"""Per-database Vertex AI Context Caching (`cachedContents`) adapter (`R1`).

Creates a server-side `CachedContent` resource per `db_id` packing full `CREATE
TABLE` DDL, table/column descriptions, and 3-gram retrieved cell value
dictionaries under a 30-to-60 minute TTL (`1800s`..`3600s`), rewrites matching
`generateContent` requests to reference `cachedContent` with stripped schema
text, and guarantees `cachedContents.delete` teardown on context manager exit.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import bisect
import dataclasses
import hashlib
import http.client
import json
import logging
import os
import pathlib
import re
import secrets
import ssl
import time
from types import TracebackType
from typing import Self
import urllib.error
import urllib.request

from argus_sql.adapters.exceptions import LLMAdapterAuthError
from argus_sql.adapters.exceptions import LLMAdapterTransportError
from argus_sql.adapters.types import BatchLLMPort
from argus_sql.adapters.types import LLMCompletionRequest
from argus_sql.adapters.types import LLMCompletionResponse
from argus_sql.adapters.types import VertexCachedContent
from argus_sql.adapters.vertex_auth import VertexTokenResolver
from argus_sql.adapters.vertex_auth import build_vertex_cached_content_item_url
from argus_sql.adapters.vertex_auth import build_vertex_cached_contents_url
from argus_sql.adapters.vertex_auth import build_vertex_publisher_model_path
from argus_sql.adapters.vertex_auth import normalize_gemini_model_name
from argus_sql.adapters.vertex_auth import validate_gcp_location
from argus_sql.adapters.vertex_auth import validate_gcp_project_id
from argus_sql.adapters.vertex_auth import validate_vertex_cached_content_name
from argus_sql.adapters.vertex_gemini_client import coerce_nonneg_int
from argus_sql.contracts.schema_ddl_renderer import render_schema_ddl
from argus_sql.contracts.types import DatabaseSchema

# Module-level logger following Google Python Style Guide lazy-% conventions.
_LOGGER = logging.getLogger(__name__)

# Callable signature `(method, url, payload_bytes, headers, timeout_sec) -> (status, body)`.
HttpTransportCallable = Callable[[str, str, bytes | None, dict[str, str], float], tuple[int, bytes]]

# Transient upstream HTTP status codes retried with bounded jittered backoff.
_RETRYABLE_HTTP_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})

# R1 TTL bounds: 30 minutes (`1800s`) minimum to 60 minutes (`3600s`) maximum.
MIN_CACHE_TTL_SECONDS: int = 1800
MAX_CACHE_TTL_SECONDS: int = 3600

# Default SQLite system prompt matching `pipelines/prompt_templates.py` verbatim
# so `adapters/` never imports upward from `argus_sql.pipelines`.
DEFAULT_SQLITE_SYSTEM_PROMPT: str = (
    "You are an expert relational database engineer specializing in SQLite. Given a CREATE TABLE schema, optional "
    "External Evidence, and a user Question, output a single valid read-only SQLite SELECT statement inside a "
    "```sql ... ``` code block. Never use `SELECT *` or `table.*` wildcard projections—project only the exact "
    "column(s) requested by the Question. Follow External Evidence mappings, ratios, and string-slicing rules "
    "faithfully (if Evidence writes boolean shorthand like `AVG(col = 'x')` for an 'average number of X per entity' "
    "count question, compute `AVG(cnt)` over a `GROUP BY entity_id` count subquery instead of a 0/1 row proportion). "
    "Preserve `LEFT JOIN` (with only the foreign-key equality in `ON` and conditional filters inside `SUM(CASE WHEN "
    "cond THEN 1 ELSE 0 END)`) on nullable foreign keys when counting or aggregating parent entities so parent rows "
    "with NULL foreign keys are never dropped by `INNER JOIN`; use `INNER JOIN` with `CASE WHEN` when aggregating "
    "required 1-to-many child rows, and never push numerator filters into `LEFT JOIN ... ON ... AND cond`. Group "
    "joined entities by their unique ID/FK key column rather than display names, avoid joining an extra "
    "entity-lookup table solely for a name column when the filtered tables already carry that name column, wrap "
    "sub-day timestamp columns (`range: 'YYYY-MM-DD HH:MM...'`) in `DATE(col)` when comparing against `'YYYY-MM-DD'` "
    "date literals, project `(entity, metric, RANK() OVER (ORDER BY metric ...))` on imperative `'Rank X by Y'` "
    "queries, aggregate with `SUM(...) + GROUP BY period` for period totals over composite PRIMARY KEY (entity, "
    "period) tables, and check column `range:`/`matched_values`/`examples` bounds before filtering."
)

# Possessive literal-anchored `Database Schema:` matcher (zero leading-ws NFA).
# Backward blank-line / prompt-start anchoring runs in `_find_schema_header`.
_DB_SCHEMA_COLON_RE = re.compile(r"database[ \t]++schema[ \t]*+:[ \t]*+\r?+\n?+", re.IGNORECASE | re.ASCII)

# Possessive SQLite `CREATE [TEMPORARY|TEMP|VIRTUAL] TABLE` matcher at offset.
_CREATE_TABLE_KW_RE = re.compile(r"CREATE[ \t\r\n]++(?:(?:TEMPORARY|TEMP|VIRTUAL)[ \t\r\n]++)?+TABLE\b", re.IGNORECASE | re.ASCII)

# Possessive single-line `/* ... */` inline block comment atom (never crosses CR/LF).
_INLINE_BC_PAT: str = r"/\*[^*\r\n]*+\*++(?:[^/*\r\n][^*\r\n]*+\*++)*+/"

# Possessive SQLite `WITHOUT ROWID`/`STRICT` (group 1), `;` (group 2), & single-line
# inline comments after `_HWS_RE` (zero cross-line `/*...*/` re-scanning).
_POST_TABLE_TAIL_RE = re.compile(r"((?:(?:\r?+\n[ \t]*+)?+(?:WITHOUT[ \t]++ROWID|STRICT)\b[ \t]*+(?:,[ \t]*+)?+)*+)((?:\r?+\n[ \t]*+)?+;[ \t]*+)?+" rf"(?:(?:--[^\r\n]*+|{_INLINE_BC_PAT})[ \t]*+)*+", re.IGNORECASE | re.ASCII)

# Validates next-line boundary after an un-semicoloned `sqlite_master` closing `)`.
_SECT_HDR_PAT: str = r"(?:Question|User[ \t]++Question|Query|Instruction|External[ \t]++Evidence|Previous[ \t]++SQL|Detected[ \t]++Issues)[ \t]*+:"
_POST_CLOSE_FOLLOW_RE, _HWS_RE = re.compile(rf"((?:[ \t]*+{_INLINE_BC_PAT})*+[ \t]*+)(?:\r?+\n|$|(?={_SECT_HDR_PAT}))", re.IGNORECASE | re.ASCII), re.compile(r"[ \t]*+")

# C-compiled close-paren tail prefilter, trivia matchers, & clean-table fast path.
_POST_SEMI_LA: str = rf"(?=[ \t]*+(?:(?:--[^\r\n]*+|{_INLINE_BC_PAT})[ \t]*+)*+(?:\r?+\n|\r?+$|(?=[cC])CREATE\b))"
_CLOSE_CAND_TAIL_PAT: str = rf"[ \t]*+(?:;(?:{_POST_SEMI_LA}|(?=[ \t]*+/\*))|(?=[wWsS])(?:WITHOUT[ \t]++ROWID|STRICT)\b|(?:(?:--[^\r\n]*+|{_INLINE_BC_PAT})[ \t]*+)*+(?:\r?+$|\r?+\n(?=[ \t]*+(?:;(?:{_POST_SEMI_LA}|(?=[ \t]*+/\*))|(?=[wWsS])(?:WITHOUT[ \t]++ROWID|STRICT)\b|(?:(?:--[^\r\n]*+|{_INLINE_BC_PAT}[ \t]*+)\r?+\n[ \t]*+)*+(?:\r?+\n|\r?+$|/\*|(?=[cCqQuUiIeEpPdD])(?:CREATE\b|{_SECT_HDR_PAT}))))))"
_SINGLE_LINE_TRIVIA_RE, _SQL_TRIVIA_RE = re.compile(rf"(?:[ \t\r\n]++|--[^\r\n]*+|{_INLINE_BC_PAT})*+"), re.compile(r"(?:[ \t\r\n]++|--[^\r\n]*+|/\*(?:[^*]*+\*++(?:[^/*][^*]*+\*++)*+/|[\s\S]*+))*+")
_DASH_RUN_RE, _LEAD_INL_RE, _INLINE_BC_RUN_RE = re.compile(r"(?:--[^\r\n]*+(?:\r?+\n[ \t]*+)*+)++"), re.compile(rf"(?:[ \t]++|{_INLINE_BC_PAT})*+"), re.compile(rf"(?:{_INLINE_BC_PAT}[ \t]*+)++")
_NON_PURE_BC_PAT: str = rf"(?!{_INLINE_BC_PAT})/\*"
_TBL_HEAD_PAT: str = r'CREATE[ \t\r\n]++(?:(?:TEMPORARY|TEMP|VIRTUAL)[ \t\r\n]++)?+TABLE[ \t\r\n]++(?:IF[ \t\r\n]++NOT[ \t\r\n]++EXISTS[ \t\r\n]++)?+(?:"(?:""|[^"\r\n])++"|`(?:``|[^`\r\n])++`|\[[^\]\r\n]++\]|[A-Za-z_][A-Za-z0-9_$.]*+)[ \t\r\n]*+(?:USING[ \t\r\n]++[A-Za-z_][A-Za-z0-9_]*+[ \t\r\n]*+)?+\('
_TABLE_OPEN_HEAD_RE = re.compile(_TBL_HEAD_PAT, re.IGNORECASE | re.ASCII)
_FAST_SIMPLE_TABLES_RE = re.compile(rf"(?:[ \t\r\n]*+{_TBL_HEAD_PAT}(?:[^()\[\]\x22'`\-/]++|\([^()\[\]\x22'`\-/]*+\))++\)[ \t]*+(?:(?:(?:WITHOUT[ \t]++ROWID|STRICT)\b[ \t]*+(?:,[ \t]*+)?+)*+;{_POST_SEMI_LA}|(?=\r?+$|\r?+\n[ \t]*+(?:\r?+\n|$|(?=[cCqQuUiIeEpPdD])(?:CREATE\b|{_SECT_HDR_PAT})))))++", re.IGNORECASE | re.ASCII)
_SL_ATOM_PAT, _FS_STOP_LA = rf"[^;\r\n\x27\x22`\[/]*+(?:(?:{_INLINE_BC_PAT}|/(?!\*)|\x27(?<=[A-Za-z]\x27)(?<!\b[xX]\x27)(?<!\b(?:IS|AS|OR|IN|ON|BY|TO|NO)\x27)(?<!\b(?:AND|NOT|ALL|FOR|SET)\x27)(?<!\b(?:LIKE|GLOB|THEN|ELSE|WHEN|CASE|FROM|JOIN|INTO|WITH|FAIL)\x27)(?<!\b(?:MATCH|WHERE|LIMIT|USING|ABORT)\x27)(?<!\b(?:ESCAPE|SELECT|VALUES|REGEXP|HAVING|OFFSET|EXISTS|UNIQUE|ISNULL|IGNORE|ACTION)\x27)(?<!\b(?:DEFAULT|COLLATE|BETWEEN|NOTNULL|REPLACE|CASCADE)\x27)(?<!\b(?:DISTINCT|ROLLBACK|RESTRICT)\x27)(?<!\b(?:REFERENCES|CONSTRAINT)\x27)(?=[st]\b)|\x27(?:\x27\x27|[^\x27;\r\n])*+\x27)[^;\r\n\x27\x22`\[/]*+)*+", rf"(?=[ \t]*+[\)cCqQuUiIeEpPdD])[ \t]*+(?:\)[ \t]*+(?:;{_POST_SEMI_LA}|(?=[wWsS])(?:WITHOUT[ \t]++ROWID|STRICT)\b|(?:\r?+\n|--[^\r\n]*+\r?+\n){(_cbc := r'(?:[ \t\r\n]++|--[^\r\n]*+|/\*[^*]*+\*++(?:[^/*][^*]*+\*++)*+/)')}*+(?:;{_POST_SEMI_LA}|\r?+\n|$|(?=[cCqQuUiIeEpPdD])(?:CREATE\b|{_SECT_HDR_PAT})))|(?=[cCqQuUiIeEpPdD])(?:CREATE\b|{_SECT_HDR_PAT}))"
_FAST_SPILLED_SEMI_RE, _FAST_SPILLED_UNSEMI_RE = re.compile(rf"(?:{_cbc}*+{_TBL_HEAD_PAT}{_SL_ATOM_PAT}(?:\r?+\n(?!{_FS_STOP_LA}){_SL_ATOM_PAT})*+\r?+\n(?:{(_fsc := rf'\)[ \t]*+(?:(?:WITHOUT[ \t]++ROWID|STRICT)\b[ \t]*+(?:,[ \t]*+)?+)*+(?:\r?+\n[ \t]*+)?+;[ \t]*+(?:(?:--[^\r\n]*+|{_INLINE_BC_PAT})[ \t]*+)*+')}(?=\r?+$|/\*|(?=[cC])CREATE\b|\r?+\n(?:(?=[qQuUiIeEpPdD]){_SECT_HDR_PAT}|[ \t]*+(?:\r?+\n|\r?+$|--|/\*|(?=[cC])CREATE\b)))|[ \t]++{_fsc}(?=\r?+$|/\*|(?=[cC])CREATE\b|\r?+\n(?:(?=[cC])CREATE\b|[ \t]*+(?:\r?+\n|\r?+$|--|/\*)))))++", re.IGNORECASE | re.ASCII), re.compile(rf"(?:{_cbc}*+{_TBL_HEAD_PAT}{_SL_ATOM_PAT}(?:\r?+\n(?!{_FS_STOP_LA}){_SL_ATOM_PAT})*+\r?+\n[ \t]*+\)[ \t]*+(?:(?:WITHOUT[ \t]++ROWID|STRICT)\b[ \t]*+(?:,[ \t]*+)?+)*+(?:(?:--[^\r\n]*+|{_INLINE_BC_PAT})[ \t]*+)*+\r?+\n{_cbc}*+(?=[cC])(?={_TBL_HEAD_PAT})(?![^\r\n]*+\)[ \t]*+\r?+\n))++", re.IGNORECASE | re.ASCII)

# Pass 2 C-compiled SQLite + VIRTUAL TABLE exit matchers skipping spilled `--` prose in O(N).
_BARE_ID_PAT: str = r'(?:"(?:""|[^"\r\n])++"|`(?:``|[^`\r\n])++`|\[[^\]\r\n]++\]|[A-Za-z_][A-Za-z0-9_]*+(?![\x27A-Za-z0-9_]))'
_IDENT_PAT: str = rf"(?:{_BARE_ID_PAT}|'(?:''|[^'\r\n])++')"
_CONSTR_LA: str = r"(?!PRIMARY\b|NOT\b|DEFAULT\b|CHECK\b|AS\b|REFERENCES\b|COLLATE\b|GENERATED\b|AUTOINCREMENT\b|UNIQUE\b|CONSTRAINT\b)"
_VTAB_OPT_PAT: str = r"(?:TOKENIZE|CONTENT(?:_ROWID)?|PREFIX|COLUMNSIZE|DETAIL|LANGUAGEID|COMPRESS|UNCOMPRESS|MATCHINFO|NOTINDEXED|RANK)[ \t]*+=[ \t]*+(?:[\x27\x22\d+\-]|[A-Za-z_][A-Za-z0-9_]*+(?=[ \t]*+(?:[,)\r\n]|--|/\*|$)))"
_TYPE_MOD_PAT: str = r"(?:UNSIGNED|SIGNED|ZEROFILL|VARYING|PRECISION|CHAR(?:ACTER)?|WITH(?:OUT)?|TIME|ZONE|BIG|SMALL|TINY|MEDIUM|LARGE|LONG|SHORT|OBJECT|INT(?:EGER)?|TEXT|BLOB|N?VARCHAR2?|ASC|DESC|UNINDEXED)"
_KNOWN_TYPE_PAT: str = (
    r"(?:(?:(?:BIG|SMALL|TINY|MEDIUM|LARGE|LONG|SHORT)[ \t]++)++(?:U?INT(?:EGER)?|N?VARCHAR2?|N?CHAR(?:ACTER)?|TEXT|BLOB|FLOAT\d*|REAL|DOUBLE)\b|(?:U?INT(?:EGER|\d+)?|TINYINT|SMALLINT|MEDIUMINT|BIGINT|LARGEINT|UNSIGNED|TEXT|CITEXT|NTEXT|MEDIUMTEXT|LONGTEXT|TINYTEXT|VARYING|NATIONAL|NATIVE|N?CHAR|N?VARCHAR2?|CHARACTER|CLOB|NCLOB|STRING"
    r"|BLOB|LONGBLOB|MEDIUMBLOB|TINYBLOB|REAL|DOUBLE|FLOAT\d*|BIGNUMERIC|BIGDECIMAL|NUM(?:ERIC|BER)?|DEC(?:IMAL)?|SMALLMONEY|MONEY|BOOL(?:EAN)?|SMALLDATETIME|DATETIME2?|DATE|TIMESTAMPTZ|TIMESTAMP|TIMETZ|TIME|INTERVAL|YEAR|JSONB?|UUID|XML|SERIAL\d*|BIGSERIAL|SMALLSERIAL|BYTEA|VARBINARY|BINARY|BIT|ANY|INET|CIDR|MACADDR|ROWID|UNINDEXED)\b)"
    rf"(?:[ \t]++{_TYPE_MOD_PAT}\b)*+(?:[ \t]*+\([ \t]*+(?:MAX|[+-]?\d++(?:[ \t]*+,[ \t]*+[+-]?\d++)?+)[ \t]*+\))?+(?:[ \t]++{_TYPE_MOD_PAT}\b)*+"
)
_COMMA_NEXT_PAT: str = rf"(?:--|/\*|\r?+\n|$|(?:PRIMARY|FOREIGN|UNIQUE|CHECK|CONSTRAINT|TOKENIZE|CONTENT|PREFIX|COLUMNSIZE|DETAIL|LANGUAGEID|COMPRESS|UNCOMPRESS|MATCHINFO|NOTINDEXED|RANK)\b|{_IDENT_PAT}[ \t]*+(?:[,)=\r\n]|--|/\*|$|[A-Za-z_][A-Za-z0-9_]*+(?![\x27A-Za-z0-9_])))"
_POST_CONSTR_LA: str = rf"(?=[ \t]*+(?:,[ \t]*+{_COMMA_NEXT_PAT}|\)(?=[ \t]*+(?:[,;\r\n]|--|/\*|$|(?:WITHOUT|STRICT)\b))|--|/\*|\r?+\n|$|(?:PRIMARY[ \t]++KEY|NOT[ \t]++(?:NULL|DEFERRABLE)|NULL|UNIQUE|CHECK|DEFAULT|COLLATE|REFERENCES|GENERATED|AS|AUTOINCREMENT|CONSTRAINT|ON[ \t]++(?:CONFLICT|DELETE|UPDATE)|ASC|DESC|MATCH|DEFERRABLE|INITIALLY)\b))"
_DEFAULT_PAT: str = rf"DEFAULT[ \t]*+(?:\(|[xX]\x27|\x27(?:\x27\x27|[^\x27\r\n])*+(?:\x27{_POST_CONSTR_LA}|(?=\r?+\n|\r?+$))|\x22(?:\x22\x22|[^\x22\r\n])*+(?:\x22{_POST_CONSTR_LA}|(?=\r?+\n|\r?+$))|(?:[+-]?(?:\d++(?:\.\d*+)?+|\.\d++)(?:[eE][+-]?\d++)?+|(?:NULL|TRUE|FALSE|CURRENT_(?:TIME(?:STAMP)?|DATE))\b){_POST_CONSTR_LA})"
_REF_COL_PAT: str = rf"(?:REFERENCES[ \t]++{_IDENT_PAT}(?:[ \t]*+\([ \t]*+(?:\r?+\n|{_IDENT_PAT}(?:[ \t]*+,[ \t]*+{_IDENT_PAT})*+[ \t]*+\)))?+|COLLATE[ \t]++{_IDENT_PAT}){_POST_CONSTR_LA}"
_COL_CONSTR_PAT: str = rf"(?:CONSTRAINT[ \t]++{_IDENT_PAT}[ \t]++)?+(?:{_DEFAULT_PAT}|(?:CHECK|AS)[ \t]*+\(|GENERATED[ \t]++ALWAYS\b|{_REF_COL_PAT}|(?:PRIMARY[ \t]++KEY(?:[ \t]++(?:ASC|DESC))?+|NOT[ \t]++NULL|AUTOINCREMENT|UNIQUE)\b{_POST_CONSTR_LA})"
_SQL_CLAUSE_PAT: str = (
    rf"(?:,[ \t]*+)?+(?:(?=[pPfFcCuU])(?:(?:PRIMARY[ \t]++KEY|FOREIGN[ \t]++KEY|CHECK)[ \t]*+\(|UNIQUE[ \t]*+(?:\(|ON[ \t]++CONFLICT\b|,[ \t]*+(?:--|/\*|\r?+\n|$)|--|/\*|\r?+\n|$)|CONSTRAINT[ \t]++{_IDENT_PAT}[ \t]++(?:PRIMARY[ \t]++KEY|FOREIGN[ \t]++KEY|UNIQUE|CHECK|NOT[ \t]++NULL|DEFAULT)\b)"
    rf"|{_VTAB_OPT_PAT}|{_COL_CONSTR_PAT}|{_IDENT_PAT}[ \t]++(?=[A-Za-z_])(?:{_COL_CONSTR_PAT}|{_CONSTR_LA}[A-Za-z_][A-Za-z0-9_]*+(?![\x27A-Za-z0-9_])(?:[ \t]++{_CONSTR_LA}[A-Za-z_][A-Za-z0-9_]*+(?![\x27A-Za-z0-9_])){{0,4}}+[ \t]*+(?:\([ \t]*+(?:MAX|[+-]?\d++(?:[ \t]*+,[ \t]*+[+-]?\d++)?+)[ \t]*+\)[ \t]*+)?+{_COL_CONSTR_PAT}"
    rf"|{_KNOWN_TYPE_PAT}[ \t]*+(?:,[ \t]*+{_COMMA_NEXT_PAT}|\)(?=[ \t]*+(?:[,;\r\n]|--|/\*|$|(?:WITHOUT|STRICT)\b))|--|/\*|\r?+\n(?=[ \t]*+(?:[),]|--|/\*|(?:NOT[ \t]++NULL|PRIMARY[ \t]++KEY|DEFAULT|CHECK|REFERENCES|UNIQUE|COLLATE|CONSTRAINT|GENERATED)\b))|$)))"
)
_BARE_COL_PAT: str = rf"{_BARE_ID_PAT}(?:[ \t]*+,[ \t]*+{_BARE_ID_PAT}(?=[ \t]*+(?:[,)\r\n]|--|/\*|$)))*+[ \t]*+(?:,[ \t]*+{_COMMA_NEXT_PAT}|\)(?=[ \t]*+(?:[,;\r\n]|--|/\*|$|(?:WITHOUT|STRICT)\b))|--|/\*|\r?+\n(?=[ \t]*+[),]))"
_UNIND_BARE_PAT: str = rf"(?:,[ \t]*+{_BARE_COL_PAT}|{_BARE_ID_PAT}[ \t]*+(?:,[ \t]*+{_COMMA_NEXT_PAT}|\)[ \t]*+(?:;|(?:WITHOUT[ \t]++ROWID|STRICT)\b)))"
_EXIT_PFX, _EXIT_NP_PFX = r"\n(?=[ \t,)/\x22\x27`\[]|[A-Za-z_][A-Za-z0-9_]*+(?:[ \t]++[A-Za-z_(]|[ \t]*+[=,)(])|UNIQUE\b)", r"\n(?![)\r\n])(?=[ \t,/\x22\x27`\[]|[A-Za-z_][A-Za-z0-9_]*+(?:[ \t]++[A-Za-z_(]|[ \t]*+[=,)(])|UNIQUE\b)"
_SPILLED_LINE_RE = re.compile(rf"(?:(?:[ \t]++|{_INLINE_BC_PAT})++(?:\){_CLOSE_CAND_TAIL_PAT}|(?=[,A-Za-z_\x22'`\[])(?:{_SQL_CLAUSE_PAT}|{_BARE_COL_PAT}))|\){_CLOSE_CAND_TAIL_PAT}|(?=[,A-Za-z_\x22'`\[])(?:{_SQL_CLAUSE_PAT}|{_UNIND_BARE_PAT}))", re.IGNORECASE | re.ASCII)
_SPILLED_EXIT_RE = re.compile(rf"{_EXIT_PFX}(?:(?:[ \t]++|{_INLINE_BC_PAT})++(?:\){_CLOSE_CAND_TAIL_PAT}|{_NON_PURE_BC_PAT}|(?=[,A-Za-z_\x22'`\[])(?:{_SQL_CLAUSE_PAT}|{_BARE_COL_PAT}))|\){_CLOSE_CAND_TAIL_PAT}|{_NON_PURE_BC_PAT}|(?=[,A-Za-z_\x22'`\[])(?:{_SQL_CLAUSE_PAT}|{_UNIND_BARE_PAT}))", re.IGNORECASE | re.ASCII)
_SPILLED_EXIT_NO_BC_RE = re.compile(rf"{_EXIT_PFX}{_SPILLED_LINE_RE.pattern}", re.IGNORECASE | re.ASCII)
_SPILLED_EXIT_NO_PAREN_RE = re.compile(rf"{_EXIT_NP_PFX}(?:(?:[ \t]++|{_INLINE_BC_PAT})++(?:{_NON_PURE_BC_PAT}|(?=[,A-Za-z_\x22\x27`\[])(?:{_SQL_CLAUSE_PAT}|{_BARE_COL_PAT}))|{_NON_PURE_BC_PAT}|(?=[,A-Za-z_\x22\x27`\[])(?:{_SQL_CLAUSE_PAT}|{_UNIND_BARE_PAT}))", re.IGNORECASE | re.ASCII)
_SPILLED_EXIT_NO_PAREN_NO_BC_RE = re.compile(rf"{_EXIT_NP_PFX}(?:(?:[ \t]++|{_INLINE_BC_PAT})++(?=[,A-Za-z_\x22\x27`\[])(?:{_SQL_CLAUSE_PAT}|{_BARE_COL_PAT})|(?=[,A-Za-z_\x22\x27`\[])(?:{_SQL_CLAUSE_PAT}|{_UNIND_BARE_PAT}))", re.IGNORECASE | re.ASCII)

# Single-pass keyword-prefix section delimiter & Pass 1/2 structural close matchers.
_NON_DDL_QUESTION_DELIM_RE, _POST_Q_SECT_START_RE = re.compile(r"\n[ \t]*+\r?+\n[ \t]*+(?=(?:Question|User[ \t]++Question|Query)[ \t]*+:)", re.IGNORECASE | re.ASCII), re.compile(r"(?:External[ \t]++Evidence|Previous[ \t]++SQL|Detected[ \t]++Issues|Instruction)[ \t]*+:", re.IGNORECASE | re.ASCII)
_BLANK_SECT_RE = re.compile(rf"\n[ \t]*+\r?+\n[ \t]*+(?=[qQuUeEiIpPdD])(?i:(?P<q>Question|User[ \t]++Question|Query)[ \t]*+:|(?P<ev>External[ \t]++Evidence[ \t]*+:(?:{(_rl := r'[^\r\n();cC]*+\r?+\n[ \t]*+\r?+\n[ \t]*+')}External[ \t]++Evidence[ \t]*+:)*+)|(?P<ins>Instruction[ \t]*+:(?:{_rl}Instruction[ \t]*+:)*+)|(?P<prev>Previous[ \t]++SQL[ \t]*+:(?:{_rl}Previous[ \t]++SQL[ \t]*+:)*+)|(?P<det>Detected[ \t]++Issues[ \t]*+:(?:{_rl}Detected[ \t]++Issues[ \t]*+:)*+))", re.ASCII)
_UNMATCHED_TBL_CLOSE_RE, _POST_DDL_SUBQ_START_RE, _SECT_HDR_RE = re.compile(rf"(?:[\x27\x22`\]\)]|\*/|\n[ \t]*+)(\))[ \t]*+(?:(?:WITHOUT[ \t]++ROWID|STRICT)\b[ \t]*+,?+[ \t]*+)*+(?:;[ \t]*+(?:(?:--[^\r\n]*+|{_INLINE_BC_PAT})[ \t]*+)*+(?:\r?+\n|\r?+$)|(?:--[^\r\n]*+\r?+\n|\r?+\n)[ \t]*+\r?+\n)", re.IGNORECASE | re.ASCII), re.compile(rf"(?:SELECT|WITH|VALUES|EXPLAIN)\b|{_SECT_HDR_PAT}", re.IGNORECASE | re.ASCII), re.compile(_SECT_HDR_PAT, re.IGNORECASE | re.ASCII)
_CLOSE_PAREN_LINE_RE, _SEMI_CLOSE_LINE_RE, _UNSEMI_SUBQ_CLOSE_RE = re.compile(rf"\n([ \t]*+\)){_CLOSE_CAND_TAIL_PAT}", re.IGNORECASE | re.ASCII), re.compile(rf"\n[ \t]*+\)[ \t]*+(?:(?:WITHOUT[ \t]++ROWID|STRICT)\b[ \t]*+(?:,[ \t]*+)?+)*+;[ \t]*+(?:(?:--[^\r\n]*+|{_INLINE_BC_PAT})[ \t]*+)*+(?=\r?+\n|\r?+$|(?=[cC])CREATE\b)", re.IGNORECASE | re.ASCII), re.compile(rf"\n[ \t]*+\)[ \t]*+(?:(?:WITHOUT[ \t]++ROWID|STRICT)\b[ \t]*+,?+[ \t]*+)*+(?:(?:--[^\r\n]*+|{_INLINE_BC_PAT})[ \t]*+)*+\r?+\n(?:[ \t]*+\r?+\n[ \t]*+(?:(?:--[^\r\n]*+|{_INLINE_BC_PAT})\r?+\n[ \t]*+)*+(?=(?:SELECT|WITH|VALUES|EXPLAIN)\b|{_SECT_HDR_PAT})|[ \t]*+(?={_SECT_HDR_PAT}))", re.IGNORECASE | re.ASCII)
_PRE_CREATE_TAIL_RE, _ORPHAN_CLOSE_RE, _SQL_NON_CODE_RE = re.compile(r"(?:(?:\)|[\r\n][ \t]*+)\)(?:[ \t]*+(?:WITHOUT[ \t]++ROWID|STRICT)\b[ \t]*+,?+)*+|\)(?:[ \t\r\n]*+(?:WITHOUT[ \t]++ROWID|STRICT)\b[ \t]*+,?+)*+[ \t\r\n]*+;)\Z", re.IGNORECASE | re.ASCII), re.compile(rf"\n[ \t]*+\)[ \t]*+(?:(?:(?:WITHOUT[ \t]++ROWID|STRICT)\b[ \t]*+,?+[ \t]*+)*+;{_POST_SEMI_LA}|(?:\r?+\n|\r?+$))", re.IGNORECASE | re.ASCII), re.compile(r"--[^\r\n]*+|/\*(?:[^*]*+\*++(?:[^/*][^*]*+\*++)*+/|[\s\S]*+)|\x27(?:\x27\x27|[^\x27])*+(?:\x27|\Z)|\x22(?:\x22\x22|[^\x22])*+(?:\x22|\Z)|`(?:``|[^`])*+(?:`|\Z)|\[[^\]]*+(?:\]|\Z)")
_BLANK_LINE_SEP_RE, _ASCII_WS_RE, _DDL_HEAD_TOKEN_RE, _DDL_BODY_TOKEN_RE, _TRAILING_COMMA_CLOSE_RE = re.compile(r"\r?+\n[ \t]*+\r?+\n[ \t]*+"), re.compile(r"[ \t\r\n]*+"), re.compile(r"""[(;\["\x27`]|--|/\*"""), re.compile(r"""[()\["\x27`]|--|/\*"""), re.compile(r",[ \t]*+--[^\r\n]*+\r?+\n[ \t]*+\)\Z")
_FLAT_PAREN_GROUP_RE, _SKIP_TO_SEMI_RE = re.compile(rf"(?:{[_pa := rf'(?:[^();\[\]\x22\x27`\-/]++|{_INLINE_BC_PAT}|/(?!\*)|\x27(?:\x27\x27|[^\x27;\r\n])*+\x27|\x22(?:\x22\x22|[^\x22;\r\n])*+\x22|`(?:``|[^`;\r\n])*+`|\[[^\]\r\n;]*+\])', _g := rf'\({_pa}*+\)', *[_g := rf'\((?:{_pa}|{_g})*+\)' for _ in range(15)]][-1]}{_pa}*+)++"), re.compile(rf"(?:[^);\-\x27\x22`\[/(]++|/(?!\*)|{_INLINE_BC_PAT}|\)(?![ \t\r\n]*+;|[ \t]*+(?:WITHOUT|STRICT)\b))++", re.IGNORECASE | re.ASCII)
_CLOSED_QUOTE_BR_RE, _HDR_NON_CODE_RE, _SQL_Q_BODY_RE, _SQL_SELECT_KW_RE, _SQL_VALUES_OPEN_RE, _UNSEMI_END_RE, _ML_DELIM_RE, _TYPE_SQ_LINE_RE = re.compile(r"--[^\r\n]*+|\x27(?:\x27\x27|[^\x27])*+\x27|\x22(?:\x22\x22|[^\x22])*+\x22|`(?:``|[^`])*+`|\[[^\[\]]*+\]"), re.compile(r"--[^\r\n]*+|/\*[^*]*+\*++(?:[^/*][^*]*+\*++)*+/|\x27(?:\x27\x27|[^\x27])*+\x27|\x22(?:\x22\x22|[^\x22])*+\x22|`(?:``|[^`])*+`|\[[^\[\]]*+\]"), re.compile(r"\bFROM\b|\([ \t\r\n]*+SELECT\b", re.IGNORECASE | re.ASCII), re.compile(r"\bSELECT\b", re.IGNORECASE | re.ASCII), re.compile(r"VALUES[ \t\r\n]*+\((?![^)\r\n]*+(?:\.\.|\bOR\b))", re.IGNORECASE | re.ASCII), re.compile(rf"\n[ \t]*+\)[ \t]*+(?:(?:WITHOUT[ \t]++ROWID|STRICT)\b[ \t]*+,?+[ \t]*+)*+(?:(?:--[^\r\n]*+|{_INLINE_BC_PAT})[ \t]*+)*+[ \t\r\n]*+\Z", re.IGNORECASE | re.ASCII), re.compile(r'\[[^\]\r\n]*+\r?+\n|`[^`\r\n]*+\r?+\n|"[^"\r\n]*+\r?+\n|/\*[^*\r\n]*+(?:\*++[^/*\r\n][^*\r\n]*+)*+\r?+\n'), re.compile(rf"(?:\n|\()[ \t]*+{_IDENT_PAT}[ \t]++{_KNOWN_TYPE_PAT}(?:[ \t]++[A-Za-z_][A-Za-z0-9_]*+)*+\x27(?![^\x27\r\n]*+\x27)", re.IGNORECASE | re.ASCII)
_OPEN_DEPTH_RUN_RE, _CLOSE_DEPTH_RUN_RE, _CLOSED_QUOTE_RUN_RE = re.compile(r"(?:\(++[^()\[\]\x22'`\-/]*+)++"), re.compile(r"(?:\)++[^()\[\]\x22'`\-/]*+)++"), re.compile(r"(?:(?:\x27(?:\x27\x27|[^\x27)\-])*+\x27|\x22(?:\x22\x22|[^\x22)\-])*+\x22)[^()\[\]\x22\x27`\-/]*+)++")
_QUOTE_RE_MAP: dict[str, re.Pattern[str]] = {"'": re.compile(r"'(?:''|[^'])*+(?:'|\Z)"), '"': re.compile(r'"(?:""|[^"])*+(?:"|\Z)'), "`": re.compile(r"`(?:``|[^`])*+(?:`|\Z)")}
_FAST_COL_LINE_RE = re.compile(r"\n([ \t]*+)[A-Za-z_][A-Za-z0-9_]*+(?![\x27A-Za-z0-9_])[ \t]++(?:U?INT(?:EGER)?|TEXT|N?VARCHAR2?|N?CHAR|REAL|FLOAT|DOUBLE|BLOB|NUMERIC|DECIMAL|DATE(?:TIME)?|TIMESTAMP(?:TZ)?|BOOL(?:EAN)?|BIGINT|SMALLINT|TINYINT|ANY)\b[ \t]*+(?:,(?=[ \t]*+(?:--|/\*|\r?+\n|$))|[\r\n(]|--|/\*|(?:PRIMARY[ \t]++KEY[ \t]*+(?:[,)\r\n]|--|/\*)|NOT[ \t]++NULL[ \t]*+(?:[,)\r\n]|--|/\*)|DEFAULT\b|(?:CHECK|AS)[ \t]*+\(|UNSIGNED\b|BIG\b|SMALL\b|WITH(?:OUT)?\b))", re.IGNORECASE | re.ASCII)


def _sha256_text(text: str) -> str:
  """Returns the lowercase hex SHA-256 digest of UTF-8 `text`."""
  # Hash normalized UTF-8 bytes so `systemInstruction` parity checks stay exact.
  return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _collapse_lines(text: str) -> str:
  """Collapses internal CR/LF line breaks into single spaces."""
  # Keep inline `--` DDL comments and description bullets on a single line.
  return " ".join(part.strip() for part in text.splitlines() if part.strip())


def _client_str_attr(client: object, attr_name: str) -> str:
  """Reads an optional string attribute from `client`, treating `None` as `""`."""
  # Coerce `None` to `""` so custom doubles with `attr = None` fall back to env.
  raw_val = getattr(client, attr_name, None)
  return str(raw_val).strip() if raw_val is not None else ""


def validate_cache_ttl_seconds(ttl_seconds: int) -> int:
  """Validates that `ttl_seconds` lies within the R1 30–60 minute window.

  Args:
    ttl_seconds: Requested `CachedContent` time-to-live in whole seconds.

  Returns:
    Validated integer TTL in `[1800, 3600]`.

  Raises:
    ValueError: If `ttl_seconds` is boolean, non-integer, or outside `[1800, 3600]`.
  """
  # Reject `bool` explicitly because `bool` subclasses `int` in Python.
  if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
    raise ValueError("ttl_seconds must be an integer between 1800 and 3600.")
  # Enforce the 30-to-60 minute TTL window mandated by R1 to bound storage cost.
  if ttl_seconds < MIN_CACHE_TTL_SECONDS or ttl_seconds > MAX_CACHE_TTL_SECONDS:
    raise ValueError(f"ttl_seconds must be between {MIN_CACHE_TTL_SECONDS} and {MAX_CACHE_TTL_SECONDS} (30-60 minutes), got {ttl_seconds}.")
  return ttl_seconds


def _format_descriptions_section(schema: DatabaseSchema | None, descriptions: Mapping[object, str] | str | None) -> str:
  """Formats deterministic table/column descriptions from mapping or schema."""
  # Prefer explicit string descriptions when caller supplies pre-rendered prose.
  if isinstance(descriptions, str) and descriptions.strip():
    return descriptions.strip()

  # Format explicit mapping entries supporting `'tbl.col'` or `('tbl', 'col')`.
  if isinstance(descriptions, Mapping):
    lines: list[str] = []
    for raw_key, raw_val in descriptions.items():
      # Collapse internal CR/LF breaks so each bullet stays single-line.
      val_str = _collapse_lines(str(raw_val))
      if not val_str:
        continue
      # Normalize 2-tuple `(table, column)` keys into dotted `table.column` form.
      raw_label = f"{raw_key[0]}.{raw_key[1]}" if isinstance(raw_key, tuple) and len(raw_key) == 2 else str(raw_key)
      key_label = _collapse_lines(raw_label)
      if key_label:
        lines.append(f"- {key_label}: {val_str}")
    return "\n".join(lines) if lines else "(none provided)"

  # Extract column descriptions, numeric/date ranges, and examples from schema.
  if schema is not None:
    schema_lines: list[str] = []
    for table in schema.tables:
      for col in table.columns:
        parts: list[str] = []
        # Include single-line sanitized BIRD metadata CSV column description.
        clean_desc = _collapse_lines(col.description) if col.description else ""
        if clean_desc:
          parts.append(clean_desc)
        # Surface min/max bounds so date and numeric filters respect domain limits.
        if col.value_range is not None:
          lo, hi = col.value_range
          parts.append(f"range: {lo}..{hi}")
        # Cap inline examples at 5 literals because full dictionary follows below.
        if col.value_examples:
          parts.append(f"examples: [{', '.join(repr(v) for v in col.value_examples[:5])}]")
        header = f"- {table.name}.{col.name} ({col.data_type})"
        schema_lines.append(f"{header}: {' | '.join(parts)}" if parts else header)
    return "\n".join(schema_lines) if schema_lines else "(none provided)"

  # Return explicit placeholder when neither descriptions nor schema was passed.
  return "(none provided)"


def _append_unique_literal(ordered_by_col: dict[str, list[str]], col_key: str, literal_val: object, max_values_per_column: int) -> None:
  """Appends a non-empty deduplicated string literal up to `max_values_per_column`."""
  clean_key = col_key.strip()
  # Ignore non-string or whitespace-only cells so SQL literals remain valid.
  if not clean_key or not isinstance(literal_val, str) or not literal_val.strip():
    return
  bucket = ordered_by_col.setdefault(clean_key, [])
  # Enforce per-column cap and exact-string deduplication to bound cache tokens.
  if len(bucket) < max_values_per_column and literal_val not in bucket:
    bucket.append(literal_val)


def _ingest_cell_mapping(ordered_by_col: dict[str, list[str]], mapping_obj: Mapping[object, object], max_values_per_column: int) -> None:
  """Ingests flat `'tbl.col' -> [vals]`, tuple-keyed, or nested table mappings."""
  for raw_key, raw_vals in mapping_obj.items():
    # Support nested `{'schools': {'county': ['Alameda']}}` dictionaries,
    # 2-tuple `(table, column)` keys, and flat dotted `'table.column'` keys.
    if isinstance(raw_vals, Mapping):
      tbl = str(raw_key).strip()
      pairs = [(f"{tbl}.{str(c).strip()}", v) for c, v in raw_vals.items()]
    elif isinstance(raw_key, tuple) and len(raw_key) == 2:
      pairs = [(f"{raw_key[0]}.{raw_key[1]}", raw_vals)]
    else:
      pairs = [(str(raw_key).strip(), raw_vals)]
    # Guard against iterating over raw `str` or `bytes` character-by-character.
    for col_key, seq_vals in pairs:
      if isinstance(seq_vals, Sequence) and not isinstance(seq_vals, (str, bytes)):
        for item in seq_vals:
          _append_unique_literal(ordered_by_col, col_key, item, max_values_per_column)


def _extract_cell_value_dictionary(schema: DatabaseSchema | None, cell_values: object, cell_value_dict: Mapping[str, Sequence[str]] | None, max_values_per_column: int) -> str:
  """Formats 3-gram cell value dictionaries via duck-typing (zero `linking` imports)."""
  # Preserve deterministic column insertion order for stable cache prefix text.
  ordered_by_col: dict[str, list[str]] = {}

  # Source A: duck-type `CellValueRetriever` (`.entries`), entry sequences, or maps.
  if cell_values is not None:
    if not hasattr(cell_values, "entries") and isinstance(cell_values, Mapping):
      _ingest_cell_mapping(ordered_by_col, cell_values, max_values_per_column)
    else:
      # Read `.table_name`, `.column_name`, and `.raw_value` without `linking` import.
      # Handles both `.entries` containers and direct `ValueIndexEntry` sequences.
      raw_seq = getattr(cell_values, "entries", cell_values)
      if isinstance(raw_seq, Sequence) and not isinstance(raw_seq, (str, bytes)):
        for entry in raw_seq:
          t_name = str(getattr(entry, "table_name", "")).strip()
          c_name = str(getattr(entry, "column_name", "")).strip()
          val = getattr(entry, "raw_value", getattr(entry, "matched_value", None))
          if t_name and c_name:
            _append_unique_literal(ordered_by_col, f"{t_name}.{c_name}", val, max_values_per_column)

  # Source B: merge optional explicit `cell_value_dict` column-to-literals map.
  if cell_value_dict is not None:
    _ingest_cell_mapping(ordered_by_col, cell_value_dict, max_values_per_column)

  # Source C: fall back to `schema` column `value_examples` when index is omitted.
  if not ordered_by_col and schema is not None:
    for table in schema.tables:
      for col in table.columns:
        for ex_val in col.value_examples:
          _append_unique_literal(ordered_by_col, f"{table.name}.{col.name}", ex_val, max_values_per_column)

  # Emit explicit marker when database contains zero indexed TEXT literals.
  if not ordered_by_col:
    return "(none indexed)"

  # Format Python `repr()` literals so embedded quotes and spaces remain unambiguous.
  lines = [f"- {col_key}: [{', '.join(repr(v) for v in vals[:max_values_per_column])}]" for col_key, vals in ordered_by_col.items() if vals]
  return "\n".join(lines) if lines else "(none indexed)"


def build_db_cached_context_text(db_id: str, *, schema: DatabaseSchema | None = None, cell_values: object = None,
    schema_ddl: str | None = None, descriptions: Mapping[object, str] | str | None = None,
    cell_value_dict: Mapping[str, Sequence[str]] | None = None,
    include_metadata_comments: bool = True, max_values_per_column: int = 64) -> str:
  """Constructs deterministic per-database context text for `cachedContents.create`.

  Args:
    db_id: Non-empty target database identifier (e.g., `'california_schools'`).
    schema: Optional `DatabaseSchema` rendered via `render_schema_ddl`.
    cell_values: Optional duck-typed `CellValueRetriever` (having `.entries`),
      sequence of entry objects (`.table_name`, `.column_name`, `.raw_value`),
      or column-to-values mapping.
    schema_ddl: Optional pre-rendered `CREATE TABLE` DDL string.
    descriptions: Optional table/column descriptions string or column mapping.
    cell_value_dict: Optional explicit `table.column -> [values]` mapping.
    include_metadata_comments: Passed to `render_schema_ddl` when rendering.
    max_values_per_column: Upper bound on distinct literals serialized per column.

  Returns:
    Formatted context block containing Schema DDL, Table/Column Descriptions,
    and 3-Gram Retrieved Cell Value Dictionary sections.

  Raises:
    ValueError: If `db_id` is blank, `max_values_per_column < 1`, or neither
      `schema` nor non-empty `schema_ddl` is provided.
  """
  # Validate `db_id` and per-column literal bounds before rendering schema text.
  clean_db_id = db_id.strip()
  if not clean_db_id:
    raise ValueError("db_id must be non-empty.")
  # Guard against `bool` (`True == 1`) or non-positive literal limits.
  if isinstance(max_values_per_column, bool) or max_values_per_column < 1:
    raise ValueError("max_values_per_column must be >= 1.")

  # Sanitize multi-line `ColumnSchema.description` fields before `render_schema_ddl`
  # so `--` inline SQL comments never spill un-commented lines into cached DDL.
  clean_schema = _sanitize_schema_descriptions(schema) if schema is not None else None

  # Render DDL from explicit string override or sanitized `DatabaseSchema`.
  if schema_ddl is not None and schema_ddl.strip():
    ddl_text = schema_ddl.strip()
  elif clean_schema is not None:
    ddl_text = render_schema_ddl(clean_schema, include_metadata_comments=include_metadata_comments)
  else:
    raise ValueError("Either `schema` or non-empty `schema_ddl` must be provided.")

  # Assemble column descriptions and 3-gram cell value dictionaries deterministically.
  desc_text = _format_descriptions_section(clean_schema, descriptions)
  cell_dict_text = _extract_cell_value_dictionary(clean_schema, cell_values, cell_value_dict, max_values_per_column)
  # Combine all three R1 database grounding artifacts into one cached prefix.
  return (f"Database ID: {clean_db_id}\n\nDatabase Schema:\n{ddl_text}\n\n"
          f"Table and Column Descriptions:\n{desc_text}\n\n3-Gram Retrieved Cell Value Dictionary:\n{cell_dict_text}")


def _sanitize_schema_descriptions(schema: DatabaseSchema) -> DatabaseSchema:
  """Returns `schema` with internal newlines collapsed in column descriptions."""
  # Normalize multi-line CSV descriptions without mutating `contracts/` modules.
  tables = tuple(dataclasses.replace(tbl, columns=tuple(
      dataclasses.replace(col, description=_collapse_lines(col.description))
      if ("\n" in col.description or "\r" in col.description) else col
      for col in tbl.columns)) for tbl in schema.tables)
  return dataclasses.replace(schema, tables=tables)


def _collect_multiline_fragments(schema: DatabaseSchema | None) -> tuple[tuple[str, str], ...]:
  """Precollects `('-- ' + raw, '-- ' + flat)` LF and CRLF description pairs."""
  # Emit both LF and CRLF `-- <desc>` forms so CRLF prompts match Layer 1
  # while dynamic `Question:` / SQL text without `-- ` stays untouched.
  pairs: dict[str, str] = {}
  for tbl in schema.tables if schema is not None else ():
    for raw in (getattr(tbl, "description", ""), *(c.description for c in tbl.columns)):
      if raw and ("\n" in (frag := str(raw).strip()) or "\r" in frag):
        flat, lf = f"-- {_collapse_lines(frag)}", frag.replace("\r\n", "\n").replace("\r", "\n")
        for variant in (frag, lf, lf.replace("\n", "\r\n")):
          pairs[f"-- {variant}"] = flat
  # Sort longest-first so nested multi-line substrings never partially replace.
  return tuple(sorted(pairs.items(), key=lambda kv: len(kv[0]), reverse=True))


def _find_schema_header(user_prompt: str) -> tuple[int, int] | None:
  """Locates `(schema_start, header_end)` for `Database Schema:` in O(N) time."""
  floor = 0
  # Inspect disjoint `[floor, idx)` slices directly on `user_prompt` so leading
  # whitespace never backtracks and Unicode characters never shift indices.
  for match in _DB_SCHEMA_COLON_RE.finditer(user_prompt):
    idx = match.start()
    if floor == 0 and not user_prompt[:idx].strip():
      return (0, match.end())
    # Verify a blank-line paragraph break (`\r?\n[ \t]*\r?\n[ \t]*`) precedes `idx`.
    nl = user_prompt.rfind("\n", floor, idx)
    if nl != -1 and not user_prompt[nl + 1 : idx].strip(" \t"):
      prev_end = nl - 1 if (nl > floor and user_prompt[nl - 1] == "\r") else nl
      prev_nl = user_prompt.rfind("\n", floor, prev_end)
      if prev_nl != -1 and not user_prompt[prev_nl + 1 : prev_end].strip(" \t"):
        return (nl + 1, match.end())
    floor = idx + 8
  return None


def _advance_comment_or_quote(text: str, pos: int) -> int:
  """Advances `pos` past SQL `--`, `/*...*/`, `'...'`, `"..."`, `` `...` ``, or `[...]`."""
  n, ch = len(text), text[pos]
  # Line comments run to LF; block comments run to `*/`; `[...]` runs to `]`.
  if ch == "-" and pos + 1 < n and text[pos + 1] == "-":
    return (nl_idx + 1) if (nl_idx := text.find("\n", pos + 2)) != -1 else n
  if ch == "/" and pos + 1 < n and text[pos + 1] == "*":
    return (end_idx + 2) if (end_idx := text.find("*/", pos + 2)) != -1 else n
  if ch == "[":
    return (close_idx + 1) if (close_idx := text.find("]", pos + 1)) != -1 else n
  # Consume doubled quotes (`''`, `""`, ````) in one C pass without Python loops.
  return m_q.end() if (q_re := _QUOTE_RE_MAP.get(ch)) is not None and (m_q := q_re.match(text, pos)) is not None else pos


def _skip_sql_trivia(text: str, pos: int, memo: list[int]) -> int:
  """Advances `pos` past SQL whitespace, `--`, and `/*...*/` in O(1) amortized."""
  # Consume disjoint single-line whitespace, `--`, and inline `/*...*/` in C.
  inl = _SINGLE_LINE_TRIVIA_RE.match(text, pos)
  q = inl.end() if inl is not None else pos
  # Guard `startswith("/*")` so unclosed spilled `/*` in `t1` never skips `t2`.
  if not text.startswith("/*", q):
    return q
  # Return memoized multi-line `[bc_lo, bc_hi)` end offset in O(1) without rescanning.
  if memo[0] <= q < memo[1]:
    return memo[1]
  full = _SQL_TRIVIA_RE.match(text, q)
  memo[0], memo[1] = q, (full.end() if full is not None else q)
  return memo[1]


def _next_create_bound(text: str, pos: int, last_create: int) -> int:
  """Finds the next inter-table boundary after `pos` in O(N) amortized time."""
  # Walk backward across inter-table `--` and `/*...*/` trivia to a table close.
  lo, sc, n, hi = pos, pos, len(text), last_create + 32
  while sc <= last_create and (m_ct := _CREATE_TABLE_KW_RE.search(text, sc, hi)) is not None:
    if (c_pos := m_ct.start()) >= lo + 3 and text[c_pos - 2 : c_pos] == ";\n" and (nl_k := text.rfind("\n", lo, c_pos - 2)) != -1 and text[nl_k + 1] == ")":
      return c_pos
    p, tr_s = c_pos - 1, c_pos
    in_cd = (dc := text.find("--", ln_c := (lo if (nl_c := text.rfind("\n", lo, c_pos)) == -1 else nl_c + 1), c_pos)) != -1 and not ("'" in (cs := text[ln_c:dc]) or '"' in cs or "`" in cs or "[" in cs or "/*" in cs)
    while not in_cd and p >= lo:
      if text[p] in " \t\r\n":
        p -= 1
      elif (dp := text.find("--", ln_p := (lo if (nl_p := text.rfind("\n", lo, p)) == -1 else nl_p + 1), p + 1)) != -1 and not ("'" in (s := text[ln_p:dp]) or '"' in s or "`" in s or "[" in s or "/*" in s):
        tr_s, p = (ln_p if not s.strip(" \t") else tr_s), dp - 1
      elif p > lo and text[p - 1 : p + 1] == "*/" and (bo1 := text.rfind("/*", lb := (lo if (pc := text.rfind("*/", lo, p - 1)) == -1 else pc + 2), p - 1)) != -1:
        bo = text.find("/*", max(lb, text.rfind(";", lb, bo1) + 1, text.rfind(")", lb, bo1) + 1), bo1 + 2)
        tr_s, p = bo, bo - 1
      elif text[p] in ");dDtT," and (text[p] == ";" or "\n" in text[p + 1 : c_pos]) and (nl_p == -1 or not text[ln_p:p].lstrip(" \t").upper().startswith(("CREATE ", "INSERT INTO ", "DROP ", "ALTER "))) and ((lo == pos and nl_p == -1 and text[p] == ")") or _PRE_CREATE_TAIL_RE.search(text, max(0, lo - (lo > pos), p - 160), p + 1) is not None):
        return (nl_a + 1) if (nl_a := text.find("\n", p + 1, tr_s)) != -1 else tr_s
      else:
        break
    in_bc = not in_cd and (bc_o := text.rfind("/*", lo, c_pos)) != -1 and text.find("*/", bc_o + 2, c_pos) == -1
    sc, lo = (((eb + 2) if (eb := text.find("*/", c_pos)) != -1 else n), lo) if in_bc else ((nxt := (nl_e + 1) if (nl_e := text.find("\n", m_ct.end())) != -1 else n), (lo if in_cd else nxt))
  return n


def _strip_closed_sql(s: str) -> str:
  """Replaces `--` comments and closed SQL quotes/brackets/comments with `' '` in O(N)."""
  # Bound `/*...*/` matching to `rfind('*/') + 2` so unclosed `/*` runs never rescan.
  return (_HDR_NON_CODE_RE.sub(" ", s[: bc + 2]) + _CLOSED_QUOTE_BR_RE.sub(" ", s[bc + 2 :])) if (bc := s.rfind("*/")) != -1 else _CLOSED_QUOTE_BR_RE.sub(" ", s)


def _is_post_ddl_subquery(text: str, start: int, sc_p: int, se: int, lim: int) -> bool:
  """Returns True when `[start, se)` is a balanced post-DDL SQL query or section."""
  # Strip only closed quotes/comments on `:` headers so English apostrophes keep `)`.
  q0 = m_t.end() if (m_t := _SQL_TRIVIA_RE.match(text, start, sc_p)) is not None else start
  if q0 >= sc_p or (m_kw := _POST_DDL_SUBQ_START_RE.match(text, q0, sc_p)) is None or not (clean_q := (_strip_closed_sql(raw_q) if (is_hdr := (":" in text[q0 : m_kw.end()])) else _SQL_NON_CODE_RE.sub(" ", raw_q)) if (raw_q := text[q0 : sc_p + 1]) else "").endswith(")") or "*/" in clean_q or "]" in clean_q or _FAST_COL_LINE_RE.search(clean_q) is not None:
    return False
  rem_i, bal_ok = (m_r.end() if (m_r := _SQL_TRIVIA_RE.match(text, se, lim)) is not None else se), (clean_q.count(")") <= clean_q.count("("))
  if is_hdr:
    return (_ORPHAN_CLOSE_RE.search(text, se, lim) is None or _POST_DDL_SUBQ_START_RE.match(text, rem_i) is not None) and (bal_ok or _BLANK_LINE_SEP_RE.match(text, se) is None or ("(" in clean_q[:-1] and (rem_i >= lim or _SECT_HDR_RE.match(text, rem_i) is not None)))
  return bal_ok and (_SQL_SELECT_KW_RE.search(clean_q, 4) is not None if (kw4 := clean_q[:4].upper()) == "WITH" else (_SQL_VALUES_OPEN_RE.match(clean_q) is not None if kw4 == "VALU" else _SQL_Q_BODY_RE.search(clean_q, 6) is not None)) and (rem_i >= lim or _SECT_HDR_RE.match(text, rem_i) is not None or (_BLANK_LINE_SEP_RE.match(text, start) is not None and _ORPHAN_CLOSE_RE.search(text, se, lim) is None))


def _needs_semi_close(text: str, paren_pos: int, tbl_lim: int, bounds: list[int]) -> bool:
  """Returns True when an explicit `;` table-close line exists in `(paren_pos, tbl_lim)`."""
  # Advance `bounds[4..5]` to the last table `;`-close in `(paren_pos, tbl_lim)`.
  if bounds[4] <= paren_pos and bounds[4] != tbl_lim:
    sc, se = (m_sc.start(), m_sc.end()) if (m_sc := _SEMI_CLOSE_LINE_RE.search(text, paren_pos + 1, tbl_lim)) is not None else (tbl_lim, tbl_lim)
    while se + 1 < tbl_lim and (m_nx := _SEMI_CLOSE_LINE_RE.search(text, se, tbl_lim)) is not None:
      if not bounds[6] and bounds[3] == len(text) and _is_post_ddl_subquery(text, se, text.find(")", m_nx.start() + 1), m_nx.end(), tbl_lim):
        break
      sc, se = m_nx.start(), m_nx.end()
    if sc < tbl_lim and ((tbl_lim < len(text) and (tbl_lim == bounds[3] or text[sc + 1] in " \t") and _UNSEMI_END_RE.search(text, se, tbl_lim) is not None) or (not bounds[6] and bounds[3] == len(text) and _POST_DDL_SUBQ_START_RE.search(text, paren_pos + 1, sc) is not None and (m_ub := _UNSEMI_SUBQ_CLOSE_RE.search(text, max(0, text.rfind("\n", 0, paren_pos + 1)), sc)) is not None and _is_post_ddl_subquery(text, m_ub.end(), text.find(")", sc + 1), se, tbl_lim))):
      sc = se = tbl_lim
    bounds[4], bounds[5] = sc, se
  return bounds[4] < tbl_lim


def _match_table_close_tail(text: str, paren_pos: int, last_create: int, memo: list[int], require_semi: bool = False) -> int | None:
  """Returns statement end offset when `text[paren_pos]` validly closes a table."""
  # Fast-path canonical `);\n` and `);\r\n` closes without regex state setup.
  if text.startswith((";\n", ";\r\n"), paren_pos + 1):
    return paren_pos + 2
  p = hws.end() if (hws := _HWS_RE.match(text, paren_pos + 1)) is not None else paren_pos + 1
  tail = _POST_TABLE_TAIL_RE.match(text, p)
  end = tail.end() if tail is not None else p
  if require_semi and (tail is None or tail.group(2) is None):
    return None
  if end >= len(text):
    return end
  # Semicolon or table modifier closes when followed by EOL or next `CREATE TABLE`.
  if tail is not None and (tail.group(1) or tail.group(2) is not None):
    ok = text[end] in "\r\n" or (end <= last_create and _CREATE_TABLE_KW_RE.match(text, _skip_sql_trivia(text, end, memo)) is not None)
    return end if ok else None
  # Un-semicoloned `)` (`sqlite_master`) must precede blank line, prompt tag, or DDL.
  if text[end] not in "\r\n":
    return None
  nxt_ln = end + (2 if text.startswith("\r\n", end) else 1)
  if (m_pf := _POST_CLOSE_FOLLOW_RE.match(text, nxt_ln)) is not None:
    return m_pf.end(1) if m_pf.end(1) > nxt_ln else end
  return end if (nxt_ln <= last_create and _CREATE_TABLE_KW_RE.match(text, _skip_sql_trivia(text, nxt_ln, memo)) is not None) else None


def _scan_spilled_prose(text: str, nl_pos: int, scan_lim: int, last_bc: int, last_create: int, bounds: list[int], memo: list[int], nxt_bc: int) -> tuple[int, int | None, int]:
  """Advances `nl_pos` monotonically across spilled `--` prose in O(N) C jumps."""
  # Fast-exit when the line right after `--` is a standard SQL column definition.
  if nl_pos != -1 and nl_pos + 1 < scan_lim and (m_fc := _FAST_COL_LINE_RE.match(text, nl_pos, scan_lim)) is not None:
    return (m_fc.end(1), None, nxt_bc)
  # Skip non-semicoloned `)` runs in C whenever `bounds[4]` proves a `;` close follows.
  while nl_pos != -1 and nl_pos + 1 < scan_lim:
    has_sc = _needs_semi_close(text, max(0, nl_pos - 1), scan_lim, bounds)
    if text.startswith(("\n);\n", "\n);\r\n"), nl_pos) and not (has_sc and bounds[4] > nl_pos) and not (scan_lim == bounds[3] < len(text) and _UNSEMI_END_RE.search(text, nl_pos + 3, scan_lim) is not None):
      return (nl_pos + 1, nl_pos + 3, nxt_bc)
    if has_sc and bounds[4] > nl_pos:
      ex = (_SPILLED_EXIT_NO_PAREN_NO_BC_RE if (nxt_bc == -1 or nl_pos >= last_bc) else _SPILLED_EXIT_NO_PAREN_RE).search(text, nl_pos, bounds[4])
      if ex is None:
        return (bounds[4] + 1, bounds[5], nxt_bc)
    else:
      ex = (_SPILLED_EXIT_NO_BC_RE if (nxt_bc == -1 or nl_pos >= last_bc) else _SPILLED_EXIT_RE).search(text, nl_pos, scan_lim)
    if ex is None or (p := m_li.end() if (m_li := _LEAD_INL_RE.match(text, ex.start() + 1, scan_lim)) is not None else ex.start() + 1) >= scan_lim:
      return (scan_lim, None, nxt_bc)
    if text[p] == ")":
      if not (text.startswith(";", p + 1) and scan_lim < len(text) and (scan_lim == bounds[3] or p > ex.start() + 1) and _UNSEMI_END_RE.search(text, p + 2, scan_lim) is not None):
        if text.startswith((";\n", ";\r\n"), p + 1) and not (has_sc and bounds[4] > p):
          return (p, p + 2, nxt_bc)
        if (ce := _match_table_close_tail(text, p, last_create, memo, _needs_semi_close(text, p, scan_lim, bounds))) is not None:
          return (p, ce, nxt_bc)
      nl_pos = text.find("\n", p + 1, scan_lim)
    elif text.startswith("/*", p):
      # Bound `nxt_bc` to `bc_lim` so unclosed `/*` never jumps past `\n);` into `t2`.
      aft, post_bc, bc_lim = p, p, (bounds[4] if _needs_semi_close(text, p, scan_lim, bounds) else scan_lim)
      while text.startswith("/*", aft):
        if p < last_bc and nxt_bc != -1 and nxt_bc < aft + 2:
          nxt_bc = text.find("*/", aft + 2, bc_lim)
        if p >= last_bc or nxt_bc == -1:
          aft = -1
          break
        post_bc = nxt_bc + 2
        aft = m_h.end() if (m_h := _LEAD_INL_RE.match(text, post_bc, scan_lim)) is not None else post_bc
      if aft == -1 or (aft < scan_lim and (text[aft] in "\r\n" or text.startswith("--", aft))):
        nl_pos = text.find("\n", (p + 2 if aft == -1 else aft), scan_lim)
      elif aft >= scan_lim:
        return (scan_lim, None, nxt_bc)
      elif text[aft] == ")":
        req_a = False if text.startswith((";\n", ";\r\n"), aft + 1) else _needs_semi_close(text, aft, scan_lim, bounds)
        if (ce := _match_table_close_tail(text, aft, last_create, memo, req_a)) is not None:
          return (aft, ce, nxt_bc)
        nl_pos = text.find("\n", aft + 1, scan_lim)
      elif _SPILLED_LINE_RE.match(text, post_bc, scan_lim) is not None:
        return (aft, None, nxt_bc)
      else:
        nl_pos = text.find("\n", aft, scan_lim)
    else:
      return (p, None, nxt_bc)
  return (scan_lim, None, nxt_bc)


def _scan_create_table_block(text: str, tbl_start: int, allow_spilled: bool, bounds: list[int], memo: list[int]) -> tuple[int | None, bool]:
  """Scans one `CREATE TABLE` statement in O(N) time (`allow_spilled` Pass 1/2)."""
  n = len(text)
  # Phase 1: Fast-match standard header or scan comments/quotes to `(` or `;`.
  if (hm := _TABLE_OPEN_HEAD_RE.match(text, tbl_start)) is not None:
    open_p, i = hm.end() - 1, hm.end()
  elif (kw_m := _CREATE_TABLE_KW_RE.match(text, tbl_start)) is None:
    return (None, False)
  else:
    i = kw_m.end()
    while tok := _DDL_HEAD_TOKEN_RE.search(text, i):
      if text[j := tok.start()] == "(":
        open_p, i = j, j + 1
        break
      if text[j] == ";":
        return (j + 1, False)
      i = _advance_comment_or_quote(text, j)
    else:
      return (None, False)
  if text.find(")", i) == -1:
    return (None, False)

  # Phase 2: Track `(...)` depth in C runs; clamp `scan_lim` eagerly in Pass 2.
  last_bc, last_create, last_q, q_lim = bounds[0], bounds[1], bounds[2], (min(n, bounds[2] + 1) if bounds[2] != -1 else n)
  if allow_spilled:
    if bounds[3] <= open_p and bounds[3] != n:
      bounds[3] = _next_create_bound(text, open_p + 1, last_create)
    scan_lim = bounds[3] if last_q <= open_p or bounds[3] < q_lim else q_lim
  else:
    scan_lim = n
  saw_dash, unclosed_tok, saw_open_p, depth, nxt_bc, q_end = False, False, False, 1, -2, open_p
  while i < scan_lim and (tok := _DDL_BODY_TOKEN_RE.search(text, i, scan_lim)) is not None:
    ch = text[j := tok.start()]
    if ch == "-":
      # Guard `last_q <= j` so an earlier table's `last_q` never underflows `scan_lim`.
      if not saw_dash or (allow_spilled and bounds[3] <= j and bounds[3] != n):
        saw_dash = True
        if allow_spilled:
          if bounds[3] <= j and bounds[3] != n:
            bounds[3] = _next_create_bound(text, j + 2, last_create)
          scan_lim = bounds[3] if last_q <= j or bounds[3] < q_lim else q_lim
      i = dm.end() if (dm := _DASH_RUN_RE.match(text, j, scan_lim)) is not None else min(scan_lim, _advance_comment_or_quote(text, j))
      bounds[5] = 1 if (not allow_spilled and (bounds[5] == 1 or depth > 1 or (nl_d := text.rfind("\n", j, i)) == -1 or (_FAST_COL_LINE_RE.match(text, nl_d, scan_lim) is None and not (text.rfind(",", open_p, j) <= text.rfind("\n", open_p, j) and _SEMI_CLOSE_LINE_RE.match(text, nl_d, scan_lim) is not None)))) else bounds[5]
      if depth == 1 and allow_spilled and i < scan_lim:
        i, ce, nxt_bc = _scan_spilled_prose(text, text.rfind("\n", j, i), scan_lim, last_bc, last_create, bounds, memo, -2 if nxt_bc == -1 and bounds[4] < j else nxt_bc)
        if ce is not None:
          return (ce, True)
    elif ch == "(":
      if not text.startswith("(((((((((((((((((", j) and (m_fp := _FLAT_PAREN_GROUP_RE.match(text, j, scan_lim)) is not None:
        saw_open_p, i = (saw_open_p or text.find("\n", j, m_fp.end()) != -1), m_fp.end()
      else:
        end_op = m_op.end() if (m_op := _OPEN_DEPTH_RUN_RE.match(text, j, scan_lim)) is not None else j + 1
        saw_open_p, depth, i = True, depth + text.count("(", j, end_op), end_op
    elif ch == ")":
      if depth > 1:
        end_cp = m_cp.end() if (m_cp := _CLOSE_DEPTH_RUN_RE.match(text, j, scan_lim)) is not None else j + 1
        if (closes := text.count(")", j, end_cp)) < depth:
          if not allow_spilled and saw_dash and text[j - 1] in "\n \t" and end_cp <= last_create and _CREATE_TABLE_KW_RE.match(text, _skip_sql_trivia(text, end_cp + (1 if text.startswith(";", end_cp) else 0), memo)) is not None:
            return (None, True)
          depth, i = depth - closes, end_cp
          continue
        for _ in range(depth - 1):
          j = text.find(")", j + 1, scan_lim)
        depth = 1
      if text.startswith((";\n", ";\r\n", "; CREATE ", "; create "), j + 1):
        return (j + 2, saw_dash)
      req_s = _needs_semi_close(text, j, scan_lim, bounds) if (allow_spilled and saw_dash) else False
      if (close_end := _match_table_close_tail(text, j, last_create, memo, req_s)) is not None:
        return (close_end, saw_dash)
      if not (allow_spilled and saw_dash):
        return (None, saw_dash)
      i, ce, nxt_bc = _scan_spilled_prose(text, text.find("\n", j + 1, scan_lim), scan_lim, last_bc, last_create, bounds, memo, nxt_bc)
      if ce is not None:
        return (ce, True)
      i = m_sk.end() if req_s and i < bounds[4] and (m_sk := _SKIP_TO_SEMI_RE.match(text, i, bounds[4])) is not None else i
    elif (r_re := _INLINE_BC_RUN_RE if ch == "/" else (_CLOSED_QUOTE_RUN_RE if ch in "'\"" else None)) is not None and (m_rn := r_re.match(text, j, scan_lim)) is not None:
      i = m_rn.end()
    else:
      nxt_i = _advance_comment_or_quote(text, j)
      if nxt_i > scan_lim and nxt_i < n and text.find("--", j, nxt_i) == -1:
        bounds[3], bounds[4] = _next_create_bound(text, nxt_i, last_create), -1
        scan_lim = bounds[3] if last_q <= nxt_i or bounds[3] < q_lim else q_lim
      if nxt_i > scan_lim or (nxt_i >= n and ch in "\"'`[/"):
        unclosed_tok = True
        if nxt_i > scan_lim:
          break
      elif text.find(")", j, nxt_i) != -1:
        q_end = nxt_i
      i = nxt_i

  # Fallback C `finditer` recovery bounded strictly to `[fb_from, scan_lim)`.
  fb_base = open_p if unclosed_tok else q_end
  has_semi = _needs_semi_close(text, fb_base, scan_lim, bounds) if allow_spilled else False
  if allow_spilled and (saw_dash or has_semi or bounds[3] < n or last_q > open_p) and (unclosed_tok or depth > 1 or saw_open_p or has_semi):
    if has_semi and (not unclosed_tok or text.startswith(("\n);\n", "\n);\r\n", "\n); "), bounds[4])):
      return (bounds[5], saw_dash)
    fb_from = bounds[4] if (has_semi and bounds[4] > fb_base) else (max(fb_base, text.rfind("\n", fb_base, max(fb_base, text.rfind("\n", fb_base, max(fb_base, scan_lim - 1))))) if (not has_semi and scan_lim == bounds[3] < n) else fb_base)
    for m in _CLOSE_PAREN_LINE_RE.finditer(text, fb_from, scan_lim):
      p = m.end(1) - 1
      req_p = False if text.startswith((";\n", ";\r\n"), p + 1) else _needs_semi_close(text, p, scan_lim, bounds)
      if (rec_end := _match_table_close_tail(text, p, last_create, memo, req_p)) is not None:
        return (rec_end, saw_dash)
  return (None, saw_dash)


def _embedded_hdr_end(text: str, hdr_pos: int, ceil: int, check_sect: bool, scan_from: int, ct_pos: list[int], ct_cache: dict[tuple[int, int], tuple[int | None, bool, int | None]]) -> int:
  """Returns close offset `> 0` when `hdr_pos` is inside DDL before `ceil`, else `0`."""
  # Short-circuit when zero `)` follow `hdr_pos`; memoize `(e1, d1, e2)` per `(last_ct, ceil)`.
  if text.find(")", hdr_pos + 1, ceil) == -1:
    return 0
  if not ct_pos:
    ct_pos.extend(m.start() for m in _CREATE_TABLE_KW_RE.finditer(text, scan_from))
  for idx in range(bisect.bisect_left(ct_pos, hdr_pos) - 1, -1, -1):
    if (key := ((last_ct := ct_pos[idx]), ceil)) not in ct_cache:
      e1, d1 = _scan_create_table_block(text, last_ct, False, [-1, ct_pos[-1], -1, -1, -1, -1, 0], [-1, -1])
      no_div = e1 is not None and (not d1 or (e1 > hdr_pos and ((dp0 := text.find("--", last_ct, hdr_pos)) == -1 or (nl0 := text.find("\n", dp0 + 2, hdr_pos)) == -1 or (_DDL_BODY_TOKEN_RE.search(text, nl0 + 1, hdr_pos) is None and text.find("--", hdr_pos, e1) == -1) or (text.find("--", nl0 + 1, text.rfind(")", hdr_pos, e1)) == -1 and _FAST_COL_LINE_RE.match(text, nl0, hdr_pos) is not None))))
      ct_cache[key] = (e1, False if no_div else d1, e1 if no_div else (min(e2_r, ceil) if (e2_r := _scan_create_table_block(text, last_ct, True, [ct_cache[(-1, ceil)][0] if (-1, ceil) in ct_cache else ct_cache.setdefault((-1, ceil), (text.rfind("*/", scan_from, ceil), False, None))[0], ct_pos[-1], (ceil - 1) if ceil < len(text) else -1, -1, -1, -1, 0], [-1, -1])[0]) is not None else None))
    if (ce0 := max((e1 := (ent := ct_cache[key])[0]) or 0, (e2 := ent[2]) or 0)) == 0 or (ce0 <= hdr_pos and (nx0 := _skip_sql_trivia(text, ce0, [-1, -1])) < hdr_pos and _POST_DDL_SUBQ_START_RE.match(text, nx0, hdr_pos) is None):
      continue
    if e1 is not None and not ent[1]:
      return e1 if (e1 > hdr_pos and _ASCII_WS_RE.match(text, e1, ceil).end() < ceil) else 0
    if e2 is not None and e2 <= hdr_pos and ((w2 := _ASCII_WS_RE.match(text, e2, hdr_pos).end()) >= hdr_pos or _SECT_HDR_RE.match(text, w2, hdr_pos) is not None) and (e1 is None or e1 <= hdr_pos or ((not any(q in (s_he := text[hdr_pos:e1]) for q in ("'", '"', "`", "]", "*/")) or (text.count("'", e2, hdr_pos) % 2 == 0 and s_he.count("'") % 2 == 1)) and _CREATE_TABLE_KW_RE.match(text, _skip_sql_trivia(text, e1, [-1, -1])) is None and (_BLANK_LINE_SEP_RE.match(text, e1) is None or _POST_Q_SECT_START_RE.match(text, _ASCII_WS_RE.match(text, e1, ceil).end(), ceil) is not None))):
      return 0
    if (ce := e1 if (e1 is not None and e1 > hdr_pos and (e2 is None or e2 <= hdr_pos or e2 >= e1)) else e2) is not None and ce > hdr_pos:
      pre, rp, hc_s = (text[last_ct:e1].rstrip(" \t\r\n") if e1 is not None else ""), _ASCII_WS_RE.match(text, ce, ceil).end(), _strip_closed_sql(text[hdr_pos:ce])
      if not (e1 is not None and e1 <= hdr_pos and ((w1 := _ASCII_WS_RE.match(text, e1, hdr_pos).end()) >= hdr_pos or _SECT_HDR_RE.match(text, w1, hdr_pos) is not None) and (pre.endswith(";") or "\n" not in pre or (_FAST_COL_LINE_RE.search(hc_s) is None and ((_ORPHAN_CLOSE_RE.search(pre, max(0, pre.rfind("\n"))) is not None and _TRAILING_COMMA_CLOSE_RE.search(pre) is None and ("(" in hc_s[:-1] or _BLANK_LINE_SEP_RE.match(text, ce) is None)) or ((m_s0 := _SEMI_CLOSE_LINE_RE.search(text, e1, hdr_pos)) is not None and text[m_s0.start() + 1] == ")" and _ASCII_WS_RE.match(text, m_s0.end(), hdr_pos).end() >= hdr_pos))))):
        return ce if rp < ceil else 0
      if (check_sect and (_BLANK_LINE_SEP_RE.match(text, ce) is None or _POST_Q_SECT_START_RE.match(text, rp, ceil) is not None)) or (not check_sect and (rp >= ceil or _SECT_HDR_RE.match(text, rp, ceil) is not None)):
        return 0
    break
  bal, prev_p = 0, hdr_pos + 1
  for m_uc in _UNMATCHED_TBL_CLOSE_RE.finditer(text, prev_p, ceil):
    bal, prev_p = bal + text.count("(", prev_p, cp := m_uc.start(1)) - text.count(")", prev_p, cp) - 1, cp + 1
    if bal < 0:
      return m_uc.end() if ((ru := _ASCII_WS_RE.match(text, m_uc.end(), ceil).end()) < ceil and (not (check_sect and text[m_uc.start()] == "\n") or _POST_Q_SECT_START_RE.match(text, ru, ceil) is None)) else 0
  return 0


def _run_ddl_pass(text: str, start_pos: int, allow_spilled: bool, last_bc: int, last_create: int, last_q: int, initial_end: int | None = None, has_q_hdr: bool = False) -> tuple[int | None, bool, bool]:
  """Runs one O(N) contiguous `CREATE TABLE` scan pass (`allow_spilled` False/True)."""
  # Initialize fresh monotonic trivia & bound cursors for this linear scan pass.
  memo, bounds = [-1, -1], [last_bc, last_create, last_q, -1, -1, -1, int(has_q_hdr)]
  last_end, any_dash, prev_dash = initial_end, False, False
  n, i = len(text), (initial_end if initial_end is not None else start_pos)
  fast_lim, semi_lim = ((fl := (min(n, last_q + 1) if last_q != -1 else n)), (m_bl.start() if (m_bl := _BLANK_LINE_SEP_RE.search(text, last_create, fl)) is not None else fl)) if allow_spilled else (n, n)
  while i < n:
    f_re = ((_FAST_SPILLED_SEMI_RE if (semi_i := text.find(";", i, i + 640)) != -1 and (text.find("'", i, semi_i) == -1 or _TYPE_SQ_LINE_RE.search(text, i, semi_i) is None) else (_FAST_SPILLED_UNSEMI_RE if i < last_create and text.find("\n)", i, i + 640) != -1 and text.find(";", i, semi_lim) == -1 else None)) if allow_spilled else (_FAST_SIMPLE_TABLES_RE if not prev_dash else None))
    if f_re is not None and i < fast_lim and (fast_m := f_re.match(text, i, fast_lim)) is not None:
      last_end = i = fast_m.end()
      continue
    if (j := i if text[i] in "cC" else (i + 1 if i + 1 < n and text[i] in "\n " and text[i + 1] in "cC" else _skip_sql_trivia(text, i, memo))) > last_create:
      break
    if allow_spilled and _ML_DELIM_RE.search(text, j, min(n, j + 4096)) is not None:
      bounds[3], bounds[4] = (_next_create_bound(text, j + 12, last_create) if j < last_create else n), -1
      if (c_end := _scan_create_table_block(text, j, False, [-1, last_create, -1, -1, -1, -1, 0], [-1, -1])[0]) is not None and c_end <= (t_lim := min(bounds[3], fast_lim)) and (_skip_sql_trivia(text, c_end, [-1, -1]) >= t_lim if t_lim < n else (_ASCII_WS_RE.match(text, c_end, n).end() < n and ((m_co := _ORPHAN_CLOSE_RE.search(text, c_end, n)) is None or _is_post_ddl_subquery(text, c_end, text.find(")", m_co.start() + 1), m_co.end(), n)))):
        last_end, i, any_dash, prev_dash = c_end, c_end, True, True
        continue
    stmt_end, saw_dash = _scan_create_table_block(text, j, allow_spilled, bounds, memo)
    any_dash, prev_dash = (any_dash or saw_dash), saw_dash
    if stmt_end is None:
      break
    last_end = i = stmt_end
  return (last_end, any_dash, bounds[5] == 1)


def _scan_ddl_prefix_end(text: str, start_pos: int) -> int | None:
  """Scans contiguous `CREATE TABLE` blocks via Two-Pass O(N) SQLite + spilled lexer."""
  # Consume any leading clean `CREATE TABLE ... (...);` prefix in one C regex call.
  init_end = fast_m.end() if (fast_m := _FAST_SIMPLE_TABLES_RE.match(text, start_pos)) is not None else None
  if (scan_from := init_end if init_end is not None else start_pos) >= (n := len(text)) or (m_fc := _CREATE_TABLE_KW_RE.search(text, scan_from)) is None:
    return init_end
  # Collect all blank-line section delimiters in one C pass (`0`–`4` items typical).
  sects, ct_pos, ct_cache = [(m.start(), m.end(), m.lastgroup) for m in _BLANK_SECT_RE.finditer(text, scan_from)], [], {}
  schema_ceil, first_q, last_ins = n, next((s for s, _, k in sects if k == "q"), n), max((s for s, _, k in sects if k == "ins"), default=-1)
  if first_q < n and any(k == "prev" for _, _, k in sects):
    det_lim = dets[-1][0] if (dets := [(s, e) for s, e, k in sects if k == "det"]) and not any(k in ("q", "ins") and s >= dets[-1][1] for s, _, k in sects) and not _embedded_hdr_end(text, dets[-1][0], n, False, scan_from, ct_pos, ct_cache) else n
    if (prv := next(((s, e) for s, e, k in reversed(sects) if k == "prev" and first_q < s < det_lim), None)) is not None and (q_prv := next((s for s, _, k in reversed(sects) if k == "q" and s < prv[0]), -1)) != -1 and (det_lim < n or (last_ins < prv[1] and text[:q_prv].rstrip(" \t\r\n").endswith((";", ")")))) and not _embedded_hdr_end(text, prv[0], det_lim, False, scan_from, ct_pos, ct_cache) and not _embedded_hdr_end(text, q_prv, prv[0], True, scan_from, ct_pos, ct_cache):
      schema_ceil = prv[0]
  if (first_ct := m_fc.start()) >= schema_ceil:
    return init_end
  last_create = max((m.start() for m in _CREATE_TABLE_KW_RE.finditer(text, max(first_ct, schema_ceil - 4096), schema_ceil)), default=-1)
  last_create, last_bc = (last_create if last_create != -1 else max(m.start() for m in _CREATE_TABLE_KW_RE.finditer(text, first_ct, schema_ceil))), text.rfind("*/", scan_from, schema_ceil)
  qs_after = [s for s, _, k in sects if k == "q" and first_ct <= s < schema_ceil]
  last_q = qs_after[-1] if qs_after and not _embedded_hdr_end(text, qs_after[-1], schema_ceil, True, scan_from, ct_pos, ct_cache) else -1
  # Pass 1: Pure standard SQLite lexer; return immediately when zero `--` exist.
  end1, saw_dash, sp_d = _run_ddl_pass(text, start_pos, False, last_bc, last_create, -1, init_end, (has_q_hdr := (last_q != -1)))
  if not saw_dash:
    return end1
  # Clamp `last_q` to any earlier non-embedded `External Evidence:` or `Instruction:` in O(N).
  p2_pre, q_ceil = (_run_ddl_pass(text, start_pos, True, last_bc, last_create, -1, init_end, has_q_hdr)[0] if sp_d and last_create > first_ct and any(k in ("ev", "ins") and first_ct < s < min(last_create, last_q if has_q_hdr else schema_ceil) for s, _, k in sects) else None), (last_q if has_q_hdr else schema_ceil)
  emb_u = (end1 or 0) if not sp_d else (min(p2_pre, last_create) if p2_pre is not None else 0)
  for s, _, k in sects:
    if k in ("ev", "ins") and first_ct < s < q_ceil and s >= emb_u and not (emb_u := _embedded_hdr_end(text, s, schema_ceil, False, scan_from, ct_pos, ct_cache)):
      last_q = s
      break
  if end1 is not None and (tail_lim := last_q if last_q != -1 else schema_ceil) >= 0 and not (sp_d and last_q > first_ct and end1 > last_q and _CLOSE_PAREN_LINE_RE.search(text, scan_from, last_q) is not None):
    if not (end1 <= last_create and _CREATE_TABLE_KW_RE.search(text, end1, tail_lim) is not None):
      if (m_or := (_ORPHAN_CLOSE_RE if has_q_hdr else _SEMI_CLOSE_LINE_RE).search(text, end1, tail_lim)) is None or ((last_q == -1 or _SECT_HDR_RE.match(text, _ASCII_WS_RE.match(text, end1, tail_lim).end()) is not None) and _is_post_ddl_subquery(text, end1, text.find(")", m_or.start() + 1), m_or.end(), tail_lim)):
        return end1
  # Pass 2: Hardened `scan_lim`-clamped spilled-prose scanner for `--` descriptions.
  end2 = p2_pre if (p2_pre is not None and last_q == -1) else _run_ddl_pass(text, start_pos, True, last_bc, last_create, last_q, last_create if (p2_pre is not None and p2_pre > last_create) else init_end, has_q_hdr)[0]
  return end2 if end2 is not None else end1


def strip_cached_schema_from_user_prompt(user_prompt: str) -> str:
  """Removes `Database Schema:\\n...\\n\\n` when `cachedContent` supplies DDL.

  Uses a deterministic single-pass `O(N)` SQL/DDL lexical scanner supporting
  both `\\n` and `\\r\\n`, Stage 4b / `B2` repair preambles, multi-line SQL
  string literals, unclosed multi-line column description comments, and
  `sqlite_master` `CREATE TABLE` blocks without trailing semicolons.

  Args:
    user_prompt: Original `LLMCompletionRequest.user_prompt` string.

  Returns:
    Prompt body retaining only dynamic question, evidence, diagnostics, and
    instructions (never returning an empty string).
  """
  # Locate `Database Schema:` at prompt start or after LF/CRLF blank line in O(N).
  hdr_span = _find_schema_header(user_prompt)
  if hdr_span is None:
    return user_prompt
  # Keep any Stage 4b / `B2` preamble + its paragraph separator untouched.
  schema_start, hdr_end = hdr_span
  ws_lead = _ASCII_WS_RE.match(user_prompt, hdr_end)
  ddl_start = ws_lead.end() if ws_lead is not None else hdr_end

  # Primary path: O(N) lexical scan across sequential `CREATE TABLE` blocks.
  ddl_end = _scan_ddl_prefix_end(user_prompt, ddl_start)
  if ddl_end is not None:
    ws_tail = _ASCII_WS_RE.match(user_prompt, ddl_end)
    cut_end = ws_tail.end() if ws_tail is not None else ddl_end
    candidate = user_prompt[:schema_start] + user_prompt[cut_end:]
  else:
    # Secondary path: non-DDL schema summary followed by `Question:`/`Query:`.
    end_match = _NON_DDL_QUESTION_DELIM_RE.search(user_prompt, ddl_start) or _BLANK_LINE_SEP_RE.search(user_prompt, ddl_start)
    candidate = user_prompt[:schema_start] + user_prompt[end_match.end() :] if end_match is not None else user_prompt

  # Never emit an empty `contents[0].parts[0].text` payload on degenerate input.
  cleaned = candidate.strip()
  return cleaned if cleaned else user_prompt.strip()


def build_cached_content_payload(*, project_id: str, location: str, model_name: str, db_id: str,
    cached_context_text: str, system_prompt: str = DEFAULT_SQLITE_SYSTEM_PROMPT,
    ttl_seconds: int = 3600, display_name: str | None = None) -> dict[str, object]:
  """Constructs a Vertex AI `CachedContent` REST JSON dictionary (`R1`).

  Args:
    project_id: Validated GCP billing project ID.
    location: Validated Vertex AI location (`'global'` or regional slug).
    model_name: Target Gemini model slug.
    db_id: Benchmark database identifier for `displayName` tagging.
    cached_context_text: Assembled schema DDL, descriptions, and cell dictionary.
    system_prompt: System instruction cached inside `CachedContent`.
    ttl_seconds: Cache time-to-live in `[1800, 3600]` seconds.
    display_name: Optional custom `CachedContent.displayName` override.

  Returns:
    JSON-serializable `CachedContent` creation payload dictionary.
  """
  # Validate TTL window and canonicalize full Vertex publisher model resource URI.
  valid_ttl = validate_cache_ttl_seconds(ttl_seconds)
  model_path = build_vertex_publisher_model_path(project_id, location, model_name)
  # Sanitize `displayName` to 128 URL-safe alphanumeric/hyphen/underscore chars.
  raw_display = display_name or f"argus-sql-db-{db_id.strip()}"
  safe_display = re.sub(r"[^A-Za-z0-9_-]++", "-", raw_display).strip("-")[:128]

  # Format `ttl` as `<seconds>s` without setting mutually exclusive `expireTime`.
  payload: dict[str, object] = {"model": model_path, "displayName": safe_display or "argus-sql-db-cache",
      "contents": [{"role": "user", "parts": [{"text": cached_context_text}]}], "ttl": f"{valid_ttl}s"}
  # Omit `"role"` inside `systemInstruction` per strict Vertex AI REST schema.
  if system_prompt.strip():
    payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
  return payload


class ContextCacheAdapter:
  """Per-`db_id` Vertex AI Context Caching context manager (`BatchLLMPort`).

  Creates a `CachedContent` resource on `__enter__`, attaches `"cachedContent"`
  while stripping `Database Schema:` on matching `batch_complete` calls, and
  guarantees `cachedContents.delete` teardown on `__exit__` (even on error).
  """

  def __init__(self, client: BatchLLMPort, db_id: str, *, schema: DatabaseSchema | None = None,
      cell_values: object = None, schema_ddl: str | None = None,
      descriptions: Mapping[object, str] | str | None = None, cell_value_dict: Mapping[str, Sequence[str]] | None = None,
      system_prompt: str = DEFAULT_SQLITE_SYSTEM_PROMPT, ttl_seconds: int = 3600,
      project_id: str | None = None, location: str | None = None, model_name: str | None = None,
      display_name: str | None = None, max_values_per_column: int = 64, include_metadata_comments: bool = True,
      fallback_on_create_error: bool = False, timeout_sec: float = 60.0, max_retries: int = 3,
      adc_file: pathlib.Path | None = None, token_supplier: Callable[[], tuple[str, float]] | None = None,
      http_transport: HttpTransportCallable | None = None) -> None:
    """Initializes per-`db_id` cached context text and Vertex REST transport.

    Args:
      client: Underlying `BatchLLMPort` (e.g., `VertexGeminiBatchClient`).
      db_id: Benchmark database identifier scoped to this context cache.
      schema: Optional `DatabaseSchema` rendered into the cached prefix.
      cell_values: Optional duck-typed `CellValueRetriever` or entry sequence.
      schema_ddl: Optional pre-rendered `CREATE TABLE` DDL string.
      descriptions: Optional table/column descriptions string or mapping.
      cell_value_dict: Optional explicit `table.column -> [values]` mapping.
      system_prompt: System prompt stored inside `CachedContent`.
      ttl_seconds: Cache TTL in seconds (`1800 <= ttl_seconds <= 3600`).
      project_id: Optional GCP project ID (defaults to `client` or env).
      location: Optional Vertex AI location (defaults to `client` or `'global'`).
      model_name: Optional Gemini model slug (defaults to `client` or env).
      display_name: Optional custom `CachedContent.displayName`.
      max_values_per_column: Maximum distinct cell literals cached per column.
      include_metadata_comments: Include column metadata comments in DDL.
      fallback_on_create_error: When True, falls back to uncached execution if
        `cachedContents.create` fails (e.g., schema `< min_tokens` HTTP 400).
      timeout_sec: Per-request HTTPS socket timeout in seconds.
      max_retries: Bounded retry attempts on transient 429/5xx HTTP errors.
      adc_file: Optional path override for Google Cloud ADC credentials JSON.
      token_supplier: Optional injectable `(token, ttl_sec)` provider for tests.
      http_transport: Optional injectable HTTP transport callable for tests.
    """
    # Store inner BatchLLMPort and precompute cached system-prompt SHA-256.
    self._llm = client
    self.db_id = db_id.strip()
    self.ttl_seconds = validate_cache_ttl_seconds(ttl_seconds)
    self._system_prompt = system_prompt
    self._system_prompt_sha256 = _sha256_text(system_prompt.strip())
    self._display_name = display_name
    self._fallback_on_create_error = fallback_on_create_error
    # Clamp retry budget to `[1, 6]` so transient failures remain bounded.
    self._timeout_sec = timeout_sec
    self._max_retries = max(1, min(max_retries, 6))
    self._http_transport = http_transport
    self._active_cache: VertexCachedContent | None = None
    self._multiline_descs = _collect_multiline_fragments(schema)

    # Resolve project, location, and model dynamically (`None` treated as `""`).
    raw_proj = project_id or _client_str_attr(client, "project_id") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    self.project_id = validate_gcp_project_id(raw_proj)

    # Default location to `'global'` matching Vertex Gemini Enterprise routing.
    raw_loc = location or _client_str_attr(client, "location") or os.environ.get("GOOGLE_CLOUD_LOCATION", "") or "global"
    self.location = validate_gcp_location(raw_loc)

    # Inherit target Gemini model slug from wrapped client unless overridden.
    raw_model = model_name or _client_str_attr(client, "model_name") or os.environ.get("ARGUS_LLM_MODEL", "") or "gemini-3.7-flash"
    self.model_name = normalize_gemini_model_name(raw_model)

    # Assemble deterministic DDL + descriptions + 3-gram cell dictionary once.
    self.cached_context_text = build_db_cached_context_text(self.db_id, schema=schema, cell_values=cell_values, schema_ddl=schema_ddl,
        descriptions=descriptions, cell_value_dict=cell_value_dict, include_metadata_comments=include_metadata_comments, max_values_per_column=max_values_per_column)

    # Enforce verified TLS 1.2+ and share `client._auth` when no override is set.
    self._ssl_ctx = ssl.create_default_context()
    self._ssl_ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    shared_auth = getattr(client, "_auth", None)
    if token_supplier is None and adc_file is None and isinstance(shared_auth, VertexTokenResolver):
      # Reuse client token cache so OAuth2 refreshes once across create + generate.
      self._auth = shared_auth
    else:
      # Instantiate isolated token resolver when custom ADC/supplier is injected.
      self._auth = VertexTokenResolver(ssl_ctx=self._ssl_ctx, adc_file=adc_file, token_supplier=token_supplier)

  @classmethod
  def for_database(cls, client: BatchLLMPort, db_id: str, *, schema: DatabaseSchema | None = None,
      cell_values: object = None, schema_ddl: str | None = None,
      descriptions: Mapping[object, str] | str | None = None, cell_value_dict: Mapping[str, Sequence[str]] | None = None,
      system_prompt: str = DEFAULT_SQLITE_SYSTEM_PROMPT, ttl_seconds: int = 3600,
      project_id: str | None = None, location: str | None = None, model_name: str | None = None,
      display_name: str | None = None, max_values_per_column: int = 64, include_metadata_comments: bool = True,
      fallback_on_create_error: bool = False, timeout_sec: float = 60.0, max_retries: int = 3,
      adc_file: pathlib.Path | None = None, token_supplier: Callable[[], tuple[str, float]] | None = None,
      http_transport: HttpTransportCallable | None = None) -> "ContextCacheAdapter":
    """Constructs a `ContextCacheAdapter` scoped to `db_id`.

    Args:
      client: Underlying `BatchLLMPort` completion client.
      db_id: Benchmark database identifier to cache.
      schema: Optional `DatabaseSchema` to render into DDL and descriptions.
      cell_values: Optional duck-typed `CellValueRetriever` or entry sequence.
      schema_ddl: Optional pre-rendered `CREATE TABLE` DDL string.
      descriptions: Optional table/column descriptions string or mapping.
      cell_value_dict: Optional explicit `table.column -> [values]` mapping.
      system_prompt: System instruction stored in `CachedContent`.
      ttl_seconds: Cache time-to-live in `[1800, 3600]` seconds.
      project_id: Optional GCP project ID override.
      location: Optional Vertex AI location override.
      model_name: Optional Gemini model slug override.
      display_name: Optional custom `CachedContent.displayName`.
      max_values_per_column: Max distinct literals serialized per column.
      include_metadata_comments: Include column metadata comments in DDL.
      fallback_on_create_error: Fall back to uncached execution on create error.
      timeout_sec: Per-request HTTPS socket timeout in seconds.
      max_retries: Bounded retry attempts on transient 429/5xx HTTP errors.
      adc_file: Optional path override for Google Cloud ADC credentials JSON.
      token_supplier: Optional injectable `(token, ttl_sec)` provider for tests.
      http_transport: Optional injectable HTTP transport callable for tests.

    Returns:
      Configured `ContextCacheAdapter` ready for `with ... as cached_llm:`.
    """
    # Forward all validated per-database context options to `__init__`.
    return cls(client, db_id, schema=schema, cell_values=cell_values, schema_ddl=schema_ddl, descriptions=descriptions,
        cell_value_dict=cell_value_dict, system_prompt=system_prompt, ttl_seconds=ttl_seconds, project_id=project_id,
        location=location, model_name=model_name, display_name=display_name, max_values_per_column=max_values_per_column,
        include_metadata_comments=include_metadata_comments, fallback_on_create_error=fallback_on_create_error,
        timeout_sec=timeout_sec, max_retries=max_retries, adc_file=adc_file, token_supplier=token_supplier, http_transport=http_transport)

  @property
  def active_cache(self) -> VertexCachedContent | None:
    """Returns the currently active `VertexCachedContent` handle, if any."""
    # Expose read-only handle so callers cannot mutate lifecycle state directly.
    return self._active_cache

  def _execute_http_once(self, method: str, url: str, payload_bytes: bytes | None, bearer_token: str) -> tuple[int, bytes]:
    """Executes one HTTPS request via injected transport or `urllib.request`."""
    # Enforce HTTPS scheme and omit sensitive headers from all log paths.
    if not url.startswith("https://"):
      raise LLMAdapterAuthError("Vertex cachedContents URL must use 'https://'.")
    # Attach `x-goog-user-project` so quota and cache storage bill `project_id`.
    headers: dict[str, str] = {"Authorization": f"Bearer {bearer_token}", "x-goog-user-project": self.project_id}
    if payload_bytes is not None:
      headers["Content-Type"] = "application/json"

    # Delegate directly to hermetic callable during offline unit tests.
    if self._http_transport is not None:
      return self._http_transport(method, url, payload_bytes, headers, self._timeout_sec)
    # Execute live TLS 1.2+ request over Python stdlib `urllib.request`.
    http_req = urllib.request.Request(url, data=payload_bytes, headers=headers, method=method)
    with urllib.request.urlopen(http_req, timeout=self._timeout_sec, context=self._ssl_ctx) as resp:
      return (int(resp.status), resp.read())

  def _request_json_with_retry(self, method: str, url: str, payload_bytes: bytes | None, *, allow_404: bool = False) -> dict[str, object]:
    """Executes a `cachedContents` HTTPS call with 401 refresh and retry."""
    # Acquire initial OAuth2 bearer token prior to entering the retry loop.
    bearer_token = self._auth.get_bearer_token()
    refreshed_on_401 = False
    last_err: Exception | None = None

    # Isolate single HTTP execution in `self._execute_http_once` (no N+1 loop).
    for attempt in range(self._max_retries):
      http_err: urllib.error.HTTPError | None = None
      status = 0
      try:
        status, raw_body = self._execute_http_once(method, url, payload_bytes, bearer_token)
        # Decode 2xx JSON response body (treating empty `DELETE` bodies as `{}`).
        if 200 <= status < 300:
          if not raw_body.strip():
            return {}
          parsed = json.loads(raw_body.decode("utf-8"))
          if not isinstance(parsed, dict):
            raise LLMAdapterTransportError("Vertex cachedContents returned non-object JSON.")
          return parsed
      except urllib.error.HTTPError as err:
        # Capture `HTTPError` so status triage below preserves `__cause__`.
        http_err, last_err, status = err, err, err.code
      except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as err:
        last_err, status = err, 0
        # Surface socket/HTTP/decode fault once retry budget is exhausted.
        if attempt + 1 >= self._max_retries:
          _LOGGER.warning("Vertex cachedContents %s failed transport.", method)
          raise LLMAdapterTransportError(f"Vertex cachedContents {method} failed transport.") from err

      # Unify 404/401/retry status triage across mock tuples and `HTTPError`.
      if status:
        # Treat HTTP 404 on `DELETE` as idempotent success when TTL expired early.
        if status == 404 and allow_404:
          _LOGGER.debug("Vertex cachedContent already absent (HTTP 404).")
          return {}
        # Evict expired OAuth2 token once on HTTP 401 and retry without sleeping.
        if status == 401 and not refreshed_on_401 and attempt + 1 < self._max_retries:
          refreshed_on_401 = True
          self._auth.invalidate_token()
          bearer_token = self._auth.get_bearer_token()
          continue
        # Fail closed immediately on non-retryable 4xx or exhausted retry budget.
        if status not in _RETRYABLE_HTTP_CODES or attempt + 1 >= self._max_retries:
          _LOGGER.warning("Vertex cachedContents %s failed with HTTP %d.", method, status)
          raise LLMAdapterTransportError(f"Vertex cachedContents {method} failed (HTTP {status}).") from http_err

      # Apply OS CSPRNG jittered exponential backoff before next transient retry.
      jitter_sec = secrets.randbelow(500) / 1000.0
      time.sleep((2.0**attempt) + jitter_sec)

    # Defensive fallback if retry loop terminates without returning or raising.
    raise LLMAdapterTransportError(f"Vertex cachedContents {method} exhausted retries.") from last_err

  def create_cache(self) -> VertexCachedContent | None:
    """Creates the server-side Vertex AI `CachedContent` resource for `db_id`.

    Returns:
      Created `VertexCachedContent` handle, or `None` when
      `fallback_on_create_error=True` and creation fails.

    Raises:
      LLMAdapterAuthError: If credentials or returned resource name are invalid.
      LLMAdapterTransportError: If creation fails and fallback is disabled.
    """
    # Return existing handle idempotently if `create_cache` was already called.
    if self._active_cache is not None:
      return self._active_cache

    # Build validated `/v1beta1/projects/{p}/locations/{l}/cachedContents` URL.
    create_url = build_vertex_cached_contents_url(self.project_id, self.location)
    body_dict = build_cached_content_payload(project_id=self.project_id, location=self.location, model_name=self.model_name,
        db_id=self.db_id, cached_context_text=self.cached_context_text,
        system_prompt=self._system_prompt, ttl_seconds=self.ttl_seconds, display_name=self._display_name)
    payload_bytes = json.dumps(body_dict).encode("utf-8")

    # Optionally degrade to uncached execution when schema is below token floor.
    try:
      resp_json = self._request_json_with_retry("POST", create_url, payload_bytes)
    except LLMAdapterTransportError as err:
      if not self._fallback_on_create_error:
        raise
      _LOGGER.warning("Vertex context cache creation failed for db_id=%s (%s); falling back to uncached execution.", self.db_id, err)
      self._active_cache = None
      return None

    # Validate canonical name (accepting numeric project IDs + matching location).
    raw_name = resp_json.get("name")
    cache_name = validate_vertex_cached_content_name(raw_name if isinstance(raw_name, str) else "")
    build_vertex_cached_content_item_url(self.location, cache_name)
    # Coalesce explicit JSON `null` on `usageMetadata`/`expireTime`; reject `bool`.
    usage = {} if resp_json.get("usageMetadata") is None else resp_json.get("usageMetadata")
    if not isinstance(usage, dict):
      raise LLMAdapterTransportError("Malformed usageMetadata in CachedContent.")
    tok_count = coerce_nonneg_int(usage.get("totalTokenCount"), 0)
    raw_exp = resp_json.get("expireTime")
    if raw_exp is not None and not isinstance(raw_exp, str):
      raise LLMAdapterTransportError("Malformed expireTime in CachedContent.")
    # Record immutable cache metadata handle for subsequent `batch_complete` calls.
    self._active_cache = VertexCachedContent(cache_name=cache_name, db_id=self.db_id, model_name=self.model_name,
        ttl_seconds=self.ttl_seconds, expire_time=raw_exp or "", total_token_count=tok_count, system_prompt_sha256=self._system_prompt_sha256)
    _LOGGER.info("Created Vertex cachedContent %s for db_id=%s (tokens=%d, ttl=%ds).", cache_name, self.db_id, tok_count, self.ttl_seconds)
    return self._active_cache

  def delete_cache(self) -> bool:
    """Deletes the active `CachedContent` resource on Vertex AI if present.

    Returns:
      True if a `DELETE` request completed (`200` or idempotent `404`); False
      when no active cache existed.

    Raises:
      LLMAdapterAuthError: If cache resource validation fails.
      LLMAdapterTransportError: If `DELETE` fails with a non-404 status.
    """
    # Clear `_active_cache` first so teardown executes at most once per handle.
    cache = self._active_cache
    self._active_cache = None
    if cache is None:
      return False

    # Construct location-verified item URL and issue idempotent `DELETE` request.
    delete_url = build_vertex_cached_content_item_url(self.location, cache.cache_name)
    self._request_json_with_retry("DELETE", delete_url, payload_bytes=None, allow_404=True)
    _LOGGER.info("Deleted Vertex cachedContent %s for db_id=%s.", cache.cache_name, self.db_id)
    return True

  def __enter__(self) -> Self:
    """Creates the `CachedContent` resource upon entering the `with` block."""
    # Provision server-side schema context cache before yielding `self`.
    self.create_cache()
    return self

  def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> bool:
    """Deletes the `CachedContent` resource on exit without masking errors."""
    # On clean block exit, surface any `delete_cache` error to the caller.
    if exc_type is None:
      self.delete_cache()
      return False
    # During active exception unwind, catch `Exception` so teardown never masks
    # the body error escaping the `with` block.
    try:
      self.delete_cache()
    except Exception as cleanup_err:
      _LOGGER.warning("Failed deleting Vertex cachedContent for db_id=%s during unwind: %s", self.db_id, cleanup_err)
    return False

  def batch_complete(self, requests: Sequence[LLMCompletionRequest]) -> tuple[LLMCompletionResponse, ...]:
    """Completes `requests` attaching `cachedContent` and stripping cached DDL.

    Args:
      requests: Sequence of `LLMCompletionRequest` items to execute.

    Returns:
      Tuple of `LLMCompletionResponse` items aligned 1-to-1 with `requests`.
    """
    # Short-circuit empty sequences and uncached fallback contexts cleanly.
    if not requests:
      return ()
    active = self._active_cache
    if active is None:
      return self._llm.batch_complete(requests)

    rewritten: list[LLMCompletionRequest] = []
    for req in requests:
      req_sys_hash = _sha256_text(req.system_prompt.strip())
      # Attach `cachedContent` only when `system_prompt` matches cached instructions
      # so `B4_LLMJudge` tie-breakers automatically bypass the SQL synthesis cache.
      if req_sys_hash == active.system_prompt_sha256:
        raw_prompt = req.user_prompt
        # Collapse `-- <multiline_desc>` inside the schema slice only (`[hdr[1]:end_p]`).
        if self._multiline_descs and (hdr := _find_schema_header(raw_prompt)) is not None:
          end_p = idx + 1 if (idx := raw_prompt.rfind(")")) > hdr[1] else len(raw_prompt)
          mid = raw_prompt[hdr[1] : end_p]
          for multi_frag, flat_frag in self._multiline_descs:
            if multi_frag in mid:
              mid = mid.replace(multi_frag, flat_frag)
          raw_prompt = raw_prompt[: hdr[1]] + mid + raw_prompt[end_p:]
        stripped_prompt = strip_cached_schema_from_user_prompt(raw_prompt)
        rewritten.append(dataclasses.replace(req, user_prompt=stripped_prompt, cached_content=req.cached_content or active.cache_name))
      else:
        # Preserve original DDL and inline `systemInstruction` on prompt mismatch.
        _LOGGER.debug("Bypassing cachedContent for request %s (system_prompt mismatch).", req.request_id)
        rewritten.append(req)

    # Dispatch the rewritten batch once through the underlying `BatchLLMPort`.
    return self._llm.batch_complete(tuple(rewritten))
