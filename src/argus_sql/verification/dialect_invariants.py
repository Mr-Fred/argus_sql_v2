"""SQLite temporal, window-rank, and projection arity dialect invariants.

Implements three domain-general SQLite invariants with zero runtime I/O:
1. `TIMESTAMP_BARE_DATE_COMPARISON` (I11): Detects bare `'YYYY-MM-DD'` literal
   comparisons (`=`, `!=`, `<>`, `>`, `<=`, `BETWEEN`) against sub-day ISO
   timestamp columns (`range: 'YYYY-MM-DD HH:MM...'`) where ASCII string
   ordering misclassifies same-day timestamps, plus a deterministic zero-token
   `DATE(col)` algebraic rewrite helper.
2. `MISSING_WINDOW_RANK` (I12): Detects imperative `'Rank <entities> by <metric>'`
   analytical queries that omit a SQLite `RANK() OVER (ORDER BY ...)` column.
3. `PROJECTION_ARITY_DRIFT`: Audits outer wildcard (`SELECT *` / `table.*`)
   over-projections (escalated to `ERROR` when `expected_projection_arity <= 2`),
   strict single-column drift, and 1-column follow-up under-projections.
"""

import re

from argus_sql.contracts.types import ExecutionOutcome
from argus_sql.contracts.types import IntentFrame
from argus_sql.contracts.types import InvariantIssue
from argus_sql.verification.relational_invariants import mask_outer_sql

# Extracts `CREATE TABLE <name> ( ... );` blocks without stopping on `);` inside
# inline `--` comments or quoted literals.
_DDL_BLOCK_RE = re.compile(
    r'CREATE\s+TABLE\s+(?:"((?:""|[^"])+)"|([a-zA-Z0-9_]+))\s*\('
    r"((?:--[^\n]*|'(?:''|[^'])*'|[^;])*?)\)\s*;",
    re.IGNORECASE | re.DOTALL,
)

# Extracts bare or double-quoted column identifier at the start of a DDL line.
_DDL_COL_HEAD_RE = re.compile(r'^(?:"((?:""|[^"])+)"|([a-zA-Z0-9_]+))\s+')

# Detects sub-day ISO timestamps (`'2014-09-01 14:23:10'` or `'2014-09-01T...'`)
# inside a rendered column `-- ... range: ...` comment.
_SUBDAY_TS_RANGE_RE = re.compile(
    r"\brange:\s*'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"
    r"|\.\.'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"
)

# Detects pure calendar date ranges (`'2012-08-23'..'2012-08-26'`) lacking time.
_PURE_DATE_RANGE_RE = re.compile(
    r"\brange:\s*'\d{4}-\d{2}-\d{2}'\.\.'\d{4}-\d{2}-\d{2}'"
)

# SQL clause keywords excluded from implicit table alias capture.
_SQL_CLAUSE_KW = (
    r"(?:AS|WHERE|ON|INNER|LEFT|RIGHT|FULL|CROSS|OUTER|NATURAL|INDEXED|JOIN"
    r"|GROUP|ORDER|HAVING|LIMIT|UNION|INTERSECT|EXCEPT|SET|VALUES|WINDOW)"
)

# Extracts `FROM/JOIN <table> [[AS] <alias>]` bindings for column resolution.
_TABLE_ALIAS_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+"
    r'(?:"((?:""|[^"])+)"|`((?:``|[^`])+)`|\[([^\]]+)\]|([a-zA-Z_][a-zA-Z0-9_]*))'
    rf"(?:\s+(?:AS\s+)?(?!{_SQL_CLAUSE_KW}\b)"
    r'(?:"((?:""|[^"])+)"|`((?:``|[^`])+)`|\[([^\]]+)\]|([a-zA-Z_][a-zA-Z0-9_]*)))?',
    re.IGNORECASE,
)

# Ordered token matcher skipping SQL comments and `'...'` literals first, then
# capturing bare or alias-qualified columns compared to `'YYYY-MM-DD'` literals.
_STRING_OR_BARE_DATE_CMP_RE = re.compile(
    r"--[^\n]*|/\*[\s\S]*?\*/|'(?:''|[^'])*'|"
    r"(?<![a-zA-Z0-9_.\"`\[])"
    r"((?:([a-zA-Z_][a-zA-Z0-9_]*)\.)?"
    r'(?:"((?:""|[^"])+)"|`((?:``|[^`])+)`|\[([^\]]+)\]|([a-zA-Z_][a-zA-Z0-9_]*)))'
    r"\s*(?:(?:=|!=|<>|>|<=)\s*'\d{4}-\d{2}-\d{2}'"
    r"|(?:NOT\s+)?BETWEEN\s+'\d{4}-\d{2}-\d{2}'\s+AND\s+'\d{4}-\d{2}-\d{2}'"
    r"|(?:NOT\s+)?IN\s*\(\s*'\d{4}-\d{2}-\d{2}'(?:\s*,\s*'\d{4}-\d{2}-\d{2}')*\s*\))",
    re.IGNORECASE,
)

# Matches imperative `'Rank <entities> by <metric>'` analytical questions.
_IMPERATIVE_RANK_RE = re.compile(
    r"^(?:please\s+)?rank\s+(?!of\b)\S+[\s\S]*\bby\b",
    re.IGNORECASE,
)

# Matches SQLite window ranking functions (`RANK() OVER`, `DENSE_RANK() OVER`).
_WINDOW_RANK_SQL_RE = re.compile(
    r"\b(?:DENSE_RANK|RANK|ROW_NUMBER)\s*\(\s*\)\s*OVER\s*\(",
    re.IGNORECASE,
)


def extract_ddl_timestamp_columns(
    ddl_prompt: str,
) -> tuple[set[tuple[str, str]], set[str]]:
  """Identifies `(lower_table, lower_col)` pairs storing sub-day ISO timestamps.

  Args:
    ddl_prompt: Rendered `CREATE TABLE` DDL string containing `range:` notes.

  Returns:
    Tuple `(timestamp_table_cols, unambiguous_timestamp_cols)` extracted purely
    in-memory from `ddl_prompt`.
  """
  ts_pairs: set[tuple[str, str]] = set()
  pure_date_cols: set[str] = set()
  if "range:" not in ddl_prompt:
    return (ts_pairs, set())

  # Inspect each table block's inline `-- ... range:` bounds without DB I/O.
  for match in _DDL_BLOCK_RE.finditer(ddl_prompt):
    t_name = (
        match.group(1).replace('""', '"') if match.group(1) else match.group(2)
    ).lower()
    for raw_line in match.group(3).splitlines():
      if "range:" not in raw_line or "--" not in raw_line:
        continue
      code_part, _, comment_part = raw_line.partition("--")
      col_m = _DDL_COL_HEAD_RE.match(code_part.strip())
      if col_m is None:
        continue
      c_name = (
          col_m.group(1).replace('""', '"') if col_m.group(1) else col_m.group(2)
      ).lower()
      if _SUBDAY_TS_RANGE_RE.search(comment_part):
        ts_pairs.add((t_name, c_name))
      elif _PURE_DATE_RANGE_RE.search(comment_part):
        pure_date_cols.add(c_name)

  # Unqualified column names are unambiguous when no rendered table has a pure
  # `YYYY-MM-DD` date column sharing the identical identifier.
  unambiguous = {col for _, col in ts_pairs} - pure_date_cols
  return (ts_pairs, unambiguous)


def _extract_table_alias_map(sql_text: str) -> dict[str, str]:
  """Maps `alias_or_table_lower -> table_lower` across `FROM` and `JOIN`."""
  # Mask string literals first so `'FROM ...'` inside text cannot fake a table.
  no_strings = re.sub(r"'(?:''|[^'])*'", "''", sql_text)
  alias_map: dict[str, str] = {}
  for m in _TABLE_ALIAS_RE.finditer(no_strings):
    raw_tbl = m.group(1) or m.group(2) or m.group(3) or m.group(4) or ""
    tbl_low = raw_tbl.replace('""', '"').replace("``", "`").lower()
    if not tbl_low:
      continue
    # Register both physical table name and optional SQL alias in lowercase.
    alias_map.setdefault(tbl_low, tbl_low)
    raw_alias = m.group(5) or m.group(6) or m.group(7) or m.group(8) or ""
    alias_low = raw_alias.replace('""', '"').replace("``", "`").lower()
    if alias_low:
      alias_map[alias_low] = tbl_low
  return alias_map


def find_timestamp_bare_date_rewrites(
    sql_text: str,
    ddl_prompt: str,
) -> list[tuple[int, int, str]]:
  """Locates bare timestamp column expressions compared to `'YYYY-MM-DD'`.

  Args:
    sql_text: Candidate SQL query string to inspect.
    ddl_prompt: Rendered `CREATE TABLE` DDL string containing `range:` notes.

  Returns:
    List of `(start_idx, end_idx, col_expr)` spans suitable for wrapping as
    `DATE(col_expr)`.
  """
  ts_pairs, unambiguous_cols = extract_ddl_timestamp_columns(ddl_prompt)
  if not ts_pairs:
    return []

  # Resolve table aliases so `t2.date` checks `t2`'s specific DDL range.
  alias_map = _extract_table_alias_map(sql_text)
  rewrites: list[tuple[int, int, str]] = []
  for match in _STRING_OR_BARE_DATE_CMP_RE.finditer(sql_text):
    # Branch 1 matches `'...'` string literals (`group(1)` is None) -> skip.
    col_expr = match.group(1)
    if col_expr is None:
      continue
    qual = (match.group(2) or "").lower()
    raw_col = (
        match.group(3)
        or match.group(4)
        or match.group(5)
        or match.group(6)
        or ""
    )
    col_low = raw_col.replace('""', '"').replace("``", "`").lower()
    # Skip boolean conjunction tokens that can precede `BETWEEN` clauses.
    if not col_low or col_low in {"and", "or", "not", "where", "on", "when"}:
      continue

    # Check base-table alias first; fall back to unambiguous cols for CTEs.
    referenced_tables = set(alias_map.values())
    is_ts_col = False
    if qual and qual in alias_map:
      is_ts_col = (alias_map[qual], col_low) in ts_pairs
    elif referenced_tables:
      in_query_ts = any((t, col_low) in ts_pairs for t in referenced_tables)
      is_ts_col = in_query_ts and col_low in unambiguous_cols
    else:
      is_ts_col = col_low in unambiguous_cols

    if is_ts_col:
      rewrites.append((match.start(1), match.end(1), col_expr))
  return rewrites


def check_timestamp_bare_date_comparison(
    sql_text: str,
    ddl_prompt: str,
) -> InvariantIssue | None:
  """Flags bare `'YYYY-MM-DD'` comparisons against sub-day timestamp columns.

  Args:
    sql_text: Candidate SQL query string.
    ddl_prompt: Rendered `CREATE TABLE` DDL string containing `range:` notes.

  Returns:
    ERROR-level InvariantIssue if a bare date comparison is found, else None.
  """
  # Reuse exact rewrite locator so verification and repair never diverge.
  rewrites = find_timestamp_bare_date_rewrites(sql_text, ddl_prompt)
  if not rewrites:
    return None
  col_expr = rewrites[0][2]
  return InvariantIssue(
      code="TIMESTAMP_BARE_DATE_COMPARISON",
      severity="ERROR",
      message=(
          f"Timestamp column `{col_expr}` stores sub-day ISO timestamps "
          "(`YYYY-MM-DD HH:MM:SS`), so bare `'YYYY-MM-DD'` string comparisons "
          "(`=`, `>`, `<=`, `BETWEEN`) misclassify same-day rows in SQLite."
      ),
      repair_hint=(
          f"Wrap `{col_expr}` in `DATE({col_expr})` before comparing against "
          "`'YYYY-MM-DD'` date literals."
      ),
  )


def rewrite_timestamp_bare_date_comparisons(
    sql_text: str,
    ddl_prompt: str,
) -> str:
  """Wraps bare timestamp columns in `DATE(col)` on `'YYYY-MM-DD'` predicates.

  Args:
    sql_text: Candidate SQL string to rewrite.
    ddl_prompt: Rendered `CREATE TABLE` DDL string containing `range:` notes.

  Returns:
    Rewritten SQL string with `DATE(<col>)` applied right-to-left.
  """
  rewrites = find_timestamp_bare_date_rewrites(sql_text, ddl_prompt)
  if not rewrites:
    return sql_text
  # Apply replacements from right to left so earlier string offsets stay valid.
  updated = sql_text
  for start_idx, end_idx, col_expr in reversed(rewrites):
    updated = f"{updated[:start_idx]}DATE({col_expr}){updated[end_idx:]}"
  return updated


def check_missing_window_rank(
    no_quotes_sql: str,
    intent: IntentFrame,
) -> InvariantIssue | None:
  """Flags imperative `'Rank X by Y'` queries lacking `RANK() OVER (...)`.

  Args:
    no_quotes_sql: Candidate SQL with quoted literals and identifiers masked.
    intent: Canonicalized IntentFrame describing question semantics.

  Returns:
    ERROR-level InvariantIssue when a `'Rank X by Y'` query omits `OVER (`,
    else None.
  """
  # Restrict to imperative `Rank <entities> by <metric>` prompts (`0` false
  # positives on scalar `'In which race did X rank highest?'` questions).
  if _IMPERATIVE_RANK_RE.match(intent.canonical_question) is None:
    return None
  # Accept `RANK() OVER`, `DENSE_RANK() OVER`, or `ROW_NUMBER() OVER`.
  if _WINDOW_RANK_SQL_RE.search(no_quotes_sql) is not None:
    return None
  return InvariantIssue(
      code="MISSING_WINDOW_RANK",
      severity="ERROR",
      message=(
          "Imperative `'Rank ... by ...'` analytical query omits a SQLite "
          "`RANK() OVER (ORDER BY ...)` window ranking column."
      ),
      repair_hint=(
          "Project the entity identifier, the ranking metric, and "
          "`RANK() OVER (ORDER BY <metric> DESC|ASC)`."
      ),
  )


# Matches quoted literals/identifiers before SQL comments for safe wildcard scans.
_SQL_QUOTED_OR_COMMENT_RE = re.compile(
    r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]"
    r"|/\*[\s\S]*?\*/|--[^\n]*"
)

# Detects interrogative/imperative head sentences for fallback follow-up checks.
_QUERY_HEAD_RE = re.compile(
    r"(?:^(?:please\s+)?(?:what|which|who|whom|whose|where|when|how|"
    r"list|give|show|provide|indicate|include|identify|find|state|"
    r"report|display|name|tell|return)\b"
    r"|,\s*(?:please\s+)?(?:list|give|show|provide|indicate|include|"
    r"identify|find|state|report|display|name|what\s+(?:is|are|was|were))\b)",
    re.IGNORECASE,
)


def has_outer_wildcard_projection(sql_text: str) -> bool:
  """Returns True when a top-level SELECT clause projects `*` or `alias.*`.

  Args:
    sql_text: Candidate SQL query string to inspect.

  Returns:
    True if an outer SELECT item is `*` or `<alias>.*` (ignoring `COUNT(*)`,
    comments, string literals, and parenthesized CTEs/subqueries).
  """
  # Strip comments while preserving quotes so `COUNT(*)` and CTE `*` mask out.
  no_comments = _SQL_QUOTED_OR_COMMENT_RE.sub(
      lambda m: m.group(0) if m.group(0)[0] in ("'", '"', "`", "[") else " ",
      sql_text,
  )
  outer_sql = mask_outer_sql(no_comments)
  for sel_m in re.finditer(
      r"\bSELECT\s+(?:DISTINCT\s+|ALL\s+)?([\s\S]+?)(?=\bFROM\b|$)",
      outer_sql,
      re.IGNORECASE,
  ):
    for item in sel_m.group(1).split(","):
      tok = item.strip()
      if tok == "*" or tok.endswith(".*"):
        return True
  return False


def _has_fallback_followup_cue(question: str) -> bool:
  """Detects multi-column follow-ups on manually constructed IntentFrames."""
  # Require the head sentence before `.`/`;` or additive joins to carry an
  # actual query directive so `[Context]. [1-col question]` never misfires.
  for m in re.finditer(
      r"(?:([.;?])\s*(?:and\s+)?(?:please\s+)?(?:also\s+)?"
      r"(?:indicate|include|provide|show|identify|state|list|give|mention|"
      r"report|display|name|what\s+(?:is|are|was|were))\s+"
      r"|\b(?:along\s+with|as\s+well\s+as|together\s+with)\s+"
      r"|\b(?:and\s+)?(?:also|please)\s+(?:indicate|include|provide|show|"
      r"identify|state|list|give|mention|report|display|name)\s+)",
      question,
      re.IGNORECASE,
  ):
    sep = m.group(1) or ""
    last_head = re.split(r"[.;?]+\s*", question[: m.start()].strip())[-1].strip()
    if sep == "?" or _QUERY_HEAD_RE.search(last_head) is not None:
      return True
  # Anchor possessive lists at clause heads and strip trailing filter tails
  # before checking for comma/`and` multi-attribute conjunctions.
  their_m = re.search(
      r"(?:^|[.;?]\s*|,\s*|\band\s+)(?:please\s+)?(?:also\s+)?"
      r"(?:list|give|show|provide|indicate|include|identify|find|state|"
      r"report|display|name|what\s+(?:are|were))\s+(?:out\s+|me\s+)?"
      r"(?:all\s+(?:of\s+)?)?(?:their|his|her|its)\s+([^?.!;]+)",
      question,
      re.IGNORECASE,
  )
  if their_m is not None:
    core = re.sub(
        r"\s+\b(?:for|between|from|during|where|who|whose|when|having|"
        r"among|that|order(?:ed)?\s+by|in\s+(?:the\s+|\d|[A-Z])).*$",
        "",
        their_m.group(1).strip(),
    )
    return bool(re.search(r",|\band\b", core, re.IGNORECASE))
  return False


def check_projection_arity_drift(
    sql_text: str,
    outcome: ExecutionOutcome,
    intent: IntentFrame,
) -> InvariantIssue | None:
  """Audits projection arity, wildcard `*` escalation, and 1-col follow-ups.

  Args:
    sql_text: Candidate SQL query string.
    outcome: Executed sandbox outcome carrying projected `column_names`.
    intent: Canonicalized IntentFrame describing expected projection arity.

  Returns:
    InvariantIssue (`ERROR` for `SELECT *` when `expected <= 2`, else
    `WARNING`) when projection arity drifts, or None if aligned.
  """
  # Skip arity checks on empty results (`EMPTY_RESULT_ANOMALY` handles 0 rows)
  # or unconstrained narrative prompts (`expected_projection_arity == 0`).
  actual_arity = len(outcome.column_names)
  if (
      outcome.row_count <= 0
      or actual_arity <= 0
      or intent.expected_projection_arity <= 0
  ):
    return None

  # Flag top-level `SELECT *` over-projections and 1-col follow-up omissions.
  has_wildcard_overproj = (
      actual_arity > intent.expected_projection_arity
      and has_outer_wildcard_projection(sql_text)
  )
  is_followup_underproj = (
      actual_arity == 1
      and intent.expected_projection_arity >= 2
      and (
          intent.has_followup_projection
          or _has_fallback_followup_cue(intent.canonical_question)
      )
  )
  is_arity_drift = (
      (intent.strict_single_column and actual_arity != 1)
      or abs(actual_arity - intent.expected_projection_arity) >= 2
      or has_wildcard_overproj
      or is_followup_underproj
  )
  if not is_arity_drift:
    return None

  # Escalate `SELECT *` over-projections to ERROR when expected arity <= 2.
  arity_severity = (
      "ERROR"
      if (has_wildcard_overproj and intent.expected_projection_arity <= 2)
      else "WARNING"
  )
  arity_hint = (
      "Replace `SELECT *` / `table.*` with only the exact column(s) "
      "requested by the user question."
      if has_wildcard_overproj
      else "Project only the exact columns requested by the user question."
  )
  return InvariantIssue(
      code="PROJECTION_ARITY_DRIFT",
      severity=arity_severity,
      message=(
          f"Projected {actual_arity} columns, whereas intent expects "
          f"~{intent.expected_projection_arity}."
      ),
      repair_hint=arity_hint,
  )
