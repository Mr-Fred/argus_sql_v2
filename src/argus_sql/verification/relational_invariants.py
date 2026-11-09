"""Relational algebra and join-cardinality invariants for SQLite verification.

Detects multi-table relational semantic bugs that execute without SQLite syntax
errors yet silently corrupt denominators, entity groups, or count averages:
1. `LEFT_JOIN_FILTERED_RATIO`: Filtering a 1-to-many child table inside
   `LEFT JOIN ... ON ... AND <pred>` while computing un-deduplicated `COUNT`
   ratios (collapsing non-matching child rows and distorting denominators).
2. `NON_KEY_ENTITY_GROUPING`: Grouping joined entities by non-unique display
   names (`GROUP BY T.team_long_name`) instead of their join key (`entity_id`),
   along with a deterministic zero-token algebraic key-substitution helper.
3. `BOOLEAN_AVG_FOR_COUNT`: Using SQLite boolean `AVG(col = 'x')` (`0..1`
   proportion) when the user asks for an average count per parent entity.
"""

import re

from argus_sql.contracts.types import IntentFrame
from argus_sql.contracts.types import InvariantIssue

# Extracts `CREATE TABLE <name> ( ... );` blocks even when `--` comments have `);`.
_DDL_BLOCK_RE = re.compile(
    r'CREATE\s+TABLE\s+(?:"((?:""|[^"])+)"|([a-zA-Z0-9_]+))\s*\('
    r"((?:--[^\n]*|'(?:''|[^'])*'|[^;])*?)\)\s*;",
    re.IGNORECASE | re.DOTALL,
)

# Matches inline single-column `PRIMARY KEY` declarations in rendered DDL lines.
_INLINE_PK_COL_RE = re.compile(
    r'^(?:"((?:""|[^"])+)"|([a-zA-Z0-9_]+))\s+[A-Z0-9_() ]*\bPRIMARY\s+KEY\b',
    re.IGNORECASE,
)

# Captures `LEFT [OUTER] JOIN <table> [AS <alias>] ON <on_expr>` up to next clause.
_LEFT_JOIN_ON_RE = re.compile(
    r"\bLEFT\s+(?:OUTER\s+)?JOIN\s+"
    r'(?:"((?:""|[^"])+)"|`((?:``|[^`])+)`|([a-zA-Z_][a-zA-Z0-9_]*))'
    r"(?:\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*))?\s+ON\s+"
    r"([\s\S]+?)"
    r"(?=\b(?:WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|INNER\s+JOIN|"
    r"LEFT\s+(?:OUTER\s+)?JOIN|CROSS\s+JOIN|JOIN|UNION|INTERSECT|EXCEPT)\b|$)",
    re.IGNORECASE,
)

# Detects non-equi filter predicates (`= ''`, `= 1`, `LIKE`, `IN`, `IS NULL`)
# appended via `AND` inside a `LEFT JOIN ... ON` clause (after literal masking).
_ON_FILTER_PRED_RE = re.compile(
    r"\bAND\b[\s\S]*?(?:"
    r"(?:=|!=|<>|<=|>=|<|>)\s*(?:''|-?\d+(?:\.\d+)?\b)"
    r"|\b(?:LIKE|IN\s*\(|IS\s+(?:NOT\s+)?NULL|BETWEEN)\b)",
    re.IGNORECASE,
)

# Matches entity display-name columns (`name`, `team_long_name`, `surname`, etc.).
_ENTITY_NAME_COL_RE = re.compile(
    r"^(?:(?:[a-z0-9]+_)*(?:name|surname|forename|title)"
    r"|displayname|fullname|firstname|lastname|school)$",
    re.IGNORECASE,
)

# Matches natural-language requests for an average count per parent entity.
_AVG_COUNT_QUESTION_RE = re.compile(
    r"\b(?:average\s+(?:number|count|amount|quantity)\s+of"
    r"|on\s+average\b[^?]*\bhow\s+many)\b",
    re.IGNORECASE,
)

# Detects SQLite boolean comparison inside `AVG(...)` (`AVG(T1.element = __q__)`).
_AVG_BOOL_EXPR_RE = re.compile(
    r"\bAVG\s*\(\s*[^()]+?(?:!=|<>|=)[^()]+?\)",
    re.IGNORECASE,
)


def mask_outer_sql(sql_text: str) -> str:
  """Masks quoted tokens and parenthesized `( ... )` spans with equal spaces.

  Args:
    sql_text: Raw SQL candidate string.

  Returns:
    Equal-length string preserving only top-level unquoted SQL tokens.
  """
  # Blank out single/double/backtick/bracket quotes first to protect `(` inside.
  no_quotes = re.sub(
      r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]",
      lambda m: " " * len(m.group(0)),
      sql_text,
  )
  chars = list(no_quotes)
  depth = 0
  for idx, ch in enumerate(chars):
    if ch == "(":
      depth += 1
      chars[idx] = " "
    elif ch == ")":
      depth = max(0, depth - 1)
      chars[idx] = " "
    elif depth > 0:
      chars[idx] = " "
  return "".join(chars)


def _extract_ddl_single_pk_columns(ddl_prompt: str) -> dict[str, str | None]:
  """Maps `lower_table -> lower_single_pk_col` (`None` if composite/no PK)."""
  pk_map: dict[str, str | None] = {}
  for match in _DDL_BLOCK_RE.finditer(ddl_prompt):
    t_name = (
        match.group(1).replace('""', '"') if match.group(1) else match.group(2)
    )
    body = match.group(3)
    if re.search(r"^\s*PRIMARY\s+KEY\s*\(", body, re.IGNORECASE | re.MULTILINE):
      pk_map[t_name.lower()] = None
      continue
    found_pks: list[str] = []
    for raw_line in body.splitlines():
      code_part = raw_line.partition("--")[0].strip().rstrip(",")
      pk_m = _INLINE_PK_COL_RE.match(code_part)
      if pk_m is not None:
        c_name = (
            pk_m.group(1).replace('""', '"') if pk_m.group(1) else pk_m.group(2)
        )
        found_pks.append(c_name.lower())
    pk_map[t_name.lower()] = found_pks[0] if len(found_pks) == 1 else None
  return pk_map


def check_left_join_filtered_ratio(
    sql_text: str,
    no_strings_sql: str,
    no_quotes_sql: str,
    ddl_prompt: str = "",
) -> InvariantIssue | None:
  """Detects denominator-corrupting `LEFT JOIN ... ON ... AND <pred>` ratios.

  Args:
    sql_text: Raw candidate SQL string.
    no_strings_sql: SQL with `'...'` literals replaced by `''`.
    no_quotes_sql: SQL with all literals and quoted identifiers masked.
    ddl_prompt: Optional rendered `CREATE TABLE` DDL for PK cardinality checks.

  Returns:
    InvariantIssue if a `1:N` filtered `LEFT JOIN` ratio is detected, else None.
  """
  # Only inspect division queries aggregating with COUNT/SUM without DISTINCT.
  if "/" not in no_quotes_sql:
    return None
  if re.search(r"\b(?:COUNT|SUM)\s*\(", no_quotes_sql, re.IGNORECASE) is None:
    return None
  if re.search(r"\bCOUNT\s*\(\s*DISTINCT\b", no_quotes_sql, re.IGNORECASE):
    return None

  single_pks = _extract_ddl_single_pk_columns(ddl_prompt) if ddl_prompt else {}
  for lj_match in _LEFT_JOIN_ON_RE.finditer(no_strings_sql):
    raw_tbl = lj_match.group(1) or lj_match.group(2) or lj_match.group(3) or ""
    tbl_low = raw_tbl.replace('""', '"').replace("``", "`").lower()
    alias_low = (lj_match.group(4) or tbl_low).lower()
    on_clause = lj_match.group(5)
    if _ON_FILTER_PRED_RE.search(on_clause) is None:
      continue

    # Check whether the joined table's ON column is its unique single-col PK
    # (1-to-0..1 extension table, where LEFT JOIN does not fan out or collapse).
    eq_cols = [
        col.lower()
        for qual, col in re.findall(
            r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b",
            on_clause.split("AND")[0],
        )
        if qual.lower() in {alias_low, tbl_low}
    ]
    if tbl_low in single_pks:
      pk_col = single_pks[tbl_low]
      if pk_col is not None and eq_cols and eq_cols[0] == pk_col:
        continue
      severity = "ERROR"
    else:
      severity = "WARNING"

    # Flag 1-to-many filtered LEFT JOIN ratio for consensus/LLM repair.
    del sql_text
    return InvariantIssue(
        code="LEFT_JOIN_FILTERED_RATIO",
        severity=severity,
        message=(
            f"Filtering 1-to-many table `{raw_tbl}` inside `LEFT JOIN ... ON "
            "... AND ...` collapses non-matching child rows and distorts "
            "`COUNT(...)` ratio denominators."
        ),
        repair_hint=(
            "Use `INNER JOIN` with only the foreign-key equality in `ON`, and "
            "move the numerator filter into `SUM(CASE WHEN <cond> THEN 1 ELSE "
            "0 END)` (or deduplicate parent entities with `COUNT(DISTINCT id)`)."
        ),
    )
  return None


def _is_key_identifier(col_name: str) -> bool:
  """Returns True when `col_name` denotes a relational primary/foreign key ID."""
  low = col_name.lower()
  return low in {"id", "uuid", "cds", "code"} or low.endswith(
      ("id", "_uuid", "_cds", "_code")
  )


def find_entity_name_grouping_rewrite(
    sql_text: str,
) -> tuple[int, int, str, str] | None:
  """Locates top-level `GROUP BY <alias>.<name>` replaceable by its join key.

  Args:
    sql_text: Candidate SQL string to inspect.

  Returns:
    Tuple `(start_idx, end_idx, old_expr, key_expr)` for deterministic rewrite,
    or None if the query does not group a joined entity by a display-name col.
  """
  outer = mask_outer_sql(sql_text)
  # Only inspect ranking/filtered `GROUP BY` queries (`ORDER BY` or `HAVING`).
  if re.search(r"\b(?:ORDER\s+BY|HAVING)\b", outer, re.IGNORECASE) is None:
    return None

  # Index equi-join key expressions per table alias across top-level JOINs.
  # Map `entity_alias_low -> preferred_join_key_expr` (preferring FK side).
  alias_join_key: dict[str, str] = {}
  join_re = re.compile(
      r"\bJOIN\s+[a-zA-Z0-9_\s]+?\s+ON\s+"
      r"([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"
      r"([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)"
      r"(?!\s+AND\b)",
      re.IGNORECASE,
  )
  for jm in join_re.finditer(outer):
    a_lhs, c_lhs, a_rhs, c_rhs = (
        jm.group(1),
        jm.group(2),
        jm.group(3),
        jm.group(4),
    )
    if not (_is_key_identifier(c_lhs) and _is_key_identifier(c_rhs)):
      continue
    # Prefer the longer/more specific FK identifier (`M.away_team_api_id`) when
    # equal on every INNER JOIN row; fall back to each alias's own key column.
    is_left = (
        re.search(
            r"\bLEFT\s+(?:OUTER\s+)?$",
            outer[: jm.start()],
            re.IGNORECASE,
        )
        is not None
    )
    fk_pref = (
        f"{a_lhs}.{c_lhs}" if len(c_lhs) >= len(c_rhs) else f"{a_rhs}.{c_rhs}"
    )
    alias_join_key.setdefault(
        a_lhs.lower(), f"{a_lhs}.{c_lhs}" if is_left else fk_pref
    )
    alias_join_key.setdefault(
        a_rhs.lower(), f"{a_rhs}.{c_rhs}" if is_left else fk_pref
    )

  # Abort when no equi-join key pair was observed across top-level JOINs.
  if not alias_join_key:
    return None

  # Locate outer-scope `GROUP BY` expression span before `HAVING`/`ORDER BY`.
  gb_match = re.search(
      r"\bGROUP\s+BY\s+([\s\S]+?)"
      r"(?=\b(?:HAVING|ORDER\s+BY|LIMIT|UNION|INTERSECT|EXCEPT)\b|$)",
      outer,
      re.IGNORECASE,
  )
  if gb_match is None:
    return None

  # Extract raw `GROUP BY` items while preserving trailing newline spacing.
  gb_start, gb_end = gb_match.start(1), gb_match.end(1)
  raw_gb = sql_text[gb_start:gb_end].strip()
  parts = [p.strip() for p in raw_gb.split(",") if p.strip()]
  if len(parts) == 1:
    # Single-column `GROUP BY T.team_long_name` on an equi-joined entity table.
    col_m = re.fullmatch(
        r"([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)", parts[0]
    )
    if (
        col_m is not None
        and col_m.group(1).lower() in alias_join_key
        and _ENTITY_NAME_COL_RE.fullmatch(col_m.group(2))
    ):
      target_key = alias_join_key[col_m.group(1).lower()]
      trim_end = gb_start + len(sql_text[gb_start:gb_end].rstrip())
      return (gb_start, trim_end, raw_gb, target_key)
  elif len(parts) == 2:
    # Handle `GROUP BY T.team_api_id, T.team_long_name` compound grouping.
    m0 = re.fullmatch(
        r"([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)", parts[0]
    )
    m1 = re.fullmatch(
        r"([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)", parts[1]
    )
    if (
        m0 is not None
        and m1 is not None
        and _is_key_identifier(m0.group(2))
        and m1.group(1).lower() in alias_join_key
        and _ENTITY_NAME_COL_RE.fullmatch(m1.group(2))
    ):
      target_key = alias_join_key[m1.group(1).lower()]
      trim_end = gb_start + len(sql_text[gb_start:gb_end].rstrip())
      return (gb_start, trim_end, raw_gb, target_key)
  return None


def check_non_key_entity_grouping(sql_text: str) -> InvariantIssue | None:
  """Flags `GROUP BY <alias>.<name>` when an equi-join entity ID key exists.

  Args:
    sql_text: Candidate SQL query string to inspect.

  Returns:
    WARNING-level InvariantIssue when a joined entity is grouped by display
    name rather than its join key; None otherwise.
  """
  # Reuse single-source locator shared with deterministic algebraic repair.
  rewrite = find_entity_name_grouping_rewrite(sql_text)
  if rewrite is None:
    return None
  _, _, old_gb, target_key = rewrite
  return InvariantIssue(
      code="NON_KEY_ENTITY_GROUPING",
      severity="WARNING",
      message=(
          f"Grouping joined entities by display name (`GROUP BY {old_gb}`) "
          "conflates homonymous entities and alters tie-break key ordering."
      ),
      repair_hint=f"Group by the unique entity join key (`GROUP BY {target_key}`).",
  )


def check_boolean_avg_for_count(
    no_quotes_sql: str,
    intent: IntentFrame,
) -> InvariantIssue | None:
  """Flags SQLite boolean `AVG(col = val)` on average-count-per-entity tasks.

  Args:
    no_quotes_sql: Candidate SQL with quoted literals/identifiers masked.
    intent: Canonicalized IntentFrame describing question semantics.

  Returns:
    ERROR-level InvariantIssue when a count-average question uses `AVG(a = b)`;
    None on percentage/proportion tasks or genuine numeric averages.
  """
  # Genuine proportion/rate/percentage tasks legitimately evaluate 0..1 means.
  if intent.requires_floating_division:
    return None
  # Restrict check to explicit 'average number/count of' or 'on average how many'.
  if _AVG_COUNT_QUESTION_RE.search(intent.canonical_question) is None:
    return None
  # Flag SQLite boolean `0/1` indicator comparisons inside `AVG(...)`.
  if _AVG_BOOL_EXPR_RE.search(no_quotes_sql) is None:
    return None
  return InvariantIssue(
      code="BOOLEAN_AVG_FOR_COUNT",
      severity="ERROR",
      message=(
          "SQLite `AVG(col = value)` evaluates a 0..1 row-level boolean "
          "proportion instead of the average count per parent entity."
      ),
      repair_hint=(
          "Compute `COUNT(...)` per parent entity ID via a `GROUP BY` subquery "
          "(with `WHERE col = value`) and take `AVG(cnt)` in the outer SELECT."
      ),
  )
