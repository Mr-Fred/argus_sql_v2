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
  # Collect CTE aliases (`WITH c1 AS (...), c2 AS (...)`) so deduplicated
  # window-function CTEs (`LEFT JOIN latest ON ... AND rn = 1`) never false-alarm.
  cte_names = {
      c.lower()
      for c in re.findall(
          r"(?:\bWITH(?:\s+RECURSIVE)?|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(",
          no_strings_sql,
          re.IGNORECASE,
      )
  }
  for lj_match in _LEFT_JOIN_ON_RE.finditer(no_strings_sql):
    raw_tbl = lj_match.group(1) or lj_match.group(2) or lj_match.group(3) or ""
    tbl_low = raw_tbl.replace('""', '"').replace("``", "`").lower()
    if tbl_low in cte_names:
      continue
    alias_low = (lj_match.group(4) or tbl_low).lower()
    on_clause = lj_match.group(5)
    if _ON_FILTER_PRED_RE.search(on_clause) is None:
      continue

    # Extract identifier-to-identifier equi-join columns on the joined table
    # regardless of `AND` casing or predicate order (`1:0..1` PK extension check).
    eq_cols: set[str] = set()
    for q1, c1, q2, c2 in re.findall(
        r"\b([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\s*=\s*([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\b",
        on_clause,
    ):
      if q1.lower() in {alias_low, tbl_low}:
        eq_cols.add(c1.lower())
      if q2.lower() in {alias_low, tbl_low}:
        eq_cols.add(c2.lower())
    if tbl_low in single_pks:
      pk_col = single_pks[tbl_low]
      # 1-to-0..1 PK extension tables never fan out or distort parent counts.
      if pk_col is not None and pk_col in eq_cols:
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

  # Track explicit parent-FK keys separately from symmetric same-name keys so a
  # child FK (`Team.country_id = Country.id`) never poisons `Team`'s own key.
  explicit_fk_key: dict[str, str] = {}
  same_name_key: dict[str, str] = {}
  join_re = re.compile(
      r"\bJOIN\s+[a-zA-Z0-9_\s]+?\s+ON\s+"
      r"([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"
      r"([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)"
      r"(?!\s+AND\b)",
      re.IGNORECASE,
  )
  for jm in join_re.finditer(outer):
    a_lhs, c_lhs, a_rhs, c_rhs = jm.groups()
    if not (_is_key_identifier(c_lhs) and _is_key_identifier(c_rhs)):
      continue
    # Preserve outer-table key expression on LEFT JOIN so NULLs cannot merge.
    is_left = bool(
        re.search(r"\bLEFT\s+(?:OUTER\s+)?$", outer[: jm.start()], re.IGNORECASE)
    )
    low_l, low_r = c_lhs.lower(), c_rhs.lower()
    if low_l == low_r:
      # Identical key names (`driverId = driverId`) identify both joined sides.
      shared_expr = f"{a_lhs}.{c_lhs}"
      same_name_key.setdefault(a_lhs.lower(), f"{a_lhs}.{c_lhs}" if is_left else shared_expr)
      same_name_key.setdefault(a_rhs.lower(), f"{a_rhs}.{c_rhs}" if is_left else shared_expr)
    elif len(low_l) != len(low_r):
      # Shorter key side is the referenced parent entity; longer side holds FK.
      a_sh, c_sh, a_lg, c_lg = (
          (a_lhs, c_lhs, a_rhs, c_rhs)
          if len(low_l) < len(low_r)
          else (a_rhs, c_rhs, a_lhs, c_lhs)
      )
      low_sh, low_lg = c_sh.lower(), c_lg.lower()
      # Require prefix/suffix stem alignment (`away_team_api_id` -> `team_api_id`).
      if not (low_lg.endswith(low_sh) or low_lg.startswith(low_sh)):
        continue
      fk_expr = f"{a_sh}.{c_sh}" if is_left else f"{a_lg}.{c_lg}"
      explicit_fk_key.setdefault(a_sh.lower(), fk_expr)
      # Support 1:1 code variants (`cds` <-> `CDSCode`) without child FK leakage.
      if low_lg == f"{low_sh}code":
        same_name_key.setdefault(a_lg.lower(), f"{a_lg}.{c_lg}" if is_left else fk_expr)

  # Explicit parent-FK bindings override symmetric fallback bindings.
  alias_join_key = {**same_name_key, **explicit_fk_key}
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

  # Extract qualified `(alias, name_col)` from single or 2-item `GROUP BY` list.
  gb_start, gb_end = gb_match.start(1), gb_match.end(1)
  raw_gb = sql_text[gb_start:gb_end].strip()
  parts = [p.strip() for p in raw_gb.split(",") if p.strip()]
  qual_re = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)")
  ent_alias: str | None = None
  if len(parts) == 1:
    m_one = qual_re.fullmatch(parts[0])
    if (
        m_one is not None
        and m_one.group(1).lower() in alias_join_key
        and _ENTITY_NAME_COL_RE.fullmatch(m_one.group(2))
    ):
      ent_alias = m_one.group(1)
  elif len(parts) == 2:
    # Support both `(id, name)` and `(name, id)` compound `GROUP BY` orderings.
    m0, m1 = qual_re.fullmatch(parts[0]), qual_re.fullmatch(parts[1])
    if m0 is not None and m1 is not None:
      for m_key, m_name in ((m0, m1), (m1, m0)):
        if (
            _is_key_identifier(m_key.group(2))
            and m_name.group(1).lower() in alias_join_key
            and _ENTITY_NAME_COL_RE.fullmatch(m_name.group(2))
        ):
          ent_alias = m_name.group(1)
          break

  if ent_alias is None:
    return None
  target_key = alias_join_key[ent_alias.lower()]
  # Never rewrite queries intentionally counting distinct entities per homonym
  # (`SELECT d.forename, COUNT(DISTINCT d.driverId) ... GROUP BY d.forename`).
  key_col = target_key.split(".")[-1]
  distinct_pat = rf"\bCOUNT\s*\(\s*DISTINCT\s+(?:{re.escape(ent_alias)}\.\w+"
  if key_col.lower() != "id":
    distinct_pat += rf"|(?:[a-zA-Z_]\w*\.)?{re.escape(key_col)}\b"
  distinct_pat += r")"
  if re.search(distinct_pat, sql_text, re.IGNORECASE):
    return None

  # Preserve any trailing whitespace/newlines before `ORDER BY` or `HAVING`.
  trim_end = gb_start + len(sql_text[gb_start:gb_end].rstrip())
  return (gb_start, trim_end, raw_gb, target_key)


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
