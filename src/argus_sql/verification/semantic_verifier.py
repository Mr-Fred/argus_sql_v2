"""Semantic and SQLite dialect invariant verifier for NL2SQL candidates.

Detects silent semantic failures—including SQLite integer-division truncation,
unfiltered NULL sort-order extrema, all-NULL aggregations, empty-result value
drift, and projection arity mismatches—that bypass syntax-only refiners.
"""

import re

from argus_sql.contracts.types import ExecutionOutcome
from argus_sql.contracts.types import IntentFrame
from argus_sql.contracts.types import InvariantIssue
from argus_sql.contracts.types import LinkedSchemaContext
from argus_sql.contracts.types import SQLCandidate
from argus_sql.verification.dialect_invariants import check_missing_window_rank
from argus_sql.verification.dialect_invariants import check_projection_arity_drift
from argus_sql.verification.dialect_invariants import check_timestamp_bare_date_comparison
from argus_sql.verification.relational_invariants import check_boolean_avg_for_count
from argus_sql.verification.relational_invariants import check_left_join_filtered_ratio
from argus_sql.verification.relational_invariants import check_non_key_entity_grouping

# Matches floating-point cast, AVG/TOTAL real aggregates, or float-literal
# promotion idioms (`* 100.0`, `100.0 *`, `/ 12.0`) in SQLite divisions.
_FLOAT_PROMOTION_RE = re.compile(
    r"\bCAST\s*\([\s\S]+?\bAS\s+(?:REAL|FLOAT|DOUBLE)\s*\)"
    r"|\b(?:AVG|TOTAL)\s*\("
    r"|\b\d+\.\d+\s*[*/]"
    r"|[*/]\s*\d+\.\d+",
    re.IGNORECASE,
)

# Matches `ORDER BY <expr> [ASC] LIMIT <k>` lacking explicit `DESC`, including
# backtick/double-quoted column identifiers (`T1.`Enrollment (K-12)``).
_ORDER_ASC_LIMIT_RE = re.compile(
    r"\bORDER\s+BY\s+((?:[a-zA-Z0-9_]+\.)?(?:`[^`]+`|\"[^\"]+\"|[a-zA-Z0-9_.]+))(?:\s+(ASC|DESC))?\s+LIMIT\s+\d+",
    re.IGNORECASE,
)

# Detects aggregate/dedup/ordered SQL clauses that intentionally shape row count.
_DEDUP_OR_AGG_SQL_RE = re.compile(
    r"\b(?:DISTINCT|GROUP\s+BY|ORDER\s+BY|UNION|INTERSECT|EXCEPT|LIMIT|COUNT\s*\("
    r"|SUM\s*\(|AVG\s*\(|MIN\s*\(|MAX\s*\(|TOTAL\s*\()\b",
    re.IGNORECASE,
)

# Extracts `CREATE TABLE <name> ( ... );` blocks without stopping on `);` inside
# inline `--` CSV descriptions (which convert `·` bullets into `; `).
_DDL_TABLE_BLOCK_RE = re.compile(
    r'CREATE\s+TABLE\s+(?:"((?:""|[^"])+)"|([a-zA-Z0-9_]+))\s*\('
    r"((?:--[^\n]*|'(?:''|[^'])*'|[^;])*?)\)\s*;",
    re.IGNORECASE | re.DOTALL,
)

# Extracts bare or double-quoted column identifier at the start of a DDL line.
_DDL_COL_HEAD_RE = re.compile(r'^(?:"((?:""|[^"])+)"|([a-zA-Z0-9_]+))\s+')


def _extract_ddl_column_ranges(
    ddl_prompt: str,
    sql_text: str,
    max_items: int = 8,
) -> list[str]:
  """Extracts prioritized `table.col=lo..hi` domain bounds from `ddl_prompt`."""
  if "range:" not in ddl_prompt:
    return []
  sql_low = sql_text.lower()
  scored_ranges: list[tuple[tuple[int, int], str]] = []
  ordinal = 0

  # Scan prompt DDL purely in-memory so `verification/` performs zero I/O.
  for match in _DDL_TABLE_BLOCK_RE.finditer(ddl_prompt):
    t_name = (
        match.group(1).replace('""', '"') if match.group(1) else match.group(2)
    )
    for raw_line in match.group(3).splitlines():
      if "range:" not in raw_line or "--" not in raw_line:
        continue
      code_part, _, comment_part = raw_line.partition("--")
      col_m = _DDL_COL_HEAD_RE.match(code_part.strip())
      rng_m = re.search(r"\brange:\s*([^\n|]+)", comment_part)
      if col_m is None or rng_m is None:
        continue
      c_name = (
          col_m.group(1).replace('""', '"') if col_m.group(1) else col_m.group(2)
      )
      c_low = c_name.lower()
      rng_val = rng_m.group(1).strip()
      in_sql = c_low in sql_low
      is_temporal = ("'" in rng_val) or any(
          k in c_low for k in ("date", "year", "yr", "season", "month", "time")
      )
      # Rank: (0) SQL temporal, (1) sibling temporal, (2) SQL numeric, (3) rest.
      prio = 0 if (in_sql and is_temporal) else (1 if is_temporal else (2 if in_sql else 3))
      scored_ranges.append(((prio, ordinal), f"{t_name}.{c_name}={rng_val}"))
      ordinal += 1

  scored_ranges.sort(key=lambda item: item[0])
  return [entry for _, entry in scored_ranges[:max_items]]


def _build_filter_recovery_hint(
    sql_text: str,
    linked_schema: LinkedSchemaContext | None,
) -> str:
  """Constructs grounded-value, domain-range, and code-padding repair hints."""
  parts: list[str] = []
  if linked_schema is not None:
    if linked_schema.value_matches:
      seen_vals: list[str] = []
      for vm in linked_schema.value_matches:
        lit_spec = f"{vm.table_name}.{vm.column_name} = {vm.matched_value!r}"
        if lit_spec not in seen_vals:
          seen_vals.append(lit_spec)
        if len(seen_vals) >= 3:
          break
      parts.append(f"Grounded literals: {', '.join(seen_vals)}.")

    ranges = _extract_ddl_column_ranges(linked_schema.ddl_prompt, sql_text)
    if ranges:
      parts.append(
          f"DDL column bounds [{'; '.join(ranges)}]: if a date/numeric filter "
          "falls outside a sample table's range, join the FK-related table "
          "whose column range covers the target value and match its format."
      )

  parts.append(
      "Verify exact string casing and zero-padded TEXT codes "
      "(or CAST(col AS INTEGER) on numeric code comparisons)."
  )
  return " " + " ".join(parts)


def audit_candidate_invariants(
    candidate: SQLCandidate,
    outcome: ExecutionOutcome,
    intent: IntentFrame,
    linked_schema: LinkedSchemaContext | None = None,
    enable_semantic_checks: bool = True,
) -> tuple[InvariantIssue, ...]:
  """Audits a SQL candidate and its ExecutionOutcome against Argus invariants.

  Args:
    candidate: Generated SQLCandidate under inspection.
    outcome: Result returned by `SQLiteReadOnlySandbox.execute`.
    intent: Canonicalized IntentFrame describing expected query semantics.
    linked_schema: Optional LinkedSchemaContext holding grounded `ValueMatch`es.
    enable_semantic_checks: When False, checks only execution success/emptiness.

  Returns:
    Tuple of InvariantIssue objects ordered from highest to lowest severity.
  """
  issues: list[InvariantIssue] = []

  # 1. Execution failure is always a hard ERROR requiring syntax/schema repair.
  if not outcome.succeeded:
    issues.append(
        InvariantIssue(
            code="EXECUTION_ERROR",
            severity="ERROR",
            message=f"SQLite execution failed: {outcome.error_message}",
            repair_hint=(
                "Verify table and column names against CREATE TABLE DDL and "
                f"fix SQLite error: {outcome.error_message}"
            ),
        )
    )
    return tuple(issues)

  sql_text = candidate.sql
  filter_hint = _build_filter_recovery_hint(sql_text, linked_schema)

  # 2. Empty result error enriched with grounded literals and DDL range bounds.
  if outcome.row_count == 0:
    issues.append(
        InvariantIssue(
            code="EMPTY_RESULT_ANOMALY",
            severity="ERROR",
            message="Query executed without error but returned 0 rows.",
            repair_hint=(
                "Relax over-restrictive WHERE predicates or switch to the "
                f"table covering the target filter.{filter_hint}"
            ),
        )
    )

  if not enable_semantic_checks:
    return tuple(issues)

  # Mask string literals for ORDER BY inspection, and mask ALL quoted tokens
  # (`'...'`, `"..."`, `` `...` ``, `[...]`) for division checks so slashes
  # inside `'2023/01'` or `` `Charter School (Y/N)` `` never false-alarm!
  no_strings_sql = re.sub(r"'(?:''|[^'])*'", "''", sql_text)
  no_quotes_sql = re.sub(
      r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]",
      "__q__",
      sql_text,
  )

  # 3. Detect all-NULL scalar aggregation results (e.g., `[(None,)]` when
  # `SUM`/`MAX`/`AVG` filters an out-of-range table or ungrounded literal).
  if outcome.row_count == 1 and outcome.rows and all(
      cell is None for cell in outcome.rows[0]
  ):
    issues.append(
        InvariantIssue(
            code="ALL_NULL_RESULT_ANOMALY",
            severity="ERROR",
            message="Query returned a single row containing only NULL values.",
            repair_hint=(
                "Guard division by zero with NULLIF(denominator, 0) and fix "
                f"empty-matching WHERE predicates.{filter_hint}"
            ),
        )
    )

  # 4. Detect SQLite integer-division truncation on ratio/rate/percentage tasks.
  # Unlike PostgreSQL numeric division, SQLite `INT / INT` truncates toward 0;
  # however, columns with native `REAL` affinity already emit Python `float`s.
  has_division_op = "/" in no_quotes_sql
  sample_cells = [cell for row in outcome.rows[:5] for cell in row]
  has_float_cell = any(isinstance(c, float) for c in sample_cells)
  has_int_cell = any(
      isinstance(c, int) and not isinstance(c, bool) for c in sample_cells
  )
  already_native_float = has_float_cell and not has_int_cell
  if (
      (intent.requires_floating_division or has_division_op)
      and has_division_op
      and not already_native_float
  ):
    if not _FLOAT_PROMOTION_RE.search(no_quotes_sql):
      issues.append(
          InvariantIssue(
              code="INTEGER_DIVISION_TRUNCATION",
              severity="ERROR",
              message=(
                  "SQLite truncates `INTEGER / INTEGER` toward zero unless "
                  "cast to REAL."
              ),
              repair_hint=(
                  "Promote numerator to floating point using "
                  "CAST(numerator AS REAL) / NULLIF(denominator, 0)."
              ),
          )
      )

  # 5. Detect unfiltered NULL extrema in `ORDER BY col [ASC] LIMIT k`.
  # SQLite sorts NULL before all numbers in ASC order, masking true minima.
  order_match = _ORDER_ASC_LIMIT_RE.search(no_strings_sql)
  if order_match is not None:
    direction = (order_match.group(2) or "ASC").upper()
    order_col = order_match.group(1)
    bare_col = order_col.split(".")[-1].strip('"`[]')
    if direction == "ASC":
      # Require explicit `<col> IS NOT NULL` guard when sorting ascending.
      null_guard_re = re.compile(
          rf"{re.escape(bare_col)}[\"`\]]?\s+IS\s+NOT\s+NULL",
          re.IGNORECASE,
      )
      if not null_guard_re.search(no_strings_sql):
        issues.append(
            InvariantIssue(
                code="UNFILTERED_NULL_ASC_LIMIT",
                severity="WARNING",
                message=(
                    f"SQLite sorts NULL values first in `ORDER BY {order_col} "
                    "ASC LIMIT`, which can return NULL rows instead of minimum."
                ),
                repair_hint=f"Add `{order_col} IS NOT NULL` to the WHERE clause.",
            )
        )

  # 6. Check projection arity alignment, 1-col follow-up under-projection, and
  # narrow (`expected <= 2`) `SELECT *` / `table.*` over-projection escalation.
  arity_issue = check_projection_arity_drift(sql_text, outcome, intent)
  if arity_issue is not None:
    issues.append(arity_issue)

  # 7. Detect unintended 1-to-many join fanout duplicates on entity listings.
  if (
      outcome.row_count >= 5
      and re.search(r"\bJOIN\b", no_quotes_sql, re.IGNORECASE) is not None
      and _DEDUP_OR_AGG_SQL_RE.search(no_quotes_sql) is None
      and not re.search(
          r"\b(?:every|each\s+row|all\s+transactions|all\s+records)\b",
          intent.canonical_question,
          re.IGNORECASE,
      )
  ):
    unique_count = len(set(outcome.rows))
    if unique_count < outcome.row_count and (unique_count / outcome.row_count) <= 0.5:
      issues.append(
          InvariantIssue(
              code="JOIN_FANOUT_DUPLICATES",
              severity="WARNING",
              message=(
                  f"Multi-table JOIN returned {outcome.row_count} rows with "
                  f"only {unique_count} unique tuple(s) without DISTINCT."
              ),
              repair_hint="Use `SELECT DISTINCT` to deduplicate 1-to-many joins.",
          )
      )

  # 8-12. Audit relational ratios, entity keys, boolean AVG, timestamps & ranks.
  ddl_text = linked_schema.ddl_prompt if linked_schema is not None else ""
  for rel_issue in (
      check_left_join_filtered_ratio(sql_text, no_strings_sql, no_quotes_sql, ddl_text),
      check_boolean_avg_for_count(no_quotes_sql, intent),
      check_non_key_entity_grouping(sql_text),
      check_timestamp_bare_date_comparison(sql_text, ddl_text),
      check_missing_window_rank(no_quotes_sql, intent),
  ):
    if rel_issue is not None:
      issues.append(rel_issue)

  return tuple(issues)
