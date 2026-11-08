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

# Matches floating-point cast or real promotion idioms in SQLite divisions,
# including nested aggregates (`CAST(SUM(x) AS REAL)`).
_FLOAT_PROMOTION_RE = re.compile(
    r"\bCAST\s*\([\s\S]+?\bAS\s+(?:REAL|FLOAT|DOUBLE)\s*\)|1\.0\s*\*|\*\s*1\.0|100\.0\s*\*",
    re.IGNORECASE,
)

# Matches `ORDER BY <expr> [ASC] LIMIT <k>` lacking explicit `DESC`, including
# backtick/double-quoted column identifiers (`T1.`Enrollment (K-12)``).
_ORDER_ASC_LIMIT_RE = re.compile(
    r"\bORDER\s+BY\s+((?:[a-zA-Z0-9_]+\.)?(?:`[^`]+`|\"[^\"]+\"|[a-zA-Z0-9_.]+))(?:\s+(ASC|DESC))?\s+LIMIT\s+\d+",
    re.IGNORECASE,
)


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

  # 2. Empty result warning when grounded literals exist in the database.
  if outcome.row_count == 0:
    val_hint = ""
    if linked_schema and linked_schema.value_matches:
      top_v = linked_schema.value_matches[0]
      val_hint = (
          f" Use grounded literal {top_v.table_name}.{top_v.column_name} = "
          f"{top_v.matched_value!r}."
      )
    issues.append(
        InvariantIssue(
            code="EMPTY_RESULT_ANOMALY",
            severity="ERROR",
            message="Query executed without error but returned 0 rows.",
            repair_hint=(
                "Relax over-restrictive WHERE predicates or check exact string "
                f"literal casing.{val_hint}"
            ),
        )
    )

  if not enable_semantic_checks:
    return tuple(issues)

  sql_text = candidate.sql
  # Mask string literals for ORDER BY inspection, and mask ALL quoted tokens
  # (`'...'`, `"..."`, `` `...` ``, `[...]`) for division checks so slashes
  # inside `'2023/01'` or `` `Charter School (Y/N)` `` never false-alarm!
  no_strings_sql = re.sub(r"'(?:''|[^'])*'", "''", sql_text)
  no_quotes_sql = re.sub(
      r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]",
      "__q__",
      sql_text,
  )

  # 3. Detect all-NULL scalar aggregation results (e.g., `[(None,)]`).
  if outcome.row_count == 1 and outcome.rows and all(
      cell is None for cell in outcome.rows[0]
  ):
    issues.append(
        InvariantIssue(
            code="ALL_NULL_RESULT_ANOMALY",
            severity="ERROR",
            message="Query returned a single row containing only NULL values.",
            repair_hint=(
                "Check WHERE filter literals and guard division by zero with "
                "NULLIF(denominator, 0)."
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

  # 6. Check projection arity alignment when non-empty rows were returned.
  # Allow +-1 column slack so optional label+value pairs are not over-penalized.
  actual_arity = len(outcome.column_names)
  if (
      outcome.row_count > 0
      and actual_arity > 0
      and abs(actual_arity - intent.expected_projection_arity) >= 2
  ):
    issues.append(
        InvariantIssue(
            code="PROJECTION_ARITY_DRIFT",
            severity="WARNING",
            message=(
                f"Projected {actual_arity} columns, whereas intent expects "
                f"~{intent.expected_projection_arity}."
            ),
            repair_hint=(
                "Project only the exact columns requested by the user question."
            ),
        )
    )

  return tuple(issues)
