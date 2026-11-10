"""Deterministic zero-token SQLite dialect repair and LLM repair prompt builder.

Repairs common high-frequency SQLite failure patterns—integer division
truncation, unfiltered NULLs in ascending `ORDER BY ... LIMIT`, and ungrounded
string literal abbreviations—in sub-millisecond time without extra API calls.
"""

from collections.abc import Sequence
import re

from argus_sql.contracts.types import InvariantIssue
from argus_sql.contracts.types import LinkedSchemaContext
from argus_sql.contracts.types import SQLCandidate
from argus_sql.verification.dialect_invariants import rewrite_timestamp_bare_date_comparisons
from argus_sql.verification.relational_invariants import find_entity_name_grouping_rewrite

# Atomic SQL token inside function arguments or parenthesized arithmetic groups.
_INNER_ATOM = (
    r"(?:[^()'\"`\[]|'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\""
    r"|`(?:``|[^`])*`|\[[^\]]*\])"
)
_NESTED_PARENS = rf"\((?:{_INNER_ATOM}|\((?:{_INNER_ATOM}|\({_INNER_ATOM}*\))*\))*\)"

# SQL clause keywords that may precede `(` without forming a function call.
_NON_FN_KW = (
    r"(?:SELECT|DISTINCT|FROM|WHERE|JOIN|ON|AND|OR|NOT|IN|AS|BY"
    r"|WHEN|THEN|ELSE|CASE|OVER|WITH|UNION|INTERSECT|EXCEPT|HAVING|LIMIT|EXISTS)"
)

# Atomic SQL operand: any SQL function/CAST call `fn(...)`, standalone balanced
# parenthesized expression `(a - b)` not preceded by an identifier, or a
# (possibly table-qualified) quoted/bare column identifier not followed by `(`.
_SQL_OPERAND_PAT = (
    rf"(?:(?!\b{_NON_FN_KW}\b)\b[a-zA-Z_][a-zA-Z0-9_]*\s*{_NESTED_PARENS}"
    rf"|(?<![a-zA-Z0-9_]){_NESTED_PARENS}"
    r"|(?:[a-zA-Z_][a-zA-Z0-9_]*\.)?"
    r"(?:`(?:``|[^`])+`|\"(?:\"\"|[^\"])+\"|\[[^\]]+\]"
    rf"|\b(?!{_NON_FN_KW}\b)[a-zA-Z_][a-zA-Z0-9_]*(?!\s*\()))"
)

# Ordered token matcher:
# 1. Single-quoted string literals (`'Free/Reduced'`) -> keep untouched.
# 2. Already-REAL-cast numerators (`CAST(... AS REAL|FLOAT|DOUBLE) /`) -> keep.
# 3. Uncast `lhs [* 100] / rhs` divisions -> promote to `CAST(...) / NULLIF(...)`.
# 4. Standalone quoted identifiers (`T1.`Charter School (Y/N)``) -> advance past
#    the closing quote so internal `(Y/N)` slashes are never split!
_QUOTED_OR_DIV_RE = re.compile(
    r"'(?:''|[^'])*'|"
    rf"(\bCAST\s*\((?:{_INNER_ATOM}|{_NESTED_PARENS})+\s+AS\s+"
    r"(?:REAL|FLOAT|DOUBLE)\s*\)(?:\s*\*\s*100(?:\.0)?)?\s*/)|"
    rf"({_SQL_OPERAND_PAT}(?:\s*\*\s*100(?:\.0)?)?)\s*/\s*({_SQL_OPERAND_PAT})|"
    r"`(?:``|[^`])*`|\"(?:\"\"|[^\"])*\"|\[[^\]]*\]",
    re.IGNORECASE,
)

# Captures `ORDER BY <col> [ASC] LIMIT <k>` (bare or quoted) for NOT NULL guards.
_ORDER_BY_ASC_RE = re.compile(
    r"\b(ORDER\s+BY\s+((?:[a-zA-Z_][a-zA-Z0-9_]*\.)?"
    r"(?:`[^`]+`|\"[^\"]+\"|[a-zA-Z_][a-zA-Z0-9_.]*))(?:\s+ASC)?\s+LIMIT\s+\d+)",
    re.IGNORECASE,
)


def _rewrite_uncast_divisions(sql_text: str) -> str:
  """Promotes uncast `lhs / rhs` arithmetic while preserving quoted tokens."""

  def _replacer(match: re.Match[str]) -> str:
    """Skips literals, standalone quoted identifiers, and existing CASTs."""
    token = match.group(0)
    # Only Branch 3 populates groups (2, 3); all other branches stay unchanged.
    if match.group(2) is None or match.group(3) is None:
      return token
    lhs, rhs = match.group(2).strip(), match.group(3).strip()
    # Avoid redundant `NULLIF(NULLIF(x, 0), 0)` when denominator is guarded.
    safe_rhs = rhs if re.match(r"^NULLIF\s*\(", rhs, re.IGNORECASE) else f"NULLIF({rhs}, 0)"
    return f"CAST({lhs} AS REAL) / {safe_rhs}"

  return _QUOTED_OR_DIV_RE.sub(_replacer, sql_text)


def _mask_subqueries_and_literals(sql_prefix: str) -> str:
  """Replaces quoted tokens and parenthesized `( ... )` spans with spaces."""
  # Mask literals & quoted identifiers first so `(K-12)` inside backticks
  # cannot skew outer subquery parenthesis depth!
  no_quotes = re.sub(
      r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]",
      lambda m: " " * len(m.group(0)),
      sql_prefix,
  )
  # Mask nested parenthesized subqueries so only outer-scope clauses remain.
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


def _inject_top_level_not_null(prefix: str, order_col: str) -> str | None:
  """Injects `<order_col> IS NOT NULL` safely at top-level query scope."""
  # Abort if `ORDER BY` lies inside an unclosed subquery or CTE parenthesis.
  no_quotes = re.sub(
      r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]",
      " ",
      prefix,
  )
  if no_quotes.count("(") != no_quotes.count(")"):
    return None

  outer_mask = _mask_subqueries_and_literals(prefix)
  # Skip rewrite if outer query aggregates via GROUP BY or compound UNION/EXCEPT.
  if (
      re.search(
          r"\b(?:GROUP\s+BY|UNION|INTERSECT|EXCEPT)\b",
          outer_mask,
          re.IGNORECASE,
      )
      is not None
  ):
    return None

  where_match = re.search(r"\bWHERE\b", outer_mask, re.IGNORECASE)
  if where_match is None:
    return f"{prefix} WHERE {order_col} IS NOT NULL"

  # Wrap existing outer WHERE predicate in parens if top-level `OR` is present.
  cond_start = where_match.end()
  outer_cond_mask = outer_mask[cond_start:]
  if re.search(r"\bOR\b", outer_cond_mask, re.IGNORECASE) is not None:
    before_where = prefix[: where_match.start()].rstrip()
    existing_cond = prefix[cond_start:].strip()
    return f"{before_where} WHERE ({existing_cond}) AND {order_col} IS NOT NULL"

  return f"{prefix} AND {order_col} IS NOT NULL"


def apply_algebraic_sql_repairs(
    candidate: SQLCandidate,
    issues: Sequence[InvariantIssue],
    linked_schema: LinkedSchemaContext | None = None,
) -> SQLCandidate | None:
  """Applies deterministic algebraic rewrites to resolve flagged dialect bugs.

  Args:
    candidate: Original SQLCandidate exhibiting one or more InvariantIssues.
    issues: Sequence of InvariantIssues detected by `audit_candidate_invariants`.
    linked_schema: Optional LinkedSchemaContext supplying grounded cell literals.

  Returns:
    New repaired SQLCandidate if a deterministic rewrite changed the SQL; None
    if no safe deterministic rewrite applies.
  """
  issue_codes = {item.code for item in issues}
  repaired = candidate.sql.strip().rstrip(";")

  # 0. Rewrite non-SQLite `ILIKE` operator outside single-quoted literals.
  if "EXECUTION_ERROR" in issue_codes and re.search(
      r"\bILIKE\b", repaired, re.IGNORECASE
  ):
    repaired = re.sub(
        r"('(?:''|[^'])*')|\bILIKE\b",
        lambda m: m.group(1) if m.group(1) is not None else "LIKE",
        repaired,
        flags=re.IGNORECASE,
    )

  # 1. Promote uncast `lhs / rhs` (SELECT and ORDER BY) while skipping literals.
  if "INTEGER_DIVISION_TRUNCATION" in issue_codes:
    repaired = _rewrite_uncast_divisions(repaired)

  # 2. Inject `<col> IS NOT NULL` at outer scope before `ORDER BY <col> ASC LIMIT`.
  if "UNFILTERED_NULL_ASC_LIMIT" in issue_codes:
    match = _ORDER_BY_ASC_RE.search(repaired)
    if match is not None:
      order_col = match.group(2)
      order_start = match.start(1)
      prefix = repaired[:order_start].rstrip()
      suffix = repaired[order_start:]
      updated_prefix = _inject_top_level_not_null(prefix, order_col)
      if updated_prefix is not None:
        repaired = f"{updated_prefix} {suffix}"

  # 3. Substitute abbreviated or mis-cased string literals on empty/all-NULL rows.
  # Tokenize single-quoted literals in one pass and preserve exact DB matches so
  # lower-scoring prefix hits ('Alameda Unified') never clobber exact ('Alameda')!
  if (
      issue_codes & {"EMPTY_RESULT_ANOMALY", "ALL_NULL_RESULT_ANOMALY"}
      and linked_schema is not None
      and linked_schema.value_matches
  ):
    exact_cased_vals = {
        vm.matched_value
        for vm in linked_schema.value_matches
        if vm.similarity_score >= 0.99
    }
    lit_replacements: dict[str, str] = {}
    for vm in linked_schema.value_matches:
      if vm.similarity_score < 0.75:
        continue
      for raw_probe in (vm.matched_value, vm.query_span):
        if raw_probe:
          lit_replacements.setdefault(raw_probe.lower(), vm.matched_value)

    if lit_replacements:

      def _rewrite_sql_string_literal(match: re.Match[str]) -> str:
        """Replaces one single-quoted SQL literal if abbreviated or mis-cased."""
        tok = match.group(0)
        inner = tok[1:-1].replace("''", "'")
        if inner in exact_cased_vals:
          return tok
        target = lit_replacements.get(inner.lower())
        if target is not None and target != inner:
          return "'" + target.replace("'", "''") + "'"
        return tok

      repaired = re.sub(r"'(?:''|[^'])*'", _rewrite_sql_string_literal, repaired)

  # 4. Deduplicate 1-to-many entity-listing join fanout via `SELECT DISTINCT`.
  if "JOIN_FANOUT_DUPLICATES" in issue_codes and re.match(
      r"^SELECT\s+(?!DISTINCT\b)", repaired, re.IGNORECASE
  ):
    repaired = re.sub(
        r"^SELECT\s+", "SELECT DISTINCT ", repaired, count=1, flags=re.IGNORECASE
    )

  # 5. Rewrite `GROUP BY <alias>.<name>` to its unique equi-join entity ID key.
  if "NON_KEY_ENTITY_GROUPING" in issue_codes:
    gb_rw = find_entity_name_grouping_rewrite(repaired)
    if gb_rw is not None:
      gb_start, gb_end, _, target_key = gb_rw
      repaired = f"{repaired[:gb_start]}{target_key}{repaired[gb_end:]}"

  # 6. Wrap sub-day timestamp columns in `DATE(col)` on `'YYYY-MM-DD'` filters.
  if (
      "TIMESTAMP_BARE_DATE_COMPARISON" in issue_codes
      and linked_schema is not None
  ):
    repaired = rewrite_timestamp_bare_date_comparisons(
        repaired, linked_schema.ddl_prompt
    )

  # Return None when rewrites produced no diff to avoid duplicate sandbox runs.
  if repaired == candidate.sql.strip().rstrip(";"):
    return None

  # Zero-token algebraic repairs incur 0 prompt/completion LLM token cost.
  return SQLCandidate(
      sql=repaired,
      strategy=f"{candidate.strategy}+argus_repair",
      reasoning="Applied deterministic Argus algebraic SQLite dialect repair.",
      prompt_tokens=0,
      completion_tokens=0,
  )


def build_diagnostic_repair_prompt(
    question: str,
    evidence: str,
    ddl_prompt: str,
    candidate: SQLCandidate,
    issues: Sequence[InvariantIssue],
    failed_sqls: Sequence[str] = (),
) -> str:
  """Formats an execution + semantic invariant repair prompt for an LLM refiner.

  Args:
    question: Canonicalized natural language user question.
    evidence: Domain formula / external knowledge string.
    ddl_prompt: Linked `CREATE TABLE` DDL schema context.
    candidate: Failing or invariant-violating SQLCandidate.
    issues: Diagnostic InvariantIssue items guiding the repair.
    failed_sqls: Optional sibling failed SQL strings so the refiner avoids
      repeating predicates that already returned 0 rows or failed.

  Returns:
    Structured prompt requesting a single corrected SQLite SELECT query.
  """
  # Render each invariant violation with its concrete remediation instruction.
  bullet_lines = "\n".join(
      f"- [{iss.code}] {iss.message} Fix: {iss.repair_hint}" for iss in issues
  )
  ev_line = f"\nExternal Evidence: {evidence}" if evidence else ""

  # Surface distinct sibling failed SQLs (capped to 2) to prevent empty-set loops.
  seed_norm = candidate.sql.strip()
  other_failed: list[str] = []
  for alt_sql in failed_sqls:
    cleaned_alt = alt_sql.strip()
    if cleaned_alt and cleaned_alt != seed_norm and cleaned_alt not in other_failed:
      other_failed.append(cleaned_alt)
    if len(other_failed) >= 2:
      break
  alt_block = ""
  if other_failed:
    joined_alts = "\n---\n".join(other_failed)
    alt_block = (
        "\n\nOther Failed SQL Attempts (do not repeat these predicates):\n"
        f"{joined_alts}"
    )

  return (
      "You are repairing a SQLite query that failed semantic or execution checks.\n\n"
      f"Database Schema:\n{ddl_prompt}\n\n"
      f"Question: {question}{ev_line}\n\n"
      f"Previous SQL:\n{candidate.sql}{alt_block}\n\n"
      f"Detected Issues:\n{bullet_lines}\n\n"
      "Return ONLY the corrected read-only SQLite SELECT query inside a "
      "```sql ... ``` code block."
  )
