"""Deterministic zero-token SQLite dialect repair and LLM repair prompt builder.

Repairs common high-frequency BIRD-SQL failure patterns—integer division
truncation, unfiltered NULLs in ascending `ORDER BY ... LIMIT`, and ungrounded
string literal abbreviations—in sub-millisecond time without extra API calls.
"""

from collections.abc import Sequence
import re

from argus_sql.contracts.types import InvariantIssue
from argus_sql.contracts.types import LinkedSchemaContext
from argus_sql.contracts.types import SQLCandidate

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

  # 3. Substitute ungrounded abbreviated string literals using high-score ValueMatch.
  if "EMPTY_RESULT_ANOMALY" in issue_codes and linked_schema is not None:
    for vm in linked_schema.value_matches:
      if vm.similarity_score >= 0.75 and vm.query_span != vm.matched_value:
        quoted_span = f"'{vm.query_span}'"
        if quoted_span.lower() in repaired.lower():
          escaped_val = vm.matched_value.replace("'", "''")
          repaired = re.sub(
              re.escape(quoted_span),
              f"'{escaped_val}'",
              repaired,
              flags=re.IGNORECASE,
          )

  # 4. Deduplicate 1-to-many entity-listing join fanout via `SELECT DISTINCT`.
  if "JOIN_FANOUT_DUPLICATES" in issue_codes and re.match(
      r"^SELECT\s+(?!DISTINCT\b)", repaired, re.IGNORECASE
  ):
    repaired = re.sub(
        r"^SELECT\s+", "SELECT DISTINCT ", repaired, count=1, flags=re.IGNORECASE
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
) -> str:
  """Formats an execution + semantic invariant repair prompt for an LLM refiner.

  Args:
    question: Canonicalized natural language user question.
    evidence: Domain formula / external knowledge string.
    ddl_prompt: Linked `CREATE TABLE` DDL schema context.
    candidate: Failing or invariant-violating SQLCandidate.
    issues: Diagnostic InvariantIssue items guiding the repair.

  Returns:
    Structured prompt requesting a single corrected SQLite SELECT query.
  """
  # Render each invariant violation with its concrete remediation instruction.
  bullet_lines = "\n".join(
      f"- [{iss.code}] {iss.message} Fix: {iss.repair_hint}" for iss in issues
  )
  ev_line = f"\nExternal Evidence: {evidence}" if evidence else ""
  return (
      "You are repairing a SQLite query that failed semantic or execution checks.\n\n"
      f"Database Schema:\n{ddl_prompt}\n\n"
      f"Question: {question}{ev_line}\n\n"
      f"Previous SQL:\n{candidate.sql}\n\n"
      f"Detected Issues:\n{bullet_lines}\n\n"
      "Return ONLY the corrected SQLite SELECT query with no markdown prose."
  )
