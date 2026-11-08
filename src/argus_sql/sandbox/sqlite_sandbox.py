"""Read-only, authorizer-locked, resource-bounded SQLite execution sandbox.

Executes untrusted LLM-generated SQL queries safely by combining OS-level
read-only URI descriptors (`?mode=ro`), SQLite C-engine authorizer callbacks
blocking all non-SELECT actions and memory-bomb functions, literal-aware
single-statement guards, and bytecode instruction-step budgets.
"""

from collections.abc import Sequence
import hashlib
import logging
import pathlib
import re
import sqlite3
import time
from typing import Any

from argus_sql.contracts.types import ExecutionOutcome
from argus_sql.sandbox.exceptions import SandboxPathViolationError
from argus_sql.sandbox.types import SandboxPolicy

# Module-level logger following Google Python Style Guide conventions.
_LOGGER = logging.getLogger(__name__)

# SQLite functions explicitly blocked inside the C-level authorizer callback.
# Blocking `randomblob`/`zeroblob` prevents single-opcode 1 GB RAM DoS attacks.
_BLOCKED_FUNCTIONS: frozenset[str] = frozenset({
    "load_extension",
    "fts3_tokenizer",
    "writefile",
    "readfile",
    "edit",
    "randomblob",
    "zeroblob",
})

# SQLite authorizer action codes permitted for pure analytical SELECT queries.
_ALLOWED_ACTION_CODES: frozenset[int] = frozenset({
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
})

# Single-pass SQL token regex matching quoted literals/identifiers BEFORE
# comments so `--`, `/*`, or `;` inside `'...'` or `` `...` `` stay intact.
_SQL_LITERAL_OR_COMMENT_RE = re.compile(
    r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]|/\*.*?\*/|--[^\n]*",
    re.DOTALL,
)


def _strip_sql_comments(sql: str) -> str:
  """Strips `--` and `/*...*/` comments while preserving quoted tokens."""
  # Keep `'...'`, `"..."`, `` `...` ``, and `[...]` intact; strip comments.
  stripped = _SQL_LITERAL_OR_COMMENT_RE.sub(
      lambda m: m.group(0) if m.group(0)[0] in ("'", '"', "`", "[") else " ",
      sql,
  )
  return stripped.strip()


def normalize_sql_for_dedup(sql: str) -> str:
  """Normalizes whitespace and trailing semicolons for batch deduplication.

  Args:
    sql: Raw SQL candidate string.

  Returns:
    Canonical single-spaced SQL string used as an execution cache key.
  """
  # Strip comments safely and collapse whitespace so equivalent queries match.
  cleaned = _strip_sql_comments(sql).rstrip(";").strip()
  return re.sub(r"\s+", " ", cleaned)


def _validate_single_readonly_select(sql: str) -> str | None:
  """Returns None if `sql` is a single SELECT/WITH query, else error text."""
  cleaned = _strip_sql_comments(sql).strip()
  if not cleaned:
    return "Empty SQL candidate."

  # Strip trailing semicolon(s), then mask string/identifier literals.
  body = cleaned.rstrip(";").strip()
  if not body:
    return "Empty SQL candidate."

  # Replace quoted strings and identifiers before checking internal `;`.
  masked = re.sub(
      r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]",
      "''",
      body,
  )
  if ";" in masked:
    return "Multi-statement SQL execution is prohibited in read-only sandbox."

  # Allow-list strictly read-only `SELECT` and `WITH` CTE entrypoints.
  first_tok = body.split(maxsplit=1)[0].upper()
  if first_tok not in {"SELECT", "WITH"}:
    return f"Only SELECT or WITH queries are permitted (got '{first_tok}')."
  return None


def _canonicalize_cell(val: Any) -> str:
  """Formats a result cell canonically so float precision noise cannot skew EX."""
  if val is None:
    return "<NULL>"
  if isinstance(val, (int, float)) and not isinstance(val, bool):
    fval = float(val)
    # Large sums (>1e5) have ~10^9 ULP spacing in IEEE-754 float64; rounding to
    # 4 decimals above 1e5 (and 6 decimals below 1e5) prevents join-order sum
    # non-associativity from splitting equivalent consensus clusters.
    if abs(fval) >= 100_000.0:
      return f"{round(fval, 4):.4f}"
    return f"{round(fval, 6):.6f}"
  return str(val).strip()


def compute_result_signature(
    rows: Sequence[Sequence[Any]],
    preserve_row_order: bool = False,
) -> str:
  """Computes a SHA-256 digest over canonicalized SQLite result tuples.

  Args:
    rows: Sequence of result tuples returned by SQLite.
    preserve_row_order: Keep row ordering when `ORDER BY` is semantically key.

  Returns:
    Hexadecimal SHA-256 signature grouping execution-equivalent candidates.
  """
  # Format each row deterministically; deduplicate unordered row sets matching
  # official BIRD-SQL `set(predicted_res) == set(ground_truth_res)` semantics.
  formatted_rows = [
      "|".join(_canonicalize_cell(cell) for cell in row) for row in rows
  ]
  if not preserve_row_order:
    formatted_rows = sorted(set(formatted_rows))
  joined = "\n".join(formatted_rows)
  return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]


class SQLiteReadOnlySandbox:
  """Executes candidate SQL queries inside a strict read-only SQLite sandbox.

  Attributes:
    policy: Active SandboxPolicy enforcing directory, opcode, and time bounds.
  """

  def __init__(self, policy: SandboxPolicy) -> None:
    """Initializes the sandbox with resolved directory containment rules.

    Args:
      policy: SandboxPolicy specifying allowed root and resource limits.
    """
    self.policy = policy
    # Canonicalize root directory once to block symlink and `..` escapes.
    self._resolved_root = policy.allowed_db_root.resolve()

  def _resolve_safe_db_uri(self, db_path: pathlib.Path) -> str:
    """Verifies `db_path` stays inside `allowed_db_root` and builds a RO URI."""
    resolved_db = db_path.resolve()
    # Enforce strict containment inside `allowed_db_root` after symlink resolve.
    if not resolved_db.is_relative_to(self._resolved_root):
      raise SandboxPathViolationError(
          "Target SQLite database lies outside allowed sandbox root."
      )
    if not resolved_db.is_file():
      raise SandboxPathViolationError("Target SQLite database file not found.")
    # Use RFC 8089 `as_uri()` so spaces, `?`, `#`, or `%` are percent-encoded.
    return f"{resolved_db.as_uri()}?mode=ro"

  def execute(self, db_path: pathlib.Path, sql: str) -> ExecutionOutcome:
    """Executes one SQL query safely against `db_path` in read-only mode.

    Args:
      db_path: Path to target `.sqlite` file inside `policy.allowed_db_root`.
      sql: Candidate SQL query string to evaluate.

    Returns:
      ExecutionOutcome containing rows, latency, signature, or sanitized error.

    Raises:
      SandboxPathViolationError: If `db_path` escapes `allowed_db_root`.
    """
    uri = self._resolve_safe_db_uri(db_path)
    precheck_err = _validate_single_readonly_select(sql)
    if precheck_err is not None:
      return ExecutionOutcome(succeeded=False, error_message=precheck_err)

    step_counter = 0
    deadline_ns = time.perf_counter_ns() + int(self.policy.timeout_ms * 1_000_000)
    max_steps = self.policy.max_vm_steps

    # Authorizer hard-denies all writes, PRAGMAs, ATTACHes, and unsafe functions.
    def _authorizer(
        action_code: int,
        arg1: str | None,
        arg2: str | None,
        db_name: str | None,
        source: str | None,
    ) -> int:
      """Enforces read-only SELECT/READ action allow-listing in SQLite."""
      del db_name, source
      # Deny DDL, DML, PRAGMA, ATTACH, and transaction mutations immediately.
      if action_code not in _ALLOWED_ACTION_CODES:
        return sqlite3.SQLITE_DENY
      # Note: SQLite passes the SQL function name in `arg2` (`arg1` is None)
      # for `SQLITE_FUNCTION`; check both slots so blocked functions fail closed!
      if action_code == sqlite3.SQLITE_FUNCTION:
        fn_candidates = ((arg1 or "").lower(), (arg2 or "").lower())
        if any(fn in _BLOCKED_FUNCTIONS for fn in fn_candidates):
          return sqlite3.SQLITE_DENY
      return sqlite3.SQLITE_OK

    # Progress callback aborts runaway cross-joins once step or time cap hits.
    progress_stride = min(500_000, max(1_000, max_steps // 2))

    def _progress_guard() -> int:
      """Returns non-zero to interrupt SQLite when VM step or time cap expires."""
      nonlocal step_counter
      step_counter += progress_stride
      # Interrupt query execution if bytecode steps or wall-clock budget is hit.
      if step_counter > max_steps or time.perf_counter_ns() > deadline_ns:
        return 1
      return 0

    start_ns = time.perf_counter_ns()
    exec_sql = _strip_sql_comments(sql).rstrip(";").strip()
    try:
      conn = sqlite3.connect(uri, uri=True, timeout=1.5)
      try:
        conn.set_authorizer(_authorizer)
        conn.set_progress_handler(_progress_guard, progress_stride)
        cursor = conn.execute(exec_sql)
        col_names = tuple(desc[0] for desc in (cursor.description or ()))
        fetched = cursor.fetchmany(self.policy.max_returned_rows)
        rows = tuple(tuple(row) for row in fetched)
      finally:
        conn.close()
    except sqlite3.Error as err:
      elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
      # Sanitize SQLite diagnostic without exposing filesystem paths.
      sanitized_msg = re.sub(r"file:[^\s]+", "<sandbox_db>", str(err))
      _LOGGER.debug("Sandbox rejected or failed SQL: %s", sanitized_msg)
      return ExecutionOutcome(
          succeeded=False,
          execution_time_ms=round(elapsed_ms, 4),
          error_message=sanitized_msg[:240],
      )

    # Record wall-clock runtime and row-order-aware SHA-256 signature.
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    has_order_by = "ORDER BY" in exec_sql.upper()
    sig = compute_result_signature(rows, preserve_row_order=has_order_by)
    return ExecutionOutcome(
        succeeded=True,
        rows=rows,
        column_names=col_names,
        row_count=len(rows),
        execution_time_ms=round(elapsed_ms, 4),
        result_signature=sig,
    )

  def execute_batch(
      self,
      db_path: pathlib.Path,
      sql_candidates: Sequence[str],
  ) -> tuple[ExecutionOutcome, ...]:
    """Executes a batch of SQL candidates, deduplicating identical queries.

    Args:
      db_path: Path to target `.sqlite` file inside `policy.allowed_db_root`.
      sql_candidates: Sequence of SQL strings to execute.

    Returns:
      Tuple of ExecutionOutcome objects aligned 1-to-1 with `sql_candidates`.
    """
    # Cache outcomes by normalized SQL so duplicate candidates execute once.
    cache: dict[str, ExecutionOutcome] = {}
    outcomes: list[ExecutionOutcome] = []
    for sql in sql_candidates:
      key = normalize_sql_for_dedup(sql)
      if key not in cache:
        cache[key] = self.execute(db_path, sql)
      outcomes.append(cache[key])
    return tuple(outcomes)
