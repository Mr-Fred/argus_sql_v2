"""Public API facade for the Argus-SQL read-only SQLite execution sandbox."""

# Re-export sandbox security exceptions.
from argus_sql.sandbox.exceptions import SandboxError
from argus_sql.sandbox.exceptions import SandboxPathViolationError
from argus_sql.sandbox.exceptions import UnsafeSQLStatementError
# Re-export sandbox engine and deterministic signature helpers.
from argus_sql.sandbox.sqlite_sandbox import SQLiteReadOnlySandbox
from argus_sql.sandbox.sqlite_sandbox import compute_result_signature
from argus_sql.sandbox.sqlite_sandbox import normalize_sql_for_dedup
# Re-export sandbox policy contract.
from argus_sql.sandbox.types import SandboxPolicy

# Explicit symbol export list.
__all__ = [
    "SQLiteReadOnlySandbox",
    "SandboxError",
    "SandboxPathViolationError",
    "SandboxPolicy",
    "UnsafeSQLStatementError",
    "compute_result_signature",
    "normalize_sql_for_dedup",
]
