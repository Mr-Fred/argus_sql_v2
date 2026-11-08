"""Domain exceptions for the read-only SQLite execution sandbox."""

from argus_sql.contracts.exceptions import ArgusSQLError


# Base error for sandbox configuration or connection failures.
class SandboxError(ArgusSQLError):
  """Raised when the read-only SQLite execution sandbox fails."""


# Raised when a caller attempts to open a database outside the allowed root.
class SandboxPathViolationError(SandboxError):
  """Raised when a target database path escapes the configured root boundary."""


# Raised when untrusted SQL attempts multi-statement or non-SELECT operations.
class UnsafeSQLStatementError(SandboxError):
  """Raised when candidate SQL violates single-statement read-only rules."""
