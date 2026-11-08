"""Domain exceptions for semantic invariant verification and arbitration."""

from argus_sql.contracts.exceptions import ArgusSQLError


# Base error for verification and consensus selection failures.
class VerificationError(ArgusSQLError):
  """Raised when candidate verification or consensus arbitration fails."""
