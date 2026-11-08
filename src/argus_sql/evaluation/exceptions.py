"""Domain exceptions for BIRD-SQL benchmark loading and metric evaluation."""

from argus_sql.contracts.exceptions import ArgusSQLError


# Base error for benchmark evaluation or report serialization failures.
class EvaluationError(ArgusSQLError):
  """Raised when metric evaluation or benchmark orchestration fails."""


# Raised when a public BIRD-SQL dataset directory is malformed or unsafe.
class BenchmarkDatasetError(EvaluationError):
  """Raised when loading benchmark JSON tasks or table schemas fails."""
