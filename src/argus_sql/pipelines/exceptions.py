"""Domain exceptions for comparative NL2SQL pipeline execution."""

from argus_sql.contracts.exceptions import ArgusSQLError


# Raised when a pipeline stage fails unrecoverably during task prediction.
class PipelineExecutionError(ArgusSQLError):
  """Raised when an NL2SQL pipeline cannot complete prediction for a task."""
