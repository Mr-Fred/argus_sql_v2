"""Domain exception hierarchy for Argus-SQL core contracts and rendering."""


# Root domain exception allowing unified error boundaries across pipelines.
class ArgusSQLError(Exception):
  """Base exception for all Argus-SQL domain and adapter errors."""


# Raised when relational table or foreign-key references are inconsistent.
class SchemaGraphError(ArgusSQLError):
  """Raised when a DatabaseSchema contains invalid table or column references."""


# Raised when DDL or prompt serialization receives malformed inputs.
class PromptBuildError(ArgusSQLError):
  """Raised when rendering a schema or task prompt fails validation."""
