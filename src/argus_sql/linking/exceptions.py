"""Domain exceptions for intent canonicalization, value indexing, and linking."""

from argus_sql.contracts.exceptions import ArgusSQLError


# Base error for schema linking and value retrieval failures.
class LinkingError(ArgusSQLError):
  """Raised when schema linking or cell-value indexing encounters an error."""


# Raised when building or querying the n-gram cell value index fails.
class ValueIndexError(LinkingError):
  """Raised when indexing database cell literals fails validation or I/O."""
