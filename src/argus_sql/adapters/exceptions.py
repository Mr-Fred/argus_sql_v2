"""Domain exceptions for external LLM batch transport and synthesis adapters."""

from argus_sql.contracts.exceptions import ArgusSQLError


# Base exception for LLM client configuration or synthesis errors.
class AdapterError(ArgusSQLError):
  """Raised when an LLM or synthesis adapter encounters an operational error."""


# Raised when required API credentials are absent or endpoint URL is insecure.
class LLMAdapterAuthError(AdapterError):
  """Raised when API authentication keys are missing or transport is unsafe."""


# Raised when an upstream HTTPS completion call fails or returns invalid JSON.
class LLMAdapterTransportError(AdapterError):
  """Raised when batch LLM transport fails or returns malformed payloads."""
