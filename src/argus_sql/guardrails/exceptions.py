"""Custom domain exceptions for clean-room IP and code-quality guardrails.

All guardrail failures inherit from GuardrailError so CLI gates and unit tests
can catch specific policy breaches cleanly without swallowing system signals.
"""


# Base class lets CLI entrypoints handle all guardrail errors uniformly.
class GuardrailError(Exception):
  """Base exception raised when a repository guardrail operation fails."""


# Raised when text or diff scanning uncovers internal markers or API secrets.
class CleanRoomViolationError(GuardrailError):
  """Raised when prohibited internal identifiers, paths, or secrets are found."""


# Raised when Git author config points to a corporate email address.
class GitIdentityViolationError(GuardrailError):
  """Raised when Git author metadata uses a prohibited corporate email domain."""
