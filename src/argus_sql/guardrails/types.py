"""Typed data contracts for clean-room IP guardrails and AST quality checks.

Defines immutable report structures used by the pre-commit scanner, commit-msg
validator, and standalone Python AST quality verifier.
"""

import dataclasses
import enum


# Severity levels determine whether findings represent direct IP/secret risks
# (CRITICAL) or workstation path/configuration hygiene issues (HIGH/MEDIUM).
class GuardrailSeverity(str, enum.Enum):
  """Severity classification for guardrail violations."""

  CRITICAL = "CRITICAL"
  HIGH = "HIGH"
  MEDIUM = "MEDIUM"


# Frozen dataclasses prevent downstream callers from mutating security rules.
@dataclasses.dataclass(frozen=True)
class LeakRule:
  """Compiled pattern rule detecting prohibited internal data or credentials.

  Attributes:
    rule_id: Stable identifier for reporting (e.g., 'CR-CORP-EMAIL').
    category: Human-readable category name ('corporate_leak', 'secret', 'path').
    severity: Violation severity tier.
    description: Actionable remediation message explaining why it was blocked.
    pattern_text: Compiled regular expression pattern string.
    case_sensitive: Whether regex matching should preserve letter casing.
  """

  rule_id: str
  category: str
  severity: GuardrailSeverity
  description: str
  pattern_text: str
  case_sensitive: bool = False


# Findings store only redacted previews so logs never echo raw secrets or PII.
@dataclasses.dataclass(frozen=True)
class LeakFinding:
  """Single prohibited pattern occurrence discovered in a file or commit metadata.

  Attributes:
    rule_id: Identifier of the triggered LeakRule.
    category: Category of the leak ('corporate_leak', 'secret', 'path', 'git').
    severity: Severity tier of the finding.
    target: Relative file path or metadata field where the match occurred.
    line_number: One-based line number of the violation.
    redacted_preview: Masked snippet showing context without echoing raw secrets.
    description: Explanation and fix guidance.
  """

  rule_id: str
  category: str
  severity: GuardrailSeverity
  target: str
  line_number: int
  redacted_preview: str
  description: str


# AST quality violations pinpoint exact source line numbers for fast repair.
@dataclasses.dataclass(frozen=True)
class CodeQualityViolation:
  """Single static AST style, documentation, typing, or batch-safety violation.

  Attributes:
    target: Relative or absolute file path audited.
    line_number: One-based source line number.
    rule_id: Short code quality rule identifier (e.g., 'AST-DOCSTRING').
    message: Detailed human-readable violation and fix guidance.
  """

  target: str
  line_number: int
  rule_id: str
  message: str


# Aggregate audit report enforces fail-closed gating across all checks.
# Any single leak, AST violation, missing Git identity, or inactive hook fails.
@dataclasses.dataclass(frozen=True)
class GuardrailAuditReport:
  """Aggregate verification summary across clean-room, git, and AST checks.

  Attributes:
    scanned_files: Total count of repository files inspected.
    leak_findings: Tuple of clean-room or secret findings detected.
    quality_violations: Tuple of AST code quality violations detected.
    git_identity_ok: True if local git user.email is non-corporate and valid.
    git_hooks_ok: True if pre-commit and commit-msg hooks are installed/active.
  """

  # Total text and PDF artifacts audited during repository traversal.
  scanned_files: int
  leak_findings: tuple[LeakFinding, ...]
  quality_violations: tuple[CodeQualityViolation, ...]
  git_identity_ok: bool
  git_hooks_ok: bool = True

  @property
  def passed(self) -> bool:
    """Returns True only when zero clean-room, git, hook, or AST issues exist."""
    # Fail closed if any leak finding, quality violation, or git gap occurs.
    return (
        self.git_identity_ok
        and self.git_hooks_ok
        and not self.leak_findings
        and not self.quality_violations
    )
