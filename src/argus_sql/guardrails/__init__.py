"""Public API facade for the Argus-SQL clean-room guardrails package."""

# Re-export core scanner builders and text/PDF inspection entrypoints.
from argus_sql.guardrails.clean_room_scanner import build_default_leak_rules
from argus_sql.guardrails.clean_room_scanner import scan_pdf_for_leaks
from argus_sql.guardrails.clean_room_scanner import scan_text_for_leaks
# Re-export Python AST style, typing, and batch-safety verifier.
from argus_sql.guardrails.code_quality_verifier import verify_python_ast_quality
# Re-export domain-specific guardrail exception classes.
from argus_sql.guardrails.exceptions import CleanRoomViolationError
from argus_sql.guardrails.exceptions import GitIdentityViolationError
from argus_sql.guardrails.exceptions import GuardrailError
# Re-export Git hook installer, CI bootstrapper, and audit coordinators.
from argus_sql.guardrails.hook_installer import bootstrap_clean_git_config
from argus_sql.guardrails.hook_installer import install_git_hooks
from argus_sql.guardrails.hook_installer import verify_git_hooks_active
from argus_sql.guardrails.repository_auditor import audit_target_path
from argus_sql.guardrails.repository_auditor import run_repository_audit
# Re-export immutable report and violation data contracts.
from argus_sql.guardrails.types import CodeQualityViolation
from argus_sql.guardrails.types import GuardrailAuditReport
from argus_sql.guardrails.types import GuardrailSeverity
from argus_sql.guardrails.types import LeakFinding
from argus_sql.guardrails.types import LeakRule

# Explicitly enumerate exported symbols for static type checkers.
__all__ = [
    "CleanRoomViolationError",
    "CodeQualityViolation",
    "GitIdentityViolationError",
    "GuardrailAuditReport",
    "GuardrailError",
    "GuardrailSeverity",
    "LeakFinding",
    "LeakRule",
    "audit_target_path",
    "bootstrap_clean_git_config",
    "build_default_leak_rules",
    "install_git_hooks",
    "run_repository_audit",
    "scan_pdf_for_leaks",
    "scan_text_for_leaks",
    "verify_git_hooks_active",
    "verify_python_ast_quality",
]
