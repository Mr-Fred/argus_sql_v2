"""Repository-wide clean-room, PDF, Git hook, and AST quality audit coordinator.

Separates pure repository auditing orchestration from the CLI entrypoint so
both unit tests and CLI commands can import it without circular runpy warnings.
"""

import pathlib

from argus_sql.guardrails.clean_room_scanner import build_default_leak_rules
from argus_sql.guardrails.clean_room_scanner import iter_scannable_repo_files
from argus_sql.guardrails.clean_room_scanner import scan_pdf_for_leaks
from argus_sql.guardrails.clean_room_scanner import scan_text_for_leaks
from argus_sql.guardrails.clean_room_scanner import verify_git_author_identity
from argus_sql.guardrails.code_quality_verifier import verify_python_ast_quality
from argus_sql.guardrails.exceptions import GuardrailError
from argus_sql.guardrails.hook_installer import verify_git_hooks_active
from argus_sql.guardrails.types import CodeQualityViolation
from argus_sql.guardrails.types import GuardrailAuditReport
from argus_sql.guardrails.types import LeakFinding


def _audit_file_list(
    resolved_root: pathlib.Path,
    files: tuple[pathlib.Path, ...],
) -> tuple[tuple[LeakFinding, ...], tuple[CodeQualityViolation, ...]]:
  """Scans a resolved sequence of repository files for leaks and AST issues."""
  rules = build_default_leak_rules()
  leak_findings: list[LeakFinding] = []
  quality_violations: list[CodeQualityViolation] = []

  # Inspect every text/source/LaTeX/PDF artifact for leaks and AST compliance.
  for file_path in files:
    rel_label = file_path.relative_to(resolved_root).as_posix()
    # Route compiled `.pdf` manuscripts through zlib stream + metadata scanning.
    if file_path.suffix.lower() == ".pdf":
      leak_findings.extend(scan_pdf_for_leaks(file_path, rel_label, rules=rules))
      continue

    # Skip unexpected non-UTF8 binary files gracefully while auditing text files.
    try:
      text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
      continue
    leak_findings.extend(scan_text_for_leaks(text, rel_label, rules=rules))
    # Run static Python AST checks on every `.py` source and test module.
    if file_path.suffix == ".py":
      quality_violations.extend(
          verify_python_ast_quality(file_path, target_label=rel_label)
      )
  return tuple(leak_findings), tuple(quality_violations)


def run_repository_audit(repo_root: pathlib.Path) -> GuardrailAuditReport:
  """Executes clean-room IP, PDF, secret, Git, and AST checks on `repo_root`.

  Args:
    repo_root: Root directory of the Argus-SQL repository.

  Returns:
    GuardrailAuditReport summarizing all scanned files and any violations.
  """
  # Canonicalize repo_root once so relative path reporting is deterministic.
  resolved_root = repo_root.resolve()
  files = iter_scannable_repo_files(resolved_root)
  leak_findings: list[LeakFinding] = []

  # Verify Git author configuration and active commit hooks when `.git` exists.
  git_ok = True
  hooks_ok = True
  if (resolved_root / ".git").is_dir():
    git_findings = verify_git_author_identity(resolved_root)
    if git_findings:
      git_ok = False
      leak_findings.extend(git_findings)
    hook_findings = verify_git_hooks_active(resolved_root)
    if hook_findings:
      hooks_ok = False
      leak_findings.extend(hook_findings)

  file_leaks, quality_violations = _audit_file_list(resolved_root, files)
  leak_findings.extend(file_leaks)
  return GuardrailAuditReport(
      scanned_files=len(files),
      leak_findings=tuple(leak_findings),
      quality_violations=quality_violations,
      git_identity_ok=git_ok,
      git_hooks_ok=hooks_ok,
  )


def audit_target_path(
    repo_root: pathlib.Path,
    target_path: pathlib.Path,
) -> GuardrailAuditReport:
  """Explicitly scans a single file or subdirectory (including `audit_import/`).

  Args:
    repo_root: Confining root directory of the Argus-SQL repository.
    target_path: Specific file or directory path to audit before ingestion.

  Returns:
    GuardrailAuditReport covering all files discovered under `target_path`.

  Raises:
    GuardrailError: If `target_path` is a symlink, missing, or escapes `repo_root`.
  """
  # Refuse symlinks before resolving so external symlink targets cannot sneak in.
  candidate = (
      target_path if target_path.is_absolute() else (repo_root / target_path)
  )
  if candidate.is_symlink():
    raise GuardrailError("Refusing to scan symlink target path.")

  # Enforce strict directory boundary containment inside `repo_root`.
  resolved_root = repo_root.resolve()
  resolved_target = candidate.resolve()
  if not resolved_target.is_relative_to(resolved_root):
    raise GuardrailError("Target path lies outside repository root boundary.")
  if not resolved_target.exists():
    raise GuardrailError(f"Target scan path does not exist: '{target_path}'.")

  # Enumerate single file directly or walk non-symlink files inside target dir.
  if resolved_target.is_file():
    files = (resolved_target,)
  else:
    # Skip nested symlinks during directory traversal to preserve containment.
    files = tuple(
        p for p in sorted(resolved_target.rglob("*"))
        if p.is_file() and not p.is_symlink()
    )

  # Run full clean-room regex, PDF stream, and Python AST gates on target files.
  file_leaks, quality_violations = _audit_file_list(resolved_root, files)
  # Scoped file scans report Git status as neutral (`True`) unless `--verify-all`.
  return GuardrailAuditReport(
      scanned_files=len(files),
      leak_findings=file_leaks,
      quality_violations=quality_violations,
      git_identity_ok=True,
      git_hooks_ok=True,
  )
