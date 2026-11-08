"""Thin CLI entrypoint for Argus-SQL clean-room and AST quality guardrails."""

import argparse
from collections.abc import Sequence
import pathlib
import sys

from argus_sql.guardrails.clean_room_scanner import scan_text_for_leaks
from argus_sql.guardrails.exceptions import GuardrailError
from argus_sql.guardrails.hook_installer import bootstrap_clean_git_config
from argus_sql.guardrails.hook_installer import install_git_hooks
from argus_sql.guardrails.repository_auditor import audit_target_path
from argus_sql.guardrails.repository_auditor import run_repository_audit
from argus_sql.guardrails.types import GuardrailAuditReport


def _emit_audit_report(report: GuardrailAuditReport) -> int:
  """Prints audit status and returns 0 when clean or 1 on any violation."""
  # Return zero immediately when every clean-room, Git, and AST check passed.
  if report.passed:
    # Summarize scanned file count on stdout for CI job logs.
    sys.stdout.write(
        f"[PASS] Clean-room & AST audit passed ({report.scanned_files} files).\n"
    )
    return 0

  # Emit actionable line-level diagnostics for any leak or AST violation.
  sys.stderr.write(
      f"[FAIL] Guardrail audit failed across {report.scanned_files} files:\n"
  )
  # Report clean-room IP/secret/path findings with redacted previews first.
  for finding in report.leak_findings:
    sys.stderr.write(
        f"  - [{finding.rule_id}] {finding.target}:{finding.line_number}: "
        f"{finding.description} -> {finding.redacted_preview}\n"
    )
  # Report AST modularity, typing, docstring, and batch-safety violations next.
  for viol in report.quality_violations:
    sys.stderr.write(
        f"  - [{viol.rule_id}] {viol.target}:{viol.line_number}: {viol.message}\n"
    )
  return 1


def _validate_commit_message_file(
    repo_root: pathlib.Path, msg_file: pathlib.Path
) -> int:
  """Scans a Git commit message file and returns 0 if clean or 1 on leak."""
  resolved_root = repo_root.resolve()
  candidate_msg = (
      msg_file if msg_file.is_absolute() else (resolved_root / msg_file)
  )
  resolved_msg = candidate_msg.resolve()
  # Confine commit-msg reads strictly to inside repo_root to prevent traversal.
  if not resolved_msg.is_relative_to(resolved_root):
    sys.stderr.write("[FAIL] Commit message path lies outside repository.\n")
    return 1

  # Scan every line of the commit message against all clean-room regex rules.
  content = resolved_msg.read_text(encoding="utf-8")
  findings = scan_text_for_leaks(content, target_label="COMMIT_EDITMSG")
  if not findings:
    return 0

  # Print redacted violation diagnostics and return non-zero to abort git commit.
  sys.stderr.write("[FAIL] Clean-room violation in commit message:\n")
  for item in findings:
    sys.stderr.write(
        f"  - [{item.rule_id}] line {item.line_number}: {item.description} "
        f"({item.redacted_preview})\n"
    )
  return 1


def main(argv: Sequence[str] | None = None) -> int:
  """Runs the Argus-SQL clean-room and AST quality CLI gate.

  Args:
    argv: Optional CLI argument sequence.

  Returns:
    Exit status 0 when all requested guardrail checks pass; 1 otherwise.
  """
  # Parse repository target, hook installation, CI bootstrap, and scan flags.
  parser = argparse.ArgumentParser(
      description="Argus-SQL Clean-Room IP, Secret & AST Guardrail CLI."
  )
  parser.add_argument(
      "--repo-root",
      type=pathlib.Path,
      default=pathlib.Path.cwd(),
      help="Path to the Argus-SQL repository root.",
  )
  parser.add_argument(
      "--install-hooks",
      action="store_true",
      help="Install executable pre-commit and commit-msg hooks in .githooks.",
  )
  parser.add_argument(
      "--bootstrap-git",
      action="store_true",
      help="Set clean local Git identity defaults (if unset) and install hooks.",
  )
  parser.add_argument(
      "--scan-path",
      type=pathlib.Path,
      default=None,
      help="Explicitly scan a quarantined file or directory (e.g., audit_import).",
  )
  parser.add_argument(
      "--verify-all",
      action="store_true",
      help="Run full clean-room leak, Git identity, and AST quality audit.",
  )
  parser.add_argument(
      "--check-commit-msg",
      type=pathlib.Path,
      default=None,
      help="Validate a Git COMMIT_EDITMSG file against clean-room leak rules.",
  )
  args = parser.parse_args(argv)

  # Bootstrap clean local Git identity + hooks on fresh clones or CI runners.
  if args.bootstrap_git:
    pre_path, msg_path = bootstrap_clean_git_config(args.repo_root)
    sys.stdout.write(
        f"[OK] Bootstrapped clean Git config & hooks: {pre_path.name}, {msg_path.name}\n"
    )
  elif args.install_hooks:
    pre_path, msg_path = install_git_hooks(args.repo_root)
    sys.stdout.write(f"[OK] Installed hooks: {pre_path.name}, {msg_path.name}\n")

  # Validate commit message file when invoked from the Git `commit-msg` hook.
  if args.check_commit_msg is not None:
    return _validate_commit_message_file(args.repo_root, args.check_commit_msg)

  # Audit an explicit quarantine path (`--scan-path`) when requested.
  if args.scan_path is not None:
    try:
      target_rep = audit_target_path(args.repo_root, args.scan_path)
    except GuardrailError as err:
      sys.stderr.write(f"[FAIL] {err}\n")
      return 1
    if _emit_audit_report(target_rep) != 0:
      return 1
    if not args.verify_all:
      return 0

  # Default to running a full audit unless only hook/git setup was requested.
  setup_only = (args.install_hooks or args.bootstrap_git) and not args.verify_all
  if args.verify_all or not setup_only:
    return _emit_audit_report(run_repository_audit(args.repo_root))

  return 0


if __name__ == "__main__":
  sys.exit(main())
