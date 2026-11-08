"""CLI for split-environment feedback bundle verification and ingestion.

Enables a zero-credential authoring loop:
1. Personal machine runs live BIRD evaluation and exports a sanitized,
   SHA-256-signed `FeedbackBundle` (`bird_feedback_bundle.json`).
2. Authoring workspace (or GitHub Actions CI) runs `argus-feedback ingest` or
   `argus-feedback verify` to validate integrity, regenerate `paper/tables/*.tex`,
   and produce an actionable failure triage Markdown report offline.
"""

import argparse
from collections.abc import Sequence
import pathlib
import sys

from argus_sql.evaluation.exceptions import EvaluationError
from argus_sql.evaluation.feedback_bundle import load_feedback_bundle_file
from argus_sql.evaluation.feedback_bundle import serialize_feedback_bundle
from argus_sql.evaluation.feedback_triage import cluster_system_failures
from argus_sql.evaluation.feedback_triage import find_clean_and_voice_suites
from argus_sql.evaluation.feedback_triage import render_feedback_triage_markdown
from argus_sql.evaluation.feedback_triage import write_confined_file
from argus_sql.evaluation.latex_exporter import escape_latex_text
from argus_sql.evaluation.latex_exporter import render_comparative_latex_table
from argus_sql.evaluation.types import BenchmarkSuiteReport
from argus_sql.evaluation.types import FeedbackBundle


def _rel_display(repo_root: pathlib.Path, path: pathlib.Path) -> str:
  """Formats `path` relative to `repo_root` so host directories stay private."""
  # Canonicalize both paths before checking repository prefix containment.
  resolved_root = repo_root.resolve()
  resolved_path = path.resolve()
  # Never echo host home directories to stdout or CI build logs.
  if resolved_path.is_relative_to(resolved_root):
    return resolved_path.relative_to(resolved_root).as_posix()
  return resolved_path.name


def _sync_latex_from_bundle(
    repo_root: pathlib.Path,
    latex_dir: pathlib.Path,
    latex_filename: str,
    bundle: FeedbackBundle,
) -> pathlib.Path | None:
  """Regenerates `bird_live.tex` when a Clean suite exists in `bundle`."""
  # Resolve Clean and Voice reports; render `--` if Voice split was omitted.
  clean_rep, voice_opt = find_clean_and_voice_suites(bundle.suites)
  if clean_rep is None:
    return None
  # Avoid duplicating Clean EX% as Voice EX% when only Clean was evaluated.
  voice_rep = voice_opt or BenchmarkSuiteReport("Voice-Omitted", (), ())
  # Escape underscores and percent signs so arbitrary model slugs compile in TeX.
  safe_model = escape_latex_text(bundle.model_label)
  tex = render_comparative_latex_table(
      clean_rep,
      voice_rep,
      caption=f"Live BIRD-SQL evaluation using {safe_model}.",
      label="tab:bird_live",
  )
  # Confine LaTeX output writes strictly inside `repo_root`.
  target = (
      latex_dir if latex_dir.is_absolute() else (repo_root.resolve() / latex_dir)
  ) / latex_filename
  return write_confined_file(repo_root, target, tex)


def _run_ingest_command(args: argparse.Namespace) -> int:
  """Executes `ingest` subcommand: verify bundle, emit triage & LaTeX table."""
  # Resolve confining workspace root and target bundle path before reading.
  repo_root: pathlib.Path = args.repo_root.resolve()
  bundle_arg: pathlib.Path = args.bundle
  bundle_path = (
      bundle_arg if bundle_arg.is_absolute() else (repo_root / bundle_arg)
  )
  try:
    # Verify file confinement, clean-room regex rules, and SHA-256 digest first.
    bundle = load_feedback_bundle_file(bundle_path, allowed_root=repo_root)
    # Optionally promote a verified quarantine bundle into `results/feedback/`.
    if args.promote_to is not None:
      write_confined_file(
          repo_root, args.promote_to, serialize_feedback_bundle(bundle)
      )
    # Render Markdown root-cause report and sync publication LaTeX fragment.
    report_md = render_feedback_triage_markdown(
        bundle, target_system=args.target_system
    )
    report_path = write_confined_file(repo_root, args.report_out, report_md)
    tex_path = _sync_latex_from_bundle(
        repo_root, args.latex_out_dir, args.latex_filename, bundle
    )
  except EvaluationError as err:
    sys.stderr.write(f"[FAIL] Feedback ingestion aborted: {err}\n")
    return 1

  # Print concise workspace-relative summary for developer and AI iteration.
  sys.stdout.write(
      f"[OK] Verified feedback bundle ({bundle.provider}/{bundle.model_label}, "
      f"sha256={bundle.payload_sha256[:12]}...).\n"
  )
  sys.stdout.write(
      f"[OK] Wrote triage report: {_rel_display(repo_root, report_path)}\n"
  )
  if tex_path is not None:
    sys.stdout.write(
        f"[OK] Updated LaTeX table: {_rel_display(repo_root, tex_path)}\n"
    )

  # Highlight top ranked failure clusters on stdout for rapid triage.
  clean_suite, _ = find_clean_and_voice_suites(bundle.suites)
  if clean_suite is not None:
    rec_map = dict(clean_suite.records_by_system)
    target_recs = rec_map.get(args.target_system, ())
    clusters = cluster_system_failures(target_recs)
    sys.stdout.write(
        f"[Triage] {args.target_system}: {len(clusters)} failure clusters across "
        f"{len(target_recs)} Clean tasks.\n"
    )
    for cl in clusters[:5]:
      sys.stdout.write(
          f"  - [{cl.task_count}x] {cl.failure_category} / {cl.sub_signature} "
          f"-> {cl.recommended_fix}\n"
      )
  return 0


def _run_verify_command(args: argparse.Namespace) -> int:
  """Executes `verify` subcommand across all `*.json` bundles in a folder."""
  # Confine feedback directory traversal strictly inside `repo_root`.
  repo_root: pathlib.Path = args.repo_root.resolve()
  fb_dir = (
      args.feedback_dir
      if args.feedback_dir.is_absolute()
      else (repo_root / args.feedback_dir)
  )
  if fb_dir.is_symlink() or not fb_dir.resolve().is_relative_to(repo_root):
    sys.stderr.write("[FAIL] Feedback directory escapes repository boundary.\n")
    return 1
  # Allow clean repositories with zero live feedback bundles to pass CI cleanly.
  if not fb_dir.exists():
    sys.stdout.write("[OK] Feedback directory absent (0 bundles to verify).\n")
    return 0

  # Validate every JSON bundle in deterministic lexicographical order.
  bundles = sorted(
      p for p in fb_dir.glob("*.json") if p.is_file() and not p.is_symlink()
  )
  for b_file in bundles:
    try:
      load_feedback_bundle_file(b_file, allowed_root=repo_root)
    except EvaluationError as err:
      sys.stderr.write(f"[FAIL] Invalid bundle '{b_file.name}': {err}\n")
      return 1

  sys.stdout.write(
      f"[PASS] Verified {len(bundles)} feedback bundle(s) in {fb_dir.name}.\n"
  )
  return 0


def main(argv: Sequence[str] | None = None) -> int:
  """Entrypoint for `argus-feedback` (`ingest` and `verify` subcommands).

  Args:
    argv: Optional CLI argument sequence.

  Returns:
    Exit code 0 on success or 1 on validation/ingestion failure.
  """
  # Configure top-level CLI parser with mutually exclusive subcommands.
  parser = argparse.ArgumentParser(
      description="Argus-SQL Split-Environment Feedback Verifier & Ingester."
  )
  sub = parser.add_subparsers(dest="subcommand", required=True)

  # Subcommand 1: `ingest` — brings personal evaluation feedback into workspace.
  ing = sub.add_parser(
      "ingest",
      help="Verify a feedback bundle, update LaTeX tables, and emit triage report.",
  )
  # Require explicit bundle path (`audit_import/*.json` or `results/feedback/*.json`).
  ing.add_argument(
      "--bundle",
      type=pathlib.Path,
      required=True,
      help="Path to `bird_feedback_bundle.json` (in `results/feedback` or `audit_import`).",
  )
  # Confine every read/write operation strictly within `--repo-root`.
  ing.add_argument(
      "--repo-root",
      type=pathlib.Path,
      default=pathlib.Path.cwd(),
      help="Confining repository root directory.",
  )
  ing.add_argument(
      "--promote-to",
      type=pathlib.Path,
      default=None,
      help="Optional canonical destination (e.g., `results/feedback/bird_feedback_bundle.json`).",
  )
  ing.add_argument(
      "--report-out",
      type=pathlib.Path,
      default=pathlib.Path("results/feedback/triage_report.md"),
      help="Destination path for the generated Markdown triage report.",
  )
  ing.add_argument(
      "--latex-out-dir",
      type=pathlib.Path,
      default=pathlib.Path("paper/tables"),
      help="Directory where `bird_live.tex` is regenerated.",
  )
  ing.add_argument(
      "--latex-filename",
      type=str,
      default="bird_live.tex",
      help="LaTeX table filename written inside `--latex-out-dir`.",
  )
  ing.add_argument(
      "--target-system",
      type=str,
      default="Argus-SQL",
      help="Primary system name analyzed in deep-dive failure sections.",
  )

  # Subcommand 2: `verify` — CI gate validating committed `results/feedback/*.json`.
  ver = sub.add_parser(
      "verify",
      help="Verify SHA-256 integrity and clean-room safety of feedback bundles.",
  )
  ver.add_argument(
      "--feedback-dir",
      type=pathlib.Path,
      default=pathlib.Path("results/feedback"),
      help="Directory containing `*.json` feedback bundles.",
  )
  ver.add_argument(
      "--repo-root",
      type=pathlib.Path,
      default=pathlib.Path.cwd(),
      help="Confining repository root directory.",
  )

  # Dispatch selected subcommand and propagate deterministic POSIX exit code.
  args = parser.parse_args(argv)
  if args.subcommand == "ingest":
    return _run_ingest_command(args)
  return _run_verify_command(args)


if __name__ == "__main__":
  sys.exit(main())
