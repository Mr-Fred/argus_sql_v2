"""Automated offline root-cause failure clustering and Markdown triage reporter.

Transforms a sanitized `FeedbackBundle` collected on a personal evaluation
machine into ranked module-level engineering actions, per-database bottleneck
tables, Clean-vs-Voice ASR regression diffs, and sample broken SQL traces.
"""

from collections.abc import Sequence
import pathlib
import subprocess

from argus_sql.evaluation.bird_metrics import compute_mcnemar_p_value
from argus_sql.evaluation.exceptions import EvaluationError
from argus_sql.evaluation.feedback_bundle import build_feedback_bundle
from argus_sql.evaluation.feedback_bundle import serialize_feedback_bundle
from argus_sql.evaluation.types import BenchmarkSuiteReport
from argus_sql.evaluation.types import FailureCluster
from argus_sql.evaluation.types import FeedbackBundle
from argus_sql.evaluation.types import TaskEvalRecord
from argus_sql.evaluation.types import VoiceDriftCase


def _classify_failure_subsignature(rec: TaskEvalRecord) -> tuple[str, str]:
  """Maps one failed `TaskEvalRecord` to `(sub_signature, recommended_fix)`."""
  # Lowercase once so SQLite error substring probes stay case-insensitive.
  err = rec.error_summary.lower()
  if rec.failure_category == "SYNTAX_OR_JOIN_KEY_ERROR":
    # Separate schema-linking omissions from SQLite dialect syntax mistakes.
    if "no such table" in err:
      return (
          "MISSING_SCHEMA_TABLE",
          "Raise FK hop cap or table lexical weight in `fk_schema_linker.py`.",
      )
    # Column lookup or join-alias ambiguity points to Stage 2 column selection.
    if "no such column" in err or "ambiguous column" in err:
      return (
          "MISSING_OR_AMBIGUOUS_COLUMN",
          "Increase column top-k retention or table-alias rules in `linking/`.",
      )
    # Parser errors indicate malformed SQL dialect constructs (e.g. TOP / ILIKE).
    if "syntax error" in err or "incomplete input" in err:
      return (
          "SQLITE_SYNTAX_ERROR",
          "Add deterministic dialect rewrite in `verification/sql_repairer.py`.",
      )
    # VM step-budget interruptions expose unbounded cross-join explosions.
    if "interrupted" in err or "timeout" in err:
      return (
          "SANDBOX_STEP_TIMEOUT",
          "Flag Cartesian missing-ON joins in `verification/semantic_verifier.py`.",
      )
    return (
        "EXECUTION_RUNTIME_ERROR",
        "Inspect `error_summary` and add sandbox execution fallback repair.",
    )

  # Zero-row outputs on valid syntax typically reflect ungrounded string literals.
  if rec.failure_category == "EMPTY_RESULT_VALUE_DRIFT":
    return (
        "EMPTY_PREDICTED_RESULT",
        "Expand 3-gram cell literal top-k or case-insensitive match in `linking/`.",
    )

  # Distinguish invariant-flagged semantic bugs from unflagged formula drift.
  if rec.invariant_codes:
    joined = ",".join(rec.invariant_codes)
    return (
        f"INVARIANT_FLAGGED:{joined}",
        "Strengthen consensus penalty or algebraic repair in `verification/`.",
    )
  return (
      "UNFLAGGED_LOGIC_OR_AGGREGATION_DRIFT",
      "Compare `predicted_sql` against domain formulas; add new SQLite invariant.",
  )


def cluster_system_failures(
    records: Sequence[TaskEvalRecord],
) -> tuple[FailureCluster, ...]:
  """Clusters failed `TaskEvalRecord` items into ranked root-cause buckets.

  Args:
    records: Sequence of evaluated task records for a single system.

  Returns:
    Tuple of `FailureCluster` objects sorted descending by `task_count`.
  """
  # Group failed tasks by `(failure_category, sub_signature)` key.
  buckets: dict[tuple[str, str], list[TaskEvalRecord]] = {}
  remedies: dict[tuple[str, str], str] = {}
  for rec in records:
    # Skip matching predictions so triage focuses strictly on errors.
    if rec.ex_correct:
      continue
    sub_sig, fix_hint = _classify_failure_subsignature(rec)
    key = (rec.failure_category, sub_sig)
    buckets.setdefault(key, []).append(rec)
    remedies[key] = fix_hint

  # Materialize immutable cluster summaries with capped sample QIDs.
  clusters: list[FailureCluster] = []
  for (cat, sub_sig), items in buckets.items():
    dbs = tuple(sorted({r.db_id for r in items}))
    sample_qids = tuple(r.question_id for r in items[:6])
    clusters.append(
        FailureCluster(
            failure_category=cat,
            sub_signature=sub_sig,
            task_count=len(items),
            affected_dbs=dbs,
            sample_question_ids=sample_qids,
            recommended_fix=remedies[(cat, sub_sig)],
        )
    )
  # Order highest-volume failure clusters first to maximize fix leverage.
  clusters.sort(key=lambda c: (-c.task_count, c.failure_category, c.sub_signature))
  return tuple(clusters)


def extract_voice_drift_cases(
    clean_records: Sequence[TaskEvalRecord],
    voice_records: Sequence[TaskEvalRecord],
) -> tuple[VoiceDriftCase, ...]:
  """Identifies tasks where Clean text accuracy and spoken Voice accuracy diverge.

  Args:
    clean_records: Task records evaluated on clean text questions.
    voice_records: Task records evaluated on spoken ASR-perturbed questions.

  Returns:
    Tuple of `VoiceDriftCase` items prioritizing `Clean=Pass -> Voice=Fail`.
  """
  # Index voice records by both raw `question_id` and stripped `*_voice` base ID
  # because `perturb_task_for_voice` appends `_voice` to Clean task identifiers.
  voice_by_qid: dict[str, TaskEvalRecord] = {}
  for v_item in voice_records:
    voice_by_qid[v_item.question_id] = v_item
    if v_item.question_id.endswith("_voice"):
      voice_by_qid.setdefault(v_item.question_id.removesuffix("_voice"), v_item)

  drifts: list[VoiceDriftCase] = []
  for c_rec in clean_records:
    v_rec = voice_by_qid.get(c_rec.question_id)
    # Retain only discordant pairs where spoken perturbations altered outcome.
    if v_rec is None or c_rec.ex_correct == v_rec.ex_correct:
      continue
    drifts.append(
        VoiceDriftCase(
            question_id=c_rec.question_id,
            db_id=c_rec.db_id,
            difficulty=c_rec.difficulty,
            clean_ex=c_rec.ex_correct,
            voice_ex=v_rec.ex_correct,
            voice_failure_category=v_rec.failure_category,
            voice_predicted_sql=v_rec.predicted_sql,
        )
    )
  # Surface spoken regressions (`clean_ex=True, voice_ex=False`) at the top.
  drifts.sort(key=lambda d: (not (d.clean_ex and not d.voice_ex), d.question_id))
  return tuple(drifts)


def compute_per_database_accuracy(
    records: Sequence[TaskEvalRecord],
) -> tuple[tuple[str, int, float], ...]:
  """Computes `(db_id, task_count, ex_pct)` sorted lowest accuracy first.

  Args:
    records: Sequence of evaluated `TaskEvalRecord` items.

  Returns:
    Tuple of `(db_id, total_tasks, ex_percentage)` ordered ascending by EX%.
  """
  # Partition task outcomes by physical database identifier.
  by_db: dict[str, list[TaskEvalRecord]] = {}
  for rec in records:
    by_db.setdefault(rec.db_id, []).append(rec)

  # Rank lowest-EX databases first (breaking ties by larger task volume).
  stats: list[tuple[str, int, float]] = []
  for db_id, items in by_db.items():
    hits = sum(1 for r in items if r.ex_correct)
    pct = round(100.0 * hits / len(items), 1)
    stats.append((db_id, len(items), pct))
  stats.sort(key=lambda row: (row[2], -row[1], row[0]))
  return tuple(stats)


def find_clean_and_voice_suites(
    suites: Sequence[BenchmarkSuiteReport],
) -> tuple[BenchmarkSuiteReport | None, BenchmarkSuiteReport | None]:
  """Resolves primary Clean and Voice suites from `FeedbackBundle.suites`.

  Args:
    suites: Ordered sequence of `BenchmarkSuiteReport` objects in a bundle.

  Returns:
    Pair `(clean_suite, voice_suite)` where absent splits are `None`.
  """
  clean_s: BenchmarkSuiteReport | None = None
  voice_s: BenchmarkSuiteReport | None = None
  # Match `voice` substring case-insensitively to pair Clean and Spoken splits.
  for suite in suites:
    lowered = suite.suite_name.lower()
    if "voice" in lowered and voice_s is None:
      voice_s = suite
    elif clean_s is None:
      clean_s = suite
  return clean_s, voice_s


def render_feedback_triage_markdown(
    bundle: FeedbackBundle,
    target_system: str = "Argus-SQL",
    max_failed_examples: int = 12,
) -> str:
  """Renders an actionable GFM Markdown triage report from a `FeedbackBundle`.

  Args:
    bundle: Integrity-verified `FeedbackBundle` imported from personal eval.
    target_system: Primary pipeline name analyzed in deep-dive sections.
    max_failed_examples: Maximum individual failed SQL traces emitted.

  Returns:
    Formatted Markdown report ready for offline engineering iteration.
  """
  # Emit run provenance header including truncated verified SHA-256 prefix.
  lines: list[str] = [
      "# Argus-SQL Split-Environment Feedback Triage Report",
      "",
      "## 1. Run Provenance & Integrity",
      "",
      f"- **Schema**: `{bundle.schema_version}`",
      f"- **Created (UTC)**: `{bundle.created_utc}`",
      f"- **Provider / Model**: `{bundle.provider}` / `{bundle.model_label}`",
      f"- **Dataset Split**: `{bundle.split_name}`",
      f"- **Source Commit**: `{bundle.git_commit}`",
      f"- **Payload SHA-256**: `{bundle.payload_sha256[:16]}...` (verified)",
      "",
      "## 2. Leaderboard Summary",
      "",
      "| Suite | System | Tasks | EX (%) | Simple | Mod. | Chall. | R-VES | p50 (ms) | Mean Tok |",
      "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
  ]

  # Render every suite/system leaderboard row for side-by-side comparison.
  for suite in bundle.suites:
    for s in suite.summaries:
      lines.append(
          f"| {suite.suite_name} | `{s.system_name}` | {s.total_tasks} | "
          f"**{s.ex_overall:.1f}** | {s.ex_simple:.1f} | {s.ex_moderate:.1f} | "
          f"{s.ex_challenging:.1f} | {s.r_ves_overall:.1f} | "
          f"{s.median_latency_ms:.1f} | {s.mean_tokens:.0f} |"
      )

  # Resolve Clean and Voice splits to compute paired statistical significance.
  clean_suite, voice_suite = find_clean_and_voice_suites(bundle.suites)
  clean_map = dict(clean_suite.records_by_system) if clean_suite else {}
  voice_map = dict(voice_suite.records_by_system) if voice_suite else {}

  # Append exact paired McNemar significance vs baselines when available.
  if target_system in clean_map:
    target_clean = clean_map[target_system]
    for base_name, base_recs in clean_map.items():
      if base_name != target_system and len(base_recs) == len(target_clean):
        p_val = compute_mcnemar_p_value(base_recs, target_clean)
        lines.append("")
        lines.append(
            f"- **Paired McNemar (`{target_system}` vs `{base_name}`, Clean, "
            f"N={len(target_clean)})**: `p = {p_val:.3e}`"
        )

  # Deep-dive into `target_system` per-DB accuracy and root-cause clusters.
  primary_recs = clean_map.get(target_system, ())
  if primary_recs:
    lines.extend([
        "",
        f"## 3. Per-Database Accuracy (`{target_system}` Clean Split)",
        "",
        "| Database (`db_id`) | Tasks | EX (%) |",
        "| :--- | :---: | :---: |",
    ])
    for db_id, count, pct in compute_per_database_accuracy(primary_recs):
      lines.append(f"| `{db_id}` | {count} | {pct:.1f}% |")

    clusters = cluster_system_failures(primary_recs)
    lines.extend([
        "",
        f"## 4. Ranked Failure Clusters (`{target_system}`)",
        "",
    ])
    if not clusters:
      lines.append("Zero failures detected on the Clean split (100.0% EX).")
    else:
      lines.extend([
          "| Count | Category | Sub-Signature | Databases | Sample QIDs | Recommended Module Action |",
          "| :---: | :--- | :--- | :--- | :--- | :--- |",
      ])
      for cl in clusters:
        dbs_str = ", ".join(f"`{d}`" for d in cl.affected_dbs[:4])
        qids_str = ", ".join(f"`{q}`" for q in cl.sample_question_ids)
        lines.append(
            f"| {cl.task_count} | `{cl.failure_category}` | `{cl.sub_signature}` "
            f"| {dbs_str} | {qids_str} | {cl.recommended_fix} |"
        )

  # Surface Clean-vs-Voice ASR canonicalization regressions when paired.
  voice_recs = voice_map.get(target_system, ())
  if primary_recs and voice_recs:
    drifts = extract_voice_drift_cases(primary_recs, voice_recs)
    lines.extend(["", "## 5. Spoken Voice-BIRD Drift Analysis", ""])
    if not drifts:
      lines.append("Zero Clean-vs-Voice discrepancies observed.")
    else:
      for dr in drifts[:10]:
        lines.append(
            f"- **QID `{dr.question_id}` (`{dr.db_id}`, `{dr.difficulty.value}`)**: "
            f"Clean EX=`{dr.clean_ex}` -> Voice EX=`{dr.voice_ex}` "
            f"(`{dr.voice_failure_category}`)"
        )

  # Emit concrete failed SQL traces so unit tests can be authored offline.
  failed = [r for r in primary_recs if not r.ex_correct][:max_failed_examples]
  if failed:
    lines.extend(["", "## 6. Sample Sanitized Failed SQL Traces", ""])
    for rec in failed:
      invs = ",".join(rec.invariant_codes) if rec.invariant_codes else "none"
      err_note = f" | Error: `{rec.error_summary}`" if rec.error_summary else ""
      lines.append(
          f"### QID `{rec.question_id}` (`{rec.db_id}` — `{rec.difficulty.value}`)"
      )
      lines.append(
          f"- **Category**: `{rec.failure_category}` | **Strategy**: "
          f"`{rec.selected_strategy or 'n/a'}` | **Invariants**: `{invs}`{err_note}"
      )
      lines.append("```sql")
      lines.append(rec.predicted_sql or "-- <empty SQL>")
      lines.append("```")
      lines.append("")

  return "\n".join(lines).rstrip() + "\n"


def write_confined_file(
    allowed_root: pathlib.Path,
    target_path: pathlib.Path,
    content: str,
) -> pathlib.Path:
  """Writes UTF-8 `content` strictly inside `allowed_root` (no symlinks).

  Args:
    allowed_root: Directory boundary that `target_path` must resolve inside.
    target_path: Target file path (relative to `allowed_root` or absolute).
    content: UTF-8 string payload written to disk.

  Returns:
    Resolved destination path written.

  Raises:
    EvaluationError: If `target_path` is a symlink or escapes `allowed_root`.
  """
  # Canonicalize the confining root directory before computing destination path.
  resolved_root = allowed_root.resolve()
  candidate = (
      target_path if target_path.is_absolute() else (resolved_root / target_path)
  )
  # Block symlink writes so callers cannot redirect output outside repository.
  if candidate.is_symlink():
    raise EvaluationError("Refusing to write through a symlink destination.")
  resolved_dest = candidate.resolve()
  # Enforce strict directory prefix containment via `Path.is_relative_to`.
  if not resolved_dest.is_relative_to(resolved_root):
    raise EvaluationError("Destination path escapes repository root boundary.")
  resolved_dest.parent.mkdir(parents=True, exist_ok=True)
  resolved_dest.write_text(content, encoding="utf-8", newline="\n")
  return resolved_dest


def resolve_short_git_commit(repo_root: pathlib.Path | None = None) -> str:
  """Returns the 12-char Git HEAD commit SHA or `'uncommitted'` when absent.

  Args:
    repo_root: Optional Git working tree directory.

  Returns:
    Sanitized short hexadecimal commit identifier or `'uncommitted'`.
  """
  # Query local Git HEAD safely without shell interpolation.
  cwd = str((repo_root or pathlib.Path.cwd()).resolve())
  try:
    proc = subprocess.run(
        ["git", "-C", cwd, "rev-parse", "--short=12", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
  except OSError:
    return "uncommitted"
  sha = proc.stdout.strip()
  # Accept strictly lowercase hex characters so branch names/errors never leak.
  if proc.returncode == 0 and sha and all(c in "0123456789abcdef" for c in sha):
    return sha
  return "uncommitted"


def export_feedback_artifacts(
    out_dir: pathlib.Path,
    suites: Sequence[BenchmarkSuiteReport],
    provider: str,
    model_label: str,
    split_name: str,
    bundle_filename: str = "bird_feedback_bundle.json",
    triage_filename: str = "triage_report.md",
    repo_root: pathlib.Path | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
  """Builds and writes a verified `FeedbackBundle` JSON + triage Markdown.

  Args:
    out_dir: Output directory where sanitized feedback artifacts are written.
    suites: Evaluated `BenchmarkSuiteReport` sequence (Clean + Voice).
    provider: Provider transport label ('vertex', 'openai', or 'diagnostic').
    model_label: Evaluated model slug.
    split_name: Dataset split filename.
    bundle_filename: Output JSON bundle filename.
    triage_filename: Output Markdown triage report filename.
    repo_root: Optional root boundary confining output writes.

  Returns:
    Pair `(bundle_json_path, triage_markdown_path)`.
  """
  # Confine writes to `out_dir.resolve()` when `repo_root` is omitted.
  boundary = (repo_root or out_dir).resolve()
  commit_sha = resolve_short_git_commit(repo_root)
  # Scrub machine paths and compute canonical SHA-256 over all suite records.
  bundle = build_feedback_bundle(
      suites=suites,
      provider=provider,
      model_label=model_label,
      split_name=split_name,
      git_commit=commit_sha,
  )
  resolved_out = out_dir.resolve()
  # Persist both the machine-verifiable JSON bundle and human triage Markdown.
  bundle_dest = write_confined_file(
      boundary, resolved_out / bundle_filename, serialize_feedback_bundle(bundle)
  )
  triage_dest = write_confined_file(
      boundary,
      resolved_out / triage_filename,
      render_feedback_triage_markdown(bundle),
  )
  return bundle_dest, triage_dest
