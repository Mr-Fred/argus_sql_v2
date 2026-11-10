"""CLI runner for Argus-SQL diagnostic ablations and public BIRD-SQL splits.

Supports two reproducible modes:
1. `--mode diagnostic`: Materializes the clean-room Mini-BIRD & Voice-BIRD
   databases in a temporary directory, runs controlled operator baselines
   (`B0_RawDDL`..`B4_LLMJudge`) + `Argus-SQL` + 5 ablations, and writes LaTeX
   tables + JSON metrics.
2. `--mode bird`: Evaluates `B0_RawDDL`..`B4_LLMJudge` + `Argus-SQL` on an
   external public BIRD-SQL split (`dev.json` / `mini_dev_sqlite.json`).
"""

import argparse
from collections.abc import Sequence
import pathlib
import sys
import tempfile

from argus_sql.adapters.client_factory import build_intent_llm_client
from argus_sql.adapters.client_factory import build_live_llm_client
from argus_sql.adapters.deterministic_synthesizer import SchemaGuidedDiagnosticSynthesizer
from argus_sql.evaluation.bird_loader import load_bird_split
from argus_sql.evaluation.bird_loader import perturb_task_for_voice
from argus_sql.evaluation.bird_loader import resolve_bird_root
from argus_sql.evaluation.bird_metrics import compute_mcnemar_p_value
from argus_sql.evaluation.feedback_triage import export_feedback_artifacts
from argus_sql.evaluation.latex_exporter import escape_latex_text
from argus_sql.evaluation.latex_exporter import render_comparative_latex_table
from argus_sql.evaluation.latex_exporter import serialize_reports_to_json
from argus_sql.evaluation.live_checkpoint import LiveRunCheckpointer
from argus_sql.evaluation.live_checkpoint import resolve_checkpoint_path
from argus_sql.evaluation.live_checkpoint import select_stratified_bird_tasks
from argus_sql.evaluation.mini_bird_databases import materialize_mini_bird_databases
from argus_sql.evaluation.mini_bird_suite import build_ablation_systems
from argus_sql.evaluation.mini_bird_suite import build_main_systems
from argus_sql.evaluation.mini_bird_suite import build_mini_bird_clean_tasks
from argus_sql.evaluation.mini_bird_suite import build_voice_bird_tasks
from argus_sql.evaluation.mini_bird_suite import run_benchmark_suite
from argus_sql.evaluation.types import BenchmarkSuiteReport
from argus_sql.pipelines.argus_pipeline import ArgusSQLPipeline
from argus_sql.pipelines.types import ArgusAblationConfig
from argus_sql.pipelines.types import NL2SQLPipeline
from argus_sql.sandbox.sqlite_sandbox import SQLiteReadOnlySandbox
from argus_sql.sandbox.types import SandboxPolicy


def _write_confined_output(
    out_root: pathlib.Path, filename: str, content: str
) -> pathlib.Path:
  """Writes `content` safely inside `out_root` after traversal validation."""
  # Resolve destination boundary before touching disk to block path traversal.
  resolved_root = out_root.resolve()
  resolved_root.mkdir(parents=True, exist_ok=True)
  target = (resolved_root / filename).resolve()
  if not target.is_relative_to(resolved_root):
    raise ValueError("Refusing to write outside configured output directory.")
  target.write_text(content, encoding="utf-8", newline="\n")
  return target


def _print_suite_summaries(
    clean_rep: BenchmarkSuiteReport, voice_rep: BenchmarkSuiteReport
) -> None:
  """Prints Clean/Voice leaderboard rows and paired McNemar p-values."""
  # Emit Clean text split accuracy, R-VES, and mean token metrics first.
  for s in clean_rep.summaries:
    sys.stdout.write(
        f"[Clean] {s.system_name:26s} | EX={s.ex_overall:5.1f}% | "
        f"R-VES={s.r_ves_overall:5.1f} | Tok={s.mean_tokens:5.0f}\n"
    )
  # Emit spoken Voice-BIRD ASR-perturbed split metrics for side-by-side comparison.
  for s in voice_rep.summaries:
    sys.stdout.write(
        f"[Voice] {s.system_name:26s} | EX={s.ex_overall:5.1f}% | "
        f"R-VES={s.r_ves_overall:5.1f} | Tok={s.mean_tokens:5.0f}\n"
    )
  # Compute exact two-sided McNemar p-values against B3 and B4 baselines.
  clean_recs = dict(clean_rep.records_by_system)
  voice_recs = dict(voice_rep.records_by_system)
  if "Argus-SQL" not in clean_recs:
    return
  joint_argus = clean_recs["Argus-SQL"] + voice_recs["Argus-SQL"]
  for base_id in ("B3_ValueMajorityVote", "B4_LLMJudge"):
    if base_id in clean_recs and base_id in voice_recs:
      joint_base = clean_recs[base_id] + voice_recs[base_id]
      p_joint = compute_mcnemar_p_value(joint_base, joint_argus)
      p_voice = compute_mcnemar_p_value(
          voice_recs[base_id], voice_recs["Argus-SQL"]
      )
      sys.stdout.write(
          f"[Stats] McNemar (Argus-SQL vs {base_id}): "
          f"Joint={p_joint:.2e} (N={len(joint_argus)}), Voice={p_voice:.2e}\n"
      )


def _build_arg_parser() -> argparse.ArgumentParser:
  """Constructs CLI argument parser for diagnostic and live BIRD modes."""
  # Configure mode selection, BIRD split locations, and provider flags.
  parser = argparse.ArgumentParser(
      description="Argus-SQL Comparative Benchmark & Ablation Runner."
  )
  parser.add_argument("--mode", choices=("diagnostic", "bird"), default="diagnostic")
  # Public BIRD-SQL / Mini-Dev dataset path and split filename parameters.
  parser.add_argument("--bird-dir", type=pathlib.Path, default=None)
  parser.add_argument("--questions-filename", type=str, default="dev.json")
  parser.add_argument("--tables-filename", type=str, default="dev_tables.json")
  parser.add_argument("--databases-subdir", type=str, default="dev_databases")
  # Live LLM adapter routing (Vertex AI / Gemini Enterprise vs. OpenAI/vLLM).
  parser.add_argument("--provider", choices=("auto", "openai", "vertex"), default="auto")
  parser.add_argument("--gcp-project", type=str, default=None)
  parser.add_argument("--gcp-location", type=str, default=None)
  parser.add_argument("--model", type=str, default="auto")
  parser.add_argument(
      "--intent-model",
      type=str,
      default="off",
      help="Optional Stage 1 flash-lite intent model ('off', 'auto', or slug).",
  )
  parser.add_argument(
      "--endpoint-url", type=str, default="https://api.openai.com/v1/chat/completions"
  )
  # Execution subsetting, checkpoint invalidation, and confined output paths.
  parser.add_argument("--only-argus", action="store_true")
  parser.add_argument("--stratified", action="store_true")
  parser.add_argument(
      "--checkpoint-dir", type=pathlib.Path, default=pathlib.Path("results/live_runs")
  )
  # Selective checkpoint eviction avoids re-billing unchanged baseline runs.
  parser.add_argument("--recompute-argus", action="store_true")
  parser.add_argument("--recompute-systems", type=str, default="")
  parser.add_argument("--fresh-checkpoint", action="store_true")
  # Confined output directories for LaTeX tables, JSON metrics, and bundles.
  parser.add_argument(
      "--latex-out-dir", type=pathlib.Path, default=pathlib.Path("paper/tables")
  )
  parser.add_argument("--json-out-dir", type=pathlib.Path, default=pathlib.Path("results"))
  parser.add_argument(
      "--feedback-out-dir", type=pathlib.Path, default=pathlib.Path("results/feedback")
  )
  parser.add_argument("--limit", type=int, default=0)
  parser.add_argument("--task-concurrency", type=int, default=8)
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  """Executes diagnostic or BIRD-SQL evaluations and exports LaTeX & JSON.

  Args:
    argv: Optional CLI argument sequence.

  Returns:
    Exit code 0 upon completion.
  """
  # Parse CLI arguments from `argv` or `sys.argv[1:]`.
  args = _build_arg_parser().parse_args(argv)

  # Hermetic Clean-Room Diagnostic Mode (`Mini-BIRD` + `Voice-BIRD` splits).
  if args.mode == "diagnostic":
    # Materialize ephemeral SQLite fixtures inside an auto-cleaned temp dir.
    with tempfile.TemporaryDirectory() as tmp_db_dir:
      db_root = pathlib.Path(tmp_db_dir)
      schemas, sqlite_paths = materialize_mini_bird_databases(db_root)
      sandbox = SQLiteReadOnlySandbox(SandboxPolicy(allowed_db_root=db_root))
      synth = SchemaGuidedDiagnosticSynthesizer()

      # Build paired 24 clean text queries and 24 spoken ASR utterances.
      clean_tasks = build_mini_bird_clean_tasks()
      voice_tasks = build_voice_bird_tasks()

      # 1. Run controlled operator baselines (B0..B4) vs Argus-SQL.
      main_pipes = build_main_systems(synth, sandbox)
      main_clean = run_benchmark_suite(
          "Mini-BIRD Clean", clean_tasks, main_pipes, schemas, sqlite_paths, sandbox
      )
      main_voice = run_benchmark_suite(
          "Voice-BIRD Spoken", voice_tasks, main_pipes, schemas, sqlite_paths, sandbox
      )

      # 2. Run Leave-One-Out Component Ablation Study.
      abl_pipes = build_ablation_systems(synth, sandbox)
      abl_clean = run_benchmark_suite(
          "Ablation Clean", clean_tasks, abl_pipes, schemas, sqlite_paths, sandbox
      )
      abl_voice = run_benchmark_suite(
          "Ablation Voice", voice_tasks, abl_pipes, schemas, sqlite_paths, sandbox
      )

      # Render booktabs LaTeX fragments included directly by `paper/main.tex`.
      main_tex = render_comparative_latex_table(
          main_clean,
          main_voice,
          caption=(
              "Controlled diagnostic evaluation on Mini-BIRD (24 clean relational "
              "queries) and Voice-BIRD (24 spoken ASR-perturbed utterances) across "
              "controlled backbone-matched operator baselines (B0--B4) and Argus-SQL."
          ),
          label="tab:diagnostic_main",
      )
      abl_tex = render_comparative_latex_table(
          abl_clean,
          abl_voice,
          caption=(
              "Leave-one-out component ablation study of Argus-SQL across Clean "
              "and Voice-BIRD diagnostic splits, isolating each module's impact "
              "on Execution Accuracy (EX), R-VES, and prompt token budget."
          ),
          label="tab:diagnostic_ablation",
      )
      # Persist publication-ready booktabs LaTeX tables and raw JSON traces.
      _write_confined_output(args.latex_out_dir, "diagnostic_main.tex", main_tex)
      _write_confined_output(
          args.latex_out_dir, "diagnostic_ablation.tex", abl_tex
      )
      _write_confined_output(
          args.json_out_dir,
          "diagnostic_benchmark.json",
          serialize_reports_to_json((main_clean, main_voice, abl_clean, abl_voice)),
      )
      # Print final summary table and exact paired McNemar statistical tests.
      _print_suite_summaries(main_clean, main_voice)
    return 0

  # Live public BIRD-SQL / BIRD Mini-Dev evaluation mode (`--mode bird`).
  if args.bird_dir is None:
    sys.stderr.write("[ERROR] `--bird-dir` is required when `--mode bird`.\n")
    return 1
  if args.limit < 0:
    sys.stderr.write("[ERROR] `--limit` must be >= 0.\n")
    return 1

  # Load official BIRD questions, schema graphs, and SQLite file paths.
  tasks, schemas, sqlite_paths = load_bird_split(
      args.bird_dir,
      questions_filename=args.questions_filename,
      tables_filename=args.tables_filename,
      databases_subdir=args.databases_subdir,
  )
  # Filter to materialized `.sqlite` files BEFORE slicing `--limit` so partial
  # downloads or stratified slices evaluate across available databases.
  tasks = tuple(t for t in tasks if t.db_id in sqlite_paths)
  if args.limit > 0:
    tasks = (
        select_stratified_bird_tasks(tasks, args.limit)
        if args.stratified
        else tasks[: args.limit]
    )
  # Generate paired spoken ASR-perturbed Voice-BIRD tasks for robustness eval.
  voice_tasks = tuple(perturb_task_for_voice(t) for t in tasks)

  # Confine sandbox execution strictly to `<resolved_bird_root>/<databases_subdir>`.
  resolved_bird = resolve_bird_root(
      args.bird_dir, tables_filename=args.tables_filename
  )
  db_root = (resolved_bird / args.databases_subdir).resolve()
  sandbox = SQLiteReadOnlySandbox(SandboxPolicy(allowed_db_root=db_root))
  # Initialize TLS-validated batch client (Vertex Gemini Enterprise or OpenAI).
  llm_client = build_live_llm_client(
      provider=args.provider,
      model_name=args.model,
      endpoint_url=args.endpoint_url,
      gcp_project=args.gcp_project,
      gcp_location=args.gcp_location,
  )
  model_label = str(getattr(llm_client, "model_name", args.model))
  provider_label = str(getattr(llm_client, "provider_name", args.provider))

  # Initialize persistent per-task JSONL checkpointer in `results/live_runs/`.
  ckpt_path = resolve_checkpoint_path(
      args.checkpoint_dir,
      split_name=args.questions_filename,
      provider=provider_label,
      model_label=model_label,
      limit=args.limit,
      stratified=args.stratified,
  )
  checkpointer = LiveRunCheckpointer(ckpt_path)
  # Apply full reset or selective system cache eviction prior to execution.
  if args.fresh_checkpoint:
    checkpointer.clear_all()
  else:
    if args.recompute_argus:
      checkpointer.invalidate_system("Argus-SQL")
    for sys_id in (s.strip() for s in args.recompute_systems.split(",") if s.strip()):
      checkpointer.invalidate_system(sys_id)

  # Optionally construct lightweight Stage 1 flash-lite intent extractor.
  intent_llm = (
      build_intent_llm_client(
          provider=args.provider,
          model_name=args.intent_model,
          endpoint_url=args.endpoint_url,
          gcp_project=args.gcp_project,
          gcp_location=args.gcp_location,
          fallback_to_none=True,
      )
      if args.intent_model.strip().lower() != "off"
      else None
  )

  # Support `--only-argus` for rapid low-token smoke checks on live endpoints.
  pipes: tuple[NL2SQLPipeline, ...] = (
      (
          ArgusSQLPipeline(
              llm_client,
              sandbox,
              ArgusAblationConfig(system_name="Argus-SQL"),
              intent_llm=intent_llm,
          ),
      )
      if args.only_argus
      else build_main_systems(llm_client, sandbox, intent_llm=intent_llm)
  )
  # Execute identical task subsets concurrently across Clean and Voice splits.
  clean_rep = run_benchmark_suite(
      "BIRD-Dev Clean",
      tasks,
      pipes,
      schemas,
      sqlite_paths,
      sandbox,
      max_workers=args.task_concurrency,
      checkpointer=checkpointer,
  )
  voice_rep = run_benchmark_suite(
      "BIRD-Dev Voice",
      voice_tasks,
      pipes,
      schemas,
      sqlite_paths,
      sandbox,
      max_workers=args.task_concurrency,
      checkpointer=checkpointer,
  )
  # Render publication-ready booktabs LaTeX table (`bird_live.tex`) and JSON.
  safe_model_tex = escape_latex_text(model_label)
  bird_tex = render_comparative_latex_table(
      clean_rep,
      voice_rep,
      caption=f"Live BIRD-SQL evaluation using {safe_model_tex}.",
      label="tab:bird_live",
  )
  # Confine output artifacts strictly within configured output directories.
  _write_confined_output(args.latex_out_dir, "bird_live.tex", bird_tex)
  _write_confined_output(
      args.json_out_dir,
      "bird_live_benchmark.json",
      serialize_reports_to_json((clean_rep, voice_rep)),
  )
  # Emit sanitized, SHA-256-signed feedback bundle for split-env round-tripping.
  bundle_file, triage_file = export_feedback_artifacts(
      out_dir=args.feedback_out_dir,
      suites=(clean_rep, voice_rep),
      provider=provider_label,
      model_label=model_label,
      split_name=args.questions_filename,
  )
  sys.stdout.write(
      f"[Feedback] Exported clean-room bundle: {bundle_file.name} & "
      f"{triage_file.name}\n"
  )
  # Print leaderboard summary rows and paired McNemar significance to stdout.
  _print_suite_summaries(clean_rep, voice_rep)
  return 0


if __name__ == "__main__":
  sys.exit(main())
