"""Public API facade for Argus-SQL BIRD metrics, loaders, and suites."""

# Re-export public BIRD-SQL loader and Voice-BIRD utterance perturber.
from argus_sql.evaluation.bird_loader import load_bird_split
from argus_sql.evaluation.bird_loader import perturb_task_for_voice
# Re-export official EX, piecewise R-VES, and paired McNemar evaluators.
from argus_sql.evaluation.bird_metrics import compute_mcnemar_p_value
from argus_sql.evaluation.bird_metrics import compute_piecewise_r_ves
from argus_sql.evaluation.bird_metrics import evaluate_prediction
from argus_sql.evaluation.bird_metrics import summarize_system_records
# Re-export evaluation exceptions.
from argus_sql.evaluation.exceptions import BenchmarkDatasetError
from argus_sql.evaluation.exceptions import EvaluationError
# Re-export split-environment feedback bundle and triage helpers.
from argus_sql.evaluation.feedback_bundle import build_feedback_bundle
from argus_sql.evaluation.feedback_bundle import load_feedback_bundle_file
from argus_sql.evaluation.feedback_bundle import parse_and_verify_feedback_bundle
from argus_sql.evaluation.feedback_bundle import sanitize_diagnostic_text
from argus_sql.evaluation.feedback_triage import cluster_system_failures
from argus_sql.evaluation.feedback_triage import export_feedback_artifacts
from argus_sql.evaluation.feedback_triage import find_clean_and_voice_suites
from argus_sql.evaluation.feedback_triage import render_feedback_triage_markdown
# Re-export LaTeX booktabs and JSON report serializers.
from argus_sql.evaluation.latex_exporter import escape_latex_text
from argus_sql.evaluation.latex_exporter import render_comparative_latex_table
from argus_sql.evaluation.latex_exporter import serialize_reports_to_json
# Re-export persistent JSONL checkpointing and stratified multi-DB sampling.
from argus_sql.evaluation.live_checkpoint import LiveRunCheckpointer
from argus_sql.evaluation.live_checkpoint import resolve_checkpoint_path
from argus_sql.evaluation.live_checkpoint import select_stratified_bird_tasks
# Re-export diagnostic SQLite database builders and benchmark suite runner.
from argus_sql.evaluation.mini_bird_databases import materialize_mini_bird_databases
from argus_sql.evaluation.mini_bird_suite import build_mini_bird_clean_tasks
from argus_sql.evaluation.mini_bird_suite import build_voice_bird_tasks
from argus_sql.evaluation.mini_bird_suite import run_benchmark_suite
# Re-export evaluation summary and feedback contracts.
from argus_sql.evaluation.types import BenchmarkSuiteReport
from argus_sql.evaluation.types import FailureCluster
from argus_sql.evaluation.types import FeedbackBundle
from argus_sql.evaluation.types import SystemEvalSummary
from argus_sql.evaluation.types import TaskEvalRecord
from argus_sql.evaluation.types import VoiceDriftCase

# Explicit symbol export list.
__all__ = [
    "BenchmarkDatasetError",
    "BenchmarkSuiteReport",
    "EvaluationError",
    "FailureCluster",
    "FeedbackBundle",
    "LiveRunCheckpointer",
    "SystemEvalSummary",
    "TaskEvalRecord",
    "VoiceDriftCase",
    "build_feedback_bundle",
    "build_mini_bird_clean_tasks",
    "build_voice_bird_tasks",
    "cluster_system_failures",
    "compute_mcnemar_p_value",
    "compute_piecewise_r_ves",
    "escape_latex_text",
    "evaluate_prediction",
    "export_feedback_artifacts",
    "find_clean_and_voice_suites",
    "load_bird_split",
    "load_feedback_bundle_file",
    "materialize_mini_bird_databases",
    "parse_and_verify_feedback_bundle",
    "perturb_task_for_voice",
    "render_comparative_latex_table",
    "render_feedback_triage_markdown",
    "resolve_checkpoint_path",
    "run_benchmark_suite",
    "sanitize_diagnostic_text",
    "select_stratified_bird_tasks",
    "serialize_reports_to_json",
    "summarize_system_records",
]
