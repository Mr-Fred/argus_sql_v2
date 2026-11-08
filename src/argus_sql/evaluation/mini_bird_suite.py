"""Batch benchmark suite runner for Mini-BIRD, Voice-BIRD, and BIRD-SQL splits.

Evaluates comparative NL2SQL pipelines over shared per-database 3-gram cell
value caches and produces aggregated `BenchmarkSuiteReport` objects.
"""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import pathlib

from argus_sql.adapters.types import BatchLLMPort
from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import NLQueryTask
from argus_sql.evaluation.bird_metrics import evaluate_prediction
from argus_sql.evaluation.bird_metrics import summarize_system_records
from argus_sql.evaluation.exceptions import BenchmarkDatasetError
# Re-export task builders so callers importing from `mini_bird_suite` work seamlessly.
from argus_sql.evaluation.mini_bird_tasks import build_mini_bird_clean_tasks
from argus_sql.evaluation.mini_bird_tasks import build_voice_bird_tasks
from argus_sql.evaluation.types import BenchmarkSuiteReport
from argus_sql.evaluation.types import SystemEvalSummary
from argus_sql.evaluation.types import TaskEvalRecord
from argus_sql.linking.value_retriever import CellValueRetriever
from argus_sql.pipelines.argus_pipeline import ArgusSQLPipeline
from argus_sql.pipelines.baseline_pipelines import CHESSValueConsistencyPipeline
from argus_sql.pipelines.baseline_pipelines import DINDecomposedPipeline
from argus_sql.pipelines.baseline_pipelines import MACRefinerPipeline
from argus_sql.pipelines.baseline_pipelines import ZeroShotDDLPipeline
from argus_sql.pipelines.types import ArgusAblationConfig
from argus_sql.pipelines.types import NL2SQLPipeline
from argus_sql.sandbox.sqlite_sandbox import SQLiteReadOnlySandbox


def build_main_systems(
    llm: BatchLLMPort, sandbox: SQLiteReadOnlySandbox
) -> tuple[NL2SQLPipeline, ...]:
  """Constructs B0, B1, B2, B3, and full Argus-SQL comparative systems.

  Args:
    llm: BatchLLMPort completion adapter shared across all systems.
    sandbox: Read-only SQLite execution sandbox.

  Returns:
    Five-element tuple of progressive comparative NL2SQL pipelines.
  """
  # Instantiate all five progressive architectures over identical ports.
  return (
      ZeroShotDDLPipeline(llm, sandbox),
      DINDecomposedPipeline(llm, sandbox),
      MACRefinerPipeline(llm, sandbox),
      CHESSValueConsistencyPipeline(llm, sandbox),
      ArgusSQLPipeline(llm, sandbox, ArgusAblationConfig(system_name="Argus-SQL")),
  )


def build_ablation_systems(
    llm: BatchLLMPort, sandbox: SQLiteReadOnlySandbox
) -> tuple[NL2SQLPipeline, ...]:
  """Constructs full Argus-SQL alongside 5 leave-one-out component ablations.

  Args:
    llm: BatchLLMPort completion adapter shared across ablations.
    sandbox: Read-only SQLite execution sandbox.

  Returns:
    Six-element tuple of full Argus-SQL and leave-one-out ablation pipelines.
  """
  # Disable exactly one subsystem per row to measure marginal contribution.
  return (
      ArgusSQLPipeline(llm, sandbox, ArgusAblationConfig(system_name="Argus-SQL")),
      # Ablate Stage 1: remove spoken ASR disfluency and number normalization.
      ArgusSQLPipeline(
          llm,
          sandbox,
          ArgusAblationConfig(
              system_name="w/o Voice Canonicalizer",
              enable_voice_canonicalizer=False,
          ),
      ),
      # Ablate Stage 2a: remove character 3-gram database cell value indexing.
      ArgusSQLPipeline(
          llm,
          sandbox,
          ArgusAblationConfig(
              system_name="w/o 3-Gram Value Grounding",
              enable_value_grounding=False,
          ),
      ),
      # Ablate Stage 2b: remove Steiner shortest-path FK bridge recovery.
      ArgusSQLPipeline(
          llm,
          sandbox,
          ArgusAblationConfig(
              system_name="w/o Steiner FK Closure",
              enable_fk_closure=False,
          ),
      ),
      # Ablate Stage 4: disable SQLite dialect invariants & algebraic repair.
      ArgusSQLPipeline(
          llm,
          sandbox,
          ArgusAblationConfig(
              system_name="w/o Semantic Verifier & Repair",
              enable_semantic_verifier=False,
              enable_algebraic_repair=False,
          ),
      ),
      # Ablate Stage 3 routing: force K=3 sampling even on simple queries.
      ArgusSQLPipeline(
          llm,
          sandbox,
          ArgusAblationConfig(
              system_name="w/o Adaptive Fast-Path (Fixed K=3)",
              enable_adaptive_routing=False,
          ),
      ),
  )


def run_benchmark_suite(
    suite_name: str,
    tasks: Sequence[NLQueryTask],
    pipelines: Sequence[NL2SQLPipeline],
    schemas: dict[str, DatabaseSchema],
    sqlite_paths: dict[str, pathlib.Path],
    sandbox: SQLiteReadOnlySandbox,
    max_workers: int = 1,
) -> BenchmarkSuiteReport:
  """Evaluates all `pipelines` across `tasks` with shared 3-gram value caches.

  Args:
    suite_name: Display title of the benchmark split ('Mini-BIRD Clean', etc.).
    tasks: Sequence of NLQueryTask items to evaluate.
    pipelines: Comparative NL2SQLPipeline instances under test.
    schemas: Mapping of `db_id -> DatabaseSchema`.
    sqlite_paths: Mapping of `db_id -> sqlite_file_path`.
    sandbox: Read-only SQLite execution sandbox.
    max_workers: Concurrent task worker count (`1` for hermetic micro-benchmarks;
      `>1` for parallel network batching on live BIRD splits).

  Returns:
    BenchmarkSuiteReport containing SystemEvalSummary rows and task traces.

  Raises:
    BenchmarkDatasetError: If `tasks` is non-empty but zero tasks match a
      materialized `.sqlite` database in `sqlite_paths`.
  """
  # Retain tasks whose target `.sqlite` file is materialized on disk so partial
  # BIRD splits (e.g., single-database subsets) evaluate cleanly without KeyError.
  active_tasks = tuple(
      t for t in tasks if t.db_id in schemas and t.db_id in sqlite_paths
  )
  if tasks and not active_tasks:
    raise BenchmarkDatasetError(
        f"Suite '{suite_name}' has {len(tasks)} tasks, but 0 match materialized "
        f"SQLite databases ({sorted(sqlite_paths)})."
    )

  # Pre-index distinct cell literals once per active database (batch optimization!).
  active_db_ids = {t.db_id for t in active_tasks}
  retrievers: dict[str, CellValueRetriever] = {
      db_id: CellValueRetriever.from_schema_and_sqlite(
          schemas[db_id], sqlite_paths.get(db_id)
      )
      for db_id in active_db_ids
  }

  # Collect per-system summary rows and full task-level records for export.
  summaries: list[SystemEvalSummary] = []
  records_by_sys: list[tuple[str, tuple[TaskEvalRecord, ...]]] = []
  eff_workers = max(1, min(max_workers, 16))

  # Evaluate each pipeline over the exact same frozen task set.
  for pipeline in pipelines:

    def _eval_one_task(
        task: NLQueryTask,
        active_pipe: NL2SQLPipeline = pipeline,
    ) -> TaskEvalRecord:
      """Runs prediction and sandbox EX/R-VES evaluation for a single task."""
      db_file = sqlite_paths[task.db_id]
      pred = active_pipe.predict(
          task=task,
          schema=schemas[task.db_id],
          db_path=db_file,
          value_retriever=retrievers[task.db_id],
      )
      return evaluate_prediction(task, pred, db_file, sandbox)

    # Dispatch tasks concurrently on live network runs while preserving 1-to-1 order.
    if eff_workers > 1 and len(active_tasks) > 1:
      with ThreadPoolExecutor(max_workers=min(eff_workers, len(active_tasks))) as pool:
        task_records = list(pool.map(_eval_one_task, active_tasks))
    else:
      task_records = [_eval_one_task(t) for t in active_tasks]

    # Aggregate stratified difficulty EX%, R-VES, latency quantiles, and tokens.
    summaries.append(summarize_system_records(pipeline.system_name, task_records))
    records_by_sys.append((pipeline.system_name, tuple(task_records)))

  # Return immutable suite report ready for LaTeX and JSON serialization.
  return BenchmarkSuiteReport(
      suite_name=suite_name,
      summaries=tuple(summaries),
      records_by_system=tuple(records_by_sys),
  )


__all__ = [
    "build_ablation_systems",
    "build_main_systems",
    "build_mini_bird_clean_tasks",
    "build_voice_bird_tasks",
    "run_benchmark_suite",
]
