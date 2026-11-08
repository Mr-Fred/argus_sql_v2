"""Immutable evaluation result and leaderboard summary contracts."""

import dataclasses

from argus_sql.contracts.types import QueryDifficulty


# Per-question evaluation record capturing EX, R-VES, latency, and failure mode.
@dataclasses.dataclass(frozen=True)
class TaskEvalRecord:
  """Per-task benchmark outcome comparing predicted SQL against gold SQL.

  Attributes:
    question_id: Unique task identifier.
    db_id: Target database identifier.
    difficulty: Official BIRD difficulty tier ('simple', 'moderate', etc.).
    is_voice: True if evaluated on a Voice-BIRD spoken utterance task.
    ex_correct: True when predicted result set equals gold result set.
    r_ves_score: Official BIRD Reward-based Valid Efficiency Score (`0..1.25`).
    latency_ms: End-to-end pipeline wall-clock latency in milliseconds.
    total_tokens: Total prompt + completion tokens consumed.
    failure_category: Taxonomic label ('CLEAN_MATCH', 'EMPTY_RESULT', etc.).
  """

  question_id: str
  db_id: str
  difficulty: QueryDifficulty
  is_voice: bool
  ex_correct: bool
  r_ves_score: float
  latency_ms: float
  total_tokens: int
  failure_category: str
  # Sanitized diagnostic trace fields enabling offline root-cause debugging.
  predicted_sql: str = ""
  error_summary: str = ""
  invariant_codes: tuple[str, ...] = ()
  was_repaired: bool = False
  selected_strategy: str = ""


# Aggregated leaderboard row for one system or ablation variant.
@dataclasses.dataclass(frozen=True)
class SystemEvalSummary:
  """Aggregated accuracy, efficiency, latency, and failure metrics per system.

  Attributes:
    system_name: Pipeline or ablation configuration identifier.
    total_tasks: Number of tasks evaluated.
    ex_overall: Overall Execution Accuracy percentage (`0.0..100.0`).
    ex_simple: Execution Accuracy percentage on `simple` tasks.
    ex_moderate: Execution Accuracy percentage on `moderate` tasks.
    ex_challenging: Execution Accuracy percentage on `challenging` tasks.
    r_ves_overall: Mean Reward-based Valid Efficiency Score percentage.
    median_latency_ms: 50th percentile end-to-end latency in milliseconds.
    p95_latency_ms: 95th percentile end-to-end latency in milliseconds.
    mean_tokens: Mean total tokens consumed per task.
    failure_counts: Tuple of `(failure_category, count)` pairs.
  """

  # Stratified EX% isolates single-table lookups from multi-hop formulas.
  system_name: str
  total_tasks: int
  ex_overall: float
  ex_simple: float
  ex_moderate: float
  ex_challenging: float
  r_ves_overall: float
  # Latency quantiles (`p50`, `p95`) measure interactive voice readiness.
  median_latency_ms: float
  p95_latency_ms: float
  mean_tokens: float
  failure_counts: tuple[tuple[str, int], ...]


# Full comparative report across Clean and Voice splits or multiple systems.
# Stores both leaderboard aggregates and raw task traces for significance tests.
@dataclasses.dataclass(frozen=True)
class BenchmarkSuiteReport:
  """Complete evaluation report across all systems on a benchmark split.

  Attributes:
    suite_name: Name of the benchmark split (e.g., 'Mini-BIRD Clean').
    summaries: Ordered tuple of SystemEvalSummary leaderboard rows.
    records_by_system: Mapping entries `(system_name, tuple_of_records)`.
  """

  # Suite identifier rendered in exported LaTeX captions and JSON headers.
  suite_name: str
  summaries: tuple[SystemEvalSummary, ...]
  records_by_system: tuple[tuple[str, tuple[TaskEvalRecord, ...]], ...]


# Actionable root-cause failure bucket surfaced during offline feedback triage.
@dataclasses.dataclass(frozen=True)
class FailureCluster:
  """Grouped failure mode linking broken tasks to concrete module remedies.

  Attributes:
    failure_category: High-level failure bucket ('SYNTAX_OR_JOIN_KEY_ERROR', etc.).
    sub_signature: Fine-grained error signature ('MISSING_SCHEMA_TABLE', etc.).
    task_count: Number of failed tasks matching this signature.
    affected_dbs: Sorted unique database identifiers affected.
    sample_question_ids: Representative task IDs for offline unit test synthesis.
    recommended_fix: Concrete Argus-SQL module action to resolve the cluster.
  """

  failure_category: str
  sub_signature: str
  task_count: int
  affected_dbs: tuple[str, ...]
  sample_question_ids: tuple[str, ...]
  recommended_fix: str


# Paired Clean-vs-Voice discrepancy isolating spoken ASR canonicalization gaps.
@dataclasses.dataclass(frozen=True)
class VoiceDriftCase:
  """Task pair where Clean text accuracy differs from spoken Voice-BIRD accuracy.

  Attributes:
    question_id: Shared benchmark task identifier.
    db_id: Target database identifier.
    difficulty: Official BIRD difficulty tier.
    clean_ex: Execution Accuracy outcome on clean text input.
    voice_ex: Execution Accuracy outcome on spoken ASR-perturbed input.
    voice_failure_category: Failure category observed on the Voice split.
    voice_predicted_sql: Sanitized SQL emitted on the Voice split.
  """

  question_id: str
  db_id: str
  difficulty: QueryDifficulty
  clean_ex: bool
  voice_ex: bool
  voice_failure_category: str
  voice_predicted_sql: str


# Portable, path-scrubbed, SHA-256-verified feedback package for split CI/CD.
@dataclasses.dataclass(frozen=True)
class FeedbackBundle:
  """Clean-room telemetry package exchanged between personal eval and authoring.

  Attributes:
    schema_version: Wire format version identifier ('argus_feedback.v1').
    created_utc: ISO-8601 UTC creation timestamp.
    provider: LLM provider identifier ('vertex', 'openai', or 'diagnostic').
    model_label: Evaluated model slug (e.g., 'gemini-2.5-flash').
    split_name: Evaluated dataset split filename (e.g., 'mini_dev_sqlite.json').
    git_commit: Short Git commit SHA of the evaluated source tree.
    suites: Sanitized BenchmarkSuiteReport tuple (typically Clean + Voice).
    payload_sha256: Hex SHA-256 integrity digest over canonical metadata + suites.
  """

  schema_version: str
  created_utc: str
  provider: str
  model_label: str
  split_name: str
  git_commit: str
  suites: tuple[BenchmarkSuiteReport, ...]
  payload_sha256: str
