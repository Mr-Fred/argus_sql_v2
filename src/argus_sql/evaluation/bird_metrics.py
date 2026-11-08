"""Official BIRD-SQL Execution Accuracy (EX), R-VES, and McNemar tests.

Implements exact set/ordered result matching, piecewise Reward-based Valid
Efficiency Score (R-VES, Li et al., BIRD Mini-Dev 2024), exact paired McNemar
statistical significance testing, and automated failure classification.
"""

from collections.abc import Sequence
import math
import pathlib
import statistics

from argus_sql.contracts.types import NLQueryTask
from argus_sql.contracts.types import PipelinePrediction
from argus_sql.contracts.types import QueryDifficulty
from argus_sql.evaluation.exceptions import EvaluationError
from argus_sql.evaluation.feedback_bundle import sanitize_diagnostic_text
from argus_sql.evaluation.types import SystemEvalSummary
from argus_sql.evaluation.types import TaskEvalRecord
from argus_sql.sandbox.sqlite_sandbox import SQLiteReadOnlySandbox
from argus_sql.sandbox.sqlite_sandbox import compute_result_signature
from argus_sql.sandbox.sqlite_sandbox import normalize_sql_for_dedup

# Syscall timer jitter deadband (1.0 ms) preventing sub-millisecond sqlite3
# connection/context-switch noise on tiny tables from flipping R-VES steps.
_SYSCALL_JITTER_FLOOR_MS = 1.0


def compute_piecewise_r_ves(
    ex_correct: bool,
    gold_time_ms: float,
    pred_time_ms: float,
) -> float:
  """Computes the official BIRD Mini-Dev piecewise R-VES reward in `[0, 1.25]`.

  Args:
    ex_correct: Whether the predicted SQL matched the gold SQL result set.
    gold_time_ms: Median warm execution time of gold SQL in milliseconds.
    pred_time_ms: Median warm execution time of predicted SQL in milliseconds.

  Returns:
    Piecewise reward value (`0.0`, `0.25`, `0.50`, `0.75`, `1.00`, or `1.25`).
  """
  # Invalid predictions always receive zero efficiency reward.
  if not ex_correct:
    return 0.0

  # Clamp times strictly positive to guard against division by zero.
  safe_pred = max(pred_time_ms, 0.001)
  safe_gold = max(gold_time_ms, 0.001)
  time_ratio = safe_gold / safe_pred

  # Piecewise step function preventing single-query outlier ratio distortion.
  if time_ratio >= 2.0:
    return 1.25
  if time_ratio >= 1.0:
    return 1.00
  if time_ratio >= 0.5:
    return 0.75
  if time_ratio >= 0.25:
    return 0.50
  return 0.25


def compute_mcnemar_p_value(
    baseline_records: Sequence[TaskEvalRecord],
    candidate_records: Sequence[TaskEvalRecord],
) -> float:
  """Computes the exact two-sided paired McNemar binomial p-value on EX.

  Args:
    baseline_records: Per-task records from a reference baseline system.
    candidate_records: Aligned per-task records from a candidate system.

  Returns:
    Exact two-sided p-value in `[0.0, 1.0]` testing paired EX equivalence.

  Raises:
    EvaluationError: If record sequence lengths or question IDs mismatch.
  """
  if len(baseline_records) != len(candidate_records):
    raise EvaluationError("Paired McNemar test requires equal-length records.")

  # Count discordant task pairs `(b: base_only_ok, c: cand_only_ok)`.
  base_only = 0
  cand_only = 0
  for b_rec, c_rec in zip(baseline_records, candidate_records, strict=True):
    if b_rec.question_id != c_rec.question_id:
      raise EvaluationError(
          f"Mismatched question_id: {b_rec.question_id} vs {c_rec.question_id}"
      )
    if b_rec.ex_correct and not c_rec.ex_correct:
      base_only += 1
    elif not b_rec.ex_correct and c_rec.ex_correct:
      cand_only += 1

  discordant = base_only + cand_only
  if discordant == 0:
    return 1.0

  # Sum exact lower-tail binomial probabilities under H0: p = 0.5.
  k_min = min(base_only, cand_only)
  tail_prob = sum(
      math.comb(discordant, k) for k in range(k_min + 1)
  ) * (0.5**discordant)
  return round(min(1.0, 2.0 * tail_prob), 12)


def evaluate_prediction(
    task: NLQueryTask,
    prediction: PipelinePrediction,
    db_path: pathlib.Path,
    sandbox: SQLiteReadOnlySandbox,
    timing_repeats: int = 5,
) -> TaskEvalRecord:
  """Evaluates one PipelinePrediction against `task.gold_sql` in the sandbox.

  Args:
    task: Ground-truth NLQueryTask containing `gold_sql` and `difficulty`.
    prediction: Candidate PipelinePrediction emitted by an NL2SQL system.
    db_path: Path to read-only SQLite database file.
    sandbox: Read-only SQLite execution sandbox.
    timing_repeats: Interleaved warm timing runs used for median R-VES ratios.

  Returns:
    TaskEvalRecord recording EX, R-VES, latency, tokens, and failure class.
  """
  # Execute both gold reference SQL and predicted candidate SQL inside sandbox.
  gold_out = sandbox.execute(db_path, task.gold_sql)
  pred_out = sandbox.execute(db_path, prediction.final_sql)

  # Compare result sets preserving row order only when gold SQL uses ORDER BY.
  order_sensitive = "ORDER BY" in task.gold_sql.upper()
  gold_sig = compute_result_signature(
      gold_out.rows, preserve_row_order=order_sensitive
  )
  # Failed executions never produce a matching result-set digest.
  pred_sig = (
      compute_result_signature(pred_out.rows, preserve_row_order=order_sensitive)
      if pred_out.succeeded
      else ""
  )
  ex_ok = bool(gold_out.succeeded and pred_out.succeeded and pred_sig == gold_sig)

  # Categorize failure mode for diagnostic ablation reporting.
  if ex_ok:
    failure_cat = "CLEAN_MATCH"
  elif not pred_out.succeeded:
    failure_cat = "SYNTAX_OR_JOIN_KEY_ERROR"
  elif pred_out.row_count == 0 and gold_out.row_count > 0:
    failure_cat = "EMPTY_RESULT_VALUE_DRIFT"
  else:
    failure_cat = "SILENT_SEMANTIC_OR_FORMULA_ERROR"

  # Compute deterministic R-VES using median interleaved warm timings.
  if not ex_ok:
    r_ves = 0.0
  elif (
      normalize_sql_for_dedup(prediction.final_sql).lower()
      == normalize_sql_for_dedup(task.gold_sql).lower()
  ):
    # Identical normalized SQL statements have a true speed ratio of 1.0.
    r_ves = 1.00
  elif timing_repeats > 1:
    g_times: list[float] = []
    p_times: list[float] = []
    for _ in range(timing_repeats):
      g_times.append(sandbox.execute(db_path, task.gold_sql).execution_time_ms)
      p_times.append(
          sandbox.execute(db_path, prediction.final_sql).execution_time_ms
      )
    g_med = statistics.median(g_times)
    p_med = statistics.median(p_times)
    # Treat sub-50us differences on micro-tables as parity rather than OS noise.
    if abs(g_med - p_med) <= _SYSCALL_JITTER_FLOOR_MS:
      r_ves = 1.00
    else:
      r_ves = compute_piecewise_r_ves(True, g_med, p_med)
  else:
    r_ves = compute_piecewise_r_ves(
        True, gold_out.execution_time_ms, pred_out.execution_time_ms
    )

  # Extract candidate provenance and triggered invariants for offline triage.
  inv_codes = sorted({
      issue.code
      for vc in prediction.verified_candidates
      for issue in vc.issues
  })
  # Track whether deterministic algebraic or LLM repair modified any candidate.
  repaired_any = any(vc.was_repaired for vc in prediction.verified_candidates)
  winning_strategy = ""
  # Match normalized final SQL back to the originating prompt strategy label.
  norm_final = normalize_sql_for_dedup(prediction.final_sql)
  for vc in prediction.verified_candidates:
    if normalize_sql_for_dedup(vc.candidate.sql) == norm_final:
      winning_strategy = vc.candidate.strategy
      break
  # Fall back to primary candidate strategy when normalization alters whitespace.
  if not winning_strategy and prediction.verified_candidates:
    winning_strategy = prediction.verified_candidates[0].candidate.strategy

  return TaskEvalRecord(
      question_id=task.question_id,
      db_id=task.db_id,
      difficulty=task.difficulty,
      is_voice=task.is_voice_perturbed,
      ex_correct=ex_ok,
      r_ves_score=r_ves,
      latency_ms=prediction.latency_ms,
      total_tokens=prediction.total_tokens,
      failure_category=failure_cat,
      predicted_sql=sanitize_diagnostic_text(prediction.final_sql, 2000),
      error_summary=sanitize_diagnostic_text(
          pred_out.error_message, 240, single_line=True
      ),
      invariant_codes=tuple(inv_codes),
      was_repaired=repaired_any,
      selected_strategy=sanitize_diagnostic_text(
          winning_strategy, 80, single_line=True
      ),
  )


def _tier_ex_pct(
    records: Sequence[TaskEvalRecord],
    tier: QueryDifficulty | None = None,
) -> float:
  """Computes Execution Accuracy percentage on `records` (optionally by tier)."""
  # Filter task records to the requested BIRD difficulty stratum when provided.
  subset = [r for r in records if tier is None or r.difficulty == tier]
  if not subset:
    return 0.0
  hits = sum(1 for r in subset if r.ex_correct)
  return round(100.0 * hits / len(subset), 2)


def summarize_system_records(
    system_name: str,
    records: Sequence[TaskEvalRecord],
) -> SystemEvalSummary:
  """Aggregates per-task TaskEvalRecords into a leaderboard SystemEvalSummary.

  Args:
    system_name: Pipeline or ablation identifier.
    records: Sequence of evaluated TaskEvalRecord objects.

  Returns:
    SystemEvalSummary with stratified EX%, R-VES%, latency quantiles, and costs.
  """
  # Return zeroed summary safely when evaluating an empty slice (`--limit 0`).
  if not records:
    return SystemEvalSummary(
        system_name=system_name,
        total_tasks=0,
        ex_overall=0.0,
        ex_simple=0.0,
        ex_moderate=0.0,
        ex_challenging=0.0,
        r_ves_overall=0.0,
        median_latency_ms=0.0,
        p95_latency_ms=0.0,
        mean_tokens=0.0,
        failure_counts=(),
    )

  # Compute p50 median and p95 tail latencies across all evaluated tasks.
  latencies = sorted(r.latency_ms for r in records)
  p50 = statistics.median(latencies)
  p95_idx = min(len(latencies) - 1, int(0.95 * len(latencies)))
  p95 = latencies[p95_idx]

  # Scale R-VES to 0..100 scale matching official BIRD leaderboard convention.
  mean_r_ves = 100.0 * statistics.fmean(r.r_ves_score for r in records)
  mean_toks = statistics.fmean(r.total_tokens for r in records)

  # Tally root-cause failure categories to expose systematic error shifts.
  counts: dict[str, int] = {}
  for rec in records:
    counts[rec.failure_category] = counts.get(rec.failure_category, 0) + 1

  return SystemEvalSummary(
      system_name=system_name,
      total_tasks=len(records),
      ex_overall=_tier_ex_pct(records, None),
      ex_simple=_tier_ex_pct(records, QueryDifficulty.SIMPLE),
      ex_moderate=_tier_ex_pct(records, QueryDifficulty.MODERATE),
      ex_challenging=_tier_ex_pct(records, QueryDifficulty.CHALLENGING),
      r_ves_overall=round(mean_r_ves, 2),
      median_latency_ms=round(p50, 3),
      p95_latency_ms=round(p95, 3),
      mean_tokens=round(mean_toks, 1),
      failure_counts=tuple(sorted(counts.items())),
  )
