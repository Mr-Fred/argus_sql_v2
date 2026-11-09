"""LLM Judge candidate-selection baseline (`B4_LLMJudge`: MCS-SQL/CHESS style).

Evaluates whether invoking an extra LLM pass as a Multiple-Choice / Unit-Test
Judge over disagreeing SQL execution previews matches or trails Argus-SQL's
sub-millisecond, zero-token 12-Invariant Verifier and Algebraic Repair stage.
"""

from collections.abc import Sequence
import pathlib
import re
import time

from argus_sql.adapters.types import BatchLLMPort
from argus_sql.adapters.types import LLMCompletionRequest
from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import ExecutionOutcome
from argus_sql.contracts.types import NLQueryTask
from argus_sql.contracts.types import PipelinePrediction
from argus_sql.contracts.types import VerifiedCandidate
from argus_sql.linking.value_retriever import CellValueRetriever
from argus_sql.pipelines.ensemble_baselines import sample_value_grounded_candidates
from argus_sql.pipelines.exceptions import PipelineExecutionError
from argus_sql.sandbox.sqlite_sandbox import SQLiteReadOnlySandbox
from argus_sql.verification.consensus_arbiter import verify_and_rank_candidates
from argus_sql.verification.types import VerifierConfig

# System instruction guiding MCS-SQL / CHESS style execution-preview selection.
LLM_JUDGE_SYSTEM_PROMPT: str = (
    "You are an expert SQLite SQL evaluation judge (multiple-choice & unit-test "
    "style). Given a relational schema, user question, optional external "
    "evidence, and competing SQL candidates with their actual SQLite execution "
    "previews, select the single candidate that most accurately answers the "
    "question. Check for SQLite integer-division truncation to 0 on rates/"
    "ratios, unfiltered NULL rows on ascending extrema, join grain accuracy, "
    "date/timestamp predicates, and requested output columns. "
    "Conclude on the final line with strictly: Selected: Option <LETTER>"
)

# Regex matching explicit `Selected: Option <LETTER>` lines in judge completions.
_SELECTED_OPTION_RE = re.compile(
    r"Selected\s*:\s*Option\s+([A-Z])\b", re.IGNORECASE
)

# Fallback regex matching standalone `Option <LETTER>` tokens if prefix omitted.
_FALLBACK_OPTION_RE = re.compile(r"\bOption\s+([A-Z])\b", re.IGNORECASE)

# Maximum competing clusters rendered into a judge prompt to bound token size.
_MAX_JUDGE_OPTIONS: int = 6


def format_outcome_preview(
    outcome: ExecutionOutcome,
    max_rows: int = 5,
    max_cols: int = 8,
    max_cell_chars: int = 60,
) -> tuple[str, str]:
  """Formats bounded, single-line sanitized column and row previews for judging.

  Args:
    outcome: Read-only SQLite ExecutionOutcome for a candidate cluster.
    max_rows: Maximum sample result rows serialized into the prompt.
    max_cols: Maximum columns per row serialized into the prompt.
    max_cell_chars: Maximum characters retained per cell `repr(...)`.

  Returns:
    Pair `(columns_preview_str, sample_rows_preview_str)`.
  """
  # Sanitize newlines/tabs inside headers and cells to block prompt spoofing.
  clean_cols = [
      re.sub(r"[\r\n\t]+", " ", str(col))[:max_cell_chars]
      for col in outcome.column_names[:max_cols]
  ]
  cols_str = repr(clean_cols)

  # Bound preview row count and cell length so wide text columns cannot bloat
  # Judge prompts or inject multi-line fake `Option` delimiters.
  preview_rows: list[str] = []
  for row in outcome.rows[:max_rows]:
    cells: list[str] = []
    for cell in row[:max_cols]:
      clean_cell = re.sub(r"[\r\n\t]+", " ", cell) if isinstance(cell, str) else cell
      flat = re.sub(r"[\r\n\t]+", " ", repr(clean_cell))
      if len(flat) > max_cell_chars:
        flat = flat[: max_cell_chars - 3] + "..."
      cells.append(flat)
    # Preserve valid single-element tuple trailing comma (`(0,)`) for clarity.
    if len(cells) == 1:
      preview_rows.append(f"({cells[0]},)")
    else:
      preview_rows.append(f"({', '.join(cells)})")

  # Serialize preview tuples as a compact single-line list representation.
  rows_str = f"[{', '.join(preview_rows)}]"
  return (cols_str, rows_str)


def build_judge_user_prompt(
    ddl_prompt: str,
    question: str,
    evidence: str,
    cluster_reps: Sequence[tuple[str, VerifiedCandidate, int]],
    total_candidates: int,
) -> str:
  """Constructs a structured Multiple-Choice SQL Judge prompt over clusters.

  Args:
    ddl_prompt: Linked SQLite `CREATE TABLE` DDL string.
    question: User natural language question or raw ASR utterance.
    evidence: Optional external domain formula/evidence string.
    cluster_reps: Sequence of `(option_label, representative, cluster_votes)`.
    total_candidates: Total number of initial SQL candidates sampled (`K`).

  Returns:
    Formatted prompt string presenting SQL text and execution previews.
  """
  # Align `Database Schema:`, `Question:`, and `External Evidence:` headers
  # with `build_strategy_request` so prompt parsers behave uniformly.
  flat_q = re.sub(r"[\r\n]+", " ", question).strip()
  flat_ev = re.sub(r"[\r\n]+", " ", evidence).strip()
  ev_line = f"External Evidence: {flat_ev}" if flat_ev else "External Evidence: None"
  blocks: list[str] = [
      f"Database Schema:\n{ddl_prompt}",
      "",
      f"Question: {flat_q}",
      ev_line,
      "",
      "Candidate SQL Options & Execution Previews:",
  ]
  # Render each distinct result-equivalence cluster once with support count.
  for label, rep, votes in cluster_reps:
    cols_str, rows_str = format_outcome_preview(rep.outcome)
    flat_sql = re.sub(r"[\r\n]+", " ", rep.candidate.sql).strip()
    blocks.append(
        f"Option {label} (support: {votes}/{total_candidates} candidates):\n"
        f"SQL: {flat_sql}\n"
        f"Columns: {cols_str} | Row Count: {rep.outcome.row_count}\n"
        f"Sample Rows: {rows_str}"
    )
    blocks.append("")

  # Enumerate allowed option labels explicitly to constrain Judge decoding.
  valid_str = ", ".join(f"Option {lbl}" for lbl, _, _ in cluster_reps)
  blocks.append(
      f"Instruction: Select the single most accurate candidate among "
      f"({valid_str}) and end with `Selected: Option <LETTER>`."
  )
  return "\n".join(blocks)


def parse_judge_choice(raw_text: str, valid_labels: Sequence[str]) -> str:
  """Extracts a validated option letter from Judge output with majority fallback.

  Args:
    raw_text: Raw completion string returned by the Judge LLM call.
    valid_labels: Ordered sequence of allowed option letters (`'A'`, `'B'`, ...).

  Returns:
    Validated uppercase option letter; defaults to `valid_labels[0]` on failure.

  Raises:
    PipelineExecutionError: If `valid_labels` is empty.
  """
  if not valid_labels:
    raise PipelineExecutionError("Judge valid_labels must be non-empty.")
  allowed = set(valid_labels)
  # Prefer the final explicit `Selected: Option <X>` declaration in output.
  explicit_hits = [
      m.group(1).upper()
      for m in _SELECTED_OPTION_RE.finditer(raw_text)
      if m.group(1).upper() in allowed
  ]
  if explicit_hits:
    return explicit_hits[-1]

  # Fall back to the final `Option <X>` mention if `Selected:` was omitted.
  fallback_hits = [
      m.group(1).upper()
      for m in _FALLBACK_OPTION_RE.finditer(raw_text)
      if m.group(1).upper() in allowed
  ]
  if fallback_hits:
    return fallback_hits[-1]

  # Fail safe to the highest-support execution cluster (`Option A`).
  return valid_labels[0]


class LLMJudgePipeline:
  """B4 controlled operator baseline: K=3 value-grounded pool + LLM Judge."""

  def __init__(
      self,
      llm: BatchLLMPort,
      sandbox: SQLiteReadOnlySandbox,
  ) -> None:
    """Initializes B4_LLMJudge with shared LLM and read-only SQLite ports.

    Args:
      llm: BatchLLMPort completion adapter.
      sandbox: Read-only SQLite execution sandbox.
    """
    # Store injected Hexagonal ports for controlled operator evaluation.
    self._llm = llm
    self._sandbox = sandbox

  @property
  def system_name(self) -> str:
    """Returns canonical controlled operator name for baseline B4."""
    return "B4_LLMJudge"

  def predict(
      self,
      task: NLQueryTask,
      schema: DatabaseSchema,
      db_path: pathlib.Path,
      value_retriever: CellValueRetriever | None = None,
  ) -> PipelinePrediction:
    """Samples K=3 value-grounded SQLs and invokes an LLM Judge on disagreement.

    Args:
      task: Target NLQueryTask.
      schema: Relational DatabaseSchema.
      db_path: Read-only SQLite database path.
      value_retriever: Optional pre-built 3-gram CellValueRetriever.

    Returns:
      PipelinePrediction selected via agreement short-circuit or LLM Judge.
    """
    start_ns = time.perf_counter_ns()
    # Sample K=3 value-grounded candidates (served from cache if B3 just ran).
    raw_intent, linked, cands, outcomes, total_tokens = (
        sample_value_grounded_candidates(
            self._llm, self._sandbox, task, schema, db_path, value_retriever
        )
    )
    # Establish unweighted execution-equivalence baseline ranking first.
    default_winner, all_ver, _ = verify_and_rank_candidates(
        candidates=cands,
        outcomes=outcomes,
        intent=raw_intent,
        linked_schema=linked,
        config=VerifierConfig(
            enable_semantic_invariants=False,
            enable_algebraic_repair=False,
            strategy_diversity_bonus=0.0,
        ),
    )

    # Group succeeded candidates by normalized execution `result_signature`.
    clusters_by_sig: dict[str, list[VerifiedCandidate]] = {}
    for ver in all_ver:
      if ver.outcome.succeeded and ver.outcome.result_signature:
        clusters_by_sig.setdefault(ver.outcome.result_signature, []).append(ver)

    # Short-circuit with 0 extra LLM calls when all succeeded SQLs agree!
    if len(clusters_by_sig) <= 1:
      elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
      return PipelinePrediction(
          question_id=task.question_id,
          db_id=task.db_id,
          system_name=self.system_name,
          final_sql=default_winner.candidate.sql,
          latency_ms=round(elapsed_ms, 3),
          total_tokens=total_tokens,
          verified_candidates=all_ver,
          metadata=(
              ("judge_invoked", "False"),
              ("distinct_clusters", str(len(clusters_by_sig))),
          ),
      )

    # Order competing clusters by support descending (breaking ties by fastest
    # sandbox execution time, matching `B3_ValueMajorityVote` for R-VES parity).
    ordered_groups = sorted(
        clusters_by_sig.values(),
        key=lambda g: (len(g), -min(v.outcome.execution_time_ms for v in g)),
        reverse=True,
    )[:_MAX_JUDGE_OPTIONS]
    cluster_reps: list[tuple[str, VerifiedCandidate, int]] = []
    rep_by_label: dict[str, VerifiedCandidate] = {}
    for idx, group in enumerate(ordered_groups):
      label = chr(ord("A") + idx)
      best_rep = min(group, key=lambda v: v.outcome.execution_time_ms)
      cluster_reps.append((label, best_rep, len(group)))
      rep_by_label[label] = best_rep

    # Dispatch a single deterministic Judge completion over bounded previews.
    judge_prompt = build_judge_user_prompt(
        ddl_prompt=linked.ddl_prompt,
        question=task.question,
        evidence=task.evidence,
        cluster_reps=cluster_reps,
        total_candidates=len(cands),
    )
    judge_req = LLMCompletionRequest(
        request_id=f"{task.question_id}-b4-judge",
        system_prompt=LLM_JUDGE_SYSTEM_PROMPT,
        user_prompt=judge_prompt,
        strategy_tag="llm_judge",
        temperature=0.0,
        max_tokens=512,
    )
    judge_resp = self._llm.batch_complete((judge_req,))[0]
    total_tokens += judge_resp.logical_tokens

    # Map Judge's letter strictly back to an already sandbox-verified candidate
    # so raw Judge output is never executed directly against SQLite.
    valid_labels = tuple(rep_by_label.keys())
    chosen_label = parse_judge_choice(judge_resp.raw_text, valid_labels)
    chosen = rep_by_label[chosen_label]
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    return PipelinePrediction(
        question_id=task.question_id,
        db_id=task.db_id,
        system_name=self.system_name,
        final_sql=chosen.candidate.sql,
        latency_ms=round(elapsed_ms, 3),
        total_tokens=total_tokens,
        verified_candidates=all_ver,
        metadata=(
            ("judge_invoked", "True"),
            ("distinct_clusters", str(len(clusters_by_sig))),
            ("selected_option", chosen_label),
        ),
    )
