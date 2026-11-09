"""Execution-refiner (B2) and value majority-vote (B3) operator baselines.

Implements controlled, backbone-matched operator baselines isolating execution
error refinement (`B2_ExecRefiner`) and steelmanned 3-gram value-grounded
majority self-consistency voting (`B3_ValueMajorityVote`) behind `NL2SQLPipeline`.
"""

import pathlib
import time

from argus_sql.adapters.types import BatchLLMPort
from argus_sql.adapters.types import LLMCompletionRequest
from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import ExecutionOutcome
from argus_sql.contracts.types import IntentFrame
from argus_sql.contracts.types import LinkedSchemaContext
from argus_sql.contracts.types import NLQueryTask
from argus_sql.contracts.types import PipelinePrediction
from argus_sql.contracts.types import QueryDifficulty
from argus_sql.contracts.types import SQLCandidate
from argus_sql.contracts.types import VerifiedCandidate
from argus_sql.linking.fk_schema_linker import link_schema_with_fk_closure
from argus_sql.linking.intent_canonicalizer import extract_raw_entities
from argus_sql.linking.types import SchemaLinkerConfig
from argus_sql.linking.value_retriever import CellValueRetriever
from argus_sql.pipelines.prompt_templates import SQLITE_SYSTEM_PROMPT
from argus_sql.pipelines.prompt_templates import build_strategy_request
from argus_sql.sandbox.sqlite_sandbox import SQLiteReadOnlySandbox
from argus_sql.verification.consensus_arbiter import verify_and_rank_candidates
from argus_sql.verification.sql_repairer import build_diagnostic_repair_prompt
from argus_sql.verification.types import VerifierConfig


def build_uncleaned_intent(
    question: str,
    evidence: str = "",
    extract_entities: bool = False,
) -> IntentFrame:
  """Builds an un-canonicalized IntentFrame for non-voice operator baselines.

  Args:
    question: Raw user question or spoken ASR transcript.
    evidence: Optional external domain knowledge / formula note.
    extract_entities: When True, populates multi-word proper nouns, quoted
      literals, and domain codes via `extract_raw_entities` without Stage 1
      ASR filler cleanup or spoken abbreviation expansions.

  Returns:
    IntentFrame preserving raw utterance text without Stage 1 voice cleanup.
  """
  # Non-voice baselines omit ASR filler stripping and spoken number rewrites,
  # while `extract_entities=True` gives B3/B4 fair multi-word entity spans.
  raw_entities = (
      extract_raw_entities(question, evidence) if extract_entities else ()
  )
  # Leave division, extrema, and voice disfluency flags unset on raw intent.
  return IntentFrame(
      raw_question=question,
      canonical_question=question.strip(),
      disfluencies_removed=0,
      entities=raw_entities,
      requires_floating_division=False,
      requires_extrema_order=False,
      expected_projection_arity=1,
      complexity=QueryDifficulty.MODERATE,
  )


def sample_value_grounded_candidates(
    llm: BatchLLMPort,
    sandbox: SQLiteReadOnlySandbox,
    task: NLQueryTask,
    schema: DatabaseSchema,
    db_path: pathlib.Path,
    value_retriever: CellValueRetriever | None = None,
) -> tuple[
    IntentFrame,
    LinkedSchemaContext,
    tuple[SQLCandidate, ...],
    tuple[ExecutionOutcome, ...],
    int,
]:
  """Samples and executes K=3 value-grounded SQL candidates shared by B3 & B4.

  Args:
    llm: BatchLLMPort completion adapter.
    sandbox: Read-only SQLite execution sandbox.
    task: Target NLQueryTask.
    schema: Relational DatabaseSchema.
    db_path: Read-only SQLite database path.
    value_retriever: Optional pre-built 3-gram CellValueRetriever.

  Returns:
    Tuple `(raw_intent, linked_schema, candidates, outcomes, logical_tokens)`.
  """
  # Extract fair surface entities (multi-word proper nouns, quotes, codes).
  raw_intent = build_uncleaned_intent(
      task.question, task.evidence, extract_entities=True
  )
  # Reuse pre-built per-database 3-gram cell value index when supplied.
  retriever = value_retriever or CellValueRetriever.from_schema_and_sqlite(
      schema, db_path
  )
  matches = retriever.retrieve(raw_intent)
  # Combine 3-gram grounded literals with FK-closed schema pruning.
  linked = link_schema_with_fk_closure(
      schema=schema,
      intent=raw_intent,
      evidence=task.evidence,
      value_matches=matches,
      config=SchemaLinkerConfig(
          enable_fk_closure=True, enable_value_grounding=True
      ),
  )

  # Sample K=3 candidates in one batched call (2 standard, 1 defensive).
  # Identical prompt payloads let B4 reuse B3's SHA-256 cache with 0 HTTP calls.
  batch_reqs = (
      build_strategy_request(
          f"{task.question_id}-vg-1",
          linked.ddl_prompt,
          task.question,
          task.evidence,
          "decomposed_cot",
          temperature=0.2,
      ),
      build_strategy_request(
          f"{task.question_id}-vg-2",
          linked.ddl_prompt,
          task.question,
          task.evidence,
          "skeleton_minimal",
          temperature=0.4,
      ),
      build_strategy_request(
          f"{task.question_id}-vg-3",
          linked.ddl_prompt,
          task.question,
          task.evidence,
          "formula_defensive",
          temperature=0.2,
      ),
  )
  # Execute all three strategy prompts concurrently in one deduplicated batch.
  resps = llm.batch_complete(batch_reqs)
  # Use `effective_*_tokens` so standalone token budgets stay exact on cache hits.
  cands = tuple(
      SQLCandidate(
          r.sql_text,
          req.strategy_tag,
          "",
          r.effective_prompt_tokens,
          r.effective_completion_tokens,
      )
      for req, r in zip(batch_reqs, resps, strict=True)
  )
  # Execute unique SQL strings once inside the read-only SQLite sandbox.
  outcomes = sandbox.execute_batch(db_path, [c.sql for c in cands])
  logical_tokens = sum(r.logical_tokens for r in resps)
  return (raw_intent, linked, cands, outcomes, logical_tokens)


class ExecRefinerPipeline:
  """B2 controlled operator baseline: FK-closed DDL + syntax/empty Refiner."""

  def __init__(
      self,
      llm: BatchLLMPort,
      sandbox: SQLiteReadOnlySandbox,
      max_refinements: int = 1,
  ) -> None:
    """Initializes B2_ExecRefiner with bounded execution-error repair rounds.

    Args:
      llm: BatchLLMPort completion adapter.
      sandbox: Read-only SQLite execution sandbox.
      max_refinements: Maximum syntax/empty-row LLM refinement iterations.
    """
    # Bound refinement passes to prevent unbounded repair loops on hard errors.
    self._llm = llm
    self._sandbox = sandbox
    self._max_refinements = max_refinements

  @property
  def system_name(self) -> str:
    """Returns canonical controlled operator name for baseline B2."""
    # Isolates FK-closed linking + syntax/empty-result execution refinement.
    return "B2_ExecRefiner"

  def predict(
      self,
      task: NLQueryTask,
      schema: DatabaseSchema,
      db_path: pathlib.Path,
      value_retriever: CellValueRetriever | None = None,
  ) -> PipelinePrediction:
    """Runs FK-closed linking -> CoT synthesis -> syntax/empty-row Refiner.

    Args:
      task: Target NLQueryTask.
      schema: Relational DatabaseSchema.
      db_path: Read-only SQLite database path.
      value_retriever: Unused in B2 (isolates FK closure + execution refiner).

    Returns:
      PipelinePrediction after optional syntax/empty-row refinement.
    """
    # B2 isolates FK-closed schema selection + syntax/empty execution repair.
    del value_retriever
    start_ns = time.perf_counter_ns()
    raw_intent = build_uncleaned_intent(task.question, task.evidence)
    # Keep intermediate FK bridge tables between lexically matched endpoints.
    linked = link_schema_with_fk_closure(
        schema=schema,
        intent=raw_intent,
        evidence=task.evidence,
        config=SchemaLinkerConfig(
            enable_fk_closure=True, enable_value_grounding=False
        ),
    )
    # Synthesize initial SQL candidate via step-by-step decomposed CoT.
    req = build_strategy_request(
        request_id=f"{task.question_id}-b2-init",
        ddl_prompt=linked.ddl_prompt,
        question=task.question,
        evidence=task.evidence,
        strategy_tag="decomposed_cot",
    )
    resp = self._llm.batch_complete((req,))[0]
    total_tokens = resp.logical_tokens
    cand = SQLCandidate(
        resp.sql_text,
        "decomposed_cot",
        "",
        resp.effective_prompt_tokens,
        resp.effective_completion_tokens,
    )
    outcome = self._sandbox.execute(db_path, cand.sql)
    history: list[VerifiedCandidate] = [VerifiedCandidate(cand, outcome)]

    # Execution Refiner triggers ONLY on SQLite exception or 0 rows (blind to
    # silent integer-division truncation or NULL-first ascending sort bugs).
    rounds_left = self._max_refinements
    while rounds_left > 0 and (not outcome.succeeded or outcome.row_count == 0):
      rounds_left -= 1
      # Check syntax/empty-only issues (`enable_semantic_invariants=False`).
      _, ver_tuple, _ = verify_and_rank_candidates(
          (cand,),
          (outcome,),
          raw_intent,
          linked,
          VerifierConfig(enable_semantic_invariants=False),
      )
      # Build repair prompt citing only SQLite error or 0-row execution status.
      rep_prompt = build_diagnostic_repair_prompt(
          task.question,
          task.evidence,
          linked.ddl_prompt,
          cand,
          ver_tuple[0].issues,
      )
      rep_req = LLMCompletionRequest(
          request_id=f"{task.question_id}-b2-ref",
          system_prompt=SQLITE_SYSTEM_PROMPT,
          user_prompt=rep_prompt,
          strategy_tag="decomposed_cot",
      )
      rep_resp = self._llm.batch_complete((rep_req,))[0]
      total_tokens += rep_resp.logical_tokens
      # Re-execute refined SQL inside the read-only SQLite sandbox.
      cand = SQLCandidate(
          rep_resp.sql_text,
          "exec_refiner",
          "",
          rep_resp.effective_prompt_tokens,
          rep_resp.effective_completion_tokens,
      )
      outcome = self._sandbox.execute(db_path, cand.sql)
      history.append(VerifiedCandidate(cand, outcome, was_repaired=True))

    # Measure total multi-round wall-clock latency in milliseconds.
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    return PipelinePrediction(
        question_id=task.question_id,
        db_id=task.db_id,
        system_name=self.system_name,
        final_sql=cand.sql,
        latency_ms=round(elapsed_ms, 3),
        total_tokens=total_tokens,
        verified_candidates=tuple(history),
    )


class ValueMajorityVotePipeline:
  """B3 controlled operator baseline: 3-gram grounding + K=3 majority vote."""

  def __init__(
      self,
      llm: BatchLLMPort,
      sandbox: SQLiteReadOnlySandbox,
  ) -> None:
    """Initializes B3_ValueMajorityVote with steelmanned K=3 majority voting.

    Args:
      llm: BatchLLMPort completion adapter.
      sandbox: Read-only SQLite execution sandbox.
    """
    # Inject identical LLM and sandbox ports shared across all 7 pipelines.
    self._llm = llm
    self._sandbox = sandbox

  @property
  def system_name(self) -> str:
    """Returns canonical controlled operator name for baseline B3."""
    # Isolates steelmanned 3-gram value grounding + unweighted K=3 voting.
    return "B3_ValueMajorityVote"

  def predict(
      self,
      task: NLQueryTask,
      schema: DatabaseSchema,
      db_path: pathlib.Path,
      value_retriever: CellValueRetriever | None = None,
  ) -> PipelinePrediction:
    """Grounds surface entities via 3-grams and takes unweighted majority vote.

    Args:
      task: Target NLQueryTask.
      schema: Relational DatabaseSchema.
      db_path: Read-only SQLite database path.
      value_retriever: Optional pre-built 3-gram CellValueRetriever.

    Returns:
      PipelinePrediction chosen by unweighted execution self-consistency.
    """
    start_ns = time.perf_counter_ns()
    # Sample K=3 value-grounded candidates with multi-word entity spans enabled.
    raw_intent, linked, cands, outcomes, total_tokens = (
        sample_value_grounded_candidates(
            self._llm, self._sandbox, task, schema, db_path, value_retriever
        )
    )
    # Unweighted self-consistency (`enable_semantic_invariants=False`,
    # `strategy_diversity_bonus=0.0`) isolates majority voting without invariants.
    winner, all_ver, _ = verify_and_rank_candidates(
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
    # Record total wall-clock latency and standalone K=3 token expenditure.
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    return PipelinePrediction(
        question_id=task.question_id,
        db_id=task.db_id,
        system_name=self.system_name,
        final_sql=winner.candidate.sql,
        latency_ms=round(elapsed_ms, 3),
        total_tokens=total_tokens,
        verified_candidates=all_ver,
    )


# Backward-compatible class aliases preserving existing caller imports.
MACRefinerPipeline = ExecRefinerPipeline
CHESSValueConsistencyPipeline = ValueMajorityVotePipeline
