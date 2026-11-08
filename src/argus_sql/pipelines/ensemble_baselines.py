"""Iterative refiner (B2) and self-consistency ensemble (B3) NL2SQL baselines.

Implements multi-step execution-error refinement (`B2_MACRefiner`) and 3-gram
value-grounded majority self-consistency voting (`B3_CHESSValueConsistency`)
from public BIRD-SQL literature behind the shared `NL2SQLPipeline` protocol.
"""

import pathlib
import time

from argus_sql.adapters.types import BatchLLMPort
from argus_sql.adapters.types import LLMCompletionRequest
from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import IntentFrame
from argus_sql.contracts.types import NLQueryTask
from argus_sql.contracts.types import PipelinePrediction
from argus_sql.contracts.types import QueryDifficulty
from argus_sql.contracts.types import SQLCandidate
from argus_sql.contracts.types import VerifiedCandidate
from argus_sql.linking.fk_schema_linker import link_schema_with_fk_closure
from argus_sql.linking.types import SchemaLinkerConfig
from argus_sql.linking.value_retriever import CellValueRetriever
from argus_sql.pipelines.prompt_templates import SQLITE_SYSTEM_PROMPT
from argus_sql.pipelines.prompt_templates import build_strategy_request
from argus_sql.sandbox.sqlite_sandbox import SQLiteReadOnlySandbox
from argus_sql.verification.consensus_arbiter import verify_and_rank_candidates
from argus_sql.verification.sql_repairer import build_diagnostic_repair_prompt
from argus_sql.verification.types import VerifierConfig


def build_uncleaned_intent(question: str) -> IntentFrame:
  """Builds a minimal un-canonicalized IntentFrame for non-voice baselines.

  Args:
    question: Raw user question or spoken ASR transcript.

  Returns:
    IntentFrame preserving raw text without ASR filler or number cleanup.
  """
  # Published B1/B2/B3 pipelines pass raw user utterances without ASR cleanup.
  return IntentFrame(
      raw_question=question,
      canonical_question=question.strip(),
      disfluencies_removed=0,
      entities=(),
      requires_floating_division=False,
      requires_extrema_order=False,
      expected_projection_arity=1,
      complexity=QueryDifficulty.MODERATE,
  )


class MACRefinerPipeline:
  """B2 baseline: Selector + Decomposer + syntax/empty-result Refiner loop."""

  def __init__(
      self,
      llm: BatchLLMPort,
      sandbox: SQLiteReadOnlySandbox,
      max_refinements: int = 1,
  ) -> None:
    """Initializes B2_MACRefiner with bounded execution-error repair rounds.

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
    """Returns canonical leaderboard name for baseline B2."""
    return "B2_MACRefiner"

  def predict(
      self,
      task: NLQueryTask,
      schema: DatabaseSchema,
      db_path: pathlib.Path,
      value_retriever: CellValueRetriever | None = None,
  ) -> PipelinePrediction:
    """Runs Selector -> Decomposer -> Execution-Error Refiner (MAC-SQL style).

    Args:
      task: Target NLQueryTask.
      schema: Relational DatabaseSchema.
      db_path: Read-only SQLite database path.
      value_retriever: Unused in B2 (MAC-SQL lacks n-gram cell indexing).

    Returns:
      PipelinePrediction after optional syntax/empty-row refinement.
    """
    # MAC-SQL Selector inspects schema metadata without n-gram cell retrieval.
    del value_retriever
    start_ns = time.perf_counter_ns()
    raw_intent = build_uncleaned_intent(task.question)
    # MAC-SQL Selector keeps FK edges between selected tables, no cell index.
    linked = link_schema_with_fk_closure(
        schema=schema,
        intent=raw_intent,
        evidence=task.evidence,
        config=SchemaLinkerConfig(
            enable_fk_closure=True, enable_value_grounding=False
        ),
    )
    # Decomposer synthesizes initial SQL candidate via step-by-step CoT.
    req = build_strategy_request(
        request_id=f"{task.question_id}-b2-init",
        ddl_prompt=linked.ddl_prompt,
        question=task.question,
        evidence=task.evidence,
        strategy_tag="decomposed_cot",
    )
    resp = self._llm.batch_complete((req,))[0]
    total_tokens = resp.prompt_tokens + resp.completion_tokens
    cand = SQLCandidate(
        resp.sql_text,
        "decomposed_cot",
        "",
        resp.prompt_tokens,
        resp.completion_tokens,
    )
    outcome = self._sandbox.execute(db_path, cand.sql)
    history: list[VerifiedCandidate] = [VerifiedCandidate(cand, outcome)]

    # MAC-SQL Refiner triggers ONLY on execution error or 0 rows (blind to
    # silent integer division truncation or NULL-first ASC sort bugs!).
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
      # Format execution-error or empty-row repair prompt for MAC-SQL Refiner.
      rep_prompt = build_diagnostic_repair_prompt(
          task.question,
          task.evidence,
          linked.ddl_prompt,
          cand,
          ver_tuple[0].issues,
      )
      # Submit single-candidate refinement request through BatchLLMPort.
      rep_req = LLMCompletionRequest(
          request_id=f"{task.question_id}-b2-ref",
          system_prompt=SQLITE_SYSTEM_PROMPT,
          user_prompt=rep_prompt,
          strategy_tag="decomposed_cot",
      )
      rep_resp = self._llm.batch_complete((rep_req,))[0]
      total_tokens += rep_resp.prompt_tokens + rep_resp.completion_tokens
      # Re-execute refined SQL inside the read-only SQLite sandbox.
      cand = SQLCandidate(
          rep_resp.sql_text,
          "mac_refiner",
          "",
          rep_resp.prompt_tokens,
          rep_resp.completion_tokens,
      )
      outcome = self._sandbox.execute(db_path, cand.sql)
      # Record `was_repaired=True` in trace history for post-hoc audit.
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


class CHESSValueConsistencyPipeline:
  """B3 baseline: 3-gram value grounding + FK pruning + naive majority vote."""

  def __init__(
      self,
      llm: BatchLLMPort,
      sandbox: SQLiteReadOnlySandbox,
  ) -> None:
    """Initializes B3_CHESSValueConsistency with K=3 naive voting.

    Args:
      llm: BatchLLMPort completion adapter.
      sandbox: Read-only SQLite execution sandbox.
    """
    # Inject identical LLM and sandbox ports used by B0, B1, B2, and Argus-SQL.
    self._llm = llm
    self._sandbox = sandbox

  @property
  def system_name(self) -> str:
    """Returns canonical leaderboard name for baseline B3."""
    return "B3_CHESSValueConsistency"

  def predict(
      self,
      task: NLQueryTask,
      schema: DatabaseSchema,
      db_path: pathlib.Path,
      value_retriever: CellValueRetriever | None = None,
  ) -> PipelinePrediction:
    """Grounds cell values, prunes schema, and takes unweighted majority vote.

    Args:
      task: Target NLQueryTask.
      schema: Relational DatabaseSchema.
      db_path: Read-only SQLite database path.
      value_retriever: Optional pre-built 3-gram CellValueRetriever.

    Returns:
      PipelinePrediction chosen by naive majority execution voting.
    """
    start_ns = time.perf_counter_ns()
    # B3 lacks voice canonicalization so raw spoken entities/numbers persist.
    raw_intent = build_uncleaned_intent(task.question)
    # Reuse pre-built 3-gram value index across benchmark tasks when provided.
    retriever = value_retriever or CellValueRetriever.from_schema_and_sqlite(
        schema, db_path
    )
    matches = retriever.retrieve(raw_intent)
    # Combine 3-gram grounded literals with FK-aware column pruning (CHESS).
    linked = link_schema_with_fk_closure(
        schema=schema,
        intent=raw_intent,
        evidence=task.evidence,
        value_matches=matches,
        config=SchemaLinkerConfig(
            enable_fk_closure=True, enable_value_grounding=True
        ),
    )

    # Sample K=3 candidates in a single batched LLM call (2 standard, 1 defensive).
    batch_reqs = (
        build_strategy_request(
            f"{task.question_id}-b3-1",
            linked.ddl_prompt,
            task.question,
            task.evidence,
            "decomposed_cot",
            temperature=0.2,
        ),
        build_strategy_request(
            f"{task.question_id}-b3-2",
            linked.ddl_prompt,
            task.question,
            task.evidence,
            "skeleton_minimal",
            temperature=0.4,
        ),
        build_strategy_request(
            f"{task.question_id}-b3-3",
            linked.ddl_prompt,
            task.question,
            task.evidence,
            "formula_defensive",
            temperature=0.2,
        ),
    )
    # Dispatch all 3 candidate prompts concurrently in a single batch call.
    resps = self._llm.batch_complete(batch_reqs)
    cands = tuple(
        SQLCandidate(
            r.sql_text,
            req.strategy_tag,
            "",
            r.prompt_tokens,
            r.completion_tokens,
        )
        for req, r in zip(batch_reqs, resps, strict=True)
    )
    # Execute unique SQL strings once against the read-only SQLite sandbox.
    outcomes = self._sandbox.execute_batch(db_path, [c.sql for c in cands])

    # Naive unweighted self-consistency (`enable_semantic_invariants=False`,
    # `strategy_diversity_bonus=0.0`): 2 truncated samples outvote 1 clean SQL!
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
    # Sum prompt and completion tokens across all K=3 candidates.
    total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in resps)
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
