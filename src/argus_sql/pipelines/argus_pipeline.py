"""Argus-SQL hybrid sweet-spot pipeline with full component ablation toggles.

Unifies four complementary mechanisms into a single low-latency architecture:
1. Voice & Intent Canonicalization (`canonicalize_intent`)
2. Character 3-Gram Value Grounding + Steiner FK-Closure (`link_schema_with_fk_closure`)
3. Complexity-Gated Batch Candidate Synthesis (Fast-Path K=1 vs Diverse K=3)
4. Semantic Invariant Verification, Zero-Token Algebraic Repair & Consensus
"""

import pathlib
import time

from argus_sql.adapters.types import BatchLLMPort
from argus_sql.adapters.types import LLMCompletionRequest
from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import ExecutionOutcome
from argus_sql.contracts.types import IntentFrame
from argus_sql.contracts.types import NLQueryTask
from argus_sql.contracts.types import PipelinePrediction
from argus_sql.contracts.types import QueryDifficulty
from argus_sql.contracts.types import SQLCandidate
from argus_sql.linking.fk_schema_linker import link_schema_with_fk_closure
from argus_sql.linking.llm_intent_extractor import canonicalize_intent_with_optional_llm
from argus_sql.linking.types import IntentExtractorPort
from argus_sql.linking.types import SchemaLinkerConfig
from argus_sql.linking.value_retriever import CellValueRetriever
from argus_sql.pipelines.prompt_templates import SQLITE_SYSTEM_PROMPT
from argus_sql.pipelines.prompt_templates import build_strategy_request
from argus_sql.pipelines.types import ArgusAblationConfig
from argus_sql.sandbox.sqlite_sandbox import SQLiteReadOnlySandbox
from argus_sql.sandbox.sqlite_sandbox import normalize_sql_for_dedup
from argus_sql.verification.consensus_arbiter import verify_and_rank_candidates
from argus_sql.verification.semantic_verifier import audit_candidate_invariants
from argus_sql.verification.sql_repairer import apply_algebraic_sql_repairs
from argus_sql.verification.sql_repairer import build_diagnostic_repair_prompt
from argus_sql.verification.types import VerifierConfig


def _is_empty_or_all_null(outcome: ExecutionOutcome) -> bool:
  """Returns True when `outcome` returned 0 rows or a single all-NULL row."""
  return outcome.row_count == 0 or (
      outcome.row_count == 1
      and bool(outcome.rows)
      and all(cell is None for cell in outcome.rows[0])
  )


def _merge_algebraic_repair(
    candidates: list[SQLCandidate],
    outcomes: list[ExecutionOutcome],
    repaired_flags: list[bool],
    orig_idx: int,
    rep_cand: SQLCandidate,
    rep_out: ExecutionOutcome,
) -> None:
  """Merges a succeeded algebraic rewrite without double-counting support."""
  # Ignore failed rewrites or whitespace-only no-ops matching parent SQL.
  if not rep_out.succeeded:
    return
  orig_cand = candidates[orig_idx]
  orig_out = outcomes[orig_idx]
  if normalize_sql_for_dedup(rep_cand.sql) == normalize_sql_for_dedup(
      orig_cand.sql
  ):
    return

  orig_empty_or_null = _is_empty_or_all_null(orig_out)
  rep_empty_or_null = _is_empty_or_all_null(rep_out)
  # Discard algebraic rewrites (e.g., `AND col IS NOT NULL`) that wipe out a
  # valid non-empty parent result into 0 rows or an all-NULL scalar row.
  if orig_out.succeeded and not orig_empty_or_null and rep_empty_or_null:
    return

  same_output = (
      orig_out.succeeded
      and bool(rep_out.result_signature)
      and rep_out.result_signature == orig_out.result_signature
  )
  both_empty_or_null = (
      orig_out.succeeded and orig_empty_or_null and rep_empty_or_null
  )
  # Upgrade in-place when execution output matches parent (or both stay empty)
  # so dialect cleanup never inflates cluster vote count!
  if orig_out.succeeded and (same_output or both_empty_or_null):
    candidates[orig_idx] = rep_cand
    outcomes[orig_idx] = rep_out
    repaired_flags[orig_idx] = True
  else:
    candidates.append(rep_cand)
    outcomes.append(rep_out)
    repaired_flags.append(True)


class ArgusSQLPipeline:
  """Proposed Argus-SQL hybrid pipeline balancing accuracy, cost, and latency."""

  def __init__(
      self,
      llm: BatchLLMPort,
      sandbox: SQLiteReadOnlySandbox,
      config: ArgusAblationConfig | None = None,
      intent_llm: BatchLLMPort | IntentExtractorPort | None = None,
  ) -> None:
    """Initializes Argus-SQL with swappable adapters and ablation flags.

    Args:
      llm: BatchLLMPort completion adapter.
      sandbox: Read-only SQLite execution sandbox.
      config: Optional ArgusAblationConfig for leave-one-out ablation sweeps.
      intent_llm: Optional fast `flash-lite` Stage 1 intent extractor port.
    """
    self._llm = llm
    self._sandbox = sandbox
    self._config = config or ArgusAblationConfig()
    self._intent_llm = intent_llm

  @property
  def system_name(self) -> str:
    """Returns configured system or ablation label."""
    return self._config.system_name

  def _build_intent_with_tokens(self, task: NLQueryTask) -> tuple[IntentFrame, int]:
    """Canonicalizes intent via optional flash-lite port or deterministic FSM."""
    if self._config.enable_voice_canonicalizer:
      extractor = (
          self._intent_llm
          if (self._intent_llm is not None or self._config.enable_llm_intent_extractor)
          else None
      )
      return canonicalize_intent_with_optional_llm(
          task.question, task.evidence, extractor=extractor
      )
    return (
        IntentFrame(
            raw_question=task.question,
            canonical_question=task.question.strip(),
            disfluencies_removed=0,
            entities=(),
            requires_floating_division=("/" in task.evidence),
            requires_extrema_order=False,
            expected_projection_arity=1,
            complexity=QueryDifficulty.MODERATE,
        ),
        0,
    )

  def _build_intent(self, task: NLQueryTask) -> IntentFrame:
    """Canonicalizes spoken/written intent when enabled, else wraps raw text."""
    intent, _ = self._build_intent_with_tokens(task)
    return intent

  def predict(
      self,
      task: NLQueryTask,
      schema: DatabaseSchema,
      db_path: pathlib.Path,
      value_retriever: CellValueRetriever | None = None,
  ) -> PipelinePrediction:
    """Executes the 4-stage Argus-SQL synthesis and verification pipeline.

    Args:
      task: Target NLQueryTask (clean text or Voice-BIRD utterance).
      schema: Relational DatabaseSchema.
      db_path: Read-only SQLite database path.
      value_retriever: Optional pre-built 3-gram CellValueRetriever.

    Returns:
      PipelinePrediction with winning SQL, telemetry, and ablation metadata.
    """
    start_ns = time.perf_counter_ns()
    cfg = self._config

    # Stage 1: Voice & Intent Canonicalization (FSM + optional flash-lite pass).
    intent, intent_tokens = self._build_intent_with_tokens(task)

    # Stage 2: 3-Gram Cell Value Grounding + Steiner FK-Closure Schema Linking.
    # Reuse caller-supplied `value_retriever` so per-DB indexing runs once.
    matches = ()
    if cfg.enable_value_grounding:
      retriever = value_retriever or CellValueRetriever.from_schema_and_sqlite(
          schema, db_path
      )
      matches = retriever.retrieve(intent)

    # Prune distractors while recovering intermediate FK join bridge tables.
    linked = link_schema_with_fk_closure(
        schema=schema,
        intent=intent,
        evidence=task.evidence,
        value_matches=matches,
        config=SchemaLinkerConfig(
            enable_fk_closure=cfg.enable_fk_closure,
            enable_value_grounding=cfg.enable_value_grounding,
        ),
    )

    # Stage 3: Adaptive Complexity-Gated Batch Candidate Generation.
    # Simple queries take the K=1 voice-latency fast path; moderate/challenging
    # queries dispatch K=3 complementary synthesis prompts in one batch!
    use_fast_path = (
        cfg.enable_adaptive_routing
        and intent.complexity == QueryDifficulty.SIMPLE
        and not task.evidence.strip()
    )
    if use_fast_path:
      strategies = (("formula_defensive", 0.0),)
    else:
      strategies = (
          ("decomposed_cot", 0.0),
          ("formula_defensive", 0.0),
          ("skeleton_minimal", 0.1),
      )

    batch_reqs = tuple(
        build_strategy_request(
            request_id=f"{task.question_id}-argus-{idx}",
            ddl_prompt=linked.ddl_prompt,
            question=intent.canonical_question,
            evidence=task.evidence,
            strategy_tag=strat,
            temperature=temp,
        )
        for idx, (strat, temp) in enumerate(strategies)
    )
    # Dispatch selected strategies concurrently via BatchLLMPort.
    resps = self._llm.batch_complete(batch_reqs)
    total_tokens = intent_tokens + sum(r.logical_tokens for r in resps)

    # Wrap raw completion strings into immutable SQLCandidate contracts.
    candidates: list[SQLCandidate] = [
        SQLCandidate(
            sql=r.sql_text,
            strategy=req.strategy_tag,
            prompt_tokens=r.effective_prompt_tokens,
            completion_tokens=r.effective_completion_tokens,
        )
        for req, r in zip(batch_reqs, resps, strict=True)
    ]
    repaired_flags: list[bool] = [False] * len(candidates)

    # Stage 4a: Batched Sandbox Execution + Zero-Token Algebraic Dialect Repair.
    # Deduplicated sandbox execution avoids re-running identical SQL strings.
    outcomes = list(
        self._sandbox.execute_batch(db_path, [c.sql for c in candidates])
    )

    # Deterministically rewrite SQLite division/NULL traps at 0 LLM token cost.
    if cfg.enable_semantic_verifier and cfg.enable_algebraic_repair:
      algebraic_repairs: list[tuple[int, SQLCandidate]] = []
      for idx, (cand, out) in enumerate(zip(candidates, outcomes, strict=True)):
        issues = audit_candidate_invariants(cand, out, intent, linked)
        if issues:
          rewritten = apply_algebraic_sql_repairs(cand, issues, linked)
          if rewritten is not None:
            algebraic_repairs.append((idx, rewritten))

      # Execute all algebraically repaired candidates together in one batch.
      if algebraic_repairs:
        rep_outcomes = self._sandbox.execute_batch(
            db_path, [c.sql for _, c in algebraic_repairs]
        )
        for (orig_idx, rep_c), rep_o in zip(
            algebraic_repairs, rep_outcomes, strict=True
        ):
          _merge_algebraic_repair(
              candidates, outcomes, repaired_flags, orig_idx, rep_c, rep_o
          )

    # Stage 4b: Optional LLM Diagnostic Repair only when zero error-free SQL exists.
    # Soft heuristic warnings (e.g., arity estimates) do not burn extra LLM calls.
    has_clean_candidate = any(
        out.succeeded
        and not any(
            iss.severity == "ERROR"
            for iss in audit_candidate_invariants(
                cand, out, intent, linked, cfg.enable_semantic_verifier
            )
        )
        for cand, out in zip(candidates, outcomes, strict=True)
    )
    if not has_clean_candidate and cfg.max_llm_repair_rounds > 0:
      # Prefer repairing a candidate that parsed & executed in SQLite (e.g., 0
      # rows or division warning) over an empty or syntax-broken completion.
      seed_idx = max(
          range(len(candidates)),
          key=lambda idx: (
              outcomes[idx].succeeded,
              bool(candidates[idx].sql.strip()),
              -idx,
          ),
      )
      seed_cand = candidates[seed_idx]
      primary_issues = audit_candidate_invariants(
          seed_cand,
          outcomes[seed_idx],
          intent,
          linked,
          cfg.enable_semantic_verifier,
      )
      if primary_issues:
        other_failed = tuple(
            c.sql
            for idx, c in enumerate(candidates)
            if idx != seed_idx and c.sql.strip()
        )
        rep_prompt = build_diagnostic_repair_prompt(
            intent.canonical_question,
            task.evidence,
            linked.ddl_prompt,
            seed_cand,
            primary_issues,
            failed_sqls=other_failed,
        )
        rep_req = LLMCompletionRequest(
            request_id=f"{task.question_id}-argus-llm-rep",
            system_prompt=SQLITE_SYSTEM_PROMPT,
            user_prompt=rep_prompt,
            strategy_tag="repair",
        )
        # Execute single-element batch through the unified BatchLLMPort adapter.
        rep_resp = self._llm.batch_complete((rep_req,))[0]
        total_tokens += rep_resp.logical_tokens
        rep_cand = SQLCandidate(
            sql=rep_resp.sql_text,
            strategy="argus_llm_repair",
            prompt_tokens=rep_resp.effective_prompt_tokens,
            completion_tokens=rep_resp.effective_completion_tokens,
        )
        rep_out = self._sandbox.execute(db_path, rep_cand.sql)
        candidates.append(rep_cand)
        outcomes.append(rep_out)
        repaired_flags.append(True)
        llm_idx = len(candidates) - 1

        # Harden LLM-repaired SQL with zero-token algebraic dialect rewrites
        # without duplicating votes if the rewrite output matches `rep_out`.
        if cfg.enable_semantic_verifier and cfg.enable_algebraic_repair:
          rep_issues = audit_candidate_invariants(
              rep_cand, rep_out, intent, linked
          )
          if rep_issues:
            post_alg = apply_algebraic_sql_repairs(rep_cand, rep_issues, linked)
            if post_alg is not None:
              post_out = self._sandbox.execute(db_path, post_alg.sql)
              _merge_algebraic_repair(
                  candidates,
                  outcomes,
                  repaired_flags,
                  llm_idx,
                  post_alg,
                  post_out,
              )

    # Stage 4c: Invariant-Weighted Execution Consensus Arbitration.
    # Penalizes integer-truncated or NULL-order-vulnerable clusters automatically.
    winner, all_verified, _ = verify_and_rank_candidates(
        candidates=candidates,
        outcomes=outcomes,
        intent=intent,
        linked_schema=linked,
        config=VerifierConfig(
            enable_semantic_invariants=cfg.enable_semantic_verifier,
            enable_algebraic_repair=cfg.enable_algebraic_repair,
        ),
        repaired_flags=repaired_flags,
    )

    # Record fine-grained stage telemetry for ablation analysis and paper plots.
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    metadata = (
        ("fast_path", str(use_fast_path)),
        ("bridge_tables", ",".join(linked.bridge_tables_added)),
        ("grounded_values", str(len(linked.value_matches))),
        ("disfluencies_removed", str(intent.disfluencies_removed)),
        ("intent_tokens", str(intent_tokens)),
    )
    # Emit immutable prediction contract containing winning SQL and trace.
    return PipelinePrediction(
        question_id=task.question_id,
        db_id=task.db_id,
        system_name=self.system_name,
        final_sql=winner.candidate.sql,
        latency_ms=round(elapsed_ms, 3),
        total_tokens=total_tokens,
        verified_candidates=all_verified,
        metadata=metadata,
    )
