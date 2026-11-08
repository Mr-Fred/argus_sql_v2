"""Single-pass comparative NL2SQL baselines (B0, B1) and baseline facade.

Implements single-pass full-DDL (`B0_ZeroShotDDL`) and lexical decomposed CoT
(`B1_DINDecomposed`), and re-exports multi-pass `MACRefinerPipeline` (`B2`) and
`CHESSValueConsistencyPipeline` (`B3`) for uniform benchmark imports.
"""

import pathlib
import time

from argus_sql.adapters.types import BatchLLMPort
from argus_sql.contracts.schema_ddl_renderer import render_schema_ddl
from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import NLQueryTask
from argus_sql.contracts.types import PipelinePrediction
from argus_sql.contracts.types import SQLCandidate
from argus_sql.contracts.types import VerifiedCandidate
from argus_sql.linking.fk_schema_linker import link_schema_with_fk_closure
from argus_sql.linking.types import SchemaLinkerConfig
from argus_sql.linking.value_retriever import CellValueRetriever
# Re-export B2 and B3 ensemble baselines so existing caller imports stay intact.
from argus_sql.pipelines.ensemble_baselines import CHESSValueConsistencyPipeline
from argus_sql.pipelines.ensemble_baselines import MACRefinerPipeline
from argus_sql.pipelines.ensemble_baselines import build_uncleaned_intent
from argus_sql.pipelines.prompt_templates import build_strategy_request
from argus_sql.sandbox.sqlite_sandbox import SQLiteReadOnlySandbox


class ZeroShotDDLPipeline:
  """B0 baseline: Full unpruned schema DDL with single-pass SQL generation."""

  def __init__(
      self,
      llm: BatchLLMPort,
      sandbox: SQLiteReadOnlySandbox,
  ) -> None:
    """Initializes B0_ZeroShotDDL with an LLM port and read-only sandbox.

    Args:
      llm: BatchLLMPort completion adapter.
      sandbox: Read-only SQLite execution sandbox.
    """
    # Store injected Hexagonal ports for fair cross-system benchmarking.
    self._llm = llm
    self._sandbox = sandbox

  @property
  def system_name(self) -> str:
    """Returns canonical leaderboard name for baseline B0."""
    return "B0_ZeroShotDDL"

  def predict(
      self,
      task: NLQueryTask,
      schema: DatabaseSchema,
      db_path: pathlib.Path,
      value_retriever: CellValueRetriever | None = None,
  ) -> PipelinePrediction:
    """Generates a single zero-shot SQL query over the full database DDL.

    Args:
      task: Target NLQueryTask.
      schema: Full DatabaseSchema.
      db_path: Read-only SQLite database path.
      value_retriever: Unused in B0 baseline.

    Returns:
      PipelinePrediction recording final SQL, latency, and token count.
    """
    # B0 intentionally ignores cell-value indexes to measure raw DDL baseline.
    del value_retriever
    start_ns = time.perf_counter_ns()
    # Render 100% of tables and columns with zero pruning or value grounding.
    full_ddl = render_schema_ddl(schema)
    req = build_strategy_request(
        request_id=f"{task.question_id}-b0",
        ddl_prompt=full_ddl,
        question=task.question,
        evidence=task.evidence,
        strategy_tag="zero_shot",
    )
    # Dispatch single-item batch and execute generated SQL inside RO sandbox.
    resp = self._llm.batch_complete((req,))[0]
    cand = SQLCandidate(
        resp.sql_text, "zero_shot", "", resp.prompt_tokens, resp.completion_tokens
    )
    outcome = self._sandbox.execute(db_path, cand.sql)
    # Record total wall-clock latency including DDL serialization and SQLite run.
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    verified = VerifiedCandidate(cand, outcome)
    return PipelinePrediction(
        question_id=task.question_id,
        db_id=task.db_id,
        system_name=self.system_name,
        final_sql=cand.sql,
        latency_ms=round(elapsed_ms, 3),
        total_tokens=resp.prompt_tokens + resp.completion_tokens,
        verified_candidates=(verified,),
    )


class DINDecomposedPipeline:
  """B1 baseline: Lexical schema pruning + decomposed CoT (DIN-SQL style)."""

  def __init__(
      self,
      llm: BatchLLMPort,
      sandbox: SQLiteReadOnlySandbox,
  ) -> None:
    """Initializes B1_DINDecomposed with an LLM port and read-only sandbox.

    Args:
      llm: BatchLLMPort completion adapter.
      sandbox: Read-only SQLite execution sandbox.
    """
    # Share identical LLM and sandbox ports across all baseline experiments.
    self._llm = llm
    self._sandbox = sandbox

  @property
  def system_name(self) -> str:
    """Returns canonical leaderboard name for baseline B1."""
    return "B1_DINDecomposed"

  def predict(
      self,
      task: NLQueryTask,
      schema: DatabaseSchema,
      db_path: pathlib.Path,
      value_retriever: CellValueRetriever | None = None,
  ) -> PipelinePrediction:
    """Prunes schema lexically (no cell index or FK closure) and runs CoT.

    Args:
      task: Target NLQueryTask.
      schema: Relational DatabaseSchema.
      db_path: Read-only SQLite database path.
      value_retriever: Unused in B1 lexical-only schema linking.

    Returns:
      PipelinePrediction recording decomposed CoT prediction.
    """
    # B1 links strictly via schema tokens without database cell inspection.
    del value_retriever
    start_ns = time.perf_counter_ns()
    raw_intent = build_uncleaned_intent(task.question)
    # Lexical linking only: no 3-gram cell grounding and no Steiner FK closure.
    linked = link_schema_with_fk_closure(
        schema=schema,
        intent=raw_intent,
        evidence=task.evidence,
        config=SchemaLinkerConfig(
            enable_fk_closure=False, enable_value_grounding=False
        ),
    )
    req = build_strategy_request(
        request_id=f"{task.question_id}-b1",
        ddl_prompt=linked.ddl_prompt,
        question=task.question,
        evidence=task.evidence,
        strategy_tag="decomposed_cot",
    )
    # Dispatch decomposed CoT prompt over lexically pruned DDL context.
    resp = self._llm.batch_complete((req,))[0]
    cand = SQLCandidate(
        resp.sql_text,
        "decomposed_cot",
        "",
        resp.prompt_tokens,
        resp.completion_tokens,
    )
    # Execute candidate in read-only sandbox without post-execution refinement.
    # Note: B1 drops unmentioned FK bridge tables (`enable_fk_closure=False`).
    outcome = self._sandbox.execute(db_path, cand.sql)
    # Measure end-to-end latency in milliseconds for efficiency comparison.
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    # Return single-candidate prediction contract for B1 evaluation.
    return PipelinePrediction(
        question_id=task.question_id,
        db_id=task.db_id,
        system_name=self.system_name,
        final_sql=cand.sql,
        latency_ms=round(elapsed_ms, 3),
        total_tokens=resp.prompt_tokens + resp.completion_tokens,
        verified_candidates=(VerifiedCandidate(cand, outcome),),
    )


# Expose all four literature baselines (B0..B3) from a single facade module.
__all__ = [
    "CHESSValueConsistencyPipeline",
    "DINDecomposedPipeline",
    "MACRefinerPipeline",
    "ZeroShotDDLPipeline",
]
