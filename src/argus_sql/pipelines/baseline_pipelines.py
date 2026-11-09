"""Controlled single-pass NL2SQL operator baselines (B0, B0+, B1) and facade.

Implements controlled, backbone-matched single-pass operator baselines:
- `RawDDLPipeline` (`B0_RawDDL`): Pure un-enriched `CREATE TABLE` DDL (`K=1`).
- `EnrichedMDDLPipeline` (`B0+_EnrichedMDDL`): Profiler-enriched full M-DDL (`K=1`).
- `LexicalCoTPipeline` (`B1_LexicalCoT`): Lexical schema pruning + CoT (`K=1`).
Also re-exports `ExecRefinerPipeline` (`B2`), `ValueMajorityVotePipeline` (`B3`),
`LLMJudgePipeline` (`B4`), and backward-compatible class aliases.
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
# Re-export multi-pass & ensemble operator baselines from dedicated modules.
from argus_sql.pipelines.ensemble_baselines import CHESSValueConsistencyPipeline
from argus_sql.pipelines.ensemble_baselines import ExecRefinerPipeline
from argus_sql.pipelines.ensemble_baselines import MACRefinerPipeline
from argus_sql.pipelines.ensemble_baselines import ValueMajorityVotePipeline
from argus_sql.pipelines.ensemble_baselines import build_uncleaned_intent
from argus_sql.pipelines.judge_baseline import LLMJudgePipeline
from argus_sql.pipelines.prompt_templates import build_strategy_request
from argus_sql.sandbox.sqlite_sandbox import SQLiteReadOnlySandbox


def _predict_full_ddl_single_pass(
    llm: BatchLLMPort,
    sandbox: SQLiteReadOnlySandbox,
    task: NLQueryTask,
    schema: DatabaseSchema,
    db_path: pathlib.Path,
    system_name: str,
    req_tag: str,
    include_metadata_comments: bool,
) -> PipelinePrediction:
  """Executes a single zero-shot `K=1` completion over full raw or M-Schema DDL."""
  start_ns = time.perf_counter_ns()
  # Render 100% of schema tables; toggle inline profiler metadata comments
  # (`range:`, `examples:`, and CSV descriptions) to isolate M-Schema profiling.
  full_ddl = render_schema_ddl(
      schema, include_metadata_comments=include_metadata_comments
  )
  # Build greedy (`temperature=0.0`) zero-shot prompt over the unpruned schema.
  req = build_strategy_request(
      request_id=f"{task.question_id}-{req_tag}",
      ddl_prompt=full_ddl,
      question=task.question,
      evidence=task.evidence,
      strategy_tag="zero_shot",
  )
  # Dispatch single-item batch and execute generated SQL inside RO sandbox.
  resp = llm.batch_complete((req,))[0]
  cand = SQLCandidate(
      resp.sql_text,
      "zero_shot",
      "",
      resp.effective_prompt_tokens,
      resp.effective_completion_tokens,
  )
  # Measure total wall-clock latency across DDL serialization and SQLite run.
  outcome = sandbox.execute(db_path, cand.sql)
  elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
  # Return immutable single-candidate prediction contract.
  return PipelinePrediction(
      question_id=task.question_id,
      db_id=task.db_id,
      system_name=system_name,
      final_sql=cand.sql,
      latency_ms=round(elapsed_ms, 3),
      total_tokens=resp.logical_tokens,
      verified_candidates=(VerifiedCandidate(cand, outcome),),
  )


class RawDDLPipeline:
  """B0 controlled operator baseline: True un-enriched raw `CREATE TABLE` DDL."""

  def __init__(
      self,
      llm: BatchLLMPort,
      sandbox: SQLiteReadOnlySandbox,
  ) -> None:
    """Initializes B0_RawDDL with shared LLM and read-only SQLite ports.

    Args:
      llm: BatchLLMPort completion adapter.
      sandbox: Read-only SQLite execution sandbox.
    """
    # Store injected Hexagonal ports for controlled operator benchmarking.
    self._llm = llm
    self._sandbox = sandbox

  @property
  def system_name(self) -> str:
    """Returns canonical controlled operator name for un-enriched raw DDL."""
    # Distinguishes pure `CREATE TABLE` DDL from profiler-enriched `B0+`.
    return "B0_RawDDL"

  def predict(
      self,
      task: NLQueryTask,
      schema: DatabaseSchema,
      db_path: pathlib.Path,
      value_retriever: CellValueRetriever | None = None,
  ) -> PipelinePrediction:
    """Generates one zero-shot SQL query over pure un-commented SQLite DDL.

    Args:
      task: Target NLQueryTask.
      schema: Full DatabaseSchema.
      db_path: Read-only SQLite database path.
      value_retriever: Unused in raw DDL baseline.

    Returns:
      PipelinePrediction recording final SQL, latency, and token count.
    """
    # Omit all statistical `range:`, `examples:`, and CSV codebook comments.
    del value_retriever
    return _predict_full_ddl_single_pass(
        self._llm,
        self._sandbox,
        task,
        schema,
        db_path,
        self.system_name,
        "b0-raw",
        include_metadata_comments=False,
    )


class EnrichedMDDLPipeline:
  """B0+ controlled operator baseline: Full profiler-enriched M-Schema DDL."""

  def __init__(
      self,
      llm: BatchLLMPort,
      sandbox: SQLiteReadOnlySandbox,
  ) -> None:
    """Initializes B0+_EnrichedMDDL with shared LLM and read-only SQLite ports.

    Args:
      llm: BatchLLMPort completion adapter.
      sandbox: Read-only SQLite execution sandbox.
    """
    # Share identical LLM and sandbox ports across all baseline experiments.
    self._llm = llm
    self._sandbox = sandbox

  @property
  def system_name(self) -> str:
    """Returns canonical controlled operator name for enriched full M-DDL."""
    # Isolates offline statistical M-Schema profiling over unpruned tables.
    return "B0+_EnrichedMDDL"

  def predict(
      self,
      task: NLQueryTask,
      schema: DatabaseSchema,
      db_path: pathlib.Path,
      value_retriever: CellValueRetriever | None = None,
  ) -> PipelinePrediction:
    """Generates one zero-shot SQL query over full profiler-enriched M-DDL.

    Args:
      task: Target NLQueryTask.
      schema: Full DatabaseSchema with offline range/example/CSV metadata.
      db_path: Read-only SQLite database path.
      value_retriever: Unused in unpruned B0+ baseline.

    Returns:
      PipelinePrediction isolating offline M-Schema profiling vs `B0_RawDDL`.
    """
    # Retain offline column metadata comments while skipping query-time linking.
    del value_retriever
    return _predict_full_ddl_single_pass(
        self._llm,
        self._sandbox,
        task,
        schema,
        db_path,
        self.system_name,
        "b0-mddl",
        include_metadata_comments=True,
    )


class LexicalCoTPipeline:
  """B1 controlled operator baseline: Lexical pruning + decomposed CoT (`K=1`)."""

  def __init__(
      self,
      llm: BatchLLMPort,
      sandbox: SQLiteReadOnlySandbox,
  ) -> None:
    """Initializes B1_LexicalCoT with shared LLM and read-only SQLite ports.

    Args:
      llm: BatchLLMPort completion adapter.
      sandbox: Read-only SQLite execution sandbox.
    """
    # Share identical LLM and sandbox ports across all baseline experiments.
    self._llm = llm
    self._sandbox = sandbox

  @property
  def system_name(self) -> str:
    """Returns canonical controlled operator name for baseline B1."""
    # Isolates lexical-only schema pruning + single-pass CoT without FK closure.
    return "B1_LexicalCoT"

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
    # B1 links strictly via schema tokens without cell inspection or FK closure.
    del value_retriever
    start_ns = time.perf_counter_ns()
    raw_intent = build_uncleaned_intent(task.question, task.evidence)
    # Disable both Steiner FK bridge recovery and 3-gram cell grounding.
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
        resp.effective_prompt_tokens,
        resp.effective_completion_tokens,
    )
    # Execute candidate in read-only sandbox without post-execution refinement.
    outcome = self._sandbox.execute(db_path, cand.sql)
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    # Return single-candidate prediction contract for B1 evaluation.
    return PipelinePrediction(
        question_id=task.question_id,
        db_id=task.db_id,
        system_name=self.system_name,
        final_sql=cand.sql,
        latency_ms=round(elapsed_ms, 3),
        total_tokens=resp.logical_tokens,
        verified_candidates=(VerifiedCandidate(cand, outcome),),
    )


# Backward-compatible class aliases preserving existing caller imports.
ZeroShotDDLPipeline = EnrichedMDDLPipeline
DINDecomposedPipeline = LexicalCoTPipeline

# Expose all six controlled operator baselines (B0..B4) and legacy aliases.
__all__ = [
    "CHESSValueConsistencyPipeline",
    "DINDecomposedPipeline",
    "EnrichedMDDLPipeline",
    "ExecRefinerPipeline",
    "LLMJudgePipeline",
    "LexicalCoTPipeline",
    "MACRefinerPipeline",
    "RawDDLPipeline",
    "ValueMajorityVotePipeline",
    "ZeroShotDDLPipeline",
]
