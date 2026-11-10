"""Configuration and protocol contracts for comparative NL2SQL systems."""

import dataclasses
import pathlib
from typing import Protocol

from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import NLQueryTask
from argus_sql.contracts.types import PipelinePrediction
from argus_sql.linking.value_retriever import CellValueRetriever


# Composable ablation toggles enabling controlled leave-one-out experiments.
@dataclasses.dataclass(frozen=True)
class ArgusAblationConfig:
  """Feature flags controlling each architectural stage inside Argus-SQL.

  Attributes:
    system_name: Display name recorded in benchmark tables and reports.
    enable_voice_canonicalizer: Strip ASR disfluencies & normalize spoken cues.
    enable_value_grounding: Retrieve character 3-gram database cell literals.
    enable_fk_closure: Recover multi-hop bridge tables via BFS FK closure.
    enable_adaptive_routing: Use K=1 fast path on simple tasks vs fixed K=3.
    enable_semantic_verifier: Audit SQLite dialect & semantic invariants.
    enable_algebraic_repair: Apply zero-token algebraic dialect rewrites.
    max_llm_repair_rounds: Maximum LLM diagnostic repair rounds on hard errors.
    enable_llm_intent_extractor: Use optional flash-lite LLM Stage 1 pass.
  """

  # Default flags enable the full 4-stage Argus-SQL sweet-spot configuration.
  # Toggles operate orthogonally so any single stage can be ablated cleanly.
  system_name: str = "Argus-SQL"
  enable_voice_canonicalizer: bool = True
  enable_value_grounding: bool = True
  enable_fk_closure: bool = True
  enable_adaptive_routing: bool = True
  enable_semantic_verifier: bool = True
  enable_algebraic_repair: bool = True
  max_llm_repair_rounds: int = 1
  enable_llm_intent_extractor: bool = False


# Shared structural interface implemented by B0, B1, B2, B3, and Argus-SQL.
# Structural subtyping (`Protocol`) lets external plugins conform without ABCs.
class NL2SQLPipeline(Protocol):
  """Common interface for evaluating any NL2SQL system on a benchmark task."""

  @property
  def system_name(self) -> str:
    """Unique human-readable system identifier for benchmark leaderboards."""
    ...

  def predict(
      self,
      task: NLQueryTask,
      schema: DatabaseSchema,
      db_path: pathlib.Path,
      value_retriever: CellValueRetriever | None = None,
  ) -> PipelinePrediction:
    """Generates and selects a final SQL query for `task` against `schema`.

    Args:
      task: Natural language or spoken benchmark task.
      schema: Relational DatabaseSchema definition.
      db_path: Path to read-only SQLite database file.
      value_retriever: Pre-built 3-gram CellValueRetriever for `schema`.

    Returns:
      PipelinePrediction containing final SQL, latency, tokens, and trace.
    """
    ...
