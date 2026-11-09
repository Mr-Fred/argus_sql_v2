"""Public API facade for Argus-SQL and controlled operator baseline pipelines."""

# Re-export proposed Argus-SQL hybrid pipeline.
from argus_sql.pipelines.argus_pipeline import ArgusSQLPipeline
# Re-export controlled operator baselines (B0..B4) and legacy class aliases.
from argus_sql.pipelines.baseline_pipelines import CHESSValueConsistencyPipeline
from argus_sql.pipelines.baseline_pipelines import DINDecomposedPipeline
from argus_sql.pipelines.baseline_pipelines import EnrichedMDDLPipeline
from argus_sql.pipelines.baseline_pipelines import ExecRefinerPipeline
from argus_sql.pipelines.baseline_pipelines import LLMJudgePipeline
from argus_sql.pipelines.baseline_pipelines import LexicalCoTPipeline
from argus_sql.pipelines.baseline_pipelines import MACRefinerPipeline
from argus_sql.pipelines.baseline_pipelines import RawDDLPipeline
from argus_sql.pipelines.baseline_pipelines import ValueMajorityVotePipeline
from argus_sql.pipelines.baseline_pipelines import ZeroShotDDLPipeline
# Re-export pipeline domain exceptions.
from argus_sql.pipelines.exceptions import PipelineExecutionError
# Re-export modular prompt template helper.
from argus_sql.pipelines.prompt_templates import build_strategy_request
# Re-export ablation config and pipeline protocol.
from argus_sql.pipelines.types import ArgusAblationConfig
from argus_sql.pipelines.types import NL2SQLPipeline

# Explicit symbol export list.
__all__ = [
    "ArgusAblationConfig",
    "ArgusSQLPipeline",
    "CHESSValueConsistencyPipeline",
    "DINDecomposedPipeline",
    "EnrichedMDDLPipeline",
    "ExecRefinerPipeline",
    "LLMJudgePipeline",
    "LexicalCoTPipeline",
    "MACRefinerPipeline",
    "NL2SQLPipeline",
    "PipelineExecutionError",
    "RawDDLPipeline",
    "ValueMajorityVotePipeline",
    "ZeroShotDDLPipeline",
    "build_strategy_request",
]
