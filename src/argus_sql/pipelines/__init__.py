"""Public API facade for Argus-SQL and comparative baseline pipelines."""

# Re-export proposed Argus-SQL hybrid pipeline.
from argus_sql.pipelines.argus_pipeline import ArgusSQLPipeline
# Re-export comparative literature baselines B0, B1, B2, and B3.
from argus_sql.pipelines.baseline_pipelines import CHESSValueConsistencyPipeline
from argus_sql.pipelines.baseline_pipelines import DINDecomposedPipeline
from argus_sql.pipelines.baseline_pipelines import MACRefinerPipeline
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
    "MACRefinerPipeline",
    "NL2SQLPipeline",
    "PipelineExecutionError",
    "ZeroShotDDLPipeline",
    "build_strategy_request",
]
