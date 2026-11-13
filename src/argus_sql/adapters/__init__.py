"""Public API facade for Argus-SQL batch LLM and deterministic adapters."""

# Re-export provider routing factories and safe `.env` defaults loader.
from argus_sql.adapters.client_factory import build_intent_llm_client
from argus_sql.adapters.client_factory import build_live_llm_client
from argus_sql.adapters.client_factory import load_dotenv_defaults
# Re-export deterministic DDL-guided SQL synthesizer for offline benchmarks.
from argus_sql.adapters.deterministic_synthesizer import SchemaGuidedDiagnosticSynthesizer
# Re-export adapter exception classes.
from argus_sql.adapters.exceptions import AdapterError
from argus_sql.adapters.exceptions import LLMAdapterAuthError
from argus_sql.adapters.exceptions import LLMAdapterTransportError
# Re-export HTTPS batch clients and SQL extraction helpers.
from argus_sql.adapters.llm_batch_client import OpenAICompatibleBatchClient
from argus_sql.adapters.llm_batch_client import extract_sql_from_completion
from argus_sql.adapters.llm_batch_client import validate_endpoint_security
# Re-export port protocol, cache handle, and completion dataclasses.
from argus_sql.adapters.types import BatchLLMPort
from argus_sql.adapters.types import LLMCompletionRequest
from argus_sql.adapters.types import LLMCompletionResponse
from argus_sql.adapters.types import VertexCachedContent
# Re-export offline Vertex AI Batch Prediction runner and job manifest types.
from argus_sql.adapters.vertex_batch_prediction import BatchJobManifest
from argus_sql.adapters.vertex_batch_prediction import BatchJobStatus
from argus_sql.adapters.vertex_batch_prediction import VertexBatchPredictionRunner
# Re-export per-database Vertex AI Context Caching adapter and helpers.
from argus_sql.adapters.vertex_context_cache import ContextCacheAdapter
from argus_sql.adapters.vertex_context_cache import build_db_cached_context_text
from argus_sql.adapters.vertex_context_cache import strip_cached_schema_from_user_prompt
# Re-export online Vertex Gemini batch client and R3 payload helpers.
from argus_sql.adapters.vertex_gemini_client import VertexGeminiBatchClient
from argus_sql.adapters.vertex_gemini_client import build_vertex_generate_payload
from argus_sql.adapters.vertex_gemini_client import parse_thinking_budget
from argus_sql.adapters.vertex_gemini_client import resolve_thinking_budget

# Explicit symbol export list sorted alphabetically for clean public imports.
__all__ = [
    "AdapterError",
    "BatchJobManifest",
    "BatchJobStatus",
    "BatchLLMPort",
    "ContextCacheAdapter",
    "LLMAdapterAuthError",
    "LLMAdapterTransportError",
    "LLMCompletionRequest",
    "LLMCompletionResponse",
    "OpenAICompatibleBatchClient",
    "SchemaGuidedDiagnosticSynthesizer",
    "VertexBatchPredictionRunner",
    "VertexCachedContent",
    "VertexGeminiBatchClient",
    "build_db_cached_context_text",
    "build_intent_llm_client",
    "build_live_llm_client",
    "build_vertex_generate_payload",
    "extract_sql_from_completion",
    "load_dotenv_defaults",
    "parse_thinking_budget",
    "resolve_thinking_budget",
    "strip_cached_schema_from_user_prompt",
    "validate_endpoint_security",
]
