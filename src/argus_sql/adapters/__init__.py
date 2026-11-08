"""Public API facade for Argus-SQL batch LLM and deterministic adapters."""

# Re-export provider routing factory and safe `.env` defaults loader.
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
# Re-export port protocol and completion dataclasses.
from argus_sql.adapters.types import BatchLLMPort
from argus_sql.adapters.types import LLMCompletionRequest
from argus_sql.adapters.types import LLMCompletionResponse
from argus_sql.adapters.vertex_gemini_client import VertexGeminiBatchClient

# Explicit symbol export list.
__all__ = [
    "AdapterError",
    "BatchLLMPort",
    "LLMAdapterAuthError",
    "LLMAdapterTransportError",
    "LLMCompletionRequest",
    "LLMCompletionResponse",
    "OpenAICompatibleBatchClient",
    "SchemaGuidedDiagnosticSynthesizer",
    "VertexGeminiBatchClient",
    "build_live_llm_client",
    "extract_sql_from_completion",
    "load_dotenv_defaults",
    "validate_endpoint_security",
]
