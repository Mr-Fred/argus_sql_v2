"""Provider-routing factory and safe `.env` reader for live batch LLM clients.

Resolves non-secret runtime environment defaults (such as `GOOGLE_CLOUD_PROJECT`,
`GOOGLE_CLOUD_LOCATION`, and `GOOGLE_GENAI_USE_ENTERPRISE`) and constructs the
appropriate `BatchLLMPort` adapter (`VertexGeminiBatchClient` or
`OpenAICompatibleBatchClient`).
"""

import logging
import os
import pathlib

from argus_sql.adapters.exceptions import LLMAdapterAuthError
from argus_sql.adapters.llm_batch_client import OpenAICompatibleBatchClient
from argus_sql.adapters.types import BatchLLMPort
from argus_sql.adapters.vertex_gemini_client import VertexGeminiBatchClient

# Module-level logger following Google Python Style Guide lazy-% conventions.
_LOGGER = logging.getLogger(__name__)

# Maximum `.env` byte size parsed to prevent accidental large-file ingestion.
_MAX_DOTENV_BYTES = 65536

# Truthy string literals recognized for boolean feature flags in `.env` / env.
_TRUTHY_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})


def load_dotenv_defaults(
    dotenv_path: pathlib.Path | None = None,
) -> dict[str, str]:
  """Reads key-value defaults from a local `.env` merged under `os.environ`.

  Process environment variables always override `.env` entries so explicit CLI
  exports take precedence without mutating global `os.environ` state.

  Args:
    dotenv_path: Optional `.env` path; defaults to `./.env` in working dir.

  Returns:
    Merged string mapping of `.env` defaults overlaid with `os.environ`.
  """
  merged: dict[str, str] = {}
  target = dotenv_path if dotenv_path is not None else pathlib.Path(".env")

  # Refuse symlinks or oversized files to maintain strict local file safety.
  if (
      not target.is_symlink()
      and target.is_file()
      and target.stat().st_size <= _MAX_DOTENV_BYTES
  ):
    # Use `utf-8-sig` so Windows Notepad / PowerShell 5.1 UTF-8 BOMs strip cleanly.
    try:
      raw_text = target.read_text(encoding="utf-8-sig")
    except OSError:
      raw_text = ""
    for raw_line in raw_text.splitlines():
      line = raw_line.strip()
      # Skip blank lines and '#' comment annotations.
      if not line or line.startswith("#") or "=" not in line:
        continue
      if line.startswith("export "):
        line = line[len("export ") :].strip()
      key, val = line.split("=", 1)
      clean_key = key.strip()
      clean_val = val.strip().strip("'\"")
      if clean_key:
        merged[clean_key] = clean_val

  # Overlay live process environment variables on top of `.env` defaults.
  for env_k, env_v in os.environ.items():
    merged[env_k] = env_v
  return merged


# Template placeholder project IDs treated as unset in `.env` files.
_PLACEHOLDER_PROJECTS: frozenset[str] = frozenset({
    "your-personal-gcp-project-id",
    "your-gcp-project-id",
    "<your-gcp-project-id>",
})

# Default public OpenAI chat completions HTTPS endpoint.
_DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"


def build_live_llm_client(
    provider: str = "auto",
    model_name: str = "auto",
    endpoint_url: str = _DEFAULT_OPENAI_ENDPOINT,
    gcp_project: str | None = None,
    gcp_location: str | None = None,
    dotenv_path: pathlib.Path | None = None,
) -> BatchLLMPort:
  """Constructs a `BatchLLMPort` client for OpenAI-compatible or Vertex Gemini.

  Args:
    provider: `'auto'`, `'vertex'`, or `'openai'`.
    model_name: Model slug (`'auto'` selects provider-appropriate default).
    endpoint_url: Chat completions URL when `provider` resolves to `'openai'`.
    gcp_project: Optional GCP project override for Vertex AI / Gemini Enterprise.
    gcp_location: Optional Vertex AI location override (defaults to `'global'`).
    dotenv_path: Optional `.env` file path for non-secret local defaults.

  Returns:
    Configured `BatchLLMPort` instance ready for concurrent batch execution.

  Raises:
    LLMAdapterAuthError: If `provider` is unknown or required settings are absent.
  """
  cfg = load_dotenv_defaults(dotenv_path)
  norm_provider = provider.strip().lower()
  if norm_provider not in {"auto", "openai", "vertex"}:
    raise LLMAdapterAuthError(f"Unsupported LLM provider '{provider}'.")

  # Ignore unedited `.env` template placeholders when resolving GCP project ID.
  raw_project = (gcp_project or cfg.get("GOOGLE_CLOUD_PROJECT", "")).strip()
  resolved_project = (
      "" if raw_project.lower() in _PLACEHOLDER_PROJECTS else raw_project
  )
  enterprise_on = (
      cfg.get("GOOGLE_GENAI_USE_ENTERPRISE", "").strip().lower() in _TRUTHY_VALUES
      or cfg.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in _TRUTHY_VALUES
  )
  resolved_api_key = (
      cfg.get("ARGUS_LLM_API_KEY", "").strip()
      or cfg.get("OPENAI_API_KEY", "").strip()
  )
  if norm_provider == "auto":
    # Prefer OpenAI when an API key is supplied without a real GCP project ID.
    use_vertex = bool(
        gcp_project
        or (enterprise_on and resolved_project)
        or (resolved_project and not resolved_api_key)
        or (enterprise_on and not resolved_api_key)
    )
    norm_provider = "vertex" if use_vertex else "openai"

  # Resolve model slug from explicit CLI flag, `ARGUS_LLM_MODEL`, or provider default.
  raw_model = model_name.strip()
  if not raw_model or raw_model.lower() == "auto":
    env_model = cfg.get("ARGUS_LLM_MODEL", "").strip()
    default_slug = "gemini-3.7-flash" if norm_provider == "vertex" else "gpt-4o-mini"
    resolved_model = env_model or default_slug
  else:
    resolved_model = raw_model

  if norm_provider == "vertex":
    # Fail closed immediately when Vertex AI lacks an explicit billing project.
    if not resolved_project:
      raise LLMAdapterAuthError(
          "Vertex AI requires `--gcp-project` or a real `GOOGLE_CLOUD_PROJECT`."
      )
    # Default Vertex AI / Gemini Enterprise routing to the `global` endpoint.
    resolved_loc = (
        gcp_location or cfg.get("GOOGLE_CLOUD_LOCATION", "") or "global"
    ).strip()
    # Log non-secret routing metadata via lazy-% formatting.
    _LOGGER.info(
        "Initialized VertexGeminiBatchClient (location=%s, model=%s).",
        resolved_loc,
        resolved_model,
    )
    return VertexGeminiBatchClient(
        project_id=resolved_project,
        model_name=resolved_model,
        location=resolved_loc,
    )

  # Allow `.env` `ARGUS_LLM_ENDPOINT_URL` to target local loopback vLLM servers.
  eff_endpoint = endpoint_url
  if endpoint_url == _DEFAULT_OPENAI_ENDPOINT:
    eff_endpoint = cfg.get("ARGUS_LLM_ENDPOINT_URL", "").strip() or endpoint_url

  # Default fallback constructs the OpenAI-compatible HTTPS/loopback client.
  _LOGGER.info("Initialized OpenAICompatibleBatchClient (model=%s).", resolved_model)
  return OpenAICompatibleBatchClient(
      model_name=resolved_model,
      endpoint_url=eff_endpoint,
      api_key=resolved_api_key or None,
  )
