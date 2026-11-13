"""Provider-routing factory and safe `.env` reader for live batch LLM clients.

Resolves non-secret runtime environment defaults (such as `GOOGLE_CLOUD_PROJECT`,
`GOOGLE_CLOUD_LOCATION`, `GOOGLE_GENAI_USE_ENTERPRISE`, and
`ARGUS_THINKING_BUDGET`) and constructs the appropriate `BatchLLMPort` adapter
(`VertexGeminiBatchClient` or `OpenAICompatibleBatchClient`).
"""

from collections.abc import Mapping
import logging
import os
import pathlib

from argus_sql.adapters.exceptions import LLMAdapterAuthError
from argus_sql.adapters.llm_batch_client import OpenAICompatibleBatchClient
from argus_sql.adapters.types import BatchLLMPort
from argus_sql.adapters.vertex_gemini_client import VertexGeminiBatchClient
from argus_sql.adapters.vertex_gemini_client import parse_thinking_budget

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


def _resolve_provider_and_credentials(
    provider: str,
    gcp_project: str | None,
    cfg: Mapping[str, str],
) -> tuple[str, str, str]:
  """Resolves canonical provider (`'vertex'` or `'openai'`), project, and key."""
  # Normalize provider flag and reject unknown backend strings early.
  norm_provider = provider.strip().lower()
  if norm_provider not in {"auto", "openai", "vertex"}:
    raise LLMAdapterAuthError(f"Unsupported LLM provider '{provider}'.")

  # Ignore unedited `.env` template placeholders when resolving GCP project ID.
  explicit_proj = (gcp_project or "").strip()
  valid_explicit_proj = (
      "" if explicit_proj.lower() in _PLACEHOLDER_PROJECTS else explicit_proj
  )
  raw_project = (valid_explicit_proj or cfg.get("GOOGLE_CLOUD_PROJECT", "")).strip()
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
    # Prefer OpenAI when an API key is supplied without Vertex Enterprise/project.
    use_vertex = bool(
        valid_explicit_proj
        or (enterprise_on and resolved_project)
        or (resolved_project and not resolved_api_key)
        or (enterprise_on and not resolved_api_key)
    )
    norm_provider = "vertex" if use_vertex else "openai"
  return (norm_provider, resolved_project, resolved_api_key)


def _resolve_validated_thinking_budget(
    thinking_budget: int | None,
    cfg: Mapping[str, str],
) -> int:
  """Validates explicit `thinking_budget` or `ARGUS_THINKING_BUDGET` eagerly."""
  # Prefer explicit caller argument; otherwise inspect `.env` / `os.environ`.
  # Validating before provider routing ensures `-5`, `True`, and `'abc'` fail
  # closed even when `provider='openai'` or `fallback_to_none=True`.
  raw_budget: int | str | None = (
      thinking_budget
      if thinking_budget is not None
      else cfg.get("ARGUS_THINKING_BUDGET", "").strip() or None
  )
  return parse_thinking_budget(raw_budget)


def build_live_llm_client(
    provider: str = "auto",
    model_name: str = "auto",
    endpoint_url: str = _DEFAULT_OPENAI_ENDPOINT,
    gcp_project: str | None = None,
    gcp_location: str | None = None,
    dotenv_path: pathlib.Path | None = None,
    thinking_budget: int | None = None,
) -> BatchLLMPort:
  """Constructs a `BatchLLMPort` client for OpenAI-compatible or Vertex Gemini.

  Args:
    provider: `'auto'`, `'vertex'`, or `'openai'`.
    model_name: Model slug (`'auto'` selects `'gemini-3.7-flash'` on Vertex AI).
    endpoint_url: Chat completions URL when `provider` resolves to `'openai'`.
    gcp_project: Optional GCP project override for Vertex AI / Gemini Enterprise.
    gcp_location: Optional Vertex AI location override (defaults to `'global'`).
    dotenv_path: Optional `.env` file path for non-secret local defaults.
    thinking_budget: Optional Gemini `thinkingBudget` override; falls back to
      `ARGUS_THINKING_BUDGET` in environment / `.env` (`2048` default).

  Returns:
    Configured `BatchLLMPort` instance ready for concurrent batch execution.

  Raises:
    LLMAdapterAuthError: If `provider` or `thinking_budget` is invalid, or if
      required credentials/settings are absent.
  """
  cfg = load_dotenv_defaults(dotenv_path)
  # Validate `thinking_budget` / `ARGUS_THINKING_BUDGET` eagerly before
  # provider resolution so invalid budgets fail closed on every backend.
  resolved_budget = _resolve_validated_thinking_budget(thinking_budget, cfg)
  norm_provider, resolved_project, resolved_api_key = (
      _resolve_provider_and_credentials(provider, gcp_project, cfg)
  )

  # Resolve model slug from explicit CLI flag, `ARGUS_LLM_MODEL`, or provider default.
  raw_model = model_name.strip()
  if not raw_model or raw_model.lower() == "auto":
    env_model = cfg.get("ARGUS_LLM_MODEL", "").strip()
    # Reserve `gemini-3.7-flash` as the Vertex default for core SQL synthesis.
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
        "Initialized VertexGeminiBatchClient (location=%s, model=%s, budget=%d).",
        resolved_loc,
        resolved_model,
        resolved_budget,
    )
    return VertexGeminiBatchClient(
        project_id=resolved_project,
        model_name=resolved_model,
        location=resolved_loc,
        thinking_budget=resolved_budget,
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


def build_intent_llm_client(
    provider: str = "auto",
    model_name: str = "auto",
    endpoint_url: str = _DEFAULT_OPENAI_ENDPOINT,
    gcp_project: str | None = None,
    gcp_location: str | None = None,
    dotenv_path: pathlib.Path | None = None,
    fallback_to_none: bool = True,
    thinking_budget: int | None = None,
) -> BatchLLMPort | None:
  """Constructs a `gemini-2.5-flash` `BatchLLMPort` for Stage 1 intent routing.

  In hermetic or unconfigured offline environments, returns `None` when
  `fallback_to_none=True` so Stage 1 automatically uses the deterministic FSM.

  Args:
    provider: `'auto'`, `'vertex'`, or `'openai'`.
    model_name: Intent model slug (`'auto'` selects `'gemini-2.5-flash'` on
      Vertex AI or `'gpt-4o-mini'` on OpenAI, overridden by
      `ARGUS_INTENT_LLM_MODEL`).
    endpoint_url: Chat completions URL when `provider` resolves to `'openai'`.
    gcp_project: Optional GCP project override for Vertex AI.
    gcp_location: Optional Vertex AI location override.
    dotenv_path: Optional `.env` file path for non-secret local defaults.
    fallback_to_none: Return `None` on missing credentials instead of raising.
    thinking_budget: Optional `thinkingBudget` override passed to Vertex client.

  Returns:
    Configured `BatchLLMPort` or `None` when offline fallback applies.

  Raises:
    LLMAdapterAuthError: If `provider` or `thinking_budget` is invalid, or if
      `fallback_to_none=False` and credentials are absent.
  """
  cfg = load_dotenv_defaults(dotenv_path)
  # Validate `thinking_budget` / `ARGUS_THINKING_BUDGET` eagerly before
  # provider routing or `fallback_to_none` catching so bad budgets never
  # silently return `None` in offline environments.
  resolved_budget = _resolve_validated_thinking_budget(thinking_budget, cfg)
  # Share exact provider auto-detection logic with `build_live_llm_client`.
  norm_provider, _, _ = _resolve_provider_and_credentials(
      provider, gcp_project, cfg
  )

  raw_model = model_name.strip()
  if not raw_model or raw_model.lower() == "auto":
    env_intent_model = cfg.get("ARGUS_INTENT_LLM_MODEL", "").strip()
    # Route Vertex AI Stage 1 intent extraction to `gemini-2.5-flash` per R4.
    is_vertex = norm_provider == "vertex"
    resolved_model = env_intent_model or (
        "gemini-2.5-flash" if is_vertex else "gpt-4o-mini"
    )
  else:
    resolved_model = raw_model

  # Delegate construction while gracefully returning `None` offline if enabled.
  try:
    return build_live_llm_client(
        provider=norm_provider,
        model_name=resolved_model,
        endpoint_url=endpoint_url,
        gcp_project=gcp_project,
        gcp_location=gcp_location,
        dotenv_path=dotenv_path,
        thinking_budget=resolved_budget,
    )
  except LLMAdapterAuthError as err:
    if not fallback_to_none:
      raise
    _LOGGER.info(
        "Intent LLM unavailable (%s); falling back to deterministic FSM.",
        type(err).__name__,
    )
    return None
