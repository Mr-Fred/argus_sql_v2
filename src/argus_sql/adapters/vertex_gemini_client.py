"""Vertex AI / Gemini Enterprise batch-optimized HTTPS transport adapter.

Executes SQL generation batches concurrently via `ThreadPoolExecutor.map` over
Vertex AI's public `generateContent` REST endpoint (`aiplatform.googleapis.com`)
using Python's standard library, TLS 1.2+, configurable `thinkingConfig`,
unclamped output capacity, `cachedContent` references, and SHA-256 deduplication.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import logging
import math
import os
import pathlib
import secrets
import ssl
import threading
import time
import urllib.error
import urllib.request

from argus_sql.adapters.exceptions import LLMAdapterAuthError
from argus_sql.adapters.exceptions import LLMAdapterTransportError
from argus_sql.adapters.llm_batch_client import extract_sql_from_completion
from argus_sql.adapters.types import LLMCompletionRequest
from argus_sql.adapters.types import LLMCompletionResponse
from argus_sql.adapters.vertex_auth import VertexTokenResolver
from argus_sql.adapters.vertex_auth import build_vertex_generate_url
from argus_sql.adapters.vertex_auth import normalize_gemini_model_name
from argus_sql.adapters.vertex_auth import validate_cached_content_name
from argus_sql.adapters.vertex_auth import validate_gcp_location
from argus_sql.adapters.vertex_auth import validate_gcp_project_id

# Module-level logger following Google Python Style Guide lazy-% conventions.
_LOGGER = logging.getLogger(__name__)

# Transient upstream HTTP status codes eligible for bounded exponential retry.
_RETRYABLE_HTTP_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})

# Default and minimum reasoning `thinkingBudget` tokens per R3 specification.
# Keeping reasoning tasks >= 2048 preserves multi-hop SQL synthesis accuracy.
DEFAULT_THINKING_BUDGET: int = 2048
MIN_REASONING_THINKING_BUDGET: int = 2048

# Deterministic baseline (`B0`, `B0+`) and Stage 1 intent tags disable thinking.
_NON_REASONING_STRATEGY_TAGS: frozenset[str] = frozenset({
    "zero_shot",
    "intent_flash_lite",
})


def parse_thinking_budget(
    raw_value: int | str | None,
    default: int = DEFAULT_THINKING_BUDGET,
) -> int:
  """Parses and validates `ARGUS_THINKING_BUDGET` or constructor `thinking_budget`.

  Args:
    raw_value: Optional integer, numeric string (`'0'`, `'2048'`), or `None`.
    default: Default thinking token budget returned when `raw_value` is unset.

  Returns:
    Validated non-negative integer thinking budget (`0` or `>= 0`).

  Raises:
    LLMAdapterAuthError: If `raw_value` is boolean, non-integer, or negative.
  """
  # Fall back to the default reasoning budget when no env/init value is set.
  if raw_value is None:
    return default
  # Reject bool explicitly because `bool` subclasses `int` in Python.
  if isinstance(raw_value, bool):
    raise LLMAdapterAuthError("Invalid thinking_budget: boolean is not allowed.")
  if isinstance(raw_value, str):
    cleaned = raw_value.strip()
    if not cleaned:
      return default
    try:
      parsed = int(cleaned)
    except ValueError as err:
      raise LLMAdapterAuthError(
          "ARGUS_THINKING_BUDGET must be a non-negative integer."
      ) from err
  elif isinstance(raw_value, int):
    parsed = raw_value
  else:
    raise LLMAdapterAuthError("Unsupported thinking_budget value type.")

  # Negative budgets (`-1` dynamic mode) are rejected to prevent unbounded spend.
  if parsed < 0:
    raise LLMAdapterAuthError("thinking_budget must be >= 0.")
  return parsed


def resolve_thinking_budget(
    strategy_tag: str,
    configured_budget: int,
    request_override: int | None = None,
) -> int:
  """Resolves effective `thinkingBudget` for a request strategy and client config.

  Non-reasoning baselines (`B0_RawDDL` and `B0+_EnrichedMDDL`, tagged
  `'zero_shot'`) and Stage 1 JSON intent extraction (`'intent_flash_lite'`)
  receive `thinkingBudget=0`, while reasoning strategies enforce `>= 2048`.

  Args:
    strategy_tag: Synthesis strategy tag from `LLMCompletionRequest`.
    configured_budget: Client-level configured budget (`0` or `>= 2048`).
    request_override: Optional per-request `thinking_budget` override.

  Returns:
    Effective integer `thinkingBudget` (`0` or `>= 2048`).

  Raises:
    LLMAdapterAuthError: If `configured_budget` or `request_override` is invalid.
  """
  # Per-request explicit override takes highest precedence when supplied.
  # Reject bool, float/NaN, and non-int runtime overrides deterministically.
  if request_override is not None:
    if (isinstance(request_override, bool)
        or not isinstance(request_override, int) or request_override < 0):
      raise LLMAdapterAuthError("Request thinking_budget override must be >= 0.")
    if request_override == 0:
      return 0
    return max(MIN_REASONING_THINKING_BUDGET, request_override)

  # Validate client-level configured budget before applying strategy rules.
  if (isinstance(configured_budget, bool)
      or not isinstance(configured_budget, int) or configured_budget < 0):
    raise LLMAdapterAuthError("Configured thinking_budget must be >= 0.")
  norm_tag = strategy_tag.strip().lower()
  # Zero out thinking on deterministic baselines (`B0`/`B0+`) or explicit `0`.
  if norm_tag in _NON_REASONING_STRATEGY_TAGS or configured_budget == 0:
    return 0
  # Enforce the 2048-token floor across all reasoning synthesis/judge/repair tags.
  return max(MIN_REASONING_THINKING_BUDGET, configured_budget)


def build_vertex_generate_payload(
    req: LLMCompletionRequest,
    configured_thinking_budget: int = DEFAULT_THINKING_BUDGET,
    default_cached_content: str | None = None,
    labels: Mapping[str, str] | None = None,
) -> dict[str, object]:
  """Builds a Vertex AI `GenerateContentRequest` JSON dict (`R1`, `R2`, `R3`).

  Omits `maxOutputTokens` clamping so the model retains full native output
  capacity, configures `thinkingConfig.thinkingBudget`, and omits inline
  `systemInstruction` whenever `cachedContent` is attached per Vertex REST rules.

  Args:
    req: Domain `LLMCompletionRequest` to serialize.
    configured_thinking_budget: Client-level `thinking_budget` setting.
    default_cached_content: Optional client-level `cachedContents/...` path.
    labels: Optional GCP request labels (e.g., `{'argus_hash': ...}` for batch).

  Returns:
    JSON-serializable Vertex AI `GenerateContentRequest` dictionary.
  """
  eff_budget = resolve_thinking_budget(
      req.strategy_tag,
      configured_thinking_budget,
      req.thinking_budget,
  )
  raw_cache = (
      req.cached_content
      if req.cached_content is not None
      else default_cached_content
  )
  # Validate any attached `cachedContent` resource path against traversal.
  active_cache = (
      validate_cached_content_name(raw_cache)
      if (raw_cache and raw_cache.strip())
      else None
  )

  # Omit `maxOutputTokens` so reasoning thoughts never truncate SQL output.
  # Canonicalize IEEE-754 `-0.0` to `0.0` and reject bool/NaN/Inf temperatures.
  temp_val = coerce_finite_float(req.temperature, 0.0)
  norm_temp = 0.0 if temp_val == 0.0 else temp_val
  body_dict: dict[str, object] = {
      "contents": [{"role": "user", "parts": [{"text": req.user_prompt}]}],
      "generationConfig": {
          "temperature": norm_temp,
          "thinkingConfig": {"thinkingBudget": eff_budget},
      },
  }
  # Vertex REST forbids `systemInstruction` when `cachedContent` is attached.
  if active_cache is not None:
    body_dict["cachedContent"] = active_cache
  elif req.system_prompt.strip():
    # Attach `systemInstruction` without `role` per canonical Vertex REST schema.
    body_dict["systemInstruction"] = {
        "parts": [{"text": req.system_prompt}],
    }
  # Attach correlation labels when serializing offline batch prediction lines.
  if labels:
    body_dict["labels"] = dict(labels)
  return body_dict


def coerce_nonneg_int(raw_val: object, fallback: int) -> int:
  """Coalesces `None` to `fallback` and validates a non-negative JSON integer.

  Args:
    raw_val: Decoded JSON value (`int`, base-10 integer `str`, or `None`).
    fallback: Default non-negative integer returned when `raw_val` is `None`.

  Returns:
    Validated non-negative integer (`>= 0`).

  Raises:
    LLMAdapterTransportError: If `raw_val` is `bool`, non-integer, or `< 0`.
  """
  # Coalesce omitted fields and explicit JSON `null` (`None`) to `fallback`.
  if raw_val is None:
    return fallback
  # Reject `bool` (`isinstance(True, int)`), non-int types, and negative counts.
  try:
    if isinstance(raw_val, bool) or not isinstance(raw_val, (int, str)):
      raise TypeError("non-integer token field")
    parsed = raw_val if isinstance(raw_val, int) else int(raw_val.strip(), 10)
    if parsed < 0:
      raise ValueError("negative token field")
  except (TypeError, ValueError) as err:
    raise LLMAdapterTransportError("Invalid integer field in JSON payload.") from err
  return parsed


def coerce_finite_float(raw_val: object, fallback: float = 0.0) -> float:
  """Coalesces `None` to `fallback` and validates a finite JSON float.

  Args:
    raw_val: Decoded numeric value (`int`, `float`, numeric `str`, or `None`).
    fallback: Default finite float returned when `raw_val` is `None`.

  Returns:
    Validated finite float (`NaN` and `+-Inf` rejected).

  Raises:
    LLMAdapterTransportError: If `raw_val` is `bool`, non-numeric, or `NaN`/`Inf`.
  """
  # Coalesce omitted fields and explicit JSON `null` (`None`) to `fallback`.
  if raw_val is None:
    return fallback
  # Reject `bool`, non-numeric scalars, and non-finite IEEE-754 `NaN`/`+-Inf`.
  try:
    if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float, str)):
      raise TypeError("non-numeric float field")
    parsed = float(raw_val)
    if not math.isfinite(parsed):
      raise ValueError("non-finite float field")
  except (TypeError, ValueError, OverflowError) as err:
    raise LLMAdapterTransportError("Invalid float field in JSON payload.") from err
  return parsed


def extract_gemini_completion_triple(
    body: dict[str, object],
    prompt_char_len: int,
) -> tuple[str, int, int]:
  """Extracts non-thought response text and token counts from Vertex JSON.

  Args:
    body: Parsed JSON dictionary returned by Vertex AI `generateContent`.
    prompt_char_len: Character length of the input prompt for fallback accounting.

  Returns:
    Triple `(raw_text, prompt_tokens, completion_tokens)`.

  Raises:
    LLMAdapterTransportError: If `candidates`, `content`, `parts`, or
      `usageMetadata` is missing or malformed.
  """
  # Validate top-level JSON object, `candidates`, `content`, and `parts` types.
  candidates = body.get("candidates") if isinstance(body, dict) else None
  first = candidates[0] if isinstance(candidates, list) and candidates else None
  if not isinstance(first, dict):
    raise LLMAdapterTransportError("Vertex Gemini response lacked valid candidate.")
  content = {} if first.get("content") is None else first.get("content")
  parts = (
      ([] if content.get("parts") is None else content.get("parts"))
      if isinstance(content, dict)
      else None
  )
  if not isinstance(parts, list):
    raise LLMAdapterTransportError("Malformed Vertex candidate content or parts.")

  # Separate final SQL parts from internal Gemini `thought=True` parts.
  visible_chunks: list[str] = []
  for part in parts:
    if not isinstance(part, dict):
      raise LLMAdapterTransportError("Malformed Vertex candidate part object.")
    txt_val, th_val = part.get("text"), part.get("thought")
    # Reject non-string `text` or non-boolean `thought` flags in candidate parts.
    if (txt_val is not None and not isinstance(txt_val, str)) or (
        th_val is not None and not isinstance(th_val, bool)
    ):
      raise LLMAdapterTransportError("Malformed Vertex candidate part fields.")
    if isinstance(txt_val, str) and not th_val:
      visible_chunks.append(txt_val)

  # Never leak `thought=True` scratchpads into SQL extraction if output truncated.
  raw_out = "\n".join(visible_chunks).strip()
  # Coalesce explicit `None` on `usageMetadata` and token fields; reject `bool`/<0.
  usage = {} if body.get("usageMetadata") is None else body.get("usageMetadata")
  if not isinstance(usage, dict):
    raise LLMAdapterTransportError("Malformed Vertex usageMetadata object.")
  p_tok = coerce_nonneg_int(usage.get("promptTokenCount"), prompt_char_len // 4)
  # Sum visible candidate tokens and reasoning thought tokens for true cost.
  c_tok = coerce_nonneg_int(
      usage.get("candidatesTokenCount"), len(raw_out) // 4
  ) + coerce_nonneg_int(usage.get("thoughtsTokenCount"), 0)
  return (raw_out, p_tok, c_tok)


class VertexGeminiBatchClient:
  """Zero-dependency Vertex AI / Gemini Enterprise `BatchLLMPort` adapter.

  Attributes:
    project_id: Validated Google Cloud project ID used for Vertex AI billing.
    location: Validated Vertex AI location (`'global'` or regional).
    model_name: Normalized Gemini model identifier (e.g., `'gemini-3.7-flash'`).
    endpoint_url: Constructed HTTPS `generateContent` URL.
    max_concurrency: Bounded worker pool size for parallel batch execution.
    thinking_budget: Configured reasoning budget (`ARGUS_THINKING_BUDGET` or `2048`).
    default_cached_content: Optional active `cachedContents/...` resource path.
  """

  def __init__(
      self,
      project_id: str,
      model_name: str = "gemini-3.7-flash",
      location: str = "global",
      max_concurrency: int = 24,
      timeout_sec: float = 120.0,
      min_output_tokens: int | None = None,
      max_retries: int = 3,
      adc_file: pathlib.Path | None = None,
      token_supplier: Callable[[], tuple[str, float]] | None = None,
      thinking_budget: int | None = None,
      default_cached_content: str | None = None,
  ) -> None:
    """Initializes the Vertex AI Gemini batch client with TLS 1.2+ and ADC auth.

    Args:
      project_id: Google Cloud project ID (e.g., from `GOOGLE_CLOUD_PROJECT`).
      model_name: Target Gemini model slug.
      location: Vertex AI location (`'global'` by default for Gemini Enterprise).
      max_concurrency: Maximum parallel HTTPS requests per batch.
      timeout_sec: Per-request HTTPS socket timeout in seconds.
      min_output_tokens: Deprecated legacy parameter retained for signature
        compatibility; `maxOutputTokens` is unclamped per R3.
      max_retries: Bounded retry attempts on transient 429/5xx responses.
      adc_file: Optional path override for Google Cloud ADC credentials JSON.
      token_supplier: Optional injectable `(token, ttl_sec)` provider for tests.
      thinking_budget: Optional `thinkingBudget` override; falls back to
        `ARGUS_THINKING_BUDGET` env var (default `2048`).
      default_cached_content: Optional `cachedContents/...` resource attached to
        requests when `req.cached_content` is unset.
    """
    # Validate all identifiers up front so invalid configs fail closed immediately.
    self.provider_name = "vertex"
    self.project_id = validate_gcp_project_id(project_id)
    self.location = validate_gcp_location(location)
    self.model_name = normalize_gemini_model_name(model_name)
    self.endpoint_url = build_vertex_generate_url(
        self.project_id, self.location, self.model_name
    )
    self.max_concurrency = max(1, min(max_concurrency, 32))
    self._timeout_sec = timeout_sec
    # Retain `_min_output_tokens` only for caller backward compatibility; never clamp.
    self._min_output_tokens = min_output_tokens
    self._max_retries = max(1, min(max_retries, 6))
    # Resolve `thinking_budget` from constructor arg or `ARGUS_THINKING_BUDGET`.
    raw_budget: int | str | None = (
        thinking_budget
        if thinking_budget is not None
        else os.environ.get("ARGUS_THINKING_BUDGET")
    )
    self.thinking_budget = parse_thinking_budget(raw_budget)
    # Validate any client-bound `cachedContents/...` resource identifier up front.
    self.default_cached_content = (
        validate_cached_content_name(default_cached_content)
        if (default_cached_content and default_cached_content.strip())
        else None
    )

    # Enforce verified TLS 1.2+ certificates on all Google Cloud HTTPS traffic.
    self._ssl_ctx = ssl.create_default_context()
    self._ssl_ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    # Delegate OAuth2 ADC refresh and caching to `VertexTokenResolver`.
    self._auth = VertexTokenResolver(
        ssl_ctx=self._ssl_ctx,
        adc_file=adc_file,
        token_supplier=token_supplier,
    )
    # In-memory SHA-256 prompt cache and synchronization primitives for batching.
    self._cache: dict[str, tuple[str, int, int]] = {}
    self._cache_lock = threading.Lock()
    self._inflight_sem = threading.Semaphore(self.max_concurrency)

  def get_bearer_token(self) -> str:
    """Returns a valid OAuth2 bearer token from the underlying ADC resolver."""
    # Expose token retrieval cleanly for `ContextCacheAdapter` lifecycle calls.
    return self._auth.get_bearer_token()

  def invalidate_token(self) -> None:
    """Evicts any cached OAuth2 bearer token following an HTTP 401 response."""
    # Allow `ContextCacheAdapter` to trigger coordinated 401 token eviction.
    self._auth.invalidate_token()

  def _cache_key(self, req: LLMCompletionRequest) -> str:
    """Computes SHA-256 digest over model, thinking budget, cache ID, and text."""
    # Include `eff_budget` and `active_cache` so distinct configs never collide.
    eff_budget = resolve_thinking_budget(
        req.strategy_tag,
        self.thinking_budget,
        req.thinking_budget,
    )
    raw_cache = (
        req.cached_content
        if req.cached_content is not None
        else self.default_cached_content
    )
    active_cache = (
        validate_cached_content_name(raw_cache)
        if (raw_cache and raw_cache.strip())
        else ""
    )
    # Omit inline system prompt from digest when superseded by `cachedContent`.
    eff_sys = "" if active_cache else req.system_prompt
    # Normalize IEEE-754 `-0.0` to `0.0` so greedy temperatures hash identically.
    # Rounding prior to zero-check prevents `-0.0001` from formatting as `-0.000`.
    temp_val = round(coerce_finite_float(req.temperature, 0.0), 3)
    norm_temp = 0.0 if temp_val == 0.0 else temp_val
    raw = (
        f"vertex|{self.project_id}|{self.location}|{self.model_name}|"
        f"{norm_temp:.3f}|tb={eff_budget}|cache={active_cache}|"
        f"{len(eff_sys)}:{eff_sys}|{req.user_prompt}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

  def _post_generate_once(
      self,
      payload_bytes: bytes,
      bearer_token: str,
  ) -> dict[str, object]:
    """Executes one HTTPS POST call to Vertex AI `generateContent`."""
    # Include `x-goog-user-project` so Gemini Enterprise bills `self.project_id`.
    # Never log headers because `Authorization` holds live bearer material.
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {bearer_token}",
        "x-goog-user-project": self.project_id,
    }
    # Construct explicit POST request targeting validated `self.endpoint_url`.
    http_req = urllib.request.Request(
        self.endpoint_url,
        data=payload_bytes,
        headers=headers,
        method="POST",
    )
    # Bound concurrent open HTTPS sockets across all active worker threads.
    with self._inflight_sem:
      with urllib.request.urlopen(
          http_req, timeout=self._timeout_sec, context=self._ssl_ctx
      ) as resp:
        parsed = json.loads(resp.read().decode("utf-8"))
    if not isinstance(parsed, dict):
      raise LLMAdapterTransportError("Vertex Gemini returned non-object JSON.")
    return parsed

  def _send_single_request(
      self,
      req: LLMCompletionRequest,
  ) -> tuple[str, int, int]:
    """Serializes one request and executes it with bounded transient retries."""
    # Build unclamped R3/R1 `GenerateContentRequest` payload dictionary.
    body_dict = build_vertex_generate_payload(
        req,
        configured_thinking_budget=self.thinking_budget,
        default_cached_content=self.default_cached_content,
    )
    payload_bytes = json.dumps(body_dict).encode("utf-8")
    bearer_token = self._auth.get_bearer_token()
    refreshed_on_401 = False

    # Retry transient 429/5xx, socket timeouts, or a single aged-token HTTP 401.
    last_err: Exception | None = None
    for attempt in range(self._max_retries):
      try:
        resp_json = self._post_generate_once(payload_bytes, bearer_token)
        return extract_gemini_completion_triple(resp_json, len(req.user_prompt))
      except urllib.error.HTTPError as err:
        last_err = err
        if err.code == 401 and not refreshed_on_401 and attempt + 1 < self._max_retries:
          # Evict expired OAuth2 bearer token and retry immediately once.
          refreshed_on_401 = True
          self._auth.invalidate_token()
          bearer_token = self._auth.get_bearer_token()
          continue
        if err.code not in _RETRYABLE_HTTP_CODES or attempt + 1 >= self._max_retries:
          _LOGGER.warning(
              "Vertex Gemini request %s failed with HTTP %d.",
              req.request_id,
              err.code,
          )
          raise LLMAdapterTransportError(
              f"Vertex Gemini request '{req.request_id}' failed (HTTP {err.code})."
          ) from err
      except (
          urllib.error.URLError,
          http.client.HTTPException,
          OSError,
          TimeoutError,
          json.JSONDecodeError,
          UnicodeDecodeError,
      ) as err:
        # Wrap `HTTPException` (`BadStatusLine`, `IncompleteRead`,
        # `RemoteDisconnected`), mid-stream `OSError`, and decode errors so
        # transient drops retry cleanly and Stage 1 falls back to the FSM.
        last_err = err
        # Fail closed with chained `LLMAdapterTransportError` on final attempt.
        if attempt + 1 >= self._max_retries:
          _LOGGER.warning("Vertex Gemini request %s failed transport.", req.request_id)
          raise LLMAdapterTransportError(
              f"Vertex Gemini request '{req.request_id}' failed transport."
          ) from err
      except (KeyError, IndexError, TypeError, ValueError) as err:
        # Wrap structural/type errors immediately so Stage 1 falls back to FSM.
        raise LLMAdapterTransportError(
            f"Malformed Vertex response for '{req.request_id}'."
        ) from err

      # Apply exponential backoff with OS CSPRNG jitter on transient 429/5xx hits.
      jitter_sec = secrets.randbelow(500) / 1000.0
      time.sleep((2.0**attempt) + jitter_sec)

    raise LLMAdapterTransportError(
        f"Vertex Gemini request '{req.request_id}' exhausted retries."
    ) from last_err

  def batch_complete(
      self,
      requests: Sequence[LLMCompletionRequest],
  ) -> tuple[LLMCompletionResponse, ...]:
    """Executes `requests` concurrently with SHA-256 prompt deduplication.

    Args:
      requests: Sequence of LLMCompletionRequest items to execute.

    Returns:
      Tuple of LLMCompletionResponse objects aligned 1-to-1 with `requests`.
    """
    if not requests:
      return ()

    # Partition batch into unique uncached prompts under lock so duplicates cost 0.
    pending_by_hash: dict[str, LLMCompletionRequest] = {}
    req_hashes: list[str] = []
    cached_hits: set[str] = set()

    with self._cache_lock:
      for req in requests:
        digest = self._cache_key(req)
        req_hashes.append(digest)
        if digest in self._cache:
          cached_hits.add(digest)
        elif digest not in pending_by_hash:
          pending_by_hash[digest] = req

    # Warm OAuth token once before spawning parallel worker threads.
    if pending_by_hash:
      self._auth.get_bearer_token()
      unique_hashes = list(pending_by_hash.keys())
      unique_reqs = [pending_by_hash[h] for h in unique_hashes]
      workers = min(self.max_concurrency, len(unique_reqs))
      with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(self._send_single_request, unique_reqs))
      with self._cache_lock:
        for digest, triple in zip(unique_hashes, results, strict=True):
          self._cache[digest] = triple

    # Reconstruct ordered responses aligned 1-to-1 with caller input sequence.
    responses: list[LLMCompletionResponse] = []
    with self._cache_lock:
      for req, digest in zip(requests, req_hashes, strict=True):
        raw_text, p_tok, c_tok = self._cache[digest]
        is_hit = digest in cached_hits
        responses.append(
            LLMCompletionResponse(
                request_id=req.request_id,
                sql_text=extract_sql_from_completion(raw_text),
                raw_text=raw_text,
                prompt_tokens=0 if is_hit else p_tok,
                completion_tokens=0 if is_hit else c_tok,
                from_cache=is_hit,
                cached_prompt_tokens=p_tok if is_hit else 0,
                cached_completion_tokens=c_tok if is_hit else 0,
            )
        )
    return tuple(responses)
