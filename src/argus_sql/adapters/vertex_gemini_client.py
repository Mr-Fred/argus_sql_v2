"""Vertex AI / Gemini Enterprise batch-optimized HTTPS transport adapter.

Executes SQL generation batches concurrently via `ThreadPoolExecutor.map` over
Vertex AI's public `generateContent` REST endpoint (`aiplatform.googleapis.com`)
using Python's standard library, TLS 1.2+, SHA-256 prompt deduplication, and
in-memory OAuth2 Application Default Credentials (zero third-party SDKs).
"""

from collections.abc import Callable
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import pathlib
import secrets
import ssl
import threading
import time
import urllib.error
import urllib.request

from argus_sql.adapters.exceptions import LLMAdapterTransportError
from argus_sql.adapters.llm_batch_client import extract_sql_from_completion
from argus_sql.adapters.types import LLMCompletionRequest
from argus_sql.adapters.types import LLMCompletionResponse
from argus_sql.adapters.vertex_auth import VertexTokenResolver
from argus_sql.adapters.vertex_auth import build_vertex_generate_url
from argus_sql.adapters.vertex_auth import normalize_gemini_model_name
from argus_sql.adapters.vertex_auth import validate_gcp_location
from argus_sql.adapters.vertex_auth import validate_gcp_project_id

# Module-level logger following Google Python Style Guide lazy-% conventions.
_LOGGER = logging.getLogger(__name__)

# Transient upstream HTTP status codes eligible for bounded exponential retry.
_RETRYABLE_HTTP_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})


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
    LLMAdapterTransportError: If `candidates` is missing or malformed.
  """
  # Validate top-level `candidates` array before indexing model output.
  candidates = body.get("candidates")
  if not isinstance(candidates, list) or not candidates:
    raise LLMAdapterTransportError("Vertex Gemini response lacked candidates.")

  first = candidates[0]
  if not isinstance(first, dict):
    raise LLMAdapterTransportError("Malformed Vertex candidate object.")
  content = first.get("content", {})
  parts = content.get("parts", []) if isinstance(content, dict) else []
  if not isinstance(parts, list):
    raise LLMAdapterTransportError("Malformed Vertex candidate parts list.")

  # Separate final SQL parts from internal Gemini 2.5/3.x `thought=True` parts.
  visible_chunks: list[str] = []
  fallback_chunks: list[str] = []
  for part in parts:
    if isinstance(part, dict) and isinstance(part.get("text"), str):
      text_piece = part["text"]
      fallback_chunks.append(text_piece)
      if not bool(part.get("thought", False)):
        visible_chunks.append(text_piece)

  # Prefer non-thought text parts; fall back only if model emitted one raw part.
  raw_out = "\n".join(visible_chunks or fallback_chunks).strip()
  usage = body.get("usageMetadata", {})
  if isinstance(usage, dict):
    p_tok = int(usage.get("promptTokenCount", prompt_char_len // 4))
    # Sum visible candidate tokens and reasoning thought tokens for true cost.
    c_tok = int(usage.get("candidatesTokenCount", len(raw_out) // 4)) + int(
        usage.get("thoughtsTokenCount", 0)
    )
  else:
    p_tok = prompt_char_len // 4
    c_tok = len(raw_out) // 4
  return (raw_out, p_tok, c_tok)


class VertexGeminiBatchClient:
  """Zero-dependency Vertex AI / Gemini Enterprise `BatchLLMPort` adapter.

  Attributes:
    project_id: Validated Google Cloud project ID used for Vertex AI billing.
    location: Validated Vertex AI location (`'global'` or regional).
    model_name: Normalized Gemini model identifier (e.g., `'gemini-2.5-flash'`).
    endpoint_url: Constructed HTTPS `generateContent` URL.
    max_concurrency: Bounded worker pool size for parallel batch execution.
  """

  def __init__(
      self,
      project_id: str,
      model_name: str = "gemini-2.5-flash",
      location: str = "global",
      max_concurrency: int = 8,
      timeout_sec: float = 60.0,
      min_output_tokens: int = 2048,
      max_retries: int = 3,
      adc_file: pathlib.Path | None = None,
      token_supplier: Callable[[], tuple[str, float]] | None = None,
  ) -> None:
    """Initializes the Vertex AI Gemini batch client with TLS 1.2+ and ADC auth.

    Args:
      project_id: Google Cloud project ID (e.g., from `GOOGLE_CLOUD_PROJECT`).
      model_name: Target Gemini model slug.
      location: Vertex AI location (`'global'` by default for Gemini Enterprise).
      max_concurrency: Maximum parallel HTTPS requests per batch.
      timeout_sec: Per-request HTTPS socket timeout in seconds.
      min_output_tokens: Floor on `maxOutputTokens` so thinking models finish SQL.
      max_retries: Bounded retry attempts on transient 429/5xx responses.
      adc_file: Optional path override for Google Cloud ADC credentials JSON.
      token_supplier: Optional injectable `(token, ttl_sec)` provider for tests.
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
    # Thinking-capable Gemini models share `maxOutputTokens` across thoughts + SQL.
    self._min_output_tokens = max(256, min_output_tokens)
    self._max_retries = max(1, min(max_retries, 6))

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

  def _cache_key(self, req: LLMCompletionRequest) -> str:
    """Computes SHA-256 digest over model, decoding params, and prompt text."""
    # Include effective output budget and temperature in the deduplication key.
    eff_max = max(req.max_tokens, self._min_output_tokens)
    raw = (
        f"vertex|{self.project_id}|{self.location}|{self.model_name}|"
        f"{req.temperature:.3f}|{eff_max}|{req.system_prompt}|{req.user_prompt}"
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
    # Apply `min_output_tokens` floor so Gemini thinking budgets cannot starve SQL.
    eff_max_tokens = max(req.max_tokens, self._min_output_tokens)
    body_dict: dict[str, object] = {
        "contents": [{"role": "user", "parts": [{"text": req.user_prompt}]}],
        "generationConfig": {
            "temperature": req.temperature,
            "maxOutputTokens": eff_max_tokens,
        },
    }
    # Attach `systemInstruction` without `role` per canonical Vertex REST schema.
    if req.system_prompt.strip():
      body_dict["systemInstruction"] = {
          "parts": [{"text": req.system_prompt}],
      }
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
      except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
        last_err = err
        if attempt + 1 >= self._max_retries:
          _LOGGER.warning("Vertex Gemini request %s failed transport.", req.request_id)
          raise LLMAdapterTransportError(
              f"Vertex Gemini request '{req.request_id}' failed transport."
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
            )
        )
    return tuple(responses)
