"""HTTPS batch-optimized OpenAI-compatible LLM transport adapter.

Executes prompt batches concurrently via `ThreadPoolExecutor.map` with SHA-256
request deduplication, strict TLS/loopback endpoint validation, and environment
variable credential resolution (zero hardcoded keys or N+1 HTTP loops).
"""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import os
import re
import ssl
import threading
import urllib.error
import urllib.parse
import urllib.request

from argus_sql.adapters.exceptions import LLMAdapterAuthError
from argus_sql.adapters.exceptions import LLMAdapterTransportError
from argus_sql.adapters.types import LLMCompletionRequest
from argus_sql.adapters.types import LLMCompletionResponse

# Module-level logger following Google Python Style Guide conventions.
_LOGGER = logging.getLogger(__name__)

# Regex extracting fenced ```sql ... ``` blocks from LLM markdown completions.
_SQL_FENCE_RE = re.compile(
    r"```(?:sql|sqlite)?\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


def extract_sql_from_completion(raw_text: str) -> str:
  """Extracts a clean SQL statement from raw or markdown-fenced model output.

  Args:
    raw_text: Raw completion string returned by a language model.

  Returns:
    Trimmed single SQL statement ready for read-only sandbox execution.
  """
  fence_match = _SQL_FENCE_RE.search(raw_text)
  candidate = fence_match.group(1).strip() if fence_match else raw_text.strip()

  # Strip leading 'SQL:' labels if emitted outside markdown code fences.
  candidate = re.sub(r"^(?:sql|sqlite)\s*:\s*", "", candidate, flags=re.IGNORECASE)
  # Locate first SELECT or WITH keyword when brief reasoning precedes raw SQL.
  kw_match = re.search(r"\b(?:SELECT|WITH)\b", candidate, re.IGNORECASE)
  if kw_match is not None and kw_match.start() > 0:
    candidate = candidate[kw_match.start() :]

  return candidate.strip().rstrip(";").strip()


def validate_endpoint_security(endpoint_url: str) -> str:
  """Validates that `endpoint_url` uses HTTPS or explicit localhost loopback.

  Args:
    endpoint_url: Target chat-completions HTTP(S) endpoint URL.

  Returns:
    Normalized endpoint URL string.

  Raises:
    LLMAdapterAuthError: If URL uses unencrypted remote HTTP or `0.0.0.0`.
  """
  parsed = urllib.parse.urlparse(endpoint_url)
  host = (parsed.hostname or "").lower()
  # Never permit binding or connecting to wildcard `0.0.0.0` addresses.
  if host == "0.0.0.0" or not host:
    raise LLMAdapterAuthError("Invalid or wildcard LLM endpoint host.")

  # Accept HTTPS globally or loopback HTTP strictly for local vLLM/Ollama runs.
  if parsed.scheme == "https":
    return endpoint_url
  if parsed.scheme == "http" and host in {"127.0.0.1", "localhost", "::1"}:
    return endpoint_url

  # Fail closed on any unencrypted non-loopback endpoint URL.
  raise LLMAdapterAuthError(
      "Remote LLM endpoints must use encrypted 'https://' transport."
  )


class OpenAICompatibleBatchClient:
  """Batch LLM adapter compatible with public OpenAI/vLLM REST endpoints.

  Attributes:
    endpoint_url: Validated HTTPS or loopback `/v1/chat/completions` URL.
    model_name: Target model identifier (e.g., 'gpt-4o', 'Qwen2.5-Coder-32B').
    max_concurrency: Bounded worker pool size for concurrent batch execution.
  """

  def __init__(
      self,
      model_name: str,
      endpoint_url: str = "https://api.openai.com/v1/chat/completions",
      api_key_env: str = "ARGUS_LLM_API_KEY",
      max_concurrency: int = 8,
      timeout_sec: float = 45.0,
      api_key: str | None = None,
  ) -> None:
    """Initializes the batch client and validates transport security.

    Args:
      model_name: Public model identifier requested in completion payloads.
      endpoint_url: HTTPS or localhost chat completions endpoint URL.
      api_key_env: Environment variable holding the bearer credential.
      max_concurrency: Maximum parallel HTTPS requests per batch.
      timeout_sec: Per-request socket timeout in seconds.
      api_key: Optional runtime credential resolved from a local `.env` file.

    Raises:
      LLMAdapterAuthError: If remote HTTPS endpoint lacks an API key in env.
    """
    self.provider_name = "openai"
    self.endpoint_url = validate_endpoint_security(endpoint_url)
    self.model_name = model_name
    self.max_concurrency = max(1, min(max_concurrency, 32))
    self._timeout_sec = timeout_sec

    # Resolve API credential from injected `.env` value or process environment.
    key = (
        (api_key or "").strip()
        or os.environ.get(api_key_env, "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    is_loopback = self.endpoint_url.startswith("http://")
    if not key and not is_loopback:
      raise LLMAdapterAuthError(
          f"Missing API credential in environment variable '{api_key_env}'."
      )
    self._api_key = key
    # In-memory SHA-256 prompt cache and synchronization primitives for batching.
    self._cache: dict[str, tuple[str, int, int]] = {}
    self._cache_lock = threading.Lock()
    self._inflight_sem = threading.Semaphore(self.max_concurrency)
    # Enforce TLS 1.2+ certificate verification on all remote HTTPS endpoints.
    self._ssl_ctx = ssl.create_default_context()
    self._ssl_ctx.minimum_version = ssl.TLSVersion.TLSv1_2

  def _cache_key(self, req: LLMCompletionRequest) -> str:
    """Computes SHA-256 digest over model, temperature, and prompt text."""
    # Include sampling temperature and max_tokens so distinct decodes stay split.
    raw = (
        f"{self.model_name}|{req.temperature:.3f}|{req.max_tokens}|"
        f"{req.system_prompt}|{req.user_prompt}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

  def _send_single_http_request(
      self,
      req: LLMCompletionRequest,
  ) -> tuple[str, int, int]:
    """Executes one HTTPS POST request inside a worker thread."""
    # Serialize standard OpenAI `/v1/chat/completions` JSON request body.
    payload = json.dumps({
        "model": self.model_name,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "messages": [
            {"role": "system", "content": req.system_prompt},
            {"role": "user", "content": req.user_prompt},
        ],
    }).encode("utf-8")

    # Attach Authorization header only in memory; never echo headers in logs.
    headers = {"Content-Type": "application/json"}
    if self._api_key:
      headers["Authorization"] = f"Bearer {self._api_key}"

    http_req = urllib.request.Request(
        self.endpoint_url,
        data=payload,
        headers=headers,
        method="POST",
    )
    ctx = self._ssl_ctx if self.endpoint_url.startswith("https://") else None
    try:
      with self._inflight_sem:
        with urllib.request.urlopen(
            http_req, timeout=self._timeout_sec, context=ctx
        ) as resp:
          body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
      # Never log authorization headers or sensitive keys on transport errors.
      _LOGGER.warning("LLM request %s failed transport.", req.request_id)
      raise LLMAdapterTransportError(
          f"Completion request '{req.request_id}' failed."
      ) from err

    # Extract completion text and fall back to ~4 chars/token if usage omitted.
    try:
      raw_out = str(body["choices"][0]["message"]["content"])
      usage = body.get("usage", {})
      p_tok = int(usage.get("prompt_tokens", len(req.user_prompt) // 4))
      c_tok = int(usage.get("completion_tokens", len(raw_out) // 4))
    except (KeyError, IndexError, TypeError, ValueError) as err:
      raise LLMAdapterTransportError("Malformed completion JSON schema.") from err

    return (raw_out, p_tok, c_tok)

  def batch_complete(
      self,
      requests: Sequence[LLMCompletionRequest],
  ) -> tuple[LLMCompletionResponse, ...]:
    """Executes `requests` as a deduplicated concurrent batch.

    Args:
      requests: Sequence of LLMCompletionRequest items to execute.

    Returns:
      Tuple of LLMCompletionResponse objects aligned 1-to-1 with `requests`.
    """
    if not requests:
      return ()

    # Partition batch into unique uncached requests under lock so duplicates cost 0.
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

    # Dispatch all unique uncached prompts concurrently via ThreadPoolExecutor.
    if pending_by_hash:
      unique_hashes = list(pending_by_hash.keys())
      unique_reqs = [pending_by_hash[h] for h in unique_hashes]
      workers = min(self.max_concurrency, len(unique_reqs))
      with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(self._send_single_http_request, unique_reqs))
      with self._cache_lock:
        for digest, triple in zip(unique_hashes, results, strict=True):
          self._cache[digest] = triple

    # Reconstruct ordered responses aligned with the caller's input sequence.
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
