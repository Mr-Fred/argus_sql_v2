"""Typed request, response, cache, and port contracts for batch LLM adapters."""

from collections.abc import Sequence
import dataclasses
from typing import Protocol


# Immutable single prompt payload submitted inside a batched completion call.
# Optional `thinking_budget` and `cached_content` fields preserve full R1/R3
# backward compatibility across all existing 6-argument pipeline call sites.
@dataclasses.dataclass(frozen=True)
class LLMCompletionRequest:
  """Single prompt completion specification within a batched LLM call.

  Attributes:
    request_id: Caller-assigned correlation identifier.
    system_prompt: System instruction framing SQL dialect and constraints.
    user_prompt: Schema DDL, question, evidence, and optional repair feedback.
    strategy_tag: Synthesis strategy name ('zero_shot', 'decomposed_cot', etc.).
    temperature: Sampling temperature (`0.0` for greedy deterministic decoding).
    max_tokens: Legacy maximum token hint (unclamped on Vertex Gemini R3).
    thinking_budget: Optional per-request Gemini `thinkingBudget` override.
    cached_content: Optional Vertex AI `cachedContents/...` resource path.
  """

  # Greedy decoding (`temperature=0.0`) maximizes SQL determinism by default.
  request_id: str
  system_prompt: str
  user_prompt: str
  strategy_tag: str = "zero_shot"
  temperature: float = 0.0
  max_tokens: int = 512
  thinking_budget: int | None = None
  cached_content: str | None = None


# Normalized completion response returned by any BatchLLMPort implementation.
# `from_cache=True` zeroes token counts so deduplicated hits are billed once.
@dataclasses.dataclass(frozen=True)
class LLMCompletionResponse:
  """Normalized SQL completion result paired with token accounting metadata.

  Attributes:
    request_id: Correlation identifier matching `LLMCompletionRequest`.
    sql_text: Extracted executable SQL query string.
    raw_text: Raw model output text prior to code-fence extraction.
    prompt_tokens: Input token count billed on this call (`0` when `from_cache`).
    completion_tokens: Output token count billed on this call (`0` when cached).
    from_cache: True when served from the SHA-256 prompt deduplication cache.
    cached_prompt_tokens: Underlying prompt tokens when `from_cache=True`.
    cached_completion_tokens: Underlying completion tokens when `from_cache=True`.
  """

  request_id: str
  sql_text: str
  raw_text: str
  prompt_tokens: int
  completion_tokens: int
  from_cache: bool = False
  cached_prompt_tokens: int = 0
  cached_completion_tokens: int = 0

  @property
  def effective_prompt_tokens(self) -> int:
    """Returns standalone input prompt tokens even when served from cache."""
    # Fall back to `cached_prompt_tokens` when cross-system dedup hits cache.
    if self.from_cache and self.cached_prompt_tokens > 0:
      return self.cached_prompt_tokens
    return self.prompt_tokens

  @property
  def effective_completion_tokens(self) -> int:
    """Returns standalone output completion tokens even when served from cache."""
    # Fall back to `cached_completion_tokens` on cross-system SHA-256 cache hits.
    if self.from_cache and self.cached_completion_tokens > 0:
      return self.cached_completion_tokens
    return self.completion_tokens

  @property
  def logical_tokens(self) -> int:
    """Returns total standalone prompt + completion tokens for a candidate."""
    # Combine effective input and output tokens for standalone baseline metrics.
    return self.effective_prompt_tokens + self.effective_completion_tokens


# Server-side Vertex AI CachedContent metadata handle scoped to one `db_id`.
@dataclasses.dataclass(frozen=True)
class VertexCachedContent:
  """Immutable handle describing an active Vertex AI `CachedContent` resource.

  Attributes:
    cache_name: Canonical `projects/.../locations/.../cachedContents/...` path.
    db_id: Benchmark database identifier cached in this resource.
    model_name: Normalized Gemini model slug bound to the cached prefix.
    ttl_seconds: Configured cache time-to-live in seconds (e.g., `1800`..`3600`).
    expire_time: RFC 3339 expiration timestamp returned by Vertex AI.
    total_token_count: Cached prefix token count reported in `usageMetadata`.
    system_prompt_sha256: SHA-256 hex digest of cached `systemInstruction` text.
  """

  # Store SHA-256 of `systemInstruction` to guard against prompt mismatches.
  cache_name: str
  db_id: str
  model_name: str
  ttl_seconds: int
  expire_time: str = ""
  total_token_count: int = 0
  system_prompt_sha256: str = ""


# Hexagonal port decoupling pipelines from specific cloud/local LLM SDKs.
class BatchLLMPort(Protocol):
  """Port interface requiring batched, deduplicated completion execution."""

  def batch_complete(
      self,
      requests: Sequence[LLMCompletionRequest],
  ) -> tuple[LLMCompletionResponse, ...]:
    """Completes a batch of prompt requests concurrently without N+1 loops.

    Args:
      requests: Sequence of LLMCompletionRequest items to execute.

    Returns:
      Tuple of LLMCompletionResponse items aligned 1-to-1 with `requests`.
    """
    ...
