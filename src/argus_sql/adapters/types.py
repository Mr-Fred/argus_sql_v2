"""Typed request, response, and port contracts for batch LLM SQL generation."""

from collections.abc import Sequence
import dataclasses
from typing import Protocol


# Immutable single prompt payload submitted inside a batched completion call.
@dataclasses.dataclass(frozen=True)
class LLMCompletionRequest:
  """Single prompt completion specification within a batched LLM call.

  Attributes:
    request_id: Caller-assigned correlation identifier.
    system_prompt: System instruction framing SQL dialect and constraints.
    user_prompt: Schema DDL, question, evidence, and optional repair feedback.
    strategy_tag: Synthesis strategy name ('zero_shot', 'decomposed_cot', etc.).
    temperature: Sampling temperature (`0.0` for greedy deterministic decoding).
    max_tokens: Maximum completion tokens allowed.
  """

  # Greedy decoding (`temperature=0.0`) maximizes SQL determinism by default.
  request_id: str
  system_prompt: str
  user_prompt: str
  strategy_tag: str = "zero_shot"
  temperature: float = 0.0
  max_tokens: int = 512


# Normalized completion response returned by any BatchLLMPort implementation.
# `from_cache=True` zeroes token counts so deduplicated hits are billed once.
@dataclasses.dataclass(frozen=True)
class LLMCompletionResponse:
  """Normalized SQL completion result paired with token accounting metadata.

  Attributes:
    request_id: Correlation identifier matching `LLMCompletionRequest`.
    sql_text: Extracted executable SQL query string.
    raw_text: Raw model output text prior to code-fence extraction.
    prompt_tokens: Input token count reported or estimated.
    completion_tokens: Output token count reported or estimated.
    from_cache: True when served from the SHA-256 prompt deduplication cache.
  """

  request_id: str
  sql_text: str
  raw_text: str
  prompt_tokens: int
  completion_tokens: int
  from_cache: bool = False


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
