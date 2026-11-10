"""Optional fast `flash-lite` structured LLM intent extractor for Stage 1.

Dispatches batched JSON intent-extraction prompts over `BatchLLMPort` to
extract `{canonical_question, entities, expected_projection_arity,
requires_floating_division, requires_extrema_order, complexity}`, with
automatic, zero-error fallback to the deterministic FSM canonicalizer in
hermetic/offline runs or on malformed model responses.
"""

from collections.abc import Sequence
import json
import logging
import re

from argus_sql.adapters.exceptions import AdapterError
from argus_sql.adapters.types import BatchLLMPort
from argus_sql.adapters.types import LLMCompletionRequest
from argus_sql.contracts.types import IntentFrame
from argus_sql.contracts.types import QueryDifficulty
from argus_sql.linking.intent_canonicalizer import canonicalize_intent
from argus_sql.linking.types import IntentExtractorPort

# Module-level logger following Google Python Style Guide lazy-% conventions.
_LOGGER = logging.getLogger(__name__)

# Regex extracting optional markdown ```json ... ``` code fences around payloads.
_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.IGNORECASE | re.DOTALL,
)

# Compact system prompt instructing fast `flash-lite` models to emit pure JSON.
INTENT_EXTRACTION_SYSTEM_PROMPT = (
    "You are a fast natural-language-to-SQL intent extractor. "
    "Strip spoken ASR fillers/stutters, normalize spoken numbers, and return "
    "ONLY a valid JSON object with keys: "
    '"canonical_question" (string), '
    '"entities" (array of up to 16 entity/literal strings), '
    '"expected_projection_arity" (integer 0..16, 0 if unconstrained narrative), '
    '"requires_floating_division" (boolean), '
    '"requires_extrema_order" (boolean), '
    '"complexity" ("simple" | "moderate" | "challenging").'
)

# Valid difficulty string literals mapped to `QueryDifficulty` enum members.
_DIFFICULTY_MAP: dict[str, QueryDifficulty] = {
    "simple": QueryDifficulty.SIMPLE,
    "moderate": QueryDifficulty.MODERATE,
    "challenging": QueryDifficulty.CHALLENGING,
}


def build_intent_extraction_request(
    request_id: str,
    question: str,
    evidence: str = "",
) -> LLMCompletionRequest:
  """Builds a lightweight structured JSON intent request for `BatchLLMPort`.

  Args:
    request_id: Caller-supplied correlation identifier for batch alignment.
    question: Raw user natural language question or ASR utterance.
    evidence: Optional external domain knowledge or formula note.

  Returns:
    Deterministic `LLMCompletionRequest` (`temperature=0.0`, `max_tokens=256`).
  """
  # Keep prompt compact so `flash-lite` responds with minimal latency/tokens.
  ev_block = f"\nEvidence: {evidence.strip()}" if evidence.strip() else ""
  user_prompt = f"Question: {question.strip()}{ev_block}\nJSON:"
  return LLMCompletionRequest(
      request_id=request_id,
      system_prompt=INTENT_EXTRACTION_SYSTEM_PROMPT,
      user_prompt=user_prompt,
      strategy_tag="intent_flash_lite",
      temperature=0.0,
      max_tokens=256,
  )


def _extract_json_object(raw_text: str) -> dict[str, object] | None:
  """Parses a JSON dictionary from raw or markdown-fenced model output."""
  stripped = raw_text.strip()
  if not stripped:
    return None
  # Prefer fenced ```json {...} ``` blocks when models wrap answers in markdown.
  fence_match = _JSON_FENCE_RE.search(stripped)
  candidate = fence_match.group(1).strip() if fence_match else stripped
  if not candidate.startswith("{"):
    # Attempt extracting the first braced object when brief prose precedes JSON.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end <= start:
      return None
    candidate = candidate[start : end + 1]
  # Reject malformed payloads cleanly so callers fall back to deterministic FSM.
  try:
    parsed = json.loads(candidate)
  except (json.JSONDecodeError, TypeError, ValueError):
    return None
  return parsed if isinstance(parsed, dict) else None


def parse_intent_json_response(
    raw_text: str,
    fallback: IntentFrame,
) -> IntentFrame | None:
  """Validates structured JSON intent fields against `IntentFrame` invariants.

  Returns `None` when `raw_text` is non-JSON (such as SQL emitted by offline
  synthesizers) or lacks expected schema structure so callers fall back safely.

  Args:
    raw_text: Unmodified text returned in `LLMCompletionResponse.raw_text`.
    fallback: Deterministic FSM `IntentFrame` supplying defaults and guards.

  Returns:
    Validated `IntentFrame` or `None` when `raw_text` is unusable.
  """
  payload = _extract_json_object(raw_text)
  if payload is None:
    return None

  # Require at least one recognized Stage 1 schema field to avoid false JSON hits.
  expected_keys = {
      "canonical_question",
      "entities",
      "expected_projection_arity",
      "requires_floating_division",
      "requires_extrema_order",
      "complexity",
  }
  if not any(k in payload for k in expected_keys):
    return None

  # 1. Canonicalized question text (must remain non-empty string).
  raw_cq = payload.get("canonical_question")
  canonical_q = (
      raw_cq.strip()
      if isinstance(raw_cq, str) and raw_cq.strip()
      else fallback.canonical_question
  )

  # 2. Candidate entity spans (deduplicated non-empty strings, capped at 16).
  raw_entities = payload.get("entities")
  if isinstance(raw_entities, list):
    clean_entities: list[str] = []
    # Preserve appearance order while dropping blanks and duplicates.
    for item in raw_entities:
      if isinstance(item, str):
        val = item.strip()
        if val and val not in clean_entities:
          clean_entities.append(val)
    entities = tuple(clean_entities[:16])
  else:
    entities = fallback.entities

  # 3. Projection arity (`bool` subclasses `int` in Python, so reject `bool`).
  raw_arity = payload.get("expected_projection_arity")
  if (
      isinstance(raw_arity, int)
      and not isinstance(raw_arity, bool)
      and 0 <= raw_arity <= 16
  ):
    arity = raw_arity
  else:
    arity = fallback.expected_projection_arity

  # 4. Boolean semantic cues (`requires_floating_division`, `requires_extrema_order`).
  raw_div = payload.get("requires_floating_division")
  needs_div = (
      raw_div if isinstance(raw_div, bool) else fallback.requires_floating_division
  )
  raw_ext = payload.get("requires_extrema_order")
  needs_extrema = (
      raw_ext if isinstance(raw_ext, bool) else fallback.requires_extrema_order
  )

  # 5. Difficulty routing tier (`simple` | `moderate` | `challenging`).
  raw_comp = payload.get("complexity")
  complexity = (
      _DIFFICULTY_MAP.get(raw_comp.strip().lower(), fallback.complexity)
      if isinstance(raw_comp, str)
      else fallback.complexity
  )

  # Preserve deterministic formula-lock and follow-up heuristics where compatible.
  has_followup = (
      arity >= 2
      if arity != fallback.expected_projection_arity
      else fallback.has_followup_projection
  )
  # Only preserve strict 1-column formula lock when predicted arity stays 1.
  strict_single = bool(fallback.strict_single_column and arity == 1)
  return IntentFrame(
      raw_question=fallback.raw_question,
      canonical_question=canonical_q,
      disfluencies_removed=fallback.disfluencies_removed,
      entities=entities,
      requires_floating_division=needs_div,
      requires_extrema_order=needs_extrema,
      expected_projection_arity=arity,
      complexity=complexity,
      strict_single_column=strict_single,
      has_followup_projection=has_followup,
      evidence=fallback.evidence,
  )


class FlashLiteIntentExtractor:
  """Stage 1 structured LLM intent extractor with automatic FSM fallback.

  Executes intent extraction concurrently via `BatchLLMPort.batch_complete` and
  automatically falls back to the deterministic FSM canonicalizer when `llm` is
  unset, offline/hermetic, raises transport errors, or returns non-JSON text.
  """

  def __init__(
      self,
      llm: BatchLLMPort | None = None,
      enabled: bool = True,
  ) -> None:
    """Initializes the intent extractor around an optional `BatchLLMPort`.

    Args:
      llm: Optional fast LLM batch adapter (e.g., `gemini-2.5-flash-lite`).
      enabled: Feature toggle; when False, skips LLM calls and uses FSM only.
    """
    self._llm = llm
    self._enabled = enabled

  def extract_intent_batch_with_usage(
      self,
      items: Sequence[tuple[str, str]],
  ) -> tuple[tuple[IntentFrame, ...], tuple[int, ...]]:
    """Extracts Stage 1 `IntentFrame` items and per-item logical token counts.

    Args:
      items: Sequence of `(question, evidence)` tuples to canonicalize.

    Returns:
      Pair `(frames, token_counts)` aligned 1-to-1 with `items`.
    """
    if not items:
      return ((), ())

    # Always build deterministic FSM baselines first (`<1 ms` per utterance).
    fallbacks = tuple(canonicalize_intent(q, ev) for q, ev in items)
    if not self._enabled or self._llm is None:
      return (fallbacks, (0,) * len(items))

    # Dispatch all intent extraction prompts in a single batched LLM request.
    reqs = tuple(
        build_intent_extraction_request(f"intent-{idx}", q, ev)
        for idx, (q, ev) in enumerate(items)
    )
    # Guard network/auth/transport faults so Stage 1 never aborts the pipeline.
    try:
      resps = self._llm.batch_complete(reqs)
    except (AdapterError, OSError, TimeoutError, ValueError, RuntimeError) as err:
      _LOGGER.warning(
          "Flash-lite intent extraction failed (%s); falling back to FSM.",
          type(err).__name__,
      )
      return (fallbacks, (0,) * len(items))

    if len(resps) != len(fallbacks):
      _LOGGER.warning(
          "Flash-lite intent batch size mismatch (%d vs %d); falling back to FSM.",
          len(resps),
          len(fallbacks),
      )
      return (fallbacks, (0,) * len(items))

    # Validate each JSON payload; fall back cleanly on SQL or malformed output.
    frames: list[IntentFrame] = []
    tokens: list[int] = []
    # Keep 1-to-1 alignment between batch responses and deterministic fallbacks.
    for resp, fb in zip(resps, fallbacks, strict=True):
      parsed = parse_intent_json_response(resp.raw_text, fb)
      if parsed is not None:
        frames.append(parsed)
      else:
        frames.append(fb)
      # Record logical prompt+completion tokens even when JSON fallback triggers.
      tokens.append(resp.logical_tokens)
    return (tuple(frames), tuple(tokens))

  def extract_intent_batch(
      self,
      items: Sequence[tuple[str, str]],
  ) -> tuple[IntentFrame, ...]:
    """Extracts `IntentFrame` contracts for `(question, evidence)` batches.

    Args:
      items: Sequence of `(question, evidence)` pairs.

    Returns:
      Tuple of `IntentFrame` items aligned 1-to-1 with `items`.
    """
    frames, _ = self.extract_intent_batch_with_usage(items)
    return frames

  def extract_intent(self, question: str, evidence: str = "") -> IntentFrame:
    """Extracts a single `IntentFrame` with automatic deterministic fallback.

    Args:
      question: Raw user natural language question or ASR transcript.
      evidence: Optional external domain knowledge / formula note.

    Returns:
      Structured `IntentFrame` contract.
    """
    return self.extract_intent_batch(((question, evidence),))[0]


def canonicalize_intent_with_optional_llm(
    question: str,
    evidence: str = "",
    extractor: BatchLLMPort | IntentExtractorPort | None = None,
) -> tuple[IntentFrame, int]:
  """Canonicalizes intent via optional `flash-lite` port or deterministic FSM.

  Args:
    question: Raw user question or spoken ASR transcript.
    evidence: Optional external domain knowledge / formula note.
    extractor: Optional `BatchLLMPort` or `IntentExtractorPort` instance.

  Returns:
    Tuple `(intent_frame, logical_tokens_consumed)`.
  """
  if extractor is None:
    return (canonicalize_intent(question, evidence), 0)
  if isinstance(extractor, FlashLiteIntentExtractor):
    frames, toks = extractor.extract_intent_batch_with_usage(
        ((question, evidence),)
    )
    return (frames[0], toks[0])
  if hasattr(extractor, "extract_intent_batch"):
    # Support arbitrary custom `IntentExtractorPort` implementations directly.
    frames = extractor.extract_intent_batch(((question, evidence),))
    return (frames[0], 0)
  # Wrap raw `BatchLLMPort` instances in `FlashLiteIntentExtractor` cleanly.
  wrapper = FlashLiteIntentExtractor(llm=extractor)
  frames, toks = wrapper.extract_intent_batch_with_usage(((question, evidence),))
  return (frames[0], toks[0])
