"""Voice-aware natural language intent canonicalizer and complexity classifier.

Transforms user questions and ASR transcripts into structured IntentFrame
contracts capturing grounded entity candidates, expected projection arity,
scalar formula constraints, and analytical complexity routing tiers.
"""

import re

from argus_sql.contracts.types import IntentFrame
from argus_sql.contracts.types import QueryDifficulty
from argus_sql.linking.asr_normalizer import ENTITY_STOPWORDS
from argus_sql.linking.asr_normalizer import clean_asr_utterance

# Common spoken geographic/domain abbreviations mapped to canonical forms.
_SPOKEN_EXPANSIONS: tuple[tuple[str, str], ...] = (
    ("calif.", "California"),
    ("calif", "California"),
    ("cali", "California"),
    ("l.a. county", "Los Angeles"),
    ("la county", "Los Angeles"),
    ("ny", "New York"),
    ("sf", "San Francisco"),
)

# Keywords signaling floating-point division / rate / percentage semantics.
_DIVISION_CUE_RE = re.compile(
    r"\b(?:ratio|rate|percentage|percent|proportion|fraction|per\s+capita"
    r"|divided\s+by|share\s+of)\b|%",
    re.IGNORECASE,
)

# Keywords signaling ORDER BY ... LIMIT k or MIN/MAX superlative semantics.
_EXTREMA_CUE_RE = re.compile(
    r"\b(?:highest|lowest|top|bottom|most|least|largest|smallest|oldest"
    r"|youngest|earliest|latest|maximum|minimum)\b",
    re.IGNORECASE,
)

# Subquery / analytical complexity triggers for adaptive candidate routing.
_CHALLENGING_CUE_RE = re.compile(
    r"\b(?:among\s+those|more\s+than\s+the\s+average|less\s+than\s+the\s+average"
    r"|difference\s+between|compared\s+to|both\s+.*\s+and|neither)\b",
    re.IGNORECASE,
)

# Matches imperative `Rank <entities> by <metric>` analytical ranking prompts.
_IMPERATIVE_RANK_RE = re.compile(
    r"^(?:please\s+)?rank\s+(?!of\b)\S+[\s\S]*\bby\b",
    re.IGNORECASE,
)


def extract_raw_entities(question: str, evidence: str = "") -> tuple[str, ...]:
  """Extracts surface entity spans without Stage 1 voice/ASR canonicalization.

  Extracts quoted literals, multi-word capitalized proper nouns, alphanumeric
  domain identifiers, hyphen/adverb status phrases, and prepositional n-grams
  so non-voice value-grounding baselines (`B3_ValueMajorityVote`, `B4_LLMJudge`)
  are evaluated fairly on multi-word entities without leaking spoken shorthand
  expansions (`_SPOKEN_EXPANSIONS`) or ASR casing normalization.

  Args:
    question: User natural language question or raw ASR transcript.
    evidence: Optional external domain knowledge / formula specification.

  Returns:
    Tuple of up to 16 candidate entity/literal spans in appearance order.
  """
  spans: list[str] = []
  combined = f"{question} {evidence}".strip()

  # 1. Capture single- or double-quoted literal values verbatim.
  for match in re.finditer(r"['\"]([^'\"]{1,50})['\"]", combined):
    val = match.group(1).strip()
    if val and val not in spans:
      spans.append(val)

  # 2. Capture capitalized Proper Noun phrases after stripping leading stopwords
  # so 'Between DC' yields 'DC' while 'There' / 'Consider' are discarded.
  for match in re.finditer(
      r"\b([A-Z][a-zA-Z0-9&.-]+(?:\s+[A-Z][a-zA-Z0-9&.-]+){0,3})\b",
      combined,
  ):
    words = match.group(1).strip().split()
    while words and words[0].lower() in ENTITY_STOPWORDS:
      words.pop(0)
    phrase = " ".join(words)
    if phrase and phrase.lower() not in ENTITY_STOPWORDS and phrase not in spans:
      spans.append(phrase)

  # 3. Capture alphanumeric domain IDs ('TR004_8_9', 'tr004_8_9', 'TR008').
  for match in re.finditer(
      r"\b([a-zA-Z]{1,8}_[a-zA-Z0-9_]+|[a-zA-Z]{2,6}\d+[a-zA-Z0-9_]*)\b",
      combined,
  ):
    code_id = match.group(1).strip()
    if code_id not in spans:
      spans.append(code_id)

  # 4. Capture hyphen-prefixed codes ('czk-paying') and '-ly funded' statuses.
  for match in re.finditer(
      r"\b([a-zA-Z]{2,6})-(?:paying|based|funded|run|level|type)\b",
      question,
      re.IGNORECASE,
  ):
    code_tok = match.group(1).strip()
    if code_tok not in spans:
      spans.append(code_tok)
  for match in re.finditer(
      r"\b([a-zA-Z]{3,15}ly\s+(?:funded|owned|operated|managed|run|based))\b",
      question,
      re.IGNORECASE,
  ):
    status_phrase = match.group(1).strip()
    if status_phrase not in spans:
      spans.append(status_phrase)

  # 5. Capture n-grams after relational prepositions while skipping determiners.
  det_skip = {
      "the", "each", "all", "any", "this", "that", "these", "those",
      "their", "his", "her", "its", "our", "a", "an",
  }
  for match in re.finditer(
      r"\b(?:in|for|named|called|county|state|status|type|currency|segment"
      r"|country|tagged|code)\s+"
      r"([a-zA-Z][a-zA-Z0-9_-]{1,25}(?:\s+[a-zA-Z][a-zA-Z0-9_-]{2,25})?)",
      question,
      re.IGNORECASE,
  ):
    cand_words = [
        w for w in match.group(1).strip().split() if w.lower() not in det_skip
    ]
    if cand_words and cand_words[0].lower() not in ENTITY_STOPWORDS:
      cand = " ".join(cand_words)
      if cand not in spans:
        spans.append(cand)

  return tuple(spans[:16])


def _extract_entity_candidates(question: str, evidence: str) -> tuple[str, ...]:
  """Extracts surface entity spans plus Stage 1 spoken geographic expansions."""
  spans = list(extract_raw_entities(question, evidence))

  # 6. Add canonical forms for common spoken geographic abbreviations.
  lower_q = question.lower()
  for abbrev, expanded in _SPOKEN_EXPANSIONS:
    if re.search(rf"\b{re.escape(abbrev)}(?:\b|$)", lower_q):
      if expanded not in spans:
        spans.append(expanded)

  return tuple(spans[:16])


def _estimate_projection_arity(canonical_q: str) -> int:
  """Estimates requested output arity (`0` = unconstrained narrative prompt)."""
  lower = canonical_q.lower()
  # Explicit single-output directives override narrative context.
  if re.search(
      r"\b(?:only|just)\s+need\s+the\s+(?:single\s+)?(?:count|total|number|"
      r"average|percentage|rate|ratio|difference|score|value|name|id)\b",
      lower,
  ):
    return 1

  # Multi-sentence narrative briefings (>38 words or >=2 periods) frequently
  # specify custom 4..16 column schemas; leave unconstrained (`0`).
  body_no_end = canonical_q.rstrip(".?!")
  if len(canonical_q.split()) > 38 or body_no_end.count(".") >= 2:
    return 0

  # Expand compound multi-field shorthand; leave plural 'full names' unconstrained
  # because schemas represent full names across 1, 2, 3, or 6 columns.
  if "full communication address" in lower or "complete address" in lower:
    return 4
  if "full names" in lower:
    return 0

  # Imperative `Rank <entities> by <metric>` projects `(entity, metric, rank)`.
  if _IMPERATIVE_RANK_RE.match(canonical_q):
    return 3

  # Detect multi-pair comparison tails ('between A and B, B and C, and C and A').
  pair_tail = re.search(
      r"\bbetween\s+([^?.!]+,\s*[^?.!]+\band\b[^?.!]+)",
      lower,
  )
  if pair_tail:
    pair_clauses = [
        seg.strip()
        for seg in pair_tail.group(1).split(",")
        if " and " in seg.lower()
    ]
    if len(pair_clauses) >= 2:
      return min(len(pair_clauses), 5)

  # Detect dual superlative pairs ('biggest and lowest', 'highest and lowest').
  if re.search(
      r"\b(?:biggest|highest|largest|most|maximum)\s+and\s+"
      r"(?:lowest|smallest|least|minimum)\b",
      lower,
  ):
    return 2

  # Inspect follow-up projection sentences after `.`, `;`, or `?` (`? Show ...`,
  # `? Indicate ...`) while ignoring decimal-rounding instructions.
  followup = re.search(
      r"(?:[.;?]\s*(?:please\s+)?(?:indicate|include|provide|show|identify|"
      r"state|list|give)\s+|\b(?:along\s+with|as\s+well\s+as)\s+)([^?.!;]{3,90})",
      lower,
  )
  if followup:
    tail_clause = followup.group(1).strip()
    if not re.match(r"^(?:your\s+answer|the\s+answer|it\s+as)\b", tail_clause):
      extra_parts = [
          p.strip()
          for p in re.split(r",|\band\b|\bas\s+well\s+as\b", tail_clause)
          if p.strip() and len(p.split()) <= 6
      ]
      return max(2, min(1 + len(extra_parts), 5)) if extra_parts else 2

  # Explicit single-sentence scalar aggregations ('how many', 'what is the avg').
  if re.match(
      r"^(?:how\s+many|what\s+(?:is|was)\s+the\s+(?:total|average|mean|ratio|"
      r"rate|percentage|number|count|highest|lowest|maximum|minimum|difference))\b",
      lower,
  ):
    if (
        " and " not in lower[:55]
        and "." not in body_no_end
        and "?" not in body_no_end
    ):
      return 1

  # Look for compact 'list/give/show/what are the X and Y of/for' projections.
  proj_match = re.search(
      r"\b(?:list|give|show|what\s+are)\s+(?:the\s+|each\s+)?"
      r"([^,?.!;]{1,40}(?:,\s*[^,?.!;]{1,30})*(?:\s+and\s+[^,?.!;]{1,30})?)\s+"
      r"(?:of|for|in|among|where|with|by)\b",
      lower,
  )
  if proj_match:
    parts = [
        p.strip()
        for p in re.split(r",|\band\b", proj_match.group(1))
        if p.strip() and len(p.split()) <= 6
    ]
    if parts:
      return max(1, min(len(parts), 5))
  return 1


def canonicalize_intent(question: str, evidence: str = "") -> IntentFrame:
  """Canonicalizes a natural language or voice question into an IntentFrame.

  Args:
    question: Raw user question or spoken ASR transcript.
    evidence: Optional external domain knowledge / formula note.

  Returns:
    Structured IntentFrame guiding value retrieval, linking, and verification.
  """
  # Delegate spoken filler removal and ASR casing alignment to `asr_normalizer`.
  cleaned, removed_count = clean_asr_utterance(question, evidence)

  combined = f"{cleaned} {evidence}"
  needs_float_div = bool(
      _DIVISION_CUE_RE.search(combined) or "/" in evidence
  )
  needs_extrema = bool(_EXTREMA_CUE_RE.search(cleaned))
  is_rank_query = bool(_IMPERATIVE_RANK_RE.match(cleaned))
  entities = _extract_entity_candidates(cleaned, evidence)
  arity = _estimate_projection_arity(cleaned)
  # Explicit `calculation =` evidence without follow-up columns mandates arity 1.
  strict_single = bool(
      arity == 1 and re.search(r"\bcalculation\s*=", evidence, re.IGNORECASE)
  )

  # Route query difficulty based on analytical cues, window ranks, and formulas.
  if _CHALLENGING_CUE_RE.search(combined) or (
      needs_float_div and needs_extrema
  ):
    complexity = QueryDifficulty.CHALLENGING
  elif (
      needs_float_div
      or needs_extrema
      or is_rank_query
      or len(entities) >= 2
      or evidence.strip()
  ):
    complexity = QueryDifficulty.MODERATE
  else:
    complexity = QueryDifficulty.SIMPLE

  return IntentFrame(
      raw_question=question,
      canonical_question=cleaned,
      disfluencies_removed=removed_count,
      entities=entities,
      requires_floating_division=needs_float_div,
      requires_extrema_order=needs_extrema,
      expected_projection_arity=arity,
      complexity=complexity,
      strict_single_column=strict_single,
  )
