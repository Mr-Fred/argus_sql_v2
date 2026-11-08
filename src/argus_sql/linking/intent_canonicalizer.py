"""Voice-aware natural language intent canonicalizer and complexity classifier.

Transforms raw user questions—including noisy spoken ASR transcripts with
fillers, stutters, spoken numbers, and colloquial ratio phrasing—into clean
IntentFrame structures before schema linking and SQL candidate synthesis.
"""

import re

from argus_sql.contracts.types import IntentFrame
from argus_sql.contracts.types import QueryDifficulty

# Spoken filler phrases ordered longest-first to avoid partial residue.
_DISFLUENCY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:uh+|um+|er+|ah+|hmm+)\b[,\s]*", re.IGNORECASE),
    re.compile(r"\b(?:you know|i mean|sort of|kind of)\b[,\s]*", re.IGNORECASE),
    re.compile(
        r"^(?:hey\s+|okay\s+|so\s+|well\s+|please\s+)*"
        r"(?:can you\s+|could you\s+)?(?:please\s+)?"
        r"(?:tell me|show me|find out|let me know)\s+",
        re.IGNORECASE,
    ),
)

# Repeated consecutive word stutter regex (e.g., 'what is the the highest').
_STUTTER_RE = re.compile(r"\b([a-zA-Z]{2,})\s+\1\b", re.IGNORECASE)

# Spoken cardinal and ordinal number words frequently emitted by ASR engines.
_SPOKEN_NUMBERS: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "fifteen": "15",
    "twenty": "20",
    "fifty": "50",
    "hundred": "100",
}

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


def _normalize_spoken_numbers(text: str) -> str:
  """Converts spoken number tokens after limit/comparison words into digits."""
  tokens = text.split()
  converted: list[str] = []
  # Convert number words when preceded by 'top', 'bottom', 'over', 'exceed', etc.
  trigger_prev = {
      "top",
      "bottom",
      "first",
      "last",
      "over",
      "under",
      "than",
      "least",
      "most",
      "above",
      "below",
  }
  for idx, tok in enumerate(tokens):
    clean = tok.strip(".,?!").lower()
    prev = tokens[idx - 1].lower() if idx > 0 else ""
    if clean in _SPOKEN_NUMBERS and (prev in trigger_prev or clean != "one"):
      suffix = tok[len(tok.rstrip(".,?!")) :]
      converted.append(_SPOKEN_NUMBERS[clean] + suffix)
    else:
      converted.append(tok)
  return " ".join(converted)


def _extract_entity_candidates(question: str, evidence: str) -> tuple[str, ...]:
  """Extracts quoted literals, capitalized spans, codes, and spoken expansions."""
  spans: list[str] = []
  combined = f"{question} {evidence}".strip()

  # 1. Capture single- or double-quoted literal values verbatim.
  for match in re.finditer(r"['\"]([^'\"]{1,50})['\"]", combined):
    val = match.group(1).strip()
    if val and val not in spans:
      spans.append(val)

  # 2. Capture capitalized multi-word Proper Noun phrases (e.g., 'Los Angeles').
  for match in re.finditer(
      r"\b([A-Z][a-zA-Z0-9&.-]+(?:\s+[A-Z][a-zA-Z0-9&.-]+){0,3})\b",
      combined,
  ):
    phrase = match.group(1).strip()
    if phrase.lower() not in {
        "what",
        "which",
        "where",
        "when",
        "who",
        "how",
        "list",
        "find",
        "give",
        "please",
        "among",
        "calculate",
        "select",
    }:
      if phrase not in spans:
        spans.append(phrase)

  # 3. Capture hyphen-prefixed domain codes (e.g., 'czk-paying' -> 'czk').
  for match in re.finditer(
      r"\b([a-zA-Z]{2,6})-(?:paying|based|funded|run|level|type)\b",
      question,
      re.IGNORECASE,
  ):
    code_tok = match.group(1).strip()
    if code_tok not in spans:
      spans.append(code_tok)

  # 4. Capture lowercase/mixed n-grams after relational prepositions and cues.
  for match in re.finditer(
      r"\b(?:in|for|named|called|county|state|status|type|currency|segment"
      r"|country|tagged|code)\s+"
      r"([a-zA-Z][a-zA-Z0-9_-]{1,25}(?:\s+[a-zA-Z][a-zA-Z0-9_-]{2,25})?)",
      question,
      re.IGNORECASE,
  ):
    candidate = match.group(1).strip()
    if candidate.lower() not in {"the", "each", "all", "any"} and candidate not in spans:
      spans.append(candidate)

  # 5. Add canonical forms for common spoken abbreviations.
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

  # Multi-sentence narrative briefings (>38 words or >=2 sentences) frequently
  # specify custom 4..16 column schemas; leave unconstrained (`0`) to avoid drift.
  body_no_end = canonical_q.rstrip(".?!")
  if len(canonical_q.split()) > 38 or body_no_end.count(".") >= 2:
    return 0

  # Expand compound multi-field shorthand common in BIRD-SQL prompts.
  if "full communication address" in lower or "complete address" in lower:
    return 4
  if "full names" in lower:
    return 4

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

  # Inspect sentence-anchored projection directives (avoiding relative 'that include').
  followup = re.search(
      r"(?:(?:^|[.;])\s*(?:please\s+)?(?:indicate|include|provide)\s+"
      r"|\b(?:along\s+with|as\s+well\s+as)\s+)([^?.!;]{3,55})",
      lower,
  )
  if followup:
    extra_parts = [
        p.strip()
        for p in re.split(r",|\band\b|\bas\s+well\s+as\b", followup.group(1))
        if p.strip() and len(p.split()) <= 6
    ]
    if extra_parts:
      return max(2, min(1 + len(extra_parts), 5))

  # Explicit single-sentence scalar aggregations ('how many', 'what is the avg').
  if re.match(
      r"^(?:how\s+many|what\s+(?:is|was)\s+the\s+(?:total|average|mean|ratio|"
      r"rate|percentage|number|count|highest|lowest|maximum|minimum|difference))\b",
      lower,
  ):
    if " and " not in lower[:55] and "." not in body_no_end:
      return 1

  # Look for compact 'list/give/show/what are the X and Y of/for' projections
  # (excluding 'state' which frequently appears as the noun 'in the state').
  proj_match = re.search(
      r"\b(?:list|give|show|what\s+are)\s+(?:the\s+|each\s+)?"
      r"([^,?.!;]{1,40}(?:,\s*[^,?.!;]{1,30})*(?:\s+and\s+[^,?.!;]{1,30})?)\s+"
      r"(?:of|for|in|among|where|with|by)\b",
      lower,
  )
  if proj_match:
    clause = proj_match.group(1)
    parts = [
        p.strip()
        for p in re.split(r",|\band\b", clause)
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
  cleaned = question.strip()
  removed_count = 0

  # Strip spoken filler tokens and conversational preambles iteratively.
  for pattern in _DISFLUENCY_PATTERNS:
    cleaned, n_subs = pattern.subn("", cleaned)
    removed_count += n_subs

  # Collapse consecutive word stutters ('the the' -> 'the').
  cleaned, n_stutters = _STUTTER_RE.subn(r"\1", cleaned)
  removed_count += n_stutters

  # Normalize spoken number words and collapse extra whitespace.
  cleaned = _normalize_spoken_numbers(cleaned)
  cleaned = re.sub(r"\s+", " ", cleaned).strip()
  if cleaned:
    cleaned = cleaned[0].upper() + cleaned[1:]

  combined = f"{cleaned} {evidence}"
  needs_float_div = bool(
      _DIVISION_CUE_RE.search(combined) or "/" in evidence
  )
  needs_extrema = bool(_EXTREMA_CUE_RE.search(cleaned))
  entities = _extract_entity_candidates(cleaned, evidence)
  arity = _estimate_projection_arity(cleaned)

  # Route query difficulty based on analytical cues, formulas, and entities.
  if _CHALLENGING_CUE_RE.search(combined) or (
      needs_float_div and needs_extrema
  ):
    complexity = QueryDifficulty.CHALLENGING
  elif needs_float_div or needs_extrema or len(entities) >= 2 or evidence.strip():
    complexity = QueryDifficulty.MODERATE
  else:
    complexity = QueryDifficulty.SIMPLE

  return IntentFrame(
      raw_question=question,
      canonical_question=cleaned or question.strip(),
      disfluencies_removed=removed_count,
      entities=entities,
      requires_floating_division=needs_float_div,
      requires_extrema_order=needs_extrema,
      expected_projection_arity=arity,
      complexity=complexity,
  )
