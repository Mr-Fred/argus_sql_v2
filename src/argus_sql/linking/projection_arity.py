"""Projection arity and follow-up clause estimator for NL2SQL canonicalization.

Estimates expected SELECT column count and multi-attribute follow-up flags
from canonicalized natural language or ASR utterances while guarding against
declarative context sentences, subordinate WHERE clauses, and filter tails.
"""

import re

# Matches imperative `Rank <entities> by <metric>` analytical ranking prompts.
IMPERATIVE_RANK_RE = re.compile(
    r"^(?:please\s+)?rank\s+(?!of\b)\S+[\s\S]*\bby\b",
    re.IGNORECASE,
)

# Detects interrogative or imperative projection cues in a head clause so
# declarative context sentences (`Charter = 1. What is ...?`) never look like
# multi-projection head questions.
_HEAD_QUERY_CUE_RE = re.compile(
    r"(?:^(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?(?:please\s+)?"
    r"(?:what|which|who|whom|whose|where|when|how|"
    r"list|give|show|provide|indicate|include|identify|find|state|"
    r"report|display|name|calculate|compute|count|rank|tell|return)\b"
    r"|,\s*(?:please\s+)?"
    r"(?:what\s+(?:is|are|was|were)|how\s+(?:many|much)|"
    r"list|give|show|provide|indicate|include|identify|find|state|"
    r"report|display|name|calculate|compute|count|rank|tell|return)\b)",
    re.IGNORECASE,
)

# Matches follow-up projection clauses after `.`, `;`, `?`, commas, or spoken
# `and [also] <verb>` joins without punctuation; requires `also` before
# `including` so appositives (`of fuels, including diesel`) never misfire.
_FOLLOWUP_PROJ_RE = re.compile(
    r"(?:([.;?])\s*(?:and\s+)?(?:please\s+)?(?:also\s+)?"
    r"(?:indicate|include|provide|show|identify|state|list|give|mention|"
    r"report|display|name|what\s+(?:is|are|was|were))\s+"
    r"(?:out\s+|down\s+|me\s+)?"
    r"|\b(?:along\s+with|as\s+well\s+as|together\s+with)\s+"
    r"|,\s*(?:and\s+)?(?:please\s+)?(?:also\s+)?"
    r"(?:include|also\s+including|show|showing|indicate|indicating|"
    r"provide|providing|list|listing|give|giving)\s+(?:me\s+|out\s+)?"
    r"|\b(?:and\s+)?(?:please\s+also|also\s+please|also|please)\s+"
    r"(?:indicate|include|provide|show|identify|state|list|give|mention|"
    r"report|display|name)\s+(?:out\s+|down\s+|me\s+)?"
    r"|\band\s+(?:show|give|include|indicate|provide|identify|mention)\s+"
    r"(?:me\s+|out\s+)?(?=(?:the|their|his|her|its|all|how|what|which)\b))"
    r"([^?.!;]{2,140})",
    re.IGNORECASE,
)

# Anchors possessive (`their`/`his`/`her`/`its`) attribute lists at clause
# heads so subordinate clauses (`users who show their email`) stay scalar.
_THEIR_PROJ_RE = re.compile(
    r"(?:^|[.;?]\s*|,\s*|\band\s+)"
    r"(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?(?:please\s+)?(?:also\s+)?"
    r"(?:list|give|show|provide|indicate|include|identify|find|state|"
    r"report|display|name|return|tell\s+me|what\s+(?:are|is|were|was))\s+"
    r"(?:out\s+|down\s+|me\s+)?(?:all\s+(?:of\s+)?)?(?:the\s+)?"
    r"(?:their|his(?:\s+or\s+her)?|her|its)\s+([^?.!;]{2,140})",
    re.IGNORECASE,
)

# Strips trailing filter/sort modifiers before counting comma/`and` attributes,
# while preserving noun complements like `withdrawals in cash`.
_ATTR_TAIL_MODIFIER_RE = re.compile(
    r"\s+\b(?:for\b.*|between\b.*|from\b.*|during\b.*|throughout\b.*|"
    r"since\b.*|until\b.*|before\b.*|after\b.*|within\b.*|across\b.*|"
    r"having\b.*|where\b.*|who\b.*|whose\b.*|when\b.*|while\b.*|if\b.*|"
    r"among\b.*|that\b.*|order(?:ed)?\s+by\b.*|sort(?:ed)?\s+by\b.*|"
    r"group(?:ed)?\s+by\b.*|"
    r"in\s+(?:(?:the|each|all|both|any|this|that|these|those)\s+|"
    r"(?:ascending|descending|alphabetical|reverse|chronological)\b|"
    r"\d|(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|spring|summer|fall|autumn|winter|q[1-4])\b|"
    r"(?:year|month|season|period|quarter|state|city|county|country|region|"
    r"district|branch|office|department|hospital|school|team|league|race|"
    r"game|match|table|segment|division|class|grade)s?\b).*)$",
    re.IGNORECASE,
)

# Strips trailing Proper-Noun locative `in New York and California` filters.
_PROPER_LOCATIVE_IN_RE = re.compile(
    r"\s+\bin\s+[A-Z][a-zA-Z0-9_.-]*(?:\s+[A-Z][a-zA-Z0-9_.-]*)*"
    r"(?:\s*(?:,|and|or)\s+[A-Z][a-zA-Z0-9_.-]*(?:\s+[A-Z][a-zA-Z0-9_.-]*)*)+$"
)

# Strips trailing `with <filter>` while preserving stranded `diagnosed with`.
_WITH_FILTER_TAIL_RE = re.compile(
    r"(?<!\bdiagnosed)(?<!\bassociated)(?<!\baffiliated)(?<!\bcombined)"
    r"(?<!\balong)(?<!\btogether)\s+\bwith\s+\S.*$",
    re.IGNORECASE,
)


def _strip_attribute_tail_modifiers(clause: str) -> str:
  """Iteratively removes trailing filter/order phrases from an attribute span."""
  # Run up to 4 passes so chained tails (`in NY and CA where x = 1`) peel off.
  trimmed = clause.strip()
  for _ in range(4):
    nxt = _PROPER_LOCATIVE_IN_RE.sub("", trimmed)
    nxt = _ATTR_TAIL_MODIFIER_RE.sub("", nxt)
    nxt = _WITH_FILTER_TAIL_RE.sub("", nxt)
    if nxt == trimmed:
      break
    trimmed = nxt
  return trimmed


def _split_projection_items(clause: str) -> list[str]:
  """Splits a projection clause into bounded attribute phrases (`<= 8` words)."""
  # Trim trailing WHERE/ORDER modifiers so final attributes stay under 8 words.
  trimmed = _strip_attribute_tail_modifiers(clause)
  return [
      seg.strip()
      for seg in re.split(
          r",|\band\b|\bas\s+well\s+as\b|\balong\s+with\b|\btogether\s+with\b",
          trimmed,
          flags=re.IGNORECASE,
      )
      if seg.strip() and len(seg.split()) <= 8
  ]


def estimate_projection_spec(canonical_q: str) -> tuple[int, bool]:
  """Estimates `(expected_projection_arity, has_followup_projection)`.

  Args:
    canonical_q: Canonicalized natural language or ASR question text.

  Returns:
    Tuple `(expected_projection_arity, has_followup_projection)` where `0`
    denotes an unconstrained multi-sentence narrative prompt.
  """
  lower = canonical_q.lower()
  # Explicit single-output directives override narrative or follow-up context.
  if re.search(
      r"\b(?:(?:only|just)\s+need\s+the\s+(?:single\s+)?(?:count|total|number|"
      r"average|percentage|rate|ratio|difference|score|value|name|id)"
      r"|(?:give|show|return)(?:\s+me)?\s+the\s+single\s+\w+)\b",
      lower,
  ):
    return (1, False)

  # Multi-sentence narrative briefings (>38 words or >=2 periods) frequently
  # specify custom 4..16 column schemas; leave unconstrained (`0`).
  body_no_end = canonical_q.rstrip(".?!")
  if len(canonical_q.split()) > 38 or body_no_end.count(".") >= 2:
    return (0, False)

  # Expand compound multi-field shorthand; leave plural 'full names' unconstrained
  # because schemas represent full names across 1, 2, 3, or 6 columns.
  if "full communication address" in lower or "complete address" in lower:
    return (4, False)
  if "full names" in lower:
    return (0, False)

  # Imperative `Rank <entities> by <metric>` projects `(entity, metric, rank)`.
  if IMPERATIVE_RANK_RE.match(canonical_q):
    return (3, False)

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
      return (min(len(pair_clauses), 5), False)

  # Detect dual superlative pairs ('biggest and lowest', 'highest and lowest').
  if re.search(
      r"\b(?:biggest|highest|largest|most|maximum)\s+and\s+"
      r"(?:lowest|smallest|least|minimum)\b",
      lower,
  ):
    return (2, False)

  # Inspect follow-up projection clauses (`? Show ...`, `and also show ...`,
  # `along with ...`) only when the preceding head clause is itself a query.
  for followup in _FOLLOWUP_PROJ_RE.finditer(canonical_q):
    sep_char = followup.group(1) or ""
    tail_clause = followup.group(2).strip()
    head_prefix = canonical_q[: followup.start()].strip()
    last_head = re.split(r"[.;?]+\s*", head_prefix)[-1].strip()
    if sep_char != "?" and _HEAD_QUERY_CUE_RE.search(last_head) is None:
      continue
    if re.match(
        r"^(?:(?:your|the)\s+(?:answer|result|output|value)\b|"
        r"it\s+(?:as|in|rounded|to)\b|them\s+(?:by|in|ordered|sorted)\b|"
        r"(?:me\s+)?the\s+single\b)",
        tail_clause,
        re.IGNORECASE,
    ):
      continue
    extra_parts = _split_projection_items(tail_clause)
    head_m = re.match(
        r"^(?:what\s+(?:is|are|was|were)|list|give|show|name)\s+(?:the\s+)?"
        r"([^?.!;]{3,55}?)\s+(?:of|for|in|among|where|with)\b",
        last_head,
        re.IGNORECASE,
    )
    head_arity = (
        max(1, len(_split_projection_items(head_m.group(1))))
        if head_m is not None
        else 1
    )
    tail_arity = len(extra_parts) if extra_parts else 1
    return (max(2, min(head_arity + tail_arity, 5)), True)

  # Inspect `'their'` / possessive attribute lists (`list their ID, sex and ...`).
  their_match = _THEIR_PROJ_RE.search(canonical_q)
  if their_match:
    their_parts = _split_projection_items(their_match.group(1))
    if their_parts:
      their_arity = max(1, min(len(their_parts), 5))
      return (their_arity, their_arity >= 2)

  # Catch conjoined entity + `'their'` lists (`list the clients and their ...`),
  # anchored at clause heads so subordinate `who` clauses do not misfire.
  entity_their = re.search(
      r"(?:^|[.;?]\s*|,\s*|\band\s+)(?:please\s+)?"
      r"(?:list|give|show|provide|indicate|include|identify|find|state|"
      r"what\s+are)\s+(?:out\s+|me\s+)?([^?.!;]{3,95}?\btheir\b[^?.!;]{2,95})",
      canonical_q,
      re.IGNORECASE,
  )
  if entity_their:
    et_head = _strip_attribute_tail_modifiers(entity_their.group(1))
    if re.search(r"\btheir\b", et_head, re.IGNORECASE):
      et_parts = _split_projection_items(et_head)
      if len(et_parts) >= 2:
        return (min(len(et_parts), 5), True)

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
      return (1, False)

  # Look for compact 'list/give/show/what are the X and Y of/for' projections.
  proj_match = re.search(
      r"\b(?:list|give|show|provide|indicate|state|what\s+are)\s+"
      r"(?:out\s+|down\s+|me\s+)?(?:the\s+|each\s+|all\s+(?:the\s+)?)?"
      r"([^,?.!;]{1,45}(?:,\s*[^,?.!;]{1,40})*(?:\s+and\s+[^,?.!;]{1,45})?)\s+"
      r"(?:of|for|in|among|where|with|by|from|that|who)\b",
      canonical_q,
      re.IGNORECASE,
  )
  if proj_match:
    parts = _split_projection_items(proj_match.group(1))
    if parts:
      return (max(1, min(len(parts), 5)), False)
  return (1, False)
