"""Voice-aware natural language intent canonicalizer and complexity classifier.

Transforms user questions and ASR transcripts into structured IntentFrame
contracts using a single-pass token-level Finite State Machine (FSM) and
Aho-Corasick multi-pattern cue matcher, avoiding brittle regex cascades.
"""

from argus_sql.contracts.types import IntentFrame
from argus_sql.contracts.types import QueryDifficulty
from argus_sql.linking.asr_normalizer import ENTITY_STOPWORDS
from argus_sql.linking.asr_normalizer import clean_asr_utterance
from argus_sql.linking.projection_arity import IMPERATIVE_RANK_RE
from argus_sql.linking.projection_arity import estimate_projection_spec
from argus_sql.linking.token_fsm import AhoCorasickAutomaton
from argus_sql.linking.token_fsm import LexToken
from argus_sql.linking.token_fsm import tokenize_utterance

# Spoken geographic/domain abbreviations matched in one Aho-Corasick pass.
_SPOKEN_EXPANSIONS_AC: AhoCorasickAutomaton[str] = AhoCorasickAutomaton(
    patterns=(
        ("calif.", "California"),
        ("calif", "California"),
        ("cali", "California"),
        ("l.a. county", "Los Angeles"),
        ("la county", "Los Angeles"),
        ("ny", "New York"),
        ("sf", "San Francisco"),
    ),
    case_sensitive=False,
)

# Unified Aho-Corasick cue automaton for division, extrema, and difficulty cues.
_SEMANTIC_CUES_AC: AhoCorasickAutomaton[str] = AhoCorasickAutomaton(
    patterns=(
        # Floating-point division / rate / percentage cues.
        ("ratio", "division"),
        ("rate", "division"),
        ("percentage", "division"),
        ("percent", "division"),
        ("proportion", "division"),
        ("fraction", "division"),
        ("per capita", "division"),
        ("divided by", "division"),
        ("share of", "division"),
        ("%", "division"),
        # Extrema / superlative ORDER BY ... LIMIT k cues.
        ("highest", "extrema"),
        ("lowest", "extrema"),
        ("top", "extrema"),
        ("bottom", "extrema"),
        ("most", "extrema"),
        ("least", "extrema"),
        ("largest", "extrema"),
        ("smallest", "extrema"),
        ("oldest", "extrema"),
        ("youngest", "extrema"),
        ("earliest", "extrema"),
        ("latest", "extrema"),
        ("maximum", "extrema"),
        ("minimum", "extrema"),
        # Analytical / subquery complexity routing cues.
        ("among those", "challenging"),
        ("more than the average", "challenging"),
        ("less than the average", "challenging"),
        ("difference between", "challenging"),
        ("compared to", "challenging"),
        ("neither", "challenging"),
        ("both", "both_head"),
        ("and", "both_tail"),
    ),
    case_sensitive=False,
)

# Evidence lock cue matched via Aho-Corasick for strict 1-column formulas.
_EVIDENCE_CALC_AC: AhoCorasickAutomaton[str] = AhoCorasickAutomaton(
    patterns=(("calculation =", "calc_eq"), ("calculation=", "calc_eq")),
    case_sensitive=False,
)

# Determiners skipped after relational prepositions (`in the`, `for all`, ...).
_PREP_DET_SKIP: frozenset[str] = frozenset({
    "the",
    "each",
    "all",
    "any",
    "this",
    "that",
    "these",
    "those",
    "their",
    "his",
    "her",
    "its",
    "our",
    "a",
    "an",
})

# Prepositional triggers launching a bounded 1..2-token entity lookahead.
_PREPOSITION_TRIGGERS: frozenset[str] = frozenset({
    "in",
    "for",
    "named",
    "called",
    "county",
    "state",
    "status",
    "type",
    "currency",
    "segment",
    "country",
    "tagged",
    "code",
})

# Hyphenated and adverbial status suffixes captured from user questions.
_HYPHEN_STATUS_SUFFIXES: frozenset[str] = frozenset({
    "paying",
    "based",
    "funded",
    "run",
    "level",
    "type",
})
_ADVERB_STATUS_SUFFIXES: frozenset[str] = frozenset({
    "funded",
    "owned",
    "operated",
    "managed",
    "run",
    "based",
})


def _is_domain_id_token(raw: str) -> bool:
  """Returns True for alphanumeric/underscore domain IDs (`TR004_8_9`, `TR008`)."""
  # Reject tokens containing punctuation outside underscores (`-`, `.`, `:`, `&`).
  if "-" in raw or "." in raw or ":" in raw or "&" in raw:
    return False
  # Branch 1: `<alpha_1_8>_<alnum_or_underscore>` (e.g., `col_1`, `db_id`).
  if "_" in raw:
    head, _, tail = raw.partition("_")
    if (
        1 <= len(head) <= 8
        and head.isalpha()
        and bool(tail)
        and all(ch.isalnum() or ch == "_" for ch in tail)
    ):
      return True
  # Branch 2: `<alpha_2_6><digits><alnum_or_underscore>*` (e.g., `TR004_8_9`).
  alpha_prefix = 0
  while alpha_prefix < len(raw) and raw[alpha_prefix].isalpha():
    alpha_prefix += 1
  if not (2 <= alpha_prefix <= 6) or alpha_prefix >= len(raw):
    return False
  rest = raw[alpha_prefix:]
  return rest[0].isdigit() and all(ch.isalnum() or ch == "_" for ch in rest)


def _is_proper_noun_token(tok: LexToken) -> bool:
  """Returns True when `tok` starts with uppercase and fits Proper Noun rules."""
  if tok.kind != "WORD" or len(tok.raw) < 2 or not tok.raw[0].isupper():
    return False
  if "_" in tok.raw or ":" in tok.raw:
    return False
  return all(ch.isalnum() or ch in "&.-" for ch in tok.raw[1:])


def _is_prep_candidate_token(tok: LexToken, min_len: int) -> bool:
  """Returns True if `tok` can join a prepositional entity n-gram."""
  if tok.kind != "WORD" or not (min_len <= len(tok.raw) <= 25):
    return False
  return tok.raw[0].isalpha() and all(
      ch.isalnum() or ch in "_-" for ch in tok.raw[1:]
  )


def _scan_tokens_fsm(
    tokens: tuple[LexToken, ...],
    spans: list[str],
    seen: set[str],
    scan_question_cues: bool,
) -> None:
  """Runs a single-pass left-to-right token FSM to collect entity candidates."""
  n = len(tokens)
  idx = 0

  def _add_span(candidate: str) -> None:
    """Appends `candidate` once in first-appearance order."""
    clean = candidate.strip()
    if clean and clean not in seen:
      seen.add(clean)
      spans.append(clean)

  # Walk tokens once; `scan_question_cues` gates question-only status/prep rules.
  while idx < n and len(spans) < 16:
    tok = tokens[idx]

    # 1. Quoted literals (`'Magnet'`, `"School Name"`).
    if tok.kind == "QUOTE":
      inner = tok.raw[1:-1].strip()
      if 1 <= len(inner) <= 50:
        _add_span(inner)
      idx += 1
      continue

    # 2. Alphanumeric or underscore domain codes (`TR004_8_9`, `TR008`).
    if tok.kind == "WORD" and _is_domain_id_token(tok.raw):
      _add_span(tok.raw)
      idx += 1
      continue

    # 3. Capitalized Proper Noun phrases (1..4 tokens, stripping lead stopwords).
    if _is_proper_noun_token(tok):
      run_end = idx + 1
      # Stop proper-noun run before domain IDs (`TR008`) or after 4 words.
      while (
          run_end < n
          and (run_end - idx) < 4
          and _is_proper_noun_token(tokens[run_end])
          and not _is_domain_id_token(tokens[run_end].raw)
      ):
        run_end += 1
      words = [t.raw for t in tokens[idx:run_end]]
      # Trim sentence-initial stopwords (`Between DC` -> `DC`).
      while words and words[0].lower() in ENTITY_STOPWORDS:
        words.pop(0)
      phrase = " ".join(words)
      if phrase and phrase.lower() not in ENTITY_STOPWORDS:
        _add_span(phrase)
      # If the proper-noun scan consumed only a single capitalized stopword
      # (e.g., `In fresno schools`), fall through so prep lookahead still runs.
      if words or run_end > idx + 1:
        idx = run_end
        continue

    if scan_question_cues and tok.kind == "WORD":
      # 4a. Hyphen-prefixed status codes (`czk-paying` -> `czk`).
      if "-" in tok.raw:
        left, sep, right = tok.raw.partition("-")
        if (
            sep
            and left.isalpha()
            and 2 <= len(left) <= 6
            and right.lower() in _HYPHEN_STATUS_SUFFIXES
        ):
          _add_span(left)

      # 4b. Adverbial status bigrams (`federally funded`, `privately owned`).
      if (
          tok.lower.endswith("ly")
          and 5 <= len(tok.raw) <= 17
          and tok.raw.isalpha()
          and (idx + 1) < n
          and tokens[idx + 1].kind == "WORD"
          and tokens[idx + 1].lower in _ADVERB_STATUS_SUFFIXES
      ):
        _add_span(f"{tok.raw} {tokens[idx + 1].raw}")

      # 5. Prepositional n-gram capture (`in Fresno schools`, `tagged r`).
      # Uses non-consuming lookahead so downstream proper nouns still fire.
      if tok.lower in _PREPOSITION_TRIGGERS:
        look = idx + 1
        # Skip determiners (`the`, `all`, `their`) after relational prepositions.
        while (
            look < n
            and (look - idx) <= 3
            and tokens[look].kind == "WORD"
            and tokens[look].lower in _PREP_DET_SKIP
        ):
          look += 1
        if look < n and _is_prep_candidate_token(tokens[look], min_len=2):
          first = tokens[look]
          if first.lower not in ENTITY_STOPWORDS:
            cand_words = [first.raw]
            nxt_idx = look + 1
            if (
                nxt_idx < n
                and _is_prep_candidate_token(tokens[nxt_idx], min_len=3)
                and tokens[nxt_idx].lower not in _PREP_DET_SKIP
            ):
              cand_words.append(tokens[nxt_idx].raw)
            _add_span(" ".join(cand_words))

    idx += 1


def extract_raw_entities(question: str, evidence: str = "") -> tuple[str, ...]:
  """Extracts surface entity spans without Stage 1 voice/ASR canonicalization.

  Runs a single-pass token-level FSM to capture quoted literals, multi-word
  capitalized proper nouns, alphanumeric domain identifiers, hyphen/adverb
  status phrases, and prepositional n-grams so non-voice baselines stay fair
  without leaking spoken shorthand expansions (`_SPOKEN_EXPANSIONS_AC`).

  Args:
    question: User natural language question or raw ASR transcript.
    evidence: Optional external domain knowledge / formula specification.

  Returns:
    Tuple of up to 16 candidate entity/literal spans in appearance order.
  """
  spans: list[str] = []
  seen: set[str] = set()

  # Scan question tokens with full structural rules, then evidence literals/IDs.
  if question.strip():
    _scan_tokens_fsm(
        tokenize_utterance(question),
        spans,
        seen,
        scan_question_cues=True,
    )
  if evidence.strip() and len(spans) < 16:
    _scan_tokens_fsm(
        tokenize_utterance(evidence),
        spans,
        seen,
        scan_question_cues=False,
    )
  return tuple(spans[:16])


def _extract_entity_candidates(question: str, evidence: str) -> tuple[str, ...]:
  """Extracts surface entity spans plus Stage 1 spoken geographic expansions."""
  spans = list(extract_raw_entities(question, evidence))

  # Add canonical forms for spoken abbreviations via one Aho-Corasick pass.
  for match in _SPOKEN_EXPANSIONS_AC.find_all(
      question, boundary_aware=True, resolve_overlaps=True
  ):
    if match.payload not in spans:
      spans.append(match.payload)

  return tuple(spans[:16])


def _estimate_projection_arity(canonical_q: str) -> int:
  """Estimates requested output arity (`0` = unconstrained narrative prompt)."""
  # Retain single-int helper compatibility around `estimate_projection_spec`.
  arity, _ = estimate_projection_spec(canonical_q)
  return arity


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

  # Run a single Aho-Corasick scan over `combined` for division/extrema/tier cues.
  combined = f"{cleaned} {evidence}"
  cue_matches = _SEMANTIC_CUES_AC.find_all(
      combined, boundary_aware=True, resolve_overlaps=False
  )
  cleaned_end = len(cleaned)
  needs_float_div = "/" in evidence
  needs_extrema = False
  is_challenging = False
  saw_both_head = False

  for match in cue_matches:
    tag = match.payload
    if tag == "division":
      needs_float_div = True
    elif tag == "extrema" and match.start < cleaned_end:
      needs_extrema = True
    elif tag == "challenging":
      is_challenging = True
    elif tag == "both_head":
      saw_both_head = True
    elif tag == "both_tail" and saw_both_head:
      is_challenging = True

  is_rank_query = bool(IMPERATIVE_RANK_RE.match(cleaned))
  entities = _extract_entity_candidates(cleaned, evidence)
  # Extract expected SELECT arity and multi-column follow-up indicator together.
  arity, has_followup = estimate_projection_spec(cleaned)
  # Explicit `calculation =` evidence without follow-up columns mandates arity 1.
  strict_single = bool(
      arity == 1
      and _EVIDENCE_CALC_AC.find_all(
          evidence, boundary_aware=True, resolve_overlaps=True
      )
  )

  # Route query difficulty based on analytical cues, window ranks, and formulas.
  if is_challenging or (needs_float_div and needs_extrema):
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

  # Return immutable IntentFrame contract consumed by linking and verification.
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
      has_followup_projection=has_followup,
      evidence=evidence,
  )
