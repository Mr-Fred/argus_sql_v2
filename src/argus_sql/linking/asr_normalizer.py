"""Single-pass FSM & Aho-Corasick ASR disfluency, cardinal, and casing normalizer.

Isolates pure lexical speech-to-text transcript cleanup—spoken filler removal,
consecutive word-stutter collapsing, cardinal number conversion, `evidence`
identifier case restoration, and multi-sentence capitalization/punctuation—via
composable finite-state token scanning and Aho-Corasick multi-pattern lookup.
"""

from collections.abc import Sequence

from argus_sql.linking.token_fsm import AhoCorasickAutomaton
from argus_sql.linking.token_fsm import LexToken
from argus_sql.linking.token_fsm import NON_TERMINAL_ABBREVS
from argus_sql.linking.token_fsm import tokenize_utterance

# Spoken cardinal and ordinal number words frequently emitted by ASR engines.
# Mapped lazily after ranking/comparison heads so idiomatic phrases stay intact.
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

# Limit/comparison predecessors enabling safe `'one' -> '1'` conversion.
_CARDINAL_TRIGGER_PREV: frozenset[str] = frozenset({
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
})

# Determiners/interrogatives guarding noun phrases (`which kind of credit card`).
_KIND_OF_GUARDS: frozenset[str] = frozenset({
    "what",
    "which",
    "any",
    "same",
    "this",
    "that",
    "a",
})

# Single-word and two-word wake tokens buffered at the start of an utterance.
_WAKE_PREFIX_WORDS: frozenset[str] = frozenset({
    "hey",
    "okay",
    "so",
    "well",
    "please",
})
_WAKE_MODAL_BIGRAMS: frozenset[tuple[str, str]] = frozenset({
    ("can", "you"),
    ("could", "you"),
})
_WAKE_TERMINAL_BIGRAMS: frozenset[tuple[str, str]] = frozenset({
    ("tell", "me"),
    ("show", "me"),
    ("find", "out"),
    ("let", "me"),
})

# Mid-clause bigram fillers detected in one lookahead step by the token FSM.
_FILLER_BIGRAMS: frozenset[tuple[str, str]] = frozenset({
    ("you", "know"),
    ("i", "mean"),
})
_GUARDED_FILLER_BIGRAMS: frozenset[tuple[str, str]] = frozenset({
    ("sort", "of"),
    ("kind", "of"),
})

# Interrogative clause heads restoring terminal `?` on stripped ASR utterances.
_INTERROGATIVE_HEADS: frozenset[str] = frozenset({
    "what",
    "which",
    "where",
    "when",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "has",
    "have",
    "had",
})

# Question pronouns allowed after conditional/clause leads inside a sentence.
_EMBEDDED_WH_WORDS: frozenset[str] = frozenset({
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "why",
    "how",
})

# Imperative/declarative directive verbs indicating terminal `.` rather than `?`.
_IMPERATIVE_VERBS: frozenset[str] = frozenset({
    "list",
    "give",
    "show",
    "find",
    "rank",
    "calculate",
    "state",
    "identify",
    "indicate",
    "provide",
    "describe",
    "name",
})

# Uppercase SQL/formula/boolean keywords excluded from entity & ASR casing.
SQL_FORMULA_UPPER: frozenset[str] = frozenset({
    "SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER",
    "ON", "AND", "OR", "NOT", "IN", "IS", "AS", "BY", "GROUP", "ORDER",
    "HAVING", "LIMIT", "DISTINCT", "UNION", "ALL", "CASE", "WHEN", "THEN",
    "ELSE", "END", "CAST", "REAL", "INT", "INTEGER", "TEXT", "DATE", "NULL",
    "SUM", "AVG", "MIN", "MAX", "COUNT", "TOTAL", "ABS", "ROUND", "IIF",
    "COALESCE", "NULLIF", "SUBSTR", "STRFTIME", "BETWEEN", "LIKE", "ASC",
    "DESC", "OVER", "RANK", "DENSE_RANK", "ROW_NUMBER", "SUBTRACT", "DIVIDE",
    "MULTIPLY", "ADD", "YES", "NO", "OK", "TRUE", "FALSE", "IF", "OF", "TO",
    "AT", "DO", "GO", "SO", "UP", "AN", "BE", "HE", "ME", "WE", "IT",
})

# Sentence-initial English words and SQL operators excluded from entity spans.
ENTITY_STOPWORDS: frozenset[str] = frozenset({
    "what", "which", "where", "when", "who", "whom", "whose", "why", "how",
    "list", "find", "give", "show", "indicate", "include", "provide", "state",
    "identify", "rank", "please", "among", "between", "consider", "under",
    "there", "the", "this", "that", "these", "those", "for", "with", "from",
    "into", "only", "just", "if", "in", "is", "are", "was", "were",
    "calculate", "select", "percent", "percentage", "ratio", "rate",
    "difference", "average", "total", "number", "count", "valid", "status",
    "full",
}) | {kw.lower() for kw in SQL_FORMULA_UPPER}


def _is_vocal_filler(word_lower: str) -> bool:
  """Returns True if `word_lower` is an elongated vocal disfluency (`uh+`, etc.)."""
  if not word_lower or not word_lower.isalpha():
    return False
  # Match exact stem + single trailing character repetition (`uhh`, `ummm`, `hmm`).
  if word_lower in {"uh", "um", "er", "ah", "hmm"}:
    return True
  head = word_lower[:2]
  if head in {"uh", "um", "er", "ah"} and len(set(word_lower[1:])) == 1:
    return True
  return head == "hm" and len(word_lower) >= 3 and set(word_lower[1:]) == {"m"}


def _reconstruct_tokens(
    source: str,
    emitted: list[tuple[LexToken, str]],
) -> str:
  """Rebuilds text from emitted `(token, replacement_raw)` pairs cleanly."""
  if not emitted:
    return ""
  parts: list[str] = []
  prev_tok: LexToken | None = None
  for tok, text_val in emitted:
    if prev_tok is not None:
      gap = source[prev_tok.end : tok.start]
      # Drop a dangling comma left right before terminal punctuation after a
      # stripped filler (`what is the rate, um?` -> `what is the rate?`).
      if gap and tok.kind == "PUNCT" and tok.raw in ".?!" and parts[-1] == ",":
        parts.pop()
      is_closing_punct = tok.kind == "PUNCT" and tok.raw in ".?!,;:)]"
      is_after_open_punct = (
          prev_tok.kind == "PUNCT" and prev_tok.raw in "([{"
      )
      # Preserve zero-space attachments around punctuation/brackets even when
      # fillers were skipped in between (`rate uh?` -> `rate?`).
      if gap and not is_closing_punct and not is_after_open_punct:
        parts.append(" ")
    parts.append(text_val)
    prev_tok = tok
  return "".join(parts)


def normalize_spoken_numbers(text: str) -> str:
  """Converts spoken number tokens after limit/comparison words into digits.

  Args:
    text: Partially cleaned natural language or ASR utterance string.

  Returns:
    Utterance with spoken cardinals ('top five') converted to digits ('top 5').
  """
  tokens = tokenize_utterance(text)
  emitted: list[tuple[LexToken, str]] = []
  prev_word = ""
  # Scan tokens linearly; guard `'one'` unless preceded by a ranking bound.
  for tok in tokens:
    if tok.kind == "WORD" and tok.lower in _SPOKEN_NUMBERS and (
        prev_word in _CARDINAL_TRIGGER_PREV or tok.lower != "one"
    ):
      emitted.append((tok, _SPOKEN_NUMBERS[tok.lower]))
    else:
      emitted.append((tok, tok.raw))
    # Ignore punctuation so `'top, five'` or `'top five'` tracks prior word.
    if tok.kind != "PUNCT":
      prev_word = tok.lower
  return _reconstruct_tokens(text, emitted)


def _extract_evidence_casing_spans(evidence: str) -> tuple[str, ...]:
  """Extracts canonical Proper Nouns and domain codes from `evidence` in 1 pass."""
  tokens = tokenize_utterance(evidence)
  spans: list[str] = []
  n = len(tokens)
  idx = 0

  # Walk evidence tokens once to harvest Proper Nouns and uppercase codes.
  while idx < n:
    tok = tokens[idx]
    if tok.kind != "WORD":
      idx += 1
      continue

    # 1. Collect multi-word Proper Noun sequences (`Marvel Comics`).
    if (
        tok.raw[0].isupper()
        and len(tok.raw) >= 2
        and tok.raw[1:].islower()
        and tok.lower not in ENTITY_STOPWORDS
    ):
      run_end = idx + 1
      # Cap multi-word proper-noun accumulation at 4 tokens to avoid full clauses.
      while (
          run_end < n
          and (run_end - idx) < 4
          and tokens[run_end].kind == "WORD"
          and len(tokens[run_end].raw) >= 2
          and tokens[run_end].raw[0].isupper()
          and tokens[run_end].raw[1:].islower()
      ):
        run_end += 1
      if run_end - idx >= 2:
        spans.append(" ".join(t.raw for t in tokens[idx:run_end]))
        idx = run_end
        continue

    # 2. Collect uppercase or alphanumeric/underscore domain codes (`TR004_8_9`, `DC`).
    # Skip SQL keywords (`SUM`, `DIVIDE`) so lowercase verbs like `sum` stay lower.
    if len(tok.raw) >= 2 and tok.raw.upper() not in SQL_FORMULA_UPPER:
      has_up = any(ch.isupper() for ch in tok.raw)
      is_domain_code = tok.raw.isupper() or (
          has_up
          and (
              any(ch.isdigit() for ch in tok.raw)
              or "_" in tok.raw
              or any(ch.isupper() for ch in tok.raw[1:])
          )
      )
      if is_domain_code:
        spans.append(tok.raw)
    idx += 1

  # Deduplicate while preserving first-seen appearance order.
  return tuple(dict.fromkeys(spans))


def _is_interrogative_segment(segment_tokens: Sequence[LexToken]) -> bool:
  """Classifies whether the trailing sentence segment should end with `?`."""
  if not segment_tokens:
    return False
  words = [t for t in segment_tokens if t.kind == "WORD"]
  if not words:
    return False
  # Direct interrogative head (`What`, `Which`, `Is`, `How`, ...).
  if words[0].lower in _INTERROGATIVE_HEADS:
    return True

  # Imperative heads (`Please list ...`, `Find ...`) remain declarative `.`.
  head_idx = 1 if (len(words) > 1 and words[0].lower == "please") else 0
  if words[head_idx].lower in _IMPERATIVE_VERBS:
    return False

  # Embedded question heads after clause boundaries (`if so, how many more`).
  for pos, tok in enumerate(segment_tokens):
    if tok.kind != "WORD" or tok.lower not in _EMBEDDED_WH_WORDS:
      continue
    if pos == 0:
      return True
    prev = segment_tokens[pos - 1]
    if prev.kind == "PUNCT" and prev.raw == ",":
      return True
    if prev.lower in {"and", "on", "average"}:
      return True
  return False


def align_asr_casing_and_punctuation(
    cleaned: str,
    raw_question: str,
    evidence: str,
) -> str:
  """Restores multi-sentence caps, evidence identifier casing, and `?`/`.`.

  Args:
    cleaned: Disfluency-stripped utterance string.
    raw_question: Unmodified input utterance (used to detect all-lower ASR).
    evidence: Optional domain formula / note supplying canonical entity casing.

  Returns:
    Canonicalized utterance aligned with written benchmark formatting.
  """
  if not cleaned:
    return cleaned

  # Mixed-case written queries already carry author casing and punctuation;
  # only capitalize the opening character in case a wake-phrase was stripped.
  if raw_question != raw_question.lower():
    return cleaned[0].upper() + cleaned[1:]

  # Restore lowercase ASR entity/code spans in one Aho-Corasick pass.
  if evidence:
    cased_spans = _extract_evidence_casing_spans(evidence)
    if cased_spans:
      span_ac = AhoCorasickAutomaton(
          tuple((s, s) for s in cased_spans),
          case_sensitive=False,
      )
      cleaned = span_ac.replace_all(
          cleaned,
          replacer=lambda m: (
              m.payload if cleaned[m.start : m.end].islower() else cleaned[m.start : m.end]
          ),
          boundary_aware=True,
      )

  # Single-pass FSM for multi-sentence capitalization and terminal `?`/`.` check.
  tokens = tokenize_utterance(cleaned)
  if not tokens:
    return cleaned
  chars = list(cleaned)
  cap_next = True
  last_segment: list[LexToken] = []
  prev_tok: LexToken | None = None

  for tok in tokens:
    if tok.kind == "PUNCT" and tok.raw in ".?!":
      is_abbrev = (
          tok.raw == "."
          and prev_tok is not None
          and prev_tok.kind == "WORD"
          and prev_tok.lower in NON_TERMINAL_ABBREVS
      )
      if not is_abbrev:
        cap_next = True
        last_segment.clear()
      else:
        last_segment.append(tok)
    else:
      last_segment.append(tok)
      if cap_next and tok.kind == "WORD" and chars[tok.start].islower():
        chars[tok.start] = chars[tok.start].upper()
        cap_next = False
      elif cap_next and tok.kind in {"WORD", "NUMBER", "QUOTE"}:
        cap_next = False
    prev_tok = tok

  result = "".join(chars)
  if result[-1] not in ".?!;":
    suffix = "?" if _is_interrogative_segment(last_segment) else "."
    result = f"{result}{suffix}"
  return result


def clean_asr_utterance(raw_question: str, evidence: str = "") -> tuple[str, int]:
  """Strips ASR disfluencies/stutters via a single-pass token FSM.

  Args:
    raw_question: Raw user question or spoken ASR transcript.
    evidence: Optional external domain knowledge / formula note.

  Returns:
    Tuple `(canonical_question, disfluencies_removed_count)`.
  """
  stripped = raw_question.strip()
  if not stripped:
    return ("", 0)
  tokens = tokenize_utterance(stripped)
  n = len(tokens)
  emitted: list[tuple[LexToken, str]] = []
  preamble_buf: list[LexToken] = []
  in_preamble = True
  removed_count = 0
  prev_word_lower = ""
  prev_was_word = False
  idx = 0

  def _skip_trailing_comma(cur_idx: int) -> int:
    """Consumes an optional comma immediately trailing a stripped filler."""
    if cur_idx < n and tokens[cur_idx].kind == "PUNCT" and tokens[cur_idx].raw == ",":
      return cur_idx + 1
    return cur_idx

  def _emit_body_token(tok: LexToken) -> None:
    """Applies stutter collapsing and spoken-number conversion in BODY state."""
    nonlocal removed_count, prev_word_lower, prev_was_word
    # Collapse consecutive identical alphabetical word stutters (`the the`).
    if (
        prev_was_word
        and tok.kind == "WORD"
        and len(tok.raw) >= 2
        and tok.raw.isalpha()
        and prev_word_lower == tok.lower
    ):
      removed_count += 1
      return
    # Convert spoken cardinals (`top five` -> `top 5`) with `one` guard.
    out_raw = tok.raw
    if tok.kind == "WORD" and tok.lower in _SPOKEN_NUMBERS and (
        prev_word_lower in _CARDINAL_TRIGGER_PREV or tok.lower != "one"
    ):
      out_raw = _SPOKEN_NUMBERS[tok.lower]
    emitted.append((tok, out_raw))
    if tok.kind == "WORD":
      prev_word_lower = tok.lower
      prev_was_word = True
    else:
      prev_was_word = False
      if tok.kind != "PUNCT":
        prev_word_lower = tok.lower

  # Single-pass PREAMBLE -> BODY finite-state transducer over `tokens`.
  while idx < n:
    tok = tokens[idx]
    nxt = tokens[idx + 1] if (idx + 1) < n else None

    # Strip elongated vocalizations (`uh`, `um`, `er`, `ah`, `hmm`) in any state.
    if tok.kind == "WORD" and _is_vocal_filler(tok.lower):
      removed_count += 1
      idx = _skip_trailing_comma(idx + 1)
      continue

    if in_preamble:
      # Buffer optional conversational wake words (`hey`, `okay`, `so`, `well`, `please`).
      if tok.kind == "WORD" and tok.lower in _WAKE_PREFIX_WORDS:
        preamble_buf.append(tok)
        idx += 1
        continue
      # Buffer optional modal polite questions (`can you`, `could you`).
      if (
          nxt is not None
          and tok.kind == "WORD"
          and nxt.kind == "WORD"
          and (tok.lower, nxt.lower) in _WAKE_MODAL_BIGRAMS
      ):
        preamble_buf.extend((tok, nxt))
        idx += 2
        continue
      # Complete wake preamble when followed by `tell me`/`show me`/`find out`/`let me know`.
      if (
          nxt is not None
          and tok.kind == "WORD"
          and nxt.kind == "WORD"
          and (tok.lower, nxt.lower) in _WAKE_TERMINAL_BIGRAMS
      ):
        if (tok.lower, nxt.lower) == ("let", "me"):
          if idx + 2 < n and tokens[idx + 2].lower == "know":
            removed_count += 1
            preamble_buf.clear()
            in_preamble = False
            idx = _skip_trailing_comma(idx + 3)
            continue
        else:
          removed_count += 1
          preamble_buf.clear()
          in_preamble = False
          idx = _skip_trailing_comma(idx + 2)
          continue
      # Non-wake token reached: flush buffered tokens into BODY state.
      in_preamble = False
      for buf_tok in preamble_buf:
        _emit_body_token(buf_tok)
      preamble_buf.clear()

    # In BODY state: strip mid-clause bigram fillers (`you know`, `i mean`).
    if (
        nxt is not None
        and tok.kind == "WORD"
        and nxt.kind == "WORD"
        and (tok.lower, nxt.lower) in _FILLER_BIGRAMS
    ):
      removed_count += 1
      idx = _skip_trailing_comma(idx + 2)
      continue

    # Strip `sort of` / `kind of` unless preceded by an interrogative/determiner.
    if (
        nxt is not None
        and tok.kind == "WORD"
        and nxt.kind == "WORD"
        and (tok.lower, nxt.lower) in _GUARDED_FILLER_BIGRAMS
        and prev_word_lower not in _KIND_OF_GUARDS
    ):
      removed_count += 1
      idx = _skip_trailing_comma(idx + 2)
      continue

    _emit_body_token(tok)
    idx += 1

  # Flush trailing preamble buffer if the entire utterance was wake words.
  for buf_tok in preamble_buf:
    _emit_body_token(buf_tok)

  cleaned = _reconstruct_tokens(stripped, emitted).strip()
  cleaned = align_asr_casing_and_punctuation(cleaned, raw_question, evidence)
  return (cleaned or stripped, removed_count)
