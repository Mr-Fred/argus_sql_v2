"""ASR disfluency stripper, spoken cardinal converter, and case/punct restorer.

Isolates pure lexical speech-to-text transcript cleanup—spoken filler removal,
consecutive word-stutter collapsing, cardinal number conversion, `evidence`
identifier case restoration, and multi-sentence capitalization/punctuation—so
voice utterances align faithfully with clean written queries.
"""

import re

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
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15",
    "twenty": "20", "fifty": "50", "hundred": "100",
}

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

# Matches interrogative sentence heads so stripped ASR questions regain `?`.
_INTERROGATIVE_HEAD_RE = re.compile(
    r"^(?:what|which|where|when|who|whom|whose|why|how|is|are|was|were"
    r"|do|does|did|can|could|has|have|had)\b",
    re.IGNORECASE,
)


def normalize_spoken_numbers(text: str) -> str:
  """Converts spoken number tokens after limit/comparison words into digits.

  Args:
    text: Partially cleaned natural language or ASR utterance string.

  Returns:
    Utterance with spoken cardinals ('top five') converted to digits ('top 5').
  """
  tokens = text.split()
  converted: list[str] = []
  # Convert 'one' only after ranking/bound words so 'which one' stays intact.
  trigger_prev = {
      "top", "bottom", "first", "last", "over", "under",
      "than", "least", "most", "above", "below",
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

  # Align lowercase ASR entity/code spans ('tr004_8_9', 'marvel comics', 'dc')
  # with canonical casing attested in external `evidence`.
  if evidence:
    cased_spans: list[str] = []
    # 1. Extract multi-word Proper Noun phrases ('Marvel Comics') from evidence.
    for m_prop in re.finditer(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", evidence
    ):
      phrase = m_prop.group(1)
      if phrase.split()[0].lower() not in ENTITY_STOPWORDS:
        cased_spans.append(phrase)
    # 2. Extract uppercase/alphanumeric domain codes ('TR004_8_9', 'DC', 'K-12').
    for m_code in re.finditer(
        r"\b([A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*)\b", evidence
    ):
      tok = m_code.group(1)
      if len(tok) < 2 or tok.upper() in SQL_FORMULA_UPPER:
        continue
      has_up = any(ch.isupper() for ch in tok)
      is_domain_code = tok.isupper() or (
          has_up
          and (
              any(ch.isdigit() for ch in tok)
              or "_" in tok
              or "-" in tok
              or any(ch.isupper() for ch in tok[1:])
          )
      )
      if is_domain_code:
        cased_spans.append(tok)
    # Apply longest-first so 'Marvel Comics' precedes single tokens.
    for span in sorted(set(cased_spans), key=len, reverse=True):
      pat = rf"(?<![A-Za-z0-9_]){re.escape(span)}(?![A-Za-z0-9_])"
      cleaned = re.sub(
          pat,
          lambda m, target=span: target if m.group(0).islower() else m.group(0),
          cleaned,
          flags=re.IGNORECASE,
      )

  # Capitalize both the opening character and any sentence after `.`, `?`, `!`.
  cleaned = re.sub(
      r"(^|[.?!]\s+)([a-z])",
      lambda m: m.group(1) + m.group(2).upper(),
      cleaned,
  )

  # Restore terminal `?` or `.` when lowercase ASR stripped ending punctuation.
  if raw_question == raw_question.lower() and cleaned[-1] not in ".?!;":
    last_seg = re.split(r"[.?!]\s+", cleaned)[-1].strip()
    is_question = bool(_INTERROGATIVE_HEAD_RE.match(last_seg)) or (
        "?" not in cleaned
        and bool(
            re.search(
                r"\b(?:what|which|who|where|when|how\s+many|how\s+much)\b",
                last_seg,
                re.IGNORECASE,
            )
        )
    )
    cleaned = f"{cleaned}?" if is_question else f"{cleaned}."
  return cleaned


def clean_asr_utterance(raw_question: str, evidence: str = "") -> tuple[str, int]:
  """Strips ASR disfluencies/stutters and restores canonical casing/punctuation.

  Args:
    raw_question: Raw user question or spoken ASR transcript.
    evidence: Optional external domain knowledge / formula note.

  Returns:
    Tuple `(canonical_question, disfluencies_removed_count)`.
  """
  cleaned = raw_question.strip()
  removed_count = 0

  # Strip spoken filler tokens and conversational wake preambles iteratively.
  for pattern in _DISFLUENCY_PATTERNS:
    cleaned, n_subs = pattern.subn("", cleaned)
    removed_count += n_subs

  # Collapse consecutive word stutters ('the the' -> 'the').
  cleaned, n_stutters = _STUTTER_RE.subn(r"\1", cleaned)
  removed_count += n_stutters

  # Normalize spoken cardinals, collapse whitespace, and align ASR casing/punct.
  cleaned = normalize_spoken_numbers(cleaned)
  cleaned = re.sub(r"\s+", " ", cleaned).strip()
  cleaned = align_asr_casing_and_punctuation(cleaned, raw_question, evidence)
  return (cleaned or raw_question.strip(), removed_count)
