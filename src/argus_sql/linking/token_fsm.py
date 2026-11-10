"""Single-pass token-level FSM lexer and Aho-Corasick multi-pattern matcher.

Provides deterministic linear-time pattern matching and tokenization primitives
replacing brittle regex cascades across Stage 1 voice normalization, entity
span extraction, spoken expansion lookup, and projection arity cue detection.
"""

from collections import deque
from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
from typing import Generic, TypeVar

# Generic payload type stored on terminal Aho-Corasick pattern states.
_PayloadT = TypeVar("_PayloadT")

# Known abbreviation stems whose trailing period does not end a sentence.
NON_TERMINAL_ABBREVS: frozenset[str] = frozenset({
    "no",
    "vs",
    "dr",
    "mr",
    "mrs",
    "ms",
    "st",
})


def _is_word_char(ch: str) -> bool:
  """Returns True when `ch` belongs to an identifier/word boundary class."""
  return ch.isalnum() or ch == "_"


# Immutable match span emitted by `AhoCorasickAutomaton.find_all`.
@dataclasses.dataclass(frozen=True)
class AhoCorasickMatch(Generic[_PayloadT]):
  """Single matched pattern span and associated typed payload.

  Attributes:
    start: Inclusive 0-based start character offset in the scanned text.
    end: Exclusive 0-based end character offset in the scanned text.
    pattern: Canonical pattern string registered in the automaton.
    payload: Caller-supplied metadata or replacement string for the pattern.
  """

  start: int
  end: int
  pattern: str
  payload: _PayloadT


# Typed lexical token produced by the single-pass utterance FSM lexer.
@dataclasses.dataclass(frozen=True)
class LexToken:
  """Single lexical token with exact character offsets in the source string.

  Attributes:
    kind: Token class (`'WORD'`, `'NUMBER'`, `'QUOTE'`, or `'PUNCT'`).
    raw: Exact unmodified substring `text[start:end]`.
    lower: Lowercased token text (or unquoted interior for `'QUOTE'`).
    start: Inclusive character start index in source text.
    end: Exclusive character end index in source text.
  """

  kind: str
  raw: str
  lower: str
  start: int
  end: int


# Internal trie node for the Aho-Corasick failure-link state machine.
class _ACNode(Generic[_PayloadT]):
  """Trie state holding outgoing transitions, failure link, and outputs."""

  __slots__ = ("children", "fail", "outputs")

  def __init__(self) -> None:
    """Initializes an empty trie node."""
    self.children: dict[str, _ACNode[_PayloadT]] = {}
    self.fail: _ACNode[_PayloadT] | None = None
    self.outputs: list[tuple[str, _PayloadT]] = []


class AhoCorasickAutomaton(Generic[_PayloadT]):
  """Linear-time multi-pattern Aho-Corasick string matcher with boundary guards.

  Compiles an arbitrary dictionary of literal phrases into a deterministic
  trie with BFS failure links so all patterns match in a single `O(N)` pass.
  """

  def __init__(
      self,
      patterns: Sequence[tuple[str, _PayloadT]],
      case_sensitive: bool = False,
  ) -> None:
    """Builds the Aho-Corasick trie and failure-link transition graph.

    Args:
      patterns: Sequence of `(pattern_text, payload)` entries to match.
      case_sensitive: When False, normalizes both patterns and input to lower.
    """
    self._case_sensitive = case_sensitive
    self._root: _ACNode[_PayloadT] = _ACNode()
    # Insert every non-empty pattern into the prefix trie.
    for raw_pat, payload in patterns:
      if not raw_pat:
        continue
      norm_pat = raw_pat if case_sensitive else raw_pat.lower()
      node = self._root
      for ch in norm_pat:
        nxt = node.children.get(ch)
        if nxt is None:
          nxt = _ACNode()
          node.children[ch] = nxt
        node = nxt
      node.outputs.append((norm_pat, payload))

    # Construct BFS failure links and propagate suffix dictionary outputs.
    queue: deque[_ACNode[_PayloadT]] = deque()
    self._root.fail = self._root
    # Depth-1 children always fail back to the trie root on mismatch.
    for child in self._root.children.values():
      child.fail = self._root
      queue.append(child)

    while queue:
      curr = queue.popleft()
      for ch, nxt in curr.children.items():
        # Walk failure links until an ancestor accepts `ch` or root is reached.
        fail_state = curr.fail if curr.fail is not None else self._root
        while fail_state is not self._root and ch not in fail_state.children:
          fail_state = fail_state.fail if fail_state.fail is not None else self._root
        nxt.fail = fail_state.children.get(ch, self._root)
        # Merge suffix outputs so nested shorter matches are never missed.
        if nxt.fail.outputs:
          nxt.outputs.extend(nxt.fail.outputs)
        queue.append(nxt)

  @staticmethod
  def _passes_boundary(text: str, start: int, end: int, pattern: str) -> bool:
    """Checks alphanumeric word boundaries around `text[start:end]`."""
    # Guard leading boundary whenever the pattern begins with a word character.
    if _is_word_char(pattern[0]) or pattern[0] == ".":
      if start > 0 and _is_word_char(text[start - 1]):
        return False
    # Guard trailing boundary whenever pattern ends with word char or dot.
    if _is_word_char(pattern[-1]) or pattern[-1] == ".":
      if end < len(text) and _is_word_char(text[end]):
        return False
    return True

  def find_all(
      self,
      text: str,
      boundary_aware: bool = True,
      resolve_overlaps: bool = True,
  ) -> tuple[AhoCorasickMatch[_PayloadT], ...]:
    """Scans `text` in a single pass and returns all matching pattern spans.

    Args:
      text: Input natural language or evidence string.
      boundary_aware: Enforce word boundaries around alphanumeric patterns.
      resolve_overlaps: Keep earliest-start, longest non-overlapping matches.

    Returns:
      Ordered tuple of `AhoCorasickMatch` records.
    """
    if not text or not self._root.children:
      return ()
    # Normalize case once upfront so the scan stays purely linear `O(N)`.
    haystack = text if self._case_sensitive else text.lower()
    raw_matches: list[AhoCorasickMatch[_PayloadT]] = []
    state = self._root

    # Stream each character once across failure-link transitions.
    for idx, ch in enumerate(haystack):
      while state is not self._root and ch not in state.children:
        state = state.fail if state.fail is not None else self._root
      state = state.children.get(ch, self._root)
      if not state.outputs:
        continue
      end = idx + 1
      # Emit every pattern terminating at `end` that passes boundary checks.
      for pat, payload in state.outputs:
        start = end - len(pat)
        if not boundary_aware or self._passes_boundary(haystack, start, end, pat):
          raw_matches.append(
              AhoCorasickMatch(
                  start=start,
                  end=end,
                  pattern=pat,
                  payload=payload,
              )
          )

    if not resolve_overlaps or len(raw_matches) <= 1:
      raw_matches.sort(key=lambda m: (m.start, -(m.end - m.start)))
      return tuple(raw_matches)

    # Greedily select earliest-start, longest-span non-overlapping matches.
    raw_matches.sort(key=lambda m: (m.start, -(m.end - m.start)))
    resolved: list[AhoCorasickMatch[_PayloadT]] = []
    cursor = 0
    for match in raw_matches:
      if match.start >= cursor:
        resolved.append(match)
        cursor = match.end
    return tuple(resolved)

  def replace_all(
      self,
      text: str,
      replacer: Callable[[AhoCorasickMatch[_PayloadT]], str] | None = None,
      boundary_aware: bool = True,
  ) -> str:
    """Replaces non-overlapping pattern matches across `text` in one pass.

    Args:
      text: Input string to rewrite.
      replacer: Optional callback mapping each match to replacement text.
      boundary_aware: Enforce word boundaries around alphanumeric patterns.

    Returns:
      Rewritten string with all non-overlapping matches substituted.
    """
    matches = self.find_all(
        text, boundary_aware=boundary_aware, resolve_overlaps=True
    )
    if not matches:
      return text
    # Stitch untouched slices and substituted spans in linear time.
    chunks: list[str] = []
    cursor = 0
    for match in matches:
      chunks.append(text[cursor : match.start])
      if replacer is not None:
        chunks.append(replacer(match))
      else:
        chunks.append(str(match.payload))
      cursor = match.end
    chunks.append(text[cursor:])
    return "".join(chunks)


def tokenize_utterance(text: str) -> tuple[LexToken, ...]:
  """Tokenizes `text` via a single-pass deterministic character FSM.

  Preserves exact offsets, quoted literals (`'...'` / `"..."`), compound domain
  codes (`TR004_8_9`, `K-12`, `0:01:27`), and punctuation boundaries without
  regex backtracking.

  Args:
    text: Input natural language, ASR transcript, or evidence string.

  Returns:
    Ordered tuple of `LexToken` instances spanning `text`.
  """
  tokens: list[LexToken] = []
  n = len(text)
  i = 0

  while i < n:
    ch = text[i]
    # Skip whitespace cleanly while preserving exact character offsets on tokens.
    if ch.isspace():
      i += 1
      continue

    # Enter quoted-literal state only when quote starts at a left word boundary
    # so English possessives (`school's`, `patients'`) are never mis-paired.
    if ch in ("'", '"') and (i == 0 or not _is_word_char(text[i - 1])):
      close_idx = text.find(ch, i + 1, min(n, i + 52))
      if close_idx != -1 and "\n" not in text[i + 1 : close_idx]:
        raw_q = text[i : close_idx + 1]
        inner = text[i + 1 : close_idx]
        tokens.append(
            LexToken(
                kind="QUOTE",
                raw=raw_q,
                lower=inner.lower(),
                start=i,
                end=close_idx + 1,
            )
        )
        i = close_idx + 1
        continue

    # Enter alphanumeric word/identifier/number state.
    if _is_word_char(ch):
      start = i
      i += 1
      while i < n:
        cur = text[i]
        if _is_word_char(cur):
          i += 1
        elif (
            cur in ("-", "&", ":", ".")
            and i + 1 < n
            and _is_word_char(text[i + 1])
        ):
          # Absorb internal hyphens, ampersands, time colons, and dots (`0.1`,
          # `0:01:27`, `K-12`, `AT&T`, `l.a`) while keeping trailing dots separate.
          i += 1
        else:
          break
      raw = text[start:i]
      has_alpha = any(c.isalpha() for c in raw)
      has_digit = any(c.isdigit() for c in raw)
      kind = "NUMBER" if (has_digit and not has_alpha) else "WORD"
      tokens.append(
          LexToken(kind=kind, raw=raw, lower=raw.lower(), start=start, end=i)
      )
      continue

    # Emit single-character punctuation / operator token.
    tokens.append(
        LexToken(kind="PUNCT", raw=ch, lower=ch.lower(), start=i, end=i + 1)
    )
    i += 1

  return tuple(tokens)
