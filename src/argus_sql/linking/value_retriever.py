"""Character 3-gram inverted index for fast database cell-value grounding.

Addresses dirty string literals and spoken entity abbreviations (CHESS-style
value retrieval) by indexing distinct TEXT cells once per database, suppressing
schema-token prefix noise and high-cardinality hierarchy shadows in < 1 ms.
"""

from collections.abc import Iterable
from collections.abc import Sequence
import difflib
import logging
import pathlib
import re
import sqlite3

from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import IntentFrame
from argus_sql.contracts.types import ValueMatch
from argus_sql.linking.exceptions import ValueIndexError
from argus_sql.linking.types import ValueIndexEntry

# Module-level logger following Google Python Style Guide conventions.
_LOGGER = logging.getLogger(__name__)

# Common English words skipped when probing bare tokens against cell literals.
_TOKEN_PROBE_STOPWORDS: frozenset[str] = frozenset({
    "all", "among", "and", "any", "are", "average", "between", "both", "but",
    "can", "consider", "count", "did", "difference", "does", "each", "eligible",
    "find", "for", "from", "full", "give", "had", "has", "have", "her",
    "highest", "his", "how", "identify", "include", "indicate", "into", "its",
    "least", "list", "lowest", "many", "more", "most", "much", "not", "number",
    "only", "other", "our", "out", "over", "paid", "per", "percentage",
    "please", "provide", "rank", "rate", "ratio", "show", "some", "state",
    "such", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "top", "total", "under", "valid", "was", "were",
    "what", "when", "where", "which", "who", "whom", "whose", "with", "year",
    "years",
})


def _compute_trigrams(normalized_text: str) -> frozenset[str]:
  """Extracts boundary-padded character 3-grams from a normalized string."""
  # Two-space boundary padding boosts prefix/suffix weighting ('  calif  ').
  padded = f"  {normalized_text}  "
  if len(padded) < 3:
    return frozenset({padded})
  return frozenset(padded[i : i + 3] for i in range(len(padded) - 2))


def _normalize_cell_text(value: str) -> str:
  """Lowercases and collapses internal whitespace for fuzzy string matching."""
  # Canonicalize casing and multi-space runs before trigram extraction.
  return " ".join(value.lower().strip().split())


def _split_column_words(identifier: str) -> set[str]:
  """Splits CamelCase/snake_case column names into lowercase noun tokens."""
  # Expand CamelCase boundaries ('MailCity' -> 'Mail City') before tokenizing.
  decamel = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", identifier)
  # Exclude generic suffix tokens ('name', 'code', 'id') shared across columns.
  return {
      tok
      for tok in re.findall(r"[a-z]{3,}", decamel.lower())
      if tok not in {"name", "code", "num", "type", "id"}
  }


def _extract_schema_vocabulary(
    schema: DatabaseSchema,
) -> tuple[frozenset[str], frozenset[str]]:
  """Extracts identifier tokens and parenthetical qualifiers from `schema`."""
  # Collect both parenthetical qualifiers ('K-12') and singular/plural stems.
  tokens: set[str] = set()
  qualifiers: set[str] = set()
  for table in schema.tables:
    for raw_ident in (table.name, *(col.name for col in table.columns)):
      # Parenthetical qualifiers inside column names are schema tags, not cells.
      for paren in re.findall(r"\(([^()]{2,25})\)", raw_ident):
        norm_q = _normalize_cell_text(paren)
        if norm_q:
          qualifiers.add(norm_q)
      decamel = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", raw_ident)
      # Record singular and plural forms so bare column words never prefix-match.
      for word in re.findall(r"[a-z]{3,}", decamel.lower()):
        tokens.add(word)
        tokens.add(f"{word}s")
        if len(word) >= 5 and word.endswith("s"):
          tokens.add(word[:-1])
  return (frozenset(tokens), frozenset(qualifiers))


def _disambiguate_hierarchy_matches(
    matches: Iterable[ValueMatch],
    question_lower: str,
    cardinality_by_col: dict[tuple[str, str], int],
) -> list[ValueMatch]:
  """Suppresses `Mail*` duplicates and high-cardinality hierarchy shadows."""
  items = list(matches)
  q_words = set(re.findall(r"[a-z]{3,}", question_lower))
  mentions_mail = "mail" in q_words or "mailing" in q_words

  # Track non-Mail matches on the same table and low-cardinality exact dimensions.
  non_mail_pairs: set[tuple[str, str]] = set()
  low_card_exact_vals: set[str] = set()
  for vm in items:
    t_low, c_low = vm.table_name.lower(), vm.column_name.lower()
    norm_val = _normalize_cell_text(vm.matched_value)
    if not c_low.startswith("mail"):
      non_mail_pairs.add((t_low, norm_val))
    card = cardinality_by_col.get((t_low, c_low), 0)
    if vm.similarity_score >= 0.99 and 0 < card <= 120:
      low_card_exact_vals.add(norm_val)

  filtered: list[ValueMatch] = []
  for vm in items:
    t_low, c_low = vm.table_name.lower(), vm.column_name.lower()
    norm_val = _normalize_cell_text(vm.matched_value)
    # 1. Drop redundant `MailCity`/`MailStreet` mirror hits unless asked for mail.
    if (
        c_low.startswith("mail")
        and not mentions_mail
        and (t_low, norm_val) in non_mail_pairs
    ):
      continue
    # 2. Drop high-cardinality shadows (`City`, `District`, `School`) when the
    # identical value matches a canonical low-cardinality dimension (`County`)
    # and the high-cardinality column's noun (`city`) is absent from the query.
    card = cardinality_by_col.get((t_low, c_low), 0)
    if (
        vm.similarity_score >= 0.99
        and norm_val in low_card_exact_vals
        and card >= 250
    ):
      col_nouns = _split_column_words(vm.column_name)
      if col_nouns and not (col_nouns & q_words):
        continue
    filtered.append(vm)
  return filtered


class CellValueRetriever:
  """Inverted character 3-gram retriever grounding query entities to DB cells.

  Attributes:
    entries: Immutable tuple of indexed distinct database cell values.
  """

  def __init__(
      self,
      entries: Sequence[ValueIndexEntry],
      schema_tokens: frozenset[str] = frozenset(),
      schema_qualifiers: frozenset[str] = frozenset(),
  ) -> None:
    """Builds the in-memory 3-gram posting list and cardinality map.

    Args:
      entries: Distinct ValueIndexEntry records extracted from a database.
      schema_tokens: Lowercase table/column identifier tokens from the schema.
      schema_qualifiers: Parenthetical column qualifiers (e.g., `'k-12'`).
    """
    self.entries: tuple[ValueIndexEntry, ...] = tuple(entries)
    self._schema_tokens = schema_tokens
    self._schema_qualifiers = schema_qualifiers
    self._postings: dict[str, list[int]] = {}
    self._exact_index: dict[str, list[int]] = {}
    self._cardinality_by_col: dict[tuple[str, str], int] = {}
    # Populate 3-gram postings, exact-value lookup, and column cardinalities.
    for idx, entry in enumerate(self.entries):
      self._exact_index.setdefault(entry.normalized_value, []).append(idx)
      col_key = (entry.table_name.lower(), entry.column_name.lower())
      if entry.column_cardinality > self._cardinality_by_col.get(col_key, 0):
        self._cardinality_by_col[col_key] = entry.column_cardinality
      for gram in entry.trigrams:
        self._postings.setdefault(gram, []).append(idx)

  @classmethod
  def from_schema_and_sqlite(
      cls,
      schema: DatabaseSchema,
      sqlite_path: pathlib.Path | None = None,
      max_distinct_per_column: int = 500,
  ) -> "CellValueRetriever":
    """Indexes distinct TEXT literals from SQLite (or schema value_examples).

    Args:
      schema: Target DatabaseSchema metadata.
      sqlite_path: Optional path to the SQLite database file opened read-only.
      max_distinct_per_column: Cap on distinct values loaded per TEXT column.

    Returns:
      Initialized CellValueRetriever ready for fast entity queries.

    Raises:
      ValueIndexError: If reading distinct values from `sqlite_path` fails.
    """
    entries: list[ValueIndexEntry] = []
    seen: set[tuple[str, str, str]] = set()
    schema_tokens, schema_qualifiers = _extract_schema_vocabulary(schema)

    # Always include schema-declared value_examples as baseline literals.
    for table in schema.tables:
      for col in table.columns:
        sample_card = len(col.value_examples)
        for sample in col.value_examples:
          norm = _normalize_cell_text(str(sample))
          key = (table.name, col.name, norm)
          if norm and len(norm) <= 80 and key not in seen:
            seen.add(key)
            entries.append(
                ValueIndexEntry(
                    table_name=table.name,
                    column_name=col.name,
                    raw_value=str(sample),
                    normalized_value=norm,
                    trigrams=_compute_trigrams(norm),
                    column_cardinality=sample_card,
                )
            )

    # Enrich with actual distinct TEXT cells via read-only RFC 8089 URI.
    if sqlite_path is not None and sqlite_path.is_file():
      uri = f"{sqlite_path.resolve().as_uri()}?mode=ro"
      try:
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
          # Scan declared TEXT/CHAR columns to skip numeric/BLOB primary keys.
          for table in schema.tables:
            t_ident = table.name.replace('"', '""')
            for col in table.columns:
              if (
                  "TEXT" not in col.data_type.upper()
                  and "CHAR" not in col.data_type.upper()
              ):
                continue
              c_ident = col.name.replace('"', '""')
              # Parameterize `max_distinct_per_column` to bound index RAM usage.
              query = (
                  f'SELECT DISTINCT "{c_ident}" FROM "{t_ident}" '
                  f'WHERE "{c_ident}" IS NOT NULL LIMIT ?'
              )
              raw_cells = [
                  row[0]
                  for row in conn.execute(query, (max_distinct_per_column,))
                  if isinstance(row[0], str)
              ]
              # Record distinct sample cardinality (`<=120` vs `>=250`) per col.
              col_card = len(raw_cells)
              # Allow 1-char enum codes ('S', 'D', 'M', 'F') on compact enums.
              min_len = 1 if col_card <= 16 else 2
              for cell_val in raw_cells:
                norm = _normalize_cell_text(cell_val)
                key = (table.name, col.name, norm)
                if min_len <= len(norm) <= 80 and key not in seen:
                  seen.add(key)
                  entries.append(
                      ValueIndexEntry(
                          table_name=table.name,
                          column_name=col.name,
                          raw_value=cell_val,
                          normalized_value=norm,
                          trigrams=_compute_trigrams(norm),
                          column_cardinality=col_card,
                      )
                  )
        finally:
          conn.close()
      except sqlite3.Error as err:
        raise ValueIndexError(
            f"Failed indexing distinct values for '{schema.db_id}'."
        ) from err

    _LOGGER.debug(
        "Indexed %d distinct cell values for database %s",
        len(entries),
        schema.db_id,
    )
    return cls(
        entries,
        schema_tokens=schema_tokens,
        schema_qualifiers=schema_qualifiers,
    )

  def retrieve(
      self,
      intent: IntentFrame,
      top_k: int = 8,
      min_similarity: float = 0.45,
  ) -> tuple[ValueMatch, ...]:
    """Retrieves top grounded database literals matching `intent` spans.

    Args:
      intent: Canonicalized IntentFrame containing extracted entity spans.
      top_k: Maximum number of ValueMatch results to return.
      min_similarity: Minimum similarity score required in `[0.0, 1.0]`.

    Returns:
      Tuple of deduplicated ValueMatch objects sorted by similarity descending.
    """
    if not self.entries:
      return ()

    entity_set = set(intent.entities)
    raw_low = intent.raw_question.lower()
    query_spans: list[str] = list(intent.entities)
    for raw_tok in re.split(r"[\s,;/()—–-]+", intent.canonical_question):
      cleaned = raw_tok.strip(".,?!'\"()").strip()
      if (
          len(cleaned) >= 2
          and not cleaned.isdigit()
          and cleaned.lower() not in _TOKEN_PROBE_STOPWORDS
          and cleaned not in query_spans
      ):
        query_spans.append(cleaned)

    best_by_cell: dict[tuple[str, str, str], ValueMatch] = {}
    for span in query_spans:
      norm_span = _normalize_cell_text(span)
      if not norm_span or norm_span in _TOKEN_PROBE_STOPWORDS:
        continue
      is_quoted = f"'{norm_span}'" in raw_low or f'"{norm_span}"' in raw_low
      # Skip parenthetical column qualifiers (`K-12` in `Enrollment (K-12)`)
      # unless the user explicitly quoted the literal string.
      if norm_span in self._schema_qualifiers and not is_quoted:
        continue

      # Short 1-3 char codes or bare schema identifier tokens ('school', 'free',
      # 'charter') require exact cell match so they never trigger prefix noise!
      is_schema_word = (
          " " not in norm_span
          and norm_span in self._schema_tokens
          and not is_quoted
      )
      if len(norm_span) <= 3 or is_schema_word:
        for entry_idx in self._exact_index.get(norm_span, ()):
          entry = self.entries[entry_idx]
          if len(norm_span) == 1 and span not in entity_set:
            continue
          cell_key = (entry.table_name, entry.column_name, entry.raw_value)
          best_by_cell[cell_key] = ValueMatch(
              table_name=entry.table_name,
              column_name=entry.column_name,
              matched_value=entry.raw_value,
              query_span=span,
              similarity_score=1.0,
          )
        continue

      has_exact_hit = norm_span in self._exact_index
      in_entities = span in entity_set or (not entity_set and span[:1].isupper())
      span_grams = _compute_trigrams(norm_span)
      hit_counts: dict[int, int] = {}
      for gram in span_grams:
        for entry_idx in self._postings.get(gram, ()):
          hit_counts[entry_idx] = hit_counts.get(entry_idx, 0) + 1

      # Stage 1: Fast O(1) 3-gram Jaccard & tight abbreviation prefix pre-filter.
      # Bare non-entity sentence words only match exact cells or whole-word
      # prefixes ('locally' -> 'Locally funded'), never random surname typos!
      shortlist: list[tuple[float, float, int]] = []
      for entry_idx, shared in hit_counts.items():
        entry = self.entries[entry_idx]
        union_len = len(span_grams) + len(entry.trigrams) - shared
        jaccard = shared / max(1, union_len)
        is_exact = norm_span == entry.normalized_value
        len_ratio = len(norm_span) / max(1, len(entry.normalized_value))
        is_tight_prefix = (
            not is_exact
            and not has_exact_hit
            and len(norm_span) >= 4
            and entry.normalized_value.startswith(norm_span)
            and (
                (" " not in entry.normalized_value and len_ratio >= 0.50)
                or (
                    in_entities
                    and len_ratio >= 0.45
                    and (
                        entry.normalized_value.startswith(f"{norm_span} ")
                        or " " in norm_span
                    )
                )
            )
        )
        allow_fuzzy = in_entities and not has_exact_hit and jaccard >= 0.25
        if is_exact or is_tight_prefix or allow_fuzzy:
          shortlist.append(
              (1.0 if (is_exact or is_tight_prefix) else jaccard, jaccard, entry_idx)
          )
      shortlist.sort(reverse=True)

      # Stage 2: Blend 3-gram Jaccard recall with SequenceMatcher edit ratio.
      # Enforce a 0.72 floor on non-exact fuzzy matches so surname typos vanish.
      effective_floor = max(min_similarity, 0.72)
      for prio, jaccard, entry_idx in shortlist[:64]:
        entry = self.entries[entry_idx]
        # Exact normalized cell equality receives full 1.0 grounding weight.
        if norm_span == entry.normalized_value:
          score = 1.0
        else:
          seq_ratio = difflib.SequenceMatcher(
              None, norm_span, entry.normalized_value
          ).ratio()
          score = (
              max(0.86, seq_ratio)
              if prio >= 1.0
              else (0.45 * jaccard + 0.55 * seq_ratio)
          )
          if score < effective_floor:
            continue

        if score < min_similarity:
          continue
        # Retain only the highest-scoring span match for each unique cell value.
        cell_key = (entry.table_name, entry.column_name, entry.raw_value)
        existing = best_by_cell.get(cell_key)
        if existing is None or score > existing.similarity_score:
          best_by_cell[cell_key] = ValueMatch(
              table_name=entry.table_name,
              column_name=entry.column_name,
              matched_value=entry.raw_value,
              query_span=span,
              similarity_score=round(score, 4),
          )

    # Disambiguate `Mail*` duplicates and high-cardinality hierarchy shadows.
    pruned = _disambiguate_hierarchy_matches(
        best_by_cell.values(),
        intent.canonical_question.lower(),
        self._cardinality_by_col,
    )
    # Sort descending so prompts prioritize the strongest grounded literals.
    ranked = sorted(pruned, key=lambda m: m.similarity_score, reverse=True)
    return tuple(ranked[:top_k])
