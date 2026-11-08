"""Character 3-gram inverted index for fast database cell-value grounding.

Addresses dirty string literals and spoken entity abbreviations (CHESS-style
value retrieval) by indexing distinct TEXT cells once per database and matching
query spans in sub-millisecond time without per-query SQL table scans.
"""

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


# Common English words skipped when probing bare tokens against cell literals.
# Filtering stopwords prevents pointless postings lookups on function words.
_TOKEN_PROBE_STOPWORDS: frozenset[str] = frozenset({
    "all", "among", "and", "any", "are", "average", "between", "both", "but",
    "can", "count", "did", "does", "each", "for", "from", "give", "had", "has",
    "have", "her", "highest", "his", "how", "into", "its", "least", "list",
    "lowest", "many", "more", "most", "much", "not", "number", "only", "other",
    "our", "out", "over", "paid", "per", "please", "percentage", "ratio",
    "rate", "show", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "top", "total", "under",
    "was", "were", "what", "when", "where", "which", "who", "whom", "with",
    "year", "years",
})


class CellValueRetriever:
  """Inverted character 3-gram retriever grounding query entities to DB cells.

  Attributes:
    entries: Immutable tuple of indexed distinct database cell values.
  """

  def __init__(self, entries: Sequence[ValueIndexEntry]) -> None:
    """Builds the in-memory 3-gram posting list over `entries`.

    Args:
      entries: Distinct ValueIndexEntry records extracted from a database.
    """
    self.entries: tuple[ValueIndexEntry, ...] = tuple(entries)
    self._postings: dict[str, list[int]] = {}
    self._exact_index: dict[str, list[int]] = {}
    # Populate both 3-gram postings and exact-value lookup in O(N).
    for idx, entry in enumerate(self.entries):
      self._exact_index.setdefault(entry.normalized_value, []).append(idx)
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

    # Always include schema-declared value_examples as baseline literals.
    for table in schema.tables:
      for col in table.columns:
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
                )
            )

    # Optionally enrich with actual distinct TEXT cells via read-only URI.
    # Using `as_uri()` percent-encodes spaces or `#`/`?` characters safely.
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
              # Allow 1-char enum codes ('S', 'D', 'M', 'F') on low-cardinality cols.
              min_len = 1 if len(raw_cells) <= 16 else 2
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
    return cls(entries)

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

    # Probe extracted entity spans and distinctive tokens (including 2-3 char
    # ISO/segment codes like 'sme', 'eur', 'czk', 'ny' on Voice transcripts).
    query_spans: list[str] = list(intent.entities)
    for raw_tok in re.split(r"[\s,;/()—–-]+", intent.canonical_question):
      cleaned = raw_tok.strip(".,?!'\"()").strip()
      if (
          len(cleaned) >= 2
          and not cleaned.isdigit()
          and cleaned.lower() not in _TOKEN_PROBE_STOPWORDS
      ):
        if cleaned not in query_spans:
          query_spans.append(cleaned)

    # Deduplicate matches by `(table, column, raw_value)` keeping max score.
    best_by_cell: dict[tuple[str, str, str], ValueMatch] = {}
    for span in query_spans:
      norm_span = _normalize_cell_text(span)
      if not norm_span:
        continue

      # Short 1-3 char codes ('eur', 'sme', 'r') require exact cell match so
      # short tokens never trigger noisy fuzzy substring collisions!
      if len(norm_span) <= 3:
        for entry_idx in self._exact_index.get(norm_span, ()):
          entry = self.entries[entry_idx]
          # Only ground 1-char codes when explicitly quoted in `intent.entities`.
          if len(norm_span) == 1 and span not in intent.entities:
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

      span_grams = _compute_trigrams(norm_span)
      # Accumulate posting-list overlap counts across matching 3-grams.
      hit_counts: dict[int, int] = {}
      for gram in span_grams:
        for entry_idx in self._postings.get(gram, ()):
          hit_counts[entry_idx] = hit_counts.get(entry_idx, 0) + 1

      # Stage 1: Fast O(1) 3-gram Jaccard & prefix pre-filter so quadratic
      # `SequenceMatcher` runs on at most top-64 candidates on 15k-cell DBs.
      shortlist: list[tuple[float, float, int]] = []
      for entry_idx, shared in hit_counts.items():
        entry = self.entries[entry_idx]
        union_len = len(span_grams) + len(entry.trigrams) - shared
        jaccard = shared / max(1, union_len)
        is_exact_or_prefix = norm_span == entry.normalized_value or (
            entry.normalized_value.startswith(norm_span) and len(norm_span) >= 4
        )
        if is_exact_or_prefix or jaccard >= 0.08:
          shortlist.append(
              (1.0 if is_exact_or_prefix else jaccard, jaccard, entry_idx)
          )
      shortlist.sort(reverse=True)

      # Stage 2: Blend 3-gram Jaccard recall with SequenceMatcher edit ratio.
      for _, jaccard, entry_idx in shortlist[:64]:
        entry = self.entries[entry_idx]
        # Reward exact or prefix matches (e.g., 'Calif' -> 'California').
        if norm_span == entry.normalized_value:
          score = 1.0
        else:
          seq_ratio = difflib.SequenceMatcher(
              None, norm_span, entry.normalized_value
          ).ratio()
          if entry.normalized_value.startswith(norm_span) and len(norm_span) >= 4:
            score = max(0.86, seq_ratio)
          else:
            score = 0.45 * jaccard + 0.55 * seq_ratio

        # Discard weak partial n-gram collisions below similarity threshold.
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

    # Sort descending so prompts prioritize the strongest grounded literals.
    ranked = sorted(
        best_by_cell.values(),
        key=lambda m: m.similarity_score,
        reverse=True,
    )
    return tuple(ranked[:top_k])
