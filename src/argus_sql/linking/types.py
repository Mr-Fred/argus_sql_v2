"""Configuration and scoring contracts for hybrid schema and value linking."""

import dataclasses


# Stores indexed cell literal metadata and precomputed character 3-grams.
@dataclasses.dataclass(frozen=True)
class ValueIndexEntry:
  """Single distinct database string value indexed for fuzzy n-gram lookup.

  Attributes:
    table_name: Physical table owning the column.
    column_name: Physical TEXT column holding `raw_value`.
    raw_value: Exact string value stored in SQLite.
    normalized_value: Lowercased whitespace-collapsed string for comparison.
    trigrams: Frozenset of padded character 3-grams for Jaccard retrieval.
  """

  table_name: str
  column_name: str
  raw_value: str
  normalized_value: str
  trigrams: frozenset[str]


# Breakdown of lexical, evidence, and cell-value relevance per column.
@dataclasses.dataclass(frozen=True)
class ColumnScore:
  """Interpretable multi-signal relevance score for one database column.

  Attributes:
    table_name: Physical table name.
    column_name: Physical column name.
    lexical_score: Overlap score between question tokens and column metadata.
    evidence_score: Overlap score with external domain evidence formulas.
    value_score: Maximum grounded cell-value similarity on this column.
  """

  table_name: str
  column_name: str
  lexical_score: float
  evidence_score: float
  value_score: float

  @property
  def total_score(self) -> float:
    """Computes weighted sum prioritizing grounded values and evidence cues."""
    # Grounded cell hits and domain evidence carry higher precision than names.
    return (
        self.lexical_score
        + (1.35 * self.evidence_score)
        + (1.65 * self.value_score)
    )


# Tunable hyperparameters enabling clean ablation studies across systems.
@dataclasses.dataclass(frozen=True)
class SchemaLinkerConfig:
  """Configuration controlling pruning aggressiveness and ablation toggles.

  Attributes:
    max_columns_per_table: Upper bound on non-key columns kept per table.
    min_column_score: Minimum relevance score to retain a non-key column.
    enable_fk_closure: Whether to recover bridge tables via FK shortest paths.
    enable_value_grounding: Whether to inject n-gram grounded cell literals.
    top_k_values: Maximum grounded `ValueMatch` items retained per query.
  """

  # Keep up to 8 non-key columns per table to avoid DDL context bloat.
  max_columns_per_table: int = 8
  # Threshold suppresses random 1-character substring collisions.
  min_column_score: float = 0.18
  enable_fk_closure: bool = True
  enable_value_grounding: bool = True
  top_k_values: int = 6
