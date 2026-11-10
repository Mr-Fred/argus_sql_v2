"""Immutable domain data contracts for Argus-SQL pipelines and benchmarks.

Models relational database schemas, natural language and spoken tasks, intent
frames, value-grounded linked contexts, SQL candidates, execution outcomes,
and semantic invariant diagnostics without external I/O dependencies.
"""

import dataclasses
import enum
from typing import Any


# Official BIRD-SQL query complexity tiers used for routing and evaluation.
class QueryDifficulty(str, enum.Enum):
  """Stratified difficulty tier aligned with the public BIRD-SQL benchmark."""

  SIMPLE = "simple"
  MODERATE = "moderate"
  CHALLENGING = "challenging"


# Represents metadata and representative literals for a single table column.
@dataclasses.dataclass(frozen=True)
class ColumnSchema:
  """Metadata describing a relational column in a target database table.

  Attributes:
    name: Exact physical column identifier in SQLite.
    data_type: Declared SQL affinity ('TEXT', 'INTEGER', 'REAL', 'DATE').
    is_primary_key: True when the column participates in the primary key.
    description: Human-readable column meaning from metadata CSVs if available.
    value_examples: Representative sample cell strings for prompt grounding.
    value_range: Optional `(min_value, max_value)` domain bounds from SQLite.
  """

  name: str
  data_type: str = "TEXT"
  is_primary_key: bool = False
  description: str = ""
  value_examples: tuple[str, ...] = ()
  value_range: tuple[str, str] | None = None


# Represents a directed foreign-key join constraint between two tables.
@dataclasses.dataclass(frozen=True)
class ForeignKeyEdge:
  """Foreign-key edge connecting `source_table.source_column` to target.

  Attributes:
    source_table: Child table holding the referencing foreign-key column.
    source_column: Referencing column name in `source_table`.
    target_table: Parent table holding the referenced key column.
    target_column: Referenced key column name in `target_table`.
  """

  source_table: str
  source_column: str
  target_table: str
  target_column: str


# Groups physical columns and optional cardinality hints for one table.
@dataclasses.dataclass(frozen=True)
class TableSchema:
  """Relational table definition containing ordered column schemas.

  Attributes:
    name: Exact physical table name in SQLite.
    columns: Ordered tuple of ColumnSchema definitions.
    row_count_hint: Optional approximate row count used in cost heuristics.
  """

  name: str
  columns: tuple[ColumnSchema, ...]
  row_count_hint: int = 0


# Complete immutable relational schema graph for a benchmark database.
@dataclasses.dataclass(frozen=True)
class DatabaseSchema:
  """Complete database schema graph including tables and foreign-key edges.

  Attributes:
    db_id: Unique database identifier (e.g., 'california_schools').
    tables: Tuple of TableSchema definitions in the database.
    foreign_keys: Tuple of ForeignKeyEdge join relationships.
  """

  db_id: str
  tables: tuple[TableSchema, ...]
  foreign_keys: tuple[ForeignKeyEdge, ...] = ()

  def table_by_name(self, name: str) -> TableSchema | None:
    """Looks up a TableSchema case-insensitively by table name.

    Args:
      name: Target table name to resolve.

    Returns:
      Matching TableSchema or None if absent.
    """
    # Normalize case because SQLite table identifiers are case-insensitive.
    lowered = name.lower()
    for table in self.tables:
      if table.name.lower() == lowered:
        return table
    return None


# Single benchmark evaluation task supporting both text and spoken utterances.
@dataclasses.dataclass(frozen=True)
class NLQueryTask:
  """Natural language or spoken voice query task evaluated against a database.

  Attributes:
    question_id: Unique integer or string benchmark task identifier.
    db_id: Target database identifier matching `DatabaseSchema.db_id`.
    question: User natural language question (or ASR voice transcript).
    evidence: Optional external domain knowledge / formula specification.
    difficulty: Stratified complexity tier ('simple', 'moderate', 'challenging').
    gold_sql: Optional reference SQL query for benchmark accuracy scoring.
    is_voice_perturbed: True if the question includes spoken ASR perturbations.
  """

  question_id: str
  db_id: str
  question: str
  evidence: str = ""
  difficulty: QueryDifficulty = QueryDifficulty.SIMPLE
  gold_sql: str = ""
  is_voice_perturbed: bool = False


# Grounded database cell literal matched via character n-gram retrieval.
@dataclasses.dataclass(frozen=True)
class ValueMatch:
  """Database cell literal matched against an entity span in the user query.

  Attributes:
    table_name: Physical table containing the matched value.
    column_name: Physical TEXT column containing the matched value.
    matched_value: Exact string literal stored inside the database cell.
    query_span: Token or phrase from the question that triggered retrieval.
    similarity_score: Normalized similarity score in `[0.0, 1.0]`.
  """

  table_name: str
  column_name: str
  matched_value: str
  query_span: str
  similarity_score: float


# Canonicalized representation of user intent extracted prior to SQL synthesis.
@dataclasses.dataclass(frozen=True)
class IntentFrame:
  """Structured semantic intent frame extracted from text or voice input.

  Attributes:
    raw_question: Original unmodified user question or ASR transcript.
    canonical_question: Cleaned question with fillers removed and numbers normalized.
    disfluencies_removed: Count of spoken filler phrases stripped.
    entities: Extracted candidate entity and literal spans for value lookup.
    requires_floating_division: True if rate/ratio/percentage cues appear.
    requires_extrema_order: True if superlative min/max/top-k cues appear.
    expected_projection_arity: Estimated column count requested (`>= 1`).
    complexity: Estimated query complexity routing tier.
    strict_single_column: True when formula evidence mandates a 1-column scalar.
    has_followup_projection: True when follow-up or `'their'` clauses require
      projected columns beyond a single scalar/identifier (`>= 2`).
  """

  # Preserves both raw utterance and normalized intent for downstream stages.
  raw_question: str
  canonical_question: str
  disfluencies_removed: int
  entities: tuple[str, ...]
  requires_floating_division: bool
  requires_extrema_order: bool
  expected_projection_arity: int
  complexity: QueryDifficulty
  strict_single_column: bool = False
  has_followup_projection: bool = False


# Output of hybrid value-grounded + Steiner FK-closure schema linking.
@dataclasses.dataclass(frozen=True)
class LinkedSchemaContext:
  """Pruned, join-complete schema context ready for SQL candidate generation.

  Attributes:
    db_id: Target database identifier.
    selected_tables: Ordered tuple of selected table names (including bridges).
    selected_columns: Mapping entries `(table_name, tuple_of_column_names)`.
    fk_closure_edges: Foreign-key edges connecting the selected subgraph.
    bridge_tables_added: Intermediate join tables recovered via FK closure.
    value_matches: Top grounded cell literals attached to selected columns.
    ddl_prompt: Compact `CREATE TABLE` DDL string formatted for LLM prompts.
  """

  # Grounded literals appear inline as `-- matched_values: [...]` DDL notes.
  db_id: str
  selected_tables: tuple[str, ...]
  selected_columns: tuple[tuple[str, tuple[str, ...]], ...]
  fk_closure_edges: tuple[ForeignKeyEdge, ...]
  bridge_tables_added: tuple[str, ...]
  value_matches: tuple[ValueMatch, ...]
  ddl_prompt: str


# Candidate SQL query emitted by one prompting or heuristic synthesis strategy.
# Tracks strategy provenance so consensus voting rewards diverse agreement.
@dataclasses.dataclass(frozen=True)
class SQLCandidate:
  """Generated SQL query candidate paired with strategy and token metadata.

  Attributes:
    sql: Executable SQLite SELECT statement.
    strategy: Name of the generation strategy (e.g., 'decomposed_cot').
    reasoning: Brief chain-of-thought or synthesis justification.
    prompt_tokens: Input prompt token count (estimated or reported by API).
    completion_tokens: Output completion token count.
  """

  # Executable SQLite SELECT or WITH query candidate string.
  sql: str
  strategy: str
  reasoning: str = ""
  prompt_tokens: int = 0
  completion_tokens: int = 0


# Deterministic result of executing a candidate inside the read-only sandbox.
# `result_signature` hashes normalized cells for O(1) equivalence clustering.
@dataclasses.dataclass(frozen=True)
class ExecutionOutcome:
  """Result and timing captured from read-only SQLite candidate execution.

  Attributes:
    succeeded: True if SQLite parsed and executed the SELECT without error.
    rows: Tuple of result rows (capped to safe preview ceiling if massive).
    column_names: Projected result column headers.
    row_count: Total number of rows returned.
    execution_time_ms: Wall-clock SQLite execution duration in milliseconds.
    error_message: Sanitized SQLite diagnostic error message when failed.
    result_signature: Deterministic SHA-256 digest of normalized result rows.
  """

  # Execution status flag governing consensus weight eligibility.
  succeeded: bool
  rows: tuple[tuple[Any, ...], ...] = ()
  column_names: tuple[str, ...] = ()
  row_count: int = 0
  execution_time_ms: float = 0.0
  error_message: str = ""
  result_signature: str = ""


# Semantic or dialect invariant issue flagged by the Argus verifier.
# `repair_hint` provides deterministic rewrite guidance without extra discovery.
@dataclasses.dataclass(frozen=True)
class InvariantIssue:
  """Actionable semantic or SQLite dialect warning attached to a SQL candidate.

  Attributes:
    code: Machine-readable invariant code (e.g., 'INTEGER_DIVISION_RISK').
    severity: 'ERROR' (blocks consensus bonus) or 'WARNING' (penalizes score).
    message: Explanation of why the SQL or result violates expected semantics.
    repair_hint: Concrete SQL rewrite instruction fed to diagnostic repair.
  """

  code: str
  severity: str
  message: str
  repair_hint: str


# Candidate enriched with sandbox execution results and invariant diagnostics.
@dataclasses.dataclass(frozen=True)
class VerifiedCandidate:
  """SQL candidate audited by both execution sandboxing and semantic invariants.

  Attributes:
    candidate: Original or repaired SQLCandidate.
    outcome: Sandbox ExecutionOutcome.
    issues: Tuple of InvariantIssue items detected during verification.
    consensus_weight: Quality weight assigned prior to equivalence voting.
    was_repaired: True if produced by a diagnostic repair pass.
  """

  candidate: SQLCandidate
  outcome: ExecutionOutcome
  issues: tuple[InvariantIssue, ...] = ()
  consensus_weight: float = 1.0
  was_repaired: bool = False


# Final output emitted by any comparative NL2SQL pipeline for one benchmark task.
@dataclasses.dataclass(frozen=True)
class PipelinePrediction:
  """End-to-end prediction and telemetry emitted by an NL2SQL pipeline.

  Attributes:
    question_id: Benchmark question identifier.
    db_id: Target database identifier.
    system_name: Identifier of the pipeline configuration (e.g., 'Argus-SQL').
    final_sql: Winning SQL query selected for evaluation.
    latency_ms: End-to-end pipeline wall-clock latency in milliseconds.
    total_tokens: Sum of prompt and completion tokens consumed across passes.
    verified_candidates: All evaluated candidates for post-hoc failure analysis.
    metadata: Key-value diagnostic tags (e.g., bridge tables, repair count).
  """

  question_id: str
  db_id: str
  system_name: str
  final_sql: str
  latency_ms: float
  total_tokens: int
  verified_candidates: tuple[VerifiedCandidate, ...] = ()
  metadata: tuple[tuple[str, str], ...] = ()
