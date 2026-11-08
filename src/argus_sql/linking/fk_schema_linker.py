"""Value-grounded Steiner Foreign-Key closure schema linker.

Combines lexical + evidence column scoring, 3-gram database cell-value matches,
and Breadth-First Shortest-Path Foreign-Key closure so pruned schemas never
drop intermediate bridge tables required for valid multi-hop SQL joins.
"""

from collections import deque
from collections.abc import Sequence
import re

from argus_sql.contracts.schema_ddl_renderer import render_schema_ddl
from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import ForeignKeyEdge
from argus_sql.contracts.types import IntentFrame
from argus_sql.contracts.types import LinkedSchemaContext
from argus_sql.contracts.types import ValueMatch
from argus_sql.linking.types import ColumnScore
from argus_sql.linking.types import SchemaLinkerConfig

# Stopwords excluded when computing lexical overlap with column identifiers.
_STOPWORDS: frozenset[str] = frozenset({
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "give",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "list",
    "many",
    "of",
    "on",
    "or",
    "show",
    "that",
    "the",
    "to",
    "what",
    "where",
    "which",
    "with",
})


def _tokenize_identifier(text: str) -> set[str]:
  """Splits snake_case, camelCase, and prose text into normalized root tokens."""
  # Split camelCase transitions before breaking on non-alphanumeric delimiters.
  spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
  raw_tokens = re.findall(r"[a-zA-Z0-9]+", spaced.lower())
  normalized: set[str] = set()
  for tok in raw_tokens:
    if tok in _STOPWORDS or len(tok) < 2:
      continue
    normalized.add(tok)
    # Add light singular stem ('schools' -> 'school') for plural invariance.
    if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
      normalized.add(tok[:-1])
  return normalized


def _overlap_score(query_tokens: set[str], target_tokens: set[str]) -> float:
  """Computes recall-oriented token overlap in `[0.0, 1.0]`."""
  if not query_tokens or not target_tokens:
    return 0.0
  shared = len(query_tokens & target_tokens)
  # Cap denominator at 4 so verbose column descriptions are not penalized.
  return min(1.0, shared / max(1, min(len(target_tokens), 4)))


def _shortest_fk_path(
    start_table: str,
    goal_table: str,
    adjacency: dict[str, list[tuple[str, ForeignKeyEdge]]],
) -> list[tuple[str, ForeignKeyEdge]]:
  """Finds shortest undirected FK join path between two tables via BFS."""
  # Unweighted BFS guarantees minimum-hop intermediate bridge table recovery.
  queue: deque[tuple[str, list[tuple[str, ForeignKeyEdge]]]] = deque(
      [(start_table, [])]
  )
  visited = {start_table}
  while queue:
    current, path = queue.popleft()
    if current == goal_table:
      return path
    # Traverse undirected FK edges so both parent->child and child->parent work.
    for neighbor, edge in adjacency.get(current, ()):
      if neighbor not in visited:
        visited.add(neighbor)
        queue.append((neighbor, [*path, (neighbor, edge)]))
  return []


def link_schema_with_fk_closure(
    schema: DatabaseSchema,
    intent: IntentFrame,
    evidence: str = "",
    value_matches: Sequence[ValueMatch] = (),
    config: SchemaLinkerConfig | None = None,
) -> LinkedSchemaContext:
  """Links relevant tables/columns and guarantees FK join-path completeness.

  Args:
    schema: Full relational DatabaseSchema.
    intent: Canonicalized IntentFrame from `canonicalize_intent`.
    evidence: Optional external domain formula / note string.
    value_matches: Grounded database cell values from `CellValueRetriever`.
    config: Optional SchemaLinkerConfig controlling pruning and ablations.

  Returns:
    LinkedSchemaContext containing selected tables, bridge tables, and DDL.
  """
  cfg = config or SchemaLinkerConfig()
  # Cap grounded cell literals to `cfg.top_k_values` when value grounding is on.
  active_values = (
      tuple(value_matches[: cfg.top_k_values])
      if cfg.enable_value_grounding
      else ()
  )

  q_tokens = _tokenize_identifier(intent.canonical_question)
  ev_tokens = _tokenize_identifier(evidence)

  # Map `(lower_table, lower_col) -> best ValueMatch similarity`.
  val_score_map: dict[tuple[str, str], float] = {}
  for vm in active_values:
    key = (vm.table_name.lower(), vm.column_name.lower())
    val_score_map[key] = max(val_score_map.get(key, 0.0), vm.similarity_score)

  # Score every column across all tables using lexical, evidence, and value hits.
  scores_by_table: dict[str, list[ColumnScore]] = {}
  seed_tables: list[str] = []

  for table in schema.tables:
    t_lower = table.name.lower()
    t_tokens = _tokenize_identifier(table.name)
    table_hit = _overlap_score(q_tokens | ev_tokens, t_tokens)
    col_scores: list[ColumnScore] = []

    for col in table.columns:
      c_tokens = _tokenize_identifier(f"{col.name} {col.description}")
      raw_lex = _overlap_score(q_tokens, c_tokens)
      # Discount pure surrogate/FK `*_id` columns on unmentioned tables so star
      # schemas do not pull in every child table; true bridges enter via BFS!
      is_key_col = col.is_primary_key or col.name.lower().endswith("_id")
      if is_key_col and table_hit == 0.0 and not col.description:
        raw_lex *= 0.25
      lex = max(raw_lex, 0.35 * table_hit)
      ev = _overlap_score(ev_tokens, c_tokens)
      val = val_score_map.get((t_lower, col.name.lower()), 0.0)
      cs = ColumnScore(table.name, col.name, lex, ev, val)
      col_scores.append(cs)

    scores_by_table[t_lower] = col_scores
    best_col = max((c.total_score for c in col_scores), default=0.0)
    if best_col >= cfg.min_column_score or table_hit >= 0.5:
      seed_tables.append(t_lower)

  # Fallback: if zero tables exceeded threshold, retain top-scoring table.
  if not seed_tables and schema.tables:
    ranked_tables = sorted(
        schema.tables,
        key=lambda t: max(
            (c.total_score for c in scores_by_table[t.name.lower()]),
            default=0.0,
        ),
        reverse=True,
    )
    seed_tables.append(ranked_tables[0].name.lower())

  # Prune isolated orphan distractor tables (0 FKs) when an FK-connected seed
  # table already covers the same matched non-key column names.
  if cfg.enable_fk_closure and len(seed_tables) >= 2 and schema.foreign_keys:
    fk_participating = {
        e.source_table.lower() for e in schema.foreign_keys
    } | {e.target_table.lower() for e in schema.foreign_keys}
    connected_cols: set[str] = set()
    for t_low in seed_tables:
      if t_low in fk_participating:
        connected_cols.update(
            cs.column_name.lower()
            for cs in scores_by_table[t_low]
            if cs.total_score >= cfg.min_column_score
        )
    seed_tables = [
        t_low
        for t_low in seed_tables
        if t_low in fk_participating
        or not {
            cs.column_name.lower()
            for cs in scores_by_table[t_low]
            if cs.total_score >= cfg.min_column_score
        }.issubset(connected_cols)
    ]

  # Recover intermediate bridge tables via Steiner-like pairwise BFS FK closure.
  selected_table_set = set(seed_tables)
  bridge_tables: list[str] = []
  fk_columns_needed: dict[str, set[str]] = {t: set() for t in selected_table_set}

  if cfg.enable_fk_closure and len(seed_tables) >= 2 and schema.foreign_keys:
    # Build an undirected FK adjacency list so joins traverse both directions.
    adjacency: dict[str, list[tuple[str, ForeignKeyEdge]]] = {}
    for edge in schema.foreign_keys:
      src = edge.source_table.lower()
      dst = edge.target_table.lower()
      adjacency.setdefault(src, []).append((dst, edge))
      adjacency.setdefault(dst, []).append((src, edge))

    # Connect every seed table to the anchor table along shortest FK paths.
    anchor = seed_tables[0]
    for other in seed_tables[1:]:
      path = _shortest_fk_path(anchor, other, adjacency)
      for hop_table, edge in path:
        if hop_table not in selected_table_set:
          selected_table_set.add(hop_table)
          bridge_tables.append(hop_table)
        # Preserve foreign-key join columns even if lexically unmentioned.
        fk_columns_needed.setdefault(edge.source_table.lower(), set()).add(
            edge.source_column.lower()
        )
        fk_columns_needed.setdefault(edge.target_table.lower(), set()).add(
            edge.target_column.lower()
        )

  # Always retain foreign-key endpoints connecting already-selected tables.
  retained_edges: list[ForeignKeyEdge] = []
  for edge in schema.foreign_keys:
    src = edge.source_table.lower()
    dst = edge.target_table.lower()
    if src in selected_table_set and dst in selected_table_set:
      retained_edges.append(edge)
      fk_columns_needed.setdefault(src, set()).add(edge.source_column.lower())
      fk_columns_needed.setdefault(dst, set()).add(edge.target_column.lower())

  # Select top-scoring columns + mandatory PK/FK columns per retained table.
  selected_cols_map: dict[str, tuple[str, ...]] = {}
  ordered_selected_tables: list[str] = []
  for table in schema.tables:
    t_lower = table.name.lower()
    if t_lower not in selected_table_set:
      continue
    ordered_selected_tables.append(table.name)

    # Sort table columns by composite lexical + evidence + cell-value score.
    ranked_cols = sorted(
        scores_by_table[t_lower],
        key=lambda c: c.total_score,
        reverse=True,
    )
    top_names = {
        cs.column_name.lower()
        for cs in ranked_cols[: cfg.max_columns_per_table]
        if cs.total_score >= cfg.min_column_score
    }
    # Ensure every table keeps at least its top-2 columns alongside PK/FK keys.
    if not top_names:
      top_names = {cs.column_name.lower() for cs in ranked_cols[:2]}

    mandatory_keys = fk_columns_needed.get(t_lower, set()) | {
        col.name.lower() for col in table.columns if col.is_primary_key
    }
    keep_set = top_names | mandatory_keys
    selected_cols_map[table.name] = tuple(
        col.name for col in table.columns if col.name.lower() in keep_set
    )

  ddl_text = render_schema_ddl(
      schema=schema,
      selected_columns=selected_cols_map,
      fk_edges=retained_edges,
      value_matches=active_values,
  )

  return LinkedSchemaContext(
      db_id=schema.db_id,
      selected_tables=tuple(ordered_selected_tables),
      selected_columns=tuple(selected_cols_map.items()),
      fk_closure_edges=tuple(retained_edges),
      bridge_tables_added=tuple(bridge_tables),
      value_matches=active_values,
      ddl_prompt=ddl_text,
  )
