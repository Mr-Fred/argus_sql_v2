"""Value-grounded Steiner Foreign-Key closure schema linker.

Combines compound-aware morphological column scoring, 3-gram database cell-value
matches, implicit PK-column foreign-key inference, and all-pairs Steiner +
junction bridge closure so schemas retain join paths and measure columns cleanly.
"""

from collections.abc import Sequence

from argus_sql.contracts.schema_ddl_renderer import render_schema_ddl
from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import ForeignKeyEdge
from argus_sql.contracts.types import IntentFrame
from argus_sql.contracts.types import LinkedSchemaContext
from argus_sql.contracts.types import ValueMatch
from argus_sql.linking.schema_graph import build_compound_vocabulary
from argus_sql.linking.schema_graph import expand_fk_subgraph
from argus_sql.linking.schema_graph import resolve_effective_foreign_keys
from argus_sql.linking.schema_graph import tokenize_identifier
from argus_sql.linking.types import ColumnScore
from argus_sql.linking.types import SchemaLinkerConfig


def _overlap_score(query_tokens: set[str], target_tokens: set[str]) -> float:
  """Computes recall-oriented token overlap in `[0.0, 1.0]`."""
  if not query_tokens or not target_tokens:
    return 0.0
  shared = len(query_tokens & target_tokens)
  # Cap denominator at 4 so rich codebook column descriptions are not penalized.
  return min(1.0, shared / max(1, min(len(target_tokens), 4)))


def _wide_table_column_priority(col_name: str, data_type: str) -> float:
  """Provides a tiny structural tie-break on wide (>32 col) tables."""
  low = col_name.lower()
  # Prioritize status/type/category/location/date/name columns over sparse codes
  # so wide tables (e.g., 115-col sports tables) retain descriptive dimensions.
  priority_keywords = (
      "type", "status", "name", "city", "county", "state", "country", "date",
      "year", "grade", "score", "rate", "charter", "email", "phone", "street",
      "goal", "win", "points", "rank", "position",
  )
  if any(kw in low for kw in priority_keywords):
    return 0.02
  # Prefer TEXT dimensions slightly over anonymous numeric betting/stat slots.
  if "TEXT" in data_type.upper():
    return 0.01
  return 0.0


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

  # Build dynamic compound sub-word vocabulary from question and evidence text.
  vocab = build_compound_vocabulary((intent.canonical_question, evidence))
  q_tokens = tokenize_identifier(intent.canonical_question, vocab)
  ev_tokens = tokenize_identifier(evidence, vocab)

  # Map `(lower_table, lower_col) -> best ValueMatch similarity`.
  val_score_map: dict[tuple[str, str], float] = {}
  for vm in active_values:
    key = (vm.table_name.lower(), vm.column_name.lower())
    val_score_map[key] = max(val_score_map.get(key, 0.0), vm.similarity_score)

  # Resolve explicit FKs plus inferred entity-PK matches when closure is active.
  effective_fks = resolve_effective_foreign_keys(
      schema, enable_implicit=cfg.enable_fk_closure
  )

  # Score every column across all tables using lexical, evidence, and value hits.
  scores_by_table: dict[str, list[ColumnScore]] = {}
  seed_tables: list[str] = []

  for table in schema.tables:
    t_lower = table.name.lower()
    t_tokens = tokenize_identifier(table.name, vocab)
    combined_q_ev = q_tokens | ev_tokens
    table_hit = (
        1.0 if t_lower in combined_q_ev else _overlap_score(combined_q_ev, t_tokens)
    )
    col_scores: list[ColumnScore] = []

    for col in table.columns:
      c_tokens = tokenize_identifier(f"{col.name} {col.description}", vocab)
      raw_lex = _overlap_score(q_tokens, c_tokens)
      # Discount pure surrogate/FK `*_id` columns on unmentioned tables so star
      # schemas do not pull in every child table; true bridges enter via BFS!
      c_low = col.name.lower()
      is_key_col = col.is_primary_key or c_low.endswith(("_id", "id"))
      if is_key_col and table_hit == 0.0 and not col.description:
        raw_lex *= 0.25
      lex = max(raw_lex, 0.35 * table_hit)
      ev = _overlap_score(ev_tokens, c_tokens)
      val = val_score_map.get((t_lower, c_low), 0.0)
      col_scores.append(ColumnScore(table.name, col.name, lex, ev, val))

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
  if cfg.enable_fk_closure and len(seed_tables) >= 2 and effective_fks:
    fk_participating = {
        e.source_table.lower() for e in effective_fks
    } | {e.target_table.lower() for e in effective_fks}
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

  # Recover Steiner bridges, M:N junction tables, and compact lookup dimensions.
  if cfg.enable_fk_closure and effective_fks:
    selected_table_set, bridge_tables, fk_columns_needed = expand_fk_subgraph(
        schema=schema,
        seed_tables=seed_tables,
        fk_edges=effective_fks,
    )
  else:
    selected_table_set = set(seed_tables)
    bridge_tables = []
    fk_columns_needed = {t: set() for t in selected_table_set}

  # Always retain foreign-key endpoints connecting already-selected tables.
  retained_edges: list[ForeignKeyEdge] = []
  for edge in effective_fks:
    src = edge.source_table.lower()
    dst = edge.target_table.lower()
    if src in selected_table_set and dst in selected_table_set:
      retained_edges.append(edge)
      fk_columns_needed.setdefault(src, set()).add(edge.source_column.lower())
      fk_columns_needed.setdefault(dst, set()).add(edge.target_column.lower())

  # Retain 100% of columns on compact/medium tables (`<= full_table_column_ceiling`)
  # so unmentioned measure or status columns are never dropped; prune ultra-wide
  # tables (`> 32` cols) to top-scoring + mandatory PK/FK columns.
  selected_cols_map: dict[str, tuple[str, ...]] = {}
  ordered_selected_tables: list[str] = []
  for table in schema.tables:
    t_lower = table.name.lower()
    if t_lower not in selected_table_set:
      continue
    ordered_selected_tables.append(table.name)

    if len(table.columns) <= cfg.full_table_column_ceiling:
      selected_cols_map[table.name] = tuple(col.name for col in table.columns)
      continue

    dtype_by_low = {c.name.lower(): c.data_type for c in table.columns}
    ranked_cols = sorted(
        scores_by_table[t_lower],
        key=lambda c: (
            c.total_score
            + _wide_table_column_priority(
                c.column_name, dtype_by_low.get(c.column_name.lower(), "")
            )
        ),
        reverse=True,
    )
    # Keep all positive hits plus top descriptive columns up to `max_columns_per_table`.
    positive_hits = {
        cs.column_name.lower() for cs in ranked_cols if cs.total_score > 0.0
    }
    budget_fill = {
        cs.column_name.lower() for cs in ranked_cols[: cfg.max_columns_per_table]
    }
    mandatory_keys = fk_columns_needed.get(t_lower, set()) | {
        col.name.lower() for col in table.columns if col.is_primary_key
    }
    keep_set = positive_hits | budget_fill | mandatory_keys
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
