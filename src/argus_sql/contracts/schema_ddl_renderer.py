"""Compact DDL prompt renderer for relational schemas and grounded cell values.

Implements DAIL-SQL and CHESS inspired `CREATE TABLE` DDL serialization that
embeds column descriptions, sample values, grounded database literals, and
explicit `FOREIGN KEY` clauses while keeping prompt token counts compact.
"""

from collections.abc import Mapping
from collections.abc import Sequence

from argus_sql.contracts.exceptions import PromptBuildError
from argus_sql.contracts.types import ColumnSchema
from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import ForeignKeyEdge
from argus_sql.contracts.types import ValueMatch


def _quote_sql_identifier(identifier: str) -> str:
  """Quotes SQLite identifiers containing spaces or punctuation safely."""
  cleaned = identifier.replace('"', '""')
  # Only wrap in double quotes if non-alphanumeric/underscore chars exist.
  if cleaned.isidentifier():
    return cleaned
  return f'"{cleaned}"'


def _format_column_line(
    col: ColumnSchema,
    grounded_literals: Sequence[str],
    inline_pk: bool = True,
) -> str:
  """Formats a single column declaration with inline comment annotations."""
  pk_suffix = " PRIMARY KEY" if (col.is_primary_key and inline_pk) else ""
  col_decl = f"  {_quote_sql_identifier(col.name)} {col.data_type}{pk_suffix}"

  # Combine column description, grounded query matches, and sample literals.
  notes: list[str] = []
  if col.description:
    notes.append(col.description.strip())
  if grounded_literals:
    joined_hits = ", ".join(repr(val) for val in grounded_literals[:3])
    notes.append(f"matched_values: [{joined_hits}]")
  elif col.value_examples:
    joined_ex = ", ".join(repr(val) for val in col.value_examples[:2])
    notes.append(f"examples: [{joined_ex}]")

  if not notes:
    return col_decl
  return f"{col_decl} -- {' | '.join(notes)}"


def render_schema_ddl(
    schema: DatabaseSchema,
    selected_columns: Mapping[str, Sequence[str]] | None = None,
    fk_edges: Sequence[ForeignKeyEdge] | None = None,
    value_matches: Sequence[ValueMatch] = (),
) -> str:
  """Serializes a full or pruned DatabaseSchema into compact SQLite DDL text.

  Args:
    schema: Source DatabaseSchema definition.
    selected_columns: Optional mapping of `table_name -> column_names` to prune
      irrelevant columns; renders all tables/columns when None.
    fk_edges: Optional subset of ForeignKeyEdge items to render; defaults to
      all edges whose endpoints belong to rendered tables.
    value_matches: Grounded cell literals attached to matching column comments.

  Returns:
    Deterministic multi-table `CREATE TABLE` DDL string for prompt injection.

  Raises:
    PromptBuildError: If `schema.tables` is empty or selection leaves 0 tables.
  """
  if not schema.tables:
    raise PromptBuildError(f"DatabaseSchema '{schema.db_id}' has no tables.")

  # Index grounded value literals by `(lower_table, lower_col)` in one pass.
  grounded_index: dict[tuple[str, str], list[str]] = {}
  for match in value_matches:
    key = (match.table_name.lower(), match.column_name.lower())
    bucket = grounded_index.setdefault(key, [])
    if match.matched_value not in bucket:
      bucket.append(match.matched_value)

  # Normalize optional table/column allow-list for case-insensitive lookup.
  allow_map: dict[str, set[str]] | None = None
  if selected_columns is not None:
    allow_map = {
        tbl.lower(): {c.lower() for c in cols}
        for tbl, cols in selected_columns.items()
    }

  active_edges = fk_edges if fk_edges is not None else schema.foreign_keys
  rendered_blocks: list[str] = []
  included_tables: set[str] = set()

  for table in schema.tables:
    tbl_lower = table.name.lower()
    if allow_map is not None and tbl_lower not in allow_map:
      continue
    included_tables.add(tbl_lower)

    # Emit composite multi-column PKs as a table constraint so SQLite DDL stays
    # syntactically valid and makes composite row grain explicit to the LLM.
    pk_cols = [col for col in table.columns if col.is_primary_key]
    is_composite_pk = len(pk_cols) >= 2

    # Filter columns according to `selected_columns` while preserving PKs.
    allowed_cols = allow_map[tbl_lower] if allow_map is not None else None
    col_lines: list[str] = []
    for col in table.columns:
      if (
          allowed_cols is not None
          and col.name.lower() not in allowed_cols
          and not col.is_primary_key
      ):
        continue
      hits = grounded_index.get((tbl_lower, col.name.lower()), ())
      col_lines.append(
          _format_column_line(col, hits, inline_pk=not is_composite_pk)
      )

    # Append composite PRIMARY KEY (...) and explicit FOREIGN KEY clauses.
    constraint_lines: list[str] = []
    if is_composite_pk:
      pk_list = ", ".join(_quote_sql_identifier(c.name) for c in pk_cols)
      constraint_lines.append(f"  PRIMARY KEY ({pk_list})")

    # Only emit FK edges whose target table also survived schema pruning.
    for edge in active_edges:
      if edge.source_table.lower() == tbl_lower:
        if allow_map is None or edge.target_table.lower() in allow_map:
          constraint_lines.append(
              f"  FOREIGN KEY ({_quote_sql_identifier(edge.source_column)}) "
              f"REFERENCES {_quote_sql_identifier(edge.target_table)}"
              f"({_quote_sql_identifier(edge.target_column)})"
          )

    # Assemble single-table CREATE TABLE block in canonical schema order.
    body = ",\n".join(col_lines + constraint_lines)
    rendered_blocks.append(
        f"CREATE TABLE {_quote_sql_identifier(table.name)} (\n{body}\n);"
    )

  # Fail fast rather than sending an empty schema prompt to an LLM adapter.
  if not rendered_blocks or not included_tables:
    raise PromptBuildError(
        f"Schema pruning left zero tables for database '{schema.db_id}'."
    )

  # Separate CREATE TABLE statements with blank lines for token clarity.
  return "\n\n".join(rendered_blocks)
