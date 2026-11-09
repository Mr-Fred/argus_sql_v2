"""Compact DDL prompt renderer for relational schemas and grounded cell values.

Implements DAIL-SQL and CHESS inspired `CREATE TABLE` DDL serialization that
embeds column descriptions, sample values, grounded database literals, and
explicit `FOREIGN KEY` clauses while keeping prompt token counts compact.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import re

from argus_sql.contracts.exceptions import PromptBuildError
from argus_sql.contracts.types import ColumnSchema
from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import ForeignKeyEdge
from argus_sql.contracts.types import ValueMatch

# Matches numeric literal bounds rendered unquoted on numeric-affinity columns.
_NUMERIC_BOUND_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:e[+-]?\d+)?$", re.IGNORECASE)

# SQLite column type prefixes that store unquoted numeric values.
_NUMERIC_TYPE_PREFIXES = ("INT", "REAL", "NUM", "FLOAT", "DOUBLE", "DEC")


# SQLite reserved keywords that must be double-quoted when used as identifiers
# (e.g., the `order` table in the public BIRD `financial` database).
_SQLITE_RESERVED_IDENTIFIERS: frozenset[str] = frozenset({
    "add", "all", "alter", "and", "as", "between", "by", "case", "check",
    "collate", "commit", "constraint", "create", "cross", "default", "delete",
    "desc", "distinct", "drop", "else", "escape", "except", "exists",
    "foreign", "from", "full", "group", "having", "in", "index", "inner",
    "insert", "intersect", "into", "is", "isnull", "join", "left", "limit",
    "natural", "not", "notnull", "null", "on", "or", "order", "outer",
    "primary", "references", "right", "select", "set", "table", "then", "to",
    "transaction", "union", "unique", "update", "using", "values", "when",
    "where", "with",
})


def _quote_sql_identifier(identifier: str) -> str:
  """Quotes SQLite identifiers containing spaces, punctuation, or keywords."""
  cleaned = identifier.replace('"', '""')
  # Only leave unquoted when identifier is alphanumeric and not a SQL keyword.
  if cleaned.isidentifier() and cleaned.lower() not in _SQLITE_RESERVED_IDENTIFIERS:
    return cleaned
  return f'"{cleaned}"'


def _format_range_note(col: ColumnSchema) -> str:
  """Formats `col.value_range` as unquoted numeric or quoted temporal bounds."""
  assert col.value_range is not None
  lo, hi = col.value_range
  dtype_upper = col.data_type.upper()
  # Emit unquoted bounds on numeric columns so SQLite literal types stay clear.
  if dtype_upper.startswith(_NUMERIC_TYPE_PREFIXES) and (
      _NUMERIC_BOUND_RE.fullmatch(lo) and _NUMERIC_BOUND_RE.fullmatch(hi)
  ):
    return f"range: {lo}..{hi}"
  # Quote temporal/text domain bounds ('2012-08-23'..'2012-08-26').
  return f"range: {lo!r}..{hi!r}"


def _format_column_entry(
    col: ColumnSchema,
    grounded_literals: Sequence[str],
    inline_pk: bool = True,
) -> tuple[str, str]:
  """Builds `(column_decl, inline_comment)` for a single DDL column line."""
  pk_suffix = " PRIMARY KEY" if (col.is_primary_key and inline_pk) else ""
  col_decl = f"  {_quote_sql_identifier(col.name)} {col.data_type}{pk_suffix}"

  # Combine description, grounded matches, domain range, or fallback examples.
  notes: list[str] = []
  if col.description:
    notes.append(col.description.strip())
  if grounded_literals:
    joined_hits = ", ".join(repr(val) for val in grounded_literals[:3])
    notes.append(f"matched_values: [{joined_hits}]")
  if col.value_range is not None:
    notes.append(_format_range_note(col))
  elif col.value_examples and not grounded_literals:
    # Omit redundant `examples:` when `range:` or `matched_values:` is present.
    joined_ex = ", ".join(repr(val) for val in col.value_examples[:2])
    notes.append(f"examples: [{joined_ex}]")

  return (col_decl, " | ".join(notes))


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
    table_entries: list[tuple[str, str]] = []
    for col in table.columns:
      if (
          allowed_cols is not None
          and col.name.lower() not in allowed_cols
          and not col.is_primary_key
      ):
        continue
      hits = grounded_index.get((tbl_lower, col.name.lower()), ())
      table_entries.append(
          _format_column_entry(col, hits, inline_pk=not is_composite_pk)
      )

    # Append composite PRIMARY KEY (...) and explicit FOREIGN KEY clauses.
    if is_composite_pk:
      pk_list = ", ".join(_quote_sql_identifier(c.name) for c in pk_cols)
      table_entries.append((f"  PRIMARY KEY ({pk_list})", ""))

    # Only emit FK edges whose target table also survived schema pruning.
    for edge in active_edges:
      if edge.source_table.lower() == tbl_lower:
        if allow_map is None or edge.target_table.lower() in allow_map:
          fk_decl = (
              f"  FOREIGN KEY ({_quote_sql_identifier(edge.source_column)}) "
              f"REFERENCES {_quote_sql_identifier(edge.target_table)}"
              f"({_quote_sql_identifier(edge.target_column)})"
          )
          table_entries.append((fk_decl, ""))

    # Place commas before `--` comments so rendered DDL is valid SQLite syntax.
    formatted_lines: list[str] = []
    last_idx = len(table_entries) - 1
    for idx, (decl, note) in enumerate(table_entries):
      comma = "," if idx < last_idx else ""
      formatted_lines.append(f"{decl}{comma} -- {note}" if note else f"{decl}{comma}")

    # Assemble single-table CREATE TABLE block in canonical schema order.
    body = "\n".join(formatted_lines)
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
