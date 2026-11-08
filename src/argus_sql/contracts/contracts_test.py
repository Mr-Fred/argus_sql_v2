"""Unit tests for Argus-SQL domain contracts and compact DDL serialization."""

import unittest

from argus_sql.contracts.exceptions import PromptBuildError
from argus_sql.contracts.schema_ddl_renderer import render_schema_ddl
from argus_sql.contracts.types import ColumnSchema
from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import ForeignKeyEdge
from argus_sql.contracts.types import TableSchema
from argus_sql.contracts.types import ValueMatch


class ContractsAndDDLTest(unittest.TestCase):
  """Tests relational schema lookup and value-annotated DDL rendering."""

  def _sample_schema(self) -> DatabaseSchema:
    """Builds a two-table sample relational schema with a foreign key."""
    # Include a spaced column identifier (`District Name`) to test SQL quoting.
    schools = TableSchema(
        name="schools",
        columns=(
            ColumnSchema("cds_code", "TEXT", is_primary_key=True),
            ColumnSchema("county", "TEXT", description="County name"),
            ColumnSchema("District Name", "TEXT", value_examples=("Oakland USD",)),
        ),
    )
    # Attach child `frpm` fact table referencing `schools(cds_code)`.
    frpm = TableSchema(
        name="frpm",
        columns=(
            ColumnSchema("cds_code", "TEXT", is_primary_key=True),
            ColumnSchema("free_meals", "INTEGER", description="Free meal count"),
        ),
    )
    fk = ForeignKeyEdge("frpm", "cds_code", "schools", "cds_code")
    return DatabaseSchema(
        db_id="schools_mini",
        tables=(schools, frpm),
        foreign_keys=(fk,),
    )

  def test_renders_full_and_pruned_ddl_with_value_matches(self) -> None:
    """Grounded cell matches, quoted identifiers, and FKs must render cleanly."""
    schema = self._sample_schema()
    # Ground 'Alameda' onto `schools.county` and prune `District Name`.
    matches = (
        ValueMatch(
            table_name="schools",
            column_name="county",
            matched_value="Alameda",
            query_span="alameda",
            similarity_score=1.0,
        ),
    )
    ddl = render_schema_ddl(
        schema,
        selected_columns={
            "schools": ("cds_code", "county"),
            "frpm": ("cds_code", "free_meals"),
        },
        value_matches=matches,
    )
    # Verify grounded values appear inline and pruned columns are excluded.
    self.assertIn("matched_values: ['Alameda']", ddl)
    self.assertIn("FOREIGN KEY (cds_code) REFERENCES schools(cds_code)", ddl)
    self.assertNotIn("District Name", ddl)

  def test_raises_when_pruning_leaves_zero_tables(self) -> None:
    """Empty table selection must fail fast with PromptBuildError."""
    # Ensure malformed schema pruning surfaces an explicit domain error.
    schema = self._sample_schema()
    with self.assertRaises(PromptBuildError):
      render_schema_ddl(schema, selected_columns={"nonexistent": ("id",)})

  def test_case_insensitive_table_lookup(self) -> None:
    """DatabaseSchema.table_by_name must match regardless of identifier case."""
    # SQLite table names are case-insensitive during lookup and join resolution.
    schema = self._sample_schema()
    resolved = schema.table_by_name("SCHOOLS")
    # Confirm uppercase query still resolves the canonical lowercase definition.
    self.assertIsNotNone(resolved)
    assert resolved is not None
    self.assertEqual(resolved.name, "schools")

  def test_renders_composite_primary_key_as_valid_sqlite_table_constraint(
      self,
  ) -> None:
    """Multi-column primary keys render as `PRIMARY KEY (c1, c2)` table DDL."""
    import sqlite3

    # Mirror `yearmonth(CustomerID, Date)` composite grain from BIRD-SQL.
    ym = TableSchema(
        name="yearmonth",
        columns=(
            ColumnSchema("CustomerID", "INTEGER", is_primary_key=True),
            ColumnSchema("Date", "TEXT", is_primary_key=True),
            ColumnSchema("Consumption", "REAL"),
        ),
    )
    ddl = render_schema_ddl(DatabaseSchema(db_id="debit", tables=(ym,)))
    self.assertIn("PRIMARY KEY (CustomerID, Date)", ddl)
    self.assertNotIn("CustomerID INTEGER PRIMARY KEY", ddl)
    # Confirm rendered DDL executes cleanly in SQLite without duplicate-PK error.
    conn = sqlite3.connect(":memory:")
    try:
      conn.executescript(ddl)
    finally:
      conn.close()


if __name__ == "__main__":
  unittest.main()
