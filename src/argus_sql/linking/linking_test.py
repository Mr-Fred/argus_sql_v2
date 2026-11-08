"""Unit tests for voice intent canonicalization, value indexing, and FK closure."""

import unittest

from argus_sql.contracts.types import ColumnSchema
from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import ForeignKeyEdge
from argus_sql.contracts.types import QueryDifficulty
from argus_sql.contracts.types import TableSchema
from argus_sql.linking.fk_schema_linker import link_schema_with_fk_closure
from argus_sql.linking.intent_canonicalizer import canonicalize_intent
from argus_sql.linking.types import SchemaLinkerConfig
from argus_sql.linking.value_retriever import CellValueRetriever


class LinkingModuleTest(unittest.TestCase):
  """Tests ASR disfluency cleanup, 3-gram value grounding, and FK bridge recovery."""

  def _three_table_bridge_schema(self) -> DatabaseSchema:
    """Creates `clients -> accounts -> loans` where `accounts` is a bridge."""
    # `clients` holds geographic literals ('California') queried by the user.
    clients = TableSchema(
        name="clients",
        columns=(
            ColumnSchema("client_id", "INTEGER", is_primary_key=True),
            ColumnSchema(
                "region",
                "TEXT",
                description="Geographic state",
                value_examples=("California", "New York"),
            ),
        ),
    )
    # `accounts` has zero lexical overlap with 'clients' or 'loans' questions.
    accounts = TableSchema(
        name="accounts",
        columns=(
            ColumnSchema("account_id", "INTEGER", is_primary_key=True),
            ColumnSchema("client_id", "INTEGER"),
            ColumnSchema("branch_code", "TEXT"),
        ),
    )
    # `loans` holds the target `loan_amount` aggregation measure.
    loans = TableSchema(
        name="loans",
        columns=(
            ColumnSchema("loan_id", "INTEGER", is_primary_key=True),
            ColumnSchema("account_id", "INTEGER"),
            ColumnSchema("loan_amount", "REAL", description="Principal USD"),
            ColumnSchema("status", "TEXT", value_examples=("Defaulted", "Paid")),
        ),
    )
    # Two-hop FK chain requires `accounts` for any valid `clients <-> loans` join.
    fks = (
        ForeignKeyEdge("accounts", "client_id", "clients", "client_id"),
        ForeignKeyEdge("loans", "account_id", "accounts", "account_id"),
    )
    return DatabaseSchema("finance_bridge", (clients, accounts, loans), fks)

  def test_voice_disfluencies_and_spoken_numbers_are_canonicalized(self) -> None:
    """Spoken fillers, stutters, and number words must normalize cleanly."""
    spoken = (
        "uh hey can you please tell me what is the the percentage of "
        "defaulted loans in Calif for the top five clients?"
    )
    intent = canonicalize_intent(spoken)
    # Disfluencies and 'the the' stutter should be stripped.
    self.assertGreaterEqual(intent.disfluencies_removed, 2)
    self.assertNotIn("the the", intent.canonical_question.lower())
    self.assertIn("top 5", intent.canonical_question.lower())
    self.assertTrue(intent.requires_floating_division)
    self.assertTrue(intent.requires_extrema_order)
    self.assertEqual(intent.complexity, QueryDifficulty.CHALLENGING)
    self.assertIn("California", intent.entities)

  def test_value_retriever_and_fk_bridge_closure(self) -> None:
    """Querying clients + loans must ground 'California' and recover 'accounts'."""
    # Build 3-table schema where `accounts` bridges `clients` and `loans`.
    schema = self._three_table_bridge_schema()
    retriever = CellValueRetriever.from_schema_and_sqlite(schema)
    intent = canonicalize_intent(
        "uh show me the total loan amount for clients in Calif."
    )
    # Confirm 'Calif.' maps onto the physical cell literal 'California'.
    matches = retriever.retrieve(intent)
    matched_vals = {m.matched_value for m in matches}
    self.assertIn("California", matched_vals)

    # With FK closure enabled, intermediate bridge table `accounts` is recovered!
    linked_with_closure = link_schema_with_fk_closure(
        schema=schema,
        intent=intent,
        value_matches=matches,
        config=SchemaLinkerConfig(enable_fk_closure=True),
    )
    self.assertIn("accounts", linked_with_closure.selected_tables)
    self.assertIn("accounts", linked_with_closure.bridge_tables_added)

    # Without FK closure (ablation), `accounts` is dropped, breaking the join!
    linked_no_closure = link_schema_with_fk_closure(
        schema=schema,
        intent=intent,
        value_matches=matches,
        config=SchemaLinkerConfig(enable_fk_closure=False),
    )
    self.assertNotIn("accounts", linked_no_closure.selected_tables)


if __name__ == "__main__":
  unittest.main()
