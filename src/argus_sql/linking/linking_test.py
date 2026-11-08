"""Unit tests for voice intent canonicalization, value indexing, and FK closure."""

import unittest

from argus_sql.contracts.types import ColumnSchema
from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import ForeignKeyEdge
from argus_sql.contracts.types import QueryDifficulty
from argus_sql.contracts.types import TableSchema
from argus_sql.linking.fk_schema_linker import link_schema_with_fk_closure
from argus_sql.linking.intent_canonicalizer import canonicalize_intent
from argus_sql.linking.schema_graph import tokenize_identifier
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

  def test_iteration2_compound_stems_implicit_fk_and_junction_closure(
      self,
  ) -> None:
    """Compound names, short voice codes, implicit FKs, and M:N junctions link."""
    # 1. Multi-pair comparison & dual-extrema arity estimation.
    pair_intent = canonicalize_intent(
        "What is the difference in consumption paid in czk between sme and lam, "
        "lam and kam, and kam and sme?"
    )
    self.assertEqual(pair_intent.expected_projection_arity, 3)
    dual_intent = canonicalize_intent(
        "Which segment has the biggest and lowest percentage increases?"
    )
    self.assertEqual(dual_intent.expected_projection_arity, 2)

    # 2. Implicit PK-FK discovery + compound table split (`gasstations`) +
    #    short lowercase ASR code grounding (`'sme'` -> `'SME'`, `'cze'` -> `'CZE'`).
    customers = TableSchema(
        "customers",
        (
            ColumnSchema("CustomerID", "INTEGER", is_primary_key=True),
            ColumnSchema("Segment", "TEXT", value_examples=("SME", "LAM")),
            ColumnSchema("Currency", "TEXT", value_examples=("EUR", "CZK")),
        ),
    )
    gasstations = TableSchema(
        "gasstations",
        (
            ColumnSchema("GasStationID", "INTEGER", is_primary_key=True),
            ColumnSchema("Country", "TEXT", value_examples=("CZE", "SVK")),
        ),
    )
    products = TableSchema(
        "products",
        (
            ColumnSchema("ProductID", "INTEGER", is_primary_key=True),
            ColumnSchema("Description", "TEXT"),
        ),
    )
    # Undeclared FKs on `transactions_1k` mirror BIRD `debit_card_specializing`.
    tx = TableSchema(
        "transactions_1k",
        (
            ColumnSchema("TransactionID", "INTEGER", is_primary_key=True),
            ColumnSchema("CustomerID", "INTEGER"),
            ColumnSchema("GasStationID", "INTEGER"),
            ColumnSchema("ProductID", "INTEGER"),
            ColumnSchema("Date", "DATE"),
        ),
    )
    star_schema = DatabaseSchema(
        "debit_star", (customers, gasstations, products, tx), foreign_keys=()
    )
    retriever = CellValueRetriever.from_schema_and_sqlite(star_schema)
    voice_intent = canonicalize_intent(
        "uh please list the countries of the gas stations for sme in cze"
    )
    v_matches = retriever.retrieve(voice_intent)
    self.assertTrue({"SME", "CZE"}.issubset({m.matched_value for m in v_matches}))

    # Querying `products` + `gasstations` must recover `transactions_1k` as an
    # M:N junction bridge via inferred `ProductID` and `GasStationID` FKs!
    prod_intent = canonicalize_intent(
        "Please list the product descriptions sold in gas stations in CZE."
    )
    linked = link_schema_with_fk_closure(
        star_schema, prod_intent, value_matches=retriever.retrieve(prod_intent)
    )
    self.assertIn("gasstations", linked.selected_tables)
    self.assertIn("products", linked.selected_tables)
    self.assertIn("transactions_1k", linked.selected_tables)

    # 3. Mixed-case acronyms ('IgG'), '-ed' verbs ('tagged'), and narrative arity.
    self.assertIn("igg", tokenize_identifier("normal IgG measurement"))
    self.assertIn("tag", tokenize_identifier("questions tagged r"))
    narrative = canonicalize_intent(
        "I am reviewing schools across California. First filter active schools. "
        "Then report name, city, county, zip, enrollment, and math score."
    )
    self.assertEqual(narrative.expected_projection_arity, 0)


if __name__ == "__main__":
  unittest.main()
