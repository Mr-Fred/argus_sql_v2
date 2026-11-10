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

  def test_iteration3_localization_stems_and_medium_table_column_retention(
      self,
  ) -> None:
    """Translation phrasing seeds foreign_data and <=50-col tables keep 100% cols."""
    # 1. `translation` stems to `foreign` + `language`, seeding `foreign_data`.
    cards = TableSchema(
        "cards",
        (
            ColumnSchema("id", "INTEGER", is_primary_key=True),
            ColumnSchema("uuid", "TEXT"),
            ColumnSchema("isStorySpotlight", "INTEGER"),
        ),
    )
    foreign_data = TableSchema(
        "foreign_data",
        (
            ColumnSchema("id", "INTEGER", is_primary_key=True),
            ColumnSchema("language", "TEXT"),
            ColumnSchema("uuid", "TEXT"),
        ),
    )
    sets = TableSchema(
        "sets",
        tuple(
            [ColumnSchema("id", "INTEGER", is_primary_key=True), ColumnSchema("code", "TEXT")]
            + [ColumnSchema(f"attr_{i}", "TEXT") for i in range(18)]
        ),
    )
    cg_schema = DatabaseSchema(
        "card_games",
        (cards, foreign_data, sets),
        (ForeignKeyEdge("foreign_data", "uuid", "cards", "uuid"),),
    )
    tr_intent = canonicalize_intent(
        "Among Story Spotlight card printings, what percentage have a French translation?"
    )
    linked_cg = link_schema_with_fk_closure(cg_schema, tr_intent)
    self.assertIn("foreign_data", linked_cg.selected_tables)

    # 2. 44-column clinical laboratory table retains 100% of columns (`<= 50`).
    lab_cols = (ColumnSchema("ID", "INTEGER", is_primary_key=True),) + tuple(
        ColumnSchema(f"LAB_{idx}", "REAL") for idx in range(43)
    )
    lab_schema = DatabaseSchema("clinical_wide", (TableSchema("Laboratory", lab_cols),))
    lab_intent = canonicalize_intent("How many abnormal lab readings occurred?")
    linked_lab = link_schema_with_fk_closure(lab_schema, lab_intent)
    kept_cols = dict(linked_lab.selected_columns)["Laboratory"]
    self.assertEqual(len(kept_cols), 44)

  def test_iteration4_asr_casing_followup_arity_and_hierarchy_grounding(
      self,
  ) -> None:
    """Voice ASR casing, `? Show`/`Rank` arity, and County-over-City grounding."""
    from argus_sql.linking.types import ValueIndexEntry

    # 1. Lowercase ASR utterances align casing/punctuation verbatim with Clean.
    # Verifies alphanumeric bond ID `TR004_8_9` and terminal `?` restoration.
    ev_bond = "TR004_8_9 bond atoms refers to bond_id = 'TR004_8_9';"
    v_bond = canonicalize_intent(
        "uh hey can you please tell me what elements are you know in the tr004_8_9 bond atoms",
        ev_bond,
    )
    self.assertEqual(
        v_bond.canonical_question,
        "What elements are in the TR004_8_9 bond atoms?",
    )
    self.assertIn("TR004_8_9", v_bond.entities)

    # Verifies `DC` + `Marvel Comics` casing, `? Find`, and strict 1-col formula.
    ev_hero = (
        "DC refers to publisher_name = 'DC Comics'; "
        "Marvel Comics refers to publisher_name = 'Marvel Comics'; "
        "calculation = SUBTRACT(SUM(publisher_name = 'Marvel Comics'), "
        "SUM(publisher_name = 'DC Comics'))"
    )
    v_hero = canonicalize_intent(
        "uh hey can you please tell me between dc and you know marvel comics, "
        "which publisher has published more superheroes? find the difference "
        "in the number of superheroes they have published",
        ev_hero,
    )
    self.assertEqual(
        v_hero.canonical_question,
        "Between DC and Marvel Comics, which publisher has published more "
        "superheroes? Find the difference in the number of superheroes they "
        "have published.",
    )
    self.assertEqual(v_hero.expected_projection_arity, 1)
    self.assertTrue(v_hero.strict_single_column)

    # 2. `? Indicate ...` / `? Show ...` follow-up sentences & `Rank X by Y`.
    # Follow-up `? Indicate` requests a 2nd column, disabling `strict_single_column`.
    ev_pct = (
        "published by Marvel Comics refers to publisher_name = 'Marvel Comics'; "
        "calculation = MULTIPLY(DIVIDE(SUM(alignment = 'Bad'), count(id)), 100)"
    )
    v_pct = canonicalize_intent(
        "what is the percentage of bad superheroes? indicate how many of the "
        "said superheroes were published by marvel comics",
        ev_pct,
    )
    self.assertEqual(v_pct.expected_projection_arity, 2)
    self.assertFalse(v_pct.strict_single_column)

    # Follow-up `? Show` and imperative `Rank ... by ...` set arity 2 and 3.
    show_intent = canonicalize_intent(
        "Who was the player that got the lap time of 0:01:27? "
        "Show his introduction website."
    )
    self.assertEqual(show_intent.expected_projection_arity, 2)
    rank_intent = canonicalize_intent(
        "Rank heroes published by Marvel Comics by their height in descending order."
    )
    self.assertEqual(rank_intent.expected_projection_arity, 3)

    # Noun-phrase 'Which kind of', 'no. of' abbreviations, and conditional `?`.
    self.assertEqual(
        canonicalize_intent("Which kind of credit card does client 9 possess?").canonical_question,
        "Which kind of credit card does client 9 possess?",
    )
    self.assertEqual(
        canonicalize_intent("List out the no. of districts with salary > 6000?").canonical_question,
        "List out the no. of districts with salary > 6000?",
    )
    self.assertEqual(
        canonicalize_intent(
            "is it true that more smes pay in koruna? if so, how many more"
        ).canonical_question,
        "Is it true that more smes pay in koruna? If so, how many more?",
    )
    self.assertEqual(
        canonicalize_intent(
            "rank schools by writing score where score > 499, showing charter numbers"
        ).canonical_question,
        "Rank schools by writing score where score > 499, showing charter numbers.",
    )

    # 3. Hierarchy cardinality disambiguation (`County` 58 vs `City`/`MailCity` 500)
    #    and parenthetical column-qualifier (`K-12`) suppression.
    geo_schema = DatabaseSchema(
        "schools_geo",
        (
            TableSchema(
                "schools",
                (
                    ColumnSchema("CDSCode", "TEXT", is_primary_key=True),
                    ColumnSchema("County", "TEXT"),
                    ColumnSchema("City", "TEXT"),
                    ColumnSchema("MailCity", "TEXT"),
                    ColumnSchema("GSserved", "TEXT"),
                    ColumnSchema("Enrollment (K-12)", "INTEGER"),
                ),
            ),
        ),
    )
    _, quals = ("schools",), ("k-12",)
    del _
    entries = [
        ValueIndexEntry("schools", "County", "Fresno", "fresno", frozenset({"  f", "fre"}), 58),
        ValueIndexEntry("schools", "City", "Fresno", "fresno", frozenset({"  f", "fre"}), 500),
        ValueIndexEntry("schools", "MailCity", "Fresno", "fresno", frozenset({"  f", "fre"}), 500),
        ValueIndexEntry("schools", "City", "Hayward", "hayward", frozenset({"  h", "hay"}), 500),
        ValueIndexEntry("schools", "GSserved", "K-12", "k-12", frozenset({"  k", "k-1"}), 80),
    ]
    ret = CellValueRetriever(
        entries,
        schema_tokens=frozenset({"schools", "school", "county", "city", "enrollment"}),
        schema_qualifiers=frozenset(quals),
    )
    # Ambiguous `'Fresno schools'` grounds `County` while suppressing `City`/`MailCity`.
    fresno_vms = ret.retrieve(canonicalize_intent("Average test takers from Fresno schools?"))
    self.assertEqual([(m.column_name, m.matched_value) for m in fresno_vms], [("County", "Fresno")])
    # Explicit `'Fresno city'` retains `City`, and city-only `'Hayward'` grounds `City`.
    city_vms = ret.retrieve(canonicalize_intent("Average test takers in Fresno city?"))
    self.assertIn(("City", "Fresno"), [(m.column_name, m.matched_value) for m in city_vms])
    hayward_vms = ret.retrieve(canonicalize_intent("How many schools are in Hayward?"))
    self.assertEqual([(m.column_name, m.matched_value) for m in hayward_vms], [("City", "Hayward")])
    # Unquoted `'K-12'` column qualifier is suppressed on `Enrollment (K-12)` queries.
    k12_vms = ret.retrieve(
        canonicalize_intent("Give names of schools with free meals in K-12 > 0.1")
    )
    self.assertEqual(k12_vms, ())
    del geo_schema

  def test_extract_raw_entities_steelmans_multiword_spans_without_voice_leaks(
      self,
  ) -> None:
    """`extract_raw_entities` captures multi-word entities without voice rules."""
    from argus_sql.linking.intent_canonicalizer import extract_raw_entities

    clean_spans = extract_raw_entities(
        "What is the overall free meal rate in Los Angeles for 'Magnet' schools?",
        "Marvel Comics refers to publisher_name = 'Marvel Comics'; TR004_8_9",
    )
    self.assertIn("Los Angeles", clean_spans)
    self.assertIn("Magnet", clean_spans)
    self.assertIn("Marvel Comics", clean_spans)
    self.assertIn("TR004_8_9", clean_spans)

    # Spoken abbreviations ('la county', 'ny') must NOT expand without Stage 1.
    spoken_spans = extract_raw_entities(
        "uh hey tell me the overall free meal rate in la county and state ny"
    )
    self.assertNotIn("Los Angeles", spoken_spans)
    self.assertNotIn("New York", spoken_spans)

  def test_iteration5_their_clauses_and_multi_attribute_followup_arity(
      self,
  ) -> None:
    """Widens `'their'` & follow-up arity (QID 1157, 866_voice) safely."""
    # 1. QID 1157 Clean & Voice `'their ID, sex and disease ...'` estimates 3.
    ev_1157 = (
        "severe degree of thrombosis refers to thrombosis = 2; "
        "disease refers to diagnosis;"
    )
    clean_1157 = canonicalize_intent(
        "For patients with severe degree of thrombosis, list their ID, sex "
        "and disease the patient is diagnosed with.",
        ev_1157,
    )
    voice_1157 = canonicalize_intent(
        "uh hey can you please tell me for patients with you know severe "
        "degree of thrombosis, list their id, sex and disease the patient is "
        "diagnosed with",
        ev_1157,
    )
    self.assertEqual(clean_1157.expected_projection_arity, 3)
    self.assertTrue(clean_1157.has_followup_projection)
    self.assertEqual(voice_1157.expected_projection_arity, 3)
    self.assertTrue(voice_1157.has_followup_projection)

    # 2. Spoken `866_voice` follow-up `? show his introduction website` sets 2
    #    with `has_followup_projection=True` so 1-col SQL outputs are flagged.
    voice_866 = canonicalize_intent(
        "uh hey can you please tell me who was the you know player that got "
        "the lap time of 0:01:27 in the race no. 161? show his introduction "
        "website",
        "player and driver are synonyms; introduction website refers to url;",
    )
    self.assertEqual(voice_866.expected_projection_arity, 2)
    self.assertTrue(voice_866.has_followup_projection)

    # 3. Multi-attribute follow-ups (`? Also list ...`, multi-head + follow-up,
    #    and conjoined entity + `'their'` lists) widen arity accurately.
    also_followup = canonicalize_intent(
        "Which drivers competed in 2009? Also list their forename and nationality."
    )
    self.assertEqual(also_followup.expected_projection_arity, 3)
    self.assertTrue(also_followup.has_followup_projection)

    multi_head_followup = canonicalize_intent(
        "What is the phone number and extension number for the school with "
        "zip code 95203? Indicate the school's name."
    )
    self.assertEqual(multi_head_followup.expected_projection_arity, 3)
    self.assertTrue(multi_head_followup.has_followup_projection)

    entity_their = canonicalize_intent(
        "List the clients and their account ID and birth date."
    )
    self.assertEqual(entity_their.expected_projection_arity, 3)
    self.assertTrue(entity_their.has_followup_projection)

    # 4. Boundary guards: `'their'` inside a WHERE filter or explicit single-value
    #    directive (`Give me the single ...`) must remain arity 1 (`False`).
    where_their = canonicalize_intent(
        "List the schools where students bring their lunch and snacks."
    )
    self.assertEqual(where_their.expected_projection_arity, 1)
    self.assertFalse(where_their.has_followup_projection)

    single_val = canonicalize_intent(
        "What was their average monthly consumption? "
        "Give me the single average value rounded to two decimal places."
    )
    self.assertEqual(single_val.expected_projection_arity, 1)
    self.assertFalse(single_val.has_followup_projection)

  def test_followup_arity_declarative_context_filter_tails_and_asr_unpunctuated(
      self,
  ) -> None:
    """Guards 1-col context/filter queries while catching ASR `and also show`."""
    # 1. Two-sentence `[Declarative context]. [1-col question]` must NOT be
    #    misclassified as an additive 2-column follow-up projection.
    for one_col_context_q in (
        "Charter schools refer to Charter = 1 in frpm. "
        "What is the total number of charter schools?",
        "A severe thrombosis has a degree of 2. "
        "List the ID of patients with severe thrombosis.",
        "Account 45 belongs to the Prague branch. "
        "State the balance of this account.",
    ):
      ctx_intent = canonicalize_intent(one_col_context_q)
      self.assertEqual(ctx_intent.expected_projection_arity, 1)
      self.assertFalse(ctx_intent.has_followup_projection)

    # 2. Single-column `'their'`/`'his'` queries with trailing prepositional
    #    filters containing `'and'` (or subordinate clauses) stay `(1, False)`.
    for one_col_filter_q in (
        "What was their consumption between 2012 and 2013?",
        "For active members, list their ID for accounts with loans and deposits.",
        "Show his lap time in 2008 and 2009.",
        "List their email from offices in New York and California.",
        "List their email in New York and California.",
        "How many users show their email and phone?",
        "What is the total consumption of fuels, including diesel and gasoline?",
    ):
      flt_intent = canonicalize_intent(one_col_filter_q)
      self.assertEqual(flt_intent.expected_projection_arity, 1)
      self.assertFalse(flt_intent.has_followup_projection)

    # 3. 3-column `'their'` query with `>8`-word `for ...` filter and `'Find
    #    their ...'` retain all 3 attributes (`(3, True)`).
    long_for = canonicalize_intent(
        "List their ID, sex and diagnosis for all patients with severe degree "
        "of thrombosis."
    )
    self.assertEqual(long_for.expected_projection_arity, 3)
    self.assertTrue(long_for.has_followup_projection)

    find_their = canonicalize_intent("Find their ID, sex and diagnosis.")
    self.assertEqual(find_their.expected_projection_arity, 3)
    self.assertTrue(find_their.has_followup_projection)

    # 4. Unpunctuated spoken ASR follow-up (`and also show ...`) widens to 2.
    asr_unpunct = canonicalize_intent(
        "uh hey can you please tell me who was the player that got the lap "
        "time of 0:01:27 in race 161 and also show his introduction website"
    )
    self.assertEqual(asr_unpunct.expected_projection_arity, 2)
    self.assertTrue(asr_unpunct.has_followup_projection)


if __name__ == "__main__":
  unittest.main()
