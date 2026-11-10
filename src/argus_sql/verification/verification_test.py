"""Unit tests for semantic invariants, algebraic repair, and consensus voting."""

import unittest

from argus_sql.contracts.types import ExecutionOutcome
from argus_sql.contracts.types import LinkedSchemaContext
from argus_sql.contracts.types import SQLCandidate
from argus_sql.contracts.types import ValueMatch
from argus_sql.linking.intent_canonicalizer import canonicalize_intent
from argus_sql.sandbox.sqlite_sandbox import compute_result_signature
from argus_sql.verification.consensus_arbiter import verify_and_rank_candidates
from argus_sql.verification.exceptions import VerificationError
from argus_sql.verification.semantic_verifier import audit_candidate_invariants
from argus_sql.verification.sql_repairer import apply_algebraic_sql_repairs
from argus_sql.verification.sql_repairer import build_diagnostic_repair_prompt


class VerificationAndArbiterTest(unittest.TestCase):
  """Proves Argus defeats integer-truncation majority traps and NULL sort bugs."""

  def test_detects_and_repairs_integer_division_and_null_order_limit(self) -> None:
    """Truncated division and unguarded ASC LIMIT 1 must be caught and repaired."""
    intent = canonicalize_intent(
        "What is the lowest free meal rate among schools?",
        evidence="free meal rate = free_meals / enrollment",
    )
    # Include `'Free/Reduced'` slash literal and `OR` predicate to test safety!
    buggy = SQLCandidate(
        sql=(
            "SELECT free_meals / enrollment FROM frpm "
            "WHERE category = 'Free/Reduced' OR state = 'CA' "
            "ORDER BY score ASC LIMIT 1"
        ),
        strategy="baseline_cot",
    )
    outcome = ExecutionOutcome(
        succeeded=True,
        rows=((0,),),
        column_names=("rate",),
        row_count=1,
        result_signature=compute_result_signature(((0,),)),
    )
    issues = audit_candidate_invariants(buggy, outcome, intent)
    codes = {i.code for i in issues}
    self.assertIn("INTEGER_DIVISION_TRUNCATION", codes)
    self.assertIn("UNFILTERED_NULL_ASC_LIMIT", codes)

    # Algebraic repair must inject REAL cast, preserve `'Free/Reduced'`, and
    # wrap the `OR` condition in parentheses before `AND score IS NOT NULL`!
    repaired = apply_algebraic_sql_repairs(buggy, issues)
    self.assertIsNotNone(repaired)
    assert repaired is not None
    self.assertIn("CAST(free_meals AS REAL) / NULLIF(enrollment, 0)", repaired.sql)
    self.assertIn("'Free/Reduced'", repaired.sql)
    self.assertIn(
        "WHERE (category = 'Free/Reduced' OR state = 'CA') AND score IS NOT NULL",
        repaired.sql,
    )

  def test_subquery_where_does_not_corrupt_outer_null_guard_injection(self) -> None:
    """Subquery WHERE must not trick outer ORDER BY ASC LIMIT repair into AND."""
    intent = canonicalize_intent("What is the lowest score?")
    subquery_cand = SQLCandidate(
        sql="SELECT score FROM (SELECT score FROM frpm WHERE active = 1) ORDER BY score ASC LIMIT 1",
        strategy="cot",
    )
    out = ExecutionOutcome(
        True, ((None,),), ("score",), 1, 0.5, "", compute_result_signature(((None,),))
    )
    issues = audit_candidate_invariants(subquery_cand, out, intent)
    repaired = apply_algebraic_sql_repairs(subquery_cand, issues)
    self.assertIsNotNone(repaired)
    assert repaired is not None
    # Outer scope lacked WHERE, so `WHERE score IS NOT NULL` must be injected.
    self.assertIn(") WHERE score IS NOT NULL ORDER BY score ASC LIMIT 1", repaired.sql)

    # Inner subquery `ORDER BY ... ASC LIMIT 1` and `UNION ALL` must not inject
    # duplicate `WHERE` clauses inside parenthesized subqueries or compound arms.
    inner_order = SQLCandidate(
        sql="SELECT s FROM (SELECT s FROM frpm WHERE active = 1 ORDER BY s ASC LIMIT 1)",
        strategy="cot",
    )
    inner_issues = audit_candidate_invariants(inner_order, out, intent)
    self.assertIsNone(apply_algebraic_sql_repairs(inner_order, inner_issues))

    union_order = SQLCandidate(
        sql="SELECT s FROM frpm UNION ALL SELECT s FROM frpm ORDER BY s ASC LIMIT 1",
        strategy="cot",
    )
    union_issues = audit_candidate_invariants(union_order, out, intent)
    self.assertIsNone(apply_algebraic_sql_repairs(union_order, union_issues))

  def test_invariant_consensus_beats_naive_majority_truncation_trap(self) -> None:
    """Two REAL-cast candidates must beat three integer-truncated candidates."""
    # Construct realistic ratio query where 3/5 samples omit REAL casting.
    intent = canonicalize_intent("Calculate the ratio of approved loans.")
    sig_zero = compute_result_signature(((0,),))
    sig_real = compute_result_signature(((0.425,),))

    # 3 naive candidates truncate to 0; 2 defensive candidates return 0.425.
    cands = (
        SQLCandidate("SELECT approved / total FROM stats", "s1"),
        SQLCandidate("SELECT approved / total FROM stats", "s2"),
        SQLCandidate("SELECT approved / total FROM stats", "s3"),
        SQLCandidate("SELECT CAST(approved AS REAL) / total FROM stats", "def1"),
        SQLCandidate("SELECT 1.0 * approved / total FROM stats", "def2"),
    )
    outs = (
        ExecutionOutcome(True, ((0,),), ("r",), 1, 1.0, "", sig_zero),
        ExecutionOutcome(True, ((0,),), ("r",), 1, 1.0, "", sig_zero),
        ExecutionOutcome(True, ((0,),), ("r",), 1, 1.0, "", sig_zero),
        ExecutionOutcome(True, ((0.425,),), ("r",), 1, 1.1, "", sig_real),
        ExecutionOutcome(True, ((0.425,),), ("r",), 1, 1.2, "", sig_real),
    )

    winner, _, clusters = verify_and_rank_candidates(cands, outs, intent)
    # Naive majority would have picked 0; Argus picks 0.425!
    self.assertEqual(winner.outcome.result_signature, sig_real)
    self.assertEqual(clusters[0].result_signature, sig_real)

    # Mismatched repaired_flags length must raise domain VerificationError.
    with self.assertRaises(VerificationError):
      verify_and_rank_candidates(cands, outs, intent, repaired_flags=(True,))

  def test_backtick_slash_identifiers_and_nested_cast_expressions(self) -> None:
    """Backtick `(Y/N)` identifiers and `CAST(SUM(x) AS REAL)` stay intact."""
    # 1. `Charter School (Y/N)` has `/` inside backticks—must NOT false-alarm!
    clean_intent = canonicalize_intent("List zip codes of charter schools.")
    yn_cand = SQLCandidate(
        "SELECT Zip FROM frpm WHERE `Charter School (Y/N)` = 1 ORDER BY `Enrollment (K-12)` ASC LIMIT 1",
        "cot",
    )
    yn_out = ExecutionOutcome(
        True, (("93706",),), ("Zip",), 1, 0.4, "", compute_result_signature((("93706",),))
    )
    yn_issues = audit_candidate_invariants(yn_cand, yn_out, clean_intent)
    self.assertNotIn("INTEGER_DIVISION_TRUNCATION", {i.code for i in yn_issues})
    yn_rep = apply_algebraic_sql_repairs(yn_cand, yn_issues)
    self.assertIsNotNone(yn_rep)
    assert yn_rep is not None
    self.assertIn("`Charter School (Y/N)` = 1", yn_rep.sql)
    self.assertIn("`Enrollment (K-12)` IS NOT NULL", yn_rep.sql)

    # 2. Nested `CAST(SUM(a) AS REAL) / COUNT(b)` is already float-promoted.
    ratio_intent = canonicalize_intent("What is the average math ratio?")
    nested_cast = SQLCandidate(
        "SELECT CAST(SUM(AvgScrMath) AS REAL) / COUNT(cds) FROM satscores",
        "defensive",
    )
    cast_out = ExecutionOutcome(
        True, ((512.4,),), ("r",), 1, 0.4, "", compute_result_signature(((512.4,),))
    )
    self.assertEqual(
        audit_candidate_invariants(nested_cast, cast_out, ratio_intent), ()
    )

  def test_avg_and_trailing_float_promotion_and_join_fanout_distinct_repair(
      self,
  ) -> None:
    """AVG()/12 and * 100.0 pass cleanly; parenthesized / and fanout repair."""
    pct_intent = canonicalize_intent("What is the percentage increase?")
    # `AVG(x) / 12` and `(a - b) * 100.0 / b` already promote to SQLite REAL.
    for clean_sql in (
        "SELECT AVG(Consumption) / 12 FROM yearmonth",
        "SELECT (c2013 - c2012) * 100.0 / c2012 FROM segs",
    ):
      cand = SQLCandidate(clean_sql, "cot")
      out = ExecutionOutcome(
          True, ((0,),), ("v",), 1, 0.3, "", compute_result_signature(((0,),))
      )
      codes = {
          i.code for i in audit_candidate_invariants(cand, out, pct_intent)
      }
      self.assertNotIn("INTEGER_DIVISION_TRUNCATION", codes)

    # Uncast parenthesized `(c2013 - c2012) * 100 / c2012` must promote cleanly
    # at zero LLM token cost while guarding division by zero via `NULLIF`.
    uncast_paren = SQLCandidate(
        "SELECT (c2013 - c2012) * 100 / c2012 FROM segs", "cot"
    )
    p_out = ExecutionOutcome(
        True, ((0,),), ("v",), 1, 0.3, "", compute_result_signature(((0,),))
    )
    p_issues = audit_candidate_invariants(uncast_paren, p_out, pct_intent)
    p_rep = apply_algebraic_sql_repairs(uncast_paren, p_issues)
    self.assertIsNotNone(p_rep)
    assert p_rep is not None
    self.assertIn(
        "CAST((c2013 - c2012) * 100 AS REAL) / NULLIF(c2012, 0)", p_rep.sql
    )

    # 1-to-many join returning duplicate entity names triggers SELECT DISTINCT
    # so consensus and R-VES favor deduplicated projections.
    list_intent = canonicalize_intent(
        "Please list the product descriptions sold in CZE gas stations."
    )
    fanout_cand = SQLCandidate(
        "SELECT T1.Description FROM products T1 JOIN transactions_1k T2 "
        "ON T1.ProductID = T2.ProductID WHERE T2.Amount > 1",
        "cot",
    )
    # Construct 10 rows containing only 2 distinct values (20% unique <= 50%).
    dup_rows = (("Nafta",),) * 6 + (("Natural 95",),) * 4
    fanout_out = ExecutionOutcome(
        True,
        dup_rows,
        ("Description",),
        len(dup_rows),
        0.5,
        "",
        compute_result_signature(dup_rows),
    )
    f_issues = audit_candidate_invariants(fanout_cand, fanout_out, list_intent)
    self.assertIn("JOIN_FANOUT_DUPLICATES", {i.code for i in f_issues})
    f_rep = apply_algebraic_sql_repairs(fanout_cand, f_issues)
    self.assertIsNotNone(f_rep)
    assert f_rep is not None
    self.assertTrue(f_rep.sql.startswith("SELECT DISTINCT T1.Description"))

    # Ordered JOIN listings (`ORDER BY`) must NOT trigger SELECT DISTINCT dedup.
    ordered_fanout = SQLCandidate(
        "SELECT T1.Description FROM products T1 JOIN transactions_1k T2 "
        "ON T1.ProductID = T2.ProductID ORDER BY T2.Amount DESC",
        "cot",
    )
    ord_codes = {
        i.code
        for i in audit_candidate_invariants(ordered_fanout, fanout_out, list_intent)
    }
    self.assertNotIn("JOIN_FANOUT_DUPLICATES", ord_codes)

    # Function-call numerators (`COALESCE`, `CAST(... AS DECIMAL)`) and `NULLIF`
    # denominators must promote cleanly without double-wrapping `NULLIF`.
    fn_div = SQLCandidate(
        "SELECT COALESCE(w.total_cost, 0) / NULLIF(a.attendees, 0) FROM att a",
        "cot",
    )
    fn_issues = audit_candidate_invariants(fn_div, p_out, pct_intent)
    fn_rep = apply_algebraic_sql_repairs(fn_div, fn_issues)
    self.assertIsNotNone(fn_rep)
    assert fn_rep is not None
    self.assertEqual(
        fn_rep.sql,
        "SELECT CAST(COALESCE(w.total_cost, 0) AS REAL) / NULLIF(a.attendees, 0) FROM att a",
    )

  def test_empty_set_consensus_immunity_and_range_aware_casing_repair(
      self,
  ) -> None:
    """Single clean repair (1.00) beats 3 empty-set votes (0.54) & fixes casing."""
    intent = canonicalize_intent(
        "List product descriptions consumed in September 2013."
    )
    empty_sig = compute_result_signature(())
    good_sig = compute_result_signature((("Diesel",),))

    # 3 distinct initial strategies return 0 rows; 1 LLM repair returns data.
    cands = (
        SQLCandidate("SELECT Description FROM tx WHERE Date LIKE '2013-09%'", "decomposed_cot"),
        SQLCandidate("SELECT Description FROM tx WHERE Date >= '2013-09-01'", "formula_defensive"),
        SQLCandidate("SELECT Description FROM tx WHERE strftime('%Y-%m', Date) = '2013-09'", "skeleton_minimal"),
        SQLCandidate("SELECT DISTINCT p.Description FROM tx JOIN yearmonth y ON tx.cid = y.cid JOIN products p ON tx.pid = p.pid WHERE y.Date = '201309'", "argus_llm_repair"),
    )
    outs = (
        ExecutionOutcome(True, (), ("Description",), 0, 1.0, "", empty_sig),
        ExecutionOutcome(True, (), ("Description",), 0, 1.0, "", empty_sig),
        ExecutionOutcome(True, (), ("Description",), 0, 1.0, "", empty_sig),
        ExecutionOutcome(True, (("Diesel",),), ("Description",), 1, 1.2, "", good_sig),
    )
    winner, _, clusters = verify_and_rank_candidates(
        cands, outs, intent, repaired_flags=(False, False, False, True)
    )
    self.assertEqual(winner.candidate.strategy, "argus_llm_repair")
    self.assertEqual(clusters[0].result_signature, good_sig)
    self.assertEqual(clusters[0].total_score, 1.0)
    self.assertAlmostEqual(clusters[1].total_score, 0.54)

    # Verify DDL `range:` extraction and exact-word casing repair on ALL_NULL!
    linked = LinkedSchemaContext(
        db_id="debit",
        selected_tables=("gasstations", "tx", "yearmonth"),
        selected_columns=(),
        fk_closure_edges=(),
        bridge_tables_added=(),
        value_matches=(
            ValueMatch("gasstations", "Segment", "Discount", "Discount", 1.0),
        ),
        ddl_prompt=(
            "CREATE TABLE tx (\n  Date DATE -- range: '2012-08-23'..'2012-08-26'\n);\n\n"
            "CREATE TABLE yearmonth (\n  Date TEXT -- range: '201112'..'201311'\n);"
        ),
    )
    null_cand = SQLCandidate(
        "SELECT SUM(Amount) FROM gasstations WHERE Segment = 'discount' AND Date = '201309'",
        "cot",
    )
    null_out = ExecutionOutcome(
        True, ((None,),), ("s",), 1, 0.4, "", compute_result_signature(((None,),))
    )
    null_issues = audit_candidate_invariants(null_cand, null_out, intent, linked)
    self.assertEqual(null_issues[0].code, "ALL_NULL_RESULT_ANOMALY")
    self.assertIn("tx.Date='2012-08-23'..'2012-08-26'", null_issues[0].repair_hint)
    self.assertIn("yearmonth.Date='201112'..'201311'", null_issues[0].repair_hint)

    # Casing mismatch `'discount'` -> `'Discount'` repairs even when span == val!
    cased_rep = apply_algebraic_sql_repairs(null_cand, null_issues, linked)
    self.assertIsNotNone(cased_rep)
    assert cased_rep is not None
    self.assertIn("Segment = 'Discount'", cased_rep.sql)

    # PostgreSQL `ILIKE` rewrites to SQLite `LIKE` without touching literals.
    ilike_cand = SQLCandidate(
        "SELECT * FROM gasstations WHERE Segment ILIKE '%disc%' AND Note = 'ILIKE'",
        "cot",
    )
    err_out = ExecutionOutcome(False, error_message="near \"ILIKE\": syntax error")
    err_issues = audit_candidate_invariants(ilike_cand, err_out, intent)
    ilike_rep = apply_algebraic_sql_repairs(ilike_cand, err_issues)
    self.assertIsNotNone(ilike_rep)
    assert ilike_rep is not None
    self.assertIn("Segment LIKE '%disc%'", ilike_rep.sql)
    self.assertIn("Note = 'ILIKE'", ilike_rep.sql)

    # Diagnostic repair prompt surfaces deduplicated sibling failed SQLs.
    prompt = build_diagnostic_repair_prompt(
        "List products",
        "",
        linked.ddl_prompt,
        cands[0],
        null_issues,
        failed_sqls=(cands[0].sql, cands[1].sql, cands[1].sql),
    )
    self.assertIn("Other Failed SQL Attempts", prompt)
    self.assertIn(cands[1].sql, prompt)

    # Exact DB match ('Alameda', 1.0) must NEVER be clobbered by a lower-score
    # prefix match ('Alameda Unified', 0.86), and `);` inside `--` DDL comments
    # must not truncate sibling temporal `range:` extraction!
    prefix_linked = LinkedSchemaContext(
        db_id="schools",
        selected_tables=("schools", "satscores"),
        selected_columns=(),
        fk_closure_edges=(),
        bridge_tables_added=(),
        value_matches=(
            ValueMatch("schools", "County", "Alameda", "Alameda", 1.0),
            ValueMatch("satscores", "dname", "Alameda Unified", "Alameda", 0.86),
        ),
        ddl_prompt=(
            'CREATE TABLE schools (\n'
            '  "Charter (Y/N)" INTEGER, -- (0: N); 1: Y | range: 0..1\n'
            '  OpenDate DATE -- range: \'1980-01-01\'..\'2017-09-22\'\n'
            ');'
        ),
    )
    exact_cand = SQLCandidate(
        "SELECT School FROM schools WHERE County = 'Alameda' AND Note = 'It''s alameda' AND \"Charter (Y/N)\" = 9",
        "cot",
    )
    empty_out = ExecutionOutcome(True, (), ("School",), 0, 0.5, "", empty_sig)
    p_issues = audit_candidate_invariants(exact_cand, empty_out, intent, prefix_linked)
    # `OpenDate` range after `);` comment survives and ranks ahead of numeric!
    self.assertIn("schools.OpenDate='1980-01-01'..'2017-09-22'", p_issues[0].repair_hint)
    # Exact `'Alameda'` and multi-word `'It''s alameda'` both remain untouched!
    self.assertIsNone(apply_algebraic_sql_repairs(exact_cand, p_issues, prefix_linked))

  def test_iteration3_relational_join_ratio_entity_key_and_boolean_avg(
      self,
  ) -> None:
    """1:N LEFT JOIN ratios, display-name GROUP BY, and boolean AVG are caught."""
    # 1. 1:N child `LEFT JOIN ... ON ... AND ...` ratio triggers ERROR, whereas
    #    1:0..1 PK-joined extension `LEFT JOIN` triggers zero false positives.
    card_ddl = (
        "CREATE TABLE cards (\n  id INTEGER PRIMARY KEY,\n  uuid TEXT,\n"
        "  isStorySpotlight INTEGER\n);\n\n"
        "CREATE TABLE foreign_data (\n  id INTEGER PRIMARY KEY,\n"
        "  language TEXT,\n  uuid TEXT\n);\n\n"
        "CREATE TABLE satscores (\n  cds TEXT PRIMARY KEY,\n  AvgScrMath INTEGER\n);"
    )
    linked_cards = LinkedSchemaContext(
        "card_games", ("cards", "foreign_data"), (), (), (), (), card_ddl
    )
    pct_intent = canonicalize_intent(
        "What is the percentage of French cards among Story Spotlight cards?"
    )
    bad_lj = SQLCandidate(
        "SELECT CAST(COUNT(T2.id) AS REAL) * 100.0 / NULLIF(COUNT(T1.id), 0) "
        "FROM cards AS T1 LEFT JOIN foreign_data AS T2 "
        "ON T1.uuid = T2.uuid AND T2.language = 'French' "
        "WHERE T1.isStorySpotlight = 1",
        "formula_defensive",
    )
    clean_ij = SQLCandidate(
        "SELECT 100.0 * COUNT(CASE WHEN T2.language = 'French' THEN 1 END) / COUNT(*) "
        "FROM cards AS T1 INNER JOIN foreign_data AS T2 ON T1.uuid = T2.uuid "
        "WHERE T1.isStorySpotlight = 1",
        "skeleton_minimal",
    )
    bad_sig = compute_result_signature(((66.7,),))
    good_sig = compute_result_signature(((12.5,),))
    bad_out = ExecutionOutcome(True, ((66.7,),), ("p",), 1, 1.0, "", bad_sig)
    good_out = ExecutionOutcome(True, ((12.5,),), ("p",), 1, 1.0, "", good_sig)
    lj_issues = audit_candidate_invariants(bad_lj, bad_out, pct_intent, linked_cards)
    self.assertEqual(lj_issues[0].code, "LEFT_JOIN_FILTERED_RATIO")
    self.assertEqual(lj_issues[0].severity, "ERROR")
    # 1 clean INNER JOIN candidate beats even 2 distorted LEFT JOIN candidates!
    lj_win, _, _ = verify_and_rank_candidates(
        (bad_lj, bad_lj, clean_ij),
        (bad_out, bad_out, good_out),
        pct_intent,
        linked_cards,
    )
    self.assertEqual(lj_win.candidate.sql, clean_ij.sql)

    # 1:0..1 PK-joined `satscores (cds PRIMARY KEY)` (including reversed/lower `and`)
    # and deduplicated window-function CTEs (`WITH latest AS (...)`) must NOT alarm!
    pk_lj = SQLCandidate(
        "SELECT CAST(COUNT(T2.cds) AS REAL) / COUNT(T1.id) FROM cards T1 "
        "LEFT JOIN satscores T2 ON T2.AvgScrMath > 500 and T1.uuid = T2.cds",
        "cot",
    )
    cte_lj = SQLCandidate(
        "WITH latest AS (SELECT uuid, ROW_NUMBER() OVER (PARTITION BY uuid) AS rn "
        "FROM foreign_data) SELECT 100.0 * SUM(CASE WHEN l.rn = 1 THEN 1 ELSE 0 END) "
        "/ COUNT(c.id) FROM cards c LEFT JOIN latest l ON l.uuid = c.uuid AND l.rn = 1",
        "cot",
    )
    self.assertEqual(
        audit_candidate_invariants(pk_lj, good_out, pct_intent, linked_cards), ()
    )
    self.assertEqual(
        audit_candidate_invariants(cte_lj, good_out, pct_intent, linked_cards), ()
    )

    # 2. `GROUP BY T.team_long_name` (and `GROUP BY T.team_api_id, T.team_long_name`
    #    or `GROUP BY T.team_long_name, T.team_api_id`) rewrites algebraically to
    #    `GROUP BY M.away_team_api_id` without child-FK poisoning (`T.country_id`),
    #    and skips intentional homonym `COUNT(DISTINCT d.driverId)` queries!
    win_intent = canonicalize_intent(
        "Which away team won the most during the 2010 season?"
    )
    name_gb = SQLCandidate(
        "SELECT T.team_long_name FROM Match AS M\n"
        "INNER JOIN League AS L ON M.league_id = L.id\n"
        "INNER JOIN Team AS T\n  ON M.away_team_api_id = T.team_api_id\n"
        "WHERE M.away_team_goal > M.home_team_goal\n"
        "GROUP BY\n  T.team_long_name\nORDER BY COUNT(*) DESC LIMIT 1",
        "formula_defensive",
    )
    pair_gb = SQLCandidate(
        "SELECT T.team_long_name FROM Team AS T "
        "INNER JOIN Country AS C ON T.country_id = C.id "
        "INNER JOIN Match AS M ON T.team_api_id = M.away_team_api_id "
        "GROUP BY T.team_long_name, T.team_api_id ORDER BY COUNT(*) DESC LIMIT 1",
        "decomposed_cot",
    )
    homonym_gb = SQLCandidate(
        "SELECT d.forename, COUNT(DISTINCT d.driverId) AS n FROM results x "
        "INNER JOIN drivers d ON x.driverId = d.driverId "
        "GROUP BY d.forename HAVING COUNT(DISTINCT d.driverId) > 1",
        "cot",
    )
    t_sig_a = compute_result_signature((("Celtic",),))
    t_sig_b = compute_result_signature((("Rangers",),))
    out_a = ExecutionOutcome(True, (("Celtic",),), ("n",), 1, 1.0, "", t_sig_a)
    out_b = ExecutionOutcome(True, (("Rangers",),), ("n",), 1, 1.0, "", t_sig_b)
    gb_issues = audit_candidate_invariants(name_gb, out_a, win_intent)
    self.assertIn("NON_KEY_ENTITY_GROUPING", {i.code for i in gb_issues})
    gb_rep = apply_algebraic_sql_repairs(name_gb, gb_issues)
    self.assertIsNotNone(gb_rep)
    assert gb_rep is not None
    self.assertIn("GROUP BY\n  M.away_team_api_id", gb_rep.sql)
    pair_issues = audit_candidate_invariants(pair_gb, out_a, win_intent)
    pair_rep = apply_algebraic_sql_repairs(pair_gb, pair_issues)
    self.assertIsNotNone(pair_rep)
    assert pair_rep is not None
    self.assertIn("GROUP BY M.away_team_api_id ORDER BY", pair_rep.sql)
    self.assertNotIn("GROUP BY T.country_id", pair_rep.sql)
    hom_out = ExecutionOutcome(True, (("John", 2),), ("f", "n"), 1, 1.0, "", t_sig_a)
    self.assertEqual(audit_candidate_invariants(homonym_gb, hom_out, win_intent), ())

    # Verify 1 clean entity-key candidate (1.00) outvotes 2 WARNING ones (0.96).
    gb_win, _, _ = verify_and_rank_candidates(
        (name_gb, pair_gb, gb_rep), (out_a, out_a, out_b), win_intent
    )
    self.assertEqual(gb_win.outcome.result_signature, t_sig_b)

    # 3. `AVG(T1.element = 'o')` on average-count-per-entity question flags ERROR.
    cnt_avg_intent = canonicalize_intent(
        "Calculate the average number of oxygen atoms in single-bonded molecules.",
        evidence="single-bonded refers to bond_type = '-'; AVG(element = 'o')",
    )
    bool_avg_cand = SQLCandidate(
        "SELECT AVG(T1.element = 'o') FROM atom T1 JOIN bond T2 "
        "ON T1.molecule_id = T2.molecule_id WHERE T2.bond_type = '-'",
        "formula_defensive",
    )
    b_issues = audit_candidate_invariants(bool_avg_cand, good_out, cnt_avg_intent)
    self.assertEqual(b_issues[0].code, "BOOLEAN_AVG_FOR_COUNT")
    self.assertEqual(b_issues[0].severity, "ERROR")

  def test_iteration4_timestamp_date_rewrite_window_rank_and_strict_arity(
      self,
  ) -> None:
    """Audits & repairs bare DATETIME comparisons (I11), window ranks (I12), & arity."""
    # 1. `TIMESTAMP_BARE_DATE_COMPARISON` (I11) flags sub-day timestamp columns
    #    and rewrites to `DATE(col)` at 0 token cost while ignoring pure DATE cols!
    ts_ddl = (
        "CREATE TABLE users (\n"
        "  Id INTEGER PRIMARY KEY,\n"
        "  LastAccessDate DATETIME, -- range: '2010-07-19 19:39:07.000'..'2014-09-14 00:10:02.000'\n"
        "  BirthDay DATE -- range: '1980-01-01'..'2005-12-31'\n"
        ");"
    )
    linked_ts = LinkedSchemaContext(
        "community", ("users",), (), (), (), (), ts_ddl
    )
    ts_intent = canonicalize_intent(
        "How many users last accessed the website after 2014/9/1?",
        evidence="last accessed after 2014/9/1 refers to LastAccessDate > '2014-09-01'",
    )
    bare_ts = SQLCandidate(
        "SELECT COUNT(*) FROM users WHERE LastAccessDate > '2014-09-01'",
        "decomposed_cot",
    )
    pure_dt = SQLCandidate(
        "SELECT COUNT(*) FROM users WHERE BirthDay > '1990-01-01'",
        "decomposed_cot",
    )
    sig_bad = compute_result_signature(((42,),))
    sig_good = compute_result_signature(((37,),))
    out_bad = ExecutionOutcome(True, ((42,),), ("cnt",), 1, 1.0, "", sig_bad)
    out_good = ExecutionOutcome(True, ((37,),), ("cnt",), 1, 1.0, "", sig_good)

    ts_issues = audit_candidate_invariants(bare_ts, out_bad, ts_intent, linked_ts)
    self.assertEqual([i.code for i in ts_issues], ["TIMESTAMP_BARE_DATE_COMPARISON"])
    self.assertEqual(ts_issues[0].severity, "ERROR")
    # Pure DATE column (`range: '1980-01-01'..'2005-12-31'`) produces 0 issues!
    self.assertEqual(
        audit_candidate_invariants(pure_dt, out_good, ts_intent, linked_ts), ()
    )
    # Zero-token algebraic repair wraps `LastAccessDate` in `DATE(...)` idempotently.
    ts_rep = apply_algebraic_sql_repairs(bare_ts, ts_issues, linked_ts)
    self.assertIsNotNone(ts_rep)
    assert ts_rep is not None
    self.assertEqual(
        ts_rep.sql,
        "SELECT COUNT(*) FROM users WHERE DATE(LastAccessDate) > '2014-09-01'",
    )
    self.assertEqual(
        audit_candidate_invariants(ts_rep, out_good, ts_intent, linked_ts), ()
    )
    # Quoted table aliases (`AS "U"`) and `IN ('YYYY-MM-DD', ...)` also rewrite!
    in_ts = SQLCandidate(
        'SELECT COUNT(*) FROM users AS "U" WHERE LastAccessDate IN (\'2014-09-01\', \'2014-09-02\')',
        "formula_defensive",
    )
    in_issues = audit_candidate_invariants(in_ts, out_bad, ts_intent, linked_ts)
    self.assertEqual([i.code for i in in_issues], ["TIMESTAMP_BARE_DATE_COMPARISON"])
    in_rep = apply_algebraic_sql_repairs(in_ts, in_issues, linked_ts)
    assert in_rep is not None
    self.assertIn("DATE(LastAccessDate) IN ('2014-09-01', '2014-09-02')", in_rep.sql)

    # 2. `MISSING_WINDOW_RANK` (I12) flags imperative `Rank X by Y` queries
    #    missing `RANK() OVER (...)`, without false-alarming on `did X rank highest?`.
    rank_intent = canonicalize_intent(
        "Rank heroes published by Marvel Comics by their height in descending order."
    )
    no_rank_sql = SQLCandidate(
        "SELECT superhero_name, height_cm, 1 FROM superhero ORDER BY height_cm DESC",
        "decomposed_cot",
    )
    with_rank_sql = SQLCandidate(
        "SELECT superhero_name, height_cm, RANK() OVER (ORDER BY height_cm DESC) "
        "FROM superhero",
        "formula_defensive",
    )
    r_out = ExecutionOutcome(
        True, (("Hulk", 244, 1),), ("n", "h", "r"), 1, 1.0, "", sig_good
    )
    r_issues = audit_candidate_invariants(no_rank_sql, r_out, rank_intent)
    self.assertIn("MISSING_WINDOW_RANK", {i.code for i in r_issues})
    self.assertEqual(audit_candidate_invariants(with_rank_sql, r_out, rank_intent), ())
    non_window_q = canonicalize_intent("In which race did Hamilton rank the highest?")
    one_out = ExecutionOutcome(True, (("Monaco",),), ("r",), 1, 1.0, "", sig_good)
    self.assertEqual(
        audit_candidate_invariants(
            SQLCandidate(
                "SELECT name FROM races WHERE rank IS NOT NULL ORDER BY rank ASC LIMIT 1",
                "cot",
            ),
            one_out,
            non_window_q,
        ),
        (),
    )

    # 3. `strict_single_column` penalizes 2-col drift on `calculation =` tasks.
    calc_intent = canonicalize_intent(
        "Between DC and Marvel Comics, which publisher published more? "
        "Find the difference.",
        evidence="calculation = SUBTRACT(SUM(a), SUM(b))",
    )
    c_two = SQLCandidate("SELECT 'Marvel', 120", "skeleton_minimal")
    c_one = SQLCandidate("SELECT 120", "decomposed_cot")
    o_two = ExecutionOutcome(
        True, (("Marvel", 120),), ("p", "d"), 1, 1.0, "", sig_bad
    )
    o_one = ExecutionOutcome(True, ((120,),), ("d",), 1, 1.0, "", sig_good)
    win_calc, _, _ = verify_and_rank_candidates(
        (c_two, c_two, c_one), (o_two, o_two, o_one), calc_intent
    )
    self.assertEqual(win_calc.outcome.result_signature, sig_good)

  def test_iteration5_lineage_dedup_projection_tiebreak_wildcard_and_followup(
      self,
  ) -> None:
    """Covers QID 24 lineage dedup, QID 159 `*`/tie-break, & QID 1157/866_voice."""
    # 1. QID 24: Derived repair copy (`skeleton_minimal+argus_repair`) sharing
    #    an execution cluster with its parent (`skeleton_minimal`) must NOT
    #    inflate cluster score to `0.18 + 1.00 = 1.18` or earn diversity bonus,
    #    and projection-name alignment (`School Name` vs `sname`) breaks the
    #    `1.0` vs `1.0` tie ahead of execution latency (`8.0ms` vs `0.2ms`)!
    intent_24 = canonicalize_intent(
        "Give the names of the schools with the percent eligible for free "
        "meals in K-12 is more than 0.1 and test takers whose test score is "
        "greater than or equal to 1500?",
        evidence=(
            "Percent eligible for free meals = "
            "Free Meal Count (K-12) / Total (Enrollment (K-12)"
        ),
    )
    sig_sname = compute_result_signature((("East Los Angeles Performing",),))
    sig_full = compute_result_signature(
        (("East Los Angeles Performing Arts Academy",),)
    )
    c_skel_raw = SQLCandidate(
        'SELECT T1.sname FROM satscores AS T1 INNER JOIN frpm AS T2 '
        'ON T1.cds = T2.CDSCode WHERE T2."Free Meal Count (K-12)" / '
        'T2."Enrollment (K-12)" > 0.1 AND T1.NumGE1500 > 0',
        "skeleton_minimal",
    )
    c_skel_rep = SQLCandidate(
        'SELECT T1.sname FROM satscores AS T1 INNER JOIN frpm AS T2 '
        'ON T1.cds = T2.CDSCode WHERE CAST(T2."Free Meal Count (K-12)" AS REAL)'
        ' / NULLIF(T2."Enrollment (K-12)", 0) > 0.1 AND T1.NumGE1500 > 0',
        "skeleton_minimal+argus_repair",
    )
    c_cot_full = SQLCandidate(
        'SELECT T2."School Name" FROM satscores AS T1 INNER JOIN frpm AS T2 '
        'ON T1.cds = T2.CDSCode WHERE CAST(T2."Free Meal Count (K-12)" AS REAL)'
        ' / T2."Enrollment (K-12)" > 0.1 AND T1.NumGE1500 > 0',
        "decomposed_cot",
    )
    # Give `sname` candidates 40x faster latency (`0.2ms` vs `8.0ms`) to prove
    # projection-name alignment takes precedence over raw latency!
    o_sname = ExecutionOutcome(
        True, (("East Los Angeles Performing",),), ("sname",), 1, 0.2, "", sig_sname
    )
    o_full = ExecutionOutcome(
        True,
        (("East Los Angeles Performing Arts Academy",),),
        ("School Name",),
        1,
        8.0,
        "",
        sig_full,
    )
    win_24, _, clusters_24 = verify_and_rank_candidates(
        (c_skel_raw, c_skel_rep, c_cot_full),
        (o_sname, o_sname, o_full),
        intent_24,
        repaired_flags=(False, True, False),
    )
    by_sig_24 = {c.result_signature: c for c in clusters_24}
    self.assertEqual(by_sig_24[sig_sname].distinct_strategies, ("skeleton_minimal",))
    self.assertEqual(by_sig_24[sig_sname].total_score, 1.0)
    self.assertEqual(win_24.outcome.result_signature, sig_full)
    self.assertEqual(win_24.candidate.strategy, "decomposed_cot")

    # Even two completely clean derived copies (`s1`, `s1+argus_repair`,
    # `s1_repair`) collapse to score 1.0 with zero fake diversity bonus!
    _, _, clean_dup_clusters = verify_and_rank_candidates(
        (
            SQLCandidate("SELECT 1", "s1"),
            SQLCandidate("SELECT 1", "s1+argus_repair"),
            SQLCandidate("SELECT 1", "s1_repair"),
        ),
        (o_sname, o_sname, o_sname),
        canonicalize_intent("How many records?"),
    )
    self.assertEqual(clean_dup_clusters[0].distinct_strategies, ("s1",))
    self.assertEqual(clean_dup_clusters[0].total_score, 1.0)

    # 2. QID 159: `SELECT T1.*` / `SELECT *` escalates to `ERROR` whenever
    #    `expected_projection_arity <= 2`, while keeping `WARNING` at arity >= 3,
    #    ignoring `COUNT(*)` / CTE subqueries, and breaking within-cluster ties
    #    by projection header (`trans_id` > `client_id` / `amount`) before speed!
    intent_159 = canonicalize_intent(
        "List all the withdrawals in cash transactions that the client with "
        "the id 3356 makes.",
        evidence="operation = 'VYBER' refers to withdrawal in cash",
    )
    self.assertEqual(intent_159.expected_projection_arity, 1)
    star_cand = SQLCandidate(
        "SELECT T1.* FROM trans AS T1 INNER JOIN disp AS T2 "
        "ON T1.account_id = T2.account_id WHERE T2.client_id = 3356 "
        "AND T1.operation = 'VYBER'",
        "decomposed_cot",
    )
    wide_cols = ("trans_id", "account_id", "date", "type", "operation", "amount")
    wide_rows = ((991, 44, "1996-01-01", "VYDAJ", "VYBER", 500),)
    star_out = ExecutionOutcome(
        True, wide_rows, wide_cols, 1, 0.2, "", compute_result_signature(wide_rows)
    )
    star_issues = audit_candidate_invariants(star_cand, star_out, intent_159)
    self.assertEqual(len(star_issues), 1)
    self.assertEqual(star_issues[0].code, "PROJECTION_ARITY_DRIFT")
    self.assertEqual(star_issues[0].severity, "ERROR")

    # Arity 2 (`expected <= 2`) `SELECT *` on a 3-col table also flags `ERROR`,
    # whereas `expected == 3` (`> 2`) `SELECT *` and explicit 3-col over-projection
    # stay `WARNING`, and `WITH cte AS (SELECT * ...) SELECT trans_id` is clean!
    intent_arity2 = canonicalize_intent(
        "Who was the player that got the lap time of 0:01:27? "
        "Show his introduction website."
    )
    three_out = ExecutionOutcome(
        True,
        ((1, "Lewis", "u"),),
        ("id", "name", "url"),
        1,
        0.3,
        "",
        compute_result_signature(((1, "Lewis", "u"),)),
    )
    arity2_star = audit_candidate_invariants(
        SQLCandidate("SELECT * FROM drivers", "cot"), three_out, intent_arity2
    )
    self.assertEqual(arity2_star[0].severity, "ERROR")

    intent_arity3 = canonicalize_intent(
        "For patients with severe degree of thrombosis, list their ID, sex "
        "and disease the patient is diagnosed with."
    )
    arity3_star = audit_candidate_invariants(
        SQLCandidate("SELECT * FROM Patient", "cot"), star_out, intent_arity3
    )
    self.assertEqual(arity3_star[0].severity, "WARNING")
    explicit_over = audit_candidate_invariants(
        SQLCandidate("SELECT trans_id, account_id, amount FROM trans", "cot"),
        three_out,
        intent_159,
    )
    self.assertEqual(explicit_over[0].severity, "WARNING")

    cte_star_clean = SQLCandidate(
        "WITH c AS (SELECT * FROM trans WHERE operation = 'VYBER') "
        "SELECT trans_id FROM c",
        "formula_defensive",
    )
    id_sig = compute_result_signature(((991,),))
    id_out_slow = ExecutionOutcome(True, ((991,),), ("trans_id",), 1, 9.5, "", id_sig)
    client_out_fast = ExecutionOutcome(
        True, ((991,),), ("client_id",), 1, 0.2, "", id_sig
    )
    self.assertEqual(
        audit_candidate_invariants(cte_star_clean, id_out_slow, intent_159), ()
    )
    # Within-cluster and vs-3-wildcard-votes: `trans_id` wins despite 9.5ms!
    win_159, _, _ = verify_and_rank_candidates(
        (
            star_cand,
            SQLCandidate(star_cand.sql, "skeleton_minimal"),
            SQLCandidate("SELECT client_id FROM disp", "s_fast"),
            cte_star_clean,
        ),
        (star_out, star_out, client_out_fast, id_out_slow),
        intent_159,
    )
    self.assertEqual(win_159.candidate.sql, cte_star_clean.sql)

    # 3. QID 1157 & `866_voice`: 3-col `'their'` query passes cleanly while
    #    1-col SQL on `866_voice` follow-up is flagged so 3-col beats 2x 1-col!
    clean_1157_sql = SQLCandidate(
        "SELECT DISTINCT T1.ID, T1.SEX, T1.Diagnosis FROM Patient AS T1 "
        "INNER JOIN Examination AS T2 ON T1.ID = T2.ID WHERE T2.Thrombosis = 2",
        "decomposed_cot",
    )
    out_1157 = ExecutionOutcome(
        True,
        ((7, "F", "SLE"),),
        ("ID", "SEX", "Diagnosis"),
        1,
        0.5,
        "",
        compute_result_signature(((7, "F", "SLE"),)),
    )
    self.assertEqual(
        audit_candidate_invariants(clean_1157_sql, out_1157, intent_arity3), ()
    )

    intent_866v = canonicalize_intent(
        "uh hey can you please tell me who was the you know player that got "
        "the lap time of 0:01:27 in the race no. 161? show his introduction "
        "website",
        evidence="player and driver are synonyms; introduction website refers to url;",
    )
    c_url_cot = SQLCandidate(
        "SELECT DISTINCT T1.url FROM drivers T1 JOIN lapTimes T2 "
        "ON T1.driverId = T2.driverId WHERE T2.raceId = 161 AND T2.time LIKE '1:27%'",
        "decomposed_cot",
    )
    c_url_skel = SQLCandidate(c_url_cot.sql, "skeleton_minimal")
    c_full_def = SQLCandidate(
        "SELECT DISTINCT T1.forename, T1.surname, T1.url FROM drivers T1 "
        "JOIN lapTimes T2 ON T1.driverId = T2.driverId "
        "WHERE T2.raceId = 161 AND T2.time LIKE '1:27%'",
        "formula_defensive",
    )
    sig_url = compute_result_signature((("http://example.org/driver",),))
    sig_drv = compute_result_signature((("Kimi", "Raikkonen", "http://example.org/driver"),))
    o_url = ExecutionOutcome(
        True, (("http://example.org/driver",),), ("url",), 1, 0.4, "", sig_url
    )
    o_drv = ExecutionOutcome(
        True,
        (("Kimi", "Raikkonen", "http://example.org/driver"),),
        ("forename", "surname", "url"),
        1,
        1.2,
        "",
        sig_drv,
    )
    url_issues = audit_candidate_invariants(c_url_cot, o_url, intent_866v)
    self.assertEqual([i.code for i in url_issues], ["PROJECTION_ARITY_DRIFT"])
    self.assertEqual(audit_candidate_invariants(c_full_def, o_drv, intent_866v), ())
    win_866v, _, _ = verify_and_rank_candidates(
        (c_url_cot, c_url_skel, c_full_def), (o_url, o_url, o_drv), intent_866v
    )
    self.assertEqual(win_866v.outcome.result_signature, sig_drv)

  def test_declarative_context_1col_attribute_tiebreak_and_error_rank_guard(
      self,
  ) -> None:
    """Prevents 2-sentence 1-col drift, `<entity>_id` inversion, & ERROR wins."""
    from argus_sql.verification.types import VerifierConfig

    # 1. Two-sentence `[Declarative context]. [1-col question]` keeps 1-column
    #    `SELECT COUNT(*)` clean so 2 agreeing strategies beat 1 noisy 2-col SQL.
    ctx_intent = canonicalize_intent(
        "Charter schools refer to Charter = 1 in frpm. "
        "What is the total number of charter schools?"
    )
    sig_two = compute_result_signature(((1, 42),))
    sig_one = compute_result_signature(((42,),))
    c_noisy = SQLCandidate(
        "SELECT Charter, COUNT(*) FROM frpm WHERE Charter = 1",
        "skeleton_minimal",
    )
    c_clean1 = SQLCandidate(
        "SELECT COUNT(*) FROM frpm WHERE Charter = 1", "decomposed_cot"
    )
    c_clean2 = SQLCandidate(c_clean1.sql, "formula_defensive")
    o_noisy = ExecutionOutcome(
        True, ((1, 42),), ("Charter", "COUNT(*)"), 1, 0.3, "", sig_two
    )
    o_clean = ExecutionOutcome(
        True, ((42,),), ("COUNT(*)",), 1, 0.4, "", sig_one
    )
    self.assertEqual(audit_candidate_invariants(c_clean1, o_clean, ctx_intent), ())
    ctx_win, _, _ = verify_and_rank_candidates(
        (c_noisy, c_clean1, c_clean2), (o_noisy, o_clean, o_clean), ctx_intent
    )
    self.assertEqual(ctx_win.outcome.result_signature, sig_one)

    # 2. Requested 1-token attribute (`email`, `date`, `amount`) must strictly
    #    outscore 2-token `<entity>_id` (`client_id`, `trans_id`) even when
    #    `"id"` appears in a `WHERE` filter (`with id 3356`) and latency lags.
    sig_target = compute_result_signature((("target_val",),))
    sig_wrong = compute_result_signature((("3356",),))
    for q_text, target_col, wrong_col in (
        ("What is the email of the client with id 3356?", "email", "client_id"),
        (
            "What is the date of the transaction that the client with id 3356 makes?",
            "date",
            "trans_id",
        ),
        ("What is the amount of the transaction with id 3356?", "amount", "trans_id"),
        ("For the client with id 3356, what is the email?", "email", "client_id"),
    ):
      q_intent = canonicalize_intent(q_text)
      o_wrong = ExecutionOutcome(
          True, (("3356",),), (wrong_col,), 1, 0.1, "", sig_wrong
      )
      o_target = ExecutionOutcome(
          True, (("target_val",),), (target_col,), 1, 8.5, "", sig_target
      )
      win_attr, _, _ = verify_and_rank_candidates(
          (
              SQLCandidate(f"SELECT {wrong_col} FROM t", "s_wrong"),
              SQLCandidate(f"SELECT {target_col} FROM t", "s_target"),
          ),
          (o_wrong, o_target),
          q_intent,
      )
      self.assertEqual(win_attr.candidate.strategy, "s_target")

    # 3. Within a cluster containing both an ERROR-free (WARNING-only) member
    #    and an ERROR member, `best_idx` always prefers the non-ERROR member.
    warn_cand = SQLCandidate(
        "SELECT email FROM clients ORDER BY pts ASC LIMIT 1", "s_warn"
    )
    err_cand = SQLCandidate("SELECT * FROM clients", "s_err")
    shared_rows = (("a@b.org", 1, 2),)
    shared_sig = compute_result_signature(shared_rows)
    shared_out = ExecutionOutcome(
        True, shared_rows, ("email", "c2", "c3"), 1, 0.5, "", shared_sig
    )
    # Configure 2 WARNINGs (`0.4 * 0.4 = 0.16`) < 1 ERROR (`0.50`) numerically
    # to prove `_candidate_rank_key` enforces `has_no_error` ahead of weight!
    custom_cfg = VerifierConfig(
        error_issue_multiplier=0.50, warning_issue_multiplier=0.40
    )
    non_err_win, _, _ = verify_and_rank_candidates(
        (err_cand, warn_cand),
        (shared_out, shared_out),
        canonicalize_intent("What is the email of the client?"),
        config=custom_cfg,
    )
    self.assertEqual(non_err_win.candidate.strategy, "s_warn")

  def test_abbreviated_surrogate_name_and_nullable_left_join_preservation(
      self,
  ) -> None:
    """Verifies schema-driven surrogate-name checks and safe LEFT JOIN ratios."""
    out_pct = ExecutionOutcome(
        True, ((12.5,),), ("pct",), 1, 1.0, "", compute_result_signature(((12.5,),))
    )

    # 1. Valid nullable-FK LEFT JOIN CASE/IIF ratios must NOT be rewritten to
    #    INNER JOIN (preserving unmatched parent rows in the denominator).
    enterprise_ddl = (
        "CREATE TABLE orders (\n  order_id INTEGER PRIMARY KEY,\n"
        "  customer_id INTEGER,\n  total REAL\n);\n\n"
        "CREATE TABLE customers (\n  customer_id INTEGER PRIMARY KEY,\n"
        "  tier TEXT\n);"
    )
    linked_orders = LinkedSchemaContext(
        "enterprise_sales", ("orders", "customers"), (), (), (), (), enterprise_ddl
    )
    intent_orders = canonicalize_intent(
        "What percentage of high-value orders were placed by VIP customers?"
    )
    cand_left_case = SQLCandidate(
        "SELECT 100.0 * COUNT(CASE WHEN T2.tier = 'VIP' THEN 1 END) "
        "/ COUNT(T1.order_id) "
        "FROM orders AS T1 LEFT JOIN customers AS T2 "
        "ON T1.customer_id = T2.customer_id WHERE T1.total > 500",
        "decomposed_cot",
    )
    self.assertEqual(
        audit_candidate_invariants(
            cand_left_case, out_pct, intent_orders, linked_orders
        ),
        (),
    )

    # 2. Schema-driven ABBREVIATED_SURROGATE_NAME works across arbitrary
    #    enterprise schemas without hardcoded domain or column vocabularies.
    hr_ddl = (
        "CREATE TABLE emp_summary (\n  emp_id INTEGER PRIMARY KEY,\n"
        "  dname TEXT,\n  salary INTEGER\n);\n\n"
        'CREATE TABLE departments (\n  dept_id INTEGER PRIMARY KEY,\n'
        '  "Department Name" TEXT\n);'
    )
    linked_hr = LinkedSchemaContext(
        "enterprise_hr", ("emp_summary", "departments"), (), (), (), (), hr_ddl
    )
    intent_dept = canonicalize_intent(
        "Give the names of the active departments with high salary."
    )
    c_dept_abbr = SQLCandidate(
        "SELECT T1.dname FROM emp_summary AS T1 INNER JOIN departments AS T2 "
        "ON T1.emp_id = T2.dept_id WHERE T1.salary > 100000",
        "skeleton_minimal",
    )
    c_dept_full = SQLCandidate(
        'SELECT T2."Department Name" FROM emp_summary AS T1 '
        "INNER JOIN departments AS T2 ON T1.emp_id = T2.dept_id "
        "WHERE T1.salary > 100000",
        "decomposed_cot",
    )
    sig_abbr = compute_result_signature((("Eng",),))
    sig_canon = compute_result_signature((("Engineering",),))
    out_abbr = ExecutionOutcome(
        True, (("Eng",),), ("dname",), 1, 0.2, "", sig_abbr
    )
    out_canon = ExecutionOutcome(
        True, (("Engineering",),), ("Department Name",), 1, 2.0, "", sig_canon
    )
    iss_hr = audit_candidate_invariants(c_dept_abbr, out_abbr, intent_dept, linked_hr)
    self.assertEqual([i.code for i in iss_hr], ["ABBREVIATED_SURROGATE_NAME"])
    self.assertEqual(iss_hr[0].severity, "WARNING")
    self.assertEqual(
        audit_candidate_invariants(c_dept_full, out_canon, intent_dept, linked_hr),
        (),
    )
    # 1 clean `"Department Name"` (1.00) outscores 2 `dname` WARNINGs (0.96).
    win_hr, _, _ = verify_and_rank_candidates(
        (c_dept_abbr, c_dept_abbr, c_dept_full),
        (out_abbr, out_abbr, out_canon),
        intent_dept,
        linked_schema=linked_hr,
    )
    self.assertEqual(win_hr.outcome.result_signature, sig_canon)

    # 3. DDL-description-driven abbreviation matching (e.g., `v_code` with
    #    `-- vendor name`) resolves to joined `vendors."Vendor Name"`.
    vendor_ddl = (
        "CREATE TABLE shipments (\n  id INTEGER PRIMARY KEY,\n"
        "  v_code TEXT -- vendor name | examples: ['ACME']\n);\n\n"
        'CREATE TABLE vendors (\n  id INTEGER PRIMARY KEY,\n'
        '  "Vendor Name" TEXT\n);'
    )
    linked_vendor = LinkedSchemaContext(
        "supply_chain", ("shipments", "vendors"), (), (), (), (), vendor_ddl
    )
    intent_vendor = canonicalize_intent("List the vendor names for shipments.")
    c_vendor_abbr = SQLCandidate(
        "SELECT T1.v_code FROM shipments AS T1 INNER JOIN vendors AS T2 "
        "ON T1.id = T2.id",
        "cot",
    )
    from argus_sql.verification.dialect_invariants import find_abbreviated_surrogate_rewrites

    rws_vendor = find_abbreviated_surrogate_rewrites(
        c_vendor_abbr.sql, linked_vendor.ddl_prompt, intent_vendor
    )
    self.assertEqual(len(rws_vendor), 1)
    self.assertEqual(rws_vendor[0][2:], ("T1.v_code", 'T2."Vendor Name"'))

    # 4. Also verifies `sname` -> `"School Name"` via generic prefix rules,
    #    while skipping `surname`, `AS sname` aliases, `GROUP BY`, `COUNT`, and
    #    non-name queries.
    schools_ddl = (
        'CREATE TABLE satscores (\n  cds TEXT PRIMARY KEY,\n  sname TEXT,\n'
        '  surname TEXT,\n  NumGE1500 INTEGER\n);\n\n'
        'CREATE TABLE frpm (\n  CDSCode TEXT PRIMARY KEY,\n'
        '  "School Name" TEXT,\n  "Percent (%) Eligible Free (K-12)" REAL\n);'
    )
    linked_schools = LinkedSchemaContext(
        "california_schools", ("satscores", "frpm"), (), (), (), (), schools_ddl
    )
    intent_school = canonicalize_intent(
        "Give the names of the schools with score >= 1500?"
    )
    c_sname_alias = SQLCandidate(
        'SELECT T1.sname AS sname FROM satscores AS T1 INNER JOIN frpm AS T2 '
        'ON T1.cds = T2.CDSCode',
        "formula_defensive",
    )
    rws_school = find_abbreviated_surrogate_rewrites(
        c_sname_alias.sql, linked_schools.ddl_prompt, intent_school
    )
    self.assertEqual(len(rws_school), 1)
    self.assertEqual(rws_school[0][2:], ("T1.sname", 'T2."School Name"'))

    c_surname = SQLCandidate(
        "SELECT T1.surname FROM satscores AS T1 INNER JOIN frpm AS T2 "
        "ON T1.cds = T2.CDSCode",
        "cot",
    )
    self.assertEqual(
        audit_candidate_invariants(c_surname, out_abbr, intent_school, linked_schools),
        (),
    )
    intent_no_name = canonicalize_intent("What are the CDS codes of the schools?")
    self.assertEqual(
        audit_candidate_invariants(c_sname_alias, out_abbr, intent_no_name, linked_schools),
        (),
    )
    c_gb_sname = SQLCandidate(
        "SELECT T1.sname FROM satscores AS T1 INNER JOIN frpm AS T2 "
        "ON T1.cds = T2.CDSCode GROUP BY T1.sname",
        "cot",
    )
    self.assertEqual(
        audit_candidate_invariants(c_gb_sname, out_abbr, intent_school, linked_schools),
        (),
    )
    c_fn_sname = SQLCandidate(
        "SELECT COUNT(T1.sname) FROM satscores AS T1 INNER JOIN frpm AS T2 "
        "ON T1.cds = T2.CDSCode",
        "cot",
    )
    self.assertEqual(
        audit_candidate_invariants(c_fn_sname, out_abbr, intent_school, linked_schools),
        (),
    )



if __name__ == "__main__":
  unittest.main()
