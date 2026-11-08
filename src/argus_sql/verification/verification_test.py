"""Unit tests for semantic invariants, algebraic repair, and consensus voting."""

import unittest

from argus_sql.contracts.types import ExecutionOutcome
from argus_sql.contracts.types import SQLCandidate
from argus_sql.linking.intent_canonicalizer import canonicalize_intent
from argus_sql.sandbox.sqlite_sandbox import compute_result_signature
from argus_sql.verification.consensus_arbiter import verify_and_rank_candidates
from argus_sql.verification.exceptions import VerificationError
from argus_sql.verification.semantic_verifier import audit_candidate_invariants
from argus_sql.verification.sql_repairer import apply_algebraic_sql_repairs


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


if __name__ == "__main__":
  unittest.main()
