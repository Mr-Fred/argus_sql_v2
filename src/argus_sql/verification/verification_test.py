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


if __name__ == "__main__":
  unittest.main()
