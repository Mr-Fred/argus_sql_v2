"""Invariant-weighted execution consensus arbiter for SQL candidate selection.

Replaces naive majority self-consistency voting—which fails whenever a majority
of samples share integer-truncation or NULL-ordering bugs—with equivalence
clustering weighted by semantic invariant cleanliness and strategy diversity.
"""

from collections.abc import Sequence

from argus_sql.contracts.types import ExecutionOutcome
from argus_sql.contracts.types import IntentFrame
from argus_sql.contracts.types import LinkedSchemaContext
from argus_sql.contracts.types import SQLCandidate
from argus_sql.contracts.types import VerifiedCandidate
from argus_sql.verification.exceptions import VerificationError
from argus_sql.verification.semantic_verifier import audit_candidate_invariants
from argus_sql.verification.types import ConsensusCluster
from argus_sql.verification.types import VerifierConfig


def compute_candidate_weight(
    outcome: ExecutionOutcome,
    issues_severities: Sequence[str],
    config: VerifierConfig,
) -> float:
  """Computes quality weight in `[0.0, 1.0]` for one executed SQL candidate.

  Args:
    outcome: SQLite execution outcome for the candidate.
    issues_severities: Sequence of issue severity strings ('ERROR', 'WARNING').
    config: Active VerifierConfig weighting penalties.

  Returns:
    Non-negative consensus weight assigned to this candidate.
  """
  # Non-executing SQL receives zero consensus weight.
  if not outcome.succeeded:
    return 0.0

  # Decay candidate weight multiplicatively per semantic invariant violation.
  weight = 1.0
  for sev in issues_severities:
    if sev == "ERROR":
      weight *= config.error_issue_multiplier
    elif sev == "WARNING":
      weight *= config.warning_issue_multiplier
  # Round to 6 decimals so cross-platform floating-point tie-breaks stay exact.
  return round(weight, 6)


def verify_and_rank_candidates(
    candidates: Sequence[SQLCandidate],
    outcomes: Sequence[ExecutionOutcome],
    intent: IntentFrame,
    linked_schema: LinkedSchemaContext | None = None,
    config: VerifierConfig | None = None,
    repaired_flags: Sequence[bool] | None = None,
) -> tuple[VerifiedCandidate, tuple[VerifiedCandidate, ...], tuple[ConsensusCluster, ...]]:
  """Verifies candidates, builds equivalence clusters, and picks the winner.

  Args:
    candidates: Sequence of SQLCandidate items evaluated.
    outcomes: Sequence of ExecutionOutcome items aligned with `candidates`.
    intent: Canonicalized IntentFrame for semantic invariant checks.
    linked_schema: Optional LinkedSchemaContext with grounded values.
    config: Optional VerifierConfig controlling invariant penalties.
    repaired_flags: Optional boolean flags marking repaired candidates.

  Returns:
    Tuple `(winning_candidate, all_verified_candidates, ranked_clusters)`.

  Raises:
    VerificationError: If `candidates` is empty or lengths mismatch.
  """
  # Fail fast with a domain VerificationError on empty or misaligned inputs.
  if not candidates or len(candidates) != len(outcomes):
    raise VerificationError(
        "Candidates must be non-empty and match outcomes 1-to-1."
    )
  if repaired_flags is not None and len(repaired_flags) != len(candidates):
    raise VerificationError(
        "repaired_flags length must match candidates 1-to-1."
    )

  cfg = config or VerifierConfig()
  # Default `repaired_flags` to False when evaluating un-repaired candidate sets.
  rep_flags = (
      tuple(repaired_flags)
      if repaired_flags is not None
      else (False,) * len(candidates)
  )

  # Track verified candidate wrappers and equivalence buckets keyed by SHA-256.
  verified_list: list[VerifiedCandidate] = []
  cluster_members: dict[str, list[int]] = {}

  # 1. Audit invariants and compute per-candidate consensus weights.
  for idx, (cand, out, is_rep) in enumerate(
      zip(candidates, outcomes, rep_flags, strict=True)
  ):
    issues = audit_candidate_invariants(
        candidate=cand,
        outcome=out,
        intent=intent,
        linked_schema=linked_schema,
        enable_semantic_checks=cfg.enable_semantic_invariants,
    )
    weight = compute_candidate_weight(
        outcome=out,
        issues_severities=[iss.severity for iss in issues],
        config=cfg,
    )
    verified = VerifiedCandidate(
        candidate=cand,
        outcome=out,
        issues=issues,
        consensus_weight=weight,
        was_repaired=is_rep,
    )
    verified_list.append(verified)
    if out.succeeded and out.result_signature:
      cluster_members.setdefault(out.result_signature, []).append(idx)

  # 2. If zero candidates executed cleanly, return the first candidate for trace.
  if not cluster_members:
    return (verified_list[0], tuple(verified_list), ())

  # 3. Score each execution equivalence cluster with diversity bonus.
  clusters: list[ConsensusCluster] = []
  for sig, indices in cluster_members.items():
    weight_sum = sum(verified_list[i].consensus_weight for i in indices)
    strategies = tuple(
        sorted({verified_list[i].candidate.strategy for i in indices})
    )
    diversity = cfg.strategy_diversity_bonus * max(0, len(strategies) - 1)
    clusters.append(
        ConsensusCluster(
            result_signature=sig,
            member_indices=tuple(indices),
            distinct_strategies=strategies,
            total_score=round(weight_sum + diversity, 6),
        )
    )

  # Rank clusters by total consensus score, breaking ties by fastest execution.
  ranked_clusters = tuple(
      sorted(
          clusters,
          key=lambda c: (
              c.total_score,
              -min(verified_list[i].outcome.execution_time_ms for i in c.member_indices),
          ),
          reverse=True,
      )
  )

  # Within the winning cluster, pick the cleanest, fastest SQL representative.
  # Preferring lower execution_time_ms directly maximizes BIRD R-VES efficiency.
  top_cluster = ranked_clusters[0]
  best_idx = max(
      top_cluster.member_indices,
      key=lambda i: (
          verified_list[i].consensus_weight,
          -len(verified_list[i].issues),
          -verified_list[i].outcome.execution_time_ms,
      ),
  )
  return (verified_list[best_idx], tuple(verified_list), ranked_clusters)
