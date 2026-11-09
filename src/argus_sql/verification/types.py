"""Configuration and equivalence-cluster contracts for consensus arbitration."""

import dataclasses


# Tunable weights governing Argus invariant-weighted execution consensus.
@dataclasses.dataclass(frozen=True)
class VerifierConfig:
  """Configuration controlling semantic verification and consensus scoring.

  Attributes:
    enable_semantic_invariants: If False, falls back to syntax/execution only.
    enable_algebraic_repair: If True, runs zero-token dialect auto-repair.
    error_issue_multiplier: Weight penalty factor per ERROR-level invariant.
    warning_issue_multiplier: Weight penalty factor per WARNING-level invariant.
    strategy_diversity_bonus: Bonus added per distinct agreeing strategy.
  """

  # Multiplicative penalty `0.18` ensures 1 clean repair (1.00) beats 3-4
  # correlated empty-set or integer-truncated candidates (3 * 0.18 = 0.54).
  enable_semantic_invariants: bool = True
  enable_algebraic_repair: bool = True
  error_issue_multiplier: float = 0.18
  # `0.48` ensures 1 invariant-clean candidate (1.00) beats 2 WARNINGs (0.96).
  warning_issue_multiplier: float = 0.48
  # Small diversity bonus (`0.15`) breaks ties across issue-free strategies.
  strategy_diversity_bonus: float = 0.15


# Equivalence cluster grouping candidates that produce identical result sets.
@dataclasses.dataclass(frozen=True)
class ConsensusCluster:
  """Equivalence class of SQL candidates sharing an execution result signature.

  Attributes:
    result_signature: Canonical SHA-256 digest of the shared result tuples.
    member_indices: Indices into the evaluated `VerifiedCandidate` sequence.
    distinct_strategies: Unique synthesis strategy names inside this cluster.
    total_score: Summed invariant-adjusted weight plus diversity bonus.
  """

  result_signature: str
  member_indices: tuple[int, ...]
  distinct_strategies: tuple[str, ...]
  total_score: float
