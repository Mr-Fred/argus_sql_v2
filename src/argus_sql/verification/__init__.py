"""Public API facade for Argus-SQL semantic verification and consensus."""

# Re-export consensus arbiter and candidate weighting helper.
from argus_sql.verification.consensus_arbiter import compute_candidate_weight
from argus_sql.verification.consensus_arbiter import verify_and_rank_candidates
# Re-export verification domain exception.
from argus_sql.verification.exceptions import VerificationError
# Re-export semantic and SQLite dialect invariant verifier.
from argus_sql.verification.semantic_verifier import audit_candidate_invariants
# Re-export deterministic dialect repairer and LLM diagnostic prompt builder.
from argus_sql.verification.sql_repairer import apply_algebraic_sql_repairs
from argus_sql.verification.sql_repairer import build_diagnostic_repair_prompt
# Re-export verifier config and cluster structures.
from argus_sql.verification.types import ConsensusCluster
from argus_sql.verification.types import VerifierConfig

# Explicit symbol export list.
__all__ = [
    "ConsensusCluster",
    "VerificationError",
    "VerifierConfig",
    "apply_algebraic_sql_repairs",
    "audit_candidate_invariants",
    "build_diagnostic_repair_prompt",
    "compute_candidate_weight",
    "verify_and_rank_candidates",
]
