"""Invariant-weighted execution consensus arbiter for SQL candidate selection.

Replaces naive majority self-consistency voting—which fails whenever a majority
of samples share integer-truncation or NULL-ordering bugs—with equivalence
clustering weighted by semantic invariant cleanliness and strategy diversity.
"""

from collections.abc import Sequence
import re

from argus_sql.contracts.types import ExecutionOutcome
from argus_sql.contracts.types import IntentFrame
from argus_sql.contracts.types import LinkedSchemaContext
from argus_sql.contracts.types import SQLCandidate
from argus_sql.contracts.types import VerifiedCandidate
from argus_sql.verification.exceptions import VerificationError
from argus_sql.verification.semantic_verifier import audit_candidate_invariants
from argus_sql.verification.types import ConsensusCluster
from argus_sql.verification.types import VerifierConfig

# Splits derived repair suffixes (`+argus_repair`, `->repair`, `:repair`, `/`).
_REPAIR_DELIM_RE = re.compile(r"\+|->|:|/")

# Strips underscore repair suffixes (`_argus_repair`, `_llm_repair`, `_repair`).
_UNDERSCORE_REPAIR_RE = re.compile(
    r"_(?:argus_)?(?:alg_|llm_)?repair$",
    re.IGNORECASE,
)

# SQL expression keywords excluded when scoring projected column-name overlap.
_SQL_PROJ_NOISE: frozenset[str] = frozenset({
    "as", "asc", "case", "cast", "coalesce", "desc", "distinct", "double",
    "else", "end", "float", "iif", "int", "integer", "null", "nullif",
    "over", "partition", "real", "round", "select", "strftime", "substr",
    "text", "then", "when",
})

# Question function words excluded from projection-name alignment tokens.
_QUESTION_STOPWORDS: frozenset[str] = frozenset({
    "a", "all", "among", "an", "and", "are", "as", "at", "be", "between",
    "both", "by", "can", "could", "did", "do", "does", "each", "for",
    "from", "give", "has", "have", "how", "in", "indicate", "is", "it",
    "its", "list", "of", "on", "or", "please", "provide", "show", "state",
    "tell", "that", "the", "their", "them", "these", "they", "this",
    "those", "to", "was", "were", "what", "when", "where", "which", "who",
    "whose", "with",
})

# Multi-character `*name` compounds mapped onto `name` (excluding `sname`).
_NAME_COMPOUNDS: frozenset[str] = frozenset({
    "displayname", "firstname", "forename", "fullname",
    "lastname", "schoolname", "surname", "username",
})


def _origin_strategy(strategy: str) -> str:
  """Extracts the originating synthesis strategy before derived repair tags."""
  # Collapse `skeleton_minimal+argus_repair` (or `_repair`) onto parent lineage
  # while keeping standalone Stage 4b `argus_llm_repair` intact.
  head = _REPAIR_DELIM_RE.split(strategy.strip(), maxsplit=1)[0].strip()
  if head.lower() not in {"argus_llm_repair", "repair"}:
    head = _UNDERSCORE_REPAIR_RE.sub("", head).strip()
  return head or strategy.strip()


# Identifies interrogative or imperative sentences carrying projection targets,
# anchored at clause heads so declarative `state`/`where` context is skipped.
_PROJECTION_SENTENCE_CUE_RE = re.compile(
    r"(?:^(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?(?:please\s+)?"
    r"(?:what|which|who|whom|whose|where|when|how|"
    r"list|give|show|provide|indicate|include|identify|find|state|"
    r"report|display|name|calculate|compute|count|rank|tell|return)\b"
    r"|,\s*(?:please\s+)?"
    r"(?:what\s+(?:is|are|was|were)|how\s+(?:many|much)|"
    r"list|give|show|provide|indicate|include|identify|find|state|"
    r"report|display|name|calculate|compute|count|rank|tell|return)\b)",
    re.IGNORECASE,
)


def _tokenize_projection_terms(text: str, is_column: bool) -> set[str]:
  """Tokenizes and stems column headers or question clauses for tie-breaking."""
  # Strip leading `table.` qualifier on plain column headers (`schools.sname`),
  # or possessive `'s` clitics on natural-language question clauses.
  if is_column:
    target = text.rsplit(".", 1)[-1] if ("(" not in text and "." in text) else text
  else:
    target = re.sub(r"'s\b", "", text, flags=re.IGNORECASE)

  # Split camelCase (`SchoolName` -> `School Name`, `CDSCode` -> `CDS Code`).
  spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", target)
  spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
  tokens: set[str] = set()
  for raw in re.findall(r"[a-zA-Z0-9]+", spaced):
    tok = raw.lower()
    if tok.isdigit():
      continue
    if is_column:
      if tok in _SQL_PROJ_NOISE or re.fullmatch(r"t\d+", tok) is not None:
        continue
    elif tok in _QUESTION_STOPWORDS:
      continue

    # Apply lightweight plural singularization (`schools` -> `school`) while
    # canonicalizing `transaction`/`transactions` onto `trans`.
    if tok.startswith("transact"):
      stem = "trans"
    elif len(tok) > 4 and tok.endswith("ies"):
      stem = tok[:-3] + "y"
    elif len(tok) > 4 and tok.endswith("oes"):
      stem = tok[:-2]
    elif (
        len(tok) > 3
        and tok != "trans"
        and tok.endswith("s")
        and not tok.endswith(("ss", "us", "is"))
    ):
      stem = tok[:-1]
    else:
      stem = tok
    tokens.add(stem)
    if stem in _NAME_COMPOUNDS:
      tokens.add("name")

  if not is_column and "how many" in text.lower():
    tokens.add("count")
  return tokens


def _projection_alignment_score(
    outcome: ExecutionOutcome,
    intent: IntentFrame,
) -> float:
  """Scores lexical alignment between projected column headers and user intent."""
  if not outcome.succeeded or not outcome.column_names:
    return 0.0
  col_tokens: set[str] = set()
  for col in outcome.column_names:
    col_tokens.update(_tokenize_projection_terms(str(col), is_column=True))
  if not col_tokens:
    return 0.0

  q_text = intent.canonical_question
  q_tokens = _tokenize_projection_terms(q_text, is_column=False)
  shared = col_tokens & q_tokens
  if not shared:
    return 0.0

  # Isolate projection-focus and requested-attribute tokens ahead of leading
  # `For ...,` filters, `of <entity>` possessors, and `with/where/that` clauses
  # so `<entity>_id` from `with id 3356` never outscores requested `<attr>`.
  sentences = [s.strip() for s in re.split(r"[.;?]+", q_text) if s.strip()]
  active_sents = [
      s for s in sentences if _PROJECTION_SENTENCE_CUE_RE.search(s)
  ] or sentences
  focus_tokens: set[str] = set()
  attr_tokens: set[str] = set()
  for sent in active_sents:
    s_main = re.sub(
        r"^(?:for|among|in|on|at|from|between|under|given|considering)\b"
        r"[^,]{1,90},\s*",
        "",
        sent,
        flags=re.IGNORECASE,
    )
    head = re.split(
        r"\b(?:that|where|who|whose|with|having|when|if)\b",
        s_main,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    head_toks = _tokenize_projection_terms(head, is_column=False)
    focus_tokens.update(head_toks)
    attr_head = re.split(
        r"\b(?:of|for|from|among)\b",
        head,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    attr_head = re.sub(
        r"\b[a-zA-Z0-9_]+'s\s+", "", attr_head, flags=re.IGNORECASE
    )
    cand_attr = _tokenize_projection_terms(attr_head, is_column=False)
    attr_tokens.update(cand_attr if cand_attr else head_toks)

  focus_shared = col_tokens & focus_tokens
  attr_shared = col_tokens & attr_tokens
  score = (
      0.65 * (len(focus_shared) / len(col_tokens))
      + 0.20 * (1.0 if attr_shared else 0.0)
      + 0.15 * (len(shared) / len(col_tokens))
      + 0.04 * min(len(focus_shared), 4)
      + 0.02 * min(len(attr_shared), 4)
  )
  return round(score, 6)


def _candidate_rank_key(
    ver: VerifiedCandidate,
    proj_score: float,
) -> tuple[bool, float, int, float, float]:
  """Orders candidates by error-freeness, weight, issue count, header, speed."""
  has_no_error = not any(iss.severity == "ERROR" for iss in ver.issues)
  return (
      has_no_error,
      ver.consensus_weight,
      -len(ver.issues),
      proj_score,
      -ver.outcome.execution_time_ms,
  )


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

  # Precompute projection-name alignment scores when semantic verification is on.
  proj_scores = [
      _projection_alignment_score(ver.outcome, intent)
      if cfg.enable_semantic_invariants
      else 0.0
      for ver in verified_list
  ]

  # 3. Score each cluster while deduplicating support per originating strategy.
  # Collapsing derived repair copies prevents one strategy (`s` + `s+argus_repair`)
  # from outvoting distinct clean strategies via duplicated lineage votes.
  clusters: list[ConsensusCluster] = []
  cluster_proj_scores: dict[str, float] = {}
  for sig, indices in cluster_members.items():
    best_by_origin: dict[str, int] = {}
    for idx in indices:
      origin = _origin_strategy(verified_list[idx].candidate.strategy)
      prev_idx = best_by_origin.get(origin)
      cand_rank = _candidate_rank_key(verified_list[idx], proj_scores[idx])
      if prev_idx is None or cand_rank > _candidate_rank_key(
          verified_list[prev_idx], proj_scores[prev_idx]
      ):
        best_by_origin[origin] = idx

    weight_sum = sum(
        verified_list[idx].consensus_weight for idx in best_by_origin.values()
    )
    strategies = tuple(sorted(best_by_origin.keys()))
    # Award diversity credit strictly to invariant-clean distinct origin lineages.
    clean_strategies = {
        orig
        for orig, idx in best_by_origin.items()
        if not verified_list[idx].issues
    }
    diversity = cfg.strategy_diversity_bonus * max(0, len(clean_strategies) - 1)
    top_qual = max(
        _candidate_rank_key(verified_list[i], proj_scores[i])[:3]
        for i in indices
    )
    cluster_proj_scores[sig] = max(
        proj_scores[i]
        for i in indices
        if _candidate_rank_key(verified_list[i], proj_scores[i])[:3] == top_qual
    )
    clusters.append(
        ConsensusCluster(
            result_signature=sig,
            member_indices=tuple(indices),
            distinct_strategies=strategies,
            total_score=round(weight_sum + diversity, 6),
        )
    )

  # Rank error-free clusters ahead of ERROR-only clusters, then by total score,
  # then by projection-name alignment ahead of fastest sandbox execution time.
  ranked_clusters = tuple(
      sorted(
          clusters,
          key=lambda c: (
              any(
                  not any(iss.severity == "ERROR" for iss in verified_list[i].issues)
                  for i in c.member_indices
              ),
              c.total_score,
              cluster_proj_scores[c.result_signature],
              -min(verified_list[i].outcome.execution_time_ms for i in c.member_indices),
          ),
          reverse=True,
      )
  )

  # Within the winning cluster, break ties by error-freeness, weight, issue
  # count, and projection-name alignment ahead of raw execution latency.
  top_cluster = ranked_clusters[0]
  best_idx = max(
      top_cluster.member_indices,
      key=lambda i: _candidate_rank_key(verified_list[i], proj_scores[i]),
  )
  return (verified_list[best_idx], tuple(verified_list), ranked_clusters)
