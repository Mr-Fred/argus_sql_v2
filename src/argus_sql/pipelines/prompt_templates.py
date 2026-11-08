"""Modular prompt templates for baseline and Argus-SQL synthesis strategies.

Provides clean, vendor-neutral SQLite system and user prompt builders for
zero-shot DDL, decomposed Chain-of-Thought, formula-defensive SQL, and
skeleton-minimal synthesis strategies.
"""

from argus_sql.adapters.types import LLMCompletionRequest

# Shared system instruction enforcing single-statement read-only SQLite syntax.
SQLITE_SYSTEM_PROMPT = (
    "You are an expert relational database engineer specializing in SQLite. "
    "Given a CREATE TABLE schema, optional External Evidence formulas, and a "
    "user Question, output a single valid read-only SQLite SELECT statement "
    "inside a ```sql ... ``` code block."
)


def build_strategy_request(
    request_id: str,
    ddl_prompt: str,
    question: str,
    evidence: str,
    strategy_tag: str,
    temperature: float = 0.0,
) -> LLMCompletionRequest:
  """Constructs an LLMCompletionRequest tailored to `strategy_tag`.

  Args:
    request_id: Unique correlation identifier for batch tracking.
    ddl_prompt: Rendered `CREATE TABLE` DDL block (full or pruned).
    question: Raw or canonicalized user natural language question.
    evidence: Optional external domain knowledge / formula note.
    strategy_tag: Synthesis strategy ('zero_shot', 'decomposed_cot',
      'formula_defensive', or 'skeleton_minimal').
    temperature: Sampling temperature (`0.0` for greedy decoding).

  Returns:
    Immutable LLMCompletionRequest ready for `BatchLLMPort.batch_complete`.
  """
  ev_block = f"\nExternal Evidence: {evidence}" if evidence.strip() else ""

  # Tailor reasoning instructions to encourage genuine candidate diversity.
  if strategy_tag == "decomposed_cot":
    # Emphasizes multi-hop foreign-key path planning (DIN-SQL style).
    guidance = (
        "Decompose step-by-step: (1) identify required tables and FK join "
        "paths, (2) verify exact string literals from matched_values, "
        "(3) write the final SQLite SELECT query."
    )
  elif strategy_tag == "formula_defensive":
    # Emphasizes explicit SQLite REAL division and NULL-sort defenses.
    guidance = (
        "Enforce SQLite defensive invariants: promote ratios using "
        "CAST(numerator AS REAL) / NULLIF(denominator, 0), filter NULLs "
        "with IS NOT NULL before ORDER BY ... ASC LIMIT, and bind exact "
        "matched_values literals."
    )
  elif strategy_tag == "skeleton_minimal":
    # Emphasizes projection arity discipline and minimal relational clauses.
    guidance = (
        "Draft the minimal relational SQL skeleton (SELECT ... FROM ... JOIN "
        "... WHERE ...) projecting only the exact columns requested, then "
        "fill column identifiers from the DDL."
    )
  else:
    # Default zero-shot directive keeps prompt overhead minimal for baselines.
    guidance = "Write the SQLite SELECT query answering the Question directly."

  # Concatenate DDL, user question, optional evidence, and strategy instruction.
  user_prompt = (
      f"Database Schema:\n{ddl_prompt}\n\n"
      f"Question: {question}{ev_block}\n\n"
      f"Instruction: {guidance}"
  )
  # Return frozen request contract for deduplicated batch LLM execution.
  return LLMCompletionRequest(
      request_id=request_id,
      system_prompt=SQLITE_SYSTEM_PROMPT,
      user_prompt=user_prompt,
      strategy_tag=strategy_tag,
      temperature=temperature,
  )
