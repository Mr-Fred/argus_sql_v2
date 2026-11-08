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
    "inside a ```sql ... ``` code block. Follow External Evidence equations "
    "and string-slicing rules verbatim, check PRIMARY KEY grain before "
    "aggregating, and inspect column examples/matched_values for exact formats."
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
    # Emphasizes multi-hop foreign-key path & PRIMARY KEY grain planning.
    guidance = (
        "Decompose step-by-step: (1) identify required tables, FK join paths, "
        "and composite PRIMARY KEY table grain (use GROUP BY + SUM when rows "
        "are per-entity-period), (2) verify exact date/code formats from "
        "matched_values and examples, (3) translate External Evidence formulas "
        "and SUBSTR rules literally into the final SQLite SELECT query."
    )
  elif strategy_tag == "formula_defensive":
    # Emphasizes explicit SQLite REAL division, NULL-sort, and DISTINCT guards.
    guidance = (
        "Enforce SQLite defensive invariants: promote ratios using "
        "CAST(numerator AS REAL) / NULLIF(denominator, 0), filter NULLs "
        "with IS NOT NULL before ORDER BY ... ASC LIMIT, follow External "
        "Evidence formulas/SUBSTR slices verbatim, use SELECT DISTINCT when "
        "listing unique entities across 1-to-many joins, and bind exact "
        "matched_values/examples literals."
    )
  elif strategy_tag == "skeleton_minimal":
    # Emphasizes projection arity discipline and direct evidence translation.
    guidance = (
        "Draft the minimal relational SQL skeleton (SELECT ... FROM ... JOIN "
        "... WHERE ...) projecting only the exact columns requested, apply "
        "External Evidence formulas directly without extra nested CTEs unless "
        "required, and fill exact column names and literals from the DDL."
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
