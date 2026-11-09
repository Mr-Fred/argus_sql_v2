"""Modular prompt templates for baseline and Argus-SQL synthesis strategies.

Provides clean, vendor-neutral SQLite system and user prompt builders for
zero-shot DDL, decomposed Chain-of-Thought, formula-defensive SQL, and
skeleton-minimal synthesis strategies.
"""

from argus_sql.adapters.types import LLMCompletionRequest

# Shared system instruction enforcing single-statement read-only SQLite syntax.
SQLITE_SYSTEM_PROMPT = (
    "You are an expert relational database engineer specializing in SQLite. "
    "Given a CREATE TABLE schema, optional External Evidence, and a user "
    "Question, output a single valid read-only SQLite SELECT statement "
    "inside a ```sql ... ``` code block. Follow External Evidence mappings, "
    "ratios, and string-slicing rules faithfully (if Evidence writes boolean "
    "shorthand like `AVG(col = 'x')` for an 'average number of X per entity' "
    "count question, compute `AVG(cnt)` over a `GROUP BY entity_id` count "
    "subquery instead of a 0/1 row proportion). For conditional percentages "
    "or ratios across joined tables, use `INNER JOIN` with `SUM(CASE WHEN "
    "cond THEN 1 ELSE 0 END)` rather than filtering inside `LEFT JOIN ... "
    "ON ... AND cond`. Group joined entities by their unique ID/FK key "
    "column rather than display names, aggregate with `SUM(...) + GROUP BY "
    "period` for period totals over composite PRIMARY KEY (entity, period) "
    "tables, and check column `range:`/`matched_values`/`examples` bounds "
    "before filtering."
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
    # Emphasizes FK join paths, entity-ID grouping, count-then-avg, & ranges.
    guidance = (
        "Decompose step-by-step: (1) trace FK join paths and entity/table "
        "grain—group joined entities by their single unique ID/FK key column, "
        "use GROUP BY period + SUM(...) for composite PRIMARY KEY (entity, "
        "period) totals, and for 'average number of X per entity' take AVG(cnt) "
        "over a GROUP BY entity_id count subquery; (2) verify column range:, "
        "matched_values, and examples so WHERE filters target the table "
        "covering the value; (3) compute conditional ratios across joined "
        "tables via INNER JOIN and CASE WHEN (do not push numerator filters "
        "into LEFT JOIN ... ON ... AND)."
    )
  elif strategy_tag == "formula_defensive":
    # Emphasizes explicit SQLite REAL division, INNER JOIN ratios, & NULL-sort.
    guidance = (
        "Enforce SQLite defensive & relational invariants: promote ratios using "
        "CAST(numerator AS REAL) / NULLIF(denominator, 0), compute conditional "
        "percentages across joins via INNER JOIN + SUM(CASE WHEN cond THEN 1 "
        "ELSE 0 END) (never filter numerator conditions inside LEFT JOIN ... ON), "
        "group joined entities by their single ID/FK key rather than text names, "
        "filter NULLs with IS NOT NULL before ORDER BY ... ASC LIMIT, use "
        "SELECT DISTINCT on duplicate-prone 1-to-many entity listings, and "
        "verify column range:/matched_values/examples bounds."
    )
  elif strategy_tag == "skeleton_minimal":
    # Direct relational synthesis anchor avoiding over-engineered join filters.
    guidance = (
        "Write the most direct relational SQLite SELECT query projecting only "
        "the exact columns requested: join via INNER JOIN + CASE WHEN for "
        "conditional ratios, group ranked entities by their single ID/FK key "
        "column rather than display names, check composite PRIMARY KEY grain "
        "and column range:/matched_values bounds, and apply External Evidence "
        "directly."
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
