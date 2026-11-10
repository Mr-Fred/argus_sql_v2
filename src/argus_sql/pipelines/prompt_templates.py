"""Modular prompt templates for baseline and Argus-SQL synthesis strategies.

Provides clean, vendor-neutral SQLite system and user prompt builders for
zero-shot DDL, decomposed Chain-of-Thought, formula-defensive SQL, and
skeleton-minimal synthesis strategies.
"""

from argus_sql.adapters.types import LLMCompletionRequest

# Shared system instruction enforcing single-statement read-only SQLite syntax.
# Encodes core SQLite relational, projection, nullable-FK, temporal, & rank rules.
SQLITE_SYSTEM_PROMPT = (
    "You are an expert relational database engineer specializing in SQLite. "
    "Given a CREATE TABLE schema, optional External Evidence, and a user "
    "Question, output a single valid read-only SQLite SELECT statement "
    "inside a ```sql ... ``` code block. Never use `SELECT *` or `table.*` "
    "wildcard projections—project only the exact column(s) requested by the "
    "Question. Follow External Evidence mappings, ratios, and string-slicing "
    "rules faithfully (if Evidence writes boolean shorthand like "
    "`AVG(col = 'x')` for an 'average number of X per entity' count question, "
    "compute `AVG(cnt)` over a `GROUP BY entity_id` count subquery instead of "
    "a 0/1 row proportion). Preserve `LEFT JOIN` (with only the foreign-key "
    "equality in `ON` and conditional filters inside `SUM(CASE WHEN cond THEN "
    "1 ELSE 0 END)`) on nullable foreign keys when counting or aggregating "
    "parent entities so parent rows with NULL foreign keys are never dropped "
    "by `INNER JOIN`; use `INNER JOIN` with `CASE WHEN` when aggregating "
    "required 1-to-many child rows, and never push numerator filters into "
    "`LEFT JOIN ... ON ... AND cond`. Group joined entities by their unique "
    "ID/FK key column rather than display names, avoid joining an extra "
    "entity-lookup table solely for a name column when the filtered tables "
    "already carry that name column, wrap sub-day timestamp columns (`range: "
    "'YYYY-MM-DD HH:MM...'`) in `DATE(col)` when comparing against "
    "`'YYYY-MM-DD'` date literals, project `(entity, metric, RANK() OVER "
    "(ORDER BY metric ...))` on imperative `'Rank X by Y'` queries, aggregate "
    "with `SUM(...) + GROUP BY period` for period totals over composite "
    "PRIMARY KEY (entity, period) tables, and check column "
    "`range:`/`matched_values`/`examples` bounds before filtering."
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
  # Omit the `External Evidence:` line altogether when no domain hint exists.
  ev_block = f"\nExternal Evidence: {evidence}" if evidence.strip() else ""

  # Tailor reasoning instructions to encourage genuine candidate diversity
  # across our K=3 batch synthesis portfolio while enforcing core invariants.
  # Note: keep literal `CAST(` and `IS NOT NULL` confined to `formula_defensive`
  # so hermetic diagnostic synthesis preserves non-defensive diversity!
  if strategy_tag == "decomposed_cot":
    # Emphasizes FK join grain, nullable-FK LEFT JOIN preservation, & no `*`.
    guidance = (
        "Decompose step-by-step: (1) trace FK join paths and entity/table "
        "grain—never use SELECT * or table.* (project only the exact "
        "requested columns), join only tables needed to answer the question "
        "(do not join an extra lookup table if the filtered tables already "
        "hold the requested name column), preserve LEFT JOIN on nullable "
        "foreign keys when counting or aggregating parent entities so parent "
        "rows with NULL foreign keys are not lost to INNER JOIN, group joined "
        "entities by their single unique ID/FK key column, use GROUP BY "
        "period + SUM(...) for composite PRIMARY KEY (entity, period) totals, "
        "and for 'average number of X per entity' take AVG(cnt) over a GROUP "
        "BY entity_id count subquery; (2) verify column range:, "
        "matched_values, and examples—wrapping sub-day timestamp columns in "
        "DATE(col) when comparing to 'YYYY-MM-DD' literals; (3) place "
        "conditional ratio numerators inside CASE WHEN (using LEFT JOIN on "
        "nullable parent-to-lookup FKs vs INNER JOIN on required child rows, "
        "and never pushing numerator filters into LEFT JOIN ... ON ... AND), "
        "and on imperative 'Rank X by Y' queries project (entity, metric, "
        "RANK() OVER (ORDER BY metric ...))."
    )
  elif strategy_tag == "formula_defensive":
    # Emphasizes SQLite REAL division, nullable-FK LEFT JOINs, & no `SELECT *`.
    guidance = (
        "Enforce SQLite defensive & relational invariants: never use SELECT * "
        "or table.* projections (project only the exact requested columns), "
        "promote ratios using CAST(numerator AS REAL) / NULLIF(denominator, 0), "
        "preserve LEFT JOIN on nullable foreign keys when counting or "
        "aggregating parent entities with conditional numerators in "
        "SUM(CASE WHEN cond THEN 1 ELSE 0 END) so NULL-FK parent rows are "
        "retained (use INNER JOIN on required child rows and never filter "
        "numerator conditions inside LEFT JOIN ... ON), group joined entities "
        "by their single ID/FK key rather than text names, wrap sub-day "
        "timestamp columns in DATE(col) on 'YYYY-MM-DD' predicates, emit "
        "RANK() OVER (ORDER BY metric ...) on imperative 'Rank X by Y' "
        "queries, filter NULLs with IS NOT NULL before ORDER BY ... ASC LIMIT, "
        "use SELECT DISTINCT on duplicate-prone 1-to-many entity listings, "
        "avoid redundant name-lookup joins, and verify column range:/"
        "matched_values/examples bounds."
    )
  elif strategy_tag == "skeleton_minimal":
    # Direct relational synthesis anchor forbidding SELECT * & INNER JOIN loss.
    guidance = (
        "Write the most parsimonious relational SQLite SELECT query projecting "
        "only the exact columns requested (never SELECT * or table.*): avoid "
        "extra lookup joins when filtered tables already hold the target name "
        "column, preserve LEFT JOIN on nullable foreign keys when counting or "
        "aggregating parent entities (with CASE WHEN for conditional ratios "
        "rather than dropping NULL-FK parent rows via INNER JOIN, and INNER "
        "JOIN on required child rows), group ranked entities by their single "
        "ID/FK key column, wrap sub-day timestamps in DATE(col) on "
        "'YYYY-MM-DD' comparisons, include RANK() OVER (ORDER BY ...) on "
        "imperative 'Rank X by Y' queries, and apply External Evidence "
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
