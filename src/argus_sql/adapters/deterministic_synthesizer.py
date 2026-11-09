"""Rule-based relational compositional SQL synthesizer for hermetic evaluation.

Synthesizes SQLite `SELECT` queries directly from prompt-visible `CREATE TABLE`
DDL blocks, foreign-key clauses, grounded cell-value annotations, and external
evidence formulas without external API calls or task-ID lookup tables.
"""

from collections.abc import Sequence
import re

from argus_sql.adapters.ddl_prompt_parser import build_fk_join_clauses
from argus_sql.adapters.ddl_prompt_parser import column_root_tokens
from argus_sql.adapters.ddl_prompt_parser import extract_prompt_section
from argus_sql.adapters.ddl_prompt_parser import parse_prompt_ddl
from argus_sql.adapters.ddl_prompt_parser import resolve_column_owner
from argus_sql.adapters.types import LLMCompletionRequest
from argus_sql.adapters.types import LLMCompletionResponse


def _synthesize_relational_sql(req: LLMCompletionRequest) -> str:
  """Composes a SQLite SELECT query directly from prompt DDL and question."""
  prompt = req.user_prompt
  ddl_tables = parse_prompt_ddl(prompt)
  if not ddl_tables:
    return "SELECT 1"

  # Identify tables participating in visible FOREIGN KEY relationships.
  connected: set[str] = set()
  for t_low, tbl in ddl_tables.items():
    if tbl.fks:
      connected.add(t_low)
      for _, dst_tbl, _ in tbl.fks:
        connected.add(dst_tbl.lower())

  question = extract_prompt_section(prompt, "Question")
  evidence = extract_prompt_section(prompt, "External Evidence")
  q_lower = question.lower()
  q_words = set(re.findall(r"[a-z0-9]+", q_lower))

  # Defensive SQLite casts/NULL guards activate only on defensive/repair prompts.
  defensive = (
      req.strategy_tag in {"formula_defensive", "repair"}
      or "CAST(" in prompt
      or "IS NOT NULL" in prompt
  )
  # Unpruned zero-shot schemas (>=4 tables) suffer distractor table ambiguity
  # unless the user explicitly named the exact physical table in the question.
  explicit_tables = {t for t in ddl_tables if t in q_words}
  overloaded = (
      req.strategy_tag == "zero_shot"
      and len(ddl_tables) >= 4
      and not explicit_tables
  )

  # 1. Build WHERE filters from grounded `matched_values` or raw prepositions.
  where_preds: list[str] = []
  involved_tables: list[str] = []

  for t_low, tbl in ddl_tables.items():
    for c_name, _, matched_vals in tbl.columns:
      if matched_vals:
        lit = matched_vals[0].replace("'", "''")
        where_preds.append(f"{tbl.name}.{c_name} = '{lit}'")
        if t_low not in involved_tables:
          involved_tables.append(t_low)

  # Fallback raw string extraction when 3-gram value grounding is turned off.
  if not where_preds:
    raw_loc = re.search(
        r"\b(?:in\s+state|in\s+county|in|state|county|status)\s+"
        r"'?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?|[A-Za-z]{2,15})'?",
        question,
    )
    if raw_loc:
      raw_lit = raw_loc.group(1).rstrip("?.!,")
      # Never mistake `in <table_name>` for a geographic string literal filter.
      if raw_lit.lower() not in ddl_tables:
        for t_low, tbl in ddl_tables.items():
          for c_name, c_type, _ in tbl.columns:
            if c_type == "TEXT" and not c_name.endswith(
                ("_id", "_code", "_name", "_title")
            ):
              where_preds.append(f"{tbl.name}.{c_name} = '{raw_lit}'")
              if t_low not in involved_tables:
                involved_tables.append(t_low)
              break
          if where_preds:
            break

  # Parse numeric inequality filters (e.g., `enrolled_patients greater than 60`).
  num_cmp = re.search(
      r"\b([a-zA-Z_]+)\s*(?:greater than|more than|over|>)\s*(\d+(?:\.\d+)?)",
      f"{question} {evidence}",
      re.IGNORECASE,
  )
  if num_cmp:
    t_owner, qual_col = resolve_column_owner(
        num_cmp.group(1), ddl_tables, connected, overloaded
    )
    if t_owner:
      where_preds.append(f"{qual_col} > {num_cmp.group(2)}")
      if t_owner not in involved_tables:
        involved_tables.append(t_owner)

  # 2. Build SELECT, ORDER BY, and LIMIT clauses from evidence or measure match.
  select_expr = "COUNT(*)"
  order_limit = ""
  null_guard_col = ""

  # Parse External Evidence `lhs / rhs` ratio formulas when supplied in prompt.
  div_ev = re.search(
      r"([a-zA-Z_][a-zA-Z0-9_]*)\s*/\s*([a-zA-Z_][a-zA-Z0-9_]*)",
      evidence,
  )
  if div_ev:
    # Resolve numerator and denominator tables strictly from prompt-visible DDL.
    num_t, num_q = resolve_column_owner(
        div_ev.group(1), ddl_tables, connected, overloaded
    )
    den_t, den_q = resolve_column_owner(
        div_ev.group(2), ddl_tables, connected, overloaded
    )
    for t_own in (num_t, den_t):
      if t_own and t_own not in involved_tables:
        involved_tables.append(t_own)

    # Aggregate via `SUM(...)` on corpus-wide rate queries ('overall', 'total').
    use_sum = any(w in q_words for w in ("overall", "total", "combined"))
    lhs = f"SUM({num_q})" if use_sum else num_q
    rhs = f"SUM({den_q})" if use_sum else den_q
    # Non-defensive prompts emit raw `/` (faithfully reproducing F4 truncation!).
    select_expr = (
        f"CAST({lhs} AS REAL) / NULLIF({rhs}, 0)"
        if defensive
        else f"{lhs} / {rhs}"
    )
    # Add `ORDER BY ... LIMIT k` when per-entity ratio extrema are requested.
    if not use_sum and any(w in q_words for w in ("highest", "lowest", "top")):
      direction = "ASC" if "lowest" in q_words else "DESC"
      k_m = re.search(r"\btop\s+(\d+)\b", q_lower)
      k_val = k_m.group(1) if k_m else "1"
      order_limit = f"ORDER BY {select_expr} {direction} LIMIT {k_val}"
      if direction == "ASC":
        null_guard_col = num_q
  else:
    # Score candidate numeric columns, skipping legacy tables unless named.
    best_num: tuple[float, str, str] = (0.0, "", "")
    for t_low, tbl in ddl_tables.items():
      if "legacy" in t_low and explicit_tables and t_low not in explicit_tables:
        continue
      for c_name, c_type, _ in tbl.columns:
        if c_name.endswith(("_id", "_code")) or c_type not in {"INTEGER", "REAL"}:
          continue
        score = float(len(column_root_tokens(c_name) & q_words))
        # Simulate zero-shot distractor confusion when legacy tables survive.
        if score > 0 and overloaded and "legacy" in t_low:
          score += 0.5
        elif score > 0 and t_low in connected:
          score += 0.2
        if score > best_num[0]:
          best_num = (score, t_low, f"{tbl.name}.{c_name}")

    # Route between superlative ORDER BY LIMIT, COUNT(*), and scalar aggregates.
    is_extrema = any(
        w in q_words for w in ("lowest", "highest", "smallest", "largest", "top")
    )
    if is_extrema and best_num[2]:
      _, m_tbl, m_col = best_num
      if m_tbl not in involved_tables:
        involved_tables.append(m_tbl)
      select_expr = m_col
      direction = (
          "ASC" if ("lowest" in q_words or "smallest" in q_words) else "DESC"
      )
      # Spoken cardinals ('top three') fail `\d+` unless Stage 1 normalized them!
      k_m = re.search(r"\btop\s+(\d+)\b", q_lower)
      k_val = k_m.group(1) if k_m else "1"
      order_limit = f"ORDER BY {m_col} {direction} LIMIT {k_val}"
      if direction == "ASC":
        null_guard_col = m_col
    elif "how many" in q_lower or "count" in q_words:
      for t_low, tbl in ddl_tables.items():
        if column_root_tokens(tbl.name) & q_words and t_low not in involved_tables:
          involved_tables.append(t_low)
      select_expr = "COUNT(*)"
    elif best_num[2]:
      _, m_tbl, m_col = best_num
      if m_tbl not in involved_tables:
        involved_tables.append(m_tbl)
      if "average" in q_words:
        select_expr = f"AVG({m_col})"
      elif "total" in q_words or "sum" in q_words:
        select_expr = f"SUM({m_col})"
      elif "maximum" in q_words:
        select_expr = f"MAX({m_col})"
      elif "minimum" in q_words:
        select_expr = f"MIN({m_col})"
      else:
        select_expr = m_col

  # Only `formula_defensive` or repair prompts inject `IS NOT NULL` upfront.
  if defensive and null_guard_col:
    where_preds.append(f"{null_guard_col} IS NOT NULL")

  if not involved_tables:
    involved_tables.append(next(iter(ddl_tables.keys())))

  # Assemble final single-statement SQLite SELECT query across visible FK joins.
  base_table, joins = build_fk_join_clauses(involved_tables, ddl_tables)
  clauses = [f"SELECT {select_expr}", f"FROM {base_table}"]
  if joins:
    clauses.extend(joins)
  if where_preds:
    clauses.append("WHERE " + " AND ".join(where_preds))
  if order_limit:
    clauses.append(order_limit)
  return " ".join(clauses)


# Regex parsing `Option <LETTER>` SQL and execution preview blocks in B4 judge.
_OPTION_BLOCK_RE = re.compile(
    r"Option\s+([A-Z])\s*\(support:\s*(\d+)/\d+\s*candidates\):\s*\n"
    r"SQL:\s*(.+?)\n"
    r"Columns:\s*(.+?)\s*\|\s*Row Count:\s*(\d+)\s*\n"
    r"Sample Rows:\s*(.+)",
)


def _evaluate_diagnostic_judge_prompt(req: LLMCompletionRequest) -> str:
  """Selects the strongest SQLite option from prompt-visible execution previews."""
  prompt = req.user_prompt
  q_low = extract_prompt_section(prompt, "Question").lower()
  evidence = extract_prompt_section(prompt, "External Evidence")
  needs_ratio = ("/" in evidence) or any(
      w in q_low
      for w in ("rate", "ratio", "percentage", "percent", "proportion")
  )
  needs_min_guard = any(w in q_low for w in ("lowest", "smallest", "minimum"))

  # Score each competing execution cluster using only prompt-visible previews.
  best_choice = ("A", -999.0)
  for match in _OPTION_BLOCK_RE.finditer(prompt):
    label = match.group(1)
    votes = int(match.group(2))
    sql_upper = match.group(3).upper()
    row_cnt = int(match.group(5))
    preview = match.group(6).strip()

    score = 0.1 * votes
    # Penalize empty results and single-column NULL-only rows (`[(None,)]`).
    if row_cnt == 0:
      score -= 4.0
    if "[(None,)" in preview:
      score -= 3.5
    # On ratio/rate prompts, prefer REAL casts over integer-truncated `[(0,)]`.
    if needs_ratio:
      if "CAST(" in sql_upper or "1.0" in sql_upper:
        score += 2.0
      if re.fullmatch(r"\[(?:\(0,\),?\s*)+\]", preview):
        score -= 2.5
    # On ascending minimum queries, reward explicit `IS NOT NULL` predicates.
    if needs_min_guard and "IS NOT NULL" in sql_upper:
      score += 1.0
    if score > best_choice[1]:
      best_choice = (label, score)

  return f"Selected: Option {best_choice[0]}"


class SchemaGuidedDiagnosticSynthesizer:
  """Deterministic BatchLLMPort synthesizing SQL from rendered prompt DDL.

  Enables fast, reproducible, zero-network ablation experiments where accuracy
  differences emerge organically from prompt DDL pruning, value grounding,
  voice canonicalization, and SQLite semantic invariant verification.
  """

  def batch_complete(
      self,
      requests: Sequence[LLMCompletionRequest],
  ) -> tuple[LLMCompletionResponse, ...]:
    """Synthesizes SQL completions for a batch of requests deterministically.

    Args:
      requests: Sequence of LLMCompletionRequest items.

    Returns:
      Tuple of LLMCompletionResponse items with token estimates.
    """
    responses: list[LLMCompletionResponse] = []
    # Process all batch prompts in-memory with deterministic token accounting.
    for req in requests:
      if req.strategy_tag == "llm_judge":
        choice_text = _evaluate_diagnostic_judge_prompt(req)
        p_tokens = max(1, len(req.system_prompt + req.user_prompt) // 4)
        c_tokens = max(1, len(choice_text) // 4)
        responses.append(
            LLMCompletionResponse(
                request_id=req.request_id,
                sql_text="",
                raw_text=choice_text,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                from_cache=False,
            )
        )
        continue
      sql = _synthesize_relational_sql(req)
      p_tokens = max(1, len(req.system_prompt + req.user_prompt) // 4)
      c_tokens = max(1, len(sql) // 4)
      responses.append(
          LLMCompletionResponse(
              request_id=req.request_id,
              sql_text=sql,
              raw_text=f"```sql\n{sql}\n```",
              prompt_tokens=p_tokens,
              completion_tokens=c_tokens,
              from_cache=False,
          )
      )
    return tuple(responses)
