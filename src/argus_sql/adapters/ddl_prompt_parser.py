"""Prompt DDL parser and FK join-path builder for hermetic SQL synthesis.

Extracts `CREATE TABLE` definitions, `FOREIGN KEY` constraints, and grounded
`matched_values` literals from rendered prompt strings so diagnostic synthesis
operates purely on prompt-visible schema context.
"""

from collections import deque
from collections.abc import Sequence
import dataclasses
import re

# Extracts `CREATE TABLE <name> ( ... );` blocks from rendered DDL prompts.
_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+\"?([a-zA-Z0-9_ ]+)\"?\s*\((.*?)\)\s*;",
    re.IGNORECASE | re.DOTALL,
)

# Extracts `FOREIGN KEY (src_col) REFERENCES dst_tbl(dst_col)` clauses.
_FK_LINE_RE = re.compile(
    r"FOREIGN\s+KEY\s*\(\"?([a-zA-Z0-9_ ]+)\"?\)\s*REFERENCES\s+"
    r"\"?([a-zA-Z0-9_ ]+)\"?\s*\(\"?([a-zA-Z0-9_ ]+)\"?\)",
    re.IGNORECASE,
)

# Extracts `matched_values: ['Literal']` inline annotations from DDL comments.
_MATCHED_VAL_RE = re.compile(r"matched_values:\s*\[([^\]]+)\]")


# Immutable representation of one `CREATE TABLE` block parsed from a prompt.
@dataclasses.dataclass(frozen=True)
class ParsedPromptTable:
  """Parsed representation of one DDL table block inside an LLM prompt.

  Attributes:
    name: Exact table identifier declared in `CREATE TABLE`.
    columns: Tuple of `(col_name, col_type, matched_values_tuple)` entries.
    fks: Tuple of `(src_col, dst_table, dst_col)` foreign-key declarations.
  """

  name: str
  columns: tuple[tuple[str, str, tuple[str, ...]], ...]
  fks: tuple[tuple[str, str, str], ...]


def parse_prompt_ddl(prompt: str) -> dict[str, ParsedPromptTable]:
  """Parses all `CREATE TABLE` blocks visible inside `prompt`.

  Args:
    prompt: Full user prompt containing rendered SQLite DDL statements.

  Returns:
    Mapping of `lowercase_table_name -> ParsedPromptTable`.
  """
  tables: dict[str, ParsedPromptTable] = {}
  # Scan every `CREATE TABLE (...)` statement in prompt order.
  for match in _CREATE_TABLE_RE.finditer(prompt):
    t_name = match.group(1).strip()
    body = match.group(2)
    cols: list[tuple[str, str, tuple[str, ...]]] = []
    fks: list[tuple[str, str, str]] = []

    for raw_line in body.splitlines():
      line = raw_line.strip().rstrip(",")
      if not line:
        continue
      # Capture explicit `FOREIGN KEY (...) REFERENCES ...` declarations first.
      fk_m = _FK_LINE_RE.search(line)
      if fk_m:
        fks.append((fk_m.group(1), fk_m.group(2), fk_m.group(3)))
        continue
      # Skip table-level composite `PRIMARY KEY (...)` constraints cleanly.
      if line.upper().startswith("PRIMARY KEY"):
        continue

      # Parse column name, SQLite type, and optional `matched_values` literals.
      code_part, _, comment_part = line.partition("--")
      tokens = code_part.strip().rstrip(",").split()
      if len(tokens) >= 2:
        c_name = tokens[0].strip('"')
        c_type = tokens[1].upper()
        matched_vals: list[str] = []
        mv_m = _MATCHED_VAL_RE.search(comment_part)
        if mv_m:
          for lit in re.findall(r"'([^']*)'", mv_m.group(1)):
            matched_vals.append(lit)
        cols.append((c_name, c_type, tuple(matched_vals)))

    tables[t_name.lower()] = ParsedPromptTable(t_name, tuple(cols), tuple(fks))
  return tables


def extract_prompt_section(prompt: str, label: str) -> str:
  """Extracts a single-line `Question:` or `External Evidence:` field.

  Args:
    prompt: Full prompt string submitted to `BatchLLMPort`.
    label: Field prefix label ('Question' or 'External Evidence').

  Returns:
    Trimmed field value or empty string when absent.
  """
  # Anchor at line start so inline colons inside DDL comments are ignored.
  pattern = re.compile(rf"^{re.escape(label)}\s*:\s*(.+)$", re.MULTILINE)
  match = pattern.search(prompt)
  return match.group(1).strip() if match else ""


def column_root_tokens(col_name: str) -> set[str]:
  """Splits snake_case column names into lowercase root words.

  Args:
    col_name: Physical SQLite column or table identifier.

  Returns:
    Set of lowercase words including light singular stems.
  """
  words = set(re.findall(r"[a-z0-9]+", col_name.lower()))
  expanded = set(words)
  # Strip trailing plural `-s` on words longer than 4 chars ('loans' -> 'loan').
  for w in list(words):
    if len(w) > 4 and w.endswith("s"):
      expanded.add(w[:-1])
  return expanded


def resolve_column_owner(
    col_ident: str,
    ddl_tables: dict[str, ParsedPromptTable],
    connected_tables: set[str],
    prefer_legacy: bool,
) -> tuple[str, str]:
  """Picks the single best `(t_low, qualified_col)` owning `col_ident`.

  Args:
    col_ident: Unqualified column identifier from question or evidence.
    ddl_tables: Visible prompt DDL tables.
    connected_tables: Lowercase names of tables with visible FK edges.
    prefer_legacy: True when simulating unpruned zero-shot distractor overload.

  Returns:
    Pair `(owner_table_lower, 'Table.col')` or `('', col_ident)` if absent.
  """
  candidates: list[tuple[int, str, str]] = []
  for t_low, tbl in ddl_tables.items():
    for c_name, _, _ in tbl.columns:
      if c_name.lower() == col_ident.lower():
        # Prefer FK-connected tables unless simulating zero-shot overload.
        prio = 1 if (t_low in connected_tables) != prefer_legacy else 0
        candidates.append((prio, t_low, f"{tbl.name}.{c_name}"))
  if not candidates:
    return ("", col_ident)
  candidates.sort(key=lambda item: item[0], reverse=True)
  return (candidates[0][1], candidates[0][2])


def build_fk_join_clauses(
    needed_tables: Sequence[str],
    ddl_tables: dict[str, ParsedPromptTable],
) -> tuple[str, list[str]]:
  """Builds `FROM t0 JOIN ...` across visible FK edges in `ddl_tables`.

  Args:
    needed_tables: Ordered lowercase table names referenced by predicates/SELECT.
    ddl_tables: Visible prompt DDL tables.

  Returns:
    Pair `(base_table_name, list_of_join_clauses)`.
  """
  # Degenerate schemas or single-table lookups emit zero JOIN clauses.
  if not needed_tables:
    first = next(iter(ddl_tables.values())).name
    return (first, [])
  if len(needed_tables) == 1:
    return (ddl_tables[needed_tables[0]].name, [])

  # Build undirected FK graph strictly from FKs visible in the prompt DDL.
  adj: dict[str, list[tuple[str, str, str]]] = {}
  for t_low, tbl in ddl_tables.items():
    for src_col, dst_tbl, dst_col in tbl.fks:
      d_low = dst_tbl.lower()
      if d_low in ddl_tables:
        d_real = ddl_tables[d_low].name
        adj.setdefault(t_low, []).append(
            (d_low, f"{tbl.name}.{src_col} = {d_real}.{dst_col}", d_real)
        )
        adj.setdefault(d_low, []).append(
            (t_low, f"{d_real}.{dst_col} = {tbl.name}.{src_col}", tbl.name)
        )

  root = needed_tables[0]
  joined_set = {root}
  join_clauses: list[str] = []

  # Connect each required table via BFS over visible FKs; if disconnected,
  # emit an unbridged join (faithfully failing when FK closure is disabled!).
  for target in needed_tables[1:]:
    if target in joined_set:
      continue
    queue: deque[tuple[str, list[tuple[str, str, str]]]] = deque([(root, [])])
    seen = {root}
    found_path: list[tuple[str, str, str]] | None = None
    while queue:
      curr, path = queue.popleft()
      if curr == target:
        found_path = path
        break
      for nxt, cond, nxt_real in adj.get(curr, ()):
        if nxt not in seen:
          seen.add(nxt)
          queue.append((nxt, [*path, (nxt, cond, nxt_real)]))

    # Materialize every intermediate bridge hop along the shortest FK path.
    if found_path is not None:
      for hop_low, cond, hop_real in found_path:
        if hop_low not in joined_set:
          joined_set.add(hop_low)
          join_clauses.append(f"JOIN {hop_real} ON {cond}")
    else:
      # When bridge tables were pruned away, guessed `.id` join fails cleanly.
      t_real = ddl_tables[target].name
      r_real = ddl_tables[root].name
      join_clauses.append(f"JOIN {t_real} ON {r_real}.id = {t_real}.id")

  return (ddl_tables[root].name, join_clauses)
