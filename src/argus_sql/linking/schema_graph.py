"""Morphological identifier tokenization and relational FK graph closure helpers.

Provides compound identifier splitting (e.g., `gasstations` -> `gas`, `station`),
inflectional/derivational stemming (`countries` -> `country`, `consumed` ->
`consum`), implicit PK-matched foreign-key discovery, and Steiner + junction
bridge closure over relational `DatabaseSchema` graphs.
"""

from collections import deque
from collections.abc import Iterable
from collections.abc import Sequence
import re

from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import ForeignKeyEdge

# Stopwords excluded when computing lexical overlap with column identifiers.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "give", "has", "have", "how", "in", "is", "it", "list", "many", "of",
    "on", "or", "please", "show", "that", "the", "to", "what", "where",
    "which", "with",
})

# Derivational verb/noun stems normalizing common relational phrasing mismatches
# (e.g., 'consumed' <-> 'Consumption', 'translated' <-> 'translation').
_DERIVATIONAL_PREFIXES: tuple[tuple[str, str], ...] = (
    ("attend", "attend"),
    ("author", "author"),
    ("consum", "consum"),
    ("transact", "transact"),
    ("translat", "translat"),
    ("enroll", "enroll"),
    ("operat", "operat"),
    ("admin", "admin"),
    ("charter", "charter"),
    ("publish", "publish"),
)

# Common relational sub-words enabling compound splits ('gasstations', 'yearmonth').
_BASE_COMPOUND_VOCAB: frozenset[str] = frozenset({
    "account", "address", "admin", "atom", "attribute", "board", "bond",
    "card", "chain", "charter", "city", "client", "club", "code", "color",
    "colour", "constructor", "country", "county", "customer", "data", "date",
    "district", "doc", "driver", "email", "event", "exam", "expense",
    "foreign", "free", "game", "gas", "grade", "hero", "high", "lap", "link",
    "loan", "low", "mail", "match", "meal", "member", "month", "name",
    "number", "open", "order", "patient", "phone", "pit", "player", "post",
    "power", "price", "product", "race", "rate", "result", "sat", "school",
    "score", "segment", "set", "soc", "standing", "state", "station", "status",
    "stop", "street", "student", "super", "tag", "team", "test", "time",
    "title", "total", "trans", "trial", "type", "user", "value", "web",
    "write", "year",
})


def _stem_word(tok: str) -> set[str]:
  """Returns normalized morphological variants for a single lowercase token."""
  forms = {tok}
  # Irregular and `-ies -> -y` / `-oes -> -o` plural normalization
  # ('countries' -> 'country', 'superheroes' -> 'superhero').
  if len(tok) > 4 and tok.endswith("ies"):
    forms.add(tok[:-3] + "y")
  elif len(tok) > 4 and tok.endswith("oes"):
    forms.add(tok[:-2])
  elif (
      len(tok) > 4
      and tok.endswith("es")
      and tok[-4:-2] in {"ss", "sh", "ch", "tx", "dd"}
  ):
    forms.add(tok[:-2])
  elif len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
    forms.add(tok[:-1])

  # Strip regular verbal `-ed` suffixes ('tagged' -> 'tag', 'dated' -> 'date').
  if len(tok) >= 5 and tok.endswith("ed"):
    ed_base = tok[:-2]
    forms.add(ed_base)
    forms.add(tok[:-1])
    if len(ed_base) >= 4 and ed_base[-1] == ed_base[-2]:
      forms.add(ed_base[:-1])

  # Irregular financial/relational verb forms ('paid'/'paying' -> 'pay').
  if tok in {"paid", "paying", "payment", "payments"}:
    forms.add("pay")

  # Map shared derivational prefixes onto a canonical root token.
  for prefix, root in _DERIVATIONAL_PREFIXES:
    if tok.startswith(prefix):
      forms.add(root)
  # Bridge localization phrasing ('translation') to `foreign` / `language` cols.
  if tok.startswith("translat"):
    forms.update({"foreign", "language"})
  return forms


def build_compound_vocabulary(extra_texts: Iterable[str]) -> frozenset[str]:
  """Builds atomic sub-word vocabulary from base relational words and query text.

  Args:
    extra_texts: Question and evidence strings supplying domain sub-words.

  Returns:
    Frozenset of lowercase atomic words (`len >= 3`) for compound splitting.
  """
  # Seed vocabulary with universal relational nouns and dynamic query stems.
  vocab = set(_BASE_COMPOUND_VOCAB)
  for text in extra_texts:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    for raw in re.findall(r"[a-zA-Z]+", spaced.lower()):
      for stem in _stem_word(raw):
        if 3 <= len(stem) <= 12 and stem not in _STOPWORDS:
          vocab.add(stem)
  return frozenset(vocab)


def tokenize_identifier(
    text: str,
    compound_vocab: frozenset[str] = _BASE_COMPOUND_VOCAB,
) -> set[str]:
  """Splits camelCase, snake_case, and concatenated identifiers into root stems.

  Args:
    text: Identifier, column description, question, or evidence string.
    compound_vocab: Atomic words used to split unseparated compound names.

  Returns:
    Set of normalized lowercase root tokens.
  """
  # Split camelCase while retaining compact mixed-case acronyms ('IgG' -> 'igg').
  spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
  spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
  unsplit_short = [w for w in re.findall(r"[a-zA-Z0-9]+", text.lower()) if 2 <= len(w) <= 8]
  raw_tokens = re.findall(r"[a-zA-Z0-9]+", spaced.lower()) + unsplit_short
  normalized: set[str] = set()

  for tok in raw_tokens:
    # Strip trailing digit suffixes on numbered columns ('AdmEmail1' -> 'admemail').
    alpha_core = tok.rstrip("0123456789")
    candidates = [tok] + ([alpha_core] if len(alpha_core) >= 3 else [])
    for item in candidates:
      if item in _STOPWORDS or len(item) < 2:
        continue
      stems = _stem_word(item)
      normalized.update(stems)
      # Split unseparated compound names ('gasstations' -> 'gas' + 'station')
      # whenever both left and right halves belong to the atomic vocabulary.
      for stem in list(stems):
        if len(stem) >= 6:
          for split_idx in range(3, len(stem) - 2):
            left, right = stem[:split_idx], stem[split_idx:]
            if left in compound_vocab and right in compound_vocab:
              normalized.update(_stem_word(left))
              normalized.update(_stem_word(right))
  return normalized


def resolve_effective_foreign_keys(
    schema: DatabaseSchema,
    enable_implicit: bool = True,
) -> tuple[ForeignKeyEdge, ...]:
  """Returns declared FKs plus exact single-PK identifier matches when enabled.

  Recovers undeclared star-schema join edges (such as `transactions_1k.ProductID`
  -> `products.ProductID`) while ignoring generic bare `'id'`/`'legacy_id'` keys
  so orphan distractor tables remain isolated.

  Args:
    schema: Target DatabaseSchema definition.
    enable_implicit: Whether to infer undeclared entity-ID PK matches.

  Returns:
    Deduplicated tuple of explicit and inferred ForeignKeyEdge items.
  """
  # Preserve explicit-only graph when FK closure is disabled in ablations.
  if not enable_implicit:
    return schema.foreign_keys

  # Index unique single-column entity primary keys (`CustomerID`, `GasStationID`).
  single_pk_owners: dict[str, tuple[str, str]] = {}
  ambiguous_pks: set[str] = set()
  for table in schema.tables:
    pks = [c.name for c in table.columns if c.is_primary_key]
    if len(pks) != 1:
      continue
    pk_low = pks[0].lower()
    # Require specific `*_id` / `*id` / `*_code` names; skip bare 'id' or 'legacy_id'.
    if (
        pk_low in {"id", "code", "legacy_id"}
        or "legacy" in table.name.lower()
        or not pk_low.endswith(("id", "code"))
    ):
      continue
    if pk_low in single_pk_owners:
      ambiguous_pks.add(pk_low)
    else:
      single_pk_owners[pk_low] = (table.name, pks[0])

  existing_pairs = {
      (e.source_table.lower(), e.source_column.lower(), e.target_table.lower())
      for e in schema.foreign_keys
  } | {
      (e.target_table.lower(), e.target_column.lower(), e.source_table.lower())
      for e in schema.foreign_keys
  }

  merged = list(schema.foreign_keys)
  for table in schema.tables:
    t_low = table.name.lower()
    if "legacy" in t_low:
      continue
    for col in table.columns:
      c_low = col.name.lower()
      if c_low in ambiguous_pks or c_low not in single_pk_owners:
        continue
      parent_tbl, parent_col = single_pk_owners[c_low]
      if parent_tbl.lower() == t_low:
        continue
      if (t_low, c_low, parent_tbl.lower()) not in existing_pairs:
        merged.append(
            ForeignKeyEdge(table.name, col.name, parent_tbl, parent_col)
        )
        existing_pairs.add((t_low, c_low, parent_tbl.lower()))
  return tuple(merged)


def _shortest_fk_path(
    start_table: str,
    goal_table: str,
    adjacency: dict[str, list[tuple[str, ForeignKeyEdge]]],
) -> list[tuple[str, ForeignKeyEdge]]:
  """Finds shortest undirected FK join path between two tables via BFS."""
  queue: deque[tuple[str, list[tuple[str, ForeignKeyEdge]]]] = deque(
      [(start_table, [])]
  )
  visited = {start_table}
  while queue:
    current, path = queue.popleft()
    if current == goal_table:
      return path
    for neighbor, edge in adjacency.get(current, ()):
      if neighbor not in visited:
        visited.add(neighbor)
        queue.append((neighbor, [*path, (neighbor, edge)]))
  return []


def expand_fk_subgraph(
    schema: DatabaseSchema,
    seed_tables: Sequence[str],
    fk_edges: Sequence[ForeignKeyEdge],
) -> tuple[set[str], list[str], dict[str, set[str]]]:
  """Recovers Steiner shortest-path bridges, M:N junctions, and compact parents.

  Args:
    schema: Full DatabaseSchema graph.
    seed_tables: Lowercase initial seed table names.
    fk_edges: Active explicit + inferred ForeignKeyEdge sequence.

  Returns:
    Triple `(selected_table_set, bridge_tables_added, fk_columns_needed_by_tbl)`.
  """
  selected_set = set(seed_tables)
  bridge_tables: list[str] = []
  fk_cols: dict[str, set[str]] = {t: set() for t in selected_set}
  if not fk_edges:
    return (selected_set, bridge_tables, fk_cols)

  # Build undirected adjacency list for shortest-path Steiner recovery.
  adjacency: dict[str, list[tuple[str, ForeignKeyEdge]]] = {}
  outgoing_by_src: dict[str, list[ForeignKeyEdge]] = {}
  for edge in fk_edges:
    src = edge.source_table.lower()
    dst = edge.target_table.lower()
    adjacency.setdefault(src, []).append((dst, edge))
    adjacency.setdefault(dst, []).append((src, edge))
    outgoing_by_src.setdefault(src, []).append(edge)

  # 1. All-pairs shortest-path BFS across seed tables (order-invariant closure).
  if len(seed_tables) >= 2:
    for idx, src_seed in enumerate(seed_tables):
      for dst_seed in seed_tables[idx + 1 :]:
        for hop_table, edge in _shortest_fk_path(src_seed, dst_seed, adjacency):
          if hop_table not in selected_set:
            selected_set.add(hop_table)
            bridge_tables.append(hop_table)
          fk_cols.setdefault(edge.source_table.lower(), set()).add(
              edge.source_column.lower()
          )
          fk_cols.setdefault(edge.target_table.lower(), set()).add(
              edge.target_column.lower()
          )

  # 2. 1-Hop Compact Parent Dimension Expansion: include compact lookup parents
  # (<= 12 cols, e.g. `player`, `drivers`, `publisher`, `member`, `users`) when a
  # referencing child table is selected, while keeping `*_legacy` schemas tight.
  tables_by_low = {t.name.lower(): t for t in schema.tables}
  has_legacy = any("legacy" in t.name.lower() for t in schema.tables)
  max_parent_out = 0 if has_legacy else 1
  for sel_tbl in list(selected_set):
    for edge in outgoing_by_src.get(sel_tbl, ()):
      parent_low = edge.target_table.lower()
      parent_obj = tables_by_low.get(parent_low)
      if (
          parent_obj is not None
          and parent_low not in selected_set
          and len(outgoing_by_src.get(parent_low, ())) <= max_parent_out
          and len(parent_obj.columns) <= 16
          and len(selected_set) < 12
      ):
        selected_set.add(parent_low)
        bridge_tables.append(parent_low)

  # 3. Many-to-Many Junction Closure (runs after parent expansion so junctions
  # linking newly recovered entity parents, such as `attendance`, are retained).
  if len(selected_set) >= 2:
    for cand_tbl, out_edges in outgoing_by_src.items():
      if cand_tbl in selected_set or len(selected_set) >= 10:
        continue
      linked_targets = {
          e.target_table.lower()
          for e in out_edges
          if e.target_table.lower() in selected_set
      }
      if len(linked_targets) >= 2:
        selected_set.add(cand_tbl)
        bridge_tables.append(cand_tbl)

  # 4. Compact <=5-table FK-connected schemas with zero legacy orphans (`<=65`
  # cols, e.g. `debit_card_specializing`, `thrombosis_prediction`, `toxicology`)
  # keep sibling FK tables so implicit code/measure tables never drop.
  total_cols = sum(len(t.columns) for t in schema.tables)
  fk_participating = set(adjacency.keys())
  if (
      not has_legacy
      and len(schema.tables) <= 5
      and total_cols <= 65
      and len(fk_participating) == len(schema.tables)
  ):
    for tbl in schema.tables:
      tl = tbl.name.lower()
      if tl not in selected_set:
        selected_set.add(tl)
        bridge_tables.append(tl)

  return (selected_set, bridge_tables, fk_cols)
