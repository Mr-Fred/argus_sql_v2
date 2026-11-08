"""Public BIRD-SQL dataset loader and deterministic Voice-BIRD perturber.

Loads standard public BIRD-SQL dev / mini-dev JSON specifications (`dev.json`,
`dev_tables.json`, and `dev_databases/<db_id>/<db_id>.sqlite`) with strict
directory traversal validation, and synthesizes reproducible spoken utterances.
"""

import json
import pathlib
import re

from argus_sql.contracts.types import ColumnSchema
from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import ForeignKeyEdge
from argus_sql.contracts.types import NLQueryTask
from argus_sql.contracts.types import QueryDifficulty
from argus_sql.contracts.types import TableSchema
from argus_sql.evaluation.bird_metadata import load_column_descriptions
from argus_sql.evaluation.bird_metadata import load_sqlite_column_examples
from argus_sql.evaluation.exceptions import BenchmarkDatasetError

# Digits mapped to spoken ASR words during Voice-BIRD utterance perturbation.
_DIGIT_TO_WORD: dict[str, str] = {
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "10": "ten",
}

# Clean entity names mapped to colloquial spoken abbreviations in Voice-BIRD.
_VOICE_ABBREVS: tuple[tuple[str, str], ...] = (
    ("California", "Calif"),
    ("Los Angeles", "LA county"),
    ("New York", "NY"),
    ("San Francisco", "SF"),
)


def perturb_task_for_voice(task: NLQueryTask) -> NLQueryTask:
  """Transforms a clean NLQueryTask into a spoken Voice-BIRD ASR utterance.

  Injects realistic conversational preambles, spoken filler tokens, abbreviated
  geographic literals, and spoken number words while preserving `gold_sql`.

  Args:
    task: Clean text NLQueryTask.

  Returns:
    Voice-perturbed NLQueryTask with `is_voice_perturbed=True`.
  """
  q = task.question.strip().rstrip("?.!")
  # Replace clean geographic literals with spoken shorthand ('California' -> 'Calif').
  for full_name, spoken_short in _VOICE_ABBREVS:
    q = re.sub(
        rf"\b{re.escape(full_name)}\b",
        spoken_short,
        q,
        flags=re.IGNORECASE,
    )

  # Convert small limit numbers ('top 5' -> 'top five') to simulate ASR transcripts.
  def _num_repl(match: re.Match[str]) -> str:
    """Replaces matched numeric limit tokens with spoken English number words."""
    prefix, num = match.group(1), match.group(2)
    return f"{prefix} {_DIGIT_TO_WORD.get(num, num)}"

  q = re.sub(r"\b(top|bottom|first)\s+(\d+)\b", _num_repl, q, flags=re.IGNORECASE)

  # Inject deterministic conversational ASR preamble and mid-sentence filler.
  words = q.split()
  if len(words) >= 4:
    # Add spoken 'you know' filler deterministically at word offset 3.
    mid = min(3, len(words) - 1)
    words.insert(mid, "you know")
  # Prepend realistic conversational voice assistant wake/request phrasing.
  spoken_q = f"uh hey can you please tell me {' '.join(words).lower()}"

  return NLQueryTask(
      question_id=f"{task.question_id}_voice",
      db_id=task.db_id,
      question=spoken_q,
      evidence=task.evidence,
      difficulty=task.difficulty,
      gold_sql=task.gold_sql,
      is_voice_perturbed=True,
  )


def _parse_questions_payload(raw_text: str) -> list[dict[str, object]]:
  """Parses either a standard BIRD JSON array or HuggingFace JSONL payload."""
  stripped = raw_text.strip()
  if not stripped:
    return []
  # Standard BIRD `dev.json` and `mini_dev_sqlite.json` store a JSON list.
  if stripped.startswith("["):
    parsed = json.loads(stripped)
    if not isinstance(parsed, list):
      raise BenchmarkDatasetError("Expected JSON array in BIRD questions file.")
    return [row for row in parsed if isinstance(row, dict)]
  # HuggingFace Hub exports sometimes serialize one JSON task object per line.
  rows: list[dict[str, object]] = []
  for line in stripped.splitlines():
    if line.strip():
      obj = json.loads(line)
      if isinstance(obj, dict):
        rows.append(obj)
  return rows


def resolve_bird_root(
    bird_dir: pathlib.Path,
    tables_filename: str = "dev_tables.json",
) -> pathlib.Path:
  """Resolves `bird_dir` or an extracted `MINIDEV/` subfolder safely inside it.

  Args:
    bird_dir: User-supplied BIRD or Mini-Dev directory path.
    tables_filename: Expected schema JSON filename (`dev_tables.json`).

  Returns:
    Resolved root directory containing `tables_filename` (confined to `bird_dir`).
  """
  root = bird_dir.resolve()
  if (root / tables_filename).is_file():
    return root
  # Support archives that unpack a single top-level `MINIDEV/` wrapper folder.
  for sub_name in ("MINIDEV", "mini_dev", "mini_dev_data", "dev_20240627"):
    candidate_dir = root / sub_name
    if candidate_dir.is_symlink() or not candidate_dir.is_dir():
      continue
    resolved_sub = candidate_dir.resolve()
    if (
        resolved_sub.is_relative_to(root)
        and (resolved_sub / tables_filename).is_file()
    ):
      return resolved_sub
  return root


def _resolve_questions_path(
    root: pathlib.Path, questions_filename: str
) -> pathlib.Path:
  """Resolves `dev.json` or falls back to `mini_dev_sqlite.json`/`.jsonl`."""
  candidate = (root / questions_filename).resolve()
  if candidate.is_relative_to(root) and candidate.is_file():
    return candidate
  # Auto-discover BIRD Mini-Dev splits across flat and `sqlite/` layouts.
  if questions_filename in {
      "dev.json",
      "mini_dev_sqlite.json",
      "mini_dev_sqlite.jsonl",
      "auto",
  }:
    for alt_name in (
        "mini_dev_sqlite.json",
        "mini_dev_sqlite.jsonl",
        "sqlite/mini_dev_sqlite.json",
        "mini_dev_data/mini_dev_sqlite.json",
        "dev.json",
    ):
      alt_path = (root / alt_name).resolve()
      if alt_path.is_relative_to(root) and alt_path.is_file():
        return alt_path
  return candidate


def load_bird_split(
    bird_dir: pathlib.Path,
    questions_filename: str = "dev.json",
    tables_filename: str = "dev_tables.json",
    databases_subdir: str = "dev_databases",
) -> tuple[
    tuple[NLQueryTask, ...],
    dict[str, DatabaseSchema],
    dict[str, pathlib.Path],
]:
  """Loads tasks, schemas, and `.sqlite` paths from a public BIRD directory.

  Args:
    bird_dir: Root folder containing BIRD JSON and `dev_databases/` files.
    questions_filename: JSON/JSONL file listing tasks (`dev.json` or `mini_dev_sqlite.json`).
    tables_filename: JSON file listing database table/FK graphs (`dev_tables.json`).
    databases_subdir: Subdirectory containing `<db_id>/<db_id>.sqlite`.

  Returns:
    Triple `(tasks, schemas_by_db_id, sqlite_paths_by_db_id)`.

  Raises:
    BenchmarkDatasetError: If required files are missing or escape `bird_dir`.
  """
  # Resolve all dataset filenames strictly inside `bird_dir` to block traversal.
  outer_root = bird_dir.resolve()
  root = resolve_bird_root(outer_root, tables_filename=tables_filename)
  q_path = _resolve_questions_path(root, questions_filename)
  t_path = (root / tables_filename).resolve()
  db_root = (root / databases_subdir).resolve()

  for p in (q_path, t_path, db_root):
    if not p.is_relative_to(outer_root) or not p.exists():
      raise BenchmarkDatasetError(f"Missing or invalid BIRD path: {p.name}")

  try:
    raw_tasks = _parse_questions_payload(q_path.read_text(encoding="utf-8"))
    raw_schemas = json.loads(t_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as err:
    raise BenchmarkDatasetError("Failed parsing BIRD JSON metadata.") from err

  schemas: dict[str, DatabaseSchema] = {}
  sqlite_paths: dict[str, pathlib.Path] = {}

  # Convert BIRD's indexed `table_names_original` / `column_names_original` format.
  for entry in raw_schemas:
    db_id = str(entry["db_id"])
    # Enforce containment so malicious `db_id` entries cannot traverse upward.
    db_folder = (db_root / db_id).resolve()
    if not db_folder.is_relative_to(db_root):
      raise BenchmarkDatasetError(f"Unsafe database folder for '{db_id}'.")
    sqlite_file = db_folder / f"{db_id}.sqlite"
    if sqlite_file.is_file():
      sqlite_paths[db_id] = sqlite_file

    # Merge CSV codebook descriptions and optional SQLite sample literals.
    col_descs = load_column_descriptions(db_folder)
    t_names: list[str] = list(entry["table_names_original"])
    c_entries: list[list[object]] = list(entry["column_names_original"])
    c_types: list[str] = [str(t).upper() for t in entry.get("column_types", [])]
    raw_col_names_by_tbl: dict[str, list[str]] = {t: [] for t in t_names}
    for pair in c_entries:
      t_idx = int(pair[0])  # type: ignore[arg-type]
      if 0 <= t_idx < len(t_names):
        raw_col_names_by_tbl[t_names[t_idx]].append(str(pair[1]))
    col_examples = load_sqlite_column_examples(sqlite_file, raw_col_names_by_tbl)

    # Flatten both scalar PK indices and composite `[col_a, col_b]` PK lists.
    pk_flat: set[int] = set()
    for pk_item in entry.get("primary_keys", []):
      if isinstance(pk_item, list):
        pk_flat.update(int(x) for x in pk_item)
      elif isinstance(pk_item, int):
        pk_flat.add(pk_item)

    cols_per_table: list[list[ColumnSchema]] = [[] for _ in t_names]
    for col_idx, pair in enumerate(c_entries):
      t_idx = int(pair[0])  # type: ignore[arg-type]
      c_name = str(pair[1])
      # Skip BIRD's sentinel `[-1, "*"]` pseudo-column at index 0.
      if t_idx < 0 or t_idx >= len(t_names):
        continue
      dtype = c_types[col_idx] if col_idx < len(c_types) else "TEXT"
      lookup_key = (t_names[t_idx].lower(), c_name.lower())
      desc = col_descs.get(lookup_key, "")
      examples = col_examples.get(lookup_key, ())
      cols_per_table[t_idx].append(
          ColumnSchema(
              name=c_name,
              data_type=dtype,
              is_primary_key=(col_idx in pk_flat),
              description=desc,
              value_examples=examples,
          )
      )

    # Reconstruct directed ForeignKeyEdge objects from column index pairs.
    fks: list[ForeignKeyEdge] = []
    for fk_pair in entry.get("foreign_keys", []):
      src_i, dst_i = int(fk_pair[0]), int(fk_pair[1])
      src_t, src_c = int(c_entries[src_i][0]), str(c_entries[src_i][1])  # type: ignore[arg-type]
      dst_t, dst_c = int(c_entries[dst_i][0]), str(c_entries[dst_i][1])  # type: ignore[arg-type]
      fks.append(
          ForeignKeyEdge(t_names[src_t], src_c, t_names[dst_t], dst_c)
      )

    tables = tuple(
        TableSchema(name=t_name, columns=tuple(cols_per_table[idx]))
        for idx, t_name in enumerate(t_names)
    )
    schemas[db_id] = DatabaseSchema(db_id, tables, tuple(fks))

  # Parse question entries aligned with official BIRD `dev.json` keys.
  tasks: list[NLQueryTask] = []
  for idx, item in enumerate(raw_tasks):
    raw_diff = str(item.get("difficulty", "simple")).lower()
    # Fall back to `SIMPLE` when evaluating custom splits lacking tier labels.
    diff = (
        QueryDifficulty(raw_diff)
        if raw_diff in {"simple", "moderate", "challenging"}
        else QueryDifficulty.SIMPLE
    )
    # Preserve `evidence` and reference `SQL` strings verbatim for EX scoring.
    tasks.append(
        NLQueryTask(
            question_id=str(item.get("question_id", idx)),
            db_id=str(item["db_id"]),
            question=str(item["question"]),
            evidence=str(item.get("evidence", "")),
            difficulty=diff,
            gold_sql=str(item.get("SQL") or item.get("sql") or ""),
        )
    )

  # Return immutable task tuple alongside indexed schema and SQLite maps.
  return (tuple(tasks), schemas, sqlite_paths)
