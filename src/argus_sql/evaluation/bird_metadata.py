"""Metadata and sample-value hydration helpers for public BIRD-SQL databases.

Combines human-readable `column_name`, `column_description`, and codebook
`value_description` fields from official BIRD CSVs and extracts representative
non-null sample literals from read-only `.sqlite` files once at load time.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import csv
import pathlib
import re
import sqlite3

# Placeholder strings in BIRD CSVs carrying zero semantic signal.
_NOISE_NOTES: frozenset[str] = frozenset({
    "unuseful", "not useful", "none", "n/a", "null", "commonsense evidence:",
})


def _clean_csv_fragment(raw: str) -> str:
  """Collapses multiline CSV bullet text into a single compact DDL-safe line."""
  # Replace bullet symbols and CRLF breaks so inline `--` SQL comments stay 1-line.
  cleaned = raw.replace("\r", " ").replace("\n", " ").replace("·", "; ")
  return re.sub(r"\s+", " ", cleaned).strip(" ;,-")


def load_column_descriptions(db_dir: pathlib.Path) -> dict[tuple[str, str], str]:
  """Reads optional BIRD `database_description/*.csv` metadata safely.

  Merges human-readable `column_name`, `column_description`, and codebook
  `value_description` fields (when distinct and informative) so enum meanings
  such as `Active / Closed / Merged` survive into schema linking and prompts.

  Args:
    db_dir: Database folder containing an optional `database_description/` dir.

  Returns:
    Mapping of `(lowercase_table, lowercase_column) -> compact_description`.
  """
  desc_dir = db_dir / "database_description"
  if not desc_dir.is_dir():
    return {}

  # Scan CSV metadata files without following symlinks outside `db_dir`.
  descriptions: dict[tuple[str, str], str] = {}
  for csv_file in sorted(desc_dir.glob("*.csv")):
    if csv_file.is_symlink():
      continue
    tbl_lower = csv_file.stem.lower()
    try:
      # Use `utf-8-sig` so official BIRD CSVs prefixed with a `\ufeff` UTF-8 BOM
      # expose `original_column_name` cleanly.
      with csv_file.open("r", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
          orig = _clean_csv_fragment(row.get("original_column_name") or "")
          alias = _clean_csv_fragment(row.get("column_name") or "")
          c_desc = _clean_csv_fragment(row.get("column_description") or "")
          v_desc = _clean_csv_fragment(row.get("value_description") or "")

          # Prefer physical `original_column_name` key; fall back to `column_name`.
          col_key = orig or alias
          if not col_key:
            continue

          # Deduplicate redundant fields that merely repeat the column name.
          # Retaining `v_desc` preserves enum codebooks ('S = School, D = District').
          parts: list[str] = []
          seen_lower = {col_key.lower()}
          for frag in (alias, c_desc, v_desc):
            low = frag.lower()
            if frag and low not in seen_lower and low not in _NOISE_NOTES:
              seen_lower.add(low)
              parts.append(frag)

          if parts:
            descriptions[(tbl_lower, col_key.lower())] = " | ".join(parts)[:200]
    except OSError:
      # Skip unreadable optional metadata CSVs without aborting split load.
      continue
  return descriptions


def load_sqlite_column_examples(
    sqlite_path: pathlib.Path,
    table_columns: Mapping[str, Sequence[str]],
    max_examples: int = 2,
) -> dict[tuple[str, str], tuple[str, ...]]:
  """Extracts up to `max_examples` short distinct non-null values per column.

  Opens `sqlite_path` strictly via read-only RFC 8089 URI (`?mode=ro`) so DDL
  prompts show concrete date/code formats (`'201201'` vs `'2012-08-24'`).

  Args:
    sqlite_path: Path to physical `.sqlite` database file.
    table_columns: Mapping of `table_name -> sequence_of_column_names`.
    max_examples: Maximum distinct values sampled per column.

  Returns:
    Mapping of `(lowercase_table, lowercase_column) -> tuple_of_string_examples`.
  """
  # Return empty map when schemas are loaded offline without `.sqlite` binaries.
  if not sqlite_path.is_file():
    return {}

  examples: dict[tuple[str, str], tuple[str, ...]] = {}
  # Percent-encode URI via RFC 8089 `as_uri()` and enforce OS read-only mode.
  uri = f"{sqlite_path.resolve().as_uri()}?mode=ro"
  try:
    conn = sqlite3.connect(uri, uri=True, timeout=2.0)
    try:
      # Sample distinct non-null cells once per database at load time (no N+1).
      for tbl_name, cols in table_columns.items():
        t_ident = tbl_name.replace('"', '""')
        for col_name in cols:
          c_ident = col_name.replace('"', '""')
          query = (
              f'SELECT DISTINCT "{c_ident}" FROM "{t_ident}" '
              f'WHERE "{c_ident}" IS NOT NULL LIMIT ?'
          )
          try:
            vals: list[str] = []
            for (raw_val,) in conn.execute(query, (max_examples,)):
              # Skip binary BLOBs and overly long free-text prose columns.
              if isinstance(raw_val, bytes):
                continue
              text = re.sub(r"\s+", " ", str(raw_val)).strip()
              if 1 <= len(text) <= 32:
                vals.append(text)
            if vals:
              examples[(tbl_name.lower(), col_name.lower())] = tuple(vals)
          except sqlite3.Error:
            continue
    finally:
      conn.close()
  except sqlite3.Error:
    return {}
  return examples
