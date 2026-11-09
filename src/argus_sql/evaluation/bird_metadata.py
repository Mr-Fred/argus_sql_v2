"""Metadata, sample-value, and domain-range hydration for BIRD-SQL databases.

Combines human-readable `column_name`, `column_description`, and codebook
`value_description` fields from official BIRD CSVs and extracts representative
non-null sample literals plus batched `MIN..MAX` temporal/numeric domain bounds
from read-only `.sqlite` files once at load time.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import csv
import pathlib
import re
import sqlite3

from argus_sql.contracts.types import ColumnSchema
from argus_sql.contracts.types import TableSchema

# Placeholder strings in BIRD CSVs carrying zero semantic signal.
_NOISE_NOTES: frozenset[str] = frozenset({
    "unuseful", "not useful", "none", "n/a", "null", "commonsense evidence:",
})

# Numeric SQLite column type prefixes eligible for measure/flag range bounds.
_NUMERIC_PREFIXES = ("INT", "REAL", "NUM", "FLOAT", "DOUBLE", "DEC")

# Identifier tokens indicating surrogate keys, codes, phones, zips, or coords.
_SURROGATE_OR_COORD_TOKENS: frozenset[str] = frozenset({
    "id", "ids", "code", "codes", "zip", "zipcode", "phone", "tel", "fax",
    "ext", "lat", "lng", "lon", "latitude", "longitude", "nces", "cds",
})

# Identifier tokens signaling compact `YYYYMM` / `YYYYMMDD` temporal columns.
_TEMPORAL_NAME_HINTS: frozenset[str] = frozenset({
    "date", "datetime", "timestamp", "time", "year", "yr", "month", "ym",
    "yearmonth", "period", "season", "day", "dob", "birth", "birthday",
    "issued", "open", "closed", "created", "updated",
})

# Unambiguously order-preserving ISO date, timestamp, or multi-year span string.
_ISO_DATE_OR_SPAN_RE = re.compile(
    r"^(?:1[89]\d{2}|20\d{2})[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
    r"|^(?:19\d{2}|20\d{2})[-/](?:\d{2}|19\d{2}|20\d{2})$"
)

# Compact big-endian `YYYYMM` or `YYYYMMDD` date code (e.g., `'201112'`).
_COMPACT_YM_RE = re.compile(
    r"^(?:19\d{2}|20\d{2})(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])?$"
)

# Zero-padded 24-hour clock time `HH:MM` or `HH:MM:SS` preserving lexical order.
_CLOCK_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d(?:\.\d+)?)?$")


def _readonly_sqlite_authorizer(
    action: int,
    arg1: str | None,
    arg2: str | None,
    db_name: str | None,
    source: str | None,
) -> int:
  """Restricts metadata inspection queries strictly to read-only SELECT ops."""
  del arg1, arg2, db_name, source
  if action in (
      sqlite3.SQLITE_SELECT,
      sqlite3.SQLITE_READ,
      sqlite3.SQLITE_FUNCTION,
  ):
    return sqlite3.SQLITE_OK
  return sqlite3.SQLITE_DENY


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
    conn.set_authorizer(_readonly_sqlite_authorizer)
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


def _format_numeric_bound(val: int | float) -> str:
  """Formats numeric MIN/MAX bounds compactly (up to 4 decimal places)."""
  if isinstance(val, int):
    return str(val)
  rounded = round(val, 4)
  # Preserve tiny non-zero rates (< 5e-5) without collapsing to `0.0`.
  if val != 0.0 and rounded == 0.0:
    return f"{val:.2g}"
  return str(rounded)


def _classify_column_range(
    col: ColumnSchema,
    raw_min: object,
    raw_max: object,
    is_fk_col: bool,
) -> tuple[str, str] | None:
  """Validates `(raw_min, raw_max)` as a genuine temporal or measure range."""
  if raw_min is None or raw_max is None:
    return None
  if isinstance(raw_min, (bytes, bool)) or isinstance(raw_max, (bytes, bool)):
    return None

  dtype = col.data_type.upper()
  c_low = col.name.lower()
  col_words = set(re.findall(r"[a-z0-9]+", c_low))
  min_s = re.sub(r"\s+", " ", str(raw_min)).strip()
  max_s = re.sub(r"\s+", " ", str(raw_max)).strip()
  if not min_s or not max_s or len(min_s) > 32 or len(max_s) > 32:
    return None

  # 1. Temporal domain bounds: keep even when `min_s == max_s` so single-year or
  # narrow-window sample tables expose their exact temporal coverage!
  if _ISO_DATE_OR_SPAN_RE.fullmatch(min_s) and _ISO_DATE_OR_SPAN_RE.fullmatch(max_s):
    return (min_s, max_s)

  has_temp_hint = dtype in {"DATE", "DATETIME", "TIMESTAMP"} or bool(
      col_words & _TEMPORAL_NAME_HINTS
  ) or any(k in c_low for k in ("date", "month", "year", "birth", "issued"))
  is_code_like = c_low.endswith("id") or bool(
      col_words & {"id", "code", "zip", "zipcode", "num", "number", "phone"}
  )
  if (
      has_temp_hint
      and not is_code_like
      and _COMPACT_YM_RE.fullmatch(min_s)
      and _COMPACT_YM_RE.fullmatch(max_s)
  ):
    return (min_s, max_s)

  if (
      (dtype == "TIME" or "time" in col_words or c_low.endswith("time"))
      and _CLOCK_TIME_RE.fullmatch(min_s)
      and _CLOCK_TIME_RE.fullmatch(max_s)
  ):
    return (min_s, max_s)

  # Integer year/season columns (`seasons.year`, `races.year`) in `1800..2100`.
  if (
      dtype.startswith(_NUMERIC_PREFIXES)
      and bool(col_words & {"year", "yr", "season"})
      and isinstance(raw_min, (int, float))
      and isinstance(raw_max, (int, float))
      and 1800 <= float(raw_min) <= float(raw_max) <= 2100
  ):
    return (str(int(raw_min)), str(int(raw_max)))

  # 2. Numeric measure/flag bounds (`min < max`, excluding PKs/FKs/IDs/coords).
  if not dtype.startswith(_NUMERIC_PREFIXES):
    return None
  if (
      col.is_primary_key
      or is_fk_col
      or c_low.endswith("id")
      or c_low in {"account", "account_to", "account_from"}
  ):
    return None
  if col_words & _SURROGATE_OR_COORD_TOKENS:
    return None
  if (
      isinstance(raw_min, (int, float))
      and isinstance(raw_max, (int, float))
      and float(raw_min) < float(raw_max)
  ):
    return (_format_numeric_bound(raw_min), _format_numeric_bound(raw_max))
  return None


def load_sqlite_column_ranges(
    sqlite_path: pathlib.Path,
    tables: Sequence[TableSchema],
    fk_columns: frozenset[tuple[str, str]] = frozenset(),
) -> dict[tuple[str, str], tuple[str, str]]:
  """Extracts `MIN..MAX` domain bounds via one batched read-only query per table.

  Ignores blank `''` text strings so CSV empty cells never shadow real dates,
  and excludes free-text prose, surrogate keys, codes, and coordinates.

  Args:
    sqlite_path: Path to physical `.sqlite` database file.
    tables: Sequence of TableSchema definitions to inspect.
    fk_columns: Lowercase `(table_name, col_name)` pairs participating in FKs.

  Returns:
    Mapping of `(lowercase_table, lowercase_column) -> (min_str, max_str)`.
  """
  if not sqlite_path.is_file() or not tables:
    return {}

  ranges: dict[tuple[str, str], tuple[str, str]] = {}
  uri = f"{sqlite_path.resolve().as_uri()}?mode=ro"
  try:
    conn = sqlite3.connect(uri, uri=True, timeout=2.0)
    conn.set_authorizer(_readonly_sqlite_authorizer)
    try:
      for table in tables:
        # Skip BLOB columns upfront; batch remaining columns in <=120-col chunks.
        cand_cols = [
            c for c in table.columns if "BLOB" not in c.data_type.upper()
        ]
        if not cand_cols:
          continue
        t_ident = table.name.replace('"', '""')
        t_low = table.name.lower()

        for offset in range(0, len(cand_cols), 120):
          chunk = cand_cols[offset : offset + 120]
          proj_exprs: list[str] = []
          for col in chunk:
            c_ident = col.name.replace('"', '""')
            c_words = set(re.findall(r"[a-z0-9]+", col.name.lower()))
            is_num_measure = col.data_type.upper().startswith(
                _NUMERIC_PREFIXES
            ) and not bool(c_words & {"year", "yr", "season"})
            if is_num_measure:
              # Filter non-numeric text placeholders ('N/A', '') on numeric cols.
              expr = f"CASE WHEN typeof(\"{c_ident}\") IN ('integer', 'real') THEN \"{c_ident}\" END"
            else:
              # Filter blank CSV strings ('') so real temporal minima survive.
              expr = f"CASE WHEN typeof(\"{c_ident}\") = 'text' AND TRIM(\"{c_ident}\") = '' THEN NULL ELSE \"{c_ident}\" END"
            proj_exprs.append(f"MIN({expr}), MAX({expr})")

          batch_sql = f'SELECT {", ".join(proj_exprs)} FROM "{t_ident}"'
          try:
            row = conn.execute(batch_sql).fetchone()
          except sqlite3.Error:
            continue
          if row is None:
            continue

          for idx, col in enumerate(chunk):
            key = (t_low, col.name.lower())
            bounds = _classify_column_range(
                col,
                row[2 * idx],
                row[2 * idx + 1],
                is_fk_col=(key in fk_columns),
            )
            if bounds is not None:
              ranges[key] = bounds
    finally:
      conn.close()
  except sqlite3.Error:
    return {}
  return ranges
