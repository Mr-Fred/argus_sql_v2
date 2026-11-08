"""Self-contained clean-room SQLite databases modeling BIRD-SQL challenges.

Materializes three multi-table relational SQLite databases (`schools_mini`,
`finance_mini`, and `clinical_mini`), each incorporating FK bridge tables,
legacy distractor tables, dirty text values, NULL extrema, and integer ratios.
"""

from collections.abc import Sequence
import pathlib
import sqlite3

from argus_sql.contracts.types import ColumnSchema
from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import ForeignKeyEdge
from argus_sql.contracts.types import TableSchema


def _make_table(
    name: str,
    col_specs: Sequence[tuple[str, str, bool, str]],
) -> TableSchema:
  """Builds a TableSchema from compact `(col_name, dtype, is_pk, desc)` tuples."""
  # Construct immutable ColumnSchema records concisely for clean schema graphs.
  cols = tuple(
      ColumnSchema(c_name, c_type, is_pk, desc)
      for c_name, c_type, is_pk, desc in col_specs
  )
  return TableSchema(name=name, columns=cols)


def _exec_ddl_and_inserts(
    db_path: pathlib.Path,
    ddl_script: str,
    inserts: Sequence[tuple[str, Sequence[tuple[object, ...]]]],
) -> None:
  """Executes `ddl_script` and parameterized batch `inserts` into `db_path`."""
  # Open transient build connection and commit all fixture tables atomically.
  conn = sqlite3.connect(db_path)
  try:
    conn.executescript(ddl_script)
    for stmt, rows in inserts:
      conn.executemany(stmt, rows)
    conn.commit()
  finally:
    conn.close()


def _build_schools_db(db_path: pathlib.Path) -> DatabaseSchema:
  """Creates the `schools_mini` SQLite database and returns its schema."""
  # Create districts -> school_bridge -> frpm plus orphan frpm_legacy table.
  ddl = """
    CREATE TABLE districts (dist_id INTEGER PRIMARY KEY, state TEXT, county TEXT);
    CREATE TABLE school_bridge (
      cds_id INTEGER PRIMARY KEY, dist_id INTEGER, school_name TEXT,
      FOREIGN KEY (dist_id) REFERENCES districts(dist_id)
    );
    CREATE TABLE frpm (
      frpm_id INTEGER PRIMARY KEY, cds_id INTEGER, free_meals INTEGER,
      enrollment INTEGER, sat_score INTEGER,
      FOREIGN KEY (cds_id) REFERENCES school_bridge(cds_id)
    );
    CREATE TABLE frpm_legacy (
      legacy_id INTEGER PRIMARY KEY, free_meals INTEGER,
      enrollment INTEGER, sat_score INTEGER
    );
  """
  # Row 5 (cds 105) carries NULL sat_score to test ASC LIMIT 1 NULL filtering.
  _exec_ddl_and_inserts(
      db_path,
      ddl,
      (
          (
              "INSERT INTO districts VALUES (?, ?, ?)",
              [
                  (1, "California", "Los Angeles"),
                  (2, "California", "San Francisco"),
                  (3, "New York", "Kings"),
              ],
          ),
          (
              "INSERT INTO school_bridge VALUES (?, ?, ?)",
              [
                  (101, 1, "Lincoln Charter"),
                  (102, 1, "Pacific Magnet"),
                  (103, 2, "Bayview Prep"),
                  (104, 3, "Brooklyn Academy"),
                  (105, 1, "Unscored Annex"),
              ],
          ),
          (
              "INSERT INTO frpm VALUES (?, ?, ?, ?, ?)",
              [
                  (1, 101, 180, 600, 1140),
                  (2, 102, 240, 800, 1290),
                  (3, 103, 120, 500, 1360),
                  (4, 104, 300, 750, 1210),
                  (5, 105, 90, 300, None),
              ],
          ),
          # Legacy orphan distractor row (`9999`) traps unpruned zero-shot DDL.
          ("INSERT INTO frpm_legacy VALUES (?, ?, ?, ?)", [(1, 9999, 10000, 400)]),
      ),
  )
  # Return typed DatabaseSchema graph with directed foreign-key join edges.
  # Note `frpm_legacy` has zero FKs so FK-aware pruning eliminates it cleanly.
  return DatabaseSchema(
      db_id="schools_mini",
      tables=(
          _make_table("districts", (("dist_id", "INTEGER", True, ""), ("state", "TEXT", False, "State name"), ("county", "TEXT", False, "County name"))),
          _make_table("school_bridge", (("cds_id", "INTEGER", True, ""), ("dist_id", "INTEGER", False, ""), ("school_name", "TEXT", False, "School title"))),
          _make_table("frpm", (("frpm_id", "INTEGER", True, ""), ("cds_id", "INTEGER", False, ""), ("free_meals", "INTEGER", False, "Free meals"), ("enrollment", "INTEGER", False, "Enrollment"), ("sat_score", "INTEGER", False, "SAT score"))),
          _make_table("frpm_legacy", (("legacy_id", "INTEGER", True, ""), ("free_meals", "INTEGER", False, ""), ("enrollment", "INTEGER", False, ""), ("sat_score", "INTEGER", False, ""))),
      ),
      foreign_keys=(
          ForeignKeyEdge("school_bridge", "dist_id", "districts", "dist_id"),
          ForeignKeyEdge("frpm", "cds_id", "school_bridge", "cds_id"),
      ),
  )


def _build_finance_db(db_path: pathlib.Path) -> DatabaseSchema:
  """Creates the `finance_mini` SQLite database and returns its schema."""
  # Create clients -> account_disp -> loans plus orphan loans_legacy table.
  # `account_disp` mirrors BIRD's `disp` bridge table between client and loan.
  ddl = """
    CREATE TABLE clients (client_id INTEGER PRIMARY KEY, state TEXT, segment TEXT);
    CREATE TABLE account_disp (
      acct_id INTEGER PRIMARY KEY, client_id INTEGER, branch_name TEXT,
      FOREIGN KEY (client_id) REFERENCES clients(client_id)
    );
    CREATE TABLE loans (
      loan_id INTEGER PRIMARY KEY, acct_id INTEGER, loan_amount REAL,
      default_count INTEGER, installment_count INTEGER, credit_score INTEGER,
      FOREIGN KEY (acct_id) REFERENCES account_disp(acct_id)
    );
    CREATE TABLE loans_legacy (
      legacy_id INTEGER PRIMARY KEY, loan_amount REAL, credit_score INTEGER
    );
  """
  # Loan 4 carries NULL credit_score to test SQLite NULL-first ordering traps.
  # `default_count / installment_count` is < 1.0, truncating to 0 as INTEGER!
  _exec_ddl_and_inserts(
      db_path,
      ddl,
      (
          (
              "INSERT INTO clients VALUES (?, ?, ?)",
              [(1, "California", "Retail"), (2, "California", "Commercial"), (3, "New York", "Commercial")],
          ),
          (
              "INSERT INTO account_disp VALUES (?, ?, ?)",
              [(11, 1, "Mission Branch"), (12, 2, "Sunset Branch"), (13, 3, "Midtown Branch"), (14, 1, "Pending Branch")],
          ),
          (
              "INSERT INTO loans VALUES (?, ?, ?, ?, ?, ?)",
              [(1, 11, 25000.0, 1, 24, 680), (2, 12, 80000.0, 3, 36, 740), (3, 13, 120000.0, 2, 48, 790), (4, 14, 15000.0, 2, 12, None)],
          ),
          # Orphan legacy backup row skews unpruned `SUM(loan_amount)` by +999k.
          ("INSERT INTO loans_legacy VALUES (?, ?, ?)", [(1, 999999.0, 300)]),
      ),
  )
  # Return typed DatabaseSchema graph for financial loan evaluation tasks.
  return DatabaseSchema(
      db_id="finance_mini",
      tables=(
          _make_table("clients", (("client_id", "INTEGER", True, ""), ("state", "TEXT", False, "Client state"), ("segment", "TEXT", False, "Banking segment"))),
          _make_table("account_disp", (("acct_id", "INTEGER", True, ""), ("client_id", "INTEGER", False, ""), ("branch_name", "TEXT", False, "Branch title"))),
          _make_table("loans", (("loan_id", "INTEGER", True, ""), ("acct_id", "INTEGER", False, ""), ("loan_amount", "REAL", False, "Loan USD"), ("default_count", "INTEGER", False, ""), ("installment_count", "INTEGER", False, ""), ("credit_score", "INTEGER", False, ""))),
          _make_table("loans_legacy", (("legacy_id", "INTEGER", True, ""), ("loan_amount", "REAL", False, ""), ("credit_score", "INTEGER", False, ""))),
      ),
      foreign_keys=(
          ForeignKeyEdge("account_disp", "client_id", "clients", "client_id"),
          ForeignKeyEdge("loans", "acct_id", "account_disp", "acct_id"),
      ),
  )


def _build_clinical_db(db_path: pathlib.Path) -> DatabaseSchema:
  """Creates the `clinical_mini` SQLite database and returns its schema."""
  # Create sponsors -> site_link -> trials plus orphan trials_legacy table.
  ddl = """
    CREATE TABLE sponsors (sponsor_id INTEGER PRIMARY KEY, state TEXT, agency_type TEXT);
    CREATE TABLE site_link (
      site_id INTEGER PRIMARY KEY, sponsor_id INTEGER, trial_title TEXT,
      FOREIGN KEY (sponsor_id) REFERENCES sponsors(sponsor_id)
    );
    CREATE TABLE trials (
      trial_id INTEGER PRIMARY KEY, site_id INTEGER, cured_patients INTEGER,
      enrolled_patients INTEGER, adverse_events INTEGER,
      FOREIGN KEY (site_id) REFERENCES site_link(site_id)
    );
    CREATE TABLE trials_legacy (
      legacy_id INTEGER PRIMARY KEY, enrolled_patients INTEGER, adverse_events INTEGER
    );
  """
  # Trial 4 carries NULL adverse_events to test ascending NULL guard invariants.
  # `cured_patients / enrolled_patients` truncates `< 1.0` ratios if uncast.
  _exec_ddl_and_inserts(
      db_path,
      ddl,
      (
          ("INSERT INTO sponsors VALUES (?, ?, ?)", [(1, "California", "Academic"), (2, "New York", "Biotech")]),
          ("INSERT INTO site_link VALUES (?, ?, ?)", [(21, 1, "CardioShield Phase3"), (22, 1, "NeuroClear Study"), (23, 2, "OncoTarget Trial"), (24, 1, "OpenLabel Screening")]),
          ("INSERT INTO trials VALUES (?, ?, ?, ?, ?)", [(1, 21, 45, 100, 4), (2, 22, 30, 80, 9), (3, 23, 60, 120, 6), (4, 24, 15, 50, None)]),
          # Orphan `trials_legacy` row (`88888`) pollutes unpruned enrollment sums.
          ("INSERT INTO trials_legacy VALUES (?, ?, ?)", [(1, 88888, 0)]),
      ),
  )
  # Return typed DatabaseSchema graph for clinical trial evaluation tasks.
  return DatabaseSchema(
      db_id="clinical_mini",
      tables=(
          _make_table("sponsors", (("sponsor_id", "INTEGER", True, ""), ("state", "TEXT", False, "Sponsor state"), ("agency_type", "TEXT", False, ""))),
          _make_table("site_link", (("site_id", "INTEGER", True, ""), ("sponsor_id", "INTEGER", False, ""), ("trial_title", "TEXT", False, "Trial title"))),
          _make_table("trials", (("trial_id", "INTEGER", True, ""), ("site_id", "INTEGER", False, ""), ("cured_patients", "INTEGER", False, ""), ("enrolled_patients", "INTEGER", False, ""), ("adverse_events", "INTEGER", False, ""))),
          _make_table("trials_legacy", (("legacy_id", "INTEGER", True, ""), ("enrolled_patients", "INTEGER", False, ""), ("adverse_events", "INTEGER", False, ""))),
      ),
      foreign_keys=(
          ForeignKeyEdge("site_link", "sponsor_id", "sponsors", "sponsor_id"),
          ForeignKeyEdge("trials", "site_id", "site_link", "site_id"),
      ),
  )


def materialize_mini_bird_databases(
    target_dir: pathlib.Path,
) -> tuple[dict[str, DatabaseSchema], dict[str, pathlib.Path]]:
  """Materializes all three clean-room diagnostic SQLite databases in `target_dir`.

  Args:
    target_dir: Directory where `.sqlite` files will be created.

  Returns:
    Pair `(schemas_by_db_id, sqlite_paths_by_db_id)`.
  """
  # Ensure destination folder exists before writing clean SQLite database files.
  target_dir.mkdir(parents=True, exist_ok=True)
  schemas: dict[str, DatabaseSchema] = {}
  paths: dict[str, pathlib.Path] = {}

  # Build each domain SQLite database deterministically from scratch.
  for db_id, builder in (
      ("schools_mini", _build_schools_db),
      ("finance_mini", _build_finance_db),
      ("clinical_mini", _build_clinical_db),
  ):
    db_file = target_dir / f"{db_id}.sqlite"
    if db_file.exists():
      db_file.unlink()
    schemas[db_id] = builder(db_file)
    paths[db_id] = db_file

  return (schemas, paths)
