"""Unit tests for the read-only, authorizer-locked SQLite execution sandbox."""

import pathlib
import sqlite3
import tempfile
import unittest

from argus_sql.sandbox.exceptions import SandboxPathViolationError
from argus_sql.sandbox.sqlite_sandbox import SQLiteReadOnlySandbox
from argus_sql.sandbox.types import SandboxPolicy


class SQLiteSandboxTest(unittest.TestCase):
  """Verifies read-only isolation, authorizer denials, timeouts, and batching."""

  def _create_temp_db(
      self, root_dir: pathlib.Path, filename: str = "sample.sqlite"
  ) -> pathlib.Path:
    """Creates a populated sample SQLite database inside `root_dir`."""
    # Support arbitrary filenames (including spaces) to test RFC 8089 URI encoding.
    db_file = root_dir / filename
    conn = sqlite3.connect(db_file)
    try:
      conn.execute(
          "CREATE TABLE metrics (id INTEGER PRIMARY KEY, name TEXT, score REAL)"
      )
      conn.executemany(
          "INSERT INTO metrics (id, name, score) VALUES (?, ?, ?)",
          [(1, "Alpha", 12.5), (2, "Beta", 25.0), (3, "Gamma", 37.5)],
      )
      conn.commit()
    finally:
      conn.close()
    return db_file

  def test_blocks_writes_ddl_pragma_and_multi_statement_injection(self) -> None:
    """Non-SELECT, PRAGMA, memory bombs, and stacked queries must fail closed."""
    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      # Exercise percent-encoded URI paths with spaces in the database filename.
      db_file = self._create_temp_db(root, filename="sample space#1.sqlite")
      sandbox = SQLiteReadOnlySandbox(SandboxPolicy(allowed_db_root=root))

      # Attempt DROP, PRAGMA, comment-in-string bypass, and blob memory bombs.
      for malicious_sql in (
          "DROP TABLE metrics",
          "DELETE FROM metrics WHERE id = 1",
          "PRAGMA table_info(metrics)",
          "SELECT * FROM metrics; DROP TABLE metrics;",
          "SELECT '--'; DROP TABLE metrics;",
          "SELECT load_extension('evil.so')",
          "SELECT zeroblob(100000000)",
          "SELECT randomblob(100000000)",
      ):
        outcome = sandbox.execute(db_file, malicious_sql)
        self.assertFalse(outcome.succeeded, msg=f"Should block: {malicious_sql}")

      # Verify `--` inside string literals executes intact on valid SELECTs.
      dash_lit = sandbox.execute(db_file, "SELECT 'a--b' AS tag; -- trailing note")
      self.assertTrue(dash_lit.succeeded)
      self.assertEqual(dash_lit.rows, (("a--b",),))

      # Verify underlying rows remain intact and valid SELECT succeeds.
      clean_outcome = sandbox.execute(
          db_file, "SELECT name, score FROM metrics ORDER BY id ASC"
      )
      self.assertTrue(clean_outcome.succeeded)
      self.assertEqual(clean_outcome.row_count, 3)

  def test_enforces_path_traversal_and_vm_step_limits(self) -> None:
    """Database paths outside root raise error and runaway joins abort safely."""
    with tempfile.TemporaryDirectory() as tmp_root, tempfile.TemporaryDirectory() as outside:
      root = pathlib.Path(tmp_root)
      db_inside = self._create_temp_db(root)
      db_outside = self._create_temp_db(pathlib.Path(outside))

      # Path outside `allowed_db_root` must raise SandboxPathViolationError.
      strict_sandbox = SQLiteReadOnlySandbox(
          SandboxPolicy(allowed_db_root=root, max_vm_steps=3_000, timeout_ms=250.0)
      )
      with self.assertRaises(SandboxPathViolationError):
        strict_sandbox.execute(db_outside, "SELECT 1")

      # Recursive infinite CTE bomb must hit the VM step guard cleanly.
      runaway_cte = (
          "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt) "
          "SELECT x FROM cnt"
      )
      aborted = strict_sandbox.execute(db_inside, runaway_cte)
      self.assertFalse(aborted.succeeded)

      # Batch execution deduplicates whitespace-equivalent SELECT queries.
      batch_res = strict_sandbox.execute_batch(
          db_inside,
          ("SELECT COUNT(*) FROM metrics;", "  SELECT   COUNT(*) FROM metrics "),
      )
      self.assertEqual(len(batch_res), 2)
      self.assertEqual(batch_res[0].result_signature, batch_res[1].result_signature)


if __name__ == "__main__":
  unittest.main()
