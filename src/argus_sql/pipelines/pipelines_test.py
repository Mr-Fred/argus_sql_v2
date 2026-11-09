"""End-to-end integration tests comparing B0..B3 baselines and Argus-SQL."""

import pathlib
import sqlite3
import tempfile
import unittest

from argus_sql.adapters.deterministic_synthesizer import SchemaGuidedDiagnosticSynthesizer
from argus_sql.contracts.types import ColumnSchema
from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import ForeignKeyEdge
from argus_sql.contracts.types import NLQueryTask
from argus_sql.contracts.types import QueryDifficulty
from argus_sql.contracts.types import TableSchema
from argus_sql.linking.value_retriever import CellValueRetriever
from argus_sql.pipelines.argus_pipeline import ArgusSQLPipeline
from argus_sql.pipelines.baseline_pipelines import CHESSValueConsistencyPipeline
from argus_sql.pipelines.baseline_pipelines import DINDecomposedPipeline
from argus_sql.pipelines.baseline_pipelines import MACRefinerPipeline
from argus_sql.pipelines.baseline_pipelines import ZeroShotDDLPipeline
from argus_sql.sandbox.sqlite_sandbox import SQLiteReadOnlySandbox
from argus_sql.sandbox.types import SandboxPolicy


class ComparativePipelinesTest(unittest.TestCase):
  """Verifies Argus-SQL resolves multi-hop, voice, and division challenges."""

  def _setup_mini_schools_db(
      self, root: pathlib.Path
  ) -> tuple[DatabaseSchema, pathlib.Path]:
    """Creates a realistic 3-table SQLite database with integer division."""
    db_file = root / "schools.sqlite"
    conn = sqlite3.connect(db_file)
    try:
      conn.execute(
          "CREATE TABLE districts (dist_id INTEGER PRIMARY KEY, state TEXT)"
      )
      conn.execute(
          "CREATE TABLE schools (cds_id INTEGER PRIMARY KEY, dist_id INTEGER, "
          "school_name TEXT)"
      )
      conn.execute(
          "CREATE TABLE frpm (frpm_id INTEGER PRIMARY KEY, cds_id INTEGER, "
          "free_meals INTEGER, enrollment INTEGER)"
      )
      # Populate districts, intermediate school bridge rows, and FRPM counts.
      conn.executemany(
          "INSERT INTO districts VALUES (?, ?)",
          [(10, "California"), (20, "Nevada")],
      )
      # Bridge table `schools` links `districts.dist_id` to `frpm.cds_id`.
      conn.executemany(
          "INSERT INTO schools VALUES (?, ?, ?)",
          [(1, 10, "Oak High"), (2, 10, "Bay High"), (3, 20, "Reno High")],
      )
      # Integer ratio `(150 + 250) / (500 + 500) = 0.4` truncates to `0` if uncast!
      conn.executemany(
          "INSERT INTO frpm VALUES (?, ?, ?, ?)",
          [(101, 1, 150, 500), (102, 2, 250, 500), (103, 3, 50, 400)],
      )
      conn.commit()
    finally:
      conn.close()

    # Declare corresponding typed DatabaseSchema with two-hop FK join edges.
    schema = DatabaseSchema(
        db_id="schools",
        tables=(
            TableSchema(
                "districts",
                (
                    ColumnSchema("dist_id", "INTEGER", is_primary_key=True),
                    ColumnSchema("state", "TEXT", value_examples=("California",)),
                ),
            ),
            TableSchema(
                "schools",
                (
                    ColumnSchema("cds_id", "INTEGER", is_primary_key=True),
                    ColumnSchema("dist_id", "INTEGER"),
                    ColumnSchema("school_name", "TEXT"),
                ),
            ),
            TableSchema(
                "frpm",
                (
                    ColumnSchema("frpm_id", "INTEGER", is_primary_key=True),
                    ColumnSchema("cds_id", "INTEGER"),
                    ColumnSchema("free_meals", "INTEGER"),
                    ColumnSchema("enrollment", "INTEGER"),
                ),
            ),
        ),
        # Connect `frpm -> schools -> districts` to test Steiner bridge recovery.
        foreign_keys=(
            ForeignKeyEdge("schools", "dist_id", "districts", "dist_id"),
            ForeignKeyEdge("frpm", "cds_id", "schools", "cds_id"),
        ),
    )
    return (schema, db_file)

  def test_argus_outperforms_baselines_on_spoken_bridge_ratio_task(self) -> None:
    """Spoken query needing bridge `schools` + `'California'` + REAL division."""
    # Create isolated SQLite database and shared read-only sandbox + 3-gram index.
    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      schema, db_file = self._setup_mini_schools_db(root)
      sandbox = SQLiteReadOnlySandbox(SandboxPolicy(allowed_db_root=root))
      synth = SchemaGuidedDiagnosticSynthesizer()
      retriever = CellValueRetriever.from_schema_and_sqlite(schema, db_file)

      # Spoken task requiring: 'Calif' -> 'California', bridge `schools`, REAL /.
      task = NLQueryTask(
          question_id="q1",
          db_id="schools",
          question="uh please tell me the overall free meal rate in Calif",
          evidence="free meal rate = free_meals / enrollment",
          difficulty=QueryDifficulty.CHALLENGING,
          gold_sql=(
              "SELECT CAST(SUM(frpm.free_meals) AS REAL) / SUM(frpm.enrollment) "
              "FROM districts JOIN schools ON schools.dist_id = districts.dist_id "
              "JOIN frpm ON frpm.cds_id = schools.cds_id "
              "WHERE districts.state = 'California'"
          ),
          is_voice_perturbed=True,
      )
      # Execute reference gold SQL first to establish ground-truth signature.
      gold_out = sandbox.execute(db_file, task.gold_sql)
      self.assertTrue(gold_out.succeeded)

      # Evaluate B0, B1, B2, B3, and Argus-SQL on the exact same task.
      for baseline in (
          ZeroShotDDLPipeline(synth, sandbox),
          DINDecomposedPipeline(synth, sandbox),
          MACRefinerPipeline(synth, sandbox),
          CHESSValueConsistencyPipeline(synth, sandbox),
      ):
        # Confirm every baseline emits a syntactically non-empty SQL string.
        pred = baseline.predict(task, schema, db_file, retriever)
        self.assertTrue(pred.final_sql)

      # Run proposed Argus-SQL pipeline and verify exact result-set equivalence.
      argus = ArgusSQLPipeline(synth, sandbox)
      argus_pred = argus.predict(task, schema, db_file, retriever)
      argus_out = sandbox.execute(db_file, argus_pred.final_sql)
      # Argus-SQL must recover `schools` bridge, `'California'`, and `0.4`!
      self.assertEqual(argus_out.result_signature, gold_out.result_signature)

  def test_argus_recovers_from_all_empty_sample_table_via_range_and_repair(
      self,
  ) -> None:
    """Single clean repair defeats 3 empty-set votes on narrow sample tables."""
    from collections.abc import Sequence
    from argus_sql.adapters.types import LLMCompletionRequest
    from argus_sql.adapters.types import LLMCompletionResponse

    class _RangeAwareMockLLM:
      """Simulates K=3 out-of-range sample-table misses followed by repair."""

      def batch_complete(
          self, requests: Sequence[LLMCompletionRequest]
      ) -> tuple[LLMCompletionResponse, ...]:
        """Emits 0-row sample predicates initially, then repairs via range.

        Args:
          requests: Batch of initial synthesis or diagnostic repair requests.

        Returns:
          Tuple of mock SQL completion responses aligned 1-to-1 with requests.
        """
        outs: list[LLMCompletionResponse] = []
        for req in requests:
          if req.strategy_tag == "repair":
            # Verify repair prompt carries both range bounds and failed SQLs!
            assert "tx_sample.Date='2012-08-23'..'2012-08-26'" in req.user_prompt
            assert "yearmonth.Date='201112'..'201311'" in req.user_prompt
            assert "Other Failed SQL Attempts" in req.user_prompt
            # Omit DISTINCT so post-repair algebraic fanout dedup is exercised!
            sql = (
                "SELECT p.Description FROM tx_sample t "
                "JOIN yearmonth y ON t.CustomerID = y.CustomerID "
                "JOIN products p ON t.ProductID = p.ProductID "
                "WHERE y.Date = '201309'"
            )
          elif req.strategy_tag == "skeleton_minimal":
            # Include mis-cased `'diesel'` on out-of-range `tx_sample` so Stage
            # 4a rewrites all 3 candidates without doubling EMPTY_SET votes!
            sql = (
                "SELECT DISTINCT p.Description FROM products p "
                "JOIN tx_sample t ON p.ProductID = t.ProductID "
                "WHERE p.Description != 'natural 95' "
                "AND strftime('%Y-%m', t.Date) = '2013-09'"
            )
          else:
            # Simulate `LIKE '2013-09%'` + mis-cased literal on narrow sample.
            sql = (
                "SELECT DISTINCT p.Description FROM products p "
                "JOIN tx_sample t ON p.ProductID = t.ProductID "
                "WHERE p.Description != 'natural 95' AND t.Date LIKE '2013-09%'"
            )
          outs.append(
              LLMCompletionResponse(req.request_id, sql, sql, 120, 30)
          )
        return tuple(outs)

    # Materialize clean-room SQLite fixture modeling narrow sample vs yearmonth.
    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      db_file = root / "debit_sample.sqlite"
      conn = sqlite3.connect(db_file)
      try:
        # Create `products`, narrow `tx_sample`, and full-range `yearmonth`.
        conn.executescript("""
          CREATE TABLE products (ProductID INTEGER PRIMARY KEY, Description TEXT);
          CREATE TABLE tx_sample (
            TxID INTEGER PRIMARY KEY, CustomerID INTEGER,
            ProductID INTEGER, Date DATE
          );
          CREATE TABLE yearmonth (
            CustomerID INTEGER, Date TEXT, Consumption REAL,
            PRIMARY KEY (CustomerID, Date)
          );
        """)
        # Populate two product labels to verify non-empty entity projection.
        conn.executemany(
            "INSERT INTO products VALUES (?, ?)",
            [(1, "Diesel"), (2, "Natural 95")],
        )
        # `tx_sample` covers only 2012-08-23..2012-08-26 with 6 fanout rows.
        conn.executemany(
            "INSERT INTO tx_sample VALUES (?, ?, ?, ?)",
            [
                (10 + i, 100, 1 if i < 4 else 2, f"2012-08-2{3 + (i % 4)}")
                for i in range(6)
            ],
        )
        # `yearmonth` spans 201112..201311 and covers target month `'201309'`.
        conn.executemany(
            "INSERT INTO yearmonth VALUES (?, ?, ?)",
            [(100, "201112", 50.0), (100, "201309", 180.0), (100, "201311", 90.0)],
        )
        conn.commit()
      finally:
        conn.close()

      # Declare typed DatabaseSchema carrying hydrated `value_range` metadata.
      schema = DatabaseSchema(
          db_id="debit_sample",
          tables=(
              TableSchema(
                  "products",
                  (
                      ColumnSchema("ProductID", "INTEGER", is_primary_key=True),
                      ColumnSchema("Description", "TEXT"),
                  ),
              ),
              # Sample table Date range reveals it lacks 2013 records.
              TableSchema(
                  "tx_sample",
                  (
                      ColumnSchema("TxID", "INTEGER", is_primary_key=True),
                      ColumnSchema("CustomerID", "INTEGER"),
                      ColumnSchema("ProductID", "INTEGER"),
                      ColumnSchema(
                          "Date", "DATE", value_range=("2012-08-23", "2012-08-26")
                      ),
                  ),
              ),
              # Composite-PK `yearmonth` Date range covers `'201112'..'201311'`.
              TableSchema(
                  "yearmonth",
                  (
                      ColumnSchema("CustomerID", "INTEGER", is_primary_key=True),
                      ColumnSchema(
                          "Date",
                          "TEXT",
                          is_primary_key=True,
                          value_range=("201112", "201311"),
                      ),
                      ColumnSchema(
                          "Consumption", "REAL", value_range=("50.0", "180.0")
                      ),
                  ),
              ),
          ),
          # Connect `tx_sample` to both `products` and `yearmonth` via FKs.
          foreign_keys=(
              ForeignKeyEdge("tx_sample", "ProductID", "products", "ProductID"),
              ForeignKeyEdge("tx_sample", "CustomerID", "yearmonth", "CustomerID"),
          ),
      )

      # Execute full Argus-SQL pipeline over the sample-range trap task.
      sandbox = SQLiteReadOnlySandbox(SandboxPolicy(allowed_db_root=root))
      task = NLQueryTask(
          question_id="1500",
          db_id="debit_sample",
          question="Please list the product descriptions (such as Natural 95) consumed in September 2013.",
          evidence="September 2013 refers to 201309.",
          difficulty=QueryDifficulty.MODERATE,
      )
      argus = ArgusSQLPipeline(_RangeAwareMockLLM(), sandbox)
      pred = argus.predict(task, schema, db_file)

      # Winning SQL must be the non-empty repaired query (with DISTINCT hardened),
      # defeating the 3 correlated 0-row sample-table queries!
      self.assertIn("yearmonth", pred.final_sql)
      self.assertTrue(pred.final_sql.startswith("SELECT DISTINCT"))
      win_out = sandbox.execute(db_file, pred.final_sql)
      self.assertEqual(set(win_out.rows), {("Diesel",), ("Natural 95",)})


if __name__ == "__main__":
  unittest.main()
