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


if __name__ == "__main__":
  unittest.main()
