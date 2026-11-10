"""End-to-end integration tests comparing B0..B4 baselines and Argus-SQL."""

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
from argus_sql.pipelines.baseline_pipelines import EnrichedMDDLPipeline
from argus_sql.pipelines.baseline_pipelines import ExecRefinerPipeline
from argus_sql.pipelines.baseline_pipelines import LLMJudgePipeline
from argus_sql.pipelines.baseline_pipelines import LexicalCoTPipeline
from argus_sql.pipelines.baseline_pipelines import MACRefinerPipeline
from argus_sql.pipelines.baseline_pipelines import RawDDLPipeline
from argus_sql.pipelines.baseline_pipelines import ValueMajorityVotePipeline
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
    # Verify legacy class aliases resolve to canonical operator baseline classes.
    self.assertIs(ZeroShotDDLPipeline, EnrichedMDDLPipeline)
    self.assertIs(DINDecomposedPipeline, LexicalCoTPipeline)
    self.assertIs(MACRefinerPipeline, ExecRefinerPipeline)
    self.assertIs(CHESSValueConsistencyPipeline, ValueMajorityVotePipeline)

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

      # Evaluate B0_RawDDL, B0+_EnrichedMDDL, B1..B4, and Argus-SQL on same task.
      for baseline in (
          RawDDLPipeline(synth, sandbox),
          EnrichedMDDLPipeline(synth, sandbox),
          LexicalCoTPipeline(synth, sandbox),
          ExecRefinerPipeline(synth, sandbox),
          ValueMajorityVotePipeline(synth, sandbox),
          LLMJudgePipeline(synth, sandbox),
      ):
        pred = baseline.predict(task, schema, db_file, retriever)
        self.assertTrue(pred.final_sql)

      # Run proposed Argus-SQL pipeline and verify exact result-set equivalence.
      argus = ArgusSQLPipeline(synth, sandbox)
      argus_pred = argus.predict(task, schema, db_file, retriever)
      argus_out = sandbox.execute(db_file, argus_pred.final_sql)
      # Argus-SQL must recover `schools` bridge, `'California'`, and `0.4`!
      self.assertEqual(argus_out.result_signature, gold_out.result_signature)

  def test_b4_judge_short_circuits_rescues_b3_and_trails_argus_on_correlated_bugs(
      self,
  ) -> None:
    """`B4_LLMJudge` rescues `B3` on 1/3 REAL split, yet fails `0/3` traps."""
    from collections.abc import Sequence
    from argus_sql.adapters.types import LLMCompletionRequest
    from argus_sql.adapters.types import LLMCompletionResponse

    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      schema, db_file = self._setup_mini_schools_db(root)
      sandbox = SQLiteReadOnlySandbox(SandboxPolicy(allowed_db_root=root))
      synth = SchemaGuidedDiagnosticSynthesizer()
      retriever = CellValueRetriever.from_schema_and_sqlite(schema, db_file)

      # 1. Agreement short-circuit: simple count produces 1 cluster -> 0 judge calls.
      simple_task = NLQueryTask(
          "s_agree", "schools", "How many districts are in state California?"
      )
      b4 = LLMJudgePipeline(synth, sandbox)
      agree_pred = b4.predict(simple_task, schema, db_file, retriever)
      # Metadata must confirm that no extra Judge completion was dispatched.
      self.assertIn(("judge_invoked", "False"), agree_pred.metadata)

      # 2. Disagreement rescue: B3 picks truncated `0` (2/3 votes), whereas B4
      #    invokes the execution-preview Judge and selects Option B (`0.4`).
      ratio_task = NLQueryTask(
          "s_ratio",
          "schools",
          "What is the overall free meal rate in California?",
          evidence="free meal rate = free_meals / enrollment",
          difficulty=QueryDifficulty.CHALLENGING,
      )
      b3_pred = ValueMajorityVotePipeline(synth, sandbox).predict(
          ratio_task, schema, db_file, retriever
      )
      b4_pred = b4.predict(ratio_task, schema, db_file, retriever)
      # Confirm B3 succumbs to majority truncation while B4 selects REAL 0.4.
      self.assertEqual(sandbox.execute(db_file, b3_pred.final_sql).rows, ((0,),))
      self.assertEqual(sandbox.execute(db_file, b4_pred.final_sql).rows, ((0.4,),))
      # Confirm B4 spent extra Judge prompt + completion tokens beyond B3's K=3.
      self.assertIn(("judge_invoked", "True"), b4_pred.metadata)
      self.assertGreater(b4_pred.total_tokens, b3_pred.total_tokens)

      # 3. Correlated `0/3` bug: all 3 candidates omit `CAST(... AS REAL)`.
      #    `B4_LLMJudge` has 1 wrong cluster (`Oracle@3 = 0%`), whereas
      #    `Argus-SQL`'s 0-token algebraic repair rewrites it to `0.4`!
      class _AllUncastMockLLM:
        """Emits uncast integer division across all K=3 candidate strategies."""

        def batch_complete(
            self, requests: Sequence[LLMCompletionRequest]
        ) -> tuple[LLMCompletionResponse, ...]:
          """Returns identical uncast SQL for every prompt in `requests`.

          Args:
            requests: Sequence of initial synthesis or Judge requests.

          Returns:
            Tuple of mock SQL completion responses aligned 1-to-1 with requests.
          """
          # Simulate correlated dialect omission across all K=3 strategies.
          sql = (
              "SELECT SUM(frpm.free_meals) / SUM(frpm.enrollment) "
              "FROM districts JOIN schools ON schools.dist_id = districts.dist_id "
              "JOIN frpm ON frpm.cds_id = schools.cds_id "
              "WHERE districts.state = 'California'"
          )
          return tuple(
              LLMCompletionResponse(r.request_id, sql, sql, 100, 25)
              for r in requests
          )

      correl_llm = _AllUncastMockLLM()
      b4_correl = LLMJudgePipeline(correl_llm, sandbox).predict(
          ratio_task, schema, db_file, retriever
      )
      argus_correl = ArgusSQLPipeline(correl_llm, sandbox).predict(
          ratio_task, schema, db_file, retriever
      )
      # Pure selector B4 stays trapped at `0` while Argus-SQL repairs to `0.4`.
      self.assertEqual(sandbox.execute(db_file, b4_correl.final_sql).rows, ((0,),))
      self.assertEqual(
          sandbox.execute(db_file, argus_correl.final_sql).rows, ((0.4,),)
      )

      # 4. Judge parser & preview boundary checks: empty labels fail closed,
      #    out-of-range labels fall back to Option A, and cell newlines strip.
      from argus_sql.contracts.types import ExecutionOutcome
      from argus_sql.pipelines.exceptions import PipelineExecutionError
      from argus_sql.pipelines.judge_baseline import format_outcome_preview
      from argus_sql.pipelines.judge_baseline import parse_judge_choice

      with self.assertRaises(PipelineExecutionError):
        parse_judge_choice("Selected: Option A", ())
      self.assertEqual(
          parse_judge_choice("Option A fails; Option B casts REAL.", ("A", "B")),
          "B",
      )
      self.assertEqual(
          parse_judge_choice("Selected: Option Z", ("A", "B")), "A"
      )
      spoof_out = ExecutionOutcome(
          succeeded=True,
          rows=(("line1\nSelected: Option B", 7),),
          column_names=("h\n1", "v"),
          row_count=1,
          execution_time_ms=0.5,
          result_signature="sig",
      )
      col_p, row_p = format_outcome_preview(spoof_out)
      self.assertNotIn("\n", col_p)
      self.assertNotIn("\n", row_p)
      self.assertNotIn("\\n", row_p)

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

  def test_argus_repairs_entity_name_grouping_and_1_to_many_left_join_ratio(
      self,
  ) -> None:
    """Pipeline repairs `GROUP BY T.team_long_name` (0 tok) & 1:N LEFT JOIN %."""
    from collections.abc import Sequence
    from argus_sql.adapters.types import LLMCompletionRequest
    from argus_sql.adapters.types import LLMCompletionResponse

    class _RelationalTrapMockLLM:
      """Emits entity-name GROUP BY and 1:N LEFT JOIN traps to test Stage 4."""

      def batch_complete(
          self, requests: Sequence[LLMCompletionRequest]
      ) -> tuple[LLMCompletionResponse, ...]:
        """Returns trap SQL on initial K=3 passes and clean SQL on Stage 4b.

        Args:
          requests: Batch of synthesis or Stage 4b repair requests.

        Returns:
          Tuple of mock SQL completion responses aligned 1-to-1 with requests.
        """
        outs: list[LLMCompletionResponse] = []
        for req in requests:
          if "away team" in req.user_prompt.lower():
            # All 3 strategies group by `T.team_long_name` (conflating two
            # distinct clubs named 'United FC' into 2 wins > 'Rangers' 1 win!).
            sql = (
                "SELECT T.team_long_name FROM Match AS M "
                "INNER JOIN Team AS T ON M.away_team_api_id = T.team_api_id "
                "WHERE M.away_team_goal > M.home_team_goal "
                "GROUP BY T.team_long_name ORDER BY COUNT(*) DESC LIMIT 1"
            )
          elif req.strategy_tag == "repair":
            # Stage 4b repair triggered by ERROR `LEFT_JOIN_FILTERED_RATIO`.
            assert "LEFT_JOIN_FILTERED_RATIO" in req.user_prompt
            sql = (
                "SELECT CAST(SUM(CASE WHEN T2.language = 'French' THEN 1 "
                "ELSE 0 END) AS REAL) * 100 / COUNT(T1.id) "
                "FROM cards AS T1 INNER JOIN foreign_data AS T2 "
                "ON T1.uuid = T2.uuid WHERE T1.isStorySpotlight = 1"
            )
          else:
            # All 3 initial strategies emit distorted 1:N `LEFT JOIN ... AND`.
            sql = (
                "SELECT CAST(COUNT(T2.id) AS REAL) * 100.0 / NULLIF(COUNT(T1.id), 0) "
                "FROM cards AS T1 LEFT JOIN foreign_data AS T2 "
                "ON T1.uuid = T2.uuid AND T2.language = 'French' "
                "WHERE T1.isStorySpotlight = 1"
            )
          outs.append(LLMCompletionResponse(req.request_id, sql, sql, 80, 25))
        return tuple(outs)

    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      db_file = root / "rel_traps.sqlite"
      conn = sqlite3.connect(db_file)
      try:
        conn.executescript("""
          CREATE TABLE Team (
            id INTEGER PRIMARY KEY, team_api_id INTEGER, team_long_name TEXT
          );
          CREATE TABLE Match (
            id INTEGER PRIMARY KEY, away_team_api_id INTEGER,
            home_team_goal INTEGER, away_team_goal INTEGER
          );
          CREATE TABLE cards (
            id INTEGER PRIMARY KEY, uuid TEXT, isStorySpotlight INTEGER
          );
          CREATE TABLE foreign_data (
            id INTEGER PRIMARY KEY, uuid TEXT, language TEXT
          );
        """)
        # Two distinct clubs named `'United FC'` have 2 wins each (4 if wrongly
        # merged by name!), whereas `'Rangers'` (`team_api_id=10`) has 3 wins!
        conn.executemany(
            "INSERT INTO Team VALUES (?, ?, ?)",
            [(1, 10, "Rangers"), (2, 20, "United FC"), (3, 30, "United FC")],
        )
        conn.executemany(
            "INSERT INTO Match VALUES (?, ?, ?, ?)",
            [
                (1, 10, 0, 2),
                (2, 10, 1, 3),
                (3, 10, 0, 1),
                (4, 20, 0, 1),
                (5, 20, 0, 2),
                (6, 30, 1, 2),
                (7, 30, 0, 3),
            ],
        )
        # Card u1 has French + German; Card u2 has German + Spanish (1/4 = 25%,
        # whereas LEFT JOIN ... AND language='French' gives 1/2 = 50%!).
        conn.executemany(
            "INSERT INTO cards VALUES (?, ?, ?)", [(1, "u1", 1), (2, "u2", 1)]
        )
        conn.executemany(
            "INSERT INTO foreign_data VALUES (?, ?, ?)",
            [
                (1, "u1", "French"),
                (2, "u1", "German"),
                (3, "u2", "German"),
                (4, "u2", "Spanish"),
            ],
        )
        conn.commit()
      finally:
        conn.close()

      schema = DatabaseSchema(
          db_id="rel_traps",
          tables=(
              TableSchema(
                  "Team",
                  (
                      ColumnSchema("id", "INTEGER", is_primary_key=True),
                      ColumnSchema("team_api_id", "INTEGER"),
                      ColumnSchema("team_long_name", "TEXT"),
                  ),
              ),
              TableSchema(
                  "Match",
                  (
                      ColumnSchema("id", "INTEGER", is_primary_key=True),
                      ColumnSchema("away_team_api_id", "INTEGER"),
                      ColumnSchema("home_team_goal", "INTEGER"),
                      ColumnSchema("away_team_goal", "INTEGER"),
                  ),
              ),
              TableSchema(
                  "cards",
                  (
                      ColumnSchema("id", "INTEGER", is_primary_key=True),
                      ColumnSchema("uuid", "TEXT"),
                      ColumnSchema("isStorySpotlight", "INTEGER"),
                  ),
              ),
              TableSchema(
                  "foreign_data",
                  (
                      ColumnSchema("id", "INTEGER", is_primary_key=True),
                      ColumnSchema("uuid", "TEXT"),
                      ColumnSchema("language", "TEXT"),
                  ),
              ),
          ),
          foreign_keys=(
              ForeignKeyEdge("Match", "away_team_api_id", "Team", "team_api_id"),
              ForeignKeyEdge("foreign_data", "uuid", "cards", "uuid"),
          ),
      )
      sandbox = SQLiteReadOnlySandbox(SandboxPolicy(allowed_db_root=root))
      argus = ArgusSQLPipeline(_RelationalTrapMockLLM(), sandbox)

      # 1. Zero-token `NON_KEY_ENTITY_GROUPING` repair recovers `'Rangers'`!
      t_team = NLQueryTask(
          "1028",
          "rel_traps",
          "Which away team won the most during the 2010 season?",
          "",
          QueryDifficulty.CHALLENGING,
      )
      p_team = argus.predict(t_team, schema, db_file)
      self.assertIn("GROUP BY M.away_team_api_id", p_team.final_sql)
      self.assertEqual(sandbox.execute(db_file, p_team.final_sql).rows, (("Rangers",),))
      # Zero Stage 4b LLM repair tokens spent (pure K=3 = 3 * 105 = 315 tokens)!
      self.assertEqual(p_team.total_tokens, 315)

      # 2. 1:N `LEFT_JOIN_FILTERED_RATIO` escalates to Stage 4b and recovers 25.0%!
      t_pct = NLQueryTask(
          "371",
          "rel_traps",
          "What is the percentage of French cards among Story Spotlight cards?",
          "Story Spotlight refers to isStorySpotlight = 1",
          QueryDifficulty.CHALLENGING,
      )
      p_pct = argus.predict(t_pct, schema, db_file)
      self.assertEqual(sandbox.execute(db_file, p_pct.final_sql).rows, ((25.0,),))

  def test_argus_repairs_timestamp_bare_date_zero_token_and_missing_window_rank(
      self,
  ) -> None:
    """Pipeline repairs bare DATETIME filters (0 tok) & missing RANK() OVER."""
    from collections.abc import Sequence
    from argus_sql.adapters.types import LLMCompletionRequest
    from argus_sql.adapters.types import LLMCompletionResponse

    class _Iter4TrapMockLLM:
      """Emits bare DATETIME comparisons and un-windowed ORDER BY rankings."""

      def batch_complete(
          self, requests: Sequence[LLMCompletionRequest]
      ) -> tuple[LLMCompletionResponse, ...]:
        """Returns trap SQL on initial K=3 passes and window SQL on Stage 4b.

        Args:
          requests: Batch of synthesis or Stage 4b repair requests.

        Returns:
          Tuple of mock SQL completion responses aligned 1-to-1 with requests.
        """
        outs: list[LLMCompletionResponse] = []
        for req in requests:
          if "last accessed" in req.user_prompt.lower():
            # All 3 strategies copy bare `LastAccessDate > '2014-09-01'` from
            # Evidence, falsely counting same-day `'2014-09-01 14:23:10'` rows!
            sql = "SELECT COUNT(Id) FROM users WHERE LastAccessDate > '2014-09-01'"
          elif req.strategy_tag == "repair":
            # Stage 4b repair triggered by `MISSING_WINDOW_RANK`.
            assert "MISSING_WINDOW_RANK" in req.user_prompt
            sql = (
                "SELECT name, height_cm, RANK() OVER (ORDER BY height_cm DESC) "
                "FROM heroes ORDER BY height_cm DESC"
            )
          else:
            # Initial K=3 strategies omit `RANK() OVER (...)`.
            sql = "SELECT name, height_cm FROM heroes ORDER BY height_cm DESC"
          outs.append(LLMCompletionResponse(req.request_id, sql, sql, 70, 20))
        return tuple(outs)

    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      db_file = root / "iter4_traps.sqlite"
      conn = sqlite3.connect(db_file)
      try:
        conn.executescript("""
          CREATE TABLE users (Id INTEGER PRIMARY KEY, LastAccessDate DATETIME);
          CREATE TABLE heroes (id INTEGER PRIMARY KEY, name TEXT, height_cm INTEGER);
        """)
        conn.executemany(
            "INSERT INTO users VALUES (?, ?)",
            [
                (1, "2014-08-31 22:00:00.000"),
                (2, "2014-09-01 14:23:10.000"),
                (3, "2014-09-02 09:15:00.000"),
            ],
        )
        conn.executemany(
            "INSERT INTO heroes VALUES (?, ?, ?)",
            [(1, "Hulk", 244), (2, "Thor", 198)],
        )
        conn.commit()
      finally:
        conn.close()

      schema = DatabaseSchema(
          db_id="iter4_traps",
          tables=(
              TableSchema(
                  "users",
                  (
                      ColumnSchema("Id", "INTEGER", is_primary_key=True),
                      ColumnSchema(
                          "LastAccessDate",
                          "DATETIME",
                          value_range=(
                              "2014-08-31 22:00:00.000",
                              "2014-09-02 09:15:00.000",
                          ),
                      ),
                  ),
              ),
              TableSchema(
                  "heroes",
                  (
                      ColumnSchema("id", "INTEGER", is_primary_key=True),
                      ColumnSchema("name", "TEXT"),
                      ColumnSchema("height_cm", "INTEGER"),
                  ),
              ),
          ),
      )
      sandbox = SQLiteReadOnlySandbox(SandboxPolicy(allowed_db_root=root))
      argus = ArgusSQLPipeline(_Iter4TrapMockLLM(), sandbox)

      # 1. Zero-token `DATE(LastAccessDate)` repair recovers `((1,),)` (not 2)
      #    without spending a single Stage 4b repair token (`3 * 90 = 270`)!
      t_ts = NLQueryTask(
          "533",
          "iter4_traps",
          "How many users last accessed the website after 2014/9/1?",
          "last accessed after 2014/9/1 refers to LastAccessDate > '2014-09-01'",
          QueryDifficulty.SIMPLE,
      )
      p_ts = argus.predict(t_ts, schema, db_file)
      self.assertIn("DATE(LastAccessDate) > '2014-09-01'", p_ts.final_sql)
      self.assertEqual(sandbox.execute(db_file, p_ts.final_sql).rows, ((1,),))
      self.assertEqual(p_ts.total_tokens, 270)

      # 2. `MISSING_WINDOW_RANK` routes imperative `Rank ... by ...` to Stage 4b!
      t_rk = NLQueryTask(
          "726",
          "iter4_traps",
          "Rank heroes by their height in descending order.",
          "",
          QueryDifficulty.MODERATE,
      )
      p_rk = argus.predict(t_rk, schema, db_file)
      self.assertIn("RANK() OVER", p_rk.final_sql)
      self.assertEqual(
          sandbox.execute(db_file, p_rk.final_sql).rows,
          (("Hulk", 244, 1), ("Thor", 198, 2)),
      )

  def test_iteration5_repair_dedup_select_star_stage4b_and_prompt_guardrails(
      self,
  ) -> None:
    """Verifies QID 24 in-place repair dedup, QID 159 `*` Stage 4b, & QID 743."""
    from collections.abc import Sequence
    from argus_sql.adapters.types import LLMCompletionRequest
    from argus_sql.adapters.types import LLMCompletionResponse
    from argus_sql.pipelines.prompt_templates import SQLITE_SYSTEM_PROMPT
    from argus_sql.pipelines.prompt_templates import build_strategy_request

    # 1. QID 743 Prompt Guardrail Invariants: `SQLITE_SYSTEM_PROMPT` and all
    #    synthesis strategies forbid `SELECT *` and preserve `LEFT JOIN` on
    #    nullable foreign keys for parent entity counts/aggregations without
    #    leaking `CAST(` or `IS NOT NULL` into non-defensive user prompts!
    sys_low = SQLITE_SYSTEM_PROMPT.lower()
    self.assertIn("select *", sys_low)
    self.assertIn("left join", sys_low)
    self.assertIn("nullable foreign keys", sys_low)
    self.assertIn("parent", sys_low)

    for strat in ("decomposed_cot", "formula_defensive", "skeleton_minimal"):
      req = build_strategy_request(
          request_id=f"r3-{strat}",
          ddl_prompt="CREATE TABLE superhero (id INTEGER PRIMARY KEY);",
          question="What is the percentage of bad superheroes?",
          evidence="alignment = 'Bad'",
          strategy_tag=strat,
      )
      u_low = req.user_prompt.lower()
      self.assertIn("select *", u_low)
      self.assertIn("left join", u_low)
      self.assertIn("nullable foreign keys", u_low)
      self.assertIn("parent", u_low)
      if strat != "formula_defensive":
        self.assertNotIn("CAST(", req.user_prompt)
        self.assertNotIn("IS NOT NULL", req.user_prompt)

    # 2. QID 24 & QID 159 End-to-End Pipeline Integration:
    class _Iter5TrapMockLLM:
      """Simulates QID 24 REAL-division `sname` trap & QID 159 `SELECT T1.*`."""

      def batch_complete(
          self, requests: Sequence[LLMCompletionRequest]
      ) -> tuple[LLMCompletionResponse, ...]:
        """Emits strategy-specific SQL traps and Stage 4b `SELECT *` repair.

        Args:
          requests: Batch of synthesis or Stage 4b repair requests.

        Returns:
          Tuple of mock SQL completion responses aligned 1-to-1 with requests.
        """
        outs: list[LLMCompletionResponse] = []
        for req in requests:
          if "withdrawals in cash" in req.user_prompt.lower():
            if req.strategy_tag == "repair":
              assert "PROJECTION_ARITY_DRIFT" in req.user_prompt
              assert "SELECT *" in req.user_prompt
              sql = (
                  "SELECT T1.trans_id FROM trans AS T1 "
                  "INNER JOIN disp AS T2 ON T1.account_id = T2.account_id "
                  "WHERE T2.client_id = 3356 AND T1.operation = 'VYBER'"
              )
            else:
              # All K=3 initial strategies over-project `SELECT T1.*`!
              sql = (
                  "SELECT T1.* FROM trans AS T1 "
                  "INNER JOIN disp AS T2 ON T1.account_id = T2.account_id "
                  "WHERE T2.client_id = 3356 AND T1.operation = 'VYBER'"
              )
          elif req.strategy_tag == "decomposed_cot":
            # Clean untruncated `frpm."School Name"` candidate (1 vote).
            sql = (
                'SELECT T2."School Name" FROM satscores AS T1 '
                "INNER JOIN frpm AS T2 ON T1.cds = T2.CDSCode "
                'WHERE CAST(T2."Free Meal Count (K-12)" AS REAL) / '
                'T2."Enrollment (K-12)" > 0.1 AND T1.NumGE1500 > 0'
            )
          elif req.strategy_tag == "formula_defensive":
            # Empty result leaves 1 clean `School Name` vs 1 repaired `sname`.
            sql = 'SELECT "School Name" FROM frpm WHERE 1 = 0'
          else:
            # `skeleton_minimal` divides REAL columns without CAST -> Stage 4a
            # rewrites to CAST(...) producing IDENTICAL rows on `T1.sname`!
            sql = (
                "SELECT T1.sname FROM satscores AS T1 "
                "INNER JOIN frpm AS T2 ON T1.cds = T2.CDSCode "
                'WHERE T2."Free Meal Count (K-12)" / '
                'T2."Enrollment (K-12)" > 0.1 AND T1.NumGE1500 > 0'
            )
          outs.append(LLMCompletionResponse(req.request_id, sql, sql, 80, 20))
        return tuple(outs)

    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      db_file = root / "iter5_traps.sqlite"
      conn = sqlite3.connect(db_file)
      try:
        conn.executescript("""
          CREATE TABLE satscores (
            cds TEXT PRIMARY KEY, sname TEXT, NumGE1500 INTEGER
          );
          CREATE TABLE frpm (
            CDSCode TEXT PRIMARY KEY,
            "School Name" TEXT,
            "Free Meal Count (K-12)" REAL,
            "Enrollment (K-12)" REAL
          );
          CREATE TABLE disp (
            disp_id INTEGER PRIMARY KEY, client_id INTEGER, account_id INTEGER
          );
          CREATE TABLE trans (
            trans_id INTEGER PRIMARY KEY,
            account_id INTEGER,
            operation TEXT,
            amount INTEGER
          );
        """)
        conn.execute(
            "INSERT INTO satscores VALUES ('01', 'East LA Performing', 25)"
        )
        conn.execute(
            "INSERT INTO frpm VALUES "
            "('01', 'East LA Performing Arts Academy', 250.0, 500.0)"
        )
        conn.execute("INSERT INTO disp VALUES (1, 3356, 77)")
        conn.execute("INSERT INTO trans VALUES (991, 77, 'VYBER', 500)")
        conn.commit()
      finally:
        conn.close()

      schema = DatabaseSchema(
          db_id="iter5_traps",
          tables=(
              TableSchema(
                  "satscores",
                  (
                      ColumnSchema("cds", "TEXT", is_primary_key=True),
                      ColumnSchema("sname", "TEXT"),
                      ColumnSchema("NumGE1500", "INTEGER"),
                  ),
              ),
              TableSchema(
                  "frpm",
                  (
                      ColumnSchema("CDSCode", "TEXT", is_primary_key=True),
                      ColumnSchema("School Name", "TEXT"),
                      ColumnSchema("Free Meal Count (K-12)", "REAL"),
                      ColumnSchema("Enrollment (K-12)", "REAL"),
                  ),
              ),
              TableSchema(
                  "disp",
                  (
                      ColumnSchema("disp_id", "INTEGER", is_primary_key=True),
                      ColumnSchema("client_id", "INTEGER"),
                      ColumnSchema("account_id", "INTEGER"),
                  ),
              ),
              TableSchema(
                  "trans",
                  (
                      ColumnSchema("trans_id", "INTEGER", is_primary_key=True),
                      ColumnSchema("account_id", "INTEGER"),
                      ColumnSchema("operation", "TEXT"),
                      ColumnSchema("amount", "INTEGER"),
                  ),
              ),
          ),
          foreign_keys=(
              ForeignKeyEdge("satscores", "cds", "frpm", "CDSCode"),
              ForeignKeyEdge("trans", "account_id", "disp", "account_id"),
          ),
      )
      sandbox = SQLiteReadOnlySandbox(SandboxPolicy(allowed_db_root=root))
      argus = ArgusSQLPipeline(_Iter5TrapMockLLM(), sandbox)

      # QID 24: Identical-output algebraic repair upgrades `skeleton_minimal`
      # in-place (`len == 3`, NOT `4`) and `School Name` defeats `sname`!
      t_24 = NLQueryTask(
          "24",
          "iter5_traps",
          "Give the names of the schools with the percent eligible for free "
          "meals in K-12 is more than 0.1 and test takers whose test score "
          "is greater than or equal to 1500?",
          "Percent eligible = Free Meal Count (K-12) / Enrollment (K-12)",
          QueryDifficulty.MODERATE,
      )
      p_24 = argus.predict(t_24, schema, db_file)
      self.assertEqual(len(p_24.verified_candidates), 3)
      self.assertEqual(
          sandbox.execute(db_file, p_24.final_sql).rows,
          (("East LA Performing Arts Academy",),),
      )

      # QID 159: `SELECT T1.*` at `expected_projection_arity == 1` escalates to
      # `ERROR`, triggering Stage 4b diagnostic repair to `SELECT T1.trans_id`!
      t_159 = NLQueryTask(
          "159",
          "iter5_traps",
          "List all the withdrawals in cash transactions that the client "
          "with the id 3356 makes.",
          "operation = 'VYBER' refers to withdrawal in cash",
          QueryDifficulty.SIMPLE,
      )
      p_159 = argus.predict(t_159, schema, db_file)
      self.assertEqual(sandbox.execute(db_file, p_159.final_sql).rows, ((991,),))

  def test_algebraic_repair_preserves_non_empty_parent_when_rewrite_returns_empty(
      self,
  ) -> None:
    """Valid non-empty parent SQL must survive when `IS NOT NULL` yields 0 rows."""
    # Import typed LLM request/response contracts for deterministic test mocking.
    from collections.abc import Sequence
    from argus_sql.adapters.types import LLMCompletionRequest
    from argus_sql.adapters.types import LLMCompletionResponse

    # Mock adapter returning an ascending ORDER BY query without IS NOT NULL.
    class _NullableAscOrderMockLLM:
      """Emits `ORDER BY pts ASC LIMIT 1` where matching rows have NULL `pts`."""

      def batch_complete(
          self, requests: Sequence[LLMCompletionRequest]
      ) -> tuple[LLMCompletionResponse, ...]:
        """Returns valid non-empty `ORDER BY pts ASC LIMIT 1` SQL.

        Args:
          requests: Batch of synthesis requests.

        Returns:
          Tuple of mock SQL completion responses aligned 1-to-1 with requests.
        """
        # Parent SQL succeeds with 1 row (`('Leclerc',)`), whereas injecting
        # `AND pts IS NOT NULL` during Stage 4a wipes out all Monaco rows.
        sql = (
            "SELECT name FROM players WHERE country = 'Monaco' "
            "ORDER BY pts ASC LIMIT 1"
        )
        return tuple(
            LLMCompletionResponse(req.request_id, sql, sql, 45, 15)
            for req in requests
        )

    # Create clean-room SQLite fixture where Monaco's only row has `pts IS NULL`.
    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      db_file = root / "players.sqlite"
      conn = sqlite3.connect(db_file)
      try:
        # Populate one NULL-pts row for Monaco and one non-NULL control row.
        conn.execute(
            "CREATE TABLE players ("
            "id INTEGER PRIMARY KEY, name TEXT, country TEXT, pts INTEGER)"
        )
        conn.executemany(
            "INSERT INTO players VALUES (?, ?, ?, ?)",
            [(1, "Leclerc", "Monaco", None), (2, "Gasly", "France", 10)],
        )
        conn.commit()
      finally:
        conn.close()

      # Declare typed DatabaseSchema matching the SQLite `players` table.
      schema = DatabaseSchema(
          db_id="players",
          tables=(
              TableSchema(
                  "players",
                  (
                      ColumnSchema("id", "INTEGER", is_primary_key=True),
                      ColumnSchema("name", "TEXT"),
                      ColumnSchema("country", "TEXT"),
                      ColumnSchema("pts", "INTEGER"),
                  ),
              ),
          ),
      )
      # Run Argus-SQL end-to-end inside SQLiteReadOnlySandbox.
      sandbox = SQLiteReadOnlySandbox(SandboxPolicy(allowed_db_root=root))
      argus = ArgusSQLPipeline(_NullableAscOrderMockLLM(), sandbox)
      task = NLQueryTask(
          question_id="alg_keep_parent",
          db_id="players",
          question="Who is the player from Monaco with the lowest pts?",
          difficulty=QueryDifficulty.MODERATE,
      )
      pred = argus.predict(task, schema, db_file)
      # Stage 4a's `AND pts IS NOT NULL` rewrite yields 0 rows and must NOT
      # clobber the valid `(('Leclerc',),)` parent candidate!
      final_out = sandbox.execute(db_file, pred.final_sql)
      self.assertEqual(final_out.rows, (("Leclerc",),))
      self.assertNotIn("IS NOT NULL", pred.final_sql)

  def test_argus_pipeline_uses_optional_intent_llm_and_falls_back_cleanly(
      self,
  ) -> None:
    """`ArgusSQLPipeline` integrates flash-lite intent tokens and hermetic fallback."""
    from argus_sql.adapters.types import LLMCompletionRequest
    from argus_sql.adapters.types import LLMCompletionResponse

    class _IntentJsonMockLLM:
      """Mock `BatchLLMPort` returning structured JSON for Stage 1."""

      def batch_complete(
          self, requests: tuple[LLMCompletionRequest, ...]
      ) -> tuple[LLMCompletionResponse, ...]:
        """Returns deterministic JSON intent completions for `requests`.

        Args:
          requests: Batch of intent extraction requests.

        Returns:
          Tuple of mock `LLMCompletionResponse` items.
        """
        payload = (
            '{"canonical_question": "How many schools are in California?", '
            '"entities": ["California"], "expected_projection_arity": 1, '
            '"requires_floating_division": false, '
            '"requires_extrema_order": false, "complexity": "simple"}'
        )
        return tuple(
            LLMCompletionResponse(req.request_id, "", payload, 30, 20)
            for req in requests
        )

    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      db_file = root / "schools.sqlite"
      conn = sqlite3.connect(db_file)
      try:
        conn.execute(
            "CREATE TABLE schools (id INTEGER PRIMARY KEY, state TEXT)"
        )
        conn.executemany(
            "INSERT INTO schools VALUES (?, ?)",
            [(1, "California"), (2, "California"), (3, "New York")],
        )
        conn.commit()
      finally:
        conn.close()

      schema = DatabaseSchema(
          db_id="schools",
          tables=(
              TableSchema(
                  "schools",
                  (
                      ColumnSchema("id", "INTEGER", is_primary_key=True),
                      ColumnSchema(
                          "state", "TEXT", value_examples=("California", "New York")
                      ),
                  ),
              ),
          ),
      )
      sandbox = SQLiteReadOnlySandbox(SandboxPolicy(allowed_db_root=root))
      synth = SchemaGuidedDiagnosticSynthesizer()
      task = NLQueryTask(
          question_id="intent_opt_1",
          db_id="schools",
          question="uh how many schools are in calif",
          difficulty=QueryDifficulty.SIMPLE,
      )

      # 1. With valid flash-lite JSON intent extractor, accounts 50 intent tokens.
      pipe_lite = ArgusSQLPipeline(synth, sandbox, intent_llm=_IntentJsonMockLLM())
      pred_lite = pipe_lite.predict(task, schema, db_file)
      self.assertEqual(dict(pred_lite.metadata)["intent_tokens"], "50")
      self.assertEqual(sandbox.execute(db_file, pred_lite.final_sql).rows, ((2,),))

      # 2. Hermetic fallback: passing SQL synthesizer as intent_llm emits 0/non-JSON
      #    and falls back automatically to deterministic FSM canonicalization.
      pipe_fallback = ArgusSQLPipeline(synth, sandbox, intent_llm=synth)
      pred_fb = pipe_fallback.predict(task, schema, db_file)
      self.assertEqual(sandbox.execute(db_file, pred_fb.final_sql).rows, ((2,),))


if __name__ == "__main__":
  unittest.main()
