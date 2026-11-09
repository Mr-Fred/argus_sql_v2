"""Unit tests for BIRD-SQL EX/R-VES, McNemar tests, loader, and ablations."""

import contextlib
import io
import json
import pathlib
import tempfile
import unittest

from argus_sql.adapters.deterministic_synthesizer import SchemaGuidedDiagnosticSynthesizer
from argus_sql.contracts.types import QueryDifficulty
from argus_sql.evaluation.bird_loader import load_bird_split
from argus_sql.evaluation.bird_metrics import compute_mcnemar_p_value
from argus_sql.evaluation.bird_metrics import compute_piecewise_r_ves
from argus_sql.evaluation.bird_metrics import summarize_system_records
from argus_sql.evaluation.exceptions import EvaluationError
from argus_sql.evaluation.feedback_bundle import build_feedback_bundle
from argus_sql.evaluation.feedback_bundle import parse_and_verify_feedback_bundle
from argus_sql.evaluation.feedback_bundle import sanitize_diagnostic_text
from argus_sql.evaluation.feedback_bundle import serialize_feedback_bundle
from argus_sql.evaluation.feedback_cli import main as feedback_cli_main
from argus_sql.evaluation.feedback_triage import cluster_system_failures
from argus_sql.evaluation.feedback_triage import extract_voice_drift_cases
from argus_sql.evaluation.latex_exporter import render_comparative_latex_table
from argus_sql.evaluation.mini_bird_databases import materialize_mini_bird_databases
from argus_sql.evaluation.mini_bird_suite import build_mini_bird_clean_tasks
from argus_sql.evaluation.mini_bird_suite import build_voice_bird_tasks
from argus_sql.evaluation.mini_bird_suite import run_benchmark_suite
from argus_sql.evaluation.types import BenchmarkSuiteReport
from argus_sql.evaluation.types import TaskEvalRecord
from argus_sql.pipelines.argus_pipeline import ArgusSQLPipeline
from argus_sql.pipelines.baseline_pipelines import EnrichedMDDLPipeline
from argus_sql.pipelines.baseline_pipelines import LLMJudgePipeline
from argus_sql.pipelines.baseline_pipelines import RawDDLPipeline
from argus_sql.pipelines.baseline_pipelines import ValueMajorityVotePipeline
from argus_sql.sandbox.sqlite_sandbox import SQLiteReadOnlySandbox
from argus_sql.sandbox.types import SandboxPolicy


class EvaluationModuleTest(unittest.TestCase):
  """Tests R-VES math, McNemar significance, BIRD loading, and ablations."""

  def test_piecewise_r_ves_rewards_faster_valid_queries_only(self) -> None:
    """Invalid SQL gets 0.0; valid faster SQL earns piecewise bonus up to 1.25."""
    # Invalid execution accuracy must always yield 0.0 regardless of speed.
    self.assertEqual(compute_piecewise_r_ves(False, 10.0, 0.1), 0.0)
    self.assertEqual(compute_piecewise_r_ves(True, 10.0, 4.0), 1.25)
    self.assertEqual(compute_piecewise_r_ves(True, 10.0, 8.0), 1.00)
    self.assertEqual(compute_piecewise_r_ves(True, 10.0, 15.0), 0.75)

  def test_mini_bird_gold_sql_validity_and_argus_statistical_significance(
      self,
  ) -> None:
    """All 24 gold queries execute non-emptily; Argus beats B3 & B4 on Voice."""
    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      schemas, paths = materialize_mini_bird_databases(root)
      sandbox = SQLiteReadOnlySandbox(SandboxPolicy(allowed_db_root=root))

      clean_tasks = build_mini_bird_clean_tasks()
      voice_tasks = build_voice_bird_tasks()
      # Verify every single gold SQL in our benchmark executes with >=1 row!
      for task in clean_tasks:
        gold_out = sandbox.execute(paths[task.db_id], task.gold_sql)
        self.assertTrue(gold_out.succeeded, msg=f"Broken gold: {task.question_id}")
        self.assertGreater(gold_out.row_count, 0, msg=f"Empty: {task.question_id}")

      # Evaluate B0, B0+, B3, B4, and Argus-SQL over identical schemas & synth.
      synth = SchemaGuidedDiagnosticSynthesizer()
      pipes = (
          RawDDLPipeline(synth, sandbox),
          EnrichedMDDLPipeline(synth, sandbox),
          ValueMajorityVotePipeline(synth, sandbox),
          LLMJudgePipeline(synth, sandbox),
          ArgusSQLPipeline(synth, sandbox),
      )
      # Run both Clean and spoken Voice-BIRD suites to verify robustness gains.
      clean_rep = run_benchmark_suite(
          "Clean", clean_tasks, pipes, schemas, paths, sandbox
      )
      voice_rep = run_benchmark_suite(
          "Voice", voice_tasks, pipes, schemas, paths, sandbox
      )
      ex_clean = {s.system_name: s.ex_overall for s in clean_rep.summaries}
      ex_voice = {s.system_name: s.ex_overall for s in voice_rep.summaries}
      tok_clean = {s.system_name: s.mean_tokens for s in clean_rep.summaries}

      # Argus-SQL must outperform B0/B0+/B3 on Clean, beat B4 on Voice & tokens!
      self.assertGreater(ex_clean["Argus-SQL"], ex_clean["B0_RawDDL"])
      self.assertGreater(ex_clean["Argus-SQL"], ex_clean["B0+_EnrichedMDDL"])
      self.assertGreater(ex_clean["Argus-SQL"], ex_clean["B3_ValueMajorityVote"])
      self.assertGreater(ex_voice["B4_LLMJudge"], ex_voice["B3_ValueMajorityVote"])
      self.assertGreater(ex_voice["Argus-SQL"], ex_voice["B4_LLMJudge"])
      self.assertLess(tok_clean["Argus-SQL"], tok_clean["B4_LLMJudge"])

      # Paired McNemar tests confirm statistical significance vs B3 and B4!
      rec_map = dict(clean_rep.records_by_system)
      voice_map = dict(voice_rep.records_by_system)
      p_val = compute_mcnemar_p_value(
          rec_map["B3_ValueMajorityVote"], rec_map["Argus-SQL"]
      )
      self.assertLess(p_val, 0.005)
      p_val_b4_voice = compute_mcnemar_p_value(
          voice_map["B4_LLMJudge"], voice_map["Argus-SQL"]
      )
      self.assertLess(p_val_b4_voice, 0.02)
      # Identical record comparisons must yield p = 1.0, and mismatches raise.
      self.assertEqual(
          compute_mcnemar_p_value(rec_map["Argus-SQL"], rec_map["Argus-SQL"]),
          1.0,
      )
      with self.assertRaises(EvaluationError):
        compute_mcnemar_p_value(rec_map["Argus-SQL"][:1], rec_map["Argus-SQL"])

      # Export LaTeX table and confirm B0+ and Argus-SQL rows render cleanly.
      tex = render_comparative_latex_table(
          clean_rep, voice_rep, "Caption", "tab:test"
      )
      self.assertIn(r"B0+\_EnrichedMDDL", tex)
      self.assertIn(r"\textbf{Argus-SQL}", tex)

  def test_bird_loader_parses_official_json_schema_structure(self) -> None:
    """Official BIRD `dev.json` + `dev_tables.json` format loads cleanly."""
    with tempfile.TemporaryDirectory() as tmp:
      bird_root = pathlib.Path(tmp)
      (bird_root / "dev_databases" / "demo").mkdir(parents=True)
      # Write minimal BIRD-formatted `dev_tables.json` and `dev.json`.
      tables_payload = [{
          "db_id": "demo",
          "table_names_original": ["users", "orders"],
          "column_names_original": [[-1, "*"], [0, "uid"], [1, "oid"], [1, "uid"]],
          "column_types": ["text", "integer", "integer", "integer"],
          "primary_keys": [1, 2],
          "foreign_keys": [[3, 1]],
      }]
      dev_payload = [{
          "question_id": 0,
          "db_id": "demo",
          "question": "How many orders?",
          "evidence": "",
          "SQL": "SELECT COUNT(*) FROM orders",
          "difficulty": "simple",
      }]
      # Write a UTF-8 BOM (`\xef\xbb\xbf`) CSV with both column_description and
      # multiline codebook `value_description` matching official BIRD CSV files.
      desc_dir = bird_root / "dev_databases" / "demo" / "database_description"
      desc_dir.mkdir(parents=True)
      bom_csv = (
          b"\xef\xbb\xbforiginal_column_name,column_description,value_description\n"
          b"uid,Unique user identifier,Active = currently enrolled\n"
      )
      (desc_dir / "users.csv").write_bytes(bom_csv)
      (bird_root / "dev_tables.json").write_text(
          json.dumps(tables_payload), encoding="utf-8"
      )
      (bird_root / "dev.json").write_text(
          json.dumps(dev_payload), encoding="utf-8"
      )
      # Verify tasks, FK edges, and merged codebook descriptions deserialize.
      tasks, schemas, sqlite_paths = load_bird_split(bird_root)
      self.assertEqual(len(tasks), 1)
      self.assertEqual(len(schemas["demo"].foreign_keys), 1)
      self.assertEqual(
          schemas["demo"].tables[0].columns[0].description,
          "Unique user identifier | Active = currently enrolled",
      )
      # Running a non-empty task list with 0 materialized SQLite DBs fails fast.
      sandbox = SQLiteReadOnlySandbox(SandboxPolicy(allowed_db_root=bird_root))
      with self.assertRaises(EvaluationError):
        run_benchmark_suite("EmptyDBs", tasks, (), schemas, sqlite_paths, sandbox)

      # Populate a real SQLite DB and verify `value_examples` + set/float EX!
      import sqlite3
      from argus_sql.contracts.types import NLQueryTask, PipelinePrediction
      from argus_sql.evaluation.bird_metrics import evaluate_prediction

      demo_sqlite = bird_root / "dev_databases" / "demo" / "demo.sqlite"
      conn = sqlite3.connect(demo_sqlite)
      try:
        conn.execute("CREATE TABLE users (uid INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE orders (oid INTEGER PRIMARY KEY, uid INTEGER)")
        conn.executemany("INSERT INTO users VALUES (?)", [(101,), (102,)])
        conn.executemany(
            "INSERT INTO orders VALUES (?, ?)", [(1, 101), (2, 101), (3, 102)]
        )
        conn.commit()
      finally:
        conn.close()

      _, hydrated_schemas, _ = load_bird_split(bird_root)
      self.assertEqual(
          hydrated_schemas["demo"].tables[0].columns[0].value_examples,
          ("101", "102"),
      )

      # Verify BIRD set-equivalence (`SELECT uid` == `SELECT DISTINCT uid`) and
      # IEEE-754 hundred-million sum join-order tolerance (`402524570.7200004`).
      set_task = NLQueryTask(
          "s1", "demo", "List user ids", gold_sql="SELECT DISTINCT uid FROM orders"
      )
      set_pred = PipelinePrediction(
          "s1", "demo", "Argus-SQL", "SELECT uid FROM orders", 1.0, 50
      )
      self.assertTrue(
          evaluate_prediction(set_task, set_pred, demo_sqlite, sandbox).ex_correct
      )

      float_task = NLQueryTask(
          "f1", "demo", "Sum", gold_sql="SELECT 402524570.7200004"
      )
      float_pred = PipelinePrediction(
          "f1", "demo", "Argus-SQL", "SELECT 402524570.7199997", 1.0, 50
      )
      self.assertTrue(
          evaluate_prediction(float_task, float_pred, demo_sqlite, sandbox).ex_correct
      )

  def test_sqlite_column_ranges_handle_blank_cells_composite_ym_and_exclusions(
      self,
  ) -> None:
    """Batched MIN..MAX extraction ignores ''/N/A and skips IDs & free text."""
    import sqlite3

    with tempfile.TemporaryDirectory() as tmp:
      bird_root = pathlib.Path(tmp)
      db_dir = bird_root / "dev_databases" / "debit_mini"
      db_dir.mkdir(parents=True)
      sqlite_path = db_dir / "debit_mini.sqlite"
      conn = sqlite3.connect(sqlite_path)
      try:
        conn.execute(
            "CREATE TABLE tx_sample (TransactionID INTEGER PRIMARY KEY, "
            "ChainID INTEGER, account_to INTEGER, Date DATE, SchoolName TEXT, "
            "LowGrade TEXT, FreeRate REAL, Latitude REAL)"
        )
        conn.execute(
            "CREATE TABLE yearmonth (CustomerID INTEGER, Date TEXT, "
            "Consumption REAL, PRIMARY KEY (CustomerID, Date))"
        )
        # Include empty string '' in Date and 'N/A' in FreeRate to verify
        # blank CSV strings and dirty text placeholders never corrupt MIN..MAX!
        conn.executemany(
            "INSERT INTO tx_sample VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, 10, 880011, "", "Alpha Charter", "1", 0.00176, 34.05),
                (2, 20, 990022, "2012-08-23", "Zeta High", "P", 1.0, 37.77),
                (3, 30, 770033, "2012-08-26", "Mid School", "9", "N/A", 38.58),
            ],
        )
        conn.executemany(
            "INSERT INTO yearmonth VALUES (?, ?, ?)",
            [(1, "201112", 12.5), (1, "201311", 450.25)],
        )
        conn.commit()
      finally:
        conn.close()

      # Serialize minimal BIRD `dev_tables.json` metadata with composite PKs.
      tables_json = [{
          "db_id": "debit_mini",
          "table_names_original": ["tx_sample", "yearmonth"],
          "column_names_original": [
              [-1, "*"],
              [0, "TransactionID"],
              [0, "ChainID"],
              [0, "account_to"],
              [0, "Date"],
              [0, "SchoolName"],
              [0, "LowGrade"],
              [0, "FreeRate"],
              [0, "Latitude"],
              [1, "CustomerID"],
              [1, "Date"],
              [1, "Consumption"],
          ],
          # Mix `date`, `text`, `integer`, and `real` declarations as in BIRD.
          "column_types": [
              "text", "integer", "integer", "integer", "date", "text", "text",
              "real", "real", "integer", "text", "real",
          ],
          # `[9, 10]` mirrors `yearmonth(CustomerID, Date)` composite primary key.
          "primary_keys": [1, [9, 10]],
          "foreign_keys": [],
      }]
      # Write metadata JSON files to the temporary split directory and load.
      (bird_root / "dev_tables.json").write_text(
          json.dumps(tables_json), encoding="utf-8"
      )
      (bird_root / "dev.json").write_text("[]", encoding="utf-8")

      # Verify `load_bird_split` hydrates `value_range` across all valid columns.
      _, schemas, _ = load_bird_split(bird_root)
      tx_cols = {c.name: c for c in schemas["debit_mini"].tables[0].columns}
      ym_cols = {c.name: c for c in schemas["debit_mini"].tables[1].columns}

      # Temporal & measure ranges extracted accurately despite '' and 'N/A'.
      self.assertEqual(tx_cols["Date"].value_range, ("2012-08-23", "2012-08-26"))
      self.assertEqual(tx_cols["FreeRate"].value_range, ("0.0018", "1.0"))
      self.assertEqual(ym_cols["Date"].value_range, ("201112", "201311"))
      self.assertEqual(ym_cols["Consumption"].value_range, ("12.5", "450.25"))

      # Surrogate IDs, account_to, coordinates, and free-text/grade strings stay None.
      for excluded in (
          "TransactionID", "ChainID", "account_to", "SchoolName", "LowGrade", "Latitude"
      ):
        self.assertIsNone(tx_cols[excluded].value_range, msg=excluded)
      self.assertIsNone(ym_cols["CustomerID"].value_range)

  def test_bird_loader_auto_resolves_mini_dev_json_and_jsonl(self) -> None:
    """`load_bird_split` auto-discovers `mini_dev_sqlite.jsonl` and lowercase sql."""
    with tempfile.TemporaryDirectory() as tmp:
      bird_root = pathlib.Path(tmp)
      # Create a minimal database directory without `dev.json` present.
      (bird_root / "dev_databases" / "demo").mkdir(parents=True)
      tables_payload = [{
          "db_id": "demo",
          "table_names_original": ["schools"],
          "column_names_original": [[-1, "*"], [0, "cds"]],
          "column_types": ["text", "text"],
          "primary_keys": [1],
          "foreign_keys": [],
      }]
      # Serialize two Mini-Dev records in JSONL format mixing `sql` and `SQL`.
      jsonl_lines = "\n".join((
          json.dumps({
              "question_id": 10,
              "db_id": "demo",
              "question": "Count schools.",
              "sql": "SELECT COUNT(*) FROM schools",
              "difficulty": "moderate",
          }),
          json.dumps({
              "question_id": 11,
              "db_id": "demo",
              "question": "List cds codes.",
              "SQL": "SELECT cds FROM schools",
              "difficulty": "simple",
          }),
      ))
      # Persist schema metadata and JSONL split to disk and verify auto-loading.
      (bird_root / "dev_tables.json").write_text(
          json.dumps(tables_payload), encoding="utf-8"
      )
      (bird_root / "mini_dev_sqlite.jsonl").write_text(
          jsonl_lines, encoding="utf-8"
      )
      tasks, _, _ = load_bird_split(bird_root)
      self.assertEqual(len(tasks), 2)
      self.assertEqual(tasks[0].gold_sql, "SELECT COUNT(*) FROM schools")
      self.assertEqual(tasks[1].gold_sql, "SELECT cds FROM schools")

      # Verify nested `MINIDEV/` archive layout + `sqlite/mini_dev_sqlite.json`.
      nested = pathlib.Path(tmp) / "outer_extract" / "MINIDEV"
      (nested / "dev_databases" / "demo").mkdir(parents=True)
      (nested / "sqlite").mkdir(parents=True)
      (nested / "dev_tables.json").write_text(
          json.dumps(tables_payload), encoding="utf-8"
      )
      (nested / "sqlite" / "mini_dev_sqlite.json").write_text(
          json.dumps([{
              "question_id": 99,
              "db_id": "demo",
              "question": "Count schools.",
              "SQL": "SELECT COUNT(*) FROM schools",
          }]),
          encoding="utf-8",
      )
      nested_tasks, _, _ = load_bird_split(
          pathlib.Path(tmp) / "outer_extract",
          questions_filename="mini_dev_sqlite.json",
      )
      self.assertEqual(len(nested_tasks), 1)
      self.assertEqual(nested_tasks[0].question_id, "99")

  def test_feedback_bundle_sanitizes_paths_verifies_sha_and_ingests(
      self,
  ) -> None:
    """Feedback bundles scrub machine paths, detect tampering, and ingest."""
    # Assemble synthetic laptop path and secret fragments dynamically.
    mac_db = "".join(("/", "Users", "/laptop/data/demo.sqlite"))
    fake_tok = "sk-" + ("B2c3D4e5" * 4)
    sql_div = "SELECT CAST(SUM(a) AS REAL) / NULLIF(SUM(b), 0) FROM schools"

    # Verify SQL division stays untouched while OS paths & tokens redact.
    self.assertEqual(sanitize_diagnostic_text(sql_div), sql_div)
    scrubbed_err = sanitize_diagnostic_text(
        f"unable to open {mac_db} with key {fake_tok}", single_line=True
    )
    self.assertIn("<PATH_REDACTED>", scrubbed_err)
    self.assertIn("<CREDENTIAL_REDACTED>", scrubbed_err)
    self.assertNotIn("laptop", scrubbed_err)

    # Construct Clean + Voice synthetic suites with failure and voice drift.
    clean_recs = (
        # Passing simple query exercising floating-point division SQL.
        TaskEvalRecord(
            question_id="q1",
            db_id="schools",
            difficulty=QueryDifficulty.SIMPLE,
            is_voice=False,
            ex_correct=True,
            r_ves_score=1.0,
            latency_ms=12.0,
            total_tokens=200,
            failure_category="CLEAN_MATCH",
            predicted_sql=sql_div,
            selected_strategy="voice_fast_path",
        ),
        # Failing challenging query embedding a laptop path inside error text.
        TaskEvalRecord(
            question_id="q2",
            db_id="schools",
            difficulty=QueryDifficulty.CHALLENGING,
            is_voice=False,
            ex_correct=False,
            r_ves_score=0.0,
            latency_ms=25.0,
            total_tokens=450,
            failure_category="SYNTAX_OR_JOIN_KEY_ERROR",
            predicted_sql="SELECT x FROM ghost_table",
            error_summary=f"no such table: ghost_table in {mac_db}",
            selected_strategy="decomposed_cot",
        ),
    )
      # Spoken Voice split uses realistic `q1_voice` ID from `perturb_task_for_voice`.
    voice_recs = (
        TaskEvalRecord(
            question_id="q1_voice",
            db_id="schools",
            difficulty=QueryDifficulty.SIMPLE,
            is_voice=True,
            ex_correct=False,
            r_ves_score=0.0,
            latency_ms=15.0,
            total_tokens=210,
            failure_category="EMPTY_RESULT_VALUE_DRIFT",
            predicted_sql="SELECT * FROM schools WHERE name = 'um alameda'",
        ),
        TaskEvalRecord(
            question_id="q2_voice",
            db_id="schools",
            difficulty=QueryDifficulty.CHALLENGING,
            is_voice=True,
            ex_correct=False,
            r_ves_score=0.0,
            latency_ms=25.0,
            total_tokens=450,
            failure_category="SYNTAX_OR_JOIN_KEY_ERROR",
            predicted_sql="SELECT x FROM ghost_table",
        ),
    )
    # Package Clean and Voice suites with summarized leaderboard metrics.
    clean_suite = BenchmarkSuiteReport(
        suite_name="BIRD Clean",
        summaries=(summarize_system_records("Argus-SQL", clean_recs),),
        records_by_system=(("Argus-SQL", clean_recs),),
    )
    voice_suite = BenchmarkSuiteReport(
        suite_name="BIRD Voice",
        summaries=(summarize_system_records("Argus-SQL", voice_recs),),
        records_by_system=(("Argus-SQL", voice_recs),),
    )

    # Verify failure clustering and Clean-to-Voice (`q1` <-> `q1_voice`) alignment.
    clusters = cluster_system_failures(clean_recs)
    self.assertEqual(clusters[0].sub_signature, "MISSING_SCHEMA_TABLE")
    drifts = extract_voice_drift_cases(clean_recs, voice_recs)
    self.assertEqual(len(drifts), 1)
    self.assertEqual(drifts[0].question_id, "q1")

    # Build, serialize, and verify cryptographic round-trip + LaTeX escaping.
    bundle = build_feedback_bundle(
        suites=(clean_suite, voice_suite),
        provider="vertex",
        model_label="Qwen2.5_Coder_32B",
        split_name=f"{mac_db}/mini_dev_sqlite.json",
        git_commit="abc123def456",
        created_utc="2026-11-08T06:00:00Z",
    )
    # Confirm basename stripping removed all laptop directory segments.
    self.assertEqual(bundle.split_name, "mini_dev_sqlite.json")
    raw_json = serialize_feedback_bundle(bundle)
    self.assertNotIn("laptop", raw_json)
    verified = parse_and_verify_feedback_bundle(raw_json)
    self.assertEqual(verified.payload_sha256, bundle.payload_sha256)

    # Flipping a metric, injecting NaN, or injecting negative tokens must fail.
    tampered = raw_json.replace('"total_tokens": 200', '"total_tokens": 999')
    with self.assertRaises(EvaluationError):
      parse_and_verify_feedback_bundle(tampered)
    nan_raw = raw_json.replace('"r_ves_score": 1.0', '"r_ves_score": NaN')
    with self.assertRaises(EvaluationError):
      parse_and_verify_feedback_bundle(nan_raw)
    neg_raw = raw_json.replace('"total_tokens": 200', '"total_tokens": -5')
    with self.assertRaises(EvaluationError):
      parse_and_verify_feedback_bundle(neg_raw)

    # Injecting an unredacted token or JSON-escaped Windows path must fail closed.
    leaked_raw = raw_json.replace("abc123def456", fake_tok)
    with self.assertRaises(EvaluationError):
      parse_and_verify_feedback_bundle(leaked_raw)
    # JSON-escaped double backslashes and space-separated Windows user dirs fail.
    win_json_leak = "".join(("C:\\\\", "Users", "\\\\First Last\\\\schools.sqlite"))
    with self.assertRaises(EvaluationError):
      parse_and_verify_feedback_bundle(raw_json.replace("abc123def456", win_json_leak))
    # Verify sanitizer strips spaced Windows user paths and file:/// URIs cleanly.
    win_space_err = "".join(("open ", "C:\\", "Users", "\\First Last\\db.sqlite failed"))
    win_uri_err = "".join(("open file:///", "C:/", "Users", "/FirstLast/db.sqlite failed"))
    self.assertEqual(
        sanitize_diagnostic_text(win_space_err), "open <PATH_REDACTED> failed"
    )
    self.assertEqual(
        sanitize_diagnostic_text(win_uri_err), "open <PATH_REDACTED> failed"
    )

    # Verify CLI `ingest` and `verify` end-to-end inside a confined workspace.
    with tempfile.TemporaryDirectory() as tmp:
      ws = pathlib.Path(tmp)
      q_file = ws / "audit_import" / "bird_feedback_bundle.json"
      q_file.parent.mkdir(parents=True)
      q_file.write_text(raw_json, encoding="utf-8")
      promoted = ws / "results" / "feedback" / "bird_feedback_bundle.json"
      report_md = ws / "results" / "feedback" / "triage_report.md"
      latex_dir = ws / "paper" / "tables"

      # Capture CLI stdout/stderr so passing test suites stay visually clean.
      out_buf = io.StringIO()
      err_buf = io.StringIO()
      # Run `ingest` to promote quarantine bundle and emit LaTeX + Markdown.
      with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
        rc_ing = feedback_cli_main([
            "ingest",
            "--repo-root",
            str(ws),
            "--bundle",
            str(q_file),
            "--promote-to",
            str(promoted),
            "--report-out",
            str(report_md),
            "--latex-out-dir",
            str(latex_dir),
        ])
      self.assertEqual(rc_ing, 0)
      self.assertIn("Verified feedback bundle", out_buf.getvalue())
      self.assertTrue(promoted.is_file())
      triage_txt = report_md.read_text(encoding="utf-8")
      self.assertIn("MISSING_SCHEMA_TABLE", triage_txt)
      self.assertIn("QID `q1`", triage_txt)
      tex_txt = (latex_dir / "bird_live.tex").read_text(encoding="utf-8")
      self.assertIn(r"Qwen2.5\_Coder\_32B", tex_txt)

      # Single-suite Clean-only bundle must render `--` for omitted Voice EX%.
      clean_only = build_feedback_bundle(
          suites=(clean_suite,),
          provider="openai",
          model_label="gpt-4o",
          split_name="dev.json",
          created_utc="2026-11-08T06:00:00Z",
      )
      clean_only_path = ws / "audit_import" / "clean_only.json"
      clean_only_path.write_text(
          serialize_feedback_bundle(clean_only), encoding="utf-8"
      )
      with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
        rc_single = feedback_cli_main([
            "ingest",
            "--repo-root",
            str(ws),
            "--bundle",
            str(clean_only_path),
            "--report-out",
            str(report_md),
            "--latex-out-dir",
            str(latex_dir),
            "--latex-filename",
            "single.tex",
        ])
      self.assertEqual(rc_single, 0)
      self.assertIn(
          r"\textbf{--}",
          (latex_dir / "single.tex").read_text(encoding="utf-8"),
      )

      # Path traversal outside `--repo-root` must fail closed with exit code 1.
      err_escape = io.StringIO()
      with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_escape):
        rc_escape = feedback_cli_main([
            "ingest",
            "--repo-root",
            str(ws),
            "--bundle",
            "../outside_bundle.json",
        ])
      self.assertEqual(rc_escape, 1)
      self.assertIn("escapes allowed root boundary", err_escape.getvalue())

      # Run `verify` gate across `results/feedback/` just as GitHub CI does.
      ver_out = io.StringIO()
      with contextlib.redirect_stdout(ver_out), contextlib.redirect_stderr(err_buf):
        rc_ver = feedback_cli_main([
            "verify",
            "--repo-root",
            str(ws),
            "--feedback-dir",
            str(ws / "results" / "feedback"),
        ])
      self.assertEqual(rc_ver, 0)
      self.assertIn("Verified 1 feedback bundle(s)", ver_out.getvalue())

  def test_stratified_sampling_and_live_run_checkpoint_resume(self) -> None:
    """Stratified sampling spans all DBs/tiers and checkpoints skip LLM re-runs."""
    from argus_sql.evaluation.live_checkpoint import LiveRunCheckpointer
    from argus_sql.evaluation.live_checkpoint import resolve_checkpoint_path
    from argus_sql.evaluation.live_checkpoint import select_stratified_bird_tasks

    clean_tasks = build_mini_bird_clean_tasks()
    # Selecting 3 tasks from 3 databases must pick exactly 1 per DB across all 3 tiers.
    strat_slice = select_stratified_bird_tasks(clean_tasks, limit=3)
    self.assertEqual(len(strat_slice), 3)
    self.assertEqual(len({t.db_id for t in strat_slice}), 3)
    self.assertEqual(len({t.difficulty for t in strat_slice}), 3)

    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      schemas, paths = materialize_mini_bird_databases(root)
      sandbox = SQLiteReadOnlySandbox(SandboxPolicy(allowed_db_root=root))
      ckpt_file = resolve_checkpoint_path(
          root / "live_runs", "mini_dev_sqlite.json", "vertex", "gemini-3.7-flash", 3, True
      )
      ckpt = LiveRunCheckpointer(ckpt_file)
      synth = SchemaGuidedDiagnosticSynthesizer()
      pipe = ArgusSQLPipeline(synth, sandbox)

      # First run writes 3 records to disk.
      rep1 = run_benchmark_suite(
          "Clean", strat_slice, (pipe,), schemas, paths, sandbox, checkpointer=ckpt
      )
      self.assertTrue(ckpt_file.is_file())
      # Reloading from disk must replay all 3 records without invoking `predict`.
      reloaded_ckpt = LiveRunCheckpointer(ckpt_file)
      rep2 = run_benchmark_suite(
          "Clean",
          strat_slice,
          (pipe,),
          schemas,
          paths,
          sandbox,
          checkpointer=reloaded_ckpt,
      )
      self.assertEqual(rep1.summaries[0].ex_overall, rep2.summaries[0].ex_overall)
      # Evicting `Argus-SQL` clears its 3 cached entries and compacts JSONL.
      self.assertEqual(reloaded_ckpt.invalidate_system("Argus-SQL"), 3)
      self.assertIsNone(
          reloaded_ckpt.get("Clean", "Argus-SQL", strat_slice[0].question_id)
      )

      # Legacy `B0_ZeroShotDDL` records migrate transparently to `B0+_EnrichedMDDL`
      # on reload, whereas `B3_CHESSValueConsistency` does NOT alias onto
      # steelmanned `B3_ValueMajorityVote` so B3 re-evaluates with raw entities.
      sample_rec = rep1.records_by_system[0][1][0]
      reloaded_ckpt.record("Clean", "B0_ZeroShotDDL", sample_rec)
      reloaded_ckpt.record("Clean", "B3_CHESSValueConsistency", sample_rec)
      migrated_ckpt = LiveRunCheckpointer(ckpt_file)
      self.assertIsNotNone(
          migrated_ckpt.get("Clean", "B0+_EnrichedMDDL", sample_rec.question_id)
      )
      self.assertIsNone(
          migrated_ckpt.get(
              "Clean", "B3_ValueMajorityVote", sample_rec.question_id
          )
      )
      self.assertEqual(migrated_ckpt.invalidate_system("B0+_EnrichedMDDL"), 1)
      migrated_ckpt.clear_all()
      self.assertFalse(ckpt_file.exists())


if __name__ == "__main__":
  unittest.main()
