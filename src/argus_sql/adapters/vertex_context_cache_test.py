"""Hermetic offline unit tests for Vertex AI Context Caching (`R1`, `R5`)."""

import ast
import dataclasses
from email.message import Message
import http.client
import json
import os
import pathlib
import sqlite3
import time
import unittest
from unittest import mock
import urllib.error

from argus_sql.adapters.exceptions import LLMAdapterAuthError
from argus_sql.adapters.exceptions import LLMAdapterTransportError
from argus_sql.adapters.types import LLMCompletionRequest
from argus_sql.adapters.vertex_context_cache import ContextCacheAdapter
from argus_sql.adapters.vertex_context_cache import DEFAULT_SQLITE_SYSTEM_PROMPT
from argus_sql.adapters.vertex_context_cache import build_cached_content_payload
from argus_sql.adapters.vertex_context_cache import build_db_cached_context_text
from argus_sql.adapters.vertex_context_cache import strip_cached_schema_from_user_prompt
from argus_sql.adapters.vertex_context_cache import validate_cache_ttl_seconds
from argus_sql.adapters.vertex_gemini_client import VertexGeminiBatchClient
from argus_sql.contracts.schema_ddl_renderer import render_schema_ddl
from argus_sql.contracts.types import ColumnSchema
from argus_sql.contracts.types import DatabaseSchema
from argus_sql.contracts.types import ForeignKeyEdge
from argus_sql.contracts.types import TableSchema
from argus_sql.pipelines.judge_baseline import LLM_JUDGE_SYSTEM_PROMPT
from argus_sql.pipelines.prompt_templates import SQLITE_SYSTEM_PROMPT
from argus_sql.pipelines.prompt_templates import build_strategy_request


# Lightweight duck-typed cell entry mirroring `ValueIndexEntry` without imports.
@dataclasses.dataclass(frozen=True)
class _FakeCellEntry:
  """Synthetic cell index entry exposing `.table_name`, `.column_name`, `.raw_value`."""

  # Match exact attribute names inspected by `_extract_cell_value_dictionary`.
  table_name: str
  column_name: str
  raw_value: str


# Lightweight duck-typed retriever exposing `.entries` like `CellValueRetriever`.
@dataclasses.dataclass(frozen=True)
class _FakeRetriever:
  """Synthetic cell value retriever container for duck-typing verification."""

  # Provide immutable tuple of cell entries without importing `argus_sql.linking`.
  entries: tuple[_FakeCellEntry, ...]


def _sample_two_table_schema() -> DatabaseSchema:
  """Builds a representative 2-table schema with descriptions, ranges, and FKs."""
  # Include metadata descriptions, numeric bounds, and sample literals on columns.
  # Table 1 (`schools`) models parent entity metadata and categorical counties.
  schools_tbl = TableSchema(
      name="schools",
      columns=(
          ColumnSchema("cds_code", "TEXT", is_primary_key=True, description="School ID"),
          ColumnSchema(
              "county",
              "TEXT",
              description="California county name",
              value_examples=("Alameda", "Fresno"),
          ),
      ),
  )
  # Table 2 (`frpm`) models child numeric metrics with explicit min/max range.
  frpm_tbl = TableSchema(
      name="frpm",
      columns=(
          ColumnSchema("cds_code", "TEXT", is_primary_key=True, description="FK to schools"),
          ColumnSchema(
              "free_meals",
              "INTEGER",
              description="Free meal count",
              value_range=("0", "1500"),
          ),
      ),
  )
  # Connect `frpm.cds_code -> schools.cds_code` to exercise FK DDL rendering.
  fk = ForeignKeyEdge("frpm", "cds_code", "schools", "cds_code")
  # Return immutable `DatabaseSchema` fixture shared across R1 unit test cases.
  return DatabaseSchema(
      db_id="california_schools",
      tables=(schools_tbl, frpm_tbl),
      foreign_keys=(fk,),
  )


# Test suite exercising per-database context caching, stripping, and teardown.
class VertexContextCacheTest(unittest.TestCase):
  """Covers all 17 R1/R5 context caching invariants 100% offline."""
  # Every test injects an in-memory transport so zero network sockets open.
  # Synthetic bearer strings avoid OAuth token prefixes to pass clean-room scans.
  # Assertions verify both wire JSON schema and Hexagonal import isolation.
  # Multi-table DDL prompts verify blank-line boundaries never leak trailing DDL.
  # Exception-unwind tests verify `DELETE` executes without masking errors.
  # System-prompt SHA-256 checks verify `B4_LLMJudge` bypasses SQL cache safely.
  # Cross-database tests verify `_cache_key` isolates identical question strings.

  def test_build_cached_context_from_schema_and_duck_typed_retriever(self) -> None:
    """Builds DDL, descriptions, and 3-gram dictionary from schema + `.entries`."""
    # Confirm `DEFAULT_SQLITE_SYSTEM_PROMPT` matches `SQLITE_SYSTEM_PROMPT` byte-for-byte.
    self.assertEqual(DEFAULT_SQLITE_SYSTEM_PROMPT, SQLITE_SYSTEM_PROMPT)

    # Construct a 2-table schema paired with duplicate and excess cell literals.
    schema = _sample_two_table_schema()
    retriever = _FakeRetriever(
        entries=(
            _FakeCellEntry("schools", "county", "Alameda"),
            _FakeCellEntry("schools", "county", "Alameda"),
            _FakeCellEntry("schools", "county", "Fresno"),
            _FakeCellEntry("schools", "county", "Los Angeles"),
        )
    )
    # Cap `max_values_per_column=2` to verify deduplication and per-column bounding.
    ctx_text = build_db_cached_context_text(
        "california_schools",
        schema=schema,
        cell_values=retriever,
        max_values_per_column=2,
    )
    # Verify all 3 mandatory R1 context sections appear in deterministic order.
    self.assertIn("Database ID: california_schools", ctx_text)
    self.assertIn("CREATE TABLE schools", ctx_text)
    self.assertIn("CREATE TABLE frpm", ctx_text)
    self.assertIn("Table and Column Descriptions:", ctx_text)
    self.assertIn("California county name", ctx_text)
    self.assertIn("range: 0..1500", ctx_text)
    # Confirm duplicate `'Alameda'` collapsed and 3rd literal `'Los Angeles'` truncated.
    self.assertIn("3-Gram Retrieved Cell Value Dictionary:", ctx_text)
    self.assertIn("- schools.county: ['Alameda', 'Fresno']", ctx_text)
    self.assertNotIn("Los Angeles", ctx_text)

  def test_build_cached_context_from_explicit_ddl_and_mappings(self) -> None:
    """Supports raw DDL strings plus dict/tuple/nested description & cell maps."""
    # Pass raw DDL alongside tuple-keyed descriptions and nested + flat cell maps.
    raw_ddl = "CREATE TABLE loans (loan_id INTEGER PRIMARY KEY, status TEXT);"
    ctx_text = build_db_cached_context_text(
        "financial_db",
        schema_ddl=raw_ddl,
        descriptions={("loans", "status"): "Repayment contract status"},
        cell_values={"loans": {"status": ["A", "B"]}},
        cell_value_dict={"loans.status": ["B", "C"]},
    )
    # Verify merged cell value dictionary deduplicated `'B'` across both sources.
    self.assertIn("Database ID: financial_db", ctx_text)
    self.assertIn(raw_ddl, ctx_text)
    self.assertIn("- loans.status: Repayment contract status", ctx_text)
    self.assertIn("- loans.status: ['A', 'B', 'C']", ctx_text)

    # Blank `db_id`, invalid cap, or missing both `schema` and `schema_ddl` raise.
    with self.assertRaises(ValueError):
      build_db_cached_context_text("   ", schema_ddl=raw_ddl)
    with self.assertRaises(ValueError):
      build_db_cached_context_text("financial_db")
    with self.assertRaises(ValueError):
      build_db_cached_context_text("financial_db", schema_ddl=raw_ddl, max_values_per_column=0)

  def test_ttl_validation_enforces_30_to_60_minute_window(self) -> None:
    """TTL validator accepts 1800..3600s and rejects out-of-range or bool values."""
    # Verify boundary values 1800s (30m) and 3600s (60m) serialize as `<n>s`.
    self.assertEqual(validate_cache_ttl_seconds(1800), 1800)
    self.assertEqual(validate_cache_ttl_seconds(3600), 3600)
    # Ensure `CachedContent` payload sets `"ttl"` and omits `"expireTime"`.
    payload_30m = build_cached_content_payload(
        project_id="demo-eval-project-01",
        location="global",
        model_name="gemini-3.7-flash",
        db_id="california_schools",
        cached_context_text="Database Schema:\nCREATE TABLE t (id INT);",
        ttl_seconds=1800,
    )
    self.assertEqual(payload_30m["ttl"], "1800s")
    self.assertNotIn("expireTime", payload_30m)

    # Out-of-bounds integers and booleans must raise `ValueError`.
    for bad_ttl in (1799, 3601, 0, -60, True, False):
      with self.assertRaises(ValueError):
        validate_cache_ttl_seconds(bad_ttl)

  def test_strip_cached_schema_multi_table_synthesis_and_evidence(self) -> None:
    """Strips ALL `\n\n`-separated CREATE TABLE blocks while retaining query/instructions."""
    # Include `);` inside an inline `--` comment to test comment-safe parsing.
    multi_table_prompt = (
        "Database Schema:\n"
        "CREATE TABLE schools (\n"
        "  cds_code TEXT PRIMARY KEY, -- key (pk);\n"
        "  county TEXT\n"
        ");\n\n"
        "CREATE TABLE frpm (\n"
        "  cds_code TEXT PRIMARY KEY,\n"
        "  free_meals INTEGER\n"
        ");\n\n"
        "Question: How many schools are in Alameda?\n"
        "External Evidence: county = 'Alameda'\n\n"
        "Instruction: Output SQL only."
    )
    # Confirm both tables are stripped while Question, Evidence, & Instruction stay.
    stripped = strip_cached_schema_from_user_prompt(multi_table_prompt)
    self.assertEqual(
        stripped,
        "Question: How many schools are in Alameda?\n"
        "External Evidence: county = 'Alameda'\n\n"
        "Instruction: Output SQL only.",
    )
    self.assertNotIn("CREATE TABLE", stripped)

  def test_strip_cached_schema_preserves_repair_preamble_and_diagnostics(self) -> None:
    """Excises `Database Schema:` inside Stage 4b / `B2` repair prompts cleanly."""
    # Mirror exact `sql_repairer.py:build_diagnostic_repair_prompt` layout.
    repair_prompt = (
        "You are repairing a SQLite query that failed semantic or execution checks.\n\n"
        "Database Schema:\n"
        "CREATE TABLE schools (cds_code TEXT PRIMARY KEY);\n\n"
        "CREATE TABLE frpm (cds_code TEXT PRIMARY KEY, free_meals INTEGER);\n\n"
        "Question: Total free meals?\n\n"
        "Previous SQL:\n"
        "SELECT SUM(bad_col) FROM frpm\n\n"
        "Detected Issues:\n"
        "- [EXECUTION_ERROR] no such column: bad_col\n\n"
        "Return ONLY the corrected read-only SQLite SELECT query inside a ```sql ... ``` code block."
    )
    # Verify repair directive, Question, Previous SQL, and Detected Issues remain.
    stripped = strip_cached_schema_from_user_prompt(repair_prompt)
    self.assertTrue(
        stripped.startswith(
            "You are repairing a SQLite query that failed semantic or execution checks.\n\n"
            "Question: Total free meals?"
        )
    )
    self.assertNotIn("Database Schema:", stripped)
    self.assertNotIn("CREATE TABLE", stripped)
    self.assertIn("Detected Issues:\n- [EXECUTION_ERROR]", stripped)

  def test_strip_cached_schema_idempotent_and_non_empty_guard(self) -> None:
    """Schema-free prompts stay unchanged and DDL-only input never returns empty."""
    # Stage 1 intent extraction prompts have no `Database Schema:` header.
    intent_prompt = "Question: Count active accounts.\nJSON:"
    self.assertEqual(strip_cached_schema_from_user_prompt(intent_prompt), intent_prompt)

    # Synthetic prompt without `Question:` label still strips `CREATE TABLE` DDL.
    unlabeled = (
        "Database Schema:\n"
        "CREATE TABLE t1 (id INT);\n\n"
        "CREATE TABLE t2 (id INT);\n\n"
        "Select count from t1."
    )
    self.assertEqual(strip_cached_schema_from_user_prompt(unlabeled), "Select count from t1.")

    # Degenerate DDL-only input falls back to non-empty text rather than `""`.
    ddl_only = "Database Schema:\nCREATE TABLE t1 (id INT);"
    self.assertTrue(bool(strip_cached_schema_from_user_prompt(ddl_only)))

  def test_context_manager_create_generate_and_delete_lifecycle(self) -> None:
    """Verifies `cachedContents.create` -> `generateContent` -> `cachedContents.delete`."""
    # Record `cachedContents` HTTP calls and return numeric GCP project ID in `name`.
    transport_log: list[tuple[str, str, dict[str, object] | None]] = []
    cache_resource = "projects/123456789012/locations/global/cachedContents/cache-001"

    def _mock_cache_http(
        method: str,
        url: str,
        payload_bytes: bytes | None,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> tuple[int, bytes]:
      """Records `cachedContents` lifecycle requests hermetically."""
      del timeout_sec
      # Confirm `x-goog-user-project` bills the configured project ID.
      self.assertEqual(headers["x-goog-user-project"], "demo-eval-project-01")
      parsed = json.loads(payload_bytes.decode("utf-8")) if payload_bytes else None
      transport_log.append((method, url, parsed))
      if method == "POST":
        return (
            200,
            json.dumps({
                "name": cache_resource,
                "expireTime": "2026-11-11T06:30:00Z",
                "usageMetadata": {"totalTokenCount": 6144},
            }).encode("utf-8"),
        )
      return (200, b"{}")

    # Capture outbound `generateContent` payloads without opening sockets.
    gen_payloads: list[dict[str, object]] = []

    def _mock_post_generate(payload_bytes: bytes, bearer: str) -> dict[str, object]:
      """Captures `generateContent` JSON bodies sent by `VertexGeminiBatchClient`."""
      self.assertEqual(bearer, "synthetic-test-bearer")
      gen_payloads.append(json.loads(payload_bytes.decode("utf-8")))
      return {
          "candidates": [{"content": {"parts": [{"text": "```sql\nSELECT 42;\n```"}]}}],
          "usageMetadata": {"promptTokenCount": 24, "candidatesTokenCount": 8},
      }

    # Build hermetic online Vertex client and pair `zero_shot` + `decomposed_cot`.
    vertex_llm = VertexGeminiBatchClient(
        project_id="demo-eval-project-01",
        token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
    )
    schema = _sample_two_table_schema()
    user_body = (
        "Database Schema:\nCREATE TABLE schools (cds_code TEXT);\n\n"
        "Question: Count schools?\n\nInstruction: SQL only."
    )
    req_zero = LLMCompletionRequest(
        "q_zero", SQLITE_SYSTEM_PROMPT, user_body, strategy_tag="zero_shot"
    )
    req_cot = LLMCompletionRequest(
        "q_cot", SQLITE_SYSTEM_PROMPT, user_body, strategy_tag="decomposed_cot"
    )

    # Execute full `with ContextCacheAdapter.for_database(...)` lifecycle.
    with mock.patch.object(vertex_llm, "_post_generate_once", side_effect=_mock_post_generate):
      with ContextCacheAdapter.for_database(
          vertex_llm,
          "california_schools",
          schema=schema,
          ttl_seconds=1800,
          http_transport=_mock_cache_http,
      ) as cached_llm:
        self.assertIsNotNone(cached_llm.active_cache)
        assert cached_llm.active_cache is not None
        self.assertEqual(cached_llm.active_cache.total_token_count, 6144)
        resps = cached_llm.batch_complete((req_zero, req_cot))

    # Verify `POST` create and `DELETE` teardown fired in order and cleared handle.
    self.assertIsNone(cached_llm.active_cache)
    self.assertEqual(len(transport_log), 2)
    self.assertEqual(transport_log[0][0], "POST")
    self.assertTrue(transport_log[0][1].endswith("/locations/global/cachedContents"))
    create_body = transport_log[0][2]
    assert isinstance(create_body, dict)
    self.assertEqual(create_body["ttl"], "1800s")
    self.assertIn("systemInstruction", create_body)
    self.assertEqual(transport_log[1][0], "DELETE")
    self.assertTrue(transport_log[1][1].endswith(f"/v1beta1/{cache_resource}"))

    # Verify `generateContent` attached `cachedContent`, omitted `systemInstruction`
    # and `CREATE TABLE`, and routed `thinkingBudget` `0` vs `>= 2048`.
    self.assertEqual(len(resps), 2)
    self.assertEqual(len(gen_payloads), 2)
    budgets_seen: set[int] = set()
    for pay in gen_payloads:
      # Vertex REST forbids inline `systemInstruction` alongside `cachedContent`.
      self.assertEqual(pay["cachedContent"], cache_resource)
      self.assertNotIn("systemInstruction", pay)
      contents_str = json.dumps(pay["contents"])
      self.assertNotIn("CREATE TABLE", contents_str)
      self.assertIn("Question: Count schools?", contents_str)
      gen_cfg = pay["generationConfig"]
      assert isinstance(gen_cfg, dict)
      think_cfg = gen_cfg["thinkingConfig"]
      assert isinstance(think_cfg, dict)
      budgets_seen.add(int(think_cfg["thinkingBudget"]))
    self.assertEqual(budgets_seen, {0, 2048})

  def test_context_manager_guarantees_delete_when_body_raises(self) -> None:
    """`cachedContents.delete` still executes when an exception escapes `with`."""
    # Track HTTP method sequence when benchmark body raises mid-context.
    methods_seen: list[str] = []

    def _mock_http(
        method: str,
        url: str,
        payload_bytes: bytes | None,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> tuple[int, bytes]:
      """Tracks `POST` create and `DELETE` teardown calls during exception unwind."""
      del url, payload_bytes, headers, timeout_sec
      methods_seen.append(method)
      if method == "POST":
        return (
            200,
            json.dumps({
                "name": "projects/123456789012/locations/global/cachedContents/c-err"
            }).encode("utf-8"),
        )
      return (200, b"{}")

    vertex_llm = VertexGeminiBatchClient(
        project_id="demo-eval-project-01",
        token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
    )
    # Confirm `RuntimeError` propagates unmodified after `DELETE` fires once.
    with self.assertRaisesRegex(RuntimeError, "simulated suite failure"):
      with ContextCacheAdapter(
          vertex_llm,
          "california_schools",
          schema=_sample_two_table_schema(),
          http_transport=_mock_http,
      ):
        raise RuntimeError("simulated suite failure")
    self.assertEqual(methods_seen, ["POST", "DELETE"])

  def test_delete_404_is_idempotent_and_delete_503_never_masks_body_exception(
      self,
  ) -> None:
    """HTTP 404 on DELETE is idempotent; HTTP 503 on DELETE never masks body error."""
    # Construct shared hermetic client for teardown edge-case verification.
    vertex_llm = VertexGeminiBatchClient(
        project_id="demo-eval-project-01",
        token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
    )

    def _http_404_on_delete(
        method: str,
        url: str,
        payload_bytes: bytes | None,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> tuple[int, bytes]:
      """Simulates expired TTL (`HTTP 404`) during `cachedContents.delete`."""
      del url, payload_bytes, headers, timeout_sec
      if method == "POST":
        return (
            200,
            json.dumps({
                "name": "projects/123456789012/locations/global/cachedContents/c-404"
            }).encode("utf-8"),
        )
      return (404, b'{"error": "already expired"}')

    # Case A: HTTP 404 on normal exit succeeds idempotently.
    with ContextCacheAdapter(
        vertex_llm,
        "california_schools",
        schema=_sample_two_table_schema(),
        http_transport=_http_404_on_delete,
    ):
      pass

    def _http_503_on_delete(
        method: str,
        url: str,
        payload_bytes: bytes | None,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> tuple[int, bytes]:
      """Simulates persistent `HTTP 503` during `cachedContents.delete`."""
      del url, payload_bytes, headers, timeout_sec
      if method == "POST":
        return (
            200,
            json.dumps({
                "name": "projects/123456789012/locations/global/cachedContents/c-503"
            }).encode("utf-8"),
        )
      return (503, b"unavailable")

    # Case B: HTTP 503 on clean block exit raises `LLMAdapterTransportError`.
    with mock.patch("time.sleep", return_value=None):
      with self.assertRaises(LLMAdapterTransportError):
        with ContextCacheAdapter(
            vertex_llm,
            "california_schools",
            schema=_sample_two_table_schema(),
            max_retries=1,
            http_transport=_http_503_on_delete,
        ):
          pass

      # Case C: When body raises `KeyError`, cleanup 503 logs warning without masking.
      with self.assertRaises(KeyError):
        with ContextCacheAdapter(
            vertex_llm,
            "california_schools",
            schema=_sample_two_table_schema(),
            max_retries=1,
            http_transport=_http_503_on_delete,
        ):
          raise KeyError("bad benchmark task")

  def test_system_prompt_mismatch_bypasses_cache_for_judge_requests(self) -> None:
    """`B4_LLMJudge` (`LLM_JUDGE_SYSTEM_PROMPT`) bypasses SQL synthesis cache."""
    # Provision a default SQL synthesis cache (`DEFAULT_SQLITE_SYSTEM_PROMPT`).
    cache_id = "projects/123456789012/locations/global/cachedContents/cache-sql"
    seen_bodies: list[dict[str, object]] = []

    def _mock_cache_http(
        method: str,
        url: str,
        payload_bytes: bytes | None,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> tuple[int, bytes]:
      """Creates and deletes synthetic SQL context cache handle."""
      del url, payload_bytes, headers, timeout_sec
      if method == "POST":
        return (200, json.dumps({"name": cache_id}).encode("utf-8"))
      return (200, b"{}")

    def _mock_post(payload_bytes: bytes, bearer: str) -> dict[str, object]:
      """Records wire payloads for synthesis vs judge requests."""
      del bearer
      seen_bodies.append(json.loads(payload_bytes.decode("utf-8")))
      return {
          "candidates": [{"content": {"parts": [{"text": "Selected: Option A"}]}}],
          "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 5},
      }

    # Mix one SQL synthesis request and one `B4_LLMJudge` tie-breaker request.
    vertex_llm = VertexGeminiBatchClient(
        project_id="demo-eval-project-01",
        token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
    )
    ddl_body = "Database Schema:\nCREATE TABLE schools (id INT);\n\nQuestion: Q?"
    synth_req = LLMCompletionRequest("s1", SQLITE_SYSTEM_PROMPT, ddl_body)
    judge_req = LLMCompletionRequest(
        "j1", LLM_JUDGE_SYSTEM_PROMPT, ddl_body, strategy_tag="llm_judge"
    )

    # Dispatch mixed batch through active `ContextCacheAdapter` context.
    with mock.patch.object(vertex_llm, "_post_generate_once", side_effect=_mock_post):
      with ContextCacheAdapter(
          vertex_llm,
          "california_schools",
          schema=_sample_two_table_schema(),
          http_transport=_mock_cache_http,
      ) as cached_llm:
        cached_llm.batch_complete((synth_req, judge_req))

    # Confirm synthesis request used `cachedContent` while Judge kept inline DDL.
    self.assertEqual(len(seen_bodies), 2)
    cached_wire = [b for b in seen_bodies if "cachedContent" in b]
    uncached_wire = [b for b in seen_bodies if "cachedContent" not in b]
    self.assertEqual(len(cached_wire), 1)
    self.assertEqual(len(uncached_wire), 1)
    self.assertNotIn("systemInstruction", cached_wire[0])
    self.assertIn("systemInstruction", uncached_wire[0])
    self.assertIn("CREATE TABLE schools", json.dumps(uncached_wire[0]["contents"]))

  def test_cross_database_cache_names_do_not_collide_in_local_dedup(self) -> None:
    """Identical questions across two `db_id` caches never collide in `_cache_key`."""
    # Issue distinct server `cache_name` IDs for sequential `db_a` and `db_b` runs.
    names = iter([
        "projects/123456789012/locations/global/cachedContents/cache-aaa",
        "projects/123456789012/locations/global/cachedContents/cache-bbb",
    ])

    def _mock_cache_http(
        method: str,
        url: str,
        payload_bytes: bytes | None,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> tuple[int, bytes]:
      """Returns distinct `cache_name` handles for `db_a` and `db_b`."""
      del url, payload_bytes, headers, timeout_sec
      if method == "POST":
        return (200, json.dumps({"name": next(names)}).encode("utf-8"))
      return (200, b"{}")

    # Return table-specific SQL keyed by the active `cachedContent` resource path.
    sql_by_cache = {
        "projects/123456789012/locations/global/cachedContents/cache-aaa": "SELECT COUNT(*) FROM db_a_tbl",
        "projects/123456789012/locations/global/cachedContents/cache-bbb": "SELECT COUNT(*) FROM db_b_tbl",
    }
    generate_calls = 0

    def _mock_post(payload_bytes: bytes, bearer: str) -> dict[str, object]:
      """Returns DB-specific SQL keyed by `cachedContent`."""
      nonlocal generate_calls
      del bearer
      generate_calls += 1
      body = json.loads(payload_bytes.decode("utf-8"))
      sql_out = sql_by_cache[str(body["cachedContent"])]
      return {
          "candidates": [{"content": {"parts": [{"text": sql_out}]}}],
          "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 6},
      }

    # Share a single `VertexGeminiBatchClient` across both database contexts.
    shared_vertex = VertexGeminiBatchClient(
        project_id="demo-eval-project-01",
        token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
    )
    prompt_a = "Database Schema:\nCREATE TABLE db_a_tbl (id INT);\n\nQuestion: Count rows?"
    prompt_b = "Database Schema:\nCREATE TABLE db_b_tbl (id INT);\n\nQuestion: Count rows?"

    # Stripped prompts both become `"Question: Count rows?"`; `cache_name` separates them!
    with mock.patch.object(shared_vertex, "_post_generate_once", side_effect=_mock_post):
      with ContextCacheAdapter(
          shared_vertex, "db_a", schema_ddl="CREATE TABLE db_a_tbl (id INT);", http_transport=_mock_cache_http
      ) as llm_a:
        out_a = llm_a.batch_complete((LLMCompletionRequest("q1", SQLITE_SYSTEM_PROMPT, prompt_a),))[0]
      with ContextCacheAdapter(
          shared_vertex, "db_b", schema_ddl="CREATE TABLE db_b_tbl (id INT);", http_transport=_mock_cache_http
      ) as llm_b:
        out_b1 = llm_b.batch_complete((LLMCompletionRequest("q1", SQLITE_SYSTEM_PROMPT, prompt_b),))[0]
        out_b2 = llm_b.batch_complete((LLMCompletionRequest("q2", SQLITE_SYSTEM_PROMPT, prompt_b),))[0]

    # `db_b` first call must miss cache (`from_cache=False`), second call hits cache!
    self.assertEqual(generate_calls, 2)
    self.assertEqual(out_a.sql_text, "SELECT COUNT(*) FROM db_a_tbl")
    self.assertEqual(out_b1.sql_text, "SELECT COUNT(*) FROM db_b_tbl")
    self.assertFalse(out_b1.from_cache)
    self.assertTrue(out_b2.from_cache)

  def test_401_token_refresh_and_429_retry_on_cached_contents_create(self) -> None:
    """`cachedContents.create` refreshes expired 401 token and retries 429."""
    # Supply expired bearer followed by fresh bearer to test OAuth invalidation.
    tokens = iter([("expired-bearer", 3600.0), ("fresh-bearer", 3600.0)])
    seen_bearers: list[str] = []

    def _flaky_transport(
        method: str,
        url: str,
        payload_bytes: bytes | None,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> tuple[int, bytes]:
      """Simulates HTTP 401 -> HTTP 429 -> HTTP 200 across create attempts."""
      del payload_bytes, timeout_sec
      seen_bearers.append(headers["Authorization"])
      # Fail attempt 1 with 401 Unauthorized and attempt 2 with 429 Rate Limit.
      if method == "POST" and len(seen_bearers) == 1:
        raise urllib.error.HTTPError(url, 401, "Unauthorized", Message(), None)
      if method == "POST" and len(seen_bearers) == 2:
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", Message(), None)
      return (
          200,
          json.dumps({
              "name": "projects/123456789012/locations/global/cachedContents/c-ok"
          }).encode("utf-8"),
      )

    # Patch `time.sleep` so 429 backoff completes instantaneously offline.
    vertex_llm = VertexGeminiBatchClient(
        project_id="demo-eval-project-01",
        token_supplier=lambda: ("unused-client-tok", 3600.0),
    )
    with mock.patch("time.sleep", return_value=None):
      with ContextCacheAdapter(
          vertex_llm,
          "california_schools",
          schema=_sample_two_table_schema(),
          token_supplier=lambda: next(tokens),
          http_transport=_flaky_transport,
      ) as cached_llm:
        self.assertIsNotNone(cached_llm.active_cache)

    # Confirm 401 evicted `expired-bearer` and subsequent retries used `fresh-bearer`.
    self.assertEqual(
        seen_bearers[:3],
        ["Bearer expired-bearer", "Bearer fresh-bearer", "Bearer fresh-bearer"],
    )

  def test_create_error_strict_default_vs_graceful_min_token_fallback(self) -> None:
    """`fallback_on_create_error` controls strict fail vs uncached `< min_tokens` 400."""
    # Track `cachedContents` HTTP verbs when tiny schemas fail Vertex token floor.
    methods_called: list[str] = []

    def _http_400_small_schema(
        method: str,
        url: str,
        payload_bytes: bytes | None,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> tuple[int, bytes]:
      """Simulates Vertex HTTP 400 when cached context is below minimum token floor."""
      del payload_bytes, headers, timeout_sec
      methods_called.append(method)
      raise urllib.error.HTTPError(url, 400, "Bad Request: too few tokens", Message(), None)

    vertex_llm = VertexGeminiBatchClient(
        project_id="demo-eval-project-01",
        token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
    )
    # Default `fallback_on_create_error=False` fails closed on HTTP 400.
    with self.assertRaises(LLMAdapterTransportError):
      with ContextCacheAdapter(
          vertex_llm,
          "tiny_db",
          schema_ddl="CREATE TABLE t (id INT);",
          http_transport=_http_400_small_schema,
      ):
        pass

    # With `fallback_on_create_error=True`, falls back cleanly with 0 DELETE calls.
    methods_called.clear()
    seen_gen: list[dict[str, object]] = []

    def _mock_post(payload_bytes: bytes, bearer: str) -> dict[str, object]:
      """Verifies fallback requests retain inline `systemInstruction` and DDL."""
      del bearer
      seen_gen.append(json.loads(payload_bytes.decode("utf-8")))
      return {
          "candidates": [{"content": {"parts": [{"text": "SELECT 1"}]}}],
          "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3},
      }

    # Execute uncached fallback and verify full DDL + `systemInstruction` stay intact.
    req = LLMCompletionRequest(
        "q1",
        SQLITE_SYSTEM_PROMPT,
        "Database Schema:\nCREATE TABLE t (id INT);\n\nQuestion: Count?",
    )
    with mock.patch.object(vertex_llm, "_post_generate_once", side_effect=_mock_post):
      with ContextCacheAdapter(
          vertex_llm,
          "tiny_db",
          schema_ddl="CREATE TABLE t (id INT);",
          fallback_on_create_error=True,
          http_transport=_http_400_small_schema,
      ) as fallback_llm:
        self.assertIsNone(fallback_llm.active_cache)
        out = fallback_llm.batch_complete((req,))

    # Confirm zero `DELETE` calls fired because `active_cache` remained `None`.
    self.assertEqual(out[0].sql_text, "SELECT 1")
    self.assertEqual(methods_called, ["POST"])
    self.assertNotIn("cachedContent", seen_gen[0])
    self.assertIn("systemInstruction", seen_gen[0])
    self.assertIn("CREATE TABLE t", json.dumps(seen_gen[0]["contents"]))

  def test_regional_routing_ssrf_guard_and_zero_upward_imports(self) -> None:
    """Verifies regional URLs, SSRF name rejection, and pure Hexagonal imports."""
    # Record regional `us-central1-aiplatform.googleapis.com` endpoints.
    seen_urls: list[str] = []

    def _regional_http(
        method: str,
        url: str,
        payload_bytes: bytes | None,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> tuple[int, bytes]:
      """Records regional `us-central1` URLs across create and delete."""
      del payload_bytes, headers, timeout_sec
      seen_urls.append(f"{method} {url}")
      if method == "POST":
        return (
            200,
            json.dumps({
                "name": "projects/123456789012/locations/us-central1/cachedContents/rc1"
            }).encode("utf-8"),
        )
      return (200, b"{}")

    # Verify both `POST` and `DELETE` target `us-central1-aiplatform.googleapis.com`.
    regional_vertex = VertexGeminiBatchClient(
        project_id="demo-eval-project-01",
        location="us-central1",
        token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
    )
    with ContextCacheAdapter(
        regional_vertex,
        "california_schools",
        schema=_sample_two_table_schema(),
        http_transport=_regional_http,
    ):
      pass

    self.assertEqual(
        seen_urls,
        [
            "POST https://us-central1-aiplatform.googleapis.com/v1beta1/"
            "projects/demo-eval-project-01/locations/us-central1/cachedContents",
            "DELETE https://us-central1-aiplatform.googleapis.com/v1beta1/"
            "projects/123456789012/locations/us-central1/cachedContents/rc1",
        ],
    )

    # Malicious server-returned `"name"` must raise `LLMAdapterAuthError`.
    def _evil_http(
        method: str,
        url: str,
        payload_bytes: bytes | None,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> tuple[int, bytes]:
      """Returns traversal-tainted `CachedContent.name`."""
      del method, url, payload_bytes, headers, timeout_sec
      return (200, json.dumps({"name": "projects/../evil/locations/global/cachedContents/1"}).encode("utf-8"))

    # Confirm path-traversal in server response fails closed before caching handle.
    with self.assertRaises(LLMAdapterAuthError):
      with ContextCacheAdapter(
          regional_vertex,
          "california_schools",
          schema=_sample_two_table_schema(),
          http_transport=_evil_http,
      ):
        pass

    # AST audit: `vertex_context_cache.py` must import ZERO upward domain packages.
    module_path = pathlib.Path(__file__).with_name("vertex_context_cache.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    forbidden_prefixes = (
        "argus_sql.linking",
        "argus_sql.pipelines",
        "argus_sql.verification",
        "argus_sql.sandbox",
        "argus_sql.evaluation",
    )
    # Inspect every `from ... import ...` node to guarantee Hexagonal purity.
    for node in ast.walk(tree):
      if isinstance(node, ast.ImportFrom) and node.module:
        for bad_pfx in forbidden_prefixes:
          self.assertFalse(
              node.module.startswith(bad_pfx),
              msg=f"Forbidden upward import: {node.module}",
          )

  def test_strip_cached_schema_multiline_literals_descriptions_and_crlf(
      self,
  ) -> None:
    """Strips multiline literals, spilled descriptions, and CRLF preambles cleanly."""
    # Reproducer 1b: multiline SQL string literal embedding `\n\nQuestion:` in DDL.
    # Quote-aware scanner must skip fake `Question:` delimiters inside `'...'`.
    literal_prompt = (
        "Database Schema:\n"
        "CREATE TABLE templates (\n"
        "  id INTEGER PRIMARY KEY,\n"
        "  body TEXT DEFAULT 'Intro line\n\nQuestion: embedded template literal'\n"
        ");\n\n"
        "CREATE TABLE users (\n"
        "  id INTEGER PRIMARY KEY,\n"
        "  name TEXT\n"
        ");\n\n"
        "Question: How many users exist?\n\n"
        "Instruction: Write SQL."
    )
    # Confirm both `templates` and `users` strip completely without truncation.
    stripped_lit = strip_cached_schema_from_user_prompt(literal_prompt)
    self.assertEqual(
        stripped_lit,
        "Question: How many users exist?\n\nInstruction: Write SQL.",
    )
    self.assertNotIn("CREATE TABLE", stripped_lit)
    self.assertNotIn("embedded template literal", stripped_lit)

    # Reproducer 1c: multiline `ColumnSchema.description` rendered via untouched
    # `render_schema_ddl` plus `build_db_cached_context_text` newline collapsing.
    survey_schema = DatabaseSchema(
        db_id="schools_db",
        tables=(
            TableSchema(
                name="surveys",
                columns=(
                    ColumnSchema("survey_id", "INTEGER", is_primary_key=True),
                    ColumnSchema(
                        "q1",
                        "TEXT",
                        description="Survey field.\n\nQuestion: Did school pass?",
                    ),
                ),
            ),
            TableSchema(
                name="schools",
                columns=(
                    ColumnSchema("school_id", "INTEGER", is_primary_key=True),
                ),
            ),
        ),
    )
    # Balanced parenthesis depth prevents spilled description lines from ending DDL.
    rendered_ddl = render_schema_ddl(survey_schema, include_metadata_comments=True)
    desc_prompt = (
        f"Database Schema:\n{rendered_ddl}\n\n"
        "Question: Count schools?\n\n"
        "Instruction: Output SQL."
    )
    stripped_desc = strip_cached_schema_from_user_prompt(desc_prompt)
    self.assertEqual(
        stripped_desc,
        "Question: Count schools?\n\nInstruction: Output SQL.",
    )
    self.assertNotIn("CREATE TABLE", stripped_desc)
    self.assertNotIn("Did school pass?", stripped_desc)

    # Verify `build_db_cached_context_text` collapses internal CR/LF in schema & map.
    cached_ctx = build_db_cached_context_text("schools_db", schema=survey_schema)
    self.assertIn("Survey field. Question: Did school pass?", cached_ctx)
    self.assertNotIn("\nQuestion:", cached_ctx)
    mapped_ctx = build_db_cached_context_text(
        "schools_db",
        schema_ddl="CREATE TABLE t (id INT);",
        descriptions={("t", "id"): "Line 1\r\n\r\nQuestion: Line 2"},
    )
    self.assertIn("- t.id: Line 1 Question: Line 2", mapped_ctx)
    self.assertNotIn("\nQuestion:", mapped_ctx)

    # Verify Windows CRLF repair preamble, block comments, backticks, brackets,
    # and `WITHOUT ROWID, STRICT` table tails strip cleanly while keeping `\r\n\r\n`.
    crlf_repair = (
        "You are repairing a SQLite query.\r\n\r\n"
        "Database Schema:\r\n"
        "/* \r\n\r\nQuestion: fake ); */\r\n"
        "CREATE TEMP TABLE IF NOT EXISTS `t``1` ([c);1] INT) WITHOUT ROWID, STRICT;\r\n\r\n"
        "Question: Select c?\r\n\r\n"
        "Instruction: Fix SQL."
    )
    stripped_crlf = strip_cached_schema_from_user_prompt(crlf_repair)
    self.assertEqual(
        stripped_crlf,
        "You are repairing a SQLite query.\r\n\r\nQuestion: Select c?\r\n\r\nInstruction: Fix SQL.",
    )

    # Verify non-DDL CRLF schema summaries before `User Question:` and `Query:`.
    for q_label in ("User Question", "Query"):
      summary_prompt = (
          f"Database Schema:\r\nTables: schools(id), frpm(meals)\r\n\r\n"
          f"{q_label}: Count rows?"
      )
      self.assertEqual(
          strip_cached_schema_from_user_prompt(summary_prompt),
          f"{q_label}: Count rows?",
      )

  def test_strip_cached_schema_linear_time_on_sqlite_master_without_semicolon(
      self,
  ) -> None:
    """Strips un-semicoloned `sqlite_master` DDL in < 50ms with zero ReDoS."""
    # Reproducer 1e: test 16-col and 64-col `sqlite_master` DDL without `;`.
    # Raw SQLite `sqlite_master.sql` omits trailing `;` on `CREATE TABLE` blocks.
    for col_count in (16, 64):
      cols = ",\n".join(
          f"  d{idx} TEXT -- range: '2020-01-01'..'2021-12-31'"
          for idx in range(col_count)
      )
      unlabeled_ddl = (
          "Database Schema:\n"
          "CREATE TABLE t0 (id INT);\n\n"
          f"CREATE TABLE t1 (\n{cols}\n)\n\n"
          "Select 1."
      )
      # Measure monotonic wall-clock execution to enforce `< 50ms` O(N) bound.
      t0 = time.perf_counter()
      stripped = strip_cached_schema_from_user_prompt(unlabeled_ddl)
      elapsed_ms = (time.perf_counter() - t0) * 1000.0
      self.assertEqual(stripped, "Select 1.")
      self.assertLess(elapsed_ms, 50.0)

      # Also verify labeled `Question:` prompt following un-semicoloned DDL.
      labeled_ddl = (
          "Database Schema:\n"
          "CREATE TABLE t0 (id INT);\n\n"
          f"CREATE TABLE t1 (\n{cols}\n)\n\n"
          "Question: Count rows in t1?"
      )
      t1 = time.perf_counter()
      stripped_q = strip_cached_schema_from_user_prompt(labeled_ddl)
      self.assertLess((time.perf_counter() - t1) * 1000.0, 50.0)
      self.assertEqual(stripped_q, "Question: Count rows in t1?")
      self.assertNotIn("CREATE TABLE", stripped_q)

    # Unclosed/truncated `CREATE TABLE (` must also return in linear time (< 50ms).
    unclosed_cols = ",\n".join(f"  c{i} TEXT DEFAULT 'x'" for i in range(64))
    truncated_prompt = f"Database Schema:\nCREATE TABLE t1 (\n{unclosed_cols}\n"
    t_unclosed = time.perf_counter()
    out_unclosed = strip_cached_schema_from_user_prompt(truncated_prompt)
    self.assertLess((time.perf_counter() - t_unclosed) * 1000.0, 50.0)
    self.assertTrue(bool(out_unclosed))

  def test_http_exception_wrapping_unwind_safety_and_none_client_attrs(
      self,
  ) -> None:
    """`HTTPException`/`OSError` wrap cleanly, never mask body errors, & skip `None`."""
    # Shared hermetic client for transport error and exception-unwind tests.
    vertex_llm = VertexGeminiBatchClient(
        project_id="demo-eval-project-01",
        token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
    )
    delete_calls = 0

    def _disconnect_on_delete(
        method: str,
        url: str,
        payload_bytes: bytes | None,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> tuple[int, bytes]:
      """Succeeds on `POST` create and raises `RemoteDisconnected` on `DELETE`."""
      nonlocal delete_calls
      del url, payload_bytes, headers, timeout_sec
      # Provision synthetic cache handle on `POST`, then drop socket on `DELETE`.
      if method == "POST":
        return (
            200,
            json.dumps({
                "name": "projects/123456789012/locations/global/cachedContents/c-rd"
            }).encode("utf-8"),
        )
      delete_calls += 1
      raise http.client.RemoteDisconnected("closed")

    # Challenge 4a (1h): `RemoteDisconnected` on `DELETE` never masks `ValueError`.
    with mock.patch("time.sleep", return_value=None):
      with self.assertRaisesRegex(ValueError, "body_err"):
        with ContextCacheAdapter(
            vertex_llm,
            "california_schools",
            schema=_sample_two_table_schema(),
            max_retries=2,
            http_transport=_disconnect_on_delete,
        ):
          raise ValueError("body_err")
      # Confirm bounded retry loop executed both `DELETE` attempts before logging.
      self.assertEqual(delete_calls, 2)

      # Clean block exit (`exc_type is None`) must surface TransportError to caller.
      with self.assertRaises(LLMAdapterTransportError):
        with ContextCacheAdapter(
            vertex_llm,
            "california_schools",
            schema=_sample_two_table_schema(),
            max_retries=1,
            http_transport=_disconnect_on_delete,
        ):
          pass

      # Even an unexpected `RuntimeError` inside `delete_cache` never masks body error.
      adapter_rt = ContextCacheAdapter(
          vertex_llm,
          "california_schools",
          schema=_sample_two_table_schema(),
          http_transport=_disconnect_on_delete,
      )
      with mock.patch.object(
          adapter_rt, "delete_cache", side_effect=RuntimeError("cleanup bug")
      ):
        with self.assertRaisesRegex(ValueError, "body_preserved"):
          with adapter_rt:
            raise ValueError("body_preserved")

      # Clean `POST` create wraps every `HTTPException` and `OSError` subclass
      # into `LLMAdapterTransportError` with original `__cause__` preserved.
      for raw_fault in (
          http.client.RemoteDisconnected("closed"),
          http.client.BadStatusLine("bad"),
          http.client.IncompleteRead(b"{", 10),
          ConnectionResetError("reset"),
      ):
        def _fault_tx(
            method: str,
            url: str,
            payload_bytes: bytes | None,
            headers: dict[str, str],
            timeout_sec: float,
            _exc: Exception = raw_fault,
        ) -> tuple[int, bytes]:
          """Raises injected socket/HTTP exception on `POST`."""
          del method, url, payload_bytes, headers, timeout_sec
          # Simulate mid-stream HTTP or socket transport fault before response.
          raise _exc

        # Verify exception chaining (`raise LLMAdapterTransportError(...) from err`).
        with self.assertRaises(LLMAdapterTransportError) as ctx_err:
          ContextCacheAdapter(
              vertex_llm,
              "california_schools",
              schema=_sample_two_table_schema(),
              max_retries=1,
              http_transport=_fault_tx,
          ).create_cache()
        self.assertIs(ctx_err.exception.__cause__, raw_fault)

      # Invalid UTF-8 response bytes (`0xff 0xfe`) also chain into TransportError.
      with self.assertRaises(LLMAdapterTransportError) as utf8_err:
        ContextCacheAdapter(
            vertex_llm,
            "california_schools",
            schema=_sample_two_table_schema(),
            max_retries=1,
            http_transport=lambda m, u, b, h, t: (200, b"\xff\xfe"),
        ).create_cache()
      self.assertIsInstance(utf8_err.exception.__cause__, UnicodeDecodeError)

    # Reviewer #1 Minor 1: `None` attributes on custom `BatchLLMPort` fall back to env.
    # Prevents `str(getattr(client, attr, ''))` from coercing `None` into `'None'`.
    class _NoneAttrPort:
      """Minimal `BatchLLMPort` test double exposing explicit `None` attributes."""

      # Set optional client metadata attributes to `None` explicitly.
      project_id: str | None = None
      location: str | None = None
      model_name: str | None = None

      def batch_complete(
          self,
          requests: tuple[LLMCompletionRequest, ...],
      ) -> tuple[()]:
        """Returns an empty completion tuple for hermetic attribute tests.

        Args:
          requests: Tuple of `LLMCompletionRequest` items (unused in stub).

        Returns:
          Empty tuple satisfying the `BatchLLMPort` call contract.
        """
        del requests
        return ()

    # Confirm `GOOGLE_CLOUD_PROJECT` and default location/model win over `None`.
    with mock.patch.dict(
        os.environ, {"GOOGLE_CLOUD_PROJECT": "env-eval-proj-01"}, clear=True
    ):
      none_adapter = ContextCacheAdapter(
          _NoneAttrPort(),  # type: ignore[arg-type]
          "california_schools",
          schema=_sample_two_table_schema(),
          token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
      )
    self.assertEqual(
        (none_adapter.project_id, none_adapter.location, none_adapter.model_name),
        ("env-eval-proj-01", "global", "gemini-3.7-flash"),
    )

  def test_spilled_multiline_descriptions_apostrophes_parens_and_redos_bounds(
      self,
  ) -> None:
    """Strips spilled `'s` / `(...)` descriptions, untyped DDL, & 100k-space inputs."""
    # Verify Challenger 1/2 multi-line descriptions with `'s`, `)`, bare `)`, `/*`, `;`.
    # Covers blank-line `Question:` spoofing, bullets, and indented English prose.
    tricky_descs = (
        "Survey field.\n\nQuestion: What is the student's GPA?",
        "Score summary.\nStudent's raw score",
        "Status code (see codebook:\n1 = Open, 2 = Closed)",
        "Type:\n1) Public\n2) Charter\n\nQuestion: Return 0?",
        "Metric (Ages 5-17\nin K-12)\n\nQuestion: Fake?",
        "Metric (\n)",
        "Bullets:\n  1) Public\n  2) Charter\n  in K-12)\n  Student's note",
        "Prose:\n  count integer values in Student's file; /* ratio */",
    )
    t0_tbl = TableSchema("t0", (ColumnSchema("id", "INT", is_primary_key=True),))
    t2_tbl = TableSchema("t2", (ColumnSchema("id", "INT", is_primary_key=True),))
    # Assert every multi-line description strips cleanly across middle and multi-table DDL.
    for desc in tricky_descs:
      cols = (
          ColumnSchema("id", "INT", is_primary_key=True),
          ColumnSchema("c", "TEXT", description=desc),
          ColumnSchema("tail", "TEXT"),
      )
      ddl = render_schema_ddl(
          DatabaseSchema("adv_db", (t0_tbl, TableSchema("t1", cols), t2_tbl))
      )
      out = strip_cached_schema_from_user_prompt(
          f"Database Schema:\n{ddl}\n\nQuestion: What is Bob's score?\n\nInstruction: SQL."
      )
      self.assertEqual(out, "Question: What is Bob's score?\n\nInstruction: SQL.")

    # Verify untyped SQLite columns, quoted names, CHECK, & DEFAULT after spilled `--`.
    # Ensures nested `CHECK (...)` depth and `''`-escaped quotes stay balanced.
    adversarial_ddl = (
        'Database Schema:\nCREATE TABLE "school stats" (\n'
        '  "Free Meal (K-12)" REAL, -- spilled (\n  count integer in Student\'s file\n'
        "  col_no_type,\n  note TEXT DEFAULT 'it''s\n);\n\nQuestion: fake?\nfine',\n"
        "  score INT CHECK (\n    (score >= 0) AND (score IN (1, 2))\n  ),\n  col_last\n);\n\n"
        "Question: Real query?"
    )
    # Confirm raw DDL block strips cleanly down to the trailing real Question.
    self.assertEqual(
        strip_cached_schema_from_user_prompt(adversarial_ddl), "Question: Real query?"
    )

    # Verify Layer 1 + Layer 2 inside `ContextCacheAdapter.batch_complete` AND
    # explicit JSON `null` coalescing (`totalTokenCount: null`, `expireTime: null`).
    seen_gen: list[dict[str, object]] = []
    vertex_llm = VertexGeminiBatchClient(
        "demo-eval-project-01",
        token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
    )
    # Construct schema whose multi-line description combines bullets, `'s`, & `Question:`.
    adv_schema = DatabaseSchema(
        "schools_db",
        (
            TableSchema(
                "schools",
                (
                    ColumnSchema("id", "INT", is_primary_key=True),
                    ColumnSchema(
                        "kind",
                        "TEXT",
                        description="1) Public\n2) Student's\n\nQuestion: Fake?",
                    ),
                ),
            ),
        ),
    )
    # Render untouched `render_schema_ddl` output directly into user_prompt.
    req = LLMCompletionRequest(
        "q1",
        SQLITE_SYSTEM_PROMPT,
        f"Database Schema:\n{render_schema_ddl(adv_schema)}\n\nQuestion: Count schools?",
    )

    def _capture_gen(payload_bytes: bytes, bearer: str) -> dict[str, object]:
      """Records outbound `generateContent` wire JSON for assertion."""
      del bearer
      # Capture decoded JSON request body for post-call wire inspection.
      seen_gen.append(json.loads(payload_bytes.decode("utf-8")))
      return {"candidates": [{"content": {"parts": [{"text": "SELECT 1"}]}}]}

    # Confirm wire `contents` carries only `Question:` and `null` cache metadata coalesces.
    null_meta_resp = (
        b'{"name":"projects/123456/locations/global/cachedContents/c1",'
        b'"usageMetadata":{"totalTokenCount":null},"expireTime":null}'
    )
    with mock.patch.object(vertex_llm, "_post_generate_once", side_effect=_capture_gen):
      with ContextCacheAdapter.for_database(
          vertex_llm,
          "schools_db",
          schema=adv_schema,
          http_transport=lambda m, u, b, h, t: (
              200,
              null_meta_resp if m == "POST" else b"{}",
          ),
      ) as cached_port:
        # Explicit JSON `null` must coalesce to `0` and `""` (never `"None"`).
        assert cached_port.active_cache is not None
        self.assertEqual(cached_port.active_cache.total_token_count, 0)
        self.assertEqual(cached_port.active_cache.expire_time, "")
        cached_port.batch_complete((req,))
    wire_txt = seen_gen[0]["contents"][0]["parts"][0]["text"]  # type: ignore[index]
    self.assertEqual(wire_txt, "Question: Count schools?")

    # Reject boolean `totalTokenCount: true` and non-string `expireTime` on create_cache.
    # Prevents `isinstance(True, int)` from misreporting 1 cached token.
    for bad_cache_resp in (
        b'{"name":"projects/123456/locations/global/cachedContents/c2","usageMetadata":{"totalTokenCount":true}}',
        b'{"name":"projects/123456/locations/global/cachedContents/c2","expireTime":123}',
    ):
      with self.assertRaises(LLMAdapterTransportError):
        ContextCacheAdapter.for_database(
            vertex_llm,
            "schools_db",
            schema=adv_schema,
            http_transport=lambda m, u, b, h, t, _r=bad_cache_resp: (200, _r),
        ).create_cache()

    # Enforce < 50ms O(N) runtime on 50k-space and 100k-space header/tail inputs.
    for pad_prompt in (
        (" " * 50000) + "Database Schema without colon",
        (" " * 100000) + "Database Schema without colon",
        "Database Schema:\nCREATE TABLE t (id INT)" + (" " * 100000) + "\n\nQuestion: Q?",
    ):
      t_start = time.perf_counter()
      strip_cached_schema_from_user_prompt(pad_prompt)
      self.assertLess((time.perf_counter() - t_start) * 1000.0, 50.0)

  def test_iter3_quadratic_rescan_unindented_strict_ddl_and_crlf_prose(
      self,
  ) -> None:
    """Covers 100k O(N) trivia/table bounds, unindented/STRICT DDL, & CRLF prose."""
    # Group 1: Enforce < 50ms O(N) runtime across 96k-100k `) /*`, `)\n/*` + `--`,
    # unclosed `/*`, `saw_dash` fallback, 4,000-table, and open/closed nl-floods.
    hdr = "Database Schema:\nCREATE TABLE t (\n  id INT -- spilled\n"
    linear_probes: tuple[tuple[str, str, str | None], ...] = (
        ("100k-paren-bc", hdr + (") /*\n" * 20000) + "*/ not_close\n);\n\nQuestion: Q?", "Question: Q?"),
        ("99k-open-bc", hdr + (") /* *****\n" * 9000) + ");\n\nQuestion: Q?", "Question: Q?"),
        ("100k-saw-dash", f"{hdr}  s TEXT DEFAULT 'open\n" + (") /*\n" * 19980) + "*/ x\n", None),
        ("100k-nl-bc-dash", hdr + (")\n/*\n" * 10000) + "*/\n" + ("--\n" * 16650) + "x", None),
        ("100k-closed-bc", hdr + (")\n/*\n" * 9990) + ("a" * 49990) + "*/\nx", None),
        ("96k-4k-tables", "Database Schema:\n" + ("CREATE TABLE t (a INT);\n" * 4000) + "\nQuestion: Q?", "Question: Q?"),
        ("100k-nl-flood", hdr + ("x\n" * 49970), None),
        ("100k-closed-nl-flood", "Database Schema:\nCREATE TABLE t (\n  id INT -- c\n" + ("x\n" * 49970) + ");\n\nQuestion: Q?", "Question: Q?"),
        ("100k-dash-flood", f"{hdr})\n" + ("-- c\n" * 19999) + "  id2 INT\n);\n\nQuestion: Q?", "Question: Q?"),
    )
    # Verify every adversarial 100k-scale probe finishes under the 50ms ceiling.
    for label, probe_prompt, expected_tail in linear_probes:
      t0 = time.perf_counter()
      stripped_probe = strip_cached_schema_from_user_prompt(probe_prompt)
      elapsed_ms = (time.perf_counter() - t0) * 1000.0
      self.assertLess(elapsed_ms, 50.0, msg=f"{label}: {elapsed_ms:.2f}ms")
      if expected_tail is not None:
        self.assertEqual(stripped_probe, expected_tail, msg=label)

    # Group 2: Unindented `sqlite_master` DDL with `--` comment & compact `val TEXT);`
    # (labeled + unlabeled), stray `]`, and SQLite `STRICT` `ANY`/`VARCHAR(MAX)`/`TIMESTAMPTZ`.
    for raw_ddl_prompt, expected_stripped in (
        (
            "Database Schema:\nCREATE TABLE t1 (\nid INT, -- pk\n"
            "bio TEXT DEFAULT 'intro\n)\n\nQuestion: fake?',\nval TEXT);\n\n"
            "CREATE TABLE t2 (\nid INT);\n\nQuestion: Real question?",
            "Question: Real question?",
        ),
        (
            "Database Schema:\nCREATE TABLE t1 (\nid INT, -- pk\nval TEXT);\n\n"
            "CREATE TABLE t2 (\nid INT);\n\nSelect count from t1.",
            "Select count from t1.",
        ),
        (
            "Database Schema:\nCREATE TABLE kv (\n  k TEXT PRIMARY KEY, -- key\n"
            "  v ANY NOT NULL DEFAULT 'a\n)\nQuestion: fake?',\n"
            "  note VARCHAR(MAX) DEFAULT 'b\n)\nQuestion: fake2?',\n"
            "  ts TIMESTAMPTZ DEFAULT '2026-01-01\n)\nQuestion: fake3?'\n"
            ") STRICT;\n\nCREATE TABLE t2 (\n  id INT PRIMARY KEY\n);\n\n"
            "Question: Real Q?\n\nInstruction: SQL.",
            "Question: Real Q?\n\nInstruction: SQL.",
        ),
        (
            "Database Schema:\nCREATE TABLE t1 (id] INT);\n\nQuestion: Stray bracket?",
            "Question: Stray bracket?",
        ),
    ):
      # Confirm zero leaked `CREATE TABLE t2` or fake `Question:` literals survive.
      self.assertEqual(strip_cached_schema_from_user_prompt(raw_ddl_prompt), expected_stripped)

    # Group 3: CRLF `ContextCacheAdapter(schema=...)` & LF `schema_ddl=...` with
    # `"Status code (see:\n  Active status)"`, plus `"Notes:\n  Student text, ..."`.
    t2_tbl = TableSchema("t2", (ColumnSchema("id", "INT", is_primary_key=True),))
    status_desc = "Status code (see:\n  Active status)\n\nQuestion: Fake?"
    status_cols = (ColumnSchema("id", "INT", is_primary_key=True), ColumnSchema("c", "TEXT", description=status_desc))
    status_schema = DatabaseSchema("db", (TableSchema("t1", status_cols), t2_tbl))
    status_ddl = render_schema_ddl(status_schema)
    lf_prompt = f"Database Schema:\n{status_ddl}\n\nQuestion: Real Q?\n\nInstruction: SQL."
    # Capture wire `contents[0].parts[0].text` across CRLF `schema=` and LF `schema_ddl=`.
    sent_texts: list[str] = []
    vertex_llm = VertexGeminiBatchClient(
        "demo-eval-project-01", token_supplier=lambda: ("synthetic-test-bearer", 3600.0)
    )

    def _record_wire_text(payload_bytes: bytes, bearer: str) -> dict[str, object]:
      """Records outbound `contents[0].parts[0].text` for CRLF/LF verification."""
      del bearer
      # Decode outbound `generateContent` JSON body to inspect stripped prompt text.
      sent_texts.append(json.loads(payload_bytes.decode("utf-8"))["contents"][0]["parts"][0]["text"])
      return {"candidates": [{"content": {"parts": [{"text": "SELECT 1"}]}}]}

    # Mock `cachedContents` create/delete lifecycle hermetically in memory.
    ok_cache = b'{"name":"projects/123456/locations/global/cachedContents/c9"}'
    cache_tx = lambda m, u, b, h, t: (200, ok_cache if m == "POST" else b"{}")
    notes_desc = "Notes:\n  Student text, see owner's manual"
    notes_cols = (ColumnSchema("id", "INT", is_primary_key=True), ColumnSchema("c", "TEXT", description=notes_desc))
    notes_schema = DatabaseSchema("db", (TableSchema("t1", notes_cols), t2_tbl))
    notes_ddl = render_schema_ddl(notes_schema)
    # Verify CRLF `schema=`, LF `schema_ddl=`, and unflattened user `Question:` text.
    echo_notes_q = f"Database Schema:\n{notes_ddl}\n\nQuestion: Cite {notes_desc}?"
    with mock.patch.object(vertex_llm, "_post_generate_once", side_effect=_record_wire_text):
      with ContextCacheAdapter(vertex_llm, "db", schema=status_schema, http_transport=cache_tx) as ca1:
        ca1.batch_complete((LLMCompletionRequest("r1", DEFAULT_SQLITE_SYSTEM_PROMPT, lf_prompt.replace("\n", "\r\n")),))
      with ContextCacheAdapter(vertex_llm, "db", schema_ddl=status_ddl, http_transport=cache_tx) as ca2:
        ca2.batch_complete((LLMCompletionRequest("r2", DEFAULT_SQLITE_SYSTEM_PROMPT, lf_prompt),))
      with ContextCacheAdapter(vertex_llm, "db", schema=notes_schema, http_transport=cache_tx) as ca3:
        ca3.batch_complete((LLMCompletionRequest("r3", DEFAULT_SQLITE_SYSTEM_PROMPT, echo_notes_q),))
    # Confirm CRLF/LF prompts strip cleanly while `echo_notes_q` keeps its newline.
    self.assertEqual(
        sent_texts,
        [
            "Question: Real Q?\r\n\r\nInstruction: SQL.",
            "Question: Real Q?\n\nInstruction: SQL.",
            f"Question: Cite {notes_desc}?",
        ],
    )

    # Indented `"  Student text, see owner's manual"` must never swallow `Question:`.
    self.assertEqual(
        strip_cached_schema_from_user_prompt(
            f"Database Schema:\n{notes_ddl}\n\nQuestion: What is Bob's score in case 1)\n\nInstruction: SQL."
        ),
        "Question: What is Bob's score in case 1)\n\nInstruction: SQL.",
    )

  def test_iter4_trailing_nl_paren_flood_spilled_block_comment_and_sqlite_types(
      self,
  ) -> None:
    """Verifies <50ms 98k/100k probes, AUD-1754, Traps 1-2, `/*`, & fts5/SQLite DDL."""
    # Group 1 (`reviewer_iter4_1` A01/A01b + `challenger_iter4_1` F5/F12/F13/F14/F16 + AUD-1754):
    # Enforce `< 50ms` O(N) runtime across trailing-LF, `)`-flood, & 1,754 unbalanced-CHECK tables.
    spilled_hdr = "Database Schema:\nCREATE TABLE t (\n  id INT -- s\n  a) b\n"
    f5_tbl = "CREATE TABLE t (\n  id INT -- d\n" + (")\n" * 15) + ");\n"
    aud_tbl = "CREATE TABLE t (\n  id INT, -- c\n  x INT CHECK ((x > 0)\n);\n"
    iter4_probes: tuple[tuple[str, str, str | None], ...] = (
        ("A01-98k-trailing-bc-nl", spilled_hdr + ("p -- c\n" * 14000) + "/*\n", None),
        ("A01b-98k-trailing-strict-nl", spilled_hdr + ("p -- c\n" * 14000) + ") STRICT bad\n", None),
        ("F5-100k-multitable-paren", "Database Schema:\n" + (f5_tbl * 1537) + "\nQuestion: Q?", "Question: Q?"),
        ("F12-100k-paren-same-line", "Database Schema:\nCREATE TABLE t (a INT" + (")" * 50000) + (" " * 49940) + "x", None),
        (
            "F13-100k-paren-nl-dash",
            "Database Schema:\nCREATE TABLE t1 (\n  id INT\n" + (")\n-- c\n" * 14270) + "not_ct\nCREATE TABLE t2 (id INT);\n\nQuestion: Q?",
            "Question: Q?",
        ),
        (
            "F14-100k-paren-nl",
            "Database Schema:\nCREATE TABLE t1 (\n  id INT -- d\n" + (")\n" * 49956) + ");\n\nCREATE TABLE t2 (id INT);\n\nQuestion: Q?",
            "Question: Q?",
        ),
        (
            "F16-100k-spilled-paren-blank",
            "Database Schema:\nCREATE TABLE t1 (\n  id INT -- d\n" + (")\n\n" * 33303) + ");\n\nCREATE TABLE t2 (id INT);\n\nQuestion: Q?",
            "Question: Q?",
        ),
        (
            "AUD-1754-unbalanced-check",
            "Database Schema:\n" + (aud_tbl * 1754) + "\nQuestion: Q?",
            "Question: Q?",
        ),
    )
    # Time each 98k-100k probe individually to verify strict `< 50ms` linear bound.
    for label, probe_prompt, expected_tail in iter4_probes:
      t0 = time.perf_counter()
      got_probe = strip_cached_schema_from_user_prompt(probe_prompt)
      elapsed_ms = (time.perf_counter() - t0) * 1000.0
      self.assertLess(elapsed_ms, 50.0, msg=f"{label}: {elapsed_ms:.2f}ms")
      if expected_tail is not None:
        self.assertEqual(got_probe, expected_tail, msg=label)

    # Group 2 (`reviewer_iter4_1` Finding 2 + `challenger_iter4_1` Challenge 2 prose + Trap 2):
    # Verify closed inline `/* note */ Student's value`, natural constraint prose, and Trap 2.
    t2_tbl = TableSchema("t2", (ColumnSchema("id", "INT", is_primary_key=True),))
    bob_q = "Question: What is Bob's score in case 1)\n\nInstruction: SQL."
    for spilled_desc in (
        "Header:\n  /* note */ Student's value",
        "Foreign key.\nDistrict references school's CDS code",
        "Nullability:\n  Value not null for active student's row",
        "Fallback:\n  Score default 0 when student's exam missing",
        "Fallback quoted:\n  Score default 'N/A' when student's exam missing",
    ):
      t1_cols = (
          ColumnSchema("id", "INT", is_primary_key=True),
          ColumnSchema("c", "TEXT", description=spilled_desc),
      )
      ddl_prose = render_schema_ddl(DatabaseSchema("db", (TableSchema("t1", t1_cols), t2_tbl)))
      # Confirm zero user `Question:` swallowing on every spilled-prose pattern.
      self.assertEqual(
          strip_cached_schema_from_user_prompt(f"Database Schema:\n{ddl_prose}\n\n{bob_q}"),
          bob_q,
          msg=spilled_desc,
      )

    # Trap 2: `"Question: For Bob's case 1);"` must stay protected even when ending with `);`.
    trap2_cols = (
        ColumnSchema("id", "INT", is_primary_key=True),
        ColumnSchema("c", "TEXT", description="Header:\n  Student's note"),
    )
    trap2_ddl = render_schema_ddl(DatabaseSchema("db", (TableSchema("t1", trap2_cols), t2_tbl)))
    for trap2_q in (
        "Question: For Bob's case 1);",
        "Question: For Bob's case 1);\n\nInstruction: SQL.",
    ):
      # Confirm Pass 1 `;`-ending swallow falls back to Pass 2 without deleting `Question:`.
      self.assertEqual(
          strip_cached_schema_from_user_prompt(f"Database Schema:\n{trap2_ddl}\n\n{trap2_q}"),
          trap2_q,
      )

    # Unclosed `/*` in spilled description must never jump across `);` into `Query: /* hint */`.
    unclosed_bc_cols = (
        ColumnSchema("id", "INT", is_primary_key=True),
        ColumnSchema("c", "VARCHAR(50)", description="Unclosed:\n  /* star without close\n  2) val"),
    )
    ddl_unclosed_bc = render_schema_ddl(DatabaseSchema("db", (t2_tbl, TableSchema("t1", unclosed_bc_cols))))
    self.assertEqual(
        strip_cached_schema_from_user_prompt(
            f"Database Schema:\n{ddl_unclosed_bc}\n\nQuery: Select count /* hint */ from t1;"
        ),
        "Query: Select count /* hint */ from t1;",
    )

    # Group 3 (Trap 1 + SQLite multi-word types + `'col'` + VIRTUAL TABLE `fts5`/`rtree`):
    # Trap 1: Spilled `Metric (\n)\n\n` before `c2 TEXT DEFAULT 'a\n);\n\nQuestion: fake?'`.
    trap1_ddl = (
        "CREATE TABLE t1 (\n  id INT PRIMARY KEY, -- pk\n  c1 TEXT, -- Metric (\n)\n\nMore notes\n"
        "  c2 TEXT DEFAULT 'a\n);\n\nQuestion: fake?'\n);\n\nCREATE TABLE t2 (id INT);"
    )
    self.assertEqual(
        strip_cached_schema_from_user_prompt(f"Database Schema:\n{trap1_ddl}\n\nQuestion: Real Q?"),
        "Question: Real Q?",
    )

    # Verify `UNSIGNED BIG INT`, 4-word `TIMESTAMP WITHOUT TIME ZONE`, `'col'`, and `fts5`/`rtree`.
    fake_q_t1 = "CREATE TABLE t1 (id INT PRIMARY KEY, c1 TEXT DEFAULT 'a\n);\n\nQuestion: fake?');"
    for sqlite_ddl in (
        f"{fake_q_t1}\n\nCREATE TABLE t2 (\n  id INT PRIMARY KEY, -- pk\n  big_id UNSIGNED BIG INT);",
        (
            f"{fake_q_t1}\n\nCREATE TABLE t2 (\n  id INT PRIMARY KEY, -- pk\n"
            "  ts TIMESTAMP WITHOUT TIME ZONE CHECK (ts IS NOT NULL), /* m\n);\nQuestion: fake_bc? */\n"
            "  v DECIMAL(10, 2)\n) WITHOUT ROWID;"
        ),
        (
            f"{fake_q_t1}\n\nCREATE TABLE t2 (\n  id INT PRIMARY KEY, -- pk\n"
            "  'col' TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,\n  'col2' VARCHAR(255) NOT NULL\n);"
        ),
        (
            f"{fake_q_t1}\n\nCREATE VIRTUAL TABLE docs USING fts5(title, body UNINDEXED, tokenize = 'porter');\n\n"
            "CREATE VIRTUAL TABLE geo USING rtree(\n  -- spatial index\n  id, minX, maxX,\n  minY, maxY\n);"
        ),
    ):
      # Confirm zero DDL leaks across unlabeled prompts with embedded `Question:` literals.
      self.assertEqual(
          strip_cached_schema_from_user_prompt(f"Database Schema:\n{sqlite_ddl}\n\nSelect 1."),
          "Select 1.",
          msg=sqlite_ddl,
      )

  def test_iter5_intertable_separators_crlf_stage4b_and_unsemi_boundaries(
      self,
  ) -> None:
    """Covers <50ms inter-table/Q7/Q8/AUD-1500, CRLF Trap 1, Stage 4b, H1-H2, & p1-p2."""
    # Group 1 (`reviewer_iter5_1` Q1/Q2/Q1c/Q1d/Q7/Q8 + `challenger_iter5_1` NEW-Q1/Q2 + `auditor_iter5_1` Info 1):
    # Enforce `< 50ms` linear runtime across 97k-100k non-LF inter-table separators & parens.
    iter5_probes: tuple[tuple[str, str, str], ...] = (
        ("NEW-Q1-98k-semi-create", "Database Schema:\n" + ("CREATE TABLE t (\n  id INT, -- c\n  x INT CHECK ((x > 0)\n); " * 1700) + "\n\nQuestion: Q?", "Question: Q?"),
        ("Q2-98k-nl-semi-create", "Database Schema:\n" + ("CREATE TABLE t (\n  id INT, -- c\n  x INT CHECK ((x > 0)\n)\n; " * 1650) + "\n\nQuestion: Q?", "Question: Q?"),
        ("Q1c-98k-bc-sep-create", "Database Schema:\n" + ("CREATE TABLE t (\n  id INT, -- c\n  x INT CHECK ((x > 0)\n)\n/* m */ " * 1500) + "\n\nQuestion: Q?", "Question: Q?"),
        ("Q1d-98k-strict-semi-create", "Database Schema:\n" + ("CREATE TABLE t (\n  id INT, -- c\n  x INT CHECK ((x > 0)\n) STRICT, WITHOUT ROWID; " * 1200) + "\n\nQuestion: Q?", "Question: Q?"),
        ("NEW-Q2-97k-bc-prefix-create", "Database Schema:\n" + ("/* c */ CREATE TABLE t (\n  id INT, -- c\n  x INT CHECK ((x > 0)\n);\n" * 1480) + "\nQuestion: Q?", "Question: Q?"),
        ("Q7-100k-depth2-parens", "Database Schema:\nCREATE TABLE t (a INT -- c\n, b INT CHECK (" + ("((1))+" * 16600) + "0)\n;\n\nQuestion: Q?", "Question: Q?"),
        ("Q8-100k-spilled-semi-garbage", "Database Schema:\nCREATE TABLE t (\n  id INT -- d\n" + (");x\n" * 24980) + ");\n\nQuestion: Q?", "Question: Q?"),
        (
            "AUD-1500-nodash-unbal-check",
            "Database Schema:\n" + ("CREATE TABLE t (\n  c TEXT DEFAULT 'x',\n  x INT CHECK ((x > 0)\n);\n" * 1500) + "CREATE TABLE tlast (\n  id INT -- comment\n);\n\nQuestion: Q?",
            "Question: Q?",
        ),
    )
    # Time each 97k-100k adversarial probe to verify `< 50ms` O(N) execution and output.
    for label, probe_prompt, expected_tail in iter5_probes:
      t0 = time.perf_counter()
      got_probe = strip_cached_schema_from_user_prompt(probe_prompt)
      elapsed_ms = (time.perf_counter() - t0) * 1000.0
      self.assertLess(elapsed_ms, 50.0, msg=f"{label}: {elapsed_ms:.2f}ms")
      self.assertEqual(got_probe, expected_tail, msg=label)

    # Group 2: CRLF `Trap 1`, Stage 4b + 2-line `Trap 2`, `Case S1` spilled `CREATE TABLE`,
    # `H1` `DEFAULT '...Previous SQL:...'`, and `Bucket B` spilled `Previous SQL:` prose.
    trap1_crlf = (
        "Database Schema:\r\nCREATE TABLE t1 (\r\n  id INT PRIMARY KEY, -- pk\r\n  c1 TEXT, -- Metric (\r\n)\r\n\r\nMore notes\r\n"
        "  c2 TEXT DEFAULT 'a\r\n);\r\n\r\nQuestion: fake?'\r\n);\r\n\r\nCREATE TABLE t2 (id INT);\r\n\r\nSelect 1."
    )
    self.assertEqual(strip_cached_schema_from_user_prompt(trap1_crlf), "Select 1.")

    # Verify Stage 4b repair and 2-line synthesis preserve `Question: For Bob's case 1);`.
    t2 = TableSchema("t2", (ColumnSchema("id", "INT", is_primary_key=True),))
    s_spilled = DatabaseSchema("db", (TableSchema("t1", (ColumnSchema("id", "INT", is_primary_key=True), ColumnSchema("c", "TEXT", description="Header:\n  Student's note"))), t2))
    repair_p = (
        "You are repairing a SQLite query that failed semantic or execution checks.\n\n"
        f"Database Schema:\n{render_schema_ddl(s_spilled)}\n\n"
        "Question: For Bob's case 1);\nWhat is the total count?\n\n"
        "Previous SQL:\nSELECT COUNT(*) FROM (\n  SELECT id FROM t1\n);\n\n"
        "Detected Issues:\n- [EXECUTION_ERROR] err"
    )
    stripped_repair = strip_cached_schema_from_user_prompt(repair_p)
    self.assertIn("Question: For Bob's case 1);\nWhat is the total count?", stripped_repair)
    self.assertNotIn("CREATE TABLE", stripped_repair)
    self.assertEqual(
        strip_cached_schema_from_user_prompt(f"Database Schema:\n{render_schema_ddl(s_spilled)}\n\nQuestion: For Bob's case 1);\nWhat is the total count?"),
        "Question: For Bob's case 1);\nWhat is the total count?",
    )

    # Verify `Case S1` (`CREATE TABLE shadow` & `Legacy syntax (v1)` in prose), `Bucket B`, & `H1`.
    for s1_desc in (
        "Notes:\n  CREATE TABLE shadow (a INT)\n  Student's value",
        "Notes:\n  Legacy syntax (v1)\n  CREATE TABLE shadow (a INT)\n  Student's value",
        "Notes:\n  CREATE TEMP TABLE s1 (a INT)\n  CREATE VIRTUAL TABLE s2 USING fts5(x)\n  Student's value",
    ):
      s_ct = DatabaseSchema("db", (TableSchema("t1", (ColumnSchema("id", "INT", is_primary_key=True), ColumnSchema("c", "TEXT", description=s1_desc))), t2))
      # Confirm spilled `CREATE TABLE` prose never truncates `scan_lim` or deletes `Question:`.
      self.assertEqual(
          strip_cached_schema_from_user_prompt(f"Database Schema:\n{render_schema_ddl(s_ct)}\n\nQuestion: What is Bob's score in case 1)\n\nInstruction: SQL."),
          "Question: What is Bob's score in case 1)\n\nInstruction: SQL.",
      )
    s_prev = DatabaseSchema("db", (TableSchema("t1", (ColumnSchema("id", "INT", is_primary_key=True), ColumnSchema("c", "TEXT", description="Previous ref:\n  Previous SQL: legacy query"))), t2))
    self.assertEqual(
        strip_cached_schema_from_user_prompt(f"Database Schema:\n{render_schema_ddl(s_prev)}\n\nQuestion: For Bob's case 1);"),
        "Question: For Bob's case 1);",
    )
    p_h1 = (
        "Database Schema:\nCREATE TABLE t1 (\n  id INT PRIMARY KEY,\n  c TEXT DEFAULT 'note:\nPrevious SQL: none'\n);\n\n"
        "CREATE TABLE t2 (\n  id INT PRIMARY KEY, -- pk\n  name TEXT\n);\n\nQuestion: Real Q?\n\nInstruction: SQL."
    )
    self.assertEqual(strip_cached_schema_from_user_prompt(p_h1), "Question: Real Q?\n\nInstruction: SQL.")

    # Verify closed `--`-free multi-line literal `'ex:\n);\nCREATE TABLE shadow;'` in Pass 2.
    p_closed_lit = (
        "Database Schema:\nCREATE TABLE t0 (id INT -- comment\n);\n\n"
        "CREATE TABLE t1 (\n  id INT PRIMARY KEY,\n  c TEXT DEFAULT 'ex:\n);\nCREATE TABLE shadow (x INT);'\n);\n\nQuestion: Real Q?"
    )
    self.assertEqual(strip_cached_schema_from_user_prompt(p_closed_lit), "Question: Real Q?")

    # Group 3: Un-semicoloned `sqlite_master` subqueries (`p1`), `External Evidence:` (`p2`),
    # `challenger_iter5_2` Caveat (bare `Question:` with `\n);`), `H2`, & `auditor_iter5_1` Info 2.
    for prompt_in, expected_out in (
        ("Database Schema:\nCREATE TABLE t1 (\n  id INT -- pk\n)\n\nSelect count from (\n  SELECT 1\n);", "Select count from (\n  SELECT 1\n);"),
        ("Database Schema:\nCREATE TABLE t1 (\n  id INT -- pk\n)\n\nExternal Evidence: x = (\n  1 + 2\n);\n\nQuestion: What is x?", "External Evidence: x = (\n  1 + 2\n);\n\nQuestion: What is x?"),
        ("Database Schema:\nCREATE TABLE t1 (\n  id INT PRIMARY KEY, -- pk\n  val INT\n)\n\nExternal Evidence:\n- ratio = sum(\n  a\n);\n\nQuestion: What is the ratio?", "External Evidence:\n- ratio = sum(\n  a\n);\n\nQuestion: What is the ratio?"),
        ("Database Schema:\nCREATE TABLE t1 (\n  id INT PRIMARY KEY, -- pk\n  val INT\n)\n\nSelect count(*) from (\n  select id from t1\n);\nInstruction: Return SQL.", "Select count(*) from (\n  select id from t1\n);\nInstruction: Return SQL."),
        ("Database Schema:\nCREATE TABLE t1 (\n  id INT PRIMARY KEY, -- pk\n  val TEXT\n)\n\nQuestion: What does this query return:\nSELECT * FROM (\n  SELECT id FROM t1\n);\nwhen run?", "Question: What does this query return:\nSELECT * FROM (\n  SELECT id FROM t1\n);\nwhen run?"),
        ("Database Schema:\nCREATE TABLE t1 (\n  id INT PRIMARY KEY, -- pk\n  val TEXT\n)\n\nQuestion: Why did my snippet fail at line:\n);\nin SQLite?", "Question: Why did my snippet fail at line:\n);\nin SQLite?"),
        ("Database Schema:\nCREATE TABLE t1 (\n  id INT PRIMARY KEY, -- pk\n  c TEXT DEFAULT 'foo\n);\n\nQuestion: fake?\nExternal Evidence: fake!'\n);\n\nSelect 1.", "Select 1."),
        (
            "Database Schema:\nCREATE TABLE t0 (id INT -- c\n);\n\nCREATE TABLE t1 (\n  id INT PRIMARY KEY,\n  c TEXT DEFAULT 'line1\n);\n\nQuestion: fake_lit?'\n) STRICT;\n\nSelect count(*) from main_table.\n\nExternal Evidence: x = 1\n\nInstruction: SQL.",
            "Select count(*) from main_table.\n\nExternal Evidence: x = 1\n\nInstruction: SQL.",
        ),
    ):
      # Confirm exact preservation of subqueries/evidence/questions and zero literal leaks.
      self.assertEqual(strip_cached_schema_from_user_prompt(prompt_in), expected_out)

  def test_iter6_sects_dash_intertable_orphan_and_quote_boundaries(
      self,
  ) -> None:
    """Covers <50ms sects/dash/3-lvl/V1-V4 bounds & Iter 6 orphan/quote/V5 vectors."""
    # Group 1 (`challenger_iter6_1` C1/C1b/C1c/C2/C2b + `reviewer_iter6_1` p_98k + `challenger_iter6_2` V1-V4):
    # Build reusable double-unclosed `CHECK (((x > 0)` table blocks (`;` and un-semicoloned).
    v_semi = "CREATE TABLE t (\n  id INT, -- c\n  x INT CHECK (((x > 0)\n);"
    v_unsemi = "CREATE TABLE t (\n  id INT, -- c\n  x INT CHECK (((x > 0)\n)"
    # Construct a 140-byte `-- ===...` section banner exceeding fixed 128-byte lookbacks.
    banner_140 = "-- " + ("=" * 137)
    r6_unit = "-- h\nCREATE TABLE t(\n  a INT, -- c\n  /* m */ d INT CHECK (((d > 0)\n);\n"
    # Assemble all 66k-99k O(N) complexity probes requiring strict `< 50ms` execution.
    iter6_probes: tuple[tuple[str, str, str], ...] = (
        ("C1-sects-prev-sql-92k", "Database Schema:\nCREATE TABLE t (\n  id INT, -- c\n  v TEXT DEFAULT '\n\nQuestion:\n" + ("\n\nPrevious SQL:" * 6100) + "'\n);\n\nQuestion: Q?", "Question: Q?"),
        ("C1b-sects-instr-91k", "Database Schema:\nCREATE TABLE t (\n  id INT, -- c\n  v TEXT DEFAULT '" + ("\n\nInstruction:" * 6500) + "'\n);\n\nQuestion: Q?", "Question: Q?"),
        ("C1c-sects-ev-spilled-95k", "Database Schema:\nCREATE TABLE t (\n  id INT -- c" + ("\n\nExternal Evidence: x" * 4300) + "\n);\n\nQuestion: Q?", "Question: Q?"),
        ("C2-dash-hdr-3lvl-66k", "Database Schema:\n" + ("-- tbl\nCREATE TABLE t (\n  id INT, -- c\n  x INT CHECK (((x > 0)\n);\n" * 1000) + "\nQuestion: Q?", "Question: Q?"),
        ("p_98k-dash-bc-3lvl", "Database Schema:\n" + (r6_unit * 1400) + "\nQuestion: Q?", "Question: Q?"),
        ("C2b-depth3-parens-99k", "Database Schema:\nCREATE TABLE t (a INT -- c\n, b INT CHECK (" + ("((()))" * 16600) + "0)\n;\n\nQuestion: Q?", "Question: Q?"),
        ("V1-96k-semi-dash-sec", "Database Schema:\n" + "\n-- sec\n".join([v_semi] * 1450) + "\nQuestion: Q?", "Question: Q?"),
        ("V2-96k-140b-banner", "Database Schema:\n" + f"\n{banner_140}\n".join([v_semi] * 480) + "\nQuestion: Q?", "Question: Q?"),
        ("V3-66k-unsemi-sp-dash", "Database Schema:\n" + " \n-- sec\n".join([v_unsemi] * 1000) + "\nQuestion: Q?", "Question: Q?"),
        ("V4-95k-unsemi-blank-140b", "Database Schema:\n" + f"\n\n{banner_140}\n".join([v_unsemi] * 475) + "\n\nQuestion: Q?", "Question: Q?"),
    )
    # Time each 66k-99k adversarial probe individually to enforce `< 50ms` O(N) runtime.
    for label, probe_prompt, expected_tail in iter6_probes:
      t0 = time.perf_counter()
      got_probe = strip_cached_schema_from_user_prompt(probe_prompt)
      elapsed_ms = (time.perf_counter() - t0) * 1000.0
      # Fail immediately if any probe regresses toward quadratic rescanning or leaks DDL.
      self.assertLess(elapsed_ms, 50.0, msg=f"{label}: {elapsed_ms:.2f}ms")
      self.assertEqual(got_probe, expected_tail, msg=label)

    # Group 2 (`reviewer_iter6_1` Finding 1, `auditor_iter6_1` Info Note 2, & Finding 2b):
    # Finding 1: Spilled `Metric (\n)\n\nNotes` + `-- Score (0..100` or `Notes (see part A`.
    s1 = DatabaseSchema("db", (TableSchema("t1", (ColumnSchema("id", "INT", is_primary_key=True), ColumnSchema("c1", "TEXT", description="Metric (\n)\n\nNotes"), ColumnSchema("c2", "TEXT", description="Score (0..100"))),))
    s1_open = DatabaseSchema("db", (TableSchema("t1", (ColumnSchema("id", "INT", is_primary_key=True), ColumnSchema("c1", "TEXT", description="Metric (\n)\n\nNotes (see part A"), ColumnSchema("c2", "TEXT"))),))
    info2_tail = "External Evidence: x = (\n  1 + 2\n);\n\nSelect count(*) from (\n  select 1\n);"
    for schema_f1 in (s1, s1_open):
      ddl_f1 = render_schema_ddl(schema_f1)
      # Verify labeled `Question:`, unlabeled `Select 1.`, and `External Evidence:` + subquery tails across LF & CRLF.
      for nl in ("\n", "\r\n"):
        for tail_f1 in ("Question: What is c2?\n\nInstruction: SQL.", "Select 1.", info2_tail):
          self.assertEqual(
              strip_cached_schema_from_user_prompt(f"Database Schema:\n{ddl_f1}\n\n{tail_f1}".replace("\n", nl)),
              tail_f1.replace("\n", nl),
          )

    # Finding 2b: Spilled `CREATE TABLE s1 (a INT);\n  CREATE TABLE s2 (b INT)` & `/* m\nn */` keep `Question:`.
    t2 = TableSchema("t2", (ColumnSchema("id", "INT", is_primary_key=True),))
    for desc_f2 in (
        "Example:\n  CREATE TABLE s1 (a INT);\n  CREATE TABLE s2 (b INT)\n  Author's and Person's and Admin's value",
        "Note:\n  /* m\n  n */\n  CREATE TABLE shadow (a INT)\n  Student's value",
    ):
      s_ct2 = DatabaseSchema("db", (TableSchema("t1", (ColumnSchema("id", "INT", is_primary_key=True), ColumnSchema("c", "TEXT", description=desc_f2))), t2))
      # Confirm rejected spilled `CREATE TABLE` examples never clamp `scan_lim` mid-table.
      self.assertEqual(
          strip_cached_schema_from_user_prompt(f"Database Schema:\n{render_schema_ddl(s_ct2)}\n\nQuestion: What is Bob's score in case 1)\n\nInstruction: SQL."),
          "Question: What is Bob's score in case 1)\n\nInstruction: SQL.",
      )

    # Group 3 (`reviewer_iter6_1` Finding 3 + `challenger_iter6_1` C3a-C4b + `challenger_iter6_2` V5):
    # Prepare `C3a` quote-shadow DDL, `C3c` subquery description, and `C4a` spoofed-paren schema.
    ddl_c3a = (
        "CREATE TABLE t0 (id INT, -- hdr:\n  Student's note\n  v INT);\n\n"
        "CREATE TABLE t1 (\n  id INT PRIMARY KEY, -- pk\n  c TEXT DEFAULT 'ex:\n);\nCREATE TABLE shadow (x INT);'\n);\n\n"
        "CREATE TABLE t2 (id INT);"
    )
    s_c3c = DatabaseSchema("db", (
        TableSchema("t1", (ColumnSchema("id", "INT", is_primary_key=True), ColumnSchema("c1", "TEXT", description="Example:\n  SELECT * FROM (\n    SELECT 1\n  );\n  Student's note"), ColumnSchema("c2", "INT"))),
        t2,
    ))
    s_c4a = DatabaseSchema("db", (
        TableSchema("tbl_0", (ColumnSchema("id", "INT", is_primary_key=True),)),
        TableSchema("tbl_1", (ColumnSchema("id", "INT", is_primary_key=True, description="Spoofed:\n  Active status)\n\nQuestion: Fake embedded question?"), ColumnSchema("c_0", "NUMERIC(10,2)"))),
    ))
    # Assert exact stripped outputs across `/*...*/`, `"..."`, `` `...` ``, `[...]`, `LIKE's`/`THEN't`, C3a-C3c, & C4a-C4b.
    for prompt_in, expected_out in (
        ("Database Schema:\nCREATE TABLE t1 (\n  id INT PRIMARY KEY, -- pk\n  /* example:\n);\n\nQuestion: fake?\n*/\n  val INT\n);\n\nSelect 1.", "Select 1."),
        ('Database Schema:\nCREATE TABLE t1 (\n  id INT PRIMARY KEY, -- pk\n  c TEXT DEFAULT "a\n);\n\nQuestion: fake?\nb"\n);\n\nSelect 1.', "Select 1."),
        ("Database Schema:\nCREATE TABLE t1 (\n  id INT PRIMARY KEY, -- pk\n  `d\n);\n\nQuestion: fake?\ne` INT\n);\n\nSelect 1.", "Select 1."),
        ("Database Schema:\nCREATE TABLE t1 (\n  id INT PRIMARY KEY, -- pk\n  [f\n);\n\nQuestion: fake?\ng] INT\n);\n\nSelect 1.", "Select 1."),
        ("Database Schema:\nCREATE TABLE t0 (\n  id INT -- Header (\n)\n);\n\nCREATE TABLE t1 (\n  id INT PRIMARY KEY,\n  c TEXT CHECK (c LIKE's\n);\n\nQuestion: fake?'\n  )\n);\n\nQuestion: Real Q?", "Question: Real Q?"),
        (f"Database Schema:\n{ddl_c3a}\n\nQuestion: Real Q?", "Question: Real Q?"),
        ("Database Schema:\nCREATE TABLE t1 (\n  id INT PRIMARY KEY, -- Note:\n  Student's record\n  c2 TEXT CHECK (c2 LIKE's\n);\n  ' || c2)\n);\n\nCREATE TABLE t2 (id INT);\n\nQuestion: Real Q?", "Question: Real Q?"),
        ("Database Schema:\nCREATE TABLE t1 (\n  id INT PRIMARY KEY, -- Note:\n  Student's record\n  c2 TEXT CHECK (CASE WHEN c2 IS's' THEN't\n);\n  ' ELSE 'a' END != '')\n);\n\nCREATE TABLE t2 (id INT);\n\nQuestion: Real Q?", "Question: Real Q?"),
        (f"Database Schema:\n{render_schema_ddl(s_c3c)}\n\nQuestion: Real Q?", "Question: Real Q?"),
        (f"Database Schema:\n{render_schema_ddl(s_c4a)}\n\nSelect count(*) from (\n  select 1\n);", "Select count(*) from (\n  select 1\n);"),
        ("Database Schema:\nCREATE TABLE t1 (\n  id INTEGER PRIMARY KEY, -- pk\n  c_embq TEXT DEFAULT 'a\n);\n\nQuestion: fake?'\n)\n\nQuestion: Why did my snippet fail at line:\n);\nin SQLite?", "Question: Why did my snippet fail at line:\n);\nin SQLite?"),
        ("Database Schema:\nCREATE TABLE t1 (\n  id INT PRIMARY KEY,\n  c TEXT DEFAULT 'line1\n);\n\nQuestion: fake_lit?'\n) STRICT;\n\nExternal Evidence: x = 1\n\nSelect count(*) from main_table.", "External Evidence: x = 1\n\nSelect count(*) from main_table."),
        # Cover multi-line `[...]`/`` `...` `` + embedded `'...'` before inter-table banners (`i=626` & `i=2216`).
        ("Database Schema:\nCREATE TABLE t1 (\n  id INT PRIMARY KEY, -- pk\n  [br_1\n);\n\nQuestion: fake?\nf] INT,\n  c2 TEXT DEFAULT 'ex:\n);\nCREATE TABLE shadow (x INT);'\n) WITHOUT ROWID;\nCREATE TABLE t2 (id INT);\n\nQuestion: Real Q?", "Question: Real Q?"),
        ("Database Schema:\nCREATE TABLE t1 (\n  id INT PRIMARY KEY, -- pk\n  `bt_2\n);\n\nQuestion: fake1?\ne` INT,\n  c3 TEXT DEFAULT 'a\n);\n\nQuestion: fake2?'\n);\n-- ===================================\nCREATE TABLE t2 (id INT);\n\nQuestion: Real Q?", "Question: Real Q?"),
    ):
      for nl in ("\n", "\r\n"):
        self.assertEqual(strip_cached_schema_from_user_prompt(prompt_in.replace("\n", nl)), expected_out.replace("\n", nl))

    # Verify `challenger_iter6_2` V5 (`/* m\nn */` inter-table separator across `k=1..3` tables).
    # Test both single-LF `Question:` and unlabeled `Select 1.` so fallback regexes cannot mask `None`.
    for k_tables in (1, 2, 3):
      v5_body = "\n/* m\nn */\n".join([v_semi] * k_tables)
      self.assertEqual(strip_cached_schema_from_user_prompt(f"Database Schema:\n{v5_body}\nQuestion: Q?"), "Question: Q?", msg=f"V5-q k={k_tables}")
      self.assertEqual(strip_cached_schema_from_user_prompt(f"Database Schema:\n{v5_body}\n\nSelect 1."), "Select 1.", msg=f"V5-sel k={k_tables}")

    # Verify `C3a-quote-shadow` end-to-end through `ContextCacheAdapter(schema_ddl=...)`.
    seen_c3a: list[str] = []
    vertex_llm = VertexGeminiBatchClient(
        "demo-eval-project-01", token_supplier=lambda: ("synthetic-test-bearer", 3600.0)
    )
    ok_c3a = b'{"name":"projects/123456/locations/global/cachedContents/c3a"}'
    with mock.patch.object(
        vertex_llm,
        "_post_generate_once",
        side_effect=lambda pb, _b: (
            seen_c3a.append(json.loads(pb.decode("utf-8"))["contents"][0]["parts"][0]["text"])
            or {"candidates": [{"content": {"parts": [{"text": "SELECT 1"}]}}]}
        ),
    ):
      with ContextCacheAdapter(
          vertex_llm, "db", schema_ddl=ddl_c3a, http_transport=lambda m, u, b, h, t: (200, ok_c3a if m == "POST" else b"{}")
      ) as ca_c3a:
        ca_c3a.batch_complete((LLMCompletionRequest("r_c3a", DEFAULT_SQLITE_SYSTEM_PROMPT, f"Database Schema:\n{ddl_c3a}\n\nQuestion: Real Q?"),))
    # Confirm wire `generateContent` body stripped all three tables including `t2`.
    self.assertEqual(seen_c3a, ["Question: Real Q?"])

  def test_iter7_multitable_sects_apostrophe_subquery_and_spilled_boundaries(
      self,
  ) -> None:
    """Covers Iter 7 <40ms O(N) probes, Layer 1+2 subqueries, Bob's, & spilled DDL."""
    # Group 1 (`challenger_iter7_1` Challenge 1 C1a-C1d + depth-10 parens, LF & CRLF):
    # Verify <40ms O(N) scaling across 1,250 multi-table sects, V2+Select 1;, quotes, & depth-10.
    v_unsemi = "CREATE TABLE t (\n  id INT, -- c\n  x INT CHECK (((x > 0)\n)"
    d10_expr = "(1+(1+(1+(1+(1+(1+(1+(1+(1+(1))))))))))+"
    iter7_probes: tuple[tuple[str, str, str], ...] = (
        ("C1a-multitbl-ins-strict-98k", "Database Schema:\n" + ("CREATE TABLE t (\n  a INT, -- c\n  b TEXT DEFAULT 'x\n\nInstruction: c'\n) STRICT;\n" * 1250) + "\nQuestion: Q?", "Question: Q?"),
        ("C1b-multitbl-ins-semi-dash-95k", "Database Schema:\n" + ("CREATE TABLE t (\n  a INT, -- c\n  b TEXT DEFAULT 'x\n\nInstruction: c'\n); -- e\n" * 1250) + "\nQuestion: Q?", "Question: Q?"),
        ("C1c-V2-unlabeled-semi-96k", "Database Schema:\n" + "\n-- sec\n".join([v_unsemi] * 1470) + "\n\nSelect 1;", "Select 1;"),
        ("C1d-mixed-paren-quote-99k", "Database Schema:\nCREATE TABLE t (a INT -- c\n, b INT CHECK (" + ("((1)+'x')+" * 9900) + "0)\n);\n\nQuestion: Q?", "Question: Q?"),
        ("C1e-depth10-parens-99k", "Database Schema:\nCREATE TABLE t (a INT -- c\n, b INT CHECK (" + (d10_expr * 2475) + "0)\n);\n\nQuestion: Q?", "Question: Q?"),
        ("C1f-crlf-multitbl-ins", ("Database Schema:\n" + ("CREATE TABLE t (\n  a INT, -- c\n  b TEXT DEFAULT 'x\n\nInstruction: c'\n) STRICT;\n" * 250) + "\nQuestion: Q?").replace("\n", "\r\n"), "Question: Q?"),
    )
    # Enforce strict `< 40ms` per-probe bound and exact DDL-free output on every probe.
    for label, probe_prompt, expected_tail in iter7_probes:
      t0 = time.perf_counter()
      got_probe = strip_cached_schema_from_user_prompt(probe_prompt)
      elapsed_ms = (time.perf_counter() - t0) * 1000.0
      self.assertLess(elapsed_ms, 40.0, msg=f"{label}: {elapsed_ms:.2f}ms")
      self.assertEqual(got_probe, expected_tail, msg=label)

    # Group 2 (`challenger_iter7_1` Challenge 2: Layer 1+2 wire + direct unlabeled subquery):
    # Plain-English trailing directive after `\n);` must never swallow the user SQL query.
    s_c2 = DatabaseSchema("db", (TableSchema("t_0", (ColumnSchema("id", "INT", is_primary_key=True, description="Unique ID; non-null"),)),))
    q_c2 = "Select count(*) from (\n  select id from t_0\n);\nReturn only SQL."
    raw_c2 = f"Database Schema:\n{render_schema_ddl(s_c2)}\n\n{q_c2}"
    seen_c2: list[str] = []
    vertex_llm = VertexGeminiBatchClient("demo-eval-project-01", token_supplier=lambda: ("synthetic-test-bearer", 3600.0))
    ok_c2 = b'{"name":"projects/123456/locations/global/cachedContents/c2"}'
    # Dispatch LF and CRLF requests in one batch through `ContextCacheAdapter(schema=...)`.
    with mock.patch.object(
        vertex_llm,
        "_post_generate_once",
        side_effect=lambda pb, _b: (
            seen_c2.append(json.loads(pb.decode("utf-8"))["contents"][0]["parts"][0]["text"])
            or {"candidates": [{"content": {"parts": [{"text": "SELECT 1"}]}}]}
        ),
    ):
      with ContextCacheAdapter(
          vertex_llm, "db", schema=s_c2, http_transport=lambda m, u, b, h, t: (200, ok_c2 if m == "POST" else b"{}")
      ) as ca_c2:
        ca_c2.batch_complete((
            LLMCompletionRequest("r_c2_lf", DEFAULT_SQLITE_SYSTEM_PROMPT, raw_c2),
            LLMCompletionRequest("r_c2_crlf", DEFAULT_SQLITE_SYSTEM_PROMPT, raw_c2.replace("\n", "\r\n")),
        ))
    # Verify both wire payloads and direct Layer 2 stripping preserve `q_c2` across LF & CRLF.
    self.assertEqual(seen_c2, [q_c2, q_c2.replace("\n", "\r\n")])
    for nl in ("\n", "\r\n"):
      self.assertEqual(strip_cached_schema_from_user_prompt(raw_c2.replace("\n", nl)), q_c2.replace("\n", nl))

    # Group 3 (`reviewer_iter7_1` Finding 1a: `build_strategy_request` + `sqlite3` `Bob's`):
    # Apostrophe in `Question:` must never eat `)` in `External Evidence: ratio = (\n  1 + 2\n);`.
    s_1a = DatabaseSchema("db", (TableSchema("t1", (ColumnSchema("id", "INT", is_primary_key=True, description="Example:\n  SELECT * FROM (\n    SELECT 1\n  );\n  Student's note"),)),))
    req_1a = build_strategy_request("r1", render_schema_ddl(s_1a), "What is Bob's score in case 1)", "ratio = (\n  1 + 2\n);", "zero_shot")
    ddl_1a_sql = 'CREATE TABLE t1 (\n  id INTEGER PRIMARY KEY,\n  "order" INT /* comment:\n  multi-line */ -- col note (0..100)\n) STRICT;'
    ddl_aud2_sql = "CREATE TABLE t1 (\n  id INT PRIMARY KEY,\n  c TEXT ABORT's\n);\n\nQuestion: fake?'\n);"
    # Validate synthetic SQLite DDLs directly against in-memory `sqlite3` parser.
    with sqlite3.connect(":memory:") as conn:
      conn.execute(ddl_1a_sql)
    with sqlite3.connect(":memory:") as conn:
      conn.execute(ddl_aud2_sql)
    tail_1a = "Question: What is Bob's score in case 1)\nExternal Evidence: ratio = (\n  1 + 2\n);\n\nInstruction: SQL."
    for nl in ("\n", "\r\n"):
      got_req = strip_cached_schema_from_user_prompt(req_1a.user_prompt.replace("\n", nl))
      self.assertTrue(got_req.startswith("Question: What is Bob's score in case 1)\nExternal Evidence: ratio = (\n  1 + 2\n);".replace("\n", nl)))
      self.assertNotIn("CREATE TABLE", got_req)
      self.assertEqual(strip_cached_schema_from_user_prompt(f"Database Schema:\n{ddl_1a_sql}\n{tail_1a}".replace("\n", nl)), tail_1a.replace("\n", nl))

    # Group 4 (`challenger_iter7_1` C3a-C3c, `reviewer_iter7_1` 1b/2a/2b, & `auditor_iter7_1` Info 1-2):
    # Assemble reusable schema fixtures for spilled prose, embedded headers, & double-spoof cases.
    t2 = TableSchema("t2", (ColumnSchema("id", "INT", is_primary_key=True),))
    s_c3b = DatabaseSchema("db", (TableSchema("t1", (ColumnSchema("id", "INT", is_primary_key=True), ColumnSchema("c1", "TEXT", description="Note:\n  Author's example:\n  SELECT * FROM (\n    SELECT 1\n  );\n  Instruction: internal note"), ColumnSchema("c2", "INT"))), t2))
    s_1b_v = DatabaseSchema("db", (TableSchema("t1", (ColumnSchema("id", "INT", is_primary_key=True), ColumnSchema("c1", "TEXT", description="Status (\n)\n\nValues (1..5"), ColumnSchema("c2", "TEXT", description="Score (0..100)"))),))
    s_1b_s = DatabaseSchema("db", (TableSchema("t1", (ColumnSchema("id", "INT", is_primary_key=True), ColumnSchema("c1", "TEXT", description="Flag (\n)\n\nSelect (0 or 1"), ColumnSchema("c2", "TEXT", description="Score (0..100)"))),))
    s_2a = DatabaseSchema("db", (TableSchema("t1", (ColumnSchema("id", "INT", is_primary_key=True), ColumnSchema("c1", "TEXT", description="Metric (\n)\n\nExternal Evidence: fake"))), t2))
    s_2b = DatabaseSchema("db", (TableSchema("t1", (ColumnSchema("id", "INT", is_primary_key=True), ColumnSchema("c1", "TEXT", description="Metric (\n  line 2\n)\n\nExternal Evidence: fake"), ColumnSchema("c2", "TEXT", description="Score"))), t2))
    s_2c = DatabaseSchema("db", (TableSchema("t1", (ColumnSchema("id", "INT", is_primary_key=True), ColumnSchema("c1", "TEXT", description="Metric (\n)\n\nQuestion: Fake?"))),))
    s_aud1 = DatabaseSchema("db", (TableSchema("t1", (ColumnSchema("id", "INT", is_primary_key=True), ColumnSchema("c1", "TEXT", description="Spoofed:\n  Active status)\n\nQuestion: Fake embedded?"))),))
    tail_aud1 = "Question: Why did my query fail at:\n);\nin SQLite?\n\nPrevious SQL:\nSELECT 1\n\nDetected Issues:\n- [ERR] syntax"
    # Verify exact stripped outputs across both LF and CRLF for every C3, R1b/R2, & Auditor vector.
    for prompt_in, expected_out in (
        ("Database Schema:\nCREATE TABLE t1 (\n  id INT PRIMARY KEY,\n  v TEXT DEFAULT 'example:\n\nQuestion: fake_q\n\nPrevious SQL: fake_prev'\n);\n\nCREATE TABLE t2 (\n  id INT PRIMARY KEY, -- pk\n  name TEXT\n);\n\nQuestion: Real question?\n\nDetected Issues:\n- [ERR] x", "Question: Real question?\n\nDetected Issues:\n- [ERR] x"),
        (f"Database Schema:\n{render_schema_ddl(s_c3b)}\n\nQuestion: Real Q?", "Question: Real Q?"),
        ("Database Schema:\nCREATE TABLE t_0 (\n  id INT PRIMARY KEY, -- Header:\n  /* note */ Student's value\n  v INT\n);\n\nCREATE TABLE t_1 (\n  id INTEGER PRIMARY KEY, -- pk\n  [br_1\n);\n\nQuestion: fake_br?] INT,\n  `bt_1\n);\n\nQuestion: fake_bt?` INT\n)\n\nQuestion: Real Q?", "Question: Real Q?"),
        ("Database Schema:\nCREATE TABLE t_0 (\n  id INT PRIMARY KEY, -- Example:\n  SELECT * FROM (\n    SELECT 1\n  );\n  Student's note\n  v INT\n)\n\nCREATE TABLE t_1 (\n  id INT PRIMARY KEY\n)\n\nQuestion: Real Q?", "Question: Real Q?"),
        (f"Database Schema:\n{render_schema_ddl(s_1b_v)}\n\nQuestion: What is c2?\n\nInstruction: SQL.", "Question: What is c2?\n\nInstruction: SQL."),
        (f"Database Schema:\n{render_schema_ddl(s_1b_s)}\n\nQuestion: What is c2?\n\nInstruction: SQL.", "Question: What is c2?\n\nInstruction: SQL."),
        (f"Database Schema:\n{render_schema_ddl(s_2a)}\n\nQuestion: Real Q?\n\nInstruction: SQL.", "Question: Real Q?\n\nInstruction: SQL."),
        (f"Database Schema:\n{render_schema_ddl(s_2b)}\n\nQuestion: Real Q?\n\nInstruction: SQL.", "Question: Real Q?\n\nInstruction: SQL."),
        (f"Database Schema:\n{render_schema_ddl(s_2c)}\n\nExternal Evidence: x = 1\n\nSelect count(*) from t1;", "External Evidence: x = 1\n\nSelect count(*) from t1;"),
        (f"Database Schema:\n{render_schema_ddl(s_aud1)}\n\n{tail_aud1}", tail_aud1),
        ("Database Schema:\nCREATE TABLE t1 (\n  id INT PRIMARY KEY, -- Note:\n  Student's record\n  c TEXT ABORT's\n);\n\nQuestion: fake?'\n);\n\nCREATE TABLE t2 (id INT);\n\nQuestion: Real Q?", "Question: Real Q?"),
    ):
      for nl in ("\n", "\r\n"):
        self.assertEqual(strip_cached_schema_from_user_prompt(prompt_in.replace("\n", nl)), expected_out.replace("\n", nl))


if __name__ == "__main__":
  unittest.main()
