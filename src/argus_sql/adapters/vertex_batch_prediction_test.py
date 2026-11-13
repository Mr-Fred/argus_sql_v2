"""Hermetic offline unit & 7-system E2E tests for Vertex Batch Prediction (`R2`–`R5`)."""

from email.message import Message
import http.client
import json
import pathlib
import tempfile
import unittest
from unittest import mock
import urllib.error
import urllib.parse

from argus_sql.adapters import BatchJobManifest
from argus_sql.adapters import VertexBatchPredictionRunner
from argus_sql.adapters import build_intent_llm_client
from argus_sql.adapters.deterministic_synthesizer import SchemaGuidedDiagnosticSynthesizer
from argus_sql.adapters.exceptions import LLMAdapterAuthError
from argus_sql.adapters.exceptions import LLMAdapterTransportError
from argus_sql.adapters.types import LLMCompletionRequest
from argus_sql.adapters.vertex_batch_prediction import _batch_request_digest
from argus_sql.adapters.vertex_batch_prediction import _build_batch_request_dict
from argus_sql.adapters.vertex_batch_prediction import parse_and_validate_gcs_uri
from argus_sql.adapters.vertex_batch_prediction import serialize_batch_requests_jsonl
from argus_sql.adapters.vertex_batch_prediction import validate_batch_job_name
from argus_sql.adapters.vertex_gemini_client import VertexGeminiBatchClient
from argus_sql.evaluation.mini_bird_databases import materialize_mini_bird_databases
from argus_sql.evaluation.mini_bird_suite import build_main_systems
from argus_sql.evaluation.mini_bird_suite import build_mini_bird_clean_tasks
from argus_sql.evaluation.mini_bird_suite import build_voice_bird_tasks
from argus_sql.evaluation.mini_bird_suite import run_benchmark_suite
from argus_sql.guardrails.clean_room_scanner import scan_text_for_leaks
from argus_sql.linking.intent_canonicalizer import canonicalize_intent
from argus_sql.sandbox.sqlite_sandbox import SQLiteReadOnlySandbox
from argus_sql.sandbox.types import SandboxPolicy


class FakeVertexBatchTransport:
  """In-memory hermetic router for GCS JSON API v1 and regional Vertex batch jobs."""

  def __init__(
      self,
      *,
      page_size: int = 10,
      poll_states: list[str] | None = None,
      auto_synthesize: bool = True,
      shard_count: int = 2,
      include_thought_parts: bool = False,
      strip_echoed_labels: bool = False,
  ) -> None:
    """Initializes in-memory GCS object storage and queued job state transitions."""
    # Configure pagination page size, state transitions, and shard count.
    self.page_size = page_size
    self.poll_states = list(poll_states) if poll_states is not None else ["JOB_STATE_SUCCEEDED"]
    self.auto_synthesize = auto_synthesize
    self.shard_count = max(1, shard_count)
    self.include_thought_parts = include_thought_parts
    self.strip_echoed_labels = strip_echoed_labels
    # Maintain in-memory `(bucket, object_name) -> bytes` map and call audit log.
    self.gcs_objects: dict[tuple[str, str], bytes] = {}
    self.calls: list[tuple[str, str, bytes | None, dict[str, str]]] = []
    self.job_counter = 0
    self.job_output_dirs: dict[str, str] = {}
    self.job_error_msg = "Simulated upstream batch quota failure"
    self.recorded_line_configs: list[tuple[int, bool]] = []
    self._synth = SchemaGuidedDiagnosticSynthesizer()

  def _synthesize_shard_rows(
      self,
      input_bytes: bytes,
      bucket: str,
      out_dir_obj: str,
  ) -> None:
    """Generates reversed/shuffled multi-shard `predictions-*.jsonl` objects."""
    raw_lines = [ln for ln in input_bytes.decode("utf-8").splitlines() if ln.strip()]
    out_rows: list[str] = []
    # Reverse input order so tests prove unordered correlation aligns 1-to-1.
    for idx, raw_ln in enumerate(reversed(raw_lines)):
      wrapper = json.loads(raw_ln)
      req_dict = dict(wrapper["request"])
      gen_cfg = dict(req_dict.get("generationConfig", {}))
      think_cfg = dict(gen_cfg.get("thinkingConfig", {}))
      tb_val = int(think_cfg.get("thinkingBudget", -1))
      has_max_out = "maxOutputTokens" in gen_cfg
      # Record `(thinkingBudget, has_maxOutputTokens)` for R3 E2E verification.
      self.recorded_line_configs.append((tb_val, has_max_out))

      # Extract prompt text and delegate SQL synthesis to offline synthesizer.
      user_txt = str(req_dict["contents"][0]["parts"][0]["text"])
      sys_txt = ""
      if "systemInstruction" in req_dict:
        sys_txt = str(req_dict["systemInstruction"]["parts"][0]["text"])
      strat_hint = "llm_judge" if "Option A" in user_txt else "decomposed_cot"
      synth_resp = self._synth.batch_complete(
          (LLMCompletionRequest(f"b_{idx}", sys_txt, user_txt, strategy_tag=strat_hint),)
      )[0]
      visible_text = (
          synth_resp.raw_text
          if strat_hint == "llm_judge"
          else f"```sql\n{synth_resp.sql_text};\n```"
      )

      # Prepend optional `thought=True` draft SQL to test thought scratchpad filtering.
      parts: list[dict[str, object]] = []
      if self.include_thought_parts:
        parts.append({"thought": True, "text": "SELECT wrong_draft FROM bad_tbl;"})
      parts.append({"text": visible_text})

      # Optionally strip `labels` to force canonical echoed-request SHA-256 fallback.
      echoed = dict(req_dict)
      if self.strip_echoed_labels:
        echoed.pop("labels", None)
        # Convert `0.0` to integer `0` to exercise canonical float normalization.
        echoed["generationConfig"] = {
            "thinkingConfig": think_cfg,
            "temperature": 0,
        }

      # Emit standard Vertex AI `BatchPredictionJob` output JSONL schema row.
      row_obj = {
          "request": echoed,
          "response": {
              "candidates": [{"content": {"role": "model", "parts": parts}}],
              "usageMetadata": {
                  "promptTokenCount": 40,
                  "candidatesTokenCount": 12,
                  "thoughtsTokenCount": 18 if self.include_thought_parts else 0,
              },
          },
          "status": "",
      }
      out_rows.append(json.dumps(row_obj))

    # Partition rows across `self.shard_count` `.jsonl` files plus a `.txt` sidecar.
    for shard_idx in range(self.shard_count):
      slice_rows = out_rows[shard_idx :: self.shard_count]
      if slice_rows or shard_idx == 0:
        shard_key = (
            f"{out_dir_obj}/predictions-{shard_idx:05d}-of-{self.shard_count:05d}.jsonl"
        )
        self.gcs_objects[(bucket, shard_key)] = ("\n".join(slice_rows)).encode("utf-8")
    # Include non-JSONL sidecar file to verify `.jsonl` suffix filtering.
    self.gcs_objects[(bucket, f"{out_dir_obj}/summary.txt")] = b"non-jsonl metadata"

  def __call__(
      self,
      method: str,
      url: str,
      body: bytes | None,
      headers: dict[str, str],
  ) -> bytes:
    """Routes GCS upload/list/download and regional `batchPredictionJobs` calls."""
    # Log every outbound HTTP request for post-call assertions.
    self.calls.append((method, url, body, dict(headers)))
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)

    # 1. GCS Simple Media Upload: `POST /upload/storage/v1/b/{bkt}/o?uploadType=media&name=...`
    if method == "POST" and parsed.netloc == "storage.googleapis.com":
      bkt = parsed.path.split("/b/", 1)[1].split("/o", 1)[0]
      obj_name = query["name"][0]
      self.gcs_objects[(bkt, obj_name)] = body or b""
      return json.dumps({"bucket": bkt, "name": obj_name}).encode("utf-8")

    # 2. Regional `batchPredictionJobs.create`: `POST .../v1/projects/.../batchPredictionJobs`
    if method == "POST" and parsed.path.endswith("/batchPredictionJobs"):
      assert body is not None
      job_req = json.loads(body.decode("utf-8"))
      loc = parsed.path.split("/locations/", 1)[1].split("/", 1)[0]
      self.job_counter += 1
      # Return server-canonicalized numeric GCP project number `123456789012`.
      job_name = (
          f"projects/123456789012/locations/{loc}/batchPredictionJobs/"
          f"70000{self.job_counter}"
      )
      in_uri = str(job_req["inputConfig"]["gcsSource"]["uris"][0])
      out_pfx = str(job_req["outputConfig"]["gcsDestination"]["outputUriPrefix"])
      in_bkt, in_obj = parse_and_validate_gcs_uri(in_uri)
      out_bkt, out_obj = parse_and_validate_gcs_uri(out_pfx)
      # Include ISO-8601 timestamp colons in Vertex output subdirectory path.
      pred_dir = f"{out_obj}/prediction-model-2026-11-11T05:00:00.000Z"
      self.job_output_dirs[job_name] = f"gs://{out_bkt}/{pred_dir}"
      if self.auto_synthesize:
        self._synthesize_shard_rows(self.gcs_objects[(in_bkt, in_obj)], out_bkt, pred_dir)
      return json.dumps({"name": job_name, "state": "JOB_STATE_PENDING"}).encode("utf-8")

    # 3. Regional `batchPredictionJobs.get`: `GET .../v1/projects/.../batchPredictionJobs/{id}`
    if method == "GET" and "/batchPredictionJobs/" in parsed.path:
      job_name = parsed.path.split("/v1/", 1)[1]
      # Advance through queued non-terminal states until the final terminal state.
      if len(self.poll_states) > 1:
        state = self.poll_states.pop(0)
      else:
        state = self.poll_states[0]
      resp_dict: dict[str, object] = {
          "name": job_name,
          "state": state,
          "outputInfo": {"gcsOutputDirectory": self.job_output_dirs.get(job_name, "")},
          "completionStats": {"successfulCount": 4, "failedCount": 0},
      }
      if state == "JOB_STATE_FAILED":
        resp_dict["error"] = {"message": self.job_error_msg}
      return json.dumps(resp_dict).encode("utf-8")

    # 4. GCS Object Media Download: `GET /storage/v1/b/{bkt}/o/{pct_encoded}?alt=media`
    if method == "GET" and parsed.netloc == "storage.googleapis.com" and query.get("alt") == ["media"]:
      raw_after_o = parsed.path.split("/o/", 1)[1]
      # Enforce `%2F` slash encoding: raw `/` inside `/o/...` returns HTTP 404!
      if "/" in raw_after_o:
        raise urllib.error.HTTPError(url, 404, "Unescaped slash in GCS path", Message(), None)
      bkt = parsed.path.split("/b/", 1)[1].split("/o/", 1)[0]
      obj_name = urllib.parse.unquote(raw_after_o)
      return self.gcs_objects[(bkt, obj_name)]

    # 5. GCS Object List: `GET /storage/v1/b/{bkt}/o?prefix=...`
    if method == "GET" and parsed.netloc == "storage.googleapis.com":
      bkt = parsed.path.split("/b/", 1)[1].split("/o", 1)[0]
      pfx = query.get("prefix", [""])[0]
      matching = sorted(k[1] for k in self.gcs_objects if k[0] == bkt and k[1].startswith(pfx))
      # Paginate deterministic sorted object keys using numeric `pageToken` offsets.
      offset = int(query.get("pageToken", ["0"])[0])
      page = matching[offset : offset + self.page_size]
      listing: dict[str, object] = {"items": [{"name": n} for n in page]}
      if offset + self.page_size < len(matching):
        listing["nextPageToken"] = str(offset + self.page_size)
      return json.dumps(listing).encode("utf-8")

    raise AssertionError(f"Unexpected transport call: {method} {url}")


class VertexBatchPredictionTest(unittest.TestCase):
  """Covers T1–T12 hermetic unit and 7-system E2E batch runner requirements."""

  def test_dynamic_env_config_and_global_to_regional_mapping(self) -> None:
    """T1: Maps `global` -> `us-central1`, honors batch env, and fails closed."""
    # Verify dynamic environment variables configure project, region, GCS, & budget.
    with mock.patch.dict(
        "os.environ",
        {
            "GOOGLE_CLOUD_PROJECT": "demo-eval-project-01",
            "GOOGLE_CLOUD_LOCATION": "global",
            "ARGUS_GCS_BATCH_URI": "gs://argus-eval-bucket/batch",
            "ARGUS_THINKING_BUDGET": "4096",
        },
        clear=True,
    ):
      runner = VertexBatchPredictionRunner(
          token_supplier=lambda: ("synthetic-test-bearer", 3600.0)
      )
      self.assertEqual(runner.location, "us-central1")
      self.assertEqual(runner.model_name, "gemini-3.7-flash")
      self.assertEqual(runner.thinking_budget, 4096)
      self.assertEqual(runner.gcs_uri_prefix, "gs://argus-eval-bucket/batch")

      # `GOOGLE_CLOUD_BATCH_LOCATION` takes precedence over `GOOGLE_CLOUD_LOCATION`.
      with mock.patch.dict("os.environ", {"GOOGLE_CLOUD_BATCH_LOCATION": "europe-west4"}):
        eu_runner = VertexBatchPredictionRunner(
            token_supplier=lambda: ("synthetic-test-bearer", 3600.0)
        )
        self.assertEqual(eu_runner.location, "europe-west4")

    # Placeholder project, missing/non-gs URI, traversal, or bad budget raise AuthError.
    with mock.patch.dict("os.environ", {}, clear=True):
      # Reject template placeholder project IDs from `.env.example`.
      with self.assertRaises(LLMAdapterAuthError):
        VertexBatchPredictionRunner(
            project_id="your-personal-gcp-project-id",
            gcs_uri_prefix="gs://argus-eval-bucket/b",
        )
      # Reject `https://` URIs where `gs://` Cloud Storage URIs are required.
      with self.assertRaises(LLMAdapterAuthError):
        VertexBatchPredictionRunner(
            project_id="demo-eval-project-01",
            gcs_uri_prefix="https://storage.googleapis.com/b",
        )
      # Reject `..` path traversal segments in GCS staging prefixes.
      with self.assertRaises(LLMAdapterAuthError):
        VertexBatchPredictionRunner(
            project_id="demo-eval-project-01",
            gcs_uri_prefix="gs://argus-eval-bucket/../bad",
        )
      # Reject boolean `thinking_budget=True` (`bool` subclasses `int`).
      with self.assertRaises(LLMAdapterAuthError):
        VertexBatchPredictionRunner(
            project_id="demo-eval-project-01",
            gcs_uri_prefix="gs://argus-eval-bucket/b",
            thinking_budget=True,
        )

  def test_jsonl_serialization_dedup_thinking_budget_and_unclamped_output(
      self,
  ) -> None:
    """T2: Collapses SHA-256 duplicates, enforces R3 budgets, and caps labels <=63."""
    # Combine B0, B0+, duplicate B1 CoT, and Stage 1 intent requests in one batch.
    reqs = (
        LLMCompletionRequest("b0", "sys", "Q1", strategy_tag="zero_shot", max_tokens=512),
        LLMCompletionRequest("b0p", "sys", "Q2", strategy_tag="zero_shot"),
        LLMCompletionRequest("b1_a", "sys", "Q_cot", strategy_tag="decomposed_cot"),
        LLMCompletionRequest("b1_dup", "sys", "Q_cot", strategy_tag="decomposed_cot"),
        LLMCompletionRequest("s1", "sys", "Q_int", strategy_tag="intent_flash_lite"),
    )
    # Configure 1024 to verify reasoning floor upgrades `decomposed_cot` to 2048.
    jsonl_bytes, entries = serialize_batch_requests_jsonl(reqs, configured_thinking_budget=1024)
    lines = jsonl_bytes.decode("utf-8").splitlines()
    self.assertEqual(len(entries), 5)
    self.assertEqual(len(lines), 4)

    # Inspect every serialized JSONL request for R3 & GCP label length rules.
    budgets: list[int] = []
    for raw_ln in lines:
      req_obj = json.loads(raw_ln)["request"]
      gen_cfg = req_obj["generationConfig"]
      self.assertNotIn("maxOutputTokens", gen_cfg)
      self.assertNotIn("role", req_obj["systemInstruction"])
      lbl = req_obj["labels"]["argus_hash"]
      self.assertEqual(len(lbl), 32)
      budgets.append(int(gen_cfg["thinkingConfig"]["thinkingBudget"]))
    self.assertEqual(budgets, [0, 0, 2048, 0])

  def test_unordered_multishard_pagination_pct2f_download_and_thought_filtering(
      self,
  ) -> None:
    """T3: Paginates shards, `%2F`-downloads, strips `thought=True`, and aligns."""
    # Force multi-page GCS listing (`page_size=2` over 4 objects) and 3 shards.
    transport = FakeVertexBatchTransport(
        page_size=2,
        poll_states=["JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_SUCCEEDED"],
        shard_count=3,
        include_thought_parts=True,
    )
    ddl_base = "CREATE TABLE schools (cds_code TEXT PRIMARY KEY, county TEXT);\n\n"
    reqs = (
        LLMCompletionRequest("r1", "sys", f"{ddl_base}Question: Count schools in Alameda?"),
        LLMCompletionRequest("r2", "sys", f"{ddl_base}Question: Count schools in Fresno?"),
        LLMCompletionRequest("r3_dup", "sys", f"{ddl_base}Question: Count schools in Alameda?"),
        LLMCompletionRequest("r4", "sys", f"{ddl_base}Question: Total schools?"),
    )
    # Execute unified `BatchLLMPort.batch_complete` with no-op `sleep_fn`.
    runner = VertexBatchPredictionRunner(
        project_id="demo-eval-project-01",
        gcs_uri_prefix="gs://argus-eval-bucket/runs",
        location="global",
        token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
        http_transport=transport,
        sleep_fn=lambda _: None,
    )
    resps = runner.batch_complete(reqs)
    self.assertEqual([r.request_id for r in resps], ["r1", "r2", "r3_dup", "r4"])
    # Ensure `thought=True` scratchpads never leak into extracted SQL text.
    for resp in resps:
      self.assertNotIn("wrong_draft", resp.sql_text)
      self.assertTrue(resp.sql_text.upper().startswith("SELECT"))

    # Duplicate `r3_dup` has 0 marginal billed tokens while preserving `logical_tokens`.
    self.assertFalse(resps[0].from_cache)
    self.assertEqual(resps[0].prompt_tokens, 40)
    self.assertEqual(resps[0].completion_tokens, 30)
    self.assertTrue(resps[2].from_cache)
    self.assertEqual(resps[2].prompt_tokens, 0)
    self.assertEqual(resps[2].completion_tokens, 0)
    self.assertEqual(resps[2].logical_tokens, 70)

    # Confirm 2 paginated GCS list calls and `%2F` media URLs (never raw `/`).
    list_urls = [u for m, u, _, _ in transport.calls if m == "GET" and "prefix=" in u]
    dl_urls = [u for m, u, _, _ in transport.calls if m == "GET" and "alt=media" in u]
    self.assertEqual(len(list_urls), 2)
    self.assertEqual(len(dl_urls), 3)
    for dl_u in dl_urls:
      self.assertIn("%2F", dl_u)
      self.assertNotIn("summary.txt", dl_u)

  def test_canonical_echoed_request_sha256_fallback_without_labels(self) -> None:
    """T4: Correlates unordered rows via canonical echoed JSON when labels strip."""
    # Strip echoed `labels` and rewrite `temperature: 0.0 -> 0` in output rows.
    transport = FakeVertexBatchTransport(strip_echoed_labels=True, shard_count=2)
    ddl = "CREATE TABLE loans (loan_id INT PRIMARY KEY);\n\n"
    reqs = (
        LLMCompletionRequest("k1", "sys", f"{ddl}Question: Count loans?", strategy_tag="zero_shot"),
        LLMCompletionRequest("k2", "sys", f"{ddl}Question: Max loan_id?", strategy_tag="decomposed_cot"),
    )
    runner = VertexBatchPredictionRunner(
        project_id="demo-eval-project-01",
        gcs_uri_prefix="gs://argus-eval-bucket/runs",
        token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
        http_transport=transport,
        sleep_fn=lambda _: None,
    )
    # Confirm canonical `_batch_request_digest` fallback aligns `k1` and `k2`.
    resps = runner.batch_complete(reqs)
    self.assertEqual([r.request_id for r in resps], ["k1", "k2"])

  def test_decoupled_ci_manifest_save_load_poll_and_collect_zero_secrets(
      self,
  ) -> None:
    """T5: Decoupled submit -> save_json -> load_json -> poll -> collect has 0 leaks."""
    # Simulate CI Stage 1: upload JSONL, submit job, and persist `manifest.json`.
    transport = FakeVertexBatchTransport(
        poll_states=["JOB_STATE_PENDING", "JOB_STATE_SUCCEEDED"]
    )
    ddl = "CREATE TABLE schools (cds_code TEXT PRIMARY KEY);\n\n"
    reqs = (LLMCompletionRequest("ci_1", "SECRET_PROMPT_TEXT", f"{ddl}Question: Count?"),)

    runner_submit = VertexBatchPredictionRunner(
        project_id="demo-eval-project-01",
        gcs_uri_prefix="gs://argus-eval-bucket/ci",
        token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
        http_transport=transport,
    )
    # Confirm numeric GCP project number in `job_name` passes allow-list validation.
    manifest = runner_submit.submit_job(reqs)
    self.assertTrue(manifest.job_name.startswith("projects/123456789012/"))

    with tempfile.TemporaryDirectory() as tmp_dir:
      manifest_file = pathlib.Path(tmp_dir) / "manifest.json"
      manifest.save_json(manifest_file)
      raw_manifest_text = manifest_file.read_text(encoding="utf-8")

      # Verify zero clean-room findings, zero bearer tokens, and zero prompt SQL.
      self.assertEqual(scan_text_for_leaks(raw_manifest_text, "manifest.json"), ())
      self.assertNotIn("synthetic-test-bearer", raw_manifest_text)
      self.assertNotIn("SECRET_PROMPT_TEXT", raw_manifest_text)

      # Symlink read/write targets must fail closed with `LLMAdapterTransportError`.
      symlink_path = pathlib.Path(tmp_dir) / "link_manifest.json"
      symlink_path.symlink_to(manifest_file)
      with self.assertRaises(LLMAdapterTransportError):
        manifest.save_json(symlink_path)
      with self.assertRaises(LLMAdapterTransportError):
        BatchJobManifest.load_json(symlink_path)

      # Simulate separate CI container loading manifest and polling to completion.
      loaded = BatchJobManifest.load_json(manifest_file)
      runner_collect = VertexBatchPredictionRunner(
          project_id="demo-eval-project-01",
          gcs_uri_prefix="gs://argus-eval-bucket/ci",
          token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
          http_transport=transport,
      )
      # First poll sees `JOB_STATE_PENDING`; second poll sees `JOB_STATE_SUCCEEDED`.
      st1 = runner_collect.poll_job(loaded.job_name)
      self.assertFalse(st1.is_terminal)
      st2 = runner_collect.poll_job(loaded.job_name)
      self.assertTrue(st2.is_terminal)
      self.assertTrue(st2.succeeded)
      collected = runner_collect.collect_job_responses(loaded)
      self.assertEqual(len(collected), 1)
      self.assertEqual(collected[0].request_id, "ci_1")

  def test_failed_cancelled_expired_job_states_and_instant_poll_timeout(
      self,
  ) -> None:
    """T6: Terminal failure states and simulated-clock timeouts fail closed fast."""
    # Verify every terminal failure enum aborts immediately without hanging.
    reqs = (LLMCompletionRequest("f1", "sys", "SELECT 1"),)
    for bad_state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
      transport = FakeVertexBatchTransport(poll_states=[bad_state], auto_synthesize=False)
      runner = VertexBatchPredictionRunner(
          project_id="demo-eval-project-01",
          gcs_uri_prefix="gs://argus-eval-bucket/fail",
          token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
          http_transport=transport,
          sleep_fn=lambda _: None,
      )
      manifest = runner.submit_job(reqs)
      with self.assertRaisesRegex(LLMAdapterTransportError, bad_state):
        runner.await_job(manifest.job_name)
      with self.assertRaisesRegex(LLMAdapterTransportError, bad_state):
        runner.collect_job_responses(manifest)

    # Non-terminating `JOB_STATE_RUNNING` terminates deterministically via `sleep_fn`.
    slept: list[float] = []
    stall_transport = FakeVertexBatchTransport(
        poll_states=["JOB_STATE_RUNNING"], auto_synthesize=False
    )
    stall_runner = VertexBatchPredictionRunner(
        project_id="demo-eval-project-01",
        gcs_uri_prefix="gs://argus-eval-bucket/stall",
        poll_interval_sec=5.0,
        poll_timeout_sec=12.0,
        token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
        http_transport=stall_transport,
        sleep_fn=slept.append,
    )
    # Confirm simulated-clock `waited_sec` bounds steps to `[5.0, 5.0, 2.0]`.
    stall_manifest = stall_runner.submit_job(reqs)
    with self.assertRaisesRegex(LLMAdapterTransportError, "timed out"):
      stall_runner.await_job(stall_manifest.job_name)
    self.assertEqual(slept, [5.0, 5.0, 2.0])

  def test_partially_succeeded_row_error_and_missing_digest_fail_closed(
      self,
  ) -> None:
    """T7: Per-row status errors or missing shard digests raise TransportError."""
    # Submit a 2-prompt batch whose job enters `JOB_STATE_PARTIALLY_SUCCEEDED`.
    reqs = (
        LLMCompletionRequest("p1", "sys", "CREATE TABLE t (id INT);\n\nQuestion: Q1?"),
        LLMCompletionRequest("p2", "sys", "CREATE TABLE t (id INT);\n\nQuestion: Q2?"),
    )
    transport = FakeVertexBatchTransport(
        poll_states=["JOB_STATE_PARTIALLY_SUCCEEDED"], auto_synthesize=False
    )
    runner = VertexBatchPredictionRunner(
        project_id="demo-eval-project-01",
        gcs_uri_prefix="gs://argus-eval-bucket/partial",
        token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
        http_transport=transport,
    )
    manifest = runner.submit_job(reqs)
    out_bkt, out_dir = parse_and_validate_gcs_uri(transport.job_output_dirs[manifest.job_name])

    # Case A: Row carries non-empty error `"status": "DEADLINE_EXCEEDED"`.
    bad_row = {
        "custom_id": manifest.entries[0][1],
        "status": "DEADLINE_EXCEEDED",
        "response": {},
    }
    transport.gcs_objects[(out_bkt, f"{out_dir}/predictions-00000.jsonl")] = (
        json.dumps(bad_row).encode("utf-8")
    )
    with self.assertRaises(LLMAdapterTransportError):
      runner.collect_job_responses(manifest)

    # Case B: First row succeeds, second manifest digest is missing from shards.
    good_only_first = {
        "custom_id": manifest.entries[0][1],
        "status": "",
        "response": {
            "candidates": [{"content": {"parts": [{"text": "SELECT 1"}]}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
        },
    }
    transport.gcs_objects[(out_bkt, f"{out_dir}/predictions-00000.jsonl")] = (
        json.dumps(good_only_first).encode("utf-8")
    )
    with self.assertRaisesRegex(LLMAdapterTransportError, "missing 1 of 2"):
      runner.collect_job_responses(manifest)

  def test_http_401_token_refresh_and_503_exponential_backoff(self) -> None:
    """T8: Refreshes bearer on HTTP 401 and backs off once via `sleep_fn` on 503."""
    # Provide rotating token iterator to verify immediate 401 token eviction.
    tokens = iter([("stale-tok", 3600.0), ("fresh-tok", 3600.0)])
    base_router = FakeVertexBatchTransport()
    slept: list[float] = []
    seen_auth: list[str] = []
    poll_attempts = 0

    def _flaky_http(
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
    ) -> bytes:
      """Injects HTTP 401 on first upload and HTTP 503 on first job poll."""
      nonlocal poll_attempts
      seen_auth.append(headers["Authorization"])
      # Trigger aged-token HTTP 401 on initial GCS upload call.
      if len(seen_auth) == 1:
        raise urllib.error.HTTPError(url, 401, "Unauthorized", Message(), None)
      # Trigger transient HTTP 503 on initial `batchPredictionJobs.get` call.
      if method == "GET" and "/batchPredictionJobs/" in url:
        poll_attempts += 1
        if poll_attempts == 1:
          raise urllib.error.HTTPError(url, 503, "Unavailable", Message(), None)
      return base_router(method, url, body, headers)

    # Execute batch with injected `sleep_fn` so 503 backoff runs in < 1 ms.
    runner = VertexBatchPredictionRunner(
        project_id="demo-eval-project-01",
        gcs_uri_prefix="gs://argus-eval-bucket/retry",
        token_supplier=lambda: next(tokens),
        http_transport=_flaky_http,
        sleep_fn=slept.append,
    )
    ddl = "CREATE TABLE t (id INT PRIMARY KEY);\n\n"
    out = runner.batch_complete((LLMCompletionRequest("r1", "sys", f"{ddl}Question: Count?"),))
    self.assertEqual(len(out), 1)
    self.assertEqual(seen_auth[0], "Bearer stale-tok")
    self.assertEqual(seen_auth[1], "Bearer fresh-tok")
    self.assertEqual(len(slept), 1)

  def test_cross_call_in_memory_cache_skips_redundant_batch_jobs(self) -> None:
    """T9: Cross-call `_cache` avoids launching redundant `BatchPredictionJob`s."""
    # Track total batch jobs created across 3 sequential `batch_complete` calls.
    transport = FakeVertexBatchTransport()
    runner = VertexBatchPredictionRunner(
        project_id="demo-eval-project-01",
        gcs_uri_prefix="gs://argus-eval-bucket/cache",
        token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
        http_transport=transport,
        sleep_fn=lambda _: None,
    )
    ddl = "CREATE TABLE t (id INT PRIMARY KEY);\n\n"
    r1 = LLMCompletionRequest("r1", "sys", f"{ddl}Question: Q1?")
    r2 = LLMCompletionRequest("r2", "sys", f"{ddl}Question: Q2?")
    r3 = LLMCompletionRequest("r3", "sys", f"{ddl}Question: Q3?")
    r4 = LLMCompletionRequest("r4", "sys", f"{ddl}Question: Q4?")

    # First call submits 1 batch job covering `(r1, r2, r3)`.
    runner.batch_complete((r1, r2, r3))
    self.assertEqual(transport.job_counter, 1)

    # Identical 3-request batch submits ZERO new GCS uploads or batch jobs!
    cached_resps = runner.batch_complete((r1, r2, r3))
    self.assertEqual(transport.job_counter, 1)
    self.assertTrue(all(r.from_cache for r in cached_resps))
    self.assertTrue(all(r.prompt_tokens == 0 and r.logical_tokens > 0 for r in cached_resps))

    # Partial overlap `(r1, r4)` uploads ONLY 1 JSONL line for `r4`!
    mixed_resps = runner.batch_complete((r1, r4))
    self.assertEqual(transport.job_counter, 2)
    self.assertTrue(mixed_resps[0].from_cache)
    self.assertFalse(mixed_resps[1].from_cache)
    uploads = [b for m, u, b, _ in transport.calls if m == "POST" and "uploadType=media" in u]
    assert uploads[-1] is not None
    self.assertEqual(len(uploads[-1].decode("utf-8").splitlines()), 1)

  def test_ssrf_and_traversal_guards_on_gcs_and_job_identifiers(self) -> None:
    """T10: Blocks SSRF, `global` batch jobs, uppercase buckets, and `..` paths."""
    # Reject traversal, non-regional `global`, and arbitrary HTTPS job URLs.
    bad_jobs = (
        "projects/../evil/locations/us-central1/batchPredictionJobs/1",
        "projects/demo-eval-project-01/locations/global/batchPredictionJobs/1",
        "https://attacker.invalid/v1/batchPredictionJobs/1",
    )
    for bad_job in bad_jobs:
      with self.assertRaises(LLMAdapterAuthError):
        validate_batch_job_name(bad_job)

    # Reject edge-hyphen regions in batch job resource identifiers (Challenge 5).
    for hyphen_job in (
        "projects/demo-eval-project-01/locations/-us-central1/batchPredictionJobs/1",
        "projects/demo-eval-project-01/locations/us-central1-/batchPredictionJobs/1",
    ):
      with self.assertRaises(LLMAdapterAuthError):
        validate_batch_job_name(hyphen_job)

    # Reject non-`gs://` schemes, uppercase buckets, `%` traversal, `//`, & queries.
    # Includes single and double percent-encoded dot segments (`%2e%2e`, `%252e%252e`).
    bad_uris = (
        "http://storage.googleapis.com/bucket/obj",
        "gs://BAD_UPPERCASE/obj",
        "gs://valid-bucket/a\\b",
        "gs://valid-bucket/a/../b",
        "gs://valid-bucket/a?query=1",
        "gs://valid-bucket/%2e",
        "gs://valid-bucket/%2e%2e",
        "gs://valid-bucket/runs/%2e%2e/secret.jsonl",
        "gs://valid-bucket/runs/%2E%2E/secret.jsonl",
        "gs://valid-bucket/runs/..%2fsecret.jsonl",
        "gs://valid-bucket/runs/%252e%252e/secret.jsonl",
        "gs://valid-bucket/runs//secret.jsonl",
    )
    for bad_uri in bad_uris:
      with self.assertRaises(LLMAdapterAuthError, msg=bad_uri):
        parse_and_validate_gcs_uri(bad_uri)

    # Reject tainted manifest `model_name`, edge-`-` `location`, and `%2e%2e` GCS URIs.
    # Ensures `BatchJobManifest.from_dict` enforces hardened `vertex_auth` guards.
    base_manifest = {
        "schema_version": "argus_batch_manifest.v1",
        "job_name": "projects/123456789012/locations/us-central1/batchPredictionJobs/7",
        "project_id": "demo-eval-project-01",
        "location": "us-central1",
        "model_name": "gemini-3.7-flash",
        "gcs_input_uri": "gs://valid-bucket/runs/r1/input.jsonl",
        "gcs_output_uri_prefix": "gs://valid-bucket/runs/r1/output/",
        "entries": [{"request_id": "q1", "digest": "a" * 64, "prompt_char_len": 12}],
    }
    # Fuzz each security-sensitive manifest field independently to confirm AuthError.
    for field_key, bad_val in (
        ("model_name", "../../evil/models/gemini-3.7-flash"),
        ("location", "-us-central1"),
        ("location", "\u212aus-central1"),
        ("gcs_input_uri", "gs://valid-bucket/runs/%2e%2e/secret.jsonl"),
        ("gcs_output_uri_prefix", "gs://valid-bucket/runs/..%2foutput/"),
    ):
      tampered = dict(base_manifest, **{field_key: bad_val})
      with self.assertRaises(LLMAdapterAuthError, msg=f"{field_key}={bad_val}"):
        BatchJobManifest.from_dict(tampered)

  def test_seven_system_mini_bird_suite_e2e_with_batch_runner_and_r4_intent(
      self,
  ) -> None:
    """T11: Runs all 7 systems (`B0`..`Argus-SQL`) over `VertexBatchPredictionRunner`."""
    # Materialize all 3 read-only SQLite Mini-BIRD databases inside `tmp_dir`.
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = pathlib.Path(tmp_dir)
      schemas, paths = materialize_mini_bird_databases(root)
      sandbox = SQLiteReadOnlySandbox(SandboxPolicy(allowed_db_root=root))

      # Wire `VertexBatchPredictionRunner` (`gemini-3.7-flash`) over hermetic router.
      transport = FakeVertexBatchTransport(shard_count=2)
      runner = VertexBatchPredictionRunner(
          project_id="demo-eval-project-01",
          gcs_uri_prefix="gs://argus-eval-bucket/e2e",
          location="global",
          token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
          http_transport=transport,
          sleep_fn=lambda _: None,
      )

      # Build R4 `gemini-2.5-flash` Stage 1 intent client with mocked transport.
      intent_llm = build_intent_llm_client(
          provider="vertex",
          gcp_project="demo-eval-project-01",
          fallback_to_none=False,
      )
      assert isinstance(intent_llm, VertexGeminiBatchClient)
      self.assertEqual(intent_llm.model_name, "gemini-2.5-flash")
      intent_llm._auth = runner._auth

      intent_budgets: list[int] = []

      def _mock_intent_post(payload_bytes: bytes, bearer: str) -> dict[str, object]:
        """Emits canonical Stage 1 JSON frames while recording `thinkingBudget`."""
        del bearer
        body = json.loads(payload_bytes.decode("utf-8"))
        gen_cfg = body["generationConfig"]
        self.assertNotIn("maxOutputTokens", gen_cfg)
        intent_budgets.append(int(gen_cfg["thinkingConfig"]["thinkingBudget"]))
        raw_prompt = str(body["contents"][0]["parts"][0]["text"])
        q_line = raw_prompt.splitlines()[0].removeprefix("Question:").strip()
        # Use `IntentFrame.entities` and `IntentFrame.complexity` fields verbatim.
        frame = canonicalize_intent(q_line, "")
        intent_json = json.dumps({
            "canonical_question": frame.canonical_question,
            "entities": list(frame.entities),
            "expected_projection_arity": frame.expected_projection_arity,
            "requires_floating_division": frame.requires_floating_division,
            "requires_extrema_order": frame.requires_extrema_order,
            "complexity": frame.complexity.value,
        })
        return {
            "candidates": [{"content": {"parts": [{"text": intent_json}]}}],
            "usageMetadata": {"promptTokenCount": 32, "candidatesTokenCount": 20},
        }

      # Slice 1 Clean + 1 Voice task across each of the 3 mini-BIRD databases.
      clean_by_db = {t.db_id: t for t in reversed(build_mini_bird_clean_tasks())}
      voice_by_db = {t.db_id: t for t in reversed(build_voice_bird_tasks())}
      e2e_tasks = tuple(clean_by_db.values()) + tuple(voice_by_db.values())
      systems = build_main_systems(runner, sandbox, intent_llm=intent_llm)
      self.assertEqual(len(systems), 7)

      # Run all 7 systems (`B0`, `B0+`, `B1`, `B2`, `B3`, `B4`, `Argus-SQL`) offline.
      with mock.patch.object(intent_llm, "_post_generate_once", side_effect=_mock_intent_post):
        report = run_benchmark_suite(
            "7-System Batch E2E",
            e2e_tasks,
            systems,
            schemas,
            paths,
            sandbox,
        )

      # Verify all 7 systems completed with positive EX and zero output clamping.
      self.assertEqual(len(report.summaries), 7)
      for summary in report.summaries:
        self.assertGreater(summary.ex_overall, 0.0, msg=summary.system_name)

      # `B0_RawDDL` & `B0+_EnrichedMDDL` emit `thinkingBudget=0`; reasoning >= 2048.
      recorded_budgets = {tb for tb, _ in transport.recorded_line_configs}
      self.assertIn(0, recorded_budgets)
      self.assertIn(2048, recorded_budgets)
      self.assertFalse(any(clamped for _, clamped in transport.recorded_line_configs))
      self.assertTrue(bool(intent_budgets) and all(tb == 0 for tb in intent_budgets))

  def test_transport_http_exception_retry_utf8_guard_and_neg_zero_digest(
      self,
  ) -> None:
    """T12: Retries `HTTPException`/`OSError`, wraps bad UTF-8, and normalizes `-0.0`."""
    # Verify `-0.0` and `+0.0` produce identical canonical digests and JSONL lines.
    req_pos = LLMCompletionRequest("p0", "sys", "SELECT 1", temperature=0.0)
    req_neg = LLMCompletionRequest("n0", "sys", "SELECT 1", temperature=-0.0)
    dict_pos = _build_batch_request_dict(req_pos, 2048)
    dict_neg = _build_batch_request_dict(req_neg, 2048)
    self.assertEqual(_batch_request_digest(dict_pos), _batch_request_digest(dict_neg))
    # Direct `-0.0` inside raw echoed dict must also normalize to `0.0`.
    raw_neg_echo = {
        "contents": dict_pos["contents"],
        "generationConfig": {"temperature": -0.0},
    }
    raw_pos_echo = {
        "contents": dict_pos["contents"],
        "generationConfig": {"temperature": 0.0},
    }
    self.assertEqual(
        _batch_request_digest(raw_neg_echo), _batch_request_digest(raw_pos_echo)
    )

    # Verify transient `IncompleteRead` and `ConnectionResetError` retry and succeed.
    base_router = FakeVertexBatchTransport(strip_echoed_labels=True)
    slept: list[float] = []
    attempt_counter = 0

    def _flaky_socket_http(
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
    ) -> bytes:
      """Simulates `IncompleteRead` then `ConnectionResetError` before succeeding."""
      nonlocal attempt_counter
      attempt_counter += 1
      # Fail first two upload attempts with HTTPException and OSError.
      if attempt_counter == 1:
        raise http.client.IncompleteRead(b"partial", 64)
      if attempt_counter == 2:
        raise ConnectionResetError("peer reset")
      return base_router(method, url, body, headers)

    # Run end-to-end `batch_complete` with `-0.0` temperature and stripped labels.
    runner = VertexBatchPredictionRunner(
        project_id="demo-eval-project-01",
        gcs_uri_prefix="gs://argus-eval-bucket/t12",
        max_retries=3,
        token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
        http_transport=_flaky_socket_http,
        sleep_fn=slept.append,
    )
    ddl = "CREATE TABLE t (id INT PRIMARY KEY);\n\n"
    res = runner.batch_complete(
        (LLMCompletionRequest("r_neg", "sys", f"{ddl}Question: Count?", temperature=-0.0),)
    )
    self.assertEqual(len(res), 1)
    self.assertEqual(len(slept), 2)

    # Persistent `BadStatusLine` or invalid UTF-8 bytes wrap in `LLMAdapterTransportError`.
    def _bad_utf8_http(
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
    ) -> bytes:
      """Returns malformed UTF-8 bytes on `batchPredictionJobs.create`."""
      # Inject undecodable byte sequence `0xff 0xfe` after GCS upload succeeds.
      if method == "POST" and url.endswith("/batchPredictionJobs"):
        return b"\xff\xfe"
      return base_router(method, url, body, headers)

    bad_utf8_runner = VertexBatchPredictionRunner(
        project_id="demo-eval-project-01",
        gcs_uri_prefix="gs://argus-eval-bucket/t12b",
        max_retries=1,
        token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
        http_transport=_bad_utf8_http,
        sleep_fn=lambda _: None,
    )
    with self.assertRaises(LLMAdapterTransportError):
      bad_utf8_runner.submit_job((req_pos,))

    # Corrupted UTF-8 in `BatchJobManifest.load_json` also raises TransportError.
    with tempfile.TemporaryDirectory() as tmp_dir:
      bad_manifest_path = pathlib.Path(tmp_dir) / "corrupt_manifest.json"
      bad_manifest_path.write_bytes(b"\xff\xfe\x00")
      with self.assertRaises(LLMAdapterTransportError):
        BatchJobManifest.load_json(bad_manifest_path)

  def test_null_and_malformed_json_fields_in_batch_digest_poll_and_collect(
      self,
  ) -> None:
    """T13: Coalesces explicit JSON `null` fields and wraps `bool`/`NaN`/bad types."""
    # Coalesce explicit `None` on `generationConfig`, `temperature`, `thinkingBudget`,
    # `systemInstruction`, and `cachedContent` without raising `TypeError`.
    base_contents = [{"role": "user", "parts": [{"text": "SELECT 1"}]}]
    d_default = _batch_request_digest(
        {"contents": base_contents, "generationConfig": {"temperature": 0.0}}
    )
    d_nulls = _batch_request_digest({
        "contents": base_contents,
        "generationConfig": {
            "temperature": None,
            "thinkingConfig": {"thinkingBudget": None},
        },
        "systemInstruction": None,
        "cachedContent": None,
    })
    self.assertEqual(d_default, d_nulls)
    # Reject boolean, NaN/Inf, negative, and non-numeric temperature/budget fields.
    for bad_gen in (
        {"temperature": True},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"temperature": "not-a-float"},
        {"thinkingConfig": {"thinkingBudget": False}},
        {"thinkingConfig": {"thinkingBudget": -2}},
        {"thinkingConfig": {"thinkingBudget": "bad"}},
    ):
      with self.assertRaises(LLMAdapterTransportError):
        _batch_request_digest({"contents": base_contents, "generationConfig": bad_gen})

    # Ensure `BatchJobManifest.from_dict` never coerces `None` into `"None"`.
    valid_manifest_dict: dict[str, object] = {
        "schema_version": "argus_batch_manifest.v1",
        "job_name": "projects/123456789012/locations/us-central1/batchPredictionJobs/7",
        "project_id": "demo-eval-project-01",
        "location": "us-central1",
        "model_name": "gemini-3.7-flash",
        "gcs_input_uri": "gs://valid-bucket/runs/r1/input.jsonl",
        "gcs_output_uri_prefix": "gs://valid-bucket/runs/r1/output/",
        "entries": [{"request_id": "q1", "digest": "a" * 64, "prompt_char_len": 12}],
    }
    for null_auth_key in ("model_name", "job_name"):
      with self.assertRaises(LLMAdapterAuthError):
        BatchJobManifest.from_dict(dict(valid_manifest_dict, **{null_auth_key: None}))
    with self.assertRaises(LLMAdapterTransportError):
      BatchJobManifest.from_dict(
          dict(
              valid_manifest_dict,
              entries=[{"request_id": None, "digest": "a" * 64, "prompt_char_len": 12}],
          )
      )

    # Verify `poll_job` coalesces `null` `completionStats` & rejects `bool`/bad counts,
    # and `collect_job_responses` raises `LLMAdapterTransportError` on `null` echo temp.
    poll_payloads = iter([
        b'{"name":"projects/123456/locations/us-central1/batchPredictionJobs/j1","state":"JOB_STATE_PARTIALLY_SUCCEEDED","outputInfo":{"gcsOutputDirectory":"gs://argus-eval-bucket/o"},"completionStats":{"successfulCount":null,"failedCount":"2","incompleteCount":null}}',
        b'{"name":"projects/123456/locations/us-central1/batchPredictionJobs/j1","state":"JOB_STATE_RUNNING","completionStats":{"successfulCount":true}}',
    ])

    def _tx(m: str, u: str, b: bytes | None, h: dict[str, str]) -> bytes:
      """Serves `null` stats, `null` `nextPageToken`, and `null`-temperature shard row."""
      del b, h
      if m == "POST" and "storage.googleapis.com" in u:
        return b"{}"
      if m == "POST":
        return b'{"name":"projects/123456/locations/us-central1/batchPredictionJobs/j1","state":"JOB_STATE_SUCCEEDED"}'
      if "/batchPredictionJobs/" in u:
        return next(poll_payloads)
      if "prefix=" in u:
        return b'{"items":[{"name":"o/p.jsonl"}],"nextPageToken":null}'
      return b'{"request":{"contents":[],"generationConfig":{"temperature":null},"labels":null},"status":"INVALID_ARGUMENT"}'

    # Confirm both failed shard collection and `bool` poll stats raise TransportError.
    runner = VertexBatchPredictionRunner(
        project_id="demo-eval-project-01",
        gcs_uri_prefix="gs://argus-eval-bucket/p",
        max_retries=1,
        token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
        http_transport=_tx,
        sleep_fn=lambda _: None,
    )
    with self.assertRaises(LLMAdapterTransportError):
      runner.batch_complete((LLMCompletionRequest("r1", "s", "SELECT 1"),))
    with self.assertRaises(LLMAdapterTransportError):
      runner.poll_job("j1")


if __name__ == "__main__":
  unittest.main()
