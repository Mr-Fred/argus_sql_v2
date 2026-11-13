"""Unit tests for HTTPS batch security, SQL extraction, and DDL synthesis."""

import http.client
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from argus_sql.adapters.client_factory import build_live_llm_client
from argus_sql.adapters.client_factory import load_dotenv_defaults
from argus_sql.adapters.deterministic_synthesizer import SchemaGuidedDiagnosticSynthesizer
from argus_sql.adapters.exceptions import LLMAdapterAuthError
from argus_sql.adapters.exceptions import LLMAdapterTransportError
from argus_sql.adapters.llm_batch_client import OpenAICompatibleBatchClient
from argus_sql.adapters.llm_batch_client import extract_sql_from_completion
from argus_sql.adapters.llm_batch_client import validate_endpoint_security
from argus_sql.adapters.types import LLMCompletionRequest
from argus_sql.adapters.vertex_gemini_client import VertexGeminiBatchClient
from argus_sql.adapters.vertex_gemini_client import build_vertex_generate_url
from argus_sql.adapters.vertex_gemini_client import extract_gemini_completion_triple


class AdaptersModuleTest(unittest.TestCase):
  """Tests endpoint security guards, Vertex Gemini batching, and synthesis."""

  def test_endpoint_security_blocks_remote_http_and_wildcard_hosts(self) -> None:
    """Remote HTTP and 0.0.0.0 must fail closed; HTTPS and 127.0.0.1 pass."""
    # Valid HTTPS and explicit localhost loopback URLs succeed.
    self.assertTrue(
        validate_endpoint_security("https://api.openai.com/v1/chat/completions")
    )
    self.assertTrue(
        validate_endpoint_security("http://127.0.0.1:8000/v1/chat/completions")
    )

    # Unencrypted remote HTTP and 0.0.0.0 wildcard hosts must raise AuthError.
    with self.assertRaises(LLMAdapterAuthError):
      validate_endpoint_security("http://example.com/v1/chat/completions")
    with self.assertRaises(LLMAdapterAuthError):
      validate_endpoint_security("http://0.0.0.0:8000/v1/chat/completions")

    # Remote HTTPS client without env API key must fail closed immediately.
    with self.assertRaises(LLMAdapterAuthError):
      OpenAICompatibleBatchClient(
          model_name="gpt-4o-mini",
          api_key_env="NONEXISTENT_ARGUS_TEST_KEY_ENV",
      )

  def test_extracts_fenced_sql_and_runs_deterministic_batch(self) -> None:
    """Markdown code fences strip cleanly and batch synthesizer aligns 1-to-1."""
    fenced = "Reasoning: join tables.\n```sql\nSELECT COUNT(*) FROM loans;\n```"
    self.assertEqual(
        extract_sql_from_completion(fenced), "SELECT COUNT(*) FROM loans"
    )

    # Synthesize two strategies in a single batch call over a sample DDL prompt.
    ddl_prompt = (
        "CREATE TABLE schools (\n"
        "  cds_code TEXT PRIMARY KEY,\n"
        "  county TEXT -- matched_values: ['Alameda']\n"
        ");\n\n"
        "Question: How many schools are in Alameda?\n"
    )
    synth = SchemaGuidedDiagnosticSynthesizer()
    reqs = (
        LLMCompletionRequest("r1", "sys", ddl_prompt, strategy_tag="zero_shot"),
        LLMCompletionRequest(
            "r2", "sys", ddl_prompt, strategy_tag="formula_defensive"
        ),
    )
    # Verify 1-to-1 response alignment and grounded literal WHERE binding.
    resps = synth.batch_complete(reqs)
    self.assertEqual(len(resps), 2)
    self.assertIn("schools.county = 'Alameda'", resps[0].sql_text)

  def test_vertex_gemini_url_validation_thought_filtering_and_dedup(self) -> None:
    """Vertex URLs block traversal, skip thought parts, and deduplicate batches."""
    # Global Gemini Enterprise routing must hit `aiplatform.googleapis.com`.
    global_url = build_vertex_generate_url(
        "demo-eval-project-01", "global", "google/gemini-3.7-flash"
    )
    self.assertEqual(
        global_url,
        "https://aiplatform.googleapis.com/v1beta1/projects/"
        "demo-eval-project-01/locations/global/publishers/google/"
        "models/gemini-3.7-flash:generateContent",
    )
    # Regional routing must prepend `<region>-` to `aiplatform.googleapis.com`.
    regional_url = build_vertex_generate_url(
        "demo-eval-project-01", "us-central1", "gemini-3.7-pro"
    )
    self.assertIn("https://us-central1-aiplatform.googleapis.com/", regional_url)

    # Reject path traversal or malformed project/location/model identifiers.
    with self.assertRaises(LLMAdapterAuthError):
      build_vertex_generate_url("../bad-project", "global", "gemini-3.7-flash")
    with self.assertRaises(LLMAdapterAuthError):
      build_vertex_generate_url("demo-eval-project-01", "bad/loc", "gemini-3.7")
    with self.assertRaises(LLMAdapterAuthError):
      build_vertex_generate_url("demo-eval-project-01", "global", "bad?model=1")

    # Verify thinking `thought=True` parts are ignored in favor of SQL output.
    sample_body: dict[str, object] = {
        "candidates": [{
            "content": {
                "parts": [
                    {"thought": True, "text": "SELECT wrong_table FROM draft;"},
                    {"text": "```sql\nSELECT COUNT(*) FROM schools;\n```"},
                ]
            }
        }],
        "usageMetadata": {
            "promptTokenCount": 120,
            "candidatesTokenCount": 14,
            "thoughtsTokenCount": 46,
        },
    }
    # Confirm extracted SQL ignores thought text and sums thought + output tokens.
    raw_txt, p_tok, c_tok = extract_gemini_completion_triple(sample_body, 400)
    self.assertEqual(extract_sql_from_completion(raw_txt), "SELECT COUNT(*) FROM schools")
    self.assertEqual(p_tok, 120)
    self.assertEqual(c_tok, 60)

    # Truncated thought-only responses must never leak internal draft SQL.
    thought_only_body: dict[str, object] = {
        "candidates": [{
            "content": {
                "parts": [{"thought": True, "text": "SELECT wrong_table FROM draft;"}]
            },
            "finishReason": "MAX_TOKENS",
        }],
        "usageMetadata": {"promptTokenCount": 80, "thoughtsTokenCount": 2048},
    }
    trunc_txt, _, trunc_c_tok = extract_gemini_completion_triple(thought_only_body, 320)
    self.assertEqual(trunc_txt, "")
    self.assertEqual(trunc_c_tok, 2048)

    # Verify batch deduplication executes only 1 upstream call for 2 identical reqs.
    token_calls = 0

    def _fake_token() -> tuple[str, float]:
      """Returns a synthetic in-memory OAuth token for hermetic unit tests."""
      nonlocal token_calls
      token_calls += 1
      return ("synthetic-test-bearer", 3600.0)

    # Construct client with injected hermetic token supplier (zero network I/O).
    client = VertexGeminiBatchClient(
        project_id="demo-eval-project-01",
        model_name="gemini-3.7-flash",
        location="global",
        token_supplier=_fake_token,
    )
    req_a = LLMCompletionRequest("q1", "sys", "SELECT prompt")
    req_b = LLMCompletionRequest("q2", "sys", "SELECT prompt")
    with mock.patch.object(
        client, "_post_generate_once", return_value=sample_body
    ) as post_mock:
      out = client.batch_complete((req_a, req_b))
    self.assertEqual(post_mock.call_count, 1)
    self.assertEqual(token_calls, 1)
    self.assertEqual(len(out), 2)
    self.assertEqual(out[0].sql_text, "SELECT COUNT(*) FROM schools")
    # Subsequent batch call hits the SHA-256 cache with 0 billed tokens while
    # preserving `logical_tokens == 180` for standalone baseline comparisons.
    with mock.patch.object(
        client, "_post_generate_once", return_value=sample_body
    ) as post_mock_cached:
      cached_out = client.batch_complete((req_a,))
    self.assertEqual(post_mock_cached.call_count, 0)
    self.assertTrue(cached_out[0].from_cache)
    self.assertEqual(cached_out[0].prompt_tokens, 0)
    self.assertEqual(cached_out[0].logical_tokens, 180)

    # Deterministic synthesizer `llm_judge` mode prefers REAL ratio over `[(0,)]`.
    judge_prompt = (
        "Question: What is the overall free meal rate?\n"
        "External Evidence: free_meals / enrollment\n\n"
        "Option A (support: 2/3 candidates):\n"
        "SQL: SELECT SUM(free_meals) / SUM(enrollment) FROM frpm\n"
        "Columns: ['rate'] | Row Count: 1\n"
        "Sample Rows: [(0,)]\n\n"
        "Option B (support: 1/3 candidates):\n"
        "SQL: SELECT CAST(SUM(free_meals) AS REAL) / NULLIF(SUM(enrollment), 0) FROM frpm\n"
        "Columns: ['rate'] | Row Count: 1\n"
        "Sample Rows: [(0.4,)]\n"
    )
    judge_resp = SchemaGuidedDiagnosticSynthesizer().batch_complete(
        (LLMCompletionRequest("j1", "judge", judge_prompt, strategy_tag="llm_judge"),)
    )[0]
    self.assertEqual(judge_resp.raw_text, "Selected: Option B")

  def test_dotenv_defaults_and_provider_auto_routing(self) -> None:
    """`.env` configures Vertex, OpenAI key, and loopback routing safely."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      # Materialize a temporary non-secret `.env` file matching Vertex workflow.
      env_file = Path(tmp_dir) / ".env"
      env_file.write_text(
          "# Sample config\n"
          "GOOGLE_GENAI_USE_ENTERPRISE=TRUE\n"
          "GOOGLE_CLOUD_PROJECT=demo-eval-project-01\n"
          "GOOGLE_CLOUD_LOCATION=global\n",
          encoding="utf-8",
      )
      # Confirm `provider='auto'` selects `VertexGeminiBatchClient` automatically
      # with `gemini-3.7-flash` on the `global` endpoint by default.
      with mock.patch.dict("os.environ", {}, clear=True):
        cfg = load_dotenv_defaults(env_file)
        self.assertEqual(cfg["GOOGLE_CLOUD_PROJECT"], "demo-eval-project-01")
        routed = build_live_llm_client(provider="auto", dotenv_path=env_file)
        self.assertIsInstance(routed, VertexGeminiBatchClient)
        assert isinstance(routed, VertexGeminiBatchClient)
        self.assertEqual(routed.provider_name, "vertex")
        self.assertEqual(routed.location, "global")
        self.assertEqual(routed.model_name, "gemini-3.7-flash")

      # Unedited template placeholder without API key must fail closed on Vertex.
      placeholder_env = Path(tmp_dir) / "placeholder.env"
      placeholder_env.write_text(
          "GOOGLE_GENAI_USE_VERTEXAI=TRUE\n"
          "GOOGLE_CLOUD_PROJECT=your-personal-gcp-project-id\n",
          encoding="utf-8",
      )
      with mock.patch.dict("os.environ", {}, clear=True):
        with self.assertRaises(LLMAdapterAuthError):
          build_live_llm_client(provider="auto", dotenv_path=placeholder_env)

      # `.env` with UTF-8 BOM (`\ufeff`, common on Windows PowerShell 5.1/Notepad)
      # and `ARGUS_LLM_API_KEY` must parse cleanly without mutating `os.environ`.
      openai_env = Path(tmp_dir) / "openai.env"
      synth_key = "test-only-" + (" token99".strip() * 3)
      openai_env.write_bytes(
          (
              f"ARGUS_LLM_API_KEY={synth_key}\n"
              "GOOGLE_GENAI_USE_VERTEXAI=TRUE\n"
              "GOOGLE_CLOUD_PROJECT=your-personal-gcp-project-id\n"
              "ARGUS_LLM_MODEL=gpt-4o\n"
              "ARGUS_LLM_ENDPOINT_URL=http://127.0.0.1:8000/v1/chat/completions\n"
          ).encode("utf-8-sig")
      )
      with mock.patch.dict("os.environ", {}, clear=True):
        oa_client = build_live_llm_client(provider="auto", dotenv_path=openai_env)
        self.assertIsInstance(oa_client, OpenAICompatibleBatchClient)
        self.assertEqual(getattr(oa_client, "provider_name", ""), "openai")
        self.assertEqual(oa_client.model_name, "gpt-4o")
        self.assertEqual(
            oa_client.endpoint_url, "http://127.0.0.1:8000/v1/chat/completions"
        )

  def test_vertex_401_refresh_and_service_account_gcloud_fallback(self) -> None:
    """Aged tokens refresh once on HTTP 401 and SA ADC falls back to gcloud."""
    tokens = iter([("expired-tok", 3600.0), ("fresh-tok", 3600.0)])
    client = VertexGeminiBatchClient(
        project_id="demo-eval-project-01",
        model_name="gemini-3.7-flash",
        token_supplier=lambda: next(tokens),
    )
    ok_body: dict[str, object] = {
        "candidates": [{"content": {"parts": [{"text": "SELECT 1"}]}}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3},
    }
    seen_payloads: list[dict[str, object]] = []
    seen_tokens: list[str] = []

    def _mock_post(payload_bytes: bytes, bearer: str) -> dict[str, object]:
      """Simulates an initial HTTP 401 followed by a valid 200 completion."""
      import json as _json
      import urllib.error as _urlerr
      from email.message import Message

      seen_payloads.append(_json.loads(payload_bytes.decode("utf-8")))
      seen_tokens.append(bearer)
      if len(seen_tokens) == 1:
        raise _urlerr.HTTPError(
            client.endpoint_url, 401, "Unauthorized", Message(), None
        )
      return ok_body

    req = LLMCompletionRequest("q401", "Be concise", "SELECT 1")
    with mock.patch.object(client, "_post_generate_once", side_effect=_mock_post):
      res = client.batch_complete((req,))
    self.assertEqual(res[0].sql_text, "SELECT 1")
    self.assertEqual(seen_tokens, ["expired-tok", "fresh-tok"])
    # `systemInstruction` must omit `"role": "system"` for strict Vertex routers,
    # and `generationConfig` must omit `maxOutputTokens` clamping while setting
    # `thinkingBudget: 0` on default `strategy_tag='zero_shot'` (R3).
    sys_inst = seen_payloads[0]["systemInstruction"]
    assert isinstance(sys_inst, dict)
    self.assertNotIn("role", sys_inst)
    gen_cfg = seen_payloads[0]["generationConfig"]
    assert isinstance(gen_cfg, dict)
    self.assertNotIn("maxOutputTokens", gen_cfg)
    self.assertEqual(gen_cfg["thinkingConfig"], {"thinkingBudget": 0})

  def test_windows_appdata_adc_discovery_and_gcloud_cmd_fallback(self) -> None:
    """Windows `%APPDATA%\\gcloud` ADC and `gcloud.cmd` resolve cleanly."""
    import ssl as _ssl
    import subprocess as _subproc
    from argus_sql.adapters.vertex_auth import VertexTokenResolver

    with tempfile.TemporaryDirectory() as tmp_dir:
      appdata_root = Path(tmp_dir) / "Roaming"
      gcloud_dir = appdata_root / "gcloud"
      gcloud_dir.mkdir(parents=True)
      adc_json = gcloud_dir / "application_default_credentials.json"
      adc_json.write_text('{"type": "service_account"}', encoding="utf-8")

      resolver = VertexTokenResolver(ssl_ctx=_ssl.create_default_context())
      # Verify `%APPDATA%\gcloud\application_default_credentials.json` is found,
      # and non-authorized_user types fall back to `gcloud.cmd` on Windows.
      fake_proc = _subproc.CompletedProcess(
          args=["gcloud.cmd"], returncode=0, stdout="win-cli-tok\n", stderr=""
      )
      with mock.patch.dict(
          "os.environ", {"APPDATA": str(appdata_root)}, clear=True
      ):
        with mock.patch.object(Path, "home", return_value=Path(tmp_dir) / "NoHome"):
          self.assertEqual(resolver._resolve_adc_path(), adc_json)
          with mock.patch(
              "shutil.which",
              side_effect=lambda name: "gcloud.cmd" if name == "gcloud.cmd" else None,
          ):
            with mock.patch("subprocess.run", return_value=fake_proc):
              self.assertEqual(resolver.get_bearer_token(), "win-cli-tok")

    # Verify `parse_prompt_ddl` handles quoted keyword tables, spaced columns,
    # and inline `--` comments containing `);` without truncating columns.
    from argus_sql.adapters.ddl_prompt_parser import parse_prompt_ddl

    tricky_ddl = (
        'CREATE TABLE "order" (\n'
        '  order_id INTEGER PRIMARY KEY, -- (pk); note\n'
        '  "Academic Year" TEXT -- matched_values: [\'2014-2015\']\n'
        ');'
    )
    parsed = parse_prompt_ddl(tricky_ddl)
    self.assertIn("order", parsed)
    self.assertEqual(
        parsed["order"].columns,
        (("order_id", "INTEGER", ()), ("Academic Year", "TEXT", ("2014-2015",))),
    )

  def test_build_intent_llm_client_offline_fallback_and_flash_lite_slug(
      self,
  ) -> None:
    """`build_intent_llm_client` falls back to None offline and routes 2.5-flash."""
    from argus_sql.adapters import build_intent_llm_client

    # Without credentials in env or `.env`, returns None cleanly when allowed.
    with tempfile.TemporaryDirectory() as tmp_dir:
      empty_env = Path(tmp_dir) / ".env"
      empty_env.write_text("", encoding="utf-8")
      with mock.patch.dict("os.environ", {}, clear=True):
        self.assertIsNone(
            build_intent_llm_client(dotenv_path=empty_env, fallback_to_none=True)
        )
        # With Vertex project configured, Stage 1 routes to gemini-2.5-flash (R4).
        with mock.patch.dict(
            "os.environ", {"GOOGLE_CLOUD_PROJECT": "demo-eval-project-01"}, clear=True
        ):
          intent_client = build_intent_llm_client(
              dotenv_path=empty_env, fallback_to_none=False
          )
          self.assertIsInstance(intent_client, VertexGeminiBatchClient)
          assert isinstance(intent_client, VertexGeminiBatchClient)
          self.assertEqual(intent_client.model_name, "gemini-2.5-flash")

  def test_vertex_auth_cached_contents_batch_jobs_and_gcs_urls(self) -> None:
    """Validators accept numeric project IDs, enforce `%2F`, and block SSRF."""
    from argus_sql.adapters.vertex_auth import build_gcs_download_url
    from argus_sql.adapters.vertex_auth import build_gcs_list_url
    from argus_sql.adapters.vertex_auth import build_gcs_upload_url
    from argus_sql.adapters.vertex_auth import build_vertex_batch_job_item_url
    from argus_sql.adapters.vertex_auth import build_vertex_batch_jobs_url
    from argus_sql.adapters.vertex_auth import build_vertex_cached_content_item_url
    from argus_sql.adapters.vertex_auth import build_vertex_cached_contents_url
    from argus_sql.adapters.vertex_auth import normalize_gemini_model_name
    from argus_sql.adapters.vertex_auth import parse_gcs_uri

    # Verify global and regional `cachedContents` URLs including numeric project ID.
    self.assertEqual(
        build_vertex_cached_contents_url("demo-eval-project-01", "global"),
        "https://aiplatform.googleapis.com/v1beta1/projects/"
        "demo-eval-project-01/locations/global/cachedContents",
    )
    item_url = build_vertex_cached_content_item_url(
        "global",
        "projects/123456789012/locations/global/cachedContents/cache-001",
    )
    self.assertEqual(
        item_url,
        "https://aiplatform.googleapis.com/v1beta1/projects/"
        "123456789012/locations/global/cachedContents/cache-001",
    )
    # Cross-region mismatch on `cachedContents` item URL must fail closed.
    with self.assertRaises(LLMAdapterAuthError):
      build_vertex_cached_content_item_url(
          "us-central1",
          "projects/123456789012/locations/global/cachedContents/cache-001",
      )

    # Verify `batchPredictionJobs` maps `'global'` -> `'us-central1'` regionally.
    self.assertEqual(
        build_vertex_batch_jobs_url("demo-eval-project-01", "global"),
        "https://us-central1-aiplatform.googleapis.com/v1/projects/"
        "demo-eval-project-01/locations/us-central1/batchPredictionJobs",
    )
    job_url = build_vertex_batch_job_item_url(
        "global",
        "projects/123456789012/locations/us-central1/batchPredictionJobs/job-9",
    )
    self.assertIn("us-central1-aiplatform.googleapis.com/v1/projects/", job_url)

    # Verify `%2F` slash encoding on GCS download and ISO timestamp listing.
    bkt, obj = parse_gcs_uri("gs://argus-eval-bucket/out/prediction-2026:05:00Z/p.jsonl")
    self.assertEqual(bkt, "argus-eval-bucket")
    dl_url = build_gcs_download_url(bkt, obj)
    self.assertIn("out%2Fprediction-2026%3A05%3A00Z%2Fp.jsonl?alt=media", dl_url)
    up_url = build_gcs_upload_url(bkt, "runs/r1/input.jsonl")
    self.assertIn("uploadType=media&name=runs%2Fr1%2Finput.jsonl", up_url)
    ls_url = build_gcs_list_url(bkt, "runs/r1/", page_token="tok-2")
    self.assertIn("prefix=runs%2Fr1%2F&pageToken=tok-2", ls_url)

    # Block dot-segment traversal on model slugs and GCS object paths.
    with self.assertRaises(LLMAdapterAuthError):
      normalize_gemini_model_name("..")
    with self.assertRaises(LLMAdapterAuthError):
      parse_gcs_uri("gs://argus-eval-bucket/runs/../secret.jsonl")

  def test_r3_thinking_budget_unclamped_output_and_cached_content_payload(
      self,
  ) -> None:
    """R3 sets 0 on zero-shot/intent, >=2048 on reasoning, and omits clamping."""
    from argus_sql.adapters import build_vertex_generate_payload
    from argus_sql.adapters import parse_thinking_budget
    from argus_sql.adapters import resolve_thinking_budget

    # Validate env/constructor `parse_thinking_budget` edge cases and rejects.
    self.assertEqual(parse_thinking_budget(None), 2048)
    self.assertEqual(parse_thinking_budget(""), 2048)
    self.assertEqual(parse_thinking_budget("0"), 0)
    self.assertEqual(parse_thinking_budget("4096"), 4096)
    for bad_val in (-1, "-5", "abc", True, False):
      with self.assertRaises(LLMAdapterAuthError):
        parse_thinking_budget(bad_val)

    # Non-reasoning tags (`B0_RawDDL`/`B0+_EnrichedMDDL` & Stage 1) get 0.
    self.assertEqual(resolve_thinking_budget("zero_shot", 2048), 0)
    self.assertEqual(resolve_thinking_budget("intent_flash_lite", 4096), 0)
    # Reasoning tags enforce the 2048 floor even when configured lower (e.g. 1024).
    for tag in (
        "decomposed_cot",
        "formula_defensive",
        "skeleton_minimal",
        "repair",
        "llm_judge",
    ):
      self.assertEqual(resolve_thinking_budget(tag, 1024), 2048)
      self.assertEqual(resolve_thinking_budget(tag, 4096), 4096)

    # Verify `ARGUS_THINKING_BUDGET` env var wiring, unclamped `generationConfig`,
    # and distinct `_cache_key` digests across budgets and `cached_content` IDs.
    with mock.patch.dict("os.environ", {"ARGUS_THINKING_BUDGET": "3072"}):
      vertex_llm = VertexGeminiBatchClient(
          project_id="demo-eval-project-01",
          min_output_tokens=8192,
          token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
      )
    self.assertEqual(vertex_llm.thinking_budget, 3072)

    # Confirm `maxOutputTokens` is absent even when legacy `min_output_tokens` was passed.
    req_zero = LLMCompletionRequest("r0", "sys", "Q?", strategy_tag="zero_shot")
    req_cot = LLMCompletionRequest("r1", "sys", "Q?", strategy_tag="decomposed_cot")
    pay_zero = build_vertex_generate_payload(req_zero, vertex_llm.thinking_budget)
    pay_cot = build_vertex_generate_payload(req_cot, vertex_llm.thinking_budget)
    # Zero-shot deterministic baselines (`B0`, `B0+`) emit `thinkingBudget: 0`.
    self.assertEqual(
        pay_zero["generationConfig"],
        {"temperature": 0.0, "thinkingConfig": {"thinkingBudget": 0}},
    )
    # Reasoning strategies preserve configured `ARGUS_THINKING_BUDGET >= 2048`.
    self.assertEqual(
        pay_cot["generationConfig"],
        {"temperature": 0.0, "thinkingConfig": {"thinkingBudget": 3072}},
    )
    # Local SHA-256 dedup key must separate `thinkingBudget=0` from `3072`.
    self.assertNotEqual(vertex_llm._cache_key(req_zero), vertex_llm._cache_key(req_cot))

    # When `cached_content` is attached, `systemInstruction` must be omitted.
    cache_a = "projects/123456789012/locations/global/cachedContents/cache-a"
    cache_b = "projects/123456789012/locations/global/cachedContents/cache-b"
    req_ca = LLMCompletionRequest("ca", "sys", "Q?", cached_content=cache_a)
    req_cb = LLMCompletionRequest("cb", "sys", "Q?", cached_content=cache_b)
    pay_ca = build_vertex_generate_payload(req_ca, vertex_llm.thinking_budget)
    self.assertEqual(pay_ca["cachedContent"], cache_a)
    self.assertNotIn("systemInstruction", pay_ca)
    self.assertNotEqual(vertex_llm._cache_key(req_ca), vertex_llm._cache_key(req_cb))
    # Reject non-int/NaN/bool/negative runtime budgets on both config & override,
    # and length-prefix `eff_sys` so `|` inside custom system prompts never collides.
    for bad_rt in (float("nan"), 2048.5, "2048", True, False, -1):
      with self.assertRaises(LLMAdapterAuthError):
        resolve_thinking_budget("decomposed_cot", bad_rt)  # type: ignore[arg-type]
      with self.assertRaises(LLMAdapterAuthError):
        resolve_thinking_budget("decomposed_cot", 2048, bad_rt)  # type: ignore[arg-type]
    # Verify `eff_sys` length prefixing prevents `|` boundary shifts in `_cache_key`.
    req_pipe_a = LLMCompletionRequest("pa", "sys|A", "B")
    req_pipe_b = LLMCompletionRequest("pb", "sys", "A|B")
    self.assertNotEqual(vertex_llm._cache_key(req_pipe_a), vertex_llm._cache_key(req_pipe_b))

  def test_r4_intent_routing_gemini_25_flash_and_provider_parity(self) -> None:
    """R4 routes Stage 1 to gemini-2.5-flash and synthesis to gemini-3.7-flash."""
    from argus_sql.adapters import build_intent_llm_client

    with tempfile.TemporaryDirectory() as tmp_dir:
      # Verify `.env` `ARGUS_THINKING_BUDGET` and Vertex 2.5-flash vs 3.7-flash.
      vertex_env = Path(tmp_dir) / "vertex.env"
      vertex_env.write_text(
          "GOOGLE_CLOUD_PROJECT=demo-eval-project-01\n"
          "GOOGLE_GENAI_USE_VERTEXAI=TRUE\n"
          "ARGUS_THINKING_BUDGET=2560\n",
          encoding="utf-8",
      )
      # Confirm Stage 1 routes to `gemini-2.5-flash` and Stage 3 to `gemini-3.7-flash`.
      with mock.patch.dict("os.environ", {}, clear=True):
        live_llm = build_live_llm_client(dotenv_path=vertex_env)
        intent_llm = build_intent_llm_client(
            dotenv_path=vertex_env, fallback_to_none=False
        )
        assert isinstance(live_llm, VertexGeminiBatchClient)
        assert isinstance(intent_llm, VertexGeminiBatchClient)
        self.assertEqual(live_llm.model_name, "gemini-3.7-flash")
        self.assertEqual(intent_llm.model_name, "gemini-2.5-flash")
        self.assertEqual(live_llm.thinking_budget, 2560)
        self.assertEqual(intent_llm.thinking_budget, 2560)

      # Verify `is_vertex` parity when `ARGUS_LLM_API_KEY` + placeholder project set.
      oa_env = Path(tmp_dir) / "oa.env"
      synth_key = "test-only-" + ("tok88" * 3)
      oa_env.write_text(
          f"ARGUS_LLM_API_KEY={synth_key}\n"
          "GOOGLE_CLOUD_PROJECT=your-personal-gcp-project-id\n",
          encoding="utf-8",
      )
      # Confirm OpenAI key overrides placeholder GCP project ID in Stage 1 too.
      with mock.patch.dict("os.environ", {}, clear=True):
        oa_intent = build_intent_llm_client(
            dotenv_path=oa_env, fallback_to_none=False
        )
        self.assertIsInstance(oa_intent, OpenAICompatibleBatchClient)
        assert isinstance(oa_intent, OpenAICompatibleBatchClient)
        self.assertEqual(oa_intent.model_name, "gpt-4o-mini")

  def test_vertex_auth_blocks_pct_gcs_traversal_tainted_models_and_bad_locations(
      self,
  ) -> None:
    """Blocks `%`-encoded GCS traversal, tainted `/models/` prefixes, and edge `-`."""
    import ssl as _ssl
    from argus_sql.adapters.vertex_auth import VertexTokenResolver
    from argus_sql.adapters.vertex_auth import build_gcs_download_url
    from argus_sql.adapters.vertex_auth import build_gcs_upload_url
    from argus_sql.adapters.vertex_auth import normalize_gemini_model_name
    from argus_sql.adapters.vertex_auth import parse_gcs_uri
    from argus_sql.adapters.vertex_auth import validate_batch_job_name
    from argus_sql.adapters.vertex_auth import validate_cached_content_name
    from argus_sql.adapters.vertex_auth import validate_gcp_location

    # Challenge 3: reject single/double percent-encoded traversal (`%2e`, `%2e%2e`,
    # `..%2f`, `%5c`, `%00`, `%252e%252e`), empty `//` segments, & boundary spaces.
    for bad_gcs in (
        "gs://my-bucket/%2e",
        "gs://my-bucket/%2e%2e",
        "gs://my-bucket/runs/%2e%2e/secret.jsonl",
        "gs://my-bucket/runs/%2E%2E/secret.jsonl",
        "gs://my-bucket/runs/..%2fsecret.jsonl",
        "gs://my-bucket/runs/%2e%2e%2fsecret.jsonl",
        "gs://my-bucket/runs/%5csecret.jsonl",
        "gs://my-bucket/runs/%00secret.jsonl",
        "gs://my-bucket/runs/%252e%252e/secret.jsonl",
        "gs://my-bucket/runs//secret.jsonl",
        "gs://my-bucket /runs/secret.jsonl",
        "gs://my-bucket/ runs/secret.jsonl",
    ):
      with self.assertRaises(LLMAdapterAuthError, msg=bad_gcs):
        parse_gcs_uri(bad_gcs)

    # Concrete media upload and download endpoints must reject trailing `/` prefixes.
    with self.assertRaises(LLMAdapterAuthError):
      build_gcs_upload_url("my-bucket", "runs/r1/")
    with self.assertRaises(LLMAdapterAuthError):
      build_gcs_download_url("my-bucket", "runs/r1/")

    # Preserve nested ISO-8601 colon timestamps in Vertex batch output shard paths.
    iso_obj = (
        "prediction-gemini-3.7-flash-2026-11-11T05:00:00.000Z/"
        "predictions-00001-of-00004.jsonl"
    )
    # Confirm `parse_gcs_uri` and `build_gcs_download_url` encode `:` and `/` cleanly.
    bkt, parsed_obj = parse_gcs_uri(f"gs://my-bucket/{iso_obj}")
    self.assertEqual((bkt, parsed_obj), ("my-bucket", iso_obj))
    self.assertIn(
        "prediction-gemini-3.7-flash-2026-11-11T05%3A00%3A00.000Z%2F"
        "predictions-00001-of-00004.jsonl?alt=media",
        build_gcs_download_url(bkt, parsed_obj),
    )

    # Challenge 5a: reject leading/trailing hyphens and non-ASCII GCP locations.
    self.assertEqual(validate_gcp_location("global"), "global")
    self.assertEqual(validate_gcp_location("us-central1"), "us-central1")
    # Ensure RFC-invalid leading/trailing `-` and Kelvin sign `\u212a` fail closed.
    for bad_loc in (
        "-us-central1",
        "us-central1-",
        "-",
        "-global",
        "global-",
        "a",
        "\u212aglobal",
    ):
      with self.assertRaises(LLMAdapterAuthError, msg=bad_loc):
        validate_gcp_location(bad_loc)
    # Resource validators for caches and batch jobs must also reject edge hyphens.
    with self.assertRaises(LLMAdapterAuthError):
      validate_cached_content_name("projects/123456/locations/-global/cachedContents/c1")
    with self.assertRaises(LLMAdapterAuthError):
      validate_batch_job_name("projects/123456/locations/us-central1-/batchPredictionJobs/j1")

    # Challenge 5b: accept all 5 canonical Gemini model formats (slug, `models/`,
    # `google/`, `publishers/google/models/`, and full project/numeric publisher URI).
    for raw_model, expected_slug in (
        ("gemini-2.5-flash", "gemini-2.5-flash"),
        ("models/gemini-2.5-flash", "gemini-2.5-flash"),
        ("google/gemini-3.7-flash", "gemini-3.7-flash"),
        ("publishers/google/models/gemini-3.7-flash", "gemini-3.7-flash"),
        (
            "projects/demo-eval-project-01/locations/global/"
            "publishers/google/models/gemini-3.7-flash",
            "gemini-3.7-flash",
        ),
        (
            "projects/123456789012/locations/us-central1/"
            "publishers/google/models/gemini-2.5-flash",
            "gemini-2.5-flash",
        ),
    ):
      self.assertEqual(normalize_gemini_model_name(raw_model), expected_slug)

    # Reject traversal, SSRF URLs, query/fragment smuggling, & non-Google publishers.
    for bad_model in (
        "../../evil/models/gemini-2.5-flash",
        "%2e%2e/models/gemini-2.5-flash",
        "https://169.254.169.254/models/gemini-2.5-flash",
        "bad?query=1#/models/gemini-2.5-flash",
        "bad\\path/models/gemini-2.5-flash",
        "publishers/evil/models/gemini-2.5-flash",
        "projects/../evil/locations/global/publishers/google/models/gemini-2.5-flash",
        "models/..",
        "...",
    ):
      with self.assertRaises(LLMAdapterAuthError, msg=bad_model):
        normalize_gemini_model_name(bad_model)

    # Verify ADC token refresh wraps `http.client.HTTPException` into AuthError.
    with tempfile.TemporaryDirectory() as tmp_dir:
      adc_path = Path(tmp_dir) / "adc.json"
      adc_path.write_text(
          '{"type":"authorized_user","client_id":"c","client_secret":"s","refresh_token":"r"}',
          encoding="utf-8",
      )
      # Simulate upstream connection drop during OAuth2 token refresh POST.
      resolver = VertexTokenResolver(ssl_ctx=_ssl.create_default_context(), adc_file=adc_path)
      with mock.patch(
          "urllib.request.urlopen",
          side_effect=http.client.RemoteDisconnected("closed"),
      ):
        with self.assertRaises(LLMAdapterAuthError):
          resolver.get_bearer_token()

  def test_challenge4b_http_exception_retry_fsm_fallback_and_neg_zero_digest(
      self,
  ) -> None:
    """`BadStatusLine`/`IncompleteRead` retry, fall back in Stage 1, and normalize `-0.0`."""
    from argus_sql.adapters import build_vertex_generate_payload
    from argus_sql.linking.intent_canonicalizer import canonicalize_intent
    from argus_sql.linking.llm_intent_extractor import FlashLiteIntentExtractor

    # Part 1: Transient `BadStatusLine` + `IncompleteRead` retry and recover.
    vertex_llm = VertexGeminiBatchClient(
        project_id="demo-eval-project-01",
        model_name="gemini-2.5-flash",
        max_retries=3,
        token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
    )
    ok_body: dict[str, object] = {
        "candidates": [{"content": {"parts": [{"text": "SELECT 1"}]}}],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 2},
    }
    # Patch `time.sleep` so exponential backoff executes instantaneously.
    with mock.patch("time.sleep", return_value=None):
      with mock.patch.object(
          vertex_llm,
          "_post_generate_once",
          side_effect=[
              http.client.BadStatusLine("malformed"),
              http.client.IncompleteRead(b"{", 32),
              ok_body,
          ],
      ) as flaky_mock:
        recovered = vertex_llm.batch_complete(
            (LLMCompletionRequest("r_rec", "sys", "SELECT 1?"),)
        )
      self.assertEqual(flaky_mock.call_count, 3)
      self.assertEqual(recovered[0].sql_text, "SELECT 1")

      # Part 2: Persistent `BadStatusLine`, `OSError`, & `UnicodeDecodeError` wrap
      # into `LLMAdapterTransportError` and trigger Stage 1 deterministic FSM fallback.
      extractor = FlashLiteIntentExtractor(llm=vertex_llm)
      utterance = "uh how many schools are in Alameda?"
      expected_fsm = canonicalize_intent(utterance, "")
      bad_utf8 = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
      # Verify every transport exception family falls back to FSM with 0 tokens.
      for transport_exc in (
          http.client.BadStatusLine("bad status"),
          ConnectionResetError("socket reset"),
          bad_utf8,
      ):
        with mock.patch.object(
            vertex_llm, "_post_generate_once", side_effect=transport_exc
        ):
          with self.assertRaises(LLMAdapterTransportError):
            vertex_llm.batch_complete(
                (LLMCompletionRequest("r_fail", "sys", f"Q:{type(transport_exc).__name__}"),)
            )
          frames, tok_counts = extractor.extract_intent_batch_with_usage(
              ((utterance, ""),)
          )
          self.assertEqual(frames[0], expected_fsm)
          self.assertEqual(tok_counts, (0,))

    # Part 3: IEEE-754 `-0.0` temperature canonicalizes identically to `0.0`.
    # Both `_cache_key` and wire `generationConfig` must emit positive `0.0`.
    req_pos = LLMCompletionRequest("p0", "sys", "SELECT 1", temperature=0.0)
    req_neg = LLMCompletionRequest("n0", "sys", "SELECT 1", temperature=-0.0)
    self.assertEqual(vertex_llm._cache_key(req_pos), vertex_llm._cache_key(req_neg))
    pay_neg = build_vertex_generate_payload(req_neg, vertex_llm.thinking_budget)
    gen_neg = pay_neg["generationConfig"]
    assert isinstance(gen_neg, dict)
    self.assertEqual(repr(gen_neg["temperature"]), "0.0")

  def test_challenge6_eager_thinking_budget_validation_across_all_providers(
      self,
  ) -> None:
    """Invalid `thinking_budget` raises `LLMAdapterAuthError` even offline or on OpenAI."""
    from argus_sql.adapters import build_intent_llm_client

    with tempfile.TemporaryDirectory() as tmp_dir:
      empty_env = Path(tmp_dir) / "empty.env"
      empty_env.write_text("", encoding="utf-8")

      # Offline environment (`fallback_to_none=True`): invalid explicit budgets
      # (`-5`, `True`, `False`) must raise `LLMAdapterAuthError`, NOT return `None`.
      with mock.patch.dict("os.environ", {}, clear=True):
        for bad_budget in (-5, -1, True, False):
          with self.assertRaises(LLMAdapterAuthError):
            build_intent_llm_client(
                dotenv_path=empty_env,
                fallback_to_none=True,
                thinking_budget=bad_budget,
            )
        # Valid or unset budget with missing credentials still returns `None`.
        self.assertIsNone(
            build_intent_llm_client(
                dotenv_path=empty_env,
                fallback_to_none=True,
                thinking_budget=0,
            )
        )

      # Malformed `ARGUS_THINKING_BUDGET` env var fails closed offline too.
      for bad_env_budget in ("abc", "-5", "2048.5"):
        with mock.patch.dict(
            "os.environ", {"ARGUS_THINKING_BUDGET": bad_env_budget}, clear=True
        ):
          with self.assertRaises(LLMAdapterAuthError):
            build_intent_llm_client(dotenv_path=empty_env, fallback_to_none=True)

      # OpenAI provider with valid API key also rejects invalid thinking budgets.
      synth_key = "test-only-" + ("key77" * 3)
      with mock.patch.dict(
          "os.environ", {"ARGUS_LLM_API_KEY": synth_key}, clear=True
      ):
        for bad_budget in (-5, True):
          with self.assertRaises(LLMAdapterAuthError):
            build_live_llm_client(
                provider="openai",
                dotenv_path=empty_env,
                thinking_budget=bad_budget,
            )
          with self.assertRaises(LLMAdapterAuthError):
            build_intent_llm_client(
                provider="openai",
                dotenv_path=empty_env,
                fallback_to_none=True,
                thinking_budget=bad_budget,
            )
        # Invalid `ARGUS_THINKING_BUDGET` env var also fails on OpenAI live client.
        with mock.patch.dict(
            "os.environ",
            {"ARGUS_LLM_API_KEY": synth_key, "ARGUS_THINKING_BUDGET": "abc"},
            clear=True,
        ):
          with self.assertRaises(LLMAdapterAuthError):
            build_live_llm_client(provider="openai", dotenv_path=empty_env)

  def test_null_usage_and_malformed_candidates_fallback_in_stage1_fsm(self) -> None:
    """Explicit JSON `null` usage/ADC fields coalesce and bad payloads trigger FSM."""
    import io as _io
    import json as _json
    import ssl as _ssl
    from argus_sql.adapters.vertex_auth import VertexTokenResolver
    from argus_sql.linking.intent_canonicalizer import canonicalize_intent
    from argus_sql.linking.llm_intent_extractor import FlashLiteIntentExtractor

    # Coalesce explicit `None` on `usageMetadata`, `content`, and individual token fields.
    # Prevents unwrapped `TypeError` when Vertex returns `null` token counts.
    null_u: dict[str, object] = {
        "candidates": [
            {"content": {"parts": [{"text": '{"canonical_question": "Count schools?"}'}]}}
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": None,
            "thoughtsTokenCount": None,
        },
    }
    raw_txt, p_tok, c_tok = extract_gemini_completion_triple(null_u, 40)
    self.assertIn("Count schools?", raw_txt)
    self.assertEqual((p_tok, c_tok), (10, len(raw_txt) // 4))
    # Explicit `content: null` and `usageMetadata: null` fall back deterministically.
    self.assertEqual(
        extract_gemini_completion_triple(
            {"candidates": [{"content": None}], "usageMetadata": None}, 20
        ),
        ("", 5, 0),
    )

    # Stage 1 `FlashLiteIntentExtractor` succeeds on `null` usage and falls back to
    # deterministic FSM on malformed `candidates`/`content`/`parts`/boolean/neg usage.
    llm = VertexGeminiBatchClient(
        "demo-eval-project-01",
        model_name="gemini-2.5-flash",
        max_retries=1,
        token_supplier=lambda: ("synthetic-test-bearer", 3600.0),
    )
    extractor = FlashLiteIntentExtractor(llm)
    with mock.patch.object(llm, "_post_generate_once", return_value=null_u):
      self.assertEqual(
          extractor.extract_intent("uh count?").canonical_question, "Count schools?"
      )

    # Verify every malformed candidate or invalid usage payload triggers FSM fallback.
    utterance = "uh how many schools?"
    expected_fsm = canonicalize_intent(utterance, "")
    for bad_body in (
        {"candidates": None},
        {"candidates": [{"content": "bad"}]},
        {"candidates": [{"content": {"parts": [123]}}]},
        {
            "candidates": [{"content": {"parts": [{"text": "SELECT 1"}]}}],
            "usageMetadata": {"candidatesTokenCount": True},
        },
        {
            "candidates": [{"content": {"parts": [{"text": "SELECT 1"}]}}],
            "usageMetadata": {"thoughtsTokenCount": -1},
        },
        {
            "candidates": [{"content": {"parts": [{"text": "SELECT 1"}]}}],
            "usageMetadata": {"promptTokenCount": "abc"},
        },
    ):
      with mock.patch.object(llm, "_post_generate_once", return_value=bad_body):
        # Direct `batch_complete` surfaces `LLMAdapterTransportError`, while Stage 1 degrades.
        with self.assertRaises(LLMAdapterTransportError):
          llm.batch_complete((LLMCompletionRequest(f"b_{id(bad_body)}", "s", "Q?"),))
        self.assertEqual(extractor.extract_intent(utterance), expected_fsm)
    # Huge 400-digit integer temperature (`float(10**400)`) wraps OverflowError cleanly.
    with self.assertRaises(LLMAdapterTransportError):
      llm.batch_complete((LLMCompletionRequest("ovf", "s", "Q?", temperature=10**400),))

    # Verify `VertexTokenResolver` fails closed on `"access_token": None` (never `"None"`),
    # coalesces `"expires_in": None` to `3600.0`, and rejects `"expires_in": True` / `<= 0`.
    with tempfile.TemporaryDirectory() as tmp_dir:
      adc_path = Path(tmp_dir) / "adc.json"
      adc_path.write_text(
          '{"type":"authorized_user","client_id":null,"client_secret":"s","refresh_token":"r"}',
          encoding="utf-8",
      )
      ssl_ctx = _ssl.create_default_context()
      # Explicit `"expires_in": null` coalesces to 3600s and caches valid bearer token.
      ok_oauth = _io.BytesIO(b'{"access_token":"fresh-adc-tok","expires_in":null}')
      ok_oauth.__enter__ = lambda s=ok_oauth: s  # type: ignore[method-assign]
      ok_oauth.__exit__ = lambda *a: False  # type: ignore[method-assign]
      with mock.patch.dict("os.environ", {}, clear=True):
        with mock.patch("urllib.request.urlopen", return_value=ok_oauth):
          self.assertEqual(
              VertexTokenResolver(ssl_ctx=ssl_ctx, adc_file=adc_path).get_bearer_token(),
              "fresh-adc-tok",
          )
        # Reject `"access_token": null` and invalid `expires_in` (`True`, `0`, `NaN`, `10**400`).
        for bad_oauth in (
            {"access_token": None, "expires_in": 3600},
            {"access_token": "tok", "expires_in": True},
            {"access_token": "tok", "expires_in": 0},
            {"access_token": "tok", "expires_in": "NaN"},
            {"access_token": "tok", "expires_in": 10**400},
        ):
          buf = _io.BytesIO(_json.dumps(bad_oauth).encode("utf-8"))
          buf.__enter__ = lambda s=buf: s  # type: ignore[method-assign]
          buf.__exit__ = lambda *a: False  # type: ignore[method-assign]
          with mock.patch("urllib.request.urlopen", return_value=buf):
            with self.assertRaises(LLMAdapterAuthError):
              VertexTokenResolver(ssl_ctx=ssl_ctx, adc_file=adc_path).get_bearer_token()


if __name__ == "__main__":
  unittest.main()

