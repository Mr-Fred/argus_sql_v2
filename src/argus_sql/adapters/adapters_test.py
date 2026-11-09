"""Unit tests for HTTPS batch security, SQL extraction, and DDL synthesis."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from argus_sql.adapters.client_factory import build_live_llm_client
from argus_sql.adapters.client_factory import load_dotenv_defaults
from argus_sql.adapters.deterministic_synthesizer import SchemaGuidedDiagnosticSynthesizer
from argus_sql.adapters.exceptions import LLMAdapterAuthError
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
    # Subsequent batch call hits the SHA-256 cache with 0 billed tokens.
    with mock.patch.object(
        client, "_post_generate_once", return_value=sample_body
    ) as post_mock_cached:
      cached_out = client.batch_complete((req_a,))
    self.assertEqual(post_mock_cached.call_count, 0)
    self.assertTrue(cached_out[0].from_cache)
    self.assertEqual(cached_out[0].prompt_tokens, 0)

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
    # and `generationConfig.maxOutputTokens` must honor the 8192 thinking floor.
    sys_inst = seen_payloads[0]["systemInstruction"]
    assert isinstance(sys_inst, dict)
    self.assertNotIn("role", sys_inst)
    gen_cfg = seen_payloads[0]["generationConfig"]
    assert isinstance(gen_cfg, dict)
    self.assertGreaterEqual(int(gen_cfg["maxOutputTokens"]), 8192)

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


if __name__ == "__main__":
  unittest.main()
