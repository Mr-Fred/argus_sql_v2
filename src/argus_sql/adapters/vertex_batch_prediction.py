"""Offline Vertex AI Batch Prediction (`BatchPredictionJob`) + GCS JSONL runner.

Serializes deduplicated `LLMCompletionRequest` batches to Google Cloud Storage
JSONL, submits regional Vertex AI `batchPredictionJobs` for `gemini-3.7-flash`,
supports both decoupled CI (`submit_job` / `poll_job` / `collect_job_responses`)
and unified `BatchLLMPort.batch_complete` workflows, and correlates unordered
multi-shard outputs back to exact caller order using Python's standard library.
"""

from collections.abc import Callable
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import datetime
import hashlib
import http.client
import json
import logging
import pathlib
import re
import secrets
import ssl
import threading
import time
import urllib.error
import urllib.request

from argus_sql.adapters.client_factory import load_dotenv_defaults
from argus_sql.adapters.exceptions import LLMAdapterAuthError
from argus_sql.adapters.exceptions import LLMAdapterTransportError
from argus_sql.adapters.llm_batch_client import extract_sql_from_completion
from argus_sql.adapters.types import LLMCompletionRequest
from argus_sql.adapters.types import LLMCompletionResponse
from argus_sql.adapters.vertex_auth import VertexTokenResolver
from argus_sql.adapters.vertex_auth import build_gcs_download_url
from argus_sql.adapters.vertex_auth import build_gcs_list_url
from argus_sql.adapters.vertex_auth import build_gcs_upload_url
from argus_sql.adapters.vertex_auth import build_vertex_batch_jobs_url
from argus_sql.adapters.vertex_auth import normalize_gemini_model_name
from argus_sql.adapters.vertex_auth import parse_gcs_uri
from argus_sql.adapters.vertex_auth import resolve_vertex_batch_location
from argus_sql.adapters.vertex_auth import validate_batch_job_name as _validate_canonical_batch_job_name
from argus_sql.adapters.vertex_auth import validate_gcp_project_id
from argus_sql.adapters.vertex_gemini_client import DEFAULT_THINKING_BUDGET
from argus_sql.adapters.vertex_gemini_client import build_vertex_generate_payload
from argus_sql.adapters.vertex_gemini_client import coerce_finite_float
from argus_sql.adapters.vertex_gemini_client import coerce_nonneg_int
from argus_sql.adapters.vertex_gemini_client import extract_gemini_completion_triple
from argus_sql.adapters.vertex_gemini_client import parse_thinking_budget

# Module-level logger using Google Python Style Guide lazy-% formatting.
_LOGGER = logging.getLogger(__name__)

# Callable signature `(method, url, body_bytes, headers) -> response_bytes`.
HttpTransport = Callable[[str, str, bytes | None, dict[str, str]], bytes]

# Default regional location used when `GOOGLE_CLOUD_LOCATION` is `'global'`.
_DEFAULT_BATCH_REGION = "us-central1"

# Maximum manifest JSON file size accepted on disk (32 MiB safety cap).
_MAX_MANIFEST_BYTES = 33_554_432

# Transient HTTP status codes retried with bounded exponential backoff.
_RETRYABLE_HTTP_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})

# Unedited `.env` template placeholders treated as missing project IDs.
_PLACEHOLDER_PROJECTS: frozenset[str] = frozenset({
    "your-personal-gcp-project-id",
    "your-gcp-project-id",
    "<your-gcp-project-id>",
})

# Hex SHA-256 digest validator enforced when loading manifests from disk.
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

# Terminal Vertex AI BatchPredictionJob states indicating output availability.
_TERMINAL_SUCCESS_STATES: frozenset[str] = frozenset({
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
})

# Terminal Vertex AI BatchPredictionJob states requiring immediate error exit.
_TERMINAL_FAILURE_STATES: frozenset[str] = frozenset({
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
})


# Immutable status snapshot returned by `poll_job` and `await_job`.
@dataclasses.dataclass(frozen=True)
class BatchJobStatus:
  """Point-in-time status of a regional Vertex AI `BatchPredictionJob`.

  Attributes:
    job_name: Canonical `projects/.../locations/.../batchPredictionJobs/...` ID.
    state: Vertex AI `JobState` enum string (e.g., `'JOB_STATE_SUCCEEDED'`).
    gcs_output_dir: Server-populated `outputInfo.gcsOutputDirectory` URI.
    error_message: Sanitized error detail when `state` is terminal failure.
    successful_count: Completed prediction row count reported by Vertex AI.
    failed_count: Failed prediction row count reported by Vertex AI.
    incomplete_count: Incomplete prediction row count reported by Vertex AI.
  """

  job_name: str
  state: str
  gcs_output_dir: str = ""
  error_message: str = ""
  successful_count: int = 0
  failed_count: int = 0
  incomplete_count: int = 0

  @property
  def is_terminal(self) -> bool:
    """Returns True when the batch job has reached any terminal state."""
    return self.state in (_TERMINAL_SUCCESS_STATES | _TERMINAL_FAILURE_STATES)

  @property
  def succeeded(self) -> bool:
    """Returns True when the batch job produced collectable output shards."""
    return self.state in _TERMINAL_SUCCESS_STATES


# Secret-free serializable manifest enabling decoupled submit/collect CI runs.
@dataclasses.dataclass(frozen=True)
class BatchJobManifest:
  """Secret-free manifest persisted between decoupled CI submit and collect runs.

  Attributes:
    job_name: Canonical Vertex AI `BatchPredictionJob` resource name.
    project_id: Validated GCP billing project identifier.
    location: Regional Vertex AI batch endpoint slug (never `'global'`).
    model_name: Normalized Gemini model slug (e.g., `'gemini-3.7-flash'`).
    gcs_input_uri: Uploaded `gs://.../input.jsonl` source URI.
    gcs_output_uri_prefix: Requested `gs://.../output/` destination prefix.
    entries: Ordered `(request_id, sha256_digest, prompt_char_len)` triples.
    created_at_utc: ISO-8601 UTC submission timestamp string.
  """

  job_name: str
  project_id: str
  location: str
  model_name: str
  gcs_input_uri: str
  gcs_output_uri_prefix: str
  entries: tuple[tuple[str, str, int], ...]
  created_at_utc: str = ""

  def to_dict(self) -> dict[str, object]:
    """Serializes manifest metadata to a JSON-compatible dictionary.

    Returns:
      Dictionary containing job routing metadata and ordered digest entries.
    """
    # Persist only request IDs, SHA-256 digests, and lengths (zero secrets).
    # Schema version allows safe forward-compatible evolution across CI runs.
    return {
        "schema_version": "argus_batch_manifest.v1",
        "job_name": self.job_name,
        "project_id": self.project_id,
        "location": self.location,
        "model_name": self.model_name,
        "gcs_input_uri": self.gcs_input_uri,
        "gcs_output_uri_prefix": self.gcs_output_uri_prefix,
        "created_at_utc": self.created_at_utc,
        "entries": [
            {"request_id": req_id, "digest": digest, "prompt_char_len": char_len}
            for req_id, digest, char_len in self.entries
        ],
    }

  @classmethod
  def from_dict(cls, raw: dict[str, object]) -> "BatchJobManifest":
    """Validates and reconstructs a `BatchJobManifest` from parsed JSON.

    Args:
      raw: Parsed JSON dictionary produced by `to_dict()`.

    Returns:
      Validated immutable `BatchJobManifest` instance.

    Raises:
      LLMAdapterAuthError: If project, location, model, job, or GCS URI fails
        allow-list validation.
      LLMAdapterTransportError: If manifest structure or entry digests are
        malformed.
    """
    # Verify manifest schema marker before trusting deserialized fields.
    if raw.get("schema_version") != "argus_batch_manifest.v1":
      raise LLMAdapterTransportError("Unsupported BatchJobManifest version.")
    # Validate all cloud identifiers against SSRF and traversal allow-lists.
    proj = validate_gcp_project_id(_opt_str(raw, "project_id"))
    raw_loc = _opt_str(raw, "location").strip()
    # Reject `'global'` explicitly because persisted manifests must be regional.
    if raw_loc.lower() == "global":
      raise LLMAdapterAuthError("BatchJobManifest location must be regional.")
    loc = resolve_vertex_batch_location(raw_loc)
    model = normalize_gemini_model_name(_opt_str(raw, "model_name"))
    job_name, job_loc = validate_batch_job_name(
        _opt_str(raw, "job_name"), proj, loc
    )
    if job_loc != loc:
      raise LLMAdapterAuthError(
          "BatchJobManifest job_name region mismatches manifest location."
      )
    # Confirm input and output Cloud Storage URIs obey strict gs:// rules.
    in_uri = _opt_str(raw, "gcs_input_uri")
    out_uri = _opt_str(raw, "gcs_output_uri_prefix")
    parse_and_validate_gcs_uri(in_uri)
    parse_and_validate_gcs_uri(out_uri)

    # Require at least one ordered request entry for 1-to-1 tuple alignment.
    raw_entries = raw.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
      raise LLMAdapterTransportError("BatchJobManifest entries must be non-empty.")

    # Validate each entry's request_id, 64-char SHA-256 hex digest, and length.
    parsed_entries: list[tuple[str, str, int]] = []
    for item in raw_entries:
      if not isinstance(item, dict):
        raise LLMAdapterTransportError("Malformed BatchJobManifest entry.")
      req_id = _opt_str(item, "request_id").strip()
      digest = _opt_str(item, "digest").strip().lower()
      char_len_raw = item.get("prompt_char_len", 0)
      # Reject booleans explicitly because bool subclasses int in Python.
      if (
          not req_id
          or not _SHA256_HEX_RE.match(digest)
          or isinstance(char_len_raw, bool)
          or not isinstance(char_len_raw, int)
          or char_len_raw < 0
      ):
        raise LLMAdapterTransportError("Invalid entry fields in BatchJobManifest.")
      parsed_entries.append((req_id, digest, char_len_raw))

    # Return frozen manifest ready for decoupled CI polling and collection.
    return cls(
        job_name=job_name,
        project_id=proj,
        location=loc,
        model_name=model,
        gcs_input_uri=in_uri,
        gcs_output_uri_prefix=out_uri,
        entries=tuple(parsed_entries),
        created_at_utc=_opt_str(raw, "created_at_utc"),
    )

  def save_json(self, path: pathlib.Path) -> None:
    """Writes this manifest to `path` as UTF-8 JSON with zero credentials.

    Args:
      path: Destination filesystem path for the JSON manifest.

    Raises:
      LLMAdapterTransportError: If `path` is a symlink or cannot be written.
    """
    # Refuse symlink targets to prevent arbitrary file overwrite attacks.
    if path.is_symlink():
      raise LLMAdapterTransportError("Refusing to write manifest to symlink.")
    # Ensure parent directories exist and write deterministic sorted UTF-8 JSON.
    try:
      path.parent.mkdir(parents=True, exist_ok=True)
      serialized = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
      path.write_text(serialized, encoding="utf-8")
    except OSError as err:
      raise LLMAdapterTransportError("Failed writing BatchJobManifest JSON.") from err

  @classmethod
  def load_json(cls, path: pathlib.Path) -> "BatchJobManifest":
    """Loads and validates a `BatchJobManifest` from a local UTF-8 JSON file.

    Args:
      path: Filesystem path to a previously saved manifest JSON file.

    Returns:
      Validated `BatchJobManifest` ready for `poll_job` / `collect_job_responses`.

    Raises:
      LLMAdapterAuthError: If cloud identifiers inside the manifest are invalid.
      LLMAdapterTransportError: If `path` is missing, a symlink, oversized, or
        invalid JSON.
    """
    # Enforce symlink and byte-size guards matching `load_dotenv_defaults`.
    if path.is_symlink() or not path.is_file():
      raise LLMAdapterTransportError("BatchJobManifest file missing or symlink.")
    # Bound input bytes before parsing JSON to prevent memory exhaustion.
    try:
      if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise LLMAdapterTransportError("BatchJobManifest exceeds size ceiling.")
      parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as err:
      raise LLMAdapterTransportError("Failed reading BatchJobManifest JSON.") from err
    # Reject top-level JSON arrays or primitives before schema validation.
    if not isinstance(parsed, dict):
      raise LLMAdapterTransportError("BatchJobManifest root must be a JSON object.")
    return cls.from_dict(parsed)


def parse_and_validate_gcs_uri(
    gcs_uri: str,
    default_prefix: str = "",
) -> tuple[str, str]:
  """Parses a `gs://` URI into a validated `(bucket, object_path)` pair.

  Args:
    gcs_uri: Candidate Cloud Storage URI (`'gs://bucket'` or `'gs://bucket/p'`).
    default_prefix: Optional default object prefix when URI omits a path.

  Returns:
    Tuple `(bucket_name, normalized_object_path)` without leading/trailing `/`.

  Raises:
    LLMAdapterAuthError: If `gcs_uri` is not `gs://` or contains traversal/bad
      bucket or object path characters.
  """
  # Delegate scheme, bucket, backslash, query/fragment, and `..` checks to `vertex_auth`.
  bucket, raw_obj = parse_gcs_uri(gcs_uri, allow_empty_object=True)
  norm_path = raw_obj.strip("/") if raw_obj.strip("/") else default_prefix.strip("/")
  if norm_path:
    # Re-validate normalized default prefix if `raw_obj` was omitted.
    parse_gcs_uri(f"gs://{bucket}/{norm_path}", allow_empty_object=False)
  return (bucket, norm_path)


def validate_batch_job_name(
    job_name: str,
    default_project: str | None = None,
    default_location: str | None = None,
) -> tuple[str, str]:
  """Validates a Vertex AI `BatchPredictionJob` resource name or bare job ID.

  Args:
    job_name: Full `projects/.../locations/.../batchPredictionJobs/...` resource
      name or bare job ID string.
    default_project: Project ID interpolated when `job_name` is a bare ID.
    default_location: Regional location interpolated when `job_name` is bare.

  Returns:
    Pair `(canonical_job_name, regional_location)`.

  Raises:
    LLMAdapterAuthError: If `job_name` is malformed, uses `'global'`, or
      attempts path traversal.
  """
  cleaned = job_name.strip()
  # Expand bare job IDs when caller supplies validated runner defaults.
  if cleaned and "/" not in cleaned and default_project and default_location:
    proj = validate_gcp_project_id(default_project)
    reg = resolve_vertex_batch_location(default_location)
    cleaned = f"projects/{proj}/locations/{reg}/batchPredictionJobs/{cleaned}"

  # Delegate strict regional & numeric-project allow-list check to `vertex_auth`.
  canonical = _validate_canonical_batch_job_name(cleaned)
  parts = canonical.split("/")
  return (canonical, parts[3])


def _opt_str(mapping: dict[str, object], key: str, default: str = "") -> str:
  """Coalesces omitted/`None` string fields to `default` and rejects non-strings."""
  # Coalesce explicit JSON `null` (`None`) to `default` and block non-str values.
  val = mapping.get(key)
  if val is not None and not isinstance(val, str):
    raise LLMAdapterTransportError(f"Malformed non-string field '{key}'.")
  return default if val is None else val


def _opt_dict(mapping: dict[str, object], key: str) -> dict[str, object]:
  """Coalesces omitted/`None` object fields to `{}` and rejects non-dicts."""
  # Coalesce explicit JSON `null` (`None`) to `{}` and block scalar/list values.
  val = mapping.get(key)
  if val is not None and not isinstance(val, dict):
    raise LLMAdapterTransportError(f"Malformed non-object field '{key}'.")
  return {} if val is None else val


def _build_batch_request_dict(
    req: LLMCompletionRequest,
    configured_thinking_budget: int,
) -> dict[str, object]:
  """Builds a Vertex `GenerateContentRequest` dict without output clamping."""
  # Reuse `build_vertex_generate_payload` so online & batch wire schemas match.
  payload = build_vertex_generate_payload(
      req,
      configured_thinking_budget=configured_thinking_budget,
  )
  gen_cfg = payload.get("generationConfig")
  if isinstance(gen_cfg, dict) and "temperature" in gen_cfg:
    # Normalize IEEE-754 `-0.0` to `0.0` before writing JSONL request lines.
    temp_val = coerce_finite_float(req.temperature, 0.0)
    gen_cfg["temperature"] = 0.0 if temp_val == 0.0 else temp_val
  return payload


def _batch_request_digest(request_obj: dict[str, object]) -> str:
  """Computes canonical SHA-256 over a Vertex `GenerateContentRequest` dict."""
  if not isinstance(request_obj, dict):
    raise LLMAdapterTransportError("Batch request object must be a JSON dict.")
  # Coalesce explicit `None` on `generationConfig` / `thinkingConfig`; reject bad types.
  gen_dict = _opt_dict(request_obj, "generationConfig")
  think_dict = _opt_dict(gen_dict, "thinkingConfig")

  # Canonicalize float temperature (`None`/`-0.0` -> `0.0`) and optional `thinkingBudget`.
  temp_val = round(coerce_finite_float(gen_dict.get("temperature"), 0.0), 6)
  norm_gen: dict[str, object] = {"temperature": 0.0 if temp_val == 0.0 else temp_val}
  if think_dict.get("thinkingBudget") is not None:
    norm_gen["thinkingConfig"] = {
        "thinkingBudget": coerce_nonneg_int(think_dict.get("thinkingBudget"), 0)
    }

  # Coalesce explicit `None` on `contents`, `systemInstruction`, and `cachedContent`.
  contents_val = [] if request_obj.get("contents") is None else request_obj.get("contents")
  if not isinstance(contents_val, list):
    raise LLMAdapterTransportError("Malformed contents list in batch request.")
  canonical: dict[str, object] = {"contents": contents_val, "generationConfig": norm_gen}
  if request_obj.get("systemInstruction") is not None:
    canonical["systemInstruction"] = _opt_dict(request_obj, "systemInstruction")
  cached_ref = _opt_str(request_obj, "cachedContent")
  if cached_ref:
    canonical["cachedContent"] = cached_ref

  # Hash compact sorted UTF-8 JSON (`allow_nan=False` rejects nested NaN/Inf).
  try:
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
  except (TypeError, ValueError) as err:
    raise LLMAdapterTransportError("Non-serializable batch request payload.") from err
  return hashlib.sha256(encoded).hexdigest()


def serialize_batch_requests_jsonl(
    requests: Sequence[LLMCompletionRequest],
    configured_thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> tuple[bytes, tuple[tuple[str, str, int], ...]]:
  """Deduplicates `requests` by SHA-256 and encodes unique GCS JSONL lines.

  Args:
    requests: Sequence of `LLMCompletionRequest` items to batch.
    configured_thinking_budget: Base reasoning token budget (`>= 2048` on CoT).

  Returns:
    Pair `(utf8_jsonl_bytes, ordered_manifest_entries)`.
  """
  # Preserve caller order in `entries` while emitting each unique digest once.
  entries: list[tuple[str, str, int]] = []
  seen_digests: set[str] = set()
  jsonl_lines: list[str] = []

  # Collapse identical prompts so Vertex Batch Prediction bills duplicates once.
  for req in requests:
    core_req = _build_batch_request_dict(req, configured_thinking_budget)
    digest = _batch_request_digest(core_req)
    entries.append((req.request_id, digest, len(req.user_prompt)))
    if digest not in seen_digests:
      seen_digests.add(digest)
      labeled_req = dict(core_req)
      # Truncate label value to 32 lowercase hex chars (`<= 63` GCP label cap).
      labeled_req["labels"] = {"argus_hash": digest[:32]}
      jsonl_lines.append(json.dumps({"request": labeled_req}, ensure_ascii=False))

  # Encode newline-delimited JSON objects without trailing blank lines.
  payload_bytes = ("\n".join(jsonl_lines)).encode("utf-8")
  return (payload_bytes, tuple(entries))


class VertexBatchPredictionRunner:
  """Offline Vertex AI `BatchPredictionJob` runner implementing `BatchLLMPort`.

  Attributes:
    provider_name: Adapter identifier (`'vertex_batch'`).
    project_id: Validated Google Cloud project ID billed for batch jobs & GCS.
    location: Regional Vertex AI batch endpoint (`'global'` -> `'us-central1'`).
    model_name: Normalized Gemini model slug (`'gemini-3.7-flash'` default).
    gcs_bucket: Validated Cloud Storage staging bucket name.
    gcs_prefix: Validated Cloud Storage staging object prefix.
    gcs_uri_prefix: Normalized `'gs://<bucket>/<prefix>'` staging root.
    thinking_budget: Configured reasoning budget (`ARGUS_THINKING_BUDGET`).
    max_concurrency: Maximum parallel GCS shard download workers.
  """

  def __init__(
      self,
      project_id: str | None = None,
      gcs_uri_prefix: str | None = None,
      model_name: str = "gemini-3.7-flash",
      location: str | None = None,
      thinking_budget: int | None = None,
      poll_interval_sec: float = 10.0,
      poll_timeout_sec: float = 3600.0,
      max_concurrency: int = 16,
      timeout_sec: float = 120.0,
      max_retries: int = 3,
      dotenv_path: pathlib.Path | None = None,
      adc_file: pathlib.Path | None = None,
      token_supplier: Callable[[], tuple[str, float]] | None = None,
      http_transport: HttpTransport | None = None,
      sleep_fn: Callable[[float], None] = time.sleep,
  ) -> None:
    """Initializes regional Vertex AI Batch Prediction and GCS JSONL transport.

    Args:
      project_id: Optional GCP project ID (defaults to `GOOGLE_CLOUD_PROJECT`).
      gcs_uri_prefix: Optional `gs://bucket[/prefix]` staging URI (defaults to
        `ARGUS_GCS_BATCH_URI`).
      model_name: Target Gemini model slug (`'gemini-3.7-flash'` by default).
      location: Optional regional Vertex AI location (reads
        `GOOGLE_CLOUD_BATCH_LOCATION` then `GOOGLE_CLOUD_LOCATION`, mapping
        `'global'` -> `'us-central1'`).
      thinking_budget: Optional reasoning token budget (defaults to
        `ARGUS_THINKING_BUDGET` or `2048`).
      poll_interval_sec: Seconds between `batchPredictionJobs.get` polls.
      poll_timeout_sec: Maximum wait duration in `await_job` before failing.
      max_concurrency: Bounded worker pool size for parallel shard downloads.
      timeout_sec: Per-request HTTPS socket timeout in seconds.
      max_retries: Bounded retry attempts on transient 429/5xx HTTP errors.
      dotenv_path: Optional `.env` path for non-secret runtime defaults.
      adc_file: Optional path override for Google Cloud ADC credentials JSON.
      token_supplier: Optional injectable `(token, ttl_sec)` provider for tests.
      http_transport: Optional injectable `(method, url, body, headers) -> bytes`
        callable for 100% hermetic offline tests.
      sleep_fn: Injectable sleep function (`time.sleep` in prod; no-op in tests).

    Raises:
      LLMAdapterAuthError: If project ID, location, model, GCS URI, or thinking
        budget configuration is missing or invalid.
    """
    cfg = load_dotenv_defaults(dotenv_path)
    self.provider_name = "vertex_batch"

    # Resolve and validate GCP project ID without accepting template placeholders.
    raw_proj = (project_id or cfg.get("GOOGLE_CLOUD_PROJECT", "")).strip()
    clean_proj = "" if raw_proj.lower() in _PLACEHOLDER_PROJECTS else raw_proj
    self.project_id = validate_gcp_project_id(clean_proj)

    # Resolve regional batch endpoint, auto-mapping `'global'` -> `'us-central1'`.
    raw_loc = (
        location
        or cfg.get("GOOGLE_CLOUD_BATCH_LOCATION", "").strip()
        or cfg.get("GOOGLE_CLOUD_LOCATION", "").strip()
        or _DEFAULT_BATCH_REGION
    )
    self.location = resolve_vertex_batch_location(raw_loc)

    # Resolve Gemini model slug from constructor, `ARGUS_LLM_MODEL`, or default.
    raw_model = model_name.strip()
    if not raw_model or raw_model.lower() == "auto":
      raw_model = cfg.get("ARGUS_LLM_MODEL", "").strip() or "gemini-3.7-flash"
    self.model_name = normalize_gemini_model_name(raw_model)

    # Validate staging GCS bucket and prefix (`ARGUS_GCS_BATCH_URI`).
    raw_gcs = (gcs_uri_prefix or cfg.get("ARGUS_GCS_BATCH_URI", "")).strip()
    self.gcs_bucket, self.gcs_prefix = parse_and_validate_gcs_uri(
        raw_gcs, default_prefix="argus-sql-batch"
    )
    self.gcs_uri_prefix = f"gs://{self.gcs_bucket}/{self.gcs_prefix}"

    # Parse `ARGUS_THINKING_BUDGET` (rejecting bool, non-int, and < 0).
    raw_budget: int | str | None = (
        thinking_budget
        if thinking_budget is not None
        else cfg.get("ARGUS_THINKING_BUDGET", "").strip() or None
    )
    self.thinking_budget = parse_thinking_budget(raw_budget)

    self._poll_interval_sec = max(0.01, float(poll_interval_sec))
    self._poll_timeout_sec = max(self._poll_interval_sec, float(poll_timeout_sec))
    self.max_concurrency = max(1, min(max_concurrency, 32))
    self._timeout_sec = timeout_sec
    self._max_retries = max(1, min(max_retries, 6))
    self._http_transport = http_transport
    self._sleep_fn = sleep_fn

    # Enforce verified TLS 1.2+ certificates on all live Google Cloud traffic.
    self._ssl_ctx = ssl.create_default_context()
    self._ssl_ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    self._auth = VertexTokenResolver(
        ssl_ctx=self._ssl_ctx,
        adc_file=adc_file,
        token_supplier=token_supplier,
    )

    # Thread-safe cross-call SHA-256 completion cache matching online client.
    self._cache: dict[str, tuple[str, int, int]] = {}
    self._cache_lock = threading.Lock()
    self._inflight_sem = threading.Semaphore(self.max_concurrency)

  def _execute_http_once(
      self,
      method: str,
      url: str,
      body: bytes | None,
      content_type: str | None,
      bearer_token: str,
  ) -> bytes:
    """Executes one HTTPS request via injected transport or `urllib.request`."""
    # Enforce HTTPS scheme on every outbound Vertex AI and GCS request.
    if not url.startswith("https://"):
      raise LLMAdapterAuthError("Cloud batch endpoints must use 'https://'.")
    # Never log `headers` so live OAuth2 bearer tokens cannot leak to stderr.
    headers: dict[str, str] = {
        "Authorization": f"Bearer {bearer_token}",
        "x-goog-user-project": self.project_id,
    }
    if content_type is not None:
      headers["Content-Type"] = content_type

    # Bound concurrent open sockets across worker threads via semaphore.
    with self._inflight_sem:
      # Short-circuit to hermetic callable during offline unit/E2E tests.
      if self._http_transport is not None:
        return self._http_transport(method, url, body, headers)
      http_req = urllib.request.Request(url, data=body, headers=headers, method=method)
      with urllib.request.urlopen(
          http_req, timeout=self._timeout_sec, context=self._ssl_ctx
      ) as resp:
        return resp.read()

  def _request_with_retry(
      self,
      method: str,
      url: str,
      body: bytes | None = None,
      content_type: str | None = None,
  ) -> bytes:
    """Executes an HTTPS call with 401 token refresh and transient retries."""
    # Acquire cached or refreshed bearer credential prior to first attempt.
    bearer_token = self._auth.get_bearer_token()
    refreshed_on_401 = False
    last_err: Exception | None = None

    # Delegate single HTTP attempt to `self._execute_http_once` (no N+1 loop).
    for attempt in range(self._max_retries):
      try:
        return self._execute_http_once(method, url, body, content_type, bearer_token)
      except urllib.error.HTTPError as err:
        last_err = err
        # Refresh aged OAuth2 token once immediately on HTTP 401 without sleeping.
        if err.code == 401 and not refreshed_on_401 and attempt + 1 < self._max_retries:
          refreshed_on_401 = True
          self._auth.invalidate_token()
          bearer_token = self._auth.get_bearer_token()
          continue
        # Fail closed immediately on non-retryable 4xx errors or final retry.
        if err.code not in _RETRYABLE_HTTP_CODES or attempt + 1 >= self._max_retries:
          _LOGGER.warning("Vertex batch %s failed with HTTP %d.", method, err.code)
          raise LLMAdapterTransportError(
              f"Vertex batch {method} failed (HTTP {err.code})."
          ) from err
      except (
          urllib.error.URLError,
          http.client.HTTPException,
          OSError,
          TimeoutError,
          json.JSONDecodeError,
          UnicodeDecodeError,
      ) as err:
        # Wrap `HTTPException` (`BadStatusLine`, `IncompleteRead`,
        # `RemoteDisconnected`), mid-stream `OSError`, and decode errors so
        # transient Cloud Storage or Vertex drops retry cleanly.
        last_err = err
        # Surface network/timeout failure once bounded retry budget is exhausted.
        if attempt + 1 >= self._max_retries:
          _LOGGER.warning("Vertex batch %s failed transport.", method)
          raise LLMAdapterTransportError(
              f"Vertex batch {method} failed transport."
          ) from err

      # Sleep with jitter via injectable `self._sleep_fn` for fast offline tests.
      jitter_sec = secrets.randbelow(500) / 1000.0
      self._sleep_fn((2.0**attempt) + jitter_sec)

    # Defensive guard when `_max_retries` loop exits without returning.
    raise LLMAdapterTransportError(f"Vertex batch {method} exhausted retries.") from last_err

  def submit_job(
      self,
      requests: Sequence[LLMCompletionRequest],
      display_name: str | None = None,
  ) -> BatchJobManifest:
    """Uploads deduplicated GCS JSONL and submits a Vertex `BatchPredictionJob`.

    Args:
      requests: Non-empty sequence of `LLMCompletionRequest` items to execute.
      display_name: Optional human-readable Vertex AI job display name.

    Returns:
      Secret-free `BatchJobManifest` suitable for `save_json` or `await_job`.

    Raises:
      LLMAdapterTransportError: If `requests` is empty or GCS/Vertex calls fail.
    """
    # Reject empty batches up front because Vertex rejects 0-line JSONL inputs.
    if not requests:
      raise LLMAdapterTransportError("Cannot submit an empty BatchPredictionJob.")

    # Deduplicate prompts by SHA-256 and build newline-delimited JSONL payload.
    jsonl_bytes, entries = serialize_batch_requests_jsonl(requests, self.thinking_budget)
    # Construct collision-free UTC run slug for isolated GCS input/output paths.
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    ts_slug = now_utc.strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts_slug}-{secrets.token_hex(4)}"

    input_obj = f"{self.gcs_prefix}/{run_id}/input.jsonl"
    gcs_input_uri = f"gs://{self.gcs_bucket}/{input_obj}"
    gcs_output_prefix = f"gs://{self.gcs_bucket}/{self.gcs_prefix}/{run_id}/output/"

    # 1. Upload JSONL via GCS Simple Media Upload with `%2F`-encoded name.
    upload_url = build_gcs_upload_url(self.gcs_bucket, input_obj)
    self._request_with_retry(
        "POST", upload_url, body=jsonl_bytes, content_type="application/jsonl; charset=utf-8"
    )

    # 2. Submit regional `batchPredictionJobs.create` payload for Gemini.
    job_display = display_name or f"argus-sql-batch-{run_id}"
    create_url = build_vertex_batch_jobs_url(self.project_id, self.location)
    job_payload: dict[str, object] = {
        "displayName": job_display,
        "model": f"publishers/google/models/{self.model_name}",
        "inputConfig": {
            "instancesFormat": "jsonl",
            "gcsSource": {"uris": [gcs_input_uri]},
        },
        "outputConfig": {
            "predictionsFormat": "jsonl",
            "gcsDestination": {"outputUriPrefix": gcs_output_prefix},
        },
    }
    raw_resp = self._request_with_retry(
        "POST",
        create_url,
        body=json.dumps(job_payload).encode("utf-8"),
        content_type="application/json",
    )
    # Parse and validate server-returned `BatchPredictionJob` resource JSON.
    try:
      job_json = json.loads(raw_resp.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
      raise LLMAdapterTransportError(
          "Vertex batchPredictionJobs.create returned invalid JSON."
      ) from err
    if not isinstance(job_json, dict):
      raise LLMAdapterTransportError("Malformed BatchPredictionJob response.")

    # Validate canonical job name (accepting numeric GCP project IDs).
    job_name, job_loc = validate_batch_job_name(
        _opt_str(job_json, "name"), self.project_id, self.location
    )
    if job_loc != self.location:
      raise LLMAdapterAuthError(
          "BatchPredictionJob response location mismatches client region."
      )
    # Log batch submission telemetry using lazy-% formatting without secrets.
    _LOGGER.info(
        "Submitted Vertex BatchPredictionJob %s (%d total, %d unique).",
        job_name,
        len(entries),
        len({d for _, d, _ in entries}),
    )
    return BatchJobManifest(
        job_name=job_name,
        project_id=self.project_id,
        location=self.location,
        model_name=self.model_name,
        gcs_input_uri=gcs_input_uri,
        gcs_output_uri_prefix=gcs_output_prefix,
        entries=entries,
        created_at_utc=now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

  def poll_job(self, job_name: str) -> BatchJobStatus:
    """Queries current status of a Vertex `BatchPredictionJob` without blocking.

    Args:
      job_name: Full `projects/.../batchPredictionJobs/...` name or bare job ID.

    Returns:
      Immutable `BatchJobStatus` snapshot.

    Raises:
      LLMAdapterAuthError: If `job_name` fails resource validation.
      LLMAdapterTransportError: If the GET call or JSON decoding fails.
    """
    # Validate job resource path and extract regional Vertex AI host prefix.
    canonical_name, job_loc = validate_batch_job_name(
        job_name, self.project_id, self.location
    )
    get_url = f"https://{job_loc}-aiplatform.googleapis.com/v1/{canonical_name}"
    raw_resp = self._request_with_retry("GET", get_url)
    # Decode Vertex AI `BatchPredictionJob` JSON payload safely.
    try:
      job_json = json.loads(raw_resp.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
      raise LLMAdapterTransportError(
          "Vertex batchPredictionJobs.get returned invalid JSON."
      ) from err
    if not isinstance(job_json, dict):
      raise LLMAdapterTransportError("Malformed BatchPredictionJob status JSON.")

    # Extract state, GCS output directory, completion counts, and error text.
    state = (
        _opt_str(job_json, "state", "JOB_STATE_UNSPECIFIED")
        or "JOB_STATE_UNSPECIFIED"
    )
    gcs_out = _opt_str(_opt_dict(job_json, "outputInfo"), "gcsOutputDirectory")
    stats_dict = _opt_dict(job_json, "completionStats")
    # Truncate upstream error detail to 240 chars so diagnostics stay bounded.
    err_msg = _opt_str(_opt_dict(job_json, "error"), "message")[:240]
    return BatchJobStatus(
        job_name=canonical_name,
        state=state,
        gcs_output_dir=gcs_out,
        error_message=err_msg,
        successful_count=coerce_nonneg_int(stats_dict.get("successfulCount"), 0),
        failed_count=coerce_nonneg_int(stats_dict.get("failedCount"), 0),
        incomplete_count=coerce_nonneg_int(stats_dict.get("incompleteCount"), 0),
    )

  def await_job(
      self,
      job_name: str,
      poll_interval_sec: float | None = None,
      timeout_sec: float | None = None,
  ) -> BatchJobStatus:
    """Polls `job_name` until completion, failing closed on error or timeout.

    Args:
      job_name: Full `projects/.../batchPredictionJobs/...` name or bare job ID.
      poll_interval_sec: Optional override for seconds between status polls.
      timeout_sec: Optional override for maximum wait time in seconds.

    Returns:
      Terminal successful `BatchJobStatus`.

    Raises:
      LLMAdapterTransportError: If the job enters `FAILED`/`CANCELLED`/`EXPIRED`
        or exceeds `timeout_sec`.
    """
    # Resolve bounded polling interval and total deadline in seconds.
    interval = (
        max(0.01, float(poll_interval_sec))
        if poll_interval_sec is not None
        else self._poll_interval_sec
    )
    deadline_sec = (
        max(interval, float(timeout_sec))
        if timeout_sec is not None
        else self._poll_timeout_sec
    )
    # Track both real monotonic elapsed time and simulated `sleep_fn` steps.
    start_mono = time.monotonic()
    waited_sec = 0.0

    # Delegate HTTP polling to `self.poll_job` so no raw HTTP call sits in loop.
    while True:
      status = self.poll_job(job_name)
      # Return immediately once Vertex AI reports terminal job completion.
      if status.state in _TERMINAL_SUCCESS_STATES:
        return status
      # Fail closed immediately on FAILED, CANCELLED, or EXPIRED states.
      if status.state in _TERMINAL_FAILURE_STATES:
        detail = f": {status.error_message}" if status.error_message else ""
        raise LLMAdapterTransportError(
            f"BatchPredictionJob '{status.job_name}' ended in {status.state}{detail}."
        )

      # Enforce deadline in both live wall-clock runs and no-op `sleep_fn` tests.
      elapsed_real = time.monotonic() - start_mono
      if elapsed_real >= deadline_sec or waited_sec >= deadline_sec:
        raise LLMAdapterTransportError(
            f"BatchPredictionJob '{status.job_name}' timed out in {status.state}."
        )
      step = min(interval, max(0.001, deadline_sec - waited_sec))
      self._sleep_fn(step)
      waited_sec += step

  def _list_output_jsonl_objects(
      self,
      bucket: str,
      dir_prefix: str,
  ) -> tuple[str, ...]:
    """Lists `.jsonl` prediction shard object names under `dir_prefix`."""
    # Ensure trailing slash on directory prefix so sibling prefixes never match.
    norm_prefix = f"{dir_prefix.strip('/')}/" if dir_prefix.strip("/") else ""
    shard_names: list[str] = []
    page_token: str | None = None

    # Paginate GCS `objects.list` via `self._request_with_retry` safely.
    while True:
      list_url = build_gcs_list_url(bucket, norm_prefix, page_token=page_token)
      raw_bytes = self._request_with_retry("GET", list_url)
      # Validate GCS object listing payload structure before reading `items`.
      try:
        listing = json.loads(raw_bytes.decode("utf-8"))
      except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise LLMAdapterTransportError("Invalid GCS object list JSON.") from err
      if not isinstance(listing, dict):
        raise LLMAdapterTransportError("Malformed GCS object list payload.")

      # Retain only `.jsonl` prediction shards, skipping summary/error sidecars.
      items = [] if listing.get("items") is None else listing.get("items")
      if not isinstance(items, list):
        raise LLMAdapterTransportError("Malformed GCS object items list.")
      for obj in items:
        if isinstance(obj, dict):
          obj_name = _opt_str(obj, "name")
          if obj_name.endswith(".jsonl"):
            shard_names.append(obj_name)

      # Continue paginating only when GCS returns a non-empty `nextPageToken`.
      next_tok = _opt_str(listing, "nextPageToken").strip()
      if not next_tok:
        break
      page_token = next_tok

    # Sort deduplicated shard names for deterministic multi-shard processing.
    return tuple(sorted(set(shard_names)))

  def _download_gcs_shard(self, target: tuple[str, str]) -> str:
    """Downloads one `.jsonl` shard from GCS using `%2F`-encoded `alt=media`."""
    bucket, obj_name = target
    # Build `%2F`-encoded media URL via `build_gcs_download_url` (`safe=""`).
    dl_url = build_gcs_download_url(bucket, obj_name.lstrip("/"))
    raw_bytes = self._request_with_retry("GET", dl_url)
    # Wrap malformed UTF-8 shard bytes in `LLMAdapterTransportError` cleanly.
    try:
      return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as err:
      raise LLMAdapterTransportError(
          "Invalid UTF-8 in GCS prediction shard."
      ) from err

  def collect_job_responses(
      self,
      manifest: BatchJobManifest,
      output_dir: str | None = None,
  ) -> tuple[LLMCompletionResponse, ...]:
    """Downloads unordered GCS `.jsonl` shards and aligns responses 1-to-1.

    Args:
      manifest: `BatchJobManifest` produced by `submit_job` or `load_json`.
      output_dir: Optional explicit `gs://...` output directory override; when
        `None`, queries `poll_job(manifest.job_name)` for `gcs_output_dir`.

    Returns:
      Tuple of `LLMCompletionResponse` items aligned 1-to-1 with `manifest`.

    Raises:
      LLMAdapterAuthError: If GCS output URI validation fails.
      LLMAdapterTransportError: If the job has not succeeded, shards are empty,
        any row reports an error status, or any expected digest is missing.
    """
    # Query `poll_job` when caller did not pass a pre-fetched GCS output URI.
    effective_out_dir = output_dir or ""
    if not effective_out_dir:
      status = self.poll_job(manifest.job_name)
      if status.state in _TERMINAL_FAILURE_STATES:
        detail = f": {status.error_message}" if status.error_message else ""
        raise LLMAdapterTransportError(
            f"BatchPredictionJob '{manifest.job_name}' failed ({status.state}){detail}."
        )
      if status.state not in _TERMINAL_SUCCESS_STATES:
        raise LLMAdapterTransportError(
            f"BatchPredictionJob '{manifest.job_name}' not ready ({status.state})."
        )
      effective_out_dir = status.gcs_output_dir

    # Fail closed if Vertex AI omitted `outputInfo.gcsOutputDirectory`.
    if not effective_out_dir:
      raise LLMAdapterTransportError("BatchPredictionJob lacked gcsOutputDirectory.")

    # Validate output bucket/prefix and enumerate `.jsonl` prediction shards.
    out_bucket, out_prefix = parse_and_validate_gcs_uri(effective_out_dir)
    shard_names = self._list_output_jsonl_objects(out_bucket, out_prefix)
    if not shard_names:
      raise LLMAdapterTransportError("No prediction .jsonl shards found in GCS.")

    # Warm OAuth2 token once, then download all `.jsonl` shards concurrently.
    self._auth.get_bearer_token()
    targets = [(out_bucket, name) for name in shard_names]
    workers = min(self.max_concurrency, len(targets))
    with ThreadPoolExecutor(max_workers=workers) as pool:
      shard_texts = list(pool.map(self._download_gcs_shard, targets))

    # Build lookup maps for O(1) `labels.argus_hash` and canonical fallback.
    digest_by_label: dict[str, str] = {}
    char_len_by_digest: dict[str, int] = {}
    for _, digest, char_len in manifest.entries:
      digest_by_label[digest[:32]] = digest
      char_len_by_digest[digest] = char_len
    expected_digests = set(char_len_by_digest.keys())

    # Parse unordered JSONL lines; name dict `echoed_req` (never `request`!).
    results_by_digest: dict[str, tuple[str, int, int]] = {}
    for shard_text in shard_texts:
      for raw_line in shard_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
          continue
        try:
          row = json.loads(stripped)
        except json.JSONDecodeError as err:
          raise LLMAdapterTransportError("Malformed JSONL prediction line.") from err
        if not isinstance(row, dict):
          raise LLMAdapterTransportError("Prediction JSONL row must be an object.")

        # 1. Primary correlation via 32-char `labels.argus_hash` prefix.
        # 2. Fallback correlation via canonical echoed `GenerateContentRequest`.
        echoed_req = _opt_dict(row, "request")
        matched_digest = ""
        if echoed_req:
          labels_map = _opt_dict(echoed_req, "labels")
          lbl_val = _opt_str(labels_map, "argus_hash").strip()
          matched_digest = digest_by_label.get(lbl_val, "")
          if not matched_digest:
            recomputed = _batch_request_digest(echoed_req)
            if recomputed in expected_digests:
              matched_digest = recomputed

        # 3. Optional fallback for synthetic fixtures carrying `custom_id`/`key`.
        if not matched_digest:
          custom_key = (_opt_str(row, "custom_id") or _opt_str(row, "key")).strip()
          if custom_key in expected_digests:
            matched_digest = custom_key
          elif custom_key in digest_by_label:
            matched_digest = digest_by_label[custom_key]

        # Reject uncorrelated rows so responses can never silently misalign.
        if not matched_digest:
          raise LLMAdapterTransportError(
              "Unable to correlate batch prediction JSONL row to manifest digest."
          )

        # Fail closed on per-row Vertex error status or missing `response` body.
        row_status = row.get("status", "")
        resp_obj = row.get("response")
        if row_status or not isinstance(resp_obj, dict):
          raise LLMAdapterTransportError(
              f"Batch prediction row {matched_digest[:12]} failed (status={row_status!r})."
          )

        # Filter `thought=True` parts and sum candidate + thought tokens.
        results_by_digest[matched_digest] = extract_gemini_completion_triple(
            resp_obj, char_len_by_digest[matched_digest]
        )

    # Verify 100% of unique manifest digests received valid completions.
    missing = expected_digests - results_by_digest.keys()
    if missing:
      raise LLMAdapterTransportError(
          f"BatchPredictionJob missing {len(missing)} of {len(expected_digests)} digests."
      )

    # Hydrate in-memory cross-call `_cache` so follow-up calls reuse completions.
    with self._cache_lock:
      for digest, triple in results_by_digest.items():
        self._cache[digest] = triple

    # Reconstruct 1-to-1 responses, marking intra-batch duplicates `from_cache`.
    seen_in_manifest: set[str] = set()
    responses: list[LLMCompletionResponse] = []
    for req_id, digest, _ in manifest.entries:
      raw_text, p_tok, c_tok = results_by_digest[digest]
      is_dup = digest in seen_in_manifest
      seen_in_manifest.add(digest)
      responses.append(
          LLMCompletionResponse(
              request_id=req_id,
              sql_text=extract_sql_from_completion(raw_text),
              raw_text=raw_text,
              prompt_tokens=0 if is_dup else p_tok,
              completion_tokens=0 if is_dup else c_tok,
              from_cache=is_dup,
              cached_prompt_tokens=p_tok if is_dup else 0,
              cached_completion_tokens=c_tok if is_dup else 0,
          )
      )
    return tuple(responses)

  def batch_complete(
      self,
      requests: Sequence[LLMCompletionRequest],
  ) -> tuple[LLMCompletionResponse, ...]:
    """Executes `requests` via GCS + `BatchPredictionJob` (`BatchLLMPort`).

    Args:
      requests: Sequence of `LLMCompletionRequest` items to complete.

    Returns:
      Tuple of `LLMCompletionResponse` items aligned 1-to-1 with `requests`.
    """
    # Short-circuit empty request sequences with zero network or GCS traffic.
    if not requests:
      return ()

    # Partition requests against cross-call `_cache` before launching any job.
    req_digests: list[str] = []
    cached_before: set[str] = set()
    pending_unique: list[LLMCompletionRequest] = []
    pending_seen: set[str] = set()

    with self._cache_lock:
      for req in requests:
        digest = _batch_request_digest(
            _build_batch_request_dict(req, self.thinking_budget)
        )
        req_digests.append(digest)
        if digest in self._cache:
          cached_before.add(digest)
        elif digest not in pending_seen:
          pending_seen.add(digest)
          pending_unique.append(req)

    # Submit, await, and collect a regional batch job only for uncached digests.
    if pending_unique:
      manifest = self.submit_job(pending_unique)
      done_status = self.await_job(manifest.job_name)
      self.collect_job_responses(
          manifest, output_dir=done_status.gcs_output_dir or None
      )

    # Align responses 1-to-1; mark cross-call and intra-batch duplicates cached.
    billed_in_call: set[str] = set()
    responses: list[LLMCompletionResponse] = []
    with self._cache_lock:
      for req, digest in zip(requests, req_digests, strict=True):
        raw_text, p_tok, c_tok = self._cache[digest]
        is_hit = (digest in cached_before) or (digest in billed_in_call)
        billed_in_call.add(digest)
        responses.append(
            LLMCompletionResponse(
                request_id=req.request_id,
                sql_text=extract_sql_from_completion(raw_text),
                raw_text=raw_text,
                prompt_tokens=0 if is_hit else p_tok,
                completion_tokens=0 if is_hit else c_tok,
                from_cache=is_hit,
                cached_prompt_tokens=p_tok if is_hit else 0,
                cached_completion_tokens=c_tok if is_hit else 0,
            )
        )
    return tuple(responses)
