"""Vertex AI / GCS URL validation and OAuth2 ADC token resolution.

Isolates allow-list validation for Google Cloud project/location/model/cache/job
identifiers, Cloud Storage JSON API v1 URLs, and thread-safe in-memory OAuth2
Application Default Credentials (ADC) token refresh with zero third-party SDKs.
"""

from collections.abc import Callable
import http.client
import json
import math
import os
import pathlib
import re
import shutil
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from argus_sql.adapters.exceptions import LLMAdapterAuthError

# Strict allow-list patterns preventing SSRF or URL path traversal injection.
# `_GCP_LOCATION_RE` forbids leading/trailing hyphens to protect RFC hostnames.
# `_GCP_REGION_RE` requires at least one hyphen so `'global'` cannot pass as regional.
# `_GEMINI_MODEL_RE` blocks `.`, `..`, and multi-dot traversal slugs.
# `_GEMINI_MODEL_RESOURCE_RE` allow-lists only canonical Gemini model prefixes.
_GCP_PROJECT_RE = re.compile(r"^[a-z][a-z0-9-]{4,61}[a-z0-9]$")
_GCP_LOCATION_RE = re.compile(r"^[a-z][a-z0-9-]*[a-z0-9]$")
_GCP_REGION_RE = re.compile(r"^[a-z]+-[a-z0-9-]*[a-z0-9]$")
_GEMINI_MODEL_RE = re.compile(r"^(?!\.\.?$)(?!.*\.\.)[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_GEMINI_MODEL_RESOURCE_RE = re.compile(
    r"^(?:"
    r"(?:projects/(?:[a-z][a-z0-9-]{4,61}[a-z0-9]|[0-9]{1,25})"
    r"/locations/[a-z][a-z0-9-]*[a-z0-9]/)?"
    r"publishers/google/models/"
    r"|models/"
    r"|google/"
    r")?(?P<slug>(?!\.\.?$)(?!.*\.\.)[A-Za-z0-9][A-Za-z0-9._-]{0,127})$"
)
_GCS_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$")

# Resource name patterns accepting canonicalized numeric GCP project numbers
# (e.g., `projects/123456789012/locations/...`) returned by Vertex AI responses.
# Both patterns enforce edge-hyphen-free locations and `..`-free resource IDs.
_CACHED_CONTENT_NAME_RE = re.compile(
    r"^projects/(?P<project>[a-z][a-z0-9-]{4,61}[a-z0-9]|[0-9]{1,25})"
    r"/locations/(?P<location>[a-z][a-z0-9-]*[a-z0-9])"
    r"/cachedContents/(?P<cache_id>(?!\.\.?$)(?!.*\.\.)[A-Za-z0-9._-]{1,128})$"
)
_BATCH_JOB_NAME_RE = re.compile(
    r"^projects/(?P<project>[a-z][a-z0-9-]{4,61}[a-z0-9]|[0-9]{1,25})"
    r"/locations/(?P<location>[a-z]+-[a-z0-9-]*[a-z0-9])"
    r"/batchPredictionJobs/(?P<job_id>(?!\.\.?$)(?!.*\.\.)[A-Za-z0-9._-]{1,128})$"
)

# Default regional fallback because `batchPredictionJobs` rejects `global`.
_DEFAULT_BATCH_REGION = "us-central1"

# Public Google OAuth 2.0 token endpoint for standard ADC refresh tokens.
_GOOGLE_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"


def validate_gcp_project_id(project_id: str) -> str:
  """Validates a Google Cloud project identifier against strict allow-list rules.

  Args:
    project_id: Candidate Google Cloud project ID string.

  Returns:
    Trimmed, validated project ID string.

  Raises:
    LLMAdapterAuthError: If `project_id` is empty or violates GCP naming syntax.
  """
  cleaned = project_id.strip()
  # Fail closed on empty, path-traversing, or malformed project identifiers.
  if not _GCP_PROJECT_RE.match(cleaned):
    raise LLMAdapterAuthError(
        "Invalid or missing Google Cloud project ID for Vertex AI."
    )
  return cleaned


def validate_gcp_location(location: str) -> str:
  """Validates a Vertex AI region or `'global'` routing identifier.

  Args:
    location: Candidate Vertex AI location (e.g., `'global'`, `'us-central1'`).

  Returns:
    Normalized lowercase Vertex AI location string.

  Raises:
    LLMAdapterAuthError: If `location` contains non-ASCII characters, invalid
      URL/host symbols, or leading/trailing hyphens.
  """
  # Reject non-ASCII glyphs before `.lower()` so Unicode case-folding (such as
  # Kelvin sign `\u212a` -> `k`) cannot bypass ASCII hostname checks.
  if not location.isascii():
    raise LLMAdapterAuthError("Invalid Vertex AI location identifier.")
  cleaned = location.strip().lower()
  # Restrict location strictly to `[a-z][a-z0-9-]*[a-z0-9]` (no edge hyphens).
  if not cleaned or not _GCP_LOCATION_RE.match(cleaned):
    raise LLMAdapterAuthError("Invalid Vertex AI location identifier.")
  return cleaned


def normalize_gemini_model_name(model_name: str) -> str:
  """Normalizes and validates a public Vertex AI Gemini model identifier.

  Accepts only a bare model slug, `models/<slug>`, `google/<slug>`,
  `publishers/google/models/<slug>`, or a canonical Vertex AI resource path
  `projects/<project_or_num>/locations/<location>/publishers/google/models/<slug>`.

  Args:
    model_name: Raw model identifier or canonical publisher model resource path.

  Returns:
    Validated bare Gemini model slug (e.g., `'gemini-3.7-flash'`).

  Raises:
    LLMAdapterAuthError: If `model_name` carries an untrusted prefix, path
      traversal segments, or unsafe characters.
  """
  # Reject non-ASCII characters up front before matching resource structure.
  if not model_name.isascii():
    raise LLMAdapterAuthError("Invalid Vertex AI Gemini model identifier.")
  cleaned = model_name.strip()
  # Validate the entire prefix + slug against `_GEMINI_MODEL_RESOURCE_RE` so
  # arbitrary traversal or URL prefixes before `/models/` fail closed.
  match = _GEMINI_MODEL_RESOURCE_RE.match(cleaned)
  if match is None:
    raise LLMAdapterAuthError("Invalid Vertex AI Gemini model identifier.")
  slug = match.group("slug")
  # Re-check extracted bare slug against `_GEMINI_MODEL_RE` as defense in depth.
  if not _GEMINI_MODEL_RE.match(slug):
    raise LLMAdapterAuthError("Invalid Vertex AI Gemini model identifier.")
  return slug


def _vertex_host_for_location(validated_location: str) -> str:
  """Returns the Vertex AI HTTPS hostname for a validated location slug."""
  # Global Gemini Enterprise routing targets `aiplatform.googleapis.com` directly.
  if validated_location == "global":
    return "aiplatform.googleapis.com"
  return f"{validated_location}-aiplatform.googleapis.com"


def build_vertex_publisher_model_path(
    project_id: str,
    location: str,
    model_name: str,
) -> str:
  """Builds `projects/{p}/locations/{l}/publishers/google/models/{m}` resource.

  Args:
    project_id: Google Cloud project ID billed for Vertex AI.
    location: Vertex AI location (`'global'` or regional slug).
    model_name: Public Gemini model identifier.

  Returns:
    Validated Vertex AI publisher model resource path string.
  """
  # Validate every segment before interpolating the immutable model resource URI.
  proj = validate_gcp_project_id(project_id)
  loc = validate_gcp_location(location)
  model = normalize_gemini_model_name(model_name)
  return f"projects/{proj}/locations/{loc}/publishers/google/models/{model}"


def build_vertex_generate_url(
    project_id: str,
    location: str,
    model_name: str,
) -> str:
  """Constructs a validated HTTPS Vertex AI `generateContent` endpoint URL.

  Args:
    project_id: Google Cloud project ID billed for Vertex AI / Gemini Enterprise.
    location: Vertex AI location (`'global'` or regional slug like `'us-central1'`).
    model_name: Public Gemini model identifier.

  Returns:
    Canonical `https://...aiplatform.googleapis.com/v1beta1/...:generateContent` URL.
  """
  # Reuse publisher model path builder so `generateContent` and `cachedContents` match.
  loc = validate_gcp_location(location)
  model_path = build_vertex_publisher_model_path(project_id, loc, model_name)
  host = _vertex_host_for_location(loc)
  return f"https://{host}/v1beta1/{model_path}:generateContent"


def validate_cached_content_name(cache_name: str) -> str:
  """Validates a Vertex AI `cachedContents` resource path (project slug or num).

  Args:
    cache_name: Resource path `projects/{id_or_num}/locations/{loc}/cachedContents/{id}`.

  Returns:
    Trimmed, validated `cachedContents` resource path string.

  Raises:
    LLMAdapterAuthError: If `cache_name` violates Vertex AI resource syntax.
  """
  cleaned = cache_name.strip()
  # Accept both project IDs and server-canonicalized numeric GCP project numbers.
  if not _CACHED_CONTENT_NAME_RE.match(cleaned):
    raise LLMAdapterAuthError("Invalid Vertex AI cachedContent resource name.")
  return cleaned


# Alias supporting callers importing `validate_vertex_cached_content_name`.
validate_vertex_cached_content_name = validate_cached_content_name


def build_vertex_cached_contents_url(project_id: str, location: str) -> str:
  """Constructs the HTTPS collection URL for `cachedContents.create`.

  Args:
    project_id: Google Cloud project ID billed for context caching.
    location: Vertex AI location (`'global'` or regional slug).

  Returns:
    Validated `https://...aiplatform.googleapis.com/v1beta1/.../cachedContents` URL.
  """
  # Target `/v1beta1` to match `build_vertex_generate_url` schema parity.
  proj = validate_gcp_project_id(project_id)
  loc = validate_gcp_location(location)
  host = _vertex_host_for_location(loc)
  return f"https://{host}/v1beta1/projects/{proj}/locations/{loc}/cachedContents"


def build_vertex_cached_content_item_url(location: str, cache_name: str) -> str:
  """Constructs the HTTPS item URL for `cachedContents.get` or `.delete`.

  Args:
    location: Client Vertex AI location (`'global'` or regional slug).
    cache_name: Server-returned `projects/{id_or_num}/locations/{loc}/cachedContents/{id}`.

  Returns:
    Validated `https://...aiplatform.googleapis.com/v1beta1/{cache_name}` URL.

  Raises:
    LLMAdapterAuthError: If `cache_name` is invalid or mismatches `location`.
  """
  loc = validate_gcp_location(location)
  canonical = validate_cached_content_name(cache_name)
  match = _CACHED_CONTENT_NAME_RE.match(canonical)
  # Prevent cross-region host/path confusion if a mismatched resource is passed.
  if match is None or match.group("location") != loc:
    raise LLMAdapterAuthError(
        "CachedContent resource location does not match client location."
    )
  host = _vertex_host_for_location(loc)
  return f"https://{host}/v1beta1/{canonical}"


def resolve_vertex_batch_location(
    location: str = "global",
    default_region: str = _DEFAULT_BATCH_REGION,
) -> str:
  """Resolves a strictly regional Vertex AI location for `batchPredictionJobs`.

  Args:
    location: Requested Vertex location (`'global'` maps to `default_region`).
    default_region: Fallback regional slug when `location` is `'global'`.

  Returns:
    Validated hyphenated GCP region string (e.g., `'us-central1'`).

  Raises:
    LLMAdapterAuthError: If the resolved location is not a valid GCP region.
  """
  loc = validate_gcp_location(location)
  # Map `'global'` automatically because `batchPredictionJobs` is regional-only.
  if loc == "global":
    loc = validate_gcp_location(default_region)
  # Reject single-token non-regional identifiers (such as `'us'` or `'eu'`).
  if not _GCP_REGION_RE.match(loc):
    raise LLMAdapterAuthError(
        f"Vertex AI batchPredictionJobs requires a regional location, got '{loc}'."
    )
  return loc


def validate_batch_job_name(job_name: str) -> str:
  """Validates a regional Vertex AI `batchPredictionJobs` resource path.

  Args:
    job_name: Resource path `projects/{id_or_num}/locations/{reg}/batchPredictionJobs/{id}`.

  Returns:
    Trimmed, validated `batchPredictionJobs` resource path string.

  Raises:
    LLMAdapterAuthError: If `job_name` is non-regional or contains unsafe chars.
  """
  cleaned = job_name.strip()
  # Accept canonicalized numeric GCP project IDs while enforcing regional routing.
  if not _BATCH_JOB_NAME_RE.match(cleaned):
    raise LLMAdapterAuthError(
        "Invalid Vertex AI batchPredictionJobs resource name."
    )
  return cleaned


def build_vertex_batch_jobs_url(
    project_id: str,
    location: str = _DEFAULT_BATCH_REGION,
) -> str:
  """Constructs the regional HTTPS collection URL for `batchPredictionJobs.create`.

  Args:
    project_id: Google Cloud project ID billed for offline batch prediction.
    location: Regional slug (or `'global'`, mapped to `'us-central1'`).

  Returns:
    Validated `https://{region}-aiplatform.googleapis.com/v1/.../batchPredictionJobs` URL.
  """
  proj = validate_gcp_project_id(project_id)
  reg = resolve_vertex_batch_location(location)
  # Vertex AI `batchPredictionJobs` uses the GA `/v1/` regional endpoint.
  return (
      f"https://{reg}-aiplatform.googleapis.com/v1/projects/{proj}"
      f"/locations/{reg}/batchPredictionJobs"
  )


def build_vertex_batch_job_item_url(location: str, job_name: str) -> str:
  """Constructs the regional HTTPS item URL for `batchPredictionJobs.get`.

  Args:
    location: Client location (`'global'` maps to `'us-central1'`, or regional).
    job_name: Server-returned `projects/{id_or_num}/locations/{reg}/batchPredictionJobs/{id}`.

  Returns:
    Validated `https://{region}-aiplatform.googleapis.com/v1/{job_name}` URL.

  Raises:
    LLMAdapterAuthError: If `job_name` is malformed or mismatches resolved region.
  """
  reg = resolve_vertex_batch_location(location)
  canonical = validate_batch_job_name(job_name)
  match = _BATCH_JOB_NAME_RE.match(canonical)
  # Require `job_name` region segment to match the resolved HTTPS host region.
  if match is None or match.group("location") != reg:
    raise LLMAdapterAuthError(
        "BatchPredictionJob location does not match resolved batch region."
    )
  return f"https://{reg}-aiplatform.googleapis.com/v1/{canonical}"


def _validate_gcs_bucket(bucket: str) -> str:
  """Validates a Cloud Storage bucket identifier against traversal and SSRF."""
  cleaned = bucket.strip()
  # Reject uppercase, empty, or double-dot traversal bucket substrings.
  if not _GCS_BUCKET_RE.match(cleaned) or ".." in cleaned:
    raise LLMAdapterAuthError("Invalid Google Cloud Storage bucket name.")
  return cleaned


def _validate_gcs_object_path(
    object_name: str,
    *,
    allow_empty: bool = False,
    allow_trailing_slash: bool = True,
) -> str:
  """Validates a GCS object name or prefix against traversal and encoding tricks."""
  cleaned = object_name.strip()
  # Allow empty prefix only when listing or resolving bucket-root URIs explicitly.
  if not cleaned:
    if allow_empty:
      return ""
    raise LLMAdapterAuthError("Cloud Storage object path must not be empty.")
  # Reject trailing `/` on concrete media upload/download object keys.
  if not allow_trailing_slash and cleaned.endswith("/"):
    raise LLMAdapterAuthError(
        "Cloud Storage media object key must not end with '/'."
    )
  # Reject leading `/`, empty `//` segments, raw `%` escapes, `\`, `?`, `#`, & CTLs.
  # Blocking `%` prevents single/double percent-encoded traversal (`%2e%2e`,
  # `..%2f`, `%252e%252e`, `%5c`, `%00`) and avoids double-encoding in REST URLs.
  if (
      cleaned.startswith("/")
      or "//" in cleaned
      or any(ch in cleaned for ch in ("%", "\\", "?", "#"))
      or any(ord(ch) < 32 or 127 <= ord(ch) <= 159 for ch in cleaned)
  ):
    raise LLMAdapterAuthError("Unsafe characters in Cloud Storage object path.")
  # Reject dot path segments (`.` or `..`) across all `/` boundaries.
  if any(seg in {".", ".."} for seg in cleaned.split("/")):
    raise LLMAdapterAuthError("Path traversal forbidden in Cloud Storage URI.")
  return cleaned


def parse_gcs_uri(
    gcs_uri: str,
    allow_empty_object: bool = False,
) -> tuple[str, str]:
  """Parses and validates a `gs://<bucket>/<object_path>` Cloud Storage URI.

  Args:
    gcs_uri: Cloud Storage URI starting with `'gs://'`.
    allow_empty_object: When True, permits bucket-only `'gs://bucket'` prefixes.

  Returns:
    Tuple `(validated_bucket, validated_object_path)`.

  Raises:
    LLMAdapterAuthError: If `gcs_uri` is not a valid, traversal-free `gs://` URI.
  """
  cleaned = gcs_uri.strip()
  # Require explicit `gs://` scheme so `http://` or `s3://` URIs fail closed.
  if not cleaned.startswith("gs://"):
    raise LLMAdapterAuthError("Cloud Storage URI must start with 'gs://'.")
  remainder = cleaned[len("gs://") :]
  # Split strictly on the first `/` into bucket and object path components.
  if "/" in remainder:
    raw_bucket, raw_obj = remainder.split("/", 1)
  else:
    raw_bucket, raw_obj = remainder, ""
  # Reject whitespace adjacent to `/` boundaries inside the `gs://` URI body.
  if raw_bucket != raw_bucket.strip() or raw_obj != raw_obj.strip():
    raise LLMAdapterAuthError("Unexpected whitespace in Cloud Storage URI.")
  # Validate bucket slug and traversal-free object path before returning tuple.
  bucket = _validate_gcs_bucket(raw_bucket)
  obj_path = _validate_gcs_object_path(
      raw_obj,
      allow_empty=allow_empty_object,
      allow_trailing_slash=True,
  )
  return (bucket, obj_path)


def build_gcs_upload_url(bucket: str, object_name: str) -> str:
  """Constructs a GCS JSON API v1 simple media upload (`uploadType=media`) URL.

  Args:
    bucket: Target Cloud Storage bucket name.
    object_name: Destination object key within `bucket`.

  Returns:
    Validated `https://storage.googleapis.com/upload/storage/v1/b/.../o?...` URL.

  Raises:
    LLMAdapterAuthError: If `bucket` or `object_name` fails allow-list checks.
  """
  bkt = _validate_gcs_bucket(bucket)
  # Require a concrete non-empty object key without a trailing directory slash.
  obj = _validate_gcs_object_path(
      object_name,
      allow_empty=False,
      allow_trailing_slash=False,
  )
  # Percent-encode slashes and colons in `name=` query parameter via `safe=''`.
  encoded_name = urllib.parse.quote(obj, safe="")
  return (
      f"https://storage.googleapis.com/upload/storage/v1/b/{bkt}/o"
      f"?uploadType=media&name={encoded_name}"
  )


def build_gcs_list_url(
    bucket: str,
    prefix: str,
    page_token: str | None = None,
) -> str:
  """Constructs a GCS JSON API v1 object list URL filtered by `prefix`.

  Args:
    bucket: Cloud Storage bucket name to list.
    prefix: Object key prefix (e.g., `outputInfo.gcsOutputDirectory` path).
    page_token: Optional pagination token from a prior `nextPageToken` response.

  Returns:
    Validated `https://storage.googleapis.com/storage/v1/b/.../o?prefix=...` URL.

  Raises:
    LLMAdapterAuthError: If `bucket`, `prefix`, or `page_token` is invalid.
  """
  bkt = _validate_gcs_bucket(bucket)
  # Permit empty or trailing-slash directory prefixes for GCS `objects.list`.
  pfx = _validate_gcs_object_path(
      prefix,
      allow_empty=True,
      allow_trailing_slash=True,
  )
  # Encode prefix characters safely so timestamps (`T05:00:00Z`) list cleanly.
  encoded_pfx = urllib.parse.quote(pfx, safe="")
  url = f"https://storage.googleapis.com/storage/v1/b/{bkt}/o?prefix={encoded_pfx}"
  # Append sanitized pagination token when iterating multi-page shard outputs.
  if page_token is not None and page_token.strip():
    tok = page_token.strip()
    if any(ord(ch) < 32 or 127 <= ord(ch) <= 159 for ch in tok):
      raise LLMAdapterAuthError("Invalid control character in GCS pageToken.")
    url += f"&pageToken={urllib.parse.quote(tok, safe='')}"
  return url


def build_gcs_download_url(bucket: str, object_name: str) -> str:
  """Constructs a `%2F`-encoded GCS JSON API v1 media download (`alt=media`) URL.

  Args:
    bucket: Cloud Storage bucket name holding the output shard.
    object_name: Full object path (all `/` slashes encoded as `%2F` via `safe=''`).

  Returns:
    Validated `https://storage.googleapis.com/storage/v1/b/.../o/{pct}?alt=media` URL.

  Raises:
    LLMAdapterAuthError: If `bucket` or `object_name` fails allow-list checks.
  """
  bkt = _validate_gcs_bucket(bucket)
  # Require a concrete non-empty shard path without a trailing directory slash.
  obj = _validate_gcs_object_path(
      object_name,
      allow_empty=False,
      allow_trailing_slash=False,
  )
  # CRITICAL: GCS REST requires `%2F` encoding for `/` inside `/o/{object_name}`.
  encoded_obj = urllib.parse.quote(obj, safe="")
  return f"https://storage.googleapis.com/storage/v1/b/{bkt}/o/{encoded_obj}?alt=media"


class VertexTokenResolver:
  """Thread-safe in-memory OAuth2 token resolver for Vertex AI requests.

  Resolves credentials strictly at runtime from environment bearer variables,
  standard Google Cloud ADC (`authorized_user` refresh token), or `gcloud auth`.
  """

  def __init__(
      self,
      ssl_ctx: ssl.SSLContext,
      adc_file: pathlib.Path | None = None,
      token_supplier: Callable[[], tuple[str, float]] | None = None,
  ) -> None:
    """Initializes the in-memory token cache and synchronization lock.

    Args:
      ssl_ctx: Verified TLS 1.2+ SSLContext for OAuth token exchange.
      adc_file: Optional path override for Google Cloud ADC JSON.
      token_supplier: Optional injectable `(token, ttl_sec)` provider for tests.
    """
    self._ssl_ctx = ssl_ctx
    self._adc_file = adc_file
    self._token_supplier = token_supplier
    # Synchronize refresh calls so concurrent batch threads refresh only once.
    self._lock = threading.Lock()
    self._cached_token = ""
    self._expiry_monotonic = 0.0

  def _resolve_adc_path(self) -> pathlib.Path | None:
    """Locates the standard Google Cloud ADC JSON file across Windows and POSIX."""
    # Prefer test/constructor ADC file injection before scanning environment.
    if self._adc_file is not None:
      return self._adc_file if self._adc_file.is_file() else None
    # Honor explicit `GOOGLE_APPLICATION_CREDENTIALS` before default SDK paths.
    env_adc = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if env_adc:
      candidate = pathlib.Path(env_adc)
      return candidate if candidate.is_file() else None

    # Probe `CLOUDSDK_CONFIG`, Windows `%APPDATA%\gcloud`, and POSIX `~/.config`.
    candidate_dirs: list[pathlib.Path] = []
    cloudsdk_cfg = os.environ.get("CLOUDSDK_CONFIG", "").strip()
    if cloudsdk_cfg:
      candidate_dirs.append(pathlib.Path(cloudsdk_cfg))
    appdata_dir = os.environ.get("APPDATA", "").strip()
    if appdata_dir:
      candidate_dirs.append(pathlib.Path(appdata_dir) / "gcloud")
    home_dir = pathlib.Path.home()
    candidate_dirs.append(home_dir / "AppData" / "Roaming" / "gcloud")
    candidate_dirs.append(home_dir / ".config" / "gcloud")

    for cfg_dir in candidate_dirs:
      adc_file = cfg_dir / "application_default_credentials.json"
      if adc_file.is_file():
        return adc_file
    return None

  def _refresh_from_adc_file(self, adc_path: pathlib.Path) -> tuple[str, float]:
    """Exchanges an `authorized_user` ADC refresh token over TLS 1.2+."""
    # Parse ADC credentials JSON strictly in memory without echoing contents.
    try:
      creds = json.loads(adc_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
      raise LLMAdapterAuthError("Failed reading Google Cloud ADC JSON.") from err

    # Require standard `authorized_user` ADC credentials created by `gcloud`.
    if not isinstance(creds, dict) or creds.get("type") != "authorized_user":
      raise LLMAdapterAuthError(
          "Unsupported ADC credential type; run `gcloud auth application-default"
          " login` or export `ARGUS_LLM_API_KEY`."
      )
    # Encode standard OAuth2 refresh_token grant payload in memory.
    form_body = urllib.parse.urlencode({
        "client_id": creds.get("client_id") if isinstance(creds.get("client_id"), str) else "",
        "client_secret": creds.get("client_secret") if isinstance(creds.get("client_secret"), str) else "",
        "refresh_token": creds.get("refresh_token") if isinstance(creds.get("refresh_token"), str) else "",
        "grant_type": "refresh_token",
    }).encode("utf-8")
    token_req = urllib.request.Request(
        _GOOGLE_OAUTH_TOKEN_URL,
        data=form_body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
      with urllib.request.urlopen(
          token_req, timeout=15.0, context=self._ssl_ctx
      ) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        OSError,
        TimeoutError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as err:
      raise LLMAdapterAuthError(
          "OAuth2 token refresh against Google Cloud ADC failed."
      ) from err

    # Verify top-level OAuth JSON object and non-empty string access_token.
    if not isinstance(payload, dict):
      raise LLMAdapterAuthError("OAuth2 token response must be a JSON object.")
    raw_tok = payload.get("access_token")
    token = raw_tok.strip() if isinstance(raw_tok, str) else ""
    if not token:
      raise LLMAdapterAuthError("OAuth2 token response lacked access_token.")
    # Coalesce explicit `None` on `expires_in`; reject `bool`, `NaN`/`Inf`, or `<= 0`.
    raw_exp = payload.get("expires_in")
    try:
      if isinstance(raw_exp, bool) or (
          raw_exp is not None and not isinstance(raw_exp, (int, float, str))
      ):
        raise TypeError("invalid expires_in")
      ttl = 3600.0 if raw_exp is None else float(raw_exp)
      if not math.isfinite(ttl) or ttl <= 0.0:
        raise ValueError("non-positive or non-finite expires_in")
    except (TypeError, ValueError, OverflowError) as err:
      raise LLMAdapterAuthError("Invalid OAuth2 expires_in value.") from err
    # Subtract a 60-second safety margin so tokens never expire mid-batch.
    return (token, max(60.0, ttl - 60.0))

  def invalidate_token(self) -> None:
    """Clears any cached OAuth2 access token so the next call refreshes it."""
    # Synchronize cache eviction across concurrent worker threads on HTTP 401.
    with self._lock:
      self._cached_token = ""
      self._expiry_monotonic = 0.0

  def _fetch_gcloud_access_token(self) -> tuple[str, float]:
    """Acquires an OAuth2 access token via `gcloud` CLI with `shell=False`."""
    # Check both `gcloud` and Windows `gcloud.cmd` SDK entrypoints safely.
    gcloud_bin = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if not gcloud_bin:
      raise LLMAdapterAuthError(
          "No Vertex AI credentials found. Run `gcloud auth"
          " application-default login` or set `ARGUS_LLM_API_KEY`."
      )
    # Try ADC token subcommand first, then standard active `gcloud auth` account.
    for subcmd in (
        ["auth", "application-default", "print-access-token"],
        ["auth", "print-access-token"],
    ):
      try:
        proc = subprocess.run(
            [gcloud_bin, *subcmd],
            capture_output=True,
            text=True,
            timeout=15.0,
            check=True,
        )
        tok = proc.stdout.strip()
        if tok:
          # Use a conservative 15-minute cache TTL for CLI-issued access tokens.
          return (tok, 900.0)
      except (OSError, subprocess.SubprocessError):
        continue
    raise LLMAdapterAuthError(
        "Failed acquiring Vertex AI access token via `gcloud auth`."
    )

  def get_bearer_token(self) -> str:
    """Returns a valid OAuth2 bearer token without logging sensitive values."""
    # 1. Check explicit environment bearer token overrides first.
    for env_key in (
        "ARGUS_LLM_API_KEY",
        "VERTEX_ACCESS_TOKEN",
        "GOOGLE_OAUTH_ACCESS_TOKEN",
    ):
      env_tok = os.environ.get(env_key, "").strip()
      if env_tok:
        return env_tok

    # 2. Reuse valid cached token across all concurrent worker threads.
    with self._lock:
      now = time.monotonic()
      if self._cached_token and now < self._expiry_monotonic:
        return self._cached_token

      if self._token_supplier is not None:
        tok, ttl = self._token_supplier()
      else:
        adc_path = self._resolve_adc_path()
        if adc_path is not None:
          try:
            tok, ttl = self._refresh_from_adc_file(adc_path)
          except LLMAdapterAuthError:
            # Re-raise immediately when testing an explicit `adc_file` override;
            # otherwise fall back to `gcloud` for service/impersonated accounts.
            if self._adc_file is not None:
              raise
            tok, ttl = self._fetch_gcloud_access_token()
        else:
          # 3. Fall back to `gcloud auth` CLI token resolution (`shell=False`).
          tok, ttl = self._fetch_gcloud_access_token()

      # Fail closed if token material is unexpectedly empty.
      if not tok:
        raise LLMAdapterAuthError("Resolved empty Vertex AI access token.")
      self._cached_token = tok
      self._expiry_monotonic = now + ttl
      return tok
