"""Vertex AI / Gemini Enterprise URL validation and OAuth2 ADC token resolution.

Isolates allow-list validation for Google Cloud project/location/model slugs
and thread-safe in-memory OAuth2 Application Default Credentials (ADC) token
refresh so batch transport remains modular and free of third-party SDKs.
"""

from collections.abc import Callable
import json
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
_GCP_PROJECT_RE = re.compile(r"^[a-z][a-z0-9-]{4,61}[a-z0-9]$")
_GCP_LOCATION_RE = re.compile(r"^[a-z0-9-]+$")
_GEMINI_MODEL_RE = re.compile(r"^[A-Za-z0-9._-]+$")

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
    LLMAdapterAuthError: If `location` contains disallowed URL/host characters.
  """
  cleaned = location.strip().lower()
  # Restrict location strictly to lowercase alphanumeric and hyphen characters.
  if not cleaned or not _GCP_LOCATION_RE.match(cleaned):
    raise LLMAdapterAuthError("Invalid Vertex AI location identifier.")
  return cleaned


def normalize_gemini_model_name(model_name: str) -> str:
  """Normalizes and validates a public Vertex AI Gemini model identifier.

  Args:
    model_name: Raw model identifier or `publishers/google/models/...` resource.

  Returns:
    Validated bare Gemini model slug (e.g., `'gemini-2.5-flash'`).

  Raises:
    LLMAdapterAuthError: If the normalized model name contains unsafe characters.
  """
  cleaned = model_name.strip()
  # Strip optional Vertex/OpenAI resource prefixes so bare model slugs remain uniform.
  if "/models/" in cleaned:
    cleaned = cleaned.rsplit("/models/", 1)[-1]
  elif cleaned.startswith("google/"):
    cleaned = cleaned.split("/", 1)[1]

  # Enforce strict character allow-list before interpolating into URL paths.
  if not cleaned or not _GEMINI_MODEL_RE.match(cleaned):
    raise LLMAdapterAuthError("Invalid Vertex AI Gemini model identifier.")
  return cleaned


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
  proj = validate_gcp_project_id(project_id)
  loc = validate_gcp_location(location)
  model = normalize_gemini_model_name(model_name)
  # Global Gemini Enterprise routing targets `aiplatform.googleapis.com` directly.
  host = (
      "aiplatform.googleapis.com"
      if loc == "global"
      else f"{loc}-aiplatform.googleapis.com"
  )
  return (
      f"https://{host}/v1beta1/projects/{proj}/locations/{loc}"
      f"/publishers/google/models/{model}:generateContent"
  )


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
    except (OSError, json.JSONDecodeError) as err:
      raise LLMAdapterAuthError("Failed reading Google Cloud ADC JSON.") from err

    # Require standard `authorized_user` ADC credentials created by `gcloud`.
    if creds.get("type") != "authorized_user":
      raise LLMAdapterAuthError(
          "Unsupported ADC credential type; run `gcloud auth application-default"
          " login` or export `ARGUS_LLM_API_KEY`."
      )
    # Encode standard OAuth2 refresh_token grant payload in memory.
    form_body = urllib.parse.urlencode({
        "client_id": str(creds.get("client_id", "")),
        "client_secret": str(creds.get("client_secret", "")),
        "refresh_token": str(creds.get("refresh_token", "")),
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
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
      raise LLMAdapterAuthError(
          "OAuth2 token refresh against Google Cloud ADC failed."
      ) from err

    token = str(payload.get("access_token", "")).strip()
    ttl = float(payload.get("expires_in", 3600))
    if not token:
      raise LLMAdapterAuthError("OAuth2 token response lacked access_token.")
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
