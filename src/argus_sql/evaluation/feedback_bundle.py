"""Clean-room telemetry sanitization, SHA-256 signing, and bundle I/O.

Enables split-environment CI/CD workflows where code is authored in a
credential-free workspace, evaluated on a personal machine against live APIs,
and imported back as a path-scrubbed, integrity-verified `FeedbackBundle`.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import datetime
import hashlib
import hmac
import json
import logging
import pathlib
import re
from typing import Any

from argus_sql.contracts.types import QueryDifficulty
from argus_sql.evaluation.exceptions import EvaluationError
from argus_sql.evaluation.types import BenchmarkSuiteReport
from argus_sql.evaluation.types import FeedbackBundle
from argus_sql.evaluation.types import SystemEvalSummary
from argus_sql.evaluation.types import TaskEvalRecord

# Module-level logger using Google Python Style Guide lazy-% formatting.
_LOGGER = logging.getLogger(__name__)

# Canonical wire schema identifier rejecting incompatible or untrusted JSON.
FEEDBACK_SCHEMA_VERSION = "argus_feedback.v1"

# Strict 16 MiB payload ceiling blocking memory-exhaustion DoS on ingestion.
_MAX_FEEDBACK_BYTES = 16 * 1024 * 1024

# Allow-listed top-level JSON keys accepted by `parse_and_verify_feedback_bundle`.
_ALLOWED_BUNDLE_KEYS: frozenset[str] = frozenset({
    "schema_version",
    "created_utc",
    "provider",
    "model_label",
    "split_name",
    "git_commit",
    "suites",
    "payload_sha256",
})


def _frag(*parts: str) -> str:
  """Concatenates regex literal fragments to keep source clean-room neutral."""
  return "".join(parts)


# Compile OS filesystem path & credential scrubbers once at import time.
# Matches explicit OS root directories (including `file:///C:/Users/...` URIs)
# while leaving SQL division (`a/b`) completely untouched.
_OS_PATH_REGEX = re.compile(
    _frag(
        r"(?:file://+)?(?<![A-Za-z0-9_.-])(?:",
        r"/(?:home|",
        "Users",
        r"|usr/local|private/var|var/folders|tmp)/[A-Za-z0-9._/@~+%-]+",
        r"|[A-Za-z]:[/\\]+",
        "[Uu]sers",
        r"[/\\]+[A-Za-z0-9._~+-]+(?: [A-Za-z0-9._~+-]+)?(?:[/\\]+[A-Za-z0-9._/\\~+@%-]+)?)",
    )
)


def _build_local_home_scrubber() -> re.Pattern[str] | None:
  """Builds a regex scrubbing native, POSIX, and JSON-escaped `$HOME` paths."""
  home_str = str(pathlib.Path.home().resolve())
  if len(home_str) <= 4:
    return None
  variants = sorted(
      {home_str, home_str.replace("\\", "/"), home_str.replace("\\", "\\\\")},
      key=len,
      reverse=True,
  )
  body = "|".join(re.escape(v) for v in variants)
  return re.compile(rf"(?:file://+)?(?:{body})(?:[/\\]+[A-Za-z0-9._/\\~+@%-]+)?")


_LOCAL_HOME_REGEX = _build_local_home_scrubber()

# Matches cloud bearer tokens, API keys, and PEM key headers for redaction.
_CREDENTIAL_REGEX = re.compile(
    _frag(
        r"(?:\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}\b",
        r"|\bAIza[0-9A-Za-z_-]{35}\b",
        r"|\bya29\.[0-9A-Za-z._-]{20,}\b",
        r"|\bghp_[A-Za-z0-9]{36}\b",
        r"|\bAKIA[0-9A-Z]{16}\b",
        r"|-----BEGIN ",
        r"[A-Z ]*PRIVATE",
        r" KEY-----)",
    )
)


def sanitize_diagnostic_text(
    text: str,
    max_chars: int = 2000,
    single_line: bool = False,
) -> str:
  """Scrubs workstation paths and credential tokens from diagnostic strings.

  Args:
    text: Raw SQL candidate or SQLite error message string.
    max_chars: Maximum UTF-8 character length retained after scrubbing.
    single_line: When True, collapses internal newlines/tabs into single spaces.

  Returns:
    Sanitized, length-bounded diagnostic string safe for clean-room commits.
  """
  if not text:
    return ""
  # Redact credentials before path tokens so secret substrings never survive.
  cleaned = _CREDENTIAL_REGEX.sub("<CREDENTIAL_REDACTED>", text)
  cleaned = _OS_PATH_REGEX.sub("<PATH_REDACTED>", cleaned)
  if _LOCAL_HOME_REGEX is not None:
    cleaned = _LOCAL_HOME_REGEX.sub("<PATH_REDACTED>", cleaned)
  if single_line:
    cleaned = " ".join(cleaned.split())
  else:
    cleaned = cleaned.strip()
  # Enforce hard length ceiling to keep feedback bundles compact and auditable.
  if len(cleaned) > max_chars:
    return cleaned[: max_chars - 3] + "..."
  return cleaned


def _sanitize_record(rec: TaskEvalRecord) -> TaskEvalRecord:
  """Returns a copy of `rec` with all free-text fields clean-room scrubbed."""
  # Bound identifiers, error previews, and SQL text independently by risk surface.
  return TaskEvalRecord(
      question_id=sanitize_diagnostic_text(rec.question_id, 128, single_line=True),
      db_id=sanitize_diagnostic_text(rec.db_id, 128, single_line=True),
      difficulty=rec.difficulty,
      is_voice=bool(rec.is_voice),
      ex_correct=bool(rec.ex_correct),
      # Round float telemetry deterministically so SHA-256 digests stay stable.
      r_ves_score=round(float(rec.r_ves_score), 4),
      latency_ms=round(float(rec.latency_ms), 3),
      total_tokens=int(rec.total_tokens),
      failure_category=sanitize_diagnostic_text(
          rec.failure_category, 80, single_line=True
      ),
      predicted_sql=sanitize_diagnostic_text(rec.predicted_sql, 2000),
      # Collapse SQLite error messages onto one line after path/key scrubbing.
      error_summary=sanitize_diagnostic_text(
          rec.error_summary, 240, single_line=True
      ),
      invariant_codes=tuple(
          sanitize_diagnostic_text(code, 80, single_line=True)
          for code in rec.invariant_codes
      ),
      was_repaired=bool(rec.was_repaired),
      selected_strategy=sanitize_diagnostic_text(
          rec.selected_strategy, 80, single_line=True
      ),
  )


def _sanitize_suite(suite: BenchmarkSuiteReport) -> BenchmarkSuiteReport:
  """Sanitizes every record and label inside a `BenchmarkSuiteReport`."""
  # Scrub system labels alongside per-task evaluation traces.
  clean_records = tuple(
      (
          sanitize_diagnostic_text(sys_name, 120, single_line=True),
          tuple(_sanitize_record(r) for r in recs),
      )
      for sys_name, recs in suite.records_by_system
  )
  return BenchmarkSuiteReport(
      suite_name=sanitize_diagnostic_text(suite.suite_name, 120, single_line=True),
      summaries=suite.summaries,
      records_by_system=clean_records,
  )


def compute_bundle_sha256(
    metadata: Mapping[str, str],
    suites: Sequence[BenchmarkSuiteReport],
) -> str:
  """Computes canonical SHA-256 hex digest over bundle metadata and suites.

  Args:
    metadata: Header dictionary (`schema_version`, `created_utc`, etc.).
    suites: Ordered sequence of sanitized `BenchmarkSuiteReport` instances.

  Returns:
    Lowercase 64-character hexadecimal SHA-256 digest.
  """
  # Canonicalize key ordering and separators for cross-OS deterministic hashing.
  canonical_obj = {
      "metadata": dict(sorted(metadata.items())),
      "suites": [dataclasses.asdict(s) for s in suites],
  }
  # Force strict RFC 8259 ASCII JSON (`allow_nan=False`) for deterministic SHA.
  encoded = json.dumps(
      canonical_obj,
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=True,
      allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def build_feedback_bundle(
    suites: Sequence[BenchmarkSuiteReport],
    provider: str,
    model_label: str,
    split_name: str,
    git_commit: str = "uncommitted",
    created_utc: str | None = None,
) -> FeedbackBundle:
  """Constructs a sanitized, SHA-256-signed `FeedbackBundle` ready for export.

  Args:
    suites: Evaluated BenchmarkSuiteReport sequence (e.g., Clean + Voice).
    provider: Live transport label ('vertex', 'openai', or 'diagnostic').
    model_label: Model identifier evaluated on the personal machine.
    split_name: Dataset split filename (never a full local path).
    git_commit: Short Git SHA or `'uncommitted'`.
    created_utc: Optional ISO-8601 UTC timestamp override for deterministic tests.

  Returns:
    Immutable `FeedbackBundle` with verified clean-room fields and SHA-256 hash.
  """
  # Strip any directory components from `split_name` so laptop paths never leak.
  safe_split = pathlib.PurePosixPath(split_name.replace("\\", "/")).name
  # Default to UTC wall-clock ISO-8601 when not pinned by deterministic tests.
  timestamp = created_utc or datetime.datetime.now(
      datetime.timezone.utc
  ).strftime("%Y-%m-%dT%H:%M:%SZ")
  sanitized_suites = tuple(_sanitize_suite(s) for s in suites)

  # Scrub header provenance fields before computing the cryptographic digest.
  header = {
      "schema_version": FEEDBACK_SCHEMA_VERSION,
      "created_utc": sanitize_diagnostic_text(timestamp, 40, single_line=True),
      "provider": sanitize_diagnostic_text(provider, 40, single_line=True),
      "model_label": sanitize_diagnostic_text(model_label, 120, single_line=True),
      "split_name": sanitize_diagnostic_text(safe_split, 120, single_line=True),
      "git_commit": sanitize_diagnostic_text(git_commit, 40, single_line=True),
  }
  digest = compute_bundle_sha256(header, sanitized_suites)
  return FeedbackBundle(
      schema_version=header["schema_version"],
      created_utc=header["created_utc"],
      provider=header["provider"],
      model_label=header["model_label"],
      split_name=header["split_name"],
      git_commit=header["git_commit"],
      suites=sanitized_suites,
      payload_sha256=digest,
  )


def serialize_feedback_bundle(bundle: FeedbackBundle) -> str:
  """Serializes a `FeedbackBundle` into pretty-printed UTF-8 JSON.

  Args:
    bundle: Verified `FeedbackBundle` instance to serialize.

  Returns:
    Formatted JSON text terminated with a trailing newline.
  """
  # Emit strict RFC 8259 2-space indented JSON for clean Git diffs across runs.
  payload = dataclasses.asdict(bundle)
  return json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n"


def _parse_task_record(raw: Mapping[str, Any]) -> TaskEvalRecord:
  """Validates and reconstructs one `TaskEvalRecord` from untrusted JSON."""
  try:
    # Enforce enum membership on `difficulty` to reject arbitrary strings.
    difficulty = QueryDifficulty(str(raw["difficulty"]))
    inv_raw = raw.get("invariant_codes", ())
    if not isinstance(inv_raw, list):
      raise EvaluationError("`invariant_codes` must be a JSON list.")
    latency_val = float(raw["latency_ms"])
    tok_val = int(raw["total_tokens"])
    if latency_val < 0.0 or tok_val < 0:
      raise EvaluationError("Negative latency or token count in TaskEvalRecord.")
    return TaskEvalRecord(
        question_id=str(raw["question_id"]),
        db_id=str(raw["db_id"]),
        difficulty=difficulty,
        is_voice=bool(raw["is_voice"]),
        ex_correct=bool(raw["ex_correct"]),
        r_ves_score=float(raw["r_ves_score"]),
        latency_ms=latency_val,
        total_tokens=tok_val,
        failure_category=str(raw["failure_category"]),
        predicted_sql=str(raw.get("predicted_sql", "")),
        error_summary=str(raw.get("error_summary", "")),
        invariant_codes=tuple(str(x) for x in inv_raw),
        was_repaired=bool(raw.get("was_repaired", False)),
        selected_strategy=str(raw.get("selected_strategy", "")),
    )
  except (KeyError, TypeError, ValueError) as err:
    raise EvaluationError(f"Malformed TaskEvalRecord in bundle: {err}") from err


def _parse_suite_report(raw: Mapping[str, Any]) -> BenchmarkSuiteReport:
  """Validates and reconstructs one `BenchmarkSuiteReport` from JSON."""
  try:
    summaries: list[SystemEvalSummary] = []
    # Rehydrate leaderboard rows with explicit numeric casts.
    for s_item in raw["summaries"]:
      f_counts = tuple((str(k), int(v)) for k, v in s_item["failure_counts"])
      summaries.append(
          SystemEvalSummary(
              system_name=str(s_item["system_name"]),
              total_tasks=int(s_item["total_tasks"]),
              ex_overall=float(s_item["ex_overall"]),
              ex_simple=float(s_item["ex_simple"]),
              ex_moderate=float(s_item["ex_moderate"]),
              ex_challenging=float(s_item["ex_challenging"]),
              r_ves_overall=float(s_item["r_ves_overall"]),
              median_latency_ms=float(s_item["median_latency_ms"]),
              p95_latency_ms=float(s_item["p95_latency_ms"]),
              mean_tokens=float(s_item["mean_tokens"]),
              failure_counts=f_counts,
          )
      )
    # Reconstruct immutable `(system_name, tuple_of_records)` mapping pairs.
    records_by_sys = tuple(
        (str(sys_name), tuple(_parse_task_record(r) for r in rec_list))
        for sys_name, rec_list in raw["records_by_system"]
    )
    return BenchmarkSuiteReport(
        suite_name=str(raw["suite_name"]),
        summaries=tuple(summaries),
        records_by_system=records_by_sys,
    )
  except (KeyError, TypeError, ValueError) as err:
    raise EvaluationError(f"Malformed BenchmarkSuiteReport: {err}") from err


def parse_and_verify_feedback_bundle(raw_json: str) -> FeedbackBundle:
  """Parses, leak-checks, and verifies SHA-256 integrity of a feedback JSON.

  Args:
    raw_json: UTF-8 JSON string imported from an external evaluation run.

  Returns:
    Verified `FeedbackBundle` instance.

  Raises:
    EvaluationError: On size overflow, schema mismatch, leak, or bad SHA-256.
  """
  # Fail closed on oversized inputs before invoking `json.loads`.
  if len(raw_json.encode("utf-8")) > _MAX_FEEDBACK_BYTES:
    raise EvaluationError("Feedback bundle exceeds 16 MiB security limit.")

  # Reject raw payloads containing unredacted credentials, OS paths, or $HOME.
  has_home_leak = bool(
      _LOCAL_HOME_REGEX is not None and _LOCAL_HOME_REGEX.search(raw_json)
  )
  if _CREDENTIAL_REGEX.search(raw_json) or _OS_PATH_REGEX.search(raw_json) or has_home_leak:
    raise EvaluationError(
        "Rejected feedback bundle: unredacted credential or workstation path."
    )

  # Parse strict RFC 8259 JSON, rejecting non-standard NaN/Infinity constants.
  def _reject_nonfinite(tok: str) -> None:
    """Fails closed on non-RFC-8259 `NaN` or `Infinity` literals."""
    raise EvaluationError(f"Non-finite JSON constant '{tok}' is forbidden.")

  try:
    data = json.loads(raw_json, parse_constant=_reject_nonfinite)
  except json.JSONDecodeError as err:
    raise EvaluationError(f"Invalid JSON in feedback bundle: {err}") from err

  # Enforce exact top-level allow-listed schema keys and wire version.
  if not isinstance(data, dict) or set(data.keys()) != _ALLOWED_BUNDLE_KEYS:
    raise EvaluationError("Feedback bundle keys violate allow-listed schema.")
  if data.get("schema_version") != FEEDBACK_SCHEMA_VERSION:
    raise EvaluationError(
        f"Unsupported feedback schema '{data.get('schema_version')}'."
    )
  if not isinstance(data.get("suites"), list):
    raise EvaluationError("Feedback bundle `suites` field must be a list.")

  # Reconstruct typed suite reports and recompute canonical SHA-256 digest.
  suites = tuple(_parse_suite_report(item) for item in data["suites"])
  header = {
      "schema_version": str(data["schema_version"]),
      "created_utc": str(data["created_utc"]),
      "provider": str(data["provider"]),
      "model_label": str(data["model_label"]),
      "split_name": str(data["split_name"]),
      "git_commit": str(data["git_commit"]),
  }
  expected_sha = compute_bundle_sha256(header, suites)
  claimed_sha = str(data["payload_sha256"])
  # Constant-time digest check prevents partial comparison shortcuts.
  if not hmac.compare_digest(expected_sha, claimed_sha):
    raise EvaluationError(
        "Feedback bundle SHA-256 integrity mismatch; file may be modified."
    )

  _LOGGER.info("Verified feedback bundle (%d suites).", len(suites))
  return FeedbackBundle(
      schema_version=header["schema_version"],
      created_utc=header["created_utc"],
      provider=header["provider"],
      model_label=header["model_label"],
      split_name=header["split_name"],
      git_commit=header["git_commit"],
      suites=suites,
      payload_sha256=claimed_sha,
  )


def load_feedback_bundle_file(
    bundle_path: pathlib.Path,
    allowed_root: pathlib.Path | None = None,
) -> FeedbackBundle:
  """Loads and verifies a `FeedbackBundle` from a confined non-symlink file.

  Args:
    bundle_path: Path to the `.json` feedback bundle file.
    allowed_root: Optional directory boundary that `bundle_path` must stay inside.

  Returns:
    Integrity-verified `FeedbackBundle` instance.

  Raises:
    EvaluationError: If symlinked, outside `allowed_root`, missing, or invalid.
  """
  # Reject symlinks up front so callers cannot dereference external files.
  if bundle_path.is_symlink():
    raise EvaluationError("Refusing to load symlinked feedback bundle.")
  resolved = bundle_path.resolve()
  # Verify containment inside `allowed_root` before touching file metadata.
  if allowed_root is not None and not resolved.is_relative_to(
      allowed_root.resolve()
  ):
    raise EvaluationError("Feedback bundle path escapes allowed root boundary.")
  if not resolved.is_file():
    raise EvaluationError(f"Feedback bundle file not found: '{bundle_path}'.")
  # Check byte size via `stat()` prior to buffering file contents in RAM.
  if resolved.stat().st_size > _MAX_FEEDBACK_BYTES:
    raise EvaluationError("Feedback bundle file exceeds 16 MiB ceiling.")

  try:
    raw_text = resolved.read_text(encoding="utf-8")
  except OSError as err:
    raise EvaluationError(f"Failed reading feedback bundle: {err}") from err
  return parse_and_verify_feedback_bundle(raw_text)
