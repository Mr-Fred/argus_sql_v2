"""Clean-room IP, workstation path, git identity, and credential leak scanner.

Scans source code, LaTeX manuscripts, BibTeX files, configurations, and commit
messages to guarantee zero internal corporate identifiers, local workstation
paths, or secret API credentials ever enter the repository.
"""

from collections.abc import Sequence
import getpass
import logging
import pathlib
import re
import subprocess
import zlib

from argus_sql.guardrails.exceptions import GuardrailError
from argus_sql.guardrails.types import GuardrailSeverity
from argus_sql.guardrails.types import LeakFinding
from argus_sql.guardrails.types import LeakRule

# Module-level logger following Google Python Style Guide lazy-% conventions.
_LOGGER = logging.getLogger(__name__)

# File extensions inspected during repository-wide clean-room audits.
# Includes `.pdf` manuscripts plus IaC/diagram/template extensions (`.tf`, `.mmd`).
_SCANNABLE_SUFFIXES: frozenset[str] = frozenset({
    ".py",
    ".tex",
    ".bib",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".sh",
    ".ps1",
    ".bat",
    ".cmd",
    ".sql",
    ".txt",
    ".csv",
    ".pdf",
    ".tf",
    ".tfvars",
    ".hcl",
    ".mmd",
    ".example",
    ".lock",
})

# Directory names skipped during file traversal (build/VCS/IDE caches and
# gitignored quarantine staging folders).
# Skipping `.git` avoids scanning binary packfiles; Git config is checked separately.
# Skipping local IDE state (`.antigravity`, `.gemini`, `.vscode`, `.idea`) avoids
# false positives when Windows IDEs record local `python.exe` interpreter paths.
# Skipping `audit_import` keeps un-audited external archives quarantined until
# explicitly audited via `--scan-path`.
_IGNORED_DIRS: frozenset[str] = frozenset({
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "build",
    "dist",
    "node_modules",
    ".antigravity",
    ".gemini",
    ".vscode",
    ".idea",
    "audit_import",
})

# Gitignored multi-segment directories holding unpacked third-party benchmarks
# (e.g., 1.5 GB `MINIDEV.zip` CSV/JSON files) or scratch live run checkpoints.
_IGNORED_REL_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("data", "bird"),
    ("data", "spider"),
    ("results", "live_runs"),
)


def _join_parts(*parts: str) -> str:
  """Concatenates token fragments so scanner source stays free of raw markers."""
  # Joining fragments at runtime ensures this file passes its own regex rules.
  return "".join(parts)


def build_default_leak_rules() -> tuple[LeakRule, ...]:
  """Constructs clean-room regex rules including runtime OS path/user guards.

  Returns:
    Tuple of compiled-ready LeakRule definitions covering corporate domains,
    intranet shortlinks, internal build markers, workstation paths, and secrets.
  """
  # Assemble corporate and internal tool markers from fragments so this file
  # itself contains zero raw corporate domain or intranet strings.
  corp_domain = _join_parts("g", "oogle", r"\.com")
  corp_host = _join_parts(r"\.corp\.", "g", "oogle")
  corp_tld = _join_parts(r"\.g", "oog", r"\b")
  mono_repo = _join_parts(r"\bg", "oogle", r"3\b")
  internal_doc = _join_parts(r"\bg", "3doc", r"\b")
  internal_tools = "|".join((
      _join_parts("bug", "anizer"),
      _join_parts("crit", "ique"),
      _join_parts("stub", "by"),
      _join_parts("blaze", "-bin"),
      _join_parts("sponge", "2"),
      _join_parts("ci", "tc"),
      _join_parts("bo", "rg", "cron"),
  ))
  # Block internal workstation and distributed filesystem mount prefixes.
  workstation_prefix = _join_parts("/usr/local/", "g", "oogle")
  depot_prefix = _join_parts("/", "g", "oogle/", "(?:src|data|bin)")
  # Block portable macOS (`/Users/<name>/`) and Windows (`C:\Users\<name>\`)
  # home paths so imported laptop snapshots cannot leak developer directories.
  mac_home_pat = _join_parts(r"(?<![A-Za-z0-9_])/", "Users", r"/[A-Za-z0-9._-]+/")
  win_home_pat = _join_parts(
      r"(?<![A-Za-z0-9_])[A-Za-z]:[/\\]+",
      "[Uu]sers",
      r"[/\\]+[A-Za-z0-9._-]+(?: [A-Za-z0-9._-]+)?[/\\]+",
  )

  # Order rules from highest-risk corporate/secret leaks to path hygiene checks.
  rules: list[LeakRule] = [
      LeakRule(
          rule_id="CR-CORP-DOMAIN",
          category="corporate_leak",
          severity=GuardrailSeverity.CRITICAL,
          description="Prohibited corporate email or intranet domain detected.",
          pattern_text=rf"(?:@{corp_domain}|{corp_host}|{corp_tld})",
      ),
      LeakRule(
          rule_id="CR-INTERNAL-SHORTLINK",
          category="corporate_leak",
          severity=GuardrailSeverity.CRITICAL,
          description="Prohibited internal shortlink (go/, b/, cl/) detected.",
          pattern_text=(
              r"(?<![A-Za-z0-9_./-])"
              r"(?:go/[A-Za-z0-9_-]{2,}|b/\d{5,}|cl/\d{5,}|http://"
              + _join_parts("g", "oogle", "3")
              + r")"
          ),
      ),
      LeakRule(
          rule_id="CR-INTERNAL-INFRA",
          category="corporate_leak",
          severity=GuardrailSeverity.CRITICAL,
          description="Prohibited internal repository or build system term.",
          pattern_text=rf"(?:{mono_repo}|{internal_doc}|\b(?:{internal_tools})\b)",
      ),
      LeakRule(
          rule_id="CR-WORKSTATION-PATH",
          category="path",
          severity=GuardrailSeverity.HIGH,
          description="Prohibited workstation or internal mount path detected.",
          pattern_text=rf"(?:{workstation_prefix}|{depot_prefix})",
      ),
      LeakRule(
          rule_id="CR-PORTABLE-HOME-PATH",
          category="path",
          severity=GuardrailSeverity.HIGH,
          description="Hardcoded macOS or Windows workstation home path detected.",
          pattern_text=rf"(?:{mac_home_pat}|{win_home_pat})",
          case_sensitive=True,
      ),
      LeakRule(
          rule_id="CR-SECRET-API-KEY",
          category="secret",
          severity=GuardrailSeverity.CRITICAL,
          description="Hardcoded API key or cloud credential token detected.",
          pattern_text=(
              r"(?:\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}\b"
              r"|\bAIza[0-9A-Za-z_-]{35}\b"
              r"|\bya29\.[0-9A-Za-z._-]{20,}\b"
              r"|\bghp_[A-Za-z0-9]{36}\b"
              r"|\bAKIA[0-9A-Z]{16}\b)"
          ),
          case_sensitive=True,
      ),
      LeakRule(
          rule_id="CR-SECRET-PRIVATE-KEY",
          category="secret",
          severity=GuardrailSeverity.CRITICAL,
          description="PEM private key header detected; never commit keys.",
          pattern_text=_join_parts("-----BEGIN ", r"[A-Z ]*PRIVATE", " KEY-----"),
          case_sensitive=True,
      ),
  ]

  # Dynamically guard native, POSIX, and JSON-escaped forms of local $HOME.
  home_str = str(pathlib.Path.home().resolve())
  if len(home_str) > 4:
    home_variants = sorted(
        {home_str, home_str.replace("\\", "/"), home_str.replace("\\", "\\\\")},
        key=len,
        reverse=True,
    )
    rules.append(
        LeakRule(
            rule_id="CR-LOCAL-HOME-PATH",
            category="path",
            severity=GuardrailSeverity.HIGH,
            description="Hardcoded local machine $HOME path detected.",
            pattern_text="(?:" + "|".join(re.escape(v) for v in home_variants) + ")",
            case_sensitive=True,
        )
    )

  # Query OS username safely; fall back to empty string in headless containers.
  try:
    os_user = getpass.getuser().strip()
  except (OSError, KeyError):
    os_user = ""

  # Skip generic OS role names (including OEM Windows defaults like 'Owner').
  generic_os_accounts = {
      "ubuntu",
      "admin",
      "administrator",
      "runner",
      "owner",
      "users",
      "public",
      "default",
      "guest",
  }
  if len(os_user) >= 5 and os_user.lower() not in generic_os_accounts:
    rules.append(
        LeakRule(
            rule_id="CR-LOCAL-OS-USER",
            category="corporate_leak",
            severity=GuardrailSeverity.HIGH,
            description="Local workstation OS username detected in content.",
            pattern_text=rf"\b{re.escape(os_user)}\b",
        )
    )

  return tuple(rules)


def _redact_match_line(line: str, match_span: tuple[int, int]) -> str:
  """Masks the matched token inside a preview line to avoid echoing secrets."""
  # Extract matched character bounds to avoid leaking raw credentials to stderr.
  start, end = match_span
  matched_len = max(1, end - start)
  # Preserve only a tiny length indicator rather than printing sensitive text.
  masked = f"[REDACTED:{matched_len}chars]"
  snippet = (line[:start] + masked + line[end:]).strip()
  # Truncate long minified JSON/TeX lines to keep pre-commit output readable.
  return snippet[:120]


def scan_text_for_leaks(
    text: str,
    target_label: str,
    rules: Sequence[LeakRule] | None = None,
) -> tuple[LeakFinding, ...]:
  """Scans an arbitrary text buffer line-by-line against clean-room rules.

  Args:
    text: Text content to inspect (source file, LaTeX, or commit message).
    target_label: Relative file path or label reported in any LeakFinding.
    rules: Optional custom sequence of LeakRule instances.

  Returns:
    Tuple of LeakFinding objects ordered by line number.
  """
  # Fall back to the full clean-room rule suite when custom rules are omitted.
  active_rules = rules if rules is not None else build_default_leak_rules()
  # Compile each regex once outside the line loop for fast O(lines * rules) scan.
  compiled = [
      (
          rule,
          re.compile(
              rule.pattern_text,
              0 if rule.case_sensitive else re.IGNORECASE,
          ),
      )
      for rule in active_rules
  ]

  # Collect every match across the file so contributors can fix all leaks at once.
  findings: list[LeakFinding] = []
  # Check every line against all compiled rules to surface complete diagnostics.
  for idx, raw_line in enumerate(text.splitlines(), start=1):
    for rule, regex in compiled:
      match = regex.search(raw_line)
      if match is not None:
        # Record redacted context so logs never echo sensitive tokens verbatim.
        findings.append(
            LeakFinding(
                rule_id=rule.rule_id,
                category=rule.category,
                severity=rule.severity,
                target=target_label,
                line_number=idx,
                redacted_preview=_redact_match_line(raw_line, match.span()),
                description=rule.description,
            )
        )
  return tuple(findings)


def scan_pdf_for_leaks(
    pdf_path: pathlib.Path,
    target_label: str,
    rules: Sequence[LeakRule] | None = None,
) -> tuple[LeakFinding, ...]:
  """Scans PDF headers, UTF-16BE metadata, and FlateDecode streams for leaks.

  Args:
    pdf_path: Path to the `.pdf` artifact to inspect.
    target_label: Relative label reported in any LeakFinding.
    rules: Optional custom sequence of LeakRule instances.

  Returns:
    Tuple of LeakFinding items discovered in raw or decompressed PDF text.
  """
  # Decode raw bytes losslessly via latin-1 and strip UTF-16BE null padding.
  raw_bytes = pdf_path.read_bytes()
  raw_text = raw_bytes.decode("latin-1")
  utf16_stripped = raw_text.replace("\x00", "")

  # Decompress zlib FlateDecode streams so rendered PDF body text is scanned too.
  chunks: list[str] = [raw_text, utf16_stripped]
  for stream_bytes in re.findall(
      rb"stream\r?\n(.*?)\r?\nendstream", raw_bytes, flags=re.DOTALL
  ):
    try:
      decompressed = zlib.decompress(stream_bytes).decode("latin-1")
      chunks.append(decompressed)
    except zlib.error:
      # Non-zlib streams (such as font subsets) are already covered via raw_text.
      continue

  return scan_text_for_leaks("\n".join(chunks), target_label, rules=rules)


def iter_scannable_repo_files(repo_root: pathlib.Path) -> tuple[pathlib.Path, ...]:
  """Enumerates scannable text files strictly confined within repo_root.

  Args:
    repo_root: Root directory of the Argus-SQL repository.

  Returns:
    Sorted tuple of resolved file paths safe for UTF-8 clean-room scanning.
  """
  resolved_root = repo_root.resolve()
  discovered: list[pathlib.Path] = []
  for path in sorted(resolved_root.rglob("*")):
    # Enforce directory containment and skip symlinks and cache/VCS folders.
    if not path.is_file() or path.is_symlink():
      continue
    rel_parts = path.relative_to(resolved_root).parts
    if any(part in _IGNORED_DIRS for part in rel_parts):
      continue
    # Skip gitignored benchmark downloads (`data/bird`, `data/spider`, etc.).
    if any(rel_parts[: len(pfx)] == pfx for pfx in _IGNORED_REL_PREFIXES):
      continue
    # Include extensionless Git hooks (.githooks/*) alongside source/doc files.
    is_hook = ".githooks" in rel_parts
    if (
        is_hook
        or path.suffix.lower() in _SCANNABLE_SUFFIXES
        or path.name in {".gitignore", ".gitattributes", "Dockerfile", "Makefile"}
    ):
      discovered.append(path)
  return tuple(discovered)


def verify_git_author_identity(repo_root: pathlib.Path) -> tuple[LeakFinding, ...]:
  """Verifies that local Git user.name and user.email pass clean-room rules.

  Args:
    repo_root: Root directory containing the standalone `.git` repository.

  Returns:
    Empty tuple if clean; otherwise LeakFinding items describing violations.

  Raises:
    GuardrailError: If invoking `git config` fails unexpectedly.
  """
  resolved_root = repo_root.resolve()
  findings: list[LeakFinding] = []
  # Query local repo-scoped Git config so global corporate configs cannot sneak in.
  for key in ("user.name", "user.email"):
    try:
      proc = subprocess.run(
          ["git", "-C", str(resolved_root), "config", "--local", "--get", key],
          capture_output=True,
          text=True,
          check=False,
      )
    except OSError as err:
      raise GuardrailError(f"Failed to query git config '{key}'.") from err

    value = proc.stdout.strip()
    if not value:
      findings.append(
          LeakFinding(
              rule_id="CR-GIT-IDENTITY-MISSING",
              category="git",
              severity=GuardrailSeverity.HIGH,
              target=f"git-config:{key}",
              line_number=1,
              redacted_preview="<unset>",
              description=f"Local git '{key}' must be explicitly configured.",
          )
      )
      continue
    findings.extend(scan_text_for_leaks(value, f"git-config:{key}"))

  _LOGGER.debug("Checked git author identity with %d findings.", len(findings))
  return tuple(findings)
