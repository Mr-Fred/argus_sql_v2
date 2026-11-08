"""Git pre-commit and commit-msg hook installer and active-hook verifier.

Generates version-controlled executable POSIX shell hooks in `.githooks/`,
mirrors them into `.git/hooks/` so commits are protected immediately out of the
box, and configures `git config --local core.hooksPath .githooks`.
"""

import logging
import os
import pathlib
import stat
import subprocess

from argus_sql.guardrails.exceptions import GuardrailError
from argus_sql.guardrails.types import GuardrailSeverity
from argus_sql.guardrails.types import LeakFinding

# Module-level logger following Google Python Style Guide conventions.
_LOGGER = logging.getLogger(__name__)

# Shared POSIX sh helper resolving `.venv` or global Python 3.11+ across
# Linux, macOS, and Windows Git while skipping Microsoft Store stubs.
_PY311_PROBE = "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
_PYTHON_RESOLVER_SH = f"""run_py() {{
  if [ -x "${{REPO_ROOT}}/.venv/Scripts/python.exe" ] && "${{REPO_ROOT}}/.venv/Scripts/python.exe" -c "{_PY311_PROBE}" >/dev/null 2>&1; then
    "${{REPO_ROOT}}/.venv/Scripts/python.exe" "$@"
  elif [ -x "${{REPO_ROOT}}/.venv/bin/python" ] && "${{REPO_ROOT}}/.venv/bin/python" -c "{_PY311_PROBE}" >/dev/null 2>&1; then
    "${{REPO_ROOT}}/.venv/bin/python" "$@"
  elif command -v python3 >/dev/null 2>&1 && python3 -c "{_PY311_PROBE}" >/dev/null 2>&1; then
    python3 "$@"
  elif command -v python >/dev/null 2>&1 && python -c "{_PY311_PROBE}" >/dev/null 2>&1; then
    python "$@"
  elif command -v py >/dev/null 2>&1 && py -3 -c "{_PY311_PROBE}" >/dev/null 2>&1; then
    py -3 "$@"
  else
    echo "[argus-guardrails] ERROR: Python 3.11+ interpreter not found on PATH." >&2
    exit 1
  fi
}}"""

# Pre-commit shell gate running clean-room scan, AST check, and full unit tests.
# Uses `sys.stdout.write(os.pathsep)` so Windows `python.exe` never emits `\r`.
_PRE_COMMIT_SCRIPT = f"""#!/usr/bin/env sh
set -eu

REPO_ROOT="$(git rev-parse --show-toplevel)"
{_PYTHON_RESOLVER_SH}
PATH_SEP="$(run_py -c "import os, sys; sys.stdout.write(os.pathsep)")"
export PYTHONPATH="${{REPO_ROOT}}/src${{PYTHONPATH:+${{PATH_SEP}}${{PYTHONPATH}}}}"

echo "[argus-guardrails] Running clean-room IP, secret & AST quality checks..."
run_py -m argus_sql.guardrails.cli --repo-root "${{REPO_ROOT}}" --verify-all

echo "[argus-guardrails] Running hermetic unit test suite..."
run_py -m unittest discover -s "${{REPO_ROOT}}/src" -p "*_test.py"
"""

# Commit-msg shell gate scanning the proposed Git commit message before commit.
_COMMIT_MSG_SCRIPT = f"""#!/usr/bin/env sh
set -eu

MSG_FILE="$1"
REPO_ROOT="$(git rev-parse --show-toplevel)"
{_PYTHON_RESOLVER_SH}
PATH_SEP="$(run_py -c "import os, sys; sys.stdout.write(os.pathsep)")"
export PYTHONPATH="${{REPO_ROOT}}/src${{PYTHONPATH:+${{PATH_SEP}}${{PYTHONPATH}}}}"

run_py -m argus_sql.guardrails.cli --repo-root "${{REPO_ROOT}}" --check-commit-msg "${{MSG_FILE}}"
"""


def _write_executable_script(target_path: pathlib.Path, content: str) -> None:
  """Writes LF-normalized UTF-8 shell script `content` with executable bits."""
  # Enforce LF (`\n`) even on Windows so Git for Windows `sh` never hits `\r`.
  target_path.write_text(content, encoding="utf-8", newline="\n")
  current_mode = target_path.stat().st_mode
  # Grant user/group/other execute permissions (`chmod +x`) for Git invocation.
  os.chmod(
      target_path,
      current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
  )


def install_git_hooks(
    repo_root: pathlib.Path,
    configure_git: bool = True,
) -> tuple[pathlib.Path, pathlib.Path]:
  """Installs pre-commit and commit-msg hooks in `.githooks/` and `.git/hooks/`.

  Args:
    repo_root: Root directory of the standalone Argus-SQL Git repository.
    configure_git: Whether to mirror into `.git/hooks/` and set `core.hooksPath`.

  Returns:
    Pair of version-controlled paths `(pre_commit_path, commit_msg_path)`.

  Raises:
    GuardrailError: If `.git` is missing or hook installation fails.
  """
  # Validate that repo_root owns a standalone `.git` folder (never parent repo).
  resolved_root = repo_root.resolve()
  git_dir = resolved_root / ".git"
  if not git_dir.is_dir():
    raise GuardrailError(
        f"Standalone '.git' directory not found inside '{resolved_root}'."
    )

  # Store version-controlled hooks inside `.githooks/` confined to repo_root.
  hooks_dir = (resolved_root / ".githooks").resolve()
  if not hooks_dir.is_relative_to(resolved_root):
    raise GuardrailError("Resolved hooks directory escaped repository root.")

  hooks_dir.mkdir(parents=True, exist_ok=True)
  pre_commit = hooks_dir / "pre-commit"
  commit_msg = hooks_dir / "commit-msg"

  # Write deterministic hook scripts into `.githooks/` first.
  _write_executable_script(pre_commit, _PRE_COMMIT_SCRIPT)
  _write_executable_script(commit_msg, _COMMIT_MSG_SCRIPT)

  # Mirror into `.git/hooks/` as belt-and-suspenders defense even before
  # `core.hooksPath` is read, and set `core.hooksPath = .githooks` locally.
  if configure_git and os.access(git_dir, os.W_OK):
    native_hooks_dir = (git_dir / "hooks").resolve()
    if native_hooks_dir.is_relative_to(resolved_root):
      try:
        native_hooks_dir.mkdir(parents=True, exist_ok=True)
        _write_executable_script(native_hooks_dir / "pre-commit", _PRE_COMMIT_SCRIPT)
        _write_executable_script(native_hooks_dir / "commit-msg", _COMMIT_MSG_SCRIPT)
      except OSError as err:
        # Read-only CI containers may lock `.git/hooks`; log and continue.
        _LOGGER.debug("Skipped .git/hooks mirror: %s", err)

    try:
      proc = subprocess.run(
          [
              "git",
              "-C",
              str(resolved_root),
              "config",
              "--local",
              "core.hooksPath",
              ".githooks",
          ],
          capture_output=True,
          text=True,
          check=False,
      )
      if proc.returncode != 0:
        _LOGGER.debug("Skipped core.hooksPath update: %s", proc.stderr.strip())
    except OSError as err:
      raise GuardrailError("Failed to invoke git config for hooksPath.") from err

  _LOGGER.info("Installed clean-room Git hooks in %s", hooks_dir)
  return (pre_commit, commit_msg)


def verify_git_hooks_active(repo_root: pathlib.Path) -> tuple[LeakFinding, ...]:
  """Verifies that executable pre-commit and commit-msg hooks are active in Git.

  Args:
    repo_root: Root directory containing the standalone `.git` repository.

  Returns:
    Empty tuple when active hooks guard commits; LeakFinding items otherwise.
  """
  resolved_root = repo_root.resolve()
  git_dir = resolved_root / ".git"
  if not git_dir.is_dir():
    return ()

  # Check whether Git resolves hooks via `core.hooksPath` or `.git/hooks`.
  # In read-only CI/sandbox mounts (`not W_OK` on `.git`), verify `.githooks/`.
  proc = subprocess.run(
      ["git", "-C", str(resolved_root), "config", "--local", "--get", "core.hooksPath"],
      capture_output=True,
      text=True,
      check=False,
  )
  configured_rel = proc.stdout.strip()
  if configured_rel:
    active_dir = (resolved_root / configured_rel).resolve()
  elif not os.access(git_dir, os.W_OK):
    active_dir = (resolved_root / ".githooks").resolve()
  else:
    active_dir = (git_dir / "hooks").resolve()

  # Fail closed if either `pre-commit` or `commit-msg` is missing or non-exec.
  findings: list[LeakFinding] = []
  for hook_name in ("pre-commit", "commit-msg"):
    hook_file = active_dir / hook_name
    if not hook_file.is_file() or not os.access(hook_file, os.X_OK):
      findings.append(
          LeakFinding(
              rule_id="CR-GIT-HOOK-INACTIVE",
              category="git",
              severity=GuardrailSeverity.HIGH,
              target=f"git-hook:{hook_name}",
              line_number=1,
              redacted_preview="<inactive>",
              description=(
                  f"Executable Git '{hook_name}' hook is not active; run "
                  "`argus-guardrails --install-hooks`."
              ),
          )
      )
  return tuple(findings)


def bootstrap_clean_git_config(
    repo_root: pathlib.Path,
    default_name: str = "Independent Researcher",
    default_email: str = "researcher@users.noreply.github.com",
) -> tuple[pathlib.Path, pathlib.Path]:
  """Bootstraps local clean-room Git identity defaults and active Git hooks.

  Preserves any existing local `user.name` or `user.email` so a developer's
  personal GitHub identity on their own computer is never overwritten, while
  populating clean defaults on fresh CI checkouts where local config is unset.

  Args:
    repo_root: Root directory of the standalone Argus-SQL Git repository.
    default_name: Fallback local `user.name` applied when unset locally.
    default_email: Fallback local `user.email` applied when unset locally.

  Returns:
    Pair of installed hook paths `(pre_commit_path, commit_msg_path)`.

  Raises:
    GuardrailError: If `.git` is missing or `git config` execution fails.
  """
  # Require a standalone `.git` directory so parent git configs are untouched.
  resolved_root = repo_root.resolve()
  git_dir = resolved_root / ".git"
  if not git_dir.is_dir():
    raise GuardrailError(
        f"Standalone '.git' directory not found inside '{resolved_root}'."
    )

  # Populate local Git author keys only when unset so personal clones keep theirs.
  defaults = (("user.name", default_name), ("user.email", default_email))
  for key, fallback in defaults:
    try:
      # Inspect repo-local `.git/config` without falling back to `--global`.
      get_proc = subprocess.run(
          ["git", "-C", str(resolved_root), "config", "--local", "--get", key],
          capture_output=True,
          text=True,
          check=False,
      )
      # Write clean no-reply identity only when local configuration is blank.
      if not get_proc.stdout.strip():
        set_proc = subprocess.run(
            ["git", "-C", str(resolved_root), "config", "--local", key, fallback],
            capture_output=True,
            text=True,
            check=False,
        )
        if set_proc.returncode != 0:
          raise GuardrailError(
              f"Failed setting local git '{key}': {set_proc.stderr.strip()}"
          )
    except OSError as err:
      raise GuardrailError(f"Failed invoking git config for '{key}'.") from err

  # Install executable `.githooks/` and activate `core.hooksPath` in local Git.
  _LOGGER.info("Bootstrapped clean local Git config in %s", resolved_root)
  return install_git_hooks(resolved_root, configure_git=True)

