#!/usr/bin/env bash
# Ingests a sanitized `bird_feedback_bundle.json` inside the credential-free
# authoring workspace (via `git pull` in `results/feedback/` or file drop in
# `audit_import/`), regenerates `paper/tables/bird_live.tex` + `triage_report.md`,
# and runs full clean-room & unit test verification with ZERO cloud login.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && (pwd -W 2>/dev/null || pwd))"
cd "${REPO_ROOT}"

PY_PROBE="import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
run_py() {
  if [ -x "${REPO_ROOT}/.venv/Scripts/python.exe" ] && "${REPO_ROOT}/.venv/Scripts/python.exe" -c "${PY_PROBE}" >/dev/null 2>&1; then
    "${REPO_ROOT}/.venv/Scripts/python.exe" "$@"
  elif [ -x "${REPO_ROOT}/.venv/bin/python" ] && "${REPO_ROOT}/.venv/bin/python" -c "${PY_PROBE}" >/dev/null 2>&1; then
    "${REPO_ROOT}/.venv/bin/python" "$@"
  elif command -v python3 >/dev/null 2>&1 && python3 -c "${PY_PROBE}" >/dev/null 2>&1; then
    python3 "$@"
  elif command -v python >/dev/null 2>&1 && python -c "${PY_PROBE}" >/dev/null 2>&1; then
    python "$@"
  elif command -v py >/dev/null 2>&1 && py -3 -c "${PY_PROBE}" >/dev/null 2>&1; then
    py -3 "$@"
  else
    echo "[ERROR] Python 3.11+ interpreter not found on PATH." >&2
    exit 1
  fi
}

PATH_SEP="$(run_py -c 'import os, sys; sys.stdout.write(os.pathsep)')"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+${PATH_SEP}${PYTHONPATH}}"

DEFAULT_QUARANTINE="${REPO_ROOT}/audit_import/bird_feedback_bundle.json"
DEFAULT_TRACKED="${REPO_ROOT}/results/feedback/bird_feedback_bundle.json"

if [ "${#}" -ge 1 ]; then
  BUNDLE_PATH="$1"
  shift
elif [ -f "${DEFAULT_QUARANTINE}" ]; then
  BUNDLE_PATH="${DEFAULT_QUARANTINE}"
else
  BUNDLE_PATH="${DEFAULT_TRACKED}"
fi

if [ ! -f "${BUNDLE_PATH}" ]; then
  echo "[ERROR] Feedback bundle not found at '${BUNDLE_PATH}'." >&2
  echo "        Run 'git pull' or place bird_feedback_bundle.json in audit_import/." >&2
  exit 1
fi

echo "[1/3] Running quarantine clean-room scan on '${BUNDLE_PATH}'..."
run_py -m argus_sql.guardrails.cli \
  --repo-root "${REPO_ROOT}" \
  --scan-path "${BUNDLE_PATH}"

PROMOTE_ARGS=()
NEED_PROMOTE="$(run_py -c 'import pathlib, sys; sys.stdout.write(str(int(pathlib.Path(sys.argv[1]).resolve() != pathlib.Path(sys.argv[2]).resolve())))' "${BUNDLE_PATH}" "${DEFAULT_TRACKED}")"
if [ "${NEED_PROMOTE}" = "1" ]; then
  PROMOTE_ARGS+=("--promote-to" "${DEFAULT_TRACKED}")
fi

echo "[2/3] Verifying SHA-256 digest, updating LaTeX table & generating triage report..."
run_py -m argus_sql.evaluation.feedback_cli ingest \
  --repo-root "${REPO_ROOT}" \
  --bundle "${BUNDLE_PATH}" \
  --report-out "${REPO_ROOT}/results/feedback/triage_report.md" \
  --latex-out-dir "${REPO_ROOT}/paper/tables" \
  "${PROMOTE_ARGS[@]}" \
  "$@"

echo "[3/3] Running full repository clean-room & hermetic unit test verification..."
run_py -m argus_sql.guardrails.cli --repo-root "${REPO_ROOT}" --verify-all
run_py -m unittest discover -s "${REPO_ROOT}/src" -p "*_test.py"

echo "[DONE] Feedback ingested cleanly! Review results/feedback/triage_report.md."
