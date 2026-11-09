#!/usr/bin/env bash
# Runs live BIRD-SQL evaluation on your PERSONAL computer or Linux VM (defaults
# to GOOGLE_CLOUD_LOCATION=global and gemini-3.7-flash on Vertex AI) and emits a
# clean-room-verified, path-scrubbed, SHA-256-signed feedback bundle ready
# to push or copy back to the credential-free authoring workspace.
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

BIRD_DIR="${ARGUS_BIRD_DIR:-data/bird/mini_dev}"
QUESTIONS_FILE="${ARGUS_QUESTIONS:-${ARGUS_QUESTIONS_FILE:-mini_dev_sqlite.json}}"
PROVIDER="${ARGUS_PROVIDER:-auto}"
MODEL="${ARGUS_MODEL:-${ARGUS_LLM_MODEL:-auto}}"
LIMIT="${ARGUS_LIMIT:-22}"
CONCURRENCY="${ARGUS_CONCURRENCY:-16}"
STRATIFIED="${ARGUS_STRATIFIED:-1}"
RECOMPUTE_ARGUS="${ARGUS_RECOMPUTE_ARGUS:-0}"
FRESH_CHECKPOINT="${ARGUS_FRESH_CHECKPOINT:-0}"

CLI_FLAGS=()
if [ "${STRATIFIED}" = "1" ] || [ "${STRATIFIED}" = "true" ]; then
  CLI_FLAGS+=(--stratified)
fi
if [ "${RECOMPUTE_ARGUS}" = "1" ] || [ "${RECOMPUTE_ARGUS}" = "true" ]; then
  CLI_FLAGS+=(--recompute-argus)
fi
if [ "${FRESH_CHECKPOINT}" = "1" ] || [ "${FRESH_CHECKPOINT}" = "true" ]; then
  CLI_FLAGS+=(--fresh-checkpoint)
fi

echo "[1/4] Bootstrapping local clean-room Git hooks..."
run_py -m argus_sql.guardrails.cli --repo-root "${REPO_ROOT}" --bootstrap-git --verify-all

if [ ! -d "${BIRD_DIR}" ]; then
  echo "[ERROR] BIRD dataset directory '${BIRD_DIR}' not found on this machine." >&2
  echo "        Extract MINIDEV.zip into '${BIRD_DIR}' or set ARGUS_BIRD_DIR." >&2
  exit 1
fi

echo "[2/4] Running live BIRD-SQL evaluation (provider=${PROVIDER}, model=${MODEL}, limit=${LIMIT}, concurrency=${CONCURRENCY}, stratified=${STRATIFIED})..."
run_py -m argus_sql.evaluation.cli \
  --mode bird \
  --bird-dir "${BIRD_DIR}" \
  --questions-filename "${QUESTIONS_FILE}" \
  --provider "${PROVIDER}" \
  --model "${MODEL}" \
  --limit "${LIMIT}" \
  --task-concurrency "${CONCURRENCY}" \
  --checkpoint-dir "${REPO_ROOT}/results/live_runs" \
  --latex-out-dir "${REPO_ROOT}/paper/tables" \
  --feedback-out-dir "${REPO_ROOT}/results/feedback" \
  ${CLI_FLAGS[@]+"${CLI_FLAGS[@]}"} \
  "$@"

echo "[3/4] Scanning exported feedback bundle & LaTeX table for zero leaks..."
run_py -m argus_sql.guardrails.cli \
  --repo-root "${REPO_ROOT}" \
  --scan-path "${REPO_ROOT}/results/feedback"
run_py -m argus_sql.guardrails.cli \
  --repo-root "${REPO_ROOT}" \
  --scan-path "${REPO_ROOT}/paper/tables/bird_live.tex"

echo "[4/4] Verifying SHA-256 cryptographic integrity of feedback bundle..."
run_py -m argus_sql.evaluation.feedback_cli verify \
  --repo-root "${REPO_ROOT}" \
  --feedback-dir "${REPO_ROOT}/results/feedback"

cat <<'EOF'

[DONE] Sanitized feedback bundle verified and ready to bring back!
  - Bundle JSON : results/feedback/bird_feedback_bundle.json
  - Triage MD   : results/feedback/triage_report.md
  - LaTeX Table : paper/tables/bird_live.tex

Next step (Git sync):
  git add results/feedback/bird_feedback_bundle.json results/feedback/triage_report.md paper/tables/bird_live.tex
  git commit -m "chore(eval): publish sanitized BIRD feedback bundle"
  git push
EOF
