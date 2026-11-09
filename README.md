# Argus-SQL

**Finding the Accuracy–Latency Sweet Spot in Natural Language and Voice-to-SQL via Value-Grounded Foreign-Key Closure and Semantic Invariant Verification**

> **Independent Clean-Room Research Notice**: This project is personal, independent scholarly research and open-source engineering. It has zero affiliation with, sponsorship from, or endorsement by any employer, and relies exclusively on public academic publications, public benchmarks (`BIRD-SQL`, `Spider`), and standard open APIs. See [`CLEAN_ROOM_POLICY.md`](CLEAN_ROOM_POLICY.md).

---

## Overview & Research Thesis

As conversational and voice-driven interfaces become a primary way humans interact with software, **Natural Language to SQL (NL2SQL)** is positioned to bridge spontaneous human intent and terabyte-scale analytical databases. However, existing leaderboard pipelines suffer from an acute **accuracy–latency–robustness trilemma**:

- Single-pass zero-shot or lexical-pruning baselines (`B0`, `B1`) break on dirty database strings, multi-hop foreign-key bridge tables, and spoken ASR disfluencies.
- Execution-error refiners (`B2`, e.g., MAC-SQL) are blind to **silent SQLite semantic bugs** such as integer-division truncation (`SUM(a) / SUM(b) -> 0`) and `NULL`-first ascending `ORDER BY ... LIMIT 1` queries.
- Brute-force self-consistency ensembles (`B3`, $K=20$) burn massive token budgets and still lose whenever $>50\%$ of samples share a systematic dialect bug.

**Argus-SQL** systematically implements, compares, and recombines modular NL2SQL operators into a fast, voice-ready four-stage architecture:

1. **Voice & Intent Canonicalizer (`linking/intent_canonicalizer.py`)**: Strips ASR disfluencies/stutters, normalizes spoken cardinals (`"top five"` -> `top 5`), extracts hyphenated/short lowercase domain codes, and estimates complexity & multi-pair/dual-extrema projection arity.
2. **Exact Short-Code + 3-Gram Cell Value Index, `MIN..MAX` Domain Ranges & Implicit/Junction FK Closure (`linking/value_retriever.py`, `linking/schema_graph.py`, `linking/fk_schema_linker.py`, `evaluation/bird_metadata.py`)**: Grounds messy string cells and 2–3 character domain codes in sub-millisecond time, hydrates batched read-only `MIN..MAX` temporal/numeric domain bounds (`range: '2012-08-23'..'2012-08-26'`, `range: 0.0018..1.0`) alongside CSV value codebooks, splits compound identifiers (`gasstations` -> `gas`, `station`), and recovers explicit, implicit PK-matched, and M:N junction bridge tables (`99%+` gold table recall across `bird_mini_dev`).
3. **Complexity-Gated Diverse Batch Synthesis (`pipelines/argus_pipeline.py`)**: Renders valid composite `PRIMARY KEY (...)` grain DDL, routes simple lookups through a $K=1$ Voice Fast-Path, and dispatches $K=3$ diverse strategies concurrently (`decomposed_cot`, `formula_defensive`, `skeleton_minimal`) for analytical queries.
4. **7-Invariant Semantic Verifier, Range-Guided Repair & Error-Free Consensus (`verification/`)**: Audits live outcomes in a C-authorizer-locked read-only SQLite sandbox (`sandbox/sqlite_sandbox.py`), deterministically repairs SQLite integer/parenthesized division, `NULL`-first sorts, `ILIKE`, exact-word literal casing, and 1-to-many `JOIN` fanout duplicates (`SELECT DISTINCT`) in `< 1 ms`, escalates all-empty/all-`NULL` queries with DDL `range:` bounds + sibling failed SQL traces, and ranks equivalence clusters via Invariant-Weighted Consensus (blocking correlated `0`-row clusters from outvoting clean repairs).

---

## Quickstart (Zero Third-Party Dependencies)

Requires only **Python 3.11+** standard library (plus `pdflatex` / `bibtex` if recompiling the paper PDF).

### 1. Enable Clean-Room Git Hooks & Run Guardrail Audit

```bash
git config --local core.hooksPath .githooks
PYTHONPATH=src python3 -m argus_sql.guardrails.cli --verify-all
```

### 2. Run Full Hermetic Unit Test Suite

```bash
PYTHONPATH=src python3 -m unittest discover -s src -p "*_test.py" -v
```

### 3. Reproduce Comparative & Leave-One-Out Ablation Tables

Runs `B0_ZeroShotDDL`, `B1_DINDecomposed`, `B2_MACRefiner`, `B3_CHESSValueConsistency`, `Argus-SQL`, and 5 leave-one-out component ablations across 24 clean `Mini-BIRD` queries and 24 spoken `Voice-BIRD` utterances, exporting `paper/tables/*.tex` and `results/diagnostic_benchmark.json`:

```bash
PYTHONPATH=src python3 -m argus_sql.evaluation.cli --mode diagnostic
```

| System | Simple EX | Mod. EX | Chall. EX | Clean EX (%) | Clean R-VES | Voice EX (%) | Mean Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `B0_ZeroShotDDL` | 75.0 | 37.5 | 0.0 | 37.5 | 37.5 | 16.7 | 375 |
| `B1_DINDecomposed` | 100.0 | 25.0 | 0.0 | 41.7 | 41.7 | 25.0 | 391 |
| `B2_MACRefiner` | 100.0 | 37.5 | 0.0 | 45.8 | 45.8 | 25.0 | 392 |
| `B3_CHESSValueConsistency` | 100.0 | 62.5 | 0.0 | 54.2 | 54.2 | 41.7 | 1175 |
| **`Argus-SQL` (Ours)** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** | **1059** |

> Paired two-sided exact McNemar test (`Argus-SQL` vs. `B3_CHESSValueConsistency`, $N = 48$ Clean + Voice queries): $p = 5.96 \times 10^{-8}$.

### 4. Split-Environment CI/CD & Live Public BIRD-SQL Evaluation

> **Credential Isolation Best Practice**: Keep your authoring workspace **100% credential-free** (no `gcloud` login or API keys). Author code and run hermetic CI here, run live LLM benchmarks on **your personal computer**, and bring the sanitized, SHA-256-signed feedback bundle (`results/feedback/bird_feedback_bundle.json`) back for offline failure triage and LaTeX table updates. Full guide: [`docs/CICD_SPLIT_WORKFLOW.md`](docs/CICD_SPLIT_WORKFLOW.md).

**Step 4a — Download `bird_mini_dev` on Your Personal Computer (500 queries + 11 DBs):**

- **Full Mini-Dev Package (`MINIDEV.zip`, ~1.5 GB)**:
  Download from [bird-bench.github.io](https://bird-bench.github.io/) / [github.com/bird-bench/mini_dev](https://github.com/bird-bench/mini_dev) and extract into `data/bird/mini_dev/`:
  - **Windows (PowerShell)**:
    ```powershell
    New-Item -ItemType Directory -Force -Path data\bird\mini_dev | Out-Null
    Expand-Archive -LiteralPath "$HOME\Downloads\MINIDEV.zip" -DestinationPath data\bird\mini_dev -Force
    ```
  - **macOS / Linux (Bash)**:
    ```bash
    mkdir -p data/bird/mini_dev
    unzip /path/to/MINIDEV.zip -d data/bird/mini_dev
    ```
- **Or Export `mini_dev_sqlite` Questions from HuggingFace Hub (`birdsql/bird_mini_dev`)**:
  ```bash
  uv run --with datasets python -c "import json, pathlib; from datasets import load_dataset; ds = load_dataset('birdsql/bird_mini_dev', split='mini_dev_sqlite'); out = pathlib.Path('data/bird/dev/mini_dev_sqlite.json'); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(list(ds), indent=2), encoding='utf-8'); print('Saved', len(ds), 'tasks to', out)"
  ```

**Step 4b — Run Live Evaluation on Your Personal Computer & Export Sanitized Feedback:**

- **Windows Personal Computer (Antigravity IDE / PowerShell)**:
  ```powershell
  # Open the cloned repo in Antigravity IDE on Windows (reads AGENTS.md automatically).
  # Configure gitignored `.env` ONLY on your personal Windows PC
  # (defaults to GOOGLE_CLOUD_LOCATION=global and ARGUS_LLM_MODEL=gemini-3.7-flash):
  Copy-Item config\personal_env.example .env
  # Edit `.env` (set GOOGLE_CLOUD_PROJECT + run `gcloud auth application-default login`,
  # OR set ARGUS_LLM_API_KEY in `.env`), then run a 20-query smoke slice or full split:
  .\scripts\run_personal_eval.ps1 -Limit 20 -OnlyArgus
  # Or run full B0..B3 + Argus-SQL evaluation:
  .\scripts\run_personal_eval.ps1
  ```

- **macOS / Linux / Remote Linux VM (Bash)**:
  ```bash
  # Defaults to GOOGLE_CLOUD_LOCATION=global and ARGUS_LLM_MODEL=gemini-3.7-flash:
  cp config/personal_env.example .env
  # On a headless Linux VM, authenticate via `gcloud auth application-default login --no-launch-browser`
  ARGUS_LIMIT=20 ./scripts/run_personal_eval.sh --only-argus
  # Or run full B0..B3 + Argus-SQL evaluation:
  ./scripts/run_personal_eval.sh
  ```

This automatically scrubs Windows/POSIX machine paths, verifies clean-room safety, and writes:
- `results/feedback/bird_feedback_bundle.json` (SHA-256-signed telemetry bundle)
- `results/feedback/triage_report.md` (ranked root-cause failure clusters & SQL traces)
- `paper/tables/bird_live.tex` (publication-ready `booktabs` comparison table)

**Step 4c — Bring Feedback Back to the Credential-Free Authoring Workspace:**

```bash
# After `git pull` (or placing bird_feedback_bundle.json in `audit_import/`):
./scripts/ingest_personal_feedback.sh
```

### 5. Compile the Research Paper (`paper/main.pdf`)

```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
