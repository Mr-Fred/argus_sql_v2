# Argus-SQL

**Finding the Accuracy–Latency Sweet Spot in Natural Language and Voice-to-SQL via Value-Grounded Foreign-Key Closure and Semantic Invariant Verification**

> **Independent Clean-Room Research Notice**: This project is personal, independent scholarly research and open-source engineering. It has zero affiliation with, sponsorship from, or endorsement by any employer, and relies exclusively on public academic publications, public benchmarks (`BIRD-SQL`, `Spider`), and standard open APIs. See [`CLEAN_ROOM_POLICY.md`](CLEAN_ROOM_POLICY.md).

---

## Overview & Research Thesis

As conversational and voice-driven interfaces become a primary way humans interact with software, **Natural Language to SQL (NL2SQL)** is positioned to bridge spontaneous human intent and terabyte-scale analytical databases. However, existing leaderboard pipelines suffer from an acute **accuracy–latency–robustness trilemma**:

- Single-pass raw or lexical-pruning operator baselines (`B0_RawDDL`, `B0+_EnrichedMDDL`, `B1_LexicalCoT`) break on dirty database strings, multi-hop foreign-key bridge tables, and spoken ASR disfluencies.
- Execution-error refiners (`B2_ExecRefiner`) are blind to **silent SQLite semantic bugs** such as integer-division truncation (`SUM(a) / SUM(b) -> 0`) and `NULL`-first ascending `ORDER BY ... LIMIT 1` queries.
- Unweighted self-consistency ensembles (`B3_ValueMajorityVote`) lose whenever $>50\%$ of samples share a dialect bug, while extra-call execution-preview LLM judges (`B4_LLMJudge`, *MCS-SQL / CHESS* selector style) add sequential latency/tokens and remain bounded by `Oracle@K` whenever spoken ASR drift or correlated bugs affect all $K$ candidates.

**Argus-SQL** systematically isolates, compares, and recombines modular NL2SQL operators under a single controlled backbone into a fast, voice-ready four-stage architecture:

1. **Voice & Intent Canonicalizer (`linking/asr_normalizer.py`, `linking/intent_canonicalizer.py`)**: Strips ASR disfluencies/stutters, normalizes spoken cardinals (`"top five"` -> `top 5`), restores multi-sentence capitalization, terminal punctuation, and `evidence`-grounded entity/code casing (`TR004_8_9`, `Marvel Comics`, `DC`), and estimates projection arity across `? Show` / `? Indicate` follow-up sentences, imperative `Rank X by Y` queries (`3`), and `calculation =` single-column locks.
2. **Hierarchy-Aware 3-Gram Cell Value Index, `MIN..MAX` Domain Ranges & Implicit/Junction FK Closure (`linking/value_retriever.py`, `linking/schema_graph.py`, `linking/fk_schema_linker.py`, `evaluation/bird_metadata.py`)**: Grounds messy string cells and 2–3 character domain codes in `< 1 ms`, suppresses unquoted column-qualifier (`(K-12)`) and schema-word prefix noise plus high-cardinality (`City` / `MailCity`) shadows when a canonical low-cardinality dimension (`County`) matches the identical string, hydrates batched read-only `MIN..MAX` temporal/numeric domain bounds (`range: '2012-08-23'..'2012-08-26'`), retains 100% of columns on tables up to 50 columns, and recovers explicit, implicit PK-matched, and M:N junction bridge tables (`99.8%+` table recall and `100%` column recall across `bird_mini_dev`).
3. **Complexity-Gated Diverse Batch Synthesis (`pipelines/argus_pipeline.py`)**: Renders valid composite `PRIMARY KEY (...)` grain DDL, routes simple lookups through a $K=1$ Voice Fast-Path, and dispatches $K=3$ diverse strategies concurrently (`decomposed_cot`, `formula_defensive`, `skeleton_minimal`) for analytical queries.
4. **12-Invariant Semantic & Relational Verifier, Zero-Token Algebraic Repair & Lexicographic Consensus (`verification/`)**: Audits live outcomes in a C-authorizer-locked read-only SQLite sandbox (`sandbox/sqlite_sandbox.py`), deterministically repairs SQLite integer/parenthesized division, `NULL`-first sorts, `ILIKE`, exact-word literal casing, 1-to-many `JOIN` fanout duplicates (`SELECT DISTINCT`), non-unique entity-name `GROUP BY` keys (`GROUP BY T.name` -> `GROUP BY fk_id`), and bare `'YYYY-MM-DD'` predicates on sub-day timestamp columns (`DATE(col)`) in `< 1 ms`, flags 1-to-many filtered `LEFT JOIN` ratios, row-level boolean `AVG(col = 'x')` count errors, and missing `RANK() OVER (...)` window columns for Stage 4b diagnostic repair, and ranks equivalence clusters via lexicographic error-free consensus.

---

### Live Public `BIRD Mini-Dev` Evaluation (`gemini-3.7-flash`, $N = 110$ Stratified Across All 11 Databases)

Evaluated across `1,100` total pipeline runs (`110` Clean + `110` spoken `Voice-BIRD` queries; `37` Simple, `37` Moderate, `36` Challenging; SHA-256 bundle `f4419f770134b474...` in `results/feedback/bird_feedback_bundle.json`, showing the controlled operator IDs recorded at commit `6d25c8f` where `B0_ZeroShotDDL` = profiler-enriched `B0+_EnrichedMDDL`):

| Controlled Operator System | Simple EX | Mod. EX | Chall. EX | Clean EX (%) | Clean R-VES | Voice EX (%) | Mean Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `B0_ZeroShotDDL` (`B0+_EnrichedMDDL`) | 81.1 | 75.7 | 61.1 | 72.7 | 73.0 | **73.6** | 3647 |
| `B1_DINDecomposed` (`B1_LexicalCoT`) | **83.8** | 75.7 | 58.3 | 72.7 | 73.6 | 70.9 | 3632 |
| `B2_MACRefiner` (`B2_ExecRefiner`) | **83.8** | **81.1** | 55.6 | 73.6 | 73.6 | 70.9 | 2889 |
| `B3_CHESSValueConsistency` | 78.4 | 78.4 | 55.6 | 70.9 | 71.6 | 70.9 | 10878 |
| **`Argus-SQL` (Ours)** | 81.1 | 78.4 | **63.9** | **74.5** | **75.2** | 71.8 | 10985 |

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

### 3. Reproduce Controlled Operator & Leave-One-Out Ablation Tables

Runs the 6 controlled backbone-matched operator baselines (`B0_RawDDL`, `B0+_EnrichedMDDL`, `B1_LexicalCoT`, `B2_ExecRefiner`, steelmanned `B3_ValueMajorityVote`, `B4_LLMJudge`), `Argus-SQL`, and 5 leave-one-out component ablations across 24 clean `Mini-BIRD` queries and 24 spoken `Voice-BIRD` utterances, exporting `paper/tables/*.tex` and `results/diagnostic_benchmark.json`:

```bash
PYTHONPATH=src python3 -m argus_sql.evaluation.cli --mode diagnostic
```

| Controlled Operator System | Simple EX | Mod. EX | Chall. EX | Clean EX (%) | Clean R-VES | Voice EX (%) | Mean Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `B0_RawDDL` | 75.0 | 37.5 | 0.0 | 37.5 | 37.5 | 16.7 | 553 |
| `B0+_EnrichedMDDL` | 75.0 | 37.5 | 0.0 | 37.5 | 37.5 | 16.7 | 568 |
| `B1_LexicalCoT` | 100.0 | 25.0 | 0.0 | 41.7 | 41.7 | 25.0 | 676 |
| `B2_ExecRefiner` | 100.0 | 37.5 | 0.0 | 45.8 | 45.8 | 25.0 | 677 |
| `B3_ValueMajorityVote` | 100.0 | 62.5 | 0.0 | 54.2 | 54.2 | 41.7 | 1984 |
| `B4_LLMJudge` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 70.8 | 2204 |
| **`Argus-SQL` (Ours)** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** | **1792** |

> Paired two-sided exact McNemar tests ($N = 48$ Clean + Voice queries): `Argus-SQL` vs. `B3_ValueMajorityVote` $p = 5.96 \times 10^{-8}$; `Argus-SQL` vs. `B4_LLMJudge` $p = 1.56 \times 10^{-2}$ (`1,792` vs. `2,204` tokens, `-18.7%`).

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
  # Full stratified B0..B4 + Argus-SQL evaluation (or -RecomputeArgus after code updates):
  .\scripts\run_personal_eval.ps1 -RecomputeArgus
  ```

- **macOS / Linux / Remote Linux VM (Bash)**:
  ```bash
  # Defaults to GOOGLE_CLOUD_LOCATION=global and ARGUS_LLM_MODEL=gemini-3.7-flash:
  cp config/personal_env.example .env
  # On a headless Linux VM, authenticate via `gcloud auth application-default login --no-launch-browser`
  ARGUS_LIMIT=20 ./scripts/run_personal_eval.sh --only-argus
  # Full stratified B0..B4 + Argus-SQL evaluation (reuses cached B0+/B1/B2 checkpoints):
  ./scripts/run_personal_eval.sh --recompute-argus
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
