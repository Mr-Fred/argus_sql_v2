# Independent Clean-Room Research & Intellectual Property Policy

## 1. Statement of Independence & Personal Scope

**Argus-SQL** is an independent, personal scientific research project and open-source software implementation authored outside of any corporate employment scope.

- **Non-Affiliation Disclaimer**: The research hypotheses, experimental designs, architectural opinions, benchmark analyses, source code, and manuscript text contained in this repository represent solely the personal, independent scholarly views of the author. Nothing in this repository represents the views, endorsements, roadmap, or proprietary systems of any employer or third-party organization.
- **100% Public Provenance**: Every citation, algorithm, prompt template, evaluation metric, and dataset referenced or implemented in **Argus-SQL** derives exclusively from peer-reviewed public publications, public arXiv preprints, open-source specifications, and public benchmarks (such as **BIRD-SQL** and **Spider**).

## 2. Non-Negotiable Clean-Room Guardrails

Every commit, source file, paper draft (`.tex`, `.bib`), documentation file (`.md`), configuration file, dataset fixture, and commit message in this repository **MUST** pass automated clean-room verification (`python3 -m argus_sql.guardrails.cli`) before it can be committed.

### Prohibited Content Categories

1. **Zero Non-Public / Corporate Internal Information**:
   - Never include internal corporate hostnames, email addresses, intranet URLs, internal issue trackers, internal code-review links, internal repository paths, or unpublished internal system names.
   - Never include local workstation home directory paths or corporate OS usernames in code, tests, logs, paper tables, or Git metadata.
2. **Zero Hardcoded Credentials or Secrets**:
   - Never commit API keys, bearer tokens, OAuth secrets, private keys, or session cookies. All external model API credentials must be injected strictly at runtime via environment variables (`ARGUS_LLM_API_KEY` or `OPENAI_API_KEY`).
3. **Zero Proprietary Datasets**:
   - Only public, permissively licensed or publicly published academic benchmarks (e.g., BIRD-SQL, Spider) and synthetic clean-room databases generated inside this repository may be loaded or evaluated.
4. **Clean Git Author Metadata**:
   - Local Git configuration (`user.name` and `user.email`) must use a personal or GitHub no-reply identity—never a corporate email address.

## 3. Mechanical Enforcement

Clean-room compliance is enforced automatically via three layers:

1. **Git Pre-Commit Hook (`.git/hooks/pre-commit`)**: Scans all repository text files, staged diffs, and Git author configuration for clean-room violations, secret leaks, AST code-quality regressions, and unit-test failures.
2. **Git Commit-Msg Hook (`.git/hooks/commit-msg`)**: Scans every commit message before commit creation and aborts immediately if any prohibited corporate identifier, local workstation path, or secret pattern appears.
3. **Hermetic Test Oracle**: Unit tests in `src/argus_sql/guardrails/clean_room_scanner_test.py` continuously verify both positive leak detection (adversarial payloads) and zero false-positive cleanliness across the entire working tree.
