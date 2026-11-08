# AI Agent & Contributor Rules for Argus-SQL

## 1. Prime Directive: Strict Clean-Room Personal Research

> [!CAUTION]
> **Argus-SQL** is strictly personal, independent academic research. **NEVER** write, reference, import, or mention any internal corporate tool, intranet shortlink, corporate email address, internal repository path, local workstation path, or non-public information anywhere in this repository (code, comments, tests, commit messages, or LaTeX paper files). Read `CLEAN_ROOM_POLICY.md` first.

## 2. Software Engineering & Architectural Invariants

1. **Hexagonal Modular Architecture (No God Modules)**:
   - Decompose every domain package under `src/argus_sql/` into cohesive, single-responsibility modules (`<= 320` executable code lines per non-test module).
   - Separate pure domain logic (`contracts`, `linking`, `verification`, `evaluation`) from outer I/O adapters (`sandbox`, `adapters`) and thin CLI entrypoints.
   - Include `types.py`, `exceptions.py`, domain modules, and co-located `*_test.py` unit tests in every package.
2. **Documentation & Readability**:
   - Every module, class, and function must include a complete Google-style Python docstring (`Args:`, `Returns:`, `Raises:`) and clear inline `#` comments (`>= 8%` comment density) explaining *why* architectural, statistical, and security invariants exist.
   - Annotate 100% of function parameter and return types.
3. **Structured Logging & Explicit Exceptions**:
   - Use module-level loggers (`_LOGGER = logging.getLogger(__name__)`) with lazy `%s` / `%d` formatting (never f-strings inside logger calls).
   - Catch specific exception types and chain domain exceptions via `raise DomainError(...) from err` (never bare `except:` or silent `except Exception: pass`).
4. **Security & Batch Network Optimization**:
   - Execute generated SQL strictly inside `SQLiteReadOnlySandbox` (`mode=ro`, `sqlite3` authorizer allow-listing `SELECT`/`READ` only, and VM step timeouts).
   - Never invoke HTTP/LLM requests sequentially inside `for`/`while` loops; always batch requests through `BatchLLMPort.batch_complete(...)` with SHA-256 prompt deduplication.
5. **Mandatory Pre-Commit Verification**:
   - Run `PYTHONPATH=src python3 -m argus_sql.guardrails.cli --verify-all` and `PYTHONPATH=src python3 -m unittest discover -s src -p "*_test.py"` before every commit.
