"""Persistent JSONL checkpointing and stratified multi-DB sampling for BIRD-SQL.

Persists every completed `TaskEvalRecord` immediately to `results/live_runs/`
with `os.fsync` durability so long comparative runs survive 10-minute remote VM
windows without losing work, and selects deterministic difficulty-balanced
slices across all 11 BIRD databases.
"""

from collections.abc import Sequence
import json
import logging
import os
import pathlib
import re
import threading

from argus_sql.contracts.types import NLQueryTask
from argus_sql.contracts.types import QueryDifficulty
from argus_sql.evaluation.exceptions import EvaluationError
from argus_sql.evaluation.feedback_bundle import sanitize_diagnostic_text
from argus_sql.evaluation.types import TaskEvalRecord

# Module-level logger following Google Python Style Guide lazy-% conventions.
_LOGGER = logging.getLogger(__name__)

# Allow-listed characters for safe checkpoint filenames inside `results/live_runs`.
_SAFE_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")

# Canonical difficulty rotation order ensuring all 3 BIRD tiers appear in slices.
_DIFFICULTY_ROTATION: tuple[QueryDifficulty, ...] = (
    QueryDifficulty.SIMPLE,
    QueryDifficulty.MODERATE,
    QueryDifficulty.CHALLENGING,
)


def _interleave_by_difficulty(
    db_tasks: Sequence[NLQueryTask],
) -> list[NLQueryTask]:
  """Orders a single database's tasks by cycling Simple, Moderate, Challenging."""
  # Group tasks by difficulty tier while preserving original JSON order within tiers.
  buckets: dict[QueryDifficulty, list[NLQueryTask]] = {
      tier: [] for tier in _DIFFICULTY_ROTATION
  }
  for task in db_tasks:
    buckets.setdefault(task.difficulty, []).append(task)

  # Interleave one task per difficulty tier in round-robin order until exhausted.
  interleaved: list[NLQueryTask] = []
  max_len = max((len(b) for b in buckets.values()), default=0)
  for idx in range(max_len):
    for tier in _DIFFICULTY_ROTATION:
      tier_list = buckets.get(tier, [])
      if idx < len(tier_list):
        interleaved.append(tier_list[idx])
  return interleaved


def select_stratified_bird_tasks(
    tasks: Sequence[NLQueryTask],
    limit: int,
) -> tuple[NLQueryTask, ...]:
  """Selects a deterministic, difficulty-balanced slice across all databases.

  Instead of taking the first `limit` tasks (which in BIRD `mini_dev_sqlite.json`
  hits only the first database), this function round-robins across all sorted
  `db_id`s while rotating through `simple`, `moderate`, and `challenging` tiers.

  Args:
    tasks: Full sequence of materialized `NLQueryTask` items.
    limit: Target number of tasks to select (`<= 0` returns all `tasks`).

  Returns:
    Tuple of up to `limit` tasks stratified across all databases and tiers.
  """
  if limit <= 0 or limit >= len(tasks):
    return tuple(tasks)

  # Partition tasks by `db_id` and interleave difficulty tiers within each DB.
  by_db: dict[str, list[NLQueryTask]] = {}
  for task in tasks:
    by_db.setdefault(task.db_id, []).append(task)

  # Sort database identifiers alphabetically so stratified slices are 100% deterministic.
  sorted_dbs = sorted(by_db.keys())
  ordered_per_db: dict[str, list[NLQueryTask]] = {
      db_id: _interleave_by_difficulty(by_db[db_id]) for db_id in sorted_dbs
  }

  # Rotate starting difficulty tier across databases so even 1 task/DB covers
  # Simple, Moderate, and Challenging queries evenly across the 11 databases.
  selected: list[NLQueryTask] = []
  round_idx = 0
  while len(selected) < limit:
    added_in_round = False
    for db_offset, db_id in enumerate(sorted_dbs):
      db_queue = ordered_per_db[db_id]
      if not db_queue:
        continue
      # On the first pass, pick a task matching `(db_offset + round_idx) % 3`
      # difficulty if available so a 1-per-DB slice spans all 3 difficulty tiers.
      target_tier = _DIFFICULTY_ROTATION[
          (db_offset + round_idx) % len(_DIFFICULTY_ROTATION)
      ]
      match_pos = next(
          (i for i, t in enumerate(db_queue) if t.difficulty == target_tier),
          0,
      )
      selected.append(db_queue.pop(match_pos))
      added_in_round = True
      if len(selected) >= limit:
        break
    if not added_in_round:
      break
    round_idx += 1

  return tuple(selected)


def resolve_checkpoint_path(
    checkpoint_dir: pathlib.Path,
    split_name: str,
    provider: str,
    model_label: str,
    limit: int,
    stratified: bool,
) -> pathlib.Path:
  """Constructs a confined JSONL checkpoint path under `checkpoint_dir`.

  Args:
    checkpoint_dir: Root directory for live checkpoint files (`results/live_runs`).
    split_name: Dataset split filename (e.g., `'mini_dev_sqlite.json'`).
    provider: LLM provider slug (`'vertex'` or `'openai'`).
    model_label: Evaluated model slug (e.g., `'gemini-3.7-flash'`).
    limit: Task cap (`0` for full split).
    stratified: Whether multi-DB stratified sampling is active.

  Returns:
    Resolved `pathlib.Path` inside `checkpoint_dir`.

  Raises:
    EvaluationError: If the constructed path escapes `checkpoint_dir`.
  """
  root = checkpoint_dir.resolve()
  # Sanitize all path components to alphanumeric/hyphen slugs to block traversal.
  clean_split = _SAFE_SLUG_RE.sub("_", pathlib.Path(split_name).stem).strip("_")
  clean_prov = _SAFE_SLUG_RE.sub("_", provider.lower()).strip("_") or "auto"
  clean_model = _SAFE_SLUG_RE.sub("_", model_label).strip("_") or "auto"
  slice_tag = f"strat{limit}" if stratified and limit > 0 else f"limit{limit}"
  filename = f"ckpt_{clean_split}_{clean_prov}_{clean_model}_{slice_tag}.jsonl"

  # Verify the resolved path stays strictly confined inside `checkpoint_dir`.
  target = (root / filename).resolve()
  if not target.is_relative_to(root):
    raise EvaluationError("Checkpoint path escaped allowed checkpoint directory.")
  return target


class LiveRunCheckpointer:
  """Thread-safe append-only JSONL checkpointer for `TaskEvalRecord` items.

  Attributes:
    checkpoint_path: Backing `.jsonl` file path on persistent disk.
  """

  def __init__(self, checkpoint_path: pathlib.Path) -> None:
    """Initializes the checkpointer and loads any existing completed records.

    Args:
      checkpoint_path: Path to the `.jsonl` checkpoint file.
    """
    self.checkpoint_path = checkpoint_path
    # Serialize cache mutations and disk appends across worker threads.
    self._lock = threading.Lock()
    # Map `(suite_name, system_name, question_id)` to completed `TaskEvalRecord`.
    self._cache: dict[tuple[str, str, str], TaskEvalRecord] = {}
    self._load_existing()

  def _load_existing(self) -> None:
    """Loads previously completed `(suite, system, question_id)` records."""
    if not self.checkpoint_path.is_file() or self.checkpoint_path.is_symlink():
      return
    loaded = 0
    try:
      raw_lines = self.checkpoint_path.read_text(encoding="utf-8").splitlines()
    except OSError as err:
      _LOGGER.warning("Could not read checkpoint file: %s", err)
      return

    # Parse each valid JSON line; ignore truncated tail lines from hard kills.
    for line in raw_lines:
      stripped = line.strip()
      if not stripped:
        continue
      try:
        obj = json.loads(stripped)
        if not isinstance(obj, dict):
          continue
        suite_name = str(obj["suite_name"])
        system_name = str(obj["system_name"])
        rec = TaskEvalRecord(
            question_id=str(obj["question_id"]),
            db_id=str(obj["db_id"]),
            difficulty=QueryDifficulty(str(obj["difficulty"])),
            is_voice=bool(obj["is_voice"]),
            ex_correct=bool(obj["ex_correct"]),
            r_ves_score=float(obj["r_ves_score"]),
            latency_ms=float(obj["latency_ms"]),
            total_tokens=int(obj["total_tokens"]),
            failure_category=str(obj["failure_category"]),
            # Sanitize SQL and error strings on reload for defense in depth.
            predicted_sql=sanitize_diagnostic_text(
                str(obj.get("predicted_sql", "")), 2000
            ),
            error_summary=sanitize_diagnostic_text(
                str(obj.get("error_summary", "")), 240, single_line=True
            ),
            invariant_codes=tuple(
                str(x) for x in obj.get("invariant_codes", [])
            ),
            was_repaired=bool(obj.get("was_repaired", False)),
            selected_strategy=sanitize_diagnostic_text(
                str(obj.get("selected_strategy", "")), 80, single_line=True
            ),
        )
        # Index by `(suite_name, system_name, question_id)` for O(1) lookup.
        self._cache[(suite_name, system_name, rec.question_id)] = rec
        loaded += 1
      except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        # Ignore malformed or truncated tail lines caused by abrupt VM stops.
        continue

    if loaded > 0:
      _LOGGER.info("Loaded %d checkpointed task evaluations.", loaded)

  def get(
      self, suite_name: str, system_name: str, question_id: str
  ) -> TaskEvalRecord | None:
    """Returns a cached `TaskEvalRecord` if already completed on disk.

    Args:
      suite_name: Benchmark suite label ('BIRD-Dev Clean' or 'BIRD-Dev Voice').
      system_name: Evaluated pipeline identifier ('Argus-SQL', 'B0_ZeroShotDDL').
      question_id: Unique task question identifier.

    Returns:
      Cached `TaskEvalRecord` if present, otherwise `None`.
    """
    # Acquire lock so concurrent worker threads read a consistent snapshot.
    with self._lock:
      return self._cache.get((suite_name, system_name, question_id))

  def record(
      self, suite_name: str, system_name: str, rec: TaskEvalRecord
  ) -> None:
    """Persists a completed `TaskEvalRecord` to disk with immediate `fsync`.

    Args:
      suite_name: Benchmark suite label ('BIRD-Dev Clean' or 'BIRD-Dev Voice').
      system_name: Evaluated pipeline identifier.
      rec: Completed `TaskEvalRecord` to append to the JSONL checkpoint.
    """
    # Sanitize diagnostic strings before persisting so checkpoint files stay clean.
    payload = {
        "suite_name": sanitize_diagnostic_text(suite_name, 120, single_line=True),
        "system_name": sanitize_diagnostic_text(system_name, 120, single_line=True),
        "question_id": sanitize_diagnostic_text(
            rec.question_id, 80, single_line=True
        ),
        "db_id": sanitize_diagnostic_text(rec.db_id, 80, single_line=True),
        "difficulty": rec.difficulty.value,
        "is_voice": rec.is_voice,
        "ex_correct": rec.ex_correct,
        "r_ves_score": rec.r_ves_score,
        "latency_ms": rec.latency_ms,
        "total_tokens": rec.total_tokens,
        "failure_category": rec.failure_category,
        "predicted_sql": sanitize_diagnostic_text(rec.predicted_sql, 2000),
        "error_summary": sanitize_diagnostic_text(
            rec.error_summary, 240, single_line=True
        ),
        "invariant_codes": list(rec.invariant_codes),
        "was_repaired": rec.was_repaired,
        "selected_strategy": sanitize_diagnostic_text(
            rec.selected_strategy, 80, single_line=True
        ),
    }
    # Serialize canonical single-line JSON record for atomic append.
    line = json.dumps(payload, sort_keys=True) + "\n"
    # Synchronize in-memory cache update and `fsync` across worker threads.
    with self._lock:
      self._cache[(suite_name, system_name, rec.question_id)] = rec
      self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
      with self.checkpoint_path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
