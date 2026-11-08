"""Security and resource-limit policy contracts for SQLite execution."""

import dataclasses
import pathlib


# Immutable sandbox policy bounding filesystem scope, CPU steps, and row caps.
@dataclasses.dataclass(frozen=True)
class SandboxPolicy:
  """Resource and security boundaries enforced on every SQLite execution.

  Attributes:
    allowed_db_root: Resolved directory under which `.sqlite` files must reside.
    max_vm_steps: Maximum SQLite virtual machine opcodes per query execution.
    timeout_ms: Maximum wall-clock milliseconds before aborting execution.
    max_returned_rows: Maximum result tuples materialized in memory.
  """

  # Confine SQLite opens strictly beneath `allowed_db_root` after resolving.
  allowed_db_root: pathlib.Path
  # Match official BIRD-SQL's 60s timeout (~5B VM steps) for unindexed CAST
  # joins while deterministically aborting infinite Cartesian cross-joins.
  max_vm_steps: int = 5_000_000_000
  timeout_ms: float = 60_000.0
  max_returned_rows: int = 10_000
