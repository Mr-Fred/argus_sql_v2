"""Standalone AST code-quality, docstring, typing, and batch-safety verifier.

Ensures every Python source module inside Argus-SQL adheres to Google Python
Style Guide standards, modular Hexagonal line limits, structured logging,
explicit exception handling, and batch-safe network execution.
"""

import ast
import pathlib

from argus_sql.guardrails.types import CodeQualityViolation

# HTTP client call names prohibited from sequential execution inside loops.
_LOOP_HTTP_ATTRS: frozenset[str] = frozenset({
    "get",
    "post",
    "put",
    "delete",
    "patch",
    "request",
    "urlopen",
})

# Standard Python logging methods that must use lazy %-formatting.
_LOGGER_METHODS: frozenset[str] = frozenset({
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "exception",
})


def _docstring_line_numbers(tree: ast.AST) -> set[int]:
  """Collects 1-based line numbers belonging to module/class/def docstrings."""
  lines: set[int] = set()
  # Walk every scope that can own a leading docstring literal expression.
  for node in ast.walk(tree):
    if isinstance(
        node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ):
      if (
          node.body
          and isinstance(node.body[0], ast.Expr)
          and isinstance(node.body[0].value, ast.Constant)
          and isinstance(node.body[0].value.value, str)
      ):
        doc_node = node.body[0]
        end_no = getattr(doc_node, "end_lineno", doc_node.lineno) or doc_node.lineno
        lines.update(range(doc_node.lineno, end_no + 1))
  return lines


def verify_python_ast_quality(
    file_path: pathlib.Path,
    target_label: str | None = None,
    max_code_lines: int = 800,
    min_comment_ratio: float = 0.08,
) -> tuple[CodeQualityViolation, ...]:
  """Audits a single Python file for modularity, docstrings, types, and safety.

  Args:
    file_path: Path to the `.py` file to inspect.
    target_label: Optional display path used in reported violations.
    max_code_lines: Maximum executable lines allowed in non-test modules.
    min_comment_ratio: Minimum ratio of `#` comment lines to non-empty lines.

  Returns:
    Tuple of CodeQualityViolation items (empty when the file is compliant).
  """
  # Resolve human-readable file label for actionable violation diagnostics.
  label = target_label or str(file_path)
  source = file_path.read_text(encoding="utf-8")
  raw_lines = source.splitlines()
  # Test modules may grow longer to cover thorough edge cases without splitting.
  is_test = file_path.name.startswith("test_") or file_path.name.endswith("_test.py")

  # Parse source into an AST; report syntax errors cleanly instead of crashing.
  try:
    tree = ast.parse(source, filename=label)
  except SyntaxError as err:
    return (
        CodeQualityViolation(
            target=label,
            line_number=err.lineno or 1,
            rule_id="AST-SYNTAX",
            message=f"SyntaxError: {err.msg}",
        ),
    )

  # Accumulate all violations in one pass rather than failing on the first error.
  violations: list[CodeQualityViolation] = []

  # 1. Verify module-level docstring exists at top of file.
  if not ast.get_docstring(tree):
    violations.append(
        CodeQualityViolation(
            target=label,
            line_number=1,
            rule_id="AST-MODULE-DOC",
            message="Missing top-of-file Google-style module docstring.",
        )
    )

  # 2. Enforce executable line budget on production modules (excluding comments/docs).
  # Counting only executable lines ensures rich docstrings are never penalized.
  if not is_test:
    doc_lines = _docstring_line_numbers(tree)
    exec_lines = [
        ln
        for idx, ln in enumerate(raw_lines, start=1)
        if ln.strip() and not ln.strip().startswith("#") and idx not in doc_lines
    ]
    if len(exec_lines) > max_code_lines:
      violations.append(
          CodeQualityViolation(
              target=label,
              line_number=1,
              rule_id="AST-MAX-LINES",
              message=(
                  f"Module has {len(exec_lines)} executable lines "
                  f"(budget: {max_code_lines}); split responsibilities."
              ),
          )
      )

  # 3. Walk AST nodes for classes, functions, exceptions, calls, and loops.
  for node in ast.walk(tree):
    # Every domain class and adapter must document its responsibility & fields.
    if isinstance(node, ast.ClassDef) and not ast.get_docstring(node):
      violations.append(
          CodeQualityViolation(
              target=label,
              line_number=node.lineno,
              rule_id="AST-CLASS-DOC",
              message=f"Class '{node.name}' is missing a docstring.",
          )
      )
    # Every function/method must carry complete PEP 484 types and a docstring.
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      _check_function_signature(label, node, violations)
    # Exception handlers must be explicit and never swallow errors silently.
    elif isinstance(node, ast.ExceptHandler):
      _check_except_handler(label, node, violations)
    # Loops must never execute single-item N+1 HTTP calls sequentially.
    elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
      _check_loop_network_safety(label, node, violations)
    # Dynamic code execution, pickle deserialization, and log f-strings are banned.
    elif isinstance(node, ast.Call):
      _check_call_safety(label, node, violations)

  # 4. Enforce explanatory '#' comment density (>= 8% by default per AGENTS.md).
  non_empty = [ln.strip() for ln in raw_lines if ln.strip()]
  if non_empty:
    comments = [
        ln for ln in non_empty if ln.startswith("#") and not ln.startswith("#!")
    ]
    ratio = len(comments) / len(non_empty)
    if ratio < min_comment_ratio:
      violations.append(
          CodeQualityViolation(
              target=label,
              line_number=1,
              rule_id="AST-COMMENT-RATIO",
              message=(
                  f"Inline '#' comment ratio ({ratio:.1%}) is below "
                  f"{min_comment_ratio:.1%}; explain key invariants."
              ),
          )
      )

  return tuple(violations)


def _check_function_signature(
    label: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    violations: list[CodeQualityViolation],
) -> None:
  """Verifies docstrings and 100% parameter/return type annotations."""
  doc = ast.get_docstring(node)
  if not doc:
    violations.append(
        CodeQualityViolation(
            target=label,
            line_number=node.lineno,
            rule_id="AST-FUNC-DOC",
            message=f"Function '{node.name}' is missing a docstring.",
        )
    )
  else:
    # Public functions accepting arguments must document an Args: block.
    params = [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
    if params and not node.name.startswith("_") and "Args:" not in doc:
      violations.append(
          CodeQualityViolation(
              target=label,
              line_number=node.lineno,
              rule_id="AST-FUNC-ARGS-DOC",
              message=f"Public function '{node.name}' needs an 'Args:' section.",
          )
      )

  # Every function and method must declare an explicit return type annotation.
  if node.returns is None:
    violations.append(
        CodeQualityViolation(
            target=label,
            line_number=node.lineno,
            rule_id="AST-RETURN-TYPE",
            message=f"Function '{node.name}' is missing a return type hint.",
        )
    )

  # Every non-self/cls argument must carry an explicit type annotation.
  all_args = (
      list(node.args.posonlyargs)
      + list(node.args.args)
      + list(node.args.kwonlyargs)
  )
  for arg in all_args:
    if arg.arg not in ("self", "cls") and arg.annotation is None:
      violations.append(
          CodeQualityViolation(
              target=label,
              line_number=node.lineno,
              rule_id="AST-PARAM-TYPE",
              message=f"Parameter '{arg.arg}' in '{node.name}' lacks a type.",
          )
      )


def _check_except_handler(
    label: str,
    handler: ast.ExceptHandler,
    violations: list[CodeQualityViolation],
) -> None:
  """Rejects bare `except:` and silent `except Exception: pass` handlers."""
  # Bare except blocks accidentally catch SystemExit and KeyboardInterrupt.
  if handler.type is None:
    violations.append(
        CodeQualityViolation(
            target=label,
            line_number=handler.lineno,
            rule_id="AST-BARE-EXCEPT",
            message="Bare 'except:' is forbidden; catch specific exceptions.",
        )
    )
  # Silent `except Exception: pass` masks real bugs and corrupts benchmark stats.
  elif (
      isinstance(handler.type, ast.Name)
      and handler.type.id == "Exception"
      and len(handler.body) == 1
      and isinstance(handler.body[0], ast.Pass)
  ):
    violations.append(
        CodeQualityViolation(
            target=label,
            line_number=handler.lineno,
            rule_id="AST-SILENT-EXCEPT",
            message="Silent 'except Exception: pass' is forbidden.",
        )
    )


def _check_loop_network_safety(
    label: str,
    loop_node: ast.For | ast.AsyncFor | ast.While,
    violations: list[CodeQualityViolation],
) -> None:
  """Flags sequential single-item HTTP calls placed directly inside loops."""
  # Prevent N+1 network latency bottlenecks during multi-query benchmarking.
  # All LLM prompts must be dispatched via `BatchLLMPort.batch_complete`.
  for child in ast.walk(loop_node):
    if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
      caller = (
          child.func.value.id if isinstance(child.func.value, ast.Name) else ""
      )
      if child.func.attr in _LOOP_HTTP_ATTRS and caller in {
          "requests",
          "httpx",
          "urllib",
          "request",
          "client",
          "session",
      }:
        violations.append(
            CodeQualityViolation(
                target=label,
                line_number=child.lineno,
                rule_id="AST-N-PLUS-ONE",
                message=f"Sequential HTTP call '{caller}.{child.func.attr}' in loop.",
            )
        )


def _check_call_safety(
    label: str,
    call_node: ast.Call,
    violations: list[CodeQualityViolation],
) -> None:
  """Flags eval/exec/pickle deserialization and f-string logger calls."""
  # Never allow arbitrary code execution or unsafe pickle deserialization.
  if isinstance(call_node.func, ast.Name) and call_node.func.id in {"eval", "exec"}:
    violations.append(
        CodeQualityViolation(
            target=label,
            line_number=call_node.lineno,
            rule_id="AST-BANNED-CALL",
            message=f"Insecure dynamic execution '{call_node.func.id}()' banned.",
        )
    )
  elif isinstance(call_node.func, ast.Attribute):
    caller = (
        call_node.func.value.id
        if isinstance(call_node.func.value, ast.Name)
        else ""
    )
    if caller == "pickle" and call_node.func.attr in {"load", "loads"}:
      violations.append(
          CodeQualityViolation(
              target=label,
              line_number=call_node.lineno,
              rule_id="AST-BANNED-PICKLE",
              message="Insecure 'pickle.load(s)' is banned.",
          )
      )
    # Enforce lazy-% formatting in logging calls so disabled levels cost zero CPU.
    if (
        caller.lower() in {"logger", "_logger", "logging", "log"}
        and call_node.func.attr in _LOGGER_METHODS
        and call_node.args
        and isinstance(call_node.args[0], ast.JoinedStr)
    ):
      violations.append(
          CodeQualityViolation(
              target=label,
              line_number=call_node.lineno,
              rule_id="AST-LOG-FSTRING",
              message="Use lazy '%s' formatting in logger calls, not f-strings.",
          )
      )
