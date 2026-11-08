"""Publication-ready LaTeX `booktabs` table and JSON report exporters.

Serializes BenchmarkSuiteReport objects directly into LaTeX tabular fragments
included by `paper/main.tex` as well as structured JSON artifacts for analysis.
"""

import dataclasses
import json

from argus_sql.evaluation.types import BenchmarkSuiteReport


def escape_latex_text(text: str) -> str:
  """Escapes special LaTeX tokens (`&`, `_`, `%`, `#`, `$`) in plain labels.

  Args:
    text: Raw system or model identifier string.

  Returns:
    LaTeX-safe escaped string suitable for tabular cells and captions.
  """
  # Prevent raw `_`, `&`, `%`, `#`, or `$` in model/system IDs from breaking TeX.
  return (
      text.replace("&", r"\&")
      .replace("_", r"\_")
      .replace("%", r"\%")
      .replace("#", r"\#")
      .replace("$", r"\$")
  )


def render_comparative_latex_table(
    clean_report: BenchmarkSuiteReport,
    voice_report: BenchmarkSuiteReport,
    caption: str,
    label: str,
) -> str:
  """Renders a joint Clean + Voice-BIRD `booktabs` LaTeX comparison table.

  Args:
    clean_report: BenchmarkSuiteReport evaluated on clean text queries.
    voice_report: BenchmarkSuiteReport evaluated on Voice-BIRD utterances.
    caption: LaTeX table caption string.
    label: LaTeX reference label (e.g., 'tab:diagnostic_main').

  Returns:
    Complete `table*` LaTeX block ready for `\\input{...}` inclusion.
  """
  # Index Voice-BIRD summaries by system_name to pair Clean and Voice columns.
  voice_by_name = {s.system_name: s for s in voice_report.summaries}

  # Emit standard two-column `booktabs` header scaled cleanly to `\textwidth`.
  lines: list[str] = [
      r"\begin{table*}[t]",
      r"\centering",
      r"\small",
      rf"\caption{{{caption}}}",
      rf"\label{{{label}}}",
      r"\resizebox{\textwidth}{!}{%",
      r"\begin{tabular}{lccccccc}",
      r"\toprule",
      (
          r"\textbf{System / Configuration} & \textbf{Simple EX} & "
          r"\textbf{Mod.\ EX} & \textbf{Chall.\ EX} & "
          r"\textbf{Clean EX (\%)} & \textbf{Clean R-VES} & "
          r"\textbf{Voice EX (\%)} & \textbf{Mean Tokens} \\"
      ),
      r"\midrule",
  ]

  # Bold the final proposed Argus-SQL row when rendering main comparison tables.
  for summary in clean_report.summaries:
    v_sum = voice_by_name.get(summary.system_name)
    # Render `--` gracefully if a custom split omitted Voice-BIRD evaluation.
    v_ex = f"{v_sum.ex_overall:.1f}" if v_sum is not None else "--"
    name_tex = escape_latex_text(summary.system_name)
    is_argus_full = summary.system_name == "Argus-SQL"
    if is_argus_full:
      # Highlight Argus-SQL accuracy columns while keeping token count unbolded.
      row = (
          rf"\textbf{{{name_tex}}} & \textbf{{{summary.ex_simple:.1f}}} & "
          rf"\textbf{{{summary.ex_moderate:.1f}}} & "
          rf"\textbf{{{summary.ex_challenging:.1f}}} & "
          rf"\textbf{{{summary.ex_overall:.1f}}} & "
          rf"\textbf{{{summary.r_ves_overall:.1f}}} & "
          rf"\textbf{{{v_ex}}} & {summary.mean_tokens:.0f} \\"
      )
    else:
      # Standard weight formatting for baselines and leave-one-out ablations.
      row = (
          f"{name_tex} & {summary.ex_simple:.1f} & "
          f"{summary.ex_moderate:.1f} & {summary.ex_challenging:.1f} & "
          f"{summary.ex_overall:.1f} & {summary.r_ves_overall:.1f} & "
          f"{v_ex} & {summary.mean_tokens:.0f} \\\\"
      )
    lines.append(row)

  # Close booktabs table environment cleanly.
  lines.extend([
      r"\bottomrule",
      r"\end{tabular}%",
      r"}",
      r"\end{table*}",
      "",
  ])
  return "\n".join(lines)


def serialize_reports_to_json(
    reports: tuple[BenchmarkSuiteReport, ...],
) -> str:
  """Serializes one or more BenchmarkSuiteReport objects into formatted JSON.

  Args:
    reports: Tuple of BenchmarkSuiteReport instances.

  Returns:
    Pretty-printed JSON string containing summaries and per-task records.
  """
  # Convert nested frozen dataclasses recursively into JSON-ready dictionaries.
  payload = {"suites": [dataclasses.asdict(rep) for rep in reports]}
  return json.dumps(payload, indent=2, sort_keys=False) + "\n"
