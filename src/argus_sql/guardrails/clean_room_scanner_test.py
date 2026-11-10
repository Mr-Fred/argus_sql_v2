"""Unit tests for Argus-SQL clean-room IP guardrails, PDF scan, and AST gates."""

import contextlib
import io
import pathlib
import subprocess
import tempfile
import unittest
import zlib

from argus_sql.guardrails.clean_room_scanner import scan_pdf_for_leaks
from argus_sql.guardrails.clean_room_scanner import scan_text_for_leaks
from argus_sql.guardrails.cli import main as guardrails_cli_main
from argus_sql.guardrails.code_quality_verifier import load_compliance_policy
from argus_sql.guardrails.code_quality_verifier import verify_python_ast_quality
from argus_sql.guardrails.exceptions import GuardrailError
from argus_sql.guardrails.hook_installer import bootstrap_clean_git_config
from argus_sql.guardrails.hook_installer import install_git_hooks
from argus_sql.guardrails.hook_installer import verify_git_hooks_active
from argus_sql.guardrails.repository_auditor import audit_target_path
from argus_sql.guardrails.repository_auditor import run_repository_audit


class CleanRoomScannerTest(unittest.TestCase):
  """Verifies prohibited leak detection, PDF scans, hooks, and clean repo state."""

  def test_detects_prohibited_corporate_domains_and_shortlinks(self) -> None:
    """Assembled corporate emails, shortlinks, and depot paths must be caught."""
    # Assemble adversarial test strings at runtime so this test file stays clean.
    corp_email = "".join(("author@", "g", "oogle", ".com"))
    short_link = "".join(("See ", "go", "/", "internal-design-doc"))
    bug_link = "".join(("Tracked in ", "b", "/", "12345678"))
    mono_marker = "".join(("Imported from ", "g", "oogle3", "/pkg"))
    payload = "\n".join((corp_email, short_link, bug_link, mono_marker))

    # Verify all corporate leak categories fire as expected.
    findings = scan_text_for_leaks(payload, target_label="synthetic.md")
    rule_ids = {item.rule_id for item in findings}
    self.assertIn("CR-CORP-DOMAIN", rule_ids)
    self.assertIn("CR-INTERNAL-SHORTLINK", rule_ids)
    self.assertIn("CR-INTERNAL-INFRA", rule_ids)
    # Redacted preview must mask sensitive substrings rather than echoing them.
    for item in findings:
      self.assertIn("[REDACTED:", item.redacted_preview)

  def test_detects_secret_api_keys_and_local_home_directory(self) -> None:
    """API keys, OAuth tokens, private keys, and home paths trigger alerts."""
    # Construct synthetic secret, OAuth token, and cross-platform home shapes.
    fake_sk = "sk-" + ("A1b2C3d4" * 4)
    fake_oauth = "ya29." + ("a0AfB_byC1" * 3)
    fake_pem = "".join(("-----BEGIN ", "RSA PRIVATE", " KEY-----"))
    home_dir = str(pathlib.Path.home().resolve())
    mac_path = "".join(("/", "Users", "/devbox/project/app.py"))
    win_path = "".join(("C:\\", "Users", "\\devbox\\project\\"))
    win_json_space = "".join(("C:\\\\", "Users", "\\\\First Last\\\\db.sqlite"))
    payload = (
        f"token = '{fake_sk}'\n{fake_pem}\npath = '{home_dir}/data'\n"
        f"mac = '{mac_path}'\nwin = '{win_path}'\n"
    )

    # Confirm secret, OAuth bearer, runtime $HOME, and laptop path rules fire.
    findings = scan_text_for_leaks(payload, target_label="config.toml")
    rule_ids = {item.rule_id for item in findings}
    self.assertIn("CR-SECRET-API-KEY", rule_ids)
    self.assertIn("CR-SECRET-PRIVATE-KEY", rule_ids)
    self.assertIn("CR-LOCAL-HOME-PATH", rule_ids)
    self.assertIn("CR-PORTABLE-HOME-PATH", rule_ids)
    json_win_hits = scan_text_for_leaks(win_json_space, target_label="bundle.json")
    self.assertTrue(any(f.rule_id == "CR-PORTABLE-HOME-PATH" for f in json_win_hits))
    oauth_hits = scan_text_for_leaks(fake_oauth, target_label="cassette.yaml")
    self.assertEqual(oauth_hits[0].rule_id, "CR-SECRET-API-KEY")
    # Lowercase REST routes (`/users/42/orders`) must never false-positive.
    self.assertEqual(scan_text_for_leaks("GET /users/42/orders", "api.md"), ())

  def test_scans_compressed_pdf_streams_for_hidden_leaks(self) -> None:
    """FlateDecode compressed PDF streams containing leaks must be flagged."""
    # Build a minimal synthetic PDF embedding a zlib-compressed stream leak.
    hidden_leak = "".join(("Contact user@", "g", "oogle", ".com")).encode("latin-1")
    compressed = zlib.compress(hidden_leak)
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nstream\n" + compressed + b"\nendstream\n"

    with tempfile.TemporaryDirectory() as tmp_dir:
      dirty_pdf = pathlib.Path(tmp_dir) / "draft.pdf"
      dirty_pdf.write_bytes(pdf_bytes)
      findings = scan_pdf_for_leaks(dirty_pdf, target_label="draft.pdf")
      self.assertTrue(any(f.rule_id == "CR-CORP-DOMAIN" for f in findings))

  def test_allows_clean_academic_sql_and_bibtex(self) -> None:
    """Public SQL queries and arXiv citations must produce zero false positives."""
    # Representative BIRD-SQL query and BibTeX citation from public literature.
    clean_academic = (
        "SELECT CAST(SUM(free_meals) AS REAL) / NULLIF(SUM(enrollment), 0)\n"
        "FROM schools WHERE state = 'California';\n"
        "@article{li2023bird,\n"
        "  title={Can LLM Already Serve as A Database Interface?},\n"
        "  journal={NeurIPS},\n"
        "  year={2023}\n"
        "}\n"
    )
    findings = scan_text_for_leaks(clean_academic, target_label="paper.tex")
    self.assertEqual(findings, ())

  def test_ast_quality_verifier_catches_bad_python_patterns(self) -> None:
    """Missing docstrings, missing types, bare except, and N+1 loops are flagged."""
    # Synthetic bad module intentionally violating six AST quality rules.
    bad_code = (
        "def broken(x):\n"
        "  try:\n"
        "    for url in x:\n"
        "      requests.get(url)\n"
        "  except:\n"
        "    pass\n"
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
      bad_file = pathlib.Path(tmp_dir) / "sample_bad.py"
      bad_file.write_text(bad_code, encoding="utf-8")
      violations = verify_python_ast_quality(bad_file)
      codes = {v.rule_id for v in violations}
      self.assertIn("AST-MODULE-DOC", codes)
      self.assertIn("AST-FUNC-DOC", codes)
      self.assertIn("AST-RETURN-TYPE", codes)
      self.assertIn("AST-PARAM-TYPE", codes)
      self.assertIn("AST-BARE-EXCEPT", codes)
      self.assertIn("AST-N-PLUS-ONE", codes)

  def test_bootstrap_git_and_quarantine_path_scanner(self) -> None:
    """CI git bootstrapper preserves personal email and scans audit_import."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      clone_root = pathlib.Path(tmp_dir)
      subprocess.run(
          ["git", "init", str(clone_root)],
          capture_output=True,
          text=True,
          check=True,
      )
      # Populate default clean identity on a fresh CI clone and verify audit.
      bootstrap_clean_git_config(clone_root)
      self.assertTrue(run_repository_audit(clone_root).passed)

      # Ensure existing custom personal email is preserved across re-runs.
      subprocess.run(
          [
              "git",
              "-C",
              str(clone_root),
              "config",
              "--local",
              "user.email",
              "scholar@users.noreply.github.com",
          ],
          check=True,
      )
      bootstrap_clean_git_config(clone_root)
      email_proc = subprocess.run(
          ["git", "-C", str(clone_root), "config", "--local", "--get", "user.email"],
          capture_output=True,
          text=True,
          check=True,
      )
      self.assertEqual(email_proc.stdout.strip(), "scholar@users.noreply.github.com")

      # Gitignored IDE metadata (`.antigravity`, `.vscode`) and `data/bird` CSVs
      # must not cause false-positive failures during `run_repository_audit`.
      win_interp = "".join(("C:\\\\", "Users", "\\\\alice\\\\python.exe"))
      ide_cfg = clone_root / ".antigravity" / "settings.json"
      ide_cfg.parent.mkdir(parents=True)
      ide_cfg.write_text(f'{{"python": "{win_interp}"}}', encoding="utf-8")
      bird_csv = clone_root / "data" / "bird" / "mini_dev" / "schools.csv"
      bird_csv.parent.mkdir(parents=True)
      bird_csv.write_text(f"note,{win_interp}\n", encoding="utf-8")
      self.assertTrue(run_repository_audit(clone_root).passed)

      # Relative `--check-commit-msg .git/COMMIT_EDITMSG` resolves against `--repo-root`.
      from argus_sql.guardrails.cli import main as guardrails_cli_main

      msg_path = clone_root / ".git" / "COMMIT_EDITMSG"
      msg_path.write_text("feat: add Windows Antigravity workflow\n", encoding="utf-8")
      self.assertEqual(
          guardrails_cli_main([
              "--repo-root",
              str(clone_root),
              "--check-commit-msg",
              ".git/COMMIT_EDITMSG",
          ]),
          0,
      )

      # Verify `audit_target_path` scans quarantined `audit_import/` explicitly.
      q_dir = clone_root / "audit_import"
      q_dir.mkdir()
      dirty_bundle = q_dir / "dirty.json"
      dirty_bundle.write_text("sk-" + ("Z9y8X7w6" * 4), encoding="utf-8")
      self.assertFalse(audit_target_path(clone_root, dirty_bundle).passed)
      # Reject path traversal outside `clone_root`.
      with self.assertRaises(GuardrailError):
        audit_target_path(clone_root, pathlib.Path("../outside.json"))

  def test_hook_installer_and_live_repository_audit(self) -> None:
    """Hook installer activates hooks and live repo passes clean-room audit."""
    # Verify unhooked repo fails `verify_git_hooks_active` until installed.
    with tempfile.TemporaryDirectory() as tmp_dir:
      fake_repo = pathlib.Path(tmp_dir)
      (fake_repo / ".git" / "hooks").mkdir(parents=True)
      self.assertEqual(len(verify_git_hooks_active(fake_repo)), 2)
      pre_hook, msg_hook = install_git_hooks(fake_repo, configure_git=True)
      self.assertTrue(pre_hook.is_file())
      self.assertTrue(msg_hook.is_file())
      self.assertEqual(verify_git_hooks_active(fake_repo), ())

    # Audit the actual Argus-SQL repository containing this test file.
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    report = run_repository_audit(repo_root)
    self.assertTrue(
        report.passed,
        msg=f"Leaks={report.leak_findings}, Quality={report.quality_violations}",
    )

  def test_healthy_range_advisory_vs_ceiling_and_policy_overrides(self) -> None:
    """Stretch modules emit non-blocking WARNING; ceiling violations fail."""

    def _build_module_with_exec_lines(exec_count: int) -> str:
      """Builds a compliant synthetic module with `exec_count` code lines."""
      header = (
          '"""Synthetic module verifying healthy-range AST thresholds."""\n'
          "# Ensure inline comment density clears >= 8% threshold.\n"
      )
      # Add 1 comment per 8 executable lines so AST-COMMENT-RATIO stays clean.
      body: list[str] = []
      for idx in range(exec_count):
        if idx % 8 == 0:
          body.append(f"# Invariant marker {idx}.")
        body.append(f"CONST_{idx}: int = {idx}")
      return header + "\n".join(body) + "\n"

    with tempfile.TemporaryDirectory() as tmp_dir:
      repo = pathlib.Path(tmp_dir)
      subprocess.run(
          ["git", "init", str(repo)],
          capture_output=True,
          text=True,
          check=True,
      )
      bootstrap_clean_git_config(repo)
      (repo / ".copilot_compliance.json").write_text(
          (
              '{\n'
              '  "healthy_min_lines": 80,\n'
              '  "healthy_max_lines": 800,\n'
              '  "ceiling_max_lines": 1000,\n'
              '  "min_comment_ratio": 0.08,\n'
              '  "file_max_lines_overrides": {\n'
              '    "pkg/exempt.py": 1050,\n'
              '    "pkg/strict.py": 250\n'
              '  }\n'
              '}\n'
          ),
          encoding="utf-8",
      )
      policy = load_compliance_policy(repo)
      self.assertEqual(policy.healthy_max_lines, 800)
      self.assertEqual(policy.ceiling_max_lines, 1000)
      self.assertEqual(policy.override_for("pkg/exempt.py"), 1050)
      self.assertEqual(policy.override_for("pkg/strict.py"), 250)

      pkg_dir = repo / "pkg"
      pkg_dir.mkdir()
      # Exact boundary 800 executable lines: inside healthy range (0 size findings).
      healthy_mod = pkg_dir / "healthy.py"
      healthy_mod.write_text(_build_module_with_exec_lines(800), encoding="utf-8")
      self.assertEqual(verify_python_ast_quality(healthy_mod), ())

      # 850 executable lines: in advisory zone (WARNING only, CLI exits 0).
      stretch_mod = pkg_dir / "stretch.py"
      stretch_mod.write_text(_build_module_with_exec_lines(850), encoding="utf-8")
      stretch_viols = verify_python_ast_quality(stretch_mod)
      self.assertEqual(len(stretch_viols), 1)
      self.assertEqual(stretch_viols[0].rule_id, "AST-SIZE-ADVISORY")
      self.assertEqual(stretch_viols[0].severity, "WARNING")
      advisory_report = run_repository_audit(repo)
      self.assertTrue(advisory_report.passed)
      self.assertEqual(len(advisory_report.advisory_quality_violations), 1)

      # Verify CLI surfaces [WARN] [AST-SIZE-ADVISORY] on stdout and returns 0.
      out_buf = io.StringIO()
      err_buf = io.StringIO()
      with (
          contextlib.redirect_stdout(out_buf),
          contextlib.redirect_stderr(err_buf),
      ):
        cli_rc = guardrails_cli_main(["--repo-root", str(repo), "--verify-all"])
      self.assertEqual(cli_rc, 0)
      self.assertIn("[WARN] [AST-SIZE-ADVISORY]", out_buf.getvalue())
      self.assertIn("[PASS]", out_buf.getvalue())
      self.assertEqual(err_buf.getvalue(), "")

      # 1020 executable lines with override at 1050 passes; without override fails.
      exempt_mod = pkg_dir / "exempt.py"
      exempt_mod.write_text(_build_module_with_exec_lines(1020), encoding="utf-8")
      self.assertTrue(run_repository_audit(repo).passed)

      # Stricter per-file override (250) rejects a 280-line module cleanly.
      strict_mod = pkg_dir / "strict.py"
      strict_mod.write_text(_build_module_with_exec_lines(280), encoding="utf-8")
      strict_report = run_repository_audit(repo)
      self.assertFalse(strict_report.passed)
      self.assertEqual(
          [v.target for v in strict_report.blocking_quality_violations],
          ["pkg/strict.py"],
      )
      strict_mod.unlink()

      # Exact global ceiling (1000) stays advisory; 1001 fails AST-MAX-LINES.
      bloat_mod = pkg_dir / "bloat.py"
      bloat_mod.write_text(_build_module_with_exec_lines(1000), encoding="utf-8")
      ceiling_viols = verify_python_ast_quality(bloat_mod)
      self.assertEqual(len(ceiling_viols), 1)
      self.assertEqual(ceiling_viols[0].rule_id, "AST-SIZE-ADVISORY")
      self.assertEqual(ceiling_viols[0].severity, "WARNING")
      bloat_mod.write_text(_build_module_with_exec_lines(1001), encoding="utf-8")
      bloat_viols = verify_python_ast_quality(bloat_mod)
      self.assertEqual(bloat_viols[0].rule_id, "AST-MAX-LINES")
      self.assertEqual(bloat_viols[0].severity, "ERROR")
      failed_report = run_repository_audit(repo)
      self.assertFalse(failed_report.passed)

  def test_compliance_policy_legacy_and_malformed_fallbacks(self) -> None:
    """Policy loader handles legacy keys and malformed/bool JSON safely."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      repo = pathlib.Path(tmp_dir)
      # Legacy `default_max_lines` maps into healthy_max_lines cleanly.
      (repo / ".copilot_compliance.json").write_text(
          '{"healthy_max_lines": null, "default_max_lines": 320}\n',
          encoding="utf-8",
      )
      legacy_policy = load_compliance_policy(repo)
      self.assertEqual(legacy_policy.healthy_max_lines, 320)
      self.assertEqual(legacy_policy.ceiling_max_lines, 1000)

      # Malformed types, bools, negative numbers, and bad overrides fall back.
      (repo / ".copilot_compliance.json").write_text(
          (
              '{\n'
              '  "healthy_min_lines": 999,\n'
              '  "healthy_max_lines": "bad",\n'
              '  "ceiling_max_lines": true,\n'
              '  "min_comment_ratio": 2.5,\n'
              '  "file_max_lines_overrides": {\n'
              '    "bad_bool.py": true,\n'
              '    "bad_neg.py": -5,\n'
              '    "ok.py": 900\n'
              '  }\n'
              '}\n'
          ),
          encoding="utf-8",
      )
      safe_policy = load_compliance_policy(repo)
      self.assertEqual(safe_policy.healthy_max_lines, 800)
      self.assertEqual(safe_policy.healthy_min_lines, 800)
      self.assertEqual(safe_policy.ceiling_max_lines, 1000)
      self.assertEqual(safe_policy.min_comment_ratio, 0.08)
      self.assertEqual(safe_policy.file_max_lines_overrides, (("ok.py", 900),))



if __name__ == "__main__":
  unittest.main()
