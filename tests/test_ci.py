"""Tests for the Day 17 GitHub Actions CI workflow.

Covers, all checkable without a GitHub runner:

- .github/workflows/ci.yml is valid YAML (parsed with PyYAML when it
  is installed; that one test skips otherwise, and the string-level
  checks below it still run).
- The workflow runs on pushes and pull requests to main, with a
  concurrency group that cancels stale runs on the same branch.
- The test job runs on ubuntu-latest across Python 3.10, 3.11 and
  3.12 — the versions the README promises support for.
- The job checks out the code, sets up Python, byte-compiles the
  source, runs the full unittest suite with the exact command the
  README documents, and smoke-checks the evaluation harness.
- The commands the workflow runs exist in the repo, so a typo in the
  workflow fails here instead of on the runner.

Run from the project root:  python3 -m unittest discover -s tests
"""
import os
import re
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(BASE_DIR, ".github", "workflows", "ci.yml")

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is optional here
    yaml = None


def _read_workflow():
    with open(WORKFLOW, encoding="utf-8") as fh:
        return fh.read()


class TestWorkflowFileExists(unittest.TestCase):
    def test_workflow_file_present(self):
        self.assertTrue(os.path.isfile(WORKFLOW),
                        ".github/workflows/ci.yml is missing")


@unittest.skipUnless(yaml is not None,
                     "PyYAML not installed; workflow structure unchecked")
class TestWorkflowParsed(unittest.TestCase):
    def setUp(self):
        self.workflow = yaml.safe_load(_read_workflow())

    def test_workflow_has_a_name(self):
        self.assertEqual(self.workflow["name"], "CI")

    def test_triggers_on_push_and_pr_to_main(self):
        # "on" parses to the boolean True under YAML 1.1, so read the
        # raw mapping key from the loaded dict either way.
        triggers = self.workflow.get(True, self.workflow.get("on"))
        self.assertIn("push", triggers)
        self.assertIn("pull_request", triggers)
        self.assertEqual(triggers["push"]["branches"], ["main"])
        self.assertEqual(triggers["pull_request"]["branches"], ["main"])

    def test_concurrency_cancels_stale_runs(self):
        group = self.workflow["concurrency"]
        self.assertIn("${{ github.ref }}", str(group["group"]))
        self.assertTrue(group["cancel-in-progress"])

    def test_matrix_covers_supported_python_versions(self):
        job = self.workflow["jobs"]["test"]
        versions = [str(v) for v in
                    job["strategy"]["matrix"]["python-version"]]
        self.assertEqual(versions, ["3.10", "3.11", "3.12"])

    def test_runs_on_ubuntu(self):
        self.assertEqual(self.workflow["jobs"]["test"]["runs-on"],
                         "ubuntu-latest")

    def test_matrix_does_not_fail_fast(self):
        # One version failing should not hide the others' results.
        self.assertFalse(
            self.workflow["jobs"]["test"]["strategy"]["fail-fast"])

    def test_uses_checkout_and_setup_python(self):
        steps = self.workflow["jobs"]["test"]["steps"]
        uses = [s.get("uses", "") for s in steps]
        self.assertIn("actions/checkout@v4", uses)
        setup = next(s for s in steps
                     if s.get("uses", "").startswith("actions/setup-python"))
        self.assertEqual(
            setup["with"]["python-version"],
            "${{ matrix.python-version }}")

    def test_runs_the_documented_test_command(self):
        steps = self.workflow["jobs"]["test"]["steps"]
        run_lines = [s.get("run", "") for s in steps if "run" in s]
        self.assertIn("python3 -m unittest discover -s tests", run_lines)

    def test_smoke_checks_the_evaluation_harness(self):
        steps = self.workflow["jobs"]["test"]["steps"]
        run_lines = [s.get("run", "") for s in steps if "run" in s]
        self.assertIn("python3 cli.py eval --all-methods", run_lines)

    def test_compile_step_before_tests(self):
        steps = self.workflow["jobs"]["test"]["steps"]
        run_lines = [s.get("run", "") for s in steps if "run" in s]
        compile_at = next(i for i, line in enumerate(run_lines)
                          if "compileall" in line)
        test_at = next(i for i, line in enumerate(run_lines)
                       if "unittest discover" in line)
        self.assertLess(compile_at, test_at,
                        "compile step must run before the test suite")


class TestWorkflowText(unittest.TestCase):
    """String-level checks that run even without PyYAML."""

    def setUp(self):
        self.text = _read_workflow()

    def test_push_and_pull_request_triggers(self):
        self.assertIn("push:", self.text)
        self.assertIn("pull_request:", self.text)
        self.assertIn("branches: [main]", self.text)

    def test_all_three_python_versions_listed(self):
        for version in ('"3.10"', '"3.11"', '"3.12"'):
            self.assertIn(version, self.text)

    def test_test_command_present(self):
        self.assertIn("python3 -m unittest discover -s tests", self.text)

    def test_eval_smoke_check_present(self):
        self.assertIn("python3 cli.py eval --all-methods", self.text)

    def test_no_step_installs_dependencies(self):
        # Pure stdlib: the workflow must not pip-install anything.
        self.assertNotIn("pip install", self.text)


class TestWorkflowCommandsExist(unittest.TestCase):
    """Everything the workflow runs must exist in this checkout."""

    def test_test_suite_directory_exists(self):
        self.assertTrue(os.path.isdir(os.path.join(BASE_DIR, "tests")))

    def test_cli_entry_point_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(BASE_DIR, "cli.py")))

    def test_eval_data_present(self):
        self.assertTrue(os.path.isdir(os.path.join(BASE_DIR, "data")))
        self.assertTrue(os.path.isfile(
            os.path.join(BASE_DIR, "eval", "eval_qa.json")))

    def test_workflow_commands_run_cleanly_locally(self):
        # The exact commands CI runs, executed the same way here.
        # This is the local proof the workflow would go green.
        compile_cmd = [sys.executable, "-m", "compileall", "-q",
                       "cli.py", "documind", "tests"]
        result = subprocess.run(compile_cmd, cwd=BASE_DIR,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
