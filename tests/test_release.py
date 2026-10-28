"""Tests for the v0.1 release artefacts (Day 7): version reporting,
the changelog and its consistency with the README.
"""
import os
import re
import subprocess
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from documind import __version__  # noqa: E402


def _read(name):
    with open(os.path.join(BASE_DIR, name), encoding="utf-8") as fh:
        return fh.read()


class TestRelease(unittest.TestCase):
    def test_version_is_semver_and_v01(self):
        self.assertRegex(__version__, r"^\d+\.\d+\.\d+$")
        self.assertEqual(__version__, "0.1.0")

    def test_cli_version_flag(self):
        proc = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "cli.py"), "--version"],
            capture_output=True, text=True, cwd=BASE_DIR)
        self.assertEqual(proc.returncode, 0)
        self.assertIn(__version__, proc.stdout)
        self.assertIn("documind", proc.stdout.lower())

    def test_changelog_documents_v01(self):
        changelog = _read("CHANGELOG.md")
        self.assertIn("## [0.1.0]", changelog)
        self.assertIn(__version__, changelog)
        for keyword in ("TF-IDF", "BM25", "rerank", "SQLite",
                        "evaluation", "citations"):
            self.assertIn(keyword, changelog,
                          "CHANGELOG should mention %s" % keyword)

    def test_changelog_eval_numbers_match_known_results(self):
        # the shipped set scores 1.000 / 1.000 / 1.000 / 0.917 on every
        # method; the release notes must not drift from that
        changelog = _read("CHANGELOG.md")
        self.assertIn("1.000", changelog)
        self.assertIn("0.917", changelog)
        self.assertIn("12-question", changelog)

    def test_readme_points_at_changelog(self):
        readme = _read("README.md")
        self.assertIn("CHANGELOG.md", readme)
        self.assertIn("v0.1", readme)


if __name__ == "__main__":
    unittest.main()
