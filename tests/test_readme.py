"""Tests for the README (Day 19): the polish must stay accurate — the
sections exist, the demo/evaluation output quoted in it matches what
the commands really print, the links resolve, and no stale claim
from an earlier day survives.
"""
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


def _read(name):
    with open(os.path.join(BASE_DIR, name), encoding="utf-8") as fh:
        return fh.read()


def _scratch_copy():
    """A throwaway copy of the runnable project (no tests, no caches,
    no saved state) so demo/eval runs cannot touch the real index."""
    tmp = tempfile.mkdtemp(prefix="documind-readme-")
    for name in ("cli.py", "config.example.json"):
        shutil.copy(os.path.join(BASE_DIR, name), tmp)
    for name in ("documind", "data", "eval"):
        shutil.copytree(os.path.join(BASE_DIR, name),
                        os.path.join(tmp, name),
                        ignore=shutil.ignore_patterns("__pycache__"))
    return tmp


def _run(tmp, *args):
    proc = subprocess.run(
        [sys.executable, os.path.join(tmp, "cli.py"), *args],
        capture_output=True, text=True, cwd=tmp)
    return proc


class TestReadme(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = _read("README.md")
        cls.tmp = _scratch_copy()
        demo = _run(cls.tmp, "demo")
        assert demo.returncode == 0, demo.stderr
        cls.demo_output = demo.stdout
        evaluation = _run(cls.tmp, "eval", "--all-methods")
        assert evaluation.returncode == 0, evaluation.stderr
        cls.eval_output = evaluation.stdout
        # demo runs in memory, so persist the same documents for the
        # server with a real ingest before starting it
        ingest = _run(cls.tmp, "ingest", "data")
        assert ingest.returncode == 0, ingest.stderr
        # the API transcript quoted in the README, from a real server
        # over the same scratch index
        cls.server = subprocess.Popen(
            [sys.executable, os.path.join(cls.tmp, "cli.py"),
             "serve", "--port", "18199"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=cls.tmp)
        try:
            cls.api_health = cls._wait_for(
                "http://127.0.0.1:18199/health")
            req = urllib.request.Request(
                "http://127.0.0.1:18199/qa",
                data=b'{"question": "What is AEM built on?"}',
                headers={"Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                cls.api_qa = resp.read().decode("utf-8")
        finally:
            cls.server.terminate()
            cls.server.wait(timeout=10)

    @classmethod
    def _wait_for(cls, url, timeout=15):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    return resp.read().decode("utf-8")
            except (urllib.error.URLError, socket.timeout):
                time.sleep(0.2)
        raise AssertionError("server did not come up at %s" % url)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_required_sections_present(self):
        for section in (
                "## Features", "## Architecture", "## Quickstart",
                "## Demo output", "## Evaluation", "## Run with Docker",
                "## REST API", "## Configuration & logging",
                "## Performance", "## Tech notes", "## Layout"):
            self.assertIn(section, self.readme,
                          "README is missing section %s" % section)

    def test_no_stale_progress_claims(self):
        # Day 17 CI and Day 18 performance are done; the README must
        # not still present them as upcoming work.
        for stale in ("next are CI", "Done (v0.2 — API + web app)",
                      "Docker packaging is done, next are CI"):
            self.assertNotIn(stale, self.readme,
                             "stale claim still in README: %r" % stale)
        self.assertIn("Days 1–20 of 21", self.readme)
        self.assertIn("CI", self.readme)
        self.assertIn("performance pass", self.readme)

    def test_demo_output_is_verbatim(self):
        demo_section = self.readme.split("## Demo output")[1].split("## Evaluation")[0]
        for line in (
                "3 file(s) seen, 3 document(s) added, 14 chunk(s) indexed",
                "Q: What is AEM built on?",
                "A: AEM is built on three open technologies from the "
                "Apache Software Foundation. [aem-basics.md]",
                "Q: What is overfitting in machine learning?"):
            self.assertIn(line, demo_section,
                          "README demo output is missing: %r" % line)
            self.assertIn(line, self.demo_output,
                          "quoted line is not real demo output: %r" % line)
        # the API transcript quoted in the README is real server output
        self.assertIn(self.api_health.strip(), demo_section)
        self.assertIn('"citation": "[aem-basics.md, chunk 0]"', self.api_qa)
        self.assertIn('"citation": "[aem-basics.md, chunk 0]"', demo_section)
        self.assertIn("ingested aem-basics.md", self.demo_output)

    def test_evaluation_table_matches_real_run(self):
        # parse the real summary table (method, hit, MRR, cited, terms)
        real = {}
        for line in self.eval_output.splitlines():
            m = re.match(r"^(tfidf|bm25|combined)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", line)
            if m:
                real[m.group(1)] = m.groups()[1:]
        self.assertEqual(set(real), {"tfidf", "bm25", "combined"})
        table = self.readme.split("## Evaluation")[1].split("## Run with Docker")[0]
        for method, values in real.items():
            row = re.search(r"\|\s*%s\s*\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|" % method, table)
            self.assertIsNotNone(row, "evaluation table lacks %s" % method)
            quoted = tuple(cell.strip() for cell in row.groups())
            self.assertEqual(quoted, values,
                             "README evaluation row for %s drifted: %s != %s"
                             % (method, quoted, values))

    def test_evaluation_q9_note_present(self):
        # the one term miss is a known, explained quirk — the README
        # must not claim a perfect QA term rate
        self.assertIn("0.917", self.readme)
        self.assertIn("question 9", self.readme)

    def test_bench_table_is_plausible(self):
        perf = self.readme.split("## Performance")[1].split("## Tech notes")[0]
        rows = re.findall(
            r"\|\s*(tfidf|bm25|combined)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)x\s*\|",
            perf)
        self.assertEqual({r[0] for r in rows}, {"tfidf", "bm25", "combined"})
        for _, cold, warm, speedup in rows:
            self.assertGreater(float(cold), float(warm),
                               "warm search must be faster than cold")
            self.assertGreater(float(speedup), 1.0)
        self.assertIn("vary by machine", perf)

    def test_internal_links_resolve(self):
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", self.readme):
            if target.startswith("http") or target.startswith("#"):
                continue
            path = target.split("#")[0]
            self.assertTrue(os.path.isfile(os.path.join(BASE_DIR, path)),
                            "README links to missing file: %s" % path)

    def test_fenced_code_blocks_balanced(self):
        fences = [line for line in self.readme.splitlines()
                  if line.strip().startswith("```")]
        self.assertEqual(len(fences) % 2, 0,
                         "unbalanced fenced code blocks in README")

    def test_screenshot_decision_is_stated(self):
        # no screenshot tooling is available on the build machine, so
        # Day 19 ships verbatim demo output instead — the README must
        # say so rather than imply screenshots exist
        self.assertIn("No screenshots are included", self.readme)

    def test_readme_still_points_at_changelog_and_roadmap(self):
        self.assertIn("ROADMAP.md", self.readme)
        self.assertIn("CHANGELOG.md", self.readme)
        self.assertIn("v0.2", self.readme)


if __name__ == "__main__":
    unittest.main()
