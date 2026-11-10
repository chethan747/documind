"""Tests for the final evaluation report (Day 20).

EVALUATION.md and eval/final_report.json are the documented outcome of
the final evaluation run, so they must stay reproducible: every number
in them is checked here against a fresh run of the harness over the
sample documents. If the pipeline changes and the numbers move, these
tests fail until the report is re-run and re-documented — the report
must never drift from what the code actually scores.
"""
import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from documind.evaluate import evaluate_all_methods, load_cases
from documind.index import Index
from documind.ingest import ingest_folder
from documind.store import DocumentStore

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH = os.path.join(PROJECT_ROOT, "eval", "eval_qa.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "eval", "final_report.json")
DOC_PATH = os.path.join(PROJECT_ROOT, "EVALUATION.md")
README_PATH = os.path.join(PROJECT_ROOT, "README.md")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _fresh_index():
    store = DocumentStore()
    index = Index()
    try:
        ingest_folder(store, index, os.path.join(PROJECT_ROOT, "data"))
    finally:
        store.close()
    return index


class TestFinalEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report_doc = _read(DOC_PATH)
        cls.cases = load_cases(EVAL_PATH)
        cls.index = _fresh_index()
        cls.headline = {r["method"]: r
                        for r in evaluate_all_methods(cls.index, cls.cases,
                                                      k=5)}
        with open(REPORT_PATH, encoding="utf-8") as fh:
            cls.saved_report = json.load(fh)

    def _headline_table(self):
        section = self.report_doc.split("## Headline results")[1]
        section = section.split("## Per-question")[0]
        rows = {}
        for line in section.splitlines():
            m = re.match(r"\|\s*(tfidf|bm25|combined)\s*\|"
                         r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
                         r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", line)
            if m:
                rows[m.group(1)] = tuple(float(v) for v in m.groups()[1:])
        return rows

    def test_required_sections_present(self):
        for section in ("## Methodology", "## Headline results",
                        "## Per-question breakdown",
                        "## Cutoff sensitivity", "## Findings",
                        "## Limitations", "## Reproducibility"):
            self.assertIn(section, self.report_doc,
                          "EVALUATION.md is missing section %s" % section)
        self.assertIn("eval/final_report.json", self.report_doc)
        self.assertIn("2026-11-10", self.report_doc)
        self.assertIn("0.2.0", self.report_doc)

    def test_headline_table_matches_fresh_run(self):
        rows = self._headline_table()
        self.assertEqual(set(rows), {"tfidf", "bm25", "combined"})
        for method, values in rows.items():
            rep = self.headline[method]
            fresh = (rep["hit_rate"], rep["mrr"], rep["qa_citation_rate"],
                     rep["qa_term_rate"])
            self.assertEqual(values, tuple(round(v, 3) for v in fresh),
                             "headline row for %s drifted: %s != %s"
                             % (method, values, fresh))

    def test_per_question_table_matches_fresh_run(self):
        section = self.report_doc.split("## Per-question breakdown")[1]
        section = section.split("## Cutoff sensitivity")[0]
        rows = {}
        for line in section.splitlines():
            m = re.match(r"\|\s*(q\d+)\s*\|[^|]*\|\s*([\w.\-]+)\s*\|"
                         r"\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
                         r"\s*(yes|no)\s*\|\s*(yes|no)\s*\|", line)
            if m:
                rows[m.group(1)] = m.groups()[1:]
        self.assertEqual(set(rows), {"q%d" % i for i in range(1, 13)})
        for method in ("tfidf", "bm25", "combined"):
            results = {r["id"]: r for r in self.headline[method]["results"]}
            for qid, (doc, r_tfidf, r_bm25, r_comb, cited, terms) in \
                    rows.items():
                r = results[qid]
                self.assertEqual(doc, r["expected_doc_id"])
                self.assertEqual({"tfidf": r_tfidf, "bm25": r_bm25,
                                  "combined": r_comb}[method],
                                 str(r["rank"]),
                                 "%s rank for %s drifted" % (method, qid))
                self.assertEqual(cited,
                                 "yes" if r["qa_cited_expected"] else "no")
                self.assertEqual(terms,
                                 "yes" if r["qa_term_hit"] else "no")

    def test_cutoff_sensitivity_table_matches_fresh_runs(self):
        section = self.report_doc.split("## Cutoff sensitivity")[1]
        section = section.split("## Findings")[0]
        rows = {}
        for line in section.splitlines():
            m = re.match(r"\|\s*(\d+)\s*\|" + r"\s*([\d.]+)\s*\|" * 6, line)
            if m:
                rows[int(m.group(1))] = tuple(float(v)
                                              for v in m.groups()[1:])
        self.assertEqual(set(rows), {1, 3, 5, 10})
        for k, values in rows.items():
            fresh = evaluate_all_methods(self.index, self.cases, k=k)
            expected = []
            for rep in fresh:
                expected.append(rep["hit_rate"])
            for rep in fresh:
                expected.append(rep["qa_term_rate"])
            self.assertEqual(values, tuple(round(v, 3) for v in expected),
                             "sensitivity row for k=%d drifted: %s != %s"
                             % (k, values, expected))

    def test_saved_report_json_matches_fresh_run(self):
        self.assertIsInstance(self.saved_report, list)
        self.assertEqual(len(self.saved_report), 3)
        for saved in self.saved_report:
            fresh = self.headline[saved["method"]]
            self.assertEqual(saved["k"], 5)
            self.assertEqual(saved["n"], 12)
            for key in ("hit_rate", "mrr", "qa_citation_rate",
                        "qa_term_rate"):
                self.assertAlmostEqual(saved[key], fresh[key], places=6,
                                       msg="%s %s drifted"
                                       % (saved["method"], key))
            self.assertEqual(len(saved["results"]), 12)
            for saved_q, fresh_q in zip(saved["results"],
                                        fresh["results"]):
                self.assertEqual(saved_q["id"], fresh_q["id"])
                self.assertEqual(saved_q["rank"], fresh_q["rank"])
                self.assertEqual(saved_q["qa_term_hit"],
                                 fresh_q["qa_term_hit"])

    def test_q9_miss_is_documented_not_hidden(self):
        # the one known miss must stay visible in the report
        self.assertIn("q9", self.report_doc)
        self.assertIn("0.917", self.report_doc)
        self.assertIn("cross-validation", self.report_doc)

    def test_readme_points_at_full_report(self):
        readme = _read(README_PATH)
        evaluation = readme.split("## Evaluation")[1].split("## Run with Docker")[0]
        self.assertIn("EVALUATION.md", evaluation)


if __name__ == "__main__":
    unittest.main()
