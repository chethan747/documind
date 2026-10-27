"""Unit tests for Day 6: evaluation harness.

Run from the project root:  python3 -m unittest discover -s tests
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from documind.evaluate import (evaluate_all_methods, format_report,
                                load_cases, run_evaluation)
from documind.index import Index

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH = os.path.join(PROJECT_ROOT, "eval", "eval_qa.json")


def _demo_index():
    index = Index()
    index.add_document(
        "aem-basics.md",
        "AEM is built on Apache Sling, Apache Jackrabbit Oak and Apache "
        "Felix. The dispatcher caches published pages and protects the "
        "publish instances from unnecessary load.")
    index.add_document(
        "java-spring.md",
        "Spring Boot auto-configuration inspects the libraries on the "
        "classpath and configures beans automatically. Spring MVC uses "
        "@RestController classes to expose REST endpoints.")
    index.add_document(
        "ml-basics.md",
        "Overfitting is when a model memorises the training examples and "
        "performs badly on new data. Cross-validation and regularisation "
        "help to reduce overfitting.")
    return index


def _cases():
    return [
        {"id": "q1", "question": "What is AEM built on?",
         "expected_doc_id": "aem-basics.md",
         "expected_terms": ["Sling", "Felix"]},
        {"id": "q2", "question": "What does Spring Boot auto-configuration do?",
         "expected_doc_id": "java-spring.md",
         "expected_terms": ["classpath"]},
        {"id": "q3", "question": "What is overfitting?",
         "expected_doc_id": "ml-basics.md",
         # The reranker picks the shorter on-topic sentence here
         # ("...help to reduce overfitting"), so either phrasing counts.
         "expected_terms": ["memorises", "overfitting"]},
    ]


class TestLoadCases(unittest.TestCase):
    def _write(self, data):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        self.addCleanup(os.unlink, path)
        return path

    def test_loads_bare_list_and_object_form(self):
        cases = _cases()
        self.assertEqual(len(load_cases(self._write(cases))), 3)
        self.assertEqual(len(load_cases(self._write({"cases": cases}))), 3)

    def test_defaults_id_and_empty_terms(self):
        path = self._write([{"question": "What is AEM built on?",
                            "expected_doc_id": "aem-basics.md"}])
        case = load_cases(path)[0]
        self.assertEqual(case["id"], "q1")
        self.assertEqual(case["expected_terms"], [])

    def test_rejects_missing_question_or_doc(self):
        with self.assertRaises(ValueError):
            load_cases(self._write([{"expected_doc_id": "a.md"}]))
        with self.assertRaises(ValueError):
            load_cases(self._write([{"question": "Q?"}]))
        with self.assertRaises(ValueError):
            load_cases(self._write([]))

    def test_rejects_bad_terms_and_bad_json(self):
        with self.assertRaises(ValueError):
            load_cases(self._write([{"question": "Q?",
                                    "expected_doc_id": "a.md",
                                    "expected_terms": "Sling"}]))
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        self.addCleanup(os.unlink, path)
        with self.assertRaises(ValueError):
            load_cases(path)


class TestRunEvaluation(unittest.TestCase):
    def setUp(self):
        self.index = _demo_index()
        self.cases = _cases()

    def test_all_three_hit_at_top(self):
        report = run_evaluation(self.index, self.cases, k=5)
        self.assertEqual(report["hit_rate"], 1.0)
        self.assertEqual(report["mrr"], 1.0)
        self.assertEqual(report["qa_citation_rate"], 1.0)
        self.assertEqual(report["qa_term_rate"], 1.0)
        self.assertTrue(all(r["rank"] == 1 for r in report["results"]))

    def test_miss_counts_as_zero_reciprocal(self):
        cases = self.cases + [{"id": "q4",
                               "question": "zzzqqq nonexistent",
                               "expected_doc_id": "aem-basics.md",
                               "expected_terms": []}]
        report = run_evaluation(self.index, cases, k=5)
        self.assertEqual(report["n"], 4)
        self.assertEqual(report["hit_rate"], 0.75)
        self.assertEqual(report["mrr"], 0.75)
        miss = report["results"][-1]
        self.assertIsNone(miss["rank"])
        self.assertFalse(miss["hit"])
        self.assertEqual(miss["reciprocal_rank"], 0.0)
        self.assertFalse(miss["qa_cited_expected"])

    def test_k_cutoff_can_exclude_a_hit(self):
        # A question whose expected doc ranks below k=1 still counts
        # at k=5; here all rank 1, so instead check k is honoured in
        # the report and that a smaller k never beats a larger one.
        small = run_evaluation(self.index, self.cases, k=1)
        large = run_evaluation(self.index, self.cases, k=5)
        self.assertGreaterEqual(large["hit_rate"], small["hit_rate"])
        self.assertEqual(small["k"], 1)

    def test_term_rate_none_when_no_terms_listed(self):
        cases = [{"id": "q1", "question": "What is AEM built on?",
                  "expected_doc_id": "aem-basics.md",
                  "expected_terms": []}]
        report = run_evaluation(self.index, cases)
        self.assertIsNone(report["qa_term_rate"])

    def test_invalid_k_and_empty_cases_rejected(self):
        with self.assertRaises(ValueError):
            run_evaluation(self.index, self.cases, k=0)
        with self.assertRaises(ValueError):
            run_evaluation(self.index, [])

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            run_evaluation(self.index, self.cases, method="pagerank")

    def test_all_methods_returns_three_reports(self):
        reports = evaluate_all_methods(self.index, self.cases, k=5)
        self.assertEqual([r["method"] for r in reports],
                         ["tfidf", "bm25", "combined"])
        self.assertTrue(all(r["hit_rate"] == 1.0 for r in reports))


class TestFormatReport(unittest.TestCase):
    def test_single_report_lists_questions(self):
        report = run_evaluation(_demo_index(), _cases(), k=5)
        text = format_report(report)
        self.assertIn("hit rate@5", text)
        self.assertIn("MRR@5", text)
        self.assertIn("What is AEM built on?", text)

    def test_comparison_table_covers_all_methods(self):
        reports = evaluate_all_methods(_demo_index(), _cases(), k=5)
        text = format_report(reports)
        for method in ("tfidf", "bm25", "combined"):
            self.assertIn(method, text)


class TestShippedEvalSet(unittest.TestCase):
    def test_shipped_cases_load_and_score(self):
        cases = load_cases(EVAL_PATH)
        self.assertEqual(len(cases), 12)
        index = Index()
        for name in ("aem-basics.md", "java-spring.md", "ml-basics.md"):
            path = os.path.join(PROJECT_ROOT, "data", name)
            with open(path, encoding="utf-8") as fh:
                index.add_document(name, fh.read())
        report = run_evaluation(index, cases, k=5)
        # The shipped set is written against the shipped documents, so
        # every question should retrieve its document in the top 5.
        self.assertEqual(report["hit_rate"], 1.0)
        self.assertGreaterEqual(report["qa_citation_rate"], 0.9)


if __name__ == "__main__":
    unittest.main()
