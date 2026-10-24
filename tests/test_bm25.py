"""Tests for the Day 3 BM25 ranker and its Index integration.

Run from the project root:  python3 -m unittest discover -s tests
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from documind.bm25 import BM25Scorer
from documind.index import Index
from documind.qa import answer_question


class TestBM25Scorer(unittest.TestCase):
    def setUp(self):
        self.docs = [
            "cats chase mice and cats sleep all day",
            "dogs chase balls and dogs bark loudly",
            "quantum computers use qubits and superposition",
        ]
        self.scorer = BM25Scorer().fit(self.docs)

    def test_rare_term_has_higher_idf_than_common_term(self):
        self.assertGreater(self.scorer.idf_["quantum"],
                           self.scorer.idf_["chase"])

    def test_relevant_document_scores_higher_than_unrelated(self):
        self.assertGreater(self.scorer.score("cats mice", 0),
                           self.scorer.score("cats mice", 2))

    def test_unknown_terms_score_zero(self):
        self.assertEqual(self.scorer.score("zzzqqq nonexistent", 0), 0.0)
        self.assertEqual(self.scorer.score("", 0), 0.0)

    def test_term_frequency_saturates(self):
        # 10 mentions must NOT score 10x a single mention (the point of k1).
        once = BM25Scorer().fit(["cats", "dogs birds"])
        ten = BM25Scorer().fit(["cats " * 10, "dogs birds"])
        ratio = ten.score("cats", 0) / once.score("cats", 0)
        self.assertGreater(ratio, 1.0)
        self.assertLess(ratio, 5.0)

    def test_shorter_document_scores_higher_for_same_match(self):
        scorer = BM25Scorer().fit([
            "cats",
            "cats " + "filler " * 20,
            "dogs birds",
        ])
        self.assertGreater(scorer.score("cats", 0),
                           scorer.score("cats", 1))

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            BM25Scorer(k1=0)
        with self.assertRaises(ValueError):
            BM25Scorer(b=1.5)

    def test_serialisation_round_trip(self):
        clone = BM25Scorer.from_dict(self.scorer.to_dict())
        self.assertEqual(clone.idf_, self.scorer.idf_)
        self.assertAlmostEqual(clone.avgdl_, self.scorer.avgdl_)
        self.assertEqual(clone.score("cats mice", 0),
                         self.scorer.score("cats mice", 0))


class TestIndexBM25(unittest.TestCase):
    def setUp(self):
        self.index = Index()
        self.index.add_document(
            "cooking.md",
            "Pasta cooks in boiling salted water for ten minutes. "
            "A good tomato sauce uses garlic, olive oil and basil. "
            "Fresh pasta needs only a few minutes in the pot.")
        self.index.add_document(
            "astronomy.md",
            "Jupiter is the largest planet in the solar system. "
            "It is a gas giant with a famous storm called the Great Red Spot. "
            "Jupiter has dozens of moons, including Europa and Ganymede.")

    def test_bm25_ranks_relevant_document_first(self):
        hits = self.index.search("largest planet gas giant", k=2,
                                 method="bm25")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["doc_id"], "astronomy.md")
        self.assertGreater(hits[0]["score"], 0.0)

    def test_bm25_other_topic(self):
        hits = self.index.search("tomato sauce pasta", k=2, method="bm25")
        self.assertEqual(hits[0]["doc_id"], "cooking.md")

    def test_combined_ranks_relevant_document_first(self):
        hits = self.index.search("largest planet gas giant", k=2,
                                 method="combined")
        self.assertEqual(hits[0]["doc_id"], "astronomy.md")

    def test_tfidf_default_unchanged(self):
        # No method argument must behave exactly as before Day 3.
        hits = self.index.search("largest planet gas giant", k=2)
        self.assertEqual(hits[0]["doc_id"], "astronomy.md")

    def test_bm25_unrelated_query_returns_nothing(self):
        self.assertEqual(
            self.index.search("zzzqqq nonexistent", method="bm25"), [])
        self.assertEqual(
            self.index.search("zzzqqq nonexistent", method="combined"), [])

    def test_unknown_method_rejected(self):
        with self.assertRaises(ValueError):
            self.index.search("planet", method="pagerank")

    def test_bm25_survives_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "index.json")
            self.index.save(path)
            loaded = Index.load(path)
            hits = loaded.search("largest planet gas giant", k=2,
                                 method="bm25")
            self.assertEqual(hits[0]["doc_id"], "astronomy.md")

    def test_pre_bm25_index_loads_and_refits(self):
        # An index dict without a "bm25" key (pre-Day 3 format) must load
        # and refit BM25 on first use instead of crashing.
        data = self.index.to_dict()
        del data["bm25"]
        loaded = Index.from_dict(data)
        hits = loaded.search("tomato sauce pasta", k=2, method="bm25")
        self.assertEqual(hits[0]["doc_id"], "cooking.md")

    def test_qa_with_bm25_still_cites_sources(self):
        result = answer_question(self.index, "What is the largest planet?",
                                 method="bm25")
        self.assertIn("[astronomy.md]", result["answer"])
        self.assertTrue(result["sources"])


if __name__ == "__main__":
    unittest.main()
