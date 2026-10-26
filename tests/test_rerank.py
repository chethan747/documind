"""Unit tests for Day 5: answer reranking + snippet extraction.

Run from the project root:  python3 -m unittest discover -s tests
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from documind.index import Index
from documind.qa import answer_question
from documind.rerank import extract_snippet, rerank_hits, sentence_score


class TestSentenceScore(unittest.TestCase):
    def test_no_overlap_scores_zero(self):
        self.assertEqual(sentence_score("quantum qubits",
                                        "Pasta cooks in boiling water."), 0.0)

    def test_full_coverage_beats_partial(self):
        full = sentence_score("largest planet gas giant",
                              "Jupiter is the largest planet and a gas giant.")
        partial = sentence_score("largest planet gas giant",
                                 "Jupiter is the largest planet.")
        self.assertGreater(full, partial)

    def test_rare_term_weights_more_than_common_term(self):
        # With idf, a sentence matching only the rare term outscores
        # one matching only the common term.
        idf = {"europa": 5.0, "planet": 1.0}
        rare = sentence_score("europa planet",
                              "Europa orbits far away.", idf)
        common = sentence_score("europa planet",
                                "The planet is large.", idf)
        self.assertGreater(rare, common)

    def test_exact_phrase_gets_bonus(self):
        with_phrase = sentence_score("great red spot",
                                     "Jupiter has the Great Red Spot.")
        without_phrase = sentence_score("great red spot",
                                        "Great storms and a red sky, spot on.")
        self.assertGreater(with_phrase, without_phrase)

    def test_short_on_topic_sentence_beats_rambling_one(self):
        short = sentence_score("tomato sauce",
                               "Tomato sauce uses basil.")
        long = sentence_score(
            "tomato sauce",
            "Tomato sauce is something people talk about when they cook "
            "in kitchens with friends and family on weekends.")
        self.assertGreater(short, long)

    def test_empty_question_or_sentence_scores_zero(self):
        self.assertEqual(sentence_score("", "Some sentence."), 0.0)
        self.assertEqual(sentence_score("cats", ""), 0.0)


class TestExtractSnippet(unittest.TestCase):
    def test_picks_best_sentence_not_first(self):
        text = ("Cooking takes patience and practice every day. "
                "Jupiter is the largest planet in the solar system. "
                "The weather was fine that morning.")
        self.assertEqual(
            extract_snippet(text, "largest planet"),
            "Jupiter is the largest planet in the solar system.")

    def test_neighbour_added_when_it_adds_uncovered_terms(self):
        text = ("Jupiter is the largest planet. "
                "It has a famous storm called the Great Red Spot.")
        snippet = extract_snippet(text, "largest planet great red spot")
        self.assertIn("largest planet", snippet)
        self.assertIn("Great Red Spot", snippet)

    def test_neighbour_skipped_when_it_adds_nothing(self):
        text = ("Jupiter is the largest planet in the solar system. "
                "The weather was fine that morning.")
        self.assertEqual(
            extract_snippet(text, "largest planet"),
            "Jupiter is the largest planet in the solar system.")

    def test_no_overlap_yields_empty_snippet(self):
        self.assertEqual(
            extract_snippet("Pasta cooks in boiling water.",
                           "zzzqqq nonexistent"), "")

    def test_overlong_snippet_trims_at_word_boundary(self):
        long_sentence = "Europa " + "very " * 100 + "moon."
        snippet = extract_snippet(long_sentence, "europa moon",
                                  max_chars=60)
        self.assertLessEqual(len(snippet), 61)
        self.assertTrue(snippet.endswith("…"))
        self.assertNotIn("  ", snippet.rstrip("…"))


class TestRerankHits(unittest.TestCase):
    def _hits(self):
        return [
            {"doc_id": "diffuse.md", "chunk_index": 0,
             "text": ("Cats are popular pets. Dogs are popular too. "
                      "Planets orbit stars in space."),
             "score": 0.9},
            {"doc_id": "focused.md", "chunk_index": 0,
             "text": "Jupiter is the largest planet in the solar system.",
             "score": 0.5},
        ]

    def test_focused_chunk_promoted_over_diffuse_high_scorer(self):
        ranked = rerank_hits(self._hits(), "largest planet")
        self.assertEqual(ranked[0]["doc_id"], "focused.md")
        self.assertGreater(ranked[0]["rerank_score"],
                           ranked[1]["rerank_score"])

    def test_original_score_untouched_and_snippet_present(self):
        ranked = rerank_hits(self._hits(), "largest planet")
        focused = next(h for h in ranked if h["doc_id"] == "focused.md")
        self.assertEqual(focused["score"], 0.5)
        self.assertIn("largest planet", focused["snippet"])

    def test_empty_hits_rerank_to_empty(self):
        self.assertEqual(rerank_hits([], "anything"), [])


class TestQAWithReranking(unittest.TestCase):
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

    def test_answer_quotes_best_sentence_with_citation(self):
        result = answer_question(self.index, "What is the largest planet?")
        self.assertIn("Jupiter is the largest planet", result["answer"])
        self.assertIn("[astronomy.md]", result["answer"])
        self.assertEqual(result["sources"][0]["doc_id"], "astronomy.md")

    def test_sources_carry_snippet_and_rerank_score(self):
        result = answer_question(self.index, "largest planet gas giant")
        source = result["sources"][0]
        self.assertIn("snippet", source)
        self.assertIn("rerank_score", source)
        self.assertIn(source["snippet"], result["answer"])

    def test_no_rerank_keeps_retriever_order(self):
        result = answer_question(self.index, "largest planet gas giant",
                                 rerank=False)
        self.assertTrue(result["sources"])
        self.assertNotIn("rerank_score", result["sources"][0])
        self.assertIn("snippet", result["sources"][0])

    def test_duplicate_snippets_appear_once(self):
        # Overlapping chunks can repeat a sentence; the answer must not.
        result = answer_question(self.index, "largest planet", k=3)
        for snippet in [s["snippet"] for s in result["sources"]
                        if s["snippet"]]:
            self.assertEqual(result["answer"].count(snippet), 1)

    def test_no_match_still_graceful(self):
        result = answer_question(self.index, "zzzqqq nonexistent")
        self.assertEqual(result["sources"], [])
        self.assertIn("could not find", result["answer"])


if __name__ == "__main__":
    unittest.main()
