"""Unit tests for the DocuMind core engine (Day 1).

Run from the project root:  python3 -m unittest discover -s tests
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from documind.chunker import chunk_text, split_sentences
from documind.embeddings import TfidfVectorizer, cosine_similarity
from documind.index import Index
from documind.qa import answer_question


class TestChunker(unittest.TestCase):
    def test_empty_text_yields_no_chunks(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n  "), [])

    def test_short_text_is_one_chunk(self):
        text = "One short sentence. And another one."
        self.assertEqual(chunk_text(text), [text])

    def test_long_text_splits_into_multiple_chunks(self):
        text = " ".join("Sentence number %d is here." % i for i in range(60))
        chunks = chunk_text(text, chunk_size=200, overlap=40)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)

    def test_chunks_overlap(self):
        text = " ".join("Alpha bravo charlie delta echo %d." % i
                        for i in range(40))
        chunks = chunk_text(text, chunk_size=120, overlap=50)
        self.assertGreater(len(chunks), 1)
        # each consecutive pair must share at least one word (the overlap tail)
        for first, second in zip(chunks, chunks[1:]):
            shared = set(first.split()) & set(second.split())
            self.assertTrue(shared, "consecutive chunks should overlap")

    def test_split_sentences(self):
        self.assertEqual(split_sentences("A. B! C?"), ["A.", "B!", "C?"])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            chunk_text("hello", chunk_size=0)
        with self.assertRaises(ValueError):
            chunk_text("hello", chunk_size=10, overlap=10)


class TestTfidf(unittest.TestCase):
    def setUp(self):
        self.docs = [
            "cats chase mice and cats sleep all day",
            "dogs chase balls and dogs bark loudly",
            "quantum computers use qubits and superposition",
        ]
        self.vec = TfidfVectorizer()
        self.vectors = self.vec.fit_transform(self.docs)

    def test_vocabulary_contains_expected_terms(self):
        self.assertIn("cats", self.vec.vocabulary_)
        self.assertIn("quantum", self.vec.vocabulary_)
        self.assertNotIn("the", self.vec.vocabulary_)  # stopword removed

    def test_rare_term_has_higher_idf_than_common_term(self):
        self.assertGreater(self.vec.idf_["quantum"], self.vec.idf_["chase"])

    def test_vectors_are_unit_length(self):
        for v in self.vectors:
            norm = sum(w * w for w in v.values()) ** 0.5
            self.assertAlmostEqual(norm, 1.0, places=6)

    def test_identical_text_has_cosine_one(self):
        v = self.vec.transform(self.docs[0])
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=6)

    def test_related_text_scores_higher_than_unrelated(self):
        query = self.vec.transform("cats and mice")
        sim_related = cosine_similarity(query, self.vectors[0])
        sim_unrelated = cosine_similarity(query, self.vectors[2])
        self.assertGreater(sim_related, sim_unrelated)

    def test_serialisation_round_trip(self):
        clone = TfidfVectorizer.from_dict(self.vec.to_dict())
        self.assertEqual(clone.vocabulary_, self.vec.vocabulary_)
        self.assertEqual(clone.idf_, self.vec.idf_)


class TestIndexAndQA(unittest.TestCase):
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

    def test_retrieval_ranks_relevant_document_first(self):
        hits = self.index.search("largest planet gas giant", k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["doc_id"], "astronomy.md")
        self.assertGreater(hits[0]["score"], 0.0)

    def test_retrieval_other_topic(self):
        hits = self.index.search("tomato sauce pasta", k=2)
        self.assertEqual(hits[0]["doc_id"], "cooking.md")

    def test_unrelated_query_returns_nothing(self):
        self.assertEqual(self.index.search("zzzqqq nonexistent"), [])

    def test_qa_answer_contains_citation(self):
        result = answer_question(self.index, "What is the largest planet?")
        self.assertIn("[astronomy.md]", result["answer"])
        self.assertTrue(result["sources"])
        self.assertEqual(result["sources"][0]["doc_id"], "astronomy.md")

    def test_qa_no_match_is_graceful(self):
        result = answer_question(self.index, "zzzqqq nonexistent")
        self.assertEqual(result["sources"], [])
        self.assertIn("could not find", result["answer"])


if __name__ == "__main__":
    unittest.main()
