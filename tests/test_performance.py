"""Tests for the Day 18 index caching + performance pass.

Three things, each verified to leave results unchanged:

- The inverted indexes: TF-IDF search now scores through term
  postings and BM25 through document postings, so a query only
  touches chunks sharing one of its terms. Both are checked against
  a brute-force reference (score every chunk / every document) on
  the same corpus — same hits, same order, same scores.
- The query cache: a bounded LRU over search results, keyed by the
  normalised query + k + method, invalidated whenever the index
  changes, sized from ``retrieval.query_cache_size`` (0 disables
  it). Cached results are copies, so a caller mutating a returned
  hit cannot poison the cache.
- The benchmark: ``documind/benchmark.py`` times build, cold and
  warm searches per method, and ``cli.py bench`` prints the table;
  the server's ``/stats`` reports the cache counters.

Run from the project root:  python3 -m unittest discover -s tests
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from documind.benchmark import (format_benchmark, run_benchmark,
                                scale_corpus)
from documind.bm25 import BM25Scorer
from documind.cache import LRUCache
from documind.config import DEFAULT_CONFIG, load_config
from documind.embeddings import cosine_similarity
from documind.index import Index
from documind.server import create_server

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOCS = [
    ("aem-basics",
     "AEM is built on Apache Sling and the Java Content Repository. "
     "Sling maps HTTP requests to JCR nodes. Components render with HTL "
     "templates and OSGi services."),
    ("spring-boot",
     "Spring Boot auto-configuration wires beans from the classpath. "
     "A starter pulls in the dependencies a web application needs. "
     "Actuator exposes health and metrics endpoints."),
    ("ml-basics",
     "Overfitting is when a model memorises training data instead of "
     "learning the pattern. Cross-validation splits data into folds "
     "to estimate how the model generalises."),
    ("caching",
     "A cache stores repeated results so later requests skip the "
     "work. An LRU cache evicts the least recently used entry when "
     "it is full. Cache invalidation keeps stale entries out."),
]

QUERIES = [
    "What is AEM built on?",
    "Spring Boot auto-configuration",
    "overfitting cross-validation",
    "cache eviction policy",
    "totally unknown zzzterm",
]


def _make_index(cache_size=128):
    index = Index(cache_size=cache_size)
    for doc_id, text in DOCS:
        index.add_document(doc_id, text)
    return index


class TestLRUCache(unittest.TestCase):
    def test_put_get_round_trip(self):
        cache = LRUCache(maxsize=4)
        cache.put("a", [1, 2])
        self.assertEqual(cache.get("a"), [1, 2])
        self.assertEqual(len(cache), 1)

    def test_miss_returns_default(self):
        cache = LRUCache(maxsize=4)
        self.assertIsNone(cache.get("missing"))
        self.assertEqual(cache.get("missing", "dflt"), "dflt")

    def test_evicts_least_recently_used(self):
        cache = LRUCache(maxsize=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # refresh a: b is now the LRU entry
        cache.put("c", 3)
        self.assertIn("a", cache)
        self.assertNotIn("b", cache)
        self.assertIn("c", cache)
        self.assertEqual(cache.evictions, 1)

    def test_updating_a_key_does_not_evict(self):
        cache = LRUCache(maxsize=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("a", 10)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.get("a"), 10)
        self.assertEqual(cache.evictions, 0)

    def test_maxsize_zero_disables_the_cache(self):
        cache = LRUCache(maxsize=0)
        cache.put("a", 1)
        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.get("a"))

    def test_invalid_maxsize_rejected(self):
        for bad in (-1, "big", 1.5, True):
            with self.assertRaises(ValueError, msg="maxsize=%r" % (bad,)):
                LRUCache(maxsize=bad)

    def test_clear_drops_entries_but_keeps_counters(self):
        cache = LRUCache(maxsize=4)
        cache.put("a", 1)
        cache.get("a")
        cache.get("nope")
        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 1)

    def test_stats(self):
        cache = LRUCache(maxsize=2)
        cache.put("a", 1)
        cache.get("a")
        cache.get("nope")
        self.assertEqual(cache.stats(), {"size": 1, "maxsize": 2,
                                         "hits": 1, "misses": 1,
                                         "evictions": 0})


class TestBM25Postings(unittest.TestCase):
    def setUp(self):
        self.texts = [text for _, text in DOCS]
        self.scorer = BM25Scorer()
        self.scorer.fit(self.texts)

    def test_score_all_matches_per_document_scoring(self):
        for query in QUERIES:
            expected = [self.scorer.score(query, i)
                        for i in range(len(self.texts))]
            actual = self.scorer.score_all(query)
            self.assertEqual(len(actual), len(expected))
            for got, want in zip(actual, expected):
                self.assertAlmostEqual(got, want, places=9,
                                       msg="query %r" % query)

    def test_unknown_terms_score_zero(self):
        self.assertEqual(self.scorer.score_all("zzzterm qqq"),
                         [0.0] * len(self.texts))

    def test_postings_cover_the_terms(self):
        # "sling" appears twice in the first document, "cache" three
        # times in the fourth ("caching" is a different token).
        self.assertEqual(self.scorer.postings_["sling"], [(0, 2)])
        self.assertEqual(self.scorer.postings_["cache"], [(3, 3)])

    def test_empty_corpus_scores_empty(self):
        scorer = BM25Scorer()
        scorer.fit([])
        self.assertEqual(scorer.score_all("anything"), [])

    def test_from_dict_rebuilds_postings(self):
        restored = BM25Scorer.from_dict(self.scorer.to_dict())
        self.assertEqual(restored.postings_, self.scorer.postings_)
        for query in QUERIES:
            self.assertEqual(restored.score_all(query),
                             self.scorer.score_all(query))


class TestInvertedIndex(unittest.TestCase):
    def _reference_tfidf(self, index, query):
        """Brute-force TF-IDF cosine scores over every chunk."""
        qvec = index.vectorizer.transform(query)
        if not qvec:
            return []
        return [cosine_similarity(qvec, vec) for vec in index._vectors]

    def test_tfidf_search_matches_brute_force(self):
        index = _make_index()
        for query in QUERIES:
            hits = index.search(query, k=5, method="tfidf")
            expected = self._reference_tfidf(index, query)
            # Same ranking as search(): score descending, ties keep
            # chunk order (search uses a stable sort on score).
            ranked = sorted(
                ((score, i) for i, score in enumerate(expected)
                 if score > 0.0),
                key=lambda item: (-item[0], item[1]))
            self.assertEqual(
                [(h["doc_id"], h["chunk_index"]) for h in hits],
                [(index.chunks[i]["doc_id"], index.chunks[i]["chunk_index"])
                 for _, i in ranked[:5]],
                msg="query %r" % query)
            for hit, (_, i) in zip(hits, ranked[:5]):
                self.assertAlmostEqual(hit["score"], round(expected[i], 4),
                                       places=4, msg="query %r" % query)

    def test_all_methods_match_brute_force_rankings(self):
        # BM25 and combined have no second reference implementation
        # here, but their top hits must be stable across rebuilds
        # and identical after a save/load round trip.
        index = _make_index()
        for method in ("bm25", "combined"):
            for query in QUERIES:
                first = index.search(query, k=5, method=method)
                index.build()  # refit + rebuild postings
                second = index.search(query, k=5, method=method)
                self.assertEqual(first, second,
                                 msg="%s query %r" % (method, query))

    def test_loaded_index_searches_identically(self):
        index = _make_index()
        restored = Index.from_dict(index.to_dict())
        for method in ("tfidf", "bm25", "combined"):
            for query in QUERIES:
                self.assertEqual(
                    index.search(query, k=5, method=method),
                    restored.search(query, k=5, method=method),
                    msg="%s query %r" % (method, query))

    def test_postings_built_on_load(self):
        index = _make_index()
        restored = Index.from_dict(index.to_dict())
        self.assertTrue(restored._postings)
        self.assertIn("sling", restored._postings)


class TestQueryCache(unittest.TestCase):
    def test_repeat_search_is_a_hit(self):
        index = _make_index()
        first = index.search("What is AEM built on?")
        info = index.cache_info()
        self.assertEqual((info["hits"], info["misses"]), (0, 1))
        second = index.search("What is AEM built on?")
        self.assertEqual(first, second)
        info = index.cache_info()
        self.assertEqual((info["hits"], info["misses"]), (1, 1))
        self.assertEqual(info["size"], 1)

    def test_cache_key_normalises_case_and_whitespace(self):
        index = _make_index()
        index.search("What is AEM built on?")
        index.search("  what   IS aem built on? ")
        self.assertEqual(index.cache_info()["hits"], 1)

    def test_k_and_method_are_separate_entries(self):
        index = _make_index()
        index.search("AEM Sling", k=5, method="tfidf")
        index.search("AEM Sling", k=3, method="tfidf")
        index.search("AEM Sling", k=5, method="bm25")
        info = index.cache_info()
        self.assertEqual(info["size"], 3)
        self.assertEqual(info["misses"], 3)
        self.assertEqual(info["hits"], 0)

    def test_add_document_invalidates(self):
        index = _make_index()
        index.search("AEM Sling")
        index.add_document("new-doc",
                           "AEM Sling extra document about Sling nodes.")
        self.assertEqual(index.cache_info()["size"], 0)
        hits = index.search("AEM Sling")
        self.assertEqual(index.cache_info()["misses"], 2)
        self.assertIn("new-doc", [h["doc_id"] for h in hits])

    def test_remove_document_invalidates(self):
        index = _make_index()
        index.search("AEM Sling")
        index.remove_document("aem-basics")
        self.assertEqual(index.cache_info()["size"], 0)
        hits = index.search("AEM Sling")
        self.assertNotIn("aem-basics", [h["doc_id"] for h in hits])

    def test_cached_results_are_copies(self):
        index = _make_index()
        hits = index.search("AEM Sling")
        hits[0]["score"] = 999.0
        hits[0]["metadata"]["title"] = "poisoned"
        fresh = index.search("AEM Sling")
        self.assertNotEqual(fresh[0]["score"], 999.0)
        self.assertNotEqual(fresh[0]["metadata"].get("title"), "poisoned")

    def test_cache_size_zero_disables_caching(self):
        index = _make_index(cache_size=0)
        index.search("AEM Sling")
        index.search("AEM Sling")
        info = index.cache_info()
        self.assertFalse(info["enabled"])
        self.assertEqual(info["size"], 0)
        self.assertEqual((info["hits"], info["misses"]), (0, 2))

    def test_set_cache_size_resizes_and_clears(self):
        index = _make_index(cache_size=2)
        index.search("AEM Sling")
        index.set_cache_size(10)
        self.assertEqual(index.cache_size, 10)
        info = index.cache_info()
        self.assertEqual(info["size"], 0)
        self.assertEqual((info["hits"], info["misses"]), (0, 0))

    def test_set_cache_size_rejects_bad_values(self):
        index = _make_index()
        for bad in (-1, "big", 1.5, True):
            with self.assertRaises(ValueError, msg="size=%r" % (bad,)):
                index.set_cache_size(bad)

    def test_lru_eviction(self):
        index = _make_index(cache_size=2)
        index.search("AEM Sling")          # cached
        index.search("Spring Boot")        # cached
        index.search("overfitting")        # evicts "AEM Sling"
        self.assertEqual(index.cache_info()["evictions"], 1)
        before = index.cache_info()["misses"]
        index.search("AEM Sling")          # miss: it was evicted
        self.assertEqual(index.cache_info()["misses"], before + 1)

    def test_empty_index_search_does_not_cache_crash(self):
        index = Index()
        self.assertEqual(index.search("anything"), [])
        self.assertEqual(index.cache_info()["size"], 0)

    def test_invalid_cache_size_rejected(self):
        for bad in (-1, "big", 1.5, True):
            with self.assertRaises(ValueError, msg="size=%r" % (bad,)):
                Index(cache_size=bad)

    def test_cache_survives_save_load(self):
        index = _make_index()
        index.search("AEM Sling")
        restored = Index.from_dict(index.to_dict())
        self.assertEqual(restored.cache_info()["size"], 0)
        self.assertEqual(restored.search("AEM Sling"),
                         index.search("AEM Sling"))


class TestCacheConfig(unittest.TestCase):
    def test_default_cache_size(self):
        self.assertEqual(DEFAULT_CONFIG["retrieval"]["query_cache_size"],
                         128)

    def test_config_file_sets_cache_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"retrieval": {"query_cache_size": 0}}, fh)
            config = load_config(path=path, env={})
        self.assertEqual(config["retrieval"]["query_cache_size"], 0)

    def test_invalid_cache_size_rejected(self):
        for bad in (-1, 10001, "big", 1.5, True):
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "config.json")
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump({"retrieval": {"query_cache_size": bad}}, fh)
                with self.assertRaises(ValueError,
                                       msg="size=%r" % (bad,)):
                    load_config(path=path, env={})

    def test_env_override_sets_cache_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(base_dir=tmp, env={
                "DOCUMIND_QUERY_CACHE_SIZE": "7"})
        self.assertEqual(config["retrieval"]["query_cache_size"], 7)

    def test_env_override_rejects_non_integer(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_config(base_dir=tmp, env={
                    "DOCUMIND_QUERY_CACHE_SIZE": "lots"})


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        self.index = _make_index()

    def test_report_shape(self):
        report = run_benchmark(self.index, QUERIES[:3], warm_repeats=2)
        self.assertEqual(report["documents"], len(DOCS))
        self.assertEqual(report["chunks"], len(self.index))
        self.assertEqual(report["questions"], 3)
        self.assertGreaterEqual(report["build_ms"], 0.0)
        self.assertEqual(sorted(report["methods"]),
                         ["bm25", "combined", "tfidf"])
        for method, stats in report["methods"].items():
            self.assertGreaterEqual(stats["cold_ms_per_search"], 0.0)
            self.assertGreaterEqual(stats["warm_ms_per_search"], 0.0)
            # Every warm pass hits the cache: 3 questions x 2 passes.
            self.assertEqual(stats["cache_hits"], 3 * 2)
            self.assertEqual(stats["cache_misses"], 0)
        self.assertIn("cache", report)

    def test_single_method_subset(self):
        report = run_benchmark(self.index, QUERIES[:1],
                               methods=["bm25"], warm_repeats=1)
        self.assertEqual(sorted(report["methods"]), ["bm25"])

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            run_benchmark(self.index, [])
        with self.assertRaises(ValueError):
            run_benchmark(self.index, QUERIES[:1], methods=["nope"])
        with self.assertRaises(ValueError):
            run_benchmark(self.index, QUERIES[:1], warm_repeats=0)

    def test_scale_corpus(self):
        docs = [(doc_id, text) for doc_id, text in DOCS]
        before = len(self.index)
        added = scale_corpus(self.index, docs, 2)
        self.assertEqual(added, 2 * len(DOCS))
        self.assertEqual(len(self.index), before * 3)
        ids = {c["doc_id"] for c in self.index.chunks}
        self.assertIn("aem-basics--copy-2", ids)

    def test_scale_corpus_rejects_bad_copies(self):
        with self.assertRaises(ValueError):
            scale_corpus(self.index, DOCS, -1)
        with self.assertRaises(ValueError):
            scale_corpus(self.index, DOCS, "many")

    def test_format_benchmark(self):
        report = run_benchmark(self.index, QUERIES[:2], warm_repeats=1)
        text = format_benchmark(report)
        for fragment in ("DocuMind benchmark", "tfidf", "bm25",
                         "combined", "query cache"):
            self.assertIn(fragment, text)

    def test_bench_cli_runs(self):
        result = subprocess.run(
            [sys.executable, "cli.py", "bench", "--scale", "2",
             "--warm-repeats", "1"],
            cwd=BASE_DIR, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DocuMind benchmark", result.stdout)
        self.assertIn("tfidf", result.stdout)

    def test_bench_cli_rejects_bad_scale(self):
        result = subprocess.run(
            [sys.executable, "cli.py", "bench", "--scale", "0"],
            cwd=BASE_DIR, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("scale", result.stderr)


class TestServerCacheStats(unittest.TestCase):
    def test_stats_reports_cache_and_searches_hit_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = create_server(tmp, "127.0.0.1", 0)
            try:
                app = server.documind_app
                status, _ = app.ingest_text(
                    "aem-basics", "AEM basics", DOCS[0][1])
                self.assertEqual(status, 201)
                stats = app.stats()
                self.assertIn("cache", stats)
                self.assertTrue(stats["cache"]["enabled"])
                app.search("AEM Sling")
                app.search("AEM Sling")
                self.assertEqual(app.stats()["cache"]["hits"], 1)
            finally:
                server.server_close()
                app.close()


if __name__ == "__main__":
    unittest.main()
