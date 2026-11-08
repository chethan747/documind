"""Benchmarks for the DocuMind index (Day 18).

Measures what the Day 18 performance pass changed, instead of
asserting it: index build time, cold search latency per ranking
method, and warm (cache-hit) latency for the same questions, plus
the cache hit rate. Timings use ``time.perf_counter`` and are
reported as milliseconds per search, averaged over one cold pass
and several warm passes.

The shipped sample corpus is tiny (3 documents, ~15 chunks), so
timings on it are sub-millisecond noise. :func:`scale_corpus`
grows an index by adding near-duplicate copies of its documents
under new ids — a load test for retrieval *speed*, not a quality
evaluation — so ``cli.py bench`` reports numbers that actually
discriminate between a full-corpus scan and the inverted index.
"""
import time

from .index import SEARCH_METHODS


def scale_corpus(index, documents, copies):
    """Add ``copies`` extra copies of each ``(doc_id, text)`` pair
    to ``index`` under new ids (``<doc_id>--copy-<n>``).

    The copies are near-duplicates by design: this is a load test,
    so the vocabulary stays the same while the chunk count grows.
    Returns the number of documents added.
    """
    if isinstance(copies, bool) or not isinstance(copies, int):
        raise ValueError("copies must be an integer, got %r" % (copies,))
    if copies < 0:
        raise ValueError("copies must be at least 0, got %d" % copies)
    added = 0
    for doc_id, text in documents:
        for n in range(1, copies + 1):
            index.add_document("%s--copy-%d" % (doc_id, n), text)
            added += 1
    return added


def _time_pass(index, questions, method, k):
    """Seconds to search every question once with ``method``."""
    start = time.perf_counter()
    for question in questions:
        index.search(question, k=k, method=method)
    return time.perf_counter() - start


def run_benchmark(index, questions, methods=SEARCH_METHODS, k=5,
                  warm_repeats=5):
    """Benchmark ``index`` over ``questions``; return a report dict.

    ``methods`` is the ranking methods to time (default: all three).
    The report carries the corpus size, the build time, and per
    method the cold (first, cache-miss) and warm (cache-hit)
    milliseconds per search, the cold/warm speedup, and the cache
    hits/misses observed during the warm passes. ``warm_repeats``
    must be at least 1. The index is left built, with its cache
    warm — callers that need a cold cache afterwards should clear
    it (``index.set_cache_size(index.cache_size)`` rebuilds it).
    """
    if not questions:
        raise ValueError("benchmark needs at least one question")
    if isinstance(warm_repeats, bool) or not isinstance(warm_repeats, int):
        raise ValueError("warm_repeats must be an integer, got %r"
                         % (warm_repeats,))
    if warm_repeats < 1:
        raise ValueError("warm_repeats must be at least 1, got %d"
                         % warm_repeats)
    methods = tuple(methods)
    for method in methods:
        if method not in SEARCH_METHODS:
            raise ValueError("unknown search method %r (expected one of "
                             "%s)" % (method, ", ".join(SEARCH_METHODS)))

    start = time.perf_counter()
    index.build()  # refits, rebuilds the postings, clears the cache
    build_ms = (time.perf_counter() - start) * 1000.0

    report = {"documents": len({c["doc_id"] for c in index.chunks}),
              "chunks": len(index),
              "questions": len(questions),
              "k": k,
              "warm_repeats": warm_repeats,
              "build_ms": round(build_ms, 3),
              "methods": {}}
    n_searches = len(questions)
    for method in methods:
        # Cold pass: the build above cleared the cache, so every
        # search here is a miss and pays the full scoring cost.
        cold_s = _time_pass(index, questions, method, k)
        before = index.cache_info()
        warm_s = 0.0
        for _ in range(warm_repeats):
            warm_s += _time_pass(index, questions, method, k)
        after = index.cache_info()
        cold_ms = cold_s / n_searches * 1000.0
        warm_ms = warm_s / (n_searches * warm_repeats) * 1000.0
        report["methods"][method] = {
            "cold_ms_per_search": round(cold_ms, 3),
            "warm_ms_per_search": round(warm_ms, 3),
            "speedup": (round(cold_ms / warm_ms, 1)
                        if warm_ms > 0 else None),
            "cache_hits": after["hits"] - before["hits"],
            "cache_misses": after["misses"] - before["misses"],
        }
    report["cache"] = index.cache_info()
    return report


def format_benchmark(report):
    """Render a benchmark report as a plain-text table."""
    lines = [
        "DocuMind benchmark — %d documents, %d chunks, %d questions "
        "(k=%d)" % (report["documents"], report["chunks"],
                    report["questions"], report["k"]),
        "build (fit + inverted index): %.3f ms" % report["build_ms"],
        "",
        "%-10s %18s %18s %10s %12s" % (
            "method", "cold ms / search", "warm ms / search",
            "speedup", "cache hits"),
    ]
    for method, stats in report["methods"].items():
        speedup = ("%.1fx" % stats["speedup"]
                   if stats["speedup"] is not None else "n/a")
        lines.append("%-10s %18.3f %18.3f %10s %12d" % (
            method, stats["cold_ms_per_search"],
            stats["warm_ms_per_search"], speedup,
            stats["cache_hits"]))
    lines.append("")
    cache = report["cache"]
    lines.append("query cache: %s, %d entries, %d hits / %d misses, "
                 "%d evictions"
                 % ("enabled (max %d)" % cache["maxsize"]
                    if cache["enabled"] else "disabled",
                    cache["size"], cache["hits"], cache["misses"],
                    cache["evictions"]))
    return "\n".join(lines) + "\n"
