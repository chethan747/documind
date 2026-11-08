"""The DocuMind index: stores document chunks and searches them.

Documents are chunked on the way in; a TF-IDF vectoriser is fitted over all
chunks and a BM25 scorer over the same chunks (Day 3). A query can be
ranked by TF-IDF cosine similarity (the default, unchanged from Day 1),
by BM25, or by a normalised combination of both. The whole index
serialises to JSON so the CLI can persist it between runs, together with
per-document metadata (Day 4: title, source path, ...) so a loaded index
alone can report where each hit came from. The saved format is versioned
and validated on load, so a corrupt or newer file fails with a clear
error instead of a cryptic one.

Day 18 — performance: both rankers now score through an inverted
index (term -> chunks containing it), so a query only touches chunks
that share one of its terms instead of scanning the whole corpus, and
repeated searches are served from a bounded LRU query cache that is
invalidated whenever the index changes. Neither changes results:
chunks with no shared query term score 0.0 and are filtered out
either way.
"""
import json

from .bm25 import BM25Scorer
from .cache import LRUCache
from .chunker import chunk_text
from .embeddings import TfidfVectorizer
from .validation import validate_k, validate_query

# Ranking methods accepted by Index.search / qa.answer_question.
SEARCH_METHODS = ("tfidf", "bm25", "combined")

# Saved-index format version. Version 1 (Days 1-3) had no version field and
# no per-document metadata; version 2 adds both. Older files still load.
INDEX_FORMAT_VERSION = 2


class Index:
    def __init__(self, chunk_size=500, overlap=80, cache_size=128):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []        # [{"doc_id", "chunk_index", "text"}]
        self.documents = {}     # doc_id -> metadata dict (title, source_path, ...)
        self.vectorizer = TfidfVectorizer()
        self.bm25 = BM25Scorer()
        self._vectors = []      # parallel to self.chunks
        self._dirty = True
        # Day 18: term -> [(chunk_index, tf-idf weight)] over the
        # fitted vectors, rebuilt by build(); and the query cache.
        self._postings = {}
        self.cache_size = cache_size
        self._cache = LRUCache(cache_size)

    # -- caching -------------------------------------------------------
    def set_cache_size(self, cache_size):
        """Change the query-cache capacity (0 disables caching).

        Replaces the cache, dropping its entries and resetting its
        hit/miss counters, so a resized cache never serves results
        cached under the old limit.
        """
        self.cache_size = cache_size
        self._cache = LRUCache(cache_size)

    def cache_info(self):
        """``{"enabled", "size", "maxsize", "hits", "misses",
        "evictions"}`` for the query cache (see cache.py)."""
        info = self._cache.stats()
        info["enabled"] = self.cache_size > 0
        return info

    @staticmethod
    def _cache_key(query, k, method):
        """Cache key for a search: tokenisation lowercases and
        ignores whitespace runs, so the key does the same — "AEM"
        and "aem" are the same query and share one cache entry."""
        return (" ".join(query.split()).lower(), k, method)

    @staticmethod
    def _copy_hits(hits):
        """Fresh copies of search hits, so a caller mutating a
        returned hit can never poison the cached originals."""
        return [{**hit, "metadata": dict(hit.get("metadata", {}))}
                for hit in hits]

    def _rebuild_postings(self):
        """(Re)build the TF-IDF inverted index from ``_vectors``."""
        postings = {}
        for chunk_index, vec in enumerate(self._vectors):
            for term, weight in vec.items():
                postings.setdefault(term, []).append((chunk_index, weight))
        self._postings = postings

    # -- ingestion -------------------------------------------------------
    def add_document(self, doc_id, text, metadata=None):
        """Chunk ``text`` and add the pieces under ``doc_id``.

        ``metadata`` (e.g. title, source path) is stored per document and
        travels with the saved index, so search hits can report where they
        came from without the document store. Passing None leaves any
        existing metadata for ``doc_id`` untouched.
        """
        pieces = chunk_text(text, self.chunk_size, self.overlap)
        for i, piece in enumerate(pieces):
            self.chunks.append({"doc_id": doc_id,
                                "chunk_index": i,
                                "text": piece})
        if metadata is not None:
            self.documents[doc_id] = dict(metadata)
        self._dirty = True
        # The index contents changed, so cached results are stale
        # even before the next build() clears them again.
        self._cache.clear()
        return len(pieces)

    def remove_document(self, doc_id):
        """Drop every chunk (and vector) belonging to ``doc_id``.

        Used by the ingestion pipeline so re-ingesting a changed file does
        not leave stale chunks behind. Returns the number of chunks removed.
        """
        if self._dirty:
            # _vectors must stay parallel to chunks before filtering
            self.build()
        before = len(self.chunks)
        if before == 0:
            return 0
        keep = [(c, v) for c, v in zip(self.chunks, self._vectors)
                if c["doc_id"] != doc_id]
        self.chunks = [c for c, _ in keep]
        self._vectors = [v for _, v in keep]
        removed = before - len(self.chunks)
        if removed:
            self._dirty = True
            self.documents.pop(doc_id, None)
            # Contents changed: cached results are stale even before
            # the next build() clears them again.
            self._cache.clear()
        return removed

    def build(self):
        """(Re)fit the TF-IDF vectoriser and BM25 scorer over all chunks.

        Also rebuilds the TF-IDF inverted index and clears the query
        cache: a build means the index contents may have changed, so
        cached results from before it must not be served.
        """
        if self.chunks:
            texts = [c["text"] for c in self.chunks]
            self._vectors = self.vectorizer.fit_transform(texts)
            self.bm25.fit(texts)
        else:
            self._vectors = []
            self.bm25.fit([])
        self._rebuild_postings()
        self._cache.clear()
        self._dirty = False

    # -- retrieval -------------------------------------------------------
    def search(self, query, k=5, method="tfidf"):
        """Return the top-k chunks for ``query``.

        ``method`` is ``"tfidf"`` (cosine similarity, the Day 1 default),
        ``"bm25"`` (BM25 score), or ``"combined"`` (the mean of the two
        scores after each is normalised by its own top score, so the two
        scales are comparable). Raises ValueError for an unknown method,
        a non-integer or out-of-range ``k`` (see validation.py), or a
        blank / non-string query.
        """
        if method not in SEARCH_METHODS:
            raise ValueError("unknown search method %r (expected one of %s)"
                             % (method, ", ".join(SEARCH_METHODS)))
        k = validate_k(k)
        query = validate_query(query)
        if self._dirty:
            self.build()
        if not self.chunks:
            return []
        cache_key = self._cache_key(query, k, method)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return self._copy_hits(cached)
        tfidf_scores = None
        bm25_scores = None
        if method in ("tfidf", "combined"):
            qvec = self.vectorizer.transform(query)
            if qvec:
                # Score through the inverted index: only chunks
                # sharing a query term can score above 0.0, and
                # cosine similarity is a dot product over the
                # shared terms, so accumulating per posting gives
                # exactly the old per-chunk dot products.
                scored = {}
                for term, weight in qvec.items():
                    for chunk_index, chunk_weight in \
                            self._postings.get(term, ()):
                        scored[chunk_index] = (
                            scored.get(chunk_index, 0.0)
                            + weight * chunk_weight)
                tfidf_scores = [scored.get(i, 0.0)
                                for i in range(len(self.chunks))]
        if method in ("bm25", "combined"):
            bm25_scores = self.bm25.score_all(query)
        if method == "combined":
            if tfidf_scores is None and bm25_scores is None:
                return []
            tfidf_scores = tfidf_scores or [0.0] * len(self.chunks)
            bm25_scores = bm25_scores or [0.0] * len(self.chunks)
            tfidf_max = max(tfidf_scores) or 1.0
            bm25_max = max(bm25_scores) or 1.0
            scores = [0.5 * (t / tfidf_max) + 0.5 * (b / bm25_max)
                      for t, b in zip(tfidf_scores, bm25_scores)]
        elif method == "tfidf":
            if tfidf_scores is None:
                return []
            scores = tfidf_scores
        else:  # bm25
            scores = bm25_scores
        scored = []
        for chunk, score in zip(self.chunks, scores):
            if score > 0.0:
                scored.append({**chunk, "score": round(score, 4),
                               "metadata": dict(
                                   self.documents.get(chunk["doc_id"], {}))})
        scored.sort(key=lambda r: r["score"], reverse=True)
        results = scored[:k]
        self._cache.put(cache_key, results)
        return self._copy_hits(results)

    # -- persistence -----------------------------------------------------
    def to_dict(self):
        if self._dirty:
            self.build()
        return {"format_version": INDEX_FORMAT_VERSION,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "chunks": self.chunks,
                "documents": self.documents,
                "vectorizer": self.vectorizer.to_dict(),
                "vectors": self._vectors,
                "bm25": self.bm25.to_dict()}

    @classmethod
    def from_dict(cls, data, cache_size=None):
        """Rebuild an index from a saved dict.

        Raises ValueError for data that is not a recognisable index
        (wrong shape, missing keys, or a newer format version), so a
        corrupt or foreign file fails loudly instead of half-loading.
        Indexes saved before Day 4 (no version, no documents map, maybe
        no BM25 state) still load; their metadata defaults to empty and
        the BM25 scorer refits on first use. ``cache_size`` sets the
        query-cache capacity of the rebuilt index (default 128).
        """
        if not isinstance(data, dict):
            raise ValueError("invalid index data: expected a JSON object")
        version = data.get("format_version", 1)
        if version > INDEX_FORMAT_VERSION:
            raise ValueError(
                "unsupported index format version %r (this build supports "
                "up to %d; re-ingest to rebuild the index)"
                % (version, INDEX_FORMAT_VERSION))
        for key in ("chunk_size", "overlap", "chunks",
                    "vectorizer", "vectors"):
            if key not in data:
                raise ValueError("invalid index data: missing key %r" % key)
        idx = cls(data["chunk_size"], data["overlap"],
                  cache_size=128 if cache_size is None else cache_size)
        idx.chunks = data["chunks"]
        idx.documents = dict(data.get("documents") or {})
        idx.vectorizer = TfidfVectorizer.from_dict(data["vectorizer"])
        idx._vectors = data["vectors"]
        if "bm25" in data:
            idx.bm25 = BM25Scorer.from_dict(data["bm25"])
            idx._dirty = False
            # The saved vectors are trusted, so no build() runs on
            # first use — rebuild the inverted index here instead.
            idx._rebuild_postings()
        else:
            # Indexes saved before Day 3 have no BM25 state; refit on use.
            idx._dirty = True
        return idx

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load(cls, path, cache_size=None):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError("corrupt index file %r: not valid JSON (%s)"
                             % (path, exc)) from exc
        return cls.from_dict(data, cache_size=cache_size)

    def __len__(self):
        return len(self.chunks)
