"""The DocuMind index: stores document chunks and searches them.

Documents are chunked on the way in; a TF-IDF vectoriser is fitted over all
chunks and a BM25 scorer over the same chunks (Day 3). A query can be
ranked by TF-IDF cosine similarity (the default, unchanged from Day 1),
by BM25, or by a normalised combination of both. The whole index
serialises to JSON so the CLI can persist it between runs.
"""
import json

from .bm25 import BM25Scorer
from .chunker import chunk_text
from .embeddings import TfidfVectorizer, cosine_similarity

# Ranking methods accepted by Index.search / qa.answer_question.
SEARCH_METHODS = ("tfidf", "bm25", "combined")


class Index:
    def __init__(self, chunk_size=500, overlap=80):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []        # [{"doc_id", "chunk_index", "text"}]
        self.vectorizer = TfidfVectorizer()
        self.bm25 = BM25Scorer()
        self._vectors = []      # parallel to self.chunks
        self._dirty = True

    # -- ingestion -------------------------------------------------------
    def add_document(self, doc_id, text):
        """Chunk ``text`` and add the pieces under ``doc_id``."""
        pieces = chunk_text(text, self.chunk_size, self.overlap)
        for i, piece in enumerate(pieces):
            self.chunks.append({"doc_id": doc_id,
                                "chunk_index": i,
                                "text": piece})
        self._dirty = True
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
        return removed

    def build(self):
        """(Re)fit the TF-IDF vectoriser and BM25 scorer over all chunks."""
        if self.chunks:
            texts = [c["text"] for c in self.chunks]
            self._vectors = self.vectorizer.fit_transform(texts)
            self.bm25.fit(texts)
        else:
            self._vectors = []
            self.bm25.fit([])
        self._dirty = False

    # -- retrieval -------------------------------------------------------
    def search(self, query, k=5, method="tfidf"):
        """Return the top-k chunks for ``query``.

        ``method`` is ``"tfidf"`` (cosine similarity, the Day 1 default),
        ``"bm25"`` (BM25 score), or ``"combined"`` (the mean of the two
        scores after each is normalised by its own top score, so the two
        scales are comparable). Raises ValueError for an unknown method.
        """
        if method not in SEARCH_METHODS:
            raise ValueError("unknown search method %r (expected one of %s)"
                             % (method, ", ".join(SEARCH_METHODS)))
        if self._dirty:
            self.build()
        if not self.chunks:
            return []
        tfidf_scores = None
        bm25_scores = None
        if method in ("tfidf", "combined"):
            qvec = self.vectorizer.transform(query)
            if qvec:
                tfidf_scores = [cosine_similarity(qvec, vec)
                                for vec in self._vectors]
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
                scored.append({**chunk, "score": round(score, 4)})
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:k]

    # -- persistence -----------------------------------------------------
    def to_dict(self):
        if self._dirty:
            self.build()
        return {"chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "chunks": self.chunks,
                "vectorizer": self.vectorizer.to_dict(),
                "vectors": self._vectors,
                "bm25": self.bm25.to_dict()}

    @classmethod
    def from_dict(cls, data):
        idx = cls(data["chunk_size"], data["overlap"])
        idx.chunks = data["chunks"]
        idx.vectorizer = TfidfVectorizer.from_dict(data["vectorizer"])
        idx._vectors = data["vectors"]
        if "bm25" in data:
            idx.bm25 = BM25Scorer.from_dict(data["bm25"])
            idx._dirty = False
        else:
            # Indexes saved before Day 3 have no BM25 state; refit on use.
            idx._dirty = True
        return idx

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def __len__(self):
        return len(self.chunks)
