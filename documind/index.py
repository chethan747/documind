"""The DocuMind index: stores document chunks and searches them.

Documents are chunked on the way in; a TF-IDF vectoriser is fitted over all
chunks; a query is vectorised with the same vocabulary and ranked against
every chunk by cosine similarity. The whole index serialises to JSON so the
CLI can persist it between runs.
"""
import json

from .chunker import chunk_text
from .embeddings import TfidfVectorizer, cosine_similarity


class Index:
    def __init__(self, chunk_size=500, overlap=80):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []        # [{"doc_id", "chunk_index", "text"}]
        self.vectorizer = TfidfVectorizer()
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
        """(Re)fit the TF-IDF vectoriser over all stored chunks."""
        if self.chunks:
            self._vectors = self.vectorizer.fit_transform(
                [c["text"] for c in self.chunks])
        else:
            self._vectors = []
        self._dirty = False

    # -- retrieval -------------------------------------------------------
    def search(self, query, k=5):
        """Return the top-k chunks for ``query`` with cosine scores."""
        if self._dirty:
            self.build()
        if not self.chunks:
            return []
        qvec = self.vectorizer.transform(query)
        if not qvec:
            return []
        scored = []
        for chunk, vec in zip(self.chunks, self._vectors):
            score = cosine_similarity(qvec, vec)
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
                "vectors": self._vectors}

    @classmethod
    def from_dict(cls, data):
        idx = cls(data["chunk_size"], data["overlap"])
        idx.chunks = data["chunks"]
        idx.vectorizer = TfidfVectorizer.from_dict(data["vectorizer"])
        idx._vectors = data["vectors"]
        idx._dirty = False
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
