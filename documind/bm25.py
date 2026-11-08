"""BM25 ranking built from scratch (no numpy, no scikit-learn).

BM25 complements the TF-IDF cosine ranker in :mod:`documind.embeddings`:

- Term-frequency *saturates*: a term appearing 10 times in a chunk is
  worth more than a term appearing once, but not 10x more. The ``k1``
  parameter controls how quickly that saturation kicks in.
- Length normalisation: a match in a short chunk counts for more than
  the same match in a long chunk. The ``b`` parameter controls how
  strongly chunk length is penalised (``b=1`` is full normalisation,
  ``b=0`` ignores length entirely).
- IDF follows the Robertson/Sparck-Jones form with a ``+1`` floor so
  terms present in every chunk still contribute a small positive weight
  instead of going negative.

Scoring for one query term against one chunk is::

    idf(term) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len))

summed over the distinct query terms. Defaults (``k1=1.5, b=0.75``)
are the widely used BM25 defaults.
"""
import math

from .embeddings import tokenize


class BM25Scorer:
    """Fits BM25 statistics over a corpus and scores queries."""

    def __init__(self, k1=1.5, b=0.75):
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self.n_documents_ = 0
        self.avgdl_ = 0.0
        self.doc_lengths_ = []     # tokens per document, parallel to corpus
        self.doc_term_freqs_ = []  # list of {term: count} per document
        self.idf_ = {}             # term -> idf weight
        self.postings_ = {}        # term -> [(doc_index, term count)]
        # Day 18: an inverted index over the fitted corpus. score_all
        # only touches documents that share a query term instead of
        # scoring every document for every query; documents with no
        # query term score 0.0 either way, so results are unchanged.

    def _build_postings(self):
        """(Re)build the inverted index from ``doc_term_freqs_``."""
        postings = {}
        for doc_index, counts in enumerate(self.doc_term_freqs_):
            for term, tf in counts.items():
                postings.setdefault(term, []).append((doc_index, tf))
        self.postings_ = postings

    def fit(self, documents):
        """Learn document frequencies, lengths and idf from ``documents``."""
        docs_tokens = [tokenize(d) for d in documents]
        self.n_documents_ = len(documents)
        self.doc_lengths_ = [len(tokens) for tokens in docs_tokens]
        self.doc_term_freqs_ = []
        df = {}
        for tokens in docs_tokens:
            counts = {}
            for term in tokens:
                counts[term] = counts.get(term, 0) + 1
            self.doc_term_freqs_.append(counts)
            for term in counts:
                df[term] = df.get(term, 0) + 1
        n = self.n_documents_
        self.avgdl_ = (sum(self.doc_lengths_) / n) if n else 0.0
        self.idf_ = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self._build_postings()
        return self

    def score(self, query, doc_index):
        """BM25 score of ``query`` against one fitted document."""
        if not self.doc_term_freqs_ or not (0 <= doc_index < len(self.doc_term_freqs_)):
            return 0.0
        if self.avgdl_ == 0.0:
            return 0.0
        counts = self.doc_term_freqs_[doc_index]
        dl = self.doc_lengths_[doc_index]
        norm = 1.0 - self.b + self.b * (dl / self.avgdl_)
        total = 0.0
        for term in set(tokenize(query)):
            tf = counts.get(term)
            if not tf:
                continue
            idf = self.idf_.get(term, 0.0)
            total += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
        return total

    def score_all(self, query):
        """BM25 scores of ``query`` against every fitted document.

        The query is tokenised once (Day 18: the old version
        re-tokenised it inside :meth:`score` for every document),
        and only documents sharing at least one query term are
        scored — the rest score 0.0 by definition, so the returned
        list is identical to scoring every document, just faster.
        """
        scores = [0.0] * self.n_documents_
        if not self.doc_term_freqs_ or self.avgdl_ == 0.0:
            return scores
        for term in set(tokenize(query)):
            postings = self.postings_.get(term)
            if not postings:
                continue
            idf = self.idf_.get(term, 0.0)
            for doc_index, tf in postings:
                dl = self.doc_lengths_[doc_index]
                norm = 1.0 - self.b + self.b * (dl / self.avgdl_)
                scores[doc_index] += (
                    idf * (tf * (self.k1 + 1.0))
                    / (tf + self.k1 * norm))
        return scores

    def to_dict(self):
        return {"k1": self.k1,
                "b": self.b,
                "n_documents": self.n_documents_,
                "avgdl": self.avgdl_,
                "doc_lengths": self.doc_lengths_,
                "doc_term_freqs": self.doc_term_freqs_,
                "idf": self.idf_}

    @classmethod
    def from_dict(cls, data):
        scorer = cls(data.get("k1", 1.5), data.get("b", 0.75))
        scorer.n_documents_ = data["n_documents"]
        scorer.avgdl_ = data["avgdl"]
        scorer.doc_lengths_ = list(data["doc_lengths"])
        scorer.doc_term_freqs_ = [
            {term: int(count) for term, count in counts.items()}
            for counts in data["doc_term_freqs"]
        ]
        scorer.idf_ = dict(data["idf"])
        # Postings are not persisted (they rebuild cheaply from the
        # term frequencies); indexes saved before Day 18 load the
        # same way.
        scorer._build_postings()
        return scorer
