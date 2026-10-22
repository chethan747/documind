"""TF-IDF vectoriser built from scratch (no numpy, no scikit-learn).

Each document becomes a sparse vector {term: weight}, where the weight is
term-frequency x inverse-document-frequency. Vectors are L2-normalised, so
cosine similarity between two vectors is just their dot product — which is
what the index uses to rank chunks against a question.
"""
import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small stopword list: very common words that carry no topical signal.
STOPWORDS = frozenset("""
a an the and or of to in on for with is are was were be been being by at as
it its this that these those from what which who whom how when where why
do does did can could should would will shall may might must not no yes
i you he she we they me him her us them my your his their our mine yours
""".split())


def tokenize(text):
    """Lowercase, split into alphanumeric tokens, drop stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def cosine_similarity(vec_a, vec_b):
    """Dot product of two sparse dict vectors (assumed L2-normalised)."""
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    return sum(w * vec_b.get(term, 0.0) for term, w in vec_a.items())


class TfidfVectorizer:
    """Fits a vocabulary and idf weights, then vectorises texts."""

    def __init__(self):
        self.vocabulary_ = []          # sorted list of terms
        self.idf_ = {}                 # term -> idf weight
        self.n_documents_ = 0

    def fit(self, documents):
        docs_tokens = [tokenize(d) for d in documents]
        self.n_documents_ = len(documents)
        df = Counter()
        for tokens in docs_tokens:
            for term in set(tokens):
                df[term] += 1
        n = self.n_documents_
        self.vocabulary_ = sorted(df)
        # smoothed idf: rare terms weigh more than terms in every document
        self.idf_ = {
            term: math.log((1 + n) / (1 + df[term])) + 1.0
            for term in self.vocabulary_
        }
        return self

    def transform(self, text):
        tokens = tokenize(text)
        if not tokens:
            return {}
        counts = Counter(t for t in tokens if t in self.idf_)
        if not counts:
            return {}
        total = float(len(tokens))
        vec = {term: (count / total) * self.idf_[term]
               for term, count in counts.items()}
        norm = math.sqrt(sum(w * w for w in vec.values()))
        if norm == 0.0:
            return {}
        return {term: w / norm for term, w in vec.items()}

    def fit_transform(self, documents):
        self.fit(documents)
        return [self.transform(d) for d in documents]

    def to_dict(self):
        return {"vocabulary": self.vocabulary_,
                "idf": self.idf_,
                "n_documents": self.n_documents_}

    @classmethod
    def from_dict(cls, data):
        vec = cls()
        vec.vocabulary_ = list(data["vocabulary"])
        vec.idf_ = dict(data["idf"])
        vec.n_documents_ = data["n_documents"]
        return vec
