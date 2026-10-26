"""Second-stage reranking and snippet extraction (Day 5).

The retriever in :mod:`documind.index` ranks whole chunks by TF-IDF /
BM25. That is fast but coarse: a chunk can score well because its terms
are spread across several unrelated sentences, while the one sentence
that actually answers the question sits in a lower-ranked chunk. So
before an answer is composed, the top-k chunks are reranked by the
quality of their *best sentence*, and the answer quotes a focused
snippet instead of an arbitrary first sentence.

Sentence score
--------------
For a question and a sentence, let ``matched`` be the question terms
present in the sentence:

* ``weighted coverage`` — the share of the question's terms the sentence
  covers, weighted by each term's idf when one is available, so a rare,
  discriminating term (``europa``) counts for more than a common one
  (``planet``). Without an idf map every term weighs the same.
* ``density`` — ``matched / sentence terms``, which favours short,
  on-topic sentences over long ones that merely mention a term in
  passing.
* ``phrase bonus`` — +0.1 when any consecutive pair of question terms
  appears consecutively in the sentence (an exact phrase match).

``score = 0.7 * weighted coverage + 0.3 * density + phrase bonus``,
capped at 1.0. Ties break towards the earlier sentence, so reranking is
deterministic.

Chunk rerank score
------------------
``0.5 * (retrieval score / best retrieval score) + 0.5 * best sentence
score``: half the retriever's judgement, half the sentence evidence, so
neither a high-scoring-but-diffuse chunk nor a single good sentence in
an otherwise irrelevant chunk can dominate on its own.

Snippet extraction
------------------
The snippet is the best-scoring sentence, plus one neighbouring
sentence when that neighbour adds question terms the best sentence
missed (a definition followed by its example, say). Neighbours that add
nothing are left out, so snippets stay focused. Over-long snippets are
trimmed to the best sentence alone, and a single over-long sentence is
cut at a word boundary with an ellipsis — the quoted text is otherwise
verbatim. A chunk with no sentence overlapping the question yields no
snippet at all, so it contributes nothing to the answer.
"""
from .chunker import split_sentences
from .embeddings import tokenize

RETRIEVAL_WEIGHT = 0.5
_MAX_SNIPPET_CHARS = 400


def _phrase_bonus(question_tokens, sentence_tokens):
    """0.1 if any consecutive question-token pair occurs in the sentence."""
    if len(question_tokens) < 2:
        return 0.0
    sentence_bigrams = set(zip(sentence_tokens, sentence_tokens[1:]))
    for first, second in zip(question_tokens, question_tokens[1:]):
        if (first, second) in sentence_bigrams:
            return 0.1
    return 0.0


def sentence_score(question, sentence, idf=None):
    """Score how well ``sentence`` answers ``question`` (0.0 to 1.0).

    ``idf`` is an optional {term: weight} map (e.g. the index
    vectoriser's ``idf_``); without it every question term weighs the
    same. A sentence sharing no question term scores 0.0.
    """
    q_terms = set(tokenize(question))
    if not q_terms:
        return 0.0
    s_tokens = tokenize(sentence)
    if not s_tokens:
        return 0.0
    matched = q_terms & set(s_tokens)
    if not matched:
        return 0.0
    if idf:
        weight = lambda term: idf.get(term, 1.0)  # noqa: E731
        weighted = (sum(weight(t) for t in matched)
                    / sum(weight(t) for t in q_terms))
    else:
        weighted = len(matched) / len(q_terms)
    density = len(matched) / len(set(s_tokens))
    score = (0.7 * weighted + 0.3 * density
             + _phrase_bonus(tokenize(question), s_tokens))
    return min(score, 1.0)


def _scored_sentences(text, question, idf=None):
    """[(score, position, sentence)] for every sentence in ``text``."""
    return [(sentence_score(question, sentence, idf), i, sentence)
            for i, sentence in enumerate(split_sentences(text))]


def extract_snippet(text, question, idf=None, max_sentences=2,
                    max_chars=_MAX_SNIPPET_CHARS):
    """Extract the best snippet of ``text`` for ``question``.

    Returns the best-scoring sentence, extended with one neighbouring
    sentence when it adds uncovered question terms, or "" when no
    sentence overlaps the question at all.
    """
    scored = _scored_sentences(text, question, idf)
    if not scored:
        return ""
    # Highest score wins; earlier sentences win ties (stable, explicit).
    best_score, best_pos, best = max(
        scored, key=lambda item: (item[0], -item[1]))
    if best_score <= 0.0:
        return ""
    if max_sentences <= 1:
        chosen = [(best_pos, best)]
    else:
        covered = set(tokenize(best))
        q_terms = set(tokenize(question))
        neighbours = []
        for score, pos, sentence in scored:
            if abs(pos - best_pos) != 1 or score <= 0.0:
                continue
            new_terms = (q_terms & set(tokenize(sentence))) - covered
            if new_terms:
                neighbours.append((len(new_terms), score, pos, sentence))
        chosen = [(best_pos, best)]
        if neighbours:
            # Most new terms wins; earlier neighbour wins ties.
            neighbours.sort(key=lambda item: (-item[0], -item[1], item[2]))
            _, _, pos, sentence = neighbours[0]
            chosen.append((pos, sentence))
            chosen.sort()
    snippet = " ".join(sentence for _, sentence in chosen)
    if len(snippet) > max_chars:
        # Fall back to the best sentence alone, then trim at a word
        # boundary rather than cutting mid-word.
        snippet = best
        if len(snippet) > max_chars:
            cut = snippet[:max_chars].rsplit(" ", 1)[0]
            snippet = (cut + "…") if cut else snippet[:max_chars] + "…"
    return snippet


def rerank_hits(hits, question, idf=None,
                retrieval_weight=RETRIEVAL_WEIGHT):
    """Rerank retriever ``hits`` for ``question`` by sentence evidence.

    Each hit (a dict with ``score`` and ``text``) is copied and given
    ``rerank_score`` (the combined score, rounded to 4 places) and
    ``snippet`` (see :func:`extract_snippet`). Hits are returned best
    first; the original retrieval ``score`` is left untouched. An empty
    hit list reranks to an empty list.
    """
    if not hits:
        return []
    top_score = max(h["score"] for h in hits) or 1.0
    reranked = []
    for hit in hits:
        scored = _scored_sentences(hit["text"], question, idf)
        best_sentence_score = max((s for s, _, _ in scored), default=0.0)
        combined = (retrieval_weight * (hit["score"] / top_score)
                    + (1.0 - retrieval_weight) * best_sentence_score)
        reranked.append({**hit,
                         "rerank_score": round(combined, 4),
                         "snippet": extract_snippet(hit["text"], question,
                                                    idf)})
    reranked.sort(key=lambda h: (-h["rerank_score"], -h["score"]))
    return reranked
