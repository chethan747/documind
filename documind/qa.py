"""Extractive question answering over the DocuMind index.

No language model is involved: the retriever finds the most relevant
chunks, then the sentences inside them that overlap most with the question
are lifted verbatim into the answer, each followed by a [source] citation.
That keeps every answer grounded in — and traceable to — the documents.
"""
from .chunker import split_sentences
from .embeddings import tokenize


def _sentence_score(question_terms, sentence):
    terms = set(tokenize(sentence))
    if not terms:
        return 0.0
    return len(question_terms & terms) / len(question_terms) if question_terms else 0.0


def answer_question(index, question, k=3, method="tfidf"):
    """Answer ``question`` from ``index`` with cited sources.

    ``method`` is passed through to ``index.search``: ``"tfidf"``
    (default), ``"bm25"`` or ``"combined"``.

    Returns {"answer": str, "sources": [{"doc_id", "chunk_index", "score"}]}.
    When nothing relevant is found, the answer says so plainly.
    """
    hits = index.search(question, k=k, method=method)
    if not hits:
        return {"answer": "I could not find anything relevant in the "
                          "indexed documents.",
                "sources": []}

    q_terms = set(tokenize(question))
    picked = []       # (sentence, doc_id)
    seen = set()
    for hit in hits:
        sentences = split_sentences(hit["text"])
        sentences.sort(key=lambda s: _sentence_score(q_terms, s),
                       reverse=True)
        for sentence in sentences[:1]:
            if sentence not in seen:
                seen.add(sentence)
                picked.append((sentence, hit["doc_id"]))

    answer = " ".join("%s [%s]" % (sentence, doc_id)
                      for sentence, doc_id in picked)
    sources = []
    for h in hits:
        source = {"doc_id": h["doc_id"],
                  "chunk_index": h["chunk_index"],
                  "score": h["score"]}
        title = h.get("metadata", {}).get("title")
        if title:
            source["title"] = title
        sources.append(source)
    return {"answer": answer, "sources": sources}
