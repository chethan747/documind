"""Extractive question answering over the DocuMind index.

No language model is involved: the retriever finds the most relevant
chunks, the Day 5 reranker re-orders them by the quality of their best
sentence, and the sentences (plus a context neighbour when it adds
uncovered question terms) are lifted verbatim into the answer, each
followed by a [source] citation. That keeps every answer grounded in —
and traceable to — the documents.

``rerank`` (default True) toggles only the re-ordering: snippet
extraction always applies, so even with ``rerank=False`` answers quote
the best sentence per chunk rather than an arbitrary one.
"""
from .rerank import extract_snippet, rerank_hits


def answer_question(index, question, k=3, method="tfidf", rerank=True):
    """Answer ``question`` from ``index`` with cited sources.

    ``method`` is passed through to ``index.search``: ``"tfidf"``
    (default), ``"bm25"`` or ``"combined"``.

    Returns {"answer": str, "sources": [{"doc_id", "chunk_index",
    "score", "snippet", ...}]}. ``sources`` also carry ``rerank_score``
    when reranking is on, and a document ``title`` when the index knows
    one. When nothing relevant is found, the answer says so plainly.
    """
    hits = index.search(question, k=k, method=method)
    if not hits:
        return {"answer": "I could not find anything relevant in the "
                          "indexed documents.",
                "sources": []}

    idf = getattr(getattr(index, "vectorizer", None), "idf_", None) or None
    if rerank:
        ranked = rerank_hits(hits, question, idf=idf)
    else:
        ranked = [{**hit,
                   "snippet": extract_snippet(hit["text"], question, idf)}
                  for hit in hits]

    picked = []       # (snippet, doc_id)
    seen = set()
    for hit in ranked:
        snippet = hit["snippet"]
        if snippet and snippet not in seen:
            seen.add(snippet)
            picked.append((snippet, hit["doc_id"]))

    if not picked:
        return {"answer": "I could not find anything relevant in the "
                          "indexed documents.",
                "sources": []}

    answer = " ".join("%s [%s]" % (snippet, doc_id)
                      for snippet, doc_id in picked)
    sources = []
    for h in ranked:
        source = {"doc_id": h["doc_id"],
                  "chunk_index": h["chunk_index"],
                  "score": h["score"],
                  "snippet": h["snippet"]}
        if rerank:
            source["rerank_score"] = h["rerank_score"]
        title = h.get("metadata", {}).get("title")
        if title:
            source["title"] = title
        sources.append(source)
    return {"answer": answer, "sources": sources}
