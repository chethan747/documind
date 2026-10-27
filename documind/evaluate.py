"""Evaluation harness for DocuMind (Day 6).

Scores the retrieval + QA pipeline against a small, hand-written set of
questions whose correct source document is known. Two things are measured
per question, both at a fixed cutoff ``k``:

* **Retrieval** — is the expected document among the top-k retrieved
  chunks? Reported as hit-rate@k (fraction of questions that hit) and
  mean reciprocal rank (MRR@k, the mean of 1/rank for hits, 0 for misses).
* **QA** — does the extractive answer cite the expected document, and
  (when the case lists ``expected_terms``) does the answer mention at
  least one of them? Reported as citation rate and term rate.

A case is a dict with ``question`` and ``expected_doc_id``, plus
optional ``id`` and ``expected_terms`` (a list of words/phrases that a
good answer should mention). Cases live in ``eval/eval_qa.json`` by
default; ``load_cases`` reads any JSON file holding either a bare list
of cases or an object with a ``cases`` key.

The harness is deliberately small and honest: it only claims what the
numbers say about *these* questions over *these* documents. It is a
regression check for ranking changes (Day 3's BM25 vs TF-IDF vs
combined, Day 5's reranking), not a claim about general accuracy.
"""
import json

from .index import SEARCH_METHODS
from .qa import answer_question


def load_cases(path):
    """Load and validate evaluation cases from a JSON file.

    Returns a list of normalised case dicts (``id``, ``question``,
    ``expected_doc_id``, ``expected_terms``). Raises ValueError for a
    file that is not a case list / ``{"cases": [...]}`` object, or for
    a case missing a non-empty ``question`` or ``expected_doc_id``.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid evaluation file %r: not valid JSON (%s)"
                         % (path, exc)) from exc
    if isinstance(data, dict):
        data = data.get("cases")
    if not isinstance(data, list) or not data:
        raise ValueError(
            "invalid evaluation file %r: expected a non-empty list of "
            "cases, or an object with a non-empty 'cases' list" % path)
    cases = []
    for i, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise ValueError("invalid evaluation case #%d: expected an "
                             "object" % (i + 1))
        question = raw.get("question")
        expected = raw.get("expected_doc_id")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("invalid evaluation case #%d: 'question' "
                             "must be a non-empty string" % (i + 1))
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError("invalid evaluation case #%d: "
                             "'expected_doc_id' must be a non-empty "
                             "string" % (i + 1))
        terms = raw.get("expected_terms") or []
        if not isinstance(terms, list) or not all(
                isinstance(t, str) and t.strip() for t in terms):
            raise ValueError("invalid evaluation case #%d: "
                             "'expected_terms' must be a list of "
                             "non-empty strings" % (i + 1))
        cases.append({"id": raw.get("id") or "q%d" % (i + 1),
                      "question": question,
                      "expected_doc_id": expected,
                      "expected_terms": list(terms)})
    return cases


def _validate_k(k):
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer, got %r" % (k,))
    return k


def _rank_of(retrieved_doc_ids, expected_doc_id):
    """1-based rank of ``expected_doc_id`` in ``retrieved_doc_ids``."""
    try:
        return retrieved_doc_ids.index(expected_doc_id) + 1
    except ValueError:
        return None


def run_evaluation(index, cases, k=5, method="tfidf"):
    """Score ``index`` on ``cases`` at cutoff ``k`` with ``method``.

    ``method`` is one of ``"tfidf"``, ``"bm25"`` or ``"combined"``; an
    unknown method raises ValueError (via ``Index.search``). An empty
    case list raises ValueError, so a vacuous 0/0 report is impossible.

    Returns a report dict: ``method``, ``k``, ``n`` (questions),
    ``hit_rate``, ``mrr``, ``qa_citation_rate``, ``qa_term_rate``
    (None when no case lists expected terms), and per-question
    ``results`` with rank, hit, reciprocal rank, QA citation flag,
    term flag and the answer itself.
    """
    _validate_k(k)
    if not cases:
        raise ValueError("no evaluation cases: nothing to score")
    results = []
    term_hits = 0
    term_total = 0
    for case in cases:
        hits = index.search(case["question"], k=k, method=method)
        retrieved = [h["doc_id"] for h in hits]
        rank = _rank_of(retrieved, case["expected_doc_id"])
        answer = answer_question(index, case["question"], k=k,
                                 method=method)
        cited = [s["doc_id"] for s in answer["sources"]]
        qa_cited = case["expected_doc_id"] in cited
        terms = case["expected_terms"]
        if terms:
            term_total += 1
            lowered = answer["answer"].lower()
            term_hit = any(t.lower() in lowered for t in terms)
            term_hits += 1 if term_hit else 0
        else:
            term_hit = None
        results.append({"id": case["id"],
                        "question": case["question"],
                        "expected_doc_id": case["expected_doc_id"],
                        "retrieved_doc_ids": retrieved,
                        "rank": rank,
                        "hit": rank is not None,
                        "reciprocal_rank": 1.0 / rank if rank else 0.0,
                        "qa_cited_expected": qa_cited,
                        "qa_term_hit": term_hit,
                        "answer": answer["answer"]})
    n = len(results)
    return {"method": method,
            "k": k,
            "n": n,
            "hit_rate": sum(r["hit"] for r in results) / n,
            "mrr": sum(r["reciprocal_rank"] for r in results) / n,
            "qa_citation_rate": sum(r["qa_cited_expected"]
                                    for r in results) / n,
            "qa_term_rate": (term_hits / term_total
                             if term_total else None),
            "results": results}


def evaluate_all_methods(index, cases, k=5):
    """Run ``run_evaluation`` for every ranking method, in order."""
    _validate_k(k)
    return [run_evaluation(index, cases, k=k, method=m)
            for m in SEARCH_METHODS]


def format_report(reports):
    """Render one report (dict) or a list of reports as text.

    A list is rendered as a comparison table plus per-method summaries;
    a single report also lists every question with its rank and QA
    citation outcome, so a regression is visible question by question.
    """
    if isinstance(reports, dict):
        reports = [reports]
    lines = []
    if len(reports) > 1:
        lines.append("method      hit@k   MRR     QA cited  QA terms")
        for rep in reports:
            terms = ("%.3f" % rep["qa_term_rate"]
                     if rep["qa_term_rate"] is not None else "  n/a")
            lines.append("%-11s %.3f   %.3f   %.3f     %s"
                         % (rep["method"], rep["hit_rate"], rep["mrr"],
                            rep["qa_citation_rate"], terms))
        lines.append("")
    for rep in reports:
        lines.append("Evaluation: method=%s, k=%d, %d question(s)"
                     % (rep["method"], rep["k"], rep["n"]))
        lines.append("  hit rate@%d : %.3f" % (rep["k"], rep["hit_rate"]))
        lines.append("  MRR@%d      : %.3f" % (rep["k"], rep["mrr"]))
        lines.append("  QA citation : %.3f" % rep["qa_citation_rate"])
        if rep["qa_term_rate"] is not None:
            lines.append("  QA terms    : %.3f" % rep["qa_term_rate"])
        if len(reports) == 1:
            for r in rep["results"]:
                rank = str(r["rank"]) if r["rank"] else "-"
                lines.append("  [%s] rank=%s hit=%s qa_cited=%s  %s"
                             % (r["id"], rank,
                                "yes" if r["hit"] else "no",
                                "yes" if r["qa_cited_expected"] else "no",
                                r["question"]))
    return "\n".join(lines) + "\n"
