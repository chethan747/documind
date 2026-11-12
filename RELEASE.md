# DocuMind v1.0.0 — 2026-11-12

The first full release of DocuMind, the AI Document Intelligence Platform:
a retrieval-augmented generation (RAG) system built from scratch in pure
Python standard library — no dependencies, no external APIs.

## What it is

DocuMind indexes a folder of text and Markdown documents, retrieves the
passages most relevant to a question, and composes an extractive answer in
which every statement is followed by the source it came from.

- Overlapping, sentence-aware chunker; TF-IDF vectoriser built from scratch
  (tokenisation, smoothed idf, sparse L2-normalised vectors, cosine index)
- BM25 ranking alongside TF-IDF (k1=1.5, b=0.75), searchable alone or
  combined (`tfidf` / `bm25` / `combined`)
- Extractive QA with `[source]` citations, plus answer reranking and
  focused snippet extraction
- Document ingestion pipeline: recursive discovery, `.txt` / `.md` / `.html`
  extraction, SHA-256 dedup, SQLite document store
- REST API (`GET /health`, `/stats`, `/search`, `/documents`;
  `POST /ingest`, `/ask`, `/qa` with structured citations, `/reindex`;
  `DELETE /documents/<doc_id>`) and a dependency-free web UI at `GET /`
- Evaluation harness: 12-question set grounded in the sample documents,
  scored by hit rate@k, MRR@k, QA citation rate and QA term rate
- Config file + structured JSON logging, Docker packaging, GitHub Actions
  CI (Python 3.10 / 3.11 / 3.12), index caching + performance pass

## Final evaluation (Day 20, 2026-11-10, k=5, 12 questions)

| method   | hit rate | MRR   | QA citation | QA terms |
|----------|----------|-------|-------------|----------|
| tfidf    | 1.000    | 1.000 | 1.000       | 0.917    |
| bm25     | 1.000    | 1.000 | 1.000       | 0.917    |
| combined | 1.000    | 1.000 | 1.000       | 0.917    |

Every question's expected document ranks first for every method. The single
QA-term miss (question 9) is a known artefact of the small sample set: the
answer quotes the cross-validation sentence rather than the memorisation
definition. Full report: `EVALUATION.md`, machine-readable:
`eval/final_report.json`. Reproduce with
`python3 cli.py eval --all-methods`.

## Performance (Day 18, sample run — varies by machine)

300 documents / 1400 chunks: cold search 1.210 / 2.327 / 1.723 ms vs warm
0.004 ms per search (279–541x) for tfidf / bm25 / combined, via the
inverted index and bounded LRU query cache. Reproduce with
`python3 cli.py bench`.

## Tests

317-test unittest suite, all passing
(`python3 -m unittest discover -s tests`), plus CI on every push / PR.

## Honest scope

A personal, learning-grade implementation built over 21 days — small,
readable and fully tested, not a production system. Sample documents in
`data/` are short explainers written for the demo.

## Try it

```bash
python3 cli.py demo
python3 cli.py eval --all-methods
python3 cli.py serve   # web UI + API on port 8000
```

Repo: https://github.com/chethan747/documind
