# Changelog

All notable changes to DocuMind are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); version numbers
follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- REST API server (`documind/server.py`, `cli.py serve`): standard-library
  `ThreadingHTTPServer` exposing `GET /health`, `GET /stats`,
  `GET /search`, `POST /ingest` and `POST /ask` over the same store and
  index the CLI uses, with the file pipeline's dedup rules on ingest and
  cited extractive answers on ask. Documented in the README.

## [0.1.0] — 2026-10-28 — Core engine (Week 1 complete)

The first releasable DocuMind: a complete, tested retrieval pipeline that
indexes a folder of documents and answers questions over them with
citations. Pure Python standard library — no dependencies, nothing to
install beyond Python 3.10+.

### Added

- Overlapping, sentence-aware text chunker (`documind/chunker.py`).
- TF-IDF vectoriser built from scratch — tokenisation, smoothed idf,
  sparse L2-normalised vectors, cosine-similarity index
  (`documind/embeddings.py`, `documind/index.py`).
- BM25 ranking alongside TF-IDF (k1=1.5, b=0.75, length normalisation),
  searchable alone or combined with TF-IDF (`documind/bm25.py`;
  `cli.py ask --method tfidf|bm25|combined`, TF-IDF default).
- Document ingestion pipeline: recursive discovery, `.txt`/`.md`/`.html`
  extraction, SHA-256 dedup, unchanged/duplicate/empty handling
  (`documind/ingest.py`).
- SQLite document store with metadata, chunk counts and upsert on
  re-ingest (`documind/store.py`).
- Persistent, versioned index save/load with document metadata and
  backward-compatible loading of pre-Day-4 saves (`documind/index.py`).
- Answer reranking and focused snippet extraction: top-k chunks are
  rescored by retrieval score plus best-sentence evidence
  (idf-weighted question-term coverage, sentence density, exact-phrase
  bonus); answers quote the best sentence per chunk, plus a neighbour
  only when it adds uncovered question terms
  (`documind/rerank.py`; `cli.py ask --no-rerank` keeps retriever order).
- Extractive question answering with `[source]` citations
  (`documind/qa.py`).
- Evaluation harness: a 12-question sample set grounded in the sample
  documents (`eval/eval_qa.json`), scored by `documind/evaluate.py` —
  hit rate@k, MRR@k, QA citation rate and QA term rate, per method or
  side by side (`cli.py eval [--all-methods] [--k N]
  [--output report.json]`).
- CLI: `ingest`, `ask`, `list`, `stats`, `demo`, `eval`, and now
  `--version` (reports the `documind.__version__` release number).
- 102-test unittest suite covering chunking, retrieval ranking, QA,
  reranking, ingestion, persistence and evaluation
  (`python3 -m unittest discover -s tests`).

### Evaluation results (shipped 12-question set, k=5)

| method   | hit rate | MRR   | QA citation | QA terms |
|----------|----------|-------|-------------|----------|
| tfidf    | 1.000    | 1.000 | 1.000       | 0.917    |
| bm25     | 1.000    | 1.000 | 1.000       | 0.917    |
| combined | 1.000    | 1.000 | 1.000       | 0.917    |

The single QA-term miss (1 of 12 questions) is a known overfitting
artefact of the small sample set: the answer quotes the
cross-validation sentence rather than the memorisation definition.
Reproduce with `python3 cli.py eval --all-methods`.

### Notes

- Honest scope: this is a personal, learning-grade implementation built
  over 21 days — small, readable and fully tested, not a production
  system. Sample documents in `data/` are short explainers written for
  the demo.
- Runtime artefacts (`index.json`, `documind.db`) are git-ignored and
  are never part of a release.

## Roadmap

- **v0.2 (Week 2)** — REST API server, document-management and QA
  endpoints, web UI, error-handling pass, integration tests.
- **v1.0 (Week 3)** — config + structured logging, Docker packaging,
  GitHub Actions CI, index caching and performance pass, final
  evaluation run, release notes and resume write-up.
