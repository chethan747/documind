# DocuMind — AI Document Intelligence Platform

**Ask questions over your own documents and get answers with citations —
a retrieval-augmented generation (RAG) system built from scratch in pure Python.**

DocuMind indexes a folder of text and Markdown documents, retrieves the
passages most relevant to a question, and composes an extractive answer in
which every statement is followed by the source it came from. No external
APIs, no frameworks, no dependencies: the chunking, TF-IDF retrieval and
ranking are implemented by hand so the whole pipeline is transparent and
easy to study.

## Features

**Done (v0.1 — core engine)**

- [x] Overlapping, sentence-aware text chunker
- [x] TF-IDF vectoriser built from scratch (tokenisation, idf, sparse vectors)
- [x] Cosine-similarity index with ranked top-k search
- [x] Extractive question answering with `[source]` citations
- [x] CLI: `ingest`, `ask` and `demo` commands; index persisted to JSON
- [x] Unit test suite covering chunking, retrieval ranking and QA
- [x] Document ingestion pipeline: recursive discovery, text extraction
      (.txt/.md/.html), SHA-256 dedup, unchanged/duplicate/empty handling
- [x] SQLite document store: document records, metadata, chunk counts,
      upsert on re-ingest, content-hash lookup
- [x] CLI: `list` and `stats` commands over the document store
- [x] BM25 ranking alongside TF-IDF (term-frequency saturation, length
      normalisation, `tfidf` / `bm25` / `combined` search methods;
      `cli.py ask --method bm25`); TF-IDF remains the default
- [x] Persistent index save/load with document metadata (Day 4): the
      saved index is versioned and validated on load, carries each
      document's title/source path, and survives a round-trip with
      identical rankings; pre-Day-4 saves still load; search and QA
      results report document titles, not just ids
- [x] Answer reranking + better snippet extraction (Day 5): top-k chunks
      are reranked by a second-stage score (half retrieval score, half
      best-sentence evidence — idf-weighted question-term coverage,
      sentence density and an exact-phrase bonus); answers quote the
      best sentence per chunk, plus a neighbouring sentence only when
      it adds uncovered question terms, with over-long snippets trimmed
      at a word boundary; QA sources carry the snippet and rerank score,
      and `cli.py ask --no-rerank` keeps plain retriever order
- [x] Evaluation harness (Day 6): a 12-question sample set in
      `eval/eval_qa.json` (grounded in the sample documents, with the
      expected source document and answer terms per question) scored by
      `documind/evaluate.py` — hit rate@k, MRR@k, QA citation rate and
      QA term rate, per ranking method or side by side via
      `cli.py eval [--all-methods] [--k N] [--output report.json]`.
      On the shipped set all three methods score hit rate 1.000,
      MRR 1.000, QA citation 1.000 and QA terms 0.917

**Roadmap** — see [ROADMAP.md](ROADMAP.md) for the full 21-day plan
and [CHANGELOG.md](CHANGELOG.md) for release notes: Week 1 is complete
and v0.1 (core engine) is released; next are REST API, web UI, Docker
packaging, CI, and a v1.0 release.

## Architecture

```
documents (.txt/.md/.html)
        │
        ▼
  ingest.py ── discovery → extraction → registration (with IngestStats)
        │                        │
        ▼                        ▼
  store.py (SQLite)        chunker.py ── overlapping sentence-aware chunks
  doc records, metadata           │
  dedup by content hash           ▼
                           embeddings.py ── TF-IDF vectors (sparse, L2-norm)
                           bm25.py ── BM25 scorer (k1/b, length norm)
                                  │
                                  ▼
                           index.py ── chunk store + tfidf / bm25 / combined top-k
                                  │
                                  ▼
                           rerank.py ── sentence scoring, chunk reranking,
                                        focused snippet extraction
                                  │
                                  ▼
                           qa.py ── extractive answers with [source] citations
                                  │
                                  ▼
                           evaluate.py ── Q&A harness: hit rate, MRR,
                                        QA citation / term rates
                                  │
                                  ▼
                           cli.py ── ingest / ask / list / stats / demo / eval
```

## Quickstart

Requires Python 3.10+; nothing to install.

```bash
# run the test suite
python3 -m unittest discover -s tests

# end-to-end demo over the sample documents in ./data
python3 cli.py demo

# index your own documents (any folder of .txt/.md/.html files)
python3 cli.py ingest /path/to/docs

# see what is stored
python3 cli.py list
python3 cli.py stats

# ask questions against the saved index
python3 cli.py ask "What is AEM built on?"

# score the pipeline on the sample Q&A set (all ranking methods)
python3 cli.py eval --all-methods
```

## Tech notes

- **Pure Python standard library** — the goal is to show the mechanics of
  RAG, not to hide them behind a framework.
- **TF-IDF retrieval from scratch** — smoothed idf, term-frequency weighting,
  L2-normalised sparse vectors, so cosine similarity is a plain dot product.
- **BM25 ranking from scratch** — Robertson/Sparck-Jones idf, saturating
  term frequency (k1=1.5) and length normalisation (b=0.75); searchable
  alone or combined (mean of per-method normalised scores) with TF-IDF.
- **Extractive answers with citations** — answers quote the retrieved
  sentences verbatim and cite their source document, so every claim can be
  traced back. (A generative LLM layer is a natural extension; the retrieval
  core stays the same.)
- **Honest scope** — this is a learning-grade implementation built over
  21 days: small, readable and fully tested, rather than a production system.

## Layout

```
documind/     core package: chunker, embeddings, bm25, index, rerank,
              qa, ingest, store, evaluate
cli.py        command-line interface
data/         sample documents the demo runs on
eval/         sample Q&A set for the evaluation harness
tests/        unittest suite
ROADMAP.md    the 21-day build plan
CHANGELOG.md  release notes (v0.1 core engine released 2026-10-28)
```
