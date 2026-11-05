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

**Done (v0.2 — API + web app)**

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

- [x] REST API server (Days 8–10): `GET /health`, `/stats`,
      `/search`, `/documents`; `POST /ingest`, `/ask`, `/qa` (structured
      citations), `/reindex`; `DELETE /documents/<doc_id>` — all JSON,
      with the file pipeline's dedup rules and JSON 400/404/405/409/413
      errors; one lock serialises store/index access across threads
- [x] Web UI (Days 11–12): dependency-free page at `GET /` with upload
      + ask forms, a sources panel with click-to-preview, and question
      history in localStorage (newest first, capped at 20); all panel
      rendering via `textContent`, so stored markup cannot inject
- [x] Error handling + input validation (Day 13): shared validators
      for k, query/question, doc_id, title and text, applied at the
      core so the API, CLI and library reject bad input identically
- [x] Integration tests (Day 14): 7 tests driving the whole stack
      together over real HTTP — full document lifecycle, restart
      rebuild from the store, library/API agreement, evaluation over
      the served index, failed requests leaving state intact, the web
      UI journey, and concurrent searches

- [x] Config file + structured logging (Day 15): `config.json`
      (host/port, index/store/data paths, default ranking method,
      retrieval cutoffs, chunk size/overlap, log level/file/JSON
      format) with `DOCUMIND_*` environment overrides and strict
      startup validation; `cli.py config` prints the effective
      config; every log line is one JSON object with a UTC timestamp
      and structured fields, the server logs one `http_access` line
      per request (path only, never the query string) plus ingest /
      delete / reindex / startup events

**Roadmap** — see [ROADMAP.md](ROADMAP.md) for the full 21-day plan
and [CHANGELOG.md](CHANGELOG.md) for release notes: Weeks 1–2 are
complete and v0.2 (API + web app) is released; next are Docker
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

# start the REST API server (default: http://127.0.0.1:8000)
python3 cli.py serve
# then open http://127.0.0.1:8000 in a browser for the upload + ask page
```

## REST API

Standard library only — no framework, no dependencies. All endpoints
speak JSON; the server loads the saved `index.json` / `documind.db`
when present, otherwise it rebuilds the index from the document store.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` or `/index.html` | web UI: upload + ask page, with sources panel and question history (HTML, not JSON) |
| GET | `/health` | status, version, document/chunk counts |
| GET | `/stats` | document/chunk counts, index presence |
| POST | `/ingest` | `{"doc_id"?, "title"?, "text"}` — add/update one document (201 on add, 200 on update/unchanged, 409 on duplicate content) |
| GET | `/search?query=...&k=5&method=tfidf\|bm25\|combined` | top-k retrieval hits |
| POST | `/ask` | `{"question"[, "k"][, "method"][, "rerank"]}` — extractive answer with source citations |
| POST | `/qa` | same request as `/ask` — dedicated QA endpoint whose response carries structured `citations` (doc id/title, chunk index, scores, snippet, readable citation label) plus `source_count` |
| GET | `/documents` | list stored documents (summary only, no full text) |
| GET | `/documents/<doc_id>` | one stored document, with its text |
| DELETE | `/documents/<doc_id>` | remove a document from store and index (404 when unknown) |
| POST | `/reindex` | rebuild the index from the document store |

Bad input returns 400, unknown paths 404, wrong-method calls 405.

## Configuration & logging

Host, port, file locations, ranking defaults, chunk size and the log
level live in a config file instead of in code — copy
`config.example.json` to `config.json` and edit it:

```json
{
  "server":    {"host": "127.0.0.1", "port": 8000},
  "paths":     {"index_file": "index.json", "store_file": "documind.db",
                "data_dir": "data"},
  "retrieval": {"default_method": "tfidf", "search_k": 5, "ask_k": 3,
                "chunk_size": 500, "chunk_overlap": 80},
  "logging":   {"level": "INFO", "file": null, "json": true}
}
```

Precedence: environment variables (`DOCUMIND_HOST`, `DOCUMIND_PORT`,
`DOCUMIND_LOG_LEVEL`, `DOCUMIND_LOG_FILE`) beat the file, which beats
the built-in defaults. Pass an explicit file with `cli.py --config
FILE <command>` or `DOCUMIND_CONFIG=FILE`; `cli.py config` prints the
effective config. Unknown keys and out-of-range values are rejected
at startup with a plain-language error, so a typo fails loudly
instead of silently doing nothing.

Every log line is one JSON object on stderr (or the configured log
file), with a UTC timestamp, level, logger, message and structured
fields — the server logs one `http_access` line per request
(`method`, `path`, `status`, `duration_ms`) plus `ingest` / `delete` /
`reindex` / `startup` events. Query strings are never logged: search
queries and questions are user content. Set `"json": false` for
plain-text logs.

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
              qa, ingest, store, evaluate, server (REST API), web (UI page),
              config (config file + env overrides),
              logging_config (structured JSON logging)
cli.py        command-line interface
config.example.json  documented config file (copy to config.json to use)
data/         sample documents the demo runs on
eval/         sample Q&A set for the evaluation harness
tests/        unittest suite
ROADMAP.md    the 21-day build plan
CHANGELOG.md  release notes (v0.2 API + web app released 2026-11-04)
```
