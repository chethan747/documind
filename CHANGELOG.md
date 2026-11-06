# Changelog

All notable changes to DocuMind are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); version numbers
follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Docker packaging (Day 16): `Dockerfile` (Python 3.12 slim, no
  dependencies to install, non-root `documind` user, `/health`
  healthcheck, state in `/data`) and `docker-compose.yml` (builds the
  image, publishes port 8000, persists the index / SQLite store /
  documents in the named volume `documind-data` mounted at `/data`,
  restarts unless stopped); `.dockerignore` keeps runtime artefacts,
  caches and dev tooling out of the image. New `DOCUMIND_INDEX_FILE`,
  `DOCUMIND_STORE_FILE` and `DOCUMIND_DATA_DIR` environment overrides
  point the index, store and data directory at a mounted volume
  without a config file. Test suite now 236 tests, including
  `tests/test_docker.py` validating the Dockerfile, compose file,
  `.dockerignore` and the new path overrides. No Docker daemon was
  available on the build machine, so `docker build` / `docker compose
  up` were not run there; the files were validated by parsing and by
  running the server with the exact environment the compose file
  sets, against a scratch directory standing in for the volume.

### Fixed

- Access-log race (Day 15): the server wrote its `http_access` line
  after writing the response body, but the socket is unbuffered
  (`wbufsize = 0`), so a client could receive the response before
  the log line existed — `test_ingest_and_error_requests_are_logged`
  failed intermittently (about 1 run in 20) with the 404 line
  missing. `_send` and `_send_html` now log before writing the
  body; the log line is guaranteed to exist by the time the client
  can read the response. Verified 0 late arrivals in 200 runs
  (was 10) and a green full suite twice in a row.
- Config file + structured logging (Day 15): `config.json` for host /
  port, index/store/data paths, default ranking method, retrieval
  cutoffs, chunk size/overlap and logging, with `DOCUMIND_*`
  environment overrides, strict startup validation, and `cli.py
  config` to print the effective config. Logs are one JSON object per
  line (UTC timestamp, level, logger, message, structured fields);
  the server logs one `http_access` line per request — path only,
  never the query string — plus ingest / delete / reindex / startup
  events. Test suite now 210 tests.

## [0.2.0] — 2026-11-04 — API + web app (Week 2 complete)

The second releasable DocuMind: everything from v0.1, plus a REST API
and a browser UI over the same store and index the CLI uses, a
validation pass so every layer rejects bad input the same way, and an
integration suite that drives the whole stack together. Still pure
Python standard library — no dependencies.

### Added

- REST API server (`documind/server.py`, `cli.py serve`): standard-library
  `ThreadingHTTPServer` exposing `GET /health`, `GET /stats`,
  `GET /search`, `POST /ingest` and `POST /ask` over the same store and
  index the CLI uses, with the file pipeline's dedup rules on ingest and
  cited extractive answers on ask. One lock serialises store/index
  access across threads.
- Document-management endpoints: `GET /documents` (summaries, no full
  text), `GET /documents/<doc_id>` (with text), `DELETE
  /documents/<doc_id>` (404 when unknown, repeat delete 404) and `POST
  /reindex`, which rebuilds the index from the document store as
  system of record.
- Dedicated QA endpoint `POST /qa`: same request shape as `/ask`, but
  the response carries structured `citations` (document id and title,
  chunk index, scores, snippet, readable `"[doc_id, chunk N]"` label)
  plus `source_count`; `/ask` is kept as a compatible alias.
- Web UI (`documind/web.py`, served at `GET /`): dependency-free page
  with an upload form posting to `/ingest`, an ask form posting to
  `/qa` with answer + structured citations, a sources panel listing
  indexed documents with click-to-preview, and question history in
  localStorage (newest first, capped at 20, clickable, clearable). All
  panel rendering uses `textContent`, so stored markup cannot inject.
- Error handling + input validation pass (`documind/validation.py`):
  shared validators for k (integer 1..100, booleans rejected),
  query/question (non-empty, max 2000 chars), doc_id (letter/digit
  start, letters/digits/`.`/`_`/`-` only, max 128), title (max 500)
  and text (non-empty, max 2,000,000 chars), applied at the core so
  the API, CLI and direct library calls reject bad input identically.
  The server answers those with JSON 400, bodies over 10 MB with 413,
  and PUT/PATCH on known paths with JSON 405 (not HTML 501); malformed
  Content-Length headers are answered 400 instead of dropping the
  connection.
- Integration test suite (`tests/test_integration.py`): 7 tests driving
  the whole stack together over real HTTP — full ingest → ask → delete
  → reindex lifecycle, index rebuild from the store after a restart
  with the saved index deleted, library/API answer agreement,
  evaluation over the served index, failed requests leaving state
  intact, the web-UI journey, and 8 concurrent search requests.
- 177-test unittest suite (up from 107 at v0.1.0), all passing
  (`python3 -m unittest discover -s tests`).

### Notes

- Honest scope, unchanged from v0.1: a personal, learning-grade
  implementation built over 21 days — small, readable and fully
  tested, not a production system.
- Runtime artefacts (`index.json`, `documind.db`) are git-ignored and
  are never part of a release.

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

- **v0.2 (Week 2)** — done and released above: REST API server,
  document-management and QA endpoints, web UI, error-handling pass,
  integration tests.
- **v1.0 (Week 3)** — config + structured logging, Docker packaging,
  GitHub Actions CI, index caching and performance pass, final
  evaluation run, release notes and resume write-up.
