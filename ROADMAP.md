# DocuMind — 21-Day Build Roadmap

DocuMind is built in public over 21 days: one focused task per day, each one
committed and verified. The goal is a complete, resume-ready AI Document
Intelligence Platform by Day 21.

## Week 1 — Core engine

- [x] **Day 1** — Project scaffold, overlapping chunker, TF-IDF vectoriser
      from scratch, cosine-similarity index, extractive QA with citations,
      CLI (ingest / ask / demo), unit tests
- [x] **Day 2** — Document ingestion pipeline + SQLite document store
- [x] **Day 3** — BM25 ranking alongside TF-IDF
- [x] **Day 4** — Persistent index save/load + document metadata
- [x] **Day 5** — Answer reranking + better snippet extraction
      (rerank.py: idf-weighted sentence scoring with phrase bonus and
      density, chunk reranking combining retrieval + sentence evidence,
      focused snippets with a context neighbour only when it adds
      uncovered question terms; qa.py reports snippet + rerank_score;
      `cli.py ask --no-rerank` keeps retriever order; 19 new tests)
- [x] **Day 6** — Evaluation harness with sample Q&A set and hit-rate metrics
      (evaluate.py: load_cases, run_evaluation, evaluate_all_methods,
      format_report; eval/eval_qa.json: 12 questions over data/; hit
      rate@5, MRR@5, QA citation rate, QA term rate; `cli.py eval`
      [--method | --all-methods] [--k] [--output]; 14 new tests.
      Result on the shipped set, all methods: hit rate 1.000, MRR 1.000,
      QA citation 1.000, QA terms 0.917 — the one term miss is q9
      (overfitting), whose answer quotes the cross-validation sentence
      rather than the memorisation definition.)
- [x] **Day 7** — Cleanup, docs, v0.1 release notes
      (CHANGELOG.md with v0.1.0 notes and the verified evaluation
      table; `cli.py --version` reporting documind 0.1.0; README
      roadmap/layout updated to the released state; cleanup pass
      confirmed no TODO/FIXME markers, no unused imports, all modules
      compile; 5 new release tests, suite now 107 tests)

## Week 2 — API + web app

- [x] **Day 8** — REST API server for ingest / search / ask
      (server.py: stdlib ThreadingHTTPServer, `cli.py serve` /
      `python3 -m documind.server`; GET /health, /stats,
      /search?query&k&method, POST /ingest with the file pipeline's
      dedup rules, POST /ask with citations; 400/404/405/409 handling;
      single lock serialising store+index access across threads,
      DocumentStore gains a check_same_thread option for it; 15 new
      tests driving a real server over HTTP, suite now 122 tests.
      Live demo verified: health, search, cited ask, ingest 201, stats.)
- [x] **Day 9** — Document management endpoints (list / delete / reindex)
      (server.py: GET /documents summaries without full text, GET
      /documents/<doc_id> with text, DELETE /documents/<doc_id> removing
      from store and index then persisting (404 when unknown, repeat
      delete 404), POST /reindex rebuilding the index from the store as
      system of record and re-persisting it; wrong-method 405s;
      10 new tests in tests/test_management.py, suite now 132 tests.
      Live demo verified: list 3, get 200, delete ml-basics.md,
      list 2, reindex 2 documents / 9 chunks.)
- [x] **Day 10** — QA endpoint with source citations
      (server.py: POST /qa as the dedicated QA endpoint — same request
      shape as /ask, but the response carries structured `citations`
      (doc_id, title, chunk_index, score, rerank_score when reranking,
      snippet, readable "[doc_id, chunk N]" label) plus `source_count`;
      /ask kept as a compatible alias; GET /qa is 405, empty index is
      a 200 with empty citations; 10 new tests in tests/test_qa.py,
      suite now 142 tests. Live demo verified: /qa 200 with 3
      structured citations for "What is AEM built on?")
- [x] **Day 11** — Web UI: upload + ask page
      (documind/web.py: dependency-free HTML page served at GET / and
      /index.html, upload form posting to /ingest, ask form posting to
      /qa with answer + structured citations, index status from
      /health, user text rendered via textContent so it cannot inject
      markup; POST / is 405; 6 new tests in tests/test_webui.py,
      suite now 148 tests. Live demo verified: page served, upload
      201, cited /qa answer, health 4 documents / 15 chunks.)
- [x] **Day 12** — UI polish: sources panel, history
      (web.py: sources panel listing indexed documents from
      GET /documents with title, id and chunk count, a refresh
      button, and click-to-preview fetching GET /documents/<doc_id>;
      question history in localStorage under "documind-history",
      newest first capped at 20, clickable to restore answer +
      citations, with a clear button; sources refresh after every
      successful upload; all panel rendering via textContent so
      stored markup cannot inject; 6 new tests in
      tests/test_ui_polish.py, suite now 154 tests. Live demo
      verified: page served with both panels, /documents list 3,
      one document preview, list grows to 4 after an upload.)
- [x] **Day 13** — Error handling + input validation pass
      (validation.py: shared validators for k (integer 1..100, booleans
      rejected), query/question (non-empty, max 2000 chars), doc_id
      (letter/digit start, letters/digits/`.`/`_`/`-` only, max 128 —
      so an id with `/` can no longer be stored yet never fetched),
      title (max 500) and text (non-empty, max 2,000,000 chars);
      Index.search and answer_question raise ValueError on bad input
      instead of silently slicing or answering a blank question;
      server returns JSON 400 for those, 413 for bodies over 10 MB,
      and JSON 405 (not HTML 501) for PUT/PATCH on known paths, and
      malformed Content-Length headers are answered 400 instead of
      dropping the connection; 16 new tests in
      tests/test_validation.py, suite now 170 tests. Live demo
      verified: health 200, ingest slash-id 400, search k=0 400,
      PUT /ask 405, /qa 200.)
- [x] **Day 14** — Integration tests + v0.2 release notes
      (tests/test_integration.py: 7 tests driving the whole stack
      together over real HTTP — full ingest → ask → delete → reindex
      lifecycle, index rebuild from the store after a restart with
      the saved index deleted, library/API answer agreement,
      evaluation over the served index, failed requests leaving
      state intact, the web-UI journey, 8 concurrent searches;
      v0.2.0 released: `documind.__version__` bumped, CHANGELOG.md
      [0.2.0] notes for Week 2, README feature list updated;
      suite now 177 tests. Live demo verified: full lifecycle over
      HTTP, restart rebuild, served-index evaluation hit rate 1.000.)

## Week 3 — Polish + ship

- [x] **Day 15** — Config file + structured logging
      (config.py: `config.json` — server host/port, index/store/data
      paths, default method, search/ask k, chunk size/overlap, log
      level/file/JSON — with `DOCUMIND_*` env overrides, strict
      startup validation of unknown keys and out-of-range values,
      and path resolution against the project root;
      logging_config.py: one JSON object per log line (UTC timestamp,
      level, logger, message, structured extra fields), file
      handler, plain-text mode; server logs one `http_access` line
      per request (path only, never the query string) plus ingest /
      delete / reindex / startup events; `cli.py --config FILE`,
      `cli.py config` prints the effective config, `serve --host` /
      `--port` override the config; 33 new tests in
      tests/test_config.py and tests/test_logging.py, suite now 210
      tests. Live demo verified: `config` prints defaults, server
      on configured port 8123 serves health/search/qa, JSON log
      file carries startup + access lines with status and
      duration_ms, no query string logged.)
- [x] **Day 16** — Dockerfile + docker-compose
      (Dockerfile: python:3.12-slim, no dependencies to install,
      non-root `documind` user, `/data` owned by it, `EXPOSE 8000`,
      `/health` healthcheck, `CMD ["python3", "cli.py", "serve"]`;
      docker-compose.yml: builds the image, publishes 8000, persists
      index/store/docs in the named volume `documind-data` mounted at
      `/data`, restarts unless stopped; `.dockerignore` keeps runtime
      artefacts, caches and dev tooling out of the image; config.py
      gains `DOCUMIND_INDEX_FILE` / `DOCUMIND_STORE_FILE` /
      `DOCUMIND_DATA_DIR` env overrides so the container points its
      state at the volume without a config file; README gains a
      "Run with Docker" section. Also fixed a Day 15 access-log race
      found while getting the suite green: `_send`/`_send_html`
      logged after writing the body to the unbuffered socket, so the
      404 access line could arrive after the client had already read
      the response — both now log before writing the body, verified
      0 late arrivals in 200 runs (was 10). 26 new tests in
      tests/test_docker.py, suite now 236 tests. Live demo verified:
      server run with the exact compose environment against a scratch
      directory standing in for the volume — health 200, ingest 201,
      cited /qa answer, index + store created in the volume dir.
      No Docker daemon on the build machine, so `docker build` /
      `docker compose up` were not run there.)
- [x] **Day 17** — GitHub Actions CI running the tests
      (.github/workflows/ci.yml: runs on push and pull requests to
      main, with a concurrency group cancelling stale runs; one job
      per Python version — 3.10, 3.11, 3.12, fail-fast off — on
      ubuntu-latest, checking out the code, setting up Python,
      byte-compiling cli.py / documind / tests, running
      `python3 -m unittest discover -s tests`, then smoke-checking
      `cli.py eval --all-methods`; nothing to install, the project
      is pure stdlib. 20 new tests in tests/test_ci.py cover the
      parsed workflow — triggers, matrix, checkout/setup-python
      versions, step order — plus text-level checks and a local run
      of the exact commands CI runs, suite now 256 tests. Live
      demo verified: the CI commands run cleanly locally — compile
      0, full suite 256 OK, eval all methods hit rate 1.000, MRR
      1.000, QA citation 1.000, QA terms 0.917.)
- [ ] **Day 18** — Index caching + performance pass
- [ ] **Day 19** — README polish with screenshots / demo output
- [ ] **Day 20** — Final evaluation run, results documented
- [ ] **Day 21** — v1.0 release notes + RESUME.md with resume bullet points
      and project description
