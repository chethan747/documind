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
- [ ] **Day 10** — QA endpoint with source citations
- [ ] **Day 11** — Web UI: upload + ask page
- [ ] **Day 12** — UI polish: sources panel, history
- [ ] **Day 13** — Error handling + input validation pass
- [ ] **Day 14** — Integration tests + v0.2 release notes

## Week 3 — Polish + ship

- [ ] **Day 15** — Config file + structured logging
- [ ] **Day 16** — Dockerfile + docker-compose
- [ ] **Day 17** — GitHub Actions CI running the tests
- [ ] **Day 18** — Index caching + performance pass
- [ ] **Day 19** — README polish with screenshots / demo output
- [ ] **Day 20** — Final evaluation run, results documented
- [ ] **Day 21** — v1.0 release notes + RESUME.md with resume bullet points
      and project description
