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
- [ ] **Day 7** — Cleanup, docs, v0.1 release notes

## Week 2 — API + web app

- [ ] **Day 8** — REST API server for ingest / search / ask
- [ ] **Day 9** — Document management endpoints (list / delete / reindex)
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
