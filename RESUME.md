# DocuMind — resume write-up

## Project description (for resume / LinkedIn Projects)

**DocuMind — AI Document Intelligence Platform** (Oct 2026 – Nov 2026)

From-scratch retrieval-augmented generation (RAG) system in pure Python
standard library — no frameworks, no external APIs. Indexes a folder of
documents, retrieves the most relevant passages for a question, and returns
extractive answers with citations. Shipped in 21 daily builds: core
retrieval engine, REST API + web UI, Docker packaging, GitHub Actions CI,
and a final evaluation run. 317-test suite, all passing.
Repo: https://github.com/chethan747/documind

## Resume bullet points

- Built DocuMind, a from-scratch RAG document-intelligence platform in
  pure Python (stdlib only, zero dependencies): TF-IDF vectoriser and
  BM25 ranker implemented by hand, with cosine-similarity retrieval and
  extractive QA that cites its sources.
- Shipped a REST API and dependency-free web UI over the same store and
  index (`/search`, `/qa` with structured citations, `/ingest`,
  `/reindex`, document management), with shared input validation so the
  CLI, API and library reject bad input identically.
- Built a 12-question evaluation harness (hit rate, MRR, QA citation and
  QA term rates); final run scores 1.000 hit rate, 1.000 MRR, 1.000 QA
  citation and 0.917 QA terms across all three ranking methods, with
  every expected document ranking first.
- Added index caching and a performance pass (inverted indexes, bounded
  LRU query cache): warm searches ~0.004 ms vs ~1–2 ms cold on a
  300-document / 1400-chunk sample run.
- Packaged with Docker and GitHub Actions CI (Python 3.10–3.12); 317-test
  unittest suite covering chunking, ranking, QA, API, integration and
  evaluation, all passing.

## Notes for Chethan

- Keep the honest-scope line: learning-grade implementation, sample
  corpus only — do not present the 1.000 scores as production results.
- The LinkedIn Projects entry already says "Oct 2026 – Present"; update
  the end date to Nov 2026 once v1.0 is pushed, or leave as-is.
