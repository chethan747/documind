# DocuMind — Final Evaluation Report

**Run:** 2026-11-10 (Day 20 of 21) · **Version:** 0.2.0 ·
**Corpus:** the 3 sample documents in `data/` (14 chunks) ·
**Question set:** the 12 questions in `eval/eval_qa.json` ·
**Harness:** `documind/evaluate.py`, run as `python3 cli.py eval --all-methods`

This is the final evaluation of the finished pipeline: the same harness
and question set used since Day 6, re-run on the Day 20 code (with
reranking, the inverted index and the query cache from Days 5, 15 and
18 in place). The machine-readable form of the headline run is
`eval/final_report.json` (`cli.py eval --all-methods --output
eval/final_report.json`); every number in this report can be reproduced
with the commands under [Reproducibility](#reproducibility).

## Methodology

Each question names the document that answers it (`expected_doc_id`)
and a list of words or phrases a good answer should mention
(`expected_terms` — any one of them counts as a term hit). Four
metrics are reported per ranking method (`tfidf`, `bm25`, `combined`):

- **hit rate@k** — share of questions whose expected document appears
  among the top-k retrieved chunks.
- **MRR@k** — mean reciprocal rank: 1/rank for a hit (rank 1 scores
  1.0), 0 for a miss, averaged over the questions.
- **QA citation rate** — share of answers whose cited sources include
  the expected document.
- **QA term rate** — share of questions whose answer mentions at least
  one of the expected terms.

Defaults for the headline run: `k=5`, answer reranking on (the
`cli.py ask` default), chunk size 500 with 80 overlap. The
[cutoff sensitivity](#cutoff-sensitivity) section repeats the run at
k=1, 3 and 10.

## Headline results (k=5)

| method   | hit rate@5 | MRR@5 | QA citation | QA terms |
|----------|-----------:|------:|------------:|---------:|
| tfidf    | 1.000      | 1.000 | 1.000       | 0.917    |
| bm25     | 1.000      | 1.000 | 1.000       | 0.917    |
| combined | 1.000      | 1.000 | 1.000       | 0.917    |

Identical to every run since Day 6: no ranking, reranking or
performance change in Weeks 2–3 moved these numbers, which is the
point of the harness — it is a regression check, and nothing regressed.

## Per-question breakdown (k=5)

Every question's expected document ranks first for every method, and
every answer cites it. The only miss anywhere is question 9's QA term
rate (see [Findings](#findings)).

| q   | question | expected doc | rank tfidf | rank bm25 | rank combined | QA cited | QA terms |
|-----|----------|--------------|-----------:|----------:|--------------:|:--------:|:--------:|
| q1  | What is AEM built on? | aem-basics.md | 1 | 1 | 1 | yes | yes |
| q2  | What does the dispatcher do in AEM? | aem-basics.md | 1 | 1 | 1 | yes | yes |
| q3  | What is HTL used for in AEM? | aem-basics.md | 1 | 1 | 1 | yes | yes |
| q4  | What is the difference between AEM author and publish instances? | aem-basics.md | 1 | 1 | 1 | yes | yes |
| q5  | What does Spring Boot auto-configuration do? | java-spring.md | 1 | 1 | 1 | yes | yes |
| q6  | How does Spring MVC expose REST endpoints? | java-spring.md | 1 | 1 | 1 | yes | yes |
| q7  | What is dependency injection in Spring? | java-spring.md | 1 | 1 | 1 | yes | yes |
| q8  | What does Spring Boot Actuator provide? | java-spring.md | 1 | 1 | 1 | yes | yes |
| q9  | What is overfitting in machine learning? | ml-basics.md | 1 | 1 | 1 | yes | no |
| q10 | How do supervised and unsupervised learning differ? | ml-basics.md | 1 | 1 | 1 | yes | yes |
| q11 | What are embeddings used for in machine learning? | ml-basics.md | 1 | 1 | 1 | yes | yes |
| q12 | Why split data into training and test sets? | ml-basics.md | 1 | 1 | 1 | yes | yes |

## Cutoff sensitivity

The same 12 questions at other cutoffs (`--all-methods --k N`):

| k  | hit rate tfidf | hit rate bm25 | hit rate combined | QA terms tfidf | QA terms bm25 | QA terms combined |
|---:|----------------:|--------------:|------------------:|---------------:|--------------:|------------------:|
| 1  | 1.000 | 1.000 | 1.000 | 0.667 | 0.917 | 0.667 |
| 3  | 1.000 | 1.000 | 1.000 | 0.917 | 0.917 | 0.917 |
| 5  | 1.000 | 1.000 | 1.000 | 0.917 | 0.917 | 0.917 |
| 10 | 1.000 | 1.000 | 1.000 | 0.917 | 0.917 | 0.917 |

MRR and QA citation rate are 1.000 at every cutoff for every method,
so they are omitted from this table. Retrieval is cutoff-proof here:
the expected document is always the top hit, so hit rate and MRR do
not move at all.

## Findings

1. **Retrieval is perfect on this set.** All 12 expected documents
   rank first for all three methods, at every cutoff from k=1 to
   k=10. Nothing in the ranking stack (TF-IDF, BM25, combined, the
   Day 18 inverted index, the query cache) can be told apart by
   retrieval quality on these questions.
2. **Every answer cites its source.** The QA citation rate is 1.000
   for every method at every cutoff — the extractive answer always
   quotes the expected document.
3. **Question 9 is the only QA term miss.** "What is overfitting in
   machine learning?" expects "memorises", "training" or "new data";
   the answer quotes the cross-validation sentence ("Techniques such
   as cross-validation, regularisation and simply collecting more data
   help to reduce overfitting") instead of the memorisation
   definition. Retrieval is right (rank 1); the reranker just picks a
   neighbouring sentence. This has been true since Day 6 and is a
   property of the sample answer, not a regression.
4. **k=1 is fragile for TF-IDF and combined.** With a single chunk to
   answer from, the QA term rate falls to 0.667 for `tfidf` and
   `combined` (misses on q2, q3, q9, q11 — the top chunk is on-topic
   but does not carry the expected terms), while `bm25` holds 0.917
   (only q9). At k=3 and above all three methods agree at 0.917.
   The shipped defaults avoid the weak spot: `cli.py ask` uses k=3
   and the API's `ask_k` is 3.
5. **No method wins on this set.** At k≥3 the three methods are
   indistinguishable on all four metrics. BM25's edge only shows at
   k=1. Method choice here is a matter of corpus shape, not of these
   results — BM25's length normalisation is the principled default
   for uneven document lengths, TF-IDF is the simpler one to explain.
6. **Performance is unchanged.** A `cli.py bench` run alongside this
   evaluation (300 documents / 1400 chunks, k=5) still shows cold
   search at ~1.1–1.9 ms and warm (cache-hit) search at ~0.005 ms
   per search — a 250–410x speedup from the query cache, in line with
   the sample run quoted in the README. Quality and speed were
   measured on the same code in the same session.

## Limitations

Read these numbers for what they are — a regression check on a small,
hand-written set, not a claim about general accuracy:

- **12 questions over 3 short documents.** A set this small cannot
  separate good rankers from great ones; every method scores 1.000
  on retrieval, so the table says "nothing regressed", not "this
  retrieval is perfect in general".
- **Questions were written against the documents.** They use the
  documents' own vocabulary, which flatters term-matching rankers.
  Paraphrased or adversarial questions are untested.
- **The term check is lenient.** Any one expected term counts as a
  hit, and the terms were chosen knowing what the answers say.
- **No held-out set.** The same questions have been run since Day 6;
  they guard against regressions but cannot detect overfitting to
  them.
- **The benchmark corpus is near-duplicates.** It measures search
  speed, not retrieval quality, and says nothing about behaviour on
  a large, varied corpus.

## Reproducibility

From the project root, on a fresh checkout (no saved index needed —
`eval` builds one in memory from `data/` when none exists):

```bash
# headline run, all three methods at k=5, with the JSON report
python3 cli.py eval --all-methods --output eval/final_report.json

# cutoff sensitivity
python3 cli.py eval --all-methods --k 1
python3 cli.py eval --all-methods --k 3
python3 cli.py eval --all-methods --k 10

# single method, per-question detail
python3 cli.py eval --method bm25

# full test suite (includes tests/test_final_eval.py, which re-runs
# this evaluation and checks every number in this report against it)
python3 -m unittest discover -s tests
```
