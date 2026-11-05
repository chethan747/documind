#!/usr/bin/env python3
"""DocuMind command-line interface.

Usage:
    python3 cli.py ingest <folder>     Index every supported document in
                                       <folder> (recursive) through the
                                       ingestion pipeline and persist the
                                       index to index.json and the document
                                       records to documind.db.
    python3 cli.py ask "<question>" [--method tfidf|bm25|combined]
                                       [--no-rerank]
                                       Answer a question from the saved index.
    python3 cli.py list                List every document in the store.
    python3 cli.py stats               Show document/index statistics.
    python3 cli.py serve [--host] [--port]
                                       Start the REST API server
                                       (ingest / search / ask over HTTP).
    python3 cli.py demo                Ingest ./data and ask sample questions.
    python3 cli.py eval                Score the index on the sample Q&A
                                       set (hit rate, MRR, QA citations).
    python3 cli.py config              Print the effective configuration
                                       (defaults + config.json + env).

Configuration (Day 15): ``python3 cli.py [--config FILE] <command>``
loads FILE, else $DOCUMIND_CONFIG, else ./config.json when present,
else the built-in defaults — see config.example.json. Host, port,
index/store paths, default ranking method, retrieval cutoffs, chunk
size and the log level all come from there; the CLI flags above
override the config for that run. Logs are structured JSON lines on
stderr (or the configured log file).
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from documind import __version__
from documind.config import (DEFAULT_CONFIG, load_config, resolve_paths)
from documind.evaluate import (evaluate_all_methods, format_report,
                               load_cases, run_evaluation)
from documind.index import Index
from documind.ingest import ingest_folder, SUPPORTED_EXTENSIONS
from documind.logging_config import configure_from_config
from documind.qa import answer_question
from documind.server import DEFAULT_HOST, DEFAULT_PORT, serve
from documind.store import DocumentStore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.json")
STORE_PATH = os.path.join(BASE_DIR, "documind.db")

# The effective config for this run. main() replaces it with the
# loaded config (defaults + config file + environment); direct calls
# to the cmd_* functions (e.g. from tests) use these defaults.
CONFIG = copy.deepcopy(DEFAULT_CONFIG)

DEMO_QUESTIONS = [
    "What is AEM built on?",
    "What does Spring Boot auto-configuration do?",
    "What is overfitting in machine learning?",
]


def _print_ingest_stats(stats):
    for entry in stats.files:
        outcome = entry["outcome"]
        if outcome == "added":
            print("ingested %-22s -> added" % os.path.basename(entry["path"]))
        elif outcome == "updated":
            print("ingested %-22s -> updated" % os.path.basename(entry["path"]))
        elif outcome == "unchanged":
            print("skipped  %-22s -> unchanged"
                  % os.path.basename(entry["path"]))
        elif outcome == "duplicate":
            print("skipped  %-22s -> duplicate content"
                  % os.path.basename(entry["path"]))
        elif outcome == "empty":
            print("skipped  %-22s -> empty file"
                  % os.path.basename(entry["path"]))
        elif outcome == "unsupported":
            print("skipped  %-22s -> unsupported format"
                  % os.path.basename(entry["path"]))
        else:
            print("error    %-22s -> %s"
                  % (os.path.basename(entry["path"]), entry["detail"]))
    print(stats.summary())


def cmd_ingest(args):
    if not os.path.isdir(args.folder):
        print("error: not a folder: %s" % args.folder, file=sys.stderr)
        return 1
    store = DocumentStore(STORE_PATH)
    try:
        retrieval = CONFIG["retrieval"]
        index = Index(retrieval["chunk_size"], retrieval["chunk_overlap"])
        # rebuild the index from everything already in the store, then add
        for doc in store.list_documents():
            metadata = dict(doc["metadata"])
            metadata.update({"title": doc["title"],
                             "source_path": doc["source_path"]})
            index.add_document(doc["doc_id"], doc["content"],
                               metadata=metadata)
        stats = ingest_folder(store, index, args.folder)
        if stats.documents_added == 0 and stats.documents_updated == 0:
            print("no new %s documents found in %s"
                  % ("/".join(SUPPORTED_EXTENSIONS), args.folder))
        _print_ingest_stats(stats)
        index.save(INDEX_PATH)
        print("index saved -> %s" % INDEX_PATH)
        return 0 if not stats.errors else 2
    finally:
        store.close()


def print_answer(question, result):
    print("\nQ: %s" % question)
    print("A: %s" % result["answer"])
    if result["sources"]:
        print("Sources:")
        for src in result["sources"]:
            label = src.get("title") or src["doc_id"]
            print("  - %s (chunk %d, score %.4f)"
                  % (label, src["chunk_index"], src["score"]))


def cmd_ask(args):
    if not os.path.exists(INDEX_PATH):
        print("error: no index found — run `ingest` first", file=sys.stderr)
        return 1
    index = Index.load(INDEX_PATH)
    method = args.method or CONFIG["retrieval"]["default_method"]
    try:
        result = answer_question(index, args.question,
                                 k=CONFIG["retrieval"]["ask_k"],
                                 method=method,
                                 rerank=not args.no_rerank)
    except ValueError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print_answer(args.question, result)
    return 0


def cmd_list(_args):
    if not os.path.exists(STORE_PATH):
        print("error: no document store found — run `ingest` first",
              file=sys.stderr)
        return 1
    store = DocumentStore(STORE_PATH)
    try:
        docs = store.list_documents()
    finally:
        store.close()
    if not docs:
        print("the document store is empty")
        return 0
    print("%-24s %-32s %6s  %s" % ("doc_id", "title", "chunks",
                                  "ingested_at"))
    for doc in docs:
        print("%-24s %-32s %6d  %s"
              % (doc["doc_id"][:24], doc["title"][:32],
                 doc["num_chunks"], doc["ingested_at"]))
    return 0


def cmd_stats(_args):
    if not os.path.exists(STORE_PATH):
        print("error: no document store found — run `ingest` first",
              file=sys.stderr)
        return 1
    store = DocumentStore(STORE_PATH)
    try:
        docs = store.list_documents()
    finally:
        store.close()
    total_chunks = sum(d["num_chunks"] for d in docs)
    print("documents : %d" % len(docs))
    print("chunks    : %d" % total_chunks)
    if docs:
        print("avg chunks/doc : %.1f" % (total_chunks / len(docs)))
    print("store     : %s" % STORE_PATH)
    print("index     : %s (%s)" % (INDEX_PATH,
                                   "present" if os.path.exists(INDEX_PATH)
                                   else "missing"))
    return 0


def cmd_eval(args):
    try:
        cases = load_cases(args.cases)
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    if os.path.exists(INDEX_PATH) and not args.data_dir:
        index = Index.load(INDEX_PATH)
    else:
        # no saved index (or an explicit folder): build one in memory
        # from the sample documents, so `eval` works on a fresh checkout
        folder = args.data_dir or resolve_paths(CONFIG, BASE_DIR)["data_dir"]
        if not os.path.isdir(folder):
            print("error: not a folder: %s" % folder, file=sys.stderr)
            return 1
        store = DocumentStore()
        try:
            retrieval = CONFIG["retrieval"]
            index = Index(retrieval["chunk_size"],
                          retrieval["chunk_overlap"])
            ingest_folder(store, index, folder)
        finally:
            store.close()
    k = args.k if args.k is not None else CONFIG["retrieval"]["search_k"]
    method = args.method or CONFIG["retrieval"]["default_method"]
    try:
        if args.all_methods:
            reports = evaluate_all_methods(index, cases, k=k)
        else:
            reports = run_evaluation(index, cases, k=k,
                                     method=method)
    except ValueError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print(format_report(reports), end="")
    if args.output:
        import json
        payload = reports if isinstance(reports, list) else [reports]
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print("evaluation written -> %s" % args.output)
    return 0


def cmd_serve(args):
    host = args.host if args.host is not None \
        else CONFIG["server"]["host"]
    port = args.port if args.port is not None \
        else CONFIG["server"]["port"]
    return serve(BASE_DIR, host, port, config=CONFIG)


def cmd_config(_args):
    """Print the effective config (defaults + file + environment)."""
    print(json.dumps(CONFIG, indent=2))
    return 0


def cmd_demo(_args):
    # a fresh in-memory store + index keeps the demo self-contained
    store = DocumentStore()
    retrieval = CONFIG["retrieval"]
    index = Index(retrieval["chunk_size"], retrieval["chunk_overlap"])
    stats = ingest_folder(store, index, os.path.join(BASE_DIR, "data"))
    _print_ingest_stats(stats)
    for question in DEMO_QUESTIONS:
        print_answer(question, answer_question(index, question))
    store.close()
    print()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="documind", description=__doc__)
    parser.add_argument("--version", action="version",
                        version="documind %s" % __version__)
    parser.add_argument("--config",
                        help="config file (default: $DOCUMIND_CONFIG, "
                             "else ./config.json when present, else "
                             "built-in defaults)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="index a folder of documents")
    p_ingest.add_argument("folder")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="ask a question over the index")
    p_ask.add_argument("question")
    p_ask.add_argument("--method", default=None,
                       choices=["tfidf", "bm25", "combined"],
                       help="ranking method (default: from config, "
                            "else tfidf)")
    p_ask.add_argument("--no-rerank", action="store_true",
                       help="keep retriever order instead of reranking "
                            "by best-sentence evidence")
    p_ask.set_defaults(func=cmd_ask)

    p_list = sub.add_parser("list", help="list stored documents")
    p_list.set_defaults(func=cmd_list)

    p_stats = sub.add_parser("stats", help="show document/index statistics")
    p_stats.set_defaults(func=cmd_stats)

    p_serve = sub.add_parser("serve", help="start the REST API server")
    p_serve.add_argument("--host", default=None,
                         help="bind address (default: from config, else "
                              "%s)" % DEFAULT_HOST)
    p_serve.add_argument("--port", type=int, default=None,
                         help="port (default: from config, else %d)"
                              % DEFAULT_PORT)
    p_serve.set_defaults(func=cmd_serve)

    p_demo = sub.add_parser("demo", help="ingest ./data and ask samples")
    p_demo.set_defaults(func=cmd_demo)

    p_eval = sub.add_parser(
        "eval", help="score the index on the sample Q&A set")
    p_eval.add_argument("--cases", default=os.path.join(BASE_DIR, "eval",
                                                        "eval_qa.json"),
                        help="evaluation Q&A JSON file "
                             "(default: eval/eval_qa.json)")
    p_eval.add_argument("--k", type=int, default=None,
                        help="retrieval cutoff (default: from config, "
                             "else 5)")
    p_eval.add_argument("--method", default=None,
                        choices=["tfidf", "bm25", "combined"],
                        help="ranking method (default: from config, "
                             "else tfidf)")
    p_eval.add_argument("--all-methods", action="store_true",
                        help="score tfidf, bm25 and combined side by side")
    p_eval.add_argument("--data-dir",
                        help="index this folder instead of the saved "
                             "index (default: saved index if present, "
                             "else the configured data dir)")
    p_eval.add_argument("--output",
                        help="write the report as JSON to this file")
    p_eval.set_defaults(func=cmd_eval)

    p_config = sub.add_parser(
        "config", help="print the effective configuration")
    p_config.set_defaults(func=cmd_config)

    args = parser.parse_args(argv)
    global CONFIG, INDEX_PATH, STORE_PATH
    try:
        CONFIG = load_config(path=args.config, base_dir=BASE_DIR)
    except ValueError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    paths = resolve_paths(CONFIG, BASE_DIR)
    INDEX_PATH = paths["index_path"]
    STORE_PATH = paths["store_path"]
    configure_from_config(CONFIG)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
