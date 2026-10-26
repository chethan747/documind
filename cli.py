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
    python3 cli.py demo                Ingest ./data and ask sample questions.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from documind.index import Index
from documind.ingest import ingest_folder, SUPPORTED_EXTENSIONS
from documind.qa import answer_question
from documind.store import DocumentStore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.json")
STORE_PATH = os.path.join(BASE_DIR, "documind.db")

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
        index = Index()
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
    try:
        result = answer_question(index, args.question, method=args.method,
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


def cmd_demo(_args):
    # a fresh in-memory store + index keeps the demo self-contained
    store = DocumentStore()
    index = Index()
    stats = ingest_folder(store, index, os.path.join(BASE_DIR, "data"))
    _print_ingest_stats(stats)
    for question in DEMO_QUESTIONS:
        print_answer(question, answer_question(index, question))
    store.close()
    print()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="documind", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="index a folder of documents")
    p_ingest.add_argument("folder")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="ask a question over the index")
    p_ask.add_argument("question")
    p_ask.add_argument("--method", default="tfidf",
                       choices=["tfidf", "bm25", "combined"],
                       help="ranking method (default: tfidf)")
    p_ask.add_argument("--no-rerank", action="store_true",
                       help="keep retriever order instead of reranking "
                            "by best-sentence evidence")
    p_ask.set_defaults(func=cmd_ask)

    p_list = sub.add_parser("list", help="list stored documents")
    p_list.set_defaults(func=cmd_list)

    p_stats = sub.add_parser("stats", help="show document/index statistics")
    p_stats.set_defaults(func=cmd_stats)

    p_demo = sub.add_parser("demo", help="ingest ./data and ask samples")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
