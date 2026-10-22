#!/usr/bin/env python3
"""DocuMind command-line interface.

Usage:
    python3 cli.py ingest <folder>     Index every .txt/.md file in <folder>
                                       and persist the index to index.json.
    python3 cli.py ask "<question>"    Answer a question from the saved index.
    python3 cli.py demo                Ingest ./data and ask sample questions.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from documind.index import Index
from documind.qa import answer_question

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.json")
SUPPORTED_EXTENSIONS = (".txt", ".md")

DEMO_QUESTIONS = [
    "What is AEM built on?",
    "What does Spring Boot auto-configuration do?",
    "What is overfitting in machine learning?",
]


def ingest_folder(index, folder):
    """Add every supported file in ``folder`` to ``index``."""
    total_chunks = 0
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith(SUPPORTED_EXTENSIONS))
    for fname in files:
        with open(os.path.join(folder, fname), "r", encoding="utf-8") as fh:
            n = index.add_document(fname, fh.read())
        print("ingested %-18s -> %d chunk(s)" % (fname, n))
        total_chunks += n
    return files, total_chunks


def cmd_ingest(args):
    if not os.path.isdir(args.folder):
        print("error: not a folder: %s" % args.folder, file=sys.stderr)
        return 1
    index = Index()
    files, total = ingest_folder(index, args.folder)
    if not files:
        print("no .txt/.md files found in %s" % args.folder)
        return 1
    index.save(INDEX_PATH)
    print("indexed %d document(s), %d chunk(s) -> %s"
          % (len(files), total, INDEX_PATH))
    return 0


def print_answer(question, result):
    print("\nQ: %s" % question)
    print("A: %s" % result["answer"])
    if result["sources"]:
        print("Sources:")
        for src in result["sources"]:
            print("  - %s (chunk %d, score %.4f)"
                  % (src["doc_id"], src["chunk_index"], src["score"]))


def cmd_ask(args):
    if not os.path.exists(INDEX_PATH):
        print("error: no index found — run `ingest` first", file=sys.stderr)
        return 1
    index = Index.load(INDEX_PATH)
    print_answer(args.question, answer_question(index, args.question))
    return 0


def cmd_demo(_args):
    index = Index()
    ingest_folder(index, os.path.join(BASE_DIR, "data"))
    index.save(INDEX_PATH)
    for question in DEMO_QUESTIONS:
        print_answer(question, answer_question(index, question))
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
    p_ask.set_defaults(func=cmd_ask)

    p_demo = sub.add_parser("demo", help="ingest ./data and ask samples")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
