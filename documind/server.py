"""REST API server for DocuMind (Day 8).

Pure standard library: ``http.server.ThreadingHTTPServer`` serves the
search index and document store built by the rest of the package, so the
API works with no third-party dependencies, exactly like the CLI.

Endpoints (all JSON in, JSON out):

    GET  /health                 {"status": "ok", "version", ...}
    GET  /stats                  document / chunk counts
    POST /ingest                 {"doc_id"?, "title"?, "text"} — add or
                                 update one document (same dedup rules as
                                 the file pipeline: unchanged, duplicate,
                                 added, updated)
    GET  /search?query=...[&k=][&method=]
                                 top-k retrieval hits
    POST /ask                    {"question"[, "k"][, "method"]
                                 [, "rerank"]} — extractive answer with
                                 source citations, same as ``cli.py ask``

Status codes: 200 on success (201 when /ingest adds a new document),
400 for missing/invalid input, 404 for unknown paths, 405 for a known
path with the wrong method, 409 when /ingest text duplicates another
document's content.

Run it from the CLI — ``python3 cli.py serve [--host] [--port]`` — or
directly: ``python3 -m documind.server [--host] [--port] [--dir DIR]``.
``--dir`` is the project directory holding ``index.json`` /
``documind.db`` (defaults to the project root); a saved index is loaded
when present, otherwise the index is rebuilt from the document store.
"""
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import __version__
from .index import SEARCH_METHODS, Index
from .ingest import sha256_text
from .qa import answer_question
from .store import DocumentStore

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_BODY_BYTES = 10 * 1024 * 1024  # refuse absurd payloads outright


def _slug(text):
    """Turn a title into a URL/document friendly slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "document"


class DocuMindApp:
    """The API's state: a document store, a search index, and a lock.

    One lock serialises every request that touches the store or index,
    so concurrent threads served by ThreadingHTTPServer cannot corrupt
    either. Reads are cheap (in-memory index), so a single lock keeps
    the code simple and correct.
    """

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.index_path = os.path.join(base_dir, "index.json")
        self.store_path = os.path.join(base_dir, "documind.db")
        self.lock = threading.Lock()
        # check_same_thread=False is safe here because every store/index
        # access goes through the single app lock above.
        self.store = DocumentStore(self.store_path,
                                   check_same_thread=False)
        self.index = self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path):
            return Index.load(self.index_path)
        index = Index()
        for doc in self.store.list_documents():
            metadata = dict(doc["metadata"])
            metadata.update({"title": doc["title"],
                             "source_path": doc["source_path"]})
            index.add_document(doc["doc_id"], doc["content"],
                               metadata=metadata)
        return index

    def close(self):
        self.store.close()

    # -- endpoints -----------------------------------------------------
    def health(self):
        return {"status": "ok", "version": __version__,
                "documents": self.store.count(),
                "chunks": len(self.index)}

    def stats(self):
        docs = self.store.list_documents()
        return {"documents": len(docs),
                "chunks": len(self.index),
                "index_present": os.path.exists(self.index_path)}

    def ingest_text(self, doc_id, title, text):
        """Add or update one document given as raw text.

        Returns (status, payload) following the file pipeline's dedup
        rules: identical text under the same id is "unchanged", identical
        text under another id is "duplicate", otherwise the document is
        added or updated and the index is persisted.
        """
        text = text.strip()
        if not text:
            return 400, {"error": "text must not be empty"}
        doc_id = doc_id or _slug(title or "document")
        title = title or doc_id
        text_hash = sha256_text(text)

        existing = self.store.get_document(doc_id)
        if existing is not None and existing["text_hash"] == text_hash:
            return 200, {"outcome": "unchanged", "doc_id": doc_id,
                         "title": existing["title"],
                         "num_chunks": existing["num_chunks"]}
        dup = self.store.find_by_hash(text_hash)
        if dup is not None and dup["doc_id"] != doc_id:
            return 409, {"outcome": "duplicate", "doc_id": doc_id,
                         "detail": "identical content already stored as %r"
                                   % dup["doc_id"]}

        metadata = {"source": "api"}
        self.store.add_document(doc_id, title, None, text, text_hash,
                                metadata=metadata)
        self.index.remove_document(doc_id)
        index_metadata = dict(metadata)
        index_metadata.update({"title": title, "source_path": None})
        num_chunks = self.index.add_document(doc_id, text,
                                             metadata=index_metadata)
        self.store.set_num_chunks(doc_id, num_chunks)
        self.index.save(self.index_path)
        status = 201 if existing is None else 200
        outcome = "added" if existing is None else "updated"
        return status, {"outcome": outcome, "doc_id": doc_id,
                        "title": title, "num_chunks": num_chunks}

    def search(self, query, k=5, method="tfidf"):
        """Return (status, payload) for a retrieval request."""
        if method not in SEARCH_METHODS:
            return 400, {"error": "unknown search method %r (expected one "
                                  "of %s)" % (method, ", ".join(SEARCH_METHODS))}
        try:
            hits = self.index.search(query, k=k, method=method)
        except ValueError as exc:
            return 400, {"error": str(exc)}
        return 200, {"query": query, "method": method, "results": hits}

    def ask(self, question, k=3, method="tfidf", rerank=True):
        """Return (status, payload) for a question-answering request."""
        if method not in SEARCH_METHODS:
            return 400, {"error": "unknown search method %r (expected one "
                                  "of %s)" % (method, ", ".join(SEARCH_METHODS))}
        try:
            result = answer_question(self.index, question, k=k,
                                     method=method, rerank=rerank)
        except ValueError as exc:
            return 400, {"error": str(exc)}
        return 200, {"question": question, "method": method,
                     "answer": result["answer"],
                     "sources": result["sources"]}


def _parse_int(value, name, default):
    """Parse a query-string integer, raising ValueError on bad input."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be an integer, got %r" % (name, value))


def make_handler(app):
    """Build a request handler class bound to ``app``."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "DocuMind/%s" % __version__

        # -- plumbing --------------------------------------------------
        def log_message(self, *args):  # keep the server quiet by default
            pass

        def _send(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                raise ValueError("request body too large")
            raw = self.rfile.read(length) if length else b""
            if not raw:
                raise ValueError("empty request body: expected JSON")
            try:
                data = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                raise ValueError("request body is not valid JSON")
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            return data

        def _require_str(self, data, field):
            value = data.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("%s must be a non-empty string" % field)
            return value.strip()

        # -- routing ---------------------------------------------------
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                with app.lock:
                    self._send(200, app.health())
            elif parsed.path == "/stats":
                with app.lock:
                    self._send(200, app.stats())
            elif parsed.path == "/search":
                params = parse_qs(parsed.query)
                try:
                    query = self._require_str(
                        {"query": (params.get("query") or [None])[0]},
                        "query")
                    k = _parse_int((params.get("k") or [None])[0],
                                   "k", 5)
                    method = (params.get("method") or ["tfidf"])[0]
                except ValueError as exc:
                    self._send(400, {"error": str(exc)})
                    return
                with app.lock:
                    status, payload = app.search(query, k=k, method=method)
                self._send(status, payload)
            elif parsed.path in ("/ingest", "/ask"):
                self._send(405, {"error": "%s expects POST" % parsed.path})
            else:
                self._send(404, {"error": "unknown path %r" % parsed.path})

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path not in ("/ingest", "/ask"):
                if parsed.path in ("/health", "/stats", "/search"):
                    self._send(405, {"error": "%s expects GET" % parsed.path})
                else:
                    self._send(404, {"error": "unknown path %r"
                                              % parsed.path})
                return
            try:
                data = self._read_json()
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            try:
                if parsed.path == "/ingest":
                    text = self._require_str(data, "text")
                    doc_id = data.get("doc_id")
                    if doc_id is not None and (
                            not isinstance(doc_id, str) or not doc_id.strip()):
                        raise ValueError("doc_id must be a non-empty string")
                    title = data.get("title")
                    if title is not None and not isinstance(title, str):
                        raise ValueError("title must be a string")
                    with app.lock:
                        status, payload = app.ingest_text(
                            doc_id.strip() if doc_id else None,
                            title.strip() if title else None, text)
                else:  # /ask
                    question = self._require_str(data, "question")
                    k = data.get("k", 3)
                    if not isinstance(k, int) or isinstance(k, bool):
                        raise ValueError("k must be an integer")
                    method = data.get("method", "tfidf")
                    if not isinstance(method, str):
                        raise ValueError("method must be a string")
                    rerank = data.get("rerank", True)
                    if not isinstance(rerank, bool):
                        raise ValueError("rerank must be a boolean")
                    with app.lock:
                        status, payload = app.ask(question, k=k,
                                                  method=method,
                                                  rerank=rerank)
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            self._send(status, payload)

    return Handler


def create_server(base_dir, host=DEFAULT_HOST, port=DEFAULT_PORT):
    """Build (but do not start) a server for ``base_dir``."""
    app = DocuMindApp(base_dir)
    server = ThreadingHTTPServer((host, port), make_handler(app))
    server.documind_app = app
    return server


def serve(base_dir, host=DEFAULT_HOST, port=DEFAULT_PORT):
    """Serve forever; Ctrl-C stops the server cleanly."""
    server = create_server(base_dir, host, port)
    print("DocuMind API listening on http://%s:%d (data dir: %s)"
          % (host, server.server_address[1], base_dir))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.documind_app.close()
    return 0


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="documind.server",
        description="DocuMind REST API server (standard library only).")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dir", default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))),
        help="project directory holding index.json / documind.db "
             "(default: project root)")
    args = parser.parse_args(argv)
    return serve(args.dir, args.host, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
