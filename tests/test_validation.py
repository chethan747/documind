"""Tests for the Day 13 error-handling + input-validation pass.

Covers the shared validators in documind/validation.py, the core
functions that apply them (Index.search, answer_question, the server's
ingest path), and the HTTP error surface (400 / 405 / 413 as JSON),
driving a real server exactly like tests/test_server.py does.

Run from the project root:  python3 -m unittest discover -s tests
"""
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cli  # noqa: E402
from documind.index import Index  # noqa: E402
from documind.ingest import ingest_folder  # noqa: E402
from documind.qa import answer_question  # noqa: E402
from documind import server as server_module  # noqa: E402
from documind.server import create_server  # noqa: E402
from documind.store import DocumentStore  # noqa: E402
from documind.validation import (MAX_DOC_ID_CHARS, MAX_QUERY_CHARS,
                                 MAX_TEXT_CHARS, MAX_TITLE_CHARS,
                                 validate_doc_id, validate_k,
                                 validate_query, validate_text,
                                 validate_title)  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _sample_index():
    index = Index()
    index.add_document("notes.md", "Spring Boot auto-configuration "
                       "configures beans automatically from the "
                       "classpath, so applications need less setup.")
    return index


def _request(method, url, payload=None, raw=None):
    body = raw if raw is not None else (
        json.dumps(payload).encode("utf-8") if payload is not None else None)
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class TestValidators(unittest.TestCase):
    def test_k_bounds(self):
        self.assertEqual(validate_k(1), 1)
        self.assertEqual(validate_k(100), 100)
        for bad in (0, -1, 101, True, False, "5", 2.5, None):
            with self.assertRaises(ValueError, msg="k=%r" % (bad,)):
                validate_k(bad)

    def test_query_blank_nonstring_and_too_long(self):
        self.assertEqual(validate_query("  hello  "), "hello")
        for bad in ("", "   ", None, 42):
            with self.assertRaises(ValueError, msg="query=%r" % (bad,)):
                validate_query(bad)
        with self.assertRaises(ValueError):
            validate_query("x" * (MAX_QUERY_CHARS + 1))

    def test_doc_id_rules(self):
        self.assertEqual(validate_doc_id("aem-basics.md"), "aem-basics.md")
        self.assertEqual(validate_doc_id("  my_notes-1 "), "my_notes-1")
        for bad in ("", "   ", "a/b", "a\\b", "..", ".", "-lead",
                    "a b", "a?b", None, 7, "x" * (MAX_DOC_ID_CHARS + 1)):
            with self.assertRaises(ValueError, msg="doc_id=%r" % (bad,)):
                validate_doc_id(bad)

    def test_title_and_text_limits(self):
        self.assertEqual(validate_title("  My notes "), "My notes")
        with self.assertRaises(ValueError):
            validate_title("t" * (MAX_TITLE_CHARS + 1))
        with self.assertRaises(ValueError):
            validate_title(42)
        self.assertEqual(validate_text("  body  "), "body")
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError, msg="text=%r" % (bad,)):
                validate_text(bad)
        with self.assertRaises(ValueError):
            validate_text("x" * (MAX_TEXT_CHARS + 1))


class TestCoreValidation(unittest.TestCase):
    def test_search_rejects_bad_k_and_blank_query(self):
        index = _sample_index()
        for bad_k in (0, -3, 101, True, "3"):
            with self.assertRaises(ValueError, msg="k=%r" % (bad_k,)):
                index.search("Spring Boot", k=bad_k)
        for bad_q in ("", "   ", None, 12):
            with self.assertRaises(ValueError, msg="query=%r" % (bad_q,)):
                index.search(bad_q)
        # a valid k at the cap still works
        self.assertTrue(index.search("Spring Boot", k=100))

    def test_negative_k_no_longer_slices_from_the_end(self):
        # Pre-Day-13, scored[:-1] silently dropped the best hit.
        index = _sample_index()
        with self.assertRaises(ValueError):
            index.search("Spring Boot", k=-1)

    def test_blank_question_is_an_error_not_a_miss(self):
        index = _sample_index()
        with self.assertRaises(ValueError):
            answer_question(index, "   ")
        # a real miss still returns the plain "could not find" answer
        result = answer_question(index, "zzzqqq nonexistent")
        self.assertEqual(result["sources"], [])

    def test_cli_ask_blank_question_exits_with_error(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            index = Index()
            store = DocumentStore(os.path.join(tmp.name, "d.db"))
            try:
                ingest_folder(store, index, DATA_DIR)
            finally:
                store.close()
            path = os.path.join(tmp.name, "index.json")
            index.save(path)
            old = cli.INDEX_PATH
            cli.INDEX_PATH = path
            try:
                code = cli.cmd_ask(SimpleNamespace(
                    question="   ", method="tfidf", no_rerank=False))
            finally:
                cli.INDEX_PATH = old
            self.assertEqual(code, 1)
        finally:
            tmp.cleanup()


class TestServerValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        store = DocumentStore(os.path.join(self.tmp.name, "documind.db"))
        index = Index()
        ingest_folder(store, index, DATA_DIR)
        store.close()
        index.save(os.path.join(self.tmp.name, "index.json"))
        self.server = create_server(self.tmp.name, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base = "http://%s:%d" % (host, port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.server.documind_app.close()
        self.thread.join(timeout=5)
        self.tmp.cleanup()

    def test_ingest_slash_doc_id_is_400_and_not_stored(self):
        status, body = _request(
            "POST", self.base + "/ingest",
            {"doc_id": "a/b", "text": "Slash id content here."})
        self.assertEqual(status, 400)
        self.assertIn("error", body)
        # and it really was not stored under either half of the id
        status, _ = _request("GET", self.base + "/documents/a")
        self.assertEqual(status, 404)

    def test_ingest_overlong_title_is_400(self):
        status, body = _request(
            "POST", self.base + "/ingest",
            {"doc_id": "long-title", "title": "t" * (MAX_TITLE_CHARS + 1),
             "text": "Some body text."})
        self.assertEqual(status, 400)

    def test_search_bad_k_is_400(self):
        for query in ("query=test&k=0", "query=test&k=-1",
                      "query=test&k=101", "query=test&k=abc"):
            status, body = _request("GET", self.base + "/search?" + query)
            self.assertEqual(status, 400, msg=query)
            self.assertIn("error", body)

    def test_search_overlong_query_is_400(self):
        status, body = _request(
            "GET", self.base + "/search?query=" + "x" * 2001)
        self.assertEqual(status, 400)

    def test_ask_bad_k_and_bool_k_are_400(self):
        for k in (0, 101, True):
            status, body = _request(
                "POST", self.base + "/ask",
                {"question": "What is AEM built on?", "k": k})
            self.assertEqual(status, 400, msg="k=%r" % (k,))
            self.assertIn("error", body)

    def test_ask_blank_question_is_400(self):
        status, body = _request(
            "POST", self.base + "/ask", {"question": "   "})
        self.assertEqual(status, 400)

    def test_put_and_patch_return_json_405(self):
        for method in ("PUT", "PATCH"):
            status, body = _request(method, self.base + "/ask", {})
            self.assertEqual(status, 405, msg=method)
            self.assertIn("error", body)
        status, body = _request("PUT", self.base + "/nope", {})
        self.assertEqual(status, 404)

    def test_oversized_body_is_413(self):
        # Exercise the 413 path with a temporarily lowered cap, so the
        # test does not have to push a real 10 MB body through the
        # socket (the cap itself is still server_module.MAX_BODY_BYTES).
        old_max = server_module.MAX_BODY_BYTES
        server_module.MAX_BODY_BYTES = 100
        try:
            raw = b'{"text": "' + b"x" * 200 + b'"}'
            status, body = _request("POST", self.base + "/ingest", raw=raw)
        finally:
            server_module.MAX_BODY_BYTES = old_max
        self.assertEqual(status, 413)
        self.assertIn("error", body)


if __name__ == "__main__":
    unittest.main()
