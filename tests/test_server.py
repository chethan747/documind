"""Tests for the DocuMind REST API server (Day 8).

The tests start a real ThreadingHTTPServer on an ephemeral localhost
port, pointed at a temporary data directory seeded from ./data, and
exercise it over HTTP with urllib — the same way a client would.

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from documind.ingest import ingest_folder
from documind.index import Index
from documind.server import create_server
from documind.store import DocumentStore

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data")


def _request(method, url, payload=None, raw=None):
    """One HTTP call; returns (status, parsed JSON body)."""
    body = raw if raw is not None else (
        json.dumps(payload).encode("utf-8") if payload is not None else None)
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class ServerTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # seed the temp data dir from ./data, exactly like cli demo does
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

    # -- health / stats --------------------------------------------------
    def test_health_reports_version_and_counts(self):
        status, body = _request("GET", self.base + "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["documents"], 3)
        self.assertGreater(body["chunks"], 0)
        self.assertIn("version", body)

    def test_stats(self):
        status, body = _request("GET", self.base + "/stats")
        self.assertEqual(status, 200)
        self.assertEqual(body["documents"], 3)
        self.assertGreater(body["chunks"], 0)
        self.assertTrue(body["index_present"])

    # -- search ----------------------------------------------------------
    def test_search_get_returns_hits(self):
        status, body = _request(
            "GET", self.base + "/search?query=Spring%20Boot%20auto-configuration")
        self.assertEqual(status, 200)
        self.assertTrue(body["results"])
        top = body["results"][0]
        self.assertIn("doc_id", top)
        self.assertIn("text", top)
        self.assertGreater(top["score"], 0)

    def test_search_method_bm25(self):
        status, body = _request(
            "GET", self.base + "/search?query=overfitting&method=bm25")
        self.assertEqual(status, 200)
        self.assertTrue(body["results"])

    def test_search_missing_query_is_400(self):
        status, body = _request("GET", self.base + "/search")
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_search_unknown_method_is_400(self):
        status, body = _request(
            "GET", self.base + "/search?query=test&method=nope")
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    # -- ask ---------------------------------------------------------------
    def test_ask_post_returns_cited_answer(self):
        status, body = _request(
            "POST", self.base + "/ask",
            {"question": "What is AEM built on?"})
        self.assertEqual(status, 200)
        self.assertIn("[", body["answer"])  # citation marker
        self.assertTrue(body["sources"])
        self.assertEqual(body["sources"][0]["doc_id"], "aem-basics.md")

    def test_ask_missing_question_is_400(self):
        status, body = _request("POST", self.base + "/ask", {})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_ask_invalid_json_is_400(self):
        status, body = _request("POST", self.base + "/ask",
                                raw=b"{not json")
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    # -- ingest ------------------------------------------------------------
    def test_ingest_then_search_finds_new_document(self):
        status, body = _request(
            "POST", self.base + "/ingest",
            {"doc_id": "api-notes", "title": "API Notes",
             "text": "Zebraquills are a rare kind of quill used only "
                     "for calligraphy in this test."})
        self.assertEqual(status, 201)
        self.assertEqual(body["outcome"], "added")
        self.assertGreater(body["num_chunks"], 0)

        status, body = _request(
            "GET", self.base + "/search?query=zebraquills%20calligraphy")
        self.assertEqual(status, 200)
        self.assertTrue(body["results"])
        self.assertEqual(body["results"][0]["doc_id"], "api-notes")

    def test_ingest_unchanged_and_duplicate(self):
        payload = {"doc_id": "dup-test", "title": "Dup Test",
                   "text": "Unique wombat facts for dedup testing."}
        status, _ = _request("POST", self.base + "/ingest", payload)
        self.assertEqual(status, 201)

        status, body = _request("POST", self.base + "/ingest", payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["outcome"], "unchanged")

        status, body = _request(
            "POST", self.base + "/ingest",
            {"doc_id": "other-id", "text": payload["text"]})
        self.assertEqual(status, 409)
        self.assertEqual(body["outcome"], "duplicate")

    def test_ingest_updated_replaces_content(self):
        _request("POST", self.base + "/ingest",
                 {"doc_id": "mutable", "text": "Old flibberty content."})
        status, body = _request(
            "POST", self.base + "/ingest",
            {"doc_id": "mutable", "text": "New glimmerwort content."})
        self.assertEqual(status, 200)
        self.assertEqual(body["outcome"], "updated")

        status, body = _request("GET", self.base + "/search?query=glimmerwort")
        self.assertTrue(any(r["doc_id"] == "mutable"
                            for r in body["results"]))
        status, body = _request("GET", self.base + "/search?query=flibberty")
        self.assertFalse(any(r["doc_id"] == "mutable"
                             for r in body["results"]))

    def test_ingest_empty_text_is_400(self):
        status, body = _request(
            "POST", self.base + "/ingest", {"doc_id": "x", "text": "   "})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    # -- routing errors ------------------------------------------------------
    def test_unknown_path_is_404(self):
        status, body = _request("GET", self.base + "/nope")
        self.assertEqual(status, 404)
        self.assertIn("error", body)

    def test_wrong_method_is_405(self):
        status, body = _request("GET", self.base + "/ask")
        self.assertEqual(status, 405)
        status, body = _request("POST", self.base + "/health", {})
        self.assertEqual(status, 405)


if __name__ == "__main__":
    unittest.main()
