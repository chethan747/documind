"""Tests for the dedicated DocuMind QA endpoint (Day 10).

POST /qa is the canonical question-answering endpoint: the same
extractive pipeline as POST /ask, but the response is built around
structured ``citations`` (document id/title, chunk index, scores,
snippet, readable citation label) instead of the looser ``sources``
shape /ask keeps for compatibility.

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


class QaEndpointTestCase(unittest.TestCase):
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

    def test_qa_returns_answer_with_structured_citations(self):
        status, body = _request(
            "POST", self.base + "/qa",
            {"question": "What is AEM built on?"})
        self.assertEqual(status, 200)
        self.assertEqual(body["question"], "What is AEM built on?")
        self.assertEqual(body["method"], "tfidf")
        self.assertTrue(body["rerank"])
        self.assertIn("[", body["answer"])  # citation marker in the text
        self.assertTrue(body["citations"])
        self.assertEqual(body["source_count"], len(body["citations"]))
        first = body["citations"][0]
        self.assertEqual(first["doc_id"], "aem-basics.md")
        self.assertIn("title", first)
        self.assertIsInstance(first["chunk_index"], int)
        self.assertGreater(first["score"], 0)
        self.assertIn("rerank_score", first)
        self.assertTrue(first["snippet"])
        self.assertIn("aem-basics.md", first["citation"])
        self.assertIn("chunk", first["citation"])
        # every cited document is actually cited in the answer text
        for citation in body["citations"]:
            self.assertIn("[%s]" % citation["doc_id"], body["answer"])

    def test_qa_bm25_method(self):
        status, body = _request(
            "POST", self.base + "/qa",
            {"question": "What is overfitting?", "method": "bm25"})
        self.assertEqual(status, 200)
        self.assertEqual(body["method"], "bm25")
        self.assertTrue(body["citations"])

    def test_qa_without_rerank_omits_rerank_score(self):
        status, body = _request(
            "POST", self.base + "/qa",
            {"question": "What is AEM built on?", "rerank": False})
        self.assertEqual(status, 200)
        self.assertFalse(body["rerank"])
        self.assertTrue(body["citations"])
        for citation in body["citations"]:
            self.assertNotIn("rerank_score", citation)

    def test_qa_missing_question_is_400(self):
        status, body = _request("POST", self.base + "/qa", {})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_qa_invalid_json_is_400(self):
        status, body = _request("POST", self.base + "/qa",
                                raw=b"{not json")
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_qa_unknown_method_is_400(self):
        status, body = _request(
            "POST", self.base + "/qa",
            {"question": "What is AEM built on?", "method": "nope"})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_qa_non_integer_k_is_400(self):
        status, body = _request(
            "POST", self.base + "/qa",
            {"question": "What is AEM built on?", "k": "three"})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_qa_get_is_405(self):
        status, body = _request("GET", self.base + "/qa")
        self.assertEqual(status, 405)
        self.assertIn("error", body)

    def test_ask_still_works_alongside_qa(self):
        status, body = _request(
            "POST", self.base + "/ask",
            {"question": "What is AEM built on?"})
        self.assertEqual(status, 200)
        self.assertTrue(body["sources"])
        self.assertEqual(body["sources"][0]["doc_id"], "aem-basics.md")


class QaEmptyIndexTestCase(unittest.TestCase):
    def test_qa_on_empty_index_is_200_with_no_citations(self):
        tmp = tempfile.TemporaryDirectory()
        server = create_server(tmp.name, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        try:
            status, body = _request(
                "POST", "http://%s:%d/qa" % (host, port),
                {"question": "Anything at all?"})
            self.assertEqual(status, 200)
            self.assertEqual(body["citations"], [])
            self.assertEqual(body["source_count"], 0)
            self.assertIn("could not find", body["answer"])
        finally:
            server.shutdown()
            server.server_close()
            server.documind_app.close()
            thread.join(timeout=5)
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
