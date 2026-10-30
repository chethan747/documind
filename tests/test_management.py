"""Tests for the DocuMind document management endpoints (Day 9).

GET /documents, GET /documents/<doc_id>, DELETE /documents/<doc_id>
and POST /reindex, exercised over HTTP against a real server seeded
from ./data, exactly like tests/test_server.py.

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


class ManagementTestCase(unittest.TestCase):
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

    # -- list ------------------------------------------------------------
    def test_list_documents(self):
        status, body = _request("GET", self.base + "/documents")
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 3)
        self.assertEqual(len(body["documents"]), 3)
        ids = {d["doc_id"] for d in body["documents"]}
        self.assertEqual(ids, {"aem-basics.md", "java-spring.md",
                               "ml-basics.md"})
        for doc in body["documents"]:
            self.assertIn("title", doc)
            self.assertIn("num_chunks", doc)
            self.assertGreater(doc["num_chunks"], 0)
            self.assertIn("ingested_at", doc)
            # summaries carry no full text
            self.assertNotIn("content", doc)

    # -- get one -----------------------------------------------------------
    def test_get_document(self):
        status, body = _request("GET", self.base + "/documents/ml-basics.md")
        self.assertEqual(status, 200)
        self.assertEqual(body["doc_id"], "ml-basics.md")
        self.assertIn("overfitting", body["content"])
        self.assertGreater(body["num_chunks"], 0)

    def test_get_unknown_document_is_404(self):
        status, body = _request("GET", self.base + "/documents/nope.md")
        self.assertEqual(status, 404)
        self.assertIn("error", body)

    # -- delete ------------------------------------------------------------
    def test_delete_removes_from_store_and_index(self):
        status, body = _request(
            "DELETE", self.base + "/documents/ml-basics.md")
        self.assertEqual(status, 200)
        self.assertEqual(body["outcome"], "deleted")
        self.assertGreater(body["chunks_removed"], 0)

        status, body = _request("GET", self.base + "/documents")
        self.assertEqual(body["count"], 2)
        self.assertNotIn("ml-basics.md",
                         {d["doc_id"] for d in body["documents"]})

        status, body = _request(
            "GET", self.base + "/search?query=overfitting")
        self.assertFalse(any(r["doc_id"] == "ml-basics.md"
                             for r in body["results"]))

        status, body = _request("GET", self.base + "/health")
        self.assertEqual(body["documents"], 2)

    def test_delete_unknown_is_404_and_repeat_delete_is_404(self):
        status, _ = _request("DELETE", self.base + "/documents/nope.md")
        self.assertEqual(status, 404)
        _request("DELETE", self.base + "/documents/aem-basics.md")
        status, _ = _request("DELETE", self.base + "/documents/aem-basics.md")
        self.assertEqual(status, 404)

    def test_deleted_id_can_be_reingested(self):
        _request("DELETE", self.base + "/documents/ml-basics.md")
        status, body = _request(
            "POST", self.base + "/ingest",
            {"doc_id": "ml-basics.md", "title": "ML Basics",
             "text": "Fresh overfitting notes, rewritten for this test."})
        self.assertEqual(status, 201)
        self.assertEqual(body["outcome"], "added")
        status, body = _request("GET", self.base + "/documents")
        self.assertEqual(body["count"], 3)

    # -- reindex -------------------------------------------------------------
    def test_reindex_reports_counts_and_keeps_search_working(self):
        status, body = _request("POST", self.base + "/reindex")
        self.assertEqual(status, 200)
        self.assertEqual(body["outcome"], "reindexed")
        self.assertEqual(body["documents"], 3)
        self.assertGreater(body["chunks"], 0)
        self.assertIn("previous_chunks", body)

        status, body = _request(
            "GET", self.base + "/search?query=Spring%20Boot%20auto-configuration")
        self.assertTrue(body["results"])
        self.assertEqual(body["results"][0]["doc_id"], "java-spring.md")

    def test_reindex_repairs_index_drift(self):
        # Simulate drift: drop a document's chunks from the index only,
        # leaving the store untouched — search then misses it.
        app = self.server.documind_app
        with app.lock:
            app.index.remove_document("ml-basics.md")
        status, body = _request(
            "GET", self.base + "/search?query=overfitting")
        self.assertFalse(any(r["doc_id"] == "ml-basics.md"
                             for r in body["results"]))

        status, body = _request("POST", self.base + "/reindex")
        self.assertEqual(status, 200)
        self.assertEqual(body["documents"], 3)

        status, body = _request(
            "GET", self.base + "/search?query=overfitting")
        self.assertTrue(any(r["doc_id"] == "ml-basics.md"
                            for r in body["results"]))

    def test_reindex_after_delete_does_not_resurrect(self):
        _request("DELETE", self.base + "/documents/ml-basics.md")
        status, body = _request("POST", self.base + "/reindex")
        self.assertEqual(body["documents"], 2)
        status, body = _request("GET", self.base + "/documents")
        self.assertEqual(body["count"], 2)

    # -- routing errors ------------------------------------------------------
    def test_wrong_methods_are_405(self):
        status, _ = _request("POST", self.base + "/documents", {})
        self.assertEqual(status, 405)
        status, _ = _request("GET", self.base + "/reindex")
        self.assertEqual(status, 405)
        status, _ = _request("DELETE", self.base + "/documents")
        self.assertEqual(status, 405)
        status, _ = _request("DELETE", self.base + "/health")
        self.assertEqual(status, 405)


if __name__ == "__main__":
    unittest.main()
