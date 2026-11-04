"""Integration tests for DocuMind (Day 14).

Unlike the unit tests, which exercise one component at a time, these
tests drive the whole stack together the way a real deployment would:
documents go in through the HTTP API, are served back through the API
and the web UI, survive a server restart, and are scored by the
evaluation harness — with the CLI/library pipeline agreeing with the
API at every step.

Each test starts a real ThreadingHTTPServer on an ephemeral localhost
port, pointed at a temporary data directory seeded from ./data, and
talks to it over HTTP with urllib.

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

from documind.evaluate import load_cases, run_evaluation
from documind.ingest import ingest_folder
from documind.index import Index
from documind.qa import answer_question
from documind.server import create_server
from documind.store import DocumentStore

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EVAL_CASES = os.path.join(BASE_DIR, "eval", "eval_qa.json")

NEW_DOC = {"doc_id": "integration-notes", "title": "Integration Notes",
           "text": "Zebraquills are a rare kind of quill used only "
                   "for calligraphy in this test. No other document "
                   "mentions zebraquills or calligraphy at all."}


def _request(method, url, payload=None):
    """One HTTP call; returns (status, parsed body).

    JSON bodies are parsed; the web UI page is returned as text.
    """
    body = (json.dumps(payload).encode("utf-8")
            if payload is not None else None)
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except ValueError:
                return resp.status, raw
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class IntegrationTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # seed the temp data dir from ./data, exactly like cli demo does
        store = DocumentStore(os.path.join(self.tmp.name, "documind.db"))
        index = Index()
        ingest_folder(store, index, DATA_DIR)
        store.close()
        index.save(os.path.join(self.tmp.name, "index.json"))
        self.server = None
        self.thread = None
        self._start_server()

    def _start_server(self):
        self.server = create_server(self.tmp.name, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base = "http://%s:%d" % (host, port)

    def _stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.server.documind_app.close()
        self.thread.join(timeout=5)
        self.server = None

    def tearDown(self):
        if self.server is not None:
            self._stop_server()
        self.tmp.cleanup()

    # -- full document lifecycle over HTTP ------------------------------
    def test_full_lifecycle_ingest_ask_delete_reindex(self):
        # ingest a new document through the API
        status, body = _request("POST", self.base + "/ingest", NEW_DOC)
        self.assertEqual(status, 201)
        self.assertEqual(body["outcome"], "added")

        # it shows up in the document list and in search results
        status, body = _request("GET", self.base + "/documents")
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 4)
        self.assertIn("integration-notes",
                      [d["doc_id"] for d in body["documents"]])
        status, body = _request(
            "GET", self.base + "/search?query=zebraquills%20calligraphy")
        self.assertEqual(body["results"][0]["doc_id"], "integration-notes")

        # and the QA endpoint cites it for a question about it
        status, body = _request(
            "POST", self.base + "/qa",
            {"question": "What are zebraquills used for?"})
        self.assertEqual(status, 200)
        self.assertIn("calligraphy", body["answer"])
        self.assertIn("integration-notes",
                      [c["doc_id"] for c in body["citations"]])

        # delete it: the list shrinks, search no longer finds it, and
        # a second delete is a visible 404 rather than a silent success
        status, body = _request(
            "DELETE", self.base + "/documents/integration-notes")
        self.assertEqual(status, 200)
        self.assertEqual(body["outcome"], "deleted")
        status, body = _request(
            "DELETE", self.base + "/documents/integration-notes")
        self.assertEqual(status, 404)
        status, body = _request(
            "GET", self.base + "/search?query=zebraquills%20calligraphy")
        self.assertFalse(any(r["doc_id"] == "integration-notes"
                             for r in body["results"]))

        # reindex from the store as system of record: back to the
        # original 3 documents, with counts consistent across endpoints
        status, body = _request("POST", self.base + "/reindex", {})
        self.assertEqual(status, 200)
        self.assertEqual(body["documents"], 3)
        status, health = _request("GET", self.base + "/health")
        self.assertEqual(health["documents"], 3)
        self.assertEqual(health["chunks"], body["chunks"])

    # -- restart: the store rebuilds the index ---------------------------
    def test_restart_rebuilds_index_from_store(self):
        _request("POST", self.base + "/ingest", NEW_DOC)
        _, before = _request("GET", self.base + "/health")

        self._stop_server()
        # lose the saved index entirely: the store must rebuild it
        os.remove(os.path.join(self.tmp.name, "index.json"))
        self._start_server()

        status, after = _request("GET", self.base + "/health")
        self.assertEqual(status, 200)
        self.assertEqual(after["documents"], before["documents"])
        self.assertEqual(after["chunks"], before["chunks"])
        status, body = _request(
            "GET", self.base + "/search?query=zebraquills%20calligraphy")
        self.assertEqual(body["results"][0]["doc_id"], "integration-notes")

    # -- library pipeline and API agree ----------------------------------
    def test_library_and_api_answers_agree(self):
        question = "What is AEM built on?"
        index = Index.load(os.path.join(self.tmp.name, "index.json"))
        library = answer_question(index, question)
        status, api = _request("POST", self.base + "/qa",
                              {"question": question})
        self.assertEqual(status, 200)
        self.assertEqual(api["answer"], library["answer"])
        self.assertEqual(api["citations"][0]["doc_id"],
                         library["sources"][0]["doc_id"])

    # -- evaluation harness over the API-served index ---------------------
    def test_evaluation_over_served_index(self):
        # the index the server is serving scores the shipped set
        index = Index.load(os.path.join(self.tmp.name, "index.json"))
        cases = load_cases(EVAL_CASES)
        report = run_evaluation(index, cases, k=5, method="tfidf")
        self.assertEqual(report["hit_rate"], 1.0)
        self.assertEqual(report["mrr"], 1.0)
        self.assertEqual(report["qa_citation_rate"], 1.0)

    # -- failed requests leave the state intact ---------------------------
    def test_failed_requests_leave_state_intact(self):
        _, before = _request("GET", self.base + "/health")
        # a batch of rejected requests: bad ids, empty text, bad k,
        # unknown document delete, wrong method on a known path
        status, _ = _request("POST", self.base + "/ingest",
                             {"doc_id": "bad/id", "text": "some text"})
        self.assertEqual(status, 400)
        status, _ = _request("POST", self.base + "/ingest",
                             {"doc_id": "x", "text": "   "})
        self.assertEqual(status, 400)
        status, _ = _request("GET", self.base + "/search?query=test&k=0")
        self.assertEqual(status, 400)
        status, _ = _request("DELETE", self.base + "/documents/no-such-doc")
        self.assertEqual(status, 404)
        status, _ = _request("PUT", self.base + "/ask", {})
        self.assertEqual(status, 405)
        # state is untouched and a valid ingest still works afterwards
        _, after = _request("GET", self.base + "/health")
        self.assertEqual(after, before)
        status, body = _request("POST", self.base + "/ingest", NEW_DOC)
        self.assertEqual(status, 201)

    # -- web UI journey: page, upload, ask, sources -----------------------
    def test_web_ui_journey(self):
        # the page itself, with upload + ask forms and both panels
        status, page = _request("GET", self.base + "/")
        self.assertEqual(status, 200)
        for marker in ("upload-form", "ask-form", "sources-panel",
                       "history-panel"):
            self.assertIn(marker, page)
        # the same journey a browser user takes: upload via the API
        # the form posts to, ask via the API the form posts to, then
        # check the sources panel's backing endpoint lists the upload
        status, _ = _request("POST", self.base + "/ingest", NEW_DOC)
        self.assertEqual(status, 201)
        status, body = _request(
            "POST", self.base + "/qa",
            {"question": "What are zebraquills used for?"})
        self.assertEqual(status, 200)
        self.assertIn("calligraphy", body["answer"])
        status, body = _request("GET", self.base + "/documents")
        self.assertIn("integration-notes",
                      [d["doc_id"] for d in body["documents"]])
        # and a click-to-preview fetch returns the full text
        status, body = _request(
            "GET", self.base + "/documents/integration-notes")
        self.assertEqual(status, 200)
        self.assertIn("zebraquills", body["content"])

    # -- concurrency: the single app lock keeps parallel reads safe -------
    def test_concurrent_search_requests_all_succeed(self):
        results = []

        def search():
            status, body = _request(
                "GET", self.base + "/search?query=overfitting")
            results.append((status, bool(body["results"])))

        threads = [threading.Thread(target=search) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(len(results), 8)
        self.assertTrue(all(status == 200 and found
                            for status, found in results))


if __name__ == "__main__":
    unittest.main()
