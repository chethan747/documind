"""Tests for the DocuMind web UI polish (Day 12).

Day 12 adds two panels to the Day 11 page, both rendered safely with
textContent only:

* **Sources panel** — a live list of indexed documents fetched from
  ``GET /documents`` (title, id, chunk count), with a refresh button
  and a click-to-preview that fetches ``GET /documents/<doc_id>``.
* **History** — past questions kept in ``localStorage`` under
  ``documind-history`` (newest first, capped at 20), each clickable to
  restore its answer and citations, with a clear button.

The browser-side storage itself cannot be driven over HTTP, so these
tests assert the page carries the exact panel markup and JavaScript
the feature needs (ids, storage key, cap, endpoints, safe rendering),
and exercise over HTTP the API round trips the panels depend on: list
documents, fetch one document's text, and see the list grow after an
upload — the same calls the page's fetch code makes.

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


def _get(url):
    """One GET; returns (status, content-type, decoded body)."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return (resp.status, resp.headers.get("Content-Type"),
                    resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type"), \
            exc.read().decode("utf-8")


def _get_json(url):
    """One GET expecting JSON; returns (status, parsed body)."""
    status, _, body = _get(url)
    return status, json.loads(body)


def _post_json(url, payload):
    """One POST with a JSON body; returns (status, parsed body)."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class UiPolishTestCase(unittest.TestCase):
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

    def test_page_has_sources_panel(self):
        _, _, body = _get(self.base + "/")
        self.assertIn('id="sources-panel"', body)
        self.assertIn('id="sources-list"', body)
        self.assertIn('id="refresh-sources"', body)
        self.assertIn('id="source-preview"', body)
        # The panel is fed by the documents endpoints, not hardcoded.
        self.assertIn('"/documents"', body)
        self.assertIn('"/documents/"', body)
        self.assertIn("encodeURIComponent", body)
        self.assertIn("refreshSources", body)

    def test_page_has_history_panel(self):
        _, _, body = _get(self.base + "/")
        self.assertIn('id="history-panel"', body)
        self.assertIn('id="history-list"', body)
        self.assertIn('id="clear-history"', body)
        self.assertIn("localStorage", body)
        self.assertIn("documind-history", body)
        self.assertIn("HISTORY_LIMIT = 20", body)
        self.assertIn("renderHistory", body)
        self.assertIn("saveHistory", body)

    def test_polish_panels_render_safely(self):
        # Sources titles, history questions and previews all go through
        # textContent, so stored markup can never run as script.
        _, _, body = _get(self.base + "/")
        self.assertIn("textContent", body)
        self.assertNotIn("innerHTML", body)

    def test_sources_round_trip_behind_the_panel(self):
        # Exactly what the sources panel fetches: the seeded list,
        # then one document's stored text for the preview.
        status, body = _get_json(self.base + "/documents")
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 3)
        ids = [d["doc_id"] for d in body["documents"]]
        self.assertIn("aem-basics.md", ids)
        for doc in body["documents"]:
            self.assertIn("num_chunks", doc)
            self.assertNotIn("content", doc)
        status, body = _get_json(self.base + "/documents/aem-basics.md")
        self.assertEqual(status, 200)
        self.assertIn("content", body)
        self.assertIn("AEM", body["content"])

    def test_sources_list_grows_after_upload(self):
        # The panel refreshes after an upload; the list behind it must
        # reflect the new document immediately.
        status, body = _post_json(
            self.base + "/ingest",
            {"doc_id": "polish-note.md", "title": "Polish note",
             "text": "The quiet archive keeps every letter filed "
                     "under the sender's name."})
        self.assertEqual(status, 201)
        status, body = _get_json(self.base + "/documents")
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 4)
        titles = {d["doc_id"]: d["title"] for d in body["documents"]}
        self.assertEqual(titles["polish-note.md"], "Polish note")

    def test_panels_present_on_empty_index(self):
        tmp = tempfile.TemporaryDirectory()
        server = create_server(tmp.name, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        try:
            status, _, body = _get("http://%s:%d/" % (host, port))
            self.assertEqual(status, 200)
            self.assertIn('id="sources-panel"', body)
            self.assertIn('id="history-panel"', body)
            self.assertIn("No documents indexed yet.", body)
            self.assertIn("No questions asked yet.", body)
            status, docs = _get_json("http://%s:%d/documents" % (host, port))
            self.assertEqual(status, 200)
            self.assertEqual(docs["count"], 0)
        finally:
            server.shutdown()
            server.server_close()
            server.documind_app.close()
            thread.join(timeout=5)
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
