"""Tests for the DocuMind web UI (Day 11).

``GET /`` (and ``GET /index.html``) serves a single dependency-free
HTML page with an upload form posting to ``POST /ingest`` and an ask
form posting to ``POST /qa``. The tests start a real
ThreadingHTTPServer on an ephemeral localhost port, pointed at a
temporary data directory seeded from ./data, and exercise it over HTTP
with urllib — the same way a browser would.

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


class WebUiTestCase(unittest.TestCase):
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

    def test_root_serves_upload_and_ask_page(self):
        status, content_type, body = _get(self.base + "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn("<title>DocuMind", body)
        # upload form: posts a document to /ingest
        self.assertIn('id="upload-form"', body)
        self.assertIn('id="upload-text"', body)
        self.assertIn('fetch', body)
        self.assertIn('"/ingest"', body)
        # ask form: posts a question to /qa
        self.assertIn('id="ask-form"', body)
        self.assertIn('id="ask-question"', body)
        self.assertIn('"/qa"', body)
        self.assertIn('"/health"', body)

    def test_index_html_serves_same_page(self):
        _, _, root = _get(self.base + "/")
        status, content_type, body = _get(self.base + "/index.html")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertEqual(body, root)

    def test_page_renders_user_text_safely(self):
        # The page never injects content as HTML: answers and snippets
        # are assigned via textContent, so markup in a document or an
        # answer cannot run as script in the browser.
        _, _, body = _get(self.base + "/")
        self.assertIn("textContent", body)
        self.assertNotIn("innerHTML", body)

    def test_post_to_root_is_405(self):
        status, body = _post_json(self.base + "/", {"question": "x"})
        self.assertEqual(status, 405)
        self.assertIn("error", body)

    def test_upload_then_ask_flow_behind_the_page(self):
        # The exact round trip the page's two forms perform: upload a
        # document, then ask a question answered from it.
        status, body = _post_json(
            self.base + "/ingest",
            {"doc_id": "webui-note.md", "title": "Web UI note",
             "text": "The harbour lighthouse flashes twice every night "
                     "to guide ships safely into port."})
        self.assertEqual(status, 201)
        self.assertEqual(body["outcome"], "added")
        status, body = _post_json(
            self.base + "/qa",
            {"question": "What does the harbour lighthouse do at night?"})
        self.assertEqual(status, 200)
        self.assertEqual(body["citations"][0]["doc_id"], "webui-note.md")
        self.assertIn("lighthouse", body["answer"].lower())

    def test_page_served_on_empty_index(self):
        tmp = tempfile.TemporaryDirectory()
        server = create_server(tmp.name, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        try:
            status, content_type, body = _get(
                "http://%s:%d/" % (host, port))
            self.assertEqual(status, 200)
            self.assertIn("text/html", content_type)
            self.assertIn('id="upload-form"', body)
            self.assertIn('id="ask-form"', body)
        finally:
            server.shutdown()
            server.server_close()
            server.documind_app.close()
            thread.join(timeout=5)
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
