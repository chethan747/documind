"""Tests for the DocuMind ingestion pipeline and SQLite document store (Day 2).

Run from the project root:  python3 -m unittest discover -s tests
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from documind.index import Index
from documind.ingest import (IngestStats, UnsupportedFormatError, extract_text,
                             ingest_file, ingest_folder, sha256_text,
                             strip_html)
from documind.store import DocumentStore


class TestExtractText(unittest.TestCase):
    def _write(self, tmpdir, name, content, binary=False):
        path = os.path.join(tmpdir, name)
        mode = "wb" if binary else "w"
        with open(path, mode, encoding=None if binary else "utf-8") as fh:
            fh.write(content)
        return path

    def test_txt_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "notes.txt", "Hello world.\nSecond line.")
            title, text = extract_text(path)
            self.assertEqual(title, "notes")
            self.assertIn("Hello world.", text)

    def test_md_extraction_keeps_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "readme.md", "# Title\n\nSome *text* here.")
            title, text = extract_text(path)
            self.assertEqual(title, "readme")
            self.assertIn("Some *text* here.", text)

    def test_html_strips_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp, "page.html",
                "<html><head><style>body{color:red}</style></head>"
                "<body><h1>Headline</h1><p>Real &amp; content.</p></body></html>")
            _title, text = extract_text(path)
            self.assertIn("Headline", text)
            self.assertIn("Real & content.", text)
            self.assertNotIn("<h1>", text)
            self.assertNotIn("color:red", text)

    def test_strip_html_removes_scripts(self):
        out = strip_html("<script>alert(1)</script><p>keep me</p>")
        self.assertEqual(out, "keep me")

    def test_unsupported_extension_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "doc.pdf", "binary-ish")
            with self.assertRaises(UnsupportedFormatError):
                extract_text(path)

    def test_missing_file_raises_oserror(self):
        with self.assertRaises(OSError):
            extract_text("/does/not/exist.txt")

    def test_latin1_fallback_decodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "enc.txt", "caf\xe9".encode("latin-1"),
                               binary=True)
            _title, text = extract_text(path)
            self.assertIn("caf", text)


class TestDocumentStore(unittest.TestCase):
    def setUp(self):
        self.store = DocumentStore()

    def tearDown(self):
        self.store.close()

    def _add(self, doc_id="doc1", text="some content here", **kw):
        params = {"title": "Doc 1", "source_path": "/tmp/doc1.txt",
                  "content": text, "text_hash": sha256_text(text)}
        params.update(kw)
        return self.store.add_document(doc_id, **params)

    def test_add_and_get_round_trip(self):
        row = self._add()
        fetched = self.store.get_document("doc1")
        self.assertEqual(fetched["content"], "some content here")
        self.assertEqual(fetched["title"], "Doc 1")
        self.assertEqual(row["doc_id"], fetched["doc_id"])

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.store.get_document("nope"))

    def test_readd_same_content_is_noop(self):
        first = self._add()
        second = self._add()
        self.assertEqual(first["ingested_at"], second["ingested_at"])
        self.assertEqual(self.store.count(), 1)

    def test_readd_changed_content_updates(self):
        self._add()
        row = self._add(text="totally different content")
        self.assertEqual(row["content"], "totally different content")
        self.assertEqual(self.store.count(), 1)

    def test_find_by_hash(self):
        self._add(doc_id="a", text="shared words")
        found = self.store.find_by_hash(sha256_text("shared words"))
        self.assertEqual(found["doc_id"], "a")
        self.assertIsNone(self.store.find_by_hash("0" * 64))

    def test_metadata_round_trip(self):
        row = self._add(metadata={"extension": ".txt", "size_bytes": 42})
        self.assertEqual(row["metadata"]["extension"], ".txt")
        self.assertEqual(row["metadata"]["size_bytes"], 42)

    def test_set_num_chunks(self):
        self._add()
        self.store.set_num_chunks("doc1", 7)
        self.assertEqual(self.store.get_document("doc1")["num_chunks"], 7)

    def test_list_documents(self):
        self._add(doc_id="b", text="second doc text")
        self._add(doc_id="a", text="first doc text")
        docs = self.store.list_documents()
        # same-second timestamps tie-break on doc_id, so the order is stable
        self.assertEqual([d["doc_id"] for d in docs], ["a", "b"])

    def test_delete_document(self):
        self._add()
        self.assertTrue(self.store.delete_document("doc1"))
        self.assertFalse(self.store.delete_document("doc1"))
        self.assertEqual(self.store.count(), 0)

    def test_persists_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "store.db")
            s1 = DocumentStore(db_path)
            s1.add_document("d", "T", "/p", "hello", sha256_text("hello"))
            s1.close()
            s2 = DocumentStore(db_path)
            try:
                self.assertEqual(s2.get_document("d")["content"], "hello")
            finally:
                s2.close()


class TestIngestionPipeline(unittest.TestCase):
    def _make_folder(self, tmpdir, files):
        folder = os.path.join(tmpdir, "docs")
        os.makedirs(folder)
        for name, content in files.items():
            mode = "wb" if isinstance(content, bytes) else "w"
            with open(os.path.join(folder, name), mode,
                      encoding=None if isinstance(content, bytes)
                      else "utf-8") as fh:
                fh.write(content)
        return folder

    def test_full_pipeline_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self._make_folder(tmp, {
                "cats.txt": "Cats chase mice. Cats are graceful hunters.",
                "dogs.md": "# Dogs\n\nDogs bark loudly at strangers.",
                "page.html": "<p>Parrots mimic human speech.</p>",
                "empty.txt": "   \n",
                "binary.pdf": b"%PDF fake",
            })
            store, index = DocumentStore(), Index()
            stats = ingest_folder(store, index, folder)
            self.assertIsInstance(stats, IngestStats)
            self.assertEqual(stats.files_seen, 5)
            self.assertEqual(stats.documents_added, 3)
            self.assertEqual(stats.empty_skipped, 1)
            self.assertEqual(stats.unsupported_skipped, 1)
            self.assertEqual(stats.errors, [])
            self.assertGreater(stats.chunks_added, 0)
            self.assertEqual(store.count(), 3)

            # documents are searchable through the index
            hits = index.search("mice hunters", k=3)
            self.assertTrue(hits)
            self.assertEqual(hits[0]["doc_id"], "cats.txt")
            hits = index.search("parrots speech", k=3)
            self.assertEqual(hits[0]["doc_id"], "page.html")
            store.close()

    def test_rerun_skips_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self._make_folder(tmp, {"a.txt": "Unique sentence one."})
            store, index = DocumentStore(), Index()
            first = ingest_folder(store, index, folder)
            second = ingest_folder(store, index, folder)
            self.assertEqual(first.documents_added, 1)
            self.assertEqual(second.documents_added, 0)
            self.assertEqual(second.documents_updated, 0)
            self.assertEqual(second.unchanged_skipped, 1)
            self.assertEqual(store.count(), 1)
            self.assertEqual(len(index), len([c for c in index.chunks]))
            store.close()

    def test_changed_file_is_updated(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self._make_folder(tmp, {"a.txt": "Original words."})
            store, index = DocumentStore(), Index()
            ingest_folder(store, index, folder)
            before = len(index)
            with open(os.path.join(folder, "a.txt"), "w") as fh:
                fh.write("Completely rewritten content about zebras.")
            stats = ingest_folder(store, index, folder)
            self.assertEqual(stats.documents_updated, 1)
            self.assertEqual(len(index), before)  # stale chunks replaced
            hits = index.search("zebras", k=3)
            self.assertEqual(hits[0]["doc_id"], "a.txt")
            store.close()

    def test_duplicate_content_under_new_name_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self._make_folder(tmp, {"copy.txt": "Same words here."})
            store, index = DocumentStore(), Index()
            ingest_folder(store, index, folder)
            with open(os.path.join(folder, "copy2.txt"), "w") as fh:
                fh.write("Same words here.")
            stats = ingest_folder(store, index, folder)
            self.assertEqual(stats.duplicates_skipped, 1)
            self.assertEqual(store.count(), 1)
            store.close()

    def test_ingest_file_empty_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "e.txt")
            open(path, "w").close()
            store, index = DocumentStore(), Index()
            self.assertEqual(ingest_file(store, index, path), "empty")
            self.assertEqual(store.count(), 0)
            store.close()

    def test_missing_folder_raises(self):
        store, index = DocumentStore(), Index()
        with self.assertRaises(NotADirectoryError):
            ingest_folder(store, index, "/does/not/exist")
        store.close()

    def test_recursive_walk(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self._make_folder(tmp, {"top.txt": "Top level words."})
            sub = os.path.join(folder, "sub")
            os.makedirs(sub)
            with open(os.path.join(sub, "nested.txt"), "w") as fh:
                fh.write("Nested words here.")
            store, index = DocumentStore(), Index()
            stats = ingest_folder(store, index, folder, recursive=True)
            self.assertEqual(stats.documents_added, 2)
            store.close()


class TestIndexRemoveDocument(unittest.TestCase):
    def test_remove_document(self):
        index = Index()
        index.add_document("a", "Cats chase mice every single day.")
        index.add_document("b", "Dogs bark loudly at night.")
        total = len(index)
        removed = index.remove_document("a")
        self.assertGreater(removed, 0)
        self.assertEqual(len(index), total - removed)
        self.assertEqual(index.search("cats mice"), [])
        hits = index.search("dogs bark")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["doc_id"], "b")

    def test_remove_missing_is_zero(self):
        index = Index()
        index.add_document("a", "Some text here.")
        self.assertEqual(index.remove_document("ghost"), 0)
        self.assertEqual(len(index), 1)


if __name__ == "__main__":
    unittest.main()
