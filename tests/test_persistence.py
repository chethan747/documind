"""Tests for Day 4: persistent index save/load + document metadata.

Run from the project root:  python3 -m unittest discover -s tests
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from documind.index import Index, INDEX_FORMAT_VERSION
from documind.ingest import ingest_file
from documind.store import DocumentStore


def _sample_index():
    index = Index()
    index.add_document("guide.md",
                       "AEM is built on Apache Sling and the Java Content "
                       "Repository. Sling maps HTTP requests to JCR nodes.",
                       metadata={"title": "AEM Guide",
                                 "source_path": "/docs/guide.md"})
    index.add_document("spring.md",
                       "Spring Boot auto-configuration sets up beans from "
                       "the classpath. A Spring Boot application starts an "
                       "embedded server.",
                       metadata={"title": "Spring Notes",
                                 "source_path": "/docs/spring.md"})
    return index


class TestIndexPersistence(unittest.TestCase):
    def test_round_trip_preserves_chunks_metadata_and_ranking(self):
        index = _sample_index()
        before = {m: index.search("AEM Sling", method=m) for m in
                  ("tfidf", "bm25", "combined")}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "index.json")
            index.save(path)
            loaded = Index.load(path)
        self.assertEqual(loaded.chunks, index.chunks)
        self.assertEqual(loaded.documents, index.documents)
        self.assertEqual(loaded.chunk_size, index.chunk_size)
        self.assertEqual(loaded.overlap, index.overlap)
        for method, hits in before.items():
            after = loaded.search("AEM Sling", method=method)
            self.assertEqual(
                [(h["doc_id"], h["chunk_index"], h["score"]) for h in after],
                [(h["doc_id"], h["chunk_index"], h["score"]) for h in hits])

    def test_saved_file_is_versioned(self):
        index = _sample_index()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "index.json")
            index.save(path)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        self.assertEqual(data["format_version"], INDEX_FORMAT_VERSION)
        self.assertEqual(data["documents"]["guide.md"]["title"], "AEM Guide")

    def test_search_hits_carry_document_metadata(self):
        hits = _sample_index().search("Spring Boot auto-configuration")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["metadata"]["title"], "Spring Notes")
        self.assertEqual(hits[0]["metadata"]["source_path"],
                         "/docs/spring.md")

    def test_documents_without_metadata_get_empty_metadata(self):
        index = Index()
        index.add_document("plain.txt", "Some text about databases.")
        hits = index.search("databases")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["metadata"], {})

    def test_remove_document_drops_metadata(self):
        index = _sample_index()
        self.assertEqual(index.remove_document("guide.md"), 1)
        self.assertNotIn("guide.md", index.documents)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "index.json")
            index.save(path)
            loaded = Index.load(path)
        self.assertNotIn("guide.md", loaded.documents)

    def test_load_pre_day4_index_still_works(self):
        """A Day 3 save (no version, no documents map) must load cleanly."""
        data = _sample_index().to_dict()
        del data["format_version"]
        del data["documents"]
        loaded = Index.from_dict(data)
        self.assertEqual(loaded.documents, {})
        hits = loaded.search("AEM Sling")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["metadata"], {})

    def test_load_rejects_corrupt_and_unsupported_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_json = os.path.join(tmp, "bad.json")
            with open(bad_json, "w", encoding="utf-8") as fh:
                fh.write("{not json")
            with self.assertRaises(ValueError):
                Index.load(bad_json)

            future = os.path.join(tmp, "future.json")
            data = _sample_index().to_dict()
            data["format_version"] = INDEX_FORMAT_VERSION + 5
            with open(future, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            with self.assertRaises(ValueError):
                Index.load(future)

            missing = os.path.join(tmp, "missing.json")
            with open(missing, "w", encoding="utf-8") as fh:
                json.dump({"format_version": INDEX_FORMAT_VERSION}, fh)
            with self.assertRaises(ValueError):
                Index.load(missing)

    def test_load_rejects_non_object_data(self):
        with self.assertRaises(ValueError):
            Index.from_dict(["not", "an", "index"])


class TestIngestMetadata(unittest.TestCase):
    def test_ingest_file_stores_metadata_in_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "my-notes.md")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("# Notes\n\nAEM runs on Apache Sling.\n")
            store = DocumentStore()
            index = Index()
            try:
                self.assertEqual(ingest_file(store, index, path), "added")
            finally:
                store.close()
            meta = index.documents["my-notes.md"]
            self.assertEqual(meta["title"], "my notes")
            self.assertEqual(meta["source_path"], path)
            self.assertEqual(meta["extension"], ".md")
            hits = index.search("AEM Sling")
            self.assertTrue(hits)
            self.assertEqual(hits[0]["metadata"]["title"], "my notes")

    def test_reingest_changed_file_refreshes_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "notes.md")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("First version about AEM and Sling.\n")
            store = DocumentStore()
            index = Index()
            try:
                ingest_file(store, index, path)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("Second version about Spring Boot beans.\n")
                self.assertEqual(ingest_file(store, index, path), "updated")
            finally:
                store.close()
            hits = index.search("Spring Boot beans")
            self.assertTrue(hits)
            self.assertEqual(hits[0]["doc_id"], "notes.md")


if __name__ == "__main__":
    unittest.main()
