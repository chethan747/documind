"""SQLite document store for DocuMind.

The store is the system of record for *documents*: their source text, where
they came from, when they were ingested, how many index chunks they produced,
and arbitrary metadata. The search index (index.py) derives its chunks from
these records and carries a copy of each document's metadata so a saved
index alone can report where its hits came from (Day 4).

Deduplication is by SHA-256 of the extracted text: identical content is only
stored once, no matter how many files or paths it arrives through.
"""
import json
import sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    source_path  TEXT,
    content      TEXT NOT NULL,
    text_hash    TEXT NOT NULL,
    num_chunks   INTEGER NOT NULL DEFAULT 0,
    metadata     TEXT NOT NULL DEFAULT '{}',
    ingested_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents (text_hash);
"""


def _row_to_dict(row):
    doc = dict(row)
    doc["metadata"] = json.loads(doc["metadata"])
    return doc


def utcnow_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DocumentStore:
    """A small SQLite-backed registry of ingested documents."""

    def __init__(self, path=":memory:", check_same_thread=True):
        self.path = path
        self.conn = sqlite3.connect(path,
                                    check_same_thread=check_same_thread)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # -- writes --------------------------------------------------------
    def add_document(self, doc_id, title, source_path, content,
                     text_hash, metadata=None):
        """Insert or update the document identified by ``doc_id``.

        Same ``doc_id`` + same ``text_hash`` is a no-op (returns the row).
        Same ``doc_id`` + different ``text_hash`` refreshes the stored
        content and resets the chunk count. Returns the stored row as a
        dict, with ``metadata`` decoded.
        """
        existing = self.get_document(doc_id)
        if existing is not None and existing["text_hash"] == text_hash:
            return existing
        meta_json = json.dumps(metadata or {}, sort_keys=True)
        now = utcnow_iso()
        if existing is None:
            self.conn.execute(
                "INSERT INTO documents (doc_id, title, source_path, content,"
                " text_hash, metadata, ingested_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_id, title, source_path, content, text_hash,
                 meta_json, now))
        else:
            self.conn.execute(
                "UPDATE documents SET title = ?, source_path = ?, content = ?,"
                " text_hash = ?, num_chunks = 0, metadata = ?, ingested_at = ?"
                " WHERE doc_id = ?",
                (title, source_path, content, text_hash, meta_json,
                 now, doc_id))
        self.conn.commit()
        return self.get_document(doc_id)

    def set_num_chunks(self, doc_id, num_chunks):
        """Record how many index chunks a document produced."""
        self.conn.execute("UPDATE documents SET num_chunks = ?"
                          " WHERE doc_id = ?",
                          (int(num_chunks), doc_id))
        self.conn.commit()

    def delete_document(self, doc_id):
        """Delete ``doc_id``. Returns True when a row was removed."""
        cur = self.conn.execute("DELETE FROM documents WHERE doc_id = ?",
                                (doc_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # -- reads ---------------------------------------------------------
    def get_document(self, doc_id):
        """Return the stored row for ``doc_id`` as a dict, or None."""
        cur = self.conn.execute("SELECT * FROM documents WHERE doc_id = ?",
                                (doc_id,))
        row = cur.fetchone()
        return _row_to_dict(row) if row else None

    def find_by_hash(self, text_hash):
        """Return the stored row whose content hashes to ``text_hash``."""
        cur = self.conn.execute(
            "SELECT * FROM documents WHERE text_hash = ? LIMIT 1",
            (text_hash,))
        row = cur.fetchone()
        return _row_to_dict(row) if row else None

    def list_documents(self):
        """All stored documents, oldest first, as a list of dicts."""
        cur = self.conn.execute(
            "SELECT * FROM documents ORDER BY ingested_at, doc_id")
        return [_row_to_dict(row) for row in cur.fetchall()]

    def count(self):
        cur = self.conn.execute("SELECT COUNT(*) AS n FROM documents")
        return cur.fetchone()["n"]

    # -- lifecycle -----------------------------------------------------
    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
