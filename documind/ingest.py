"""Document ingestion pipeline for DocuMind.

A document enters through three stages:

1. **Discovery** — walk a folder (recursively) and pick up supported files.
2. **Extraction** — read the raw bytes, decode the text, and strip markup
   for formats that carry it (Markdown, HTML).
3. **Registration** — hash the text, skip duplicates and empties, store the
   document record in the SQLite ``DocumentStore``, and feed its chunks into
   the search ``Index``.

Every run returns an ``IngestStats`` describing what happened, so the CLI
(and tests) can report it faithfully.
"""
import hashlib
import html as html_module
import os
import re

from .index import Index
from .store import DocumentStore

SUPPORTED_EXTENSIONS = (".txt", ".md", ".html")

_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


class UnsupportedFormatError(ValueError):
    """Raised when a file's extension is not a supported document format."""


def sha256_text(text):
    """Hex SHA-256 of ``text`` — the document's deduplication key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _decode(raw_bytes):
    """Decode bytes, tolerating non-UTF-8 files instead of crashing."""
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1")


def strip_html(text):
    """Remove scripts, styles and tags from HTML, leaving readable text."""
    text = _TAG_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = html_module.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _title_from_path(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[_-]+", " ", stem).strip() or stem


def extract_text(path):
    """Extract (title, text) from a document file.

    Raises UnsupportedFormatError for unknown extensions and
    FileNotFoundError / OSError for unreadable files.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            "unsupported extension %r (supported: %s)"
            % (ext, ", ".join(SUPPORTED_EXTENSIONS)))
    with open(path, "rb") as fh:
        raw = fh.read()
    text = _decode(raw)
    if ext == ".html":
        text = strip_html(text)
    return _title_from_path(path), text.strip()


class IngestStats:
    """Counters describing one ingestion run."""

    def __init__(self):
        self.files_seen = 0
        self.documents_added = 0
        self.documents_updated = 0
        self.unchanged_skipped = 0
        self.duplicates_skipped = 0
        self.empty_skipped = 0
        self.unsupported_skipped = 0
        self.errors = []          # [(path, message)]
        self.chunks_added = 0
        self.files = []           # [{"path", "outcome", "detail"}]

    def record(self, path, outcome, detail=""):
        self.files.append({"path": path, "outcome": outcome,
                           "detail": detail})

    def summary(self):
        parts = ["%d file(s) seen" % self.files_seen,
                 "%d document(s) added" % self.documents_added]
        if self.documents_updated:
            parts.append("%d updated" % self.documents_updated)
        if self.unchanged_skipped:
            parts.append("%d unchanged skipped" % self.unchanged_skipped)
        if self.duplicates_skipped:
            parts.append("%d duplicate(s) skipped" % self.duplicates_skipped)
        if self.empty_skipped:
            parts.append("%d empty skipped" % self.empty_skipped)
        if self.unsupported_skipped:
            parts.append("%d unsupported skipped" % self.unsupported_skipped)
        if self.errors:
            parts.append("%d error(s)" % len(self.errors))
        parts.append("%d chunk(s) indexed" % self.chunks_added)
        return ", ".join(parts)


def ingest_file(store, index, path, doc_id=None):
    """Run one file through the pipeline.

    Returns one of "added", "updated", "duplicate", "empty".
    Raises UnsupportedFormatError for unknown extensions and
    OSError for unreadable files.
    """
    doc_id = doc_id or os.path.basename(path)
    title, text = extract_text(path)
    if not text:
        return "empty"
    text_hash = sha256_text(text)

    existing = store.get_document(doc_id)
    if existing is not None and existing["text_hash"] == text_hash:
        # byte-identical re-ingest: nothing to do
        return "unchanged"

    dup = store.find_by_hash(text_hash)
    if dup is not None and dup["doc_id"] != doc_id:
        # identical content already stored under another id: skip it
        return "duplicate"
    ext = os.path.splitext(path)[1].lower()
    metadata = {"source": "file",
                "extension": ext,
                "size_bytes": os.path.getsize(path)}
    store.add_document(doc_id, title, path, text, text_hash,
                       metadata=metadata)

    # drop stale chunks so re-ingesting a changed file stays consistent
    if hasattr(index, "remove_document"):
        index.remove_document(doc_id)
    index_metadata = dict(metadata)
    index_metadata.update({"title": title, "source_path": path})
    num_chunks = index.add_document(doc_id, text, metadata=index_metadata)
    store.set_num_chunks(doc_id, num_chunks)

    return "updated" if existing is not None else "added"


def ingest_folder(store, index, folder, recursive=True):
    """Ingest every supported file under ``folder``; return IngestStats."""
    if not os.path.isdir(folder):
        raise NotADirectoryError("not a folder: %s" % folder)
    stats = IngestStats()
    paths = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for fname in files:
                paths.append(os.path.join(root, fname))
    else:
        paths = [os.path.join(folder, f) for f in os.listdir(folder)]
    paths.sort()

    for path in paths:
        stats.files_seen += 1
        try:
            outcome = ingest_file(store, index, path)
        except UnsupportedFormatError as exc:
            stats.unsupported_skipped += 1
            stats.record(path, "unsupported", str(exc))
            continue
        except OSError as exc:
            stats.errors.append((path, str(exc)))
            stats.record(path, "error", str(exc))
            continue
        if outcome == "added":
            stats.documents_added += 1
        elif outcome == "updated":
            stats.documents_updated += 1
        elif outcome == "unchanged":
            stats.unchanged_skipped += 1
        elif outcome == "duplicate":
            stats.duplicates_skipped += 1
        elif outcome == "empty":
            stats.empty_skipped += 1
        if outcome in ("added", "updated"):
            row = store.get_document(os.path.basename(path))
            stats.chunks_added += row["num_chunks"] if row else 0
        stats.record(path, outcome)
    return stats
