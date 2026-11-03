"""Shared input validation for DocuMind (Day 13).

Every public entry point — the REST API, the CLI (via the core
functions below) and direct library use — funnels through these
helpers, so a bad value fails the same way everywhere: with a
``ValueError`` carrying a plain-language message, which the server
turns into a 400 (or 409/413 where noted) and the CLI turns into an
error exit, instead of a crash, a silent empty result, or a weird
slice of the index.

Limits are deliberately generous for real documents and strict
against abuse:

* ``k`` (retrieval cutoff): an integer, 1..100. Before this pass a
  negative ``k`` silently sliced from the end of the ranked list
  (``scored[:-1]`` drops the best hit) and ``k=0`` silently returned
  nothing, so a typo looked like "no results".
* queries and questions: non-empty after stripping, at most 2000
  characters. Before this pass a blank question returned the generic
  "could not find anything" answer, indistinguishable from a real miss.
* document ids: 1..128 characters, starting with a letter or digit,
  then letters, digits, ``.``, ``_`` or ``-`` only. Before this pass
  an id containing ``/`` could be stored via ``POST /ingest`` but
  could never be fetched again, because ``GET /documents/<id>``
  treats the slash as a path separator and answers 404.
* titles: at most 500 characters.
* document text: at most 2,000,000 characters (the 10 MB request-body
  cap in server.py still applies first to raw HTTP payloads).
"""
import re

MAX_K = 100
MIN_K = 1
MAX_QUERY_CHARS = 2000
MAX_TITLE_CHARS = 500
MAX_DOC_ID_CHARS = 128
MAX_TEXT_CHARS = 2_000_000

_DOC_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def validate_k(k):
    """Return ``k`` as an int, or raise ValueError.

    Booleans are rejected explicitly: ``True`` is an ``int`` in
    Python, but ``k=True`` meaning "top 1" would be a silent surprise.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("k must be an integer, got %r" % (k,))
    if k < MIN_K or k > MAX_K:
        raise ValueError("k must be between %d and %d, got %d"
                         % (MIN_K, MAX_K, k))
    return k


def validate_query(text, field="query"):
    """Return ``text`` stripped, or raise ValueError.

    Rejects non-strings, blank input, and absurdly long input, so
    callers never have to distinguish "empty query" from "no match".
    """
    if not isinstance(text, str):
        raise ValueError("%s must be a string, got %r" % (field, text))
    stripped = text.strip()
    if not stripped:
        raise ValueError("%s must not be empty" % field)
    if len(stripped) > MAX_QUERY_CHARS:
        raise ValueError("%s must be at most %d characters, got %d"
                         % (field, MAX_QUERY_CHARS, len(stripped)))
    return stripped


def validate_doc_id(doc_id):
    """Return ``doc_id`` stripped, or raise ValueError.

    The pattern keeps ids usable as one URL path segment (see the
    module docstring) and rules out ``..`` / path traversal, since
    ids are also used as store keys and, for file ingests, filenames.
    """
    if not isinstance(doc_id, str):
        raise ValueError("doc_id must be a string, got %r" % (doc_id,))
    stripped = doc_id.strip()
    if not stripped:
        raise ValueError("doc_id must not be empty")
    if len(stripped) > MAX_DOC_ID_CHARS:
        raise ValueError("doc_id must be at most %d characters, got %d"
                         % (MAX_DOC_ID_CHARS, len(stripped)))
    if not _DOC_ID_RE.match(stripped) or stripped in (".", ".."):
        raise ValueError(
            "doc_id %r is invalid: use letters, digits, '.', '_' or "
            "'-', starting with a letter or digit" % stripped)
    return stripped


def validate_title(title):
    """Return ``title`` stripped, or raise ValueError."""
    if not isinstance(title, str):
        raise ValueError("title must be a string, got %r" % (title,))
    stripped = title.strip()
    if len(stripped) > MAX_TITLE_CHARS:
        raise ValueError("title must be at most %d characters, got %d"
                         % (MAX_TITLE_CHARS, len(stripped)))
    return stripped


def validate_text(text):
    """Return ``text`` stripped, or raise ValueError."""
    if not isinstance(text, str):
        raise ValueError("text must be a string, got %r" % (text,))
    stripped = text.strip()
    if not stripped:
        raise ValueError("text must not be empty")
    if len(stripped) > MAX_TEXT_CHARS:
        raise ValueError("text must be at most %d characters, got %d"
                         % (MAX_TEXT_CHARS, len(stripped)))
    return stripped
