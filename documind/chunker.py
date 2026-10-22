"""Text chunking for DocuMind.

Retrieval works on small, focused passages rather than whole documents, so
each document is split into overlapping chunks. Sentences are packed greedily
into chunks of at most ``chunk_size`` characters; the tail of each chunk is
repeated at the start of the next one (the overlap), so an idea that straddles
a boundary is still retrievable from both sides.
"""
import re

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    """Split text into sentences on ., ! and ? boundaries."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _hard_split(sentence, chunk_size, overlap):
    """Split one over-long sentence into overlapping character windows."""
    pieces = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(sentence), step):
        pieces.append(sentence[start:start + chunk_size])
        if start + chunk_size >= len(sentence):
            break
    return pieces


def chunk_text(text, chunk_size=500, overlap=80):
    """Split ``text`` into a list of overlapping chunk strings.

    - Empty / whitespace-only text yields no chunks.
    - Sentences longer than ``chunk_size`` are window-split with overlap.
    - Consecutive chunks share roughly ``overlap`` characters of context.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    chunks = []
    current = ""
    for sentence in split_sentences(text):
        if len(sentence) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(sentence, chunk_size, overlap))
            continue
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            # carry the tail of the finished chunk into the next one
            tail = current[-overlap:] if overlap else ""
            tail = tail.split(" ", 1)[-1] if " " in tail else tail
            current = (tail + " " + sentence).strip() if tail else sentence
    if current:
        chunks.append(current)
    return chunks
