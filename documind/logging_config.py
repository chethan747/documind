"""Structured logging for DocuMind (Day 15).

Every log line is one JSON object, so logs are readable by a human
*and* parseable by a machine without a regex: each record carries a
UTC timestamp, the level, the logger name, the message, and any
structured fields the call site attached with ``extra={...}`` — for
example the server's access log attaches ``event="http_access"`` plus
``method``, ``path``, ``status`` and ``duration_ms``, and the ingest
path attaches ``event="ingest"`` plus ``doc_id`` and ``outcome``.

    {"ts": "2026-11-05T15:50:01.123456+00:00", "level": "INFO",
     "logger": "documind.server", "message": "http request",
     "event": "http_access", "method": "GET", "path": "/health",
     "status": 200, "duration_ms": 1.2}

Log to stderr by default; set ``logging.file`` in the config (or
``DOCUMIND_LOG_FILE``) to also append to a file. ``logging.json:
false`` switches the console/file output back to plain text for
humans who prefer it. Levels come from ``logging.level`` in the
config (``DOCUMIND_LOG_LEVEL`` overrides).

Usage::

    from documind.logging_config import get_logger, setup_logging

    setup_logging(level="INFO")            # once, at startup
    log = get_logger("server")
    log.info("server started", extra={"event": "startup", "port": 8000})

Call sites pass structured fields through ``extra``; the JSON
formatter below picks them up automatically. Two rules, enforced by
the formatter: the reserved ``logging`` attribute names (``level``,
``message`` and friends) are never overwritten, and any extra value
that is not JSON-serialisable is rendered with ``str()`` instead of
crashing the log call.
"""
import json
import logging
import sys
from datetime import datetime, timezone

LOGGER_NAME = "documind"

# Attributes every LogRecord already has; anything else in
# ``record.__dict__`` was attached by the call site via ``extra`` and
# is emitted as a structured field.
_RESERVED = frozenset((
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
))


class JsonFormatter(logging.Formatter):
    """Format a LogRecord as a single-line JSON object."""

    def format(self, record):
        payload = {
            "ts": datetime.fromtimestamp(record.created,
                                         tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in sorted(record.__dict__.items()):
            if key in _RESERVED or key in payload:
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                value = str(value)
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class PlainFormatter(logging.Formatter):
    """The human-readable fallback when ``logging.json`` is false."""

    def __init__(self):
        super().__init__("%(asctime)s %(levelname)-8s %(name)s %(message)s",
                         datefmt="%Y-%m-%dT%H:%M:%S")


def setup_logging(level="INFO", log_file=None, json_format=True):
    """Configure the ``documind`` logger and return it.

    Safe to call more than once: previous handlers are removed first,
    so reconfiguration (e.g. a test, or a server restart) never
    duplicates lines. Logs go to stderr, plus ``log_file`` when given.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers = []
    logger.propagate = False
    formatter = JsonFormatter() if json_format else PlainFormatter()
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file:
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def configure_from_config(config):
    """Configure logging from a loaded config dict (see config.py)."""
    logging_cfg = config["logging"]
    return setup_logging(level=logging_cfg["level"],
                         log_file=logging_cfg["file"],
                         json_format=logging_cfg["json"])


def get_logger(name):
    """Return a child logger, e.g. ``get_logger("server")``."""
    return logging.getLogger("%s.%s" % (LOGGER_NAME, name))
