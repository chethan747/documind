"""Tests for Day 15 structured logging (documind/logging_config.py),
the server's JSON access log, and the CLI's config command.

The formatter tests drive the formatter directly; the server tests
attach a capturing handler to the ``documind.server`` logger and make
real HTTP requests, exactly like tests/test_server.py does.

Run from the project root:  python3 -m unittest discover -s tests
"""
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from documind.ingest import ingest_folder  # noqa: E402
from documind.index import Index  # noqa: E402
from documind.logging_config import (JsonFormatter, PlainFormatter,
                                     configure_from_config, get_logger,
                                     setup_logging)  # noqa: E402
from documind.server import create_server  # noqa: E402
from documind.store import DocumentStore  # noqa: E402
from documind.config import load_config  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _request(method, url, payload=None):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class _CaptureHandler(logging.Handler):
    """Collect formatted log lines in a list."""

    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(self.format(record))


class TestJsonFormatter(unittest.TestCase):
    def _format(self, level, message, **extra):
        record = logging.LogRecord("documind.test", level, __file__, 1,
                                   message, None, None)
        record.__dict__.update(extra)
        return json.loads(JsonFormatter().format(record))

    def test_line_is_one_json_object_with_required_fields(self):
        payload = self._format(logging.INFO, "hello",
                               event="greeting", answer=42)
        self.assertEqual(payload["level"], "INFO")
        self.assertEqual(payload["logger"], "documind.test")
        self.assertEqual(payload["message"], "hello")
        self.assertEqual(payload["event"], "greeting")
        self.assertEqual(payload["answer"], 42)
        self.assertIn("T", payload["ts"])  # ISO timestamp

    def test_non_serialisable_extra_is_stringified_not_lost(self):
        payload = self._format(logging.INFO, "hello",
                               weird=object())
        self.assertIn("weird", payload)
        self.assertIsInstance(payload["weird"], str)

    def test_plain_formatter_is_not_json(self):
        record = logging.LogRecord("documind.test", logging.INFO,
                                   __file__, 1, "hello", None, None)
        line = PlainFormatter().format(record)
        self.assertIn("INFO", line)
        self.assertIn("hello", line)
        with self.assertRaises(ValueError):
            json.loads(line)


class TestSetupLogging(unittest.TestCase):
    def test_json_lines_written_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "app.log")
            logger = setup_logging(level="INFO", log_file=path)
            try:
                logger.info("file line", extra={"event": "test"})
                logger.debug("suppressed by level")
            finally:
                for handler in list(logger.handlers):
                    handler.close()
                logger.handlers = []
            with open(path, encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["message"], "file line")
        self.assertEqual(lines[0]["event"], "test")

    def test_level_filtering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "app.log")
            logger = setup_logging(level="WARNING", log_file=path)
            try:
                logger.info("too quiet")
                logger.warning("loud enough")
            finally:
                for handler in list(logger.handlers):
                    handler.close()
                logger.handlers = []
            with open(path, encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh if line.strip()]
        self.assertEqual([line["level"] for line in lines], ["WARNING"])

    def test_configure_from_config(self):
        config = load_config(base_dir=BASE_DIR, env={})
        logger = configure_from_config(config)
        self.assertEqual(logger.name, "documind")
        self.assertEqual(logger.level, logging.INFO)
        logger.handlers = []

    def test_get_logger_is_a_documind_child(self):
        self.assertEqual(get_logger("server").name, "documind.server")


class TestCliConfigCommand(unittest.TestCase):
    def test_config_command_prints_effective_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "custom.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"server": {"port": 9123},
                           "logging": {"level": "WARNING"}}, fh)
            proc = subprocess.run(
                [sys.executable, os.path.join(BASE_DIR, "cli.py"),
                 "--config", path, "config"],
                capture_output=True, text=True, cwd=BASE_DIR)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        config = json.loads(proc.stdout)
        self.assertEqual(config["server"]["port"], 9123)
        self.assertEqual(config["logging"]["level"], "WARNING")
        self.assertEqual(config["server"]["host"], "127.0.0.1")

    def test_bad_config_file_exits_2_with_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write('{"server": {"port": 0}}')
            proc = subprocess.run(
                [sys.executable, os.path.join(BASE_DIR, "cli.py"),
                 "--config", path, "config"],
                capture_output=True, text=True, cwd=BASE_DIR)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("error:", proc.stderr)


class TestServerAccessLog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        store = DocumentStore(os.path.join(self.tmp.name, "documind.db"))
        index = Index()
        ingest_folder(store, index, DATA_DIR)
        store.close()
        index.save(os.path.join(self.tmp.name, "index.json"))
        self.server = create_server(self.tmp.name, "127.0.0.1", 0,
                                    config=load_config(
                                        base_dir=self.tmp.name, env={}))
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base = "http://%s:%d" % (host, port)
        # capture the structured log lines the server emits
        self.capture = _CaptureHandler()
        self.capture.setFormatter(JsonFormatter())
        logger = get_logger("server")
        logger.addHandler(self.capture)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        self.logger = logger

    def tearDown(self):
        self.logger.removeHandler(self.capture)
        self.logger.propagate = True
        self.server.shutdown()
        self.server.server_close()
        self.server.documind_app.close()
        self.thread.join(timeout=5)
        self.tmp.cleanup()

    def _events(self, name):
        return [json.loads(line) for line in self.capture.lines
                if json.loads(line).get("event") == name]

    def test_health_request_logs_structured_access_line(self):
        status, _ = _request("GET", self.base + "/health")
        self.assertEqual(status, 200)
        events = self._events("http_access")
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["method"], "GET")
        self.assertEqual(event["path"], "/health")
        self.assertEqual(event["status"], 200)
        self.assertIsInstance(event["duration_ms"], (int, float))
        self.assertGreaterEqual(event["duration_ms"], 0)

    def test_query_string_is_not_logged(self):
        status, _ = _request("GET",
                             self.base + "/search?query=topsecretquestion")
        self.assertEqual(status, 200)
        events = self._events("http_access")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["path"], "/search")
        self.assertNotIn("topsecretquestion",
                         "\n".join(self.capture.lines))

    def test_ingest_and_error_requests_are_logged(self):
        status, _ = _request("POST", self.base + "/ingest",
                             {"doc_id": "log-test",
                              "text": "Structured logging records "
                                      "events as JSON lines."})
        self.assertEqual(status, 201)
        ingest = self._events("ingest")
        self.assertEqual(len(ingest), 1)
        self.assertEqual(ingest[0]["doc_id"], "log-test")
        self.assertEqual(ingest[0]["outcome"], "added")
        status, _ = _request("GET", self.base + "/nope")
        self.assertEqual(status, 404)
        access = self._events("http_access")
        self.assertEqual([e["status"] for e in access], [201, 404])


if __name__ == "__main__":
    unittest.main()
