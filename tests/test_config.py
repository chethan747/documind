"""Tests for the Day 15 config file (documind/config.py).

Covers the built-in defaults, loading and merging a config file,
environment-variable overrides, validation of every value, and path
resolution. ``env={}`` is passed throughout so the real environment
cannot leak into a test.

Run from the project root:  python3 -m unittest discover -s tests
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from documind.config import (DEFAULT_CONFIG, load_config, resolve_paths)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_config(directory, payload):
    path = os.path.join(directory, "config.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


class TestDefaults(unittest.TestCase):
    def test_no_file_means_builtin_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(base_dir=tmp, env={})
        self.assertEqual(config, DEFAULT_CONFIG)
        self.assertEqual(config["server"], {"host": "127.0.0.1",
                                            "port": 8000})
        self.assertEqual(config["retrieval"]["default_method"], "tfidf")
        self.assertEqual(config["logging"],
                         {"level": "INFO", "file": None, "json": True})

    def test_defaults_are_not_mutated_by_loading(self):
        before = json.dumps(DEFAULT_CONFIG, sort_keys=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(tmp, {"server": {"port": 9000}})
            load_config(path=path, env={})
        self.assertEqual(json.dumps(DEFAULT_CONFIG, sort_keys=True), before)
        self.assertEqual(DEFAULT_CONFIG["server"]["port"], 8000)

    def test_shipped_example_matches_defaults(self):
        example = os.path.join(BASE_DIR, "config.example.json")
        config = load_config(path=example, env={})
        self.assertEqual(config, DEFAULT_CONFIG)


class TestFileLoading(unittest.TestCase):
    def test_partial_file_merges_onto_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(tmp, {"server": {"port": 9000},
                                       "logging": {"level": "DEBUG"}})
            config = load_config(path=path, env={})
        self.assertEqual(config["server"]["port"], 9000)
        self.assertEqual(config["server"]["host"], "127.0.0.1")
        self.assertEqual(config["logging"]["level"], "DEBUG")
        self.assertEqual(config["retrieval"]["search_k"], 5)

    def test_config_json_in_base_dir_is_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_config(tmp, {"retrieval": {"default_method": "bm25"}})
            config = load_config(base_dir=tmp, env={})
        self.assertEqual(config["retrieval"]["default_method"], "bm25")

    def test_explicit_missing_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_config(path=os.path.join(tmp, "nope.json"), env={})

    def test_malformed_json_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{not json")
            with self.assertRaises(ValueError):
                load_config(path=path, env={})

    def test_non_object_json_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "list.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("[1, 2, 3]")
            with self.assertRaises(ValueError):
                load_config(path=path, env={})


class TestValidation(unittest.TestCase):
    def _bad(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(tmp, payload)
            with self.assertRaises(ValueError, msg="%r" % (payload,)):
                load_config(path=path, env={})

    def test_unknown_section_and_key_are_rejected(self):
        self._bad({"surprise": {}})
        self._bad({"server": {"prot": 8000}})
        self._bad({"server": "not-a-dict"})

    def test_bad_port(self):
        self._bad({"server": {"port": 0}})
        self._bad({"server": {"port": 65536}})
        self._bad({"server": {"port": "8000"}})
        self._bad({"server": {"port": True}})

    def test_bad_host(self):
        self._bad({"server": {"host": ""}})
        self._bad({"server": {"host": 42}})

    def test_bad_method_and_k(self):
        self._bad({"retrieval": {"default_method": "neural"}})
        self._bad({"retrieval": {"search_k": 0}})
        self._bad({"retrieval": {"search_k": 101}})
        self._bad({"retrieval": {"ask_k": "3"}})

    def test_bad_chunk_settings(self):
        self._bad({"retrieval": {"chunk_size": 0}})
        self._bad({"retrieval": {"chunk_overlap": -1}})
        self._bad({"retrieval": {"chunk_size": 100,
                                 "chunk_overlap": 100}})

    def test_bad_logging_settings(self):
        self._bad({"logging": {"level": "VERBOSE"}})
        self._bad({"logging": {"level": "info"}})  # must be upper-case
        self._bad({"logging": {"json": "yes"}})
        self._bad({"logging": {"file": 42}})


class TestEnvOverrides(unittest.TestCase):
    def test_host_port_level_and_file(self):
        env = {"DOCUMIND_HOST": "0.0.0.0", "DOCUMIND_PORT": "9001",
               "DOCUMIND_LOG_LEVEL": "debug",
               "DOCUMIND_LOG_FILE": "app.log"}
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(base_dir=tmp, env=env)
        self.assertEqual(config["server"]["host"], "0.0.0.0")
        self.assertEqual(config["server"]["port"], 9001)
        self.assertEqual(config["logging"]["level"], "DEBUG")
        self.assertEqual(config["logging"]["file"], "app.log")

    def test_env_wins_over_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(tmp, {"server": {"port": 9000}})
            config = load_config(path=path,
                                 env={"DOCUMIND_PORT": "7000"})
        self.assertEqual(config["server"]["port"], 7000)

    def test_bad_env_port_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_config(base_dir=tmp,
                            env={"DOCUMIND_PORT": "not-a-port"})

    def test_documind_config_env_points_at_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(tmp, {"server": {"port": 9123}})
            config = load_config(base_dir=tmp,
                                 env={"DOCUMIND_CONFIG": path})
        self.assertEqual(config["server"]["port"], 9123)

    def test_empty_env_values_are_ignored(self):
        env = {"DOCUMIND_HOST": "", "DOCUMIND_PORT": ""}
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(base_dir=tmp, env=env)
        self.assertEqual(config["server"], DEFAULT_CONFIG["server"])


class TestResolvePaths(unittest.TestCase):
    def test_relative_paths_resolve_against_base_dir(self):
        config = load_config(base_dir="/nowhere", env={})
        paths = resolve_paths(config, "/some/project")
        self.assertEqual(paths["index_path"], "/some/project/index.json")
        self.assertEqual(paths["store_path"], "/some/project/documind.db")
        self.assertEqual(paths["data_dir"], "/some/project/data")

    def test_absolute_paths_pass_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(tmp, {"paths": {
                "index_file": "/var/lib/documind/index.json",
                "store_file": "/var/lib/documind/store.db",
                "data_dir": "/srv/docs"}})
            config = load_config(path=path, env={})
        paths = resolve_paths(config, "/some/project")
        self.assertEqual(paths["index_path"],
                         "/var/lib/documind/index.json")
        self.assertEqual(paths["store_path"], "/var/lib/documind/store.db")
        self.assertEqual(paths["data_dir"], "/srv/docs")


if __name__ == "__main__":
    unittest.main()
