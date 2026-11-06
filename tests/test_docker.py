"""Tests for the Day 16 Docker packaging.

Covers three things, all checkable without a Docker daemon:

- The Dockerfile: a slim Python base, non-root user, the server
  exposed on port 8000 with a /health healthcheck, container-sensible
  defaults (bind 0.0.0.0, state in /data), and a CMD that starts the
  same server the CLI starts locally.
- docker-compose.yml: one service built from the Dockerfile, port
  8000 published, state persisted in the named volume mounted at
  /data, with the path environment variables pointing there.
- The config support the container relies on (Day 15 extended):
  DOCUMIND_INDEX_FILE / DOCUMIND_STORE_FILE / DOCUMIND_DATA_DIR
  environment overrides, so the index, store and data directory can
  live on a mounted volume without shipping a config file.
- .dockerignore: runtime artefacts and caches stay out of the image.

The compose file is parsed with PyYAML when it is installed; that
one test skips otherwise, and the string-level checks below it still
run. The rest of the suite is standard library only.

Run from the project root:  python3 -m unittest discover -s tests
"""
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from documind.config import (DEFAULT_CONFIG, load_config, resolve_paths)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is optional here
    yaml = None


def _read(name):
    with open(os.path.join(BASE_DIR, name), encoding="utf-8") as fh:
        return fh.read()


class TestDockerfile(unittest.TestCase):
    def setUp(self):
        self.text = _read("Dockerfile")

    def test_slim_python_base(self):
        self.assertRegex(self.text, r"(?m)^FROM python:3\.12-slim\s*$")

    def test_workdir_is_app(self):
        self.assertIn("WORKDIR /app", self.text)

    def test_runs_as_non_root_user(self):
        # A USER line naming the app user, and no later switch back
        # to root.
        users = re.findall(r"^USER\s+(\S+)\s*$", self.text, re.MULTILINE)
        self.assertTrue(users, "no USER instruction in Dockerfile")
        self.assertEqual(users[-1], "documind")
        self.assertNotIn("USER root", self.text)

    def test_data_dir_created_and_owned_by_app_user(self):
        self.assertIn("mkdir -p /data/docs", self.text)
        self.assertIn("chown -R documind:documind /data", self.text)

    def test_copies_code_data_eval_and_tests(self):
        for fragment in ("COPY --chown=documind:documind cli.py",
                         "COPY --chown=documind:documind documind/",
                         "COPY --chown=documind:documind data/",
                         "COPY --chown=documind:documind eval/",
                         "COPY --chown=documind:documind tests/"):
            self.assertIn(fragment, self.text)

    def test_container_defaults_bind_all_interfaces_on_8000(self):
        self.assertIn("DOCUMIND_HOST=0.0.0.0", self.text)
        self.assertIn("DOCUMIND_PORT=8000", self.text)

    def test_container_defaults_point_state_at_data_volume(self):
        self.assertIn("DOCUMIND_INDEX_FILE=/data/index.json", self.text)
        self.assertIn("DOCUMIND_STORE_FILE=/data/documind.db", self.text)
        self.assertIn("DOCUMIND_DATA_DIR=/data/docs", self.text)

    def test_exposes_8000(self):
        self.assertRegex(self.text, r"(?m)^EXPOSE 8000\s*$")

    def test_healthcheck_hits_health_on_the_exposed_port(self):
        match = re.search(r"^HEALTHCHECK.*$", self.text,
                          re.MULTILINE | re.DOTALL)
        self.assertIsNotNone(match, "no HEALTHCHECK in Dockerfile")
        check = match.group(0)
        self.assertIn("/health", check)
        self.assertIn("localhost:8000", check)

    def test_cmd_starts_the_cli_server(self):
        self.assertRegex(
            self.text,
            r'(?m)^CMD \["python3", "cli\.py", "serve"\]\s*$')


class TestDockerignore(unittest.TestCase):
    def setUp(self):
        self.lines = [line.strip() for line in
                      _read(".dockerignore").splitlines()
                      if line.strip() and not line.strip().startswith("#")]

    def test_excludes_runtime_artefacts(self):
        for excluded in ("index.json", "documind.db", "config.json",
                         "*.log"):
            self.assertIn(excluded, self.lines)

    def test_excludes_caches_and_vcs(self):
        for excluded in (".git", "__pycache__/", "*.py[cod]"):
            self.assertIn(excluded, self.lines)

    def test_excludes_dev_tooling(self):
        self.assertIn("bin/", self.lines)


@unittest.skipUnless(yaml is not None,
                     "PyYAML not installed; compose structure unchecked")
class TestComposeFileParsed(unittest.TestCase):
    def setUp(self):
        self.compose = yaml.safe_load(_read("docker-compose.yml"))

    def test_one_documind_service_built_from_dockerfile(self):
        service = self.compose["services"]["documind"]
        self.assertEqual(service["build"], ".")
        self.assertEqual(service["image"], "documind:latest")

    def test_port_8000_published(self):
        service = self.compose["services"]["documind"]
        self.assertIn("8000:8000", [str(p) for p in service["ports"]])

    def test_state_volume_mounted_at_data(self):
        service = self.compose["services"]["documind"]
        self.assertIn("documind-data:/data",
                      [str(v) for v in service["volumes"]])
        self.assertIn("documind-data", self.compose["volumes"])

    def test_environment_points_state_at_the_volume(self):
        env = self.compose["services"]["documind"]["environment"]
        self.assertEqual(str(env["DOCUMIND_HOST"]), "0.0.0.0")
        self.assertEqual(str(env["DOCUMIND_PORT"]), "8000")
        self.assertEqual(env["DOCUMIND_INDEX_FILE"], "/data/index.json")
        self.assertEqual(env["DOCUMIND_STORE_FILE"], "/data/documind.db")
        self.assertEqual(env["DOCUMIND_DATA_DIR"], "/data/docs")


class TestComposeFileText(unittest.TestCase):
    """String-level checks that run even without PyYAML."""

    def setUp(self):
        self.text = _read("docker-compose.yml")

    def test_service_and_volume_present(self):
        self.assertIn("documind:", self.text)
        self.assertIn("documind-data:/data", self.text)
        self.assertIn("8000:8000", self.text)

    def test_restart_policy_keeps_it_up(self):
        self.assertIn("restart: unless-stopped", self.text)


class TestConfigPathEnvOverrides(unittest.TestCase):
    """DOCUMIND_* path overrides (new in Day 16)."""

    def test_path_overrides_land_in_config_and_resolve_absolute(self):
        env = {"DOCUMIND_INDEX_FILE": "/data/index.json",
               "DOCUMIND_STORE_FILE": "/data/documind.db",
               "DOCUMIND_DATA_DIR": "/data/docs"}
        config = load_config(base_dir=BASE_DIR, env=env)
        self.assertEqual(config["paths"]["index_file"], "/data/index.json")
        self.assertEqual(config["paths"]["store_file"], "/data/documind.db")
        self.assertEqual(config["paths"]["data_dir"], "/data/docs")
        paths = resolve_paths(config, BASE_DIR)
        self.assertEqual(paths["index_path"], "/data/index.json")
        self.assertEqual(paths["store_path"], "/data/documind.db")
        self.assertEqual(paths["data_dir"], "/data/docs")

    def test_relative_path_override_resolves_against_base_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"DOCUMIND_INDEX_FILE": "state/index.json"}
            config = load_config(base_dir=tmp, env=env)
            paths = resolve_paths(config, tmp)
            self.assertEqual(paths["index_path"],
                             os.path.join(tmp, "state", "index.json"))

    def test_empty_path_override_is_ignored(self):
        env = {"DOCUMIND_INDEX_FILE": ""}
        config = load_config(base_dir=BASE_DIR, env=env)
        self.assertEqual(config["paths"]["index_file"],
                         DEFAULT_CONFIG["paths"]["index_file"])

    def test_blank_path_override_is_rejected(self):
        env = {"DOCUMIND_INDEX_FILE": "   "}
        with self.assertRaises(ValueError):
            load_config(base_dir=BASE_DIR, env=env)

    def test_path_override_beats_config_file(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"paths": {"index_file": "from-file.json"}}, fh)
            env = {"DOCUMIND_INDEX_FILE": "/data/index.json"}
            config = load_config(path=path, base_dir=tmp, env=env)
            self.assertEqual(config["paths"]["index_file"],
                             "/data/index.json")

    def test_other_values_unchanged_by_path_overrides(self):
        env = {"DOCUMIND_INDEX_FILE": "/data/index.json"}
        config = load_config(base_dir=BASE_DIR, env=env)
        self.assertEqual(config["server"], DEFAULT_CONFIG["server"])
        self.assertEqual(config["retrieval"], DEFAULT_CONFIG["retrieval"])
        self.assertEqual(config["logging"], DEFAULT_CONFIG["logging"])


class TestDockerPortConsistency(unittest.TestCase):
    """The port is 8000 in the Dockerfile, compose and the defaults."""

    def test_expose_compose_and_default_port_agree(self):
        dockerfile = _read("Dockerfile")
        exposed = re.search(r"^EXPOSE (\d+)\s*$", dockerfile, re.MULTILINE)
        self.assertIsNotNone(exposed)
        port = exposed.group(1)
        self.assertIn('"%s:%s"' % (port, port),
                      _read("docker-compose.yml"))
        self.assertIn("DOCUMIND_PORT=%s" % port, dockerfile)
        self.assertEqual(DEFAULT_CONFIG["server"]["port"], int(port))


if __name__ == "__main__":
    unittest.main()
