"""Configuration for DocuMind (Day 15).

Everything DocuMind used to hard-code — the server host and port, the
index/store file locations, the default ranking method and retrieval
cutoffs, the chunk size, and the log level — now lives in one place and
can be changed without editing code.

Precedence, highest first:

1. Environment variables — ``DOCUMIND_HOST``, ``DOCUMIND_PORT``,
   ``DOCUMIND_LOG_LEVEL``, ``DOCUMIND_LOG_FILE``.
2. A config file — the path passed to :func:`load_config`, else the
   ``DOCUMIND_CONFIG`` environment variable, else ``config.json`` in the
   project directory when it exists.
3. The built-in defaults in :data:`DEFAULT_CONFIG`.

Config files are JSON with four sections (see ``config.example.json``
in the project root). Unknown sections or keys are rejected with a
``ValueError`` naming the key, so a typo fails loudly at startup
instead of silently doing nothing. Every value is type- and
range-checked the same way: a bad port, an unknown ranking method or
an out-of-range ``k`` is a startup error, not a surprise at runtime.
"""
import copy
import json
import os

from .index import SEARCH_METHODS
from .validation import MAX_K, MIN_K

CONFIG_FILE_NAME = "config.json"
ENV_CONFIG_PATH = "DOCUMIND_CONFIG"

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

DEFAULT_CONFIG = {
    "server": {
        "host": "127.0.0.1",
        "port": 8000,
    },
    "paths": {
        "index_file": "index.json",
        "store_file": "documind.db",
        "data_dir": "data",
    },
    "retrieval": {
        "default_method": "tfidf",
        "search_k": 5,
        "ask_k": 3,
        "chunk_size": 500,
        "chunk_overlap": 80,
    },
    "logging": {
        "level": "INFO",
        "file": None,
        "json": True,
    },
}

# Environment variables that override a single config value. The key
# is (section, key) in the config dict; the value is the env var name.
_ENV_OVERRIDES = {
    ("server", "host"): "DOCUMIND_HOST",
    ("server", "port"): "DOCUMIND_PORT",
    ("logging", "level"): "DOCUMIND_LOG_LEVEL",
    ("logging", "file"): "DOCUMIND_LOG_FILE",
}


def _fail(message):
    raise ValueError("invalid config: %s" % message)


def _check_int(value, section, key, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("%s.%s must be an integer, got %r" % (section, key, value))
    if low is not None and value < low:
        _fail("%s.%s must be at least %d, got %d"
              % (section, key, low, value))
    if high is not None and value > high:
        _fail("%s.%s must be at most %d, got %d"
              % (section, key, high, value))
    return value


def _check_str(value, section, key, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        _fail("%s.%s must be a non-empty string, got %r"
              % (section, key, value))
    return value


def _check_bool(value, section, key):
    if not isinstance(value, bool):
        _fail("%s.%s must be true or false, got %r" % (section, key, value))
    return value


def _validate(config):
    """Check every value in a merged config; raise ValueError on the
    first bad one. Returns the config unchanged."""
    server = config["server"]
    _check_str(server["host"], "server", "host")
    _check_int(server["port"], "server", "port", low=1, high=65535)

    paths = config["paths"]
    for key in ("index_file", "store_file", "data_dir"):
        _check_str(paths[key], "paths", key)

    retrieval = config["retrieval"]
    if retrieval["default_method"] not in SEARCH_METHODS:
        _fail("retrieval.default_method must be one of %s, got %r"
              % (", ".join(SEARCH_METHODS), retrieval["default_method"]))
    _check_int(retrieval["search_k"], "retrieval", "search_k",
               low=MIN_K, high=MAX_K)
    _check_int(retrieval["ask_k"], "retrieval", "ask_k",
               low=MIN_K, high=MAX_K)
    _check_int(retrieval["chunk_size"], "retrieval", "chunk_size", low=1)
    _check_int(retrieval["chunk_overlap"], "retrieval", "chunk_overlap",
               low=0)
    if retrieval["chunk_overlap"] >= retrieval["chunk_size"]:
        _fail("retrieval.chunk_overlap (%d) must be smaller than "
              "retrieval.chunk_size (%d)"
              % (retrieval["chunk_overlap"], retrieval["chunk_size"]))

    logging_cfg = config["logging"]
    if logging_cfg["level"] not in LOG_LEVELS:
        _fail("logging.level must be one of %s, got %r"
              % (", ".join(LOG_LEVELS), logging_cfg["level"]))
    if logging_cfg["file"] is not None:
        _check_str(logging_cfg["file"], "logging", "file")
    _check_bool(logging_cfg["json"], "logging", "json")
    return config


def _merge(base, override, where=""):
    """Deep-merge ``override`` onto ``base`` (both plain dicts).

    Unknown sections and keys are rejected, so a misspelled key in a
    config file cannot silently do nothing.
    """
    merged = copy.deepcopy(base)
    if not isinstance(override, dict):
        _fail("expected a JSON object%s, got %s"
              % ((" at %s" % where) if where else "", type(override).__name__))
    for section, values in override.items():
        if section not in merged:
            _fail("unknown section %r (expected one of %s)"
                  % (section, ", ".join(sorted(merged))))
        if not isinstance(values, dict):
            _fail("section %r must be a JSON object, got %s"
                  % (section, type(values).__name__))
        for key, value in values.items():
            if key not in merged[section]:
                _fail("unknown key %r in section %r (expected one of %s)"
                      % (key, section, ", ".join(sorted(merged[section]))))
            merged[section][key] = value
    return merged


def _apply_env_overrides(config, env=None):
    """Apply the DOCUMIND_* environment variables onto ``config``.

    ``env`` defaults to ``os.environ``; passing a dict keeps the
    function testable without touching the real environment.
    """
    env = os.environ if env is None else env
    for (section, key), var in _ENV_OVERRIDES.items():
        raw = env.get(var)
        if raw is None or raw == "":
            continue
        if key == "port":
            try:
                value = int(raw)
            except ValueError:
                _fail("%s must be an integer, got %r" % (var, raw))
        elif key == "level":
            value = raw.upper()
        else:
            value = raw
        config[section][key] = value
    return config


def default_base_dir():
    """The project root: the parent of the ``documind`` package."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None, base_dir=None, env=None):
    """Load the effective config as a plain nested dict.

    ``path`` is an explicit config file; when it is given but does not
    exist, that is an error rather than a silent fallback. Otherwise
    the ``DOCUMIND_CONFIG`` environment variable wins, then
    ``<base_dir>/config.json`` when it exists. With no file at all the
    built-in defaults apply. Environment overrides are applied last,
    so they win over both the file and the defaults.
    """
    base_dir = base_dir or default_base_dir()
    env = os.environ if env is None else env
    config = copy.deepcopy(DEFAULT_CONFIG)

    if path is None:
        path = env.get(ENV_CONFIG_PATH)
    if path is None:
        candidate = os.path.join(base_dir, CONFIG_FILE_NAME)
        path = candidate if os.path.exists(candidate) else None

    if path is not None:
        if not os.path.exists(path):
            _fail("config file not found: %s" % path)
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError) as exc:
            _fail("cannot read config file %s: %s" % (path, exc))
        config = _merge(config, raw)

    _apply_env_overrides(config, env=env)
    return _validate(config)


def resolve_paths(config, base_dir=None):
    """Resolve the ``paths`` section against ``base_dir``.

    Relative entries are interpreted relative to the project root, so
    a config that says ``"index_file": "index.json"`` keeps working
    wherever the process was started from. Absolute entries pass
    through unchanged. Returns ``{"index_path", "store_path",
    "data_dir"}`` as absolute paths.
    """
    base_dir = base_dir or default_base_dir()
    paths = config["paths"]

    def _resolve(value):
        return value if os.path.isabs(value) else os.path.join(base_dir,
                                                               value)

    return {"index_path": _resolve(paths["index_file"]),
            "store_path": _resolve(paths["store_file"]),
            "data_dir": _resolve(paths["data_dir"])}
