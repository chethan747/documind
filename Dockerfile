# DocuMind — AI Document Intelligence Platform (Day 16)
#
# Pure Python standard library: no third-party dependencies, so the
# image is just Python plus the source — nothing to pip-install.
# The server is the same one `python3 cli.py serve` starts locally.
#
# Build:   docker build -t documind .
# Run:     docker run --rm -p 8000:8000 documind
#          (or `docker compose up`, which also persists /data)
# Then open http://localhost:8000 for the upload + ask page.
FROM python:3.12-slim

# The image defaults are the container-sensible ones: listen on all
# interfaces so a published port can reach the server, and keep the
# index, the SQLite store and the data directory in /data, which
# docker-compose mounts as a named volume so documents and the index
# survive container restarts. All three are plain Day 15 config
# values, so any of them can be overridden with the matching
# DOCUMIND_* environment variable or a mounted config.json.
ENV PYTHONUNBUFFERED=1 \
    DOCUMIND_HOST=0.0.0.0 \
    DOCUMIND_PORT=8000 \
    DOCUMIND_INDEX_FILE=/data/index.json \
    DOCUMIND_STORE_FILE=/data/documind.db \
    DOCUMIND_DATA_DIR=/data/docs

WORKDIR /app

# Run as a non-root user: the server only needs to read its own code
# and write to /data.
RUN groupadd -r documind && useradd -r -g documind documind \
    && mkdir -p /data/docs \
    && chown -R documind:documind /data

# .dockerignore keeps runtime artefacts (index.json, documind.db,
# config.json, __pycache__, logs) out of the build context.
COPY --chown=documind:documind cli.py config.example.json README.md ./
COPY --chown=documind:documind documind/ ./documind/
COPY --chown=documind:documind data/ ./data/
COPY --chown=documind:documind eval/ ./eval/
COPY --chown=documind:documind tests/ ./tests/

USER documind

EXPOSE 8000

# The server's own /health endpoint: 200 with {"status": "ok"} once
# it is listening. Uses the stdlib urllib so no extra tool is needed
# in the slim image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"

# Host, port and paths come from the DOCUMIND_* environment above;
# `serve` with no flags reads them from the config, exactly as the
# CLI does on a laptop.
CMD ["python3", "cli.py", "serve"]
