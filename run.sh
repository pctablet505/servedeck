#!/usr/bin/env bash
# run.sh — foreground dev runner for Servedeck. Not the way the deployed
# service starts (that's systemd/servedeck.service, installed by setup.sh);
# this is for running it by hand at a terminal to watch its own logs while
# developing/debugging.
#
# ASSUMPTION FLAGGED (see systemd/servedeck.service's own comment): this
# imports `servedeck.app:app` as the FastAPI application object. Update the
# module path below if whichever agent builds the API layer names it
# something else.
#
# Binds 127.0.0.1:8010 only — SPEC.md §1: "Runtime is network-free:
# 127.0.0.1 only". Loading this never touches port 8000/8001 or starts any
# model server; it only serves Servedeck's own GUI/API/gateway process.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
HOST="127.0.0.1"
PORT="8010"

if [ ! -x "$VENV/bin/uvicorn" ]; then
    echo "No venv at $VENV (or uvicorn isn't installed in it)." >&2
    echo "Run ./setup.sh first." >&2
    exit 1
fi

# cd into the project root so `servedeck` resolves as an importable package
# (uvicorn's default --app-dir is the current working directory) without
# needing an editable install of this project itself.
cd "$HERE"

reload_flag=()
if [ "${SERVEDECK_RELOAD:-0}" != "0" ]; then
    reload_flag=(--reload --reload-dir "$HERE/servedeck")
fi

echo "Servedeck starting on http://$HOST:$PORT (Ctrl-C to stop)"
exec "$VENV/bin/uvicorn" servedeck.app:app --host "$HOST" --port "$PORT" "${reload_flag[@]}" "$@"
