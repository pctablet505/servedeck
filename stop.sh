#!/usr/bin/env bash
# stop.sh — stop the Servedeck dashboard (NOT the model server).
#
# Finds the process by the port it is listening on, never by matching a
# command line: `pkill -f uvicorn` would also kill any other uvicorn you are
# running, and on this box `-f` patterns have matched the calling shell.
set -euo pipefail

PORT="${SERVEDECK_PORT:-8010}"

pid_on_port() {
    ss -H -ltnp "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1
}

PID="$(pid_on_port || true)"
if [ -z "$PID" ]; then
    echo "Nothing is listening on 127.0.0.1:$PORT — already stopped."
    exit 0
fi

echo "Stopping Servedeck (pid $PID) on :$PORT..."
kill "$PID" 2>/dev/null || true

for _ in $(seq 1 20); do
    sleep 0.5
    [ -z "$(pid_on_port || true)" ] && { echo "Stopped. :$PORT is free."; exit 0; }
done

echo "Still running after 10s; sending SIGKILL." >&2
kill -9 "$PID" 2>/dev/null || true
sleep 1
if [ -z "$(pid_on_port || true)" ]; then
    echo "Stopped. :$PORT is free."
else
    echo "Could not stop pid $PID — check it by hand: ps -p $PID" >&2
    exit 1
fi
