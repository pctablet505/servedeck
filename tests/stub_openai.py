"""A minimal OpenAI-shaped server, stdlib only, for tests/test_control_e2e.py.

Exists so the lifecycle half of the e2e — transient unit, cgroup, MainPID
parentage, stop, collection — can be proven in about a second and without a
GPU. The vLLM half of that file proves the same code against a real model when
there is VRAM to spare; this one proves it always.

Stdlib only and no imports from `servedeck`, because it is exec'd as the
payload of a systemd unit whose environment is the user manager's, not the
test runner's. A dependency here would be a dependency of the thing under
test.

    python tests/stub_openai.py --port 8030 --model sd-test-stub-model
    python tests/stub_openai.py --port 8030 --model x --delay-ready 5
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

READY = False
MODEL_ID = "sd-test-stub-model"

#: Printed at the points a real vLLM prints them, so a test can drive the
#: marker detector without a model. Wording is copied from vLLM's own log
#: lines — if vLLM changes them, control.READY_MARKERS is what must change,
#: and this file is where the expectation is written down.
BOOT_LINES = (
    "Loading weights took 0.01 seconds",
    "GPU KV cache size: 4,096 tokens",
    "Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)",
)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/models":
            self._send(404, {"error": "not found"})
            return
        if not READY:
            # Listening but not serving — the state a real model spends most
            # of its boot in, and the one a port-is-open readiness check
            # mistakes for ready.
            self._send(503, {"error": "still loading"})
            return
        self._send(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        self._send(200, {
            "id": "stub-1",
            "object": "chat.completion",
            "model": MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
        })

    def log_message(self, fmt: str, *args) -> None:
        # journald captures stderr; keep it quiet but not silent.
        sys.stderr.write("stub %s\n" % (fmt % args))


def main() -> int:
    global READY, MODEL_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--delay-ready", type=float, default=0.0,
                        help="seconds to answer 503 before reporting the model")
    parser.add_argument("--ready-grace", type=float, default=2.0,
                        help="pause between the last boot line and becoming ready")
    parser.add_argument("--emit-boot-lines", action="store_true",
                        help="print vLLM's boot markers to stdout as it starts")
    args = parser.parse_args()

    MODEL_ID = args.model
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True

    def become_ready_after_booting() -> None:
        """Markers first, readiness strictly afterwards.

        A real vLLM prints "Application startup complete." and begins serving
        at very nearly the same instant, so the ordering a test relies on is
        made explicit here rather than left to a race: the grace pause is the
        margin `control._READY_DRAIN_S` exists to cover.
        """
        global READY
        if args.emit_boot_lines:
            step = args.delay_ready / (len(BOOT_LINES) + 1) if args.delay_ready else 0.0
            for line in BOOT_LINES:
                time.sleep(step)
                print(line, flush=True)
            time.sleep(step)
            print("Application startup complete.", flush=True)
            time.sleep(args.ready_grace)
        else:
            time.sleep(args.delay_ready)
        READY = True

    threading.Thread(target=become_ready_after_booting, daemon=True).start()

    print(f"stub listening on {args.host}:{args.port} as {MODEL_ID}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
