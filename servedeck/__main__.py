"""Console entry point: `servedeck` starts the dashboard."""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    from . import config

    cfg = config.get()
    ap = argparse.ArgumentParser(prog="servedeck", description="Local LLM server dashboard")
    ap.add_argument("--host", default=cfg.listen_host)
    ap.add_argument("--port", type=int, default=cfg.listen_port)
    ap.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    # P9: the master switch for request/response capture (glm_policies.py). A
    # route's own `capture = true` in models.toml is consent, not an enable --
    # capture writes whatever the user typed to disk, so turning it on has to be
    # a thing somebody did on purpose, today, at the command line. Passed as an
    # env var because the app is handed to uvicorn by import string, so there is
    # no object to pass it to.
    ap.add_argument(
        "--glm-capture",
        action="store_true",
        help="capture the last few request/response bodies of any model whose "
        "registry entry sets capture = true (state/captures/, bodies on disk)",
    )
    args = ap.parse_args(argv)
    if args.glm_capture:
        from .glm_policies import CAPTURE_ENV

        os.environ[CAPTURE_ENV] = "1"

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Try: pip install servedeck", file=sys.stderr)
        return 1

    if not cfg.backends:
        print(
            "No backends configured — the dashboard will start, but it cannot\n"
            "identify your model server. Copy servedeck.toml.example to\n"
            "servedeck.toml and add one. See docs/CONFIGURATION.md.",
            file=sys.stderr,
        )
    uvicorn.run("servedeck.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
