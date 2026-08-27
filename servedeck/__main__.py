"""Console entry point: `servedeck` starts the dashboard."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    from . import config

    cfg = config.get()
    ap = argparse.ArgumentParser(prog="servedeck", description="Local LLM server dashboard")
    ap.add_argument("--host", default=cfg.listen_host)
    ap.add_argument("--port", type=int, default=cfg.listen_port)
    ap.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    args = ap.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Try: pip install 'servedeck-llm[server]'", file=sys.stderr)
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
