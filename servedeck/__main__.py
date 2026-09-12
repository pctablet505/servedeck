"""``python -m servedeck`` — start the dashboard + gateway.

Separate from ``servedeck.cli`` (the ``servedeck`` console script) on purpose:
this starts the long-running server, that talks to one. The systemd unit runs
this; a terminal runs the CLI.
"""

from __future__ import annotations

import argparse
import sys

from . import settings as _settings


def main(argv: list[str] | None = None) -> int:
    cfg = _settings.get()
    ap = argparse.ArgumentParser(prog="python -m servedeck", description="servedeck v2 server")
    ap.add_argument("--host", default=cfg.listen_host)
    ap.add_argument("--port", type=int, default=cfg.listen_port)
    ap.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    ap.add_argument(
        "--no-reconcile",
        action="store_true",
        help="serve, but do not start anything from desired.json",
    )
    args = ap.parse_args(argv)

    try:
        import uvicorn
    except ImportError:  # pragma: no cover - packaging failure
        print("uvicorn is not installed. Try: pip install -e '.[dev]'", file=sys.stderr)
        return 1

    if not cfg.models_path.is_file():
        print(
            f"{cfg.models_path} does not exist. servedeck v2 is driven entirely by\n"
            "that file; see docs/CONFIGURATION.md. ($SERVEDECK_MODELS overrides it.)",
            file=sys.stderr,
        )
        return 1

    # --host/--port on the command line must reach the app too: it polls its own
    # listen URL before reconciling, and a factory that read only the
    # environment would poll :8010 while uvicorn bound something else — the
    # reconcile would then wait out its timeout and start nothing.
    import os

    os.environ["SERVEDECK_HOST"] = args.host
    os.environ["SERVEDECK_PORT"] = str(args.port)
    _settings.reset()

    from . import app as _app

    factory = (lambda: _app.create_app(reconcile=not args.no_reconcile)) if not args.reload else None
    if args.reload:
        # --reload needs an import string, and the string form cannot carry
        # --no-reconcile; say so rather than silently ignoring the flag.
        if args.no_reconcile:
            print("--no-reconcile is not supported with --reload", file=sys.stderr)
            return 2
        uvicorn.run("servedeck.app:create_app", factory=True, host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(factory, factory=True, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
