"""``python -m servedeck`` — start the dashboard + gateway.

Separate from ``servedeck.cli`` (the ``servedeck`` console script) on purpose:
this starts the long-running server, that talks to one. The systemd unit runs
this; a terminal runs the CLI.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys

import uvicorn

from . import settings as _settings


#: Everything servedeck's own modules log, at this level, goes to the journal.
#: Without it the ``servedeck.*`` loggers had no handler at all, so only
#: WARNING+ escaped (via logging.lastResort) and the unit's journal held
#: nothing but uvicorn access lines: which model reconcile decided to start at
#: boot, the utilisation it chose, every refusal reason the code takes care to
#: word well — none of it was recorded anywhere a person could read later.
LOG_LEVEL_ENV = "SERVEDECK_LOG_LEVEL"


def _configure_logging() -> None:
    """One stderr handler for the ``servedeck`` logger tree.

    stderr because systemd captures it into the journal with the unit's
    identifier; no timestamp in the format because journald already stamps
    every line, and two timestamps per line is how a log becomes unreadable.
    """
    level_name = os.environ.get(LOG_LEVEL_ENV, "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger = logging.getLogger("servedeck")
    logger.setLevel(level)
    if not any(getattr(h, "_servedeck", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        handler._servedeck = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    # uvicorn's access log stays ON. It is noisy (900+ lines an hour here) and
    # thin — method, path, status, nothing about the model or the tokens — but
    # it is the only durable record that the gateway was used at all, and
    # silencing it to make room for these lines would trade one gap for
    # another. servedeck's own decisions are greppable by their prefix:
    #     journalctl --user -u servedeck | grep 'servedeck\.'


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
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

    if args.reload:
        # --reload needs an import string, and the string form cannot carry
        # --no-reconcile; say so rather than silently ignoring the flag.
        if args.no_reconcile:
            print("--no-reconcile is not supported with --reload", file=sys.stderr)
            return 2
        uvicorn.run(
            "servedeck.app:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
            # --reload is given an import string, so there is no app instance
            # to take a hub from and the early close below cannot be installed.
            # The bound still applies, which is all a dev-only flag needs.
            timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S,
        )
        return 0

    application = _app.create_app(reconcile=not args.no_reconcile)
    config = uvicorn.Config(
        application,
        host=args.host,
        port=args.port,
        # The hard bound. Nothing below can make a stop exceed this, and the
        # unit's TimeoutStopSec=15 is comfortably above it, so systemd never
        # reaches the SIGKILL that used to end every single stop.
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S,
    )
    _ShutdownClosesTheHub(config, application.state.rt.hub).run()
    return 0


#: How long uvicorn may spend draining connections. A streaming chat completion
#: relayed through the gateway is cut off at this point — correct on a stop: the
#: MODEL keeps running in its own transient unit, only the relay stops, and the
#: client retries against a dashboard that is back in seconds.
GRACEFUL_SHUTDOWN_S = 5


class _ShutdownClosesTheHub(uvicorn.Server):
    """Close the SSE hub on the way into shutdown, not on the way out.

    MEASURED, 2026-09-12: with one ``/api/events`` connection open, a plain
    ``uvicorn.Server`` did not exit at all — still running after 40 s — while
    the same server with the hub closed first exits in 1.2 s.

    The reason is the order of uvicorn's own shutdown. It (1) stops accepting,
    (2) asks each live connection to finish, (3) WAITS for them, and only then
    (4) runs the ASGI lifespan's shutdown. ``app.py`` closes the hub in the
    lifespan — step 4 — which is after the wait it was supposed to shorten. An
    SSE response never finishes on its own, so step 3 blocked forever.

    ``handle_exit`` is the earliest hook: it is what the signal handler calls,
    so closing the hub here happens before step 2. Every parked subscriber gets
    its sentinel, every generator returns, and the drain has nothing to wait
    for. The lifespan still closes the hub as well — that path covers a
    shutdown this override does not see (a test driving ``should_exit``
    directly), and closing twice is idempotent.
    """

    def __init__(self, config: uvicorn.Config, hub: object) -> None:
        super().__init__(config)
        # Handed in, not discovered. `config.loaded_app` is only populated once
        # `run()` has called `config.load()`, and it is the MIDDLEWARE-WRAPPED
        # app by then, so `loaded_app.state` does not exist — the first cut
        # looked there, found nothing, silently closed no hub, and every
        # shutdown quietly fell back to the 5 s timeout instead of 1.2 s. A
        # missing hub that degrades to "slower but still correct" is exactly
        # the kind of failure nothing reports.
        self._hub = hub

    def handle_exit(self, sig: int, frame: object) -> None:  # type: ignore[override]
        with contextlib.suppress(Exception):
            self._hub.close()  # type: ignore[attr-defined]
        super().handle_exit(sig, frame)  # type: ignore[arg-type]


if __name__ == "__main__":
    raise SystemExit(main())
