"""Every machine- and environment-specific value servedeck needs, in one place
(REDESIGN-2026-09-12.md §2.1, R1).

This module replaces ``config.py`` (283 lines of per-backend launcher tables,
env-var maps, log paths and `servedeck.toml` parsing) and ``paths.py`` (104
lines of absolute paths into two sibling projects). Both existed because a
model's identity lived in many files; in v2 it lives in ``models.toml``, so
what is left over is genuinely small:

* where to listen,
* where ``models.toml`` is,
* where mutable state goes,
* which systemd unit namespace we own.

That is the whole surface. Nothing here knows a model's port, flags, venv or
context length — asking this module for one of those is the bug the redesign
removes.

Deliberately dependency-free: it imports nothing from ``servedeck``. Anything
that needs both the registry and a setting composes them itself, so there is
no import cycle to route around and no "settings" that quietly reads a TOML
file on import.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = [
    "Settings",
    "PACKAGE_DIR",
    "PROJECT_ROOT",
    "UNIT_PREFIXES",
    "DEFAULT_UNIT_PREFIX",
    "get",
    "reset",
    "load",
]

#: ``<project>/servedeck`` and ``<project>``. The registry and the state dir
#: default next to the package because the repo *is* the deployment: one
#: checkout, one unit, one models.toml (REDESIGN §2 "one repository").
PACKAGE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = PACKAGE_DIR.parent

#: The two systemd unit namespaces that exist. ``model-`` is production;
#: ``sd-test-`` is the suite's, and ``units.py`` refuses every other shape
#: before spawning anything. An arbitrary prefix is rejected here rather than
#: at spawn time so a typo in the environment fails at startup, loudly, instead
#: of producing a Control whose discovery glob matches nothing and which
#: therefore reports a serving box as empty.
UNIT_PREFIXES: tuple[str, ...] = ("model-", "sd-test-")
DEFAULT_UNIT_PREFIX = "model-"

DEFAULT_LISTEN_HOST = "127.0.0.1"
#: Not 8000 (ats-optimizer.service) and not a model's own port: :8010 is the
#: one URL every client is configured with (REDESIGN decision 3).
DEFAULT_LISTEN_PORT = 8010


class SettingsError(Exception):
    """An environment variable names something servedeck will not accept."""


@dataclass(frozen=True)
class Settings:
    """Resolved settings. Immutable: a process serves one configuration, and a
    value that could change under a running reconcile loop is a bug waiting to
    be written."""

    listen_host: str
    listen_port: int
    #: ``models.toml`` — the single source of truth for every model.
    models_path: Path
    #: Mutable state: ``desired.json``, ``wire``'s dated backups, the
    #: measurement store. Created on first write, never on import.
    state_dir: Path
    #: ``model-`` in production. Control is constructed with this, and so is
    #: every unit name and the list-units glob — one value, so discovery and
    #: action can never disagree about which units are ours.
    unit_prefix: str

    @property
    def gateway_url(self) -> str:
        """The URL clients are configured with: ``http://host:port/v1``."""
        return f"http://{self.listen_host}:{self.listen_port}/v1"

    @property
    def base_url(self) -> str:
        return f"http://{self.listen_host}:{self.listen_port}"

    @property
    def health_url(self) -> str:
        """What the reconcile task polls before touching a model: servedeck's
        own listen socket, answering its own health route. See
        ``app.py``'s ``_reconcile_after_bind`` — reconciling from inside the
        ASGI lifespan is what launched the 27B seventy times on 2026-09-11
        (REDESIGN §4 R4), because uvicorn runs the lifespan BEFORE it binds
        and a failed bind then retried the whole thing."""
        return f"{self.base_url}/api/health"

    @property
    def desired_path(self) -> Path:
        return self.state_dir / "desired.json"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SettingsError(f"{name}={raw!r} is not an integer") from exc


def load(environ: dict[str, str] | None = None) -> Settings:
    """Resolve settings from ``environ`` (default: ``os.environ``).

    Takes the mapping as an argument so a test can exercise every branch
    without mutating the process environment — the pattern that makes
    ``reset()`` a convenience rather than a requirement.
    """
    env = os.environ if environ is None else environ

    models_raw = env.get("SERVEDECK_MODELS") or env.get("SERVEDECK_MODELS_TOML")
    models_path = Path(models_raw) if models_raw else PROJECT_ROOT / "models.toml"

    state_raw = env.get("SERVEDECK_STATE_DIR")
    state_dir = Path(state_raw) if state_raw else PROJECT_ROOT / "state"

    prefix = env.get("SERVEDECK_UNIT_PREFIX") or DEFAULT_UNIT_PREFIX
    if prefix not in UNIT_PREFIXES:
        raise SettingsError(
            f"SERVEDECK_UNIT_PREFIX={prefix!r} is not one of {UNIT_PREFIXES}. "
            "Only these two namespaces exist: 'model-' is production and "
            "'sd-test-' is the test suite's, and units.py refuses anything else."
        )

    port = _env_int("SERVEDECK_PORT", DEFAULT_LISTEN_PORT)
    host = env.get("SERVEDECK_HOST") or DEFAULT_LISTEN_HOST

    return Settings(
        listen_host=host,
        listen_port=port,
        models_path=models_path.expanduser(),
        state_dir=state_dir.expanduser(),
        unit_prefix=prefix,
    )


@lru_cache(maxsize=1)
def _cached() -> Settings:
    return load()


def get() -> Settings:
    """The process's settings, resolved once."""
    return _cached()


def reset() -> None:
    """Forget the cached settings, so the next :func:`get` re-reads the
    environment. For tests and for ``__main__`` after it parses ``--port``."""
    _cached.cache_clear()
