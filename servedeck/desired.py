"""Desired state — the one thing servedeck remembers across restarts
(REDESIGN-2026-09-12.md §2.2).

    {"version": 2, "main": "flashnext" | null, "residents": ["lfm2"]}

That is the whole schema. Not a mirror of what is running — *what an operator
last explicitly asked for*. It is written only by ``control.start`` /
``control.stop`` / ``control.switch`` / ``control.adopt``; nothing derives it
from a probe, because a model that crashed must stay "desired" so reconcile
brings it back, and a model an operator stopped must stay stopped even though
adopting a stray unit would otherwise silently re-desire it.

v1 (``{"version": 1, "desired_state": "RUNNING", "backend": "inline", "port":
8004, "util": 0.91, ...}``) described a single server with its launch flags
inline. v2 keeps none of that: flags come from the registry now (R1), and the
"one server" assumption is what made a resident model impossible. Reading a v1
file yields ``main = backend if desired_state == "RUNNING" else None`` and no
residents, with a warning — v1's ``backend`` was a launcher name ("inline"),
not necessarily a v2 registry key, so the caller is told to check.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Desired", "SCHEMA_VERSION", "load", "save", "default_path"]

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2


@dataclass
class Desired:
    """What an operator last asked for."""

    main: str | None = None
    residents: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, object]:
        return {
            "version": SCHEMA_VERSION,
            "main": self.main,
            "residents": list(self.residents),
        }

    def with_main(self, key: str | None) -> "Desired":
        return Desired(main=key, residents=list(self.residents))

    def with_resident(self, key: str) -> "Desired":
        if key in self.residents:
            return Desired(main=self.main, residents=list(self.residents))
        return Desired(main=self.main, residents=[*self.residents, key])

    def without_resident(self, key: str) -> "Desired":
        return Desired(main=self.main, residents=[k for k in self.residents if k != key])


def default_path() -> Path:
    """``<project>/state/desired.json`` — the same filename v1 used, on
    purpose: one file per box, migrated in place, never two."""
    from servedeck import paths

    return paths.STATE_DIR / "desired.json"


def _from_v1(raw: dict[str, object], source: str) -> Desired:
    backend = raw.get("backend")
    state = raw.get("desired_state")
    main = backend if (state == "RUNNING" and isinstance(backend, str) and backend) else None
    log.warning(
        "%s is schema version 1 (desired_state=%r backend=%r); migrating to "
        "version 2 as main=%r, residents=[]. v1's `backend` was a launcher "
        "name, not a registry key — confirm it names a model in models.toml "
        "before trusting reconcile.",
        source,
        state,
        backend,
        main,
    )
    return Desired(main=main, residents=[])


def load(path: str | os.PathLike[str] | None = None) -> Desired:
    """Read desired state. A missing, empty or unparseable file yields the
    empty ``Desired()`` — never raises, because failing to read this file must
    degrade to "want nothing", not to a servedeck that will not start."""
    target = Path(path) if path is not None else default_path()
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return Desired()
    except OSError as exc:
        log.warning("%s unreadable (%s); treating desired state as empty", target, exc)
        return Desired()

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        log.warning("%s is not valid JSON (%s); treating desired state as empty", target, exc)
        return Desired()
    if not isinstance(raw, dict):
        log.warning("%s is not a JSON object; treating desired state as empty", target)
        return Desired()

    version = raw.get("version")
    if version == 1:
        return _from_v1(raw, str(target))
    if version != SCHEMA_VERSION:
        log.warning(
            "%s has unknown schema version %r (expected %d); treating desired "
            "state as empty rather than guessing",
            target,
            version,
            SCHEMA_VERSION,
        )
        return Desired()

    main = raw.get("main")
    if not isinstance(main, str) or not main:
        main = None
    residents_raw = raw.get("residents")
    residents: list[str] = []
    if isinstance(residents_raw, list):
        for item in residents_raw:
            if isinstance(item, str) and item and item not in residents:
                residents.append(item)
    return Desired(main=main, residents=residents)


def save(desired: Desired, path: str | os.PathLike[str] | None = None) -> Path:
    """Write atomically: temp file in the same directory, fsync, ``os.replace``.

    Same-directory matters — ``os.replace`` is only atomic within a filesystem
    — and the fsync matters because the file whose whole job is surviving a
    restart must survive the kind of restart that is a power cut.
    """
    target = Path(path) if path is not None else default_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(desired.to_json(), indent=2, sort_keys=True) + "\n"

    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".desired-", suffix=".json")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target
