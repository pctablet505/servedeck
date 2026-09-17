"""Desired state — the one thing servedeck remembers across restarts
(REDESIGN-2026-09-12.md §2.2).

    {"version": 3, "main": "flashnext" | null, "residents": ["lfm2"],
     "launch": {"flashnext": {"util": 0.96,
                              "argv": {"--max-model-len": "262144",
                                       "--max-num-seqs": "16",
                                       "--kv-offloading-size": "40"}}}}

That is the whole schema. Not a mirror of what is running — *what an operator
last explicitly asked for*. It is written only by ``control.start`` /
``control.stop`` / ``control.switch`` / ``control.adopt``; nothing derives it
from a probe, because a model that crashed must stay "desired" so reconcile
brings it back, and a model an operator stopped must stay stopped even though
adopting a stray unit would otherwise silently re-desire it.

``launch`` (v3, 2026-09-18) is the second thing an operator asks for: the
values they applied on the page. Without it the allocator was write-only —
the tuned utilisation, context, agent count and KV-offload size lived in the
argv of a running process and nowhere else, so the next servedeck restart or
reboot relaunched the model at the registry defaults with the utilisation
recomputed from free VRAM (0.98 on an idle card, the value with the known
OOM-under-concurrency history) and nothing said so. Written only after a
launch that actually became ready, so a configuration that fails to boot is
never the one a reboot repeats.

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

__all__ = ["Desired", "Launch", "SCHEMA_VERSION", "load", "save", "default_path"]

log = logging.getLogger(__name__)

SCHEMA_VERSION = 3


@dataclass(frozen=True)
class Launch:
    """The settings one model was last launched READY with.

    ``argv`` holds whole flags (``{"--max-num-seqs": "16"}``) rather than named
    fields, so a control the page grows tomorrow persists without a schema
    change; ``None`` as a value means "remove this flag", exactly as
    ``control.apply_argv_overrides`` reads it.
    """

    util: float | None = None
    argv: dict[str, str | None] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        out: dict[str, object] = {}
        if self.util is not None:
            out["util"] = self.util
        if self.argv:
            out["argv"] = dict(self.argv)
        return out

    def __bool__(self) -> bool:
        return self.util is not None or bool(self.argv)


@dataclass
class Desired:
    """What an operator last asked for."""

    main: str | None = None
    residents: list[str] = field(default_factory=list)
    #: Per-model launch settings from the last ready boot; see :class:`Launch`.
    launch: dict[str, Launch] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "version": SCHEMA_VERSION,
            "main": self.main,
            "residents": list(self.residents),
            "launch": {k: v.to_json() for k, v in sorted(self.launch.items()) if v},
        }

    def _copy(self, **changes: object) -> "Desired":
        out = Desired(main=self.main, residents=list(self.residents), launch=dict(self.launch))
        for name, value in changes.items():
            setattr(out, name, value)
        return out

    def with_main(self, key: str | None) -> "Desired":
        return self._copy(main=key)

    def with_resident(self, key: str) -> "Desired":
        if key in self.residents:
            return self._copy()
        return self._copy(residents=[*self.residents, key])

    def without_resident(self, key: str) -> "Desired":
        return self._copy(residents=[k for k in self.residents if k != key])

    def with_launch(self, key: str, launch: Launch) -> "Desired":
        merged = dict(self.launch)
        if launch:
            merged[key] = launch
        else:
            merged.pop(key, None)
        return self._copy(launch=merged)

    def launch_for(self, key: str) -> Launch:
        return self.launch.get(key, Launch())


def default_path() -> Path:
    """``<project>/state/desired.json`` — the same filename v1 used, on
    purpose: one file per box, migrated in place, never two."""
    from servedeck import settings

    return settings.get().desired_path


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
    if version == 2:
        # v2 -> v3 is purely additive (no `launch` block means "no settings
        # recorded yet"), so it migrates silently and in place: a file left at
        # the old version would be re-read and re-migrated on every poll.
        migrated = _parse(raw)
        try:
            save(migrated, target)
        except OSError as exc:
            log.warning("%s: could not persist the v3 migration (%s)", target, exc)
        return migrated
    if version == 1:
        migrated = _from_v1(raw, str(target))
        # Migrate in place, once. Left as v1 on disk, every poll re-read and
        # re-warned (30 lines a minute on 2026-09-17) and nothing ever wrote
        # v2 unless an operator action happened to change the desired set.
        # v1's file is kept beside it for the rollback the cutover runbook
        # describes.
        try:
            backup = target.with_name(target.name + ".v1")
            if not backup.exists():
                backup.write_text(text, encoding="utf-8")
            save(migrated, target)
        except OSError as exc:
            log.warning("%s: could not persist the v2 migration (%s)", target, exc)
        return migrated
    if version != SCHEMA_VERSION:
        log.warning(
            "%s has unknown schema version %r (expected %d); treating desired "
            "state as empty rather than guessing",
            target,
            version,
            SCHEMA_VERSION,
        )
        return Desired()

    return _parse(raw)


def _parse(raw: dict[str, object]) -> Desired:
    """The v2/v3 body. Every field degrades to its empty value rather than
    raising: this file must never be the reason servedeck will not start."""
    main = raw.get("main")
    if not isinstance(main, str) or not main:
        main = None
    residents_raw = raw.get("residents")
    residents: list[str] = []
    if isinstance(residents_raw, list):
        for item in residents_raw:
            if isinstance(item, str) and item and item not in residents:
                residents.append(item)
    launch: dict[str, Launch] = {}
    launch_raw = raw.get("launch")
    if isinstance(launch_raw, dict):
        for key, entry in launch_raw.items():
            if not isinstance(key, str) or not isinstance(entry, dict):
                continue
            util = entry.get("util")
            argv_raw = entry.get("argv")
            argv: dict[str, str | None] = {}
            if isinstance(argv_raw, dict):
                for flag, value in argv_raw.items():
                    if isinstance(flag, str) and flag.startswith("-"):
                        argv[flag] = value if isinstance(value, str) else None
            parsed = Launch(
                util=float(util) if isinstance(util, (int, float)) and not isinstance(util, bool) else None,
                argv=argv,
            )
            if parsed:
                launch[key] = parsed
    return Desired(main=main, residents=residents, launch=launch)


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
