"""Every surviving `coldstart` spelling, in one module.

Servedeck was forked as `coldstart` and has now been renamed back. The rename
is complete inside this repository; what it cannot reach is the outside world,
which still spells things the old way:

  * ``~/Projects/local_llm/.config`` carries ``COLDSTART_URL=""``, and
    ``codex-qwen.sh``'s own ``CONFIG_ALLOWED_KEYS`` still lists that key. Both
    files are owned by another workstream. Servedeck reads the old spelling as
    a DEPRECATED ALIAS of the new one, preferring the new one when both are
    set, so the two readers of that one file agree during the changeover.
  * ``~/Projects/coldstart/state/`` holds the boot history and the KV
    measurements this box actually accumulated, because the coldstart fork is
    what has been serving :8010. Servedeck reads it and never writes to it.

Every alias in this module is deprecated. The point of collecting them here
rather than scattering ``or cfg.get("COLDSTART_URL")`` through the package is
that dropping compatibility later is one file to read and one grep to trust.

WHY READS ARE MERGED, NOT FALLEN BACK TO. "Read the old file only when the new
one is missing" orphans the old file the instant anything writes a new one:
the first boot after the cutover would make every earlier boot disappear.
Instead both files are read and their records concatenated, de-duplicated on
exact content so that an operator who also copies the old file across (which
docs/MIGRATION.md tells them to do) does not end up counting every historic
boot twice.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import paths

#: Deprecated ``.config`` / environment key -> the name that replaced it.
#: Read-compatibility only: Servedeck prefers the new name wherever both
#: appear, and writes only the name the outside world still requires.
DEPRECATED_KEYS: dict[str, str] = {
    "COLDSTART_URL": "SERVEDECK_URL",
    "USE_COLDSTART": "USE_SERVEDECK",
}

#: Deprecated environment-variable PREFIX. ``COLDSTART_PORT`` is honoured as
#: ``SERVEDECK_PORT`` and so on, so a shell that exported the old names keeps
#: working. Only consulted when the new spelling is unset or empty.
DEPRECATED_ENV_PREFIX = "COLDSTART_"
ENV_PREFIX = "SERVEDECK_"


def env(name: str) -> str | None:
    """``$SERVEDECK_<name>``, falling back to the deprecated ``$COLDSTART_<name>``.

    An empty value counts as unset on BOTH spellings, matching how the rest of
    config.py treats ``""`` — otherwise exporting ``SERVEDECK_PORT=`` would
    shadow a perfectly good ``COLDSTART_PORT`` with nothing.
    """
    value = os.environ.get(ENV_PREFIX + name, "")
    if value:
        return value
    return os.environ.get(DEPRECATED_ENV_PREFIX + name, "") or None


def preferred(mapping: Any, key: str) -> str:
    """The value of ``key``, or of whichever deprecated name aliases it.

    ``key`` may be given as either spelling; the NEW name always wins when
    both are present, so a config file mid-migration is not ambiguous.
    """
    new = DEPRECATED_KEYS.get(key, key)
    old = next((o for o, n in DEPRECATED_KEYS.items() if n == new), None)
    value = (mapping.get(new, "") or "").strip()
    if value:
        return value
    if old is None:
        return ""
    return (mapping.get(old, "") or "").strip()


def legacy_state_dir() -> Path | None:
    """The pre-rename state directory to read alongside the current one.

    ``SERVEDECK_LEGACY_STATE_DIR`` overrides it; setting that to the empty
    string turns the compatibility read OFF, which is what a machine that
    never ran the coldstart fork should do once it is sure. ``None`` means
    "nothing to merge".
    """
    raw = os.environ.get("SERVEDECK_LEGACY_STATE_DIR")
    if raw is not None:
        raw = raw.strip()
        return Path(raw).expanduser() if raw else None
    return paths.LEGACY_COLDSTART_STATE_DIR


def _legacy_file(name: str, *, current: Path) -> Path | None:
    """``<legacy state dir>/<name>``, if it exists and is not the file we are
    already reading. The identity check matters when a test (or an operator)
    points both directories at the same place: merging a file with itself
    would double every record before the de-duplication could help.
    """
    directory = legacy_state_dir()
    if directory is None:
        return None
    candidate = directory / name
    if not candidate.is_file():
        return None
    try:
        if candidate.resolve() == Path(current).resolve():
            return None
    except OSError:
        pass
    return candidate


def merge_jsonl(current: Path, name: str) -> list[dict[str, Any]]:
    """Records from the legacy JSONL file followed by the current one's.

    Legacy first because these logs are chronological and the legacy tree is
    the older one. De-duplicated on the exact serialized record, which is what
    makes copying the old file into the new state directory harmless.
    """
    legacy = _legacy_file(name, current=current)
    if legacy is None:
        return []
    return _read_jsonl(legacy)


def merge_json_list(current: Path, name: str) -> list[Any]:
    """The legacy JSON array for ``name``, or [] when there is nothing to merge."""
    legacy = _legacy_file(name, current=current)
    if legacy is None:
        return []
    try:
        data = json.loads(legacy.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def dedupe(records: list[Any]) -> list[Any]:
    """Order-preserving de-duplication on serialized content.

    ``sort_keys=True`` so that two records differing only in key order — one
    written by the coldstart fork, one by Servedeck — collapse to one.
    """
    seen: set[str] = set()
    out: list[Any] = []
    for record in records:
        try:
            key = json.dumps(record, sort_keys=True, default=str)
        except (TypeError, ValueError):
            key = repr(record)
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Same tolerance as history.load_all(): a torn trailing line is skipped,
    not fatal, and an unreadable file yields nothing rather than raising. A
    legacy file that has gone bad must never take the dashboard down."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out
