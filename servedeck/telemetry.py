"""GPU telemetry sampler — REDESIGN-2026-09-12.md §2.5, packet P6.

Why this module exists, verbatim from the owner: the GPU fell off the bus at
14:25:44 on 2026-09-12 (Xid 79, then Xid 154 "Node Reboot Required"), and when
asked "what was the temperature at that time", the honest answer was that
**nothing on this box recorded it** — no sampler, no log, no kernel thermal
line kept anywhere a human could read it back. That must never be the answer
again.

This module is the sampler half of the fix: an asyncio task that lives inside
the app and appends one small JSON record every ``interval_s`` (default 5) to
a rotating, size-capped, day-bucketed JSONL under
``state/telemetry/YYYY-MM-DD.jsonl``. The Xid-watcher / crash-report half
lives in :mod:`servedeck.xid_watch`, which reads the same directory this
module writes.

Every record's ``ts`` and every day-file's bucket date are UTC (matching
``history.py``'s ``now_iso()`` convention elsewhere in this package) — never
local time, so a box whose timezone changes, or that is read from a different
timezone, never re-derives a different day boundary for the same instant.

Query fields, and why exactly these
------------------------------------
The eleven fields the packet asked for
(``temperature.gpu, power.draw, power.limit, utilization.gpu,
utilization.memory, clocks.sm, clocks.mem, memory.used, memory.free, pstate,
clocks_throttle_reasons.active``) were checked against
``nvidia-smi --help-query-gpu`` on this box (driver 595.84) and every single
one is present — see :func:`detect_supported_fields`. Nothing was dropped
here, but the detection is real (parses the help text, does not hardcode "all
present"), because a driver downgrade or a different box is exactly the
situation this function exists to survive: an unsupported field would make
the WHOLE query line fail on some drivers, not just that field, so silently
querying for something this driver doesn't have is worse than leaving it out.

One quirk worth recording: on this driver, ``clocks_throttle_reasons.active``
is NOT the human-readable comma list ("SW Power Cap, HW Slowdown") some older
docs show — it is a hex bitmask string (``"0x0000000000000000"``). This
module stores it verbatim as a string and does not attempt to decode the
bits; a record that needs decoding still has the raw value to decode from.

Tolerating a GPU that is not there
-----------------------------------
Every failure mode is a valid, forensically useful record, never a skipped
tick and never a raised exception:

- ``nvidia-smi`` missing from PATH -> ``gpu_unavailable`` record, reason
  "nvidia-smi: not found on PATH" (or similar OSError text).
- timeout -> ``gpu_unavailable`` record, reason names the timeout.
- non-zero exit (today's exact case: the GPU falls off the bus mid-run and
  ``nvidia-smi`` exits non-zero with a line like "Unable to determine the
  device handle for GPU 0000:01:00.0: ... No devices were found") ->
  ``gpu_unavailable`` record, reason is the first line of stderr (falling
  back to stdout if stderr is empty).
- a field reported as ``[Not Supported]`` or ``N/A`` -> that one field is
  ``None`` in the record; the rest of the record is still written. This is
  NOT the same as ``gpu_unavailable`` — the query succeeded, one field just
  has nothing to report.

The failed-sample record is never dropped silently: that is the one the
owner's incident needed and did not have.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "REQUESTED_GPU_FIELDS",
    "DEFAULT_INTERVAL_S",
    "DEFAULT_RETENTION_DAYS",
    "DEFAULT_MAX_BYTES",
    "TELEMETRY_DIRNAME",
    "Runner",
    "default_runner",
    "default_state_dir",
    "default_telemetry_dir",
    "detect_supported_fields",
    "take_sample",
    "append_jsonl_capped",
    "prune_old_days",
    "Sampler",
    "last_samples",
    "samples_since",
    "parse_duration",
    "now_iso",
    "read_jsonl",
    "local_tz",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUESTED_GPU_FIELDS: tuple[str, ...] = (
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "utilization.gpu",
    "utilization.memory",
    "clocks.sm",
    "clocks.mem",
    "memory.used",
    "memory.free",
    "pstate",
    "clocks_throttle_reasons.active",
)

#: Record keys, in REQUESTED_GPU_FIELDS order, and whether each is numeric
#: (int/float) or opaque text (pstate, the throttle-reason bitmask).
_NUMERIC_KEYS: dict[str, str] = {
    "temperature.gpu": "temperature_c",
    "power.draw": "power_draw_w",
    "power.limit": "power_limit_w",
    "utilization.gpu": "util_gpu_percent",
    "utilization.memory": "util_memory_percent",
    "clocks.sm": "clocks_sm_mhz",
    "clocks.mem": "clocks_mem_mhz",
    "memory.used": "memory_used_mib",
    "memory.free": "memory_free_mib",
}
_TEXT_KEYS: dict[str, str] = {
    "pstate": "pstate",
    "clocks_throttle_reasons.active": "clocks_throttle_reasons_active",
}
_FLOAT_FIELDS = {"power.draw", "power.limit"}

DEFAULT_INTERVAL_S = 5.0
DEFAULT_RETENTION_DAYS = 14
DEFAULT_MAX_BYTES = 64 * 1024 * 1024  # 64 MiB per day-file

TELEMETRY_DIRNAME = "telemetry"

_NVIDIA_SMI_TIMEOUT_S = 5.0
_NOT_SUPPORTED_MARKERS = {"[not supported]", "n/a", "[n/a]", ""}

_DAY_FILE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.jsonl$")

_WRITE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Runner — injectable subprocess boundary, same shape as gpu.py/units.py
# ---------------------------------------------------------------------------

Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


def default_runner(timeout_s: float = _NVIDIA_SMI_TIMEOUT_S) -> Runner:
    """A :data:`Runner` that actually shells out.

    Never raises: a missing binary or a timeout is turned into a synthetic
    ``CompletedProcess`` with returncode 1 and an explanatory stderr line, so
    every caller has exactly one shape to handle (see :func:`take_sample`) —
    "the process could not even be started" and "the process ran and failed"
    are both just a non-zero-exit CompletedProcess to the rest of this
    module.
    """

    def _run(argv: Sequence[str]) -> "subprocess.CompletedProcess[str]":
        try:
            return subprocess.run(
                list(argv),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except FileNotFoundError:
            return subprocess.CompletedProcess(
                list(argv), 1, "", f"{argv[0]}: not found on PATH"
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                list(argv), 1, "", f"{' '.join(argv)}: timed out after {timeout_s}s"
            )

    return _run


# ---------------------------------------------------------------------------
# State directory
# ---------------------------------------------------------------------------


def default_state_dir() -> Path:
    """The configured state directory, from ``settings.get()``.

    v2 replaced ``config``/``paths`` with ``settings`` (P4); the fallback below
    keeps telemetry recording even if settings cannot load, because a box whose
    configuration is broken is exactly when a GPU fault record matters most.
    """
    try:
        from . import settings as _settings

        return _settings.get().state_dir
    except Exception:  # noqa: BLE001 - a broken config must not silence telemetry
        return Path(__file__).resolve().parent.parent / "state"


def default_telemetry_dir() -> Path:
    return default_state_dir() / TELEMETRY_DIRNAME


# ---------------------------------------------------------------------------
# Field detection — nvidia-smi --help-query-gpu
# ---------------------------------------------------------------------------

# A field-name line in --help-query-gpu's output looks like:
#     "temperature.gpu"
# or, for a field with aliases:
#     "clocks.current.sm" or "clocks.sm"
# Matching every quoted token, not just the first, so an alias-only field
# (the driver documents it only under its alias) is still recognized.
_HELP_FIELD_RE = re.compile(r'"([a-zA-Z0-9_.]+)"')


def _fields_in_help_text(help_text: str) -> set[str]:
    return set(_HELP_FIELD_RE.findall(help_text))


def detect_supported_fields(
    run: Runner | None = None, *, requested: Sequence[str] = REQUESTED_GPU_FIELDS
) -> tuple[list[str], list[str]]:
    """Which of ``requested`` this driver's ``nvidia-smi --help-query-gpu``
    actually documents, in requested order, and which were dropped.

    On any failure to run the help command at all (missing binary, timeout),
    this optimistically returns ``(list(requested), [])`` — detection
    couldn't run, but a real sample attempt will surface the truth anyway
    (an unsupported field usually makes the whole query line fail non-zero,
    which becomes a ``gpu_unavailable`` record, not a silent hang).
    """
    run = run or default_runner()
    proc = run(["nvidia-smi", "--help-query-gpu"])
    if proc.returncode != 0 or not (proc.stdout or "").strip():
        return list(requested), []
    available = _fields_in_help_text(proc.stdout)
    supported = [f for f in requested if f in available]
    dropped = [f for f in requested if f not in available]
    return supported, dropped


# ---------------------------------------------------------------------------
# Sampling — one `nvidia-smi --query-gpu=...` call -> one record
# ---------------------------------------------------------------------------


def now_iso(wall: float | None = None) -> str:
    """UTC ISO-8601 timestamp for ``wall`` (a ``time.time()``-style float),
    or for right now if omitted. Shared by this module and
    :mod:`servedeck.xid_watch`, which stamps its own records the same way."""
    return datetime.fromtimestamp(wall if wall is not None else time.time(), tz=timezone.utc).isoformat()


def local_tz() -> "Any":
    """This process's local timezone, as attached to ``datetime.now().astimezone()``.

    ``journalctl`` (without ``--utc``) prints in the box's local timezone, so
    :mod:`servedeck.xid_watch` uses this to reattach a timezone to the
    year-less, tz-less timestamps in ``-o short-precise`` output. Assumes the
    box parsing the log is the box that wrote it — true for every use in this
    package (nothing here ships a journal to a different machine).
    """
    return datetime.now().astimezone().tzinfo


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _parse_field(field: str, raw: str) -> Any:
    raw = raw.strip()
    if field in _TEXT_KEYS:
        return raw
    if raw.lower() in _NOT_SUPPORTED_MARKERS:
        return None
    try:
        return float(raw) if field in _FLOAT_FIELDS else int(raw)
    except ValueError:
        return None


def take_sample(
    *,
    run: Runner | None = None,
    fields: Sequence[str] = REQUESTED_GPU_FIELDS,
    live_metrics_provider: Callable[[], dict[str, Any] | None] | None = None,
    wall: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """One telemetry record: wall-clock ISO timestamp, one
    ``nvidia-smi --query-gpu=...`` call's worth of fields (or a
    ``gpu_unavailable`` record explaining why not), plus whatever
    ``live_metrics_provider()`` returns (or ``None`` if there is none / no
    model is live) under ``"live_metrics"``.

    Never raises. This is the function :class:`Sampler` calls once per tick,
    and the function the tests exercise directly without any asyncio
    involved.
    """
    run = run or default_runner()
    now = wall()
    ts = now_iso(now)
    live_metrics = None
    if live_metrics_provider is not None:
        try:
            live_metrics = live_metrics_provider()
        except Exception as exc:  # noqa: BLE001 - a metrics glitch must not cost the sample
            live_metrics = {"error": f"live_metrics_provider raised: {exc!r}"}

    argv = ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]
    proc = run(argv)
    if proc.returncode != 0:
        reason = _first_line(proc.stderr) or _first_line(proc.stdout) or f"exit {proc.returncode}"
        return {
            "ts": ts,
            "gpu_unavailable": True,
            "error": reason,
            "live_metrics": live_metrics,
        }

    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if not lines:
        return {
            "ts": ts,
            "gpu_unavailable": True,
            "error": "nvidia-smi exited 0 but produced no output",
            "live_metrics": live_metrics,
        }

    try:
        # skipinitialspace=True: nvidia-smi's CSV rows put a space after every
        # comma (including before an opening quote), which plain csv.reader
        # treats as part of the field -- "SW Power Cap, HW Slowdown" would
        # then parse as two fields ('"SW Power Cap', ' HW Slowdown"') instead
        # of one quoted one.
        row = next(csv.reader(io.StringIO(lines[0]), skipinitialspace=True))
    except csv.Error as exc:
        return {
            "ts": ts,
            "gpu_unavailable": True,
            "error": f"could not parse nvidia-smi CSV output: {exc}",
            "live_metrics": live_metrics,
        }

    if len(row) != len(fields):
        return {
            "ts": ts,
            "gpu_unavailable": True,
            "error": (
                f"nvidia-smi returned {len(row)} field(s), expected {len(fields)} "
                f"for query {','.join(fields)!r}: {lines[0]!r}"
            ),
            "live_metrics": live_metrics,
        }

    record: dict[str, Any] = {"ts": ts, "gpu_unavailable": False, "live_metrics": live_metrics}
    for field, raw in zip(fields, row):
        key = _NUMERIC_KEYS.get(field) or _TEXT_KEYS.get(field) or field
        record[key] = _parse_field(field, raw)
    return record


# ---------------------------------------------------------------------------
# JSONL append with a size cap, and day-file pruning
# ---------------------------------------------------------------------------


def append_jsonl_capped(path: Path, record: dict[str, Any], max_bytes: int) -> bool:
    """Append ``record`` as one JSON line, unless ``path`` is already at or
    over ``max_bytes`` — then write ONE truncation-marker line (if the file's
    last line isn't already one) and return ``False`` without growing the
    file further. Growth is therefore bounded at roughly ``max_bytes`` plus
    one marker line, forever, never unbounded.

    Returns ``True`` iff ``record`` itself was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        size = path.stat().st_size if path.is_file() else 0
        if size >= max_bytes:
            if not _last_line_is_truncation_marker(path):
                marker = {
                    "ts": record.get("ts", now_iso()),
                    "telemetry_truncated": True,
                    "reason": f"day file reached the {max_bytes}-byte cap; further samples today are dropped",
                }
                _raw_append(path, marker)
            return False
        _raw_append(path, record)
        return True


def _raw_append(path: Path, record: dict[str, Any]) -> None:
    line = json.dumps(record, sort_keys=False) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def _last_line_is_truncation_marker(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        # The marker line is short; reading the last 4 KiB is comfortably
        # enough without loading a potentially 64 MiB file to check one flag.
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 4096))
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return False
        return bool(obj.get("telemetry_truncated"))
    return False


def prune_old_days(directory: Path, retention_days: int, *, today: "datetime | None" = None) -> list[Path]:
    """Delete ``YYYY-MM-DD.jsonl`` files in ``directory`` older than
    ``retention_days`` days before ``today`` (UTC, if not given). Files that
    don't match the day-file name pattern (``xid.jsonl``, anything else) are
    left alone — this only prunes telemetry day-files. Missing directory is a
    no-op. Returns the paths actually deleted.
    """
    if not directory.is_dir():
        return []
    today_date = (today or datetime.now(timezone.utc)).date()
    cutoff = today_date - timedelta(days=retention_days)
    deleted: list[Path] = []
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        m = _DAY_FILE_RE.match(entry.name)
        if not m:
            continue
        try:
            day = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            continue
        if day < cutoff:
            entry.unlink(missing_ok=True)
            deleted.append(entry)
    return deleted


# ---------------------------------------------------------------------------
# Sampler — the asyncio task
# ---------------------------------------------------------------------------


class Sampler:
    """Runs inside the app as an asyncio task; every ``interval_s`` seconds it
    takes one sample (:func:`take_sample`) and appends it
    (:func:`append_jsonl_capped`) to today's day-file under
    ``state_dir/telemetry/``, pruning day-files older than ``retention_days``
    whenever the day rolls over.

    ``sample_once()`` is synchronous and does the real work; ``run_forever()``
    is a thin asyncio loop around it (via ``asyncio.to_thread`` so the
    blocking ``nvidia-smi`` subprocess call never stalls the event loop), so
    tests exercise the former directly and never need an event loop unless
    they are specifically testing the loop/cancellation plumbing.
    """

    def __init__(
        self,
        *,
        state_dir: Path | None = None,
        interval_s: float = DEFAULT_INTERVAL_S,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        max_bytes: int = DEFAULT_MAX_BYTES,
        run: Runner | None = None,
        fields: Sequence[str] | None = None,
        live_metrics_provider: Callable[[], dict[str, Any] | None] | None = None,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self.state_dir = state_dir if state_dir is not None else default_state_dir()
        self.interval_s = interval_s
        self.retention_days = retention_days
        self.max_bytes = max_bytes
        self._run = run or default_runner()
        if fields is None:
            supported, dropped = detect_supported_fields(self._run)
            self.dropped_fields = dropped
            fields = supported
        else:
            self.dropped_fields = []
        self.fields = list(fields)
        self._live_metrics_provider = live_metrics_provider
        self._wall = wall
        self._known_days: set[str] = set()
        self._task: "Any | None" = None

    @property
    def telemetry_dir(self) -> Path:
        return self.state_dir / TELEMETRY_DIRNAME

    def _path_for_day(self, day: str) -> Path:
        return self.telemetry_dir / f"{day}.jsonl"

    def sample_once(self) -> dict[str, Any]:
        """Take one sample, write it, return it. No sleeping."""
        record = take_sample(
            run=self._run,
            fields=self.fields,
            live_metrics_provider=self._live_metrics_provider,
            wall=self._wall,
        )
        day = record["ts"][:10]
        if day not in self._known_days:
            prune_old_days(self.telemetry_dir, self.retention_days)
            self._known_days.add(day)
        append_jsonl_capped(self._path_for_day(day), record, self.max_bytes)
        return record

    async def run_forever(self) -> None:
        """The asyncio task body. Runs until cancelled; a single bad sample
        never breaks the loop because ``sample_once`` (via ``take_sample``)
        never raises."""
        import asyncio

        while True:
            await asyncio.to_thread(self.sample_once)
            await asyncio.sleep(self.interval_s)

    def start(self) -> "Any":
        """Create and return the asyncio task. The caller (app.py's startup
        hook) owns the returned Task and is responsible for cancelling it on
        shutdown — see :meth:`stop`."""
        import asyncio

        self._task = asyncio.create_task(self.run_forever(), name="gpu-telemetry-sampler")
        return self._task

    async def stop(self) -> None:
        """Cancel the task started by :meth:`start` and wait for it to
        actually finish. A no-op if :meth:`start` was never called."""
        import asyncio

        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Surfacing — read-only API for the app / CLI to build on
# ---------------------------------------------------------------------------


def _day_files_newest_first(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    days = []
    for entry in directory.iterdir():
        if entry.is_file() and _DAY_FILE_RE.match(entry.name):
            days.append(entry)
    return sorted(days, key=lambda p: p.name, reverse=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn trailing line from an interrupted write
    return out


def last_samples(n: int = 12, *, state_dir: Path | None = None) -> list[dict[str, Any]]:
    """The last ``n`` telemetry records, oldest first (chronological), reading
    backwards from today's day-file into yesterday's (and further back, if
    ``n`` is large) as needed. ``[]`` if nothing has been sampled yet. Never
    raises.

    For ``GET /api/state``: call this with a small ``n`` (e.g. 1-12) to show
    the current reading and a short recent trend.
    """
    directory = (state_dir if state_dir is not None else default_state_dir()) / TELEMETRY_DIRNAME
    collected: list[dict[str, Any]] = []
    for path in _day_files_newest_first(directory):
        records = read_jsonl(path)
        collected = records + collected
        if len(collected) >= n:
            break
    return collected[-n:] if n > 0 else []


def parse_duration(spec: str) -> timedelta:
    """Parse a duration like ``"10m"``, ``"1h"``, ``"30s"``, ``"2d"``, or a
    bare integer (seconds). Raises ``ValueError`` on anything else — the CLI
    is the only caller and should report that to the user, not swallow it.
    """
    spec = spec.strip().lower()
    m = re.fullmatch(r"(\d+)\s*([smhd]?)", spec)
    if not m:
        raise ValueError(f"not a duration: {spec!r} (expected e.g. '10m', '1h', '30s', '2d')")
    n = int(m.group(1))
    unit = m.group(2) or "s"
    seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return timedelta(seconds=n * seconds)


def samples_since(since: str, *, state_dir: Path | None = None, tail: int | None = None) -> list[dict[str, Any]]:
    """Telemetry records with ``ts`` within ``since`` (a :func:`parse_duration`
    string) of now, oldest first, optionally capped to the last ``tail`` of
    them. Used by ``servedeck gpu-log``.
    """
    cutoff = datetime.now(timezone.utc) - parse_duration(since)
    directory = (state_dir if state_dir is not None else default_state_dir()) / TELEMETRY_DIRNAME
    out: list[dict[str, Any]] = []
    # Walk day-files oldest-first among those that could possibly overlap the
    # window (today, and yesterday in case the window crosses midnight UTC).
    candidates = sorted(_day_files_newest_first(directory), key=lambda p: p.name)
    for path in candidates:
        for rec in read_jsonl(path):
            ts = rec.get("ts")
            if not isinstance(ts, str):
                continue
            try:
                rec_dt = datetime.fromisoformat(ts)
            except ValueError:
                continue
            if rec_dt >= cutoff:
                out.append(rec)
    if tail is not None and tail > 0:
        out = out[-tail:]
    return out
