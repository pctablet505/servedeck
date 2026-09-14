"""Xid watcher and crash-report writer — REDESIGN-2026-09-12.md §2.5, packet P6.

Companion to :mod:`servedeck.telemetry` (which owns the periodic GPU
sampler); this module owns the other half of "never again say nothing
recorded the temperature": watching the kernel journal for NVRM Xid lines,
recording each one, and — for a FATAL one — bundling everything a human would
want five minutes later into one crash report file.

A surprising thing found while building this, worth knowing before touching
this module again
------------------------------------------------------------------------------
``journalctl -k`` (``--dmesg``) **implies ``--boot=0``** (the current boot)
*unless a `-b`/`--boot` is given explicitly* — this is documented (``man
journalctl``: "-k, --dmesg: ... This implies --boot=0 unless explicitly
specified otherwise"), not a bug, but it is exactly the trap this module must
not fall into: the box in front of this packet rebooted between the fault
(2026-09-12 14:25:44) and this code being written, so

    journalctl -k --since "2026-09-12 14:25:40" --until "2026-09-12 14:26:20"

(the plain form, and the form ``servedeck/gpu.py``'s existing ``xid_events()``
uses) returns **nothing** on this box right now — it silently searches only
the CURRENT boot's kernel ring, which starts at 16:23:05, well after the
fault. The exact same query with ``-b all`` inserted finds both lines
immediately. Since a FATAL Xid (79/154) is often exactly the kind of event
that causes a reboot, a watcher that does not pass ``-b all`` would reliably
fail to see the one event class it most needs to see, the instant the reboot
it warned about actually happens. Every ``journalctl`` call in this module
therefore passes ``-b all`` explicitly. (``servedeck/gpu.py``'s
``xid_events()`` has the same gap and was not touched here — it belongs to
another packet.)

Record shapes
-------------
``state/telemetry/xid.jsonl`` — one line per Xid or companion line, oldest
first::

    {"ts": "<UTC ISO8601>", "raw_ts": "Sep 12 14:25:44.984246",
     "code": 79, "raw_line": "<the full journal line, verbatim>",
     "severity": "fatal", "needs_reboot": true, "note": "..."}

``state/crash_reports/<anchor-ts>.json`` — one file per fault, never
overwritten::

    {"written_at": "<UTC ISO8601>", "anchor_ts": "<UTC ISO8601 of the fatal event>",
     "xid_events": [...],                 # the Xid/companion lines from that scan
     "telemetry_window_60s": [...],       # last 60s of GPU telemetry samples
     "live_models": [{"unit": "model-flashnext", "state": {...},
                       "journal_tail": ["...", ...]}, ...]}
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import telemetry as _telemetry
from . import units as _units
from .doctor import CheckResult

__all__ = [
    "CRASH_REPORTS_DIRNAME",
    "classify_xid",
    "parse_xid_lines",
    "default_xid_log_path",
    "default_crash_reports_dir",
    "record_xid_events",
    "build_crash_report",
    "write_crash_report",
    "XidWatcher",
    "recent_xids",
    "last_crash_report",
    "check_fatal_xid_needs_reboot",
]

CRASH_REPORTS_DIRNAME = "crash_reports"

_JOURNALCTL_TIMEOUT_S = 10.0
_DEFAULT_BOOTSTRAP_WINDOW = "24h"
#: How many recent telemetry samples to fetch before filtering to the actual
#: 60s window around a fault. At the default 5s interval this is ~25 minutes
#: of headroom -- generous on purpose, since a slower Sampler interval must
#: not silently shrink the crash report's window.
_CRASH_REPORT_SAMPLE_LOOKBACK = 720

# ---------------------------------------------------------------------------
# Xid line parsing and classification
# ---------------------------------------------------------------------------

# "NVRM: Xid (PCI:0000:01:00): 79, pid=552867, name=nvidia-smi, GPU has ..."
_XID_RE = re.compile(r"NVRM:\s*Xid\s*\(PCI:[0-9a-fA-F:.]+\):\s*(\d+)")

#: Companion lines that carry no Xid number of their own but are direct
#: evidence of the same fault (per the packet: "GPU has fallen off the bus" /
#: "GPU recovery action changed"). Matched only when `_XID_RE` did NOT already
#: match the line -- the recovery-action line in practice carries its own Xid
#: number (154) and is handled by the branch above; this branch exists for
#: the plain "NVRM: GPU 0000:01:00.0: GPU has fallen off the bus." line that
#: has no "Xid (PCI:...)" clause at all.
_COMPANION_PHRASES: tuple[str, ...] = (
    "GPU has fallen off the bus",
    "GPU recovery action changed",
)

# journalctl -o short-precise: "Sep 12 14:25:44.984246 host kernel: ...".
# No year -- the caller supplies one (see parse_xid_lines).
_TS_RE = re.compile(r"^(?P<mon>[A-Za-z]{3})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s")

#: code -> (severity, needs_reboot, note). Per the packet's own taxonomy --
#: deliberately NOT the restart-policy taxonomy in gpu.py's
#: `_classify_xid_code` (mmu_fault/off_bus/reboot_required/unrecognized),
#: which answers "should the supervisor restart the process?". This one
#: answers "how bad is this for a human reading a crash report?", and the
#: packet asked for these four buckets specifically.
_SEVERITY_BY_CODE: dict[int, tuple[str, bool, str]] = {
    79: ("fatal", True, "GPU has fallen off the bus. Hardware/driver-level; needs a reboot."),
    154: ("fatal", True, "Driver GPU-recovery action asserted (\"Node Reboot Required\"); needs a reboot."),
    13: ("channel", False, "Channel / illegal-access fault (GPU MMU fault, illegal or misaligned address)."),
    31: ("channel", False, "Channel / illegal-access fault (GPU MMU fault, illegal or misaligned address)."),
    43: ("app_level", False, "Application-level GPU error (e.g. the GPU stopped processing a context)."),
    45: ("app_level", False, "Application-level GPU error (preemptive context cleanup after a prior error)."),
}


def classify_xid(code: int | None) -> tuple[str, bool, str]:
    """(severity, needs_reboot, note) for an Xid ``code``.

    ``severity`` is one of ``"fatal"`` / ``"channel"`` / ``"app_level"`` /
    ``"unknown"``. ``code=None`` (a companion line with no numeric Xid of its
    own) and any numeric code this table doesn't recognize both classify as
    ``"unknown"`` -- per the packet: "unknown = report as unknown", never a
    guess.
    """
    if code is None:
        return "unknown", False, "no numeric Xid code on this line (a companion line only)."
    if code in _SEVERITY_BY_CODE:
        return _SEVERITY_BY_CODE[code]
    return "unknown", False, f"Xid {code} is not one of the codes this classifier recognizes; look it up."


def _parse_short_precise_ts(line: str, *, year: int, tzinfo: Any) -> tuple[datetime | None, str | None]:
    m = _TS_RE.match(line)
    if not m:
        return None, None
    raw_ts = m.group(0).strip()
    text = f"{m['mon']} {m['day']} {year} {m['time']}"
    for fmt in ("%b %d %Y %H:%M:%S.%f", "%b %d %Y %H:%M:%S"):
        try:
            naive = datetime.strptime(text, fmt)
            break
        except ValueError:
            continue
    else:
        return None, raw_ts
    return naive.replace(tzinfo=tzinfo).astimezone(timezone.utc), raw_ts


def parse_xid_lines(
    text: str, *, year: int | None = None, tzinfo: Any = None
) -> list[dict[str, Any]]:
    """Every Xid / companion line in ``text`` (raw ``journalctl -o
    short-precise`` output, or any text shaped like it), classified, deduped
    by ``(timestamp, raw line)``, oldest first.

    ``year``/``tzinfo`` are injectable so this is a pure, deterministic
    function to test against a real fixture without depending on wall-clock
    "now" — real callers (:class:`XidWatcher`) default them to the current
    year and :func:`servedeck.telemetry.local_tz`.

    Never raises: an unparseable timestamp yields ``ts: None`` on that one
    event rather than dropping the line — a raw Xid line with no usable
    timestamp is still evidence worth keeping.
    """
    year = year if year is not None else datetime.now().year
    tzinfo = tzinfo if tzinfo is not None else _telemetry.local_tz()

    events: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for line in text.splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        m = _XID_RE.search(line)
        code: int | None = None
        if m:
            code = int(m.group(1))
        elif not any(phrase in line for phrase in _COMPANION_PHRASES):
            continue  # not an Xid-related line

        ts_dt, raw_ts = _parse_short_precise_ts(line, year=year, tzinfo=tzinfo)
        ts_iso = ts_dt.isoformat() if ts_dt is not None else None

        dedupe_key = (ts_iso or "", line)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        severity, needs_reboot, note = classify_xid(code)
        events.append(
            {
                "ts": ts_iso,
                "raw_ts": raw_ts,
                "code": code,
                "raw_line": line,
                "severity": severity,
                "needs_reboot": needs_reboot,
                "note": note,
            }
        )
    return events


# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------


def default_xid_log_path(*, state_dir: Path | None = None) -> Path:
    base = state_dir if state_dir is not None else _telemetry.default_state_dir()
    return base / _telemetry.TELEMETRY_DIRNAME / "xid.jsonl"


def default_crash_reports_dir(*, state_dir: Path | None = None) -> Path:
    base = state_dir if state_dir is not None else _telemetry.default_state_dir()
    return base / CRASH_REPORTS_DIRNAME


def _append_line(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=False) + "\n")
        fh.flush()


def record_xid_events(events: Sequence[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    """Append the events in ``events`` that are not already present in
    ``path`` (by ``(ts, raw_line)``), return the ones actually written.

    A second layer of the same dedupe :func:`parse_xid_lines` already does —
    belt and braces against two overlapping scan windows re-parsing the same
    journal lines.
    """
    existing = {(e.get("ts"), e.get("raw_line")) for e in _telemetry.read_jsonl(path)}
    written: list[dict[str, Any]] = []
    for event in events:
        key = (event.get("ts"), event.get("raw_line"))
        if key in existing:
            continue
        existing.add(key)
        _append_line(path, event)
        written.append(event)
    return written


# ---------------------------------------------------------------------------
# Crash report
# ---------------------------------------------------------------------------


def build_crash_report(
    trigger_events: Sequence[dict[str, Any]],
    *,
    state_dir: Path | None = None,
    units_runner: _units.Runner | None = None,
) -> dict[str, Any]:
    """Assemble a crash report from the Xid events that triggered it: the
    last 60s of telemetry, which models were live and their unit states, and
    the last 40 journal lines of each. Never raises -- every sub-lookup is
    wrapped so one flaky piece (e.g. ``systemctl`` briefly unavailable)
    cannot cost the rest of the report.
    """
    fatal_ts = sorted(e["ts"] for e in trigger_events if e.get("severity") == "fatal" and e.get("ts"))
    anchor_ts = fatal_ts[-1] if fatal_ts else None

    window_samples: list[dict[str, Any]] = []
    if anchor_ts is not None:
        anchor_dt = datetime.fromisoformat(anchor_ts)
        for sample in _telemetry.last_samples(_CRASH_REPORT_SAMPLE_LOOKBACK, state_dir=state_dir):
            ts = sample.get("ts")
            if not isinstance(ts, str):
                continue
            try:
                sample_dt = datetime.fromisoformat(ts)
            except ValueError:
                continue
            delta = (anchor_dt - sample_dt).total_seconds()
            if 0 <= delta <= 60:
                window_samples.append(sample)

    live_models: list[dict[str, Any]] = []
    try:
        unit_names = _units.list_model_units(run=units_runner)
    except _units.UnitError as exc:
        unit_names = []
        live_models.append({"error": f"could not list model-* units: {exc}"})
    for unit_name in unit_names:
        try:
            state = _units.show(unit_name, run=units_runner)
            state_dict: dict[str, Any] = {
                "active_state": state.active_state,
                "sub_state": state.sub_state,
                "result": state.result,
                "n_restarts": state.n_restarts,
                "main_pid": state.main_pid,
                "exec_main_start_ts": state.exec_main_start_ts,
            }
        except _units.UnitError as exc:
            state_dict = {"error": str(exc)}
        try:
            tail_lines = _units.journal_tail(unit_name, lines=40, run=units_runner)
        except _units.UnitError as exc:
            tail_lines = [f"<could not read journal for {unit_name}: {exc}>"]
        live_models.append({"unit": unit_name, "state": state_dict, "journal_tail": tail_lines})

    return {
        "written_at": _telemetry.now_iso(),
        "anchor_ts": anchor_ts,
        "xid_events": list(trigger_events),
        "telemetry_window_60s": window_samples,
        "live_models": live_models,
    }


def write_crash_report(report: dict[str, Any], *, state_dir: Path | None = None) -> Path:
    """Write ``report`` to ``state/crash_reports/<anchor-ts>.json``. One file
    per fault, never overwritten: a filename collision (two reports with the
    exact same anchor timestamp — should not happen given the events are
    already deduped by timestamp) gets a ``-2``, ``-3``, ... suffix rather
    than clobbering the earlier file.
    """
    directory = default_crash_reports_dir(state_dir=state_dir)
    directory.mkdir(parents=True, exist_ok=True)
    anchor = report.get("anchor_ts") or report.get("written_at") or _telemetry.now_iso()
    safe = anchor.replace(":", "-")
    path = directory / f"{safe}.json"
    n = 2
    while path.exists():
        path = directory / f"{safe}-{n}.json"
        n += 1
    path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------


def _journalctl_since_str(dt_utc: datetime, *, tzinfo: Any) -> str:
    return dt_utc.astimezone(tzinfo).strftime("%Y-%m-%d %H:%M:%S")


class XidWatcher:
    """Periodically scans the kernel journal for new Xid/companion lines,
    records them, and writes a crash report the moment a fatal one appears.

    Like :class:`servedeck.telemetry.Sampler`, ``scan_once()`` is the
    synchronous unit of work tests exercise directly; ``run_forever()`` is a
    thin asyncio loop around it.
    """

    def __init__(
        self,
        *,
        state_dir: Path | None = None,
        run: _telemetry.Runner | None = None,
        units_runner: _units.Runner | None = None,
        bootstrap_window: str = _DEFAULT_BOOTSTRAP_WINDOW,
    ) -> None:
        self.state_dir = state_dir if state_dir is not None else _telemetry.default_state_dir()
        self._run = run or _telemetry.default_runner(_JOURNALCTL_TIMEOUT_S)
        self._units_runner = units_runner
        self.bootstrap_window = bootstrap_window
        self._task: Any | None = None

    def xid_log_path(self) -> Path:
        return default_xid_log_path(state_dir=self.state_dir)

    def _last_seen_ts(self) -> str | None:
        events = _telemetry.read_jsonl(self.xid_log_path())
        ts_values = [e.get("ts") for e in events if isinstance(e.get("ts"), str)]
        return max(ts_values) if ts_values else None

    def scan_once(self) -> list[dict[str, Any]]:
        """One poll: run journalctl, parse, record new events, write a crash
        report for any newly-seen fatal one. Returns the newly recorded
        events (possibly empty). Never raises: an unreachable journalctl is
        the same as "no new evidence this tick", not an error.
        """
        tzinfo = _telemetry.local_tz()
        last_seen_iso = self._last_seen_ts()
        if last_seen_iso is not None:
            since_dt = datetime.fromisoformat(last_seen_iso) - timedelta(seconds=1)
        else:
            since_dt = datetime.now(timezone.utc) - _telemetry.parse_duration(self.bootstrap_window)
        since_str = _journalctl_since_str(since_dt, tzinfo=tzinfo)

        proc = self._run(
            ["journalctl", "-k", "-b", "all", "--no-pager", "-o", "short-precise", "--since", since_str]
        )
        if proc.returncode != 0:
            return []

        events = parse_xid_lines(proc.stdout or "", year=datetime.now().year, tzinfo=tzinfo)
        if last_seen_iso is not None:
            events = [e for e in events if e.get("ts") and e["ts"] > last_seen_iso]

        new_events = record_xid_events(events, self.xid_log_path())
        fatal = [e for e in new_events if e.get("severity") == "fatal"]
        if fatal:
            report = build_crash_report(new_events, state_dir=self.state_dir, units_runner=self._units_runner)
            write_crash_report(report, state_dir=self.state_dir)
        return new_events

    async def run_forever(self, poll_s: float = 15.0) -> None:
        import asyncio

        while True:
            await asyncio.to_thread(self.scan_once)
            await asyncio.sleep(poll_s)

    def start(self, poll_s: float = 15.0) -> Any:
        import asyncio

        self._task = asyncio.create_task(self.run_forever(poll_s), name="gpu-xid-watcher")
        return self._task

    async def stop(self) -> None:
        import asyncio

        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Surfacing — read-only API for the app / CLI / doctor to build on
# ---------------------------------------------------------------------------


def recent_xids(n: int = 20, *, state_dir: Path | None = None) -> list[dict[str, Any]]:
    """The ``n`` most recently recorded Xid/companion events, NEWEST FIRST
    (most relevant on top, for an alerts panel). ``[]`` if none recorded.
    Never raises.
    """
    events = _telemetry.read_jsonl(default_xid_log_path(state_dir=state_dir))
    events.sort(key=lambda e: e.get("ts") or "")
    if n <= 0:
        return []
    return list(reversed(events[-n:]))


def last_crash_report(*, state_dir: Path | None = None) -> dict[str, Any] | None:
    """The most recently written crash report, or ``None`` if none exist.
    Never raises."""
    directory = default_crash_reports_dir(state_dir=state_dir)
    if not directory.is_dir():
        return None
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix == ".json")
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Doctor check
# ---------------------------------------------------------------------------


def _current_boot_time() -> datetime | None:
    """``btime`` out of ``/proc/stat`` — the kernel's own record of when this
    boot started, in seconds since the epoch. ``None`` off Linux or if the
    file is unreadable; never raises."""
    try:
        with open("/proc/stat", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("btime "):
                    return datetime.fromtimestamp(int(line.split()[1]), tz=timezone.utc)
    except (OSError, ValueError, IndexError):
        return None
    return None


def check_fatal_xid_needs_reboot(
    *, state_dir: Path | None = None, boot_time: datetime | None = None
) -> CheckResult:
    """One :class:`servedeck.doctor.CheckResult`: fails (``ok=False``) iff the
    newest crash report's timestamp is AFTER the current boot's start time —
    i.e. a fatal Xid was recorded during the boot session that is still
    running right now, so the reboot it called for has not happened yet.
    Once the box reboots, the new boot's start time overtakes the old crash
    report's timestamp and this goes quiet on its own.

    Not owned by ``doctor.py`` — see the packet's instructions; P4 wires this
    into ``run_doctor()``'s result list.
    """
    name = "gpu fatal xid"
    report = last_crash_report(state_dir=state_dir)
    if report is None:
        return CheckResult(name, True, "no crash reports recorded")

    anchor = report.get("anchor_ts") or report.get("written_at")
    if not anchor:
        return CheckResult(name, True, "latest crash report has no usable timestamp")
    try:
        report_dt = datetime.fromisoformat(anchor)
    except ValueError:
        return CheckResult(name, True, f"latest crash report timestamp {anchor!r} is unparseable")

    bt = boot_time if boot_time is not None else _current_boot_time()
    if bt is None:
        return CheckResult(
            name,
            True,
            f"a fatal Xid was recorded at {anchor}; could not determine this boot's start "
            "time to check whether a reboot already happened",
        )
    if report_dt > bt:
        return CheckResult(name, False, f"a fatal Xid was recorded at {anchor}; the card needs a reboot")
    return CheckResult(
        name,
        True,
        f"latest fatal Xid ({anchor}) predates the current boot ({bt.isoformat()}); already recovered",
    )
