"""servedeck.xid_watch — Xid parsing/classification, the watcher, and the
crash-report writer.

The fixture in tests/fixtures/xid-2026-09-12.txt is copied VERBATIM from this
box's real journal (`journalctl -k -b all -o short-precise --since
"2026-09-12 14:25:40" --until "2026-09-12 14:26:20"`), covering the real
fault: the GPU fell off the bus at 14:25:44 IST (Xid 79), followed
immediately by Xid 154 ("Node Reboot Required"). Every assertion against it
must derive its expected values from the real lines, not from anything typed
by hand.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from servedeck import xid_watch

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "xid-2026-09-12.txt"
# The box that captured the fixture runs IST (UTC+5:30) -- see
# servedeck/xid_watch.py's module docstring for why parse_xid_lines takes
# tzinfo explicitly rather than reading the CURRENT machine's timezone: this
# keeps the fixture test deterministic regardless of what timezone the box
# running the test suite happens to be in.
IST = timezone(timedelta(hours=5, minutes=30))


class FakeRunner:
    def __init__(self, *results: subprocess.CompletedProcess) -> None:
        self.calls: list[list[str]] = []
        self._results = list(results)

    def __call__(self, argv):
        self.calls.append(list(argv))
        if len(self._results) > 1:
            return self._results.pop(0)
        return self._results[0] if self._results else subprocess.CompletedProcess(list(argv), 0, "", "")


def ok(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["journalctl"], 0, stdout, "")


def fail(code: int = 1, stderr: str = "boom") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["journalctl"], code, "", stderr)


class FakeUnitsRunner:
    """Scripts the three systemctl/journalctl shapes build_crash_report()
    needs: list-units, show, journal tail -- see servedeck/units.py's own
    argv builders for the exact shapes being matched here."""

    def __init__(self, unit_line: str = "model-flashnext.service loaded active running Flash-Next\n"):
        self.unit_line = unit_line
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        if argv[:3] == ["systemctl", "--user", "list-units"]:
            return subprocess.CompletedProcess(argv, 0, self.unit_line, "")
        if argv[:3] == ["systemctl", "--user", "show"]:
            props = (
                "ActiveState=active\nSubState=running\nResult=success\n"
                "NRestarts=0\nMainPID=12345\nExecMainStartTimestamp=Fri 2026-09-12 14:00:00 IST\n"
            )
            return subprocess.CompletedProcess(argv, 0, props, "")
        if argv[:1] == ["journalctl"]:
            return subprocess.CompletedProcess(argv, 0, "log line 1\nlog line 2\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")


# --------------------------------------------------------------------------
# classify_xid
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code,expected_severity,expected_needs_reboot",
    [
        (79, "fatal", True),
        (154, "fatal", True),
        (13, "channel", False),
        (31, "channel", False),
        (43, "app_level", False),
        (45, "app_level", False),
        (999, "unknown", False),
        (None, "unknown", False),
    ],
)
def test_classify_xid(code, expected_severity, expected_needs_reboot):
    severity, needs_reboot, note = xid_watch.classify_xid(code)
    assert severity == expected_severity
    assert needs_reboot == expected_needs_reboot
    assert note  # never blank -- always something to read


# --------------------------------------------------------------------------
# parse_xid_lines against the REAL fixture
# --------------------------------------------------------------------------


def test_real_fixture_finds_xid_79_and_154_as_fatal_with_correct_timestamps():
    text = FIXTURE_PATH.read_text(encoding="utf-8")
    events = xid_watch.parse_xid_lines(text, year=2026, tzinfo=IST)

    by_code = {e["code"]: e for e in events if e["code"] is not None}
    assert 79 in by_code, "must find the real Xid 79 line (GPU fell off the bus)"
    assert 154 in by_code, "must find the real Xid 154 line (Node Reboot Required)"

    for code in (79, 154):
        assert by_code[code]["severity"] == "fatal"
        assert by_code[code]["needs_reboot"] is True

    # The real journal line's own local timestamp, "Sep 12 14:25:44.984246"
    # / "...984940", converted from IST to UTC (the module's own convention).
    expected_79 = datetime(2026, 9, 12, 14, 25, 44, 984246, tzinfo=IST).astimezone(timezone.utc).isoformat()
    expected_154 = datetime(2026, 9, 12, 14, 25, 44, 984940, tzinfo=IST).astimezone(timezone.utc).isoformat()
    assert by_code[79]["ts"] == expected_79
    assert by_code[154]["ts"] == expected_154

    # 154 happened a fraction of a second after 79, in that order.
    assert by_code[79]["ts"] < by_code[154]["ts"]

    # The raw line is preserved verbatim, PCI address and all.
    assert "NVRM: Xid (PCI:0000:01:00): 79" in by_code[79]["raw_line"]
    assert "NVRM: Xid (PCI:0000:01:00): 154" in by_code[154]["raw_line"]
    assert "GPU has fallen off the bus" in by_code[79]["raw_line"]
    assert "Node Reboot Required" in by_code[154]["raw_line"]


def test_real_fixture_also_catches_the_codeless_companion_line():
    """The plain 'NVRM: GPU 0000:01:00.0: GPU has fallen off the bus.' line
    carries no Xid number of its own but is real evidence of the same fault
    -- it must be captured too, classified unknown (no code), not dropped."""
    text = FIXTURE_PATH.read_text(encoding="utf-8")
    events = xid_watch.parse_xid_lines(text, year=2026, tzinfo=IST)
    codeless = [e for e in events if e["code"] is None]
    assert codeless, "the codeless 'fallen off the bus' companion line must still be recorded"
    assert any("GPU has fallen off the bus" in e["raw_line"] for e in codeless)
    assert all(e["severity"] == "unknown" for e in codeless)


def test_real_fixture_does_not_pick_up_unrelated_noise():
    """The fixture's ~3500 lines are mostly unrelated RPC-retry spam from the
    same crash; the parser must not treat every NVRM line as an event."""
    text = FIXTURE_PATH.read_text(encoding="utf-8")
    total_nvrm_lines = sum(1 for ln in text.splitlines() if "NVRM:" in ln)
    events = xid_watch.parse_xid_lines(text, year=2026, tzinfo=IST)
    assert total_nvrm_lines > 100  # sanity: the fixture really is noisy
    assert len(events) < 10  # but only the real Xid/companion lines are events


# --------------------------------------------------------------------------
# parse_xid_lines: dedupe and synthetic edge cases
# --------------------------------------------------------------------------


def test_parse_xid_lines_dedupes_identical_repeated_lines():
    line = "Sep 12 14:25:44.984246 host kernel: NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.\n"
    events = xid_watch.parse_xid_lines(line + line, year=2026, tzinfo=IST)
    assert len(events) == 1


def test_parse_xid_lines_ignores_lines_with_no_xid_and_no_companion_phrase():
    text = "Sep 12 14:25:44.000000 host kernel: some unrelated line\n"
    assert xid_watch.parse_xid_lines(text, year=2026, tzinfo=IST) == []


def test_parse_xid_lines_unparseable_timestamp_still_kept():
    text = "no timestamp at all NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.\n"
    events = xid_watch.parse_xid_lines(text, year=2026, tzinfo=IST)
    assert len(events) == 1
    assert events[0]["ts"] is None
    assert events[0]["code"] == 79


# --------------------------------------------------------------------------
# record_xid_events: cross-call dedupe on disk
# --------------------------------------------------------------------------


def test_record_xid_events_writes_new_and_skips_seen(tmp_path):
    path = tmp_path / "xid.jsonl"
    e1 = {"ts": "t1", "raw_line": "line1", "code": 79, "severity": "fatal", "needs_reboot": True, "note": "n"}
    e2 = {"ts": "t2", "raw_line": "line2", "code": 154, "severity": "fatal", "needs_reboot": True, "note": "n"}

    written_first = xid_watch.record_xid_events([e1, e2], path)
    assert written_first == [e1, e2]
    assert len(path.read_text().splitlines()) == 2

    written_second = xid_watch.record_xid_events([e1, e2], path)
    assert written_second == []
    assert len(path.read_text().splitlines()) == 2, "re-recording the same events must not duplicate them"

    e3 = {"ts": "t3", "raw_line": "line3", "code": None, "severity": "unknown", "needs_reboot": False, "note": "n"}
    written_third = xid_watch.record_xid_events([e2, e3], path)
    assert written_third == [e3]
    assert len(path.read_text().splitlines()) == 3


# --------------------------------------------------------------------------
# build_crash_report / write_crash_report
# --------------------------------------------------------------------------


def _telemetry_sample(ts: str, temp: int) -> dict:
    return {
        "ts": ts,
        "gpu_unavailable": False,
        "temperature_c": temp,
        "power_draw_w": 400.0,
        "power_limit_w": 600.0,
        "util_gpu_percent": 90,
        "util_memory_percent": 50,
        "clocks_sm_mhz": 2000,
        "clocks_mem_mhz": 10000,
        "memory_used_mib": 50000,
        "memory_free_mib": 40000,
        "pstate": "P0",
        "clocks_throttle_reasons_active": "0x0",
        "live_metrics": None,
    }


def test_build_crash_report_contains_60s_window_and_xid_events(tmp_path):
    from servedeck import telemetry

    anchor = datetime(2026, 9, 12, 8, 55, 44, 984246, tzinfo=timezone.utc)
    telem_dir = tmp_path / "telemetry"
    day_file = telem_dir / f"{anchor.date().isoformat()}.jsonl"
    # Samples at -90s (outside window), -30s, -5s (inside), +10s (after fault,
    # must be excluded -- the window is the 60s BEFORE the fault).
    outside_before = (anchor - timedelta(seconds=90)).isoformat()
    inside_1 = (anchor - timedelta(seconds=30)).isoformat()
    inside_2 = (anchor - timedelta(seconds=5)).isoformat()
    outside_after = (anchor + timedelta(seconds=10)).isoformat()
    for ts, temp in [(outside_before, 41), (inside_1, 60), (inside_2, 75), (outside_after, 30)]:
        telemetry.append_jsonl_capped(day_file, _telemetry_sample(ts, temp), 10_000_000)

    trigger = [
        {
            "ts": anchor.isoformat(),
            "raw_ts": "Sep 12 14:25:44.984246",
            "code": 79,
            "raw_line": "NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.",
            "severity": "fatal",
            "needs_reboot": True,
            "note": "GPU has fallen off the bus.",
        }
    ]
    units_runner = FakeUnitsRunner()
    report = xid_watch.build_crash_report(trigger, state_dir=tmp_path, units_runner=units_runner)

    assert report["anchor_ts"] == anchor.isoformat()
    assert report["xid_events"] == trigger
    window_ts = {s["ts"] for s in report["telemetry_window_60s"]}
    assert window_ts == {inside_1, inside_2}
    assert outside_before not in window_ts
    assert outside_after not in window_ts

    assert report["live_models"], "the fake unit must show up as a live model"
    model = report["live_models"][0]
    assert model["unit"] == "model-flashnext"
    assert model["state"]["active_state"] == "active"
    assert model["journal_tail"] == ["log line 1", "log line 2"]


def test_write_crash_report_one_file_per_fault_never_overwritten(tmp_path):
    report1 = {"written_at": "w1", "anchor_ts": "2026-09-12T08:55:44.984246+00:00", "xid_events": [], "telemetry_window_60s": [], "live_models": []}
    report2 = {"written_at": "w2", "anchor_ts": "2026-09-12T08:55:44.984246+00:00", "xid_events": ["different"], "telemetry_window_60s": [], "live_models": []}

    path1 = xid_watch.write_crash_report(report1, state_dir=tmp_path)
    path2 = xid_watch.write_crash_report(report2, state_dir=tmp_path)

    assert path1 != path2, "a second fault with the same anchor timestamp must not overwrite the first"
    assert json.loads(path1.read_text())["written_at"] == "w1"
    assert json.loads(path2.read_text())["written_at"] == "w2"

    directory = tmp_path / "crash_reports"
    assert len(list(directory.glob("*.json"))) == 2


# --------------------------------------------------------------------------
# XidWatcher.scan_once
# --------------------------------------------------------------------------


def test_scan_once_records_new_events_and_writes_a_crash_report(tmp_path):
    text = FIXTURE_PATH.read_text(encoding="utf-8")
    runner = FakeRunner(ok(text))
    units_runner = FakeUnitsRunner()
    watcher = xid_watch.XidWatcher(state_dir=tmp_path, run=runner, units_runner=units_runner)

    new_events = watcher.scan_once()

    assert new_events, "the fixture contains real Xid lines that must be recorded"
    codes = {e["code"] for e in new_events}
    assert {79, 154} <= codes

    xid_log = xid_watch.default_xid_log_path(state_dir=tmp_path)
    assert xid_log.is_file()
    assert len(xid_log.read_text().splitlines()) == len(new_events)

    reports = list((tmp_path / "crash_reports").glob("*.json"))
    assert reports, "a fatal Xid must produce a crash report"

    # journalctl was called WITH -b all -- the exact fix for the -k-implies-
    # --boot=0 trap documented in the module.
    assert runner.calls
    assert "-b" in runner.calls[0] and "all" in runner.calls[0]
    assert "-k" in runner.calls[0]


def test_scan_once_second_call_finds_nothing_new(tmp_path):
    text = FIXTURE_PATH.read_text(encoding="utf-8")
    runner = FakeRunner(ok(text), ok(""))
    watcher = xid_watch.XidWatcher(state_dir=tmp_path, run=runner, units_runner=FakeUnitsRunner())

    first = watcher.scan_once()
    assert first

    second = watcher.scan_once()
    assert second == []
    # exactly the events from the first call remain on disk
    xid_log = xid_watch.default_xid_log_path(state_dir=tmp_path)
    assert len(xid_log.read_text().splitlines()) == len(first)


def test_scan_once_journalctl_failure_is_quiet_not_raising(tmp_path):
    runner = FakeRunner(fail(1, "journalctl: No journal files were found"))
    watcher = xid_watch.XidWatcher(state_dir=tmp_path, run=runner)
    assert watcher.scan_once() == []


# --------------------------------------------------------------------------
# check_fatal_xid_needs_reboot
# --------------------------------------------------------------------------


def test_doctor_check_no_crash_reports_is_ok(tmp_path):
    result = xid_watch.check_fatal_xid_needs_reboot(state_dir=tmp_path)
    assert result.ok is True
    assert "no crash reports" in result.detail


def test_doctor_check_fails_when_crash_report_is_newer_than_boot(tmp_path):
    report = {
        "written_at": "2026-09-12T08:56:00+00:00",
        "anchor_ts": "2026-09-12T08:55:44.984246+00:00",
        "xid_events": [],
        "telemetry_window_60s": [],
        "live_models": [],
    }
    xid_watch.write_crash_report(report, state_dir=tmp_path)
    boot_time = datetime(2026, 9, 11, 21, 36, 10, tzinfo=timezone.utc)  # the boot BEFORE the fault

    result = xid_watch.check_fatal_xid_needs_reboot(state_dir=tmp_path, boot_time=boot_time)
    assert result.ok is False
    assert "needs a reboot" in result.detail
    assert "2026-09-12T08:55:44.984246+00:00" in result.detail


def test_doctor_check_passes_once_rebooted_after_the_fault(tmp_path):
    report = {
        "written_at": "2026-09-12T08:56:00+00:00",
        "anchor_ts": "2026-09-12T08:55:44.984246+00:00",
        "xid_events": [],
        "telemetry_window_60s": [],
        "live_models": [],
    }
    xid_watch.write_crash_report(report, state_dir=tmp_path)
    boot_time = datetime(2026, 9, 12, 10, 53, 5, tzinfo=timezone.utc)  # the boot AFTER the fault

    result = xid_watch.check_fatal_xid_needs_reboot(state_dir=tmp_path, boot_time=boot_time)
    assert result.ok is True
    assert "already recovered" in result.detail


def test_current_boot_time_reads_the_real_proc_stat():
    """Light real-environment check: /proc/stat's btime, if present on this
    OS, parses to a timezone-aware datetime no later than now."""
    boot_time = xid_watch._current_boot_time()
    if boot_time is None:
        pytest.skip("/proc/stat not available on this platform")
    assert boot_time.tzinfo is not None
    assert boot_time <= datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# recent_xids / last_crash_report surfacing
# --------------------------------------------------------------------------


def test_recent_xids_newest_first_and_capped(tmp_path):
    path = xid_watch.default_xid_log_path(state_dir=tmp_path)
    events = [
        {"ts": f"2026-09-12T00:0{i}:00+00:00", "raw_line": f"line{i}", "code": 79, "severity": "fatal", "needs_reboot": True, "note": "n"}
        for i in range(5)
    ]
    xid_watch.record_xid_events(events, path)

    top2 = xid_watch.recent_xids(2, state_dir=tmp_path)
    assert [e["raw_line"] for e in top2] == ["line4", "line3"]


def test_recent_xids_empty_when_nothing_recorded(tmp_path):
    assert xid_watch.recent_xids(5, state_dir=tmp_path) == []


def test_last_crash_report_none_when_none_written(tmp_path):
    assert xid_watch.last_crash_report(state_dir=tmp_path) is None


def test_last_crash_report_returns_the_most_recent(tmp_path):
    xid_watch.write_crash_report({"written_at": "a", "anchor_ts": "2026-09-12T00:00:00+00:00"}, state_dir=tmp_path)
    xid_watch.write_crash_report({"written_at": "b", "anchor_ts": "2026-09-12T01:00:00+00:00"}, state_dir=tmp_path)
    latest = xid_watch.last_crash_report(state_dir=tmp_path)
    assert latest["written_at"] == "b"
