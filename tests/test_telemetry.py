"""servedeck.telemetry — the GPU sampler.

Every failure mode nvidia-smi can produce is exercised with a FakeRunner, not
the real binary, EXCEPT the one test explicitly marked "live": that one hits
the box's real, healthy GPU on purpose, per the packet's instructions.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from datetime import datetime, timedelta, timezone

import pytest

from servedeck import telemetry


class FakeRunner:
    """Records every argv and replays canned CompletedProcess results in
    order; once exhausted, repeats the last one (so a test only scripts the
    calls it cares about, and a Sampler loop that ticks more than scripted
    doesn't crash the test)."""

    def __init__(self, *results: subprocess.CompletedProcess) -> None:
        self.calls: list[list[str]] = []
        self._results = list(results)

    def __call__(self, argv):
        self.calls.append(list(argv))
        if len(self._results) > 1:
            return self._results.pop(0)
        return self._results[0] if self._results else subprocess.CompletedProcess(list(argv), 0, "", "")


def ok(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["nvidia-smi"], 0, stdout, "")


def fail(stderr: str, code: int = 6) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["nvidia-smi"], code, "", stderr)


HAPPY_ROW = "42, 85.31, 375.00, 12, 3, 2610, 13365, 6785, 90465, P1, 0x0000000000000000\n"


# --------------------------------------------------------------------------
# take_sample: happy path
# --------------------------------------------------------------------------


def test_happy_path_parses_every_field():
    runner = FakeRunner(ok(HAPPY_ROW))
    record = telemetry.take_sample(run=runner, wall=lambda: 1_800_000_000.0)

    assert record["gpu_unavailable"] is False
    assert record["temperature_c"] == 42
    assert record["power_draw_w"] == 85.31
    assert record["power_limit_w"] == 375.0
    assert record["util_gpu_percent"] == 12
    assert record["util_memory_percent"] == 3
    assert record["clocks_sm_mhz"] == 2610
    assert record["clocks_mem_mhz"] == 13365
    assert record["memory_used_mib"] == 6785
    assert record["memory_free_mib"] == 90465
    assert record["pstate"] == "P1"
    assert record["clocks_throttle_reasons_active"] == "0x0000000000000000"
    # wall-clock ISO timestamp, UTC
    assert record["ts"] == datetime.fromtimestamp(1_800_000_000.0, tz=timezone.utc).isoformat()
    # exactly one nvidia-smi call, querying every requested field
    assert len(runner.calls) == 1
    assert runner.calls[0][0] == "nvidia-smi"
    assert "--query-gpu=" in runner.calls[0][1]
    for field in telemetry.REQUESTED_GPU_FIELDS:
        assert field in runner.calls[0][1]


def test_not_supported_field_becomes_none_without_failing_the_sample():
    row = "38, [Not Supported], 375.00, 0, 0, 180, 405, 3656, 93595, P8, N/A\n"
    runner = FakeRunner(ok(row))
    record = telemetry.take_sample(run=runner)

    assert record["gpu_unavailable"] is False
    assert record["temperature_c"] == 38
    assert record["power_draw_w"] is None  # [Not Supported]
    assert record["clocks_throttle_reasons_active"] == "N/A"  # text field: kept verbatim, not None


def test_quoted_comma_field_parses_via_csv_not_naive_split():
    """clocks_throttle_reasons.active can legitimately contain a comma-joined
    list; nvidia-smi's CSV format quotes it. A naive `line.split(",")` would
    shift every field after it. Using csv.reader must not."""
    row = '38, 11.58, 375.00, 0, 0, 180, 405, 3656, 93595, P8, "SW Power Cap, HW Slowdown"\n'
    runner = FakeRunner(ok(row))
    record = telemetry.take_sample(run=runner)

    assert record["gpu_unavailable"] is False
    assert record["clocks_throttle_reasons_active"] == "SW Power Cap, HW Slowdown"
    assert record["pstate"] == "P8"  # the field just before the quoted one is unaffected


# --------------------------------------------------------------------------
# take_sample: failure modes
# --------------------------------------------------------------------------


def test_non_zero_exit_gpu_fallen_off_bus_writes_gpu_unavailable_record():
    """Today's exact case: nvidia-smi exits non-zero once the GPU is gone."""
    runner = FakeRunner(
        fail("Unable to determine the device handle for GPU 0000:01:00.0: Unknown Error\nNo devices were found\n")
    )
    record = telemetry.take_sample(run=runner)

    assert record["gpu_unavailable"] is True
    assert "Unable to determine the device handle" in record["error"]
    # only the FIRST stderr line is kept, not the whole trace
    assert "No devices were found" not in record["error"]
    assert "ts" in record


def test_missing_binary_via_default_runner_is_gpu_unavailable_not_an_exception(monkeypatch, tmp_path):
    """A PATH with no `nvidia-smi` on it at all -- through the REAL
    default_runner(), not a fake -- must still come back as one clean
    gpu_unavailable record, never a raised FileNotFoundError."""
    monkeypatch.setenv("PATH", str(tmp_path))  # empty dir: nothing resolves
    record = telemetry.take_sample(run=telemetry.default_runner())
    assert record["gpu_unavailable"] is True
    assert "not found on PATH" in record["error"]


def test_timeout_is_gpu_unavailable_via_real_default_runner(monkeypatch, tmp_path):
    """A `nvidia-smi` that hangs -- through the REAL default_runner()'s
    subprocess timeout, not a scripted one -- must still come back as a
    clean gpu_unavailable record."""
    fake_bin = tmp_path / "nvidia-smi"
    fake_bin.write_text("#!/bin/sh\n/bin/sleep 5\n")  # absolute path: PATH is about to be emptied
    fake_bin.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    record = telemetry.take_sample(run=telemetry.default_runner(timeout_s=0.2))
    assert record["gpu_unavailable"] is True
    assert "timed out" in record["error"]


def test_wrong_field_count_is_gpu_unavailable_not_a_crash():
    runner = FakeRunner(ok("38, 11.58\n"))  # far fewer fields than requested
    record = telemetry.take_sample(run=runner)
    assert record["gpu_unavailable"] is True
    assert "expected" in record["error"]


def test_empty_stdout_is_gpu_unavailable():
    runner = FakeRunner(ok(""))
    record = telemetry.take_sample(run=runner)
    assert record["gpu_unavailable"] is True


# --------------------------------------------------------------------------
# live_metrics_provider
# --------------------------------------------------------------------------


def test_live_metrics_provider_is_embedded_when_present():
    runner = FakeRunner(ok(HAPPY_ROW))
    record = telemetry.take_sample(
        run=runner, live_metrics_provider=lambda: {"running": 2, "waiting": 0, "kv_usage_perc": 0.31}
    )
    assert record["live_metrics"] == {"running": 2, "waiting": 0, "kv_usage_perc": 0.31}


def test_live_metrics_provider_none_when_not_given():
    runner = FakeRunner(ok(HAPPY_ROW))
    record = telemetry.take_sample(run=runner)
    assert record["live_metrics"] is None


def test_live_metrics_provider_raising_does_not_cost_the_sample():
    def boom():
        raise RuntimeError("metrics endpoint mid-restart")

    runner = FakeRunner(ok(HAPPY_ROW))
    record = telemetry.take_sample(run=runner, live_metrics_provider=boom)
    assert record["gpu_unavailable"] is False
    assert record["temperature_c"] == 42
    assert "metrics endpoint mid-restart" in record["live_metrics"]["error"]


# --------------------------------------------------------------------------
# detect_supported_fields
# --------------------------------------------------------------------------

_HELP_TEXT_ALL = "\n".join(f'    "{f}"' for f in telemetry.REQUESTED_GPU_FIELDS)


def test_detect_supported_fields_all_present_on_this_driver_shape():
    runner = FakeRunner(ok(_HELP_TEXT_ALL))
    supported, dropped = telemetry.detect_supported_fields(runner)
    assert supported == list(telemetry.REQUESTED_GPU_FIELDS)
    assert dropped == []


def test_detect_supported_fields_drops_what_the_help_text_lacks():
    reduced = "\n".join(
        f'    "{f}"' for f in telemetry.REQUESTED_GPU_FIELDS if f != "clocks_throttle_reasons.active"
    )
    runner = FakeRunner(ok(reduced))
    supported, dropped = telemetry.detect_supported_fields(runner)
    assert "clocks_throttle_reasons.active" not in supported
    assert dropped == ["clocks_throttle_reasons.active"]
    # order is preserved, not alphabetized or reversed
    assert supported == [f for f in telemetry.REQUESTED_GPU_FIELDS if f != "clocks_throttle_reasons.active"]


def test_detect_supported_fields_optimistic_when_help_itself_fails():
    runner = FakeRunner(fail("nvidia-smi: not found on PATH", code=1))
    supported, dropped = telemetry.detect_supported_fields(runner)
    assert supported == list(telemetry.REQUESTED_GPU_FIELDS)
    assert dropped == []


# --------------------------------------------------------------------------
# append_jsonl_capped / size cap
# --------------------------------------------------------------------------


def test_append_jsonl_capped_writes_normally_below_cap(tmp_path):
    path = tmp_path / "2026-09-12.jsonl"
    wrote = telemetry.append_jsonl_capped(path, {"ts": "x", "n": 1}, max_bytes=1_000_000)
    assert wrote is True
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"ts": "x", "n": 1}


def test_append_jsonl_capped_stops_growing_past_the_cap(tmp_path):
    path = tmp_path / "2026-09-12.jsonl"
    record = {"ts": "2026-09-12T00:00:00+00:00", "temperature_c": 40, "padding": "x" * 50}
    max_bytes = 40  # smaller than one serialized record, so ONE write already reaches the cap

    # The very first write is always allowed (the file starts empty, below
    # the cap) -- the cap is enforced on the NEXT attempt, once the file is
    # already at/over it. This bounds growth at "cap + one record", never
    # exactly at the byte, and never unbounded.
    first = telemetry.append_jsonl_capped(path, record, max_bytes)
    assert first is True
    assert path.stat().st_size >= max_bytes

    second = telemetry.append_jsonl_capped(path, record, max_bytes)
    assert second is False  # now over cap -> refused, one marker written instead
    lines_after_second = path.read_text().splitlines()
    assert len(lines_after_second) == 2
    marker = json.loads(lines_after_second[1])
    assert marker["telemetry_truncated"] is True
    assert "cap" in marker["reason"]

    # Ten more attempts must add nothing further -- one marker, forever.
    for _ in range(10):
        wrote = telemetry.append_jsonl_capped(path, record, max_bytes)
        assert wrote is False
    lines_after_more = path.read_text().splitlines()
    assert len(lines_after_more) == 2, "size cap must not grow unboundedly once truncated"


def test_append_jsonl_capped_never_unbounded_over_many_records(tmp_path):
    path = tmp_path / "2026-09-12.jsonl"
    max_bytes = 2000
    for i in range(500):
        telemetry.append_jsonl_capped(path, {"ts": f"t{i}", "i": i, "pad": "y" * 20}, max_bytes)
    size = path.stat().st_size
    assert size < max_bytes + 500, f"file grew to {size} bytes despite a {max_bytes}-byte cap"


# --------------------------------------------------------------------------
# prune_old_days
# --------------------------------------------------------------------------


def test_prune_old_days_deletes_only_stale_day_files(tmp_path):
    today = datetime(2026, 9, 12, tzinfo=timezone.utc)
    names_and_ages = {
        "2026-08-20.jsonl": 23,  # older than retention (14) -> deleted
        "2026-08-29.jsonl": 14,  # exactly at cutoff boundary -> deleted (< cutoff check)
        "2026-09-01.jsonl": 11,  # within retention -> kept
        "2026-09-12.jsonl": 0,  # today -> kept
        "xid.jsonl": None,  # not a day-file at all -> untouched
        "not-a-date.jsonl": None,
    }
    for name in names_and_ages:
        (tmp_path / name).write_text("{}\n")

    deleted = telemetry.prune_old_days(tmp_path, retention_days=14, today=today)
    deleted_names = {p.name for p in deleted}

    assert "2026-08-20.jsonl" in deleted_names
    assert not (tmp_path / "2026-08-20.jsonl").exists()
    assert (tmp_path / "2026-09-01.jsonl").exists()
    assert (tmp_path / "2026-09-12.jsonl").exists()
    assert (tmp_path / "xid.jsonl").exists()
    assert (tmp_path / "not-a-date.jsonl").exists()


def test_prune_old_days_missing_directory_is_a_noop(tmp_path):
    assert telemetry.prune_old_days(tmp_path / "does-not-exist", retention_days=14) == []


# --------------------------------------------------------------------------
# Sampler: rotation across a day boundary, pruning trigger
# --------------------------------------------------------------------------


def test_sampler_sample_once_writes_to_todays_file(tmp_path):
    runner = FakeRunner(ok(HAPPY_ROW))
    clock = iter([1_757_000_000.0])  # a fixed instant
    sampler = telemetry.Sampler(
        state_dir=tmp_path,
        run=runner,
        fields=telemetry.REQUESTED_GPU_FIELDS,
        wall=lambda: next(clock),
    )
    record = sampler.sample_once()
    day = record["ts"][:10]
    path = tmp_path / "telemetry" / f"{day}.jsonl"
    assert path.is_file()
    assert json.loads(path.read_text().splitlines()[0])["temperature_c"] == 42


def test_sampler_rotates_to_a_new_file_when_the_utc_day_changes(tmp_path):
    runner = FakeRunner(ok(HAPPY_ROW))
    # Two instants either side of a UTC midnight.
    before_midnight = datetime(2026, 9, 11, 23, 59, 59, tzinfo=timezone.utc).timestamp()
    after_midnight = datetime(2026, 9, 12, 0, 0, 1, tzinfo=timezone.utc).timestamp()
    times = iter([before_midnight, after_midnight])
    sampler = telemetry.Sampler(
        state_dir=tmp_path, run=runner, fields=telemetry.REQUESTED_GPU_FIELDS, wall=lambda: next(times)
    )
    sampler.sample_once()
    sampler.sample_once()

    telem_dir = tmp_path / "telemetry"
    assert (telem_dir / "2026-09-11.jsonl").is_file()
    assert (telem_dir / "2026-09-12.jsonl").is_file()


def test_sampler_field_detection_reports_dropped_fields(tmp_path):
    reduced_help = "\n".join(
        f'    "{f}"' for f in telemetry.REQUESTED_GPU_FIELDS if f != "power.limit"
    )
    runner = FakeRunner(ok(reduced_help), ok(HAPPY_ROW))
    sampler = telemetry.Sampler(state_dir=tmp_path, run=runner)
    assert sampler.dropped_fields == ["power.limit"]
    assert "power.limit" not in sampler.fields


def test_sampler_start_and_stop_run_the_asyncio_loop():
    runner = FakeRunner(ok(HAPPY_ROW))

    async def go(tmp_path):
        sampler = telemetry.Sampler(
            state_dir=tmp_path,
            run=runner,
            fields=telemetry.REQUESTED_GPU_FIELDS,
            interval_s=0.01,
        )
        sampler.start()
        await asyncio.sleep(0.05)
        await sampler.stop()
        return sampler

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path

        sampler = asyncio.run(go(Path(d)))
        telem_dir = Path(d) / "telemetry"
        files = list(telem_dir.glob("*.jsonl"))
        assert files, "the asyncio loop must have written at least one sample"
        total_lines = sum(len(f.read_text().splitlines()) for f in files)
        assert total_lines >= 2, "a 0.01s interval over 0.05s must produce more than one sample"


# --------------------------------------------------------------------------
# last_samples / samples_since / parse_duration
# --------------------------------------------------------------------------


def test_last_samples_reads_across_day_files_oldest_first(tmp_path):
    telem_dir = tmp_path / "telemetry"
    telem_dir.mkdir(parents=True)
    for i, ts in enumerate(["2026-09-11T23:59:00+00:00", "2026-09-11T23:59:30+00:00"]):
        telemetry.append_jsonl_capped(telem_dir / "2026-09-11.jsonl", {"ts": ts, "i": i}, 10_000_000)
    for i, ts in enumerate(["2026-09-12T00:00:00+00:00", "2026-09-12T00:00:05+00:00"]):
        telemetry.append_jsonl_capped(telem_dir / "2026-09-12.jsonl", {"ts": ts, "i": 10 + i}, 10_000_000)

    result = telemetry.last_samples(3, state_dir=tmp_path)
    assert [r["ts"] for r in result] == [
        "2026-09-11T23:59:30+00:00",
        "2026-09-12T00:00:00+00:00",
        "2026-09-12T00:00:05+00:00",
    ]


def test_last_samples_empty_when_nothing_sampled(tmp_path):
    assert telemetry.last_samples(5, state_dir=tmp_path) == []


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("10m", timedelta(minutes=10)),
        ("1h", timedelta(hours=1)),
        ("30s", timedelta(seconds=30)),
        ("2d", timedelta(days=2)),
        ("45", timedelta(seconds=45)),
    ],
)
def test_parse_duration_accepts_documented_forms(spec, expected):
    assert telemetry.parse_duration(spec) == expected


@pytest.mark.parametrize("spec", ["", "10x", "abc", "-5m"])
def test_parse_duration_rejects_garbage(spec):
    with pytest.raises(ValueError):
        telemetry.parse_duration(spec)


def test_samples_since_filters_by_window_and_applies_tail(tmp_path):
    telem_dir = tmp_path / "telemetry"
    telem_dir.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    day_file = telem_dir / f"{now.date().isoformat()}.jsonl"
    old_ts = (now - timedelta(hours=2)).isoformat()
    recent_ts_1 = (now - timedelta(minutes=5)).isoformat()
    recent_ts_2 = (now - timedelta(minutes=1)).isoformat()
    for ts in (old_ts, recent_ts_1, recent_ts_2):
        telemetry.append_jsonl_capped(day_file, {"ts": ts}, 10_000_000)

    result = telemetry.samples_since("10m", state_dir=tmp_path)
    assert [r["ts"] for r in result] == [recent_ts_1, recent_ts_2]

    tailed = telemetry.samples_since("10m", state_dir=tmp_path, tail=1)
    assert [r["ts"] for r in tailed] == [recent_ts_2]


# --------------------------------------------------------------------------
# Live smoke test — the real, healthy GPU on this box (driver 595.84)
# --------------------------------------------------------------------------


def test_live_three_real_samples_have_plausible_fields():
    """No fakes: three real `nvidia-smi` calls against this box's actual,
    healthy RTX PRO 6000. Confirms the query line this driver accepts and
    that every numeric field lands in a physically sane range."""
    from servedeck import gpu as _gpu

    total = _gpu.total_mib()
    if total is None:
        pytest.skip("nvidia-smi not available on this runner")

    samples = []
    for _ in range(3):
        record = telemetry.take_sample()
        samples.append(record)
        time.sleep(1)

    for record in samples:
        assert record["gpu_unavailable"] is False, record.get("error")
        assert 20 <= record["temperature_c"] <= 95
        assert 0 <= record["power_draw_w"] <= 600
        assert 0 <= record["power_limit_w"] <= 600
        assert 0 <= record["util_gpu_percent"] <= 100
        assert 0 <= record["util_memory_percent"] <= 100
        assert record["memory_free_mib"] <= total
        assert record["memory_used_mib"] + record["memory_free_mib"] <= total + 64  # rounding slack
        assert record["pstate"].startswith("P")
