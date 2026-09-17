"""servedeck.cli_gpu — the `servedeck gpu-log` subcommand.

`cli.py` is P1/P4's module and is not touched by this packet; these tests
build a standalone parser the same way `cli.py.build_parser()` would (one
`add_parser(sub)` call), never importing `servedeck.cli` itself.
"""

from __future__ import annotations

import argparse

from servedeck import cli_gpu, telemetry, xid_watch


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="servedeck")
    sub = parser.add_subparsers(dest="command", required=True)
    cli_gpu.add_parser(sub)
    return parser


def test_add_parser_wires_gpu_log_with_documented_defaults():
    parser = _build_parser()
    args = parser.parse_args(["gpu-log"])
    assert args.command == "gpu-log"
    assert args.since == "10m"
    assert args.tail == 50
    assert args.xid is False
    assert args.func is cli_gpu.run


def test_add_parser_accepts_all_documented_flags():
    parser = _build_parser()
    args = parser.parse_args(["gpu-log", "--since", "1h", "--tail", "5", "--xid"])
    assert args.since == "1h"
    assert args.tail == 5
    assert args.xid is True


def test_run_telemetry_path_prints_samples(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(telemetry, "default_state_dir", lambda: tmp_path)
    day_file = tmp_path / "telemetry" / f"{__import__('datetime').date.today().isoformat()}.jsonl"
    telemetry.append_jsonl_capped(day_file, {"ts": telemetry.now_iso(), "gpu_unavailable": False, "temperature_c": 55, "power_draw_w": 300.0, "power_limit_w": 600.0, "util_gpu_percent": 40, "util_memory_percent": 10, "clocks_sm_mhz": 2000, "clocks_mem_mhz": 10000, "memory_used_mib": 1000, "memory_free_mib": 2000, "pstate": "P0", "clocks_throttle_reasons_active": "0x0", "live_metrics": None}, 10_000_000)

    parser = _build_parser()
    args = parser.parse_args(["gpu-log", "--since", "1h"])
    rc = args.func(args)

    assert rc == 0
    out = capsys.readouterr().out
    assert "temp=55C" in out
    assert "pstate=P0" in out


def test_run_telemetry_path_empty_window_prints_placeholder(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(telemetry, "default_state_dir", lambda: tmp_path)
    parser = _build_parser()
    args = parser.parse_args(["gpu-log"])
    rc = args.func(args)
    assert rc == 0
    assert "no telemetry samples" in capsys.readouterr().out


def test_run_telemetry_path_bad_since_reports_error(capsys):
    parser = _build_parser()
    args = parser.parse_args(["gpu-log", "--since", "not-a-duration"])
    rc = args.func(args)
    assert rc == 1
    assert "error" in capsys.readouterr().err


def test_run_xid_path_prints_events_oldest_first(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(telemetry, "default_state_dir", lambda: tmp_path)
    path = xid_watch.default_xid_log_path(state_dir=tmp_path)
    events = [
        {"ts": "2026-09-12T08:55:44.984246+00:00", "raw_line": "...Xid 79...fallen off the bus", "code": 79, "severity": "fatal", "needs_reboot": True, "note": "n"},
        {"ts": "2026-09-12T08:55:44.984940+00:00", "raw_line": "...Xid 154...Node Reboot Required", "code": 154, "severity": "fatal", "needs_reboot": True, "note": "n"},
    ]
    xid_watch.record_xid_events(events, path)

    parser = _build_parser()
    args = parser.parse_args(["gpu-log", "--xid"])
    rc = args.func(args)

    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2
    assert "Xid=79" in out[0]  # oldest first
    assert "Xid=154" in out[1]
    assert "NEEDS_REBOOT" in out[0]


def test_run_xid_path_empty_prints_placeholder(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(telemetry, "default_state_dir", lambda: tmp_path)
    parser = _build_parser()
    args = parser.parse_args(["gpu-log", "--xid"])
    rc = args.func(args)
    assert rc == 0
    assert "no Xid events" in capsys.readouterr().out
