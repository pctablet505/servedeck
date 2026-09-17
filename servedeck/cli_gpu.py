"""``servedeck gpu-log`` — packet P6's CLI surface over
:mod:`servedeck.telemetry` / :mod:`servedeck.xid_watch`.

Deliberately its OWN module, not a function added to ``cli.py``: ``cli.py``
belongs to P1/P4. Wiring this in is one call — see :func:`add_parser` and the
module docstring's exact snippet.
"""

from __future__ import annotations

import argparse
import sys

from . import telemetry as _telemetry
from . import xid_watch as _xid_watch

__all__ = ["add_parser", "run"]


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Add the ``gpu-log`` subcommand to an existing subparsers group.

    In ``servedeck/cli.py``'s ``build_parser()``, P4 adds:

        from . import cli_gpu as _cli_gpu
        _cli_gpu.add_parser(sub)

    right next to the other ``sub.add_parser(...)`` calls (``sub`` is the
    ``argparse._SubParsersAction`` `build_parser()` already creates).
    """
    p = subparsers.add_parser(
        "gpu-log",
        help="tail recorded GPU telemetry samples or Xid/crash events",
    )
    p.add_argument(
        "--since",
        default="10m",
        help="how far back to look, e.g. 10m / 1h / 2d (telemetry samples only; ignored with --xid)",
    )
    p.add_argument("--tail", type=int, default=50, help="maximum rows to print (default: 50)")
    p.add_argument(
        "--xid",
        action="store_true",
        help="show recorded Xid/companion events instead of telemetry samples",
    )
    p.set_defaults(func=run)


def _fmt_sample(s: dict) -> str:
    if s.get("gpu_unavailable"):
        return f"{s.get('ts', '?')}  GPU_UNAVAILABLE  {s.get('error', '')}"
    if s.get("telemetry_truncated"):
        return f"{s.get('ts', '?')}  TRUNCATED  {s.get('reason', '')}"
    used = s.get("memory_used_mib")
    free = s.get("memory_free_mib")
    total = f"{used + free}" if isinstance(used, int) and isinstance(free, int) else "?"
    live = s.get("live_metrics") or {}
    load = ""
    if live:
        load = (
            f"  running={live.get('running')} waiting={live.get('waiting')} "
            f"kv={live.get('kv_usage_perc')}"
        )
    return (
        f"{s.get('ts', '?')}  temp={s.get('temperature_c')}C  "
        f"power={s.get('power_draw_w')}/{s.get('power_limit_w')}W  "
        f"util(gpu/mem)={s.get('util_gpu_percent')}%/{s.get('util_memory_percent')}%  "
        f"clocks(sm/mem)={s.get('clocks_sm_mhz')}/{s.get('clocks_mem_mhz')}MHz  "
        f"mem={used}/{total}MiB  pstate={s.get('pstate')}  "
        f"throttle={s.get('clocks_throttle_reasons_active')}{load}"
    )


def _fmt_xid(e: dict) -> str:
    code = e.get("code")
    code_str = str(code) if code is not None else "?"
    reboot = " NEEDS_REBOOT" if e.get("needs_reboot") else ""
    return f"{e.get('ts', '?')}  Xid={code_str}  [{e.get('severity')}]{reboot}  {e.get('raw_line', '')}"


def run(args: argparse.Namespace) -> int:
    if args.xid:
        events = _xid_watch.recent_xids(args.tail)
        if not events:
            print("(no Xid events recorded)")
            return 0
        for event in reversed(events):  # oldest-first: reads like a log tail
            print(_fmt_xid(event))
        return 0

    try:
        samples = _telemetry.samples_since(args.since, tail=args.tail)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not samples:
        print("(no telemetry samples in this window)")
        return 0
    for sample in samples:
        print(_fmt_sample(sample))
    return 0
