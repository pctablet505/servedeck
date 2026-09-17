"""``servedeck`` — the one CLI (REDESIGN-2026-09-12.md §2.4, §2.5).

It replaces ``llm``, three launcher scripts and ``smoke.py``. Everything the
page can do, a terminal can do, because the owner's agents (Codex, Claude Code)
live in terminals.

Two shapes of subcommand:

* **Local** (``models``, ``wire``, ``doctor``) read ``models.toml`` and the
  filesystem. They need no server and are unchanged from P1.
* **Control** (``status``, ``start``, ``stop``, ``switch``, ``log``, ``smoke``)
  drive the dashboard over HTTP: one POST, then ``/api/events`` streamed until
  the model is ready or has failed. The stream is opened *before* the POST, so
  a boot that produces its first marker in under a millisecond cannot slip
  through the gap between the two calls.

The fallback that matters
-------------------------
``servedeck start`` must work when the dashboard is down — that is the state an
operator is most likely to be in, and a CLI whose only mode is "ask the thing
that is not running" is a CLI that fails exactly when it is needed. So the
control commands fall back to driving ``Control`` in-process, and say so on the
first line. It is the same ``Control`` the dashboard builds
(``app.build_control``), so a model started this way is a model the dashboard
will adopt unchanged when it comes back.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from . import doctor as _doctor
from . import models as _models
from . import settings as _settings
from . import wire as _wire

__all__ = ["build_parser", "main"]

#: A boot is minutes. The read timeout has to outlast the slowest one, and the
#: SSE stream is silent for whole minutes at a time between markers.
_STREAM_TIMEOUT = httpx.Timeout(connect=2.0, read=None, write=10.0, pool=10.0)
_QUICK_TIMEOUT = 5.0

#: Notices that end a `servedeck start|stop|switch` wait, and the exit code.
_TERMINAL: dict[str, int] = {
    "ready": 0,
    "stopped": 0,
    "adopted": 0,
    "boot_failed": 1,
    "exception": 1,
    # Every control.Refusal reason: the mutation was accepted and then refused
    # against reality (a race the precheck could not see).
    "unknown_model": 1,
    "already_live": 1,
    "main_slot_busy": 1,
    "not_main_slot": 1,
    "not_enough_vram": 1,
    "gpu_unavailable": 1,
    "no_vram_budget": 1,
    "start_failed": 1,
    "stop_failed": 1,
    "vram_not_released": 1,
    "bad_key": 1,
}


def default_models_toml() -> Path:
    return _settings.get().models_path


def _load_or_report(models_toml: Path) -> _models.Registry | None:
    try:
        return _models.load(models_toml)
    except _models.RegistryError as e:
        print(f"error: {e}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
# Local commands (P1)
# --------------------------------------------------------------------------- #


def _table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
        for i, h in enumerate(headers)
    ]

    def fmt(row: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip()

    out = [fmt(headers), fmt(tuple("-" * w for w in widths))]
    out.extend(fmt(r) for r in rows)
    return "\n".join(out)


def _cmd_models(args: argparse.Namespace) -> int:
    reg = _load_or_report(args.models_toml)
    if reg is None:
        return 1
    print(
        _table(
            ("key", "id", "slot", "port", "build", "ctx"),
            [
                (key, m.id, m.slot, str(m.port), m.build, str(m.ctx))
                for key, m in reg.models.items()
            ],
        )
    )
    return 0


def _backup_and_write(path: Path, old_content: str) -> Path:
    backup_dir = _wire.BACKUP_ROOT / date.today().isoformat()
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / str(path).lstrip("/").replace("/", "_")
    backup_path.write_text(old_content)
    return backup_path


def _cmd_wire(args: argparse.Namespace) -> int:
    reg = _load_or_report(args.models_toml)
    if reg is None:
        return 1
    resolve_ctx = _wire.make_default_ctx_resolver(reg)

    for target in _wire.WIRE_TARGETS:
        before = _wire.read_existing(target.path)
        after = target.render(reg, before, resolve_ctx=resolve_ctx)
        if before == after:
            print(f"== {target.name}: no changes")
            continue
        if args.apply:
            backup_path = _backup_and_write(target.path, before)
            target.path.parent.mkdir(parents=True, exist_ok=True)
            target.path.write_text(after)
            print(f"== {target.name}: applied (backup: {backup_path})")
        else:
            print(f"== {target.name}: would change (dry-run; pass --apply to write)")
            diff = _wire.unified_diff(target.name, before, after)
            sys.stdout.write(diff if diff.endswith("\n") else diff + "\n")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    results = _doctor.run_doctor(args.models_toml)
    print(_doctor.format_table(results))
    return 0 if _doctor.all_ok(results) else 1


# --------------------------------------------------------------------------- #
# Talking to the dashboard
# --------------------------------------------------------------------------- #


def _base_url(args: argparse.Namespace) -> str:
    return (args.url or _settings.get().base_url).rstrip("/")


def _get_json(base: str, path: str) -> Any | None:
    """GET, or None if the dashboard is not answering. Never raises: "is it
    up" and "what did it say" are the same call, and a traceback here would
    read as a bug in servedeck rather than as a server that is down."""
    try:
        response = httpx.get(f"{base}{path}", timeout=_QUICK_TIMEOUT)
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    try:
        return response.json()
    except ValueError:
        return None


def _dashboard_up(base: str) -> bool:
    payload = _get_json(base, "/api/health")
    return isinstance(payload, dict) and payload.get("ok") is True


def _fmt_secs(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    seconds = int(value)
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}"


def _fmt_num(value: Any, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.1f}{suffix}"
    return f"{value:,}{suffix}"


def _cmd_status(args: argparse.Namespace) -> int:
    base = _base_url(args)
    state = _get_json(base, "/api/state")
    if state is None:
        print(f"dashboard not running at {base}", file=sys.stderr)
        print("start it with: systemctl --user start servedeck", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(state, indent=2))
        return 0

    rows: list[tuple[str, ...]] = []
    for m in state["models"]:
        met = m.get("metrics") or {}
        rows.append(
            (
                m["key"],
                m["id"],
                m["slot"],
                "ready" if m["ready"] else ("booting" if m["live"] else "-"),
                m["unit_state"],
                _fmt_num(m.get("ctx")),
                f"{met['kv_usage_perc'] * 100:.0f}%" if met.get("kv_usage_perc") is not None else "-",
                f"{met.get('running', '-')}/{met.get('waiting', '-')}" if met else "-",
                _fmt_num(met.get("gen_tok_s")),
                _fmt_secs(m.get("uptime_s")),
                str(m.get("restarts", 0)),
            )
        )
    print(
        _table(
            ("key", "id", "slot", "state", "unit", "ctx", "kv", "run/wait", "tok/s", "up", "rst"),
            rows,
        )
    )
    gpu = state.get("gpu") or {}
    head = state.get("headroom") or {}
    print()
    print(f"gateway  {state['gateway_url']}")
    print(f"gpu      {_fmt_num(gpu.get('free_mib'))} MiB free of {_fmt_num(gpu.get('total_mib'))}")
    if head.get("source"):
        print(
            f"headroom {head['full_context_requests']} full-context "
            f"({_fmt_num(head['full_ctx'])}) + {head['small_requests']} "
            f"{_fmt_num(head['small_request_tokens'])}-token requests "
            f"— {head['source']}"
        )
    elif head.get("unavailable"):
        print(f"headroom unknown — {head['unavailable']}")
    for stray in state.get("unknown_units") or []:
        print(f"warning: {stray['unit']} is running but is not in models.toml", file=sys.stderr)
    return 0


def _cmd_log(args: argparse.Namespace) -> int:
    base = _base_url(args)
    payload = _get_json(base, f"/api/log/{args.key}?lines={args.lines}")
    if payload is None:
        if _dashboard_up(base):
            print(f"error: no model {args.key!r} in the registry", file=sys.stderr)
            return 1
        # journalctl is right here; there is no reason to need a web server to
        # read a log the operator could read by hand.
        print("dashboard not running, reading the journal directly", file=sys.stderr)
        from . import units as _units

        unit = f"{_settings.get().unit_prefix}{args.key}"
        for line in _units.journal_tail(unit, args.lines):
            print(line)
        return 0
    for line in payload["lines"]:
        print(line)
    return 0


# --------------------------------------------------------------------------- #
# start / stop / switch
# --------------------------------------------------------------------------- #


def _print_progress(kind: str, text: str) -> None:
    prefix = {"marker": "  * ", "ready": "  = ", "failed": "  ! "}.get(kind, "    ")
    print(f"{prefix}{text}", flush=True)


def _drive_via_dashboard(base: str, action: str, key: str, timeout_s: float) -> int:
    """POST the action, then print ``/api/events`` until it ends.

    The stream is opened first and the POST issued from inside it. Reversing
    those two loses every frame emitted in between, which on a stub or a warm
    model is all of them — the command then hangs waiting for a boot that has
    already finished.
    """
    path = {
        "start": f"/api/models/{key}/start",
        "stop": f"/api/models/{key}/stop",
        "switch": f"/api/switch/{key}",
    }[action]

    with httpx.Client(timeout=_STREAM_TIMEOUT) as client:
        with client.stream("GET", f"{base}/api/events") as stream:
            if stream.status_code != 200:
                print(f"error: /api/events answered {stream.status_code}", file=sys.stderr)
                return 1
            try:
                response = client.post(f"{base}{path}", timeout=_QUICK_TIMEOUT)
            except httpx.HTTPError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            if response.status_code != 202:
                body = _error_of(response)
                print(f"refused ({response.status_code}): {body}", file=sys.stderr)
                return 1
            print(f"{action} {key}: accepted", flush=True)
            return _consume_events(stream, key, action)


def _error_of(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or "(no body)"
    err = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(err, dict):
        return f"{err.get('reason')}: {err.get('message')}"
    return json.dumps(payload)


def _consume_events(stream: httpx.Response, key: str, action: str) -> int:
    """Print progress for ``key`` until a terminal notice arrives.

    Events for other models are ignored rather than printed: a resident
    restarting while you watch a switch is not your switch, and interleaving
    the two is how an operator reads the wrong journal line as the cause of a
    failure.
    """
    event_type = ""
    for line in stream.iter_lines():
        line = line.rstrip("\r")
        if line.startswith("event: "):
            event_type = line[7:].strip()
            continue
        if not line.startswith("data: "):
            continue
        try:
            data = json.loads(line[6:])
        except ValueError:
            continue
        if event_type == "progress" and data.get("key") in (key, "reconcile"):
            _print_progress(data.get("kind", ""), data.get("text", ""))
        elif event_type == "notice":
            if data.get("replay"):
                # The backlog the stream opens with. It describes things that
                # already happened -- possibly a previous boot of this same
                # model -- and treating one as terminal would exit 0 the
                # instant we connected, reporting the last run's outcome.
                continue
            reason = data.get("reason", "")
            if data.get("key") not in (key, None, ""):
                continue
            code = _TERMINAL.get(reason)
            if code is None:
                continue
            message = data.get("message", reason)
            print(message, file=sys.stderr if code else sys.stdout, flush=True)
            for journal_line in (data.get("journal") or [])[-20:]:
                print(f"  | {journal_line}", file=sys.stderr)
            return code
    print(f"error: the event stream ended before {action} {key} finished", file=sys.stderr)
    return 1


def _drive_directly(action: str, key: str, timeout_s: float) -> int:
    """The fallback: build the same Control the dashboard builds, and use it.

    Prints one line saying so. An operator who does not know which of the two
    paths ran cannot tell "the model started" from "the model started and the
    dashboard does not know about it" — and the answer to the second is
    ``servedeck adopt``, which they will not think to run.
    """
    print("dashboard not running, acting directly", file=sys.stderr)
    from . import app as _app
    from . import control as _control

    settings = _settings.get()
    registry = _load_or_report(settings.models_path)
    if registry is None:
        return 1
    control = _app.build_control(settings, registry)

    def on_progress(event: Any) -> None:
        _print_progress(event.kind, event.text)

    if action == "start":
        result: Any = control.start(key, timeout_s=timeout_s, on_progress=on_progress)
    elif action == "stop":
        result = control.stop(key)
    else:
        result = control.switch(key, timeout_s=timeout_s, on_progress=on_progress)

    if isinstance(result, _control.SwitchResult):
        if result.stopped is not None:
            print(f"stopped {result.stopped.unit} (held ~{result.stopped.held_mib} MiB)")
        result = result.started
    if isinstance(result, _control.Refusal):
        print(f"refused ({result.reason}): {result.message}", file=sys.stderr)
        return 1
    if isinstance(result, _control.StopResult):
        print(f"{result.unit} stopped")
        return 0
    if isinstance(result, _control.StartResult):
        if result.ready:
            print(f"{key} ready in {result.elapsed_s:.0f}s")
            return 0
        print(f"{key} failed: {result.failure}", file=sys.stderr)
        for line in result.journal[-20:]:
            print(f"  | {line}", file=sys.stderr)
        return 1
    print(f"{key}: {result}")
    return 0


def _cmd_control(args: argparse.Namespace) -> int:
    base = _base_url(args)
    if _dashboard_up(base):
        return _drive_via_dashboard(base, args.command, args.key, args.timeout)
    return _drive_directly(args.command, args.key, args.timeout)


def _cmd_adopt(args: argparse.Namespace) -> int:
    base = _base_url(args)
    if _dashboard_up(base):
        try:
            response = httpx.post(f"{base}/api/adopt", timeout=_QUICK_TIMEOUT)
        except httpx.HTTPError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"adopt: {'accepted' if response.status_code == 202 else _error_of(response)}")
        return 0 if response.status_code == 202 else 1
    print("dashboard not running, acting directly", file=sys.stderr)
    from . import app as _app

    settings = _settings.get()
    registry = _load_or_report(settings.models_path)
    if registry is None:
        return 1
    result = _app.build_control(settings, registry).adopt()
    print(f"adopted: {result.adopted or 'nothing'}")
    for unit in result.unknown_units:
        print(f"warning: {unit} is running but is not in models.toml", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #

_SMOKE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _cmd_smoke(args: argparse.Namespace) -> int:
    """Two requests through the gateway: a plain chat and a tool call.

    Through the GATEWAY, on the model's public name — never at the model's own
    port. That is the whole point: the thing being proved is that the name a
    client is configured with reaches the weights, which is precisely the
    failure that produced two 404 outages (REDESIGN §4 R1). Hitting :8004
    directly would pass in exactly the situation the check exists to catch.
    """
    base = _base_url(args)
    reg = _load_or_report(_settings.get().models_path)
    if reg is None:
        return 1
    found = reg.resolve(args.key)
    if found is None:
        print(f"error: no model {args.key!r} in the registry", file=sys.stderr)
        return 1
    name = args.key if args.key != found.model.key else found.model.id
    url = f"{base}/v1/chat/completions"
    failures = 0

    print(f"smoke {name} via {url}")
    try:
        plain = httpx.post(
            url,
            json={
                "model": name,
                "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
                "max_tokens": 32,
                "temperature": 0.0,
            },
            timeout=args.timeout,
        )
    except httpx.HTTPError as exc:
        print(f"  FAIL chat: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if plain.status_code != 200:
        print(f"  FAIL chat: HTTP {plain.status_code} {_error_of(plain)}", file=sys.stderr)
        failures += 1
    else:
        message = (plain.json()["choices"][0] or {}).get("message") or {}
        content = message.get("content")
        if not content:
            print("  FAIL chat: 200 but the message has no content", file=sys.stderr)
            failures += 1
        else:
            print(f"  ok   chat: {content.strip()[:60]!r}")
        if message.get("reasoning_content"):
            print("  ok   reasoning_content is present (the mirror is working)")

    try:
        tooled = httpx.post(
            url,
            json={
                "model": name,
                "messages": [{"role": "user", "content": "What is the weather in Pune? Use the tool."}],
                "tools": [_SMOKE_TOOL],
                "tool_choice": "auto",
                "max_tokens": 128,
                "temperature": 0.0,
            },
            timeout=args.timeout,
        )
    except httpx.HTTPError as exc:
        print(f"  FAIL tools: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if tooled.status_code != 200:
        print(f"  FAIL tools: HTTP {tooled.status_code} {_error_of(tooled)}", file=sys.stderr)
        failures += 1
    else:
        message = (tooled.json()["choices"][0] or {}).get("message") or {}
        calls = message.get("tool_calls") or []
        if calls:
            print(f"  ok   tools: called {calls[0]['function']['name']}")
        else:
            # Not a failure: whether a model chooses to call is the model's
            # business. That the request was ACCEPTED and parsed is what this
            # proves, and saying otherwise would make the check fail for
            # reasons the operator cannot act on.
            print("  warn tools: 200, but the model answered in prose (no tool call)")
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--models-toml",
        type=Path,
        default=default_models_toml(),
        help="path to models.toml (default: the repo root's, or $SERVEDECK_MODELS)",
    )
    remote = argparse.ArgumentParser(add_help=False)
    remote.add_argument(
        "--url",
        default=None,
        help="dashboard base URL (default: http://127.0.0.1:8010, or $SERVEDECK_HOST/$SERVEDECK_PORT)",
    )

    parser = argparse.ArgumentParser(prog="servedeck", description="servedeck v2 control CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_models = sub.add_parser("models", parents=[common], help="list models from the registry")
    p_models.set_defaults(func=_cmd_models)

    p_wire = sub.add_parser("wire", parents=[common], help="generate/update client configs")
    # Dry run is the DEFAULT, and `--dry-run` is the spelling REDESIGN §2.4 and
    # §3 step 1 both use ("`servedeck wire --dry-run` diff"). Without the flag
    # argparse answered "unrecognized arguments: --dry-run" with exit 2 — which,
    # to an operator following the design doc during a cutover, reads as "wire is
    # broken" rather than "the flag is spelled differently".
    #
    # Mutually exclusive rather than two independent booleans: `--dry-run
    # --apply` has no correct resolution, and picking one silently would either
    # write when the operator asked not to or refuse when they asked to.
    mode = p_wire.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply", action="store_true", help="write changes (default: dry-run diff only)"
    )
    mode.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="show the diff and write nothing (the default; accepted explicitly)",
    )
    p_wire.set_defaults(func=_cmd_wire)

    p_doctor = sub.add_parser("doctor", parents=[common], help="check the registry against reality")
    p_doctor.set_defaults(func=_cmd_doctor)

    p_status = sub.add_parser("status", parents=[remote], help="what is live right now")
    p_status.add_argument("--json", action="store_true", help="print /api/state verbatim")
    p_status.set_defaults(func=_cmd_status)

    for name, helptext in (
        ("start", "start a model and wait for it"),
        ("stop", "stop a model"),
        ("switch", "replace whatever holds the main slot"),
    ):
        p = sub.add_parser(name, parents=[remote], help=helptext)
        p.add_argument("key", help="registry key from `servedeck models`")
        p.add_argument("--timeout", type=float, default=900.0, help="seconds to wait")
        p.set_defaults(func=_cmd_control, command=name)

    p_adopt = sub.add_parser("adopt", parents=[remote], help="record running units as desired")
    p_adopt.set_defaults(func=_cmd_adopt)

    p_log = sub.add_parser("log", parents=[remote], help="tail a model's journal")
    p_log.add_argument("key")
    p_log.add_argument("-n", "--lines", type=int, default=80)
    p_log.set_defaults(func=_cmd_log)

    p_smoke = sub.add_parser("smoke", parents=[remote], help="one chat + one tool call via the gateway")
    p_smoke.add_argument("key", help="registry key, id, alias or preset")
    p_smoke.add_argument("--timeout", type=float, default=120.0)
    p_smoke.set_defaults(func=_cmd_smoke)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
