"""P7 acceptance gate: the merged v2 stack against REAL vLLM servers.

REDESIGN-2026-09-12 §2.7 is explicit that "nearly all tests run against mocks"
and that "the live boot/switch/adopt paths, the ones that actually fail, have
no end-to-end test".  ``tests/test_control_e2e.py`` (P3) proves the lifecycle
with a stdlib stub and one LFM2 boot; ``tests/test_gateway_e2e.py`` (P2) proves
the gateway against the *production* resident on :8007 with a hand-written
route table.  Neither exercises the three packets **joined**: a registry entry
from ``models.py``, launched by ``control.py``, served through ``gateway.py``.

This file does that, with two ~350M models as fixtures:

* ``LiquidAI/LFM2.5-350M`` — no thinking mode, ``lfm2`` tool parser.  The
  workhorse: control plane, main-slot exclusivity, switch, reconcile, failed
  boot, the gateway, the output floor, the KV prediction.
* ``Qwen/Qwen3-0.6B`` with ``--reasoning-parser qwen3`` — the only fixture on
  this box small enough to boot in a test that HAS a thinking mode, so it is
  the only way to prove the reasoning mirror (the amnesia fix) end to end.

Blast radius
------------
Every unit created here is named ``sd-test-vfy-*``: ``units.py`` refuses every
other shape before spawning anything, every ``Control`` is constructed with
``unit_prefix="sd-test-"`` so even its discovery glob cannot see a ``model-*``
unit, and the module finaliser stops every ``sd-test-*`` unit whether the tests
passed, failed or raised.  Only 127.0.0.1 ports 8050-8055 are bound.  No
existing unit is touched, nothing is enabled or disabled, and no process this
file did not start is ever signalled.

What is NOT tested here, and cannot be
--------------------------------------
``servedeck/routes.py`` (the production ``RouteTable``) and
``ModelSpecAdapter`` (the production ``models.Model`` -> ``control.ModelSpec``
bridge) both live on the **unmerged** ``v2-p4`` branch.  ``v2`` as merged has
no join between P1's registry and P3's supervisor at all.  So the adapters in
this file (:class:`RegistrySpec`, :class:`ControlRoutes`) are the *test's own*
minimal implementations of those two Protocols, written to the Protocols rather
than copied from P4.  They prove the Protocols are satisfiable and that the
real modules behave against a real server; they do NOT prove P4's code.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import threading
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from servedeck import control, gateway, gpu, kvcalc, models as models_mod, units
from servedeck.control import Control, Refusal
from servedeck.gateway import Route, RoutePolicies

pytestmark = pytest.mark.skipif(
    shutil.which("systemd-run") is None or not os.environ.get("XDG_RUNTIME_DIR"),
    reason="needs a systemd user manager (systemd-run + XDG_RUNTIME_DIR)",
)

# --------------------------------------------------------------------------
# Fixture constants
# --------------------------------------------------------------------------

VENV_BIN = Path("/home/pctablet505/Projects/local_llm/.venv-llm-029/bin")
VLLM = VENV_BIN / "vllm"

LFM2_REPO = "LiquidAI/LFM2.5-350M"
LFM2_ID = "LFM2.5-350M"
QWEN_REPO = "Qwen/Qwen3-0.6B"
QWEN_ID = "Qwen3-0.6B"

#: 127.0.0.1 only, and only inside the range this packet was given.
PORT_MAIN_A = 8050
PORT_MAIN_B = 8051
PORT_RESIDENT = 8052
PORT_QWEN = 8053
PORT_BAD = 8054
PORT_GATEWAY = 8055

#: Per-instance footprint at util 0.03 on a 97,887 MiB card: ~2.9 GiB of
#: declared budget plus the worker's CUDA context.  Measured 3,130 MiB.
UTIL = 0.03
CTX = 8192
TOTAL_MIB = 97887

#: Below this, SKIP rather than boot: the owner's 90 GiB model may be on the
#: card and an OOM here would be an outage, not a test failure.
MIN_FREE_MIB = 6144

#: A cold LFM2 boot measured 102 s on this box, of which 79 s is CUDA graph
#: capture (owner rule: no --enforce-eager, so that cost is real and stays).
BOOT_TIMEOUT_S = 420.0

#: Collected across the run and printed by the `measurements` finaliser, so the
#: numbers this gate reports are the ones the tests actually observed.
MEASURED: dict[str, object] = {}


# --------------------------------------------------------------------------
# The two Protocol adapters (see the module docstring: P4 is unmerged)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RegistrySpec:
    """One ``models.Model`` as a ``control.ModelSpec``.

    Deliberately thin, and deliberately delegating to the REAL
    ``models.render_argv`` / ``models.render_env``: the point of the whole
    exercise is that the argv a model boots with comes out of ``models.toml``
    through P1's renderer, not out of a test's string literals.  The only thing
    this class adds is the two-argument ``render_argv(util, port)`` shape P3's
    Protocol asks for, plus the resolved ``ctx_tokens``.
    """

    model: models_mod.Model
    build: models_mod.Build
    ctx_tokens: int

    @property
    def key(self) -> str:
        return self.model.key

    @property
    def id(self) -> str:
        return self.model.id

    @property
    def slot(self) -> str:
        return self.model.slot

    @property
    def port(self) -> int:
        return self.model.port

    @property
    def vram_mib(self) -> int | None:
        return self.model.vram_mib

    @property
    def venv_bin(self) -> str:
        return str(Path(self.build.venv).expanduser() / "bin")

    def served_names(self) -> list[str]:
        return self.model.served_names()

    def render_argv(self, util: float, port: int) -> list[str]:
        return models_mod.render_argv(
            self.model, str(Path(self.venv_bin) / "vllm"), util, self.ctx_tokens, port
        )

    def render_env(self) -> dict[str, str]:
        env = dict(models_mod.render_env(self.model, self.build))
        # A transient unit inherits the user MANAGER's environment, not the
        # test runner's, and vLLM needs both of these to find its cache.
        env.setdefault("HOME", str(Path.home()))
        env.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env


class SpecRegistry:
    """``control.Registry``: one ``get(key)``, backed by a real ``models.Registry``."""

    def __init__(self, registry: models_mod.Registry, ctx_tokens: int = CTX) -> None:
        self.registry = registry
        self.specs = {
            key: RegistrySpec(
                model=model,
                build=registry.builds[model.build],
                ctx_tokens=ctx_tokens if isinstance(model.ctx, str) else int(model.ctx),
            )
            for key, model in registry.models.items()
        }

    def get(self, key: str) -> RegistrySpec:
        return self.specs[key]


class ControlRoutes:
    """``gateway.RouteTable`` over a real ``Control``.

    Liveness is a *pushed* snapshot, refreshed by :meth:`refresh`, never
    computed inside ``resolve()`` — ``control.live()`` runs a ``systemctl show``
    per unit plus an HTTP probe, and putting that on the critical path of every
    chat completion would run two subprocesses on the event loop thread.  An
    un-refreshed table reports nothing live, which fails toward 503+Retry-After
    rather than toward proxying at a port with nothing behind it.
    """

    def __init__(self, ctl: Control, registry: models_mod.Registry, ctx_tokens: int = CTX) -> None:
        self.ctl = ctl
        self.registry = registry
        self.ctx_tokens = ctx_tokens
        self._live: set[str] = set()

    def refresh(self) -> set[str]:
        self._live = {m.key for m in self.ctl.live() if m.ready}
        return set(self._live)

    def _route(self, model: models_mod.Model) -> Route:
        ctx = self.ctx_tokens if isinstance(model.ctx, str) else int(model.ctx)
        reasoning = model.reasoning
        return Route(
            model_id=model.id,
            port=model.port,
            live=model.key in self._live,
            aliases=tuple(model.aliases),
            presets=tuple(model.presets),
            policies=RoutePolicies(
                ctx=ctx,
                mirror_reasoning=bool(reasoning and reasoning.mirror_content),
                min_output_tokens=model.min_output_tokens,
            ),
        )

    # -- gateway.RouteTable ------------------------------------------------
    def resolve(self, name: str) -> Route | None:
        found = self.registry.resolve(name)
        if found is None:
            return None
        route = self._route(found.model)
        if found.preset is not None and found.overlay:
            route = Route(
                model_id=route.model_id,
                port=route.port,
                live=route.live,
                aliases=route.aliases,
                presets=route.presets,
                policies=RoutePolicies(
                    ctx=route.policies.ctx,
                    mirror_reasoning=route.policies.mirror_reasoning,
                    effort_overlay=dict(found.overlay),
                    min_output_tokens=route.policies.min_output_tokens,
                ),
            )
        return route

    def live_routes(self) -> list[Route]:
        return [
            self._route(m) for m in self.registry.models.values() if m.key in self._live
        ]

    def main(self) -> Route | None:
        for m in self.registry.models.values():
            if m.slot == "main" and m.key in self._live:
                return self._route(m)
        for m in self.registry.models.values():
            if m.slot == "main":
                return self._route(m)
        return None

    def known_names(self) -> list[str]:
        out: list[str] = []
        for m in self.registry.models.values():
            out.extend(m.served_names())
        return out


# --------------------------------------------------------------------------
# models.toml fixtures — written as TOML and loaded by the REAL loader
# --------------------------------------------------------------------------

_FIXTURE_TOML = f"""
[gpu]
total_mib  = {TOTAL_MIB}
margin_mib = 1024

[defaults.env]
VLLM_USE_FLASHINFER_SAMPLER = "0"

[builds.stock]
venv      = "/home/pctablet505/Projects/local_llm/.venv-llm-029"
cuda_home = "/home/pctablet505/Projects/local_llm/.venv-llm-029/lib/python3.13/site-packages/nvidia/cu13"

# Main slot A.  Aliases are the point: vLLM is started with every one of them
# as --served-model-name, so a client that still says "lfm2-vfy-a" keeps working.
[models.vfy-a]
id      = "{LFM2_ID}"
aliases = ["lfm2-vfy-a", "vfy-a"]
repo    = "{LFM2_REPO}"
slot    = "main"
port    = {PORT_MAIN_A}
build   = "stock"
ctx     = {CTX}
max_output_tokens = 1024
flags   = ["--max-num-seqs", "8", "--dtype", "bfloat16"]

[models.vfy-a.tools]
parser = "lfm2"

# Main slot B: a DIFFERENT id and port, so "the main slot is exclusive" is a
# claim about the slot and not about the port or the weights.
[models.vfy-b]
id      = "vfy-lfm2-b"
aliases = ["vfy-b"]
repo    = "{LFM2_REPO}"
slot    = "main"
port    = {PORT_MAIN_B}
build   = "stock"
ctx     = {CTX}
flags   = ["--max-num-seqs", "8", "--dtype", "bfloat16"]

# The resident.  Its util is DERIVED from vram_mib, so it exercises the other
# half of compute_util's arithmetic, and it carries the output floor.
[models.vfy-res]
id       = "vfy-lfm2-resident"
aliases  = ["vfy-res"]
repo     = "{LFM2_REPO}"
slot     = "resident"
vram_mib = 3300
port     = {PORT_RESIDENT}
build    = "stock"
ctx      = {CTX}
min_output_tokens = 24
flags    = ["--max-num-seqs", "8", "--dtype", "bfloat16"]

[models.vfy-res.tools]
parser = "lfm2"

# A boot that cannot succeed: --max-model-len past the checkpoint's ceiling.
[models.vfy-bad]
id      = "vfy-lfm2-bad"
repo    = "{LFM2_REPO}"
slot    = "resident"
vram_mib = 3300
port    = {PORT_BAD}
build   = "stock"
ctx     = 999999999
flags   = ["--max-num-seqs", "8", "--dtype", "bfloat16"]

# The only tiny checkpoint on this box with a thinking mode.  mirror_content is
# the whole subject of item 8.
[models.vfy-qwen]
id      = "{QWEN_ID}"
aliases = ["vfy-qwen"]
repo    = "{QWEN_REPO}"
slot    = "resident"
# Measured 2026-09-14: at 3300 MiB (util 0.03) vLLM reports
# "Available KV cache memory: -0.08 GiB" and the engine core refuses to start —
# weights 1.12 GiB + CUDA graphs 0.25 GiB + context leave nothing for KV.
vram_mib = 6000
port    = {PORT_QWEN}
build   = "stock"
ctx     = {CTX}

[models.vfy-qwen.reasoning]
parser = "qwen3"
mirror_content = true
"""


@pytest.fixture(scope="module")
def registry(tmp_path_factory) -> models_mod.Registry:
    """The fixture registry, through ``models.load`` — real validation included."""
    path = tmp_path_factory.mktemp("vfy") / "models.toml"
    path.write_text(_FIXTURE_TOML)
    return models_mod.load(path)


@pytest.fixture(scope="module")
def desired_path(tmp_path_factory) -> Path:
    """Desired state goes to a tmp dir, never to the worktree's state/."""
    return tmp_path_factory.mktemp("vfy-state") / "desired.json"


def make_control(registry: models_mod.Registry, desired_path: Path) -> Control:
    return Control(
        SpecRegistry(registry),
        unit_prefix="sd-test-",
        total_mib=TOTAL_MIB,
        desired_path=desired_path,
    )


# --------------------------------------------------------------------------
# Cleanup and guards — run whatever happens
# --------------------------------------------------------------------------

#: Units this file may create.  Named with a `vfy-` infix so they cannot
#: collide with another agent's `sd-test-*` units, and listed explicitly so the
#: finaliser's assertion is about THIS file's leaks.
OUR_UNITS = (
    "sd-test-vfy-a",
    "sd-test-vfy-b",
    "sd-test-vfy-res",
    "sd-test-vfy-bad",
    "sd-test-vfy-qwen",
)


def _stop_our_units() -> list[str]:
    """Stop every ``sd-test-vfy-*`` unit, by name, one at a time.

    By name and never by pattern: a glob handed to systemctl, or a ``pkill
    -f``, is the self-match trap this whole design exists to delete.
    Discovery is the same ``list-units`` call production uses, so this can only
    ever name units it could also have created.
    """
    stopped: list[str] = []
    for unit in units.list_units("sd-test-vfy-*"):
        try:
            units.stop(unit, timeout_s=90)
        except units.UnitError:  # pragma: no cover - best-effort teardown
            pass
        stopped.append(unit)
    return stopped


@pytest.fixture(scope="module", autouse=True)
def _no_units_left_behind():
    _stop_our_units()
    try:
        yield
    finally:
        _stop_our_units()
        leftover = [u for u in units.list_units("sd-test-*") if u.startswith("sd-test-vfy-")]
        assert leftover == [], f"this file leaked transient units: {leftover}"


@pytest.fixture(scope="module", autouse=True)
def _production_services_untouched():
    """The production units must be in the same state at the end as at the start.

    Read-only: ``systemctl --user show`` and ``systemctl is-active`` only.  The
    point is not to catch this file doing something wrong on purpose — it never
    signals a unit it did not start — but to catch it doing so by accident,
    e.g. through a glob or a prefix that turned out to match more than intended.
    """
    watched = ("lfm2-350m", "servedeck", "flashnext-reasoning-proxy", "qwen27b-reasoning-proxy")

    def snapshot() -> dict[str, tuple[str, str]]:
        out = {}
        for name in watched:
            props = units._parse_properties(
                units.default_runner()(
                    ["systemctl", "--user", "show", "-p", "ActiveState,MainPID", name]
                ).stdout
            )
            out[name] = (props.get("ActiveState", ""), props.get("MainPID", ""))
        return out

    before = snapshot()
    try:
        yield before
    finally:
        after = snapshot()
        assert after == before, (
            "a production unit changed state during this run: "
            + ", ".join(f"{k}: {before[k]} -> {after[k]}" for k in watched if before[k] != after[k])
        )


@pytest.fixture(scope="module", autouse=True)
def measurements():
    yield MEASURED
    if MEASURED:
        print("\n\n=== P7 measured numbers ===")
        for key in sorted(MEASURED):
            print(f"  {key}: {MEASURED[key]}")


def require_gpu_room(need_mib: int = MIN_FREE_MIB) -> int:
    free = gpu.free_mib()
    if free is None:
        pytest.skip("nvidia-smi did not report free VRAM")
    if free < need_mib:
        pytest.skip(
            f"only {free} MiB VRAM free, need >= {need_mib} MiB. A 90 GiB model "
            f"may be serving on this card; booting into the remainder would be "
            f"an outage, not a test failure."
        )
    if not VLLM.exists():
        pytest.skip(f"{VLLM} not present")
    return free


def wait_until(predicate, timeout_s: float, interval_s: float = 0.25):
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval_s)
    return last


def http_json(url: str, payload: dict | None = None, timeout_s: float = 120.0) -> dict | None:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            return json.loads(response.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def journal_has_boot_marker(unit: str, lines: int = 4000) -> int:
    """How many times ``Application startup complete.`` appears in the journal.

    The witness for "nothing was restarted": one boot, one marker.  MainPID is
    the other half, and the two together are much harder to fool than either —
    a unit could in principle be restarted onto the same pid number, and a pid
    can survive a re-exec.
    """
    return sum(
        1 for ln in units.journal_tail(unit, lines) if "Application startup complete." in ln
    )


# --------------------------------------------------------------------------
# Item 1 — the control plane against a real model, whole cycle
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_1_control_plane_starts_serves_and_releases_a_real_model(registry, desired_path):
    """start -> markers in order -> /v1/models -> chat -> stop -> VRAM back -> gone.

    The one test that walks the entire control plane on a real model, with the
    argv coming out of ``models.toml`` through P1's ``render_argv`` rather than
    out of a literal in this file.
    """
    free_at_start = require_gpu_room()
    ctl = make_control(registry, desired_path)

    progress: list[control.Progress] = []
    t0 = time.monotonic()
    result = ctl.start("vfy-a", timeout_s=BOOT_TIMEOUT_S, on_progress=progress.append)
    try:
        assert not isinstance(result, Refusal), result
        assert result.ready, f"{result.failure}\n" + "\n".join(result.journal)
        MEASURED["item1_boot_s"] = round(result.elapsed_s, 1)
        MEASURED["item1_wall_s"] = round(time.monotonic() - t0, 1)

        # (a) The four readiness markers, in the order vLLM emits them.
        assert result.markers == list(control.READY_MARKERS), (
            f"saw {result.markers}; tail:\n" + "\n".join(units.journal_tail("sd-test-vfy-a", 40))
        )
        assert [e.marker_index for e in progress if e.kind == "marker"] == [0, 1, 2, 3]
        assert progress[-1].kind == "ready"

        # (b) The argv is the registry's, computed util included, and the
        # owner's rule about eager mode survived the round trip.
        argv = result.argv
        assert argv[0] == str(VLLM) and argv[1] == "serve" and argv[2] == LFM2_REPO
        assert "--enforce-eager" not in argv
        assert argv[argv.index("--tool-call-parser") + 1] == "lfm2"
        assert argv[argv.index("--max-model-len") + 1] == str(CTX)
        # main slot: util is (free - margin)/total, floored to 2dp.
        assert result.util == control.floor2((free_at_start - 1024) / TOTAL_MIB)
        assert argv[argv.index("--gpu-memory-utilization") + 1] == f"{result.util:.2f}"

        # (c) The model is servable under every name vLLM was told to serve.
        listed = http_json(f"http://127.0.0.1:{PORT_MAIN_A}/v1/models")
        assert listed is not None
        ids = {entry["id"] for entry in listed["data"]}
        assert {LFM2_ID, "lfm2-vfy-a", "vfy-a"} <= ids, ids

        # (d) It answers.
        t_chat = time.perf_counter()
        completion = http_json(
            f"http://127.0.0.1:{PORT_MAIN_A}/v1/chat/completions",
            {
                "model": LFM2_ID,
                "messages": [{"role": "user", "content": "Reply with exactly: pineapple"}],
                "max_tokens": 32,
                "temperature": 0,
            },
        )
        assert completion is not None, "chat completion did not answer"
        content = completion["choices"][0]["message"]["content"]
        assert content, completion
        dt = time.perf_counter() - t_chat
        MEASURED["item1_direct_tok_s"] = round(
            completion["usage"]["completion_tokens"] / dt, 1
        )

        # (e) Discoverable the way servedeck rediscovers it after a restart.
        live = {m.key: m for m in ctl.live()}
        assert live["vfy-a"].ready and live["vfy-a"].pid > 0
        assert ctl.live_main(list(live.values())).key == "vfy-a"
        assert ctl.load_desired().main == "vfy-a"
    finally:
        stopped = ctl.stop("vfy-a")

    assert not isinstance(stopped, Refusal), stopped
    assert stopped.was_live
    held = stopped.held_mib
    assert held and held > 500, f"the driver attributed only {held} MiB to the unit's cgroup"
    MEASURED["item1_held_mib"] = held

    # The GPU really lets go.  >= 2 GiB, measured, not inferred from an exit code.
    free_after = wait_until(
        lambda: (gpu.free_mib() or 0) >= (stopped.free_before_mib or 0) + 2048, 180
    )
    final = gpu.free_mib() or 0
    MEASURED["item1_vram_released_mib"] = final - (stopped.free_before_mib or 0)
    assert free_after, (
        f"free VRAM went {stopped.free_before_mib} -> {final} MiB after the stop; "
        f"expected at least 2048 MiB back (the unit held {held})"
    )
    assert units.gone("sd-test-vfy-a") is True
    assert ctl.load_desired().main is None
    assert final >= free_at_start - 256


# --------------------------------------------------------------------------
# Item 5 — a boot that cannot succeed must not crash-loop
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_5_a_failed_boot_reports_honestly_and_does_not_crash_loop(registry, desired_path):
    """R4, against a real vLLM that really refuses.

    ``--max-model-len 999999999`` is past the checkpoint's ceiling, so vLLM
    exits non-zero every time.  Three things must hold together, and each was a
    real bug: ``wait_ready`` must report FAILURE (not a timeout) with the
    journal tail; ``Control`` must have STOPPED the unit rather than leave
    ``Restart=on-failure`` retrying unattended after the caller walked away;
    and nothing may be left in ``activating (auto-restart)``.
    """
    require_gpu_room()
    ctl = make_control(registry, desired_path)
    unit = "sd-test-vfy-bad"
    before = ctl.load_desired()

    result = ctl.start("vfy-bad", timeout_s=240.0, restart_sec=2)
    assert not isinstance(result, Refusal), f"it was refused before launching: {result}"
    assert not result.ready, "a --max-model-len of 999999999 booted; the premise is gone"
    assert result.failure, "a failed boot must say why"
    assert result.journal, "a failed boot must carry the journal tail"
    MEASURED["item5_failure"] = result.failure.split(";")[0][:120]
    MEASURED["item5_elapsed_s"] = round(result.elapsed_s, 1)

    # The failure is diagnosed, not merely timed out: the message names the
    # unit's fate, and the journal tail carries vLLM's own complaint.
    assert "timed out" not in result.failure, (
        f"reported a timeout for a unit that failed in {result.elapsed_s:.0f}s: {result.failure}"
    )
    tail = "\n".join(result.journal).lower()
    assert "max" in tail or "error" in tail or "traceback" in tail, result.journal

    # Nothing is left running or retrying.  `gone` and not `exists`: one
    # LoadState read is ~0.2% unreliable (units.exists).
    assert units.gone(unit), units.properties(unit, ("ActiveState", "SubState", "NRestarts"))
    state = units.show(unit)
    assert state.sub_state != "auto-restart", state
    assert not units.list_units(f"{unit}*")

    # And it is still there 12 s later — long enough for RestartSec=2 to have
    # fired six more times if the ceiling did not hold.
    time.sleep(12)
    assert units.gone(unit, attempts=2, delay_s=0.2), (
        "the unit came back after Control stopped it: "
        + str(units.properties(unit, ("ActiveState", "NRestarts")))
    )

    # A failed start must not erase an operator's recorded intent (control.py's
    # `units.stop, NOT self.stop` comment), and must not record a new one.
    after = ctl.load_desired()
    assert after.main == before.main
    assert "vfy-bad" not in after.residents


# --------------------------------------------------------------------------
# The shared live stack: one resident + one main, booted once
# --------------------------------------------------------------------------


@dataclass
class Stack:
    ctl: Control
    routes: ControlRoutes
    main_key: str = "vfy-a"
    resident_pid: int = 0
    starts: dict[str, control.StartResult] = field(default_factory=dict)
    free_before: dict[str, int] = field(default_factory=dict)


@pytest.fixture(scope="module")
def stack(registry, desired_path) -> Stack:
    """A resident and a main model, co-resident on the real card.

    Module-scoped because each boot is ~100 s of CUDA graph capture; the tests
    that share it only read, except the switch test, which is defined last and
    leaves the main slot holding ``b``.
    """
    require_gpu_room(MIN_FREE_MIB + 3300)
    ctl = make_control(registry, desired_path)
    st = Stack(ctl=ctl, routes=ControlRoutes(ctl, registry))

    for key in ("vfy-res", "vfy-a"):
        st.free_before[key] = gpu.free_mib() or 0
        result = ctl.start(key, timeout_s=BOOT_TIMEOUT_S)
        if isinstance(result, Refusal) or not result.ready:
            _stop_our_units()
            detail = result if isinstance(result, Refusal) else (
                result.failure, result.journal[-15:]
            )
            pytest.fail(f"the shared stack could not boot {key}: {detail}")
        st.starts[key] = result

    st.resident_pid = units.show("sd-test-vfy-res").main_pid
    st.routes.refresh()
    MEASURED["stack_boot_s"] = {k: round(v.elapsed_s, 1) for k, v in st.starts.items()}
    try:
        yield st
    finally:
        for key in ("vfy-a", "vfy-b", "vfy-res"):
            try:
                ctl.stop(key)
            except Exception:  # pragma: no cover - best-effort teardown
                pass


# --------------------------------------------------------------------------
# Item 3 — resident coexistence and the util arithmetic
# --------------------------------------------------------------------------


def _kv_line(unit: str) -> str:
    for line in units.journal_tail(unit, 4000):
        if "GPU KV cache size:" in line:
            return line
    return ""


@pytest.mark.slow
def test_3_resident_and_main_coexist_and_neither_was_starved(stack):
    """The §2.1 property: a resident can never make a main model refuse to boot.

    Two independent witnesses.  (1) Both models logged a KV cache size, which a
    vLLM that refused at startup never does — it dies in the memory profiler
    with "Free memory ... is less than desired".  (2) For each launch,
    ``ceil(total * util) <= free_at_launch``: whatever utilisation was computed,
    the fraction of the card it asks for was actually available at that moment.
    The resident's 3.3 GiB is simply not free when the main model computes its
    util, so it is never offered.
    """
    import math

    for key, unit in (("vfy-res", "sd-test-vfy-res"), ("vfy-a", "sd-test-vfy-a")):
        result = stack.starts[key]
        kv = _kv_line(unit)
        assert kv, f"{unit} never logged a KV cache size — it may have been starved"
        MEASURED[f"item3_kv_line_{key}"] = kv.split("GPU KV cache size:")[-1].strip()[:80]

        util = result.util
        assert util is not None and util > 0
        asked = math.ceil(TOTAL_MIB * util)
        free_at_launch = stack.free_before[key]
        assert asked <= free_at_launch, (
            f"{key}: util {util} asks for {asked} MiB of a card with "
            f"{free_at_launch} MiB free at launch"
        )
        # compute_util's stated post-condition: the launching worker's CUDA
        # context cushion survives.  It is not counted inside the fraction.
        assert asked <= free_at_launch - 700, (
            f"{key}: util {util} leaves only {free_at_launch - asked} MiB for the "
            f"worker's own CUDA context"
        )
        MEASURED[f"item3_util_{key}"] = (util, asked, free_at_launch)

    # The resident's util came from its budget; the main's from free memory.
    assert stack.starts["vfy-res"].util == control.floor2(3300 / TOTAL_MIB)
    assert stack.starts["vfy-a"].util > stack.starts["vfy-res"].util

    # Both are answering at the same time.  This is the state §2.1 describes
    # and that v1 could not reach.
    live = {m.key: m for m in stack.ctl.live()}
    assert live["vfy-res"].ready and live["vfy-a"].ready
    assert http_json(f"http://127.0.0.1:{PORT_RESIDENT}/v1/models") is not None
    assert http_json(f"http://127.0.0.1:{PORT_MAIN_A}/v1/models") is not None


# --------------------------------------------------------------------------
# Item 4 — reconcile restarts nothing
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_4_reconcile_leaves_running_models_alone(stack, registry, desired_path):
    """§2.2's rule with teeth: servedeck restarting is not a reason for a model to.

    A FRESH ``Control`` and a fresh ``Desired`` — the state a just-started
    servedeck is in — must adopt what is running instead of relaunching it.  v1
    launched the 27B seventy times in nine minutes doing exactly this.
    """
    from servedeck.desired import Desired

    pids_before = {u: units.show(u).main_pid for u in ("sd-test-vfy-res", "sd-test-vfy-a")}
    boots_before = {u: journal_has_boot_marker(u) for u in pids_before}
    # NOT "== 1": the stack fixture is module-scoped, so earlier items in this
    # file legitimately stopped and restarted these units.  The property under
    # test is that reconcile adds NO boot of its own, whatever the count is now.
    assert all(v >= 1 for v in boots_before.values()), (
        f"a unit under test never booted at all: {boots_before}"
    )

    fresh = make_control(registry, desired_path)
    outcome = fresh.reconcile(Desired(main="vfy-a", residents=["vfy-res"]), timeout_s=60.0)

    assert sorted(outcome.already_live) == ["vfy-a", "vfy-res"], outcome
    assert outcome.started == [], f"reconcile launched something: {outcome.started}"
    assert outcome.booting == [] and outcome.refused == []

    pids_after = {u: units.show(u).main_pid for u in pids_before}
    assert pids_after == pids_before, f"MainPID changed: {pids_before} -> {pids_after}"
    assert {u: journal_has_boot_marker(u) for u in pids_before} == boots_before
    assert all(units.show(u).n_restarts == 0 for u in pids_before)


# --------------------------------------------------------------------------
# Items 6 and 7 — the gateway over the real control plane
# --------------------------------------------------------------------------


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _Server(uvicorn.Server):
    def install_signal_handlers(self) -> None:  # background thread: no signals
        pass


@pytest.fixture(scope="module")
def gateway_url(stack):
    """The real gateway router, under uvicorn, over the real route table.

    A real socket and not ``ASGITransport``: httpx's in-process transport
    collects a streaming body before returning it, so an in-process test cannot
    tell a stream from a buffer however the gateway behaves.
    """
    router = gateway.build_router(stack.routes)
    client = router.gateway_client

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)

    config = uvicorn.Config(
        app, host="127.0.0.1", port=PORT_GATEWAY, log_level="warning",
        loop="asyncio", http="h11",
    )
    server = _Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    if not server.started:
        raise RuntimeError("the gateway under test did not start")
    try:
        yield f"http://127.0.0.1:{PORT_GATEWAY}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.mark.slow
def test_6a_models_lists_ids_and_aliases_of_live_routes_only(gateway_url, stack):
    """``GET /v1/models`` is the union of every LIVE model's every name.

    ``b``, ``bad`` and ``qwen`` are registered and not running, so they must be
    absent: an entry a client would pick and then get a 503 from is worse than
    no entry at all.
    """
    r = httpx.get(f"{gateway_url}/v1/models", timeout=15)
    assert r.status_code == 200
    entries = {m["id"]: m for m in r.json()["data"]}
    assert set(entries) == {
        LFM2_ID, "lfm2-vfy-a", "vfy-a", "vfy-lfm2-resident", "vfy-res",
    }, sorted(entries)
    assert entries["vfy-a"]["root"] == LFM2_ID
    assert entries["vfy-res"]["root"] == "vfy-lfm2-resident"
    assert entries[LFM2_ID]["max_model_len"] == CTX
    for name in ("vfy-lfm2-b", "vfy-b", "vfy-lfm2-bad", QWEN_ID):
        assert name not in entries


@pytest.mark.slow
def test_6b_a_request_naming_an_alias_reaches_the_model(gateway_url):
    """The 404-after-a-rename class of bug, gone by construction.

    ``vfy-a`` IS one of A's ``--served-model-name`` values, so this also proves
    the alias survives the launch; the resident's alias below is the stronger
    half — the gateway rewrites it to the served id either way, and an
    unrewritten name comes back as a 404 from vLLM itself.
    """
    for alias, expect in (("vfy-a", LFM2_ID), ("vfy-res", "vfy-lfm2-resident")):
        r = httpx.post(
            f"{gateway_url}/v1/chat/completions",
            json={
                "model": alias,
                "messages": [{"role": "user", "content": "Say OK"}],
                "max_tokens": 16,
                "temperature": 0,
            },
            timeout=120,
        )
        assert r.status_code == 200, f"{alias}: {r.status_code} {r.text}"
        assert r.json()["model"] == expect


@pytest.mark.slow
def test_6c_a_known_but_stopped_model_is_503_naming_the_slot_occupant(gateway_url, stack):
    """503 + Retry-After, and the body names what is actually in the main slot.

    That last part is the difference between a client that waits and a user who
    goes looking for a config file.  ``b`` holds the same exclusive slot ``a``
    is in, so the honest answer is "a has it".
    """
    r = httpx.post(
        f"{gateway_url}/v1/chat/completions",
        json={"model": "vfy-b", "messages": [{"role": "user", "content": "hi"}]},
        timeout=30,
    )
    assert r.status_code == 503, r.text
    assert r.headers["Retry-After"] == "15"
    body = r.json()["error"]
    assert body["code"] == "not_running"
    assert "vfy-lfm2-b" in body["message"]
    assert LFM2_ID in body["message"], (
        f"the 503 does not name the main-slot occupant: {body['message']!r}"
    )


@pytest.mark.slow
def test_6d_an_unknown_model_is_404_listing_the_known_names(gateway_url):
    r = httpx.post(
        f"{gateway_url}/v1/chat/completions",
        json={"model": "a-name-nobody-registered", "messages": []},
        timeout=30,
    )
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "model_not_found"
    # Every known name, live or not — the client is misconfigured and waiting
    # will not help, so it is told what it could have said.
    for name in (LFM2_ID, "vfy-a", "vfy-lfm2-b", QWEN_ID):
        assert name in err["message"], f"{name} missing from the 404 body"


@pytest.mark.slow
def test_6e_streaming_deltas_arrive_incrementally(gateway_url):
    """Timing AND read sizes, as P2's own e2e does.

    Either witness alone can pass for the wrong reason on a fast or a slow box;
    together they cannot.  A buffered 60 KB response reaches the client in a
    handful of ~16 KB reads, never in hundreds of ~200-byte ones.
    """
    chunks: list[tuple[float, bytes]] = []
    body = {
        "model": "vfy-a",
        "messages": [
            {"role": "user", "content": "Write every number from 1 to 200, one per line, nothing else."}
        ],
        "max_tokens": 700,
        "temperature": 0,
        "stream": True,
    }
    start = time.perf_counter()
    with httpx.stream("POST", f"{gateway_url}/v1/chat/completions", json=body, timeout=180) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for chunk in r.iter_bytes():
            if chunk:
                chunks.append((time.perf_counter() - start, chunk))

    total = sum(len(c) for _, c in chunks)
    t_first, t_last = chunks[0][0], chunks[-1][0]
    MEASURED["item6_gateway_ttft_s"] = round(t_first, 4)
    MEASURED["item6_gateway_stream_s"] = round(t_last, 3)
    MEASURED["item6_gateway_reads"] = len(chunks)
    MEASURED["item6_gateway_bytes_per_read"] = round(total / len(chunks), 1)

    assert len(chunks) >= 20, f"only {len(chunks)} network reads — the stream was buffered"
    assert total / len(chunks) < 2000, (
        f"{total / len(chunks):.0f} bytes per read — a buffer being drained, not SSE frames"
    )
    assert t_last - t_first > 0.05, "the generation was too short to tell a stream from a buffer"
    assert t_first < t_last * 0.4, (
        f"first chunk at {t_first:.3f}s of a {t_last:.3f}s stream — the response is buffered"
    )
    raw = b"".join(c for _, c in chunks)
    assert raw.rstrip().endswith(b"data: [DONE]")


@pytest.mark.slow
def test_6f_time_to_first_token_direct_versus_through_the_gateway(gateway_url):
    """How much the normalising hop costs, measured rather than assumed.

    Reported, not asserted below a threshold: the number that matters to the
    owner is the delta, and pinning it to a constant would make this test a
    weather report about the box.
    """
    body = {
        "model": LFM2_ID,
        "messages": [{"role": "user", "content": "Write every number from 1 to 120, one per line."}],
        "max_tokens": 400,
        "temperature": 0,
        "stream": True,
    }

    def ttft(url: str, model: str) -> tuple[float, float, int]:
        payload = dict(body, model=model)
        start = time.perf_counter()
        first = None
        n = 0
        with httpx.stream("POST", f"{url}/v1/chat/completions", json=payload, timeout=180) as r:
            assert r.status_code == 200, r.read()
            for chunk in r.iter_bytes():
                if chunk:
                    n += 1
                    if first is None:
                        first = time.perf_counter() - start
        return first or 0.0, time.perf_counter() - start, n

    direct = ttft(f"http://127.0.0.1:{PORT_MAIN_A}", LFM2_ID)
    through = ttft(gateway_url, "vfy-a")
    MEASURED["item6_ttft_direct_s"] = round(direct[0], 4)
    MEASURED["item6_ttft_gateway_s"] = round(through[0], 4)
    MEASURED["item6_ttft_overhead_ms"] = round((through[0] - direct[0]) * 1000, 1)
    assert direct[2] > 10 and through[2] > 10


@pytest.mark.slow
def test_6g_a_tool_call_round_trips_through_the_lfm2_parser(gateway_url):
    """``--tool-call-parser lfm2`` came out of ``models.toml``; the parsed call
    must reach the client with its arguments intact."""
    body = {
        "model": "vfy-a",
        "messages": [{"role": "user", "content": "What is the weather in Paris right now?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string", "description": "City name"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 200,
        "temperature": 0,
    }
    r = httpx.post(f"{gateway_url}/v1/chat/completions", json=body, timeout=180)
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    assert calls, f"no tool call in {choice}"
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"])["city"].lower().startswith("paris")
    assert choice["finish_reason"] == "tool_calls"


@pytest.mark.slow
def test_6h_json_schema_response_format_returns_valid_json(gateway_url):
    """Structured output survives the hop.  The gateway does not touch the
    response body on this path, so a failure here means it disturbed something
    it should have relayed."""
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
        "required": ["city", "country"],
        "additionalProperties": False,
    }
    r = httpx.post(
        f"{gateway_url}/v1/chat/completions",
        json={
            "model": "vfy-a",
            "messages": [{"role": "user", "content": "Paris is in France. Fill the schema."}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "place", "schema": schema, "strict": True},
            },
            "max_tokens": 120,
            "temperature": 0,
        },
        timeout=180,
    )
    assert r.status_code == 200, r.text
    content = r.json()["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    assert set(parsed) == {"city", "country"}, parsed


@pytest.mark.slow
def test_6i_a_3mb_body_with_no_model_key_reaches_the_model_unbuffered(gateway_url, stack):
    """Two properties in one request, and they pull in opposite directions.

    (1) **Routing**: a body with no top-level ``model`` goes to the main slot,
    not to a 503 — that is what makes ``/health``-style probes and every client
    that leaves the field out work.  The reply's ``model`` names A, so it really
    went there.

    (2) **Memory**: the body is 3 MB, past ``_MODEL_SCAN_LIMIT`` (2 MiB), so the
    scan must give up and stream the remainder.  The witness is the peak RSS of
    the gateway process across the request: buffering 3 MB would show, streaming
    it cannot.  ``tracemalloc`` would not see it — the bytes are in a
    ``bytearray`` the gateway owns, which is exactly what RSS measures.
    """
    import resource

    filler = "q" * 3_000_000
    payload = {
        "messages": [
            {"role": "system", "content": filler},
            {"role": "user", "content": "Reply with exactly: ok"},
        ],
        "max_tokens": 16,
        "temperature": 0,
    }
    raw = json.dumps(payload).encode()
    assert b'"model"' not in raw
    assert len(raw) > gateway._MODEL_SCAN_LIMIT

    def gateway_rss_kib() -> int:
        # The gateway runs in THIS process (a uvicorn thread), so our own RSS is
        # its RSS.  ru_maxrss is a high-water mark, which is the right shape for
        # "did it ever hold the whole body".
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    before = gateway_rss_kib()
    r = httpx.post(
        f"{gateway_url}/v1/chat/completions",
        content=raw,
        headers={"content-type": "application/json"},
        timeout=300,
    )
    after = gateway_rss_kib()

    # The whole 3 MB reached vLLM: only the model itself can measure the prompt
    # against its own context window, and it did (a BadRequestError naming the
    # 8192-token limit).  A gateway that truncated, buffered-and-failed, or
    # routed nowhere would have produced its own 4xx/5xx instead, with a
    # `servedeck_*` error type.  That vLLM answered at all IS the delivery proof.
    assert r.status_code == 400, r.text[:400]
    err = r.json()["error"]
    assert err.get("type") == "BadRequestError", err
    assert "maximum context length" in err.get("message", ""), err
    assert not str(err.get("type", "")).startswith("servedeck"), (
        "the gateway answered this itself; the body never reached the model"
    )
    growth_mib = (after - before) / 1024
    MEASURED["item6_3mb_rss_growth_mib"] = round(growth_mib, 1)
    MEASURED["item6_3mb_scan_limit_mib"] = gateway._MODEL_SCAN_LIMIT / 1024 / 1024
    # The scan ceiling is 2 MiB, so up to that much is expected; a gateway that
    # read the body to EOF would need the full 3 MB plus httpx's copy of it.
    assert growth_mib < 6.0, (
        f"the gateway's peak RSS grew {growth_mib:.1f} MiB across a 3 MB request — "
        f"that is the whole body being held, not a 2 MiB scan prefix"
    )


@pytest.mark.slow
def test_7_the_output_floor_is_applied_upstream(gateway_url, stack):
    """``min_output_tokens`` on the resident route, proven on a real server.

    The GLM thinking-budget fix: a small ``max_tokens`` is spent thinking and
    the turn returns ``finish_reason="length"`` with ``content: null``.  So the
    floor must RAISE an explicitly-small budget, and — the half that was a bug
    once — must leave a large one alone.
    """
    floor = stack.routes.resolve("vfy-res").policies.min_output_tokens
    assert floor == 24, "the resident route lost its min_output_tokens"

    small = httpx.post(
        f"{gateway_url}/v1/chat/completions",
        json={
            "model": "vfy-res",
            "messages": [{"role": "user", "content": "Write a long paragraph about rain."}],
            "max_tokens": 1,
            "temperature": 0,
        },
        timeout=120,
    )
    assert small.status_code == 200, small.text
    used = small.json()["usage"]["completion_tokens"]
    MEASURED["item7_completion_tokens_for_max_tokens_1"] = used
    assert used > 1, (
        f"max_tokens: 1 produced {used} completion tokens — the floor of {floor} "
        f"was not applied to the upstream request"
    )
    assert used <= floor, f"{used} tokens exceeds the floor of {floor}; it was not a clamp"

    # A caller who asked for more keeps what they asked for.  Proven by the
    # finish reason: a request capped at 24 would stop on length, this one does
    # not, and the completion is longer than the floor.
    big = httpx.post(
        f"{gateway_url}/v1/chat/completions",
        json={
            "model": "vfy-res",
            "messages": [
                {"role": "user", "content": "Write out every number from 1 to 90, one per line."}
            ],
            "max_tokens": 400,
            "temperature": 0,
        },
        timeout=180,
    )
    assert big.status_code == 200, big.text
    big_used = big.json()["usage"]["completion_tokens"]
    MEASURED["item7_large_budget_completion_tokens"] = big_used
    assert big_used > floor, (
        f"a request with max_tokens: 400 produced only {big_used} tokens — the "
        f"floor is behaving as a cap"
    )


# --------------------------------------------------------------------------
# Item 10 — measured beats estimated (closes sweep item F10)
# --------------------------------------------------------------------------


def _prom_labels(metrics_text: str, metric: str) -> dict[str, str]:
    """Labels of the first sample of ``metric`` in a Prometheus exposition."""
    for line in metrics_text.splitlines():
        if not line.startswith(metric + "{"):
            continue
        inner = line[len(metric) + 1 : line.rindex("}")]
        out: dict[str, str] = {}
        for pair in inner.split('",'):
            key, _, value = pair.partition('="')
            out[key.strip()] = value.strip().strip('"')
        return out
    return {}


@pytest.mark.slow
def test_10_predicted_kv_size_versus_what_vllm_reports(stack):
    """The estimator against the instrument, on the same server.

    ``vllm:cache_config_info`` carries ``kv_cache_size_tokens`` — vLLM's own
    count after it profiled the card.  ``kvcalc.geometry`` predicts the same
    number from ``config.json`` plus the KV bytes the worker said it had.
    "Measured beats estimated" (§2.5) is only a rule if somebody checks the
    estimate, and nothing in the repository did.
    """
    from servedeck import discovery as registry_mod

    metrics = httpx.get(f"http://127.0.0.1:{PORT_MAIN_A}/metrics", timeout=15).text
    labels = _prom_labels(metrics, "vllm:cache_config_info")
    assert labels, "vllm:cache_config_info is not on /metrics; the premise is gone"
    actual_tokens = int(labels["kv_cache_size_tokens"])
    MEASURED["item10_vllm_kv_tokens"] = actual_tokens
    MEASURED["item10_vllm_gpu_blocks"] = labels.get("num_gpu_blocks")

    # The KV byte budget the worker actually settled on, from its own log line.
    kv_gib = None
    for line in units.journal_tail("sd-test-vfy-a", 4000):
        if "Available KV cache memory:" in line:
            kv_gib = float(line.split("Available KV cache memory:")[-1].split("GiB")[0].strip())
    assert kv_gib, "vLLM did not log 'Available KV cache memory'"
    MEASURED["item10_vllm_kv_gib"] = kv_gib

    cfg = registry_mod.load_model_config(LFM2_REPO)
    assert cfg is not None, "LFM2's config.json is not in the local hub cache"
    geo = kvcalc.geometry(cfg)
    predicted_tokens = geo.tokens_for(kv_gib * 1024**3, CTX)
    implied_bytes = kv_gib * 1024**3 / actual_tokens

    MEASURED["item10_kvcalc_family"] = geo.family
    MEASURED["item10_kvcalc_bytes_per_token"] = round(geo.bytes_per_token(CTX), 1)
    MEASURED["item10_vllm_implied_bytes_per_token"] = round(implied_bytes, 1)
    MEASURED["item10_predicted_tokens"] = predicted_tokens
    error = (predicted_tokens - actual_tokens) / actual_tokens
    MEASURED["item10_relative_error"] = f"{error:+.1%}"

    # This is the assertion the fix below has to earn.  Before it, kvcalc
    # classified LFM2 as `dense_gqa` and counted all 16 layers as attention
    # layers -- 32,768 B/token against vLLM's 12,373 -- so it predicted 46,530
    # tokens where the server reported 123,229: -62.2%.
    assert abs(error) <= 0.20, (
        f"kvcalc predicts {predicted_tokens} KV tokens for {kv_gib} GiB where vLLM "
        f"reports {actual_tokens} ({error:+.1%}). Its per-token rate is "
        f"{geo.bytes_per_token(CTX):.0f} B against the server's implied "
        f"{implied_bytes:.0f} B; family={geo.family}."
    )


# --------------------------------------------------------------------------
# Item 2 — main-slot exclusivity and the switch (leaves the slot holding `b`)
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_2_main_slot_is_exclusive_and_switch_waits_for_the_card(stack):
    """Defined last among the stack's tests, because it replaces the main model.

    Three claims: starting a second main model is REFUSED and the refusal names
    the holder; ``switch`` stops the holder, **measures** the memory coming back
    and only then boots the replacement; and the resident is untouched
    throughout — same MainPID, one boot marker, never signalled.
    """
    ctl = stack.ctl
    resident_pid_before = units.show("sd-test-vfy-res").main_pid
    resident_boots_before = journal_has_boot_marker("sd-test-vfy-res")
    assert resident_pid_before == stack.resident_pid

    # (1) Exclusivity.  Returned, not raised: this is an operational answer.
    refusal = ctl.start("vfy-b", timeout_s=30.0)
    assert isinstance(refusal, Refusal), f"a second main model was allowed to start: {refusal}"
    assert refusal.reason == "main_slot_busy"
    assert refusal.live_key == "vfy-a"
    assert "vfy-a" in refusal.message and "switch" in refusal.message
    assert units.gone("sd-test-vfy-b"), "the refusal still created a unit"
    MEASURED["item2_refusal"] = refusal.message[:120]

    # (2) The switch.
    free_before_switch = gpu.free_mib() or 0
    t0 = time.monotonic()
    result = ctl.switch("vfy-b", timeout_s=BOOT_TIMEOUT_S)
    MEASURED["item2_switch_wall_s"] = round(time.monotonic() - t0, 1)
    assert not isinstance(result, Refusal), result

    assert result.stopped is not None and result.stopped.key == "vfy-a"
    held = result.stopped.held_mib
    assert held and held > 500, f"the stopped model was credited with {held} MiB"
    # The MEASURED release, not the return value: free_mib rose past the bar
    # `_wait_for_target` was aiming at, and `switch` waited for it.
    target = (result.stopped.free_before_mib or 0) + control.RELEASE_FRACTION * held
    assert result.released is True
    assert result.free_after_mib is not None and result.free_after_mib >= target, (
        f"released=True with {result.free_after_mib} MiB free against a target of {int(target)}"
    )
    assert result.waited_s >= 0.0
    MEASURED["item2_held_mib"] = held
    MEASURED["item2_release_wait_s"] = round(result.waited_s, 2)
    MEASURED["item2_free_before_after"] = (result.stopped.free_before_mib, result.free_after_mib)
    assert units.gone("sd-test-vfy-a"), "the old main model's unit is still loaded"

    started = result.started
    assert not isinstance(started, Refusal), started
    assert started is not None and started.ready, (
        f"{started.failure}\n" + "\n".join(started.journal[-15:])
    )
    MEASURED["item2_b_boot_s"] = round(started.elapsed_s, 1)
    assert started.markers == list(control.READY_MARKERS)
    listed = http_json(f"http://127.0.0.1:{PORT_MAIN_B}/v1/models")
    assert listed is not None
    assert "vfy-lfm2-b" in {e["id"] for e in listed["data"]}
    assert ctl.load_desired().main == "vfy-b"
    stack.main_key = "vfy-b"

    # (3) The resident never noticed.  Same pid, one boot, no restarts.
    assert units.show("sd-test-vfy-res").main_pid == resident_pid_before
    assert units.show("sd-test-vfy-res").n_restarts == 0
    # Unchanged ACROSS THE SWITCH, not "exactly one boot ever": the stack is
    # module-scoped and earlier items may legitimately have restarted it.
    assert journal_has_boot_marker("sd-test-vfy-res") == resident_boots_before
    assert http_json(f"http://127.0.0.1:{PORT_RESIDENT}/v1/models") is not None


# --------------------------------------------------------------------------
# Item 8 — the reasoning mirror on a real stream
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qwen(registry, desired_path):
    """Qwen3-0.6B with ``--reasoning-parser qwen3``, through the real Control."""
    ctl = make_control(registry, desired_path)
    # These are the LAST items in the file, and the reasoning mirror is the most
    # load-bearing behaviour in the gateway — it must never be skipped merely
    # because the earlier items' models are still holding the card.  Stop them
    # first and wait for the memory to come back, then check the room.
    for key in ("vfy-a", "vfy-b", "vfy-res"):
        try:
            ctl.stop(key)
        except Exception:
            pass
    deadline = time.time() + 120
    while time.time() < deadline and (gpu.free_mib() or 0) < 8192:
        time.sleep(2)
    require_gpu_room()
    result = ctl.start("vfy-qwen", timeout_s=BOOT_TIMEOUT_S)
    if isinstance(result, Refusal) or not result.ready:
        try:
            units.stop("sd-test-vfy-qwen", timeout_s=90)
        except units.UnitError:
            pass
        detail = result if isinstance(result, Refusal) else (result.failure, result.journal[-15:])
        pytest.skip(f"Qwen3-0.6B did not boot: {detail}")
    MEASURED["item8_qwen_boot_s"] = round(result.elapsed_s, 1)
    try:
        yield ctl
    finally:
        ctl.stop("vfy-qwen")


@pytest.fixture(scope="module")
def qwen_gateway(qwen, registry):
    """A second gateway instance whose route table has the Qwen route live.

    Its own uvicorn on an ephemeral loopback port, so it does not depend on the
    main stack's fixture still holding :8055 — these tests must be runnable on
    their own.
    """
    routes = ControlRoutes(qwen, registry)
    routes.refresh()
    assert routes.resolve(QWEN_ID).live, "the route table does not see the Qwen unit"
    assert routes.resolve(QWEN_ID).policies.mirror_reasoning is True

    router = gateway.build_router(routes)
    client = router.gateway_client

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    port = _free_port()
    server = _Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                       loop="asyncio", http="h11")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    if not server.started:
        raise RuntimeError("the qwen gateway under test did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


THINK = {"role": "user", "content": "What is 17 times 3? Think it through, then answer."}


@pytest.mark.slow
def test_8a_the_premise_upstream_emits_reasoning_and_not_reasoning_content(qwen):
    """THE PREMISE of the whole mirror.  If this fails, say so loudly.

    vLLM >= 0.29 renamed the chat-completions reasoning field to ``reasoning``
    and offers no flag to emit the old name.  A chat-completions client that
    reads ``reasoning_content`` therefore sees no thinking, cannot store it and
    cannot send it back — the measured mechanism behind multi-turn agent
    amnesia.  If ``reasoning_content`` ever appears here on its own, the server
    changed and the mirror is no longer the fix; if ``reasoning`` disappears,
    the mirror has nothing to copy.
    """
    body = http_json(
        f"http://127.0.0.1:{PORT_QWEN}/v1/chat/completions",
        {"model": QWEN_ID, "messages": [THINK], "max_tokens": 256, "temperature": 0},
    )
    assert body is not None, "Qwen did not answer on its own port"
    message = body["choices"][0]["message"]
    MEASURED["item8_upstream_message_keys"] = sorted(message)
    assert isinstance(message.get("reasoning"), str) and message["reasoning"].strip(), (
        f"vLLM 0.29 did not emit a 'reasoning' field: keys were {sorted(message)}. "
        f"THE MIRROR'S PREMISE IS GONE — re-derive what the server emits before "
        f"trusting any mirroring test."
    )
    assert "reasoning_content" not in message or message["reasoning_content"] is None, (
        f"vLLM now emits reasoning_content itself ({message.get('reasoning_content')!r}); "
        f"the gateway's mirror is no longer what supplies it and the amnesia fix "
        f"must be re-justified."
    )


@pytest.mark.slow
def test_8b_the_mirror_supplies_both_fields_in_a_json_response(qwen_gateway):
    """Through the gateway, both names are present and EQUAL."""
    r = httpx.post(
        f"{qwen_gateway}/v1/chat/completions",
        json={"model": QWEN_ID, "messages": [THINK], "max_tokens": 256, "temperature": 0},
        timeout=180,
    )
    assert r.status_code == 200, r.text
    message = r.json()["choices"][0]["message"]
    assert isinstance(message.get("reasoning"), str) and message["reasoning"].strip()
    assert message.get("reasoning_content") == message["reasoning"], (
        f"reasoning_content={message.get('reasoning_content')!r} != "
        f"reasoning={message.get('reasoning')!r}"
    )
    MEASURED["item8_json_reasoning_chars"] = len(message["reasoning"])


@pytest.mark.slow
def test_8c_the_mirror_holds_across_a_real_sse_stream(qwen_gateway):
    """The load-bearing half: a real SSE stream, mirrored delta by delta.

    Chunk boundaries land anywhere — mid-JSON, mid-UTF-8 — so this is where
    ``policies.sse_stream``'s hold-back either works against a real server or
    does not.  Reassembling both fields across every delta and requiring them
    to be the same string is the check a client's own accumulator performs.
    """
    body = {
        "model": "vfy-qwen",
        "messages": [THINK],
        "max_tokens": 400,
        "temperature": 0,
        "stream": True,
    }
    chunks: list[tuple[float, bytes]] = []
    start = time.perf_counter()
    with httpx.stream("POST", f"{qwen_gateway}/v1/chat/completions", json=body, timeout=180) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for chunk in r.iter_bytes():
            if chunk:
                chunks.append((time.perf_counter() - start, chunk))

    raw = b"".join(c for _, c in chunks)
    events = []
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload in (b"[DONE]", b""):
            continue
        events.append(json.loads(payload))

    def joined(field: str) -> str:
        return "".join(
            (ch.get("delta") or {}).get(field) or ""
            for ev in events
            for ch in ev.get("choices") or []
            if isinstance((ch.get("delta") or {}).get(field), str)
        )

    reasoning = joined("reasoning")
    mirrored = joined("reasoning_content")
    MEASURED["item8_sse_events"] = len(events)
    MEASURED["item8_sse_reasoning_chars"] = len(reasoning)
    MEASURED["item8_sse_reads"] = len(chunks)

    assert reasoning.strip(), "no reasoning deltas in the stream at all"
    assert mirrored == reasoning, (
        f"reassembled reasoning_content ({len(mirrored)} chars) != reasoning "
        f"({len(reasoning)} chars) across {len(events)} SSE events"
    )
    # Every delta that carried one name carried the other, not just the totals.
    for ev in events:
        for ch in ev.get("choices") or []:
            delta = ch.get("delta") or {}
            if isinstance(delta.get("reasoning"), str) and delta["reasoning"]:
                assert delta.get("reasoning_content") == delta["reasoning"], delta
    # Still a stream, not a buffer, with the transform in the path.
    assert len(chunks) >= 20, f"only {len(chunks)} reads — sse_stream buffered the response"
    assert chunks[0][0] < chunks[-1][0] * 0.5


@pytest.mark.slow
def test_8d_a_second_turn_that_resends_reasoning_content_is_accepted(qwen_gateway):
    """The round trip closed: the client stores what it was given and sends it back.

    This is the shape that broke.  A client that receives ``reasoning_content``
    puts the assistant turn back into ``messages`` with that key present; if the
    server rejected it the client would have to strip it, which is precisely the
    knowledge the gateway exists to stop clients needing.
    """
    first = httpx.post(
        f"{qwen_gateway}/v1/chat/completions",
        json={"model": QWEN_ID, "messages": [THINK], "max_tokens": 256, "temperature": 0},
        timeout=180,
    )
    assert first.status_code == 200, first.text
    message = first.json()["choices"][0]["message"]
    assert message.get("reasoning_content"), "nothing to echo back"

    echoed = {
        "role": "assistant",
        "content": message.get("content") or "",
        "reasoning_content": message["reasoning_content"],
    }
    second = httpx.post(
        f"{qwen_gateway}/v1/chat/completions",
        json={
            "model": QWEN_ID,
            "messages": [THINK, echoed, {"role": "user", "content": "Now double that answer."}],
            "max_tokens": 256,
            "temperature": 0,
        },
        timeout=180,
    )
    assert second.status_code == 200, (
        f"the server rejected a turn carrying reasoning_content: "
        f"HTTP {second.status_code} {second.text[:400]}"
    )
    reply = second.json()["choices"][0]["message"]
    assert (reply.get("content") or "").strip() or (reply.get("reasoning") or "").strip(), reply
    MEASURED["item8_second_turn"] = "accepted"


# --------------------------------------------------------------------------
# Item 9 — doctor and wire against the real box (no GPU, no boot)
# --------------------------------------------------------------------------

REAL_MODELS_TOML = Path(__file__).resolve().parent.parent / "models.toml"


def _servedeck_cli(*args: str) -> tuple[int, str, str]:
    """Run the real console script in a subprocess, as an operator would.

    A subprocess and not ``cli.main([...])`` in-process: the exit code is half
    of what item 9 is about, and ``main`` returning an int is not the same
    claim as the command exiting with it.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "servedeck.cli", *args],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(REAL_MODELS_TOML.parent),
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_9a_doctor_reports_the_real_box_honestly():
    """``servedeck doctor`` against the REAL ``models.toml``.

    The honest expected picture on this box: the production LFM2 answers on
    :8007 with ``LFM2.5-350M``; the three big models (27B :8004, Flash-Next
    :8001, GLM :8002) are not running.  "Not running" must NOT be a failure —
    the main slot is exclusive, so at most one of the three can ever be up, and
    a doctor that goes red for the other two is a doctor nobody reads.
    """
    from servedeck import doctor

    results = doctor.run_doctor(REAL_MODELS_TOML, timeout=3.0)
    by_name = {r.name: r for r in results}
    table = doctor.format_table(results)
    MEASURED["item9_doctor_fail_count"] = sum(1 for r in results if not r.ok)
    MEASURED["item9_doctor_failures"] = [r.name for r in results if not r.ok]

    assert by_name["registry loads"].ok, table

    # The production resident really is live, and doctor sees it.
    lfm2 = by_name["port 8007 (lfm2)"]
    assert lfm2.ok and "listening" in lfm2.detail, lfm2

    # A model that is simply not running is reported as such, and is OK.
    for key, port in (("qwen27b", 8004), ("flashnext", 8001), ("glm53", 8002)):
        port_check = by_name[f"port {port} ({key})"]
        unit_check = by_name[f"unit (model-{key})"]
        assert port_check.ok, (
            f"doctor FAILS for {key}, which is merely not running: {port_check.detail}. "
            f"A model not occupying the exclusive main slot is the normal case."
        )
        assert unit_check.ok and unit_check.detail == "not running", unit_check

    # No stray model-* unit belongs to a key models.toml does not know.
    strays = [r for r in results if r.name.startswith("unit (model-") and not r.ok]
    assert strays == [], strays

    # And the exit code is explainable: it is 0 iff nothing failed, and every
    # failure is a CLIENT CONFIG reference, which is the drift doctor exists to
    # surface (REDESIGN §4 R1 measured six broken client entries).
    code, out, err = _servedeck_cli("doctor", "--models-toml", str(REAL_MODELS_TOML))
    MEASURED["item9_doctor_exit_code"] = code
    failures = [r for r in results if not r.ok]
    assert code == (0 if not failures else 1), f"exit {code} with {len(failures)} failures"
    assert all(
        r.name.split(":")[0] in ("vscode", "codex", "kimi") for r in failures
    ), (
        "doctor exits non-zero for something other than client-config drift: "
        + str([(r.name, r.detail) for r in failures])
    )
    assert "[OK  ]" in out or "[FAIL]" in out, out


def test_9b_wire_dry_run_diffs_the_real_client_configs_and_writes_nothing():
    """``servedeck wire`` must be able to show a diff without touching a byte.

    Both real targets that exist on this box are checked, and the proof that
    nothing was written is the file's mtime, size and sha256 before and after —
    an assertion about content alone would pass for a write that happened to
    produce the same bytes, which is not the claim.
    """
    import hashlib

    from servedeck import wire

    targets = [t for t in wire.WIRE_TARGETS if t.path.is_file()]
    assert targets, "no real client config exists on this box; nothing to diff"
    MEASURED["item9_wire_targets"] = [t.path.name for t in targets]

    def fingerprint(path: Path) -> tuple[float, int, str]:
        stat = path.stat()
        return (stat.st_mtime, stat.st_size, hashlib.sha256(path.read_bytes()).hexdigest())

    before = {t.path: fingerprint(t.path) for t in targets}

    code, out, err = _servedeck_cli("wire", "--models-toml", str(REAL_MODELS_TOML))
    assert code == 0, f"exit {code}\n{out}\n{err}"

    after = {path: fingerprint(path) for path in before}
    assert after == before, (
        "wire wrote to a real client config in dry-run mode: "
        + str({str(p): (before[p], after[p]) for p in before if before[p] != after[p]})
    )

    # A diff for each target that would change, naming the gateway URL.
    changed = 0
    for target in targets:
        text = wire.read_existing(target.path)
        reg = models_mod.load(REAL_MODELS_TOML)
        rendered = target.render(reg, text, resolve_ctx=wire.make_default_ctx_resolver(reg))
        if rendered != text:
            changed += 1
            assert f"== {target.name}: would change" in out, out
    assert changed >= 2, (
        f"only {changed} of the real client configs would change; the point of "
        f"item 9 is a diff against the box's actual drift"
    )
    MEASURED["item9_wire_would_change"] = changed
    assert "8010" in out, "the generated config does not point clients at the gateway"
    assert "--apply" in out, "the dry-run does not say how to apply"


def test_9c_the_documented_dry_run_flag_exists():
    """§2.4 and §3 step 1 both spell the command ``servedeck wire --dry-run``.

    An operator following the design doc types that, and before the fix below
    argparse answered ``unrecognized arguments: --dry-run`` and exit 2 — which
    during a cutover reads as "wire is broken", not as "the flag is spelled
    differently".  Dry run stays the DEFAULT; the flag is explicit and
    mutually exclusive with ``--apply``, so ``--dry-run --apply`` is refused
    rather than silently resolved in one direction.
    """
    code, out, err = _servedeck_cli(
        "wire", "--dry-run", "--models-toml", str(REAL_MODELS_TOML)
    )
    assert code == 0, f"`servedeck wire --dry-run` exited {code}: {err}"
    assert "would change" in out or "no changes" in out, out

    code2, _out2, err2 = _servedeck_cli(
        "wire", "--dry-run", "--apply", "--models-toml", str(REAL_MODELS_TOML)
    )
    assert code2 != 0, "--dry-run --apply was accepted; one of them silently lost"
    assert "not allowed with" in err2 or "argument" in err2, err2
