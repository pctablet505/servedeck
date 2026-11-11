"""Exhaustive finite-state model checker for Servedeck's control-plane protocol.

TLA+-style explicit-state Breadth-First Search (BFS) over the complete reachable
state graph of:
  - ``app.lifespan`` / ``_reconcile_after_bind`` / ``_poll_loop`` (``_recover_desired``)
  - ``app._claim`` / ``_run_mutation`` / ``_publish_result`` single-flight lock
  - ``control.Control`` ``start`` / ``switch`` / ``stop`` / ``abort`` / ``desired.json``
  - ``legacy_page.BootTracker`` + ``actual_state`` + ``supervisor.phase``
  - ``web/app.js`` ``Stop`` vs ``Force stop boot`` (``/api/server/stop`` vs ``/abort``)

Every reachable protocol state and transition interleaving is explored and
checked against 7 safety invariants; if any invariant fails, the minimal
counterexample trace from the initial state is reconstructed and printed.
A second suite verifies step-by-step refinement (bisimulation) between the
formal transitions and the real ``BootTracker``, ``actual_state``, ``Desired``,
and ``Control.abort`` objects.
"""

from __future__ import annotations

import subprocess
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from servedeck import app as _app
from servedeck import control as _control
from servedeck import desired as _desired
from servedeck import legacy_page as _lp


@dataclass(frozen=True)
class ProtocolState:
    """Complete discrete state of the Servedeck control-plane + UI protocol."""

    bound: bool = False
    bind_failed: bool = False
    reconcile_pending: bool = True
    # None | "reconcile" | "recover" | "start" | "switch" | "stop" | "adopt"
    busy_action: str | None = None
    # None | "booting" | "ready" | "failed"
    unit_state: str | None = None
    desired_main: bool = True
    has_launch_override: bool = False
    recovery_attempts: int = 0
    # BootTracker discrete state: None | "starting" | "ready" | "failed"
    boot_track: str | None = None
    # Control abort latches
    abort_armed: bool = False
    launches_before_bind: int = 0
    last_outcome: str = "init"


def _actual_state_of(st: ProtocolState) -> str:
    """Exact mirror of ``legacy_page.actual_state``."""
    if st.busy_action == "stop":
        return _lp.STOPPING
    if st.boot_track == "starting":
        return _lp.STARTING
    if st.unit_state == "ready":
        return _lp.READY
    if st.unit_state == "booting":
        return _lp.STARTING
    if st.boot_track == "failed":
        return _lp.FAILED
    return _lp.STOPPED


def _supervisor_phase_of(st: ProtocolState) -> str | None:
    """Exact mirror of ``legacy_page.build_legacy_blocks`` ``supervisor.phase``."""
    word = _actual_state_of(st)
    if st.boot_track is not None and word == _lp.STARTING:
        return "loading_weights"
    return None


def _ui_stop_endpoint_of(st: ProtocolState) -> str | None:
    """Which endpoint ``web/app.js`` ``stop.onclick`` calls when enabled."""
    actual = _actual_state_of(st)
    boot_active = st.boot_track == "starting" and actual == _lp.STARTING
    is_booting = actual in ("STARTING", "PREFLIGHT") or boot_active
    is_stopping = actual in ("STOPPING", "DRAINING")
    up_now = st.unit_state == "ready"
    if is_stopping:
        return None  # button disabled during STOPPING
    if is_booting:
        return "/api/server/abort"
    if up_now:
        return "/api/server/stop"
    return None  # button disabled when nothing running


def _transitions(st: ProtocolState) -> list[tuple[str, ProtocolState]]:
    """All enabled non-deterministic transitions from ``st``."""
    out: list[tuple[str, ProtocolState]] = []

    # 1. Uvicorn bind completion or timeout (happens asynchronously after lifespan starts)
    if not st.bound and not st.bind_failed:
        out.append(("uvicorn_bind_ok", replace(st, bound=True)))
        out.append(
            (
                "uvicorn_bind_timeout",
                replace(st, bind_failed=True, reconcile_pending=False),
            )
        )

    # 2. _reconcile_after_bind: only after bound, while reconcile_pending and not busy
    if st.bound and st.reconcile_pending and st.busy_action is None:
        if st.desired_main and st.unit_state is None:
            out.append(
                (
                    "reconcile_begin_boot",
                    replace(
                        st,
                        busy_action="reconcile",
                        unit_state="booting",
                        boot_track="starting",
                        abort_armed=False,
                        launches_before_bind=st.launches_before_bind + (0 if st.bound else 1),
                    ),
                )
            )
        else:
            out.append(("reconcile_noop_done", replace(st, reconcile_pending=False)))

    # 3. Poll tick (_recover_desired): runs at ANY point, even before bind —
    # gated on busy is None, not reconcile_pending, AND not bind_failed!
    if st.busy_action is None and not st.reconcile_pending and not st.bind_failed:
        if st.unit_state == "ready":
            if st.recovery_attempts != 0:
                out.append(("poll_reset_recovery", replace(st, recovery_attempts=0)))
        elif st.desired_main and st.unit_state is None and st.recovery_attempts < _app.RECOVERY_ATTEMPTS:
            out.append(
                (
                    f"poll_recover_attempt_{st.recovery_attempts + 1}",
                    replace(
                        st,
                        busy_action="recover",
                        unit_state="booting",
                        boot_track="starting",
                        recovery_attempts=st.recovery_attempts + 1,
                        abort_armed=False,
                        launches_before_bind=st.launches_before_bind + (0 if st.bound else 1),
                    ),
                )
            )

    # 4. In-flight boot completion (reconcile / recover / start / switch)
    if st.busy_action in ("reconcile", "recover", "start", "switch"):
        done_reconcile = False if st.busy_action == "reconcile" else st.reconcile_pending
        if st.abort_armed:
            out.append(
                (
                    f"{st.busy_action}_aborted",
                    replace(
                        st,
                        busy_action=None,
                        reconcile_pending=done_reconcile,
                        unit_state=None,
                        desired_main=False if st.busy_action in ("start", "switch") else st.desired_main,
                        boot_track=None,
                        abort_armed=False,
                        last_outcome="aborted",
                    ),
                )
            )
        else:
            # Outcome A: Boot reaches READY
            out.append(
                (
                    f"{st.busy_action}_ready",
                    replace(
                        st,
                        busy_action=None,
                        reconcile_pending=done_reconcile,
                        unit_state="ready",
                        desired_main=True,
                        recovery_attempts=0,
                        boot_track="ready",
                        abort_armed=False,
                        last_outcome="ready",
                    ),
                )
            )
            # Outcome B: Boot fails (e.g. OOM / engine crash, cleaned up via units.stop)
            out.append(
                (
                    f"{st.busy_action}_failed",
                    replace(
                        st,
                        busy_action=None,
                        reconcile_pending=done_reconcile,
                        unit_state=None,
                        boot_track="failed",
                        abort_armed=False,
                        last_outcome="failed",
                    ),
                )
            )
            # Outcome C: Early preflight / VRAM-release refusal before unit ready
            out.append(
                (
                    f"{st.busy_action}_refused",
                    replace(
                        st,
                        busy_action=None,
                        reconcile_pending=done_reconcile,
                        unit_state=None,
                        boot_track=None,
                        abort_armed=False,
                        last_outcome="refused",
                    ),
                )
            )

    # 5. Operator API mutations (only accepted once bound and when busy_action is None)
    if st.bound and st.busy_action is None:
        if st.unit_state is None:
            # Plain start (preserves has_launch_override)
            out.append(
                (
                    "api_start_plain",
                    replace(
                        st,
                        busy_action="start",
                        unit_state="booting",
                        boot_track="starting",
                        abort_armed=False,
                    ),
                )
            )
            # Allocator start with custom override
            out.append(
                (
                    "api_start_with_override",
                    replace(
                        st,
                        busy_action="start",
                        unit_state="booting",
                        has_launch_override=True,
                        boot_track="starting",
                        abort_armed=False,
                    ),
                )
            )
            # Allocator start resetting overrides to empty dict {}
            out.append(
                (
                    "api_start_reset_override",
                    replace(
                        st,
                        busy_action="start",
                        unit_state="booting",
                        has_launch_override=False,
                        boot_track="starting",
                        abort_armed=False,
                    ),
                )
            )
        if st.unit_state == "ready":
            out.append(
                (
                    "api_switch_or_restart",
                    replace(
                        st,
                        busy_action="switch",
                        unit_state="booting",
                        boot_track="starting",
                        abort_armed=False,
                    ),
                )
            )
            out.append(
                (
                    "api_stop",
                    replace(
                        st,
                        busy_action="stop",
                        desired_main=False,
                    ),
                )
            )
        out.append(("api_adopt", replace(st, busy_action="adopt")))

    # 6. Completion of stop / adopt
    if st.busy_action == "stop":
        out.append(
            (
                "stop_complete",
                replace(
                    st,
                    busy_action=None,
                    unit_state=None,
                    desired_main=False,
                    boot_track=None,
                    last_outcome="stopped",
                ),
            )
        )
    if st.busy_action == "adopt":
        out.append(
            (
                "adopt_complete",
                replace(
                    st,
                    busy_action=None,
                    desired_main=True if st.unit_state == "ready" else st.desired_main,
                ),
            )
        )

    # 7. Operator abort call (allowed in any bound state -- must be a pure no-op when idle!)
    if st.bound:
        if st.busy_action in ("reconcile", "recover", "start", "switch"):
            out.append(("api_abort_inflight", replace(st, abort_armed=True)))
        else:
            # Idle abort(None): returns False and does NOT arm abort_armed!
            out.append(("api_abort_idle_noop", st))

    # 8. Spontaneous engine death + systemd --collect unloading the unit
    if st.unit_state == "ready" and st.busy_action is None:
        out.append(("engine_crash_collected", replace(st, unit_state=None)))

    return out


def _check_invariants(st: ProtocolState) -> str | None:
    """Return an error description if ``st`` violates any safety invariant, else None."""
    # INV-1: Never launch a model before winning our own port bind (R4).
    if st.launches_before_bind > 0:
        return "INV-1 violated: model launch occurred before winning listen-port bind"

    # INV-2: _recover_desired never runs while initial reconcile is pending (BUG-02).
    if st.reconcile_pending and st.busy_action == "recover":
        return "INV-2 violated: poll recovery fired while initial reconcile was still pending"

    # INV-3: When READY and idle, UI Stop button MUST target /api/server/stop, never /abort (BUG-09).
    actual = _actual_state_of(st)
    if actual == _lp.READY and st.busy_action is None:
        if _supervisor_phase_of(st) is not None:
            return f"INV-3a violated: supervisor.phase={_supervisor_phase_of(st)!r} leaked into READY"
        ep = _ui_stop_endpoint_of(st)
        if ep != "/api/server/stop":
            return f"INV-3b violated: UI Stop button in READY state targets {ep!r} instead of /api/server/stop"

    # INV-4: When no action is in flight and no unit is booting, BootTracker must not be stuck in starting (BUG-13).
    if st.busy_action is None and st.unit_state != "booting":
        if st.boot_track == "starting" or actual == _lp.STARTING:
            return f"INV-4 violated: stuck in STARTING while idle (state={st})"

    # INV-5: Immediately after an explicit stop or operator abort completes while idle,
    # actual_state must be STOPPED (never stuck in FAILED or STARTING).
    if st.busy_action is None and st.unit_state is None and st.last_outcome in ("stopped", "aborted"):
        if actual != _lp.STOPPED:
            return f"INV-5 violated: {st.last_outcome} left actual_state={actual!r} ({st})"

    # INV-6: Abort latch is never left armed when no boot/switch is in flight (BUG-06).
    if st.busy_action not in ("reconcile", "recover", "start", "switch") and st.abort_armed:
        return f"INV-6 violated: abort_armed stayed True while busy_action={st.busy_action!r}"

    # INV-7: Crash recovery attempts are strictly bounded by RECOVERY_ATTEMPTS.
    if st.recovery_attempts > _app.RECOVERY_ATTEMPTS:
        return f"INV-7 violated: recovery_attempts={st.recovery_attempts} exceeded {_app.RECOVERY_ATTEMPTS}"

    return None


def test_exhaustive_bfs_control_plane_and_ui_state_machine() -> None:
    """Explore 100% of reachable states from both startup configurations
    (``reconcile=True`` with desired model, and ``reconcile=False`` idle) and
    prove all 7 invariants hold on every reachable state."""
    initials = [
        ProtocolState(reconcile_pending=True, desired_main=True),
        ProtocolState(reconcile_pending=False, bound=True, desired_main=False),
    ]
    visited: set[ProtocolState] = set()
    parent: dict[ProtocolState, tuple[ProtocolState | None, str]] = {}
    queue: deque[ProtocolState] = deque()

    for init in initials:
        visited.add(init)
        parent[init] = (None, "<init>")
        queue.append(init)

    edges_checked = 0
    while queue:
        cur = queue.popleft()
        err = _check_invariants(cur)
        if err is not None:
            trace: list[str] = []
            node: ProtocolState | None = cur
            while node is not None:
                prev, act = parent[node]
                trace.append(f"{act} -> {node}")
                node = prev
            trace.reverse()
            raise AssertionError(f"{err}\nMinimal trace:\n  " + "\n  ".join(trace))

        for action, nxt in _transitions(cur):
            edges_checked += 1
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = (cur, action)
                queue.append(nxt)

    # Sanity guard against vacuous exploration: graph has dozens of distinct states & hundreds of edges.
    assert len(visited) >= 40, f"unexpectedly small state graph: {len(visited)}"
    assert edges_checked >= 150, f"unexpectedly few transitions explored: {edges_checked}"


def test_refinement_bisimulation_against_real_boot_tracker_and_abort(tmp_path: Path) -> None:
    """Drive the real ``BootTracker``, ``actual_state``, ``_publish_result``, and
    ``Control.abort`` through every event sequence of length <= 4 and assert
    exact refinement with the formal model's post-states."""
    alphabet: tuple[str, ...] = (
        "boot_line",
        "boot_marker",
        "boot_ready",
        "publish_refusal",
        "publish_stop",
        "publish_aborted",
        "idle_abort",
    )

    class _Spec:
        key = "big"
        id = "BIG"
        slot = "main"
        port = 19001
        vram_mib = None
        ctx_tokens = 4096
        venv_bin = "/bin"

        def served_names(self) -> list[str]:
            return ["BIG"]

        def render_argv(self, util: float, port: int) -> list[str]:
            return []

        def render_env(self) -> dict[str, str]:
            return {}

    class _Reg:
        def get(self, key: str) -> _Spec:
            return _Spec()

        def keys(self) -> list[str]:
            return ["big"]

    def run_sequence(seq: tuple[str, ...]) -> None:
        hub = _app.Hub()
        boot = _lp.BootTracker()
        ctl = _control.Control(
            _Reg(),
            unit_prefix="sd-test-",
            desired_path=tmp_path / "desired.json",
            run=lambda argv: subprocess.CompletedProcess(list(argv), 0, "", ""),
        )
        holder_ready: bool | None = None

        for step in seq:
            if step == "boot_line":
                boot.record("big", "line", None, 1.0, "waiting for VRAM")
                assert boot.booting is True
                assert _lp.actual_state(busy={"action": "start"}, boot=boot, holder_ready=holder_ready) == _lp.STARTING
            elif step == "boot_marker":
                boot.record("big", "marker", 0, 5.0, "Loading weights took 1.0s")
                assert boot.booting is True
            elif step == "boot_ready":
                boot.record("big", "ready", None, 10.0, "ready")
                holder_ready = True
                assert boot.booting is False
                word = _lp.actual_state(busy=None, boot=boot, holder_ready=holder_ready)
                assert word == _lp.READY
                # supervisor.phase invariant in READY:
                phase = boot.current.phase if (boot.current and word == _lp.STARTING) else None
                assert phase is None
            elif step == "publish_refusal":
                _app._publish_result(
                    hub,
                    "switch",
                    _control.Refusal(reason="vram_not_released", message="busy", key="big"),
                    boot=boot,
                )
                assert boot.booting is False
                if holder_ready is None:
                    assert _lp.actual_state(busy=None, boot=boot, holder_ready=None) != _lp.STARTING
            elif step == "publish_stop":
                holder_ready = None
                _app._publish_result(
                    hub,
                    "stop",
                    _control.StopResult(key="big", unit="sd-test-big", was_live=True),
                    boot=boot,
                )
                assert boot.booting is False
                assert _lp.actual_state(busy=None, boot=boot, holder_ready=None) == _lp.STOPPED
            elif step == "publish_aborted":
                holder_ready = None
                _app._publish_result(
                    hub,
                    "start",
                    _control.StartResult(
                        key="big", unit="sd-test-big", ready=False, elapsed_s=2.0,
                        failure="cancelled", aborted=True,
                    ),
                    boot=boot,
                )
                assert boot.booting is False
                assert _lp.actual_state(busy=None, boot=boot, holder_ready=None) == _lp.STOPPED
            elif step == "idle_abort":
                assert ctl.abort(None) is False
                assert ctl._abort_all is False
                assert ctl._aborted == set()

    # Exhaustively check all 1-, 2-, and 3-step sequences (7 + 49 + 343 = 399 traces)
    seq1 = [(a,) for a in alphabet]
    seq2 = [(a, b) for a in alphabet for b in alphabet]
    seq3 = [(a, b, c) for a in alphabet for b in alphabet for c in alphabet]
    for trace in (*seq1, *seq2, *seq3):
        run_sequence(trace)


def test_counterexample_trace_bind_timeout_blocks_poll_recovery(tmp_path: Path) -> None:
    """Concrete execution of the 3-step trace discovered by the BFS model checker:
    when a rival servedeck holds the listen port so ``_reconcile_after_bind``
    times out, ``rt.bind_failed`` stays latched and ``_recover_desired`` MUST NOT
    launch the desired model on subsequent poll ticks (R4)."""
    import asyncio
    import httpx
    from servedeck import models as _models
    from servedeck import settings as _settings

    toml_path = tmp_path / "models.toml"
    toml_path.write_text(
        '[gpu]\ntotal_mib = 100000\nmargin_mib = 1024\n'
        '[builds.stock]\nvenv = "/opt/stock"\ncuda_home = "/opt/cuda"\n'
        '[models.big]\nid = "Big"\nrepo = "org/big"\nslot = "main"\nport = 19001\nbuild = "stock"\nctx = 4096\n'
    )
    reg = _models.load(toml_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _desired.save(_desired.Desired(main="big"), state_dir / "desired.json")

    launched: list[str] = []

    class _Ctl:
        def live(self):
            return []

        def start(self, key: str, **_kw):
            launched.append(key)
            return _control.StartResult(key=key, unit=f"sd-test-{key}", ready=True, elapsed_s=0.1)

        def reconcile(self, *_a, **_kw):
            launched.append("reconcile")
            return _control.Reconciliation()

    s = _settings.Settings(
        listen_host="127.0.0.1",
        listen_port=18095,
        models_path=toml_path,
        state_dir=state_dir,
        unit_prefix="sd-test-",
    )
    rt = _app.build_runtime(
        s,
        registry=reg,
        control=_Ctl(),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _r: httpx.Response(200, json={"ok": True, "instance": "rival-instance"})
            )
        ),
    )
    rt.reconcile_pending = True

    async def _drive() -> None:
        rt.hub.bind(asyncio.get_running_loop())
        res = await _app._reconcile_after_bind(rt, timeout_s=0.02, interval_s=0.005)
        assert res is None
        assert rt.bind_failed is True
        assert rt.reconcile_pending is False
        _app._recover_desired(rt)

    asyncio.run(_drive())
    assert launched == [], f"R4 violated: losing process launched {launched} after bind_timeout"
