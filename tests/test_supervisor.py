"""Supervisor regression tests.

The code review found four supervisor bugs and noted there were no supervisor
tests at all — which is why they survived. These cover the two most severe,
both of which the reviewer reproduced by execution.

Uses asyncio.run() rather than pytest-asyncio: one async test does not justify
a plugin dependency.
"""

from __future__ import annotations

import asyncio
import types
from pathlib import Path

from servedeck import phases, preflight, procctl, supervisor

_CFG = dict(
    repo_id="RadixArk/Qwen3.8-Flash-Next-NVFP4",
    backend="flashnext",
    served_name="qwen38-flash-next",
    port=8001,
    util=0.95,
    max_model_len=262144,
    max_num_seqs=1,
)


def _fake_supervisor(tmp: Path) -> tuple[supervisor.Supervisor, list]:
    """A Supervisor with launch/stop/preflight/config/monitor all faked out."""
    launches: list = []

    def fake_launch(argv, env, cwd, log_path):
        launches.append(argv)
        return procctl.ServerHandle(
            pid=1000 + len(launches), pgid=1000 + len(launches),
            argv=list(argv), cwd="/tmp", log_path="/tmp/fake.log", started_at=0.0,
        )

    s = supervisor.Supervisor(
        state_dir=tmp,
        clock=lambda: 1000.0,
        launch_fn=fake_launch,
        stop_fn=lambda h, **kw: procctl.StopResult(True, "term", 0.0, "faked"),
        history_path=tmp / "history.jsonl",
    )
    # preflight touches the GPU and real files; the monitor polls HTTP and
    # tails logs. Neither belongs in a unit test of the state machine.
    preflight.run_preflight = lambda **kw: []           # type: ignore[assignment]
    preflight.blocking_failures = lambda checks: []     # type: ignore[assignment]
    s._sync_shell_config = types.MethodType(lambda self, **kw: None, s)          # type: ignore[assignment]
    s._run_monitor = types.MethodType(lambda self, *a, **k: asyncio.sleep(0), s)  # type: ignore[assignment]
    return s, launches


def test_restart_actually_relaunches(tmp_path: Path) -> None:
    """Regression: restart() used to be a silent no-op stop.

    stop() leaves actual_state == "STOPPING"; start()'s idempotence guard
    returns early on that state, so the relaunch never happened — 0 launches,
    ending at actual=STOPPING / desired=STOPPED.
    """
    s, launches = _fake_supervisor(tmp_path)

    async def scenario() -> None:
        await s.start(**_CFG)
        assert len(launches) == 1, "initial start should launch once"
        s.actual_state = "READY"          # pretend the boot completed
        await s.restart(mode="immediate")

    asyncio.run(scenario())

    assert len(launches) == 2, (
        f"restart() must relaunch; got {len(launches)} launch(es) — the "
        "regression where restart silently degraded into a plain stop"
    )
    assert s.desired.desired_state == "RUNNING", (
        "restart() must leave intent RUNNING, or auto-restart treats the "
        "relaunched server as unwanted and never revives it"
    )
    assert s.actual_state != "STOPPING"


def test_adopted_server_reports_reached_ready() -> None:
    """Regression: an adopted server's crash was filed as a failed boot.

    _adopt_ready() left _tracker None, so reached_ready computed False and
    _on_exit took the failed-boot branch — disabling auto-restart for every
    server that was already running when Coldstart started.
    """
    t = phases.PhaseTracker()
    assert t.reached_ready is False

    t.mark_adopted_ready()
    assert t.reached_ready is True

    for line in (
        "torch.AcceleratorError: CUDA error: misaligned address",
        "CUDA out of memory",
    ):
        f = phases.classify(line, reached_ready=t.reached_ready)
        assert f is not None and f.auto_restart is True, (
            f"{line!r} must be auto-restartable for an adopted, serving server"
        )


def test_failed_boot_is_never_auto_restarted() -> None:
    """The other half of the rule: a boot that never served must not loop.

    Six consecutive boots failed on 2026-08-27, each with a different root
    cause. Blind restarting would have hidden every one.
    """
    t = phases.PhaseTracker()  # never reached READY
    f = phases.classify(
        "torch.AcceleratorError: CUDA error: misaligned address",
        reached_ready=t.reached_ready,
    )
    assert f is not None and f.auto_restart is False
