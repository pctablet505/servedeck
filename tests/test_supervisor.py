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


def _fake_supervisor(tmp: Path, monkeypatch) -> tuple[supervisor.Supervisor, list]:
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
    #
    # monkeypatch, not assignment: these used to be rebound on the module
    # permanently, so every test that ran AFTER one of these in the same
    # session had preflight silently disabled -- including the tests whose
    # whole subject is a preflight check refusing a start.
    monkeypatch.setattr(preflight, "run_preflight", lambda **kw: [])
    monkeypatch.setattr(preflight, "blocking_failures", lambda checks: [])
    s._sync_shell_config = types.MethodType(lambda self, **kw: None, s)          # type: ignore[assignment]
    s._run_monitor = types.MethodType(lambda self, *a, **k: asyncio.sleep(0), s)  # type: ignore[assignment]
    return s, launches


def test_restart_actually_relaunches(tmp_path: Path, monkeypatch) -> None:
    """Regression: restart() used to be a silent no-op stop.

    stop() leaves actual_state == "STOPPING"; start()'s idempotence guard
    returns early on that state, so the relaunch never happened — 0 launches,
    ending at actual=STOPPING / desired=STOPPED.
    """
    s, launches = _fake_supervisor(tmp_path, monkeypatch)

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


def test_adopted_server_reports_reached_ready(monkeypatch) -> None:
    """Regression: an adopted server's crash was filed as a failed boot.

    _adopt_ready() left _tracker None, so reached_ready computed False and
    _on_exit took the failed-boot branch — disabling auto-restart for every
    server that was already running when Servedeck started.
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


def test_failed_boot_is_never_auto_restarted(monkeypatch) -> None:
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


# --------------------------------------------------------------------------
# Monitor / adoption regressions
# --------------------------------------------------------------------------
def test_adopted_server_monitor_notices_the_process_exiting(tmp_path: Path, monkeypatch) -> None:
    """Regression: _run_monitor died on its first tick for an ADOPTED server.

    _adopt_ready() calls _run_monitor(already_ready=True), which leaves
    `client` None — but the poll body is guarded on `self._tracker is not
    None`, which _adopt_ready has just made true. `assert client is not None`
    therefore fired on iteration 1, the exception was swallowed by the
    fire-and-forget task, and nothing ever checked liveness again.

    Observed consequence on a live instance: /api/state reported actual_state
    READY with nothing listening on the port at all.
    """
    s, _ = _fake_supervisor(tmp_path, monkeypatch)
    # _fake_supervisor stubs _run_monitor out; this test is about the real one.
    del s._run_monitor
    s.desired.backend = "flashnext"
    s.desired.port = 8001
    s._tracker = phases.PhaseTracker()
    s._tracker.mark_adopted_ready()
    s.actual_state = "READY"

    handle = procctl.ServerHandle(
        pid=424242, pgid=424242, argv=[], cwd="/tmp", log_path="", started_at=0.0,
    )
    alive = [True]
    s._try_reap = types.MethodType(lambda self, pid: None, s)          # type: ignore[assignment]
    s._pid_alive = types.MethodType(lambda self, pid: alive[0], s)     # type: ignore[assignment]
    exits: list = []

    async def fake_on_exit(self, handle, *, reaped_status):  # noqa: ANN001
        exits.append(reaped_status)

    s._on_exit = types.MethodType(fake_on_exit, s)                     # type: ignore[assignment]

    async def scenario() -> None:
        task = asyncio.create_task(
            s._run_monitor(handle, log_paths=[], port=8001, already_ready=True)
        )
        for _ in range(200):                 # let a few poll ticks happen
            await asyncio.sleep(0)
        alive[0] = False                     # the server dies
        await asyncio.wait_for(task, timeout=5.0)

    asyncio.run(scenario())

    assert exits == [None], (
        "the monitor of an adopted server must survive its poll loop and "
        "report the exit; instead it crashed on tick 1 and the UI sat on a "
        "stale READY forever"
    )
