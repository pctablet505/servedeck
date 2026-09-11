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


# --------------------------------------------------------------------------
# Phase-1 sweep regressions (2026-09-10, driven against the live 27B)
# --------------------------------------------------------------------------
# F1 (Flash-Next's EXTRA_ARGS reaching the 27B) is covered end to end, with
# the real config sync writing a real file, in tests/test_extra_args_ownership.py.


def test_restart_waits_for_the_old_engine_to_exit(tmp_path, monkeypatch) -> None:
    """F4(a) GPU SAFETY: restart-during-boot ran two vLLM engines at once.

    ``stop()`` deliberately does not await the SIGTERM; ``restart()`` then
    slept 0.2 s and started the next engine. A vLLM parent mid-boot takes tens
    of seconds to die, and PORT_IN_USE cannot catch it because vLLM binds its
    port only after loading weights. Observed 2026-09-10 23:16:48: pids
    3780019 and 3783809 both loading weights on the same card for ~60 s.
    """
    import time as _time

    s, launches = _fake_supervisor(tmp_path, monkeypatch)

    def slow_stop(h, **kw):  # noqa: ANN001 - the real one waits for the pgid to die
        _time.sleep(0.5)
        return procctl.StopResult(True, "sigterm", 0.5, "exited after SIGTERM")

    s._stop_fn = slow_stop  # type: ignore[assignment]

    async def scenario() -> int:
        await s.start(**_CFG)
        s.actual_state = "READY"
        task = asyncio.create_task(s.restart(mode="immediate"))
        await asyncio.sleep(0.3)      # the old process is still dying here
        mid = len(launches)
        await task
        return mid

    mid = asyncio.run(scenario())
    assert len(launches) == 2, "restart must still relaunch"
    assert mid == 1, (
        f"a second engine was launched while the first was still being "
        f"stopped ({mid} launches 0.3 s into a 0.5 s stop)"
    )


def test_a_stale_handles_exit_does_not_orphan_the_running_engine(tmp_path, monkeypatch) -> None:
    """F4(b) GPU SAFETY: _on_exit() cleared _handle unconditionally.

    The previous boot's monitor fires AFTER start() has installed the new
    handle, so the live process was orphaned: /api/state reported FAILED (and
    later STOPPED) while a 44 GiB engine went on serving on :8004, and Stop
    was inert because stop() took its ``self._handle is None`` branch.
    """
    s, _ = _fake_supervisor(tmp_path, monkeypatch)

    async def scenario():
        await s.start(**_CFG)
        old = s._handle
        s.actual_state = "STOPPED"        # pretend the first boot settled
        await s.start(**_CFG)
        new = s._handle
        assert old is not None and new is not None and old is not new
        await s._on_exit(old, reaped_status=0)   # the OLD process finally dies
        return new

    new = asyncio.run(scenario())

    assert s._handle is new, (
        "the exit of a superseded handle cleared the handle of the engine "
        "that is actually running — Stop then signals nothing"
    )
    assert s.actual_state == "STARTING", (
        f"actual_state was overwritten to {s.actual_state!r} by an exit that "
        "belongs to a previous run"
    )


def test_stop_does_not_claim_stopped_while_a_listener_holds_the_port(tmp_path, monkeypatch) -> None:
    """F4(b), second half: Stop reported the GPU released while it was not.

    With no handle installed, stop() set actual_state=STOPPED and signalled
    nothing — while pid 3783809 kept listening on :8004 and holding 44,472
    MiB, and the SAME /api/state payload reported upstream.up=true.
    """
    s, _ = _fake_supervisor(tmp_path, monkeypatch)
    s.desired.port = 8004
    monkeypatch.setattr(procctl, "listener_pid", lambda port: 3783809 if port == 8004 else None)

    asyncio.run(s.stop())

    assert s.actual_state != "STOPPED", (
        "stop() reported STOPPED with a listener still on the port it was "
        "asked to free"
    )
    assert "3783809" in (s.last_error or ""), (
        f"the operator is not told what is still holding the port: {s.last_error!r}"
    )


def test_a_refused_start_never_persists_the_port_it_was_refused(tmp_path, monkeypatch) -> None:
    """F5 SEVERE: a start refused by preflight still published its intent.

    start() wrote desired.json (port + desired_state RUNNING) BEFORE running
    preflight. updetect ranks desired.json's port, found ats-optimizer's
    listener on the refused :8000 and followed it: /api/state reported
    upstream http://localhost:8000, backend "inline", pid 2922 and
    server_uptime_s 105114 — 29 h of an unrelated service displayed as the
    model server — and app.py's catch_all proxied every /v1 request there.
    """
    s, launches = _fake_supervisor(tmp_path, monkeypatch)
    blocked = preflight.PreflightCheck(
        "PORT_IN_USE", False, "block", "port already in use",
        "pid 2922 is already listening on :8000.", None,
    )
    monkeypatch.setattr(preflight, "run_preflight", lambda **kw: [blocked])
    monkeypatch.setattr(preflight, "blocking_failures", lambda checks: [blocked])

    asyncio.run(s.start(**{**_CFG, "port": 8000}))

    assert not launches
    assert s.actual_state == "FAILED"
    assert s.desired.port != 8000, (
        "the refused port was written into the supervisor's desired state, "
        "and updetect follows it straight to the unrelated listener"
    )
    on_disk = supervisor.load_desired(tmp_path)
    assert on_disk.port != 8000, f"state/desired.json persisted the refused port: {on_disk.port}"
    assert on_disk.desired_state != "RUNNING", (
        "a start that never launched must not leave intent RUNNING"
    )


def test_an_out_of_range_util_is_refused_before_anything_is_written(tmp_path, monkeypatch) -> None:
    """F6 SEVERE: util 5.0 was accepted, half-written, and wedged PREFLIGHT.

    _sync_shell_config() wrote BACKEND/MODEL_REPO/SERVED_NAME/PORT/
    MAX_MODEL_LEN/MAX_NUM_SEQS and only THEN did shellconfig.set_util() raise
    ValueError — which start() does not catch — so .config was left describing
    a configuration nobody asked for and actual_state stayed "PREFLIGHT"
    forever. PREFLIGHT is in the UI's BUSY_PHASES, so Apply and Stop were both
    disabled: the page had no control left.
    """
    s, launches = _fake_supervisor(tmp_path, monkeypatch)
    writes: list = []
    s._sync_shell_config = types.MethodType(lambda self, **kw: writes.append(kw), s)  # type: ignore[assignment]

    asyncio.run(s.start(**{**_CFG, "util": 5.0}))

    assert writes == [], "an invalid configuration was written to .config anyway"
    assert not launches
    assert s.actual_state == "FAILED", (
        f"actual_state left at {s.actual_state!r} — anything but a settled "
        "state disables every control on the page"
    )
    assert "5.0" in (s.last_error or ""), (
        f"the operator is not told which value was rejected: {s.last_error!r}"
    )
    assert s.desired.util != 5.0, "the rejected value was persisted as intent"


def test_a_config_write_that_raises_does_not_wedge_the_supervisor(tmp_path, monkeypatch) -> None:
    """F6, second half: the ValueError escaped start() entirely.

    start() catches only ShellConfigError/ServerRunningError, so any other
    failure inside the config write left actual_state at "PREFLIGHT" and the
    next VALID start returned 202 and did nothing (the idempotence guard
    returns early on PREFLIGHT).
    """
    s, launches = _fake_supervisor(tmp_path, monkeypatch)

    def boom(self, **kw):  # noqa: ANN001
        raise ValueError("GPU_MEM_UTIL must be a number in (0, 1], got 5.0")

    s._sync_shell_config = types.MethodType(boom, s)  # type: ignore[assignment]
    asyncio.run(s.start(**_CFG))

    assert s.actual_state == "FAILED", f"wedged at {s.actual_state!r}"
    assert "GPU_MEM_UTIL" in (s.last_error or ""), s.last_error

    # ... and the supervisor still accepts the next, valid start.
    s._sync_shell_config = types.MethodType(lambda self, **kw: None, s)  # type: ignore[assignment]
    asyncio.run(s.start(**_CFG))
    assert len(launches) == 1, "the supervisor was wedged: a valid start did nothing"


# --------------------------------------------------------------------------
# reached_ready is a fact about the run, not a liveness reading (F11d, and
# the stale boot panel that the F2 fix made visible)
# --------------------------------------------------------------------------
def test_reached_ready_survives_the_probe_failing_during_shutdown() -> None:
    """The monitor keeps probing /v1/models after READY, and the probe fails
    as soon as the server starts shutting down. reached_ready used to be
    recomputed from the latest probe, so it went back to False on every stop."""
    t = phases.PhaseTracker()
    t.feed("INFO:     Application startup complete.")
    assert t.set_http_probe_ok(True) is not None, "READY must be announced once"
    assert t.reached_ready is True

    assert t.set_http_probe_ok(False) is None      # shutting down
    assert t.reached_ready is True, "a failing probe un-latched READY"
    assert t.phase is phases.Phase.READY
    assert t.set_http_probe_ok(True) is None, "READY must not be announced twice"


def test_a_boot_that_never_served_is_still_not_ready() -> None:
    """Over-correction guard: the latch must not make READY easier to reach.
    Each criterion alone, in either order, is not READY."""
    probe_only = phases.PhaseTracker()
    assert probe_only.set_http_probe_ok(True) is None
    assert probe_only.reached_ready is False

    log_only = phases.PhaseTracker()
    log_only.feed("INFO:     Application startup complete.")
    assert log_only.set_http_probe_ok(False) is None
    assert log_only.reached_ready is False

    # Criteria that were both true only at different moments never latch.
    flaky = phases.PhaseTracker()
    flaky.set_http_probe_ok(True)
    flaky.set_http_probe_ok(False)
    flaky.feed("INFO:     Application startup complete.")
    assert flaky.reached_ready is False


def test_a_boot_that_served_and_was_stopped_is_recorded_as_having_reached_ready(
    tmp_path: Path, monkeypatch
) -> None:
    """The real monitor, end to end. The boot reaches READY, the operator
    stops it, the /v1/models probe fails while it shuts down, and only then
    does the process exit. Every history record written on 2026-09-11 said
    reached_ready: false, including runs that had served for minutes, so
    history.py never had a boot to learn an ETA from (F10/F11d)."""
    import time as _time

    from servedeck import paths

    monkeypatch.setattr(paths, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(supervisor, "MONITOR_POLL_S", 0.01)
    monkeypatch.setattr(preflight, "run_preflight", lambda **kw: [])
    monkeypatch.setattr(preflight, "blocking_failures", lambda checks: [])

    boot_logs: list[str] = []

    def launch(argv, env, cwd, log_path):  # noqa: ANN001
        boot_logs.append(log_path)          # what the monitor tails
        return procctl.ServerHandle(pid=4242, pgid=4242, argv=list(argv), cwd="/tmp",
                                    log_path=log_path, started_at=_time.time())

    hist = tmp_path / "history.jsonl"
    s = supervisor.Supervisor(
        state_dir=tmp_path / "state", clock=_time.time, launch_fn=launch,
        stop_fn=lambda h, **kw: procctl.StopResult(True, "sigterm", 0.1, "exited after SIGTERM"),
        death_exec_fn=lambda env: None, history_path=hist,
    )
    s._sync_shell_config = types.MethodType(lambda self, **kw: None, s)  # type: ignore[assignment]
    alive = [True]
    answering = [False]
    s._try_reap = types.MethodType(lambda self, pid: None, s)          # type: ignore[assignment]
    s._pid_alive = types.MethodType(lambda self, pid: alive[0], s)     # type: ignore[assignment]

    async def probe(self, client, port):  # noqa: ANN001
        return answering[0]

    s._probe_ready = types.MethodType(probe, s)                         # type: ignore[assignment]

    async def until(cond, what):  # noqa: ANN001
        for _ in range(500):
            if cond():
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"timed out waiting for {what}")

    async def scenario() -> None:
        await s.start(**_CFG)
        with open(boot_logs[0], "a") as fh:
            fh.write("INFO:     Application startup complete.\n")
        answering[0] = True
        await until(lambda: s.actual_state == "READY", "READY")

        await s.stop()
        answering[0] = False               # the server stops answering first ...
        await asyncio.sleep(0.1)           # ... for several monitor ticks ...
        alive[0] = False                   # ... and only then does the process exit
        await until(lambda: s.actual_state == "STOPPED" and s._handle is None, "the exit")

    asyncio.run(scenario())

    import json

    records = [json.loads(line) for line in hist.read_text().splitlines() if line.strip()]
    assert len(records) == 1, records
    rec = records[0]
    assert rec["outcome"] == "stopped_by_user", rec
    assert rec["reached_ready"] is True, (
        "a boot that served and was then stopped is recorded as never having "
        f"reached READY: {rec}"
    )
    assert rec["total_s"] is not None, "a READY boot must record its boot time"
    snap = s.snapshot()
    assert snap["reached_ready"] is True and snap["phase"] == "ready", snap
