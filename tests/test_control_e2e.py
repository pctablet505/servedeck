"""End to end against the REAL systemd user manager, with a stub standing in
for vLLM.

§2.7: "the live boot/switch/adopt paths, the ones that actually fail, have no
end-to-end test". This is that test. Everything here runs against the real
thing: real `systemd-run --user`, real journald, a real HTTP probe — served by
`stub_openai.py`, never a real model, so nothing here needs GPU room. (The one
test that used to boot a real LFM2.5-350M here now lives in
tests/test_e2e_real.py, which is where a GPU-needing test belongs — see the
2026-09-16 note near STUB_MODEL below.)

Blast radius. Every unit created here is named `sd-test-*`; `units.py` refuses
every other shape before spawning anything, `Control` is constructed with
`unit_prefix="sd-test-"` so even its discovery glob cannot see a `model-*`
unit, and the module-scoped fixture below stops every `sd-test-*` unit on the
way out whether the tests passed, failed or raised. No existing unit is
touched, nothing is enabled or disabled, no process this file did not start is
ever signalled, and the only port bound is 8030 on loopback.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pytest

from servedeck import control, gpu, units
from servedeck.control import Control, Refusal

pytestmark = pytest.mark.skipif(
    shutil.which("systemd-run") is None or not os.environ.get("XDG_RUNTIME_DIR"),
    reason="needs a systemd user manager (systemd-run + XDG_RUNTIME_DIR)",
)

STUB = Path(__file__).resolve().parent / "stub_openai.py"
STUB_UNIT = "sd-test-stub"
STUB_PORT = 8030
STUB_MODEL = "sd-test-stub-model"

# The real-LFM2, needs-a-free-GPU test that used to live here (LFM2_UNIT,
# LFM2_PORT, LFM2_VENV_BIN, MIN_FREE_MIB, _Lfm2Spec, _skip_unless_gpu_has_room)
# moved to tests/test_e2e_real.py on 2026-09-16 as
# test_real_lfm2_boots_serves_and_gives_the_memory_back: the always-run unit
# suite (pytest --ignore=tests/test_e2e_real.py) must not depend on GPU room
# being free, and this file has no such gate of its own.


# --------------------------------------------------------------------------
# Cleanup — runs whatever happens
# --------------------------------------------------------------------------


def _stop_every_sd_test_unit() -> list[str]:
    """Stop every `sd-test-*` unit, by name, one at a time.

    By name and not by pattern: `systemctl stop 'sd-test-*'` would be a glob
    handed to a privileged tool, and a pkill would be the self-match trap this
    whole design exists to delete. Discovery is the same `list-units` call
    production uses, so this can only ever name units it could also have
    created.
    """
    stopped: list[str] = []
    for unit in units.list_units("sd-test-*"):
        try:
            units.stop(unit, timeout_s=60)
        except units.UnitError:  # pragma: no cover - best effort teardown
            pass
        stopped.append(unit)
    return stopped


@pytest.fixture(scope="module", autouse=True)
def _no_sd_test_units_left_behind():
    _stop_every_sd_test_unit()
    try:
        yield
    finally:
        _stop_every_sd_test_unit()
        leftover = units.list_units("sd-test-*")
        assert leftover == [], f"the suite leaked transient units: {leftover}"


def wait_until(predicate, timeout_s: float, interval_s: float = 0.25):
    """Poll `predicate` until it returns something truthy, or time out."""
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval_s)
    return last


def parent_pid(pid: int) -> int:
    """The ppid from /proc/<pid>/stat.

    Split on the LAST ')': the comm field is parenthesised and may itself
    contain spaces and parentheses, so field-splitting the whole line gets the
    wrong column for a process named e.g. `(sd-pam)`.
    """
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[1])  # [0] is state, [1] is ppid


def ancestors(pid: int) -> list[int]:
    chain = []
    while pid > 1:
        pid = parent_pid(pid)
        chain.append(pid)
    return chain


def my_cgroup() -> str:
    return Path("/proc/self/cgroup").read_text().strip().split("::", 1)[-1]


def http_json(url: str, payload: dict | None = None, timeout_s: float = 30.0) -> dict | None:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            return json.loads(response.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return None


# --------------------------------------------------------------------------
# 1. The transient-unit lifecycle, with a stub instead of a model
# --------------------------------------------------------------------------


@pytest.fixture
def stub_unit():
    """`sd-test-stub` on :8030, started through units.start_transient itself."""
    units.start_transient(
        STUB_UNIT,
        [sys.executable, str(STUB), "--port", str(STUB_PORT), "--model", STUB_MODEL],
        {"PYTHONUNBUFFERED": "1"},
        cwd=str(STUB.parent),
        restart="no",
        description="servedeck P3 e2e stub",
    )
    try:
        yield STUB_UNIT
    finally:
        units.stop(STUB_UNIT, timeout_s=30)


def test_start_transient_really_starts_a_unit_and_stop_really_removes_it(stub_unit) -> None:
    assert wait_until(lambda: units.show(stub_unit).active_state == "active", 20), (
        units.journal_tail(stub_unit, 20)
    )
    state = units.show(stub_unit)
    assert state.main_pid > 0
    assert state.sub_state == "running" and state.result == "success"
    assert not units.gone(stub_unit)

    ids = wait_until(lambda: control.http_probe(STUB_PORT), 20)
    assert ids == [STUB_MODEL]

    units.stop(stub_unit, timeout_s=30)
    # `--collect` unloads the unit as soon as it is inactive: gone, not merely
    # inactive. That is what makes the name reusable for the next start
    # without a `reset-failed` dance.
    assert units.gone(stub_unit) is True
    assert stub_unit not in units.list_units("sd-test-*")
    assert control.http_probe(STUB_PORT) is None


def test_the_unit_is_not_in_the_cgroup_of_whatever_started_it(stub_unit) -> None:
    """R4, stated as a fact instead of a hope — the KillMode fix by construction.

    v1 launched vLLM as a child of the dashboard, inside the dashboard's
    cgroup, with `KillMode=control-group`: stopping servedeck killed the model,
    and a model started from a terminal died with the terminal. Nothing about
    that is fixable by choosing a better KillMode, because the child really was
    in that cgroup.

    systemd-run --user puts the payload in its OWN cgroup under the user
    manager's app.slice. This test is the proof: the unit's ControlGroup is
    not ours, does not contain ours, and is not contained by ours.
    """
    wait_until(lambda: units.show(stub_unit).main_pid > 0, 20)
    unit_cgroup = units.control_group(stub_unit)
    mine = my_cgroup()

    assert unit_cgroup.endswith(f"/{stub_unit}.service"), unit_cgroup
    assert "/app.slice/" in unit_cgroup
    assert unit_cgroup != mine
    assert not unit_cgroup.startswith(mine.rstrip("/") + "/")
    assert not mine.startswith(unit_cgroup.rstrip("/") + "/")


def test_the_units_main_pid_is_a_child_of_systemd_not_of_the_test(stub_unit) -> None:
    """The same fact from the process tree's side: the model is forked by the
    user manager, not by us, so it is in no shell's job control and dies with
    nothing we can close.

    Note what is NOT asserted: that the manager is absent from the test's own
    ancestry. Every process in a user session descends from `systemd --user`,
    so a common ancestor proves nothing either way — which is exactly why the
    cgroup test above, not this one, is the load-bearing proof. What this adds
    is the other direction: the unit is not OUR descendant.
    """
    pid = wait_until(lambda: units.show(stub_unit).main_pid or None, 20)
    assert pid
    ppid = parent_pid(pid)
    comm = Path(f"/proc/{ppid}/comm").read_text().strip()
    assert ppid == 1 or comm == "systemd", f"MainPID {pid} has parent {ppid} ({comm})"
    assert ppid != os.getpid()
    assert os.getpid() not in ancestors(pid), "the unit must not be a child of the test"
    # One hop to the manager, where the test sits many hops below it.
    assert len(ancestors(pid)) < len(ancestors(os.getpid()))


def test_the_start_limit_properties_are_really_set_on_the_unit(stub_unit) -> None:
    """Note the asymmetry: the property is WRITTEN as `StartLimitIntervalSec`
    and READ back as `StartLimitIntervalUSec`. Asserting on the name we wrote
    would silently read a default and pass, which is why this reads the name
    systemd actually reports."""
    props = units.properties(
        stub_unit, ("StartLimitIntervalUSec", "StartLimitBurst", "RestartUSec")
    )
    assert props["StartLimitIntervalUSec"] == "5min"
    assert props["StartLimitBurst"] == "3"


def test_a_unit_that_cannot_boot_stops_restarting_instead_of_looping_forever() -> None:
    """The crash loop actually terminates — the whole point of item 1.

    With systemd's defaults (5 starts per 10 s) and RestartSec=10 every retry
    lands outside the window, the counter resets, and a model that cannot boot
    restarts every ten seconds for as long as the machine is up. Here the
    command fails immediately and restart_sec is compressed to 1 s so the
    ceiling is reached in about three seconds instead of thirty; the property
    under test — that there IS a ceiling — is the same.
    """
    unit = "sd-test-doomed"
    try:
        units.start_transient(
            unit,
            ["/bin/false"],
            {},
            cwd="/tmp",
            restart="on-failure",
            restart_sec=1,
            start_limit_interval_sec=300,
            start_limit_burst=3,
        )
        # It gives up and (with --collect) is removed. If the limit could not
        # fire, this unit would still be here, restarting, when the timeout
        # expires.
        assert wait_until(lambda: units.gone(unit, attempts=2, delay_s=0.2), 30), (
            f"still alive after 30s: {units.properties(unit, ('ActiveState', 'NRestarts'))}"
        )
    finally:
        units.stop(unit, timeout_s=30)


def test_a_transient_unit_does_not_inherit_the_callers_environment() -> None:
    """Measured, and the reason ModelSpec.render_env() must be COMPLETE.

    A transient unit's environment is the USER MANAGER's, not the calling
    shell's. An operator who exported a variable before running `servedeck
    start` does not pass it to the model — which is the honest version of R2's
    `.config` problem: there is now exactly one place a model's environment
    comes from, and it is the registry.
    """
    os.environ["SD_TEST_SHELL_ONLY"] = "leaked"
    unit = "sd-test-env"
    try:
        units.start_transient(
            unit,
            ["/usr/bin/env"],
            {"SD_TEST_FROM_REGISTRY": "present"},
            cwd="/tmp",
            restart="no",
        )
        lines = wait_until(
            lambda: [ln for ln in units.journal_tail(unit, 200) if ln.startswith("SD_TEST_")],
            20,
        )
        assert "SD_TEST_FROM_REGISTRY=present" in (lines or [])
        assert not any(ln.startswith("SD_TEST_SHELL_ONLY") for ln in lines or [])
    finally:
        os.environ.pop("SD_TEST_SHELL_ONLY", None)
        units.stop(unit, timeout_s=30)


@pytest.fixture
def manager_canary():
    """Put a harmless secret-shaped variable into the REAL user manager.

    `systemctl --user set-environment` is how a login session's exports get
    into the manager in the first place, so this reproduces the actual leak
    path rather than simulating it. Removed again in the finaliser whatever
    happens; the name is unique to this suite and matches nothing else.
    """
    name, value = "SD_TEST_SECRET", "canary-value-123"
    subprocess.run(["systemctl", "--user", "set-environment", f"{name}={value}"], check=True)
    try:
        assert name in units.manager_environment_names()
        yield name, value
    finally:
        subprocess.run(["systemctl", "--user", "unset-environment", name], check=False)
        assert name not in units.manager_environment_names()


def _env_lines_of(unit: str) -> list[str]:
    """What /usr/bin/env printed to the journal for this unit."""
    return wait_until(
        lambda: [ln for ln in units.journal_tail(unit, 300) if ln.startswith("SD_TEST_")] or None,
        20,
    ) or []


def test_without_the_property_the_managers_secrets_really_do_leak(manager_canary) -> None:
    """The control that gives the next two tests their power.

    If this ever stops leaking, `UnsetEnvironment=` has stopped being the thing
    that makes the difference and the tests below are passing for free.
    """
    name, value = manager_canary
    unit = "sd-test-leak"
    try:
        units.start_transient(unit, ["/usr/bin/env"], {}, cwd="/tmp", restart="no")
        lines = _env_lines_of(unit)
        assert f"{name}={value}" in lines, lines
    finally:
        units.stop(unit, timeout_s=30)


def test_unset_env_keeps_the_managers_secrets_out_of_the_unit(manager_canary) -> None:
    """The fix, against real systemd: the property is honoured, the variable is
    absent from what the process itself can read, and nothing innocent went
    with it."""
    name, value = manager_canary
    unit = "sd-test-unset"
    try:
        units.start_transient(unit, ["/usr/bin/env"], {"SD_TEST_KEPT": "yes"},
                              cwd="/tmp", restart="no", unset_env=[name])
        assert units.properties(unit, ("UnsetEnvironment",))["UnsetEnvironment"] == name
        lines = _env_lines_of(unit)
        assert "SD_TEST_KEPT=yes" in lines, lines
        assert not any(ln.startswith(f"{name}=") for ln in lines), lines
        assert value not in "\n".join(units.journal_tail(unit, 300))
    finally:
        units.stop(unit, timeout_s=30)


def test_a_model_started_through_control_cannot_read_the_managers_secrets(manager_canary) -> None:
    """End to end through `control.start`, checking the one place that cannot
    be argued with: the live process's own `/proc/<pid>/environ`.

    The stub is a stand-in for vLLM here, but the path is identical — the
    redaction list is computed from the real `systemctl --user
    show-environment` and reaches the real unit.
    """
    name, value = manager_canary
    spec = _StubSpec()
    ctl = Control(
        _OneModelRegistry(spec),
        unit_prefix="sd-test-",
        desired_path=Path(os.environ["PYTEST_TMP"]) / "desired.json",
        total_mib=97887,
        free_mib=lambda: 97887,
    )
    assert name in ctl.secret_env_names()

    result = ctl.start("stub", timeout_s=90)
    try:
        assert not isinstance(result, Refusal), result
        assert result.ready, result.failure
        pid = units.show(STUB_UNIT).main_pid
        assert pid > 0
        environ = Path(f"/proc/{pid}/environ").read_bytes().decode("utf-8", "replace")
        entries = [e for e in environ.split("\0") if e]
        assert not any(e.startswith(f"{name}=") for e in entries), "the canary reached the model"
        assert value not in environ
        # The unit is genuinely populated, so the absence above means something.
        assert any(e.startswith("HOME=") for e in entries)
        assert "PYTHONUNBUFFERED=1" in entries
    finally:
        ctl.stop("stub")


def test_wait_ready_sees_real_journal_markers_in_order() -> None:
    """The marker detector, against real journald rather than a pipe.

    The stub prints vLLM's four lines with the real delays a boot has, so this
    exercises the actual failure shape: long silences between markers, the
    port open but answering 503, and readiness arriving last.
    """
    unit = STUB_UNIT
    spec = _StubSpec()
    ctl = Control(
        _OneModelRegistry(spec),
        unit_prefix="sd-test-",
        desired_path=Path(os.environ["PYTEST_TMP"]) / "desired.json",
        # The GPU is the ONE thing faked here, because the stub holds no VRAM:
        # the card's real state has nothing to do with what this test proves,
        # and leaving it real would skip the journal/marker path — the part
        # that has no coverage — every time the box happens to be busy. The
        # utilisation arithmetic itself is covered exhaustively in
        # tests/test_control.py, and for real in the LFM2 test below.
        total_mib=97887,
        free_mib=lambda: 97887,
    )
    seen: list[control.Progress] = []
    result = ctl.start("stub", timeout_s=90, on_progress=seen.append)
    try:
        assert not isinstance(result, Refusal), result
        assert result.ready, result.failure
        assert result.markers == list(control.READY_MARKERS), (
            f"saw {result.markers}; journal tail: {units.journal_tail(unit, 30)}"
        )
        assert [e.marker_index for e in seen if e.kind == "marker"] == [0, 1, 2, 3]
        assert seen[-1].kind == "ready"
    finally:
        ctl.stop("stub")
    assert units.gone(unit) is True


@dataclass
class _StubSpec:
    """A ModelSpec whose 'model' is the stdlib stub server."""

    key: str = "stub"
    id: str = STUB_MODEL
    slot: str = "resident"
    port: int = STUB_PORT
    vram_mib: int | None = 2000
    ctx_tokens: int = 4096
    venv_bin: str = str(Path(sys.executable).parent)

    def served_names(self) -> list[str]:
        return [STUB_MODEL]

    def render_argv(self, util: float, port: int) -> list[str]:
        return [
            sys.executable,
            str(STUB),
            "--port",
            str(port),
            "--model",
            self.id,
            "--delay-ready",
            "6",
            "--emit-boot-lines",
        ]

    def render_env(self) -> dict[str, str]:
        return {"PYTHONUNBUFFERED": "1"}


class _OneModelRegistry:
    def __init__(self, spec) -> None:
        self._spec = spec

    def get(self, key: str):
        if key != self._spec.key:
            raise KeyError(key)
        return self._spec

    def keys(self) -> list[str]:
        return [self._spec.key]


@pytest.fixture(autouse=True)
def _tmp_for_desired(tmp_path, monkeypatch):
    """Desired state goes to a tmp dir, never to the worktree's state/."""
    monkeypatch.setenv("PYTEST_TMP", str(tmp_path))


def test_gpu_module_reports_numbers_or_none_but_never_raises() -> None:
    """gpu.py's new primitives, against the real nvidia-smi on this box."""
    free, total, used = gpu.free_mib(), gpu.total_mib(), gpu.used_by_pids()
    if shutil.which("nvidia-smi") is None:
        assert free is None and total is None and used is None
        return
    assert isinstance(free, int) and isinstance(total, int)
    assert 0 <= free <= total
    assert used is not None and all(isinstance(k, int) and isinstance(v, int) for k, v in used.items())


def test_nvidia_smi_absent_returns_none(monkeypatch) -> None:
    """The tool being gone must be distinguishable from 'nothing is free'.

    Returning 0 here would make control compute a negative utilisation and
    make switch conclude the card emptied the instant nvidia-smi broke.
    """
    def boom(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", boom)
    assert gpu.free_mib() is None
    assert gpu.total_mib() is None
    assert gpu.used_by_pids() is None
