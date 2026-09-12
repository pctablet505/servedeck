"""control.py against a fake systemd, a fake registry and a fake probe.

These are the decisions that cost money when they are wrong — which model may
boot, how much of the card it is handed, whether the last one has really let go
— so each one is exercised as arithmetic and as argv, with no GPU and no
systemd in the room.

The end-to-end counterpart (tests/test_control_e2e.py) proves the same code
path against the real thing; neither file replaces the other. A mock cannot
show that `systemd-run` puts the child in its own cgroup, and a live boot
cannot be made to have exactly 1023 MiB free.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field

import pytest

from servedeck import control, desired as desired_mod, units
from servedeck.control import Control, Refusal
from servedeck.desired import Desired

# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


@dataclass
class FakeSpec:
    """A ModelSpec. Asserted against the Protocol in
    test_fake_spec_satisfies_the_protocol, which is the contract P1's real
    registry has to meet."""

    key: str
    id: str
    slot: str
    port: int
    served_names: list[str] = field(default_factory=list)
    vram_mib: int | None = None
    ctx_tokens: int | None = None
    venv_bin: str = "/opt/venv/bin"

    def __post_init__(self) -> None:
        if not self.served_names:
            self.served_names = [self.id]

    def render_argv(self, util: float, port: int) -> list[str]:
        return [
            f"{self.venv_bin}/vllm",
            "serve",
            self.id,
            "--gpu-memory-utilization",
            str(util),
            "--port",
            str(port),
        ]

    def render_env(self) -> dict[str, str]:
        return {"VLLM_USE_FLASHINFER_SAMPLER": "0"}


class FakeRegistry:
    def __init__(self, *specs: FakeSpec) -> None:
        self._by_key = {spec.key: spec for spec in specs}

    def models(self) -> list[FakeSpec]:
        return list(self._by_key.values())

    def get(self, key: str) -> FakeSpec:
        return self._by_key[key]


class FakeSystemd:
    """An in-memory systemd that answers the five argv shapes units.py builds.

    It reproduces the two behaviours that matter and are easy to get wrong:
    `show` on an unknown unit exits 0 with property defaults, and `--collect`
    removes a unit that failed.
    """

    def __init__(self) -> None:
        self.units: dict[str, dict[str, object]] = {}
        self.journal: dict[str, list[str]] = {}
        self.calls: list[list[str]] = []
        self.started: list[list[str]] = []

    def add(self, name, active_state="active", sub_state="running", result="success",
            n_restarts=0, main_pid=4242) -> None:
        self.units[name] = {
            "ActiveState": active_state,
            "SubState": sub_state,
            "Result": result,
            "NRestarts": str(n_restarts),
            "MainPID": str(main_pid),
            "ExecMainStartTimestamp": "Fri 2026-09-12 13:20:01 IST",
            "LoadState": "loaded",
            "ControlGroup": f"/user.slice/user-1000.slice/user@1000.service/app.slice/{name}.service",
        }

    def collect(self, name: str) -> None:
        """What `--collect` does to a unit that failed: it is simply gone."""
        self.units.pop(name, None)

    def __call__(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        if argv[0] == "systemd-run":
            name = next(a.split("=", 1)[1] for a in argv if a.startswith("--unit="))
            if name in self.units:
                return subprocess.CompletedProcess(
                    argv, 1, "", f"Unit {name}.service already exists."
                )
            self.started.append(argv)
            self.add(name)
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:3] == ["systemctl", "--user", "show"]:
            name, props = argv[5], argv[4].split(",")
            unit = self.units.get(name)
            defaults = {
                "ActiveState": "inactive", "SubState": "dead", "Result": "success",
                "NRestarts": "0", "MainPID": "0", "ExecMainStartTimestamp": "",
                "LoadState": "not-found", "ControlGroup": "",
            }
            source = unit if unit is not None else defaults
            out = "".join(f"{p}={source.get(p, defaults.get(p, ''))}\n" for p in props)
            return subprocess.CompletedProcess(argv, 0, out, "")
        if argv[:4] == ["systemctl", "--user", "list-units", "model-*"]:
            rows = "".join(
                f"{n}.service loaded {u['ActiveState']} {u['SubState']} desc\n"
                for n, u in sorted(self.units.items())
                if n.startswith("model-")
            )
            return subprocess.CompletedProcess(argv, 0, rows, "")
        if argv[:3] == ["systemctl", "--user", "stop"]:
            name = argv[3]
            if name not in self.units:
                return subprocess.CompletedProcess(argv, 5, "", f"Unit {name}.service not loaded.")
            del self.units[name]
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[0] == "journalctl":
            name = argv[argv.index("-u") + 1]
            return subprocess.CompletedProcess(
                argv, 0, "\n".join(self.journal.get(name, [])), ""
            )
        raise AssertionError(f"unexpected argv: {argv}")


def fake_spawn(lines):
    """A journal follower that replays `lines` then reaches EOF."""

    def spawn(argv):
        read_fd, write_fd = os.pipe()
        os.write(write_fd, ("\n".join(lines) + "\n").encode() if lines else b"")
        os.close(write_fd)

        class FakeProc:
            stdout = os.fdopen(read_fd, "rb")

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

        return FakeProc()

    return spawn


class FakeClock:
    """Monotonic time that advances on every read, so a loop that polls
    cannot spin forever and the test never sleeps."""

    def __init__(self, step: float = 0.6) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


BOOT_LINES = [
    "INFO ... Starting vLLM API server on http://127.0.0.1:8031",
    "INFO ... Loading weights took 0.50 seconds",
    "INFO ... GPU KV cache size: 262,144 tokens",
    "INFO ... Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)",
    "INFO ... Application startup complete.",
]


def make_control(systemd, registry, tmp_path, *, probe=None, free=(50000,),
                 total=97887, margin=1024, lines=None, cgroup_pids=None, used=None):
    free_values = list(free)

    def free_mib():
        return free_values[0] if len(free_values) == 1 else free_values.pop(0)

    return Control(
        registry,
        margin_mib=margin,
        total_mib=total,
        desired_path=tmp_path / "desired.json",
        run=systemd,
        spawn=fake_spawn(BOOT_LINES if lines is None else lines),
        probe=probe if probe is not None else (lambda port: None),
        free_mib=free_mib,
        used_by_pids=used if used is not None else (lambda: {}),
        cgroup_pids=cgroup_pids or (lambda unit: [4242]),
        clock=FakeClock(),
        sleep=lambda _s: None,
    )


LFM2 = FakeSpec(key="lfm2", id="LFM2.5-350M", slot="resident", port=8007, vram_mib=3300)
FLASH = FakeSpec(key="flashnext", id="Qwen3.8-Flash-Next", slot="main", port=8001,
                 served_names=["Qwen3.8-Flash-Next", "flashnext"])
BIG27 = FakeSpec(key="qwen27b", id="Qwen3.8-27B-NVFP4", slot="main", port=8004)


# --------------------------------------------------------------------------
# The Protocol P1 must satisfy
# --------------------------------------------------------------------------


def test_fake_spec_satisfies_the_protocol() -> None:
    assert isinstance(LFM2, control.ModelSpec)
    assert isinstance(FakeRegistry(LFM2), control.Registry)


# --------------------------------------------------------------------------
# Utilisation arithmetic (design decision 4)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "free,total,margin,expected",
    [
        # The whole card free minus the margin: 96863/97887 = 0.98953... -> 0.98
        (97887, 97887, 1024, 0.98),
        # With the 3.3 GiB resident already loaded, the main model is simply
        # offered less. This is R4's "a resident can block a big-model boot"
        # turned into arithmetic that cannot block anything.
        (94587, 97887, 1024, 0.95),
        (50000, 97887, 1024, 0.50),
        # FLOOR, not round: 0.9151 must not become 0.92 and eat the margin.
        (90600, 97887, 1024, 0.91),
    ],
)
def test_main_util_is_everything_free_minus_the_margin(free, total, margin, expected, tmp_path) -> None:
    ctl = make_control(FakeSystemd(), FakeRegistry(FLASH), tmp_path,
                       free=(free,), total=total, margin=margin)
    assert ctl.compute_util(FLASH) == expected


def test_resident_util_comes_from_its_budget_not_from_free_memory(tmp_path) -> None:
    """A resident is sized by models.toml, not by what happens to be free —
    otherwise the first model to boot would claim the card."""
    ctl = make_control(FakeSystemd(), FakeRegistry(LFM2), tmp_path, free=(97000,), total=97887)
    assert ctl.compute_util(LFM2) == 0.03  # floor2(3300/97887) = 0.033... -> 0.03


def test_refuses_when_free_is_at_or_below_the_margin(tmp_path) -> None:
    """The failure this box actually produced: 1344 MiB free with a 90 GiB
    model resident. (free - margin)/total is 0.003 — a 'successful' launch
    into a card with nothing in it, which dies minutes later as a CUDA OOM.
    Refuse with a number instead."""
    ctl = make_control(FakeSystemd(), FakeRegistry(FLASH), tmp_path, free=(1024,), margin=1024)
    refusal = ctl.compute_util(FLASH)
    assert isinstance(refusal, Refusal) and refusal.reason == "not_enough_vram"
    # The MESSAGE, not just the reason: there are two guards that both end in
    # `not_enough_vram`, and only this one can say why in terms an operator can
    # act on. Asserting the reason alone lets the margin check be deleted
    # entirely — the floors-to-zero guard downstream would catch this input and
    # the test would still pass, reporting a number instead of a cause.
    assert refusal.message == (
        "1024 MiB free is not more than the 1024 MiB margin; "
        "nothing can be offered to flashnext"
    )


def test_refuses_when_the_remainder_floors_to_zero(tmp_path) -> None:
    """free is above the margin, but only just: 1500-1024 = 476 MiB is 0.00 of
    the card. A util of 0.0 is not a small model, it is a crash."""
    ctl = make_control(FakeSystemd(), FakeRegistry(FLASH), tmp_path, free=(1500,), margin=1024)
    assert isinstance(ctl.compute_util(FLASH), Refusal)


def test_refuses_rather_than_guessing_when_nvidia_smi_is_mute(tmp_path) -> None:
    ctl = Control(FakeRegistry(FLASH), total_mib=97887, desired_path=tmp_path / "d.json",
                  run=FakeSystemd(), free_mib=lambda: None)
    refusal = ctl.compute_util(FLASH)
    assert isinstance(refusal, Refusal) and refusal.reason == "gpu_unavailable"


def test_resident_without_a_budget_is_refused(tmp_path) -> None:
    spec = FakeSpec(key="x", id="X", slot="resident", port=8009, vram_mib=None)
    ctl = make_control(FakeSystemd(), FakeRegistry(spec), tmp_path)
    assert isinstance(ctl.compute_util(spec), Refusal)


def test_floor2_truncates() -> None:
    assert control.floor2(0.98953) == 0.98
    assert control.floor2(0.9999) == 0.99
    assert control.floor2(0.03371) == 0.03


# --------------------------------------------------------------------------
# start
# --------------------------------------------------------------------------


def test_start_builds_the_unit_from_the_registry_and_the_live_card(tmp_path) -> None:
    systemd = FakeSystemd()
    ctl = make_control(systemd, FakeRegistry(FLASH), tmp_path, free=(94587,),
                       probe=lambda port: ["Qwen3.8-Flash-Next"] if port == 8001 else None)
    result = ctl.start("flashnext")
    assert not isinstance(result, Refusal)
    assert result.ready and result.util == 0.95
    assert result.markers == list(control.READY_MARKERS)

    argv = systemd.started[0]
    assert argv[:4] == ["systemd-run", "--user", "--unit=model-flashnext", "--collect"]
    assert "-p" in argv and "Restart=on-failure" in argv
    assert "--setenv=VLLM_USE_FLASHINFER_SAMPLER=0" in argv
    # The computed utilisation reaches vLLM, not a configured one.
    command = argv[argv.index("--") + 1 :]
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.95"


def test_start_records_intent_in_desired_state(tmp_path) -> None:
    systemd = FakeSystemd()
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path,
                       probe=lambda port: ["LFM2.5-350M"])
    ctl.start("lfm2")
    assert desired_mod.load(tmp_path / "desired.json") == Desired(main=None, residents=["lfm2"])


def test_a_failed_start_does_not_become_desired(tmp_path) -> None:
    """Desired state is intent that survives a crash, so it must only record a
    model that actually came up. A model that never booted, marked desired,
    is a reconcile loop that retries it at every servedeck restart forever."""
    systemd = FakeSystemd()
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path, probe=lambda port: None, lines=[])
    result = ctl.start("lfm2", timeout_s=3.0)
    assert not isinstance(result, Refusal) and not result.ready
    assert desired_mod.load(tmp_path / "desired.json") == Desired()


def test_main_slot_is_exclusive(tmp_path) -> None:
    """R4: two main models on one card is the failure mode the slot exists to
    prevent. The refusal names the model in the way, because the operator's
    next question is always 'what is holding it'."""
    systemd = FakeSystemd()
    systemd.add("model-qwen27b")
    ctl = make_control(systemd, FakeRegistry(FLASH, BIG27, LFM2), tmp_path,
                       probe=lambda port: ["Qwen3.8-27B-NVFP4"] if port == 8004 else None)
    refusal = ctl.start("flashnext")
    assert isinstance(refusal, Refusal)
    assert refusal.reason == "main_slot_busy" and refusal.live_key == "qwen27b"
    assert systemd.started == [], "nothing may be launched by a refused start"


def test_a_resident_may_start_while_the_main_slot_is_held(tmp_path) -> None:
    """The exclusivity is per-slot. A resident booting alongside the main
    model is the normal steady state, not a conflict."""
    systemd = FakeSystemd()
    systemd.add("model-qwen27b")
    ctl = make_control(systemd, FakeRegistry(BIG27, LFM2), tmp_path,
                       probe=lambda port: ["LFM2.5-350M"] if port == 8007 else None)
    assert not isinstance(ctl.start("lfm2"), Refusal)


def test_start_refuses_an_unknown_model(tmp_path) -> None:
    ctl = make_control(FakeSystemd(), FakeRegistry(LFM2), tmp_path)
    assert ctl.start("nope").reason == "unknown_model"


def test_start_refuses_a_model_that_is_already_up(tmp_path) -> None:
    systemd = FakeSystemd()
    systemd.add("model-lfm2")
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path, probe=lambda port: ["LFM2.5-350M"])
    refusal = ctl.start("lfm2")
    assert isinstance(refusal, Refusal) and refusal.reason == "already_live"
    assert systemd.started == []


# --------------------------------------------------------------------------
# wait_ready
# --------------------------------------------------------------------------


def test_wait_ready_reports_each_marker_in_order_through_the_callback(tmp_path) -> None:
    systemd = FakeSystemd()
    systemd.add("model-lfm2")
    seen: list[control.Progress] = []
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path, probe=lambda port: ["LFM2.5-350M"])
    result = ctl.wait_ready("lfm2", timeout_s=60, on_progress=seen.append)
    assert result.ready
    markers = [event for event in seen if event.kind == "marker"]
    assert [event.marker_index for event in markers] == [0, 1, 2, 3]
    assert result.markers == list(control.READY_MARKERS)
    assert seen[-1].kind == "ready"


def test_markers_out_of_order_do_not_count(tmp_path) -> None:
    """The sequence is the assertion. "Application startup complete." before
    the weights loaded is a stale journal line from the PREVIOUS boot, which
    is exactly how a re-used unit name reports ready while it is still
    loading."""
    systemd = FakeSystemd()
    systemd.add("model-lfm2")
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path, probe=lambda port: None,
                       lines=["Application startup complete.", "Loading weights took 1s"])
    result = ctl.wait_ready("lfm2", timeout_s=3.0)
    assert result.markers == ["Loading weights took"]
    assert not result.ready


def test_readiness_is_the_probe_not_the_markers(tmp_path) -> None:
    """A model that skips CUDA graph capture still serves. Markers are the
    progress bar; `GET /v1/models` answering with the id is the fact."""
    systemd = FakeSystemd()
    systemd.add("model-lfm2")
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path, probe=lambda port: ["LFM2.5-350M"],
                       lines=["Loading weights took 1s"])
    result = ctl.wait_ready("lfm2", timeout_s=60)
    assert result.ready and result.markers == ["Loading weights took"]


def test_the_last_marker_still_counts_when_the_port_answers_first(tmp_path) -> None:
    """The race `_READY_DRAIN_S` exists for, made deterministic.

    vLLM logs "Application startup complete." and begins serving at nearly the
    same instant, and the journal line has further to travel than the HTTP
    response. The probe therefore usually wins, and a wait that returned the
    moment it won would drop the final marker from almost every successful
    boot — a progress display permanently stuck at 3/4, and a marker assertion
    that fails at random.

    Here the last line is written into the journal pipe BY the probe, at the
    moment it first reports ready: the worst ordering, every run.
    """
    systemd = FakeSystemd()
    systemd.add("model-lfm2")
    read_fd, write_fd = os.pipe()
    os.write(write_fd, "\n".join(BOOT_LINES[:-1]).encode() + b"\n")
    written = []

    def probe(port):
        if not written:  # the log line loses the race, once
            written.append(True)
            os.write(write_fd, BOOT_LINES[-1].encode() + b"\n")
        return ["LFM2.5-350M"]

    class FakeProc:
        stdout = os.fdopen(read_fd, "rb")

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    try:
        ctl = Control(
            FakeRegistry(LFM2), total_mib=97887, desired_path=tmp_path / "desired.json",
            run=systemd, spawn=lambda argv: FakeProc(), probe=probe,
            free_mib=lambda: 50000, used_by_pids=lambda: {}, clock=FakeClock(),
            sleep=lambda _s: None,
        )
        result = ctl.wait_ready("lfm2", timeout_s=60)
    finally:
        os.close(write_fd)
    assert result.ready
    assert result.markers == list(control.READY_MARKERS)


def test_a_probe_answering_with_the_wrong_model_is_not_ready(tmp_path) -> None:
    """Something is listening on the port — the previous model, or another
    tool. Port-is-open was v1's readiness test and it is how a switch reports
    the new model up while the old one is still answering."""
    systemd = FakeSystemd()
    systemd.add("model-lfm2")
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path,
                       probe=lambda port: ["Qwen3.8-27B-NVFP4"])
    assert not ctl.wait_ready("lfm2", timeout_s=3.0).ready


def test_an_alias_counts_as_ready(tmp_path) -> None:
    systemd = FakeSystemd()
    systemd.add("model-flashnext")
    ctl = make_control(systemd, FakeRegistry(FLASH), tmp_path, probe=lambda port: ["flashnext"])
    assert ctl.wait_ready("flashnext", timeout_s=60).ready


def test_a_unit_that_vanished_is_a_failure_not_a_clean_stop(tmp_path) -> None:
    """The `--collect` trap, at the level that matters.

    The unit crashed, systemd collected it, and `show` now answers
    inactive/success — which reads as "stopped normally". If wait_ready
    believed that, every crashed boot would be reported as a success with an
    empty log, and the dashboard would show a model that is not there.
    """
    systemd = FakeSystemd()
    systemd.add("model-lfm2")
    systemd.journal["model-lfm2"] = ["ValueError: unsupported dtype", "engine core failed"]
    systemd.collect("model-lfm2")
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path, probe=lambda port: None, lines=[])
    result = ctl.wait_ready("lfm2", timeout_s=60)
    assert not result.ready
    assert "no longer exists" in (result.failure or "")
    assert result.journal == ["ValueError: unsupported dtype", "engine core failed"]


def test_a_crash_loop_is_a_failure_not_a_slow_boot(tmp_path) -> None:
    systemd = FakeSystemd()
    systemd.add("model-lfm2", active_state="activating", sub_state="auto-restart", n_restarts=3)
    systemd.journal["model-lfm2"] = ["CUDA out of memory"]
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path, probe=lambda port: None, lines=[])
    result = ctl.wait_ready("lfm2", timeout_s=60)
    assert not result.ready and "crash-looping" in (result.failure or "")
    assert result.journal == ["CUDA out of memory"]


def test_a_timeout_reports_how_far_the_boot_got(tmp_path) -> None:
    systemd = FakeSystemd()
    systemd.add("model-lfm2")
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path, probe=lambda port: None,
                       lines=["Loading weights took 1s", "GPU KV cache size: 12 tokens"])
    result = ctl.wait_ready("lfm2", timeout_s=5.0)
    assert not result.ready and "2/4 boot markers" in (result.failure or "")


# --------------------------------------------------------------------------
# live()
# --------------------------------------------------------------------------


def test_live_joins_units_with_the_probe(tmp_path) -> None:
    systemd = FakeSystemd()
    systemd.add("model-lfm2")
    systemd.add("model-flashnext", active_state="activating", sub_state="start", main_pid=0)
    ctl = make_control(systemd, FakeRegistry(LFM2, FLASH), tmp_path,
                       probe=lambda port: ["LFM2.5-350M"] if port == 8007 else None)
    live = {model.key: model for model in ctl.live()}
    assert live["lfm2"].ready and live["lfm2"].port == 8007 and live["lfm2"].pid == 4242
    assert not live["flashnext"].ready and live["flashnext"].state == "activating"


def test_live_reports_an_unregistered_unit_without_guessing_at_it(tmp_path) -> None:
    """A `model-*` unit whose key is not in models.toml has no port to probe
    and no slot to reason about. Reported, never acted on — inventing a slot
    for it is how a stray unit ends up blocking the main model."""
    systemd = FakeSystemd()
    systemd.add("model-mystery")
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path)
    (model,) = ctl.live()
    assert model.unknown and model.port is None and not model.ready


def test_live_never_sees_a_unit_outside_the_model_namespace(tmp_path) -> None:
    systemd = FakeSystemd()
    systemd.add("qwen-vllm")
    systemd.add("servedeck")
    systemd.add("lfm2-350m")
    systemd.add("model-lfm2")
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path, probe=lambda port: ["LFM2.5-350M"])
    assert [model.unit for model in ctl.live()] == ["model-lfm2"]


# --------------------------------------------------------------------------
# stop / switch
# --------------------------------------------------------------------------


def test_stop_clears_the_main_slot_from_desired_state(tmp_path) -> None:
    systemd = FakeSystemd()
    systemd.add("model-flashnext")
    desired_mod.save(Desired(main="flashnext", residents=["lfm2"]), tmp_path / "desired.json")
    ctl = make_control(systemd, FakeRegistry(FLASH, LFM2), tmp_path)
    result = ctl.stop("flashnext")
    assert not isinstance(result, Refusal) and result.was_live
    assert desired_mod.load(tmp_path / "desired.json") == Desired(main=None, residents=["lfm2"])
    assert "model-flashnext" not in systemd.units


def test_stop_accounts_for_the_whole_cgroup_not_just_mainpid(tmp_path) -> None:
    """vLLM v1's MainPID is the API server and holds no GPU memory; the engine
    core and workers hold all of it. Charging the unit only its MainPID reports
    ~0 MiB held, and switch() then believes a 90 GiB model released instantly.
    """
    systemd = FakeSystemd()
    systemd.add("model-flashnext", main_pid=1000)
    ctl = make_control(
        systemd, FakeRegistry(FLASH), tmp_path, free=(1344,),
        used=lambda: {1000: 0, 1001: 45000, 1002: 45000, 999: 3300},  # 999 = the resident
        cgroup_pids=lambda unit: [1000, 1001, 1002],
    )
    result = ctl.stop("flashnext")
    assert result.held_mib == 90000  # not 0, and not 93300


def test_switch_waits_for_the_card_to_actually_empty(tmp_path) -> None:
    """The driver frees a context asynchronously. A switch that trusts the
    stop's exit code launches the next model into a card that still has the
    last one in it; the OOM surfaces a minute later with no obvious cause.
    """
    systemd = FakeSystemd()
    systemd.add("model-qwen27b", main_pid=1000)
    # free_mib readings, in order: the initial reading, the pre-stop capture,
    # then the release poll — still full, still full, then released.
    free_readings = [1344, 1344, 1500, 2000, 80000, 80000, 80000, 80000]
    ctl = make_control(
        systemd, FakeRegistry(BIG27, FLASH), tmp_path, free=tuple(free_readings),
        used=lambda: {1000: 90000},
        cgroup_pids=lambda unit: [1000],
        probe=lambda port: ["Qwen3.8-Flash-Next"] if port == 8001 else None,
    )
    result = ctl.switch("flashnext")
    assert not isinstance(result, Refusal)
    assert result.stopped is not None and result.stopped.held_mib == 90000
    assert result.released
    # Crucially: it did not launch until the memory came back.
    assert result.free_after_mib == 80000
    assert not isinstance(result.started, Refusal) and result.started.ready


def test_switch_refuses_to_launch_when_the_memory_never_comes_back(tmp_path) -> None:
    systemd = FakeSystemd()
    systemd.add("model-qwen27b", main_pid=1000)
    ctl = make_control(
        systemd, FakeRegistry(BIG27, FLASH), tmp_path, free=(1344,),  # never rises
        used=lambda: {1000: 90000},
        cgroup_pids=lambda unit: [1000],
        probe=lambda port: ["Qwen3.8-Flash-Next"],
    )
    result = ctl.switch("flashnext", release_timeout_s=5.0)
    assert not isinstance(result, Refusal)
    assert not result.released
    assert isinstance(result.started, Refusal)
    assert result.started.reason == "vram_not_released"
    assert systemd.started == [], "nothing may boot into a card that is still occupied"


def test_switch_refuses_a_resident(tmp_path) -> None:
    ctl = make_control(FakeSystemd(), FakeRegistry(LFM2), tmp_path)
    refusal = ctl.switch("lfm2")
    assert isinstance(refusal, Refusal) and refusal.reason == "not_main_slot"


def test_switch_leaves_residents_alone(tmp_path) -> None:
    """§2.2: 'Residents are untouched.' A switch that stopped the resident
    would take the small always-on model down with every main-model change."""
    systemd = FakeSystemd()
    systemd.add("model-qwen27b", main_pid=1000)
    systemd.add("model-lfm2", main_pid=999)
    ctl = make_control(
        systemd, FakeRegistry(BIG27, FLASH, LFM2), tmp_path, free=(80000,),
        used=lambda: {1000: 90000, 999: 3300},
        cgroup_pids=lambda unit: [1000] if unit == "model-qwen27b" else [999],
        probe=lambda port: {8001: ["Qwen3.8-Flash-Next"], 8007: ["LFM2.5-350M"]}.get(port),
    )
    ctl.switch("flashnext")
    assert "model-lfm2" in systemd.units
    assert ["systemctl", "--user", "stop", "model-lfm2"] not in systemd.calls


# --------------------------------------------------------------------------
# adopt
# --------------------------------------------------------------------------


def test_adopt_files_live_units_under_their_slot(tmp_path) -> None:
    systemd = FakeSystemd()
    systemd.add("model-qwen27b")
    systemd.add("model-lfm2")
    ctl = make_control(systemd, FakeRegistry(BIG27, LFM2), tmp_path,
                       probe=lambda port: ["x"])
    result = ctl.adopt()
    assert sorted(result.adopted) == ["lfm2", "qwen27b"]
    assert result.desired == Desired(main="qwen27b", residents=["lfm2"])
    assert desired_mod.load(tmp_path / "desired.json") == result.desired


def test_adopt_reports_but_never_adopts_an_unregistered_unit(tmp_path) -> None:
    systemd = FakeSystemd()
    systemd.add("model-mystery")
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path)
    result = ctl.adopt()
    assert result.unknown_units == ["model-mystery"]
    assert result.adopted == [] and result.desired == Desired()
    assert not (tmp_path / "desired.json").exists()


def test_adopt_writes_nothing_when_there_is_nothing_to_adopt(tmp_path) -> None:
    systemd = FakeSystemd()
    systemd.add("model-lfm2")
    desired_mod.save(Desired(residents=["lfm2"]), tmp_path / "desired.json")
    before = (tmp_path / "desired.json").read_text()
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path, probe=lambda port: ["LFM2.5-350M"])
    assert ctl.adopt().adopted == []
    assert (tmp_path / "desired.json").read_text() == before


# --------------------------------------------------------------------------
# reconcile
# --------------------------------------------------------------------------


def test_reconcile_does_not_restart_a_live_model(tmp_path) -> None:
    """The rule with teeth (R4). Servedeck restarting is not a reason for a
    90 GiB model to restart — that is the failure that made 'stopping the
    dashboard killed the model' a sentence anyone had to write."""
    systemd = FakeSystemd()
    systemd.add("model-qwen27b")
    systemd.add("model-lfm2")
    ctl = make_control(systemd, FakeRegistry(BIG27, LFM2), tmp_path,
                       probe=lambda port: ["Qwen3.8-27B-NVFP4"] if port == 8004 else ["LFM2.5-350M"])
    result = ctl.reconcile(Desired(main="qwen27b", residents=["lfm2"]))
    assert sorted(result.already_live) == ["lfm2", "qwen27b"]
    assert result.started == []
    assert systemd.started == []
    assert not any(call[:3] == ["systemctl", "--user", "stop"] for call in systemd.calls)


def test_reconcile_leaves_a_still_booting_unit_alone(tmp_path) -> None:
    """Not ready is not the same as not there. A model 40 seconds into a
    90-second boot must survive the dashboard coming up."""
    systemd = FakeSystemd()
    systemd.add("model-qwen27b", active_state="activating", sub_state="start")
    ctl = make_control(systemd, FakeRegistry(BIG27), tmp_path, probe=lambda port: None)
    result = ctl.reconcile(Desired(main="qwen27b"))
    assert result.booting == ["qwen27b"] and result.started == []
    assert systemd.started == []


def test_reconcile_starts_what_is_missing_residents_first(tmp_path) -> None:
    """Residents before the main model, because the main model's utilisation
    is computed from what is free AFTER the residents have taken their share.
    The other order hands the main model memory the resident then cannot get."""
    systemd = FakeSystemd()
    ctl = make_control(systemd, FakeRegistry(BIG27, LFM2), tmp_path, free=(94587,),
                       probe=lambda port: ["Qwen3.8-27B-NVFP4"] if port == 8004 else ["LFM2.5-350M"])
    result = ctl.reconcile(Desired(main="qwen27b", residents=["lfm2"]))
    assert [r.key for r in result.started] == ["lfm2", "qwen27b"]
    assert [a[2] for a in systemd.started] == ["--unit=model-lfm2", "--unit=model-qwen27b"]


def test_reconcile_never_writes_desired_state(tmp_path) -> None:
    """Desired state is explicit intent (start/stop/switch/adopt). If
    reconcile wrote it, a failed start at boot would quietly erase the operator's
    intent and the model would never be retried."""
    systemd = FakeSystemd()
    path = tmp_path / "desired.json"
    desired_mod.save(Desired(main="qwen27b", residents=["lfm2"]), path)
    before = path.read_text()
    ctl = make_control(systemd, FakeRegistry(BIG27, LFM2), tmp_path, free=(1024,))
    result = ctl.reconcile()
    assert [r.reason for r in result.refused] == ["not_enough_vram", "not_enough_vram"]
    assert path.read_text() == before


# --------------------------------------------------------------------------
# desired.json — schema and migration
# --------------------------------------------------------------------------


def test_desired_roundtrips_as_version_2(tmp_path) -> None:
    path = tmp_path / "desired.json"
    desired_mod.save(Desired(main="flashnext", residents=["lfm2"]), path)
    assert json.loads(path.read_text()) == {
        "version": 2,
        "main": "flashnext",
        "residents": ["lfm2"],
    }
    assert desired_mod.load(path) == Desired(main="flashnext", residents=["lfm2"])


def test_a_v1_running_file_migrates_to_a_main_model(tmp_path, caplog) -> None:
    """The live file on this box, verbatim (2026-09-12)."""
    path = tmp_path / "desired.json"
    path.write_text(json.dumps({
        "version": 1, "desired_state": "RUNNING", "repo_id": "RadixArk/Qwen3.8-27B-NVFP4",
        "backend": "inline", "served_name": "Qwen3.8-27B-NVFP4", "port": 8004, "util": 0.91,
        "max_model_len": 262144, "max_num_seqs": 16, "auto_restart": True,
        "suspended": False, "attempts": [], "updated_at": "2026-09-12T06:45:39.888274+00:00",
    }))
    with caplog.at_level("WARNING"):
        assert desired_mod.load(path) == Desired(main="inline", residents=[])
    assert "version 1" in caplog.text
    assert "launcher name, not a registry key" in caplog.text


def test_a_v1_stopped_file_migrates_to_no_main(tmp_path) -> None:
    path = tmp_path / "desired.json"
    path.write_text(json.dumps({"version": 1, "desired_state": "STOPPED", "backend": "inline"}))
    assert desired_mod.load(path) == Desired(main=None, residents=[])


def test_migration_does_not_rewrite_the_v1_file_on_read(tmp_path) -> None:
    """Reading must not destroy the rollback. Step 4 of the migration keeps
    the old units on disk precisely so it can be undone; a read that rewrote
    desired.json would take v1's port/util/max_model_len with it."""
    path = tmp_path / "desired.json"
    body = json.dumps({"version": 1, "desired_state": "RUNNING", "backend": "inline", "util": 0.91})
    path.write_text(body)
    desired_mod.load(path)
    assert path.read_text() == body


@pytest.mark.parametrize("body", ["", "not json", "[]", '{"version": 99}', '{"version": 2}'])
def test_an_unreadable_desired_file_degrades_to_wanting_nothing(tmp_path, body) -> None:
    """Never raise out of here: failing to parse this file must not be the
    reason servedeck will not start."""
    path = tmp_path / "desired.json"
    path.write_text(body)
    assert desired_mod.load(path) == Desired()


def test_a_missing_desired_file_is_not_an_error(tmp_path) -> None:
    assert desired_mod.load(tmp_path / "nothing.json") == Desired()


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path) -> None:
    path = tmp_path / "desired.json"
    desired_mod.save(Desired(main="a"), path)
    desired_mod.save(Desired(main="b"), path)
    assert [p.name for p in tmp_path.iterdir()] == ["desired.json"]
    assert desired_mod.load(path).main == "b"


def test_residents_are_deduplicated_on_read_and_on_add(tmp_path) -> None:
    path = tmp_path / "desired.json"
    path.write_text(json.dumps({"version": 2, "main": None, "residents": ["lfm2", "lfm2"]}))
    assert desired_mod.load(path).residents == ["lfm2"]
    assert Desired(residents=["lfm2"]).with_resident("lfm2").residents == ["lfm2"]


# --------------------------------------------------------------------------
# Blast radius
# --------------------------------------------------------------------------


def test_control_can_only_ever_name_model_units(tmp_path) -> None:
    """Every systemctl call this module makes, for every operation, names a
    unit in the `model-` namespace. This is the guard that keeps a bug in
    control.py from reaching `qwen-vllm` or `servedeck` itself."""
    systemd = FakeSystemd()
    systemd.add("model-lfm2")
    systemd.add("qwen-vllm")
    ctl = make_control(systemd, FakeRegistry(LFM2, FLASH), tmp_path, free=(50000,),
                       probe=lambda port: ["LFM2.5-350M"] if port == 8007 else None)
    ctl.live()
    ctl.adopt()
    ctl.stop("lfm2")
    ctl.start("lfm2")
    ctl.reconcile(Desired(residents=["lfm2"]))

    for call in systemd.calls:
        named = [a for a in call if a.startswith("model-") or a.startswith("--unit=model-")]
        if call[:4] == ["systemctl", "--user", "list-units", "model-*"]:
            continue
        assert named, f"a call that names no model unit: {call}"
        for name in named:
            assert units.valid_unit_name(name.removeprefix("--unit="))
    assert "qwen-vllm" in systemd.units, "an untouched unit must stay untouched"
