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
import math
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
    names: list[str] = field(default_factory=list)
    vram_mib: int | None = None
    ctx_tokens: int = 4096
    venv_bin: str = "/opt/venv/bin"

    def __post_init__(self) -> None:
        if not self.names:
            self.names = [self.id]

    def served_names(self) -> list[str]:
        return list(self.names)

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

    def get(self, key: str) -> FakeSpec:
        return self._by_key[key]

    def keys(self) -> list[str]:
        return list(self._by_key)


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
        # The user manager's environment as measured on this box: session
        # variables plus two real credentials an operator exported at login.
        self.environment: list[str] = [
            "HOME=/home/pctablet505",
            "PATH=/usr/bin:/bin",
            "LANG=en_US.UTF-8",
            "KITE_API_KEY=4eabvmt7jnne18w2",
            "KITE_API_SECRET=v41zj8cxazl09fz8gniaffo3xeqhqmlh",
        ]
        self.env_query_fails = False

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
        if argv == ["systemctl", "--user", "show-environment"]:
            if self.env_query_fails:
                return subprocess.CompletedProcess(argv, 1, "", "Failed to connect to bus.")
            return subprocess.CompletedProcess(argv, 0, "\n".join(self.environment) + "\n", "")
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
        # NOT `{}`: an empty attribution sends every stop down the plateau
        # fallback, so the default fixture would silently exercise the
        # degraded path and the target-based wait — the one that actually runs
        # on this box — would be tested only where a test opted in.
        used_by_pids=used if used is not None else (lambda: {4242: 2000}),
        cgroup_pids=cgroup_pids or (lambda unit: [4242]),
        clock=FakeClock(),
        sleep=lambda _s: None,
        # No socket table in the fakes: a port that "answers" in a test is a
        # unit's port, never an adoption, unless the test installs a listener.
        listener_pid=lambda port: None,
    )


LFM2 = FakeSpec(key="lfm2", id="LFM2.5-350M", slot="resident", port=8007, vram_mib=3300)
FLASH = FakeSpec(key="flashnext", id="Qwen3.8-Flash-Next", slot="main", port=8001,
                 names=["Qwen3.8-Flash-Next", "flashnext"])
BIG27 = FakeSpec(key="qwen27b", id="Qwen3.8-27B-NVFP4", slot="main", port=8004)


# --------------------------------------------------------------------------
# The Protocol P1 must satisfy
# --------------------------------------------------------------------------


def test_fake_spec_satisfies_the_protocol() -> None:
    assert isinstance(LFM2, control.ModelSpec)
    assert isinstance(FakeRegistry(LFM2), control.Registry)


def test_served_names_is_a_method_and_the_protocol_says_so() -> None:
    """P1's real model object exposes served_names() as a method. A Protocol
    that declared it an attribute would still pass isinstance against that
    object — and then fail inside wait_ready with "method object is not
    iterable", at model-boot time. So the contract must name the callable, and
    control must call it.
    """
    assert callable(LFM2.served_names)
    assert list(LFM2.served_names()) == ["LFM2.5-350M"]
    assert list(FLASH.served_names()) == ["Qwen3.8-Flash-Next", "flashnext"]


def test_a_spec_whose_served_names_is_a_bare_list_is_not_a_ModelSpec() -> None:
    """The whole point of making it a method: the shape that would explode
    later must fail the conformance check now."""

    @dataclass
    class ListSpec:
        key: str = "x"
        id: str = "X"
        slot: str = "resident"
        port: int = 8009
        served_names: list[str] = field(default_factory=list)  # the wrong shape
        vram_mib: int | None = 1000
        ctx_tokens: int = 4096
        venv_bin: str = "/opt/venv/bin"

        def render_argv(self, util, port):
            return []

        def render_env(self):
            return {}

    # isinstance() on a runtime_checkable Protocol only checks that the
    # attribute EXISTS, which is exactly why the docstring above matters: it
    # passes here and breaks at boot. Pin the real discriminator instead.
    assert not callable(ListSpec().served_names)


def test_the_registry_protocol_asks_for_nothing_it_does_not_call() -> None:
    """`models()` was in the contract and called by nothing. A method P1 must
    implement for no reason is where a wrong assumption about ordering or
    laziness hides."""
    assert not hasattr(control.Registry, "models")


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


@pytest.mark.parametrize("free", [2000, 5000, 20000, 50000, 90000, 97887])
@pytest.mark.parametrize("slot", ["main", "resident"])
def test_the_margin_leaves_the_launching_worker_its_cuda_context(free, slot, tmp_path) -> None:
    """`--gpu-memory-utilization` is a fraction vLLM will FILL; the process
    filling it must first build a CUDA context and load cuBLAS/cuDNN — a few
    hundred MiB that are not in the fraction and are taken while the
    allocation is happening. Hand a model everything free and its own startup
    is what pushes it over.

    So whatever compute_util returns, at least 700 MiB of the free pool must
    still be there for the worker about to claim it.
    """
    total = 97887
    spec = FakeSpec(key="k", id="K", slot=slot, port=8009,
                    vram_mib=(None if slot == "main" else 1500))
    ctl = make_control(FakeSystemd(), FakeRegistry(spec), tmp_path,
                       free=(free,), total=total, margin=1024)
    util = ctl.compute_util(spec)
    if isinstance(util, Refusal):
        return  # refusing is always safe; this test is about what it hands out
    assert math.ceil(total * util) <= free - 700, (
        f"util {util} of {total} MiB is {math.ceil(total * util)} MiB out of {free} free"
    )


def test_a_resident_whose_budget_does_not_fit_is_refused(tmp_path) -> None:
    """A resident's utilisation comes from its BUDGET, so nothing in the
    arithmetic notices the budget does not fit. Without this the unit
    launches, vLLM asks for 3.3 GiB of a card with 2 GiB free, and the failure
    arrives minutes later as a CUDA OOM in journald with no mention of the
    number that was wrong."""
    ctl = make_control(FakeSystemd(), FakeRegistry(LFM2), tmp_path, free=(4000,), margin=1024)
    refusal = ctl.compute_util(LFM2)  # budget 3300, available 4000-1024 = 2976
    assert isinstance(refusal, Refusal) and refusal.reason == "not_enough_vram"
    assert "3300" in refusal.message and "2976" in refusal.message


def test_a_resident_that_exactly_fits_is_allowed(tmp_path) -> None:
    """The boundary, so the guard is `>` and not `>=`: a budget equal to what
    is available is fundable, and refusing it would make the margin count
    twice."""
    ctl = make_control(FakeSystemd(), FakeRegistry(LFM2), tmp_path,
                       free=(3300 + 1024,), margin=1024)
    assert ctl.compute_util(LFM2) == 0.03


def test_start_refuses_a_resident_that_does_not_fit(tmp_path) -> None:
    systemd = FakeSystemd()
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path, free=(4000,))
    assert isinstance(ctl.start("lfm2"), Refusal)
    assert systemd.started == []


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


@pytest.mark.parametrize(
    "name",
    [
        # Measured in this box's own user-manager environment.
        "KITE_API_KEY", "KITE_API_SECRET",
        "GITHUB_TOKEN", "HF_TOKEN", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY",
        "PASSWORD", "DB_PASSWORD", "PASS", "SECRET", "TOKEN", "KEY",
        "GOOGLE_APPLICATION_CREDENTIALS", "CREDENTIAL_FILE",
        "api_key", "Secret_Thing",  # case-insensitive
    ],
)
def test_secret_names_are_recognised(name: str) -> None:
    assert control.SECRET_NAME_RE.search(name)


@pytest.mark.parametrize(
    "name",
    [
        # Over-redaction is not free: an unset variable a model needed is a
        # boot failure with no message, so the word must be a whole
        # underscore-delimited component, not a substring.
        "MONKEY_BUSINESS", "KEYBOARD_LAYOUT", "PASSTHROUGH", "TOKENIZERS_PARALLELISM",
        "HOME", "PATH", "LANG", "DISPLAY", "XDG_RUNTIME_DIR", "SSH_AUTH_SOCK",
        "HF_HOME", "VLLM_USE_FLASHINFER_SAMPLER", "CUDA_VISIBLE_DEVICES",
        "KEYS_DIR", "LOW_PASSES",
    ],
)
def test_innocent_names_are_left_alone(name: str) -> None:
    assert not control.SECRET_NAME_RE.search(name)


def test_secret_env_names_comes_from_the_live_manager_environment(tmp_path) -> None:
    """Computed at start time, not written down: the manager's environment is
    whatever the login session put there and changes without anyone editing
    this repo, so a hardcoded deny-list would be correct once and quietly
    wrong afterwards."""
    systemd = FakeSystemd()
    systemd.environment.append("NEWLY_EXPORTED_TOKEN=abc")
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path)
    assert ctl.secret_env_names() == ["KITE_API_KEY", "KITE_API_SECRET", "NEWLY_EXPORTED_TOKEN"]


def test_start_unsets_every_inherited_credential(tmp_path) -> None:
    """The finding this closes: a transient unit inherits the USER MANAGER's
    environment, and on this box that contains two live broker credentials. A
    model process runs arbitrary prompts, logs freely and can be asked to print
    its own environment, so the only safe amount of unrelated credential in it
    is none."""
    systemd = FakeSystemd()
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path, probe=lambda port: ["LFM2.5-350M"])
    assert not isinstance(ctl.start("lfm2"), Refusal)
    argv = systemd.started[0]
    assert "UnsetEnvironment=KITE_API_KEY" in argv
    assert "UnsetEnvironment=KITE_API_SECRET" in argv
    # ...and nothing innocent was swept up with them.
    unset = [a.split("=", 1)[1] for a in argv if a.startswith("UnsetEnvironment=")]
    assert unset == ["KITE_API_KEY", "KITE_API_SECRET"]


def test_a_variable_the_registry_declares_is_the_models_own_and_survives(tmp_path) -> None:
    """The registry is the single source of truth for what a model needs (R1).
    An HF_TOKEN declared in models.toml for a gated repo is a deliberate,
    source-controlled decision, not ambient leakage — redacting it would break
    the boot, and `units.start_transient` would refuse the contradiction
    anyway."""
    systemd = FakeSystemd()
    systemd.environment.append("HF_TOKEN=from-the-session")

    spec = FakeSpec(key="gated", id="Gated", slot="resident", port=8008, vram_mib=2000)
    spec.render_env = lambda: {"HF_TOKEN": "from-the-registry"}  # type: ignore[method-assign]
    ctl = make_control(systemd, FakeRegistry(spec), tmp_path, probe=lambda port: ["Gated"])

    assert not isinstance(ctl.start("gated"), Refusal)
    argv = systemd.started[0]
    assert "--setenv=HF_TOKEN=from-the-registry" in argv
    assert "UnsetEnvironment=HF_TOKEN" not in argv
    assert "UnsetEnvironment=KITE_API_KEY" in argv


def test_start_refuses_rather_than_leaking_when_the_environment_cannot_be_read(tmp_path) -> None:
    """Fail closed. With no way to know what to redact, starting anyway would
    hand the model every ambient credential — silently. Refusing is loud."""
    systemd = FakeSystemd()
    systemd.env_query_fails = True
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path, probe=lambda port: ["LFM2.5-350M"])
    refusal = ctl.start("lfm2")
    assert isinstance(refusal, Refusal) and refusal.reason == "env_scan_failed"
    assert systemd.started == [], "nothing may launch when the redaction list is unknown"


def test_no_secret_value_is_ever_read_logged_or_returned(tmp_path, caplog) -> None:
    """Names in, values never. `units.manager_environment_names` discards the
    value half before returning, so there is nothing on this path for a log
    line, an exception or a Refusal message to leak."""
    systemd = FakeSystemd()
    secret = "v41zj8cxazl09fz8gniaffo3xeqhqmlh"
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path, probe=lambda port: ["LFM2.5-350M"])
    with caplog.at_level("DEBUG"):
        result = ctl.start("lfm2")
    assert not isinstance(result, Refusal)
    assert secret not in caplog.text
    assert "KITE_API_SECRET" in caplog.text  # the NAME is reported, so it is auditable
    # and the value does not travel in the argv, the result, or anything read back
    assert not any(secret in arg for call in systemd.calls for arg in call)
    assert secret not in repr(result)


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


@pytest.mark.parametrize(
    "setup,why",
    [
        ("timeout", "a boot that never answered"),
        ("crash_loop", "a unit restarting on failure"),
    ],
)
def test_a_start_that_did_not_come_up_stops_the_unit(setup, why, tmp_path) -> None:
    """Restart=on-failure keeps retrying after the caller has been handed a
    failure and moved on: the unit holds the port and takes the GPU on every
    attempt, unattended. That is R4's crash-loop with nobody watching, and it
    follows a TIMEOUT (where the unit is often healthy, just slow) as surely
    as a crash.
    """
    systemd = FakeSystemd()
    if setup == "crash_loop":
        original_add = systemd.add

        def add_crashing(name, **kwargs):
            original_add(name, active_state="activating", sub_state="auto-restart",
                         n_restarts=3)

        systemd.add = add_crashing  # the unit comes up already crash-looping
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path, probe=lambda port: None, lines=[])

    result = ctl.start("lfm2", timeout_s=5.0)
    assert not isinstance(result, Refusal)
    assert not result.ready, why
    assert ["systemctl", "--user", "stop", "model-lfm2"] in systemd.calls
    assert "model-lfm2" not in systemd.units


def test_cleaning_up_a_failed_start_does_not_erase_the_operators_intent(tmp_path) -> None:
    """The cleanup stops the UNIT, not the intent.

    `self.stop()` would also remove the key from desired state — and a start
    invoked by reconcile() is acting on an intent the operator already
    recorded. Erasing it because one boot attempt failed means the model is
    never retried and nothing says why.
    """
    systemd = FakeSystemd()
    desired_mod.save(Desired(residents=["lfm2"]), tmp_path / "desired.json")
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path, probe=lambda port: None, lines=[])

    assert not ctl.start("lfm2", timeout_s=5.0).ready
    assert ["systemctl", "--user", "stop", "model-lfm2"] in systemd.calls
    assert desired_mod.load(tmp_path / "desired.json") == Desired(residents=["lfm2"])


def test_a_successful_start_is_not_stopped(tmp_path) -> None:
    systemd = FakeSystemd()
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path, probe=lambda port: ["LFM2.5-350M"])
    assert ctl.start("lfm2").ready
    assert ["systemctl", "--user", "stop", "model-lfm2"] not in systemd.calls


def test_a_unit_that_never_appeared_says_so_instead_of_blaming_the_model(tmp_path) -> None:
    """"Never appeared" and "vanished after starting" are different faults with
    different fixes — a rejected unit definition versus a model that crashed —
    and both would otherwise arrive as the same "--collect removed it"
    sentence, sending the reader to the journal of a unit that never existed."""
    systemd = FakeSystemd()
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path, probe=lambda port: None, lines=[])
    result = ctl.wait_ready("lfm2", timeout_s=5.0)  # nothing was ever started
    assert not result.ready
    assert "never appeared" in (result.failure or "")
    assert "vanished" not in (result.failure or "")


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
    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path, probe=lambda port: None, lines=[])

    # Present when the wait starts, collected while it is running — a crash
    # partway through the boot, which is what actually happens.
    real_call = systemd.__call__
    reads = []

    def collect_mid_boot(argv):
        argv = list(argv)
        if argv[4:5] == ["LoadState"]:
            reads.append(True)
            if len(reads) > 1:
                systemd.collect("model-lfm2")
        return real_call(argv)

    ctl._run = collect_mid_boot
    result = ctl.wait_ready("lfm2", timeout_s=60)
    assert not result.ready
    assert "vanished after starting" in (result.failure or "")
    # ...and it confirmed before saying so, rather than trusting one read.
    load_state_reads = [c for c in systemd.calls if c[4:5] == ["LoadState"]]
    assert len(load_state_reads) >= 3
    assert result.journal == ["ValueError: unsupported dtype", "engine core failed"]


def test_one_glitched_load_state_read_does_not_kill_a_healthy_boot(tmp_path) -> None:
    """The measured 0.2% `LoadState=not-found` transient, at the level it
    would have done damage: a model that is booting normally must not be
    reported dead because one `systemctl show` hiccupped."""
    systemd = FakeSystemd()
    systemd.add("model-lfm2")
    real_call = systemd.__call__
    reads: list[int] = []
    glitched = []

    def glitchy(argv):
        argv = list(argv)
        if argv[4:5] == ["LoadState"]:
            reads.append(1)
            # The SECOND read, so the glitch lands in the in-loop health check
            # rather than in the "did it appear at all" check before it. Both
            # confirm, but only the loop's verdict ends the boot.
            if len(reads) == 2:
                glitched.append(True)
                systemd.calls.append(argv)
                return subprocess.CompletedProcess(argv, 0, "LoadState=not-found\n", "")
        return real_call(argv)

    ctl = make_control(systemd, FakeRegistry(LFM2), tmp_path, probe=lambda port: ["LFM2.5-350M"])
    ctl._run = glitchy
    result = ctl.wait_ready("lfm2", timeout_s=60)
    assert glitched, "the test did not actually inject the glitch"
    assert result.ready, result.failure


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


def test_switch_never_reports_released_for_a_wait_it_did_not_do(tmp_path) -> None:
    """The silent branch that was here before: with no per-process
    attribution, `_wait_for_release` returned `True, 0.0` immediately. At the
    call site that is indistinguishable from a successful wait — and it fires
    exactly when the instrument is broken. A measurement failing downward into
    a confident yes.

    Now it waits for the plateau: free VRAM stops rising for three consecutive
    reads. Weaker than a target, far stronger than nothing.
    """
    systemd = FakeSystemd()
    systemd.add("model-qwen27b", main_pid=1000)
    ctl = make_control(
        systemd, FakeRegistry(BIG27, FLASH), tmp_path,
        # rising, rising, then flat: the plateau is only declared on the third
        # non-increasing read, so a single flat sample cannot end the wait.
        free=(1344, 1344, 20000, 60000, 91000, 91000, 91000, 91000, 91000),
        used=lambda: None,  # nvidia-smi could not enumerate compute apps
        cgroup_pids=lambda unit: [1000],
        probe=lambda port: ["Qwen3.8-Flash-Next"] if port == 8001 else None,
    )
    result = ctl.switch("flashnext")
    assert not isinstance(result, Refusal)
    assert result.stopped is not None and result.stopped.held_mib is None
    assert result.released and result.waited_s > 0, "it must have actually waited"
    assert result.free_after_mib == 91000
    assert not isinstance(result.started, Refusal) and result.started.ready


def test_the_plateau_wait_does_not_end_on_one_flat_reading(tmp_path) -> None:
    """VRAM comes back in steps with pauses between them. Ending on the first
    non-increasing read would call a pause a plateau and boot into a card that
    is still emptying."""
    systemd = FakeSystemd()
    systemd.add("model-qwen27b", main_pid=1000)
    # Three reads are consumed before the wait loop (switch's opening read,
    # stop's free_before capture, the instrument check), then: a two-read
    # PAUSE at 5000 — long enough to fool a 1-read rule, short of the 3-read
    # rule — before the memory actually finishes coming back at 91000.
    readings = [1000, 1000, 1000, 5000, 5000, 5000, 91000, 91000, 91000, 91000]
    ctl = make_control(
        systemd, FakeRegistry(BIG27, FLASH), tmp_path, free=tuple(readings),
        used=lambda: None, cgroup_pids=lambda unit: [1000],
        probe=lambda port: ["Qwen3.8-Flash-Next"] if port == 8001 else None,
    )
    result = ctl.switch("flashnext")
    assert not isinstance(result, Refusal)
    # Ending on the first flat read would have settled for 5000 MiB and booted
    # a 90 GiB model into a card that was still emptying.
    assert result.free_after_mib == 91000


def test_switch_refuses_when_vram_cannot_be_measured_at_all(tmp_path) -> None:
    """No instrument: not "released", not "not released", a distinct answer —
    because the caller must refuse rather than retry.

    The interesting case is nvidia-smi working when the model is stopped and
    breaking during the wait: there IS a target (90 GiB must come back) and no
    way to see it. Without the check at the top of the wait, that spends the
    full 120 s timeout and then reports `vram_not_released` — blaming the
    driver for a broken instrument, and sending the operator to look at a GPU
    that is probably fine.
    """
    systemd = FakeSystemd()
    systemd.add("model-qwen27b", main_pid=1000)
    ctl = make_control(
        systemd, FakeRegistry(BIG27, FLASH), tmp_path,
        free=(1344, 1344, None),  # readable at stop time, then the tool breaks
        used=lambda: {1000: 90000},
        cgroup_pids=lambda unit: [1000],
        probe=lambda port: ["Qwen3.8-Flash-Next"],
    )
    result = ctl.switch("flashnext")
    assert not isinstance(result, Refusal)
    assert result.stopped is not None and result.stopped.held_mib == 90000
    assert result.released is False
    assert isinstance(result.started, Refusal)
    assert result.started.reason == "vram_accounting_unavailable"
    assert result.waited_s == 0.0, "it must not burn the timeout on a broken instrument"
    assert systemd.started == []


def test_switch_refuses_when_vram_was_never_measurable(tmp_path) -> None:
    systemd = FakeSystemd()
    systemd.add("model-qwen27b", main_pid=1000)
    ctl = make_control(
        systemd, FakeRegistry(BIG27, FLASH), tmp_path,
        probe=lambda port: ["Qwen3.8-Flash-Next"],
        cgroup_pids=lambda unit: [1000],
    )
    ctl._free_mib = lambda: None
    result = ctl.switch("flashnext")
    assert not isinstance(result, Refusal)
    assert result.released is False
    assert isinstance(result.started, Refusal)
    assert result.started.reason == "vram_accounting_unavailable"
    assert systemd.started == []


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
        systemd, FakeRegistry(BIG27, FLASH, LFM2), tmp_path,
        # A RISING sequence. With a constant `free` the release wait can never
        # be satisfied, so the switch bails out before it ever starts anything
        # and the assertion below ("the resident was not stopped") would hold
        # for a switch that did nothing at all.
        free=(4000, 4000, 10000, 50000, 80000, 80000, 80000, 80000),
        used=lambda: {1000: 90000, 999: 3300},
        cgroup_pids=lambda unit: [1000] if unit == "model-qwen27b" else [999],
        probe=lambda port: {8001: ["Qwen3.8-Flash-Next"], 8007: ["LFM2.5-350M"]}.get(port),
    )
    result = ctl.switch("flashnext")
    # The switch must actually have COMPLETED, or "the resident survived" is
    # true of a switch that never ran.
    assert not isinstance(result, Refusal)
    assert result.released and not isinstance(result.started, Refusal)
    assert result.started.ready
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


def test_reconcile_does_not_start_registry_residents_by_default(tmp_path) -> None:
    """Owner directive 2026-09-16: a resident in models.toml is opt-in, not
    always-on. reconcile only ever starts what desired.json names (`want.residents`
    / `want.main`) — it must never walk the registry looking for slot="resident"
    entries to bring up on its own, even when one is defined and would fit."""
    systemd = FakeSystemd()
    ctl = make_control(systemd, FakeRegistry(BIG27, LFM2), tmp_path, free=(94587,),
                       probe=lambda port: None)
    result = ctl.reconcile(Desired())
    assert result.started == [] and result.already_live == [] and result.booting == []
    assert systemd.started == []


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


def test_migration_rewrites_the_v1_file_once_and_keeps_the_original(tmp_path) -> None:
    """Reading must not destroy the rollback, and must not re-warn on every
    poll either (30 warnings a minute on 2026-09-17, because nothing ever
    wrote v2 unless an operator action changed the set). The first read
    rewrites desired.json as schema 2 and keeps the verbatim v1 body beside
    it as desired.json.v1, which is what the cutover's rollback restores."""
    path = tmp_path / "desired.json"
    body = json.dumps({
        "version": 1, "desired_state": "RUNNING", "backend": "inline", "util": 0.91,
    })
    path.write_text(body)
    assert desired_mod.load(path) == Desired(main="inline", residents=[])
    assert json.loads(path.read_text())["version"] == 2
    assert (tmp_path / "desired.json.v1").read_text() == body
    assert desired_mod.load(path) == Desired(main="inline", residents=[])
    assert (tmp_path / "desired.json.v1").read_text() == body, "a second read leaves the v1 copy alone"


@pytest.mark.parametrize("body", ["", "not json", "[]", '{"version": 99}', '{"version": 2}'])
def test_an_unreadable_desired_file_degrades_to_wanting_nothing(tmp_path, body) -> None:
    """Never raise out of here: failing to parse this file must not be the
    reason servedeck will not start."""
    path = tmp_path / "desired.json"
    path.write_text(body)
    assert desired_mod.load(path) == Desired()


def test_a_missing_desired_file_is_not_an_error(tmp_path) -> None:
    assert desired_mod.load(tmp_path / "nothing.json") == Desired()


def test_fresh_desired_has_no_residents(tmp_path) -> None:
    """Owner directive 2026-09-16: residents are opt-in, never on by default.
    A fresh install (no desired.json on disk at all) must want nothing running
    — in particular no resident — regardless of how many `slot = "resident"`
    entries live in models.toml. Nothing about this file's shape lets a
    registry entry opt itself in."""
    assert desired_mod.load(tmp_path / "desired.json") == Desired(main=None, residents=[])
    assert desired_mod.load(tmp_path / "desired.json").residents == []


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

    # Exactly two calls are allowed to name no unit, both read-only and both
    # manager-wide by necessity: discovery (scoped to the model-* glob) and
    # reading the manager's environment to decide what to redact. Any THIRD
    # unit-less call is a new way for this module to reach outside its
    # namespace, and this test is where that has to be argued for.
    manager_wide = (
        ["systemctl", "--user", "list-units", "model-*"],
        ["systemctl", "--user", "show-environment"],
    )
    for call in systemd.calls:
        named = [a for a in call if a.startswith("model-") or a.startswith("--unit=model-")]
        if any(call[: len(allowed)] == allowed for allowed in manager_wide):
            continue
        assert named, f"a call that names no model unit: {call}"
        for name in named:
            assert units.valid_unit_name(name.removeprefix("--unit="))
    assert "qwen-vllm" in systemd.units, "an untouched unit must stay untouched"


# --------------------------------------------------------------------------
# Adoption of an unmanaged listener (2026-09-17)
# --------------------------------------------------------------------------


def _probe_for(port_ids: dict[int, list[str]]):
    return lambda port: port_ids.get(port)


def test_live_adopts_a_registered_model_serving_without_a_unit(tmp_path) -> None:
    """Cutover: Flash-Next keeps serving on :8001 from v1's launch while v2
    takes over. No `model-flashnext` unit exists, but the port answers with
    the registered id, so it is live, ready, and routable — as adopted."""
    systemd = FakeSystemd()
    c = make_control(systemd, FakeRegistry(FLASH, BIG27), tmp_path,
                     probe=_probe_for({8001: ["Qwen3.8-Flash-Next"]}))
    c._listener_pid = lambda port: 27110 if port == 8001 else None
    rows = c.live()
    assert [r.key for r in rows] == ["flashnext"]
    row = rows[0]
    assert row.adopted and row.ready and row.pid == 27110 and row.port == 8001
    assert row.unit == control.ADOPTED_UNIT and row.sub_state == "adopted"
    assert c.live_main() == row, "an adopted main-slot model holds the main slot"


def test_a_foreign_listener_on_a_registered_port_is_not_adopted(tmp_path) -> None:
    """:8002 held by an unrelated web app (seen 2026-09-17) answers /v1/models
    with nothing of ours, or not at all. Not a model; doctor's business."""
    c = make_control(FakeSystemd(), FakeRegistry(FLASH), tmp_path,
                     probe=_probe_for({8001: ["something-else"]}))
    assert c.live() == []


def test_a_unit_owned_model_is_never_double_listed_as_adopted(tmp_path) -> None:
    systemd = FakeSystemd()
    systemd.add("model-flashnext", main_pid=555)
    c = make_control(systemd, FakeRegistry(FLASH), tmp_path,
                     probe=_probe_for({8001: ["Qwen3.8-Flash-Next"]}))
    rows = c.live()
    assert len(rows) == 1 and not rows[0].adopted and rows[0].pid == 555


def test_start_refuses_while_an_adopted_model_holds_the_main_slot(tmp_path) -> None:
    c = make_control(FakeSystemd(), FakeRegistry(FLASH, BIG27), tmp_path,
                     probe=_probe_for({8001: ["Qwen3.8-Flash-Next"]}))
    c._listener_pid = lambda port: 27110
    got = c.start("qwen27b")
    assert isinstance(got, control.Refusal) and got.reason == "main_slot_busy"
    assert got.live_key == "flashnext"


def test_stop_of_an_adopted_model_signals_its_pid_and_waits_for_the_port(tmp_path) -> None:
    """No unit to `systemctl stop`: SIGTERM the listener, wait until the port
    stops answering, account the VRAM of the pid tree (the API server holds
    nothing; its engine children hold it all)."""
    alive = {"up": True}
    kills: list[tuple[int, int]] = []

    def kill(pid, sig):
        kills.append((pid, sig))
        if pid == 27110:
            alive["up"] = False

    c = make_control(
        FakeSystemd(), FakeRegistry(FLASH), tmp_path,
        probe=lambda port: ["Qwen3.8-Flash-Next"] if alive["up"] else None,
        used=lambda: {27110: 0, 27200: 81000, 27201: 900},
    )
    c._listener_pid = lambda port: 27110 if alive["up"] else None
    c._kill = kill
    c._descendants = lambda pid: [27200, 27201] if pid == 27110 else []
    c.adopt()
    assert c.load_desired().main == "flashnext"
    got = c.stop("flashnext")
    assert isinstance(got, control.StopResult)
    assert got.was_live and got.unit == control.ADOPTED_UNIT
    assert got.held_mib == 81900
    assert kills == [(27110, 15)], "one SIGTERM to the listener, no SIGKILL when it exits"
    assert c.load_desired().main is None
    assert c.live() == []


def test_stop_of_an_adopted_model_escalates_to_sigkill_on_timeout(tmp_path) -> None:
    kills: list[tuple[int, int]] = []
    c = make_control(FakeSystemd(), FakeRegistry(FLASH), tmp_path,
                     probe=lambda port: ["Qwen3.8-Flash-Next"])
    c._listener_pid = lambda port: 27110
    c._kill = lambda pid, sig: kills.append((pid, sig))
    c._descendants = lambda pid: [27200]
    got = c.stop("flashnext", timeout_s=1.0)
    assert isinstance(got, control.StopResult)
    assert kills[0] == (27110, 15)
    assert (27200, 9) in kills and (27110, 9) in kills, "the whole tree gets SIGKILL after the deadline"


def test_reconcile_leaves_an_adopted_desired_main_alone(tmp_path) -> None:
    """Desired says flashnext; an adopted process already serves it. Starting a
    second copy would boot 80 GiB into an occupied card."""
    systemd = FakeSystemd()
    c = make_control(systemd, FakeRegistry(FLASH), tmp_path,
                     probe=_probe_for({8001: ["Qwen3.8-Flash-Next"]}))
    c._listener_pid = lambda port: 27110
    c._save_desired(c.load_desired().with_main("flashnext"))
    rec = c.reconcile()
    assert rec.already_live == ["flashnext"] and rec.started == [] and systemd.started == []
