"""units.py: the exact argv, and the names it refuses.

Every assertion here is on an argument LIST, never on a formatted string.
That is the point of the module: a model id, a description or an env value can
never be re-parsed as shell, so there is no quoting to get right and no
injection to defend against. A test that matched a joined command line would
pass for a `shell=True` implementation too, and would therefore be testing
nothing that matters.
"""

from __future__ import annotations

import subprocess

import pytest

from servedeck import units


class FakeRunner:
    """Records every argv and replays canned results.

    ``results`` is consumed in order; when it runs out, a 0/empty result is
    returned, so a test only has to script the calls it cares about.
    """

    def __init__(self, *results: subprocess.CompletedProcess[str]) -> None:
        self.calls: list[list[str]] = []
        self._results = list(results)

    def __call__(self, argv):
        self.calls.append(list(argv))
        if self._results:
            return self._results.pop(0)
        return subprocess.CompletedProcess(list(argv), 0, "", "")

    @property
    def last(self) -> list[str]:
        return self.calls[-1]


def ok(stdout: str = "", argv=("x",)) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(list(argv), 0, stdout, "")


def fail(code: int = 1, stderr: str = "boom", stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["x"], code, stdout, stderr)


# --------------------------------------------------------------------------
# Name validation — the blast-radius guard
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["model-lfm2", "model-flashnext", "model-glm53-flash", "model-27b", "sd-test-stub", "sd-test-lfm2"],
)
def test_accepts_model_and_test_names(name: str) -> None:
    assert units.valid_unit_name(name)


@pytest.mark.parametrize(
    "name",
    [
        # The live units on this box. Every one of these must be unreachable
        # through this module, by name, before any process is spawned.
        "servedeck",
        "qwen-vllm",
        "lfm2-350m",
        "flashnext-reasoning-proxy",
        "coldstart",
        "ats-optimizer",
        # Shapes that would smuggle one of the above past the prefix check.
        "model-LFM2",  # uppercase
        "model-",  # empty key
        "Model-lfm2",
        "model lfm2",
        "model-lfm2.service",
        "model-lfm2/../servedeck",
        "sd-test",
        "../model-x",
        "",
        "*",
        "model-*",
    ],
)
def test_refuses_every_other_name(name: str) -> None:
    assert not units.valid_unit_name(name)
    runner = FakeRunner()
    for call in (
        lambda: units.start_transient(name, ["/bin/true"], {}, "/tmp", run=runner),
        lambda: units.stop(name, run=runner),
        lambda: units.show(name, run=runner),
        lambda: units.journal_tail(name, 10, run=runner),
        lambda: units.journal_follow(name, "now"),
    ):
        with pytest.raises(units.UnitError):
            call()
    assert runner.calls == [], "a refused name must never reach a subprocess"


# --------------------------------------------------------------------------
# start_transient
# --------------------------------------------------------------------------


def test_start_transient_argv_is_exact() -> None:
    runner = FakeRunner()
    units.start_transient(
        "model-lfm2",
        ["/opt/venv/bin/vllm", "serve", "LiquidAI/LFM2.5-350M", "--port", "8007"],
        {"VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_HOME": "/home/u/.cache/huggingface"},
        cwd="/opt/venv",
        restart="on-failure",
        restart_sec=10,
        description="servedeck model LFM2.5-350M (lfm2)",
        run=runner,
    )
    assert runner.last == [
        "systemd-run",
        "--user",
        "--unit=model-lfm2",
        "--collect",
        "--description=servedeck model LFM2.5-350M (lfm2)",
        "-p",
        # this test passes restart= explicitly: the override wins
        "Restart=on-failure",
        "-p",
        "RestartSec=10",
        # The crash-loop ceiling. systemd's defaults (10s window, 5 starts)
        # can NEVER fire with RestartSec=10, because each retry lands outside
        # the window — so a model that cannot boot restarts every ten seconds
        # forever, unattended, taking the GPU each time.
        "-p",
        "StartLimitIntervalSec=300",
        "-p",
        "StartLimitBurst=3",
        # Long enough for vLLM's --shutdown-timeout drain plus the unwind of
        # pinned host memory; systemd killing the unit first is what orphaned
        # 40 GiB of /dev/shm on every restart.
        "-p",
        "TimeoutStopSec=120",
        "-p",
        "WorkingDirectory=/opt/venv",
        # sorted by key, so the argv is deterministic and assertable
        "--setenv=HF_HOME=/home/u/.cache/huggingface",
        "--setenv=VLLM_USE_FLASHINFER_SAMPLER=0",
        "--",
        "/opt/venv/bin/vllm",
        "serve",
        "LiquidAI/LFM2.5-350M",
        "--port",
        "8007",
    ]


def test_the_start_limit_can_actually_fire_with_the_restart_delay_we_use() -> None:
    """The defaults are not merely weak here, they are unreachable.

    systemd's defaults are 5 starts per 10 s. With RestartSec=10 the second
    attempt lands at t=10s, outside the window, which resets the counter — so
    the burst is never exceeded and the unit restarts forever. The property
    that matters is arithmetic: the interval must be long enough to contain
    `burst` restarts spaced `restart_sec` apart.
    """
    runner = FakeRunner()
    units.start_transient("model-x", ["/bin/true"], {}, "/tmp", run=runner)

    def value_of(prefix: str) -> int:
        (entry,) = [a for a in runner.last if a.startswith(prefix + "=")]
        return int(entry.split("=", 1)[1])

    interval = value_of("StartLimitIntervalSec")
    burst = value_of("StartLimitBurst")
    restart_sec = value_of("RestartSec")
    assert interval > burst * restart_sec, (
        f"{burst} restarts {restart_sec}s apart span {burst * restart_sec}s, which must "
        f"fit inside StartLimitIntervalSec={interval} or the limit can never fire"
    )


def test_start_transient_separates_options_from_command() -> None:
    """`--` must precede the command, or a model flag becomes a systemd-run flag.

    Without it, `systemd-run --user --unit=... --version` would print
    systemd-run's version and exit 0: a "successful start" that started
    nothing. The separator is what makes argv[0] onwards unambiguously the
    command.
    """
    runner = FakeRunner()
    units.start_transient("model-x", ["/bin/echo", "--version"], {}, "/tmp", run=runner)
    argv = runner.last
    assert "--" in argv
    assert argv[argv.index("--") + 1 :] == ["/bin/echo", "--version"]


def test_start_transient_defaults_description_to_the_unit_name() -> None:
    runner = FakeRunner()
    units.start_transient("sd-test-stub", ["/bin/true"], {}, "/tmp", run=runner)
    assert "--description=sd-test-stub" in runner.last


def test_unset_env_becomes_one_unset_environment_property_per_name() -> None:
    """The redaction reaches systemd as unit properties, in a fixed place.

    Measured on this box (systemd 259): with `-p UnsetEnvironment=X` the
    variable is absent from the unit's /proc/<MainPID>/environ and from what
    /usr/bin/env logs to the journal, while HOME survives. Without it, it
    leaks. tests/test_control_e2e.py proves both halves live.
    """
    runner = FakeRunner()
    units.start_transient(
        "model-lfm2",
        ["/opt/venv/bin/vllm", "serve", "m"],
        {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
        cwd="/opt/venv",
        unset_env=["KITE_API_SECRET", "KITE_API_KEY"],
        run=runner,
    )
    assert runner.last == [
        "systemd-run",
        "--user",
        "--unit=model-lfm2",
        "--collect",
        "--description=model-lfm2",
        "-p",
        # always, not on-failure: vLLM exits 0 when its engine core dies
        # (launcher.py's watchdog returns from serve_http), so on-failure
        # never fires and a dead model stays dead. See units.DEFAULT_RESTART.
        "Restart=always",
        "-p",
        "RestartSec=10",
        # The crash-loop ceiling. systemd's defaults (10s window, 5 starts)
        # can NEVER fire with RestartSec=10, because each retry lands outside
        # the window — so a model that cannot boot restarts every ten seconds
        # forever, unattended, taking the GPU each time.
        "-p",
        "StartLimitIntervalSec=300",
        "-p",
        "StartLimitBurst=3",
        # Long enough for vLLM's --shutdown-timeout drain plus the unwind of
        # pinned host memory; systemd killing the unit first is what orphaned
        # 40 GiB of /dev/shm on every restart.
        "-p",
        "TimeoutStopSec=120",
        "-p",
        "WorkingDirectory=/opt/venv",
        # sorted, and before --setenv, so the argv is deterministic
        "-p",
        "UnsetEnvironment=KITE_API_KEY",
        "-p",
        "UnsetEnvironment=KITE_API_SECRET",
        "--setenv=VLLM_USE_FLASHINFER_SAMPLER=0",
        "--",
        "/opt/venv/bin/vllm",
        "serve",
        "m",
    ]


def test_unset_env_is_absent_from_the_argv_when_there_is_nothing_to_unset() -> None:
    runner = FakeRunner()
    units.start_transient("model-x", ["/bin/true"], {}, "/tmp", run=runner)
    assert not any(a.startswith("UnsetEnvironment=") for a in runner.last)


def test_unset_env_deduplicates() -> None:
    runner = FakeRunner()
    units.start_transient("model-x", ["/bin/true"], {}, "/tmp",
                          unset_env=["A_KEY", "A_KEY"], run=runner)
    assert runner.last.count("UnsetEnvironment=A_KEY") == 1


@pytest.mark.parametrize("name", ["", "2FA_TOKEN", "A-KEY", "A KEY", "A=B", "a.key", "$KEY"])
def test_unset_env_refuses_a_name_that_is_not_an_environment_variable(name: str) -> None:
    """systemd rejects the whole unit for a malformed property value, so a
    typo in a redaction list would become a model that will not boot, with the
    reason buried in journald. Refuse it here, where the message is about the
    typo."""
    runner = FakeRunner()
    with pytest.raises(units.UnitError, match="not a usable environment variable name"):
        units.start_transient("model-x", ["/bin/true"], {}, "/tmp", unset_env=[name], run=runner)
    assert runner.calls == []


def test_a_name_in_both_env_and_unset_env_is_refused() -> None:
    """systemd applies UnsetEnvironment= LAST, after Environment= and
    --setenv, so the unit would start without a variable the caller explicitly
    set and nothing would report it. Silent is the one outcome not allowed."""
    runner = FakeRunner()
    with pytest.raises(units.UnitError, match="appear in both env and unset_env"):
        units.start_transient("model-x", ["/bin/true"], {"HF_TOKEN": "hunter2"}, "/tmp",
                              unset_env=["HF_TOKEN"], run=runner)
    assert runner.calls == []


def test_manager_environment_names_returns_names_and_never_values() -> None:
    """The value half is discarded inside the function, so there is nothing
    for a caller, a log line or an exception message to leak."""
    runner = FakeRunner(ok("HOME=/home/u\nKITE_API_SECRET=v41zj8cxazl09fz8gniaffo3xeqhqmlh\n"
                           "PATH=/usr/bin\n"))
    names = units.manager_environment_names(run=runner)
    assert names == ["HOME", "KITE_API_SECRET", "PATH"]
    assert runner.last == ["systemctl", "--user", "show-environment"]
    assert not any("v41zj8" in name for name in names)


def test_manager_environment_names_raises_rather_than_returning_empty() -> None:
    """An empty list reads as 'nothing to redact'. If the manager cannot be
    asked, that answer would hand every ambient secret to the model — a broken
    measurement failing downward. Raise instead."""
    with pytest.raises(units.UnitError):
        units.manager_environment_names(run=FakeRunner(fail(1, "Failed to connect to bus.")))


def test_start_transient_refuses_a_bogus_restart_policy() -> None:
    runner = FakeRunner()
    with pytest.raises(units.UnitError, match="not a systemd restart policy"):
        units.start_transient("model-x", ["/bin/true"], {}, "/tmp", restart="on-fail", run=runner)
    assert runner.calls == []


def test_start_transient_refuses_an_empty_command() -> None:
    runner = FakeRunner()
    with pytest.raises(units.UnitError, match="empty command"):
        units.start_transient("model-x", [], {}, "/tmp", run=runner)
    assert runner.calls == []


def test_start_transient_raises_when_systemd_run_fails() -> None:
    runner = FakeRunner(fail(1, "Unit model-x.service already exists."))
    with pytest.raises(units.UnitError, match="already exists"):
        units.start_transient("model-x", ["/bin/true"], {}, "/tmp", run=runner)


# --------------------------------------------------------------------------
# stop / show
# --------------------------------------------------------------------------


def test_stop_argv() -> None:
    runner = FakeRunner()
    units.stop("model-lfm2", run=runner)
    assert runner.last == ["systemctl", "--user", "stop", "model-lfm2"]


def test_stop_of_an_already_collected_unit_is_not_an_error() -> None:
    """`--collect` deletes a unit that failed, so `stop` after a crash hits a
    unit that is not loaded. That is the normal path, not an error."""
    runner = FakeRunner(fail(5, "Failed to stop model-x.service: Unit model-x.service not loaded."))
    units.stop("model-x", run=runner)  # must not raise


def test_stop_still_raises_on_a_real_failure() -> None:
    runner = FakeRunner(fail(1, "Interactive authentication required."))
    with pytest.raises(units.UnitError, match="Interactive authentication"):
        units.stop("model-x", run=runner)


SHOW_OUTPUT = """ActiveState=active
SubState=running
Result=success
NRestarts=0
MainPID=278740
ExecMainStartTimestamp=Fri 2026-09-12 13:20:01 IST
"""


def test_show_argv_and_parse() -> None:
    runner = FakeRunner(ok(SHOW_OUTPUT))
    state = units.show("model-lfm2", run=runner)
    assert runner.last == [
        "systemctl",
        "--user",
        "show",
        "-p",
        "ActiveState,SubState,Result,NRestarts,MainPID,ExecMainStartTimestamp",
        "model-lfm2",
    ]
    assert state == units.UnitState(
        active_state="active",
        sub_state="running",
        result="success",
        n_restarts=0,
        main_pid=278740,
        exec_main_start_ts="Fri 2026-09-12 13:20:01 IST",
    )
    assert state.active and not state.failed


def test_show_tolerates_empty_and_non_numeric_fields() -> None:
    runner = FakeRunner(ok("ActiveState=inactive\nSubState=dead\nResult=success\n"
                           "NRestarts=\nMainPID=\nExecMainStartTimestamp=\n"))
    state = units.show("model-x", run=runner)
    assert (state.n_restarts, state.main_pid, state.exec_main_start_ts) == (0, 0, "")


def test_failed_is_true_for_a_loaded_failed_unit() -> None:
    runner = FakeRunner(ok("ActiveState=failed\nSubState=failed\nResult=exit-code\n"
                           "NRestarts=3\nMainPID=0\nExecMainStartTimestamp=\n"))
    assert units.show("model-x", run=runner).failed


def test_a_collected_unit_looks_healthy_to_show_and_exists_is_what_catches_it() -> None:
    """The measured `--collect` trap, pinned.

    On this box (systemd 259) `systemctl --user show` exits 0 for a unit it
    has never heard of and prints property DEFAULTS. A crashed, collected unit
    therefore reads `ActiveState=inactive Result=success` — indistinguishable
    from a clean stop, and `failed` is False. If this test ever starts failing
    because `failed` became True, the module has begun guessing. The only
    honest discriminator is LoadState.
    """
    collected = ("ActiveState=inactive\nSubState=dead\nResult=success\n"
                 "NRestarts=0\nMainPID=0\nExecMainStartTimestamp=\n")
    runner = FakeRunner(ok(collected), ok("LoadState=not-found\n"))
    state = units.show("model-gone", run=runner)
    assert state.result == "success" and not state.failed  # the lie
    assert units.exists("model-gone", run=runner) is False  # the truth
    assert runner.last == ["systemctl", "--user", "show", "-p", "LoadState", "model-gone"]


def test_exists_is_true_only_for_loaded() -> None:
    assert units.exists("model-x", run=FakeRunner(ok("LoadState=loaded\n"))) is True
    assert units.exists("model-x", run=FakeRunner(ok("LoadState=masked\n"))) is False


def test_gone_does_not_believe_a_single_not_found(monkeypatch) -> None:
    """MEASURED on this box (systemd 259, 2026-09-12): sampling a live,
    running transient unit every 20 ms, `systemctl --user show -p LoadState`
    answered `LoadState=not-found` for 2 of ~1030 reads — exit 0, empty
    stderr, unit healthy throughout.

    0.2% is not small where it is used: wait_ready checks every 2 s, so a
    five-minute boot is ~150 reads and a single-read verdict would declare a
    healthy 90 GiB model dead about a quarter of the time. This is exactly
    that transient, replayed.
    """
    runner = FakeRunner(
        ok("LoadState=not-found\n"),  # the glitch
        ok("LoadState=loaded\n"),  # the truth
    )
    slept: list[float] = []
    assert units.gone("model-x", run=runner, sleep=slept.append) is False
    assert len(runner.calls) == 2
    assert slept == [0.3]


def test_gone_is_true_when_every_attempt_agrees() -> None:
    runner = FakeRunner(*[ok("LoadState=not-found\n")] * 3)
    assert units.gone("model-x", run=runner, sleep=lambda _s: None) is True
    assert len(runner.calls) == 3


def test_gone_short_circuits_on_a_live_unit() -> None:
    """A healthy unit must cost one read and no sleep — the confirmation
    delay is paid only by a unit that really is absent, where nobody is
    waiting."""
    runner = FakeRunner(ok("LoadState=loaded\n"))
    slept: list[float] = []
    assert units.gone("model-x", run=runner, sleep=slept.append) is False
    assert len(runner.calls) == 1 and slept == []


def test_control_group_argv_and_parse() -> None:
    runner = FakeRunner(ok("ControlGroup=/user.slice/user-1000.slice/user@1000.service/"
                           "app.slice/model-lfm2.service\n"))
    assert units.control_group("model-lfm2", run=runner).endswith("/model-lfm2.service")
    assert runner.last[:5] == ["systemctl", "--user", "show", "-p", "ControlGroup"]


# --------------------------------------------------------------------------
# list_model_units
# --------------------------------------------------------------------------


LIST_OUTPUT = (
    "model-lfm2.service     loaded active   running servedeck model LFM2.5-350M (lfm2)\n"
    "model-flashnext.service loaded activating start   servedeck model Qwen3.8 (flashnext)\n"
)


def test_list_model_units_argv_and_parse() -> None:
    runner = FakeRunner(ok(LIST_OUTPUT))
    assert units.list_model_units(run=runner) == ["model-lfm2", "model-flashnext"]
    assert runner.last == [
        "systemctl",
        "--user",
        "list-units",
        "model-*",
        "--all",
        "--plain",
        "--no-legend",
    ]


def test_list_model_units_never_returns_a_unit_it_could_not_act_on() -> None:
    """Discovery and action share one name predicate.

    A row that `valid_unit_name` would refuse is dropped here rather than
    handed to a caller that will later try to `stop` it and get an exception
    from deep inside. (`model-*` cannot match `servedeck.service`, but it can
    match `model-Foo.service` if someone hand-creates one.)
    """
    runner = FakeRunner(ok("model-Foo.service loaded active running x\n"
                           "model-lfm2.service loaded active running y\n"))
    assert units.list_model_units(run=runner) == ["model-lfm2"]


def test_list_model_units_is_empty_when_the_glob_matches_nothing() -> None:
    """systemd exits nonzero for a glob with no matches; that is not an error."""
    assert units.list_model_units(run=FakeRunner(fail(1, "", ""))) == []


# --------------------------------------------------------------------------
# journal
# --------------------------------------------------------------------------


def test_journal_tail_argv() -> None:
    runner = FakeRunner(ok("line one\nline two\n"))
    assert units.journal_tail("model-lfm2", 40, run=runner) == ["line one", "line two"]
    assert runner.last == [
        "journalctl",
        "--user",
        "-u",
        "model-lfm2",
        "--no-pager",
        "-n",
        "40",
        "-o",
        "cat",
    ]


def test_journal_tail_is_empty_not_fatal_when_journalctl_fails() -> None:
    """A failure report must not itself fail. journal_tail is called on the
    error path; raising there would replace the diagnosis with a traceback."""
    assert units.journal_tail("model-x", 40, run=FakeRunner(fail(1, "no journal"))) == []


def test_journal_follow_argv() -> None:
    seen: list[list[str]] = []

    class FakeProc:
        stdout = None

        def poll(self):
            return 0

    units.journal_follow("model-lfm2", "2026-09-12 13:00:00",
                         spawn=lambda argv: (seen.append(list(argv)), FakeProc())[1])
    assert seen[0] == [
        "journalctl",
        "--user",
        "-u",
        "model-lfm2",
        "--no-pager",
        "-o",
        "cat",
        "--since",
        "2026-09-12 13:00:00",
        "-f",
    ]


def test_journal_stream_reassembles_lines_split_across_reads() -> None:
    """journalctl writes into a pipe; a marker can straddle two reads.

    "Application startup complete." arriving as two chunks must still be one
    line, or readiness detection depends on kernel pipe timing.
    """
    import os

    read_fd, write_fd = os.pipe()

    class FakeProc:
        stdout = os.fdopen(read_fd, "rb")

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    stream = units.JournalStream(FakeProc())
    os.write(write_fd, b"Loading weights to")
    assert stream.poll_lines(0.1) == []  # partial line is held back
    os.write(write_fd, b"ok 0.5 seconds\nGPU KV cache size: 1,024 tokens\n")
    assert stream.poll_lines(0.1) == [
        "Loading weights took 0.5 seconds",
        "GPU KV cache size: 1,024 tokens",
    ]
    os.close(write_fd)
    assert stream.poll_lines(0.1) == []
    assert stream.eof is True
