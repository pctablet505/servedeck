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
        "Restart=on-failure",
        "-p",
        "RestartSec=10",
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
