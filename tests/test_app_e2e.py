"""The whole surface, end to end, against a REAL systemd user manager.

One model, started the way an operator starts one:

    POST /api/models/stub/start
      -> Control -> systemd-run --user --unit=sd-test-stub
      -> models.toml's [builds] venv -> bin/vllm -> the stub server
      -> the journal markers arrive on /api/events
      -> GET /v1/models through the GATEWAY lists it
      -> a chat completion through the GATEWAY reaches it
    POST /api/models/stub/stop
      -> the unit is gone

Nothing is faked except ``nvidia-smi``, and only that, for the reason P3's own
e2e gives: the stub holds no VRAM, the card's real state has nothing to do with
what this proves, and leaving it real would SKIP the path that has no coverage
every time the box happens to be busy — which, with a 90 GiB model serving, is
always.

The registry is a real ``models.toml``. The shim at ``<build>/bin/vllm`` is
what makes that possible: ``models.render_argv`` renders a genuine ``vllm serve
… --served-model-name … --port …`` line, and the shim reads its own arguments
back out. So this exercises the argv contract too, not only the lifecycle.

Blast radius. Every unit is named ``sd-test-*``; ``units.py`` refuses any other
shape before spawning anything, and ``Control`` is built with
``unit_prefix="sd-test-"`` so even its discovery glob cannot see a ``model-*``
unit. The only ports bound are 8040 (the app) and 8042 (the stub), both on
loopback and both inside the range this packet owns. The fixture stops every
``sd-test-*`` unit on the way out whether the tests passed, failed or raised.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import stat
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from servedeck import __main__ as _entry
from servedeck import app as _app
from servedeck import control as _control
from servedeck import models as _models
from servedeck import units as _units
from servedeck import wire as _wire
from servedeck.settings import Settings

pytestmark = pytest.mark.skipif(
    shutil.which("systemd-run") is None or not os.environ.get("XDG_RUNTIME_DIR"),
    reason="needs a systemd user manager (systemd-run + XDG_RUNTIME_DIR)",
)

ROOT = Path(__file__).resolve().parent.parent
STUB = Path(__file__).resolve().parent / "stub_openai.py"

APP_PORT = 8040
STUB_PORT = 8042
STUB_UNIT = "sd-test-stub"
STUB_ID = "sd-test-stub-model"
BASE = f"http://127.0.0.1:{APP_PORT}"

MODELS_TOML = f"""
[gpu]
total_mib = 97887
margin_mib = 1024

[builds.stub]
venv = "{{build_root}}"
cuda_home = "{{build_root}}/cuda"

[models.stub]
id = "{STUB_ID}"
aliases = ["stubby"]
repo = "org/stub"
slot = "resident"
vram_mib = 2000
port = {STUB_PORT}
build = "stub"
ctx = 4096
"""

#: A ``vllm`` that is not vLLM. It reads the two arguments it needs out of the
#: argv ``models.render_argv`` produced -- which is the point: a shim that took
#: its port from an environment variable would pass even if the renderer
#: stopped emitting ``--port``.
SHIM = f'''#!{sys.executable}
import subprocess, sys
argv = sys.argv[1:]
assert argv[0] == "serve", argv
port = argv[argv.index("--port") + 1]
name = argv[argv.index("--served-model-name") + 1]
assert "--max-model-len" in argv, argv
assert "--gpu-memory-utilization" in argv, argv
sys.exit(subprocess.call([
    sys.executable, "{STUB}", "--port", port, "--model", name,
    "--delay-ready", "3", "--emit-boot-lines", "--ready-grace", "1",
]))
'''


def stop_every_sd_test_unit() -> list[str]:
    """Stop every ``sd-test-*`` unit, by name, one at a time.

    By name and not by pattern: ``systemctl stop 'sd-test-*'`` would be a glob
    handed to a privileged tool. Discovery is the same ``list-units`` call
    production uses, so this can only ever name units it could have created.
    """
    stopped = []
    for unit in _units.list_units("sd-test-*"):
        try:
            _units.stop(unit, timeout_s=60)
        except _units.UnitError:  # pragma: no cover - best-effort teardown
            pass
        stopped.append(unit)
    return stopped


@pytest.fixture(scope="module", autouse=True)
def _no_units_left_behind():
    stop_every_sd_test_unit()
    try:
        yield
    finally:
        stop_every_sd_test_unit()
        leftover = _units.list_units("sd-test-*")
        assert leftover == [], f"the suite leaked transient units: {leftover}"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    build_root = tmp / "build"
    (build_root / "bin").mkdir(parents=True)
    shim = build_root / "bin" / "vllm"
    shim.write_text(SHIM)
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    models_path = tmp / "models.toml"
    models_path.write_text(MODELS_TOML.format(build_root=build_root))
    return tmp, models_path


@pytest.fixture(scope="module")
def server(workspace):
    tmp, models_path = workspace
    settings = Settings(
        listen_host="127.0.0.1",
        listen_port=APP_PORT,
        models_path=models_path,
        state_dir=tmp / "state",
        unit_prefix="sd-test-",
    )
    registry = _models.load(models_path)
    # The real Control, the real units module, the real journal -- with
    # nvidia-smi injected. `total_mib`/`free_mib` are constructor parameters of
    # Control precisely so this substitution needs no patching.
    control = _control.Control(
        _app._RegistryAdapter(registry),
        margin_mib=registry.gpu.margin_mib,
        total_mib=registry.gpu.total_mib,
        unit_prefix="sd-test-",
        desired_path=settings.desired_path,
        free_mib=lambda: registry.gpu.total_mib,
    )
    original_targets = _wire.WIRE_TARGETS
    _app._wire.WIRE_TARGETS = ()  # never touch ~/.config from a test

    app = _app.create_app(
        settings, registry=registry, control=control, reconcile=False, poll=True
    )
    # The production server class, so this e2e also covers the stop path an
    # operator actually triggers (SIGTERM -> handle_exit -> hub closed before
    # uvicorn starts draining). A plain uvicorn.Server here would pass the
    # lifecycle assertions and tell us nothing about the shutdown.
    config = uvicorn.Config(
        app, host="127.0.0.1", port=APP_PORT, log_level="error",
        timeout_graceful_shutdown=_entry.GRACEFUL_SHUTDOWN_S,
    )
    uv = _entry._ShutdownClosesTheHub(config, app.state.rt.hub)
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{BASE}/api/health", timeout=1.0).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.05)
    else:  # pragma: no cover
        uv.should_exit = True
        pytest.fail(f"the app never answered on {BASE}")

    try:
        yield settings
    finally:
        _app._wire.WIRE_TARGETS = original_targets
        started = time.monotonic()
        uv.handle_exit(15, None)  # SIGTERM, exactly as systemd sends it
        thread.join(timeout=10)
        elapsed = time.monotonic() - started
        assert not thread.is_alive(), "uvicorn did not exit"
        # The TimeoutStopSec fix, measured. The old app hung on every stop
        # because each open SSE generator was parked on a queue nothing would
        # fill again; systemd SIGKILLed the unit 15 s later.
        assert elapsed < 5.0, f"shutdown took {elapsed:.1f}s with an SSE stream open"


@pytest.fixture(autouse=True)
def _stub_is_stopped_between_tests(server):
    """Leave the stub stopped, before AND after every test.

    Without this a test that fails mid-lifecycle leaves the unit running, and
    every test after it reports 409 — so one real defect is reported as five,
    four of them at the wrong place.
    """
    yield
    if _units.exists(STUB_UNIT):
        with contextlib.suppress(Exception):
            httpx.post(f"{BASE}/api/models/stub/stop", timeout=10.0)
        wait_until(lambda: not _units.exists(STUB_UNIT), 60)
    wait_until(lambda: state().get("busy") is None, 30)


def wait_until(predicate, timeout_s: float = 60.0, interval_s: float = 0.2):
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval_s)
    return last


def state() -> dict:
    return httpx.get(f"{BASE}/api/state", timeout=10.0).json()


def stub_row(doc: dict) -> dict:
    return next(m for m in doc["models"] if m["key"] == "stub")


class EventCollector:
    """Reads ``/api/events`` in a thread for the duration of a test.

    A thread rather than a poll of some accumulated list, because what is being
    proved is that the frames ARRIVE as things happen — an SSE stream that only
    ever delivered on disconnect would pass any after-the-fact assertion.
    """

    def __init__(self) -> None:
        self.frames: list[tuple[str, dict]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            with httpx.Client(timeout=httpx.Timeout(5.0, read=None)) as client:
                with client.stream("GET", f"{BASE}/api/events") as stream:
                    event = ""
                    for line in stream.iter_lines():
                        if self._stop.is_set():
                            return
                        if line.startswith("event: "):
                            event = line[7:].strip()
                        elif line.startswith("data: "):
                            self.frames.append((event, json.loads(line[6:])))
        except Exception:  # pragma: no cover - the stream ends at shutdown
            return

    def __enter__(self) -> EventCollector:
        self._thread.start()
        time.sleep(0.3)  # let the subscription register before the POST
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()

    def of(self, kind: str) -> list[dict]:
        return [data for event, data in self.frames if event == kind and not data.get("replay")]

    def notice_reasons(self) -> list[str]:
        return [n.get("reason") for n in self.of("notice")]


# ==========================================================================


def test_the_whole_lifecycle_through_the_api(server) -> None:
    """start -> markers on SSE -> ready -> gateway lists it -> gateway reaches
    it -> stop -> the unit is gone."""
    assert stub_row(state())["live"] is False

    with EventCollector() as events:
        response = httpx.post(f"{BASE}/api/models/stub/start", timeout=10.0)
        assert response.status_code == 202, response.text
        assert response.json() == {"accepted": True, "action": "start", "model": "stub"}

        ready = wait_until(lambda: "ready" in events.notice_reasons(), timeout_s=90)
        assert ready, (
            f"no ready notice; reasons={events.notice_reasons()} "
            f"journal={_units.journal_tail(STUB_UNIT, 30)}"
        )

        # The boot markers arrived AS PROGRESS, in vLLM's own order. This is
        # the thing the 462-line phase machine used to do.
        markers = [p["text"] for p in events.of("progress") if p["kind"] == "marker"]
        expected = [
            "Loading weights took 0.01 seconds",
            "GPU KV cache size: 4,096 tokens",
            "Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)",
        ]
        if markers[:3] != expected:
            # Two very different failures produce an empty marker list, and the
            # message has to say which. If the journal HAS the lines, the
            # follower missed them, and the cause seen on this box was inotify
            # watch exhaustion — systemd logs "Failed to add control inotify
            # watch descriptor ... No space left on device" right beside them
            # and `journalctl -f` then never learns about new entries. That is
            # the machine, not the code. If the journal does NOT have them, the
            # model never got that far and this is a real regression.
            tail = _units.journal_tail(STUB_UNIT, 200)
            logged = [m for m in expected if any(m in line for line in tail)]
            inotify = [line for line in tail if "inotify watch descriptor" in line]
            raise AssertionError(
                f"progress markers were {markers[:3]}, expected {expected}.\n"
                f"markers present in the journal: {logged}\n"
                f"inotify exhaustion lines: {len(inotify)}\n"
                + ("The journal HAS the lines, so the follower missed them "
                   "(see the inotify count above)." if len(logged) == 3
                   else "The journal does NOT have the lines: the stub never "
                        "reached that point. This is a real failure.")
            )

    # The unit is real, and it is not in our cgroup.
    assert _units.exists(STUB_UNIT)
    cgroup = _units.control_group(STUB_UNIT)
    assert cgroup.endswith(f"/{STUB_UNIT}.service"), cgroup
    mine = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[-1]
    assert cgroup != mine and not cgroup.startswith(mine.rstrip("/") + "/")

    # /api/state sees it.
    row = wait_until(lambda: stub_row(state()) if stub_row(state())["ready"] else None, 30)
    assert row and row["ready"] is True
    assert row["unit_state"] == "active (running)"
    assert row["pid"] > 0
    assert row["uptime_s"] is not None and row["uptime_s"] >= 0
    assert row["ctx"] == 4096

    # THE GATEWAY lists it, under its id and its alias.
    listed = httpx.get(f"{BASE}/v1/models", timeout=10.0).json()
    ids = [entry["id"] for entry in listed["data"]]
    assert ids == [STUB_ID, "stubby"], ids
    assert all(entry["max_model_len"] == 4096 for entry in listed["data"])

    # A chat request THROUGH THE GATEWAY reaches the stub.
    chat = httpx.post(
        f"{BASE}/v1/chat/completions",
        json={"model": STUB_ID, "messages": [{"role": "user", "content": "hi"}]},
        timeout=30.0,
    )
    assert chat.status_code == 200, chat.text
    assert chat.json()["choices"][0]["message"]["content"] == "ok"

    # ...and so does one addressed by the ALIAS, which the gateway rewrites to
    # the id before forwarding. This is the 404-after-rename class of bug.
    aliased = httpx.post(
        f"{BASE}/v1/chat/completions",
        json={"model": "stubby", "messages": [{"role": "user", "content": "hi"}]},
        timeout=30.0,
    )
    assert aliased.status_code == 200, aliased.text

    # Stop it.
    with EventCollector() as events:
        response = httpx.post(f"{BASE}/api/models/stub/stop", timeout=10.0)
        assert response.status_code == 202
        assert wait_until(lambda: "stopped" in events.notice_reasons(), timeout_s=60), (
            events.notice_reasons()
        )

    # `--collect` unloads the unit as soon as it is inactive: gone, not merely
    # inactive, so the name is reusable without a reset-failed dance.
    assert _units.exists(STUB_UNIT) is False
    assert STUB_UNIT not in _units.list_units("sd-test-*")
    assert wait_until(lambda: stub_row(state())["live"] is False, 30)
    assert httpx.get(f"{BASE}/v1/models", timeout=10.0).json()["data"] == []


def test_the_registry_is_what_launched_it(server) -> None:
    """The argv really came from ``models.toml``.

    The shim asserts on its own arguments, so a boot that reached the ready
    state has already proved that ``--port``, ``--served-model-name``,
    ``--max-model-len`` and ``--gpu-memory-utilization`` were all rendered. The
    journal is where that is visible after the fact.
    """
    response = httpx.post(f"{BASE}/api/models/stub/start", timeout=10.0)
    assert response.status_code == 202
    try:
        assert wait_until(lambda: stub_row(state())["ready"], timeout_s=90), (
            _units.journal_tail(STUB_UNIT, 30)
        )
        # The stub prints the name it was told to serve.
        tail = "\n".join(_units.journal_tail(STUB_UNIT, 40))
        assert f"as {STUB_ID}" in tail, tail
        assert f"127.0.0.1:{STUB_PORT}" in tail, tail
    finally:
        httpx.post(f"{BASE}/api/models/stub/stop", timeout=10.0)
        wait_until(lambda: not _units.exists(STUB_UNIT), 60)


def test_desired_state_records_the_start_and_forgets_the_stop(server) -> None:
    """``desired.json`` is what an operator last asked for, not a mirror of
    what is running — that is what makes reconcile bring a crashed model back
    and leave a stopped one alone."""
    httpx.post(f"{BASE}/api/models/stub/start", timeout=10.0)
    try:
        assert wait_until(lambda: stub_row(state())["ready"], timeout_s=90)
        assert wait_until(lambda: "stub" in state()["desired"]["residents"], 30), state()["desired"]
    finally:
        httpx.post(f"{BASE}/api/models/stub/stop", timeout=10.0)
        wait_until(lambda: not _units.exists(STUB_UNIT), 60)
    assert wait_until(lambda: "stub" not in state()["desired"]["residents"], 30)


def test_starting_it_twice_is_refused_without_touching_the_unit(server) -> None:
    httpx.post(f"{BASE}/api/models/stub/start", timeout=10.0)
    try:
        assert wait_until(lambda: stub_row(state())["ready"], timeout_s=90)
        # `busy` outranks `already_live` in the precheck, and the first start
        # holds it until its outcome has been published — which can be after
        # the poller has already seen the model answer. Wait for the mutation
        # to finish, so this asserts the refusal it means to.
        assert wait_until(lambda: state().get("busy") is None, 30), state().get("busy")
        pid = stub_row(state())["pid"]
        again = httpx.post(f"{BASE}/api/models/stub/start", timeout=10.0)
        assert again.status_code == 409
        assert again.json()["error"]["reason"] == "already_live"
        time.sleep(1.0)
        assert stub_row(state())["pid"] == pid, "the refused start restarted the model"
    finally:
        httpx.post(f"{BASE}/api/models/stub/stop", timeout=10.0)
        wait_until(lambda: not _units.exists(STUB_UNIT), 60)


def test_the_journal_route_reads_the_real_journal(server) -> None:
    httpx.post(f"{BASE}/api/models/stub/start", timeout=10.0)
    try:
        assert wait_until(lambda: stub_row(state())["ready"], timeout_s=90)
        # 1000, not 40: the dashboard polls /v1/models and /metrics every two
        # seconds and the stub logs both, so the boot lines scroll out of a
        # short tail within a minute. A journal route that could only show the
        # last 40 lines of a chatty unit would never show a boot failure.
        payload = httpx.get(f"{BASE}/api/log/stub?lines=1000", timeout=20.0).json()
        assert payload["unit"] == STUB_UNIT
        assert any("stub listening" in line for line in payload["lines"]), payload["lines"][:10]
        assert any("Loading weights took" in line for line in payload["lines"])
    finally:
        httpx.post(f"{BASE}/api/models/stub/stop", timeout=10.0)
        wait_until(lambda: not _units.exists(STUB_UNIT), 60)


def test_a_request_for_a_stopped_model_is_503_not_a_connection_refused(server) -> None:
    """While nothing is up the gateway answers 503 with Retry-After, naming
    what is in the main slot. A client then waits instead of concluding it is
    misconfigured."""
    assert wait_until(lambda: stub_row(state())["live"] is False, 30)
    response = httpx.post(
        f"{BASE}/v1/chat/completions",
        json={"model": STUB_ID, "messages": []},
        timeout=10.0,
    )
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "15"
    assert response.json()["error"]["code"] == "not_running"
