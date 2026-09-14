"""``servedeck status|start|stop|log`` as a real subprocess, against a real
HTTP server.

Not TestClient: the CLI's whole job is to be a separate process talking to a
port, and the two things most likely to break in it — the SSE stream being read
incrementally, and an exit code — are exactly the things an in-process call
would paper over. So uvicorn really binds 127.0.0.1:8041 (inside the range this
packet owns) and the console script really runs.

The dashboard's Control is a fake, so nothing here touches systemd or the GPU.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import uvicorn

from servedeck import app as _app
from servedeck import control as _control
from servedeck import gpu as _gpu
from servedeck import models as _models
from servedeck import units as _units
from servedeck import wire as _wire
from servedeck.settings import Settings

ROOT = Path(__file__).resolve().parent.parent
SERVEDECK_BIN = ROOT / ".venv" / "bin" / "servedeck"
PORT = 8041
BASE = f"http://127.0.0.1:{PORT}"

TOML = """
[gpu]
total_mib = 100000
margin_mib = 1024

[builds.stock]
venv = "/opt/stock"
cuda_home = "/opt/stock/lib/python3.13/site-packages/nvidia/cu13"

[models.big]
id = "Big-Model"
aliases = ["big-alias"]
repo = "org/big"
slot = "main"
port = 19001
build = "stock"
ctx = 32768

[models.small]
id = "Small-Model"
repo = "org/small"
slot = "resident"
vram_mib = 3300
port = 19002
build = "stock"
ctx = 4096
"""

pytestmark = pytest.mark.skipif(
    not SERVEDECK_BIN.exists(), reason="the console script is not installed in .venv"
)


@dataclass
class FakeLive:
    key: str
    unit: str
    ready: bool = True
    state: str = "active"
    sub_state: str = "running"
    pid: int = 4242
    restarts: int = 0
    port: int | None = None
    unknown: bool = False


class FakeControl:
    def __init__(self) -> None:
        self.rows: list[FakeLive] = []
        self.calls: list[str] = []
        self.fail_start = False

    def live(self) -> list[FakeLive]:
        return list(self.rows)

    def start(self, key: str, timeout_s: float = 900.0, on_progress=None, **_kw):
        self.calls.append(f"start:{key}")
        for index, text in enumerate(
            ["Loading weights took 1.2 seconds", "GPU KV cache size: 400,000 tokens"]
        ):
            if on_progress is not None:
                on_progress(_control.Progress(kind="marker", text=text, marker_index=index))
        if self.fail_start:
            return _control.StartResult(
                key=key, unit=f"sd-test-{key}", ready=False, elapsed_s=3.0,
                failure="CUDA out of memory", journal=["torch.OutOfMemoryError"],
            )
        self.rows.append(FakeLive(key=key, unit=f"sd-test-{key}"))
        return _control.StartResult(key=key, unit=f"sd-test-{key}", ready=True, elapsed_s=3.0)

    def stop(self, key: str, **_kw):
        self.calls.append(f"stop:{key}")
        self.rows = [r for r in self.rows if r.key != key]
        return _control.StopResult(key=key, unit=f"sd-test-{key}", was_live=True, held_mib=3300)

    def switch(self, key: str, timeout_s: float = 900.0, on_progress=None, **_kw):
        self.calls.append(f"switch:{key}")
        return _control.SwitchResult(
            stopped=None, released=True, waited_s=0.0, free_after_mib=1,
            started=self.start(key, on_progress=on_progress),
        )

    def adopt(self, **_kw):
        self.calls.append("adopt")
        return _control.Adoption(adopted=["small"])

    def reconcile(self, desired=None, on_progress=None, **_kw):
        return _control.Reconciliation()


@pytest.fixture(scope="module")
def _patched_host():
    """Patch the host calls for the whole module.

    Module-scoped and applied by hand, because the server thread below outlives
    any one test and a function-scoped monkeypatch would restore the real
    nvidia-smi underneath a running poller.
    """
    originals = {
        (_app._gpu, "total_mib"): _gpu.total_mib,
        (_app._gpu, "free_mib"): _gpu.free_mib,
        (_app._units, "properties"): _units.properties,
        (_app._units, "journal_tail"): _units.journal_tail,
        (_app._wire, "WIRE_TARGETS"): _wire.WIRE_TARGETS,
    }
    _app._gpu.total_mib = lambda: 100000
    _app._gpu.free_mib = lambda: 42000
    _app._units.properties = lambda *a, **k: {}
    _app._units.journal_tail = lambda unit, lines, **k: [f"{unit}: line {i}" for i in range(lines)]
    _app._wire.WIRE_TARGETS = ()
    try:
        yield
    finally:
        for (module, name), original in originals.items():
            setattr(module, name, original)


@pytest.fixture(scope="module")
def server(tmp_path_factory, _patched_host):
    tmp = tmp_path_factory.mktemp("cli")
    models_path = tmp / "models.toml"
    models_path.write_text(TOML)
    settings = Settings(
        listen_host="127.0.0.1",
        listen_port=PORT,
        models_path=models_path,
        state_dir=tmp / "state",
        unit_prefix="sd-test-",
    )
    control = FakeControl()
    app = _app.create_app(
        settings, registry=_models.load(models_path), control=control, reconcile=False, poll=True
    )
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error")
    uv = uvicorn.Server(config)
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{BASE}/api/health", timeout=1.0).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.05)
    else:  # pragma: no cover - the server failed to come up
        uv.should_exit = True
        pytest.fail(f"the test server never answered on {BASE}")

    try:
        yield control
    finally:
        uv.should_exit = True
        thread.join(timeout=10)
        assert not thread.is_alive(), (
            "uvicorn did not exit within 10s — a parked SSE subscriber is the "
            "TimeoutStopSec bug this packet exists to fix"
        )


def run_cli(*args: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SERVEDECK_BIN), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(ROOT),
    )


@pytest.fixture(autouse=True)
def _reset(server):
    """Clear the fake, then WAIT for the dashboard to have noticed.

    The dashboard's prechecks read a cached liveness snapshot the poller
    refreshes every two seconds — that cache is the whole reason nothing in a
    request handler shells out. Resetting the fake without waiting for the next
    poll leaves the previous test's model "live" for up to two seconds, and the
    next start is refused with ``already_live``.
    """
    server.rows = []
    server.calls = []
    server.fail_start = False
    wait_for_state(lambda s: not any(m["live"] for m in s["models"]))
    server.calls = []
    yield


def wait_for_state(predicate, timeout_s: float = 10.0):
    """The dashboard's poller refreshes on its own clock; a CLI assertion that
    raced it would be flaky rather than wrong."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = httpx.get(f"{BASE}/api/state", timeout=5.0).json()
        if predicate(state):
            return state
        time.sleep(0.1)
    raise AssertionError("the dashboard state never reached the expected shape")


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------


def test_status_prints_a_table_of_every_model(server) -> None:
    result = run_cli("status", "--url", BASE)
    assert result.returncode == 0, result.stderr
    assert "Big-Model" in result.stdout
    assert "Small-Model" in result.stdout
    assert "gateway  http://127.0.0.1:8041/v1" in result.stdout
    assert "42,000 MiB free of 100,000" in result.stdout


def test_status_shows_a_live_model_as_ready(server) -> None:
    server.rows = [FakeLive(key="small", unit="sd-test-small")]
    wait_for_state(lambda s: any(m["ready"] for m in s["models"]))
    result = run_cli("status", "--url", BASE)
    assert "ready" in result.stdout
    assert "active (running)" in result.stdout


def test_status_json_is_the_state_document_verbatim(server) -> None:
    import json

    result = run_cli("status", "--json", "--url", BASE)
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["gateway_url"] == f"{BASE}/v1"
    assert {m["key"] for m in payload["models"]} == {"big", "small"}


def test_status_against_a_dead_dashboard_exits_1_and_says_so() -> None:
    result = run_cli("status", "--url", "http://127.0.0.1:8048")
    assert result.returncode == 1
    assert "dashboard not running" in result.stderr


# --------------------------------------------------------------------------
# start / stop / switch
# --------------------------------------------------------------------------


def test_start_streams_progress_and_exits_zero(server) -> None:
    """The POST is fire-and-forget; everything the operator sees comes off the
    SSE stream. An exit code that did not depend on it would report success for
    a boot that failed four minutes later."""
    result = run_cli("start", "small", "--url", BASE)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "start small: accepted" in result.stdout
    assert "Loading weights took 1.2 seconds" in result.stdout
    assert "GPU KV cache size" in result.stdout
    assert "ready" in result.stdout
    assert "start:small" in server.calls


def test_a_failed_boot_exits_one_and_prints_the_journal(server) -> None:
    server.fail_start = True
    result = run_cli("start", "small", "--url", BASE)
    assert result.returncode == 1
    assert "CUDA out of memory" in result.stderr
    assert "torch.OutOfMemoryError" in result.stderr


def test_starting_an_unknown_model_is_refused_without_waiting(server) -> None:
    result = run_cli("start", "nope", "--url", BASE, timeout=20)
    assert result.returncode == 1
    assert "refused (404)" in result.stderr
    assert "unknown_model" in result.stderr
    assert server.calls == [], "a refused start must not reach Control"


def test_starting_a_second_main_model_is_refused_with_the_holder(server) -> None:
    server.rows = [FakeLive(key="big", unit="sd-test-big")]
    wait_for_state(lambda s: any(m["key"] == "big" and m["live"] for m in s["models"]))
    result = run_cli("start", "big", "--url", BASE, timeout=20)
    assert result.returncode == 1
    assert "already_live" in result.stderr


def test_stop_exits_zero_and_reports_the_unit(server) -> None:
    server.rows = [FakeLive(key="small", unit="sd-test-small")]
    wait_for_state(lambda s: any(m["key"] == "small" and m["live"] for m in s["models"]))
    result = run_cli("stop", "small", "--url", BASE)
    assert result.returncode == 0, result.stderr
    assert "sd-test-small stopped" in result.stdout
    assert "stop:small" in server.calls


def test_switch_streams_and_exits_zero(server) -> None:
    result = run_cli("switch", "big", "--url", BASE)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "switch big: accepted" in result.stdout
    assert "switch:big" in server.calls


def test_adopt_reports_what_it_recorded(server) -> None:
    result = run_cli("adopt", "--url", BASE)
    assert result.returncode == 0, result.stderr
    assert "accepted" in result.stdout


# --------------------------------------------------------------------------
# log
# --------------------------------------------------------------------------


def test_log_prints_the_units_journal(server) -> None:
    result = run_cli("log", "small", "-n", "4", "--url", BASE)
    assert result.returncode == 0
    assert result.stdout.splitlines() == [f"sd-test-small: line {i}" for i in range(4)]


def test_log_for_an_unknown_model_exits_one(server) -> None:
    result = run_cli("log", "nope", "--url", BASE)
    assert result.returncode == 1
    assert "no model 'nope'" in result.stderr


# --------------------------------------------------------------------------
# The fallback: the dashboard is down
# --------------------------------------------------------------------------


def test_start_falls_back_to_driving_control_directly(tmp_path) -> None:
    """``servedeck start`` must work when the dashboard is down — that is the
    state an operator is most likely to be in, and a CLI whose only mode is
    "ask the thing that is not running" fails exactly when it is needed.

    The key is deliberately unknown, so the fallback proves it reached a real
    ``Control`` (which refuses unknown keys before touching systemd) without
    this test being able to start anything.
    """
    models_path = tmp_path / "models.toml"
    models_path.write_text(TOML)
    env = {
        **_clean_env(),
        "SERVEDECK_MODELS": str(models_path),
        "SERVEDECK_STATE_DIR": str(tmp_path / "state"),
        "SERVEDECK_UNIT_PREFIX": "sd-test-",
    }
    result = subprocess.run(
        [str(SERVEDECK_BIN), "start", "definitely-not-a-model", "--url", "http://127.0.0.1:8048"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=str(ROOT),
    )
    assert "dashboard not running, acting directly" in result.stderr
    assert result.returncode == 1
    assert "unknown_model" in result.stderr


def test_the_fallback_says_which_path_it_took(tmp_path) -> None:
    """An operator who cannot tell the two paths apart cannot tell "the model
    started" from "the model started and the dashboard does not know about
    it" — and the answer to the second is `servedeck adopt`, which they will
    not think to run."""
    models_path = tmp_path / "models.toml"
    models_path.write_text(TOML)
    env = {
        **_clean_env(),
        "SERVEDECK_MODELS": str(models_path),
        "SERVEDECK_STATE_DIR": str(tmp_path / "state"),
        "SERVEDECK_UNIT_PREFIX": "sd-test-",
    }
    result = subprocess.run(
        [str(SERVEDECK_BIN), "adopt", "--url", "http://127.0.0.1:8048"],
        capture_output=True, text=True, timeout=60, env=env, cwd=str(ROOT),
    )
    assert "dashboard not running, acting directly" in result.stderr


def _clean_env() -> dict[str, str]:
    """A subprocess environment without this suite's own overrides."""
    import os

    env = {k: v for k, v in os.environ.items() if not k.startswith("SERVEDECK_")}
    env.setdefault("PATH", "/usr/bin:/bin")
    env["PYTEST_ADDOPTS"] = ""
    return env


# --------------------------------------------------------------------------
# The local commands still work with no server at all
# --------------------------------------------------------------------------


def test_models_needs_no_server(tmp_path) -> None:
    models_path = tmp_path / "models.toml"
    models_path.write_text(TOML)
    result = run_cli("models", "--models-toml", str(models_path))
    assert result.returncode == 0
    assert "Big-Model" in result.stdout and "Small-Model" in result.stdout
    assert sys.executable  # the console script ran out of the same venv
