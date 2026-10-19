"""Tests for force-stopping and aborting in-flight model boots."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest

from servedeck import app as _app
from servedeck import control as _control
from servedeck import models as _models
from servedeck.settings import Settings


TOML = """
[gpu]
total_mib = 100000
margin_mib = 1024

[builds.stock]
venv = "/opt/v"
cuda_home = "/opt/cuda"

[models.a]
id = "Model A"
repo = "org/A"
slot = "main"
port = 8001
build = "stock"
ctx = 32768
"""


class AbortableFakeControl:
    def __init__(self) -> None:
        self.aborted: list[str | None] = []
        self._stopped: list[str] = []

    def live(self) -> list[Any]:
        return []

    def adopt(self, **_kw: Any) -> None:
        return None

    def reconcile(self, *_a: Any, **_kw: Any) -> None:
        return None

    def abort(self, key: str | None = None) -> bool:
        self.aborted.append(key)
        return True

    def stop(self, key: str, **_kw: Any) -> _control.StopResult:
        self._stopped.append(key)
        return _control.StopResult(
            key=key, unit=f"model-{key}", was_live=True, held_mib=1000,
            free_before_mib=50000, reaped=(),
        )


def _app_with_abort(tmp_path: Path, ctl: Any) -> Any:
    models_path = tmp_path / "models.toml"
    models_path.write_text(TOML)
    settings = Settings(
        listen_host="127.0.0.1", listen_port=8099, models_path=models_path,
        state_dir=tmp_path / "state", unit_prefix="sd-test-",
    )
    return _app.create_app(
        settings, registry=_models.load(models_path), control=ctl,
        reconcile=False, poll=False,
    )


def test_control_abort_stops_wait_ready_loop(tmp_path: Path) -> None:
    models_path = tmp_path / "models.toml"
    models_path.write_text(TOML)
    adapter = _app._RegistryAdapter(_models.load(models_path))
    stopped_units: list[str] = []

    def mock_run(argv: list[str]) -> Any:
        m = MagicMock()
        m.returncode = 0
        m.stdout = ""
        m.stderr = ""
        if "stop" in argv:
            stopped_units.append(argv[-1])
        return m

    clock_val = 100.0
    def mock_clock() -> float:
        nonlocal clock_val
        clock_val += 0.1
        return clock_val

    class FakeFollowStream:
        def poll_lines(self, tick: float) -> list[str]:
            return []

    ctl = _control.Control(
        adapter,
        margin_mib=1024,
        total_mib=100000,
        unit_prefix="sd-test-",
        run=mock_run,
        spawn=lambda argv: FakeFollowStream(),
        clock=mock_clock,
        sleep=lambda dt: None,
        free_mib=lambda: 90000,
    )

    # Signal abort in ctl
    ctl.abort("a")
    assert "a" in ctl._aborted

    # When wait_ready is called, it detects the abort on the first tick and finishes cleanly
    res = ctl.wait_ready("a", timeout_s=100.0)
    assert not res.ready
    assert "cancelled by operator" in (res.failure or "")


def test_post_models_abort_endpoint(tmp_path: Path) -> None:
    ctl = AbortableFakeControl()
    app = _app_with_abort(tmp_path, ctl)
    with TestClient(app) as client:
        r = client.post("/api/models/a/abort")
        assert r.status_code == 202
        assert r.json() == {"accepted": True, "action": "abort", "model": "a"}
        assert "a" in ctl.aborted


def test_post_server_abort_endpoint(tmp_path: Path) -> None:
    ctl = AbortableFakeControl()
    app = _app_with_abort(tmp_path, ctl)
    with TestClient(app) as client:
        r = client.post("/api/server/abort")
        assert r.status_code == 202
        assert r.json()["accepted"] is True
        assert r.json()["action"] == "abort"


def test_post_server_stop_while_busy_triggers_abort(tmp_path: Path) -> None:
    ctl = AbortableFakeControl()
    app = _app_with_abort(tmp_path, ctl)
    rt = app.state.rt
    rt.busy = {"action": "start", "key": "a", "label": "start a"}
    with TestClient(app) as client:
        # Instead of 409 busy refusal, stop when busy triggers abort of the in-flight boot
        r = client.post("/api/server/stop")
        assert r.status_code == 202
        assert r.json()["action"] == "abort"
        assert "a" in ctl.aborted
