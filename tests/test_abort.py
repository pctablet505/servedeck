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

    # Signal abort in ctl; start() puts the key in _booting before wait_ready
    ctl._booting.add("a")
    ctl.abort("a")
    assert "a" in ctl._aborted

    # When wait_ready is called, it detects the abort on the first tick and finishes cleanly
    res = ctl.wait_ready("a", timeout_s=100.0)
    assert not res.ready
    assert "cancelled by operator" in (res.failure or "")


def test_post_models_abort_endpoint(tmp_path: Path) -> None:
    ctl = AbortableFakeControl()
    app = _app_with_abort(tmp_path, ctl)
    app.state.rt.busy = {"action": "start", "key": "a", "label": "start a"}
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


def test_abort_none_only_stops_booting_units_not_unrelated_live_models(tmp_path: Path) -> None:
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

    ctl = _control.Control(
        adapter,
        margin_mib=1024,
        total_mib=100000,
        unit_prefix="sd-test-",
        run=mock_run,
        free_mib=lambda: 90000,
    )
    # When nothing is in _booting, abort(None) must not stop any units
    ctl.abort(None)
    assert stopped_units == []

    # When 'a' is tracked as booting, abort(None) stops only 'sd-test-a.service'
    ctl._booting.add("a")
    ctl.abort(None)
    assert "a" in ctl._aborted
    assert any("sd-test-a" in u for u in stopped_units)


def test_switch_abort_during_vram_release_does_not_launch_replacement(tmp_path: Path) -> None:
    models_path = tmp_path / "models.toml"
    models_path.write_text(
        TOML
        + """
[models.b]
id = "Model B"
repo = "org/B"
slot = "main"
port = 8002
build = "stock"
ctx = 32768
"""
    )
    adapter = _app._RegistryAdapter(_models.load(models_path))
    started_units: list[str] = []

    def mock_run(argv: list[str]) -> Any:
        m = MagicMock()
        m.returncode = 0
        m.stdout = ""
        m.stderr = ""
        if "systemd-run" in argv:
            started_units.append(argv[-1])
        return m

    ctl = _control.Control(
        adapter,
        margin_mib=1024,
        total_mib=100000,
        unit_prefix="sd-test-",
        run=mock_run,
        free_mib=lambda: 90000,
    )
    ctl.live_main = lambda *_a, **_kw: _control.LiveModel(  # type: ignore[method-assign]
        key="a", unit="sd-test-a.service", port=8001, state="active",
        pid=1234, ready=True, restarts=0, sub_state="running", adopted=False,
    )
    ctl.stop = lambda key, **_kw: _control.StopResult(  # type: ignore[method-assign]
        key=key, unit=f"sd-test-{key}.service", was_live=True, held_mib=20000,
        free_before_mib=50000, reaped=(),
    )

    def fake_wait_for_release(**_kw: Any) -> tuple[bool, float, int | None, str | None]:
        # Operator aborts 'b' while waiting for 'a's VRAM to release
        ctl.abort("b", stop_unit=False)
        return True, 1.5, 90000, None

    ctl._wait_for_release = fake_wait_for_release  # type: ignore[method-assign]
    res = ctl.switch("b")
    assert isinstance(res, _control.SwitchResult)
    assert isinstance(res.started, _control.StartResult)
    assert not res.started.ready
    assert "cancelled by operator" in (res.started.failure or "")
    assert started_units == []


def test_publish_result_classifies_operator_abort_as_warn_aborted() -> None:
    hub = _app.Hub()
    sub = hub.subscribe()
    res = _control.StartResult(
        key="a",
        unit="sd-test-a.service",
        ready=False,
        elapsed_s=2.0,
        markers=[],
        failure="boot of a was cancelled by operator",
        journal=[],
    )
    _app._publish_result(hub, "start a", res)
    ev = sub.queue.get_nowait()
    assert ev["type"] == "notice"
    assert ev["data"]["level"] == "warn"
    assert ev["data"]["reason"] == "aborted"



def _fake_control(tmp_path: Path, toml: str = TOML) -> tuple[_control.Control, list[str]]:
    """A Control whose systemctl, journal and clock are fakes, with its own desired.json."""
    models_path = tmp_path / "models.toml"
    models_path.write_text(toml)
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

        def close(self) -> None:
            return None

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
        desired_path=tmp_path / "desired.json",
    )
    return ctl, stopped_units


def test_abort_leaves_a_model_that_is_not_booting_alone(tmp_path: Path) -> None:
    # An abort naming a model that is serving (or stopped) is not a boot to
    # cancel. Stopping its unit from here would skip stop()'s bookkeeping and,
    # with desired state still naming it, recovery would boot it again.
    ctl, stopped_units = _fake_control(tmp_path)
    assert ctl.abort("a") is False
    assert stopped_units == []
    assert "a" not in ctl._aborted


def test_abort_endpoint_refuses_a_model_that_is_not_booting(tmp_path: Path) -> None:
    ctl = AbortableFakeControl()
    app = _app_with_abort(tmp_path, ctl)
    with TestClient(app) as client:
        r = client.post("/api/models/a/abort")
    assert r.status_code == 409
    assert r.json()["error"]["reason"] == "not_booting"
    assert ctl.aborted == []


def test_a_cancelled_boot_is_not_left_for_recovery_to_relaunch(tmp_path: Path) -> None:
    # A restart, or recovery's own relaunch, boots a model desired state names.
    # If the cancel leaves it named, the next poll relaunches the very boot the
    # operator just stopped.
    ctl, _ = _fake_control(tmp_path)
    ctl._save_desired(ctl.load_desired().with_main("a"))
    ctl._booting.add("a")  # start() puts the key there before wait_ready
    ctl.abort("a", stop_unit=False)
    res = ctl.wait_ready("a", timeout_s=100.0)
    assert "cancelled by operator" in (res.failure or "")
    assert ctl.load_desired().main is None


def test_a_cancelled_switch_does_not_bring_back_the_old_holder(tmp_path: Path) -> None:
    # switch() stops the old holder with forget_intent=False so a FAILED boot
    # gets it back. An operator cancel is not a failure: relaunching the old
    # holder is a boot nobody asked for either.
    ctl, _ = _fake_control(
        tmp_path,
        TOML
        + """
[models.b]
id = "Model B"
repo = "org/B"
slot = "main"
port = 8002
build = "stock"
ctx = 32768
""",
    )
    ctl._save_desired(ctl.load_desired().with_main("a"))
    ctl.live_main = lambda *_a, **_kw: _control.LiveModel(  # type: ignore[method-assign]
        key="a", unit="sd-test-a.service", port=8001, state="active",
        pid=1234, ready=True, restarts=0, sub_state="running", adopted=False,
    )
    ctl.stop = lambda key, **_kw: _control.StopResult(  # type: ignore[method-assign]
        key=key, unit=f"sd-test-{key}.service", was_live=True, held_mib=20000,
        free_before_mib=50000, reaped=(),
    )

    def fake_wait_for_release(**_kw: Any) -> tuple[bool, float, int | None, str | None]:
        ctl.abort("b", stop_unit=False)
        return True, 1.5, 90000, None

    ctl._wait_for_release = fake_wait_for_release  # type: ignore[method-assign]
    res = ctl.switch("b")
    assert isinstance(res, _control.SwitchResult)
    assert "cancelled by operator" in (res.started.failure or "")  # type: ignore[union-attr]
    assert ctl.load_desired().main is None
