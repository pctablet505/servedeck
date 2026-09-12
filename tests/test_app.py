"""The HTTP surface, driven with fakes: no systemd, no GPU, no model.

Everything that shells out (``systemctl``, ``journalctl``, ``nvidia-smi``) or
touches a real client config is injected or monkeypatched, so this file can run
anywhere and says nothing about the box it runs on. The two facts it is really
here to pin are the two that cost outages:

* a mutation is refused **before** it is accepted, with a typed reason;
* ``reconcile`` runs only after the listen port answers.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from servedeck import app as _app
from servedeck import control as _control
from servedeck import gpu as _gpu
from servedeck import models as _models
from servedeck import units as _units
from servedeck import wire as _wire
from servedeck.settings import Settings

TOML = """
[gpu]
total_mib = 100000
margin_mib = 1024

[builds]
stock = "/opt/stock"

[models.big]
id = "Big-Model"
aliases = ["big-alias"]
repo = "org/big"
slot = "main"
port = 19001
build = "stock"
ctx = 32768
reasoning = { parser = "glm", mirror_content = true }

[models.big.presets.big-high]
reasoning_effort = "high"

[models.other]
id = "Other-Model"
repo = "org/other"
slot = "main"
port = 19003
build = "stock"
ctx = 8192

[models.small]
id = "Small-Model"
repo = "org/small"
slot = "resident"
vram_mib = 3300
port = 19002
build = "stock"
ctx = 4096
"""


@pytest.fixture
def anyio_backend():
    return "asyncio"


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


@dataclass
class FakeLive:
    key: str
    unit: str
    ready: bool = True
    state: str = "active"
    sub_state: str = "running"
    pid: int = 999
    restarts: int = 0
    port: int | None = None
    unknown: bool = False


class FakeControl:
    """Records what it was asked to do, and answers with whatever the test set.

    Deliberately not a Mock: the assertions below are about ORDER and about
    which arguments arrived, and a Mock's call list reads as plumbing where
    this reads as a narrative.
    """

    def __init__(self, rows: list[FakeLive] | None = None) -> None:
        self.rows = rows or []
        self.calls: list[str] = []
        self.results: dict[str, Any] = {}
        self.reconcile_arg: Any = None
        self.progress_lines: list[str] = []

    def live(self) -> list[FakeLive]:
        return list(self.rows)

    def _result(self, name: str, key: str, default: Any) -> Any:
        self.calls.append(f"{name}:{key}")
        return self.results.get(name, default)

    def start(self, key: str, timeout_s: float = 900.0, on_progress=None, **_kw) -> Any:
        for line in self.progress_lines:
            if on_progress is not None:
                on_progress(_control.Progress(kind="line", text=line))
        return self._result(
            "start",
            key,
            _control.StartResult(key=key, unit=f"model-{key}", ready=True, elapsed_s=1.0),
        )

    def stop(self, key: str, **_kw) -> Any:
        return self._result(
            "stop", key, _control.StopResult(key=key, unit=f"model-{key}", was_live=True, held_mib=10)
        )

    def switch(self, key: str, timeout_s: float = 900.0, on_progress=None, **_kw) -> Any:
        return self._result(
            "switch",
            key,
            _control.SwitchResult(
                stopped=None,
                released=True,
                waited_s=0.0,
                free_after_mib=1000,
                started=_control.StartResult(key=key, unit=f"model-{key}", ready=True, elapsed_s=2.0),
            ),
        )

    def adopt(self, **_kw) -> Any:
        return self._result("adopt", "", _control.Adoption(adopted=["small"]))

    def reconcile(self, desired=None, on_progress=None, **_kw) -> Any:
        self.calls.append("reconcile")
        self.reconcile_arg = desired
        return self.results.get("reconcile", _control.Reconciliation(already_live=["small"]))


@pytest.fixture
def registry(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(TOML)
    return _models.load(path)


@pytest.fixture
def settings(tmp_path, registry):
    return Settings(
        listen_host="127.0.0.1",
        listen_port=8049,
        models_path=tmp_path / "models.toml",
        state_dir=tmp_path / "state",
        unit_prefix="sd-test-",
    )


@pytest.fixture
def control() -> FakeControl:
    return FakeControl()


@pytest.fixture(autouse=True)
def _no_real_host_calls(monkeypatch):
    """Nothing in this file may reach nvidia-smi, systemd or a client config."""
    monkeypatch.setattr(_gpu, "total_mib", lambda: 100000)
    monkeypatch.setattr(_gpu, "free_mib", lambda: 42000)
    monkeypatch.setattr(_app._gpu, "total_mib", lambda: 100000)
    monkeypatch.setattr(_app._gpu, "free_mib", lambda: 42000)
    monkeypatch.setattr(_units, "properties", lambda *a, **k: {})
    monkeypatch.setattr(_app._units, "properties", lambda *a, **k: {})
    monkeypatch.setattr(_app._units, "journal_tail", lambda unit, lines, **k: [f"{unit} line {i}" for i in range(lines)])
    monkeypatch.setattr(_app._discovery, "discover_models", lambda *a, **k: [])
    monkeypatch.setattr(_wire, "WIRE_TARGETS", ())
    monkeypatch.setattr(_app._wire, "WIRE_TARGETS", ())


@pytest.fixture
def runtime(settings, registry, control):
    return _app.build_runtime(settings, registry=registry, control=control)


@pytest.fixture
def client(settings, registry, control):
    app = _app.create_app(
        settings, registry=registry, control=control, reconcile=False, poll=False
    )
    with TestClient(app) as test_client:
        test_client.rt = app.state.rt  # type: ignore[attr-defined]
        test_client.control = control  # type: ignore[attr-defined]
        yield test_client


def set_live(client_or_rt, *rows: FakeLive) -> None:
    rt = getattr(client_or_rt, "rt", client_or_rt)
    rt.control.rows = list(rows)
    rt.routes.set_live(rows)


# ==========================================================================
# health + state
# ==========================================================================


def test_health_is_cheap_and_does_not_depend_on_a_model(client) -> None:
    """``_reconcile_after_bind`` polls this to learn whether it won the port.

    A health route that consulted systemd or the GPU would make the reconcile
    decision depend on something other than the bind — and on a box where
    nvidia-smi is slow, on nothing at all for ten seconds.
    """
    set_live(client)  # nothing running
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert client.control.calls == []


def test_state_lists_every_registry_model_whether_running_or_not(client) -> None:
    set_live(client, FakeLive(key="small", unit="sd-test-small"))
    state = client.get("/api/state").json()
    assert [m["key"] for m in state["models"]] == ["big", "other", "small"]
    by_key = {m["key"]: m for m in state["models"]}
    assert by_key["small"]["live"] is True and by_key["small"]["ready"] is True
    assert by_key["big"]["live"] is False
    assert by_key["big"]["unit_state"] == "not started"


def test_state_carries_every_field_the_page_renders(client) -> None:
    set_live(client, FakeLive(key="big", unit="sd-test-big"))
    state = client.get("/api/state").json()
    assert set(state) >= {
        "gateway_url", "gpu", "desired", "models", "headroom", "unknown_units", "busy",
    }
    assert state["gateway_url"] == "http://127.0.0.1:8049/v1"
    assert state["gpu"] == {"total_mib": 100000, "free_mib": 42000}
    row = next(m for m in state["models"] if m["key"] == "big")
    assert set(row) >= {
        "key", "id", "aliases", "presets", "slot", "port", "ctx", "live", "ready",
        "unit_state", "restarts", "pid", "uptime_s",
    }
    assert row["aliases"] == ["big-alias"]
    assert row["presets"] == ["big-high"]
    assert row["ctx"] == 32768


def test_state_reports_a_unit_the_registry_does_not_know(client) -> None:
    """A stray ``model-*`` unit is surfaced, never acted on: with no spec there
    is no slot, no port and no way to tell a stray from a model."""
    set_live(client, FakeLive(key="ghost", unit="sd-test-ghost", ready=False, unknown=True))
    state = client.get("/api/state").json()
    assert state["unknown_units"] == [
        {"key": "ghost", "unit": "sd-test-ghost", "unit_state": "active (running)"}
    ]


def test_headroom_says_it_is_unknown_rather_than_estimating(client) -> None:
    """With nothing in the main slot there is no engine to have measured, and
    the panel must say so. A plausible-looking estimate printed where a
    measurement belongs is indistinguishable from one on screen."""
    set_live(client)
    head = client.get("/api/state").json()["headroom"]
    assert head["source"] is None
    assert head["full_context_requests"] is None
    assert "main slot" in head["unavailable"]


def test_headroom_is_computed_by_parallelism_and_labelled_measured(runtime) -> None:
    """The page does no capacity maths (REDESIGN §2.5), so these numbers must
    be the library's, and the source string must name where the pool came
    from."""
    from servedeck import parallelism

    head = _app._headroom(
        main_key="big", main_id="Big-Model", snapshot={"kv_cache_size_tokens": 400_000},
        full_ctx=32768, free_mib=4200,
    )
    assert head["source"] == "measured from the running engine"
    assert head["pool_tokens"] == 400_000
    assert head["full_context_requests"] == parallelism.recommend(
        pool_tokens=400_000, prompt_tokens=32768
    ).n_before_clamp
    assert head["small_requests"] == parallelism.recommend(
        pool_tokens=400_000, prompt_tokens=4096
    ).n_before_clamp
    assert head["small_requests"] > head["full_context_requests"]


def test_headroom_without_a_kv_pool_names_the_missing_metric(runtime) -> None:
    head = _app._headroom(
        main_key="big", main_id="Big-Model", snapshot={}, full_ctx=32768, free_mib=4200
    )
    assert head["full_context_requests"] is None
    assert "cache_config_info" in head["unavailable"]


# ==========================================================================
# Mutations: 202 / 409 / 404
# ==========================================================================


def test_start_returns_202_naming_the_action_and_the_model(client) -> None:
    set_live(client)
    response = client.post("/api/models/small/start")
    assert response.status_code == 202
    assert response.json() == {"accepted": True, "action": "start", "model": "small"}


def test_an_unknown_key_is_404_and_lists_what_it_could_have_said(client) -> None:
    set_live(client)
    response = client.post("/api/models/nope/start")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["reason"] == "unknown_model"
    assert set(error["known"]) == {"big", "other", "small"}
    assert client.control.calls == [], "an unknown key must not reach Control"


def test_starting_something_already_live_is_409_already_live(client) -> None:
    set_live(client, FakeLive(key="small", unit="sd-test-small"))
    response = client.post("/api/models/small/start")
    assert response.status_code == 409
    assert response.json()["error"]["reason"] == "already_live"
    assert client.control.calls == []


def test_starting_a_second_main_model_is_409_and_names_the_holder(client) -> None:
    """The main slot is exclusive. The refusal names what holds it, because
    "switch" is the answer and the operator needs to know what they are
    replacing."""
    set_live(client, FakeLive(key="big", unit="sd-test-big"))
    response = client.post("/api/models/other/start")
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["reason"] == "main_slot_busy"
    assert error["live_key"] == "big"
    assert "/api/switch/other" in error["message"]


def test_a_resident_can_start_while_the_main_slot_is_busy(client) -> None:
    """The exclusivity is the MAIN slot's, not the card's. Refusing a resident
    here would be the "a resident can block a big-model boot" failure (R4)
    inverted, and just as wrong."""
    set_live(client, FakeLive(key="big", unit="sd-test-big"))
    assert client.post("/api/models/small/start").status_code == 202


def test_stopping_something_that_is_not_running_is_409_not_live(client) -> None:
    set_live(client)
    response = client.post("/api/models/small/stop")
    assert response.status_code == 409
    assert response.json()["error"]["reason"] == "not_live"


def test_switching_to_a_resident_is_409_not_main_slot(client) -> None:
    set_live(client)
    response = client.post("/api/switch/small")
    assert response.status_code == 409
    assert response.json()["error"]["reason"] == "not_main_slot"


def test_switch_returns_202(client) -> None:
    set_live(client, FakeLive(key="big", unit="sd-test-big"))
    response = client.post("/api/switch/other")
    assert response.status_code == 202
    assert response.json()["action"] == "switch"


def test_switching_to_the_model_that_already_holds_the_slot_is_409(client) -> None:
    set_live(client, FakeLive(key="big", unit="sd-test-big"))
    assert client.post("/api/switch/big").status_code == 409


def test_adopt_returns_202(client) -> None:
    set_live(client)
    response = client.post("/api/adopt")
    assert response.status_code == 202
    assert response.json()["action"] == "adopt"


def test_a_model_whose_native_ctx_is_unreadable_is_refused_before_launch(
    tmp_path, settings, control
) -> None:
    """``ctx = "native"`` for a checkpoint that is not downloaded.

    Refused at the POST, with the reason, rather than raised inside the worker
    thread — where it would reach the operator only as a notice they had to be
    watching a stream to see.
    """
    path = tmp_path / "native.toml"
    path.write_text(TOML.replace("ctx = 32768", 'ctx = "native"', 1))
    registry = _models.load(path)
    app = _app.create_app(settings, registry=registry, control=control, reconcile=False, poll=False)
    with TestClient(app) as test_client:
        response = test_client.post("/api/models/big/start")
    assert response.status_code == 409
    assert response.json()["error"]["reason"] == "ctx_unresolved"
    assert control.calls == []


# ==========================================================================
# log / wire / doctor / models
# ==========================================================================


def test_log_tails_the_journal_of_the_units_prefix(client) -> None:
    payload = client.get("/api/log/small?lines=3").json()
    assert payload["unit"] == "sd-test-small"
    assert payload["lines"] == ["sd-test-small line 0", "sd-test-small line 1", "sd-test-small line 2"]


def test_log_for_an_unknown_key_is_404(client) -> None:
    assert client.get("/api/log/nope").status_code == 404


def test_log_line_count_is_clamped(client) -> None:
    """An unbounded ``lines`` is a journalctl call that can return a gigabyte."""
    assert len(client.get("/api/log/small?lines=99999").json()["lines"]) == 1000
    assert len(client.get("/api/log/small?lines=0").json()["lines"]) == 1


def test_models_joins_the_registry_with_the_hub_cache(client, monkeypatch) -> None:
    @dataclass
    class Entry:
        repo_id: str
        servable: bool = True
        disk_bytes: int = 3 * 1024**3
        architectures0: str | None = "X"
        reason: str | None = None

    monkeypatch.setattr(_app._discovery, "discover_models", lambda *a, **k: [Entry("org/big")])
    payload = client.get("/api/models").json()
    rows = {m["key"]: m for m in payload["models"]}
    assert rows["big"]["on_disk"] is True
    assert rows["big"]["disk_gib"] == 3.0
    assert rows["small"]["on_disk"] is False
    assert "hub cache" in rows["small"]["reason"]
    assert payload["cache"][0]["in_registry"] is True


def test_wire_is_a_dry_run_until_apply(client, tmp_path, monkeypatch) -> None:
    """The diff shown is literally the diff applied — same code path, one flag
    apart. That is what makes an Apply button trustworthy."""
    target_path = tmp_path / "client.json"

    @dataclass
    class Target:
        name: str = "fake"
        path: Any = None

        def render(self, registry, before, resolve_ctx=None):
            return "generated\n"

    target = Target(path=target_path)
    monkeypatch.setattr(_app._wire, "WIRE_TARGETS", (target,))

    payload = client.get("/api/wire").json()
    assert payload["applied"] is False
    assert payload["targets"][0]["changed"] is True
    assert "generated" in payload["targets"][0]["diff"]
    assert not target_path.exists(), "a dry run must not write"

    applied = client.post("/api/wire/apply").json()
    assert applied["applied"] is True
    assert target_path.read_text() == "generated\n"
    assert client.get("/api/wire").json()["targets"][0]["changed"] is False


def test_doctor_reports_every_check(client, monkeypatch) -> None:
    from servedeck import doctor as _doctor

    monkeypatch.setattr(
        _app._doctor,
        "run_doctor",
        lambda *a, **k: [
            _doctor.CheckResult("a", True, "fine"),
            _doctor.CheckResult("b", False, "broken"),
        ],
    )
    payload = client.get("/api/doctor").json()
    assert payload["ok"] is False
    assert payload["checks"][1] == {"name": "b", "ok": False, "detail": "broken"}


# ==========================================================================
# The gateway is mounted, and mounted LAST
# ==========================================================================


def test_the_gateway_serves_v1_models_from_the_registry(client) -> None:
    set_live(client, FakeLive(key="small", unit="sd-test-small"))
    payload = client.get("/v1/models").json()
    assert [entry["id"] for entry in payload["data"]] == ["Small-Model"]
    assert payload["data"][0]["max_model_len"] == 4096


def test_the_gateway_lists_every_alias_and_preset_of_a_live_model(client) -> None:
    set_live(client, FakeLive(key="big", unit="sd-test-big"))
    ids = [entry["id"] for entry in client.get("/v1/models").json()["data"]]
    assert ids == ["Big-Model", "big-alias", "big-high"]
    assert all(e["root"] == "Big-Model" for e in client.get("/v1/models").json()["data"])


def test_an_unknown_model_through_the_gateway_is_404_not_503(client) -> None:
    set_live(client, FakeLive(key="small", unit="sd-test-small"))
    response = client.post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_a_registered_but_stopped_model_is_503_with_retry_after(client) -> None:
    set_live(client)
    response = client.post("/v1/chat/completions", json={"model": "Big-Model", "messages": []})
    assert response.status_code == 503
    assert response.headers["Retry-After"]


def test_api_routes_are_not_swallowed_by_the_gateways_catch_all(client) -> None:
    """The gateway's ``/v1/{path:path}`` and the page's ``/{asset:path}`` are
    both catch-alls; Starlette matches in registration order. The previous app
    registered its catch-all first and the StaticFiles mount it added later was
    never reached — a dead route that fails silently."""
    assert client.get("/api/health").status_code == 200
    assert client.get("/v1/models").status_code == 200
    assert client.get("/definitely-not-a-file").status_code == 404


# ==========================================================================
# SSE
# ==========================================================================


class FakeRequest:
    def __init__(self, disconnect_after: int = 10_000) -> None:
        self.checks = 0
        self.disconnect_after = disconnect_after

    async def is_disconnected(self) -> bool:
        self.checks += 1
        return self.checks > self.disconnect_after


def parse_frames(chunks: list[bytes]) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for chunk in chunks:
        text = chunk.decode()
        if text.startswith(":"):
            out.append(("keepalive", None))
            continue
        event = text.split("event: ", 1)[1].split("\n", 1)[0]
        data = text.split("data: ", 1)[1].rstrip("\n")
        out.append((event, json.loads(data)))
    return out


@pytest.mark.anyio
async def test_the_stream_opens_with_the_current_state_and_recent_notices(runtime) -> None:
    """A page loaded after everything interesting happened must not be blank
    until the next change."""
    runtime.hub.bind(asyncio.get_running_loop())
    runtime.state = {"models": [], "gpu": {}}
    runtime.hub.publish("notice", {"level": "error", "message": "something broke"})

    frames: list[bytes] = []
    stream = _app._event_stream(runtime, FakeRequest())
    frames.append(await anext(stream))
    frames.append(await anext(stream))
    runtime.hub.close()
    with pytest.raises(StopAsyncIteration):
        await anext(stream)

    parsed = parse_frames(frames)
    assert parsed[0][0] == "state"
    assert parsed[1] == (
        "notice",
        {"level": "error", "message": "something broke", "replay": True},
    )


@pytest.mark.anyio
async def test_replayed_notices_are_flagged_and_live_ones_are_not(runtime) -> None:
    """A `servedeck start` that stopped on a replayed notice would exit 0 the
    instant it connected, reporting the PREVIOUS boot's outcome as this one's.

    Measured, not hypothetical: it made two CLI tests report a stale "ready in
    3s" for a start that had not happened yet. The page still wants the
    backlog, so the fix is a flag rather than dropping it.
    """
    runtime.hub.bind(asyncio.get_running_loop())
    runtime.hub.publish("notice", {"reason": "ready", "key": "big", "message": "an hour ago"})

    stream = _app._event_stream(runtime, FakeRequest())
    replayed = parse_frames([await anext(stream)])[0]
    assert replayed[0] == "notice"
    assert replayed[1]["replay"] is True

    task = asyncio.ensure_future(anext(stream))
    await asyncio.sleep(0)
    runtime.hub.publish("notice", {"reason": "ready", "key": "big", "message": "just now"})
    live_notice = parse_frames([await asyncio.wait_for(task, timeout=2)])[0]
    assert live_notice[1]["message"] == "just now"
    assert "replay" not in live_notice[1], "a live notice must not look like a replay"
    runtime.hub.close()


@pytest.mark.anyio
async def test_progress_and_notice_frames_reach_a_subscriber(runtime) -> None:
    runtime.hub.bind(asyncio.get_running_loop())
    stream = _app._event_stream(runtime, FakeRequest())
    task = asyncio.ensure_future(anext(stream))
    await asyncio.sleep(0)
    runtime.hub.publish("progress", {"key": "big", "kind": "marker", "text": "Loading weights took 4s"})
    frame = await asyncio.wait_for(task, timeout=2)
    event, data = parse_frames([frame])[0]
    assert event == "progress"
    assert data["kind"] == "marker" and data["key"] == "big"
    runtime.hub.close()


@pytest.mark.anyio
async def test_shutdown_closes_every_subscriber(runtime) -> None:
    """The ``TimeoutStopSec`` fix, stated as the thing that was wrong.

    Each open page used to hold a coroutine parked on ``queue.get()`` that
    nothing would ever complete; uvicorn waited for them at shutdown and
    systemd SIGKILLed the unit 15 s later, every single stop. The sentinel is
    what makes the generator return.
    """
    runtime.hub.bind(asyncio.get_running_loop())
    streams = [_app._event_stream(runtime, FakeRequest()) for _ in range(3)]
    tasks = [asyncio.ensure_future(anext(s)) for s in streams]
    await asyncio.sleep(0)
    assert runtime.hub.subscriber_count == 3

    runtime.hub.close()
    done, pending = await asyncio.wait(tasks, timeout=1.0)
    assert not pending, "a subscriber did not return within a second of close()"
    for task in done:
        with pytest.raises(StopAsyncIteration):
            task.result()


@pytest.mark.anyio
async def test_a_slow_subscriber_loses_its_oldest_events_not_the_publisher(runtime) -> None:
    """Drop-oldest, and never block. A page that stopped reading must not stall
    the poller that is trying to publish to every other page."""
    runtime.hub.bind(asyncio.get_running_loop())
    sub = runtime.hub.subscribe()
    for i in range(_app.QUEUE_MAXSIZE + 10):
        runtime.hub.publish("notice", {"n": i})
    assert sub.queue.qsize() == _app.QUEUE_MAXSIZE
    assert sub.dropped == 10
    first = sub.queue.get_nowait()
    assert first["data"]["n"] == 10, "the newest events must survive, not the oldest"


@pytest.mark.anyio
async def test_the_stream_ends_when_the_client_goes_away(runtime) -> None:
    runtime.hub.bind(asyncio.get_running_loop())
    stream = _app._event_stream(runtime, FakeRequest(disconnect_after=0))
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert runtime.hub.subscriber_count == 0, "the subscriber was not unregistered"


def test_publishing_from_a_worker_thread_reaches_the_loop(runtime) -> None:
    """``control.start``'s progress callback fires in a worker thread.

    ``asyncio.Queue`` is not thread-safe and the failure that mistake produces
    is a lost event, not an exception — which is why the hop to the loop is
    written out rather than assumed.
    """

    async def scenario() -> list[bytes]:
        runtime.hub.bind(asyncio.get_running_loop())
        stream = _app._event_stream(runtime, FakeRequest())
        task = asyncio.ensure_future(anext(stream))
        await asyncio.sleep(0)
        await asyncio.to_thread(
            _app._progress_publisher(runtime.hub, "big"),
            _control.Progress(kind="line", text="from another thread"),
        )
        frame = await asyncio.wait_for(task, timeout=2)
        runtime.hub.close()
        return [frame]

    event, data = parse_frames(asyncio.run(scenario()))[0]
    assert event == "progress"
    assert data["text"] == "from another thread"


# ==========================================================================
# Reconcile only after the listen port answers  (REDESIGN §4 R4)
# ==========================================================================


@dataclass
class BindRecorder:
    """One ordered log of health probes and reconciles.

    The assertion that matters is not "reconcile happened" but "reconcile
    happened AFTER a 200" — a boolean flag could not tell those apart, and the
    bug being pinned is purely one of order.
    """

    events: list[str] = field(default_factory=list)
    answer_after: int = 3
    probes: int = 0

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.probes += 1
            if self.probes <= self.answer_after:
                self.events.append("port-refused")
                raise httpx.ConnectError("connection refused", request=request)
            self.events.append("port-answered")
            return httpx.Response(200, json={"ok": True})

        return httpx.MockTransport(handler)


@pytest.fixture
def bind_runtime(settings, registry, control):
    def make(recorder: BindRecorder):
        rt = _app.build_runtime(
            settings,
            registry=registry,
            control=control,
            client=httpx.AsyncClient(transport=recorder.transport()),
        )
        original = control.reconcile

        def recording_reconcile(*args, **kwargs):
            recorder.events.append("reconcile")
            return original(*args, **kwargs)

        control.reconcile = recording_reconcile  # type: ignore[method-assign]
        return rt

    return make


@pytest.mark.anyio
async def test_reconcile_waits_for_the_listen_port(bind_runtime) -> None:
    """The 70-boots-in-9-minutes bug (REDESIGN §4 R4), as an ordering test.

    uvicorn runs the ASGI lifespan BEFORE it binds. A reconcile called from the
    lifespan therefore runs in the process that is about to die with "address
    already in use" — and under Restart=on-failure that turned one lost race
    into an unbounded loop with a real vLLM launch on every lap.
    """
    recorder = BindRecorder(answer_after=3)
    rt = bind_runtime(recorder)
    rt.hub.bind(asyncio.get_running_loop())

    result = await _app._reconcile_after_bind(rt, timeout_s=5.0, interval_s=0.001)

    assert result is not None
    assert recorder.events == ["port-refused"] * 3 + ["port-answered", "reconcile"]
    assert recorder.events.index("port-answered") < recorder.events.index("reconcile")
    assert "reconcile" not in recorder.events[: recorder.events.index("port-answered")]


@pytest.mark.anyio
async def test_nothing_is_started_when_the_port_never_answers(bind_runtime) -> None:
    """If the bind never succeeds, no model is ever started. Refusing to act is
    the correct failure for a dashboard that could not start itself."""
    recorder = BindRecorder(answer_after=10_000)
    rt = bind_runtime(recorder)
    rt.hub.bind(asyncio.get_running_loop())

    result = await _app._reconcile_after_bind(rt, timeout_s=0.05, interval_s=0.001)

    assert result is None
    assert "reconcile" not in recorder.events
    assert recorder.probes > 0, "it must actually have tried"
    reasons = [n["data"]["reason"] for n in rt.hub.notices]
    assert "bind_timeout" in reasons


@pytest.mark.anyio
async def test_reconcile_is_handed_the_desired_state_from_disk(bind_runtime, settings) -> None:
    """Reconcile must act on what an operator last asked for, not on what is
    running: a model that crashed has to stay desired so it comes back."""
    from servedeck import desired as _desired

    settings.state_dir.mkdir(parents=True, exist_ok=True)
    _desired.save(_desired.Desired(main="big", residents=["small"]), settings.desired_path)

    recorder = BindRecorder(answer_after=0)
    rt = bind_runtime(recorder)
    rt.hub.bind(asyncio.get_running_loop())
    await _app._reconcile_after_bind(rt, timeout_s=5.0, interval_s=0.001)

    want = rt.control.reconcile_arg
    assert want.main == "big"
    assert want.residents == ["small"]


@pytest.mark.anyio
async def test_reconcile_runs_exactly_once(bind_runtime) -> None:
    """Once per process. A reconcile that ran on every health poll would be the
    boot loop again, with a slower clock."""
    recorder = BindRecorder(answer_after=1)
    rt = bind_runtime(recorder)
    rt.hub.bind(asyncio.get_running_loop())
    await _app._reconcile_after_bind(rt, timeout_s=5.0, interval_s=0.001)
    assert recorder.events.count("reconcile") == 1


# ==========================================================================
# Mutations report over SSE
# ==========================================================================


@pytest.mark.anyio
async def test_a_refusal_from_control_becomes_an_error_notice(runtime) -> None:
    """A refusal decided against reality (a race the precheck could not see)
    must reach the operator with the same vocabulary as one decided from the
    snapshot."""
    runtime.hub.bind(asyncio.get_running_loop())
    runtime.control.results["start"] = _control.Refusal(
        reason="not_enough_vram", message="only 12 MiB free", key="big"
    )
    await _app._run_mutation(runtime, "start big", lambda: runtime.control.start("big"))
    notice = [n["data"] for n in runtime.hub.notices if n["data"]["reason"] == "not_enough_vram"]
    assert notice and notice[0]["level"] == "error"
    assert "only 12 MiB free" in notice[0]["message"]


@pytest.mark.anyio
async def test_a_failed_boot_carries_its_journal(runtime) -> None:
    """``--collect`` deletes a failed unit and ``systemctl show`` then reports
    property defaults that read as a clean stop. The journal outlives the unit,
    which is why a failure report is built from it."""
    runtime.hub.bind(asyncio.get_running_loop())
    runtime.control.results["start"] = _control.StartResult(
        key="big", unit="sd-test-big", ready=False, elapsed_s=9.0,
        failure="sd-test-big no longer exists", journal=["CUDA out of memory"],
    )
    await _app._run_mutation(runtime, "start big", lambda: runtime.control.start("big"))
    notice = [n["data"] for n in runtime.hub.notices if n["data"]["reason"] == "boot_failed"][0]
    assert notice["journal"] == ["CUDA out of memory"]
    assert notice["level"] == "error"


@pytest.mark.anyio
async def test_two_mutations_cannot_interleave(runtime) -> None:
    """A switch is between its stop and its start for a while, and in that
    window the card is empty and every precheck would say "go ahead". One lock
    makes the main slot's exclusivity true of the API and not only of the GPU.
    """
    runtime.hub.bind(asyncio.get_running_loop())
    order: list[str] = []

    def slow(name: str):
        def work():
            order.append(f"{name}-in")
            import time as _time

            _time.sleep(0.05)
            order.append(f"{name}-out")
            return _control.StopResult(key=name, unit=f"u-{name}", was_live=True)

        return work

    await asyncio.gather(
        _app._run_mutation(runtime, "a", slow("a")),
        _app._run_mutation(runtime, "b", slow("b")),
    )
    assert order in (
        ["a-in", "a-out", "b-in", "b-out"],
        ["b-in", "b-out", "a-in", "a-out"],
    ), order


def test_a_mutation_in_flight_refuses_the_next_one(client) -> None:
    set_live(client)
    client.rt.busy = "switch big"
    response = client.post("/api/models/small/start")
    assert response.status_code == 409
    assert response.json()["error"]["reason"] == "busy"


# ==========================================================================
# State change detection
# ==========================================================================


def test_only_a_real_change_publishes_a_state_frame() -> None:
    """``generated_at`` and the throughput figures move on every poll. Pushing
    on those would make "on change" mean "every two seconds", and the page
    would re-render continuously."""
    base = {
        "generated_at": 1.0,
        "models": [{"key": "a", "ready": True, "uptime_s": 5, "metrics": {"gen_tok_s": 10.0, "running": 1}}],
    }
    later = json.loads(json.dumps(base))
    later["generated_at"] = 99.0
    later["models"][0]["uptime_s"] = 400
    later["models"][0]["metrics"]["gen_tok_s"] = 55.5
    assert _app._state_changed(base, later) is False

    real = json.loads(json.dumps(later))
    real["models"][0]["ready"] = False
    assert _app._state_changed(base, real) is True

    busier = json.loads(json.dumps(later))
    busier["models"][0]["metrics"]["running"] = 4
    assert _app._state_changed(base, busier) is True
