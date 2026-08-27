"""Tests for coldstart.gateway — SPEC.md §7, corrected by the 2026-08-27
addendum (C2 per-path policy, C7 disconnect propagation, C8 finite hold).

Every test here drives the module against a stubbed supervisor (a plain
object satisfying gateway.SupervisorView) and a stubbed upstream
(httpx.MockTransport) — SPEC's own verification instruction: "unit-test
the state matrix with a stubbed upstream (do not proxy to the live :8001
server for destructive cases)". No network is used anywhere in this file.

Handlers are exercised directly (``gateway._handle_hold_eligible`` /
``gateway._handle_pass_through``) against hand-built Starlette Requests
rather than through TestClient. That's a deliberate choice, not a
shortcut: TestClient runs the ASGI app on a separate portal thread, and
several of these tests (park-then-wake, park-detects-FAILED-mid-hold,
shed-while-another-request-is-parked) need to flip supervisor state from
the *same* asyncio loop that's awaiting inside the handler — asyncio.Event
isn't thread-safe to .set() across loops without call_soon_threadsafe
gymnastics that would test the harness, not the gateway. Calling the
handler coroutines directly keeps everything on one loop and is exactly
as much "the real code" as going through the router, since the router's
own dispatch (`v1_dispatch`) is a two-line method/path check covered
separately in test_router_dispatches_hold_eligible_paths_only.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from starlette.requests import Request

from coldstart import gateway

# ---------------------------------------------------------------------------
# Request builder — no ASGI server involved, just a scope + receive callable.
# ---------------------------------------------------------------------------


def make_request(method: str = "GET", path: str = "/v1/models", body: bytes = b"", headers: dict | None = None) -> Request:
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": raw_headers,
        "http_version": "1.1",
        "scheme": "http",
        "server": ("127.0.0.1", 8010),
        "client": ("127.0.0.1", 12345),
    }
    state = {"sent": False}

    async def receive():
        if state["sent"]:
            return {"type": "http.disconnect"}
        state["sent"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


# ---------------------------------------------------------------------------
# Stub supervisor — satisfies gateway.SupervisorView structurally.
# ---------------------------------------------------------------------------


class StubSupervisor:
    def __init__(self, *, desired: str = gateway.DESIRED_RUNNING, actual: str = gateway.ACTUAL_READY):
        self._desired = desired
        self._actual = actual
        self.ready_event = asyncio.Event()
        if actual == gateway.ACTUAL_READY:
            self.ready_event.set()
        self._failure_code: str | None = None
        self._failure_detail: str | None = None
        self._hold = gateway.HoldStatus(
            phase_code="loading_weights", phase_label="Loading weights", eta_s=190, attempt=1
        )

    def desired_state(self) -> str:
        return self._desired

    def actual_state(self) -> str:
        return self._actual

    def upstream_base_url(self) -> str:
        return "http://upstream.invalid"

    def hold_status(self) -> gateway.HoldStatus:
        return self._hold

    def failure_code(self) -> str | None:
        return self._failure_code

    def failure_detail(self) -> str | None:
        return self._failure_detail

    # test helpers, not part of the protocol
    def set_actual(self, actual: str) -> None:
        self._actual = actual
        if actual == gateway.ACTUAL_READY:
            self.ready_event.set()
        else:
            self.ready_event.clear()

    def set_desired(self, desired: str) -> None:
        self._desired = desired

    def set_failure(self, code: str, detail: str) -> None:
        self._failure_code = code
        self._failure_detail = detail


assert isinstance(StubSupervisor(), gateway.SupervisorView)


# ---------------------------------------------------------------------------
# Mock transports
# ---------------------------------------------------------------------------


class CountingOkTransport(httpx.AsyncBaseTransport):
    """Every request gets a fixed 200 JSON body. Records call count and the
    last request seen (for header-forwarding assertions)."""

    def __init__(self, body: bytes = b'{"ok": true}'):
        self.calls = 0
        self.last_request: httpx.Request | None = None
        self._body = body

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.last_request = request

        async def _one_chunk():
            yield self._body

        return httpx.Response(
            200,
            headers={"content-type": "application/json", "connection": "keep-alive"},
            content=_one_chunk(),
        )


class RefusingTransport(httpx.AsyncBaseTransport):
    """Simulates "nothing is listening" — every request raises a connect error."""

    def __init__(self):
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        raise httpx.ConnectError("Connection refused", request=request)


def client_for(transport: httpx.AsyncBaseTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(connect=5.0, read=None, write=None, pool=None))


# ---------------------------------------------------------------------------
# READY / UNMANAGED — transparent stream proxy
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_ready_streams_through():
    supervisor = StubSupervisor(actual=gateway.ACTUAL_READY)
    transport = CountingOkTransport()
    client = client_for(transport)
    runtime = gateway.GatewayRuntime(max_parked=64)
    try:
        request = make_request(method="POST", path="/v1/chat/completions", body=b'{"model":"m"}')
        resp = await gateway._handle_hold_eligible(request, supervisor, client, runtime, hold_max_s=5.0, park_poll_s=0.05)
        assert resp.status_code == 200
        body = b"".join([chunk async for chunk in resp.body_iterator])
        assert json.loads(body) == {"ok": True}
        assert transport.calls == 1
        assert runtime.parked_count == 0  # never parked
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_unmanaged_streams_through_without_parking():
    supervisor = StubSupervisor(desired=gateway.DESIRED_RUNNING, actual=gateway.ACTUAL_UNMANAGED)
    transport = CountingOkTransport()
    client = client_for(transport)
    runtime = gateway.GatewayRuntime(max_parked=64)
    try:
        request = make_request(method="POST", path="/v1/responses", body=b"{}")
        resp = await gateway._handle_hold_eligible(request, supervisor, client, runtime, hold_max_s=5.0, park_poll_s=0.05)
        assert resp.status_code == 200
        assert transport.calls == 1
        assert runtime.parked_count == 0
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# desired == STOPPED -> immediate 503, never park, never touch transport
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_desired_stopped_is_immediate_503_never_parks():
    # actual is deliberately still READY, to prove the desired==STOPPED
    # check wins the race described in _handle_hold_eligible's docstring
    # (stop() sets desired_state before signalling).
    supervisor = StubSupervisor(desired=gateway.DESIRED_STOPPED, actual=gateway.ACTUAL_READY)
    transport = CountingOkTransport()
    client = client_for(transport)
    runtime = gateway.GatewayRuntime(max_parked=64)
    try:
        request = make_request(method="POST", path="/v1/chat/completions", body=b"{}")
        resp = await gateway._handle_hold_eligible(request, supervisor, client, runtime, hold_max_s=5.0, park_poll_s=0.05)
        assert resp.status_code == 503
        assert resp.headers["retry-after"] == "5"
        body = json.loads(bytes(resp.body))
        assert body["error"]["code"] == "stopped"
        assert body["error"]["type"] == "coldstart_upstream_unavailable"
        assert transport.calls == 0
        assert runtime.parked_count == 0
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# actual == FAILED -> immediate 503 with classified code, never park
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_failed_is_immediate_503_with_classified_code():
    supervisor = StubSupervisor(desired=gateway.DESIRED_RUNNING, actual=gateway.ACTUAL_FAILED)
    supervisor.set_failure("KV_TOO_SMALL", "estimated maximum model length is 131072")
    transport = CountingOkTransport()
    client = client_for(transport)
    runtime = gateway.GatewayRuntime(max_parked=64)
    try:
        request = make_request(method="POST", path="/v1/chat/completions", body=b"{}")
        resp = await gateway._handle_hold_eligible(request, supervisor, client, runtime, hold_max_s=5.0, park_poll_s=0.05)
        assert resp.status_code == 503
        body = json.loads(bytes(resp.body))
        assert body["error"]["code"] == "kv_too_small"
        assert "estimated maximum model length is 131072" in body["error"]["message"]
        assert transport.calls == 0
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_failed_with_broken_failure_accessors_still_returns_503():
    """A supervisor whose failure_code()/failure_detail() raise must not
    take the whole response path down with it (_safe() wrapping)."""

    class BrokenFailureSupervisor(StubSupervisor):
        def failure_code(self):
            raise RuntimeError("not wired yet")

        def failure_detail(self):
            raise RuntimeError("not wired yet")

    supervisor = BrokenFailureSupervisor(desired=gateway.DESIRED_RUNNING, actual=gateway.ACTUAL_FAILED)
    transport = CountingOkTransport()
    client = client_for(transport)
    runtime = gateway.GatewayRuntime(max_parked=64)
    try:
        request = make_request(method="POST", path="/v1/responses", body=b"{}")
        resp = await gateway._handle_hold_eligible(request, supervisor, client, runtime, hold_max_s=5.0, park_poll_s=0.05)
        assert resp.status_code == 503
        body = json.loads(bytes(resp.body))
        assert body["error"]["code"] == "failed"
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# STARTING/PREFLIGHT/DRAINING/STOPPING + desired==RUNNING -> PARK
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "actual_during_hold",
    [gateway.ACTUAL_STARTING, gateway.ACTUAL_PREFLIGHT, gateway.ACTUAL_DRAINING, gateway.ACTUAL_STOPPING],
)
@pytest.mark.anyio
async def test_park_then_ready_streams_through(actual_during_hold):
    supervisor = StubSupervisor(desired=gateway.DESIRED_RUNNING, actual=actual_during_hold)
    transport = CountingOkTransport()
    client = client_for(transport)
    runtime = gateway.GatewayRuntime(max_parked=64)
    try:
        request = make_request(method="POST", path="/v1/chat/completions", body=b"{}")
        task = asyncio.create_task(
            gateway._handle_hold_eligible(request, supervisor, client, runtime, hold_max_s=5.0, park_poll_s=0.02)
        )
        await asyncio.sleep(0.05)
        assert runtime.parked_count == 1  # confirms it actually parked, not raced ahead
        assert transport.calls == 0
        supervisor.set_actual(gateway.ACTUAL_READY)  # also sets ready_event
        resp = await asyncio.wait_for(task, timeout=2.0)
        assert resp.status_code == 200
        assert transport.calls == 1
        assert runtime.parked_count == 0  # slot released
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_park_timeout_returns_503_restarting_with_eta():
    supervisor = StubSupervisor(desired=gateway.DESIRED_RUNNING, actual=gateway.ACTUAL_STARTING)
    transport = CountingOkTransport()
    client = client_for(transport)
    runtime = gateway.GatewayRuntime(max_parked=64)
    try:
        request = make_request(method="POST", path="/v1/chat/completions", body=b"{}")
        loop = asyncio.get_event_loop()
        start = loop.time()
        resp = await gateway._handle_hold_eligible(request, supervisor, client, runtime, hold_max_s=0.2, park_poll_s=0.05)
        elapsed = loop.time() - start
        assert resp.status_code == 503
        assert elapsed < 1.0  # bounded near hold_max_s, not left hanging
        body = json.loads(bytes(resp.body))
        assert body["error"]["code"] == "restarting"
        assert body["error"]["coldstart"]["phase"] == "loading_weights"
        assert body["error"]["coldstart"]["eta_s"] == 190
        assert "3m10s" in body["error"]["message"]
        assert transport.calls == 0
        assert runtime.parked_count == 0  # slot released even on timeout
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_park_notices_failed_transition_before_full_ceiling():
    """SPEC corrections C8: 'Never hold when reached_ready was false.' A
    boot that fails mid-hold must not keep this request parked for the
    remainder of hold_max_s."""
    supervisor = StubSupervisor(desired=gateway.DESIRED_RUNNING, actual=gateway.ACTUAL_STARTING)
    transport = CountingOkTransport()
    client = client_for(transport)
    runtime = gateway.GatewayRuntime(max_parked=64)
    try:
        request = make_request(method="POST", path="/v1/chat/completions", body=b"{}")
        task = asyncio.create_task(
            gateway._handle_hold_eligible(request, supervisor, client, runtime, hold_max_s=30.0, park_poll_s=0.02)
        )
        await asyncio.sleep(0.05)
        supervisor.set_actual(gateway.ACTUAL_FAILED)
        supervisor.set_failure("RUNTIME_OOM", "CUDA out of memory")
        resp = await asyncio.wait_for(task, timeout=1.0)  # must NOT take anywhere near 30s
        assert resp.status_code == 503
        body = json.loads(bytes(resp.body))
        assert body["error"]["code"] == "runtime_oom"
        assert transport.calls == 0
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_park_notices_stopped_transition_before_full_ceiling():
    supervisor = StubSupervisor(desired=gateway.DESIRED_RUNNING, actual=gateway.ACTUAL_STARTING)
    transport = CountingOkTransport()
    client = client_for(transport)
    runtime = gateway.GatewayRuntime(max_parked=64)
    try:
        request = make_request(method="POST", path="/v1/responses", body=b"{}")
        task = asyncio.create_task(
            gateway._handle_hold_eligible(request, supervisor, client, runtime, hold_max_s=30.0, park_poll_s=0.02)
        )
        await asyncio.sleep(0.05)
        supervisor.set_desired(gateway.DESIRED_STOPPED)  # user clicked Stop while we were parked
        resp = await asyncio.wait_for(task, timeout=1.0)
        assert resp.status_code == 503
        body = json.loads(bytes(resp.body))
        assert body["error"]["code"] == "stopped"
        assert transport.calls == 0
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# max_parked shedding
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_max_parked_sheds_with_503_and_releases_after():
    supervisor = StubSupervisor(desired=gateway.DESIRED_RUNNING, actual=gateway.ACTUAL_STARTING)
    transport = CountingOkTransport()
    client = client_for(transport)
    runtime = gateway.GatewayRuntime(max_parked=1)
    try:
        first_req = make_request(method="POST", path="/v1/chat/completions", body=b"{}")
        first_task = asyncio.create_task(
            gateway._handle_hold_eligible(first_req, supervisor, client, runtime, hold_max_s=5.0, park_poll_s=0.02)
        )
        await asyncio.sleep(0.05)
        assert runtime.parked_count == 1

        second_req = make_request(method="POST", path="/v1/chat/completions", body=b"{}")
        shed_resp = await gateway._handle_hold_eligible(
            second_req, supervisor, client, runtime, hold_max_s=5.0, park_poll_s=0.02
        )
        assert shed_resp.status_code == 503
        shed_body = json.loads(bytes(shed_resp.body))
        assert shed_body["error"]["code"] == "queue_full"
        assert shed_body["error"]["coldstart"]["parked"] == 1
        assert runtime.parked_count == 1  # the shed request never occupied a slot

        supervisor.set_actual(gateway.ACTUAL_READY)
        first_resp = await asyncio.wait_for(first_task, timeout=2.0)
        assert first_resp.status_code == 200
        assert runtime.parked_count == 0
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Pass-through paths — SPEC corrections C2: never consult state, never hold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("actual", [gateway.ACTUAL_FAILED, gateway.ACTUAL_STOPPED, gateway.ACTUAL_STARTING])
@pytest.mark.anyio
async def test_get_models_passes_through_ignoring_state(actual):
    """C2: GET /v1/models must NEVER be held and NEVER synthesized —
    it's a plain pass-through regardless of what actual_state says."""
    supervisor = StubSupervisor(desired=gateway.DESIRED_STOPPED, actual=actual)
    transport = CountingOkTransport(body=b'{"object":"list","data":[{"id":"m"}]}')
    client = client_for(transport)
    request = make_request(method="GET", path="/v1/models")
    resp = await gateway._handle_pass_through(request, supervisor, client)
    assert resp.status_code == 200
    assert transport.calls == 1


@pytest.mark.anyio
async def test_passthrough_503_when_upstream_down_and_never_retries():
    supervisor = StubSupervisor(desired=gateway.DESIRED_RUNNING, actual=gateway.ACTUAL_READY)
    transport = RefusingTransport()
    client = client_for(transport)
    request = make_request(method="GET", path="/health")
    resp = await gateway._handle_pass_through(request, supervisor, client)
    assert resp.status_code == 503
    body = json.loads(bytes(resp.body))
    assert body["error"]["code"] == "upstream_unreachable"
    assert transport.calls == 1  # exactly one attempt — no internal retry


def test_filter_headers_strips_hop_by_hop_and_configured_extras():
    items = [
        ("Host", "should-be-dropped"),
        ("Connection", "keep-alive"),
        ("Transfer-Encoding", "chunked"),
        ("Authorization", "Bearer x"),
        ("Content-Type", "application/json"),
    ]
    out = gateway._filter_headers(items, gateway._REQUEST_STRIP_HEADERS)
    assert out == {"Authorization": "Bearer x", "Content-Type": "application/json"}

    out_response_side = gateway._filter_headers(items, gateway._HOP_BY_HOP_HEADERS)
    # response-side stripping does NOT strip "host" — that's a request-only concern.
    assert out_response_side == {
        "Host": "should-be-dropped",
        "Authorization": "Bearer x",
        "Content-Type": "application/json",
    }


@pytest.mark.anyio
async def test_passthrough_does_not_leak_the_incoming_host_header():
    # httpx respects an explicitly-set Host header verbatim (verified
    # separately) — so if the gateway forwarded the client's Host as-is,
    # upstream would see "should-be-dropped" instead of its own address.
    # This is therefore a real end-to-end check of _REQUEST_STRIP_HEADERS,
    # not just of _filter_headers in isolation.
    supervisor = StubSupervisor()
    transport = CountingOkTransport()
    client = client_for(transport)
    request = make_request(
        method="GET",
        path="/v1/models",
        headers={"host": "should-be-dropped", "authorization": "Bearer x"},
    )
    resp = await gateway._handle_pass_through(request, supervisor, client)
    assert resp.status_code == 200
    sent = transport.last_request
    assert sent is not None
    assert sent.headers["host"] == "upstream.invalid"
    assert sent.headers["authorization"] == "Bearer x"


# ---------------------------------------------------------------------------
# Router wiring — the two-line method/path check that picks hold-eligible
# vs pass-through is exercised for real here (build_gateway_router itself),
# everything above tests the handlers it dispatches to.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_router_dispatches_hold_eligible_paths_only():
    supervisor = StubSupervisor(desired=gateway.DESIRED_STOPPED, actual=gateway.ACTUAL_STOPPED)
    transport = CountingOkTransport()
    router, client, runtime = gateway.build_gateway_router(supervisor, transport=transport, hold_max_s=1.0)
    try:
        route_map = {(r.methods and tuple(sorted(r.methods)), r.path): r for r in router.routes}
        v1_route = next(r for r in router.routes if r.path == "/v1/{full_path:path}")

        # POST /v1/chat/completions is hold-eligible: desired==STOPPED -> 503, transport untouched.
        req = make_request(method="POST", path="/v1/chat/completions", body=b"{}")
        resp = await v1_route.endpoint(full_path="chat/completions", request=req)
        assert resp.status_code == 503
        assert transport.calls == 0

        # GET /v1/models is NOT hold-eligible: passes straight through even though STOPPED.
        req2 = make_request(method="GET", path="/v1/models")
        resp2 = await v1_route.endpoint(full_path="models", request=req2)
        assert resp2.status_code == 200
        assert transport.calls == 1

        # POST /v1/embeddings is not one of the two hold-eligible sub-paths either.
        req3 = make_request(method="POST", path="/v1/embeddings", body=b"{}")
        resp3 = await v1_route.endpoint(full_path="embeddings", request=req3)
        assert resp3.status_code == 200
        assert transport.calls == 2
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_fixed_passthrough_paths_registered():
    supervisor = StubSupervisor()
    transport = CountingOkTransport()
    router, client, runtime = gateway.build_gateway_router(supervisor, transport=transport)
    try:
        registered = {r.path for r in router.routes}
        for p in gateway._FIXED_PASSTHROUGH_PATHS:
            assert p in registered
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# C7 — client-disconnect propagation: abandoning the response's body
# generator early must close the upstream response, not leave it dangling
# for a --max-num-seqs=1 backend to stay stuck on.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_abandoning_stream_closes_upstream_connection():
    close_calls = {"n": 0}

    class TrackedResponse(httpx.Response):
        async def aclose(self) -> None:
            close_calls["n"] += 1
            await super().aclose()

    async def body_gen():
        for i in range(5):
            yield f"chunk{i}".encode()
            await asyncio.sleep(0.01)

    class TrackingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return TrackedResponse(200, content=body_gen())

    supervisor = StubSupervisor()
    client = client_for(TrackingTransport())
    try:
        request = make_request(method="GET", path="/v1/models")
        resp = await gateway._stream_proxy(client, supervisor.upstream_base_url(), request)
        gen = resp.body_iterator
        first = await gen.__anext__()
        assert first == b"chunk0"
        assert close_calls["n"] == 0  # not closed yet — still mid-stream
        await gen.aclose()  # simulates Starlette tearing the task down on client disconnect
        assert close_calls["n"] == 1  # our try/finally ran upstream.aclose() exactly once
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_full_stream_consumption_still_closes_upstream_once():
    close_calls = {"n": 0}

    class TrackedResponse(httpx.Response):
        async def aclose(self) -> None:
            close_calls["n"] += 1
            await super().aclose()

    async def body_gen():
        yield b"only-chunk"

    class TrackingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return TrackedResponse(200, content=body_gen())

    supervisor = StubSupervisor()
    client = client_for(TrackingTransport())
    try:
        request = make_request(method="GET", path="/v1/models")
        resp = await gateway._stream_proxy(client, supervisor.upstream_base_url(), request)
        chunks = [c async for c in resp.body_iterator]
        assert chunks == [b"only-chunk"]
        # httpx itself auto-closes a response once its stream is fully
        # read, on top of our own try/finally's aclose() call — two calls
        # is the expected, harmless outcome (httpx.Response.aclose() is
        # idempotent, guarded by `if not self.is_closed`), not a bug.
        assert close_calls["n"] >= 1
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# mount_gateway wiring
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_mount_gateway_exposes_runtime_on_app_state():
    from fastapi import FastAPI

    supervisor = StubSupervisor()
    transport = CountingOkTransport()
    app = FastAPI()
    runtime = gateway.mount_gateway(app, supervisor, transport=transport)
    assert app.state.gateway is runtime
    assert app.state.gateway_client is not None
    assert isinstance(runtime, gateway.GatewayRuntime)
    # shutdown handler must close the client without raising
    for handler in app.router.on_shutdown:
        await handler()


# ---------------------------------------------------------------------------
# anyio backend selection — asyncio only, no trio dependency in this repo
# ---------------------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"
