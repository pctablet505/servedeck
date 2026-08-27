"""Coldstart gateway — SPEC.md §7, corrected by the 2026-08-27 addendum
(C2, C7, C8 specifically).

This module is a transparent reverse proxy from Coldstart's own listener
(127.0.0.1:8010, bound by whoever owns app.py/run.sh — not this file) to
whichever upstream vLLM server is currently configured (127.0.0.1:8001 for
flashnext, 127.0.0.1:8000 for inline). Two things make it more than a dumb
proxy:

1. Two endpoints — POST /v1/chat/completions and POST /v1/responses — may
   be *held* ("parked") for up to ``hold_max_s`` while the backend is
   mid-restart, so a Codex turn issued during a reboot doesn't just fail;
   it waits and then proceeds once the backend is READY. Every other path
   is plain, unconditional pass-through: connect now, 503 if that fails.
   This split is corrections addendum C2, and it OVERRIDES the plain
   text of SPEC §7, which said to park all of ``/v1/*`` — that would hang
   ``codex-qwen.sh``'s ``is_server_up()`` (an un-timed ``curl`` against
   GET /v1/models) forever. See ``_HOLD_ELIGIBLE_SUBPATHS`` below.
2. THE HARD RULE (SPEC §7, restated by C7): once any upstream byte has
   reached the client, this module never retries internally, and it
   propagates a client disconnect to the upstream connection (vLLM's
   ``/v1/responses`` wraps generation in ``with_cancellation`` — closing
   the *upstream* socket is what actually aborts an in-flight generation,
   which matters a great deal at ``--max-num-seqs 1``: one leaked
   generation holds the only sequence slot and starves every other
   request). See ``_stream_proxy``'s ``try/finally: await upstream.aclose()``
   for how that propagation happens — Starlette cancels the streaming
   task when it detects the downstream client is gone, which raises
   inside our body generator at whatever await it is suspended on, and
   the ``finally`` runs from there.

--------------------------------------------------------------------------
INTEGRATION CONTRACT — read this if you are wiring app.py / supervisor.py
--------------------------------------------------------------------------
coldstart/supervisor.py (SPEC §6) is owned by a different agent and did
not exist yet when this file was written. Rather than import it (and
either break at import time or freeze this file to a guess at its
exact shape), this module depends on it only *structurally*, through the
``SupervisorView`` Protocol below. Anything — a class instance, or a bare
module (``import coldstart.supervisor as supervisor_mod``; modules satisfy
Protocols too, since this is duck typing) — that exposes:

  desired_state()      -> "STOPPED" | "RUNNING"
  actual_state()        -> "STOPPED"|"PREFLIGHT"|"STARTING"|"READY"|
                            "DRAINING"|"STOPPING"|"FAILED"|"UNMANAGED"
  upstream_base_url()   -> str, e.g. "http://127.0.0.1:8001" (no path)
  ready_event            : asyncio.Event, set exactly while actual_state
                            == "READY" (supervisor's job to keep this in
                            sync with its own state transitions)
  hold_status()          -> HoldStatus  (best-effort; used only to build
                            the human-readable / machine-readable parts
                            of a park-timeout or shed error body)
  failure_code()         -> str | None  (set when actual_state == FAILED;
                            SPEC §5's Failure.code, e.g. "KV_TOO_SMALL")
  failure_detail()       -> str | None

...can be passed to :func:`build_gateway_router` / :func:`mount_gateway`.
``failure_code``/``failure_detail``/``hold_status`` are consulted only for
error-message cosmetics and are wrapped in ``_safe()`` — a supervisor
that hasn't implemented them yet (or raises) still gets correct 503/park/
stream routing, just a plainer error body.

Two numeric choices in the 503 bodies are NOT specified anywhere in
SPEC.md and are flagged here rather than silently invented as fact:
  - Retry-After for the park-timeout and FAILED cases (SPEC gives an
    exact value, 5s, only for the desired==STOPPED case). This module
    uses 30s for FAILED and a 5-60s ETA-derived value for park-timeout.
  - The "queue_full" error code/message for the max_parked=64 shed case
    (SPEC says only "immediate 503 + Retry-After", no code name).
Both are marked ``# UNSPECIFIED IN SPEC`` at their definition.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, runtime_checkable

import httpx
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# State constants — the exact strings from SPEC.md §6. Gateway does not own
# these enums (supervisor.py does); it only compares against the literal
# values, so it has no import-time dependency on however supervisor.py ends
# up spelling them (str enum, plain str constants, whatever).
# ---------------------------------------------------------------------------

DESIRED_STOPPED = "STOPPED"
DESIRED_RUNNING = "RUNNING"

ACTUAL_STOPPED = "STOPPED"
ACTUAL_PREFLIGHT = "PREFLIGHT"
ACTUAL_STARTING = "STARTING"
ACTUAL_READY = "READY"
ACTUAL_DRAINING = "DRAINING"
ACTUAL_STOPPING = "STOPPING"
ACTUAL_FAILED = "FAILED"
ACTUAL_UNMANAGED = "UNMANAGED"

# The only two paths that are ever held (SPEC corrections addendum C2).
# Matched against the tail of the path *after* the "/v1/" prefix the route
# below already consumes, e.g. "chat/completions" for "/v1/chat/completions".
_HOLD_ELIGIBLE_SUBPATHS = frozenset({"chat/completions", "responses"})

# RFC 7230 §6.1 hop-by-hop headers, stripped in both directions — carrying
# these across a proxy hop is always wrong. "host" is stripped only on the
# request side (httpx sets it correctly from the target URL itself).
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
_REQUEST_STRIP_HEADERS = _HOP_BY_HOP_HEADERS | {"host"}

_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

# Exact-path pass-through endpoints from SPEC §7 (everything except the
# "/v1/*" family, which gets its own path-parameter route so it can single
# out the two hold-eligible sub-paths).
_FIXED_PASSTHROUGH_PATHS = (
    "/health",
    "/ping",
    "/metrics",
    "/tokenize",
    "/detokenize",
    "/invocations",
    "/generative_scoring",
)


# ---------------------------------------------------------------------------
# Supervisor contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldStatus:
    """Best-effort description of "what's happening right now", used only
    to fill in the ``coldstart`` block of a 503 body while a request is
    parked or being shed. Never consulted for routing decisions."""

    phase_code: str  # e.g. "loading_weights" (phases.Phase value) or a
    #                  lowercased actual_state for non-boot holds like
    #                  "draining" / "stopping"
    phase_label: str  # human label, e.g. "Loading weights"
    eta_s: int | None  # remaining-seconds estimate, or None if unknown
    attempt: int  # current restart/boot attempt number (0 if not a restart)


@runtime_checkable
class SupervisorView(Protocol):
    """Structural contract this module needs from coldstart.supervisor.
    See the module docstring's INTEGRATION CONTRACT section."""

    def desired_state(self) -> str: ...
    def actual_state(self) -> str: ...
    def upstream_base_url(self) -> str: ...
    ready_event: asyncio.Event
    def hold_status(self) -> HoldStatus: ...
    def failure_code(self) -> str | None: ...
    def failure_detail(self) -> str | None: ...


def _safe(fn):
    """Call a best-effort supervisor accessor; None on any failure so a
    supervisor that's missing/broken on one of the cosmetic methods never
    takes the gateway down with it."""
    try:
        return fn()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Parked-request bookkeeping (SPEC §7 max_parked=64; SPEC §8's "3 agent
# requests held · oldest 2m14s" queue-visibility line reads this too).
# ---------------------------------------------------------------------------


@dataclass
class GatewayRuntime:
    """Live gateway state exposed for introspection (e.g. by the SSE
    ``gateway`` event in api.py, which is not owned by this file — this
    class is the read surface it's expected to poll, via
    ``app.state.gateway``)."""

    max_parked: int = 64
    parked_count: int = 0
    _park_started_at: list[float] = field(default_factory=list, repr=False)

    def try_enter_park(self) -> bool:
        """Reserve one parking slot. False means "shed" (SPEC §7:
        "max_parked=64 -> beyond that immediate 503 + Retry-After")."""
        if self.parked_count >= self.max_parked:
            return False
        self.parked_count += 1
        self._park_started_at.append(time.monotonic())
        return True

    def leave_park(self) -> None:
        self.parked_count = max(0, self.parked_count - 1)
        if self._park_started_at:
            # FIFO-ish; exact ordering under concurrent leaves isn't
            # load-bearing — this list only ever backs a UX gauge, never a
            # routing or safety decision.
            self._park_started_at.pop(0)

    def oldest_parked_age_s(self) -> float | None:
        if not self._park_started_at:
            return None
        return time.monotonic() - self._park_started_at[0]


# ---------------------------------------------------------------------------
# Header / URL helpers
# ---------------------------------------------------------------------------


def _filter_headers(items, strip: frozenset[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in items:
        if k.lower() in strip:
            continue
        out[k] = v
    return out


def _build_target_url(base_url: str, request: Request) -> str:
    url = base_url.rstrip("/") + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    return url


def _format_duration(seconds: float | int) -> str:
    """190 -> "3m10s"; 45 -> "45s" — matches SPEC §7's error-body example
    ("~3m10s remaining" for eta_s=190) exactly."""
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


# ---------------------------------------------------------------------------
# Error envelope — SPEC §7, OpenAI-compatible
# ---------------------------------------------------------------------------


def _error_body(
    *,
    message: str,
    code: str,
    phase: str | None = None,
    eta_s: int | None = None,
    parked: int = 0,
    attempt: int = 0,
) -> dict:
    return {
        "error": {
            "message": message,
            "type": "coldstart_upstream_unavailable",
            "code": code,
            "coldstart": {
                "phase": phase,
                "eta_s": eta_s,
                "parked": parked,
                "attempt": attempt,
            },
        }
    }


def _json_503(body: dict, *, retry_after: int) -> JSONResponse:
    return JSONResponse(body, status_code=503, headers={"Retry-After": str(retry_after)})


def _stopped_response() -> JSONResponse:
    body = _error_body(message="Coldstart: backend is stopped.", code="stopped")
    return _json_503(body, retry_after=5)  # SPEC §7: exact value given


def _failed_response(supervisor: SupervisorView) -> JSONResponse:
    code = _safe(supervisor.failure_code) or "failed"
    detail = _safe(supervisor.failure_detail)
    message = f"Coldstart: backend failed to start — {detail or code}"
    body = _error_body(message=message, code=str(code).lower())
    return _json_503(body, retry_after=30)  # UNSPECIFIED IN SPEC


def _shed_response(runtime: GatewayRuntime, supervisor: SupervisorView) -> JSONResponse:
    hold = _safe(supervisor.hold_status)
    message = (
        f"Coldstart: too many requests waiting for the backend to become "
        f"ready ({runtime.max_parked} already parked)."
    )
    body = _error_body(
        message=message,
        code="queue_full",  # UNSPECIFIED IN SPEC (name only; the 503+Retry-After behavior is specified)
        phase=hold.phase_code if hold else None,
        eta_s=hold.eta_s if hold else None,
        parked=runtime.max_parked,
        attempt=hold.attempt if hold else 0,
    )
    return _json_503(body, retry_after=5)  # UNSPECIFIED IN SPEC


def _timeout_response(supervisor: SupervisorView, runtime: GatewayRuntime) -> JSONResponse:
    hold = _safe(supervisor.hold_status)
    phase_label = hold.phase_label if hold else "starting"
    phase_code = hold.phase_code if hold else None
    eta_s = hold.eta_s if hold else None
    attempt = hold.attempt if hold else 0
    eta_str = _format_duration(eta_s) if eta_s is not None else "an unknown time"
    message = f"Coldstart: backend restarting — phase '{phase_label}', ~{eta_str} remaining"
    body = _error_body(
        message=message,
        code="restarting",
        phase=phase_code,
        eta_s=eta_s,
        parked=runtime.parked_count,
        attempt=attempt,
    )
    retry_after = max(5, min(int(eta_s) if eta_s is not None else 30, 60))  # UNSPECIFIED IN SPEC
    return _json_503(body, retry_after=retry_after)


def _unreachable_response(exc: BaseException) -> JSONResponse:
    body = _error_body(message=f"Coldstart: upstream unreachable — {exc}", code="upstream_unreachable")
    return _json_503(body, retry_after=5)  # UNSPECIFIED IN SPEC


# ---------------------------------------------------------------------------
# The actual proxy — transparent stream, no buffering either direction
# ---------------------------------------------------------------------------


async def _stream_proxy(client: httpx.AsyncClient, base_url: str, request: Request) -> Response:
    """READY / UNMANAGED behavior, and every plain-pass-through path: open
    the upstream request with the client's own body stream as content
    (never buffered — SPEC §7: "body NOT buffered (TCP backpressure)"),
    grab status+headers as soon as they arrive, then hand back a
    StreamingResponse that lazily pulls raw bytes from upstream.

    C7 (client-disconnect propagation): Starlette's StreamingResponse
    races its body-streaming task against a disconnect listener and
    cancels the streaming task the moment the client goes away. That
    cancellation lands inside ``body()`` below, at whatever
    ``upstream.aiter_raw()`` await it's suspended on, and the
    ``try/finally`` ensures ``upstream.aclose()`` still runs — which is
    what actually closes the socket vLLM's ``with_cancellation`` is
    watching. THE HARD RULE this satisfies: once a chunk has been yielded
    here, this function makes no further decision that could re-issue the
    request — a disconnect after that point only ever tears the one
    upstream connection down, never retries it.
    """
    target = _build_target_url(base_url, request)
    headers = _filter_headers(request.headers.items(), _REQUEST_STRIP_HEADERS)
    req = client.build_request(request.method, target, headers=headers, content=request.stream())

    try:
        upstream = await client.send(req, stream=True)
    except httpx.HTTPError as exc:
        return _unreachable_response(exc)

    response_headers = _filter_headers(upstream.headers.items(), _HOP_BY_HOP_HEADERS)

    async def body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(body(), status_code=upstream.status_code, headers=response_headers)


async def _park(supervisor: SupervisorView, hold_max_s: float, poll_s: float) -> str:
    """Wait for supervisor.ready_event, bounded at hold_max_s total, but
    polled at poll_s granularity so a transition to FAILED or a
    desired_state flip to STOPPED *during* the hold is noticed promptly
    instead of only once the full 240s elapses (corrections addendum C8:
    "Never hold when reached_ready was false" — a boot that has already
    failed must stop blocking this request well before the outer ceiling).

    Returns one of "ready" | "failed" | "stopped" | "timeout".
    """
    deadline = time.monotonic() + hold_max_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "timeout"
        wait_s = min(poll_s, remaining)
        try:
            await asyncio.wait_for(supervisor.ready_event.wait(), timeout=wait_s)
        except TimeoutError:
            pass
        else:
            return "ready"
        if _safe(supervisor.actual_state) == ACTUAL_FAILED:
            return "failed"
        if _safe(supervisor.desired_state) == DESIRED_STOPPED:
            return "stopped"


async def _handle_hold_eligible(
    request: Request,
    supervisor: SupervisorView,
    client: httpx.AsyncClient,
    runtime: GatewayRuntime,
    hold_max_s: float,
    park_poll_s: float,
) -> Response:
    """POST /v1/chat/completions and POST /v1/responses only (C2). Every
    other path goes through :func:`_handle_pass_through` instead, which
    never parks and never consults state at all.

    Ordering below matters and is deliberate:
    desired==STOPPED is checked first because it is an unconditional
    "never park" per SPEC §7, independent of whatever actual_state
    happens to still read as mid-race (supervisor.stop() sets
    desired_state=STOPPED *before* signalling, per SPEC §6, so a request
    can legitimately arrive with actual_state still READY for one more
    tick while desired is already STOPPED — SPEC's intent is clearly not
    to start a brand-new generation against a server that a human just
    told to stop).
    """
    desired = supervisor.desired_state()
    actual = supervisor.actual_state()

    if desired == DESIRED_STOPPED:
        return _stopped_response()

    if actual == ACTUAL_FAILED:
        return _failed_response(supervisor)

    if actual in (ACTUAL_READY, ACTUAL_UNMANAGED):
        return await _stream_proxy(client, supervisor.upstream_base_url(), request)

    # STARTING / PREFLIGHT / DRAINING / STOPPING (desired==RUNNING, per the
    # STOPPED check above) — and, defensively, any other actual_state this
    # module doesn't otherwise recognize while desired==RUNNING: SPEC §6's
    # startup reconciliation means such a state is transient (about to
    # become PREFLIGHT/STARTING), so parking is the correct call rather
    # than either an indefinite hold (hold_max_s bounds it regardless) or
    # a spurious immediate 503.
    if not runtime.try_enter_park():
        return _shed_response(runtime, supervisor)

    try:
        outcome = await _park(supervisor, hold_max_s, park_poll_s)
    finally:
        runtime.leave_park()

    if outcome == "ready":
        return await _stream_proxy(client, supervisor.upstream_base_url(), request)
    if outcome == "failed":
        return _failed_response(supervisor)
    if outcome == "stopped":
        return _stopped_response()
    return _timeout_response(supervisor, runtime)  # outcome == "timeout"


async def _handle_pass_through(request: Request, supervisor: SupervisorView, client: httpx.AsyncClient) -> Response:
    """Every path except the two hold-eligible ones (C2): GET /v1/models,
    GET/POST /health, /ping, /metrics, /tokenize, /detokenize,
    /invocations, /generative_scoring, and anything else under /v1/*.
    Deliberately consults NO supervisor state — "pass through; 503 when
    down" means literally attempt the connection now and translate a
    connect/transport failure into 503; never hold, never synthesize a
    200 (C2's explicit warning: faking GET /v1/models as 200 would make
    ``is_server_up()`` believe a dead server is alive and nothing would
    ever restart it).
    """
    base_url = _safe(supervisor.upstream_base_url)
    if not base_url:
        return _unreachable_response(RuntimeError("no upstream configured"))
    return await _stream_proxy(client, base_url, request)


# ---------------------------------------------------------------------------
# Router construction
# ---------------------------------------------------------------------------


def build_gateway_router(
    supervisor: SupervisorView,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    hold_max_s: float = 240.0,
    max_parked: int = 64,
    park_poll_s: float = 0.5,
) -> tuple[APIRouter, httpx.AsyncClient, GatewayRuntime]:
    """Build the gateway's routes against one supervisor. Returns the
    router plus the httpx client and runtime-info object it owns, so a
    caller (production: :func:`mount_gateway`; tests: directly) controls
    the client's lifecycle explicitly rather than this module reaching
    for a hidden global.

    `transport` lets tests substitute `httpx.MockTransport` for the real
    network — production callers leave it None and get a real
    `httpx.AsyncClient`. No read timeout is set (`read=None`): a
    generation can legitimately run for minutes, and the only thing that
    should ever end a stream early is the client disconnecting (C7) — not
    a server-side clock. `connect` stays short since upstream is always
    127.0.0.1.
    """
    timeout = httpx.Timeout(connect=10.0, read=None, write=None, pool=None)
    client = httpx.AsyncClient(transport=transport, timeout=timeout)
    runtime = GatewayRuntime(max_parked=max_parked)
    router = APIRouter()

    @router.api_route("/v1/{full_path:path}", methods=_METHODS)
    async def v1_dispatch(full_path: str, request: Request) -> Response:
        if request.method == "POST" and full_path in _HOLD_ELIGIBLE_SUBPATHS:
            return await _handle_hold_eligible(request, supervisor, client, runtime, hold_max_s, park_poll_s)
        return await _handle_pass_through(request, supervisor, client)

    def _make_fixed_handler():
        async def _handler(request: Request) -> Response:
            return await _handle_pass_through(request, supervisor, client)

        return _handler

    for fixed_path in _FIXED_PASSTHROUGH_PATHS:
        router.add_api_route(fixed_path, _make_fixed_handler(), methods=_METHODS)

    return router, client, runtime


def mount_gateway(
    app: FastAPI,
    supervisor: SupervisorView,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    hold_max_s: float = 240.0,
    max_parked: int = 64,
    park_poll_s: float = 0.5,
) -> GatewayRuntime:
    """Convenience for whoever owns app.py: mounts the gateway routes on
    `app`, wires the owned httpx client's shutdown into the app lifecycle,
    and stashes the runtime-info object at ``app.state.gateway`` (read by
    the SSE ``gateway`` event, SPEC §8) before returning it too.

    Shutdown wiring uses ``app.router.add_event_handler`` (FastAPI's own
    backward-compat shim over Starlette's removed on_shutdown list; NOT
    ``app.add_event_handler``, which this FastAPI version no longer has).
    That shim only fires if app.py leaves FastAPI's *default* lifespan in
    place. If app.py instead supplies its own ``lifespan=`` context
    manager, on_shutdown handlers registered this way are never called —
    in that case, close ``app.state.gateway_client`` (stashed here
    specifically as a fallback) from within that lifespan's teardown.
    """
    router, client, runtime = build_gateway_router(
        supervisor,
        transport=transport,
        hold_max_s=hold_max_s,
        max_parked=max_parked,
        park_poll_s=park_poll_s,
    )
    app.include_router(router)
    app.state.gateway = runtime
    app.state.gateway_client = client  # fallback close target — see docstring

    async def _close_client() -> None:
        await client.aclose()

    app.router.add_event_handler("shutdown", _close_client)
    return runtime
