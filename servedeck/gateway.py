"""The one normalising gateway — REDESIGN-2026-09-12 §1 (R3), §2.3, §2.7.

``http://127.0.0.1:8010/v1`` becomes the only URL any client is ever
configured with.  The gateway resolves the request's ``model`` (id, alias, or
effort preset) to a live model's port, applies that route's policies, and
streams the exchange through in both directions.

Why this file replaces three things
-----------------------------------
R3, the third of the five root causes the redesign removes: *normalisation
lived in per-model side proxies, not in the gateway*.  ``:8005`` mirrored the
reasoning field for Flash-Next, ``:8003`` injected reasoning effort for GLM,
``:8006`` did the same for the 27B — each fronting one **fixed** upstream port,
so a port swap broke a proxy and clients had to know which port was "safe" for
which model.  Meanwhile the real gateway forwarded raw bytes and was unusable
for VS Code.  All three proxies' transforms now live in ``policies.py`` and are
selected per route, by the registry, here.

What was dropped on purpose
---------------------------
The previous ``gateway.py`` was dead code (nothing imported it) built around
*parking*: holding a request for up to 240 s while the backend restarted, with
a 64-slot queue, shed responses and ETA-derived ``Retry-After`` values.  None
of it was ever exercised against a live boot.  It is replaced by an immediate
``503`` + ``Retry-After: 15`` naming what is actually in the main slot — the
honest answer, and one a client can act on.  Parking can come back later, on
top of a gateway that is tested, if a real client turns out to need it.

Two rules this file does not bend
---------------------------------
1. **Neither body is ever buffered when it does not have to be.**  The request
   body is read only far enough to resolve ``model`` (``policies.scan_model``)
   and the remainder is streamed; the response is streamed chunk by chunk.
   A body is held in full only when a policy must rewrite it — and every
   policy is off by default.  ``app.py``'s ``catch_all``, which this replaces,
   buffers every request including megabyte base64 images.
2. **A client disconnect reaches the upstream socket.**  Starlette cancels the
   streaming task when the downstream client goes away; that cancellation
   lands inside the body generator and its ``finally`` closes the upstream
   response.  Closing that socket is what actually aborts an in-flight vLLM
   generation, which matters a great deal at small ``--max-num-seqs``: one
   leaked generation holds a sequence slot and starves every other request.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from servedeck import glm_policies, policies

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The route table contract — implemented by the registry (P1/P3), not here
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutePolicies:
    """Which normalisations this model's traffic needs.  All off by default.

    ``mirror_reasoning``
        Copy ``reasoning`` into ``reasoning_content`` (and the reverse) in the
        JSON body and in every SSE delta, so a chat-completions client can
        store the thinking and echo it back next turn.  Registry field
        ``reasoning.mirror_content``.
    ``effort_overlay``
        ``chat_template_kwargs`` entries an effort *preset* carries, merged
        with ``setdefault`` so an explicit caller value always wins — e.g.
        ``{"reasoning_effort": "high"}`` for ``glm53-flash-high``.
    ``min_output_tokens``
        Raise an explicitly-small output budget to this floor, clamped to
        ``ctx``.  Never lowers one, never invents one.
    ``ctx``
        The model's context length.  Reported as ``max_model_len`` by
        ``/v1/models`` and used as the clamp for ``min_output_tokens``.

    ``ctx`` has **no default**, and is first so it cannot have one.  A zero ctx
    publishes ``max_model_len: 0`` to every client and makes the output floor
    clamp to nothing — the GLM thinking-budget fix would read as configured and
    do nothing at all.  A registry that cannot say how long a model's context
    is has not finished loading that model.

    The last three are P9's GLM-5.3 client-compatibility switches; every one of
    them is implemented in ``glm_policies.py`` and every one is off here, so a
    route that does not set them is byte-for-byte unaffected.  Registry fields
    ``sanitize_tool_tags`` / ``restore_reasoning`` / ``capture``.

    ``sanitize_tool_tags``
        Strip GLM template markup (``</arg_key>`` and friends) that its
        tool-call parser leaks into parsed arguments.  Default true for glm53
        in ``models.toml``: a recorded failure, and a pure repair.
    ``restore_reasoning``
        Put reasoning back on the in-flight assistant turns of a request whose
        client dropped it.  Default **false**, including for glm53 — see
        ``glm_policies`` for the Xid-31 correlation that made it switchable.
    ``capture``
        This route *consents* to request/response capture.  It is not an
        enable: capture also needs ``SERVEDECK_GLM_CAPTURE=1``.
    """

    ctx: int
    mirror_reasoning: bool = False
    effort_overlay: Mapping[str, Any] | None = None
    min_output_tokens: int | None = None
    sanitize_tool_tags: bool = False
    restore_reasoning: bool = False
    capture: bool = False


@dataclass(frozen=True)
class Route:
    """One model, as the gateway needs to see it.

    ``aliases`` and ``presets`` are what ``GET /v1/models`` publishes alongside
    ``model_id``; ``resolve()`` must accept every one of those names.  They are
    kept apart because a preset is not a synonym — ``glm53-flash-high`` is the
    same weights with a different request overlay — and the distinction is
    worth keeping visible in the table even though ``/v1/models`` lists both.
    """

    model_id: str
    port: int
    policies: RoutePolicies
    live: bool = False
    aliases: tuple[str, ...] = ()
    presets: tuple[str, ...] = ()
    host: str = "127.0.0.1"

    def served_names(self) -> tuple[str, ...]:
        """Every name this route answers to, id first."""
        return (self.model_id, *self.aliases, *self.presets)

    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@runtime_checkable
class RouteTable(Protocol):
    """What the gateway needs from the registry.  Structural, so anything —
    a class instance or a bare module — that exposes these four methods can be
    passed to :func:`build_router`.

    ``resolve`` must accept an id, an alias **and** a preset name, and must
    return routes that are not currently running too (that is what separates a
    503 "not running" from a 404 "no such model" — collapsing the two would
    tell a client to reconfigure itself when all it had to do was wait).
    """

    def resolve(self, name: str) -> Route | None:
        """Route serving ``name`` (id, alias or preset), live or not."""

    def live_routes(self) -> list[Route]:
        """Every route whose model is up right now, for ``GET /v1/models``."""

    def main(self) -> Route | None:
        """The route occupying the exclusive main GPU slot, or None."""

    def known_names(self) -> list[str]:
        """Every name ``resolve`` would accept, for the 404 body."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

#: Passthrough endpoints outside ``/v1``.  They carry no ``model`` (except
#: ``/tokenize`` and ``/detokenize``, which do), so without one they go to the
#: main slot — which is what "is the server up" probes mean by them anyway.
#: ``/ping`` is here because the proxy this replaces forwarded it
#: (``app.py:1783``'s ``_PROXY_PREFIXES``) and something is probing it.
_EXTRA_PATHS = ("/health", "/ping", "/metrics", "/tokenize", "/detokenize")

#: Methods whose body can carry a ``model``.  Everything else routes to main.
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})

#: Ceiling on how much of a request body is read looking for ``model``.  Not a
#: correctness bound — a body that hides ``model`` past this point simply
#: routes to the main slot — but a memory bound, so a hostile or malformed
#: body cannot make the gateway accumulate without limit.  2 MiB leaves room
#: for a client that puts a ~1.5 MB base64 image ahead of ``model`` (the
#: OpenAI SDK does not, but a hand-built body may) while keeping the worst
#: case per in-flight request to something a dashboard can afford.
_MODEL_SCAN_LIMIT = 2 * 1024 * 1024

#: Cheap pre-filter before the structural scan.  ``scan_model`` is a
#: byte-at-a-time Python loop; running it over a megabyte of base64 image on
#: every chunk would cost more than the request.  If these bytes are not
#: present at all there is no ``model`` key to find, and ``bytes.__contains__``
#: settles that at C speed.
_MODEL_MARK = b'"model"'

#: Connect fast (upstream is always 127.0.0.1, so a slow connect means it is
#: not there), read slowly (a generation can legitimately run for an hour).
_TIMEOUT = httpx.Timeout(connect=10.0, read=3600.0, write=3600.0, pool=10.0)

#: How long a client should wait before retrying a model that is starting.  A
#: cold boot of the main slot is minutes, but a 15 s poll is what makes a
#: client's own retry loop feel like "it came back" rather than "it failed".
_RETRY_AFTER_S = 15

#: The responses API, matched as a prefix so ``/v1/responses/{id}`` (GET,
#: DELETE, and the ``/cancel`` sub-path) is treated the same as the POST.
#: Codex speaks this API, where reasoning is already a first-class output
#: item, so mirroring there would add a field to a shape that never lacked it.
_RESPONSES_PATH = "/v1/responses"


def _is_responses_path(path: str) -> bool:
    return path == _RESPONSES_PATH or path.startswith(_RESPONSES_PATH + "/")


# ---------------------------------------------------------------------------
# Error envelopes — OpenAI-shaped, because every client already parses that
# ---------------------------------------------------------------------------


def _error(
    message: str, *, type_: str, code: str, status: int, headers: dict[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": type_, "code": code}},
        status_code=status,
        headers=headers,
    )


def unknown_model_response(name: str, known: list[str]) -> JSONResponse:
    """404. The name is not in the registry at all — the client is misconfigured
    and no amount of waiting will help, so list what it could have said."""
    listed = ", ".join(known) if known else "none"
    return JSONResponse(
        {
            "error": {
                "message": f"The model '{name}' does not exist. Known models: {listed}",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            }
        },
        status_code=404,
    )


def _main_slot_description(route: Route | None, main: Route | None) -> str:
    if main is None:
        return "empty"
    if route is not None and main.model_id == route.model_id:
        return f"{main.model_id}, still starting"
    return main.model_id


def not_running_response(route: Route | None, main: Route | None) -> JSONResponse:
    """503 + Retry-After. The model is registered but not up: either it is
    booting or something else holds the exclusive main slot.  Naming what *is*
    in the slot is the difference between a client that waits and a user who
    goes looking for a config file."""
    slot = _main_slot_description(route, main)
    message = (
        f"{route.model_id} is not running (main slot: {slot})"
        if route is not None
        else f"no model is running (main slot: {slot})"
    )
    return _error(
        message,
        type_="model_not_running",
        code="not_running",
        status=503,
        headers={"Retry-After": str(_RETRY_AFTER_S)},
    )


def upstream_unavailable_response(route: Route, exc: BaseException) -> JSONResponse:
    """502. The registry says this model is live and the socket says otherwise.
    That is a disagreement between servedeck and reality, not a client error,
    and it must not be dressed up as a 503 the client will silently retry."""
    return _error(
        f"upstream {route.base_url()} for {route.model_id} is unreachable: "
        f"{type(exc).__name__}: {exc}",
        type_="upstream_unavailable",
        code="upstream_unavailable",
        status=502,
    )


# ---------------------------------------------------------------------------
# Request plumbing
# ---------------------------------------------------------------------------


def _declares_a_body(request: Request) -> bool:
    if request.headers.get("transfer-encoding"):
        return True
    try:
        return int(request.headers.get("content-length", "0")) > 0
    except ValueError:
        return False


async def _peek_model(request: Request) -> tuple[str | None, bytes, AsyncIterator[bytes] | None]:
    """Read just enough of the request body to learn its ``model``.

    Returns ``(model, bytes_read, unread_remainder)``.  ``unread_remainder`` is
    ``None`` only when the body ended while we were still reading; otherwise it
    is the live client stream, suspended wherever the scan stopped, and the
    caller must forward it (``policies.chain_body``) or drain it.

    The scan is re-run as the buffer grows, but only on a doubling schedule
    (and only when the bytes ``"model"`` are present at all), which bounds the
    total scanning work at roughly twice the body size instead of the O(n²) a
    naive rescan-per-chunk would cost on a chunked upload.
    """
    if request.method not in _BODY_METHODS:
        # A body on GET/DELETE/HEAD is unusual but legal, and dropping one
        # silently is the kind of bug that only shows up as a confusing
        # upstream 400.  It is forwarded unread: these methods carry no
        # ``model``, so they route to the main slot either way.
        return None, b"", (request.stream() if _declares_a_body(request) else None)

    stream = request.stream()
    buf = bytearray()
    model: str | None = None
    exhausted = True
    scanned_len = -1
    next_scan_at = 0
    mark_seen = False

    async for chunk in stream:
        if not chunk:
            continue
        previous = len(buf)
        buf += chunk
        if len(buf) >= _MODEL_SCAN_LIMIT:
            # Checked FIRST, before anything that could `continue`: a body that
            # never mentions a model must stop accumulating here and stream the
            # rest.  With this test below the mark test, such a body was read to
            # EOF — exactly the unbounded buffering this rewrite removes.
            exhausted = False
            break
        if not mark_seen:
            # Sticky, and searched only over the newly arrived bytes plus an
            # overlap of len(mark)-1 so a mark straddling a chunk boundary is
            # still found.  Re-searching the whole buffer per chunk is O(n²)
            # over the body — at C speed, but still quadratic, and on a
            # multi-megabyte upload that is the dominant cost of the request.
            tail = buf[max(0, previous - (len(_MODEL_MARK) - 1)) :]
            mark_seen = _MODEL_MARK in tail
        if not mark_seen:
            continue
        if len(buf) < next_scan_at:
            continue
        scanned_len = len(buf)
        scan = policies.scan_model(bytes(buf))
        if scan.complete:
            model = scan.model
            exhausted = False
            break
        next_scan_at = len(buf) * 2

    if model is None and mark_seen and len(buf) != scanned_len:
        # The body ended (or hit the ceiling) between scheduled scans; one last
        # look, so a small body split across chunks is not mis-routed to main.
        model = policies.scan_model(bytes(buf)).model

    return model, bytes(buf), (None if exhausted else stream)


def _forward_headers(request: Request) -> list[tuple[bytes, bytes]]:
    """Client headers minus the hop-by-hop set, with encoding forced to
    identity so upstream's bytes are the bytes we relay."""
    out = [
        (k, v)
        for k, v in request.headers.raw
        if k.decode("latin-1").lower() not in policies.DROP_REQUEST_HEADERS
    ]
    out.append((b"accept-encoding", b"identity"))
    return out


def _response_headers(upstream: httpx.Response) -> dict[str, str]:
    return {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() not in policies.DROP_RESPONSE_HEADERS
    }


async def _drain(rest: AsyncIterator[bytes] | None) -> bytes:
    if rest is None:
        return b""
    out = bytearray()
    async for chunk in rest:
        if chunk:
            out += chunk
    return bytes(out)


# ---------------------------------------------------------------------------
# The proxy
# ---------------------------------------------------------------------------


async def _proxy(
    request: Request, route: Route, client: httpx.AsyncClient, prefix: bytes,
    rest: AsyncIterator[bytes] | None, *, rewrite_model_to: str | None,
    glm_state: glm_policies.GlmState | None = None,
) -> Response:
    pol = route.policies
    # /v1/responses: mirroring OFF (reasoning is already a first-class output
    # item there, so there is no missing name to supply), effort overlay and
    # output floor ON — glm53-effort-proxy/proxy.py:496-509 applies the
    # alias → chat_template_kwargs.reasoning_effort mapping on this endpoint
    # for exactly the reason the preset exists: Codex speaks wire_api
    # "responses" only, so dropping the overlay here would rewrite
    # `glm53-flash-high` to `glm53-flash` and silently deliver max effort.
    is_responses = _is_responses_path(request.url.path)

    # --- P9 hook: GLM-5.3 client-compatibility policies --------------------
    # One object per request, or None when this route has all three switches
    # off — which is every route but glm53, so nothing below changes for them.
    # Everything it does lives in glm_policies.py; the five call sites here are
    # all one-liners, marked "P9".
    glm = glm_policies.begin(
        pol,
        model_id=route.model_id,
        state=glm_state,
        responses_api=is_responses,
        headers=request.headers,
    )

    needs_body = (
        rewrite_model_to is not None
        or bool(pol.effort_overlay)
        or bool(pol.min_output_tokens)
        or (glm is not None and glm.needs_request_body)  # P9
    )
    content: Any
    if needs_body and request.method in _BODY_METHODS:
        raw = prefix + await _drain(rest)
        content = policies.apply_request_policies(
            raw,
            model_id=rewrite_model_to,
            overlay=pol.effort_overlay,
            floor=pol.min_output_tokens,
            ctx=pol.ctx,
        )
        if glm is not None:  # P9
            content = glm.on_request_body(content)
    elif prefix or rest is not None:
        content = policies.chain_body(prefix, rest)
    else:
        content = None

    url = route.base_url() + request.url.path
    if request.url.query:
        url = f"{url}?{request.url.query}"

    upstream_request = client.build_request(
        request.method, url, headers=_forward_headers(request), content=content
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except Exception as exc:  # noqa: BLE001 — any transport failure is a 502
        return upstream_unavailable_response(route, exc)

    headers = _response_headers(upstream)
    media = upstream.headers.get("content-type", "")
    # Whether a response is transformed is decided by its **upstream
    # Content-Type**, never by whether the request said stream=true: the proxy
    # then never has to guess, and a server that answers a streaming request
    # with a JSON error is handled by the JSON path automatically.
    mirror = pol.mirror_reasoning and not is_responses

    if media.startswith("text/event-stream"):
        source = policies.sse_stream(upstream.aiter_bytes()) if mirror else upstream.aiter_bytes()
        if glm is not None:  # P9 — composes on top of the mirror, line by line
            source = glm.sse(source)

        async def sse_body() -> AsyncIterator[bytes]:
            """The stream, with a terminal error EVENT if the upstream dies.

            Status and headers are already on the wire by the time a stream
            breaks, so there is no status code left to change: the only honest
            way to tell a client is in-band. Without this the stream simply
            stopped, which a client cannot tell from a completed answer — the
            agent kept whatever half-sentence it had as the model's reply.
            """
            try:
                async for chunk in source:
                    yield chunk
            except Exception as exc:  # noqa: BLE001 — mid-stream upstream death
                log.warning("%s: upstream stream failed mid-response: %r", route.model_id, exc)
                yield policies.sse_error_chunk(
                    f"the upstream for {route.model_id} stopped mid-response "
                    f"({type(exc).__name__}); the answer is incomplete"
                )
            finally:
                await upstream.aclose()

        return StreamingResponse(
            sse_body(), status_code=upstream.status_code, headers=headers, media_type=media
        )

    if (mirror or glm is not None) and (  # P9 adds the second reason to buffer
        media.startswith("application/json") or media.startswith("application/vnd")
    ):
        # The only place a response body is held in full, and only because a
        # complete JSON object cannot be mirrored a chunk at a time.
        #
        # The read can fail after the headers arrived: the engine died, or the
        # GPU fault took it, mid-body. Uncaught, that reached the client as a
        # bare "500 Internal Server Error" in text/plain — not an OpenAI error
        # object, so every client reported it as a parse failure or a crash of
        # its own rather than "the server went away, retry".
        try:
            payload = await upstream.aread()
        except Exception as exc:  # noqa: BLE001 — a dead upstream is a 502
            return upstream_unavailable_response(route, exc)
        finally:
            await upstream.aclose()
        body = policies.transform_json_body(payload) if mirror else payload
        if glm is not None:  # P9
            body = glm.on_json_body(body)
        return Response(
            content=body,
            status_code=upstream.status_code,
            headers=headers,
            media_type=media or None,
        )

    async def raw_body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        raw_body(), status_code=upstream.status_code, headers=headers, media_type=media or None
    )


def _model_less_route(routes: RouteTable) -> Route | None:
    """Where a request that named no model goes: the main slot if it is up,
    otherwise any live route, otherwise nowhere."""
    main = routes.main()
    if main is not None and main.live:
        return main
    live = routes.live_routes()
    return live[0] if live else None


async def _dispatch(
    request: Request,
    routes: RouteTable,
    client: httpx.AsyncClient,
    glm_state: glm_policies.GlmState | None = None,  # P9
) -> Response:
    model, prefix, rest = await _peek_model(request)

    if model is None:
        # No model named: a GET probe (/health, /ping, /metrics), or a POST
        # that left it out.  The main slot is what "the local model" means to
        # a client that did not say — but a box with only residents up has no
        # main slot and is not down, so any live route will answer for it.
        # Reporting 503 there would make `doctor` and every is-server-up probe
        # call a serving machine dead.
        route = _model_less_route(routes)
        if route is None:
            await _drain(rest)
            return not_running_response(None, routes.main())
    else:
        resolved = routes.resolve(model)
        if resolved is None:
            await _drain(rest)
            return unknown_model_response(model, routes.known_names())
        if not resolved.live:
            await _drain(rest)
            return not_running_response(resolved, routes.main())
        route = resolved

    # The requested name is an alias or a preset, not the name vLLM was started
    # with: rewrite it, or upstream answers 404 for a model it is serving.
    rewrite = route.model_id if (model is not None and model != route.model_id) else None
    return await _proxy(
        request, route, client, prefix, rest, rewrite_model_to=rewrite, glm_state=glm_state
    )


def models_payload(routes: RouteTable, *, created: int | None = None) -> dict:
    """``GET /v1/models``: one entry per served name of every live route.

    Every name a client has ever been configured with appears, so the 404 class
    of bug ("model does not exist" after a rename) is gone by construction; and
    every entry carries ``max_model_len``, so a client can size its context
    from the same place it learned the name instead of a second config file.
    ``root`` is the model id, which is what tells a client that three of these
    entries are the same weights.
    """
    stamp = int(time.time()) if created is None else created
    data = []
    for route in routes.live_routes():
        for name in route.served_names():
            data.append(
                {
                    "id": name,
                    "object": "model",
                    "created": stamp,
                    "owned_by": "servedeck",
                    "root": route.model_id,
                    "parent": None,
                    "max_model_len": route.policies.ctx,
                }
            )
    return {"object": "list", "data": data}


# ---------------------------------------------------------------------------
# Router construction
# ---------------------------------------------------------------------------


def build_router(
    routes: RouteTable,
    *,
    client: httpx.AsyncClient | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> APIRouter:
    """Build the gateway's ``/v1/*`` routes against one route table.

    ``client`` / ``transport`` exist so tests can substitute
    ``httpx.MockTransport`` for the network.  When neither is given the router
    owns a real client and stashes it at ``router.gateway_client`` — the caller
    that mounts the router is responsible for closing it at shutdown, because
    a module-level client would outlive the app it belongs to and is exactly
    the kind of hidden global this rewrite exists to remove.
    """
    owned = client is None
    if client is None:
        client = httpx.AsyncClient(transport=transport, timeout=_TIMEOUT, follow_redirects=False)
    router = APIRouter()
    # Not an APIRouter field; attached deliberately so the owner can close it.
    router.gateway_client = client  # type: ignore[attr-defined]
    router.gateway_owns_client = owned  # type: ignore[attr-defined]
    # P9: this router's GLM policy state — one bounded reasoning bucket and one
    # set of counters per route. Attached, not module-global, so two gateways in
    # one process (the :8011 rehearsal of cutover step 1) share nothing. Its
    # ``stats()`` is what the dashboard reads for the answerless-turn count.
    glm_state = glm_policies.GlmState()
    router.glm_state = glm_state  # type: ignore[attr-defined]

    @router.get("/v1/models")
    async def list_models() -> JSONResponse:
        return JSONResponse(models_payload(routes))

    @router.api_route("/v1/{path:path}", methods=_METHODS)
    async def v1(path: str, request: Request) -> Response:
        return await _dispatch(request, routes, client, glm_state)

    def _make_handler():
        async def handler(request: Request) -> Response:
            return await _dispatch(request, routes, client, glm_state)

        return handler

    for extra in _EXTRA_PATHS:
        router.add_api_route(extra, _make_handler(), methods=_METHODS)

    return router


__all__ = [
    "Route",
    "RoutePolicies",
    "RouteTable",
    "build_router",
    "models_payload",
    "not_running_response",
    "unknown_model_response",
    "upstream_unavailable_response",
]
