"""Tests for ``servedeck.gateway`` — the one normalising gateway.

The upstream is an ``httpx.MockTransport`` that records exactly what the
gateway sent it and replies with whatever the test asks for (including a
byte-stream cut at boundaries the test chooses).  The gateway itself is driven
over ``httpx.ASGITransport``, so every assertion goes through the real ASGI
request/response path — path parameters, header raw lists, streaming response
bodies — rather than calling handler functions directly.

What is deliberately NOT asserted here: time-to-first-chunk.  httpx's
``ASGITransport`` collects a streaming response into a list of body parts
before handing it back, so an in-process test cannot see incremental delivery
however the gateway behaves.  That property is proved in
``tests/test_gateway_e2e.py`` against a real server over a real socket.  What
*is* asserted here is the property that makes incremental delivery correct:
the output must not depend on where the upstream's chunk boundaries fell.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI

from servedeck import gateway
from servedeck.gateway import Route, RoutePolicies
from tests.fake_routes import FakeRouteTable, load, lfm2_route, reassemble, sse_events

STREAM = "chat_stream_reasoning.sse"
NONSTREAM = "chat_nonstream_reasoning.json"
NOREASON = "chat_nonstream_noreasoning.json"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class Upstream:
    """A recorded fake vLLM behind ``httpx.MockTransport``."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.status = 200
        self.media = "application/json"
        self.body: bytes = b"{}"
        self.chunks: list[bytes] | None = None
        self.headers: dict[str, str] = {}
        self.raise_on_send: Exception | None = None

    @property
    def last(self) -> httpx.Request:
        assert self.requests, "upstream was never called"
        return self.requests[-1]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raise_on_send is not None:
            raise self.raise_on_send
        headers = {"content-type": self.media, **self.headers}
        if self.chunks is None:
            return httpx.Response(self.status, headers=headers, content=self.body)

        pieces = list(self.chunks)

        async def gen():
            for piece in pieces:
                yield piece

        return httpx.Response(self.status, headers=headers, content=gen())


def rig(table: FakeRouteTable) -> tuple[FastAPI, Upstream]:
    upstream = Upstream()
    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream.handler))
    app = FastAPI()
    app.include_router(gateway.build_router(table, client=client))
    return app, upstream


def call(app: FastAPI, method: str, path: str, **kwargs) -> httpx.Response:
    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
            return await c.request(method, path, **kwargs)

    return asyncio.run(go())


def two_model_table(*, flash_live: bool = True, lfm_live: bool = True) -> FakeRouteTable:
    """The shape the box actually runs: one exclusive main model with an alias
    and two effort presets, plus an always-on resident."""
    flash = Route(
        model_id="glm53-flash",
        port=8002,
        live=flash_live,
        aliases=("glm53",),
        presets=("glm53-flash-high", "glm53-flash-low"),
        policies=RoutePolicies(mirror_reasoning=True, ctx=327680),
    )
    return FakeRouteTable([flash, lfm2_route(live=lfm_live)], main_name="glm53-flash")


# ---------------------------------------------------------------------------
# The Protocol itself
# ---------------------------------------------------------------------------


def test_fake_route_table_satisfies_the_protocol():
    """P1/P3 implement ``RouteTable``; this pins that the four methods the
    gateway calls are the four methods the Protocol declares."""
    assert isinstance(two_model_table(), gateway.RouteTable)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,port,upstream_model",
    [
        ("glm53-flash", 8002, "glm53-flash"),
        ("glm53", 8002, "glm53-flash"),
        ("glm53-flash-high", 8002, "glm53-flash"),
        ("LFM2.5-350M", 8007, "LFM2.5-350M"),
        ("lfm2", 8007, "LFM2.5-350M"),
    ],
)
def test_routes_by_id_alias_and_preset(name, port, upstream_model):
    app, up = rig(two_model_table())
    call(app, "POST", "/v1/chat/completions", json={"model": name, "messages": []})
    assert up.last.url.port == port
    assert up.last.url.path == "/v1/chat/completions"
    # vLLM was started with the id, not with the alias or the preset name, so
    # the name the client used must be rewritten or upstream answers 404.
    assert json.loads(up.last.content)["model"] == upstream_model


def test_get_without_a_model_goes_to_the_main_slot():
    app, up = rig(two_model_table())
    call(app, "GET", "/v1/anything?a=1&b=two")
    assert up.last.url.port == 8002
    assert up.last.url.query == b"a=1&b=two"


def test_post_without_a_model_goes_to_the_main_slot_untouched():
    app, up = rig(two_model_table())
    sent = b'{"messages":[{"role":"user","content":"hi"}]}'
    call(app, "POST", "/v1/chat/completions", content=sent,
         headers={"content-type": "application/json"})
    assert up.last.url.port == 8002
    # No model was named, so none is invented: the body crosses as it stands.
    assert up.last.content == sent


@pytest.mark.parametrize("path", ["/health", "/metrics", "/tokenize", "/detokenize"])
def test_non_v1_paths_pass_through_to_the_routed_model(path):
    app, up = rig(two_model_table())
    call(app, "GET", path)
    assert up.last.url.port == 8002
    assert up.last.url.path == path


def test_tokenize_routes_by_the_model_in_its_body():
    app, up = rig(two_model_table())
    call(app, "POST", "/tokenize", json={"model": "lfm2", "prompt": "hi"})
    assert up.last.url.port == 8007
    assert json.loads(up.last.content)["model"] == "LFM2.5-350M"


def test_model_named_late_in_a_large_body_still_routes_correctly():
    """A client is free to put ``model`` after a megabyte of base64 image. The
    scan must keep reading rather than give up and route to the main slot."""
    blob = "A" * 1_200_000
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + blob}}],
            }
        ],
        "model": "lfm2",
    }
    sent = json.dumps(body).encode()
    app, up = rig(two_model_table())
    call(app, "POST", "/v1/chat/completions", content=sent,
         headers={"content-type": "application/json"})
    assert up.last.url.port == 8007
    assert len(up.last.content) > 1_000_000
    assert json.loads(up.last.content)["model"] == "LFM2.5-350M"


# ---------------------------------------------------------------------------
# Error bodies
# ---------------------------------------------------------------------------


def test_unknown_model_is_404_listing_the_known_names():
    app, up = rig(two_model_table())
    r = call(app, "POST", "/v1/chat/completions", json={"model": "qwen38-flash-next", "messages": []})
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "model_not_found"
    assert err["type"] == "invalid_request_error"
    assert "qwen38-flash-next" in err["message"]
    for name in ("glm53-flash", "glm53", "glm53-flash-high", "LFM2.5-350M", "lfm2"):
        assert name in err["message"]
    assert not up.requests, "an unknown model must never reach a port"


def test_known_but_not_live_is_503_naming_what_holds_the_main_slot():
    app, up = rig(two_model_table(flash_live=False))
    r = call(app, "POST", "/v1/chat/completions", json={"model": "lfm2", "messages": []})
    assert r.status_code == 200  # the resident is live and unaffected
    r = call(app, "POST", "/v1/chat/completions", json={"model": "glm53", "messages": []})
    assert r.status_code == 503
    assert r.headers["Retry-After"] == "15"
    err = r.json()["error"]
    assert err["type"] == "model_not_running"
    assert err["code"] == "not_running"
    assert err["message"] == "glm53-flash is not running (main slot: glm53-flash, still starting)"


def test_503_names_the_other_model_that_holds_the_slot():
    flash = Route(model_id="glm53-flash", port=8002, live=True, policies=RoutePolicies(ctx=327680))
    booting = Route(model_id="Qwen3.8-27B-NVFP4", port=8004, live=False, policies=RoutePolicies(ctx=262144))
    table = FakeRouteTable([flash, booting], main_name="glm53-flash")
    app, _ = rig(table)
    r = call(app, "POST", "/v1/chat/completions", json={"model": "Qwen3.8-27B-NVFP4"})
    assert r.status_code == 503
    assert r.json()["error"]["message"] == (
        "Qwen3.8-27B-NVFP4 is not running (main slot: glm53-flash)"
    )


def test_no_model_named_and_no_main_slot_is_503():
    table = FakeRouteTable([lfm2_route()], main_name=None)
    app, up = rig(table)
    r = call(app, "GET", "/health")
    assert r.status_code == 503
    assert r.headers["Retry-After"] == "15"
    assert r.json()["error"]["message"] == "no model is running (main slot: empty)"
    assert not up.requests


def test_upstream_connect_failure_is_502_not_503():
    """The registry says live and the socket disagrees. That is servedeck being
    wrong about the world, not a model that is merely starting — a 503 here
    would tell every client to retry forever against a port with nothing on
    it."""
    app, up = rig(two_model_table())
    up.raise_on_send = httpx.ConnectError("Connection refused")
    r = call(app, "POST", "/v1/chat/completions", json={"model": "lfm2", "messages": []})
    assert r.status_code == 502
    err = r.json()["error"]
    assert err["type"] == "upstream_unavailable"
    assert err["code"] == "upstream_unavailable"
    assert "127.0.0.1:8007" in err["message"]


def test_upstream_error_status_and_body_are_relayed_unchanged():
    app, up = rig(two_model_table())
    up.status = 400
    up.body = b'{"error":{"message":"bad","type":"BadRequestError","code":400}}'
    r = call(app, "POST", "/v1/chat/completions", json={"model": "lfm2", "messages": []})
    assert r.status_code == 400
    assert r.content == up.body


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------


def test_models_lists_every_served_name_of_every_live_route():
    app, up = rig(two_model_table())
    r = call(app, "GET", "/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    assert [e["id"] for e in data] == [
        "glm53-flash",
        "glm53",
        "glm53-flash-high",
        "glm53-flash-low",
        "LFM2.5-350M",
        "lfm2",
    ]
    by_id = {e["id"]: e for e in data}
    # root is the model id, which is how a client learns that four of these
    # entries are the same weights.
    assert by_id["glm53-flash-high"]["root"] == "glm53-flash"
    assert by_id["lfm2"]["root"] == "LFM2.5-350M"
    # max_model_len is published so a client sizes context from the same place
    # it learned the name.
    assert by_id["glm53"]["max_model_len"] == 327680
    assert by_id["lfm2"]["max_model_len"] == 32768
    assert not up.requests, "/v1/models must be answered from the registry, not proxied"


def test_models_omits_a_route_that_is_not_live():
    app, _ = rig(two_model_table(flash_live=False))
    ids = [e["id"] for e in call(app, "GET", "/v1/models").json()["data"]]
    assert ids == ["LFM2.5-350M", "lfm2"]


def test_models_is_empty_when_nothing_is_live():
    app, _ = rig(two_model_table(flash_live=False, lfm_live=False))
    assert call(app, "GET", "/v1/models").json() == {"object": "list", "data": []}


# ---------------------------------------------------------------------------
# Header hygiene
# ---------------------------------------------------------------------------


def test_request_header_hygiene():
    app, up = rig(two_model_table())
    call(
        app,
        "POST",
        "/v1/chat/completions",
        json={"model": "lfm2"},
        headers={
            "authorization": "Bearer secret",
            "x-request-id": "abc",
            "accept-encoding": "gzip, br",
            "connection": "keep-alive",
            "te": "trailers",
            "proxy-connection": "keep-alive",
        },
    )
    sent = up.last.headers
    # Forwarded untouched: anything the model or its logs might want.
    assert sent["authorization"] == "Bearer secret"
    assert sent["x-request-id"] == "abc"
    # Forced, so what upstream writes is what we can relay byte-for-byte.
    assert sent["accept-encoding"] == "identity"
    # The client's Host would have named the gateway's own listener.
    assert sent["host"] == "127.0.0.1:8007"
    assert "proxy-connection" not in sent
    assert "te" not in sent


def test_response_header_hygiene():
    """A gzipped upstream reply (which forcing identity should prevent, but
    which must still be handled) is delivered decoded, so relaying its
    ``content-encoding`` would tell the client to decompress plain bytes."""
    import gzip

    app, up = rig(two_model_table())
    up.body = gzip.compress(b'{"ok":true}')
    up.headers = {
        "x-request-id": "abc",
        "content-encoding": "gzip",
        "connection": "keep-alive",
        "transfer-encoding": "chunked",
    }
    r = call(app, "POST", "/v1/chat/completions", json={"model": "lfm2"})
    assert r.content == b'{"ok":true}'
    assert r.headers["x-request-id"] == "abc"
    # Dropped: the body we hand back is decoded, and may be a different length
    # than upstream's once a policy has mirrored a field into it.
    assert "content-encoding" not in r.headers
    assert "connection" not in r.headers
    assert "transfer-encoding" not in r.headers


def test_request_body_is_streamed_not_buffered_when_no_policy_applies():
    """With no policy to apply and no rewrite needed, the body crosses as a
    stream — which httpx frames as chunked, and which is visible here as the
    absence of a content-length the gateway would only have if it had read the
    whole body first."""
    app, up = rig(two_model_table())
    sent = json.dumps({"model": "LFM2.5-350M", "messages": [{"role": "user", "content": "x" * 5000}]}).encode()
    call(app, "POST", "/v1/chat/completions", content=sent,
         headers={"content-type": "application/json"})
    assert up.last.headers.get("transfer-encoding") == "chunked"
    assert "content-length" not in up.last.headers
    assert up.last.content == sent  # byte-for-byte, not merely equivalent


def test_a_body_on_a_non_post_method_is_forwarded_not_dropped():
    """DELETE with a body is unusual but legal; swallowing it would surface as
    a confusing 400 from the model rather than as a gateway bug."""
    app, up = rig(two_model_table())
    sent = b'{"reason":"cancelled"}'
    call(app, "DELETE", "/v1/responses/resp_123", content=sent,
         headers={"content-type": "application/json"})
    assert up.last.method == "DELETE"
    assert up.last.content == sent


def test_a_get_carries_no_body_upstream():
    app, up = rig(two_model_table())
    call(app, "GET", "/v1/models/LFM2.5-350M")
    assert up.last.content == b""
    assert "transfer-encoding" not in up.last.headers


def test_query_string_and_unknown_paths_reach_upstream():
    app, up = rig(two_model_table())
    call(app, "GET", "/v1/anything/else?a=1&b=two")
    assert up.last.url.path == "/v1/anything/else"
    assert up.last.url.query == b"a=1&b=two"


# ---------------------------------------------------------------------------
# Byte-identical passthrough
# ---------------------------------------------------------------------------


def test_no_policy_means_the_response_is_byte_identical():
    table = FakeRouteTable([lfm2_route()], main_name="LFM2.5-350M")
    app, up = rig(table)
    up.body = load(NONSTREAM)
    r = call(app, "POST", "/v1/chat/completions", json={"model": "LFM2.5-350M"})
    assert r.content == load(NONSTREAM)
    assert "reasoning_content" not in r.text


def test_no_policy_means_the_request_is_byte_identical():
    table = FakeRouteTable([lfm2_route()], main_name="LFM2.5-350M")
    app, up = rig(table)
    sent = b'{"model": "LFM2.5-350M", "messages": [], "trailing garbage'
    call(app, "POST", "/v1/chat/completions", content=sent,
         headers={"content-type": "application/json"})
    assert up.last.content == sent


def test_mirroring_a_stream_that_carries_no_reasoning_is_byte_identical():
    table = FakeRouteTable([lfm2_route(mirror_reasoning=True)], main_name="LFM2.5-350M")
    app, up = rig(table)
    raw = load("chat_stream_noreasoning.sse")
    up.media = "text/event-stream"
    up.chunks = [raw]
    r = call(app, "POST", "/v1/chat/completions", json={"model": "lfm2", "stream": True})
    assert r.content == raw


# ---------------------------------------------------------------------------
# Mirroring, and where it does and does not apply
# ---------------------------------------------------------------------------


def test_mirror_applies_to_a_json_response():
    table = FakeRouteTable([lfm2_route(mirror_reasoning=True)], main_name="LFM2.5-350M")
    app, up = rig(table)
    up.body = load(NONSTREAM)
    r = call(app, "POST", "/v1/chat/completions", json={"model": "LFM2.5-350M"})
    msg = r.json()["choices"][0]["message"]
    assert msg["reasoning_content"] == msg["reasoning"]


def test_mirror_applies_to_an_sse_response():
    table = FakeRouteTable([lfm2_route(mirror_reasoning=True)], main_name="LFM2.5-350M")
    app, up = rig(table)
    up.media = "text/event-stream"
    up.chunks = [load(STREAM)]
    r = call(app, "POST", "/v1/chat/completions", json={"model": "LFM2.5-350M", "stream": True})
    events = sse_events(r.content)
    assert reassemble(events, "reasoning_content") == reassemble(sse_events(load(STREAM)), "reasoning")


@pytest.mark.parametrize("media", ["text/plain; charset=utf-8", "application/octet-stream"])
def test_mirror_is_decided_by_the_upstream_content_type(media):
    """Whether a response is transformed is decided by what upstream said it
    is, never by whether the request asked to stream.  A metrics scrape whose
    text happens to contain the field name must cross untouched."""
    table = FakeRouteTable([lfm2_route(mirror_reasoning=True)], main_name="LFM2.5-350M")
    app, up = rig(table)
    up.media = media
    up.body = b'vllm:reasoning{"reasoning":"not json at all"} 1.0\n'
    r = call(app, "POST", "/v1/chat/completions", json={"model": "LFM2.5-350M"})
    assert r.content == up.body
    assert b"reasoning_content" not in r.content


def test_responses_endpoint_is_never_mirrored():
    """Codex speaks /v1/responses, where reasoning is already a first-class
    output item: mirroring there would add a field to a shape that never
    lacked it."""
    table = FakeRouteTable([lfm2_route(mirror_reasoning=True)], main_name="LFM2.5-350M")
    app, up = rig(table)
    up.body = load(NONSTREAM)
    r = call(app, "POST", "/v1/responses", json={"model": "LFM2.5-350M"})
    assert r.content == load(NONSTREAM)
    assert "reasoning_content" not in r.text


# ---------------------------------------------------------------------------
# Streaming chunk boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [1, 3, 17, 250])
def test_stream_output_does_not_depend_on_chunk_boundaries(size):
    table = FakeRouteTable([lfm2_route(mirror_reasoning=True)], main_name="LFM2.5-350M")
    raw = load(STREAM)

    def run(chunks):
        app, up = rig(table)
        up.media = "text/event-stream"
        up.chunks = chunks
        return call(app, "POST", "/v1/chat/completions",
                    json={"model": "lfm2", "stream": True}).content

    shredded = [raw[i : i + size] for i in range(0, len(raw), size)]
    assert run(shredded) == run([raw])


def test_stream_survives_a_split_mid_utf8_character():
    """A chunk boundary inside a multi-byte character must not corrupt it: a
    naive per-chunk ``.decode()`` would raise, and a lossy one would replace
    the character."""
    table = FakeRouteTable([lfm2_route(mirror_reasoning=True)], main_name="LFM2.5-350M")
    frame = json.dumps(
        {"choices": [{"index": 0, "delta": {"reasoning": "héllo — wörld ✅"}, "finish_reason": None}]},
        ensure_ascii=False,
    ).encode()
    raw = b"data: " + frame + b"\n\ndata: [DONE]\n\n"
    app, up = rig(table)
    up.media = "text/event-stream"
    up.chunks = [raw[i : i + 1] for i in range(len(raw))]
    out = call(app, "POST", "/v1/chat/completions", json={"model": "lfm2", "stream": True}).content
    delta = sse_events(out)[0]["choices"][0]["delta"]
    assert delta["reasoning"] == delta["reasoning_content"] == "héllo — wörld ✅"


def test_stream_with_a_partial_trailing_line_is_still_delivered():
    """An upstream that ends without a final newline must not have its last
    event swallowed by the hold-back."""
    table = FakeRouteTable([lfm2_route(mirror_reasoning=True)], main_name="LFM2.5-350M")
    app, up = rig(table)
    up.media = "text/event-stream"
    up.chunks = [b'data: {"choices":[{"delta":{"reasoning":"tail"}}]}']
    out = call(app, "POST", "/v1/chat/completions", json={"model": "lfm2", "stream": True}).content
    assert b'"reasoning_content":"tail"' in out


# ---------------------------------------------------------------------------
# Per-route request policies
# ---------------------------------------------------------------------------


def test_effort_preset_injects_its_overlay_and_rewrites_the_model():
    high = Route(
        model_id="glm53-flash",
        port=8002,
        live=True,
        presets=("glm53-flash-high",),
        policies=RoutePolicies(effort_overlay={"reasoning_effort": "high"}, ctx=327680),
    )
    app, up = rig(FakeRouteTable([high], main_name="glm53-flash"))
    call(app, "POST", "/v1/chat/completions", json={"model": "glm53-flash-high", "messages": []})
    sent = json.loads(up.last.content)
    assert sent["model"] == "glm53-flash"
    assert sent["chat_template_kwargs"] == {"reasoning_effort": "high"}


def test_output_floor_is_applied_per_route():
    table = FakeRouteTable(
        [lfm2_route(min_output_tokens=4096)], main_name="LFM2.5-350M"
    )
    app, up = rig(table)
    call(app, "POST", "/v1/chat/completions",
         json={"model": "LFM2.5-350M", "max_tokens": 64, "messages": []})
    assert json.loads(up.last.content)["max_tokens"] == 4096


def test_output_floor_applies_on_the_responses_endpoint_too():
    """/v1/responses is routed and floored and nothing else — the empty-turn
    failure the floor prevents is not specific to chat/completions."""
    table = FakeRouteTable([lfm2_route(min_output_tokens=4096)], main_name="LFM2.5-350M")
    app, up = rig(table)
    call(app, "POST", "/v1/responses",
         json={"model": "lfm2", "max_output_tokens": 32, "input": "hi"})
    sent = json.loads(up.last.content)
    assert sent["max_output_tokens"] == 4096
    assert sent["model"] == "LFM2.5-350M"


def test_a_route_with_no_policies_never_parses_the_request():
    """Proved by handing it a body that json.loads cannot read: if any policy
    had run, the request would have been dropped or mangled."""
    table = FakeRouteTable([lfm2_route()], main_name="LFM2.5-350M")
    app, up = rig(table)
    sent = b'{"model":"LFM2.5-350M","messages":[}}}not json'
    call(app, "POST", "/v1/chat/completions", content=sent,
         headers={"content-type": "application/json"})
    assert up.last.content == sent
