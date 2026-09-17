"""End-to-end: the gateway in front of the REAL LFM2.5-350M on :8007.

REDESIGN-2026-09-12 §2.7's point: the paths that actually fail are the live
ones, and nearly every test in this repository runs against a mock.  So this
file runs the router under uvicorn on an ephemeral port and drives it with a
real HTTP client against the always-on 350M resident — a model that costs
3 GiB and answers in under a second, so a full pass is seconds, not minutes.

It is the only place three things can be checked at all:

* that a **chunked** request body (what the gateway sends when no policy needs
  the body, so that it never has to buffer one) is accepted by a real vLLM;
* that response deltas arrive **incrementally** — httpx's ``ASGITransport``
  collects a streaming body before returning it, so an in-process test cannot
  see the difference between a stream and a buffer, however the gateway
  behaves;
* that a tool call survives the round trip with its parsed ``tool_calls``
  intact (``--tool-call-parser lfm2`` is on upstream).

If :8007 is not answering, every test here skips with the reason.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from servedeck import gateway
from tests.fake_routes import FakeRouteTable, lfm2_route

UPSTREAM = "http://127.0.0.1:8007"
MODEL_ID = "LFM2.5-350M"
ALIAS = "lfm2"


def _upstream_reason() -> str | None:
    """None when :8007 is serving LFM2.5-350M; otherwise why it is not."""
    try:
        r = httpx.get(f"{UPSTREAM}/v1/models", timeout=5.0)
    except Exception as exc:
        return f"{UPSTREAM} is not answering ({type(exc).__name__}: {exc})"
    if r.status_code != 200:
        return f"{UPSTREAM}/v1/models returned HTTP {r.status_code}"
    ids = [m.get("id") for m in r.json().get("data", [])]
    if MODEL_ID not in ids:
        return f"{UPSTREAM} serves {ids}, not {MODEL_ID}"
    return None


SKIP_REASON = _upstream_reason()
pytestmark = pytest.mark.skipif(SKIP_REASON is not None, reason=SKIP_REASON or "")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _Server(uvicorn.Server):
    def install_signal_handlers(self) -> None:  # background thread: no signals
        pass


@pytest.fixture(scope="module")
def base_url():
    """The gateway, on an ephemeral loopback port, in front of the real model.

    The route table has exactly one live route: LFM2.5-350M on 8007 with the
    alias ``lfm2``, mirroring on (the model answers ``"reasoning": null`` on
    every turn, so the mirror must prove it can leave a real reply alone).
    """
    table = FakeRouteTable([lfm2_route(port=8007, mirror_reasoning=True)], main_name=MODEL_ID)
    router = gateway.build_router(table)
    client = router.gateway_client

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)

    port = _free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        # h11 + the plain asyncio loop: uvloop/httptools buffer differently and
        # would mask the flush behaviour this file exists to observe.
        loop="asyncio",
        http="h11",
    )
    server = _Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    if not server.started:
        raise RuntimeError("gateway under test did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


# ---------------------------------------------------------------------------


def test_models_lists_the_id_and_the_alias(base_url):
    r = httpx.get(f"{base_url}/v1/models", timeout=15)
    assert r.status_code == 200
    entries = {m["id"]: m for m in r.json()["data"]}
    assert set(entries) == {MODEL_ID, ALIAS}
    assert entries[ALIAS]["root"] == MODEL_ID
    assert entries[ALIAS]["max_model_len"] == 32768


def test_nonstreaming_chat_reaches_the_model(base_url):
    r = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "Reply with exactly the word: pineapple"}],
            "max_tokens": 32,
            "temperature": 0,
        },
        timeout=120,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == MODEL_ID
    assert "pineapple" in body["choices"][0]["message"]["content"].lower()
    # The mirror is on and the model reports no reasoning: it must stay absent
    # rather than gain an invented empty twin.
    msg = body["choices"][0]["message"]
    assert msg.get("reasoning") is None and msg.get("reasoning_content") is None


def test_a_request_naming_the_alias_reaches_the_model(base_url):
    """:8007 was started with ``--served-model-name LFM2.5-350M`` only, so a
    request naming ``lfm2`` reaching the model at all is proof the gateway
    rewrote it.  An unrewritten alias comes back as a 404 from vLLM."""
    r = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": ALIAS,
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 16,
            "temperature": 0,
        },
        timeout=120,
    )
    assert r.status_code == 200, r.text
    assert r.json()["model"] == MODEL_ID


def test_unknown_model_is_404_before_the_model_is_touched(base_url):
    r = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={"model": "a-model-that-was-never-registered", "messages": []},
        timeout=30,
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"
    assert MODEL_ID in r.json()["error"]["message"]


def test_streaming_chat_delivers_deltas_incrementally(base_url):
    """Time-to-first-chunk must be a small fraction of time-to-last-chunk.

    A gateway that buffered the response would deliver everything at once and
    make these two times equal — which is exactly the failure that would make
    an editor's chat view sit blank for the whole generation and then paint.
    """
    chunks: list[tuple[float, bytes]] = []
    body = {
        "model": ALIAS,
        "messages": [
            {"role": "user", "content": "Write out every number from 1 to 200, one per line, nothing else."}
        ],
        "max_tokens": 700,
        "temperature": 0,
        "stream": True,
    }
    start = time.perf_counter()
    with httpx.stream("POST", f"{base_url}/v1/chat/completions", json=body, timeout=180) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for chunk in r.iter_bytes():
            if chunk:
                chunks.append((time.perf_counter() - start, chunk))

    total = sum(len(c) for _, c in chunks)
    t_first, t_last = chunks[0][0], chunks[-1][0]
    # Two independent witnesses, so neither a fast box nor a slow one can make
    # this pass for the wrong reason.  (1) Timing: the first chunk lands in the
    # first few percent of the stream.  (2) Shape: the body arrives in many
    # small reads rather than a few large ones — a buffered 60 KB response
    # reaches the client in a handful of ~16 KB reads, never in hundreds of
    # ~200-byte ones, whatever the clock says.
    assert len(chunks) >= 20, f"only {len(chunks)} network reads — the stream was buffered"
    assert total / len(chunks) < 2000, (
        f"{total / len(chunks):.0f} bytes per read — that is a buffer being drained, not SSE frames"
    )
    assert t_last - t_first > 0.05, "the generation was too short to tell a stream from a buffer"
    assert t_first < t_last * 0.4, (
        f"first chunk at {t_first:.3f}s of a {t_last:.3f}s stream — the response is being buffered"
    )

    raw = b"".join(c for _, c in chunks)
    assert raw.rstrip().endswith(b"data: [DONE]")
    text = "".join(
        (ch.get("delta") or {}).get("content") or ""
        for line in raw.split(b"\n")
        if line.startswith(b"data: ") and line[6:].strip() not in (b"[DONE]", b"")
        for ch in json.loads(line[6:]).get("choices") or []
    )
    assert "1" in text and "2" in text


def test_streaming_tool_call_round_trips(base_url):
    """LFM2 emits a parsed ``tool_calls`` entry for a weather question when
    ``tools`` are offered; the gateway must carry the assembled call through
    without disturbing its arguments."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City name"}},
                    "required": ["city"],
                },
            },
        }
    ]
    body = {
        "model": ALIAS,
        "messages": [{"role": "user", "content": "What is the weather in Paris right now?"}],
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 200,
        "temperature": 0,
    }
    r = httpx.post(f"{base_url}/v1/chat/completions", json=body, timeout=180)
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    assert calls, f"no tool call in {choice}"
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"])["city"].lower().startswith("paris")
    assert choice["finish_reason"] == "tool_calls"


def test_a_chunked_request_body_is_accepted_by_the_real_server(base_url):
    """The no-policy path streams the request body upstream, which httpx frames
    as ``Transfer-Encoding: chunked``.  Nothing else in the suite can tell
    whether a real vLLM accepts that; if it did not, every request through the
    gateway would fail the moment a route had no policies."""
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "x" * 20000 + "\nReply with the word: ok"}],
        "max_tokens": 16,
        "temperature": 0,
    }

    async def send_chunked() -> httpx.Response:
        raw = json.dumps(payload).encode()

        async def body():
            for i in range(0, len(raw), 4096):
                yield raw[i : i + 4096]

        async with httpx.AsyncClient(timeout=180) as c:
            return await c.post(
                f"{base_url}/v1/chat/completions",
                content=body(),
                headers={"content-type": "application/json"},
            )

    r = asyncio.run(send_chunked())
    assert r.status_code == 200, r.text
    assert r.json()["model"] == MODEL_ID


def test_health_and_tokenize_pass_through_to_the_routed_model(base_url):
    assert httpx.get(f"{base_url}/health", timeout=30).status_code == 200
    r = httpx.post(
        f"{base_url}/tokenize",
        json={"model": ALIAS, "prompt": "hello world"},
        timeout=30,
    )
    assert r.status_code == 200, r.text
    assert r.json()["count"] > 0
