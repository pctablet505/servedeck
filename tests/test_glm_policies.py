"""Tests for ``servedeck.glm_policies`` — P9, the GLM-5.3 client-compat policies.

GLM-5.3 needs 90+ GiB to boot and is not running, so nothing here talks to it.
Three instruments stand in, and which one proves what is stated per test:

* **hand-built payloads carrying the real leaked-tag shapes** quoted in
  ``glm53-effort-proxy/proxy.py``'s own comments — ``tests/fixtures/glm/``.
  The failure those reproduce (``list_dir({"path</arg_key>": ...})``) was
  captured from a live Copilot turn; the *envelopes* around it are synthetic,
  modelled on the recorded vLLM replies in ``tests/fixtures/reasoning/``.
* **recorded fixtures** (``tests/fixtures/reasoning/``) from a real vLLM, so
  the transforms are proved not to disturb genuine traffic.
* **the real LFM2.5-350M on :8007** — ``tests/test_glm_policies_e2e.py``, for
  anything shaped like a live round trip.

What only a live GLM can settle is listed at the bottom of this file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib

import httpx
import pytest
from fastapi import FastAPI

from servedeck import gateway, glm_policies, models
from servedeck.gateway import Route, RoutePolicies
from tests.fake_routes import FakeRouteTable, lfm2_route, sse_events

GLM_FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "glm"
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

LEAKED_JSON = "chat_nonstream_leaked_tag.json"
CLEAN_JSON = "chat_nonstream_clean_toolcall.json"
ANSWERLESS_JSON = "chat_nonstream_answerless.json"
ECHO_REQUEST = "chat_request_copilot_echo.json"
LEAKED_SSE = "chat_stream_leaked_tag.sse"
ANSWERLESS_SSE = "chat_stream_answerless.sse"


def glm_fixture(name: str) -> bytes:
    return (GLM_FIXTURES / name).read_bytes()


# ---------------------------------------------------------------------------
# Harness (same shape as tests/test_gateway.py's, so the two read alike)
# ---------------------------------------------------------------------------


class Upstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.bodies: list[bytes] = []
        self.status = 200
        self.media = "application/json"
        self.body: bytes = b"{}"
        self.chunks: list[bytes] | None = None

    @property
    def last(self) -> httpx.Request:
        assert self.requests, "upstream was never called"
        return self.requests[-1]

    @property
    def last_body(self) -> dict:
        return json.loads(self.bodies[-1])

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.bodies.append(request.read())
        headers = {"content-type": self.media}
        if self.chunks is None:
            return httpx.Response(self.status, headers=headers, content=self.body)
        pieces = list(self.chunks)

        async def gen():
            for piece in pieces:
                yield piece

        return httpx.Response(self.status, headers=headers, content=gen())


def glm_route(*, live: bool = True, **policy_kwargs) -> Route:
    """The glm53 route as ``models.toml`` describes it, unless overridden."""
    kwargs = {"mirror_reasoning": True, "min_output_tokens": 8192, "sanitize_tool_tags": True}
    kwargs.update(policy_kwargs)
    return Route(
        model_id="glm53-flash",
        port=8002,
        live=live,
        aliases=("glm53",),
        presets=("glm53-flash-high",),
        policies=RoutePolicies(ctx=327680, **kwargs),
    )


def rig(*routes: Route) -> tuple[FastAPI, Upstream, glm_policies.GlmState]:
    table = FakeRouteTable(list(routes) or [glm_route()], main_name="glm53-flash")
    upstream = Upstream()
    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream.handler))
    router = gateway.build_router(table, client=client)
    app = FastAPI()
    app.include_router(router)
    return app, upstream, router.glm_state


def call(app: FastAPI, method: str, path: str, **kwargs) -> httpx.Response:
    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
            return await c.request(method, path, **kwargs)

    return asyncio.run(go())


def post_chat(app, body: dict | bytes, **kwargs) -> httpx.Response:
    if isinstance(body, bytes):
        return call(app, "POST", "/v1/chat/completions", content=body,
                    headers={"content-type": "application/json"}, **kwargs)
    return call(app, "POST", "/v1/chat/completions", json=body, **kwargs)


def sse_chunks(raw: bytes, size: int) -> list[bytes]:
    return [raw[i : i + size] for i in range(0, len(raw), size)] or [b""]


def tool_arguments(events: list[dict], index: int = 0) -> str:
    """Reassemble one tool call's ``arguments`` across a parsed SSE stream —
    what a client actually ends up with."""
    parts = []
    for ev in events:
        for ch in ev.get("choices") or []:
            for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                if tc.get("index", 0) != index:
                    continue
                args = (tc.get("function") or {}).get("arguments")
                if isinstance(args, str):
                    parts.append(args)
    return "".join(parts)


def only_message(reply: httpx.Response) -> dict:
    return reply.json()["choices"][0]["message"]


# ===========================================================================
# Item 1 — tool-tag sanitiser
# ===========================================================================


def test_leaked_tag_reaches_the_client_without_the_sanitiser():
    """FAIL-BEFORE for item 1, non-streaming.

    With ``sanitize_tool_tags`` off — which is what the gateway did before this
    packet, for every route including glm53 — the exact captured shape reaches
    the client: an argument object whose key is ``path</arg_key>``, which is
    the key VS Code rejects with "must have required property 'path'".
    """
    app, up, _ = rig(glm_route(sanitize_tool_tags=False))
    up.body = glm_fixture(LEAKED_JSON)
    reply = post_chat(app, {"model": "glm53-flash", "messages": []})
    args = json.loads(only_message(reply)["tool_calls"][0]["function"]["arguments"])
    assert list(args) == ["path</arg_key>"]


def test_leaked_tag_in_a_parsed_key_is_repaired():
    """The same turn with the switch on: the client gets a usable call."""
    app, up, state = rig(glm_route())
    up.body = glm_fixture(LEAKED_JSON)
    reply = post_chat(app, {"model": "glm53-flash", "messages": []})
    args = json.loads(only_message(reply)["tool_calls"][0]["function"]["arguments"])
    assert args == {"path": "/Users/user/Desktop/project"}
    assert state.stats()["glm53-flash"]["tool_call_repairs"] == 1


def test_a_healthy_tool_call_is_relayed_byte_identically():
    """Over-correction guard, and the reason this ships enabled.

    A reply with no leaked markup must come back exactly as upstream wrote it —
    including an argument value that legitimately ends in a space, which the
    original's unconditional ``.strip()`` rewrote and counted as a repair.
    """
    app, up, state = rig(glm_route(mirror_reasoning=False))
    up.body = glm_fixture(CLEAN_JSON)
    reply = post_chat(app, {"model": "glm53-flash", "messages": []})
    assert reply.content == up.body
    assert state.stats()["glm53-flash"]["tool_call_repairs"] == 0


def test_clean_fragment_leaves_a_tagless_string_alone():
    """The same property one layer down, where it is a one-line contract."""
    assert glm_policies.clean_fragment("line one ") == "line one "
    assert glm_policies.clean_fragment(" padded ") == " padded "
    assert glm_policies.clean_fragment("path</arg_key>") == "path"


def test_leaked_tag_split_across_two_deltas_is_repaired():
    """The streaming case, and the one a per-frame sanitiser cannot see.

    The tag leaked upstream *because* it straddled a token boundary, so it
    arrives split: ``{"path</arg`` then ``_key>": "/Users/...``.  What the
    client reassembles must still be the JSON object the tool declared.
    """
    app, up, state = rig(glm_route())
    up.media = "text/event-stream"
    up.chunks = [glm_fixture(LEAKED_SSE)]
    reply = post_chat(app, {"model": "glm53-flash", "messages": [], "stream": True})
    args = tool_arguments(sse_events(reply.content))
    assert json.loads(args) == {"path": "/Users/user/Desktop/project"}
    assert state.stats()["glm53-flash"]["tool_call_repairs"] >= 1


def test_split_leaked_tag_survives_the_stream_without_the_sanitiser():
    """FAIL-BEFORE for item 1, streaming."""
    app, up, _ = rig(glm_route(sanitize_tool_tags=False))
    up.media = "text/event-stream"
    up.chunks = [glm_fixture(LEAKED_SSE)]
    reply = post_chat(app, {"model": "glm53-flash", "messages": [], "stream": True})
    args = tool_arguments(sse_events(reply.content))
    # The reassembled arguments are still *valid JSON* — that is what makes
    # this defect so confusing in an editor. The KEY is wrong, so the client
    # rejects the call with "must have required property 'path'".
    assert list(json.loads(args)) == ["path</arg_key>"]


@pytest.mark.parametrize("size", [1, 3, 17, 250, 10_000])
def test_stream_output_does_not_depend_on_chunk_boundaries(size):
    """The property that makes incremental delivery correct: upstream may cut
    the byte stream anywhere, including mid-tag and mid-JSON."""
    raw = glm_fixture(LEAKED_SSE)
    app, up, _ = rig(glm_route())
    up.media = "text/event-stream"
    up.chunks = sse_chunks(raw, size)
    reply = post_chat(app, {"model": "glm53-flash", "messages": [], "stream": True})
    args = tool_arguments(sse_events(reply.content))
    assert json.loads(args) == {"path": "/Users/user/Desktop/project"}


def test_a_recorded_real_tool_call_stream_is_relayed_unchanged():
    """Over-correction guard on the streaming path, against traffic recorded
    from a real vLLM (``tests/fixtures/reasoning/chat_stream_toolcall.sse``)."""
    from tests.fake_routes import load

    raw = load("chat_stream_toolcall.sse")
    app, up, state = rig(glm_route(mirror_reasoning=False))
    up.media = "text/event-stream"
    up.chunks = [raw]
    reply = post_chat(app, {"model": "glm53-flash", "messages": [], "stream": True})
    assert reply.content == raw
    assert state.stats()["glm53-flash"]["tool_call_repairs"] == 0


def test_held_back_bytes_are_delivered_when_a_stream_ends_early():
    """A ``<`` at the end of an argument is held back (it could start a tag).
    If the stream then ends without a finish_reason, those bytes are real
    argument text and must still reach the client."""
    frame = {
        "id": "x", "object": "chat.completion.chunk", "created": 1, "model": "glm53-flash",
        "choices": [{"index": 0, "finish_reason": None, "delta": {
            "tool_calls": [{"index": 0, "function": {"arguments": "{\"expr\": \"a<"}}]}}],
    }
    app, up, _ = rig(glm_route())
    up.media = "text/event-stream"
    up.chunks = [b"data: " + json.dumps(frame).encode() + b"\n\n"]
    reply = post_chat(app, {"model": "glm53-flash", "messages": [], "stream": True})
    assert tool_arguments(sse_events(reply.content)) == '{"expr": "a<'


def test_nested_leaked_tag_is_repaired():
    """The captured failure was a top-level key; the same leak inside a nested
    edit list is the same defect, and the original stopped at depth 1."""
    msg = {"tool_calls": [{"function": {"name": "apply_edits", "arguments": json.dumps(
        {"edits": [{"path</arg_key>": "/a", "body<arg_value>": "x"}]})}}]}
    assert glm_policies.sanitize_tool_calls(msg) == 1
    args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
    assert args == {"edits": [{"path": "/a", "body": "x"}]}


def test_arguments_that_only_parse_after_stripping_are_recovered():
    """The leak can break the JSON itself, not just a key inside it."""
    msg = {"tool_calls": [{"function": {"arguments": '{"path": "/a"}</tool_call>'}}]}
    assert glm_policies.sanitize_tool_calls(msg) == 1
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"path": "/a"}


def test_unparseable_arguments_with_no_tags_are_left_exactly_alone():
    """Handing a client a *differently* broken string is not an improvement,
    and the untouched original is what a failure report needs."""
    msg = {"tool_calls": [{"function": {"arguments": "{not json at all"}}]}
    assert glm_policies.sanitize_tool_calls(msg) == 0
    assert msg["tool_calls"][0]["function"]["arguments"] == "{not json at all"


def test_removal_that_exposes_a_new_tag_reaches_a_fixed_point():
    assert glm_policies.strip_leaked_tags("<<arg_key>arg_key>") == ""


def test_the_request_history_is_sanitised_too():
    """A transcript poisoned before this packet shipped is still in editor
    sessions, and it goes back upstream on every turn."""
    app, up, _ = rig(glm_route())
    body = {"model": "glm53-flash", "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"id": "t1", "type": "function", "function": {
            "name": "list_dir",
            "arguments": "{\"path</arg_key>\": \"/Users/user/Desktop/project\"}"}}]},
    ]}
    post_chat(app, body)
    sent = up.last_body["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(sent) == {"path": "/Users/user/Desktop/project"}


def test_other_routes_are_untouched_by_the_sanitiser():
    """The switch travels with the model: LFM2 sets none of the three, so it
    gets no hook at all and its bytes are relayed as they stand."""
    app, up, state = rig(lfm2_route())
    up.body = glm_fixture(LEAKED_JSON)
    reply = post_chat(app, {"model": "LFM2.5-350M", "messages": []})
    assert reply.content == up.body
    assert state.stats() == {}


# ===========================================================================
# Item 2 — reasoning restore by tool-call id
# ===========================================================================


def remember_then_restore(state_route: str = "glm53-flash", **policy) -> tuple:
    """Two turns: a reply carrying reasoning + a tool_call id, then a request
    that echoes the id back with the reasoning dropped (what Copilot does)."""
    route = glm_route(**policy)
    app, up, state = rig(route)
    up.body = glm_fixture(LEAKED_JSON)          # carries reasoning + the id
    post_chat(app, {"model": state_route, "messages": []})
    up.body = b'{"choices":[]}'
    post_chat(app, glm_fixture(ECHO_REQUEST))
    return app, up, state


def test_reasoning_is_not_restored_without_the_switch():
    """FAIL-BEFORE for item 2 — and the DEFAULT.

    ``restore_reasoning`` is false in ``models.toml``, so out of the box the
    echoed assistant turn still reaches GLM with no reasoning and its template
    still renders an empty ``<think></think>``.  That is deliberate, not a
    gap: see ``glm_policies`` on the Xid-31 correlation.
    """
    _, up, state = remember_then_restore()
    assert "reasoning_content" not in up.last_body["messages"][2]
    assert state.stats()["glm53-flash"]["reasoning_remembered"] == 0


def test_reasoning_is_restored_onto_the_echoed_tool_call():
    _, up, state = remember_then_restore(restore_reasoning=True)
    turn = up.last_body["messages"][2]
    assert turn["reasoning_content"].startswith("The user asked what is in the project")
    assert state.stats()["glm53-flash"]["reasoning_restored"] == 1
    assert state.stats()["glm53-flash"]["reasoning_remembered"] == 1


def test_reasoning_is_remembered_from_a_real_stream():
    """The path that actually matters: Copilot streams, so a restore that only
    learned from non-streaming replies would have nothing to restore on the
    one client that hits the bug (``proxy.py:376-382``)."""
    app, up, state = rig(glm_route(restore_reasoning=True))
    up.media = "text/event-stream"
    up.chunks = [glm_fixture(LEAKED_SSE)]
    post_chat(app, {"model": "glm53-flash", "messages": [], "stream": True})
    up.media, up.chunks, up.body = "application/json", None, b'{"choices":[]}'
    post_chat(app, glm_fixture(ECHO_REQUEST))
    turn = up.last_body["messages"][2]
    assert turn["reasoning_content"] == "They want a directory listing. I will call list_dir."
    assert state.stats()["glm53-flash"]["reasoning_remembered"] == 1


def test_a_model_switch_cannot_inject_the_other_models_reasoning():
    """The guard the original did not have.

    GLM answers, its reasoning is remembered against a tool_call id.  GLM is
    then switched out and another model takes the slot.  A client that echoes
    the same id must NOT have GLM's thinking pasted into the new model's
    prompt — a prompt built from another model's markup is not merely wrong,
    it is malformed.
    """
    glm = glm_route(restore_reasoning=True)
    other = Route(
        model_id="qwen38-27b", port=8001, live=True,
        policies=RoutePolicies(ctx=262144, restore_reasoning=True),
    )
    table = FakeRouteTable([glm, other], main_name="glm53-flash")
    up = Upstream()
    client = httpx.AsyncClient(transport=httpx.MockTransport(up.handler))
    router = gateway.build_router(table, client=client)
    app = FastAPI()
    app.include_router(router)

    up.body = glm_fixture(LEAKED_JSON)
    post_chat(app, {"model": "glm53-flash", "messages": []})
    assert router.glm_state.stats()["glm53-flash"]["reasoning_remembered"] == 1

    up.body = b'{"choices":[]}'
    echo = json.loads(glm_fixture(ECHO_REQUEST))
    echo["model"] = "qwen38-27b"
    post_chat(app, echo)
    assert "reasoning_content" not in up.last_body["messages"][2]
    assert router.glm_state.stats()["qwen38-27b"]["reasoning_restored"] == 0


def test_the_store_itself_refuses_another_models_id():
    """Belt and braces: the per-route bucket already separates models, and the
    entry carries its owner as well, because one guard on a path that writes
    into a prompt is one guard too few."""
    store = glm_policies.ReasoningStore("glm53-flash")
    store.remember(["t1"], "thinking")
    assert store.get("t1", model_id="glm53-flash") == "thinking"
    assert store.get("t1", model_id="qwen38-27b") is None


def test_forget_drops_the_reasoning_and_keeps_the_counters():
    state = glm_policies.GlmState()
    route = state.route("glm53-flash")
    route.reasoning.remember(["t1"], "thinking")
    route.stats.answerless_turns = 3
    state.forget("glm53-flash")
    assert state.route("glm53-flash").reasoning.get("t1", model_id="glm53-flash") is None
    assert state.answerless_turns("glm53-flash") == 3


def test_the_store_is_bounded_and_evicts_the_oldest_entry():
    """The original cleared the WHOLE dict at the cap, throwing away the
    in-flight thread's reasoning along with the stale entries."""
    store = glm_policies.ReasoningStore("m", max_entries=3)
    for i in range(5):
        store.remember([f"t{i}"], f"r{i}")
    assert len(store) == 3
    assert store.get("t0", model_id="m") is None
    assert store.get("t4", model_id="m") == "r4"


def test_restored_text_is_bounded_and_prompt_markup_is_stripped():
    """A restored block is prompt tokens on every later turn, and a ``</think>``
    inside it would terminate the block the template puts it in."""
    text = "</think>keep<|user|>" + "x" * 6000
    safe = glm_policies.safe_reasoning(text)
    assert "</think>" not in safe and "<|user|>" not in safe
    assert safe.startswith("keep")
    assert len(safe) == glm_policies.MAX_RESTORED_CHARS + len(" ...")


def test_only_turns_after_the_last_user_message_are_restored():
    """Earlier turns are closed; their thinking is not read by the template,
    and paying prompt tokens for it every turn would be a regression."""
    store = glm_policies.ReasoningStore("m")
    store.remember(["old", "new"], "thinking")
    body = {"messages": [
        {"role": "assistant", "tool_calls": [{"id": "old"}]},
        {"role": "user", "content": "next"},
        {"role": "assistant", "tool_calls": [{"id": "new"}]},
    ]}
    assert glm_policies.restore_reasoning(body, store=store, model_id="m") == 1
    assert "reasoning_content" not in body["messages"][0]
    assert body["messages"][2]["reasoning_content"] == "thinking"


def test_a_turn_that_already_carries_reasoning_is_left_alone():
    store = glm_policies.ReasoningStore("m")
    store.remember(["t1"], "stored")
    body = {"messages": [{"role": "user", "content": "go"},
                         {"role": "assistant", "reasoning_content": "client sent this",
                          "tool_calls": [{"id": "t1"}]}]}
    assert glm_policies.restore_reasoning(body, store=store, model_id="m") == 0
    assert body["messages"][1]["reasoning_content"] == "client sent this"


def test_nothing_is_remembered_when_restore_is_off():
    """Memory is not spent on a switch nobody turned on."""
    app, up, state = rig(glm_route())
    up.body = glm_fixture(LEAKED_JSON)
    post_chat(app, {"model": "glm53-flash", "messages": []})
    assert len(state.route("glm53-flash").reasoning) == 0


# ===========================================================================
# Item 3 — answerless-turn detection (no retry)
# ===========================================================================


def test_an_answerless_reply_is_counted_and_not_retried(caplog):
    """The whole of item 3.  FAIL-BEFORE is the count: without the detection
    the counter does not exist and nothing in the journal names the cause.

    The refusal is the ``len(up.requests) == 1`` line: the original re-ran the
    request at ``reasoning_effort=high``, which on a box with
    ``--max-num-seqs 1`` does not merely cost tokens, it serialises behind and
    delays every other request — and it hides a floor that is set too low.
    """
    app, up, state = rig(glm_route())
    up.body = glm_fixture(ANSWERLESS_JSON)
    with caplog.at_level(logging.WARNING, logger="servedeck.glm_policies"):
        reply = post_chat(app, {"model": "glm53-flash", "messages": []})
    assert len(up.requests) == 1
    assert state.stats()["glm53-flash"]["answerless_turns"] == 1
    assert reply.json()["choices"][0]["message"]["content"] is None
    assert "answerless turn" in caplog.text
    assert "min_output_tokens floor is 8192" in caplog.text
    assert "Not retried on purpose" in caplog.text


def test_an_answerless_stream_is_counted_and_relayed_unchanged(caplog):
    """The original's streaming rescue buffered the whole stream until it saw
    text and, seeing none, re-ran the turn NON-streaming and synthesised two
    chunks.  Here the client gets the model's own stream, byte for byte."""
    raw = glm_fixture(ANSWERLESS_SSE)
    app, up, state = rig(glm_route(mirror_reasoning=False))
    up.media = "text/event-stream"
    up.chunks = [raw]
    with caplog.at_level(logging.WARNING, logger="servedeck.glm_policies"):
        reply = post_chat(app, {"model": "glm53-flash", "messages": [], "stream": True})
    assert len(up.requests) == 1
    assert reply.content == raw
    assert state.stats()["glm53-flash"]["answerless_turns"] == 1
    assert "streaming path" in caplog.text


def test_a_tool_call_with_null_content_is_never_answerless():
    """``content: null`` + ``tool_calls`` is how every tool call is returned;
    counting it would make the metric useless and the warning a log storm."""
    app, up, state = rig(glm_route())
    up.body = glm_fixture(LEAKED_JSON)
    post_chat(app, {"model": "glm53-flash", "messages": []})
    assert state.stats()["glm53-flash"]["answerless_turns"] == 0


def test_a_streamed_tool_call_is_never_answerless():
    app, up, state = rig(glm_route())
    up.media = "text/event-stream"
    up.chunks = [glm_fixture(LEAKED_SSE)]
    post_chat(app, {"model": "glm53-flash", "messages": [], "stream": True})
    assert state.stats()["glm53-flash"]["answerless_turns"] == 0


def test_an_error_reply_is_not_counted_as_answerless():
    """A 400 from vLLM has no ``choices`` and is not a turn at all."""
    app, up, state = rig(glm_route())
    up.status = 400
    up.body = b'{"error":{"message":"bad request","type":"invalid_request_error"}}'
    post_chat(app, {"model": "glm53-flash", "messages": []})
    assert state.stats()["glm53-flash"]["answerless_turns"] == 0


def test_whitespace_only_content_is_still_an_answerless_turn():
    """``content: "   "`` renders as nothing in an editor and goes back into the
    history as an all-but-empty assistant turn, which is the poison. It is the
    same failure as ``content: null`` and is counted the same way."""
    app, up, state = rig(glm_route())
    up.body = b'{"choices":[{"index":0,"message":{"role":"assistant","content":"   "},' \
              b'"finish_reason":"stop"}]}'
    post_chat(app, {"model": "glm53-flash", "messages": []})
    assert state.stats()["glm53-flash"]["answerless_turns"] == 1


def test_structured_content_is_not_an_answerless_turn():
    """A reply whose content is a list of parts (the multimodal shape) has an
    answer in it; ``"".strip()`` on a list would raise, and treating it as
    empty would count every such reply."""
    app, up, state = rig(glm_route())
    up.body = (b'{"choices":[{"index":0,"message":{"role":"assistant",'
               b'"content":[{"type":"text","text":"here"}]},"finish_reason":"stop"}]}')
    post_chat(app, {"model": "glm53-flash", "messages": []})
    assert state.stats()["glm53-flash"]["answerless_turns"] == 0


def test_the_last_arguments_chunk_holds_nothing_back():
    """A chunk that carries arguments AND the finish_reason ends the choice, so
    nothing may be held: held bytes would be flushed in a frame *before* that
    line and the client would reassemble the argument in the wrong order."""
    frame = {
        "id": "x", "object": "chat.completion.chunk", "created": 1, "model": "glm53-flash",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {
            "tool_calls": [{"index": 0, "function": {"arguments": "{\"expr\": \"a<"}}]}}],
    }
    app, up, _ = rig(glm_route())
    up.media = "text/event-stream"
    up.chunks = [b"data: " + json.dumps(frame).encode() + b"\n\n"]
    reply = post_chat(app, {"model": "glm53-flash", "messages": [], "stream": True})
    assert tool_arguments(sse_events(reply.content)) == '{"expr": "a<'


def test_redaction_is_case_insensitive():
    """Starlette lowercases what arrives over the wire, but a capture built
    from a plain dict (a test, a replay, a future caller) need not be."""
    out = glm_policies.redact_headers(
        {"Authorization": "Bearer sk-SECRET", "X-Api-Key": "k", "User-Agent": "vscode"}
    )
    assert out == {"Authorization": "<redacted>", "X-Api-Key": "<redacted>",
                   "User-Agent": "vscode"}


def test_blank_reasoning_is_not_remembered():
    """Storing whitespace would inflate the counter and put an entry in the
    bounded store that can never restore anything (``safe_reasoning`` empties
    it), evicting one that could."""
    store = glm_policies.ReasoningStore("m")
    assert store.remember(["t1"], "   ") == 0
    assert store.remember(["t1"], "") == 0
    assert store.remember([], "real thinking") == 0
    assert len(store) == 0


def test_an_sse_line_we_change_nothing_in_is_not_reformatted():
    """Byte-identical passthrough, inherited from ``policies.py``: a client
    that diffs two streams must see no difference where we made no change —
    including the whitespace upstream chose."""
    line = b'data: {"id": "x", "object": "chat.completion.chunk", "choices": []}\n\n'
    app, up, _ = rig(glm_route(mirror_reasoning=False))
    up.media = "text/event-stream"
    up.chunks = [line + b"data: [DONE]\n\n"]
    reply = post_chat(app, {"model": "glm53-flash", "messages": [], "stream": True})
    assert reply.content == line + b"data: [DONE]\n\n"


def test_stats_are_counted_per_route():
    glm = glm_route()
    other = Route(model_id="qwen38-27b", port=8001, live=True,
                  policies=RoutePolicies(ctx=262144, sanitize_tool_tags=True))
    table = FakeRouteTable([glm, other], main_name="glm53-flash")
    up = Upstream()
    client = httpx.AsyncClient(transport=httpx.MockTransport(up.handler))
    router = gateway.build_router(table, client=client)
    app = FastAPI()
    app.include_router(router)
    up.body = glm_fixture(ANSWERLESS_JSON)
    post_chat(app, {"model": "glm53-flash", "messages": []})
    post_chat(app, {"model": "qwen38-27b", "messages": []})
    post_chat(app, {"model": "qwen38-27b", "messages": []})
    assert router.glm_state.answerless_turns("glm53-flash") == 1
    assert router.glm_state.answerless_turns("qwen38-27b") == 2
    assert sorted(router.glm_state.stats()) == ["glm53-flash", "qwen38-27b"]


# ===========================================================================
# Item 5 — capture
# ===========================================================================


def capture_rig(tmp_path, *, consent: bool, monkeypatch, enabled: bool):
    monkeypatch.setenv(glm_policies.CAPTURE_ENV, "1" if enabled else "")
    route = glm_route(capture=consent)
    table = FakeRouteTable([route], main_name="glm53-flash")
    up = Upstream()
    client = httpx.AsyncClient(transport=httpx.MockTransport(up.handler))
    router = gateway.build_router(table, client=client)
    router.glm_state._capture_dir = tmp_path / "captures"
    app = FastAPI()
    app.include_router(router)
    return app, up, router.glm_state, tmp_path / "captures"


def test_a_routes_consent_alone_does_not_write_anything(tmp_path, monkeypatch):
    """The registry field is consent, not an enable.  A ``capture = true``
    somebody left in ``models.toml`` months ago must not start writing
    everything a user types to disk."""
    app, up, _, capdir = capture_rig(tmp_path, consent=True, monkeypatch=monkeypatch,
                                     enabled=False)
    up.body = glm_fixture(LEAKED_JSON)
    post_chat(app, {"model": "glm53-flash", "messages": [{"role": "user", "content": "secret"}]})
    assert not capdir.exists()


def test_the_master_switch_alone_does_not_write_anything(tmp_path, monkeypatch):
    app, up, _, capdir = capture_rig(tmp_path, consent=False, monkeypatch=monkeypatch,
                                     enabled=True)
    up.body = glm_fixture(LEAKED_JSON)
    post_chat(app, {"model": "glm53-flash", "messages": []})
    assert not capdir.exists()


def test_capture_writes_a_pair_and_redacts_credential_headers(tmp_path, monkeypatch):
    app, up, state, capdir = capture_rig(tmp_path, consent=True, monkeypatch=monkeypatch,
                                        enabled=True)
    up.body = glm_fixture(LEAKED_JSON)
    post_chat(app, {"model": "glm53-flash", "messages": [{"role": "user", "content": "hi"}]},
              headers={"authorization": "Bearer sk-REAL-SECRET", "x-api-key": "k",
                       "user-agent": "vscode"})
    files = sorted(p.name for p in (capdir / "glm53-flash").iterdir())
    assert files == ["req-0.json", "resp-0.json"]
    req = json.loads((capdir / "glm53-flash" / "req-0.json").read_text())
    assert req["headers"]["authorization"] == "<redacted>"
    assert req["headers"]["x-api-key"] == "<redacted>"
    assert req["headers"]["user-agent"] == "vscode"
    assert "sk-REAL-SECRET" not in (capdir / "glm53-flash" / "req-0.json").read_text()
    assert req["body"]["messages"][0]["content"] == "hi"
    assert state.stats()["glm53-flash"]["captures_written"] == 2


def test_the_capture_ring_wraps_and_that_is_the_retention_policy(tmp_path):
    """The original had none: it wrote to /tmp/glm-capture and, on a failure,
    froze a full copy of the request and reply OUTSIDE the ring, for ever."""
    ring = glm_policies.CaptureRing(tmp_path / "c", size=2)
    for i in range(5):
        turn = ring.next_turn()
        ring.write("req", turn, {"i": i})
    assert sorted(p.name for p in (tmp_path / "c").iterdir()) == ["req-0.json", "req-1.json"]
    assert json.loads((tmp_path / "c" / "req-0.json").read_text()) == {"i": 4}


def test_a_capture_that_cannot_be_written_never_breaks_the_request(tmp_path):
    ring = glm_policies.CaptureRing(tmp_path / "nope" / "c")
    (tmp_path / "nope").write_text("a file, not a directory")
    assert ring.write("req", 0, {"a": 1}) is None
    assert ring.errors == 1


def test_capture_files_are_not_world_readable(tmp_path):
    ring = glm_policies.CaptureRing(tmp_path / "c")
    path = ring.write("req", 0, {"a": 1})
    assert path is not None
    assert (path.stat().st_mode & 0o077) == 0
    assert ((tmp_path / "c").stat().st_mode & 0o077) == 0


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
    ("", False), ("0", False), ("false", False), ("no", False), ("maybe", False),
])
def test_the_master_switch_only_accepts_affirmatives(value, expected):
    assert glm_policies.capture_enabled({glm_policies.CAPTURE_ENV: value}) is expected


def test_capture_enabled_is_false_when_the_variable_is_absent():
    assert glm_policies.capture_enabled({}) is False


# ===========================================================================
# Wiring: the switches, the registry, and what gets no hook
# ===========================================================================


def test_models_toml_ships_the_three_switches_as_decided():
    """The defaults are the decision, so they are pinned in a test: the
    sanitiser on (recorded failure, pure repair), restore off (Xid-31
    correlation), capture off (it writes what the user typed to disk)."""
    reg = models.load(REPO_ROOT / "models.toml")
    glm = reg.models["glm53"]
    assert glm.sanitize_tool_tags is True
    assert glm.restore_reasoning is False
    assert glm.capture is False
    for key, model in reg.models.items():
        if key == "glm53":
            continue
        assert (model.sanitize_tool_tags, model.restore_reasoning, model.capture) == (
            False, False, False), f"{key} opted into a GLM-only policy"


def test_a_non_boolean_switch_is_a_registry_error(tmp_path):
    """``capture = "false"`` is a truthy string.  A switch whose whole job is
    to keep request bodies off the disk unless asked must not turn ON when its
    config says "false"."""
    path = tmp_path / "models.toml"
    path.write_text(
        (REPO_ROOT / "models.toml").read_text().replace("capture = false", 'capture = "false"')
    )
    with pytest.raises(models.RegistryError, match="capture must be true or false"):
        models.load(path)


def test_a_route_with_no_switches_gets_no_hook():
    """``begin()`` returning None is what keeps the gateway's no-buffering rule
    intact for every model that does not need these repairs."""
    pol = RoutePolicies(ctx=32768, mirror_reasoning=True)
    state = glm_policies.GlmState()
    assert glm_policies.begin(pol, model_id="LFM2.5-350M", state=state) is None
    assert state.stats() == {}


def test_the_responses_api_gets_no_hook():
    """Its envelope is ``output`` items, not ``choices[].message``; none of
    these repairs is written for it and the original did not wire them there
    either.  Silently applying a chat-shaped transform would be worse."""
    pol = RoutePolicies(ctx=327680, sanitize_tool_tags=True, restore_reasoning=True)
    state = glm_policies.GlmState()
    assert glm_policies.begin(pol, model_id="glm53-flash", state=state,
                              responses_api=True) is None


def test_the_responses_endpoint_still_gets_the_effort_overlay():
    """Proof the hook's absence there did not disturb what P2 ported: Codex
    speaks only this API, so dropping the overlay would silently downgrade
    ``glm53-flash-high`` to unbounded effort."""
    route = Route(
        model_id="glm53-flash", port=8002, live=True, presets=("glm53-flash-high",),
        policies=RoutePolicies(ctx=327680, sanitize_tool_tags=True,
                               effort_overlay={"reasoning_effort": "high"}),
    )
    table = FakeRouteTable([route], main_name="glm53-flash")
    up = Upstream()
    client = httpx.AsyncClient(transport=httpx.MockTransport(up.handler))
    app = FastAPI()
    app.include_router(gateway.build_router(table, client=client))
    call(app, "POST", "/v1/responses", json={"model": "glm53-flash-high", "input": "hi"})
    assert up.last_body["chat_template_kwargs"] == {"reasoning_effort": "high"}


def test_the_state_belongs_to_the_router_not_the_module():
    """Two gateways in one process — the :8011 rehearsal of cutover step 1
    alongside the live :8010 — must share nothing."""
    a = rig(glm_route(restore_reasoning=True))
    b = rig(glm_route(restore_reasoning=True))
    a[1].body = glm_fixture(LEAKED_JSON)
    post_chat(a[0], {"model": "glm53-flash", "messages": []})
    assert a[2].stats()["glm53-flash"]["reasoning_remembered"] == 1
    assert b[2].stats() == {}


def test_the_hook_composes_with_the_reasoning_mirror():
    """P2's mirror runs first, line by line; this packet's transform runs on
    its output.  Both must survive the composition."""
    app, up, _ = rig(glm_route())
    up.media = "text/event-stream"
    up.chunks = [glm_fixture(LEAKED_SSE)]
    reply = post_chat(app, {"model": "glm53-flash", "messages": [], "stream": True})
    events = sse_events(reply.content)
    mirrored = [
        (ch.get("delta") or {}).get("reasoning_content")
        for ev in events for ch in ev.get("choices") or []
    ]
    assert "They want a directory listing" in mirrored
    assert json.loads(tool_arguments(events)) == {"path": "/Users/user/Desktop/project"}


def test_the_output_floor_still_applies_alongside_the_hook():
    """``max_tokens: 200`` in the Copilot fixture is exactly the shape that
    truncates mid-``<think>``; P2's floor must still raise it."""
    app, up, _ = rig(glm_route())
    post_chat(app, glm_fixture(ECHO_REQUEST))
    assert up.last_body["max_tokens"] == 8192


# ===========================================================================
# What only a live GLM-5.3 can settle
# ===========================================================================
#
# * That GLM's parser leaks these tags in the arrival *shape* fixture 5
#   assumes.  The leaked text is real (captured); that it arrives split across
#   two `arguments` deltas rather than inside one is inference from *why* it
#   leaked (a tag straddling a token boundary).  Both shapes are covered here,
#   so the sanitiser is right either way, but only a live run says which
#   happens.
# * That restoring reasoning by tool_call id still collapses the amnesia on
#   GLM's current template, and whether the Xid-31 correlation reappears. That
#   is the measurement the default-off switch exists to make possible.
# * Whether the answerless-turn count stays at zero once the ported output
#   floor is doing its job. If it does not, the floor is wrong, and that is
#   precisely what the count is for.
