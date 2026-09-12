"""Tests for ``servedeck.policies`` — the pure transforms the gateway applies.

Every reasoning-mirror assertion here runs against fixtures **recorded from a
live vLLM**: ``tests/fixtures/reasoning/`` was captured on 2026-09-10 from the
build serving Flash-Next on :8001 (``vllm-0.29.0.dev0+qwen38next-79646177``,
model ``qwen38-flash-next``), and carried over from
``flashnext-reasoning-proxy/tests/`` together with the tests that read them.
None of them are synthetic; the constructed payloads (the shredded chunk
boundaries, the oversized body) are inputs, not expected outputs.

The premise test is the first one below, and it is the reason the fixtures were
kept rather than replaced with hand-written JSON: if a future vLLM starts
emitting both field names, it fails, and the mirror can be retired instead of
left in place normalising something that no longer needs it.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from servedeck import policies
from servedeck.policies import (
    ModelScan,
    apply_effort_overlay,
    apply_model_rewrite,
    apply_output_floor,
    apply_request_policies,
    mirror_reasoning,
    scan_model,
    sse_stream,
    transform_json_body,
    transform_sse_line,
)
from tests.fake_routes import load, reassemble, sse_events

NONSTREAM = "chat_nonstream_reasoning.json"
NOREASON = "chat_nonstream_noreasoning.json"
TOOLCALL = "chat_nonstream_toolcall.json"
STREAM = "chat_stream_reasoning.sse"
STREAM_NOREASON = "chat_stream_noreasoning.sse"
STREAM_TOOLCALL = "chat_stream_toolcall.sse"

RECORDED = (NONSTREAM, NOREASON, TOOLCALL, STREAM, STREAM_NOREASON, STREAM_TOOLCALL)


def run_sse(raw: bytes, chunks: list[bytes] | None = None) -> bytes:
    """Drive ``sse_stream`` over a byte stream cut at the given boundaries."""
    pieces = chunks if chunks is not None else [raw]

    async def source():
        for piece in pieces:
            yield piece

    async def drive():
        return b"".join([c async for c in sse_stream(source())])

    return asyncio.run(drive())


# ---------------------------------------------------------------------------
# 1. the premise
# ---------------------------------------------------------------------------


def test_recorded_upstream_really_lacks_reasoning_content():
    """If a future server starts emitting both names, this fails and the mirror
    policy can be switched off in the registry rather than left mirroring a
    field that is already there."""
    msg = json.loads(load(NONSTREAM))["choices"][0]["message"]
    assert msg["reasoning"], "fixture has no reasoning to mirror"
    assert "reasoning_content" not in msg
    for name in (STREAM, STREAM_TOOLCALL):
        assert b'"reasoning_content"' not in load(name), f"{name} already carries both names"


@pytest.mark.parametrize("name", RECORDED)
def test_dumps_round_trips_recorded_upstream_byte_for_byte(name):
    """``_dumps`` claims to serialise exactly the way vLLM's JSONResponse does.
    That claim is what lets a payload needing no mirroring survive a
    parse/serialise round trip unchanged, so it is checked, not asserted in a
    comment — on every recorded JSON body and every recorded SSE frame."""
    if name.endswith(".json"):
        raw = load(name)
        assert policies._dumps(json.loads(raw)) == raw
        return
    frames = 0
    for line in load(name).split(b"\n"):
        if not line.startswith(b"data: ") or line[6:] in (b"[DONE]", b""):
            continue
        assert policies._dumps(json.loads(line[6:])) == line[6:]
        frames += 1
    assert frames > 3, "fixture carries no SSE frames"


# ---------------------------------------------------------------------------
# 2. the mirror itself
# ---------------------------------------------------------------------------


def test_mirror_leaves_non_string_and_empty_values_alone():
    assert mirror_reasoning({"reasoning": None}) == {"reasoning": None}
    assert mirror_reasoning({"reasoning": ""}) == {"reasoning": ""}
    assert mirror_reasoning({"reasoning": {"effort": "high"}}) == {"reasoning": {"effort": "high"}}
    assert mirror_reasoning({"type": "reasoning", "summary": []}) == {
        "type": "reasoning",
        "summary": [],
    }


def test_mirror_fills_either_direction_and_never_overwrites():
    assert mirror_reasoning({"reasoning_content": "abc"}) == {
        "reasoning_content": "abc",
        "reasoning": "abc",
    }
    both = {"reasoning": "a", "reasoning_content": "b"}
    assert mirror_reasoning(dict(both)) == both


def test_mirror_reaches_nested_shapes():
    """It keys on the field name at any depth, so it covers the message, the
    delta and the /v1/responses item shapes without knowing any of them."""
    obj = {"output": [{"type": "reasoning", "parts": [{"reasoning": "deep"}]}]}
    mirror_reasoning(obj)
    assert obj["output"][0]["parts"][0]["reasoning_content"] == "deep"


# ---------------------------------------------------------------------------
# 3. JSON bodies
# ---------------------------------------------------------------------------


def test_nonstream_emits_both_fields_with_equal_values():
    got = json.loads(transform_json_body(load(NONSTREAM)))
    msg = got["choices"][0]["message"]
    original = json.loads(load(NONSTREAM))["choices"][0]["message"]
    assert msg["reasoning"] == msg["reasoning_content"] == original["reasoning"]


def test_nonstream_never_rewrites_content_or_other_fields():
    got = json.loads(transform_json_body(load(NONSTREAM)))
    original = json.loads(load(NONSTREAM))
    got["choices"][0]["message"].pop("reasoning_content")
    assert got == original


def test_nonstream_usage_and_finish_reason_survive():
    got = json.loads(transform_json_body(load(NONSTREAM)))
    original = json.loads(load(NONSTREAM))
    assert got["usage"] == original["usage"]
    assert got["usage"]["completion_tokens_details"]["reasoning_tokens"] == 40
    assert got["choices"][0]["finish_reason"] == original["choices"][0]["finish_reason"]


def test_response_with_no_reasoning_is_byte_identical():
    """``"reasoning": null`` must stay null: no empty twin invented.

    This body *does* take the slow path — the pre-filter matches the literal
    ``"reasoning"`` key even though its value is null — so what it proves is
    the other half of the contract: a parse/serialise round trip that changed
    nothing produces the same bytes, down to key order and separators.
    """
    raw = load(NOREASON)
    assert transform_json_body(raw) == raw
    out = json.loads(transform_json_body(raw))
    assert "reasoning_content" not in out["choices"][0]["message"]
    assert out["choices"][0]["message"]["reasoning"] is None


def test_tool_calls_and_null_content_survive_with_reasoning_mirrored():
    got = json.loads(transform_json_body(load(TOOLCALL)))
    original = json.loads(load(TOOLCALL))
    msg, omsg = got["choices"][0]["message"], original["choices"][0]["message"]
    assert msg["tool_calls"] == omsg["tool_calls"]
    assert msg["content"] is None
    assert got["choices"][0]["finish_reason"] == "tool_calls"
    assert msg["reasoning_content"] == msg["reasoning"] == omsg["reasoning"]


def test_reasoning_tokens_key_does_not_trigger_a_reparse():
    """The byte pre-filter must not match ``"reasoning_tokens"``; if it did,
    every usage frame would take the slow path."""
    raw = b'{"usage":{"completion_tokens_details":{"reasoning_tokens":40}}}'
    assert transform_json_body(raw) is raw


def test_malformed_json_is_relayed_verbatim():
    raw = b'{"reasoning": "x", trailing garbage'
    assert transform_json_body(raw) is raw


# ---------------------------------------------------------------------------
# 4. SSE
# ---------------------------------------------------------------------------


def test_stream_mirrors_every_reasoning_delta():
    seen = 0
    for ev in sse_events(run_sse(load(STREAM))):
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if "reasoning" in d or "reasoning_content" in d:
                assert d["reasoning"] == d["reasoning_content"]
                seen += 1
    assert seen >= 5, f"only {seen} reasoning deltas seen"


def test_stream_reassembles_to_the_same_text_and_reasoning():
    before, after = sse_events(load(STREAM)), sse_events(run_sse(load(STREAM)))
    assert reassemble(after, "content") == reassemble(before, "content")
    assert reassemble(after, "reasoning") == reassemble(before, "reasoning")
    assert reassemble(after, "reasoning_content") == reassemble(before, "reasoning")


def test_stream_usage_and_finish_reason_survive():
    out = run_sse(load(STREAM))
    events = sse_events(out)
    finals = [e for e in events if e.get("usage")]
    assert finals, "no usage frame"
    assert finals[-1]["usage"]["completion_tokens_details"]["reasoning_tokens"] == 43
    assert any(ch.get("finish_reason") == "stop" for e in events for ch in e.get("choices") or [])
    assert b"data: [DONE]" in out


def test_stream_without_reasoning_is_byte_identical():
    raw = load(STREAM_NOREASON)
    assert run_sse(raw) == raw


def test_stream_tool_calls_survive():
    def tcs(evs):
        return [
            tc
            for e in evs
            for ch in e.get("choices") or []
            for tc in (ch.get("delta") or {}).get("tool_calls") or []
        ]

    after = tcs(sse_events(run_sse(load(STREAM_TOOLCALL))))
    assert after == tcs(sse_events(load(STREAM_TOOLCALL)))
    assert after, "fixture carries no tool_call deltas"


@pytest.mark.parametrize("size", [1, 3, 17, 250])
def test_stream_survives_arbitrary_chunk_boundaries(size):
    """Upstream chunk boundaries fall anywhere, including mid-JSON and
    mid-UTF-8; the output must not depend on where."""
    raw = load(STREAM)
    shredded = [raw[i : i + size] for i in range(0, len(raw), size)]
    assert run_sse(raw, chunks=shredded) == run_sse(raw)


def test_stream_holds_back_only_the_partial_trailing_line():
    """The hold-back is the whole correctness argument for the line transform.
    Feed a stream cut inside the JSON of its third frame: the two complete
    frames before the cut must come out on the first yield, and nothing else."""

    raw = load(STREAM)
    cut = raw.index(b"\n\n", raw.index(b"\n\n") + 2) + 2 + 10
    chunks = [raw[:cut], raw[cut:]]

    async def source():
        for c in chunks:
            yield c

    async def drive():
        return [c async for c in sse_stream(source())]

    out = asyncio.run(drive())
    assert len(out) == 2, "the transform did not flush on the chunk that completed a line"
    # The two complete frames before the cut came out at once, already
    # mirrored; the partial third was held back, not emitted broken.
    assert out[0].count(b"data:") == 2
    assert out[0].endswith(b"\n")
    assert b"reasoning_content" in out[0]
    assert b"".join(out) == run_sse(raw)


def test_sse_line_passthrough_shapes():
    """Comments, event: lines, blank separators and [DONE] are returned as the
    same bytes, and both line endings survive."""
    for line in (b": ping\n", b"event: message\n", b"\n", b"data: [DONE]\n\n", b"data:\n"):
        assert transform_sse_line(line) is line
    crlf = b'data: {"reasoning":"a"}\r\n'
    assert transform_sse_line(crlf).endswith(b"\r\n")
    assert b'"reasoning_content":"a"' in transform_sse_line(crlf)
    # The space after "data:" is preserved exactly as upstream wrote it.
    assert transform_sse_line(b'data:{"reasoning":"a"}\n').startswith(b'data:{"')
    assert transform_sse_line(b'data: {"reasoning":"a"}\n').startswith(b'data: {"')


# ---------------------------------------------------------------------------
# 5. the effort overlay (ported from glm53-effort-proxy)
# ---------------------------------------------------------------------------


def test_overlay_is_byte_identical_when_there_is_nothing_to_apply():
    raw = json.dumps({"model": "m", "messages": []}).encode()
    assert apply_effort_overlay(raw, None) is raw
    assert apply_effort_overlay(raw, {}) is raw


def test_overlay_injects_reasoning_effort():
    raw = json.dumps({"model": "glm53-flash-high", "messages": []}).encode()
    got = json.loads(apply_effort_overlay(raw, {"reasoning_effort": "high"}))
    assert got["chat_template_kwargs"] == {"reasoning_effort": "high"}


def test_overlay_only_fills_a_gap_an_explicit_caller_value_wins():
    raw = json.dumps(
        {"model": "x", "chat_template_kwargs": {"reasoning_effort": "low", "other": 1}}
    ).encode()
    out = apply_effort_overlay(raw, {"reasoning_effort": "high"})
    assert out is raw, "setdefault semantics must leave the body untouched"
    assert json.loads(out)["chat_template_kwargs"]["reasoning_effort"] == "low"


def test_overlay_merges_alongside_existing_kwargs():
    raw = json.dumps({"model": "x", "chat_template_kwargs": {"other": 1}}).encode()
    got = json.loads(apply_effort_overlay(raw, {"reasoning_effort": "high"}))
    assert got["chat_template_kwargs"] == {"other": 1, "reasoning_effort": "high"}


def test_overlay_survives_a_null_chat_template_kwargs():
    raw = json.dumps({"model": "x", "chat_template_kwargs": None}).encode()
    got = json.loads(apply_effort_overlay(raw, {"reasoning_effort": "low"}))
    assert got["chat_template_kwargs"] == {"reasoning_effort": "low"}


# ---------------------------------------------------------------------------
# 6. the output floor (ported from glm53-effort-proxy / flashnext)
# ---------------------------------------------------------------------------


def test_floor_off_is_byte_identical():
    raw = json.dumps({"max_tokens": 64, "messages": []}).encode()
    assert apply_output_floor(raw, 0, 262144) is raw
    assert apply_output_floor(raw, None, 262144) is raw


def test_floor_raises_an_explicitly_small_budget():
    raw = json.dumps({"max_tokens": 64, "messages": []}).encode()
    assert json.loads(apply_output_floor(raw, 8192, 262144))["max_tokens"] == 8192


def test_floor_never_lowers_a_large_budget():
    raw = json.dumps({"max_tokens": 30000, "messages": []}).encode()
    assert apply_output_floor(raw, 8192, 262144) is raw


def test_floor_never_invents_an_absent_budget():
    """A caller who sent no budget already has the whole remaining context,
    which is strictly more room than any floor.  Setting one here is how an
    earlier version of this CAPPED unbounded callers at 8k."""
    raw = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    assert apply_output_floor(raw, 8192, 262144) is raw
    assert "max_tokens" not in json.loads(apply_output_floor(raw, 8192, 262144))


def test_floor_applies_to_every_budget_spelling():
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        raw = json.dumps({key: 16, "messages": []}).encode()
        assert json.loads(apply_output_floor(raw, 4096, 262144))[key] == 4096


def test_floor_is_clamped_so_prompt_plus_budget_still_fits():
    """ctx 1000, a ~1000-token prompt and a 2048-token reserve leave no room at
    all, so the small budget the caller sent is left exactly as it was."""
    prompt = "x" * 4000  # ~1000 tokens at 4 chars/token
    raw = json.dumps({"max_tokens": 64, "messages": [{"role": "user", "content": prompt}]}).encode()
    assert apply_output_floor(raw, 8192, 1000) is raw
    # With room for some of the floor but not all of it, the clamp binds.
    got = json.loads(apply_output_floor(raw, 8192, 6000))
    assert got["max_tokens"] == 6000 - 1000 - 2048


def test_floor_counts_multimodal_text_parts_not_image_bytes():
    """An image's base64 is not text; counting it as prompt tokens would make
    the clamp refuse to raise anything on a multimodal turn."""
    body = {
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 40000}},
                ],
            }
        ],
    }
    got = json.loads(apply_output_floor(json.dumps(body).encode(), 8192, 32768))
    assert got["max_tokens"] == 8192


def test_floor_ignores_a_boolean_budget():
    """``True`` is an ``int`` in Python; treating it as a budget of 1 and
    raising it to 8192 would change a (nonsensical but harmless) field into a
    real token budget."""
    raw = json.dumps({"max_tokens": True, "messages": []}).encode()
    assert apply_output_floor(raw, 8192, 262144) is raw


def test_floor_leaves_a_null_budget_alone():
    raw = json.dumps({"max_tokens": None, "messages": []}).encode()
    assert apply_output_floor(raw, 8192, 262144) is raw


def test_floor_counts_the_responses_api_input_field():
    raw = json.dumps({"max_output_tokens": 64, "input": "y" * 8000}).encode()
    got = json.loads(apply_output_floor(raw, 8192, 6000))
    assert got["max_output_tokens"] == 6000 - 2000 - 2048


# ---------------------------------------------------------------------------
# 7. model rewrite and composition
# ---------------------------------------------------------------------------


def test_model_rewrite_is_byte_identical_when_the_name_already_matches():
    raw = json.dumps({"model": "LFM2.5-350M", "messages": []}).encode()
    assert apply_model_rewrite(raw, "LFM2.5-350M") is raw
    assert apply_model_rewrite(raw, None) is raw


def test_model_rewrite_replaces_an_alias():
    raw = json.dumps({"model": "lfm2", "messages": []}).encode()
    assert json.loads(apply_model_rewrite(raw, "LFM2.5-350M"))["model"] == "LFM2.5-350M"


def test_model_rewrite_never_invents_a_model_field():
    raw = json.dumps({"messages": []}).encode()
    assert apply_model_rewrite(raw, "LFM2.5-350M") is raw


def test_all_policies_compose_in_one_round_trip():
    raw = json.dumps({"model": "glm53-flash-high", "max_tokens": 64, "messages": []}).encode()
    got = json.loads(
        apply_request_policies(
            raw,
            model_id="glm53-flash",
            overlay={"reasoning_effort": "high"},
            floor=8192,
            ctx=327680,
        )
    )
    assert got["model"] == "glm53-flash"
    assert got["chat_template_kwargs"] == {"reasoning_effort": "high"}
    assert got["max_tokens"] == 8192


def test_request_policies_relay_non_json_and_non_objects_verbatim():
    garbage = b'{"model": "x", "messages": [], trailing garbage'
    assert apply_request_policies(garbage, model_id="y", floor=8192, ctx=100) is garbage
    array = b'[1,2,3]'
    assert apply_request_policies(array, model_id="y") is array


def test_prior_turn_reasoning_content_survives_a_policy_round_trip():
    """The multi-turn request fixture carries the reasoning a client echoed
    back.  Whatever a policy rewrites, that field must reach the model: losing
    it is the whole amnesia mechanism the mirror exists to prevent."""
    raw = load("chat_nonstream_multiturn_request.json")
    out = apply_request_policies(raw, model_id="rewritten", floor=8192, ctx=262144)
    assert out is not raw
    assert json.loads(out)["messages"][1]["reasoning_content"] == "SECRET_PRIOR_THOUGHT: 17*23=391"


# ---------------------------------------------------------------------------
# 8. scan_model — routing without buffering the request
# ---------------------------------------------------------------------------


def test_scan_model_finds_a_leading_model():
    assert scan_model(b'{"model": "lfm2", "messages": []}') == ModelScan("lfm2", True)


def test_scan_model_finds_a_trailing_model():
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "model": "lfm2"}).encode()
    assert scan_model(body) == ModelScan("lfm2", True)


def test_scan_model_reports_incomplete_rather_than_guessing():
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "model": "lfm2"}).encode()
    for cut in range(1, len(body) - len(b'"lfm2"}')):
        assert scan_model(body[:cut]).model is None
    # ... and the truncations that stop before the value are not "complete"
    assert scan_model(body[:20]) == ModelScan(None, False)


def test_scan_model_ignores_a_nested_model_key():
    """A tool schema with a ``model`` property must not decide the route.

    The shape matters: ``model`` is the *second* property, after one whose own
    value is an object.  A scanner that tracked key/value position but not
    **depth** survives the single-property version of this body (the nested key
    lands where a value was expected and is discarded), and then returns
    ``"type"`` — the first word of the schema — for this one.
    """
    body = json.dumps(
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "model": {"type": "string"},
                            },
                        },
                    },
                }
            ],
            "model": "real",
        }
    ).encode()
    assert scan_model(body) == ModelScan("real", True)


def test_scan_model_ignores_the_word_model_inside_a_string():
    body = json.dumps({"messages": [{"role": "user", "content": 'the "model": "fake" is'}], "model": "real"}).encode()
    assert scan_model(body) == ModelScan("real", True)


def test_scan_model_handles_escapes_and_braces_inside_strings():
    body = rb'{"messages":[{"content":"a \" } quote and a \\\\ slash"}],"model":"real"}'
    assert scan_model(body) == ModelScan("real", True)


def test_scan_model_completes_with_none_when_there_is_no_model():
    assert scan_model(b'{"messages": []}') == ModelScan(None, True)


def test_scan_model_rejects_non_objects():
    assert scan_model(b"[1,2,3]") == ModelScan(None, True)
    assert scan_model(b"not json at all") == ModelScan(None, True)
    assert scan_model(b"") == ModelScan(None, False)


def test_scan_model_rejects_a_non_string_model():
    assert scan_model(b'{"model": 7, "messages": []}') == ModelScan(None, True)
    assert scan_model(b'{"model": null}') == ModelScan(None, True)


def test_scan_model_survives_utf8_split_mid_character():
    body = '{"messages":[{"content":"héllo — wörld"}],"model":"real"}'.encode()
    for cut in range(len(body)):
        # No truncation may ever produce a *wrong* answer; only None or "real".
        assert scan_model(body[:cut]).model in (None, "real")
    assert scan_model(body) == ModelScan("real", True)
