"""Per-route request/response policies for the gateway — pure, no I/O.

Everything here is a byte-level transform. No sockets, no files, no clock, no
globals that a request can change: a policy takes bytes (or a parsed object)
and returns bytes. That is what makes the whole normalising layer testable
against *recorded* upstream traffic instead of a live model, and it is why the
gateway can decide per route which policies are on without any of them
reaching for configuration of their own.

What lives here, and where it came from
---------------------------------------
* ``mirror_reasoning`` / ``_dumps`` / ``transform_json_body`` /
  ``transform_sse_line`` / ``sse_stream`` and the two header-hygiene lists are
  ported from ``flashnext-reasoning-proxy/proxy.py`` (2026-09-10), together
  with its recorded fixtures.  vLLM ≥ 0.29 renamed the chat-completions
  reasoning field to ``reasoning`` and offers no flag to emit the old name; a
  client that reads ``reasoning_content`` therefore sees no thinking, cannot
  store it, and cannot send it back — which is the measured mechanism behind
  multi-turn agent amnesia (restoring the round trip collapsed it ~33,000x).
* The effort overlay and the output floor inside ``apply_request_policies``
  are ported from ``glm53-effort-proxy/proxy.py``.  The overlay is how a preset
  (``glm53-flash-high``) becomes a real request: neither VS Code's BYOK
  provider nor Codex lets a user set ``chat_template_kwargs``, but both let a
  user pick a model, so the preset name carries the parameter.  The floor
  exists because reasoning shares ``max_tokens`` with the answer: a small
  budget is spent thinking and the turn returns ``finish_reason="length"``
  with ``content: null`` — an empty assistant turn, which then poisons the
  next request's history.
* ``ModelScan`` / ``scan_model`` are new.  The gateway routes by the ``model``
  field of the request body, and the only way to do that *without* buffering
  the body (``app.py``'s ``catch_all`` buffers every request, including 1 MB
  base64 images) is to read just far enough to resolve that one field.

The rule every function here obeys
----------------------------------
**Byte-identical passthrough when there is nothing to do.**  A policy that
finds no work returns the *same object* it was given, not an equal one — no
parse, no re-serialise, no reordered keys, no changed float formatting.  The
tests assert identity, not equality, wherever that is the contract.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Header hygiene (ported verbatim from flashnext-reasoning-proxy/proxy.py)
# ---------------------------------------------------------------------------

#: Request headers that must never be relayed upstream.
#:
#: ``host`` — httpx sets it from the target URL, and relaying the client's
#: would send the gateway's own ``127.0.0.1:8010`` to a model on :8007.
#: ``content-length`` — the body length changes when a policy rewrites it, and
#: when it does not we forward the body as a stream, so httpx sets whichever of
#: content-length / transfer-encoding is right.
#: ``accept-encoding`` — dropped and then *forced to identity* by the gateway,
#: so what we read from upstream is what upstream wrote.  Without this a
#: payload that needs no mirroring could arrive gzipped and could not be
#: relayed as the same bytes.
#:
#: **Credentials are stripped, not relayed.**  ``authorization``, ``api-key``,
#: ``x-api-key``, ``openai-organization`` and ``cookie`` carry a client's own
#: secrets — a real OpenAI key a user left in a VS Code profile, a session
#: cookie from a browser extension — and the models here run without
#: ``--api-key``, so a model process has no use for any of them.  Forwarding
#: them would copy a user's credentials into vLLM's memory and, on a bad
#: request, into its logs.  If a route ever needs to authenticate to its
#: upstream, the gateway must *inject* that route's own credential here rather
#: than pass the client's through.
#:
#: The rest are RFC 7230 §6.1 hop-by-hop headers plus ``proxy-connection``.
DROP_REQUEST_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "upgrade",
        "te",
        "trailer",
        "proxy-connection",
        "accept-encoding",
        "authorization",
        "api-key",
        "x-api-key",
        "openai-organization",
        "cookie",
    }
)

#: Response headers that must never be relayed downstream.  ``content-length``
#: and ``content-encoding`` because the body we hand back may be a different
#: length than upstream's (mirroring adds a field) and is always decoded; the
#: rest are hop-by-hop.
DROP_RESPONSE_HEADERS = frozenset(
    {
        "content-length",
        "content-encoding",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "trailer",
    }
)

# ---------------------------------------------------------------------------
# Reasoning mirror
# ---------------------------------------------------------------------------

#: Cheap pre-filter.  A payload without this byte sequence cannot need
#: mirroring, so it is relayed verbatim — no parse, no re-serialise.  It does
#: NOT match ``"reasoning_tokens"`` or ``"reasoning_content"`` (both continue
#: past the closing quote), which is why those are tested for separately.
_REASONING_MARK = b'"reasoning"'
_REASONING_CONTENT_MARK = b'"reasoning_content"'


def mirror_reasoning(obj: Any) -> Any:
    """Give every ``reasoning`` a twin ``reasoning_content`` and vice versa.

    Mutates in place, recursively, so it covers the chat-completions message,
    the streaming delta, and the ``/v1/responses`` item shapes without knowing
    any of them.  It keys on the **field name**, so a ``{"type": "reasoning"}``
    marker or a ``{"reasoning": {"effort": ...}}`` request object is untouched.

    A missing, null or empty value is left exactly as it is: inventing an empty
    ``reasoning_content`` would be a change to a response that had no
    reasoning.  LFM2.5-350M, for instance, answers with ``"reasoning": null``
    on every turn, and that must survive as ``null``.
    """
    if isinstance(obj, dict):
        r = obj.get("reasoning")
        rc = obj.get("reasoning_content")
        if isinstance(r, str) and r and rc is None:
            obj["reasoning_content"] = r
        elif isinstance(rc, str) and rc and r is None:
            obj["reasoning"] = rc
        for v in obj.values():
            if isinstance(v, (dict, list)):
                mirror_reasoning(v)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, (dict, list)):
                mirror_reasoning(v)
    return obj


def _dumps(data: Any) -> bytes:
    """Serialise the way vLLM's own JSONResponse does.

    Verified byte-identical on recorded upstream replies (see
    ``tests/fixtures/``), so a payload that needed no mirroring comes out of a
    parse/serialise round trip unchanged.  Every one of these arguments is
    load-bearing: ``ensure_ascii=False`` keeps non-ASCII as UTF-8 rather than
    ``\\uXXXX``, and ``separators=(",", ":")`` removes the spaces json.dumps
    would otherwise put after every comma and colon.
    """
    return json.dumps(
        data, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":")
    ).encode("utf-8")


def transform_json_body(raw: bytes) -> bytes:
    """Mirror reasoning in a complete JSON body, or relay it verbatim.

    Returns ``raw`` itself — the same object — when there is nothing to do.
    """
    if _REASONING_MARK not in raw and _REASONING_CONTENT_MARK not in raw:
        return raw
    try:
        data = json.loads(raw)
    except Exception:
        return raw
    return _dumps(mirror_reasoning(data))


def transform_sse_line(line: bytes) -> bytes:
    """Mirror reasoning inside one SSE line, preserving its line ending.

    Anything that is not a ``data:`` payload — comments, ``event:``, blank
    separator lines, ``[DONE]`` — is returned as the same bytes.  The space
    after ``data:`` is preserved exactly as upstream wrote it (present or
    absent): it is not part of the payload per the SSE spec, but a client that
    diffs streams should see no difference where we made no change.
    """
    if b"data:" not in line:
        return line
    body = line
    ending = b""
    if body.endswith(b"\r\n"):
        body, ending = body[:-2], b"\r\n"
    elif body.endswith(b"\n"):
        body, ending = body[:-1], b"\n"
    if not body.startswith(b"data:"):
        return line
    payload = body[5:]
    lead = b""
    if payload.startswith(b" "):
        lead, payload = b" ", payload[1:]
    if payload == b"[DONE]" or not payload:
        return line
    if _REASONING_MARK not in payload and _REASONING_CONTENT_MARK not in payload:
        return line
    try:
        data = json.loads(payload)
    except Exception:
        return line
    return b"data:" + lead + _dumps(mirror_reasoning(data)) + ending


async def sse_stream(source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Transform an SSE byte stream line by line, without buffering it.

    Only an incomplete trailing line is held back — upstream chunk boundaries
    fall anywhere, including mid-JSON and mid-UTF-8 — so at most one partial
    SSE event is ever in hand.  Every complete line is flushed on the same
    iteration the chunk that completed it arrived, so time-to-first-token is
    one hop, not one response.

    The hold-back is what makes the transform correct at arbitrary chunk
    boundaries: without it a ``data:`` line split across two reads would be
    parsed twice as two broken halves.  ``tests/test_gateway.py`` shreds a
    recorded stream into 1-, 3-, 17- and 250-byte chunks and requires the
    output to be identical to the unshredded run.
    """
    buf = b""
    async for chunk in source:
        if not chunk:
            continue
        buf += chunk
        if b"\n" not in buf:
            continue
        head, _, buf = buf.rpartition(b"\n")
        out = bytearray()
        for line in head.split(b"\n"):
            out += transform_sse_line(line + b"\n")
        yield bytes(out)
    if buf:
        yield transform_sse_line(buf)


# ---------------------------------------------------------------------------
# Request-body policies: model rewrite, effort overlay, output floor
# ---------------------------------------------------------------------------

#: The three spellings of "how many tokens may the model emit" across the
#: chat-completions (``max_tokens``, and its OpenAI successor
#: ``max_completion_tokens``) and responses (``max_output_tokens``) APIs.
_OUTPUT_BUDGET_KEYS = ("max_tokens", "max_completion_tokens", "max_output_tokens")

#: Slack left between the approximate prompt and the context ceiling when
#: clamping a raised budget, so prompt + budget still fits after the estimate
#: (4 chars/token) under-counts.
_CTX_RESERVE = 2048


def _approx_prompt_tokens(body: Mapping[str, Any]) -> int:
    """Rough prompt size, at the industry-standard 4 characters per token.

    Deliberately approximate: it is only used to clamp a raised budget so the
    result still fits in the context window, and being a little pessimistic
    there costs nothing.  Counts chat ``messages`` (string content and the
    ``text`` parts of multimodal content — an image's base64 is not text and
    must not be counted as if it were) and the responses API's ``input``.
    """
    total = 0
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str):
            total += len(c) // 4
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += len(part["text"]) // 4
    inp = body.get("input")
    if isinstance(inp, str):
        total += len(inp) // 4
    return total


def _apply_floor(body: dict, floor: int | None, ctx: int) -> bool:
    """Raise an explicitly-small output budget to ``floor``.  Returns changed.

    Three rules, each of which was a bug at some point:

    1. **Only ever raises.**  A caller who asked for more than the floor keeps
       what they asked for.
    2. **Never touches an absent budget.**  A caller who sent no budget at all
       already has the whole remaining context, which is strictly more room
       than any floor — an earlier version of this in ``glm53-effort-proxy``
       set the floor unconditionally and therefore *capped* unbounded callers
       at 8k, truncating long planning turns mid-thought.
    3. **Clamped to the context.**  ``ctx - prompt - 2048``, so raising the
       budget can never make the request itself unservable.
    """
    if not isinstance(floor, int) or isinstance(floor, bool) or floor <= 0:
        return False
    if ctx <= 0:
        # A floor with no context to clamp it against silently computes
        # ``min(floor, 0 - prompt - 2048)`` <= 0 and therefore never raises
        # anything — the GLM thinking-budget fix would be configured, reported
        # as on, and do nothing.  A registry that sets a floor must set a ctx.
        raise ValueError(
            f"min_output_tokens={floor} requires a positive ctx, got {ctx}: "
            "an unclamped floor is a silent no-op"
        )
    approx: int | None = None
    changed = False
    for key in _OUTPUT_BUDGET_KEYS:
        if key not in body:
            continue
        current = body[key]
        if not isinstance(current, int) or isinstance(current, bool):
            continue
        if current >= floor:
            continue
        if approx is None:
            approx = _approx_prompt_tokens(body)
        target = max(0, min(floor, ctx - approx - _CTX_RESERVE))
        if target > current:
            body[key] = target
            changed = True
    return changed


def _apply_overlay(body: dict, overlay: Mapping[str, Any] | None) -> bool:
    """Merge a preset's ``chat_template_kwargs`` overlay in.  Returns changed.

    ``setdefault`` semantics: the overlay only ever **fills a gap**.  An
    explicit caller-supplied value always wins, because a preset is a default
    a client could not otherwise express, not an override of one it could.
    """
    if not overlay:
        return False
    existing = body.get("chat_template_kwargs")
    kw = dict(existing) if isinstance(existing, dict) else {}
    changed = False
    for k, v in overlay.items():
        if k not in kw:
            kw[k] = v
            changed = True
    if changed:
        body["chat_template_kwargs"] = kw
    return changed


def _apply_model(body: dict, model_id: str | None) -> bool:
    """Rewrite ``model`` to the upstream's served name.  Returns changed.

    Only rewrites a ``model`` the body actually carried: a request that named
    no model is forwarded as it stands rather than having one invented for it.
    """
    if model_id is None or "model" not in body:
        return False
    if body["model"] == model_id:
        return False
    body["model"] = model_id
    return True


def apply_request_policies(
    raw: bytes,
    *,
    model_id: str | None = None,
    overlay: Mapping[str, Any] | None = None,
    floor: int | None = None,
    ctx: int = 0,
) -> bytes:
    """Apply every enabled request policy in **one** parse/serialise round trip.

    This is the single entry point on purpose: parsing a request body that may
    carry a megabyte of base64 image once per policy would be three times the
    cost for the same answer, and three near-identical wrappers around it were
    three places for the byte-identity contract to drift apart.

    Returns ``raw`` itself when no policy is enabled, when the body is not
    JSON, when it is not a JSON *object*, or when every enabled policy decided
    there was nothing to change.
    """
    if model_id is None and not overlay and not (isinstance(floor, int) and floor > 0):
        return raw
    try:
        body = json.loads(raw)
    except Exception:
        return raw
    if not isinstance(body, dict):
        return raw
    changed = False
    if _apply_model(body, model_id):
        changed = True
    if _apply_overlay(body, overlay):
        changed = True
    if _apply_floor(body, floor, ctx):
        changed = True
    return _dumps(body) if changed else raw


# ---------------------------------------------------------------------------
# Routing: find the request's `model` without buffering the request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelScan:
    """Verdict of :func:`scan_model` on one prefix of a request body.

    ``complete`` is the important half: ``ModelScan(None, False)`` means "ask
    me again with more bytes", while ``ModelScan(None, True)`` means "this body
    has no top-level ``model``" and the caller should stop reading.
    """

    model: str | None
    complete: bool


_WS = frozenset(b" \t\r\n")
_MODEL_KEY = b'"model"'


def _scan_string(buf: bytes, i: int) -> tuple[int, bytes | None]:
    """``buf[i]`` is ``"``.  Return (index past the closing quote, raw token).

    ``(i, None)`` means the string is not finished inside ``buf`` yet.  Byte
    level scanning is safe for UTF-8 here: every continuation byte is ≥ 0x80
    and so can never be mistaken for ``"`` (0x22) or ``\\`` (0x5c), which is
    why a chunk boundary falling mid-character cannot confuse this.
    """
    n = len(buf)
    j = i + 1
    while j < n:
        c = buf[j]
        if c == 0x5C:  # backslash: skip the escaped byte, whatever it is
            j += 2
            continue
        if c == 0x22:
            return j + 1, buf[i : j + 1]
        j += 1
    return i, None


def scan_model(buf: bytes) -> ModelScan:
    """Extract the **top-level** ``model`` of a JSON object from a prefix.

    A structural scan, not a substring search, because ``"model"`` appears
    inside request bodies for reasons that have nothing to do with routing —
    a tool schema with a ``model`` property, a user message discussing models,
    a nested ``response_format`` — and routing a request to the wrong upstream
    on one of those would be a 404 in somebody's editor.  Only a ``model`` key
    at depth 1 of the top-level object, with a string value, counts.

    Tolerant of truncation: given half a body it reports ``complete=False``
    rather than guessing, so the caller can read one more chunk and ask again.

    A ``\\u``-escaped spelling of the key itself (``"mo\\u0064el"``) is not
    recognised.  No client emits one, and the gateway pre-filters on the plain
    bytes before calling this at all, so supporting it here would be dead code.
    """
    n = len(buf)
    i = 0
    depth = 0
    key: bytes | None = None
    after_colon = False
    while i < n:
        c = buf[i]
        if c in _WS:
            i += 1
            continue
        if depth == 0 and c != 0x7B:  # '{'
            # Not a JSON object at the top level (an array, a bare value, or
            # something that is not JSON at all): there is no model to find.
            return ModelScan(None, True)
        if c == 0x22:  # '"'
            j, raw = _scan_string(buf, i)
            if raw is None:
                return ModelScan(None, False)
            if depth == 1:
                if after_colon:
                    if key == _MODEL_KEY:
                        try:
                            value = json.loads(raw.decode("utf-8"))
                        except Exception:
                            return ModelScan(None, True)
                        return ModelScan(value if isinstance(value, str) else None, True)
                    key = None
                    after_colon = False
                else:
                    key = raw
            i = j
            continue
        if c == 0x7B or c == 0x5B:  # '{' '['
            depth += 1
            i += 1
            continue
        if c == 0x7D or c == 0x5D:  # '}' ']'
            depth -= 1
            if depth <= 0:
                # The top-level object closed and no model key turned up.
                return ModelScan(None, True)
            if depth == 1:
                key = None
                after_colon = False
            i += 1
            continue
        if c == 0x3A:  # ':'
            if depth == 1:
                after_colon = True
            i += 1
            continue
        if c == 0x2C:  # ','
            if depth == 1:
                key = None
                after_colon = False
            i += 1
            continue
        # A bare token: number, true, false, null.  Its bytes are never
        # structural, so stepping through them one at a time is safe.
        if depth == 1 and after_colon and key == _MODEL_KEY:
            return ModelScan(None, True)  # present, but not a string
        i += 1
    return ModelScan(None, False)


def chain_body(prefix: bytes, rest: AsyncIterator[bytes] | None) -> AsyncIterator[bytes]:
    """Re-join the bytes already read for routing with the unread remainder.

    The gateway reads a request body only until :func:`scan_model` reaches a
    verdict; everything after that point is still in the client's socket.  This
    hands httpx one stream made of both halves, so the body crosses the gateway
    without ever being held in full — TCP backpressure all the way through.
    """

    async def _gen() -> AsyncIterator[bytes]:
        if prefix:
            yield prefix
        if rest is not None:
            async for chunk in rest:
                if chunk:
                    yield chunk

    return _gen()


__all__ = [
    "DROP_REQUEST_HEADERS",
    "DROP_RESPONSE_HEADERS",
    "ModelScan",
    "apply_request_policies",
    "chain_body",
    "mirror_reasoning",
    "scan_model",
    "sse_stream",
    "transform_json_body",
    "transform_sse_line",
]
