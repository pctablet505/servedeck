"""GLM-5.3 client-compatibility policies — the half of ``glm53-effort-proxy``
that P2 (the gateway packet) deliberately left behind.

P2 ported the transforms that every model may need: the reasoning mirror, the
effort overlay, the output floor, the model rewrite.  What it left in
``~/Projects/glm53-effort-proxy/proxy.py`` were the repairs that exist because
of **one model's tool-call template and one client's history handling** — and
retiring that proxy (cutover step 6) without them would put back, silently,
failures that were captured live.  They live here rather than in
``policies.py`` so that the generic layer stays generic, and so that this
file's switches can be argued about on their own evidence.

What is here, and the evidence for each
---------------------------------------
1. **Tool-tag sanitiser** (``proxy.py`` ~301-349).  GLM's streaming tool-call
   parser can miss a closing tag that straddles a token boundary, and the tag
   then leaks into the parsed argument.  Captured from a real VS Code Copilot
   turn::

       list_dir({"path</arg_key>": "/Users/user/Desktop/project"})

   The client rejects the call ("must have required property 'path'"), the
   agent loses its step and starts inventing paths.  **On by default** for
   glm53: it is a pure repair of markup that is never valid argument text, and
   there is a recorded failure it prevents.

   *Ported with a change of direction.*  The original sanitised only the
   **request** (the history a client echoes back), which cleans the transcript
   but does nothing about the client that already rejected the call.  Here the
   repair runs on the **response** too — non-streaming and streaming — so the
   broken key never reaches the client at all.  The request side is kept as
   well, because a transcript poisoned before this packet shipped is still out
   there in editor sessions.

2. **Reasoning restore by tool-call id** (``proxy.py`` ~352-466).  Copilot
   echoes ``tool_call`` ids back verbatim but drops the reasoning that came
   with them, so GLM's template renders the in-flight assistant turn as an
   empty ``<think></think>``; the model reads its own blank thinking and
   concludes no task was ever given.  This is an **input**-side repair and the
   gateway's output-side mirror cannot substitute for it: the mirror makes the
   thinking *available* to a client, it cannot make a client send it back.

   **OFF by default.**  The source's own comment records why it was made
   switchable: *"three Xid 31 GPU faults followed within 40 minutes of it
   first firing, after 14 hours clean."*  That is correlation, not proof — but
   this box has an active, unrelated GPU-fault problem, so nothing that
   plausibly perturbs the GPU may default to on.  Turning it on is a decision
   with evidence attached, not a default.

3. **Answerless-turn detection** (``proxy.py`` ~216-247).  A turn with no
   ``content`` and no ``tool_calls`` is useless to a client *and* poisons the
   next request: the blank assistant turn goes back in the history and the
   model reads the thread as fresh.  The detection is ported.  The original's
   automatic **retry at effort=high is NOT** — see :func:`note_answerless`.

4. **Fresh-thread detection** (``proxy.py`` ~177-213) is **refused**.  See the
   ``REFUSED:`` block at the bottom of this module for the argument.

5. **Trace/capture** (``proxy.py`` ~249-298) is replaced, not ported.  The
   original defaults capture to *on* and writes full request and response
   bodies — everything the user typed, plus whatever headers a client sent —
   to ``/tmp/glm-capture`` with no retention policy.  Here it is opt-in, needs
   an explicit env/CLI flag *in addition to* the route's own consent, keeps a
   bounded ring under ``state/captures/`` (gitignored, mode 0700) and redacts
   credential headers.

Statefulness
------------
The original kept one process-global ``dict`` for the reasoning map.  Here the
state is an explicit :class:`GlmState` object owned by the gateway's router,
holding one bucket **per route**, each bucket bounded and tagged with its
model id.  A model switch therefore cannot inject one model's reasoning into
another model's prompt — the original had no such guard, and it is the failure
mode that matters most here, because the injected text goes into the *prompt*
and a prompt from the wrong model is not merely wrong, it is malformed.

Purity
------
Everything except :class:`CaptureRing` (files) and :func:`note_answerless`
(one log line) is a pure transform over bytes or parsed objects, for the same
reason ``policies.py`` is: the whole of it can be proved against recorded
traffic and hand-built payloads without a GPU.  The
**byte-identical-passthrough** rule is inherited verbatim — a transform that
finds no work returns the *same object* it was given.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The serialiser is imported rather than re-implemented on purpose: "a body we
# did not change comes back byte-identical" is one contract with one
# implementation, verified against recorded vLLM replies in test_policies.py.
# A second copy here would be a second thing to keep in step.
from .policies import _dumps

logger = logging.getLogger(__name__)

__all__ = [
    "CAPTURE_ENV",
    "LEAKED_TAGS",
    "MAX_RESTORED_CHARS",
    "REDACTED_HEADERS",
    "UNSAFE_IN_PROMPT",
    "CaptureRing",
    "GlmHook",
    "GlmRouteState",
    "GlmState",
    "GlmStats",
    "ReasoningStore",
    "begin",
    "capture_enabled",
    "clean_fragment",
    "is_answerless",
    "redact_headers",
    "restore_reasoning",
    "safe_reasoning",
    "sanitize_message_list",
    "sanitize_tool_calls",
    "strip_leaked_tags",
]


# ===========================================================================
# 1. Tool-tag sanitiser
# ===========================================================================

#: The tags GLM's tool-call format uses.  None of them is ever valid inside a
#: JSON argument key or value, which is what makes removing them a repair
#: rather than a guess.  Longest-first so no removal can leave a shorter tag's
#: fragment behind.
LEAKED_TAGS: tuple[str, ...] = (
    "</arg_value>",
    "</tool_call>",
    "<arg_value>",
    "<tool_call>",
    "</arg_key>",
    "<arg_key>",
)

#: Longest tag, in characters.  Bounds how much of a streamed argument
#: fragment must be held back to catch a tag split across two deltas.
MAX_TAG_LEN = max(len(t) for t in LEAKED_TAGS)

#: Removal can expose a tag that was not there before (``<<arg_key>arg_key>``),
#: so the strip iterates to a fixed point.  Capped, because a fixed point is
#: guaranteed only by the string getting shorter and a cap is cheaper to
#: reason about than that argument.
_STRIP_PASSES = 4


def strip_leaked_tags(text: str) -> str:
    """Remove every leaked template tag.  Whitespace is **not** touched.

    Separate from :func:`clean_fragment` because a *streamed* argument
    fragment must not be whitespace-stripped: ``{"city": "Paris"}`` arrives
    split at arbitrary points and trimming a fragment's edges would corrupt
    the JSON the client is reassembling.
    """
    for _ in range(_STRIP_PASSES):
        before = text
        for tag in LEAKED_TAGS:
            if tag in text:
                text = text.replace(tag, "")
        if text == before:
            break
    return text


def clean_fragment(text: str) -> str:
    """Remove leaked tags from a *parsed* key or value.

    Trailing/leading whitespace is stripped **only when a tag was actually
    removed** — the leak arrives glued to real text (``"path</arg_key>"``) and
    leaves whitespace behind, so trimming is part of the repair.  The original
    (``proxy.py:_clean_fragment``) trimmed unconditionally, which silently
    rewrote arguments that had no leak at all: a legitimate ``{"text": "a "}``
    came out as ``{"text": "a"}`` and was counted as a fix.  A sanitiser that
    edits healthy payloads is not one you can leave on by default, so this is
    the version that ships enabled.
    """
    stripped = strip_leaked_tags(text)
    if stripped == text:
        return text
    return stripped.strip()


def _clean_tree(obj: Any) -> tuple[Any, bool]:
    """Clean every string key and value in a parsed argument object.

    Recursive, unlike the original, which only looked at the top level: the
    captured failure was a top-level key, but a leak inside a nested edit list
    (``{"edits": [{"path</arg_key>": ...}]}``) is the same defect and the same
    repair, and a sanitiser that stops at depth 1 would report success on it.
    """
    if isinstance(obj, str):
        cleaned = clean_fragment(obj)
        return cleaned, cleaned != obj
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        changed = False
        for k, v in obj.items():
            nk: Any = k
            if isinstance(k, str):
                nk = clean_fragment(k)
                if nk != k:
                    changed = True
            nv, sub = _clean_tree(v)
            changed = changed or sub
            out[nk] = nv
        return out, changed
    if isinstance(obj, list):
        items = []
        changed = False
        for v in obj:
            nv, sub = _clean_tree(v)
            changed = changed or sub
            items.append(nv)
        return items, changed
    return obj, False


def sanitize_tool_calls(msg: Mapping[str, Any] | dict) -> int:
    """Strip leaked markup out of one message's tool-call arguments.

    Mutates ``msg["tool_calls"][*]["function"]["arguments"]`` in place and
    returns the number of tool calls repaired.  Two shapes, both from the
    original:

    * ``arguments`` parses as JSON — clean its keys and values, re-serialise
      only if something changed;
    * ``arguments`` does not parse at all — strip the tags and try once more,
      which is the case where the leak broke the JSON itself.
    """
    fixed = 0
    calls = msg.get("tool_calls")
    if not isinstance(calls, list):
        return 0
    for tc in calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict):
            continue
        raw = fn.get("arguments")
        if not isinstance(raw, str) or not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            cleaned = strip_leaked_tags(raw)
            if cleaned == raw:
                continue
            try:
                parsed = json.loads(cleaned)
            except Exception:
                # Still not JSON.  Leave it alone: handing the client a
                # different broken string is not an improvement, and the
                # untouched original is what the failure report will need.
                continue
            fn["arguments"] = json.dumps(parsed, ensure_ascii=False)
            fixed += 1
            continue
        repaired, changed = _clean_tree(parsed)
        if changed:
            fn["arguments"] = json.dumps(repaired, ensure_ascii=False)
            fixed += 1
    return fixed


def sanitize_message_list(messages: Any) -> int:
    """Sanitise the tool calls of every assistant turn in a request history."""
    if not isinstance(messages, list):
        return 0
    return sum(sanitize_tool_calls(m) for m in messages if isinstance(m, dict))


def _partial_tag_suffix(text: str) -> int:
    """Length of the trailing run of ``text`` that could still become a tag.

    Called on text from which every *complete* tag has already been removed,
    so any suffix matching the start of a tag is necessarily a partial one.
    This is the streaming hold-back: it is what makes the sanitiser correct
    when a leaked tag is split across two ``arguments`` deltas — which is the
    likely case, since the tag leaked *because* it straddled a token boundary
    in the first place.
    """
    for n in range(min(len(text), MAX_TAG_LEN - 1), 0, -1):
        suffix = text[-n:]
        if any(tag.startswith(suffix) and n < len(tag) for tag in LEAKED_TAGS):
            return n
    return 0


# ===========================================================================
# 2. Reasoning restore, keyed by tool-call id
# ===========================================================================

#: Markup that must never be re-injected into a prompt.  The stored text is
#: the model's own output; a think/tool/role marker inside it would nest or
#: terminate the ``<think>`` block the template puts it in, and the rendered
#: prompt would stop being well-formed.  Stripped rather than skipped, so the
#: reasoning is still restored (that is what fixes the amnesia) without the
#: structural risk.  Ported verbatim from ``proxy.py:_UNSAFE_IN_PROMPT``.
UNSAFE_IN_PROMPT: tuple[str, ...] = (
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<|assistant|>",
    "<|user|>",
    "<|system|>",
    "<|observation|>",
    "<tool_response>",
    "</tool_response>",
    "[gMASK]",
    "<sop>",
)

#: A restored block is prompt tokens on *every* later turn of the thread, so
#: it is bounded.  4000 characters ≈ 1000 tokens, the original's number.
MAX_RESTORED_CHARS = 4000

#: How many tool-call ids one route remembers.  The original cleared the whole
#: dict when it passed this (throwing away the in-flight thread's reasoning
#: along with the stale entries); this evicts oldest-first instead, so the
#: entries that are about to be asked for are the ones that survive.
DEFAULT_MAX_REASONING_ENTRIES = 400


def safe_reasoning(text: str) -> str:
    """Make stored reasoning safe to put back into a prompt, and bound it."""
    for tag in UNSAFE_IN_PROMPT:
        if tag in text:
            text = text.replace(tag, "")
    text = text.strip()
    if len(text) > MAX_RESTORED_CHARS:
        text = text[:MAX_RESTORED_CHARS] + " ..."
    return text


class ReasoningStore:
    """One route's bounded ``tool_call id -> reasoning`` map.

    Tagged with ``model_id`` and every entry re-checked against it on read.
    That is redundant with :meth:`GlmState.route` handing out one store per
    model — deliberately, because the failure it prevents is injecting one
    model's reasoning into another model's prompt, and a single guard on a
    path that writes into a *prompt* is one guard too few.
    """

    def __init__(self, model_id: str, *, max_entries: int = DEFAULT_MAX_REASONING_ENTRIES) -> None:
        self.model_id = model_id
        self.max_entries = max(1, int(max_entries))
        #: insertion-ordered; Python dicts are, which is the whole LRU here
        self._entries: dict[str, tuple[str, str]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def remember(self, tool_call_ids: Iterable[str], reasoning: str) -> int:
        """Record ``reasoning`` against each id.  Returns how many were stored."""
        text = reasoning.strip() if isinstance(reasoning, str) else ""
        if not text:
            return 0
        stored = 0
        for tcid in tool_call_ids:
            if not isinstance(tcid, str) or not tcid:
                continue
            self._entries.pop(tcid, None)  # re-insert so it counts as newest
            self._entries[tcid] = (self.model_id, text)
            stored += 1
            while len(self._entries) > self.max_entries:
                self._entries.pop(next(iter(self._entries)))
        return stored

    def get(self, tool_call_id: Any, *, model_id: str) -> str | None:
        """Reasoning for ``tool_call_id``, **only** if it was this model's.

        A mismatch returns ``None`` rather than raising: the caller is about to
        build a prompt and the right answer to "I am not sure whose thinking
        this is" is to restore nothing.
        """
        if not isinstance(tool_call_id, str):
            return None
        entry = self._entries.get(tool_call_id)
        if entry is None:
            return None
        owner, text = entry
        if owner != model_id:
            return None
        return text


def restore_reasoning(body: dict, *, store: ReasoningStore, model_id: str) -> int:
    """Put reasoning back on the assistant turns after the last user message.

    Only those turns: they are the in-flight tool-call loop, the one GLM's
    template renders as an empty ``<think></think>`` when the reasoning is
    absent.  Earlier turns are closed and their thinking is not read.

    Returns how many turns were repaired.  Mutates ``body["messages"]``.
    """
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return 0
    last_user = -1
    for i, m in enumerate(msgs):
        if isinstance(m, dict) and m.get("role") == "user":
            last_user = i
    restored = 0
    for i, m in enumerate(msgs):
        if i <= last_user or not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        if m.get("reasoning_content") or m.get("reasoning"):
            continue
        calls = m.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for tc in calls:
            if not isinstance(tc, dict):
                continue
            stored = store.get(tc.get("id"), model_id=model_id)
            if not stored:
                continue
            safe = safe_reasoning(stored)
            if not safe:
                break
            m["reasoning_content"] = safe
            restored += 1
            break
    return restored


def _message_reasoning(msg: Mapping[str, Any]) -> str:
    for key in ("reasoning", "reasoning_content"):
        value = msg.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _message_tool_ids(msg: Mapping[str, Any]) -> list[str]:
    calls = msg.get("tool_calls")
    if not isinstance(calls, list):
        return []
    return [
        tc["id"]
        for tc in calls
        if isinstance(tc, dict) and isinstance(tc.get("id"), str) and tc["id"]
    ]


# ===========================================================================
# 3. Answerless-turn detection — detection only, no retry
# ===========================================================================


def is_answerless(msg: Mapping[str, Any]) -> bool:
    """True if this assistant message is useless to a client.

    No ``content`` and no ``tool_calls``.  Two ways GLM-5.3 produces one:
    ``finish_reason="length"`` (the template ends the prompt with a bare
    ``<think>``, so at ``reasoning_effort=max`` the model can exhaust the
    budget before emitting ``</think>``), or ``finish_reason="stop"`` (it
    finishes thinking and stops without writing an answer).  The thinking
    survives under ``reasoning``, but no client renders that, so the user sees
    a blank turn — and the blank goes back into the history on the next
    request, where the template renders it as an empty assistant message and
    the model reads the thread as fresh.

    A message with ``tool_calls`` and ``content: null`` is **normal** and must
    never be counted: that is how every tool call is returned.
    """
    if msg.get("tool_calls"):
        return False
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return False
    if isinstance(content, list) and content:
        # Multimodal/structured content: any part at all counts as an answer.
        return False
    return True


def note_answerless(model_id: str, *, floor: int | None, stream: bool) -> None:
    """Log the one warning an answerless turn earns.  **No retry.**

    The original retried the whole request at ``reasoning_effort=high`` — for a
    streaming request it re-ran it non-streaming — and, if that also came back
    empty, promoted the reasoning into ``content``.  That is refused here:

    * a gateway that silently re-runs a request **doubles GPU work** on a box
      whose main slot runs at ``--max-num-seqs 1``, so the retry does not
      merely cost tokens, it serialises behind and delays every other request;
    * it **hides the cause**.  The measured root cause is a too-small output
      budget spent inside ``<think>``, and the fix for that — the output floor
      (``min_output_tokens``, ``policies._apply_floor``) — is already ported
      and already on for glm53.  A retry that papers over a floor that is set
      wrong means nobody ever finds out the floor is set wrong;
    * the retry changed the caller's own ``chat_template_kwargs`` and, on the
      streaming path, silently converted a streaming request into a
      non-streaming one.  A proxy that answers a different request than the
      one it was given is not a proxy.

    What a user sees instead: the upstream's own answerless reply, unmodified
    (an empty assistant turn — the honest report that the model produced
    nothing), this warning in the journal naming the likely cause, and the
    count on ``GlmState.stats()``.  The empty turn is not *good*; it is the
    truth, and the floor is what stops it happening.
    """
    logger.warning(
        "%s: answerless turn (no content, no tool_calls) on the %s path. "
        "Most likely the whole output budget was spent inside an unterminated "
        "<think> block; this route's min_output_tokens floor is %s. Not "
        "retried on purpose: a silent retry doubles GPU work at "
        "--max-num-seqs 1 and hides a floor that is set too low.",
        model_id,
        "streaming" if stream else "non-streaming",
        floor if floor else "unset",
    )


# ===========================================================================
# 5. Capture — opt-in, bounded, redacted
# ===========================================================================

#: The env var that is the **master** switch.  A route's own ``capture = true``
#: is consent, not an enable: a config file that someone edited months ago
#: must not be able to start writing everything a user types to disk.  Both
#: must be true, and this one is deliberately not in ``models.toml``.
CAPTURE_ENV = "SERVEDECK_GLM_CAPTURE"

_TRUE = frozenset({"1", "true", "yes", "on"})

#: Headers whose *values* are credentials.  The gateway already refuses to
#: relay these upstream (``policies.DROP_REQUEST_HEADERS``); a capture file is
#: the other way they could escape, so they are redacted rather than omitted —
#: knowing a client sent an ``authorization`` header is diagnostic, knowing
#: what was in it is a liability.
REDACTED_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "api-key",
        "x-api-key",
        "openai-api-key",
        "openai-organization",
        "cookie",
        "set-cookie",
        "x-auth-token",
    }
)

#: Turns kept in the ring.  The original's number; six turns is enough to hold
#: the failing turn plus the ones that led to it, and small enough that a
#: forgotten flag cannot fill a disk.
DEFAULT_CAPTURE_RING = 6


def capture_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether the master capture switch is set.  Read at call time, so a test
    (or an operator) can change it without restarting anything."""
    source = os.environ if env is None else env
    return source.get(CAPTURE_ENV, "").strip().lower() in _TRUE


def redact_headers(headers: Any) -> dict[str, str]:
    """Header map with every credential value replaced by ``<redacted>``."""
    out: dict[str, str] = {}
    try:
        items = headers.items()
    except AttributeError:
        return out
    for k, v in items:
        key = str(k)
        out[key] = "<redacted>" if key.lower() in REDACTED_HEADERS else str(v)
    return out


class CaptureRing:
    """A fixed-size ring of turn captures under ``state/captures/``.

    ``<dir>/req-<n>.json`` and ``<dir>/resp-<n>.json`` for the same ``n`` are
    the two halves of one turn, so a capture can be read as a request/response
    pair the way the original's was.  ``n`` wraps at :attr:`size`, which *is*
    the retention policy the original did not have.

    Every write is best-effort: a capture that cannot be written must never
    turn into a failed request for the user.
    """

    def __init__(self, directory: Path, *, size: int = DEFAULT_CAPTURE_RING) -> None:
        self.directory = Path(directory)
        self.size = max(1, int(size))
        self._turn = -1
        self.written = 0
        self.errors = 0

    def next_turn(self) -> int:
        self._turn = (self._turn + 1) % self.size
        return self._turn

    def write(self, kind: str, turn: int, payload: Any) -> Path | None:
        try:
            # 0700: these files hold whatever the user typed at their editor.
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = self.directory / f"{kind}-{turn}.json"
            path.write_text(json.dumps(payload, indent=1, default=str))
            path.chmod(0o600)
            self.written += 1
            return path
        except Exception:  # noqa: BLE001 — diagnostics never break a request
            self.errors += 1
            return None


def _json_or_text(raw: bytes) -> Any:
    """Parsed JSON when it parses, otherwise a bounded text excerpt."""
    try:
        return json.loads(raw)
    except Exception:
        return {"_unparsed": raw[:4096].decode("utf-8", "replace"), "_bytes": len(raw)}


# ===========================================================================
# State: one bucket per route, owned by the router
# ===========================================================================


@dataclass
class GlmStats:
    """Counters for one route.  Read by the dashboard; never by a policy."""

    answerless_turns: int = 0
    tool_call_repairs: int = 0
    reasoning_remembered: int = 0
    reasoning_restored: int = 0
    captures_written: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "answerless_turns": self.answerless_turns,
            "tool_call_repairs": self.tool_call_repairs,
            "reasoning_remembered": self.reasoning_remembered,
            "reasoning_restored": self.reasoning_restored,
            "captures_written": self.captures_written,
        }


@dataclass
class GlmRouteState:
    """Everything this packet remembers about one model."""

    model_id: str
    reasoning: ReasoningStore
    stats: GlmStats = field(default_factory=GlmStats)


class GlmState:
    """The router-owned home for every route's GLM policy state.

    Created once in ``gateway.build_router`` and reachable as
    ``router.glm_state``, which is what the dashboard reads for
    ``stats()`` — there is no module-level dictionary anywhere in this file,
    which is the difference from the original and the reason two gateways in
    one process (the rehearsal on :8011 during cutover step 1, for instance)
    cannot see each other's reasoning.
    """

    def __init__(
        self,
        *,
        capture_dir: Path | None = None,
        capture_size: int = DEFAULT_CAPTURE_RING,
        max_reasoning_entries: int = DEFAULT_MAX_REASONING_ENTRIES,
    ) -> None:
        self._routes: dict[str, GlmRouteState] = {}
        self._capture_dir = capture_dir
        self._capture_size = capture_size
        self._max_reasoning_entries = max_reasoning_entries
        self._rings: dict[str, CaptureRing] = {}

    # -- per-route -----------------------------------------------------------
    def route(self, model_id: str) -> GlmRouteState:
        state = self._routes.get(model_id)
        if state is None:
            state = GlmRouteState(
                model_id=model_id,
                reasoning=ReasoningStore(model_id, max_entries=self._max_reasoning_entries),
            )
            self._routes[model_id] = state
        return state

    def forget(self, model_id: str) -> None:
        """Drop a model's remembered reasoning — the hook for a model switch.

        Counters survive, because a count of answerless turns is history and
        history does not become false when a model stops.  Not *required* for
        correctness (the model-id guard already makes cross-model injection
        impossible); it is how the memory is reclaimed when a model that will
        not come back is stopped.
        """
        state = self._routes.get(model_id)
        if state is not None:
            state.reasoning = ReasoningStore(
                model_id, max_entries=self._max_reasoning_entries
            )

    def ring(self, model_id: str) -> CaptureRing:
        ring = self._rings.get(model_id)
        if ring is None:
            base = self._capture_dir
            if base is None:
                from . import settings as _settings  # local: keep import cheap

                base = _settings.get().state_dir / "captures"
            safe = "".join(c if (c.isalnum() or c in "-._") else "_" for c in model_id)
            ring = CaptureRing(Path(base) / safe, size=self._capture_size)
            self._rings[model_id] = ring
        return ring

    # -- read side -----------------------------------------------------------
    def stats(self) -> dict[str, dict[str, int]]:
        """``{model_id: {counter: value}}`` for every route seen this session."""
        return {mid: st.stats.as_dict() for mid, st in sorted(self._routes.items())}

    def answerless_turns(self, model_id: str) -> int:
        state = self._routes.get(model_id)
        return 0 if state is None else state.stats.answerless_turns


# ===========================================================================
# The per-request hook the gateway calls
# ===========================================================================


class GlmHook:
    """One request's worth of GLM policy.  Built by :func:`begin`.

    The gateway holds exactly one of these per request and calls at most four
    methods on it, so the whole of this packet's presence in ``gateway.py`` is
    ``begin()`` plus four one-line call sites.
    """

    def __init__(
        self,
        *,
        route_state: GlmRouteState,
        sanitize: bool,
        restore: bool,
        ring: CaptureRing | None,
        floor: int | None,
        headers: Any = None,
    ) -> None:
        self._st = route_state
        self.model_id = route_state.model_id
        self._sanitize = sanitize
        self._restore = restore
        self._ring = ring
        self._floor = floor
        self._headers = headers
        self._turn = ring.next_turn() if ring is not None else -1
        # streaming state
        self._carry: dict[tuple[Any, Any], str] = {}
        self._envelope: dict[str, Any] = {}
        self._reasoning: list[str] = []
        self._tool_ids: list[str] = []
        self._saw_answer = False
        self._saw_chunk = False
        #: tool calls whose streamed arguments actually had a tag removed,
        #: counted once each at end of stream rather than once per frame — a
        #: tag split across two deltas is ONE repair, not two.
        self._repaired: set[tuple[Any, Any]] = set()

    # -- what the gateway needs to know before it reads the body -------------
    @property
    def needs_request_body(self) -> bool:
        """True when the request body must be in hand.

        ``capture`` is included: capturing half a body would be worse than not
        capturing it, because the half would look complete.
        """
        return self._sanitize or self._restore or self._ring is not None

    # -- request -------------------------------------------------------------
    def on_request_body(self, raw: bytes) -> bytes:
        """Repair the outgoing request.  Returns ``raw`` itself if unchanged."""
        if self._ring is not None:
            self._ring.write(
                "req",
                self._turn,
                {
                    "model_id": self.model_id,
                    "headers": redact_headers(self._headers),
                    "body": _json_or_text(raw),
                },
            )
            self._st.stats.captures_written = self._ring.written
        if not (self._sanitize or self._restore):
            return raw
        try:
            body = json.loads(raw)
        except Exception:
            return raw
        if not isinstance(body, dict):
            return raw
        changed = False
        if self._sanitize:
            fixed = sanitize_message_list(body.get("messages"))
            if fixed:
                self._st.stats.tool_call_repairs += fixed
                changed = True
        if self._restore:
            restored = restore_reasoning(
                body, store=self._st.reasoning, model_id=self.model_id
            )
            if restored:
                self._st.stats.reasoning_restored += restored
                changed = True
        return _dumps(body) if changed else raw

    # -- non-streaming response ---------------------------------------------
    def on_json_body(self, raw: bytes) -> bytes:
        """Repair the incoming reply.  Returns ``raw`` itself if unchanged."""
        data: Any
        try:
            data = json.loads(raw)
        except Exception:
            self._capture_response(_json_or_text(raw))
            return raw
        changed = False
        if isinstance(data, dict):
            for choice in data.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                msg = choice.get("message")
                if not isinstance(msg, dict):
                    continue
                if self._sanitize:
                    fixed = sanitize_tool_calls(msg)
                    if fixed:
                        self._st.stats.tool_call_repairs += fixed
                        changed = True
                if self._restore:
                    self._remember(_message_reasoning(msg), _message_tool_ids(msg))
                if is_answerless(msg):
                    self._st.stats.answerless_turns += 1
                    note_answerless(self.model_id, floor=self._floor, stream=False)
        self._capture_response(data)
        return _dumps(data) if changed else raw

    # -- streaming response --------------------------------------------------
    async def sse(self, source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """Line-transform an SSE stream, holding back only a partial line.

        Same hold-back discipline as ``policies.sse_stream`` (and it composes
        on top of it when the reasoning mirror is also on): upstream chunk
        boundaries fall anywhere, including mid-JSON and mid-UTF-8, so at most
        one incomplete SSE event is ever in hand and every complete line is
        flushed on the iteration that completed it.
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
                out += self._line(line + b"\n")
            if out:
                yield bytes(out)
        if buf:
            rest = self._line(buf)
            if rest:
                yield rest
        tail = self._end_of_stream()
        if tail:
            yield tail

    def _line(self, line: bytes) -> bytes:
        if b"data:" not in line:
            return line
        body, ending = line, b""
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
        try:
            data = json.loads(payload)
        except Exception:
            return line
        if not isinstance(data, dict):
            return line

        if not self._envelope:
            self._envelope = {
                "id": data.get("id", "chatcmpl-servedeck"),
                "object": data.get("object", "chat.completion.chunk"),
                "created": data.get("created", 0),
                "model": data.get("model", self.model_id),
            }

        pre = bytearray()
        changed = False
        for choice in data.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            self._saw_chunk = True
            cidx = choice.get("index", 0)
            delta = choice.get("delta")
            final = choice.get("finish_reason") is not None
            if isinstance(delta, dict):
                self._watch(delta)
                # Sanitise BEFORE any flush frame is emitted, and tell it when
                # this is the choice's last chunk so it holds nothing back.
                if self._sanitize and self._sanitize_delta(delta, cidx, final=final):
                    changed = True
            if final:
                # Any carry left on a tool call this chunk did NOT touch is
                # complete text with nothing following it: emit it as its own
                # well-formed chunk before the finish chunk, rather than
                # appending it to a delta the client is about to close.
                pre += self._flush_frame(cidx)
        if not pre and not changed:
            return line
        return bytes(pre) + b"data:" + lead + _dumps(data) + ending

    def _watch(self, delta: Mapping[str, Any]) -> None:
        content = delta.get("content")
        if isinstance(content, str) and content.strip():
            self._saw_answer = True
        if delta.get("tool_calls"):
            self._saw_answer = True
        if self._restore:
            r = delta.get("reasoning")
            if not isinstance(r, str) or not r:
                r = delta.get("reasoning_content")
            if isinstance(r, str) and r:
                self._reasoning.append(r)
            for tc in delta.get("tool_calls") or []:
                if isinstance(tc, dict) and isinstance(tc.get("id"), str) and tc["id"]:
                    self._tool_ids.append(tc["id"])

    def _sanitize_delta(self, delta: dict, cidx: Any, *, final: bool) -> bool:
        calls = delta.get("tool_calls")
        if not isinstance(calls, list):
            return False
        changed = False
        for tc in calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if not isinstance(fn, dict):
                continue
            raw = fn.get("arguments")
            if not isinstance(raw, str):
                continue
            key = (cidx, tc.get("index", 0))
            joined = self._carry.pop(key, "") + raw
            text = strip_leaked_tags(joined)
            if len(text) < len(joined):
                # Something was actually removed. Tracked separately from
                # `changed` below, which is also true when all that happened
                # was a carry being prepended or held back.
                self._repaired.add(key)
            hold = 0 if final else _partial_tag_suffix(text)
            if hold:
                self._carry[key] = text[len(text) - hold :]
                text = text[: len(text) - hold]
            if text != raw:
                fn["arguments"] = text
                changed = True
        return changed

    def _flush_frame(self, cidx: Any) -> bytes:
        """One SSE chunk carrying whatever is still held back for a choice."""
        pending = [(key[1], text) for key, text in self._carry.items() if key[0] == cidx and text]
        if not pending:
            return b""
        for key in [k for k in self._carry if k[0] == cidx]:
            self._carry.pop(key, None)
        frame = {
            **self._envelope,
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": cidx,
                    "delta": {
                        "tool_calls": [
                            {"index": tcidx, "function": {"arguments": text}}
                            for tcidx, text in pending
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        return b"data: " + _dumps(frame) + b"\n\n"

    def _end_of_stream(self) -> bytes:
        """Flush, remember, count — after the last upstream byte."""
        tail = bytearray()
        if self._carry:
            # Only reachable when the stream ended without a finish_reason
            # (upstream aborted).  Held-back bytes are real argument text, so
            # they are delivered rather than dropped.
            for cidx in sorted({k[0] for k in self._carry}, key=repr):
                tail += self._flush_frame(cidx)
        if self._repaired:
            self._st.stats.tool_call_repairs += len(self._repaired)
            self._repaired.clear()
        if self._restore:
            self._remember("".join(self._reasoning), self._tool_ids)
        if self._saw_chunk and not self._saw_answer:
            self._st.stats.answerless_turns += 1
            note_answerless(self.model_id, floor=self._floor, stream=True)
        self._capture_response(
            {
                "stream": True,
                "saw_answer": self._saw_answer,
                "reasoning_chars": sum(len(p) for p in self._reasoning),
                "tool_call_ids": list(self._tool_ids),
            }
        )
        return bytes(tail)

    # -- shared --------------------------------------------------------------
    def _remember(self, reasoning: str, tool_ids: list[str]) -> None:
        if not reasoning or not tool_ids:
            return
        stored = self._st.reasoning.remember(tool_ids, reasoning)
        self._st.stats.reasoning_remembered += stored

    def _capture_response(self, payload: Any) -> None:
        if self._ring is None:
            return
        self._ring.write("resp", self._turn, {"model_id": self.model_id, "body": payload})
        self._st.stats.captures_written = self._ring.written


def begin(
    pol: Any,
    *,
    model_id: str,
    state: GlmState | None,
    responses_api: bool = False,
    headers: Any = None,
) -> GlmHook | None:
    """Build this request's hook, or ``None`` when nothing is switched on.

    ``None`` is the common case — every route but glm53 — and it is what keeps
    the gateway's no-buffering rule intact for everyone else: with no hook
    there is no reason to hold a body, so nothing about the hot path changes
    for a model that does not need these repairs.

    Answerless **detection** rides along whenever any switch is on rather than
    having a switch of its own.  It cannot be free-standing: counting it means
    reading the reply, reading a non-streaming reply means buffering it, and
    buffering every route's replies to count something only GLM's template
    produces would be a real cost for no gain.  ``sanitize_tool_tags``
    defaults true for glm53, so the count is on exactly where it means
    something.

    ``/v1/responses`` gets **no** hook.  Its envelope is a different shape
    (``output`` items, not ``choices[].message``) and none of these repairs is
    written for it — the original did not wire them there either
    (``proxy.py:492-500``: *"The answerless-turn rescue is NOT wired for this
    shape yet"*).  Silently applying a chat-completions-shaped transform to it
    would be worse than not applying one.  Codex is the client on that API and
    is not the client with these bugs.
    """
    if state is None or responses_api:
        return None
    sanitize = bool(getattr(pol, "sanitize_tool_tags", False))
    restore = bool(getattr(pol, "restore_reasoning", False))
    capture = bool(getattr(pol, "capture", False)) and capture_enabled()
    if not (sanitize or restore or capture):
        return None
    route_state = state.route(model_id)
    return GlmHook(
        route_state=route_state,
        sanitize=sanitize,
        restore=restore,
        ring=state.ring(model_id) if capture else None,
        floor=getattr(pol, "min_output_tokens", None),
        headers=headers,
    )


# ===========================================================================
# REFUSED: fresh-thread detection (proxy.py:177-213)
# ===========================================================================
#
# `_FRESH_THREAD_OPENERS` / `_reads_as_fresh_thread` match the START of a reply
# against six phrases ("i'm ready to help", "what would you like to work on",
# ...) and, when >= 6 messages preceded it, treat the reply as evidence that
# the model lost the thread — in the original, grounds for a silent retry and
# for freezing a capture outside the ring.
#
# Not ported, for three reasons:
#
# 1. **It fires on legitimate replies.**  "What would you like to work on?" is
#    a correct answer to "what should we do next?", and an agent asking a
#    clarifying question at turn 7 of a session is normal behaviour, not a
#    fault.  The guard cannot tell those apart, because the only thing it
#    looks at is the first 120 characters of the reply.
#
# 2. **The defect it was built for has since been diagnosed, and it is not
#    this.**  The greeting is a *symptom* of an empty `<think>` block — either
#    a blank assistant turn already in the history (`is_answerless`, item 3)
#    or reasoning the client dropped (`restore_reasoning`, item 2).  Both
#    causes are now addressed at the cause.  The source's own docstring says
#    as much: "a symptom-level guard, not a fix ... the cause is upstream of
#    the transcript and is still being traced."
#
# 3. **What it did on a match was the part that was actually harmful** — a
#    silent retry (refused for the reasons in `note_answerless`) and a capture
#    of the full request and reply, written outside the ring and never
#    deleted.
#
# If the greeting is ever wanted as an observation rather than an action, the
# place for it is P6's telemetry, offline, over captured turns — not in the
# request path, where a false positive costs the user a real answer.
