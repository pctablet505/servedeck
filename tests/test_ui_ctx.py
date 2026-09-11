"""The context control launches at the model's full native context.

THE DEFECT
----------
The owner started the Qwen3.8-27B from the dashboard and it came up with
``--max-model-len 110592``; the next long prompt failed with "This model's
maximum context length is 110592 tokens". The model's own
max_position_embeddings is 262,144, and at util 0.95 its KV pool (1,595,321
tokens on vLLM 0.29.0) holds 6.1 requests of that length at once.

The page divided the pool by the agent count (capacity.compute's
``ctx_max_fit = kv_tokens // max_num_seqs``, 16 agents) and treated the
quotient as the longest ANY request could be: renderCtx() clamped the slider
to it and the Apply button posted it. But --max-model-len is a per-REQUEST
ceiling; the pool is shared and vLLM admits what fits and queues the rest.
Capping every request at pool / agents made the long ones fail outright.

HOW THESE TESTS WORK
--------------------
Same method as tests/test_ui.py: a grep assertion survives any behaviour
mutation, so these EXECUTE web/app.js in the JS engine against the DOM shim.
The estimate payloads are built by the REAL app._estimate() (only its I/O --
the hub scan, the measurement store, nvidia-smi -- is scripted), and the page's
own Apply handler is clicked, so each test covers the whole chain from the
capacity arithmetic to the body POSTed at /api/server/start.
"""

from __future__ import annotations

import json

import pytest

from servedeck import app as capp
from servedeck import capacity, registry

from .test_ui import (
    APP_JS,
    INDEX_HTML,
    _PRELUDE,
    _const_line,
    _element_ids,
    _fn_body,
    _reads_of,
    _run_js,
)

pytest.importorskip("dukpy", reason="pip install -e '.[dev]' brings in the JS engine")

REPO = "RadixArk/Qwen3.8-27B-NVFP4"
NATIVE = 262_144
#: The live 27B's KV pool at util 0.95 on vLLM 0.29.0 (vllm:cache_config_info).
POOL = 1_595_321
WEIGHTS = 20.82
OVERHEAD = 4.33


# --------------------------------------------------------------------------
# The estimate, computed by the real app._estimate()
# --------------------------------------------------------------------------
def _entry(max_pe: int = NATIVE) -> registry.ModelEntry:
    return registry.ModelEntry(
        repo_id=REPO, hub_dirname="models--RadixArk--Qwen3.8-27B-NVFP4",
        snapshot_path="/nowhere", skipped=False, servable=True, backend="inline",
        safetensors_gib=19.0, safetensors_count=1, disk_bytes=20 * 1024**3,
        disk_local_bytes=20 * 1024**3, config_exists=True,
        architectures0="Qwen3_5ForConditionalGeneration", model_type="qwen3_5",
        max_position_embeddings=max_pe, num_hidden_layers=64, num_key_value_heads=4,
        head_dim=256, full_attention_interval=4, quant_algo="NVFP4", reason=None,
    )


def _pool_rate() -> float:
    """The KiB/token that makes util 0.95 buy exactly the live 1,595,321-token
    pool: (0.95 x 95.5928 - 20.82 - 4.33) GiB over 1,595,321.5 tokens."""
    kv_gib = 0.95 * capacity.GPU_TOTAL_GIB - WEIGHTS - OVERHEAD
    return kv_gib * 1048576 / (POOL + 0.5)


@pytest.fixture
def estimate(monkeypatch: pytest.MonkeyPatch):
    """app._estimate() with its I/O scripted. `rate(ctx)` is the KV rate the
    measurement store / calculator would resolve at that context length."""

    def go(util: float, ctx: int, agents: int, *, rate=None, weights: float = WEIGHTS,
           max_pe: int = NATIVE, source: str = "measured") -> dict:
        rate = rate or (lambda _ctx: _pool_rate())

        def resolve(repo_id, u, length, **_kw):
            return registry.ResolvedInputs(
                repo_id=repo_id, backend="inline", requested_util=u, requested_ctx=length,
                weights_gib=weights, weights_source="measured",
                kv_kib_per_token=rate(length), kv_source=source, trust="measured",
                overhead_gib=OVERHEAD, model_max_ctx=max_pe, model_type="qwen3_5",
                architectures0="Qwen3_5ForConditionalGeneration", matched_ctx=length,
            )

        monkeypatch.setattr(registry, "discover_models", lambda *a, **k: [_entry(max_pe)])
        monkeypatch.setattr(registry, "resolve_inputs", resolve)
        monkeypatch.setattr(capp, "_cache_flags", lambda backend: (None, None))
        monkeypatch.setattr(capp, "_kv_geometry", lambda *a, **k: None)
        monkeypatch.setattr(capp, "_own_gpu_mib", lambda: 0)
        monkeypatch.setattr(capp, "_training_marker_hits", lambda: [])
        monkeypatch.setattr(capp.gpu, "gpu_summary", lambda: None)
        monkeypatch.setattr(capp.gpu, "gpu_alive", lambda: True)
        return capp._estimate(REPO, util, ctx, agents)

    return go


# --------------------------------------------------------------------------
# The page, executed
# --------------------------------------------------------------------------
_FNS = (
    "set", "ctxLabel", "ctxTop", "ctxRange", "ctxChoice", "ctxNoteOf",
    "agentsFitNote", "renderCtx", "badgeOf", "paintPreflight", "showFindings",
    "paintEstimate", "busyPhase", "runCtxOf", "syncRunCtx", "dirtyBits",
    "paintDirty", "paintState", "wireControls", "init",
)


def _optional_fn(name: str) -> str:
    """_fn_body, or nothing when app.js has no such function -- so the harness
    also loads an app.js from before this fix and fails on the ASSERTION, with
    the wrong number in it, rather than on a missing name.

    Keeps an `async` prefix: _fn_body() starts at `function`, and init()
    without it is an `await` outside an async function."""
    if f"function {name}(" not in APP_JS:
        return ""
    body = _fn_body(name)
    return ("async " + body) if f"async function {name}(" in APP_JS else body


_MODEL = {"repo_id": REPO, "name": "Qwen3.8-27B-NVFP4", "backend": "inline",
          "servable": True, "model_max_ctx": NATIVE, "trust": "measured"}


def _page(scenario: str, *, agents: int = 16) -> dict:
    """Load the page's real control code, run init() -- which attaches the
    real slider handlers, then waits forever on /api/models -- and
    wireControls(), run `scenario`, and report what the control shows and what
    Apply POSTed.

    In a scenario, `showEstimate(d)` is what doEstimate() does with a payload
    that came back: remember it, then paint it.
    """
    ids = _element_ids()
    src = [
        *_PRELUDE,
        *[_const_line(c) for c in ("MIN_CTX", "CTX_STEP", "DEFAULT_MAX_CTX", "BUSY_PHASES")],
        *[_optional_fn(n) for n in _FNS],
        f"var MODELS = [{json.dumps(_MODEL)}], sel = 0, util = 0.95, agents = {agents};",
        "var userPicked = false, controlEnabled = true, lastEstimate = null;",
        "var liveFacts = {}, liveSizing = {};",
        "var ctxSource = 'default', ctxRun = null, lastRunSig = null;",
        "var location = {origin: 'http://127.0.0.1:8010'};",
        "var __posts = [];",
        "function post(path, body) { __posts.push({path: path, body: body}); return {}; }",
        "function confirm() { return true; }",
        "var __asked = [];",
        "function estimate() { __asked.push(ctx); }",
        "function log() {}",
        "function fetch() { return new Promise(function () {}); }",
        "function paintRequestStats() {} function paintBoot() {}",
        "function paintServingMeta() {} function renderModels() {}",
        "function showEstimate(d) { lastEstimate = d; paintEstimate(d); }",
        f"__mk({json.dumps(ids)});",
        "init();",
        # init() wires the buttons only after the model list and the event
        # stream are up, which here is never; wire them the same way it does.
        "wireControls();",
        scenario,
        """JSON.stringify((function () {
          var note = __els.ctxNote;
          return {ctx: ctx, value: __els.ctx.value, min: __els.ctx.min, max: __els.ctx.max,
                  readout: __els.ctxV.textContent, bound: __els.ctxBound.textContent,
                  note: note ? note.textContent : null, noteCls: note ? note.className : null,
                  agentsFit: __els.agentsFit.textContent, dirty: __els.dirty.className,
                  posts: __posts, asked: __asked};
        })());""",
    ]
    return _run_js("\n".join(src))


def _apply() -> str:
    return "__els.apply.onclick();"


def _state(max_model_len: int, pid: int = 64771) -> dict:
    """/api/state for the live 27B, trimmed to what the page reads."""
    return {
        "upstream": {
            "up": True, "port": 8004, "model": "Qwen3.8-27B-NVFP4", "model_id": REPO,
            "identity": {"repo_id": REPO, "served_names": ["Qwen3.8-27B-NVFP4"],
                         "source": "process", "mismatch": False},
            "max_model_len": max_model_len,
            "resolution": {"port": 8004, "pid": pid, "reason": "live"},
            "live": {"kv_tokens": POOL, "util_effective": 0.95, "ctx": max_model_len},
        },
        "supervisor": {"actual_state": "READY", "desired_state": "RUNNING"},
        "sizing": {"max_num_seqs": 16},
        "boot": {},
        "control_enabled": True,
    }


# --------------------------------------------------------------------------
# 1. The default is the model's full native context
# --------------------------------------------------------------------------
def test_apply_launches_at_the_models_full_context_not_pool_over_agents(estimate) -> None:
    """The owner's case: a 1,595,321-token pool, 16 agents, a 262,144 native
    maximum. Before the fix the estimate said the ceiling was 1,595,321 / 16 =
    99,707, the slider clamped to 98,304, and Apply posted that (the live box,
    whose estimated pool was 1,814,367, posted 110,592)."""
    est = estimate(0.95, NATIVE, 16)
    assert est["kv_tokens"] == POOL, "precondition: the live pool"

    got = _page(f"showEstimate({json.dumps(est)}); {_apply()}")
    assert [p["path"] for p in got["posts"]] == ["/api/server/start"]
    assert got["posts"][0]["body"]["ctx"] == NATIVE, got["posts"]
    # The root, in the payload the page was handed.
    assert est["ctx_max_fit"] == NATIVE, (
        f"ctx_max_fit is {est['ctx_max_fit']:,}: the pool was divided by the agents"
    )
    assert est["ctx_fit_reason"] is None
    assert got["value"] == "262144" and got["readout"] == "262,144", got
    assert got["max"] == "262144", "the full native context must be reachable"
    assert got["note"] == "", "nothing to explain at full native context"
    # The agent count is still sent -- as --max-num-seqs, not as a divisor.
    assert got["posts"][0]["body"]["max_num_seqs"] == 16
    # And the line beside it is advice about sharing the pool, not a verdict.
    assert got["agentsFit"] == "3 at 256k at once", got["agentsFit"]


def test_the_agent_count_does_not_move_the_context_at_all(estimate) -> None:
    """Over-correction guard for the same root: 1, 4 and 64 agents all launch
    at 262,144, because none of them changes how long one request may be."""
    for agents in (1, 4, 64):
        est = estimate(0.95, NATIVE, agents)
        got = _page(f"showEstimate({json.dumps(est)}); {_apply()}", agents=agents)
        assert got["posts"][0]["body"]["ctx"] == NATIVE, (agents, got["posts"])


def test_the_default_is_the_native_context_before_any_estimate() -> None:
    """With no estimate back yet the only bound known is the model's own, and
    that is what the control rests on -- never a placeholder below it."""
    got = _page(_apply())
    assert got["value"] == "262144" and got["posts"][0]["body"]["ctx"] == NATIVE, got


# --------------------------------------------------------------------------
# 2. Lowered only when one request of the native length cannot fit
# --------------------------------------------------------------------------
def test_a_pool_too_small_for_one_full_request_lowers_it_and_says_why(estimate) -> None:
    """Util 0.30 buys (0.30 x 95.5928 - 20.82 - 4.33) = 3.5278 GiB, which at
    the live rate (43.159 KiB/token) is 85,710 tokens: one 262,144-token request
    cannot fit. The launch drops to the longest one request CAN use, snapped
    down onto the slider grid (85,710 -> 81,920), and the page prints why.

    Before the fix the same budget was divided by 16 agents, 5,356, and the
    slider bottomed out at 8,192 with nothing on screen explaining it."""
    est = estimate(0.30, NATIVE, 16)
    assert est["kv_tokens"] == 85_710, est["kv_tokens"]
    assert est["ctx_max_fit"] == 85_710, est["ctx_max_fit"]
    assert est["can_apply"] is False, "precondition: 262,144 itself is blocked"

    # The page drops to what fits, and asks for an estimate AT that length:
    # the one above describes 262,144 and its block would refuse the Apply.
    at_fit = estimate(0.30, 81_920, 16)
    assert at_fit["can_apply"] is True, at_fit["findings"]
    got = _page(f"showEstimate({json.dumps(est)});"
                f"showEstimate({json.dumps(at_fit)}); {_apply()}")
    assert got["asked"] == [81_920], f"the page re-estimated at {got['asked']}"
    assert got["posts"][0]["body"]["ctx"] == 81_920, got["posts"]
    assert got["value"] == "81920" and got["max"] == "81920", got
    note = got["note"] or ""
    for part in ("262,144", "does not fit", "0.30", "85,710", "Raise GPU utilization"):
        assert part in note, (part, note)
    assert "on" in (got["noteCls"] or "").split() and "warn" in got["noteCls"].split(), (
        f"the reason is not visible: {got['noteCls']!r}"
    )
    assert got["bound"] == "KV fits 84k", got["bound"]


def test_the_longest_single_request_accounts_for_the_per_sequence_state(estimate) -> None:
    """A hybrid model's KV rate rises as the context shrinks (the per-sequence
    recurrent state is spread over fewer tokens), so "the pool at 262,144" is
    NOT a length one request can use. Here: 8 GiB of KV, 36,000 B/token of
    attention, a 1 GiB state per sequence. One request of L needs 36,000 L +
    2^30 bytes, so the longest that fits is 7 x 2^30 / 36,000 = 208,783.

    Checked by its defining property through the real estimate: at the
    default, one request fits (no KV_TOO_SMALL_FOR_ONE_CTX); one token more
    and it does not. Before the fix the default was the pool at 262,144
    (~211,500) -- a launch the engine refuses."""
    weights = 0.5 * capacity.GPU_TOTAL_GIB - OVERHEAD - 8.0     # leaves exactly 8 GiB
    rate = lambda length: (36_000 + 2**30 / length) / 1024      # noqa: E731 - KiB/token

    def blocked(ctx: int) -> bool:
        e = estimate(0.5, ctx, 1, rate=rate, weights=weights)
        return any(f["code"] == "KV_TOO_SMALL_FOR_ONE_CTX" for f in e["findings"])

    fit = estimate(0.5, NATIVE, 1, rate=rate, weights=weights)["ctx_max_fit"]
    assert 208_782 <= fit <= 208_783, fit   # float flooring of the rate; see docstring
    assert not blocked(fit), "the default is a length one request cannot use"
    assert blocked(fit + 1), "the default is not the LONGEST length that fits"


# --------------------------------------------------------------------------
# 3. The operator can still lower it, deliberately
# --------------------------------------------------------------------------
def test_an_operator_lowered_context_is_honoured(estimate) -> None:
    """Trading context for concurrency is a legitimate choice, and only an
    explicit one. 200,704 is below the native 262,144 but above the old
    pool / agents cap, which is why the page used to pull it back to 98,304."""
    est = estimate(0.95, NATIVE, 16)
    after = estimate(0.95, 200_704, 16)       # the estimate the input triggers
    got = _page(
        f"showEstimate({json.dumps(est)});"
        "__els.ctx.oninput({target: {value: '200704'}});"
        f"showEstimate({json.dumps(after)}); {_apply()}"
    )
    assert got["posts"][0]["body"]["ctx"] == 200_704, got["posts"]
    assert got["readout"] == "200,704", got
    assert got["note"].startswith("Lowered by hand to 200,704"), got["note"]
    assert "warn" not in (got["noteCls"] or ""), "the operator's own choice is not a warning"


def test_an_operator_value_still_cannot_exceed_one_request(estimate) -> None:
    """Over-correction guard: honouring the operator never means offering a
    length no single request can hold. At util 0.30 the ceiling is 81,920."""
    est = estimate(0.30, 131_072, 16)
    at_fit = estimate(0.30, 81_920, 16)
    got = _page(
        f"showEstimate({json.dumps(at_fit)});"
        "__els.ctx.oninput({target: {value: '131072'}});"
        f"showEstimate({json.dumps(est)}); showEstimate({json.dumps(at_fit)}); {_apply()}"
    )
    assert got["posts"][0]["body"]["ctx"] == 81_920, got["posts"]


# --------------------------------------------------------------------------
# 4. The page shows what the server runs
# --------------------------------------------------------------------------
def test_the_control_reads_back_the_running_servers_max_model_len(estimate) -> None:
    """The control used to keep whatever the slider said (an earlier review
    saw 32,768 on screen beside a server running 262,144), and a later Apply
    silently sent it. It now reads /api/state's upstream.max_model_len -- the
    live process's own --max-model-len -- whenever the selected model is the
    one serving, and says when that is below the model's own."""
    est = estimate(0.95, NATIVE, 16)
    got = _page(
        f"showEstimate({json.dumps(est)});"
        f"lastState = {json.dumps(_state(110_592))}; paintState(lastState); {_apply()}"
    )
    assert got["value"] == "110592" and got["readout"] == "110,592", got
    assert got["posts"][0]["path"] == "/api/server/restart"
    assert got["posts"][0]["body"]["ctx"] == 110_592, got["posts"]
    assert "running server was started at 110,592" in got["note"], got["note"]
    assert "262,144" in got["note"] and "warn" in got["noteCls"].split(), got
    assert "on" not in got["dirty"].split(), "the page and the server agree"


def test_a_new_server_is_read_back_but_an_operator_pick_survives_the_same_one(
    estimate,
) -> None:
    """After a start the control must show what came up, not the slider value
    that was sent: a launcher that capped it, or a start from another client,
    would otherwise leave a stale number for the next Apply. But the SAME
    server's next state event must not undo a pick the operator just made."""
    est = estimate(0.95, NATIVE, 16)
    got = _page(
        f"showEstimate({json.dumps(est)});"
        f"paintState({json.dumps(_state(110_592, pid=1))});"
        "__els.ctx.oninput({target: {value: '200704'}});"
        f"paintState({json.dumps(_state(110_592, pid=1))});"
        "var __kept = ctx;"
        f"paintState({json.dumps(_state(196_608, pid=2))});"
        f"lastState = {json.dumps(_state(196_608, pid=2))}; {_apply()}"
        "__posts.push({kept: __kept});"
    )
    assert got["posts"][-1] == {"kept": 200_704}, (
        "the same server's state event overwrote the operator's pick"
    )
    assert got["value"] == "196608", got
    assert got["posts"][0]["body"]["ctx"] == 196_608, got["posts"]


def test_a_running_length_is_not_carried_into_a_budget_that_cannot_hold_it(
    estimate,
) -> None:
    """The read-back describes the running server at ITS utilization. Planning
    a relaunch at util 0.30 is a different KV budget, one that cannot hold a
    single 262,144-token request, so the default rules again: the control drops
    to what fits and says why. Found by driving the real page against the live
    27B: the control stayed at the running 262,144 while the estimate said
    85,710, and Apply was refused by the stale length's block instead."""
    at95 = estimate(0.95, NATIVE, 16)
    at30 = estimate(0.30, NATIVE, 16)
    at30_fit = estimate(0.30, 81_920, 16)
    got = _page(
        f"showEstimate({json.dumps(at95)});"
        f"lastState = {json.dumps(_state(NATIVE))}; paintState(lastState);"
        "var __before = ctx;"
        "__els.util.oninput({target: {value: '30'}});"
        f"showEstimate({json.dumps(at30)}); showEstimate({json.dumps(at30_fit)});"
        f"{_apply()} __posts.push({{before: __before}});"
    )
    assert got["posts"][-1] == {"before": NATIVE}, "precondition: the read-back"
    assert got["posts"][0]["body"]["ctx"] == 81_920, got["posts"]
    assert got["posts"][0]["body"]["util"] == 0.30
    assert "does not fit" in got["note"] and "85,710" in got["note"], got["note"]

    # Back at the running server's own utilization, it reads back again.
    back = _page(
        f"showEstimate({json.dumps(at95)});"
        f"lastState = {json.dumps(_state(NATIVE))}; paintState(lastState);"
        "__els.util.oninput({target: {value: '30'}});"
        f"showEstimate({json.dumps(at30)});"
        "__els.util.oninput({target: {value: '95'}});"
        f"showEstimate({json.dumps(at95)}); {_apply()}"
    )
    assert back["posts"][0]["body"]["ctx"] == NATIVE, back["posts"]


def test_selecting_a_model_that_is_not_serving_uses_its_own_default(estimate) -> None:
    """Over-correction guard: the read-back belongs to the model that is
    running. Another model on the card list is a different launch and starts
    from ITS native context."""
    est = estimate(0.95, NATIVE, 16)
    other = dict(_state(110_592))
    other["upstream"] = dict(other["upstream"], model_id="Other/Model",
                             identity={"repo_id": "Other/Model", "served_names": ["x"]})
    got = _page(f"showEstimate({json.dumps(est)}); paintState({json.dumps(other)});"
                f"{_apply()}")
    assert got["posts"][0]["body"]["ctx"] == NATIVE, got["posts"]


# --------------------------------------------------------------------------
# 5. Long and short requests sharing the live pool
# --------------------------------------------------------------------------
def _mix_panel(sizing: dict) -> dict:
    """paintRequestStats() -- which paints the mixed view -- on a real sizing
    payload, returning the head line, the table's cell text and the note."""
    ids = _element_ids()
    helpers = ("bytesTxt", "secsTxt", "agoTxt", "uptimeTxt", "durTxt", "ctxLabel")
    src = [
        *_PRELUDE,
        _const_line("BUSY_PHASES"),
        *[_fn_body(n) for n in (*helpers, "set", "winSpan", "histData", "reqHist")],
        *[_optional_fn(n) for n in ("mixModel", "paintMix")],
        _fn_body("paintRequestStats"),
        f"__mk({json.dumps(ids)});",
        f"var liveSizing = {json.dumps(sizing)};",
        "paintRequestStats();",
        """JSON.stringify((function () {
          var t = __els.mixTbl, rows = [];
          for (var i = 0; t && i < t.children.length; i++) {
            var tr = t.children[i], cells = [];
            for (var j = 0; j < tr.children.length; j++) cells.push(tr.children[j].textContent);
            rows.push(cells);
          }
          return {head: __els.mixFull ? __els.mixFull.textContent : null,
                  title: __els.mixFull ? __els.mixFull.title : null,
                  note: __els.mixNote ? __els.mixNote.textContent : null, rows: rows,
                  rec: __els.recN.textContent};
        })());""",
    ]
    return _run_js("\n".join(src))


@pytest.fixture
def sizing(monkeypatch: pytest.MonkeyPatch):
    """The real _sizing_payload() on a scripted scrape and command line."""
    def go(window: dict, *, max_model_len: int | None = NATIVE, max_num_seqs=None) -> dict:
        monkeypatch.setattr(capp.rt, "metrics", {
            "reachable": True, "kv_cache_size_tokens": POOL, "running": 3,
            "preemptions": 0, "prompt_stats": window,
        }, raising=False)
        monkeypatch.setattr(capp.rt, "serving_model", "Qwen3.8-27B-NVFP4", raising=False)
        monkeypatch.setattr(capp, "_running_max_num_seqs", lambda: max_num_seqs)
        monkeypatch.setattr(capp, "_running_max_model_len", lambda: max_model_len)
        return capp._sizing_payload()
    return go


def _window(p50: dict, p90: dict) -> dict:
    return {"n": 100, "capacity": 100, "exact_n": 100, "age_s": 60.0,
            "p50": p50, "p90": p90, "p99": p90, "max": p90, "partial": False}


def test_the_live_panel_shows_long_and_short_requests_together(sizing) -> None:
    """"Only 2-3 primary agents have large context and rest smaller": the
    panel answers how many full-length requests fit at once and how many
    smaller ones fit beside 1, 2 or 3 of them, on the LIVE pool and the
    calibrated per-request cost. Numbers worked by hand in
    tests/test_parallelism.py::test_mixed_capacity_reproduces_the_hand_computed_split."""
    exact = lambda v: {"lo": v, "hi": v, "exact": True}   # noqa: E731
    got = _mix_panel(sizing(_window(exact(8_102), exact(30_116))))
    assert got["head"] == "3 at 262,144 at once", got
    assert got["rows"] == [
        ["long", "+ p50 8,102", "+ p90 30,116"],
        ["1 at 262,144", "+ 35", "+ 16"],
        ["2 at 262,144", "+ 20", "+ 9"],
        ["3 at 262,144", "+ 5", "+ 2"],
    ], got["rows"]
    for part in ("1,595,321", "445,115", "0.94", "12,772"):
        assert part in got["title"], (part, got["title"])
    # The recommendation beside it stays: advice, never a cap on the context.
    assert got["rec"] == "22", got["rec"]


def test_the_live_panel_marks_a_bucket_edge_and_the_scheduler_cap(sizing) -> None:
    """The stress-run window: p50 and p90 both known only to the 100,000
    bucket, so the sizes are written "<=". Beside three full-length requests
    nothing that size fits, and --max-num-seqs is named when it cuts a row."""
    edge = {"lo": 50_001, "hi": 100_000, "exact": False}
    got = _mix_panel(sizing(_window(edge, edge)))
    assert got["rows"][0] == ["long", "+ p50 ≤ 100,000", "+ p90 ≤ 100,000"]
    assert [r[2] for r in got["rows"][1:]] == ["+ 5", "+ 3", "+ 0"], got["rows"]

    small = {"lo": 8_102, "hi": 8_102, "exact": True}
    capped = _mix_panel(sizing(_window(small, small), max_num_seqs=16))
    assert capped["rows"][1] == ["1 at 262,144", "+ 15", "+ 15"], capped["rows"]
    assert "--max-num-seqs 16" in capped["note"], capped["note"]


def test_the_live_panel_says_why_there_is_no_split(sizing) -> None:
    """No --max-model-len on the running command line: the panel says so
    rather than splitting the pool at an assumed length."""
    exact = {"lo": 8_102, "hi": 8_102, "exact": True}
    got = _mix_panel(sizing(_window(exact, exact), max_model_len=None))
    assert "--max-model-len" in got["head"], got
    assert got["rows"] == []


# --------------------------------------------------------------------------
# Contracts: the page reads only what the backend sends
# --------------------------------------------------------------------------
def test_the_context_control_reads_only_fields_the_estimate_emits(estimate) -> None:
    est = estimate(0.95, NATIVE, 16)
    reads = _reads_of("d", _fn_body("renderCtx"))
    assert reads <= set(est), f"renderCtx reads fields _estimate never sends: {reads - set(est)}"
    full = _reads_of("full", _fn_body("agentsFitNote"))
    assert full <= set(est["full_at_once"]), full - set(est["full_at_once"])


def test_the_mixed_view_reads_only_fields_the_payload_emits(sizing) -> None:
    exact = {"lo": 8_102, "hi": 8_102, "exact": True}
    sz = sizing(_window(exact, exact))
    mx = sz["mixed"]
    body = _fn_body("mixModel") + _fn_body("paintMix")
    assert _reads_of("mx", body) <= set(mx), _reads_of("mx", body) - set(mx)
    assert _reads_of("z", body) <= set(mx["sizes"][0]), _reads_of("z", body)
    assert _reads_of("r", body) <= set(mx["rows"][0]), _reads_of("r", body)
    assert _reads_of("sz", body) <= set(sz), _reads_of("sz", body) - set(sz)
    for el in ("mixFull", "mixTbl", "mixNote", "ctxNote"):
        assert f'id="{el}"' in INDEX_HTML, f"#{el} is painted but not in the page"
