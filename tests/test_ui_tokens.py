"""The token strip on the page: input (prompt) tokens, output (generated)
tokens, and the prefix-cache share of the input.

Same method as tests/test_ui.py's rendering tests, and for the same reason: a
grep assertion survives any behaviour mutation, so these EXECUTE web/app.js's
functions in duktape against the DOM shim and assert on the text the painter
leaves in each element. The payloads come from the real MetricsPoller scraping
exposition text recorded off live vLLM servers -- not from dicts written here --
so the page is tested against the shapes the backend actually sends.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Any

import pytest

from servedeck import metrics, reqstats, tokens

from .test_tokens import (
    _FLASHNEXT_START,
    _QWEN27B,
    _QWEN27B_RESTARTED,
    _QWEN27B_START,
    _RESTART,
    _WIN_DT,
    _WIN_T1,
    _WIN_T2,
    _WINDOWS,
    _FIXTURES,
    _Client,
    _bump,
    _drop,
)
from .test_ui import APP_JS, INDEX_HTML, _PRELUDE, _fn_body, _rendered_text, _run_js

pytest.importorskip("dukpy", reason="pip install -e '.[dev]' brings in the JS engine")


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------
def _strip_markup() -> str:
    """index.html's token strip, from its opening div to the phases block."""
    start = INDEX_HTML.index('id="tok"')
    return INDEX_HTML[start: INDEX_HTML.index('class="phases"', start)]


def _strip_ids() -> list[str]:
    return re.findall(r'id="(tk\w+)"', _strip_markup())


def _static_text() -> dict[str, str]:
    """Each strip element's shipped text, tags stripped: what a visitor sees
    before the painter runs, and what the shim is seeded with so a painter
    that overwrote a LABEL would be caught."""
    out = {}
    for m in re.finditer(r'id="(tk\w+)"[^>]*>(.*?)</div>', _strip_markup(), re.S):
        out[m.group(1)] = re.sub(r"<[^>]+>", "", m.group(2)).strip()
    return out


_LABELS = {"tkInK": "Input (prompt)", "tkOutK": "Output (generated)",
           "tkCacheK": "Input from prefix cache"}
_VALUE_IDS = ("tkIn", "tkOut", "tkCache")
#: Every line the painter writes. Labels are markup and must NOT be in here.
_LINE_IDS = ("tkIn", "tkInS", "tkInW", "tkInR", "tkOut", "tkOutS", "tkOutW",
             "tkOutR", "tkCache", "tkCacheS", "tkCacheW")

_TOKEN_FNS = ("durTxt", "rateTxt", "winSpan", "pctTxt", "shareTxt", "spanTxt",
              "tokWindowTxt", "perRequestTxt", "tokenCell", "cacheCell",
              "paintTokens")


def _js_fns(names: tuple[str, ...]) -> list[str]:
    return [*_PRELUDE, _fn_body("set"), *[_fn_body(n) for n in dict.fromkeys(names)]]


def _paint(payload: dict) -> dict[str, dict]:
    """Run paintTokens() on a metrics payload; return every strip element."""
    ids = _strip_ids()
    seed = _static_text()
    src = [
        *_js_fns(_TOKEN_FNS),
        f"__mk({json.dumps(ids)});",
        *[f"__els[{json.dumps(k)}].textContent = {json.dumps(v)};" for k, v in seed.items()],
        f"liveMetrics = {json.dumps(payload)};",
        "paintTokens();",
        f"__dump({json.dumps(ids)});",
    ]
    return _run_js("\n".join(src))


def _snaps(script: list[Any], clock: list[float], wall: float = 0.0) -> list[dict]:
    """Full metrics payloads (what /api/events carries as `vllm`) from the
    real poller over recorded exposition text."""
    ticks = iter(clock)
    poller = metrics.MetricsPoller(
        "http://127.0.0.1:8001", monotonic=lambda: next(ticks), wall=lambda: wall
    )
    client = _Client(script)

    async def run() -> list[dict]:
        return [(await poller.scrape(client)).to_dict()  # type: ignore[arg-type]
                for _ in range(len(script))]

    return asyncio.run(run())


def _last(script: list[Any], clock: list[float], wall: float = 0.0) -> dict:
    return _snaps(script, clock, wall)[-1]


_PREFILL_LAG = _WINDOWS["prefill_lag"]
_LAG_T1 = (_FIXTURES / _PREFILL_LAG["t1"]).read_text()
_LAG_T2 = (_FIXTURES / _PREFILL_LAG["t2"]).read_text()


def _busy() -> dict:
    """A recorded 10.016 s interval in which one request finished (so the
    request windows hold one exact observation), with one agent turn's prompt
    added -- 30,000 input tokens, 26,624 of them from the cache."""
    t2 = _bump(_LAG_T2, metrics.PROMPT_TOK_TOTAL, 30_000)
    t2 = _bump(t2, metrics.PROMPT_TOK_CACHED_TOTAL, 26_624)
    return _last([_LAG_T1, t2], [0.0, float(_PREFILL_LAG["dt_s"])],
                 wall=_FLASHNEXT_START + 11_000)


def _fresh() -> str:
    """A server that has served nothing yet: the 27B's first scrape after the
    restart recorded live, every token counter 0.0."""
    return _QWEN27B_RESTARTED


def _states() -> dict[str, dict]:
    old_build = _drop(_drop(_WIN_T2, metrics.PROMPT_TOK_CACHED_TOTAL), metrics.PROCESS_START)
    return {
        "busy": _busy(),
        "decoding, no new input": _last([_WIN_T1, _WIN_T2], [0.0, _WIN_DT],
                                        wall=_FLASHNEXT_START + 11_520),
        "idle": _last([_WIN_T2, _WIN_T2], [0.0, 60.0], wall=_FLASHNEXT_START + 11_520),
        "first poll": _last([_WIN_T2], [0.0], wall=_FLASHNEXT_START + 11_520),
        "restart / switch of model": _last([_WIN_T2, _QWEN27B], [0.0, 2.0],
                                           wall=_QWEN27B_START + 300),
        "fresh server, nothing served": _last([_fresh(), _fresh()], [0.0, 2.0],
                                              wall=_RESTART["after"]["wall_s"]),
        "nothing cached yet (live 27B)": _last([_QWEN27B], [0.0], wall=_QWEN27B_START + 300),
        "older build": _last([old_build], [0.0]),
        "nothing serving": metrics.unreachable_snapshot(),
        "backend went away": _last([_WIN_T2, ConnectionError("refused")], [0.0, 2.0]),
    }


_BAD = ("NaN", "Infinity", "undefined", "null", "[object")


# --------------------------------------------------------------------------
# Every state: both cells labelled, every line a reading or a reason
# --------------------------------------------------------------------------
def test_input_and_output_render_with_their_labels_in_every_server_state() -> None:
    """The cells are never blank, never a zero standing in for "unknown", and
    never lose the words that say which figure they are -- including when the
    server is idle, has just restarted, and when nothing is serving at all."""
    seeded = _static_text()
    for name, payload in _states().items():
        dom = _paint(payload)
        for lid, words in _LABELS.items():
            assert words in dom[lid]["text"], f"{name}: {lid} lost its label: {dom[lid]}"
            assert dom[lid]["text"] == seeded[lid], f"{name}: the painter wrote a label"
        for lid in _LINE_IDS:
            text = dom[lid]["text"]
            assert text.strip(), f"{name}: #{lid} is blank"
            assert text != "—", f"{name}: #{lid} was never painted"
            for bad in _BAD:
                assert bad not in text, f"{name}: #{lid} renders {bad!r}: {text!r}"
            assert not re.search(r"(?<![\w])-\d", text), f"{name}: negative figure in #{lid}: {text!r}"
        for vid in ("tkIn", "tkOut"):
            v = dom[vid]["text"]
            assert v == "n/a" or re.fullmatch(r"\d{1,3}(,\d{3})* tokens", v), f"{name}: {vid}={v!r}"
            assert ("na" in dom[vid]["cls"].split()) == (v == "n/a"), f"{name}: {vid} class"
        c = dom["tkCache"]["text"]
        assert c == "n/a" or re.fullmatch(r"\d{1,3}\.\d% of input", c), f"{name}: tkCache={c!r}"


def test_the_recorded_window_renders_as_a_delta_over_its_span() -> None:
    """The recorded 60.007 s Flash-Next pair, exactly as the page shows it."""
    dom = _paint(_states()["decoding, no new input"])
    text = {k: v["text"] for k, v in dom.items()}
    assert text["tkOut"] == "6,461,615 tokens"
    assert text["tkOutS"] == "server started 3h 12m ago"
    assert text["tkOutW"] == "last 60 s: 14,940 tokens · 249.0 tok/s"
    assert text["tkIn"] == "90,749,177 tokens"
    assert text["tkInW"] == "last 60 s: idle — no input tokens"
    assert text["tkCache"] == "72.0% of input"
    assert text["tkCacheS"] == "since server start: 65,366,496 cached · 25,382,681 computed"
    assert text["tkCacheW"] == "last 60 s: idle — no input tokens"


def test_an_input_window_names_its_rate_and_its_cache_split() -> None:
    dom = _paint(_busy())
    assert dom["tkInW"]["text"] == "last 10 s: 30,000 tokens · 2,995.2 tok/s"
    assert dom["tkCacheW"]["text"] == "last 10 s: 88.8% · 26,624 cached · 3,376 computed"
    # One request finished in the interval, so its size is exact.
    assert dom["tkInR"]["text"] == (
        "per request, last 1/100: p50 69,134 · p90 69,134 · max 69,134"
    )
    assert dom["tkOutR"]["text"] == (
        "per request, last 1/100: p50 5,231 · p90 5,231 · max 5,231"
    )


def test_a_restart_shows_the_new_servers_totals_and_start() -> None:
    """Flash-Next at 90.7M input tokens, then the 27B at 39,529: the strip must
    show the 27B's own totals, since the 27B's own start, and say the window
    started again -- never the old totals, a sum, or a negative rate."""
    dom = _paint(_states()["restart / switch of model"])
    assert dom["tkIn"]["text"] == "39,529 tokens"
    assert dom["tkOut"]["text"] == "4,111 tokens"
    assert dom["tkInS"]["text"] == "server started 5m 00s ago"
    assert dom["tkInW"]["text"] == "window: " + tokens.RESTARTED
    assert dom["tkOutW"]["text"] == "window: " + tokens.RESTARTED
    assert "90,749,177" not in json.dumps(dom)


def test_nothing_serving_says_why_in_every_line() -> None:
    dom = _paint(metrics.unreachable_snapshot())
    for vid in _VALUE_IDS:
        assert dom[vid]["text"] == "n/a" and "na" in dom[vid]["cls"]
    assert dom["tkInS"]["text"] == tokens.UNREACHABLE
    assert dom["tkInW"]["text"] == "window: " + tokens.UNREACHABLE
    assert dom["tkInR"]["text"] == "per request: " + tokens.UNREACHABLE
    assert dom["tkCacheS"]["text"] == tokens.UNREACHABLE


# --------------------------------------------------------------------------
# The cached share: zero cached, zero input, non-numbers
# --------------------------------------------------------------------------
def test_zero_cached_renders_zero_percent_and_zero_input_renders_a_reason() -> None:
    zero = _paint(_states()["nothing cached yet (live 27B)"])
    assert zero["tkCache"]["text"] == "0.0% of input"
    assert zero["tkCacheS"]["text"] == "since server start: 0 cached · 39,529 computed"

    empty = _paint(_states()["fresh server, nothing served"])
    assert empty["tkCache"]["text"] == "n/a"
    assert empty["tkCacheS"]["text"] == tokens.NO_INPUT_YET
    assert empty["tkCacheW"]["text"] == "last 2 s: idle — no input tokens"
    assert empty["tkIn"]["text"] == "0 tokens", "zero input is a real total, not n/a"


def _cell(expr: str) -> Any:
    return _run_js("\n".join(_js_fns(_TOKEN_FNS) + [f"JSON.stringify({expr});"]))


def test_a_share_that_is_not_a_finite_number_is_never_drawn() -> None:
    """Belt and braces for the backend's guard: whatever reaches the page, a
    null, a NaN or an Infinity prints the reason -- never "0.0%" (which is what
    `(null * 100).toFixed(1)` gives) and never "NaN%"."""
    for share in ("null", "NaN", "Infinity", "undefined"):
        got = _cell(f"cacheCell({{cached: {{share: {share}, share_reason: 'why'}}}})")
        assert got["value"] == "n/a", (share, got)
        assert got["since"] == "why", (share, got)
    assert _cell("shareTxt(0)") == "0.0%"
    assert _cell("shareTxt(1)") == "100.0%"
    assert _cell("shareTxt(0.8875)") == "88.8%"


def test_a_rate_with_decimals_is_grouped_like_every_other_number() -> None:
    """rateTxt(v, 1) was a bare toFixed(): an input rate printed as
    "2995.2 tok/s" on a strip whose totals read "90,749,177". The integer part
    is grouped now; the decimals, and every rate below 1,000, are unchanged."""
    assert _cell("rateTxt(2995.24, 1)") == "2,995.2 tok/s"
    assert _cell("rateTxt(1234567.25, 1)") == "1,234,567.3 tok/s"
    # Over-correction guard: the strip above already renders these.
    assert _cell("rateTxt(249, 1)") == "249.0 tok/s"
    assert _cell("rateTxt(43.66, 1)") == "43.7 tok/s"
    assert _cell("rateTxt(1240)") == "1,240 tok/s"
    assert _cell("rateTxt(null, 1)") == "—"
    # The window's span, in the unit the window is sized in.
    assert _cell("spanTxt(60.007)") == "60 s"
    assert _cell("spanTxt(61.6)") == "62 s"
    assert _cell("spanTxt(null)") == "—"


# --------------------------------------------------------------------------
# Per-request figures keep their bucket provenance
# --------------------------------------------------------------------------
def _window(*scrapes: tuple[list[tuple[float, float]], float, float]) -> dict:
    win = reqstats.RequestWindow()
    for buckets, s, n in scrapes:
        win.observe(buckets, hist_sum=s, hist_count=n)
    return win.stats().to_dict()


_EDGES = [5000.0, 20000.0, 50000.0, 200000.0, math.inf]


def test_the_per_request_line_says_how_much_is_only_bucket_bounded() -> None:
    zero = [(e, 0.0) for e in _EDGES]
    three = [(5000.0, 0.0), (20000.0, 0.0), (50000.0, 3.0), (200000.0, 3.0), (math.inf, 3.0)]
    w = _window((zero, 0.0, 0.0), (three, 99_000.0, 3.0))
    line = _cell(f"perRequestTxt({json.dumps(w)}, {{}}, true)")
    assert line == (
        "per request, last 3/100 (3 bucket-bounded): "
        "p50 20,001–50,000 · p90 20,001–50,000 · max 20,001–50,000"
    ), line

    one = [(5000.0, 0.0), (20000.0, 0.0), (50000.0, 1.0), (200000.0, 1.0), (math.inf, 1.0)]
    exact = _window((zero, 0.0, 0.0), (one, 31_234.0, 1.0))
    assert _cell(f"perRequestTxt({json.dumps(exact)}, {{}}, true)") == (
        "per request, last 1/100: p50 31,234 · p90 31,234 · max 31,234"
    )
    empty = json.dumps(reqstats.EMPTY_STATS)
    assert _cell(f"perRequestTxt({empty}, {{}}, true)") == "per request: 0/100 requests observed"
    # The open top bucket has no upper edge; it is a lower bound, not a number.
    assert _cell('pctTxt({lo: 200001, hi: null, exact: false})') == "> 200,001"


# --------------------------------------------------------------------------
# Wiring and contracts
# --------------------------------------------------------------------------
def test_telemetry_paints_the_token_strip() -> None:
    """paintTokens() is tested above; this proves the page calls it. Executed:
    the whole telemetry painter runs against the shim (its canvases return
    early without a 2D context), and the strip must come out painted."""
    ids = sorted(set(re.findall(r'id="([A-Za-z0-9_]+)"', INDEX_HTML)))
    fns = ("agoTxt", "secsTxt", "rateTxt", "windowFigure", "lifeTxt", "streamLifeTxt",
           "uptimeTxt",
           "busyPhase", "resolutionNote", "resolutionDetail", "paintThroughput",
           "paintServingMeta", "winSpan", "durTxt", "histData", "markRows",
           "reqHist", "mixModel", "paintMix", "paintRequestStats", "spark",
           "paintTelemetry", *_TOKEN_FNS)
    payload = _states()["decoding, no new input"]
    src = [
        *_js_fns(fns),
        "var hist = [], liveSizing = {}, liveFacts = {};",
        f"__mk({json.dumps(ids)});",
        f"paintTelemetry({{vllm: {json.dumps(payload)}, gpu: {{}}, sizing: null}});",
        '__dump(["tkIn", "tkOutW", "tkCache"]);',
    ]
    dom = _run_js("\n".join(src))
    assert dom["tkIn"]["text"] == "90,749,177 tokens", dom
    assert dom["tkOutW"]["text"] == "last 60 s: 14,940 tokens · 249.0 tok/s", dom
    assert dom["tkCache"]["text"] == "72.0% of input", dom


def test_the_strip_ships_no_fabricated_readings() -> None:
    """Until the painter runs, a shipped digit in a value slot would be a
    reading nothing produced (web/app.js: "No simulated data")."""
    static = _static_text()
    for lid in _LINE_IDS:
        assert lid in static, f"#{lid} is missing from index.html"
        assert not re.search(r"\d", static[lid]), f"#{lid} ships {static[lid]!r}"


def _payload_keys() -> tuple[set[str], set[str], set[str]]:
    live = _states()["busy"]["tokens"]
    return set(live), set(live["input"]), set(live["cached"])


def _reads(obj: str, src: str) -> set[str]:
    return set(re.findall(rf"\b{re.escape(obj)}\.([A-Za-z_]\w*)", _rendered_text(src)))


def test_the_page_reads_only_token_fields_the_backend_emits() -> None:
    """Key for key against the real serialiser, as test_ui.py does for every
    other payload: a field the backend never sends renders "undefined"."""
    top, counter, cache = _payload_keys()
    body = "\n".join(_fn_body(n) for n in ("tokWindowTxt", "tokenCell", "cacheCell"))
    assert _reads("t", body) <= top, _reads("t", body) - top
    assert _reads("f", body) <= counter | cache, _reads("f", body) - (counter | cache)
    assert _reads("g", body) <= counter | cache, _reads("g", body) - (counter | cache)
    assert _reads("c", _fn_body("cacheCell")) <= cache, _reads("c", _fn_body("cacheCell")) - cache
    win = set(reqstats.EMPTY_STATS)
    assert _reads("w", _fn_body("perRequestTxt")) <= win
    assert "tokens" in metrics.MetricsSnapshot().to_dict()
    assert {"tokens", "prompt_stats", "gen_stats"} <= _reads("m", _fn_body("paintTokens"))


def test_the_strip_switches_on_state_codes_not_on_prose() -> None:
    """Rewording a reason in Python must not change what the page does."""
    codes = set(metrics.REASON_CODE.values()) | {"ok"}
    for fn in ("tokWindowTxt", "tokenCell", "cacheCell", "perRequestTxt"):
        for lit in re.findall(r'state === "([^"]+)"', _fn_body(fn)):
            assert lit in codes, f"{fn} compares against {lit!r}, not a state code"
    for prose in (tokens.UNREACHABLE, tokens.RESTARTED, tokens.NO_INPUT_YET,
                  tokens.NOT_EXPOSED, tokens.NO_BASELINE):
        assert prose not in APP_JS, f"the page hardcodes backend prose {prose!r}"
