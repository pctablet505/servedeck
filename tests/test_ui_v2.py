"""What the v2 page actually renders, run in a real JS engine.

The v1 suite this replaces was mostly ``assert "prefill" in source``. Those
assertions survive every behavioural mutation of the code they claim to cover,
which is how the page shipped ``${liveMetrics.gen_tok_s || 0} tok/s`` -- an
idle engine and a stalled one both printed "0 tok/s" and no test noticed.

So the rendering tests here EXECUTE ``web/app.js``'s pure helpers in duktape
(via dukpy) against payloads produced by the real backend, and assert on the
strings that come back. The pure helpers are a contiguous, marker-delimited
block at the top of ``app.js`` precisely so this file can lift them verbatim:
anything that touches ``document`` lives below the marker and is not loaded
here, which is the separation that makes these tests behavioural.

The remaining tests are contracts that no execution can check:

* every ``m.<field>`` the page reads is a key ``app.build_state`` emits -- built
  by CALLING build_state on a real registry, never by hand;
* the page does no capacity arithmetic of its own (the constants and the
  division are both absent), because a second formula here would eventually
  disagree with ``parallelism.py`` and the screen would contradict itself;
* every element id the painters write to exists in ``index.html``, and every id
  in the HTML is one a painter writes -- a painter writing to a missing element
  fails silently, and an element nobody writes is a permanently stale label.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

import httpx
import pytest

from servedeck import app as _app
from servedeck import metrics as _metrics
from servedeck import models as _models
from servedeck import parallelism as _parallelism
from servedeck import settings as _settings

#: The assets ship inside the package, so this is app.WEB, never a guess at
#: the project root.
APP_JS = (_app.WEB / "app.js").read_text()
INDEX_HTML = (_app.WEB / "index.html").read_text()

EM = "—"
MAX_APP_JS_LINES = 900

_BEGIN = "/* ===== PURE HELPERS BEGIN ===== */"
_END = "/* ===== PURE HELPERS END ===== */"


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------
def _pure_block() -> str:
    """Every pure helper, lifted verbatim out of app.js."""
    assert _BEGIN in APP_JS and _END in APP_JS, "app.js lost its pure-helper markers"
    start = APP_JS.index(_BEGIN) + len(_BEGIN)
    return APP_JS[start : APP_JS.index(_END)]


def _fn_body(name: str) -> str:
    """The source of a top-level ``function name(...) { ... }``, by brace match."""
    start = APP_JS.index(f"function {name}(")
    depth, i = 0, APP_JS.index("{", start)
    for j in range(i, len(APP_JS)):
        if APP_JS[j] == "{":
            depth += 1
        elif APP_JS[j] == "}":
            depth -= 1
            if depth == 0:
                return APP_JS[start : j + 1]
    raise AssertionError(f"unbalanced braces in {name}()")


#: duktape implements toLocaleString() but ignores the locale, so fmtNum would
#: return "280813" where every browser returns "280,813" -- and the grouping is
#: one of the things under test. Give the engine the browser's behaviour.
_LOCALE_SHIM = """
Number.prototype.toLocaleString = function () {
  var neg = this < 0, n = Math.abs(this), i = Math.floor(n), frac = n - i;
  var out = "", str = String(i);
  while (str.length > 3) { out = "," + str.slice(-3) + out; str = str.slice(0, -3); }
  out = str + out;
  if (frac > 0) out += String(Math.round(frac * 1000) / 1000).slice(1);
  return (neg ? "-" : "") + out;
};
"""


def _run(expression: str) -> object:
    """Evaluate ``expression`` with app.js's pure helpers in scope."""
    dukpy = pytest.importorskip(
        "dukpy", reason="pip install -e '.[dev]' brings in the JS engine"
    )
    src = _LOCALE_SHIM + _pure_block() + "\nJSON.stringify(" + expression + ");"
    return json.loads(dukpy.evaljs(src))


#: Just enough DOM to RUN the painters. They are the half of app.js the pure
#: tests deliberately do not load, and the failure they produce in a browser is
#: silent -- one handler throws, the rest of the page merely looks stale -- so
#: "every painter survives a real /api/state document" needs its own proof.
_DOM_SHIM = """
function El(t){ this.tagName=t; this._text=""; this.className=""; this.childNodes=[];
  this.hidden=false; this.disabled=false; this.value=""; this.type=""; }
El.prototype.appendChild=function(c){ this._text=""; this.childNodes.push(c); return c; };
El.prototype.removeChild=function(c){ var i=this.childNodes.indexOf(c);
  if(i>=0) this.childNodes.splice(i,1); return c; };
El.prototype.addEventListener=function(){};
El.prototype.scrollIntoView=function(){};
El.prototype.querySelectorAll=function(){ return this.childNodes.slice(); };
Object.defineProperty(El.prototype,"firstChild",
  {get:function(){ return this.childNodes.length?this.childNodes[0]:null; }});
Object.defineProperty(El.prototype,"textContent",{
  get:function(){ if(!this.childNodes.length) return this._text;
    var o=""; for(var i=0;i<this.childNodes.length;i++) o+=this.childNodes[i].textContent+"\\n";
    return o; },
  set:function(v){ this.childNodes=[]; this._text=String(v); }});
var __els={};
var document={ getElementById:function(id){ return __els[id]||null; },
  createElement:function(t){ return new El(t); } };
var location={ origin:"http://127.0.0.1:8010" };
function fetch(){ throw new Error("a painter must not fetch"); }
function EventSource(){ this.close=function(){}; this.addEventListener=function(){}; }
function setTimeout(){ return 0; } function setInterval(){ return 0; }
function clearTimeout(){} function clearInterval(){}
function __mk(ids){ for (var i=0;i<ids.length;i++) __els[ids[i]]=new El("div"); }
function __dump(ids){ var o={}; for(var i=0;i<ids.length;i++){ var e=__els[ids[i]];
  o[ids[i]]={text:e.textContent, cls:e.className}; } return o; }
"""


def _paint(state: dict, extra: list[str]) -> dict:
    """Load all of app.js except its final ``init()`` call, paint, and return
    every element's text."""
    dukpy = pytest.importorskip("dukpy")
    assert APP_JS.rstrip().endswith("init();"), "app.js must end with its init() call"
    body = APP_JS.rstrip()[: -len("init();")]
    ids = sorted(set(re.findall(r'id="([A-Za-z0-9_]+)"', INDEX_HTML)))
    src = "\n".join(
        [
            _LOCALE_SHIM,
            _DOM_SHIM,
            "__mk(" + json.dumps(ids) + ");",
            body,
            "STATE = " + json.dumps(state) + ";",
            *extra,
            "JSON.stringify(__dump(" + json.dumps(ids) + "));",
        ]
    )
    return json.loads(dukpy.evaljs(src))


def _code_only(src: str) -> str:
    """``src`` with its comment lines removed.

    Without this a prose comment mentioning a calibration constant satisfies --
    or fails -- an assertion about what the page computes. Only what survives
    here can reach the screen.
    """
    out = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//") or stripped.startswith("*") or stripped.startswith("/*"):
            continue
        out.append(line)
    return "\n".join(out)


# --------------------------------------------------------------------------
# A real /api/state document, built by the real backend
# --------------------------------------------------------------------------
_MODELS_TOML = """
[gpu]
total_mib = 97887
margin_mib = 1024

[builds.stock]
venv = "/nonexistent/venv"
cuda_home = "/nonexistent/venv/cuda"

[models.tinymain]
id      = "tiny-main"
aliases = ["tm", "tiny"]
repo    = "example/tiny-main"
slot    = "main"
port    = 8099
build   = "stock"
ctx     = 262144

[models.tinyres]
id     = "tiny-resident"
repo   = "example/tiny-resident"
slot   = "resident"
port   = 8098
build  = "stock"
ctx    = 4096
vram_mib = 3000

# ctx = "native" for a repo that is not in the local hub cache: the registry
# cannot read a context length and says so rather than guessing one. This is
# the real source of a null `ctx` in /api/state.
[models.tinynative]
id     = "tiny-native"
repo   = "example/not-in-the-hub-cache"
slot   = "resident"
port   = 8097
build  = "stock"
ctx    = "native"
vram_mib = 3000
"""


@dataclass
class _FakeLive:
    """One row of ``control.live()``. pid = 0 on purpose: ``_unit_started_at``
    returns immediately for a pid it cannot attribute, so building this state
    document spawns no ``systemctl``."""

    key: str
    unit: str
    ready: bool = True
    state: str = "active"
    sub_state: str = "running"
    pid: int = 0
    restarts: int = 0
    port: int | None = None
    unknown: bool = False


class _FakeControl:
    def __init__(self, rows: list[_FakeLive]) -> None:
        self._rows = rows

    def live(self) -> list[_FakeLive]:
        return list(self._rows)


@pytest.fixture(scope="module")
def state_doc(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """``build_state`` run against a two-model registry and a fake control.

    Real code end to end: the real registry loader, the real RegistryRoutes,
    the real MetricsPoller (pointed at a transport that answers 404, which is
    the "engine not scrapeable" path and still fills every metrics key), and
    the real ``_headroom``. The key sets every contract below compares against
    come out of THIS document, so they cannot drift from app.py.
    """
    tmp = tmp_path_factory.mktemp("ui-v2")
    path = tmp / "models.toml"
    path.write_text(_MODELS_TOML)
    settings = _settings.Settings(
        listen_host="127.0.0.1",
        listen_port=8010,
        models_path=path,
        state_dir=tmp / "state",
        unit_prefix="sd-test-",
    )
    registry = _models.load(path)
    control = _FakeControl(
        [
            _FakeLive(key="tinymain", unit="sd-test-tinymain"),
            _FakeLive(key="tinynative", unit="sd-test-tinynative", ready=False,
                      sub_state="start"),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async def build() -> dict:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        rt = _app.build_runtime(
            settings, registry=registry, control=control, client=client
        )
        try:
            return await _app.build_state(rt)
        finally:
            await client.aclose()

    # nvidia-smi is a subprocess and its answer depends on the box; pin it so
    # the document is the same everywhere.
    real_free, real_total = _app._gpu.free_mib, _app._gpu.total_mib
    _app._gpu.free_mib = lambda: 40960  # type: ignore[assignment]
    _app._gpu.total_mib = lambda: 97887  # type: ignore[assignment]
    try:
        return asyncio.run(build())
    finally:
        _app._gpu.free_mib = real_free  # type: ignore[assignment]
        _app._gpu.total_mib = real_total  # type: ignore[assignment]


def _main_row(state_doc: dict) -> dict:
    rows = [m for m in state_doc["models"] if m["key"] == "tinymain"]
    assert rows, state_doc["models"]
    return rows[0]


# --------------------------------------------------------------------------
# The file itself
# --------------------------------------------------------------------------
def test_app_js_parses_in_a_real_js_engine() -> None:
    """A single unbalanced brace white-screens the whole dashboard.

    Wrapped in a function body so parsing does not RUN it: app.js ends by
    calling init(), which opens an EventSource.
    """
    dukpy = pytest.importorskip("dukpy")
    dukpy.evaljs("function __parse_only() {\n" + APP_JS + "\n}\n1;")


def test_app_js_stays_within_its_line_budget() -> None:
    """v1's app.js was 2,210 lines and nobody could say what any of it drew.
    The budget is the mechanism that keeps the page a renderer."""
    n = len(APP_JS.splitlines())
    assert n <= MAX_APP_JS_LINES, f"web/app.js is {n} lines, budget is {MAX_APP_JS_LINES}"


def test_the_gateway_chip_is_derived_from_where_the_page_is_served() -> None:
    """The chip names the URL a client configures. Hardcoding it is R1 in one
    line: the page would keep advertising :8010 from a servedeck started on a
    different port, and the client that believed it would 404."""
    code = _code_only(APP_JS)
    assert "`${location.origin}/v1`" in code, "the chip must be built from location.origin"
    assert not re.search(r"https?://\d", code), (
        "a hardcoded host:port in app.js is a second source of truth for the gateway URL"
    )


# --------------------------------------------------------------------------
# Formatting, executed
# --------------------------------------------------------------------------
def test_a_missing_number_is_an_em_dash_and_never_a_zero() -> None:
    """`null` is "the engine did not report this", `0` is "it reported none".
    Printing the first as the second is the v1 defect, and it is unreadable in
    exactly the situation it matters: an idle server looks like a stalled one."""
    got = _run("[fmtNum(null), fmtNum(undefined), fmtNum(NaN), fmtNum(0), fmtNum(280813)]")
    assert got[0] == EM
    assert got[1] == EM
    assert got[2] == EM
    assert got[3] == "0", "0 is a real reading and must print as 0, not as a dash"
    assert got[4] == "280,813", "thousands must be grouped"


def test_uptime_switches_units_at_ninety_seconds_and_at_an_hour() -> None:
    got = _run("[fmtUptime(89), fmtUptime(90), fmtUptime(5400), fmtUptime(null)]")
    assert got == ["89s", "1m", "1h 30", EM], got


def test_a_live_row_with_no_throughput_never_says_zero_tok_s(state_doc: dict) -> None:
    """`gen_tok_s: null` with `gen_state: "idle"` means nothing ran in the
    sampling window. The cell says so; it does not invent a rate."""
    row = dict(_main_row(state_doc))
    emitted = set(_metrics.MetricsSnapshot().to_dict())
    keys = set(row["metrics"] or {})
    assert keys <= emitted
    snap = _metrics.MetricsSnapshot(
        reachable=True, running=2, waiting=1, kv_usage_perc=0.1234,
        gen_reason=_metrics.IDLE,
    ).to_dict()
    assert snap["gen_tok_s"] is None and snap["gen_state"] == "idle"
    row["metrics"] = {k: snap[k] for k in keys}
    row["uptime_s"] = 5400.0

    cells = _run("liveRow(" + json.dumps(row) + ")")
    blob = json.dumps(cells)
    assert "0 tok/s" not in blob, f"a fabricated zero rate: {cells}"
    assert cells["rate"] == "idle", cells
    assert cells["kv"] == "12.3%", cells
    assert cells["queue"] == "2 / 1", cells
    assert cells["uptime"] == "1h 30", cells
    assert cells["ctx"] == "262,144 tok", cells


def test_an_unreachable_engine_renders_dashes_not_zeroes(state_doc: dict) -> None:
    """The row as build_state really produced it: the poller could not scrape
    the engine, so running/waiting/KV are structural zeroes. None of them is a
    measurement and none may be printed as one."""
    row = _main_row(state_doc)
    assert row["metrics"] is not None, "the ready model should carry a metrics block"
    assert row["metrics"]["reachable"] is False
    cells = _run("liveRow(" + json.dumps(row) + ")")
    for field in ("kv", "queue", "rate"):
        assert cells[field] == EM, f"{field} printed {cells[field]!r} for an unreachable engine"


def test_an_unreadable_context_length_is_not_printed_as_zero(state_doc: dict) -> None:
    """``ctx = "native"`` for a checkpoint that is not downloaded resolves to
    nothing, and /api/state carries `ctx: null` with the reason beside it.
    "0 tok" would be a context nobody measured, and a client sizing prompts
    against it is exactly the failure resolve_ctx refuses to cause."""
    rows = [m for m in state_doc["models"] if m["key"] == "tinynative"]
    assert rows and rows[0]["ctx"] is None, rows
    assert rows[0]["ctx_error"], "the reason must travel with the missing number"
    cells = _run("liveRow(" + json.dumps(rows[0]) + ")")
    assert cells["ctx"] == "unknown", cells
    assert "0" not in cells["ctx"], cells
    assert cells["unit"].endswith("not ready"), cells


def test_headroom_prints_the_backends_source_string_verbatim(state_doc: dict) -> None:
    """The panel's numbers are only worth reading with their provenance beside
    them: "measured from the running engine" is the difference between a fact
    and an estimate, and it is the backend's sentence, not the page's."""
    headroom = dict(state_doc["headroom"])
    headroom.update(
        unavailable=None,
        pool_tokens=280813,
        full_ctx=262144,
        full_context_requests=1,
        small_requests=18,
        source="measured from the running engine",
        note="KV cost curve calibrated on qwen38-flash-next.",
    )
    lines = _run("headroomLines(" + json.dumps(headroom) + ")")
    assert "measured from the running engine" in lines, lines
    assert any("280,813" in ln for ln in lines), lines
    assert any("40,960 MiB" in ln for ln in lines), lines


def test_headroom_prints_the_reason_and_no_capacity_number_when_unavailable(
    state_doc: dict,
) -> None:
    """This is the real document: nothing has reported a KV pool, so
    ``_headroom`` refused to answer. A capacity rendered in the same font as a
    measured one cannot be told apart from it on screen, so there must be no
    number to mistake."""
    headroom = state_doc["headroom"]
    assert headroom["unavailable"], headroom
    lines = _run("headroomLines(" + json.dumps(headroom) + ")")
    blob = " | ".join(lines)
    assert headroom["unavailable"] in blob, blob
    # The reason sentence itself names the KV pool; everything the PAGE added
    # around it is what must carry no capacity figure.
    rest = blob.replace(headroom["unavailable"], "")
    assert "requests" not in rest, f"a capacity figure beside an 'unavailable': {blob}"
    assert "KV pool" not in rest, blob
    assert "headroom" not in rest, blob
    assert len(lines) == 2, f"only free VRAM and the reason may be printed: {lines}"
    assert headroom["source"] is None


def test_a_restarts_start_waits_for_the_state_to_stop_listing_the_unit(
    state_doc: dict,
) -> None:
    """Restart is stop-then-start, and the start may not be fired while
    /api/state still lists the unit.

    app.py publishes the `stopped` notice BEFORE it re-reads systemd, so a
    start sequenced off that notice races the refresh and comes back 409
    `already_live` -- reliably, on a localhost round trip. The state document
    is the first thing that agrees with what the precheck will see.
    """
    still_live = json.dumps(state_doc)
    assert _run("restartsReadyToStart(" + still_live + ', {tinymain: 1})') == [], (
        "the start was released while /api/state still listed the unit as live"
    )

    gone = json.loads(still_live)
    for m in gone["models"]:
        if m["key"] == "tinymain":
            m["live"] = False
    assert _run("restartsReadyToStart(" + json.dumps(gone) + ", {tinymain: 1})") == ["tinymain"]
    # A model nobody asked to restart is never started by this path.
    assert _run("restartsReadyToStart(" + json.dumps(gone) + ", {})") == []


def test_notice_levels_style_differently() -> None:
    got = _run('[noticeClass("info"), noticeClass("error"), noticeClass(undefined)]')
    assert got[0] != got[1], got
    assert "error" in got[1] and "error" not in got[0], got
    assert got[2] == got[0], "an unlabelled notice must not be styled as an error"


def test_a_model_that_is_not_downloaded_says_so_in_the_dropdown() -> None:
    """The main-slot dropdown is one click away from a 90 GiB download. The
    option carries the on-disk answer /api/models already computed."""
    got = _run(
        'modelOption({id:"tiny-main", on_disk:true, disk_gib:12.5, reason:""})'
        + ""
    )
    assert "12.5 GiB" in got and "on disk" in got, got
    missing = _run(
        'modelOption({id:"tiny-main", on_disk:false, disk_gib:null,'
        ' reason:"not in the local hub cache"})'
    )
    assert "not in the local hub cache" in missing, missing
    assert "0" not in missing.replace("tiny-main", ""), (
        f"a null disk size must not become a 0 GiB claim: {missing}"
    )


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------
def _reads_of(obj: str, src: str) -> set[str]:
    return set(re.findall(rf"\b{re.escape(obj)}\.([A-Za-z_]\w*)", src))


def test_every_model_field_the_page_reads_is_one_build_state_emits(state_doc: dict) -> None:
    """The v1 card read ``m.trust`` and ``m.weights_gib``; neither had ever been
    in a payload, so a provenance badge silently said "estimated" for every
    model. A substring test cannot see that; a key-for-key comparison against
    the real serialiser can. The expected set is built by CALLING build_state.
    """
    code = _code_only(APP_JS)
    emitted = set(_main_row(state_doc))
    read = _reads_of("m", code) | _reads_of("holder", code)
    missing = read - emitted
    assert not missing, f"web/app.js reads model field(s) /api/state never sends: {sorted(missing)}"

    met_emitted = set(_main_row(state_doc)["metrics"])
    met_read = _reads_of("met", code)
    met_missing = met_read - met_emitted
    assert not met_missing, (
        f"web/app.js reads metrics field(s) /api/state never sends: {sorted(met_missing)}"
    )

    hr_emitted = set(state_doc["headroom"])
    hr_read = _reads_of("h", code)
    hr_missing = hr_read - hr_emitted
    assert not hr_missing, (
        f"web/app.js reads headroom field(s) /api/state never sends: {sorted(hr_missing)}"
    )


def test_the_state_document_still_carries_the_top_level_keys_the_page_paints(
    state_doc: dict,
) -> None:
    """The painters reach into the document by name. These are not derived from
    a variable the regex above can follow, so they are listed -- and checked
    against the document build_state really returned, so the list cannot rot."""
    for key in ("models", "headroom", "busy", "unknown_units"):
        assert key in state_doc, f"/api/state no longer carries {key!r}"
        assert f"STATE.{key}" in APP_JS or f'"{key}"' in APP_JS, (
            f"the page never reads {key!r}"
        )


def test_the_page_does_no_capacity_arithmetic() -> None:
    """REDESIGN §2.5: every capacity number is printed as parallelism.py
    computed it. The calibration constants must not exist here, and neither may
    a division by a KV pool -- a second formula would disagree with the first,
    and nothing on screen would say which one to believe."""
    code = _code_only(APP_JS)
    forbidden = {
        "the calibration pool": str(_parallelism.CALIBRATION_POOL_TOKENS),
        "the calibration pool, grouped": f"{_parallelism.CALIBRATION_POOL_TOKENS:,}",
        "the headroom factor": str(_parallelism.HEADROOM),
        "the fixed per-sequence cost": str(_parallelism.FIXED_COST_TOKENS),
        "the calibration model": _parallelism.CALIBRATION_MODEL,
    }
    for tokens, frac in _parallelism.KV_COST_ANCHORS:
        forbidden[f"KV anchor {tokens}"] = str(frac)
    for what, literal in forbidden.items():
        assert literal not in code, f"app.js contains {what} ({literal!r}) -- it must print, not compute"

    for pattern in (
        r"pool_tokens\s*/[^/*]",
        r"/\s*[A-Za-z_.]*pool_tokens",
        r"kv_cache_size_tokens\s*/[^/*]",
        r"/\s*[A-Za-z_.]*kv_cache_size_tokens",
    ):
        assert not re.search(pattern, code), f"app.js divides by the KV pool ({pattern})"


def test_a_refusal_shows_the_message_the_backend_wrote() -> None:
    """409/404 bodies are ``{error: {reason, message}}`` and the message names
    what refused and why ("the main slot is held by qwen27b -- use POST
    /api/switch/..."). Replacing it with "failed" throws away the only sentence
    that tells the operator what to do next."""
    body = _code_only(_fn_body("req"))
    assert "err.message" in body, body
    assert "body.error" in body, body
    assert "res.status" in body, "the fallback branch must at least name the status"
    # Every mutating caller goes through req(), so no route can answer into a
    # silent no-op.
    code = _code_only(APP_JS)
    assert len(re.findall(r"\bfetch\(", code)) == 1, (
        "every request must go through req(), which is what shows the refusal"
    )


def test_every_element_the_painters_write_to_exists_in_the_page() -> None:
    """A painter writing to an element that is not there is silent: the
    document.getElementById returns null, the assignment throws inside one
    handler, and the rest of the page looks merely stale."""
    html_ids = set(re.findall(r'id="([A-Za-z0-9_]+)"', INDEX_HTML))
    js_ids = set(re.findall(r'\$\("([A-Za-z0-9_]+)"\)', APP_JS))
    js_ids |= set(re.findall(r'getElementById\("([A-Za-z0-9_]+)"\)', APP_JS))
    assert js_ids, "no element lookups found -- the regex no longer matches app.js"
    assert not (js_ids - html_ids), (
        f"app.js writes to element(s) index.html does not have: {sorted(js_ids - html_ids)}"
    )
    assert not (html_ids - js_ids), (
        f"index.html has element(s) no painter ever fills: {sorted(html_ids - js_ids)}"
    )


def test_the_page_ships_no_external_reference() -> None:
    """It is served from 127.0.0.1 and has to render with the network down."""
    for name, text in (("index.html", INDEX_HTML), ("app.js", APP_JS),
                       ("style.css", (_app.WEB / "style.css").read_text())):
        assert "//cdn" not in text and "https://" not in text, f"{name} reaches off the box"


def test_every_painter_survives_a_real_state_document(state_doc: dict) -> None:
    """The painters, run end to end against what build_state really returned.

    This is the only check that the DOM half of the file works at all: a
    painter that throws takes out the rest of its own pass and leaves the page
    looking stale rather than broken, which is indistinguishable from a slow
    backend until someone reads the console.
    """
    dom = _paint(
        state_doc,
        [
            'REG = [{key:"tinymain", id:"tiny-main", slot:"main", on_disk:true,'
            ' disk_gib:12.5, reason:""}];',
            "paintRegistry();",
            '$("mainSel").value = "tinymain";',
            "paintAll();",
            'paintDoctor({ok:false, checks:[{name:"codex", ok:false, detail:"points at :8005"}]});',
            'onNotice({level:"error", reason:"main_slot_busy", message:"the main slot is held"});',
            'onProgress({key:"tinymain", kind:"marker", text:"Loading weights",'
            " marker_index:3, elapsed_s:12.4});",
        ],
    )
    live = dom["liveBody"]["text"]
    assert "tiny-main" in live and "262,144 tok" in live, live
    assert "0 tok/s" not in live, live
    # The table is the LIVE table: a registry model with no unit has no row.
    assert "tiny-resident" not in live, f"a model that is not running got a Live row: {live}"
    assert "Capacity unavailable" in dom["hrList"]["text"], dom["hrList"]
    events = dom["evList"]["text"]
    assert "the main slot is held" in events, events
    assert "main_slot_busy" in events, f"the notice lost its reason: {events}"
    assert "Loading weights" in dom["mainProg"]["text"], dom["mainProg"]
    assert dom["doctorDot"]["cls"] == "dot bad", dom["doctorDot"]
    assert "points at :8005" in dom["docTable"]["text"], dom["docTable"]
    assert dom["mainBtn"]["text"] == "Switch", dom["mainBtn"]
    assert "tiny-resident" in dom["resList"]["text"], dom["resList"]


def test_a_state_frame_is_what_releases_a_pending_restart(state_doc: dict) -> None:
    """The wiring, not just the predicate: onState must consult the pending
    restarts. Without the call, a Restart leaves the model stopped forever and
    the page shows no error, because nothing failed.

    The shim's fetch throws, which req() catches synchronously into a notice --
    so the notice text is the evidence that the start was attempted, and which
    path it was attempted on.
    """
    gone = json.loads(json.dumps(state_doc))
    for m in gone["models"]:
        if m["key"] == "tinymain":
            m["live"] = False
    dom = _paint(
        state_doc,
        [
            "RESTARTING['tinymain'] = 1;",
            "onState(" + json.dumps(gone) + ");",
        ],
    )
    events = dom["evList"]["text"]
    assert "/api/models/tinymain/start" in events, (
        f"a pending restart was never started when its unit went away: {events}"
    )


def test_the_js_engine_is_a_declared_dev_dependency() -> None:
    """Otherwise every rendering test above silently skips on a clean checkout
    and the only coverage left is the greps."""
    pyproject = (_app.WEB.parent.parent / "pyproject.toml").read_text()
    assert "dukpy" in pyproject, "add dukpy to the dev extra"
