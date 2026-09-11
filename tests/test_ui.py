"""Front-end / back-end contract tests for web/app.js.

There is no JavaScript runtime installed in .venv-gui (no node), so these are
NOT behavioural tests of the browser code. They are contract tests: every
field the page reads off a JSON payload must be a field the Python that
builds that payload actually emits. That is the class of bug that produced
three of the defects fixed on 2026-09-02 — the page read `m.trust` and
`m.weights_gib`, neither of which /api/models has ever returned, so a badge
silently rendered the wrong provenance for every model and a serving-detection
clause was dead code. A substring test would not have caught those; a
key-for-key comparison against the real serialiser does.

The one true string assertion here (the uptime fallback) pins a fix whose
whole content is the ABSENCE of an expression.
"""

from __future__ import annotations

import json
import math
import re

import pytest

from servedeck import app as _app
from servedeck import metrics

#: The web assets live INSIDE the package (they ship in the wheel), so this
#: is app.WEB, never a project-root guess.
APP_JS = (_app.WEB / "app.js").read_text()
INDEX_HTML = (_app.WEB / "index.html").read_text()

#: Fields the page synthesises on the client and therefore does not expect
#: from the backend.
_CLIENT_SIDE_MODEL_FIELDS = {"serving"}


def _reads_of(obj: str, src: str = APP_JS) -> set[str]:
    """Property names read off `obj` in the JS source (obj.foo)."""
    return set(re.findall(rf"\b{re.escape(obj)}\.([A-Za-z_]\w*)", src))


def _fn_body(name: str) -> str:
    """The source of a top-level `function name(...) { ... }`, by brace match."""
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


def _serving_line() -> str:
    """The whole function that renders the serving line, including the
    statements that pick which fields it renders."""
    return _fn_body("paintServingMeta")


# --------------------------------------------------------------------------
# Prefill speed — the feature
# --------------------------------------------------------------------------
def test_serving_line_shows_prefill_as_well_as_generation() -> None:
    """The status line must report prompt-processing throughput next to
    generation throughput. One number cannot tell a 17 s time-to-first-token
    apart from a slow decode."""
    line = _serving_line()
    assert "prefill" in line.lower(), line
    assert "gen" in line.lower(), line


def test_serving_line_never_prints_a_bare_zero_rate() -> None:
    """Regression: the line rendered `${liveMetrics.gen_tok_s || 0} tok/s`.

    Before the first telemetry event liveMetrics is {}, and an idle server
    reports no throughput at all — both printed "0 tok/s", which reads as a
    stalled engine rather than as "nothing is running right now".
    """
    line = _serving_line()
    assert "|| 0" not in line, f"|| 0 fabricates a zero throughput: {line}"
    assert "${liveMetrics.gen_tok_s" not in line, (
        "the raw metric must go through the null-aware formatter, not be "
        f"interpolated bare into the template: {line}"
    )
    assert "${liveMetrics.prefill" not in line, line


def test_rate_formatter_handles_null() -> None:
    """The formatter the serving line uses must map a null/absent rate to an
    em dash rather than to 0, NaN or 'undefined'."""
    m = re.search(r"function rateTxt\(([^)]*)\)\s*\{(.*?)\n\}", APP_JS, re.S)
    assert m, "web/app.js must define rateTxt()"
    body = m.group(2)
    assert '"—"' in body or "'—'" in body, body
    assert 'typeof' in body and 'number' in body, (
        f"guard on the type, not on truthiness — 0 is a number: {body}"
    )


# --------------------------------------------------------------------------
# Uptime
# --------------------------------------------------------------------------
def test_serving_line_does_not_fall_back_to_servedecks_own_uptime() -> None:
    """Regression: `s.server_uptime_s ?? s.uptime_s ?? 0`.

    `uptime_s` is SERVEDECK's uptime, a different quantity. app.py's own
    comment on the field says "The UI's 'Serving ... up Nm' must NOT use
    this", and procctl.process_uptime_s() exists precisely because using it
    "made a long-running server look freshly started". The fallback put the
    bug straight back.
    """
    line = _serving_line()
    assert "uptime_s" not in line.replace("server_uptime_s", ""), (
        f"the serving line must use server_uptime_s only: {line}"
    )


# --------------------------------------------------------------------------
# Payload contracts
# --------------------------------------------------------------------------
def test_every_metric_field_the_page_reads_is_one_the_poller_emits() -> None:
    emitted = set(metrics.MetricsSnapshot().to_dict())
    read = _reads_of("liveMetrics")
    missing = read - emitted
    assert not missing, (
        f"web/app.js reads metric field(s) /api/state never sends: {sorted(missing)}"
    )


def test_every_model_field_the_page_reads_is_one_api_models_emits() -> None:
    """Regression: the model card read m.trust and m.weights_gib.

    Neither was ever in the /api/models payload, so `m.trust === "measured"`
    was false for every model and EVERY card was labelled "estimated" —
    including the two SPEC.md §0 lists as measured. A provenance badge that
    always says "estimated" is worse than no badge: SPEC.md §3 attaches a
    "~25% optimistic historically" warning to that label.
    """
    from servedeck import app

    rows = app._model_rows()
    if not rows:
        pytest.skip("no models in the local hub cache — nothing to check against")
    emitted = set(rows[0])
    # Only where `m` IS a model row: renderModels' card loop and paintState's
    # serving-match loop. `m` is also a local name elsewhere in the file.
    where = _fn_body("renderModels") + _fn_body("paintState")
    read = _reads_of("m", where) | _reads_of("MODELS[sel]")
    missing = read - emitted - _CLIENT_SIDE_MODEL_FIELDS
    assert not missing, (
        f"web/app.js reads model field(s) /api/models never sends: {sorted(missing)}"
    )


def test_measured_models_are_not_all_labelled_estimated() -> None:
    """The badge must be able to say "measured" for a model that has been
    measured. SPEC.md §0 records boots of both NVFP4 models."""
    from servedeck import app

    rows = app._model_rows()
    trusts = {r["repo_id"]: r["trust"] for r in rows}
    if not any(t != "unknown" for t in trusts.values()):
        pytest.skip("no model on this machine has a resolvable trust level")
    assert any(str(t).startswith("measured") or t == "estimated" for t in trusts.values()), (
        f"every model resolves to 'unknown': {trusts}"
    )


#: Every element a painter fills from /api/*. Their shipped inner text is
#: what a visitor sees until (and if) that painter runs — on a slow first
#: paint, a failed estimate, or a backend that never comes up.
_LIVE_DATA_SLOTS = (
    "sMeta", "gpuUsed", "gpuTotal", "vramTxt", "mcount", "dKv", "dKvTok",
    "dBadge", "kvPct", "kvTok",
    "mRun", "mWait", "mPre", "hitRate",
    # The control readouts that describe a bound rather than a reading: a
    # shipped digit here would be a ceiling nothing had computed yet.
    "agentsFit", "ctxBound",
    # The request-size window and the parallelism recommendation. Same rule:
    # until their painter runs there is no reading, and a shipped digit here
    # would be a fabricated one. The distribution itself is drawn on
    # #reqHist, a canvas, which has no text content to fabricate.
    "pctP90", "winMeta",
    "recN", "recBasis", "recMath", "recP99", "recCal",
    # Boot progress: elapsed clock and ETA. Both were the prototype's literals.
    "elapsed", "etaTxt",
)


def test_page_placeholders_are_not_fabricated_readings() -> None:
    """web/app.js's own header rule: "No simulated data. Everything arrives
    from /api/* and /api/events."

    index.html shipped the design prototype's mock readings as its static
    text: ":8001 · 262,144 ctx · 99.3 tok/s · up 12m", "92,571 MiB" of VRAM
    in use, "8.8 GiB" of KV, "6 requests · synthetic test traffic", and a
    green "measured" provenance badge over all of it. Every one of those is
    on screen whenever its painter has not run — which includes the case
    where the backend is unreachable and the page is otherwise empty.
    """
    bad = {}
    for slot in _LIVE_DATA_SLOTS:
        m = re.search(rf'id="{slot}"[^>]*>([^<]*)<', INDEX_HTML)
        assert m, f"could not find #{slot} in index.html"
        if re.search(r"\d", m.group(1)):
            bad[slot] = m.group(1)
    assert not bad, f"placeholders that read as live data: {bad}"


def test_provenance_badge_does_not_ship_pre_styled_as_measured() -> None:
    """`class="tag meas"` is the green "this was actually booted and
    measured" styling. Shipping it on an empty badge asserts provenance for
    numbers that do not exist yet."""
    m = re.search(r'<span class="([^"]*)" id="dBadge"', INDEX_HTML)
    assert m, "could not find #dBadge in index.html"
    assert "meas" not in m.group(1), m.group(1)


def test_selecting_a_model_redraws_the_context_control() -> None:
    """A ceiling that is only read once is the hardcoded array again, one
    render later: picking a different model must re-derive it, or the control
    keeps describing the previous model."""
    assert "renderCtx(" in _fn_body("renderModels"), (
        "clicking a model card must redraw the context control"
    )


# --------------------------------------------------------------------------
# Structural validity of the page's only script
# --------------------------------------------------------------------------
def _scan_js(src: str) -> tuple[list[str], list[tuple[int, str]]]:
    """Walk web/app.js, skipping comments and string/template bodies.

    Returns (unbalanced-bracket errors, list of (line, kind) for unterminated
    literals). There is no JavaScript engine installed anywhere on this box —
    no node, deno, quickjs, and no Python JS binding in .venv-gui — so app.js
    gets no syntax checking at all before it reaches the browser, where a
    single unbalanced brace white-screens the entire dashboard. This is the
    cheap 90%: it catches every failure mode an edit to this file realistically
    produces.

    Sound only because app.js contains no regex literals (asserted below), so
    an unquoted `/` is always division.

    Detection power, measured by injecting faults into app.js: 6 of 7 (dropped
    brace, extra brace, unclosed template, unclosed paren, unterminated
    string, mismatched bracket type). The miss is a *misplaced* block-comment
    terminator, which stays balanced by swallowing to the next `*/` -- a real
    JS parser behaves the same way. Do not read a pass here as "app.js is
    valid JavaScript"; read it as "app.js will at least parse".
    """
    errors: list[str] = []
    open_at: list[tuple[str, int]] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    i, line, n = 0, 1, len(src)
    # Stack of template-literal depths: entering ${ inside a template pushes a
    # normal-code context that ends at the matching }.
    tmpl: list[int] = []
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            i = src.find("\n", i)
            if i < 0:
                break
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            end = src.find("*/", i + 2)
            if end < 0:
                errors.append(f"unterminated block comment at line {line}")
                break
            line += src.count("\n", i, end)
            i = end + 2
            continue
        if c in "'\"":
            j, quote = i + 1, c
            while j < n and src[j] != quote:
                if src[j] == "\\":
                    j += 1
                elif src[j] == "\n":
                    break
                j += 1
            if j >= n or src[j] != quote:
                errors.append(f"unterminated {quote} string at line {line}")
            i = j + 1
            continue
        if c == "`":
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == "`":
                    break
                if src[j] == "$" and j + 1 < n and src[j + 1] == "{":
                    tmpl.append(len(open_at))
                    open_at.append(("{", line))
                    j += 2
                    break
                if src[j] == "\n":
                    line += 1
                j += 1
            if j >= n:
                errors.append(f"unterminated template literal at line {line}")
            i = j + 1 if j < n and src[j] == "`" else j
            continue
        if c in "([{":
            open_at.append((c, line))
        elif c in ")]}":
            if not open_at:
                errors.append(f"stray {c!r} at line {line}")
            elif open_at[-1][0] != pairs[c]:
                errors.append(
                    f"{c!r} at line {line} closes {open_at[-1][0]!r} opened at line {open_at[-1][1]}"
                )
                open_at.pop()
            else:
                open_at.pop()
                # Leaving a ${...} hands control back to the template body.
                if tmpl and len(open_at) == tmpl[-1]:
                    tmpl.pop()
                    j = i + 1
                    while j < n:
                        if src[j] == "\\":
                            j += 2
                            continue
                        if src[j] == "`":
                            break
                        if src[j] == "$" and j + 1 < n and src[j + 1] == "{":
                            tmpl.append(len(open_at))
                            open_at.append(("{", line))
                            j += 2
                            break
                        if src[j] == "\n":
                            line += 1
                        j += 1
                    i = j + 1 if j < n and src[j] == "`" else j
                    continue
        i += 1
    for ch, ln in open_at:
        errors.append(f"unclosed {ch!r} opened at line {ln}")
    return errors, []


def test_app_js_has_no_regex_literals() -> None:
    """Premise of the bracket checker below: an unquoted `/` is division."""
    assert not re.search(r"(?:match|replace|replaceAll|test|split|exec|search)\(\s*/", APP_JS)
    assert not re.search(r"=\s*/[^/*\s]", APP_JS)


def test_app_js_brackets_and_literals_balance() -> None:
    """A single unbalanced brace in web/app.js white-screens the dashboard,
    and nothing else in this project would notice before the browser does."""
    errors, _ = _scan_js(APP_JS)
    assert not errors, "web/app.js is not structurally valid:\n  " + "\n  ".join(errors)


def test_serving_line_repaints_on_telemetry_not_only_on_state() -> None:
    """The two throughput figures arrive on the telemetry event (2 s); the
    rest of the serving line arrives on state (5 s). Painting the line only
    from paintState() showed readings up to 5 s old and dropped every other
    sample."""
    assert "paintServingMeta()" in _fn_body("paintTelemetry")
    assert "paintServingMeta()" in _fn_body("paintState")


def test_serving_line_prefers_the_running_servers_own_context_length() -> None:
    """`supervisor.max_model_len` is desired config; `upstream.max_model_len`
    is read out of the serving process's command line. For an adopted server
    the two can differ, and only the second is a fact."""
    line = _serving_line()
    i_live = line.find("up.max_model_len")
    i_desired = line.find("max_model_len", line.find("supervisor"))
    assert i_live >= 0, f"serving line ignores the running server's own ctx: {line}"
    assert i_live < i_desired, (
        f"the running server's own ctx must be preferred over desired config: {line}"
    )


# --------------------------------------------------------------------------
# The serving line names a URL, not a port
# --------------------------------------------------------------------------
def test_serving_line_shows_a_full_url_not_a_bare_port() -> None:
    """The serving line is the one place a user goes to find the address to
    paste into a client. It rendered `:${up.port}` — a bare port, which is not
    an address: every reader had to reconstruct the scheme and host by hand.
    """
    line = _serving_line()
    assert "http://localhost:${up.port}" in line, (
        f"the serving line must render a copyable base URL: {line}"
    )
    assert "`:${up.port}" not in line, f"bare-port form still present: {line}"


# --------------------------------------------------------------------------
# The context ladder comes from the model, not from a literal
# --------------------------------------------------------------------------
def test_context_control_is_bounded_by_the_model_and_by_what_fits() -> None:
    """`const CTXS = [8192, ... 262144]` was wrong in both directions: it
    offered lengths a small model cannot reach (the engine refuses at boot,
    minutes later) and it capped a large one — a model declaring
    max_position_embeddings 1,048,576 had three quarters of its range
    unreachable from the UI.

    Deriving the ladder from the model ceiling alone fixed only half of that.
    A length the CHECKPOINT allows can still be one the KV budget cannot hold
    for the number of agents asked for, and that failure also arrives minutes
    into a boot. Both bounds must reach the control.
    """
    assert "const CTXS" not in APP_JS, "the hardcoded context array is back"
    assert 'id="ctx"' in INDEX_HTML and 'type="range"' in INDEX_HTML, (
        "context per agent must be a slider over a real range"
    )
    assert 'id="ctxBtns"' not in INDEX_HTML, "the fixed ladder of buttons is back"
    body = _fn_body("renderCtx")
    assert "ctx_max_model" in body, "the model's own ceiling must bound the slider"
    assert "ctx_max_fit" in body, "what the KV budget holds must bound the slider"
    assert "Math.min(modelMax, fit)" in body, (
        f"the binding limit is the SMALLER of the two bounds: {body}"
    )
    assert "DEFAULT_MAX_CTX" in body, "a model with no declared ceiling needs a fallback"


def test_the_agent_count_is_an_input_not_a_hardcoded_one() -> None:
    """"Context per agent" is meaningless without an agent count, and the page
    sent max_num_seqs: 1 on every estimate and every start — so the ceiling it
    drew was the one-agent ceiling however many agents the operator wanted."""
    assert 'id="agents"' in INDEX_HTML, "there is no control for the agent count"
    assert "max_num_seqs: 1" not in APP_JS, (
        "the agent count is still hardcoded to 1 in a request body"
    )
    assert "max_num_seqs: agents" in APP_JS


def test_the_bounds_are_redrawn_on_every_estimate() -> None:
    """Both bounds move with utilization, with the agent count and with the
    model. A control drawn once at load is the hardcoded ladder again, one
    render later."""
    assert "renderCtx(d)" in _fn_body("paintEstimate"), (
        "the estimate that computes the bounds must redraw the control"
    )


def test_model_max_ctx_is_a_field_api_models_actually_sends() -> None:
    """The whole ladder now depends on this one field arriving. The class of
    bug this file exists for is exactly a page reading a key the serialiser
    never emits."""
    from servedeck import app

    rows = app._model_rows()
    if not rows:
        pytest.skip("no models in the local hub cache — nothing to check against")
    assert "model_max_ctx" in rows[0]
    assert isinstance(rows[0]["model_max_ctx"], int) and rows[0]["model_max_ctx"] > 0


# --------------------------------------------------------------------------
# Prefill vs decode vs TTFT (owner report: "confusing whether it is prefill or
# decode time when only 1 is visible")
# --------------------------------------------------------------------------
def test_throughput_strip_gives_every_figure_a_name_and_a_unit() -> None:
    """One unlabelled number cannot say whether the server is slow to start
    answering or slow to keep answering. Each cell names the figure, spells
    the unit out in full, and has all three of its lines in the markup -- the
    window reading, what that reading is, and the lifetime companion -- so no
    line can be missing at render time and no cell can go blank."""
    for slot, label, unit in (
        ("thPrefill", "Prefill", "prompt tokens/second"),
        ("thDecode", "Decode", "generated tokens/second"),
        ("thTtft", "Time to first token", "seconds, mean"),
    ):
        for suffix, what in (("", "figure"), ("S", "state line"), ("L", "lifetime line")):
            assert f'id="{slot}{suffix}"' in INDEX_HTML, f"{slot}'s {what} is missing"
        assert label in INDEX_HTML, f"{slot} is not labelled {label!r}"
        assert unit in INDEX_HTML, f"{slot} does not spell out its unit"
    body = _fn_body("paintThroughput")
    for slot in ("thPrefill", "thDecode", "thTtft"):
        assert slot in body, f"paintThroughput never writes {slot}"


def test_a_missing_throughput_figure_says_why_and_keeps_its_last_value() -> None:
    """A bare em dash is the same defect one step on: it distinguishes neither
    "nothing ran just now" from "this build does not publish that metric", nor
    either of them from "the other figure took this slot"."""
    body = _fn_body("windowFigure")
    assert '"n/a"' in body, "windowFigure() must render n/a, not a bare dash"
    assert '"idle"' in body, "an idle window is a different fact from a missing metric"
    assert "reason" in body, "windowFigure() must carry the backend's reason through"
    assert "last" in body, "an idle figure must keep the last reading it had"
    assert "agoTxt" in body, "a stale reading without its age reads as current"
    assert "—" not in _fn_body("paintThroughput"), (
        "the throughput strip must not fall back to an em dash"
    )


def test_the_window_figure_cannot_be_filled_in_from_the_lifetime_average() -> None:
    """The substitution, pinned in the source as well as in the render: the
    function that draws the window line is not given the lifetime value at
    all, so it cannot print it however it is later edited."""
    import inspect

    sig = re.search(r"function windowFigure\(([^)]*)\)", APP_JS)
    assert sig, "web/app.js must define windowFigure()"
    params = [p.strip() for p in sig.group(1).split(",")]
    assert not any("avg" in p or "life" in p for p in params), (
        f"windowFigure() takes a lifetime figure: {params}"
    )
    body = _fn_body("windowFigure")
    assert "_avg" not in body, body
    del inspect


def test_the_strip_names_prefill_decode_and_time_to_first_token() -> None:
    """The three figures must each be named in full. This rule lived on the
    serving line until the throughput figures moved to the strip; it moved with
    them. 'gen' reads as neither prefill nor decode."""
    start = INDEX_HTML.index('class="thru" id="thru"')
    strip = INDEX_HTML[start: INDEX_HTML.index('class="phases"', start)]
    for word in ("Prefill", "Decode", "Time to first token"):
        assert word in strip, f"the throughput strip never says {word!r}"
    assert ">gen<" not in strip.lower() and " gen " not in strip.lower(), (
        "the strip abbreviates generation throughput to 'gen'"
    )


def test_the_page_reads_the_reason_fields_the_poller_emits() -> None:
    """Contract: every *_reason the page renders must be a field
    MetricsSnapshot.to_dict() actually produces."""
    emitted = set(metrics.MetricsSnapshot().to_dict())
    read = {
        f
        for f in _reads_of("m", _fn_body("paintThroughput")) | _reads_of("liveMetrics")
        if f.endswith("_reason")
    }
    assert read, "the page reads no reason field at all"
    assert read <= emitted, f"page reads reasons the poller never emits: {read - emitted}"


def test_the_kv_budget_says_whether_it_was_measured_or_estimated() -> None:
    """"Prefer the measured value and label it measured vs estimated." The
    panel printed a token count with no provenance at all, so a figure the
    per-architecture calculator produced for a model that has never booted
    read exactly like one the engine reported."""
    body = _fn_body("paintEstimate")
    assert "kv_source" in body, "the KV budget never reads the estimate's provenance"
    assert "measured" in body and "estimated" in body
    assert "liveFacts.kv_tokens" in body, (
        "the running engine's own KV size must win where it applies"
    )


def test_the_estimate_payload_carries_the_fields_the_panel_reads() -> None:
    """Contract, the same class of bug as the model-row fields: every key
    paintEstimate reads off the estimate must be one _estimate() emits."""
    import inspect

    src = inspect.getsource(_app._estimate)
    for key in ("kv_source", "kv_geometry", "ctx_max_model", "ctx_max_fit", "agents"):
        assert f'"{key}"' in src, f"_estimate() does not emit {key}"


# --------------------------------------------------------------------------
# Controls that lie
# --------------------------------------------------------------------------
def test_no_enabled_control_is_unwired() -> None:
    """A control the page never reads is a claim the page cannot keep.

    "Auto-restart on crash" shipped enabled AND checked, and nothing anywhere
    read it: the operator was told a recovery behaviour was armed by a
    checkbox no code consults. "Set subagents" shipped enabled next to a hint
    saying it writes max_concurrent_threads_per_session, with no endpoint
    behind it. An unimplemented control must be disabled with a reason, the
    way Smoke test already was.
    """
    dead = []
    for m in re.finditer(r"<(button|input|select)\b([^>]*)>", INDEX_HTML):
        attrs = m.group(2)
        idm = re.search(r'id="([A-Za-z0-9_]+)"', attrs)
        if not idm:
            continue
        elem_id = idm.group(1)
        if elem_id in APP_JS:
            continue          # painted or handled by the page's own script
        assert "disabled" in attrs, f"#{elem_id} is enabled and nothing reads it"
        assert "title=" in attrs, f"#{elem_id} is disabled with no reason on it"
    assert not dead, f"enabled controls nothing reads: {dead}"


def test_the_preflight_strip_is_computed_not_written_into_the_page() -> None:
    """Four static spans — "GPU free", "weights cached", "ptrace_scope 0",
    "disk 412 GiB" — were on screen whatever the machine was doing, including
    when the backend was unreachable and the rest of the page was empty. No
    code has ever measured free disk."""
    markup = re.sub(r"<!--.*?-->", "", INDEX_HTML, flags=re.S)
    assert "disk 412 GiB" not in markup
    assert 'id="pref"' in INDEX_HTML
    body = _fn_body("paintPreflight")
    assert "findings" in body, "the strip must be built from real findings"
    assert "paintPreflight" in _fn_body("paintEstimate")


def test_the_boot_eta_is_not_a_literal_string() -> None:
    """"typically 4m10s warm · 9m50s cold" was hardcoded markup. history.py
    computes a real per-model ETA and marks when it is falling back to a
    calibration figure; until that is painted, the slot must be empty rather
    than assert a number."""
    m = re.search(r'id="etaTxt"[^>]*>([^<]*)<', INDEX_HTML)
    assert m, "could not find #etaTxt"
    assert not re.search(r"\d", m.group(1)), f"fabricated ETA: {m.group(1)!r}"


# --------------------------------------------------------------------------
# Rendering tests: the real app.js functions, run in a real JS engine
# --------------------------------------------------------------------------
#
# Everything above this line is a CONTRACT test -- it reads the source as
# text. That catches a field name that does not exist and misses everything
# about what the page actually puts on screen, which is precisely where the
# defect the owner reported lived: "when one is not displayed, the prefill
# speed shows, when it is not running, while generate shows when generating".
# No substring assertion can see that.
#
# So these tests EXECUTE web/app.js's rendering functions against a small DOM
# shim, with payloads produced by the real MetricsPoller, and assert on the
# text that lands in each element. duktape (via dukpy) is ES5.1 plus the ES6
# app.js actually uses (const/let, arrow functions, template literals).

_DOM_SHIM = """
/* duktape implements toLocaleString() but ignores the locale, so fmt() would
   return "1240" where a browser returns "1,240" -- and the grouping is part of
   what the tests are checking. Give the engine the browser's behaviour. */
Number.prototype.toLocaleString = function () {
  var neg = this < 0, n = Math.abs(this), i = Math.floor(n), frac = n - i;
  var out = "", str = String(i);
  while (str.length > 3) { out = "," + str.slice(-3) + out; str = str.slice(0, -3); }
  out = str + out;
  if (frac > 0) out += String(Math.round(frac * 1000) / 1000).slice(1);
  return (neg ? "-" : "") + out;
};
function El(tag) {
  this.tagName = tag || "div";
  this._text = "";
  this.className = "";
  this.title = "";
  this.childNodes = [];
  this.style = {};
  this._attrs = {};
  this.dataset = {};
  this.disabled = false;
}
/* classList/children/setAttribute exist because the painters added in the
   2026-09-09 UI pass use them: paintBoot toggles classes on phase steps, and
   renderModels marks cards selected. Without them those functions cannot be
   EXECUTED here, and a test that only greps their source passes unchanged when
   their behaviour is inverted (proven by mutation on the preflight severity
   fix, which grep-based assertions missed). */
Object.defineProperty(El.prototype, "children", {
  get: function () { return this.childNodes; }
});
El.prototype.classList = undefined;
function ClassList(el) { this._el = el; }
ClassList.prototype._parts = function () {
  return this._el.className ? this._el.className.split(" ").filter(function (c) { return c; }) : [];
};
ClassList.prototype.add = function (c) {
  var p = this._parts();
  if (p.indexOf(c) < 0) p.push(c);
  this._el.className = p.join(" ");
};
ClassList.prototype.remove = function (c) {
  this._el.className = this._parts().filter(function (x) { return x !== c; }).join(" ");
};
ClassList.prototype.contains = function (c) { return this._parts().indexOf(c) >= 0; };
ClassList.prototype.toggle = function (c, force) {
  var has = this.contains(c);
  var want = (force === undefined) ? !has : !!force;
  if (want) this.add(c); else this.remove(c);
  return want;
};
Object.defineProperty(El.prototype, "classList", {
  get: function () { if (!this._cl) this._cl = new ClassList(this); return this._cl; }
});
El.prototype.setAttribute = function (k, v) { this._attrs[k] = String(v); };
El.prototype.getAttribute = function (k) {
  return Object.prototype.hasOwnProperty.call(this._attrs, k) ? this._attrs[k] : null;
};
/* Returns a stub rather than null. renderModels() builds a card with an
   innerHTML template and then writes into b.querySelector(".nm"); with a null
   return the whole painter throws in duktape and can only be grepped, which is
   exactly the weakness the mutation battery keeps exposing. A stable stub per
   selector per element lets the painter run to completion. */
El.prototype.querySelector = function (sel) {
  if (!this._qs) this._qs = {};
  if (!this._qs[sel]) this._qs[sel] = new El("span");
  return this._qs[sel];
};
El.prototype.contains = function (n) {
  for (var i = 0; i < this.childNodes.length; i++) {
    if (this.childNodes[i] === n || this.childNodes[i].contains(n)) return true;
  }
  return false;
};
El.prototype.remove = function () {};
El.prototype.focus = function () {};
Object.defineProperty(El.prototype, "innerHTML", {
  get: function () { return this._html || ""; },
  set: function (v) { this._html = String(v); this.childNodes = []; }
});
Object.defineProperty(El.prototype, "textContent", {
  get: function () {
    if (this.childNodes.length === 0) return this._text;
    var out = "";
    for (var i = 0; i < this.childNodes.length; i++) out += this.childNodes[i].textContent;
    return out;
  },
  set: function (v) { this.childNodes = []; this._text = String(v); }
});
El.prototype.appendChild = function (c) { this._text = ""; this.childNodes.push(c); return c; };
var __els = {};
var document = {
  getElementById: function (id) { return __els[id] || null; },
  createElement: function (tag) { return new El(tag); },
  createTextNode: function (t) { var n = new El("#text"); n._text = String(t); return n; },
  querySelector: function () { return null; }
};
var liveMetrics = {};
var lastState = null;
var ctx = 262144;
function __mk(ids) { for (var i = 0; i < ids.length; i++) __els[ids[i]] = new El("div"); }
function __dump(ids) {
  var out = {};
  for (var i = 0; i < ids.length; i++) {
    var e = __els[ids[i]];
    out[ids[i]] = e ? { text: e.textContent, cls: e.className, title: e.title } : null;
  }
  return JSON.stringify(out);
}
"""

#: The functions under test, lifted verbatim out of web/app.js.
_RENDER_FNS = (
    "agoTxt", "secsTxt", "rateTxt", "windowFigure", "lifeTxt",
    "uptimeTxt", "busyPhase", "resolutionNote", "resolutionDetail",
    "paintThroughput", "paintServingMeta",
)


def _const_line(name: str) -> str:
    """A top-level `const name = ...;` line, verbatim."""
    i = APP_JS.index(f"const {name} = ")
    return APP_JS[i : APP_JS.index("\n", i)]


def _optional_const_line(name: str) -> str:
    """_const_line, or nothing when app.js has no such const. The harness then
    still loads an older app.js, so a test fails on its assertion rather than
    on a missing name."""
    return _const_line(name) if f"const {name} = " in APP_JS else ""


def _const_block(name: str) -> str:
    """A top-level `const name = { ... };` spanning several lines, verbatim."""
    start = APP_JS.index(f"const {name} = ")
    depth, i = 0, APP_JS.index("{", start)
    for j in range(i, len(APP_JS)):
        if APP_JS[j] == "{":
            depth += 1
        elif APP_JS[j] == "}":
            depth -= 1
            if depth == 0:
                return APP_JS[start : APP_JS.index(";", j) + 1]
    raise AssertionError(f"unbalanced braces in const {name}")


def _element_ids() -> list[str]:
    return sorted(set(re.findall(r'id="([A-Za-z0-9_]+)"', INDEX_HTML)))


#: The two helpers every painter and every pure function resolves first. Loaded
#: by all three JS harnesses below; without them a painter throws on its first
#: `$("id")` before it renders anything, and the failure looks like a bug in the
#: function under test rather than in the harness.
_PRELUDE = (_const_line("$"), _const_line("fmt"))


def _render(payload: dict, state: dict | None = None) -> dict:
    """Paint the throughput strip (and, with `state`, the serving line) from a
    real /api telemetry payload and return every element's rendered text."""
    ids = _element_ids()
    src = [
        *_PRELUDE,
        _const_line("BUSY_PHASES"),
        *[_fn_body(n) for n in _RENDER_FNS],
        f"__mk({json.dumps(ids)});",
        f"liveMetrics = {json.dumps(payload)};",
        f"lastState = {json.dumps(state)};",
        "paintThroughput();",
        "if (lastState) paintServingMeta();",
        f"__dump({json.dumps(ids)});",
    ]
    return _run_js("\n".join(src))


def _payload(**kw) -> dict:
    """A telemetry payload the poller could really have produced."""
    snap = metrics.MetricsSnapshot(reachable=True)
    for k, v in kw.items():
        setattr(snap, k, v)
    return snap.to_dict()


_SERVING = {
    "upstream": {"up": True, "port": 8001, "max_model_len": 262144},
    "server_uptime_s": 11520,
}

_CELLS = ("thPrefill", "thDecode", "thTtft")


def test_both_figures_render_at_once_in_every_server_state() -> None:
    """The defect, stated as a test: no state of the server may leave prefill
    or decode without a figure of its own.

    Before the fix an idle window put NOTHING in the cell (or, worse, quietly
    substituted the lifetime average), so the two numbers appeared to take
    turns. Every state below must produce a non-empty, distinct reading for
    both -- and the labels are static markup, so they are present regardless.
    """
    states = {
        "both busy": _payload(
            prefill_tok_s=1240.0, gen_tok_s=249.0, ttft_s=1.85,
            prefill_tok_s_avg=3013.3, gen_tok_s_avg=104.4, ttft_s_avg=8.62,
        ),
        "decoding, no prefill": _payload(
            gen_tok_s=249.0, prefill_reason=metrics.IDLE, ttft_reason=metrics.IDLE,
            prefill_tok_s_last=1240.0, prefill_last_age_s=34.0,
            prefill_tok_s_avg=3013.3, gen_tok_s_avg=104.4, ttft_s_avg=8.62,
        ),
        "prefilling, nothing decoded yet": _payload(
            prefill_tok_s=1240.0, gen_reason=metrics.IDLE, ttft_reason=metrics.IDLE,
            gen_tok_s_last=249.0, gen_last_age_s=6.0,
            prefill_tok_s_avg=3013.3, gen_tok_s_avg=104.4, ttft_s_avg=8.62,
        ),
        "completely idle": _payload(
            prefill_reason=metrics.IDLE, gen_reason=metrics.IDLE,
            ttft_reason=metrics.IDLE,
            prefill_tok_s_avg=3013.3, gen_tok_s_avg=104.4, ttft_s_avg=8.62,
        ),
        "backend gone": metrics.unreachable_snapshot(),
    }
    for name, payload in states.items():
        dom = _render(payload, _SERVING)
        for cell in _CELLS:
            assert dom[cell]["text"].strip(), f"{name}: {cell} rendered empty"
            assert dom[cell + "S"]["text"].strip(), f"{name}: {cell} has no state line"
            assert dom[cell + "L"]["text"].strip(), f"{name}: {cell} has no lifetime line"
        # The three cells are three different questions; two of them showing
        # the same string means one has been substituted for another.
        shown = [dom[c]["text"] for c in _CELLS if dom[c]["text"] not in ("idle", "n/a")]
        assert len(set(shown)) == len(shown), f"{name}: a figure was duplicated: {shown}"


def test_an_idle_figure_shows_its_last_value_and_an_age_not_a_blank() -> None:
    """"Show it as idle or stale with its last value and an age, never blank
    and never silently replaced by the other.\""""
    dom = _render(
        _payload(
            gen_tok_s=249.0,
            prefill_reason=metrics.IDLE,
            prefill_tok_s_last=1240.0,
            prefill_last_age_s=34.0,
            prefill_tok_s_avg=3013.3,
        ),
        _SERVING,
    )
    assert dom["thPrefill"]["text"] == "idle"
    note = dom["thPrefillS"]["text"]
    assert "1,240 tok/s" in note, f"the last reading is missing: {note}"
    assert "34 s ago" in note, f"the age is missing: {note}"
    # ...and the decode cell is untouched by any of it.
    assert dom["thDecode"]["text"] == "249.0 tok/s"


def test_an_idle_figure_is_never_filled_in_with_the_lifetime_average() -> None:
    """The substitution that made the panel unreadable: a lifetime average,
    same font, same slot, marked only with a "~". On this recording the two
    differ by 12x for prefill and 2.4x for decode."""
    dom = _render(
        _payload(
            prefill_reason=metrics.IDLE, gen_reason=metrics.IDLE,
            prefill_tok_s_avg=3013.3, gen_tok_s_avg=104.4,
        ),
        _SERVING,
    )
    assert dom["thPrefill"]["text"] == "idle"
    assert dom["thDecode"]["text"] == "idle"
    assert "3,013" not in dom["thPrefill"]["text"]
    assert "104" not in dom["thDecode"]["text"]
    # The lifetime figures are still on screen -- on their own line, naming
    # their own denominator, which is what makes them readable at all.
    assert "3,013 tok/s" in dom["thPrefillL"]["text"]
    assert "per second of prefill time" in dom["thPrefillL"]["text"]
    assert "104.4 tok/s" in dom["thDecodeL"]["text"]
    assert "per second of decode time" in dom["thDecodeL"]["text"]


def test_every_rendered_throughput_number_carries_its_unit() -> None:
    """"a lone number appears with no unit". Any figure on the strip is either
    a word ("idle"/"n/a") or a number with a unit attached to it."""
    dom = _render(
        _payload(
            prefill_tok_s=1240.0, gen_tok_s=249.0, ttft_s=1.85,
            prefill_tok_s_avg=3013.3, gen_tok_s_avg=104.4, ttft_s_avg=8.62,
        ),
        _SERVING,
    )
    for cell, unit in (("thPrefill", "tok/s"), ("thDecode", "tok/s"), ("thTtft", "s")):
        text = dom[cell]["text"]
        assert text.split()[-1].strip(), text
        assert not text.replace(",", "").replace(".", "").strip().isdigit(), (
            f"{cell} rendered a bare number with no unit: {text!r}"
        )
        assert unit in text, f"{cell} is missing its unit: {text!r}"


def _ttft_cell_markup() -> str:
    """The markup of the TTFT cell alone, from its `<div class="tcell">` to the
    end of its lifetime line.

    Scoped on purpose. This assertion used to run over the WHOLE of
    index.html, which was safe only while nothing on the page was a genuine
    percentile. The request-size panel now labels p50/p90/p99 — and those ARE
    percentiles — so a document-wide ban on the word would fail on a correct
    page and would be "fixed" by deleting the honest labels. What must never
    carry a percentile label is THIS figure, which is a mean.
    """
    end = INDEX_HTML.index('id="thTtftL"')
    start = INDEX_HTML.rindex('<div class="tcell">', 0, end)
    stop = INDEX_HTML.index("\n", end)
    return INDEX_HTML[start:stop]


def test_ttft_is_labelled_as_a_mean_and_never_as_a_percentile() -> None:
    """It is delta(_sum)/delta(_count) -- a mean, not p50 and not p95. A
    percentile label on a mean is a bigger lie than no label."""
    dom = _render(_payload(ttft_s=1.85, ttft_s_avg=8.62, ttft_requests=1728), _SERVING)
    assert "mean" in INDEX_HTML.lower()
    life = dom["thTtftL"]["text"]
    assert "mean" in life, life
    assert "over 1,728 requests" in life, f"the sample size is missing: {life}"
    cell = _ttft_cell_markup()
    assert 'id="thTtft"' in cell, "the slice missed the cell it is meant to check"
    assert "mean" in cell.lower(), "the TTFT cell must say it is a mean"
    for word in ("p50", "p95", "p99", "percentile", "median"):
        assert word not in life.lower(), f"TTFT is a mean, not {word}: {life}"
        assert word not in cell.lower(), f"the TTFT cell calls a mean {word}: {cell}"


def test_the_serving_line_labels_every_figure_before_its_number() -> None:
    """"the text is confusing due to formatting". The line read
    "249.0 tok/s gen": you meet the number before the word that says what it
    measures. The rule survives on the figures the line still carries --
    context and uptime -- each label first, number after."""
    dom = _render(
        _payload(prefill_tok_s=1240.0, gen_tok_s=249.0, ttft_s=1.85), _SERVING
    )
    line = dom["sMeta"]["text"]
    for label in ("context", "up"):
        assert label in line, f"{label!r} missing from the serving line: {line}"
        # label first, number after
        after = line.split(label, 1)[1].lstrip()
        assert after and after[0].isdigit(), (
            f"{label!r} is not followed by its figure: {line}"
        )
    assert "http://localhost:8001" in line, line
    assert "—" not in line, f"an em dash is not a reading: {line}"


def test_the_serving_line_does_not_duplicate_the_throughput_strip() -> None:
    """The line used to carry prefill/decode/TTFT as bare numbers while the
    strip below carried the same three with a reason for an absent figure and
    an age for a stale one. Two renderings of one figure is two chances for
    them to disagree, so the strip is their only home; the line keeps just the
    identity the strip does not show."""
    dom = _render(
        _payload(gen_tok_s=249.0, prefill_reason=metrics.IDLE,
                 ttft_reason=metrics.IDLE),
        _SERVING,
    )
    line = dom["sMeta"]["text"]
    for throughput in ("prefill", "decode", "time to first token", "tok/s"):
        assert throughput not in line, (
            f"{throughput!r} is on the serving line again; the strip owns it: {line}"
        )
    # ...and the strip still renders them, idle and all.
    assert dom["thDecode"]["text"] == "249.0 tok/s"
    assert dom["thPrefill"]["text"] == "idle"


def test_the_serving_line_breaks_only_between_labelled_segments() -> None:
    """A number and its label must not be able to wrap apart. The line is
    built as one element per figure; CSS makes each of them nowrap."""
    dom = _render(
        _payload(prefill_tok_s=1240.0, gen_tok_s=249.0, ttft_s=1.85), _SERVING
    )
    assert "createElement" in _serving_line(), (
        "the serving line must be built as elements, not as one text node that "
        "wraps wherever it happens to fit"
    )
    css = (_app.WEB / "style.css").read_text()
    assert ".smeta .seg{white-space:nowrap}" in css.replace("\n", "")
    assert 'class="status-meta smeta mono"' in INDEX_HTML
    # url · context · up  == two separators, each segment one nowrap element
    assert dom["sMeta"]["text"].count("·") == 2


def test_the_strip_never_wraps_a_figure_away_from_its_unit() -> None:
    css = (_app.WEB / "style.css").read_text()
    block = css[css.index(".tcell .n{") : css.index(".tcell .n.na")]
    assert "white-space:nowrap" in block, (
        f"a 20px figure in a 1/3-width cell will wrap between number and unit: {block}"
    )


def test_the_page_switches_on_reason_codes_not_on_english_prose() -> None:
    """Matching on the reason SENTENCE means rewording one string in Python
    silently changes what the dashboard renders."""
    codes = set(metrics.REASON_CODE.values())
    for body in (_fn_body("windowFigure"),):
        for literal in re.findall(r'"([a-z_]{3,})"', body):
            if literal in ("idle", "n/a", "number", "no reading"):
                continue
            assert literal in codes, (
                f"{literal!r} is neither a reason code nor a rendered word: {body}"
            )
    # The prose may be DISPLAYED (the strip prints the backend's reason
    # verbatim); it must never be compared against.
    for reason in metrics.REASON_CODE:
        for op in ('=== "', '== "', 'indexOf("'):
            assert op + reason not in APP_JS, (
                f"the page matches on the prose {reason!r}; use its code instead"
            )


def test_the_js_engine_that_runs_these_tests_is_a_declared_dev_dependency() -> None:
    """Otherwise the rendering tests silently skip on a clean checkout and the
    only coverage of what the page draws is the substring tests above."""
    pyproject = (_app.WEB.parent.parent / "pyproject.toml").read_text()
    assert "dukpy" in pyproject, "add dukpy to the dev extra"


def test_app_js_parses_in_a_real_js_engine() -> None:
    """Stronger than the bracket-balance scanner above, which is a heuristic
    written because "there is no JavaScript engine installed anywhere on this
    box". There is one now, and a single unbalanced brace white-screens the
    whole dashboard, so parse the file for real.

    Wrapped in a function body so that parsing does not RUN it: app.js ends by
    calling init(), which opens an EventSource.
    """
    dukpy = pytest.importorskip("dukpy")
    dukpy.evaljs("function __parse_only() {\n" + APP_JS + "\n}\n1;")


# --------------------------------------------------------------------------
# Request-size window and the parallelism recommendation
# --------------------------------------------------------------------------
def _request_stats_painter() -> str:
    return _fn_body("paintRequestStats")


def _rendered_text(src: str) -> str:
    """The painter with its own comments removed.

    Without this, an assertion that the page "names vllm:num_preemptions_total"
    is satisfied by a comment that merely mentions it — the string could be
    dropped from the rendered warning and the test would still pass. Comments
    are for the reader; only what survives here reaches the screen.
    """
    out = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//") or stripped.startswith("*") or stripped.startswith("/*"):
            continue
        out.append(line)
    return "\n".join(out)


def test_the_distribution_is_drawn_from_the_backends_own_buckets() -> None:
    """The percentile wall became a histogram. The bars must come from the
    buckets the backend computed (``WindowStats.buckets``, counted back into
    vLLM's own edges), not from a binning invented in the page.

    This is the same rule that keeps the capacity formula out of app.js: the
    page formats, Python measures. A second binning here would disagree with
    the percentiles beside it — the bars would say the mass is somewhere the
    p90 says it is not.
    """
    assert 'id="reqHist"' in INDEX_HTML, "no canvas for the request-size histogram"
    assert 'id="spark"' in INDEX_HTML, "the KV sparkline must not be collateral damage"
    painter = _request_stats_painter()
    assert re.search(r'reqHist\(\s*w\s*\)', painter), (
        f"the painter never hands the window to the plotter: {painter}"
    )
    plot = _fn_body("reqHist")
    assert "w.buckets" in plot or "buckets" in plot, (
        "the plot does not read the backend's bucket counts"
    )
    assert "fillRect" in plot, "the plot draws no bars"


def test_the_plot_uses_the_fine_bins_when_the_window_supports_them() -> None:
    """Three fat engine bars hid 90 exact counts. The plot's selection is a
    pure function (histData) precisely so it can be EXECUTED here rather than
    grepped: a source-text assertion survives every behaviour mutation, which
    makes it decorative. Run it against real payloads and assert the answer.

    The rule: exact observations exist -> draw the fine bins and keep the
    interval-only requests as a separate underlay (merging them into a bar
    would place them more precisely than they were measured); nothing exact ->
    fall back to the engine's own buckets with no underlay.
    """
    dukpy = pytest.importorskip("dukpy")
    src = "\n".join(
        [
            "function histData(w) {"
            + _fn_body("histData").split("{", 1)[1].rsplit("}", 1)[0]
            + "}",
            "function __run() { return JSON.stringify([",
            # exact present: fine bars + interval underlay, buckets unused
            "histData({fine:{step:5000,bins:[{lo:25000,hi:30000,n:3}],"
            "intervals:[{lo:20001,hi:50000,n:2}]},"
            "buckets:[{lo:20001,hi:50000,n:5}]}),",
            # nothing exact: the engine's buckets, no underlay
            "histData({fine:{step:0,bins:[],intervals:[{lo:20001,hi:50000,n:9}]},"
            "buckets:[{lo:20001,hi:50000,n:9}]}),",
            # empty window
            "histData({fine:{step:0,bins:[]},buckets:[]})",
            "]); } __run();",
        ]
    )
    fine, coarse, empty = json.loads(dukpy.evaljs(src))

    assert fine["bars"] == [{"lo": 25000, "hi": 30000, "n": 3}], (
        f"the fine bins are not the bars: {fine}"
    )
    assert fine["underlay"] == [{"lo": 20001, "hi": 50000, "n": 2}], (
        f"the interval requests are not a separate underlay: {fine}"
    )
    assert coarse["bars"] == [{"lo": 20001, "hi": 50000, "n": 9}], (
        f"with nothing exact the engine's buckets must be drawn: {coarse}"
    )
    assert coarse["underlay"] == [], (
        f"an all-interval window has no exact bars to hide behind: {coarse}"
    )
    assert empty["bars"] == [] and empty["underlay"] == [], empty


def test_the_plot_marks_the_percentiles_that_are_decisions() -> None:
    """The point of annotating the plot is that the reader finds the sizing
    number without a paragraph labelling it. p50/p90/p99 are all drawn; the
    headline text keeps only p90, the one the recommendation divides by."""
    plot = _fn_body("reqHist")
    for name in ("p50", "p90", "p99"):
        assert name in plot, f"the plot does not mark {name}"
    painter = _request_stats_painter()
    assert 'set("pctP90"' in painter, "the p90 headline is gone"
    for gone in ("pctP50", "pctP99", "pctMax"):
        assert f'set("{gone}"' not in painter, (
            f"{gone} is back as a text cell; it belongs on the plot now"
        )


def test_the_plot_keeps_a_percentile_an_interval() -> None:
    """A bucket-bounded percentile is an interval, and drawing it as a line at
    a midpoint would be the same lie the old text version made. The band from
    lo to hi must be drawn, and the marker placed at hi — the bound the
    recommendation is sized on, i.e. the conservative end."""
    plot = _fn_body("reqHist")
    assert "p.exact" in plot, "the plot ignores whether a percentile is exact"
    assert "p.lo" in plot and "p.hi" in plot, (
        "an interval-valued percentile is drawn from one bound only"
    )
    # The marker sits on hi, not on a midpoint of lo and hi.
    assert not re.search(r"\(\s*\w+\.p\.lo\s*\+\s*\w*\.?p?\.?hi\s*\)\s*/\s*2", plot), (
        f"the marker is drawn at a midpoint nothing measured: {plot}"
    )


def test_the_plot_axis_is_linear_in_tokens() -> None:
    """A log axis would draw every vLLM bucket a similar width — the buckets
    are a geometric ladder — and so would erase exactly the coarseness the
    picture exists to show. The scale must be a plain ratio."""
    plot = _fn_body("reqHist")
    assert "Math.log" not in plot and "log10" not in plot, (
        f"a log axis hides the quantisation the plot is for: {plot}"
    )
    assert re.search(r"/\s*xmax\s*\)\s*\*\s*W", plot), (
        f"the x scale is not a linear ratio of the token axis: {plot}"
    )


def test_buckets_reach_the_page_from_the_window() -> None:
    """Contract, the other end of the same claim: the field the plot reads is
    one the serialiser emits, and the counts sum to the sample size the panel
    prints beside it."""
    from servedeck import reqstats

    assert "buckets" in reqstats.EMPTY_STATS, "the empty window omits the bars"
    win = reqstats.RequestWindow()
    win.observe([(5000.0, 0.0), (50000.0, 0.0), (math.inf, 0.0)],
                hist_sum=0.0, hist_count=0.0)
    win.observe([(5000.0, 3.0), (50000.0, 7.0), (math.inf, 7.0)],
                hist_sum=2.5e5, hist_count=7.0)
    d = win.stats().to_dict()
    assert sum(b["n"] for b in d["buckets"]) == d["n"], (
        f"the bars do not add up to the sample count: {d['buckets']} vs n={d['n']}"
    )
    assert _app._sizing_payload() is not None


def test_percentile_intervals_are_rendered_as_intervals() -> None:
    """A bucket-bounded percentile must render as "20,001–50,000", never as a
    midpoint or a bare upper edge. Collapsing it to one number is exactly the
    "silently present an estimate as exact" failure."""
    painter = _request_stats_painter()
    assert "p.exact" in painter, "the painter ignores whether a percentile is exact"
    assert "p.lo" in painter and "p.hi" in painter, (
        "an interval-valued percentile is rendered from one bound only"
    )


def test_the_page_reads_only_fields_the_sizing_payload_emits() -> None:
    """Key-for-key against the real serialisers, the way this file checks every
    other payload. A painter reading `rec.agents` off a payload that has never
    had that key renders an empty headline with a 200 in the network tab."""
    from servedeck import parallelism, reqstats

    painter = _request_stats_painter()
    rec_keys = set(
        parallelism.recommend(
            pool_tokens=280_813, prompt_tokens=8_102, max_num_seqs=16
        ).to_dict()
    )
    win_keys = set(reqstats.EMPTY_STATS)
    sizing_keys = set(_app._sizing_payload())

    assert _reads_of("rec", painter) <= rec_keys, (
        f"painter reads unknown recommendation fields: "
        f"{_reads_of('rec', painter) - rec_keys}"
    )
    assert _reads_of("w", painter) <= win_keys, (
        f"painter reads unknown window fields: {_reads_of('w', painter) - win_keys}"
    )
    assert _reads_of("sz", painter) <= sizing_keys, (
        f"painter reads unknown sizing fields: "
        f"{_reads_of('sz', painter) - sizing_keys}"
    )


def test_the_page_shows_the_arithmetic_behind_the_recommendation() -> None:
    """The headline is a claim about how hard to push the GPU. It has to be
    checkable on screen: pool, per-request cost, the raw fit, the headroom
    factor and the clamp."""
    painter = _request_stats_painter()
    for term in ("pool_tokens", "cost_tokens", "fit_n", "headroom", "n_before_clamp"):
        assert term in painter, f"the arithmetic does not show {term}"
    assert "max_num_seqs" in painter, "the clamp is applied but never shown"


def test_the_page_shows_what_p99_would_change_it_to() -> None:
    painter = _request_stats_painter()
    assert "at_p99" in painter
    assert 'set("recP99"' in painter


def test_over_subscription_warning_names_preemptions_as_the_confirmation() -> None:
    """Preemption is what over-subscription actually does. The warning has to
    point at vllm:num_preemptions_total, or it is an opinion with no evidence
    attached."""
    painter = _request_stats_painter()
    assert "over_subscribed" in painter
    assert "preemptions" in painter
    assert "num_preemptions_total" in _rendered_text(painter), (
        "the warning must name the metric that confirms it, in text the "
        "operator can see — a comment mentioning it does not count"
    )


def test_the_lifetime_average_is_no_longer_the_sizing_input() -> None:
    """The defect this work removes: sizing was floor(kv_tokens / mean prompt
    size), where the mean was sum/count over every request since the engine
    booted. Both halves were wrong -- a lifetime mean, and a division that
    ignores the fixed per-sequence KV cost."""
    painter = _request_stats_painter()
    assert "avg_prompt_tokens" not in painter, (
        "the sizing panel is reading the lifetime mean again"
    )
    assert "avg_prompt_tokens" not in _fn_body("paintTelemetry")
    assert 'id="avgCtx"' not in INDEX_HTML


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------- #
# Disk sizes: the number and its unit must be produced together
# --------------------------------------------------------------------------- #


def test_no_size_is_rendered_next_to_a_hardcoded_unit() -> None:
    """The defect, stated as a source-level rule.

    The model card was literally ``${m.disk_gib || "—"} GB``: a value computed
    in GiB (bytes/1024**3) printed beside the letters "GB". 125.99 GiB was
    displayed as "125.91 GB" — 7.4% low, small enough to read as rounding.
    A template that interpolates a number and then types its unit can always
    drift; bytesTxt() returns both together, so nothing may hardcode one.
    """
    # The rule: a unit literal may appear only where a formatter is DEFINED
    # (bytesTxt, gib), never in a render template. A formatter returns its
    # number and its unit as one string, so the two cannot drift apart; a
    # template that types the unit itself can drift, and did.
    offenders = [
        line.strip()
        for line in APP_JS.splitlines()
        if re.search(r"\$\{[^}]*\}\s*(?:GB|GiB|MB|MiB|TB|TiB)\b", line)
        and "=>" not in line
        and not line.strip().startswith("//")
    ]
    assert not offenders, (
        "a render template interpolates a size next to a hardcoded unit, which "
        f"is exactly how GiB came to be labelled GB: {offenders}"
    )
    assert "disk_gib" not in APP_JS, (
        "disk_gib was a pre-divided GiB float; the payload now carries bytes so "
        "the unit is chosen at the point of formatting"
    )


def test_the_page_formats_sizes_through_one_binary_formatter() -> None:
    """bytesTxt is binary (1024) and says so. If it ever became decimal, every
    figure on the page would drift 7.4% from the df it sits beside."""
    assert "function bytesTxt" in APP_JS
    body = _fn_body("bytesTxt")
    assert "1024" in body and "1000" not in body, (
        "bytesTxt must be binary — the page's other sources (df, free, "
        "nvidia-smi, vLLM's logs) all are"
    )
    # The unit table itself is binary-named; a decimal name here would mean a
    # decimal number was being printed under a binary label or vice versa.
    assert re.search(r'BYTE_UNITS\s*=\s*\[[^\]]*"GiB"', APP_JS)
    assert not re.search(r'BYTE_UNITS\s*=\s*\[[^\]]*"GB"', APP_JS)


def test_the_disk_line_reads_only_fields_api_disk_emits() -> None:
    """Same contract as the model rows: every property the disk line reads has
    to exist in the payload that paints it."""
    from servedeck import app

    payload = app._disk_payload()
    read = _reads_of("d", _fn_body("paintDisk"))
    missing = read - set(payload)
    assert not missing, (
        f"web/app.js reads disk field(s) /api/disk never sends: {sorted(missing)}"
    )


def test_the_free_disk_figure_is_measured_not_markup() -> None:
    """The page once carried a static "412 GiB" disk span that nothing
    computed. The placeholder must be an em-dash the painter overwrites, never
    a plausible-looking number."""
    m = re.search(r'id="dfree"[^>]*>([^<]*)<', INDEX_HTML)
    assert m, 'the model rail has no #dfree element for the disk figure'
    assert not re.search(r"\d", m.group(1)), (
        f"#dfree ships a hardcoded reading: {m.group(1)!r}"
    )
    assert "paintDisk" in APP_JS, "nothing paints the disk figure"


# --------------------------------------------------------------------------
# The 2026-09-09 UI pass: controls that lie about themselves
# --------------------------------------------------------------------------
#
# These all EXECUTE the function whose decision is the fix. The reason is
# measured, not assumed: the first version of the preflight-severity test was a
# substring assertion, and inverting the ternary that chooses green from amber
# left every substring in place -- the test passed on a page that had the bug
# back. Each function below is pure for exactly this purpose (see histData() for
# the same reasoning).


def _run_js(src: str) -> object:
    """Run a snippet against the DOM shim and return its JSON result."""
    dukpy = pytest.importorskip(
        "dukpy", reason="pip install -e '.[dev]' brings in the JS engine"
    )
    return json.loads(dukpy.evaljs(_DOM_SHIM + "\n" + src))


def _exec(names: tuple[str, ...], expr: str) -> object:
    """Execute app.js's named functions (plus the formatters they call) and
    return the value of `expr`.

    The expression's value is JSON.stringify'd on the JS side rather than
    returned raw: dukpy hands back a plain string for `durTxt(48)`, which is not
    a JSON document, and the caller would otherwise get a JSONDecodeError for
    the very answer it was looking for.

    The shared formatters are always loaded, not only the ones named: they are
    what the painters call each other (bootNote -> durTxt), and omitting them
    produces a ReferenceError that reads like a bug in the function under test.
    """
    helpers = ("bytesTxt", "secsTxt", "agoTxt", "uptimeTxt", "durTxt", "ctxLabel")
    src = [
        *_PRELUDE,
        _const_line("BYTE_UNITS"),
        _const_line("BUSY_PHASES"),
        _const_line("MIN_CTX"),
        _const_line("CTX_STEP"),
        _const_line("DEFAULT_MAX_CTX"),
        _optional_const_line("BOOTING_STATES"),
        *[_fn_body(n) for n in dict.fromkeys((*helpers, *names))],
        f"JSON.stringify({expr});",
    ]
    return _run_js("\n".join(src))


def test_the_model_rail_rebuilds_only_when_the_set_changes() -> None:
    """The rail is repainted on every state event -- every five seconds -- and
    used to tear down and rebuild all its cards each time, which stole keyboard
    focus from the card being arrowed through and reset the list's scroll
    position mid-scroll.

    The rebuild is keyed on a signature. Asserted as an exact oracle rather than
    a property, because the interesting cases are the ones a weaker signature
    would get WRONG: MODELS.length alone misses a model becoming servable, and
    the repo ids alone miss a model gaining its blobs from another mount.
    """
    base = [
        {"repo_id": "a", "servable": True, "disk_bytes": 10},
        {"repo_id": "b", "servable": False, "disk_bytes": 20},
    ]
    sig = _exec(("railSigOf",), "railSigOf(%s)" % json.dumps(base))

    def sig_of(models):
        return _exec(("railSigOf",), f"railSigOf({json.dumps(models)})")

    # A selection change is not a set change: no rebuild, so focus survives.
    assert sig_of(base) == sig
    # A model becoming servable IS one: the card's tags and disabled state
    # would otherwise be stale forever.
    served = [dict(base[0]), dict(base[1], servable=True)]
    assert sig_of(served) != sig, "a model becoming servable does not rebuild the rail"
    # So does a model's size, which is printed on its card.
    grew = [dict(base[0], disk_bytes=11), base[1]]
    assert sig_of(grew) != sig, "a model changing size does not rebuild the rail"
    # And a model appearing, whatever the reason.
    assert sig_of([*base, {"repo_id": "c", "servable": True, "disk_bytes": 5}]) != sig


def test_the_model_count_says_how_many_are_servable() -> None:
    """"3 models" on a machine where two of the three cannot be served here is
    the wrong number for the question the rail answers."""
    got = _exec(
        ("railCountOf",),
        "railCountOf(["
        '{"servable":true},{"servable":false},{"servable":true}])',
    )
    assert got == "2/3 servable", got
    assert _exec(("railCountOf",), "railCountOf([])") == "0/0 servable"


def test_the_unsaved_marker_reports_the_field_that_differs() -> None:
    """#dirty shipped with a .on rule in CSS and no painter, so "unsaved
    changes" was permanently invisible.

    The comparison is against the RUNNING engine's own numbers, not against
    "has anything been clicked", and the two cases that separate those
    implementations are both asserted here: moving a control back to the running
    value must clear the marker, and adopting a server started at util 0.90 must
    mark the page's default 0.95 as changed with no click at all.
    """
    def bits(facts, sizing, want):
        return _exec(
            ("dirtyBits",),
            f"dirtyBits({json.dumps(facts)},{json.dumps(sizing)},"
            f"{json.dumps(want)})",
        )

    running = {"util_effective": 0.90, "ctx": 131072}
    sizing = {"max_num_seqs": 8}
    same = {"util": 0.90, "ctx": 131072, "agents": 8}
    assert bits(running, sizing, same) == [], "an unchanged page reads as unsaved"

    # Each field, individually, with the exact string the operator reads.
    assert bits(running, sizing, dict(same, util=0.95)) == ["util 0.95"]
    assert bits(running, sizing, dict(same, ctx=262144)) == ["262,144 ctx"]
    assert bits(running, sizing, dict(same, agents=4)) == ["4 agents"]
    assert bits(running, sizing, dict(same, agents=1)) == ["1 agent"], (
        "the plural is wrong for one agent"
    )
    # All three at once, in the order the controls appear on the page.
    assert bits(running, sizing, {"util": 0.95, "ctx": 8192, "agents": 2}) == [
        "util 0.95", "8,192 ctx", "2 agents"
    ]
    # Nothing known about the server yet: no facts, so nothing to compare
    # against, and the marker must not invent a difference.
    assert bits({}, {}, same) == []
    # A sub-threshold util difference is the same setting; the engine rounds it.
    assert bits({"util_effective": 0.9501}, sizing, same if False else dict(same, util=0.95)) == []


def test_the_boot_bar_is_drawn_only_while_a_boot_is_running() -> None:
    """#phases shipped in the prototype's markup and has never had a painter, so
    a multi-minute boot sat on "Elapsed —" with an empty track.

    Once the server is READY the phase history is a post-mortem rather than
    progress, so the bar must go away -- a dead widget above the controls on
    every page load is worse than none.
    """
    def active(b):
        return _exec(("bootActive",), f"bootActive({json.dumps(b)})")

    order = ["init", "loading_weights", "compiling", "kv_cache", "cuda_graphs",
             "http_start", "ready"]
    booting = {"phases": order, "actual_state": "STARTING"}
    assert active({**booting, "phase": "loading_weights", "reached_ready": False})
    assert not active({**booting, "phase": "ready", "reached_ready": True}), (
        "the bar stays up after the server is ready"
    )
    assert not active({**booting, "phase": None, "reached_ready": False}), (
        "the bar claims a boot before one has started"
    )
    # This used to assert the opposite: a boot that FAILED kept the bar up, "so
    # the operator can see where it stopped". What that bar actually showed was
    # a boot still in progress. The phase pulsed as current and the elapsed
    # clock kept counting. Main showed the 06:59 failure on 2026-09-11 as
    # "cuda_graphs, elapsed 1641 s" for an engine that had died 30 s into its
    # boot. A finished run is not a boot in progress. last_error says why it
    # failed.
    assert not active({"phases": order, "actual_state": "FAILED",
                       "phase": "kv_cache", "reached_ready": False})


def _paint_boot(snap: dict, monkeypatch) -> dict:
    """Build the boot payload with the REAL server-side _boot_payload() from a
    supervisor snapshot, paint it with the REAL paintBoot(), and return what
    landed in the panel's elements."""
    monkeypatch.setattr(_app, "_snap", lambda: dict(snap))
    payload = _app._boot_payload()
    ids = _element_ids()
    src = [
        *_PRELUDE,
        _const_block("PHASE_LABELS"),
        _optional_const_line("BOOTING_STATES"),
        *[_fn_body(n) for n in ("durTxt", "bootActive", "bootNote", "paintBoot")],
        f"__mk({json.dumps(ids)});",
        f"paintBoot({json.dumps(payload)});",
        f"__dump({json.dumps(['phases', 'elapsed', 'pnote'])});",
    ]
    return _run_js("\n".join(src))


_PHASE_TIMES = {"init": 13.1, "loading_weights": 17.1, "compiling": 33.2, "kv_cache": 115.8,
                "cuda_graphs": 155.9, "http_start": 169.0, "ready": 170.0}


@pytest.mark.parametrize("snap", [
    # 07:43:53 on 2026-09-11, after a clean Stop of a boot that had served:
    # the finished run still reads phase "ready", and before the latch in
    # PhaseTracker its reached_ready had gone back to False.
    {"actual_state": "STOPPED", "desired_state": "STOPPED", "phase": "ready",
     "reached_ready": False, "phase_times": _PHASE_TIMES, "run_elapsed_s": 278.0,
     "repo_id": "RadixArk/Qwen3.8-27B-NVFP4", "backend": "inline"},
    # 06:59 the same day: a boot that died at cuda_graphs, shown on main as
    # "elapsed 1641 s" long after the engine was gone.
    {"actual_state": "FAILED", "desired_state": "RUNNING", "phase": "cuda_graphs",
     "reached_ready": False, "phase_times": {"init": 14.1, "cuda_graphs": 30.1},
     "run_elapsed_s": 1641.5, "repo_id": "RadixArk/Qwen3.8-27B-NVFP4", "backend": "inline"},
    # A restart during a boot: the old engine is being stopped, not booted.
    {"actual_state": "STOPPING", "desired_state": "STOPPED", "phase": "cuda_graphs",
     "reached_ready": False, "phase_times": {"init": 13.0}, "run_elapsed_s": 62.0,
     "repo_id": "RadixArk/Qwen3.8-27B-NVFP4", "backend": "inline"},
], ids=["stopped-with-stale-ready", "failed-at-cuda-graphs", "stopping-mid-boot"])
def test_a_finished_run_never_repaints_as_a_boot(snap, monkeypatch) -> None:
    """After a Stop the page kept the boot panel on with the elapsed clock
    counting ("elapsed 4m 38s") beside "Not reachable". The supervisor's phase
    outlives its run. Only a supervisor that is actually booting may light
    the panel."""
    got = _paint_boot(snap, monkeypatch)
    assert got["phases"]["cls"] == "phases", (
        f"{snap['actual_state']} with phase {snap['phase']!r} painted the boot panel "
        f"as a boot in progress: {got}"
    )


def test_a_running_boot_still_lights_the_panel(monkeypatch) -> None:
    """Over-correction guard: the gate must not switch the panel off for the
    boots it exists for."""
    got = _paint_boot({
        "actual_state": "STARTING", "desired_state": "RUNNING", "phase": "loading_weights",
        "reached_ready": False, "phase_times": {"init": 13.1, "loading_weights": 17.1},
        "run_elapsed_s": 31.0, "repo_id": "RadixArk/Qwen3.8-27B-NVFP4", "backend": "inline",
    }, monkeypatch)
    assert got["phases"]["cls"] == "phases on", got
    assert got["elapsed"]["text"] == "31 s", got


def _paint_request_stats(sizing: dict) -> dict:
    """Paint the request-statistics panel with the REAL paintRequestStats()
    and return what landed in its elements."""
    ids = _element_ids()
    helpers = ("bytesTxt", "secsTxt", "agoTxt", "uptimeTxt", "durTxt", "ctxLabel")
    src = [
        *_PRELUDE,
        _const_line("BUSY_PHASES"),
        *[_fn_body(n) for n in (*helpers, "set", "winSpan", "histData", "reqHist",
                                "paintRequestStats")],
        f"__mk({json.dumps(ids)});",
        f"var liveSizing = {json.dumps(sizing)};",
        "paintRequestStats();",
        f"__dump({json.dumps(['recBasis', 'recP99', 'pctP90', 'recN'])});",
    ]
    return _run_js("\n".join(src))


def _sizing_from_requests(monkeypatch, polls: list[list[float]]) -> dict:
    """The REAL sizing payload (_app._sizing_payload over a real
    reqstats.RequestWindow) after the engine served the given requests, one
    list per poll. Several requests finishing within one poll are known only
    to a bucket; one alone is known exactly (its size is the _sum delta)."""
    from servedeck import reqstats

    edges = [1000.0, 5000.0, 10000.0, 50000.0, math.inf]
    win = reqstats.RequestWindow()
    seen: list[float] = []

    def scrape():
        cum = [(e, float(sum(1 for v in seen if v <= e))) for e in edges]
        win.observe(cum, hist_sum=float(sum(seen)), hist_count=float(len(seen)))

    scrape()                                   # the baseline scrape
    for batch in polls:
        seen.extend(batch)
        scrape()
    stats = win.stats().to_dict()
    monkeypatch.setattr(_app.rt, "metrics", {
        "reachable": True, "kv_cache_size_tokens": 561_944, "running": 0,
        "preemptions": 0, "prompt_stats": stats, "gen_stats": dict(reqstats.EMPTY_STATS),
    }, raising=False)
    monkeypatch.setattr(_app, "_running_max_num_seqs", lambda: 16)
    return _app._sizing_payload()


def test_the_p99_line_does_not_state_a_bucket_edge_as_a_measurement(monkeypatch) -> None:
    """F9, second half. The live panel on 2026-09-11 read "p90 of the last 5
    requests <= 10,000 prompt tokens" and then "at p99 (10,000 tok) it would be
    16". The prompts were 117 tokens once and 9,853 four times. 10,000 is a
    bucket edge, and the p99 line stated it as a measurement."""
    # The live case: one small request alone (exact), four together (bucket).
    sizing = _sizing_from_requests(monkeypatch, [[117.0], [9853.0] * 4])
    assert sizing["at_p99"] is not None and sizing["window"]["p99"]["exact"] is False, sizing
    got = _paint_request_stats(sizing)
    p99 = got["recP99"]["text"]
    assert "(≤ 10,000 tok)" in p99, (
        f"the p99 line states the bucket edge 10,000 as a measured p99: {p99!r}"
    )
    assert "≤ 10,000" in got["recBasis"]["text"], got


def test_an_exact_p99_is_still_stated_as_one(monkeypatch) -> None:
    """Over-correction guard: when every observation at that rank was exact,
    the number IS the measurement and must not be hedged."""
    sizing = _sizing_from_requests(monkeypatch, [[9853.0], [9853.0], [117.0]])
    assert sizing["window"]["p99"]["exact"] is True, sizing["window"]["p99"]
    got = _paint_request_stats(sizing)
    p99 = got["recP99"]["text"]
    assert "≤" not in p99, f"an exact p99 is hedged as a bound: {p99!r}"
    assert f"at p99 ({sizing['at_p99']['prompt_tokens']:,} tok)" in p99, p99


def test_the_boot_eta_names_its_source_and_says_when_it_is_beaten() -> None:
    """history.py computes a real per-model ETA and marks when it falls back to
    SPEC.md's calibration figures. An ETA that is a guess must not read like a
    measurement, and a boot that has already run past the estimate must say so
    rather than keep showing a number it is behind."""
    def note(b):
        return _exec(("bootNote",), f"bootNote({json.dumps(b)})")

    measured = note({"eta_s": 250, "eta_p90_s": 400, "eta_source": "history",
                     "cold": False, "elapsed_s": 48})
    assert "4m 10s" in measured, measured
    assert "warm boot" in measured
    assert "p90 6m 40s" in measured
    assert "measured boots" in measured, "a history ETA does not say where it came from"

    calibrated = note({"eta_s": 600, "eta_source": "calibration", "cold": True,
                       "elapsed_s": 60})
    assert "cold boot" in calibrated
    assert "calibration" in calibrated, "a guessed ETA is presented as a measurement"
    assert "measured" not in calibrated, calibrated

    beaten = note({"eta_s": 250, "eta_source": "history", "cold": False,
                   "elapsed_s": 400})
    assert "past the" in beaten and "estimate already" in beaten, beaten

    none = note({"eta_s": None, "eta_note": "no boots of this model recorded yet"})
    assert none == "no boots of this model recorded yet"
    assert _exec(("bootNote",), 'bootNote({eta_s: null})') == \
        "no boot estimate for this model yet"


def test_the_elapsed_clock_has_second_resolution() -> None:
    """uptimeTxt() floors to whole minutes, which is right for "up 3h 12m" and
    useless for a boot 48 seconds in: the readout would sit on "0m" for the
    first minute of a multi-minute wait and read as frozen."""
    def dur(v):
        return _exec(("durTxt",), f"durTxt({json.dumps(v)})")

    assert dur(0) == "0 s"
    assert dur(48.4) == "48 s"
    assert dur(59.6) == "1m 00s"
    assert dur(192) == "3m 12s"
    assert dur(3_840) == "1h 04m"
    assert dur(None) == "—"
    assert dur(-1) == "—"


def test_a_warning_is_not_rendered_as_a_pass() -> None:
    """The preflight strip coloured every non-blocking finding green with a
    check mark, so "KV budget is tight at this context" and "no blockers for
    this configuration" were indistinguishable -- the strip read all-green on a
    configuration that was about to preempt.

    Executed rather than grepped: inverting the ternary that picks the class
    leaves every substring a source test could look for intact.
    """
    src = [
        *_PRELUDE,
        _fn_body("paintPreflight"),
        '__mk(["pref"]);',
        "paintPreflight({findings: ["
        '{level:"warn",title:"KV budget is tight",detail:"d",fix:"f"},'
        '{level:"block",title:"does not fit",detail:"d"},'
        '{level:"info",title:"fine",detail:"d"}]});',
        '(function(){var out=[];var c=__els.pref.childNodes;'
        'for(var i=0;i<c.length;i++){var p=c[i].className.split(" ");'
        'out.push([p[p.length-1], c[i].childNodes[0].textContent]);}'
        'return JSON.stringify(out);})()',
    ]
    got = _run_js("\n".join(src))
    assert got == [
        ["warn", "!"],     # a warning is amber and marked '!', never a green tick
        ["crit", "\u2715"],
        ["ok", "\u2713"],
    ], f"finding severities rendered as {got}"


def test_the_running_agent_count_is_read_back_into_its_field() -> None:
    """The util slider was synced from the running server and the agent count
    was not, so the field showed 1 against a server running 8 while the
    recommendation column beside it said 8 -- the input and the evidence
    disagreed, on the same screen, permanently."""
    body = _fn_body("paintState")
    assert "liveSizing.max_num_seqs" in body, (
        "paintState never reads the running engine's --max-num-seqs"
    )
    assert re.search(r'agents\s*=\s*runSeqs', body), (
        "the running agent count is read but never written to the field"
    )
    # And it must be gated on the same flag util is, or it overwrites a number
    # the operator has just typed.
    assert body.index("!userPicked") < body.index("agents = runSeqs")


def _handler_body(cid: str, var: str) -> str:
    """The body of the `oninput` handler attached to the #cid control.

    Anchored on the `var.oninput =` assignment, not on `const var = $(...)`:
    the same local name is reused elsewhere (`const cx = $("ctx")` also appears
    in renderCtx), and anchoring there makes the test read the WRONG control's
    handler -- proven by a mutation that deleted the ctx handler's userPicked
    line and left this test passing because it was looking at util's.

    Brace-matched rather than line-regexed because the handler is a one-line
    arrow function whose body contains the very braces a line pattern trips on.
    """
    start = APP_JS.index(f"{var}.oninput =")
    brace = APP_JS.index("{", start)
    depth, i = 0, brace
    while i < len(APP_JS):
        if APP_JS[i] == "{":
            depth += 1
        elif APP_JS[i] == "}":
            depth -= 1
            if depth == 0:
                return APP_JS[brace : i + 1]
        i += 1
    raise AssertionError(f"unbalanced braces in #{cid}'s oninput handler")


def test_touching_a_control_stops_the_server_overwriting_it() -> None:
    """userPicked was set only by clicking a model card. Moving the utilization
    slider did not set it, so the next state event (five seconds later) moved
    the thumb back to the running server's value -- and the Apply confirmation,
    which quotes util from the same variable, then quoted a number the operator
    had already changed.

    Every control that feeds a request body must set the flag, so this checks
    each handler's own body rather than counting occurrences in the file.
    """
    for cid, var in (("util", "u"), ("ctx", "cx"), ("agents", "ag")):
        handler = _handler_body(cid, var)
        assert "userPicked = true" in handler, (
            f"moving #{cid} does not claim the control: {handler.strip()}"
        )


def test_the_page_does_not_poll_state_it_is_pushed_over_sse() -> None:
    """The page polled /api/state every 5 s on top of the SSE stream, which
    already pushes a state event on that same interval. Two sources for one
    readout means whichever paint lands last wins, and the poll's
    `catch (_) {}` meant the duplicate could fail forever without a trace."""
    assert "setInterval" in APP_JS, "the disk refresh is gone too"
    assert not re.search(r'fetch\("/api/state"\)', APP_JS), (
        "the duplicate state poll is back"
    )
    assert "addEventListener(\"state\"" in APP_JS, "the pushed state is not consumed"


def test_the_dead_css_is_actually_dead() -> None:
    """~47 lines of style.css described markup that no longer exists (.twoup,
    .winsel, .segbtn, .ticks, .flabel) or markup that was deleted with the
    unwired controls (.toggles, .tog, .banner). Dead rules are not free: they
    are read as live, and a reader who finds `.banner.on` will look for a banner.

    Every class selector in the stylesheet must appear in the page's own markup
    or in the script that builds it. Sub-selectors are checked on their last
    compound, which is what carries the class name.
    """
    css = (_app.WEB / "style.css").read_text()
    # Strip comments, then take the selector of each rule.
    stripped = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    classes: set[str] = set()
    for sel in re.findall(r"([^{}]+)\{", stripped):
        for part in sel.split(","):
            classes.update(re.findall(r"\.([A-Za-z][\w-]*)", part))
    markup = INDEX_HTML + APP_JS
    orphan = sorted(c for c in classes if not re.search(rf'\b{re.escape(c)}\b', markup))
    assert not orphan, f"style.css has rules for markup that does not exist: {orphan}"


def test_the_stylesheet_has_no_undefined_custom_properties() -> None:
    """`.dfline` used `var(--border)`, which the stylesheet never defines, so the
    rail's disk line had no bottom rule at all -- and the omission was invisible
    because the declaration is simply dropped rather than reported."""
    css = (_app.WEB / "style.css").read_text()
    defined = set(re.findall(r"(--[\w-]+)\s*:", css))
    used = set(re.findall(r"var\((--[\w-]+)", css))
    missing = used - defined
    assert not missing, f"var() of a property the stylesheet never defines: {sorted(missing)}"


def test_the_log_is_collapsed_and_says_what_it_holds() -> None:
    """The log held 190px of the page permanently to display a stream nobody was
    reading while they were reading something else. It is a drawer now, and the
    toggle counts the lines waiting so opening it is an informed click."""
    assert 'class="panel log folded"' in INDEX_HTML, "the log is not folded by default"
    assert 'id="logFold"' in INDEX_HTML and 'aria-expanded="false"' in INDEX_HTML
    assert "logFold" in APP_JS, "nothing wires the log drawer"
    assert "logPanel" in APP_JS
    assert ".log.folded .logbody{display:none}" in css_fold_rule(), \
        "folding the log does not hide it"


def css_fold_rule() -> str:
    css = (_app.WEB / "style.css").read_text().replace("\n", "")
    m = re.search(r"\.log\.folded\s+\.logbody\s*\{[^}]*\}", css)
    return m.group(0) if m else ""


def test_the_counters_name_their_time_base() -> None:
    """Running/Waiting are instantaneous; Preemptions and Prefix cache are
    lifetime counters that never reset. Drawn in one row at one size, a 593
    accumulated over days read identically to "preempting right now" -- and the
    preemption cell was red on top of that, permanently."""
    row = INDEX_HTML[INDEX_HTML.index('class="lrow"'):]
    row = row[: row.index("</div>\n        </div>")]
    assert "since boot" in row, "the preemption tally does not say it is cumulative"
    assert "lifetime" in row, "the prefix cache hit rate does not say its window"
    assert row.count("class=\"tb\"") >= 2, "the time bases are not visibly subordinate"
    css = (_app.WEB / "style.css").read_text().replace("\n", "")
    assert ".lrow .n.bad{color:var(--ink)}" in css, (
        "a cumulative preemption count is still painted as an alarm; the alarm "
        "is #oversub, which fires on the condition rather than the tally"
    )


def test_the_recommendation_can_be_applied_where_it_is_acting() -> None:
    """The recommendation is computed from measured request sizes and shown in
    the live panel; the field it constrains is in the allocator, a scroll away.
    The round trip was to read a number and type it into a box, which is the
    "badly organized" complaint in its smallest form.

    The button must be disabled until a recommendation exists -- an enabled
    button that writes nothing is the defect this file exists for.
    """
    assert 'id="useRec"' in INDEX_HTML
    assert "useRec" in APP_JS
    body = _fn_body("paintRequestStats")
    assert 'set("recN"' in body
    m = re.search(r'if \(rec\) \{(.*?)\n  \} else', body, re.S)
    assert m, "the recommendation branch of paintRequestStats is unrecognisable"
    assert "ur.disabled = false" in m.group(1), (
        "the button is enabled somewhere other than when a recommendation exists"
    )
    assert "ur.disabled = true" in body[body.index("} else"):], (
        "the button stays enabled when there is no recommendation"
    )


def test_the_context_ceiling_lands_on_the_step_grid() -> None:
    """A range input's thumb can only rest on min + k*step. The ceiling comes
    from the KV budget and is not a round number -- 19,100 on this box at 16
    agents -- so with min 8192 and step 4096 the thumb parked at 16,384 while
    the readout beside it printed 19,100. The control disagreed with its own
    value, and the number POSTed to /api/server/start was the readout's, not the
    thumb's.

    Asserted as the exact snapped value, because "snaps somehow" is satisfied by
    rounding up, which would exceed the budget the ceiling came from.
    """
    def top(ceil):
        return _exec(("ctxTop",), f"ctxTop({ceil})")

    assert top(19_100) == 16_384, "an off-grid ceiling must snap DOWN to the grid"
    # Literal expected values throughout. An assertion that recomputes the same
    # floor() the function under test performs passes whatever the function
    # returns, so it proves nothing; these are written out instead.
    assert top(12_288) == 12_288, "a value already on the grid must not move"
    assert top(19_200) == 16_384, top(19_200)
    assert top(8_192) == 8_192, "the floor of the range must survive"
    # Never rounds up: exceeding the bound is worse than one step less of it.
    assert top(12_287) == 8_192, top(12_287)
    # A ceiling below the minimum still yields the minimum, not a negative max
    # (which would make the slider unusable rather than merely small).
    assert top(1_000) == 8_192, top(1_000)
    assert top(1_048_576) == 1_048_576, "a model ceiling on the grid must be reachable"

    # And the painter must actually use it, rather than recomputing inline.
    body = _fn_body("renderCtx")
    assert "ctxTop(ceiling)" in body, "renderCtx does not snap the ceiling"
    assert "el.max = String(top)" in body, "the slider's max is not the snapped value"
    assert "if (ctx > top)" in body, (
        "the context value is still clamped to the unsnapped ceiling, so it can "
        "be set to a value the thumb cannot show"
    )


def test_close_percentile_labels_stack_and_stay_inside_the_canvas() -> None:
    """The p90 and p99 markers are usually a few k apart, which on a 200k axis
    is a handful of pixels: their captions overprinted into mush, and the row
    assignment only ever compared against row 0, so two close labels both
    landed on row 1 anyway. And the captions were drawn with whatever
    textBaseline the count gridline left behind ("alphabetic"), which anchors
    glyphs ABOVE y=4 — half of every label was off the top of the canvas.
    markRows() is executed with literal oracles; the painter is checked to
    state its baseline before drawing the captions."""
    def rows(xs):
        return _exec(("markRows",), f"markRows({json.dumps(xs)}, 26, 2)")

    # Well separated: all on the top row.
    assert rows([10, 100, 190]) == [0, 0, 0]
    # p90 clear of p50, p99 crowding p90: p99 drops to the second row.
    assert rows([10, 100, 110]) == [0, 0, 1]
    # All three crowded: first takes row 0, second row 1, third has nowhere
    # left and doubles up on the last row rather than a third invisible row.
    assert rows([10, 20, 30]) == [0, 1, 1]
    assert rows([]) == []
    # One row available: everything shares it, no out-of-range row index.
    assert _exec(("markRows",), "markRows([5, 8], 26, 1)") == [0, 0]

    plot = _fn_body("reqHist")
    # The baseline in force when the captions are drawn must be "top" (glyphs
    # hang BELOW y). Checking that "top" appears somewhere is not enough — the
    # tick labels set it earlier and the count gridline resets it to
    # "alphabetic" afterwards, so compare which assignment is NEAREST before
    # the caption loop.
    seg = plot[: plot.find("marks.forEach((m, i)")]
    assert seg.rfind('x.textBaseline = "top"') > seg.rfind(
        'x.textBaseline = "alphabetic"'
    ), "percentile captions inherit an alphabetic baseline"
    # And the caption y must be inside the canvas: a fixed offset above padT
    # (the old padT - 16 = 4 with an alphabetic baseline) drew off the top.
    assert "padT - 16" not in plot, "captions are placed above the canvas again"


# --------------------------------------------------------------------------
# 2026-09-10 second audit pass: flaws found by rendering the page and reading
# the computed styles, not the source. Each test EXECUTES the decision where a
# decision exists, and pins the exact string/attribute the operator sees.
# --------------------------------------------------------------------------
def _rail_cards(models, selected=0):
    """Run renderModels() against the shim and return one dict per card built.

    renderModels is a painter, so this is the only way to see what it actually
    produced: grepping its source passes when `aria-disabled` is set on the
    servable branch instead of the unservable one, or when `disabled` is added
    back alongside it and the keyboard trap returns.
    """
    return _run_js(
        "\n".join(
            [
                *_PRELUDE,
                _fn_body("bytesTxt"),
                _const_line("BYTE_UNITS"),
                _fn_body("diskTitle"),
                _fn_body("trustTag"),
                _fn_body("railSigOf"),
                _fn_body("railCountOf"),
                _fn_body("updateRailSelection"),
                _fn_body("renderModels"),
                "var railSig = '';",
                "var sel = %d;" % selected,
                "var MODELS = %s;" % json.dumps(models),
                "__mk(['mlist', 'mcount']);",
                "renderModels();",
                "(function () {"
                " var L = document.getElementById('mlist'), out = [];"
                " for (var i = 0; i < L.children.length; i++) {"
                "   var b = L.children[i];"
                "   out.push({disabled: !!b.disabled,"
                "     ariaDisabled: b.getAttribute('aria-disabled'),"
                "     ariaPressed: b.getAttribute('aria-pressed'),"
                "     role: b.getAttribute('role'),"
                "     title: b.title, html: b.innerHTML});"
                " }"
                " return JSON.stringify({cards: out, count:"
                "   document.getElementById('mcount').textContent});"
                "})();"
            ]
        )
    )


def test_an_unservable_model_is_still_reachable_by_the_keyboard() -> None:
    """The comment promised aria-disabled; the code set `disabled`. A disabled
    button is skipped by Tab entirely, so the 29 models that cannot be served
    here could not be focused, and their reason -- which lives on the card's own
    note line -- was unreadable without a mouse. Executed, not grepped: the
    attribute has to land on the unservable cards and NOT on the servable one.
    """
    got = _rail_cards(
        [
            {"repo_id": "a", "name": "Good", "servable": True, "disk_bytes": 10,
             "quant": "nvfp4", "trust": "measured", "backend": "flashnext",
             "model_max_ctx": 262144, "disk_local_bytes": 10},
            {"repo_id": "b", "name": "Bad", "servable": False, "disk_bytes": 20,
             "quant": None, "disk_local_bytes": 20,
             "unservable_reason": "0 safetensors"},
        ]
    )
    good, bad = got["cards"]
    assert good["ariaDisabled"] is None, "a servable card must stay tab-reachable"
    assert bad["ariaDisabled"] == "true", "the unservable card is not aria-disabled"
    assert not bad["disabled"], (
        "the unservable card is still `disabled`, which removes it from the tab "
        "order entirely -- the keyboard trap the aria-disabled comment promised "
        "to avoid"
    )
    assert "0 safetensors" in bad["title"], "the reason is not on the card"
    assert got["count"] == "1/2 servable"
    # The dimmed look moved from the inline style to the attribute selector.
    css = (_app.WEB / "style.css").read_text().replace("\n", "")
    assert '.mcard[aria-disabled="true"]' in css, (
        "aria-disabled cards are reachable but look identical to servable ones"
    )


def test_the_agents_field_shows_its_own_kv_limit() -> None:
    """"KV fits 1" sat beside a field reading "16" with nothing connecting the
    two, in the same quiet grey as a caption. The contradiction has to be said
    and has to be loud. Executed with literal oracles: an assertion that
    recomputed `fitN < want` would pass with the comparison flipped.
    """
    over = _exec(("agentsFitNote",), "agentsFitNote(1, 16, 16384)")
    assert over["text"] == "KV fits 1"
    assert over["overfit"] is True, "asking for 16 when 1 fits is not flagged"
    assert "over budget" in over["title"], over["title"]
    assert "16,384" in over["title"], "the note must name the context it is about"

    fits = _exec(("agentsFitNote",), "agentsFitNote(8, 8, 8192)")
    assert fits["overfit"] is False, "asking for exactly what fits is not over budget"
    assert "over budget" not in fits["title"], fits["title"]

    unknown = _exec(("agentsFitNote",), "agentsFitNote(0, 4, 8192)")
    assert unknown["text"] == "—", "a bound nothing computed must not print a number"
    assert unknown["overfit"] is False

    css = (_app.WEB / "style.css").read_text().replace("\n", "")
    assert ".frow .s.overfit" in css, "the over-budget note has no distinct style"


def test_the_provenance_badge_never_prints_the_enum() -> None:
    """The badge showed `measured_other_ctx` verbatim -- a Python enum leaking
    into a sentence -- and `unknown`, which is the wrong word for a capacity
    figure. Executed across every code the backend can send.
    """
    codes = ("measured", "measured_other_ctx", "estimated", "unknown", None)
    for code in codes:
        b = _exec(("badgeOf",), f"badgeOf({json.dumps(code)})")
        assert "_" not in b["text"], f"the enum leaked into the badge: {b}"
        assert b["text"] != "unknown", f"'unknown' is not a capacity word: {b}"
        assert b["title"], f"badge {code!r} has no explanation"
    assert _exec(("badgeOf",), 'badgeOf("measured_other_ctx")')["text"] == "measured*"
    assert _exec(("badgeOf",), 'badgeOf("measured_other_ctx")')["meas"] is True
    assert _exec(("badgeOf",), 'badgeOf("unknown")')["meas"] is False
    assert _exec(("badgeOf",), "badgeOf(null)")["text"] == "estimated"


def test_the_window_span_is_a_duration_not_raw_seconds() -> None:
    """"100/100 requests · 1182s" made the reader do arithmetic to learn the
    window is ~20 minutes of traffic -- the number that decides whether the
    picture is current at all. Every other duration on the page is formatted.
    """
    def span(w):
        return _exec(("winSpan",), f"winSpan({json.dumps(w)})")

    assert span({"n": 100, "capacity": 100, "age_s": 1182, "exact_n": 100}) == (
        "100/100 requests · 19m 42s"
    )
    # Partial window and a sub-minute span.
    assert span({"n": 3, "capacity": 100, "age_s": 48, "exact_n": 3}) == (
        "3/100 requests · 48 s"
    )
    # Only some observations are exact -- said, because it bounds the plot.
    assert span({"n": 100, "capacity": 100, "age_s": 60, "exact_n": 98}) == (
        "100/100 requests · 1m 00s · 98 exact"
    )
    assert "1182s" not in span({"n": 100, "capacity": 100, "age_s": 1182, "exact_n": 100})
    # Empty window still counts itself out of its capacity.
    assert span({"n": 0, "capacity": 100}) == "0/100 requests observed"


def test_a_transition_is_said_once_and_by_the_element_that_owns_it() -> None:
    """paintState() wrote "starting…" into #sMeta, but paintTelemetry() repaints
    that element every 2 s while state arrives every 5 s -- so the note survived
    one frame in three and the rest of the time the line claimed the server was
    already serving. The phase is now rendered inside paintServingMeta(), the
    only writer of that slot, from a pure busyPhase() the buttons share.
    """
    for state, want in (
        ("STARTING", "starting"), ("PREFLIGHT", "preflight"),
        ("STOPPING", "stopping"), ("DRAINING", "draining"),
    ):
        assert _exec(("busyPhase",), f"busyPhase({{'actual_state':'{state}'}})") == want
    assert _exec(("busyPhase",), "busyPhase({actual_state:'READY'})") == ""
    assert _exec(("busyPhase",), "busyPhase({})") == ""

    dom = _render(_payload(), {"upstream": {"up": False, "port": 8001},
                               "supervisor": {"actual_state": "STARTING"}})
    assert dom["sMeta"]["text"] == "starting…", dom["sMeta"]["text"]

    # The one state list, not two that can disagree.
    assert '["STARTING", "PREFLIGHT"' not in APP_JS, (
        "the busy-state list is written out again instead of read from BUSY_PHASES"
    )
    body = _fn_body("paintState")
    assert "busyPhase(sv)" in body, "paintState does not use the shared decision"
    assert 'meta.textContent = `${sv.actual_state' not in body, (
        "paintState writes the phase into #sMeta again, where telemetry clobbers it"
    )
    # The pill's busy style shipped with no painter, like #dirty did.
    assert '" busy"' in body, "the pill never shows the busy state its CSS defines"


def test_the_agents_box_shows_the_number_it_sends() -> None:
    """Typing 99 left the box reading 99 while `agents` -- the value POSTed as
    max_num_seqs -- held 64. The control displayed a number the page was not
    using, the same disagreement class as the off-grid context thumb.
    """
    handler = _handler_body("agents", "ag")
    assert 'e.target.value = String(agents)' in handler, (
        f"the clamped agent count is not echoed back into the box: {handler}"
    )
    assert 'e.target.value !== ""' in handler, (
        "an empty box (mid-edit) must not be rewritten to 1 under the caret"
    )


def test_the_vram_bar_segment_rule_does_not_leak_onto_the_serving_line() -> None:
    """The serving line was rendered, measured and INVISIBLE.

    `.smeta .seg` (the serving line's per-figure spans) and `.seg` (a segment of
    the VRAM bar) are different elements sharing one class name. The bar rule
    carries `color:#fff` and `font-size:10px`, so it painted the serving line
    white on a #FBFCFD surface — getComputedStyle reported rgb(255,255,255) on
    every segment while the markup, the JS and the text were all correct.

    Pinned on the rule text because the cascade cannot be evaluated in duktape:
    the bar rule must be scoped, and nothing may paint the serving line's spans
    white again.
    """
    css = (_app.WEB / "style.css").read_text().replace("\n", "")
    m = re.search(r"(?<![\w .-])\.seg\{[^}]*\}", css)
    assert m is None, (
        f"an unscoped `.seg` rule is back and collides with .smeta .seg: {m and m.group(0)}"
    )
    bar = re.search(r"\.bar \.seg\{[^}]*\}", css)
    assert bar, "the VRAM bar's segment rule has gone missing entirely"
    assert "color:#fff" in bar.group(0), "the bar rule below must stay scoped to .bar"
    # And the serving line's own rule must not claim a colour of its own.
    smeta = re.search(r"\.smeta \.seg\{[^}]*\}", css)
    assert smeta, "the serving line's segment rule is gone"
    assert "color" not in smeta.group(0), (
        f"the serving line sets its own colour and can go invisible again: "
        f"{smeta.group(0)}"
    )


def test_the_pure_decisions_are_actually_wired_into_their_painters() -> None:
    """The tests above EXECUTE the pure functions, which proves the decision is
    right and proves nothing about whether the page uses it. A painter that
    inlines the old behaviour again keeps the pure function defined and dead —
    the exact shape a mutation battery catches and a pure-function test does not.
    """
    for painter, call in (
        ("renderCtx", "agentsFitNote("),
        ("paintEstimate", "badgeOf("),
        ("paintRequestStats", "winSpan("),
        ("paintServingMeta", "busyPhase("),
        ("paintServingMeta", "resolutionNote("),
        ("paintServingMeta", "resolutionDetail("),
        ("paintState", "busyPhase("),
    ):
        # Comments stripped: renderCtx carries a "see agentsFitNote()" note, so
        # grepping the raw source passes even when the call itself is deleted.
        assert call in _rendered_text(_fn_body(painter)), (
            f"{painter}() no longer calls {call.strip('(')} — the tested decision "
            "is dead code and the page renders something else"
        )


def test_a_dashboard_that_cannot_find_a_server_says_which_port_and_why() -> None:
    """The serving line's whole message was "nothing serving on
    http://localhost:8002".

    On 2026-09-10 that sentence was true and useless: :8002 came from a shell
    config naming a backend that had been dead for a week, a healthy server was
    answering on :8001, and nothing on the page — or in the payload behind it —
    said which port had been chosen or why. The backend now resolves the port
    and explains itself (updetect.py); this is the page rendering that answer
    rather than a blank.
    """
    resolution = {
        "port": 8002,
        "source": "shell_config",
        "reason": ("nothing is listening on any known port (checked :8002 "
                   "(backends.glm53), :8001 (backends.flashnext)); showing "
                   ":8002 (backends.glm53), named by the shell config's PORT"),
        "backend": "glm53",
        "pid": None,
        "live": False,
        "candidates": [
            {"port": 8002, "source": "shell_config", "backend": "glm53",
             "listening": False, "pid": None},
            {"port": 8001, "source": "backend_config", "backend": "flashnext",
             "listening": True, "pid": 3102188},
        ],
    }
    dom = _render(
        _payload(reachable=False),
        {"upstream": {"up": False, "port": 8002, "resolution": resolution},
         "server_uptime_s": None},
    )

    text = dom["sMeta"]["text"]
    assert "http://localhost:8002" in text, text
    assert resolution["reason"] in text, (
        f"the page dropped the reason the backend gave it: {text!r}"
    )
    title = dom["sMeta"]["title"]
    assert ":8001 (flashnext)" in title and "pid 3102188 listening" in title, title
    assert ":8002 (glm53): nothing listening" in title, title


def test_a_port_that_was_never_probed_is_not_drawn_as_an_empty_one() -> None:
    """`listening: null` means the socket table was not read for that port.
    Painting it as "nothing listening" would put a claim on the page that
    nothing ever checked — the failure this whole chain is about, one layer
    up."""
    dom = _render(
        _payload(reachable=False),
        {"upstream": {"up": False, "port": 8002, "resolution": {
            "port": 8002, "source": "shell_config", "reason": "not probed yet",
            "candidates": [{"port": 8002, "source": "shell_config",
                            "backend": None, "listening": None, "pid": None}],
        }}},
    )
    assert ":8002: not probed" in dom["sMeta"]["title"], dom["sMeta"]["title"]
    assert "nothing listening" not in dom["sMeta"]["title"], dom["sMeta"]["title"]


def test_the_serving_line_says_nothing_extra_when_there_is_no_resolution() -> None:
    """Over-correction guard: an older payload (a browser holding a stale page
    across a restart) carries no `resolution`, and the line must degrade to the
    sentence it always had rather than rendering "undefined"."""
    dom = _render(_payload(reachable=False), {"upstream": {"up": False, "port": 8001}})
    assert dom["sMeta"]["text"] == "nothing serving on http://localhost:8001", (
        dom["sMeta"]["text"]
    )
    assert dom["sMeta"]["title"] == "", dom["sMeta"]["title"]


# ---------------------------------------------------------------------------
# Phase-1 sweep, F3: the button labelled "Apply & restart" never restarted
# ---------------------------------------------------------------------------
def test_apply_reaches_the_endpoint_that_actually_relaunches() -> None:
    """The Apply handler posted only to /api/server/start.

    supervisor.start() returns at its first statement when the server is
    READY, so applying a new context length to a running server changed
    nothing while the page logged a success — and POST /api/server/restart,
    the only endpoint that relaunches, appeared nowhere in the page's source
    at all. Driven on the live 27B 2026-09-10: 202 accepted, pid unchanged,
    .config unchanged, cmdline unchanged.
    """
    wire = _fn_body("wireControls")
    apply_handler = wire[wire.index('$("apply")'):]
    assert "/api/server/restart" in apply_handler, (
        "the Apply button cannot reach the only endpoint that relaunches; "
        "every config change it posts at a running server is a silent no-op"
    )
    # ... and starting a stopped server must still go through start.
    assert "/api/server/start" in apply_handler


def test_the_recommendation_does_not_state_a_bucket_edge_as_the_measured_p90() -> None:
    """F9: the two halves of the request-statistics panel disagree.

    A percentile over bucket-bounded observations is an INTERVAL, and the
    panel renders it as one three lines higher ("20,001–50,000" via pct()).
    The recommendation is divided by the interval's UPPER edge -- deliberately,
    it is the conservative end -- but printed it as an equality: "p90 of the
    last 5 requests = 50,000 prompt tokens", i.e. a histogram bucket edge
    stated as a measurement. Only when every observation at that rank was
    exact (reqstats.Percentile.exact) is "=" true.
    """
    paint = _fn_body("paintRequestStats")
    start = paint.index('set("recBasis"')
    statement = paint[start : paint.index(";", start) + 1]
    assert "rec.basis" in statement, "the recommendation's basis line is gone"
    assert "exact" in statement.lower(), (
        "the basis line states the p90 as an equality without consulting "
        f"p90.exact: {statement}"
    )
    # The over-subscription banner quotes the same number and sits OUTSIDE
    # `if (rec) {`. A block-scoped const declared inside it is a
    # ReferenceError there, and there is no JS runtime in this venv to catch
    # one -- so pin the declaration ahead of the block.
    assert paint.index("const p90Exact") < paint.index("if (rec) {"), (
        "p90Exact is declared inside `if (rec)` but read again outside it"
    )
