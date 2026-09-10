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
    "dAgents", "dAgentsS", "dCtx", "dCtxS", "dBadge", "kvPct", "kvTok",
    "mRun", "mWait", "mPre", "hitRate",
    # The request-size window and the parallelism recommendation. Same rule:
    # until their painter runs there is no reading, and a shipped digit here
    # would be a fabricated one.
    "pctP90", "pctP50", "pctP99", "pctMax", "winMeta", "genMeta", "statsProv",
    "recN", "recBasis", "recMath", "recP99", "recCal",
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


def test_serving_line_names_prefill_decode_and_time_to_first_token() -> None:
    line = _serving_line()
    for word in ("prefill", "decode", "time to first token"):
        assert word in line, f"the serving line never says {word!r}: {line}"
    assert "gen`" not in line, (
        "the serving line abbreviated generation throughput to 'gen', which "
        "reads as neither prefill nor decode"
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
}
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
    "agoTxt", "secsTxt", "rateTxt", "windowFigure", "lifeTxt", "segFigure",
    "uptimeTxt", "paintThroughput", "paintServingMeta",
)


def _const_line(name: str) -> str:
    """A top-level `const name = ...;` line, verbatim."""
    i = APP_JS.index(f"const {name} = ")
    return APP_JS[i : APP_JS.index("\n", i)]


def _element_ids() -> list[str]:
    return sorted(set(re.findall(r'id="([A-Za-z0-9_]+)"', INDEX_HTML)))


def _render(payload: dict, state: dict | None = None) -> dict:
    """Paint the throughput strip (and, with `state`, the serving line) from a
    real /api telemetry payload and return every element's rendered text."""
    dukpy = pytest.importorskip(
        "dukpy", reason="pip install -e '.[dev]' brings in the JS engine"
    )
    ids = _element_ids()
    src = [
        _DOM_SHIM,
        _const_line("$"),
        _const_line("fmt"),
        *[_fn_body(n) for n in _RENDER_FNS],
        f"__mk({json.dumps(ids)});",
        f"liveMetrics = {json.dumps(payload)};",
        f"lastState = {json.dumps(state)};",
        "paintThroughput();",
        "if (lastState) paintServingMeta();",
        f"__dump({json.dumps(ids)});",
    ]
    return json.loads(dukpy.evaljs("\n".join(src)))


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
    measures, and an absent figure collapsed to a bare em dash with no unit,
    beside a neighbour that still had a number."""
    dom = _render(
        _payload(prefill_tok_s=1240.0, gen_tok_s=249.0, ttft_s=1.85), _SERVING
    )
    line = dom["sMeta"]["text"]
    for label in ("prefill", "decode", "time to first token"):
        assert label in line, f"{label!r} missing from the serving line: {line}"
        # label first, number after
        after = line.split(label, 1)[1].lstrip()
        assert after and (after[0].isdigit() or after.startswith(("idle", "n/a"))), (
            f"{label!r} is not followed by its figure: {line}"
        )
    assert "—" not in line, f"an em dash is not a reading: {line}"


def test_the_serving_line_says_idle_rather_than_dropping_a_figure() -> None:
    """The state the owner was looking at: decoding, nothing prefilling. Both
    words stay on the line, and the one with no reading says why."""
    dom = _render(
        _payload(gen_tok_s=249.0, prefill_reason=metrics.IDLE,
                 ttft_reason=metrics.IDLE),
        _SERVING,
    )
    line = dom["sMeta"]["text"]
    assert "prefill idle" in line, line
    assert "decode 249.0 tok/s" in line, line


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
    # every segment is one nowrap element, so nothing renders as a lone number
    assert dom["sMeta"]["text"].count("·") >= 4


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
    for body in (_fn_body("windowFigure"), _fn_body("segFigure")):
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


def test_provenance_label_is_rendered_beside_the_percentiles() -> None:
    """The estimate's provenance is not optional decoration.

    The percentiles are bucket-quantised (vLLM publishes a histogram, not
    per-request rows) and cover only requests that finished while Servedeck was
    watching. A p90 shown without that caveat reads as an exact measurement of
    the server's whole traffic, which it is not. This pins BOTH halves: the
    markup has the slot, and the painter fills it from the backend's own
    string rather than from a copy in the page that could drift.
    """
    assert 'id="statsProv"' in INDEX_HTML, "no slot for the provenance label"
    painter = _request_stats_painter()
    assert 'set("statsProv"' in painter, "the provenance slot is never filled"
    assert "sz.provenance" in painter, (
        "the label must come from reqstats.PROVENANCE via the payload, not be "
        "restated in the page where it can drift from the module that computes "
        "the estimate"
    )


def test_provenance_string_actually_reaches_the_page_payload() -> None:
    """The other end of the same contract: the field the painter reads is the
    field the backend emits, carrying the real sentence."""
    from servedeck import reqstats

    payload = _app._sizing_payload()
    assert payload["provenance"] == reqstats.PROVENANCE
    assert "bucket" in payload["provenance"].lower()


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
