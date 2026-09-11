/* Servedeck frontend.
 *
 * Ported from the approved design prototype. Two deliberate differences:
 *  1. The prototype's capacity() is GONE. The browser never computes capacity —
 *     one implementation lives server-side and is used by both the estimate and
 *     the validation, so they cannot drift.
 *  2. No simulated data. Everything arrives from /api/* and /api/events.
 */
const $ = (id) => document.getElementById(id);
const fmt = (n) => (n == null ? "—" : Number(n).toLocaleString("en-US"));

/* Write text into an element by id, if it exists.
 *
 * Every painter needs this and four of them used to define their own identical
 * copy. Hoisted once: the null guard is the whole content, and a per-function
 * copy is one more place for the guard to be forgotten.
 */
function set(id, v) { const e = $(id); if (e) e.textContent = v; }

/* Bytes -> a string that carries its own unit.
 *
 * Binary (1024), because everything else on this page is: df, free,
 * nvidia-smi and every vLLM log line. The unit is returned WITH the number
 * on purpose. The model cards used to render a GiB value next to hardcoded
 * markup reading "GB" -- 125.99 GiB shown as "125.91 GB", a 7.4% error that
 * looks exactly like a rounding slip and is not one. Nothing here may print
 * a size without calling this.
 */
const BYTE_UNITS = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
function bytesTxt(n, digits) {
  if (n == null || !isFinite(n)) return "—";
  const d = digits == null ? 2 : digits;
  let v = Math.abs(Number(n));
  const sign = Number(n) < 0 ? "-" : "";
  let i = 0;
  while (v >= 1024 && i < BYTE_UNITS.length - 1) { v /= 1024; i++; }
  return i === 0 ? `${sign}${Math.round(v)} B` : `${sign}${v.toFixed(d)} ${BYTE_UNITS[i]}`;
}

/* A throughput reading, or an em dash.
 *
 * null/undefined means "not known right now" — an idle server has no tok/s,
 * and a counter that reset means this window spans two processes. Printing 0
 * for either reads as "the engine got slow", which is the opposite of true.
 * Guard on the type: `x || 0` also swallows a genuine 0, and `x ?? 0` still
 * prints a number where there is no measurement.
 */
function rateTxt(v, digits) {
  if (typeof v !== "number" || !isFinite(v)) return "—";
  return (digits ? v.toFixed(digits) : fmt(Math.round(v))) + " tok/s";
}

/* How long ago, in words. Whole seconds/minutes: this is the age of a
 * reading, not a reading. */
function agoTxt(sec) {
  if (typeof sec !== "number" || !isFinite(sec) || sec < 0) return "";
  if (sec < 60) return Math.round(sec) + " s ago";
  const m = Math.round(sec / 60);
  return m < 60 ? m + " min ago" : Math.round(m / 60) + " h ago";
}

function secsTxt(v) {
  if (typeof v !== "number" || !isFinite(v)) return "n/a";
  return v >= 10 ? v.toFixed(0) + " s" : v >= 1 ? v.toFixed(1) + " s"
       : Math.round(v * 1000) + " ms";
}

/* The WINDOW line of one throughput cell: {value, note}.
 *
 * The rule this function exists to enforce: a cell's window line shows the
 * window figure or says why it cannot, and it NEVER falls back to a different
 * quantity. The panel used to substitute a lifetime average here behind a "~",
 * so a cell read "3,013 tok/s" whether that was a rate measured over the last
 * two seconds or an average over three thousand requests. Those differ by
 * 2.4x for decode on this box (measured: window 249.0 tok/s vs lifetime
 * 104.4 tok/s) — same font, same slot, no way to tell.
 *
 * When there is no window reading, the value is "idle" (or "n/a") and the note
 * carries the LAST window reading and its age. That is a fact about this
 * figure, so the cell is never blank and never looks as though its neighbour
 * has taken it over.
 */
function windowFigure(live, last, ageS, state, reason, fmtFn) {
  if (typeof live === "number" && isFinite(live)) {
    return { value: fmtFn(live), na: false, note: "now · last 2 s window" };
  }
  // Switch on the CODE, never on the English. metrics.REASON_CODE is the
  // table; test_ui.py checks the codes named here are all in it.
  const head = (state === "idle") ? "idle" : "n/a";
  let note = reason || "no reading";
  if (typeof last === "number" && isFinite(last)) {
    const ago = agoTxt(ageS);
    note = note + " · last " + fmtFn(last) + (ago ? ", " + ago : "");
  }
  return { value: head, na: true, note: note };
}

/* The LIFETIME line of one throughput cell.
 *
 * Always rendered, in its own line, in its own smaller type, and always
 * naming its denominator — because it is not the same quantity as the line
 * above it. "per second of prefill time" is what makes 3,013 tok/s and
 * 249 tok/s both true at once.
 */
function lifeTxt(v, fmtFn, basis, n) {
  if (typeof v !== "number" || !isFinite(v)) return "lifetime n/a";
  const count = (typeof n === "number" && n > 0) ? " over " + fmt(n) + " requests" : "";
  return "lifetime " + fmtFn(v) + " " + basis + count;
}

/* The throughput strip: prefill, decode and TTFT, always all three, and for
 * each of them a window figure and a lifetime figure, always both.
 *
 * Rendering one number where the other had been is the defect this replaces.
 * Owner: "when one is not displayed, the prefill speed shows, when it is not
 * running, while generate shows when generating". Both causes are gone: no
 * cell can go blank (an idle window prints its last value and an age) and no
 * cell can borrow another cell's quantity.
 */
function paintThroughput() {
  const m = liveMetrics || {};
  const rate0 = (v) => rateTxt(v);
  const rate1 = (v) => rateTxt(v, 1);
  const cells = [
    ["thPrefill",
     windowFigure(m.prefill_tok_s, m.prefill_tok_s_last, m.prefill_last_age_s,
                  m.prefill_state, m.prefill_reason, rate0),
     lifeTxt(m.prefill_tok_s_avg, rate0, "per second of prefill time",
             m.prefill_requests)],
    ["thDecode",
     windowFigure(m.gen_tok_s, m.gen_tok_s_last, m.gen_last_age_s,
                  m.gen_state, m.gen_reason, rate1),
     lifeTxt(m.gen_tok_s_avg, rate1, "per second of decode time")],
    ["thTtft",
     windowFigure(m.ttft_s, m.ttft_s_last, m.ttft_last_age_s,
                  m.ttft_state, m.ttft_reason, secsTxt),
     lifeTxt(m.ttft_s_avg, secsTxt, "mean", m.ttft_requests)],
  ];
  cells.forEach(function (row) {
    const id = row[0], f = row[1], life = row[2];
    const n = $(id), sub = $(id + "S"), lifeEl = $(id + "L");
    if (n) { n.textContent = f.value; n.className = "n mono" + (f.na ? " na" : ""); }
    if (sub) sub.textContent = f.note;
    if (lifeEl) lifeEl.textContent = life;
  });
}

/* Seconds since the SERVING process started -> "3h 12m".
 *
 * null when the listener's start time could not be read. It must never fall
 * back to Servedeck's own uptime: that is a different quantity, and using it
 * made a long-running server look freshly started every time the UI was
 * restarted (procctl.process_uptime_s' docstring; app.py's comment on the
 * uptime_s field says the serving line "must NOT use this").
 */
function uptimeTxt(sec) {
  if (typeof sec !== "number" || !isFinite(sec)) return "—";
  const m = Math.floor(sec / 60);
  return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}m`;
}

/* SPEC.md §4: a measurement taken at a DIFFERENT context length is still a
 * measurement, but not of this configuration — it renders "measured*".
 * "unknown" is its own answer (the safetensors estimator is refused for this
 * architecture, SPEC.md §3 UNKNOWN_CAPACITY) and must not be laundered into
 * "estimated", which carries a specific "~25% optimistic" meaning.
 */
/* Why a model's size may not be space on THIS filesystem.
 *
 * Half the hub cache here is snapshots whose blobs are symlinks onto an
 * exfat stick mounted under /run/media. Those bytes are real -- the model
 * loads from them -- but deleting the model frees none of the free space
 * shown next to it. The card marks the difference instead of silently
 * adding 126 GiB to a figure the operator will compare against df.
 */
function diskTitle(m) {
  const total = bytesTxt(m.disk_bytes);
  if (!(m.disk_bytes > m.disk_local_bytes)) {
    return `${total} on disk (each blob counted once; hardlinks and repeated snapshot symlinks are not double-counted)`;
  }
  return (
    `${total} on disk, of which only ${bytesTxt(m.disk_local_bytes)} is on this filesystem — ` +
    `the rest lives on another mount, so deleting this model frees ${bytesTxt(m.disk_local_bytes)} here, not ${total}`
  );
}

function trustTag(trust) {
  if (trust === "measured") return '<span class="tag meas">measured</span>';
  if (trust === "measured_other_ctx")
    return '<span class="tag meas" title="measured, but at a different context length">measured*</span>';
  if (trust === "unknown")
    return '<span class="tag est" title="this architecture\'s VRAM footprint cannot be predicted from disk size">unknown</span>';
  return '<span class="tag est">estimated</span>';
}

/* The context control.
 *
 * It used to be a fixed ladder of buttons, which is wrong in two directions
 * at once. It offered lengths the KV budget cannot hold — the engine loads
 * weights for several minutes and only then refuses — and it stopped at a
 * hardcoded ceiling, so a model declaring max_position_embeddings 1,048,576
 * had three quarters of its range unreachable from the page.
 *
 * The slider's bounds are two real numbers the backend computes:
 *   ctx_max_model — the checkpoint's own max_position_embeddings
 *   ctx_max_fit   — (KV tokens the budget buys) / (parallel agents)
 * and the smaller of the two is the ceiling. Both move when the utilization
 * slider, the agent count or the selected model moves, so the control is
 * redrawn on every estimate rather than once at load.
 */
const MIN_CTX = 8192;
const CTX_STEP = 4096;
const DEFAULT_MAX_CTX = 262144;   // mirrors app.DEFAULT_MODEL_MAX_CTX

function ctxLabel(c) {
  if (!c) return "—";
  return c >= 1048576 ? +(c / 1048576).toFixed(2) + "M"
       : c >= 1024    ? Math.round(c / 1024) + "k"
       : String(c);
}

/* The highest context the slider can actually rest on.
 *
 * A range input's thumb only stops at min + k*step, so a ceiling that is not on
 * the grid is unreachable and the thumb parks one step below it — while the
 * readout beside the slider shows the unsnapped ceiling. The control then
 * disagrees with its own value, which is the exact defect this page has tests
 * for everywhere else. Snapping down (never up) keeps the thumb, the readout and
 * the value POSTed to /api/server/start identical, and never exceeds the bound
 * the ceiling came from.
 */
function ctxTop(ceiling) {
  const steps = Math.floor((ceiling - MIN_CTX) / CTX_STEP);
  return MIN_CTX + Math.max(0, steps) * CTX_STEP;
}

/* Redraw the context slider against the bounds in an estimate.
 *
 * `d` is the /api/capacity/estimate payload, or null before the first one has
 * come back — in which case the only bound known is the selected model's
 * ceiling, and the fit bound is left blank rather than guessed at.
 */
/* What the agents field's constraint line says, and how loud it is.
 *
 * Painted beside the field rather than only in the live panel because it is a
 * limit ON that field, and a limit nobody can see while typing into the box is
 * not a limit: the field read "16" next to "KV fits 1" with nothing connecting
 * the two. Pure so the test engine can execute the comparison — the contradiction
 * is the whole content, and a grep assertion passes unchanged when `fitN <
 * agents` is flipped to `fitN > agents`.
 */
function agentsFitNote(fitN, want, ctxLen) {
  if (!fitN) {
    return {text: "—", overfit: false,
            title: "the KV budget for this configuration is not known yet"};
  }
  const over = fitN < want;
  return {
    text: "KV fits " + fitN,
    overfit: over,
    title: `at ${fmt(ctxLen)} context the KV budget holds ${fitN} agent`
      + `${fitN === 1 ? "" : "s"} — the field above asks for ${want}`
      + (over ? " and is over budget" : ""),
  };
}

function renderCtx(d) {
  const el = $("ctx");
  if (!el) return;
  const modelMax = Number((d && d.ctx_max_model) || MODELS[sel]?.model_max_ctx) || DEFAULT_MAX_CTX;
  const fit = d ? Number(d.ctx_max_fit) || 0 : 0;
  // The ceiling is whichever real limit binds first. A zero fit means the
  // budget holds nothing at this utilization: keep the model ceiling as the
  // range so the slider still moves, and let the blocker explain why.
  const ceiling = Math.max(MIN_CTX, fit > 0 ? Math.min(modelMax, fit) : modelMax);
  // The ceiling is snapped DOWN onto the step grid, and the reason is a lie the
  // slider otherwise tells: a range input's thumb can only rest on
  // min + k*step, so with min 8192 / step 4096 a max of 19,100 leaves the thumb
  // at 16,384 while the readout beside it says 19,100 -- the control and its own
  // value disagreeing, which is the defect class this page is full of tests for.
  // Snapping keeps thumb, readout and the value sent to the backend identical.
  // It never raises the ceiling, so the KV budget is still respected.
  const top = ctxTop(ceiling);
  el.min = String(MIN_CTX);
  el.max = String(top);
  el.step = String(CTX_STEP);
  if (ctx > top) ctx = top;
  if (ctx < MIN_CTX) ctx = MIN_CTX;
  el.value = String(ctx);

  set("ctxV", fmt(ctx));
  // Name which of the two limits is binding, so a ceiling that moves when the
  // agent count changes is not mistaken for the model's own limit. The range
  // itself is in the slider's title; the tick row that used to spell it out
  // cost a line per field.
  set("ctxBound", fit > 0 && fit < modelMax
    ? "KV fits " + ctxLabel(fit)
    : "model " + ctxLabel(modelMax));
  const bound = $("ctxBound");
  if (bound) bound.title = fit > 0 && fit < modelMax
    ? `the KV budget holds ${fmt(fit)} tokens per agent at this utilization`
    : `the checkpoint's own max_position_embeddings is ${fmt(modelMax)}`;
  const cx = $("ctx");
  if (cx) cx.title = `${ctxLabel(MIN_CTX)} to ${ctxLabel(ceiling)} — ${
    fit > 0 && fit < modelMax ? "limited by the KV budget" : "limited by the model"
  }`;
  // The agents field's own constraint. Painted here rather than only in the
  // live panel because it is a limit ON that field; see agentsFitNote().
  const fitNote = agentsFitNote(d && d.agents_at_ctx, agents, ctx);
  const af = $("agentsFit");
  if (af) {
    af.classList.toggle("overfit", fitNote.overfit);
    af.title = fitNote.title;
  }
  set("agentsFit", fitNote.text);
}
let MODELS = [];
let sel = 0;
let util = 0.95;   // overwritten from the running server via /api/state
let ctx = 262144;
let agents = 1;   // parallel agents the context is being sized for
let lastEstimate = null;
let controlEnabled = false;
let liveFacts = {};      // the RUNNING engine's own numbers, from /api/state
let liveSizing = {};     // request-size window + parallelism recommendation
let lastState = null;    // last /api/state payload, so telemetry can repaint
let userPicked = false;  // once the user picks a model, stop auto-selecting

/* ------------------------------------------------------------- logging -- */
function log(msg, cls) {
  const L = $("log");
  if (!L) return;
  const t = new Date().toLocaleTimeString("en-GB", { hour12: false });
  const d = document.createElement("div");
  const span = document.createElement("span");
  span.className = cls || "";
  span.textContent = msg;
  const ts = document.createElement("span");
  ts.className = "t";
  ts.textContent = t + "  ";
  d.appendChild(ts);
  d.appendChild(span);
  L.appendChild(d);
  while (L.children.length > 300) L.removeChild(L.firstChild);
  L.scrollTop = L.scrollHeight;
}

/* -------------------------------------------------------- model rail ---- */

/* The rail is repainted on every state event — every five seconds — and used to
 * tear down and rebuild all its cards each time. That stole the keyboard focus
 * from whatever card the operator was arrowing through and reset the list's
 * scroll position mid-scroll, so reaching the eighth model meant scrolling to it
 * again every five seconds.
 *
 * So the cards are rebuilt only when the SET changes (a model appears, vanishes,
 * becomes servable, or changes size); selection and the serving tag are updated
 * in place otherwise. The signature is deliberately not just MODELS.length,
 * which would miss a model becoming servable.
 */
let railSig = "";

/* What a rebuild of the rail is keyed on, and what its header says.

   Both are separate from renderModels() so they can be EXECUTED by the test
   engine: the decision "did the set change" is the whole fix, and a canvas-
   free pure function is assertable while a DOM teardown is not.

   The signature is deliberately not MODELS.length, which would miss a model
   becoming servable, nor repo_id alone, which would miss a model changing size
   or gaining its blobs. */
function railSigOf(models) {
  return (models || []).map((m) => `${m.repo_id}|${m.servable ? 1 : 0}|${m.disk_bytes}`).join(";");
}

/* "N models" alone hid the number that actually matters: how many of them this
 * machine can serve at all. Say both. */
function railCountOf(models) {
  const n = (models || []).length;
  const servable = (models || []).filter((m) => m.servable).length;
  return `${servable}/${n} servable`;
}

function renderModels() {
  const L = $("mlist");
  if (!L) return;
  const c = $("mcount");
  if (c) c.textContent = railCountOf(MODELS);
  const sig = railSigOf(MODELS);
  if (sig === railSig) { updateRailSelection(); return; }
  railSig = sig;
  // Which card had focus, and how far the list was scrolled, are restored after
  // the rebuild. Focus is tracked by repo_id rather than index, because a model
  // appearing earlier in the list would otherwise move the focus to a different
  // model under the operator's keyboard.
  const focused = L.querySelector ? L.querySelector(":focus") : null;
  const focusRepo = focused && focused.dataset ? focused.dataset.repo : null;
  const scrollTop = L.scrollTop;
  L.innerHTML = "";
  MODELS.forEach((m, i) => {
    const b = document.createElement("button");
    b.className = "mcard";
    b.type = "button";
    b.setAttribute("aria-pressed", i === sel ? "true" : "false");
    // The rail is a list of choices, not a list of unrelated buttons.
    b.setAttribute("role", "option");
    b.dataset.repo = m.repo_id || "";
    const tags = [];
    tags.push(`<span class="tag">${m.quant || "—"}</span>`);
    if (m.serving) tags.push('<span class="tag live">serving</span>');
    if (!m.servable) tags.push('<span class="tag est">unservable</span>');
    else tags.push(trustTag(m.trust));
    b.innerHTML =
      `<span class="r1"><span class="nm"></span><span class="sz mono" title="${diskTitle(m)}">${bytesTxt(m.disk_bytes)}${m.disk_bytes > m.disk_local_bytes ? '<span class="off">↗</span>' : ""}</span></span>
       <span class="r2">${tags.join("")}</span>
       <span class="note"></span>`;
    b.querySelector(".nm").textContent = m.name;
    b.querySelector(".note").textContent =
      m.unservable_reason || `${m.backend || "?"} · ${fmt(m.model_max_ctx)} ctx`;
    if (!m.servable) {
      // aria-disabled, NOT the disabled attribute. A disabled button is skipped
      // by Tab entirely, so an operator who cannot serve a model also cannot
      // reach the card to read why -- and the reason lives on the card's own
      // note line, which is only readable if the card can be focused. The
      // click handler is simply not attached below, and .mcard[aria-disabled]
      // in the stylesheet carries the dimmed look the attribute used to set.
      b.setAttribute("aria-disabled", "true");
      b.title = m.unservable_reason || "this model cannot be served here";
    } else {
      b.onclick = () => {
        sel = i;
        userPicked = true;
        renderModels();
        renderCtx(null);   // a different model has a different ceiling
        estimate();
      };
    }
    L.appendChild(b);
  });
  L.scrollTop = scrollTop;
  if (focusRepo && L.querySelector) {
    const back = L.querySelector(`[data-repo="${focusRepo}"]`);
    if (back && back.focus) back.focus();
  }
}

/* Selection and the serving tag without a rebuild. Both change on every state
 * event; neither is a reason to throw away the operator's focus. */
function updateRailSelection() {
  const L = $("mlist");
  if (!L || !L.children) return;
  MODELS.forEach((m, i) => {
    const b = L.children[i];
    if (!b) return;
    b.setAttribute("aria-selected", i === sel ? "true" : "false");
    b.setAttribute("aria-pressed", i === sel ? "true" : "false");
  });
}

/* ----------------------------------------------- capacity (server-side) -- */
let estTimer = null;
function estimate() {
  clearTimeout(estTimer);
  estTimer = setTimeout(doEstimate, 120);
}

async function doEstimate() {
  const m = MODELS[sel];
  if (!m) return;
  const uv = $("utilV");
  if (uv) uv.textContent = util.toFixed(2);
  try {
    const r = await fetch("/api/capacity/estimate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo_id: m.repo_id, util, ctx, max_num_seqs: agents }),
    });
    var d = await r.json();
  } catch (e) {
    // Only fetch/parse failures belong here. Rendering used to be inside this
    // try, so a TypeError in the view was reported as a network failure and
    // sent people looking at a backend that was fine.
    showFindings([{ level: "block", title: "Backend unreachable", detail: String(e) }]);
    return;
  }
  if (d.error) {
    showFindings([{ level: "block", title: "Estimate failed", detail: d.error }]);
    return;
  }
  lastEstimate = d;
  try {
    paintEstimate(d);
  } catch (e) {
    showFindings([{ level: "block", title: "Display error", detail: String(e) }]);
    console.error("paintEstimate failed", e, d);
  }
}

/* The preflight strip: what would stop this configuration starting.
 *
 * It used to be four hardcoded spans that were on screen no matter what the
 * machine was doing, including one ("disk 412 GiB") that nothing measured.
 * Every entry here is a finding capacity.compute() actually produced for the
 * configuration on screen. Real free disk lives in the model rail's .dfline,
 * measured by statvfs.
 */
function paintPreflight(d) {
  const el = $("pref");
  if (!el) return;
  el.innerHTML = "";
  const findings = (d && d.findings) || [];
  const add = (cls, mark, text, title) => {
    const s = document.createElement("span");
    s.className = "chk " + cls;
    const m = document.createElement("span");
    m.className = "m";
    m.textContent = mark;
    s.appendChild(m);
    s.appendChild(document.createTextNode(" " + text));
    if (title) s.title = title;
    el.appendChild(s);
  };
  if (!findings.length) {
    add("ok", "\u2713", "no blockers for this configuration");
    return;
  }
  findings.forEach((f) => {
    // A warning is not a pass. This used to colour every non-blocking finding
    // green with a tick, so "KV budget is tight at this context" and "no
    // blockers" were indistinguishable — the strip read all-green on a
    // configuration that was about to preempt.
    const cls = f.level === "block" ? "crit" : f.level === "warn" ? "warn" : "ok";
    const mark = f.level === "block" ? "\u2715" : f.level === "warn" ? "!" : "\u2713";
    add(cls, mark, f.title,
        (f.detail || "") + (f.fix ? "  Fix: " + f.fix : ""));
  });
}

/* The provenance badge's label and explanation, from the estimate's confidence
 * code.
 *
 * The badge is read as prose, so it must not print the enum: the page showed
 * "measured_other_ctx" verbatim, and "unknown" is the wrong word for a capacity
 * figure (nothing is unknown about the model — its footprint is what cannot be
 * predicted). Pure so the mapping is executed by the tests rather than grepped.
 */
function badgeOf(confidence) {
  if (confidence === "measured_other_ctx") {
    return {text: "measured*", meas: true,
            title: "measured on this machine, but at a different context length"};
  }
  if (confidence === "measured") {
    return {text: "measured", meas: true,
            title: "this model has been booted here at this context length"};
  }
  if (confidence === "unknown") {
    return {text: "unpredictable", meas: false,
            title: "this architecture's VRAM footprint cannot be predicted from its files"};
  }
  return {text: confidence || "estimated", meas: false,
          title: "from the per-architecture calculator, not from a boot here"};
}

function paintEstimate(d) {
  // The bounds move with util, agents and model, so redraw them here rather
  // than once at load.
  renderCtx(d);
  const kv = $("dKv");
  if (kv) {
    kv.innerHTML = (typeof d.kv_gib === "number")
      ? `${d.kv_gib.toFixed(1)}<span style="font-size:13px;font-weight:400"> GiB</span>`
      : "—";
  }
  // Measured beats estimated, and the two are never presented alike. When the
  // model on screen IS the one running, at the context it is running at, the
  // engine's own resolved KV size is the answer — read off
  // vllm:cache_config_info, not computed here. Otherwise this is the
  // per-architecture calculator's estimate and says so.
  const servingNow = MODELS[sel] && MODELS[sel].serving;
  const liveCtx = liveFacts.ctx;
  const measured = servingNow && liveFacts.kv_tokens &&
                   (!liveCtx || liveCtx === ctx) ? liveFacts.kv_tokens : null;
  set("dKvTok", measured
    ? fmt(measured) + " tokens · measured"
    : fmt(d.kv_tokens) + " tokens · " + (d.kv_source === "measured" ? "measured"
      : d.kv_source === "measured_other_ctx" ? "measured at another context"
      : d.kv_source === "estimated" ? "estimated" : "unknown"));
  const kvTok = $("dKvTok");
  if (kvTok) {
    const g = d.kv_geometry;
    kvTok.title = measured
      ? "the running engine's own resolved KV size, from /metrics " +
        "(vllm:cache_config_info)"
      : g
        ? `${g.kib_per_token} KiB per token — ${g.family}, from this ` +
          `checkpoint's config.json (allocator correction ${g.allocator_factor}, ` +
          `${g.factor_note})`
        : "";
  }
  // d.agents_at_ctx — how many agents the KV budget holds at this context — is
  // not painted here. It is painted on the agents control itself, where the
  // number is a constraint on a field rather than a fourth readout; see
  // renderCtx()'s agentsFit. Painting it in both places was the defect: the
  // derived cell restated the input above it, and the badge over it claimed the
  // division had been measured.
  const badge = $("dBadge");
  if (badge) {
    const b = badgeOf(d.confidence);
    badge.textContent = b.text;
    badge.title = b.title;
    badge.className = "tag " + (b.meas ? "meas" : "est");
  }

  const bar = d.bar || {};
  const seg = (id, pct, label) => {
    const e = $(id);
    if (!e) return;
    e.style.width = (pct || 0) + "%";
    if (label !== undefined) e.textContent = (pct > 10 ? label : "");
  };
  paintPreflight(d);
  seg("segW", bar.weights_pct, "weights");
  seg("segK", bar.kv_pct, "KV");
  seg("segO", bar.overhead_pct);
  seg("segF", bar.free_pct);
  // `(x ?? 0).toFixed` is truthy even when x is null — (0).toFixed is a
  // function — so the old guard passed and then threw on the real null.
  // weights_gib IS null for models whose weights cannot be estimated
  // (host-offload architectures), which is a normal state, not an error.
  // Each figure carries its own unit rather than three numbers sharing a
  // trailing one. Same rule as bytesTxt above, and for the same reason: a
  // unit that lives in the template instead of with the value is a unit that
  // can drift away from it. These are VRAM gibibytes from capacity.compute(),
  // already in GiB — NOT bytes, so bytesTxt is the wrong formatter here.
  const gib = (v) => (typeof v === "number" ? `${v.toFixed(1)} GiB` : "—");
  set("vramTxt",
    `weights ${gib(d.weights_gib)} · KV ${gib(d.kv_gib)} · budget ${gib(d.budget_gib)}`);

  showFindings(d.findings || []);
}

function showFindings(findings) {
  const crit = $("alertC"), warn = $("alertW");
  if (!crit || !warn) return;
  crit.classList.remove("on");
  warn.classList.remove("on");
  const block = findings.find((f) => f.level === "block");
  const wf = findings.find((f) => f.level === "warn");
  if (block) {
    crit.classList.add("on");
    $("alertCT").innerHTML = `<b>${block.title}</b> ${block.detail}`;
  } else if (wf) {
    warn.classList.add("on");
    $("alertWT").innerHTML = `<b>${wf.title}</b> ${wf.detail}`;
  }
}

/* --------------------------------------------------- live utilization --- */
let hist = [];
let liveMetrics = {};

/* Size a canvas's bitmap to its CSS box x devicePixelRatio and return the
 * drawing size in CSS pixels.
 *
 * Both canvases shipped a hardcoded width/height attribute (420x42, 440x96)
 * while CSS stretched them to the column's width, so every figure was
 * stretched by an unknown factor and text came out blurry — on a 2x display,
 * permanently soft. Painting in device pixels and scaling the context back
 * means the painters below keep working in CSS-pixel coordinates and come out
 * sharp at any DPI.
 *
 * width/height are only assigned when they actually change: writing to
 * canvas.width clears the bitmap, which would blank the plot on every repaint.
 */
function fitCanvas(c) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round((c.clientWidth || c.width) * dpr));
  const h = Math.max(1, Math.round((c.clientHeight || c.height) * dpr));
  if (c.width !== w) c.width = w;
  if (c.height !== h) c.height = h;
  const x = c.getContext("2d");
  x.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { W: w / dpr, H: h / dpr };
}

/* The palette a canvas painter needs, read from the stylesheet.
 *
 * The colours are CSS custom properties so the plots follow the theme — a
 * hardcoded hex would be invisible in one of the two modes. Each name carries
 * its own fallback because getComputedStyle returns "" for a property the
 * stylesheet does not define, and an empty strokeStyle is silently ignored by
 * the canvas rather than reported.
 */
function palette(names) {
  const cs = getComputedStyle(document.documentElement);
  const out = {};
  names.forEach(([k, fallback]) => {
    out[k] = (cs.getPropertyValue("--" + k) || "").trim() || fallback;
  });
  return out;
}

function spark() {
  const c = $("spark");
  if (!c || !c.getContext) return;
  const x = c.getContext("2d");
  const size = fitCanvas(c);
  const col = palette([["accent", "#0E7C7B"], ["line", "#ccc"]]);
  const acc = col.accent, ln = col.line;
  const w = size.W, h = size.H;
  x.clearRect(0, 0, w, h);
  x.strokeStyle = ln; x.lineWidth = 1;
  [0.25, 0.5, 0.75].forEach((g) => { x.beginPath(); x.moveTo(0, h * g); x.lineTo(w, h * g); x.stroke(); });
  if (hist.length < 2) return;
  const step = w / (Math.max(hist.length, 40) - 1);
  x.beginPath(); x.moveTo(0, h - hist[0] * h);
  hist.forEach((v, i) => x.lineTo(i * step, h - v * h));
  x.strokeStyle = acc; x.lineWidth = 1.75; x.stroke();
  x.lineTo((hist.length - 1) * step, h); x.lineTo(0, h); x.closePath();
  x.globalAlpha = 0.13; x.fillStyle = acc; x.fill(); x.globalAlpha = 1;
}

/* The request-size histogram: the last 100 finished requests, drawn in vLLM's
 * own buckets, with the percentiles marked where they fall.
 *
 * WHY A PICTURE INSTEAD OF FOUR NUMBERS
 * The window's observations are bucket-bounded, so every percentile it can
 * report is an interval — "45,314–49,216". Printed as text, four of those in a
 * row, they read as noise and the one number that is a decision input (the p90,
 * which the agent count is divided by) has to be labelled to death to be found.
 * Drawn, the same data answers the two questions that actually get asked:
 *   - WHERE is the mass?  The bars, on a linear token axis.
 *   - HOW COARSE is it?   A bar's WIDTH is the range it covers. A 30,000-token
 *     bar with nothing beside it says "nothing here is known finer than 30k"
 *     without a sentence saying so. That is the honesty the provenance
 *     paragraph used to carry in prose, and it is the reason the paragraph is
 *     gone rather than merely shortened.
 *
 * The x axis is LINEAR, not log. A log axis would draw every bucket a similar
 * width and quietly destroy the one thing the picture is for.
 *
 * A percentile is an interval, so it is drawn as one: a band from lo to hi,
 * with the marker line at hi — the bound the recommendation is sized on, which
 * is the conservative end. An exact percentile draws as the line alone.
 */
/* Which bars to draw, and which requests to draw behind them.

   Split out of reqHist() so the SELECTION is testable in the JS engine the
   rendering tests already use: a canvas painter cannot be asserted on in
   duktape (no getContext), but this decision can, and it is the decision the
   owner's "this can have more granularity" is about. The rule it encodes:

     exact observations exist  -> draw the fine bins; the interval-only
                                  requests go behind them as an underlay.
     nothing is exact          -> fall back to the engine's own buckets.

   The two are never merged into one bar: a request known only to
   (20000, 50000] placed in a single 5k bin would be drawn at a position no
   measurement supports.
*/
function histData(w) {
  const fine = (w && w.fine) || {};
  const bins = fine.bins || [];
  const intervals = fine.intervals || [];
  const buckets = (w && w.buckets) || [];
  return bins.length
    ? { bars: bins, underlay: intervals }
    : { bars: buckets, underlay: [] };
}

// Which label row each percentile marker's caption goes on, so two close
// percentiles stack instead of overprinting. A label may only sit where the
// previous caption on that row has cleared it, so take the first row whose
// last label is at least `gap` px to the left; if every row is still crowded,
// double up on the last one — overlapping beats running off the canvas.
function markRows(xs, gap, rows) {
  const last = [];
  return xs.map((cx) => {
    let r = rows - 1;
    for (let i = 0; i < rows; i++) {
      if (last[i] === undefined || cx - last[i] >= gap) { r = i; break; }
    }
    last[r] = cx;
    return r;
  });
}

function reqHist(w) {
  const c = $("reqHist");
  if (!c || !c.getContext) return;
  const x = c.getContext("2d");
  const col = palette([["accent", "#0E7C7B"], ["line", "#ccc"], ["muted", "#888"]]);
  const acc = col.accent, line = col.line, muted = col.muted;
  const size = fitCanvas(c);
  const W = size.W, H = size.H;
  const padT = 20, padB = 15;
  const plotH = H - padT - padB;
  x.clearRect(0, 0, W, H);

  // The fine histogram when the window has exact observations to fill it, and
  // the engine's own coarse buckets otherwise. Most observations here are exact
  // (a poll that catches one finished request gets its token count exactly), so
  // the fine bins carry real precision the three fat engine bars would hide;
  // the requests that are only known to a bucket interval are drawn behind them
  // as a translucent "somewhere in this range" band, never merged into a bar
  // that would place them more precisely than they were measured.
  const sel = histData(w);
  const bins = sel.bars;
  const intervals = sel.underlay;

  // The axis is drawn whatever the data is, so an empty window still shows
  // where the readings would land rather than a blank box.
  x.strokeStyle = line; x.lineWidth = 1;
  x.beginPath(); x.moveTo(0, H - padB + 0.5); x.lineTo(W, H - padB + 0.5); x.stroke();
  if (!bins.length) return;

  // Right edge of the axis: the widest thing the window actually knows about,
  // so no bar or marker is clipped and no empty space is invented.
  const hiOf = (v) => (v == null ? 0 : v);
  let xmax = 0;
  const grow = (b) => { xmax = Math.max(xmax, hiOf(b.hi) || b.lo); };
  bins.forEach(grow);
  intervals.forEach(grow);
  [w.p50, w.p90, w.p99, w.max].forEach((p) => {
    if (p) xmax = Math.max(xmax, hiOf(p.hi) || hiOf(p.lo));
  });
  if (!(xmax > 0)) return;
  const px = (v) => Math.max(0, Math.min(W, (v / xmax) * W));

  // Tick labels, at a round step that yields four to six of them.
  const STEPS = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000];
  let step = STEPS[STEPS.length - 1];
  for (let i = 0; i < STEPS.length; i++) {
    if (xmax / STEPS[i] <= 6) { step = STEPS[i]; break; }
  }
  x.font = '9px "IBM Plex Mono",monospace';
  x.fillStyle = muted; x.textBaseline = "top";
  for (let v = step; v <= xmax; v += step) {
    const tx = px(v);
    const lab = v >= 1000 ? Math.round(v / 1000) + "k" : String(v);
    x.fillText(lab, Math.min(tx + 2, W - x.measureText(lab).width - 1), H - padB + 3);
  }

  // One shared height scale across the exact bars and the interval underlay,
  // so a tall underlay band and a tall exact bar mean the same count.
  const maxn = bins.concat(intervals).reduce((m, b) => Math.max(m, b.n), 0);
  const barH = (n) => (maxn ? Math.max(1, (n / maxn) * plotH) : 0);

  // The count scale. Without it a bar is only relatively tall — you can see
  // one bin is twice another but not whether that "another" is 2 requests or
  // 200. One faint gridline at the peak, labelled with the count it is, is
  // enough to read every bar against; drawing a full y axis would crowd a
  // 96px-tall plot for no extra information.
  if (maxn > 0) {
    const gy = H - padB - plotH;
    x.strokeStyle = line; x.lineWidth = 1; x.setLineDash([2, 3]);
    x.beginPath(); x.moveTo(0, gy + 0.5); x.lineTo(W, gy + 0.5); x.stroke();
    x.setLineDash([]);
    x.fillStyle = muted; x.textBaseline = "bottom";
    x.fillText(String(maxn), 1, gy - 1);
    x.textBaseline = "alphabetic";
  }

  // One bar-drawing routine, called twice. The interval underlay and the exact
  // bars are the same rectangle computation at two opacities; they were two
  // copy-pasted loops, which is two places for the "+Inf bucket runs to the
  // axis end" rule to drift apart. The underlay is drawn first so the exact
  // bars sit in front of it.
  const drawBars = (list, alpha) => {
    x.globalAlpha = alpha; x.fillStyle = acc;
    list.forEach((b) => {
      const x0 = px(b.lo);
      // An unbounded +Inf bucket has no right edge; it runs to the axis end.
      const x1 = b.hi == null ? W : px(b.hi);
      const bh = barH(b.n);
      x.fillRect(x0, H - padB - bh, Math.max(1, x1 - x0 - 1), bh);
    });
    x.globalAlpha = 1;
  };
  drawBars(intervals, 0.18);
  // The bars themselves: the fine bins when the window has exact
  // observations, the engine's own buckets when it has none.
  drawBars(bins, 0.5);

  // The percentile markers. p90 is the accent because it is the input to the
  // recommendation; the others are context and are drawn quieter. Labels
  // stagger into two rows so two close percentiles cannot overprint.
  const marks = [
    { k: "p50", p: w.p50, c: muted },
    { k: "p90", p: w.p90, c: acc },
    { k: "p99", p: w.p99, c: muted },
  ].filter((m) => m.p);
  const markX = marks.map((m) => px(hiOf(m.p.hi) || hiOf(m.p.lo)));
  const rows = markRows(markX, 26, 2);
  // The captions live in the 20px band above the plot, two rows of 10px. The
  // baseline has to be stated: the count gridline above leaves it "alphabetic",
  // which anchors glyphs ABOVE the y coordinate and pushes a row-0 caption
  // half off the top of the canvas — the labels were drawn and invisible.
  x.textBaseline = "top";
  marks.forEach((m, i) => {
    const hi = hiOf(m.p.hi) || hiOf(m.p.lo);
    const cx = markX[i];
    if (!m.p.exact && m.p.lo != null && hiOf(m.p.lo) < hi) {
      x.globalAlpha = 0.16; x.fillStyle = m.c;
      x.fillRect(px(m.p.lo), padT, Math.max(1, cx - px(m.p.lo)), plotH);
      x.globalAlpha = 1;
    }
    x.setLineDash([3, 2]); x.strokeStyle = m.c; x.lineWidth = 1;
    x.beginPath(); x.moveTo(cx + 0.5, padT); x.lineTo(cx + 0.5, H - padB); x.stroke();
    x.setLineDash([]);
    x.fillStyle = m.c;
    const lab = m.k;
    x.fillText(lab, Math.min(cx + 3, W - x.measureText(lab).width - 1), 1 + rows[i] * 10);
  });
  x.textBaseline = "alphabetic";
}

function paintTelemetry(t) {
  liveMetrics = t.vllm || {};
  if (t.sizing) liveSizing = t.sizing;
  const g = t.gpu || {};

  // The card's identity and size come from nvidia-smi, not from the markup.
  // index.html used to name one specific GPU and one specific total, which is
  // a claim about the machine that the page has no way to know is still true.
  if (g.used_mib != null) set("gpuUsed", fmt(g.used_mib));
  if (g.total_mib != null) set("gpuTotal", fmt(g.total_mib));
  if (g.name) set("gpuName", g.name);

  const reachable = liveMetrics.reachable;
  const kvp = reachable ? liveMetrics.kv_usage_perc : 0;
  hist.push(kvp || 0);
  if (hist.length > 40) hist.shift();

  set("kvPct", reachable ? Math.round(kvp * 100) + "%" : "—");
  // Prefix cache hit rate is the "is caching working?" number. It is NOT
  // kv_cache_usage_perc, which is in-flight occupancy and drops to 0 when idle.
  const hr = liveMetrics.prefix_hit_rate;
  const hrEl = $("hitRate");
  if (hrEl) hrEl.textContent = (reachable && hr != null) ? (hr * 100).toFixed(1) + "%" : "—";
  // Must be a fraction of what the RUNNING engine allocated. Using the
  // selected model's estimate here showed 1.6M tokens for a server that has
  // 272k - the selected model was not the one running.
  const liveTotal = liveFacts.kv_tokens;
  set("kvTok", (reachable && liveTotal)
    ? fmt(Math.round((kvp || 0) * liveTotal)) + " / " + fmt(liveTotal) + " tokens"
    : "—");
  set("mRun", reachable ? liveMetrics.running : "—");
  // The header chip said "— agents holding" forever. The engine knows how
  // many requests are in flight; say that, and only show the chip when there
  // are any.
  const q = document.querySelector(".queued");
  const running = reachable ? (liveMetrics.running || 0) : 0;
  set("qN", running);
  if (q) q.className = "queued mono" + (running > 0 ? " on" : "");
  set("mWait", reachable ? liveMetrics.waiting : "—");
  set("mPre", reachable ? liveMetrics.preemptions : "—");

  const w = $("mWait");
  if (w) w.className = "n mono" + (reachable && liveMetrics.waiting > 0 ? " hot" : "");
  const p = $("mPre");
  if (p) p.className = "n mono" + (reachable && liveMetrics.preemptions > 8 ? " bad" : "");

  paintRequestStats();
  paintThroughput();
  paintServingMeta();
  spark();
}

/* Request-size distribution and the parallelism recommendation.
 *
 * Every number here is computed in Python (app._sizing_payload ->
 * parallelism.recommend) and arrives whole. The formula deliberately does NOT
 * live here: it has to reproduce a measured concurrency table, and a copy of
 * it in the page would be a second, untested implementation. This function
 * only formats.
 *
 * The distribution is DRAWN rather than listed. This column used to print
 * p50/p90/p99/max as four interval strings, a sample-count line, a
 * generated-token line and a provenance paragraph — six lines of text whose
 * combined content is "the data is coarse, here is roughly where the tail
 * sits". reqHist() says the same thing in one picture, and says the
 * coarseness better, because a bar's width IS the range of sizes it covers.
 */
/* The window's own honesty line: how many requests were observed out of how
 * many it wants, over how long a span, and how many are exact rather than
 * bucket-bounded. "n of 100" is said explicitly whenever the window is partial
 * -- never padded.
 *
 * The span is a DURATION and gets the duration formatter. A raw "1182s" made
 * the reader do arithmetic to learn the window is 20 minutes of traffic, which
 * is the number that decides whether the picture is current at all.
 *
 * Pure so the test engine can execute it: the whole content of this line is
 * its formatting, and a grep for `durTxt` passes even when the arguments are
 * swapped or the exact-count condition is inverted.
 */
function winSpan(w) {
  if (!w || !w.n) return `0/${(w && w.capacity) || 100} requests observed`;
  const age = w.age_s == null ? "" : ` · ${durTxt(w.age_s)}`;
  const ex = w.exact_n < w.n ? ` · ${w.exact_n} exact` : "";
  return `${w.n}/${w.capacity} requests${age}${ex}`;
}

function paintRequestStats() {
  const sz = liveSizing || {};
  const w = sz.window || {};
  const rec = sz.recommended;

  // A percentile over bucket-bounded observations IS an interval. Render it as
  // one -- "20,001–50,000", not a midpoint nothing measured. Only when every
  // observation at that rank was exact does it collapse to a single number.
  const pct = (p) => {
    if (!p) return "—";
    if (p.hi == null) return "> " + fmt(p.lo);
    if (p.exact || p.lo === p.hi) return fmt(p.hi);
    return fmt(p.lo) + "–" + fmt(p.hi);
  };

  // The one figure that is a decision input rather than a description: the
  // p90 is what the recommendation below is divided by. p50/p99/max are on
  // the plot instead of here.
  set("pctP90", pct(w.p90));

  // Is that p90 a measurement or a bucket edge? A percentile over
  // bucket-bounded observations is an interval, rendered as one by pct()
  // just above; the recommendation divides by its UPPER edge (the
  // conservative end), and both places that quoted that number stated it as
  // an equality -- a histogram bucket edge printed as a measurement, on the
  // same panel that had just said 20,001-50,000. Declared out here, not
  // inside `if (rec)`, because the over-subscription banner below reads it
  // too and a const is block-scoped.
  const p90Exact = !!(w.p90 && (w.p90.exact || w.p90.lo === w.p90.hi));
  // The p99 line has the same problem. at_p99 is sized on the UPPER edge of
  // the p99 interval, so it is also a bucket edge unless every observation at
  // that rank was exact.
  const p99Exact = !!(w.p99 && (w.p99.exact || w.p99.lo === w.p99.hi));

  // The window's own honesty line, as short as it can be; see winSpan().
  set("winMeta", winSpan(w));
  reqHist(w);

  if (rec) {
    set("recN", rec.n);
    const ur = $("useRec");
    if (ur) {
      ur.disabled = false;
      ur.title = `write ${rec.n} into the Parallel agents field`;
    }
    set("recBasis",
      `${rec.basis} ${p90Exact ? "=" : "\u2264"} ${fmt(rec.prompt_tokens)} prompt tokens`);
    // The arithmetic, in full. pool / cost gives the raw fit; the headroom
    // factor keeps the last admitted sequence off the preemption edge; the
    // clamp is max_num_seqs, which the scheduler enforces whatever the KV says.
    const clamp = rec.clamped
      ? ` → clamped to --max-num-seqs ${rec.max_num_seqs}`
      : (rec.max_num_seqs ? ` (--max-num-seqs ${rec.max_num_seqs})` : "");
    set("recMath",
      `${fmt(rec.pool_tokens)} KV ÷ ${fmt(rec.cost_tokens)} per request `
      + `= ${rec.fit_n} × ${rec.headroom} headroom = ${rec.n_before_clamp}${clamp}`);
    const alt = sz.at_p99;
    const where = rec.extrapolated
      ? " · cost extrapolated beyond the measured range"
      : (rec.segment ? ` · cost interpolated between ${fmt(rec.segment[0])} and ${fmt(rec.segment[1])} tok` : "");
    set("recP99", (alt
      ? `at p99 (${p99Exact ? "" : "\u2264 "}${fmt(alt.prompt_tokens)} tok) it would be ${alt.n}`
      : "") + where);
  } else {
    set("recN", "—");
    set("recBasis", sz.reason || "—");
    set("recMath", "");
    set("recP99", "");
    const ur = $("useRec");
    if (ur) { ur.disabled = true; ur.title = sz.reason || "nothing recommended yet"; }
  }
  set("recCal", sz.calibration_note || "—");

  // The warning. Live concurrency above the recommendation is the condition;
  // num_preemptions_total is the CONFIRMATION, because preemption is what
  // over-subscription actually does -- vLLM evicts a sequence's KV and
  // recomputes it, so work already paid for is thrown away.
  const os = $("oversub");
  if (!os) return;
  const pre = liveMetrics.preemptions;
  if (rec && sz.over_subscribed) {
    os.className = "oversub on";
    os.innerHTML = `<b>Over-subscribed: ${sz.running} requests in flight, `
      + `${rec.n} recommended</b> at a p90 of ${p90Exact ? "" : "at most "}`
      + `${fmt(rec.prompt_tokens)} prompt tokens. `
      + (pre ? `${pre} preemptions since this server started — that is vLLM `
             + `evicting and recomputing KV, i.e. work already paid for being thrown away.`
             : `No preemptions yet; <span class="mono">vllm:num_preemptions_total</span> `
             + `climbing is what confirms it.`);
  } else if (liveMetrics.waiting_capacity > 0) {
    os.className = "oversub on";
    os.innerHTML = `<b>KV-bound right now.</b> ${liveMetrics.waiting_capacity} request(s) waiting on capacity. Reduce agents, lower per-agent context, or raise utilization.`;
  } else if (pre > 8) {
    os.className = "oversub on";
    os.innerHTML = `<b>${pre} preemptions since restart.</b> vLLM is evicting and recomputing KV — real work is being thrown away. This is the empirical signal that the agent count is too high.`;
  } else {
    os.className = "oversub";
  }
}

/* ------------------------------------------------------------- state ---- */

/* The supervisor's transition, in the operator's words, or "" when it is not
 * mid-transition.
 *
 * Pure, and the single owner of the state list, for two reasons. (1) The list
 * used to be written twice -- once to disable the buttons, once to paint a
 * note -- so the two could disagree about what counts as busy. (2) The note
 * used to be painted by paintState() straight into #sMeta, which
 * paintTelemetry() repaints every two seconds while state arrives every five:
 * "starting…" was on screen for one frame in three and the rest of the time
 * the line claimed the server was already serving. Whatever owns a slot must
 * be the only thing that writes it, so the phase is rendered inside
 * paintServingMeta() and this is the decision it renders from.
 */
const BUSY_PHASES = {STARTING: "starting", PREFLIGHT: "preflight", STOPPING: "stopping", DRAINING: "draining"};

function busyPhase(sv) {
  return BUSY_PHASES[(sv || {}).actual_state] || "";
}

/* Why the page is watching THIS port, in one clause.
 *
 * When the dashboard could not find a server it said "nothing serving on
 * http://localhost:8002" and stopped there — no hint that :8002 came from a
 * shell config naming a backend that had been dead for a week, and none that
 * a healthy server was answering on :8001. The backend now resolves the port
 * and explains itself (updetect.py); the page is where that explanation has
 * to land, because the page is the only thing the operator reads.
 */
function resolutionNote(res) {
  const reason = (res || {}).reason;
  return reason ? " \u2014 " + reason : "";
}

/* Every port the resolver looked at and what it found there, for the tooltip.
 *
 * `listening` is tri-state: false is "probed, nothing there" and null is "not
 * probed at all". Rendering the second as the first would put a claim on the
 * page that nothing ever checked.
 */
function resolutionDetail(res) {
  const cand = ((res || {}).candidates) || [];
  if (!cand.length) return "";
  return "ports checked \u2014 " + cand.map(function (c) {
    const where = c.backend ? `:${c.port} (${c.backend})` : `:${c.port}`;
    const state = c.listening === true
      ? `pid ${c.pid} listening`
      : c.listening === false ? "nothing listening" : "not probed";
    return `${where}: ${state} [${c.source}]`;
  }).join("; ");
}

/* The serving line.
 *
 * Split out of paintState() because its two throughput figures come from the
 * TELEMETRY event (every 2 s) while the rest of it comes from state (every
 * 5 s). Repainting it only on state meant the prefill and generation numbers
 * were up to 5 s old and skipped every other reading.
 */
function paintServingMeta() {
  const meta = $("sMeta");
  if (!meta || !lastState) return;
  const s = lastState, up = s.upstream || {};
  // A transition is its own answer, and it is painted HERE rather than by
  // paintState(): telemetry repaints this element every 2 s, so a note written
  // by the 5 s state painter was overwritten before it could be read.
  const phase = busyPhase(s.supervisor);
  if (phase) {
    meta.textContent = `${phase}…`;
    meta.title = `the supervisor is in ${s.supervisor.actual_state}; ` +
      "the figures below are the last ones the previous server produced";
    return;
  }
  // The SERVING line reports what the server ACTUALLY serves, never the
  // planner's inputs. ctx sources, most authoritative first: the running
  // process's own --max-model-len; then desired config, which is what
  // Servedeck wants and can differ for a server it merely adopted; then, only
  // if both are missing, the slider — local UI state that describes nothing
  // that is running, and which showed a stale figure for a server started at
  // a different length.
  const liveCtx = up.max_model_len || (s.supervisor || {}).max_model_len || ctx;
  // The whole URL, not `:8002`. This is the string a user pastes into a
  // client, and a bare port is not pasteable — they had to reconstruct the
  // scheme and host by hand every time.
  const base = up.port ? `http://localhost:${up.port}` : "—";
  // One segment per figure, each an element of its own, each LABEL FIRST.
  //
  // Two defects fixed here. (1) The whole line was one text node, so it
  // wrapped wherever it happened to fit — including between a number and its
  // unit, and between a number and the word saying which number it was. Each
  // segment is now white-space:nowrap and the line breaks only between them.
  // (2) The label trailed the number ("249.0 tok/s gen"), so you read the
  // figure before learning what it measured, and when a figure was missing
  // the line read "— gen": an em dash with no unit and no explanation, while
  // the OTHER figure still showed a number. That is what made it look as
  // though prefill and generate were swapping places.
  //
  // Identity only. The three throughput figures used to be here too, as bare
  // numbers with no reason for an absent one and no age for a stale one, while
  // the strip below carried the same three done properly. Two renderings of one
  // figure is two chances for them to disagree, so the strip is their only home
  // and this line keeps the three facts only it carries.
  const segs = [
    ["", base],
    ["context", fmt(liveCtx) + " tokens"],
    ["up", uptimeTxt(s.server_uptime_s)],
  ];
  meta.textContent = "";
  if (!up.up) {
    // The reason, not just the blank. An operator who cannot see WHICH port
    // was chosen and why cannot tell a dead box from a misaimed dashboard.
    meta.textContent = `nothing serving on ${base}${resolutionNote(up.resolution)}`;
  } else {
    segs.forEach((seg, i) => {
      if (i > 0) {
        const sep = document.createElement("span");
        sep.className = "sep";
        sep.textContent = "·";
        meta.appendChild(sep);
      }
      // createElement + textContent, not innerHTML: a served model name is
      // upstream data and must never be parsed as markup.
      const el = document.createElement("span");
      el.className = "seg";
      if (seg[0]) {
        const b = document.createElement("b");
        b.textContent = seg[0] + " ";
        el.appendChild(b);
      }
      el.appendChild(document.createTextNode(seg[1]));
      meta.appendChild(el);
    });
  }
  meta.title = up.up
    ? "the address a client points at, the context the running engine accepts, " +
      "and how long it has been up. Throughput is NOT here: the strip below " +
      "carries prefill, decode and time-to-first-token with their reasons and " +
      "lifetime companions."
    : resolutionDetail(up.resolution);
}

/* A duration for the boot bar: "48 s", "3m 12s", "1h 04m".
 *
 * uptimeTxt() is the serving line's formatter and floors to whole minutes,
 * which is right for "up 3h 12m" and useless for a boot that is 48 seconds in
 * and moving every tick. The boot bar needs the seconds to visibly advance or
 * it reads as frozen.
 */
function durTxt(sec) {
  if (typeof sec !== "number" || !isFinite(sec) || sec < 0) return "—";
  const s = Math.round(sec);
  if (s < 60) return s + " s";
  const p2 = (n) => (n < 10 ? "0" + n : String(n));
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${p2(s % 60)}s`;
  return `${Math.floor(m / 60)}h ${p2(m % 60)}m`;
}

/* Is a boot in progress? Only then is the bar drawn: once the server is READY
 * the phase history is a post-mortem, not progress, and a dead widget above the
 * controls every time the page opens is worse than no widget.
 *
 * The supervisor state gates it too. A run that has ended still carries its
 * last phase: "ready" after a Stop, or the phase a failed boot died in. Gating
 * on phase alone painted those as boots in progress, with the elapsed clock
 * counting beside "Not reachable" ("elapsed 4m 38s", 2026-09-11). The server
 * applies the same gate (app.py _boot_in_progress). */
const BOOTING_STATES = ["PREFLIGHT", "STARTING"];
function bootActive(b) {
  return !!(b && b.phase && !b.reached_ready && BOOTING_STATES.indexOf(b.actual_state) >= 0);
}

/* The ETA line, with its provenance. history.py decides whether the figure is a
 * median of this model's measured boots or SPEC.md's calibration fallback, and
 * the note says which — an ETA that is a guess must never read like a
 * measurement, and being past the estimate must be said rather than hidden. */
function bootNote(b) {
  if (typeof b.eta_s !== "number") return b.eta_note || "no boot estimate for this model yet";
  const past = typeof b.elapsed_s === "number" && b.elapsed_s > b.eta_s;
  const bits = [past
    ? `past the ${durTxt(b.eta_s)} estimate already`
    : `typically ${durTxt(b.eta_s)} (${b.cold ? "cold" : "warm"} boot)`];
  if (typeof b.eta_p90_s === "number") bits.push(`p90 ${durTxt(b.eta_p90_s)}`);
  bits.push(b.eta_source === "history" ? "from this model's measured boots"
                                       : "from calibration figures");
  return bits.join(" · ");
}

/* The boot-progress bar.
 *
 * The markup for this shipped with the prototype and has never had a painter:
 * a multi-minute boot sat on "Elapsed —" with an empty track while the
 * supervisor was publishing the phase, the per-phase times and a real ETA the
 * whole time. This paints all of it from s.boot (app._boot_payload).
 *
 * The bar is drawn ONLY while a boot is in progress. Once the server is READY
 * the phase history is a post-mortem, not a progress bar, and leaving it up
 * would put a dead widget above the controls every time the page is opened.
 */
const PHASE_LABELS = {
  init: "init",
  loading_weights: "weights",
  compiling: "compile",
  kv_cache: "KV cache",
  cuda_graphs: "graphs",
  http_start: "HTTP",
  ready: "ready",
};

function paintBoot(b) {
  const wrap = $("phases");
  if (!wrap) return;
  const order = b.phases || [];
  const booting = bootActive(b);
  wrap.className = "phases" + (booting ? " on" : "");
  if (!booting) return;

  const el = $("elapsed");
  if (el) el.textContent = durTxt(b.elapsed_s);

  const idx = order.indexOf(b.phase);
  const times = b.phase_times || {};
  const track = $("ptrack"), labels = $("plabels");
  if (track && labels) {
    if (track.childNodes.length !== order.length) {
      track.textContent = "";
      labels.textContent = "";
      order.forEach((p) => {
        const step = document.createElement("div");
        step.className = "pstep";
        const fill = document.createElement("div");
        fill.className = "fill";
        step.appendChild(fill);
        track.appendChild(step);
        const lab = document.createElement("span");
        lab.textContent = PHASE_LABELS[p] || p;
        labels.appendChild(lab);
      });
    }
    // A phase that has been seen is done; the current one pulses; unseen ones
    // stay empty. The per-phase times are what prove a phase actually ran
    // rather than the index merely being ahead of it.
    order.forEach((p, i) => {
      const done = times[p] != null && i < idx;
      const act = i === idx;
      track.children[i].className = "pstep" + (done ? " done" : act ? " act" : "");
      labels.children[i].className = done ? "done" : act ? "act" : "";
    });
  }

  const note = $("pnote");
  if (note) {
    note.textContent = bootNote(b);
    note.style.display = "block";
  }
}

/* The unsaved-changes marker.
 *
 * It shipped as markup with a .on class in CSS and nothing ever added the
 * class, so "unsaved changes" was permanently invisible. It means: the controls
 * on screen differ from what the running server was started with.
 */
/* Which controls differ from the running server, as human-readable bits.
 *
 * Pure so the test engine can execute it: the marker's whole value is the
 * comparison, and "has anything been clicked" is the wrong question — moving a
 * slider back to the running value must clear it, and a server adopted at util
 * 0.90 must mark the page's default 0.95 as changed with no click at all. */
function dirtyBits(facts, sizing, want) {
  const bits = [];
  if (facts.util_effective && Math.abs(facts.util_effective - want.util) > 0.002) {
    bits.push(`util ${want.util.toFixed(2)}`);
  }
  if (facts.ctx && facts.ctx !== want.ctx) bits.push(`${fmt(want.ctx)} ctx`);
  if (sizing.max_num_seqs && sizing.max_num_seqs !== want.agents) {
    bits.push(`${want.agents} agent${want.agents === 1 ? "" : "s"}`);
  }
  return bits;
}

function paintDirty(upIsUp) {
  const d = $("dirty");
  if (!d) return;
  if (!upIsUp) { d.classList.remove("on"); d.textContent = "unsaved changes"; return; }
  const diff = dirtyBits(liveFacts || {}, liveSizing || {}, { util: util, ctx: ctx, agents: agents });
  d.classList.toggle("on", diff.length > 0);
  d.textContent = diff.length ? `unsaved: ${diff.join(", ")}` : "unsaved changes";
}

function paintState(s) {
  lastState = s;
  const up = s.upstream || {};
  liveFacts = up.live || {};
  // /api/state carries the same sizing block as the telemetry event, so the
  // panel is populated on first paint rather than staying blank for up to 2 s.
  if (s.sizing) { liveSizing = s.sizing; paintRequestStats(); }
  paintBoot(s.boot || {});
  // Show the configuration the server is ACTUALLY running at, not the page's
  // defaults, until the operator touches a control. Both of these used to be
  // one-directional: util was read back but the agent count never was, so the
  // "Parallel agents" field showed 1 against a server running 8, and the
  // recommendation column below it disagreed with the input above it.
  if (!userPicked && liveFacts.util_effective && Math.abs(liveFacts.util_effective - util) > 0.002) {
    util = liveFacts.util_effective;
    const u = $("util");
    if (u) u.value = String(Math.round(util * 100));
    estimate();
  }
  // The running engine's --max-num-seqs, read back into the field that sets
  // it. liveSizing.max_num_seqs is that value (app._sizing_payload reads it off
  // the live process), so the input and the recommendation can never disagree.
  const runSeqs = liveSizing.max_num_seqs;
  if (!userPicked && runSeqs && runSeqs !== agents) {
    agents = runSeqs;
    const a = $("agents");
    if (a) a.value = String(agents);
    estimate();
  }
  paintDirty(!!up.up);
  const sv = s.supervisor || {};
  const phase = busyPhase(sv);
  const pill = $("pill"), pillTxt = $("pillTxt");
  if (pill && pillTxt) {
    // `.pill.busy` shipped in the stylesheet with no painter, exactly like
    // #dirty. During a boot the upstream is down, so the pill read
    // "Not reachable" in grey while the operator had just pressed the button
    // that started it -- the page describing a boot in progress as a fault.
    pill.className = "pill" + (phase ? " busy" : up.up ? "" : " off");
    pillTxt.textContent = phase
      ? phase + "\u2026"
      : up.up ? "Serving" : "Not reachable";
  }
  const sm = $("sModel");
  if (sm) sm.textContent = up.model || "—";
  const urlChip = $("serverUrl");
  if (urlChip) urlChip.title = `Codex base URL — click to copy · upstream ${up.url || "—"}`;
  paintServingMeta();

  controlEnabled = !!s.control_enabled;
  const busy = !!phase;
  const upNow = !!up.up;
  ["apply", "stop"].forEach((id) => {
    const b = $(id);
    if (!b) return;
    // Stop only makes sense when something is running; Apply only when the
    // machine is not mid-transition.
    b.disabled = !controlEnabled || busy || (id === "stop" && !upNow);
    b.title = !controlEnabled
      ? (s.control_note || "supervisor unavailable")
      : busy ? `busy: ${sv.actual_state}` : "";
  });
  const smoke = $("smoke");
  if (smoke) { smoke.disabled = true; smoke.title = "not implemented yet"; }

  // A server running while intent says STOPPED is deliberately untouched.
  // Offer to adopt it rather than silently leaving Stop inert.
  const unadopted = upNow && controlEnabled &&
    (sv.desired_state === "STOPPED" || sv.actual_state === "UNMANAGED");
  let btn = $("adoptBtn");
  if (unadopted && !btn) {
    const bar = document.querySelector(".actions");
    if (bar) {
      btn = document.createElement("button");
      btn.id = "adoptBtn"; btn.className = "btn"; btn.type = "button";
      btn.textContent = "Manage running server";
      btn.title = "Bring the already-running server under Servedeck's control";
      bar.insertBefore(btn, bar.firstChild);
      wireControls();
    }
  } else if (!unadopted && btn) {
    btn.remove();
  }
  const stopBtn = $("stop");
  if (stopBtn && unadopted) {
    stopBtn.disabled = true;
    stopBtn.title = "Not managed yet — click 'Manage running server' first";
  }

  // Which row is serving comes from /api/state's `identity`: the live
  // process's own --model first, then /v1/models resolved against the cache.
  // It used to be a served-model-name match with a "weights within 0.5 GiB"
  // fallback -- an alias an operator reuses across models, plus a heuristic
  // that cheerfully marks a DIFFERENT model of similar size as the one that
  // is running. Both were wrong on this box at the same time, and the result
  // was a dashboard that could not name the model it was serving.
  const ident = up.identity || {};
  const names = ident.served_names || (up.model ? [up.model] : []);
  if (MODELS.length && (ident.repo_id || names.length)) {
    let servingIdx = -1;
    MODELS.forEach((m, i) => {
      m.serving = ident.repo_id
        ? m.repo_id === ident.repo_id
        : names.includes(m.repo_id) || names.includes(m.name);
      if (m.serving) servingIdx = i;
    });
    if (ident.mismatch) {
      log(
        `served-model-name ${names.join(", ")} does not belong to the model ` +
          `actually loaded (${ident.repo_id}) — it was reused from another run`,
        "w"
      );
    }
    // Select what is ACTUALLY running, not whatever sorted first. Otherwise
    // every panel describes a model the user is not using.
    if (servingIdx >= 0 && !userPicked) { sel = servingIdx; renderCtx(null); estimate(); }
    renderModels();
  }
}

/* ------------------------------------------------------------ control --- */
async function post(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
  return d;
}

function wireControls() {
  const adopt = $("adoptBtn");
  if (adopt) {
    adopt.onclick = async () => {
      try {
        const d = await post("/api/server/adopt");
        log(`adopted the running server (pid ${d.pid})`, "g");
      } catch (e) { log("adopt failed: " + e.message, "e"); }
    };
  }

  const stop = $("stop");
  if (stop) {
    stop.onclick = async () => {
      // Stopping frees the GPU and kills in-flight requests. Confirm, and say
      // exactly what is running so the click is informed.
      const running = liveMetrics.running || 0;
      const what = liveFacts.kv_tokens ? ` (${fmt(liveFacts.kv_tokens)} KV tokens allocated)` : "";
      const extra = running > 0 ? `\n\n${running} request(s) are in flight and will fail.` : "";
      if (!confirm(`Stop the model server${what}?${extra}`)) return;
      stop.disabled = true;
      try { await post("/api/server/stop"); log("stop requested", "w"); }
      catch (e) { log("stop failed: " + e.message, "e"); stop.disabled = false; }
    };
  }

  const apply = $("apply");
  if (apply) {
    apply.onclick = async () => {
      const m = MODELS[sel];
      if (!m) return;
      if (lastEstimate && lastEstimate.can_apply === false) {
        log("refused: this configuration cannot start — see the blocker above", "e");
        return;
      }
      const unknown = lastEstimate && lastEstimate.confidence === "unknown";
      const msg = unknown
        ? `Start ${m.name} at util ${util.toFixed(2)}, ${fmt(ctx)} context?\n\n` +
          `Its VRAM footprint cannot be predicted for this architecture, so ` +
          `there is no capacity estimate to check against. Starting it is how ` +
          `the real figure gets measured — the engine will refuse safely if it ` +
          `does not fit.\n\nThe server will be unavailable for several minutes.`
        : `Restart with ${m.name} at util ${util.toFixed(2)}, ${fmt(ctx)} context?\n\n` +
          `The server will be unavailable for several minutes.`;
      if (!confirm(msg)) return;
      apply.disabled = true;
      // "Apply & restart" means RESTART when something is already serving.
      // /api/server/start is idempotent on the supervisor's side -- it returns
      // early when the server is READY -- so posting a new context length at a
      // running server returned 202 "accepted" and changed nothing at all:
      // same pid, same cmdline, same .config, and a success in the log.
      // /api/server/restart is the only endpoint that relaunches, and it
      // carries the same settings body.
      const sv = (lastState && lastState.supervisor) || {};
      const upNow = !!(lastState && lastState.upstream && lastState.upstream.up);
      const path = (upNow || sv.actual_state === "READY")
        ? "/api/server/restart" : "/api/server/start";
      try {
        await post(path, {
          repo_id: m.repo_id, backend: m.backend, util, ctx, max_num_seqs: agents,
        });
        log(`applying ${m.name} …`, "g");
      } catch (e) { log("apply failed: " + e.message, "e"); apply.disabled = false; }
    };
  }
}

/* -------------------------------------------------------------- init ---- */
async function init() {
  renderCtx(null);

  // The URL every client (Codex) must point at. Served from the same origin
  // as this page, so location.origin IS the gateway origin — no server round-trip.
  const urlChip = $("serverUrl");
  if (urlChip) {
    const url = `${location.origin}/v1`;
    urlChip.textContent = url;
    let resetTimer = null;
    urlChip.onclick = async () => {
      let ok = false;
      try {
        await navigator.clipboard.writeText(url);
        ok = true;
      } catch (_) {
        // navigator.clipboard is undefined/throws outside secure contexts.
        // Select the text so a manual Ctrl-C still works.
        const range = document.createRange();
        range.selectNodeContents(urlChip);
        const selRange = getSelection();
        selRange?.removeAllRanges();
        selRange?.addRange(range);
      }
      urlChip.classList.toggle("copied", ok);
      clearTimeout(resetTimer); // rapid clicks must not let an old timer wipe a fresh flash
      resetTimer = setTimeout(() => urlChip.classList.remove("copied"), 1200);
      log(ok ? `copied ${url}` : `clipboard unavailable — ${url} selected, copy manually`, ok ? "g" : "w");
    };
  }

  const u = $("util");
  if (u) {
    u.value = String(Math.round(util * 100));
    u.oninput = (e) => { userPicked = true; util = +e.target.value / 100; estimate(); };
  }

  const cx = $("ctx");
  if (cx) {
    cx.oninput = (e) => {
      // Touching a control is the operator speaking. Without this flag the
      // next state event overwrites the value the operator just set with the
      // one the running server was started with, which is how a moved slider
      // sprang back and how the Apply confirmation could quote a util the
      // operator had already changed.
      userPicked = true;
      ctx = +e.target.value;
      const v = $("ctxV");
      if (v) v.textContent = fmt(ctx);   // move the readout with the thumb
      estimate();
    };
  }

  const ag = $("agents");
  if (ag) {
    ag.oninput = (e) => {
      // A number input, not a slider: the value is the whole point and a thumb
      // cannot be aimed at 7. An empty or out-of-range box must not send
      // max_num_seqs: NaN to the backend.
      const n = Math.round(+e.target.value);
      agents = isFinite(n) ? Math.min(64, Math.max(1, n)) : 1;
      // Echo the clamped value back into the box. Typing 99 left the box
      // reading 99 while `agents` -- the number POSTed as max_num_seqs -- held
      // 64, so the control displayed a value the page was not using. An empty
      // box is left alone: that is someone mid-edit, not an out-of-range
      // entry, and the request falls back to 1 without rewriting their text.
      if (e.target.value !== "" && e.target.value !== String(agents)) {
        e.target.value = String(agents);
      }
      userPicked = true;
      estimate();
    };
  }

  // The recommendation is computed from measured request sizes and lives in
  // the live panel; this writes it into the field it constrains. It is a
  // suggestion the operator confirms by clicking, never an automatic write:
  // paintState deliberately does not move `agents` on its own.
  const useRec = $("useRec");
  if (useRec) {
    useRec.onclick = () => {
      const rec = (liveSizing || {}).recommended;
      if (!rec || !isFinite(rec.n)) return;
      agents = Math.min(64, Math.max(1, Math.round(rec.n)));
      const a = $("agents");
      if (a) a.value = String(agents);
      userPicked = true;
      estimate();
      log(`agent count set to the recommendation: ${agents}`, "g");
    };
  }

  try {
    const r = await fetch("/api/models");
    const d = await r.json();
    MODELS = d.models || [];
    paintDisk(d.disk);
    const i = MODELS.findIndex((m) => m.servable);
    sel = i >= 0 ? i : 0;
    renderModels();
    // Only now is the selected model's own ceiling known — the range drawn at
    // init() was the DEFAULT_MAX_CTX fallback.
    renderCtx(null);
    estimate();
    log(`${MODELS.length} models on disk, ${MODELS.filter((m) => m.servable).length} servable`);
  } catch (e) {
    log("could not load models: " + e, "e");
  }

  // The scan behind these numbers has a 5 s TTL server-side; repolling it is
  // what makes a 95 GiB deletion visible without a reload. The old panel
  // cached its scan for the life of the process and went on reporting sizes
  // for files that no longer existed.
  setInterval(refreshDisk, 15000);

  const es = new EventSource("/api/events");
  es.addEventListener("state", (ev) => paintState(JSON.parse(ev.data)));
  es.addEventListener("telemetry", (ev) => paintTelemetry(JSON.parse(ev.data)));
  es.addEventListener("notice", (ev) => {
    const n = JSON.parse(ev.data);
    log(n.body || n.code, n.level === "warn" ? "w" : "e");
  });
  // The page used to ALSO poll /api/state every 5 s on top of this stream,
  // which already pushes a state event on that same interval. Two sources for
  // one readout means whichever paint lands last wins, and the poll's
  // `catch (_) {}` meant the duplicate could fail forever without a trace.
  // EventSource reconnects by itself; what it cannot do is refresh the disk
  // scan, which is what the interval above is for. The drop is reported once
  // per outage rather than once per reconnect attempt.
  let streamDown = false;
  es.onerror = () => {
    if (streamDown) return;
    streamDown = true;
    log("event stream dropped — reconnecting", "w");
  };
  es.onopen = () => { streamDown = false; };

  // The log drawer. Collapsed by default because it is the least read thing on
  // the page and it used to hold 190px of it permanently. The button says how
  // many lines are waiting, so opening it is an informed click, and the count
  // keeps updating while folded — an error arriving is visible as a number
  // changing even when the stream itself is hidden.
  const fold = $("logFold");
  if (fold) {
    const panel = $("logPanel");
    const count = () => {
      const L = $("log");
      const n = L && L.children ? L.children.length : 0;
      fold.textContent = (panel && panel.classList.contains("folded") ? "show " : "hide ") + n;
    };
    fold.onclick = () => {
      const folded = panel.classList.toggle("folded");
      fold.setAttribute("aria-expanded", folded ? "false" : "true");
      count();
      if (!folded) { const L = $("log"); if (L) L.scrollTop = L.scrollHeight; }
    };
    // log() appends without asking the drawer, so the count is refreshed on a
    // short tick of its own rather than by instrumenting every call site.
    count();
    setInterval(count, 2000);
  }

  wireControls();
  log("connected");
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    spark();
    reqHist((liveSizing || {}).window || {});
  });
  // The canvases are sized to their CSS box (fitCanvas), so a width change —
  // a resize, or the drawer opening — invalidates the bitmap. Repaint rather
  // than leave a stretched old drawing behind. Debounced: resize fires per
  // pixel of drag, and each repaint re-reads the CSS custom properties.
  let rsTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(rsTimer);
    rsTimer = setTimeout(() => {
      spark();
      reqHist((liveSizing || {}).window || {});
    }, 120);
  });
}

init();


/* The disk line under "Models on disk".
 *
 * Filesystem figures come from statvfs (the kernel's own answer, identical to
 * df byte-for-byte), never from adding up a directory walk: a walk cannot see
 * free space at all, and under-reports "used" by every tree it may not read.
 *
 * The percentage is used / (used + avail), which is how df computes Use%.
 * Using used / total instead would ignore ext4's root-reserved blocks (45 GiB
 * here) and read four points low against the df the operator will check.
 */
function paintDisk(d) {
  const el = $("dfree");
  if (!el || !d) return;
  el.textContent = `${bytesTxt(d.avail_bytes, 0)} free`;
  el.title =
    `${bytesTxt(d.used_bytes, 1)} used of ${bytesTxt(d.total_bytes, 1)} (${d.used_pct}% full) on ${d.path}\n` +
    `hub cache: ${bytesTxt(d.hub_bytes, 1)} of models, ${bytesTxt(d.hub_local_bytes, 1)} of it on this filesystem\n` +
    `from statvfs; binary units (1 GiB = 1024³ bytes)`;
  const hub = $("dhub");
  if (hub) {
    hub.textContent = `${bytesTxt(d.hub_local_bytes, 1)} of models here`;
    hub.title =
      d.hub_foreign_bytes > 0
        ? `${bytesTxt(d.hub_bytes, 1)} of models in the hub cache, but ${bytesTxt(d.hub_foreign_bytes, 1)} of that is on another mount`
        : `${bytesTxt(d.hub_bytes, 1)} of models in the hub cache, each blob counted once`;
  }
}

async function refreshDisk() {
  try {
    paintDisk(await (await fetch("/api/disk")).json());
  } catch (_) {
    /* the panel keeps its last real figure rather than inventing one */
  }
}
