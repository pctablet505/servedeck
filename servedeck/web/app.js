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

/* Prefill throughput for the serving line.
 *
 * Prefill is bursty: at --max-num-seqs 1 a 10k-token prompt prefills for
 * ~10-17 s and then nothing prefills for minutes, so the windowed rate is
 * genuinely unknown most of the time. Fall back to the lifetime figure —
 * computed prompt tokens per second OF PREFILL TIME, which does not decay
 * while the server sits idle — and mark it "~" so a live reading and a
 * lifetime average are never mistaken for each other.
 */
function prefillTxt(m) {
  if (typeof m.prefill_tok_s === "number") return rateTxt(m.prefill_tok_s);
  if (typeof m.prefill_tok_s_avg === "number") return "~" + rateTxt(m.prefill_tok_s_avg);
  return "—";
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
function trustTag(trust) {
  if (trust === "measured") return '<span class="tag meas">measured</span>';
  if (trust === "measured_other_ctx")
    return '<span class="tag meas" title="measured, but at a different context length">measured*</span>';
  if (trust === "unknown")
    return '<span class="tag est" title="this architecture\'s VRAM footprint cannot be predicted from disk size">unknown</span>';
  return '<span class="tag est">estimated</span>';
}

/* The context lengths offered for the SELECTED model.
 *
 * This used to be a literal array, which is wrong in both directions: it
 * offered lengths a small model cannot reach (the engine refuses at boot,
 * minutes later) and hid the top of a large one — GLM-5.3 declares
 * max_position_embeddings 1,048,576 and the list stopped at 262,144, so a
 * quarter of the model was unreachable from the UI. The registry has parsed
 * that ceiling all along; build the ladder from it.
 *
 * Powers of two up to the model's own ceiling, with the ceiling itself always
 * last even when it is not a power of two. DEFAULT_MAX_CTX only applies when
 * no model is selected or its config.json declares no ceiling.
 */
const MIN_CTX = 8192;
const DEFAULT_MAX_CTX = 262144;   // mirrors app.DEFAULT_MODEL_MAX_CTX

function ctxChoices() {
  const ceiling = Number(MODELS[sel]?.model_max_ctx) || DEFAULT_MAX_CTX;
  const out = [];
  for (let c = MIN_CTX; c < ceiling; c *= 2) out.push(c);
  out.push(ceiling);
  return out;
}
let MODELS = [];
let sel = 0;
let util = 0.95;   // overwritten from the running server via /api/state
let ctx = 262144;
let lastEstimate = null;
let controlEnabled = false;
let liveFacts = {};      // the RUNNING engine's own numbers, from /api/state
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
function renderModels() {
  const L = $("mlist");
  if (!L) return;
  L.innerHTML = "";
  MODELS.forEach((m, i) => {
    const b = document.createElement("button");
    b.className = "mcard";
    b.type = "button";
    b.setAttribute("aria-pressed", i === sel ? "true" : "false");
    const tags = [];
    tags.push(`<span class="tag">${m.quant || "—"}</span>`);
    if (m.serving) tags.push('<span class="tag live">serving</span>');
    if (!m.servable) tags.push('<span class="tag est">unservable</span>');
    else tags.push(trustTag(m.trust));
    b.innerHTML =
      `<span class="r1"><span class="nm"></span><span class="sz mono">${m.disk_gib || "—"} GB</span></span>
       <span class="r2">${tags.join("")}</span>
       <span class="note"></span>`;
    b.querySelector(".nm").textContent = m.name;
    b.querySelector(".note").textContent =
      m.unservable_reason || `${m.backend || "?"} · ${fmt(m.model_max_ctx)} ctx`;
    if (!m.servable) {
      b.disabled = true;
      b.style.opacity = ".5";
      b.style.cursor = "not-allowed";
    } else {
      b.onclick = () => {
        sel = i;
        userPicked = true;
        renderModels();
        renderCtx();   // a different model has a different ceiling
        estimate();
      };
    }
    L.appendChild(b);
  });
  const c = $("mcount");
  if (c) c.textContent = MODELS.length;
}

function renderCtx() {
  const B = $("ctxBtns");
  if (!B) return;
  const choices = ctxChoices();
  // Switching to a smaller model must not leave a selection the model cannot
  // serve: the engine would refuse it at boot, minutes later, with an error
  // that reads nothing like "you picked too big a number here".
  if (!choices.includes(ctx)) {
    ctx = choices.reduce((best, c) => (c <= ctx && c > best ? c : best), choices[0]);
  }
  B.innerHTML = "";
  choices.forEach((c) => {
    const b = document.createElement("button");
    b.className = "segbtn";
    b.type = "button";
    b.setAttribute("aria-pressed", c === ctx ? "true" : "false");
    // "1024k" for a 1,048,576-token ceiling reads as a typo. Once the ladder
    // is derived from the model, million-token contexts are reachable.
    b.textContent = c >= 1048576 ? +(c / 1048576).toFixed(2) + "M"
                  : c >= 1024    ? c / 1024 + "k"
                  : String(c);
    b.onclick = () => { ctx = c; renderCtx(); estimate(); };
    B.appendChild(b);
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
      body: JSON.stringify({ repo_id: m.repo_id, util, ctx, max_num_seqs: 1 }),
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

function paintEstimate(d) {
  const set = (id, v) => { const e = $(id); if (e) e.textContent = v; };
  const kv = $("dKv");
  if (kv) {
    kv.innerHTML = (typeof d.kv_gib === "number")
      ? `${d.kv_gib.toFixed(1)}<span style="font-size:13px;font-weight:400"> GiB</span>`
      : "—";
  }
  set("dKvTok", fmt(d.kv_tokens) + " tokens");
  set("dAgents", d.agents_at_ctx);
  const a = $("dAgents");
  if (a) a.classList.toggle("bad", d.agents_at_ctx < 1);
  set("dAgentsS", "at " + fmt(ctx) + " context");
  set("dCtx", fmt(d.max_single_ctx));
  set("dCtxS", d.max_single_ctx >= (MODELS[sel].model_max_ctx || 262144)
    ? "model ceiling reached" : "below the model's " + fmt(MODELS[sel].model_max_ctx));

  const badge = $("dBadge");
  if (badge) {
    const meas = d.confidence && d.confidence.startsWith("measured");
    badge.textContent = d.confidence || "estimated";
    badge.className = "tag " + (meas ? "meas" : "est");
  }

  const bar = d.bar || {};
  const seg = (id, pct, label) => {
    const e = $(id);
    if (!e) return;
    e.style.width = (pct || 0) + "%";
    if (label !== undefined) e.textContent = (pct > 10 ? label : "");
  };
  seg("segW", bar.weights_pct, "weights");
  seg("segK", bar.kv_pct, "KV");
  seg("segO", bar.overhead_pct);
  seg("segF", bar.free_pct);
  // `(x ?? 0).toFixed` is truthy even when x is null — (0).toFixed is a
  // function — so the old guard passed and then threw on the real null.
  // weights_gib IS null for models whose weights cannot be estimated
  // (host-offload architectures), which is a normal state, not an error.
  const gib = (v) => (typeof v === "number" ? v.toFixed(1) : "—");
  set("vramTxt",
    `weights ${gib(d.weights_gib)} · KV ${gib(d.kv_gib)} · budget ${gib(d.budget_gib)} GiB`);

  showFindings(d.findings || []);
  paintAgentSizing();
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

function spark() {
  const c = $("spark");
  if (!c || !c.getContext) return;
  const x = c.getContext("2d");
  const cs = getComputedStyle(document.documentElement);
  const acc = cs.getPropertyValue("--accent").trim() || "#0E7C7B";
  const ln = cs.getPropertyValue("--line").trim() || "#ccc";
  const w = c.width, h = c.height;
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

function paintTelemetry(t) {
  liveMetrics = t.vllm || {};
  const g = t.gpu || {};
  const set = (id, v) => { const e = $(id); if (e) e.textContent = v; };

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
  set("mWait", reachable ? liveMetrics.waiting : "—");
  set("mPre", reachable ? liveMetrics.preemptions : "—");

  const w = $("mWait");
  if (w) w.className = "n mono" + (reachable && liveMetrics.waiting > 0 ? " hot" : "");
  const p = $("mPre");
  if (p) p.className = "n mono" + (reachable && liveMetrics.preemptions > 8 ? " bad" : "");

  set("avgCtx", reachable && liveMetrics.avg_prompt_tokens ? fmt(liveMetrics.avg_prompt_tokens) : "—");
  set("avgSrc", reachable
    ? `${liveMetrics.prompt_token_count || 0} requests since server start`
    : "backend not reachable");

  paintAgentSizing();
  paintServingMeta();
  spark();
}

function paintAgentSizing() {
  if (!lastEstimate) return;
  const set = (id, v) => { const e = $(id); if (e) e.textContent = v; };
  set("capWorst", lastEstimate.agents_at_ctx);
  set("capWorstD", "every agent at full " + fmt(ctx));
  const avg = liveMetrics.avg_prompt_tokens;
  // prefer the running engine's real KV total; fall back to the estimate for
  // the selected model when nothing is serving
  const total = liveFacts.kv_tokens || lastEstimate.kv_tokens;
  if (avg && avg > 0) {
    set("capAvg", Math.floor(total / avg));
    set("capAvgD", "at " + fmt(avg) + " avg");
  } else {
    set("capAvg", "—");
    set("capAvgD", "no traffic measured yet");
  }
  const os = $("oversub");
  if (!os) return;
  if (liveMetrics.waiting_capacity > 0) {
    os.className = "oversub on";
    os.innerHTML = `<b>KV-bound right now.</b> ${liveMetrics.waiting_capacity} request(s) waiting on capacity. Reduce agents, lower per-agent context, or raise utilization.`;
  } else if (liveMetrics.preemptions > 8) {
    os.className = "oversub on";
    os.innerHTML = `<b>${liveMetrics.preemptions} preemptions since restart.</b> vLLM is evicting and recomputing KV — real work is being thrown away. This is the empirical signal that the agent count is too high.`;
  } else {
    os.className = "oversub";
  }
}

/* ------------------------------------------------------------- state ---- */

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
  // Both throughputs, never one: generation tok/s alone cannot tell a 17 s
  // time-to-first-token apart from a slow decode, and the two can differ by
  // ~70x on a PCIe-bound decode.
  meta.textContent = up.up
    ? `${base} · ${fmt(liveCtx)} ctx` +
      ` · ${rateTxt(liveMetrics.gen_tok_s, 1)} gen` +
      ` · ${prefillTxt(liveMetrics)} prefill` +
      ` · up ${uptimeTxt(s.server_uptime_s)}`
    : `nothing serving on ${base}`;
  meta.title = up.up
    ? "generation and prefill throughput over the last 2 s poll window, from " +
      "the engine's own counters. \u2014 means nothing was running in that " +
      "window; a ~ prefix marks the lifetime prefill average (tokens per " +
      "second of prefill time), shown when no prefill happened just now."
    : "";
}

function paintState(s) {
  lastState = s;
  const up = s.upstream || {};
  liveFacts = up.live || {};
  // Show the utilization the server is ACTUALLY running at, not a hardcoded
  // default, until the user moves the slider themselves.
  if (!userPicked && liveFacts.util_effective && Math.abs(liveFacts.util_effective - util) > 0.002) {
    util = liveFacts.util_effective;
    const u = $("util");
    if (u) u.value = String(Math.round(util * 100));
    estimate();
  }
  const pill = $("pill"), pillTxt = $("pillTxt");
  if (pill && pillTxt) {
    pill.className = "pill" + (up.up ? "" : " off");
    pillTxt.textContent = up.up ? "Serving" : "Not reachable";
  }
  const sm = $("sModel");
  if (sm) sm.textContent = up.model || "—";
  const urlChip = $("serverUrl");
  if (urlChip) urlChip.title = `Codex base URL — click to copy · upstream ${up.url || "—"}`;
  paintServingMeta();

  controlEnabled = !!s.control_enabled;
  const sv = s.supervisor || {};
  const busy = ["STARTING", "PREFLIGHT", "STOPPING", "DRAINING"].includes(sv.actual_state);
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
  if (sv.actual_state && sv.actual_state !== "READY") {
    const meta = $("sMeta");
    if (meta && busy) meta.textContent = `${sv.actual_state.toLowerCase()}…`;
  }

  if (MODELS.length && up.model) {
    let servingIdx = -1;
    MODELS.forEach((m, i) => {
      m.serving = (m.repo_id === up.model) || (m.name === up.model) ||
                  (liveFacts.weights_gib != null && Math.abs((m.weights_gib || -1) - liveFacts.weights_gib) < 0.5);
      if (m.serving) servingIdx = i;
    });
    // Select what is ACTUALLY running, not whatever sorted first. Otherwise
    // every panel describes a model the user is not using.
    if (servingIdx >= 0 && !userPicked) { sel = servingIdx; renderCtx(); estimate(); }
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
      try {
        await post("/api/server/start", {
          repo_id: m.repo_id, backend: m.backend, util, ctx, max_num_seqs: 1,
        });
        log(`applying ${m.name} …`, "g");
      } catch (e) { log("apply failed: " + e.message, "e"); apply.disabled = false; }
    };
  }
}

/* -------------------------------------------------------------- init ---- */
async function init() {
  renderCtx();

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
    u.oninput = (e) => { util = +e.target.value / 100; estimate(); };
  }

  try {
    const r = await fetch("/api/models");
    const d = await r.json();
    MODELS = d.models || [];
    const i = MODELS.findIndex((m) => m.servable);
    sel = i >= 0 ? i : 0;
    renderModels();
    // Only now is the selected model's own ceiling known — the ladder drawn
    // at init() was the DEFAULT_MAX_CTX fallback.
    renderCtx();
    estimate();
    log(`${MODELS.length} models on disk, ${MODELS.filter((m) => m.servable).length} servable`);
  } catch (e) {
    log("could not load models: " + e, "e");
  }

  const es = new EventSource("/api/events");
  es.addEventListener("state", (ev) => paintState(JSON.parse(ev.data)));
  es.addEventListener("telemetry", (ev) => paintTelemetry(JSON.parse(ev.data)));
  es.addEventListener("notice", (ev) => {
    const n = JSON.parse(ev.data);
    log(n.body || n.code, n.level === "warn" ? "w" : "e");
  });
  es.onerror = () => log("event stream dropped — reconnecting", "w");

  setInterval(async () => {
    try { paintState(await (await fetch("/api/state")).json()); } catch (_) {}
  }, 5000);

  wireControls();
  log("connected");
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", spark);
}

init();
