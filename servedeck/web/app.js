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
  return "n/a";
}

/* One figure of the throughput strip.
 *
 * Returns {value, note}. `value` is never a bare number with no unit and
 * never an unexplained dash: when there is no reading it is the string "n/a"
 * and `note` carries the reason the backend gave (idle, counters reset, not
 * published by this build, backend unreachable). A lone unlabelled number was
 * the original complaint — you could not tell prefill from decode — and a
 * lone em dash is the same defect one step further along: it does not say
 * whether the server is quiet or the metric is missing.
 *
 * `live` is the window reading, `life` the lifetime companion. A lifetime
 * figure is shown with a "~" and says so, so it is never read as "now".
 */
function figure(live, life, reason, fmtFn, lifeNote) {
  if (typeof live === "number" && isFinite(live)) {
    return { value: fmtFn(live), note: "last 2 s window" };
  }
  if (typeof life === "number" && isFinite(life)) {
    return { value: "~" + fmtFn(life), note: lifeNote + (reason ? " — " + reason : "") };
  }
  return { value: "n/a", note: reason || "no reading" };
}

function secsTxt(v) {
  if (typeof v !== "number" || !isFinite(v)) return "n/a";
  return v >= 10 ? v.toFixed(0) + " s" : v >= 1 ? v.toFixed(1) + " s"
       : Math.round(v * 1000) + " ms";
}

/* The throughput strip: prefill, decode and TTFT, always all three.
 *
 * Rendering only one of them is what the dashboard did, and it made a slow
 * time-to-first-token indistinguishable from a slow decode — the two differ
 * by ~70x on this box, so the single number was not merely ambiguous, it was
 * off by nearly two orders of magnitude depending on which one you assumed.
 */
function paintThroughput() {
  const m = liveMetrics || {};
  const cells = [
    ["thPrefill", figure(m.prefill_tok_s, m.prefill_tok_s_avg, m.prefill_reason,
                         (v) => rateTxt(v), "lifetime, per second of prefill time")],
    ["thDecode", figure(m.gen_tok_s, m.gen_tok_s_avg, m.gen_reason,
                        (v) => rateTxt(v, 1), "lifetime, per second of decode time")],
    ["thTtft", figure(m.ttft_s, m.ttft_s_avg, m.ttft_reason,
                      secsTxt, "lifetime mean")],
  ];
  cells.forEach(([id, f]) => {
    const n = $(id), sub = $(id + "S");
    if (n) { n.textContent = f.value; n.className = "n mono" + (f.value === "n/a" ? " na" : ""); }
    if (sub) sub.textContent = f.note;
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

/* Redraw the context slider against the bounds in an estimate.
 *
 * `d` is the /api/capacity/estimate payload, or null before the first one has
 * come back — in which case the only bound known is the selected model's
 * ceiling, and the fit bound is left blank rather than guessed at.
 */
function renderCtx(d) {
  const el = $("ctx");
  if (!el) return;
  const modelMax = Number((d && d.ctx_max_model) || MODELS[sel]?.model_max_ctx) || DEFAULT_MAX_CTX;
  const fit = d ? Number(d.ctx_max_fit) || 0 : 0;
  // The ceiling is whichever real limit binds first. A zero fit means the
  // budget holds nothing at this utilization: keep the model ceiling as the
  // range so the slider still moves, and let the blocker explain why.
  const ceiling = Math.max(MIN_CTX, fit > 0 ? Math.min(modelMax, fit) : modelMax);
  el.min = String(MIN_CTX);
  el.max = String(ceiling);
  el.step = String(CTX_STEP);
  if (ctx > ceiling) ctx = ceiling;
  if (ctx < MIN_CTX) ctx = MIN_CTX;
  el.value = String(ctx);

  const set = (id, v) => { const e = $(id); if (e) e.textContent = v; };
  set("ctxV", fmt(ctx));
  set("ctxMin", ctxLabel(MIN_CTX));
  set("ctxMax", ctxLabel(ceiling));
  // Name which of the two limits is binding, so a ceiling that moves when the
  // agent count changes is not mistaken for the model's own limit.
  set("ctxBound", fit > 0 && fit < modelMax
    ? "KV fits " + ctxLabel(fit)
    : "model " + ctxLabel(modelMax));
  const bound = $("ctxBound");
  if (bound) bound.title = fit > 0 && fit < modelMax
    ? `the KV budget holds ${fmt(fit)} tokens per agent at this utilization`
    : `the checkpoint's own max_position_embeddings is ${fmt(modelMax)}`;
  set("agentsV", agents);
  set("agentsFit", d && d.agents_at_ctx ? "fits " + d.agents_at_ctx : "—");
}
let MODELS = [];
let sel = 0;
let util = 0.95;   // overwritten from the running server via /api/state
let ctx = 262144;
let agents = 1;   // parallel agents the context is being sized for
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
        renderCtx(null);   // a different model has a different ceiling
        estimate();
      };
    }
    L.appendChild(b);
  });
  const c = $("mcount");
  if (c) c.textContent = MODELS.length;
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

function paintEstimate(d) {
  const set = (id, v) => { const e = $(id); if (e) e.textContent = v; };
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
  paintThroughput();
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
  // Every figure is named. "99 tok/s" on its own does not say whether the
  // server is slow to START answering or slow to KEEP answering, and those
  // have different fixes.
  const decode = typeof liveMetrics.gen_tok_s === "number"
    ? rateTxt(liveMetrics.gen_tok_s, 1)
    : typeof liveMetrics.gen_tok_s_avg === "number"
      ? "~" + rateTxt(liveMetrics.gen_tok_s_avg, 1) : "n/a";
  const ttft = typeof liveMetrics.ttft_s === "number"
    ? secsTxt(liveMetrics.ttft_s)
    : typeof liveMetrics.ttft_s_avg === "number"
      ? "~" + secsTxt(liveMetrics.ttft_s_avg) : "n/a";
  meta.textContent = up.up
    ? `${base} · ${fmt(liveCtx)} ctx` +
      ` · prefill ${prefillTxt(liveMetrics)}` +
      ` · decode ${decode}` +
      ` · TTFT ${ttft}` +
      ` · up ${uptimeTxt(s.server_uptime_s)}`
    : `nothing serving on ${base}`;
  meta.title = up.up
    ? "prefill (prompt processing), decode (token generation) and " +
      "time-to-first-token, from the engine's own counters over the last 2 s " +
      "poll window. A ~ prefix marks a lifetime average, shown when nothing " +
      "happened in the window; n/a with a reason is shown when there is no " +
      "figure at all. The full strip below carries the reasons."
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
      try {
        await post("/api/server/start", {
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
    u.oninput = (e) => { util = +e.target.value / 100; estimate(); };
  }

  const cx = $("ctx");
  if (cx) {
    cx.oninput = (e) => {
      ctx = +e.target.value;
      const v = $("ctxV");
      if (v) v.textContent = fmt(ctx);   // move the readout with the thumb
      estimate();
    };
  }

  const ag = $("agents");
  if (ag) {
    ag.oninput = (e) => {
      agents = +e.target.value;
      const v = $("agentsV");
      if (v) v.textContent = agents;
      estimate();
    };
  }

  try {
    const r = await fetch("/api/models");
    const d = await r.json();
    MODELS = d.models || [];
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
