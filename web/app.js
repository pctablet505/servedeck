/* Coldstart frontend.
 *
 * Ported from the approved design prototype. Two deliberate differences:
 *  1. The prototype's capacity() is GONE. The browser never computes capacity —
 *     one implementation lives server-side and is used by both the estimate and
 *     the validation, so they cannot drift.
 *  2. No simulated data. Everything arrives from /api/* and /api/events.
 */
const $ = (id) => document.getElementById(id);
const fmt = (n) => (n == null ? "—" : Number(n).toLocaleString("en-US"));

const CTXS = [8192, 32768, 65536, 131072, 262144];
let MODELS = [];
let sel = 0;
let util = 0.95;   // overwritten from the running server via /api/state
let ctx = 262144;
let lastEstimate = null;
let controlEnabled = false;
let liveFacts = {};      // the RUNNING engine's own numbers, from /api/state
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
    else if (m.trust === "measured") tags.push('<span class="tag meas">measured</span>');
    else tags.push('<span class="tag est">estimated</span>');
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
      b.onclick = () => { sel = i; userPicked = true; renderModels(); estimate(); };
    }
    L.appendChild(b);
  });
  const c = $("mcount");
  if (c) c.textContent = MODELS.length;
}

function renderCtx() {
  const B = $("ctxBtns");
  if (!B) return;
  B.innerHTML = "";
  CTXS.forEach((c) => {
    const b = document.createElement("button");
    b.className = "segbtn";
    b.type = "button";
    b.setAttribute("aria-pressed", c === ctx ? "true" : "false");
    b.textContent = c >= 1024 ? c / 1024 + "k" : String(c);
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
    const d = await r.json();
    if (d.error) { showFindings([{ level: "block", title: "Estimate failed", detail: d.error }]); return; }
    lastEstimate = d;
    paintEstimate(d);
  } catch (e) {
    showFindings([{ level: "block", title: "Backend unreachable", detail: String(e) }]);
  }
}

function paintEstimate(d) {
  const set = (id, v) => { const e = $(id); if (e) e.textContent = v; };
  const kv = $("dKv");
  if (kv) kv.innerHTML = `${d.kv_gib.toFixed(1)}<span style="font-size:13px;font-weight:400"> GiB</span>`;
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
  set("vramTxt",
    `weights ${(d.weights_gib ?? 0).toFixed ? d.weights_gib.toFixed(1) : "—"} · ` +
    `KV ${d.kv_gib.toFixed(1)} · budget ${d.budget_gib.toFixed(1)} GiB`);

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

  if (g.used_mib != null) set("gpuUsed", fmt(g.used_mib));

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
  spark();
}

function paintAgentSizing() {
  if (!lastEstimate) return;
  const set = (id, v) => { const e = $(id); if (e) e.textContent = v; };
  set("capWorst", lastEstimate.agents_at_ctx);
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
function paintState(s) {
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
  const meta = $("sMeta");
  if (meta) {
    meta.textContent = up.up
      ? `:${up.port} · ${fmt(ctx)} ctx · ${liveMetrics.gen_tok_s || 0} tok/s · up ${Math.floor((s.uptime_s || 0) / 60)}m`
      : `nothing serving on :${up.port}`;
  }

  controlEnabled = !!s.control_enabled;
  ["apply", "stop", "smoke"].forEach((id) => {
    const b = $(id);
    if (b) {
      b.disabled = !controlEnabled;
      if (!controlEnabled) b.title = s.control_note || "not wired yet";
    }
  });

  if (MODELS.length && up.model) {
    let servingIdx = -1;
    MODELS.forEach((m, i) => {
      m.serving = (m.repo_id === up.model) || (m.name === up.model) ||
                  (liveFacts.weights_gib != null && Math.abs((m.weights_gib || -1) - liveFacts.weights_gib) < 0.5);
      if (m.serving) servingIdx = i;
    });
    // Select what is ACTUALLY running, not whatever sorted first. Otherwise
    // every panel describes a model the user is not using.
    if (servingIdx >= 0 && !userPicked) { sel = servingIdx; estimate(); }
    renderModels();
  }
}

/* -------------------------------------------------------------- init ---- */
async function init() {
  renderCtx();

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

  log("connected to Coldstart");
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", spark);
}

init();
