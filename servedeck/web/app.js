/* servedeck v2 dashboard (REDESIGN-2026-09-12.md §2.5).
 *
 * The page renders `GET /api/state` and the `/api/events` stream. It computes
 * nothing. In particular it does NO capacity arithmetic: the KV pool, the
 * per-request cost, the headroom factor and both request counts arrive already
 * computed by servedeck/parallelism.py. A second formula here would eventually
 * disagree with that one, and two numbers on one screen that contradict each
 * other are worse than one number nobody checked -- there is no way to tell
 * from the screen which of them is the lie.
 *
 * Two rules the formatters exist to enforce:
 *
 *   * A missing number is an em dash, never 0. `gen_tok_s: null` means the
 *     engine reported no throughput in the sampling window; printing "0 tok/s"
 *     for that reads as "the machine got slow", which is the opposite of the
 *     truth, and it is the defect the v1 page shipped.
 *   * Every refusal is shown. A 409/404 carries `error.message` written by the
 *     backend that made the decision; the page prints that sentence, never a
 *     generic "failed" and never nothing at all.
 *
 * Everything above PURE HELPERS END is a pure function of its arguments --
 * no `document`, no fetch, no globals but EM and the two lookup tables. That
 * is what lets tests/test_ui_v2.py EXECUTE them in dukpy against real
 * /api/state payloads and assert on the strings, instead of grepping source.
 */
'use strict';

/* ===== PURE HELPERS BEGIN ===== */

/* U+2014. The one thing the page prints where a number is unknown. */
const EM = "—";

/* metrics.REASON_CODE -> what the tok/s cell says when there is no rate.
 * Matching on the CODE, never on the prose reason: the prose is English that
 * gets reworded, the code is the contract. */
const RATE_STATE = {
  ok: "",
  idle: "idle",
  no_baseline: "no baseline yet",
  reset: "counters reset",
  not_exposed: "not published by this build",
  unreachable: EM,
  unknown: EM
};

/* A number, grouped, or an em dash. Guards on the TYPE: 0 is a number and
 * must print as "0"; null/undefined/NaN are not and must not print as "0". */
function fmtNum(v) {
  if (typeof v !== "number" || !isFinite(v)) return EM;
  return v.toLocaleString();
}

/* A fraction (vLLM publishes KV usage as 0..1) as a percentage. */
function fmtPct(v) {
  if (typeof v !== "number" || !isFinite(v)) return EM;
  return (v * 100).toFixed(1) + "%";
}

/* Seconds under 90 s, then whole minutes, then "Nh MM". Truncating, not
 * rounding: an uptime that reads longer than it is would let a unit that
 * restarted 30 s ago look like it survived the last minute. */
function fmtUptime(s) {
  if (typeof s !== "number" || !isFinite(s) || s < 0) return EM;
  if (s < 90) return Math.floor(s) + "s";
  if (s < 3600) return Math.floor(s / 60) + "m";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s - h * 3600) / 60);
  return h + "h " + (m < 10 ? "0" : "") + m;
}

/* The tok/s cell. Never "0 tok/s" for an absent reading. */
function rateCell(met) {
  if (!met || !met.reachable) return EM;
  if (typeof met.gen_tok_s === "number") return fmtNum(met.gen_tok_s) + " tok/s";
  const label = RATE_STATE[met.gen_state];
  return label ? label : EM;
}

/* One row of the Live table, as strings. Returned rather than painted so the
 * test can read what a browser would see. */
function liveRow(m) {
  const met = m.metrics;
  const ok = !!(met && met.reachable);
  const aliases = m.aliases && m.aliases.length ? m.aliases.join(", ") : "";
  return {
    key: m.key,
    name: m.id,
    aliases: aliases,
    slot: m.slot + " · :" + m.port,
    unit: m.unit_state + (m.ready ? " · ready" : " · not ready"),
    ctx: m.ctx ? fmtNum(m.ctx) + " tok" : (m.ctx_error ? "unknown" : EM),
    kv: ok ? fmtPct(met.kv_usage_perc) : EM,
    queue: ok ? fmtNum(met.running) + " / " + fmtNum(met.waiting) : EM,
    rate: rateCell(met),
    uptime: fmtUptime(m.uptime_s),
    restarts: fmtNum(m.restarts),
    error: met && met.error ? String(met.error) : ""
  };
}

/* The Headroom panel, line by line.
 *
 * When the backend says the capacity is unavailable, that sentence is the
 * whole answer and NO capacity figure is printed beside it -- an estimate
 * rendered in the same font as a measurement is indistinguishable from one,
 * and acting on the wrong one over-subscribes the engine. `source` is printed
 * verbatim because it is the provenance of every number above it.
 */
function headroomLines(h) {
  if (!h) return [EM];
  const out = [];
  out.push("Free VRAM: " + (typeof h.free_mib === "number" ? fmtNum(h.free_mib) + " MiB" : EM));
  if (h.unavailable) {
    out.push("Capacity unavailable: " + h.unavailable);
    return out;
  }
  out.push("KV pool: " + fmtNum(h.pool_tokens) + " tokens");
  out.push("Full-context requests (" + fmtNum(h.full_ctx) + " tokens): " +
           fmtNum(h.full_context_requests));
  out.push(fmtNum(h.small_request_tokens) + "-token requests: " + fmtNum(h.small_requests));
  out.push("Per-request fixed cost " + fmtNum(h.fixed_cost_tokens) +
           " tokens, headroom factor " + fmtNum(h.headroom_fraction));
  if (h.source) out.push(h.source);
  if (h.note) out.push(h.note);
  return out;
}

function noticeClass(level) {
  return level === "error" ? "ev ev-error" : "ev ev-info";
}

function noticeText(n) {
  if (!n) return "";
  const reason = n.reason ? "[" + n.reason + "] " : "";
  return reason + (n.message || "");
}

/* One option of the main-slot select: the model and whether it is downloaded.
 * `r` is a /api/models row, not a /api/state model. */
function modelOption(r) {
  let disk;
  if (r.on_disk) {
    disk = typeof r.disk_gib === "number" ? "on disk, " + fmtNum(r.disk_gib) + " GiB" : "on disk";
  } else {
    disk = r.reason ? r.reason : "not on disk";
  }
  return r.id + " — " + disk;
}

/* An SSE `progress` frame as one line. */
function progressText(p) {
  if (!p) return "";
  const bits = [];
  if (p.kind) bits.push(String(p.kind));
  if (typeof p.marker_index === "number" && p.marker_index >= 0) {
    bits.push("step " + fmtNum(p.marker_index));
  }
  if (typeof p.elapsed_s === "number") bits.push(fmtNum(p.elapsed_s) + "s");
  const head = bits.join(" · ");
  const text = p.text ? String(p.text) : "";
  if (!head) return text;
  return text ? head + " — " + text : head;
}

/* The primary panel's state word. A holder beats everything else -- once the
 * unit is live, its own `ready` flag is the truth, not what the page
 * remembers about how it got there. `bootActive` and `failed` only matter
 * while nothing holds the slot yet. */
function mainStateText(holder, bootActive, failed) {
  if (holder) return holder.ready ? "READY" : "booting";
  if (bootActive) return "booting";
  if (failed) return "FAILED";
  return "stopped";
}

function doctorCells(c) {
  return { name: c.name, ok: c.ok ? "ok" : "FAIL", detail: c.detail ? String(c.detail) : "" };
}

/* What the Regenerate button reports once /api/wire has answered. */
function wireSummary(w) {
  if (!w || !w.targets) return "no wire information";
  const changed = w.targets.filter(function (t) { return t.changed; });
  const verb = w.applied ? "changed" : "would change";
  if (!changed.length) return "every client config is already up to date";
  return changed.length + " of " + w.targets.length + " client configs " + verb + ": " +
         changed.map(function (t) { return t.name; }).join(", ");
}

/* Which pending restarts may be started now: the ones /api/state no longer
 * lists as live.
 *
 * The obvious trigger is the `stopped` notice, and it is wrong. app.py
 * publishes that notice BEFORE it re-reads systemd, so the liveness snapshot
 * `_precheck` reads is still the one that has the unit running -- the start
 * fired on that notice comes back 409 `already_live` on a localhost round
 * trip. The state document is published after the refresh, so it is the first
 * thing that agrees with what the precheck will see.
 */
function restartsReadyToStart(state, pending) {
  const models = state && state.models ? state.models : [];
  return models
    .filter(function (m) { return !!pending[m.key] && !m.live; })
    .map(function (m) { return m.key; });
}

function wireDiffText(w) {
  if (!w || !w.targets) return "";
  const parts = [];
  for (let i = 0; i < w.targets.length; i++) {
    const t = w.targets[i];
    if (!t.changed) continue;
    parts.push(t.diff ? t.diff : "--- " + t.name + " (" + t.path + "): changed");
  }
  return parts.length ? parts.join("\n") : "no client config would change";
}

/* ===== PURE HELPERS END ===== */


// --------------------------------------------------------------------------
// Page state. Everything below touches the DOM or the network.
// --------------------------------------------------------------------------

let STATE = null;      // the last /api/state document
let REG = [];          // /api/models rows, for the main-slot select only
let WIRE = null;       // the last /api/wire payload
let NOTICES = [];      // newest first, capped at 50
let BOOT = null;       // { key } while a boot is in flight
let LOG_TIMER = null;  // journal poll while BOOT is set
let LOG_FAILED = false; // the last tracked boot ended in boot_failed/exception
let ES = null;
let BACKOFF = 1000;
const RESTARTING = {}; // key -> timer id, while waiting for its unit to leave /api/state
let SUPPRESS_PERSIST = null; // a block name while app.js itself is setting .open

const MAX_NOTICES = 50;
const JOURNAL_LINES = 20;
const LOG_POLL_MS = 3000;
const BACKOFF_MAX_MS = 15000;
const RESTART_TIMEOUT_MS = 120000;

function $(id) { return document.getElementById(id); }

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function button(label, cls, onclick) {
  const b = el("button", cls, label);
  b.type = "button";
  b.addEventListener("click", onclick);
  return b;
}

function busy() { return !!(STATE && STATE.busy); }

function note(level, reason, message) {
  pushNotice({ level: level, reason: reason, message: message });
}

function pushNotice(n) {
  NOTICES.unshift(n);
  if (NOTICES.length > MAX_NOTICES) NOTICES = NOTICES.slice(0, MAX_NOTICES);
  paintEvents();
}


// --------------------------------------------------------------------------
// Network
// --------------------------------------------------------------------------

/* Every request goes through here so that no caller can silently drop a
 * refusal. A 202 returns its body; a 404/409 turns `error.message` -- the
 * sentence the backend wrote about this exact refusal -- into a notice and
 * returns null. The generic branch only fires when there is no error body at
 * all, and it still names the method, the path and the status. */
async function req(method, path) {
  let res;
  try {
    res = await fetch(path, { method: method, headers: { "Accept": "application/json" } });
  } catch (e) {
    note("error", "network", method + " " + path + " did not reach servedeck: " + e);
    return null;
  }
  let body = null;
  try { body = await res.json(); } catch (e) { body = null; }
  if (res.ok) return body === null ? {} : body;
  const err = body && body.error;
  if (err && err.message) note("error", err.reason || "refused", err.message);
  else note("error", "http_" + res.status, method + " " + path + " returned HTTP " + res.status);
  return null;
}

async function startModel(key) {
  const r = await req("POST", "/api/models/" + encodeURIComponent(key) + "/start");
  if (r) setBoot(key);
}

async function stopModel(key) {
  await req("POST", "/api/models/" + encodeURIComponent(key) + "/stop");
}

async function switchMain(key) {
  const r = await req("POST", "/api/switch/" + encodeURIComponent(key));
  if (r) setBoot(key);
}

/* Restart = stop, then start once /api/state stops listing the unit as live.
 *
 * There is no /restart route, and inventing one in the page would mean firing
 * the start while the old unit still holds the GPU, which the backend refuses
 * with `already_live`. So the two POSTs are sequenced, and the notice says
 * exactly how far it has got: after this call the model is STOPPED, and it is
 * started again only when the state document says the unit is gone.
 */
async function restartModel(key) {
  if (RESTARTING[key]) {
    note("info", "restart", "restart " + key + ": already waiting for it to stop");
    return;
  }
  const r = await req("POST", "/api/models/" + encodeURIComponent(key) + "/stop");
  if (!r) return;
  RESTARTING[key] = setTimeout(function () {
    delete RESTARTING[key];
    note("error", "restart_timeout",
      "restart " + key + ": it never left /api/state; it is STOPPED and the start " +
      "was NOT sent");
  }, RESTART_TIMEOUT_MS);
  note("info", "restart", "restart " + key + ": stop accepted; the start is sent once " +
    "/api/state no longer lists the unit");
}

/* Fire the second half of every restart whose unit has gone. */
function resumeRestarts() {
  restartsReadyToStart(STATE, RESTARTING).forEach(function (key) {
    clearTimeout(RESTARTING[key]);
    delete RESTARTING[key];
    note("info", "restart", "restart " + key + ": the unit is gone; starting it again");
    startModel(key);
  });
}

async function showLog(key) {
  const r = await req("GET", "/api/log/" + encodeURIComponent(key) + "?lines=200");
  if (!r) return;
  $("logOut").textContent = (r.unit || key) + "\n" + (r.lines || []).join("\n");
}

async function loadModels() {
  const r = await req("GET", "/api/models");
  if (!r) return;
  REG = r.models || [];
  paintRegistry();
  paintMain();
}

async function loadDoctor() {
  const r = await req("GET", "/api/doctor");
  if (!r) return;
  paintDoctor(r);
}

async function loadState() {
  const r = await req("GET", "/api/state");
  if (r) onState(r);
}

async function wireDiff() {
  const r = await req("GET", "/api/wire");
  if (!r) return;
  WIRE = r;
  $("wireOut").textContent = wireDiffText(r);
  $("wireApply").hidden = !r.targets.some(function (t) { return t.changed; });
  note("info", "wire", wireSummary(r));
}

async function wireApply() {
  const r = await req("POST", "/api/wire/apply");
  if (!r) return;
  WIRE = r;
  $("wireOut").textContent = wireDiffText(r);
  $("wireApply").hidden = true;
  note("info", "wired", wireSummary(r));
  loadDoctor();
}


// --------------------------------------------------------------------------
// Boot tracking: the progress line and the journal tail
// --------------------------------------------------------------------------

function setBoot(key) {
  BOOT = key ? { key: key } : null;
  if (key) LOG_FAILED = false; // a fresh attempt supersedes the last failure
  if (LOG_TIMER) { clearInterval(LOG_TIMER); LOG_TIMER = null; }
  if (BOOT) {
    $("mainLog").textContent = "";
    pollJournal();
    LOG_TIMER = setInterval(pollJournal, LOG_POLL_MS);
  }
  applyBlockOpen();
}

async function pollJournal() {
  if (!BOOT) return;
  const key = BOOT.key;
  const r = await req("GET",
    "/api/log/" + encodeURIComponent(key) + "?lines=" + JOURNAL_LINES);
  if (!r || !BOOT || BOOT.key !== key) return;
  $("mainLog").textContent = (r.lines || []).join("\n");
}


// --------------------------------------------------------------------------
// Painters. Each one calls the pure helpers above and does no formatting of
// its own, so what the tests assert on is what the browser draws.
// --------------------------------------------------------------------------

function paintAll() {
  paintHeader();
  paintLive();
  paintMain();
  paintResidents();
  paintHeadroom();
  applyBlockOpen();
}

function paintHeader() {
  $("busyTxt").textContent = busy() ? "busy: " + STATE.busy : "";
}

function paintLive() {
  const body = $("liveBody");
  clear(body);
  const models = STATE && STATE.models ? STATE.models : [];
  const rows = models.filter(function (m) { return m.live; });
  rows.forEach(function (m) {
    const r = liveRow(m);
    const tr = el("tr");
    const name = el("td");
    name.appendChild(el("div", "strong", r.name));
    if (r.aliases) name.appendChild(el("div", "sub", r.aliases));
    if (r.error) name.appendChild(el("div", "sub bad", r.error));
    tr.appendChild(name);
    ["slot", "unit", "ctx", "kv", "queue", "rate", "uptime", "restarts"].forEach(function (f) {
      tr.appendChild(el("td", f === "restarts" || f === "kv" ? "num" : "", r[f]));
    });
    const act = el("td", "act");
    act.appendChild(button("Stop", "", function () { stopModel(m.key); }));
    act.appendChild(button("Restart", "", function () { restartModel(m.key); }));
    act.appendChild(button("Log", "", function () { showLog(m.key); }));
    if (busy()) {
      act.querySelectorAll("button").forEach(function (b) { b.disabled = true; });
    }
    tr.appendChild(act);
    body.appendChild(tr);
  });
  const unknown = STATE && STATE.unknown_units ? STATE.unknown_units : [];
  const bits = [];
  if (!rows.length) bits.push("nothing is running");
  if (unknown.length) {
    bits.push("units with no registry entry: " + unknown.map(function (u) {
      return u.unit + " (" + u.unit_state + ")";
    }).join(", "));
  }
  $("liveNote").textContent = bits.join(" · ");
  // The primary panel already covers a single live unit; the table earns its
  // place only once there is more than one thing to compare.
  $("liveWrap").hidden = (rows.length + unknown.length) <= 1;
}

/* The select's options change only when /api/models is re-read, so they are
 * built there and not on every state frame -- rebuilding a <select> under the
 * user's cursor loses the selection they were about to act on. */
function paintRegistry() {
  const sel = $("mainSel");
  const want = sel.value;
  clear(sel);
  REG.filter(function (r) { return r.slot === "main"; }).forEach(function (r) {
    const o = el("option", "", modelOption(r));
    o.value = r.key;
    sel.appendChild(o);
  });
  if (want) sel.value = want;
}

function mainHolder() {
  const models = STATE && STATE.models ? STATE.models : [];
  const held = models.filter(function (m) { return m.slot === "main" && m.live; });
  return held.length ? held[0] : null;
}

function paintMain() {
  const sel = $("mainSel");
  const chosen = sel.value;
  const holder = mainHolder();

  const startBtn = $("mainBtn");
  startBtn.textContent = holder ? "Switch" : "Start";
  startBtn.disabled = !chosen || busy() || !!(holder && holder.key === chosen);
  $("mainStopBtn").disabled = !holder || busy();
  $("mainRestartBtn").disabled = !holder || busy();

  $("mainName").textContent = holder ? holder.id : "";
  $("mainState").textContent = mainStateText(holder, !!BOOT, LOG_FAILED);

  let text;
  if (!holder) {
    text = "nothing holds the main slot";
  } else {
    text = holder.id + " holds the main slot (" + holder.unit_state + ")";
    if (holder.key === chosen) text += " — it is already the selection";
  }
  $("mainNote").textContent = text;

  // Same columns the Live table shows, computed the same way -- no second
  // formula, just the same pure helper pointed at the holder's own row.
  const row = holder ? liveRow(holder) : null;
  $("mainCtx").textContent = row ? row.ctx : EM;
  $("mainKv").textContent = row ? row.kv : EM;
  $("mainQueue").textContent = row ? row.queue : EM;
  $("mainRate").textContent = row ? row.rate : EM;
  $("mainUptime").textContent = row ? row.uptime : EM;
  $("mainRestarts").textContent = row ? row.restarts : EM;
}

function paintResidents() {
  const box = $("resList");
  clear(box);
  const models = STATE && STATE.models ? STATE.models : [];
  const rows = models.filter(function (m) { return m.slot === "resident"; });
  if (!rows.length) {
    box.appendChild(el("p", "note", "no resident models in the registry"));
    return;
  }
  rows.forEach(function (m) {
    const card = el("div", "card");
    card.appendChild(el("div", "strong", m.id));
    card.appendChild(el("div", "sub", m.unit_state + (m.ready ? " · ready" : "")));
    card.appendChild(el("div", "sub", "up " + fmtUptime(m.uptime_s) +
      " · restarts " + fmtNum(m.restarts)));
    const b = m.live
      ? button("Stop", "", function () { stopModel(m.key); })
      : button("Start", "", function () { startModel(m.key); });
    b.disabled = busy();
    card.appendChild(b);
    box.appendChild(card);
  });
}

function paintHeadroom() {
  const list = $("hrList");
  clear(list);
  headroomLines(STATE ? STATE.headroom : null).forEach(function (line) {
    list.appendChild(el("li", "", line));
  });
}

function paintDoctor(doc) {
  const dot = $("doctorDot");
  dot.className = "dot " + (doc.ok ? "good" : "bad");
  const checks = doc.checks || [];
  const failed = checks.filter(function (c) { return !c.ok; }).length;
  $("doctorTxt").textContent = doc.ok
    ? "doctor: " + checks.length + " checks ok"
    : "doctor: " + failed + " of " + checks.length + " failing";
  const body = $("docTable");
  clear(body);
  checks.forEach(function (c) {
    const cells = doctorCells(c);
    const tr = el("tr", c.ok ? "" : "bad");
    tr.appendChild(el("td", "", cells.name));
    tr.appendChild(el("td", "", cells.ok));
    tr.appendChild(el("td", "", cells.detail));
    body.appendChild(tr);
  });
}

function paintEvents() {
  const list = $("evList");
  clear(list);
  NOTICES.forEach(function (n) {
    list.appendChild(el("li", noticeClass(n.level), noticeText(n)));
  });
}

function setConn(cls, text) {
  $("connDot").className = "dot " + cls;
  $("connTxt").textContent = text;
}


// --------------------------------------------------------------------------
// Collapsed blocks: open/closed persists per block in localStorage, except
// while an auto-open condition holds -- that always wins, and releasing it
// falls back to whatever the user last chose (closed, by default).
// --------------------------------------------------------------------------

function detailsEl(name) {
  switch (name) {
    case "more": return $("moreDetails");
    case "log": return $("logDetails");
    case "headroom": return $("headroomDetails");
    case "wiring": return $("wiringDetails");
    case "events": return $("eventsDetails");
    default: return null;
  }
}

function blockPrefKey(name) { return "servedeck.block." + name; }

function loadBlockPref(name) {
  try {
    return localStorage.getItem(blockPrefKey(name)) === "1";
  } catch (e) {
    return false;
  }
}

function saveBlockPref(name, open) {
  try {
    localStorage.setItem(blockPrefKey(name), open ? "1" : "0");
  } catch (e) { /* private window, blocked storage -- the block still works */ }
}

/* Set a block's open state without that change itself being recorded as the
 * user's preference -- otherwise an auto-open would overwrite what the user
 * actually asked for the moment the auto-open condition lets go. */
function setBlockOpen(name, open) {
  const node = detailsEl(name);
  if (!node || node.open === open) return;
  SUPPRESS_PERSIST = name;
  node.open = open;
  SUPPRESS_PERSIST = null;
}

function applyBlockOpen() {
  const models = STATE && STATE.models ? STATE.models : [];
  const residentRunning = models.some(function (m) { return m.slot === "resident" && !!m.live; });
  setBlockOpen("more", residentRunning || loadBlockPref("more"));
  setBlockOpen("log", !!BOOT || LOG_FAILED || loadBlockPref("log"));
  setBlockOpen("headroom", loadBlockPref("headroom"));
  setBlockOpen("wiring", loadBlockPref("wiring"));
  setBlockOpen("events", loadBlockPref("events"));
}

function initBlocks() {
  ["more", "log", "headroom", "wiring", "events"].forEach(function (name) {
    const node = detailsEl(name);
    if (!node) return;
    node.addEventListener("toggle", function () {
      if (SUPPRESS_PERSIST === name) return;
      saveBlockPref(name, node.open);
    });
  });
  applyBlockOpen();
}


// --------------------------------------------------------------------------
// SSE
// --------------------------------------------------------------------------

function onState(doc) {
  STATE = doc;
  paintAll();
  resumeRestarts();
}

function onProgress(p) {
  if (p.key && (!BOOT || BOOT.key !== p.key)) {
    // A reconcile at startup publishes progress under the key "reconcile";
    // adopting it as the boot key is how the journal tail follows a boot the
    // page did not itself request.
    setBoot(p.key);
  }
  $("mainProg").textContent = (p.key ? p.key + ": " : "") + progressText(p);
}

const BOOT_DONE = { ready: 1, boot_failed: 1, exception: 1 };

function onNotice(n) {
  pushNotice(n);
  if (BOOT && n.key === BOOT.key && BOOT_DONE[n.reason]) {
    LOG_FAILED = n.reason !== "ready";
    if (n.journal && n.journal.length) $("mainLog").textContent = n.journal.join("\n");
    setBoot(null);
  }
}

/* EventSource reconnects on its own, but only on its own schedule and only
 * while the server answers at all. The explicit backoff below is for the case
 * that matters here -- servedeck restarting -- where a fixed retry would hammer
 * a port that is not listening yet.
 *
 * Keepalives arrive as SSE comment frames (`: keepalive`). EventSource drops
 * comments before any listener sees them, so there is deliberately no handler
 * for them: they must change nothing on screen, not even a timestamp. */
function connect() {
  if (ES) { ES.close(); ES = null; }
  setConn("warn", "connecting");
  const es = new EventSource("/api/events");
  ES = es;
  const bind = function (type, fn) {
    es.addEventListener(type, function (e) {
      let data;
      try { data = JSON.parse(e.data); } catch (err) {
        note("error", "bad_frame", "unparseable " + type + " frame from /api/events");
        return;
      }
      BACKOFF = 1000;
      fn(data);
    });
  };
  bind("state", onState);
  bind("progress", onProgress);
  bind("notice", onNotice);
  es.onopen = function () { BACKOFF = 1000; setConn("good", "live"); };
  es.onerror = function () {
    setConn("bad", "reconnecting");
    es.close();
    if (ES === es) ES = null;
    setTimeout(connect, BACKOFF);
    BACKOFF = Math.min(BACKOFF * 2, BACKOFF_MAX_MS);
  };
}


// --------------------------------------------------------------------------
// Init
// --------------------------------------------------------------------------

function init() {
  // The URL a client configures, derived from where this page was served.
  $("gwChip").textContent = `${location.origin}/v1`;

  $("doctorBtn").addEventListener("click", function () {
    const node = $("wiringDetails");
    if (!node) return;
    node.open = !node.open;
    if (node.open) {
      loadDoctor();
      node.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  });
  $("mainSel").addEventListener("change", paintMain);
  $("mainBtn").addEventListener("click", function () {
    const key = $("mainSel").value;
    if (!key) return;
    if (mainHolder()) switchMain(key); else startModel(key);
  });
  $("mainStopBtn").addEventListener("click", function () {
    const holder = mainHolder();
    if (holder) stopModel(holder.key);
  });
  $("mainRestartBtn").addEventListener("click", function () {
    const holder = mainHolder();
    if (holder) restartModel(holder.key);
  });
  $("wireBtn").addEventListener("click", wireDiff);
  $("wireApply").addEventListener("click", wireApply);

  initBlocks();
  paintAll();
  loadState();
  loadModels();
  loadDoctor();
  connect();
}

init();
