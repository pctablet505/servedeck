"""Regression tests for the comprehensive repository audit (BUG-01 .. BUG-28).

Every test in this file reproduces a concrete defect found during the line-by-line
audit and fails before its corresponding fix. None boots an LLM or touches real
GPU/systemd/user-home state.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import shutil
import subprocess
import tomllib
import typing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from servedeck import (
    __main__ as _entry,
    app as _app,
    batch as _batch,
    bootfacts as _bootfacts,
    capacity as _capacity,
    cli as _cli,
    control as _control,
    desired as _desired,
    discovery as _discovery,
    doctor as _doctor,
    glm_policies as _glm,
    gpu as _gpu,
    legacy_page as _lp,
    limits as _limits,
    models as _models,
    parallelism as _parallelism,
    policies as _policies,
    settings as _settings,
    units as _units,
    wire as _wire,
    xid_watch as _xid_watch,
)

from tests.test_app import TOML as APP_TOML
from tests.test_app import FakeControl as AppFakeControl
from tests.test_control import BIG27, FLASH, LFM2, FakeRegistry, FakeSystemd, make_control

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "servedeck" / "web"


def _node_bin() -> str | None:
    found = shutil.which("node")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "node"
    return str(local) if local.is_file() and os.access(local, os.X_OK) else None


NODE = _node_bin()


BASE_TOML = """
[gpu]
total_mib = 100000
margin_mib = 1024

[builds.stock]
venv = "/opt/stock"
cuda_home = "/opt/stock/cu13"

[models.big]
id = "Big-Model"
aliases = ["big"]
repo = "org/big"
slot = "main"
port = 19001
build = "stock"
ctx = 32768
util = 0.96
flags = ["--max-num-seqs", "16", "--kv-offloading-size", "40"]

[models.other]
id = "Other-Model"
aliases = ["other"]
repo = "org/other"
slot = "main"
port = 19002
build = "stock"
ctx = 16384
util = 0.40

[models.small]
id = "LFM2.5-350M"
repo = "org/small"
slot = "resident"
vram_mib = 4895
port = 19003
build = "stock"
ctx = 4096
"""


# ---------------------------------------------------------------------------
# BUG-01: __main__.py --glm-capture UnboundLocalError & log level validation
# ---------------------------------------------------------------------------


def test_bug01_main_glm_capture_does_not_raise_unbound_local_error(tmp_path, monkeypatch):
    toml_path = tmp_path / "models.toml"
    toml_path.write_text(BASE_TOML)
    monkeypatch.setenv("SERVEDECK_MODELS", str(toml_path))
    monkeypatch.setenv("SERVEDECK_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("SERVEDECK_UNIT_PREFIX", "sd-test-")
    monkeypatch.delenv(_glm.CAPTURE_ENV, raising=False)
    monkeypatch.setattr(_app, "create_app", lambda *a, **kw: type("A", (), {"state": type("S", (), {"rt": type("R", (), {"hub": None})()})()})())
    monkeypatch.setattr(_entry.uvicorn, "Config", lambda *a, **kw: object())

    class _NoopServer:
        def __init__(self, config, hub):
            self.ran = False

        def run(self):
            self.ran = True

    monkeypatch.setattr(_entry, "_ShutdownClosesTheHub", _NoopServer)
    rc = _entry.main(["--glm-capture", "--no-reconcile", "--port", "18099"])
    assert rc == 0
    assert os.environ.get(_glm.CAPTURE_ENV) == "1"


def test_bug01_configure_logging_handles_non_int_logging_attribute(monkeypatch):
    monkeypatch.setenv("SERVEDECK_LOG_LEVEL", "basic_format")
    _entry._configure_logging()
    assert logging.getLogger("servedeck").level == logging.INFO


# ---------------------------------------------------------------------------
# BUG-02: app.py startup race between _poll_loop recovery and reconcile
# ---------------------------------------------------------------------------


def test_bug02_recovery_waits_while_initial_reconcile_is_pending(tmp_path, monkeypatch):
    toml_path = tmp_path / "models.toml"
    toml_path.write_text(BASE_TOML)
    reg = _models.load(toml_path)
    s = _settings.Settings(
        listen_host="127.0.0.1",
        listen_port=18091,
        models_path=toml_path,
        state_dir=tmp_path / "state",
        unit_prefix="sd-test-",
    )
    calls: list[str] = []

    class _Ctl:
        def live(self):
            return []

        def start(self, key, **_kw):
            calls.append(f"start:{key}")
            return _control.StartResult(key=key, unit=f"sd-test-{key}", ready=True, elapsed_s=0.1)

        def reconcile(self, desired=None, **_kw):
            calls.append("reconcile")
            assert rt.busy is not None, "reconcile must claim busy so mutations/recovery cannot race it"
            return _control.Reconciliation(started=[])

    s.state_dir.mkdir(parents=True, exist_ok=True)
    _desired.save(_desired.Desired(main="big"), s.desired_path)
    rt = _app.build_runtime(
        s,
        registry=reg,
        control=_Ctl(),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _r: httpx.Response(200, json={"ok": True, "instance": rt.instance_id})
            )
        ),
    )
    rt.reconcile_pending = True
    _app._recover_desired(rt)
    assert calls == [], "poll recovery must not fire before initial reconcile finishes"

    async def _run():
        rt.hub.bind(asyncio.get_running_loop())
        await _app._reconcile_after_bind(rt, timeout_s=2.0, interval_s=0.001)

    asyncio.run(_run())
    assert rt.reconcile_pending is False
    assert calls == ["reconcile"]


# ---------------------------------------------------------------------------
# BUG-03 & BUG-04: control.py util override validation & clearing stale launch
# ---------------------------------------------------------------------------


class _Spec:
    def __init__(self, key="big", slot="main", port=19001, util=0.96, flags=("--max-num-seqs", "16")):
        self.key = key
        self.id = key.upper()
        self.repo = f"org/{key}"
        self.slot = slot
        self.port = port
        self.vram_mib = None
        self.util = util
        self.ctx_tokens = 32768
        self.host_ram_gib = None
        self.venv_bin = "/opt/stock/bin"
        self.cwd = "."
        self.flags = flags

    def served_names(self):
        return [self.id]

    def render_argv(self, util: float, port: int) -> list[str]:
        return ["/bin/vllm", "serve", "org/m", "--gpu-memory-utilization", f"{util:.2f}", "--port", str(port), *self.flags]

    def render_env(self) -> dict[str, str]:
        return {}


class _SimpleReg:
    def __init__(self, *specs: _Spec):
        self._by_key = {s.key: s for s in specs}

    def get(self, key: str):
        return self._by_key[key]

    def keys(self):
        return list(self._by_key)


def _fake_ready_ctl(tmp_path: Path, spec: _Spec, *, free: int) -> tuple[_control.Control, list[list[str]]]:
    launched: list[list[str]] = []

    def runner(argv):
        cmd = list(argv)
        if cmd[0] == "systemd-run":
            launched.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if "show-environment" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "HOME=/tmp\n", "")
        if "list-units" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if "show" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, "LoadState=loaded\nActiveState=active\nSubState=running\nMainPID=1234\nResult=success\nNRestarts=0\n", ""
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    ctl = _control.Control(
        _SimpleReg(spec),
        margin_mib=1024,
        total_mib=100_000,
        unit_prefix="sd-test-",
        desired_path=tmp_path / "desired.json",
        run=runner,
        probe=lambda port: [spec.id],
        free_mib=lambda: free,
        listener_pid=lambda _p: None,
        reap_offload=lambda **_k: [],
        mem_available_gib_fn=lambda: 200.0,
    )
    return ctl, launched


def test_bug03_explicit_lower_util_succeeds_and_excess_or_bad_util_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(_control.Control, "wait_ready", lambda self, key, *a, **kw: _control.StartResult(key=key, unit=f"sd-test-{key}", ready=True, elapsed_s=0.1))

    # Pinned at 0.96, 85_000 MiB free -> 0.96 does not fit, but explicit 0.80 DOES fit.
    ctl_low, launched = _fake_ready_ctl(tmp_path, _Spec("big", util=0.96), free=85_000)
    res_ok = ctl_low.start("big", util=0.80)
    assert isinstance(res_ok, _control.StartResult) and res_ok.ready is True
    assert "0.80" in launched[-1]

    # Pinned at 0.40, 50_000 MiB free -> 0.40 fits, but explicit 0.90 does NOT fit!
    (tmp_path / "sub").mkdir()
    ctl_high, _ = _fake_ready_ctl(tmp_path / "sub", _Spec("other", util=0.40), free=50_000)
    res_high = ctl_high.start("other", util=0.90)
    assert isinstance(res_high, _control.Refusal) and res_high.reason == "not_enough_vram"

    for bad in (0.0, -0.2, 1.05, float("nan")):
        res_bad = ctl_high.start("other", util=bad)
        assert isinstance(res_bad, _control.Refusal)


def test_bug04_default_start_clears_stale_launch_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(_control.Control, "wait_ready", lambda self, key, *a, **kw: _control.StartResult(key=key, unit=f"sd-test-{key}", ready=True, elapsed_s=0.1))

    ctl, _ = _fake_ready_ctl(tmp_path, _Spec("big", util=0.90), free=98_000)
    prior = _desired.Desired(main="big").with_launch(
        "big", _desired.Launch(util=0.75, argv={"--max-num-seqs": "1"})
    )
    _desired.save(prior, tmp_path / "desired.json")

    # Plain start(argv_overrides=None) preserves stored overrides, whereas an
    # explicit allocator launch with empty overrides (overrides_from({}) -> (None, {}))
    # must clear the stale overrides instead of skipping `with_launch` on truthiness!
    res_plain = ctl.start("big")
    assert isinstance(res_plain, _control.StartResult) and res_plain.ready is True
    assert "big" in _desired.load(tmp_path / "desired.json").launch

    res_reset = ctl.start("big", argv_overrides={})
    assert isinstance(res_reset, _control.StartResult) and res_reset.ready is True
    after = _desired.load(tmp_path / "desired.json")
    assert "big" not in after.launch, f"stale overrides survived an explicit reset launch: {after.launch}"


# ---------------------------------------------------------------------------
# BUG-05, BUG-06, BUG-07, BUG-08: offload_scan, abort, failed units, argv
# ---------------------------------------------------------------------------


def test_bug05_offload_scan_fails_closed_when_proc_missing_or_witness_unverified(tmp_path):
    missing = _control.offload_scan(proc=tmp_path / "does-not-exist")
    assert missing.trusted is False

    empty_proc = tmp_path / "empty_proc"
    empty_proc.mkdir()
    # Our own PID is alive, yet absent from empty_proc -> cannot trust scan!
    unverified = _control.offload_scan(witness_pids=[os.getpid()], proc=empty_proc)
    assert unverified.trusted is False


def test_bug06_abort_none_when_idle_does_not_poison_next_boot(tmp_path):
    import io

    spec = _Spec("big", util=0.50)

    def runner(argv):
        cmd = list(argv)
        if "show" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, "LoadState=loaded\nActiveState=active\nSubState=running\nMainPID=1234\nResult=success\nNRestarts=0\n", ""
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    class _DummyProc:
        stdout = None
        def poll(self):
            return None
        def terminate(self):
            pass
        def kill(self):
            pass
        def wait(self, timeout=None):
            return 0

    ctl = _control.Control(
        _SimpleReg(spec),
        margin_mib=1024,
        total_mib=100_000,
        unit_prefix="sd-test-",
        desired_path=tmp_path / "desired.json",
        run=runner,
        spawn=lambda _argv: _DummyProc(),
        probe=lambda _p: [spec.id],
        free_mib=lambda: 90_000,
        sleep=lambda _s: None,
    )
    assert ctl.abort(None) is False
    # Next wait_ready must NOT immediately abort with "cancelled by operator"!
    res = ctl.wait_ready("big", timeout_s=2.0, since="2 seconds ago")
    assert res.ready is True, f"idle abort(None) poisoned next boot: {res}"


def test_bug07_failed_unit_is_not_treated_as_live_or_adoptable(tmp_path):
    spec = _Spec("big", util=0.50)

    def runner(argv):
        cmd = list(argv)
        if "list-units" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, "sd-test-big.service loaded failed failed Model big\n", ""
            )
        if "show" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, "LoadState=loaded\nActiveState=failed\nSubState=failed\nMainPID=0\nResult=exit-code\nNRestarts=3\n", ""
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    ctl = _control.Control(
        _SimpleReg(spec),
        margin_mib=1024,
        total_mib=100_000,
        unit_prefix="sd-test-",
        desired_path=tmp_path / "desired.json",
        run=runner,
        probe=lambda _p: None,
        free_mib=lambda: 90_000,
    )
    assert ctl.live_main() is None
    adopted = ctl.adopt()
    assert adopted.adopted == []


def test_bug08_apply_argv_overrides_handles_equals_form_and_units_cpu_weight(tmp_path, caplog):
    argv = ["/bin/vllm", "serve", "m", "--max-num-seqs=16", "--kv-offloading-size=40", "--keep"]
    updated = _control.apply_argv_overrides(
        argv, {"--max-num-seqs": "4", "--kv-offloading-size": None}
    )
    assert updated == ["/bin/vllm", "serve", "m", "--max-num-seqs", "4", "--keep"]

    captured: list[list[str]] = []
    _units.start_transient(
        "sd-test-cpuw",
        ["/bin/true"],
        {},
        str(tmp_path),
        cpu_weight=250,
        run=lambda cmd: (captured.append(list(cmd)), subprocess.CompletedProcess(list(cmd), 0, "", ""))[1],
    )
    assert any("CPUWeight=250" in tok for tok in captured[0])

    v1 = tmp_path / "desired.json"
    v1.write_text(json.dumps({"version": 1, "desired_state": "RUNNING", "backend": "big"}))
    with caplog.at_level(logging.WARNING):
        _desired.load(v1)
    assert "version 3" in caplog.text


# ---------------------------------------------------------------------------
# BUG-09 .. BUG-15: Web UI (app.js, index.html, style.css) + legacy_page.py
# ---------------------------------------------------------------------------


@pytest.mark.skipif(NODE is None, reason="needs node runtime to execute app.js DOM harnesses")
def test_bug09_bug10_bug11_bug14_app_js_behaviors():
    script = r"""
const fs = require("fs");
const vm = require("vm");
const src = fs.readFileSync(process.argv[1], "utf8");

function makeEl(id = "") {
  const el = {
    id,
    tagName: "DIV",
    textContent: "",
    _innerHTML: "",
    className: "",
    title: "",
    value: "",
    disabled: false,
    open: false,
    style: {},
    dataset: {},
    attrs: {},
    children: [],
    listeners: {},
    classList: {
      _set: new Set(),
      add(...xs) { xs.forEach(x => this._set.add(x)); },
      remove(...xs) { xs.forEach(x => this._set.delete(x)); },
      toggle(x, f) { if (f === undefined ? !this._set.has(x) : f) this._set.add(x); else this._set.delete(x); },
      contains(x) { return this._set.has(x); },
    },
    setAttribute(k, v) { this.attrs[k] = String(v); },
    getAttribute(k) { return this.attrs[k] ?? null; },
    appendChild(c) { this.children.push(c); return c; },
    replaceChildren(...cs) { this.children = [...cs]; this._innerHTML = ""; },
    addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); },
    querySelector() { return makeEl(); },
    querySelectorAll() { return []; },
    getContext() {
      return { setTransform(){}, clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, closePath(){}, stroke(){}, fill(){}, arc(){}, fillRect(){}, fillText(){} };
    },
    getBoundingClientRect() { return { width: 200, height: 40 }; },
    showModal() { if (this.open) throw new Error("InvalidStateError: already open"); this.open = true; },
    close() { this.open = false; },
    remove() { delete els[this.id]; },
    insertBefore(c) { if (c && c.id) els[c.id] = c; this.children.unshift(c); return c; },
    click() { if (typeof this.onclick === "function") this.onclick(); },
  };
  Object.defineProperty(el, "innerHTML", {
    get() {
      if (this.children.length && !this._innerHTML) {
        return this.children.map(c => c.textContent || c._innerHTML || "").join("");
      }
      return this._innerHTML;
    },
    set(v) { this._innerHTML = String(v); this.children = []; },
  });
  return el;
}

const els = {};
function getEl(id) {
  if (id === "adoptBtn") return els[id] || null;
  if (!els[id]) els[id] = makeEl(id);
  return els[id];
}

const posted = [];
const sseHandlers = {};

const sandbox = {
  console,
  Math, Number, String, Boolean, Array, Object, Set, Map, JSON, RegExp, Date, Promise, Error,
  parseInt, parseFloat, isNaN, isFinite, encodeURIComponent,
  setTimeout: () => 1,
  clearTimeout: () => {},
  setInterval: () => 1,
  requestAnimationFrame: () => 1,
  devicePixelRatio: 1,
  localStorage: { _m: {}, getItem(k){ return this._m[k] ?? null; }, setItem(k, v){ this._m[k] = String(v); } },
  location: { reload(){} },
  matchMedia: () => ({ addEventListener(){} }),
  getComputedStyle: () => ({ getPropertyValue(){ return ""; } }),
  window: { addEventListener(){}, devicePixelRatio: 1 },
  document: {
    documentElement: makeEl("html"),
    getElementById: (id) => getEl(id),
    createElement: (tag) => { const e = makeEl(); e.tagName = tag.toUpperCase(); return e; },
    createTextNode: (txt) => ({ textContent: String(txt) }),
    querySelector: () => makeEl(),
    querySelectorAll: () => [],
    addEventListener: () => {},
  },
  EventSource: function(url) {
    this.url = url;
    this.addEventListener = (name, fn) => { sseHandlers[name] = fn; };
  },
  fetch: async (url, opts = {}) => {
    posted.push({ url, method: opts.method || "GET", body: opts.body ? JSON.parse(opts.body) : null });
    const payload = url === "/api/models" ? { models: [], disk: null } : { ok: true, findings: [] };
    return { ok: true, status: 200, json: async () => payload, text: async () => JSON.stringify(payload) };
  },
};
vm.createContext(sandbox);
vm.runInContext(src, sandbox);
vm.runInContext(
  "globalThis.__setTestState = (p) => { if ('MODELS' in p) MODELS = p.MODELS; if ('sel' in p) sel = p.sel; if ('lastEstimate' in p) lastEstimate = p.lastEstimate; if ('ctxSource' in p) ctxSource = p.ctxSource; }; globalThis.__getModels = () => MODELS;",
  sandbox
);

(async () => {
  await new Promise((r) => setImmediate(r));
  // 1. SSE must listen to "progress" events (BUG-15)
  if (typeof sseHandlers.progress !== "function") {
    throw new Error("missing progress SSE listener");
  }

  // 2. Duplicate repo_id + resident model handling (BUG-10, BUG-11)
  sandbox.__setTestState({
    MODELS: [
      { key: "flashnext", name: "Flash-Next", repo_id: "mazinb/FN", backend: "flashnext", slot: "main", serving: true, downloaded: true, servable: true, model_max_ctx: 262144 },
      { key: "flashnext-batch", name: "Flash-Next-Batch", repo_id: "mazinb/FN", backend: "flashnext", slot: "main", serving: false, downloaded: true, servable: true, model_max_ctx: 262144 },
      { key: "lfm2", name: "LFM2.5-350M", repo_id: "LiquidAI/LFM2.5-350M", backend: "stock", slot: "resident", serving: false, downloaded: true, servable: true, model_max_ctx: 32768 },
    ],
    sel: 2,
  });
  sandbox.paintTargetBanner();
  if (getEl("apply").textContent !== "Start LFM2.5-350M") {
    throw new Error("resident stage button wrong: " + getEl("apply").textContent);
  }
  if (getEl("cfgBannerSub").textContent.includes("evict")) {
    throw new Error("resident stage falsely claimed eviction: " + getEl("cfgBannerSub").textContent);
  }

  // Paint state with BOTH flashnext (main) and lfm2 (resident) live, while flashnext-batch (same repo_id!) is NOT live
  sandbox.paintState({
    models: [
      { key: "flashnext", live: true, ready: true },
      { key: "flashnext-batch", live: false, ready: false },
      { key: "lfm2", live: true, ready: true },
    ],
    upstream: { up: true, reachable: true, identity: { repo_id: "mazinb/FN" }, max_model_len: 262144 },
    supervisor: { control_enabled: true, actual_state: "READY", phase: null },
    boot: { active: false, phase: "ready", elapsed_s: 12 },
    gpu_v1: { name: "RTX", total_mib: 97887, used_mib: 90000, free_mib: 7887 },
    vllm: {},
    sizing: {},
    config: {},
  });
  const curModels = sandbox.__getModels();
  if (!curModels[0].serving || curModels[1].serving || !curModels[2].serving) {
    throw new Error("serving flags wrong across shared repo_id & resident: " + JSON.stringify(curModels.map(m => [m.key, m.serving])));
  }

  // 3. Stop button after first boot completed (boot.phase === "ready") MUST call /api/server/stop, NOT /api/server/abort! (BUG-09)
  sandbox.__setTestState({ sel: 0 });
  sandbox.promptConfirm = async () => true;
  posted.length = 0;
  await getEl("stop").onclick();
  const stopCall = posted.find(p => p.method === "POST" && p.url.startsWith("/api/server/"));
  if (!stopCall || stopCall.url !== "/api/server/stop") {
    throw new Error("Stop while READY hit wrong endpoint: " + JSON.stringify(posted));
  }

  // 4. XSS check in showFindings and renderCardTags (BUG-14)
  sandbox.showFindings([
    { severity: "block", code: "B1", title: "<img src=x onerror=alert(1)>", detail: "<script>bad()</script>" },
    { severity: "warn", code: "W1", title: "<b>warn</b>", detail: "<svg onload=x()>" },
  ]);
  const rawHtml = getEl("alertCT")._innerHTML + getEl("alertWT")._innerHTML;
  if (rawHtml.includes("<img") || rawHtml.includes("<script>") || rawHtml.includes("<svg")) {
    throw new Error("unescaped HTML in alert banners: " + rawHtml);
  }
  const tagHtml = sandbox.renderCardTags({ quant: "<img src=x>", servable: true, trust: "measured" }, false);
  if (tagHtml.includes("<img")) {
    throw new Error("unescaped HTML in renderCardTags: " + tagHtml);
  }

  // 5. Non-4096 ceiling preserved by renderCtx + commitCtxNum (BUG-15)
  curModels[1].model_max_ctx = 202752;
  sandbox.__setTestState({ sel: 1, ctxSource: "default", lastEstimate: { ctx_max_model: 202752 } });
  sandbox.renderCtx({ ctx_max_model: 202752 });
  getEl("ctxNum").value = "202752";
  getEl("ctxNum").onchange();
  if (Number(getEl("ctx").value) !== 202752) {
    throw new Error("commitCtxNum failed to preserve non-4096 ceiling 202752: got " + getEl("ctx").value);
  }
  getEl("ctxNum").value = "198656"; // 10240 + 46 * 4096: exact stop on top-anchored grid!
  getEl("ctxNum").onchange();
  if (Number(getEl("ctx").value) !== 198656) {
    throw new Error("commitCtxNum snapped off slider grid for 198656: got " + getEl("ctx").value);
  }

  console.log("ALL_JS_CHECKS_PASSED");
})().catch((e) => {
  console.error(e && e.stack ? e.stack : String(e));
  process.exit(1);
});
"""
    res = subprocess.run([NODE, "-e", script, str(WEB_DIR / "app.js")], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "ALL_JS_CHECKS_PASSED" in res.stdout


def test_bug10_bug12_bug13_legacy_page_shared_repo_cgroup_vram_and_boot_reset(tmp_path, monkeypatch):
    dup_toml = """
[gpu]
total_mib = 97887
margin_mib = 1024

[builds.stock]
venv = "/opt/stock"
cuda_home = "/opt/stock/cu13"

[models.flashnext]
id = "Flash-Next"
repo = "mazinb/Shared"
slot = "main"
port = 19001
build = "stock"
ctx = 262144
util = 0.96
flags = ["--max-num-seqs", "16", "--kv-offloading-size", "40"]

[models.flashnext-batch]
id = "Flash-Next-Batch"
repo = "mazinb/Shared"
slot = "main"
port = 19002
build = "stock"
ctx = 262144
util = 0.96
flags = ["--max-num-seqs", "64"]
"""
    toml_path = tmp_path / "models.toml"
    toml_path.write_text(dup_toml)
    reg = _models.load(toml_path)
    s = _settings.Settings(
        listen_host="127.0.0.1",
        listen_port=18092,
        models_path=toml_path,
        state_dir=tmp_path / "state",
        unit_prefix="sd-test-",
    )
    started_keys: list[str] = []

    @dataclasses.dataclass
    class _Live:
        key: str
        unit: str
        ready: bool = True
        state: str = "active"
        sub_state: str = "running"
        pid: int = 100  # MainPID holds 0 MiB VRAM; worker 200 holds 90_000 MiB!
        restarts: int = 0
        port: int = 19001
        unknown: bool = False

    class _Ctl:
        def __init__(self):
            self.rows: list[_Live] = []

        def live(self):
            return list(self.rows)

        def start(self, key, **_kw):
            started_keys.append(key)
            return _control.StartResult(key=key, unit=f"sd-test-{key}", ready=True, elapsed_s=0.1)

        def switch(self, key, **_kw):
            started_keys.append(key)
            return _control.SwitchResult(
                stopped=None,
                released=True,
                waited_s=0.0,
                free_after_mib=90_000,
                started=_control.StartResult(key=key, unit=f"sd-test-{key}", ready=True, elapsed_s=0.1),
            )

        def stop(self, key, **_kw):
            self.rows = [r for r in self.rows if r.key != key]
            return _control.StopResult(key=key, unit=f"sd-test-{key}", was_live=True, held_mib=90_000)

        def adopt(self, **_kw):
            return _control.Adoption(adopted=[])

        def reconcile(self, *_a, **_kw):
            return _control.Reconciliation()

    import time as _time

    ctl = _Ctl()
    monkeypatch.setattr(_lp.discovery, "discover_models", lambda *a, **k: [])
    monkeypatch.setattr(
        _lp.discovery,
        "resolve_inputs",
        lambda repo_id, util, ctx, **k: _discovery.ResolvedInputs(
            repo_id=repo_id,
            backend="stock",
            requested_util=util,
            requested_ctx=ctx,
            weights_gib=75.0,
            weights_source="measured",
            kv_kib_per_token=30.0,
            kv_source="measured",
            trust="measured",
            overhead_gib=5.0,
            model_max_ctx=262144,
            model_type="qwen2",
            architectures0="Qwen2ForCausalLM",
            matched_ctx=ctx,
            other_ctx_kv_rates={ctx: 30.0},
        ),
    )
    monkeypatch.setattr(
        _lp._gpu,
        "compute_apps",
        lambda: [_gpu.ComputeApp(pid=200, process_name="/opt/stock/bin/python", used_mib=90_000)],
    )
    monkeypatch.setattr(_lp._gpu, "free_mib", lambda: 7_000)
    monkeypatch.setattr(_lp._gpu, "total_mib", lambda: 97_887)
    # The estimate reads the card through gpu_summary()/gpu_alive(). Unpatched,
    # a box with a GPU answers with its real numbers and a box without one
    # skips the VRAM check, so the discount below was proven nowhere.
    monkeypatch.setattr(
        _lp._gpu,
        "gpu_summary",
        lambda: _gpu.GpuSummary(
            index=0, name="test GPU", total_mib=97_887, used_mib=90_887, free_mib=7_000,
            util_percent=0, temperature_c=40, power_draw_w=100.0,
        ),
    )
    monkeypatch.setattr(_lp._gpu, "gpu_alive", lambda: True)
    monkeypatch.setattr(_control, "_cgroup_pids", lambda unit, **k: [100, 200])

    app = _app.create_app(s, registry=reg, control=ctl, reconcile=False, poll=False)
    with TestClient(app) as c:
        # BUG-10: launching flashnext-batch by key when repo_id is shared with flashnext
        r = c.post("/api/server/start", json={"key": "flashnext-batch", "repo_id": "mazinb/Shared", "util": 0.96, "ctx": 262144})
        assert r.status_code == 202, r.text
        for _ in range(100):
            if app.state.rt.busy is None and started_keys:
                break
            _time.sleep(0.01)
        assert started_keys == ["flashnext-batch"]

        # BUG-12: while flashnext is serving (MainPID=100 holds 0 MiB, worker PID=200 holds 90_000 MiB),
        # /api/capacity/estimate must discount worker PID 200's 90_000 MiB!
        ctl.rows = [_Live(key="flashnext", unit="sd-test-flashnext")]
        app.state.rt.routes.refresh()
        est = c.post(
            "/api/capacity/estimate",
            json={"key": "flashnext", "repo_id": "mazinb/Shared", "util": 0.96, "ctx": 262144, "max_num_seqs": 16},
        ).json()
        blocks = [f["code"] for f in est["findings"] if f.get("level") == "block" or f.get("severity") == "block"]
        assert "NOT_ENOUGH_FREE_VRAM" not in blocks, f"failed to discount worker cgroup VRAM: {est['findings']}"

        # BUG-13: malformed JSON field types on /api/server/start return 400, not 500
        bad = c.post("/api/server/start", json={"key": "flashnext", "repo_id": "mazinb/Shared", "ctx": "not-a-number"})
        assert bad.status_code == 400
        neg = c.post("/api/server/start", json={"key": "flashnext", "repo_id": "mazinb/Shared", "ctx": -10})
        assert neg.status_code == 400

        # BUG-13: BootTracker must not stay stuck in booting=True after a Refusal
        rt = app.state.rt
        rt.boot.record("flashnext-batch", "line", None, 1.0)
        assert rt.boot.booting is True
        _app._publish_result(
            rt.hub,
            "switch",
            _control.Refusal(reason="vram_not_released", message="held", key="flashnext-batch"),
            boot=rt.boot,
        )
        assert rt.boot.booting is False


def test_bug15_index_html_and_style_css_fixes():
    index_html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    style_css = (WEB_DIR / "style.css").read_text(encoding="utf-8")
    assert "serve.log" not in index_html, "stale v1 serve.log label still in index.html"
    assert ".gpuchip.err .dot" in style_css or ".gpuchip .dot.err" in style_css


# ---------------------------------------------------------------------------
# BUG-16 & BUG-17: Telemetry, XidWatcher, gpu-log CLI & Doctor integration
# ---------------------------------------------------------------------------


def test_bug16_cli_exposes_gpu_log_and_doctor_checks_fatal_xid(tmp_path, monkeypatch):
    parser = _cli.build_parser()
    args = parser.parse_args(["gpu-log", "--xid", "--tail", "5"])
    assert args.command == "gpu-log" and args.xid is True and args.tail == 5

    # Doctor must surface a fatal Xid crash report newer than boot time as FAIL
    toml_path = tmp_path / "models.toml"
    toml_path.write_text(BASE_TOML)
    hub = tmp_path / "hub"
    for repo in ("org/big", "org/other", "org/small"):
        snap = hub / ("models--" + repo.replace("/", "--")) / "snapshots" / "r1"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
        (snap / "model.safetensors").write_bytes(b"0")
    shm = tmp_path / "shm"
    shm.mkdir()
    state_dir = tmp_path / "state"
    crash_dir = state_dir / "crash_reports"
    crash_dir.mkdir(parents=True)
    (crash_dir / "2026-09-12T10-00-00Z.json").write_text(
        json.dumps({"anchor_ts": "2099-01-01T00:00:00+00:00"})
    )
    monkeypatch.setattr(_xid_watch, "_current_boot_time", lambda: datetime(2026, 9, 12, tzinfo=timezone.utc))

    results = _doctor.run_doctor(
        toml_path,
        client_files={},
        unit_run=lambda argv: subprocess.CompletedProcess(list(argv), 0, "", ""),
        marker_paths=[],
        ptrace_scope=0,
        desired_path=state_dir / "desired.json",
        shm_dir=shm,
        hub_dir=str(hub),
        power_limit_w=None,
    )
    by_name = {r.name: r for r in results}
    assert "gpu fatal xid" in by_name
    assert by_name["gpu fatal xid"].ok is False


def test_bug16_app_lifespan_starts_and_stops_telemetry_and_xid_watcher(tmp_path, monkeypatch):
    toml_path = tmp_path / "models.toml"
    toml_path.write_text(BASE_TOML)
    reg = _models.load(toml_path)
    state_dir = tmp_path / "state"
    s = _settings.Settings(
        listen_host="127.0.0.1",
        listen_port=18093,
        models_path=toml_path,
        state_dir=state_dir,
        unit_prefix="sd-test-",
    )
    events: list[str] = []

    class _FakeSampler:
        def __init__(self, *, state_dir=None, **kw):
            events.append(f"sampler:init:{state_dir}")

        def start(self):
            events.append("sampler:start")

        async def stop(self):
            events.append("sampler:stop")

    class _FakeWatcher:
        def __init__(self, *, state_dir=None, **kw):
            events.append(f"xid:init:{state_dir}")

        def start(self, *a, **kw):
            events.append("xid:start")

        async def stop(self):
            events.append("xid:stop")

    class _Ctl:
        def live(self):
            return []

        def reconcile(self, *a, **kw):
            return _control.Reconciliation()

    from servedeck import telemetry as _telemetry_mod

    monkeypatch.setattr(_telemetry_mod, "Sampler", _FakeSampler)
    monkeypatch.setattr(_xid_watch, "XidWatcher", _FakeWatcher)

    app = _app.create_app(s, registry=reg, control=_Ctl(), reconcile=False, poll=False, telemetry=True)
    with TestClient(app):
        assert f"sampler:init:{state_dir}" in events
        assert "sampler:start" in events
        assert f"xid:init:{state_dir}" in events
        assert "xid:start" in events
    assert "sampler:stop" in events
    assert "xid:stop" in events


def test_bug17_gpu_xid_events_passes_all_boots_and_xid_watch_new_year_rollover():
    seen_argv: list[list[str]] = []
    _gpu.xid_events(
        since="-1h",
        run=lambda argv: (seen_argv.append(list(argv)), subprocess.CompletedProcess(list(argv), 0, "", ""))[1],
    )
    assert "-b" in seen_argv[0] and "all" in seen_argv[0]

    # On Jan 1, 2027 00:05 UTC, a kernel line "Dec 31 23:55:00.000000" must be 2026, NOT 2027!
    line = "Dec 31 23:55:00.123456 host kernel: NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.\n"
    now_jan1 = datetime(2027, 1, 1, 0, 5, 0, tzinfo=timezone.utc)
    parsed = _xid_watch.parse_xid_lines(line, tzinfo=timezone.utc, now=now_jan1)
    assert len(parsed) == 1
    assert parsed[0]["ts"].startswith("2026-12-31"), parsed[0]["ts"]


# ---------------------------------------------------------------------------
# BUG-18 .. BUG-22: bootfacts, discovery, capacity, parallelism, settings, models
# ---------------------------------------------------------------------------


def test_bug18_bootfacts_marks_measured_trust_and_discovery_falls_back_on_partial_obs(tmp_path):
    facts = _bootfacts.BootFacts(
        device_total_gib=94.97,
        util=0.96,
        kv_gib=12.0,
        kv_tokens=262_144,
    )
    obs = _bootfacts.observation(
        repo_id="org/m",
        backend="stock",
        facts=facts,
        weights_gib=75.0,
        basis_gib=95.59,
        ctx=262_144,
        max_num_seqs=16,
        kv_cache_dtype=None,
        ts="2026-09-18T00:00:00Z",
    )
    assert obs is not None
    assert obs.get("trust") == "measured"

    # Latest observation has weights_gib but kv_kib_per_token=None; earlier observation has kv_kib_per_token=64.0
    observations = [
        {
            "repo_id": "org/m",
            "backend": "stock",
            "trust": "measured",
            "inputs": {"max_model_len": 32768},
            "measured": {
                "weights_gib": 75.0,
                "kv_kib_per_token": 64.0,
                "overhead_gib": 5.0,
            },
        },
        {
            "repo_id": "org/m",
            "backend": "stock",
            "trust": "measured",
            "inputs": {"max_model_len": 32768},
            "measured": {
                "weights_gib": 75.2,
                "kv_kib_per_token": None,
                "overhead_gib": None,
            },
        },
    ]
    resolved = _discovery.resolve_inputs("org/m", 0.9, 32768, observations=observations, hub_dir=tmp_path)
    assert resolved.weights_gib == 75.2
    assert resolved.kv_kib_per_token == 64.0
    assert resolved.overhead_gib == 5.0


def test_bug19_capacity_zero_gpu_total_invalid_ctx_and_unknown_blue_green(monkeypatch):
    mi = _capacity.ModelInputs(
        repo_id="org/m",
        backend="stock",
        model_max_ctx=32768,
        weights_gib=20.0,
        weights_source="measured",
        kv_kib_per_token=50.0,
        overhead_gib=4.5,
        trust="measured",
    )
    monkeypatch.setenv("SERVEDECK_GPU_TOTAL_MIB", "0")
    # 0 is "unset" to limits.get(), which then asks nvidia-smi: on a box with a
    # GPU that answered with the real card and this test never saw a zero.
    monkeypatch.setattr(_limits, "detect_gpu_total_mib", lambda: 0)
    _limits.reset()
    _capacity.refresh_limits()
    try:
        # Must NOT raise ZeroDivisionError when GPU total is 0!
        est = _capacity.compute(mi, util=0.9, ctx=32768, max_num_seqs=1)
        assert est.can_apply is False
        assert any(f.code == "GPU_TOTAL_UNKNOWN" for f in est.findings)
    finally:
        monkeypatch.setenv("SERVEDECK_GPU_TOTAL_MIB", "97887")
        _limits.reset()
        _capacity.refresh_limits()

    bad_ctx = _capacity.compute(mi, util=0.9, ctx=0, max_num_seqs=1)
    assert bad_ctx.can_apply is False
    assert any(f.code == "CTX_INVALID" for f in bad_ctx.findings)

    unknown_a = _capacity.ModelInputs(repo_id="org/a", backend="stock", model_max_ctx=32768, weights_gib=None, weights_source="unknown")
    unknown_b = _capacity.ModelInputs(repo_id="org/b", backend="stock", model_max_ctx=32768, weights_gib=None, weights_source="unknown")
    bg = _capacity.blue_green(unknown_a, unknown_b, util_a=0.45, util_b=0.45, ctx=32768)
    assert bg.feasible is False


def test_bug20_parallelism_flashnext_id_and_max_num_seqs_clamps():
    assert "different architecture" not in _parallelism.calibration_note("Qwen3.8-Flash-Next-Uncensored-NVFP4")
    assert "different architecture" not in _parallelism.calibration_note("Qwen3.8-Flash-Next-FP4-Local")

    head = _app._headroom(
        main_key="glm53",
        main_id="glm53-flash",
        snapshot={"kv_cache_size_tokens": 400_000},
        full_ctx=32768,
        free_mib=4000,
        max_num_seqs=1,
    )
    assert head["small_requests"] == 1
    assert head["full_context_requests"] == 1

    mixed = _parallelism.mixed_capacity(pool_tokens=400_000, full_ctx=32768, max_num_seqs=1)
    assert all(big <= 1 for big, _small in mixed.rows)


def test_bug21_settings_load_uses_passed_environ_and_validates_port_and_ipv6(tmp_path):
    toml_path = tmp_path / "models.toml"
    toml_path.write_text(BASE_TOML)
    loaded = _settings.load(
        {
            "SERVEDECK_PORT": "19123",
            "SERVEDECK_HOST": "::1",
            "SERVEDECK_MODELS": str(toml_path),
            "SERVEDECK_STATE_DIR": str(tmp_path / "state"),
        }
    )
    assert loaded.listen_port == 19123
    assert loaded.base_url == "http://[::1]:19123"

    with pytest.raises(_settings.SettingsError):
        _settings.load({"SERVEDECK_PORT": "70000"})
    with pytest.raises(_settings.SettingsError):
        _settings.load({"SERVEDECK_PORT": "0"})


def test_bug22_models_type_and_collision_validation(tmp_path):
    # get_type_hints must not raise NameError for Mapping
    hints = typing.get_type_hints(_models._as_util)
    assert "raw" in hints

    bad_cases = [
        # ctx = true (bool must not pass isinstance(..., int))
        BASE_TOML.replace("ctx = 4096", "ctx = true"),
        # negative ctx
        BASE_TOML.replace("ctx = 4096", "ctx = -1024"),
        # port out of range
        BASE_TOML.replace("port = 19003", "port = 70000"),
        # string "false" for vision
        BASE_TOML + 'vision = "false"\n',
        # min_output_tokens > max_output_tokens
        BASE_TOML + "min_output_tokens = 8192\nmax_output_tokens = 4096\n",
        # alias of big colliding with unaliased resident key "small"
        BASE_TOML.replace('aliases = ["big"]', 'aliases = ["big", "small"]'),
    ]
    for idx, text in enumerate(bad_cases):
        p = tmp_path / f"bad_{idx}.toml"
        p.write_text(text)
        with pytest.raises(_models.RegistryError):
            _models.load(p)


# ---------------------------------------------------------------------------
# BUG-23 .. BUG-28: wire, doctor, cli, policies, glm_policies, batch, builds
# ---------------------------------------------------------------------------


def test_bug23_wire_kimi_comment_in_thinking_and_prunes_stale_models_and_keeps_backup(tmp_path, monkeypatch):
    toml_path = tmp_path / "models.toml"
    toml_path.write_text(BASE_TOML)
    reg = _models.load(toml_path)

    existing_kimi = (
        f"{_wire._OWNED_COMMENT}\n"
        '[models."servedeck/retired-old-model"]\n'
        'provider = "servedeck"\n'
        'model = "Old"\n'
        "max_context_size = 4096\n\n"
        "[thinking]\n"
        "# user note inside thinking table\n"
        "enabled = false\n"
        'effort = "low"\n'
    )
    rendered = _wire.render_kimi(reg, existing_kimi, resolve_ctx=lambda m: m.ctx if isinstance(m.ctx, int) else 0)
    parsed = tomllib.loads(rendered)  # used to raise TOMLDecodeError!
    assert parsed["thinking"]["enabled"] is True
    assert parsed["thinking"]["effort"] == "low"
    assert "servedeck/retired-old-model" not in parsed.get("models", {})

    # Same-day second `wire --apply` must not overwrite the original pre-wire backup!
    vscode = tmp_path / "chatLanguageModels.json"
    vscode.write_text('[{"name": "original-user-config", "models": []}]')
    backup_root = tmp_path / "backups"
    monkeypatch.setattr(_wire, "BACKUP_ROOT", backup_root)
    monkeypatch.setattr(
        _wire,
        "WIRE_TARGETS",
        (_wire.WireTarget("vscode", vscode, _wire.render_vscode),),
    )
    assert _cli.main(["wire", "--models-toml", str(toml_path), "--apply"]) == 0
    # Modify models.toml so a second --apply on the same day writes again
    toml2 = tmp_path / "models2.toml"
    toml2.write_text(BASE_TOML.replace("ctx = 4096", "ctx = 8192"))
    assert _cli.main(["wire", "--models-toml", str(toml2), "--apply"]) == 0
    all_backups = [p.read_text() for p in backup_root.rglob("*") if p.is_file()]
    assert any("original-user-config" in b for b in all_backups), (
        f"second same-day wire --apply clobbered original backup: {all_backups}"
    )


def test_bug24_doctor_main_alias_codex_top_level_and_corrupt_config(tmp_path):
    toml_path = tmp_path / "models.toml"
    toml_path.write_text(BASE_TOML)
    reg = _models.load(toml_path)

    vscode = tmp_path / "vscode.json"
    vscode.write_text(
        json.dumps([{"name": "servedeck", "models": [{"id": "main", "url": "http://localhost:8010/v1/chat/completions"}]}])
    )
    codex = tmp_path / "codex.toml"
    codex.write_text(_wire.render_codex(reg, "", resolve_ctx=lambda m: m.ctx if isinstance(m.ctx, int) else 0))
    corrupt = tmp_path / "kimi.toml"
    corrupt.write_text("not valid [[[ toml")

    hub = tmp_path / "hub"
    for repo in ("org/big", "org/other", "org/small"):
        snap = hub / ("models--" + repo.replace("/", "--")) / "snapshots" / "r1"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
        (snap / "model.safetensors").write_bytes(b"0")
    shm = tmp_path / "shm"
    shm.mkdir()

    def fake_get(url: str, timeout: float) -> httpx.Response:
        if "8010" in url:
            # Gateway up, no model currently running -> data: []
            return httpx.Response(200, json={"object": "list", "data": []})
        raise httpx.ConnectError("not listening")

    results = _doctor.run_doctor(
        toml_path,
        client_files={"vscode": vscode, "codex": codex, "kimi": corrupt},
        unit_run=lambda argv: subprocess.CompletedProcess(list(argv), 0, "", ""),
        http_get=fake_get,
        marker_paths=[],
        ptrace_scope=0,
        desired_path=tmp_path / "desired.json",
        shm_dir=shm,
        hub_dir=str(hub),
        power_limit_w=None,
    )
    by_name = {r.name: r for r in results}
    assert by_name["vscode: main"].ok is True, by_name["vscode: main"]
    assert "codex: main" in by_name and by_name["codex: main"].ok is True
    assert by_name["kimi config"].ok is False


def test_bug25_cli_handles_all_terminal_notices_and_smoke_main(tmp_path):
    for reason in (
        "aborted",
        "not_enough_host_ram",
        "conflicting_offload",
        "port_busy",
        "env_scan_failed",
        "vram_accounting_unavailable",
    ):
        assert reason in _cli._TERMINAL, f"{reason} missing from cli._TERMINAL"

    toml_path = tmp_path / "models.toml"
    toml_path.write_text(BASE_TOML)

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content.decode())
        assert body["model"] == "main"
        if "tools" in body:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"tool_calls": [{"function": {"name": "add", "arguments": '{"a":2,"b":3}'}}]}}]},
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

    rc = _cli._cmd_smoke(
        _cli.build_parser().parse_args(["smoke", "main", "--models-toml", str(toml_path)]),
        transport=httpx.MockTransport(handler),
    )
    assert rc == 0


def test_bug26_policies_responses_list_input_glm_multipass_stream_capture_and_batch(tmp_path, monkeypatch):
    # List-shaped /v1/responses input (~40,000 chars = ~10,000 tokens) with ctx=12,000 and floor=8,192:
    # headroom is ~2,000 tokens, so max_output_tokens must clamp <= 2,000, NOT jump to 8,192!
    long_text = "abcd" * 10_000
    raw = json.dumps(
        {
            "model": "glm53",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": long_text}]}],
            "max_output_tokens": 500,
        }
    ).encode()
    transformed = json.loads(
        _policies.apply_request_policies(
            raw,
            model_id="glm53-flash",
            overlay=None,
            floor=8192,
            ctx=12_000,
        )
    )
    assert transformed["max_output_tokens"] <= 2500, transformed["max_output_tokens"]

    # Multi-pass safe_reasoning
    assert "</think>" not in _glm.safe_reasoning("hello </th</think>ink> world")

    # Streaming capture records reasoning_chars and tool_call_ids even when restore_reasoning=False
    monkeypatch.setenv(_glm.CAPTURE_ENV, "1")
    st = _glm.GlmState(capture_dir=tmp_path / "state" / "captures")
    pol = type(
        "Pol",
        (),
        {"sanitize_tool_tags": True, "restore_reasoning": False, "capture": True, "min_output_tokens": None},
    )()
    hook = _glm.begin(pol, model_id="glm53-flash", state=st)
    assert hook is not None
    hook.on_request_body(b'{"model":"glm53-flash","messages":[{"role":"user","content":"hi"}],"stream":true}')

    async def _stream():
        async def _chunks():
            yield b'data: {"choices":[{"index":0,"delta":{"reasoning":"thought-123"}}]}\n\n'
            yield b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"name":"f","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}\n\n'
            yield b"data: [DONE]\n\n"

        out = [c async for c in hook.sse(_chunks())]
        assert out

    asyncio.run(_stream())
    resp_files = sorted((tmp_path / "state" / "captures" / "glm53-flash").glob("resp-*.json"))
    assert resp_files, "expected a streaming response capture file"
    captured_body = json.loads(resp_files[-1].read_text())["body"]
    assert captured_body["reasoning_chars"] == len("thought-123")
    assert captured_body["tool_call_ids"] == ["call_abc"]

    # batch.run must NOT swallow KeyboardInterrupt!
    def raise_kb(_item):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        _batch.run([1], raise_kb, limit=1, metrics_url=None)


def test_bug27_bug28_build_scripts_and_requirements_sync():
    stock_sh = (REPO_ROOT / "builds" / "stock" / "build.sh").read_text(encoding="utf-8")
    assert "$CU/cu13/lib64" not in stock_sh, "builds/stock/build.sh still has double cu13/cu13/lib64 bug"

    for name in ("glm53", "qwen38next", "stock", "stock032"):
        body = (REPO_ROOT / "builds" / name / "build.sh").read_text(encoding="utf-8")
        assert "set -euo pipefail" in body, f"builds/{name}/build.sh missing -e in set -euo pipefail"

    reqs = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "tomli-w" in reqs


def test_bug02_poll_does_not_launch_before_the_port_is_won(tmp_path):
    """Through create_app's own lifespan, not a hand-set flag: our port never
    answers as us (a rival holds it), so the poll's recovery must not launch the
    wanted model. Main's first poll tick ran _recover_desired before the bind."""
    (tmp_path / "models.toml").write_text(APP_TOML)
    reg = _models.load(tmp_path / "models.toml")
    s = _settings.Settings(
        listen_host="127.0.0.1",
        listen_port=8049,
        models_path=tmp_path / "models.toml",
        state_dir=tmp_path / "state",
        unit_prefix="sd-test-",
    )
    s.state_dir.mkdir(parents=True)
    _desired.save(_desired.Desired(main="big"), s.desired_path)
    ctl = AppFakeControl()
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: httpx.Response(503)))
    app = _app.create_app(
        s, registry=reg, control=ctl, client=client, reconcile=True, poll=True, telemetry=False
    )

    async def run() -> None:
        async with app.router.lifespan_context(app):
            await asyncio.sleep(1.5)

    asyncio.run(run())
    assert "start:big" not in ctl.calls, f"launched before winning the port: {ctl.calls}"


def test_bug06_stale_abort_of_another_key_does_not_break_a_later_switch(tmp_path):
    """A cancel that lands on a resident's start which is then refused leaves
    that key in _aborted. The switch's VRAM-release wait must only end early
    for an abort of the model it is switching to."""
    systemd = FakeSystemd()
    systemd.add("model-qwen27b", main_pid=1000)
    box = {"free": 1344}
    ctl = make_control(
        systemd,
        FakeRegistry(BIG27, FLASH, LFM2),
        tmp_path,
        used=lambda: {1000: 90000},
        cgroup_pids=lambda unit: [1000],
        probe=lambda port: ["Qwen3.8-Flash-Next"] if port == 8001 else None,
    )
    ctl._free_mib = lambda: box["free"]
    real_reap = ctl.reap_offload
    # The operator's cancel arrives during the resident start's pre-launch phase.
    ctl.reap_offload = lambda: (ctl.abort("lfm2", stop_unit=False), real_reap())[1]
    refused = ctl.start("lfm2")
    assert isinstance(refused, _control.Refusal), refused
    assert "lfm2" in ctl._aborted
    ctl.reap_offload = real_reap

    reads = iter([1344, 1344, 1500, 2000, 80000])
    ctl._free_mib = lambda: next(reads, 80000)
    result = ctl.switch("flashnext")
    assert not isinstance(result, _control.Refusal), result
    assert not isinstance(result.started, _control.Refusal), result.started
    assert result.started.ready
