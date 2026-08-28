"""Coldstart — FastAPI application.

Module is named `app` because run.sh and systemd/servedeck.service both import
`servedeck.app:app`. SPEC.md called it api.py; those two files won the tie
because they are already installed.

Scope of THIS file today: read-only observability + capacity estimation + a
pass-through proxy. Server control (start/stop/restart) belongs to
supervisor.py and is NOT wired here yet — the buttons that would call it are
rendered disabled rather than lying about what they do.

Binds 127.0.0.1 only. Never runs sudo. Never starts a model server.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import capacity, config, events, gpu, registry, shellconfig, supervisor as _sup
from .metrics import MetricsPoller

HERE = Path(__file__).resolve().parent
WEB = HERE / "web"   # inside the package, so it ships in the wheel
def _candidate_logs() -> list[Path]:
    """Backend log files, the one serving our port first.

    Ordering matters: these logs are append-only across restarts, so we read
    the LAST match in whichever file belongs to the running backend.
    """
    cfg = config.get()
    ours = [b for b in cfg.backends if b.port == rt.port]
    others = [b for b in cfg.backends if b.port != rt.port]
    return [b.log_path for b in (*ours, *others)]

UPSTREAM_HOST = "http://localhost"
STARTED_AT = time.time()

app = FastAPI(title="Coldstart", docs_url=None, redoc_url=None)


# ----------------------------------------------------------------- state --
class Runtime:
    """Process-wide mutable state. One instance, created at startup."""

    def __init__(self) -> None:
        cfg = _safe_config()
        self.port = int(cfg.get("PORT") or _default_port())
        self.upstream = f"{UPSTREAM_HOST}:{self.port}"
        self.poller = MetricsPoller(self.upstream)
        self.metrics: dict[str, Any] = {"reachable": False}
        self.gpu: dict[str, Any] = {}
        self.serving_model: str | None = None
        self.upstream_up = False
        self.client: httpx.AsyncClient | None = None
        self._models_cache: list[dict[str, Any]] | None = None

    def config(self) -> dict[str, str]:
        return _safe_config()


def _default_port() -> int:
    """Port of the first configured backend, or 8000 if none are declared."""
    backends = config.get().backends
    return backends[0].port if backends else 8000


def _safe_config() -> dict[str, str]:
    try:
        return shellconfig.read_config()
    except Exception:  # noqa: BLE001 - a missing .config must not break the UI
        return {}


rt = Runtime()

# One supervisor for the process. Created lazily: constructing it touches the
# state directory, and an import-time failure would take the whole UI down
# rather than just disabling the controls.
_supervisor: _sup.Supervisor | None = None
_supervisor_error: str | None = None


def sup() -> _sup.Supervisor | None:
    global _supervisor, _supervisor_error
    if _supervisor is None and _supervisor_error is None:
        try:
            _supervisor = _sup.Supervisor()
        except Exception as exc:  # noqa: BLE001
            _supervisor_error = f"{type(exc).__name__}: {exc}"
    return _supervisor




# ------------------------------------------------------------- SSE hub ----
# events.EventHub, not a local one: it supports Last-Event-ID replay, so a
# browser that reconnects does not silently lose the events it missed.
hub = events.EventHub()


# ----------------------------------------------------------- background ---
async def _poll_loop() -> None:
    assert rt.client is not None
    while True:
        try:
            snap = await rt.poller.scrape(rt.client)
            rt.metrics = snap.to_dict()
            rt.upstream_up = snap.reachable

            g = gpu.gpu_summary()
            rt.gpu = (
                {
                    "used_mib": g.used_mib,
                    "total_mib": g.total_mib,
                    "free_mib": g.free_mib,
                    "util_percent": g.util_percent,
                    "name": g.name,
                }
                if g
                else {}
            )

            if rt.upstream_up and rt.serving_model is None:
                rt.serving_model = await _fetch_served_model()

            hub.publish(
                "telemetry",
                {"gpu": rt.gpu, "vllm": rt.metrics, "uptime_s": int(time.time() - STARTED_AT)},
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the poller must never die
            hub.publish("notice", {"level": "warn", "code": "poll_error", "body": str(exc)[:200]})
        await asyncio.sleep(2.0)


_KV_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
_KVGIB_RE = re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB")
_CONC_RE = re.compile(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x")
_WEIGHTS_RE = re.compile(r"Model loading took\s*([\d.]+)\s*GiB")


def _live_boot_facts() -> dict[str, Any]:
    """Read the RUNNING server's real KV size from its boot log.

    The live 'KV in use' readout must be a fraction of what the running engine
    actually allocated - NOT of some other model's estimate. vLLM's /metrics
    exposes kv_cache_usage_perc but not the absolute token total, and that
    total is only ever printed once, at boot.
    """
    facts: dict[str, Any] = {}
    # Read the LAST match: launcher logs are append-only across restarts, so
    # the first match is the oldest boot's numbers.
    for log in _candidate_logs():
        try:
            if not log.exists():
                continue
            text = log.read_text(errors="replace")
        except OSError:
            continue
        kv_all = _KV_RE.findall(text)
        if not kv_all:
            continue
        facts["kv_tokens"] = int(kv_all[-1].replace(",", ""))
        gib_all = _KVGIB_RE.findall(text)
        if gib_all:
            facts["kv_gib"] = float(gib_all[-1])
        conc_all = _CONC_RE.findall(text)
        if conc_all:
            facts["ctx"] = int(conc_all[-1][0].replace(",", ""))
            facts["concurrency_x"] = float(conc_all[-1][1])
        w_all = _WEIGHTS_RE.findall(text)
        if w_all:
            facts["weights_gib"] = float(w_all[-1])
        # Derive the utilization actually in force. vLLM never prints it, but
        # budget = weights + kv + overhead, and util = budget / total. Without
        # this the UI's slider shows a value the server is not running at.
        if "weights_gib" in facts and "kv_gib" in facts:
            budget = facts["weights_gib"] + facts["kv_gib"] + capacity.OVERHEAD_GIB_DEFAULT
            facts["util_effective"] = round(budget / capacity.GPU_TOTAL_GIB, 3)
        facts["source"] = str(log)
        break
    return facts


async def _fetch_served_model() -> str | None:
    try:
        r = await rt.client.get(f"{rt.upstream}/v1/models", timeout=3.0)  # type: ignore[union-attr]
        if r.status_code == 200:
            data = r.json().get("data") or []
            if data:
                return data[0].get("id")
    except Exception:  # noqa: BLE001
        pass
    return None


@app.on_event("startup")
async def _startup() -> None:
    rt.client = httpx.AsyncClient()
    app.state.poller_task = asyncio.create_task(_poll_loop())
    # Adopt a server that is already running, so the UI shows READY rather
    # than STOPPED and a later crash is classified as crash-while-serving.
    s = sup()
    if s is not None:
        try:
            outcome = await s.reconcile_startup()
            hub.publish("notice", {"level": "info", "code": "reconciled", "body": outcome})
        except Exception as exc:  # noqa: BLE001
            hub.publish("notice", {"level": "warn", "code": "reconcile_failed", "body": str(exc)[:200]})


@app.on_event("shutdown")
async def _shutdown() -> None:
    task = getattr(app.state, "poller_task", None)
    if task:
        task.cancel()
    if rt.client:
        await rt.client.aclose()


# ------------------------------------------------------------- helpers ----
def _model_rows() -> list[dict[str, Any]]:
    if rt._models_cache is not None:
        return rt._models_cache
    rows: list[dict[str, Any]] = []
    for e in registry.discover_models():
        if getattr(e, "skipped", False):
            continue
        # Direct attribute access, NOT getattr-with-default: a field rename must
        # raise here, not silently render "262144 ctx / no quant" for every row.
        rows.append(
            {
                "repo_id": e.repo_id,
                "name": e.repo_id.split("/")[-1],
                "backend": e.backend,
                "servable": e.servable,
                "unservable_reason": e.reason,
                "disk_gib": round(e.safetensors_gib or 0.0, 2),
                "quant": e.quant_algo or "—",
                "model_max_ctx": e.max_position_embeddings or 262144,
            }
        )
    rows.sort(key=lambda r: (not r["servable"], r["name"]))
    rt._models_cache = rows
    return rows


def _own_gpu_mib() -> int:
    """VRAM held by OUR vLLM server, to discount when checking free memory.

    Matching any process whose name contains "python" is wrong: a training job
    holding 60 GiB would be counted as ours, making NOT_ENOUGH_FREE_VRAM
    unreachable and green-lighting a config that then OOMs. Attribute only
    processes that are actually part of the server on our upstream port.
    """
    try:
        from . import procctl

        listener = procctl.listener_pid(rt.port)
    except Exception:  # noqa: BLE001
        listener = None
    if listener is None:
        return 0

    family = {listener}
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                status = (entry / "status").read_text(errors="replace")
            except OSError:
                continue
            for line in status.splitlines():
                if line.startswith("PPid:"):
                    if int(line.split()[1]) in family:
                        family.add(int(entry.name))
                    break
    except OSError:
        pass

    own = 0
    try:
        for a in gpu.compute_apps():
            if a.pid in family or (a.process_name or "").startswith("VLLM::"):
                own += a.used_mib
    except Exception:  # noqa: BLE001
        return 0
    return own


def _estimate(repo_id: str, util: float, ctx: int, seqs: int) -> dict[str, Any]:
    ri = registry.resolve_inputs(repo_id, util, ctx)
    # servable/reason live on ModelEntry, not ResolvedInputs - look them up
    # rather than defaulting servable=True, which made MODEL_UNSERVABLE dead.
    entry = next((e for e in registry.discover_models() if e.repo_id == repo_id), None)
    mi = capacity.ModelInputs(
        repo_id=repo_id,
        backend=ri.backend or "inline",
        model_max_ctx=ri.model_max_ctx or 262144,
        weights_gib=ri.weights_gib,
        weights_source=ri.weights_source,
        kv_kib_per_token=ri.kv_kib_per_token,
        overhead_gib=ri.overhead_gib or capacity.OVERHEAD_GIB_DEFAULT,
        trust=ri.trust,
        servable=entry.servable if entry else True,
        unservable_reason=(entry.reason if entry else None),
        model_type=ri.model_type or "",
        used_ctx_for_rate=ri.matched_ctx or ctx,
        known_kv_rates=ri.other_ctx_kv_rates or {},
    )
    g = gpu.gpu_summary()
    # own_mib matters: the VRAM held by the server we are ALREADY running is not
    # a competitor for the config being estimated - restarting reclaims it first.
    # Without this, every estimate blocks with "not enough free VRAM" simply
    # because the model is currently up.
    own = _own_gpu_mib()
    live = capacity.LiveFacts(
        gpu_responsive=gpu.gpu_alive(),
        total_mib=g.total_mib if g else None,
        used_mib=g.used_mib if g else None,
        own_mib=own,
    )
    r = capacity.compute(mi, util=util, ctx=ctx, max_num_seqs=seqs, live=live)
    return {
        "kv_gib": round(r.kv_gib, 2),
        "kv_tokens": r.kv_tokens,
        "budget_gib": round(r.budget_gib, 2),
        "concurrency_x": round(r.concurrency_x, 2),
        "agents_at_ctx": r.agents_at_ctx,
        "effective_parallel": r.effective_parallel,
        "max_single_ctx": r.max_single_ctx,
        "confidence": r.confidence,
        "can_apply": r.can_apply,
        "weights_gib": mi.weights_gib,
        "kv_kib_per_token": mi.kv_kib_per_token,
        "bar": {
            "weights_pct": round(r.bar.weights_pct, 2),
            "kv_pct": round(r.bar.kv_pct, 2),
            "overhead_pct": round(r.bar.overhead_pct, 2),
            "free_pct": round(r.bar.free_pct, 2),
        },
        "findings": [
            {
                "code": f.code,
                "level": f.level,
                "title": f.title,
                "detail": f.detail,
                "fix": f.fix,
                "fix_action": f.fix_action,
            }
            for f in r.findings
        ],
    }


def _state() -> dict[str, Any]:
    cfg = rt.config()
    return {
        "upstream": {
            "url": rt.upstream,
            "up": rt.upstream_up,
            "model": rt.serving_model,
            "port": rt.port,
            # the RUNNING engine's own numbers, not an estimate for some other model
            "live": _live_boot_facts() if rt.upstream_up else {},
        },
        "config": {
            "backend": cfg.get("BACKEND", "unknown"),
            "gpu_mem_util": cfg.get("GPU_MEM_UTIL"),
            "max_subagents": cfg.get("CODEX_MAX_SUBAGENTS"),
        },
        "gpu": rt.gpu,
        "vllm": rt.metrics,
        "control_enabled": sup() is not None,
        "control_note": _supervisor_error or "",
        "supervisor": (sup().snapshot() if sup() is not None else {}),
        "uptime_s": int(time.time() - STARTED_AT),
    }


# -------------------------------------------------------------- routes ----
@app.get("/api/health")
async def api_health() -> dict[str, Any]:
    return {"ok": True, "upstream_up": rt.upstream_up}


@app.get("/api/state")
async def api_state() -> dict[str, Any]:
    return _state()


@app.get("/api/models")
async def api_models() -> dict[str, Any]:
    return {"models": _model_rows(), "serving": rt.serving_model}


@app.post("/api/capacity/estimate")
async def api_estimate(body: dict[str, Any]) -> Any:
    repo = body.get("repo_id")
    if not repo:
        return JSONResponse({"error": "repo_id is required"}, status_code=400)
    try:
        return _estimate(
            repo,
            float(body.get("util", 0.96)),
            int(body.get("ctx", 262144)),
            int(body.get("max_num_seqs", 1)),
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=400)


def _need_sup() -> Any:
    s = sup()
    if s is None:
        return JSONResponse(
            {"error": _supervisor_error or "supervisor unavailable"}, status_code=503
        )
    return s


@app.post("/api/server/adopt")
async def api_adopt(body: dict[str, Any] | None = None) -> Any:
    """Bring an already-running server under management.

    On startup, a server that is running while desired_state is STOPPED is
    deliberately left alone -- Servedeck does not assume a process it did not
    start is wanted. Adopting is the explicit human act that says it is, and
    it is what makes Stop and crash-detection work for that process.
    """
    s = _need_sup()
    if isinstance(s, JSONResponse):
        return s
    port = int((body or {}).get("port") or s.desired.port or rt.port)
    from . import procctl

    pid = procctl.listener_pid(port)
    if pid is None:
        return JSONResponse({"error": f"nothing is listening on port {port}"}, status_code=409)
    if not procctl.is_attributable(pid):
        return JSONResponse(
            {
                "error": (
                    f"a server is serving on :{port} but its process (pid {pid}) cannot be "
                    "attributed, so Servedeck cannot control it. Stop it from the terminal "
                    "that launched it."
                )
            },
            status_code=409,
        )
    d = s.desired
    d.desired_state = "RUNNING"
    d.port = port
    if not d.repo_id:
        d.repo_id = rt.serving_model
    if not d.backend:
        d.backend = _safe_config().get("BACKEND") or "flashnext"
    _sup.save_desired(d, s.state_dir)
    s._run_repo_id, s._run_backend = d.repo_id, d.backend
    s._adopt_ready(pid)
    hub.publish("state", _state())
    return JSONResponse({"adopted": True, "pid": pid, "port": port}, status_code=200)


@app.post("/api/server/stop")
async def api_stop() -> Any:
    s = _need_sup()
    if isinstance(s, JSONResponse):
        return s
    # Returns immediately; progress arrives on /api/events. Stop sets
    # desired_state=STOPPED BEFORE signalling, so the resulting exit is
    # recorded as intent and never auto-restarted.
    asyncio.create_task(_run_and_report(s.stop(), "stop"))
    return JSONResponse({"accepted": True, "action": "stop"}, status_code=202)


@app.post("/api/server/start")
async def api_start(body: dict[str, Any] | None = None) -> Any:
    s = _need_sup()
    if isinstance(s, JSONResponse):
        return s
    b = body or {}
    d = s.desired
    asyncio.create_task(
        _run_and_report(
            s.start(
                repo_id=b.get("repo_id") or d.repo_id,
                backend=b.get("backend") or d.backend,
                served_name=b.get("served_name") or d.served_name,
                port=int(b.get("port") or d.port or rt.port),
                util=float(b["util"]) if b.get("util") is not None else d.util,
                max_model_len=int(b["ctx"]) if b.get("ctx") is not None else d.max_model_len,
                max_num_seqs=int(b["max_num_seqs"]) if b.get("max_num_seqs") is not None else d.max_num_seqs,
            ),
            "start",
        )
    )
    return JSONResponse({"accepted": True, "action": "start"}, status_code=202)


@app.post("/api/server/restart")
async def api_restart(body: dict[str, Any] | None = None) -> Any:
    s = _need_sup()
    if isinstance(s, JSONResponse):
        return s
    mode = (body or {}).get("mode", "immediate")
    asyncio.create_task(_run_and_report(s.restart(mode=mode), f"restart:{mode}"))
    return JSONResponse({"accepted": True, "action": "restart", "mode": mode}, status_code=202)


async def _run_and_report(coro: Any, label: str) -> None:
    """Await a supervisor action, reporting the outcome on the event stream.

    Without this, a failure inside a fire-and-forget task is swallowed and the
    UI simply never changes state.
    """
    try:
        await coro
        hub.publish("notice", {"level": "info", "code": label, "body": f"{label} completed"})
    except Exception as exc:  # noqa: BLE001
        hub.publish(
            "notice",
            {"level": "error", "code": f"{label}_failed", "body": f"{type(exc).__name__}: {exc}"},
        )
    finally:
        hub.publish("state", _state())


@app.get("/api/events")
async def api_events(request: Request) -> StreamingResponse:
    # Resume from where a reconnecting browser left off, if it tells us.
    try:
        last_id: int | None = int(request.headers.get("last-event-id", ""))
    except ValueError:
        last_id = None
    sub, backlog = hub.subscribe(last_event_id=last_id)

    def frame(ev: events.Event) -> str:
        return f"id: {ev.id}\nevent: {ev.type}\ndata: {json.dumps(ev.data)}\n\n"

    async def gen():
        try:
            yield f"event: state\ndata: {json.dumps(_state())}\n\n"
            for ev in backlog:
                yield frame(ev)
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(sub.queue.get(), timeout=15.0)
                    yield frame(ev)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"   # keep proxies from closing an idle stream
        finally:
            hub.unsubscribe(sub)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------- proxy ----
# Pass-through only. NOTE (SPEC C2): /v1/models must never be held or
# synthesised — codex-qwen.sh's is_server_up() probes it with no timeout, so a
# held response hangs the CLI forever and a fake 200 makes it believe a dead
# server is alive. Request holding belongs in gateway.py when that exists.
_PROXY_PREFIXES = ("/v1", "/health", "/ping", "/metrics", "/tokenize", "/detokenize")


@app.api_route("/{path:path}", methods=["GET", "POST", "DELETE", "PUT", "PATCH"])
async def catch_all(path: str, request: Request) -> Any:
    full = "/" + path

    if any(full == p or full.startswith(p + "/") for p in _PROXY_PREFIXES):
        if rt.client is None:
            return JSONResponse({"error": "not ready"}, status_code=503)
        url = f"{rt.upstream}{full}"
        try:
            req = rt.client.build_request(
                request.method,
                url,
                content=await request.body(),
                headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
                params=dict(request.query_params),
                timeout=None,
            )
            resp = await rt.client.send(req, stream=True)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                {
                    "error": {
                        "message": f"Coldstart: upstream {rt.upstream} unreachable ({type(exc).__name__})",
                        "type": "servedeck_upstream_unavailable",
                        "code": "unreachable",
                    }
                },
                status_code=503,
            )

        async def body_iter():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()

        hop = {"content-length", "transfer-encoding", "connection"}
        return StreamingResponse(
            body_iter(),
            status_code=resp.status_code,
            headers={k: v for k, v in resp.headers.items() if k.lower() not in hop},
        )

    # ---- static site --------------------------------------------------
    # NOTE: this catch-all is registered before any StaticFiles mount, so it
    # must serve assets itself - a mount added later never gets reached.
    # index.html references /assets/<file>; the files live flat in web/, so
    # strip the prefix rather than requiring a web/assets/ directory.
    if not WEB.exists():
        return JSONResponse({"error": "web/ is not built yet"}, status_code=503)

    rel = path[len("assets/"):] if path.startswith("assets/") else path

    if rel:
        candidate = (WEB / rel).resolve()
        root = WEB.resolve()
        if candidate.is_file() and str(candidate).startswith(str(root)):
            return FileResponse(candidate)   # FileResponse infers the media type
        # A request with a file extension wants a FILE. Returning index.html
        # with content-type text/html for a missing .css/.js is worse than a
        # 404: the browser drops it silently and the page renders unstyled with
        # a 200 in the network tab. Fail loudly instead.
        if "." in Path(rel).name:
            return JSONResponse({"error": f"not found: {rel}"}, status_code=404)

    return FileResponse(WEB / "index.html")

# (no StaticFiles mount: the catch-all above is registered first and would
# shadow it. Assets are served there, including the /assets/ prefix strip.)
