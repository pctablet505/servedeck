"""The page the owner had, on the backend the owner has (2026-09-17).

v1's dashboard (``web/index.html`` + ``web/app.js`` + ``web/style.css``, as
they were at tag ``pre-cutover-2026-09-17``) is the page. It reads the v1
payload shapes: a ``state`` document with ``upstream`` / ``vllm`` / ``sizing``
/ ``boot`` / ``supervisor`` blocks, a ``telemetry`` event every poll, a
``/api/models`` rail with disk sizes, ``/api/disk``, a capacity estimate, and
``/api/server/*`` controls. This module builds those shapes from v2's runtime
(control plane, registry, metrics poller, discovery) and registers the
routes. The v2 API is left intact — every v1 key is ADDED to the v2 state
document, never substituted — so the CLI, the gate and the tests keep
working.

Nothing here is a second implementation of the arithmetic: sizing calls
``parallelism``, the estimate calls ``capacity`` and ``kvcalc``, disk sizes
come from ``disksize`` and ``discovery`` — the same modules v1 called.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from servedeck import capacity, discovery, disksize, kvcalc, parallelism, reqstats
from servedeck import gpu as _gpu
from servedeck import models as _models

log = logging.getLogger(__name__)

DEFAULT_MODEL_MAX_CTX = 262144
STARTED_AT = time.time()

#: The phase names the page knows (app.js PHASE_LABELS), in boot order.
PHASES: tuple[str, ...] = (
    "init", "loading_weights", "compiling", "kv_cache", "cuda_graphs", "http_start", "ready",
)
#: control.READY_MARKERS index -> the phase that marker *ends*.
MARKER_PHASE: dict[int, str] = {0: "loading_weights", 1: "kv_cache", 2: "cuda_graphs", 3: "http_start"}

#: The supervisor states the page treats as "a boot is running" (app.js
#: BOOTING_STATES / BUSY_PHASES).
STARTING = "STARTING"
STOPPING = "STOPPING"
READY = "READY"
STOPPED = "STOPPED"
FAILED = "FAILED"


# --------------------------------------------------------------------------
# argv of the live process: the only source for what it is really running
# --------------------------------------------------------------------------


def argv_flag(argv: Sequence[str], flag: str) -> str | None:
    """Space-separated form only, which is how vLLM's launchers spell flags."""
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


def int_flag(argv: Sequence[str], flag: str) -> int | None:
    raw = argv_flag(argv, flag)
    try:
        value = int(raw) if raw is not None else 0
    except ValueError:
        return None
    return value if value > 0 else None


def float_flag(argv: Sequence[str], flag: str) -> float | None:
    raw = argv_flag(argv, flag)
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None


def host_ram(meminfo: str | None = None) -> dict[str, Any]:
    """``MemTotal`` / ``MemAvailable`` in GiB, from /proc/meminfo — the room a
    KV offload buffer (pinned host RAM) can be given."""
    try:
        text = meminfo if meminfo is not None else Path("/proc/meminfo").read_text()
    except OSError:
        return {}
    out: dict[str, Any] = {}
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            out["total_gib"] = round(int(line.split()[1]) / 1048576, 1)
        elif line.startswith("MemAvailable:"):
            out["available_gib"] = round(int(line.split()[1]) / 1048576, 1)
    return out


def process_uptime_s(pid: int) -> int | None:
    """Seconds since ``pid`` started, from ``/proc/<pid>/stat`` — for an adopted
    process no unit start time exists, and "up —" beside a model that has
    served for hours is a lie of omission."""
    if not pid:
        return None
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        uptime = float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    try:
        fields = stat.rsplit(")", 1)[1].split()
        start_ticks = int(fields[19])  # field 22 of stat, 20th after the comm
        hz = os.sysconf("SC_CLK_TCK")
    except (IndexError, ValueError, OSError):
        return None
    return max(0, int(uptime - start_ticks / hz))


def cmdline_of(pid: int) -> list[str]:
    """``/proc/<pid>/cmdline`` split on NUL; empty when the pid is gone."""
    if not pid:
        return []
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [p.decode("utf-8", "replace") for p in data.split(b"\0") if p]


# --------------------------------------------------------------------------
# Boot progress, for the phase bar
# --------------------------------------------------------------------------


@dataclass
class BootTrack:
    key: str
    phase: str = "init"
    #: phase -> elapsed seconds when it was entered (a cumulative clock, the
    #: shape v1's supervisor recorded and the page's track draws).
    phase_times: dict[str, float] = field(default_factory=lambda: {"init": 0.0})
    elapsed_s: float = 0.0
    reached_ready: bool = False
    failed: bool = False


class BootTracker:
    """Folds ``control.Progress`` events into the phase bar's payload."""

    def __init__(self) -> None:
        self.current: BootTrack | None = None

    def record(self, key: str, kind: str, marker_index: int | None, elapsed_s: float) -> None:
        t = self.current
        if t is None or t.key != key or t.reached_ready or t.failed:
            t = BootTrack(key=key)
            self.current = t
        t.elapsed_s = float(elapsed_s or 0.0)
        if kind == "marker" and marker_index is not None:
            phase = MARKER_PHASE.get(int(marker_index))
            if phase:
                t.phase = phase
                t.phase_times.setdefault(phase, t.elapsed_s)
        elif kind == "ready":
            t.phase = "ready"
            t.phase_times.setdefault("ready", t.elapsed_s)
            t.reached_ready = True
        elif kind == "failed":
            t.failed = True

    @property
    def booting(self) -> bool:
        t = self.current
        return t is not None and not t.reached_ready and not t.failed

    def payload(self, actual_state: str) -> dict[str, Any]:
        t = self.current
        return {
            "phases": list(PHASES),
            "phase": t.phase if t else None,
            "phase_times": dict(t.phase_times) if t else {},
            "elapsed_s": t.elapsed_s if t else None,
            "reached_ready": bool(t and t.reached_ready),
            "actual_state": actual_state,
            # No boot history in v2 yet: the page leaves the ETA slot empty
            # rather than asserting a number, exactly as v1 did without one.
            "eta_s": None,
            "eta_p90_s": None,
            "eta_source": None,
            "eta_note": None,
            "cold": None,
        }


def actual_state(
    *, busy: Mapping[str, Any] | None, boot: BootTracker, holder_ready: bool | None
) -> str:
    """The one word the page keys its pill and gates on.

    ``holder_ready`` is None when nothing holds the main slot, True/False
    otherwise. A boot in flight outranks the holder's own readiness (the holder
    may be the model being replaced), and a stop in flight is STOPPING.
    """
    action = (busy or {}).get("action") if busy else None
    if action == "stop":
        return STOPPING
    if boot.booting or action in ("start", "switch", "reconcile"):
        return STARTING
    if holder_ready is None:
        t = boot.current
        return FAILED if (t is not None and t.failed) else STOPPED
    return READY if holder_ready else STARTING


# --------------------------------------------------------------------------
# The state document's v1 blocks
# --------------------------------------------------------------------------


def gpu_payload() -> dict[str, Any]:
    g = _gpu.gpu_summary()
    if g is None:
        return {}
    return {
        "used_mib": g.used_mib,
        "total_mib": g.total_mib,
        "free_mib": g.free_mib,
        "util_percent": g.util_percent,
        "name": g.name,
    }


def live_facts(snapshot: Mapping[str, Any] | None, argv: Sequence[str]) -> dict[str, Any]:
    """The running engine's own numbers (``/metrics`` cache_config_info)."""
    m = snapshot or {}
    facts: dict[str, Any] = {}
    if m.get("reachable") and m.get("kv_cache_size_tokens"):
        facts["kv_tokens"] = int(m["kv_cache_size_tokens"])
        facts["kv_source"] = "engine"
        facts["kv_trust"] = "measured"
        if m.get("kv_cache_max_concurrency"):
            facts["concurrency_x"] = round(float(m["kv_cache_max_concurrency"]), 3)
        if m.get("kv_cache_gpu_util"):
            facts["util_effective"] = float(m["kv_cache_gpu_util"])
    ctx = int_flag(argv, "--max-model-len")
    if ctx:
        facts["ctx"] = ctx
    offload = float_flag(argv, "--kv-offloading-size")
    if offload:
        facts["kv_offload_gib"] = offload
    if "util_effective" not in facts:
        util = float_flag(argv, "--gpu-memory-utilization")
        if util:
            facts["util_effective"] = util
    return facts


def upstream_payload(
    *,
    model: _models.Model | None,
    key: str | None,
    ready: bool,
    pid: int,
    adopted: bool,
    ctx_tokens: int,
    snapshot: Mapping[str, Any] | None,
    argv: Sequence[str],
    fallback_port: int | None,
    reason_when_down: str,
) -> dict[str, Any]:
    port = model.port if model else fallback_port
    url = f"http://localhost:{port}" if port else None
    if model is None:
        return {
            "url": url, "up": False, "model": None, "model_id": None,
            "identity": {"repo_id": None, "source": "unknown", "backend": None,
                         "served_names": [], "mismatch": False},
            "max_model_len": None, "port": port,
            "resolution": {"port": port, "source": "registry", "reason": reason_when_down,
                           "backend": None, "pid": None, "live": False, "candidates": []},
            "live": {},
        }
    names = [model.id, *model.aliases]
    return {
        "url": url,
        "up": bool(ready),
        "model": model.id,
        "model_id": model.repo,
        "identity": {
            "repo_id": model.repo,
            "source": "process" if argv else "unit",
            "backend": model.build,
            "served_names": names,
            "mismatch": False,
        },
        "max_model_len": int_flag(argv, "--max-model-len") or (ctx_tokens or None),
        "port": port,
        "resolution": {
            "port": port,
            "source": "adopted" if adopted else "unit",
            "reason": (
                f"{model.id} holds the main slot"
                + (" (adopted: launched outside servedeck)" if adopted else f" (unit model-{key})")
            ),
            "backend": key,
            "pid": pid or None,
            "live": bool(ready),
            "candidates": [],
        },
        "live": live_facts(snapshot, argv) if ready else {},
    }


def sizing_payload(
    snapshot: Mapping[str, Any] | None,
    *,
    served_name: str | None,
    max_num_seqs: int | None,
    full_ctx: int | None,
) -> dict[str, Any]:
    """v1's parallelism panel, verbatim in shape: window stats + recommendation.

    Every branch degrades to a stated REASON, never to a number that was not
    computed — "we do not know yet" and "8 agents" must never look alike.
    """
    m = snapshot or {}
    window = m.get("prompt_stats") or dict(reqstats.EMPTY_STATS)
    out: dict[str, Any] = {
        "window": window,
        "gen_window": m.get("gen_stats") or dict(reqstats.EMPTY_STATS),
        "calibration_note": parallelism.calibration_note(served_name),
        "running": m.get("running") if m.get("reachable") else None,
        "preemptions": m.get("preemptions") if m.get("reachable") else None,
        "max_num_seqs": max_num_seqs,
        "recommended": None,
        "at_p99": None,
        "over_subscribed": False,
        "reason": None,
        "mixed": None,
        "mixed_reason": None,
    }
    if not m.get("reachable"):
        out["reason"] = "backend not reachable"
        out["mixed_reason"] = out["reason"]
        return out
    pool = m.get("kv_cache_size_tokens")
    if not pool:
        out["reason"] = "engine has not published its KV pool size yet"
        out["mixed_reason"] = out["reason"]
        return out
    if full_ctx:
        sizes: dict[str, tuple[float, bool]] = {}
        for label in ("p50", "p90"):
            pct = window.get(label) or {}
            if pct.get("hi"):
                exact = bool(pct.get("exact") or pct.get("lo") == pct.get("hi"))
                sizes[label] = (float(pct["hi"]), exact)
        out["mixed"] = parallelism.mixed_capacity(
            pool_tokens=int(pool), full_ctx=int(full_ctx), sizes=sizes, max_num_seqs=max_num_seqs,
        ).to_dict()
    else:
        out["mixed_reason"] = "the running server's --max-model-len is not known"
    p90 = (window.get("p90") or {}).get("hi")
    if not p90:
        out["reason"] = (
            f"no requests observed yet — {window.get('n', 0)} of "
            f"{window.get('capacity', reqstats.WINDOW_SIZE)} in the window"
        )
        return out
    n = window.get("n", 0)
    rec = parallelism.recommend(
        pool_tokens=int(pool), prompt_tokens=float(p90), max_num_seqs=max_num_seqs,
        basis=f"p90 of the last {n} request{'' if n == 1 else 's'}",
    )
    out["recommended"] = rec.to_dict()
    p99 = (window.get("p99") or {}).get("hi")
    if p99:
        out["at_p99"] = parallelism.recommend(
            pool_tokens=int(pool), prompt_tokens=float(p99), max_num_seqs=max_num_seqs,
            basis=f"p99 of the last {n} request{'' if n == 1 else 's'}",
        ).to_dict()
    running = m.get("running") or 0
    out["over_subscribed"] = bool(running > rec.n)
    return out


def telemetry_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    """The ``telemetry`` SSE event v1 published every poll."""
    return {
        "gpu": state.get("gpu_v1") or {},
        "vllm": state.get("vllm") or {},
        "sizing": state.get("sizing") or {},
        "uptime_s": int(time.time() - STARTED_AT),
    }


# --------------------------------------------------------------------------
# The model rail and the disk line
# --------------------------------------------------------------------------


def model_rows(
    registry: _models.Registry,
    entries: Sequence[Any],
    *,
    ctx_for: Any,
    live_keys: Mapping[str, bool],
) -> list[dict[str, Any]]:
    """One rail card per registry model, with what the hub cache knows about
    it. ``live_keys`` maps key -> ready for the models that are up."""
    by_repo = {e.repo_id: e for e in entries}
    rows: list[dict[str, Any]] = []
    for key, model in registry.models.items():
        e = by_repo.get(model.repo)
        ctx = 0
        try:
            ctx = int(ctx_for(model) or 0)
        except Exception:  # noqa: BLE001 - a ctx error must not blank the rail
            ctx = 0
        g = lambda name, default=None: getattr(e, name, default) if e is not None else default  # noqa: E731
        servable = bool(e is not None and g("servable", False))
        rows.append(
            {
                "key": key,
                "repo_id": model.repo,
                "name": model.id,
                "backend": model.build,
                "servable": servable,
                "unservable_reason": (
                    None if servable
                    else (g("reason") if e is not None else "not in the local hub cache")
                ),
                "disk_bytes": int(g("disk_bytes", 0) or 0),
                "disk_local_bytes": int(g("disk_local_bytes", g("disk_bytes", 0)) or 0),
                "quant": g("quant_algo") or "—",
                "model_max_ctx": ctx or g("max_position_embeddings") or DEFAULT_MODEL_MAX_CTX,
                "trust": "measured" if key in live_keys else "estimated",
                "weights_gib": (round(float(g("safetensors_gib")), 2) if g("safetensors_gib") is not None else None),
                "serving": bool(live_keys.get(key)),
            }
        )
    return rows


def disk_payload(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hub = discovery.default_hub_dir()
    fs = disksize.filesystem_usage(hub if hub.exists() else Path("/"))
    hub_bytes = sum(int(r.get("disk_bytes") or 0) for r in rows)
    hub_local = sum(int(r.get("disk_local_bytes") or 0) for r in rows)
    return {
        "path": fs.path,
        "total_bytes": fs.total_bytes,
        "used_bytes": fs.used_bytes,
        "avail_bytes": fs.avail_bytes,
        "used_pct": round(fs.used_pct, 1),
        "hub_bytes": hub_bytes,
        "hub_local_bytes": hub_local,
        "hub_foreign_bytes": hub_bytes - hub_local,
        "unit": "bytes",
    }


# --------------------------------------------------------------------------
# The capacity estimate behind the allocator
# --------------------------------------------------------------------------


def cache_flags(model: _models.Model | None) -> tuple[str | None, str | None]:
    """``--kv-cache-dtype`` / ``--mamba-ssm-cache-dtype`` from the registry's
    own flags: they change the cache LAYOUT, so an estimate without them
    describes a server nobody starts."""
    if model is None:
        return None, None
    flags = list(model.flags)
    kv = argv_flag(flags, "--kv-cache-dtype")
    ssm = argv_flag(flags, "--mamba-ssm-cache-dtype")
    return (None if kv in (None, "auto") else kv), ssm


def _model_inputs(
    repo_id: str, util: float, ctx: int, entry: Any, kv_dtype: str | None, ssm_dtype: str | None
) -> tuple[Any, capacity.ModelInputs]:
    ri = discovery.resolve_inputs(
        repo_id, util, ctx, kv_cache_dtype=kv_dtype, mamba_ssm_dtype=ssm_dtype
    )
    mi = capacity.ModelInputs(
        repo_id=repo_id,
        backend=ri.backend or "stock",
        model_max_ctx=ri.model_max_ctx or DEFAULT_MODEL_MAX_CTX,
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
    return ri, mi


def _native_ctx_fit(
    repo_id: str, util: float, entry: Any, kv_dtype: str | None, ssm_dtype: str | None,
    *, native: int, known: dict[int, tuple[Any, capacity.CapacityResult]],
) -> tuple[int, str | None]:
    def at(length: int) -> tuple[Any, capacity.CapacityResult]:
        if length not in known:
            ri, mi = _model_inputs(repo_id, util, length, entry, kv_dtype, ssm_dtype)
            known[length] = (ri, capacity.compute(mi, util=util, ctx=length, max_num_seqs=1))
        return known[length]

    ri_max, r_max = at(native)
    if ri_max.weights_source == "unknown" or not ri_max.kv_kib_per_token or r_max.kv_tokens <= 0:
        return 0, None
    fit = capacity.single_request_fit(native, lambda length: at(length)[1].kv_tokens)
    reason = capacity.ctx_fit_reason(
        model_max_ctx=native, pool_at_max=r_max.kv_tokens, fit=fit, util=util,
        kv_source=ri_max.kv_source,
    )
    return fit, reason


def kv_geometry(repo_id: str, ctx: int, kv: str | None, ssm: str | None) -> dict[str, Any] | None:
    try:
        cfg = discovery.load_model_config(repo_id)
        if cfg is None:
            return None
        geo = kvcalc.geometry(cfg, kv_cache_dtype=kv, mamba_ssm_dtype=ssm)
        out = kvcalc.summarise(geo, ctx)
        out["launch_flags"] = {"kv_cache_dtype": kv, "mamba_ssm_cache_dtype": ssm}
        return out
    except Exception:  # noqa: BLE001 - a bad config must not blank the panel
        return None


def offload_tokens_for(offload_gib: float | None, kv_tokens: int, kv_gib: float) -> int | None:
    """How many tokens the KV offload parks: the host buffer at the same
    bytes-per-token the GPU pool resolved to. None when either is unknown."""
    if not offload_gib or offload_gib <= 0 or kv_tokens <= 0 or kv_gib <= 0:
        return None
    return int(offload_gib * (kv_tokens / kv_gib))


def own_gpu_mib(pids: Sequence[int]) -> int:
    """VRAM held by our own model processes: the holder's pid tree, plus any
    ``VLLM::`` worker (vLLM renames its engine processes) — never "anything
    called python", which would count a training job as ours."""
    family = set(pids)
    own = 0
    try:
        for a in _gpu.compute_apps():
            if a.pid in family or (a.process_name or "").startswith("VLLM::"):
                own += a.used_mib
    except Exception:  # noqa: BLE001
        return 0
    return own


def training_marker_hits() -> list[str]:
    hits = []
    for marker in capacity.TRAINING_MARKER_PATHS:
        try:
            if Path(marker).expanduser().exists():
                hits.append(marker)
        except OSError:
            continue
    return hits


def estimate_payload(
    *,
    repo_id: str,
    util: float,
    ctx: int,
    seqs: int,
    model: _models.Model | None,
    entries: Sequence[Any],
    own_pids: Sequence[int],
    ptrace_scope: int | None,
    state: str | None,
    kv_offload_gib: float | None = None,
    running_offload_gib: float | None = None,
) -> dict[str, Any]:
    entry = next((e for e in entries if e.repo_id == repo_id), None)
    kv_dtype, ssm_dtype = cache_flags(model)
    ri, mi = _model_inputs(repo_id, util, ctx, entry, kv_dtype, ssm_dtype)
    g = _gpu.gpu_summary()
    live = capacity.LiveFacts(
        gpu_responsive=_gpu.gpu_alive(),
        total_mib=g.total_mib if g else None,
        used_mib=g.used_mib if g else None,
        own_mib=own_gpu_mib(own_pids),
        training_markers=training_marker_hits(),
        ptrace_scope=ptrace_scope,
        ptrace_scope_pinned=ptrace_scope_pinned(),
        actual_state=state,
    )
    r = capacity.compute(mi, util=util, ctx=ctx, max_num_seqs=seqs, live=live)
    known: dict[int, tuple[Any, capacity.CapacityResult]] = {}
    if mi.model_max_ctx == ctx:
        known[ctx] = (ri, r)
    fit, fit_reason = _native_ctx_fit(
        repo_id, util, entry, kv_dtype, ssm_dtype, native=mi.model_max_ctx, known=known,
    )
    offload_gib = (
        float(kv_offload_gib) if kv_offload_gib is not None
        else (float_flag(list(model.flags), "--kv-offloading-size") if model else None)
    ) or 0.0
    ram = host_ram()
    # What can be pinned: what is available now plus what the running server
    # already holds for this purpose (a restart gives it back first).
    offload_max = None
    if ram.get("available_gib") is not None:
        offload_max = int(ram["available_gib"] + (running_offload_gib or 0.0) - 4)  # keep 4 GiB for the box
        offload_max = max(0, offload_max)
    return {
        "kv_gib": round(r.kv_gib, 2),
        "kv_tokens": r.kv_tokens,
        "offload_gib": offload_gib,
        "offload_tokens": offload_tokens_for(offload_gib, r.kv_tokens, r.kv_gib),
        "offload_max_gib": offload_max,
        "host_ram": ram,
        "budget_gib": round(r.budget_gib, 2),
        "concurrency_x": round(r.concurrency_x, 2),
        "agents_at_ctx": r.agents_at_ctx,
        "effective_parallel": r.effective_parallel,
        "max_single_ctx": r.max_single_ctx,
        "ctx_max_model": r.ctx_max_model,
        "ctx_max_fit": fit,
        "ctx_fit_reason": fit_reason,
        "full_at_once": (
            parallelism.recommend(
                pool_tokens=r.kv_tokens, prompt_tokens=ctx, basis=f"one {ctx:,}-token request",
            ).to_dict()
            if r.kv_tokens > 0 else None
        ),
        "agents": seqs,
        "confidence": r.confidence,
        "can_apply": r.can_apply,
        "weights_gib": mi.weights_gib,
        "kv_kib_per_token": mi.kv_kib_per_token,
        "kv_source": ri.kv_source,
        "kv_geometry": kv_geometry(repo_id, ctx, kv_dtype, ssm_dtype),
        "bar": {
            "weights_pct": round(r.bar.weights_pct, 2),
            "kv_pct": round(r.bar.kv_pct, 2),
            "overhead_pct": round(r.bar.overhead_pct, 2),
            "free_pct": round(r.bar.free_pct, 2),
        },
        "findings": [
            {
                "code": f.code, "level": f.level, "title": f.title, "detail": f.detail,
                "fix": f.fix, "fix_action": f.fix_action,
            }
            for f in r.findings
        ],
    }


# --------------------------------------------------------------------------
# Wiring into the app: the state document, the routes
# --------------------------------------------------------------------------


def key_for_repo(registry: _models.Registry, repo_id: str | None, backend: str | None) -> str | None:
    """The page posts what v1 knew a model by: its repo id, or a name."""
    if not repo_id:
        return None
    for key, m in registry.models.items():
        if m.repo == repo_id or m.id == repo_id or key == repo_id:
            return key
    if backend and backend in registry.models:
        return backend
    return None


def overrides_from(body: Mapping[str, Any]) -> tuple[float | None, dict[str, str | None]]:
    """``util`` / ``ctx`` / ``max_num_seqs`` from the allocator's POST body."""
    util = None
    if body.get("util") is not None:
        util = max(0.05, min(0.99, float(body["util"])))
    argv: dict[str, str | None] = {}
    if body.get("ctx") is not None:
        argv["--max-model-len"] = str(int(body["ctx"]))
    if body.get("max_num_seqs") is not None:
        argv["--max-num-seqs"] = str(max(1, int(body["max_num_seqs"])))
    if body.get("kv_offload_gib") is not None:
        gib = max(0.0, float(body["kv_offload_gib"]))
        # 0 means no offload at all: the flag is removed, not passed as 0.
        argv["--kv-offloading-size"] = f"{gib:g}" if gib > 0 else None
    return util, argv


def augment_state(
    *,
    registry: _models.Registry,
    state: Mapping[str, Any],
    boot: BootTracker,
    busy: Mapping[str, Any] | None,
    main_key: str | None,
    main_ctx: int,
    main_snapshot: Mapping[str, Any] | None,
    desired_main: str | None,
    uptimes: Mapping[str, float | None],
) -> dict[str, Any]:
    """The v1 blocks, computed from what ``build_state`` already has."""
    rows = {m["key"]: m for m in state.get("models", [])}
    holder = rows.get(main_key) if main_key else None
    model = registry.models.get(main_key) if main_key else None
    ready = bool(holder and holder.get("ready"))
    pid = int(holder.get("pid") or 0) if holder else 0
    argv = cmdline_of(pid) if pid else []
    word = actual_state(busy=busy, boot=boot, holder_ready=(ready if holder else None))
    adopted = bool(holder and "adopted" in str(holder.get("unit_state", "")))
    first_main = next((m for m in registry.models.values() if m.slot == "main"), None)
    seqs = int_flag(argv, "--max-num-seqs") or (int_flag(list(model.flags), "--max-num-seqs") if model else None)
    full_ctx = int_flag(argv, "--max-model-len") or (main_ctx or None)
    return {
        "upstream": upstream_payload(
            model=model, key=main_key, ready=ready, pid=pid, adopted=adopted,
            ctx_tokens=main_ctx, snapshot=main_snapshot, argv=argv,
            fallback_port=(first_main.port if first_main else None),
            reason_when_down="no model holds the main slot; pick one on the left and Apply",
        ),
        "config": {
            "backend": model.build if model else None,
            "gpu_mem_util": float_flag(argv, "--gpu-memory-utilization"),
            "max_subagents": None,
        },
        "gpu_v1": gpu_payload(),
        "vllm": dict(main_snapshot) if main_snapshot else {"reachable": False},
        "sizing": sizing_payload(
            main_snapshot, served_name=(model.id if model else None),
            max_num_seqs=seqs, full_ctx=full_ctx,
        ),
        "boot": boot.payload(word),
        "control_enabled": True,
        "control_note": "",
        "supervisor": {
            "actual_state": word,
            "desired_state": "RUNNING" if desired_main else "STOPPED",
            "phase": boot.current.phase if boot.current else None,
            "repo_id": model.repo if model else None,
            "backend": model.build if model else None,
            "max_model_len": full_ctx,
        },
        "host": host_ram(),
        "uptime_s": int(time.time() - STARTED_AT),
        "server_uptime_s": (
            int(uptimes[main_key]) if main_key and uptimes.get(main_key) is not None
            else process_uptime_s(pid)
        ),
    }


def register(app: FastAPI, rt: Any) -> None:
    """The v1 routes the page calls. ``rt`` is the app's Runtime; the imports
    of the app's helpers are local to keep this module import-free of app."""
    from servedeck import app as _app

    def entries() -> list[Any]:
        return discovery.discover_models()

    async def _rail() -> tuple[list[dict[str, Any]], str | None]:
        ents = await _app.asyncio.to_thread(entries)
        live = rt.routes.live()
        live_keys = {k: bool(v.ready) for k, v in live.items()}
        rows = model_rows(rt.registry, ents, ctx_for=rt.routes.ctx_for, live_keys=live_keys)
        serving = next((r["name"] for r in rows if r["serving"]), None)
        return rows, serving

    @app.get("/api/disk")
    async def api_disk() -> dict[str, Any]:
        rows, _serving = await _rail()
        return await _app.asyncio.to_thread(disk_payload, rows)

    @app.post("/api/capacity/estimate")
    async def api_estimate(body: dict[str, Any]) -> Any:
        repo = body.get("repo_id")
        if not repo:
            return JSONResponse({"error": "repo_id is required"}, status_code=400)
        key = key_for_repo(rt.registry, repo, body.get("backend"))
        model = rt.registry.models.get(key) if key else None
        holder = _app._main_holder(rt, rt.routes.live())
        view = rt.routes.live_view(holder) if holder else None
        pids = [view.pid] if view and view.pid else []
        running_argv = cmdline_of(view.pid) if view and view.pid else []
        raw_off = body.get("kv_offload_gib")
        try:
            return await _app.asyncio.to_thread(
                estimate_payload,
                kv_offload_gib=(float(raw_off) if raw_off is not None else None),
                running_offload_gib=float_flag(running_argv, "--kv-offloading-size"),
                repo_id=model.repo if model else str(repo),
                util=float(body.get("util", 0.96)),
                ctx=int(body.get("ctx", DEFAULT_MODEL_MAX_CTX)),
                seqs=int(body.get("max_num_seqs", 1)),
                model=model,
                entries=entries(),
                own_pids=pids,
                ptrace_scope=_read_ptrace_scope(),
                state=(rt.state.get("supervisor") or {}).get("actual_state") if rt.state else None,
            )
        except Exception as exc:  # noqa: BLE001 - the page shows the text
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=400)

    def _launch(body: dict[str, Any], action: str) -> JSONResponse:
        key = key_for_repo(rt.registry, body.get("repo_id"), body.get("backend"))
        if key is None:
            return JSONResponse(
                {"error": f"no registry model matches {body.get('repo_id')!r}"}, status_code=404
            )
        if rt.busy is not None:
            return JSONResponse(
                {"error": f"servedeck is busy: {rt.busy.get('label')}"}, status_code=409
            )
        util, argv = overrides_from(body)
        holder = _app._main_holder(rt, rt.routes.live())
        model = rt.registry.models[key]
        if holder is not None and model.slot == "main":
            same = holder == key
            label, verb = (f"restart {key}", "switch") if same else (f"switch {key}", "switch")
            work = lambda: rt.control.switch(  # noqa: E731
                key, on_progress=_app._progress_publisher(rt.hub, key, rt.boot),
                util=util, argv_overrides=argv, relaunch=same,
            )
        else:
            label, verb = f"start {key}", "start"
            work = lambda: rt.control.start(  # noqa: E731
                key, on_progress=_app._progress_publisher(rt.hub, key, rt.boot),
                util=util, argv_overrides=argv,
            )
        _app.asyncio.create_task(_app._run_mutation(rt, label, work, action=verb, key=key))
        return JSONResponse({"accepted": True, "action": action, "key": key}, status_code=202)

    @app.post("/api/server/start")
    async def api_server_start(body: dict[str, Any] | None = None) -> Any:
        return _launch(body or {}, "start")

    @app.post("/api/server/restart")
    async def api_server_restart(body: dict[str, Any] | None = None) -> Any:
        return _launch(body or {}, "restart")

    @app.post("/api/server/stop")
    async def api_server_stop() -> Any:
        holder = _app._main_holder(rt, rt.routes.live())
        if holder is None:
            return JSONResponse({"error": "nothing holds the main slot"}, status_code=409)
        if rt.busy is not None:
            return JSONResponse(
                {"error": f"servedeck is busy: {rt.busy.get('label')}"}, status_code=409
            )
        _app.asyncio.create_task(
            _app._run_mutation(
                rt, f"stop {holder}", lambda: rt.control.stop(holder), action="stop", key=holder
            )
        )
        return JSONResponse({"accepted": True, "action": "stop", "key": holder}, status_code=202)

    @app.post("/api/server/adopt")
    async def api_server_adopt(body: dict[str, Any] | None = None) -> Any:
        result = await _app.asyncio.to_thread(rt.control.adopt)
        holder = _app._main_holder(rt, rt.routes.refresh())
        view = rt.routes.live_view(holder) if holder else None
        return {
            "adopted": list(getattr(result, "adopted", []) or []),
            "pid": view.pid if view else None,
            "key": holder,
        }


def ptrace_scope_pinned(sysctl_dir: str | Path = "/etc/sysctl.d") -> bool:
    """Does a sysctl.d file set ``kernel.yama.ptrace_scope = 0``? Then 0 is the
    box's configured state (the owner's 90-servedeck.conf) and no finding
    should call it "left relaxed"."""
    try:
        files = sorted(Path(sysctl_dir).glob("*.conf"))
    except OSError:
        return False
    for f in files:
        try:
            for line in f.read_text().splitlines():
                bare = line.split("#", 1)[0].replace(" ", "")
                if bare == "kernel.yama.ptrace_scope=0":
                    return True
        except OSError:
            continue
    return False


def _read_ptrace_scope() -> int | None:
    try:
        return int(Path("/proc/sys/kernel/yama/ptrace_scope").read_text().strip())
    except (OSError, ValueError):
        return None
