"""Servedeck — FastAPI application.

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

from . import capacity
from . import disksize
from . import kvcalc, config, events, gpu, registry, shellconfig, supervisor as _sup
from . import updetect
from . import metrics as _metrics_mod
from . import parallelism, reqstats
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
    # log_path is optional: a launcher with no log management of its own has
    # no fixed file to name, and None is not a path to try opening.
    return [b.log_path for b in (*ours, *others) if b.log_path is not None]

UPSTREAM_HOST = "http://localhost"
#: Context length assumed when a model's config.json declares no
#: max_position_embeddings. A fallback, never a claim: it exists so the UI has
#: something to draw, and every real number overrides it.
DEFAULT_MODEL_MAX_CTX = 262144
STARTED_AT = time.time()

app = FastAPI(title="Servedeck", docs_url=None, redoc_url=None)


# ----------------------------------------------------------------- state --
class Runtime:
    """Process-wide mutable state. One instance, created at startup."""

    def __init__(self) -> None:
        #: How this port was chosen, in words. Never None: a dashboard that
        #: cannot find a server must still be able to say what it looked at.
        #: Files only at construction -- this runs at import, and a /proc walk
        #: plus a socket-table read there would be paid by every consumer of
        #: the module, tests included. The first poll resolves it properly.
        self.resolution = _resolve_upstream(discover=False)
        self.port = self.resolution.port
        self.upstream = f"{UPSTREAM_HOST}:{self.port}"
        self.poller = MetricsPoller(self.upstream)
        self.metrics: dict[str, Any] = _metrics_mod.unreachable_snapshot()
        self.gpu: dict[str, Any] = {}
        self.serving_model: str | None = None
        #: Every id /v1/models advertises on the live port. A vLLM server can
        #: advertise several aliases for one loaded model, and taking data[0]
        #: alone made an alias that happened to sort first the whole answer.
        self.serving_models: list[str] = []
        self.upstream_up = False
        self.client: httpx.AsyncClient | None = None
        self._models_cache: list[dict[str, Any]] | None = None
        #: When the model scan was taken. The cache used to have no expiry at
        #: all: scanned once on the first /api/models and then served for the
        #: life of the process. On the day a 95.37 GiB BF16 PLE table was
        #: deleted and a 47.68 GiB FP8 one installed, the panel went on
        #: reporting the pre-deletion sizes for hours -- the whole reason the
        #: displayed disk figure "looked wrong". A whole-hub scan is ~25 ms
        #: warm, so there is nothing to protect with a permanent cache.
        self._models_cache_at: float = 0.0

    def config(self) -> dict[str, str]:
        return _safe_config()

    def retarget(self, *, discover: bool | None = None) -> bool:
        """Re-resolve the upstream and follow it if it moved. True if it moved.

        The port used to be read from the shell config exactly once, at
        construction, and then only ever re-read from that same file. Both
        halves were wrong. The first left the metrics poller, the uptime
        lookup, the running-model probe, the own-VRAM discount and the /v1
        proxy watching a dead port for the life of the process. The second
        made ONE hand-edited file the whole truth: on 2026-09-10 that file's
        last uncommented lines said ``BACKEND="glm53"`` / ``PORT="8002"``,
        GLM had been dead for a week, Flash-Next was serving on :8001, and the
        dashboard reported the box as dead with every figure blank.

        ``discover`` decides whether this tick re-resolves at all. Left as
        None it does so exactly when the answer can change: when what we are
        watching is not answering. While the upstream IS answering, the port
        stays put -- a server we are talking to outranks every file, and the
        cheap files-only re-resolution would happily move us onto whatever the
        shell config names (on this box, the dead backend), whereupon the next
        poll would find that port dead and move us back. A flap every 2 s,
        rebuilding the poller each way, so no throughput figure would ever
        live long enough to be computed.

        The poller is rebuilt rather than re-pointed: it carries a two-sample
        throughput baseline belonging to the OLD server, and carrying that
        across would produce one fabricated rate spanning two processes.
        """
        if discover is None:
            discover = not self.upstream_up
        if not discover:
            return False
        return self.follow(_resolve_upstream(discover=True, current_port=self.port))

    def follow(self, resolution: updetect.Upstream) -> bool:
        """Adopt `resolution` as the answer, moving the port if it changed."""
        self.resolution = resolution
        if resolution.port == self.port:
            return False
        self.port = resolution.port
        self.upstream = f"{UPSTREAM_HOST}:{self.port}"
        self.poller = MetricsPoller(self.upstream)
        self.metrics = _metrics_mod.unreachable_snapshot()
        self.serving_model = None
        self.serving_models = []
        # The cached socket-table answer belongs to the OLD port.
        _invalidate_listener()
        return True


def _configured_ports() -> list[tuple[str, int]]:
    """(backend name, port) for every backend servedeck.toml declares, in
    declaration order. Empty when there is no readable config at all."""
    try:
        return [(b.name, b.port) for b in config.get().backends if b.port]
    except Exception:  # noqa: BLE001 - an unreadable config must not blind the UI
        return []


def _desired_target() -> tuple[object, str | None]:
    """(port, backend) out of state/desired.json — what Servedeck last WANTED.

    Read through the supervisor when one already exists, and straight off the
    file otherwise: constructing a Supervisor touches the state directory and
    starts reconciliation, which is far too much to do just to read a port.
    """
    if _supervisor is not None:
        return _supervisor.desired.port, _supervisor.desired.backend
    try:
        d = _sup.load_desired()
    except Exception:  # noqa: BLE001
        return None, None
    return d.port, d.backend


def _live_servers() -> list[updetect.LiveServer]:
    """Every live vLLM api-server process holding a listening socket.

    From ``/proc`` and the socket table — the system itself — never from a
    file. This is the fact that outranks every configuration: a process that
    is running is not somebody's intention.
    """
    from . import procctl

    try:
        procs = procctl.scan_vllm_processes()
    except Exception:  # noqa: BLE001 - a scan failure means "found nothing"
        return []
    return [
        updetect.LiveServer(
            pid=p.pid, port=p.port, backend=p.venv, cmdline=tuple(p.cmdline)
        )
        for p in procs
        if p.role == procctl.ROLE_SERVER and p.port
    ]


def _port_probe():
    """A ``port -> pid`` probe backed by ONE socket-table read.

    The resolver asks about several ports; ``procctl.listener_pid`` forks an
    ``ss`` per question, and asking four times a poll for one table is how the
    dashboard's own cost grows faster than the box it watches.
    """
    from . import procctl

    try:
        table = procctl.listening_pids()
    except Exception:  # noqa: BLE001
        table = {}
    return lambda port: table.get(port)


def _resolve_upstream(
    *, discover: bool = True, current_port: int | None = None
) -> updetect.Upstream:
    """Which port to watch and why — see updetect.py for the precedence.

    ``discover=False`` resolves from files alone (no process scan, no socket
    probing), which is what construction wants.
    """
    cfg = _safe_config()
    desired_port, desired_backend = _desired_target()
    return updetect.resolve(
        scan=_live_servers if discover else (lambda: []),
        # None, not a probe that always says no: an unprobed port must be
        # reported as unprobed, never as "nothing is listening there".
        probe=_port_probe() if discover else None,
        shell_port=cfg.get("PORT"),
        shell_backend=(cfg.get("BACKEND") or "").strip() or None,
        desired_port=desired_port,
        desired_backend=desired_backend,
        backends=_configured_ports(),
        current_port=current_port,
        fallback_port=_default_port(),
    )


def _training_marker_hits() -> list[str]:
    """Marker files that exist right now. Empty when none do — and empty when
    none are configured, which is the same thing to the caller."""
    hits = []
    for marker in capacity.TRAINING_MARKER_PATHS:
        try:
            if Path(marker).expanduser().exists():
                hits.append(marker)
        except OSError:
            continue
    return hits


def _server_uptime_s() -> int | None:
    """Uptime of the process actually serving on rt.port, or None.

    procctl is imported here, not at module scope: app.py deliberately keeps
    that import local (see api_adopt), and referencing it globally silently
    raised NameError into the except below — which read as "no uptime".
    """
    from . import procctl

    try:
        pid = _listener_pid_now()
        if not pid:
            return None
        # Whose uptime is this? The pid holding the port is not necessarily a
        # model server: with the shell config (or, before the ordering fix in
        # supervisor.start(), a refused start) naming someone else's port, this
        # reported 105,114 s of an unrelated always-on service as the model
        # server's uptime while no model was running at all. procctl's own rule
        # 4 -- a port that answers with no attributable pid is UNMANAGED, not
        # guessed at -- applies to every figure taken off that pid.
        if not procctl.is_attributable(pid):
            return None
        return procctl.process_uptime_s(pid)
    except Exception:  # noqa: BLE001
        return None


def _default_port() -> int:
    """Port of the first configured backend, or 8000 if none are declared."""
    backends = config.get().backends
    return backends[0].port if backends else 8000


def _safe_config() -> dict[str, str]:
    try:
        return shellconfig.read_config()
    except Exception:  # noqa: BLE001 - a missing .config must not break the UI
        return {}


# One supervisor for the process. Created lazily: constructing it touches the
# state directory, and an import-time failure would take the whole UI down
# rather than just disabling the controls. Declared BEFORE the runtime because
# resolving the upstream port reads desired state through it when one exists.
_supervisor: _sup.Supervisor | None = None
_supervisor_error: str | None = None

rt = Runtime()


def sup() -> _sup.Supervisor | None:
    global _supervisor, _supervisor_error
    if _supervisor is None and _supervisor_error is None:
        try:
            _supervisor = _sup.Supervisor()
        except Exception as exc:  # noqa: BLE001
            _supervisor_error = f"{type(exc).__name__}: {exc}"
    return _supervisor


def _snap() -> dict[str, Any]:
    """The supervisor's snapshot, or {} when there is no supervisor at all.

    The guard belongs here rather than at each of the two call sites that build
    a payload: `sup().snapshot() if sup() is not None else {}` calls sup() twice,
    and a third copy written as `sup() and sup().snapshot()` would return None
    rather than {} and put a null where the page expects an object.
    """
    s = sup()
    return s.snapshot() if s is not None else {}


# ------------------------------------------------------------- SSE hub ----
# events.EventHub, not a local one: it supports Last-Event-ID replay, so a
# browser that reconnects does not silently lose the events it missed.
hub = events.EventHub()


# ----------------------------------------------------------- background ---
async def _poll_loop() -> None:
    assert rt.client is not None
    while True:
        try:
            if rt.retarget():
                hub.publish(
                    "notice",
                    {"level": "info", "code": "retargeted",
                     "body": f"now watching {rt.upstream} — {rt.resolution.reason}"},
                )
            # A request cannot exceed the engine's own --max-model-len, so
            # that is the ceiling for the histogram's open-ended +Inf bucket.
            # Set every poll rather than once: a restart at a different
            # context length must not keep the old ceiling.
            rt.poller.ceiling_tokens = _running_max_model_len()
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

            if rt.upstream_up and not rt.serving_models:
                _SERVED_NAME_REPO.clear()   # a new server may be a new model
                rt.serving_models = await _fetch_served_models()
                rt.serving_model = rt.serving_models[0] if rt.serving_models else None
            if not rt.upstream_up:
                rt.serving_model = None
                rt.serving_models = []

            # A server that comes back on its own — started from a terminal
            # after a blocker was cleared — must be noticed. Without this the
            # dashboard sits on a stale FAILED while the model serves happily,
            # and a later crash is misfiled because nothing is tracking it.
            await _recover_if_server_returned()

            hub.publish(
                "telemetry",
                {
                    "gpu": rt.gpu,
                    "vllm": rt.metrics,
                    "sizing": _sizing_payload(),
                    "uptime_s": int(time.time() - STARTED_AT),
                },
            )
            _publish_state_if_changed()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the poller must never die
            hub.publish("notice", {"level": "warn", "code": "poll_error", "body": str(exc)[:200]})
        await asyncio.sleep(2.0)


#: Signature of the last `state` event published by the poll loop, so an
#: idle box does not re-broadcast an unchanged payload every 2 s.
_LAST_STATE_SIG: tuple[Any, ...] | None = None


def _publish_state_if_changed() -> None:
    """Publish a `state` event when the state a human can see has moved.

    The page paints the phase strip, the elapsed clock and the boot ETA only
    from `state` (app.js: paintState -> paintBoot), and the 5 s /api/state poll
    was deliberately removed in favour of this stream. The poll loop published
    only `telemetry`, so across two real boots the browser received three
    `state` events -- all three published by a control POST returning, at the
    instant phase was still null. The boot panel therefore never rendered at
    all: bootActive() needs a phase, and by the time one existed nothing was
    publishing.

    While a boot is in progress every poll publishes, because the elapsed
    counter is part of the payload and a counter that only moves on a state
    CHANGE does not count.
    """
    global _LAST_STATE_SIG
    snap = _snap()
    booting = snap.get("phase") is not None and not snap.get("reached_ready")
    sig = (
        snap.get("actual_state"),
        snap.get("desired_state"),
        snap.get("phase"),
        bool(snap.get("reached_ready")),
        snap.get("last_error"),
        snap.get("unmanaged_pid"),
        rt.upstream_up,
        rt.upstream,
        rt.serving_model,
    )
    if booting or sig != _LAST_STATE_SIG:
        _LAST_STATE_SIG = sig
        hub.publish("state", _state())


_KV_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
_KVGIB_RE = re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB")
_CONC_RE = re.compile(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x")
_WEIGHTS_RE = re.compile(r"Model loading took\s*([\d.]+)\s*GiB")
_MODEL_RE = re.compile(r"'model_tag':\s*'([^']+)'")


def _argv_flag(argv: list[str], flag: str) -> str | None:
    """The value following `flag` in a command line, or None.

    Space-separated form only — that is how vLLM's own launchers spell their
    flags. A flag in trailing position has no value and must not read off the
    end.
    """
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _max_model_len_from(argv: list[str]) -> int | None:
    """--max-model-len as an int, or None if absent or not a length."""
    raw = _argv_flag(argv, "--max-model-len")
    try:
        value = int(raw) if raw is not None else 0
    except ValueError:
        return None
    return value if value > 0 else None


#: (port, monotonic deadline, pid) of the last socket-table lookup.
_LISTENER_CACHE: tuple[int, float, int | None] | None = None

#: How long one socket-table answer is reused. Deliberately far shorter than
#: the dashboard's own poll interval, so no two polls ever share an answer:
#: the point is only that the several questions asked WITHIN one poll — the
#: running model id, the running max_model_len, the serving identity, the
#: server's uptime — share the one `ss` between them instead of spawning one
#: each. A longer window would start hiding a server that just came up.
_LISTENER_TTL_S = 0.25


def _invalidate_listener() -> None:
    """Forget the cached socket-table answer.

    Called wherever Servedeck itself changes what is listening (a start, a
    stop, a port move), so the next question re-reads rather than waiting out
    the TTL.
    """
    global _LISTENER_CACHE
    _LISTENER_CACHE = None


def _listener_pid_now() -> int | None:
    """PID listening on rt.port, from a lookup shared across one poll.

    ``procctl.listener_pid`` shells out to ``ss``: one fork+exec per call, and
    /api/state asked it five separate times for one listening socket, several
    times a second, for a fact that cannot change between the questions.

    Keyed on the port, so a repoint (``Runtime.repoint``) can never be
    answered from the previous port's lookup — that would be the stale-port
    class of bug this file already carries two fixes for.
    """
    global _LISTENER_CACHE
    from . import procctl

    now = time.monotonic()
    cached = _LISTENER_CACHE
    if cached is not None and cached[0] == rt.port and now < cached[1]:
        return cached[2]
    pid = procctl.listener_pid(rt.port)
    _LISTENER_CACHE = (rt.port, now + _LISTENER_TTL_S, pid)
    return pid


def _listener_argv() -> list[str]:
    """The command line of whatever is listening on rt.port."""
    from . import procctl

    pid = _listener_pid_now()
    return procctl.cmdline_of(pid) if pid is not None else []


def _running_max_model_len() -> int | None:
    """The context length the LIVE server is actually serving.

    The serving line's "N ctx" has now been wrong twice from two different
    stale sources: first the UI's own slider, then supervisor.max_model_len.
    The second is desired config — what Servedeck WANTS — and for a server it
    adopted rather than launched, the two need not agree at all. The process's
    own command line is the only authoritative source, exactly as
    _running_model_id() already argues for the model name.
    """
    return _max_model_len_from(_listener_argv())


def _running_max_num_seqs() -> int | None:
    """``--max-num-seqs`` of the LIVE server, from its own command line.

    The hard ceiling on parallelism: however much KV is free, the scheduler
    will not run more sequences than this concurrently, so a recommendation
    above it is a recommendation to build a queue. Read from the process for
    the same reason as --max-model-len -- servedeck.toml says what Servedeck
    would launch, which for an adopted server need not be what is running.
    vLLM's /metrics does not publish it (cache_config_info carries the cache
    settings only), so the command line is the only live source.
    """
    raw = _argv_flag(_listener_argv(), "--max-num-seqs")
    try:
        value = int(raw) if raw is not None else 0
    except ValueError:
        return None
    return value if value > 0 else None


def _running_model_id() -> str | None:
    """The model the LIVE process is serving, from its own command line.

    Authoritative, unlike a log file (which can be stale, or belong to a
    different run) and unlike --served-model-name (which an operator may reuse
    across different models, leaving two models indistinguishable over the
    API).
    """
    return _model_id_from(_listener_argv())


def _model_id_from(argv: list[str]) -> str | None:
    """The repo a command line loads: ``--model X`` or ``vllm serve X``.

    Split out of _running_model_id() so adoption can ask the same question
    about a process on a port we are NOT currently watching -- which is the
    whole of discovering a server somewhere else.
    """
    if not argv:
        return None
    for i, a in enumerate(argv):
        if a == "--model" and i + 1 < len(argv):
            return argv[i + 1]
    # positional form: `vllm serve <model>`
    for i, a in enumerate(argv):
        if a.endswith("vllm") and i + 2 < len(argv) and argv[i + 1] == "serve":
            return argv[i + 2]
    return None


def _boot_log_candidates(
    backend: str | None = None, *, boot_log_dir: Path | None = None
) -> tuple[Path, ...]:
    """Logs that could hold the running server's boot numbers, best first.

    Chosen by BACKEND, not by port. The old rule ordered by which backend owns
    rt.port, which cannot name the log of a backend that declares none — so
    such a deployment opened some other backend's log first, and that file can
    be a stale symlink to a months-old boot of a different model carrying a
    real "GPU KV cache size: N tokens" line. Only _live_boot_facts()'
    model_tag guard kept that number off the dashboard, and that guard is a
    backstop, not a selection rule.

    The other backends' logs stay on the list as fallbacks — a server can be
    adopted after a hand launch into any of them — but behind the one that
    belongs to the backend we believe is running.
    """
    if backend is None:
        s = sup()
        backend = (s.desired.backend if s is not None else None) or _safe_config().get("BACKEND")
    if boot_log_dir is None:
        s = sup()
        if s is not None:
            boot_log_dir = s.state_dir / _sup.BOOT_LOG_DIRNAME
    primary = tuple(Path(p) for p in _sup._default_log_paths(backend, boot_log_dir=boot_log_dir))
    rest = tuple(p for p in _candidate_logs() if p not in primary)
    return primary + rest


def _boot_payload() -> dict[str, Any]:
    """Boot progress for the phase bar: the ordered phase list, the per-phase
    times the supervisor recorded, the elapsed clock, and the ETA.

    The page has drawn none of this since the prototype — #phases/#elapsed
    shipped markup with no painter, so a multi-minute boot sat on "Elapsed —"
    with an empty track. The supervisor already tracks the phase and its
    per-phase times; the only thing computed here is the ETA, which needs the
    boot history and must stay attributed when it falls back to SPEC.md's
    calibration figures (history.eta_for handles that; this only forwards it).

    The history read happens ONLY while a boot is in progress. In the steady
    state — server up, nothing booting — the bar is not drawn, so the 5 s
    state poll must not open history.jsonl twice for a bar nobody sees.
    """
    from . import history, phases

    snap = _snap()
    out: dict[str, Any] = {
        "phases": [p.value for p in phases.PHASE_ORDER],
        "phase": snap.get("phase"),
        "phase_times": snap.get("phase_times") or {},
        "elapsed_s": snap.get("run_elapsed_s"),
        "reached_ready": bool(snap.get("reached_ready")),
        "eta_s": None,
        "eta_p90_s": None,
        "eta_source": None,
        "eta_note": None,
        "cold": None,
    }
    repo_id = snap.get("repo_id") or rt.serving_model
    booting = out["phase"] is not None and not out["reached_ready"]
    if not booting or not repo_id:
        return out
    backend = snap.get("backend") or rt.config().get("BACKEND") or ""
    try:
        cold = not history.has_prior_success(repo_id, backend)
        eta = history.eta_for(repo_id, backend, cold=cold)
    except Exception:  # noqa: BLE001 - an unreadable history must not blank the bar
        return out
    out["cold"] = cold
    out["eta_s"] = eta.median_total_s
    out["eta_p90_s"] = eta.p90_total_s
    out["eta_source"] = eta.source
    out["eta_note"] = eta.note
    return out


def _sizing_payload() -> dict[str, Any]:
    """The parallelism panel's whole payload: window stats + the recommendation.

    Computed in Python, not in the page, for one reason: the formula has to be
    testable. ``tests/test_parallelism.py`` reproduces the measured
    concurrency table against :func:`parallelism.recommend`; a copy of the same
    arithmetic living in app.js would be a second, untested implementation that
    silently disagrees.

    Everything here degrades to a stated REASON rather than to a number. "We do
    not know yet" and "8 agents" must never look alike on a panel whose whole
    job is to tell you how hard to push the GPU.
    """
    m = rt.metrics or {}
    window = m.get("prompt_stats") or dict(reqstats.EMPTY_STATS)
    out: dict[str, Any] = {
        "window": window,
        "gen_window": m.get("gen_stats") or dict(reqstats.EMPTY_STATS),
        "calibration_note": parallelism.calibration_note(rt.serving_model),
        "running": m.get("running") if m.get("reachable") else None,
        "preemptions": m.get("preemptions") if m.get("reachable") else None,
        "max_num_seqs": _running_max_num_seqs(),
        "recommended": None,
        "at_p99": None,
        "over_subscribed": False,
        "reason": None,
    }
    if not m.get("reachable"):
        out["reason"] = "backend not reachable"
        return out
    pool = m.get("kv_cache_size_tokens")
    if not pool:
        # The pool must be the RUNNING engine's own kv_cache_size_tokens.
        # Falling back to the 280,813 the cost curve was calibrated on would
        # produce a confident recommendation for a server launched at a
        # different --gpu-memory-utilization, which is how you over-subscribe.
        out["reason"] = "engine has not published its KV pool size yet"
        return out
    p90 = (window.get("p90") or {}).get("hi")
    if not p90:
        out["reason"] = (
            f"no requests observed yet — {window.get('n', 0)} of "
            f"{window.get('capacity', reqstats.WINDOW_SIZE)} in the window"
        )
        return out

    seqs = out["max_num_seqs"]
    n = window.get("n", 0)
    rec = parallelism.recommend(
        pool_tokens=int(pool),
        prompt_tokens=float(p90),
        max_num_seqs=seqs,
        basis=f"p90 of the last {n} request{'' if n == 1 else 's'}",
    )
    out["recommended"] = rec.to_dict()

    p99 = (window.get("p99") or {}).get("hi")
    if p99:
        out["at_p99"] = parallelism.recommend(
            pool_tokens=int(pool),
            prompt_tokens=float(p99),
            max_num_seqs=seqs,
            basis=f"p99 of the last {n} request{'' if n == 1 else 's'}",
        ).to_dict()

    # The warning. Live concurrency above the recommendation is the condition;
    # num_preemptions_total is the confirmation, because preemption is what
    # over-subscription actually DOES -- vLLM evicts a sequence's KV and
    # recomputes it, so work already paid for is thrown away.
    running = m.get("running") or 0
    out["over_subscribed"] = bool(running > rec.n)
    return out


def _live_boot_facts() -> dict[str, Any]:
    """Read the RUNNING server's real KV size from its boot log.

    The live 'KV in use' readout must be a fraction of what the running engine
    actually allocated - NOT of some other model's estimate.

    FIRST from /metrics. This build publishes the engine's whole resolved cache
    configuration on vllm:cache_config_info, whose LABELS carry
    kv_cache_size_tokens - the same figure the boot log prints once as
    "GPU KV cache size: N tokens". Reading it there needs no log at all, which
    matters because a server launched by hand in a terminal writes to no log
    Servedeck knows about, and the boot log of a PREVIOUS run of another model
    is the wrong file to fall back to.

    The boot log is still read, for the two things /metrics does not carry:
    the KV pool in GiB and the weights measurement.
    """
    facts: dict[str, Any] = {}
    m = rt.metrics or {}
    if m.get("reachable") and m.get("kv_cache_size_tokens"):
        facts["kv_tokens"] = int(m["kv_cache_size_tokens"])
        facts["kv_source"] = "engine"
        facts["kv_trust"] = "measured"
        if m.get("kv_cache_max_concurrency"):
            facts["concurrency_x"] = round(float(m["kv_cache_max_concurrency"]), 3)
        if m.get("kv_cache_gpu_util"):
            facts["util_effective"] = float(m["kv_cache_gpu_util"])
    # Pick the log belonging to the backend actually running on our port, and
    # read the LAST match: launcher logs are append-only across restarts, so
    # the first match is the oldest boot's numbers.
    want = _running_model_id()
    for log in _boot_log_candidates():
        try:
            if not log.exists():
                continue
            text = log.read_text(errors="replace")
        except OSError:
            continue
        # Reject a log written by a different model: these files are reused
        # across runs, and a foreground launch may not write to one at all.
        if want:
            tags = _MODEL_RE.findall(text)
            if tags and tags[-1] != want:
                continue
        kv_all = _KV_RE.findall(text)
        if not kv_all:
            continue
        # /metrics already answered, and it is the running engine rather than
        # a file that outlives it. Do not overwrite it with a log line.
        facts.setdefault("kv_tokens", int(kv_all[-1].replace(",", "")))
        facts.setdefault("kv_source", "boot log")
        facts.setdefault("kv_trust", "measured")
        gib_all = _KVGIB_RE.findall(text)
        if gib_all:
            facts["kv_gib"] = float(gib_all[-1])
        conc_all = _CONC_RE.findall(text)
        if conc_all:
            facts["ctx"] = int(conc_all[-1][0].replace(",", ""))
            facts.setdefault("concurrency_x", float(conc_all[-1][1]))
        w_all = _WEIGHTS_RE.findall(text)
        if w_all:
            facts["weights_gib"] = float(w_all[-1])
        # Derive the utilization actually in force. vLLM never prints it, but
        # budget = weights + kv + overhead, and util = budget / total. Without
        # this the UI's slider shows a value the server is not running at.
        if "weights_gib" in facts and "kv_gib" in facts:
            budget = facts["weights_gib"] + facts["kv_gib"] + capacity.OVERHEAD_GIB_DEFAULT
            facts.setdefault("util_effective", round(budget / capacity.GPU_TOTAL_GIB, 3))
        facts["source"] = str(log)
        break
    return facts


async def _recover_if_server_returned() -> None:
    """Adopt a server that reappeared while we were in a terminal state."""
    s = sup()
    if s is None or not rt.upstream_up:
        return
    if s.actual_state not in ("FAILED", "STOPPED"):
        return
    if s.desired.desired_state != "RUNNING":
        return  # intent says stopped: leave it alone, offer adoption in the UI
    from . import procctl

    pid = _listener_pid_now()
    if pid is None or not procctl.is_attributable(pid):
        return
    try:
        s._run_repo_id = s.desired.repo_id or rt.serving_model
        s._run_backend = s.desired.backend
        s._adopt_ready(pid)
        s.last_error = None
        hub.publish(
            "notice",
            {"level": "info", "code": "recovered",
             "body": f"server returned on :{rt.port} (pid {pid}) — now tracking it"},
        )
        hub.publish("state", _state())
    except Exception as exc:  # noqa: BLE001
        hub.publish("notice", {"level": "warn", "code": "recover_failed", "body": str(exc)[:200]})


async def _fetch_served_models() -> list[str]:
    """Every model id the live server advertises on ``/v1/models``.

    This is the ONLY authority for what the server calls itself. It is not the
    authority for what the server actually loaded -- see _serving_identity().
    """
    return await _probe_models(rt.port)


async def _probe_models(port: int) -> list[str]:
    """``/v1/models`` on any local port: the ids it advertises, or [].

    Takes a port rather than reading rt.upstream because discovery asks this
    of ports the dashboard is NOT watching -- that is how it finds a server
    the configuration has lost track of. Loopback GET only, the read-only kind
    of request SPEC.md's rule 1b allows against a running server.

    Borrows the poller's client when there is one; a request that arrives
    before startup (a test, a CLI call) opens its own rather than failing.
    """
    url = f"{UPSTREAM_HOST}:{port}/v1/models"
    try:
        if rt.client is not None:
            r = await rt.client.get(url, timeout=3.0)
        else:
            async with httpx.AsyncClient() as client:
                r = await client.get(url, timeout=3.0)
        if r.status_code == 200:
            data = r.json().get("data") or []
            return [str(d.get("id")) for d in data if d.get("id")]
    except Exception:  # noqa: BLE001
        pass
    return []


async def _discover_servers() -> list[dict[str, Any]]:
    """Every known port that answers as a model server, best first.

    "Known" is the resolver's own candidate list -- the live processes, the
    shell config's port, desired state's port, every configured backend's port
    -- so adoption and the dashboard can never be looking at two different
    boxes. A port qualifies only if BOTH halves agree: something holds the
    socket, and ``/v1/models`` answers on it. The process's command line is
    read for the same reason adoption reads it at all: it says what was
    actually loaded, while ``/v1/models`` says only what the server calls
    itself.
    """
    from . import procctl

    resolution = _resolve_upstream(discover=True, current_port=rt.port)
    probe = _port_probe()
    found: list[dict[str, Any]] = []
    for candidate in resolution.candidates:
        if any(f["port"] == candidate.port for f in found):
            continue
        pid = candidate.pid if candidate.listening else probe(candidate.port)
        if pid is None:
            continue
        models = await _probe_models(candidate.port)
        if not models:
            continue
        argv = procctl.cmdline_of(pid)
        found.append(
            {
                "port": candidate.port,
                "pid": pid,
                "models": models,
                "repo_id": _model_id_from(argv),
                "backend": procctl.backend_of_pid(pid),
                "attributable": procctl.is_attributable(pid),
                "cmdline": argv,
            }
        )
    # An attributable server first: it is the only kind Servedeck can then
    # stop or watch for a crash. The rest are still reported -- "a server is
    # there and I cannot control it" is an answer, and it is the one the UI
    # needs to explain itself.
    found.sort(key=lambda f: (not f["attributable"], f["port"] != rt.port))
    return found


#: Memo for _repo_for_served_name, keyed on the name. discover_models() walks
#: the whole model cache -- a directory listing and a config.json parse per
#: repo -- and _state() runs on every dashboard poll and every SSE state
#: publish. A served name changes only when a server restarts, so resolving it
#: once per name is the difference between a lookup and a filesystem scan
#: several times a second.
_SERVED_NAME_REPO: dict[str, str | None] = {}


def _repo_for_served_name(name: str | None) -> str | None:
    """The cached repo a served-model-name refers to, if exactly one does.

    Exact match on the repo id or on its final path segment, and only when the
    match is unique -- a name that fits two cached repos identifies neither.
    """
    if not name:
        return None
    if name not in _SERVED_NAME_REPO:
        hits = [
            e.repo_id
            for e in registry.discover_models()
            if e.repo_id == name or e.repo_id.rsplit("/", 1)[-1] == name
        ]
        _SERVED_NAME_REPO[name] = hits[0] if len(hits) == 1 else None
    return _SERVED_NAME_REPO[name]


def _serving_identity() -> dict[str, Any]:
    """Which model is REALLY serving on rt.port, and how we know.

    Sources, in decreasing order of trust:

    1. the live process's own ``--model`` argument -- what vLLM was actually
       told to load, read from ``/proc/<pid>/cmdline``;
    2. ``/v1/models`` on the live port, resolved against the model cache --
       what the server calls itself;
    3. nothing, and then we say nothing.

    The shell config header is deliberately NOT a source. ``BACKEND`` /
    ``MODEL_REPO`` record what somebody last INTENDED; on this box that header
    said GLM while Qwen was serving, and every reader that trusted it was
    wrong together. ``mismatch`` is true when 1 and 2 disagree -- the
    ``--served-model-name`` was reused from another model, which is exactly
    when a name-matching UI shows the wrong row (or no row at all).
    """
    from_process = _running_model_id()
    served_names = list(rt.serving_models)
    from_name = next(
        (r for r in (_repo_for_served_name(n) for n in served_names) if r), None
    )
    repo_id = from_process or from_name
    source = "process" if from_process else ("served_name" if from_name else "unknown")
    pid = None
    backend = None
    if rt.upstream_up:
        from . import procctl

        try:
            pid = _listener_pid_now()
            backend = procctl.backend_of_pid(pid) if pid else None
        except Exception:  # noqa: BLE001
            backend = None
    return {
        "repo_id": repo_id,
        "source": source,
        "backend": backend,
        "served_names": served_names,
        "mismatch": bool(from_process and from_name and from_process != from_name),
    }


@app.on_event("startup")
async def _startup() -> None:
    rt.client = httpx.AsyncClient()
    # Resolve the upstream BEFORE anything can ask for it. Construction is
    # files-only (a /proc walk on import would be paid by every importer), so
    # without this the first page load -- and reconcile_startup below it --
    # would be answered from a port nothing had yet probed.
    rt.retarget(discover=True)
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
#: Seconds a model scan stays fresh. Short enough that a deletion shows up
#: while the operator is still looking at the screen, long enough that the
#: 2 s telemetry tick does not re-walk the hub cache on every poll.
MODELS_CACHE_TTL_S = 5.0


def _model_rows(*, now: float | None = None) -> list[dict[str, Any]]:
    t = time.monotonic() if now is None else now
    if rt._models_cache is not None and (t - rt._models_cache_at) < MODELS_CACHE_TTL_S:
        return rt._models_cache
    rows: list[dict[str, Any]] = []
    for e in registry.discover_models():
        if getattr(e, "skipped", False):
            continue
        # trust/weights_gib are what the model card's provenance badge and the
        # serving-model match are built from. The card read both long before
        # this payload carried either: every model therefore rendered
        # "estimated" (SPEC.md §3 attaches "~25% optimistic historically" to
        # that label, so it is a claim, not decoration) and the weights-based
        # arm of the serving match was dead code.
        #
        # Resolved at the model's OWN max context, which is the only ctx that
        # is a property of the model rather than of the slider — the badge is
        # a coarse "has this ever been booted and measured", and the exact
        # per-configuration answer comes from /api/capacity/estimate.
        ctx_for_trust = e.max_position_embeddings or DEFAULT_MODEL_MAX_CTX
        try:
            ri = registry.resolve_inputs(e.repo_id, capacity.UTIL_THIN_MARGIN, ctx_for_trust)
            trust, weights_gib = ri.trust, ri.weights_gib
        except Exception:  # noqa: BLE001 - a broken measurements store must not blank the rail
            trust, weights_gib = "unknown", None
        # Direct attribute access, NOT getattr-with-default: a field rename must
        # raise here, not silently render "262144 ctx / no quant" for every row.
        rows.append(
            {
                "repo_id": e.repo_id,
                "name": e.repo_id.split("/")[-1],
                "backend": e.backend,
                "servable": e.servable,
                "unservable_reason": e.reason,
                # Bytes, not a pre-rounded GiB float: the unit belongs to the
                # formatter. Shipping "disk_gib" and rendering it beside the
                # letters "GB" is how a 125.99 GiB model came to be displayed
                # as 125.91 GB -- a 7.4% error that reads as a rounding slip.
                "disk_bytes": e.disk_bytes,
                "disk_local_bytes": e.disk_local_bytes,
                "quant": e.quant_algo or "—",
                "model_max_ctx": ctx_for_trust,
                "trust": trust,
                "weights_gib": weights_gib,
            }
        )
    rows.sort(key=lambda r: (not r["servable"], r["name"]))
    rt._models_cache = rows
    rt._models_cache_at = t
    return rows


def _disk_payload() -> dict[str, Any]:
    """Filesystem free/used from statvfs, plus what the hub cache costs.

    The filesystem half never comes from a directory walk. A walk sees only
    what it can read, so it under-reports "used" by every tree the server
    cannot enter, and it cannot see free space at all. ``statvfs`` is the
    kernel's own answer and is what ``df`` prints.

    ``hub_bytes`` versus ``hub_local_bytes``: blobs reached through symlinks
    to another mount are real bytes but sit on another filesystem, so they
    must not be subtracted from the ``df`` figure shown beside them.
    """
    hub = registry.default_hub_dir()
    fs = disksize.filesystem_usage(hub if hub.exists() else Path("/"))
    rows = _model_rows()
    hub_bytes = sum(r["disk_bytes"] for r in rows)
    hub_local = sum(r["disk_local_bytes"] for r in rows)
    return {
        "path": fs.path,
        "total_bytes": fs.total_bytes,
        "used_bytes": fs.used_bytes,
        "avail_bytes": fs.avail_bytes,
        "used_pct": round(fs.used_pct, 1),
        "hub_bytes": hub_bytes,
        "hub_local_bytes": hub_local,
        "hub_foreign_bytes": hub_bytes - hub_local,
        # Named so no consumer has to guess. Everything above is bytes; the
        # UI divides by 1024 and says GiB.
        "unit": "bytes",
    }


def _own_gpu_mib() -> int:
    """VRAM held by OUR vLLM server, to discount when checking free memory.

    Matching any process whose name contains "python" is wrong: a training job
    holding 60 GiB would be counted as ours, making NOT_ENOUGH_FREE_VRAM
    unreachable and green-lighting a config that then OOMs. Attribute only
    processes that are actually part of the server on our upstream port.
    """
    try:
        listener = _listener_pid_now()
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


def _cache_flags(backend: str | None) -> tuple[str | None, str | None]:
    """(kv_cache_dtype, mamba_ssm_cache_dtype) this backend would launch with.

    They belong in the estimate because they change the cache LAYOUT, not just
    its speed: --mamba-ssm-cache-dtype bfloat16 halves the GDN recurrent state
    and is worth 6.9% of the token count on this box's delivered Flash-Next
    configuration. Estimating without them describes a server nobody starts.

    Sources, in the order a launch resolves them: the backend's own fixed
    `env` in servedeck.toml, then EXTRA_ARGS out of the shell config -- which
    supervisor.shell_extra_args() already guards on BACKEND, so another
    backend's flags can never be read as this one's.
    """
    kv = ssm = None
    b = config.get().backend(backend) if backend else None
    if b is not None:
        kv = b.env.get("KV_DTYPE") or None
    argv = _sup.shell_extra_args(backend).split()
    kv = _argv_flag(argv, "--kv-cache-dtype") or kv
    ssm = _argv_flag(argv, "--mamba-ssm-cache-dtype") or ssm
    return kv, ssm


def _kv_geometry(repo_id: str, ctx: int, backend: str | None = None) -> dict[str, Any] | None:
    """The per-architecture KV breakdown, for the panel's tooltip.

    None when the checkpoint's config.json cannot be read locally — never a
    network fetch, and never a fabricated breakdown.
    """
    try:
        cfg = registry.load_model_config(repo_id)
        if cfg is None:
            return None
        kv, ssm = _cache_flags(backend)
        geo = kvcalc.geometry(cfg, kv_cache_dtype=kv, mamba_ssm_dtype=ssm)
        out = kvcalc.summarise(geo, ctx)
        out["launch_flags"] = {"kv_cache_dtype": kv, "mamba_ssm_cache_dtype": ssm}
        return out
    except Exception:  # noqa: BLE001 - a bad config must not blank the panel
        return None


def _estimate(repo_id: str, util: float, ctx: int, seqs: int) -> dict[str, Any]:
    # servable/reason live on ModelEntry, not ResolvedInputs - look them up
    # rather than defaulting servable=True, which made MODEL_UNSERVABLE dead.
    # The entry also names the backend, which is what decides WHICH launch
    # flags apply -- so it has to be read before the estimate, not after.
    entry = next((e for e in registry.discover_models() if e.repo_id == repo_id), None)
    kv_dtype, ssm_dtype = _cache_flags(entry.backend if entry else None)
    ri = registry.resolve_inputs(
        repo_id, util, ctx, kv_cache_dtype=kv_dtype, mamba_ssm_dtype=ssm_dtype
    )
    mi = capacity.ModelInputs(
        repo_id=repo_id,
        backend=ri.backend or "inline",
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
        # capacity.py is pure and never stat()s: somebody allowed I/O has to
        # do it, and nobody was. The TRAINING_MARKER block was therefore
        # unreachable — a guard against starting a server on a GPU a training
        # run is using, that could never fire.
        training_markers=_training_marker_hits(),
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
        # Bounds for the context control. The UI must not offer a length that
        # either the model or the KV budget cannot serve.
        "ctx_max_model": r.ctx_max_model,
        "ctx_max_fit": r.ctx_max_fit,
        "agents": seqs,
        "confidence": r.confidence,
        "can_apply": r.can_apply,
        "weights_gib": mi.weights_gib,
        "kv_kib_per_token": mi.kv_kib_per_token,
        # measured (this repo booted at this context) vs estimated (the
        # per-architecture calculator). The panel labels them differently and
        # must never present the second as the first.
        "kv_source": ri.kv_source,
        "kv_geometry": _kv_geometry(repo_id, ctx, entry.backend if entry else ri.backend),
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
            "model": rt.serving_model,          # --served-model-name
            "model_id": _running_model_id(),     # what is REALLY loaded
            # Which model is serving, and how we know -- never the config
            # header. The UI matches its model list on this.
            "identity": _serving_identity(),
            # The running engine's OWN --max-model-len, not desired config.
            "max_model_len": _running_max_model_len(),
            "port": rt.port,
            # Which port this is, and WHY it is this one. A dashboard that
            # cannot find a server used to render blanks and nothing else:
            # no port, no reason, nothing to act on. The reason is written for
            # a human and the candidate list is what was checked.
            "resolution": rt.resolution.to_dict(),
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
        "sizing": _sizing_payload(),
        "boot": _boot_payload(),
        "control_enabled": sup() is not None,
        "control_note": _supervisor_error or "",
        "supervisor": _snap(),
        # Servedeck's own uptime. The UI's "Serving ... up Nm" must NOT use
        # this: restarting the UI would make a long-running server look fresh.
        "uptime_s": int(time.time() - STARTED_AT),
        "server_uptime_s": _server_uptime_s(),
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
    return {"models": _model_rows(), "serving": rt.serving_model, "disk": _disk_payload()}


@app.get("/api/disk")
async def api_disk() -> dict[str, Any]:
    return _disk_payload()


@app.post("/api/capacity/estimate")
async def api_estimate(body: dict[str, Any]) -> Any:
    repo = body.get("repo_id")
    if not repo:
        return JSONResponse({"error": "repo_id is required"}, status_code=400)
    try:
        return _estimate(
            repo,
            float(body.get("util", 0.96)),
            int(body.get("ctx", DEFAULT_MODEL_MAX_CTX)),
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
    from . import procctl

    explicit = (body or {}).get("port")
    if explicit:
        # A named port is a named port: probe exactly it, and say precisely
        # why it cannot be adopted rather than quietly adopting something else.
        port = int(explicit)
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
    else:
        # No port named: DISCOVER one. This used to fall back to
        # `s.desired.port or rt.port` -- both files -- so pressing Adopt while
        # the configuration named a dead backend re-probed the dead port and
        # reported nothing listening, with a healthy server one port away.
        rt.retarget(discover=True)
        found = await _discover_servers()
        adoptable = next((f for f in found if f["attributable"]), None)
        if adoptable is None:
            if found:
                other = found[0]
                return JSONResponse(
                    {
                        "error": (
                            f"a server is serving on :{other['port']} "
                            f"({', '.join(other['models'])}) but its process "
                            f"(pid {other['pid']}) cannot be attributed, so Servedeck "
                            "cannot control it. Stop it from the terminal that launched it."
                        ),
                        "resolution": rt.resolution.to_dict(),
                    },
                    status_code=409,
                )
            checked = ", ".join(f":{c.port}" for c in rt.resolution.candidates) or "no ports"
            return JSONResponse(
                {
                    "error": (
                        f"no server answered /v1/models on any known port (checked {checked})"
                    ),
                    "reason": rt.resolution.reason,
                    "resolution": rt.resolution.to_dict(),
                },
                status_code=409,
            )
        port, pid = adoptable["port"], adoptable["pid"]
    # Watch what we just adopted. Adoption used to leave rt.port alone, so
    # adopting a server on another port produced a managed server the
    # dashboard still could not see -- and _serving_identity() below would
    # have read the command line of whatever was on the OLD port.
    rt.follow(
        updetect.Upstream(
            port=port,
            source=updetect.LIVE_PROCESS,
            reason=f"adopted by hand: pid {pid} is serving on :{port}",
            pid=pid,
            live=True,
            candidates=(updetect.Candidate(port, updetect.LIVE_PROCESS, None, True, pid),),
        )
    )
    d = s.desired
    d.desired_state = "RUNNING"
    d.port = port
    # Adopt what is RUNNING, not what a header says was intended. `d.repo_id`
    # was filled from the served-model-name (an alias an operator reuses) and
    # `d.backend` from the shell config's BACKEND -- the exact header that
    # said GLM while Qwen was serving. Both now come from the live process,
    # and only fall back when the process cannot be read at all.
    #
    # Read the process DIRECTLY here rather than only through
    # _serving_identity(): that helper answers about the port the dashboard is
    # watching and only once the upstream has been observed up, which at the
    # instant of an adoption it has not been -- so on this path its answer was
    # empty and the shell config's header was silently taking over again.
    ident = _serving_identity()
    d.repo_id = (
        _model_id_from(procctl.cmdline_of(pid))
        or ident["repo_id"]
        or d.repo_id
        or rt.serving_model
    )
    d.backend = (
        procctl.backend_of_pid(pid)
        or ident["backend"]
        or d.backend
        or (_safe_config().get("BACKEND") or None)
    )
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
    # supervisor.start() is idempotent: it returns at its first statement when
    # the server is READY or mid-transition. Reporting that as 202 accepted is
    # how the dashboard's "Apply & restart" came to change nothing at all while
    # the page logged a success. Say so instead, and name the endpoint that
    # does work.
    busy = getattr(s, "actual_state", "STOPPED")
    if busy in ("READY", "STARTING", "PREFLIGHT", "STOPPING", "DRAINING"):
        return JSONResponse(
            {
                "error": (
                    f"the server is {busy}; a start would be ignored. "
                    "POST /api/server/restart to apply new settings, or stop it first."
                ),
                "actual_state": busy,
            },
            status_code=409,
        )
    repo_id = b.get("repo_id") or d.repo_id
    backend = b.get("backend") or d.backend
    # Port and served name come from the backend/model being started, not from
    # whatever the PREVIOUS run left in desired.json -- see the two resolvers'
    # docstrings. Falling through to d.port/d.served_name is what launched one
    # backend on another's port under the other's model name.
    port = _sup.resolve_port(backend, d, explicit=b.get("port"), fallback=rt.port)
    served_name = _sup.resolve_served_name(
        backend, repo_id, d, explicit=b.get("served_name")
    )
    asyncio.create_task(
        _run_and_report(
            s.start(
                repo_id=repo_id,
                backend=backend,
                served_name=served_name,
                port=int(port or rt.port),
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
    b = body or {}
    mode = b.get("mode", "immediate")
    # The same body the Apply button sends to /api/server/start, honoured here
    # too: this is the only endpoint that actually relaunches, so it is the
    # only one that can change ctx / max_num_seqs / util on a running server.
    # Port and served name are re-derived from the backend being started, for
    # the same reason api_start does it -- never inherited from the previous run.
    d = s.desired
    backend = b.get("backend") or d.backend
    repo_id = b.get("repo_id") or d.repo_id
    settings: dict[str, Any] = {"repo_id": repo_id, "backend": backend}
    if b.get("backend") or b.get("port") is not None:
        settings["port"] = int(_sup.resolve_port(backend, d, explicit=b.get("port"), fallback=rt.port) or rt.port)
        settings["served_name"] = _sup.resolve_served_name(
            backend, repo_id, d, explicit=b.get("served_name")
        )
    if b.get("util") is not None:
        settings["util"] = float(b["util"])
    if b.get("ctx") is not None:
        settings["max_model_len"] = int(b["ctx"])
    if b.get("max_num_seqs") is not None:
        settings["max_num_seqs"] = int(b["max_num_seqs"])
    asyncio.create_task(_run_and_report(s.restart(mode=mode, **settings), f"restart:{mode}"))
    return JSONResponse({"accepted": True, "action": "restart", "mode": mode}, status_code=202)


async def _run_and_report(coro: Any, label: str) -> None:
    """Await a supervisor action, reporting the outcome on the event stream.

    Without this, a failure inside a fire-and-forget task is swallowed and the
    UI simply never changes state.
    """
    try:
        await coro
    except Exception as exc:  # noqa: BLE001
        hub.publish(
            "notice",
            {"level": "error", "code": f"{label}_failed", "body": f"{type(exc).__name__}: {exc}"},
        )
    else:
        # The outcome is the supervisor's state afterwards, not the mere fact
        # that the coroutine returned. Every control action used to report
        # '<action> completed' at level info -- including a start PORT_IN_USE
        # had just refused, a start that did nothing because the server was
        # already up, and the loser of two concurrent starts. An action that
        # failed must not read like one that worked.
        hub.publish("notice", _outcome_notice(label))
    finally:
        hub.publish("state", _state())


#: Supervisor states in which an action is still in flight -- "start
#: completed" then means "the boot is under way", not "the server is up".
_IN_FLIGHT_STATES = ("PREFLIGHT", "STARTING", "STOPPING", "DRAINING")


def _outcome_notice(label: str) -> dict[str, Any]:
    """The notice for an action that returned, derived from what it achieved."""
    s = sup()
    state = getattr(s, "actual_state", None)
    error = getattr(s, "last_error", None)
    if state == "FAILED":
        return {"level": "error", "code": f"{label}_failed",
                "body": f"{label} failed: {error or 'no reason recorded'}"}
    if state == "UNMANAGED":
        return {"level": "warn", "code": f"{label}_unmanaged",
                "body": error or f"{label}: a server is running that Servedeck does not manage"}
    if state in _IN_FLIGHT_STATES:
        return {"level": "info", "code": label, "body": f"{label} accepted — {state.lower()}"}
    return {"level": "info", "code": label, "body": f"{label} completed"}


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
                        "message": f"Servedeck: upstream {rt.upstream} unreachable ({type(exc).__name__})",
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
            # no-store on the dashboard's own assets. These change whenever the
            # app is updated, and a browser holding a stale app.js reports bugs
            # that were already fixed -- with the old error text, which sends
            # everyone looking in the wrong place. This is a localhost tool;
            # there is nothing to gain from caching them.
            return FileResponse(
                candidate,
                headers={"Cache-Control": "no-store, must-revalidate"},
            )   # FileResponse infers the media type
        # A request with a file extension wants a FILE. Returning index.html
        # with content-type text/html for a missing .css/.js is worse than a
        # 404: the browser drops it silently and the page renders unstyled with
        # a 200 in the network tab. Fail loudly instead.
        if "." in Path(rel).name:
            return JSONResponse({"error": f"not found: {rel}"}, status_code=404)

    return FileResponse(
        WEB / "index.html", headers={"Cache-Control": "no-store, must-revalidate"}
    )

# (no StaticFiles mount: the catch-all above is registered first and would
# shadow it. Assets are served there, including the /assets/ prefix strip.)
