"""The HTTP surface: one gateway, one control API, one page
(REDESIGN-2026-09-12.md §2.3, §2.5; P4).

This is a rewrite, not an edit. The file it replaces was 1,865 lines that
re-derived everything on every request: four ``/proc`` argv scans per
``/api/state``, a hand-rolled boot-phase machine, a 240-second request park, a
``catch_all`` that buffered every request body including megabyte images, and
a lifespan that started a model *before uvicorn had bound the port*. That last
one is not a stylistic complaint: on 2026-09-11 it launched the 27B seventy
times in nine minutes (REDESIGN §4 R4), because uvicorn runs the ASGI lifespan
before ``bind()``, the bind then failed against a hand-started copy, systemd
restarted the unit, and the whole thing went round again — each lap leaving a
real vLLM boot behind it.

So the shape of this file is dictated by four rules:

1. **Reconcile only after our own port answers.**  ``_reconcile_after_bind``
   polls ``/api/health`` on our own listen socket and only then calls
   ``control.reconcile``. If the bind never succeeds, nothing is ever started.
   ``tests/test_app.py::test_reconcile_waits_for_the_listen_port`` is the
   proof, and it is written against a fake control that records ordering.

2. **Nothing in a request handler shells out or blocks.**  ``systemctl``,
   ``journalctl`` and ``nvidia-smi`` are subprocesses; a coroutine that waits
   on one stops the whole server, including the gateway that is streaming a
   model's tokens. Every one of them runs in a worker thread
   (``asyncio.to_thread``), and the request path reads a snapshot.

3. **Mutations answer immediately and report over SSE.**  A start is minutes
   long. The POST returns 202 with the action it accepted; progress arrives on
   ``/api/events`` as it happens. A refusal that can be decided from the
   snapshot (unknown key, slot busy, already live) is answered synchronously
   with 404/409 and a typed reason, so a client never has to watch a stream to
   learn that its request was rejected.

4. **Shutdown closes every subscriber.**  The old app hung on
   ``TimeoutStopSec`` at every stop, because its SSE generators waited forever
   on a queue nothing would ever fill again. Here the hub hands each one a
   sentinel and the generators return.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)

from servedeck import control as _control
from servedeck import desired as _desired
from servedeck import discovery as _discovery
from servedeck import doctor as _doctor
from servedeck import gateway as _gateway
from servedeck import gpu as _gpu
from servedeck import legacy_page as _legacy
from servedeck import metrics as _metrics
from servedeck import models as _models
from servedeck import parallelism as _parallelism
from servedeck import routes as _routes
from servedeck import settings as _settings
from servedeck import units as _units
from servedeck import wire as _wire

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
#: The page lives INSIDE the package so it ships in the wheel.
WEB = HERE / "web"

#: How often the poller rebuilds ``/api/state``. Two seconds is the metrics
#: window the throughput figures are differenced over; polling faster would
#: divide counter deltas by a dt small enough for scheduling jitter to show up
#: as throughput noise.
POLL_INTERVAL_S = 2.0

#: SSE keepalive. Any comment frame will do; 15 s is short enough that a proxy
#: with a 30 s idle timeout never closes a connection that is merely quiet.
KEEPALIVE_S = 15.0

#: Per-subscriber queue depth. Drop-oldest on overflow: a slow page loses
#: history rather than stalling the poller that is trying to publish.
QUEUE_MAXSIZE = 256

#: The small-request size the headroom panel plans against.
SMALL_REQUEST_TOKENS = 4096

#: How long ``_reconcile_after_bind`` waits for our own port before giving up.
#: Generous: a cold page cache can make uvicorn's first bind slow. If it
#: expires, reconcile does not run — refusing to start models is the correct
#: failure for a dashboard that could not start itself.
BIND_WAIT_TIMEOUT_S = 60.0
BIND_POLL_INTERVAL_S = 0.25


# ==========================================================================
# SSE hub
# ==========================================================================


class _Subscriber:
    """One open ``GET /api/events`` connection."""

    __slots__ = ("queue", "dropped")

    def __init__(self) -> None:
        self.queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(QUEUE_MAXSIZE)
        self.dropped = 0

    def offer(self, event: dict[str, Any] | None) -> None:
        """Never blocks and never raises. A full queue loses its OLDEST event,
        which is what SSE ordering expects: the page is behind, and the newest
        state is the one worth having."""
        while True:
            try:
                self.queue.put_nowait(event)
                return
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    self.queue.get_nowait()
                    self.dropped += 1


class Hub:
    """In-process pub/sub for ``/api/events``.

    Publishable from a worker thread — ``control.start`` runs in one and its
    progress callback fires there — by hopping to the hub's loop with
    ``call_soon_threadsafe``. ``asyncio.Queue`` is not thread-safe, and the
    failure that mistake produces is a lost event rather than an exception, so
    the hop is not optional.
    """

    def __init__(self) -> None:
        self._subs: set[_Subscriber] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.closed = False
        #: Every notice published this session, newest last, capped. The page's
        #: Events panel renders the last 50 of these on load, so a page opened
        #: after a failure still shows the failure.
        self.notices: list[dict[str, Any]] = []

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> _Subscriber:
        sub = _Subscriber()
        self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: _Subscriber) -> None:
        self._subs.discard(sub)

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)

    def publish(self, event_type: str, data: Any) -> None:
        """Fan ``data`` out to every subscriber. Safe from any thread."""
        event = {"type": event_type, "data": data, "ts": time.time()}
        if event_type == "notice":
            self.notices.append(event)
            del self.notices[:-200]
        loop = self._loop
        if loop is None:
            self._fanout(event)
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._fanout(event)
        else:
            with contextlib.suppress(RuntimeError):  # loop already closed
                loop.call_soon_threadsafe(self._fanout, event)

    def _fanout(self, event: dict[str, Any] | None) -> None:
        for sub in list(self._subs):
            sub.offer(event)

    def close(self) -> None:
        """Hand every subscriber the sentinel so its generator returns.

        This is the whole of the ``TimeoutStopSec`` fix. Without it each open
        page holds a coroutine parked on ``queue.get()`` that nothing will ever
        complete, uvicorn waits for them on shutdown, and systemd SIGKILLs the
        unit 15 seconds later — every single stop.
        """
        self.closed = True
        self._fanout(None)


# ==========================================================================
# Runtime
# ==========================================================================


@dataclass
class Runtime:
    """Everything a request handler needs, assembled once by the lifespan."""

    settings: _settings.Settings
    registry: _models.Registry
    control: Any
    routes: _routes.RegistryRoutes
    hub: Hub
    client: httpx.AsyncClient
    #: key -> ModelSpecAdapter, built once at load. Empty when a test injects
    #: its own control and never needed the adapters.
    specs: dict[str, Any] = field(default_factory=dict)
    #: key -> MetricsPoller. One per model, kept across polls because the
    #: throughput figures are counter deltas and a fresh poller has no baseline.
    pollers: dict[str, _metrics.MetricsPoller] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    #: (unit, pid) -> unit start time (epoch seconds). systemd's
    #: ExecMainStartTimestamp is one more `systemctl show` per unit per poll
    #: and it cannot change while the pid does not, so it is asked for once.
    _uptime_cache: dict[tuple[str, int], float] = field(default_factory=dict)
    #: Serialises mutations. Two concurrent switches would each stop what the
    #: other started; one lock makes "the main slot is exclusive" true of the
    #: API and not only of the GPU.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    #: The mutation in flight, as ``{"action", "key", "label"}``, or None.
    #: A dict rather than the label string alone: the page needs to know WHICH
    #: model is booting so it can put the progress line on the right row, and
    #: parsing that back out of "start flashnext" would be a client reading
    #: English.
    busy: dict[str, str] | None = None
    first_state: asyncio.Event = field(default_factory=asyncio.Event)
    #: A nonce this process invents at startup and echoes from /api/health.
    #: ``_reconcile_after_bind`` requires it back before it will start a model.
    #:
    #: A 200 alone does not prove we won the port: on 2026-09-11 a
    #: hand-started copy of servedeck held :8010, and it answers /api/health
    #: with 200 too. Believing that answer is precisely what turned one lost
    #: bind into 70 real vLLM launches (REDESIGN §4 R4) — the losing process
    #: concluded it was listening and reconciled. The nonce makes "is that me"
    #: answerable instead of assumed.
    instance_id: str = field(default_factory=lambda: secrets.token_hex(8))
    #: key -> (attempts_made, last_attempt_monotonic). What the poll's
    #: desired-vs-live check has already tried, so a model that cannot boot is
    #: retried a few times and then left alone with a notice, rather than
    #: relaunched every two seconds forever.
    recovery: dict[str, tuple[int, float]] = field(default_factory=dict)
    #: Boot progress folded into the phase bar the page draws (legacy_page).
    boot: _legacy.BootTracker = field(default_factory=_legacy.BootTracker)

    def poller_for(self, key: str, port: int) -> _metrics.MetricsPoller:
        base = f"http://127.0.0.1:{port}"
        existing = self.pollers.get(key)
        if existing is None or existing.base_url != base:
            existing = _metrics.MetricsPoller(base)
            self.pollers[key] = existing
        return existing


# --------------------------------------------------------------------------
# Blocking helpers — every one of these runs in a worker thread
# --------------------------------------------------------------------------


def _unit_started_at(rt: Runtime, unit: str, pid: int) -> float | None:
    """Epoch seconds the unit's main process started, or None.

    ``ExecMainStartTimestamp`` is a localised human string
    (``Fri 2026-09-12 13:49:02 IST``); the numeric companion is asked for
    instead, because parsing a timezone abbreviation is a locale bug waiting
    to happen and systemd already publishes the microseconds.
    """
    if pid <= 0:
        return None
    hit = rt._uptime_cache.get((unit, pid))
    if hit is not None:
        return hit
    try:
        props = _units.properties(unit, ("ExecMainStartTimestampMonotonic",))
    except _units.UnitError:
        return None
    raw = props.get("ExecMainStartTimestampMonotonic", "")
    try:
        monotonic_us = int(raw)
    except ValueError:
        return None
    if monotonic_us <= 0:
        return None
    # systemd's monotonic clock is CLOCK_MONOTONIC, the same one
    # time.monotonic() reads, so the conversion needs no boot time.
    started = time.time() - (time.monotonic() - monotonic_us / 1_000_000)
    rt._uptime_cache[(unit, pid)] = started
    return started


def _collect_facts(rt: Runtime) -> dict[str, Any]:
    """The blocking half of a state build: systemd, nvidia-smi, desired state."""
    # A crashed unit is restarted by systemd (Restart=on-failure) without
    # passing through control.start, so the reaper also runs here, every poll:
    # a buffer the dead engine left in /dev/shm is gone seconds later, long
    # before the restarted engine allocates its own.
    reap = getattr(rt.control, "reap_offload", None)
    if callable(reap):
        try:
            for path, size in reap():
                rt.hub.publish("notice", {
                    "level": "warn", "reason": "offload_reaped",
                    "message": f"freed {size / 2**30:.1f} GiB of host RAM: {path} was left "
                               "behind by a vLLM engine that did not shut down cleanly",
                })
        except Exception:  # noqa: BLE001 - the reaper must never break the poll
            log.exception("offload reaper failed")
    live = rt.routes.refresh()
    uptimes: dict[str, float | None] = {}
    for key, view in live.items():
        started = _unit_started_at(rt, view.unit, view.pid)
        uptimes[key] = None if started is None else max(0.0, time.time() - started)
    return {
        "live": live,
        "uptimes": uptimes,
        "gpu": {"total_mib": _gpu.total_mib(), "free_mib": _gpu.free_mib()},
        "desired": _desired.load(rt.settings.desired_path),
    }


def _headroom(
    *,
    main_key: str | None,
    main_id: str | None,
    snapshot: dict[str, Any] | None,
    full_ctx: int,
    free_mib: int | None,
) -> dict[str, Any]:
    """How much more work the running engine can take on.

    The pool is ``vllm:cache_config_info``'s ``kv_cache_size_tokens`` — the
    engine's own resolved KV capacity, the same number its boot log prints as
    "GPU KV cache size". That is a measurement, and the panel says so. When it
    is absent the answer is "unknown" with a reason, never an estimate: an
    estimated capacity presented beside a measured one is indistinguishable
    from it on screen, and acting on the wrong one over-subscribes the engine.

    All the arithmetic is ``parallelism``'s, called here — the page does no
    capacity maths of its own (REDESIGN §2.5).
    """
    base: dict[str, Any] = {
        "free_mib": free_mib,
        "model": main_id,
        "key": main_key,
        "full_ctx": full_ctx or None,
        "pool_tokens": None,
        "full_context_requests": None,
        "small_request_tokens": SMALL_REQUEST_TOKENS,
        "small_requests": None,
        "fixed_cost_tokens": _parallelism.FIXED_COST_TOKENS,
        "headroom_fraction": _parallelism.HEADROOM,
        "source": None,
        "unavailable": None,
        "note": None,
        # Present on EVERY branch, null when unknown. A key that only appears
        # in the available case makes the document's shape depend on its
        # content: a client (or a contract test) reading the unavailable branch
        # concludes these are never sent, and a renderer written against the
        # available branch throws on the other one.
        "full_cost_tokens": None,
        "small_cost_tokens": None,
    }
    if main_key is None:
        base["unavailable"] = "no model holds the main slot"
        return base
    pool = (snapshot or {}).get("kv_cache_size_tokens")
    if not pool:
        base["unavailable"] = (
            "the engine has not reported vllm:cache_config_info yet, so its KV "
            "pool size is unknown"
        )
        return base
    if not full_ctx:
        base["unavailable"] = (
            "the model's context length is unknown (ctx = \"native\" and the "
            "checkpoint is not in the local hub cache)"
        )
        return base
    full = _parallelism.recommend(
        pool_tokens=int(pool), prompt_tokens=full_ctx, basis="full context"
    )
    small = _parallelism.recommend(
        pool_tokens=int(pool),
        prompt_tokens=SMALL_REQUEST_TOKENS,
        basis=f"{SMALL_REQUEST_TOKENS}-token request",
    )
    base.update(
        pool_tokens=int(pool),
        full_context_requests=full.n_before_clamp,
        full_cost_tokens=full.cost_tokens,
        small_requests=small.n_before_clamp,
        small_cost_tokens=small.cost_tokens,
        source="measured from the running engine",
        note=_parallelism.calibration_note(main_id),
    )
    return base


async def build_state(rt: Runtime) -> dict[str, Any]:
    """The one JSON document the page renders from.

    Every field is either a registry fact, a systemd fact, or a number the
    running engine published about itself. Nothing here is estimated, and
    nothing is computed twice: ``/api/state`` is the only shape, and the page
    has no second source to disagree with it.
    """
    facts = await asyncio.to_thread(_collect_facts, rt)
    live: dict[str, _routes.LiveView] = facts["live"]
    desired: _desired.Desired = facts["desired"]

    model_rows: list[dict[str, Any]] = []
    main_key: str | None = None
    main_id: str | None = None
    main_ctx = 0
    main_snapshot: dict[str, Any] | None = None

    for key, model in rt.registry.models.items():
        view = live.get(key)
        ctx = rt.routes.ctx_for(model)
        row: dict[str, Any] = {
            "key": key,
            "id": model.id,
            "aliases": list(model.aliases),
            "presets": list(model.presets),
            "slot": model.slot,
            "port": model.port,
            "ctx": ctx or None,
            "ctx_error": rt.routes.ctx_error(key),
            "build": model.build,
            "repo": model.repo,
            "vram_mib": model.vram_mib,
            "needs_tty": model.needs_tty,
            "live": view is not None,
            "ready": bool(view and view.ready),
            "unit": view.unit if view else f"{rt.settings.unit_prefix}{key}",
            "unit_state": view.unit_state if view else "not started",
            "restarts": view.restarts if view else 0,
            "pid": view.pid if view else 0,
            "uptime_s": facts["uptimes"].get(key),
            "metrics": None,
        }
        if view is not None and view.ready:
            poller = rt.poller_for(key, model.port)
            # The request-size histogram's open-ended top bucket is drawn up
            # to the engine's own --max-model-len; without a ceiling the page
            # cannot place a 129k-token request. v1 set this every poll too.
            poller.ceiling_tokens = ctx or None
            snapshot = await poller.scrape(rt.client)
            payload = snapshot.to_dict()
            row["metrics"] = {
                "reachable": payload["reachable"],
                "running": payload["running"],
                "waiting": payload["waiting"],
                "kv_usage_perc": payload["kv_usage_perc"],
                "gen_tok_s": payload["gen_tok_s"],
                "gen_tok_s_avg": payload["gen_tok_s_avg"],
                "gen_state": payload["gen_state"],
                "kv_cache_size_tokens": payload["kv_cache_size_tokens"],
                "error": payload["error"],
            }
            if model.slot == "main":
                main_snapshot = payload
        if model.slot == "main" and view is not None:
            main_key, main_id, main_ctx = key, model.id, ctx
        model_rows.append(row)

    unknown_units = [
        {"key": view.key, "unit": view.unit, "unit_state": view.unit_state}
        for view in live.values()
        if view.unknown
    ]

    state = {
        "generated_at": time.time(),
        "gateway_url": rt.settings.gateway_url,
        "unit_prefix": rt.settings.unit_prefix,
        "gpu": facts["gpu"],
        "desired": {"main": desired.main, "residents": list(desired.residents)},
        "busy": rt.busy,
        "models": model_rows,
        "unknown_units": unknown_units,
        "headroom": _headroom(
            main_key=main_key,
            main_id=main_id,
            snapshot=main_snapshot,
            full_ctx=main_ctx,
            free_mib=facts["gpu"]["free_mib"],
        ),
    }
    # The v1 page's blocks, ADDED to the document (legacy_page): upstream,
    # vllm, sizing, boot, supervisor, config, gpu_v1, uptimes.
    try:
        state.update(
            await asyncio.to_thread(
                _legacy.augment_state,
                registry=rt.registry,
                state=state,
                boot=rt.boot,
                busy=rt.busy,
                main_key=main_key,
                main_ctx=main_ctx,
                main_snapshot=main_snapshot,
                desired_main=desired.main,
                uptimes=facts["uptimes"],
            )
        )
    except Exception:  # noqa: BLE001 - the v1 blocks must not take the v2 state down
        log.exception("legacy state blocks failed")
    return state


def _state_changed(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """Is this state worth an SSE frame?

    ``generated_at`` and the throughput figures change on every poll; pushing
    on those would make "on change" mean "every two seconds" and the page would
    re-render continuously. Compare the document with the perpetually-moving
    parts removed.
    """
    return _comparable(old) != _comparable(new)


_VOLATILE_METRIC_KEYS = ("gen_tok_s", "gen_tok_s_avg", "kv_usage_perc", "gen_state")


def _comparable(doc: dict[str, Any]) -> str:
    stripped = dict(doc)
    stripped.pop("generated_at", None)
    rows = []
    for row in stripped.get("models") or []:
        row = dict(row)
        row.pop("uptime_s", None)
        met = row.get("metrics")
        if isinstance(met, dict):
            row["metrics"] = {k: v for k, v in met.items() if k not in _VOLATILE_METRIC_KEYS}
        rows.append(row)
    stripped["models"] = rows
    return json.dumps(stripped, sort_keys=True, default=str)


# ==========================================================================
# Background tasks
# ==========================================================================


async def _poll_loop(rt: Runtime) -> None:
    """Rebuild ``/api/state`` on a timer and publish it when it changes."""
    while True:
        try:
            state = await build_state(rt)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a bad poll must not end the poller
            log.exception("state poll failed")
            await asyncio.sleep(POLL_INTERVAL_S)
            continue
        changed = _state_changed(rt.state, state)
        rt.state = state
        rt.first_state.set()
        if changed:
            rt.hub.publish("state", state)
        # v1's page paints the live strip from this every poll, changed or not.
        rt.hub.publish("telemetry", _legacy.telemetry_payload(state))
        _recover_desired(rt)
        await asyncio.sleep(POLL_INTERVAL_S)


#: How many times the poll will relaunch a model that is wanted but absent
#: before it stops and leaves the notice standing. Three, because the failure
#: this exists for is a transient one (an Xid fault, a driver hiccup); a model
#: that cannot boot at all must not be launched every two seconds forever.
RECOVERY_ATTEMPTS = 3
#: Minimum gap between those attempts.
RECOVERY_BACKOFF_S = 60.0


def _recover_desired(rt: Runtime) -> None:
    """Relaunch a model that desired state names but nothing is running.

    This is the gap systemd cannot close. When vLLM's engine dies the process
    exits 0, and even with ``Restart=always`` systemd gives up after
    ``StartLimitBurst`` attempts — at which point ``--collect`` removes the
    unit, so ``systemctl show`` reports an unknown unit's defaults and a
    crashed model becomes indistinguishable from one nobody wanted. Reconcile
    runs once, at startup, so nothing looked again: on this card, which has a
    documented Xid history, one persistent fault meant the box served nothing
    until somebody noticed.
    """
    if rt.busy is not None:  # a mutation is already in flight
        return
    try:
        want = _desired.load(rt.settings.desired_path)
    except Exception:  # noqa: BLE001 - never break the poll over this
        log.exception("desired state could not be read for the recovery check")
        return
    live = rt.routes.live()
    now = time.monotonic()
    for key in [*want.residents, *([want.main] if want.main else [])]:
        if key in live:
            rt.recovery.pop(key, None)  # it is up: forget the attempts
            continue
        if key not in rt.registry.models:
            continue
        attempts, last = rt.recovery.get(key, (0, 0.0))
        if attempts >= RECOVERY_ATTEMPTS:
            continue
        if last and now - last < RECOVERY_BACKOFF_S:
            continue
        rt.recovery[key] = (attempts + 1, now)
        left = RECOVERY_ATTEMPTS - attempts - 1
        log.warning(
            "%s is wanted but not running; relaunching (attempt %d of %d)",
            key, attempts + 1, RECOVERY_ATTEMPTS,
        )
        rt.hub.publish("notice", {
            "level": "warn", "reason": "recovering", "key": key,
            "message": (
                f"{key} is wanted but nothing is running it — its engine died or its unit "
                f"was collected. Relaunching (attempt {attempts + 1} of {RECOVERY_ATTEMPTS}"
                + (f", {left} left after this)" if left else ", the last one)")
            ),
        })
        _claim(rt, "start", key, f"recover {key}")
        asyncio.create_task(
            _run_mutation(
                rt,
                f"recover {key}",
                lambda k=key: rt.control.start(k, on_progress=_progress_publisher(rt.hub, k, rt.boot)),
                action="start",
                key=key,
            )
        )
        return  # one at a time; the next poll takes the next model


async def _own_port_answers(client: httpx.AsyncClient, url: str, instance_id: str) -> bool:
    """Is the thing answering ``url`` THIS process?

    Three distinct answers collapse into False, and all three mean "do not
    start a model yet":

    * nothing is listening (the normal case, for the first few hundred ms);
    * something is listening but is not a servedeck (a 404, a 503);
    * something is listening and IS a servedeck — just not us.

    The third is the one that mattered: a hand-started copy held :8010 on
    2026-09-11 and answers ``/api/health`` with a perfectly good 200. A check
    on the status code alone cannot see the difference, so the process that
    LOST the bind reconciled anyway and launched the 27B on every restart.
    """
    try:
        response = await client.get(url, timeout=2.0)
    except Exception:  # noqa: BLE001 - not listening yet is the expected case
        return False
    if response.status_code != 200:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    return isinstance(payload, dict) and payload.get("instance") == instance_id


async def _reconcile_after_bind(
    rt: Runtime,
    *,
    timeout_s: float = BIND_WAIT_TIMEOUT_S,
    interval_s: float = BIND_POLL_INTERVAL_S,
) -> Any:
    """Wait for **our own** listen port to answer, then reconcile.

    The rule this enforces, stated as a sequence: no model is started until
    servedeck has demonstrably won the race for :8010. uvicorn runs the ASGI
    lifespan *before* it binds, so a reconcile called from the lifespan runs
    even in the process that is about to exit with "address already in use" —
    and under ``Restart=on-failure`` that turns one lost race into an
    unbounded boot loop with a real vLLM launch on every lap (REDESIGN §4 R4,
    measured: 70 launches in 9 minutes on 2026-09-11).

    Returns the Reconciliation, or None if the port never answered.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if await _own_port_answers(rt.client, rt.settings.health_url, rt.instance_id):
            break
        await asyncio.sleep(interval_s)
    else:
        rt.hub.publish(
            "notice",
            {
                "level": "error",
                "reason": "bind_timeout",
                "message": (
                    f"{rt.settings.base_url} did not answer as THIS process within "
                    f"{timeout_s:.0f}s; not reconciling. Either the bind failed, or "
                    "another servedeck is holding the port."
                ),
            },
        )
        return None

    want = await asyncio.to_thread(_desired.load, rt.settings.desired_path)
    rt.hub.publish(
        "notice",
        {
            "level": "info",
            "reason": "reconciling",
            "message": f"listening on {rt.settings.base_url}; reconciling desired state",
            "desired": {"main": want.main, "residents": list(want.residents)},
        },
    )
    result = await asyncio.to_thread(
        lambda: rt.control.reconcile(want, on_progress=_progress_publisher(rt.hub, "reconcile", rt.boot))
    )
    _publish_reconciliation(rt.hub, result)
    return result


def _progress_publisher(
    hub: Hub, key: str, boot: _legacy.BootTracker | None = None
) -> Callable[[Any], None]:
    """A ``control.ProgressCallback`` that turns each event into an SSE frame.

    Called from the worker thread ``control.start`` runs in, which is why
    ``Hub.publish`` hops to the loop rather than touching a queue directly.
    """

    def on_progress(event: Any) -> None:
        if boot is not None:
            boot.record(key, event.kind, event.marker_index, event.elapsed_s)
        hub.publish(
            "progress",
            {
                "key": key,
                "kind": event.kind,
                "text": event.text,
                "marker_index": event.marker_index,
                "elapsed_s": round(event.elapsed_s, 1),
            },
        )

    return on_progress


def _publish_reconciliation(hub: Hub, result: Any) -> None:
    if result is None:
        return
    hub.publish(
        "notice",
        {
            "level": "info",
            "reason": "reconciled",
            "message": (
                f"reconcile: {len(result.already_live)} already live, "
                f"{len(result.booting)} booting, {len(result.started)} started, "
                f"{len(result.refused)} refused"
            ),
            "already_live": list(result.already_live),
            "booting": list(result.booting),
            "started": [r.key for r in result.started],
            "refused": [{"key": r.key, "reason": r.reason, "message": r.message} for r in result.refused],
        },
    )


# ==========================================================================
# Refusals
# ==========================================================================


def _refusal(status: int, reason: str, message: str, **extra: Any) -> JSONResponse:
    """The one refusal shape. ``reason`` is the machine-readable half and is
    the same vocabulary ``control.Refusal`` uses, so a refusal decided here
    from the snapshot and one decided there against reality are
    indistinguishable to a client."""
    return JSONResponse({"error": {"reason": reason, "message": message, **extra}}, status_code=status)


def _accepted(action: str, key: str, **extra: Any) -> JSONResponse:
    """202. Names the action and the model, so a caller that fires and forgets
    still has something to correlate the SSE frames against."""
    return JSONResponse({"accepted": True, "action": action, "model": key, **extra}, status_code=202)


def _precheck(rt: Runtime, key: str, action: str) -> JSONResponse | None:
    """Answer from the snapshot whatever can be answered without a subprocess.

    Everything here is also re-checked inside ``Control`` against reality; this
    exists so the common refusals come back on the POST rather than as a notice
    the caller has to be watching a stream to see.
    """
    model = rt.registry.models.get(key)
    if model is None:
        return _refusal(
            404,
            "unknown_model",
            f"no model {key!r} in the registry",
            known=list(rt.registry.models),
        )
    live = rt.routes.live()
    if rt.busy is not None:
        return _refusal(
            409,
            "busy",
            f"servedeck is already running {rt.busy['label']}",
            busy=rt.busy,
        )
    if action in ("start", "switch") and key in live:
        return _refusal(
            409,
            "already_live",
            f"{live[key].unit} already exists ({live[key].unit_state}); stop it first",
            live_key=key,
        )
    if action == "start" and model.slot == "main":
        holder = _main_holder(rt, live)
        if holder is not None:
            return _refusal(
                409,
                "main_slot_busy",
                f"the main slot is held by {holder} — use POST /api/switch/{key}",
                live_key=holder,
            )
    if action == "switch" and model.slot != "main":
        return _refusal(
            409,
            "not_main_slot",
            f"{key} is a {model.slot} model; switch only replaces the main slot",
        )
    if action == "stop" and key not in live:
        return _refusal(409, "not_live", f"no unit exists for {key}", live_key=None)
    if action in ("start", "switch"):
        spec = rt.specs.get(key)
        if spec is not None and not spec.ctx_tokens:
            # `ctx = "native"` that could not be read. Caught here so the
            # operator gets a sentence naming the checkpoint, instead of a
            # RegistryError raised inside a background thread and surfacing as
            # a notice they have to be watching a stream to see.
            return _refusal(
                409,
                "ctx_unresolved",
                spec.ctx_error or f"{key} has no known context length",
            )
    return None


def _main_holder(rt: Runtime, live: dict[str, _routes.LiveView]) -> str | None:
    for key, model in rt.registry.models.items():
        if model.slot == "main" and key in live:
            return key
    return None


#: Methods that change something. GET/HEAD/OPTIONS are readable by anyone who
#: can reach the port, which on loopback is the operator.
_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


async def _same_origin_only(request: Request, call_next: Callable[[Request], Any]) -> Any:
    """Refuse a cross-origin write to the control API.

    Every mutation route takes its key in the PATH and no body, so
    ``POST /api/models/flashnext/stop`` is a CORS "simple request": any page
    in the operator's browser — any tab, any ad frame — can send it with
    ``fetch(..., {mode: "no-cors"})``, get an opaque response back, and stop
    the model, switch the card or rewrite every client's config, with nothing
    in the journal but an access line. There is no API key and no CORS
    middleware, so the only thing distinguishing "the dashboard" from "some
    web page" is the Origin header the browser attaches to both.

    Non-browser callers (the CLI, curl, a script) send no Origin at all and
    are unaffected; the gateway under ``/v1`` is deliberately NOT covered,
    because a local tool calling the OpenAI API from a browser page is a use
    case, not an attack, and it cannot change anything.
    """
    origin = request.headers.get("origin")
    if (
        origin
        and request.method in _UNSAFE_METHODS
        and request.url.path.startswith("/api/")
        and not _origin_is_ours(request, origin)
    ):
        return JSONResponse(
            {
                "error": "cross_origin",
                "message": (
                    f"refusing a {request.method} from {origin}: the servedeck control API "
                    "is only writable from its own page or from a terminal"
                ),
            },
            status_code=403,
        )
    return await call_next(request)


def _origin_is_ours(request: Request, origin: str) -> bool:
    """True when ``origin`` names this server's own host:port.

    Both loopback spellings count: the page is served at 127.0.0.1 but VS Code
    and the operator's bookmarks reach it as localhost, and a browser sends
    whichever one is in the address bar.
    """
    from urllib.parse import urlsplit

    parts = urlsplit(origin)
    if parts.scheme not in ("http", "https"):
        return False
    port = parts.port or (443 if parts.scheme == "https" else 80)
    # The configured listen port, not only the request URL's: behind a test
    # client (and behind any proxy) the request URL carries no port at all,
    # and a guard that then rejected the page's own origin would break the
    # dashboard instead of protecting it.
    own_ports = {request.url.port or 0}
    rt = getattr(request.app.state, "rt", None)
    if rt is not None:
        own_ports.add(rt.settings.listen_port)
    return parts.hostname in ("127.0.0.1", "::1", "localhost") and port in own_ports


def _claim(rt: Runtime, action: str, key: str, label: str) -> None:
    """Mark servedeck busy in the handler, before the work is scheduled.

    ``_run_mutation`` used to set ``busy`` itself, which happens only once the
    event loop gets round to the task — so two Applies arriving in the same
    tick both passed ``_precheck`` and both ran, restarting the model twice.
    A handler calls this immediately after a clean precheck, with no ``await``
    in between, which is what makes the check-then-claim atomic.
    """
    rt.busy = {"action": action, "key": key, "label": label}


async def _run_mutation(
    rt: Runtime, label: str, work: Callable[[], Any], *, action: str = "", key: str = ""
) -> None:
    """Run one blocking Control call in a thread, reporting whatever it says.

    Holds ``rt.lock`` for the whole operation and advertises it as ``busy`` in
    ``/api/state``, so a second start cannot interleave with a switch that is
    between its stop and its start — the window in which the card is empty and
    every precheck would say "go ahead".

    ``busy`` is cleared only AFTER the outcome has been published. Clearing it
    first (in a ``finally`` before the publish) left a tick in which
    ``/api/state`` said nothing was running while the notice stream was still
    about to announce the result — two views of the same instant disagreeing,
    which is the shape of every bug this rewrite exists to remove.
    """
    async with rt.lock:
        rt.busy = {"action": action or label, "key": key, "label": label}
        rt.hub.publish(
            "notice",
            {"level": "info", "reason": "started", "message": f"{label}: running", "key": key or None},
        )
        try:
            result = await asyncio.to_thread(work)
        except Exception as exc:  # noqa: BLE001 - a failed mutation is a notice
            log.exception("%s failed", label)
            rt.hub.publish(
                "notice",
                {
                    "level": "error",
                    "reason": "exception",
                    "message": f"{label}: {type(exc).__name__}: {exc}",
                    "key": key or None,
                },
            )
            return
        else:
            _publish_result(rt.hub, label, result)
        finally:
            # `else` runs before `finally`, and the except branch publishes
            # before returning, so every exit path has announced its outcome by
            # the time `busy` clears. That ordering is the whole fix.
            rt.busy = None
    with contextlib.suppress(Exception):
        state = await build_state(rt)
        rt.state = state
        rt.hub.publish("state", state)


def _publish_result(hub: Hub, label: str, result: Any) -> None:
    if isinstance(result, _control.Refusal):
        hub.publish(
            "notice",
            {
                "level": "error",
                "reason": result.reason,
                "message": f"{label}: {result.message}",
                "key": result.key,
                "live_key": result.live_key,
            },
        )
        return
    if isinstance(result, _control.StartResult):
        hub.publish(
            "notice",
            {
                "level": "info" if result.ready else "error",
                "reason": "ready" if result.ready else "boot_failed",
                "message": (
                    f"{label}: {result.key} ready in {result.elapsed_s:.0f}s"
                    if result.ready
                    else f"{label}: {result.failure}"
                ),
                "key": result.key,
                "markers": list(result.markers),
                "journal": list(result.journal),
            },
        )
        return
    if isinstance(result, _control.StopResult):
        hub.publish(
            "notice",
            {
                "level": "info",
                "reason": "stopped",
                "message": (
                    f"{label}: {result.unit} stopped"
                    + (f" (released ~{result.held_mib} MiB)" if result.held_mib else "")
                ),
                "key": result.key,
            },
        )
        return
    if isinstance(result, _control.SwitchResult):
        _publish_result(hub, label, result.started)
        return
    if isinstance(result, _control.Adoption):
        hub.publish(
            "notice",
            {
                "level": "info",
                "reason": "adopted",
                "message": (
                    f"{label}: adopted {result.adopted or 'nothing'}"
                    + (f"; unknown units {result.unknown_units}" if result.unknown_units else "")
                ),
                "adopted": list(result.adopted),
                "unknown_units": list(result.unknown_units),
            },
        )
        return
    if isinstance(result, _control.Reconciliation):
        _publish_reconciliation(hub, result)


# ==========================================================================
# App
# ==========================================================================


def build_control(
    settings: _settings.Settings,
    registry: _models.Registry,
    adapter: _RegistryAdapter | None = None,
) -> _control.Control:
    """A ``Control`` wired to this registry and this box's settings.

    Public because ``servedeck.cli`` needs exactly this when the dashboard is
    not running: the CLI's fallback drives the same supervisor in-process
    rather than a second, subtly different one.
    """
    return _control.Control(
        adapter or _RegistryAdapter(registry),
        margin_mib=registry.gpu.margin_mib,
        total_mib=registry.gpu.total_mib,
        unit_prefix=settings.unit_prefix,
        desired_path=settings.desired_path,
    )


def build_runtime(
    settings: _settings.Settings | None = None,
    *,
    registry: _models.Registry | None = None,
    control: Any | None = None,
    client: httpx.AsyncClient | None = None,
) -> Runtime:
    """Assemble a Runtime. Every dependency is injectable, which is how the API
    tests drive every route with a fake Control and no systemd."""
    settings = settings or _settings.get()
    registry = registry if registry is not None else _models.load(settings.models_path)
    adapter = _RegistryAdapter(registry)
    if control is None:
        control = build_control(settings, registry, adapter)
    # ONE context resolution for the whole process. The adapter already read
    # every `ctx = "native"` off disk at load; handing the same answers to the
    # route table means the gateway's `max_model_len`, the page's ctx column
    # and the `--max-model-len` a model is actually launched with cannot
    # disagree — which they could when each resolved its own.
    routes = _routes.RegistryRoutes(
        registry, control, ctx_resolver=lambda m: adapter.specs[m.key].ctx_tokens
    )
    return Runtime(
        settings=settings,
        registry=registry,
        specs=adapter.specs,
        control=control,
        routes=routes,
        hub=Hub(),
        client=client or httpx.AsyncClient(timeout=10.0),
    )


def _build_venv_bin(build: _models.Build | None) -> str:
    """``<venv>/bin`` for one ``[builds.<name>]`` table.

    ``~`` is expanded here rather than passed through: this string becomes
    argv[0]'s directory, and ``execve`` does not expand ``~`` — a literal one
    produces "No such file or directory" naming a path that plainly exists,
    which is among the least legible failures available. ``render_env`` expands
    the same two paths for ``PATH``/``CUDA_HOME``, for the same reason.
    """
    if build is None:
        return ""
    root = Path(build.venv).expanduser()
    return str(root if root.name == "bin" else root / "bin")


@dataclass(frozen=True)
class ModelSpecAdapter:
    """One ``models.Model`` as a ``control.ModelSpec``.

    Built ONCE, at load, for every model — not per call. ``ctx_tokens`` is the
    reason: resolving ``ctx = "native"`` reads the checkpoint's config.json off
    disk, and a property would do that on every ``control.live()`` probe.
    Resolving it here also means the value is settled before anything can try
    to launch with it.

    ``ctx_tokens`` is 0, and only 0, when ``native`` could not be read.
    :meth:`render_argv` then refuses rather than launching a model with a
    context length nobody measured — and ``app._precheck`` catches that case
    before a POST is ever accepted, so the refusal reaches the operator as a
    409 instead of as a traceback in a background task.
    """

    model: _models.Model
    #: The resolved ``[builds.<name>]`` table this model launches from. Carried
    #: rather than looked up per call because ``render_env`` needs it on every
    #: launch and a registry lookup inside the Protocol's methods would put the
    #: registry back into the supervisor's dependency set.
    build: _models.Build | None
    key: str
    id: str
    slot: str
    port: int
    vram_mib: int | None
    ctx_tokens: int
    venv_bin: str
    ctx_error: str | None = None

    @classmethod
    def from_model(cls, model: _models.Model, registry: _models.Registry) -> ModelSpecAdapter:
        """Resolve one registry model into a launchable spec.

        NOT named ``build``: that is the name of the field above, and a
        classmethod sharing a field's name becomes that field's default as far
        as ``dataclasses`` is concerned — every field after it then raises
        "non-default argument follows default argument" at import time.
        """
        ctx, error = 0, None
        if isinstance(model.ctx, int):
            ctx = model.ctx
        else:
            try:
                ctx = _models.native_ctx(model.repo)
            except _models.RegistryError as exc:
                error = str(exc)
        build = registry.builds.get(model.build)
        return cls(
            model=model,
            build=build,
            key=model.key,
            id=model.id,
            slot=model.slot,
            port=model.port,
            vram_mib=model.vram_mib,
            ctx_tokens=ctx,
            venv_bin=_build_venv_bin(build),
            ctx_error=error,
        )

    def served_names(self) -> list[str]:
        """A method, matching ``models.Model.served_names`` — one shape for
        "every name this model answers to", in the registry and in the
        supervisor's Protocol both."""
        return self.model.served_names()

    def render_argv(self, util: float, port: int) -> list[str]:
        if not self.ctx_tokens:
            raise _models.RegistryError(
                f"{self.key}: {self.ctx_error or 'no context length is known'}"
            )
        if self.build is None or not self.venv_bin:
            raise _models.RegistryError(
                f"{self.key}: build {self.model.build!r} is not in [builds], so "
                "there is no venv to launch from"
            )
        return _models.render_argv(
            self.model, str(Path(self.venv_bin) / "vllm"), util, self.ctx_tokens, port
        )

    def render_env(self) -> dict[str, str]:
        """The COMPLETE launch environment, including ``CUDA_HOME`` and a full
        ``PATH`` built from this model's build.

        A transient unit inherits the USER MANAGER's environment, never the
        caller's shell, so anything missing here is missing at boot — and
        FlashInfer's JIT needs nvcc/ptxas on ``PATH`` at *request* time, not
        only at process start.
        """
        if self.build is None:
            raise _models.RegistryError(
                f"{self.key}: build {self.model.build!r} is not in [builds], so "
                "CUDA_HOME and PATH cannot be derived"
            )
        return _models.render_env(self.model, self.build)


class _RegistryAdapter:
    """P1's ``Registry`` as P3's ``control.Registry`` Protocol.

    ``get`` for every keyed action, ``keys`` for port adoption only (see
    ``control.Registry``). There is still no ``models()``: nothing decides
    order here.
    """

    def __init__(self, registry: _models.Registry) -> None:
        self.registry = registry
        self.specs: dict[str, ModelSpecAdapter] = {
            key: ModelSpecAdapter.from_model(model, registry)
            for key, model in registry.models.items()
        }

    def get(self, key: str) -> ModelSpecAdapter:
        spec = self.specs.get(key)
        if spec is None:
            raise KeyError(key)
        return spec

    def keys(self) -> list[str]:
        return list(self.specs)


def create_app(
    settings: _settings.Settings | None = None,
    *,
    registry: _models.Registry | None = None,
    control: Any | None = None,
    client: httpx.AsyncClient | None = None,
    gateway_transport: httpx.AsyncBaseTransport | None = None,
    reconcile: bool = True,
    poll: bool = True,
) -> FastAPI:
    """Build the ASGI app.

    ``reconcile``/``poll`` default on and are turned off by tests that want the
    routes without the background machinery. They are parameters rather than
    environment variables because a test that has to set an env var to stop the
    app launching a model is one forgotten fixture away from launching one.
    """
    rt = build_runtime(settings, registry=registry, control=control, client=client)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        rt.hub.bind(asyncio.get_running_loop())
        tasks: list[asyncio.Task[Any]] = []
        # One synchronous refresh before anything serves, so the first request
        # sees the truth rather than an empty table that 503s every model.
        with contextlib.suppress(Exception):
            await asyncio.to_thread(rt.routes.refresh)
        if poll:
            tasks.append(asyncio.create_task(_poll_loop(rt), name="servedeck-poll"))
        if reconcile:
            tasks.append(asyncio.create_task(_reconcile_after_bind(rt), name="servedeck-reconcile"))
        try:
            yield
        finally:
            # Subscribers first: a generator parked on queue.get() is what made
            # every shutdown take TimeoutStopSec.
            rt.hub.close()
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            await rt.client.aclose()
            gw_client = getattr(_app.state, "gateway_client", None)
            if gw_client is not None and getattr(_app.state, "gateway_owns_client", False):
                await gw_client.aclose()

    app = FastAPI(title="servedeck", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.rt = rt
    app.middleware("http")(_same_origin_only)

    gw = _gateway.build_router(rt.routes, transport=gateway_transport)
    app.state.gateway_client = gw.gateway_client
    app.state.gateway_owns_client = gw.gateway_owns_client
    _register_api(app, rt)
    # The gateway LAST: its `/v1/{path:path}` is a catch-all, and Starlette
    # matches in registration order, so a route registered after it is dead.
    app.include_router(gw)
    _register_page(app)
    return app


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


def _register_api(app: FastAPI, rt: Runtime) -> None:
    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        """Deliberately the cheapest possible answer: no systemd, no GPU, no
        registry walk. ``_reconcile_after_bind`` polls this to learn whether we
        won the port, and a health check that could itself fail would make the
        reconcile decision depend on something other than the bind."""
        return {
            "ok": True,
            "service": "servedeck",
            "port": rt.settings.listen_port,
            "instance": rt.instance_id,
        }

    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        if not rt.state:
            rt.state = await build_state(rt)
            rt.first_state.set()
        return rt.state

    @app.get("/api/models")
    async def api_models() -> dict[str, Any]:
        """The registry, joined with what is actually on disk.

        Two different questions share this route on purpose: "what can I run"
        is the registry, and "is it downloaded" is the hub cache. Separating
        them put the main-slot dropdown one round trip away from knowing
        whether the model it lists would have to download 90 GiB first.
        """
        entries = await asyncio.to_thread(_discovery.discover_models)
        by_repo = {e.repo_id: e for e in entries}
        live = rt.routes.live()
        legacy_rows = {
            r["key"]: r
            for r in _legacy.model_rows(
                rt.registry, entries, ctx_for=rt.routes.ctx_for,
                live_keys={k: bool(v.ready) for k, v in live.items()},
            )
        }
        registry_rows = []
        for key, model in rt.registry.models.items():
            entry = by_repo.get(model.repo)
            registry_rows.append(
                {
                    "key": key,
                    "id": model.id,
                    "repo": model.repo,
                    "slot": model.slot,
                    "build": model.build,
                    "on_disk": entry is not None and entry.servable,
                    "disk_gib": round(entry.disk_bytes / 1024**3, 2) if entry else None,
                    "reason": entry.reason if entry else "not in the local hub cache",
                    # v1 rail fields (legacy_page.model_rows), same row.
                    **{k: v for k, v in legacy_rows.get(key, {}).items()
                       if k not in ("key", "id", "repo", "slot", "build", "on_disk", "disk_gib", "reason")},
                }
            )
        return {
            "serving": next((r["name"] for r in legacy_rows.values() if r["serving"]), None),
            "disk": await asyncio.to_thread(_legacy.disk_payload, list(legacy_rows.values())),
            "models": registry_rows,
            "cache": [
                {
                    "repo_id": e.repo_id,
                    "servable": e.servable,
                    "disk_gib": round(e.disk_bytes / 1024**3, 2),
                    "arch": e.architectures0,
                    "in_registry": e.repo_id in {m.repo for m in rt.registry.models.values()},
                }
                for e in entries
            ],
        }

    @app.post("/api/models/{key}/start")
    async def start(key: str) -> JSONResponse:
        refusal = _precheck(rt, key, "start")
        if refusal is not None:
            return refusal
        _claim(rt, "start", key, f"start {key}")
        asyncio.create_task(
            _run_mutation(
                rt,
                f"start {key}",
                lambda: rt.control.start(key, on_progress=_progress_publisher(rt.hub, key, rt.boot)),
                action="start",
                key=key,
            )
        )
        return _accepted("start", key)

    @app.post("/api/models/{key}/stop")
    async def stop(key: str) -> JSONResponse:
        refusal = _precheck(rt, key, "stop")
        if refusal is not None:
            return refusal
        _claim(rt, "stop", key, f"stop {key}")
        asyncio.create_task(
            _run_mutation(rt, f"stop {key}", lambda: rt.control.stop(key), action="stop", key=key)
        )
        return _accepted("stop", key)

    @app.post("/api/switch/{key}")
    async def switch(key: str) -> JSONResponse:
        refusal = _precheck(rt, key, "switch")
        if refusal is not None:
            return refusal
        _claim(rt, "switch", key, f"switch {key}")
        asyncio.create_task(
            _run_mutation(
                rt,
                f"switch {key}",
                lambda: rt.control.switch(key, on_progress=_progress_publisher(rt.hub, key, rt.boot)),
                action="switch",
                key=key,
            )
        )
        return _accepted("switch", key)

    @app.post("/api/adopt")
    async def adopt() -> JSONResponse:
        if rt.busy is not None:
            return _refusal(409, "busy", f"servedeck is already running {rt.busy['label']}", busy=rt.busy)
        _claim(rt, "adopt", "", "adopt")
        asyncio.create_task(
            _run_mutation(rt, "adopt", lambda: rt.control.adopt(), action="adopt")
        )
        return _accepted("adopt", "")

    @app.get("/api/log/{key}")
    async def log_tail(key: str, lines: int = 80) -> JSONResponse:
        """``journalctl --user -u <unit> -n <lines>``.

        The journal outlives the unit (``--collect`` deletes the unit object,
        not its log), which is exactly why a failure report is built from this
        and not from ``Result=``."""
        if key not in rt.registry.models:
            return _refusal(404, "unknown_model", f"no model {key!r} in the registry")
        lines = max(1, min(int(lines), 1000))
        unit = f"{rt.settings.unit_prefix}{key}"
        tail = await asyncio.to_thread(_units.journal_tail, unit, lines)
        return JSONResponse({"key": key, "unit": unit, "lines": tail})

    @app.get("/api/wire")
    async def wire_diff() -> dict[str, Any]:
        return await asyncio.to_thread(_wire_payload, rt, False)

    @app.post("/api/wire/apply")
    async def wire_apply() -> dict[str, Any]:
        payload = await asyncio.to_thread(_wire_payload, rt, True)
        rt.hub.publish(
            "notice",
            {
                "level": "info",
                "reason": "wired",
                "message": "wire --apply: "
                + (", ".join(t["name"] for t in payload["targets"] if t["changed"]) or "no changes"),
            },
        )
        return payload

    @app.get("/api/doctor")
    async def doctor() -> dict[str, Any]:
        results = await asyncio.to_thread(_doctor.run_doctor, rt.settings.models_path)
        return {
            "ok": _doctor.all_ok(results),
            "checks": [{"name": r.name, "ok": r.ok, "detail": r.detail} for r in results],
        }

    @app.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        return StreamingResponse(
            _event_stream(rt, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    _legacy.register(app, rt)

def _wire_payload(rt: Runtime, apply: bool) -> dict[str, Any]:
    """The dry-run diff for every client config, and optionally the write.

    Same code path either way — the diff shown is literally the diff applied,
    which is the property that makes an Apply button trustworthy.
    """
    resolve_ctx = _wire.make_default_ctx_resolver(rt.registry)
    targets = []
    for target in _wire.WIRE_TARGETS:
        before = _wire.read_existing(target.path)
        after = target.render(rt.registry, before, resolve_ctx=resolve_ctx)
        changed = before != after
        backup: str | None = None
        if changed and apply:
            backup_dir = rt.settings.state_dir / "backups" / date.today().isoformat()
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / str(target.path).lstrip("/").replace("/", "_")
            backup_path.write_text(before)
            target.path.parent.mkdir(parents=True, exist_ok=True)
            target.path.write_text(after)
            backup = str(backup_path)
        targets.append(
            {
                "name": target.name,
                "path": str(target.path),
                "changed": changed,
                "diff": _wire.unified_diff(target.name, before, after) if changed else "",
                "backup": backup,
            }
        )
    return {"applied": apply, "targets": targets}


async def _event_stream(rt: Runtime, request: Request) -> AsyncIterator[bytes]:
    """One SSE connection.

    Opens with the current state and the recent notices, so a page that loads
    after everything interesting happened is not blank until the next change.
    Ends when the hub hands it the sentinel (shutdown) or the client goes away.
    """
    sub = rt.hub.subscribe()
    try:
        if rt.state:
            yield _frame("state", rt.state)
        for notice in rt.hub.notices[-50:]:
            # ``replay: true`` is not decoration. The page wants this backlog —
            # a page opened after a failure should still show the failure — but
            # a CLI watching for the end of the start IT just requested must
            # not stop on a "ready" from an hour ago. Without the flag,
            # `servedeck start` exits 0 the instant it connects, reporting the
            # previous boot's outcome as this one's.
            yield _frame("notice", {**notice["data"], "replay": True})
        while True:
            if await request.is_disconnected():
                return
            try:
                event = await asyncio.wait_for(sub.queue.get(), timeout=KEEPALIVE_S)
            except asyncio.TimeoutError:
                yield b": keepalive\n\n"
                continue
            if event is None:  # shutdown sentinel
                return
            yield _frame(event["type"], event["data"])
    finally:
        rt.hub.unsubscribe(sub)


def _frame(event_type: str, data: Any) -> bytes:
    if event_type == "notice" and isinstance(data, dict):
        # The v1 page reads `body` and `code`; the v2 shape is `message` and
        # `reason`. The frame carries both names; the stored notice keeps one.
        data = {**data, "body": data.get("body", data.get("message")),
                "code": data.get("code", data.get("reason"))}
    return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n".encode()


def _register_page(app: FastAPI) -> None:
    @app.get("/")
    async def index() -> Any:
        target = WEB / "index.html"
        if not target.is_file():  # pragma: no cover - only in a broken install
            return PlainTextResponse("servedeck: web/index.html is missing", status_code=500)
        return FileResponse(target)

    @app.get("/{asset:path}")
    async def asset(asset: str) -> Any:
        """Serve ``web/`` by explicit name.

        Registered after every API and gateway route, and resolved against
        ``WEB`` so a ``..`` cannot escape it. A StaticFiles mount would be
        shorter, but the previous app's catch-all was registered first and the
        mount it added later was never reached — an asset route that is dead
        and silent is worse than one written out.
        """
        candidate = (WEB / asset).resolve()
        if not str(candidate).startswith(str(WEB.resolve()) + "/") or not candidate.is_file():
            return PlainTextResponse("not found", status_code=404)
        return FileResponse(candidate)
