"""The registry, seen through the gateway's eyes (REDESIGN-2026-09-12.md §2.3).

``gateway.py`` (P2) declares a ``RouteTable`` Protocol and nothing that
implements it; ``models.py`` (P1) knows every model's names, port and policies;
``control.py`` (P3) knows which of them is answering right now. This module is
the one join between the three, and it is the only place in servedeck where the
two questions a client's request asks — *what is this name?* and *is it up?* —
are answered together.

Two design rules, both load-bearing:

**1. Resolution is total; liveness is a snapshot.**  ``resolve()`` answers from
the registry, which is a file loaded once, so a name that is spelled in
``models.toml`` always resolves — even while nothing is running. That is what
separates a 404 ("reconfigure yourself") from a 503 ("wait"), the distinction
``gateway.not_running_response`` exists to draw. Collapsing them would send a
user hunting for a config file every time a model was booting.

**2. Nothing in a request path shells out.**  ``control.live()`` runs
``systemctl show`` per unit and probes every port; doing that inside
``resolve()`` would put two subprocesses and a socket connect on the critical
path of every chat completion, on the event loop thread. So liveness is a
*pushed* snapshot: :meth:`RegistryRoutes.refresh` is called from a worker
thread by ``app.py``'s poller (and after every mutation), and the request path
only ever reads the dict it left behind. An un-refreshed table reports nothing
live, which is the safe direction: a 503 with Retry-After, not a proxy attempt
at a port with nothing behind it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from servedeck import models as _models
from servedeck.gateway import Route, RoutePolicies

__all__ = ["RegistryRoutes", "LiveView", "resolve_ctx", "make_ctx_resolver"]

log = logging.getLogger(__name__)


class _LiveSource(Protocol):
    """The half of :class:`servedeck.control.Control` this module uses."""

    def live(self) -> Sequence[Any]: ...


@dataclass(frozen=True)
class LiveView:
    """One model's liveness, flattened out of ``control.LiveModel``.

    A frozen snapshot rather than the live object, because the gateway reads it
    from the event loop while the poller replaces it from a worker thread: a
    reader can only ever see a complete, self-consistent record, never a
    half-updated one.
    """

    key: str
    unit: str
    ready: bool
    state: str
    sub_state: str
    pid: int
    restarts: int
    port: int | None = None
    unknown: bool = False
    adopted: bool = False

    @property
    def unit_state(self) -> str:
        """``active (running)`` — what the page shows in the unit column."""
        if not self.state:
            return "unknown"
        return f"{self.state} ({self.sub_state})" if self.sub_state else self.state


CtxResolver = Callable[[_models.Model], int]


def resolve_ctx(model: _models.Model, hub_dir: str | None = None) -> int:
    """A model's context length in tokens, with ``ctx = "native"`` resolved.

    Returns 0 — never a guess — when the checkpoint is not in the local hub
    cache and its native length therefore cannot be read. 0 travels honestly:
    ``/v1/models`` reports ``max_model_len: 0`` and ``/api/state`` carries the
    ``ctx_error`` string beside it, so the page can say *why* rather than print
    a plausible-looking number nobody measured. Substituting a default here is
    how a client ends up sizing its prompts against a context the model does
    not have.
    """
    if isinstance(model.ctx, int):
        return model.ctx
    return _models.native_ctx(model.repo, hub_dir)


def make_ctx_resolver(hub_dir: str | None = None) -> CtxResolver:
    """A :data:`CtxResolver` that memoises ``native_ctx`` per model key.

    ``native_ctx`` reads a JSON file off disk; the gateway asks for ``ctx`` on
    every ``/v1/models`` request and on every request that carries an output
    floor, so without the memo the answer would be re-read from disk hundreds
    of times a minute for a value that cannot change while the process runs.
    """
    cache: dict[str, int] = {}

    def resolve(model: _models.Model) -> int:
        hit = cache.get(model.key)
        if hit is not None:
            return hit
        try:
            value = resolve_ctx(model, hub_dir)
        except _models.RegistryError as exc:
            log.warning("ctx for %s is unknown: %s", model.key, exc)
            value = 0
        cache[model.key] = value
        return value

    return resolve


class RegistryRoutes:
    """``gateway.RouteTable`` over P1's registry and P3's ``Control``.

    ``ctx_resolver`` and the initial ``live`` map are injectable so the whole
    class is testable with a registry, a dict and no systemd — which is what
    ``tests/test_routes.py`` does.
    """

    def __init__(
        self,
        registry: _models.Registry,
        control: _LiveSource | None = None,
        *,
        ctx_resolver: CtxResolver | None = None,
        hub_dir: str | None = None,
    ) -> None:
        self.registry = registry
        self.control = control
        self._ctx = ctx_resolver or make_ctx_resolver(hub_dir)
        #: key -> LiveView. Replaced wholesale by refresh(); never mutated in
        #: place, so a reader on the event loop cannot observe a partial update.
        self._live: dict[str, LiveView] = {}
        self._ctx_errors: dict[str, str] = {}

    # -- liveness ---------------------------------------------------------

    def refresh(self) -> dict[str, LiveView]:
        """Re-read liveness from Control. **Blocking** — subprocesses and a
        socket probe per unit — so call it from a worker thread, never from a
        coroutine. Returns the new snapshot.

        A failure to reach systemd leaves the previous snapshot in place rather
        than emptying it: one flaky ``systemctl`` call must not take a serving
        model off the gateway for two seconds.
        """
        if self.control is None:
            return self._live
        try:
            live = self.control.live()
        except Exception:  # noqa: BLE001 - discovery failing must not blank the table
            log.exception("control.live() failed; keeping the previous snapshot")
            return self._live
        self.set_live(live)
        return self._live

    def set_live(self, live: Sequence[Any]) -> dict[str, LiveView]:
        """Install a liveness snapshot from ``control.LiveModel``-shaped rows."""
        snapshot: dict[str, LiveView] = {}
        for row in live:
            view = LiveView(
                key=row.key,
                unit=row.unit,
                ready=bool(row.ready),
                state=getattr(row, "state", ""),
                sub_state=getattr(row, "sub_state", ""),
                pid=getattr(row, "pid", 0),
                restarts=getattr(row, "restarts", 0),
                port=getattr(row, "port", None),
                unknown=bool(getattr(row, "unknown", False)),
                adopted=bool(getattr(row, "adopted", False)),
            )
            snapshot[view.key] = view
        self._live = snapshot
        return snapshot

    def live(self) -> dict[str, LiveView]:
        """The current liveness snapshot, keyed by registry key."""
        return self._live

    def live_view(self, key: str) -> LiveView | None:
        return self._live.get(key)

    # -- ctx --------------------------------------------------------------

    def ctx_for(self, model: _models.Model) -> int:
        """Context length in tokens; 0 when ``native`` could not be read."""
        try:
            value = self._ctx(model)
        except _models.RegistryError as exc:
            self._ctx_errors[model.key] = str(exc)
            return 0
        if value:
            self._ctx_errors.pop(model.key, None)
        elif model.ctx == "native":
            self._ctx_errors.setdefault(
                model.key,
                f"{model.repo} is not in the local hub cache, so its native "
                "context length could not be read",
            )
        return value

    def ctx_error(self, key: str) -> str | None:
        return self._ctx_errors.get(key)

    # -- gateway.RouteTable -----------------------------------------------

    def route_for(self, model: _models.Model, *, preset: str | None = None) -> Route:
        """One registry model as a gateway ``Route``.

        The policies are assembled here and nowhere else:

        ``mirror_reasoning``   the model's ``reasoning.mirror_content``.
        ``effort_overlay``     the *preset's* overlay, or None for a plain
                               id/alias route — a preset is the only thing that
                               carries one, which is what makes an alias a
                               synonym and a preset a different request.
        ``min_output_tokens``  the model's floor (the GLM thinking-budget fix).
        ``ctx``                resolved context, the clamp for that floor.
        """
        overlay: Mapping[str, Any] | None = None
        if preset is not None:
            overlay = model.presets.get(preset) or None
        return Route(
            model_id=model.id,
            port=model.port,
            live=self.is_live(model.key),
            aliases=tuple(model.aliases),
            presets=tuple(model.presets),
            policies=RoutePolicies(
                mirror_reasoning=bool(model.reasoning and model.reasoning.mirror_content),
                effort_overlay=overlay,
                min_output_tokens=model.min_output_tokens,
                ctx=self.ctx_for(model),
            ),
        )

    def is_live(self, key: str) -> bool:
        view = self._live.get(key)
        return bool(view and view.ready)

    def resolve(self, name: str) -> Route | None:
        """Route for ``name`` — id, alias, preset or registry key — live or not.

        Delegates the name lookup to ``Registry.resolve`` rather than keeping a
        second index: one place decides what a name means, so the gateway and
        ``servedeck doctor`` can never disagree about whether a model exists.
        """
        found = self.registry.resolve(name)
        if found is None:
            return None
        return self.route_for(found.model, preset=found.preset)

    def live_routes(self) -> list[Route]:
        """Every ready model, for ``GET /v1/models``.

        Ready, not merely started: a unit that exists but is 40 s into a boot
        must not be advertised, or a client picks it off the list and gets a
        connection refused instead of the 503 that tells it to wait.
        """
        return [
            self.route_for(model)
            for key, model in self.registry.models.items()
            if self.is_live(key)
        ]

    def main(self) -> Route | None:
        """The model in the exclusive main slot, live or booting.

        Answered from the registry plus the snapshot, so it names the model
        that *holds* the slot even before it is ready — which is the whole
        content of the 503 body ("main slot: GLM-5.3, still starting").
        """
        for key, model in self.registry.models.items():
            if model.slot != "main":
                continue
            if key in self._live:
                return self.route_for(model)
        return None

    def known_names(self) -> list[str]:
        """Every name ``resolve`` accepts, for the 404 body: each model's id,
        its aliases and its presets, in registry order."""
        out: list[str] = []
        for model in self.registry.models.values():
            out.extend(model.served_names())
        return out
