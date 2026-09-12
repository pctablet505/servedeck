"""The v2 supervisor: models as transient systemd units
(REDESIGN-2026-09-12.md §1 R2/R4, §2.2).

What this module deliberately does NOT contain, because systemd or the
registry already does it — this list is the design:

* **No `.config`, no launcher script, no `EXTRA_ARGS` stash.** The argv comes
  from the registry (R1/R2); nothing is written to a mutable bash file that a
  second control plane also writes.
* **No `/proc` walking and no pattern matching.** Discovery is
  ``systemctl --user list-units 'model-*'`` plus a port probe. There is no
  `pgrep -f` to match its own shell, and no way to signal a process we did not
  start: we stop *units*, by name, and only names matching ``model-<key>``.
* **No boot-phase state machine.** Boot progress is four substrings in the
  journal; readiness is ``GET /v1/models`` answering 200 with the model listed.
  The probe is the truth, the markers are the progress bar.
* **No KillMode question.** The model is in its own cgroup under the user
  manager, not in servedeck's. See ``tests/test_control_e2e.py``.

Two failure modes this module is built around, both measured on this box:

1. ``--collect`` deletes a *failed* unit, after which ``systemctl show``
   reports ``ActiveState=inactive Result=success`` — property defaults for an
   unknown unit. The natural failure check therefore reads "clean stop" for
   every crash. Every place that judges a unit here asks
   :func:`units.exists` first, and a unit that vanishes mid-boot is a failure.
2. vLLM v1 does not hold GPU memory in its ``MainPID``; the engine core and
   workers are child processes. Asking ``used_by_pids()[MainPID]`` how much a
   model holds returns ~0, so a switch would decide memory came back the
   instant it stopped the old model and launch the new one into a full card.
   :meth:`Control.switch` sums the *whole unit cgroup*.
"""

from __future__ import annotations

import json
import logging
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol, runtime_checkable

from servedeck import desired as desired_mod
from servedeck import gpu, units
from servedeck.desired import Desired
from servedeck.units import Runner, UnitError

__all__ = [
    "ModelSpec",
    "Registry",
    "LiveModel",
    "Refusal",
    "StartResult",
    "StopResult",
    "SwitchResult",
    "Adoption",
    "Reconciliation",
    "Progress",
    "Control",
    "READY_MARKERS",
    "UNIT_PREFIX",
    "DEFAULT_MARGIN_MIB",
    "floor2",
]

log = logging.getLogger(__name__)

UNIT_PREFIX = "model-"

#: The four lines vLLM prints on the way up, in the order it prints them.
#: They drive the progress bar only — readiness is decided by the port probe,
#: so a model that skips one (no CUDA graphs, a cached compile) still becomes
#: ready. Treating a missing marker as "not ready" would be an instrument
#: reporting on itself.
READY_MARKERS: tuple[str, ...] = (
    "Loading weights took",
    "GPU KV cache size:",
    "Capturing CUDA graphs",
    "Application startup complete.",
)

#: VRAM never handed to any model (models.toml ``[gpu] margin_mib``).
DEFAULT_MARGIN_MIB = 1024

_PROBE_TIMEOUT_S = 2.0
_TICK_S = 0.5
_PROBE_EVERY_S = 2.0
_HEALTH_EVERY_S = 2.0
#: When the probe says ready, keep draining the journal for this long before
#: reporting. The port answering and the last marker reaching journald race,
#: and the probe usually wins — so without this the progress display drops
#: "Application startup complete." from almost every successful boot, and a
#: test asserting the four markers would be flaky rather than wrong.
_READY_DRAIN_S = 1.0
#: switch() waits for this fraction of the stopped model's VRAM to come back.
RELEASE_FRACTION = 0.8
RELEASE_TIMEOUT_S = 120.0


# --------------------------------------------------------------------------
# The contract P1's registry must satisfy
# --------------------------------------------------------------------------


@runtime_checkable
class ModelSpec(Protocol):
    """One model, fully resolved from ``models.toml``.

    This is the entire surface :mod:`servedeck.control` needs. Anything else
    the registry knows (aliases' request overlays, reasoning parser, build
    name) is the gateway's business, not the supervisor's.
    """

    #: Registry key; also the unit name suffix (``model-<key>``). Must match
    #: ``[a-z0-9-]+`` or :func:`units.start_transient` will refuse it.
    key: str
    #: The model's canonical public id, e.g. ``"LFM2.5-350M"``. This is what
    #: ``GET /v1/models`` must list for the model to count as ready.
    id: str
    #: Every name vLLM is told to serve (``--served-model-name``): the id plus
    #: every alias any client has ever used. Never renamed, only added to.
    served_names: Sequence[str]
    #: ``"main"`` (exclusive GPU slot, one at a time) or ``"resident"``
    #: (co-resident, budgeted, always on).
    slot: str
    #: Loopback port this model's own vLLM listens on. Clients never see it;
    #: only the gateway and this module do.
    port: int
    #: For a resident: its VRAM budget in MiB, from which its utilisation is
    #: derived. For a main model: None — a main model gets everything free.
    vram_mib: int | None
    #: Context length in tokens, or None for "native" (vLLM decides).
    ctx_tokens: int | None
    #: Absolute path to the venv's ``bin`` DIRECTORY (e.g.
    #: ``/home/u/Projects/local_llm/.venv-llm-029/bin``). Control prepends it
    #: to ``PATH`` in the unit environment so every subprocess vLLM spawns
    #: resolves to the same interpreter. ``render_argv`` is still expected to
    #: return an absolute argv[0]; PATH is for vLLM's children, not for us.
    venv_bin: str

    def render_argv(self, util: float, port: int) -> list[str]:
        """The full argv to exec, argv[0] absolute.

        ``util`` is the ``--gpu-memory-utilization`` this module computed from
        live free memory; the registry must use the value it is handed rather
        than a configured one — that is decision 4 of the design ("main-slot
        utilisation is computed from free memory, not configured").
        """
        ...

    def render_env(self) -> Mapping[str, str]:
        """Environment for the unit. Must be COMPLETE for anything the model
        needs: a transient unit inherits the user manager's environment, never
        the calling shell's."""
        ...


@runtime_checkable
class Registry(Protocol):
    def models(self) -> Sequence[ModelSpec]: ...

    def get(self, key: str) -> ModelSpec:
        """The spec for ``key``. Raises :class:`KeyError` if unknown."""
        ...


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveModel:
    key: str
    unit: str
    port: int | None
    state: str
    pid: int
    ready: bool
    restarts: int
    sub_state: str = ""
    #: True when a ``model-*`` unit exists whose key the registry does not
    #: know. Never acted on automatically: without a spec there is no slot,
    #: no port and no way to tell a stray from a model.
    unknown: bool = False


@dataclass(frozen=True)
class Refusal:
    """A refusal to act, with a machine-readable reason. Returned, not
    raised: every one of these is an expected operational answer."""

    reason: str
    message: str
    key: str | None = None
    #: For ``main_slot_busy``: the model currently holding the main slot.
    live_key: str | None = None


@dataclass(frozen=True)
class StartResult:
    key: str
    unit: str
    ready: bool
    elapsed_s: float
    #: Markers seen, in order. A short list on a failure says how far it got.
    markers: list[str] = field(default_factory=list)
    #: None when ready; otherwise why the wait ended.
    failure: str | None = None
    #: Last 40 journal lines, populated on failure.
    journal: list[str] = field(default_factory=list)
    util: float | None = None
    argv: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StopResult:
    key: str
    unit: str
    was_live: bool
    #: VRAM the unit's cgroup held immediately before the stop, in MiB; None
    #: if nvidia-smi could not say.
    held_mib: int | None = None
    free_before_mib: int | None = None


@dataclass(frozen=True)
class SwitchResult:
    stopped: StopResult | None
    released: bool
    waited_s: float
    free_after_mib: int | None
    started: StartResult | Refusal | None


@dataclass(frozen=True)
class Adoption:
    """Units that exist but were not in desired state."""

    adopted: list[str] = field(default_factory=list)
    unknown_units: list[str] = field(default_factory=list)
    desired: Desired = field(default_factory=Desired)
    written: bool = False


@dataclass(frozen=True)
class Reconciliation:
    already_live: list[str] = field(default_factory=list)
    #: Unit exists but is not answering yet — left strictly alone. Restarting
    #: a booting model is the failure this whole design exists to remove.
    booting: list[str] = field(default_factory=list)
    started: list[StartResult] = field(default_factory=list)
    refused: list[Refusal] = field(default_factory=list)


@dataclass(frozen=True)
class Progress:
    """One boot-progress event, shaped for an SSE frame."""

    kind: str  # "line" | "marker" | "ready" | "failed"
    text: str
    marker_index: int | None = None
    elapsed_s: float = 0.0


ProgressCallback = Callable[[Progress], None]
#: ``(port) -> list of model ids reported by GET /v1/models``, or None when the
#: port did not answer 200.
Probe = Callable[[int], list[str] | None]


def floor2(value: float) -> float:
    """Truncate to two decimals. FLOOR, never round: rounding 0.9151 up to
    0.92 hands a model 10 MiB of the safety margin, and the margin exists
    precisely because the last 10 MiB is what OOMs a boot."""
    return math.floor(value * 100) / 100


def http_probe(port: int, timeout_s: float = _PROBE_TIMEOUT_S) -> list[str] | None:
    """``GET http://127.0.0.1:<port>/v1/models`` -> the ids listed, or None.

    None covers every not-yet/never case identically (connection refused, a
    half-open socket during boot, a non-200, unparseable JSON) because the
    caller's question is only ever "can I use this model right now".
    """
    url = f"http://127.0.0.1:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310
            if response.status != 200:
                return None
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return None
    return [item["id"] for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)]


def _cgroup_pids(unit: str, run: Runner | None = None) -> list[int]:
    """Every pid in the unit's cgroup, from
    ``/sys/fs/cgroup/<ControlGroup>/cgroup.procs``.

    This is not ``/proc`` walking and not pattern matching: systemd tells us
    the cgroup, and the kernel tells us exactly which processes are in it. It
    is the only way to account for vLLM v1's memory, which lives in the engine
    core and worker children rather than in ``MainPID``.
    """
    try:
        path = units.control_group(unit, run=run)
    except UnitError:
        return []
    if not path:
        return []
    procs = Path("/sys/fs/cgroup") / path.lstrip("/") / "cgroup.procs"
    try:
        text = procs.read_text(encoding="utf-8")
    except OSError:
        return []
    pids: list[int] = []
    for line in text.split():
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids


class Control:
    """Start, stop, switch, adopt and reconcile models.

    Every external dependency is injectable so the whole thing is testable
    without a systemd, a GPU or a model: ``run`` (systemctl/systemd-run),
    ``spawn`` (journalctl -f), ``probe`` (/v1/models), ``free_mib`` /
    ``used_by_pids`` (nvidia-smi), ``cgroup_pids``, ``clock`` and ``sleep``.
    """

    def __init__(
        self,
        registry: Registry,
        *,
        margin_mib: int = DEFAULT_MARGIN_MIB,
        total_mib: int | None = None,
        unit_prefix: str = UNIT_PREFIX,
        desired_path: str | Path | None = None,
        run: Runner | None = None,
        spawn: Callable[[Sequence[str]], object] | None = None,
        probe: Probe | None = None,
        free_mib: Callable[[], int | None] | None = None,
        used_by_pids: Callable[[], dict[int, int] | None] | None = None,
        cgroup_pids: Callable[[str], list[int]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if unit_prefix not in (UNIT_PREFIX, "sd-test-"):
            raise ValueError(
                f"unit_prefix must be {UNIT_PREFIX!r} (production) or 'sd-test-' "
                f"(the suite's own namespace), not {unit_prefix!r}"
            )
        self.registry = registry
        self.margin_mib = margin_mib
        self._total_mib = total_mib
        # The namespace this Control owns. Production is 'model-'; the e2e
        # suite drives this exact code under 'sd-test-' so a test can never
        # create, adopt or stop a real model unit. Both discovery (the
        # list-units glob) and action (the unit name) come from this one
        # value, so there is no way for them to disagree.
        self.unit_prefix = unit_prefix
        self.desired_path = Path(desired_path) if desired_path is not None else None
        self._run = run
        self._spawn = spawn
        self._probe: Probe = probe or http_probe
        self._free_mib = free_mib or gpu.free_mib
        self._used_by_pids = used_by_pids or gpu.used_by_pids
        self._cgroup_pids = cgroup_pids or (lambda unit: _cgroup_pids(unit, run=self._run))
        self._clock = clock
        self._sleep = sleep

    # -- small helpers ----------------------------------------------------

    def unit_for(self, key: str) -> str:
        return f"{self.unit_prefix}{key}"

    def key_for(self, unit: str) -> str:
        prefix = self.unit_prefix
        return unit[len(prefix) :] if unit.startswith(prefix) else unit

    def total_mib(self) -> int | None:
        if self._total_mib is not None:
            return self._total_mib
        return gpu.total_mib()

    def load_desired(self) -> Desired:
        return desired_mod.load(self.desired_path)

    def _save_desired(self, value: Desired) -> None:
        desired_mod.save(value, self.desired_path)

    def _spec(self, key: str) -> ModelSpec | None:
        try:
            return self.registry.get(key)
        except KeyError:
            return None

    # -- discovery --------------------------------------------------------

    def live(self) -> list[LiveModel]:
        """Every ``model-*`` unit systemd knows, joined with a port probe.

        This replaces adoption-by-``/proc``-scan wholesale (R4). A unit the
        registry does not know is reported with ``unknown=True`` rather than
        guessed at: no spec means no port to probe and no slot to reason about.
        """
        out: list[LiveModel] = []
        for unit in units.list_units(f"{self.unit_prefix}*", run=self._run):
            key = self.key_for(unit)
            state = units.show(unit, run=self._run)
            spec = self._spec(key)
            if spec is None:
                out.append(
                    LiveModel(
                        key=key,
                        unit=unit,
                        port=None,
                        state=state.active_state,
                        sub_state=state.sub_state,
                        pid=state.main_pid,
                        ready=False,
                        restarts=state.n_restarts,
                        unknown=True,
                    )
                )
                continue
            ids = self._probe(spec.port) if state.active_state == "active" else None
            names = {spec.id, *spec.served_names}
            ready = ids is not None and bool(names & set(ids))
            out.append(
                LiveModel(
                    key=key,
                    unit=unit,
                    port=spec.port,
                    state=state.active_state,
                    sub_state=state.sub_state,
                    pid=state.main_pid,
                    ready=ready,
                    restarts=state.n_restarts,
                )
            )
        return out

    def live_main(self, live: Sequence[LiveModel] | None = None) -> LiveModel | None:
        """The model currently holding the exclusive main slot, if any."""
        for model in live if live is not None else self.live():
            spec = self._spec(model.key)
            if spec is not None and spec.slot == "main":
                return model
        return None

    # -- utilisation ------------------------------------------------------

    def compute_util(self, spec: ModelSpec) -> float | Refusal:
        """``--gpu-memory-utilization`` for this model, right now.

        main:     floor2((free_mib - margin_mib) / total_mib)
        resident: floor2(vram_mib / total_mib)

        Decision 4 of the design: a main model is never configured with a
        utilisation, it is handed everything that is free minus the margin.
        That is what makes "a resident can block a big-model boot" (R4)
        impossible — the resident's 3.3 GiB is simply not free any more, so it
        is not offered.
        """
        total = self.total_mib()
        if total is None or total <= 0:
            return Refusal(
                reason="gpu_unavailable",
                message="nvidia-smi could not report total VRAM; refusing to guess a utilisation",
                key=spec.key,
            )
        free = self._free_mib()
        if free is None:
            return Refusal(
                reason="gpu_unavailable",
                message="nvidia-smi could not report free VRAM; refusing to guess a utilisation",
                key=spec.key,
            )
        if free <= self.margin_mib:
            return Refusal(
                reason="not_enough_vram",
                message=(
                    f"{free} MiB free is not more than the {self.margin_mib} MiB margin; "
                    f"nothing can be offered to {spec.key}"
                ),
                key=spec.key,
            )
        if spec.slot == "main":
            util = floor2((free - self.margin_mib) / total)
        else:
            if spec.vram_mib is None or spec.vram_mib <= 0:
                return Refusal(
                    reason="no_vram_budget",
                    message=f"resident {spec.key} has no vram_mib budget; utilisation is underived",
                    key=spec.key,
                )
            util = floor2(spec.vram_mib / total)
        if util <= 0:
            return Refusal(
                reason="not_enough_vram",
                message=(
                    f"computed utilisation for {spec.key} rounds down to {util}; "
                    f"{free} MiB free, {total} MiB total, {self.margin_mib} MiB margin"
                ),
                key=spec.key,
            )
        return util

    # -- start ------------------------------------------------------------

    def start(
        self,
        key: str,
        timeout_s: float = 900.0,
        on_progress: ProgressCallback | None = None,
        restart: str = "on-failure",
        restart_sec: int = 10,
    ) -> StartResult | Refusal:
        """Launch ``key`` as ``model-<key>.service`` and wait for it to answer.

        Refuses rather than raises for every operational "no": unknown model,
        already running, main slot taken, not enough VRAM.
        """
        spec = self._spec(key)
        if spec is None:
            return Refusal(reason="unknown_model", message=f"no model {key!r} in the registry", key=key)

        unit = self.unit_for(key)
        current = self.live()
        for model in current:
            if model.key == key:
                return Refusal(
                    reason="already_live",
                    message=f"{unit} already exists ({model.state}/{model.sub_state}); "
                    f"stop it first if you mean to relaunch",
                    key=key,
                    live_key=key,
                )
        if spec.slot == "main":
            holder = self.live_main(current)
            if holder is not None:
                return Refusal(
                    reason="main_slot_busy",
                    message=(
                        f"the main slot is held by {holder.key} ({holder.unit}); "
                        f"use switch({key!r}) to replace it"
                    ),
                    key=key,
                    live_key=holder.key,
                )

        util = self.compute_util(spec)
        if isinstance(util, Refusal):
            return util

        argv = list(spec.render_argv(util, spec.port))
        env = dict(spec.render_env())
        env.setdefault("PATH", f"{spec.venv_bin}:/usr/local/bin:/usr/bin:/bin")

        # Start the follower BEFORE the unit, from a timestamp a couple of
        # seconds in the past: journalctl --since has one-second granularity,
        # so anchoring at "now" can drop the first lines of a fast boot. Seeing
        # a line twice is free; missing "Loading weights took" is not.
        since = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 2))
        started_at = self._clock()
        try:
            units.start_transient(
                unit,
                argv,
                env,
                cwd=str(Path(spec.venv_bin).parent),
                restart=restart,
                restart_sec=restart_sec,
                description=f"servedeck model {spec.id} ({key})",
                run=self._run,
            )
        except UnitError as exc:
            return Refusal(reason="start_failed", message=str(exc), key=key)

        result = self.wait_ready(
            key,
            timeout_s=timeout_s,
            on_progress=on_progress,
            since=since,
            started_at=started_at,
        )
        result = replace(result, util=util, argv=argv)
        if result.ready:
            current_desired = self.load_desired()
            if spec.slot == "main":
                self._save_desired(current_desired.with_main(key))
            else:
                self._save_desired(current_desired.with_resident(key))
        return result

    def wait_ready(
        self,
        key: str,
        timeout_s: float = 900.0,
        on_progress: ProgressCallback | None = None,
        since: str = "-1min",
        started_at: float | None = None,
    ) -> StartResult:
        """Follow the unit's journal until the model answers, fails or times out.

        Readiness is the port probe, not the markers (see :data:`READY_MARKERS`).
        Failure is any of: the unit went to ``failed``; it restarted at least
        once (a crash loop at boot); or **it vanished** — ``--collect`` deletes a
        failed unit, and the state call then reports property defaults that look
        exactly like a clean stop.
        """
        spec = self._spec(key)
        unit = self.unit_for(key)
        started_at = self._clock() if started_at is None else started_at
        deadline = started_at + timeout_s
        markers: list[str] = []
        next_marker = 0
        last_probe = 0.0
        last_health = started_at

        def emit(kind: str, text: str, marker_index: int | None = None) -> None:
            if on_progress is None:
                return
            on_progress(
                Progress(
                    kind=kind,
                    text=text,
                    marker_index=marker_index,
                    elapsed_s=self._clock() - started_at,
                )
            )

        def finish(ready: bool, failure: str | None) -> StartResult:
            journal = [] if ready else units.journal_tail(unit, 40, run=self._run)
            emit("ready" if ready else "failed", failure or f"{key} is ready")
            return StartResult(
                key=key,
                unit=unit,
                ready=ready,
                elapsed_s=self._clock() - started_at,
                markers=list(markers),
                failure=failure,
                journal=journal,
            )

        def consume(lines: list[str]) -> None:
            nonlocal next_marker
            for line in lines:
                emit("line", line)
                if next_marker < len(READY_MARKERS) and READY_MARKERS[next_marker] in line:
                    markers.append(READY_MARKERS[next_marker])
                    emit("marker", line, next_marker)
                    next_marker += 1

        stream = units.journal_follow(unit, since, spawn=self._spawn)
        try:
            while True:
                now = self._clock()
                if now >= deadline:
                    return finish(
                        False,
                        f"timed out after {timeout_s:.0f}s waiting for {key}; "
                        f"saw {len(markers)}/{len(READY_MARKERS)} boot markers",
                    )

                consume(stream.poll_lines(_TICK_S))

                now = self._clock()
                if now - last_health >= _HEALTH_EVERY_S:
                    last_health = now
                    if not units.exists(unit, run=self._run):
                        return finish(
                            False,
                            f"{unit} no longer exists: it failed and --collect removed it "
                            f"(systemctl show would report inactive/success for it now)",
                        )
                    state = units.show(unit, run=self._run)
                    if state.failed:
                        return finish(
                            False,
                            f"{unit} failed: ActiveState={state.active_state} "
                            f"Result={state.result} NRestarts={state.n_restarts}",
                        )
                    if state.n_restarts > 0:
                        return finish(
                            False,
                            f"{unit} restarted {state.n_restarts}x while booting; "
                            f"it is crash-looping, not starting",
                        )

                if spec is not None and now - last_probe >= _PROBE_EVERY_S:
                    last_probe = now
                    ids = self._probe(spec.port)
                    if ids is not None and ({spec.id, *spec.served_names} & set(ids)):
                        drain_until = self._clock() + _READY_DRAIN_S
                        while self._clock() < drain_until:
                            consume(stream.poll_lines(min(0.2, _READY_DRAIN_S)))
                        return finish(True, None)
        finally:
            stream.close()

    # -- stop -------------------------------------------------------------

    def stop(self, key: str, timeout_s: float = units.DEFAULT_STOP_TIMEOUT_S) -> StopResult | Refusal:
        """Stop ``model-<key>`` and record that it is no longer wanted.

        The VRAM accounting is captured *before* the stop, because after it the
        pids are gone and there is nothing left to attribute memory to.
        """
        unit = self.unit_for(key)
        if not units.valid_unit_name(unit):
            return Refusal(reason="bad_key", message=f"{key!r} is not a usable model key", key=key)
        was_live = units.exists(unit, run=self._run)
        free_before = self._free_mib()
        held = self._held_mib(unit) if was_live else 0
        try:
            units.stop(unit, timeout_s=timeout_s, run=self._run)
        except UnitError as exc:
            return Refusal(reason="stop_failed", message=str(exc), key=key)

        current = self.load_desired()
        if current.main == key:
            current = current.with_main(None)
        current = current.without_resident(key)
        self._save_desired(current)
        return StopResult(
            key=key, unit=unit, was_live=was_live, held_mib=held, free_before_mib=free_before
        )

    def _held_mib(self, unit: str) -> int | None:
        """GPU memory held by every process in the unit's cgroup, in MiB.

        Not ``used_by_pids()[MainPID]``: in vLLM v1 the API server is MainPID
        and holds nothing, while the engine-core and worker children hold all
        of it. Summing MainPID alone would report ~0 and make
        :meth:`switch` believe a 90 GiB model released instantly.
        """
        usage = self._used_by_pids()
        if usage is None:
            return None
        pids = set(self._cgroup_pids(unit))
        if not pids:
            return 0
        return sum(mib for pid, mib in usage.items() if pid in pids)

    # -- switch -----------------------------------------------------------

    def switch(
        self,
        key: str,
        timeout_s: float = 900.0,
        release_timeout_s: float = RELEASE_TIMEOUT_S,
        on_progress: ProgressCallback | None = None,
    ) -> SwitchResult | Refusal:
        """Replace whatever holds the main slot with ``key``.

        Stop, then **wait for the memory to actually come back** before
        starting the next model. The driver frees a context asynchronously; a
        switch that trusts the stop's exit code launches a 90 GiB model into a
        card that still has the last one in it, and the failure surfaces
        minutes later as a CUDA OOM with no obvious cause.

        The bar is ``free_mib`` rising by ``RELEASE_FRACTION`` (80%) of what
        the stopped unit's cgroup held — not 100%, because the driver keeps
        some context allocated, and not a fixed sleep, because a fixed sleep is
        either wrong or slow.
        """
        spec = self._spec(key)
        if spec is None:
            return Refusal(reason="unknown_model", message=f"no model {key!r} in the registry", key=key)
        if spec.slot != "main":
            return Refusal(
                reason="not_main_slot",
                message=f"{key} is a {spec.slot} model; switch only replaces the main slot",
                key=key,
            )

        holder = self.live_main()
        stopped: StopResult | None = None
        released = True
        waited = 0.0
        free_after = self._free_mib()

        if holder is not None and holder.key != key:
            result = self.stop(holder.key)
            if isinstance(result, Refusal):
                return result
            stopped = result
            released, waited, free_after = self._wait_for_release(
                free_before=result.free_before_mib,
                held_mib=result.held_mib,
                timeout_s=release_timeout_s,
                on_progress=on_progress,
            )
            if not released:
                return SwitchResult(
                    stopped=stopped,
                    released=False,
                    waited_s=waited,
                    free_after_mib=free_after,
                    started=Refusal(
                        reason="vram_not_released",
                        message=(
                            f"{result.unit} held {result.held_mib} MiB; after {waited:.0f}s only "
                            f"{(free_after or 0) - (result.free_before_mib or 0)} MiB came back. "
                            f"Refusing to boot {key} into a card that is still occupied."
                        ),
                        key=key,
                        live_key=holder.key,
                    ),
                )
        elif holder is not None and holder.key == key:
            return Refusal(
                reason="already_live",
                message=f"{key} already holds the main slot",
                key=key,
                live_key=key,
            )

        started = self.start(key, timeout_s=timeout_s, on_progress=on_progress)
        return SwitchResult(
            stopped=stopped,
            released=released,
            waited_s=waited,
            free_after_mib=free_after,
            started=started,
        )

    def _wait_for_release(
        self,
        free_before: int | None,
        held_mib: int | None,
        timeout_s: float,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[bool, float, int | None]:
        started = self._clock()
        if not held_mib or free_before is None:
            # Nothing was attributed (a CPU-only unit) or nvidia-smi is mute:
            # there is no number to wait for, so waiting would be theatre.
            return True, 0.0, self._free_mib()
        target = free_before + RELEASE_FRACTION * held_mib
        free_now = self._free_mib()
        while True:
            free_now = self._free_mib()
            if free_now is not None and free_now >= target:
                return True, self._clock() - started, free_now
            waited = self._clock() - started
            if waited >= timeout_s:
                return False, waited, free_now
            if on_progress is not None:
                on_progress(
                    Progress(
                        kind="line",
                        text=f"waiting for VRAM: {free_now} MiB free, need {int(target)} MiB",
                        elapsed_s=waited,
                    )
                )
            self._sleep(1.0)

    # -- adopt / reconcile ------------------------------------------------

    def adopt(self, write: bool = True) -> Adoption:
        """Bring desired state into line with the units that actually exist.

        An explicit operator action, like start/stop/switch — which is why it
        may write ``desired.json`` while :meth:`reconcile` may not. Units whose
        key is not in the registry are *reported*, never adopted: without a
        spec there is no slot to file them under.
        """
        current = self.load_desired()
        adopted: list[str] = []
        unknown: list[str] = []
        for model in self.live():
            if model.unknown:
                unknown.append(model.unit)
                continue
            spec = self._spec(model.key)
            assert spec is not None  # model.unknown is False
            if spec.slot == "main":
                if current.main != model.key:
                    current = current.with_main(model.key)
                    adopted.append(model.key)
            elif model.key not in current.residents:
                current = current.with_resident(model.key)
                adopted.append(model.key)
        if write and adopted:
            self._save_desired(current)
        return Adoption(
            adopted=adopted, unknown_units=unknown, desired=current, written=bool(write and adopted)
        )

    def reconcile(
        self,
        desired: Desired | None = None,
        timeout_s: float = 900.0,
        on_progress: ProgressCallback | None = None,
    ) -> Reconciliation:
        """At servedeck start: make desired state true, restarting nothing.

        The rule with teeth is the negative one — **a unit that exists is left
        alone**, ready or not. Servedeck restarting is not a reason for a model
        to restart (R4); and a model that is 40 seconds into a 90-second boot
        must not be killed by the dashboard coming up. Reconcile never stops
        anything and never writes desired state.
        """
        want = self.load_desired() if desired is None else desired
        existing = {model.key: model for model in self.live()}
        already: list[str] = []
        booting: list[str] = []
        started: list[StartResult] = []
        refused: list[Refusal] = []

        order = [*want.residents, *([want.main] if want.main else [])]
        for key in order:
            model = existing.get(key)
            if model is not None:
                (already if model.ready else booting).append(key)
                continue
            result = self.start(key, timeout_s=timeout_s, on_progress=on_progress)
            if isinstance(result, Refusal):
                refused.append(result)
            else:
                started.append(result)
                existing[key] = LiveModel(
                    key=key,
                    unit=result.unit,
                    port=None,
                    state="active",
                    pid=0,
                    ready=result.ready,
                    restarts=0,
                )
        return Reconciliation(
            already_live=already, booting=booting, started=started, refused=refused
        )
