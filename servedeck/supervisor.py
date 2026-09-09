"""Servedeck supervisor — SPEC.md §6. TOP-PRIORITY MODULE.

Owns the desired/actual state machine for the one vLLM server Servedeck
manages, startup reconciliation, the boot-phase monitor, the
crash-while-serving vs failed-boot distinction, backoff-based auto-restart
with a crash-loop ceiling, and death recording. Every other piece of this
file exists in service of the two rules SPEC.md calls out as essential:

1. Auto-restart is gated on ``desired_state == "RUNNING"``, full stop.
   :meth:`Supervisor.stop` persists ``desired_state = "STOPPED"`` (via
   :func:`save_desired`) *before* it signals the process group — so the
   exit that follows is always classified ``deliberate`` and never reaches
   the restart-decision logic at all. This is the fix for the watchdog
   defect SETUP.md:393 documents: a supervisor that cannot tell "the user
   meant to stop this" from "this crashed" will eventually resurrect a
   server someone deliberately stopped.

2. ``reached_ready`` (SPEC.md §5's phase-FSM flag, true the instant BOTH
   "Application startup complete." *and* a 200 from ``GET /v1/models``
   have been observed) separates two failure populations that must be
   handled oppositely: a crash *after* successfully serving is restarted
   (with backoff); a boot that *never* reached READY is not restarted at
   all — it would fail identically every time, and a blind retry loop
   would burn the backoff schedule hiding a deterministic, fixable error.
   SPEC.md's own evidence: six consecutive boots failed on 2026-08-27 with
   six *different* root causes; auto-restarting any of them would have
   hidden the other five.

Everything else here — startup reconciliation's five cases, the backoff
schedule and its window/ceiling, crash-loop suspension with a persisted
acknowledgement, Xid classification, and death recording — is built on top
of those two rules, not around them.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Mapping, Sequence

import httpx

from servedeck import config, gpu, history, logtail, paths, phases, preflight, procctl, shellconfig

# ---------------------------------------------------------------------------
# Types & constants — SPEC.md §6
# ---------------------------------------------------------------------------

DesiredValue = Literal["STOPPED", "RUNNING"]
ActualValue = Literal[
    "STOPPED", "PREFLIGHT", "STARTING", "READY", "DRAINING", "STOPPING", "FAILED", "UNMANAGED"
]

BACKEND_FLASHNEXT = "flashnext"
BACKEND_INLINE = "inline"

#: "BACKOFF=[15,30,60,120,240]s, window 1800s, max 5 attempts." Five values,
#: five allowed restart attempts per window — the crash that would need a
#: SIXTH restart within the window is the one that gets suspended instead
#: (SPEC.md: "Exceeded -> FAILED + suspended=true"). See decide_after_exit().
BACKOFF_S: tuple[float, ...] = (15.0, 30.0, 60.0, 120.0, 240.0)
CRASH_WINDOW_S: float = 1800.0
MAX_ATTEMPTS: int = len(BACKOFF_S)

#: phases.py §5's OWN READY criterion, verbatim: "GET /v1/models -> 200
#: (BOTH required; the HTTP probe is authoritative, same criterion as
#: is_server_up())." NOTE this is deliberately NOT SPEC.md correction C2's
#: `/health` guidance — C2 is about gateway.py's *upstream-liveness* probe
#: (is_server_up(), which must never hang codex's CLI). This module answers
#: a different question — "has THIS boot's phase FSM reached READY" — and
#: phases.py's own docstring says its regexes (and the criterion they
#: complete) are transcribed verbatim from SPEC.md §5 and must not be
#: "improved" independently of it.
READY_PROBE_PATH = "/v1/models"
READY_PROBE_TIMEOUT_S = 2.0
MONITOR_POLL_S = 1.0
DEATH_RECORD_TIMEOUT_S = 15.0
STOP_RESULT_GRACE_S = 3.0

_DESIRED_VERSION = 1


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _parse_iso(ts: str) -> float:
    return datetime.fromisoformat(ts).timestamp()


def _atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


# ---------------------------------------------------------------------------
# state/desired.json — SPEC.md §6's schema, verbatim field set
# ---------------------------------------------------------------------------


@dataclass
class DesiredState:
    version: int = _DESIRED_VERSION
    desired_state: DesiredValue = "STOPPED"
    repo_id: str | None = None
    backend: str | None = None
    served_name: str | None = None
    port: int | None = None
    util: float | None = None
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    auto_restart: bool = True
    suspended: bool = False
    suspended_reason: str | None = None
    #: ISO-8601 timestamps of crash-triggered restarts still inside the
    #: current CRASH_WINDOW_S window (pruned lazily — see prune_attempts()).
    #: NOT user-initiated starts/restarts; those reset it (SPEC.md: a fresh,
    #: deliberate action is not part of an automatic restart chain).
    attempts: list[str] = field(default_factory=list)
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DesiredState":
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        try:
            return cls(**filtered)
        except TypeError:
            return cls()


def _desired_path(state_dir: Path) -> Path:
    return state_dir / "desired.json"


def load_desired(state_dir: Path | None = None) -> DesiredState:
    state_dir = state_dir or paths.STATE_DIR
    try:
        raw = json.loads(_desired_path(state_dir).read_text())
    except (OSError, json.JSONDecodeError):
        return DesiredState()
    if not isinstance(raw, dict):
        return DesiredState()
    return DesiredState.from_dict(raw)


def save_desired(d: DesiredState, state_dir: Path | None = None) -> None:
    state_dir = state_dir or paths.STATE_DIR
    d.updated_at = now_iso()
    _atomic_write_json(_desired_path(state_dir), d.to_dict())


# ---------------------------------------------------------------------------
# state/ack.json — crash-loop-suspension acknowledgement, SPEC.md §6
# ---------------------------------------------------------------------------


@dataclass
class Ack:
    suspended_at: str | None = None
    reason: str | None = None
    #: The ISO timestamps of the attempts that triggered this suspension —
    #: exactly what "[Show all 5]" in SPEC.md's red-card copy needs to render.
    attempts: list[str] = field(default_factory=list)
    acked: bool = False
    acked_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Ack":
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        try:
            return cls(**filtered)
        except TypeError:
            return cls()


def _ack_path(state_dir: Path) -> Path:
    return state_dir / "ack.json"


def load_ack(state_dir: Path | None = None) -> Ack | None:
    state_dir = state_dir or paths.STATE_DIR
    try:
        raw = json.loads(_ack_path(state_dir).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return Ack.from_dict(raw)


def save_ack(a: Ack, state_dir: Path | None = None) -> None:
    state_dir = state_dir or paths.STATE_DIR
    _atomic_write_json(_ack_path(state_dir), a.to_dict())


def clear_ack(state_dir: Path | None = None) -> None:
    state_dir = state_dir or paths.STATE_DIR
    with contextlib.suppress(FileNotFoundError):
        _ack_path(state_dir).unlink()


# ---------------------------------------------------------------------------
# The pure restart decision — SPEC.md §6's "CRASH-WHILE-SERVING vs
# FAILED-BOOT" tree, plus SPEC.md correction C3 (Flash-Next's own
# blocked-needs-human carve-out). No I/O: every fact it needs (gpu_ok,
# xid_blocks_restart, the pruned attempts list, `now`) is collected by the
# caller. This is the function tests target directly for the FSM's core
# behaviour (VERIFY (a)-(d)) — no process, no clock, no disk needed to
# exercise it.
# ---------------------------------------------------------------------------

RestartAction = Literal["none", "restart", "blocked", "suspend"]

#: SPEC.md correction C3: "for BACKEND=flashnext, auto-restart must resolve
#: to blocked-needs-human ... NOT a silent retry loop." serve.sh:53-66
#: relaxes kernel.yama.ptrace_scope via `sudo sysctl`, which silently no-ops
#: without an interactive tty (a systemd/asyncio-launched Servedeck has
#: none) — "it only appears to work right now because ptrace_scope happens
#: to be 0 on this boot." preflight.py's PTRACE_BLOCKS_PLE check already
#: refuses to *launch* flashnext when ptrace_scope != 0, but C3 asks for
#: something stronger for the AUTOMATIC-restart path specifically: never
#: attempt it unattended at all, regardless of ptrace_scope's value at this
#: exact instant, because "it will break after the next reboot" — the
#: current 0 is not an invariant Servedeck can lean on for an unattended
#: retry. A manual Start (this module's start(), called directly, not via
#: the backoff scheduler) is NOT gated by this — only the crash-while-
#: serving auto-restart path is.
FLASHNEXT_HUMAN_GATE_CODE = "FLASHNEXT_NEEDS_HUMAN"


@dataclass(frozen=True)
class RestartDecision:
    action: RestartAction
    delay_s: float | None = None
    code: str = ""
    reason: str = ""


def prune_attempts(attempts: Sequence[str], *, now: float, window_s: float = CRASH_WINDOW_S) -> list[str]:
    """Drop attempt timestamps older than `window_s` relative to `now`.
    Malformed entries are dropped too rather than raising — a corrupt
    desired.json should degrade to "no history", never crash the FSM."""
    out: list[str] = []
    for a in attempts:
        try:
            t = _parse_iso(a)
        except (ValueError, TypeError):
            continue
        if now - t <= window_s:
            out.append(a)
    return out


def decide_after_exit(
    *,
    desired_state: str,
    auto_restart: bool,
    reached_ready: bool,
    backend: str | None,
    gpu_ok: bool,
    xid_blocks_restart: bool,
    xid_note: str | None,
    attempts: Sequence[float],
    now: float,
    window_s: float = CRASH_WINDOW_S,
    backoff: Sequence[float] = BACKOFF_S,
    max_attempts: int = MAX_ATTEMPTS,
) -> RestartDecision:
    """SPEC.md §6's decision tree. `attempts` is the ALREADY-PRUNED list of
    epoch-second timestamps for crash-triggered restarts still inside the
    window (see prune_attempts()) — this function does not re-prune, so a
    caller can pass a synthetic list directly in tests without needing
    ISO-string round-tripping.
    """
    if desired_state != "RUNNING":
        # THE essential rule: a deliberate stop set this to STOPPED BEFORE
        # signalling (Supervisor.stop()), so an exit arriving with
        # desired_state already STOPPED is intent, not a crash. Nothing to
        # decide — there is no restart to schedule.
        return RestartDecision("none", code="DESIRED_STOPPED", reason="desired_state is not RUNNING.")

    if not reached_ready:
        # A boot that never served ANYTHING fails identically every time a
        # blind retry would try it again — SPEC.md's six-different-root-
        # causes-in-one-day evidence is exactly why this is unconditional,
        # not merely the first item on a list of soft preferences.
        return RestartDecision(
            "blocked",
            code="FAILED_BOOT",
            reason="never reached READY — restarting would fail identically; fix the reported error first.",
        )

    if not auto_restart:
        return RestartDecision("blocked", code="AUTO_RESTART_DISABLED", reason="auto_restart is disabled.")

    if not gpu_ok:
        return RestartDecision(
            "blocked",
            code="GPU_UNRESPONSIVE",
            reason="`nvidia-smi -L` failed — GPU unresponsive, no restart until it recovers.",
        )

    if xid_blocks_restart:
        return RestartDecision(
            "blocked", code="XID_NO_RESTART", reason=xid_note or "GPU fault (Xid) is not a restartable pattern."
        )

    # `needs_tty` in servedeck.toml, not a backend name: whether a launcher
    # needs a terminal is a property of that launcher (a `sudo sysctl` that
    # silently no-ops without a tty — SPEC.md correction C3), and the answer
    # for the same backend differs from machine to machine. Falls back to the
    # historical flashnext-only gate when nothing declares the backend, so an
    # install with no config behaves exactly as before.
    _b = config.get().backend(backend)
    if _b.needs_tty if _b is not None else backend == BACKEND_FLASHNEXT:
        return RestartDecision(
            "blocked",
            code=FLASHNEXT_HUMAN_GATE_CODE,
            reason=(
                f"{backend} cannot be auto-restarted unattended (SPEC.md correction C3): "
                "its launcher needs an interactive terminal — e.g. serve.sh relaxes "
                "kernel.yama.ptrace_scope via `sudo sysctl`, which silently fails without "
                "a tty. Start it manually from a terminal instead."
            ),
        )

    pruned = [a for a in attempts if now - a <= window_s]
    if len(pruned) >= max_attempts:
        span_min = (now - min(pruned)) / 60.0 if pruned else 0.0
        return RestartDecision(
            "suspend",
            code="CRASH_LOOP",
            reason=f"{len(pruned)} restarts in {span_min:.0f} min — crash-loop suspended, acknowledgement required.",
        )

    idx = min(len(pruned), len(backoff) - 1)
    return RestartDecision(
        "restart",
        delay_s=backoff[idx],
        code="RESTART",
        reason=f"crash-while-serving — scheduling restart attempt {len(pruned) + 1}/{max_attempts}.",
    )


# ---------------------------------------------------------------------------
# Death classification — systemd-compatible SERVICE_RESULT/EXIT_CODE/
# EXIT_STATUS, for bin/qwen-server-record-death.sh (SPEC.md §6).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExitInfo:
    service_result: str
    exit_code: str
    exit_status: str


def classify_exit(
    *,
    reaped_status: int | None,
    deliberate: bool,
    stop_result: procctl.StopResult | None,
) -> ExitInfo:
    """`reaped_status` is a raw `os.waitpid` status — only available when
    the dead process was genuinely OUR OWN CHILD (procctl.launch() started
    it, and this same Servedeck process is still alive to reap it).
    procctl.py exposes no exit-code API of its own (its ServerHandle drops
    the Popen object once persisted, by design — see its own docstring on
    surviving a Servedeck restart), so an ADOPTED process (found via a port
    probe, not launched by us) can never be reaped here; for those, the
    best available evidence is `stop_result` from our own stop() call, and
    failing that, the honest answer is "unknown" — this never guesses at a
    real exit code it did not actually observe.
    """
    if reaped_status is not None:
        if os.WIFSIGNALED(reaped_status):
            sig = os.WTERMSIG(reaped_status)
            try:
                sig_name = signal.Signals(sig).name
            except ValueError:
                sig_name = str(sig)
            return ExitInfo(
                service_result="success" if deliberate else "signal",
                exit_code="killed",
                exit_status=sig_name,
            )
        if os.WIFEXITED(reaped_status):
            code = os.WEXITSTATUS(reaped_status)
            if deliberate or code == 0:
                # record-death.sh's own NOTE for this exact case (SERVICE_RESULT
                # success + EXIT_STATUS 0) already warns the reader not to read
                # this as "healthy" — vLLM's watchdog_loop exits 0 on EngineCore
                # death by design. Mirrored, not reinvented, here.
                result = "success"
            else:
                result = "exit-code"
            return ExitInfo(service_result=result, exit_code="exited", exit_status=str(code))

    if deliberate:
        if stop_result is not None and stop_result.method in ("sigterm", "sigkill"):
            return ExitInfo(
                service_result="success",
                exit_code="killed",
                exit_status="SIGTERM" if stop_result.method == "sigterm" else "SIGKILL",
            )
        return ExitInfo(service_result="success", exit_code="unknown", exit_status="unknown")

    return ExitInfo(service_result="unknown", exit_code="unknown", exit_status="unknown")


def _read_metric(text: str, name: str) -> float | None:
    """Smallest possible Prometheus text-format scalar reader: the first
    line starting with `name` (optionally followed by a `{...}` label set),
    whitespace, then a float. Good enough for the single-series gauges this
    module reads (vllm:num_requests_running) — SPEC.md §8 confirms the
    metric name; it does not confirm a label set, so this does not assume
    one either way.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith(name):
            continue
        rest = line[len(name) :]
        if rest and rest[0] == "{":
            end = rest.find("}")
            if end == -1:
                continue
            rest = rest[end + 1 :]
        rest = rest.strip()
        try:
            return float(rest.split()[0])
        except (ValueError, IndexError):
            continue
    return None


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


#: One file per boot, under Servedeck's own state dir. _build_launch() writes
#: them; _default_log_paths() reads the newest back when adopting.
BOOT_LOG_DIRNAME = "boot_logs"


def resolve_port(
    backend: str | None, desired: "DesiredState", *,
    explicit: object = None, fallback: int | None = None,
) -> int | None:
    """Which port a start should use.

    The dashboard's Apply button sends a model and a backend but no port, and
    the old fallback chain was ``explicit or desired.port or rt.port`` -- the
    PREVIOUS run's port. Switching backends therefore launched the new backend
    on the old one's port: on this box, picking a Flash-Next model while
    ``desired.json`` still said 8002 started Flash-Next on GLM's port, and the
    gateway (pointed at 8001) reported "backend not reachable" about a server
    that had booted perfectly.

    A backend's port is part of the launch contract it declares in
    servedeck.toml, so that is the answer whenever the backend is changing.
    An explicit port always wins, and a port already chosen for the SAME
    backend is preserved -- otherwise a deliberate one-off move would be
    silently undone on the next restart.
    """
    if explicit:
        return int(explicit)  # type: ignore[arg-type]
    if backend == desired.backend and desired.port:
        return desired.port
    b = config.get().backend(backend)
    if b is not None:
        return b.port
    return desired.port or fallback


def resolve_served_name(
    backend: str | None, repo_id: str | None, desired: "DesiredState", *,
    explicit: str | None = None,
) -> str | None:
    """The ``--served-model-name`` a start should use.

    The dashboard sends no served name either, so the old ``desired.served_name``
    fallback carried the PREVIOUS model's alias onto the new one: launching a
    Qwen checkpoint after a GLM one advertised it over ``/v1/models`` as
    ``glm53-flash``. Nothing downstream could then tell the two apart -- which
    is exactly the "unable to detect the model correctly" report.

    A name chosen for the SAME repo is kept (clients are configured against
    it); a different repo gets its own name, derived from the repo id.
    """
    if explicit:
        return explicit
    if repo_id and repo_id == desired.repo_id and desired.served_name:
        return desired.served_name
    if not repo_id:
        return desired.served_name
    return repo_id.rsplit("/", 1)[-1]


def shell_extra_args(backend: str | None) -> str:
    """``EXTRA_ARGS`` from the shell config, for this backend only.

    ``local_llm/llm start`` exports ``EXTRA_ARGS="${EXTRA_ARGS:-}"`` from
    ``.config`` on every launch, and on this box that string carries flags the
    model does not boot correctly without. Servedeck passed nothing, so a
    launch from the dashboard and a launch from the CLI produced two different
    servers from the same configuration.

    Guarded on ``BACKEND``: ``.config`` describes ONE backend at a time, and
    handing GLM's extra flags to a Qwen launcher is worse than handing it
    none. Never raises -- an unreadable ``.config`` means "no extra args".
    """
    if not backend:
        return ""
    try:
        cfg = shellconfig.read_config()
    except Exception:  # noqa: BLE001 - the shell config is optional
        return ""
    if (cfg.get("BACKEND") or "").strip() != backend:
        return ""
    return (cfg.get("EXTRA_ARGS") or "").strip()


def _default_log_paths(
    backend: str | None, *, boot_log_dir: Path | None = None
) -> list[Path]:
    """Where the named backend's log is, when we did NOT launch it.

    A launch knows its own log path; this is the adoption case. Returning the
    WRONG backend's log is worse than returning nothing: phases.classify()
    would match an error line from another model's run and file it as this
    run's failure_code. So when a backend declares no `log_path`, fall back
    to the newest log Servedeck opened for that backend itself, and to an
    empty list when there is none — a quiet phase machine beats a lying one.
    """
    b = config.get().backend(backend)
    if b is not None and b.log_path is not None:
        return [b.log_path]
    if not backend:
        return []
    d = boot_log_dir if boot_log_dir is not None else paths.STATE_DIR / BOOT_LOG_DIRNAME
    try:
        candidates = sorted(
            d.glob(f"{backend}-*.log"), key=lambda f: f.stat().st_mtime, reverse=True
        )
    except OSError:
        return []
    return candidates[:1]


class Supervisor:
    """The desired/actual state machine for Servedeck's one managed vLLM
    server. Every process/network side effect is reached through one of a
    small number of injectable callables (`launch_fn`, `stop_fn`,
    `scheduler`, `death_exec_fn`) so the FSM itself — the part SPEC.md §6
    cares about being correct — can be driven with fakes in tests, per the
    task brief's VERIFY section, without spawning a real vLLM or waiting
    out a real backoff timer.
    """

    def __init__(
        self,
        *,
        state_dir: Path | None = None,
        clock: Callable[[], float] = time.time,
        launch_fn: Callable[[Sequence[str], Mapping[str, str], str, str], procctl.ServerHandle] = procctl.launch,
        stop_fn: Callable[[procctl.ServerHandle], procctl.StopResult] = procctl.stop,
        scheduler: Callable[[float, Callable[[], Awaitable[None]]], object] | None = None,
        death_exec_fn: Callable[[Mapping[str, str]], None] | None = None,
        history_path: Path | None = None,
    ) -> None:
        self.state_dir = state_dir or paths.STATE_DIR
        self._clock = clock
        self._launch_fn = launch_fn
        self._stop_fn = stop_fn
        self._scheduler = scheduler or self._default_scheduler
        self._death_exec_fn = death_exec_fn or self._exec_record_death
        self._history_path = history_path or history.default_history_path()

        self.desired: DesiredState = load_desired(self.state_dir)
        self.actual_state: ActualValue = "STOPPED"
        self.last_error: str | None = None

        self._handle: procctl.ServerHandle | None = None
        self._tracker: phases.PhaseTracker | None = None
        self._run_started_at: float | None = None
        self._run_repo_id: str | None = None
        self._run_backend: str | None = None
        self._phase_times: dict[str, float] = {}
        self._last_failure: phases.Failure | None = None

        self._monitor_task: "asyncio.Task[None] | None" = None
        self._pending_restart_task: object | None = None
        self._next_restart_at: float | None = None

        self._stopping_deliberately = False
        self._last_stop_result: procctl.StopResult | None = None
        self._unmanaged_pid: int | None = None

    # ----------------------------------------------------------------- #
    # Startup reconciliation — SPEC.md §6, all five cases
    # ----------------------------------------------------------------- #

    async def reconcile_startup(self) -> str:
        """Runs once, when Servedeck itself starts. Returns a short case
        name for logging/tests; state is mutated in place."""
        d = self.desired
        port = d.port
        pid = procctl.listener_pid(port) if port else None
        attributable = pid is not None and procctl.is_attributable(pid)

        if d.desired_state == "RUNNING":
            if pid is not None and attributable:
                self._run_repo_id, self._run_backend = d.repo_id, d.backend
                self._adopt_ready(pid)
                return "adopted_ready"

            if pid is not None and not attributable:
                # "port answers with no attributable pid is UNMANAGED, not
                # guessed at" — procctl.py rule 4, SPEC.md §2.
                self.actual_state = "UNMANAGED"
                self._unmanaged_pid = pid
                return "unmanaged"

            # RUNNING + nothing listening -> preflight -> start. If a prior
            # crash-loop suspension is still on record, honor it here too —
            # otherwise a Servedeck restart would silently resume the exact
            # loop the suspension existed to stop (see acknowledge_and_resume()).
            if d.suspended:
                self.actual_state = "FAILED"
                self.last_error = d.suspended_reason or "auto-restart suspended; acknowledgement required."
                return "suspended_on_reconcile"

            await self.start(
                repo_id=d.repo_id,
                backend=d.backend,
                served_name=d.served_name,
                port=d.port,
                util=d.util,
                max_model_len=d.max_model_len,
                max_num_seqs=d.max_num_seqs,
            )
            return "preflight_start"

        # desired == STOPPED
        if pid is not None:
            # "DO NOTHING. Banner offering Adopt or Stop. Never auto-kill."
            self.actual_state = "STOPPED"
            self._unmanaged_pid = pid
            return "stopped_port_answers"

        self.actual_state = "STOPPED"
        return "stopped_nothing"

    def _adopt_ready(self, pid: int) -> None:
        handle = procctl.load_handle()
        if handle is None or handle.pid != pid:
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                pgid = pid
            handle = procctl.ServerHandle(
                pid=pid, pgid=pgid, argv=[], cwd="", log_path="", started_at=self._clock()
            )
        self._handle = handle
        self.actual_state = "READY"
        self._run_started_at = handle.started_at
        # An adopted server IS serving: it answered /v1/models before we got
        # here. Leaving reached_ready False makes _on_exit take the failed-boot
        # branch, so a crash-while-serving is recorded as "failed_boot" and
        # auto-restart never runs for any adopted server.
        self._tracker = phases.PhaseTracker()
        self._tracker.mark_adopted_ready()
        self._stopping_deliberately = False
        # Tail the real log even for an adopted server. With log_paths=[] there
        # is nothing to classify, so CUDA_FAULT / RUNTIME_OOM -- the only codes
        # whose auto_restart depends on reached_ready -- can never fire and a
        # crash records failure_code: null.
        self._monitor_task = asyncio.create_task(
            self._run_monitor(
                handle,
                log_paths=_default_log_paths(
                    self.desired.backend,
                    boot_log_dir=self.state_dir / BOOT_LOG_DIRNAME,
                ),
                port=self.desired.port or 0,
                already_ready=True,
            )
        )

    # ----------------------------------------------------------------- #
    # start / stop / restart
    # ----------------------------------------------------------------- #

    async def start(
        self,
        *,
        repo_id: str | None,
        backend: str | None,
        served_name: str | None,
        port: int | None,
        util: float | None,
        max_model_len: int | None,
        max_num_seqs: int | None,
        _preserve_attempts: bool = False,
    ) -> None:
        if self.actual_state in ("READY", "STARTING", "PREFLIGHT", "STOPPING"):
            return  # idempotent: already up or already coming up

        # A backend is startable iff servedeck.toml declares it. The gate used
        # to be a literal two-name tuple, which silently refused every backend
        # added by configuration -- the one thing config exists to allow.
        if not repo_id or not backend or not port or config.get().backend(backend) is None:
            self.actual_state = "FAILED"
            self.last_error = (
                "no model/backend/port configured — choose a model first."
                if not (repo_id and backend and port)
                else f"backend {backend!r} is not declared in servedeck.toml — "
                "add a [backends.<name>] section naming its launcher."
            )
            return

        d = self.desired
        d.desired_state = "RUNNING"
        d.repo_id, d.backend, d.served_name, d.port = repo_id, backend, served_name, port
        d.util, d.max_model_len, d.max_num_seqs = util, max_model_len, max_num_seqs
        d.suspended, d.suspended_reason = False, None
        if not _preserve_attempts:
            d.attempts = []
        save_desired(d, self.state_dir)
        self._write_flat_desired_state_file()

        self.actual_state = "PREFLIGHT"
        checks = preflight.run_preflight(
            backend=backend, actual_state=self.actual_state, port=port
        )
        failures = preflight.blocking_failures(checks)
        if failures:
            self.actual_state = "FAILED"
            self.last_error = "; ".join(f"{c.id}: {c.detail}" for c in failures)
            self._record_boot_failure_before_launch(reason="preflight_blocked")
            return

        try:
            self._sync_shell_config(repo_id=repo_id, backend=backend, served_name=served_name, port=port, util=util, max_model_len=max_model_len, max_num_seqs=max_num_seqs)
        except (shellconfig.ShellConfigError, shellconfig.ServerRunningError) as exc:
            self.actual_state = "FAILED"
            self.last_error = f"failed to write local_llm/.config: {exc}"
            self._record_boot_failure_before_launch(reason="config_write_failed")
            return

        self.actual_state = "STARTING"
        self._tracker = phases.PhaseTracker()
        self._phase_times = {}
        self._last_failure = None
        self._run_started_at = self._clock()
        self._run_repo_id, self._run_backend = repo_id, backend
        self._stopping_deliberately = False
        self._last_stop_result = None

        argv, env, cwd, log_paths = self._build_launch(
            backend=backend, repo_id=repo_id, served_name=served_name or repo_id,
            port=port, util=util,
            max_model_len=max_model_len, max_num_seqs=max_num_seqs,
        )
        try:
            handle = self._launch_fn(argv, env, cwd, log_paths[0])
        except OSError as exc:
            self.actual_state = "FAILED"
            self.last_error = f"failed to launch {argv[0]}: {exc}"
            self._record_boot_failure_before_launch(reason="launch_failed")
            return
        self._handle = handle

        self._monitor_task = asyncio.create_task(
            self._run_monitor(handle, log_paths=log_paths, port=port, already_ready=False)
        )

    async def stop(self) -> None:
        # THE essential rule: persist STOPPED before signalling anything.
        d = self.desired
        d.desired_state = "STOPPED"
        d.suspended, d.suspended_reason = False, None
        save_desired(d, self.state_dir)
        self._write_flat_desired_state_file()
        self._stopping_deliberately = True
        self._cancel_pending_restart()

        if self._handle is None:
            self.actual_state = "STOPPED"
            return

        self.actual_state = "STOPPING"
        handle = self._handle
        # procctl.stop() blocks its calling thread for up to ~70s
        # (SIGTERM wait + SIGKILL wait) — run it off the event loop, and
        # don't await it here: the already-running monitor task notices
        # the death (usually within one MONITOR_POLL_S tick, well before
        # procctl.stop()'s own timeout) and does the actual bookkeeping via
        # _on_exit(). Awaiting it here would block every other request
        # this single-process server is handling for up to 70s.
        asyncio.create_task(self._run_stop_signal(handle))

    async def _run_stop_signal(self, handle: procctl.ServerHandle) -> None:
        loop = asyncio.get_running_loop()
        self._last_stop_result = await loop.run_in_executor(None, self._stop_fn, handle)

    async def restart(self, *, mode: str = "immediate") -> None:
        if mode == "blue_green":
            raise NotImplementedError(
                "blue-green execution is a v1 non-goal (SPEC.md §10); use mode='immediate' or 'drain'."
            )
        if mode not in ("immediate", "drain"):
            raise ValueError(f"unknown restart mode {mode!r}")

        d = self.desired
        if mode == "drain" and d.port:
            self.actual_state = "DRAINING"
            await self._drain(d.port)

        await self.stop()
        # stop() leaves actual_state == "STOPPING", and start()'s idempotence
        # guard returns early on any non-STOPPED state -- so without settling
        # to STOPPED here, restart() silently degrades into a plain stop
        # (0 launches). Also restore desired_state: stop() sets it to STOPPED
        # by design, and leaving it there would make the supervisor treat the
        # relaunch as an unwanted server and never auto-restart it.
        self.actual_state = "STOPPED"
        self.desired.desired_state = "RUNNING"
        save_desired(self.desired, self.state_dir)
        self._write_flat_desired_state_file()
        # Give the just-signalled process a moment to actually vacate the port
        # before start()'s preflight runs.
        await asyncio.sleep(0.2)
        await self.start(
            repo_id=d.repo_id, backend=d.backend, served_name=d.served_name, port=d.port,
            util=d.util, max_model_len=d.max_model_len, max_num_seqs=d.max_num_seqs,
        )

    async def _drain(self, port: int, timeout_s: float = 120.0) -> None:
        deadline = self._clock() + timeout_s
        async with httpx.AsyncClient(timeout=3.0) as client:
            while self._clock() < deadline:
                try:
                    resp = await client.get(f"http://127.0.0.1:{port}/metrics")
                    if resp.status_code == 200:
                        running = _read_metric(resp.text, "vllm:num_requests_running")
                        if running is not None and running <= 0.0:
                            return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1.0)

    async def acknowledge_and_resume(self) -> None:
        """SPEC.md §6's "[Resume auto-restart]" action. Requires an explicit
        call (never automatic) — persists the acknowledgement to
        state/ack.json so a page reload does not lose it, clears the
        suspension, resets the crash-attempt window, and retries once
        immediately if desired_state is still RUNNING."""
        ack = load_ack(self.state_dir) or Ack()
        ack.acked, ack.acked_at = True, now_iso()
        save_ack(ack, self.state_dir)

        d = self.desired
        d.suspended, d.suspended_reason = False, None
        d.attempts = []
        save_desired(d, self.state_dir)
        self.last_error = None

        if d.desired_state == "RUNNING" and self.actual_state != "READY":
            await self.start(
                repo_id=d.repo_id, backend=d.backend, served_name=d.served_name, port=d.port,
                util=d.util, max_model_len=d.max_model_len, max_num_seqs=d.max_num_seqs,
            )

    def sweep_orphans(self) -> list[int]:
        """POST /api/server/sweep-orphans (SPEC.md §8). procctl's own
        find_orphaned_engine_cores() is deliberately read-only (its
        docstring says so); the actual SIGKILL has to live somewhere that
        is allowed to signal, which is here."""
        killed: list[int] = []
        for pid in procctl.find_orphaned_engine_cores():
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except (ProcessLookupError, PermissionError):
                continue
        return killed

    # ----------------------------------------------------------------- #
    # Launch construction — SPEC.md §1 "delegation, not reimplementation"
    # ----------------------------------------------------------------- #

    def _sync_shell_config(
        self, *, repo_id: str, backend: str, served_name: str | None, port: int,
        util: float | None, max_model_len: int | None, max_num_seqs: int | None,
    ) -> None:
        """"Restart sequence is always: stop -> write config -> start."
        Called only once actual_state has passed PREFLIGHT (i.e. we already
        know nothing is up), so `server_up=False` is trusted directly
        rather than re-probing (shellconfig.py's own docstring names this
        exact case). Keeps local_llm/.config's view of BACKEND/MODEL_REPO/
        SERVED_NAME/PORT/etc in sync with whichever backend Servedeck is
        actually about to run, for every backend — not just inline —
        because codex-qwen.sh's own status/deaths/base-url logic reads it
        regardless of which backend is live (SPEC.md §9(d)'s use_systemd()
        gate, in particular, depends on .config's BACKEND being accurate).
        """
        for key, value in (
            ("BACKEND", backend),
            ("MODEL_REPO", repo_id),
            ("SERVED_NAME", served_name or repo_id),
            ("PORT", str(port)),
            ("MAX_MODEL_LEN", str(max_model_len) if max_model_len else None),
            ("MAX_NUM_SEQS", str(max_num_seqs) if max_num_seqs else None),
        ):
            if value is not None:
                shellconfig.set_key(key, value)
        if util is not None:
            shellconfig.set_util(util, server_up=False)

    def _build_launch(
        self, *, backend: str, repo_id: str, served_name: str, port: int,
        util: float | None, max_model_len: int | None, max_num_seqs: int | None,
    ) -> tuple[list[str], dict[str, str], str, list[str]]:
        """Returns (argv, env, cwd, log_paths). `log_paths[0]` is what gets
        passed to launch_fn() as the process's own stdout/stderr sink;
        `log_paths` (plural) is every file the boot monitor should tail.

        Servedeck never builds a `vllm serve` command line: it runs the
        launcher named in servedeck.toml and hands it settings through the
        environment (`env_map`), so the launcher keeps owning the flags. Any
        machine-specific tuning the launcher also reads goes in that
        backend's `env` table and is passed through verbatim — deliberately
        NOT interpreted here, because a knob Servedeck understands is a knob
        Servedeck can get wrong.
        """
        b = config.get().backend(backend)
        if b is None:                       # start() already gated on this
            raise ValueError(f"backend {backend!r} is not configured")

        boot_log_dir = self.state_dir / BOOT_LOG_DIRNAME
        boot_log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        # Servedeck's OWN log, never the launcher's conventional one: that
        # file may be a read-only fixture, or a previous boot of a different
        # model, and a fresh launch must not append to or replay either.
        my_log = str(boot_log_dir / f"{backend}-{stamp}.log")

        settings: dict[str, str] = {
            "repo_id": repo_id,
            "port": str(port),
            "max_model_len": str(max_model_len or ""),
            "util": f"{util:.6g}" if util is not None else "",
            "max_num_seqs": str(max_num_seqs or ""),
            "served_name": served_name,
            "kv_dtype": "auto",
            "extra_args": shell_extra_args(backend),
        }
        env = {
            var: settings[key]
            for key, var in b.env_map.items()
            if key in settings and settings[key] != ""
        }
        # EXTRA_ARGS is exported whether or not env_map names it, exactly as
        # `local_llm/llm start` does: it is the launcher's own verbatim
        # passthrough slot, and a launcher that does not read it ignores it.
        # Dropping it made a dashboard launch boot a DIFFERENT configuration
        # from the CLI's -- on this box that silently lost
        # `--language-model-only --mamba-ssm-cache-dtype bfloat16
        # --prefix-match-unit 208` for Flash-Next.
        extra_var = b.env_map.get("extra_args", "EXTRA_ARGS")
        if settings["extra_args"] and extra_var not in env:
            env[extra_var] = settings["extra_args"]
        env.update(b.env)
        # SPEC.md correction C9: "Add HF_HUB_OFFLINE=1 to the unit
        # Environment — verified to remove the DNS class" (17 of 72 recorded
        # exits). procctl.launch() merges this onto (not over) the current
        # environment, same semantics as the `env KEY=VAL cmd` prefix
        # SPEC.md §1 describes.
        env["HF_HUB_OFFLINE"] = "1"

        # A launcher that redirects into its own log stops writing to the pipe
        # we opened partway through the boot, so both files have to be tailed.
        log_paths = [my_log]
        if b.writes_own_log and b.log_path is not None:
            log_paths.append(str(b.log_path))
        # run_cwd, not launcher.parent: a launcher under bin/ resolves its own
        # .config, run/ and logs/ relative to the PROJECT root, and running it
        # from bin/ makes it create a second, empty state tree there.
        return [str(b.launcher)], env, str(b.run_cwd), log_paths

    # ----------------------------------------------------------------- #
    # Boot / liveness monitor
    # ----------------------------------------------------------------- #

    async def _run_monitor(
        self, handle: procctl.ServerHandle, *, log_paths: list[str], port: int, already_ready: bool
    ) -> None:
        # Always tail. For an adopted server start at EOF so we see new lines
        # (crashes) without replaying prior boots into the phase FSM.
        tailers = [logtail.LogTailer(p, from_end=already_ready) for p in log_paths]
        client: httpx.AsyncClient | None = None if already_ready else httpx.AsyncClient(timeout=READY_PROBE_TIMEOUT_S)
        try:
            while True:
                reaped = self._try_reap(handle.pid)
                if reaped is not None:
                    await self._on_exit(handle, reaped_status=reaped)
                    return
                if not self._pid_alive(handle.pid):
                    await self._on_exit(handle, reaped_status=None)
                    return

                # NOT gated on already_ready: an adopted server still needs its
                # log tailed so crashes get classified. Feeding a tracker that
                # is already READY is harmless - the FSM never regresses.
                if self._tracker is not None:
                    for tailer in tailers:
                        result = tailer.poll()
                        if result.rotated or result.truncated:
                            # A fresh boot's log rotated under us (or was
                            # truncated in place) — logtail.py's own
                            # contract: start a fresh phase FSM.
                            self._tracker = phases.PhaseTracker()
                            self._phase_times = {}
                        for line in result.lines:
                            self._feed_line(line)

                    # An adopted server has no probe client: it answered
                    # /v1/models before we ever got here, so READY is already
                    # established and there is nothing for the probe to
                    # decide. This used to be `assert client is not None`,
                    # which is reachable for exactly that case — the guard
                    # above is on `self._tracker`, which _adopt_ready() has
                    # just set. The assertion fired on tick 1, the
                    # fire-and-forget task swallowed it, and liveness was
                    # never polled again: the UI reported READY for a server
                    # whose port was dead.
                    if client is not None:
                        probe_ok = await self._probe_ready(client, port)
                        event = self._tracker.set_http_probe_ok(probe_ok)
                        if event is not None:
                            self._on_reached_ready()
                            already_ready = True

                await asyncio.sleep(MONITOR_POLL_S)
        finally:
            if client is not None:
                await client.aclose()

    async def _probe_ready(self, client: httpx.AsyncClient, port: int) -> bool:
        try:
            resp = await client.get(f"http://127.0.0.1:{port}{READY_PROBE_PATH}")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def _feed_line(self, line: str) -> None:
        assert self._tracker is not None
        self._tracker.feed(line)
        if self._run_started_at is not None:
            idx = self._tracker.phase_index
            if idx >= 0:
                phase_name = phases.PHASE_ORDER[idx].value
                # First arrival only — SPEC.md §5: "Timing uses WALL-CLOCK
                # arrival at the tailer", and history.py's ETA stats want
                # one number per phase per run, not the last line matched.
                self._phase_times.setdefault(phase_name, self._clock() - self._run_started_at)
        failure = phases.classify(line, reached_ready=self._tracker.reached_ready)
        if failure is not None and failure.code != "INFORMATIONAL":
            self._last_failure = failure

    def _on_reached_ready(self) -> None:
        self.actual_state = "READY"
        self.last_error = None
        # Deliberately does NOT clear desired.attempts: the crash-window is
        # time-based (CRASH_WINDOW_S), not reset-on-success — a server that
        # crashes, restarts, serves for two minutes, then crashes again is
        # exactly the pattern the ceiling exists to catch. See
        # decide_after_exit()'s docstring.

    # ----------------------------------------------------------------- #
    # Exit handling — the crash-while-serving / failed-boot / deliberate-
    # stop fork, backoff scheduling, crash-loop suspension, death recording
    # ----------------------------------------------------------------- #

    async def _on_exit(self, handle: procctl.ServerHandle, *, reaped_status: int | None) -> None:
        reached_ready = bool(self._tracker and self._tracker.reached_ready)
        now = self._clock()
        deliberate = self._stopping_deliberately

        if deliberate and self._last_stop_result is None:
            # Small grace window for _run_stop_signal()'s background
            # procctl.stop() call to populate the nicer SIGTERM/SIGKILL
            # detail before we classify — see stop()'s own comment on why
            # that call isn't awaited directly. Harmless either way:
            # classify_exit() still reports SERVICE_RESULT=success from
            # `deliberate` alone if this races past the grace window.
            grace_deadline = self._clock() + STOP_RESULT_GRACE_S
            while self._last_stop_result is None and self._clock() < grace_deadline:
                await asyncio.sleep(0.1)

        exit_info = classify_exit(reaped_status=reaped_status, deliberate=deliberate, stop_result=self._last_stop_result)

        total_s = (now - self._run_started_at) if (reached_ready and self._run_started_at is not None) else None
        repo_id, backend = self._run_repo_id, self._run_backend
        cold = (
            not history.has_prior_success(
                repo_id, backend, before_ts=_iso(self._run_started_at) if self._run_started_at else None,
                path=self._history_path,
            )
            if repo_id and backend
            else False
        )

        if deliberate:
            outcome = "stopped_by_user"
        elif not reached_ready:
            outcome = "failed_boot"
        else:
            outcome = "crashed"

        record: dict[str, Any] = {
            "repo_id": repo_id,
            "backend": backend,
            "started_at": _iso(self._run_started_at) if self._run_started_at else None,
            "reached_ready": reached_ready,
            "cold": cold,
            "total_s": total_s,
            "phases": dict(self._phase_times),
            "outcome": outcome,
            "failure_code": self._last_failure.code if self._last_failure else None,
            "exit": {
                "service_result": exit_info.service_result,
                "exit_code": exit_info.exit_code,
                "exit_status": exit_info.exit_status,
            },
        }

        self._handle = None
        self._monitor_task = None

        if deliberate:
            self.actual_state = "STOPPED"
            self.last_error = None
            record["restart_scheduled_s"] = None
            record["attempt_in_window"] = None
            self._record_death(record)
            return

        if not reached_ready:
            self.actual_state = "FAILED"
            self.last_error = self._describe_boot_failure()
            record["failure_code"] = self._last_failure.code if self._last_failure else record["failure_code"]
            record["restart_scheduled_s"] = None
            record["attempt_in_window"] = None
            self._record_death(record)
            return

        # Crash-while-serving: gather the facts decide_after_exit() needs.
        gpu_ok = gpu.gpu_alive()
        xids = gpu.xid_events(since=f"@{int(self._run_started_at)}") if self._run_started_at else []
        xid_blocks = any(not e.restartable for e in xids)
        xid_note = xids[-1].note if xids else None

        pruned_iso = prune_attempts(self.desired.attempts, now=now, window_s=CRASH_WINDOW_S)
        decision = decide_after_exit(
            desired_state=self.desired.desired_state,
            auto_restart=self.desired.auto_restart,
            reached_ready=True,
            backend=backend,
            gpu_ok=gpu_ok,
            xid_blocks_restart=xid_blocks,
            xid_note=xid_note,
            attempts=[_parse_iso(a) for a in pruned_iso],
            now=now,
        )

        record["restart_scheduled_s"] = decision.delay_s
        record["attempt_in_window"] = (len(pruned_iso) + 1) if decision.action == "restart" else None
        if decision.action == "suspend":
            record["outcome"] = "suspended"
        self._record_death(record)

        self.last_error = decision.reason
        if decision.action == "restart":
            assert decision.delay_s is not None
            self.actual_state = "FAILED"  # waiting to retry; see next_restart_at in snapshot()
            self.desired.attempts = pruned_iso + [_iso(now)]
            save_desired(self.desired, self.state_dir)
            self._schedule(decision.delay_s, self._do_scheduled_restart)
            return

        self.actual_state = "FAILED"
        if decision.action in ("suspend",) or decision.code == FLASHNEXT_HUMAN_GATE_CODE:
            self.desired.suspended = True
            self.desired.suspended_reason = decision.reason
            save_desired(self.desired, self.state_dir)
            save_ack(
                Ack(suspended_at=_iso(now), reason=decision.reason, attempts=pruned_iso, acked=False),
                self.state_dir,
            )

    async def _do_scheduled_restart(self) -> None:
        self._pending_restart_task = None
        self._next_restart_at = None
        d = self.desired
        if d.desired_state != "RUNNING" or d.suspended:
            # A Stop click (or a suspension from elsewhere) arrived during
            # the backoff wait — honor it, never resurrect.
            return
        await self.start(
            repo_id=d.repo_id, backend=d.backend, served_name=d.served_name, port=d.port,
            util=d.util, max_model_len=d.max_model_len, max_num_seqs=d.max_num_seqs,
            _preserve_attempts=True,
        )

    def _describe_boot_failure(self) -> str:
        if self._last_failure is not None:
            f = self._last_failure
            msg = f"{f.code}: {f.line.strip()}"
            if f.hint:
                msg += f" ({f.hint})"
            return msg
        return "boot never reached READY (no specific error line matched)."

    def _record_boot_failure_before_launch(self, *, reason: str) -> None:
        """A preflight/config-write/launch-exec failure — never even got a
        PID, so there is nothing to death-watch, but it is still a boot
        attempt that belongs in history (SPEC.md §4: "written ... after a
        boot failing at KV, which still yields weights" — the general
        principle that a failed attempt is itself data, not noise)."""
        now = self._clock()
        record = {
            "repo_id": self.desired.repo_id,
            "backend": self.desired.backend,
            "started_at": _iso(now),
            "reached_ready": False,
            "cold": False,
            "total_s": None,
            "phases": {},
            "outcome": "failed_boot",
            "failure_code": reason.upper(),
            "restart_scheduled_s": None,
            "attempt_in_window": None,
            "exit": {"service_result": "unknown", "exit_code": "unknown", "exit_status": "unknown"},
        }
        history.append(record, path=self._history_path)

    # ----------------------------------------------------------------- #
    # Death recording — SPEC.md §6: "append state/history.jsonl AND exec
    # bin/qwen-server-record-death.sh ... so `./codex-qwen.sh deaths`
    # keeps working (one death record)."
    # ----------------------------------------------------------------- #

    def _record_death(self, record: Mapping[str, Any]) -> None:
        history.append(dict(record), path=self._history_path)
        exit_info = record.get("exit") or {}
        self._death_exec_fn(
            {
                "SERVICE_RESULT": str(exit_info.get("service_result", "unknown")),
                "EXIT_CODE": str(exit_info.get("exit_code", "unknown")),
                "EXIT_STATUS": str(exit_info.get("exit_status", "unknown")),
            }
        )

    def _exec_record_death(self, env: Mapping[str, str]) -> None:
        script = paths.RECORD_DEATH_SH
        if not (script.is_file() and os.access(script, os.X_OK)):
            return  # best-effort; a missing/non-executable script must never break exit handling
        full_env = {**os.environ, **env}
        try:
            subprocess.run(
                [str(script)], env=full_env, cwd=str(paths.LOCAL_LLM),
                timeout=DEATH_RECORD_TIMEOUT_S, check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    # ----------------------------------------------------------------- #
    # Scheduling
    # ----------------------------------------------------------------- #

    def _schedule(self, delay_s: float, fn: Callable[[], Awaitable[None]]) -> None:
        self._pending_restart_task = self._scheduler(delay_s, fn)
        self._next_restart_at = self._clock() + delay_s

    def _default_scheduler(self, delay_s: float, fn: Callable[[], Awaitable[None]]) -> "asyncio.Task[None]":
        async def _runner() -> None:
            try:
                await asyncio.sleep(delay_s)
                await fn()
            finally:
                self._pending_restart_task = None
                self._next_restart_at = None

        return asyncio.create_task(_runner())

    def _cancel_pending_restart(self) -> None:
        task = self._pending_restart_task
        if task is not None and hasattr(task, "cancel"):
            task.cancel()  # type: ignore[attr-defined]
        self._pending_restart_task = None
        self._next_restart_at = None

    # ----------------------------------------------------------------- #
    # Liveness primitives — no procctl.py equivalent exists (its
    # ServerHandle deliberately drops the Popen object once persisted), so
    # this module implements its own, using only os.kill/os.waitpid —
    # never a pattern-matching process-table tool (SPEC.md §2 rule 1).
    # ----------------------------------------------------------------- #

    @staticmethod
    def _try_reap(pid: int) -> int | None:
        try:
            reaped_pid, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return None  # not our child (e.g. adopted) — nothing to reap
        return status if reaped_pid == pid else None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    # ----------------------------------------------------------------- #
    # C4 prep — see decide_after_exit()'s neighbours for the rationale
    # ----------------------------------------------------------------- #

    def _write_flat_desired_state_file(self) -> None:
        """SPEC.md correction C4: qwen-server-run.sh's proposed guard 0
        (`desired_state == stopped -> exit 69`, §9 territory — NOT owned by
        this file) needs a plain file it can `cat` without JSON parsing.
        "Store desired_state in run/desired_state on ext4 (survives
        reboot). Not /run/user/1000 (tmpfs), not systemd is-enabled."
        Writing this now costs nothing and makes that shell patch, whenever
        it lands, a pure read against an already-correct file.
        """
        try:
            paths.RUN_DIR.mkdir(parents=True, exist_ok=True)
            target = paths.RUN_DIR / "desired_state"
            fd, tmp_name = tempfile.mkstemp(dir=str(paths.RUN_DIR), prefix=".desired_state-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write("running\n" if self.desired.desired_state == "RUNNING" else "stopped\n")
                os.replace(tmp_name, target)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_name)
                raise
        except OSError:
            pass  # best-effort; never block state transitions on this

    # ----------------------------------------------------------------- #
    # Snapshot for the API layer (SPEC.md §8 GET /api/state)
    # ----------------------------------------------------------------- #

    def snapshot(self) -> dict[str, Any]:
        now = self._clock()
        return {
            "desired_state": self.desired.desired_state,
            "actual_state": self.actual_state,
            "repo_id": self.desired.repo_id,
            "backend": self.desired.backend,
            "served_name": self.desired.served_name,
            "port": self.desired.port,
            "util": self.desired.util,
            "max_model_len": self.desired.max_model_len,
            "max_num_seqs": self.desired.max_num_seqs,
            "auto_restart": self.desired.auto_restart,
            "suspended": self.desired.suspended,
            "suspended_reason": self.desired.suspended_reason,
            "attempts_in_window": len(prune_attempts(self.desired.attempts, now=now)),
            "last_error": self.last_error,
            "phase": (self._tracker.phase.value if (self._tracker and self._tracker.phase) else None),
            "reached_ready": bool(self._tracker and self._tracker.reached_ready),
            "next_restart_at": self._next_restart_at,
            "unmanaged_pid": self._unmanaged_pid,
        }
