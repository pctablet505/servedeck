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
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

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
    "SECRET_NAME_RE",
    "UNIT_PREFIX",
    "DEFAULT_MARGIN_MIB",
    "floor2",
]

log = logging.getLogger(__name__)


def _argv_int(argv: Sequence[str], flag: str) -> int | None:
    """The int value of ``flag`` in an argv list, or None."""
    value = _argv_str(argv, flag)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _spec_ctx(spec: Any) -> int | None:
    """The registry's context length for a model, when it is a number.

    ``models.toml`` allows ``ctx = "native"``, which means "whatever the
    checkpoint says" and is not a measurement key.
    """
    for attr in ("ctx_tokens", "ctx"):
        value = getattr(spec, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _argv_str(argv: Sequence[str], flag: str) -> str | None:
    """The value of ``flag`` in an argv list, or None if absent or last."""
    for i, token in enumerate(argv):
        if token == flag and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


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

#: A co-resident model's own CUDA-context headroom, on top of the fraction of
#: the card its budget derives. The global margin is the MAIN model's cushion
#: and is already spent by the time a resident starts, so charging the
#: resident for it as well refused every co-resident launch — which is the
#: only kind a resident has.
RESIDENT_CUSHION_MIB = 512

#: Environment variable names that must never reach a model process.
#:
#: A transient unit inherits the user manager's environment, which on a
#: workstation is the login session's — so an operator who exported an API key
#: in their ~/.profile has put it in every model's environment. Measured on
#: this box: ``KITE_API_KEY`` and ``KITE_API_SECRET`` are both in
#: ``systemctl --user show-environment``. A model runs arbitrary prompts, logs
#: freely, and can simply be asked to print its own environment, so the only
#: safe amount of unrelated credential in it is none.
#:
#: The word must be a whole underscore-delimited component, which is what
#: keeps ``MONKEY_BUSINESS``, ``KEYBOARD_LAYOUT`` and ``PASSTHROUGH`` out of
#: the list while catching ``KITE_API_KEY``, ``GITHUB_TOKEN``, ``PASSWORD``
#: and ``GOOGLE_APPLICATION_CREDENTIALS``. Over-redaction is not free: an
#: unset variable a model needed is a boot failure with no message.
SECRET_NAME_RE = re.compile(
    r"(^|_)(KEY|SECRET|TOKEN|PASS|PASSWORD|CREDENTIALS?)($|_)", re.IGNORECASE
)

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
    #: ``"main"`` (exclusive GPU slot, one at a time) or ``"resident"``
    #: (co-resident, budgeted, always on).
    slot: str
    #: Loopback port this model's own vLLM listens on. Clients never see it;
    #: only the gateway and this module do.
    port: int
    #: For a resident: its VRAM budget in MiB, from which its utilisation is
    #: derived. For a main model: None — a main model gets everything free.
    vram_mib: int | None
    #: Context length in tokens. REQUIRED, never None: "native" is resolved to
    #: a number by the registry adapter before launch, and ``render_argv`` must
    #: always emit ``--max-model-len``. Leaving it to vLLM means the dashboard
    #: cannot size KV headroom and two callers disagree about the context a
    #: client may ask for — R1 in miniature.
    ctx_tokens: int
    #: Absolute path to the venv's ``bin`` DIRECTORY (e.g.
    #: ``/home/u/Projects/local_llm/.venv-llm-029/bin``). Control prepends it
    #: to ``PATH`` in the unit environment so every subprocess vLLM spawns
    #: resolves to the same interpreter. ``render_argv`` is still expected to
    #: return an absolute argv[0]; PATH is for vLLM's children, not for us.
    venv_bin: str

    def served_names(self) -> Sequence[str]:
        """Every name vLLM is told to serve (``--served-model-name``): the id
        plus every alias any client has ever used. Never renamed, only added to.

        A METHOD, not an attribute, and deliberately so: P1's real model object
        already exposes it as a method, and a Protocol that declared it an
        attribute would still pass ``isinstance`` against that object — then
        blow up inside :meth:`Control.wait_ready` with "method object is not
        iterable", at model-boot time, on the box. A conformance check that
        accepts the shape it will later choke on is worse than none.
        """
        ...

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
    """What the supervisor needs from the registry: a lookup and the key list.

    There is no ``models()`` here. Every action entry point is given a key.
    ``keys()`` exists for exactly one reader, :meth:`Control.live`'s adoption
    of an unmanaged listener (2026-09-17): a registered model may be serving
    on its port without a ``model-*`` unit — the process v1 launched at
    cutover, or a hand launch — and the only way to notice is to probe the
    registered ports. Order carries no meaning here.
    """

    def get(self, key: str) -> ModelSpec:
        """The spec for ``key``. Raises :class:`KeyError` if unknown."""
        ...

    def keys(self) -> Sequence[str]:
        """Every registry key, for port adoption. Order is not significant."""
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
    #: True when no unit owns this model but its registered port answers
    #: ``/v1/models`` with its id: a process servedeck did not launch (v1's,
    #: or a hand launch). Routed and stoppable (by pid), never restarted.
    adopted: bool = False


#: The ``unit`` an adopted model reports. Not a systemd name on purpose.
ADOPTED_UNIT = "(adopted)"


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
    #: KV offload buffers removed from /dev/shm after the engine exited
    #: (``(path, bytes)``); non-empty means it did not clean up after itself.
    reaped: tuple[tuple[str, int], ...] = ()


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


#: vLLM's CPU KV offload buffer (vllm/v1/kv_offload/cpu/shared_offload_region.py):
#: a named file in /dev/shm — host RAM — sized by --kv-offloading-size.
OFFLOAD_REGION_GLOB = "vllm_offload_*.mmap"


@dataclass(frozen=True)
class OffloadScan:
    """Which offload buffers are mapped, and whether ``/proc`` could be read.

    ``trusted`` is the whole point: an unreadable holder is indistinguishable
    from no holder, and "no holder" means "delete 40 GiB". Untrusted scans
    delete nothing and say why.
    """

    paths: frozenset[str] = frozenset()
    trusted: bool = True
    why: str = ""


def _engine_pids(proc: Path) -> tuple[list[Path], bool]:
    """Every vLLM process's ``/proc`` dir, and whether the listing worked.

    Only a vLLM process can hold a vLLM offload buffer (the engine, its
    ``VLLM::EngineCore`` and its ``VLLM::Worker`` children all carry the name
    in their argv), so only those are read. Reading ``maps`` for all ~500
    processes on the box every poll would cost far more than it proves.
    """
    try:
        pids = [p for p in proc.iterdir() if p.name.isdigit()]
    except OSError:
        return [], False
    out: list[Path] = []
    for p in pids:
        try:
            cmdline = (p / "cmdline").read_bytes()
        except OSError:
            continue  # the process exited, or it is not ours: not an engine we launched
        if b"vllm" in cmdline.lower():
            out.append(p)
    return out, True


def offload_scan(
    proc: Path = Path("/proc"),
    witness_pids: Collection[int] = (),
    engine_pids: Callable[[Path], tuple[list[Path], bool]] = _engine_pids,
) -> OffloadScan:
    """The offload buffers vLLM processes map or hold open.

    Every pid in ``witness_pids`` (the pids of the models servedeck knows are
    live) must be readable, and so must every vLLM process found in ``/proc``.
    ``kernel.yama.ptrace_scope`` decides whether we can read a sibling
    engine's ``maps`` at all — it is 0 on this box for Flash-Next's PLE
    handoff, not for this — so the scan states its own trustworthiness
    instead of letting a permission error read as an absent holder.
    """
    dirs, listed = engine_pids(proc)
    if not listed:
        return OffloadScan(trusted=False, why=f"{proc} could not be listed")
    seen = {int(d.name) for d in dirs}
    for pid in witness_pids:
        if pid not in seen:
            path = proc / str(pid)
            if path.exists():
                dirs.append(path)
    paths: set[str] = set()
    for d in dirs:
        read_something = exited = False
        try:
            for line in (d / "maps").read_text().splitlines():
                parts = line.split(None, 5)
                if len(parts) == 6:
                    paths.add(parts[5].removesuffix(" (deleted)"))
            read_something = True
        except FileNotFoundError:
            exited = True  # it went away while we walked: it holds nothing now
        except OSError:
            pass  # a permission error, which is the case this guard is for
        try:
            for fd in (d / "fd").iterdir():
                try:
                    paths.add(os.readlink(fd))
                except OSError:
                    pass
            read_something = True
        except OSError:
            pass
        if not (read_something or exited):
            return OffloadScan(
                trusted=False,
                why=f"/proc/{d.name} (a vLLM process) could not be read; "
                    "kernel.yama.ptrace_scope hides it",
            )
    return OffloadScan(paths=frozenset(paths), trusted=True)


def reap_offload_regions(
    shm_dir: Path = Path("/dev/shm"),
    scan: Callable[[], OffloadScan] = offload_scan,
) -> list[tuple[str, int]]:
    """Delete KV-offload buffers that no vLLM process maps or holds open.

    vLLM unlinks the buffer only in its graceful ``cleanup()``. An engine that
    is SIGKILLed, crashes, or loses the GPU (Xid 79/154 on this card) leaves
    the whole ``--kv-offloading-size`` (40 GiB for Flash-Next) pinned in tmpfs
    until reboot, and every relaunch adds another under a fresh engine id.
    A buffer some process still maps is live and is never touched; neither is
    any buffer at all when the scan could not see every engine.
    Returns ``(path, bytes)`` for what was removed.
    """
    try:
        candidates = sorted(shm_dir.glob(OFFLOAD_REGION_GLOB))
    except OSError:
        return []
    if not candidates:
        return []
    result = scan()
    if not result.trusted:
        log.warning(
            "not reaping %d KV offload buffer(s): %s", len(candidates), result.why or "unknown"
        )
        return []
    removed: list[tuple[str, int]] = []
    for path in candidates:
        if str(path) in result.paths:
            continue
        try:
            size = path.stat().st_size
            path.unlink()
        except OSError:
            continue
        log.warning("reaped orphaned KV offload buffer %s (%.1f GiB)", path, size / 2**30)
        removed.append((str(path), size))
    return removed


def apply_argv_overrides(argv: list[str], overrides: Mapping[str, str | None] | None) -> list[str]:
    """Replace a flag's value in ``argv`` (or append the pair when absent).
    A value of ``None`` removes the flag and its value."""
    if not overrides:
        return argv
    out = list(argv)
    for flag, value in overrides.items():
        for i, a in enumerate(out):
            if a == flag and i + 1 < len(out):
                if value is None:
                    del out[i : i + 2]
                else:
                    out[i + 1] = str(value)
                break
        else:
            if value is not None:
                out += [flag, str(value)]
    return out


def listener_pid(port: int) -> int | None:
    """The pid listening on ``127.0.0.1:port``/``0.0.0.0:port``, from ``ss``."""
    try:
        out = subprocess.run(
            ["ss", "-Hltnp", f"sport = :{port}"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"pid=(\d+)", out)
    return int(m.group(1)) if m else None


def _cmdline_of(pid: int) -> str:
    """``/proc/<pid>/cmdline``, flattened and clipped — for naming the process
    that is holding a port in a refusal an operator has to act on."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return " ".join(raw.decode("utf-8", "replace").split("\0")).strip()[:120]


def mem_available_gib(meminfo: Path = Path("/proc/meminfo")) -> float | None:
    """MemAvailable + SwapFree, in GiB. None when /proc/meminfo is unreadable.

    Both terms, because a box WITH swap can survive an overshoot; this one has
    none, which is why the guard that uses this exists at all.
    """
    try:
        text = meminfo.read_text(encoding="utf-8")
    except OSError:
        return None
    kib = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        if name in ("MemAvailable", "SwapFree"):
            try:
                kib[name] = int(rest.split()[0])
            except (IndexError, ValueError):
                return None
    if "MemAvailable" not in kib:
        return None
    return (kib["MemAvailable"] + kib.get("SwapFree", 0)) / 1048576


def descendants(pid: int) -> list[int]:
    """``pid``'s descendants from ``/proc/*/task/*/children`` (empty off Linux)."""
    out: list[int] = []
    stack = [pid]
    while stack:
        p = stack.pop()
        try:
            for task in Path(f"/proc/{p}/task").iterdir():
                kids = (task / "children").read_text().split()
                for k in kids:
                    child = int(k)
                    out.append(child)
                    stack.append(child)
        except OSError:
            continue
    return out


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
        listener_pid: Callable[[int], int | None] | None = None,
        kill: Callable[[int, int], None] | None = None,
        descendants: Callable[[int], list[int]] | None = None,
        reap_offload: Callable[[], list[tuple[str, int]]] | None = None,
        mem_available_gib_fn: Callable[[], float | None] | None = None,
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
        self._listener_pid = listener_pid or globals()["listener_pid"]
        self._kill = kill or os.kill
        self._descendants = descendants or globals()["descendants"]
        self._reap_offload = reap_offload or reap_offload_regions
        self._mem_available_gib = mem_available_gib_fn or mem_available_gib

    def offload_witnesses(self, live: Sequence[LiveModel] | None = None) -> list[int] | None:
        """The pids that must be visible before a buffer counts as orphaned.

        ``None`` means servedeck believes a model is live but cannot list its
        processes, which is a reason to reap nothing at all.
        """
        pids: list[int] = []
        for model in self.live() if live is None else live:
            if model.state != "active" and model.sub_state != "adopted":
                continue
            if model.adopted:
                if not model.pid:
                    return None
                pids.append(model.pid)
                pids.extend(self._descendants(model.pid))
                continue
            found = self._cgroup_pids(model.unit)
            if not found:
                return None
            pids.extend(found)
        return pids

    def reap_offload(self, live: Sequence[LiveModel] | None = None) -> list[tuple[str, int]]:
        """Remove KV offload buffers no vLLM process maps (see reap_offload_regions)."""
        witnesses = self.offload_witnesses(live)
        if witnesses is None:
            log.warning(
                "not reaping KV offload buffers: servedeck cannot list the processes of a "
                "model it believes is live"
            )
            return []
        return self._reap_offload(scan=lambda: offload_scan(witness_pids=witnesses))

    # -- small helpers ----------------------------------------------------

    def unit_for(self, key: str) -> str:
        return f"{self.unit_prefix}{key}"

    def key_for(self, unit: str) -> str:
        prefix = self.unit_prefix
        return unit[len(prefix) :] if unit.startswith(prefix) else unit

    def secret_env_names(self, keep: Collection[str] = ()) -> list[str]:
        """Names in the user manager's environment that a model must not see.

        Computed at start time rather than written down, because the manager's
        environment is whatever the login session put there and changes without
        anyone editing this repo — a hardcoded deny-list would be correct on
        the day it was written and quietly wrong afterwards.

        ``keep`` is the model's own declared environment. The registry is the
        single source of truth for what a model needs (R1), so a variable it
        declares on purpose — an ``HF_TOKEN`` for a gated repo — is the
        model's, not ambient leakage, and is left alone. Only what the model
        never asked for is redacted.

        Returns NAMES. No value is read, stored, returned or logged anywhere on
        this path: :func:`units.manager_environment_names` discards the value
        half before returning, so there is nothing here to leak.
        """
        keep = set(keep)
        return [
            name
            for name in units.manager_environment_names(run=self._run)
            if SECRET_NAME_RE.search(name) and name not in keep
        ]

    def total_mib(self) -> int | None:
        if self._total_mib is not None:
            return self._total_mib
        return gpu.total_mib()

    def _record_boot(
        self,
        key: str,
        spec: Any,
        unit: str,
        argv: Sequence[str],
        since: str,
    ) -> None:
        """Re-fit this model's capacity constants from the boot that just came up.

        Best-effort by construction: a model is serving by the time this runs,
        and no failure to *measure* it may turn into a failure to *start* it.
        Every branch that cannot produce a number returns instead of guessing,
        because a fabricated observation is worse than none — the resolver
        picks the most recent entry, so one bad record shadows every good one.
        """
        try:
            from servedeck import bootfacts as _bootfacts
            from servedeck import capacity as _capacity
            from servedeck import discovery as _discovery

            # Anchored at this unit's own start, not a tail: a tail's line
            # count would have to cover both the chattiest possible boot
            # (torch.compile plus 97 FlashInfer JIT objects) and any traffic
            # that arrived before this ran. Measured against the live unit, a
            # 2000-line tail found nothing at all.
            lines = units.journal_since(unit, since, run=self._run)
            facts = _bootfacts.parse(lines)
            if facts.kv_tokens is None and facts.kv_gib is None:
                log.debug("boot of %s printed no KV figures; not recorded", key)
                return

            repo_id = getattr(spec, "repo", None) or getattr(spec, "id", key)
            # The backend string the RESOLVER keys on, which is the one
            # discover_models() derives from the hub cache -- not models.toml's
            # `build`. They differ here: the hub says "flashnext" where the
            # registry says "qwen38next", and resolve_inputs filters
            # observations by the hub's value. An entry tagged with the other
            # one is written, kept, and then silently skipped at read time,
            # which is indistinguishable from never having measured at all.
            backend = next(
                (
                    e.backend
                    for e in _discovery.discover_models()
                    if e.repo_id == repo_id and e.backend
                ),
                getattr(spec, "build", None) or "",
            )
            store = _discovery.load_observations()
            # Weights are a property of the checkpoint, not of the launch, and
            # this boot does not print them in a form worth re-deriving. Carry
            # the last measured value for this repo forward so the overhead
            # residual is fitted against the same weights the panel predicts
            # with; without it the residual would absorb the weights error too.
            weights_gib = next(
                (
                    (o.get("measured") or {}).get("weights_gib")
                    for o in reversed(store)
                    if o.get("repo_id") == repo_id
                    and (o.get("measured") or {}).get("weights_gib") is not None
                ),
                None,
            )
            obs = _bootfacts.observation(
                repo_id=repo_id,
                backend=backend,
                facts=facts,
                weights_gib=weights_gib,
                basis_gib=_capacity.GPU_TOTAL_GIB,
                # The resolver's tier 1 is an EXACT ctx match, so an
                # observation recorded with ctx=None can never be found by it
                # -- it sits in the store looking like data and shadows
                # nothing, while the panel keeps using an older entry. A launch
                # that does not pass --max-model-len still has a context: the
                # registry's.
                ctx=_argv_int(argv, "--max-model-len") or _spec_ctx(spec),
                max_num_seqs=_argv_int(argv, "--max-num-seqs"),
                kv_cache_dtype=_argv_str(argv, "--kv-cache-dtype"),
                ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            )
            if obs is None:
                return
            _discovery.append_observation(obs)
            log.info(
                "measured %s: kv %s tokens, overhead %s GiB at util %s",
                key,
                facts.kv_tokens,
                (obs["measured"] or {}).get("overhead_gib"),
                facts.util,
            )
        except Exception:  # noqa: BLE001 - measuring must never fail a boot
            log.warning("could not record a measurement for %s", key, exc_info=True)

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
            names = {spec.id, *spec.served_names()}
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
        # Adoption (2026-09-17): a registered model serving on its port with
        # no unit — v1's launch at cutover, or a hand launch. Only a listener
        # that names the registered id counts; a foreign process on the port
        # is doctor's finding, not a model.
        seen = {model.key for model in out}
        for key in self.registry.keys():
            if key in seen:
                continue
            spec = self._spec(key)
            if spec is None:
                continue
            # A real listener first (the socket table), then the identity
            # question. A probe answer with no listener pid cannot happen on
            # a box; in the suite it is the shape of every "port answers"
            # fake, and none of those is an adoption.
            pid = self._listener_pid(spec.port)
            if pid is None:
                continue
            ids = self._probe(spec.port)
            if not ids:
                continue
            names = {spec.id, *spec.served_names()}
            if not (names & set(ids)):
                continue
            out.append(
                LiveModel(
                    key=key,
                    unit=ADOPTED_UNIT,
                    port=spec.port,
                    state="active",
                    sub_state="adopted",
                    pid=pid,
                    ready=True,
                    restarts=0,
                    adopted=True,
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

        **The margin is not only a safety buffer; it is the launching worker's
        CUDA-context cushion.** ``--gpu-memory-utilization`` is a fraction of
        the card that vLLM will fill, but the process that fills it must first
        create a CUDA context, load cuBLAS/cuDNN kernels and allocate NCCL
        buffers — a few hundred MiB that are NOT counted in the fraction and
        are taken while the allocation is in progress. Hand a model literally
        everything free and its own startup overhead is what pushes it over.
        Hence the post-condition
        ``ceil(total * util) <= free - 700``, pinned by a test: whatever this
        function returns, at least 700 MiB of the free pool is still there for
        the worker that is about to claim it.
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
            pinned = getattr(spec, "util", None)
            if pinned is not None:
                # A registry `util` is the value this model is PROVEN to serve
                # at (models.toml says which launcher proved it). Handing it
                # more because the card happens to be empty is how a launch
                # ends up at 0.98 and dies of CUDA OOM once agents arrive; and
                # silently handing it LESS would change the KV budget, the
                # context that fits and the agent count with no mention of it,
                # so that is a refusal naming what is holding the card.
                if util < pinned:
                    return Refusal(
                        reason="not_enough_vram",
                        message=(
                            f"{spec.key} is proven at utilisation {pinned:.2f} but only "
                            f"{util:.2f} fits right now ({free} MiB free of {total} MiB, "
                            f"less the {self.margin_mib} MiB margin). Free the card, or "
                            f"launch it explicitly at a lower utilisation"
                        ),
                        key=spec.key,
                    )
                util = pinned
        else:
            if spec.vram_mib is None or spec.vram_mib <= 0:
                return Refusal(
                    reason="no_vram_budget",
                    message=f"resident {spec.key} has no vram_mib budget; utilisation is underived",
                    key=spec.key,
                )
            # A resident's utilisation comes from its budget, so nothing in the
            # arithmetic above notices that the budget does not fit. Without
            # this check the unit launches, vLLM asks for 3.3 GiB of a card
            # with 2 GiB free, and the failure arrives minutes later as a CUDA
            # OOM in journald with no mention of the number that was wrong.
            # What the resident will actually take is its DERIVED fraction of
            # the card, and the margin is the main model's cushion — it has
            # already been spent by the main model that is running. Charging
            # the resident for it too is what made the opt-in resident
            # unstartable in exactly the situation it exists for: 3,300 MiB
            # budget, 3,735 MiB free, refused as "not enough VRAM" while its
            # own allocation is 2,937 MiB and the v1 unit demonstrably ran it
            # co-resident with a 0.96 main model (its header records ~0.6 GiB
            # of slack in both boot orders at util 0.03).
            need = math.ceil(total * floor2(spec.vram_mib / total))
            if need + RESIDENT_CUSHION_MIB > free:
                return Refusal(
                    reason="not_enough_vram",
                    message=(
                        f"resident {spec.key} needs {need} MiB at utilisation "
                        f"{floor2(spec.vram_mib / total):.2f} plus a "
                        f"{RESIDENT_CUSHION_MIB} MiB CUDA-context cushion, and only "
                        f"{free} MiB is free"
                    ),
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
        restart: str = units.DEFAULT_RESTART,
        restart_sec: int = units.DEFAULT_RESTART_SEC,
        timeout_stop_sec: int = units.DEFAULT_TIMEOUT_STOP_SEC,
        util: float | None = None,
        argv_overrides: Mapping[str, str | None] | None = None,
    ) -> StartResult | Refusal:
        """Launch ``key`` as ``model-<key>.service`` and wait for it to answer.

        Refuses rather than raises for every operational "no": unknown model,
        already running, main slot taken, not enough VRAM.

        ``util`` and ``argv_overrides`` are the page's allocator (2026-09-17):
        an explicit utilisation and flag values (``--max-model-len``,
        ``--max-num-seqs``) for THIS launch only. The registry stays the
        default; the VRAM refusal still applies when nothing would fit.
        """
        spec = self._spec(key)
        if spec is None:
            return Refusal(reason="unknown_model", message=f"no model {key!r} in the registry", key=key)

        # A buffer leaked by a crashed or killed engine would otherwise hold
        # host RAM this boot's own offload buffer needs.
        self.reap_offload()

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

        requested_util = None if util is None else float(util)
        computed = self.compute_util(spec)
        if isinstance(computed, Refusal):
            return computed
        chosen = computed if util is None else float(util)
        if util is not None and chosen != computed:
            log.info("%s: launching at util %.2f (operator override; computed %.2f)", key, chosen, computed)
        util = chosen

        argv = apply_argv_overrides(list(spec.render_argv(util, spec.port)), argv_overrides)
        env = dict(spec.render_env())
        env.setdefault("PATH", f"{spec.venv_bin}:/usr/local/bin:/usr/bin:/bin")

        # Fail closed. If the manager's environment cannot be enumerated there
        # is no way to know what to redact, and starting anyway would hand the
        # model every ambient credential — the exact outcome this exists to
        # prevent. Refusing is loud; leaking is silent.
        try:
            unset_env = self.secret_env_names(keep=env)
        except UnitError as exc:
            return Refusal(
                reason="env_scan_failed",
                message=(
                    f"could not read the user manager's environment ({exc}); refusing to "
                    f"start {key} rather than hand it whatever credentials are in it"
                ),
                key=key,
            )
        if unset_env:
            log.info(
                "%s: unsetting %d inherited variable(s) for the model: %s",
                unit,
                len(unset_env),
                ", ".join(unset_env),  # names only; values are never read
            )

        # --- preflight: refusals that cost a second instead of a model outage --
        #
        # Everything below was learned the expensive way and then lost in the
        # port from the shell launchers (2026-09-18 audit).

        # Host RAM. No swap on this box and pinned pages cannot be reclaimed,
        # so overshooting is an OOM kill of the desktop, not a slowdown; one
        # happened on 2026-08-28. serve-opt.sh refused to launch below
        # CPU_OFFLOAD_GB + 30 and servedeck had no equivalent.
        need_ram = getattr(spec, "host_ram_gib", None)
        if need_ram:
            have = self._mem_available_gib()
            if have is not None and have < need_ram:
                return Refusal(
                    reason="not_enough_host_ram",
                    message=(
                        f"{key} needs about {need_ram} GiB of host RAM (its pinned host-side "
                        f"caches plus the loader's transient copies) and only {have:.0f} GiB is "
                        f"available. There is no swap on this box, so starting anyway risks an "
                        f"OOM kill of the session rather than a slow launch"
                    ),
                    key=key,
                )

        # Two host-memory offloads at once. The page can add
        # --kv-offloading-size to any model; on one that already parks its
        # experts in host RAM via --cpu-offload-gb, the two add up to more
        # than the box has.
        if "--cpu-offload-gb" in argv and "--kv-offloading-size" in argv:
            return Refusal(
                reason="conflicting_offload",
                message=(
                    f"{key} already offloads to host RAM with --cpu-offload-gb; adding "
                    f"--kv-offloading-size on top of it double-books memory this box "
                    f"does not have"
                ),
                key=key,
            )

        # The port. Without this, a switch stops the running model, waits for
        # VRAM, loads a 90 GiB engine for minutes and only then fails to bind
        # — leaving the card empty and every client on a 503. Registry ports
        # get squatted by unrelated projects on this box (:8002 and :8003 by
        # cvi-scratch servers, repeatedly).
        holder_pid = self._listener_pid(spec.port)
        if holder_pid is not None:
            return Refusal(
                reason="port_busy",
                message=(
                    f"port {spec.port} is already held by pid {holder_pid} "
                    f"({_cmdline_of(holder_pid) or 'unknown process'}), so {key} could not "
                    f"bind it. Stop that process, or change this model's port in models.toml"
                ),
                key=key,
            )

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
                timeout_stop_sec=timeout_stop_sec,
                description=f"servedeck model {spec.id} ({key})",
                unset_env=unset_env,
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
            # Measure this boot before anything else. `append_observation`'s
            # own docstring claimed it was "written after every boot reaching
            # READY"; it had three tests and zero callers, so the store had not
            # been written since 2026-09-02 and the capacity panel spent three
            # weeks replaying an August measurement of a model that had since
            # gained MTP-3 and the offload connector. The panel was 1.15 GiB
            # optimistic, which is how it came to green-light a Flash-Next
            # launch at util 0.95 that the engine then refused.
            self._record_boot(key, spec, unit, argv, since)
            current_desired = self.load_desired()
            # Record what actually worked, so a servedeck restart or a reboot
            # repeats THIS launch rather than the registry defaults with the
            # utilisation recomputed from an idle card (0.98 — the value with
            # the OOM-under-concurrency history). Only a ready boot is stored:
            # a configuration that failed is not one to repeat unattended.
            if requested_util is not None or argv_overrides:
                current_desired = current_desired.with_launch(
                    key,
                    desired_mod.Launch(
                        util=requested_util,
                        argv={k: v for k, v in (argv_overrides or {}).items()},
                    ),
                )
            if spec.slot == "main":
                self._save_desired(current_desired.with_main(key))
            else:
                self._save_desired(current_desired.with_resident(key))
            return result

        # A start that did not come up must not be left behind. With
        # Restart=on-failure the unit keeps retrying after we stop watching,
        # holding the port and taking the GPU on each attempt — unattended,
        # because the caller has already been handed a failure and moved on.
        # That is R4's crash-loop with nobody watching, and it survives a
        # timeout (where the unit is often perfectly healthy and just slow)
        # just as much as a crash.
        #
        # units.stop, NOT self.stop: self.stop also REMOVES the key from
        # desired state, and a start invoked by reconcile() is acting on an
        # intent the operator already recorded. Erasing it because one boot
        # attempt failed means the model is never retried and nothing says
        # why. The unit is stopped; the intent is not.
        try:
            units.stop(unit, run=self._run)
        except UnitError as exc:
            log.warning("%s: failed start could not be cleaned up: %s", unit, exc)
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
        # "Never appeared" and "vanished after starting" are different faults
        # with different fixes — a rejected unit definition or a refused
        # transient name versus a model that crashed — and both would otherwise
        # arrive as the same "--collect removed it" sentence, sending the
        # reader to the journal of a unit that was never created.
        appeared = not units.gone(unit, run=self._run, sleep=self._sleep)
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

        if not appeared:
            return finish(
                False,
                f"{unit} never appeared: systemd-run reported success but the unit "
                f"is not loaded, so nothing was started under that name. Look at "
                f"the unit definition, not at the model.",
            )

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
                    # `gone`, not `exists`: one LoadState read is ~0.2%
                    # unreliable (see units.exists), and at one health check
                    # every 2s a five-minute boot takes ~150 of them — so
                    # trusting a single negative would report a healthy model
                    # dead about a quarter of the time, with a confident
                    # message naming the wrong cause.
                    if units.gone(unit, run=self._run, sleep=self._sleep):
                        return finish(
                            False,
                            f"{unit} vanished after starting: it failed and --collect "
                            f"removed it (systemctl show would report inactive/success "
                            f"for it now)",
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
                    if ids is not None and ({spec.id, *spec.served_names()} & set(ids)):
                        drain_until = self._clock() + _READY_DRAIN_S
                        while self._clock() < drain_until:
                            consume(stream.poll_lines(min(0.2, _READY_DRAIN_S)))
                        return finish(True, None)
        finally:
            stream.close()

    # -- stop -------------------------------------------------------------

    def stop(
        self,
        key: str,
        timeout_s: float = units.DEFAULT_STOP_TIMEOUT_S,
        forget_intent: bool = True,
    ) -> StopResult | Refusal:
        """Stop ``model-<key>`` and record that it is no longer wanted.

        The VRAM accounting is captured *before* the stop, because after it the
        pids are gone and there is nothing left to attribute memory to.

        ``forget_intent=False`` stops the unit WITHOUT erasing desired state,
        which is what a switch needs: a switch whose replacement then fails to
        boot must leave a box that still knows which model it wants. Erasing
        it first meant a failed Apply & restart left ``main: null`` — nothing
        serving, no record of what had been, and a reboot that started nothing.
        """
        unit = self.unit_for(key)
        if not units.valid_unit_name(unit):
            return Refusal(reason="bad_key", message=f"{key!r} is not a usable model key", key=key)
        adopted = next((m for m in self.live() if m.key == key and m.adopted), None)
        if adopted is not None:
            return self._stop_adopted(key, adopted, timeout_s, forget_intent=forget_intent)
        # Confirmed, because a false "not there" skips the VRAM accounting
        # below and would let the next switch boot into an occupied card.
        was_live = not units.gone(unit, run=self._run, sleep=self._sleep)
        free_before = self._free_mib()
        held = self._held_mib(unit) if was_live else 0
        try:
            units.stop(unit, timeout_s=timeout_s, run=self._run)
        except UnitError as exc:
            return Refusal(reason="stop_failed", message=str(exc), key=key)

        if forget_intent:
            current = self.load_desired()
            if current.main == key:
                current = current.with_main(None)
            current = current.without_resident(key)
            self._save_desired(current)
        return StopResult(
            key=key, unit=unit, was_live=was_live, held_mib=held, free_before_mib=free_before,
            reaped=tuple(self.reap_offload()),
        )

    def _stop_adopted(
        self, key: str, model: LiveModel, timeout_s: float, forget_intent: bool = True
    ) -> StopResult | Refusal:
        """Stop a model no unit owns: SIGTERM its listener pid, wait for the
        port to stop answering, SIGKILL the tree if it will not. The VRAM
        accounting sums the pid and its descendants, because vLLM's API
        server holds nothing and its engine-core children hold everything."""
        free_before = self._free_mib()
        usage = self._used_by_pids()
        tree = [model.pid, *self._descendants(model.pid)]
        held = None if usage is None else sum(usage.get(p, 0) for p in tree)
        try:
            self._kill(model.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            return Refusal(reason="stop_failed", message=f"kill {model.pid}: {exc}", key=key)
        deadline = self._clock() + timeout_s
        while self._probe(model.port) is not None or self._listener_pid(model.port) is not None:
            if self._clock() >= deadline:
                for p in reversed(tree):
                    try:
                        self._kill(p, signal.SIGKILL)
                    except OSError:
                        pass
                break
            self._sleep(0.5)
        if forget_intent:
            current = self.load_desired()
            if current.main == key:
                current = current.with_main(None)
            current = current.without_resident(key)
            self._save_desired(current)
        return StopResult(
            key=key, unit=ADOPTED_UNIT, was_live=True, held_mib=held, free_before_mib=free_before,
            reaped=tuple(self.reap_offload()),
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
        util: float | None = None,
        argv_overrides: Mapping[str, str | None] | None = None,
        relaunch: bool = False,
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

        When there is no number to aim at — ``used_by_pids()`` returned None,
        or the driver attributed nothing to the unit's cgroup — the wait falls
        back to a plateau: poll until ``free_mib()`` has stopped rising for
        three consecutive reads. It never reports ``released=True`` for a wait
        that did not happen, because "we did not check" and "the memory came
        back" must not arrive at the caller as the same answer.
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

        # relaunch=True (2026-09-17): the page's "Apply & restart" on the model
        # that already holds the slot, with new util/context/agents/offload.
        # Without it this branch refused "already holds the main slot", the
        # POST was accepted, and nothing restarted.
        if holder is not None and (holder.key != key or relaunch):
            # forget_intent=False: if the replacement fails to boot, desired
            # state must still name a model, so reconcile and the next reboot
            # have something to bring back.
            result = self.stop(holder.key, forget_intent=False)
            if isinstance(result, Refusal):
                return result
            stopped = result
            released, waited, free_after, reason = self._wait_for_release(
                free_before=result.free_before_mib,
                held_mib=result.held_mib,
                timeout_s=release_timeout_s,
                on_progress=on_progress,
            )
            if reason == "vram_accounting_unavailable":
                return SwitchResult(
                    stopped=stopped,
                    released=False,
                    waited_s=waited,
                    free_after_mib=free_after,
                    started=Refusal(
                        reason="vram_accounting_unavailable",
                        message=(
                            f"nvidia-smi cannot report free VRAM, so there is no way to tell "
                            f"whether {result.unit} let go. Refusing to boot {key} on the "
                            f"assumption that it did."
                        ),
                        key=key,
                        live_key=holder.key,
                    ),
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

        started = self.start(key, timeout_s=timeout_s, on_progress=on_progress, util=util, argv_overrides=argv_overrides)
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
    ) -> tuple[bool, float, int | None, str | None]:
        """``(released, waited_s, free_after_mib, reason)``.

        ``released=True`` is only ever returned by a wait that actually
        observed something. There is no "nothing to wait for, call it done"
        branch: that branch is indistinguishable, at the call site, from a
        successful wait, and it fires precisely when the instrument is broken —
        a measurement failing downward into a confident yes.
        """
        started = self._clock()
        if self._free_mib() is None:
            # No instrument at all. Not "released", not "not released" — a
            # distinct answer, because the caller must refuse rather than
            # retry.
            return False, 0.0, None, "vram_accounting_unavailable"

        if held_mib and free_before is not None:
            return self._wait_for_target(
                free_before + RELEASE_FRACTION * held_mib, started, timeout_s, on_progress
            )

        # No attribution: nvidia-smi could not enumerate compute apps, or the
        # driver credited this unit's cgroup with nothing. There is no target,
        # but there is still a signal — memory comes back in steps and then
        # stops. Waiting for the plateau is weaker than waiting for a number
        # and much stronger than not waiting.
        return self._wait_for_plateau(started, timeout_s, on_progress)

    def _wait_for_target(
        self,
        target: float,
        started: float,
        timeout_s: float,
        on_progress: ProgressCallback | None,
    ) -> tuple[bool, float, int | None, str | None]:
        while True:
            free_now = self._free_mib()
            if free_now is not None and free_now >= target:
                return True, self._clock() - started, free_now, None
            waited = self._clock() - started
            if waited >= timeout_s:
                return False, waited, free_now, "vram_not_released"
            if on_progress is not None:
                on_progress(
                    Progress(
                        kind="line",
                        text=f"waiting for VRAM: {free_now} MiB free, need {int(target)} MiB",
                        elapsed_s=waited,
                    )
                )
            self._sleep(1.0)

    def _wait_for_plateau(
        self,
        started: float,
        timeout_s: float,
        on_progress: ProgressCallback | None,
        stable_reads: int = 3,
    ) -> tuple[bool, float, int | None, str | None]:
        best: int | None = None
        stable = 0
        while True:
            free_now = self._free_mib()
            if free_now is None:
                return False, self._clock() - started, None, "vram_accounting_unavailable"
            if best is None or free_now > best:
                best = free_now
                stable = 0
            else:
                stable += 1
                if stable >= stable_reads:
                    return True, self._clock() - started, free_now, None
            waited = self._clock() - started
            if waited >= timeout_s:
                return False, waited, free_now, "vram_not_released"
            if on_progress is not None:
                on_progress(
                    Progress(
                        kind="line",
                        text=(
                            f"waiting for VRAM to settle (no per-process attribution): "
                            f"{free_now} MiB free, stable for {stable}/{stable_reads} reads"
                        ),
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

        **This call BLOCKS for as long as the models take to boot** — minutes
        for a 90 GiB model — so it must not be run from an ASGI lifespan
        handler: uvicorn does not bind the socket until startup returns, and
        servedeck would be unreachable for the whole boot, including to the
        dashboard that is meant to be showing its progress. P4 runs it as a
        task created after the bind.
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
            # The settings the operator last applied, not the registry
            # defaults: without this a reboot silently re-tuned the box.
            stored = want.launch_for(key)
            result = self.start(
                key,
                timeout_s=timeout_s,
                on_progress=on_progress,
                util=stored.util,
                argv_overrides=stored.argv or None,
            )
            if isinstance(result, Refusal):
                log.warning(
                    "reconcile: %s not started (%s): %s", key, result.reason, result.message
                )
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
