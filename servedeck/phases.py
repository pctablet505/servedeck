"""Servedeck phase detection — SPEC.md §5.

Turns a stream of vLLM boot-log lines into an ordered phase state machine
(:class:`PhaseTracker`) and classifies terminal/near-terminal error lines
into typed :class:`Failure` objects (:func:`classify`).

Every regex here is transcribed VERBATIM from SPEC.md §5 (which was in turn
measured against real boot logs in vllm-qwen38next/ and local_llm/logs/).
Do not "improve", generalize, or re-derive these patterns — if a real log
line stops matching, fix SPEC.md first, then this file.

This module does no I/O: it consumes line strings and returns data. File
tailing lives in logtail.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


class Phase(str, Enum):
    """Ordered boot phases. Values are the wire/JSON-friendly names."""

    INIT = "init"
    LOADING_WEIGHTS = "loading_weights"
    COMPILING = "compiling"
    KV_CACHE = "kv_cache"
    CUDA_GRAPHS = "cuda_graphs"
    HTTP_START = "http_start"
    READY = "ready"


# Fixed FSM order. Index defines "how far along" a phase is — the tracker
# only ever advances forward through this tuple, never regresses.
PHASE_ORDER: tuple[Phase, ...] = (
    Phase.INIT,
    Phase.LOADING_WEIGHTS,
    Phase.COMPILING,
    Phase.KV_CACHE,
    Phase.CUDA_GRAPHS,
    Phase.HTTP_START,
    Phase.READY,
)

_PHASE_INDEX: dict[Phase, int] = {p: i for i, p in enumerate(PHASE_ORDER)}


# ---------------------------------------------------------------------------
# §5 regexes — verbatim
# ---------------------------------------------------------------------------

# INIT
RE_INIT = re.compile(r"\[core\.py:\d+\]\s+Initializing a V1 LLM engine")

# LOADING_WEIGHTS
# Wildcard file:line as \w+\.py:\d+ per SPEC — but this anchor specifically
# differs only in the optional "gpu_" prefix between vLLM 0.27.1 / 0.29.0.dev0.
RE_LOADING_WEIGHTS = re.compile(
    r"\[(?:gpu_)?model_runner\.py:\d+\]\s+Starting to load model\s+(\S+)"
)
RE_SHARD_PROGRESS = re.compile(
    r"Loading safetensors checkpoint shards:\s+(\d+)%\s+Completed\s+\|\s+(\d+)/(\d+)"
)
RE_WEIGHTS_MEASURED = re.compile(
    r"\[(?:gpu_)?model_runner\.py:\d+\]\s+Model loading took\s+([\d.]+)\s+GiB memory and\s+([\d.]+)\s+seconds"
)

# COMPILING
RE_COMPILING = re.compile(
    r"\[backends\.py:\d+\]\s+Using cache directory:\s+(\S+)\s+for vLLM's torch\.compile"
)
# Fires TWICE for MTP models (backbone then eagle_head) — callers must
# accumulate, never treat the second hit as a phase reset.
RE_COMPILE_EXIT = re.compile(r"\[monitor\.py:\d+\]\s+torch\.compile took\s+([\d.]+)\s+s in total")

# KV_CACHE
RE_KV_CACHE = re.compile(r"\[gpu_worker\.py:\d+\]\s+Available KV cache memory:\s+([\d.]+)\s+GiB")
# Flash-Next puts both of these on ONE line (kv_cache_utils.py:2258); the 27B
# SPLITS them across :2235/:2236. Two independent re.search calls, applied
# per-line, cover both layouts without needing to know which one we're in.
RE_KV_TOKENS = re.compile(r"GPU KV cache size:\s+([\d,]+)\s+tokens")
RE_KV_CONCURRENCY = re.compile(r"Maximum concurrency for\s+([\d,]+)\s+tokens per request:\s+([\d.]+)x")

# CUDA_GRAPHS
# tqdm \r-joins many updates into one physical line — the capture regex only
# grabs the phase label; percentage is pulled separately by taking the LAST
# "NN%|" occurrence on the line (see _parse_cuda_graph_pct).
RE_CUDA_GRAPHS = re.compile(r"Capturing CUDA graphs\s+\(([^)]+)\):")
RE_CUDA_GRAPHS_PCT = re.compile(r"(\d+)%\|")

# HTTP_START
RE_HTTP_START = re.compile(r"\[api_server\.py:\d+\]\s+Starting vLLM server on\s+http://([\d.]+):(\d+)")

# READY — BOTH required; the HTTP probe is authoritative (same criterion as
# is_server_up()). This module only recognizes the log-text half; the probe
# result is fed in externally via PhaseTracker.set_http_probe_ok().
RE_STARTUP_COMPLETE = re.compile(r"INFO:\s+Application startup complete\.")


# ---------------------------------------------------------------------------
# Phase-tracked data
# ---------------------------------------------------------------------------


@dataclass
class WeightsInfo:
    model: str | None = None
    shard_pct: int | None = None
    shard_done: int | None = None
    shard_total: int | None = None
    gib: float | None = None
    seconds: float | None = None


@dataclass
class KvCacheInfo:
    available_gib: float | None = None
    tokens: int | None = None
    max_ctx_per_request: int | None = None
    concurrency_x: float | None = None


@dataclass
class CudaGraphsInfo:
    # label -> last-seen percent complete, e.g. {"PIECEWISE": 100, "FULL": 100}
    progress: dict[str, int] = field(default_factory=dict)


@dataclass
class HttpStartInfo:
    host: str | None = None
    port: int | None = None


@dataclass
class PhaseEvent:
    """One phase-relevant observation, emitted by PhaseTracker.feed()."""

    phase: Phase
    line: str
    kind: str  # "enter" | "shard_progress" | "measurement" | "compile_exit" |
    #            "kv_available" | "kv_tokens" | "kv_concurrency" |
    #            "cuda_graphs_progress" | "http_start" | "startup_complete"
    advanced: bool  # True if this observation moved self.phase forward


class PhaseTracker:
    """Stateful FSM over vLLM boot-log lines.

    Advance-only: a line matching phase N's regex moves ``phase`` to
    ``max(current, N)``; it never regresses. This tolerates backends that
    skip an anchor line (Flash-Next never prints the LOADING_WEIGHTS anchor
    "Starting to load model ..." — it goes straight to the safetensors
    shard-progress / "Model loading took" lines, which still count as
    LOADING_WEIGHTS evidence and advance the phase).
    """

    def __init__(self) -> None:
        self.phase: Phase | None = None
        self.weights = WeightsInfo()
        self.kv = KvCacheInfo()
        self.cuda_graphs = CudaGraphsInfo()
        self.http = HttpStartInfo()
        self.compile_exits: list[float] = []
        self._startup_complete_seen = False
        self._http_probe_ok = False
        self._ready_latched = False

    @property
    def phase_index(self) -> int:
        return -1 if self.phase is None else _PHASE_INDEX[self.phase]

    @property
    def reached_ready(self) -> bool:
        """True from the instant BOTH READY criteria are first satisfied, and
        for the rest of this run.

        This is a fact about the run's history ("did this boot ever serve"),
        not a liveness reading. The monitor keeps probing /v1/models after
        READY, and those probes fail as soon as the server starts shutting
        down. This flag used to be recomputed from the latest probe, so every
        stop turned it back to False. Every history record then said
        reached_ready: false, including runs that served for minutes (F11d),
        and the stopped server's stale "ready" phase painted as a boot in
        progress.
        """
        return self._ready_latched

    def _latch_if_ready(self) -> bool:
        """Latch READY once both criteria hold. True only on the call that
        latches it."""
        if not self._ready_latched and self._startup_complete_seen and self._http_probe_ok:
            self._ready_latched = True
            return True
        return False

    def mark_adopted_ready(self) -> None:
        """Declare READY for a server that was already serving when adopted.

        Adoption skips the whole boot sequence: there is no log to replay, so
        neither READY criterion can ever be satisfied from evidence. Without
        this, an adopted server's reached_ready stays False and a crash while
        serving is misfiled as a failed boot - which disables auto-restart for
        exactly the case auto-restart exists to handle.
        """
        self._startup_complete_seen = True
        self._http_probe_ok = True
        self._ready_latched = True
        self.phase = Phase.READY

    def set_http_probe_ok(self, ok: bool) -> PhaseEvent | None:
        """Record the result of an external GET /v1/models probe.

        Not log-derived, so it lives outside feed(). Returns a PhaseEvent iff
        this call is the one that completes the READY criteria. A failing
        probe after that point does not undo READY (see reached_ready).
        """
        self._http_probe_ok = ok
        if ok and self._latch_if_ready():
            self.phase = Phase.READY
            return PhaseEvent(phase=Phase.READY, line="", kind="ready", advanced=True)
        return None

    def _advance(self, phase: Phase) -> bool:
        idx = _PHASE_INDEX[phase]
        if idx > self.phase_index:
            self.phase = phase
            return True
        return False

    def feed(self, line: str) -> list[PhaseEvent]:
        """Process one log line. Returns zero or more PhaseEvents."""
        events: list[PhaseEvent] = []

        if RE_INIT.search(line):
            events.append(PhaseEvent(Phase.INIT, line, "enter", self._advance(Phase.INIT)))

        m = RE_LOADING_WEIGHTS.search(line)
        if m:
            self.weights.model = m.group(1)
            events.append(
                PhaseEvent(Phase.LOADING_WEIGHTS, line, "enter", self._advance(Phase.LOADING_WEIGHTS))
            )

        m = RE_SHARD_PROGRESS.search(line)
        if m:
            self.weights.shard_pct = int(m.group(1))
            self.weights.shard_done = int(m.group(2))
            self.weights.shard_total = int(m.group(3))
            events.append(
                PhaseEvent(
                    Phase.LOADING_WEIGHTS, line, "shard_progress", self._advance(Phase.LOADING_WEIGHTS)
                )
            )

        m = RE_WEIGHTS_MEASURED.search(line)
        if m:
            self.weights.gib = float(m.group(1))
            self.weights.seconds = float(m.group(2))
            events.append(
                PhaseEvent(Phase.LOADING_WEIGHTS, line, "measurement", self._advance(Phase.LOADING_WEIGHTS))
            )

        if RE_COMPILING.search(line):
            events.append(PhaseEvent(Phase.COMPILING, line, "enter", self._advance(Phase.COMPILING)))

        m = RE_COMPILE_EXIT.search(line)
        if m:
            # Fires twice (backbone, eagle_head for MTP) — accumulate, don't reset.
            self.compile_exits.append(float(m.group(1)))
            events.append(
                PhaseEvent(Phase.COMPILING, line, "compile_exit", self._advance(Phase.COMPILING))
            )

        m = RE_KV_CACHE.search(line)
        if m:
            self.kv.available_gib = float(m.group(1))
            events.append(PhaseEvent(Phase.KV_CACHE, line, "kv_available", self._advance(Phase.KV_CACHE)))

        m = RE_KV_TOKENS.search(line)
        if m:
            self.kv.tokens = int(m.group(1).replace(",", ""))
            events.append(PhaseEvent(Phase.KV_CACHE, line, "kv_tokens", self._advance(Phase.KV_CACHE)))

        m = RE_KV_CONCURRENCY.search(line)
        if m:
            self.kv.max_ctx_per_request = int(m.group(1).replace(",", ""))
            self.kv.concurrency_x = float(m.group(2))
            events.append(
                PhaseEvent(Phase.KV_CACHE, line, "kv_concurrency", self._advance(Phase.KV_CACHE))
            )

        m = RE_CUDA_GRAPHS.search(line)
        if m:
            label = m.group(1)
            pct = _parse_cuda_graph_pct(line)
            if pct is not None:
                self.cuda_graphs.progress[label] = pct
            events.append(
                PhaseEvent(Phase.CUDA_GRAPHS, line, "cuda_graphs_progress", self._advance(Phase.CUDA_GRAPHS))
            )

        m = RE_HTTP_START.search(line)
        if m:
            self.http.host = m.group(1)
            self.http.port = int(m.group(2))
            events.append(PhaseEvent(Phase.HTTP_START, line, "http_start", self._advance(Phase.HTTP_START)))

        if RE_STARTUP_COMPLETE.search(line):
            self._startup_complete_seen = True
            advanced = False
            self._latch_if_ready()
            if self.reached_ready:
                advanced = self._advance(Phase.READY)
            events.append(PhaseEvent(Phase.READY, line, "startup_complete", advanced))

        return events


def _parse_cuda_graph_pct(line: str) -> int | None:
    """Take the LAST "NN%|" occurrence on a tqdm \\r-joined line."""
    matches = RE_CUDA_GRAPHS_PCT.findall(line)
    if not matches:
        return None
    return int(matches[-1])


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

RE_KV_TOO_SMALL = re.compile(
    r"ValueError: To serve at least one request with the model's max seq len \((\d+)\), "
    r"\(([\d.]+) GiB KV cache is needed, which is larger than the available KV cache memory "
    r"\(([\d.]+) GiB\)\. Based on the available memory, the estimated maximum model length is (\d+)\."
)
RE_PLE_FP8_PATCH_MISSING = re.compile(r"no module or parameter named 'ngram_embedding\.weight_scale'")
RE_PTRACE_DENIED = re.compile(r"pidfd_getfd: Operation not permitted")
RE_CUDA_SYMLINKS = re.compile(r"Could NOT find CUDA_CUDART_LIBRARY")
RE_CUDA_TOOLCHAIN = re.compile(r"the provided PTX was compiled with an unsupported toolchain")
RE_FLASHINFER_LINK = re.compile(r"Ninja build failed|cannot find -l(?:cudart|nvrtc|nvvm)")
RE_STARTUP_OOM = re.compile(r"Free memory .* is less than desired GPU memory utilization")
RE_ENGINE_INIT_FAILED = re.compile(r"RuntimeError: Engine core initialization failed")
RE_CUDA_FAULT = re.compile(r"torch\.AcceleratorError: CUDA error: (misaligned address|an illegal memory access)")
RE_RUNTIME_OOM = re.compile(r"CUDA out of memory")
# Written descriptively in SPEC ("shm_broadcast: No available shared memory
# broadcast block found in 60 seconds"); the real log line is
# "[shm_broadcast.py:801] No available shared memory broadcast block found
# in 60 seconds." — match on the module name plus the exact message so this
# never confuses a genuine error with the same message from elsewhere.
RE_INFORMATIONAL_SHM = re.compile(
    r"shm_broadcast.*No available shared memory broadcast block found in 60 seconds"
)


@dataclass(frozen=True)
class Failure:
    code: str
    line: str
    groups: tuple[str, ...] = ()
    auto_restart: bool = False
    fix_action: dict[str, str] | None = None
    hint: str | None = None


def classify(line: str, reached_ready: bool) -> Failure | None:
    """Classify one log line as a Failure, or None if it isn't one.

    ``reached_ready`` is the run's reached_ready flag (whether phase READY
    ever completed before this line), consulted only for the two failure
    codes whose auto-restart eligibility depends on it per SPEC §5/§6.
    """

    # INFORMATIONAL first: this line is a superset match risk (contains
    # "shm_broadcast" and looks alarming) but must NEVER be surfaced as an
    # error — SETUP.md:407.
    if RE_INFORMATIONAL_SHM.search(line):
        return Failure(code="INFORMATIONAL", line=line, auto_restart=False)

    m = RE_KV_TOO_SMALL.search(line)
    if m:
        return Failure(
            code="KV_TOO_SMALL",
            line=line,
            groups=m.groups(),
            auto_restart=False,
            fix_action={"ctx": m.group(4)},
        )

    if RE_PLE_FP8_PATCH_MISSING.search(line):
        return Failure(code="PLE_FP8_PATCH_MISSING", line=line, auto_restart=False)

    if RE_PTRACE_DENIED.search(line):
        return Failure(code="PTRACE_DENIED", line=line, auto_restart=False)

    if RE_CUDA_SYMLINKS.search(line):
        return Failure(code="CUDA_SYMLINKS", line=line, auto_restart=False)

    if RE_CUDA_TOOLCHAIN.search(line):
        return Failure(code="CUDA_TOOLCHAIN", line=line, auto_restart=False)

    if RE_FLASHINFER_LINK.search(line):
        return Failure(
            code="FLASHINFER_LINK",
            line=line,
            auto_restart=False,
            hint='grep -nE "FAILED:|cannot find -l" serve.log',
        )

    if RE_STARTUP_OOM.search(line):
        return Failure(code="STARTUP_OOM", line=line, auto_restart=False, hint="offer orphan sweep")

    if RE_ENGINE_INIT_FAILED.search(line):
        return Failure(code="ENGINE_INIT_FAILED", line=line, auto_restart=False, hint="attach preceding 40 lines")

    m = RE_CUDA_FAULT.search(line)
    if m:
        return Failure(code="CUDA_FAULT", line=line, groups=m.groups(), auto_restart=reached_ready)

    if RE_RUNTIME_OOM.search(line):
        return Failure(code="RUNTIME_OOM", line=line, auto_restart=reached_ready)

    return None


# ---------------------------------------------------------------------------
# Log line severity (display styling) — §5 final paragraph
# ---------------------------------------------------------------------------

RE_SEV_ERROR_WORD = re.compile(r"\sERROR\s")
RE_SEV_TRACEBACK = re.compile(r"Traceback \(most recent call last\)")
RE_SEV_ERROR_LINE_START = re.compile(r"^\s*\w*(Error|Exception):")
RE_SEV_WARNING = re.compile(r"\sWARNING\s")


def line_severity(line: str, is_phase_match: bool = False) -> str:
    """Return one of "e" (error), "w" (warning), "g" (good/phase), "" (default).

    Precedence exactly as SPEC §5 lists it: error checks first, then
    warning, then phase/measurement match, else default.
    ``is_phase_match`` should be True when the caller already knows this
    line matched one of the phase/measurement regexes above (PhaseTracker
    returned a non-empty event list for it).
    """
    if (
        RE_SEV_ERROR_WORD.search(line)
        or RE_SEV_TRACEBACK.search(line)
        or RE_SEV_ERROR_LINE_START.search(line)
    ):
        return "e"
    if RE_SEV_WARNING.search(line):
        return "w"
    if is_phase_match:
        return "g"
    return ""
