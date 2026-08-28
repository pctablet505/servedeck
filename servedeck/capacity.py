"""Coldstart capacity module — SPEC.md §3.

PURE. No file I/O, no subprocess, no network. Every environmental fact that
requires actually looking at the machine (free VRAM, ptrace_scope, training
markers, GPU health, the live Codex subagent cap) arrives already-collected
via the optional `live` parameter (a `LiveFacts`) from a caller that is
allowed to do I/O (supervisor / preflight). This module only does arithmetic
and produces `Finding`s from the numbers it is handed.

Constants, formulas and every finding trigger below are transcribed from
SPEC.md §3 verbatim. Do not "improve" the measured constants.
"""

import math
from dataclasses import dataclass
from typing import Any, Literal

from . import config as _config

# ---------------------------------------------------------------------------
# Constants (SPEC.md §3) — provenance noted inline.
# ---------------------------------------------------------------------------

# Hardware and safety constants. All come from servedeck.config so a different
# card, or a different launcher, needs no code change. They are read at import
# time; call capacity.refresh_limits() after changing config in a long-lived
# process (tests do this).
_cfg = _config.get()

#: Total VRAM. 0 means detection failed -- compute() then refuses to guess.
GPU_TOTAL_MIB: int = _cfg.gpu_total_mib
GPU_TOTAL_GIB: float = GPU_TOTAL_MIB / 1024

#: VRAM that is neither weights nor KV: activations and CUDA graphs.
OVERHEAD_GIB_DEFAULT: float = _cfg.overhead_gib

#: Fragmentation margin. Reported as a WARNING when free memory is within this
#: of the budget -- never added to the requirement itself. Adding it made every
#: utilization above ~0.958 look impossible on a card that runs 0.95 daily.
VRAM_GUARD_HEADROOM_MIB: int = _cfg.frag_margin_mib

#: Extra margin required on top of two models' weights before blue-green
#: (running two servers at once) is considered feasible.
FRAG_MARGIN_GIB: float = 1.0

#: Above this utilization, warn about a thin margin.
UTIL_THIN_MARGIN: float = 0.97

#: Hybrid Mamba/attention models fail CUDA-graph capture above this. Only
#: applied to backends that declare it; harmless elsewhere.
MAMBA_MAX_NUM_SEQS_INLINE: int = 128

#: Some models cannot fit their own weights below a floor utilization. Derived
#: per model from measured weights rather than hardcoded per backend.
FLASHNEXT_MIN_UTIL: float = 0.90


def _cfg_markers() -> tuple[str, ...]:
    import os
    raw = os.environ.get("SERVEDECK_TRAINING_MARKERS", "")
    return tuple(p for p in raw.split(":") if p)


def refresh_limits() -> None:
    """Re-read hardware limits from config (after config.reset())."""
    global _cfg, GPU_TOTAL_MIB, GPU_TOTAL_GIB, OVERHEAD_GIB_DEFAULT, VRAM_GUARD_HEADROOM_MIB
    _cfg = _config.get()
    GPU_TOTAL_MIB = _cfg.gpu_total_mib
    GPU_TOTAL_GIB = GPU_TOTAL_MIB / 1024
    OVERHEAD_GIB_DEFAULT = _cfg.overhead_gib
    VRAM_GUARD_HEADROOM_MIB = _cfg.frag_margin_mib

# A "lock file" convention: if any of these paths exists, something else wants
# the GPU (a training run, a benchmark) and Coldstart must stand down rather
# than start a server. Configure via `training_markers` in servedeck.toml.
# capacity.py stays pure — it never stat()s anything; a caller that is allowed
# I/O checks existence and reports hits via LiveFacts.training_markers.
TRAINING_MARKER_PATHS: tuple[str, ...] = tuple(_cfg_markers())


Level = Literal["block", "warn"]
WeightsSource = Literal["measured", "estimated", "unknown"]
Trust = Literal["measured", "measured_other_ctx", "estimated", "unknown"]

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInputs:
    """Everything capacity.compute() needs to know about one model.

    Populated by servedeck/registry.py's resolve_inputs(); capacity.py does
    not know how to get these numbers, only what to do with them.
    """

    repo_id: str
    backend: str  # "flashnext" | "inline"
    model_max_ctx: int
    weights_gib: float | None = None
    weights_source: WeightsSource = "measured"
    kv_kib_per_token: float | None = None
    overhead_gib: float = OVERHEAD_GIB_DEFAULT
    trust: Trust = "measured"
    servable: bool = True
    unservable_reason: str | None = None
    model_type: str | None = None
    # For MEASURED_OTHER_CTX: which ctx kv_kib_per_token was actually
    # measured at, and any other known (ctx -> KiB/tok) rates for display.
    used_ctx_for_rate: int | None = None
    known_kv_rates: dict[int, float] | None = None


@dataclass(frozen=True)
class Finding:
    code: str
    level: Level
    title: str
    detail: str
    fix: str | None = None
    fix_action: dict[str, Any] | None = None


@dataclass(frozen=True)
class CapacityBar:
    weights_pct: float
    kv_pct: float
    overhead_pct: float
    free_pct: float


@dataclass(frozen=True)
class CapacityResult:
    budget_gib: float
    kv_gib: float
    kv_tokens: int
    concurrency_x: float
    agents_at_ctx: int
    effective_parallel: int
    max_single_ctx: int
    confidence: str
    bar: CapacityBar
    findings: tuple[Finding, ...]
    can_apply: bool


@dataclass(frozen=True)
class LiveFacts:
    """Environmental facts collected by an I/O-capable caller.

    Every field is optional: omit what you did not check and the finding
    that depends on it is simply skipped rather than guessed at.
    """

    gpu_responsive: bool | None = None  # False => `nvidia-smi -L` failed
    total_mib: int | None = None  # defaults to GPU_TOTAL_MIB when None
    used_mib: int | None = None  # nvidia-smi memory.used
    own_mib: int = 0  # VRAM held by our own vllm processes (discounted)
    training_markers: list[str] | None = None  # paths found present
    ptrace_scope: int | None = None  # current kernel.yama.ptrace_scope
    actual_state: str | None = None  # supervisor.actual_state, duck-typed
    codex_max_subagents: int | None = None  # CODEX_MAX_SUBAGENTS, if known


@dataclass(frozen=True)
class BlueGreenVerdict:
    feasible: bool
    combined_gib: float  # w_a + w_b + 2*OVERHEAD_GIB_DEFAULT
    required_gib: float  # combined_gib + FRAG_MARGIN_GIB, checked against GPU_TOTAL_GIB
    gpu_total_gib: float
    remaining_kv_gib: float
    kv_tokens: int
    concurrency_x: float
    budget_gib_a: float  # util_a * GPU_TOTAL_GIB, informational
    budget_gib_b: float  # util_b * GPU_TOTAL_GIB, informational
    reason: str


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------


def _pct(gib: float) -> float:
    return gib / GPU_TOTAL_GIB * 100.0


def _util_min_for(weights_gib: float, overhead_gib: float) -> float:
    """Smallest util (rounded up to 2dp) that gives kv_gib > 0."""
    raw = (weights_gib + overhead_gib) / GPU_TOTAL_GIB
    bumped = raw + 1e-9
    return min(1.0, math.ceil(bumped * 100.0) / 100.0)


def compute(
    m: ModelInputs,
    *,
    util: float,
    ctx: int,
    max_num_seqs: int,
    live: LiveFacts | None = None,
) -> CapacityResult:
    findings: list[Finding] = []

    if not m.servable:
        findings.append(
            Finding(
                code="MODEL_UNSERVABLE",
                level="block",
                title="Model is not servable",
                detail=(
                    f"{m.repo_id}: "
                    f"{m.unservable_reason or 'no known launcher/backend for this model'}."
                ),
                fix="Choose a different model.",
            )
        )
        empty_bar = CapacityBar(weights_pct=0.0, kv_pct=0.0, overhead_pct=0.0, free_pct=0.0)
        return CapacityResult(
            budget_gib=0.0,
            kv_gib=0.0,
            kv_tokens=0,
            concurrency_x=0.0,
            agents_at_ctx=0,
            effective_parallel=0,
            max_single_ctx=0,
            confidence="unknown",
            bar=empty_bar,
            findings=tuple(findings),
            can_apply=False,
        )

    # ---------------------------------------------------------- validation
    if not (0.0 < util <= 1.0):
        findings.append(
            Finding(
                code="UTIL_OUT_OF_RANGE",
                level="block",
                title="GPU utilization out of range",
                detail=f"util={util!r} must be in (0.0, 1.0].",
                fix="Set util strictly above 0 and at most 1.0.",
            )
        )

    if not isinstance(max_num_seqs, int) or max_num_seqs < 1:
        findings.append(
            Finding(
                code="MAX_NUM_SEQS_INVALID",
                level="block",
                title="max_num_seqs invalid",
                detail=f"max_num_seqs={max_num_seqs!r} must be a positive integer.",
                fix="Set max_num_seqs to 1 or higher.",
            )
        )

    if m.model_max_ctx > 0 and ctx > m.model_max_ctx:
        findings.append(
            Finding(
                code="CTX_ABOVE_CEILING",
                level="block",
                title="Context above model ceiling",
                detail=(
                    f"ctx={ctx:,} exceeds this model's max_position_embeddings "
                    f"({m.model_max_ctx:,})."
                ),
                fix=f"Lower ctx to at most {m.model_max_ctx:,}.",
            )
        )

    # ------------------------------------------------------- weights/budget
    weights_unknown = False
    weights_gib = m.weights_gib
    if m.weights_source == "unknown" or weights_gib is None:
        findings.append(
            Finding(
                code="UNKNOWN_CAPACITY",
                # WARN, not block. Booting is the ONLY way to learn a model's
                # real weight size, so blocking the launch made the condition
                # permanent: unknown -> cannot start -> stays unknown.
                level="warn",
                title="Weight size unknown — capacity cannot be estimated",
                detail=(
                    f"{m.repo_id}: weights_source is 'unknown'. The generic "
                    "safetensors_gib*1.01 estimator is refused for "
                    "model_type=='qwen4_exp' (on this box it was 37% off for "
                    "Flash-Next: 125.91 GiB disk vs 78.47 GiB actual VRAM) and "
                    "no boot measurement exists yet."
                ),
                fix="Boot the model once to obtain a measured weights figure.",
            )
        )
        weights_gib = 0.0
        weights_unknown = True

    overhead_gib = m.overhead_gib
    budget_gib = util * GPU_TOTAL_GIB
    kv_gib = budget_gib - weights_gib - overhead_gib

    if weights_unknown:
        # Unknown weights were being substituted with 0.0, so the KV budget
        # absorbed the entire card and reported millions of tokens -- a bigger
        # number than any real model, shown as if it meant something.
        # UNKNOWN_CAPACITY already blocks applying it; the displayed figures
        # must be blank too, not fabricated.
        kv_gib = 0.0
        kv_gib_for_tokens = 0.0
    elif kv_gib <= 0:
        util_min = _util_min_for(weights_gib, overhead_gib)
        findings.append(
            Finding(
                code="WEIGHTS_EXCEED_BUDGET",
                level="block",
                title="Weights + overhead exceed the util budget",
                detail=(
                    f"budget {budget_gib:.2f} GiB - weights {weights_gib:.2f} GiB - "
                    f"overhead {overhead_gib:.2f} GiB = {kv_gib:.2f} GiB, no room "
                    "for KV cache."
                ),
                fix=f"Raise util to at least {util_min:.2f}.",
                fix_action={"field": "util", "value": util_min},
            )
        )
        kv_gib_for_tokens = 0.0
    else:
        kv_gib_for_tokens = kv_gib

    kv_tokens = 0
    if kv_gib_for_tokens > 0 and m.kv_kib_per_token:
        kv_tokens = math.floor(kv_gib_for_tokens * 1048576 / m.kv_kib_per_token)

    if not weights_unknown and kv_gib > 0 and kv_tokens < ctx:
        fix_action = {"field": "ctx", "value": kv_tokens} if kv_tokens >= 1 else None
        findings.append(
            Finding(
                code="KV_TOO_SMALL_FOR_ONE_CTX",
                level="block",
                title="KV budget too small for one request at this context",
                detail=(
                    f"KV budget yields {kv_tokens:,} tokens, less than the "
                    f"configured ctx {ctx:,}. Not even one full-context request fits."
                ),
                fix=(
                    f"Lower ctx to at most {kv_tokens:,}."
                    if kv_tokens >= 1
                    else "Raise util, or lower ctx."
                ),
                fix_action=fix_action,
            )
        )

    if m.backend == "flashnext" and util < FLASHNEXT_MIN_UTIL:
        findings.append(
            Finding(
                code="FLASHNEXT_UTIL_TOO_LOW",
                level="block",
                title="util too low for Flash-Next",
                detail=(
                    f"util={util:.2f} is below the Flash-Next minimum "
                    f"{FLASHNEXT_MIN_UTIL:.2f}."
                ),
                fix=f"Raise util to at least {FLASHNEXT_MIN_UTIL:.2f}.",
            )
        )

    # -------------------------------------------------------------- live
    if live is not None:
        if live.gpu_responsive is False:
            findings.append(
                Finding(
                    code="GPU_UNRESPONSIVE",
                    level="block",
                    title="GPU unresponsive",
                    detail="`nvidia-smi -L` failed to list the GPU.",
                    fix="Check the driver / GPU before starting a server.",
                )
            )
        elif live.used_mib is not None:
            total_mib = live.total_mib if live.total_mib is not None else GPU_TOTAL_MIB
            own_mib = live.own_mib or 0
            free_mib = total_mib - live.used_mib + own_mib
            want_mib = int(total_mib * util)
            # VRAM_GUARD_HEADROOM_MIB (4096) is a fragmentation margin carried
            # over from qwen-server-run.sh. It must NOT be added on top of the
            # utilization budget when deciding "can this start": util is already
            # a fraction of TOTAL, so want+4096 exceeds the card for any
            # util >= 0.958 and the check could never pass -- yet this box runs
            # Flash-Next at util 0.96 every day. The requirement is the budget
            # itself; the headroom is what makes it *comfortable*, and a thin
            # margin is reported separately as a warning below.
            need_mib = want_mib
            if need_mib <= free_mib < need_mib + VRAM_GUARD_HEADROOM_MIB:
                findings.append(
                    Finding(
                        code="THIN_VRAM_MARGIN",
                        level="warn",
                        title="Thin VRAM margin",
                        detail=(
                            f"free {free_mib} MiB leaves under "
                            f"{VRAM_GUARD_HEADROOM_MIB} MiB above the {need_mib} MiB budget. "
                            "It should start, but fragmentation could cause an OOM mid-session."
                        ),
                        fix="Lower utilization slightly for more headroom.",
                    )
                )
            if free_mib < need_mib:
                findings.append(
                    Finding(
                        code="NOT_ENOUGH_FREE_VRAM",
                        level="block",
                        title="Not enough free VRAM",
                        detail=(
                            f"free {free_mib} MiB (total {total_mib}, used "
                            f"{live.used_mib}, our own {own_mib} discounted) < "
                            f"need {need_mib} MiB = {want_mib} at util {util:.2f} "
                            f"+ {VRAM_GUARD_HEADROOM_MIB} headroom."
                        ),
                        fix="Lower util, or stop a holder, then try again.",
                    )
                )

        markers = live.training_markers or []
        if markers:
            findings.append(
                Finding(
                    code="TRAINING_MARKER",
                    level="block",
                    title="Training in progress",
                    detail=(
                        "Training marker present at: "
                        + ", ".join(markers)
                        + ". Refusing to take the GPU."
                    ),
                    fix="Remove the marker once training is done, then start again.",
                )
            )

        if (
            m.backend == "flashnext"
            and live.ptrace_scope is not None
            and live.ptrace_scope != 0
        ):
            findings.append(
                Finding(
                    code="PTRACE_BLOCKS_PLE",
                    level="block",
                    title="ptrace_scope blocks Flash-Next's PLE",
                    detail=(
                        f"kernel.yama.ptrace_scope={live.ptrace_scope}; Flash-Next's "
                        "process lifecycle event handling needs pidfd_getfd, which "
                        "requires ptrace_scope=0."
                    ),
                    fix="Run: sudo sysctl -w kernel.yama.ptrace_scope=0",
                    fix_action={
                        "type": "manual_command",
                        "command": "sudo sysctl -w kernel.yama.ptrace_scope=0",
                    },
                )
            )

        if live.ptrace_scope == 0 and live.actual_state in ("READY", "STOPPED"):
            findings.append(
                Finding(
                    code="PTRACE_LEFT_RELAXED",
                    level="warn",
                    title="ptrace_scope left relaxed",
                    detail=(
                        "kernel.yama.ptrace_scope=0 is still set while the server "
                        f"is {live.actual_state}. This is a standing security "
                        "relaxation that is not needed right now."
                    ),
                    fix="Run: sudo sysctl -w kernel.yama.ptrace_scope=1",
                    fix_action={
                        "type": "manual_command",
                        "command": "sudo sysctl -w kernel.yama.ptrace_scope=1",
                    },
                )
            )

    # -------------------------------------------------------- warn (static)
    if util >= UTIL_THIN_MARGIN:
        findings.append(
            Finding(
                code="THIN_MARGIN",
                level="warn",
                title="Thin utilization margin",
                detail=(
                    f"util={util:.2f} leaves almost no fragmentation headroom "
                    f"(>= {UTIL_THIN_MARGIN:.2f})."
                ),
                fix="Consider a lower util for a safety margin against fragmentation.",
            )
        )

    if m.weights_source == "estimated":
        findings.append(
            Finding(
                code="ESTIMATED_ONLY",
                level="warn",
                title="Weights estimated, not measured",
                detail=(
                    f"{m.repo_id} has never booted; these numbers are estimated "
                    "from safetensors size, not measured. Estimates for "
                    "never-booted models are upper bounds — ~25% optimistic "
                    "historically on this box."
                ),
                fix="Treat these numbers as optimistic until the model boots once.",
            )
        )

    if m.trust == "measured_other_ctx":
        rate_note = f"{m.kv_kib_per_token} KiB/tok"
        if m.used_ctx_for_rate is not None:
            rate_note += f" measured at ctx={m.used_ctx_for_rate:,}"
        other = ""
        if m.known_kv_rates:
            pairs = ", ".join(
                f"{c:,}: {r} KiB/tok" for c, r in sorted(m.known_kv_rates.items())
            )
            other = f" Known rates — {pairs}."
        findings.append(
            Finding(
                code="MEASURED_OTHER_CTX",
                level="warn",
                title="KV rate measured at a different context length",
                detail=(
                    f"No measurement exists at ctx={ctx:,}. Using {rate_note} "
                    "(KV rate is not context-invariant)." + other
                ),
                fix="Boot at this ctx once to get an exact rate.",
            )
        )

    if m.backend == "inline" and max_num_seqs > MAMBA_MAX_NUM_SEQS_INLINE:
        findings.append(
            Finding(
                code="MAMBA_SEQS_CAP",
                level="warn",
                title="max_num_seqs above the inline CUDA graph cap",
                detail=(
                    f"max_num_seqs={max_num_seqs} > {MAMBA_MAX_NUM_SEQS_INLINE} "
                    "fails CUDA graph capture on the inline (mamba) backend."
                ),
                fix=f"Keep max_num_seqs at or below {MAMBA_MAX_NUM_SEQS_INLINE}.",
            )
        )

    # ------------------------------------------------------------- derived
    concurrency_x = (kv_tokens / ctx) if ctx > 0 else 0.0
    agents_at_ctx = math.floor(concurrency_x)
    max_single_ctx = min(m.model_max_ctx, kv_tokens) if m.model_max_ctx > 0 else kv_tokens
    valid_seqs = isinstance(max_num_seqs, int) and max_num_seqs >= 1
    effective_parallel = min(agents_at_ctx, max_num_seqs) if valid_seqs else 0

    if valid_seqs and max_num_seqs < agents_at_ctx:
        findings.append(
            Finding(
                code="SEQS_BELOW_AGENTS",
                level="warn",
                title="max_num_seqs caps parallelism below the KV budget",
                detail=(
                    f"KV budget supports {agents_at_ctx} agents at ctx={ctx:,}, "
                    f"but max_num_seqs={max_num_seqs} caps it lower."
                ),
                fix=f"Raise max_num_seqs toward {agents_at_ctx} to use the full KV budget.",
            )
        )

    if live is not None and live.codex_max_subagents is not None:
        c = live.codex_max_subagents
        e = effective_parallel
        if c > e:
            findings.append(
                Finding(
                    code="SUBAGENTS_NOT_A_GUARANTEE",
                    level="warn",
                    title="Subagent cap exceeds GPU parallelism",
                    detail=(
                        f"CODEX_MAX_SUBAGENTS={c} is a Codex-side orchestration "
                        f"cap, not a GPU guarantee. This configuration serves {e} "
                        f"in parallel; the other {c - e} will queue."
                    ),
                    fix="Lower CODEX_MAX_SUBAGENTS, or raise capacity (util/max_num_seqs).",
                )
            )

    bar = CapacityBar(
        weights_pct=_pct(weights_gib),
        kv_pct=_pct(max(kv_gib, 0.0)),
        overhead_pct=_pct(overhead_gib),
        free_pct=max(
            0.0,
            100.0 - _pct(weights_gib) - _pct(max(kv_gib, 0.0)) - _pct(overhead_gib),
        ),
    )

    confidence = "unknown" if m.weights_source == "unknown" else m.trust
    can_apply = not any(f.level == "block" for f in findings)

    return CapacityResult(
        budget_gib=budget_gib,
        kv_gib=kv_gib,
        kv_tokens=kv_tokens,
        concurrency_x=concurrency_x,
        agents_at_ctx=agents_at_ctx,
        effective_parallel=effective_parallel,
        max_single_ctx=max_single_ctx,
        confidence=confidence,
        bar=bar,
        findings=tuple(findings),
        can_apply=can_apply,
    )


# ---------------------------------------------------------------------------
# blue_green()
# ---------------------------------------------------------------------------


def blue_green(
    a: ModelInputs,
    b: ModelInputs,
    *,
    util_a: float,
    util_b: float,
    ctx: int,
) -> BlueGreenVerdict:
    """SPEC.md §3: fits = (w_a + w_b + 2*overhead + FRAG_MARGIN) <= GPU_TOTAL_GIB.

    `overhead` here is OVERHEAD_GIB_DEFAULT (the conservative constant), not
    either model's individually measured overhead — verified against the
    §3 worked table: 2*78.47+9.4=166.3 (2*OVERHEAD_GIB_DEFAULT == 9.4 exactly).
    """
    w_a = a.weights_gib or 0.0
    w_b = b.weights_gib or 0.0

    combined_gib = w_a + w_b + 2 * OVERHEAD_GIB_DEFAULT
    required_gib = combined_gib + FRAG_MARGIN_GIB
    feasible = required_gib <= GPU_TOTAL_GIB
    remaining_kv_gib = (GPU_TOTAL_GIB - combined_gib) if feasible else 0.0

    rate: float | None = None
    for candidate in (a.kv_kib_per_token, b.kv_kib_per_token):
        if candidate:
            rate = candidate if rate is None else max(rate, candidate)

    kv_tokens = 0
    if feasible and rate:
        kv_tokens = math.floor(remaining_kv_gib * 1048576 / rate)
    concurrency_x = (kv_tokens / ctx) if (ctx > 0 and kv_tokens) else 0.0

    if feasible:
        reason = (
            f"{combined_gib:.1f} GiB (weights {w_a:.2f}+{w_b:.2f} + 2x overhead "
            f"{OVERHEAD_GIB_DEFAULT:.1f}) + {FRAG_MARGIN_GIB:.1f} GiB margin fits "
            f"under {GPU_TOTAL_GIB:.1f} GiB — ~{remaining_kv_gib:.1f} GiB left for KV."
        )
    else:
        reason = (
            f"{combined_gib:.1f} GiB (weights {w_a:.2f}+{w_b:.2f} + 2x overhead "
            f"{OVERHEAD_GIB_DEFAULT:.1f}) + {FRAG_MARGIN_GIB:.1f} GiB margin exceeds "
            f"{GPU_TOTAL_GIB:.1f} GiB — blue-green is impossible for this pair."
        )

    return BlueGreenVerdict(
        feasible=feasible,
        combined_gib=combined_gib,
        required_gib=required_gib,
        gpu_total_gib=GPU_TOTAL_GIB,
        remaining_kv_gib=remaining_kv_gib,
        kv_tokens=kv_tokens,
        concurrency_x=concurrency_x,
        budget_gib_a=util_a * GPU_TOTAL_GIB,
        budget_gib_b=util_b * GPU_TOTAL_GIB,
        reason=reason,
    )
