"""Servedeck preflight checks — SPEC.md §3's environmental blockers.

capacity.py is deliberately pure (no I/O — SPEC.md §3's docstring says so
explicitly): every environmental fact it reacts to has to be collected by
something else and handed in via ``LiveFacts``. This module is that
something else, for the checks that are about the MACHINE rather than the
chosen (util, ctx, max_num_seqs) numbers for one model — those live-facts
constitute a standalone go/no-go read a caller can ask for before even
picking a config (SPEC.md §8: ``GET /api/preflight``), and supervisor.py
runs the backend-specific subset of them as its "RUNNING + nothing ->
preflight -> start" reconciliation step (SPEC.md §6).

Every check returns one :class:`PreflightCheck` — ``{id, ok, level, title,
detail, fix_command}`` — never raises, and never runs a privileged command
itself: where a fix needs root (relaxing ``ptrace_scope``), ``fix_command``
is a copyable string for a human to paste, exactly like capacity.py's own
``PTRACE_BLOCKS_PLE``/``PTRACE_LEFT_RELAXED`` findings (SPEC.md absolute
rule 4 — Servedeck itself never runs ``sudo``).

Two checks here have no equivalent in capacity.py's finding list at all:
``VENV_MISSING`` and ``LAUNCHER_MISSING``. SPEC.md's correction C9 names
"FATAL: vLLM venv not found" (a stale ``AlgoTrading-llm-audit/.venv-llm``
path) as 12 of 72 recorded exits — the single biggest boot-failure class on
this box — and it is a check that can be answered with a `Path.is_dir()`
before ever invoking a launcher, so it belongs in a *preflight* set even
though SPEC.md §3's finding-code table doesn't name it. Refusing to check
it just because it's not in that table would be exactly the kind of
substitution SPEC.md's "measured, do not substitute" warns against in the
other direction: a real, previously-observed failure mode left unguarded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from servedeck import config, gpu, paths

Level = Literal["block", "warn", "info"]

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreflightCheck:
    id: str
    ok: bool
    level: Level
    title: str
    detail: str
    fix_command: str | None = None


# ---------------------------------------------------------------------------
# Backend name literals — mirrors registry.KNOWN_ARCHS's value set. Not
# imported from registry.py (that module is about model discovery, not
# backend identity) — these two strings are the whole of it.
# ---------------------------------------------------------------------------

BACKEND_FLASHNEXT = "flashnext"
BACKEND_INLINE = "inline"

# capacity.py's own PTRACE_BLOCKS_PLE / PTRACE_LEFT_RELAXED thresholds,
# duplicated here as literals rather than imported: capacity.py is pure and
# deliberately has no knowledge of *how* ptrace_scope was read, and this
# module has no need of capacity.py's arithmetic — only its threshold. The
# value is a single, load-bearing constant (0), unlikely to drift silently.
_PTRACE_SCOPE_REQUIRED = 0

_STOPPED_OR_READY = frozenset({"STOPPED", "READY"})


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_gpu_responsive() -> PreflightCheck:
    """SPEC.md §3 GPU_UNRESPONSIVE: `nvidia-smi -L` failing => never start,
    never restart (also SPEC.md §6's Xid-classification rule, reused
    verbatim by supervisor.py for the same predicate on an already-running
    server)."""
    alive = gpu.gpu_alive()
    return PreflightCheck(
        id="GPU_UNRESPONSIVE",
        ok=alive,
        level="block",
        title="GPU responsive" if alive else "GPU unresponsive",
        detail=(
            "`nvidia-smi -L` listed the GPU."
            if alive
            else "`nvidia-smi -L` failed to list the GPU — check the driver "
            "before starting or restarting anything."
        ),
        fix_command=None,
    )


def check_training_marker() -> PreflightCheck:
    """SPEC.md §3 TRAINING_MARKER — qwen-server-run.sh:75-79's own guard 1:
    presence of ANY of the three marker paths means "keep off the GPU"."""
    hits = [str(p) for p in paths.TRAINING_MARKERS if p.exists()]
    ok = not hits
    return PreflightCheck(
        id="TRAINING_MARKER",
        ok=ok,
        level="block",
        title="No training in progress" if ok else "Training in progress",
        detail=(
            "None of the three training-marker paths are present."
            if ok
            else "Training marker present at: " + ", ".join(hits) + ". Refusing to take the GPU."
        ),
        fix_command=None if ok else f"rm {hits[0]}   # only once training has actually finished",
    )


def _read_ptrace_scope() -> int | None:
    try:
        return int(paths.PTRACE_PATH.read_text().strip())
    except (OSError, ValueError):
        return None


def check_ptrace_scope(backend: str | None, actual_state: str | None = None) -> PreflightCheck | None:
    """SPEC.md §3 PTRACE_BLOCKS_PLE (block, flashnext only) / PTRACE_LEFT_RELAXED
    (warn, any backend, only while READY/STOPPED). Returns None when neither
    applies (backend is not flashnext AND actual_state doesn't call for the
    "left relaxed" warning) — there is nothing meaningful to report.

    SPEC.md correction C3: Flash-Next's own relaxation of this sysctl
    (serve.sh:53-66) goes through `sudo sysctl`, which silently no-ops
    without an interactive tty (a systemd ExecStart has none) — "it only
    appears to work right now because ptrace_scope happens to be 0 on this
    boot." This check is what lets Servedeck surface that BEFORE a launch
    attempt burns 4-10 minutes discovering it the hard way.
    """
    scope = _read_ptrace_scope()
    if scope is None:
        if backend != BACKEND_FLASHNEXT:
            return None
        return PreflightCheck(
            id="PTRACE_BLOCKS_PLE",
            ok=False,
            level="block",
            title="ptrace_scope unreadable",
            detail=f"Could not read {paths.PTRACE_PATH} to confirm Flash-Next's PLE handoff will work.",
            fix_command="cat /proc/sys/kernel/yama/ptrace_scope",
        )

    if backend == BACKEND_FLASHNEXT and scope != _PTRACE_SCOPE_REQUIRED:
        return PreflightCheck(
            id="PTRACE_BLOCKS_PLE",
            ok=False,
            level="block",
            title="ptrace_scope blocks Flash-Next's PLE",
            detail=(
                f"kernel.yama.ptrace_scope={scope}; Flash-Next's process lifecycle "
                "event handling needs pidfd_getfd, which requires ptrace_scope=0. "
                "NOTE (SPEC.md C3): serve.sh relaxes this itself via `sudo sysctl`, "
                "which silently fails with no tty (e.g. under systemd) — do not "
                "assume serve.sh will fix this for you."
            ),
            fix_command="sudo sysctl -w kernel.yama.ptrace_scope=0",
        )

    if scope == _PTRACE_SCOPE_REQUIRED and actual_state in _STOPPED_OR_READY:
        return PreflightCheck(
            id="PTRACE_LEFT_RELAXED",
            ok=True,
            level="warn",
            title="ptrace_scope left relaxed",
            detail=(
                "kernel.yama.ptrace_scope=0 is still set while the server is "
                f"{actual_state}. This is a standing security relaxation that is "
                "not needed right now."
            ),
            fix_command="sudo sysctl -w kernel.yama.ptrace_scope=1",
        )

    if backend == BACKEND_FLASHNEXT:
        return PreflightCheck(
            id="PTRACE_BLOCKS_PLE",
            ok=True,
            level="block",
            title="ptrace_scope OK for Flash-Next",
            detail=f"kernel.yama.ptrace_scope={scope} — Flash-Next's PLE handoff will work.",
            fix_command=None,
        )
    return None


def _configured(backend: str | None, attr: str) -> Path | None:
    """A path this backend declares in servedeck.toml, or None.

    Never raises: an unreadable config must degrade to the built-in defaults
    below, not turn every preflight into a crash.
    """
    try:
        b = config.get().backend(backend)
    except Exception:  # noqa: BLE001
        return None
    value = getattr(b, attr, None) if b is not None else None
    return Path(value) if value is not None else None


def check_venv(backend: str | None) -> PreflightCheck | None:
    """SPEC.md correction C9: "FATAL: vLLM venv not found" at a stale path
    was 12 of 72 recorded exits — the single largest boot-failure class.
    `backend=None` (not yet chosen) returns None: there is nothing to check.
    The venv comes from servedeck.toml when the backend declares one, so a
    backend added by configuration gets this check too rather than silently
    skipping the largest boot-failure class there is."""
    venv_dir = _configured(backend, "venv")
    if venv_dir is None:
        if backend == BACKEND_FLASHNEXT:
            venv_dir = paths.VENV_NEXT_DIR
        elif backend == BACKEND_INLINE:
            venv_dir = paths.VENV_LLM_DIR
        else:
            return None
    ok = venv_dir.is_dir() and (venv_dir / "bin" / "python").exists()
    return PreflightCheck(
        id="VENV_MISSING",
        ok=ok,
        level="block",
        title="vLLM venv present" if ok else "vLLM venv not found",
        detail=(
            f"{venv_dir} exists with a bin/python."
            if ok
            else f"{venv_dir} is missing or has no bin/python — this is SPEC.md "
            "correction C9's dominant boot-failure class (a stale venv path)."
        ),
        fix_command=None,
    )


def check_launcher(backend: str | None) -> PreflightCheck | None:
    """The launcher script SPEC.md §1's "delegation, not reimplementation"
    invokes must actually exist and be executable, or a start attempt fails
    before it can even produce a log to diagnose from. The path comes from
    servedeck.toml when the backend is declared there."""
    script = _configured(backend, "launcher")
    if script is None:
        if backend == BACKEND_FLASHNEXT:
            script = paths.SERVE_SH
        elif backend == BACKEND_INLINE:
            script = paths.SERVER_RUN_SH
        else:
            return None
    ok = script.is_file() and os.access(script, os.X_OK)
    return PreflightCheck(
        id="LAUNCHER_MISSING",
        ok=ok,
        level="block",
        title="Launcher script present" if ok else "Launcher script missing or not executable",
        detail=(f"{script} exists and is executable." if ok else f"{script} is missing or not executable."),
        fix_command=None if ok else f"chmod +x {script}",
    )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def run_preflight(backend: str | None = None, actual_state: str | None = None) -> list[PreflightCheck]:
    """Run every applicable check and return the full list (blocking AND
    passing AND warn/info) — callers decide what to do with ``ok``/``level``
    themselves (SPEC.md §8's ``GET /api/preflight`` renders all of them;
    supervisor.py's own preflight-before-start gate only treats
    ``level == "block" and not ok`` as a hard stop).

    ``backend`` is optional: with it omitted (or unrecognized), the two
    backend-specific checks (``VENV_MISSING``, ``LAUNCHER_MISSING``) and the
    Flash-Next-only half of the ptrace check are simply skipped rather than
    guessed at — there is no "environmental blocker" to report about a
    model that hasn't been chosen yet.
    """
    checks: list[PreflightCheck] = [
        check_gpu_responsive(),
        check_training_marker(),
    ]
    ptrace = check_ptrace_scope(backend, actual_state)
    if ptrace is not None:
        checks.append(ptrace)
    venv_check = check_venv(backend)
    if venv_check is not None:
        checks.append(venv_check)
    launcher_check = check_launcher(backend)
    if launcher_check is not None:
        checks.append(launcher_check)
    return checks


def blocking_failures(checks: list[PreflightCheck]) -> list[PreflightCheck]:
    """The subset that should actually stop a start attempt: level=='block'
    and not ok. (A passing 'block'-level check, e.g. PTRACE_BLOCKS_PLE when
    the sysctl is fine, is still returned by run_preflight() for display —
    it is not itself a failure.)"""
    return [c for c in checks if c.level == "block" and not c.ok]
