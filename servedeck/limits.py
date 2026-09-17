"""The four hardware numbers the capacity arithmetic needs, and nothing else.

This is what survives of ``config.py``. That file was 283 lines because a
model's identity lived in it — launcher paths, ports, env-var maps, log files,
per-backend architecture lists — and all of that moved to ``models.toml``
(REDESIGN-2026-09-12 §2.1, R1). What could not move is genuinely about the
*card*, not about any model:

* how much VRAM it has,
* how much of it is neither weights nor KV (activations, CUDA graphs),
* how much fragmentation margin to warn at,
* which lock files mean "something else wants the GPU".

``settings.py`` deliberately does not carry these: it answers "where do I
listen and where is the registry", is imported by everything, and must stay
free of the nvidia-smi call that resolving a GPU size needs.

Every value is environment-overridable and every default is the measured one,
with its provenance written beside it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Limits",
    "DEFAULT_OVERHEAD_GIB",
    "DEFAULT_FRAG_MARGIN_MIB",
    "DEFAULT_TRAINING_MARKERS",
    "detect_gpu_total_mib",
    "training_markers",
    "get",
    "reset",
]

#: Activations + CUDA graphs: VRAM that is neither weights nor KV. Measured at
#: 4.3-4.5 GiB for two very different models; 4.7 errs high, which makes every
#: prediction built on it conservative.
DEFAULT_OVERHEAD_GIB = 4.7

#: Fragmentation margin above the utilisation budget. Reported as "thin",
#: never as "impossible" — adding it to the *requirement* makes any util above
#: ~0.958 unsatisfiable on a card whose util is a fraction of total, which is a
#: bug this project shipped once.
DEFAULT_FRAG_MARGIN_MIB = 4096

#: A "lock file" convention: if one of these exists, something else wants the
#: GPU and servedeck stands down. Both paths are read-only existence checks,
#: and both belong to tools outside this repo — which is why they are written
#: here rather than discovered.
DEFAULT_TRAINING_MARKERS: tuple[str, ...] = (
    str(Path.home() / "Projects" / "local_llm" / "run" / "training_in_progress"),
    str(Path.home() / ".cache" / "algotrading" / "training_in_progress"),
)

#: 0, not a guess: capacity treats an unknown total as "cannot compute" and
#: blocks, which is safer than sizing a KV cache against a number nobody
#: measured.
FALLBACK_GPU_TOTAL_MIB = 0


@dataclass(frozen=True)
class Limits:
    gpu_total_mib: int
    overhead_gib: float
    frag_margin_mib: int


def detect_gpu_total_mib() -> int:
    """Total VRAM in MiB from nvidia-smi, or 0 if it cannot be determined."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return FALLBACK_GPU_TOTAL_MIB
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode != 0:
            return FALLBACK_GPU_TOTAL_MIB
        return int(out.stdout.strip().splitlines()[0].strip())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return FALLBACK_GPU_TOTAL_MIB


def training_markers() -> tuple[str, ...]:
    """Marker paths, from ``$SERVEDECK_TRAINING_MARKERS`` (colon-separated) or
    the built-in defaults. Read at call time, not cached, because a test that
    sets the variable expects the next check to see it."""
    raw = os.environ.get("SERVEDECK_TRAINING_MARKERS")
    if raw is not None:
        return tuple(p for p in raw.split(":") if p)
    return DEFAULT_TRAINING_MARKERS


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


_cached: Limits | None = None


def get() -> Limits:
    global _cached
    if _cached is None:
        total = _env_int("SERVEDECK_GPU_TOTAL_MIB", 0) or detect_gpu_total_mib()
        _cached = Limits(
            gpu_total_mib=total,
            overhead_gib=_env_float("SERVEDECK_OVERHEAD_GIB", DEFAULT_OVERHEAD_GIB),
            frag_margin_mib=_env_int("SERVEDECK_FRAG_MARGIN_MIB", DEFAULT_FRAG_MARGIN_MIB),
        )
    return _cached


def reset() -> None:
    """Drop the cache so the next :func:`get` re-reads the environment."""
    global _cached
    _cached = None
