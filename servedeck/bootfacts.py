"""vLLM's own memory accounting, read back off a boot that succeeded.

The capacity panel predicts a KV pool; vLLM measures one. Until 2026-09-18 the
two were never compared, and they had drifted 1.14 GiB apart — enough that the
panel green-lit ``util 0.95`` at 262,144 tokens for Flash-Next (``can_apply:
true``, ``findings: []``) and the engine died 2.5 minutes later with::

    ValueError: To serve at least one request with the model's max seq len
    (262144), (7.17 GiB KV cache is needed, which is larger than the available
    KV cache memory (6.74 GiB).

The whole of that drift is one number going stale. ``capacity`` works in
``kv = budget - weights - overhead``; ``overhead`` for this model was 4.47 GiB,
fitted from a boot on **2026-08-28**. Since then the model gained MTP-3
speculative decoding (a draft model with its own activations and CUDA graphs)
and the ``OffloadingConnector``. Re-fitted from the live boots of 2026-09-18 it
is **5.62 GiB** (5.615 at util 0.97, 5.621 at util 0.98 — two boots, 0.006 GiB
apart, so this is a constant and not a fit to one point). 5.62 - 4.47 = 1.15 GiB
is the gap, and it is the whole of it: with 5.62 the model reproduces every
measured pool on this card, including the 6.72 GiB at util 0.95 that the engine
refused as too small for a 262,144-token request.

It went stale because nothing ever re-measured it. ``append_observation``
existed, had three tests asserting it appends atomically, creates parent
directories and never overwrites — and was called by **nothing**. The store had
not been written since 2026-09-02, so every prediction replayed August, and
would have kept replaying it for as long as the panel existed.

A note on which quantity to store. vLLM prints its own decomposition
(``consumed``/``activation``/``CUDAGraph``), and it is tempting to sum those.
Do not: that sum is 5.29 GiB, and the residual against vLLM's own budget is
5.01, because the parts do not name memory already on the card at startup and
because vLLM sizes its budget against CUDA's device total (94.97 GiB here)
while ``capacity`` sizes against nvidia-smi's (95.59 GiB). Only the residual
fitted against **capacity's own basis** — 5.62 — makes the panel reproduce the
engine, because it absorbs both discrepancies by construction. Re-basing
capacity onto CUDA's total instead was tried and reverted: every stored
``overhead`` in the repo — including the 27B's, measured on a card that has not
booted under v2 — is fitted against the nvidia-smi basis and silently absorbs
the same 0.62 GiB, so moving the basis without re-fitting all of them broke six
calibrations and improved no prediction.

The lines parsed here are vLLM 0.29's; a release that reformats them makes
every field ``None``, which callers treat as "learned nothing this boot" and
never as zero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

__all__ = ["BootFacts", "parse", "overhead_gib_from"]


#: vLLM 0.29 `worker.py`, printed once after profiling, before the KV cache is
#: allocated. Captured live 2026-09-18 22:23 from model-flashnext:
#:
#:     Free memory on device (94.41/94.97 GiB) on startup. Desired GPU memory
#:     utilization is (0.97, 92.12 GiB). Actual usage is 81.7 GiB for consumed
#:     memory (weights + non-torch), 1.78 GiB for peak activation, and 0.28 GiB
#:     for CUDAGraph memory. [...] Current kv cache memory in use is 8.64 GiB.
#:
#: Split into three independent patterns rather than one long alternation: a
#: release that changes the tail of the sentence then costs the tail's fields,
#: not the device total that the budget basis depends on.
_FREE = re.compile(
    r"Free memory on device \(([\d.]+)/([\d.]+) GiB\) on startup",
)
_DESIRED = re.compile(
    r"Desired GPU memory utilization is \(([\d.]+), ([\d.]+) GiB\)",
)
_USAGE = re.compile(
    r"Actual usage is ([\d.]+) GiB for consumed memory \(weights \+ non-torch\), "
    r"([\d.]+) GiB for peak activation, and ([\d.]+) GiB for CUDAGraph memory",
)
_KV_IN_USE = re.compile(r"Current kv cache memory in use is ([\d.]+) GiB")

#: The measured KV pool, printed by `kv_cache_utils.py`. Thousands separators
#: are locale-independent here (vLLM formats with `,`), hence the explicit
#: comma class rather than `\d+`.
_KV_SIZE = re.compile(
    r"GPU KV cache size: ([\d,]+) tokens"
    r"(?:, Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x)?",
)


@dataclass(frozen=True)
class BootFacts:
    """What one successful boot measured about its own memory.

    Every field is optional and independently sourced: a partial parse is
    useful (``device_total_gib`` alone fixes the budget basis) and is never
    filled in with a default that would read as a measurement.
    """

    #: CUDA's total for the device — the number vLLM multiplies by
    #: `--gpu-memory-utilization`. NOT nvidia-smi's total, which is larger by
    #: the driver's reserve (~638 MiB on the RTX PRO 6000 here).
    device_total_gib: float | None = None
    #: Free on the device when the engine started, i.e. before it allocated
    #: anything. Below `device_total_gib` by whatever else holds the card.
    device_free_gib: float | None = None
    #: The utilisation this boot actually ran at, as vLLM resolved it.
    util: float | None = None
    #: `util * device_total_gib` — the whole budget, KV included.
    budget_gib: float | None = None
    #: Weights + non-torch allocations (NCCL, cuBLAS, the CUDA context).
    consumed_gib: float | None = None
    activation_gib: float | None = None
    cudagraph_gib: float | None = None
    #: The KV pool the engine ended up with.
    kv_gib: float | None = None
    #: The same pool in tokens — the figure the panel is really predicting.
    kv_tokens: int | None = None
    max_concurrency: float | None = None

    @property
    def complete(self) -> bool:
        """True when this boot can calibrate the budget AND the overhead."""
        return (
            self.device_total_gib is not None
            and self.budget_gib is not None
            and self.kv_gib is not None
            and self.kv_tokens is not None
        )


def _f(value: str) -> float:
    return float(value.replace(",", ""))


def parse(lines: Iterable[str]) -> BootFacts:
    """Read a unit's journal into :class:`BootFacts`.

    Later lines win. A unit restarted in place has several boots in one
    journal, and the most recent one is the one that is running.
    """
    found: dict[str, object] = {}
    for line in lines:
        if (m := _FREE.search(line)) is not None:
            found["device_free_gib"] = _f(m.group(1))
            found["device_total_gib"] = _f(m.group(2))
        if (m := _DESIRED.search(line)) is not None:
            found["util"] = _f(m.group(1))
            found["budget_gib"] = _f(m.group(2))
        if (m := _USAGE.search(line)) is not None:
            found["consumed_gib"] = _f(m.group(1))
            found["activation_gib"] = _f(m.group(2))
            found["cudagraph_gib"] = _f(m.group(3))
        if (m := _KV_IN_USE.search(line)) is not None:
            found["kv_gib"] = _f(m.group(1))
        if (m := _KV_SIZE.search(line)) is not None:
            found["kv_tokens"] = int(_f(m.group(1)))
            if m.group(3) is not None:
                found["max_concurrency"] = _f(m.group(3))
    return BootFacts(**found)  # type: ignore[arg-type]


def overhead_gib_from(
    facts: BootFacts, weights_gib: float | None, basis_gib: float | None
) -> float | None:
    """The capacity model's ``overhead`` term, as a calibration residual.

    ``capacity.compute`` works in ``kv = budget - weights - overhead`` with
    ``budget = util * GPU_TOTAL_GIB``, so the only definition of ``overhead``
    that makes the panel reproduce what the engine measured is the leftover::

        overhead = util * GPU_TOTAL_GIB - weights - kv

    ``basis_gib`` is that ``GPU_TOTAL_GIB`` — passed in rather than imported so
    this module stays free of capacity's import-time GPU detection, and so a
    caller can fit against whatever basis it actually predicts with. It is
    deliberately NOT ``facts.budget_gib``: that is vLLM's budget, computed from
    CUDA's smaller device total, and fitting against it would leave every
    prediction 0.62 GiB adrift from the basis the panel uses.

    ``None`` when the boot did not print enough, or when the numbers do not
    hang together (a negative residual means the parse is wrong, not that
    overhead is negative).
    """
    if weights_gib is None or basis_gib is None or facts.util is None:
        return None
    if facts.kv_gib is None:
        return None
    residual = facts.util * basis_gib - weights_gib - facts.kv_gib
    if residual <= 0:
        return None
    return round(residual, 3)


def observation(
    *,
    repo_id: str,
    backend: str,
    facts: BootFacts,
    weights_gib: float | None,
    basis_gib: float | None,
    ctx: int | None,
    max_num_seqs: int | None,
    kv_cache_dtype: str | None,
    ts: str,
) -> dict | None:
    """One observation, shaped exactly like the entries already in the store.

    Pure: the caller does the I/O. ``None`` when the boot printed nothing
    worth recording — a store entry with no measurement in it would still be
    picked as "most recent" by the resolver and would blank out a good older
    one, so silence is the correct output, not an empty record.

    ``inputs.util`` is the utilisation vLLM RESOLVED, read back off its own
    log, not the one the caller requested. A launch that passes no
    ``--gpu-memory-utilization`` gets one derived from free VRAM, and that
    derived value is the one the measurement belongs to.
    """
    if facts.kv_tokens is None and facts.kv_gib is None:
        return None
    overhead = overhead_gib_from(facts, weights_gib, basis_gib)
    measured: dict = {
        "weights_gib": weights_gib,
        "kv_gib": facts.kv_gib,
        "kv_tokens": facts.kv_tokens,
        "concurrency_x": facts.max_concurrency,
        "overhead_gib": overhead,
        # Not from nvidia-smi: the engine's own view of what it took, which is
        # the only one that excludes whatever else is on the card.
        "gpu_used_mib": (
            round((facts.device_total_gib - facts.device_free_gib) * 1024)
            if facts.device_total_gib is not None and facts.device_free_gib is not None
            else None
        ),
        "device_total_gib": facts.device_total_gib,
    }
    return {
        "repo_id": repo_id,
        "backend": backend,
        "ts": ts,
        "inputs": {
            "util": facts.util,
            "max_model_len": ctx,
            "max_num_seqs": max_num_seqs,
            "kv_cache_dtype": kv_cache_dtype,
        },
        "measured": measured,
    }
