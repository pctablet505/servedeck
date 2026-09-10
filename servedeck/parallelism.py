"""How many agents can run in parallel against the live server, right now.

THE QUESTION
------------
"How many subagents do I keep running?" is a KV-*admission* question on this
box, not a compute one. The GPU is not the constraint; the KV pool is. Once the
requests in flight need more pool than exists, vLLM preempts — evicts a
sequence's KV and recomputes it later — and aggregate throughput goes DOWN, not
sideways. The measured table below has a cell where that is explicit: at 30k
prompts, N=8 is *slower* than N=4.

THE FORMULA THAT DOES NOT WORK
------------------------------
The obvious one is ``floor(kv_pool_tokens / prompt_tokens)``. It is what vLLM
itself publishes as ``kv_cache_max_concurrency`` (280,813 / 262,144 = 1.0712 on
this server — literally pool over max_model_len). At 8k prompts it predicts
280,813 / 8,102 = **34** concurrent requests. Measurement says 8. It is wrong by
4x, and wrong in the dangerous direction, because it assumes a request's KV cost
is proportional to its length with no fixed part. On a hybrid attention/Mamba
model it is not: every sequence takes a fixed state page whatever its length.
That fixed part is ~12.8k pool-tokens here — bigger than a whole 8k prompt.

THE CALIBRATION
---------------
So the cost model is fitted to measurement, not derived from token counts. The
anchors are the measured KV-pool cost of one request, taken on this exact
server (Qwen3.8-Flash-Next NVFP4, FP8 PLE table, bf16 Mamba SSM cache,
``--gpu-memory-utilization 0.96``, pool = 280,813 tokens):

    615 prompt tokens ->  5.0% of the pool
  8,102               -> 10.5%
 30,116               -> 23.3%
105,108               -> 67%

Cost is interpolated piecewise-linearly between anchors and extrapolated with
the nearest segment's slope outside them, in ABSOLUTE pool tokens — the fixed
per-sequence page does not shrink when the pool does, so scaling the anchors as
fractions of a different pool would be wrong.

THE HEADROOM CONSTANT
---------------------
``floor(pool / cost(L))`` alone reproduces three of the four measured cells and
misses the 8k one by one (it says 9.5 fit; measurement says 8 is the last
useful). Planning to sit at 100% pool occupancy is wrong anyway — the engine
runs with ``--watermark 0.10`` and the last admitted sequence is the one that
gets preempted first. So the recommendation applies a headroom factor.

``HEADROOM = 0.94`` is not a taste decision; it is the only round value in the
band that reproduces every measured cell:

  * 8k needs ``floor(9.524 * m) == 8``  -> m in [0.840, 0.945)
  * 30k needs ``floor(4.292 * m) == 4`` -> m in [0.932, 1.165)
  * intersection: **[0.932, 0.945)**

Below that band the 8k cell over-recommends 9; above it the 30k cell
under-recommends 3. ``test_parallelism.py`` pins both edges, so a later "let's
round it to 0.9" cannot pass silently.

WHAT THIS IS NOT
----------------
Model-specific. The anchors were measured on one model at one quantisation. A
different model — especially a pure-attention one, which has no fixed Mamba
page — has a different cost curve, and the recommendation for it will be too
conservative. The UI says so; :func:`calibration_note` is the string it says it
with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

#: The KV pool the anchors below were measured against. Used only to turn the
#: measured *fractions* into absolute pool-token costs once, at import.
CALIBRATION_POOL_TOKENS = 280_813

#: The model the anchors were measured on. Compared against the served name so
#: the UI can downgrade its own claim when something else is running.
CALIBRATION_MODEL = "qwen38-flash-next"

#: (prompt tokens, fraction of the KV pool one such request occupies).
KV_COST_ANCHORS: tuple[tuple[int, float], ...] = (
    (615, 0.050),
    (8_102, 0.105),
    (30_116, 0.233),
    (105_108, 0.67),
)

#: See "THE HEADROOM CONSTANT" above. Admissible band, from the measurements:
HEADROOM = 0.94
HEADROOM_BAND = (0.932, 0.945)

#: The measured table this module must reproduce: prompt tokens -> the largest
#: concurrency that still helped. Exported so the test does not restate it and
#: so the two can never drift apart.
MEASURED_MAX_USEFUL_CONCURRENCY: tuple[tuple[int, int], ...] = (
    (615, 16),
    (8_102, 8),
    (30_116, 4),
    (105_108, 1),
    (245_000, 1),
)

_ANCHORS: tuple[tuple[float, float], ...] = tuple(
    (float(tokens), frac * CALIBRATION_POOL_TOKENS) for tokens, frac in KV_COST_ANCHORS
)


def kv_cost_tokens(prompt_tokens: float) -> float:
    """Pool tokens one request of ``prompt_tokens`` occupies.

    Piecewise-linear through the measured anchors; the first/last segment's
    slope extrapolates below/above them. Never returns <= 0: a request always
    costs at least the fixed per-sequence page.
    """
    length = max(0.0, float(prompt_tokens))
    pts = _ANCHORS
    if length <= pts[0][0]:
        slope = (pts[1][1] - pts[0][1]) / (pts[1][0] - pts[0][0])
        return max(1.0, pts[0][1] - (pts[0][0] - length) * slope)
    for (l0, c0), (l1, c1) in zip(pts, pts[1:]):
        if length <= l1:
            return c0 + (c1 - c0) * (length - l0) / (l1 - l0)
    (l0, c0), (l1, c1) = pts[-2], pts[-1]
    return c1 + (length - l1) * (c1 - c0) / (l1 - l0)


#: The per-sequence floor: what a zero-length request would still cost. This is
#: the number that kills the naive pool/prompt formula, so the UI shows it.
FIXED_COST_TOKENS = round(kv_cost_tokens(0))


@dataclass(frozen=True)
class Recommendation:
    """A recommended parallel-agent count, with every step of its arithmetic.

    Every field here is rendered somewhere on the page. The point is that an
    operator can check the number rather than trust it.
    """

    #: The prompt size the recommendation was computed for, and where it came
    #: from ("p90 of the last 47 requests", say).
    prompt_tokens: int
    basis: str
    #: Live KV pool, from vllm:cache_config_info -- a measurement, not an
    #: estimate.
    pool_tokens: int
    cost_tokens: int
    #: Which two anchors the cost interpolates between, or None when the
    #: prompt size is outside the calibrated range and the cost is
    #: extrapolated.
    segment: tuple[int, int] | None
    extrapolated: bool
    headroom: float
    #: pool/cost, before headroom and before the clamp.
    fit_n: float
    #: floor(fit_n * headroom), before the clamp.
    n_before_clamp: int
    max_num_seqs: int | None
    n: int
    clamped: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "basis": self.basis,
            "pool_tokens": self.pool_tokens,
            "cost_tokens": self.cost_tokens,
            "fixed_cost_tokens": FIXED_COST_TOKENS,
            "segment": list(self.segment) if self.segment else None,
            "extrapolated": self.extrapolated,
            "headroom": self.headroom,
            "fit_n": round(self.fit_n, 2),
            "n_before_clamp": self.n_before_clamp,
            "max_num_seqs": self.max_num_seqs,
            "n": self.n,
            "clamped": self.clamped,
        }


def recommend(
    *,
    pool_tokens: int,
    prompt_tokens: float,
    max_num_seqs: int | None = None,
    headroom: float = HEADROOM,
    basis: str = "",
) -> Recommendation:
    """The headline number: how many agents to run in parallel.

    ``pool_tokens`` must be the LIVE pool (``kv_cache_size_tokens`` off
    ``vllm:cache_config_info``), never a constant — a server relaunched at a
    different ``--gpu-memory-utilization`` has a different pool, and a
    recommendation computed from yesterday's is a recommendation to
    over-subscribe.
    """
    cost = kv_cost_tokens(prompt_tokens)
    fit = pool_tokens / cost if cost > 0 else 0.0
    # max(1, ...): the answer to "how many agents" is never zero. One request
    # too big for the pool still runs -- vLLM chunks its prefill -- it just
    # runs alone. That is the 245k row of the measured table.
    n_before = max(1, int(math.floor(fit * headroom)))
    n = n_before
    clamped = False
    if max_num_seqs is not None and max_num_seqs > 0 and n > max_num_seqs:
        n, clamped = max_num_seqs, True

    segment: tuple[int, int] | None = None
    extrapolated = True
    length = max(0.0, float(prompt_tokens))
    for (l0, _c0), (l1, _c1) in zip(_ANCHORS, _ANCHORS[1:]):
        if l0 <= length <= l1:
            segment, extrapolated = (int(l0), int(l1)), False
            break

    return Recommendation(
        prompt_tokens=int(round(prompt_tokens)),
        basis=basis,
        pool_tokens=int(pool_tokens),
        cost_tokens=int(round(cost)),
        segment=segment,
        extrapolated=extrapolated,
        headroom=headroom,
        fit_n=fit,
        n_before_clamp=n_before,
        max_num_seqs=max_num_seqs,
        n=n,
        clamped=clamped,
    )


def calibration_note(served_model: str | None) -> str:
    """The provenance sentence the panel prints under the recommendation.

    Never omitted. A number this actionable, derived from a curve fitted to one
    model, has to carry the name of that model on screen.
    """
    base = (
        f"KV cost curve calibrated on {CALIBRATION_MODEL} "
        f"({CALIBRATION_POOL_TOKENS:,}-token pool, 4 measured points)"
    )
    if served_model and served_model != CALIBRATION_MODEL:
        return (
            f"{base}. The running model is {served_model} — a different "
            f"architecture has a different per-sequence cost, so treat this as "
            f"an upper bound, not a measurement."
        )
    return base + "."
