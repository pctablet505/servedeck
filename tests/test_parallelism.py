"""The recommended parallel-agent count.

The load-bearing test here is :func:`test_formula_reproduces_the_measured_table`.
The recommendation is a claim about what this GPU will do, and the only thing
that makes it a claim rather than an opinion is that it reproduces the
concurrency actually measured on it. If the formula and the measurements
disagree, the formula is wrong.
"""

from __future__ import annotations

import math

import pytest

from servedeck import parallelism
from servedeck.parallelism import (
    HEADROOM,
    HEADROOM_BAND,
    KV_COST_ANCHORS,
    MEASURED_MAX_USEFUL_CONCURRENCY,
    kv_cost_tokens,
    recommend,
)

#: The live pool on this box: vllm:cache_config_info's kv_cache_size_tokens,
#: which is the same number the boot log prints as "GPU KV cache size".
POOL = 280_813
#: --max-num-seqs of the running server, read off its own command line.
MAX_SEQS = 16


# --------------------------------------------------------------------------
# The measurements
# --------------------------------------------------------------------------
@pytest.mark.parametrize(("prompt_tokens", "measured"), MEASURED_MAX_USEFUL_CONCURRENCY)
def test_formula_reproduces_the_measured_table(prompt_tokens: int, measured: int) -> None:
    """Benchmarked on this exact server: ~500-token prompts saturate at 16
    concurrent, 8k at 8, 30k at 4 (N=8 there goes BACKWARDS), 105k and 245k at
    1. The recommendation must land on each of those, not near them."""
    rec = recommend(pool_tokens=POOL, prompt_tokens=prompt_tokens, max_num_seqs=MAX_SEQS)
    assert rec.n == measured, (
        f"{prompt_tokens} tok: recommended {rec.n}, measured {measured} "
        f"(fit={rec.fit_n:.3f}, cost={rec.cost_tokens})"
    )


def test_the_naive_pool_over_prompt_formula_is_the_thing_being_rejected() -> None:
    """floor(pool / prompt_tokens) -- what vLLM itself publishes as
    kv_cache_max_concurrency -- says 34 agents fit at 8k prompts. Measurement
    says 8. This test exists so nobody "simplifies" the cost curve back into
    that division: it pins the 4x gap the curve is there to close."""
    naive = POOL // 8_102
    assert naive >= 30, "the naive formula's answer changed; re-derive the gap"
    rec = recommend(pool_tokens=POOL, prompt_tokens=8_102, max_num_seqs=MAX_SEQS)
    assert rec.n * 3 < naive, (
        "the recommendation has drifted back toward pool/prompt_tokens"
    )


def test_cost_curve_passes_through_every_measured_anchor() -> None:
    """Interpolation, so each anchor must come back exactly. A fitted curve
    that misses its own anchors is a curve nobody measured."""
    for prompt_tokens, frac in KV_COST_ANCHORS:
        assert kv_cost_tokens(prompt_tokens) == pytest.approx(frac * POOL, rel=1e-9)


def test_a_request_has_a_fixed_cost_no_matter_how_short() -> None:
    """The whole reason pool/prompt_tokens fails: on this hybrid model every
    sequence takes a fixed state page. It is ~12.8k pool-tokens -- larger than
    an entire 8k prompt -- so a zero-length request is not free."""
    assert parallelism.FIXED_COST_TOKENS > 10_000
    assert kv_cost_tokens(0) == pytest.approx(parallelism.FIXED_COST_TOKENS, abs=1.0)
    assert kv_cost_tokens(1) > 0


def test_cost_is_monotone_in_prompt_size() -> None:
    """A longer prompt can never cost less pool. A non-monotone curve would let
    the recommendation go UP as the workload got heavier."""
    prev = -1.0
    for n in (0, 1, 100, 615, 4_000, 8_102, 20_000, 30_116, 60_000, 105_108, 262_144):
        cost = kv_cost_tokens(n)
        assert cost > prev, f"cost fell at {n}"
        prev = cost


def test_recommendation_is_monotone_non_increasing_in_prompt_size() -> None:
    last = math.inf
    for n in range(500, 260_000, 2_500):
        rec = recommend(pool_tokens=POOL, prompt_tokens=n, max_num_seqs=MAX_SEQS)
        assert rec.n <= last, f"recommendation rose at {n} tokens"
        last = rec.n


# --------------------------------------------------------------------------
# The headroom constant is determined by the measurements, not chosen
# --------------------------------------------------------------------------
def test_headroom_sits_inside_the_band_the_measurements_allow() -> None:
    lo, hi = HEADROOM_BAND
    assert lo <= HEADROOM < hi


def test_below_the_band_the_30k_cell_under_recommends() -> None:
    """Pinning the lower edge: this is why the constant cannot just be 0.9.
    At 0.931 the 30k workload is told to run 3 agents where 4 was measured --
    a third of the throughput thrown away for nothing."""
    lo, _hi = HEADROOM_BAND
    below = recommend(
        pool_tokens=POOL, prompt_tokens=30_116, max_num_seqs=MAX_SEQS, headroom=lo - 0.001
    )
    assert below.n == 3


def test_above_the_band_the_8k_cell_over_recommends() -> None:
    """Pinning the upper edge: at 0.945 the 8k workload is told to run 9 agents
    where 8 was the last useful one. Over-recommending is the failure that
    costs money -- it is the one that ends in preemption."""
    _lo, hi = HEADROOM_BAND
    above = recommend(
        pool_tokens=POOL, prompt_tokens=8_102, max_num_seqs=MAX_SEQS, headroom=hi
    )
    assert above.n == 9


# --------------------------------------------------------------------------
# The clamp
# --------------------------------------------------------------------------
def test_recommendation_is_clamped_to_max_num_seqs() -> None:
    """However much KV is free, the scheduler will not run more sequences than
    --max-num-seqs concurrently. Recommending more is recommending a queue."""
    rec = recommend(pool_tokens=POOL, prompt_tokens=615, max_num_seqs=4)
    assert rec.n == 4
    assert rec.clamped is True
    assert rec.n_before_clamp > 4, "the clamp must be visible, not silent"


def test_the_clamp_never_raises_the_recommendation() -> None:
    """max_num_seqs is a ceiling, not a target. A server started with
    --max-num-seqs 64 must not be told to run 64 agents at 100k prompts."""
    rec = recommend(pool_tokens=POOL, prompt_tokens=105_108, max_num_seqs=64)
    assert rec.n == 1
    assert rec.clamped is False


def test_an_unknown_max_num_seqs_does_not_fabricate_a_ceiling() -> None:
    """When the process cmdline is unreadable, there is no ceiling to apply.
    Defaulting to some number would be a claim about a server we cannot see."""
    rec = recommend(pool_tokens=POOL, prompt_tokens=615, max_num_seqs=None)
    assert rec.max_num_seqs is None
    assert rec.clamped is False
    assert rec.n == rec.n_before_clamp


# --------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------
def test_a_prompt_too_big_for_the_pool_still_recommends_one() -> None:
    """245k tokens costs more than the whole pool, but the request runs -- vLLM
    chunks its prefill. The answer is 1 agent, never 0."""
    rec = recommend(pool_tokens=POOL, prompt_tokens=245_000, max_num_seqs=MAX_SEQS)
    assert rec.n == 1
    assert rec.fit_n < 1.0, "this case is only meaningful when the fit is below 1"


def test_a_smaller_pool_recommends_fewer_agents() -> None:
    """The pool must be the LIVE one. A server relaunched at a lower
    --gpu-memory-utilization has less KV, and the same workload then supports
    fewer agents."""
    big = recommend(pool_tokens=POOL, prompt_tokens=8_102, max_num_seqs=MAX_SEQS)
    small = recommend(pool_tokens=POOL // 2, prompt_tokens=8_102, max_num_seqs=MAX_SEQS)
    assert small.n < big.n


# --------------------------------------------------------------------------
# Auditability: every step of the arithmetic travels to the page
# --------------------------------------------------------------------------
def test_payload_carries_the_whole_arithmetic() -> None:
    """The panel shows the working so an operator can check the number. Every
    term of `floor(pool / cost * headroom)` clamped to max_num_seqs has to be
    in the payload."""
    d = recommend(
        pool_tokens=POOL, prompt_tokens=30_116, max_num_seqs=MAX_SEQS, basis="p90 of 100"
    ).to_dict()
    for key in (
        "pool_tokens", "cost_tokens", "fixed_cost_tokens", "headroom",
        "fit_n", "n_before_clamp", "max_num_seqs", "n", "clamped",
        "prompt_tokens", "basis", "segment", "extrapolated",
    ):
        assert key in d, f"missing {key}"
    # ...and the arithmetic in it must actually reconstruct the answer.
    assert d["fit_n"] == pytest.approx(d["pool_tokens"] / d["cost_tokens"], rel=1e-3)
    assert d["n_before_clamp"] == max(1, math.floor(d["fit_n"] * d["headroom"]))


def test_interpolated_and_extrapolated_are_distinguishable() -> None:
    """Inside the calibrated range the panel can name the two anchors it sat
    between. Outside it, it must say the cost was extrapolated rather than
    imply a measurement exists there."""
    inside = recommend(pool_tokens=POOL, prompt_tokens=20_000, max_num_seqs=MAX_SEQS)
    assert inside.extrapolated is False
    assert inside.segment == (8_102, 30_116)
    outside = recommend(pool_tokens=POOL, prompt_tokens=245_000, max_num_seqs=MAX_SEQS)
    assert outside.extrapolated is True
    assert outside.segment is None


def test_calibration_note_names_the_model_it_was_measured_on() -> None:
    assert parallelism.CALIBRATION_MODEL in parallelism.calibration_note(None)


def test_calibration_note_warns_when_a_different_model_is_serving() -> None:
    """The curve was fitted to one hybrid model. A pure-attention model has no
    fixed Mamba page and a completely different cost curve; the panel must not
    keep presenting this one as if it applied."""
    note = parallelism.calibration_note("llama-3-70b")
    assert "llama-3-70b" in note
    assert "upper bound" in note
