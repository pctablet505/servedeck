"""The rolling window over the last 100 finished requests.

These test the two things the panel's honesty rests on: that a percentile over
bucket-bounded observations comes back as an INTERVAL (never a fabricated
point), and that a window holding fewer than 100 requests says so instead of
padding.
"""

from __future__ import annotations

import math

from servedeck import reqstats
from servedeck.reqstats import RequestWindow, nearest_rank

#: The bucket edges vLLM actually publishes for request_prompt_tokens, read off
#: http://127.0.0.1:8001/metrics on 2026-09-10 (recorded at
#: tests/fixtures/metrics_hist_live.txt). Coarse on purpose: this is the real
#: quantisation the percentiles inherit, and a test using invented fine-grained
#: edges would not exercise it.
EDGES = [
    1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0,
    5000.0, 10000.0, 20000.0, 50000.0, 100000.0, 200000.0, math.inf,
]


def cumulative(counts: dict[float, int]) -> list[tuple[float, float]]:
    """Turn {le: per-bucket count} into the cumulative (le, count) vector."""
    running = 0.0
    out = []
    for e in EDGES:
        running += counts.get(e, 0)
        out.append((e, running))
    return out


def feed(win: RequestWindow, *polls: dict[float, int], ceiling=None, ts=None) -> None:
    """Push a baseline then each poll's per-bucket counts."""
    total: dict[float, int] = {}
    win.observe(cumulative(total), hist_sum=0.0, hist_count=0.0, ceiling=ceiling, ts=ts)
    hist_sum = 0.0
    hist_count = 0.0
    for poll in polls:
        for e, c in poll.items():
            total[e] = total.get(e, 0) + c
            # A plausible sum: each request sits at its bucket's upper edge.
            hist_sum += c * (e if e != math.inf else 200000.0)
            hist_count += c
        win.observe(
            cumulative(total), hist_sum=hist_sum, hist_count=hist_count,
            ceiling=ceiling, ts=ts,
        )


# --------------------------------------------------------------------------
# nearest_rank on known inputs
# --------------------------------------------------------------------------
def test_percentiles_on_a_known_sequence() -> None:
    """1..100 has an unambiguous nearest-rank answer for every percentile:
    rank = ceil(p/100 * n), so p50 is the 50th value and p99 the 99th."""
    vals = list(range(1, 101))
    assert nearest_rank(vals, 50) == 50
    assert nearest_rank(vals, 90) == 90
    assert nearest_rank(vals, 99) == 99
    assert nearest_rank(vals, 100) == 100


def test_percentile_rank_rounds_up_so_p90_really_covers_90_percent() -> None:
    """The rank is ceil(p/100 * n), not floor.

    With 3 samples, floor puts p50 at rank 1 -- the SMALLEST value -- so a
    "median" prompt size of 30k reports as 10k and the panel recommends three
    times as many agents as fit. The two only differ when p/100*n is not a
    whole number, which is why a 1..100 fixture alone does not catch it.
    """
    assert nearest_rank([10, 20, 30], 50) == 20
    assert nearest_rank([1, 2, 3, 4, 5, 6, 7], 90) == 7
    assert nearest_rank(list(range(1, 11)), 95) == 10


def test_percentile_is_at_or_above_the_requested_fraction_of_samples() -> None:
    """The defining property, checked directly: at least ceil(p/100 * n) of the
    samples are <= the value returned. A percentile that under-reports the tail
    is a recommendation to over-subscribe."""
    for n in range(1, 40):
        vals = list(range(1, n + 1))
        for p in (50, 90, 99):
            v = nearest_rank(vals, p)
            covered = sum(1 for x in vals if x <= v)
            assert covered >= math.ceil(p / 100 * n), (
                f"p{p} of {n} samples covered only {covered}"
            )


def test_percentile_of_all_identical_input_is_that_value() -> None:
    """The degenerate case the panel hits with a steady agent loop: every
    request the same size. Every percentile must be that size -- an
    interpolating percentile can drift off it, which would make the
    recommendation move while the workload did not."""
    vals = [4096] * 37
    assert nearest_rank(vals, 50) == 4096
    assert nearest_rank(vals, 90) == 4096
    assert nearest_rank(vals, 99) == 4096


def test_percentile_of_one_sample_is_that_sample() -> None:
    assert nearest_rank([777], 50) == 777
    assert nearest_rank([777], 99) == 777


def test_percentile_of_nothing_is_none_not_zero() -> None:
    """An empty window has no p90. Returning 0 would divide the KV pool by
    zero-ish and recommend an unbounded number of agents."""
    assert nearest_rank([], 90) is None


def test_p99_never_exceeds_the_max() -> None:
    """Nearest rank is an ORDER STATISTIC: it is always an observed value, so
    p99 cannot land above the largest sample however few samples there are."""
    for n in range(1, 60):
        vals = list(range(1, n + 1))
        assert nearest_rank(vals, 99) <= vals[-1]


# --------------------------------------------------------------------------
# The window: fewer than 100
# --------------------------------------------------------------------------
def test_a_short_window_reports_its_real_size_and_says_it_is_partial() -> None:
    """The owner asked for "the statistics of last 100 requests". With 7
    requests observed, the answer is statistics over 7 -- flagged as such --
    not seven values padded out to a hundred."""
    win = RequestWindow()
    feed(win, {5000.0: 7})
    st = win.stats()
    assert st.n == 7
    assert st.capacity == 100
    assert st.partial is True
    assert st.to_dict()["n"] == 7


def test_a_full_window_is_not_partial_and_rolls() -> None:
    win = RequestWindow()
    feed(win, {5000.0: 100})
    assert win.stats().partial is False
    # 20 more requests, all much larger: the window must have rolled, so the
    # p50 has moved off the small bucket.
    feed(RequestWindow(), {5000.0: 1})   # unrelated, keeps flake pressure off
    win.observe(
        cumulative({5000.0: 100, 50000.0: 100}),
        hist_sum=1.0e9, hist_count=200.0,
    )
    st = win.stats()
    assert st.n == 100
    assert st.p50.hi == 50000.0, "the window kept 100 stale requests instead of rolling"


def test_first_scrape_is_a_baseline_and_adds_nothing() -> None:
    """The lifetime histogram must not be poured into the window on the first
    poll: those requests finished before Servedeck was watching, and counting
    them would make a freshly-started dashboard claim a 100-request window."""
    win = RequestWindow()
    added = win.observe(cumulative({50000.0: 900}), hist_sum=2.7e7, hist_count=900.0)
    assert added == 0
    assert win.stats().n == 0


# --------------------------------------------------------------------------
# Bucket quantisation is reported, never hidden
# --------------------------------------------------------------------------
def test_a_bucket_bounded_percentile_is_an_interval() -> None:
    """Ten requests in (20000, 50000]. The p90 is not "50,000" and not
    "35,000" -- it is the interval, and `exact` says so."""
    win = RequestWindow()
    feed(win, {50000.0: 10})
    p90 = win.stats().p90
    assert (p90.lo, p90.hi) == (20001.0, 50000.0)
    assert p90.exact is False
    assert win.stats().to_dict()["p90"] == {"lo": 20001, "hi": 50000, "exact": False}


def test_a_lone_request_in_a_poll_window_is_recorded_exactly() -> None:
    """delta(_count) == 1 makes delta(_sum) that one request's token count.
    This is what keeps the window from being all intervals under interactive
    traffic."""
    win = RequestWindow()
    win.observe(cumulative({}), hist_sum=0.0, hist_count=0.0)
    win.observe(cumulative({50000.0: 1}), hist_sum=31234.0, hist_count=1.0)
    st = win.stats()
    assert st.n == 1
    assert st.exact_n == 1
    assert (st.p90.lo, st.p90.hi) == (31234.0, 31234.0)
    assert st.p90.exact is True


def test_an_impossible_exact_value_falls_back_to_the_bucket() -> None:
    """If _sum and _bucket disagree -- a scrape torn mid-write, say -- the
    interval is the safe statement. Trusting the sum here would put a 3-token
    request in the panel as the p90 of a 30k workload."""
    win = RequestWindow()
    win.observe(cumulative({}), hist_sum=0.0, hist_count=0.0)
    win.observe(cumulative({50000.0: 1}), hist_sum=3.0, hist_count=1.0)
    p90 = win.stats().p90
    assert (p90.lo, p90.hi) == (20001.0, 50000.0)
    assert p90.exact is False


def test_exact_count_is_reported_so_the_page_can_qualify_the_number() -> None:
    win = RequestWindow()
    win.observe(cumulative({}), hist_sum=0.0, hist_count=0.0)
    win.observe(cumulative({50000.0: 1}), hist_sum=31234.0, hist_count=1.0)
    # cumulative(): 7 total in that bucket, i.e. 6 new since the last poll.
    win.observe(cumulative({50000.0: 7}), hist_sum=31234.0 + 6 * 30000, hist_count=7.0)
    st = win.stats()
    assert (st.n, st.exact_n) == (7, 1)


def test_the_open_ended_bucket_is_capped_by_max_model_len() -> None:
    """A request above the last finite edge is bounded by --max-model-len,
    which the engine enforces. Without the ceiling the p90 is infinite and the
    recommendation divides by infinity."""
    win = RequestWindow()
    feed(win, {math.inf: 4}, ceiling=262144)
    p90 = win.stats().p90
    assert p90.hi == 262144.0
    assert p90.lo == 200001.0


def test_without_a_ceiling_the_open_bucket_stays_unbounded_rather_than_guessing() -> None:
    win = RequestWindow()
    feed(win, {math.inf: 4})
    assert win.stats().p90.hi == math.inf
    # ...and serialises as null, so the page prints "> 200,000" instead of a
    # number nothing measured.
    assert win.stats().to_dict()["p90"]["hi"] is None


# --------------------------------------------------------------------------
# Restarts and outages
# --------------------------------------------------------------------------
def test_a_counter_reset_empties_the_window() -> None:
    """Counters going backwards means a NEW engine process -- possibly a new
    model. Requests from the old one must not survive into statistics used to
    size the new one."""
    win = RequestWindow()
    feed(win, {50000.0: 20})
    assert win.stats().n == 20
    win.observe(cumulative({}), hist_sum=0.0, hist_count=0.0)   # restart
    assert win.stats().n == 0


def test_a_failed_scrape_drops_the_baseline_but_keeps_the_window() -> None:
    """An outage's worth of requests must not all be stamped with the instant
    of the next successful scrape (which would make age_s claim the window
    covers a second). They are dropped; what we did watch is kept."""
    win = RequestWindow()
    feed(win, {5000.0: 5})
    assert win.stats().n == 5
    win.drop_baseline()
    # 400 requests happened during the outage; the next scrape re-baselines.
    win.observe(cumulative({5000.0: 405}), hist_sum=2.0e6, hist_count=405.0)
    assert win.stats().n == 5, "outage traffic was back-filled into the window"


# --------------------------------------------------------------------------
# Interval soundness
# --------------------------------------------------------------------------
def test_percentile_intervals_are_always_valid() -> None:
    """lo <= hi at every rank, even when exact points interleave with
    intervals. This holds because lo_j <= hi_j for each observation, so order
    statistics preserve it -- the test pins the property, not the proof."""
    win = RequestWindow()
    win.observe(cumulative({}), hist_sum=0.0, hist_count=0.0)
    win.observe(cumulative({50000.0: 1}), hist_sum=21000.0, hist_count=1.0)
    win.observe(cumulative({20000.0: 3, 100000.0: 2}), hist_sum=3.0e5, hist_count=6.0)
    st = win.stats()
    for p in (st.p50, st.p90, st.p99, st.peak):
        assert p.lo <= p.hi, p


def test_empty_stats_payload_has_every_key_a_filled_one_has() -> None:
    """The page reads the same fields in both states; a missing key would make
    "no data yet" render as a crash rather than as a sentence."""
    win = RequestWindow()
    feed(win, {5000.0: 3})
    assert set(reqstats.EMPTY_STATS) == set(win.stats().to_dict())


def test_provenance_string_names_both_limitations() -> None:
    """The one sentence that has to be on screen. Bucket quantisation and the
    partial view are separate defects in the estimate and both must be said."""
    assert "bucket" in reqstats.PROVENANCE.lower()
    assert "watching" in reqstats.PROVENANCE.lower()
