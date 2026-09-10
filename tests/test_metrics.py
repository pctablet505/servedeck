"""MetricsPoller regression tests — SPEC.md §8 telemetry.

There were no tests for metrics.py at all, which is why every bug below
survived. Each test names the defect it pins.

The poller is exercised through a fake httpx client rather than a live
server: the real /metrics endpoint belongs to a vLLM process that must not
be started, stopped or leaned on by a unit test.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from servedeck import metrics


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class _Resp:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


class _FakeClient:
    """Serves a scripted list of responses (str body, int status, or an
    Exception to raise) one per get()."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls = 0

    async def get(self, url: str, timeout: float | None = None) -> _Resp:
        self.calls += 1
        item = self.script.pop(0) if self.script else ""
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):
            return _Resp(item[0], item[1])
        return _Resp(200, item)


def _exposition(
    *,
    prompt_total: float = 0.0,
    prompt_cached: float = 0.0,
    gen_total: float = 0.0,
    prefill_sum: float | None = None,
    prefill_count: float | None = None,
) -> str:
    """A minimal /metrics page in the real exposition format.

    Label sets copy the shape vLLM actually emits (every family carries
    model_name), so the parser is exercised on the labelled form.
    """
    m = 'model_name="glm53-flash",engine="0"'
    lines = [
        "# HELP vllm:prompt_tokens_total Number of prefill tokens processed.",
        "# TYPE vllm:prompt_tokens_total counter",
        f"vllm:prompt_tokens_total{{{m}}} {prompt_total}",
        f"vllm:prompt_tokens_cached_total{{{m}}} {prompt_cached}",
        f"vllm:generation_tokens_total{{{m}}} {gen_total}",
        f"vllm:kv_cache_usage_perc{{{m}}} 0.0",
        f"vllm:num_requests_running{{{m}}} 0",
        f"vllm:num_requests_waiting{{{m}}} 0",
    ]
    if prefill_sum is not None:
        lines.append(f"vllm:request_prefill_time_seconds_sum{{{m}}} {prefill_sum}")
    if prefill_count is not None:
        lines.append(f"vllm:request_prefill_time_seconds_count{{{m}}} {prefill_count}")
    return "\n".join(lines) + "\n"


def _scrape_all(script: list[Any], clock: list[float]) -> list[metrics.MetricsSnapshot]:
    """Run one scrape per scripted response, advancing a fake monotonic clock."""
    ticks = iter(clock)
    poller = metrics.MetricsPoller("http://127.0.0.1:8002", monotonic=lambda: next(ticks))
    client = _FakeClient(script)
    out: list[metrics.MetricsSnapshot] = []

    async def run() -> None:
        for _ in range(len(script)):
            out.append(await poller.scrape(client))  # type: ignore[arg-type]

    asyncio.run(run())
    return out


# --------------------------------------------------------------------------
# Prefill throughput — the feature
# --------------------------------------------------------------------------
def test_prefill_tok_s_is_a_rate_over_the_poll_window() -> None:
    """Prefill speed must be derived the same way gen tok/s is: a delta of a
    Prometheus counter over the poll interval.

    2058 computed prompt tokens in 2 s is 1029 tok/s — the figure vLLM's own
    log prints as "Avg prompt throughput" for the same window.
    """
    snaps = _scrape_all(
        [
            _exposition(prompt_total=1000, gen_total=100),
            _exposition(prompt_total=3058, gen_total=128),
        ],
        clock=[100.0, 102.0],
    )
    assert snaps[0].prefill_tok_s is None, "no baseline yet — must not invent a rate"
    assert snaps[1].prefill_tok_s == pytest.approx(1029.0)
    assert snaps[1].gen_tok_s == pytest.approx(14.0)


def test_prefill_excludes_prefix_cache_hits() -> None:
    """vllm:prompt_tokens_total counts cached tokens too, but a token served
    from the prefix cache costs no prefill compute.

    vLLM's own throughput line uses prompt_token_stats.computed
    (v1/metrics/loggers.py:147), i.e. total minus cached. Counting the cached
    ones inflates prefill speed without bound on a cache hit — 4.16 s of real
    work reported as if it were 0.53 s.
    """
    snaps = _scrape_all(
        [
            _exposition(prompt_total=1000, prompt_cached=0),
            _exposition(prompt_total=11000, prompt_cached=8000),
        ],
        clock=[0.0, 2.0],
    )
    # computed delta = (11000-8000) - (1000-0) = 2000 over 2 s
    assert snaps[1].prefill_tok_s == pytest.approx(1000.0)


def test_prefill_is_none_not_zero_when_idle() -> None:
    """An idle server has no prefill speed. "0 tok/s" reads as "the machine
    got slow"; the honest render is "—"."""
    snaps = _scrape_all(
        [
            _exposition(prompt_total=5000, gen_total=900),
            _exposition(prompt_total=5000, gen_total=900),
        ],
        clock=[0.0, 2.0],
    )
    assert snaps[1].prefill_tok_s is None
    assert snaps[1].gen_tok_s is None
    d = snaps[1].to_dict()
    assert d["prefill_tok_s"] is None and d["gen_tok_s"] is None


def test_prefill_avg_survives_an_idle_window() -> None:
    """Prefill is bursty: at --max-num-seqs 1 a 10k prompt prefills for ~10 s
    and then nothing prefills for minutes. A windowed rate is "—" almost
    always, so the lifetime figure (computed tokens per second of prefill
    time) is what the serving line can actually show.

    Not a stale reading: the denominator is prefill seconds, not wall
    seconds, so it does not decay while the server sits idle.
    """
    snaps = _scrape_all(
        [_exposition(prompt_total=12000, prompt_cached=2000, prefill_sum=10.0, prefill_count=3)],
        clock=[0.0],
    )
    assert snaps[0].prefill_tok_s_avg == pytest.approx(1000.0)
    assert snaps[0].prefill_requests == 3


def test_prefill_avg_is_none_before_any_request_finishes() -> None:
    """No finished prefill means no average. Dividing by a zero histogram sum
    must not produce inf, nan, or 0."""
    snaps = _scrape_all(
        [_exposition(prompt_total=0, prefill_sum=0.0, prefill_count=0)],
        clock=[0.0],
    )
    assert snaps[0].prefill_tok_s_avg is None


def test_prefill_avg_absent_family_does_not_crash() -> None:
    """vLLM renames metrics between versions. A missing histogram means
    "unknown", never an exception and never a fabricated number."""
    snaps = _scrape_all([_exposition(prompt_total=500)], clock=[0.0])
    assert snaps[0].prefill_tok_s_avg is None
    assert snaps[0].reachable is True


# --------------------------------------------------------------------------
# Rate-baseline bugs
# --------------------------------------------------------------------------
def test_rate_baseline_is_dropped_when_the_backend_goes_away() -> None:
    """Regression: the baseline survived an unreachable window.

    scrape() returned early on a transport failure without touching the
    stored (ts, total) pair, so the next successful scrape divided a fresh
    counter delta by a dt spanning the whole outage — a backend down for
    10 minutes reported a plausible-looking throughput averaged over its own
    downtime.
    """
    snaps = _scrape_all(
        [
            _exposition(prompt_total=1000, gen_total=1000),
            ConnectionError("backend down"),
            _exposition(prompt_total=1600, gen_total=1600),
        ],
        clock=[0.0, 1.0, 600.0],
    )
    assert snaps[1].reachable is False
    assert snaps[2].gen_tok_s is None, (
        "a rate must never be computed across an interval in which the "
        f"backend was unreachable; got {snaps[2].gen_tok_s}"
    )
    assert snaps[2].prefill_tok_s is None


def test_http_error_also_drops_the_baseline() -> None:
    """A 500 from a half-restarted /metrics is the same situation as a
    refused connection."""
    snaps = _scrape_all(
        [
            _exposition(prompt_total=1000, gen_total=1000),
            (503, "unavailable"),
            _exposition(prompt_total=1600, gen_total=1600),
        ],
        clock=[0.0, 1.0, 300.0],
    )
    assert snaps[1].reachable is False and snaps[1].error == "HTTP 503"
    assert snaps[2].gen_tok_s is None


def test_counter_reset_reports_unknown_not_zero() -> None:
    """vLLM's counters restart at 0 with the process. A backwards counter
    means "new server", and the honest answer for that window is "unknown" —
    reporting 0 tok/s for a server that is generating is a lie."""
    snaps = _scrape_all(
        [
            _exposition(prompt_total=90000, gen_total=50000),
            _exposition(prompt_total=120, gen_total=40),
        ],
        clock=[0.0, 2.0],
    )
    assert snaps[1].gen_tok_s is None
    assert snaps[1].prefill_tok_s is None


def test_rates_use_a_monotonic_clock() -> None:
    """dt must come from a clock that cannot step. time.time() jumps on an
    NTP correction, which silently scales every throughput number."""
    seen: list[float] = []

    def clock() -> float:
        seen.append(len(seen) * 2.0)
        return seen[-1]

    poller = metrics.MetricsPoller("http://127.0.0.1:8002", monotonic=clock)
    client = _FakeClient([_exposition(gen_total=0), _exposition(gen_total=30)])

    async def run() -> list[metrics.MetricsSnapshot]:
        return [await poller.scrape(client), await poller.scrape(client)]  # type: ignore[arg-type]

    snaps = asyncio.run(run())
    assert seen, "the injected clock must be the one the rate is computed from"
    assert snaps[1].gen_tok_s == pytest.approx(15.0)


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
def test_parser_reads_labelled_counters() -> None:
    p = metrics.parse_prometheus(_exposition(prompt_total=7, prompt_cached=2))
    assert p["vllm:prompt_tokens_total"][0][1] == 7.0
    assert p["vllm:prompt_tokens_total"][0][0]["model_name"] == "glm53-flash"
    assert p["vllm:prompt_tokens_cached_total"][0][1] == 2.0


# --------------------------------------------------------------------------
# Prefill vs decode vs TTFT — the ambiguity the dashboard shipped
# --------------------------------------------------------------------------
#
# The panel showed ONE throughput number. "Confusing whether it is prefill or
# decode time when only 1 is visible" — and when a figure was missing it
# rendered a bare em dash, which says neither which figure is missing nor why.
# These tests pin the three separate figures and the requirement that a
# missing one always arrives with a reason attached.

_LIVE_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "metrics_flashnext_live.txt"


def _live_text() -> str:
    """The exposition recorded off the running Flash-Next server on :8001,
    2026-09-09. Recorded rather than invented so the metric NAMES are the ones
    this build actually publishes — the reason the older short name
    `vllm:time_per_output_token_seconds` would have read as absent forever."""
    return _LIVE_FIXTURE.read_text()


def _bump(text: str, name: str, delta: float) -> str:
    """Return `text` with one metric's value increased by `delta`."""
    out = []
    hit = 0
    for line in text.splitlines():
        if line.startswith(name + "{"):
            head, _, value = line.rpartition(" ")
            line = f"{head} {float(value) + delta}"
            hit += 1
        out.append(line)
    assert hit == 1, f"{name} appears {hit} times in the fixture, expected 1"
    return "\n".join(out) + "\n"


def test_ttft_is_read_from_the_metric_name_this_build_publishes() -> None:
    """TTFT was not scraped at all, so the panel could not show it.

    The window figure is the mean of the requests that reached their first
    token IN the window: delta(_sum)/delta(_count). Two requests taking 3 s
    and 5 s must read 4 s, not the server's lifetime average.
    """
    first = _live_text()
    second = _bump(first, metrics.TTFT_SUM, 8.0)
    second = _bump(second, metrics.TTFT_COUNT, 2.0)
    snaps = _scrape_all([first, second], [10.0, 12.0])

    assert snaps[1].ttft_s == pytest.approx(4.0)
    assert snaps[1].ttft_reason is None
    # And the lifetime average, which does not decay while the server idles.
    assert snaps[0].ttft_s_avg == pytest.approx(1871.7741420269012 / 552.0)


def test_prefill_decode_and_ttft_are_three_separate_readings() -> None:
    """One number cannot answer "is it slow to start or slow to run".

    Same window, same fixture pair: prefill tok/s, decode tok/s and TTFT must
    each come out as their own figure.
    """
    first = _live_text()
    second = _bump(first, metrics.PROMPT_TOK_TOTAL, 20_000.0)
    second = _bump(second, metrics.GEN_TOK_TOTAL, 200.0)
    second = _bump(second, metrics.TTFT_SUM, 3.0)
    second = _bump(second, metrics.TTFT_COUNT, 1.0)
    snaps = _scrape_all([first, second], [0.0, 2.0])
    s = snaps[1]

    assert s.prefill_tok_s == pytest.approx(10_000.0)
    assert s.gen_tok_s == pytest.approx(100.0)
    assert s.ttft_s == pytest.approx(3.0)
    d = s.to_dict()
    assert d["prefill_tok_s"] == pytest.approx(10_000.0)
    assert d["gen_tok_s"] == pytest.approx(100.0)
    assert d["ttft_s"] == pytest.approx(3.0)


def test_every_missing_figure_carries_a_reason_never_a_bare_dash() -> None:
    """A missing reading must say WHY. "no traffic yet" and "this build does
    not publish that metric" are different facts and only one of them is fixed
    by sending a request; both used to render as the same em dash."""
    text = _live_text()
    # Two identical scrapes: counters do not move, so the server was idle.
    snaps = _scrape_all([text, text], [0.0, 2.0])

    first, second = snaps
    assert first.gen_tok_s is None and first.gen_reason == metrics.NO_BASELINE
    assert first.prefill_tok_s is None and first.prefill_reason == metrics.NO_BASELINE
    assert first.ttft_s is None and first.ttft_reason == metrics.NO_BASELINE

    assert second.gen_tok_s is None and second.gen_reason == metrics.IDLE
    assert second.prefill_tok_s is None and second.prefill_reason == metrics.IDLE
    assert second.ttft_s is None and second.ttft_reason == metrics.IDLE

    d = second.to_dict()
    assert d["gen_reason"] == metrics.IDLE and d["ttft_reason"] == metrics.IDLE


def test_unreachable_backend_reports_unreachable_not_idle() -> None:
    snaps = _scrape_all([ConnectionError("refused")], [0.0])
    s = snaps[0]
    assert s.reachable is False
    assert s.gen_reason == metrics.UNREACHABLE
    assert s.prefill_reason == metrics.UNREACHABLE
    assert s.ttft_reason == metrics.UNREACHABLE


def test_a_build_without_the_ttft_family_says_so() -> None:
    """Absent family -> "not published by this build", never a fabricated 0."""
    a = _exposition(prompt_total=10.0, gen_total=5.0)
    b = _exposition(prompt_total=20.0, gen_total=10.0)
    snaps = _scrape_all([a, b], [0.0, 1.0])
    assert snaps[1].ttft_s is None
    assert snaps[1].ttft_reason == metrics.NOT_EXPOSED
    assert snaps[1].ttft_s_avg is None


def test_lifetime_decode_rate_survives_an_idle_window() -> None:
    """The window decode rate is unknown whenever nothing is generating, which
    on an agent workload is most of the time. The lifetime rate — generated
    tokens per second OF DECODE TIME — is the figure that still answers "how
    fast does this server decode"."""
    text = _live_text()
    snaps = _scrape_all([text, text], [0.0, 2.0])
    assert snaps[1].gen_tok_s is None                     # idle window
    assert snaps[1].gen_tok_s_avg == pytest.approx(
        1_196_208.0 / 9371.17950598198, rel=1e-9
    )
    assert snaps[1].to_dict()["gen_tok_s_avg"] == pytest.approx(127.6, abs=0.1)


def test_kv_capacity_is_read_from_the_engines_own_cache_config() -> None:
    """The RUNNING engine publishes its resolved KV size on
    vllm:cache_config_info. That is a measurement — the same number the boot
    log prints as "GPU KV cache size: N tokens" — and it must be preferred
    over any estimate. Servedeck used to be able to read it only by scraping a
    boot log, which does not exist for a server started by hand."""
    snaps = _scrape_all([_live_text()], [0.0])
    s = snaps[0]
    assert s.kv_cache_size_tokens == 290_925
    assert s.kv_cache_max_concurrency == pytest.approx(1.1097922848664687)
    assert s.kv_cache_gpu_util == pytest.approx(0.95)
    assert s.to_dict()["kv_cache_size_tokens"] == 290_925


def test_cache_config_none_labels_do_not_become_zero() -> None:
    """vLLM writes the string "None" for unset numeric config
    (kv_cache_memory_bytes="None"). Parsing that as 0 would report a server
    with no KV cache at all."""
    line = 'vllm:cache_config_info{kv_cache_size_tokens="None",gpu_memory_utilization="None"} 1.0'
    snaps = _scrape_all([_exposition() + line + "\n"], [0.0])
    assert snaps[0].kv_cache_size_tokens is None
    assert snaps[0].kv_cache_gpu_util is None


def test_the_prescrape_placeholder_has_the_same_shape_as_a_snapshot() -> None:
    """The dashboard publishes a placeholder before its first scrape. A bare
    {"reachable": False} carries none of the reason fields, so the one state
    that most needs explaining — nothing has been scraped yet — rendered as
    the unexplained blank the reasons exist to replace."""
    placeholder = metrics.unreachable_snapshot()
    assert set(placeholder) == set(metrics.MetricsSnapshot().to_dict())
    assert placeholder["reachable"] is False
    for key in ("gen_reason", "prefill_reason", "ttft_reason"):
        assert placeholder[key] == metrics.UNREACHABLE


# --------------------------------------------------------------------------
# Rates derived from two RECORDED live snapshots with a known interval
# --------------------------------------------------------------------------
#
# Owner, on the panel that showed one throughput number at a time: "prefill
# might be badly wired". These pin what the two denominators are, using
# exposition text scraped off the running Flash-Next server on :8001 rather
# than invented numbers -- the pathology below is not one anybody would have
# thought to write down.

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_WINDOWS = json.loads((_FIXTURES / "metrics_live_windows.json").read_text())


def _pair(key: str) -> tuple[str, str, float]:
    """The recorded (t1, t2, dt) of one live pair."""
    spec = _WINDOWS[key]
    return (
        (_FIXTURES / spec["t1"]).read_text(),
        (_FIXTURES / spec["t2"]).read_text(),
        float(spec["dt_s"]),
    )


def _scrape_pair(key: str) -> metrics.MetricsSnapshot:
    """Scrape both halves of a recorded pair with the recorded dt, and return
    the second snapshot -- the one that has a window to report on."""
    t1, t2, dt = _pair(key)
    return _scrape_all([t1, t2], [0.0, dt])[1]


def test_window_rate_is_a_delta_over_the_recorded_interval() -> None:
    """tokens/second, from two real scrapes 60.007 s apart.

    d(generation_tokens_total) = 14,940 over dt = 60.007 s, so the decode
    window figure is 248.97 tok/s and nothing else. A "total divided by
    uptime" would have produced 6,424,697 / (server uptime), a completely
    different number that happens to look just as plausible.
    """
    t1, t2, dt = _pair("win")
    p1, p2 = metrics.parse_prometheus(t1), metrics.parse_prometheus(t2)
    d_gen = metrics._first(p2, metrics.GEN_TOK_TOTAL) - metrics._first(
        p1, metrics.GEN_TOK_TOTAL
    )
    assert d_gen == 14940.0, "fixture changed; the arithmetic below is pinned to it"

    snap = _scrape_pair("win")
    assert snap.gen_tok_s == pytest.approx(d_gen / dt, rel=1e-9)
    assert snap.gen_tok_s == pytest.approx(248.97, abs=0.01)


def test_the_live_window_that_produced_the_owners_report() -> None:
    """The exact state the owner described: "when one is not displayed, the
    prefill speed shows, when it is not running, while generate shows when
    generating."

    In this recorded 60 s window the engine decoded 14,940 tokens and computed
    ZERO prompt tokens. So decode has a window reading and prefill does not --
    and the panel used to fill the prefill slot with a lifetime average
    instead, in the same font, marked only with a "~". Both facts are asserted
    here because the fix is that the second one no longer follows from the
    first.
    """
    snap = _scrape_pair("win")
    assert snap.gen_tok_s is not None, "decode had traffic in this window"
    assert snap.prefill_tok_s is None, "no prompt tokens were computed"
    assert snap.prefill_reason == metrics.IDLE
    payload = snap.to_dict()
    assert payload["gen_state"] == "ok"
    assert payload["prefill_state"] == "idle"
    # The lifetime companion is still published -- in its own field, which the
    # panel renders on its own line. It must never arrive as prefill_tok_s.
    assert payload["prefill_tok_s"] is None
    assert payload["prefill_tok_s_avg"] == pytest.approx(3013.3, abs=1.0)


def test_lifetime_prefill_divides_by_prefill_time_not_wall_clock() -> None:
    """"Prefill throughput is prompt tokens per second of prefill time."

    Computed prompt tokens 25,244,710 over 8,378.13 s of prefill time =
    3,013.3 tok/s. Over the server's WALL uptime the same numerator gives a
    figure smaller by orders of magnitude, and over the poll window it gives
    0. The denominator is what makes this figure mean anything, so the test
    perturbs the denominator and requires the answer to move with it.
    """
    _t1, t2, _dt = _pair("win")
    p = metrics.parse_prometheus(t2)
    computed = metrics._first(p, metrics.PROMPT_TOK_TOTAL) - metrics._first(
        p, metrics.PROMPT_TOK_CACHED_TOTAL
    )
    prefill_seconds = metrics._first(p, metrics.PREFILL_TIME_SUM)
    assert prefill_seconds > 0

    snap = _scrape_pair("win")
    assert snap.prefill_tok_s_avg == pytest.approx(computed / prefill_seconds, rel=1e-9)

    # Halve the prefill seconds; the rate must double. A wall-clock (or
    # uptime) denominator would not move at all.
    halved = _bump(t2, "vllm:request_prefill_time_seconds_sum", -prefill_seconds / 2)
    snap2 = _scrape_all([t2, halved], [0.0, 60.0])[1]
    assert snap2.prefill_tok_s_avg == pytest.approx(2 * snap.prefill_tok_s_avg, rel=1e-6)


def test_lifetime_decode_divides_by_decode_time_not_wall_clock() -> None:
    """The decode companion, same rule, different denominator -- and the two
    denominators are why the window figure and the lifetime figure of the SAME
    cell disagree by 2.4x on this recording (248.97 vs 104.4 tok/s). They are
    different quantities; the panel must not swap one for the other."""
    _t1, t2, _dt = _pair("win")
    p = metrics.parse_prometheus(t2)
    gen = metrics._first(p, metrics.GEN_TOK_TOTAL)
    decode_seconds = metrics._first(p, metrics.DECODE_TIME_SUM)
    snap = _scrape_pair("win")
    assert snap.gen_tok_s_avg == pytest.approx(gen / decode_seconds, rel=1e-9)
    assert snap.gen_tok_s_avg == pytest.approx(104.4, abs=0.1)
    assert snap.gen_tok_s == pytest.approx(248.97, abs=0.01)


def test_the_window_rate_cannot_use_the_prefill_time_histogram() -> None:
    """Why prefill's WINDOW figure divides by wall clock even though prefill
    throughput is properly per second of prefill time.

    Recorded live, 10.016 s apart: d(prompt_tokens_total) = 0 while
    d(request_prefill_time_seconds_sum) = 20.005 over ONE observed request.
    The counter accrues per engine iteration; the histogram is observed once,
    at first token, carrying the request's whole prefill duration. Their
    deltas over a short window are not a ratio of anything -- dividing them
    gives 0 tok/s for a window in which the engine had been prefilling.

    This test exists so that a future "fix" that switches the window figure to
    that denominator fails here with the reason written down.
    """
    t1, t2, dt = _pair("prefill_lag")
    p1, p2 = metrics.parse_prometheus(t1), metrics.parse_prometheus(t2)
    d_prompt = metrics._first(p2, metrics.PROMPT_TOK_TOTAL) - metrics._first(
        p1, metrics.PROMPT_TOK_TOTAL
    )
    d_prefill_s = metrics._first(p2, metrics.PREFILL_TIME_SUM) - metrics._first(
        p1, metrics.PREFILL_TIME_SUM
    )
    d_prefill_n = metrics._first(p2, metrics.PREFILL_TIME_COUNT) - metrics._first(
        p1, metrics.PREFILL_TIME_COUNT
    )
    assert d_prompt == 0.0
    assert d_prefill_s == pytest.approx(20.005, abs=0.01)
    assert d_prefill_n == 1.0

    snap = _scrape_all([t1, t2], [0.0, dt])[1]
    # Wall clock over this window: zero prompt tokens computed, which is
    # reported as idle -- a true statement -- and NOT as "0 tok/s".
    assert snap.prefill_tok_s is None
    assert snap.prefill_reason == metrics.IDLE
    # And the lifetime figure, whose denominator IS prefill seconds, is still
    # a number, because thousands of requests average the observation lag out.
    assert snap.prefill_tok_s_avg is not None and snap.prefill_tok_s_avg > 1000


def test_no_family_on_this_build_gives_engine_prefill_seconds() -> None:
    """The reason the window figure has no better denominator available.

    Only per-request latency histograms carry prefill time. If a future vLLM
    grows a cumulative engine-side prefill-seconds counter, this test fails and
    the window figure should be rewired to use it.
    """
    _t1, t2, _dt = _pair("win")
    families = {
        line.split()[2]
        for line in t2.splitlines()
        if line.startswith("# TYPE") and len(line.split()) > 2
    }
    prefill_time_families = {
        f
        for f in families
        if "prefill" in f
        and ("time" in f or "seconds" in f)
        # `_created` is prometheus_client's per-family creation TIMESTAMP, not
        # a duration; it is not a candidate denominator.
        and not f.endswith("_created")
    }
    assert prefill_time_families == {"vllm:request_prefill_time_seconds"}, (
        f"this build now publishes {prefill_time_families}"
    )


def test_the_parser_reads_a_whole_unedited_live_exposition() -> None:
    """Reality check on the parser: 678 lines of the real page, histograms,
    `le` labels, `+Inf`, scientific notation and all."""
    _t1, t2, _dt = _pair("win")
    p = metrics.parse_prometheus(t2)
    assert metrics._first(p, metrics.GEN_TOK_TOTAL) == 6461615.0
    assert metrics._first(p, metrics.PROMPT_TOK_TOTAL) == 90749177.0
    # Scientific notation is what the exposition actually uses for these: the
    # value on the wire is the string "9.0749177e+07", not 90749177.
    assert "9.0749177e+07" in t2
    # A bucketed histogram parses into many rows, one per `le`.
    assert len(p["vllm:time_to_first_token_seconds_bucket"]) > 10
    # And the engine's own resolved KV size comes off a LABEL, not a value.
    snap = _scrape_pair("win")
    assert snap.kv_cache_size_tokens == 280813


# --------------------------------------------------------------------------
# The idle/stale path: a figure that has no reading keeps its last one
# --------------------------------------------------------------------------
def test_an_idle_figure_keeps_its_last_reading_and_an_age() -> None:
    """A blank cell is what let one figure look as though it had been replaced
    by the other. An idle window must publish the last value it DID see and
    how old that value is."""
    snaps = _scrape_all(
        [
            _exposition(gen_total=0),
            _exposition(gen_total=100),   # 50 tok/s over 2 s
            _exposition(gen_total=100),   # idle
            _exposition(gen_total=100),   # still idle
        ],
        [0.0, 2.0, 4.0, 10.0],
    )
    assert snaps[1].gen_tok_s == pytest.approx(50.0)
    assert snaps[2].gen_tok_s is None and snaps[2].gen_reason == metrics.IDLE
    assert snaps[2].gen_tok_s_last == pytest.approx(50.0)
    assert snaps[2].gen_last_age_s == pytest.approx(2.0)
    # The age grows; the value does not change.
    assert snaps[3].gen_tok_s_last == pytest.approx(50.0)
    assert snaps[3].gen_last_age_s == pytest.approx(8.0)
    assert snaps[3].to_dict()["gen_last_age_s"] == 8


def test_a_last_reading_survives_the_backend_going_away() -> None:
    """"last 50.0 tok/s, 12 s ago" beats an empty box for a backend that just
    died, so the unreachable path publishes it too."""
    snaps = _scrape_all(
        [_exposition(gen_total=0), _exposition(gen_total=100), ConnectionError("boom")],
        [0.0, 2.0, 20.0],
    )
    assert snaps[2].reachable is False
    assert snaps[2].gen_tok_s_last == pytest.approx(50.0)
    assert snaps[2].gen_last_age_s == pytest.approx(18.0)
    assert snaps[2].to_dict()["gen_state"] == "unreachable"


def test_a_counter_reset_throws_the_last_reading_away() -> None:
    """A reading from the process that just died is not a stale reading of
    THIS server; it is a reading of a different one. Keeping it would put a
    number from the old model beside an age that makes it look current."""
    snaps = _scrape_all(
        [_exposition(gen_total=0), _exposition(gen_total=100), _exposition(gen_total=3)],
        [0.0, 2.0, 4.0],
    )
    assert snaps[1].gen_tok_s_last == pytest.approx(50.0)
    assert snaps[2].gen_reason == metrics.COUNTER_RESET
    assert snaps[2].gen_tok_s_last is None
    assert snaps[2].gen_last_age_s is None
    assert snaps[2].to_dict()["gen_state"] == "reset"


def test_every_figure_publishes_a_state_code_the_ui_can_switch_on() -> None:
    """The page must not match on the English of a reason string: an edit to
    one word here would silently change how the dashboard renders."""
    payload = metrics.unreachable_snapshot()
    for key in ("gen_state", "prefill_state", "ttft_state"):
        assert payload[key] == "unreachable"
    assert set(metrics.REASON_CODE.values()) == {
        "unreachable", "no_baseline", "reset", "idle", "not_exposed",
    }
    for reason, code in metrics.REASON_CODE.items():
        assert metrics._state(None, reason) == code
    assert metrics._state(12.0, metrics.IDLE) == "ok", "a reading always wins"
