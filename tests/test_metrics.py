"""MetricsPoller regression tests — SPEC.md §8 telemetry.

There were no tests for metrics.py at all, which is why every bug below
survived. Each test names the defect it pins.

The poller is exercised through a fake httpx client rather than a live
server: the real /metrics endpoint belongs to a vLLM process that must not
be started, stopped or leaned on by a unit test.
"""

from __future__ import annotations

import asyncio
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
