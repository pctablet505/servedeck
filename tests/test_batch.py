"""batch.py: client-side admission. No server, no GPU: the engine is a canned
/metrics text or a fake ``fetch``, and ``call`` is a function that sleeps.

The incident this pins (2026-10-06, WM-2B pilot 2): 64 calls sent where about
10 fit, 27 waiting, 54 preemptions, 21 client timeouts at 1,800 s that were all
time spent queued. Each test below names the part of that it guards against.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from servedeck import batch, parallelism

FIXTURES = Path(__file__).parent / "fixtures"
POOL = 280_813


def busy_metrics(waiting: int, preemptions: float, running: int = 10) -> str:
    return (
        f'vllm:num_requests_running{{engine="0",model_name="m"}} {running}\n'
        f'vllm:num_requests_waiting{{engine="0",model_name="m"}} {waiting}\n'
        f'vllm:num_requests_waiting_by_reason{{engine="0",model_name="m",reason="capacity"}} {waiting}\n'
        f'vllm:num_preemptions_total{{engine="0",model_name="m"}} {preemptions}\n'
        f'vllm:cache_config_info{{block_size="8",kv_cache_size_tokens="{POOL}",engine="0"}} 1.0\n'
    )


# --------------------------------------------------------------------------
# How many fit
# --------------------------------------------------------------------------


def test_sizing_counts_the_output_a_call_will_generate() -> None:
    """A call holds KV for its prompt AND everything it generates. Sizing on
    the prompt alone is what admitted 64 WM-2B calls (4-7k prompts) whose
    reasoning then grew to 12-17k tokens each."""
    n = batch.safe_concurrency(pool_tokens=POOL, prompt_tokens=7_000, output_tokens=12_000, max_num_seqs=None)
    assert n == parallelism.recommend(pool_tokens=POOL, prompt_tokens=19_000).n
    assert n < parallelism.recommend(pool_tokens=POOL, prompt_tokens=7_000).n


def test_sizing_never_exceeds_the_engines_own_sequence_cap() -> None:
    assert batch.safe_concurrency(pool_tokens=POOL, prompt_tokens=300, output_tokens=100, max_num_seqs=16) == 16
    assert batch.safe_concurrency(pool_tokens=POOL, prompt_tokens=300, output_tokens=100, max_num_seqs=None) > 16


def test_sizing_reproduces_the_measured_ceilings_when_output_is_small() -> None:
    """With no output the arithmetic is exactly the dashboard's: the measured
    table in parallelism.MEASURED_MAX_USEFUL_CONCURRENCY."""
    for prompt, measured in parallelism.MEASURED_MAX_USEFUL_CONCURRENCY:
        assert batch.safe_concurrency(
            pool_tokens=parallelism.CALIBRATION_POOL_TOKENS, prompt_tokens=prompt, output_tokens=0,
            max_num_seqs=16,
        ) == measured


def test_negative_token_counts_are_refused() -> None:
    with pytest.raises(ValueError):
        batch.safe_concurrency(pool_tokens=POOL, prompt_tokens=-1, output_tokens=0)


# --------------------------------------------------------------------------
# What the engine says
# --------------------------------------------------------------------------


def test_read_load_parses_a_live_flashnext_scrape() -> None:
    load = batch.read_load((FIXTURES / "metrics_flashnext_live.txt").read_text())
    assert load.pool_tokens and load.pool_tokens > 100_000
    assert load.waiting == 0 and load.running == 0 and load.preemptions == 37.0


def test_read_load_does_not_count_the_by_reason_series_twice() -> None:
    load = batch.read_load(busy_metrics(waiting=23, preemptions=109))
    assert (load.waiting, load.preemptions, load.pool_tokens, load.running) == (23, 109.0, POOL, 10)


def test_an_unreadable_scrape_is_unknown_not_zero() -> None:
    assert batch.read_load("") == batch.Load()


# --------------------------------------------------------------------------
# Limiter
# --------------------------------------------------------------------------


def test_a_new_preemption_cuts_the_limit() -> None:
    lim = batch.Limiter(8)
    lim.observe(batch.Load(waiting=0, preemptions=54))  # first reading: no delta yet
    assert lim.limit == 8
    lim.observe(batch.Load(waiting=0, preemptions=55))
    assert lim.limit == 6


def test_one_waiting_reading_is_not_a_queue_but_two_are() -> None:
    """A just-sent call waits one engine step; one poll can see it."""
    lim = batch.Limiter(8)
    lim.observe(batch.Load(waiting=1, preemptions=0))
    assert lim.limit == 8
    lim.observe(batch.Load(waiting=3, preemptions=0))
    assert lim.limit == 6


def test_the_limit_never_drops_below_the_floor() -> None:
    lim = batch.Limiter(2, floor=1)
    for p in range(10):
        lim.observe(batch.Load(waiting=5, preemptions=float(p)))
    assert lim.limit == 1


def test_calm_polls_grow_it_back_but_never_past_the_ceiling() -> None:
    lim = batch.Limiter(4, calm_polls=2)
    lim.observe(batch.Load(waiting=0, preemptions=0))
    lim.observe(batch.Load(waiting=0, preemptions=1))
    assert lim.limit == 3
    for _ in range(10):
        lim.observe(batch.Load(waiting=0, preemptions=1))
    assert lim.limit == 4


def test_an_unreadable_engine_changes_nothing() -> None:
    lim = batch.Limiter(5)
    lim.observe(None)
    lim.observe(batch.Load())
    assert lim.limit == 5


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


class Gauge:
    """Counts calls in flight and remembers the maximum."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.now = 0
        self.peak = 0

    def __enter__(self):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)

    def __exit__(self, *exc):
        with self.lock:
            self.now -= 1


def test_never_more_than_the_limit_in_flight_and_results_keep_their_order() -> None:
    g = Gauge()

    def call(x):
        with g:
            time.sleep(0.02)
        return x * 10

    out = batch.run(range(20), call, limit=3, metrics_url=None)
    assert g.peak == 3
    assert [o.result for o in out] == [x * 10 for x in range(20)]


def test_the_call_clock_starts_at_admission_not_at_submission() -> None:
    """The 21 WM-2B timeouts: the client clock ran while calls sat queued.
    Here the second call queues behind the first for ~0.3 s; its own clock
    must not include that."""
    out = batch.run([0.3, 0.3], lambda s: time.sleep(s), limit=1, metrics_url=None)
    assert out[1].admitted_after_s >= 0.25
    assert out[1].call_s < 0.45  # 0.3 s of work, not 0.6 s of wait + work


def test_one_failing_call_does_not_stop_the_batch() -> None:
    def call(x):
        if x == 2:
            raise TimeoutError("engine took too long")
        return x

    out = batch.run(range(5), call, limit=2, metrics_url=None)
    assert [o.ok for o in out] == [True, True, False, True, True]
    assert isinstance(out[2].error, TimeoutError)


def test_warm_first_finishes_the_first_item_before_any_other_starts() -> None:
    events: list[tuple[str, int]] = []
    lock = threading.Lock()

    def call(x):
        with lock:
            events.append(("start", x))
        time.sleep(0.05)
        with lock:
            events.append(("end", x))

    batch.run(range(4), call, limit=4, metrics_url=None, warm_first=True)
    assert events[:2] == [("start", 0), ("end", 0)]


def test_the_runner_shrinks_in_flight_when_the_engine_preempts() -> None:
    """A fake engine that preempts on every poll: the limiter must bring the
    in-flight count down, not just record a lower number."""
    g = Gauge()
    counter = iter(range(1000))
    seen_after_cut: list[int] = []
    lim = batch.Limiter(4, calm_polls=10_000)

    def fetch(_url):
        return batch.Load(waiting=0, preemptions=float(next(counter)))

    def call(_x):
        with g:
            if lim.limit == 1:
                seen_after_cut.append(g.now)
            time.sleep(0.03)

    batch.run(range(40), call, limit=lim, metrics_url="http://unused", poll_s=0.01, fetch=fetch)
    assert lim.limit == 1
    assert seen_after_cut and max(seen_after_cut[-5:]) == 1
