"""Client-side admission for batch jobs against the local model.

THE PROBLEM (2026-10-06)
------------------------
Flash-Next's KV pool holds about 280,000 tokens and every sequence also pins a
fixed per-sequence state page (see :mod:`servedeck.parallelism`), so the engine
can only run a handful of long calls at once. WM-2B pilot 2 sent 64 calls where
about 10 fit: 27 waited, the engine preempted 54 times, and 21 calls hit the
client's 1,800 s timeout. None of those timeouts was the model being slow; the
client's clock started when the call was SENT, and most of it was spent in the
engine's waiting queue.

WHAT THIS MODULE DOES
---------------------
* :func:`safe_concurrency` -- how many calls fit, from prompt PLUS expected
  output tokens: the same calibrated cost and headroom as the dashboard's
  "recommended" (:func:`servedeck.parallelism.recommend`), applied to the
  length a call reaches when it finishes, not the length it starts at. A call
  holds KV for its whole prompt and every token it has generated, so sizing on
  the prompt alone over-admits exactly the long-reasoning jobs that preempt.
* :class:`Limiter` -- the in-flight cap, adapted to what the engine reports:
  it shrinks by a quarter (never below 1) when ``/metrics`` shows a new
  preemption, or requests waiting on two polls in a row, and grows back by one
  after ``calm_polls`` quiet polls, never above its ceiling.
* :func:`run` -- a thread runner: calls ``call(item)`` for every item with at
  most ``limiter.limit`` in flight. ``call`` is only invoked once its slot is
  granted, so the timeout a caller sets INSIDE ``call`` measures the engine's
  work (prefill and generation), not time spent queued behind this job's own
  backlog. Each :class:`Outcome` records when it was admitted and how long the
  call itself took.

Use it like this::

    import sys; sys.path.insert(0, "/home/pctablet505/Projects/servedeck")
    from servedeck import batch
    n = batch.plan(prompt_tokens=7_000, output_tokens=12_000).n   # p90s of your job
    outcomes = batch.run(items, my_call, limit=n)

It needs only the standard library, :mod:`servedeck.parallelism` and
:func:`servedeck.metrics.parse_prometheus` (httpx), so a job's own venv can
import it. Frozen pipelines are not changed; new code adopts it.
"""

from __future__ import annotations

import threading
import time
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from servedeck import parallelism
from servedeck.metrics import parse_prometheus

#: The servedeck gateway. ``/metrics`` there is passed through to the model in
#: the main slot (gateway._EXTRA_PATHS), so no client needs the engine's port.
DEFAULT_METRICS_URL = "http://127.0.0.1:8010/metrics"
#: Flash-Next's ``--max-num-seqs`` (models.toml). The engine never runs more
#: than this many at once, so a client cap above it only builds a queue.
DEFAULT_MAX_NUM_SEQS = 16

T = TypeVar("T")
R = TypeVar("R")


# --------------------------------------------------------------------------
# What the engine reports
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Load:
    """The few ``/metrics`` figures admission needs. None = not reported."""

    running: int | None = None
    waiting: int | None = None
    preemptions: float | None = None
    pool_tokens: int | None = None


def read_load(text: str) -> Load:
    """Parse a ``/metrics`` exposition into a :class:`Load`."""
    p = parse_prometheus(text)

    def gauge(name: str) -> float | None:
        rows = p.get(name)
        return sum(v for _labels, v in rows) if rows else None

    pool = None
    for labels, _v in p.get("vllm:cache_config_info", []):
        try:
            pool = int(labels["kv_cache_size_tokens"])
        except (KeyError, ValueError):
            pool = None
        break
    running, waiting = gauge("vllm:num_requests_running"), gauge("vllm:num_requests_waiting")
    return Load(
        running=None if running is None else int(running),
        waiting=None if waiting is None else int(waiting),
        preemptions=gauge("vllm:num_preemptions_total"),
        pool_tokens=pool,
    )


def fetch_load(url: str = DEFAULT_METRICS_URL, timeout_s: float = 5.0) -> Load | None:
    """GET ``url`` and parse it; None if the engine cannot be read."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return read_load(resp.read().decode("utf-8", "replace"))
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------
# How many fit
# --------------------------------------------------------------------------


def safe_concurrency(
    *,
    pool_tokens: int,
    prompt_tokens: float,
    output_tokens: float,
    max_num_seqs: int | None = DEFAULT_MAX_NUM_SEQS,
) -> int:
    """Calls in flight that fit the KV pool: the dashboard's recommendation,
    computed for a call's FINAL length (prompt + output). Pass p90s, not
    means: the calls that preempt are the long ones."""
    return plan_for(
        pool_tokens=pool_tokens, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        max_num_seqs=max_num_seqs,
    ).n


def plan_for(
    *,
    pool_tokens: int,
    prompt_tokens: float,
    output_tokens: float,
    max_num_seqs: int | None = DEFAULT_MAX_NUM_SEQS,
) -> parallelism.Recommendation:
    """:func:`safe_concurrency` with every step of its arithmetic."""
    if prompt_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts cannot be negative")
    return parallelism.recommend(
        pool_tokens=int(pool_tokens),
        prompt_tokens=float(prompt_tokens) + float(output_tokens),
        max_num_seqs=max_num_seqs,
        basis=f"prompt {prompt_tokens:.0f} + output {output_tokens:.0f} tokens",
    )


def plan(
    *,
    prompt_tokens: float,
    output_tokens: float,
    metrics_url: str = DEFAULT_METRICS_URL,
    max_num_seqs: int | None = DEFAULT_MAX_NUM_SEQS,
) -> parallelism.Recommendation:
    """:func:`plan_for` against the LIVE pool. Raises if the engine is not
    up or has not published its pool: a guessed pool is how a job over-admits."""
    load = fetch_load(metrics_url)
    if load is None or not load.pool_tokens:
        raise RuntimeError(f"no KV pool size at {metrics_url}; is the model up?")
    return plan_for(
        pool_tokens=load.pool_tokens, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        max_num_seqs=max_num_seqs,
    )


# --------------------------------------------------------------------------
# Adapting to what the engine says
# --------------------------------------------------------------------------


class Limiter:
    """The in-flight cap. Shrinks when the engine queues or preempts, grows back
    slowly when it does not.

    * a preemption counter that went UP since the last poll, or requests
      waiting on ``waiting_polls`` polls in a row, cuts the limit by a quarter
      (at least one, never below ``floor``). One waiting reading is not enough:
      a call that has just been sent waits for one engine step before it is
      scheduled, so a single poll can catch it in flight.
    * ``calm_polls`` polls in a row with neither grow it by one, never above
      ``ceiling``.
    * an unreadable poll changes nothing.
    """

    def __init__(self, ceiling: int, *, floor: int = 1, calm_polls: int = 3, waiting_polls: int = 2) -> None:
        if ceiling < 1 or floor < 1 or floor > ceiling:
            raise ValueError(f"need 1 <= floor <= ceiling, got floor={floor} ceiling={ceiling}")
        self.ceiling, self.floor = ceiling, floor
        self.calm_polls, self.waiting_polls = calm_polls, waiting_polls
        self.limit = ceiling
        self._last_preemptions: float | None = None
        self._waiting_run = 0
        self._calm_run = 0
        #: (monotonic time, limit, reason) for every change, for the job's log.
        self.history: list[tuple[float, int, str]] = []

    def observe(self, load: Load | None) -> int:
        if load is None:
            return self.limit
        preempted = (
            load.preemptions is not None
            and self._last_preemptions is not None
            and load.preemptions > self._last_preemptions
        )
        if load.preemptions is not None:
            self._last_preemptions = load.preemptions
        self._waiting_run = self._waiting_run + 1 if (load.waiting or 0) > 0 else 0
        queued = self._waiting_run >= self.waiting_polls
        if preempted or queued:
            self._calm_run = 0
            new = max(self.floor, self.limit - max(1, self.limit // 4))
            if new != self.limit:
                why = "preemption" if preempted else f"{load.waiting} waiting"
                self._set(new, why)
            return self.limit
        self._calm_run += 1
        if self._calm_run >= self.calm_polls and self.limit < self.ceiling:
            self._calm_run = 0
            self._set(self.limit + 1, "calm")
        return self.limit

    def _set(self, value: int, reason: str) -> None:
        self.limit = value
        self.history.append((time.monotonic(), value, reason))


# --------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------


@dataclass
class Outcome(Generic[T, R]):
    """One item's result. ``error`` is set instead of ``result`` when ``call``
    raised; the runner never lets one item's exception stop the batch."""

    index: int
    item: T
    result: R | None = None
    error: BaseException | None = None
    #: Seconds from run() start until this item was handed its slot.
    admitted_after_s: float = 0.0
    #: Seconds ``call(item)`` itself took: the engine's work, not our queue.
    call_s: float = 0.0
    #: The in-flight limit when this item was admitted.
    limit_at_admission: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


def run(
    items: Iterable[T],
    call: Callable[[T], R],
    *,
    limit: int | Limiter,
    metrics_url: str | None = DEFAULT_METRICS_URL,
    poll_s: float = 5.0,
    warm_first: bool = False,
    on_done: Callable[[Outcome[T, R]], None] | None = None,
    fetch: Callable[[str], Load | None] | None = None,
) -> list[Outcome[T, R]]:
    """Call ``call(item)`` for every item, at most ``limit`` at once.

    ``limit`` is a ceiling (an int, wrapped in a default :class:`Limiter`) or
    a :class:`Limiter`. With ``metrics_url`` set, a poller reads the engine
    every ``poll_s`` seconds and the limiter adapts; ``None`` keeps the limit
    fixed. ``warm_first`` runs the first item alone, to completion, before the
    rest start: when the items share a long prefix (one document, several
    questions) the rest then find it in the prefix cache instead of all
    prefilling it side by side. Results come back in input order.
    """
    seq: Sequence[T] = list(items)
    limiter = limit if isinstance(limit, Limiter) else Limiter(int(limit))
    fetch = fetch or (lambda url: fetch_load(url))
    outcomes: list[Outcome[T, R]] = [Outcome(index=i, item=it) for i, it in enumerate(seq)]
    cond = threading.Condition()
    state = {"in_flight": 0, "next": 0, "done": 0}
    stop = threading.Event()
    t0 = time.monotonic()

    def worker() -> None:
        while True:
            with cond:
                while True:
                    if state["next"] >= len(seq):
                        return
                    warming = warm_first and state["done"] == 0 and state["next"] >= 1
                    if not warming and state["in_flight"] < limiter.limit:
                        break
                    cond.wait(timeout=0.5)
                index = state["next"]
                state["next"] += 1
                state["in_flight"] += 1
                out = outcomes[index]
                out.admitted_after_s = time.monotonic() - t0
                out.limit_at_admission = limiter.limit
            start = time.monotonic()
            try:
                out.result = call(out.item)
            except BaseException as exc:  # noqa: BLE001 - recorded per item, the batch goes on
                out.error = exc
            out.call_s = time.monotonic() - start
            if on_done is not None:
                try:
                    on_done(out)
                except Exception:  # noqa: BLE001 - a logging hook must not stop the batch
                    pass
            with cond:
                state["in_flight"] -= 1
                state["done"] += 1
                cond.notify_all()

    def poller() -> None:
        assert metrics_url is not None
        while not stop.wait(poll_s):
            limiter.observe(fetch(metrics_url))
            with cond:
                cond.notify_all()  # a grown limit admits waiting workers now

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(limiter.ceiling, len(seq)))]
    poll_thread = threading.Thread(target=poller, daemon=True) if metrics_url else None
    if poll_thread is not None:
        poll_thread.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    if poll_thread is not None:
        poll_thread.join(timeout=poll_s + 1)
    return outcomes
