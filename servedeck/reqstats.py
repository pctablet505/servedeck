"""A rolling window over the last N *finished* requests, rebuilt from vLLM's
histograms.

WHY THIS EXISTS
---------------
The dashboard used to show one number for request size: ``avg_prompt_tokens``,
computed as ``vllm:request_prompt_tokens_sum / vllm:request_prompt_tokens_count``.
That is the arithmetic mean over **every request since the engine booted** — a
lifetime figure that a single 250k-token request drags upward for the rest of
the process's life, and that a morning of benchmark traffic keeps poisoning all
afternoon. It cannot answer "how many agents can I run *now*", because sizing
is governed by the tail of the distribution (a p90 that does not fit is a
preemption) and a mean has no tail.

WHAT VLLM ACTUALLY EXPOSES
--------------------------
There is no per-request feed. ``/metrics`` publishes ``request_prompt_tokens``
and ``request_generation_tokens`` as **cumulative histograms**: a ``_bucket``
series per ``le`` threshold, plus ``_sum`` and ``_count``. The buckets are
coarse where this workload lives — the live exposition on 2026-09-10 had edges
at 5,000 / 10,000 / 20,000 / 50,000 / 100,000 / 200,000 tokens, and 938 of
1,717 requests fell in the single (20,000, 50,000] bucket.

The alternative source considered and rejected: Servedeck's own ``/v1`` proxy
(``app.catch_all``) could record exact token counts for every request it
forwards. It was rejected because on this box the agents talk to the backend
directly (``:8001``) or through the reasoning proxy (``:8005``); the dashboard's
proxy sees a small and *unrepresentative* slice, and sizing advice computed from
a slice that excludes the big requests is advice that over-subscribes the GPU.
The histogram sees every request the engine finished. Coarse and complete beats
exact and partial, for this question.

HOW THE WINDOW IS RECONSTRUCTED
-------------------------------
Each poll takes the delta of the cumulative bucket vector. The delta of a
cumulative histogram is itself a valid cumulative histogram, so per-bucket
counts are always >= 0, and each one says "this many requests finished in this
poll window, each somewhere in (prev_le, le]". Those become individual
:class:`Observation` rows in a ``deque(maxlen=100)`` — a genuine rolling window
of the last 100 finished requests, each known only to a bucket *interval*.

One refinement makes most of them exact: when a poll window contains exactly
ONE new request, ``delta(_sum)`` **is** that request's prompt-token count. At a
2 s poll interval that is the common case for interactive agent traffic, so the
window is typically a mix of exact points and bucket intervals, and
:class:`WindowStats` reports how many of each.

WHAT THIS CANNOT TELL YOU (stated in the UI, not just here)
-----------------------------------------------------------
* Only requests that finished **while Servedeck was polling** are in the window.
  A server that has served 1,717 requests shows n=0 the moment the dashboard
  restarts. The sample count is the count of what was *observed*, never the
  engine's lifetime ``_count``.
* A percentile over interval-valued observations is itself an interval. Nothing
  here narrows ``(20000, 50000]`` to a point, and nothing here pretends to.
* The histogram observes a request at FINISH, so a long-running request is
  absent from the window for as long as it runs.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

#: How many finished requests the window keeps. The owner asked for "the
#: statistics of last 100 requests"; this is that 100.
WINDOW_SIZE = 100

#: A prompt is at least one token, so the first bucket's lower bound is 1 and
#: not 0. It matters only for tiny-prompt workloads, where a lower bound of 0
#: would let the p50 interval start at a value no request can have.
MIN_PROMPT_TOKENS = 1


@dataclass(frozen=True)
class Observation:
    """One finished request, known either exactly or to a bucket interval.

    ``lo``/``hi`` are inclusive bounds on the token count. ``hi`` is
    ``math.inf`` only for the ``+Inf`` bucket when no ceiling was supplied;
    with a ceiling (``--max-model-len``, a real hard limit the engine enforces)
    it is that ceiling.
    """

    lo: float
    hi: float
    ts: float

    @property
    def exact(self) -> bool:
        return self.lo == self.hi


def nearest_rank(values: Sequence[float], p: float) -> float | None:
    """The nearest-rank percentile: the smallest value at or above rank
    ``ceil(p/100 * n)``.

    Chosen over linear interpolation deliberately. Interpolating between two
    *bucket bounds* would invent a value that no measurement supports — the
    exact failure this module exists to avoid — and nearest-rank always returns
    an observed bound. With n=1 every percentile is that one observation, and
    with all-identical input every percentile is that value; both are properties
    the tests pin.
    """
    n = len(values)
    if n == 0:
        return None
    k = int(math.ceil(p / 100.0 * n))
    k = min(max(k, 1), n)
    return values[k - 1]


@dataclass(frozen=True)
class Percentile:
    """A percentile of interval-valued data: itself an interval.

    ``lo == hi`` exactly when every observation at that rank was exact.
    """

    lo: float
    hi: float

    @property
    def exact(self) -> bool:
        return self.lo == self.hi

    def to_dict(self) -> dict[str, Any]:
        return {
            "lo": None if self.lo == math.inf else round(self.lo),
            "hi": None if self.hi == math.inf else round(self.hi),
            "exact": self.exact,
        }


@dataclass(frozen=True)
class WindowStats:
    """Everything the panel needs to describe the window honestly."""

    n: int
    capacity: int
    exact_n: int
    age_s: float | None
    p50: Percentile | None
    p90: Percentile | None
    p99: Percentile | None
    peak: Percentile | None
    #: True when the window has not yet filled. The UI must say "n of 100",
    #: never pad to 100 and never imply the window is representative.
    partial: bool

    def to_dict(self) -> dict[str, Any]:
        pct = lambda p: None if p is None else p.to_dict()  # noqa: E731
        return {
            "n": self.n,
            "capacity": self.capacity,
            "exact_n": self.exact_n,
            "age_s": None if self.age_s is None else round(self.age_s, 1),
            "p50": pct(self.p50),
            "p90": pct(self.p90),
            "p99": pct(self.p99),
            "max": pct(self.peak),
            "partial": self.partial,
        }


#: The one sentence the panel MUST print beside every percentile it shows.
#: Both halves are load-bearing: a bucket-quantised percentile is not an exact
#: one, and a window that only covers what Servedeck watched is not the
#: server's history. Presenting either as exact is the failure this whole
#: module was written to avoid, so the string lives here rather than in the
#: markup, and tests/test_ui.py asserts the page renders it.
PROVENANCE = (
    "estimate: percentiles are bucket-quantised from vLLM's histogram, "
    "and cover only requests that finished while Servedeck was watching"
)


#: The empty window's payload. Published before the first delta, and after a
#: counter reset. Every key a filled window has is present, so the page never
#: has to distinguish "no stats yet" from "stats field missing".
EMPTY_STATS = WindowStats(
    n=0, capacity=WINDOW_SIZE, exact_n=0, age_s=None,
    p50=None, p90=None, p99=None, peak=None, partial=True,
).to_dict()


class RequestWindow:
    """Rolling window of the last ``maxlen`` finished requests.

    Fed one cumulative bucket vector per poll; keeps the previous vector and
    turns the difference into individual observations.
    """

    def __init__(self, maxlen: int = WINDOW_SIZE) -> None:
        self._obs: deque[Observation] = deque(maxlen=maxlen)
        self._maxlen = maxlen
        #: (cumulative counts, _sum, _count) of the previous poll, or None
        #: before the first one.
        self._prev: tuple[list[float], float, float] | None = None

    def __len__(self) -> int:
        return len(self._obs)

    def clear(self) -> None:
        """Drop every observation AND the baseline.

        Called on a counter reset, which means the engine restarted: the
        requests in the window were served by a different process, possibly a
        different model, and averaging across that boundary is exactly the
        lifetime-mean mistake this module replaces.
        """
        self._obs.clear()
        self._prev = None

    def drop_baseline(self) -> None:
        """Forget the previous bucket vector, keep the observations.

        Called when a scrape FAILS. The requests the engine finished while
        Servedeck could not reach it are real, but nothing observed them, and
        attributing a whole outage's worth of them to the single instant of the
        next successful scrape would make ``age_s`` claim the window covers
        half a second when it covers ten minutes. Losing them is the honest
        outcome: the window is "requests we watched finish", and we did not
        watch these.
        """
        self._prev = None

    def observe(
        self,
        buckets: Iterable[tuple[float, float]],
        *,
        hist_sum: float,
        hist_count: float,
        ceiling: float | None = None,
        ts: float | None = None,
    ) -> int:
        """Ingest one scrape. Returns how many requests were added.

        ``buckets`` is ``(le, cumulative_count)`` pairs in any order; ``+Inf``
        is expected as ``math.inf``.
        """
        now = time.time() if ts is None else ts
        rows = sorted(buckets, key=lambda b: b[0])
        edges = [le for le, _c in rows]
        cum = [c for _le, c in rows]
        if not cum:
            return 0

        prev = self._prev
        self._prev = (list(cum), hist_sum, hist_count)
        if prev is None:
            # First scrape is a baseline only. Counting the whole lifetime
            # histogram here would fill the "last 100 requests" window with
            # requests from before the dashboard was even running.
            return 0

        prev_cum, prev_sum, _prev_count = prev
        if (
            len(prev_cum) != len(cum)
            or any(c < p for c, p in zip(cum, prev_cum))
            or hist_sum < prev_sum
        ):
            # Counters went backwards: new engine process. Everything already
            # in the window belongs to the old one.
            self._obs.clear()
            self._prev = (list(cum), hist_sum, hist_count)
            return 0

        total_new = int(round(cum[-1] - prev_cum[-1]))
        if total_new <= 0:
            return 0
        sum_new = hist_sum - prev_sum

        # Per-bucket counts, as the difference of two cumulative histograms.
        # d[i] >= 0 always: cum[i] - prev_cum[i] is itself nondecreasing in i.
        added: list[Observation] = []
        for i in range(len(cum)):
            below = (cum[i - 1] - prev_cum[i - 1]) if i else 0.0
            d = int(round((cum[i] - prev_cum[i]) - below))
            if d <= 0:
                continue
            lo = float(MIN_PROMPT_TOKENS) if i == 0 else edges[i - 1] + 1.0
            hi = edges[i]
            if hi == math.inf and ceiling is not None:
                hi = float(ceiling)
            # Defensive: a ceiling below the last finite edge would invert the
            # interval. Keep it a valid interval rather than emit lo > hi.
            if hi < lo:
                hi = lo
            for _ in range(d):
                added.append(Observation(lo=lo, hi=hi, ts=now))

        # Exactness for free: one new request means delta(_sum) IS its token
        # count. Only accepted when it lands inside the bucket the histogram
        # put it in — otherwise the two series disagree and the interval is
        # the safer statement.
        if len(added) == 1 and sum_new > 0:
            exact = float(round(sum_new))
            o = added[0]
            if o.lo <= exact <= o.hi:
                added = [Observation(lo=exact, hi=exact, ts=now)]

        self._obs.extend(added)
        return len(added)

    def stats(self) -> WindowStats:
        obs = list(self._obs)
        n = len(obs)
        if n == 0:
            return WindowStats(
                n=0, capacity=self._maxlen, exact_n=0, age_s=None,
                p50=None, p90=None, p99=None, peak=None, partial=True,
            )
        # Sorted separately, which is sound: lo_j <= hi_j for every
        # observation, so the k-th order statistic of the lows is <= the k-th
        # of the highs. The pair is therefore always a valid interval.
        los = sorted(o.lo for o in obs)
        his = sorted(o.hi for o in obs)

        def pct(p: float) -> Percentile:
            return Percentile(lo=nearest_rank(los, p), hi=nearest_rank(his, p))

        oldest = min(o.ts for o in obs)
        return WindowStats(
            n=n,
            capacity=self._maxlen,
            exact_n=sum(1 for o in obs if o.exact),
            age_s=max(0.0, time.time() - oldest),
            p50=pct(50),
            p90=pct(90),
            p99=pct(99),
            peak=Percentile(lo=los[-1], hi=his[-1]),
            partial=n < self._maxlen,
        )
