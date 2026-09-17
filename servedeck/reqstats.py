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

    ``bucket`` is the index of the engine's bucket this request was counted in.
    Kept because it cannot be recovered from ``hi`` afterwards: a ceiling that
    happens to equal a finite edge makes an observation's ``hi`` name the wrong
    bin, and the bars must be counted in the bin the engine itself used.
    """

    lo: float
    hi: float
    ts: float
    bucket: int = -1
    #: True for an observation seeded from the engine's lifetime histogram at
    #: the dashboard's first scrape (bucket-bounded, never exact) rather than
    #: watched finishing. Reported as ``seeded_n`` so the page can say so.
    seeded: bool = False

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
class Bucket:
    """One bar of the request-size histogram: ``count`` finished requests.

    The edges are vLLM's own, not a binning invented here. That is the whole
    point of drawing bars instead of printing percentile text: a bar is the
    quantisation, so the coarseness of the source is visible rather than
    explained away in a sentence beside the number.

    ``hi`` is ``None`` only for the ``+Inf`` bucket when the engine has not
    published a context limit, which leaves the last bar with no right edge.
    """

    lo: float
    hi: float
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "lo": round(self.lo),
            "hi": None if self.hi == math.inf else round(self.hi),
            "n": self.count,
        }


#: Round bin widths for the fine histogram, ascending. The step is the smallest
#: of these that keeps the exact observations inside ~40 bars, so a workload
#: spanning 20k..185k gets ~5k bars and a narrow one gets finer ones. A ladder,
#: not a fixed width, because a fixed width is either too coarse to add
#: granularity or too fine to be readable depending on the range.
_FINE_STEPS: tuple[int, ...] = (
    500, 1000, 2000, 2500, 5000, 10000, 20000, 25000, 50000, 100000,
)


@dataclass(frozen=True)
class WindowStats:
    """Everything the panel needs to describe the window honestly."""

    n: int
    capacity: int
    exact_n: int
    age_s: float | None
    buckets: tuple[Bucket, ...]
    fine: dict[str, Any]
    p50: Percentile | None
    p90: Percentile | None
    p99: Percentile | None
    peak: Percentile | None
    #: True when the window has not yet filled. The UI must say "n of 100",
    #: never pad to 100 and never imply the window is representative.
    partial: bool
    seeded_n: int = 0

    def to_dict(self) -> dict[str, Any]:
        pct = lambda p: None if p is None else p.to_dict()  # noqa: E731
        return {
            "n": self.n,
            "capacity": self.capacity,
            "exact_n": self.exact_n,
            "age_s": None if self.age_s is None else round(self.age_s, 1),
            "buckets": [b.to_dict() for b in self.buckets],
            "fine": self.fine,
            "p50": pct(self.p50),
            "p90": pct(self.p90),
            "p99": pct(self.p99),
            "max": pct(self.peak),
            "partial": self.partial,
            "seeded_n": self.seeded_n,
        }


#: The empty window's payload. Published before the first delta, and after a
#: counter reset. Every key a filled window has is present, so the page never
#: has to distinguish "no stats yet" from "stats field missing".
EMPTY_STATS = WindowStats(
    n=0, capacity=WINDOW_SIZE, exact_n=0, age_s=None, buckets=(), fine={"step": 0, "bins": []},
    p50=None, p90=None, p99=None, peak=None, partial=True,
).to_dict()


def _went_backwards(
    cum: Sequence[float], hist_sum: float, previous: tuple[list[float], float, float]
) -> bool:
    """Do these cumulative counters describe a DIFFERENT process than
    `previous` did? A Prometheus counter only ever goes up within one process,
    so any decrease -- or a change in the bucket layout -- means the engine
    restarted.
    """
    prev_cum, prev_sum, _prev_count = previous
    return (
        len(prev_cum) != len(cum)
        or any(c < p for c, p in zip(cum, prev_cum))
        or hist_sum < prev_sum
    )


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
        #: The last cumulative vector we ever saw, kept even across
        #: drop_baseline(). Without it a restart was invisible: the failed
        #: scrapes an engine restart causes drop the baseline, so the new
        #: engine's first scrape took the "first scrape is a baseline only"
        #: path and the backwards-counter check below never ran -- leaving the
        #: dead process's requests in the window for good.
        self._last_seen: tuple[list[float], float, float] | None = None
        #: The engine's own bucket edges from the last scrape, ascending, with
        #: ``+Inf`` last. Kept so the window can be counted back up into the
        #: SAME bins it was built from — see :meth:`_bars`.
        self._edges: list[float] = []
        #: The ceiling passed to the last scrape (--max-model-len), if any. The
        #: +Inf bar is drawn to it rather than to an open end, because the
        #: engine will not admit a request longer than it.
        self._ceiling: float | None = None

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
        self._last_seen = None
        self._edges = []
        self._ceiling = None

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
        self._edges = list(edges)
        self._ceiling = ceiling

        prev = self._prev
        last_seen = self._last_seen
        self._prev = (list(cum), hist_sum, hist_count)
        self._last_seen = self._prev
        if prev is None:
            if last_seen is None and not self._obs:
                # The dashboard's first look at this engine (or at a new
                # process after clear()). v1 took a baseline only, so every
                # dashboard restart emptied the picture; on 2026-09-17 ten
                # restarts left it showing one request. Seed the window
                # with the engine's lifetime histogram instead — bucket-
                # bounded, scaled to the window, marked as seeded — and let
                # real observations replace it from here on.
                return self._seed(edges, cum, ceiling, now)
            # No baseline, but the window has history: a scrape failed and
            # dropped the baseline. Nothing is seeded twice.
            #
            # But "no baseline" is also what a restart looks like from here:
            # the scrapes that fail while the engine is down drop the
            # baseline, so this path is where a NEW process's counters arrive.
            # Compare against the last vector we ever saw, not just the last
            # one we could difference -- otherwise the window goes on
            # reporting requests served by a process that no longer exists.
            if last_seen is not None and _went_backwards(cum, hist_sum, last_seen):
                self._obs.clear()
                return self._seed(edges, cum, ceiling, now)
            return 0

        if _went_backwards(cum, hist_sum, prev):
            # Counters went backwards: new engine process. Everything already
            # in the window belongs to the old one; the new one's (short)
            # lifetime seeds the picture until its requests are watched.
            self._obs.clear()
            return self._seed(edges, cum, ceiling, now)

        prev_cum, prev_sum, _prev_count = prev
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
                added.append(Observation(lo=lo, hi=hi, ts=now, bucket=i))

        # Exactness for free: one new request means delta(_sum) IS its token
        # count. Only accepted when it lands inside the bucket the histogram
        # put it in — otherwise the two series disagree and the interval is
        # the safer statement.
        if len(added) == 1 and sum_new > 0:
            exact = float(round(sum_new))
            o = added[0]
            if o.lo <= exact <= o.hi:
                added = [Observation(lo=exact, hi=exact, ts=now, bucket=o.bucket)]

        self._obs.extend(added)
        return len(added)

    def _bars(self, obs: list[Observation]) -> tuple[Bucket, ...]:
        """Count the window back into the engine's own bins, for the bars.

        Re-binning here (rather than storing per-bucket counts as requests
        arrive) is what keeps the bars honest across the deque's boundary: an
        observation that has aged out of the last-100 window disappears from
        its bar too. Empty bins are omitted — a bar of height zero is not a
        reading, and dropping them leaves the page nothing to misread.

        The bin is the index recorded when the request was observed, so the
        bars are the engine's own counts and not a re-derivation of them.
        """
        edges = self._edges
        if not edges:
            return ()
        counts = [0] * len(edges)
        for o in obs:
            if 0 <= o.bucket < len(edges):
                counts[o.bucket] += 1
        ceiling = self._ceiling
        return tuple(
            Bucket(
                lo=float(MIN_PROMPT_TOKENS) if i == 0 else edges[i - 1] + 1.0,
                # The open bucket is bounded by the engine's context limit when
                # it has published one; the observations inside it already are.
                hi=ceiling if (edges[i] == math.inf and ceiling is not None) else edges[i],
                count=counts[i],
            )
            for i in range(len(edges))
            if counts[i]
        )

    def _fine(self, obs: list[Observation]) -> dict[str, Any]:
        """A finer histogram, for the part of the window that supports it.

        ``_bars`` answers "what did the engine measure", and its answer is
        three fat bars because that is all vLLM's histogram resolves. But most
        observations here are EXACT -- a poll that catches one finished request
        gets its token count exactly from delta(_sum) -- and collapsing 90 exact
        counts into 3 bars throws away precision the window actually has.

        So the exact observations are binned on a round step chosen to keep them
        inside ~40 bars, and the interval observations are returned separately,
        still as intervals, for the page to draw as a translucent underlay. The
        two are never added into one bar: an exact count and a "somewhere in
        (20k,50k]" count are different kinds of knowledge, and stacking them
        would imply the interval requests are as precisely placed as the exact
        ones. The page draws the exact bars solid and the intervals behind them.

        Returns ``{"step", "bins": [{lo,hi,n}], "intervals": [{lo,hi,n}]}``.
        ``step`` is 0 when there are no exact observations to bin.
        """
        exact = sorted(o.lo for o in obs if o.exact)
        intervals: dict[tuple[float, float], int] = {}
        for o in obs:
            if not o.exact:
                key = (o.lo, o.hi)
                intervals[key] = intervals.get(key, 0) + 1

        if not exact:
            return {
                "step": 0,
                "bins": [],
                "intervals": [
                    {"lo": round(lo), "hi": None if hi == math.inf else round(hi), "n": c}
                    for (lo, hi), c in sorted(intervals.items())
                ],
            }

        span = exact[-1] - exact[0]
        step = _FINE_STEPS[-1]
        for cand in _FINE_STEPS:
            if span / cand <= 40 or cand == _FINE_STEPS[-1]:
                step = cand
                break
        base = (exact[0] // step) * step
        nbins = int((exact[-1] - base) // step) + 1
        counts = [0] * nbins
        for v in exact:
            counts[int((v - base) // step)] += 1
        return {
            "step": step,
            "bins": [
                {"lo": base + i * step, "hi": base + (i + 1) * step, "n": c}
                for i, c in enumerate(counts)
                if c
            ],
            "intervals": [
                {"lo": round(lo), "hi": None if hi == math.inf else round(hi), "n": c}
                for (lo, hi), c in sorted(intervals.items())
            ],
        }

    def _seed(
        self, edges: list[float], cum: list[float], ceiling: float | None, now: float
    ) -> int:
        """Fill the window from a cumulative histogram, keeping its shape."""
        counts: list[int] = []
        for i in range(len(cum)):
            below = cum[i - 1] if i else 0.0
            counts.append(max(0, int(round(cum[i] - below))))
        total = sum(counts)
        if total <= 0:
            return 0
        want = min(total, self._maxlen)
        # Largest-remainder scaling: the seeded picture sums to `want` and
        # keeps each bucket's share.
        raw = [c * want / total for c in counts]
        take = [int(x) for x in raw]
        short = want - sum(take)
        for i in sorted(range(len(raw)), key=lambda k: raw[k] - take[k], reverse=True)[:short]:
            take[i] += 1
        for i, d in enumerate(take):
            if d <= 0:
                continue
            lo = float(MIN_PROMPT_TOKENS) if i == 0 else edges[i - 1] + 1.0
            hi = edges[i]
            if hi == math.inf and ceiling is not None:
                hi = float(ceiling)
            if hi < lo:
                hi = lo
            for _ in range(d):
                self._obs.append(Observation(lo=lo, hi=hi, ts=now, bucket=i, seeded=True))
        return want

    def stats(self) -> WindowStats:
        obs = list(self._obs)
        n = len(obs)
        if n == 0:
            return WindowStats(
                n=0, capacity=self._maxlen, exact_n=0, age_s=None, buckets=(),
                fine={"step": 0, "bins": [], "intervals": []},
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
            seeded_n=sum(1 for o in obs if o.seeded),
            age_s=max(0.0, time.time() - oldest),
            buckets=self._bars(obs),
            fine=self._fine(obs),
            p50=pct(50),
            p90=pct(90),
            p99=pct(99),
            peak=Percentile(lo=los[-1], hi=his[-1]),
            partial=n < self._maxlen,
        )
