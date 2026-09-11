"""Token accounting for the running server: input (prompt) tokens, output
(generated) tokens, and how much of the input the prefix cache served instead
of the engine computing it.

WHERE THE NUMBERS COME FROM
---------------------------
Three vLLM counters, read off /metrics on every poll (metrics.MetricsPoller):

    vllm:prompt_tokens_total          input tokens, cached ones INCLUDED
    vllm:prompt_tokens_cached_total   input tokens served from the prefix cache
    vllm:generation_tokens_total      output tokens

plus prometheus_client's own ``process_start_time_seconds`` from the SAME
exposition, which says when the process that owns those counters started. It
comes off the same scrape as the counters, so the two can never describe
different servers -- which a start time looked up anywhere else could.

Checked against the serving build (vllm-qwen38next, v1/metrics/loggers.py
record(): the two prompt counters are incremented back to back from one
``PromptTokenStats``; v1/metrics/stats.py ``PrefillStats.set`` asserts
cached <= prompt). Both move once per request, by that request's whole prompt,
when its prefill completes -- at its first token. Two consequences:

* Within one process, d(cached) <= d(prompt) over any interval, so the cached
  share is a fraction in [0, 1]. A share outside that range means the two
  counters did not come from one process, and is reported as such, never drawn.
* A prompt still prefilling is in neither counter yet.

WHY THE WINDOW IS ~60 s AND NOT THE 2 s POLL
--------------------------------------------
Because the prompt counter moves by a WHOLE prompt at once, a 2 s window reads
either 0 or a 30,000-token jump: "15,000 tok/s input" for the one poll that
happened to contain a prefill completion. That is a counting artifact, not a
rate, and the cached share of a 2 s window is "no input" nearly every time.
So the window here is the last ``WINDOW_S`` seconds of CONTIGUOUS scrapes of
one process, and it travels with the span it was actually measured over -- the
page prints "last 61 s", never an assumed 60. A rate is always
delta(counter) / that span, never a total divided by an uptime.

WHAT A TOTAL MEANS
------------------
"Since this server started." The counters are born at 0 with the process, so
the raw counter IS the since-start total. Nothing here adds across processes or
subtracts one process's counter from another's: a counter that went DOWN, or a
start time that moved, starts the window again from the new process. A server
that is not answering yields no totals at all, only the reason -- its last
figures describe a process that may no longer exist.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

#: Target span of the rolling window, seconds. The window's baseline is the
#: YOUNGEST scrape at least this old, so at a 2 s poll the span sits in
#: [60, 62) s -- and whatever it is, the payload states it.
WINDOW_S = 60.0

#: A process start time that moved by more than this is a different process.
#: prometheus_client publishes one float for the life of a process; the
#: tolerance only absorbs a formatting round trip, and no vLLM boots in 1 s.
_START_TOLERANCE_S = 1.0

# Reasons. The CODE beside each is metrics.REASON_CODE's vocabulary -- the page
# switches on the code and only displays the prose (tests/test_tokens.py pins
# that every code emitted here is one the page already knows).
UNREACHABLE = "backend not reachable"
NOT_EXPOSED = "not published by this vLLM build"
NO_BASELINE = "no baseline yet — first poll of this server"
RESTARTED = "counters reset — the server restarted; the window starts again here"
START_NOT_EXPOSED = "this vLLM build does not publish process_start_time_seconds"
NO_INPUT_YET = "no input tokens since this server started"
INCONSISTENT = "cached exceeds input — these counters are not from one process"


@dataclass(frozen=True)
class Sample:
    """One scrape's counters.

    ``None`` means the family is absent from the exposition (an older build),
    and is never interchangeable with 0.0: "not published" and "nothing yet"
    are different facts and the page says different things for them.
    """

    ts: float                   # the poll's own monotonic clock read
    prompt: float | None        # vllm:prompt_tokens_total
    cached: float | None        # vllm:prompt_tokens_cached_total
    gen: float | None           # vllm:generation_tokens_total
    started: float | None       # process_start_time_seconds, unix epoch


def is_new_process(prev: Sample | None, cur: Sample) -> bool:
    """Do these counters belong to a different process than ``prev``'s?

    A moved start time says so directly. Without one (a build that does not
    publish it), a counter that went DOWN says so: Prometheus counters only
    ever rise within one process.
    """
    if prev is None:
        return False
    if (
        prev.started is not None
        and cur.started is not None
        and abs(cur.started - prev.started) > _START_TOLERANCE_S
    ):
        return True
    return any(
        a is not None and b is not None and b < a
        for a, b in (
            (prev.prompt, cur.prompt),
            (prev.cached, cur.cached),
            (prev.gen, cur.gen),
        )
    )


def _share(part: float, whole: float, empty_reason: str) -> tuple[float | None, str | None]:
    """``part / whole`` as a fraction, or ``(None, why)``.

    ``whole <= 0`` is "there was no input to take a share of", which is not
    "0% cached" -- the first is unknown, the second a measurement -- so it is
    None with a reason, never 0.0 and never a division by zero.
    """
    if whole <= 0:
        return None, empty_reason
    if part < 0 or part > whole:
        return None, INCONSISTENT
    return round(part / whole, 4), None


def _counter(
    now_v: float | None,
    base_v: float | None,
    span: float,
    state: str,
    reason: str | None,
    noun: str,
) -> dict[str, Any]:
    """One of the input/output figures: its since-start total and its window.

    Exactly one of ``rate`` / ``reason`` is set, so the page can always print
    either a reading or why there is none.
    """
    if now_v is None:
        return {
            "total": None, "total_reason": NOT_EXPOSED, "window": None,
            "rate": None, "state": "not_exposed", "reason": NOT_EXPOSED,
        }
    fig: dict[str, Any] = {
        "total": int(round(now_v)), "total_reason": None, "window": None,
        "rate": None, "state": state, "reason": reason,
    }
    if state != "ok":
        return fig
    if base_v is None:
        # The family appeared mid-process: nothing to difference against.
        fig.update(state="no_baseline", reason=NO_BASELINE)
        return fig
    delta = now_v - base_v
    if delta < 0:
        # Unreachable while is_new_process() guards the window; kept so a
        # negative delta can never reach the page as a figure.
        fig.update(state="reset", reason=RESTARTED)
        return fig
    fig["window"] = int(round(delta))
    if delta > 0:
        fig.update(rate=round(delta / span, 1), state="ok", reason=None)
    else:
        fig.update(state="idle", reason=f"idle — no {noun} tokens")
    return fig


def _cache(
    cur: Sample, base: Sample, span: float, state: str, reason: str | None
) -> dict[str, Any]:
    """The prefix-cache split of the input: cached vs computed, as a share of
    input since the server started and over the window."""
    out: dict[str, Any] = {
        "total": None, "computed": None, "share": None, "share_reason": None,
        "window": None, "window_computed": None, "share_window": None,
        "state": state, "reason": reason,
    }
    p, c = cur.prompt, cur.cached
    if p is None or c is None:
        out.update(share_reason=NOT_EXPOSED, state="not_exposed", reason=NOT_EXPOSED)
        return out
    out["total"] = int(round(c))
    out["computed"] = int(round(max(0.0, p - c)))
    out["share"], out["share_reason"] = _share(c, p, NO_INPUT_YET)
    if state != "ok":
        return out
    if base.prompt is None or base.cached is None:
        out.update(state="no_baseline", reason=NO_BASELINE)
        return out
    dp, dc = p - base.prompt, c - base.cached
    if dp < 0 or dc < 0:
        out.update(state="reset", reason=RESTARTED)   # see _counter()
        return out
    out["window"] = int(round(dc))
    out["window_computed"] = int(round(max(0.0, dp - dc)))
    share, why = _share(dc, dp, "idle — no input tokens")
    if share is not None:
        out.update(share_window=share, state="ok", reason=None)
    elif dp <= 0:
        out.update(state="idle", reason=why)
    else:
        out.update(state="unknown", reason=why)
    return out


def unreachable_payload(window_s: float = WINDOW_S) -> dict[str, Any]:
    """The ``tokens`` block when the backend is not answering.

    Same keys as a live one, every figure None with the reason beside it. NOT
    the last totals seen: those belong to a process that may since have been
    replaced, and a figure that predates the current server must not appear
    as current.
    """
    def fig() -> dict[str, Any]:
        return {
            "total": None, "total_reason": UNREACHABLE, "window": None,
            "rate": None, "state": "unreachable", "reason": UNREACHABLE,
        }

    return {
        "reachable": False,
        "started_ago_s": None,
        "started_reason": UNREACHABLE,
        "window_s": None,
        "window_cap_s": window_s,
        "input": fig(),
        "output": fig(),
        "cached": {
            "total": None, "computed": None, "share": None,
            "share_reason": UNREACHABLE, "window": None,
            "window_computed": None, "share_window": None,
            "state": "unreachable", "reason": UNREACHABLE,
        },
    }


class TokenLedger:
    """Rolling window over the token counters of ONE process.

    Lives on the MetricsPoller, so a repoint (a new poller) starts it empty.
    """

    def __init__(
        self, window_s: float = WINDOW_S, *, wall: Callable[[], float] = time.time
    ) -> None:
        self.window_s = window_s
        #: Wall clock, for the start time's AGE only. process_start_time_seconds
        #: is a unix timestamp, so its age needs the wall clock; every interval
        #: a rate divides by comes from the poll's monotonic clock instead.
        self._wall = wall
        #: Contiguous scrapes of the current process, oldest first.
        self._samples: deque[Sample] = deque()
        #: The last scrape ever seen, kept across failed scrapes. Without it a
        #: restart hidden behind the failures a restart causes would be
        #: invisible: the window is empty by then, and "empty" looks the same
        #: whether the next process is the old one or a new one.
        self._last_seen: Sample | None = None

    def is_new_process(self, cur: Sample) -> bool:
        """Would ``cur`` start a new process's window? (No side effects.)"""
        return is_new_process(self._last_seen, cur)

    def unreachable(self) -> dict[str, Any]:
        """A failed scrape. Drops the window's baseline, keeps ``_last_seen``.

        Same rule as MetricsPoller's rate baseline: a window may not span an
        interval nothing observed, because the server may have been replaced
        inside it.
        """
        self._samples.clear()
        return unreachable_payload(self.window_s)

    def observe(self, cur: Sample) -> dict[str, Any]:
        """Ingest one successful scrape and return the ``tokens`` block."""
        reset = self.is_new_process(cur)
        self._last_seen = cur
        if reset:
            self._samples.clear()
        self._samples.append(cur)
        # The baseline is the youngest sample at least window_s old, so the
        # span reaches the target as soon as the history allows. Drop the
        # oldest only while the next one would still be old enough.
        while len(self._samples) >= 2 and cur.ts - self._samples[1].ts >= self.window_s:
            self._samples.popleft()
        base = self._samples[0]
        span = cur.ts - base.ts

        if reset:
            state, reason = "reset", RESTARTED
        elif base is cur or span <= 0:
            state, reason = "no_baseline", NO_BASELINE
        else:
            state, reason = "ok", None

        started_ago: int | None = None
        started_reason: str | None = START_NOT_EXPOSED
        if cur.started is not None:
            # max(0, ...): a wall clock stepped backwards must not print
            # "started -3 s ago".
            started_ago = int(max(0.0, self._wall() - cur.started))
            started_reason = None

        return {
            "reachable": True,
            "started_ago_s": started_ago,
            "started_reason": started_reason,
            "window_s": round(span, 1) if state == "ok" else None,
            "window_cap_s": self.window_s,
            "input": _counter(cur.prompt, base.prompt, span, state, reason, "input"),
            "output": _counter(cur.gen, base.gen, span, state, reason, "output"),
            "cached": _cache(cur, base, span, state, reason),
        }
