"""Input / output token accounting (servedeck/tokens.py via MetricsPoller).

Owner: "can we also have input and output tokens in servedeck". The counters
were already scraped; what was missing was the arithmetic that makes them
honest to display -- since WHEN a total is, what interval a rate covers, what
a restart does to both, and a cached share that cannot divide by zero.

Every test drives the real MetricsPoller over exposition text recorded off a
live vLLM server, through a fake client and an injected clock. Nothing here
talks to a real server.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from pathlib import Path
from typing import Any

from servedeck import metrics, tokens

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_WINDOWS = json.loads((_FIXTURES / "metrics_live_windows.json").read_text())

#: Flash-Next on :8001, 2026-09-10, 60.007 s apart (metrics_live_windows.json
#: "win"): 14,940 tokens decoded, 0 prompt tokens computed, 3h+ into a process
#: that started at 1.78901026252e+09.
_WIN_T1 = (_FIXTURES / "metrics_live_win_t1.txt").read_text()
_WIN_T2 = (_FIXTURES / "metrics_live_win_t2.txt").read_text()
_WIN_DT = float(_WINDOWS["win"]["dt_s"])
_FLASHNEXT_START = 1.78901026252e09

#: Qwen3.8-27B-NVFP4 on :8004, recorded 2026-09-11 by a plain GET: a DIFFERENT
#: process (started 1.78909218635e+09) that had served five requests, none of
#: them from the prefix cache -- vllm:prompt_tokens_cached_total is a real 0.0.
_QWEN27B = (_FIXTURES / "metrics_27b_live_cached_zero.txt").read_text()
_QWEN27B_START = 1.78909218635e09

#: The same server RESTARTING, caught live (metrics_27b_live_restart.json): the
#: process above went away, 85 scrapes failed, and a new process came up with
#: every token counter at 0.0.
_RESTART = json.loads((_FIXTURES / "metrics_27b_live_restart.json").read_text())
_QWEN27B_RESTARTED = (_FIXTURES / _RESTART["after"]["file"]).read_text()
_QWEN27B_RESTARTED_START = 1.78909259294e09


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------
class _Resp:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


class _Client:
    """One scripted response per get(): exposition text, (status, body), or an
    exception to raise."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)

    async def get(self, url: str, timeout: float | None = None) -> _Resp:
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):
            return _Resp(*item)
        return _Resp(200, item)


def _scrape(script: list[Any], clock: list[float], wall: float = 0.0) -> list[dict]:
    """The `tokens` block of every snapshot, in order, as the page receives it
    (to_dict(), i.e. after serialisation)."""
    ticks = iter(clock)
    poller = metrics.MetricsPoller(
        "http://127.0.0.1:8001", monotonic=lambda: next(ticks), wall=lambda: wall
    )
    client = _Client(script)

    async def run() -> list[dict]:
        return [
            (await poller.scrape(client)).to_dict()["tokens"]  # type: ignore[arg-type]
            for _ in range(len(script))
        ]

    return asyncio.run(run())


def _set(text: str, name: str, value: float) -> str:
    """`text` with every series of `name` set to `value` (one series here)."""
    out, hit = [], 0
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            head, _, _old = line.rpartition(" ")
            line = f"{head} {value}"
            hit += 1
        out.append(line)
    assert hit == 1, f"{name} appears {hit} times, expected 1"
    return "\n".join(out) + "\n"


def _bump(text: str, name: str, delta: float) -> str:
    return _set(text, name, _value(text, name) + delta)


def _drop(text: str, name: str) -> str:
    """`text` without the family `name` (an older build that never had it)."""
    return "\n".join(
        ln for ln in text.splitlines()
        if not (ln.startswith(name + "{") or ln.startswith(name + " "))
    ) + "\n"


def _value(text: str, name: str) -> float:
    m = re.search(rf"^{re.escape(name)}(?:\{{[^}}]*\}})? (\S+)$", text, re.M)
    assert m, f"{name} is not in the exposition"
    return float(m.group(1))


def _no_nan(payload: dict) -> None:
    """The payload must be strict JSON: a NaN or an Infinity anywhere in it
    reaches the page as a number and prints as "NaN%"."""
    json.dumps(payload, allow_nan=False)


# --------------------------------------------------------------------------
# The rate is a delta over the measured interval
# --------------------------------------------------------------------------
def test_the_window_rate_is_a_delta_over_the_recorded_interval() -> None:
    """Two real scrapes of one live process, 60.007 s apart.

    d(generation_tokens_total) = 6,461,615 - 6,446,675 = 14,940 tokens, so the
    output window reads 14,940 tokens and 14,940 / 60.007 = 249.0 tok/s. The
    two quantities it must NOT be, both of which look just as plausible:
    the total over the process's uptime (6,461,615 / 11,520 s = 560.9 tok/s)
    and the delta over the nominal 60 s target rather than the measured span.
    """
    d_gen = _value(_WIN_T2, metrics.GEN_TOK_TOTAL) - _value(_WIN_T1, metrics.GEN_TOK_TOTAL)
    assert d_gen == 14_940.0, "fixture changed; the arithmetic below is pinned to it"

    first, second = _scrape([_WIN_T1, _WIN_T2], [0.0, _WIN_DT], wall=_FLASHNEXT_START + 11_520)
    out = second["output"]
    assert out["window"] == 14_940
    assert out["rate"] == round(14_940 / _WIN_DT, 1) == 249.0
    assert out["state"] == "ok" and out["reason"] is None
    assert second["window_s"] == round(_WIN_DT, 1)
    uptime_rate = round(_value(_WIN_T2, metrics.GEN_TOK_TOTAL) / 11_520, 1)
    assert out["rate"] != uptime_rate, "a total divided by an uptime is not a window rate"

    # The same recording, input side: the engine prefilled nothing in those
    # 60 s. That is a measured zero -- the window holds 0 tokens -- and it is
    # stated as idle, never as "0 tok/s".
    inp = second["input"]
    assert inp["window"] == 0 and inp["rate"] is None
    assert inp["state"] == "idle" and "no input tokens" in inp["reason"]
    # The first scrape has nothing to difference against.
    assert first["output"]["state"] == "no_baseline" and first["output"]["rate"] is None


def test_the_rate_divides_by_the_span_it_actually_measured() -> None:
    """The same 14,940-token delta scraped 30 s apart is 498.0 tok/s, and the
    payload says the window is 30 s. A rate hardwired to the 60 s target would
    halve it and still look plausible."""
    _first, second = _scrape([_WIN_T1, _WIN_T2], [0.0, 30.0])
    assert second["window_s"] == 30.0
    assert second["output"]["rate"] == 498.0


def test_the_second_recorded_pair_gives_its_own_rate() -> None:
    """The other recorded pair (10.016 s): 438 tokens decoded. A second real
    interval, so a formula that happened to fit one pair cannot pass."""
    spec = _WINDOWS["prefill_lag"]
    t1 = (_FIXTURES / spec["t1"]).read_text()
    t2 = (_FIXTURES / spec["t2"]).read_text()
    d_gen = _value(t2, metrics.GEN_TOK_TOTAL) - _value(t1, metrics.GEN_TOK_TOTAL)
    assert d_gen == 438.0
    _first, second = _scrape([t1, t2], [0.0, float(spec["dt_s"])])
    assert second["output"]["rate"] == round(438 / 10.016, 1) == 43.7


def test_input_and_cache_windows_from_one_recorded_interval() -> None:
    """Input traffic over a real interval: the recorded 60.007 s pair with one
    agent turn added to its second scrape -- 30,000 prompt tokens of which
    26,624 came from the prefix cache (the increments vLLM makes together, once,
    when that prefill completes). Input 30,000 / 60.007 = 499.9 tok/s; cached
    share of the window 26,624 / 30,000 = 88.75%; computed 3,376."""
    t2 = _bump(_WIN_T2, metrics.PROMPT_TOK_TOTAL, 30_000)
    t2 = _bump(t2, metrics.PROMPT_TOK_CACHED_TOTAL, 26_624)
    _first, second = _scrape([_WIN_T1, t2], [0.0, _WIN_DT])
    inp, cache = second["input"], second["cached"]
    assert inp["window"] == 30_000 and inp["rate"] == 499.9 and inp["state"] == "ok"
    assert cache["state"] == "ok"
    assert cache["share_window"] == 0.8875
    assert (cache["window"], cache["window_computed"]) == (26_624, 3_376)


def test_the_baseline_is_the_youngest_scrape_at_least_a_window_old() -> None:
    """Scraped every 2 s for 100 s, 10 output tokens per scrape. The window must
    span the last 60 s -- not 2 s, not the whole 100 s -- and its rate must be
    the delta over exactly the span it reports."""
    base = _WIN_T1
    script, clock = [], []
    for i in range(51):                 # t = 0, 2, ..., 100
        script.append(_bump(base, metrics.GEN_TOK_TOTAL, 10.0 * i))
        clock.append(2.0 * i)
    last = _scrape(script, clock)[-1]
    assert last["window_s"] == 60.0
    assert last["output"]["window"] == 300       # 30 scrapes x 10 tokens
    assert last["output"]["rate"] == 5.0


# --------------------------------------------------------------------------
# A restart, or a switch of model, starts again -- never a negative delta
# --------------------------------------------------------------------------
def test_a_counter_that_goes_down_is_a_restart_not_a_negative_delta() -> None:
    """Two real processes, back to back: Flash-Next at 90.7M input tokens, then
    the 27B at 39,529. Every counter went DOWN. The totals must be the new
    server's own (never the old one's, never a sum), the window must start
    again, and nothing may carry a negative delta."""
    after = _bump(_QWEN27B, metrics.GEN_TOK_TOTAL, 500)
    a, b, c = _scrape(
        [_WIN_T2, _QWEN27B, after], [0.0, 2.0, 4.0], wall=_QWEN27B_START + 300
    )
    assert a["input"]["total"] == 90_749_177

    assert b["input"]["total"] == 39_529
    assert b["output"]["total"] == 4_111
    assert b["cached"]["total"] == 0
    assert b["started_ago_s"] == 300, "'since' must be the NEW process's start"
    for key in ("input", "output", "cached"):
        assert b[key]["state"] == "reset", (key, b[key])
        assert b[key]["reason"] == tokens.RESTARTED
    assert b["input"]["rate"] is None and b["output"]["rate"] is None
    assert b["window_s"] is None
    assert not re.search(r'": -\d', json.dumps(b)), f"a negative figure escaped: {b}"
    # The window after the restart is measured from the NEW process: 500
    # tokens over 2 s. Still differencing against the dead process's
    # counters would read "reset" (or negative) for the next 60 s.
    assert c["output"]["state"] == "ok", c["output"]
    assert (c["output"]["window"], c["output"]["rate"]) == (500, 250.0)
    assert c["window_s"] == 2.0


def test_a_restart_is_detected_from_the_counters_alone() -> None:
    """A build that publishes no process_start_time_seconds: the decrease is
    the only evidence, and it is enough."""
    old = _drop(_WIN_T2, metrics.PROCESS_START)
    new = _drop(_QWEN27B, metrics.PROCESS_START)
    newer = _bump(new, metrics.PROMPT_TOK_TOTAL, 8_000)
    _a, b, c = _scrape([old, new, newer], [0.0, 2.0, 4.0])
    assert b["input"]["state"] == "reset"
    assert b["started_ago_s"] is None
    assert b["started_reason"] == tokens.START_NOT_EXPOSED
    assert (c["input"]["state"], c["input"]["window"]) == ("ok", 8_000), c["input"]


def test_a_moved_start_time_is_a_restart_even_when_counters_rose() -> None:
    """A new process can have served MORE than the old one by the time it is
    scraped (a long outage). Counters alone would difference the two; the start
    time says they are different processes."""
    older = _set(_QWEN27B, metrics.PROCESS_START, _QWEN27B_START - 5_000)
    older = _set(older, metrics.PROMPT_TOK_TOTAL, 10_000)
    older = _set(older, metrics.GEN_TOK_TOTAL, 1_000)
    _a, b = _scrape([older, _QWEN27B], [0.0, 2.0])
    assert b["input"]["state"] == "reset", b["input"]
    assert b["input"]["window"] is None, "a delta was taken across two processes"


def test_a_restart_hidden_behind_failed_scrapes_is_still_a_restart() -> None:
    """A restart takes the server down for minutes, so the scrapes around it
    FAIL, and a failure drops the window. The next process then arrives with no
    window to compare against -- and without a memory of the last counters
    seen, "no baseline yet" would hide that the server was replaced."""
    got = _scrape([_WIN_T2, ConnectionError("down"), (503, ""), _QWEN27B], [0.0, 2.0, 4.0, 300.0])
    assert got[1]["reachable"] is False and got[2]["reachable"] is False
    assert got[3]["input"]["state"] == "reset", got[3]["input"]
    assert got[3]["input"]["total"] == 39_529


def test_the_restart_recorded_live_resets_and_then_measures_the_new_process() -> None:
    """The real sequence, at the recorded times: the old process's last scrape,
    85 failed scrapes, the new process's first two. Its counters are all 0.0 --
    lower than the old ones -- and its start time moved, so it is a restart
    twice over; the next scrape measures a window of the NEW process only."""
    before, after = _RESTART["before"], _RESTART["after"]
    n_fail = _RESTART["failed_scrapes_between"]
    script = [_QWEN27B, *[ConnectionError("down")] * n_fail, _QWEN27B_RESTARTED, _QWEN27B_RESTARTED]
    step = (after["mono_s"] - before["mono_s"]) / (n_fail + 1)
    clock = [before["mono_s"] + step * i for i in range(n_fail + 1)]
    clock += [after["mono_s"], after["next_mono_s"]]
    got = _scrape(script, clock, wall=after["wall_s"])

    first, second = got[-2], got[-1]
    assert got[-3]["reachable"] is False
    for key in ("input", "output"):
        assert first[key]["total"] == 0, "the new process's own total, not the old one's"
        assert first[key]["state"] == "reset", first[key]
    assert first["started_ago_s"] == int(after["wall_s"] - _QWEN27B_RESTARTED_START) == 169
    assert first["cached"]["share"] is None
    assert first["cached"]["share_reason"] == tokens.NO_INPUT_YET

    assert second["window_s"] == 2.0
    for key in ("input", "output"):
        assert second[key]["state"] == "idle" and second[key]["window"] == 0, second[key]


def test_the_same_process_after_an_outage_is_not_called_a_restart() -> None:
    """Over-correction guard: a /metrics timeout on a server that never went
    away. Same start time, counters higher -- the window starts again (nothing
    watched the gap) but it is NOT a restart, and consecutive healthy scrapes
    are never one either."""
    later = _bump(_WIN_T2, metrics.GEN_TOK_TOTAL, 500)
    got = _scrape([_WIN_T1, _WIN_T2, ConnectionError("timeout"), later], [0.0, 60.0, 62.0, 64.0])
    assert got[1]["output"]["state"] == "ok"
    assert got[3]["output"]["state"] == "no_baseline", got[3]["output"]
    assert got[3]["output"]["total"] == _value(later, metrics.GEN_TOK_TOTAL)


def test_a_backend_that_is_down_shows_no_totals_not_its_last_ones() -> None:
    """The last totals seen describe a process that may since have been
    replaced. Down means no figure and the reason, in every slot."""
    got = _scrape([_WIN_T2, ConnectionError("refused")], [0.0, 2.0])
    down = got[1]
    assert down["reachable"] is False
    for key in ("input", "output"):
        assert down[key]["total"] is None
        assert down[key]["total_reason"] == tokens.UNREACHABLE
        assert down[key]["state"] == "unreachable"
    assert down["cached"]["share"] is None and down["cached"]["total"] is None
    assert down["started_ago_s"] is None and down["started_reason"] == tokens.UNREACHABLE


def test_the_per_request_figures_do_not_outlive_the_process_that_served_them() -> None:
    """The token strip prints per-request p50/p90/max beside the totals, from
    the request windows. Those windows were emptied only when their OWN bucket
    counters went backwards -- and a restart hides behind failed scrapes, which
    drop the windows' baseline, so the new process's first scrape was taken as
    a fresh baseline and the old process's requests stayed in the window.

    Recorded: Flash-Next finishes one 69,134-token request, the backend goes
    away, the 27B comes up. The strip then read "39,529 tokens" beside
    "per request p50 69,134" -- a request the 27B never served.
    """
    lag = _WINDOWS["prefill_lag"]
    t1 = (_FIXTURES / lag["t1"]).read_text()
    t2 = (_FIXTURES / lag["t2"]).read_text()
    ticks = iter([0.0, float(lag["dt_s"]), 12.0, 300.0])
    poller = metrics.MetricsPoller("http://127.0.0.1:8001", monotonic=lambda: next(ticks))
    client = _Client([t1, t2, ConnectionError("restarting"), _QWEN27B])

    async def run() -> list[dict]:
        return [(await poller.scrape(client)).to_dict() for _ in range(4)]  # type: ignore[arg-type]

    snaps = asyncio.run(run())
    assert snaps[1]["prompt_stats"]["n"] == 1, "fixture: one request finished"
    assert snaps[1]["gen_stats"]["n"] == 1
    new = snaps[3]
    assert new["tokens"]["input"]["state"] == "reset"
    assert new["prompt_stats"]["n"] == 0, (
        f"the new process shows the old one's requests: {new['prompt_stats']['p90']}"
    )
    assert new["gen_stats"]["n"] == 0


def test_the_request_windows_survive_an_outage_of_the_same_process() -> None:
    """Over-correction guard: a /metrics timeout on a server that never went
    away must not throw its request history away."""
    lag = _WINDOWS["prefill_lag"]
    t1 = (_FIXTURES / lag["t1"]).read_text()
    t2 = (_FIXTURES / lag["t2"]).read_text()
    t3 = _bump(t2, metrics.GEN_TOK_TOTAL, 100)
    ticks = iter([0.0, 10.0, 12.0, 14.0])
    poller = metrics.MetricsPoller("http://127.0.0.1:8001", monotonic=lambda: next(ticks))
    client = _Client([t1, t2, ConnectionError("timeout"), t3])

    async def run() -> list[dict]:
        return [(await poller.scrape(client)).to_dict() for _ in range(4)]  # type: ignore[arg-type]

    snaps = asyncio.run(run())
    assert snaps[3]["prompt_stats"]["n"] == 1
    assert snaps[3]["tokens"]["input"]["state"] == "no_baseline"


# --------------------------------------------------------------------------
# The cached share: 0 cached, 0 input, absent family -- no NaN, no Infinity
# --------------------------------------------------------------------------
def test_a_cached_share_of_zero_is_a_measurement() -> None:
    """The live 27B served 39,529 input tokens and none from the cache. That is
    0.0%, a real reading -- not None, not "unknown"."""
    (got,) = _scrape([_QWEN27B], [0.0])
    c = got["cached"]
    assert c["share"] == 0.0 and c["share_reason"] is None
    assert (c["total"], c["computed"]) == (0, 39_529)
    _no_nan(got)


def test_no_input_at_all_is_not_a_share() -> None:
    """A server that has served nothing: 0 / 0. That is "no input yet", which
    must not become 0% (a claim about the cache) and must not divide by zero."""
    fresh = _set(_QWEN27B, metrics.PROMPT_TOK_TOTAL, 0)
    fresh = _set(fresh, metrics.GEN_TOK_TOTAL, 0)
    a, b = _scrape([fresh, fresh], [0.0, 2.0])
    for snap in (a, b):
        c = snap["cached"]
        assert c["share"] is None and c["share_reason"] == tokens.NO_INPUT_YET
        assert snap["input"]["total"] == 0
        _no_nan(snap)
    # ...and over the window: no input in it, so no share of it either.
    assert b["cached"]["share_window"] is None and b["cached"]["state"] == "idle"


def test_a_window_with_input_and_no_cache_hits_is_zero_percent() -> None:
    """Input arrived, none of it cached: the window share is 0.0 exactly."""
    t2 = _bump(_QWEN27B, metrics.PROMPT_TOK_TOTAL, 12_000)
    _a, b = _scrape([_QWEN27B, t2], [0.0, 10.0])
    assert b["cached"]["share_window"] == 0.0 and b["cached"]["state"] == "ok"
    assert b["cached"]["window_computed"] == 12_000


def test_the_share_helper_never_divides_by_zero_or_exceeds_one() -> None:
    """Direct oracles on the arithmetic every share goes through."""
    assert tokens._share(0.0, 0.0, "why") == (None, "why")
    assert tokens._share(5.0, 0.0, "why") == (None, "why")
    assert tokens._share(0.0, 10.0, "why") == (0.0, None)
    assert tokens._share(10.0, 10.0, "why") == (1.0, None)
    assert tokens._share(1.0, 3.0, "why") == (0.3333, None)
    # Cached above input cannot come from one process; it is not a figure.
    assert tokens._share(11.0, 10.0, "why") == (None, tokens.INCONSISTENT)


def test_a_build_without_the_cached_family_says_so() -> None:
    """An older vLLM without prompt_tokens_cached_total: the input total is
    still real, the cache split is "not published" -- not 0% and not 100%."""
    old = _drop(_WIN_T2, metrics.PROMPT_TOK_CACHED_TOTAL)
    (got,) = _scrape([old], [0.0])
    assert got["input"]["total"] == 90_749_177
    c = got["cached"]
    assert c["state"] == "not_exposed" and c["share"] is None
    assert c["share_reason"] == tokens.NOT_EXPOSED and c["total"] is None
    _no_nan(got)


def test_a_build_without_a_counter_says_not_published_not_zero() -> None:
    """metrics._first() turns an absent family into 0.0. The token totals must
    not: "not published by this build" and "no tokens yet" are different."""
    (got,) = _scrape([_drop(_WIN_T2, metrics.GEN_TOK_TOTAL)], [0.0])
    assert got["output"]["total"] is None
    assert got["output"]["state"] == "not_exposed"
    assert got["output"]["total_reason"] == tokens.NOT_EXPOSED


# --------------------------------------------------------------------------
# Shape and vocabulary
# --------------------------------------------------------------------------
def test_the_totals_are_since_the_process_that_published_them_started() -> None:
    """The start time comes off the SAME scrape as the counters, so the two can
    never describe different servers. 11,520 s after 1.78901026252e+09."""
    (got,) = _scrape([_WIN_T2], [0.0], wall=_FLASHNEXT_START + 11_520.4)
    assert got["started_ago_s"] == 11_520 and got["started_reason"] is None


def test_a_series_per_engine_is_summed() -> None:
    """With data parallelism vLLM publishes one series per engine; the server's
    total is their sum, not the first engine's."""
    line = 'vllm:generation_tokens_total{engine="1",model_name="m"} 250.0\n'
    text = _set(_WIN_T2, metrics.GEN_TOK_TOTAL, 1_000) + line
    (got,) = _scrape([text], [0.0])
    assert got["output"]["total"] == 1_250


def test_a_start_time_is_read_not_summed() -> None:
    """The counters are summed across series; the start time must not be --
    two start times added together are a date a billion seconds in the
    future, and the age would clamp to "started 0 s ago"."""
    text = _WIN_T2 + 'process_start_time_seconds{worker="1"} 1.0e9\n'
    (got,) = _scrape([text], [0.0], wall=_FLASHNEXT_START + 11_520)
    assert got["started_ago_s"] == 11_520, got["started_ago_s"]


def _all_codes(payload: dict) -> set[str]:
    return {payload[k]["state"] for k in ("input", "output", "cached")}


def test_every_state_code_is_one_the_page_already_switches_on() -> None:
    """The page switches on codes, never on prose. Every code the ledger can
    emit is from metrics.REASON_CODE (plus "ok" for a reading, and "unknown",
    which metrics._state already falls back to)."""
    known = set(metrics.REASON_CODE.values()) | {"ok", "unknown"}
    seen: set[str] = set()
    runs = [
        ([_WIN_T1, _WIN_T2], [0.0, _WIN_DT]),                     # ok / idle
        ([_WIN_T2, _QWEN27B], [0.0, 2.0]),                        # reset
        ([_WIN_T2], [0.0]),                                       # no_baseline
        ([ConnectionError("x")], [0.0]),                          # unreachable
        ([_drop(_WIN_T2, metrics.PROMPT_TOK_CACHED_TOTAL)], [0.0]),  # not_exposed
    ]
    for script, clock in runs:
        for snap in _scrape(script, clock):
            seen |= _all_codes(snap)
    assert seen <= known, f"codes the page does not know: {seen - known}"
    assert {"ok", "idle", "reset", "no_baseline", "unreachable", "not_exposed"} <= seen


def _shape(d: Any) -> Any:
    return {k: _shape(v) for k, v in d.items()} if isinstance(d, dict) else None


def test_every_state_has_the_same_shape() -> None:
    """The page reads the same keys in every state; a key that exists only when
    the server is up renders "undefined" the moment it goes down."""
    live = _scrape([_WIN_T1, _WIN_T2], [0.0, _WIN_DT])[1]
    down = _scrape([ConnectionError("x")], [0.0])[0]
    assert _shape(down) == _shape(live)
    assert _shape(metrics.unreachable_snapshot()["tokens"]) == _shape(live)
    assert set(metrics.unreachable_snapshot()) == set(metrics.MetricsSnapshot().to_dict())


def test_a_non_advancing_clock_is_no_baseline_not_a_division_by_zero() -> None:
    """dt = 0 would divide by zero. It means "no interval", so no rate."""
    later = _bump(_WIN_T2, metrics.GEN_TOK_TOTAL, 100)
    _a, b = _scrape([_WIN_T2, later], [5.0, 5.0])
    assert b["output"]["state"] == "no_baseline" and b["output"]["rate"] is None
    _no_nan(b)
    assert not math.isnan(b["output"]["total"])
