"""Prometheus text-format scraping for the live vLLM server.

Deliberately a small hand-rolled parser rather than prometheus_client: we need
a handful of families out of a ~200-line exposition page, and pulling in a
metrics *library* to read someone else's metrics is not worth the dependency.

Every metric name here was checked against a real source, never guessed --
vLLM renames these between versions. Most were verified live against
http://localhost:8001/metrics on 2026-08-27; the three prefill names added on
2026-09-02 (PROMPT_TOK_CACHED_TOTAL, PREFILL_TIME_SUM/_COUNT) were read out of
the serving build's own v1/metrics/loggers.py because the server was mid-
benchmark and could not be scraped. A family that is absent degrades to
"unknown" (None), never to a fabricated 0.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

# --- metric names, verified live -----------------------------------------
KV_USAGE = "vllm:kv_cache_usage_perc"
RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
WAITING_BY_REASON = "vllm:num_requests_waiting_by_reason"
PREEMPTIONS = "vllm:num_preemptions_total"
PROMPT_TOK_SUM = "vllm:request_prompt_tokens_sum"
PROMPT_TOK_COUNT = "vllm:request_prompt_tokens_count"
GEN_TOK_TOTAL = "vllm:generation_tokens_total"
PROMPT_TOK_TOTAL = "vllm:prompt_tokens_total"
# Prompt tokens served straight out of the prefix cache. They are INCLUDED in
# vllm:prompt_tokens_total (v1/metrics/loggers.py:1194 increments it with
# iteration_stats.num_prompt_tokens) but cost no prefill compute, so prefill
# throughput has to subtract them -- which is exactly what vLLM's own
# "Avg prompt throughput" line does (:147 uses prompt_token_stats.computed).
PROMPT_TOK_CACHED_TOTAL = "vllm:prompt_tokens_cached_total"
# Histogram of per-request prefill wall time (first SCHEDULED -> first token,
# v1/metrics/stats.py:544). _sum over all finished requests is the denominator
# of the lifetime prefill rate.
#
# WHY IT IS NOT THE DENOMINATOR OF THE *WINDOW* RATE.
# "Prefill throughput is prompt tokens per second of prefill time, not per
# second of wall clock" is right about the quantity and impossible to compute
# per window on this build. prompt_tokens_total accrues CONTINUOUSLY, one
# increment per engine iteration; this histogram is observed ONCE, when a
# request reaches its first token, and the observation carries the request's
# whole prefill duration. The two are not co-timed, so their deltas over a
# short window are not a ratio of anything.
#
# Measured on the live Flash-Next server (:8001), 10.016 s apart, recorded at
# tests/fixtures/metrics_live_prefill_lag_t{1,2}.txt:
#     d(prompt_tokens_total)               =      0 tokens
#     d(request_prefill_time_seconds_sum)  = 20.005 s
#     d(request_prefill_time_seconds_count)=      1 request
# A request that had been prefilling across earlier windows reached its first
# token inside this one. Dividing gives 0 / 20.005 = 0 tok/s for a window in
# which the engine had, in fact, prefilled. The wall-clock denominator gives
# 0 tok/s too, but says something TRUE: no prompt tokens were computed in
# this window.
#
# And there is no substitute: this build exposes no cumulative counter of
# seconds the ENGINE spent prefilling (checked against the full family list of
# a live scrape -- the only prefill-time family is this per-request latency
# histogram). So:
#   window   figure = computed prompt tokens / wall seconds   (aggregate; the
#            same quantity vLLM's own log prints as "Avg prompt throughput")
#   lifetime figure = computed prompt tokens / prefill seconds (per second of
#            prefill time, as asked -- the completion lag averages out over
#            thousands of requests)
# They are DIFFERENT QUANTITIES and the panel must never substitute one for
# the other in the same slot. It used to, marked only by a "~".
PREFILL_TIME_SUM = "vllm:request_prefill_time_seconds_sum"
PREFILL_TIME_COUNT = "vllm:request_prefill_time_seconds_count"
#: Time-to-first-token histogram. _sum/_count over the poll window give the
#: MEAN TTFT of the requests that started producing tokens in that window;
#: over the lifetime they give the server's average since it booted. Verified
#: live against http://127.0.0.1:8001/metrics on 2026-09-09 (the exposition
#: recorded at tests/fixtures/metrics_flashnext_live.txt).
TTFT_SUM = "vllm:time_to_first_token_seconds_sum"
TTFT_COUNT = "vllm:time_to_first_token_seconds_count"
#: Per-request mean time between output tokens. NOTE the `request_` prefix:
#: this build exposes `vllm:request_time_per_output_token_seconds`, not the
#: `vllm:time_per_output_token_seconds` name older vLLM used. A guess at the
#: shorter name silently yields "not exposed" forever.
TPOT_SUM = "vllm:request_time_per_output_token_seconds_sum"
TPOT_COUNT = "vllm:request_time_per_output_token_seconds_count"
#: Seconds spent DECODING, the denominator of the lifetime decode rate. Like
#: prefill time, it does not decay while the server is idle.
DECODE_TIME_SUM = "vllm:request_decode_time_seconds_sum"
DECODE_TIME_COUNT = "vllm:request_decode_time_seconds_count"
#: The engine's own resolved cache configuration, exposed as one info metric
#: whose LABELS carry the numbers. `kv_cache_size_tokens` here is the same
#: figure the boot log prints as "GPU KV cache size: N tokens" — a MEASURED
#: capacity, read from the running engine, with no estimate in it.
CACHE_CONFIG = "vllm:cache_config_info"
SUCCESS_TOTAL = "vllm:request_success_total"
PREFIX_HITS = "vllm:prefix_cache_hits_total"
PREFIX_QUERIES = "vllm:prefix_cache_queries_total"


def parse_prometheus(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Parse exposition format into {name: [(labels, value), ...]}.

    Ignores # HELP/# TYPE. Tolerates the label-less form. Never raises on a
    malformed line — a metrics endpoint that is half-written during a restart
    must not take the poller down with it.
    """
    out: dict[str, list[tuple[dict[str, str], float]]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            if "{" in line:
                name, rest = line.split("{", 1)
                labelpart, valuepart = rest.rsplit("}", 1)
                labels: dict[str, str] = {}
                for pair in _split_labels(labelpart):
                    if "=" not in pair:
                        continue
                    k, v = pair.split("=", 1)
                    labels[k.strip()] = v.strip().strip('"')
                value = float(valuepart.split()[0])
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                name, labels, value = parts[0], {}, float(parts[1])
        except (ValueError, IndexError):
            continue
        out.setdefault(name.strip(), []).append((labels, value))
    return out


def _split_labels(s: str) -> list[str]:
    """Split on commas that are not inside quotes."""
    parts, cur, inq = [], [], False
    for ch in s:
        if ch == '"':
            inq = not inq
        if ch == "," and not inq:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return parts


def _first(parsed: dict, name: str, default: float = 0.0) -> float:
    rows = parsed.get(name)
    return rows[0][1] if rows else default


def _rate(delta: float, dt: float) -> float | None:
    """A per-second rate, or None when nothing moved.

    Zero tokens over the window means the server was idle for it. That is a
    different fact from "the server produced tokens slowly", and the UI must
    be able to tell them apart -- so idle is None, never 0.0.
    """
    return (delta / dt) if delta > 0 else None


#: Why a throughput/latency figure is not a number right now. The UI renders
#: "n/a" plus one of these, never a bare em dash and never 0: "no traffic in
#: the window" and "this build does not publish that metric" are different
#: facts, and only one of them is fixed by sending a request.
UNREACHABLE = "backend not reachable"
NO_BASELINE = "no baseline yet — first poll of this server"
COUNTER_RESET = "counters reset — the server restarted during this window"
IDLE = "idle — nothing ran in the sampling window"
NOT_EXPOSED = "not published by this vLLM build"

#: A short machine-readable code beside every reason. The page has to render
#: an idle figure differently from an absent one (idle keeps its last value and
#: an age; "this build does not publish that metric" has no last value to
#: keep), and matching on the PROSE above would mean an edit to one word of
#: English here silently changed how the dashboard behaves. tests/test_ui.py
#: asserts the page switches on codes from this table and on nothing else.
REASON_CODE = {
    UNREACHABLE: "unreachable",
    NO_BASELINE: "no_baseline",
    COUNTER_RESET: "reset",
    IDLE: "idle",
    NOT_EXPOSED: "not_exposed",
}


def _state(value: float | None, reason: str | None) -> str:
    """"ok" when there is a window reading, else the reason's code."""
    if value is not None:
        return "ok"
    return REASON_CODE.get(reason or "", "unknown")


def _label_int(labels: dict[str, str], key: str) -> int | None:
    """An integer label, or None. vLLM writes the string "None" for unset
    numeric config, which int() would raise on."""
    raw = labels.get(key)
    try:
        return int(raw) if raw not in (None, "", "None") else None
    except ValueError:
        return None


def _label_float(labels: dict[str, str], key: str) -> float | None:
    raw = labels.get(key)
    try:
        return float(raw) if raw not in (None, "", "None") else None
    except ValueError:
        return None


def _age(v: float | None) -> int | None:
    """Seconds since a reading, whole seconds. Sub-second precision on the age
    of a stale figure is noise -- the poll interval is 2 s -- and printing
    "0.7 s ago" invites reading it as a measurement rather than as a clock."""
    return None if v is None else int(v)


def _by_label(parsed: dict, name: str, key: str, val: str) -> float:
    for labels, v in parsed.get(name, []):
        if labels.get(key) == val:
            return v
    return 0.0


@dataclass
class MetricsSnapshot:
    reachable: bool = False
    kv_usage_perc: float = 0.0
    running: int = 0
    waiting: int = 0
    waiting_capacity: int = 0
    preemptions: int = 0
    avg_prompt_tokens: float = 0.0
    prompt_token_count: int = 0
    # None means "not known right now", and is NOT interchangeable with 0.0.
    # An idle server has no throughput; rendering that as "0 tok/s" reads as
    # "the machine got slow", which is the opposite of the truth. The UI shows
    # None as an em dash.
    gen_tok_s: float | None = None
    #: Prefill (prompt-processing) throughput over the last poll window, in
    #: computed prompt tokens per second. Same derivation as gen_tok_s, and
    #: the same quantity vLLM's own log prints as "Avg prompt throughput".
    prefill_tok_s: float | None = None
    #: Lifetime prefill throughput: computed prompt tokens per second OF
    #: PREFILL TIME (not of wall time). Prefill is bursty -- at
    #: --max-num-seqs 1 a 10k prompt prefills for ~10 s and then nothing
    #: prefills for minutes -- so the windowed rate above is unknown almost
    #: always. This one does not decay while the server is idle, because its
    #: denominator is prefill seconds.
    prefill_tok_s_avg: float | None = None
    #: Requests that have finished a prefill (the sample count behind the avg).
    prefill_requests: int = 0
    #: Lifetime decode throughput: generated tokens per second OF DECODE TIME.
    #: The windowed gen_tok_s above is unknown whenever nothing is generating;
    #: this one is the answer to "how fast does this server decode", and it
    #: survives an idle poll.
    gen_tok_s_avg: float | None = None
    #: Mean time-to-first-token of the requests that FINISHED PREFILLING in the
    #: last window (delta of the histogram sum over the delta of its count).
    ttft_s: float | None = None
    #: Same quantity over the server's whole life, so the panel still has a
    #: TTFT to show when no request started in the last two seconds.
    ttft_s_avg: float | None = None
    #: Requests whose time-to-first-token has been observed (the sample count
    #: behind ttft_s_avg), so the panel can say "mean of N", not just "mean".
    ttft_requests: int = 0
    #: The last WINDOW reading this poller saw for each figure, and how many
    #: seconds ago it was taken. A cell whose window is idle renders these
    #: instead of blanking: "idle - last 249.0 tok/s, 34 s ago" is a fact, an
    #: empty box is not, and the empty box is what let the panel look as
    #: though one figure had replaced the other. Cleared when the counters
    #: reset, because a reading from the previous process describes a server
    #: that no longer exists.
    prefill_tok_s_last: float | None = None
    prefill_last_age_s: float | None = None
    gen_tok_s_last: float | None = None
    gen_last_age_s: float | None = None
    ttft_s_last: float | None = None
    ttft_last_age_s: float | None = None
    #: Why each of the four figures above is not a number, when it is not.
    #: Never None at the same time as its value: exactly one of the pair is
    #: set, so the UI can always print either a reading or a reason.
    gen_reason: str | None = None
    prefill_reason: str | None = None
    ttft_reason: str | None = None
    #: The RUNNING engine's own resolved KV capacity, straight off
    #: vllm:cache_config_info. This is a measurement, not an estimate: it is
    #: the same number the boot log prints as "GPU KV cache size".
    kv_cache_size_tokens: int | None = None
    kv_cache_max_concurrency: float | None = None
    kv_cache_gpu_util: float | None = None
    requests_succeeded: int = 0
    prefix_hit_rate: float | None = None   # None = no queries yet, NOT 0%
    error: str | None = None
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        rate = lambda v: None if v is None else round(v, 1)  # noqa: E731
        return {
            "reachable": self.reachable,
            "kv_usage_perc": round(self.kv_usage_perc, 4),
            "running": self.running,
            "waiting": self.waiting,
            "waiting_capacity": self.waiting_capacity,
            "preemptions": self.preemptions,
            "avg_prompt_tokens": round(self.avg_prompt_tokens),
            "prompt_token_count": self.prompt_token_count,
            "gen_tok_s": rate(self.gen_tok_s),
            "gen_tok_s_avg": rate(self.gen_tok_s_avg),
            "prefill_tok_s": rate(self.prefill_tok_s),
            "prefill_tok_s_avg": rate(self.prefill_tok_s_avg),
            "prefill_requests": self.prefill_requests,
            # Seconds, not rounded to 1dp: a 0.04 s TTFT is a real reading and
            # round(_, 1) would print it as 0.0.
            "ttft_s": None if self.ttft_s is None else round(self.ttft_s, 3),
            "ttft_s_avg": None if self.ttft_s_avg is None else round(self.ttft_s_avg, 3),
            "ttft_requests": self.ttft_requests,
            "prefill_tok_s_last": rate(self.prefill_tok_s_last),
            "prefill_last_age_s": _age(self.prefill_last_age_s),
            "gen_tok_s_last": rate(self.gen_tok_s_last),
            "gen_last_age_s": _age(self.gen_last_age_s),
            "ttft_s_last": (
                None if self.ttft_s_last is None else round(self.ttft_s_last, 3)
            ),
            "ttft_last_age_s": _age(self.ttft_last_age_s),
            "gen_reason": self.gen_reason,
            "prefill_reason": self.prefill_reason,
            "ttft_reason": self.ttft_reason,
            "gen_state": _state(self.gen_tok_s, self.gen_reason),
            "prefill_state": _state(self.prefill_tok_s, self.prefill_reason),
            "ttft_state": _state(self.ttft_s, self.ttft_reason),
            "kv_cache_size_tokens": self.kv_cache_size_tokens,
            "kv_cache_max_concurrency": self.kv_cache_max_concurrency,
            "kv_cache_gpu_util": self.kv_cache_gpu_util,
            "requests_succeeded": self.requests_succeeded,
            "prefix_hit_rate": (
                None if self.prefix_hit_rate is None else round(self.prefix_hit_rate, 4)
            ),
            "error": self.error,
        }


def unreachable_snapshot() -> dict[str, Any]:
    """The payload to publish before the first scrape, and after a repoint.

    A bare ``{"reachable": False}`` is not the same shape as a real snapshot:
    the page reads a reason beside every missing figure and would find none,
    so the very state that most needs explaining ("nothing has been scraped
    yet") rendered as the unexplained blank the reasons exist to replace.
    """
    snap = MetricsSnapshot()
    snap.gen_reason = snap.prefill_reason = snap.ttft_reason = UNREACHABLE
    return snap.to_dict()


class MetricsPoller:
    """Scrapes /metrics and derives generation and prefill throughput.

    The endpoint is absent or 500 while the backend restarts; that is a normal
    state here, not an error worth propagating. reachable=False says so and the
    caller renders it as "no backend", never as a crash.

    Both rates are counter deltas over the poll interval — the same mechanism,
    so the two numbers on the serving line are always comparable and always
    describe the same window.
    """

    def __init__(
        self, base_url: str, *, monotonic: Callable[[], float] = time.monotonic
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # (ts, generation_tokens_total, computed_prompt_tokens, ttft_sum,
        # ttft_count). One baseline for every windowed figure, so the numbers
        # on the panel can never describe different windows.
        self._prev: tuple[float, float, float, float, float] | None = None
        # Injectable, and MONOTONIC: time.time() steps on an NTP correction,
        # and a backwards step silently scales every throughput number the UI
        # has ever shown.
        self._monotonic = monotonic
        # key -> (monotonic ts, value) of the last WINDOW reading that was a
        # number. The panel renders these while a window is idle so a figure
        # is never blank and never appears to have been replaced by its
        # neighbour. Not a rate cache: it is only ever shown with its age.
        self._last: dict[str, tuple[float, float]] = {}

    def _remember(self, snap: MetricsSnapshot, now: float) -> None:
        """Record this scrape's window readings, then attach the newest one
        for every figure -- including the figures this scrape has no reading
        for, which is the whole point.

        Called on EVERY exit path, unreachable ones included: a backend that
        just went away is exactly when "last 249.0 tok/s, 12 s ago" beats an
        empty box.
        """
        for key, live in (
            ("prefill", snap.prefill_tok_s),
            ("gen", snap.gen_tok_s),
            ("ttft", snap.ttft_s),
        ):
            if live is not None:
                self._last[key] = (now, live)
        for key, val_attr, age_attr in (
            ("prefill", "prefill_tok_s_last", "prefill_last_age_s"),
            ("gen", "gen_tok_s_last", "gen_last_age_s"),
            ("ttft", "ttft_s_last", "ttft_last_age_s"),
        ):
            rec = self._last.get(key)
            if rec is None:
                continue
            ts, value = rec
            setattr(snap, val_attr, value)
            # max(0.0, ...) because a caller may inject a clock that does not
            # advance; a negative age would render as "-0 s ago".
            setattr(snap, age_attr, max(0.0, now - ts))

    async def scrape(self, client: httpx.AsyncClient) -> MetricsSnapshot:
        snap = MetricsSnapshot()
        snap.gen_reason = snap.prefill_reason = snap.ttft_reason = UNREACHABLE
        # ONE clock read per scrape, taken before the request and used for
        # both the rate window and the age of every stale reading. Two reads
        # would date the window and the ages from different instants, which is
        # how an age of "-1 s ago" gets onto a dashboard.
        now = self._monotonic()
        try:
            r = await client.get(f"{self.base_url}/metrics", timeout=4.0)
            if r.status_code != 200:
                snap.error = f"HTTP {r.status_code}"
                self._prev = None
                self._remember(snap, now)
                return snap  # reasons already say UNREACHABLE
            text = r.text
        except Exception as exc:  # noqa: BLE001 - any transport failure means "down"
            snap.error = type(exc).__name__
            # Drop the baseline. Keeping it made the next successful scrape
            # divide a fresh counter delta by a dt spanning the whole outage,
            # so a backend that was down for ten minutes came back reporting a
            # plausible-looking throughput averaged over its own downtime.
            self._prev = None
            self._remember(snap, now)
            return snap

        p = parse_prometheus(text)
        snap.reachable = True
        snap.gen_reason = snap.prefill_reason = snap.ttft_reason = None
        snap.kv_usage_perc = _first(p, KV_USAGE)
        snap.running = int(_first(p, RUNNING))
        snap.waiting = int(_first(p, WAITING))
        snap.waiting_capacity = int(_by_label(p, WAITING_BY_REASON, "reason", "capacity"))
        snap.preemptions = int(_first(p, PREEMPTIONS))

        total = _first(p, PROMPT_TOK_SUM)
        count = _first(p, PROMPT_TOK_COUNT)
        snap.prompt_token_count = int(count)
        snap.avg_prompt_tokens = (total / count) if count else 0.0

        snap.requests_succeeded = int(sum(v for _, v in p.get(SUCCESS_TOTAL, [])))

        # Prefix cache hit rate is what people mean when they ask "is caching
        # working". kv_cache_usage_perc is unrelated: it is in-flight block
        # occupancy and correctly drops to 0 between requests.
        hits, queries = _first(p, PREFIX_HITS), _first(p, PREFIX_QUERIES)
        snap.prefix_hit_rate = (hits / queries) if queries > 0 else None

        gen_total = _first(p, GEN_TOK_TOTAL)
        # Prefill work = prompt tokens that were actually computed. A token
        # served out of the prefix cache is counted in prompt_tokens_total but
        # cost nothing to "prefill", and including it makes a cache hit look
        # like a throughput record. An older vLLM without the cached counter
        # yields 0.0 here and degrades to the raw prompt-token rate.
        prompt_computed = max(
            0.0, _first(p, PROMPT_TOK_TOTAL) - _first(p, PROMPT_TOK_CACHED_TOTAL)
        )

        # TTFT is a histogram, so the window figure is the mean of the requests
        # that reached their first token during it: delta(_sum)/delta(_count).
        # Dividing the lifetime _sum by the lifetime _count instead would print
        # an average over every request since boot and call it "now".
        ttft_sum = _first(p, TTFT_SUM)
        ttft_count = _first(p, TTFT_COUNT)
        has_ttft = bool(p.get(TTFT_COUNT))

        if self._prev is None:
            snap.gen_reason = snap.prefill_reason = snap.ttft_reason = NO_BASELINE
        else:
            prev_ts, prev_gen, prev_prompt, prev_ttft_sum, prev_ttft_count = self._prev
            dt = now - prev_ts
            # Counters restart at 0 with the process. A counter that went
            # backwards means "new server", so this window spans two different
            # processes and has no throughput -- which is "unknown", not 0.
            reset = gen_total < prev_gen or prompt_computed < prev_prompt
            if dt <= 0 or reset:
                reason = COUNTER_RESET if reset else NO_BASELINE
                snap.gen_reason = snap.prefill_reason = snap.ttft_reason = reason
                if reset:
                    # A reading taken from the process that just died is not a
                    # stale reading of THIS server, it is a reading of a
                    # different one. Showing it with an age would be a lie
                    # with a timestamp on it.
                    self._last.clear()
            else:
                snap.gen_tok_s = _rate(gen_total - prev_gen, dt)
                snap.prefill_tok_s = _rate(prompt_computed - prev_prompt, dt)
                if snap.gen_tok_s is None:
                    snap.gen_reason = IDLE
                if snap.prefill_tok_s is None:
                    snap.prefill_reason = IDLE
                if not has_ttft:
                    snap.ttft_reason = NOT_EXPOSED
                elif ttft_count > prev_ttft_count:
                    snap.ttft_s = (ttft_sum - prev_ttft_sum) / (
                        ttft_count - prev_ttft_count
                    )
                else:
                    snap.ttft_reason = IDLE
        self._prev = (now, gen_total, prompt_computed, ttft_sum, ttft_count)

        # Lifetime companions. They do not decay while the server is idle, so
        # the panel always has all three of prefill / decode / TTFT to show —
        # marked as lifetime rather than silently substituted for the window.
        if has_ttft and ttft_count > 0:
            snap.ttft_s_avg = ttft_sum / ttft_count
            snap.ttft_requests = int(ttft_count)
        decode_seconds = _first(p, DECODE_TIME_SUM)
        if decode_seconds > 0 and gen_total > 0:
            snap.gen_tok_s_avg = gen_total / decode_seconds

        # The engine's own resolved cache configuration. Every number is a
        # LABEL on a single info metric whose value is always 1.0, so it has to
        # be read out of the labels, not out of the value.
        for labels, _v in p.get(CACHE_CONFIG, []):
            snap.kv_cache_size_tokens = _label_int(labels, "kv_cache_size_tokens")
            snap.kv_cache_max_concurrency = _label_float(labels, "kv_cache_max_concurrency")
            snap.kv_cache_gpu_util = _label_float(labels, "gpu_memory_utilization")
            break

        # Lifetime prefill rate. Denominator is seconds spent prefilling, not
        # seconds elapsed, so it stays meaningful while the server sits idle.
        # The numerator counts in-flight requests the denominator has not seen
        # yet (the histogram only observes finished ones), so it overshoots
        # briefly during a large prefill and then settles.
        prefill_seconds = _first(p, PREFILL_TIME_SUM)
        snap.prefill_requests = int(_first(p, PREFILL_TIME_COUNT))
        if prefill_seconds > 0 and prompt_computed > 0:
            snap.prefill_tok_s_avg = prompt_computed / prefill_seconds
        self._remember(snap, now)
        return snap
