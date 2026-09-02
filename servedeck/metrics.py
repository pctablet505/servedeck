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
PREFILL_TIME_SUM = "vllm:request_prefill_time_seconds_sum"
PREFILL_TIME_COUNT = "vllm:request_prefill_time_seconds_count"
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
            "prefill_tok_s": rate(self.prefill_tok_s),
            "prefill_tok_s_avg": rate(self.prefill_tok_s_avg),
            "prefill_requests": self.prefill_requests,
            "requests_succeeded": self.requests_succeeded,
            "prefix_hit_rate": (
                None if self.prefix_hit_rate is None else round(self.prefix_hit_rate, 4)
            ),
            "error": self.error,
        }


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
        # (ts, generation_tokens_total, computed_prompt_tokens). One baseline
        # for both rates so they can never be measured over different windows.
        self._prev: tuple[float, float, float] | None = None
        # Injectable, and MONOTONIC: time.time() steps on an NTP correction,
        # and a backwards step silently scales every throughput number the UI
        # has ever shown.
        self._monotonic = monotonic

    async def scrape(self, client: httpx.AsyncClient) -> MetricsSnapshot:
        snap = MetricsSnapshot()
        try:
            r = await client.get(f"{self.base_url}/metrics", timeout=4.0)
            if r.status_code != 200:
                snap.error = f"HTTP {r.status_code}"
                self._prev = None
                return snap
            text = r.text
        except Exception as exc:  # noqa: BLE001 - any transport failure means "down"
            snap.error = type(exc).__name__
            # Drop the baseline. Keeping it made the next successful scrape
            # divide a fresh counter delta by a dt spanning the whole outage,
            # so a backend that was down for ten minutes came back reporting a
            # plausible-looking throughput averaged over its own downtime.
            self._prev = None
            return snap

        p = parse_prometheus(text)
        snap.reachable = True
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

        now = self._monotonic()
        if self._prev is not None:
            prev_ts, prev_gen, prev_prompt = self._prev
            dt = now - prev_ts
            # Counters restart at 0 with the process. A counter that went
            # backwards means "new server", so this window spans two different
            # processes and has no throughput -- which is "unknown", not 0.
            if dt > 0 and gen_total >= prev_gen and prompt_computed >= prev_prompt:
                snap.gen_tok_s = _rate(gen_total - prev_gen, dt)
                snap.prefill_tok_s = _rate(prompt_computed - prev_prompt, dt)
        self._prev = (now, gen_total, prompt_computed)

        # Lifetime prefill rate. Denominator is seconds spent prefilling, not
        # seconds elapsed, so it stays meaningful while the server sits idle.
        # The numerator counts in-flight requests the denominator has not seen
        # yet (the histogram only observes finished ones), so it overshoots
        # briefly during a large prefill and then settles.
        prefill_seconds = _first(p, PREFILL_TIME_SUM)
        snap.prefill_requests = int(_first(p, PREFILL_TIME_COUNT))
        if prefill_seconds > 0 and prompt_computed > 0:
            snap.prefill_tok_s_avg = prompt_computed / prefill_seconds
        return snap
