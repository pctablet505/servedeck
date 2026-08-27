"""Prometheus text-format scraping for the live vLLM server.

Deliberately a small hand-rolled parser rather than prometheus_client: we need
exactly six families out of a ~200-line exposition page, and pulling in a
metrics *library* to read someone else's metrics is not worth the dependency.

Every metric name here was verified live against
http://localhost:8001/metrics on 2026-08-27 — do not guess at names, vLLM
renames these between versions.
"""

from __future__ import annotations

import time
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
    gen_tok_s: float = 0.0
    requests_succeeded: int = 0
    prefix_hit_rate: float | None = None   # None = no queries yet, NOT 0%
    error: str | None = None
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reachable": self.reachable,
            "kv_usage_perc": round(self.kv_usage_perc, 4),
            "running": self.running,
            "waiting": self.waiting,
            "waiting_capacity": self.waiting_capacity,
            "preemptions": self.preemptions,
            "avg_prompt_tokens": round(self.avg_prompt_tokens),
            "prompt_token_count": self.prompt_token_count,
            "gen_tok_s": round(self.gen_tok_s, 1),
            "requests_succeeded": self.requests_succeeded,
            "prefix_hit_rate": (
                None if self.prefix_hit_rate is None else round(self.prefix_hit_rate, 4)
            ),
            "error": self.error,
        }


class MetricsPoller:
    """Scrapes /metrics and derives a rate for generation throughput.

    The endpoint is absent or 500 while the backend restarts; that is a normal
    state here, not an error worth propagating. reachable=False says so and the
    caller renders it as "no backend", never as a crash.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._prev_gen: tuple[float, float] | None = None  # (ts, total)

    async def scrape(self, client: httpx.AsyncClient) -> MetricsSnapshot:
        snap = MetricsSnapshot()
        try:
            r = await client.get(f"{self.base_url}/metrics", timeout=4.0)
            if r.status_code != 200:
                snap.error = f"HTTP {r.status_code}"
                return snap
            text = r.text
        except Exception as exc:  # noqa: BLE001 - any transport failure means "down"
            snap.error = type(exc).__name__
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
        now = time.time()
        if self._prev_gen is not None:
            prev_ts, prev_total = self._prev_gen
            dt = now - prev_ts
            # counters reset to 0 on a server restart; a negative delta means
            # "new process", not negative throughput.
            if dt > 0 and gen_total >= prev_total:
                snap.gen_tok_s = (gen_total - prev_total) / dt
        self._prev_gen = (now, gen_total)
        return snap
