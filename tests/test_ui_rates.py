"""The prefill and decode cells say which kind of rate each line is.

THE DEFECT (owner, 2026-09-11)
------------------------------
The Decode cell read "858.6 tok/s · now · last 2 s window" over "lifetime 68.0
tok/s per second of decode time", and was read as "lifetime throughput is
low". They are different kinds of number:

  * the window figure is delta(vllm:generation_tokens_total) / wall seconds:
    every running request TOGETHER;
  * the lifetime figure is generation_tokens_total /
    vllm:request_decode_time_seconds_sum, and that sum is each finished
    request's decode time added up -- request-seconds, not wall seconds. It is
    ONE request's decode speed. Live: 398,172 tokens / 6,020 s over 235
    requests = 66 tok/s per stream, while ~13 streams together made 858.6.

Prefill is built the same way (request_prefill_time_seconds is per request,
and requests prefill side by side under load), so it had the same mix-up. TTFT
is a mean per request on both lines and did not.

Executed, as everywhere in these UI tests: the payload comes from the real
MetricsPoller scraping exposition text, and paintThroughput() runs against the
DOM shim.
"""

from __future__ import annotations

import pytest

from servedeck import metrics

from .test_metrics import _scrape_all
from .test_ui import _render

pytest.importorskip("dukpy", reason="pip install -e '.[dev]' brings in the JS engine")

_LABELS = 'engine="0",model_name="Qwen3.8-27B-NVFP4"'


def _exposition(*, gen: float, prompt: float, cached: float) -> str:
    """One /metrics page of the live 27B during the 2026-09-11 stress run:
    lifetime 398,172 generated tokens over 6,020 s of summed decode time in
    235 finished requests, and 3,211,629 computed prompt tokens (22,283,661
    minus 19,072,032 cached) over 632.71 s of summed prefill time in 586."""
    rows = {
        "vllm:generation_tokens_total": gen,
        "vllm:prompt_tokens_total": prompt,
        "vllm:prompt_tokens_cached_total": cached,
        "vllm:request_decode_time_seconds_sum": 6020.0,
        "vllm:request_decode_time_seconds_count": 235.0,
        "vllm:request_prefill_time_seconds_sum": 632.71,
        "vllm:request_prefill_time_seconds_count": 586.0,
        "vllm:num_requests_running": 13.0,
        "vllm:num_requests_waiting": 0.0,
        "vllm:kv_cache_usage_perc": 0.41,
    }
    return "".join(f"{name}{{{_LABELS}}} {value}\n" for name, value in rows.items())


def _stress_run() -> dict:
    """Two scrapes 2 s apart: 1,717.2 tokens generated (858.6 tok/s across
    the 13 streams) and 6,240 prompt tokens computed (3,120 tok/s)."""
    t1 = _exposition(gen=398_172 - 1_717.2, prompt=22_283_661 - 6_240, cached=19_072_032)
    t2 = _exposition(gen=398_172, prompt=22_283_661, cached=19_072_032)
    return _scrape_all([t1, t2], [0.0, 2.0])[1].to_dict()


_SERVING = {"upstream": {"up": True, "port": 8004, "max_model_len": 262144},
            "server_uptime_s": 2171}


def test_the_decode_cell_says_aggregate_above_and_per_request_below() -> None:
    payload = _stress_run()
    assert payload["gen_tok_s"] == pytest.approx(858.6, abs=0.05)
    assert payload["gen_tok_s_avg"] == pytest.approx(66.1, abs=0.05)   # 398,172 / 6,020
    dom = _render(payload, _SERVING)
    assert dom["thDecode"]["text"] == "858.6 tok/s"
    assert dom["thDecodeL"]["text"] == "per request: 66.1 tok/s (lifetime mean over 235 requests)", (
        dom["thDecodeL"]["text"]
    )
    assert dom["thDecodeS"]["text"] == "now · all requests together, last 2 s window", (
        dom["thDecodeS"]["text"]
    )


def test_the_prefill_cell_had_the_same_mix_up_and_says_so_too() -> None:
    """3,211,629 computed prompt tokens / 632.71 request-seconds of prefill =
    5,076 tok/s for one request; the window figure is the engine's 3,120 tok/s
    across everything prefilling at once."""
    payload = _stress_run()
    dom = _render(payload, _SERVING)
    assert dom["thPrefill"]["text"] == "3,120 tok/s"
    assert "all requests together" in dom["thPrefillS"]["text"], dom["thPrefillS"]["text"]
    assert dom["thPrefillL"]["text"] == (
        "per request: 5,076 tok/s (lifetime mean over 586 requests)"
    ), dom["thPrefillL"]["text"]


def test_no_lifetime_line_still_reads_as_a_throughput_of_the_server() -> None:
    """The phrasing that was misread, gone from both cells in every state --
    including idle, where the headline says "idle" and the window line
    carries the last aggregate reading, labelled as aggregate."""
    busy = _stress_run()
    idle = dict(busy, gen_tok_s=None, gen_state="idle", gen_reason="idle — nothing ran",
                gen_tok_s_last=858.6, gen_last_age_s=12.0,
                prefill_tok_s=None, prefill_state="idle", prefill_reason="idle — nothing ran",
                prefill_tok_s_last=3120.0, prefill_last_age_s=12.0)
    for payload in (busy, idle):
        dom = _render(payload, _SERVING)
        for cell in ("thPrefillL", "thDecodeL"):
            text = dom[cell]["text"]
            assert text.startswith("per request: "), text
            assert "per second of" not in text, text
    dom = _render(idle, _SERVING)
    assert dom["thDecode"]["text"] == "idle"
    assert "last 858.6 tok/s (all requests together)" in dom["thDecodeS"]["text"], (
        dom["thDecodeS"]["text"]
    )


def test_ttft_keeps_its_mean_label_on_both_lines() -> None:
    """Over-correction guard: TTFT is a mean per request in the window AND over
    the lifetime, so it must not be relabelled as an aggregate or given the
    per-stream wording of the rate cells."""
    dom = _render(metrics.MetricsSnapshot(reachable=True, ttft_s=1.85, ttft_s_avg=9.1,
                                          ttft_requests=598).to_dict(), _SERVING)
    assert dom["thTtftS"]["text"] == "now · mean per request, last 2 s window"
    assert dom["thTtftL"]["text"] == "lifetime 9.1 s mean over 598 requests"
    assert "all requests together" not in dom["thTtftS"]["text"]


def test_the_decode_request_count_comes_from_the_engine() -> None:
    """"over N requests" needs N: vllm:request_decode_time_seconds_count, off
    the recorded live Flash-Next exposition (551 finished requests)."""
    from .test_metrics import _live_text

    text = _live_text()
    snap = _scrape_all([text, text], [0.0, 2.0])[1]
    assert snap.gen_requests == 551
    assert snap.to_dict()["gen_requests"] == 551
