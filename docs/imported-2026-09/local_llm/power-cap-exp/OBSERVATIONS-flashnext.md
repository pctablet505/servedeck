# GPU Power-Cap Experiment — Qwen3.8-Flash-Next ABLITERATED

**Server:** vLLM, model `qwen38-flash-next` on `http://127.0.0.1:8001`.
**GPU:** RTX PRO 6000 Blackwell (power limit range 150–600 W; original cap 500 W, auto-restored at the end).
**Prefill Workload:** 39,760-token prompt (`prompt40k.txt`) with unique cache-busting salt line per cap (guaranteed cold prefill).
**Decode Workload:** continuous generation prompt (`ignore_eos=True`, `max_tokens=8192`, `temp=0.7`) to measure true sustained decode throughput.
**Windows:** decode-1 and decode-6 each run a 45 s window starting at the first token; tokens measured via vLLM continuous usage stats; power/util/temp sampled every 0.5 s.
**Order:** caps descend 585 W -> 200 W in 10% steps; each cap: warmup -> cold prefill -> decode-1 -> settle -> decode-6 -> settle.

## Results

| # | Cap (W) | Cold Prefill TTFT (s) | Prefill (tok/s) | Decode-1 (tok/s) | D1 tokens/45s | Decode-6 (tok/s) | D6 tokens/45s | PWR avg→max (W) | Temp max (°C) | Status / notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 585 | 2.78 | 14302.2 | 133.5 | 6009 | 537.7 | 24197 | 414.8→439.7 / 484.7→525.6 | 81.0 | ok |
| 2 | 526 | 3.02 | 13165.6 | 127.2 | 5725 | 552.3 | 24853 | 438.1→454.3 / 508.7→528.8 | 85.0 | ok |
| 3 | 474 | 3.19 | 12463.9 | 136.0 | 6118 | 552.5 | 24862 | 444.2→459.3 / 465.7→481.0 | 83.0 | ok |
| 4 | 426 | 3.44 | 11558.1 | 133.4 | 6002 | 527.5 | 23737 | 416.5→431.0 / 419.7→429.1 | 80.0 | ok |
| 5 | 384 | 3.71 | 10717.0 | 145.7 | 6558 | 523.1 | 23539 | 378.1→387.9 / 377.4→399.4 | 77.0 | ok |
| 6 | 345 | 4.04 | 9841.6 | 127.9 | 5757 | 492.5 | 22161 | 338.4→347.3 / 338.8→358.5 | 75.0 | ok |
| 7 | 311 | 4.37 | 9098.4 | 128.8 | 5794 | 494.6 | 22259 | 305.8→311.5 / 306.2→320.8 | 72.0 | ok |
| 8 | 280 | 4.85 | 8197.9 | 119.5 | 5379 | 457.2 | 20575 | 275.7→280.4 / 276.3→283.6 | 69.0 | ok |
| 9 | 252 | 5.35 | 7431.8 | 114.9 | 5171 | 415.3 | 18690 | 248.2→267.1 / 248.5→261.1 | 67.0 | ok |
| 10 | 227 | 6.03 | 6593.7 | 105.8 | 4759 | 367.9 | 16555 | 223.4→243.2 / 224.6→227.7 | 64.0 | ok |
| 11 | 204 | — | — | — | — | — | — | — | — | pending |
| 12 | 200 | — | — | — | — | — | — | — | — | pending |

## Notes
- Cold prefill is measured on 39,760 tokens with a unique salt to bust the prefix cache.
- Decode phases run an open-ended continuous generation prompt with ignore_eos to measure sustained generation speed.
- Concurrency is 6 parallel requests (`Decode-6`).
- Step interval is 10% (585W -> 200W).
- Tokens are tracked via vLLM's internal token counters (continuous_usage_stats).
- `idle_warn` / notes flag rows where other GPU workloads were active.
- power columns: avg→max during decode-1 / decode-6.
- raw data: `/home/pctablet505/Projects/local_llm/power-cap-exp/results-flashnext.csv`; full log: `/home/pctablet505/Projects/local_llm/power-cap-exp/experiment-flashnext.log`; original cap restored automatically on exit.

## Log (last 100 lines)

[2026-09-11T23:47:00] === sweep starting: caps=[585, 526, 474, 426, 384, 345, 311, 280, 252, 227, 204, 200]W original_cap=500.0W window=45.0s prefill_prompt_tokens=39760 server=http://127.0.0.1:8001 parallel=6 ===
[2026-09-11T23:47:00] NOTE: stop other GPU workloads for clean numbers.
[2026-09-11T23:47:00] --- [1/12] power cap 585W ---
[2026-09-11T23:47:00] sudo authenticated via SUDO_PASS env - continuing.
[2026-09-11T23:47:00]   cap set: 585W
[2026-09-11T23:47:01]   warmup: 16 tok @ current cap
[2026-09-11T23:47:09]   prefill: TTFT 2.78 s (14302.2 tok/s) @ 585W
[2026-09-11T23:47:59]   decode-1: 6009 tok in 45.0 s = 133.5 tok/s (pwr avg 414.8W)
[2026-09-11T23:48:49]   decode-6: 24197 tok in 45.0 s = 537.7 tok/s (pwr avg 484.7W)
[2026-09-11T23:48:54] --- [2/12] power cap 526W ---
[2026-09-11T23:48:54]   cap set: 526W
[2026-09-11T23:48:54]   warmup: 16 tok @ current cap
[2026-09-11T23:49:02]   prefill: TTFT 3.02 s (13165.6 tok/s) @ 526W
[2026-09-11T23:49:53]   decode-1: 5725 tok in 45.0 s = 127.2 tok/s (pwr avg 438.1W)
[2026-09-11T23:50:43]   decode-6: 24853 tok in 45.0 s = 552.3 tok/s (pwr avg 508.7W)
[2026-09-11T23:50:48] --- [3/12] power cap 474W ---
[2026-09-11T23:50:48]   cap set: 474W
[2026-09-11T23:50:48]   warmup: 16 tok @ current cap
[2026-09-11T23:50:56]   prefill: TTFT 3.19 s (12463.9 tok/s) @ 474W
[2026-09-11T23:51:46]   decode-1: 6118 tok in 45.0 s = 136.0 tok/s (pwr avg 444.2W)
[2026-09-11T23:52:36]   decode-6: 24862 tok in 45.0 s = 552.5 tok/s (pwr avg 465.7W)
[2026-09-11T23:52:41] --- [4/12] power cap 426W ---
[2026-09-11T23:52:41]   cap set: 426W
[2026-09-11T23:52:42]   warmup: 16 tok @ current cap
[2026-09-11T23:52:50]   prefill: TTFT 3.44 s (11558.1 tok/s) @ 426W
[2026-09-11T23:53:40]   decode-1: 6002 tok in 45.0 s = 133.4 tok/s (pwr avg 416.5W)
[2026-09-11T23:54:30]   decode-6: 23737 tok in 45.0 s = 527.5 tok/s (pwr avg 419.7W)
[2026-09-11T23:54:35] --- [5/12] power cap 384W ---
[2026-09-11T23:54:35]   cap set: 384W
[2026-09-11T23:54:36]   warmup: 16 tok @ current cap
[2026-09-11T23:54:44]   prefill: TTFT 3.71 s (10717.0 tok/s) @ 384W
[2026-09-11T23:55:34]   decode-1: 6558 tok in 45.0 s = 145.7 tok/s (pwr avg 378.1W)
[2026-09-11T23:56:25]   decode-6: 23539 tok in 45.0 s = 523.1 tok/s (pwr avg 377.4W)
[2026-09-11T23:56:30] --- [6/12] power cap 345W ---
[2026-09-11T23:56:30]   cap set: 345W
[2026-09-11T23:56:30]   warmup: 16 tok @ current cap
[2026-09-11T23:56:39]   prefill: TTFT 4.04 s (9841.6 tok/s) @ 345W
[2026-09-11T23:57:29]   decode-1: 5757 tok in 45.0 s = 127.9 tok/s (pwr avg 338.4W)
[2026-09-11T23:58:19]   decode-6: 22161 tok in 45.0 s = 492.5 tok/s (pwr avg 338.8W)
[2026-09-11T23:58:24] --- [7/12] power cap 311W ---
[2026-09-11T23:58:24]   cap set: 311W
[2026-09-11T23:58:24]   warmup: 16 tok @ current cap
[2026-09-11T23:58:34]   prefill: TTFT 4.37 s (9098.4 tok/s) @ 311W
[2026-09-11T23:59:24]   decode-1: 5794 tok in 45.0 s = 128.8 tok/s (pwr avg 305.8W)
[2026-09-12T00:00:14]   decode-6: 22259 tok in 45.0 s = 494.6 tok/s (pwr avg 306.2W)
[2026-09-12T00:00:19] --- [8/12] power cap 280W ---
[2026-09-12T00:00:19]   cap set: 280W
[2026-09-12T00:00:19]   warmup: 16 tok @ current cap
[2026-09-12T00:00:29]   prefill: TTFT 4.85 s (8197.9 tok/s) @ 280W
[2026-09-12T00:01:19]   decode-1: 5379 tok in 45.0 s = 119.5 tok/s (pwr avg 275.7W)
[2026-09-12T00:02:09]   decode-6: 20575 tok in 45.0 s = 457.2 tok/s (pwr avg 276.3W)
[2026-09-12T00:02:14] --- [9/12] power cap 252W ---
[2026-09-12T00:02:15]   cap set: 252W
[2026-09-12T00:02:15]   warmup: 16 tok @ current cap
[2026-09-12T00:02:25]   prefill: TTFT 5.35 s (7431.8 tok/s) @ 252W
[2026-09-12T00:03:15]   decode-1: 5171 tok in 45.0 s = 114.9 tok/s (pwr avg 248.2W)
[2026-09-12T00:04:05]   decode-6: 18690 tok in 45.0 s = 415.3 tok/s (pwr avg 248.5W)
[2026-09-12T00:04:10] --- [10/12] power cap 227W ---
[2026-09-12T00:04:10]   cap set: 227W
[2026-09-12T00:04:11]   warmup: 16 tok @ current cap
[2026-09-12T00:04:22]   prefill: TTFT 6.03 s (6593.7 tok/s) @ 227W
[2026-09-12T00:05:12]   decode-1: 4759 tok in 45.0 s = 105.8 tok/s (pwr avg 223.4W)
[2026-09-12T00:06:02]   decode-6: 16555 tok in 45.0 s = 367.9 tok/s (pwr avg 224.6W)
