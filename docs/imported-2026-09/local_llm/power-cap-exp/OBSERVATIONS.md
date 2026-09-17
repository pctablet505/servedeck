# GPU Power-Cap Experiment — Live Observations

**Server:** vLLM 0.29.0, model `Qwen3.8-27B-NVFP4` on `:8004` — same server throughout, never restarted.
**GPU:** RTX PRO 6000 Blackwell (power limit range 150–600 W; original cap 500 W, auto-restored at the end).
**Prefill Workload:** 39,760-token prompt (`prompt40k.txt`) with unique cache-busting salt line per cap (guaranteed cold prefill).
**Decode Workload:** continuous generation prompt (`ignore_eos=True`, `max_tokens=8192`, `temp=0.7`) to measure true sustained decode throughput.
**Windows:** decode-1 and decode-16 each run a 45 s window starting at the first token; tokens measured via vLLM continuous usage stats; power/util/temp sampled every 0.5 s.
**Order:** caps descend 585 W → 200 W in 5% steps; each cap: warmup → cold prefill → decode-1 → settle → decode-16 → settle.

## Results

| # | Cap (W) | Cold Prefill TTFT (s) | Prefill (tok/s) | Decode-1 (tok/s) | D1 tokens/45s | Decode-16 (tok/s) | D16 tokens/45s | PWR avg→max (W) | Temp max (°C) | Status / notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 585 | 4.48 | 8875.0 | 140.2 | 6310 | 1467.0 | 66014 | 439.8→460.7 / 571.8→588.9 | 86.0 | ok |
| 2 | 556 | 4.72 | 8423.7 | 137.2 | 6175 | 1402.3 | 63104 | 466.5→483.8 / 545.2→563.2 | 89.0 | ok |
| 3 | 528 | 4.89 | 8130.9 | 138.6 | 6238 | 1409.5 | 63428 | 471.3→487.7 / 518.9→528.4 | 88.0 | ok |
| 4 | 502 | 5.06 | 7857.7 | 144.7 | 6511 | 1351.6 | 60820 | 472.3→487.3 / 494.2→503.3 | 87.0 | ok |
| 5 | 476 | 5.25 | 7573.3 | 197.2 | 8192 | 1358.9 | 61149 | 466.5→479.0 / 467.3→488.6 | 86.0 | ok |
| 6 | 453 | 5.49 | 7242.3 | 128.1 | 5764 | 1320.9 | 59441 | 446.0→460.8 / 445.1→454.4 | 85.0 | ok |
| 7 | 430 | 5.74 | 6926.8 | 131.3 | 5909 | 1295.0 | 58277 | 422.2→443.1 / 422.3→441.5 | 83.0 | ok |
| 8 | 409 | 5.79 | 6867.0 | 133.3 | 5997 | 1318.7 | 59340 | 402.0→410.0 / 403.9→410.9 | 82.0 | ok |
| 9 | 388 | 6.01 | 6615.6 | 136.0 | 6122 | 1220.7 | 54930 | 380.7→401.5 / 380.2→388.0 | 80.0 | ok |
| 10 | 369 | 6.28 | 6331.2 | 139.9 | 6294 | 1171.6 | 52720 | 363.4→370.3 / 363.2→372.0 | 78.0 | ok |
| 11 | 350 | 6.66 | 5970.0 | 131.5 | 5919 | 1152.6 | 51865 | 344.8→351.4 / 344.4→354.2 | 77.0 | ok |
| 12 | 333 | 6.79 | 5855.7 | 124.3 | 5593 | 1101.4 | 49561 | 326.9→336.7 / 326.3→333.0 | 75.0 | ok |
| 13 | 316 | 7.08 | 5615.8 | 141.0 | 6344 | 1062.3 | 47805 | 310.2→335.7 / 311.2→320.6 | 75.0 | ok |
| 14 | 300 | 7.44 | 5344.1 | 141.4 | 6364 | 1023.9 | 46077 | 295.6→301.8 / 294.5→316.4 | 72.0 | ok |
| 15 | 285 | — | — | — | — | — | — | — | — | pending |
| 16 | 271 | — | — | — | — | — | — | — | — | pending |
| 17 | 257 | — | — | — | — | — | — | — | — | pending |
| 18 | 245 | — | — | — | — | — | — | — | — | pending |
| 19 | 232 | — | — | — | — | — | — | — | — | pending |
| 20 | 221 | — | — | — | — | — | — | — | — | pending |
| 21 | 210 | — | — | — | — | — | — | — | — | pending |
| 22 | 200 | — | — | — | — | — | — | — | — | pending |

## Notes
- Cold prefill is measured on 39,760 tokens with a unique salt to bust the prefix cache.
- Decode phases run an open-ended continuous generation prompt with ignore_eos to measure sustained generation speed.
- Tokens are tracked via vLLM's internal token counters (continuous_usage_stats), not SSE chunk counts.
- `idle_warn` / notes flag rows where other GPU workloads were active.
- power columns: avg→max during decode-1 / decode-16.
- raw data: `results.csv`; full log: `experiment.log`; original cap restored automatically on exit (or `./restore.sh`).

## Log (last 100 lines)

[2026-09-11T23:07:50]   prefill: TTFT 4.41 s (9015.9 tok/s) @ 585W
[2026-09-11T23:08:00]   decode-1: 742 tok in 5.0 s = 148.4 tok/s (pwr avg 359.5W)
[2026-09-11T23:08:10]   decode-16: 6537 tok in 5.0 s = 1307.4 tok/s (pwr avg 408.3W)
[2026-09-11T23:08:15] === sweep complete, restoring original cap ===
[2026-09-11T23:08:15] restored power cap to 500.0W (verified 500.0W)
[2026-09-11T23:08:15] Summary (cap W | prefill TTFT | prefill tok/s | d1 tok/s | d16 tok/s):
[2026-09-11T23:08:15]    585W |  4.41s |  9015.9 tok/s |   148.4 tok/s |  1307.4 tok/s
[2026-09-11T23:08:33] === sweep starting: caps=[585, 556, 528, 502, 476, 453, 430, 409, 388, 369, 350, 333, 316, 300, 285, 271, 257, 245, 232, 221, 210, 200]W original_cap=500.0W window=45.0s prefill_prompt_tokens=39760 server=http://127.0.0.1:8004 ===
[2026-09-11T23:08:33] NOTE: stop other GPU workloads (VS Code agents on :8004/:8006) for clean numbers.
[2026-09-11T23:08:33] --- [1/22] power cap 585W ---
[2026-09-11T23:08:33] sudo authenticated via SUDO_PASS env - continuing.
[2026-09-11T23:08:33]   cap set: 585W
[2026-09-11T23:08:33]   warmup: 16 tok @ current cap
[2026-09-11T23:08:43]   prefill: TTFT 4.48 s (8875.0 tok/s) @ 585W
[2026-09-11T23:09:33]   decode-1: 6310 tok in 45.0 s = 140.2 tok/s (pwr avg 439.8W)
[2026-09-11T23:10:23]   decode-16: 66014 tok in 45.0 s = 1467.0 tok/s (pwr avg 571.8W)
[2026-09-11T23:10:28] --- [2/22] power cap 556W ---
[2026-09-11T23:10:28]   cap set: 556W
[2026-09-11T23:10:28]   warmup: 16 tok @ current cap
[2026-09-11T23:10:38]   prefill: TTFT 4.72 s (8423.7 tok/s) @ 556W
[2026-09-11T23:11:28]   decode-1: 6175 tok in 45.0 s = 137.2 tok/s (pwr avg 466.5W)
[2026-09-11T23:12:18]   decode-16: 63104 tok in 45.0 s = 1402.3 tok/s (pwr avg 545.2W)
[2026-09-11T23:12:23] --- [3/22] power cap 528W ---
[2026-09-11T23:12:23]   cap set: 528W
[2026-09-11T23:12:23]   warmup: 16 tok @ current cap
[2026-09-11T23:12:33]   prefill: TTFT 4.89 s (8130.9 tok/s) @ 528W
[2026-09-11T23:13:23]   decode-1: 6238 tok in 45.0 s = 138.6 tok/s (pwr avg 471.3W)
[2026-09-11T23:14:14]   decode-16: 63428 tok in 45.0 s = 1409.5 tok/s (pwr avg 518.9W)
[2026-09-11T23:14:19] --- [4/22] power cap 502W ---
[2026-09-11T23:14:19]   cap set: 502W
[2026-09-11T23:14:19]   warmup: 16 tok @ current cap
[2026-09-11T23:14:29]   prefill: TTFT 5.06 s (7857.7 tok/s) @ 502W
[2026-09-11T23:15:19]   decode-1: 6511 tok in 45.0 s = 144.7 tok/s (pwr avg 472.3W)
[2026-09-11T23:16:09]   decode-16: 60820 tok in 45.0 s = 1351.6 tok/s (pwr avg 494.2W)
[2026-09-11T23:16:14] --- [5/22] power cap 476W ---
[2026-09-11T23:16:14]   cap set: 476W
[2026-09-11T23:16:14]   warmup: 16 tok @ current cap
[2026-09-11T23:16:25]   prefill: TTFT 5.25 s (7573.3 tok/s) @ 476W
[2026-09-11T23:17:11]   decode-1: 8192 tok in 41.54 s = 197.2 tok/s (pwr avg 466.5W)
[2026-09-11T23:18:01]   decode-16: 61149 tok in 45.0 s = 1358.9 tok/s (pwr avg 467.3W)
[2026-09-11T23:18:06] --- [6/22] power cap 453W ---
[2026-09-11T23:18:06]   cap set: 453W
[2026-09-11T23:18:06]   warmup: 16 tok @ current cap
[2026-09-11T23:18:17]   prefill: TTFT 5.49 s (7242.3 tok/s) @ 453W
[2026-09-11T23:19:07]   decode-1: 5764 tok in 45.0 s = 128.1 tok/s (pwr avg 446.0W)
[2026-09-11T23:19:57]   decode-16: 59441 tok in 45.0 s = 1320.9 tok/s (pwr avg 445.1W)
[2026-09-11T23:20:02] --- [7/22] power cap 430W ---
[2026-09-11T23:20:02]   cap set: 430W
[2026-09-11T23:20:02]   warmup: 16 tok @ current cap
[2026-09-11T23:20:13]   prefill: TTFT 5.74 s (6926.8 tok/s) @ 430W
[2026-09-11T23:21:03]   decode-1: 5909 tok in 45.0 s = 131.3 tok/s (pwr avg 422.2W)
[2026-09-11T23:21:53]   decode-16: 58277 tok in 45.0 s = 1295.0 tok/s (pwr avg 422.3W)
[2026-09-11T23:21:58] --- [8/22] power cap 409W ---
[2026-09-11T23:21:58]   cap set: 409W
[2026-09-11T23:21:59]   warmup: 16 tok @ current cap
[2026-09-11T23:22:09]   prefill: TTFT 5.79 s (6867.0 tok/s) @ 409W
[2026-09-11T23:22:59]   decode-1: 5997 tok in 45.0 s = 133.3 tok/s (pwr avg 402.0W)
[2026-09-11T23:23:50]   decode-16: 59340 tok in 45.0 s = 1318.7 tok/s (pwr avg 403.9W)
[2026-09-11T23:23:55] --- [9/22] power cap 388W ---
[2026-09-11T23:23:55]   cap set: 388W
[2026-09-11T23:23:55]   warmup: 16 tok @ current cap
[2026-09-11T23:24:06]   prefill: TTFT 6.01 s (6615.6 tok/s) @ 388W
[2026-09-11T23:24:56]   decode-1: 6122 tok in 45.0 s = 136.0 tok/s (pwr avg 380.7W)
[2026-09-11T23:25:46]   decode-16: 54930 tok in 45.0 s = 1220.7 tok/s (pwr avg 380.2W)
[2026-09-11T23:25:51] --- [10/22] power cap 369W ---
[2026-09-11T23:25:51]   cap set: 369W
[2026-09-11T23:25:51]   warmup: 16 tok @ current cap
[2026-09-11T23:26:03]   prefill: TTFT 6.28 s (6331.2 tok/s) @ 369W
[2026-09-11T23:26:53]   decode-1: 6294 tok in 45.0 s = 139.9 tok/s (pwr avg 363.4W)
[2026-09-11T23:27:43]   decode-16: 52720 tok in 45.0 s = 1171.6 tok/s (pwr avg 363.2W)
[2026-09-11T23:27:48] --- [11/22] power cap 350W ---
[2026-09-11T23:27:48]   cap set: 350W
[2026-09-11T23:27:48]   warmup: 16 tok @ current cap
[2026-09-11T23:28:00]   prefill: TTFT 6.66 s (5970.0 tok/s) @ 350W
[2026-09-11T23:28:50]   decode-1: 5919 tok in 45.0 s = 131.5 tok/s (pwr avg 344.8W)
[2026-09-11T23:29:40]   decode-16: 51865 tok in 45.0 s = 1152.6 tok/s (pwr avg 344.4W)
[2026-09-11T23:29:45] --- [12/22] power cap 333W ---
[2026-09-11T23:29:45]   cap set: 333W
[2026-09-11T23:29:45]   warmup: 16 tok @ current cap
[2026-09-11T23:29:57]   prefill: TTFT 6.79 s (5855.7 tok/s) @ 333W
[2026-09-11T23:30:47]   decode-1: 5593 tok in 45.0 s = 124.3 tok/s (pwr avg 326.9W)
[2026-09-11T23:31:37]   decode-16: 49561 tok in 45.0 s = 1101.4 tok/s (pwr avg 326.3W)
[2026-09-11T23:31:42] --- [13/22] power cap 316W ---
[2026-09-11T23:31:42]   cap set: 316W
[2026-09-11T23:31:42]   warmup: 16 tok @ current cap
[2026-09-11T23:31:54]   prefill: TTFT 7.08 s (5615.8 tok/s) @ 316W
[2026-09-11T23:32:44]   decode-1: 6344 tok in 45.0 s = 141.0 tok/s (pwr avg 310.2W)
[2026-09-11T23:33:35]   decode-16: 47805 tok in 45.0 s = 1062.3 tok/s (pwr avg 311.2W)
[2026-09-11T23:33:40] --- [14/22] power cap 300W ---
[2026-09-11T23:33:40]   cap set: 300W
[2026-09-11T23:33:40]   warmup: 16 tok @ current cap
[2026-09-11T23:33:52]   prefill: TTFT 7.44 s (5344.1 tok/s) @ 300W
[2026-09-11T23:34:42]   decode-1: 6364 tok in 45.0 s = 141.4 tok/s (pwr avg 295.6W)
[2026-09-11T23:35:33]   decode-16: 46077 tok in 45.0 s = 1023.9 tok/s (pwr avg 294.5W)
[2026-09-11T23:35:38] --- [15/22] power cap 285W ---
[2026-09-11T23:35:38]   cap set: 285W
[2026-09-11T23:35:38]   warmup: 16 tok @ current cap
[2026-09-11T23:35:51]   prefill: TTFT 7.76 s (5123.7 tok/s) @ 285W
[2026-09-11T23:36:10] interrupted — restoring cap
[2026-09-11T23:36:10] restored power cap to 500.0W (verified 500.0W)
