# Handoff — GPU power-cap sweep (585W → 200W) on the running 27B vLLM server

**Written:** 2026-09-11 ~22:50 IST, for a clean re-run by another agent.
**Goal:** measure how decode/prefill throughput on the local 27B degrades as the GPU power cap is lowered from 585W to 200W in 5% steps — single request and 16 concurrent requests — while the vLLM server keeps running (never restarted).

---

## 0. State as of handoff

- Experiment assets: **`/home/pctablet505/Projects/local_llm/power-cap-exp/`** (persistent).
  `/tmp/power-cap-exp/` is a STALE copy from the aborted attempt — ignore it.
- The full sweep was started once but **aborted: every row collected was contaminated** by
  concurrent coding-agent traffic on the same server (their 50k–200k-token prompts vs our
  39.76k-token prompt; all rows carry `idle_warn=1`). Archived, excluded from results:
  `power-cap-exp/results-contaminated-2026-09-11.csv`.
- **No clean data exists yet.** Power cap has been restored to 500W and verified.
- The pipeline itself is fully debugged and validated (cap setting, prefill TTFT, 45s
  decode windows, power sampling, live OBSERVATIONS.md). What failed was *environment
  discipline*, not the code: the server must be idle or the numbers are meaningless.

## 1. Owner rules (do not re-litigate)

1. **Keep the same vLLM server.** Do not restart, reload, or reconfigure anything on :8004.
2. **Never touch servedeck or its ports** (:8010 dashboard; it manages/adopts the server and
   can kill it on stop). Never touch :8000 (`ats-optimizer` — unrelated).
3. **:8006** is a reasoning-mirroring proxy → :8004 used by VS Code. It's a *consumer*, not
   the experiment target; experiment traffic goes straight to **:8004**.
4. The power cap must be **restored to 500W at the end** (the script does this in `finally`;
   `./restore.sh` is the manual fallback). The GPU's allowed range is 150–600W; 500W is the
   current/original setting, 600W is the factory default.

## 2. Environment facts (verified 2026-09-11)

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation, driver 595.84, ~97,887 MiB |
| Power limit range | min 150W / max 600W / current **500W** / default 600W |
| Model server | vLLM **0.29.0** (stock wheel), venv `~/Projects/local_llm/.venv-llm-029`, port **8004**, served name `Qwen3.8-27B-NVFP4`, `--max-num-seqs 16`, `--gpu-memory-utilization 0.95`, `--max-model-len 262144`, prefix caching ON, reasoning ON |
| KV pool | ~1.81M tokens (plenty for 16 × 40k contexts) |
| Metrics | `http://127.0.0.1:8004/metrics` (Prometheus; `vllm:num_requests_running/waiting`) |
| Dashboard | servedeck on :8010 (read-only for this experiment; its tok/s strip shows blended traffic — do not compare to our numbers) |
| Python for the script | `~/Projects/local_llm/.venv-llm-029/bin/python` (needs `httpx`; system python3 does not have it) |

## 3. The experiment protocol (implemented in `power-cap-exp/run_experiment.py`)

Caps: `585, 556, 528, 502, 476, 453, 430, 409, 388, 369, 350, 333, 316, 300, 285, 271,
257, 245, 232, 221, 210, 200` (585 × 0.95^n, floored at 200W). 22 caps.

For **each cap**, in order:
1. `sudo nvidia-smi -pl <cap>` (script authenticates itself via the `SUDO_PASS` env var —
   see §5 — and verifies the applied cap with `nvidia-smi --query-gpu=power.limit`).
2. **Warmup:** one 40k-prompt request, 8 output tokens (brings clocks to state; caches the prompt prefix).
3. **PREFILL:** one request with the 40k prompt **plus a unique cache-bust salt line**,
   `max_tokens=1`, streamed → **TTFT = cold prefill time** at that cap.
   (vLLM 0.29 has **no** HTTP `reset_prefix_cache` endpoint — the salt is how we get a
   guaranteed-cold prefill every cap; the salt changes block 0 of the prefix chain so all
   ~40k blocks miss.)
4. **DECODE-1:** 1 request, same 40k prompt (cached prefix), `temperature=0`,
   `ignore_eos=true`, `max_tokens=8192`, streamed → **45 s window starting at the first
   token**; tokens counted inside the window → `d1_tok_s`, `d1_tokens`.
5. Settle 5 s.
6. **DECODE-16:** 16 identical concurrent requests, same settings → one shared **45 s
   window starting at the first token of the batch**; total tokens in window →
   `d16_tok_s`, `d16_tokens`.
7. Settle 5 s. Row written to `results.csv` + `OBSERVATIONS.md`; next cap.

Controls making caps comparable: identical frozen prompt (`prompt40k.txt`, **39,760
prompt tokens**, do not regenerate/modify), temperature 0, top_p 1, 45 s windows, decode
windows start at first token (prefill excluded), power draw / util / temp sampled every
0.5 s throughout, external-load detector flags `idle_warn=1` rows.

**Live observation file: `power-cap-exp/OBSERVATIONS.md`** — protocol header, the 22-cap
table filling in live, and a log tail. Raw: `results.csv`. Full log: `experiment.log`.
State: `state.json` (records original cap 500W; needed by `restore.sh`).

## 4. THE IDLE GATE — this is what broke the first attempt

The first run was contaminated because coding agents were hammering the same server
(46 requests 50k–100k + 31 requests 100k–200k tokens in 22 min; our 16-request phases were
sharing 16 `max_num_seqs` slots with external requests). **Do not start, and ideally verify
between caps, that the server is idle:**

```bash
# GPU side: expect ~0% util, ~115W draw when idle
nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader

# Queue side: expect 0.0 / 0.0
curl -s http://127.0.0.1:8004/metrics | grep -E '^vllm:num_requests_(running|waiting)\{'
```

Rules:
- **Start only when both show idle for a few consecutive samples (≥3, spaced 2 s).**
- The script's `idle_warn` flag is a backstop, not a gate: if any completed row has
  `idle_warn=1`, treat it as contaminated — stop the run (SIGINT), move the bad cap out of
  `results.csv` (or `mv results.csv results-bad-<stamp>.csv` and let the resume skip the
  good ones), re-check idle, and resume. The script **resumes automatically**: caps already
  present in `results.csv` are skipped.
- If the machine can't be made idle (owner's agents actively using :8004/:8006), **wait or
  stop — never average over load.**

## 5. Running it

```bash
cd /home/pctablet505/Projects/local_llm/power-cap-exp
SUDO_PASS=REDACTED-ROTATE-THIS-PASSWORD /home/pctablet505/Projects/local_llm/.venv-llm-029/bin/python run_experiment.py
```

- Run it **in the foreground of a live TTY** (a real terminal, or a foreground TTY exec
  session in an agent CLI). ~2.1 min/cap → **~50–70 min total**.
- **`SUDO_PASS` is mandatory.** The owner's sudo password is `REDACTED-ROTATE-THIS-PASSWORD`. It is passed as an
  env var and used only by the script's own `sudo -S` call. Why: in agent sandbox
  environments each exec gets its own private `/run`, so sudo timestamps never carry
  across execs; the script must authenticate inside its own process tree. Without
  `SUDO_PASS` it falls back to waiting for a human to run `sudo -v` in a real terminal.
- Do **not** use `nohup ... &` from an agent exec session: backgrounded processes don't
  survive session teardown in this environment.
- `launch.sh` is the human-terminal variant (prompts for the password interactively, then
  backgrounds). Either path is fine; the foreground one is simplest to supervise.

Monitoring (from anywhere):
```bash
tail -f /home/pctablet505/Projects/local_llm/power-cap-exp/OBSERVATIONS.md
```

Abort cleanly (restores 500W cap in `finally`):
```bash
# in a real terminal: Ctrl+C in the running session.
# if the TTY won't deliver it (agent exec sessions), send it to the pid directly —
# this worked where TTY Ctrl+C did not:
kill -INT "$(pgrep -f 'run_experi[m]ent.py')"
```
Manual cap restore, ever: `./restore.sh` (prompts for sudo, or prefix `SUDO_PASS=REDACTED-ROTATE-THIS-PASSWORD

## 6. Known pitfalls (already hit & fixed — don't re-litigate)

- **vLLM 0.29 streaming:** request body must include `"stream": true` (otherwise you get a
  single non-streamed completion). Reasoning tokens arrive in **`delta.reasoning`** (not
  `reasoning_content`) — the script counts `content` + `reasoning` + `reasoning_content`.
- **No `/reset_prefix_cache` endpoint in 0.29** (offline `LLM` class only) → the salt-line
  trick in §3.
- `nvidia-smi -pl` prints a harmless "persistence mode disabled" warning — ignore it.
- **Never `pgrep -f`/`pkill -f` on plain patterns** — it matches the searching shell
  itself. Use bracketed patterns (`run_experi[m]ent`) or verify via `/proc/<pid>/cmdline`.
- The vLLM engine process renames itself (`VLLM::EngineCore`); the `/proc/<pid>/environ`
  trick is not reliable for it (irrelevant here, but don't chase it).
- Power cap is a **ceiling, not a target**: at 585–528W the card's natural draw is
  ~500–575W, so those caps are *not* constraining — expect flat numbers there. The
  informative region is where the cap meets or undercuts natural draw (roughly ≤575W for
  decode-16, lower for decode-1). 200W will be the steep part.
- GPU idles around 115W / 60–70°C on this box; under full load it hit ~499W / 86°C at the
  500W cap. No thermal-throttle surprises expected at lower caps.

## 7. Sanity ranges (from the contaminated first pass — directional only)

| Cap | prefill TTFT | decode-1 | decode-16 | (all idle_warn=1) |
|---|---|---|---|---|
| 585W | 4.70 s | 47.9 tok/s | 356.7 tok/s | yes |
| 556W | 4.79 s | 49.2 tok/s | 350.3 tok/s | yes |
| 528W | 4.98 s | 49.2 tok/s | 344.6 tok/s | yes |
| 502W | 5.33 s | 48.6 tok/s | 337.8 tok/s | yes |
| 476W | 5.29 s | 47.8 tok/s | 316.0 tok/s | yes |
| 453W | 5.76 s | 38.9 tok/s* | 318.3 tok/s | yes (*mid-window interrupt) |

A clean run should show: prefill TTFT rising smoothly as caps fall; decode-1 roughly flat
until cap ≈ natural single-stream draw, then falling; decode-16 falling earlier and
steeply (it's the phase that hits the power wall at ~575W). If a clean decode-1 at 585W
comes out far from ~50–60 tok/s, something else changed — stop and re-verify.

## 8. Reporting / done-when

1. All 22 rows in `results.csv` with `idle_warn=0` (any flagged row: re-run that cap after
   re-verifying idle).
2. Cap verified restored to 500W (`nvidia-smi --query-gpu=power.limit`) — the script logs
   "restored power cap" on exit; verify anyway.
3. Deliver to the owner: the final table (cap | prefill | decode-1 | decode-16 | avg/max
   power | temp), the tok/s-vs-cap curve summary (where does performance start falling,
   where does it cliff), and the files: `results.csv`, `OBSERVATIONS.md`,
   `experiment.log` (all in `~/Projects/local_llm/power-cap-exp/` — persistent, no /tmp).
