> **Provenance.** This is the implementation spec the `coldstart` fork was
> built to, preserved here because 130 comments in `servedeck/` cite it by
> section (`SPEC.md §6`, `SPEC.md correction C3`, …) and the repository did not
> contain the file those comments point at. It is the fork's text with
> `coldstart` renamed to `servedeck`; nothing has been rewritten to look
> better in hindsight.
>
> Where it names a path, a port, a GPU size or a model, that value is now
> **configuration** — see `docs/CONFIGURATION.md` and `servedeck.toml.example`.
> The spec's *reasoning* about those values still holds; its literals do not.

# Servedeck — implementation spec (authoritative)

> ## ⚠ CORRECTIONS — 2026-08-27 22:20, from a 26-agent adversarial verification.
> **These OVERRIDE anything later in this file. 12 of 20 load-bearing claims were refuted.**
>
> ### C1. Codex retry values were WRONG — do not use 12
> `stream_max_retries = 12` in §9(g) is **wrong and harmful**. A hung endpoint costs **~154 s per
> attempt**, and that per-attempt ceiling is **not settable by any key** (`stream_idle_timeout_ms` at
> 3 s and 600 s produced identical 155 s wall times). 12 retries = ~30 min of a frozen turn.
> **USE:**
> ```toml
> stream_max_retries    = 2      # VERIFIED to control the reconnect loop; default is 5
> request_max_retries   = 2      # key exists, but NO observable effect on connect-level failures
> stream_idle_timeout_ms = 900000 # key exists; effect on a HELD connection is UNVERIFIED
> ```
> All four keys work **only** inside `[model_providers.<name>]`. At top level they parse silently and
> do nothing. There is no global knob and no environment-variable override.
>
> ### C2. `/v1/models` must NEVER be held or faked — this would hang the CLI forever
> §7 says park all of `/v1/*`. **Wrong.** `is_server_up()` (codex-qwen.sh:262) probes `/v1/models` with
> `curl -s` and **no `-m` timeout**. Holding it hangs the CLI forever; faking a 200 makes the CLI believe
> a dead server is up, so `start_server()` returns "already up" and nothing ever restarts.
> **Per-path policy — replace §7's blanket rule:**
> | path | behaviour |
> |---|---|
> | `POST /v1/responses`, `POST /v1/chat/completions` | **hold** (both are first-class; the live log shows 8 chat/completions and zero /v1/responses) |
> | `GET /v1/models` | **pass through; 503 when down. NEVER hold, NEVER synthesize 200.** |
> | `GET /health` | **the real health gate** — returns 503 on `EngineDeadError` (health.py:22-32), which is the observed death mode. `/v1/models` is served from app state and stays **200 with a dead engine**. |
> | everything else | plain pass-through |
> Also: switch `is_server_up()` to probe `/health`, and give its curl a `-m` timeout.
>
> ### C3. Flash-Next CANNOT be auto-restarted unattended (highest-impact finding)
> `serve.sh:53-66` relaxes `kernel.yama.ptrace_scope` via **`sudo sysctl`**. `sudo -n true` fails and a
> systemd `ExecStart` has no tty. It only appears to work right now because `ptrace_scope` happens to be
> **0 on this boot** — **it will break after the next reboot.**
> Therefore: for `BACKEND=flashnext`, auto-restart must resolve to **`blocked-needs-human`** with a
> prompt for an interactive launch — NOT a silent retry loop. Implement the state, do not pretend.
>
> ### C4. `desired_state` must be enforced in ExecStart, not only in the supervisor
> `Restart=on-failure` does **not** work: vLLM **exits 0 when EngineCore dies**. Add
> `desired_state == stopped → exit 69` as **guard 0** in `qwen-server-run.sh`, reusing the existing
> `RestartPreventExitStatus=69` contract. That makes intent supervisor-independent — the watchdog, a boot
> start, a stray `systemctl --user start`, and a GUI bug all become harmless stand-downs.
> Store `desired_state` in `run/desired_state` on ext4 (survives reboot). **Not** `/run/user/1000` (tmpfs),
> **not** systemd `is-enabled`. `actual_state` is **computed, never stored** — storing it is precisely what
> lets a restart resurrect a stale belief.
>
> ### C5. Cross-kill risk — TESTED AND REFUTED for this topology (do not chase it)
> The verifier warned `find_orphaned_engine_cores()` would SIGKILL a wrapper-launched EngineCore because
> "legitimate" is defined as *parent cmdline contains `vllm serve`*. **Tested against the live tree: it
> does NOT fire.** `serve.sh` execs the real binary, so EngineCore's direct parent is
> `.../.venv-next/bin/python .../bin/vllm serve RadixArk/...` — the literal `vllm serve` is present and
> the process is correctly classified legitimate. Servedeck launching via `serve.sh`/`qwen-server-run.sh`
> preserves that topology, so it is safe.
> Keep one guard anyway: if a future launcher ever interposes a process between the wrapper and `vllm`,
> the rule silently starts killing live engines. Assert the ownership check in a test.
> Separately: the bash `find_orphaned_engine_cores()` legitimately uses `pgrep -f` because the kernel
> truncates `comm` at 15 chars and `VLLM::EngineCore` is 16. **Servedeck's Python port must instead use
> `comm.startswith("VLLM::")`** — same rule, no self-match.

> ### C6. Blue-green FP8 does NOT fit — §3's table is wrong for FP8
> Those repos declare no `kv_cache_scheme`, so KV is bf16 (~72 KiB/token). Two full-context FP8 instances
> need ~0.55 util per side. **Only 27B-NVFP4 (2×0.47, the shipped default) and AWQ are safe.**
> Flash-Next remains impossible. Recompute blue_green() with per-model KV, not weights alone.
>
> ### C7. Client-disconnect propagation is mandatory, not optional
> The server runs `--max-num-seqs 1`. One un-propagated disconnect holds the only sequence slot and blocks
> everything — worse than the outage being hidden. vLLM's `/v1/responses` is wrapped in
> `with_cancellation`, so closing the upstream socket does abort generation. Propagate it.
>
> ### C8. Hold ceiling must be finite and boot failures must NOT be held
> 44 of 72 recorded exits **never reached serving**. A hold with no ceiling turns a deterministic config
> error into a silent multi-minute freeze. Ceiling = **2× measured cold boot**, then a real 503 whose body
> names the reason. Never hold when `reached_ready` was false.
>
> ### C9. Fix the boot path — it is the dominant failure class
> Of 72 exits: 44 boot failures (17 DNS, 12 `FATAL: vLLM venv not found` at a stale
> `AlgoTrading-llm-audit/.venv-llm` path, 9 guard-69 stand-downs, 6 NVML), 19 deliberate stops, 9 crashes.
> Add `HF_HUB_OFFLINE=1` to the unit `Environment` — verified to remove the DNS class. Note
> `After=/Wants=network-online.target` is **inert** for a *user* unit.
>
> ### C10. Deaths of the CURRENT server are recorded nowhere
> `qwen_deaths.log` only captures exits of `qwen-vllm.service`, which is disabled/inactive. Flash-Next runs
> outside systemd via `serve.sh`, so its deaths are unrecorded — the crash population is partly unmeasured.
> Kernel Xid history is volatile (`/var/log/journal` empty); read the persisted record, not `journalctl`.
>
> ### C11. A model swap changes wire identity
> Codex replays the old model name from its own session state on `resume --last` — this already caused a
> live 404 storm. Every backend must be served under the **same `--served-model-name`** for a swap to be
> transparent.
>
> ### Watchdog corrections
> Drop `inactive` from the watchdog's actionable set (`qwen-vllm-watchdog.sh:47-50`) — that is exactly what
> a deliberate stop leaves, and it resurrected the unit at 20:13:58 today. Stop calling `reset-failed`
> (`:60`), which erases the `StartLimitBurst` backstop.


Local web GUI managing a vLLM server on one RTX PRO 6000 (97,887 MiB).
Design prototype (visual/UX reference, NOT logic reference): a single-file
HTML mock, kept outside the repository. `servedeck/web/` is what shipped.

## 0. GROUND TRUTH — measured, do not substitute

| model | backend | weights GiB | KV KiB/tok | ctx | trust |
|---|---|---|---|---|---|
| RadixArk/Qwen3.8-Flash-Next-NVFP4 | flashnext | 78.47 | 30.39 @262144 | 262144 | measured |
| RadixArk/Qwen3.8-Flash-Next-NVFP4 | flashnext | 78.47 | 33.85 @131072 | — | measured |
| RadixArk/Qwen3.8-27B-NVFP4 | inline | 20.75 | 37.99 @262144 | 262144 | measured |
| Qwen/Qwen3.8-27B-FP8 | inline | 28.51 | 37.99 (family est) | 262144 | weights measured |
| twolven/Qwen3.8-27B-abliterated-AWQ-MTP | inline | 18.21 (est) | ~35.4 (est) | 262144 | estimated |
| orcarouter/Qwen3.8-27B-Uncensored-FP8 | inline | 28.75 (est) | 37.99 (est) | 262144 | estimated |
| OBLITERATUS/Qwen3.8-27B-OBLITERATED | — | — | — | — | **UNSERVABLE** GGUF-only |

KV rate is NOT context-invariant (30.39 @262k vs 33.85 @131k) — vLLM block sizing depends on max_model_len.
tok/s: Flash-Next 99.3 (measured), 27B NVFP4 140.4, AWQ-MTP 173.8, FP8 = null (unverified, do NOT fabricate).

## 1. ARCHITECTURE

Python 3.13 + FastAPI + uvicorn + httpx in a THIRD venv `~/Projects/servedeck/.venv`.
NEVER install into `.venv-llm` or `.venv-next` (documented foot-gun: SETUP.md:229 — uv re-resolves and
downgrades pinned CUDA 13.2.86, reintroducing the PTX blocker).

Deps pinned: fastapi==0.136.3, uvicorn==0.52.4, httpx==0.28.* (wheels already in uv cache).
Runtime is network-free: 127.0.0.1 only, no CDN, no external fonts.

Process model: uvicorn on 127.0.0.1:8010 serves GUI + /api/* + gateway proxy to upstream vLLM (8001 flashnext / 8000 inline).
vLLM launched with `start_new_session=True` so it gets its OWN session+PGID and survives Servedeck restarting.
Servedeck re-adopts on startup from state/server.json + port probe.

### Delegation, not reimplementation
Servedeck NEVER builds a `vllm serve` command line. It invokes existing launchers:
- flashnext: `setsid env PORT= MAX_LEN= GPU_UTIL= MAX_SEQS= KV_DTYPE=auto SERVED_NAME= ~/Projects/vllm-qwen38next/serve.sh`
  (serve.sh ALREADY reads PORT/MAX_LEN/GPU_UTIL/MAX_SEQS/KV_DTYPE from env — only SERVED_NAME needs adding)
- inline: `setsid ~/Projects/local_llm/bin/qwen-server-run.sh` (sources .config)

`.config` is written ONLY by shelling out to `codex-qwen.sh set-mem|set-subagents|set-config`.
Never edit .config directly (preserves save_config_kv line semantics + validators).
HARD RULE: `set_mem()` auto-restarts when server is up. Servedeck therefore only calls it while DOWN.
Restart sequence is always: stop -> write config -> start.

## 2. PROCESS CONTROL — the self-match trap

SETUP.md:401 — `pgrep -f`/`pkill -f` match the calling shell. This killed shells repeatedly during development.

Rules in `servedeck/procctl.py`:
1. NEVER pgrep -f / pkill -f / pkill / killall. A test greps the package and fails on any hit.
2. Popen(..., start_new_session=True); record pgid = os.getpgid(pid); persist.
3. Stop = os.killpg(pgid, SIGTERM) -> SIGKILL after timeout (60s). BEFORE signalling assert
   pgid != os.getpgid(0) and pgid != os.getpid(); refuse otherwise.
4. Discovery of a server we did not launch: `ss -H -ltnp "sport = :PORT"` -> parse pid=(\d+);
   cross-check /proc/<pid>/comm == "vllm" and exe/cwd resolving inside .venv-next or .venv-llm.
   If port answers but no PID attributable -> actual_state = UNMANAGED, Stop/Restart disabled, explained.
5. Orphan sweep uses `comm`, NOT cmdline. Kernel truncates comm at 15 chars and "VLLM::EngineCore" is 16,
   so the safe predicate is comm.startswith("VLLM::"). Orphan iff /proc/<ppid>/cmdline lacks "vllm serve".

Signatures:
  scan_vllm_processes() -> list[VllmProc]
  find_orphaned_engine_cores() -> list[int]
  listener_pid(port) -> int | None
  launch(argv, env, cwd, log_path) -> ServerHandle
  stop(handle, timeout_s=60, escalate=True) -> StopResult

## 3. CAPACITY MODULE (servedeck/capacity.py) — pure, no I/O, ONE implementation

Constants:
  GPU_TOTAL_MIB=97887; GPU_TOTAL_GIB=97887/1024 (95.5928)
  OVERHEAD_GIB_DEFAULT=4.7 (measured: 4.47 flashnext, 4.33 27B — 4.7 is conservative)
  FRAG_MARGIN_GIB=1.0; UTIL_THIN_MARGIN=0.97
  VRAM_GUARD_HEADROOM_MIB=4096 (== qwen-server-run.sh:70)
  MAMBA_MAX_NUM_SEQS_INLINE=128 (codex-qwen.sh:445 — >128 fails CUDA graph capture)
  FLASHNEXT_MIN_UTIL=0.90

Formulas:
  budget_gib     = util * GPU_TOTAL_GIB
  kv_gib         = budget_gib - weights_gib - overhead_gib
  kv_tokens      = floor(kv_gib * 1048576 / kv_kib_per_token)  if kv_gib>0 else 0
  concurrency_x  = kv_tokens / ctx
  agents_at_ctx  = floor(concurrency_x)
  max_single_ctx = min(model_max_ctx, kv_tokens)
  effective_parallel = min(agents_at_ctx, max_num_seqs)

ACCEPTANCE TEST (tests/test_capacity.py) — must reproduce both real boots to <0.2%:
  Flash-Next util 0.96, w 78.47, ovh 4.47 -> kv 8.83 GiB, 304,655 tok (vLLM printed 304,653), 1.16x
  27B NVFP4  util 0.50, w 20.75, ovh 4.33 -> kv 22.72 GiB, 627,113 tok (vLLM printed 627,117), 2.39x

Dataclasses: ModelInputs, Finding(code, level: "block"|"warn", title, detail, fix, fix_action),
CapacityResult(budget_gib, kv_gib, kv_tokens, concurrency_x, agents_at_ctx, effective_parallel,
max_single_ctx, confidence, bar{weights_pct,kv_pct,overhead_pct,free_pct}, findings, can_apply)

  compute(m, *, util, ctx, max_num_seqs, live=None) -> CapacityResult
  blue_green(a, b, *, util_a, util_b, ctx) -> BlueGreenVerdict
    fits = (w_a + w_b + 2*overhead + FRAG_MARGIN) <= GPU_TOTAL_GIB
    Flash-Next: 2*78.47+9.4 = 166.3 > 95.6 -> impossible
    27B NVFP4:  2*20.75+9.4 =  50.9 -> feasible, ~44.7 GiB KV
    27B FP8:    2*28.51+9.4 =  66.4 -> feasible
    AWQ-MTP:    2*18.21+9.4 =  45.8 -> feasible

BLOCKING findings: MODEL_UNSERVABLE, UNKNOWN_CAPACITY (weights_source unknown — refuse the
safetensors*1.01 estimator for model_type=="qwen4_exp": 125.91 GiB disk vs 78.47 VRAM, 37% error),
UTIL_OUT_OF_RANGE, MAX_NUM_SEQS_INVALID, CTX_ABOVE_CEILING, WEIGHTS_EXCEED_BUDGET (kv_gib<=0,
offer fix_action util_min), KV_TOO_SMALL_FOR_ONE_CTX (offer ctx fix), FLASHNEXT_UTIL_TOO_LOW (<0.90),
NOT_ENOUGH_FREE_VRAM (same arithmetic as qwen-server-run.sh:122, discount our own processes),
TRAINING_MARKER (3 paths per qwen-server-run.sh:75-79), PTRACE_BLOCKS_PLE (flashnext + scope!=0;
NEVER run sudo — show copyable command), GPU_UNRESPONSIVE (nvidia-smi -L fails).

WARN findings: THIN_MARGIN(util>=0.97), ESTIMATED_ONLY ("~25% optimistic historically"),
MEASURED_OTHER_CTX (show both rates), SEQS_BELOW_AGENTS, SUBAGENTS_NOT_A_GUARANTEE (REQUIRED label:
"CODEX_MAX_SUBAGENTS={c} is a Codex-side orchestration cap, not a GPU guarantee. This configuration
serves {e} in parallel; the other {c-e} will queue."), MAMBA_SEQS_CAP(inline & seqs>128),
PTRACE_LEFT_RELAXED (scope==0 while READY/STOPPED — persistent amber + re-harden command).

THE BROWSER NEVER COMPUTES CAPACITY. Delete the prototype's JS capacity(). All derived numbers come
from POST /api/capacity/estimate (debounce slider 120ms; <2ms over loopback).

## 4. REGISTRY (servedeck/registry.py)

Scan ~/.cache/huggingface/hub/models--*/. repo_id = dirname after "models--", "--"->"/".
snapshot = refs/main content else newest snapshots/ dir. Skip if no snapshots/ (excludes the 12KB stub
models--Qwen--Qwen3.8-27B). safetensors_gib = sum st_size(follow_symlinks=True) of *.safetensors.
Parse config.json -> architectures[0], model_type, and from text_config (fallback root):
max_position_embeddings, num_hidden_layers, num_key_value_heads, head_dim, full_attention_interval,
quantization_config.quant_algo||quant_method.

servable = config.json exists AND safetensors>0 AND architectures[0] in KNOWN_ARCHS
KNOWN_ARCHS = {"Qwen3_5ForConditionalGeneration":"inline", "Qwen4ExpForConditionalGeneration":"flashnext"}
(this map also selects venv/launcher/backend)

state/measurements.json = APPEND-ONLY list of observations (never one scalar per model), written after
every boot reaching READY (and after a boot failing at KV, which still yields weights):
  {ts, repo_id, backend, inputs{util,max_model_len,max_num_seqs,kv_cache_dtype},
   measured{weights_gib,kv_gib,kv_tokens,concurrency_x,kv_kib_per_token,overhead_gib,gpu_used_mib},
   timing{cold,total_s,phases{}}, throughput{decode_tok_s,samples,source}, source, trust}
Derived, not stored twice: kv_kib_per_token = kv_gib*1048576/kv_tokens

resolve_inputs(repo_id, util, ctx) lookup order:
 1. exact (repo_id, backend, max_model_len==ctx) -> trust="measured"
 2. same repo_id any ctx -> trust="measured_other_ctx" (render "measured*")
 3. estimate: weights_gib = safetensors_gib*1.01 (validated 20.42->20.75 +1.6%, 28.75->28.51 -0.8%)
    BUT refuse for model_type=="qwen4_exp" -> weights_source="unknown" -> UNKNOWN_CAPACITY blocker.
    kv rate = median of measured observations sharing architectures[0] (currently 37.99 for Qwen3_5*).

## 5. PHASE DETECTION (servedeck/phases.py)

Unanchored re.search. Wildcard file:line as \w+\.py:\d+ (differs between vLLM 0.27.1 and 0.29.0.dev0).
Timing uses WALL-CLOCK arrival at the tailer, not log timestamps (format "INFO 08-27 21:22:13" has no
year; uvicorn's "INFO:     Application startup complete." has no timestamp).

INIT             \[core\.py:\d+\]\s+Initializing a V1 LLM engine
LOADING_WEIGHTS  \[(?:gpu_)?model_runner\.py:\d+\]\s+Starting to load model\s+(\S+)
  sub-progress   Loading safetensors checkpoint shards:\s+(\d+)%\s+Completed\s+\|\s+(\d+)/(\d+)
  MEASUREMENT    \[(?:gpu_)?model_runner\.py:\d+\]\s+Model loading took\s+([\d.]+)\s+GiB memory and\s+([\d.]+)\s+seconds
COMPILING        \[backends\.py:\d+\]\s+Using cache directory:\s+(\S+)\s+for vLLM's torch\.compile
  exit           \[monitor\.py:\d+\]\s+torch\.compile took\s+([\d.]+)\s+s in total   (fires TWICE:
                 backbone then eagle_head for MTP — count both, do not reset the phase)
KV_CACHE         \[gpu_worker\.py:\d+\]\s+Available KV cache memory:\s+([\d.]+)\s+GiB
  tokens         GPU KV cache size:\s+([\d,]+)\s+tokens
                 (Flash-Next puts both on ONE line kv_cache_utils.py:2258; the 27B SPLITS them across
                  :2235/:2236 — use two independent searches to cover both layouts)
  concurrency    Maximum concurrency for\s+([\d,]+)\s+tokens per request:\s+([\d.]+)x
CUDA_GRAPHS      Capturing CUDA graphs\s+\(([^)]+)\):   (tqdm \r-joins updates into one line — take the
                 LAST (\d+)%\| on the line)
HTTP_START       \[api_server\.py:\d+\]\s+Starting vLLM server on\s+http://([\d.]+):(\d+)
READY            "INFO:\s+Application startup complete\." AND GET /v1/models -> 200 (BOTH required;
                 the HTTP probe is authoritative, same criterion as is_server_up())

Prototype's 5 labels map: Stopping=0-1, Loading weights=2-4, Compiling=5, KV cache=6, Ready=7-9.

classify(line, reached_ready) -> Failure | None:
  KV_TOO_SMALL      ValueError: To serve at least one request with the model's max seq len \((\d+)\), \(([\d.]+) GiB KV cache is needed, which is larger than the available KV cache memory \(([\d.]+) GiB\)\. Based on the available memory, the estimated maximum model length is (\d+)\.
                    -> NO auto-restart; fix_action set ctx = group(4)
  PLE_FP8_PATCH_MISSING  no module or parameter named 'ngram_embedding\.weight_scale'   (Blocker 1)
  PTRACE_DENIED     pidfd_getfd: Operation not permitted                                (Blocker 2)
  CUDA_SYMLINKS     Could NOT find CUDA_CUDART_LIBRARY                                  (Blocker 3)
  CUDA_TOOLCHAIN    the provided PTX was compiled with an unsupported toolchain         (Blocker 4)
  FLASHINFER_LINK   Ninja build failed | cannot find -l(cudart|nvrtc|nvvm)              (Blocker 5)
                    -> hint: grep -nE "FAILED:|cannot find -l" serve.log
  STARTUP_OOM       Free memory .* is less than desired GPU memory utilization -> offer orphan sweep
  ENGINE_INIT_FAILED RuntimeError: Engine core initialization failed -> attach preceding 40 lines
  CUDA_FAULT        torch\.AcceleratorError: CUDA error: (misaligned address|an illegal memory access)
                    -> auto-restart ONLY if reached_ready
  RUNTIME_OOM       CUDA out of memory -> auto-restart only if reached_ready
  INFORMATIONAL     shm_broadcast: No available shared memory broadcast block found in 60 seconds
                    -> NEVER shown as an error (SETUP.md:407)

Log severity: \sERROR\s | Traceback \(most recent call last\) | ^\s*\w*(Error|Exception): -> .e
              \sWARNING\s -> .w ; phase/measurement matches -> .g ; else default

## 6. SUPERVISOR (servedeck/supervisor.py) — TOP PRIORITY

desired_state: "STOPPED"|"RUNNING"  (persisted, changed ONLY by explicit user action)
actual_state:  "STOPPED"|"PREFLIGHT"|"STARTING"|"READY"|"DRAINING"|"STOPPING"|"FAILED"|"UNMANAGED"

state/desired.json (atomic: temp + os.replace):
 {version,desired_state,repo_id,backend,served_name,port,util,max_model_len,max_num_seqs,
  auto_restart,suspended,suspended_reason,attempts[],updated_at}

Startup reconciliation:
  RUNNING + port answers + PID attributable -> adopt READY
  RUNNING + port answers + no PID           -> UNMANAGED (Stop/Restart disabled, explained)
  RUNNING + nothing                          -> preflight -> start
  STOPPED + port answers                     -> DO NOTHING. Banner offering Adopt or Stop. Never auto-kill.
  STOPPED + nothing                          -> STOPPED

### THE RULE THAT FIXES THE KNOWN WATCHDOG DEFECT
SETUP.md:393 — a watchdog cannot tell "deliberately stopped" from "crashed" and previously resurrected
a server the user meant to stop. THEREFORE: auto-restart is gated on desired_state=="RUNNING", full stop.
Stop sets desired_state="STOPPED" BEFORE signalling, so the ensuing exit is intent, not a crash
(recorded outcome:"stopped_by_user", triggers nothing). setup.sh must REFUSE to enable
qwen-vllm-watchdog.timer.

### CRASH-WHILE-SERVING vs FAILED-BOOT (essential)
Per-run flag reached_ready set the instant phase READY completes.
  if not reached_ready:  actual=FAILED; auto_restart=NO   # a failed boot fails identically forever
                         surface the matched error line + blocker number + fix_action
  else:
     if failure in {XID_79, XID_154} or not gpu_alive(): FAILED, NO restart (reboot required)
     elif backoff exhausted: FAILED, suspended=True
     else: schedule_restart(BACKOFF[attempt])
Rationale: six consecutive failed boots on 2026-08-27 each had a DIFFERENT root cause; blind restarting
would have hidden every one.

Xid classification reuses bin/qwen-server-record-death.sh:54-69 — 13/31 MMU fault (restartable);
79 off-bus (no restart); 154 reboot-required (no restart); else "not a code this script recognizes;
do not assume" (no restart, human ack). `nvidia-smi -L` failing => never restart.

BACKOFF=[15,30,60,120,240]s, window 1800s, max 5 attempts. Exceeded -> FAILED + suspended=true,
desired_state STAYS RUNNING (never silently rewrite intent). Red card requires explicit click:
"Crash loop — auto-restart suspended. 5 restarts in 28 min. Last: <error>." [Show all 5]
[Resume auto-restart] [Stop]. Ack persisted in state/ack.json so a reload does not clear it.

On EVERY exit: append state/history.jsonl AND exec bin/qwen-server-record-death.sh with
SERVICE_RESULT/EXIT_CODE/EXIT_STATUS set, so `./codex-qwen.sh deaths` keeps working (one death record).

Restart modes: immediate (default) | drain (poll vllm:num_requests_running until 0 or 120s) |
blue_green (v2, only when capacity.blue_green().feasible).
Single path: stage -> preflight -> [drain] -> stop -> shellconfig.set_*() -> start -> tail -> READY|FAILED.
CONFIG IS NEVER MUTATED ON A LIVE SERVER.

## 7. GATEWAY (servedeck/gateway.py)

Codex targets http://127.0.0.1:8010/v1. Upstream stays 8001/8000. Bind 127.0.0.1 only.
Proxy prefixes: /v1/*, /health, /ping, /metrics, /tokenize, /detokenize, /invocations, /generative_scoring.

NO SSE keep-alive trick, NO fake-200. Never write a byte until an upstream response exists.
  READY                              -> transparent stream proxy via httpx.AsyncClient.stream +
                                        StreamingResponse; body NOT buffered (TCP backpressure)
  STARTING/PREFLIGHT/DRAINING/STOPPING and desired==RUNNING
                                     -> PARK: await wait_for(ready_event.wait(), hold_max_s=240)
                                        BEFORE reading the body. On timeout -> 503 + Retry-After
  desired==STOPPED                   -> immediate 503, Retry-After 5. Never park; nothing is coming.
  FAILED                             -> immediate 503 with classified code. Never park.
  UNMANAGED                          -> proxy normally
max_parked=64 -> beyond that immediate 503 + Retry-After.

Error body (OpenAI-compatible):
 {"error":{"message":"Servedeck: backend restarting — phase 'Loading weights', ~3m10s remaining",
  "type":"coldstart_upstream_unavailable","code":"restarting",
  "servedeck":{"phase":"loading_weights","eta_s":190,"parked":3,"attempt":1}}}

REPLAY SAFETY — THE ONE HARD RULE: once ANY upstream byte has been forwarded, NEVER retry internally.
Propagate the disconnect; Codex's stream_max_retries re-issues the whole turn. Concatenating a fresh
generation onto a partial one produces duplicated incoherent output. Parked-but-never-issued requests
are not "replayed" — they simply proceed.

## 8. API

SSE: GET /api/events (one stream; EventSource auto-reconnects with Last-Event-ID).
events: state (on change) | phase (transition + 1s while STARTING) | telemetry (2s) |
        log (<=250ms, batched <=20) | gateway (1s while parked>0) | notice
telemetry pulls http://127.0.0.1:{port}/metrics — CONFIRMED PRESENT:
  vllm:kv_cache_usage_perc, vllm:num_requests_running, vllm:num_requests_waiting,
  vllm:num_requests_waiting_by_reason{reason="capacity"|"deferred"}, vllm:num_preemptions_total,
  vllm:request_prompt_tokens_sum/_count, vllm:request_generation_tokens_sum/_count,
  vllm:prompt_tokens_total, vllm:generation_tokens_total, vllm:prefix_cache_hits_total/queries_total
gen_tok_s = delta(vllm:generation_tokens_total)/dt. Fallback log regex:
  Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%

REST: GET /api/state, /api/health, /api/models, /api/history?limit, /api/preflight, /api/log?lines
POST /api/capacity/estimate {repo_id,util,ctx,max_num_seqs} -> CapacityResult
POST /api/config/stage (MEMORY ONLY) | /api/config/revert
POST /api/server/start | stop | restart{mode} | apply{mode} | adopt | resume | sweep-orphans
POST /api/codex/subagents {n} | /api/smoke
All mutating endpoints return 202 immediately; progress on SSE. NO endpoint blocks for a boot.

### LIVE-METRICS AGENT SIZING (user requirement)
Show TWO numbers side by side, never one:
  "Safe floor"       = agents at configured max ctx        (a guarantee)
  "At observed avg"  = kv_total_tokens / avg_prompt_tokens (a BET — style it differently)
avg_prompt_tokens = vllm:request_prompt_tokens_sum / _count, attributable to a window
(5m / 1h / since restart) and resettable. Show a sparkline of kv_cache_usage_perc + avg over time —
a single instantaneous average is not a safe basis for setting agent count.
OVERSUBSCRIPTION SIGNALS (empirical, beat any estimate):
  num_preemptions_total rising => too many agents (real work being thrown away)
  num_requests_waiting_by_reason{reason="capacity"} > 0 => KV-bound now
  kv_cache_usage_perc sustained near 1.0 => at the edge
Closing the loop: subagent field pre-filled with the observed-average recommendation, writes
max_concurrent_threads_per_session via codex-qwen.sh set-subagents, labelled "applies on NEXT Codex
launch, not the running session" and "Codex-side cap, not a GPU guarantee".
vLLM metrics RESET on restart — persist history in the backend for trend lines.

### SMOKE TEST — runs THROUGH the gateway
1. tool call: one function get_time(timezone), tool_choice auto, temperature 0. Pass iff HTTP 200 and
   choices[0].message.tool_calls[0].function.name=="get_time" and arguments parses as JSON.
2. plain: "Reply with exactly: OK". Pass iff content OR reasoning_content non-empty, and REPORT WHICH
   FIELD — the qwen3 reasoning parser routes output to reasoning_content and an empty content "looks
   exactly like a model failure" (SETUP.md:277).
Failures must show the actual HTTP status AND body.

### WAIT-TIME UX (top priority — the delay is accepted, so make it legible)
- 5-phase bar from the prototype, driven by real phases, with the note line per phase.
- ETA from history: median and p90 of total_s and per-phase over prior runs matching (repo_id, backend,
  cold). Render "typically 4m10s · cold boot 9m50s". NEVER a fake percentage. cold = no prior successful
  boot for (repo_id, backend). With <2 samples cite SETUP.md figures WITH attribution.
  Calibration: Flash-Next warm boot 122s (21:22:11->21:24:13), 27B ~145s.
- OVERRUN HONESTY: when elapsed > p90 for the phase, amber label "Loading weights — 7m12s, usually 3m04s".
  Never a silent spinner.
- Queue visibility: "3 agent requests held · oldest 2m14s" from the gateway event.
- Notify when ready: Notification.requestPermission() INSIDE the click handler (required user gesture).
  Fallback always on: while document.hidden alternate document.title every 1s; stop on visibilitychange.
  Optional WebAudio 2-tone beep behind a localStorage toggle, DEFAULT OFF. No assets, no CDN.
- One-click restart: POST /api/server/restart with empty body. No form, no confirm.
- Every control that is not semantically invalid stays interactive during a boot.

### HONEST LIMITS — rendered in the UI, not buried
1. Prefix cache is lost on every restart: measured 0.53s -> 4.16s for a 30k shared prefix; 27.1s for a
   253k prefill. Banner for 60s after READY.
2. A cold Flash-Next boot is 4-10 min and nothing hides that.
3. CODEX_MAX_SUBAGENTS is a Codex-side cap, not a GPU guarantee.
4. Blue-green impossible for Flash-Next: 2*78.5+9.4 = 166.3 vs 95.6 GiB.
5. If Servedeck dies the gateway dies (~1s RestartSec window).
6. Servedeck NEVER runs sudo. ptrace is a copyable manual one-liner.
7. Estimates for never-booted models are upper bounds (~25% optimistic historically on this box).

## 9. SHELL PATCHES (backup each as .bak-precoldstart first; all additive and guarded)

codex-qwen.sh:
 a. recompute_derived() { BASE_URL="${COLDSTART_URL:+$COLDSTART_URL/v1}";
    [ -z "$BASE_URL" ] && BASE_URL="http://localhost:${PORT}/v1";
    PID_FILE="$RUN_DIR/qwen_server.pid"; LOG_FILE="$LOG_DIR/qwen_server.log"; }
    call at end of load_config() and once at top level. (Today BASE_URL freezes before load_config runs,
    so .config's PORT never takes effect.)
 b. recognise .config keys: BACKEND, MODEL, MODEL_REPO, SERVED_NAME, PORT, MAX_MODEL_LEN,
    MAX_NUM_SEQS, COLDSTART_URL, USE_COLDSTART
 c. add `set-config <KEY> <VALUE>` -> save_config_kv with an allow-list
 d. ALREADY FIXED (2026-08-27): use_systemd() gates the systemd paths on BACKEND != flashnext.
    Do NOT undo. Verify it is still present.
 e. start_server(): new FIRST branch — if USE_COLDSTART != 0 and
    `curl -sf -m 2 "$COLDSTART_URL/api/health"` succeeds, POST /api/server/start then poll /api/state
    printing phase names until READY|FAILED; return.
 f. stop_server(): same guard -> POST /api/server/stop
 g. write_provider_config(): add the VERIFIED keys (present in codex-cli 0.147.0 ModelProviderInfo):
      request_max_retries = 6
      stream_max_retries = 12
      stream_idle_timeout_ms = 900000
    Keep wire_api = "responses" (the binary contains: `wire_api = "chat"` is no longer supported).
    DO NOT repoint base_url to the gateway yet — that is a separate, user-gated step.

bin/qwen-server-run.sh: MODEL="${MODEL_REPO:-RadixArk/Qwen3.8-27B-NVFP4}"; add
 --served-model-name "${SERVED_NAME:-$MODEL}", --max-model-len "${MAX_MODEL_LEN:-262144}",
 --max-num-seqs "${MAX_NUM_SEQS:-128}" to the exec; add early
 `[ "${1:-}" = "--preflight-only" ] && exit 0` right after the orphan sweep.

vllm-qwen38next/serve.sh: ONE token — --served-model-name "${SERVED_NAME:-qwen38-flash-next}".

## 10. NON-GOALS (v1)
blue-green execution; ANY sudo; model download/delete/convert; GGUF; remote access/auth/TLS/CORS;
multi-GPU; arbitrary vLLM flag editing (only util, max-model-len, max-num-seqs); Codex session
management; benchmarking; replacing codex-qwen.sh or the systemd unit; mobile/i18n/packaging;
installing anything into .venv-llm or .venv-next.

## 11. UNVERIFIED ASSUMPTIONS (flag, do not paper over)
1. Whether Codex's request_max_retries covers HTTP 503 vs only transport errors; the backoff schedule;
   any clamp on stream_idle_timeout_ms; whether a pre-first-byte connect timeout is separately
   configurable (no such key found — which is why the gateway hold is bounded at 240s rather than
   relying on the client).
2. max_concurrent_threads_per_session table never fully confirmed (LOCAL_LLM_SETUP.md:554).
3. 27B FP8 101.7 tok/s appears in no log — seed null.
4. AWQ-MTP 18.21 GiB is a safetensors sum, not a boot measurement.
