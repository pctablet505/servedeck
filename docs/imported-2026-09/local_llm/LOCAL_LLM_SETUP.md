> **Background document.** The day-to-day entry point is now `./llm` — see
> [README.md](README.md) for how to start, stop and wire clients. This file is the
> long-form history: supervision, systemd units, and how earlier numbers were
> measured. Parts of it predate the current GLM-5.3 setup and describe the Qwen
> Flash-Next era; treat specific commands here as historical unless README agrees.

# Local LLM Setup — Qwen3.8 (27B and Flash-Next)

> ## Read this first — the backend changed on 2026-08-27
>
> This document was written for **Qwen3.8-27B on port 8000** and much of it still applies to
> that backend. The active backend is now **`qwen38-flash-next` on port 8001**, served from a
> source build of vLLM in a separate venv.
>
> Sections below marked **[27B]** describe the old backend and remain accurate *for it*.
> Anything about ports, model names, or capacity numbers should be checked against the
> current guides before you rely on it.
>
> | For | Go to |
> |---|---|
> | Day-to-day operation, health checks, incident triage | **[docs/RUNBOOK.md](docs/RUNBOOK.md)** |
> | Measured capacity, context vs agent-count sizing | **[docs/CAPACITY.md](docs/CAPACITY.md)** |
> | Codex CLI and VS Code Copilot wiring | **[docs/CLIENTS.md](docs/CLIENTS.md)** |
> | Building Flash-Next from source (6 blockers, by symptom) | **[../vllm-qwen38next/SETUP.md](../vllm-qwen38next/SETUP.md)** |
> | Everything, indexed | **[docs/README.md](docs/README.md)** |
>
> **Three things that supersede text further down:**
> 1. The **watchdog is now disabled on purpose** — see the correction in the Supervision
>    section. It could not distinguish a deliberate stop from a crash.
> 2. The **Copilot BYOK section is superseded** by [docs/CLIENTS.md](docs/CLIENTS.md), which
>    has verified values for the current backend.
> 3. **`max_threads` must never be written** alongside `max_concurrent_threads_per_session` —
>    it is a serde alias for the same field and writing both makes Codex reject the config.


How to run the local model this repo uses for test-quality triage
(`scripts/llm_test_audit.py`), and how `codex-qwen.sh` drives Codex CLI
against it as a fast, fully offline coding agent.

## What's already in place

- **Default model**: `RadixArk/Qwen3.8-27B-NVFP4` — a mixed-precision
  quantization of `Qwen/Qwen3.8-27B` (MLP layers at 4-bit NVFP4, attention at
  FP8, MTP head and vision tower left at full precision). Cached at
  `~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4` (~26 GB).
  See [Speeding it up](#speeding-it-up) for why this is the default and the
  evidence it doesn't cost meaningful accuracy.
- **Original checkpoint**: `Qwen/Qwen3.8-27B-FP8`, still cached
  (~29 GB) — what `scripts/llm_test_audit.py` (the batch triage tool) uses,
  and the documented fallback if the NVFP4 build ever needs reverting.
- **venv**: `.venv-llm` in this directory (Python 3.13, `vllm==0.27.1`). Used
  to live in the sibling `AlgoTrading-llm-audit` project; moved in-tree
  2026-08-24 after that worktree was deleted (routine AlgoTrading worktree
  cleanup) and took the venv with it, silently breaking the server for over a
  day since the failure exits 0-looking-clean at the shell level. Lesson:
  never put a load-bearing venv inside a disposable git worktree. Rebuilt
  identically (`vllm==0.27.1` + matching `flashinfer-jit-cache`, see below);
  model weights were untouched since they live in the separate HF cache.
- **GPU**: RTX PRO 6000 Blackwell Workstation Edition, 97 GB VRAM,
  1,792 GB/s memory bandwidth ([spec source](https://boston.co.uk/content-hub/nvidia-rtx-pro-6000-blackwell-workstation-edition/)).
- **Batch triage script**: `scripts/llm_test_audit.py` — loads
  `Qwen/Qwen3.8-27B-FP8` in-process via vLLM's `LLM()` class (no server) and
  scores every test file in one pass. Last full run: 1,267/1,268 files in
  1,035s. Output: `audit_reports/qwen_test_audit.{json,md}`.

The checked-in `activate_gpu_env_wsl.sh` is **stale** (Windows/WSL paths,
`/mnt/c/...`) — this machine is not WSL. Ignore it; use the env block below.

## Environment

`nvcc` and `ninja` ship inside the `.venv-llm` package tree rather than as
system packages, so `CUDA_HOME` has to point there explicitly. FlashInfer's
JIT sampler is incompatible with this CUDA build and must be disabled.

```bash
cd /home/pctablet505/Projects/local_llm
export CUDA_HOME=$(pwd)/.venv-llm/lib/python3.13/site-packages/nvidia/cu13
export PATH="$(pwd)/.venv-llm/bin:$CUDA_HOME/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
```

### flashinfer-jit-cache — required for NVFP4 {#flashinfer-jit-cache}

NVFP4 inference needs FlashInfer's grouped-GEMM FP4 kernel. This CUDA-13
toolchain **cannot JIT-compile it from source** — a genuine CCCL/nvcc header
incompatibility (`error: CUDA compiler and CUDA toolkit headers are
incompatible`), the same failure class as the sampler issue above, but this
kernel isn't optional the way the sampler was. The fix is the matching
**precompiled** wheel, a one-time install into this shared venv:

```bash
cd /home/pctablet505/Projects/local_llm
uv pip install --python .venv-llm/bin/python --no-deps \
  "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.16.post3/flashinfer_jit_cache-0.6.16.post3+cu130-cp39-abi3-manylinux_2_28_x86_64.whl"
```

That version must match the installed `flashinfer_python` build exactly
(check with `.venv-llm/bin/python -c "import flashinfer; print(flashinfer.__version__)"`,
then find the matching file under `https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/`).
`codex-qwen.sh` checks for this before starting the server and fails with
this section's link rather than a several-hundred-line compiler error.

## Running the batch triage

```bash
.venv-llm/bin/python scripts/llm_test_audit.py --repo /home/pctablet505/Projects/AlgoTrading
```

> **Currently broken, separately from the server fix above.** This script
> lived in `AlgoTrading-llm-audit/scripts/llm_test_audit.py` — the same
> deleted directory that held the venv. Only the venv was rebuilt
> (2026-08-24); the script itself has not been recovered or relocated. Find
> it in git history / another checkout before relying on this section again.

Re-run after significant test-suite rewrites to see whether scores moved.
Scores are structured-output constrained (`StructuredOutputsParams`), so the
run is unattended-safe; treat it as a triage signal, not ground truth — ground
truth for detection is mutation testing (`scripts/mutation_probe.py`).

## Speeding it up {#speeding-it-up}  **[27B]**

> These measurements are for the 27B on port 8000. Flash-Next's numbers differ
> substantially — see [docs/CAPACITY.md](docs/CAPACITY.md).


Four changes, each measured live on this card, each stacking with the
others. All are lossless or independently accuracy-audited — nothing here
trades quality for speed the way, say, aggressive temperature or truncation
would.

| Change | Mechanism | Measured throughput | Cumulative |
|---|---|---:|---:|
| (baseline) | plain autoregressive decode, FP8 | 39 tok/s | 1x |
| + MTP speculative decoding (n=1) | see below | 60 tok/s | 1.55x |
| + NVFP4 quantization | see below | 81 tok/s | 2.1x |
| + tuned `num_speculative_tokens` (n=4) | see below | **140 tok/s** | **3.6x** |

**MTP speculative decoding** (`--speculative-config '{"method": "mtp", ...}'`):
this checkpoint family ships its own multi-token-prediction head
(`mtp.layers.0.*` in the safetensors index, already on disk, no extra
download). It's **lossless by construction** — the MTP head proposes
tokens, the full model verifies them all in the same forward pass it
would've done anyway, and any mismatch is rejected and regenerated. Output
distribution is provably identical to plain decoding, whatever
`num_speculative_tokens` is set to; only the number of tokens confirmed per
memory-bandwidth-bound pass changes.

**Tuning `num_speculative_tokens`**: swept 1..6 with a fixed-prompt,
fixed-`max_tokens` benchmark (`/v1/completions`, 3 timed calls after a
warmup call, same GPU/model/every-other-flag held constant) to isolate the
effect of this one value:

| `num_speculative_tokens` | avg tok/s | mean acceptance length | per-position acceptance |
|---:|---:|---:|---|
| 1 | 68.2 | — | ~65% (1 position) |
| 2 | 107.8 | 2.38 | 0.80, 0.58 |
| 3 | 128.8 | 2.73 | 0.80, 0.56, 0.36 |
| **4 (current default)** | **140.4** | **3.02** | 0.78, 0.58, 0.42, 0.24 |
| 5 | 131.5 | — | (regressing) |
| 6 | 115.6 | — | (regressing further) |

Peaks at 4, then falls off: later draft positions get accepted less often
(later speculative tokens condition on earlier *guesses*, not confirmed
tokens, so error compounds), so past 4 the extra draft compute is spent
mostly on tokens that get thrown away. This is a pure config-value change —
same weights, same VRAM, same lossless-by-construction guarantee at every
setting — re-sweep if the model or hardware changes, since the peak is
empirical, not derived from a formula.

**NVFP4 quantization**: see [What's already in place](#whats-already-in-place)
for the scheme. Accuracy evidence, from the checkpoint's own shipped audit
trail (`qualification.json`, SHA-256'd) — not just a README claim:

| | |
|---|---|
| Eval | GSM8K, 1,319 examples |
| Accuracy | **97.27%** (1,283 correct) |
| Gate | ≥96.5% required — **pass** |
| MTP kept | excluded from quantization entirely (`ignore: ["mtp*"]`) |

Their own eval ran on different hardware/stack (4x NVIDIA GB300 via SGLang)
with an honest caveat about uncalibrated FP8 KV-cache scaling in *their* run
— worth knowing, but not something this deployment inherits, since KV cache
here stays at its default dtype. Correctness re-verified independently on
this card (`12 * 7` → `84`, deterministic).

**Why weights got smaller but total VRAM didn't, by default**: NVFP4 weights
measure 20.75 GiB vs 28.51 GiB for FP8 — a real ~27% cut. But vLLM's default
`--gpu-memory-utilization` targets a *fixed percentage* of free VRAM for the
KV-cache pool, not a fixed size — so the freed weight memory just became
more idle KV-cache capacity, and total usage went *up* slightly (92 GB vs
90 GB) despite smaller weights. `codex-qwen.sh` sets `--gpu-memory-utilization`
explicitly rather than take the default — see `GPU_MEM_UTIL` in the script.

**Choosing the value — one full context vs. several subagents at once**:
the fixed floor (weights + activation + CUDA graph) is ~24.7 GiB regardless
of setting; each 262,144-token full context costs ~8.84 GiB of KV cache
(measured consistently at two different util values). That gives a direct
trade between headroom for other GPU work and how many *simultaneous*
full-context sessions — several subagents each working a large task at
once, not just one — the pool can hold before requests start queuing:

| `--gpu-memory-utilization` | Total VRAM | Free for other work | Full-context concurrency |
|---:|---:|---:|---:|
| **0.47 (current default)** | **~45 GiB** | **~53 GiB** | **~2.0x** |
| 0.5 | 47.5 GiB | 50.4 GiB | 2.6x |
| 0.7 | 66.5 GiB | 31.4 GiB | 4.7x |
| 0.85 | 80.7 GiB | 17.2 GiB | 6.3x |
| 0.92 (vLLM's own default) | 87.4 GiB | 10.5 GiB | 7.1x |

Set to **0.47**, measured live: ~45 GB total, room for ~2 concurrent
full-context sessions. Deliberately dialed back from the 0.85 (~6.3x)
setting this ran at before — that much concurrent-subagent headroom was
provisioned but never actually exercised (no concurrent-session benchmark
was ever run against it), so it was reserved capacity rather than a proven
need. Confirmed live: throughput is unaffected by this setting either way
(140.4 tok/s at both 0.47 and 0.85) — `--gpu-memory-utilization` only trades
concurrency headroom for idle VRAM, it does not touch single-stream speed.
Revisit upward if real concurrent-subagent usage says otherwise.

### Prefix caching — enabled

`--enable-prefix-caching --mamba-cache-mode all`. An earlier attempt at
`--enable-prefix-caching --kv-cache-dtype fp8` together crashed `EngineCore`
silently on this hybrid Mamba/attention architecture (process zombied, no
error surfaced — the boot just stopped advancing). Root cause: vLLM ties
prefix caching to a separate `mamba_cache_mode` setting for hybrid models,
which defaults to `none` — only valid when prefix caching is off. Retested
live with `--mamba-cache-mode all` alone (not re-adding `--kv-cache-dtype
fp8`, which was never isolated as the actual cause): no crash, engine
survived real multi-turn generation, and repeat-prefill of a 20K-token
shared prefix dropped from 2.41s to 0.40s (turn 2 hit cache, `prompt_tokens`
identical both turns). This is the main win for tool-call-heavy agent
sessions — without it, every turn re-prefills the entire growing history
from scratch, since Codex resends full conversation state each request
(the server itself is stateless per-request; nothing is held in GPU memory
between turns either way, idle agents cost nothing).

`--kv-cache-dtype fp8` stays out — it was the other half of the original
crashing combination and hasn't been retested in isolation now that
`mamba-cache-mode` is confirmed as the real fix. Untested, not assumed safe.

### Tried and deliberately not kept

- **`--optimization-level 3`** (vLLM default is 2): regressed throughput to
  ~60 tok/s (vs. 140.4 at default), and got slower across repeated calls
  rather than stabilizing — the opposite of expected JIT-warmup behavior.
  Not investigated further; default level 2 stays.
- **Longer context via RoPE/YaRN extrapolation**: not attempted. 262,144
  tokens is the model's own trained ceiling, confirmed from its config, not
  a VRAM limit — the current KV-cache pool already holds 2.5-7x that much
  concurrently depending on `--gpu-memory-utilization`. Extrapolating past
  the trained length is a genuine, well-documented accuracy trade that gets
  worse the further you push it, unlike everything above — explicitly out
  of scope given the goal of not compromising accuracy for speed.
- **NVFP4 for the batch triage script**: `scripts/llm_test_audit.py` still
  targets `Qwen/Qwen3.8-27B-FP8` — untouched, out of scope for this file.

## Why the server kept dying {#why-the-server-kept-dying}  **[27B]**

Investigated 2026-08-22 after the server was found dead repeatedly. It had
died **four times in two days**, and nothing restarted it or recorded why.

### The mechanism

A CUDA kernel in the decode path hits an illegal / misaligned memory address.
That poisons the CUDA context, EngineCore dies, and vLLM then **shuts the whole
server down on purpose**. In order:

1. **GPU MMU fault.** Recorded only by the kernel driver:

   ```
   NVRM: Xid (PCI:0000:01:00): 13, pid=665187, name=VLLM::EngineCor, Graphics Exception: ...
   NVRM: Xid (PCI:0000:01:00): 31, pid=665187, name=VLLM::EngineCor, ... MMU Fault:
     ENGINE GRAPHICS GPC0 GPCCLIENT_T1_9 faulted @ 0x17_00001000.
     Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_READ
   ```

2. **EngineCore dies** with `torch.AcceleratorError: CUDA error: misaligned
   address`, surfacing at `async_copy_ready_event.synchronize()` in
   `gpu_model_runner.py`. The fault is reported asynchronously, so the Python
   traceback points at the sync, not at the offending kernel.

3. **In-flight streams 500.** The API server raises `EngineDeadError`. Because
   SSE headers have already gone out, starlette cannot replace the body and
   raises `RuntimeError: Caught handled exception, but response already
   started.` That message is a **symptom, not a cause** — it is per-request and
   never fatal on its own.

4. **vLLM shuts itself down.** `watchdog_loop` in
   `vllm/entrypoints/launcher.py` polls every 5 s and, on `engine.errored`,
   sets `server.should_exit = True`. uvicorn drains and the process exits.

**The exit status is 0.** This is the trap: vLLM exits *successfully* when its
engine has died. Any supervisor using `Restart=on-failure` would treat that as
a clean run and never restart it. Verified live — killing EngineCore produced
`SERVICE_RESULT=success, EXIT_STATUS=0`. The unit uses `Restart=always`.
**Do not change that to `on-failure`.**

### The four deaths

All four are `VLLM::EngineCor`, from `journalctl -k` (note: `dmesg` is
unreadable on this box — `kernel.dmesg_restrict=1` — but `journalctl -k` works
without privileges):

| When | PID | Xid | Fault |
|---|---|---|---|
| 2026-08-21 00:51:16 | 703500 | 31 | GPC6 `FAULT_PTE` VIRT_READ @ `0x768f_b91d9000` |
| 2026-08-21 01:04:22 | 1153842 | 31 | GPC1 `FAULT_PTE` VIRT_READ @ `0x72b6_a0432000` |
| 2026-08-21 23:24:59 | 200652 | 13+31 | GPC7 `FAULT_PDE` VIRT_WRITE @ `0x989_86b66000` |
| 2026-08-22 05:20:22 | 665187 | 13+31 | GPC0 `FAULT_PDE` VIRT_READ @ `0x17_00001000` |

### What it was NOT

Ruled out with evidence, not assumed:

- **Linux OOM killer** — zero OOM records in `journalctl -k` across all boots;
  `systemd-oomd` logged no kills. The box has 182 GB RAM with ~119 GB free.
- **CUDA OOM / VRAM exhaustion** — a real CUDA OOM says `CUDA out of memory`
  and names the allocation. An MMU `FAULT_PDE` is an *unmapped address*, not a
  failed allocation. KV-cache usage at the moment of the 05:20 crash was 14-21%.
- **SIGHUP / terminal or SSH going away** — the old launcher used `nohup` (so
  SIGHUP was ignored anyway), `KillUserProcesses` is at its default `no`, and
  the process exited through its own watchdog, not a signal.
- **An external killer** — no systemd unit existed, no supervisor, no stray
  `pkill`. The shutdown sequence in the log is vLLM's own, start to finish.
- **A bad request shape from Codex** — `POST /v1/responses` is fully supported
  by this build and returns 200; it served hundreds of 200s right up to the
  crash. The run of 500s *followed* the engine's death.

### Correlates (suspected, not proven)

The exact faulting kernel is not recorded. At the 05:20 crash the scheduler
dump showed five concurrent long-context streams (41k-53k computed tokens
each) and MTP draft slots as placeholders:

```
scheduled_spec_decode_tokens={resp_...: [-1, -1, -1, -1], ... }
num_spec_tokens_to_schedule=4
```

Present in every crash: MTP speculative decoding, async scheduling
(`step_with_batch_queue`), prefix caching on a hybrid Mamba/attention model,
and FP8 KV cache. If the faults continue, disable these **one at a time** and
keep the run logs — the death record now makes that a measurable experiment.

> **The FP8 KV cache is on, and not by choice.** The launcher deliberately does
> not pass `--kv-cache-dtype fp8`, and a comment says so. It is enabled anyway:
> the checkpoint's own `hf_quant_config.json` declares
> `"kv_cache_quant_algo": "FP8"`, and vLLM's modelopt loader honours it. The
> boot log confirms `kv_cache_dtype=fp8_e4m3`. So "we left FP8 KV cache out" is
> **false in practice** — a prime suspect that was believed to be excluded.

### A different failure: GPU fell off the bus (2026-08-24)

Not the mechanism above — worth keeping separate, not lumped in as "another
MMU fault." `journalctl -k` showed:

```
NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.
NVRM: Xid (PCI:0000:01:00): 154, GPU recovery action changed from 0x0 (None)
  to 0x2 (Node Reboot Required)
```

plus a kernel oops inside the nvidia driver itself
(`nvidia_dev_put_uuid`, bad pointer in the UVM release path).
`nvidia-smi` afterwards: `Unable to determine the device handle ... No
devices were found`. Xid 154 says the recovery action explicitly —
**Node Reboot Required** — no process-level restart, driver module
reload, or vLLM setting reaches a GPU that's off the bus. Root cause of
*why* it fell off the bus is not established (first occurrence; unlike the
Xid 13/31 pattern above there's no correlate list yet — if it recurs, that's
the point to start one).

Two things this exposed, both fixed the same day:

1. `qwen-server-record-death.sh` grepped for "any Xid in the journal" and
   unconditionally labeled it the known MMU-fault pattern — so all six
   automatic restart attempts got logged as Xid 13/31 when the real codes
   were 79/154. Fixed to classify by code instead of assuming.
2. `Restart=always` dutifully retried six times in 21 seconds (each one
   failing instantly on `NVMLError_Unknown` — the GPU wasn't there),
   burned through `StartLimitBurst`, and the unit sat `failed` with nothing
   watching it. See `qwen-vllm-watchdog.timer` below.

## Supervision — the systemd user unit {#supervision}

> **CORRECTION (2026-08-27): the watchdog described in this section is now DISABLED, and
> should stay disabled.**
>
> `qwen-vllm-watchdog.timer` treated a `inactive` unit as actionable. But `inactive` is
> exactly what a *deliberate stop* leaves behind, so the watchdog resurrected a server that
> had been stopped on purpose (observed 20:13:58, 2026-08-27) — including once during a
> deliberate teardown for testing. It also called `systemctl reset-failed`, which erases the
> `StartLimitBurst` backstop protecting against restart storms.
>
> A supervisor cannot infer intent from observed state. Any replacement must gate restarts on
> a **persisted `desired_state`**, and must distinguish *crashed while serving* (restart is
> right) from *failed to boot* (restarting loops forever and hides the cause — 44 of 72
> recorded exits never reached serving).
>
> Current state: `systemctl --user is-enabled qwen-vllm-watchdog.timer` -> `disabled`.


The server runs under a systemd **user** unit, `qwen-vllm.service`. Lingering
is enabled (`loginctl enable-linger`, no root needed), so it survives logout
and starts at boot.

### The one command to stop it

```bash
systemctl --user stop qwen-vllm
```

This stops the server **and** the supervisor — nothing brings it back until you
start it again. Run it before a scheduled training block. `./codex-qwen.sh stop`
does exactly this.

To start it again: `systemctl --user start qwen-vllm` (or `./codex-qwen.sh qwen`).
To take it out of the boot sequence entirely: `systemctl --user disable --now qwen-vllm`.

### Files

| Path | Role |
|---|---|
| `~/.config/systemd/user/qwen-vllm.service` | the unit |
| `bin/qwen-server-run.sh` | `ExecStart` — guards, log rotation, then `exec vllm serve` |
| `bin/qwen-server-record-death.sh` | `ExecStopPost` — writes the death record |
| `logs/qwen_deaths.log` | every exit, with status, Xid check and log tail |
| `logs/qwen_server-<stamp>.log` | one log per run; `qwen_server.log` symlinks to the current one |
| `~/.config/systemd/user/qwen-vllm-watchdog.timer` | fires `qwen-vllm-watchdog.service` every 5 min |
| `bin/qwen-vllm-watchdog.sh` | un-sticks the unit after a burst-limit lockout — see below |
| `logs/qwen_watchdog.log` | what the watchdog checked and did, each run |

### It will not seize the GPU from training

`bin/qwen-server-run.sh` exits **69** (`EX_UNAVAILABLE`) and the unit sets
`RestartPreventExitStatus=69`, so systemd stands down rather than retrying,
when either:

- a training marker exists at `run/training_in_progress` (also checked:
  `~/.cache/algotrading/training_in_progress`,
  `<AlgoTrading>/run/training_in_progress`); or
- free VRAM can't fit what vLLM is about to ask for (`GPU_MEM_UTIL x total`)
  plus a fragmentation margin (`QWEN_GPU_HEADROOM_MIB`, default 4096 MiB).
  This replaced an earlier fixed-8-GB-holder threshold that couldn't tell
  "the card is busy" from "the card is full" and didn't scale with
  `GPU_MEM_UTIL` — see the comment above guard 2 in `qwen-server-run.sh`.

```bash
touch /home/pctablet505/Projects/local_llm/run/training_in_progress   # keep it off the GPU
rm    /home/pctablet505/Projects/local_llm/run/training_in_progress   # allow it back
```

Both guards are tested live.

`StartLimitBurst=5` / `StartLimitIntervalSec=600` stop a hot restart loop: five
starts in ten minutes and the unit goes `failed` and stays there — systemd
itself will never try again. `qwen-vllm-watchdog.timer` exists for exactly
that: every 5 min it checks whether the unit is `failed` or `inactive`
(guard-69 stand-down), and if the GPU actually responds to `nvidia-smi -L`,
clears the lockout and retries (`reset-failed` + `start`). This is also how a
guard-69 stand-down now recovers on its own once training finishes or VRAM
frees up, instead of needing a human to remember to run `systemctl --user
start qwen-vllm`.

It deliberately does **not** retry if `nvidia-smi -L` itself fails — that
means the GPU isn't there at all (see the next section), and hammering
`start` against a genuinely absent GPU only burns through the same burst
limit again in seconds for nothing. It logs that it declined, to
`logs/qwen_watchdog.log`, and leaves the unit failed for a human to act on.

### Logs no longer get destroyed

The old launcher opened the single log with `>`, truncating it on every start.
That is why the 2026-08-21 23:24 crash left no log at all — only the kernel
journal still had the Xid. Each run now gets its own timestamped file, the last
10 are kept, and everything also goes to the journal
(`journalctl --user -u qwen-vllm`).

```bash
./codex-qwen.sh deaths     # the death record
./codex-qwen.sh status     # includes restart count
```

## Multimodal — currently text-only in practice

Both checkpoints are Qwen3-VL-based — `vision_config`, `image_token_id`, and
`video_token_id` are genuinely present in the model config, and on the
NVFP4 build vision tensors are explicitly excluded from quantization (kept
at full precision, same treatment as the MTP head). The weights are there.

**But vLLM 0.27.1 doesn't actually serve images for this model right now.**
Confirmed from the server's own boot log:
```
WARNING [registry.py:117] Model ... is treated as multimodal but has no
registered multimodal processor; running in text-only mode.
INFO [model.py:842] Disabled mm_prefix attention mode because multimodal
inputs are configuration-disabled.
```
So despite the checkpoint carrying vision weights, this vLLM version has no
registered processor for this specific architecture and disables image
input at the registry level — not a config flag to flip, a real gap in this
vLLM build's model support. Sending an image through Codex's `-i` flag would
not currently work against this server. Revisit on a vLLM upgrade; check
this same log line first before assuming it's fixed.

## Using it with Codex CLI — `codex-qwen.sh`

**Claude Code (this tool) cannot be pointed at a local model** — it's
Anthropic-specific and talks only to the Anthropic API. There is no supported
way to swap its backend.

**OpenAI's Codex CLI is different**: it's designed around pluggable model
providers. This directory has a single launcher, `codex-qwen.sh`, that
handles all of it — installs Codex, starts/stops the Qwen server, and
launches Codex against either the local model or your normal OpenAI account.
It's self-contained: everything it writes lives under this directory
(`bin/`, `logs/`, `run/`, `.codex/`) except the vLLM venv and model weights,
which it reuses from `AlgoTrading-llm-audit` by absolute path only — it never
`cd`s there.

```bash
./codex-qwen.sh               # interactive menu
./codex-qwen.sh qwen [args]   # start the server if needed, launch codex on it
./codex-qwen.sh resume [args] # start the server if needed, resume the last
                               #  local-Qwen session (--last)
./codex-qwen.sh openai [args] # launch codex against your OpenAI account, unchanged
./codex-qwen.sh stop          # stop the local Qwen server (frees the GPU)
./codex-qwen.sh restart       # stop then start again, same settings
./codex-qwen.sh status        # report server / codex install status
./codex-qwen.sh health        # GPU memory, current util, OOM/orphan checks
./codex-qwen.sh set-mem 0.65  # persist a new util, restart to apply it
./codex-qwen.sh tail-log      # tail the server log (Ctrl-C to exit)
```

Verified end-to-end on 2026-08-15: `./codex-qwen.sh qwen` against a running
server produced a correct response through the full Codex agent loop, on
both the original FP8 model and the current NVFP4 default.

**Sessions persist independent of the server.** Codex saves full sessions to
disk (`.codex/sessions/...`) regardless of whether the vLLM process is
running, so stop-to-free-the-GPU-then-`resume`-later loses nothing:
```bash
./codex-qwen.sh stop       # free the GPU for anything else
# ... do other GPU work ...
./codex-qwen.sh resume     # pick up exactly where you left off
```

### What it actually does

- **Installs Codex CLI from GitHub releases** (`bin/codex`), not npm — this
  machine has no `node`/`npm`. Codex ships as a standalone Rust binary
  (`codex-x86_64-unknown-linux-musl.tar.gz`); no runtime dependency beyond
  that.
- **Checks for `flashinfer-jit-cache`** before starting (see above) and
  fails fast with a pointer here instead of a wall of compiler errors.
- **Writes `.codex/config.toml`** (under this directory's own `CODEX_HOME`,
  never `~/.codex` — so it can't collide with a real OpenAI-account config):
  ```toml
  model_provider = "local-qwen"
  model = "RadixArk/Qwen3.8-27B-NVFP4"

  [model_providers.local-qwen]
  name = "local-qwen"
  base_url = "http://localhost:8000/v1"
  wire_api = "responses"
  ```
  **`wire_api` must be `"responses"`, not `"chat"`.** Codex 0.147.0 dropped
  chat-completions support entirely — `"chat"` is a hard config error now.
  vLLM 0.27.1 does serve `/v1/responses` (confirmed by reading
  `vllm/entrypoints/openai/responses/api_router.py` and by a live smoke test),
  so this works, but it's new enough that older vLLM builds won't have it.
- **Starts the server** with the flags detailed in
  [Speeding it up](#speeding-it-up), waits on `/v1/models` until it answers
  200 (first start: ~2-3 min for weight load + `torch.compile` warmup;
  cached afterwards), then execs Codex.

### Changing GPU memory allocation on the fly — `set-mem`

You'll want less VRAM reserved for Qwen when you need the GPU for something
else, and more when running several concurrent Codex subagents. vLLM sizes
its KV-cache block pool once, during boot profiling — there's no API to grow
or shrink it on a live process, confirmed by the vLLM source (no such
endpoint exists) and by design (the block pool is allocated up front so every
request can assume it exists). So this can't be done without a restart; what
`set-mem` gives you is the least-friction version of that restart:

```bash
./codex-qwen.sh set-mem 0.65   # persists to .config, restarts the server now
                                #  if it's running, or just persists if it's
                                #  down (takes effect next start)
```

Codex needs no reconfiguration either way — it always talks to
`localhost:8000`, whether that's the server that was already running or the
one `set-mem` just started. A resumed session picks up wherever it left off
(sessions live on disk, independent of the server — see above); a
mid-request session gets one dropped connection, same as any `stop`.

The number of concurrent subagents is Codex's own setting, not this script's
— `set-mem` only controls how much VRAM is available for them to share. Use
the measured table in [Speeding it up](#speeding-it-up) to pick a value:
lower (e.g. 0.47) when you need the GPU for something else, higher (e.g.
0.75-0.85) before a session with several subagents running full-context work
concurrently.

### Raising the subagent cap — `set-subagents` {#subagent-concurrency}

Codex's own default limit on concurrent subagents is **6**. Unrelated to
`GPU_MEM_UTIL` — this is CPU-side orchestration (how many subagent threads
Codex will run at once), not GPU memory.

```bash
./codex-qwen.sh set-subagents 64   # persists to .config
```

The config key is `max_concurrent_threads_per_session`, found in the Codex
binary's own struct layout (`CodeModeHostConfigToml`) and in a real example
from a shipped Codex security plugin (`capability-profiles.toml`, using the
same key name under a different feature table). **What table it lives under
for this install's active setup (Codex 0.147.0, `multi_agent` v1 +
`code_mode_host`, both `stable`/enabled — check with `codex features list`)
was not fully confirmed** — `codex --strict-config` rejects the nested form
under `features.multi_agent.*` and `features.code_mode_host.*` (those are
plain booleans there — a real type error, confirmed), but doesn't validate
deeply enough elsewhere to prove a specific top-level table is correct
either. `write_provider_config()` in `codex-qwen.sh` writes it to *both*
`[code_mode_host]` and `[multi_agent]` as top-level tables — an unrecognized
table is silently ignored by this build, so this is safe either way, just
not guaranteed effective.

**To actually verify**: start a Codex session and ask it to spawn more than
6 concurrent subagents on independent tasks; if more than 6 run at once, it
worked. Hasn't been done yet (written same day the GPU was down for a
reboot). If it turns out neither table is right, the confirmed-working
alternative is `features.multi_agent_v2.max_concurrent_threads_per_session`
— but that requires enabling `multi_agent_v2` first (`multi_agent_v2` is
`stable` but disabled by default), which switches orchestrator versions, a
bigger change than this setting alone. Don't flip that without deciding to.

Takes effect on the **next** `qwen`/`resume` launch — `config.toml` is
regenerated at launch time, not live, so an already-running Codex session
won't pick it up.

If something crashes or the server gets stuck, `./codex-qwen.sh health`
reports current util, orphaned `EngineCore` processes, and any OOM lines
from the tail of the log; `./codex-qwen.sh restart` recovers a stuck server
without changing any settings.

## Using it with VS Code Copilot Chat (BYOK)  [superseded]

> **SUPERSEDED by [docs/CLIENTS.md](docs/CLIENTS.md).** The values below target the 27B on
> port 8000. The current backend is `qwen38-flash-next` on **8001**, and tool-calling has since
> been **verified live** (this section said it was unverified).
>
> Two findings here are still correct and important, and are carried into CLIENTS.md:
> `vendor` must be `customendpoint` (not `openai`), and `maxInputTokens + maxOutputTokens`
> must total the real context window or VS Code silently uses a much smaller default.
>
> Also note: **Copilot is not currently installed** (`code --list-extensions` shows only
> `anthropic.claude-code` and `openai.chatgpt`), and **BYOK does not cover inline code
> completions** — chat and utility tasks only.


The VS Code **Codex extension** was tried first and rejected: it hardcodes
its config path to `~/.codex/config.toml` (no `CODEX_HOME` override like the
CLI has, so it can't be sandboxed the way `codex-qwen.sh` is), and has an
open bug where new conversations started in the extension ignore custom
providers and silently fall back to `gpt-5-codex`
([openai/codex#4558](https://github.com/openai/codex/issues/4558),
[#7971](https://github.com/openai/codex/issues/7971)). Only conversations
*started from the CLI and continued in the extension* are confirmed to work
correctly with a custom provider — not useful as a primary workflow.

**Copilot Chat's own BYOK ("Bring Your Own Key") support works instead**,
with no known equivalent bug, and needs zero changes to the running server
— it talks plain `/v1/chat/completions`, which vLLM already serves with
tool-calling on (`--enable-auto-tool-choice --tool-call-parser qwen3_xml`,
already part of the standard startup flags).

Setup (one-time, interactive — the API key/URL step goes into VS Code's
secret storage, not a file, so it can't be scripted):

1. Command Palette → **`Chat: Manage Language Models`**
2. **Add Models** → **Custom Endpoint** (OpenAI-compatible) provider
3. Group name: anything, e.g. `Local Qwen`
4. Fields:
   - **vendor**: `customendpoint` — **not** `openai`. VS Code has an open bug
     where `vendor: "openai"` silently ignores `maxInputTokens` /
     `maxOutputTokens` even when set
     ([microsoft/vscode#322216](https://github.com/microsoft/vscode/issues/322216)) —
     this was the cause of a real "limited context length" report, not a
     server-side issue.
   - **url**: `http://localhost:8000/v1/chat/completions`
   - **apiType**: `chat-completions`
   - **apiKey**: any non-empty string (`local`) — vLLM doesn't check it
   - **id** / **name**: `RadixArk/Qwen3.8-27B-NVFP4`
   - **toolCalling**: `true`
   - **maxInputTokens**: `246144`
   - **maxOutputTokens**: `16000`
     (sum = 262144, the model's actual trained context limit from its own
     `config.json` — VS Code treats `maxInputTokens + maxOutputTokens` as the
     model's total context window, and defaults to something much smaller if
     these are simply left out.)
5. Server must already be running (`./codex-qwen.sh qwen` or `resume` starts
   it) — the model then appears in Copilot Chat's model picker.

Not yet verified end-to-end (setup steps only, sourced from VS Code's own
BYOK docs — https://code.visualstudio.com/blogs/2026/06/18/byok-vscode).
Confirm it actually connects and tool-calls correctly before relying on it.

### Caveats found while building this

- **Qwen's `<think>...</think>` reasoning leaks into the visible transcript.**
  Codex doesn't parse Qwen3's thinking-tag format as a distinct reasoning
  channel the way it does for GPT models, so the raw chain-of-thought prints
  as ordinary agent output before the final answer. Harmless, just noisy.
- **Patch/tool-call reliability.** Codex's edit/patch tool-calling format was
  tuned against GPT models. A 27B open model follows that format less
  reliably — expect more malformed patches and weaker multi-step agentic
  performance than Codex's default hosted model. It's a reasonable fit for
  constrained, single-file tasks (explain this code, review this diff, small
  edits) fully offline and at zero API cost; it is not a like-for-like
  replacement for large autonomous changes.
- **A `SIGTERM` sent mid-startup can be silently swallowed.** Confirmed live:
  killing the server while it was still loading weights / running
  `torch.compile` / capturing CUDA graphs did nothing — it finished booting
  to fully serving as if no signal had arrived. A second `SIGTERM`, sent once
  idle, worked within seconds. `stop_server()` accounts for this: it verifies
  the process actually exited rather than trusting `kill`'s return code, and
  escalates to `SIGKILL` after 30s if it hasn't. So `stop` can occasionally
  take up to ~30s — that's this working as designed, not hanging.
- The server holds ~45 GB of the card (per `GPU_MEM_UTIL` above) while
  running. `./codex-qwen.sh stop` (or option 4 in the menu) releases it —
  nothing here leaves it running unattended.
- **Backgrounding a `qwen`/`resume` invocation and killing the wrapper does
  NOT stop the server.** `start_server()` `nohup`s and `disown`s the actual
  `vllm serve` process almost immediately, specifically so it survives its
  parent exiting normally — that same design means a quick
  `codex-qwen.sh qwen & sleep 1; kill $!`-style probe kills only the
  wrapper, not the now-detached server, which keeps running untracked
  (`PID_FILE` still gets written, so it's recoverable via `status`/`stop`,
  but it's easy to think nothing started when something real did). Confirmed
  live: this is exactly how the 0.47 config's server first came up — always
  use `stop` to tear a server down, never a timed background-kill.
- **Reasoning-effort selection is a no-op for this model.** Codex's
  `-c model_reasoning_effort=...` is accepted and shown in the session
  banner, but vLLM only maps it to thinking on/off for non-catalog models —
  there's no graduated low→max depth control the way GPT-5 has. Thinking is
  already on by default via the model's own chat template regardless of what
  effort is requested.
