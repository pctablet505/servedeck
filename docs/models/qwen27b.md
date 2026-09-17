# Qwen3.8-27B-NVFP4 (`qwen27b`)

`RadixArk/Qwen3.8-27B-NVFP4`, a mixed-precision quantisation of `Qwen/Qwen3.8-27B`: MLP at NVFP4
group 16, attention and linear-attn at FP8, MTP head and vision tower unquantised (`ignore: ["mtp*",
"mtp.layers.0*"]`). Aliases `qwen27b`, `inline`, `Qwen3.8-27B-NVFP4`, the repo id. Port 8004, slot
`main`, build `stock` (`~/Projects/local_llm/.venv-llm-029`, vLLM **0.29.0**), context native
262,144, 20.4 GiB of weights in the hub cache.

It is **not running** (2026-09-18: Flash-Next holds 93.5 GiB of the card) and has **never been
launched under servedeck v2** — the user journal holds zero `model-qwen27b` invocations; its last run
was under the old shell launcher on 2026-09-12. Read
[First launch under servedeck](#first-launch-under-servedeck--what-to-watch) first.

Prefer it over [Flash-Next](flashnext.md) when you want this checkpoint's answers (its shipped audit
trail records GSM8K 1,319 examples at 97.27% against a ≥96.5% gate, on the publisher's 4x GB300
SGLang stack, not here), or when you need a **stock pip wheel** engine: GLM and Flash-Next run from
patched forks, so this is the only main model that tests an unmodified vLLM release on this card.
Being main-slot it evicts the live main model, so use `servedeck switch qwen27b` — a plain `start` is
refused while another main model holds the VRAM ([../ARCHITECTURE.md](../ARCHITECTURE.md)).

## Launch configuration

Rendered from `[models.qwen27b]`; argv verified 2026-09-18 by rendering it.

| Flag | Value | Why |
|---|---|---|
| `--served-model-name` | `Qwen3.8-27B-NVFP4 RadixArk/Qwen3.8-27B-NVFP4 qwen27b inline` | Every name a client was ever wired to answers. v1 launched one name and broke configs written against the repo id. |
| `--max-model-len` | 262,144 | The trained ceiling, and the owner's rule: full native context, never pool ÷ agents ([../DECISIONS.md](../DECISIONS.md)). |
| `--gpu-memory-utilization` | 0.95 | Pinned, not computed: the owner's deliberate choice for KV headroom (2026-09-11). Make it safe, do not lower it. |
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":4}` | The checkpoint ships its own MTP head (`mtp.layers.0.*`), so speculation is lossless by construction — drafts are verified in the forward pass the model would have done anyway. 4 is the measured peak. |
| `--max-num-seqs` | 16 | The live v1 override (script default 128); sized for 2-3 large-context agents plus small ones. The pool fits far more short requests, so this can rise once saturation stability is re-checked. |
| `--enable-prefix-caching` | — | Codex and the chat clients resend the whole conversation each turn; without it every turn re-prefills the full history. |
| `--mamba-cache-mode` | `all` | Required by the flag above on this hybrid Mamba/attention model: `mamba_cache_mode` defaults to `none`, valid only with prefix caching off. Repeat prefill of a 20k shared prefix fell 2.41 s → 0.40 s (measured on or before 2026-08-27; not re-measured). |
| `--enable-auto-tool-choice --tool-call-parser qwen3_xml`, `--reasoning-parser qwen3` | — | Tool calls and thinking; the gateway mirrors `reasoning` into `reasoning_content` so chat-completions clients keep it ([../CLIENTS.md](../CLIENTS.md)). |
| env | `VLLM_USE_FLASHINFER_SAMPLER=0`, `HF_HUB_OFFLINE=1`, `CUDA_HOME`/`PATH` into the venv's `nvidia/cu13` | No CUDA toolkit on the box, so FlashInfer's JIT sampler cannot compile; the hub cache is the only weight source. |

Deliberately absent: **`--kv-cache-dtype fp8`**, because FP8 KV is on anyway and not by choice — the
loaded snapshot's `config.json` declares `quantization_config.kv_cache_scheme` as 8-bit float (a
sibling snapshot's `hf_quant_config.json` says `kv_cache_quant_algo: FP8`), vLLM's modelopt loader
honours it, and the 2026-09-11 and 09-12 boot logs both report `kv_cache_dtype=fp8_e4m3`. A crash
suspect the launcher's comments called excluded was live throughout. The explicit flag stays out
because it was the other half of a combination that once hung `EngineCore` silently and was never
isolated. **`--enforce-eager`** is refused on principle (below). **`--shutdown-timeout`** is
unnecessary — this model pins no host memory and writes no `/dev/shm` offload buffer — but its unit
still gets `Restart=always` and `TimeoutStopSec=120`.

## Utilisation, context and KV pool

| util | ctx | KV pool | Full-length concurrency | When |
|---:|---:|---:|---:|---|
| 0.95 | 262,144 | 1,817,628 tokens | 6.93x | 2026-09-11, vLLM 0.29.0 |
| 0.91 | 262,144 | 1,707,556 tokens | 6.51x | 2026-09-12, vLLM 0.29.0, from its boot log |

Utilisation buys concurrent full-length sessions, not speed: 140.4 tok/s single-stream at both 0.47
and 0.85 (on or before 2026-08-27; not re-measured). The fixed floor is ~24.7 GiB of weights,
activations and CUDA graphs and each 262,144-token context costs ~8.84 GiB of KV (same era).
`compute_util` treats the pinned 0.95 as a requirement: it refuses when that much does not fit,
naming what is free, and never quietly launches lower.

## Measured performance

The 3.6x, each step measured live on this card, cumulative (port-8000 era, on or before 2026-08-27;
not re-measured):

| Change | Single-stream | Cumulative |
|---|---:|---:|
| Baseline: plain autoregressive decode, FP8 weights | 39 tok/s | 1x |
| + MTP speculative decode, n=1 | 60 tok/s | 1.55x |
| + NVFP4 weights | 81 tok/s | 2.1x |
| + `num_speculative_tokens` 4 | 140 tok/s | 3.6x |

`num_speculative_tokens` swept 1-6, fixed prompt and `max_tokens`, three timed `/v1/completions`
calls after a warmup, everything else held (same era):

| n | avg tok/s | mean acceptance | per-position acceptance |
|---:|---:|---:|---|
| 1 | 68.2 | — | ~0.65 |
| 2 | 107.8 | 2.38 | 0.80, 0.58 |
| 3 | 128.8 | 2.73 | 0.80, 0.56, 0.36 |
| **4** | **140.4** | **3.02** | 0.78, 0.58, 0.42, 0.24 |
| 5 | 131.5 | — | regressing |
| 6 | 115.6 | — | regressing further |

It peaks at 4 because later draft positions condition on guesses rather than confirmed tokens, so
error compounds and the extra draft compute goes on tokens that are thrown away. The peak is
empirical: re-sweep if the model, the vLLM version or the hardware changes.

The 0.29.0 upgrade was also a throughput upgrade, at ctx 110,592 on 2026-09-11: single stream
124 → 143 tok/s; 16-way 15 tok/s (with two 130 s stalls) → **1,207 tok/s**; identical answers. The
16-way figure on 0.27.1 is a symptom of the race below, not a baseline.

## Crash history and the root cause

GPU faults attributable to this model, from `journalctl -k -b all` read 2026-09-18 (the default
current-boot filter shows nothing on a box that reboots this often):

| When | Signature | Conditions |
|---|---|---|
| 2026-08-21 – 08-24 | Paired Xid **13 + 31** on `VLLM::EngineCor`, 6 incidents, last 08-24 23:45:39 (`Graphics Exception: ESR 0x5bb730=0x9000d`) | util 0.62 and 0.92, vLLM 0.27.1 |
| 2026-09-11 18:16:56 | Xid **31**, `FAULT_PDE ACCESS_TYPE_VIRT_WRITE` on `VLLM::EngineCor`, ~8 min after boot | util 0.95, vLLM 0.27.1 |
| 2026-09-11 19:26:27 | **No kernel Xid**: host-side `Segfault encountered` in `cuGraphLaunch` → `at::cuda::CUDAGraph::replay()`, then `EngineDeadError` | util 0.95, vLLM 0.27.1, under benchmark |

**Root cause (upstream-converged, never reproduced locally):** a race in the hybrid-GDN Mamba
state-copy plus CUDA-graph replay path on SM120, present in vLLM 0.26.0 through 0.28.0 and absent in
0.24.0. `vllm-project/vllm#52225` reports this exact checkpoint (MTP-4, FP8 KV) with byte-identical
ESR words; `#54331` reports the same `CUDAGraph::replay` SIGSEGV on the same card class after 2.5-8
min of saturation, where `cudagraph_mode=PIECEWISE`, TRITON_ATTN, `expandable_segments` and
`max_num_seqs=128` all still crashed. MTP and higher utilisation are accelerants, not the cause. The
fix is **PR #50729** (`is_left_overlap`, merged 2026-08-17), first released in **v0.29.0**.

**So the fix is the version, not the utilisation.** Verified 2026-09-18: `is_left_overlap` is present
in `.venv-llm-029/.../vllm/v1/worker/mamba_utils.py`, that venv reports vLLM 0.29.0, and
`models.toml` makes it `[builds.stock]` — the build servedeck would launch carries the fix. The
unpatched 0.27.1 survives only as the rollback venv `.venv-llm` and is not a place to retreat to; if
0.29.0 itself faults, stop it and keep the evidence rather than loop or mitigate.

Two Xid **79 + 154** events ("GPU has fallen off the bus", node reboot required), 2026-09-11 21:24:41
and 2026-09-12 14:25:44, are a **different family** — the host power path, see
[../HOST.md](../HOST.md) — and must not be counted against 0.29.0. The 09-12 one killed the 27B with
`CUDA error: unspecified launch failure` and a cascade of `EngineDeadError`: consequences of losing
the card, not causes.

### Recognising it

A crash with no kernel Xid is not a clean bill of health. Check both places:

```
grep -a -n 'Segfault encountered\|EngineDeadError\|illegal memory' \
     "$(ls -t ~/Projects/local_llm/logs/qwen_server-*.log | head -1)"
journalctl -k -b all | grep -i xid | grep -vi r8169     # r8169 "XID 64a" is the NIC, not the GPU
```

`Segfault encountered` in `CUDAGraph::replay` and Xid 31 are the same race: one crashes the host
launching a graph, the other corrupts device state. Under servedeck the engine's output goes to the
journal (`servedeck log qwen27b`), not to `local_llm/logs/`.

## Evidence on vLLM 0.29.0

2026-09-11: 37 min of stress over two boots, 810 requests all HTTP 200, 5 saturation bursts (≥7.6 min
at the 16-request cap), prompts of 122k and 204k tokens answered correctly, CUDA graphs on, no Xid —
but the owner stopped the stress before the 60-min bar, so it is **partial**. 2026-09-12: one run of
**2 h 10 min** (12:15:40 → 14:25:46) at util 0.91, ctx 262,144, zero Xid 31 and zero `Segfault
encountered`, ended only by the card falling off the bus. That is the longest clean window on record
and still far short of an overnight soak. `bin/soak-27b-overnight.sh` (28,800 s default) closes that
gap, and **the owner runs it, not an agent**: it writes its own frozen `.config.27b-soak` and passes
it via `CONFIG_FILE` so whatever `.config` says cannot misdirect it, samples the kernel log every
60 s, and aborts if the port or GPU is busy. Known defect: it re-sends `reasoning_content`, which
this server does not return, so its conversations lose their reasoning.

## Build

A stock pip wheel with no code patches: on 2026-09-11 the install matched its package RECORD hashes on
24,416 of 24,417 files (the odd one is flashinfer's build-time `build_backend.py`), nothing untracked
under `vllm/`. All of this model's speed lives in launch flags, so a venv swap keeps it and upgrading
it cannot touch the GLM or Flash-Next forks ([../BUILDS.md](../BUILDS.md)). Two symlinks that the
`nvidia/cu13` wheel does not ship and FlashInfer's JIT assumes, both verified present 2026-09-18:

```
.venv-llm-029/lib/python3.13/site-packages/nvidia/cu13/lib64            -> lib
.venv-llm-029/lib/python3.13/site-packages/nvidia/cu13/lib/libcudart.so -> libcudart.so.13
```

Plus `flashinfer_jit_cache-0.6.18+cu130` installed `--no-deps` (present). If the venv is rebuilt,
recreate all three: a miss shows as a **failed request**, not a boot error, so a healthy-looking boot
proves nothing. Other prebuilt shapes may still be missing.

## Tried and refuted

- **`--enforce-eager`** (CUDA graphs off): the one mitigation that survives in every upstream report
  that tried it, at +23.6% time per round in #54331's measurements. The owner's ruling (2026-09-11)
  is that it is not acceptable, not temporarily and not as an end state. Root-cause instead.
- **util 0.47**: the Xid-free operating point for 9+ days after the August bisection, and still the
  old launcher's hardcoded default. Superseded twice — it is not a fix for the race (the 2026-09-11
  fault hit at 0.95 on 0.27.1, and the race is version-dependent), and the owner pinned 0.95. Three
  arms at 0.47 / 0.62 / 0.62+eager ran clean for 11-15 min each on 2026-09-09, which proves nothing:
  the historical triggers took 30 min to 30 h.
- **`--optimization-level 3`** (default is 2): regressed to ~60 tok/s from 140.4 and got slower over
  repeated calls instead of warming up. Not investigated further.
- **RoPE/YaRN context extension**: not attempted. 262,144 is the trained ceiling, not a VRAM limit.
- **Images**: the checkpoint carries vision weights (excluded from quantisation), but vLLM 0.27.1
  served it text-only — "treated as multimodal but has no registered multimodal processor". On 0.29.0
  that registry warning is **gone**; the only remaining line is `Disabled mm_prefix attention mode
  because multimodal inputs are configuration-disabled`, the consequence of passing no
  `--limit-mm-per-prompt`. `models.toml` carries `vision = true` and `max_output_tokens = 36000`, but
  no image has ever been served by this model on this box: treat vision as untested.

## First launch under servedeck — what to watch

1. **Offline resolution.** `HF_HUB_OFFLINE=1` is new for this model. The cache holds five snapshots
   and `refs/main` points at `319f741c` (13 files, 20.4 GiB), the one the 09-11 and 09-12 runs loaded,
   so it should resolve; a sibling snapshot (`58e2c08a`) holds no weights at all, so if `refs/main`
   ever moves there the launch fails fast with a local-files-only error. `servedeck doctor`'s
   `weights (qwen27b)` row reports that before a launch.
2. **The eviction**: the card is full while Flash-Next serves, so `switch`, not `start`.
3. **Utilisation actually applied**: confirm `--gpu-memory-utilization 0.95` and a KV pool near
   1.82 M tokens at ctx 262,144. `desired.json` (schema 3) records the launch settings and replays
   them on restart and reboot, so a wrong first launch repeats itself until corrected.
4. **The race**: confirm `kv_cache_dtype=fp8_e4m3` in the boot log, then watch for Xid 31 and
   `Segfault encountered`. Under `Restart=always` a fault becomes a restart, and the poll relaunches a
   wanted-but-absent model 3 times, 60 s apart with a notice each time — read the notices rather than
   reading a quiet box as a healthy one.
5. **No host-RAM preflight**: `[models.qwen27b]` has no `host_ram_gib` because this model pins no host
   memory, and its resident host use has never been measured. Measure it and set the key if a launch
   is seen to consume real host RAM.
6. **Power**: the live cap on 2026-09-18 is 375 W, below the 485 W boot service the crash notes
   assume, and 600 W is known unsafe. Check [../HOST.md](../HOST.md) before saturating this model.
