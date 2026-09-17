# Flash-Next

`mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4`, registry key `flashnext`, port 8001, main slot — the
model that serves this box every day, and what the gateway aliases `main` and `local` resolve to
unless an operator switched slots. Start it with `servedeck switch flashnext`. Served ids:
`Qwen3.8-Flash-Next-Uncensored-NVFP4`, `qwen38-flash-next`, `flashnext`. Registry section
`[models.flashnext]` in [../../models.toml](../../models.toml); build `qwen38next`, a fork
([../BUILDS.md](../BUILDS.md)).

## The checkpoint

NVFP4, 48 layers, `Qwen4ExpForConditionalGeneration`, with an abliterated refusal direction — it
will comply with requests a stock checkpoint refuses. Research use. The owner chose this graft over
the RadixArk base, deleted from disk on 2026-08-29.

**Native context 262,144.** `max_position_embeddings` with `rope_type: default` and no YaRN, so
that is the *model's* ceiling, not a memory limit: freeing VRAM buys concurrency and never context.
`/v1/models` reports 262144 for all three ids (2026-09-18).

**A 47.68 GiB PLE n-gram table in host RAM.** 43 `model-plefp8-*.safetensors` shards,
51,200,268,090 bytes, FP8 E4M3 with one global scale, held by a sibling `PleOffloadWorker` process
and prefetched asynchronously. Without `VLLM_PLE_CPU_OFFLOAD=1` the model does not fit on one card.

**Vision on.** The tower is 0.84 GiB and all BF16 while the rest is NVFP4 (safetensors headers,
2026-09-09). Two images per request, verified across five tests including a 2920x1944 screenshot
and a correct HTTP 400 on a third image (2026-09-10).

The checkpoint is 125.91 GiB on disk against 79.31 GiB of weights resident, so the "disk x 1.01"
estimator that works for ordinary checkpoints is 37% wrong here: the PLE table never reaches the
card. Only 12 of the 48 layers hold a growing KV cache (`full_attention_interval=4`); the other 36 are
Gated DeltaNet linear attention with fixed per-sequence state. Hence ~30.4 KiB/token against the
27B's 37.99 (2026-09-03; not re-measured). The rate is **not** context-invariant — 30.39 KiB/token
at `max_model_len` 262144 but 33.85 at 131072, because vLLM's block sizing follows the configured
length. Never reuse a rate measured at another context.

## Launch configuration, flag by flag

What `models.toml` renders, which is what the next launch gets. The engine serving right now
started 2026-09-17 22:58:27, before `--shutdown-timeout 30` was added, so that one flag is in the
table and not in the live argv; everything else matches. (`ps -eo args | grep '[v]llm serve'` —
with the bracket, or the grep matches its own command line.)

| Flag | Value | Why it is there |
|---|---|---|
| `--max-model-len` | 262144 | The model's own ceiling; owner rule is full native context. |
| `--gpu-memory-utilization` | 0.96 | Pinned in the registry, not computed. See below. |
| `--max-num-seqs` | 16 | Parallel agents. At 1 the box gave 141 aggregate tok/s; at 16, ~1,040 (2026-09-03). |
| `--max-num-batched-tokens` | 8192 | Memory-locked here; 16384 and 32768 do not boot at 262k ctx. |
| `--limit-mm-per-prompt` | `{"image":2,"video":0}` | Images on. Costs 0.81 GiB of weights plus 0.49 GiB of profiling peak and encoder cache (encoder budget 16,384 tokens). |
| `--kv-cache-dtype` | auto (bf16) | fp8 is hard-blocked by the QSA kernel. |
| `--enable-prefix-caching` | — | Effective granularity 832 tokens; see below. |
| `--speculative-config` | mtp, 3 draft tokens | 4 raises acceptance and not throughput; 5 asserts. |
| `--mamba-ssm-cache-dtype` | bfloat16 | Lowers the GDN recurrent state from fp32, shrinking the derived attention block 1600 → 832. This, not `--language-model-only`, bought +18,863 KV tokens. |
| `--long-prefill-token-threshold` | 1024 | Unsettled; see below. |
| `--watermark` | 0.10 | Measured: preemptions at 8k prompts 4 → 0 at N=16 and 11 → 1 at N=32 (2026-09-10). |
| `--kv-offloading-size` | 40 | 40 GiB of pinned host RAM for evicted contexts; see below. |
| `--shutdown-timeout` | 30 | Without it, 40 GiB of host RAM leaks on every restart. Added 2026-09-18; arrives on the next launch. |
| `--reasoning-parser` | qwen3 | Splits thinking out of `content`; see *The reasoning field*. |
| `--tool-call-parser` | qwen3_coder, with `--enable-auto-tool-choice` | Required for agentic clients. |
| `--tensor-parallel-size 1`, `--distributed-executor-backend mp` | — | One card; the PLE worker is a spawned sibling. |

`--language-model-only` is **not** in the live argv and should not come back as a way to buy KV:
the KV-memory difference between the configs that flag distinguished is 0.12 GiB (2026-09-09).

`--long-prefill-token-threshold 1024` was added to stop a reported crash when several long prefills
are padded to a common rectangle, but the adversarial verifier could not reproduce that crash, and
the flag is global and costs roughly 20% on long-prompt prefill. Kept because a crash is worse than
20%. The durable fix, not done, is a variable-length convolution in
`ple_layer.py::_short_conv_dilated_prefill_batched`, which today pads every prefill in a batch to
the longest chunk so the allocation grows as (batch x longest).

`--shutdown-timeout 30` exists because vLLM's default of 0 means abort: the API server SIGTERMs the
engine and force-kills it at once, so the engine never reaches `CPUOffloadingWorker.shutdown()`,
`cudaHostUnregister` never runs, and `/dev/shm/vllm_offload_*.mmap` is never unlinked — 40 GiB of
host RAM orphaned on *every* restart, not only on a crash. 30 s also drains in-flight requests
instead of dropping an agent's generation mid-stream. The unit's `Restart=always` and
`TimeoutStopSec=120` are described in [../ARCHITECTURE.md](../ARCHITECTURE.md).

### Why 0.96, and not 0.98 or 0.95

0.96 is the highest utilisation this model has served at without a CUDA OOM under concurrency, and
it is pinned in the registry so a launch on an empty card cannot drift upward: `compute_util` uses
the pinned value and refuses when the card cannot fit it, naming what holds it. 0.98 boots and then
dies under concurrency with a CUDA OOM (wanted 842 MiB, 880 MiB free), measured 2026-09-03 — and
free-VRAM arithmetic on an idle card yields exactly 0.98, which is why the pin exists. Two launches
on 2026-09-17 (22:43, 22:45) did run at 0.98 and reported 9.59 GiB of KV, 350,043 tokens, 1.34x:
69,230 tokens more than 0.96 gives, given up on purpose. 0.95 in the other direction does not boot
at all with images on — the KV pool falls to 6.68 GiB while a full 262,144-token request needs
7.17 GiB, and the server refuses (2026-09-10). The safer lever for the last gigabyte is explicit
KV sizing, not a higher fraction: at the 0.96 boot vLLM offered `--kv-cache-memory=10856775168`
(10.11 GiB) to fully use the card. Untried under load.

### Environment

| Variable | Value | Why |
|---|---|---|
| `VLLM_PLE_CPU_OFFLOAD` | 1 | The single flag that makes one-GPU serving possible. |
| `VLLM_PLE_FP8_CHECKPOINT` | 1 | Must match the PLE table's dtype. Wrong either way, loading dies on `ngram_embedding.weight_scale`. |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` | Fragmentation under the PLE handoff. |
| `TORCH_CUDA_ARCH_LIST` | `12.0f` | SM120. |
| `VLLM_USE_FLASHINFER_SAMPLER` | 0, from `[defaults.env]` | No CUDA toolkit on the host PATH, so FlashInfer's JIT top-k/top-p sampler cannot compile. |
| `HF_HUB_OFFLINE` | 1, from `[defaults.env]` | A gated repo: the first v2 launch went online, asked the hub about it and died on a 401. Independently, 17 of 72 recorded boot exits in the coldstart investigation were DNS failures, and this flag was verified to remove that class. |

`kernel.yama.ptrace_scope` must be **0** or this model cannot start. The PLE handoff passes a CUDA
IPC handle for the GPU output buffer, torch transfers it with `pidfd_getfd()`, that syscall needs
`PTRACE_MODE_ATTACH`, and the two processes are siblings — Ubuntu's default of 1 permits attach
only to descendants and startup dies with `pidfd_getfd: Operation not permitted`. The old
`serve.sh` asked for sudo at every launch; `/etc/sysctl.d/90-servedeck.conf` now pins it to 0 at
boot, so the model launches unattended, and the same sysctl is what lets the offload reaper see a
sibling engine's mappings ([../HOST.md](../HOST.md), [../ARCHITECTURE.md](../ARCHITECTURE.md)).

### The fork patch this configuration needs

`--kv-offloading-size` does not work on this architecture with stock fork code. Three boots failed
first, each a different assert in the offloading connector, all traceable to one thing: the QSA
raw-key ring (`CircularBufferSpec`, one 8-token block per request, `prefix_cacheable=False`) is not
a `FullAttentionSpec`, and no match unit divides both it and the 832-token hash. The fix is a `skip`
flag on `GroupOffloadConfig` for non-prefix-cacheable groups, folded into
`builds/patches/qwen38next/working-tree-tracked.patch` with the standalone diff at
`builds/patches/flashnext-kv-offload-skip-qsa-ring.patch`. Confirm it is in force from two boot-log
lines: `EAGLE/MTP draft attention groups [0, 1, 2, 3, 4, 5] detected` and `groups [1] are not
prefix-cacheable and will not be offloaded`.

## Capacity

From the live boot of 2026-09-17 23:00 at util 0.96, images on, KV offload on: model loading
79.31 GiB in 82.6 s; weights + non-torch on the card 81.7 GiB; peak activation 1.78 GiB; CUDA
graphs 0.66 GiB; **available KV cache 7.69 GiB = 280,813 tokens = 1.07x concurrency at 262,144
tokens per request.** Two older figures still in circulation are superseded by that line, because
offload's own metadata costs KV: 304,653 tokens / 1.16x (text-only, no offload) and 278,479 tokens
/ 1.06x (images on, no offload).

## Measured performance

Always measure **warm**: the first passes after a restart pay FlashInfer autotune and CUDA-graph
capture, and an early baseline read 365 tok/s prefill (12x too low) and 332 tok/s aggregate (3x too
low) from that alone. Two boots of an identical config differ by 16-24% here (121.8 vs 141.6
single-stream, 699 vs 919 aggregate, 2026-09-09), so any A/B smaller than that is noise; on-boot
interleaving is impossible, so alternate at least three boots per arm on a quiet host.

| Measurement | Value | Conditions |
|---|---|---|
| Decode, single-stream warm | 92-132 tok/s | 2026-09-03 to 2026-09-17; not re-measured |
| Decode, aggregate | ~1,040 tok/s | 16 concurrent at `--max-num-seqs 16` (2026-09-03) |
| Prefill | 27.1 s for 253k tokens, ~9,300 tok/s | uncached (2026-09-03; not re-measured) |
| Needle at 220k, depths 10/50/93% | 3/3 pass | uncached (2026-09-03) |
| Cold prefill TTFT, 35k prompt | 3.78 s | 2026-09-17 |
| GPU prefix-cache hit, same prompt | 0.22 s | 2026-09-17 |
| Reload from host RAM after eviction | 0.36 s, ~1 GB moved | 2026-09-17 |
| Boot time | 100-370 s | the 81 s once recorded was one warm boot |

Cumulative counters from the engine running since 2026-09-17 23:00, read 2026-09-18 on a mixed
agent workload: 218,049 draft steps, 654,147 draft tokens, 395,933 accepted — 60.5% acceptance,
2.82 accepted tokens per step, per-position 75.3 / 58.9 / 47.4%. MTP-3 measured 3.11 accepted per
step (rate 0.833) on prose in 2026-09-03 and 1.557 under 16-way mixed load in 2026-09-10, so a
lifetime figure between them is what a mixed workload should look like. Prefix-cache hit rate over
the same window 30.4% (12,791,168 of 42,022,241 queries), with 9 preemptions.

Under decode the card ran at 94.5% utilisation and 379 W against a 485 W cap (2026-09-03); the cap
is now 375 W, so check [../HOST.md](../HOST.md) before comparing power figures. The "GPU is
underutilised" premise is refuted — the 6% / 40 W reading behind it was an idle GLM server.

### Concurrency is admission-bound, not compute-bound

Measured 2026-09-10 on an idle box, FP8 PLE table, images on, reasoning on, against a
280,813-token KV pool.

| Prompt shape | Honest max concurrency | Peak aggregate | KV cost of one request |
|---|---|---|---|
| ~500 tokens | 16 | 937 tok/s at N=16 | 5% of the pool |
| 8k tokens | 8 | 500 tok/s at N=8 | 10.5% |
| 30k tokens | 4 | 227 tok/s at N=4; N=8 goes backwards | 23.3% |
| 105k tokens | 1 | — | 67% |
| 245k tokens | 1 | — | — |

Offering more concurrency than this table allows never helps; it converts throughput into
preemption. The noise band on aggregate throughput is ±8%. At full context the config serves one
agent at a time.

### Prefix caching

The effective match unit is **832 tokens** and `--prefix-match-unit 208` does not change it (see
*Refuted*). With the usual "never cache the final block" rule, a growing agent conversation needs
more than 1,664 shared tokens before its first hit and then re-prefills up to 831 tail tokens every
turn. That is the real prefix-caching limit here, and making 208 effective needs an upstream root
cause.

### KV offload

`--kv-offloading-size 40` parks evicted contexts in 40 GiB of pinned host RAM: native CPU backend,
LRU, 832-token chunks, dedicated stream, `store_threshold` 0. Those defaults are right for agents;
only the pool size is a real knob. The payoff is the 0.36 s reload against 3.78 s of cold prefill
above, with decode unchanged; live transfer rate 2026-09-18 was one 655,220,736-byte load in
0.0117 s, about 56 GB/s, consistent with pinned-host DMA. The cost is one
`/dev/shm/vllm_offload_<engine-id>.mmap` of 42,945,576,960 bytes, created at boot and unlinked only
by a graceful engine exit — what `--shutdown-timeout 30` protects and what the offload reaper
cleans up after when protection fails. Outputs after a reload differ textually from the cold
answer, but so do two consecutive GPU-cache hits, and first-token logprobs vary in the same band
across cold, hit and reload: baseline MTP and batching nondeterminism, not a reload defect.

## Host RAM

This model commits roughly 114 GiB while serving (2026-09-17): ~48 GiB of PLE table, the 40 GiB
pinned KV buffer, and the loader's transient copies. Live 2026-09-18 the `PleOffloadWorker` holds
50,940,620 KiB of anonymous RSS (48.58 GiB) and 51,487,120 KiB VmRSS (49.10 GiB) — the ~0.9 GiB
above the 47.68 GiB table is mapping and allocator overhead; the API server adds ~2.4 GiB.
`models.toml` sets `host_ram_gib = 95` and `start()` refuses the launch below that, because there
is no swap here: overshooting is an OOM kill of the session, not a slowdown. The old
`serve-abliterated.sh` set `vm.swappiness=10` because swap faults on PLE lookups cost ~35% of
decode throughput; there is no swap now, so that preflight replaces the mitigation rather than
inheriting it.

## The reasoning field

**This engine emits `reasoning`, never `reasoning_content`.** Verified live 2026-09-18 on port
8001: message keys are `annotations`, `audio`, `content`, `function_call`, `reasoning`, `refusal`,
`role`; `reasoning` is populated, `reasoning_content` absent, `content` clean with no `<think>`
leak. `SETUP.md`'s warning that output goes to `reasoning_content` is wrong for this build.

The gateway mirrors it: through `http://127.0.0.1:8010/v1` the same response carries both
`reasoning` and an identical `reasoning_content` (verified 2026-09-18), on the JSON and SSE paths
alike, driven by `[models.flashnext.reasoning] mirror_content = true`. Mirroring is off on
`/v1/responses`, where reasoning is already a first-class output item Codex round-trips
structurally. A client that talks to port 8001 directly and reads `reasoning_content` sees no
thinking and cannot re-send it — the exact mechanism behind GLM's multi-turn amnesia. Point
chat-completions clients at the gateway, never at the model port; the standalone
`flashnext-reasoning-proxy.service` on :8005 that used to do this is retired. Server *input*
accepts a re-sent `reasoning_content` and normalises it to `reasoning`, so the gap was only ever
output-side. See [../CLIENTS.md](../CLIENTS.md).

## Failure modes and their signatures

**Empty answer, whole budget billed.** A too-small `max_tokens` at reasoning effort `xhigh` returns
both `content` and `reasoning` empty with usage reporting the full budget spent (1,200 billed and
empty; 6,000 fine), because generation truncates mid-`<think>` and leaves the qwen3 parser an
unterminated block. It looks exactly like a dead server; the gateway's minimum-output floor exists
for this class. `max_output_tokens` is 32000 here.

| Signature | Cause |
|---|---|
| HTTP 400 `Unexpected reasoning effort ultra. Supported types are xhigh (default), medium, and low.` | Only those three exist. For scale, xhigh against low on one task measured 5,329 tokens in 36.1 s versus 1,266 in 10.6 s. |
| `pidfd_getfd: Operation not permitted` during startup | `kernel.yama.ptrace_scope` is not 0. `servedeck doctor` has a row for it. |
| A 401 from the hub minutes into a launch | `HF_HUB_OFFLINE` was not set. With it set, a missing shard instead fails fast with a local-files-only error, and doctor's `weights (flashnext)` row catches that before a launch starts. |
| Load dies on `ngram_embedding.weight_scale` | `VLLM_PLE_FP8_CHECKPOINT` disagrees with the PLE table on disk, in one direction or the other. |
| 40 GiB missing from host RAM after a stop | The engine was force-killed before it could unlink its offload buffer. Look for a `vllm_offload_*.mmap` in `/dev/shm` with no live engine behind it; the reaper removes only buffers it can prove are orphaned. |
| A log line saying `max_num_batched_tokens=2048` | That is the `PleOffloadWorker`'s own scheduler, not the engine's. The `APIServer` non-default-args line says 8192, and 8192 is served. |

## Refuted, and do not retry

**`--kv-cache-dtype fp8`, any spelling.** `NotImplementedError: Qwen4Exp QSA requires a BF16 main
KV cache`, at `src/vllm/models/qwen4_exp/nvidia/qsa.py:108` and `:185`. The single biggest
unavailable concurrency lever: it would roughly double the four-concurrent ceiling at 30k prompts.

**`--max-num-batched-tokens` 16384 or 32768 at 262k ctx.** No memory left for KV; the boot fails.

**`num_speculative_tokens 4`.** Acceptance length rises 2.75 → 4.30 and throughput does *not*
improve (median 128 vs 148 tok/s), because the QSA backend falls back to rebuilding attention
metadata between draft steps. 5 fails the QSA assert; legal K is 0, 1, 2, 3, 4, 6, 9, 10, 11, 12
(`qsa_cache.py:779-784`: capacity `4*cdiv(4+K,4)` must divide the block size). Hence the pinned 3.

**`--prefix-match-unit 208`.** Accepted, logged, and inert. Proven by a three-point hit-count
discriminator against the live server: shared prefixes of 500 / 1,000 / 2,000 tokens gave
0 / 0 / 832 hits, which fits an 832-token unit exactly and refutes 208 at all three points. The
earlier "93.8% hits, TTFT 0.65 s → 0.18 s" result was real but came from an *identical* prompt,
which hits at any granularity. It is also incompatible with offload. Dropped from the registry.

**MARLIN NvFp4 MoE.** Identical to FlashInfer CUTLASS here, 624-656 against 640-651 tok/s. The GLM
MARLIN finding does not transfer: Qwen's experts are 512x640, GLM's 288x2048.

**Offloading experts to host RAM.** Routed experts are 68 GiB of the 82 GiB on the card and four
full contexts need ~30 GiB of KV, so ~22 GiB of experts would have to leave. PCIe at 52 GB/s with
no compute overlap puts single-stream at ~50 tok/s random and ~80 tok/s with per-layer hot-expert
pinning, and prefill would stream all 22 GiB per chunk. Do not re-propose without a measurement.

**Halving context to 131k.** No prefill or decode gain.

**Fused multi-step draft decode for QSA.** Reverted as inconclusive, not as a regression: correct
output, no CUDA errors, acceptance 1.47 against 1.50, but 126.6 against 141.6 tok/s single-stream
and 724 against 919 at 16-way — inside the box's own 16-24% boot-to-boot band. Do not re-run
without a noise-controlled design.

**The BF16 PLE table.** Superseded by FP8 on 2026-09-10 and deleted. The switch halved worker RSS
(96.27 → 48.59 GiB), cut host RAM while serving from 120-122 GiB to 73 GiB, model load from 152.8 s
to 81.0 s and PLE offload load from ~60 s to 11 s, and left KV unchanged (7.63 → 7.68 GiB). It is
lossless in the arithmetic vLLM performs — the BF16 table was itself a dequantised FP8 table, so
recovering the grid scale round-trips bit-identically, and a float32 re-derivation's 9.16e-05 "max
error" is bf16 representation granularity, not conversion loss. Do not re-verify in float32 and
conclude it is lossy. Both index sidecars were renamed `*.STALE-ple-bf16-deleted`, which
deliberately disarms `ple_fp8/rollback.sh`; the only recovery path is a targeted re-download of the
43 BF16 shards (95.37 GiB), documented in `vllm-qwen38next/ple_fp8_cleanup/RECOVERY.md`, and
restoring it buys no accuracy.

**Open caveat on a shipped flag.** `--mamba-ssm-cache-dtype bfloat16` lowers the GDN recurrent state
from fp32 and coherence was spot-checked only. Drop it first if long-context quality ever looks
wrong — and re-measure KV, because it is worth +18,863 tokens.
