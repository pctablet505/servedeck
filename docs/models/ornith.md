---
title: "Ornith-1.5-122B on one GPU: MTP and prefix caching"
description: "Serving Ornith-1.5-122B NVFP4 on vLLM 0.32 with one 96 GB card: measured flags, KV capacity, the MTP sweep, the prefix-cache rule and failure modes."
---

# Ornith-1.5-122B-NVFP4 (`ornith`)

The main model from 2026-10-22, replacing Flash-Next at the owner's word. Since 2026-10-23 the weights
are dealignai's uncensored build, `dealignai/Ornith-1.5-122B-UNCENSORED-NVFP4`, revision
`e6a1c1785949cc6e1ccd8af766f0a6148f575e6b`, MIT, 75.7 GB in 50 shards (the owner asked for the
uncensored build "that has the best numbers"; it is the only one vLLM can serve, the rest are GGUF).
The registry key, the served name `Ornith-1.5-122B-NVFP4` and the old aliases are unchanged, so no
client config moved. The first build, `ornith-ai/Ornith-1.5-122B-NVFP4` revision `f049036f794622db`
(77.7 GB, 8 shards), is gone from the disk; `hf download` brings it back in about two hours.
The checkpoint's facts (architecture, quantisation, sampling) are in the comment above
`[models.ornith]` in `models.toml`; this page holds what was measured on this box. Sections dated
2026-10-22 were measured on the ornith-ai build.

The publisher's own numbers for the uncensored build (not re-measured here): 309 of 320 HarmBench
prompts complied (96.6%), and MMLU on 5,000 questions 82.38% against 84.07% for the NVFP4 base
(-1.69 points). Checked here on 2026-10-23: arithmetic, translation, code, an image, a tool call and
reasoning all answer; lock-picking and a violent villain monologue are answered without a refusal.
It introduces itself as Claude from Anthropic, as the model's training data does.

## Launch configuration

| Flag | Value | Why |
|---|---|---|
| build | `stock032` (vLLM 0.32.0) | Equal to 0.29.0 or slightly better on every measure below; `build = "stock"` (0.29.0) also serves it unchanged and is the fallback. |
| `--gpu-memory-utilization` | 0.94 | KV pool 12.9 GiB. 0.96 together with 16,384 batched tokens ran out of memory at boot (wanted 792 MiB, 478 MiB free). |
| `--max-num-seqs` | 32 | 16 queued the 17th request at 32 in flight; 64 added no throughput (the card is at its power cap) and shrank the pool by 20k tokens. |
| `--max-num-batched-tokens` | 8192 | At 16,384 and util 0.96 the KV pool came out at 12.0 GiB, below the 12.8 GiB of 8,192 at 0.94, and the boot then ran out of memory. |
| `--speculative-config` | MTP, 3 draft tokens | The checkpoint's own MTP layer. See the sweep below. |
| `--enable-prefix-caching`, `--mamba-cache-mode align` | — | `align` is the only prefix-caching mode 0.32 has; on 0.29, `all` and `align` measured the same. |
| `--kv-cache-dtype` | fp8 | The ornith-ai build shipped KV scales; the uncensored build has none (vLLM logs the accuracy warning and uses scale 1.0). Answers on the checks above were normal; a long-context quality check is not done. |
| `VLLM_USE_DEEP_GEMM`, `VLLM_MOE_USE_DEEP_GEMM` | 0 | The uncensored build's MTP experts are FP8 block-quantized. 0.32 sends them to DeepGEMM, whose JIT compile fails on this box (nvcc and the cu13 headers disagree), and the boot dies after the weights load. With it off they take the Triton path. |
| `--limit-mm-per-prompt` | `{"image":256,"video":0}` | Images on, as for Flash-Next (OpenWork's browser screenshots). |
| `--shutdown-timeout` | 30 | Drain in-flight requests on stop. |
| parsers | `qwen3` reasoning, `qwen3_xml` tools | The model card's. |

## Capacity

- **KV pool:** uncensored build 15.3 GiB = 990,700 tokens = 3.78 requests of 262,144 (0.32, 32
  sequences, util 0.94). The ornith-ai build: 12.9 GiB = 834,094 tokens = 3.18x; on 0.29 at 16
  sequences 827,285 tokens, 3.16x. The pool grew because the uncensored weights take 70.6 GiB
  against 72.5.
- **Attention block:** 4,240 tokens. The checkpoint keeps its linear-attention state in float32
  (`mamba_ssm_dtype`), and vLLM sizes an attention page to match one state page.
- **Weights:** 70.6 GiB loaded (uncensored build; 72.5 GiB for the ornith-ai build). Boot: 61 to 147 s
  on the ornith-ai build; 94 s through `servedeck switch` on the uncensored one.
- **Host RAM:** the engine's anonymous memory peaked at 5,570 MiB across 12 measured boots
  (`host_ram_gib = 8`). The unit's MemoryPeak reads about 54 GB, but all of that above 6 GB is the
  page cache of the weight files, which the kernel reclaims.

## Measured performance

2026-10-22, `vllm bench serve` against the unit's port, the GPU capped at 425 W. "Agent tasks" are
264 real task prompts (a system prompt and one coding task, median 411 tokens) answered with
reasoning on, at temperature 0.6, top-p 0.95, top-k 20 unless marked greedy. Raw results:
`~/Projects/t0-work/ornith-2026-10-22/bench/`. Boots of one config differ by several percent, so
treat differences under about 5% as noise.

| Measurement | Value |
|---|---|
| One stream, agent tasks, greedy | 235 tok/s (0.29: 230) |
| One stream, agent tasks, sampled | 194 to 230 tok/s |
| 32 agent tasks in flight | 667 to 728 tok/s in total, 46 to 48 tok/s each |
| 16 prompts of 30,000 tokens in flight | first token after 26.7 s (median), no preemptions; prefill-bound |
| One 60,000-token prompt, cold (0.29) | 9.0 s to first token (about 6,700 tok/s of prefill) |
| GPU power under load | 400 to 424 W: the 425 W cap is the limit |

### Uncensored build, 2026-10-23

Quick runs (3 to 64 prompts, same prompt set and flags, GPU at the 425 W cap), for direction only.
Raw results: `~/Projects/t0-work/ornith-uncensored-2026-10-23/bench/`.

| Measurement | Uncensored | ornith-ai build |
|---|---|---|
| One stream, agent tasks, sampled (8 prompts) | 190 tok/s overall, 4.34 ms per token, 2.02 tokens per MTP step | 194 to 230 tok/s, 1.9 to 2.1 per step |
| 32 agent tasks in flight (64 prompts) | 649 tok/s in total, 23.4 ms per token, 1.90 per step | 667 to 728 tok/s, 21.1 ms, 1.95 |

The MTP head loads and is accepted at the same rate, so the speed is within the noise of the first
build; the aggregate at 32 in flight is about 3% lower, which fits MTP experts on the Triton path.

### Draft tokens (MTP), 16 sequences, 0.29, ornith-ai build

| Draft tokens | One stream | 16 in flight | KV pool |
|---|---|---|---|
| 0 | 134 tok/s | 460 tok/s | 1,451,289 tokens |
| 1 | 169 | 650 | 906,581 |
| 2 | 212 | 654 | 852,841 |
| **3** | **230** | **668** | 827,285 |
| 4 | 215 | 462 | 783,155 |

Each draft token also reserves state pages per sequence, which is why the pool shrinks as they rise.

## Prefix caching

A request whose first S tokens match an earlier one's is served `(floor(S / 4,240) - 1) * 4,240`
tokens from the cache: state exists only at block boundaries, and the MTP drafter drops the last
matched block (the same rule as Flash-Next's, [flashnext.md](flashnext.md#prefix-caching), with a
larger block). So **nothing below 8,480 shared tokens is reused**. Measured: two 16,384-token
prefixes, each with 14 followers, gave 43.9% of prompt tokens from the cache, which is 8,480 of each
follower's 16,896.

## Failure modes

- **No `refs/main` in the hub cache.** A checkpoint fetched with `hf download --revision <sha>` has
  none, and every launch is offline without `--revision`, so the engine dies in 4 s with
  `LocalEntryNotFoundError`. `servedeck doctor`'s `weights (ornith)` row now fails on it; the fix is
  `printf <sha> > <repo dir>/refs/main`.
- **DeepGEMM compile at boot (uncensored build).** Without the two `VLLM_*DEEP_GEMM` variables set
  to 0 the engine dies about 40 s in, after the weights load, in `deepgemm_post_process_weight_scale_block`
  with `CUDA compiler and CUDA toolkit headers are incompatible`. Only the MTP layer reaches that path.
- **A registry edit needs a daemon restart.** `servedeck.service` holds the registry in memory;
  after editing `models.toml`, `systemctl --user restart servedeck` before `servedeck switch`, or the
  unit starts with the old repo name (the 2026-10-23 first attempt died in 5 s on a missing snapshot).
- **FlashInfer kernels compiled at boot (`stock032`).** A missing `libcudart.so` link fails the XQA
  compile after the weights load (`ld: cannot find -lcudart`); without the prebuilt sm120f kernels
  the JIT stops on `CUDA compiler and CUDA toolkit headers are incompatible`. Both are set up by
  `builds/stock032/build.sh` and checked by its `check.sh`.

## Not done

- `--mamba-ssm-cache-dtype bfloat16` would halve the 4,240-token block (finer prefix reuse, smaller
  per-sequence state), as it did for Flash-Next. Untried, because the checkpoint asks for float32
  state and the change needs a quality check first.
- servedeck's parallel-agent recommendation (`parallelism.py`) is fitted to Flash-Next's per-request
  cost and says one full-context request fits, against the engine's 3.18. It needs Ornith's own
  measured points.
