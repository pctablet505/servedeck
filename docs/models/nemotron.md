---
title: "Nemotron-3.5-Super-VL-120B on one GPU: KV at 1M context"
description: "Serving an abliterated NVFP4 Nemotron-3.5-Super-VL-120B on one 96 GB card: launch flags, KV capacity at 1M context, what was checked."
---

# Nemotron-3.5-Super-VL-120B-A12B, abliterated (`nemotron-abl`)

Wired 2026-10-24 in place of ThinkingCap, whose weights were deleted the same day at the owner's word.
Weights: `armin1/Nemotron-3.5-Super-VL-120B-A12B-NVFP4-abliterated`, revision
`1e1c8ecf1976ee0ec387c114c1964a6f3506e804`, 75.06 GB in 88 shards, OpenMDW 1.1 (the base model's
licence). It is an edit of `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`; the publisher does
not disclose the method. **Uncensored: not for automated pipelines.**

The checkpoint's facts (layers, experts, quantisation) are in the comment above
`[models.nemotron-abl]` in `models.toml`; this page holds what was run on this box.

## Launch configuration

| Setting | Value | Why |
|---|---|---|
| build | `stock032` (vLLM 0.32.0) | The qwen38next fork has no `NemotronH_Omni_Reasoning_V3`. 0.29.0 registers it too; only 0.32.0 was booted. |
| `ctx` | `native` (1,048,576) | Read from `llm_config.max_position_embeddings`; until 2026-10-25 servedeck looked only at `text_config`, the launch failed, and the entry had to be pinned to the number. |
| `--trust-remote-code` | on | Needed for the HF config class (`auto_map`); vLLM uses its own model code. The repo's `.py` files were read first. |
| `--gpu-memory-utilization` | 0.90 | A first guess; it booted. Not tuned. |
| `--max-num-seqs` | 16 | A first guess. 32 or 64 raise the total when more than 16 are in flight (below); neither was tried together with 5 draft tokens. |
| `--speculative-config` | MTP, 5 draft tokens | The checkpoint's own MTP head. One request 110 -> 312 tok/s (below). |
| `--max-num-batched-tokens` | 8192 | The card's recipe. |
| `--enable-prefix-caching`, `--mamba-cache-mode align` | — | Needed for the KV offload on a hybrid model; `align` is the only prefix-caching mode 0.32 has. |
| `--kv-offloading-size` | 12 | The offload connector started and mapped a 12.87 GB buffer. Never filled, so eviction to host RAM is unexercised. |
| `--limit-mm-per-prompt` | `{"image":16,"video":0}` | Images on, video off (the card's video flags were not used). |
| `VLLM_USE_DEEP_GEMM`, `VLLM_MOE_USE_DEEP_GEMM` | 0 | Set as for ornith, where DeepGEMM's JIT fails on this box. Not tested without them. |
| parsers | `nemotron_v3` reasoning, `qwen3_coder` tools | The card's. |

## Capacity

2026-10-24 boot, 165 s from the start request to ready, the weight files freshly downloaded:

- **Weights:** 68.45 GiB loaded.
- **KV pool:** 13.29 GiB = 3,272,828 tokens = 3.12 requests of 1,048,576 tokens. Only 8 of 88 layers are attention (2 KV heads of 128), so a million tokens cost little; the Mamba state is the other consumer.
- **Attention block:** 4,224 tokens, sized so one attention page equals one Mamba state page.
- **KV dtype:** fp8 (e4m3), chosen from the checkpoint's quantisation config. The backbone's attention layers have no k/v scales (only the MTP layer does), so vLLM logs its warning and uses scale 1.0, as on the uncensored Ornith build. Answers below were normal; no long-context accuracy check was run.
- **GPU in use after boot:** 87,552 MiB.
- **With the MTP head (5 draft tokens):** KV pool 2,180,132 tokens (2.08 requests of 1,048,576); the drafter's weights and its own cache take about 2 GiB. The pool shrinks by about 3% for each further draft token.

## Tuning sweep, 2026-10-25

36 variants, each a flag change in `models.toml`, a servedeck reboot and a quick bench: `vllm bench serve`
over the chat endpoint with the 264 real agent prompts of the Ornith bench (ShareGPT file, at most 512
output tokens, temperature 0.6, top-p 0.95, top-k 20, seed 0), 6 prompts one at a time, 48 at 16 in
flight, 96 at 32. One run per variant, so the numbers show direction, not precision. Noise: the
no-MTP baseline repeated at 110 and 110 tok/s (one request) and 670 and 668 (16 in flight); with MTP,
near-identical configurations read 1,123 to 1,396 at 16 in flight, so treat 10% as noise there. The GPU
sat at its 425 W cap in every run (mean 379 to 417 W). Raw data: `~/Projects/t0-work/nemotron-abl-2026-10-24/bench/`.

| Configuration | One request, tok/s | 16 in flight | 32 in flight | KV pool, tokens |
|---|---|---|---|---|
| no MTP | 110 | 670 | | 3.27M |
| no MTP, 32 sequences | 110 | 671 | 1,016 | 3.25M |
| no MTP, 64 sequences | 111 | 681 | 960 (64 in flight: 1,356) | 3.20M |
| MTP, 1 draft token | 162 | 906 | | 2.45M |
| MTP, 2 | 204 | 1,232 | | 2.37M |
| MTP, 3 | 249 | 1,396 (three other 3-token runs: 1,123 to 1,266) | | 2.32M |
| MTP, 3, 32 sequences | 246 | 1,266 | 1,607 | 2.25M |
| MTP, 3, 64 sequences | 239 | 1,123 | 1,585 (64 in flight: 1,620) | 2.15M |
| MTP, 4 | 287 | 1,145 | | 2.24M |
| **MTP, 5 (in `models.toml`)** | **312** | **1,232** | | 2.18M |
| MTP, 6 | 265 | 1,263 | | 2.12M |
| MTP, 7 | 286 | 1,185 | | 2.07M |
| MTP, 8 | 284 | 1,009 | | 2.02M |

- **MTP is the lever.** Acceptance on agent prompts is 3.7 of 4 at 3 draft tokens and 5.2 of 6 at 5. One request keeps gaining up to 5 draft tokens and then flattens; the total at 16 in flight peaks near 3 and falls slowly. 5 favours the one-request case, which is what an interactive session sees.
- **More sequences scale the total** once more than 16 requests are in flight (no MTP: 670 -> 1,016 -> 1,356 at 16, 32, 64 sequences and as many in flight; with MTP 3: 1,607 at 32, flat at 64) and cost nothing for one request.
- **The default kernels are the best ones.** FlashInfer attention beat `TRITON_ATTN` or tied it. FlashInfer CUTLASS MoE beat vLLM's own CUTLASS MoE (92 tok/s, -16%); Marlin MoE (activations kept in bf16) read 115 and 701 without MTP, inside the noise, and 1,172 at 16 in flight with MTP 3, with a smaller pool. FlashInfer autotune off cost 25% for one request (83 against 110, and 192 against 249 with MTP 3).
- **Did not boot:** `--moe-backend b12x` and `flashinfer_cutedsl` ("kernel does not support current device cuda"), `flashinfer_b12x` (CUDA out of memory during boot), `--mamba-backend flashinfer` (FlashInfer's JIT cannot find the CUDA headers), `--attention-backend FLASH_ATTN` with MTP (engine init failed; the cause was not read).
- **No effect on speed:** KV offload on or off, prefix caching on or off, `--performance-mode` throughput or interactivity (only 16 sequences tried), async scheduling forced on (off: -17%), CUDA graph capture caps of 32 and 64, `VLLM_TRITON_FORCE_FIRST_CONFIG`, 16 Triton JIT threads, bf16 KV (the same speed with half the pool: 1.74M tokens, 1.27M with MTP 3; the fp8 cache with scale 1.0 stays), 16,384 batched tokens (720 against 670 at 16 in flight, inside the noise, with a smaller pool).

**Boot time follows cache warmth and CPU load, not flags.** With the compile caches warm and the box quiet the same
config reached ready in 50 to 60 s (engine start to listening, plus about 15 s of process start). The first boot of a
new flag set took 125 to 155 s: `torch.compile` 10 to 20 s, and the Mamba SSD Triton warmup 29 to 32 s when other
sessions were loading the CPUs, about 1 s when they were not. Turning prefix caching and offload off did not shorten
a warm boot (50 s against 55 s), and the one flag that cut boot time, FlashInfer autotune off, cost too much speed.
Weights load in 16 to 27 s from the page cache.

## Checked

- **Reasoning split:** `/v1/chat/completions` returns `reasoning` and a separate `content` ("391" for 17 × 23); no `</think>` in the content.
- **Tool calls, chat:** `tool_choice: auto` returns a parsed `tool_calls` entry with valid JSON arguments; no `<tool_call>` text leaks into the content.
- **Tool calls, Codex's wire:** `/v1/responses` with `stream: true` emits `reasoning_text` deltas, then a `function_call` item with arguments; a second turn carrying a `function_call_output` returns a `reasoning` item and a final message.
- **Gateway:** `/v1/models` on :8010 lists the registry id, `nemotron-abl`, `nemotron` and `main`; a request for `main` is answered by this model.
- **Speed:** see the sweep above; the first measurement, one stream without MTP, was 111.7 tok/s.
- **Compliance:** a locksmith-style lock-picking question is answered, not refused.

## Not checked

The card's own numbers (refusal rate 98% to 2%, quality unchanged, KL 0.0157 against the base) are the
publisher's, from a private harness, and were not re-measured. Not run here: refusal rate on any
set, quality against the base model, vision, MTP 5 with 32 sequences, the 3-draft-token run repeated to size
the noise under MTP, a request near 1,048,576 tokens, KV eviction to the host buffer, utilisation tuning,
`--performance-mode` above 16 sequences, vLLM 0.29.0. Speculative decoding and batching can change token
choices at temperature 0 between runs; the greedy outputs of four short prompts agreed with the no-MTP run
from 68% to 100% of their length for settings that should be numerically identical, so that comparison
is noise, not a quality measure.
