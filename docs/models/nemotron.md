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
| `ctx` | 1,048,576, a number | `"native"` reads `max_position_embeddings` at the top level or in `text_config`; this config keeps it in `llm_config`, and the launch failed with "config.json has no max_position_embeddings". |
| `--trust-remote-code` | on | Needed for the HF config class (`auto_map`); vLLM uses its own model code. The repo's `.py` files were read first. |
| `--gpu-memory-utilization` | 0.90 | A first guess; it booted. Not tuned. |
| `--max-num-seqs` | 16 | A first guess. Not tuned. |
| `--max-num-batched-tokens` | 8192 | The card's recipe. |
| `--enable-prefix-caching`, `--mamba-cache-mode align` | — | Needed for the KV offload on a hybrid model; `align` is the only prefix-caching mode 0.32 has. |
| `--kv-offloading-size` | 12 | The offload connector started and mapped a 12.87 GB buffer. Never filled, so eviction to host RAM is unexercised. |
| `--limit-mm-per-prompt` | `{"image":16,"video":0}` | Images on, video off (the card's video flags were not used). |
| `VLLM_USE_DEEP_GEMM`, `VLLM_MOE_USE_DEEP_GEMM` | 0 | Set as for ornith, where DeepGEMM's JIT fails on this box. Not tested without them. |
| parsers | `nemotron_v3` reasoning, `qwen3_coder` tools | The card's. |

No `--speculative-config`: the checkpoint has an MTP layer, which was not tried.

## Capacity

2026-10-24 boot, 165 s from the start request to ready, the weight files freshly downloaded:

- **Weights:** 68.45 GiB loaded.
- **KV pool:** 13.29 GiB = 3,272,828 tokens = 3.12 requests of 1,048,576 tokens. Only 8 of 88 layers are attention (2 KV heads of 128), so a million tokens cost little; the Mamba state is the other consumer.
- **Attention block:** 4,224 tokens, sized so one attention page equals one Mamba state page.
- **KV dtype:** fp8 (e4m3), chosen from the checkpoint's quantisation config. The backbone's attention layers have no k/v scales (only the MTP layer does), so vLLM logs its warning and uses scale 1.0, as on the uncensored Ornith build. Answers below were normal; no long-context accuracy check was run.
- **GPU in use after boot:** 87,552 MiB.

## Checked

- **Reasoning split:** `/v1/chat/completions` returns `reasoning` and a separate `content` ("391" for 17 × 23); no `</think>` in the content.
- **Tool calls, chat:** `tool_choice: auto` returns a parsed `tool_calls` entry with valid JSON arguments; no `<tool_call>` text leaks into the content.
- **Tool calls, Codex's wire:** `/v1/responses` with `stream: true` emits `reasoning_text` deltas, then a `function_call` item with arguments; a second turn carrying a `function_call_output` returns a `reasoning` item and a final message.
- **Gateway:** `/v1/models` on :8010 lists the registry id, `nemotron-abl`, `nemotron` and `main`; a request for `main` is answered by this model.
- **Speed:** one stream, a 1,200-token answer with reasoning on: 111.7 tok/s, one run.
- **Compliance:** a locksmith-style lock-picking question is answered, not refused.

## Not checked

The card's own numbers (refusal rate 98% to 2%, quality unchanged, KL 0.0157 against the base) are the
publisher's, from a private harness, and were not re-measured. Not run here: refusal rate on any
set, quality against the base model, vision, an MTP draft head, 16 requests in flight, a request
near 1,048,576 tokens, KV eviction to the host buffer, utilisation or sequence-count tuning, vLLM 0.29.0.
