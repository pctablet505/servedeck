# GLM-5.3-Flash on a single RTX PRO 6000 (SM120)

Working configuration, and the seven bugs that stood between here and there.
vLLM built from PR #53906 (`0.29.0.dev0+glm53`), FlashInfer 0.6.18.

## Run it

```bash
cd ~/Projects/vllm-glm53
SPEC=1 GPU_UTIL=0.96 MAX_LEN=262144 MAX_BATCHED=4096 ./serve.sh
```

The two non-obvious mandatory settings are now `serve.sh` defaults:
`MOE_BACKEND=marlin` (bug 7 — without it the model loads, serves, and emits one
token forever with no error) and `VLLM_USE_DEEP_GEMM=0` (bug 8).
`SPEC=1` enables MTP; `SPEC_TOKENS` defaults to 2 (1 crashes mid-generation).

**Measured: 10.6-12.1 tok/s, stable over a 600-token generation.**

## Hardware reality

| | |
|---|---|
| checkpoint | 181.28 GiB NVFP4 (163.27 GiB of it routed experts) |
| VRAM | 95.6 GiB — cannot hold the model |
| approach | keep attention/embeddings/shared-expert resident, stream routed experts from host RAM over PCIe via UVA zero-copy |
| measured | 4.43 GiB of expert weights touched per decoded token; at 65% offloaded, 2.87 GiB crosses PCIe → ~25 GB/s effective |

Decode is **bandwidth-bound, not compute-bound**. The MoE kernel choice changes
throughput by ~6%; bytes-per-token is what matters.

## The seven bugs

### 1. Offloaded GPU memory was never released
`UVAOffloader` moves a parameter to host, but the caching allocator keeps the
freed block and — under `expandable_segments` — extends the segment for the next
layer rather than reusing it. Measured: `gpu_reserved` grew **4.06 GiB/layer**
while live tensors grew 0.30. Over 43 MoE layers that is 175 GiB reserved on a
96 GiB card, so **offloading bought nothing**.
*Fix:* `torch.cuda.empty_cache()` after each offloaded module (`uva.py`).

### 2. Host RSS was 1.94x the offloaded bytes
Pre-pinning routes the tensor through PyTorch's caching *host* allocator, which
rounds to power-of-two blocks: 2.42 GiB `w13` → 4 GiB, 1.21 GiB `w2` → 2 GiB.
19.0 GiB of weights occupied 33.8 GiB of RAM.
*Fix:* hand `get_cuda_view_from_cpu_tensor` an **unpinned** tensor; it pins with
an exact-size `cudaHostAlloc` itself (`VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1`).
Ratio → ~1.0x.

### 3-4. `fp8_ds_mla` cache kernel assumed DeepSeek's shape
GLM-5.3 is **NoPE** MLA (`qk_rope_head_dim=0`). `concat_and_cache_mla` asserted
`pe_dim == 64` and a fixed 656-byte entry; the launch hardcoded `dim3 block(96)`,
i.e. 64 NoPE threads + a 32-thread RoPE warp that must not run when pe_dim=0.
*Fix:* allow `pe_dim` 0 or 64, size the entry `512 + 16 + 2*pe_dim`, launch
`64 + pe_dim/2` threads (`csrc/libtorch_stable/cache_kernels.cu`).

### 5-6. The rope-free MLA lane was unreachable on SM120
`platforms/cuda.py` listed only `[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]` for
capability 12, and SM120's sparse MLA implements only DeepSeek's 576-wide shape.
The rope-free lane is `FLASHINFER_MLA_SPARSE_SM90`, gated to `major == 9` and
hardcoding Hopper-only `backend="fa3"`.
*Fix:* list the SM90 backend for `head_size == 512`; gate `(9, 12)`; use `fa2`
on SM12x (`fa3` compiles but emits no kernel image → "no kernel image is
available for execution on the device"). Needs FlashInfer >= 0.6.18.
Matches upstream issue vllm-project/vllm#53963.

### 7. NVFP4 MoE kernels silently corrupt on SM120  ← the expensive one
| backend | result |
|---|---|
| `flashinfer_trtllm` | refuses — "kernel does not support current device" |
| `flashinfer_cutedsl` | refuses — same |
| `flashinfer_cutlass` (**auto default**) | **silently wrong** |
| `cutlass` (vLLM) | **silently wrong** |
| `marlin` | **correct** |

Both CUTLASS-family kernels accept the work and return garbage; the model emits
one token forever (`"lock"`, id 1023) with no error or warning. TRTLLM and
CuteDSL refuse loudly on the same hardware — the silent one is what auto picks.

### 8. MTP hits an SM100-only DeepGEMM kernel
The MTP draft head routes through DeepGEMM's paged-MQA attention, which asserts
`arch_major == 10` (`deepgemm-src/csrc/apis/attention.hpp:320`); SM120 is 12.
*Fix:* `VLLM_USE_DEEP_GEMM=0` — only the spec-decode path needs it, plain decode
works with DeepGEMM enabled.

## Also found, not fixed

- **Fused-w13 scale collapse (affects ALL modelopt NVFP4 MoE checkpoints, not
  just the abliterated one):** the checkpoint stores `experts.N.gate_proj.weight_scale_2`
  and `experts.N.up_proj.weight_scale_2` as separate tensors, quantized
  independently, so their global scales differ — measured 179 of 288 experts,
  ~1.4x apart. vLLM fuses them into `w13` and keeps only `[:, 0]` (the gate
  scale), mis-scaling the up-projection. Inherent to how modelopt quantizes,
  NOT caused by abliteration; the official NVFP4 build will warn the same way.
- **Non-UVA offload fallback is broken:** `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1`
  dies with `CUDA error: an illegal memory access`, leaving a D-state thread
  holding all VRAM (only a reboot clears it).

## Measured

- 8.7 tok/s (marlin, no MTP)
- 10.6-12.1 tok/s (marlin + MTP-2), 11.9 sustained over 600 tokens — +31%, not the ~2x a
  naive bytes-per-token model suggests. GLM's MTP head is one layer re-run per
  speculative token, so acceptance falls off and each rejected draft costs a
  full forward pass; vLLM warns about this explicitly.
- 9.2 tok/s (cutlass, wrong output) — kernel choice moves throughput only ~6%,
  confirming decode is bandwidth-bound rather than compute-bound
- KV cache ~40 KB/token (MLA latent + KDA state + indexer/kpool, page-aligned)
- Model natively supports 1,048,576 context; 1M costs ~23% decode vs 256k
