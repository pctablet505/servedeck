# Running GLM-5.3-Flash (NVFP4) on a single RTX PRO 6000 / SM120

A complete reproduction guide. GLM-5.3-Flash is 320B total / 18B active; the
NVFP4 checkpoint is **181.3 GiB** and the card has **95.6 GiB**. It runs, but
eight separate bugs stand between a fresh checkout and correct output — and
**one of them produces no error at all**, so you can spend a day chasing it.

Everything below was measured on:
Ubuntu 26.04 (glibc 2.43) · RTX PRO 6000 Blackwell Workstation (SM120, 96 GB)
· 182 GiB RAM · driver 595.84 · vLLM PR #53906 @ `878631b60` · FlashInfer 0.6.18

Final result: **10–12 tok/s**, 512k context, correct output.
Companion files: `GLM53-SM120.patch` (all source changes), `FINDINGS.md`
(condensed), `serve.sh` (the launcher, defaults already correct).

---

## 0. Read this first

Two facts decide whether this is worth your time:

1. **Decode is PCIe-bound and cannot be tuned out.** A decoded token touches
   **4.43 GiB** of expert weights (42 MoE layers x top-8 of 288 x ~13.5 MiB).
   With ~65% of experts in host RAM, ~2.9 GiB crosses PCIe *per token*.
   That is the 10-12 tok/s ceiling. Kernel choice moves it 6%; MTP moves it
   31%; config tuning ~10%. Nothing moves it 3x.
2. **The default MoE kernel silently returns garbage on SM120** (bug 7). The
   model loads, serves, reports healthy, and emits one token forever. If you
   take nothing else from this document, take `--kernel-config
   '{"moe_backend":"marlin"}'`.

If you have two GPUs, put the whole model in VRAM and skip all of this.

---

## 1. Build environment

CUDA is a **triple constraint** on this platform:

| | |
|---|---|
| driver 595.84 | caps at CUDA **13.2** |
| Ubuntu 26.04, glibc 2.43 | CUDA **13.0** fails to compile (`rsqrt` exception-spec conflict in `mathcalls.h`) |
| ⇒ | **13.2.86 is the only satisfying point** |

```bash
uv venv --python 3.13 .venv-glm53
uv pip install --python .venv-glm53/bin/python --no-deps \
  --index-url https://download.pytorch.org/whl/cu130 "torch==2.13.0+cu130"
# nvcc, cudart and nvrtc are THREE separate pip packages and drift apart.
# torch pulls cudart/nvrtc at 13.0; nvcc must be 13.2.86. Pin all of them:
uv pip install --python .venv-glm53/bin/python --no-deps \
  nvidia-cuda-nvcc==13.2.86 nvidia-cuda-runtime==13.2.86 nvidia-cuda-nvrtc==13.2.86
uv pip install --python .venv-glm53/bin/python --no-deps \
  --extra-index-url https://flashinfer.ai/whl/ \
  flashinfer-python==0.6.18 flashinfer-cubin==0.6.18   # 0.6.17 lacks SM90 NoPE MLA
```

**Two traps that cost hours:**

* **`nvcc` vs headers must match.** If they differ, Torch's `cuda.cmake` aborts
  configure with *"FindCUDA says CUDA version is 13.2, but the CUDA headers say
  13.0"* — 30 s in, after a long wait. Guard it before building:
  ```bash
  nvcc_ver=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
  hdr=$(sed -n 's/^#define CUDART_VERSION *\([0-9]*\).*/\1/p' "$CUDA_HOME/include/cuda_runtime_api.h")
  [ "$nvcc_ver" = "$((hdr/1000)).$(((hdr%1000)/10))" ] || echo "MISMATCH"
  ```
* **Anything that re-resolves dependencies will undo the pins.** A plain
  `uv pip install -r requirements/*.txt` silently replaced `torch 2.13.0+cu130`
  with `+cpu` mid-session. Always `--no-deps`, and re-check `torch.version.cuda`
  is not `None` before building.

Then build (~20 min, `MAX_JOBS=44 NVCC_THREADS=2`, single arch
`TORCH_CUDA_ARCH_LIST=12.0f`), plus two more traps:

* pip CUDA wheels ship `lib/`, but FlashInfer's JIT hardcodes `-L$cuda_home/lib64`.
  Without `ln -sfn lib "$CUDA_HOME/lib64"` the first forward pass fails to LINK
  (`ld: cannot find -lcudart`) and surfaces as the misleading `Ninja build failed`.
* Pre-clone flash-attn and set `VLLM_FLASH_ATTN_SRC_DIR`; CMake's FetchContent
  path drags in ROCm/composable_kernel, which a CUDA build never uses.

**Run from a directory with no `vllm/` in it.** If the repo root is itself a
vLLM checkout, `sys.path[0]` shadows the compiled editable install and you get
`ImportError: vllm.vllm_flash_attn requires the CUDA flash attention extensions`
— which reads as a broken build when the build is fine.

---

## 2. Memory: how 181 GiB fits in 96 GiB

90% of the checkpoint is routed MoE experts:

| component | GiB | |
|---|---|---|
| routed experts | **163.27** | offload these |
| attention (MLA + KDA) | 11.52 | resident |
| embeddings / lm_head | 2.36 | resident |
| shared expert | 2.02 | resident |
| norms, gates, dense MLP | 1.07 | resident |
| vision tower | 1.05 | disabled via `--limit-mm-per-prompt` |

Use **UVA** offload with segment matching, not the prefetch backend:

```
--offload-backend uva --cpu-offload-gb 105 --cpu-offload-params experts
```

Top-8-of-288 routing means a decode step reads only the experts it selected, and
UVA's zero-copy reads exactly those. The prefetch backend moves *whole layers*
(~3.9 GiB each) regardless of routing — catastrophic for a sparse MoE.

### Bug 1 — offloaded GPU memory is never released

Moving a parameter to host drops its GPU storage, but the caching allocator
keeps the block and, under `expandable_segments`, **extends the segment for the
next layer instead of reusing it**. Measured: `gpu_reserved` grew **4.06
GiB/layer** while live tensors grew 0.30. Over 43 MoE layers that is 175 GiB
reserved on a 96 GiB card — **offloading buys nothing** and you OOM around
layer 22, no matter what `--cpu-offload-gb` you pass.

*Fix:* `torch.cuda.empty_cache()` after each offloaded module (`uva.py`).
Afterwards `gpu_reserved` tracks `gpu_alloc` (4.44 vs 4.40).

### Bug 2 — host RSS is 1.94x the offloaded bytes

Pre-pinning routes the tensor through PyTorch's caching **host** allocator,
which rounds to power-of-two blocks: a 2.42 GiB `w13` takes 4 GiB, a 1.21 GiB
`w2` takes 2 GiB. 19.0 GiB of weights occupied 33.8 GiB of RAM.

*Fix:* hand `get_cuda_view_from_cpu_tensor` an **unpinned** tensor. The C++ op
then pins it itself with an exact-size `cudaHostAlloc`
(`csrc/libtorch_stable/cuda_view.cu`). Set
`VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1`. Ratio → ~1.0x.

> Counter-intuitive, and the opposite of what the flag name suggests. Pinning
> *earlier* is the pessimisation.

**Also:** there is no swap on many workstations. 100 GB of swap turns a bad
estimate into a slowdown instead of an OOM kill that takes down your desktop.
Pinned pages never swap, so this adds no capacity — only survivability.

---

## 3. The model is NoPE MLA — four DeepSeek assumptions break

GLM-5.3 sets **`qk_rope_head_dim = 0`** (verified in the upstream config *and*
as the default in vLLM's own `Glm5NextConfig`). Almost every sparse-MLA code
path assumes DeepSeek's 64-dim RoPE.

### Bugs 3-4 — the fp8_ds_mla cache kernel

`concat_and_cache_mla` asserts `pe_dim == 64` and a fixed 656-byte entry, and
launches `dim3 block(96)` — 64 NoPE threads **plus a 32-thread RoPE warp that
must not run** when there is no RoPE data (it would read past an empty `k_pe`
and write past the entry).

*Fix:* allow `pe_dim` 0 or 64; size the entry `512 + 16 + 2*pe_dim` (**528**
bytes for NoPE, 656 for DeepSeek); launch `64 + pe_dim/2` threads. Mirror the
size in `mla_attention.py`'s `state_content_bytes`, which hardcoded 656.
Requires a rebuild.

### Bugs 5-6 — the rope-free lane is unreachable on SM120

`platforms/cuda.py` offers capability 12 only
`[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]`, and the SM120 sparse kernel
implements only DeepSeek's 576-wide shape. FlashInfer's `_resolve_model_type`
supports exactly two layouts and GLM-5.3 matches **neither**:

| | latent | rope | d_qk | bytes |
|---|---|---|---|---|
| DSv3.2 / GLM-NSA | 512 | 64 | 576 | 656 |
| DSv4 | 448 | 64 | 512 | 584 |
| **GLM-5.3** | **512** | **0** | **512** | **528** |

The rope-free lane is **`FLASHINFER_MLA_SPARSE_SM90`** — despite the name — but
it is gated to `capability.major == 9` and hardcodes Hopper-only `backend="fa3"`.

*Fix:* list the SM90 backend for `head_size == 512`; gate `(9, 12)`; select
`fa2` on SM12x. FA3 compiles on SM120 but emits no device code, so you get
`no kernel image is available for execution on the device`. Needs FlashInfer
≥ 0.6.18. Matches upstream [vllm#53963](https://github.com/vllm-project/vllm/issues/53963);
the same approach was independently used on SM121 (DGX Spark).

---

## 4. Bug 7 — the one that produces no error

**Symptom:** the model loads, serves, `/health` is green, and every prompt
returns the same token forever (`"lock"`, id 1023). No error, no warning, no
NaN. Prompt-independent.

Sweeping all five NVFP4 MoE backends on SM120:

| backend | result |
|---|---|
| `flashinfer_trtllm` | refuses — *"kernel does not support current device"* |
| `flashinfer_cutedsl` | refuses — same |
| `flashinfer_cutlass` ← **auto picks this** | **silently wrong** |
| `cutlass` (vLLM) | **silently wrong** |
| **`marlin`** | **correct** |

Both CUTLASS-family kernels accept the work and return garbage. Their siblings
refuse loudly on the *same* hardware — so the failure mode you get is decided
by which kernel auto-selection happens to pick.

```
--kernel-config '{"moe_backend":"marlin"}'
```

Marlin warns *"your GPU does not have native support for FP4"* and falls back to
weight-only decompression. It costs only ~6% versus the (broken) CUTLASS path,
because decode is bandwidth-bound — the kernel is mostly waiting on PCIe.

**Debugging note.** What finally isolated this was a differential test: swap one
component, hold everything else fixed. Before reaching it we had eliminated,
each by direct measurement, the indexer's uninitialised top-k buffer, PDL on
unvalidated silicon, zero expert scales, and the TileLang hyper-connection
kernels. Use `--load-format dummy` for this: it exercises construction and the
offloader without reading a byte of checkpoint, turning a 6-minute 181 GiB
experiment into a 30-second one. Truncating the config to 8 layers helps too.

---

## 5. Bug 8 — MTP hits an SM100-only kernel

Speculative decoding (the checkpoint carries a real MTP head at layer 45) routes
through DeepGEMM's paged-MQA attention, which asserts `arch_major == 10`.

*Fix:* `VLLM_USE_DEEP_GEMM=0`. Only the spec-decode path needs it; plain decode
is fine with DeepGEMM on. Use **`num_speculative_tokens: 2`** — 1 returns
correct text then dies with `CUDA error: an illegal memory access` partway
through a longer generation, and 3 lowers acceptance (vLLM warns about this:
>1 re-runs the same MTP layer).

MTP is worth **+31%** (8.7 → 11.9 tok/s sustained) and is the only lever that
materially moves a bandwidth-bound decode.

---

## 6. Working command

```bash
VLLM_USE_DEEP_GEMM=0 \
VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
vllm serve <checkpoint> \
  --served-model-name glm53-flash \
  --port 8002 \
  --max-model-len 524288 \
  --gpu-memory-utilization 0.96 \
  --offload-backend uva --cpu-offload-gb 112 --cpu-offload-params experts \
  --kernel-config '{"moe_backend":"marlin"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --max-num-seqs 1 --max-num-batched-tokens 4096 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enable-prefix-caching --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47
```

Boot is ~6 min: 181 GiB load, then a **silent** multi-minute Marlin repack
(96% CPU, no log output — do not mistake it for a hang), then graph capture.

## 7. Measured

| | |
|---|---|
| decode | 8.7 tok/s (no MTP) → **10–12 tok/s** (MTP-2) |
| prefill | ~350–570 tok/s cold; ~1,270 tok/s prefix-cached |
| KV cache | **~14 KB/token** at long context (measure at *your* context: at 16k it appears to be 40 KB/token because per-sequence KDA state dominates) |
| VRAM at 512k | 71.5 weights+ctx · 5.1 activation · 14.5 KV · 0.3 graphs |
| context cost | 512k costs ~7% decode vs 256k; 1M ~6% more again |

**Sizing for one agent:** vLLM sizes KV from *leftover* budget, not need — at
512k it provisioned 2.1x concurrency. Cap it with `--kv-cache-memory` and give
the difference to resident experts; each GiB moved off the host is a GiB that
does not cross PCIe every token.

## 8. Not fixed

* **Fused-`w13` scale collapse (all modelopt NVFP4 MoE checkpoints).**
  `gate_proj.weight_scale_2` and `up_proj.weight_scale_2` are separate tensors,
  quantized independently — 179 of 288 experts differ by ~1.4x. vLLM fuses them
  and keeps only the gate scale, mis-scaling the up-projection. Not caused by
  abliteration; hardware-independent.
* **The non-UVA offload fallback is broken.**
  `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1` dies with an illegal memory access and
  leaves a **D-state thread holding all VRAM** — only a reboot clears it, and
  `nvidia-smi --gpu-reset` refuses if the card is primary.

## 9. Honest assessment

It works, and as far as the public record goes this was the first single-GPU
SM120 deployment. But **10-12 tok/s is a hard ceiling set by PCIe**, and a
model that fits entirely in VRAM will beat it by a wide margin for interactive
use. This is worth doing if you need *this* model's quality on *this* hardware
and can tolerate batch-style latency; otherwise fit the model to the card.
