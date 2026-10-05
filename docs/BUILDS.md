# The three vLLM builds

Every model in `models.toml` names a `build`, and every build is one venv. Two of the three are patched source checkouts; one is a plain wheel. Nothing about a build is discovered at launch time — `[builds.<key>]` in `models.toml` gives the venv and its `CUDA_HOME`, and that is all servedeck knows.

| build | venv | vLLM | serves | source |
|---|---|---|---|---|
| `stock` | `~/Projects/local_llm/.venv-llm-029` | `0.29.0` | `qwen27b`, `lfm2` | plain PyPI wheel, no patches |
| `qwen38next` | `~/Projects/vllm-qwen38next/.venv-next` | `0.29.0.dev0+qwen38next.cu132` | `flashnext` | editable install of `~/Projects/vllm-qwen38next/src`, branch `release/qwen38next_offload` at `95dc96d1d` |
| `glm53` | `~/Projects/vllm-glm53/.venv-glm53` | `0.29.0.dev0+glm53.cu132` | `glm53` | editable install of `~/Projects/vllm-glm53/src`, branch `glm53` at `878631b60` |

All three run Python 3.13.15, torch 2.13.0+cu130, transformers 5.16.1, triton 3.7.1, numpy 2.4.6. FlashInfer differs on purpose: `stock` and `glm53` carry 0.6.18, `qwen38next` carries 0.6.16.post3. Do not harmonise them — GLM-5.3 needs >= 0.6.18 for the SM90 NoPE MLA kernel that 0.6.17 lacks, and the Flash-Next tree was built and validated against 0.6.16.post3.

`lfm2` is declared on `stock` per the v2 registry decision, but the live systemd unit launches it from `.venv-next`'s python. Either venv serves a 350M model; the declaration is what a rebuild would follow.

## Why each fork exists

**Flash-Next (`qwen38next`).** The architecture `qwen4_exp` is in no released vLLM — it lives in unmerged PR #53899, which is what the fork branch carries. On top of that branch the working tree holds four changes that are not upstream anywhere:

- `vllm/envs.py` registers `VLLM_PLE_FP8_CHECKPOINT`, and `vllm/models/qwen4_exp/nvidia/ple_layer.py` returns `Qwen4ExpPLEFp8EmbeddingMethod` *before* the `Fp8Config` checks. Without both, loading dies on `no module or parameter named 'ngram_embedding.weight_scale'`: the checkpoint's top-level quant config is NVFP4 while its PLE n-gram table is separately FP8 with one global `weight_scale`, a shape the upstream gate never fires for.
- `kv_connector/v1/offloading/{config,scheduler}.py` skip non-prefix-cacheable KV groups, which is what lets `--kv-offloading-size 40` boot at all; without it the engine asserts on the QSA ring group. The standalone diff is `builds/patches/flashnext-kv-offload-skip-qsa-ring.patch` (note: `models.toml:138` names it under `builds/patches/qwen38next/`, which is wrong — it sits one directory up; the same diff is also inside `patches/qwen38next/working-tree-tracked.patch`, re-exported 2026-09-17).
- `v1/core/sched/scheduler.py` defers a request whose admission fails while nothing is running, instead of `break`ing out of the waiting loop. Without it a request whose host-KV hit is larger than the free pool head-of-line blocks everything behind it: the engine re-runs the same failed admission about 790 times a second with the GPU at 0%. Measured 2026-09-30 on 262,144-token Flash-Next: four stalls in 5h51m, longest 14.7 min, one logging `hit 122304 offloaded tokens` 149,584 times while five requests waited. Diff: `builds/patches/flashnext-sched-hol-deadlock-on-external-kv-hit.patch`. It is NOT inside `patches/qwen38next/working-tree-tracked.patch` (that export predates it), so a rebuild must apply it separately.

**GLM-5.3 (`glm53`).** GLM-5.3 is NoPE MLA (`qk_rope_head_dim = 0`, latent 512, `d_qk` 512, a 528-byte cache entry) and that breaks four DeepSeek assumptions in the sparse-MLA path: `concat_and_cache_mla` asserted `pe_dim == 64` with a fixed 656-byte entry and launched `dim3 block(96)`, a 32-thread RoPE warp that must not run; `state_content_bytes` in `mla_attention.py` hardcoded 656; and `platforms/cuda.py` offered capability 12 only `TRITON_MLA` + `FLASHINFER_MLA_SPARSE_SM120`, whose kernel implements only DeepSeek's 576-wide shape. The rope-free lane is `FLASHINFER_MLA_SPARSE_SM90` despite the name; the patch lists it for `head_size == 512`, gates `(9, 12)`, and selects fa2 on SM12x — fa3 compiles but emits no device code, which surfaces as `no kernel image is available for execution on the device`. (Documented 2026-08-29 in the fork's `RUNNING-GLM53-ON-ONE-GPU.md`; not re-derived since.) The September tuning pass added the hot-expert Marlin cache, DMA staging and grouped prefill on top.

**stock.** No fork, because the 27B needed no code patches: its 0.27.1 install matched its package RECORD hashes on 24,416 of 24,417 files with nothing untracked under `vllm/` (checked 2026-09-11). Its speed-ups are launch flags, which survive a venv swap. The move to 0.29.0 was made for one upstream fix — see *Upstream distance* below.

## The CUDA pin, and the two symlinks

CUDA is pinned to **13.2.86** across all five `nvidia-cuda-*` pip packages (nvcc, crt, nvvm, runtime, nvrtc). It is the only version that satisfies both sides: 13.3+ compiles cleanly and then dies at the first kernel launch with `CUDA error: the provided PTX was compiled with an unsupported toolchain` (driver 595.84 caps at 13.2), and 13.0 fails to compile on this host's glibc 2.43 with an exception-specification conflict on `rsqrt`/`rsqrtf` in `mathcalls.h`. All five must move together; mismatched nvcc and headers abort torch's `cuda.cmake` about 30 s into configure with `FindCUDA says CUDA version is 13.2, but the CUDA headers say 13.0`. (Recorded 2026-08-27; not re-derived.)

The editable install line is `uv pip install -e . --no-build-isolation --no-deps`. **`--no-deps` is load-bearing**: without it uv re-resolves mid-build and silently downgrades the pinned CUDA packages, reinstating the PTX failure, and it will also replace `torch 2.13.0+cu130` with `+cpu`. That cost two full rebuild cycles to find.

Two symlink shims inside each venv's `nvidia/cu13`:

- unversioned `.so` links next to the `.so.N` that pip CUDA wheels ship, or CMake reports `Could NOT find CUDA_CUDART_LIBRARY`;
- `lib64 -> lib`, because FlashInfer's JIT hardcodes `-L$CUDA_HOME/lib64`. Without it the fused-MoE kernel fails to *link* at the first forward pass (`ld: cannot find -lcudart`), minutes after a successful weight load, and reports it as the generic `Ninja build failed`. Find that one with `grep -nE 'FAILED:|cannot find -l'`.

Both forks' serve scripts recreate `lib64` if it is missing (`serve.sh:66-71`, `serve-opt.sh:170-175`) and `builds/*/build.sh` recreates it on a rebuild. The `stock` venv has it too, and nothing on disk creates it there — `qwen-server-run.sh` and the lfm2 unit contain no `lib64` reference. It exists and is correct; its origin is unexplained, so do not delete it.

`VLLM_USE_FLASHINFER_SAMPLER=0` is set for every model in `[defaults.env]` for the same family of reasons: there is no system CUDA toolkit on `PATH`, so FlashInfer's JIT top-k/top-p sampler cannot compile.

## Patches are files, never a hand-edited checkout

`builds/patches/<name>/` holds the fork's state as exported diffs: `working-tree-tracked.patch` (`git diff` of modified tracked files), `working-tree-untracked.patch` (added files, with `*.bak*` and `*.reviewed` scratch deliberately excluded), `branches/<topic>/000N-*.patch` for local topic branches that are **not** merged into the installed branch, and `MANIFEST.txt`, the raw capture log from 2026-09-12.

`builds/<name>/MANIFEST.toml` is the structured version: venv and package versions, editable target, tree remote/branch/HEAD/describe, the ordered patch list, the excluded branches, and the CUDA facts. `builds/<name>/check.sh` verifies the live venv against it, read-only, no GPU, no network. The check that earns its keep is the last one: it re-derives the tracked diff and the untracked-file diff straight from the live tree and hashes them against the exported patches, so an edit that was never re-exported is caught here rather than discovered months later.

**Nothing runs `check.sh` automatically.** `tests/test_builds.py` parses every `MANIFEST.toml`, cross-checks `[models.*].build` against `[builds]`, asserts every listed patch file exists, and runs `check.sh --help` and `build.sh --dry-run` — it never runs the real check, because that needs the live venvs. Run it by hand after touching a fork, and before trusting a `MANIFEST.toml`:

```bash
for b in stock qwen38next glm53; do bash ~/Projects/servedeck/builds/$b/check.sh; done
```

State on 2026-09-18: all three pass, including both exported diffs hash-matching in both forks and all six excluded branches still unmerged.

The rule is: change the checkout, re-export, re-run `check.sh`. Editing a `.patch` by hand, or fixing the checkout without re-exporting, is exactly the drift `check.sh` exists to catch.

`builds/patches/glm53/GLM53-SM120-2026-08-29.patch` is history only and is excluded from `build.sh`'s apply list. It touches 9 files, all 9 also in `working-tree-tracked.patch` (which touches 10 — `marlin_moe.py` is the addition), and it predates the September tuning pass. Rebuilding from it alone would lose the hot-expert cache, DMA staging and grouped prefill, leaving GLM at roughly 10.8 tok/s instead of ~16 and making its `models.toml` env vars inert.

## A fork's working tree is not a backup

Each fork carries a branch `local/servedeck-2026-09-17` holding a snapshot commit of its uncommitted **tracked** changes. Untracked files are not in it. The sharpest case: `vllm/model_executor/offloader/dma_stage.py`, GLM's DMA staging module, is in neither `HEAD` nor `local/servedeck-2026-09-17` (verified 2026-09-18) — it exists only as an untracked file in the working tree and as 623 lines of `builds/patches/glm53/working-tree-untracked.patch`. Lose the tree without that patch file and the feature is gone.

`~/Projects/vllm-glm53` holds **two** vLLM checkouts and only `src/` runs. The outer repo is on branch `glm-release` at `3c01e04e` with a pristine `vllm/` tree; `src/.git` is a separate checkout on branch `glm53` at `878631b60` that the venv editable-installs. The outer `git status` looks clean while real modifications exist — always `git -C src status`. A raw `diff -rq vllm/ src/vllm/` (~43 paths) is **not** the patch set; most of it is the two checkouts sitting one commit apart.

## Rebuilding

A rebuild is a multi-hour source compile that needs the GPU idle for its whole duration. **Never run a real `build.sh` while the card is serving.** `builds/*/build.sh` supports `--dry-run`, which prints every step and runs none; on this box only `--dry-run` has ever been exercised, for all three builds. The full recipe from a clean directory was ~3-4 h, most of it compile and download, when it was last done (2026-08-27); with `CCACHE_DIR` populated a rebuild in place is ~11 min warm and ~40 min cold. `~/.cache/ccache` is 2.1 G today, so warm is the realistic case.

Build environment that matters:

| var | value | why |
|---|---|---|
| `TORCH_CUDA_ARCH_LIST` | `12.0f` | single arch. Building all archs is a multi-hour waste on one known card |
| `MAX_JOBS` / `NVCC_THREADS` | `40` / `2` | 32 threads, but RAM is the real limit at ~1.5 GB per nvcc |
| `CUDA_HOME` | the venv's `nvidia/cu13` | there is no system CUDA install |
| `VLLM_FLASH_ATTN_SRC_DIR` | a pre-cloned flash-attention | keeps CMake from fetching it mid-compile, where an interruption corrupts the clone |
| `SETUPTOOLS_SCM_PRETEND_VERSION` | `0.29.0.dev0+<fork>` | neither fork branch has tags |
| `CCACHE_DIR` | set | the difference between 11 min and 40 |

What has to be fetched: the fork remote at the recorded SHA (`peakcrosser7/vllm` for `qwen38next`, `pctablet505/vllm` for `glm53` — `build.sh` checks out the SHA, not a branch tip, so an advanced remote is not silently picked up), the five CUDA pins, flash-attention at `617264c1c7955c9e84817654ebeedff069f3c5f1` with only `csrc/cutlass` initialised (skipping `csrc/composable_kernel`, which is ROCm-only: 393 MB instead of multi-GB — the live `fa-src` is 393 MB), and the CMake `FetchContent` deps that land in `src/.deps`.

One trap to expect: both installed venvs report versions ending `.cu132`, and no `build.sh` on disk sets that suffix — the fork scripts and `builds/*/build.sh` export `0.29.0.dev0+qwen38next` and `0.29.0.dev0+glm53`. A rebuild that follows them produces a version string `check.sh` will flag as drift on the `vllm version` row. Set `SETUPTOOLS_SCM_PRETEND_VERSION` to the manifest's value, or update the manifest deliberately.

Verify before serving — import alone does not prove the PTX toolchain, which only fails on first launch:

```bash
.venv-next/bin/python -c "
import vllm, torch
from vllm.model_executor.models.registry import ModelRegistry
print(vllm.__version__, [a for a in ModelRegistry.get_supported_archs() if 'Qwen4' in a])
torch.randn(8, device='cuda').sum().item()"
```

GLM's own boot is slow and looks broken: ~6 minutes, in which 181 GiB of weights load, then a **silent** multi-minute Marlin repack at 96% CPU with no log output, then graph capture (measured 2026-08-29; not re-measured). Do not read the silent phase as a hang.

## What is safe to delete after a build

`src/.deps` (fetched CMake dependency sources and their build trees) and `src/rust/target` are build-time only. The runtime artefacts are the `*.abi3.so` extension modules inside `src/vllm/`, which the editable install imports directly; nothing at runtime reads `.deps` or `rust/target`. Deleting them costs a re-fetch and a cold Rust build on the next rebuild, nothing else. Current sizes (2026-09-18): `vllm-qwen38next/src` 6.6 G of which `.deps` 2.2 G and `rust/target` 2.6 G; `vllm-glm53/src` 9.7 G of which `.deps` 2.2 G and `rust/target` 5.1 G — about 12 GiB across both, on a root filesystem at 86% with 123 G free.

**Venv sizes as `du` reports them are not what deleting one frees.** uv hardlinks package files into `~/.cache/uv/archive-v0`, and the venvs share those inodes with each other. Measured 2026-09-18:

| venv | `du -sh` alone | of which multiply-linked |
|---|---|---|
| `.venv-llm` (vLLM 0.27.1, superseded) | 9.6 G | 9.3 GiB |
| `.venv-llm-029` (`stock`) | 9.1 G | 8.8 GiB |
| `.venv-next` (`qwen38next`) | 7.0 G | 6.7 GiB |
| `.venv-glm53` (`glm53`) | 14 G | 0 |

`libtorch_cuda.so` (469 MB) has 6 links: the uv archive plus `.venv-llm`, `.venv-llm-029` and `.venv-next`. So deleting the superseded `.venv-llm` reclaims roughly **0.3 GiB**, not 9.5 — the rest stays alive through the uv cache (60 G) and the two live venvs. An earlier note claiming ~9.5 GiB, and a plan claiming 3.5 G, are both artefacts of measurement: a single `du -sh A B` credits shared inodes to whichever path it walks first, which is how `.venv-llm-029` came out as 3.4 G. Measure each path in its own `du` call, subtract `find -links +1`, and prune the uv cache if the space is the point.

`tests/test_e2e_real.py:79` hardcodes the `stock` venv path, so moving or deleting it breaks that test as well as `models.toml`.

## The FP8 PLE table

Flash-Next's PLE n-gram table was converted from BF16 to FP8 locally and the BF16 original deleted on 2026-09-10. What serves today is **43 locally generated files, 51,200,268,090 B = 47.684 GiB**, named `model-plefp8-*.safetensors`, living *inside* the HF snapshot directory `~/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f/`, referenced by a **hand-edited** `model.safetensors.index.json` that is a regular file where HF wrote a symlink. They are not in any repo and cannot be re-downloaded. Each has link count 2 (verified 2026-09-18): its only second name is the hardlink in `~/.cache/huggingface/ple-fp8-staging` (48 G), which shares the same extents — deleting staging frees nothing while the snapshot copies exist, and deleting both frees everything.

If the FP8 table is lost, regenerate it from BF16 with `ple_fp8/convert_ple_bf16_to_fp8.py` — but the BF16 table is gone, so that first means a 95.368 GiB download. The full route, including the two rollback sidecars now renamed `*.STALE-ple-bf16-deleted` (a deliberate interlock: without the rename, `rollback.sh` would restore the BF16 index *and* delete the 43 FP8 shards, leaving a checkpoint naming files that do not exist) is in [imported-2026-09/vllm-qwen38next/ple_fp8_cleanup/RECOVERY.md](imported-2026-09/vllm-qwen38next/ple_fp8_cleanup/RECOVERY.md). Two things from it that are easy to get wrong: re-arm both sidecars *before* running `rollback.sh`, and do not sweep `blobs/2d2c3617...e407` (28.7 MiB) — it has no symlink pointing at it but it is the rollback target, not an orphan.

The conversion report claims `bit_exact_fraction: 1.0`, `rms_error: 0.0` and also `clamped_elements: 1`, which cannot both be true; the worst case is one element in 51.2 billion off by ~6.1e-05. With the BF16 table deleted, that can no longer be checked.

Two reusable rules from that cleanup. Deleting a snapshot symlink frees nothing — the bytes are in `blobs/<sha256>`, so a safe reclaim checks hardlink count and referrer count and removes both names; and an mmapped blob keeps its extents until the holder exits, so `df` will not move while a model is serving. And `$SNAP` is exported by snapd into every shell launched from the snap-packaged VS Code: a script reading a bare `$SNAP` silently retargets itself at `/snap/code/260`. The PLE scripts renamed theirs to `PLE_SNAP` / `PLE_CLEANUP_SNAP`; if you are running an older copy, use `env -u SNAP`.

## Upstream distance, and the one fix that matters

`qwen38next` tracks `origin/release/qwen38next_offload` and was 2 commits behind it at the last fetch; `build.sh` pins the SHA, so those two are not picked up by accident. There is no way in this checkout to say how far the branch is from a released vLLM: it has no tags, and PR #53899's base commit is recorded nowhere. `glm53` describes as `v0.28.1rc0-56-g878631b60` and its installed branch has no configured upstream, so it is not confirmed pushed under that name — the commit *is* reachable from the fork remote via `fork/glm53-nope-mla-sm12x`.

The upstream fix that matters is **PR #50729** (merged 2026-08-17), which fixes a race in the hybrid-GDN Mamba state-copy plus CUDA-graph-replay path on SM120 that made vLLM 0.26.0 through 0.28.0 crash on Qwen hybrid-GDN checkpoints (paired Xid 13+31, or a host-side segfault in `CUDAGraph::replay`; upstream #54331 reports the same failure on the same hardware class, with only `--enforce-eager` surviving). Its marker `is_left_overlap` is present in `vllm/v1/worker/mamba_utils.py` in the `stock` 0.29.0 wheel and in the `glm53` fork tree, and **absent from the `qwen38next` fork tree** (`git grep` at HEAD, 2026-09-18). Moving the 27B onto `stock` 0.29.0 is what closed that crash family for it; Flash-Next is a hybrid-GDN model running on a tree that predates the fix, so a rebase of that fork onto a post-0.29 base is the outstanding work. See [HOST.md](HOST.md) for the Xid history and [models/qwen27b.md](models/qwen27b.md) for the decision trail.

## Known defect, fix not applied

The fused-w13 scale collapse affects **all** modelopt NVFP4 MoE checkpoints, not only GLM's and not because of abliteration: `gate_proj.weight_scale_2` and `up_proj.weight_scale_2` are quantised independently (179 of 288 experts differ by ~1.4x), vLLM fuses them into `w13` and keeps only the gate scale, mis-scaling the up-projection. The official NVFP4 build warns the same way, so do not chase it as a checkpoint problem. A fix exists on the local branch `modelopt-nvfp4-w13-scale-mismatch` (`e473a88a7`, "requantize mismatched w1/w3 global scales instead of silently using the gate's") with tests on `nvfp4-w13-reconcile-tests`; neither is merged into the installed `glm53` branch, so what serves today still has the defect.
