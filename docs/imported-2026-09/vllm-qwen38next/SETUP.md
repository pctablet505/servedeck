# Qwen3.8-Flash-Next on a single RTX PRO 6000 — reproducible setup

Working as of **2026-08-27**. Serves `RadixArk/Qwen3.8-Flash-Next-NVFP4` at the
model's **full 262,144-token context** on one 96 GB Blackwell card.

This is not a normal `pip install vllm` job. The architecture (`qwen4_exp`) is
**not in any released vLLM** — it lives in an unmerged PR — and the checkpoint
exercises code paths that have two genuine upstream bugs. Six distinct blockers
sit between a clean machine and a serving model. Every one is documented below
with its *symptom*, because the symptoms are misleading: four of the six report
an error that names the wrong subsystem.

Budget ~3–4 h for a first run, most of it compile and download.

---

## 0. Baseline this was built on

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation, 97,887 MiB, **SM 12.0** |
| Driver | **595.84** — supports CUDA **13.2 max** (see Blocker 4) |
| OS | Ubuntu 26.04 LTS, glibc **2.43** |
| Python | 3.13.15 |
| torch | 2.13.0 |
| vLLM | `0.29.0.dev0+qwen38next` built from source |
| flashinfer | 0.6.16.post3 |
| CPU/RAM | 32 threads; **≥64 GB RAM required** (the n-gram table is offloaded to host) |

Disk: **~150 GB** for the model + ~30 GB for source/build artifacts.

---

## 1. Layout

```
~/Projects/vllm-qwen38next/
├── src/            # vLLM source, branch release/qwen38next_offload
├── .venv-next/     # isolated venv — NEVER touch the production .venv-llm
├── flash-attn/     # pre-cloned flash-attention (see Blocker 5)
├── build.sh        # source build
├── serve.sh        # launch server
├── fetch_fa.sh     # clone flash-attn without the ROCm tree
└── download.sh     # resumable model download
```

The isolation is deliberate: the production 27B server keeps running from
`~/Projects/local_llm/.venv-llm` throughout, so a failed experiment never costs
you a working setup.

---

## 2. Model download (do this first — it's the long pole)

146 GB. Run it under **systemd**, not a shell — this machine rebooted three
times mid-download during development and killed unsupervised transfers.

```bash
systemctl --user enable --now qwen38next-download.service
journalctl --user -u qwen38next-download -f
```

The unit uses `Restart=on-failure` + `StartLimitIntervalSec=0`, and `hf download`
resumes from partial blobs, so interruptions are free.

> **HF token:** read it from your environment; never echo it. If two `HF_TOKEN`
> lines exist in `~/.bashrc` they will concatenate into a broken 75-char token —
> use `grep -oP '(?<=export HF_TOKEN=")[^"]+' ~/.bashrc | tail -1`.

---

## 3. Build prerequisites

```bash
sudo apt install -y build-essential cmake ninja-build ccache git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
```

**Install rustup exactly once, serialized.** Running concurrent rustup
processes corrupts `~/.rustup/downloads` with rename errors; recovery is
`pkill rustup; rm -rf ~/.rustup/downloads` and a single clean retry.

Rust is required because vLLM builds a native tokenizer extension via
`setuptools_rust`. Ensure `~/.cargo/bin` is on `PATH` for the build.

---

## 4. Source checkout

```bash
git clone https://github.com/vllm-project/vllm.git src
cd src && git checkout release/qwen38next_offload   # PR #53899
```

Verify you're on the right tree:

```bash
git log --oneline -1     # 95dc96d [BugFix] fix PLE-offload initialization ...
```

---

## 5. Apply the two source patches

Both are **required** — without them the model cannot load. Both are genuine
upstream gaps, not local workarounds, and both carry explanatory comments in
the diff.

### Patch A — `vllm/envs.py`: add `VLLM_PLE_FP8_CHECKPOINT`

Registers a new env flag (declaration + lambda in `environment_variables`).

### Patch B — `vllm/models/qwen4_exp/nvidia/ple_layer.py`

In `_get_ple_embedding_quant_method`, **before** the `Fp8Config` checks:

```python
if envs.VLLM_PLE_FP8_CHECKPOINT:
    return Qwen4ExpPLEFp8EmbeddingMethod()
```

**Why (Blocker 1):** loading dies with

```
no module or parameter named 'ngram_embedding.weight_scale'
```

The checkpoint's top-level quant config is **NVFP4**, but its PLE n-gram table
is separately **FP8 with one global `weight_scale`** (131 shard weights + 1
global scale). The upstream gate only fires for a top-level `Fp8Config`, so the
`weight_scale` parameter is never created. The flag forces the FP8 PLE path for
exactly this checkpoint shape.

Re-apply after any `git pull`:
```bash
cd src && git diff > ../patches.diff     # save
cd src && git apply ../patches.diff      # restore
```

---

## 6. Pin CUDA to 13.2.86 — this is the narrowest constraint in the whole setup

```bash
uv pip install --python .venv-next/bin/python \
  "nvidia-cuda-nvcc==13.2.86" "nvidia-cuda-crt==13.2.86" "nvidia-nvvm==13.2.86" \
  "nvidia-cuda-runtime==13.2.86" "nvidia-cuda-nvrtc==13.2.86"
```

**13.2 is the only version that works, squeezed from both sides:**

- **13.3+ fails at runtime.** Driver 595.84 supports CUDA 13.2 max. Building
  with 13.3 compiles cleanly, then dies at first kernel launch with
  `CUDA error: the provided PTX was compiled with an unsupported toolchain`
  (Blocker 4). The build gives no warning.
- **13.0 fails at compile.** On Ubuntu 26.04's glibc 2.43 you get an
  exception-specification conflict on `rsqrt`/`rsqrtf` in `mathcalls.h`.

All five packages must move together — mismatched nvcc/headers produce
confusing version errors.

Then create the unversioned symlinks CMake's `FindCUDA` needs (**Blocker 3**):

```bash
CU=.venv-next/lib/python3.13/site-packages/nvidia/cu13/lib
cd $CU && for f in *.so.*; do
  base=$(echo "$f" | sed 's/\.so\..*/.so/'); [ -e "$base" ] || ln -s "$f" "$base"
done
```

pip CUDA wheels ship only versioned `.so.N`. Without this:
`Could NOT find CUDA_CUDART_LIBRARY`.

Also create the **lib64 shim** (Blocker 6 — `serve.sh` does this automatically):

```bash
ln -sfn lib .venv-next/lib/python3.13/site-packages/nvidia/cu13/lib64
```

Afterwards, check for dangling links — a CUDA version change leaves symlinks
pointing at files that no longer exist:
```bash
find $CU -maxdepth 1 -xtype l
```

---

## 7. Pre-clone flash-attention (Blocker 5)

```bash
./fetch_fa.sh
```

Clones at pinned `617264c1c7955c9e84817654ebeedff069f3c5f1`, inits
`csrc/cutlass`, and **skips `csrc/composable_kernel`** (ROCm/AMD only, huge,
unused on CUDA). Result: 393 MB instead of multi-GB.

Doing this ahead of time and pointing the build at it via
`VLLM_FLASH_ATTN_SRC_DIR` avoids the build fetching it mid-compile, where an
interruption corrupts the clone.

---

## 8. Build

```bash
MAX_JOBS=40 NVCC_THREADS=2 ./build.sh          # ~11 min warm, ~40 min cold
```

Key environment in `build.sh`:

| var | value | why |
|---|---|---|
| `TORCH_CUDA_ARCH_LIST` | `12.0f` | **single arch only.** Building all archs is a multi-hour waste on one known card |
| `CUDA_HOME` | pip `nvidia/cu13` | not a system CUDA install |
| `MAX_JOBS` / `NVCC_THREADS` | 40 / 2 | 32-thread CPU; RAM is the real limit, ~1.5 GB per nvcc |
| `SETUPTOOLS_SCM_PRETEND_VERSION` | `0.29.0.dev0+qwen38next` | no tags on the branch |
| `VLLM_FLASH_ATTN_SRC_DIR` | `../flash-attn` | see step 7 |
| `CCACHE_DIR` | set | rebuilds drop to ~11 min |

The install line:

```bash
uv pip install --python .venv-next/bin/python -e . --no-build-isolation --no-deps
```

**`--no-deps` is load-bearing.** Without it, uv re-resolves dependencies
mid-build and silently *downgrades your pinned CUDA packages*, undoing step 6
and reintroducing Blocker 4. This cost two full rebuild cycles to find.

### Verify the build before serving

```bash
.venv-next/bin/python -c "
import vllm, torch
from vllm.model_executor.models.registry import ModelRegistry
print('vllm  :', vllm.__version__)
print('archs :', [a for a in ModelRegistry.get_supported_archs() if 'Qwen4' in a])
x = torch.randn(8, device='cuda'); print('kernels OK:', bool((x*2).sum().item() or True))
"
```

You need `Qwen4Exp*` in the arch list **and** a real kernel launch. Import
alone does not prove the PTX toolchain is right — that only fails on first
launch.

---

## 9. Serve

```bash
./serve.sh
```

### Flags that matter, and why

| flag | value | reason |
|---|---|---|
| `--max-model-len` | `262144` | model's architectural ceiling (`max_position_embeddings`; `rope_type: default`, no YaRN) |
| `--gpu-memory-utilization` | `0.96` | 0.92 leaves ~4 GiB idle; no display attached to this card |
| `--max-num-seqs` | `1` | full context admits only one sequence (1.16x) |
| `--kv-cache-dtype` | `auto` (bf16) | **fp8 is rejected**: `Qwen4Exp QSA requires a BF16 main KV cache` |
| `--limit-mm-per-prompt` | `{"image":0,"video":0}` | frees the 16k-token image encoder cache; saves 0.84 GiB of weights |
| `--speculative-config` | mtp, 3 tokens | speed; costs KV (see capacity) |
| `--tool-call-parser` | `qwen3_coder` | required for agentic use |
| `--reasoning-parser` | `qwen3` | **see the warning below** |

Environment set by `serve.sh`:

```
VLLM_PLE_CPU_OFFLOAD=1      # keeps the ~51 GB n-gram table in host RAM
VLLM_PLE_FP8_CHECKPOINT=1   # the patched flag from step 5
PYTORCH_ALLOC_CONF=expandable_segments:True
VLLM_USE_FLASHINFER_SAMPLER=0
```

> ### ⚠ The `qwen3` reasoning parser splits the response
> Output goes to `reasoning_content`, **not** `content`. A short `max_tokens`
> can be consumed entirely by reasoning, leaving `content` empty — which looks
> exactly like a model failure. Verify any client (Codex included) reads the
> right field before concluding the model is broken.

---

## 10. `ptrace_scope` — required at startup only

PLE CPU-offload hands the GPU output buffer to a sibling worker process via
**CUDA IPC**, which needs `pidfd_getfd()` → `PTRACE_MODE_ATTACH`. Ubuntu's
default `kernel.yama.ptrace_scope=1` permits attach only to *descendants*, and
the GPU worker and `PleOffloadWorker` are **siblings**.

Symptom (**Blocker 2**):
```
PleOffloadWorker ... RuntimeError: pidfd_getfd: Operation not permitted
```

`serve.sh` handles this correctly and minimally:

1. reads the current value, saving it in `PTRACE_ORIG`
2. if non-zero, `sudo sysctl -w kernel.yama.ptrace_scope=0`
3. a background poller watches `/v1/models` and **re-hardens as soon as the
   server answers** — the IPC handoff happens once, at startup
4. `trap restore_ptrace EXIT INT TERM` (the server runs with `&` + `wait`, not
   `exec`, so the trap fires)

Running the server therefore requires sudo, by design. **Do not** set
`ptrace_scope=0` globally or persist it in `sysctl.conf`.

> If you ever relax it by hand (e.g. `pkexec` while debugging), `serve.sh` sees
> it already at 0 and will not restore it. Put it back yourself:
> `sudo sysctl -w kernel.yama.ptrace_scope=1`

---

## 11. The six blockers, as symptoms

Four of these name the wrong subsystem. Match on the symptom:

| # | Symptom | Real cause | Fix |
|---|---|---|---|
| 1 | `no module or parameter named 'ngram_embedding.weight_scale'` | NVFP4 checkpoint, FP8 PLE table | patches in §5 |
| 2 | `pidfd_getfd: Operation not permitted` | `ptrace_scope=1` blocks sibling CUDA IPC | §10 |
| 3 | `Could NOT find CUDA_CUDART_LIBRARY` | pip wheels have no unversioned `.so` | symlinks, §6 |
| 4 | `the provided PTX was compiled with an unsupported toolchain` | built with CUDA 13.3, driver caps at 13.2 | pin 13.2.86, §6 |
| 5 | `Ninja build failed` *(at first forward pass)* | flashinfer JIT link: `ld: cannot find -lcudart` — it hardcodes `-L$cuda_home/lib64`, wheels use `lib/` | lib64 shim, §6 |
| 6 | `To serve at least one request ... 7.57 GiB needed ... available 3.89 GiB` | KV budget too small at util 0.92 | §9 flags |

**Blocker 5 is the nastiest.** vLLM reports the generic `Ninja build failed`
and dumps 97 compile lines; the actual cause is a single `ld` line buried in
the output. It also appears *minutes after* a successful weight load, during
memory profiling, so it reads like a runtime fault rather than a link error.
Find it with:
```bash
grep -nE "FAILED:|cannot find -l" serve.log
```

---

## 12. Measured capacity and performance

From the boot log, not estimates:

```
Model loading took          78.47 GiB
Available KV cache memory    8.83 GiB
GPU KV cache size          304,653 tokens
Maximum concurrency @262k     1.16x
GPU used                   92,571 / 97,887 MiB
```

**KV cost: 30.4 KiB/token** (12 full-attention layers = 24 KiB; remainder is
MTP + linear-attention state). Of 48 layers, `full_attention_interval=4` means
only 12 hold a growing KV cache; the other 36 are Gated DeltaNet linear
attention with fixed per-sequence state.

Verified behaviour:

| test | result |
|---|---|
| short-prompt decode | **99.3 tok/s** |
| prefill @253k tokens | 27.1 s → **~9,300 tok/s** |
| needle-in-haystack @220k, depths 10/50/93% | **3/3 PASS**, uncached |

### Capacity is the real trade-off

262,144 is the **model's** ceiling, not a memory limit — so freeing memory buys
*concurrency*, never more context. At full length this config serves **one
agent at a time**.

For comparison, the 27B abliterated model on the same card had ~62 GiB of KV
budget and ran ~7 full-context agents at 173.8 tok/s. **Flash-Next costs
roughly 8× the concurrency and is slower per token.** That is inherent to 79 GiB
of weights plus a 51 GB n-gram table, not a tuning failure. Decide deliberately.

Levers to trade speed for concurrency (estimated, unmeasured):
- **drop MTP** — frees the 4B layer's weights *and* cuts KV/token 30.4 → ~24 KiB
- `--enforce-eager` — frees CUDA-graph memory, ~20–30% slower decode
- util 0.96 → 0.98 — ~+1.9 GiB, thinner OOM margin

---

## 13. Rollback

The production 27B setup is untouched and independent:

```bash
systemctl --user enable --now qwen-vllm     # 27B on :8000
```

Its venv (`~/Projects/local_llm/.venv-llm`), unit file, and `.config` all
survive a Flash-Next failure.

> **Watchdog caution:** a restart watchdog cannot distinguish *deliberately
> stopped* from *crashed*, and will resurrect a server you meant to stop.
> `qwen-vllm-watchdog` is currently **disabled** for this reason.

---

## 14. Traps worth knowing

- **`pgrep -f` / `pkill -f` match your own command line.** Every pattern-based
  process check here caused a false positive or killed the calling shell. Use
  `ps -eo comm` or explicit PIDs.
- **Don't kill orphan processes by pattern during a build** — this corrupted the
  flash-attn clone and a rustup download.
- **Boot is slow but cached.** torch.compile AOT artifacts and the 97 flashinfer
  JIT objects persist in `~/.cache/`; first boot ~10 min, later boots ~4 min.
- **`shm_broadcast: No available shared memory broadcast block found in 60
  seconds`** during startup is informational, not an error.
