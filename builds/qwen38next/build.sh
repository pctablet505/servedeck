#!/usr/bin/env bash
# builds/qwen38next/build.sh -- reproduce the "qwen38next" fork venv from
# scratch in a NEW directory. Recipe transcribed from
# ~/Projects/vllm-qwen38next/SETUP.md (sections 3-9) and build.sh, cited
# inline below.
#
# Usage:
#   build.sh --dry-run [<new-dir>]     # print every step, run nothing
#   build.sh <new-dir>                 # actually build -- multi-hour, needs
#                                       # the GPU idle. NOT exercised by this
#                                       # packet; see builds/README.md.
#
# <new-dir> becomes the wrapper directory: <new-dir>/src (fork checkout) and
# <new-dir>/.venv-next (the venv), mirroring ~/Projects/vllm-qwen38next/.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PATCH_DIR="$REPO_ROOT/builds/patches/qwen38next"

REMOTE="https://github.com/peakcrosser7/vllm.git"
HEAD_SHA="95dc96d1d012a25ff5c3823a1e77197c8dae4654"

DRY_RUN=0
TARGET_DIR=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --help|-h)
            cat <<'EOF'
usage: build.sh [--dry-run] <new-dir>

Reproduces the "qwen38next" fork venv (vllm-project/vllm fork at
peakcrosser7/vllm, release/qwen38next_offload, pinned commit) from scratch.
--dry-run prints every command without running it. A real run is a multi-hour
source build (SETUP.md: ~3-4 h first run) and needs the GPU idle -- not
exercised by this packet.
EOF
            exit 0
            ;;
        *) TARGET_DIR="$arg" ;;
    esac
done

if [ -z "$TARGET_DIR" ]; then
    if [ "$DRY_RUN" -eq 1 ]; then
        TARGET_DIR="<new-dir>"
    else
        echo "usage: build.sh [--dry-run] <new-dir>" >&2
        exit 2
    fi
fi

run() {
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '+ %s\n' "$*"
    else
        echo "+ $*"
        eval "$@"
    fi
}

echo "=== builds/qwen38next/build.sh -- plan for $TARGET_DIR ==="
echo

echo "# 0. Host prerequisites (SETUP.md §3) -- one-time, not repeated per build:"
echo "#    sudo apt install -y build-essential cmake ninja-build ccache git"
echo "#    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
echo "#    (rustup exactly once, serialized -- concurrent installs corrupt"
echo "#    ~/.rustup/downloads per SETUP.md §3)"
echo

echo "# 1. Clone the fork and pin the exact recorded commit (SETUP.md §4)."
echo "#    Pin the SHA, not the branch tip: MANIFEST.toml [base].note records"
echo "#    origin/release/qwen38next_offload as 2 commits ahead of this SHA"
echo "#    as of 2026-09-12, so 'git checkout release/qwen38next_offload'"
echo "#    alone would NOT reproduce the running tree."
run "git clone \"$REMOTE\" \"$TARGET_DIR/src\""
run "git -C \"$TARGET_DIR/src\" checkout \"$HEAD_SHA\""
echo

echo "# 2. Apply the exported patches, in order (MANIFEST.toml [patches].files)."
echo "#    Do NOT apply builds/patches/qwen38next/branches/*/*.patch -- those"
echo "#    are unmerged topic branches, not part of the running tree."
run "git -C \"$TARGET_DIR/src\" apply \"$PATCH_DIR/working-tree-tracked.patch\""
run "git -C \"$TARGET_DIR/src\" apply \"$PATCH_DIR/working-tree-untracked.patch\""
echo

echo "# 3. Pre-clone flash-attention at the pinned ref, skipping the ROCm"
echo "#    submodule (SETUP.md §7 / fetch_fa.sh)."
run "git clone https://github.com/Dao-AILab/flash-attention.git \"$TARGET_DIR/flash-attn\""
run "git -C \"$TARGET_DIR/flash-attn\" checkout 617264c1c7955c9e84817654ebeedff069f3c5f1"
run "git -C \"$TARGET_DIR/flash-attn\" submodule update --init csrc/cutlass"
echo

echo "# 4. Create the venv (MANIFEST.toml [venv].python)."
run "uv venv --python 3.13.15 \"$TARGET_DIR/.venv-next\""
echo

echo "# 5. Pin CUDA to 13.2.86 -- SETUP.md §6, 'the narrowest constraint in"
echo "#    the whole setup': driver 595.84 caps at CUDA 13.2 (13.3 built"
echo "#    kernels fail at first launch with an unsupported-toolchain error;"
echo "#    13.0 fails to compile on this host's glibc 2.43)."
run "uv pip install --python \"$TARGET_DIR/.venv-next/bin/python\" \\
  \"nvidia-cuda-nvcc==13.2.86\" \"nvidia-cuda-crt==13.2.86\" \"nvidia-nvvm==13.2.86\" \\
  \"nvidia-cuda-runtime==13.2.86\" \"nvidia-cuda-nvrtc==13.2.86\""
echo

echo "# 6. torch itself, matching MANIFEST.toml [packages].torch"
echo "#    (2.13.0+cu130) -- installed --no-deps in the build step below, but"
echo "#    vLLM's own dependency resolution needs it present first; SETUP.md's"
echo "#    build.sh treats a missing/cu130-less torch as fatal before compiling."
run "uv pip install --python \"$TARGET_DIR/.venv-next/bin/python\" torch==2.13.0 --torch-backend=cu130"
echo

echo "# 7. Unversioned .so symlinks CMake's FindCUDA needs, and the lib64"
echo "#    shim (SETUP.md §6, Blockers 3 and 5/6; MANIFEST.toml [cuda])."
run "CU=\"$TARGET_DIR/.venv-next/lib/python3.13/site-packages/nvidia/cu13/lib\"; cd \"\$CU\" && for f in *.so.*; do base=\$(echo \"\$f\" | sed 's/\\.so\\..*/.so/'); [ -e \"\$base\" ] || ln -s \"\$f\" \"\$base\"; done"
run "ln -sfn lib \"$TARGET_DIR/.venv-next/lib/python3.13/site-packages/nvidia/cu13/lib64\""
echo

echo "# 8. Build and install editable, --no-deps load-bearing (SETUP.md §8:"
echo "#    without it uv re-resolves deps mid-build and silently downgrades"
echo "#    the pinned CUDA packages, reintroducing the toolchain mismatch)."
run "export CUDA_HOME=\"$TARGET_DIR/.venv-next/lib/python3.13/site-packages/nvidia/cu13\""
run "export PATH=\"$TARGET_DIR/.venv-next/bin:\$CUDA_HOME/bin:\$HOME/.cargo/bin:\$PATH\""
run "export TORCH_CUDA_ARCH_LIST=12.0f"
run "export VLLM_FLASH_ATTN_SRC_DIR=\"$TARGET_DIR/flash-attn\""
run "export SETUPTOOLS_SCM_PRETEND_VERSION=0.29.0.dev0+qwen38next"
run "export VLLM_TARGET_DEVICE=cuda"
run "cd \"$TARGET_DIR/src\" && MAX_JOBS=\${MAX_JOBS:-40} NVCC_THREADS=\${NVCC_THREADS:-2} \\
  uv pip install --python \"$TARGET_DIR/.venv-next/bin/python\" -e . --no-build-isolation --no-deps"
echo

echo "# 9. Verify (SETUP.md §8's own verification block)."
run "\"$TARGET_DIR/.venv-next/bin/python\" -c \"
import vllm, torch
from vllm.model_executor.models.registry import ModelRegistry
print('vllm  :', vllm.__version__)
print('archs :', [a for a in ModelRegistry.get_supported_archs() if 'Qwen4' in a])
\""
echo

echo "NOTE: not exercised on this box. SETUP.md budgets ~3-4 h for a first"
echo "run (~11 min warm / ~40 min cold once the toolchain is staged), and the"
echo "whole thing needs the GPU idle. builds/qwen38next/check.sh is what"
echo "verifies a resulting venv actually matches MANIFEST.toml."
