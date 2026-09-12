#!/usr/bin/env bash
# builds/glm53/build.sh -- reproduce the "glm53" fork venv from scratch in a
# NEW directory. Recipe transcribed from ~/Projects/vllm-glm53/build.sh and
# FINDINGS.md, cited inline below.
#
# Usage:
#   build.sh --dry-run [<new-dir>]     # print every step, run nothing
#   build.sh <new-dir>                 # actually build -- multi-hour, needs
#                                       # the GPU idle. NOT exercised by this
#                                       # packet; see builds/README.md.
#
# <new-dir> becomes the wrapper directory: <new-dir>/src (fork checkout) and
# <new-dir>/.venv-glm53 (the venv), mirroring ~/Projects/vllm-glm53/.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PATCH_DIR="$REPO_ROOT/builds/patches/glm53"

REMOTE="https://github.com/pctablet505/vllm.git"
HEAD_SHA="878631b6079d2cf9fb80830ef9cb41b43aded098"

DRY_RUN=0
TARGET_DIR=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --help|-h)
            cat <<'EOF'
usage: build.sh [--dry-run] <new-dir>

Reproduces the "glm53" fork venv (vllm-project/vllm fork at
pctablet505/vllm, branch glm53, PR #53906, pinned commit) from scratch.
--dry-run prints every command without running it. A real run is a
multi-hour source build and needs the GPU idle -- not exercised by this
packet.
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

echo "=== builds/glm53/build.sh -- plan for $TARGET_DIR ==="
echo

echo "# 1. Clone the fork, FULL (not --depth 1): MANIFEST.toml [base].note --"
echo "#    the local 'glm53' branch has no configured upstream, but the SHA"
echo "#    below IS reachable from this remote (an ancestor of its"
echo "#    glm53-nope-mla-sm12x branch); a shallow clone of just one branch"
echo "#    tip risks missing it, and a shallow clone also breaks"
echo "#    setuptools_scm's tag lookup (build.sh's own"
echo "#    SETUPTOOLS_SCM_PRETEND_VERSION workaround, step 7 below, exists"
echo "#    because of exactly this)."
run "git clone \"$REMOTE\" \"$TARGET_DIR/src\""
run "git -C \"$TARGET_DIR/src\" checkout \"$HEAD_SHA\""
echo

echo "# 2. Apply the exported patches, in order (MANIFEST.toml [patches].files)."
echo "#    Do NOT apply GLM53-SM120-2026-08-29.patch -- historical, superseded"
echo "#    by working-tree-tracked.patch (MANIFEST.toml [patches].note)."
echo "#    Do NOT apply builds/patches/glm53/branches/main/*.patch -- 39"
echo "#    commits on an unrelated local branch named 'main', not merged into"
echo "#    'glm53'."
run "git -C \"$TARGET_DIR/src\" apply \"$PATCH_DIR/working-tree-tracked.patch\""
run "git -C \"$TARGET_DIR/src\" apply \"$PATCH_DIR/working-tree-untracked.patch\""
echo

echo "# 3. Pre-clone flash-attention at the pinned ref, skipping the ROCm"
echo "#    submodule (fetch_fa.sh, same pinned commit as the qwen38next build)."
run "git clone https://github.com/Dao-AILab/flash-attention.git \"$TARGET_DIR/fa-src\""
run "git -C \"$TARGET_DIR/fa-src\" checkout 617264c1c7955c9e84817654ebeedff069f3c5f1"
run "git -C \"$TARGET_DIR/fa-src\" submodule update --init csrc/cutlass"
echo

echo "# 4. Create the venv (MANIFEST.toml [venv].python)."
run "uv venv --python 3.13.15 \"$TARGET_DIR/.venv-glm53\""
echo

echo "# 5. torch, matching MANIFEST.toml [packages].torch (2.13.0+cu130)."
echo "#    build.sh (lines ~26-37) treats a CPU-only or mismatched torch as"
echo "#    fatal before ever invoking cmake, because that failure otherwise"
echo "#    surfaces 30s into an unrelated-looking compile error."
run "uv pip install --python \"$TARGET_DIR/.venv-glm53/bin/python\" torch==2.13.0 --torch-backend=cu130"
echo

echo "# 6. Realign nvcc with torch's cudart/nvrtc (build.sh's nvcc_ver=hdr_ver"
echo "#    guard, lines ~53-68): nvcc, cudart and nvrtc are three separate pip"
echo "#    packages and drift apart easily. Pin all three to the version this"
echo "#    driver (595.84, CUDA 13.2 max) actually supports."
run "uv pip install --python \"$TARGET_DIR/.venv-glm53/bin/python\" --no-deps \\
  nvidia-cuda-nvcc==13.2.86 nvidia-cuda-runtime==13.2.86 nvidia-cuda-nvrtc==13.2.86"
echo

echo "# 7. lib64 shim (MANIFEST.toml [cuda].lib64_shim; build's own"
echo "#    serve-opt.sh:170-175 creates this at serve time -- doing it here"
echo "#    too means a fresh venv also links correctly before first serve)."
run "CUDA_HOME=\"$TARGET_DIR/.venv-glm53/lib/python3.13/site-packages/nvidia/cu13\"; ln -sfn lib \"\$CUDA_HOME/lib64\""
echo

echo "# 8. Build and install editable (build.sh:110, verbatim). --no-deps is"
echo "#    load-bearing for the same reason as the qwen38next build: without"
echo "#    it uv re-resolves deps mid-build and silently reintroduces the"
echo "#    CUDA version mismatch step 6 just fixed."
run "export CUDA_HOME=\"$TARGET_DIR/.venv-glm53/lib/python3.13/site-packages/nvidia/cu13\""
run "export PATH=\"$TARGET_DIR/.venv-glm53/bin:\$CUDA_HOME/bin:\$HOME/.cargo/bin:\$PATH\""
run "export CARGO_HOME=\"\$HOME/.cargo\""
run "export TORCH_CUDA_ARCH_LIST=12.0f"
run "export VLLM_FLASH_ATTN_SRC_DIR=\"$TARGET_DIR/fa-src\""
run "export SETUPTOOLS_SCM_PRETEND_VERSION=0.29.0.dev0+glm53"
run "export VLLM_TARGET_DEVICE=cuda"
run "cd \"$TARGET_DIR/src\" && MAX_JOBS=\${MAX_JOBS:-40} NVCC_THREADS=\${NVCC_THREADS:-2} \\
  uv pip install --python \"$TARGET_DIR/.venv-glm53/bin/python\" -e . --no-build-isolation --no-deps"
echo

echo "# 9. Verify (same commands used to populate MANIFEST.toml)."
run "\"$TARGET_DIR/.venv-glm53/bin/python\" -c \"import vllm; print(vllm.__version__)\""
run "\"$TARGET_DIR/.venv-glm53/bin/python\" -c \"import torch; print(torch.__version__)\""
echo

echo "NOTE: not exercised on this box. FINDINGS.md documents seven kernel/"
echo "runtime bugs this fork's patches fix on top of the base tag (v0.28.1rc0"
echo "+56 commits, PR #53906); this is a multi-hour compile (see"
echo "vllm-qwen38next/SETUP.md's comparable timings) and needs the GPU idle."
echo "builds/glm53/check.sh is what verifies a resulting venv actually"
echo "matches MANIFEST.toml."
