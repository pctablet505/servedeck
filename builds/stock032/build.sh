#!/usr/bin/env bash
# builds/stock032/build.sh -- reproduce the "stock032" venv (plain vllm==0.32.0
# wheel plus FlashInfer's prebuilt sm120f kernels) from scratch in a NEW directory.
#
# Usage:
#   build.sh --dry-run [<new-venv-dir>]     # print every step, run nothing
#   build.sh <new-venv-dir>                 # actually build
#
# The live venv was built by these steps by hand on 2026-10-22; this script
# has been run only with --dry-run.
set -uo pipefail

DRY_RUN=0
TARGET_DIR=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --help|-h)
            cat <<'USAGE'
usage: build.sh [--dry-run] <new-venv-dir>

Reproduces the "stock032" vllm==0.32.0 venv from scratch. --dry-run prints every
command without running it; when a target directory is required and none was
given, a placeholder is shown instead so the plan can still be printed.
USAGE
            exit 0
            ;;
        *) TARGET_DIR="$arg" ;;
    esac
done

if [ -z "$TARGET_DIR" ]; then
    if [ "$DRY_RUN" -eq 1 ]; then
        TARGET_DIR="<new-venv-dir>"
    else
        echo "usage: build.sh [--dry-run] <new-venv-dir>" >&2
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

CU="$TARGET_DIR/lib/python3.13/site-packages/nvidia/cu13"

echo "=== builds/stock032/build.sh -- plan for $TARGET_DIR ==="
echo

echo "# 1. Python version pinned in MANIFEST.toml [venv].python"
run "uv venv --python 3.13.15 \"$TARGET_DIR\""
echo

echo "# 2. vllm==0.32.0 from PyPI on the cu130 torch build. On Wi-Fi the first"
echo "#    try timed out mid-download; a longer UV_HTTP_TIMEOUT got through."
run "UV_HTTP_TIMEOUT=300 uv pip install --python \"$TARGET_DIR/bin/python\" vllm==0.32.0 --torch-backend cu130"
echo

echo "# 3. FlashInfer's prebuilt kernels for this card only (MANIFEST.toml"
echo "#    [cuda].jit_cache): the shim with --no-deps, plus its sm120f package."
run "uv pip install --python \"$TARGET_DIR/bin/python\" --no-deps 'flashinfer-jit-cache==0.7.0.post1+cu130' 'flashinfer-jit-cache-sm120f==0.7.0.post1+cu130' --extra-index-url https://flashinfer.ai/whl/cu130"
echo

echo "# 4. The two shims FlashInfer's boot-time JIT links against (MANIFEST.toml [cuda])."
run "[ -e \"$CU/lib64\" ] || ln -s lib \"$CU/lib64\""
run "[ -e \"$CU/lib/libcudart.so\" ] || ln -s libcudart.so.13 \"$CU/lib/libcudart.so\""
echo

echo "# 5. Verify: point MANIFEST.toml [venv].path at $TARGET_DIR (or build in"
echo "#    place) and run builds/stock032/check.sh."
