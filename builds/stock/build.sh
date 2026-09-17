#!/usr/bin/env bash
# builds/stock/build.sh -- reproduce the "stock" venv (plain vllm==0.29.0
# wheel) from scratch in a NEW directory.
#
# Usage:
#   build.sh --dry-run [<new-venv-dir>]     # print every step, run nothing
#   build.sh <new-venv-dir>                 # actually build (NOT exercised
#                                            # by this packet -- see below)
#
# This is the cheap build of the three (no compile, minutes not hours), but
# it still was NOT run for real as part of packet P5 -- only --dry-run was,
# per the packet's hard rules (no pip/uv install into any tracked venv, no
# build). Read builds/README.md before running this against a real target.
set -uo pipefail

DRY_RUN=0
TARGET_DIR=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --help|-h)
            cat <<'EOF'
usage: build.sh [--dry-run] <new-venv-dir>

Reproduces the "stock" vllm==0.29.0 venv from scratch. --dry-run prints every
command without running it; when a target directory is required and none was
given, a placeholder is shown instead so the plan can still be printed.
EOF
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

echo "=== builds/stock/build.sh -- plan for $TARGET_DIR ==="
echo

echo "# 1. Python version pinned in builds/stock/MANIFEST.toml [venv].python"
run "uv venv --python 3.13.15 \"$TARGET_DIR\""
echo

echo "# 2. Install vllm==0.29.0 from PyPI, forcing the same torch build the"
echo "#    live venv actually has: torch.__version__ there is 2.13.0+cu130"
echo "#    (not plain 2.13.0 -- see MANIFEST.toml [packages].torch and the"
echo "#    check.sh note on why torch.__version__, not the dist METADATA"
echo "#    Version field, is what distinguishes a cu130 build). uv's"
echo "#    --torch-backend selects the matching PyTorch index; pinned to"
echo "#    cu130 explicitly rather than 'auto' because this box has no nvcc"
echo "#    to probe (see MANIFEST.toml [cuda])."
run "uv pip install --python \"$TARGET_DIR/bin/python\" vllm==0.29.0 --torch-backend=cu130"
echo

echo "# 3. lib64 shim -- see MANIFEST.toml [cuda].lib64_shim. Present in the"
echo "#    live venv though this build's own launchers never create it (the"
echo "#    packet could not pin down which step put it there); create it"
echo "#    defensively so a rebuilt venv behaves the same way."
run "CU=\"\$(dirname \"\$(find \\\"$TARGET_DIR\\\" -path '*/nvidia/cu13/lib' -maxdepth 6 -type d | head -1)\")\"; [ -e \"\$CU/cu13/lib64\" ] || ln -sfn lib \"\$CU/cu13/lib64\""
echo

echo "# 4. Verify (same commands used to populate MANIFEST.toml):"
run "\"$TARGET_DIR/bin/python\" --version"
run "\"$TARGET_DIR/bin/python\" -c \"import vllm; print(vllm.__version__)\""
run "\"$TARGET_DIR/bin/python\" -c \"import torch; print(torch.__version__)\""
echo

echo "# 5. Point a model at it: models.toml [builds].stock, or run"
echo "#    builds/stock/check.sh against \"$TARGET_DIR\" to confirm it matches."
echo
echo "NOTE: this recipe has not been run for real on this box (packet P5 ran"
echo "only --dry-run). It is the cheapest of the three builds -- no compile --"
echo "but still needs VLLM_USE_FLASHINFER_SAMPLER=0 at serve time (no nvcc on"
echo "this host; see MANIFEST.toml [cuda])."
