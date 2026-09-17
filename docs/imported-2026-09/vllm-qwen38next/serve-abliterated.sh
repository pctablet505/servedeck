#!/usr/bin/env bash
# Launch the ABLITERATED Flash-Next graft (mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4)
# through the same serve.sh that serves RadixArk, so all the Flash-Next plumbing
# (PLE CPU offload, ptrace_scope relax/restore, lib64 shim, MTP spec-decode,
# qwen3_coder tool parser) is reused rather than duplicated.
#
# Differences this wrapper has to handle, and how:
#   * MODEL           -> serve.sh now honors ${MODEL:-...} (was hardcoded RadixArk).
#   * PLE table dtype -> mazinb ships the 51B n-gram table in BF16 (ple-bf16-*.safetensors),
#                        RadixArk ships it FP8. serve.sh auto-detects from the cached
#                        snapshot: FP8 -> VLLM_PLE_FP8_CHECKPOINT=1, BF16 -> offload
#                        timeout 1800s (the ~95 GB BF16 table blows the 600s default).
#   * SERVED_NAME     -> stays "qwen38-flash-next" so ~/.codex config and any client
#                        keep working with no change. Override with SERVED_NAME=... if
#                        you want the model id to advertise the abliterated name.
#   * VRAM            -> weights in VRAM are the NVFP4 experts (~68 GB) + residual BF16;
#                        the BF16 PLE table lives in HOST RAM (needs ~95 GB free; this
#                        box has 182 GB). util 0.95 matches the RadixArk default.
#
# SAFETY: this checkpoint has had its refusal direction abliterated. It will comply
# with harmful/illegal requests. Research use only; you own the consequences.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4}"
export SERVED_NAME="${SERVED_NAME:-qwen38-flash-next}"
export PORT="${PORT:-8001}"
export MAX_LEN="${MAX_LEN:-262144}"
export GPU_UTIL="${GPU_UTIL:-0.95}"
export MAX_SEQS="${MAX_SEQS:-1}"
export KV_DTYPE="${KV_DTYPE:-auto}"
# Fully cached (108/108 files verified); go offline so a DNS hiccup can't stall boot.
export HF_HUB_OFFLINE=1

# The PLE worker's peak RSS (~103 G) briefly collides with the main worker's
# weight load + safetensors page cache. Default swappiness=60 makes the kernel
# evict the cold-once-written PLE table to swap (~45 G) instead of dropping
# clean file pages. Low swappiness = reclaim page cache first, keep the table
# in RAM (swap faults on PLE lookups cost ~35% decode throughput).
sudo sysctl -q vm.swappiness=10 2>/dev/null || true

echo "MODEL=$MODEL SERVED_NAME=$SERVED_NAME PORT=$PORT GPU_UTIL=$GPU_UTIL MAX_LEN=$MAX_LEN"
# This banner used to hardcode "PLE: BF16 table". serve.sh decides the dtype by
# globbing the cached snapshot, so once ple_fp8/enable.sh swapped the table to
# FP8 the banner kept announcing BF16 while VLLM_PLE_FP8_CHECKPOINT=1 was in
# force -- the first line of the log said the opposite of what was booting.
# Report what serve.sh will actually detect, using the SAME glob it uses.
_ple_snap="$HOME/.cache/huggingface/hub/models--${MODEL%%/*}--${MODEL##*/}/snapshots"
if ls "$_ple_snap"/*/model-plefp8-*.safetensors >/dev/null 2>&1; then
    echo "PLE: FP8 table (model-plefp8-*) -> host RAM offload, VLLM_PLE_FP8_CHECKPOINT=1"
else
    echo "PLE: BF16 table (ple-bf16-*) -> host RAM offload, ready timeout ${VLLM_PLE_OFFLOAD_READY_TIMEOUT:-1800}s"
fi
unset _ple_snap
echo "sudo prompt expected (ptrace_scope relax for the CUDA IPC handoff)."

exec "$HERE/serve.sh"
