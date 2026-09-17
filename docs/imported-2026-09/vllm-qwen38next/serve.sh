#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next NVFP4 on a single RTX PRO 6000 (96GB).
#
# HOW THIS FITS ON ONE CARD: the checkpoint is ~130GB, but ~51GB of that is the
# n-gram embedding table, which VLLM_PLE_CPU_OFFLOAD keeps in host RAM and
# prefetches asynchronously. That leaves ~76GB of weights in VRAM plus KV cache.
# Needs the qwen4_exp support from vLLM PR #53899 -- not in any released vLLM.
#
# NOTE vs the reference config this was based on:
#   * VLLM_PLE_FP8_CHECKPOINT IS set, and we had to add it. PR #53899 does not
#     ship it, and without it loading dies on 'ngram_embedding.weight_scale':
#     this checkpoint is NVFP4 overall but its PLE n-gram table is FP8 with one
#     global scale, and upstream only creates that scale parameter when the
#     top-level quant_config is Fp8Config. Local patch:
#     src/vllm/models/qwen4_exp/nvidia/ple_layer.py + src/vllm/envs.py.
#   * --max-num-seqs is left at a usable value rather than 2. 2 was a
#     single-large-context benchmark choice; it would cripple subagent use.
set -uo pipefail
# --- DRY_RUN dump hook (test-only) -------------------------------------------
# Prints what this script WOULD run instead of running it, so
# local_llm/tests/test_serve_model_golden.sh can prove that
# local_llm/bin/serve-model.sh resolves byte-identically. Never reached unless
# DRY_RUN=1 is set in the environment.
#   ENV  lines: one per name in $DRY_ENV_VARS (unset -> "<unset>"; HF_TOKEN is
#               hashed so a token never lands in test output). The caller owns
#               the vocabulary, so the launcher and this script can never
#               compare different sets of names.
#   ARGV lines: one per argument, in order.
# DRY_RUN_FORMAT=nul emits the argv NUL-separated and nothing else, for the
# case where an argument could itself contain a newline.
_dry_run_dump() {
    local _v
    if [ "${DRY_RUN_FORMAT:-line}" = nul ]; then
        printf '%s\0' "$@"
        return 0
    fi
    for _v in ${DRY_ENV_VARS:-}; do
        if [ -z "${!_v+x}" ]; then
            printf 'ENV %s=<unset>\n' "$_v"
        elif [ "$_v" = HF_TOKEN ]; then
            printf 'ENV %s=sha256:%s\n' "$_v" "$(printf '%s' "${!_v}" | sha256sum | cut -c1-12)"
        else
            printf 'ENV %s=%s\n' "$_v" "${!_v}"
        fi
    done
    printf 'ARGV %s\n' "$@"
}
# -----------------------------------------------------------------------------


ROOT="/home/pctablet505/Projects/vllm-qwen38next"
VENV="$ROOT/.venv-next"
# Overridable so Coldstart (or a shell) can point this launcher at any
# qwen4_exp checkpoint, e.g. the abliterated graft
# mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4. Keep the default as-is.
# Default is the abliterated graft, which is what actually gets served here.
# The base RadixArk checkpoint was deleted on 2026-08-29 (archived to Ventoy
# at hf-archive/); leaving it as the default would have made a bare run
# silently re-download 126 GB.
MODEL="${MODEL:-mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4}"
PORT="${PORT:-8001}"

export CUDA_HOME="$VENV/lib/python3.13/site-packages/nvidia/cu13"

# --- lib64 shim (required) ---------------------------------------------------
# FlashInfer's JIT hardcodes "-L$cuda_home/lib64" (flashinfer/jit/cpp_ext.py),
# but the pip CUDA wheels lay libraries out in lib/, not lib64/. Without this
# symlink the SM120 fused-MoE kernel fails to LINK at first forward pass with
# "ld: cannot find -lcudart / -lnvrtc", which surfaces as "Ninja build failed".
[ -e "$CUDA_HOME/lib64" ] || ln -sfn lib "$CUDA_HOME/lib64"
# -----------------------------------------------------------------------------
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0

# The single flag that makes one-GPU serving possible.
export VLLM_PLE_CPU_OFFLOAD=1
# VLLM_PLE_FP8_CHECKPOINT must MATCH the checkpoint's PLE n-gram table dtype:
# RadixArk/Inferact ship it FP8 (model-plefp8-*.safetensors), while
# primitive-ai/mazinb ship it BF16 (ple-bf16-*.safetensors) -- setting the flag
# for a BF16 table dies on 'ngram_embedding.weight_scale', and NOT setting it
# for an FP8 one dies the other way. Detect from the cached snapshot instead of
# hardcoding. BF16 tables are ~95 GB and blow past the 600 s offload default,
# so raise that timeout only for them (mazinb README: 1800).
_PLE_SNAP="$HOME/.cache/huggingface/hub/models--${MODEL%%/*}--${MODEL##*/}/snapshots"
if ls "$_PLE_SNAP"/*/model-plefp8-*.safetensors >/dev/null 2>&1; then
    export VLLM_PLE_FP8_CHECKPOINT=1
else
    export VLLM_PLE_OFFLOAD_READY_TIMEOUT="${VLLM_PLE_OFFLOAD_READY_TIMEOUT:-1800}"
fi
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST=12.0f

# ---------------------------------------------------------------- ptrace ----
# PLE offload hands the CPU worker a CUDA IPC handle for the GPU output buffer,
# and torch 2.13 transfers that handle with pidfd_getfd(). That syscall needs
# PTRACE_MODE_ATTACH; the two processes are siblings, so Ubuntu's default
# kernel.yama.ptrace_scope=1 (attach limited to descendants) denies it and
# startup dies with "pidfd_getfd: Operation not permitted".
#
# We do NOT weaken this permanently. It is relaxed only while the server runs
# and restored on exit, so the box sits at the hardened default the rest of the
# time. That is why starting the server asks for sudo.
PTRACE_PATH=/proc/sys/kernel/yama/ptrace_scope
PTRACE_ORIG=""
if [ -r "$PTRACE_PATH" ]; then
    PTRACE_ORIG=$(cat "$PTRACE_PATH")
    if [ "$PTRACE_ORIG" != "0" ]; then
        echo "ptrace_scope is $PTRACE_ORIG; PLE offload needs 0. Requesting sudo..."
        if ! sudo sysctl -w kernel.yama.ptrace_scope=0; then
            echo "FATAL: could not relax ptrace_scope; PLE offload cannot start." >&2
            exit 1
        fi
    else
        PTRACE_ORIG=""   # already 0, nothing of ours to restore
    fi
fi

restore_ptrace() {
    [ -n "$PTRACE_ORIG" ] || return 0
    # -n: never prompt. The sudo timestamp usually expires during a long run,
    # so say plainly what is left undone rather than hanging on a password.
    if sudo -n sysctl -w kernel.yama.ptrace_scope="$PTRACE_ORIG" >/dev/null 2>&1; then
        echo "restored ptrace_scope=$PTRACE_ORIG"
    else
        echo "WARNING: could not restore ptrace_scope (sudo timed out)." >&2
        echo "  Run: sudo sysctl -w kernel.yama.ptrace_scope=$PTRACE_ORIG" >&2
    fi
}
# Stop the server too, not just the sysctl. Without this a TERM to serve.sh
# restores ptrace_scope and exits while vllm keeps running and holding the GPU.
#
# OWNERSHIP: this trap signals exactly one pid -- $VLLM_PID, the child this
# script itself forked -- and never a port, a pattern, or a process group. The
# PPid check makes that positive rather than merely recorded: between the fork
# and the trap the pid could in principle have been reaped and reused, and a
# cleanup trap that kills a stranger is the whole defect class this file is
# being audited for (llm's cmd_start had it on 2026-09-10 and SIGTERM'd two
# healthy servers it did not own).
cleanup() {
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null \
       && [ "$(awk '/^PPid:/{print $2}' "/proc/$VLLM_PID/status" 2>/dev/null)" = "$$" ]; then
        echo "stopping vllm (pid $VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null
        for _ in $(seq 1 30); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
        kill -0 "$VLLM_PID" 2>/dev/null && kill -KILL "$VLLM_PID" 2>/dev/null
    fi
    restore_ptrace
}
trap cleanup EXIT INT TERM

# The relaxed setting is only needed while the CUDA IPC handle is handed over,
# which happens once in accept_registrations() during startup. The steady-state
# path just copies into the already-mapped buffer and never re-shares. So drop
# back to the hardened value the moment the server answers, rather than leaving
# the box relaxed for the whole run.
restore_when_ready() {
    [ -n "$PTRACE_ORIG" ] || return 0
    for _ in $(seq 1 150); do   # ~25 min ceiling; cold start is 2-3 min
        if curl -sf -m 3 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
            restore_ptrace
            return 0
        fi
        sleep 10
    done
    echo "WARNING: server never became ready; ptrace_scope left relaxed." >&2
}

HF_TOKEN=$(grep -oP '(?<=export HF_TOKEN=")[^"]+' "$HOME/.bashrc" 2>/dev/null | tail -1)
export HF_TOKEN

# FlashInfer autotune is deliberately LEFT ON (vLLM's own -O2 default). This
# invocation used to pass --no-enable-flashinfer-autotune, inherited from the
# reference config with no rationale. Measured 2026-08-31 on the abliterated
# checkpoint: forward-pass rate 57.9 -> 59.9/s (+3.4%), consistent across code,
# prose and repetitive workloads. MTP depth was also tested: k=4 is a wash (the
# extra draft step costs ~8.5% of pass rate, cancelling the higher accept count)
# and k>=5 trips "QSA ring capacity must divide the attention block size", so 3
# stays. Re-tune only against bench.py.
# --- multimodal limits -------------------------------------------------------
# Was hardcoded to '{"image":0,"video":0}'. The vision tower of this checkpoint
# is only 0.84 GiB, and the owner asked for images on :8001 (2026-09-09), so it
# is now an env knob. DEFAULT IS UNCHANGED, so any caller that does not set
# MM_LIMIT_JSON gets byte-identical argv to before (local_llm/tests/
# test_serve_model_golden.sh compares this argv against bin/serve-model.sh).
# serve-abliterated.sh / the operator sets MM_LIMIT_JSON to override.
# NOTE: this CANNOT be driven from local_llm/.config's EXTRA_ARGS -- llm parses
# that file with ([A-Z_]+)="([^\"]*)" , so a value containing a double quote is
# SILENTLY SKIPPED, and vLLM only accepts strict JSON here (single quotes,
# image=2 and {image:2} were all tested and rejected).
#
# DEFAULT CHANGED 2026-09-10 from '{"image":0,"video":0}' to images-on. The
# owner's requirement is "we want images and multimodal to be enabled"; with the
# old default a bare `llm start` silently served text-only, so images survived
# only until the next restart. Measured cost of image=2 on this box: weights
# +0.81 GiB (the 0.84 GiB vision tower) plus 0.49 GiB of multimodal profiling
# peak and encoder cache, i.e. KV 7.98 -> 7.63 GiB and 290,925 -> 278,479
# tokens at GPU_MEM_UTIL 0.96, which still holds a full 262,144-token request
# (1.06x). Set MM_LIMIT_JSON='{"image":0,"video":0}' to get the old text-only
# behaviour back, e.g. when trading KV headroom for a long-context run.
if [ -z "${MM_LIMIT_JSON:-}" ]; then MM_LIMIT_JSON='{"image":2,"video":0}'; fi
# -----------------------------------------------------------------------------
# --- MM_LIMIT_JSON validation ------------------------------------------------
# A malformed value must not reach vLLM. vLLM rejects it in argparse, but only
# after this launcher has already relaxed ptrace_scope and lowered swappiness,
# and the two launch paths would have "agreed" on a value neither can serve.
# The SAME regex, message and exit status live in serve-tuned.sh and in
# local_llm/bin/serve-model.sh, so local_llm/tests/test_serve_model_golden.sh
# can assert that a bad value fails IDENTICALLY on every path instead of one
# path silently dropping it. Accepts {} and {"name": <non-negative int>, ...}
# with optional whitespace -- i.e. exactly the shape vLLM parses.
_MM_LIMIT_RE='^\{[[:space:]]*("[A-Za-z_][A-Za-z0-9_]*"[[:space:]]*:[[:space:]]*[0-9]+([[:space:]]*,[[:space:]]*"[A-Za-z_][A-Za-z0-9_]*"[[:space:]]*:[[:space:]]*[0-9]+)*[[:space:]]*)?\}$'
if [[ ! "$MM_LIMIT_JSON" =~ $_MM_LIMIT_RE ]]; then
    echo "FATAL: MM_LIMIT_JSON is not strict JSON of \"modality\": <count> pairs: $MM_LIMIT_JSON" >&2
    exit 2
fi
# -----------------------------------------------------------------------------

VLLM_ARGV=( "$VENV/bin/vllm" serve "$MODEL" \
    --served-model-name "${SERVED_NAME:-qwen38-flash-next}" \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len "${MAX_LEN:-262144}" \
    --gpu-memory-utilization "${GPU_UTIL:-0.95}" \
    --tensor-parallel-size 1 \
    --distributed-executor-backend mp \
    --max-num-seqs "${MAX_SEQS:-1}" \
    --max-num-batched-tokens "${MAX_BATCHED:-8192}" \
    --limit-mm-per-prompt "$MM_LIMIT_JSON" \
    --kv-cache-dtype "${KV_DTYPE:-auto}" \
    --enable-prefix-caching \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS:-3}}" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    ${EXTRA_ARGS:-} )

if [ "${DRY_RUN:-0}" = 1 ]; then _dry_run_dump "${VLLM_ARGV[@]}"; exit 0; fi

"${VLLM_ARGV[@]}" &

VLLM_PID=$!

# Re-harden as soon as it is serving, without waiting for the process to end.
restore_when_ready &

wait "$VLLM_PID"
