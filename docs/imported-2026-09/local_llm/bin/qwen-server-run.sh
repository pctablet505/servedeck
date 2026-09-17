#!/usr/bin/env bash
# qwen-server-run.sh -- ExecStart wrapper for the qwen-vllm.service user unit.
#
# WHY THIS EXISTS
# ---------------
# The vLLM server was dying repeatedly and silently. The mechanism, established
# 2026-08-22 from the kernel journal and the server log (see
# LOCAL_LLM_SETUP.md#why-the-server-kept-dying):
#
#   1. A CUDA kernel in the decode path hits an illegal/misaligned address.
#      The kernel driver records it:
#        NVRM: Xid ...: 31, pid=..., name=VLLM::EngineCor, MMU Fault:
#        ENGINE GRAPHICS ... Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_READ
#   2. That poisons the CUDA context. EngineCore raises
#        torch.AcceleratorError: CUDA error: misaligned address
#      and the EngineCore process exits.
#   3. vLLM's own watchdog_loop (vllm/entrypoints/launcher.py, 5 s poll) sees
#      engine.errored and sets server.should_exit = True.
#   4. uvicorn shuts down cleanly and the process exits with status 0.
#
# Step 4 is the part that matters for supervision: vLLM exits *successfully*
# when its engine dies. `Restart=on-failure` would therefore NEVER restart it.
# The unit uses Restart=always for exactly this reason -- do not "fix" that.
#
# This wrapper adds the two guards the restart loop needs so it can never
# seize the GPU during a scheduled training block, then execs vLLM so systemd
# tracks the real server process as the unit's main PID.

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


HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$HERE/logs"
RUN_DIR="$HERE/run"
# CONFIG_FILE: env-overridable (soak isolation, 2026-09-09) so a caller like
# bin/soak-27b-overnight.sh can source a frozen snapshot of its own instead
# of the live .config, which a concurrent model switch could change out from
# under it. Defaults to the live .config, unchanged, when unset.
CONFIG_FILE="${CONFIG_FILE:-$HERE/.config}"
DEATH_LOG="$LOG_DIR/qwen_deaths.log"

# VLLM_VENV: env-overridable (2026-09-09, T3 of the v0.29.0 validation pass --
# the launcher had no override before this, so testing a second venv meant
# either editing this file or duplicating it).
# Default switched .venv-llm (vLLM 0.27.1) -> .venv-llm-029 (vLLM 0.29.0) on
# 2026-09-11: 0.27.1 at util 0.95 died twice that day (18:16:56 Xid 31 MMU
# fault; 19:26:27 SIGSEGV in CUDAGraph::replay), the GDN/Mamba state-copy race
# that PR #50729 (is_left_overlap in vllm/v1/worker/mamba_utils.py) fixes in
# 0.29.0. .venv-llm stays on disk as the rollback: pass
# VLLM_VENV=.../.venv-llm to use it, and not at util 0.95. Must match
# servedeck.toml's [backends.inline] venv and P_VENV in profiles/qwen27b.env.
VLLM_VENV="${VLLM_VENV:-$HERE/.venv-llm-029}"
MODEL="RadixArk/Qwen3.8-27B-NVFP4"
# PORT default moved 8000 -> 8004 on 2026-09-09: 8000 is permanently held by
# the unrelated ats-optimizer.service (autostarting user unit); .config's
# PORT overrides this anyway via the source below, but keep the fallback
# consistent so a missing .config still boots on the right port.
PORT=8004
GPU_MEM_UTIL="0.47"

mkdir -p "$LOG_DIR" "$RUN_DIR"

# Exit 69 (EX_UNAVAILABLE) is the agreed "do not restart me" signal. The unit
# sets RestartPreventExitStatus=69, so systemd stops trying instead of
# hammering the GPU while training owns it.
EX_UNAVAILABLE=69

# The guard asks the question that actually matters: IS THERE ROOM FOR US?
#
# It used to ask a different one -- "is any non-vLLM process holding more than
# 8 GB?" -- and that is the wrong comparison on a 96 GB card. On 2026-08-22 it
# refused to start three times while 70 GB were free, because three small
# training jobs held 8,854 MiB each against an 8,192 MiB constant. A fixed MiB
# threshold cannot distinguish "the card is busy" from "the card is full", and
# it does not scale with the card or with GPU_MEM_UTIL.
#
# It also could not do the job it was written for: the project's flagship
# training run holds under 10 GB itself, so an 8 GB trip-wire fires on a
# throwaway probe and on the flagship alike.
#
# What replaces it: refuse only when free VRAM cannot fit what vLLM is about to
# ask for (GPU_MEM_UTIL x total) plus a fragmentation margin. That is
# self-adjusting to the card, to the util setting, and to whatever else is
# resident. The explicit "keep off the GPU" mechanism remains the training
# marker in guard 1, which is a deliberate statement of intent rather than a
# guess from a byte count.
#
# QWEN_GPU_HEADROOM_MIB overrides the margin, purely so the guard is testable.
GPU_HEADROOM_MIB="${QWEN_GPU_HEADROOM_MIB:-4096}"

# Presence of ANY of these files means "training in progress -- keep off the
# GPU". Cheap, explicit, and greppable. Create one before a training block and
# remove it afterwards; see LOCAL_LLM_SETUP.md#training-blocks.
TRAINING_MARKERS=(
    "$RUN_DIR/training_in_progress"
    "$HOME/.cache/algotrading/training_in_progress"
    "/home/pctablet505/Projects/AlgoTrading/run/training_in_progress"
)

note() {
    printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$DEATH_LOG" >&2
}

[ -f "$CONFIG_FILE" ] && . "$CONFIG_FILE"

# MODEL_REPO/SERVED_NAME/MAX_MODEL_LEN/MAX_NUM_SEQS are recognised .config
# keys (Coldstart integration, additive/opt-in). Deferred until after the
# source above so a value set in .config actually takes effect -- assigning
# these before sourcing would freeze in the hardcoded default the same way
# codex-qwen.sh's BASE_URL used to (see codex-qwen.sh recompute_derived()).
# Unset in .config today, so MODEL stays exactly "RadixArk/Qwen3.8-27B-NVFP4".
MODEL="${MODEL_REPO:-$MODEL}"

# ------------------------------------------------------------ guard 1: marker
# DRY_RUN (2026-09-09, config-resolution test harness -- see the RESOLVED
# print near the exec line below) skips this and the other startup guards so
# variable resolution can be inspected without a real training marker, VRAM
# state, or venv in play.
[ "${DRY_RUN:-0}" = 1 ] || for marker in "${TRAINING_MARKERS[@]}"; do
    if [ -e "$marker" ]; then
        note "REFUSING TO START: training-in-progress marker present at $marker."
        note "  Remove it (or run: systemctl --user start qwen-vllm) once training is done."
        exit "$EX_UNAVAILABLE"
    fi
done

# --------------------------------------------------------------- guard 2: VRAM
# Sum VRAM held by compute processes that are not this unit's own leftovers.
# A stale EngineCore from our own previous crash is swept below, not counted
# here as "somebody else's training run".
[ "${DRY_RUN:-0}" = 1 ] || if command -v nvidia-smi >/dev/null 2>&1; then
    read -r total_mib used_mib < <(
        nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null |
        head -1 | tr -d ' ' | tr ',' ' '
    )
    case "${total_mib:-}" in ''|*[!0-9]*) total_mib="" ;; esac
    case "${used_mib:-}"  in ''|*[!0-9]*) used_mib=""  ;; esac

    if [ -n "$total_mib" ] && [ -n "$used_mib" ]; then
        # Our own leftovers do not count against us -- they are swept below.
        own_mib=0
        while IFS=, read -r pid mem _rest; do
            pid="${pid// /}"; mem="${mem// /}"; mem="${mem%MiB}"
            [ -z "$pid" ] && continue
            case "$mem" in ''|*[!0-9]*) continue ;; esac
            cmd=""
            [ -r "/proc/$pid/cmdline" ] && cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
            case "$cmd" in
                *"vllm serve"*|*VLLM::EngineCore*) own_mib=$((own_mib + mem)) ;;
            esac
        done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)

        free_mib=$(( total_mib - used_mib + own_mib ))
        want_mib=$(awk -v t="$total_mib" -v u="$GPU_MEM_UTIL" 'BEGIN{printf "%d", t*u}')
        need_mib=$want_mib

        if [ "$free_mib" -lt "$need_mib" ]; then
            note "REFUSING TO START: not enough free VRAM."
            note "  free ${free_mib} MiB (total ${total_mib}, used ${used_mib}, our own ${own_mib})"
            note "  need ${need_mib} MiB at util ${GPU_MEM_UTIL}"
            note "  Holders:"
            nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null |
                while IFS=, read -r p m _r; do
                    p="${p// /}"; [ -z "$p" ] && continue
                    c=""; [ -r "/proc/$p/cmdline" ] && c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
                    note "    pid $p  ${m// /}  ${c:0:110}"
                done
            note "  Lower GPU_MEM_UTIL in ${CONFIG_FILE}, or stop a holder, then start again."
            exit "$EX_UNAVAILABLE"
        fi
        if [ "$free_mib" -lt "$(( want_mib + GPU_HEADROOM_MIB ))" ]; then
            note "WARNING: thin VRAM margin: free ${free_mib} MiB leaves under ${GPU_HEADROOM_MIB} MiB above ${want_mib} MiB budget (util ${GPU_MEM_UTIL})."
        else
            note "VRAM check OK: free ${free_mib} MiB >= need ${need_mib} MiB + ${GPU_HEADROOM_MIB} headroom (util ${GPU_MEM_UTIL})."
        fi
    else
        note "VRAM check SKIPPED: nvidia-smi returned no parseable total/used."
    fi
fi

# --------------------------------------------- sweep our own orphaned children
# A SIGKILL on the API server does not cascade to its EngineCore child (the
# parent normally forwards shutdown; a hard kill bypasses that). An orphan left
# sitting on ~48 GiB makes every subsequent start fail with "Free memory ... is
# less than desired GPU memory utilization". Note the -f: the kernel's `comm`
# field truncates at 15 chars and "VLLM::EngineCore" is 16, so `pgrep -x` never
# matches it.
[ "${DRY_RUN:-0}" = 1 ] || for candidate in $(pgrep -u "$(id -u)" -f "VLLM::EngineCore" 2>/dev/null); do
    ppid=$(awk '/^PPid:/{print $2}' "/proc/$candidate/status" 2>/dev/null)
    parent_cmd=""
    if [ -n "${ppid:-}" ] && [ -r "/proc/$ppid/cmdline" ]; then
        parent_cmd=$(tr '\0' ' ' < "/proc/$ppid/cmdline" 2>/dev/null)
    fi
    case "$parent_cmd" in
        *"vllm serve"*) ;;
        *) note "Sweeping orphaned VLLM::EngineCore pid $candidate"
           kill -KILL "$candidate" 2>/dev/null || true ;;
    esac
done

# Coldstart integration: run every guard above (training marker, VRAM check,
# orphan sweep) without actually launching vllm. Exit 0 means "clear to
# start"; the guards above already exit 69 on their own if not. No effect
# unless something explicitly passes --preflight-only.
[ "${1:-}" = "--preflight-only" ] && exit 0

# ----------------------------------------------------------------- log rotation
# The old launcher used `> "$LOG_FILE"`, truncating the single log on every
# start. That is why the 2026-08-21 23:24 crash left no log at all -- only the
# kernel journal still had the Xid. Keep one timestamped file per run instead,
# with `qwen_server.log` as a symlink to the current one so `tail-log` and the
# existing docs keep working.
# Skipped under DRY_RUN (2026-09-09, second pass): a dry run resolves variables
# and must not touch the log estate. Rotating here unconditionally repointed
# logs/qwen_server.log -- the symlink servedeck tails and `tail-log` follows --
# at a fresh EMPTY file while a real server was still writing the old one,
# appended a false "STARTING vllm serve" line to qwen_deaths.log, and could
# delete the 11th-oldest run log. Proven before the guard by
# tests/test_dry_run_no_side_effects.sh.
if [ "${DRY_RUN:-0}" != 1 ]; then
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_LOG="$LOG_DIR/qwen_server-$STAMP.log"
: > "$RUN_LOG"
ln -sfn "$RUN_LOG" "$LOG_DIR/qwen_server.log"
# Keep the 10 most recent run logs.
ls -1t "$LOG_DIR"/qwen_server-*.log 2>/dev/null | tail -n +11 | xargs -r rm -f

note "STARTING vllm serve $MODEL (util=$GPU_MEM_UTIL, log=$RUN_LOG)"
fi

# ------------------------------------------------------------------- launch
[ "${DRY_RUN:-0}" = 1 ] || if [ ! -x "$VLLM_VENV/bin/vllm" ]; then
    note "FATAL: vLLM venv not found at $VLLM_VENV"
    exit "$EX_UNAVAILABLE"
fi

export CUDA_HOME="$VLLM_VENV/lib/python3.13/site-packages/nvidia/cu13"
export PATH="$VLLM_VENV/bin:$CUDA_HOME/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0

# exec, so systemd's main PID is the real server and its SIGTERM reaches vLLM
# directly rather than a shell that would have to forward it.
#
# EXTRA_ARGS (2026-09-09): found dead here during the Xid-31/13 forensics --
# .config documents it as a "recognised" key (see the comment above MODEL's
# assignment) and vllm-qwen38next/serve.sh actually consumes it, but this
# script never referenced it, so setting EXTRA_ARGS with BACKEND="inline"
# was silently a no-op. Wired in below, unquoted for word-splitting (matches
# serve.sh:155's `${EXTRA_ARGS:-}`). Default is "" (see .config), so this is
# not a behavior change by itself.
#
# The escalation this exists for: this model (GDN-hybrid + Mamba, MTP-4,
# FlashInfer, fp8 KV cache, SM120) matches an open upstream bug family
# (vllm-project/vllm#52225, #54331, #54225 -- Xid 13/31, CUDA-graph-replay
# corruption under sustained load, unfixed as of vLLM 0.29.0) that three
# other operators of this exact RadixArk/Qwen3.8-27B-NVFP4 checkpoint have
# also hit. The one mitigation that consistently survives across every
# report that tried it is `--enforce-eager` (disables CUDA graphs). It was
# NOT added to the default args below: GPU_MEM_UTIL=0.47 has run Xid-free for
# 9+ real days since the Aug 2026 crash bisection (which found the two hits
# at util=.62 and util=.92, never at .47), and enforce-eager costs real
# decode throughput that hasn't been shown necessary at .47. If Xid 13/31
# recurs, set EXTRA_ARGS="--enforce-eager" in .config before raising util
# further.
# DRY_RUN (2026-09-09): print the fully-resolved launch variables and exit 0
# instead of exec'ing vllm. Used by tests/test_soak_config_isolation.sh to
# prove which .config (or CONFIG_FILE override) a given invocation would
# actually launch, without booting a server or touching the GPU.
VLLM_ARGV=( "$VLLM_VENV/bin/vllm" serve "$MODEL" --port "$PORT" \
    --served-model-name "${SERVED_NAME:-$MODEL}" \
    --max-model-len "${MAX_MODEL_LEN:-262144}" \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 4}' \
    --max-num-seqs "${MAX_NUM_SEQS:-128}" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-prefix-caching --mamba-cache-mode all \
    ${EXTRA_ARGS:-} )

if [ "${DRY_RUN:-0}" = 1 ]; then
    # The RESOLVED line predates the argv dump and stays: tests/
    # test_soak_config_isolation.sh greps for it. The ARGV/ENV lines below are
    # what tests/test_serve_model_golden.sh compares against
    # bin/serve-model.sh.
    if [ "${DRY_RUN_FORMAT:-line}" != nul ]; then
        printf 'RESOLVED model=%s port=%s util=%s max_num_seqs=%s max_model_len=%s extra_args=[%s] venv=%s\n' \
            "$MODEL" "$PORT" "$GPU_MEM_UTIL" "${MAX_NUM_SEQS:-128}" "${MAX_MODEL_LEN:-262144}" "${EXTRA_ARGS:-}" "$VLLM_VENV"
    fi
    _dry_run_dump "${VLLM_ARGV[@]}"
    exit 0
fi

exec "${VLLM_ARGV[@]}" >> "$RUN_LOG" 2>&1
