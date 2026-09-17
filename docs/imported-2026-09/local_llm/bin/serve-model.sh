#!/usr/bin/env bash
# serve-model.sh -- ONE launcher for every local vLLM model on this box.
#
# WHAT THIS REPLACES
#   vllm-glm53/serve-opt.sh          -> profiles/glm53.env
#   vllm-glm53/serve.sh              -> profiles/glm53-legacy.env
#   vllm-qwen38next/serve-abliterated.sh (-> serve.sh)
#                                    -> profiles/flashnext.env
#   vllm-qwen38next/serve.sh (bare)  -> profiles/flashnext-base.env
#   vllm-qwen38next/serve-tuned.sh   -> profiles/flashnext-tuned.env
#   local_llm/bin/qwen-server-run.sh -> profiles/qwen27b.env
#
# THE CONTRACT: the resolved `vllm serve` argument vector, and the environment
# it is resolved in, are IDENTICAL to the script the profile replaces. That is
# not a claim, it is a test: tests/test_serve_model_golden.sh runs each old
# script and this launcher with DRY_RUN=1 and diffs the two argv dumps
# (order-sensitive) and the two env dumps (order-insensitive). A profile typo
# fails that test. Nothing here "improves" a flag; every value lives in the
# profile and every default is the one the old script had.
#
# WHAT A PROFILE MAY AND MAY NOT SAY
#   A profile carries VALUES (checkpoint, port, offload size, spec depth,
#   which guards apply). It does not carry CODE. Three families -- glm53,
#   flashnext, qwen27b -- differ in argument order, guard set and process
#   model, and each has its own build/preflight function below. Adding a
#   4th model that fits an existing family is a new .env file and nothing
#   else; a genuinely different serving shape needs a new family here.
#
# PRECEDENCE (matches the old scripts exactly)
#   caller environment  >  profile default
# except for the qwen27b family, where $CONFIG_FILE is SOURCED after the
# defaults and therefore wins over both -- which is what bin/qwen-server-run.sh
# does today, and what bin/soak-27b-overnight.sh depends on.
#
# USAGE
#   serve-model.sh <profile>                 # e.g. serve-model.sh glm53
#   PROFILE=glm53 serve-model.sh
#   serve-model.sh --list
#   serve-model.sh <profile> --preflight-only    # guards only, exit 0 if clear
#   DRY_RUN=1 serve-model.sh <profile>       # print env+argv, launch nothing
#   DRY_RUN=1 DRY_RUN_FORMAT=nul ...         # NUL-separated argv, nothing else
#   CONFIG_FILE=/path/.config serve-model.sh qwen27b
#
# NEVER pgrep -f / pkill -f with a pattern that also appears in the caller's
# own command line -- see memory/pgrep-pkill-self-match.md. The one pgrep here
# is the duplicate-launch guard inherited verbatim from serve-opt.sh, and it
# matches a venv path that this script's own command line does not contain.
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERE="$(cd "$SELF_DIR/.." && pwd)"                 # ~/Projects/local_llm
PROFILE_DIR="${PROFILE_DIR:-$HERE/profiles}"
LOG_DIR="$HERE/logs"
RUN_DIR="$HERE/run"

# --- DRY_RUN dump hook -------------------------------------------------------
# Byte-for-byte the same helper the old scripts now carry, so the golden test
# is comparing two dumps produced by the same code, not two formatters.
#   ENV  lines: one per name in $DRY_ENV_VARS (unset -> "<unset>"; HF_TOKEN is
#               hashed so a token never lands in test output).
#   ARGV lines: one per argument, in order.
# DRY_RUN_FORMAT=nul emits the argv NUL-separated and nothing else.
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

# knob NAME -- true when this profile's replaced script honoured $NAME.
knob() { case " ${P_ENV_KNOBS:-} " in *" $1 "*) return 0 ;; esac; return 1; }
die()  { echo "FATAL: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }
dry()  { [ "${DRY_RUN:-0}" = 1 ]; }

# ---------------------------------------------------------------- profile ----
PREFLIGHT_ONLY=0
ARG_PROFILE=""
for a in "$@"; do
    case "$a" in
        --list)            ls -1 "$PROFILE_DIR"/*.env 2>/dev/null | xargs -r -n1 basename | sed 's/\.env$//'; exit 0 ;;
        --preflight-only)  PREFLIGHT_ONLY=1 ;;
        -h|--help)         sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*)                die "unknown option '$a'" ;;
        *)                 [ -n "$ARG_PROFILE" ] && die "more than one profile named ('$ARG_PROFILE', '$a')"; ARG_PROFILE="$a" ;;
    esac
done

# PROFILE resolution, most explicit first. The .config fallback exists so `llm`
# can dispatch here with nothing but the BACKEND it already reads; LAUNCHER_
# PROFILE in .config overrides that mapping for a variant (e.g. flashnext-tuned)
# without changing BACKEND, which other readers (codex-qwen.sh, servedeck) use.
PROFILE="${PROFILE:-$ARG_PROFILE}"
if [ -z "$PROFILE" ] && [ -n "${CONFIG_FILE:-}" ] && [ -f "$CONFIG_FILE" ]; then
    _cfg_profile=$(sed -nE 's/^[[:space:]]*LAUNCHER_PROFILE="([^"]*)".*/\1/p' "$CONFIG_FILE" | tail -1)
    _cfg_backend=$(sed -nE 's/^[[:space:]]*BACKEND="([^"]*)".*/\1/p'          "$CONFIG_FILE" | tail -1)
    if [ -n "$_cfg_profile" ]; then PROFILE="$_cfg_profile"
    else
        case "$_cfg_backend" in
            glm53)     PROFILE=glm53 ;;
            flashnext) PROFILE=flashnext ;;
            inline)    PROFILE=qwen27b ;;
        esac
    fi
fi
[ -n "$PROFILE" ] || die "no profile. Pass one (serve-model.sh glm53), set PROFILE=, or set BACKEND=/LAUNCHER_PROFILE= in \$CONFIG_FILE. Known: $(ls -1 "$PROFILE_DIR"/*.env 2>/dev/null | xargs -r -n1 basename | sed 's/\.env$//' | tr '\n' ' ')"

PROFILE_FILE="$PROFILE_DIR/$PROFILE.env"
[ -f "$PROFILE_FILE" ] || die "no such profile: $PROFILE_FILE"

# Profile defaults, so a profile that omits a key gets a defined value rather
# than a set -u abort. Every one of these is overridden by the profile and/or
# the caller below; none of them is a serving default on its own.
P_FAMILY=""; P_ROOT=""; P_VENV=""; P_RUNDIR=""
P_MODEL=""; P_SERVED_NAME=""; P_PORT=""; P_MAX_LEN=""; P_GPU_UTIL=""
P_MAX_SEQS=""; P_MAX_BATCHED="8192"
P_KV_DTYPE="auto"; P_LOAD_FORMAT="auto"
# P_MM_LIMIT: the --limit-mm-per-prompt argument, i.e. whether the model can
# see images at all. It is a PROFILE value because the scripts disagree: the
# flashnext family enables images ({"image":2,"video":0}, the owner's stated
# requirement, defaulted in vllm-qwen38next/serve.sh and serve-tuned.sh since
# 2026-09-10) while both glm53 scripts hardcode zero. The default below is the
# conservative one; every profile states its own explicitly.
P_MM_LIMIT='{"image":0,"video":0}'
P_SPEC=0; P_SPEC_METHOD="mtp"; P_SPEC_TOKENS=""; P_SPEC_EXTRA=""; P_SPEC_JSON=""
P_AUTOTUNE=1; P_MOE_BACKEND=""; P_CPU_OFFLOAD_GB=""
P_HOT_SLOTS=""; P_HOT_PROFILE=""; P_PREFILL_GROUPS=""; P_BREAKABLE_CUDAGRAPH=""
P_DEEP_GEMM=""; P_WEIGHT_OFFLOAD_DISABLE_PIN=""
P_PYTORCH_ALLOC_CONF=""; P_TORCH_CUDA_ARCH_LIST="12.0f"; P_FLASHINFER_SAMPLER="0"
# P_ENV_KNOBS: the optional environment knobs the replaced script honoured.
# A knob NOT listed is IGNORED here, because the script this profile replaces
# ignored it -- otherwise the unified launcher would quietly accept a knob the
# old path did not, and the two would resolve differently under it. The old
# scripts disagree about all of these: only serve-tuned.sh reads $FI_SAMPLER,
# only serve-opt.sh reads $ENFORCE_EAGER/$BLOCK_SIZE/$SPEC_METHOD/$SPEC_EXTRA,
# only serve-opt.sh and serve-tuned.sh read $AUTOTUNE, and only serve-tuned.sh
# ignores $MAX_BATCHED.
# Recognised: fi_sampler autotune enforce_eager block_size spec_method
#             spec_extra max_batched mm_limit
P_ENV_KNOBS=""
# serve-opt.sh EXPORTED CPU_OFFLOAD_GB into the server's environment; the
# older serve.sh kept it a shell-local. vLLM takes the value from
# --cpu-offload-gb either way, but the environments differ and the golden
# test compares environments.
P_EXPORT_CPU_OFFLOAD=1
P_RAM_OVERHEAD_GIB=""; P_KV_BYTES_PER_TOKEN=""
P_EXTRA_ARGS_POS="none"          # none | mid | end
P_TOOL_PARSER=""; P_REASONING_PARSER=""
P_PLE_OFFLOAD=0; P_PLE_TIMEOUT_BF16=""; P_HF_HUB_OFFLINE=""; P_SWAPPINESS=""
# P_EXPORT_LAUNCH_VARS: names the replaced script EXPORTED into the server's
# environment after resolving them. serve-abliterated.sh does exactly that for
# MODEL/SERVED_NAME/PORT/MAX_LEN/MAX_SEQS/GPU_UTIL/KV_DTYPE before exec'ing
# serve.sh, so the vLLM child sees them; the bare serve.sh and serve-tuned.sh do
# not, and neither does either glm53 script. vLLM itself reads none of these
# (it reads VLLM_*/HF_*/CUDA_*/PYTORCH_*/TORCH_*), so this is fidelity, not
# function -- but "identical environment" has to mean identical.
P_EXPORT_LAUNCH_VARS=""
P_HF_TOKEN_FROM_BASHRC=0; P_NEEDS_PTRACE=0
P_GUARD_CWD_SHADOW=0; P_GUARD_HOST_RAM=0; P_GUARD_PORT=0; P_GUARD_DUP_PATTERN=""
P_GUARD_TRAINING_MARKER=0; P_GUARD_VRAM=0; P_SWEEP_ORPHAN_ENGINECORE=0
P_TRAINING_MARKERS=""; P_GPU_HEADROOM_MIB="4096"
P_READS_CONFIG=0; P_CONFIG_DEFAULT=""
P_LOG_MODE="inherit"             # inherit | rotate
P_LOG_STEM=""; P_LOG_KEEP=10; P_LOG_SYMLINK=""
P_PIDFILE=""; P_DEATH_LOG=""; P_DEATH_HOOK=""; P_EXEC=0
P_LIB64_SHIM=1

# shellcheck disable=SC1090
. "$PROFILE_FILE" || die "could not read $PROFILE_FILE"
[ -n "$P_FAMILY" ] || die "$PROFILE_FILE sets no P_FAMILY"

VENV="$P_VENV"
# DEATH_LOG is env-overridable so a test can run this launcher without
# appending to the real logs/qwen_deaths.log.
DEATH_LOG="${DEATH_LOG:-${P_DEATH_LOG:-}}"
note() {
    if [ -n "$DEATH_LOG" ]; then printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$DEATH_LOG" >&2
    else printf '%s %s\n' "$(date -Is)" "$*" >&2; fi
}

# ------------------------------------------------------- config + overrides ---
# qwen27b only: bin/qwen-server-run.sh sources $CONFIG_FILE AFTER its own
# defaults, so a value in .config beats both the default and the environment.
# That asymmetry is load-bearing -- bin/soak-27b-overnight.sh pins the 27B by
# handing this script a frozen CONFIG_FILE -- so it is reproduced, not tidied.
if [ "$P_READS_CONFIG" = 1 ]; then
    CONFIG_FILE="${CONFIG_FILE:-$P_CONFIG_DEFAULT}"
    VLLM_VENV="${VLLM_VENV:-$P_VENV}"
    MODEL="$P_MODEL"
    PORT="$P_PORT"
    GPU_MEM_UTIL="$P_GPU_UTIL"
    # shellcheck disable=SC1090
    [ -f "$CONFIG_FILE" ] && . "$CONFIG_FILE"
    # Deferred until after the source, exactly as qwen-server-run.sh does it:
    # assigning before would freeze in the hardcoded default.
    MODEL="${MODEL_REPO:-$MODEL}"
    VENV="$VLLM_VENV"
    GPU_HEADROOM_MIB="${QWEN_GPU_HEADROOM_MIB:-$P_GPU_HEADROOM_MIB}"
else
    MODEL="${MODEL:-$P_MODEL}"
    # SERVED_NAME is NOT resolved here: serve.sh and serve-tuned.sh never assign
    # it (they read "${SERVED_NAME:-qwen38-flash-next}" inline), so each family
    # below resolves it the way its own replaced script does.
    PORT="${PORT:-$P_PORT}"
    GPU_HEADROOM_MIB="${QWEN_GPU_HEADROOM_MIB:-$P_GPU_HEADROOM_MIB}"
fi

# ------------------------------------------------------------------ guards ---
guard_cwd_shadow() {
    # $P_ROOT is itself a second, uncompiled vLLM checkout ($P_ROOT/vllm/ next
    # to the real $P_ROOT/src/vllm/). Python puts cwd first on sys.path, so
    # launching from $P_ROOT imports the copy with no compiled extensions and
    # the failure names the wrong subsystem ("vllm.vllm_flash_attn requires the
    # CUDA flash attention extensions"). Run from a directory with no vllm/.
    mkdir -p "$P_RUNDIR"; cd "$P_RUNDIR" || die "cannot cd $P_RUNDIR"
    dry && return 0
    local resolved
    resolved=$("$VENV/bin/python" -c "import vllm; print(vllm.__file__)" 2>/dev/null)
    case "$resolved" in
        "$P_ROOT/src/vllm/"*) ;;
        *) echo "FATAL: vllm resolves to $resolved, expected $P_ROOT/src/vllm/..." >&2
           die "something is shadowing the editable install" ;;
    esac
    echo "vllm:    $resolved"
}

guard_host_ram() {
    # No swap on this box: overshooting host RAM is an OOM kill, not a
    # slowdown, and pinned pages cannot be reclaimed. Refuse instead.
    dry && return 0
    local avail_gib need_gib
    avail_gib=$(awk '/MemAvailable/{m=$2} /SwapFree/{s=$2} END{print int((m+s)/1048576)}' /proc/meminfo)
    need_gib=$(( ${CPU_OFFLOAD_GB%.*} + P_RAM_OVERHEAD_GIB ))
    if [ "$avail_gib" -lt "$need_gib" ]; then
        echo "FATAL: need ~${need_gib} GiB host RAM (${CPU_OFFLOAD_GB} offload + ${P_RAM_OVERHEAD_GIB} overhead)," >&2
        die "but only ${avail_gib} GiB is available. Free memory or lower CPU_OFFLOAD_GB."
    fi
    echo "host RAM: ${avail_gib} GiB available, need ~${need_gib} GiB -- ok"
}

guard_port_free() {
    # A vLLM that OOMs on the GPU HANGS instead of exiting -- it keeps its VRAM
    # and its process tree. A second launch then stalls forever behind it with
    # an empty log, which reads as a slow start rather than a collision.
    dry && return 0
    if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
        echo "FATAL: port $PORT is already bound. Stop the running server first:" >&2
        die "  kill the pid holding it (ss -ltnp | grep :$PORT)"
    fi
}

guard_duplicate_launch() {
    dry && return 0
    [ -n "$P_GUARD_DUP_PATTERN" ] || return 0
    local others
    others=$(pgrep -f "$P_GUARD_DUP_PATTERN" | grep -v "^$$\$" | head -3)
    if [ -n "$others" ]; then
        echo "FATAL: a vllm from this tree is already running (pids: $(echo $others | tr '\n' ' '))." >&2
        die "       It may be a hung GPU-OOM. Kill it before relaunching."
    fi
}

guard_training_marker() {
    # Presence of ANY marker means "training in progress -- keep off the GPU".
    # Exit 69 (EX_UNAVAILABLE) is the agreed do-not-restart-me signal;
    # qwen-vllm.service sets RestartPreventExitStatus=69.
    dry && return 0
    local marker
    for marker in $P_TRAINING_MARKERS; do
        marker="${marker/#\~/$HOME}"
        if [ -e "$marker" ]; then
            note "REFUSING TO START: training-in-progress marker present at $marker."
            note "  Remove it (or run: systemctl --user start qwen-vllm) once training is done."
            exit 69
        fi
    done
}

guard_vram() {
    # Ask the question that matters -- IS THERE ROOM FOR US -- not "is anyone
    # else on the card". A fixed MiB trip-wire refused to start three times on
    # 2026-08-22 while 70 GB were free.
    dry && return 0
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local total_mib used_mib own_mib free_mib want_mib need_mib pid mem cmd
    read -r total_mib used_mib < <(
        nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null |
        head -1 | tr -d ' ' | tr ',' ' '
    )
    case "${total_mib:-}" in ''|*[!0-9]*) total_mib="" ;; esac
    case "${used_mib:-}"  in ''|*[!0-9]*) used_mib=""  ;; esac
    if [ -z "$total_mib" ] || [ -z "$used_mib" ]; then
        note "VRAM check SKIPPED: nvidia-smi returned no parseable total/used."
        return 0
    fi
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
        note "  Lower GPU_MEM_UTIL in ${CONFIG_FILE:-the profile}, or stop a holder, then start again."
        exit 69
    fi
    if [ "$free_mib" -lt "$(( want_mib + GPU_HEADROOM_MIB ))" ]; then
        note "WARNING: thin VRAM margin: free ${free_mib} MiB leaves under ${GPU_HEADROOM_MIB} MiB above ${want_mib} MiB budget (util ${GPU_MEM_UTIL})."
    else
        note "VRAM check OK: free ${free_mib} MiB >= need ${need_mib} MiB + ${GPU_HEADROOM_MIB} headroom (util ${GPU_MEM_UTIL})."
    fi
}

sweep_orphan_enginecore() {
    # A SIGKILL on the API server does not cascade to its EngineCore child. An
    # orphan sitting on ~48 GiB makes every later start fail. -f is required:
    # the kernel's `comm` truncates at 15 chars and "VLLM::EngineCore" is 16.
    dry && return 0
    local candidate ppid parent_cmd
    for candidate in $(pgrep -u "$(id -u)" -f "VLLM::EngineCore" 2>/dev/null); do
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
}

lib64_shim() {
    # FlashInfer's JIT hardcodes "-L$cuda_home/lib64" (flashinfer/jit/cpp_ext.py)
    # but the pip CUDA wheels lay libraries out in lib/. Without this the fused
    # MoE kernel fails to LINK at first forward pass ("Ninja build failed").
    [ "$P_LIB64_SHIM" = 1 ] || return 0
    [ -e "$CUDA_HOME/lib64" ] || ln -sfn lib "$CUDA_HOME/lib64" 2>/dev/null || true
}

# --------------------------------------------------------------- ptrace ------
# PLE offload hands the CPU worker a CUDA IPC handle and torch transfers it with
# pidfd_getfd(), which needs PTRACE_MODE_ATTACH. The two processes are siblings,
# so ptrace_scope=1 denies it and startup dies with "Operation not permitted".
# Relaxed only while the handoff happens, restored the moment the server answers
# -- the box sits at the hardened default the rest of the time.
PTRACE_PATH=/proc/sys/kernel/yama/ptrace_scope
PTRACE_ORIG=""
relax_ptrace() {
    [ "$P_NEEDS_PTRACE" = 1 ] || return 0
    dry && return 0
    [ -r "$PTRACE_PATH" ] || return 0
    PTRACE_ORIG=$(cat "$PTRACE_PATH")
    if [ "$PTRACE_ORIG" != "0" ]; then
        echo "ptrace_scope is $PTRACE_ORIG; PLE offload needs 0. Requesting sudo..."
        sudo sysctl -w kernel.yama.ptrace_scope=0 || die "could not relax ptrace_scope; PLE offload cannot start."
    else
        PTRACE_ORIG=""   # already 0, nothing of ours to restore
    fi
}
restore_ptrace() {
    [ -n "$PTRACE_ORIG" ] || return 0
    # -n: never prompt. The sudo timestamp usually expires during a long run.
    if sudo -n sysctl -w kernel.yama.ptrace_scope="$PTRACE_ORIG" >/dev/null 2>&1; then
        echo "restored ptrace_scope=$PTRACE_ORIG"
    else
        warn "could not restore ptrace_scope (sudo timed out)."
        echo "  Run: sudo sysctl -w kernel.yama.ptrace_scope=$PTRACE_ORIG" >&2
    fi
}
restore_when_ready() {
    [ -n "$PTRACE_ORIG" ] || return 0
    for _ in $(seq 1 150); do   # ~25 min ceiling; cold start is 2-3 min
        if curl -sf -m 3 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
            restore_ptrace; return 0
        fi
        sleep 10
    done
    warn "server never became ready; ptrace_scope left relaxed."
}

# ------------------------------------------------------------- environment ---
setup_env_common() {
    export CUDA_HOME="$VENV/lib/python3.13/site-packages/nvidia/cu13"
    lib64_shim
    export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
    if knob fi_sampler; then
        export VLLM_USE_FLASHINFER_SAMPLER="${FI_SAMPLER:-$P_FLASHINFER_SAMPLER}"
    else
        export VLLM_USE_FLASHINFER_SAMPLER="$P_FLASHINFER_SAMPLER"
    fi
    [ -n "$P_PYTORCH_ALLOC_CONF" ]   && export PYTORCH_ALLOC_CONF="$P_PYTORCH_ALLOC_CONF"
    [ -n "$P_TORCH_CUDA_ARCH_LIST" ] && export TORCH_CUDA_ARCH_LIST="$P_TORCH_CUDA_ARCH_LIST"
}

setup_env_glm53() {
    # These four knobs are a MATCHED SET (offload / hot slots / prefill groups /
    # hot profile), not independent preferences -- see profiles/glm53.env.
    CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-$P_CPU_OFFLOAD_GB}"
    [ "$P_EXPORT_CPU_OFFLOAD" = 1 ] && export CPU_OFFLOAD_GB
    [ -n "$P_HOT_SLOTS" ]            && export VLLM_MOE_HOT_SLOTS="${VLLM_MOE_HOT_SLOTS:-$P_HOT_SLOTS}"
    [ -n "$P_BREAKABLE_CUDAGRAPH" ]  && export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-$P_BREAKABLE_CUDAGRAPH}"
    [ -n "$P_PREFILL_GROUPS" ]       && export VLLM_MOE_PREFILL_GROUPS="${VLLM_MOE_PREFILL_GROUPS:-$P_PREFILL_GROUPS}"
    [ -n "$P_HOT_PROFILE" ]          && export VLLM_MOE_HOT_PROFILE="${VLLM_MOE_HOT_PROFILE:-$P_HOT_PROFILE}"
    [ -n "$P_WEIGHT_OFFLOAD_DISABLE_PIN" ] && export VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY="$P_WEIGHT_OFFLOAD_DISABLE_PIN"
    [ -n "$P_DEEP_GEMM" ]            && export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-$P_DEEP_GEMM}"
    # The hot-expert profile records the offload it was captured at. A mismatch
    # is not fatal (the pinner falls back per layer) but silently costs ~5 tok/s.
    if [ -n "${VLLM_MOE_HOT_SLOTS:-}" ] && [ "${VLLM_MOE_HOT_SLOTS:-0}" != "0" ]; then
        if [ ! -f "${VLLM_MOE_HOT_PROFILE:-}" ]; then
            warn "hot-expert profile '${VLLM_MOE_HOT_PROFILE:-}' not found;"
            echo "         pinning is OFF and throughput drops to ~11-14 tok/s." >&2
        else
            local prof_off
            prof_off=$("$VENV/bin/python" -c "import json,sys;print(json.load(open(sys.argv[1])).get('offload_gb',''))" "$VLLM_MOE_HOT_PROFILE" 2>/dev/null)
            if [ -n "$prof_off" ] && [ "$prof_off" != "${CPU_OFFLOAD_GB%.*}" ]; then
                warn "profile was captured at CPU_OFFLOAD_GB=${prof_off} but this"
                echo "         launch uses ${CPU_OFFLOAD_GB}. Layer indices will not line up;" >&2
                echo "         regenerate the profile or match the offload." >&2
            fi
        fi
    fi
}

setup_env_flashnext() {
    export VLLM_PLE_CPU_OFFLOAD=1
    # VLLM_PLE_FP8_CHECKPOINT must MATCH the checkpoint's PLE n-gram table
    # dtype: RadixArk/Inferact ship it FP8 (model-plefp8-*.safetensors),
    # primitive-ai/mazinb ship it BF16 (ple-bf16-*). Setting the flag for a
    # BF16 table dies on 'ngram_embedding.weight_scale' and not setting it for
    # an FP8 one dies the other way, so DETECT it from the cached snapshot --
    # never hardcode it in a profile. BF16 tables are ~95 GB and blow past the
    # 600 s offload default, so raise the timeout only for those.
    local snap="$HOME/.cache/huggingface/hub/models--${MODEL%%/*}--${MODEL##*/}/snapshots"
    if ls "$snap"/*/model-plefp8-*.safetensors >/dev/null 2>&1; then
        export VLLM_PLE_FP8_CHECKPOINT=1
    else
        export VLLM_PLE_OFFLOAD_READY_TIMEOUT="${VLLM_PLE_OFFLOAD_READY_TIMEOUT:-$P_PLE_TIMEOUT_BF16}"
    fi
    [ -n "$P_HF_HUB_OFFLINE" ] && export HF_HUB_OFFLINE="$P_HF_HUB_OFFLINE"
    if [ "$P_HF_TOKEN_FROM_BASHRC" = 1 ]; then
        HF_TOKEN=$(grep -oP '(?<=export HF_TOKEN=")[^"]+' "$HOME/.bashrc" 2>/dev/null | tail -1)
        export HF_TOKEN
    fi
    # The PLE worker's peak RSS (~103 G) briefly collides with the main
    # worker's weight load. Default swappiness=60 evicts the cold-once-written
    # PLE table to swap instead of dropping clean file pages, and swap faults on
    # PLE lookups cost ~35% of decode throughput.
    if [ -n "$P_SWAPPINESS" ] && ! dry; then
        sudo sysctl -q "vm.swappiness=$P_SWAPPINESS" 2>/dev/null || true
    fi
}

# ---------------------------------------------------------------- argv -------
# MM_LIMIT_RESOLVED is the --limit-mm-per-prompt argument this run will pass.
# Resolved by resolve_mm_limit() below before the argv array is built, never
# inside it: a $(...) substitution would put the `exit 2` in a subshell and the
# script would carry on with an empty argument.
MM_LIMIT_RESOLVED=""
# The SAME regex, message and exit status as vllm-qwen38next/serve.sh and
# serve-tuned.sh, so tests/test_serve_model_golden.sh can assert that a
# malformed MM_LIMIT_JSON fails IDENTICALLY on every launch path rather than
# one path silently dropping it. Accepts {} and {"name": <non-negative int>,
# ...} with optional whitespace -- exactly the shape vLLM parses.
_MM_LIMIT_RE='^\{[[:space:]]*("[A-Za-z_][A-Za-z0-9_]*"[[:space:]]*:[[:space:]]*[0-9]+([[:space:]]*,[[:space:]]*"[A-Za-z_][A-Za-z0-9_]*"[[:space:]]*:[[:space:]]*[0-9]+)*[[:space:]]*)?\}$'
resolve_mm_limit() {
    # $MM_LIMIT_JSON is honoured ONLY where the replaced script honours it --
    # i.e. where the profile lists the mm_limit knob (the flashnext family,
    # from 2026-09-10). serve-opt.sh and vllm-glm53/serve.sh hardcode the flag
    # and ignore the environment, so glm53 profiles must ignore it too or the
    # unified launcher would resolve differently under a variable the old path
    # never read. An empty value counts as unset, matching serve.sh's
    # `[ -z "${MM_LIMIT_JSON:-}" ]`.
    MM_LIMIT_RESOLVED="$P_MM_LIMIT"
    knob mm_limit || return 0
    MM_LIMIT_RESOLVED="${MM_LIMIT_JSON:-$P_MM_LIMIT}"
    if [[ ! "$MM_LIMIT_RESOLVED" =~ $_MM_LIMIT_RE ]]; then
        echo "FATAL: MM_LIMIT_JSON is not strict JSON of \"modality\": <count> pairs: $MM_LIMIT_RESOLVED" >&2
        exit 2
    fi
    # Assign it back, exactly as serve.sh and serve-tuned.sh do, so the shell
    # state -- not just the argv -- matches the replaced script. That is what
    # lets MM_LIMIT_JSON sit in the golden test's $DRY_ENV_VARS vocabulary and
    # be compared as well. Only where the knob applies: serve-opt.sh never
    # assigns it, so the glm53 profiles must not either.
    MM_LIMIT_JSON="$MM_LIMIT_RESOLVED"
}

spec_json() {
    # SPEC_METHOD: mtp (default) | ngram | ngram_gpu | suffix | eagle ...
    # SPEC_EXTRA is raw extra JSON fields, e.g. ',"prompt_lookup_max":6'.
    if [ -n "$P_SPEC_JSON" ]; then printf '%s' "$P_SPEC_JSON"; return 0; fi
    local method="$P_SPEC_METHOD" extra="$P_SPEC_EXTRA"
    knob spec_method && method="${SPEC_METHOD:-$P_SPEC_METHOD}"
    knob spec_extra  && extra="${SPEC_EXTRA:-$P_SPEC_EXTRA}"
    printf '{"method":"%s","num_speculative_tokens":%s%s}' \
        "$method" "${SPEC_TOKENS:-$P_SPEC_TOKENS}" "$extra"
}

build_argv_glm53() {
    local extra_mid=() extra_end=() spec=() eager="" kvflag="" blockflag="" autotune=""
    resolve_mm_limit
    [ "$P_EXTRA_ARGS_POS" = mid ] && extra_mid=(${EXTRA_ARGS:-})
    [ "$P_EXTRA_ARGS_POS" = end ] && extra_end=(${EXTRA_ARGS:-})
    [ "${SPEC:-$P_SPEC}" = 1 ] && spec=(--speculative-config "$(spec_json)")
    # ENFORCE_EAGER: DMA expert staging is data-dependent and needs a
    # device->host sync that CUDA graph capture forbids. At ~207 ms per forward
    # pass graphs save single-digit ms while staging is worth ~5x on the
    # transfer, so trading graphs away is net-positive HERE only.
    knob enforce_eager && [ "${ENFORCE_EAGER:-0}" = 1 ] && eager="--enforce-eager"
    # AUTOTUNE=1 leaves FlashInfer autotune at vLLM's -O2 default (+3.4%
    # measured); 0 restores the inherited --no-enable-flashinfer-autotune.
    if knob autotune; then [ "${AUTOTUNE:-$P_AUTOTUNE}" = 1 ] || autotune="--no-enable-flashinfer-autotune"
    else [ "$P_AUTOTUNE" = 1 ] || autotune="--no-enable-flashinfer-autotune"; fi
    # BLOCK_SIZE: on SM120 DeepGEMM paged-MQA needs block_kv == 64 for the fp8
    # indexer, so block_size must be a multiple of 256.
    knob block_size && [ -n "${BLOCK_SIZE:-}" ] && blockflag="--block-size ${BLOCK_SIZE}"
    # KV_BYTES caps the KV cache instead of letting vLLM size it from LEFTOVER
    # budget. Derived from MAX_LEN so the two can never disagree -- Coldstart
    # passes MAX_LEN but not KV_BYTES, and a stale KV size makes vLLM refuse to
    # start ("N GiB KV cache is needed, larger than the available").
    if [ -n "$P_KV_BYTES_PER_TOKEN" ]; then
        KV_BYTES="${KV_BYTES:-$(( ${MAX_LEN:-$P_MAX_LEN} * P_KV_BYTES_PER_TOKEN ))}"
        [ -n "$KV_BYTES" ] && kvflag="--kv-cache-memory-bytes ${KV_BYTES}"
    fi
    VLLM_ARGV=( "$VENV/bin/vllm" serve "$MODEL"
        --served-model-name "$SERVED_NAME"
        --host 0.0.0.0 --port "$PORT"
        --max-model-len "${MAX_LEN:-$P_MAX_LEN}"
        --gpu-memory-utilization "${GPU_UTIL:-$P_GPU_UTIL}"
        --tensor-parallel-size 1
        --distributed-executor-backend mp
        --offload-backend uva
        --cpu-offload-gb "$CPU_OFFLOAD_GB"
        --cpu-offload-params experts
        --max-num-seqs "${MAX_SEQS:-$P_MAX_SEQS}"
        --max-num-batched-tokens "$(knob max_batched && printf '%s' "${MAX_BATCHED:-$P_MAX_BATCHED}" || printf '%s' "$P_MAX_BATCHED")"
        --limit-mm-per-prompt "$MM_LIMIT_RESOLVED"
        ${ATTN_BACKEND:+--attention-backend "$ATTN_BACKEND"}
        ${MOE_BACKEND:+--kernel-config "{\"moe_backend\":\"$MOE_BACKEND\"}"}
        "${extra_mid[@]}"
        --load-format "${LOAD_FORMAT:-$P_LOAD_FORMAT}"
        --kv-cache-dtype "${KV_DTYPE:-$P_KV_DTYPE}"
        --enable-prefix-caching
        ${eager}
        ${kvflag}
        ${blockflag}
        ${autotune}
        "${spec[@]}"
        --enable-auto-tool-choice
        --tool-call-parser "$P_TOOL_PARSER"
        --reasoning-parser "$P_REASONING_PARSER"
        "${extra_end[@]}" )
}

build_argv_flashnext() {
    local extra_end=() autotune="" batched
    resolve_mm_limit
    [ "$P_EXTRA_ARGS_POS" = end ] && extra_end=(${EXTRA_ARGS:-})
    if knob autotune; then [ "${AUTOTUNE:-$P_AUTOTUNE}" = 1 ] || autotune="--no-enable-flashinfer-autotune"
    else [ "$P_AUTOTUNE" = 1 ] || autotune="--no-enable-flashinfer-autotune"; fi
    # serve-tuned.sh hardcoded 8192 and ignored MAX_BATCHED; serve.sh honoured
    # it -- hence the max_batched knob, rather than quietly picking one.
    if knob max_batched; then batched="${MAX_BATCHED:-$P_MAX_BATCHED}"; else batched="$P_MAX_BATCHED"; fi
    VLLM_ARGV=( "$VENV/bin/vllm" serve "$MODEL"
        --served-model-name "${SERVED_NAME:-$P_SERVED_NAME}"
        --host 0.0.0.0 --port "$PORT"
        --max-model-len "${MAX_LEN:-$P_MAX_LEN}"
        --gpu-memory-utilization "${GPU_UTIL:-$P_GPU_UTIL}"
        --tensor-parallel-size 1
        --distributed-executor-backend mp
        --max-num-seqs "${MAX_SEQS:-$P_MAX_SEQS}"
        --max-num-batched-tokens "$batched"
        --limit-mm-per-prompt "$MM_LIMIT_RESOLVED"
        --kv-cache-dtype "${KV_DTYPE:-$P_KV_DTYPE}"
        --enable-prefix-caching
        ${autotune}
        --speculative-config "$(spec_json)"
        --enable-auto-tool-choice
        --tool-call-parser "$P_TOOL_PARSER"
        --reasoning-parser "$P_REASONING_PARSER"
        "${extra_end[@]}" )
}

build_argv_qwen27b() {
    # No --host here: bin/qwen-server-run.sh never passed one, so the 27B binds
    # vLLM's own default rather than 0.0.0.0. Preserved deliberately.
    # The spec JSON is written with the same interior spaces the old script
    # used, because the golden test compares the argument byte for byte.
    VLLM_ARGV=( "$VENV/bin/vllm" serve "$MODEL" --port "$PORT"
        --served-model-name "${SERVED_NAME:-$MODEL}"
        --max-model-len "${MAX_MODEL_LEN:-$P_MAX_LEN}"
        --enable-auto-tool-choice --tool-call-parser "$P_TOOL_PARSER"
        --reasoning-parser "$P_REASONING_PARSER"
        --speculative-config "$(spec_json)"
        --max-num-seqs "${MAX_NUM_SEQS:-$P_MAX_SEQS}"
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --enable-prefix-caching --mamba-cache-mode all
        ${EXTRA_ARGS:-} )
}

# ------------------------------------------------------------------ logging --
open_log() {
    # rotate: one timestamped file per run with a stable symlink, keeping the
    # newest $P_LOG_KEEP. The old launcher used `> "$LOG_FILE"`, which is why
    # the 2026-08-21 23:24 crash left no log at all.
    # inherit: this script writes to its own stdout/stderr and the caller (llm
    # cmd_start) owns the redirect and the rotation.
    [ "$P_LOG_MODE" = rotate ] || return 0
    dry && return 0
    mkdir -p "$LOG_DIR"
    local stamp; stamp="$(date +%Y%m%d-%H%M%S)"
    RUN_LOG="$LOG_DIR/${P_LOG_STEM}-$stamp.log"
    : > "$RUN_LOG"
    [ -n "$P_LOG_SYMLINK" ] && ln -sfn "$RUN_LOG" "$LOG_DIR/$P_LOG_SYMLINK"
    ls -1t "$LOG_DIR/${P_LOG_STEM}-"*.log 2>/dev/null | tail -n +$((P_LOG_KEEP + 1)) | xargs -r rm -f
}

# The pidfile is SHARED MUTABLE STATE between launcher instances, so this
# launcher only ever writes one it is not stealing, and only ever deletes one it
# still owns. Without both halves, a second launch overwrites the record of the
# first and the first's cleanup then deletes the second's -- which is exactly
# how `llm` came to SIGTERM two healthy servers on 2026-09-10. P_PIDFILE is
# unset in every shipped profile today; this is fixed now so that setting it
# does not reintroduce the defect at the LAUNCHER=unified cutover.
P_PIDFILE_WRITTEN=""
write_pidfile() {
    [ -n "$P_PIDFILE" ] || return 0
    mkdir -p "$RUN_DIR"
    local f="${P_PIDFILE/#\~/$HOME}" rec
    rec=$(tr -d ' \n' < "$f" 2>/dev/null || true)
    if [ -n "$rec" ] && [ "$rec" != "$1" ] && kill -0 "$rec" 2>/dev/null; then
        echo "WARNING: $f already names live pid $rec -- not overwriting it; this launch is pid $1" >&2
        return 0
    fi
    printf '%s\n' "$1" > "$f"
    P_PIDFILE_WRITTEN="$1"
}

cleanup() {
    # Only the child THIS script forked, verified by PPid so a reused pid can
    # never be signalled. Never a port, never a pattern, never a group.
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null \
       && [ "$(awk '/^PPid:/{print $2}' "/proc/$VLLM_PID/status" 2>/dev/null)" = "$$" ]; then
        echo "stopping vllm (pid $VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null
        for _ in $(seq 1 30); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
        kill -0 "$VLLM_PID" 2>/dev/null && kill -KILL "$VLLM_PID" 2>/dev/null
    fi
    if [ -n "$P_PIDFILE" ] && [ -n "$P_PIDFILE_WRITTEN" ]; then
        local f="${P_PIDFILE/#\~/$HOME}"
        [ "$(tr -d ' \n' < "$f" 2>/dev/null || true)" = "$P_PIDFILE_WRITTEN" ] && rm -f "$f"
    fi
    restore_ptrace
    # Post-exit death record: the same structured entry qwen-vllm.service gets
    # from its ExecStopPost hook, for the paths where systemd is not the
    # supervisor. Skipped when this script execs (systemd still runs the hook).
    if [ -n "${P_DEATH_HOOK:-}" ] && [ -x "${P_DEATH_HOOK}" ]; then
        SERVICE_RESULT="${SERVICE_RESULT:-unknown}" "$P_DEATH_HOOK" || true
    fi
}

# =============================================================== main ========
case "$P_FAMILY" in
  glm53)
    SERVED_NAME="${SERVED_NAME:-$P_SERVED_NAME}"
    MOE_BACKEND="${MOE_BACKEND:-$P_MOE_BACKEND}"
    [ "$P_GUARD_CWD_SHADOW" = 1 ] && guard_cwd_shadow
    setup_env_common
    setup_env_glm53
    [ "$P_GUARD_HOST_RAM" = 1 ] && guard_host_ram
    [ "$P_GUARD_PORT" = 1 ] && guard_port_free
    guard_duplicate_launch
    [ "$PREFLIGHT_ONLY" = 1 ] && exit 0
    trap cleanup EXIT INT TERM
    echo "model:   $MODEL"
    echo "offload: ${CPU_OFFLOAD_GB} GiB of experts -> host RAM (uva)"
    echo "serving: $SERVED_NAME on :$PORT, max_len ${MAX_LEN:-$P_MAX_LEN}"
    build_argv_glm53
    ;;
  flashnext)
    # SERVED_NAME is resolved into a shell variable ONLY for the profile whose
    # replaced script did that (serve-abliterated.sh, via the export block
    # below). The bare serve.sh and serve-tuned.sh never assign it -- they read
    # "${SERVED_NAME:-qwen38-flash-next}" inline -- so assigning it here would
    # leave the launcher holding a variable the replaced script does not have.
    if [ -n "$P_EXPORT_LAUNCH_VARS" ]; then
        SERVED_NAME="${SERVED_NAME:-$P_SERVED_NAME}"
        # Resolve first, then export -- the values are the ones the argv below
        # uses, so this cannot make the environment and the command line
        # disagree. Assigning MAX_LEN/MAX_SEQS/GPU_UTIL/KV_DTYPE here is a
        # no-op for the argv, which reads the same "${X:-$P_X}" expressions.
        MAX_LEN="${MAX_LEN:-$P_MAX_LEN}"; MAX_SEQS="${MAX_SEQS:-$P_MAX_SEQS}"
        GPU_UTIL="${GPU_UTIL:-$P_GPU_UTIL}"; KV_DTYPE="${KV_DTYPE:-$P_KV_DTYPE}"
        # shellcheck disable=SC2086
        export $P_EXPORT_LAUNCH_VARS
    fi
    setup_env_common
    setup_env_flashnext
    [ "$P_GUARD_PORT" = 1 ] && guard_port_free
    guard_duplicate_launch
    [ "$PREFLIGHT_ONLY" = 1 ] && exit 0
    relax_ptrace
    trap cleanup EXIT INT TERM
    echo "MODEL=$MODEL SERVED_NAME=${SERVED_NAME:-$P_SERVED_NAME} PORT=$PORT GPU_UTIL=${GPU_UTIL:-$P_GPU_UTIL} MAX_LEN=${MAX_LEN:-$P_MAX_LEN}"
    build_argv_flashnext
    ;;
  qwen27b)
    [ "$P_GUARD_TRAINING_MARKER" = 1 ] && guard_training_marker
    [ "$P_GUARD_VRAM" = 1 ] && guard_vram
    [ "$P_SWEEP_ORPHAN_ENGINECORE" = 1 ] && sweep_orphan_enginecore
    [ "$PREFLIGHT_ONLY" = 1 ] && exit 0
    open_log
    # Not under DRY_RUN: note() appends to $DEATH_LOG (logs/qwen_deaths.log by
    # default), and a dry run must leave the log estate exactly as it found it.
    # Same guard as bin/qwen-server-run.sh's rotation block; proven by
    # tests/test_dry_run_no_side_effects.sh.
    dry || note "STARTING vllm serve $MODEL (util=$GPU_MEM_UTIL, log=${RUN_LOG:-<inherited>})"
    if ! dry && [ ! -x "$VENV/bin/vllm" ]; then
        note "FATAL: vLLM venv not found at $VENV"
        exit 69
    fi
    export CUDA_HOME="$VENV/lib/python3.13/site-packages/nvidia/cu13"
    export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
    export VLLM_USE_FLASHINFER_SAMPLER="$P_FLASHINFER_SAMPLER"
    build_argv_qwen27b
    ;;
  *) die "profile $PROFILE names an unknown P_FAMILY '$P_FAMILY'" ;;
esac

if [ "${DRY_RUN:-0}" = 1 ]; then
    if [ "$P_FAMILY" = qwen27b ] && [ "${DRY_RUN_FORMAT:-line}" != nul ]; then
        # Kept identical to bin/qwen-server-run.sh's line: anything that greps
        # for it (tests/test_soak_config_isolation.sh) keeps working after the
        # cutover.
        printf 'RESOLVED model=%s port=%s util=%s max_num_seqs=%s max_model_len=%s extra_args=[%s] venv=%s\n' \
            "$MODEL" "$PORT" "$GPU_MEM_UTIL" "${MAX_NUM_SEQS:-$P_MAX_SEQS}" "${MAX_MODEL_LEN:-$P_MAX_LEN}" "${EXTRA_ARGS:-}" "$VENV"
    fi
    _dry_run_dump "${VLLM_ARGV[@]}"
    exit 0
fi

if [ "$P_EXEC" = 1 ]; then
    # exec, so systemd's main PID is the real server and its SIGTERM reaches
    # vLLM directly rather than a shell that would have to forward it.
    if [ -n "${RUN_LOG:-}" ]; then exec "${VLLM_ARGV[@]}" >> "$RUN_LOG" 2>&1
    else exec "${VLLM_ARGV[@]}"; fi
fi

"${VLLM_ARGV[@]}" &
VLLM_PID=$!
write_pidfile "$VLLM_PID"
# Re-harden as soon as it is serving, without waiting for the process to end.
[ "$P_NEEDS_PTRACE" = 1 ] && restore_when_ready &
wait "$VLLM_PID"
