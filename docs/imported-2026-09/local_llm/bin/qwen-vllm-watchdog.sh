#!/usr/bin/env bash
# qwen-vllm-watchdog.sh -- periodic check that un-sticks qwen-vllm.service.
#
# WHY THIS EXISTS
# ----------------
# Restart=always + RestartSec=15 recovers from an ordinary crash (the
# documented Xid 13/31 MMU fault -- see LOCAL_LLM_SETUP.md#why-the-server-
# kept-dying). It does NOT recover from two things, both observed live on
# 2026-08-24:
#
#   1. StartLimitBurst=5 / StartLimitIntervalSec=600 is hit (five failures in
#      ten minutes) and the unit enters `failed` for good -- systemd will
#      never restart it again on its own. This is the "stuck" state.
#   2. A deliberate guard-69 stand-down (training marker / not enough VRAM,
#      see qwen-server-run.sh) needed a human to remember to come back and
#      run `systemctl --user start qwen-vllm` once the condition cleared.
#
# This timer-driven script re-arms the unit when it's safe to, and -- just as
# important -- declines to when it isn't, rather than hammering a dead GPU.
#
# THE ONE THING IT CHECKS BEFORE RETRYING: is the GPU even present.
# 2026-08-24's real incident was Xid 79 (GPU fell off the bus) + Xid 154
# ("Node Reboot Required"). Six automatic restarts burned through the burst
# limit in 21 seconds because nothing checked that first -- every attempt
# failed instantly on `NVMLError_Unknown`. A reboot is the only fix for that
# (NVRM says so, not a guess); retrying is pure noise until then.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$HERE/logs"
WATCHDOG_LOG="$LOG_DIR/qwen_watchdog.log"
UNIT="qwen-vllm.service"
mkdir -p "$LOG_DIR"

log() { printf '%s %s\n' "$(date -Is)" "$*" >> "$WATCHDOG_LOG"; }

state="$(systemctl --user is-active "$UNIT" 2>/dev/null || true)"

# Nothing to do if it's already up, or already trying (activating).
case "$state" in
    active|activating|reloading) exit 0 ;;
esac

# Only `failed` (or `inactive` after a guard-69 stand-down) is worth acting
# on. Anything else (unit not found, deactivating, ...) -- leave it alone.
case "$state" in
    failed|inactive) ;;
    *) exit 0 ;;
esac

# Is the GPU actually there? A start attempt against a bus-fallen-off GPU
# fails in well under a second and just re-burns the retry budget.
if ! nvidia-smi -L >/dev/null 2>&1; then
    log "GPU unresponsive (nvidia-smi -L failed) -- not retrying $UNIT. Needs a reboot; see logs/qwen_deaths.log for the Xid that caused it."
    exit 0
fi

log "GPU responsive, $UNIT is '$state' -- clearing and retrying."
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
systemctl --user start "$UNIT" 2>&1 | sed "s/^/  /" >> "$WATCHDOG_LOG" || true
