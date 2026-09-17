#!/usr/bin/env bash
# qwen-server-record-death.sh -- ExecStopPost hook for qwen-vllm.service.
#
# Runs on EVERY exit of the server, clean or otherwise, and appends one
# structured record to logs/qwen_deaths.log. This is the "clear record when it
# dies" that was missing before: previously the server exited with status 0
# (vLLM shuts itself down when its engine dies), the single log file was
# truncated by the next start, and nothing anywhere said what happened.
#
# systemd hands us these in the environment:
#   $SERVICE_RESULT  -- success | exit-code | signal | timeout | ...
#   $EXIT_CODE       -- exited | killed | dumped
#   $EXIT_STATUS     -- the numeric status, or the signal name
#
# It also pulls the most recent NVIDIA Xid line out of the kernel journal,
# because a GPU MMU fault is the known cause here and it is recorded ONLY in
# the journal -- never in the vLLM log itself.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$HERE/logs"
DEATH_LOG="$LOG_DIR/qwen_deaths.log"
mkdir -p "$LOG_DIR"

{
    echo "=================================================================="
    echo "$(date -Is)  SERVER EXITED"
    echo "  SERVICE_RESULT = ${SERVICE_RESULT:-unset}"
    echo "  EXIT_CODE      = ${EXIT_CODE:-unset}"
    echo "  EXIT_STATUS    = ${EXIT_STATUS:-unset}"

    if [ "${SERVICE_RESULT:-}" = "success" ] && [ "${EXIT_STATUS:-}" = "0" ]; then
        echo "  NOTE: exit status 0 does NOT mean a healthy shutdown here."
        echo "        vLLM's watchdog_loop deliberately exits 0 when EngineCore"
        echo "        dies. Check the GPU fault line and the log tail below."
    fi

    # The decisive evidence for GPU-side failure modes. Requires no privileges:
    # journalctl -k is readable by this user even though dmesg is restricted
    # (kernel.dmesg_restrict=1 on this box).
    #
    # Xid codes are NOT interchangeable -- classify by number, don't assume
    # "an Xid happened" means the documented MTP/MMU-fault pattern (that bug
    # mislabeled a bus-fall-off/reboot-required event as the known MMU fault
    # on 2026-08-24; fixed here). See NVIDIA's Xid reference for codes not
    # listed below.
    echo "  --- most recent NVIDIA Xid in kernel journal ---"
    if xid=$(journalctl -k --since "30 min ago" 2>/dev/null | grep -E "NVRM: Xid" | tail -5) \
       && [ -n "$xid" ]; then
        echo "$xid" | sed 's/^/  /'
        codes=$(echo "$xid" | grep -oE ': [0-9]+,' | grep -oE '[0-9]+' | sort -un)
        for code in $codes; do
            case "$code" in
                13|31)
                    echo "  => Xid $code: GPU MMU fault (illegal/misaligned address)."
                    echo "     This is the known, documented, MTP-correlated crash --"
                    echo "     see LOCAL_LLM_SETUP.md#why-the-server-kept-dying." ;;
                79)
                    echo "  => Xid $code: GPU has fallen off the bus. Hardware/driver-level,"
                    echo "     NOT the documented MMU fault. No process restart fixes this." ;;
                154)
                    echo "  => Xid $code: driver GPU-recovery action asserted -- often means"
                    echo "     'Node Reboot Required'. Check the line above for the actual"
                    echo "     recovery action; no process restart fixes this either." ;;
                *)
                    echo "  => Xid $code: not one of the codes this script recognizes."
                    echo "     Do not assume it's the known MMU-fault pattern -- look it up." ;;
            esac
        done
    else
        echo "  (none in the last 30 minutes -- so this exit was NOT a GPU fault)"
    fi

    echo "  --- last 15 lines of the run log ---"
    latest=$(ls -1t "$LOG_DIR"/qwen_server-*.log 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        echo "  log: $latest"
        tail -15 "$latest" 2>/dev/null | sed 's/^/  | /'
    else
        echo "  (no run log found)"
    fi
    echo
} >> "$DEATH_LOG" 2>&1

exit 0
