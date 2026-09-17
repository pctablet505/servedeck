#!/usr/bin/env bash
# battery-watchdog.sh — pauses the audit campaign when the UPS goes on battery.
# UPS = USB HID device (ups_hiddev0). Polls upower every 15 s.
#   on battery : writes run/BATTERY_PAUSE, tries to stop qwen-vllm.service
#   back on AC : writes run/POWER_RESUMED
# The orchestrator (codex) checks the flags each turn and pauses/resumes.
HERE="/home/pctablet505/Projects/local_llm"
RUN="$HERE/run"; LOG="$HERE/logs/battery-watchdog.log"
DEV="/org/freedesktop/UPower/devices/ups_hiddev0"
mkdir -p "$RUN" "$HERE/logs"
while :; do
  state=$(upower -i "$DEV" 2>/dev/null | awk '/state:/{print $2; exit}')
  onbatt=$(upower -i "$DEV" 2>/dev/null | awk '/on-battery:/{print $2; exit}')
  if [ "$state" = "discharging" ] || [ "$onbatt" = "yes" ]; then
    if [ ! -f "$RUN/BATTERY_PAUSE" ]; then
      echo "$(date '+%F %T') UPS on battery (state=$state on-batt=$onbatt) -> PAUSING" >> "$LOG"
      echo "$(date '+%F %T') state=$state" > "$RUN/BATTERY_PAUSE"
      systemctl --user stop qwen-vllm 2>>"$LOG" || echo "$(date '+%F %T') systemctl stop failed (orchestrator/user must stop qwen-vllm)" >> "$LOG"
    fi
  else
    if [ -f "$RUN/BATTERY_PAUSE" ] && [ ! -f "$RUN/POWER_RESUMED" ]; then
      echo "$(date '+%F %T') back on AC (state=$state) -> RESUME" >> "$LOG"
      echo "$(date '+%F %T')" > "$RUN/POWER_RESUMED"
      rm -f "$RUN/BATTERY_PAUSE"
    fi
  fi
  sleep 15
done
