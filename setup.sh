#!/usr/bin/env bash
# setup.sh — one-time (idempotent, re-runnable) environment setup for
# Coldstart. Never touches port 8000/8001, never starts a model server,
# never runs sudo (SPEC.md's absolute rules). The only process this script
# may itself start is `systemctl --user daemon-reload`, which starts
# nothing — see the systemd section below for why enabling/starting
# coldstart.service is left as a printed, copyable command rather than run
# automatically.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
PYTHON_VERSION="3.13"

echo "== Coldstart setup =="
echo "project root: $HERE"
echo

# ---------------------------------------------------------------- uv -----
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' is not on PATH. Install it first (see https://docs.astral.sh/uv/)." >&2
    exit 1
fi
echo "uv: $(uv --version)"

# ------------------------------------------------------------- venv ------
# THIRD venv, per SPEC.md §1 — NEVER .venv-llm or .venv-next. This is the
# only venv this script (or anything else in this project) may write into.
if [ -x "$VENV/bin/python" ]; then
    echo "venv already present at $VENV — leaving it, just syncing deps below."
else
    echo "Creating venv at $VENV (python $PYTHON_VERSION)..."
    uv venv --python "$PYTHON_VERSION" "$VENV"
fi

echo "Installing pinned deps from requirements.txt..."
uv pip install --python "$VENV/bin/python" -r "$HERE/requirements.txt"

echo
echo "Installed versions:"
"$VENV/bin/python" -c "
import fastapi, httpx, uvicorn
print(f'  fastapi  {fastapi.__version__}')
print(f'  uvicorn  {uvicorn.__version__}')
print(f'  httpx    {httpx.__version__}')
try:
    import pytest
    print(f'  pytest   {pytest.__version__}')
except ImportError:
    print('  pytest   NOT INSTALLED (unexpected)')
"

# ------------------------------------------------------------ state ------
mkdir -p "$HERE/state"
echo
echo "state dir: $HERE/state (desired.json/server.json/history.jsonl/ack.json"
echo "  are created on first write by supervisor.py/procctl.py, not by this script)"

# --------------------------------------------------------- systemd -------
UNIT_NAME="coldstart.service"
USER_UNIT_DIR="$HOME/.config/systemd/user"
SRC_UNIT="$HERE/systemd/$UNIT_NAME"
DST_UNIT="$USER_UNIT_DIR/$UNIT_NAME"

echo
echo "== systemd (--user, no sudo) =="

mkdir -p "$USER_UNIT_DIR"
if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
    cp "$SRC_UNIT" "$DST_UNIT"
    systemctl --user daemon-reload
    echo "Installed $DST_UNIT and reloaded the user systemd daemon."
    echo "NOT enabling or starting it automatically (see below) — when you're ready:"
    echo
    echo "    systemctl --user enable --now coldstart.service"
    echo
    echo "  (then: systemctl --user status coldstart.service"
    echo "         journalctl --user -u coldstart -f)"
else
    echo "No usable systemd --user manager here; skipping unit install."
    echo "The unit file is still at: $SRC_UNIT"
fi

# --- THE EXPLICIT REFUSAL (SPEC.md §1's task 7 / §6's watchdog section) --
#
# This script deliberately does NOT run, and will NEVER run:
#     systemctl --user enable qwen-vllm-watchdog.timer
#     systemctl --user enable --now qwen-vllm-watchdog.timer
#
# Why: that watchdog (bin/qwen-vllm-watchdog.sh) cannot distinguish "the
# user deliberately stopped the server" from "the server crashed" — it
# resurrected a deliberately-stopped unit at least once already (SETUP.md
# :393, and again live at 20:13:58 on the day this spec's corrections were
# written). Coldstart's own supervisor.py fixes exactly this defect by
# gating auto-restart on a persisted `desired_state` that a Stop click
# clears *before* signalling (SPEC.md §6, "THE RULE THAT FIXES THE KNOWN
# WATCHDOG DEFECT"). Running both supervisors at once would let the old,
# defective one fight the new, correct one over the exact same server —
# so this script refuses to turn the old one on, full stop.
#
# This script also does not disable an *already*-enabled watchdog timer —
# that is a real mutation of state this project doesn't own, and it is
# currently disabled anyway (confirmed: `systemctl --user list-unit-files`
# shows qwen-vllm-watchdog.timer as "disabled"). If it were enabled, the
# copyable command to fix that by hand is:
#
#     systemctl --user disable --now qwen-vllm-watchdog.timer
#
echo
echo "== watchdog timer (deliberately NOT touched) =="
if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
    watchdog_state="$(systemctl --user is-enabled qwen-vllm-watchdog.timer 2>/dev/null || true)"
    echo "qwen-vllm-watchdog.timer is-enabled: ${watchdog_state:-not-found}"
    if [ "$watchdog_state" = "enabled" ]; then
        echo "WARNING: it is currently ENABLED. It will fight Coldstart's supervisor"
        echo "  (known defect: it cannot tell a deliberate stop from a crash)."
        echo "  This script will not disable it for you. To do so yourself:"
        echo
        echo "    systemctl --user disable --now qwen-vllm-watchdog.timer"
        echo
    else
        echo "Good — not enabled. setup.sh refuses to enable it (see the comment"
        echo "  block in this script for why) and will never do so."
    fi
else
    echo "(no usable systemd --user manager to check against)"
fi

# ------------------------------------------------------- shell patches ---
echo
echo "== SPEC.md §9 shell patches =="
echo "Not applied by this script (out of scope — those files belong to"
echo "codex-qwen.sh / serve.sh / qwen-server-run.sh, owned elsewhere in this"
echo "project, each patch additive and backed up as *.bak-precoldstart)."
echo "Known-applied already, per live inspection: retry values (C1) and the"
echo "BACKEND!=flashnext systemd gate (§9d)."

echo
echo "== done =="
echo "Run tests:   $VENV/bin/python -m pytest"
echo "Dev server:  ./run.sh"
