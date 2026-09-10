"""Servedeck path constants — SPEC.md §2.

Every absolute path Servedeck touches, anywhere outside its own project
tree, is a module constant here. No other file in this package may spell
out one of these paths itself — import the constant instead. That is what
makes it possible to trust a `grep` for a stray literal, and it is the only
place a future move of local_llm/ or vllm-qwen38next/ would need editing.

Nothing in this module does I/O beyond `Path.home()` (a getenv, not a
syscall against these specific paths) — importing it is always safe, even
before any of the directories below exist.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Home and sibling-project roots
# --------------------------------------------------------------------------

HOME: Path = Path.home()

# The inline-backend project (Qwen3.8-27B family, codex-qwen.sh, the
# qwen-vllm.service systemd unit). Servedeck reads and shells out into this
# tree; it never writes inside it except via codex-qwen.sh subcommands
# (see shellconfig.py) and its own launched server's log files.
LOCAL_LLM: Path = HOME / "Projects" / "local_llm"

# The flashnext-backend project (Qwen3.8-Flash-Next-NVFP4, its own vLLM
# source build in .venv-next, serve.sh).
VLLM_QWEN38NEXT: Path = HOME / "Projects" / "vllm-qwen38next"

# --------------------------------------------------------------------------
# local_llm/ — inline backend
# --------------------------------------------------------------------------

CODEX_QWEN_SH: Path = LOCAL_LLM / "codex-qwen.sh"
CONFIG_FILE: Path = LOCAL_LLM / ".config"
LOG_DIR: Path = LOCAL_LLM / "logs"
RUN_DIR: Path = LOCAL_LLM / "run"
DEATH_LOG: Path = LOG_DIR / "qwen_deaths.log"
RECORD_DEATH_SH: Path = LOCAL_LLM / "bin" / "qwen-server-record-death.sh"
SERVER_RUN_SH: Path = LOCAL_LLM / "bin" / "qwen-server-run.sh"
VENV_LLM_DIR: Path = LOCAL_LLM / ".venv-llm"

# The inline backend's own PID file / current-run log, written by
# qwen-server-run.sh / codex-qwen.sh. Servedeck only ever reads these.
QWEN_PID_FILE: Path = RUN_DIR / "qwen_server.pid"
QWEN_LOG_FILE: Path = LOG_DIR / "qwen_server.log"

# Training-block markers (qwen-server-run.sh guard 1 — presence of ANY of
# these means "keep off the GPU"). The third lives under a sibling project
# Servedeck otherwise never touches; it is read-only, existence-check only.
TRAINING_MARKERS: tuple[Path, ...] = (
    RUN_DIR / "training_in_progress",
    HOME / ".cache" / "algotrading" / "training_in_progress",
)

# --------------------------------------------------------------------------
# vllm-qwen38next/ — flashnext backend
# --------------------------------------------------------------------------

SERVE_SH: Path = VLLM_QWEN38NEXT / "serve.sh"
VENV_NEXT_DIR: Path = VLLM_QWEN38NEXT / ".venv-next"

# --------------------------------------------------------------------------
# Shared / system
# --------------------------------------------------------------------------

HF_HUB: Path = HOME / ".cache" / "huggingface" / "hub"
PTRACE_PATH: Path = Path("/proc/sys/kernel/yama/ptrace_scope")

# --------------------------------------------------------------------------
# Servedeck's own tree
# --------------------------------------------------------------------------

# This file lives at <project_root>/servedeck/paths.py.
SERVEDECK_PKG_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SERVEDECK_PKG_DIR.parent

# state/desired.json, state/server.json, state/history.jsonl, state/ack.json,
# state/measurements.json — SPEC.md §4 / §6. Created on first write; nothing
# in this module creates it eagerly.
STATE_DIR: Path = PROJECT_ROOT / "state"

# --------------------------------------------------------------------------
# Pre-rename tree (DEPRECATED)
# --------------------------------------------------------------------------

# Servedeck was forked as `coldstart`, and that fork is what has been serving
# :8010 on this box — so it, not this tree, is where the boot history and the
# KV measurements were actually accumulated. Renaming the project without
# reading this directory would not lose the file, but it would lose the
# HISTORY: every model would silently revert to "never booted", the boot-ETA
# statistics would restart from zero, and nothing would report an error.
#
# READ-ONLY, always. Nothing in Servedeck may write inside this tree: the
# coldstart dashboard may still be running out of it, and a writer would be
# corrupting the state of a live process. See legacy.py.
LEGACY_COLDSTART_ROOT: Path = HOME / "Projects" / "coldstart"
LEGACY_COLDSTART_STATE_DIR: Path = LEGACY_COLDSTART_ROOT / "state"

GUI_VENV: Path = PROJECT_ROOT / ".venv"
