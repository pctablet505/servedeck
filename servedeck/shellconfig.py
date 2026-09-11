"""Servedeck's ONLY writer of local_llm/.config — SPEC.md §1 / §9(c).

`.config` is never edited directly (that would bypass save_config_kv()'s
line-preserving semantics and codex-qwen.sh's own validators — see that
script's save_config_kv() docstring: an earlier naive `echo ... >
$CONFIG_FILE` clobbered the whole file). Every mutation in this module goes
through `codex-qwen.sh set-mem|set-subagents|set-config` as a subprocess,
exactly as a human running that script by hand would. There is one
exception: `set_extra_args()`. codex-qwen.sh will not write EXTRA_ARGS, but
Servedeck must keep that key matched to BACKEND, so it writes the key itself
using save_config_kv()'s rule. Its docstring explains why.

The one hard rule this module enforces on Servedeck's behalf (SPEC.md §1):
`set_mem()` in codex-qwen.sh auto-restarts the server as a side effect when
one is already up. Servedeck's own restart sequence is always
stop -> write config -> start, owned by supervisor.py — so `set_util()`
here refuses outright (`ServerRunningError`) rather than ever letting
codex-qwen.sh's implicit restart race Servedeck's own supervisor loop.
"""

from __future__ import annotations

import contextlib
import os
import re
import stat
import subprocess
import tempfile
import urllib.error
import urllib.request
from typing import Mapping

from servedeck import legacy
from servedeck import paths

# codex-qwen.sh's save_config_kv() always emits exactly `KEY="value"\n`
# (one assignment per line, double-quoted, no escaping of embedded quotes)
# — this parses that shape directly rather than sourcing the file as shell.
_KV_LINE_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)="(.*)"$')

# Mirrors codex-qwen.sh's CONFIG_ALLOWED_KEYS (set_config()'s own allow
# list, SPEC.md §9(b)). Not a filesystem path, so it lives here rather than
# in paths.py — but it is a second copy of that list, and codex-qwen.sh
# will independently re-reject anything not on ITS list regardless of what
# this module thinks; keep the two in sync by hand if codex-qwen.sh's ever
# changes.
ALLOWED_SET_KEYS: frozenset[str] = frozenset(
    {
        "BACKEND",
        "MODEL",
        "MODEL_REPO",
        "SERVED_NAME",
        "PORT",
        "MAX_MODEL_LEN",
        "MAX_NUM_SEQS",
        "SERVEDECK_URL",
        "USE_SERVEDECK",
    }
    # DEPRECATED ALIASES. The shell side has not been renamed: codex-qwen.sh's
    # own CONFIG_ALLOWED_KEYS still spells these COLDSTART_*, and .config on
    # this box carries COLDSTART_URL="". Dropping them here would mean
    # Servedeck reads a key that never exists, and set_key("SERVEDECK_URL")
    # would be re-rejected by codex-qwen.sh's own allow list anyway. Both
    # files are owned by another workstream -- see docs/MIGRATION.md for what
    # has to change there before these can go. legacy.DEPRECATED_KEYS is the
    # one list; spelling them out again here is how they drift.
    | frozenset(legacy.DEPRECATED_KEYS)
)

_PROBE_TIMEOUT_S = 2.0
_SUBPROCESS_TIMEOUT_S = 30.0


class ServerRunningError(RuntimeError):
    """`set_util()` was attempted while the inline server answers as up.

    codex-qwen.sh's `set-mem` auto-restarts the server as a side effect
    (SPEC.md §1's hard rule). Servedeck's own restart sequence (stop ->
    write config -> start) must own that restart instead — this exception
    is the refusal that keeps the two from racing.
    """


class ShellConfigError(RuntimeError):
    """A `codex-qwen.sh` subcommand invoked by this module exited non-zero,
    or could not be run at all (missing file, not executable, timed out)."""


def read_config() -> dict[str, str]:
    """Parse local_llm/.config's `KEY="value"` lines into a dict.

    Reading directly (not via `source`) is safe precisely because this
    module is the *only* writer and every write goes through
    save_config_kv()'s fixed line shape — see the module docstring. A
    missing file returns `{}`, matching codex-qwen.sh's own
    `[ -f "$CONFIG_FILE" ] && source "$CONFIG_FILE"`: no file means every
    key just falls back to that script's compiled-in default.
    """
    try:
        text = paths.CONFIG_FILE.read_text()
    except FileNotFoundError:
        return {}
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _KV_LINE_RE.match(line)
        if match:
            out[match.group(1)] = match.group(2)
    return out


def _base_url(cfg: Mapping[str, str]) -> str:
    """Mirrors codex-qwen.sh's recompute_derived(): SERVEDECK_URL, if set
    and non-empty, wins outright; otherwise http://localhost:$PORT/v1.
    PORT itself defaults to "8001" here — codex-qwen.sh's own current
    top-level default (BACKEND=flashnext, PORT=8001) — because
    read_config() only ever sees keys that were actually *written* via
    set-mem/set-subagents/set-config; an empty or PORT-less .config is
    completely normal and must fall back to the same default the shell
    script itself would use, not an arbitrary one.
    """
    # Either spelling: SERVEDECK_URL is what this project calls it, and
    # COLDSTART_URL is the DEPRECATED alias the shell script still writes.
    # Preferring the new name while still honouring the old one is the only
    # way both readers of one file agree during the rename.
    gateway_url = legacy.preferred(cfg, "SERVEDECK_URL")
    if gateway_url:
        return gateway_url.rstrip("/") + "/v1"
    port = cfg.get("PORT", "").strip() or "8001"
    return f"http://localhost:{port}/v1"


def probe_server_up(timeout_s: float = _PROBE_TIMEOUT_S) -> bool:
    """Same criterion codex-qwen.sh's `is_server_up()` uses: GET
    `{BASE_URL}/models` returns HTTP 200. Read-only, loopback-only — the
    kind of request SPEC.md's absolute rule 1b explicitly allows against
    the already-running server. Any failure (connection refused, timeout,
    non-200, malformed URL) is treated as "not up"; this never raises.
    """
    url = _base_url(read_config()) + "/models"
    try:
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _run_codex_qwen(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke `codex-qwen.sh <args>` and raise ShellConfigError on any
    non-zero exit (with stdout/stderr attached) or failure to launch."""
    try:
        result = subprocess.run(
            [str(paths.CODEX_QWEN_SH), *args],
            cwd=str(paths.LOCAL_LLM),
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ShellConfigError(f"failed to run codex-qwen.sh {' '.join(args)}: {exc}") from exc
    if result.returncode != 0:
        raise ShellConfigError(
            f"codex-qwen.sh {' '.join(args)} exited {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result


def set_util(value: float, *, server_up: bool | None = None) -> subprocess.CompletedProcess[str]:
    """Persist GPU_MEM_UTIL via `codex-qwen.sh set-mem <value>`.

    Raises `ServerRunningError` — refusing to run `set-mem` at all — if the
    server answers as up. `server_up`, when given explicitly, is trusted
    as-is (a caller that already knows its own `actual_state`
    authoritatively, i.e. supervisor.py mid-restart-sequence, can skip a
    redundant network probe); left as `None` (the default), this probes
    for itself via `probe_server_up()` so the refusal holds even when
    called from somewhere that has no independent liveness tracking of its
    own.
    """
    if not (0 < value <= 1):
        raise ValueError(f"GPU_MEM_UTIL must be a number in (0, 1], got {value!r}")
    up = probe_server_up() if server_up is None else server_up
    if up:
        raise ServerRunningError(
            "set_util() refused: the server currently answers as up. "
            "Stop it first — Servedeck's restart sequence is always "
            "stop -> write config -> start (SPEC.md §1)."
        )
    return _run_codex_qwen("set-mem", f"{value:.6g}")


def set_subagents(n: int) -> subprocess.CompletedProcess[str]:
    """Persist CODEX_MAX_SUBAGENTS via `codex-qwen.sh set-subagents <n>`.

    Unlike `set_util()`, this never touches the running server — it only
    rewrites Codex's own config.toml, and only takes effect on the next
    Codex launch (codex-qwen.sh's own set_subagents() docstring) — so there
    is no `ServerRunningError` gate here.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"CODEX_MAX_SUBAGENTS must be a positive integer, got {n!r}")
    return _run_codex_qwen("set-subagents", str(n))


def set_key(key: str, value: str) -> subprocess.CompletedProcess[str]:
    """Persist one Servedeck-integration key via
    `codex-qwen.sh set-config <KEY> <VALUE>` (SPEC.md §9(c)'s allow-listed
    escape hatch — BACKEND, MODEL, PORT, etc.). Never restarts anything;
    codex-qwen.sh's own set_config() says so explicitly and independently
    re-validates `key` against its own allow list regardless of what this
    module thinks.
    """
    if key not in ALLOWED_SET_KEYS:
        raise ValueError(f"unknown config key {key!r}; allowed: {sorted(ALLOWED_SET_KEYS)}")
    return _run_codex_qwen("set-config", key, str(value))


#: The one key this module writes itself instead of through codex-qwen.sh.
EXTRA_ARGS_KEY = "EXTRA_ARGS"

#: Characters an EXTRA_ARGS value may not contain. `"`: every line reader of
#: this file (`llm`'s read_config_file, _KV_LINE_RE above, serve-model.sh's
#: sed) accepts only KEY="value" with no double quote inside, and `llm` skips
#: any other line. `$`, backquote and backslash: bin/qwen-server-run.sh SOURCES
#: the file as shell, which expands them inside double quotes, so the 27B's
#: launcher would read a different value from every line reader. Newlines:
#: one assignment is one line.
_UNWRITABLE_CHARS = frozenset('"$`\\\n\r')


def set_extra_args(value: str) -> None:
    """Persist EXTRA_ARGS using exactly codex-qwen.sh's save_config_kv() rule.

    The rule: drop every line that starts with ``EXTRA_ARGS=`` and append
    ``EXTRA_ARGS="value"``. Every other line is kept byte for byte. Every
    reader takes the last assignment, so the appended line is the one they
    all see.

    This is the one write that does not go through codex-qwen.sh, because
    its ``set-config`` allow list does not include EXTRA_ARGS. Servedeck
    still has to write it. EXTRA_ARGS holds one backend's flags, and
    ``bin/qwen-server-run.sh`` (the 27B's launcher) sources ``.config`` after
    its defaults, so the value in the FILE beats anything Servedeck passes in
    the environment. Rewriting BACKEND without rewriting EXTRA_ARGS is how
    the 27B got Flash-Next's ``--prefix-match-unit 208`` on 2026-09-10 22:54
    and again at the 2026-09-11 06:59 reboot, and died both times with
    "Invalid prefix_match_unit=208".

    The file is replaced atomically (temp file in the same directory, then
    os.replace) and keeps its mode (``.config`` is 0600). Raises ValueError
    for a value no reader could read back the same way.
    """
    if not isinstance(value, str):
        raise ValueError(f"EXTRA_ARGS must be a string, got {type(value).__name__}")
    bad = sorted({ch for ch in value if ch in _UNWRITABLE_CHARS})
    if bad:
        raise ValueError(
            f"EXTRA_ARGS cannot be written to {paths.CONFIG_FILE}: it contains {bad!r}, "
            "which `llm` skips and bin/qwen-server-run.sh (which sources the file) "
            "would expand, so the two would launch different servers. "
            f"Value: {value!r}"
        )
    path = paths.CONFIG_FILE
    try:
        text = path.read_text()
        mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        text, mode = "", 0o600
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()                       # the file's own trailing newline
    kept = [ln for ln in lines if not ln.startswith(f"{EXTRA_ARGS_KEY}=")]
    kept.append(f'{EXTRA_ARGS_KEY}="{value}"')
    body = "\n".join(kept) + "\n"

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.servedeck-")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
