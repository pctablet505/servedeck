"""Coldstart process control — SPEC.md §2, "the self-match trap".

SETUP.md:401 recorded a real incident: the pattern-matching lookup tools in
this family (their names are deliberately never spelled out below — a
regex-anchored token match runs against this file's own source in CI-style
checks, and this module must never trip it, comments included) match
against the FULL COMMAND LINE, including the calling shell's own. Filtering
a process list by a substring of its own invocation kills the shell that
ran the filter. That happened repeatedly during this project's development.

The five rules this module follows instead:

1. Never invoke any command-line tool that greps a process table by regex
   against full cmdline text. Enumerate /proc directly and compare fields
   ourselves; go through ``ss`` (socket-table lookup, not process-table
   pattern matching) for "who holds this port".
2. A launched server gets ``start_new_session=True`` (the ``Popen``
   equivalent of ``setsid``): its pid becomes both its own pgid and its own
   session id, so it survives Coldstart being restarted and can be
   signalled as a whole group without touching anything else on the box.
   The resulting handle is recorded (pid, pgid, argv, cwd, log path) and
   persisted so a restarted Coldstart can find it again.
3. Stopping a group always signals the *group*
   (``os.killpg``), never a bare pid — vLLM's own EngineCore/Worker
   children are not the process ``stop()`` was handed. SIGTERM first,
   SIGKILL only after a timeout. Before signalling anything we assert the
   target pgid is neither Coldstart's own process group nor its own pid;
   refuse rather than ever signal our own tree.
4. A server Coldstart did not launch (found by whatever is listening on the
   configured port) is only ever "adopted" after cross-checking it really
   looks like a vLLM server living in one of the two known venvs. A port
   that answers with no attributable pid is reported as unmanaged, not
   guessed at.
5. Orphan detection reads ``comm`` (the kernel's 15-byte-truncated process
   name — ``VLLM::EngineCore`` truncates to ``VLLM::EngineCor``, so the
   live predicate is a prefix match on ``VLLM::``), never cmdline text
   matching. A ``VLLM::*`` process is orphaned iff its parent's cmdline
   does not contain the two words that mean "this is a live vllm server":
   v-l-l-m space s-e-r-v-e (spelled out here only as literal Python source
   the reader can see is a fixed 9-character needle, not a shell pattern
   handed to any table-scanning tool).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from coldstart import paths

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

#: Role of one live process in a vLLM tree, as classified from `comm`.
_ROLE_SERVER = "server"
_ROLE_ENGINE_CORE = "engine_core"
_ROLE_WORKER = "worker"
_ROLE_OTHER_VLLM = "vllm_internal"


@dataclass(frozen=True)
class VllmProc:
    """One live process belonging to (or plausibly belonging to) a vLLM
    tree, as found by a full /proc scan."""

    pid: int
    ppid: int
    pgid: int
    comm: str
    cmdline: list[str]
    cwd: str | None
    exe: str | None
    role: str  # "server" | "engine_core" | "worker" | "vllm_internal"
    venv: str | None  # "next" | "llm" | None (unattributed)
    port: int | None  # populated only when role == "server" and it holds a listening socket


@dataclass(frozen=True)
class ServerHandle:
    """Everything needed to find and stop a server Coldstart launched."""

    pid: int
    pgid: int
    argv: list[str]
    cwd: str
    log_path: str
    started_at: float  # time.time() at launch

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ServerHandle":
        return cls(
            pid=int(data["pid"]),  # type: ignore[arg-type]
            pgid=int(data["pgid"]),  # type: ignore[arg-type]
            argv=list(data["argv"]),  # type: ignore[arg-type]
            cwd=str(data["cwd"]),
            log_path=str(data["log_path"]),
            started_at=float(data["started_at"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class StopResult:
    """Outcome of a stop() call."""

    ok: bool
    method: str  # "already_dead" | "sigterm" | "sigkill" | "refused"
    waited_s: float
    detail: str


# ---------------------------------------------------------------------------
# Low-level /proc readers — every one tolerates a pid vanishing mid-read
# (the process table is inherently racy) by returning None/[] rather than
# raising.
# ---------------------------------------------------------------------------


def _read_comm(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return None


def _read_ppid(pid: int) -> int | None:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("PPid:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return None
    return None


def _read_cmdline_list(pid: int) -> list[str]:
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    if not data:
        return []
    text = data.decode("utf-8", errors="replace")
    return [part for part in text.split("\x00") if part != ""]


def _read_cmdline_raw(pid: int) -> str | None:
    """Space-joined cmdline, or None if the pid is gone/unreadable."""
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return data.replace(b"\x00", b" ").decode("utf-8", errors="replace")


def _read_cwd(pid: int) -> str | None:
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return None


def _read_exe(pid: int) -> str | None:
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return None


def _iter_pids() -> list[int]:
    pids: list[int] = []
    try:
        entries = os.scandir("/proc")
    except OSError:
        return pids
    with entries:
        for entry in entries:
            if entry.name.isdigit():
                pids.append(int(entry.name))
    return pids


def _classify_role(comm: str) -> str | None:
    if comm == "vllm":
        return _ROLE_SERVER
    if comm.startswith("VLLM::Engine"):
        return _ROLE_ENGINE_CORE
    if comm.startswith("VLLM::Worker"):
        return _ROLE_WORKER
    if comm.startswith("VLLM::"):
        return _ROLE_OTHER_VLLM
    return None


def _detect_venv(cmdline: Sequence[str], cwd: str | None) -> str | None:
    """Rule 4's "exe/cwd resolving inside .venv-next or .venv-llm".

    The api-server process's cmdline literally contains the venv's
    ``bin/python``/``bin/vllm`` paths (confirmed live). Its EngineCore /
    Worker children have no useful cmdline (multiprocessing.spawn rewrites
    argv down to just the comm string) — for those, cwd is what's left,
    and it is the project root that *contains* the venv (vLLM does not
    chdir into the venv itself), so a cwd prefix match against the project
    root is the correct — not merely a fallback — test for that case.
    """
    joined = " ".join(cmdline)
    next_dir = str(paths.VENV_NEXT_DIR)
    next_root = str(paths.VLLM_QWEN38NEXT)
    llm_dir = str(paths.VENV_LLM_DIR)
    llm_root = str(paths.LOCAL_LLM)

    if next_dir in joined or (cwd is not None and (cwd == next_root or cwd.startswith(next_root + "/"))):
        return "next"
    if llm_dir in joined or (cwd is not None and (cwd == llm_root or cwd.startswith(llm_root + "/"))):
        return "llm"
    return None


# ---------------------------------------------------------------------------
# ss-based port lookup (rule 1: socket table, never a process-table pattern
# match)
# ---------------------------------------------------------------------------

_PID_RE = re.compile(r"pid=(\d+)")


def listener_pid(port: int) -> int | None:
    """PID of whatever process holds the listening socket on `port`, or
    None if nothing is listening (or `ss` itself is unavailable).

    This makes no claim about *what* that process is — a caller that needs
    to know whether it is safe to adopt should cross-check the result with
    :func:`is_attributable` or :func:`scan_vllm_processes` (SPEC.md §2 rule
    4: a port that answers with no attributable pid is UNMANAGED, not
    guessed at).
    """
    try:
        result = subprocess.run(
            ["ss", "-H", "-ltnp", f"sport = :{port}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in result.stdout.splitlines():
        match = _PID_RE.search(line)
        if match:
            return int(match.group(1))
    return None


def _listening_ports_by_pid() -> dict[int, list[int]]:
    """One `ss` call covering every listening TCP socket, for scan_vllm_processes
    to attribute ports to server-role pids without one `ss` call per pid."""
    try:
        result = subprocess.run(
            ["ss", "-H", "-ltnp"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    out: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        local = parts[3]
        if ":" not in local:
            continue
        port_str = local.rsplit(":", 1)[1]
        try:
            port = int(port_str)
        except ValueError:
            continue
        pid_match = _PID_RE.search(line)
        if not pid_match:
            continue
        pid = int(pid_match.group(1))
        out.setdefault(pid, []).append(port)
    return out


def is_attributable(pid: int) -> bool:
    """SPEC.md §2 rule 4's cross-check, standalone: does this pid look like
    a genuine vLLM api-server process living in one of the two known venvs?
    """
    comm = _read_comm(pid)
    if comm != "vllm":
        return False
    cmdline = _read_cmdline_list(pid)
    cwd = _read_cwd(pid)
    return _detect_venv(cmdline, cwd) is not None


# ---------------------------------------------------------------------------
# Full-tree scan
# ---------------------------------------------------------------------------


def scan_vllm_processes() -> list[VllmProc]:
    """Enumerate every live process that looks like part of a vLLM tree
    (the api-server process itself, plus its ``VLLM::*``-named children),
    by direct /proc iteration — never a process-table pattern-match tool.
    """
    port_map = _listening_ports_by_pid()
    procs: list[VllmProc] = []
    for pid in _iter_pids():
        comm = _read_comm(pid)
        if comm is None:
            continue
        role = _classify_role(comm)
        if role is None:
            continue
        cmdline = _read_cmdline_list(pid)
        cwd = _read_cwd(pid)
        exe = _read_exe(pid)
        ppid = _read_ppid(pid)
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            continue
        venv = _detect_venv(cmdline, cwd)
        ports = port_map.get(pid) if role == _ROLE_SERVER else None
        port = ports[0] if ports else None
        procs.append(
            VllmProc(
                pid=pid,
                ppid=ppid if ppid is not None else 0,
                pgid=pgid,
                comm=comm,
                cmdline=cmdline,
                cwd=cwd,
                exe=exe,
                role=role,
                venv=venv,
                port=port,
            )
        )
    return procs


# ---------------------------------------------------------------------------
# Orphan sweep — rule 5
# ---------------------------------------------------------------------------

# Spelled out as a literal Python string comparison, never handed to any
# process-table pattern-matching tool.
_VLLM_SERVE_NEEDLE = "vllm serve"

# Safety cap on the ancestor walk below — real chains terminate in 1-2 hops
# (VLLM::Worker -> VLLM::EngineCore -> vllm), this only guards a pathological
# /proc read racing a fork storm from looping indefinitely.
_MAX_ANCESTOR_HOPS = 16


def _root_ancestor_cmdline(start_pid: int) -> str | None:
    """Walk up the parent chain past any further ``VLLM::*``-named
    ancestors and return the first non-``VLLM::*`` ancestor's cmdline.

    Needed because the tree is two levels deep: a ``VLLM::Worker``'s
    *immediate* parent is ``VLLM::EngineCore``, not the ``vllm serve``
    process — checking only the immediate parent's cmdline for "vllm
    serve" would misclassify every healthy Worker as orphaned (confirmed
    against the live tree: EngineCore's own cmdline is just
    "VLLM::EngineCore", never "vllm serve"). Walking past VLLM::-named
    hops finds the real root: the ``vllm`` api-server process for a
    healthy tree, or whatever the OS reparented an orphan to (systemd/init)
    for a genuinely severed one — neither of which needs special-casing
    once the walk reaches it.
    """
    pid = start_pid
    for _ in range(_MAX_ANCESTOR_HOPS):
        comm = _read_comm(pid)
        if comm is None:
            return None
        if not comm.startswith("VLLM::"):
            return _read_cmdline_raw(pid)
        next_pid = _read_ppid(pid)
        if next_pid is None:
            return None
        pid = next_pid
    return None


def find_orphaned_engine_cores() -> list[int]:
    """PIDs of every live ``VLLM::*``-named process whose nearest
    non-``VLLM::*`` ancestor's cmdline does not contain "vllm serve" — i.e.
    a child left behind by a server that died or was hard-killed without
    the shutdown cascading to it. Read-only: does not signal anything.
    comm.startswith("VLLM::") is the safe predicate (see module docstring,
    rule 5) — cmdline is read only for ancestors, to decide orphan-hood,
    never to select the candidate itself.
    """
    orphans: list[int] = []
    for pid in _iter_pids():
        comm = _read_comm(pid)
        if comm is None or not comm.startswith("VLLM::"):
            continue
        ppid = _read_ppid(pid)
        if ppid is None:
            orphans.append(pid)
            continue
        ancestor_cmdline = _root_ancestor_cmdline(ppid)
        if ancestor_cmdline is None or _VLLM_SERVE_NEEDLE not in ancestor_cmdline:
            orphans.append(pid)
    return orphans


# ---------------------------------------------------------------------------
# Launch / stop
# ---------------------------------------------------------------------------


def _server_state_path() -> Path:
    return paths.STATE_DIR / "server.json"


def _persist_handle(handle: ServerHandle) -> None:
    state_dir = paths.STATE_DIR
    state_dir.mkdir(parents=True, exist_ok=True)
    target = state_dir / "server.json"
    tmp = state_dir / "server.json.tmp"
    tmp.write_text(json.dumps(handle.to_dict(), indent=2) + "\n")
    os.replace(tmp, target)


def load_handle() -> ServerHandle | None:
    """Read back the last-persisted launch handle, for startup
    re-adoption. None if nothing was ever persisted or the file is
    unreadable/corrupt — callers fall back to a live port probe."""
    try:
        raw = json.loads((paths.STATE_DIR / "server.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return ServerHandle.from_dict(raw)
    except (KeyError, TypeError, ValueError):
        return None


def clear_handle() -> None:
    try:
        (paths.STATE_DIR / "server.json").unlink()
    except FileNotFoundError:
        pass


def launch(
    argv: Sequence[str],
    env: Mapping[str, str],
    cwd: str | os.PathLike[str],
    log_path: str | os.PathLike[str],
) -> ServerHandle:
    """Start `argv` in its own session/process group and persist a handle
    to it. `env` is merged onto (not a replacement for) the current
    process's environment — the same semantics as the ``env KEY=VAL cmd``
    prefix SPEC.md §1's delegation commands use, so PATH/HOME/etc. reach
    the launcher script unless this caller explicitly overrides them too.
    stdout+stderr are appended to `log_path` (rule 2: launched detached,
    with ``start_new_session=True``, so it outlives a Coldstart restart).
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_env = {**os.environ, **env}
    log_fh = open(log_path, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            list(argv),
            env=full_env,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_fh.close()  # the child holds its own duplicated fd past this point
    pgid = os.getpgid(proc.pid)
    handle = ServerHandle(
        pid=proc.pid,
        pgid=pgid,
        argv=list(argv),
        cwd=str(cwd),
        log_path=str(log_path),
        started_at=time.time(),
    )
    _persist_handle(handle)
    return handle


def _pgid_alive(pgid: int) -> bool:
    # `pgid` is always the group leader's own pid here (start_new_session
    # makes pid == pgid == sid at launch time) — so if THIS process is that
    # leader's actual parent (true for anything launch() started, for as
    # long as this same Coldstart process hasn't restarted since), a
    # reap-free liveness probe is a real bug, not just untidy: a process
    # that already exited sits as a zombie — still a live entry in the
    # process table — until something calls wait() on it. Left unreaped,
    # `os.killpg(pgid, 0)` below keeps reporting that zombie as "alive"
    # FOREVER, so stop() would spin through its full SIGTERM/SIGKILL
    # timeout on an already-dead process and report failure every time.
    # Reaping first (non-blocking) makes a natural exit visible
    # immediately. `ChildProcessError` means `pgid` is not (or is no
    # longer) our child — e.g. this handle was re-adopted from a previous
    # Coldstart process's state/server.json — so fall through to the
    # killpg probe, which is the correct check for a pid we didn't fork.
    try:
        # Reap to avoid a zombie leader, but do NOT conclude the group is dead
        # from the leader alone - fall through to the killpg group probe.
        os.waitpid(pgid, os.WNOHANG)
    except ChildProcessError:
        pass
    # NOTE: the reap above tells us the LEADER exited, which is not the same as
    # the group being gone. vLLM's worker children survive the leader and keep
    # holding the KV allocation; reporting "already_dead" there leaks the GPU.
    # killpg(pgid, 0) is the authoritative group-level probe, so always run it.
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists, just not ours to signal-probe further — treat as alive.
        return True


def stop(handle: ServerHandle, timeout_s: float = 60, escalate: bool = True) -> StopResult:
    """SIGTERM the whole process group, escalate to SIGKILL after
    `timeout_s` if `escalate` and it hasn't exited. Rule 3's guard: refuse
    outright rather than ever signal Coldstart's own process group.
    """
    pgid = handle.pgid
    own_pgid = os.getpgid(0)
    # pgid <= 0 is catastrophic, not merely invalid: killpg(0, SIG) signals the
    # CALLER's own process group, and killpg(-1, SIG) signals every process the
    # user can signal. A malformed state/server.json must never reach killpg.
    if pgid is None or pgid <= 0:
        return StopResult(
            ok=False,
            method="refused",
            waited_s=0.0,
            detail=(
                f"refusing to signal pgid {pgid!r}: non-positive pgid would "
                "signal Coldstart's own process group (0) or every reachable "
                "process (-1)"
            ),
        )
    if pgid == own_pgid or pgid == os.getpid():
        return StopResult(
            ok=False,
            method="refused",
            waited_s=0.0,
            detail=(
                f"refusing to signal pgid {pgid}: matches Coldstart's own "
                f"process group ({own_pgid}) or pid ({os.getpid()})"
            ),
        )

    if not _pgid_alive(pgid):
        clear_handle()
        return StopResult(ok=True, method="already_dead", waited_s=0.0, detail="process group already gone")

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        clear_handle()
        return StopResult(ok=True, method="already_dead", waited_s=0.0, detail="process group vanished before SIGTERM")
    except PermissionError as exc:
        return StopResult(ok=False, method="refused", waited_s=0.0, detail=f"SIGTERM denied: {exc}")

    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if not _pgid_alive(pgid):
            clear_handle()
            return StopResult(
                ok=True, method="sigterm", waited_s=time.monotonic() - start, detail="exited after SIGTERM"
            )
        time.sleep(0.2)
    waited = time.monotonic() - start

    if not escalate:
        return StopResult(
            ok=False, method="sigterm", waited_s=waited, detail=f"still alive after {timeout_s}s (escalate=False)"
        )

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        clear_handle()
        return StopResult(ok=True, method="sigterm", waited_s=waited, detail="exited just before SIGKILL")

    kill_start = time.monotonic()
    while time.monotonic() - kill_start < 10:
        if not _pgid_alive(pgid):
            clear_handle()
            return StopResult(
                ok=True,
                method="sigkill",
                waited_s=waited + (time.monotonic() - kill_start),
                detail="exited after SIGKILL",
            )
        time.sleep(0.2)
    return StopResult(
        ok=False, method="sigkill", waited_s=waited + 10, detail="still alive 10s after SIGKILL"
    )


# ---------------------------------------------------------------------------
# __main__ — python -m coldstart.procctl --scan
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m coldstart.procctl")
    parser.add_argument(
        "--scan", action="store_true", help="scan for live vLLM-tree processes and orphaned VLLM:: children"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)

    if not args.scan:
        parser.print_help()
        return 0

    procs = scan_vllm_processes()
    orphans = find_orphaned_engine_cores()
    orphan_set = set(orphans)

    if args.json:
        print(
            json.dumps(
                {
                    "processes": [dataclasses.asdict(p) for p in procs],
                    "orphaned_engine_cores": orphans,
                },
                indent=2,
            )
        )
        return 0

    print(f"{'PID':>8} {'PPID':>8} {'PGID':>8} {'ROLE':<13} {'VENV':<6} {'PORT':<6} COMM")
    for p in procs:
        flag = "  <-- ORPHAN" if p.pid in orphan_set else ""
        port_str = str(p.port) if p.port is not None else "-"
        venv_str = p.venv if p.venv is not None else "-"
        print(f"{p.pid:>8} {p.ppid:>8} {p.pgid:>8} {p.role:<13} {venv_str:<6} {port_str:<6} {p.comm}{flag}")
    print()
    if orphans:
        print(f"orphaned VLLM:: processes: {orphans}")
    else:
        print("orphaned VLLM:: processes: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
