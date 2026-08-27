"""SPEC.md §2 rule 1: no pattern-matching process-table lookup tool may
ever be invoked anywhere in the servedeck package — that family of tools
matches against a process's FULL COMMAND LINE, including the calling
shell's own, and has repeatedly killed shells during this project's
development (SETUP.md:401). This test greps the actual shipped package
source (`servedeck/`, not `tests/`) for those tool names and fails on any
hit, comments included — matching procctl.py's own module docstring, which
deliberately avoids spelling the names out for exactly this reason.

This file is the one place in the whole tree allowed to spell them out: a
search needs its own search terms, and grepping this file against itself
would be a tautology, so only `servedeck/` is walked below.

It also exercises SPEC.md §2 rules 2/3 end to end against a real (harmless,
non-vLLM) child process: `launch()` must give the child its own session
and process group, distinct from Coldstart's own, and `stop()` must refuse
outright rather than ever signal Coldstart's own process group.
"""

from __future__ import annotations

import os

from servedeck import paths, procctl

_PACKAGE_DIR = paths.SERVEDECK_PKG_DIR

# The literal names. Spelled out ONLY here (and nowhere under servedeck/)
# precisely because this is the file whose job is to make sure of that.
_FORBIDDEN_TOKENS = ("pgrep", "pkill", "killall")


def _iter_package_py_files() -> list[str]:
    found: list[str] = []
    for root, dirnames, filenames in os.walk(_PACKAGE_DIR):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name.endswith(".py"):
                found.append(os.path.join(root, name))
    return found


def test_no_pattern_matching_process_lookup_anywhere_in_package() -> None:
    hits: list[str] = []
    for path in _iter_package_py_files():
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        for token in _FORBIDDEN_TOKENS:
            if token in text:
                hits.append(f"{path}: contains {token!r}")
    assert not hits, "SPEC.md §2 rule 1 violation(s):\n" + "\n".join(hits)


def test_launch_gets_its_own_session_and_process_group(tmp_path, monkeypatch) -> None:
    # launch() unconditionally persists a handle to STATE_DIR/server.json.
    # Redirect that to an isolated tmp dir so this test can never touch (or
    # race) the real one, which may describe the live, must-not-be-disturbed
    # vLLM server.
    monkeypatch.setattr(paths, "STATE_DIR", tmp_path / "state")

    own_pgid = os.getpgid(0)
    log_path = tmp_path / "launch_test.log"

    # A short-lived, entirely harmless child (never vLLM, never a port).
    handle = procctl.launch(argv=["sleep", "2"], env={}, cwd=str(tmp_path), log_path=str(log_path))
    try:
        pgid = os.getpgid(handle.pid)
        assert pgid == handle.pid, "launched process must be its own group leader (start_new_session=True)"
        assert pgid != own_pgid, "launched process must NOT share Coldstart's own process group"
        assert handle.pgid == pgid
    finally:
        result = procctl.stop(handle, timeout_s=5, escalate=True)
        assert result.ok, f"cleanup stop() failed: {result}"


def test_stop_refuses_to_signal_own_process_group(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "STATE_DIR", tmp_path / "state")
    own_pgid = os.getpgid(0)
    # A handle that (falsely) claims to BE Coldstart's own process group —
    # rule 3's guard, exercised directly rather than only trusting that no
    # caller ever constructs one of these by accident.
    fake_handle = procctl.ServerHandle(
        pid=os.getpid(),
        pgid=own_pgid,
        argv=["not", "actually", "launched"],
        cwd=str(tmp_path),
        log_path=str(tmp_path / "unused.log"),
        started_at=0.0,
    )
    result = procctl.stop(fake_handle, timeout_s=1, escalate=False)
    assert result.ok is False
    assert result.method == "refused"
