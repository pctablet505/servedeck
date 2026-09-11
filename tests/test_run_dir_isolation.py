"""The suite must never write the launcher's real run/desired_state file.

Supervisor._write_flat_desired_state_file() writes
paths.RUN_DIR / "desired_state", and paths.RUN_DIR is the REAL
~/Projects/local_llm/run: a directory owned by the local_llm launcher, not by
servedeck. Before tests/conftest.py redirected it, every suite run rewrote
that file (observed: "running" at the end of every run on 2026-09-11), so a
test run silently changed a file another tool reads to decide what should be
serving. These tests pin the redirection, and pin that the writer honours it.
"""

from __future__ import annotations

from pathlib import Path

from servedeck import paths, supervisor

# Captured at import, before any fixture runs: the path the module was
# configured with. conftest must not change this constant, only paths.RUN_DIR.
_REAL_RUN_DIR = paths.LOCAL_LLM / "run"


def test_run_dir_is_redirected_away_from_the_real_launcher_directory() -> None:
    assert Path(paths.RUN_DIR).resolve() != _REAL_RUN_DIR.resolve(), (
        f"paths.RUN_DIR still points at the real launcher directory {_REAL_RUN_DIR}; "
        "every test that writes desired_state would overwrite the launcher's file"
    )


def test_the_desired_state_writer_lands_in_the_redirected_directory(tmp_path: Path) -> None:
    real = _REAL_RUN_DIR / "desired_state"
    before = (real.exists(), real.stat().st_mtime_ns if real.exists() else None,
              real.read_text() if real.exists() else None)

    s = supervisor.Supervisor(state_dir=tmp_path, clock=lambda: 1000.0,
                              history_path=tmp_path / "history.jsonl")
    s.desired.desired_state = "RUNNING"
    s._write_flat_desired_state_file()

    written = Path(paths.RUN_DIR) / "desired_state"
    assert written.read_text() == "running\n"
    after = (real.exists(), real.stat().st_mtime_ns if real.exists() else None,
             real.read_text() if real.exists() else None)
    assert after == before, f"the real {real} was modified by the test suite"
