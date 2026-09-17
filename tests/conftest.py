"""Suite-wide isolation.

Two things must never leak in either direction:

* **This box's configuration into the tests.** ``limits.py`` falls back to
  ``nvidia-smi`` for the card's size, so without a fixed value every capacity
  assertion would depend on which machine ran it — a test that passes on a
  clean checkout and fails on the maintainer's box, or worse, the reverse.
* **The tests' writes onto this box.** ``settings.state_dir`` defaults to the
  worktree's ``state/``, and ``desired.json`` lives in it. A test that wrote
  there would be telling the *real* servedeck what to start at its next
  reconcile. Every test gets a tmp state dir instead.
"""

from __future__ import annotations

import os

import pytest

from servedeck import capacity, limits, settings

#: The card this project was written on, so every capacity number in the suite
#: is checked against one known total rather than against whatever is plugged
#: in. Matches the ``[gpu] total_mib`` in the repo's models.toml.
GPU_TOTAL_MIB = 97887
OVERHEAD_GIB = 4.7
FRAG_MARGIN_MIB = 4096


@pytest.fixture(autouse=True)
def _fixed_environment(tmp_path_factory: pytest.TempPathFactory):
    """Pin the hardware limits and redirect every write to a tmp directory.

    Saved and restored BY HAND, not through the shared ``monkeypatch``
    fixture. Requesting ``monkeypatch`` from an autouse fixture sets it up
    before ``_no_permanent_module_stubs`` below, so its undo would run *after*
    that guard — and the guard would then see every test's own
    ``monkeypatch.setattr`` still in place and fail the innocent test that
    used it correctly.
    """
    state_dir = tmp_path_factory.mktemp("state")
    wanted = {
        "SERVEDECK_GPU_TOTAL_MIB": str(GPU_TOTAL_MIB),
        "SERVEDECK_OVERHEAD_GIB": str(OVERHEAD_GIB),
        "SERVEDECK_FRAG_MARGIN_MIB": str(FRAG_MARGIN_MIB),
        "SERVEDECK_STATE_DIR": str(state_dir),
        # No training marker exists under a fresh tmp dir, so the default
        # "is something else using the GPU" answer is a clean "no" — and a
        # test that is ABOUT markers sets the variable itself.
        "SERVEDECK_TRAINING_MARKERS": str(state_dir / "training_in_progress"),
    }
    previous = {name: os.environ.get(name) for name in wanted}
    os.environ.update(wanted)
    limits.reset()
    settings.reset()
    capacity.refresh_limits()
    try:
        yield state_dir
    finally:
        for name, old in previous.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
        limits.reset()
        settings.reset()
        capacity.refresh_limits()


# --------------------------------------------------------------------------
# Guard: a test that permanently rebinds a module entry point
# --------------------------------------------------------------------------
_GUARDED = (("servedeck.discovery", "discover_models"),)


@pytest.fixture(autouse=True)
def _no_permanent_module_stubs():
    """Fail the test that leaves a stub behind, not the innocent test after it.

    Plain assignment to a module attribute is not undone at teardown, so a test
    that stubs a discovery call disables it for every test that runs later in
    the same session -- and the tests that notice are the ones whose whole
    subject is that call. They fail in a full run and pass in isolation, which
    reads as flakiness rather than as the pollution it is.
    """
    import importlib

    before = {
        (mod, attr): getattr(importlib.import_module(mod), attr) for mod, attr in _GUARDED
    }
    yield
    leaked = [
        f"{mod}.{attr}"
        for (mod, attr), original in before.items()
        if getattr(importlib.import_module(mod), attr) is not original
    ]
    assert not leaked, (
        "this test replaced " + ", ".join(leaked) + " and did not restore it; "
        "use monkeypatch.setattr so the next test is not affected"
    )
