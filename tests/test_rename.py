"""The coldstart -> servedeck rename: what must keep working across it.

A rename is a refactor only if nothing outside the tree notices. Three things
outside this tree do:

  * ``~/Projects/local_llm/.config`` and ``codex-qwen.sh``, which still spell
    the gateway URL ``COLDSTART_URL`` and are owned by another workstream;
  * ``~/Projects/coldstart/state/``, which holds every boot and every KV
    measurement this box has actually recorded, because the coldstart fork is
    what has been serving :8010;
  * the console entry point, which anything scripted invokes by name.

Each gets a test here. The grep gate at the bottom is what stops the rename
from quietly coming undone one edit at a time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from servedeck import config, history, legacy, paths, registry, shellconfig

_REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# 1. The deprecated .config / environment aliases still resolve
# --------------------------------------------------------------------------
def test_the_deprecated_config_key_still_resolves() -> None:
    """``.config`` on this box carries ``COLDSTART_URL=""`` and codex-qwen.sh's
    CONFIG_ALLOWED_KEYS still lists that spelling. If Servedeck stopped reading
    it, it would read a key that does not exist in that file and silently fall
    back to ``http://localhost:$PORT/v1`` -- pointing Codex straight at the
    model server and around the gateway, with nothing reporting an error."""
    assert shellconfig._base_url({"COLDSTART_URL": "http://127.0.0.1:8010"}) == \
        "http://127.0.0.1:8010/v1"
    assert "COLDSTART_URL" in shellconfig.ALLOWED_SET_KEYS
    assert "USE_COLDSTART" in shellconfig.ALLOWED_SET_KEYS


def test_the_new_name_wins_when_both_are_present() -> None:
    """During the changeover one file can carry both. The new name has to win,
    or writing SERVEDECK_URL would appear to do nothing."""
    assert shellconfig._base_url(
        {"COLDSTART_URL": "http://old:1", "SERVEDECK_URL": "http://new:2"}
    ) == "http://new:2/v1"


def test_an_empty_deprecated_key_falls_through_to_the_port() -> None:
    """``.config`` ships ``COLDSTART_URL=""``. Empty must mean "unset", not
    "the base URL is the empty string"."""
    assert shellconfig._base_url({"COLDSTART_URL": "", "PORT": "8001"}) == \
        "http://localhost:8001/v1"


def test_both_spellings_are_writable_keys() -> None:
    """set_key() rejects anything not on the allow list. The new name has to be
    on it (Servedeck's own spelling) and so does the old one (what
    codex-qwen.sh will accept back)."""
    for key in ("SERVEDECK_URL", "USE_SERVEDECK", "COLDSTART_URL", "USE_COLDSTART"):
        assert key in shellconfig.ALLOWED_SET_KEYS, key


def test_deprecated_env_vars_are_read_under_the_old_prefix(monkeypatch) -> None:
    """A shell that exported COLDSTART_PORT before the rename keeps working."""
    monkeypatch.delenv("SERVEDECK_PORT", raising=False)
    monkeypatch.setenv("COLDSTART_PORT", "8777")
    assert legacy.env("PORT") == "8777"


def test_the_new_env_prefix_wins_over_the_deprecated_one(monkeypatch) -> None:
    monkeypatch.setenv("SERVEDECK_PORT", "8010")
    monkeypatch.setenv("COLDSTART_PORT", "8777")
    assert legacy.env("PORT") == "8010"


def test_an_empty_new_env_var_does_not_shadow_the_deprecated_one(monkeypatch) -> None:
    """``SERVEDECK_PORT=`` is how a shell unsets by assignment. It must not
    blank out a COLDSTART_PORT that is actually set."""
    monkeypatch.setenv("SERVEDECK_PORT", "")
    monkeypatch.setenv("COLDSTART_PORT", "8777")
    assert legacy.env("PORT") == "8777"


def test_every_deprecated_key_names_its_replacement() -> None:
    """The map is the documentation. A deprecated key with no replacement is
    an alias nobody can migrate off."""
    assert legacy.DEPRECATED_KEYS
    for old, new in legacy.DEPRECATED_KEYS.items():
        assert old.startswith("COLDSTART") or "COLDSTART" in old, old
        assert "SERVEDECK" in new, new
        assert new in shellconfig.ALLOWED_SET_KEYS, new


# --------------------------------------------------------------------------
# 2. State written under the old layout is still read
# --------------------------------------------------------------------------
_OLD_BOOT = {
    "ts": "2026-08-27T21:24:13+00:00",
    "repo_id": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
    "backend": "flashnext",
    "outcome": "ready",
    "reached_ready": True,
    "cold": True,
    "total_s": 122.0,
}
_NEW_BOOT = dict(_OLD_BOOT, ts="2026-09-10T09:00:00+00:00", cold=False, total_s=61.0)

_OLD_OBS = {
    "repo_id": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
    "measured": {"kv_gib": 7.68, "kv_tokens": 280813},
}


@pytest.fixture
def split_state(tmp_path: Path, config_path, monkeypatch):
    """A current state directory and a pre-rename one, both empty."""
    current = tmp_path / "servedeck-state"
    legacy_dir = tmp_path / "coldstart-state"
    current.mkdir()
    legacy_dir.mkdir()
    monkeypatch.setenv("SERVEDECK_LEGACY_STATE_DIR", str(legacy_dir))
    config_path(f'state_dir = "{current}"\n')
    return current, legacy_dir


def test_boot_history_written_by_the_old_tree_is_still_read(split_state) -> None:
    """The rename must not orphan the run history. Every boot on this box was
    recorded by the coldstart fork; a Servedeck that ignores that directory
    reports "never booted" for every model and restarts its boot-ETA
    statistics from zero, with nothing reporting an error."""
    current, legacy_dir = split_state
    (legacy_dir / "history.jsonl").write_text(json.dumps(_OLD_BOOT) + "\n")
    assert history.load_all() == [_OLD_BOOT]


def test_old_and_new_history_are_merged_not_replaced(split_state) -> None:
    """"Read the old file only when the new one is missing" is the trap: the
    first boot recorded after the cutover would make every earlier one
    disappear. Both are read, oldest tree first."""
    current, legacy_dir = split_state
    (legacy_dir / "history.jsonl").write_text(json.dumps(_OLD_BOOT) + "\n")
    history.append(dict(_NEW_BOOT))
    loaded = history.load_all()
    assert loaded == [_OLD_BOOT, _NEW_BOOT], loaded


def test_copying_the_old_history_across_does_not_double_count(split_state) -> None:
    """docs/MIGRATION.md tells the operator to copy the old state files over.
    Doing that AND reading the old directory must not count every historic boot
    twice -- doubled boot counts would silently halve every failure rate the
    dashboard reports."""
    current, legacy_dir = split_state
    line = json.dumps(_OLD_BOOT) + "\n"
    (legacy_dir / "history.jsonl").write_text(line)
    (current / "history.jsonl").write_text(line)
    assert history.load_all() == [_OLD_BOOT]


def test_a_record_that_differs_only_in_key_order_is_one_record(split_state) -> None:
    """The two trees serialize with the same writer today, but a de-duplication
    that depends on byte order would fail the moment either side reorders a
    field -- and would fail by silently doubling, not by raising."""
    current, legacy_dir = split_state
    reordered = {k: _OLD_BOOT[k] for k in reversed(list(_OLD_BOOT))}
    (legacy_dir / "history.jsonl").write_text(json.dumps(_OLD_BOOT) + "\n")
    (current / "history.jsonl").write_text(json.dumps(reordered) + "\n")
    assert len(history.load_all()) == 1


def test_kv_measurements_from_the_old_tree_are_still_read(split_state) -> None:
    """Same directory, the other file. Losing it makes every model that HAS
    been booted and measured report "estimated" again -- the one thing this
    store exists to prevent."""
    current, legacy_dir = split_state
    (legacy_dir / "measurements.json").write_text(json.dumps([_OLD_OBS]))
    assert registry.load_observations() == [_OLD_OBS]


def test_writes_never_land_in_the_pre_rename_tree(split_state) -> None:
    """The coldstart dashboard may still be RUNNING out of that directory.
    Servedeck reads it and must never write into it -- a second writer would be
    corrupting a live process's state."""
    current, legacy_dir = split_state
    (legacy_dir / "history.jsonl").write_text(json.dumps(_OLD_BOOT) + "\n")
    before = (legacy_dir / "history.jsonl").read_text()
    history.append(dict(_NEW_BOOT))
    registry.append_observation({"repo_id": "x", "measured": {}})
    assert (legacy_dir / "history.jsonl").read_text() == before
    assert not (legacy_dir / "measurements.json").exists()
    assert (current / "history.jsonl").is_file()
    assert (current / "measurements.json").is_file()


def test_the_compatibility_read_can_be_turned_off(split_state, monkeypatch) -> None:
    """A machine that never ran the coldstart fork should be able to say so."""
    current, legacy_dir = split_state
    (legacy_dir / "history.jsonl").write_text(json.dumps(_OLD_BOOT) + "\n")
    monkeypatch.setenv("SERVEDECK_LEGACY_STATE_DIR", "")
    assert legacy.legacy_state_dir() is None
    assert history.load_all() == []


def test_the_history_file_follows_the_configured_state_dir(split_state) -> None:
    """default_history_path() read paths.STATE_DIR -- the directory next to the
    package -- while registry.default_measurements_path() read config.state_dir.
    Set state_dir at all (a systemd StateDirectory= does) and boots were
    appended to one directory while capacity was read from another. Nothing
    failed; the ETA statistics just stayed empty forever."""
    current, _legacy_dir = split_state
    assert history.default_history_path().parent == current
    assert registry.default_measurements_path().parent == current
    assert history.default_history_path().parent == \
        registry.default_measurements_path().parent


def test_an_explicit_path_reads_exactly_that_file(split_state, tmp_path) -> None:
    """The merge is for the DEFAULT store only. A caller that names a file --
    every test that builds a fixture, and the CLI -- must get that file and
    nothing else off the machine it happens to be running on."""
    _current, legacy_dir = split_state
    (legacy_dir / "history.jsonl").write_text(json.dumps(_OLD_BOOT) + "\n")
    named = tmp_path / "elsewhere.jsonl"
    named.write_text(json.dumps(_NEW_BOOT) + "\n")
    assert history.load_all(named) == [_NEW_BOOT]


def test_the_pre_rename_state_dir_is_not_inside_this_tree() -> None:
    """It names the fork's tree, not a subdirectory of this one. If these ever
    became the same path the merge would read a file into itself."""
    assert paths.LEGACY_COLDSTART_STATE_DIR != paths.STATE_DIR
    assert paths.PROJECT_ROOT not in paths.LEGACY_COLDSTART_STATE_DIR.parents


# --------------------------------------------------------------------------
# 3. The entry point starts under the new name
# --------------------------------------------------------------------------
def test_the_console_entry_point_is_declared_as_servedeck() -> None:
    text = (_REPO / "pyproject.toml").read_text()
    assert 'servedeck = "servedeck.__main__:main"' in text
    assert "coldstart" not in text.lower()


def test_the_entry_point_runs_under_the_new_name() -> None:
    """Actually invoke it. --help exits before uvicorn.run(), so this starts the
    real entry point without binding a port or touching the GPU."""
    out = subprocess.run(
        [sys.executable, "-m", "servedeck", "--help"],
        cwd=str(_REPO), capture_output=True, text=True, timeout=60,
        env={**os.environ, "SERVEDECK_CONFIG": os.environ.get("SERVEDECK_CONFIG", "")},
    )
    assert out.returncode == 0, out.stderr
    assert "usage: servedeck" in out.stdout, out.stdout
    assert "coldstart" not in out.stdout.lower()


def test_the_entry_point_serves_the_servedeck_asgi_app(monkeypatch) -> None:
    """The name in the ExecStart of systemd/servedeck.service and in run.sh.
    A rename that left this string behind would fail only at boot."""
    from servedeck import __main__ as entry

    seen: dict[str, object] = {}

    def _fake_run(app: str, **kw: object) -> None:
        seen["app"] = app
        seen.update(kw)

    monkeypatch.setitem(sys.modules, "uvicorn", type("M", (), {"run": staticmethod(_fake_run)}))
    assert entry.main(["--port", "18099"]) == 0
    assert seen["app"] == "servedeck.app:app"
    assert seen["port"] == 18099


def test_the_deployment_files_name_servedeck_only() -> None:
    for name in ("run.sh", "stop.sh", "setup.sh", "systemd/servedeck.service"):
        text = (_REPO / name).read_text()
        assert "coldstart" not in text.lower(), f"{name} still names coldstart"
        assert "servedeck" in text.lower(), name


# --------------------------------------------------------------------------
# 4. The grep gate
# --------------------------------------------------------------------------
#
# Every file below is allowed to say "coldstart", for the reason given. A file
# that is not on this list may not, so a future edit cannot reintroduce the old
# name without either failing here or being justified here.
_ALLOWED = {
    "docs/MIGRATION.md": "the cutover document; coldstart is its subject",
    "docs/SPEC.md": "provenance header, plus the shell contract that is still "
                    "spelled COLDSTART_* outside this repo",
    "MERGE-NOTES.md": "the record of folding the fork back in",
    "servedeck/legacy.py": "the deprecation shims themselves",
    "servedeck/paths.py": "LEGACY_COLDSTART_STATE_DIR, the directory read for "
                          "compatibility",
    "servedeck/shellconfig.py": "names the external .config contract in a comment",
    "tests/test_rename.py": "this file",
    "tests/test_launch_contract.py": "asserts the deprecated alias still resolves",
    "tests/test_kvcalc.py": "reads the fork's measurements as a real-data fixture",
    "tests/conftest.py": "turns the compatibility read off for the suite",
    "servedeck/config.py": "docstring names the deprecated COLDSTART_* env prefix",
    "servedeck/history.py": "comment explaining the merge of the pre-rename store",
    "servedeck/registry.py": "comment explaining the merge of the pre-rename store",
    "servedeck.toml.example": "documents SERVEDECK_LEGACY_STATE_DIR and its default",
}


def _tracked_files() -> list[str]:
    """Tracked files PLUS untracked, non-ignored ones.

    ``git ls-files`` alone would let a brand-new module reintroduce the old
    name and pass this gate until the moment it was committed -- i.e. it would
    pass in exactly the run where it could still be fixed cheaply.
    """
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=str(_REPO), capture_output=True, text=True, check=True,
    )
    return sorted({line for line in out.stdout.splitlines() if line})


def test_no_unlisted_file_still_says_coldstart() -> None:
    offenders = []
    for rel in _tracked_files():
        if rel in _ALLOWED:
            continue
        path = _REPO / rel
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "coldstart" in text.lower():
            offenders.append(rel)
    assert not offenders, (
        "these files still name coldstart and are not on the allow list in "
        f"tests/test_rename.py: {offenders}"
    )


def test_the_allow_list_has_no_dead_entries() -> None:
    """An allow-list entry for a file that no longer says coldstart is an
    exemption nobody is watching -- it would silently permit the name coming
    back into that file later."""
    tracked = set(_tracked_files())
    stale = []
    for rel in _ALLOWED:
        if rel not in tracked:
            stale.append(f"{rel} (not tracked)")
            continue
        if "coldstart" not in (_REPO / rel).read_text(encoding="utf-8", errors="ignore").lower():
            stale.append(f"{rel} (no longer says coldstart)")
    assert not stale, f"stale allow-list entries: {stale}"


def test_no_python_identifier_is_named_coldstart() -> None:
    """Comments and doc prose may name the old project. A module, class,
    function, or variable may not -- except the deprecation shims, whose whole
    job is to spell it."""
    import ast

    offenders: list[str] = []
    for rel in _tracked_files():
        if not rel.endswith(".py") or rel in ("servedeck/legacy.py", "tests/test_rename.py"):
            continue
        tree = ast.parse((_REPO / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
            elif isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Attribute):
                name = node.attr
            if not name or "coldstart" not in name.lower():
                continue
            # LEGACY_/legacy_ is the marker that says "this spells the old
            # name on purpose". Anything else is a rename that was missed.
            if name.lower().startswith("legacy"):
                continue
            offenders.append(f"{rel}:{node.lineno} {name}")
    assert not offenders, offenders


def test_no_string_literal_in_the_package_says_coldstart() -> None:
    """A UI string, a config key, a path, an event type. This is the class the
    grep gate above cannot see through prose: a comment that mentions coldstart
    is documentation, a STRING that does is behaviour. legacy.py and paths.py
    are the two files whose job is to hold those strings."""
    import ast

    exempt = {"servedeck/legacy.py", "servedeck/paths.py"}
    offenders: list[str] = []
    for rel in _tracked_files():
        if not rel.startswith("servedeck/") or not rel.endswith(".py") or rel in exempt:
            continue
        tree = ast.parse((_REPO / rel).read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in docstrings:
                continue
            if "coldstart" in node.value.lower():
                offenders.append(f"{rel}:{node.lineno} {node.value[:60]!r}")
    assert not offenders, offenders
