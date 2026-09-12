"""Package-wide rules that no single module's tests can enforce.

Two of these are inherited from suites P4 deleted along with the modules they
covered. The modules are gone; the rules are not, and a rule whose only
enforcement disappeared with its subject is a rule that quietly stops holding.

* ``tests/test_procctl_no_pattern_kill.py`` banned pattern-based process
  lookup, but did it by grepping every file for the *string* "pgrep". P3's
  ``units.py`` and ``control.py`` both mention it in comments explaining why it
  must never be used — so the old test failed on the documentation of its own
  rule. Here the ban is on an INVOCATION, found in the AST, where a comment
  cannot reach.
* ``tests/test_rename.py`` swept for the pre-rename "coldstart" spelling. That
  sweep is worth keeping (a stale name in a generated config is a real defect)
  with an allow-list for the places that are *about* the old tree.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "servedeck"
TESTS = ROOT / "tests"

PY_FILES = sorted(PACKAGE.glob("*.py")) + sorted(TESTS.glob("*.py"))

#: This file DEFINES the bans below, so it is the one file that necessarily
#: spells the banned words out — in the pattern that detects them and in the
#: fixture source that proves the detector fires. Excluding it is not a hole:
#: ``test_the_ban_would_catch_a_real_invocation`` runs the detector against
#: both shapes, so the detector cannot be weakened without a test failing.
SWEPT = [p for p in PY_FILES if p.name != "test_hygiene.py"]


# --------------------------------------------------------------------------
# 1. No pattern-based process lookup, ever
# --------------------------------------------------------------------------

_BANNED = re.compile(r"\b(pgrep|pkill)\b")


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """``id()`` of every string constant that is a docstring.

    Docstrings are in the AST (comments are not), and this rule is about what
    the program DOES, not about what it explains. ``control.py``'s module
    docstring says "there is no `pgrep -f` to match its own shell" — which is
    the rule being stated, and the old test failed on it.
    """
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            out.add(id(first.value))
    return out


def _executable_strings(path: Path) -> list[tuple[int, str]]:
    """Every string literal in the file that is not a docstring."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    skip = _docstring_nodes(tree)
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in skip
    ]


@pytest.mark.parametrize("path", SWEPT, ids=lambda p: p.name)
def test_nothing_invokes_pgrep_or_pkill(path: Path) -> None:
    """``pgrep -f <pattern>`` matches the shell that ran it (exit 144 on this
    box), and ``pkill -f`` then signals a process nobody identified.

    v2 has no need for either: discovery is ``systemctl --user list-units``,
    and everything that is stopped is stopped by UNIT NAME, validated against
    ``model-``/``sd-test-`` before a single process is spawned. This test bans
    the *call*, so the comments that explain the ban do not trip it.
    """
    offenders = [
        f"{path.name}:{lineno}: {value!r}"
        for lineno, value in _executable_strings(path)
        if _BANNED.search(value)
    ]
    assert not offenders, (
        "pattern-based process lookup is banned; stop units by name instead:\n"
        + "\n".join(offenders)
    )


def test_the_ban_would_catch_a_real_invocation(tmp_path: Path) -> None:
    """Over-correction guard: an AST-based check that skipped docstrings could
    just as easily skip everything. Prove it still fires on the real shapes —
    and that it still passes the comment/docstring form it exists to tolerate.
    """
    caught = tmp_path / "bad.py"
    caught.write_text(
        'import subprocess\n'
        'subprocess.run(["pgrep", "-f", "vllm"])\n'
        'subprocess.run("pkill -f vllm", shell=True)\n'
    )
    hits = [v for _ln, v in _executable_strings(caught) if _BANNED.search(v)]
    assert len(hits) == 2, hits

    tolerated = tmp_path / "good.py"
    tolerated.write_text(
        '"""Never use pgrep -f: it matches its own shell."""\n'
        '# and never pkill -f either\n'
        'def f():\n'
        '    """pkill is banned here too."""\n'
        '    return 1\n'
    )
    assert [v for _ln, v in _executable_strings(tolerated) if _BANNED.search(v)] == []


# --------------------------------------------------------------------------
# 2. The pre-rename spelling
# --------------------------------------------------------------------------

#: Files allowed to say "coldstart", each because it is ABOUT the pre-rename
#: tree rather than carrying a stale name. Every entry is a place a reader
#: would be confused by the word's absence, not by its presence.
_COLDSTART_ALLOWED = {
    # Both read the real measurement store, which was accumulated by the
    # coldstart fork that served :8010 until 09-12.
    "tests/test_kvcalc.py",
    "servedeck/discovery.py",
    # P3 cites the old unit name when explaining what transient units replaced.
    "tests/test_units.py",
    # This file, which names the rule.
    "tests/test_hygiene.py",
}

_SWEEP_SUFFIXES = {".py", ".md", ".toml", ".js", ".html", ".css", ".service", ".sh"}


def _tracked_files() -> list[Path]:
    out: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in _SWEEP_SUFFIXES:
            continue
        rel = path.relative_to(ROOT)
        if rel.parts[0] in {".git", ".venv", "builds", "dist", "state", ".pytest_cache"}:
            continue
        out.append(path)
    return sorted(out)


def test_no_unlisted_file_still_says_coldstart() -> None:
    """The project was forked as ``coldstart`` and renamed.

    The word surviving in a generated client config, a unit file or a docs
    path is a real defect — it points at a tree that is not this one. Where it
    is deliberate the file is on the allow-list above, which is short on
    purpose: a growing allow-list is the signal that the rename is not done.
    """
    offenders = sorted(
        str(p.relative_to(ROOT))
        for p in _tracked_files()
        if "coldstart" in p.read_text(encoding="utf-8", errors="replace").lower()
        and str(p.relative_to(ROOT)) not in _COLDSTART_ALLOWED
    )
    assert not offenders, f"these still spell the pre-rename name: {offenders}"


# --------------------------------------------------------------------------
# 3. Packaging facts the rename test used to hold
# --------------------------------------------------------------------------


def test_the_console_script_is_the_cli_not_the_server() -> None:
    """``servedeck`` is the control CLI; ``python -m servedeck`` is the server.

    v1 pointed the console script at ``__main__:main``, so typing ``servedeck``
    started a web server — which is why the dashboard kept getting started from
    terminals (REDESIGN §4 R4) and two of them ended up on one state dir.
    """
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert data["project"]["scripts"]["servedeck"] == "servedeck.cli:main"


def test_the_deleted_v1_modules_are_really_gone() -> None:
    """P4 deletes the surface, not just its callers. A module left on disk with
    no importer is still 1,700 lines a reader has to rule out."""
    gone = [
        "supervisor", "procctl", "phases", "updetect", "preflight", "history",
        "logtail", "shellconfig", "legacy", "smoke", "config", "paths", "events",
        "registry",
    ]
    present = [name for name in gone if (PACKAGE / f"{name}.py").exists()]
    assert not present, present


_DELETED = frozenset({
    "supervisor", "procctl", "phases", "updetect", "preflight", "history",
    "logtail", "shellconfig", "legacy", "smoke", "config", "paths", "events",
    "registry",
})


def _imported_servedeck_modules(path: Path) -> set[str]:
    """Submodules of ``servedeck`` this file imports, by their REAL name.

    Read from the AST, not by regex: ``from servedeck import discovery as
    registry`` imports ``discovery``, and a regex over the line sees the word
    ``registry`` and calls it a resurrection of the module that was renamed.
    The alias is the caller's business; the module name is the fact.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if base == "servedeck" or (node.level and not base):
                found.update(alias.name for alias in node.names)
            elif base.startswith("servedeck."):
                found.add(base.split(".", 1)[1].split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("servedeck."):
                    found.add(alias.name.split(".", 1)[1].split(".")[0])
    return found


def test_nothing_imports_a_deleted_module() -> None:
    """An import of a module that no longer exists fails at import time, which
    is loud — but only if something imports THAT file. This sweeps all of them
    at once, including the ones only a rare code path reaches."""
    offenders = sorted(
        f"{path.name}: {sorted(_imported_servedeck_modules(path) & _DELETED)}"
        for path in PY_FILES
        if _imported_servedeck_modules(path) & _DELETED
    )
    assert not offenders, offenders


def test_the_import_sweep_reads_the_module_not_the_alias(tmp_path: Path) -> None:
    """Over-correction guard for the fix above: the sweep must still catch a
    real import of a deleted module, and must not be fooled by an alias in
    either direction."""
    aliased = tmp_path / "aliased.py"
    aliased.write_text("from servedeck import discovery as registry\n")
    assert _imported_servedeck_modules(aliased) == {"discovery"}

    real = tmp_path / "real.py"
    real.write_text("from servedeck import supervisor as sup\nimport servedeck.paths\n")
    assert _imported_servedeck_modules(real) & _DELETED == {"supervisor", "paths"}
