"""What ships: the unit file and the docs, checked as artefacts.

Neither is imported by anything, so nothing else in this suite would notice
them going wrong -- and both are things a new user copies verbatim before they
have any way to tell that a line is stale.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
UNIT = ROOT / "systemd" / "servedeck.service"
DOCS = sorted((ROOT / "docs").glob("*.md")) + [ROOT / "README.md"]

#: Directives systemd genuinely allows more than once in one section.
_REPEATABLE = {
    "Environment", "EnvironmentFile", "ExecStartPre", "ExecStartPost",
    "ExecStopPost", "ExecReload", "After", "Before", "Wants", "Requires",
    "WantedBy", "RequiredBy", "Also", "Documentation", "Conflicts",
    "BindsTo", "PartOf", "ReadWritePaths", "ReadOnlyPaths",
    "InaccessiblePaths", "SupplementaryGroups", "AppArmorProfile",
}


def _directives_by_section(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line
            continue
        if "=" in line:
            out.setdefault(section, []).append(line.split("=", 1)[0].strip())
    return out


def test_the_systemd_unit_declares_each_directive_once() -> None:
    """A directive written twice is one of them being silently discarded.

    systemd keeps the LAST value for a non-repeatable key and says nothing, so
    an operator editing the copy they can see can be editing the one that does
    not apply. This unit shipped with WorkingDirectory declared twice, with a
    comment in between that made the second look like the only one.
    """
    for section, names in _directives_by_section(UNIT.read_text()).items():
        dupes = sorted(
            {n for n in names if names.count(n) > 1 and n not in _REPEATABLE}
        )
        assert not dupes, f"{UNIT.name} {section} declares {dupes} more than once"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_shipped_docs_do_not_hardcode_one_persons_home_directory(doc: Path) -> None:
    """This repository is public and model-agnostic; ``/home/<someone>/...``
    in its documentation is both a privacy leak and an instruction nobody else
    can follow. Paths are written ``~/Projects/...``; a URL that merely
    contains the account name is not a path and is fine.
    """
    text = doc.read_text()
    hits = [
        m.group(0)
        for m in re.finditer(r"/home/[A-Za-z0-9._-]+/\S*", text)
        if not m.group(0).startswith("/home/user/")
    ]
    assert not hits, f"{doc.name} hardcodes {hits}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_shipped_docs_do_not_point_at_a_scratch_directory(doc: Path) -> None:
    """A path under /tmp is gone by the next boot. One reached the spec as the
    location of the design prototype it tells the reader to consult."""
    hits = re.findall(r"/tmp/\S+", doc.read_text())
    assert not hits, f"{doc.name} points at {hits}"
