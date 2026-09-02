"""Servedeck GPU introspection — SPEC.md §2's nvidia-smi wrappers and the
Xid classification described there (reusing bin/qwen-server-record-death.sh
lines 54-69's case statement, read verbatim from that file).

Every function here shells out to `nvidia-smi` or `journalctl` rather than
touching the driver directly, and every one tolerates the tool being
slow/absent/erroring by returning a clearly-empty/None result rather than
raising into a caller that is mid-render of a status page. `gpu_alive()` in
particular backs SPEC.md §3's GPU_UNRESPONSIVE finding and §6's rule
"nvidia-smi -L failing => never restart" — it must never raise, and a
caller may treat any exception escaping this module as a bug in the module,
not a signal about the GPU.

This module does not decide restart policy (that is supervisor.py's job,
per SPEC.md §6) — it only classifies. What supervisor.py does with
`XidEvent.restartable` is out of scope here.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

_NVIDIA_SMI_TIMEOUT_S = 5.0
_JOURNALCTL_TIMEOUT_S = 10.0

_GPU_QUERY_FIELDS = (
    "index,name,memory.total,memory.used,memory.free,"
    "utilization.gpu,temperature.gpu,power.draw"
)
_COMPUTE_APPS_FIELDS = "pid,process_name,used_memory"


@dataclass(frozen=True)
class GpuSummary:
    """One row of `nvidia-smi --query-gpu=...`. SPEC.md's box has exactly
    one GPU (multi-GPU is an explicit v1 non-goal, SPEC.md §10), so
    gpu_summary() below returns the first (only) row rather than a list."""

    index: int
    name: str
    total_mib: int
    used_mib: int
    free_mib: int
    util_percent: int
    temperature_c: int
    power_draw_w: float


@dataclass(frozen=True)
class ComputeApp:
    """One row of `nvidia-smi --query-compute-apps=...` — a process
    currently holding GPU memory, as the driver itself attributes it."""

    pid: int
    process_name: str
    used_mib: int


@dataclass(frozen=True)
class XidEvent:
    """One `NVRM: Xid` line pulled from the kernel journal, classified per
    bin/qwen-server-record-death.sh:54-69."""

    raw_line: str
    code: int | None
    classification: str  # "mmu_fault" | "off_bus" | "reboot_required" | "unrecognized"
    restartable: bool
    note: str


def gpu_alive() -> bool:
    """True iff `nvidia-smi -L` runs and reports at least one GPU line.
    False on ANY failure — missing binary, timeout, nonzero exit, empty
    stdout — never raises. This is the exact predicate SPEC.md §6 names:
    "`nvidia-smi -L` failing => never restart"."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "GPU" in result.stdout


def gpu_summary() -> GpuSummary | None:
    """The single GPU's current headline numbers, or None if nvidia-smi is
    unavailable / its output doesn't parse as expected. Never raises."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={_GPU_QUERY_FIELDS}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        return None
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) != 8:
        return None
    try:
        return GpuSummary(
            index=int(parts[0]),
            name=parts[1],
            total_mib=int(parts[2]),
            used_mib=int(parts[3]),
            free_mib=int(parts[4]),
            util_percent=int(parts[5]),
            temperature_c=int(parts[6]),
            power_draw_w=float(parts[7]),
        )
    except ValueError:
        return None


def compute_apps() -> list[ComputeApp]:
    """Every process nvidia-smi currently attributes GPU memory to. Empty
    list on any failure (missing binary, timeout, nonzero exit) — never
    raises. Used by SPEC.md §3's NOT_ENOUGH_FREE_VRAM finding ("discount
    our own processes") and by procctl's rule-4 cross-checks upstream of
    this module.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-compute-apps={_COMPUTE_APPS_FIELDS}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    apps: list[ComputeApp] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            apps.append(ComputeApp(pid=int(parts[0]), process_name=parts[1], used_mib=int(parts[2])))
        except ValueError:
            continue
    return apps


# ---------------------------------------------------------------------------
# Xid classification — bin/qwen-server-record-death.sh:54-69, read verbatim:
#
#   case "$code" in
#       13|31)  "GPU MMU fault (illegal/misaligned address)... the known,
#               documented, MTP-correlated crash"
#       79)     "GPU has fallen off the bus. Hardware/driver-level, NOT the
#               documented MMU fault. No process restart fixes this."
#       154)    "driver GPU-recovery action asserted -- often means 'Node
#               Reboot Required'. ... no process restart fixes this either."
#       *)      "not one of the codes this script recognizes. Do not assume
#               it's the known MMU-fault pattern -- look it up."
#
# `restartable` here mirrors that script's judgement, not a general claim:
# 13/31 is the one pattern that script and SPEC.md §6 both call restartable;
# everything else — including an unrecognized code — is not.
# ---------------------------------------------------------------------------

# Matches the ": NNN," that immediately follows "NVRM: Xid (PCI:...)" in a
# real line, e.g. "NVRM: Xid (PCI:0000:01:00): 13, pid=237949, name=...".
# Same regex shape as record-death.sh's own
# `grep -oE ': [0-9]+,' | grep -oE '[0-9]+'` extraction.
_XID_CODE_RE = re.compile(r":\s*(\d+),")


def _classify_xid_code(code: int) -> tuple[str, bool, str]:
    if code in (13, 31):
        return (
            "mmu_fault",
            True,
            f"Xid {code}: GPU MMU fault (illegal/misaligned address). "
            "The known, documented, MTP-correlated crash.",
        )
    if code == 79:
        return (
            "off_bus",
            False,
            f"Xid {code}: GPU has fallen off the bus. Hardware/driver-level, "
            "NOT the documented MMU fault. No process restart fixes this.",
        )
    if code == 154:
        return (
            "reboot_required",
            False,
            f"Xid {code}: driver GPU-recovery action asserted — often means "
            "'Node Reboot Required'. No process restart fixes this either.",
        )
    return (
        "unrecognized",
        False,
        f"Xid {code}: not one of the codes qwen-server-record-death.sh "
        "recognizes. Do not assume it's the known MMU-fault pattern — look it up.",
    )


def xid_events(since: str) -> list[XidEvent]:
    """NVRM Xid lines out of the kernel journal since `since`, classified
    per bin/qwen-server-record-death.sh:54-69.

    `since` is passed straight through to `journalctl -k --no-pager
    --since <since>` — anything that flag accepts works ("30 min ago",
    "2026-08-27 21:00:00", "today", ...). Equivalent to that script's
    ``journalctl -k --since "..." | grep "NVRM: Xid"`` pipeline, just done
    without shelling out to `grep` (the filter is a plain Python substring
    check on each line).

    Empty list on any failure (missing journalctl, timeout, no persistent
    journal for the window asked) — never raises. An empty result means
    "no evidence found in the journal for this window", not "definitely
    nothing happened": kernel Xid history is not guaranteed durable on
    this box across boots.
    """
    try:
        result = subprocess.run(
            ["journalctl", "-k", "--no-pager", "--since", since],
            capture_output=True,
            text=True,
            timeout=_JOURNALCTL_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []

    events: list[XidEvent] = []
    for line in result.stdout.splitlines():
        if "NVRM: Xid" not in line:
            continue
        match = _XID_CODE_RE.search(line)
        if match is None:
            events.append(
                XidEvent(
                    raw_line=line,
                    code=None,
                    classification="unrecognized",
                    restartable=False,
                    note="Xid line matched but no numeric code could be parsed from it.",
                )
            )
            continue
        code = int(match.group(1))
        classification, restartable, note = _classify_xid_code(code)
        events.append(
            XidEvent(
                raw_line=line,
                code=code,
                classification=classification,
                restartable=restartable,
                note=note,
            )
        )
    return events
