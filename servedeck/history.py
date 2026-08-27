"""Coldstart boot/death history — SPEC.md §6 ("append state/history.jsonl
[...] on EVERY exit") and §8's WAIT-TIME UX ("ETA from history: median and
p90 of total_s and per-phase over prior runs matching (repo_id, backend,
cold)").

``state/history.jsonl`` is an append-only JSON-Lines log, one record per
boot attempt (successful or not) or per exit of an already-serving run.
supervisor.py is the only writer (see its ``_record_exit`` /
``_record_boot_failure``); this module owns the on-disk shape, the
append/query primitives, and the ETA statistics derived from it. It does no
process/GPU/log-tailing I/O of its own — purely a small time-series store
plus arithmetic over it.

Record shape (one JSON object per line — every field optional except
``ts``/``repo_id``/``backend``/``outcome``, so a partial record from a
still-evolving caller never breaks ``load_all()``)::

    {
      "ts": "2026-08-27T21:24:13+00:00",   # when this record was written
      "repo_id": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
      "backend": "flashnext",
      "started_at": "2026-08-27T21:22:11+00:00",  # launch() wall-clock time
      "reached_ready": true,
      "cold": false,                        # SPEC §8: no prior *successful*
                                              # boot existed for (repo_id,
                                              # backend) before this run
      "total_s": 122.0,                     # launch -> READY, only when
                                              # reached_ready
      "phases": {"init": 0.4, "loading_weights": 1.1, ...},  # SPEC §5's
                                              # phase names -> wall-clock
                                              # seconds from launch to FIRST
                                              # entry into that phase
      "outcome": "ready" | "crashed" | "failed_boot" | "stopped_by_user"
               | "suspended" | "adopted",
      "failure_code": "RUNTIME_OOM" | null,  # phases.classify() code, if any
      "restart_scheduled_s": 15 | null,      # backoff delay chosen, if any
      "attempt_in_window": 1 | null,         # 1-based position in the
                                              # current crash-loop window
      "exit": {"service_result": "...", "exit_code": "...", "exit_status": "..."}
    }

Nothing here is invented: every field above is either directly observable
by supervisor.py (timestamps, which phase lines arrived) or a value already
defined elsewhere in this codebase (phases.classify() codes, SPEC §6's
BACKOFF schedule, SPEC §5's phase names).
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from servedeck import paths

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

# A single process-wide lock: uvicorn runs Coldstart as one worker (SPEC.md
# §1's process model names no --workers flag, and the systemd unit doesn't
# either), so this only needs to serialize concurrent async tasks within
# that one process, not cross-process — but it costs nothing to be safe
# about interleaved writes to a shared file handle either way.
_WRITE_LOCK = threading.Lock()


def default_history_path() -> Path:
    return paths.STATE_DIR / "history.jsonl"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append(record: dict[str, Any], path: Path | str | None = None) -> None:
    """Append one record as a single JSON line. Never overwrites prior
    entries. Adds ``ts`` if the caller didn't already set one."""
    p = Path(path) if path is not None else default_history_path()
    record = dict(record)
    record.setdefault("ts", now_iso())
    line = json.dumps(record, sort_keys=False) + "\n"
    with _WRITE_LOCK:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())


def load_all(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Read every record. Missing file -> []. A malformed trailing line
    (e.g. a write that was interrupted mid-flush) is skipped, not fatal —
    every earlier line still parses and is returned; never raises."""
    p = Path(path) if path is not None else default_history_path()
    if not p.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def query(
    *,
    repo_id: str | None = None,
    backend: str | None = None,
    cold: bool | None = None,
    outcome: str | None = None,
    reached_ready: bool | None = None,
    limit: int | None = None,
    path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Filter the full history by any combination of fields, oldest first.
    ``limit`` (if given) keeps the most recent ``limit`` matches (SPEC.md §8:
    ``GET /api/history?limit``)."""
    records = load_all(path)
    if repo_id is not None:
        records = [r for r in records if r.get("repo_id") == repo_id]
    if backend is not None:
        records = [r for r in records if r.get("backend") == backend]
    if cold is not None:
        records = [r for r in records if r.get("cold") == cold]
    if outcome is not None:
        records = [r for r in records if r.get("outcome") == outcome]
    if reached_ready is not None:
        records = [r for r in records if r.get("reached_ready") == reached_ready]
    if limit is not None and limit >= 0:
        # records[-0:] is the WHOLE list, not an empty one. limit=0 must mean
        # "no records", which is what every caller passing a computed limit
        # expects.
        records = records[-limit:] if limit > 0 else []
    return records


def has_prior_success(
    repo_id: str,
    backend: str,
    *,
    before_ts: str | None = None,
    path: Path | str | None = None,
) -> bool:
    """SPEC.md §4/§8's definition of "cold": whether ANY prior record for
    (repo_id, backend) reached READY. ``before_ts`` (ISO-8601, comparable
    lexically since both this module and history records always use
    timezone-aware ``isoformat()``) restricts the search to records
    strictly before it — the caller starting a NEW boot right now passes
    its own not-yet-written start time so a record from that same boot
    (once it later gets appended) can never count as its own "prior"
    success.
    """
    records = query(repo_id=repo_id, backend=backend, reached_ready=True, path=path)
    if before_ts is None:
        return bool(records)
    return any(str(r.get("ts", "")) < before_ts for r in records)


# ---------------------------------------------------------------------------
# ETA statistics
# ---------------------------------------------------------------------------

# SPEC.md §8's WAIT-TIME UX calibration figures — cited WITH attribution
# whenever a stat is built from these instead of real history (fewer than 2
# matching samples). Never treated as if they were measured for the
# specific repo_id being asked about; see EtaStats.source/EtaStats.note.
SPEC_WARM_BOOT_S: dict[str, float] = {
    # "Flash-Next warm boot 122s (21:22:11->21:24:13)" — SPEC.md §8.
    "RadixArk/Qwen3.8-Flash-Next-NVFP4": 122.0,
}
# "27B ~145s" — SPEC.md §8 gives this for the *inline* backend generically,
# not tied to one repo_id among the several 27B variants (NVFP4/FP8/AWQ/
# Uncensored) in SPEC.md §0's ground-truth table.
SPEC_WARM_BOOT_INLINE_GENERIC_S = 145.0

# "A cold Flash-Next boot is 4-10 min and nothing hides that" — SPEC.md §8
# HONEST LIMITS item 2. A RANGE, not a point estimate — do not collapse it
# to a single fabricated number.
SPEC_COLD_BOOT_RANGE_S: dict[str, tuple[float, float]] = {
    "flashnext": (240.0, 600.0),
}


@dataclass(frozen=True)
class EtaStats:
    """What supervisor.py / the API layer needs to render SPEC.md §8's
    "typically 4m10s · cold boot 9m50s" line, and the amber
    overrun-vs-p90 comparison, without ever fabricating a percentage."""

    repo_id: str
    backend: str
    cold: bool
    samples: int
    median_total_s: float | None
    p90_total_s: float | None
    median_phase_s: dict[str, float] = field(default_factory=dict)
    p90_phase_s: dict[str, float] = field(default_factory=dict)
    source: str = "history"  # "history" | "spec_calibration" | "unknown"
    note: str | None = None


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (0..100) over already-collected
    values. ``values`` must be non-empty; callers check that first."""
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _phase_stat(records: list[dict[str, Any]], stat_fn) -> dict[str, float]:
    by_phase: dict[str, list[float]] = {}
    for r in records:
        phases = r.get("phases") or {}
        if not isinstance(phases, dict):
            continue
        for name, secs in phases.items():
            if isinstance(secs, (int, float)):
                by_phase.setdefault(str(name), []).append(float(secs))
    return {name: stat_fn(vals) for name, vals in by_phase.items() if vals}


def _spec_calibration(repo_id: str, backend: str, cold: bool) -> EtaStats:
    if cold:
        rng = SPEC_COLD_BOOT_RANGE_S.get(backend)
        if rng is not None:
            lo, hi = rng
            return EtaStats(
                repo_id=repo_id,
                backend=backend,
                cold=True,
                samples=0,
                median_total_s=None,
                p90_total_s=None,
                source="spec_calibration",
                note=(
                    f"No cold-boot history yet for this model. SPEC.md §8: a cold "
                    f"{backend} boot is {lo / 60:.0f}-{hi / 60:.0f} min."
                ),
            )
        return EtaStats(
            repo_id=repo_id,
            backend=backend,
            cold=True,
            samples=0,
            median_total_s=None,
            p90_total_s=None,
            source="unknown",
            note="No cold-boot history yet for this model, and SPEC.md gives no cold-boot figure for this backend.",
        )

    warm = SPEC_WARM_BOOT_S.get(repo_id)
    if warm is None and backend == "inline":
        warm = SPEC_WARM_BOOT_INLINE_GENERIC_S
    if warm is not None:
        return EtaStats(
            repo_id=repo_id,
            backend=backend,
            cold=False,
            samples=0,
            median_total_s=warm,
            p90_total_s=None,
            source="spec_calibration",
            note=f"No warm-boot history yet for this model. SPEC.md §8 calibration: ~{warm:.0f}s.",
        )
    return EtaStats(
        repo_id=repo_id,
        backend=backend,
        cold=False,
        samples=0,
        median_total_s=None,
        p90_total_s=None,
        source="unknown",
        note="No warm-boot history yet for this model, and SPEC.md gives no calibration figure for it.",
    )


def eta_for(
    repo_id: str,
    backend: str,
    *,
    cold: bool,
    path: Path | str | None = None,
) -> EtaStats:
    """Median/p90 total_s and per-phase timings over prior runs matching
    (repo_id, backend, cold) that reached READY. Fewer than 2 samples ->
    fall back to SPEC.md §8's calibration figures, clearly attributed
    (SPEC.md §8: "With <2 samples cite SETUP.md figures WITH attribution");
    NEVER a fake percentage or an interpolated-from-nothing number.
    """
    records = query(repo_id=repo_id, backend=backend, cold=cold, reached_ready=True, path=path)
    totals = [float(r["total_s"]) for r in records if isinstance(r.get("total_s"), (int, float))]

    if len(totals) < 2:
        return _spec_calibration(repo_id, backend, cold)

    median_phase = _phase_stat(records, statistics.median)
    p90_phase = _phase_stat(records, lambda vs: _percentile(vs, 90))

    return EtaStats(
        repo_id=repo_id,
        backend=backend,
        cold=cold,
        samples=len(totals),
        median_total_s=statistics.median(totals),
        p90_total_s=_percentile(totals, 90),
        median_phase_s=median_phase,
        p90_phase_s=p90_phase,
        source="history",
        note=None,
    )
