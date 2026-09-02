"""Servedeck log tailing — SPEC.md §5 / §6.

vLLM's boot log for the inline backend lives behind a SYMLINK
(``local_llm/logs/qwen_server.log``) that ``qwen-server-run.sh`` re-points
with ``ln -sfn "$RUN_LOG" "$LOG_DIR/qwen_server.log"`` on every start — a
fresh, differently-named regular file underneath, atomically swapped in.
The flashnext backend's ``serve.sh`` writes a plain file directly and may
also be restarted with a fresh path by its caller.

:class:`LogTailer` must therefore never trust a held-open file handle or a
byte offset alone: it re-``stat``s the path (inode + size) on every poll and
treats an inode change as a rotation (seek to start of the new file) and a
same-inode size decrease as an in-place truncation (also seek to start).
Only a genuine same-inode size increase is a normal "more bytes appended"
poll.

No phase/regex logic lives here — that's phases.py. This module hands back
plain line strings; callers feed them to a phases.PhaseTracker themselves,
and should construct a *fresh* tracker whenever ``rotated`` is True, since a
rotation means a new vLLM boot (new PID, new phase sequence from INIT).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# A single poll reads at most this many new bytes before returning; callers
# polling on a short interval (SPEC §8: log event <=250ms) will drain a
# fast-growing log across a few polls rather than blocking on one giant read.
DEFAULT_MAX_BYTES = 8 * 1024 * 1024

# Size of the file-head snapshot kept for same-inode truncation detection.
# A plain size/offset comparison misses the case where a launcher truncates
# a STABLE filename in place (`open(path, "w")`, same inode) and the fresh
# content happens to end up longer than our old read offset -- the size
# check alone then reads as ordinary growth. qwen-server-run.sh's own
# history is the proof this is a real failure mode, not a hypothetical: it
# used to do exactly `> "$LOG_FILE"` on every start (replaced by the
# timestamped-symlink scheme for this reason) and a future regression or a
# different launcher could do it again. Real boot logs diverge (differing
# PID/timestamp) well inside the first 100 bytes, so 512 is ample headroom
# without making the extra per-poll read meaningfully expensive.
HEAD_CHARS = 512


@dataclass
class TailResult:
    lines: list[str]
    rotated: bool
    truncated: bool
    exists: bool
    size: int
    resolved_path: Path | None  # the real file this poll actually read, if any


class LogTailer:
    """Follows ``path`` (which may be a symlink re-pointed between polls)
    and yields newly-appended complete lines.

    Usage::

        tailer = LogTailer("/path/to/qwen_server.log")
        while True:
            result = tailer.poll()
            if result.rotated:
                tracker = PhaseTracker()  # new boot, new FSM
            for line in result.lines:
                for event in tracker.feed(line):
                    ...
    """

    def __init__(self, path: str | Path, *, from_end: bool = False) -> None:
        """`from_end` skips existing content on the first poll.

        Needed when adopting a server that is already running: its log holds
        every previous boot, and replaying those would drive the phase FSM
        through old, irrelevant transitions.
        """
        self.path = Path(path)
        self._from_end = from_end
        self._ino: int | None = None
        self._dev: int | None = None
        self._offset: int = 0
        self._partial: str = ""
        # Snapshot of the current inode's first HEAD_CHARS characters, as of
        # the last time self._offset was (re)established at 0. None means
        # "nothing to compare yet" (no generation read from the start yet).
        self._head: str | None = None

    def reset(self) -> None:
        """Forget all tailing state (next poll re-reads from the top)."""
        self._ino = None
        self._dev = None
        self._offset = 0
        self._partial = ""
        self._head = None

    def _read_head(self) -> str:
        """Best-effort re-read of the file's current first HEAD_CHARS chars.

        Only used to detect a same-inode truncate-then-rewrite; any failure
        here (raced away between our stat() and this open()) just yields ""
        which safely never trips the mismatch check on its own (compare_len
        would be 0).
        """
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                return f.read(HEAD_CHARS)
        except OSError:
            return ""

    def poll(self, max_bytes: int = DEFAULT_MAX_BYTES) -> TailResult:
        try:
            st = os.stat(self.path)  # follows the symlink to its current target
        except OSError:
            # Missing, or a dangling/broken symlink mid-swap. Not a rotation
            # of a file we were reading — it's simply not there right now.
            # Keep prior offset state: `ln -sfn` is atomic, so a momentarily
            # missing stat here means "hasn't been created yet", and once it
            # exists it'll look like a fresh inode and rotate correctly.
            was_seen = self._ino is not None
            if was_seen:
                # The path existed before and doesn't now (deleted, not just
                # swapped) — clear state so the next appearance reads fresh.
                self.reset()
            return TailResult(lines=[], rotated=False, truncated=False, exists=False, size=0, resolved_path=None)

        rotated = False
        truncated = False

        is_new_target = self._ino is None or st.st_ino != self._ino or st.st_dev != self._dev
        first_ever = self._ino is None
        if is_new_target:
            rotated = True
            self._offset = 0
            self._partial = ""
            self._head = None
            if first_ever and self._from_end:
                # Adopting a running server: its log holds every previous boot.
                # Replaying those would drive the phase FSM through old, long
                # finished transitions. Start at EOF and report no rotation,
                # since nothing actually rotated.
                self._offset = st.st_size
                self._head = self._read_head()
                self._from_end = False
                rotated = False
        elif st.st_size < self._offset:
            # Same inode, shorter than what we already consumed: truncated
            # in place (e.g. a launcher that does `open(path, "w")` on a
            # stable filename instead of swapping the symlink).
            truncated = True
            self._offset = 0
            self._partial = ""
            self._head = None
        elif self._head is not None:
            # Same inode, size unchanged-or-grew: on its own that reads as
            # ordinary appended growth, but a same-inode O_TRUNC-then-rewrite
            # can ALSO land here whenever the fresh content happens to be at
            # least as long as our old offset. Confirm the file's head is
            # still the head we last saw before trusting self._offset.
            current_head = self._read_head()
            compare_len = min(len(self._head), len(current_head))
            if compare_len > 0 and current_head[:compare_len] != self._head[:compare_len]:
                truncated = True
                self._offset = 0
                self._partial = ""
                self._head = None

        self._ino = st.st_ino
        self._dev = st.st_dev

        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._offset)
                chunk = f.read(max_bytes)
                self._offset = f.tell()
        except OSError:
            # Raced: file vanished between stat() and open(). Report what we
            # know from the stat; state stays as set above so a later poll
            # either rotates (new file appears) or resumes cleanly.
            return TailResult(
                lines=[], rotated=rotated, truncated=truncated, exists=False, size=st.st_size, resolved_path=None
            )

        data = self._partial + chunk
        parts = data.split("\n")
        self._partial = parts[-1]
        lines = parts[:-1]

        if self._head is None and data:
            # This poll (re)started reading from offset 0 (rotation, a
            # detected truncation, or the very first poll ever) — `data` IS
            # the file's new head. Snapshot it now so the NEXT poll can
            # detect a same-inode truncate-then-rewrite even if it grows
            # past today's offset.
            #
            # `and data` is load-bearing: storing "" for an empty first read
            # pins _head non-None forever, and every later comparison has
            # compare_len == 0, which silently disables truncation detection
            # for the life of the tailer. Staying None re-arms the snapshot on
            # the first poll that actually sees content.
            self._head = data[:HEAD_CHARS]

        return TailResult(
            lines=lines,
            rotated=rotated,
            truncated=truncated,
            exists=True,
            size=st.st_size,
            resolved_path=self.path,
        )
