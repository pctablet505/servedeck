"""How much space something actually takes, and how much the filesystem has left.

Every number servedeck showed about disk was wrong in one of five ways, and
this module exists so that each one has exactly one right answer:

1. **Apparent size instead of blocks.** ``st_size`` is what the file claims;
   ``st_blocks * 512`` is what the filesystem gave it. They differ for sparse
   files (a 1 GiB sparse file costs a few KiB) and they differ on exfat, where
   the cluster size rounds a 354,098,208-byte shard up to 354,156,544 bytes.
   "How much room would deleting this give me back" is the block figure.

2. **Following a symlink and counting the blob twice.** A HuggingFace snapshot
   directory is entirely symlinks into ``../../blobs/<sha>``. Two snapshot
   entries can point at the same blob, and two snapshots of one repo almost
   always do. ``stat()`` follows the link by default, so a naive sum counts the
   same bytes once per name.

3. **Counting hardlinked content once per link.** The FP8 PLE table on this box
   lives in the model snapshot *and* in ``~/.cache/huggingface/ple-fp8-staging``
   as 43 hardlinks (``stat -c %h`` says 2). It occupies 47.68 GiB once. A
   directory walk that adds up the two trees reports 95.36 GiB of space that
   deleting both would not return.

   (2) and (3) are the same bug: the identity of stored bytes is the inode, not
   the path. Deduplicating on ``(st_dev, st_ino)`` after resolving symlinks
   fixes both at once, which is exactly what ``du`` does.

4. **Mixing GiB and GB.** 1024**3 versus 1000**3 is a 7.37% error — big enough
   that the number looks wrong, small enough that it looks like it might just
   be rounding. :func:`format_bytes` always prints the unit it used.

5. **Reading a cached value that is never refreshed.** Not this module's
   problem to hold, but its callers': a size scanned once at process start
   still claimed 95.37 GiB of BF16 PLE table hours after that table was
   deleted.

One trap this module deliberately does NOT paper over: bytes on a *different
filesystem*. Half the hub cache here symlinks out to an exfat stick mounted at
/run/media/.../Ventoy. Those blobs are real, but deleting them frees nothing on
``/``, so :class:`TreeUsage` keeps them in a separate total instead of quietly
adding them to a figure the operator will compare against ``df``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: Binary and decimal bases, named so a call site cannot silently pick the
#: wrong one by writing a bare 1024**3.
GIB = 1024**3
GB = 1000**3

#: ``st_blocks`` is defined by POSIX in 512-byte units regardless of the
#: filesystem's own block size. It is NOT ``st_blksize``.
_BLOCK_UNIT = 512

_BINARY_UNITS = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
_DECIMAL_UNITS = ("B", "kB", "MB", "GB", "TB", "PB")


def format_bytes(n: float, *, binary: bool = True, digits: int = 2) -> str:
    """``n`` bytes as a string that names its own unit.

    Binary by default, because that is what ``df``, ``free``, nvidia-smi and
    every vLLM log line on this box mean when they say "GB". The unit is part
    of the return value on purpose: a bare number labelled by markup somewhere
    else is how ``125.91`` GiB came to be rendered as ``125.91 GB``.
    """
    units = _BINARY_UNITS if binary else _DECIMAL_UNITS
    base = 1024.0 if binary else 1000.0
    sign = "-" if n < 0 else ""
    v = float(abs(n))
    i = 0
    while v >= base and i < len(units) - 1:
        v /= base
        i += 1
    if i == 0:
        return f"{sign}{int(v)} {units[0]}"
    return f"{sign}{v:.{digits}f} {units[i]}"


@dataclass(frozen=True)
class FilesystemUsage:
    """What ``statvfs`` says about the filesystem holding one path.

    ``free_bytes`` and ``avail_bytes`` differ by the reserved-blocks pool
    (ext4 keeps 5% for root by default). ``df``'s "Avail" column and its
    "Use%" are both computed from ``avail``, so that is the pair the UI
    shows — an operator who compares servedeck against ``df`` must not find
    a 45 GiB discrepancy that is really just ext4's root reserve.
    """

    path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    avail_bytes: int

    @property
    def used_pct(self) -> float:
        """Percent full the way ``df`` computes it: used / (used + avail).

        NOT used / total. On a 915 GiB ext4 with 45 GiB of root reserve those
        differ by four percentage points, and ``df`` is the number the
        operator will check against.
        """
        denom = self.used_bytes + self.avail_bytes
        if denom <= 0:
            return 0.0
        return 100.0 * self.used_bytes / denom


def filesystem_usage(path: str | os.PathLike[str] = "/") -> FilesystemUsage:
    """Filesystem totals from ``statvfs`` — never from a directory walk.

    A walk can only ever see files it is allowed to read, so it under-reports
    "used" by every unreadable tree on the box; and it cannot see free space
    at all. The kernel already knows both.
    """
    s = os.statvfs(path)
    # f_frsize is the fragment size the block counts are in. f_bsize is the
    # preferred I/O size and is NOT what f_blocks counts; on filesystems where
    # they differ, using f_bsize inflates every figure. Fall back only if
    # f_frsize is zero, which some exotic filesystems report.
    frsize = s.f_frsize or s.f_bsize
    total = s.f_blocks * frsize
    free = s.f_bfree * frsize
    avail = s.f_bavail * frsize
    return FilesystemUsage(
        path=str(path),
        total_bytes=total,
        used_bytes=total - free,
        free_bytes=free,
        avail_bytes=avail,
    )


@dataclass(frozen=True)
class TreeUsage:
    """The cost of a directory tree, with each stored blob counted once.

    ``allocated_bytes`` answers "how much would deleting this free"; it is the
    sum of ``st_blocks`` over distinct inodes, which is what ``du`` reports.
    ``apparent_bytes`` is the sum of ``st_size`` over the same distinct inodes
    — what ``du --apparent-size`` reports, and the right input to "how much
    will this weigh once loaded", since a sparse hole still becomes zeros in
    RAM.

    ``links`` counts names visited, ``blobs`` counts distinct inodes behind
    them. ``links > blobs`` is the signature of hardlinked or repeatedly
    symlinked content, and is the evidence that the naive sum would have been
    wrong.
    """

    allocated_bytes: int
    apparent_bytes: int
    links: int
    blobs: int
    dangling: int
    #: Allocated bytes living on a filesystem other than the one the scan root
    #: is on. Included in ``allocated_bytes``; broken out so nothing subtracts
    #: them from the wrong ``df``.
    foreign_bytes: int
    #: Distinct ``st_dev`` values seen behind the tree's files.
    devices: frozenset[int]

    @property
    def local_bytes(self) -> int:
        """Allocated bytes on the same filesystem as the scan root."""
        return self.allocated_bytes - self.foreign_bytes


def _resolved_stat(path: Path) -> os.stat_result | None:
    """``stat`` through symlinks, or None if it does not resolve.

    A hub snapshot whose blobs live on an unmounted drive is all dangling
    symlinks; that must read as "unresolved", never as "zero bytes".
    """
    try:
        return path.stat()  # follow_symlinks=True
    except OSError:
        return None


def scan_tree(
    root: str | os.PathLike[str],
    *,
    recursive: bool = True,
    match: str | None = None,
) -> TreeUsage:
    """Walk ``root`` and total it with every blob counted exactly once.

    ``match`` restricts which *names* contribute (a filename suffix such as
    ``".safetensors"``); the deduplication set still spans everything visited,
    so a blob already counted under one name cannot be counted again under a
    matching one.

    Directories are not counted: their own inodes hold metadata, not payload,
    and including them makes a tree of many small files look bigger than the
    bytes it stores. ``du`` counts them; the question here is "how big is this
    model", not "what does the metadata cost".
    """
    root_path = Path(root)
    try:
        root_dev: int | None = os.stat(root_path).st_dev
    except OSError:
        return TreeUsage(0, 0, 0, 0, 0, 0, frozenset())

    seen: set[tuple[int, int]] = set()
    allocated = apparent = links = dangling = foreign = 0
    devices: set[int] = set()

    if recursive:
        entries = root_path.rglob("*")
    else:
        try:
            entries = iter(root_path.iterdir())
        except OSError:
            return TreeUsage(0, 0, 0, 0, 0, 0, frozenset())

    for p in entries:
        st = _resolved_stat(p)
        if st is None:
            # Only a symlink can dangle; an unreadable real entry is a
            # different problem and must not be reported as a missing blob.
            if p.is_symlink():
                dangling += 1
            continue
        if not _is_regular_file(st):
            continue
        if match is not None and not p.name.endswith(match):
            continue
        links += 1
        key = (st.st_dev, st.st_ino)
        if key in seen:
            # Second name for bytes already counted: a hardlink, or a second
            # snapshot entry symlinked to the same blob. This is the branch
            # whose absence made the FP8 PLE table read as 95.36 GiB.
            continue
        seen.add(key)
        devices.add(st.st_dev)
        blocks = st.st_blocks * _BLOCK_UNIT
        allocated += blocks
        apparent += st.st_size
        if root_dev is not None and st.st_dev != root_dev:
            foreign += blocks

    return TreeUsage(
        allocated_bytes=allocated,
        apparent_bytes=apparent,
        links=links,
        blobs=len(seen),
        dangling=dangling,
        foreign_bytes=foreign,
        devices=frozenset(devices),
    )


def _is_regular_file(st: os.stat_result) -> bool:
    import stat as _stat

    return _stat.S_ISREG(st.st_mode)
