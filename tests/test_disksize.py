"""What a size figure is allowed to mean.

The disk numbers servedeck showed were wrong in five separable ways, and each
test here pins exactly one of them so a regression names its own cause:

* a blob reachable under two names counted twice (hardlink, or two snapshot
  symlinks into one blob) -- :func:`test_a_hardlinked_blob_is_counted_once`,
  :func:`test_a_blob_two_snapshot_symlinks_point_at_is_counted_once`
* apparent size mistaken for space consumed (sparse files) --
  :func:`test_a_sparse_file_costs_its_blocks_not_its_apparent_size`
* GiB rendered with the letters "GB" -- :func:`test_binary_and_decimal_units_are_not_interchangeable`
* filesystem totals guessed from a walk instead of read from statvfs --
  :func:`test_filesystem_usage_is_statvfs_not_a_walk`
* bytes on another mount added to a figure compared against this ``df`` --
  :func:`test_blobs_on_another_filesystem_are_not_counted_as_local_space`

Every fixture is built the way the box builds the real thing: os.link for the
FP8 PLE table's 43 hardlinks, a blobs/ + snapshots/ pair for the hub cache's
symlinks, and a truncate-then-seek sparse file.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from servedeck.disksize import (
    GB,
    GIB,
    FilesystemUsage,
    filesystem_usage,
    format_bytes,
    scan_tree,
)

#: st_blocks is POSIX-defined in 512-byte units on every filesystem.
BLOCK = 512


def _write(path: Path, nbytes: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(b"\0" * nbytes)
    return path


# --------------------------------------------------------------------------- #
# One blob, many names
# --------------------------------------------------------------------------- #


def test_a_hardlinked_blob_is_counted_once(tmp_path: Path) -> None:
    """The FP8 PLE table shape: one set of bytes, two directory entries.

    43 shards of 1.2 GB each were installed into the model snapshot as
    HARDLINKS of the files in ~/.cache/huggingface/ple-fp8-staging (``stat -c
    %h`` says 2). They occupy 47.68 GiB once. Adding up the two trees per-name
    reports 95.36 GiB -- and tells the operator that deleting both would free
    twice the space that exists.
    """
    size = 4 * 1024 * 1024
    blob = _write(tmp_path / "snapshot" / "model-plefp8-00.safetensors", size)
    link = tmp_path / "staging" / "model-plefp8-00.safetensors"
    link.parent.mkdir()
    os.link(blob, link)
    assert blob.stat().st_nlink == 2, "fixture is not actually hardlinked"

    u = scan_tree(tmp_path)

    assert u.links == 2, "both names should have been visited"
    assert u.blobs == 1, "two names for one inode are one blob"
    assert u.apparent_bytes == size
    # The whole point: the total is one file's worth, not two.
    assert u.allocated_bytes == blob.stat().st_blocks * BLOCK
    assert u.allocated_bytes < 2 * size


def test_a_blob_two_snapshot_symlinks_point_at_is_counted_once(tmp_path: Path) -> None:
    """The HuggingFace cache shape: snapshots/ is symlinks into blobs/.

    Walking a ``models--*`` directory follows each snapshot symlink back to the
    blob it already visited under blobs/. On this box that turns a 126.02 GiB
    model into 204.26 GiB.
    """
    size = 3 * 1024 * 1024
    blob = _write(tmp_path / "blobs" / "deadbeef", size)
    snap = tmp_path / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "model-00001.safetensors").symlink_to(blob)
    (snap / "also-model-00001.safetensors").symlink_to(blob)

    u = scan_tree(tmp_path)

    assert u.links == 3, "blobs/ entry plus both snapshot names"
    assert u.blobs == 1
    assert u.apparent_bytes == size


def test_counting_once_does_not_collapse_distinct_files(tmp_path: Path) -> None:
    """Over-correction guard.

    Deduplicating on (st_dev, st_ino) must not merge two files that merely
    have the same size, the same name in different directories, or identical
    content. A dedup keyed on anything but the inode does exactly that, and
    would report a 5-shard model as one shard.
    """
    size = 1024 * 1024
    for i in range(5):
        _write(tmp_path / f"dir{i}" / "model-00001.safetensors", size)

    u = scan_tree(tmp_path)

    assert u.blobs == 5, "five distinct inodes are five blobs"
    assert u.links == 5
    assert u.apparent_bytes == 5 * size


# --------------------------------------------------------------------------- #
# Apparent size vs space consumed
# --------------------------------------------------------------------------- #


def test_a_sparse_file_costs_its_blocks_not_its_apparent_size(tmp_path: Path) -> None:
    """``st_size`` is a claim; ``st_blocks`` is the bill.

    A sparse file reports gigabytes and occupies kilobytes. "How much would
    deleting this free" is the block figure; "how much will this weigh once
    loaded" is the apparent one, because a hole still becomes zeros in RAM.
    Both are kept, and they must not be confused for each other.
    """
    p = tmp_path / "sparse.safetensors"
    with p.open("wb") as fh:
        fh.truncate(512 * 1024 * 1024)  # 512 MiB of nothing
        fh.seek(0)
        fh.write(b"x")
    st = p.stat()
    if st.st_blocks * BLOCK >= st.st_size:
        pytest.skip("filesystem does not support sparse files")

    u = scan_tree(tmp_path)

    assert u.apparent_bytes == 512 * 1024 * 1024
    assert u.allocated_bytes == st.st_blocks * BLOCK
    assert u.allocated_bytes < u.apparent_bytes / 100


# --------------------------------------------------------------------------- #
# Symlinks that do not resolve
# --------------------------------------------------------------------------- #


def test_a_dangling_symlink_is_reported_not_silently_zero(tmp_path: Path) -> None:
    """Half this box's hub cache symlinks onto a removable exfat stick.

    With the stick unplugged those links resolve to nothing. Counting them as
    0-byte files makes a 126 GiB model read as an empty download; the walker
    has to say the bytes are unreachable, not that they are absent.
    """
    snap = tmp_path / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "model-00001.safetensors").symlink_to(tmp_path / "not-mounted" / "blob")
    _write(snap / "config.json", 100)

    u = scan_tree(tmp_path)

    assert u.dangling == 1
    assert u.blobs == 1, "only config.json resolved"
    assert u.apparent_bytes == 100


def test_blobs_on_another_filesystem_are_not_counted_as_local_space(tmp_path: Path) -> None:
    """A model can be real and still free nothing on THIS filesystem.

    RadixArk/Qwen3.8-Flash-Next-NVFP4 is 126.00 GiB of blobs reached through
    symlinks onto an exfat stick under /run/media. It occupies 4 KiB of ``/``.
    Adding its 126 GiB to a total shown beside a df figure invites the
    operator to delete it expecting 126 GiB back.
    """
    # pytest's tmp_path is on /tmp, which is a tmpfs on this box; HOME is on
    # the nvme root filesystem. If they ever share a device this fixture
    # cannot express "another mount" and the test says so rather than passing
    # vacuously.
    import tempfile

    with tempfile.TemporaryDirectory(dir=str(Path.home())) as other:
        other_path = Path(other)
        if os.stat(other_path).st_dev == os.stat(tmp_path).st_dev:
            pytest.skip("no second filesystem available to express the split")
        size = 2 * 1024 * 1024
        far = _write(other_path / "blob", size)
        near = _write(tmp_path / "here.safetensors", 4096)
        snap = tmp_path / "snapshots" / "rev"
        snap.mkdir(parents=True)
        (snap / "model-00001.safetensors").symlink_to(far)

        u = scan_tree(tmp_path)

        assert u.blobs == 2
        assert u.allocated_bytes == (
            far.stat().st_blocks * BLOCK + near.stat().st_blocks * BLOCK
        )
        assert u.foreign_bytes == far.stat().st_blocks * BLOCK
        assert u.local_bytes == near.stat().st_blocks * BLOCK
        assert len(u.devices) == 2


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #


def test_binary_and_decimal_units_are_not_interchangeable() -> None:
    """The 7.4% error that reads as a rounding slip.

    The model cards rendered a GiB value inside markup that said "GB": 125.99
    GiB was displayed as "125.91 GB". The two differ by 1024**3/1000**3, and
    the only defence is that the number and its unit are produced together.
    """
    n = 135 * GB  # 135 GB exactly
    assert format_bytes(n, binary=False) == "135.00 GB"
    assert format_bytes(n, binary=True) == "125.73 GiB"
    # The bug, stated as arithmetic: the same byte count named two ways is two
    # different numbers, ~7.4% apart.
    assert 1.07 < GIB / GB < 1.08

    assert format_bytes(126 * GIB) == "126.00 GiB"
    assert format_bytes(126 * GIB, binary=False) == "135.29 GB"


def test_format_bytes_always_carries_its_unit() -> None:
    """No call site may receive a bare number to label itself."""
    for n in (0, 1, 999, 1024, 5 * 1024**2, 3 * GIB, 7 * 1024**4):
        out = format_bytes(n)
        assert out.split(" ")[-1] in {"B", "KiB", "MiB", "GiB", "TiB", "PiB"}
    assert format_bytes(0) == "0 B"
    assert format_bytes(1023) == "1023 B", "bytes must not be shown as fractional KiB"
    assert format_bytes(1024) == "1.00 KiB"


# --------------------------------------------------------------------------- #
# The filesystem's own answer
# --------------------------------------------------------------------------- #


def test_filesystem_usage_is_statvfs_not_a_walk() -> None:
    """Every field must equal statvfs, byte for byte.

    A directory walk cannot see free space at all and under-reports "used" by
    every tree the process may not read. This asserts the figures come from
    the kernel and that the frsize/bsize distinction is respected: on a
    filesystem where they differ, using f_bsize inflates all four numbers.
    """
    u = filesystem_usage("/")
    s = os.statvfs("/")
    frsize = s.f_frsize or s.f_bsize

    assert u.total_bytes == s.f_blocks * frsize
    assert u.free_bytes == s.f_bfree * frsize
    assert u.avail_bytes == s.f_bavail * frsize
    assert u.used_bytes == (s.f_blocks - s.f_bfree) * frsize
    # shutil.disk_usage is the same syscall; agreeing with it rules out a
    # units slip that happens to be self-consistent.
    import shutil

    du = shutil.disk_usage("/")
    assert u.total_bytes == du.total
    assert u.used_bytes == du.used
    assert u.avail_bytes == du.free


def test_used_pct_is_the_df_formula_not_used_over_total() -> None:
    """``df`` computes Use% as used/(used+avail), excluding root-reserved blocks.

    On this 915 GiB ext4 the root reserve is ~45 GiB, so used/total reads about
    four points low -- and the operator is comparing against ``df``.
    """
    # 100 units total, 50 used, 40 available to a non-root user, 10 reserved.
    u = FilesystemUsage(
        path="/x", total_bytes=100, used_bytes=50, free_bytes=50, avail_bytes=40
    )
    assert u.used_pct == pytest.approx(100 * 50 / 90)
    assert u.used_pct != pytest.approx(50.0), "used/total ignores the root reserve"


def test_filesystem_usage_of_an_empty_filesystem_does_not_divide_by_zero() -> None:
    u = FilesystemUsage(path="/x", total_bytes=0, used_bytes=0, free_bytes=0, avail_bytes=0)
    assert u.used_pct == 0.0


def test_scan_tree_of_a_missing_directory_is_empty_not_an_exception(tmp_path: Path) -> None:
    """A hub dir that was deleted between the listing and the scan must not
    take the whole panel down with it."""
    u = scan_tree(tmp_path / "gone")
    assert (u.allocated_bytes, u.apparent_bytes, u.links, u.blobs) == (0, 0, 0, 0)
