# BF16 PLE table cleanup — prepared, not executed

Everything needed to delete the 95.368 GiB BF16 PLE n-gram table from the
Flash-Next checkpoint, once a human confirms the FP8 switch serves correctly.

**Nothing here has deleted anything.** The reconnaissance was read-only; the
deletion script has only ever been run in dry-run mode; the test suite runs
against throwaway fixtures under the session scratchpad.

## Files

| File | What it is |
|---|---|
| `bf16-ple-manifest.json` | Machine-readable inventory: all 43 files, symlink targets, blob paths, sha256s, sizes, inodes, hardlink counts, referrer counts, tensor names |
| `bf16-ple-manifest.md` | The same, readable, with totals and the space-semantics warning |
| `SAFETY-ANALYSIS.md` | Every safety claim with the command that produced it and that command's output |
| `delete-bf16-ple.sh` | The guarded deletion. **Dry run by default.** |
| `test_delete_guards.sh` | Guard tests on fixture trees, plus `--mutate` for fail-before proof |
| `RECOVERY.md` | How to get the BF16 table back, what it costs, and why it probably is not worth it |

## The numbers

| | |
|---|---|
| Files | 43 × `ple-bf16-NN.safetensors` |
| Tensors | 128 n-gram shards, BF16, `[2500012, 160]` — and nothing else |
| Reclaimable | **102,400,512,064 B = 95.3679 GiB** |
| FP8 replacement | 43 files, 47.684 GiB, already installed |
| Blobs shared with anything else | **none** |
| Processes holding them open | **none** |

## To execute

```bash
cd ~/Projects/vllm-qwen38next/ple_fp8_cleanup

# 1. see the plan (safe, changes nothing)
./delete-bf16-ple.sh

# 2. only after verifying the FP8 switch actually serves correctly:
echo "verified by <name> on $(date -Is): FP8 switch serving correctly" > FP8_SWITCH_VERIFIED

# 3. dry run again -- now every guard passes and the full plan prints
./delete-bf16-ple.sh

# 4. act
./delete-bf16-ple.sh --yes-really-delete
```

Step 2 is the whole point of the marker: it is a human asserting something no
script can check. Do not create it to make the script stop complaining.

Before step 2, run `ple_fp8/verify_fp8_table.py` — it compares FP8 against BF16,
and after the deletion it can never run again without a 95 GiB re-download.

## What the guards check

| | |
|---|---|
| G1 | manifest parses and describes this snapshot |
| G2 | the index lists the FP8 shards and **no** BF16 PLE shard |
| G3 | all 43 FP8 shards exist at their expected byte sizes |
| G4 | the operator-intent marker exists (+ `--yes-really-delete` to act) |
| G5 | every blob: right size, hardlink count 1, exactly one referrer in the whole hub |
| G6 | no live process has any target blob mmapped or open |

Any failure aborts before a single `unlink()`. The script removes each snapshot
symlink together with its blob, then verifies no dangling symlink is left
anywhere under the hub, and reports reclaimed bytes from `df` before and after.

## Things worth knowing

**Deleting a symlink frees nothing.** The bytes are in
`blobs/<sha256>`. That is why G5 counts referrers instead of trusting the
snapshot listing, and why the script removes both names together.

**`SNAP` is a snapd environment variable.** Snapd exports
`SNAP=/snap/<app>/<revision>` into every shell launched from a snap-packaged
editor. The first draft of `enable.sh`/`rollback.sh` read a bare `$SNAP` and was
silently retargeted at `/snap/code/260`; both now use `PLE_SNAP`/`PLE_STAGE` and
ignore it. This script uses `PLE_CLEANUP_SNAP` for the same reason.

**Rollback becomes a trap once the shards are gone.** `rollback.sh` would restore
the BF16 index *and delete the 43 FP8 shards*, leaving a checkpoint naming files
that no longer exist. The deletion script defuses this by renaming **both**
`model.safetensors.index.json.prev-state` and `...bf16` aside, so `rollback.sh`
aborts before removing anything. Renaming only the `.bf16` copy is not enough --
the symlink branch never reads it. See `RECOVERY.md` to re-arm them.

**The old index blob is not an orphan.** `blobs/2d2c3617...e407` (28.7 MiB) has no
symlink pointing at it but is `rollback.sh`'s restore target. Do not sweep it.

## Tests

```bash
./test_delete_guards.sh                     # 11 assertions, all green
./test_delete_guards.sh --mutate 2          # weakens G2       -> its test fails
./test_delete_guards.sh --mutate 5          # weakens G5       -> its test fails
./test_delete_guards.sh --mutate rollback   # regresses the interlock -> its test fails
```

The mutation runs are the fail-before proof: they patch a copy of the script,
never the original, and demonstrate that each guard's test actually fails when
that guard is removed.
