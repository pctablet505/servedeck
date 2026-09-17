# Restoring the BF16 PLE n-gram table

If the 43 `ple-bf16-*.safetensors` files have been deleted and you need them back.

Repo: `mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4`
Revision pinned in the local cache: `f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f`

---

## Read this before you download 95 GiB

The FP8 table that replaced these files is a **re-encoding of the same numbers**,
not a different model. Per `ple-fp8-staging/conversion_report.json`, the
conversion scanned all 51,200,245,760 elements and reports
`bit_exact_fraction: 1.0`, `rms_error: 0.0`, `max_abs_error: 0.0`.

So restoring BF16 buys you **no accuracy**. The only things it buys:

1. a return to the old code path (`VLLM_PLE_FP8_CHECKPOINT` unset, the BF16
   n-gram loader), if the FP8 path turns out to have a *code* bug; and
2. the ability to re-verify the FP8 table against its original.

It costs 95.368 GiB of download and 95.368 GiB of disk.

### One honest caveat about "lossless"

The same conversion report contains a number that contradicts the bit-exact
claim: `clamped_elements: 1`, and in the scale-selection candidates,
`clipped: 2` for the chosen `grid/2^0` scale.

The arithmetic says a clamp there is not free:

```
amax                      0.08935546875
chosen scale              0.00019931793212890625
max representable  448 x scale = 0.08929443359375
amax / scale            = 448.306...   -> clamps to 448
error on that element   = 6.103515625e-05   (6.8e-4 relative)
```

`bit_exact_fraction: 1.0` and `clamped_elements: 1` cannot both be true. The
worst case is that **one element out of 51.2 billion** differs by ~6.1e-05.
That is almost certainly irrelevant to model behaviour, but it means "lossless"
is accurate to within one clamped element, not literally exact.

This matters here for one reason only: **once the BF16 files are gone, that
discrepancy can no longer be checked without re-downloading 95 GiB.** Run
`ple_fp8/verify_fp8_table.py` while both tables are still on disk.

---

## The restore command

Downloads only the 43 n-gram shard files, not the ~172 GiB repo.

```bash
# the venv that already has huggingface_hub 1.29.0
HF=~/Projects/vllm-qwen38next/.venv-next/bin/hf

"$HF" download mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4 \
    --revision f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f \
    --include 'ple-bf16-*.safetensors'
```

`hf download` writes into the standard cache (`~/.cache/huggingface/hub`), so
the files land back in the same snapshot directory as symlinks into
`../../blobs/<sha256>`, exactly as they were.

<details>
<summary>Equivalent Python, and the older <code>huggingface-cli</code> spelling</summary>

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4",
    revision="f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f",
    allow_patterns=["ple-bf16-*.safetensors"],
)
```

```bash
# deprecated alias, still present in huggingface_hub 1.29.0
huggingface-cli download mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4 \
    --revision f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f \
    --include 'ple-bf16-*.safetensors'
```

</details>

### Download size

| | |
|---|---|
| Files | 43 (`ple-bf16-00` … `ple-bf16-42`) |
| Bytes | 102,400,512,064 |
| Size | **95.368 GiB** (102.401 GB) |
| Per shard | 2,400,012,000 B (×34), 2,400,012,008 B (×8), 1,600,008,000 B (×1) |

safetensors are uncompressed, so wire size equals disk size. Check you have the
room first — the filesystem was at 90% before this cleanup.

Exact per-file sizes and sha256 blob names are in `bf16-ple-manifest.json`;
after downloading you can confirm nothing changed upstream by comparing the
blob filenames, which HuggingFace derives from the file's sha256.

---

## Restoring the BF16 *index* as well

Downloading the shards is only half of a rollback. The snapshot's
`model.safetensors.index.json` must also map the n-gram shards back to the BF16
files, and the FP8 files must stop matching `serve.sh`'s
`model-plefp8-*.safetensors` glob (that glob is what sets
`VLLM_PLE_FP8_CHECKPOINT=1`).

`ple_fp8/rollback.sh` does both, driven by two sidecars that `enable.sh` wrote:

| Sidecar | What it does |
|---|---|
| `model.safetensors.index.json.prev-state` | records that the index was a **symlink** and to which blob, plus mode/mtime |
| `model.safetensors.index.json.bf16` | a plain copy of the BF16 index, used only when the sidecar is absent |
| `model.safetensors.index.json.fp8-manifest` | the 43 FP8 filenames to remove |

**`delete-bf16-ple.sh` deliberately renames the first two aside:**

```
model.safetensors.index.json.prev-state  ->  ....prev-state.STALE-ple-bf16-deleted
model.safetensors.index.json.bf16        ->  ....bf16.STALE-ple-bf16-deleted
```

That is a safety interlock, not damage. Without it, `rollback.sh` run after the
deletion would restore the BF16 index **and delete the 43 FP8 shards**, leaving a
checkpoint that names 43 files which no longer exist. With both renamed,
`rollback.sh` stops at `FATAL: no BF16 index backup` before removing anything.

> Renaming only the `.bf16` copy would not work: when `.prev-state` says
> `kind=symlink` and the target still resolves, `rollback.sh` recreates the symlink
> directly and never opens the backup.

So the full restore is:

```bash
SNAPDIR=~/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f

# 1. get the shards back (command above), then
# 2. re-arm the rollback sidecars
for sc in model.safetensors.index.json.prev-state model.safetensors.index.json.bf16; do
    mv "$SNAPDIR/$sc.STALE-ple-bf16-deleted" "$SNAPDIR/$sc"
done

# 3. now the normal rollback works
~/Projects/vllm-qwen38next/ple_fp8/rollback.sh
```

Step 2 is the one that is easy to forget. Do not skip step 1 and jump to step 3.

### Do not sweep the old index blob

`enable.sh` replaced the index symlink with a regular file, which left
`blobs/2d2c3617...e407` (28.7 MiB) with no symlink pointing at it. It looks like an
orphan to any cache-cleaning tool. It is not: it is the exact blob `.prev-state`
tells `rollback.sh` to re-link. Deleting it downgrades rollback to the regular-file
path. `delete-bf16-ple.sh` leaves it alone.

> Historical note: earlier versions of `enable.sh` / `rollback.sh` read a bare
> `SNAP` environment variable, which snapd overrides to `/snap/<app>/<rev>` in any
> terminal launched from a snap-packaged editor. Both scripts now use `PLE_SNAP` /
> `PLE_STAGE` and ignore bare `SNAP`, so this no longer applies -- but if you are
> looking at an older copy, run it as `env -u SNAP ./rollback.sh`.

## Recovering the FP8 table instead

If it is the FP8 side you lost, there is no download: regenerate it from the
BF16 table with `ple_fp8/convert_ple_bf16_to_fp8.py`, or re-copy from
`~/.cache/huggingface/ple-fp8-staging/`, which `rollback.sh` deliberately leaves
in place. Note that the staging copy is what the snapshot files are hardlinked
to — after `enable.sh` the two names share one set of extents, so deleting
staging does **not** free 47.68 GiB while the snapshot copies exist, and
deleting both does.
