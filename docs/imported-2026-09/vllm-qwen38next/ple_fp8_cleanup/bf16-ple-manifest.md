# BF16 PLE n-gram table — deletion manifest

Generated (UTC): 2026-09-09T20:21:03.864452+00:00  
Repo: `mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4`  
Revision: `f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f`  
Snapshot: `/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f`

**Nothing has been deleted.** This file is an inventory produced by read-only reconnaissance.

## The one thing to understand about space

> Deleting a snapshot **symlink** frees **zero bytes**.
>
> Every `ple-bf16-NN.safetensors` entry in the snapshot directory is a symlink into
> `../../blobs/<sha256>`. The 95 GiB lives in the blob. Space comes back only when the
> **blob's last link** is unlinked *and* no live process still holds the file open.
> A blob whose `stat -c %h` is greater than 1, or which more than one snapshot symlink
> resolves to, must not be removed — another checkpoint is using the same bytes.
>
> If a process has the blob **mmapped**, `unlink()` removes the name but the extents survive
> until that process exits. `df` will not move, and the server keeps reading phantom bytes.

## Totals

| | |
|---|---|
| BF16 PLE files | **43** |
| Tensors (n-gram shards) | **128** — every one `…ngram_embedding.shard_N.weight`, BF16, shape `[2500012, 160]` |
| Total bytes | **102,400,512,064** |
| Total | **95.3679 GiB** (102.401 GB) |
| Distinct blobs | 43 |
| Max hardlink count over all blobs | 1 |
| Max snapshot referrers over all blobs | 1 |

## FP8 replacement — INSTALLED

| | |
|---|---|
| Files | 43 × `model-plefp8-NN.safetensors` |
| Total bytes | 51,200,268,090 |
| Total | 47.684 GiB |
| Installed as | regular files, hardlink count 2 (staging + snapshot) |
| Net reclaim if BF16 removed | **95.3679 GiB** |

ple_fp8/enable.sh hardlinked the staging files DIRECTLY into the snapshot dir as regular files (NOT blobs+symlinks) and overwrote model.safetensors.index.json with a regular file. Observed installed hardlink_count is 2 (staging + snapshot), so the FP8 table occupies 47.684 GiB once, not twice.

Because the snapshot copies are hardlinks to the staging copies, the FP8 table
occupies its 47.684 GiB **once**, not twice.

## Index state

- Path: `/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f/model.safetensors.index.json`
- Is a symlink: `False`  (a regular file, as `enable.sh` leaves it)
- BF16 PLE files referenced: **0**
- FP8 PLE files referenced: **43**
- `metadata.total_size`: {'total_size': 135195583208}

**FP8 IS IN FORCE - the index maps all 128 n-gram shards to model-plefp8-* and references no ple-bf16-* file.**

> An earlier pass of this analysis, at 2026-09-10T01:39 IST, found the BF16 index still in force. The switch landed at ~01:49 IST while the analysis was running. Both states were observed; this record is the later one.

## Incidental: one orphan blob left behind by the switch

- `/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/blobs/2d2c36171af073a7a527f57a074349eaf78bde5fa390ca68a15557d98f87e407`
- 30,129,035 bytes (28.7 MiB), 0 referrers
- the old BF16 model.safetensors.index.json blob. enable.sh replaced the snapshot symlink with a regular file, orphaning this blob. 28.7 MiB, harmless, NOT removed by delete-bf16-ple.sh -- flagged for the operator.

## File inventory

`SIZE` is the blob's size in bytes. `LNK` is `stat -c %h` on the blob. `REF` is the number of
symlinks anywhere under `~/.cache/huggingface/hub` that resolve to that blob.

| # | Snapshot name | Symlink target | Blob sha256 | Size (B) | LNK | REF | Tensors |
|---:|---|---|---|---:|---:|---:|---:|
| 0 | `ple-bf16-00.safetensors` | `../../blobs/36bc5eef3691812ef9b7868019b2c4493fe5a8797d7a778bea93ce6a93039889` | `36bc5eef3691812ef9b7868019b2c4493fe5a8797d7a778bea93ce6a93039889` | 2,400,012,000 | 1 | 1 | 3 |
| 1 | `ple-bf16-01.safetensors` | `../../blobs/d89fd9d7ae5d03a355611b5c8cce54e825d719765cdb0b7bae88dd4ce4cbd9e9` | `d89fd9d7ae5d03a355611b5c8cce54e825d719765cdb0b7bae88dd4ce4cbd9e9` | 2,400,012,000 | 1 | 1 | 3 |
| 2 | `ple-bf16-02.safetensors` | `../../blobs/fbe16136ac35c6eeedca68da6902fcf94f945a789f1fecb3675f9536b6236ba9` | `fbe16136ac35c6eeedca68da6902fcf94f945a789f1fecb3675f9536b6236ba9` | 2,400,012,000 | 1 | 1 | 3 |
| 3 | `ple-bf16-03.safetensors` | `../../blobs/32e075d76dffcdb29dd47c3f89622dc7f6c5941815e89494fbaf88659eec10af` | `32e075d76dffcdb29dd47c3f89622dc7f6c5941815e89494fbaf88659eec10af` | 2,400,012,000 | 1 | 1 | 3 |
| 4 | `ple-bf16-04.safetensors` | `../../blobs/92e349d0c32400d8f0a966c91ddfd39c58bd3cc052447b771e295560bf044ea3` | `92e349d0c32400d8f0a966c91ddfd39c58bd3cc052447b771e295560bf044ea3` | 2,400,012,000 | 1 | 1 | 3 |
| 5 | `ple-bf16-05.safetensors` | `../../blobs/fd4d43370eee0332bd78694da5ca33f441c222777d879d1cebbd306a17ff89df` | `fd4d43370eee0332bd78694da5ca33f441c222777d879d1cebbd306a17ff89df` | 2,400,012,000 | 1 | 1 | 3 |
| 6 | `ple-bf16-06.safetensors` | `../../blobs/078439aec94e4782463ce7b48233d87fa9dfccd2014cf19847dcb4c7c6112286` | `078439aec94e4782463ce7b48233d87fa9dfccd2014cf19847dcb4c7c6112286` | 2,400,012,000 | 1 | 1 | 3 |
| 7 | `ple-bf16-07.safetensors` | `../../blobs/04527a968343e7a886582ff4feaa3b9b1b55aba01a2619c4a1cb468857d3504e` | `04527a968343e7a886582ff4feaa3b9b1b55aba01a2619c4a1cb468857d3504e` | 2,400,012,000 | 1 | 1 | 3 |
| 8 | `ple-bf16-08.safetensors` | `../../blobs/60d254f38f40c93d9b2d1f9e8314bc86297fdbcc9503a46a64551262ac68b7aa` | `60d254f38f40c93d9b2d1f9e8314bc86297fdbcc9503a46a64551262ac68b7aa` | 2,400,012,000 | 1 | 1 | 3 |
| 9 | `ple-bf16-09.safetensors` | `../../blobs/674370eb446aa1ff9135d49fa9c07bd0993cb5c7760b74d44482e145970fb96e` | `674370eb446aa1ff9135d49fa9c07bd0993cb5c7760b74d44482e145970fb96e` | 2,400,012,000 | 1 | 1 | 3 |
| 10 | `ple-bf16-10.safetensors` | `../../blobs/4cefd07d846e1258d38b5df205509a9d0aa20423a32566a990aa4227ba9aa561` | `4cefd07d846e1258d38b5df205509a9d0aa20423a32566a990aa4227ba9aa561` | 2,400,012,000 | 1 | 1 | 3 |
| 11 | `ple-bf16-11.safetensors` | `../../blobs/853709d10c5ba27714ab4bf461a932413e87a9ac4392a3b1e2d02f671ba5f36a` | `853709d10c5ba27714ab4bf461a932413e87a9ac4392a3b1e2d02f671ba5f36a` | 2,400,012,000 | 1 | 1 | 3 |
| 12 | `ple-bf16-12.safetensors` | `../../blobs/cba1b1101cf8aae40f13a808d6f9ca50acbe23925e8b899e71c4063b722d291e` | `cba1b1101cf8aae40f13a808d6f9ca50acbe23925e8b899e71c4063b722d291e` | 2,400,012,000 | 1 | 1 | 3 |
| 13 | `ple-bf16-13.safetensors` | `../../blobs/95402ae538d6f130ed64feebc940185485a8106f0ce57a44d005b2600dd9661b` | `95402ae538d6f130ed64feebc940185485a8106f0ce57a44d005b2600dd9661b` | 2,400,012,000 | 1 | 1 | 3 |
| 14 | `ple-bf16-14.safetensors` | `../../blobs/a1e68c50b06d22351499f2207bf6e512fc894564df4f0ee649bd4a2b19188375` | `a1e68c50b06d22351499f2207bf6e512fc894564df4f0ee649bd4a2b19188375` | 2,400,012,000 | 1 | 1 | 3 |
| 15 | `ple-bf16-15.safetensors` | `../../blobs/9b8fb40c0fc5ccc8b1f77e0236e435a7d86509ec48bd236e4c58b20e328c488a` | `9b8fb40c0fc5ccc8b1f77e0236e435a7d86509ec48bd236e4c58b20e328c488a` | 2,400,012,000 | 1 | 1 | 3 |
| 16 | `ple-bf16-16.safetensors` | `../../blobs/88bde76c6a2fd63d62ed184d299e074c50b1ef9318e2140432faa03a58c3696c` | `88bde76c6a2fd63d62ed184d299e074c50b1ef9318e2140432faa03a58c3696c` | 2,400,012,000 | 1 | 1 | 3 |
| 17 | `ple-bf16-17.safetensors` | `../../blobs/42f39e7f4cdbe6a47f66bb7a0fb4a0660985866ea0fcef39189499f0a308670b` | `42f39e7f4cdbe6a47f66bb7a0fb4a0660985866ea0fcef39189499f0a308670b` | 2,400,012,000 | 1 | 1 | 3 |
| 18 | `ple-bf16-18.safetensors` | `../../blobs/9719cacf7b248a9afc6699cc5521db36ae3760b613e5d7e67e76f3e30a8cb966` | `9719cacf7b248a9afc6699cc5521db36ae3760b613e5d7e67e76f3e30a8cb966` | 2,400,012,000 | 1 | 1 | 3 |
| 19 | `ple-bf16-19.safetensors` | `../../blobs/8fb678cd10e3377f3d0d14f77a1660d907990878906e18d38fd28225c24f61ff` | `8fb678cd10e3377f3d0d14f77a1660d907990878906e18d38fd28225c24f61ff` | 2,400,012,000 | 1 | 1 | 3 |
| 20 | `ple-bf16-20.safetensors` | `../../blobs/6624c4a8fe18ba3f43cc5c3c9f13441f93f099b0836700ad5360167691b3b19a` | `6624c4a8fe18ba3f43cc5c3c9f13441f93f099b0836700ad5360167691b3b19a` | 2,400,012,000 | 1 | 1 | 3 |
| 21 | `ple-bf16-21.safetensors` | `../../blobs/2fc9806c94b0cc1315c6439ad385e64ea94604718f0e5fae163163f4b015548d` | `2fc9806c94b0cc1315c6439ad385e64ea94604718f0e5fae163163f4b015548d` | 2,400,012,000 | 1 | 1 | 3 |
| 22 | `ple-bf16-22.safetensors` | `../../blobs/5f6f336593d5047f18aa8c35b265e5f387d8c820d6841d10b682f1b636b58282` | `5f6f336593d5047f18aa8c35b265e5f387d8c820d6841d10b682f1b636b58282` | 2,400,012,000 | 1 | 1 | 3 |
| 23 | `ple-bf16-23.safetensors` | `../../blobs/a64ae69728d45d520a0c81e23f4c597becb7910e90e691d4f3dadbc0db5580ca` | `a64ae69728d45d520a0c81e23f4c597becb7910e90e691d4f3dadbc0db5580ca` | 2,400,012,000 | 1 | 1 | 3 |
| 24 | `ple-bf16-24.safetensors` | `../../blobs/11539f10957639850e666176c941abe757ce708e915f39f6a2b1f3be6b0d5f11` | `11539f10957639850e666176c941abe757ce708e915f39f6a2b1f3be6b0d5f11` | 2,400,012,000 | 1 | 1 | 3 |
| 25 | `ple-bf16-25.safetensors` | `../../blobs/fa57c7aae696bae8a7c4af660933adf3ee40b589d5b1564ebca4b91b3d5cdc0d` | `fa57c7aae696bae8a7c4af660933adf3ee40b589d5b1564ebca4b91b3d5cdc0d` | 2,400,012,000 | 1 | 1 | 3 |
| 26 | `ple-bf16-26.safetensors` | `../../blobs/e06755c4775c7c42804a5167950fe0dee41da019c9ce53ec3f879a6bd392cf65` | `e06755c4775c7c42804a5167950fe0dee41da019c9ce53ec3f879a6bd392cf65` | 2,400,012,000 | 1 | 1 | 3 |
| 27 | `ple-bf16-27.safetensors` | `../../blobs/89b96a77b3e173f282acecf301c32bf63ad9d2fd3ae520efc8c990f2286e7739` | `89b96a77b3e173f282acecf301c32bf63ad9d2fd3ae520efc8c990f2286e7739` | 2,400,012,000 | 1 | 1 | 3 |
| 28 | `ple-bf16-28.safetensors` | `../../blobs/608b1cbd485f8b5d6ec000c087a51e11b541a02364843fbfa395c0ecffd5d34d` | `608b1cbd485f8b5d6ec000c087a51e11b541a02364843fbfa395c0ecffd5d34d` | 2,400,012,000 | 1 | 1 | 3 |
| 29 | `ple-bf16-29.safetensors` | `../../blobs/9abaa42b378518897caf7d12d98124c90d974216a9e053516edb30ec2a53f870` | `9abaa42b378518897caf7d12d98124c90d974216a9e053516edb30ec2a53f870` | 2,400,012,000 | 1 | 1 | 3 |
| 30 | `ple-bf16-30.safetensors` | `../../blobs/15fc3936a39ec9f1bf49418c98fcd59cadded22fd9a2fbc8500182d60b27b65e` | `15fc3936a39ec9f1bf49418c98fcd59cadded22fd9a2fbc8500182d60b27b65e` | 2,400,012,000 | 1 | 1 | 3 |
| 31 | `ple-bf16-31.safetensors` | `../../blobs/67e98ed4610112941e19393db9840233024ef80a491f721248da76d6bd0c0608` | `67e98ed4610112941e19393db9840233024ef80a491f721248da76d6bd0c0608` | 2,400,012,000 | 1 | 1 | 3 |
| 32 | `ple-bf16-32.safetensors` | `../../blobs/d5155cc1203a370956f311451d2b56bdae0f948f3af82c7f015e98574c6a918f` | `d5155cc1203a370956f311451d2b56bdae0f948f3af82c7f015e98574c6a918f` | 2,400,012,000 | 1 | 1 | 3 |
| 33 | `ple-bf16-33.safetensors` | `../../blobs/20502b0f5684c2f002e3038ef56796cc862d48e8265bd956cc99dc259c177e17` | `20502b0f5684c2f002e3038ef56796cc862d48e8265bd956cc99dc259c177e17` | 2,400,012,000 | 1 | 1 | 3 |
| 34 | `ple-bf16-34.safetensors` | `../../blobs/40fd74bd770fc80fc822a3665ac42aa6d113fdb6b13bd97614ae92bbe7289d8a` | `40fd74bd770fc80fc822a3665ac42aa6d113fdb6b13bd97614ae92bbe7289d8a` | 2,400,012,008 | 1 | 1 | 3 |
| 35 | `ple-bf16-35.safetensors` | `../../blobs/7799ba456f375f9ab6e9679d117fcb344c2fd5746750ef8f9835680f04c74e21` | `7799ba456f375f9ab6e9679d117fcb344c2fd5746750ef8f9835680f04c74e21` | 2,400,012,008 | 1 | 1 | 3 |
| 36 | `ple-bf16-36.safetensors` | `../../blobs/584163fa9d0fb032a4f9cdf694846e6a5b2b03f175059a467117a53f015adb0b` | `584163fa9d0fb032a4f9cdf694846e6a5b2b03f175059a467117a53f015adb0b` | 2,400,012,008 | 1 | 1 | 3 |
| 37 | `ple-bf16-37.safetensors` | `../../blobs/b0234869d6fd9c3e6ad76b3d498811f077027b45552d89522c816ce8b21b44ea` | `b0234869d6fd9c3e6ad76b3d498811f077027b45552d89522c816ce8b21b44ea` | 2,400,012,008 | 1 | 1 | 3 |
| 38 | `ple-bf16-38.safetensors` | `../../blobs/4f62d9311f2fb142432144bdb37af426e6eac58cb7c835f8f43761fcfad8824b` | `4f62d9311f2fb142432144bdb37af426e6eac58cb7c835f8f43761fcfad8824b` | 2,400,012,008 | 1 | 1 | 3 |
| 39 | `ple-bf16-39.safetensors` | `../../blobs/105dd462aa99603cda36e4e64077af5fec9b7c7c82b92866d239077c464642f8` | `105dd462aa99603cda36e4e64077af5fec9b7c7c82b92866d239077c464642f8` | 2,400,012,008 | 1 | 1 | 3 |
| 40 | `ple-bf16-40.safetensors` | `../../blobs/60748b9b42623a0956b22561f83a8f635035d63728ac741e2a95e579e466ad0f` | `60748b9b42623a0956b22561f83a8f635035d63728ac741e2a95e579e466ad0f` | 2,400,012,008 | 1 | 1 | 3 |
| 41 | `ple-bf16-41.safetensors` | `../../blobs/6bf5769d4b2dce0203f112fe66a173ef383cd4adbc53fb7a85b67525578e9f7b` | `6bf5769d4b2dce0203f112fe66a173ef383cd4adbc53fb7a85b67525578e9f7b` | 2,400,012,008 | 1 | 1 | 3 |
| 42 | `ple-bf16-42.safetensors` | `../../blobs/a59f5a108d449bf9768a65fbfbb00cef668d05c52626ae0c084f058b2da9bc4f` | `a59f5a108d449bf9768a65fbfbb00cef668d05c52626ae0c084f058b2da9bc4f` | 1,600,008,000 | 1 | 1 | 2 |
| | **TOTAL** | | | **102,400,512,064** | | | **128** |

All blob paths are `/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/blobs/<sha256>`.

## Tensor coverage proof

Read from each file's safetensors header (not from the index), all 43 files:

- 128 tensors total, **zero** non-`ngram_embedding.shard_N.weight` tensors.
- All shapes `(2500012, 160)`, all dtype `BF16`.
- Therefore these files contain the n-gram table and nothing else.

The other 9 `.ple.` tensors of the model (`conv1d`, `key_proj`, `value_proj`, `norm_conv`,
`norm_key`, `norm_query`, `ple_embedding.layer_multipliers`, `ngram_heads_offsets`,
`ngram_heads_vocab_sizes`) live in `model-bf16-00001.safetensors` and
`model-bf16-00010.safetensors`, which are **not** in this manifest and are still required by
the FP8 index.

## Files required by the installed FP8 index (all 95)

None of them is a `ple-bf16-*` file.

```
experts-layer00.safetensors
experts-layer01.safetensors
experts-layer02.safetensors
experts-layer03.safetensors
experts-layer04.safetensors
experts-layer05.safetensors
experts-layer06.safetensors
experts-layer07.safetensors
experts-layer08.safetensors
experts-layer09.safetensors
experts-layer10.safetensors
experts-layer11.safetensors
experts-layer12.safetensors
experts-layer13.safetensors
experts-layer14.safetensors
experts-layer15.safetensors
experts-layer16.safetensors
experts-layer17.safetensors
experts-layer18.safetensors
experts-layer19.safetensors
experts-layer20.safetensors
experts-layer21.safetensors
experts-layer22.safetensors
experts-layer23.safetensors
experts-layer24.safetensors
experts-layer25.safetensors
experts-layer26.safetensors
experts-layer27.safetensors
experts-layer28.safetensors
experts-layer29.safetensors
experts-layer30.safetensors
experts-layer31.safetensors
experts-layer32.safetensors
experts-layer33.safetensors
experts-layer34.safetensors
experts-layer35.safetensors
experts-layer36.safetensors
experts-layer37.safetensors
experts-layer38.safetensors
experts-layer39.safetensors
experts-layer40.safetensors
experts-layer41.safetensors
experts-layer42.safetensors
experts-layer43.safetensors
experts-layer44.safetensors
experts-layer45.safetensors
experts-layer46.safetensors
experts-layer47.safetensors
model-bf16-00001.safetensors
model-bf16-00010.safetensors
model-bf16-00011.safetensors
model-bf16-00012.safetensors
model-plefp8-00.safetensors
model-plefp8-01.safetensors
model-plefp8-02.safetensors
model-plefp8-03.safetensors
model-plefp8-04.safetensors
model-plefp8-05.safetensors
model-plefp8-06.safetensors
model-plefp8-07.safetensors
model-plefp8-08.safetensors
model-plefp8-09.safetensors
model-plefp8-10.safetensors
model-plefp8-11.safetensors
model-plefp8-12.safetensors
model-plefp8-13.safetensors
model-plefp8-14.safetensors
model-plefp8-15.safetensors
model-plefp8-16.safetensors
model-plefp8-17.safetensors
model-plefp8-18.safetensors
model-plefp8-19.safetensors
model-plefp8-20.safetensors
model-plefp8-21.safetensors
model-plefp8-22.safetensors
model-plefp8-23.safetensors
model-plefp8-24.safetensors
model-plefp8-25.safetensors
model-plefp8-26.safetensors
model-plefp8-27.safetensors
model-plefp8-28.safetensors
model-plefp8-29.safetensors
model-plefp8-30.safetensors
model-plefp8-31.safetensors
model-plefp8-32.safetensors
model-plefp8-33.safetensors
model-plefp8-34.safetensors
model-plefp8-35.safetensors
model-plefp8-36.safetensors
model-plefp8-37.safetensors
model-plefp8-38.safetensors
model-plefp8-39.safetensors
model-plefp8-40.safetensors
model-plefp8-41.safetensors
model-plefp8-42.safetensors
```
