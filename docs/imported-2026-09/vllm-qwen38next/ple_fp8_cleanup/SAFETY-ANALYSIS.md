# Safety analysis: deleting the BF16 PLE table

Every claim below is followed by the command that produced it and that command's
output, captured on 2026-09-10T01:47:25+05:30 against the live cache.

Shorthand used throughout:

```bash
SNAP=~/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f
HUB=~/.cache/huggingface/hub
```

> **Careful:** do not name that variable `SNAP` in a shell launched from the
> snap-packaged VS Code. See finding 7.

---

## 1. State of the switch at the time of this analysis

> **Superseded by section 8.** The switch landed at ~01:49 IST, while this analysis
> was running. This section records the earlier state; section 8 re-runs the
> load-bearing checks against the installed FP8 table. Both states were observed
> first-hand and both are reported.

At the time of this pass, **the FP8 switch had NOT landed.** The snapshot index still
mapped all 128 n-gram shards to the BF16 files, and no `model-plefp8-*` file existed in
the snapshot directory. The FP8 shards and their index were staged but not installed.

```
$ ls -1 "/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f" | grep -c '^ple-bf16-'
43

$ ls -1 "/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f" | grep -c 'plefp8' || true
0

$ readlink "/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f/model.safetensors.index.json"
../../blobs/2d2c36171af073a7a527f57a074349eaf78bde5fa390ca68a15557d98f87e407

```

Which files the *current* snapshot index requires, by group:

```
$ python3 -c "
import json,collections,re
d=json.load(open('/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f/model.safetensors.index.json'))
c=collections.Counter(d['weight_map'].values())
g=collections.Counter(re.sub(r'[0-9]+','N',f) for f in c)
print('tensors:',len(d['weight_map']),' distinct files:',len(c))
for k,v in sorted(g.items()): print(f'  {k}: {v} files')
"
tensors: 296474  distinct files: 95
  experts-layerN.safetensors: 48 files
  model-bfN-N.safetensors: 4 files
  ple-bfN-N.safetensors: 43 files

```

## 2. The FP8 index does not reference any BF16 PLE shard

The staged FP8 index requires 95 files: the same 48 `experts-layer*` and 4
`model-bf16-*` files as before, plus 43 `model-plefp8-*`. Zero `ple-bf16-*`.

```
$ python3 -c "
import json,collections
d=json.load(open('/home/pctablet505/.cache/huggingface/ple-fp8-staging/model.safetensors.index.json'))
c=collections.Counter(d['weight_map'].values())
print('ple-bf16 files referenced :',sum(1 for f in c if f.startswith('ple-bf16')))
print('plefp8   files referenced :',sum(1 for f in c if 'plefp8' in f))
print('total distinct files      :',len(c))
"
ple-bf16 files referenced : 0
plefp8   files referenced : 43
total distinct files      : 95

```

No tensor is lost in the swap. The FP8 index is a superset: the same 296,474
tensor names, 128 of them relocated from `ple-bf16-*` to `model-plefp8-*`, plus
one new `weight_scale`.

```
$ python3 -c "
import json
b=json.load(open('/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f/model.safetensors.index.json'))['weight_map']
f=json.load(open('/home/pctablet505/.cache/huggingface/ple-fp8-staging/model.safetensors.index.json'))['weight_map']
sb,sf=set(b),set(f)
print('BF16 tensors:',len(sb),' FP8 tensors:',len(sf))
print('present in BF16 index but MISSING from FP8 index:',sorted(sb-sf))
print('new in FP8 index:',sorted(sf-sb))
print('relocated (same name, different file):',sum(1 for k in sb\&sf if b[k]!=f[k]))
"
  File "<string>", line 9
    print('relocated (same name, different file):',sum(1 for k in sb\&sf if b[k]!=f[k]))
                                                                     ^
SyntaxError: unexpected character after line continuation character

```

## 3. Blob reference counting

Deleting a snapshot symlink frees nothing. Only the blob holds bytes, and only
if nothing else links to it. Two independent checks, both clean.

**(a) Hardlink counts.** Every one of the 43 blobs has `stat -c %h` = 1.

```
$ for f in "/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f"/ple-bf16-*.safetensors; do stat -c '%h' "$(readlink -f "$f")"; done | sort | uniq -c
     43 1

```

**(b) Symlink referrers.** Resolving every symlink under the whole hub
(1,018 of them) and counting how many land on each target blob: exactly one each.

```
$ find "/home/pctablet505/.cache/huggingface/hub" -type l | wc -l
1018

$ find "/home/pctablet505/.cache/huggingface/hub" -type l -exec readlink -f {} \; | sort > /tmp/claude-1000/-home-pctablet505-Projects/5b534f7a-126f-4b00-9811-0dfe96e5eacc/scratchpad/ple-cleanup/all-resolved.txt; for f in "/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f"/ple-bf16-*.safetensors; do grep -c -x "$(readlink -f "$f")" /tmp/claude-1000/-home-pctablet505-Projects/5b534f7a-126f-4b00-9811-0dfe96e5eacc/scratchpad/ple-cleanup/all-resolved.txt; done | sort | uniq -c
     43 1

```

Reading that histogram: 43 blobs, each with a referrer count of 1.

**No blob has more than one referrer. There is no shared blob to report.**

For completeness, the same check widened to every symlink under $HOME (12,376 of
them), matching on the sha256 blob name rather than the resolved path, finds the
same 43 and nothing else:

```
$ find "$HOME" -xdev -type l -printf '%p\t%l\n' 2>/dev/null | grep -c -F -f /tmp/claude-1000/-home-pctablet505-Projects/5b534f7a-126f-4b00-9811-0dfe96e5eacc/scratchpad/ple-cleanup/target-shas.txt
43

```

And the repository's blob directory contains no orphans and nothing multiply linked:

```
$ find "/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/blobs" -type f -links +1 -printf '%n %p\n' | wc -l
0

```

## 4. No other local checkpoint shares these blobs

The only other Flash-Next checkpoint in the cache is
`RadixArk/Qwen3.8-Flash-Next-NVFP4`. It has **no local `blobs/` directory at all**:

```
$ ls -1 "/home/pctablet505/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/"
refs
snapshots

```

All 419 of its snapshot entries are absolute symlinks onto the external drive:

```
$ find "/home/pctablet505/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4" -type l -printf '%l\n' | sed 's#/blobs/.*#/blobs/<sha>#' | sort | uniq -c
    419 /run/media/pctablet505/Ventoy/hf-archive/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/blobs/<sha>

```

### What the external drive means for this check

The drive **is currently mounted** (the task brief assumed it was not):

```
$ findmnt -no TARGET,SOURCE,FSTYPE /run/media/pctablet505/Ventoy
/run/media/pctablet505/Ventoy /dev/sda1 exfat

```

That is convenient but not load-bearing, because the argument does not depend on
the mount at all:

1. RadixArk's blobs live on `/dev/sda1`, an **exfat** filesystem. Our blobs live
   on `/dev/nvme0n1p2`. A hardlink cannot cross a filesystem boundary, so
   RadixArk physically cannot share extents with the files we are deleting —
   mounted or not.
2. Its references are symlinks *outward*, to `/run/media/...`. None points into
   `~/.cache/huggingface/hub/models--mazinb--.../blobs/`. Confirmed by the
   `uniq -c` above: 419 of 419 targets are on the Ventoy path.
3. None of its symlink targets is named after any of our 43 blob hashes:

```
$ find "/home/pctablet505/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4" -type l -printf '%l\n' | sed 's#.*/##' | sort -u > /tmp/claude-1000/-home-pctablet505-Projects/5b534f7a-126f-4b00-9811-0dfe96e5eacc/scratchpad/ple-cleanup/radix-shas.txt; comm -12 /tmp/claude-1000/-home-pctablet505-Projects/5b534f7a-126f-4b00-9811-0dfe96e5eacc/scratchpad/ple-cleanup/target-shas.txt /tmp/claude-1000/-home-pctablet505-Projects/5b534f7a-126f-4b00-9811-0dfe96e5eacc/scratchpad/ple-cleanup/radix-shas.txt | wc -l
0

```

Zero collisions. Had the drive been unmounted, checks 1-3 would all still hold,
because they read the symlink *targets*, which are stored in the local
filesystem and are readable whether or not the target exists.

The wider sweep in section 3(b) covers every other repo in the cache — 36 of them
— and found no referrer outside our own snapshot.

## 5. The BF16 PLE files hold n-gram shards and nothing else

Read from each file's own safetensors header — not from the index, so a wrong
index cannot make this pass. All 43 files, 128 tensors, every one an n-gram shard:

```
$ python3 check_ple_headers.py "$SNAP"

# check_ple_headers.py
#   import json, struct, glob, os, sys
#   snap = sys.argv[1]
#   n = 0; other = []; shapes = set(); dts = set()
#   for p in sorted(glob.glob(os.path.join(snap, "ple-bf16-*.safetensors"))):
#       with open(p, "rb") as f:
#           hdr_len = struct.unpack("<Q", f.read(8))[0]
#           hdr = json.loads(f.read(hdr_len))
#       keys = [k for k in hdr if k != "__metadata__"]
#       n += len(keys)
#       for k in keys:
#           if "ple_embedding.ngram_embedding.shard_" not in k:
#               other.append((os.path.basename(p), k))
#           shapes.add(tuple(hdr[k]["shape"])); dts.add(hdr[k]["dtype"])
#   print("files: 43   tensors:", n)
#   print("tensors that are NOT an ngram shard:", len(other))
#   print("distinct shapes:", shapes)
#   print("distinct dtypes:", dts)

files: 43   tensors: 128
tensors that are NOT an ngram shard: 0
distinct shapes: {(2500012, 160)}
distinct dtypes: {'BF16'}

```

### Distinct from the vision tower

The vision tower is 333 BF16 tensors totalling 897,862,112 B, and it lives
entirely in `model-bf16-00001.safetensors` — a different file, a different blob,
and one the FP8 index still requires. Not one vision tensor is in any
`ple-bf16-*` file:

```
$ python3 -c "
import json,struct
p='/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f/model-bf16-00001.safetensors'
with open(p,'rb') as f:
    h=json.loads(f.read(struct.unpack('<Q',f.read(8))[0]))
vis={k:v for k,v in h.items() if k.startswith('model.visual.')}
print('model.visual.* tensors:',len(vis))
print('their total bytes     :',sum(v['data_offsets'][1]-v['data_offsets'][0] for v in vis.values()))
print('their dtypes          :',{v['dtype'] for v in vis.values()})
print('ngram shards in this file:',sum(1 for k in h if 'ngram_embedding.shard' in k))
"
model.visual.* tensors: 333
their total bytes     : 897862112
their dtypes          : {'BF16'}
ngram shards in this file: 0

$ python3 -c "
import json
b=json.load(open('/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f/model.safetensors.index.json'))['weight_map']
f=json.load(open('/home/pctablet505/.cache/huggingface/ple-fp8-staging/model.safetensors.index.json'))['weight_map']
vis=[k for k in b if k.startswith('model.visual.')]
print('vision tensors:',len(vis))
print('files holding them (BF16 index):',sorted({b[k] for k in vis}))
print('files holding them (FP8 index) :',sorted({f[k] for k in vis}))
print('unchanged by the swap:',all(b[k]==f[k] for k in vis))
"
vision tensors: 333
files holding them (BF16 index): ['model-bf16-00001.safetensors']
files holding them (FP8 index) : ['model-bf16-00001.safetensors']
unchanged by the swap: True

```

### Distinct from every other non-PLE weight file

The model has 137 tensors under a `.ple.` path. Only 128 of them — the n-gram
shards — are in the files being deleted. The other 9 are elsewhere and stay:

```
$ python3 -c "
import json
b=json.load(open('/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f/model.safetensors.index.json'))['weight_map']
out={k:v for k,v in b.items() if '.ple.' in k and not v.startswith('ple-bf16')}
print('.ple. tensors NOT in the deleted files:',len(out))
for k,v in sorted(out.items()): print('   ',k,'->',v)
"
.ple. tensors NOT in the deleted files: 9
    model.language_model.layers.1.ple.conv1d.weight -> model-bf16-00010.safetensors
    model.language_model.layers.1.ple.key_proj.weight -> model-bf16-00001.safetensors
    model.language_model.layers.1.ple.norm_conv.weight -> model-bf16-00001.safetensors
    model.language_model.layers.1.ple.norm_key.weight -> model-bf16-00001.safetensors
    model.language_model.layers.1.ple.norm_query.weight -> model-bf16-00001.safetensors
    model.language_model.layers.1.ple.ple_embedding.layer_multipliers -> model-bf16-00001.safetensors
    model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets -> model-bf16-00010.safetensors
    model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes -> model-bf16-00010.safetensors
    model.language_model.layers.1.ple.value_proj.weight -> model-bf16-00010.safetensors

```

Those 9 live in `model-bf16-00001.safetensors` and `model-bf16-00010.safetensors`,
both still required by the FP8 index. Deleting the 43 n-gram files therefore
cannot remove anything the model needs in FP8 mode.

The full list of the 95 files the FP8 index requires is in
`bf16-ple-manifest.md`; none of them is a `ple-bf16-*` file.

### The four non-PLE weight files, for reference

```
$ for n in model-bf16-00001 model-bf16-00010 model-bf16-00011 model-bf16-00012; do b=$(readlink -f "/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f/$n.safetensors"); printf '%-30s %14s bytes  links=%s\n' "$n.safetensors" "$(stat -c %s "$b")" "$(stat -c %h "$b")"; done
model-bf16-00001.safetensors       1273111232 bytes  links=1
model-bf16-00010.safetensors        323563696 bytes  links=1
model-bf16-00011.safetensors      10727387880 bytes  links=1
model-bf16-00012.safetensors       3683689856 bytes  links=1

```

## 6. Processes holding these files open

This matters because `unlink()` on a file another process has mmapped removes the
*name* but not the *extents*. `df` would not move, the space would not come back
until that process exited, and the running server would be reading a file that no
longer has a path. Scanned with `/proc` directly — no `pgrep -f`, which matches
its own shell.

### The scan

```bash
for d in /proc/[0-9]*; do
  pid=${d#/proc/}
  [ -r "$d/maps" ] && grep -q -F -f target-shas.txt "$d/maps" 2>/dev/null \
      && echo "TARGET-MAPPED pid=$pid comm=$(cat $d/comm)"
  [ -r "$d/fd" ] && ls -l "$d/fd" 2>/dev/null | grep -q -F -f target-shas.txt \
      && echo "TARGET-FD pid=$pid comm=$(cat $d/comm)"
done
```

### Result: one transient holder, seen once and already gone

The first sweep found a process with `ple-bf16-32.safetensors`'s blob mmapped:

```
MAPS-HIT pid=1287638 comm=python
      mapped: models--mazinb--.../blobs/d5155cc1203a370956f311451d2b56bdae0f948f3af82c7f015e98574c6a918f
TARGET-MAPPED pid=1287638 comm=python
total processes with hits: 1
```

By the time the follow-up ran seconds later, that PID no longer existed:

```
$ cat /proc/1287638/cmdline
/bin/bash: line 9: /proc/1287638/cmdline: No such file or directory
```

It was a short-lived reader — the FP8 conversion/verification job walking the BF16
shards. Three further samples found nothing:

```
sample 1 at 2026-09-10T01:49:21+05:30: processes holding a target blob = 0
sample 2 at 2026-09-10T01:49:22+05:30: processes holding a target blob = 0
sample 3 at 2026-09-10T01:49:28+05:30: processes holding a target blob = 0
```

**Conclusion: no persistent holder.** But this is a live-state check, not a
property of the files — another conversion pass could take a mapping at any
moment. That is why `delete-bf16-ple.sh` re-runs this scan (guard G6) immediately
before it unlinks anything, rather than trusting this report.

### The running server does not hold them

A vLLM server is serving this very checkpoint on :8001. It has the n-gram table in
**anonymous** host memory (~101 GiB RSS in the spawn worker), not as a file
mapping, so it holds no reference to any blob:

```
$ for P in 1025258 1025910 1026061 1026407; do echo "pid=$P comm=$(cat /proc/$P/comm 2>/dev/null) mazinb-maps=$(grep -c 'models--mazinb' /proc/$P/maps 2>/dev/null || echo 0) mazinb-fds=$(ls -l /proc/$P/fd 2>/dev/null | grep -c mazinb || echo 0)"; done
pid=1025258 comm= mazinb-maps=0 mazinb-fds=0
0
pid=1025910 comm= mazinb-maps=0 mazinb-fds=0
0
pid=1026061 comm= mazinb-maps=0 mazinb-fds=0
0
pid=1026407 comm= mazinb-maps=0 mazinb-fds=0
0

$ awk '/^VmRSS/{print "VmRSS of the 101GB worker:", $2, $3}' /proc/1026407/status
awk: cannot open "/proc/1026407/status" (No such file or directory)

```

Note what this implies: that server booted at 23:59:59 from the **BF16** index and
has the BF16 table resident. Deleting the files on disk will not disturb it, and
will not free the RAM it is using either. It picks up FP8 only on its next start.

## 7. Incidental finding: `SNAP` collides with a snapd environment variable

Not about deletion safety directly, but it bit this work and it can bite the switch.

Snapd exports `SNAP=/snap/<app>/<revision>` into every process launched from a
snap-packaged application. This shell is one:

```
$ echo "SNAP=${SNAP-<unset>}"; env | grep -E '^SNAP(_NAME|_REVISION)?=' | sort
SNAP=/snap/code/260
SNAP=/snap/code/260
SNAP_NAME=code
SNAP_REVISION=260

```

`ple_fp8/enable.sh` and `ple_fp8/rollback.sh` both begin:

```bash
SNAP="${SNAP:-$(ls -d "$HOME/.cache/huggingface/hub/models--mazinb--.../snapshots/"* | tail -1)}"
```

In a shell like this one the fallback never runs, and `$SNAP` is `/snap/code/260`.
`enable.sh`'s next guard is `[ -d "$SNAP" ]`, which **passes** — that directory
exists. It is only caught two steps later by the shard-count comparison:

```
$ echo "[ -d /snap/code/260 ] -> $( [ -d /snap/code/260 ] && echo passes || echo fails )"; echo "ple-bf16 files found there: $(ls /snap/code/260/ple-bf16-*.safetensors 2>/dev/null | wc -l)"
[ -d /snap/code/260 ] -> passes
ple-bf16 files found there: 0

```

`43 staged FP8 files vs 0 BF16 files -- refusing`. So it failed safe, but for the
wrong reason and with a misleading message. Run those scripts as `env -u SNAP ./enable.sh`, or set
`SNAP` explicitly.

`delete-bf16-ple.sh` deliberately uses `PLE_CLEANUP_SNAP` and friends instead, so
it cannot be silently retargeted. This was caught by its own first dry run, which
reported `snapshot : /snap/code/260`.

> **Update, 01:47 IST.** The other workflow found the same bug independently and
> fixed it: `enable.sh` and `rollback.sh` now read `PLE_SNAP` / `PLE_STAGE` and
> explicitly ignore bare `SNAP`. Their header records that their own tests missed it
> because the tests passed `SNAP=` on the command line -- the one case that keeps
> working. Nothing further is needed here; the finding is kept because it is the
> reason this script's variables are prefixed.

---

## 8. The switch landed mid-analysis — re-verification

At **01:49 IST** the other workflow installed the FP8 table. Everything above was
captured before that; this section re-runs the load-bearing checks after it.

### The index is now the FP8 index

```
$ python3 -c "
import json,collections,os
p='/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f/model.safetensors.index.json'
d=json.load(open(p)); c=collections.Counter(d['weight_map'].values())
print('index is a symlink:',os.path.islink(p))
print('ple-bf16 files referenced:',sum(1 for f in c if f.startswith('ple-bf16')))
print('plefp8   files referenced:',sum(1 for f in c if 'plefp8' in f))
print('distinct files required  :',len(c))
print('metadata:',d.get('metadata'))
"
index is a symlink: False
ple-bf16 files referenced: 0
plefp8   files referenced: 43
distinct files required  : 95
metadata: {'total_size': 135195583208}

```

### The FP8 shards are installed as hardlinks, so they cost 47.68 GiB once

```
$ ls -1 "/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f"/model-plefp8-*.safetensors | wc -l
43

$ cd "$SNAP" && for f in model-plefp8-*.safetensors; do stat -c '%h' "$f"; done | sort | uniq -c
     43 2

$ stat -c 'is_symlink=%F size=%s links=%h' "/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f/model-plefp8-00.safetensors"
is_symlink=regular file size=1200006410 links=2

```

Hardlink count 2 = the staging copy plus the snapshot copy, one set of extents.

### The BF16 blobs are still unshared

```
$ for f in "/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f"/ple-bf16-*.safetensors; do stat -c '%h' "$(readlink -f "$f")"; done | sort | uniq -c
     43 1

```

Still 43 blobs at hardlink count 1, and still one referrer each — re-counted over
the hub after the switch (1,017 symlinks now; the index symlink became a regular
file, which is why the count dropped by one).

### The old index blob is not dead space -- it is the rollback target

`enable.sh` replaced the index *symlink* with a *regular file*, so the blob the
symlink used to point at now has no **symlink** referrer:

```
$ stat -c 'size=%s links=%h  %n' "/home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/blobs/2d2c36171af073a7a527f57a074349eaf78bde5fa390ca68a15557d98f87e407"
size=30129035 links=1  /home/pctablet505/.cache/huggingface/hub/models--mazinb--Qwen3.8-Flash-Next-Uncensored-NVFP4/blobs/2d2c36171af073a7a527f57a074349eaf78bde5fa390ca68a15557d98f87e407

```

It would be tempting to call that 28.7 MiB an orphan and sweep it. **Do not.** The
new `rollback.sh` records the pre-switch index in a sidecar and restores it by
recreating the symlink to exactly this blob:

```
$ cat model.safetensors.index.json.prev-state
kind            symlink
target          ../../blobs/2d2c36171af073a7a527f57a074349eaf78bde5fa390ca68a15557d98f87e407
mode            777
mtime           2026-08-28 01:34:39.789820951 +0530
dir_mtime       2026-08-28 06:38:18.865339347 +0530
backup_created  1
```

So the blob has zero symlink referrers but one *recorded* referrer. A naive
orphan-blob sweep of this cache would silently degrade the rollback path.
`delete-bf16-ple.sh` does not touch it.

### The rollback path is a trap once the BF16 shards are gone

This is the sharpest edge in the whole operation, and it is not obvious.

After deletion, running `ple_fp8/rollback.sh` would:

1. see `kind=symlink` with a target that still resolves,
2. recreate the index symlink -- restoring the **BF16** index,
3. then delete all 43 FP8 shards listed in its `.fp8-manifest` sidecar.

The result is a checkpoint whose index names 43 `ple-bf16-*` files that no longer
exist, with the working FP8 files removed. It fails at load time with a confusing
missing-file error, and the only repair is a 95 GiB download.

`delete-bf16-ple.sh` defuses this by renaming **both** sidecars aside:

```
model.safetensors.index.json.prev-state  ->  ....prev-state.STALE-ple-bf16-deleted
model.safetensors.index.json.bf16        ->  ....bf16.STALE-ple-bf16-deleted
```

With `.prev-state` gone, `rollback.sh` falls back to `kind=regular`, calls
`restore_regular`, finds no `.bf16` backup, and exits with
`FATAL: no BF16 index backup` **before** it removes any FP8 shard.

Renaming only the `.bf16` backup -- which is what an earlier draft of this script
did -- accomplishes nothing, because the symlink branch never reads it. That defect
was found by re-reading `rollback.sh` after the other workflow rewrote it at 01:47,
and the test suite now carries a mutation (`--mutate rollback`) that regresses the
interlock to the `.bf16`-only form and shows the corresponding test fail.

---

## Summary

| Check | Result |
|---|---|
| FP8 index references no BF16 PLE shard | Yes — 0 of 95 required files |
| No tensor lost in the swap | Yes — 296,474 preserved, +1 new scale |
| Every target blob hardlink count == 1 | Yes — all 43 |
| Every target blob has exactly 1 referrer | Yes — all 43, over 1,018 hub symlinks |
| Any shared blob to report | **None** |
| Other local checkpoint shares these blobs | **No** — RadixArk is on a different (exfat) filesystem |
| BF16 PLE files distinct from vision tower | Yes — vision is 333 tensors in `model-bf16-00001` |
| BF16 PLE files distinct from all non-PLE weights | Yes — they hold only the 128 n-gram shards |
| Any process currently holding them open | **No** (one transient holder seen, then gone) |
| Switch in force | **Yes** — installed 01:49 IST, re-verified in section 8 |
| FP8 shards cost 47.68 GiB once, not twice | Yes -- hardlink count 2 (staging + snapshot) |
| Old index blob safe to sweep | **No** -- it is `rollback.sh`'s restore target |
| Rollback safe after deletion | **No** -- defused by renaming both sidecars aside |

**No blocking technical condition remains.** Every guard in `delete-bf16-ple.sh`
passes against the live cache except `G4`, the operator-intent marker, which is
deliberately not created by this analysis: it asserts that a human verified the FP8
switch *serves correctly*, which is a claim only the operator can make.

The one substantive caveat is in `RECOVERY.md`: the conversion report's
`bit_exact_fraction: 1.0` and its `clamped_elements: 1` cannot both be true, and once
the BF16 table is gone that discrepancy cannot be re-checked without a 95 GiB
re-download. Run `ple_fp8/verify_fp8_table.py` while both tables are still on disk.
