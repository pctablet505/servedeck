# The host

Read live 2026-09-18 01:13 IST, uptime 2 h 37 m, Flash-Next serving; re-read rather than trust it.

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB, compute capability 12.0 (SM120) |
| Driver | 595.84 (`nvidia-driver-595-open` 595.84-0ubuntu0.26.04.1), CUDA 13.2 driver-side |
| Power cap | 375 W live; card range 150-600 W, factory default 600 W |
| Host RAM | 182 GiB, **no swap device**; `/dev/shm` and `/tmp` are each a 92 G tmpfs |
| Disk | `/dev/nvme0n1p2` 915 G, 746 G used, 123 G free (86%) |
| Board, kernel | Gigabyte X870E AORUS ELITE X3D, 32 cores, Ubuntu, kernel 7.0.0-31-generic |

## GPU

One card, shared by everything: at 01:13 on 2026-09-18, 94,435 of 97,887 MiB were in use by the
one serving model. `models.toml [gpu]` records `total_mib = 97887` and `margin_mib = 1024`, and
that margin is never handed to a model. `gnome-shell` is on this GPU as a type `G` process
(8 MiB), so the card is not headless — `nvidia-smi --query-compute-apps` never lists a graphics
process, which is where the "no display attached to this card" line in
`vllm-qwen38next/SETUP.md:260` came from; it is wrong. The cost is single-digit MiB, but a
utilisation value with zero slack has the compositor inside it.

SM120 is the expensive part: kernels the ecosystem ships for SM90 either fail to select or select
and then assert here — see [models/glm53.md](models/glm53.md) for the attention-backend case and
[BUILDS.md](BUILDS.md) for what had to be compiled. There is **no system CUDA toolkit**: no
`nvcc` on `PATH`, no `/usr/local/cuda`. `ptxas` comes from pip wheels, which is why every
launcher exports `CUDA_HOME` into its own venv's `nvidia/cu13` and why
`VLLM_USE_FLASHINFER_SAMPLER=0` is in `[defaults.env]` — FlashInfer's JIT sampler cannot compile
without one.

**Driver upgrades are a hazard, and they are unattended.** `APT::Periodic::Unattended-Upgrade
"1"` is set and no NVIDIA package is held (`apt-mark showhold` is empty), while three serving
venvs are built against this driver's stack, all on torch 2.13.0+cu130: `.venv-llm-029` (vLLM
0.29.0, stock wheel), `.venv-next` and `.venv-glm53` (editable forks). A driver bump replaces the
kernel module under running engines and can move the ABI away from the cu13 wheels those forks
were compiled against; recovering means a rebuild, not a restart. Check the driver after an
upgrade window before blaming servedeck for a model that stops booting.

## The power cap — an open decision

`/etc/systemd/system/nvidia-power-limit.service` (oneshot, `RemainAfterExit=yes`, after and
requiring `nvidia-persistenced.service`, enabled and active) runs `nvidia-smi -pl 375` at every
boot, so a cap set by hand does not survive a reboot. Nothing in `models.toml` pins an expected
value, so `servedeck doctor`'s power-cap row reports the live number and says it has nothing to
compare against. Older notes claim this unit sets 485 W and other docs say 450 W or 500 W — the
unit file says 375 W and the live cap is 375 W.

### The card has fallen off the bus three times

| When | Xids | Cap in force | What was running |
|---|---|---|---|
| 2026-08-24 20:12:26 | 79, then 154 | not recorded | `qwen-vllm.service` (27B NVFP4), 22 min into its run |
| 2026-09-11 21:24:41 | 79, then 154 | 600 W, raised from 550 W 3 min 52 s earlier | 27B in a heavy prefill burst: 16 running, 1 waiting, prompt 9,228 tok/s, KV 73-83% |
| 2026-09-12 14:25:44 | 79 (`pid=nvidia-smi`), then 154 | 450 W, unchanged for 90 min | 27B at ordinary load, ~1,000 tok/s prompt, KV 29% |

Xid 79 is "GPU has fallen off the bus", Xid 154 is "recovery action changed ... to 0x2 (Node
Reboot Required)". Each time `nvidia-smi` reported "No devices were found", the rest of the
machine stayed up, only a reboot brought the card back, and no PCIe AER or thermal message
preceded it.

**The "600 W did it, 485 W is the validated stable cap" conclusion is not supported.** It was
written after the 2026-09-11 event alone, where a cap change four minutes earlier made a power
transition the obvious suspect. The other two kill it: 2026-09-12 happened at a cap that had not
moved in 90 minutes, under light load, with the fault attributed to a plain `nvidia-smi` query,
and 2026-08-24 happened while the 27B merely served. The owner-supplied external report
(dbirks/home-k8s#46) describes the same card class (GB202) and the same Xid 79 + 154 pair,
reproduced across motherboards and drivers 570-595, as a confirmed NVIDIA GSP firmware/driver bug
with affected cards treated as RMA candidates. On that reading a lower cap lowers the odds and
does not remove them.

The cap is therefore **an open owner decision**. Mitigations named in that report are not applied:
`NVreg_DynamicPowerManagement=0x00` is unset and persistence mode is Disabled (Ubuntu runs
`nvidia-persistenced --no-persistence-mode`). What a cap costs is measured below; what it buys in
fault rate is unknown.

### What a cap costs, measured

Both sweeps, 2026-09-11: idle box, one frozen 39,760-token prompt with a per-cap cache-busting
salt (so every cold prefill misses the whole prefix chain), 45 s decode windows timed from the
first token, power sampled every 0.5 s, `idle_warn=0` on every row kept. Raw CSVs in
`docs/imported-2026-09/local_llm/power-cap-exp/`; the protocol handoff beside them contains a
credential, and a credential was redacted from this source.

27B = `Qwen3.8-27B-NVFP4`, vLLM 0.29.0 stock wheel, util 0.95, ctx 262,144, `--max-num-seqs 16`,
port 8004, batched at 16. FN = Flash-Next on port 8001, batched at 6.

| Model | Cap (W) | Cold prefill TTFT | Prefill tok/s | Decode-1 | Batched decode | Temp max |
|---|---|---|---|---|---|---|
| 27B | 585 | 4.48 s | 8,875 | 140 | 1,467 | 86 C |
| 27B | 502 | 5.06 s | 7,858 | 145 | 1,352 | 87 C |
| 27B | 388 | 6.01 s | 6,616 | 136 | 1,221 | 80 C |
| 27B | 300 | 7.44 s | 5,344 | 141 | 1,024 | 72 C |
| FN | 585 | 2.78 s | 14,302 | 134 | 538 | 81 C |
| FN | 474 | 3.19 s | 12,464 | 136 | 553 | 83 C |
| FN | 384 | 3.71 s | 10,717 | 146 | 523 | 77 C |
| FN | 227 | 6.03 s | 6,594 | 106 | 368 | 64 C |

**Batched decode and prefill pay; single-stream decode does not.** 27B decode-1 stays in
124-145 tok/s from 300 W to 585 W and Flash-Next decode-1 in 106-146 tok/s from 227 W to 585 W,
inside run-to-run scatter. What costs power is concurrency and prefill.

**Neither 375 W nor 485 W was measured directly.** Interpolating the bracketing rows, 375 W
against 485 W costs the 27B about 12% of batched decode (~1,187 vs ~1,356 tok/s) and about 20%
more cold-prefill time (~6.2 s vs ~5.2 s), and costs Flash-Next about 7% of batched decode (~516
vs ~552 tok/s) and about 20% more cold prefill (~3.8 s vs ~3.2 s). Against 585 W the 27B figures
are about -19% and +38%. Derived, not measured.

**A cap is a ceiling, not a target.** At 585 W the 27B's batched decode drew 571.8 W average, so
that cap barely binds; Flash-Next drew only 484.7 W average there and its batched decode is
flat-to-non-monotonic from 474 W up (553, 552, 538 tok/s at 474/526/585 W), so above roughly
530 W there is nothing left to buy for it. Neither sweep is complete: the 27B run was interrupted
at 285 W (285-200 W never collected) and Flash-Next never got 204 W or 200 W.

### Measuring power at all

A Corsair HX1200i exposes USB-HID telemetry at `/sys/class/hwmon/hwmon3` (`corsairpsu`, still
present 2026-09-18), world-readable, ~2 ms per read: whole-system power needs no root and no
external meter. `power1` is **DC output**, not AC input — true AC draw is 8-12% higher. DC power
has sd ≈ 50 W with a ~16 s autocorrelation time, so the smallest provable difference is ~22 W
over 10 min and ~9 W over 60 min: **any power claim below ~10 W is unfalsifiable here.** Under
sustained load the card is simultaneously power- and thermally-capped (598.76 W, 91 C,
`clocks_event_reasons.sw_power_cap = Active`), so it has no headroom to surrender. (Measured
2026-08-17, not re-measured.) These figures are why a context-aware power daemon was rejected in
favour of a static cap — see [DECISIONS.md](DECISIONS.md).

## Xid history and what the codes mean

`journalctl -k` implies `--boot=0` and this box reboots often: 0 Xid lines on the current boot,
790 across `journalctl -k -b all` (2026-09-18), so always pass `-b all`. `dmesg` is unreadable
(`kernel.dmesg_restrict = 1`). `servedeck/xid_watch.py` passes `-b all` for exactly this reason —
a fatal Xid is often the event that caused the reboot that hid it.

| Code | Occurrences here | Action |
|---|---|---|
| 13, SM warp exception | 752 lines in one storm, 2026-08-24 23:03 and 23:45 | may recover on restart |
| 31, MMU fault | ~20 between 2026-08-15 and 2026-09-11, 10 of them on 2026-09-02 | may recover on restart |
| 79 + 154 | 3 events: 2026-08-24, 2026-09-11, 2026-09-12 | **reboot; nothing else recovers it** |
| 109, CTX SWITCH TIMEOUT | 1, 2026-08-29 14:19, followed instantly by an Xid 31 | treat as 31 |

Anything not in that table is "do not assume", and **if `nvidia-smi -L` fails at all, never
restart a model** — the card is gone and a restart only produces a confusing second failure.
`servedeck/gpu.py:264` carries this classification in code.

Two Xid 31 causes must not be conflated: GLM-5.3's MMU faults reproduce under an agent loop in
the SM90 sparse-MLA path and are unresolved ([models/glm53.md](models/glm53.md)), while the 27B's
appear as Xid 31 or a segfault in `CUDAGraph::replay` and are the upstream vLLM GDN-Mamba race
([models/qwen27b.md](models/qwen27b.md)). Neither is related to the 79/154 events, and a decision
made on the wrong family will be wrong.

## Host RAM, /dev/shm and /tmp

`swapon --show` prints nothing. With no swap a host-RAM spike does not slow down, it reaches the
OOM killer. The structural commitments while Flash-Next serves are its FP8 PLE table (~47.7 GiB
resident) and the 40 GiB pinned KV-offload buffer in `/dev/shm`; `free -g` read 182 total / 108
used / 40 shared / 74 available on 2026-09-18, but run it rather than trusting that — it moves by
tens of GiB with what is loaded. `start()` preflights host RAM against `models.toml host_ram_gib`
(flashnext 95, glm53 155) and refuses rather than letting the kernel arbitrate; the reason is
2026-08-27, when a second model load OOM-killed a 103 GiB process and throughput fell from 93 to
2.8 tok/s while thrashing (source-only, not re-measured). Never start a second big model.

`/dev/shm` is a 92 G tmpfs and `/tmp` is a *separate* 92 G tmpfs; both are host RAM and both
sizes are ceilings, not reservations. `/dev/shm` currently holds `vllm_offload_<uuid>.mmap`,
42,945,576,960 bytes (40.0 GiB) — vLLM's CPU KV-offload region, sized by `--kv-offloading-size`,
and the `shared` column in `free -g`. So a multi-GB write to `/tmp` is a multi-GB claim on the RAM
a serving model holds, with the OOM killer as arbiter: put scratch files on disk. And everything
in `/tmp` is gone at reboot — evidence written there has been lost twice, which is why the
cutover doc's original advice to put a state dir under `/tmp` was corrected.

vLLM unlinks its offload buffer only inside its own graceful shutdown path, so `flashnext` and
`glm53` pass `--shutdown-timeout 30` and the units carry `TimeoutStopSec=120`; a force-kill leaks
40 GiB until reboot. servedeck's reaper cleans an orphan but fails closed: it deletes only when it
could read every vLLM process's and every live model's mappings, because an unreadable holder is
indistinguishable from no holder and "no holder" means "delete 40 GiB" — see
[ARCHITECTURE.md](ARCHITECTURE.md).

## sysctl

`/etc/sysctl.d/90-servedeck.conf`, all three confirmed live 2026-09-18:

| Key | Value | Why |
|---|---|---|
| `kernel.yama.ptrace_scope` | 0 | Flash-Next's PLE CUDA-IPC handoff needs it to boot from a unit with no tty, and the offload reaper needs it to read a sibling engine's `maps` at all |
| `fs.inotify.max_user_watches` | 1048576 | the default 65,536 was exhausted |
| `fs.inotify.max_user_instances` | 1024 | default 128, of which 84 were in use |

The inotify raise has a specific cause: on 2026-09-16 `kimi` died at startup with `ENOSPC: System
limit for number of file watchers reached`. VS Code's file-watcher process held 59,628 of the
then-default 65,536 watches, its workspace being all of `~/Projects` (over 1M files: vllm-glm53
227k, cvi-scratch 153k, vllm-qwen38next 129k, local_llm 125k); nothing in the serving stack holds
watches. `files.watcherExclude` entries were also added to `~/.config/Code/User/settings.json`,
effective only after a window reload. Diagnostic: count per-process watches from `/proc/*/fdinfo`
before blaming the tool.

Docs elsewhere still name `/etc/sysctl.d/90-vllm.conf`; that file does not exist.
`/etc/sysctl.d/99-vllm-swap.conf` is a leftover setting `vm.swappiness = 10` on a box with no
swap, so it is inert, and its rationale (a swapped PLE table cost roughly 35% of decode) cannot
happen without swap. `serve-abliterated.sh:42` still makes a pointless
`sudo sysctl -q vm.swappiness=10` call. Removing both is cleanup, not a fix.

## PCIe, and why the CPU cannot rescue host offload

The link is PCIe 5.0 x16, and `nvidia-smi` reports `pcie.link.gen.current = 1` at idle — the
driver downclocks link and memory clock when the card is unused, returning to gen 5 under load.
Measured 2026-09-01 with FreeToken's `ft bench bw` (not re-measured): CPU STREAM read 64.6 GB/s
at 16c/16t against PCIe H2D 56.4 and D2H 57.3 GB/s. Host memory is only **1.15x** PCIe here,
because this is an X870E desktop board — dual-channel DDR5, not the quad-channel that was
assumed. CPU co-execution for weight offload therefore caps at about 1.36x aggregate (70.2 vs
51.8 GB/s) and cannot pay: FreeToken's own policy requires a 2.0x CPU/PCIe ratio before picking
its hybrid backend and selects plain offload here, and a hybrid attempt measured 0.895x. Published
results that look better came from a Xeon Platinum 8559C at ~178 GB/s, about 3x this box.

## Disk

`/` is one 915 G partition on `nvme0n1p2`: 746 G used, 123 G free, 86% (2026-09-18). The Hugging
Face hub cache is 360 G of that — GLM-5.3 182 G, Flash-Next 127 G, the 27B 24 G, LFM2.5-350M
681 M, all four in the registry. Reclaimable: `models--openai--gpt-oss-20b` 25 G and two
`models--CedricHwang--qwen2.5-0.5b-modelopt-fp8-*` repos totalling 1.2 G, none in `models.toml`.
Keep `models--Qwen--Qwen3-0.6B` (1.5 G) — the real-vLLM gate tests load it. Everything launches
from this cache and never the network (`HF_HUB_OFFLINE=1` is in `[defaults.env]` for every model),
so a deleted repo is a failed boot, not a silent re-download.

The other large trees are `vllm-glm53` 24 G, `local_llm` 15 G, `vllm-qwen38next` 7.7 G and
`servedeck` 80 M; `/var/log/journal` is 811 M and `/var/cache/apt` 593 M. Ten servedeck checkouts
exist (main, `v2`, worktrees `v2-p1`-`v2-p7` and `v2-p9`; there is no p8) at 13-53 MB each, and
all nine packet branches are already ancestors of main, so they carry no unique work. Do not `du`
across `~/Projects` as a whole — the AlgoTrading worktrees make that scan cost minutes.
