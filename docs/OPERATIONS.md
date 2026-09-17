# Operations

```bash
servedeck status        # what is live, what it is doing, how much room is left
servedeck doctor        # run this when status surprises you (exits 1 on any red row)
```

`status` reads `/api/state` and prints one row per registry model — slot,
ready/booting, unit state, context, KV in use, running/waiting, decode tok/s, uptime,
restarts — then the gateway URL, free VRAM and the headroom estimate. If it says
`dashboard not running at http://127.0.0.1:8010`, the models may still be serving:
they live in their own units.

## Everyday actions, and what each costs

| Action | Command | Cost |
| --- | --- | --- |
| See what serves | `servedeck status` | instant |
| Read a model's journal | `servedeck log flashnext -n 200` | instant, works with the dashboard down |
| Prove a name reaches the weights | `servedeck smoke main` | one chat + one tool call through `:8010/v1` |
| Put a model in the main slot | `servedeck switch glm53` | stop + VRAM wait + boot (below) |
| Start the small resident | `servedeck start lfm2` | ~102 s cold, 79 s of it CUDA-graph capture (e2e gate 2026-09-12; not re-measured) |
| Stop a model | `servedeck stop flashnext` | ~11 s for Flash-Next (2026-09-17: last request 22:58:13, unit gone 22:58:24) |
| Re-tune the live model | page → **Configure** → Apply & restart | a stop and a full boot, not a live change |
| Rewrite client configs | `servedeck wire --apply` | instant; no flag is a dry-run diff |

One model holds the `main` slot, so `switch` is the command for that slot; `start`
refuses it with `main_slot_busy`. `lfm2` is `slot = "resident"` and is started with
`start`; nothing adds it to desired state on its own (owner directive 2026-09-16: by
default this box runs one model).

**Boot, Flash-Next, measured live.** 2026-09-17 22:58:27 unit start → 23:00:44
`Application startup complete` = **137 s** warm: 66 s weight load, 4 s graph capture,
22.68 s engine init. An earlier boot the same evening took 179 s with a 90 s weight
load. A genuinely cold boot — no torch.compile AOT artifacts and none of the 97
FlashInfer JIT objects under `~/.cache` — was **up to ~10 min** under v1's `serve.sh`
(measured 2026-08-27; not re-measured under servedeck). `shm_broadcast: No available
shared memory broadcast block found in 60 seconds` during startup is informational.

**Switch** stops the old model, waits for `nvidia-smi` to give back **80 %** of what
that unit's cgroup held (up to **120 s**), then boots the next one. The driver frees a
context asynchronously, and without the wait a 90 GiB launch goes into a card that is
still occupied and fails minutes later as a CUDA OOM with no stated cause;
`vram_not_released` is the honest refusal. The accounting is over the whole cgroup,
not `MainPID`: in vLLM v1 the API server is `MainPID` and holds nothing while the
engine-core and worker children hold all of it.

**Apply & restart** (`POST /api/server/restart`) passes an explicit utilisation and
argv overrides (`--max-model-len`, `--max-num-seqs`, `--kv-offloading-size`) for that
one launch. A launch above a model's pinned `ctx` is refused, not clamped — GLM is
validated to 327,680 tokens here and its checkpoint claims 1,048,576. A plain `start`
on the model already serving returns 409; only restart relaunches. Utilisation is not
recomputed freely either: `models.toml` pins the value each main model is proven at
(flashnext 0.96, qwen27b 0.95, glm53 0.95) and `compute_util` refuses with
`not_enough_vram` when the card cannot fit it, rather than quietly serving at a
smaller KV budget. The card always keeps `margin_mib = 1024` plus a 700 MiB cushion
for the CUDA context, cuBLAS/cuDNN kernels and NCCL buffers that
`--gpu-memory-utilization` does not count.

## Restarts and reboots

**A model survives a servedeck restart.** Each runs as
`systemd-run --user --unit=model-<key> --collect` in its own cgroup, and servedeck's
unit sets `KillMode=process`, so stopping, restarting or crashing servedeck cannot
touch an engine — nor can closing the terminal a `servedeck start` was typed into. In
v1 vLLM ran inside the dashboard's cgroup with the default `KillMode=control-group`
and every dashboard stop SIGKILLed the model (journal 2026-09-11 21:38:12). Prove it
with `systemctl --user show -p ControlGroup model-flashnext` against
`/proc/self/cgroup`.

**Reconcile restarts nothing.** At startup servedeck starts only what
`state/desired.json` wants and is absent; a unit that exists is left alone, ready or
not, so the dashboard coming up cannot kill a model 40 s into a boot. It never
*writes* desired state — only an explicit start/stop/switch/adopt does — so a stopped
model stays stopped and a crashed model stays wanted.

**At reboot** `servedeck.service` is enabled (`WantedBy=default.target`) and launches
whatever `desired.json` names, with no tty and no human step. Desired state is schema
3: besides `main` and `residents` it stores each model's `launch` settings
(utilisation plus argv flags) from the last boot that actually became **ready**, and
reconcile replays them. Before that a reboot relaunched at registry defaults with
utilisation recomputed from free VRAM — 0.98 on an idle card, the value with the
OOM-under-concurrency history — and said nothing. A configuration that fails to boot
is never recorded, so a reboot never repeats one. A reboot with nothing wanted starts
nothing: at the 2026-09-17 22:36 reboot servedeck came up at 22:36:41 and launched no
model, because the operator had stopped the main model at 22:32:07.

**Models run with `Restart=always`** and `TimeoutStopSec=120`; flashnext and glm53
pass `--shutdown-timeout 30` so the engine unwinds pinned host memory and unlinks its
own `/dev/shm` buffer instead of being force-killed. `Restart=on-failure` never fired,
because vLLM's API server exits **0** when its engine dies. The unit serving right now
was launched 2026-09-17 22:58 and still carries the old `Restart=on-failure` /
`TimeoutStopSec=90`; the next launch picks up the new values.

**The poll is the last line of defence.** Every 2 s servedeck re-reads state; if
desired state names a model and nothing is running it, it relaunches — **3 times, 60 s
apart**, with a `recovering` notice each time — then stops and leaves the notice
standing. Three, because the failure this exists for is transient (an Xid fault, a
driver hiccup) and a model that cannot boot at all must not be relaunched forever.
Systemd cannot close this gap: after `StartLimitBurst` it gives up and `--collect`
removes the unit, making a crashed model indistinguishable from one nobody wanted.

## Logs

```bash
journalctl --user -u servedeck -f              # servedeck itself, its own logger included
journalctl --user -u model-flashnext -f        # one model's engine
journalctl -k --since "-2h" | grep -iE "NVRM: Xid|oom-kill|killed process"
servedeck log flashnext -n 200                 # the same journal, via the CLI
```

A model's log **is** its journal; there is no log file, and `servedeck log` falls back
to `journalctl` when the dashboard is down. servedeck's own Python logger reaches the
journal (a handler in `__main__`) alongside uvicorn's access log, so a `recovering` or
`not reaping` warning is visible there. `state/boot_logs/` holds v1-era captures that
nothing in servedeck reads.

The four lines a healthy boot prints, in order: `Loading weights took` →
`GPU KV cache size:` → `Capturing CUDA graphs` / `Graph capturing finished` →
`Application startup complete.` A boot that ends without the first never loaded
weights; without the second the KV cache did not fit. Both appear in the last 40
journal lines a failed start hands back. **Readiness is the port probe**, never the
markers: `GET /v1/models` on the model's own port answering 200 with one of its served
names — a cached compile or a skipped graph capture drops a marker and the model is
ready anyway.

Two endpoints on a model's own port disagree on purpose: `/health` returns **503 on
`EngineDeadError`**, while `/v1/models` is served from app state and stays **200 with
a dead engine** (RUNBOOK 2026-08-27; not re-measured), so only `/health` or a real
generation proves liveness — and an empty stream counts as a failure, not a 200: a v1
load test reported "60 ok, 0 failed" when 26 turns were post-mortem (2026-09-02). For
oversubscription read the engine's own counters: `vllm:num_preemptions_total` rising
means KV is being evicted and recomputed, and
`vllm:num_requests_waiting_by_reason{reason="capacity"} > 0` means KV-bound right now.
All counters reset on restart.

## Recovery

Confirm before fixing. Boot failures dominate here — **44 of 72 recorded exits never
reached serving** (v1 tally, 2026-08-27) — and a crash while serving usually recovers
on a restart while a boot failure loops and hides its cause.

### A boot failed

`servedeck start`/`switch` exits 1 with a reason and up to 20 journal lines. Confirm
with `journalctl --user -u model-<key> -n 200 --no-pager` and read the error before
restarting. Three causes seen here:

- **Gated-repo 401** — `GatedRepoError: 401 Client Error` on a companion file such as
  `processor_config.json`, ~15 s in. Flash-Next's first v2 launch died this way at
  2026-09-17 22:43:32: the weights were on disk, the loader asked the hub anyway.
  Fixed by `HF_HUB_OFFLINE = "1"` in `[defaults.env]` for every model; doctor's
  `weights (<key>)` row answers the other half, whether the snapshot is local.
- **CUDA OOM** — either the card was not empty (the switch VRAM wait exists for this)
  or the utilisation was too high for the concurrency that arrived.
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` names the holder.
- **An assert or `ValueError` naming a flag or a shape** — the flag is wrong for the
  build, not the box. Reasons per model in [models/](models/), builds in
  [BUILDS.md](BUILDS.md).

A start now preflights the three that used to cost an outage: host RAM against
`host_ram_gib` (flashnext 95, glm53 155 — no swap here, so overshooting is an OOM kill
of the session, and one happened 2026-08-28); `--cpu-offload-gb` together with
`--kv-offloading-size`, which double-books host memory; and a port that already has a
listener, naming the pid.

### The engine died and the unit was collected

Clients get "not running" or a gateway 503, `status` shows the model absent, and
nothing looks broken. `servedeck doctor` turns the `desired (<key>)` row red: "wanted
but no `model-<key>` unit exists". Fix with `servedeck start <key>` and read the
journal — the poll will already have tried up to 3 times.

Why it hides: `--collect` sets `CollectMode=inactive-or-failed`, so systemd unloads a
transient unit *including when it failed*, and `systemctl show` then exits 0 for a unit
it has never heard of, printing property **defaults** (`LoadState=not-found`,
`Result=success`) — so the obvious predicate `Result != "success"` reads every crash as
a clean stop. Every judgement in `control.py` pairs `units.show()` with
`units.exists()`, and the diagnosis comes from `journalctl`, which survives collection.

Apparent *context loss* in a client is usually this failure, not a client bug: all
three `chat/completions` 500s in one GLM session coincided exactly with an engine death
(two CUDA OOMs and an Xid 31) while 83 of 86 agent requests returned 200 (2026-09-02).
A dead turn leaves an empty assistant message in the client's history and the chat
template renders that as a fresh thread, so verify the server first.

### A leaked 40 GiB `/dev/shm` buffer

Host RAM is short after a crash, or a start refuses with `not_enough_host_ram`, for no
visible reason.

```bash
ls -la /dev/shm | grep vllm_offload    # one 42,945,576,960-byte file per engine
grep -l vllm_offload /proc/*/maps      # who still maps them
```

A file no pid maps is a leak; doctor's `offload buffers` row says the same. Any
`servedeck stop` or `start` reaps it, and the poll reaps on every lap. vLLM unlinks the
buffer only in its own graceful shutdown, which a SIGKILL, a crash or a lost GPU skips,
and every relaunch adds another under a fresh engine id — 40 GiB of the 182 gone per
crash until reboot. The reaper **fails closed**: it deletes nothing unless it could
read the mappings of every vLLM process and every live model, because an unreadable
holder is indistinguishable from no holder and "no holder" means "delete 40 GiB". That
needs `kernel.yama.ptrace_scope = 0`; at 1 it logs `not reaping N KV offload buffer(s)`
and stops. Baseline: one live Flash-Next = one 40 GiB buffer, `/dev/shm` 41G of 92G
(2026-09-18).

### A foreign process holds a model port

A start refuses with `port_busy` and a pid, or doctor reports `port N (<key>)` red.
Confirm with `ss -Hltnp 'sport = :8002'`, then `tr '\0' ' ' < /proc/<pid>/cmdline`;
never with `pgrep -f`, which matches the calling shell's own command line (exit 144)
and has killed the invoking shell here. Stop that process, or change the model's port
in `models.toml`. Registry ports get squatted by unrelated projects — `:8002` and
`:8003` repeatedly by `cvi-scratch` servers, and as of 2026-09-18 `:8002` is held by
`scripts/serve.py --db .../cvi-L1.db --port 8002`, so doctor shows one red row and
`servedeck start glm53` would refuse. Without the preflight a `switch` stops the
running model, waits for VRAM, loads a 90 GiB engine for minutes and only then fails
to bind, leaving the card empty and every client on a 503. Not listening is **green**;
red means the port answers and serves something else.

To find engine processes, match on `comm`:
`ps -eo pid,comm --no-headers | awk '$2=="vllm" || $2 ~ /^VLLM::/'` — the `VLLM::`
prefix, never the full name, because `VLLM::EngineCore` is 16 characters and `comm`
truncates at 15.

### The GPU fell off the bus

Every request fails and `nvidia-smi` is slow or errors. Confirm with
`journalctl -k --no-pager | grep "NVRM: Xid" | tail -5`.

| Xid | Meaning | Restart? |
| --- | --- | --- |
| 13, 31 | GPU MMU fault (illegal/misaligned address); the documented, MTP-correlated crash | yes, a restart may recover |
| 79 | GPU has fallen off the bus — hardware/driver level | **no: reboot** |
| 154 | Driver recovery action asserted ("Node Reboot Required") | **no: reboot** |
| anything else | unrecognised | **no — do not assume it is the MMU fault** |

If `nvidia-smi -L` fails at all, never restart: a reboot is the only fix. Six restarts
in 21 s once burned the systemd burst budget against a card that was off the bus.
`servedeck/xid_watch.py` can classify these and write `state/crash_reports/<ts>.json`,
but nothing starts it, so read the kernel journal. Xid history: [HOST.md](HOST.md).

### A client says the model is not running

A 404 naming the model, or a 503 from the gateway. `servedeck doctor` probes each
configured id at its configured URL. **`missing`** means that URL serves something
else: the 404 body lists every name that would have worked, so add it to `aliases`
(names are only added, never renamed), restart the model and `servedeck wire --apply`.
**`unreachable`** means nothing answers there — check servedeck itself. **`wired`** is
green: the id is in the registry but not running, and the gateway answers it with a 503
and a reason until it is; without that distinction three stopped models made the page
report "13 of 29 failing" with nothing wrong (2026-09-17). A 503 with `Retry-After: 15`
naming what holds the slot is a switch in progress. Details: [CLIENTS.md](CLIENTS.md).

### servedeck itself will not start

`systemctl --user status servedeck` and `ss -ltnp 'sport = :8010'` — something else
holds the port; stop that first, and never run a second servedeck against the same
state directory, or both react to a crash. On 2026-09-11 a hand-started copy held
`:8010` and the unit crash-looped **67 times in 9 minutes**, launching a real vLLM boot
on every lap, because reconcile ran inside the ASGI lifespan, which uvicorn executes
*before* it binds. Now reconcile waits for servedeck's own `/api/health` and compares
the instance nonce — a 200 from a *different* servedeck does not count — and starts
nothing if the port has not answered in 60 s (`bind_timeout`); the unit carries
`StartLimitIntervalSec=60` / `StartLimitBurst=3`.

## `servedeck doctor`, row by row

Loopback-only, read-only, 2 s per probe, exit 1 if any row is red. Read the failing
row, not the summary.

| Row | Red means | Do this |
| --- | --- | --- |
| `registry loads` | `models.toml` is invalid; every other check is skipped | fix the one error it names |
| `vscode:` / `codex:` / `kimi: <id>` | `missing` or `unreachable` | see the section above |
| `port N (<key>)` | the port answers and serves a different model, or answers unusably | `ss -Hltnp 'sport = :N'` |
| `unit (model-<key>)` | only a **stray** is red: a live `model-*` unit whose key the registry does not know | add it to the registry, or `systemctl --user stop model-<key>` |
| `training marker` | a lock file exists — something else wants the GPU | leave the card alone; remove it only once that run has finished |
| `ptrace_scope (host)` | not 0: Flash-Next's PLE handoff (`pidfd_getfd` between sibling workers) fails and the offload reaper goes blind | owner action, `/etc/sysctl.d/90-servedeck.conf` |
| `desired (<key>)` | wanted, but no unit — the engine died and the unit was collected, or it never started | `servedeck start <key>`, read the journal |
| `weights (<key>)` | the checkpoint is missing or unservable in the local hub cache; with `HF_HUB_OFFLINE=1` that is a dead launch, not a slow one | fetch the named repo |
| `offload buffers` | `vllm_offload_*.mmap` files hold host RAM no engine maps | a stop or start reaps them; if they survive, check `ptrace_scope` |
| `power cap` | the live cap disagrees with `models.toml [gpu] power_limit_w` | a boot unit re-applies its own value: `/etc/systemd/system/nvidia-power-limit.service` |

`<kind> config` and `<kind> config (remote)` never go red: a missing client file is
nothing to check yet, and entries pointing off this box are listed but deliberately
not probed. `power cap` cannot go red until `[gpu] power_limit_w` is set — as of
2026-09-18 it is not, and the row reports the live value only (375 W).
`unit (model-<key>)` deliberately does not look for a unit file: v2's models are
transient units under `/run/user/<uid>/systemd/transient`, so a file check there could
never pass.

A 409 from `POST /api/…` is the snapshot already answering, with `reason` one of
`busy` (a mutation is in flight; one lock serialises them), `already_live`,
`main_slot_busy` (use `switch`), `not_main_slot` (residents use `start`), `not_live`,
`ctx_unresolved` or `unknown_model` — each re-checked inside `Control` against reality,
the pre-check existing only so the common refusals arrive on the POST rather than as an
SSE notice you had to be watching for. A cross-origin write to `/api` is refused;
`/v1` stays open.
