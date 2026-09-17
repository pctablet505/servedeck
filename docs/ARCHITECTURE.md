# Architecture

One Python process holds the page, the JSON API, the `/v1` gateway, the poller
and reconcile. Every model runs outside it, as its own transient systemd unit.
`models.toml` is the only place a model's identity lives.

```
clients ──► servedeck  (one uvicorn process, 127.0.0.1:8010)
              ├── /v1/*          gateway: name → port, policies, streaming
              ├── /api/*         control: state, mutations, wire, doctor, SSE
              ├── /api/server/*  the v1 page's controls (legacy_page.py)
              └── /              the page (servedeck/web/)
                      │ systemd-run --user --unit=model-<key> --collect
                      ▼
              model-<key>.service ──► vllm serve   (its own cgroup)
```

servedeck runs from `~/.config/systemd/user/servedeck.service` (`Type=exec`,
`Restart=on-failure`, 3 starts per 60 s, `KillMode=process`,
`TimeoutStopSec=15`) and starts no model itself.

## The one process, and what its death costs

Page, API, gateway, poller and reconcile share one event loop. That buys one
thing worth having — no second control plane to disagree with the first — and
costs a single point of failure: when the process dies every client loses the
only URL it knows, while the models keep serving on their own ports and notice
nothing. Recovery is servedeck coming back and re-reading reality.

Two rules keep the loop alive. **Nothing in a request handler blocks**:
`systemctl`, `journalctl` and `nvidia-smi` run under `asyncio.to_thread` and
request paths read only the poller's snapshot, because a coroutine parked on a
subprocess stalls the gateway mid-stream. **Reconcile runs only once our own
port answers as us**: `_reconcile_after_bind` polls `/api/health` until its
`instance` matches this process's id (60 s ceiling, 0.25 s interval). uvicorn
runs the ASGI lifespan *before* `bind()`, and on 2026-09-11 a lifespan
reconcile in the process that *lost* the port launched the 27B 70 times in
9 minutes; a status-code check cannot tell a rival servedeck's good 200 from
its own.

Registration order in `create_app` is load-bearing — `/api/*` (including
`legacy_page.register`), then the gateway, then the page — because
`/v1/{path:path}` and `/{asset:path}` are catch-alls and Starlette matches in
registration order. A same-origin middleware refuses writes to `/api/*` from a
foreign `Origin`, since mutations take their key in the path and no body and so
are CORS simple requests any browser tab could send; `/v1` is not covered and
callers with no `Origin` are unaffected.

## The registry is the source of truth

`models.py` loads and validates `models.toml` and renders one entry into a
`vllm serve` argv and env. `render_argv(util, port)` is pure, which is what
lets `tests/test_models_golden.py` compare it byte-for-byte against a legacy
launcher's dry run; resolving `ctx = "native"` needs the hub cache and is a
separate call. Utilisation lives in `control.compute_util`, not here — one
implementation, not two: a `main` model gets
`floor2((free_mib - margin_mib) / total_mib)`, unless `models.toml` pins the
`util` it is proven at, in which case a card that cannot fit that value is a
refusal naming the holder rather than a quiet launch at a different KV budget.
Derived flags follow the same rule: GLM's `--kv-cache-memory-bytes` is
`ctx_tokens * kv_cache_bytes_per_token`, so context and KV cap cannot drift.

## Models as transient units

`units.py` translates only: build an argv, run it, parse it back. Never
`shell=True`; only `model-<key>` and the suite's `sd-test-<name>` are accepted
as names. The point of transient units is cgroup separation — the model sits
under the user manager, not under servedeck, so restarting or crashing
servedeck cannot take it along and there is no `KillMode` to get wrong. v1's
in-process launcher took the model down with the dashboard on every restart.

| Property | Value | Why |
|---|---|---|
| `Restart=` | `always` | vLLM exits **0** when its engine dies (the watchdog returns from `serve_http` normally), so `on-failure` fires zero times. |
| `StartLimitIntervalSec=`/`Burst=` | 300 / 3 | systemd's default 5-per-10 s cannot fire with `RestartSec=10`, so a model that cannot boot would restart forever; 3 per 300 s reaches `failed` in ~30 s. |
| `TimeoutStopSec=` | 120 | Must exceed the model's `--shutdown-timeout` (30 for flashnext, glm53): `cudaHostUnregister` on a 40 GiB offload buffer and its `/dev/shm` unlink happen inside engine shutdown. |
| `UnsetEnvironment=` | per launch | A transient unit inherits the user manager's environment, which here holds credentials for unrelated services. Unreadable environment ⇒ refuse the start. |

`--collect` is required and is a trap: it unloads a *failed* unit at once,
after which `systemctl show` returns an unknown unit's defaults
(`ActiveState=inactive Result=success`), so the naive check reads "clean stop"
for every crash. Every judgement in `control.py` is written around that, and
failure reports come from the journal, which outlives the unit.

`start()` preflights what otherwise costs an outage rather than a second: host
RAM against `host_ram_gib` (no swap here, so overshooting OOM-kills the
session), a refusal when `--cpu-offload-gb` and `--kv-offloading-size` would
both be passed, and a refusal when the port already has a listener, naming the
pid. A start that never became ready is cleaned up with `units.stop`, not
`control.stop`, so the operator's recorded intent survives.

## Readiness

The four journal markers (`Loading weights took`, `GPU KV cache size:`,
`Capturing CUDA graphs`, `Application startup complete.`) drive the progress
bar only. Truth is `GET 127.0.0.1:<port>/v1/models` answering 200 with one of
that model's served names, so a boot that skips a marker still becomes ready.
The journal follower starts before the unit, anchored two seconds in the past,
because `journalctl --since` has one-second granularity.

## Adoption

`control.live()` lists every `model-*` unit, joins each with a port probe, and
marks a unit whose key the registry does not know `unknown=True` rather than
guessing. It then adopts: a registered model with *no* unit whose port has a
real listener (socket table) answering with that model's registered id appears
as `unit="(adopted)"`, `sub_state="adopted"`, with the listener's pid — which
is how a hand launch, or v1's server at the cutover, becomes visible without
being restarted. A foreign process on a registry port is doctor's finding, not
a model. `control.adopt()` is the operator action that writes adopted units
into desired state; unknown keys are reported, never adopted, because without
a spec there is no slot to file them under.

## desired.json and reconcile

`state/desired.json` is intent, not a mirror. Schema 3 (2026-09-18):

```json
{"version": 3, "main": "flashnext", "residents": ["lfm2"],
 "launch": {"flashnext": {"util": 0.96, "argv": {"--max-model-len": "262144",
            "--max-num-seqs": "16", "--kv-offloading-size": "40"}}}}
```

`launch` records the settings of the last boot that actually became **ready**,
so a configuration that failed is never the one a reboot repeats. Without it
the allocator was write-only: the tuned utilisation, context, agent count and
offload size lived in a running process's argv and nowhere else, so the next
restart relaunched at registry defaults with utilisation recomputed from an
idle card (0.98 — the value with the OOM-under-concurrency history) and nothing
said so. `argv` holds whole flags rather than named fields, so a control the
page grows tomorrow persists without a schema change; a `null` value removes a
flag. Writes come only from `control.start`/`stop`/`switch`/`adopt`, atomically
(same-directory temp file, `fsync`, `os.replace`). Missing, empty, unparseable
or unknown-version reads as "want nothing" with a warning, never an exception;
v2 migrates to v3 in place and silently, v1 is read as
`main = backend if desired_state == "RUNNING"`, copied to `desired.json.v1`
and rewritten, because a file left at the old version re-warned every poll.

`reconcile` makes desired state true and **restarts nothing** — a unit that
exists is left alone, ready or not, so servedeck coming up cannot kill a model
40 seconds into a 90-second boot. It never stops anything and never writes
desired state: a crashed model must stay desired, a stopped one stopped. Each
key it does start is started from that key's stored `launch` values. It blocks
for as long as the boots take, hence a task created after the bind.

## The poll loop

Every 2 s the poller rebuilds the `/api/state` document and publishes it. Its
blocking half runs in a thread: the offload reaper, `routes.refresh()` (the
pushed liveness snapshot every request path reads), unit start times,
`nvidia-smi` total/free, `desired.json`, one `/metrics` scrape per ready model.

| SSE event | Cadence | Payload |
|---|---|---|
| `state` | on change | The whole state document. `generated_at`, `uptime_s` and the volatile throughput fields are excluded from the comparison, or "on change" would mean "every 2 s". |
| `telemetry` | every poll | v1's live strip (`legacy_page.telemetry_payload`), changed or not. |
| `progress` | per journal line | `{key, kind, text, marker_index, elapsed_s}`; `kind` ∈ `line`/`marker`/`ready`/`failed`, `key` is `"reconcile"` at startup. |
| `notice` | per event | `{level, reason, message, …}`. The last 50 replay to a new subscriber, flagged `replay: true` so a CLI waiting on its own start does not stop on an hour-old `ready`. |
| keepalive | 15 s | The comment line `: keepalive` — not an event. |

Publishing is thread-safe (`call_soon_threadsafe`); each subscriber's 256-deep
queue drops its **oldest** event, so a slow page loses history rather than
stalling the poller. The poll also closes the gap systemd cannot:
`_recover_desired` relaunches a model desired state names while nothing is
running — the state `--collect` plus an exit-0 engine death makes
indistinguishable from "nobody wanted it" — three times, ≥60 s apart, one model
per poll, with a `recovering` notice each time, then stops and leaves the
notice standing.

## The KV-offload reaper

vLLM's CPU KV offload buffer is `/dev/shm/vllm_offload_*.mmap`, sized by
`--kv-offloading-size` (40 GiB for Flash-Next) — host RAM, on a box with no
swap. vLLM unlinks it only in its graceful `cleanup()`, so an engine that is
SIGKILLed, crashes or loses the GPU leaves the whole buffer pinned until
reboot, and each relaunch adds another under a fresh engine id. The reaper runs
in `control.start` and in every poll (a systemd restart never passes through
`start`), publishing an `offload_reaped` notice with the size freed.

It **fails closed**. The scan reads `/proc/<pid>/maps` and `/proc/<pid>/fd` for
every process whose cmdline contains `vllm`, plus every pid servedeck believes
belongs to a live model. If `/proc` cannot be listed, any one of those
processes cannot be read, or servedeck thinks a model is live but cannot
enumerate its cgroup's pids, the scan is `trusted=False` with a reason and
**nothing is deleted** — an unreadable holder is indistinguishable from no
holder, and "no holder" means "delete 40 GiB". Reading a sibling engine's maps
needs `kernel.yama.ptrace_scope = 0`, which is also what Flash-Next's PLE
handoff needs; doctor reports it as a host row. See [HOST.md](HOST.md).

## The state directory

`state/`, resolved from the checkout (the checkout is the deployment).

| Path | Written by | What it is |
|---|---|---|
| `desired.json` | `desired.save` | Operator intent, schema 3. The only file reconcile reads. |
| `desired.json.v1` | the v1 migration | The pre-migration file, kept for the runbook's rollback. |
| `backups/<date>/<flattened path>` | `wire --apply` | Prior content of every client config a write changed. |
| `measurements.json` | `discovery` | Append-only store of boot-measured KV rates and contexts. |
| `telemetry/*.jsonl`, `crash_reports/*.json` | `telemetry`, `xid_watch` | GPU samples, Xid events, one report per fatal Xid — written only when something runs the sampler, and the app does not (see debt). |

v1 leftovers, referenced by no code in `servedeck/` and readable as history
only: `server.json` (+ `.bak-dead-glm53-pid214759`), `ack.json`,
`extra_args.json`, `history.jsonl`, `boot_logs/`, `desired.json.bak-glm53-stale`.

## The legacy page surface

`servedeck/legacy_page.py` exists because **the page is v1's page**: v1's
`web/index.html` + `app.js` + `style.css` as at tag `pre-cutover-2026-09-17`
is what the owner uses and what is served, and the owner rejected a redesign.
The module builds v1's shapes from the v2 runtime — `upstream`, `vllm`,
`sizing`, `boot`, `supervisor`, `config`, `gpu_v1`, `uptimes` blocks *added* to
the v2 state document rather than substituted, a `telemetry` event per poll,
the `/api/models` rail with disk sizes, `/api/disk`, `/api/capacity/estimate`
and the `/api/server/*` controls — so the CLI, doctor and the tests keep
reading the v2 document unchanged. None of the arithmetic is reimplemented:
sizing calls `parallelism`, the estimate calls `capacity` and `kvcalc`, disk
sizes come from `disksize` and `discovery`. A failure in the legacy blocks is
caught so it cannot take the v2 state down.

## The mutation contract

A mutation answers `202 {"accepted": true, …}` and reports over SSE; the
blocking `Control` call runs in a thread. Anything decidable from the snapshot
is refused **synchronously**, so no client has to watch a stream to learn it
was rejected: `404 unknown_model`, `409 busy`, `409 already_live`,
`409 main_slot_busy`, `409 not_main_slot`, `409 not_live`,
`409 ctx_unresolved`. The `reason` vocabulary is exactly `control.Refusal`'s,
so a refusal decided from the snapshot and one decided against reality are
indistinguishable to a client; refusals that need reality
(`not_enough_vram`, `not_enough_host_ram`, `conflicting_offload`, `port_busy`,
`env_scan_failed`, `gpu_unavailable`, `start_failed`, `boot_failed`) arrive as
notices. Single-flight has two layers and the outer one matters: `_claim` sets
`rt.busy` **in the handler**, immediately after a clean precheck with no
`await` between, which is what makes check-then-claim atomic — setting it
inside the task let two Applies in one tick both pass and both restart the
model. `_run_mutation` then holds `rt.lock` for the whole operation and clears
`busy` only *after* publishing the outcome, because clearing it first left a
tick where `/api/state` said nothing was running while the notice stream was
about to say ready.

## Module map

| Module | One line |
|---|---|
| `settings.py` | Where to listen, where `models.toml` and `state/` are, which unit namespace we own. Imports nothing from `servedeck`. |
| `limits.py` | The four *hardware* numbers: VRAM total, non-KV overhead, fragmentation margin, training markers. |
| `models.py` | Loads and validates `models.toml`; renders one model's `vllm serve` argv and env. |
| `desired.py` | `state/desired.json`: operator intent, schema 3, atomic writes, v1/v2 migration. |
| `units.py` | `systemd-run`/`systemctl`/`journalctl` wrappers. No policy, no `shell=True`, two allowed name shapes. |
| `control.py` | The supervisor: preflight, start, wait-ready, stop, switch, adopt, reconcile, `compute_util`, the reaper. |
| `routes.py` | The one join between registry names and live ports: resolution total, liveness a pushed snapshot. |
| `gateway.py` | The `/v1` proxy: resolve `model` → port, apply policies, stream both ways, 404/503/502 envelopes. |
| `policies.py` | Pure byte transforms: reasoning mirror, effort overlay, output floor, model rewrite, `scan_model`. |
| `glm_policies.py` | GLM-5.3-only client repairs (tool-tag sanitiser and kin), kept out of `policies.py` so the generic layer stays generic. |
| `app.py` | The ASGI app: `/api/*`, the SSE hub, the poller, reconcile-after-bind, the origin guard, the page. |
| `legacy_page.py` | v1-shaped payloads and `/api/server/*` on the v2 runtime, because v1's page is the page. |
| `__main__.py` | `python -m servedeck`: servedeck's own logs into the journal, then uvicorn with a shutdown that closes the hub first. |
| `cli.py` | The `servedeck` console script; control commands drive the server over HTTP, falling back to in-process `Control`. |
| `cli_gpu.py` | `servedeck gpu-log`'s parser over `telemetry`/`xid_watch` — one call from being wired into `cli.py`, and not wired. |
| `wire.py` | Rewrites VS Code / Codex / Kimi configs from the registry, in place, its own tables only, with a diff. |
| `doctor.py` | Proves the registry against reality: registry, client configs, ports, units, and host rows (ptrace_scope, desired, weights, offload buffers, power cap). |
| `discovery.py` | The local hub cache — downloaded, servable, how big — plus `measurements.json`. |
| `disksize.py` | Deduplicated on-disk size: blocks not `st_size`, inode identity not path. |
| `gpu.py` | `nvidia-smi` total/free/per-pid usage and Xid classification. Never raises. |
| `capacity.py` | Pure VRAM/KV arithmetic and findings; every environmental fact is injected. |
| `kvcalc.py` | Per-architecture KV cache arithmetic from a model's own `config.json`. |
| `parallelism.py` | How many agents fit, as a KV-admission question fitted to measurement. |
| `metrics.py` | Hand-rolled Prometheus scraping of the live engine; an absent family degrades to `None`, never 0. |
| `tokens.py` | Prompt / generated / prefix-cached token accounting from three counters and one start time. |
| `reqstats.py` | A rolling window over the last N finished requests, rebuilt from vLLM's histograms. |
| `telemetry.py` | The GPU sampler: one JSON record every 5 s into `state/telemetry/`. |
| `xid_watch.py` | Watches the kernel journal for Xid lines; writes a crash report per fatal one. |
| `web/` | v1's page: `index.html`, `app.js`, `style.css`. |

Gone in v2 and not returning: `config.py`, `paths.py`, `supervisor.py`,
`procctl.py`, `phases.py`, `updetect.py`, `shellconfig.py`, `legacy.py`,
`history.py`, `logtail.py`, `preflight.py`, `smoke.py`, `events.py`, and
`registry.py` (now `discovery.py`, because the model registry owns that name).

## Debt

**Two mutation API families.** `/api/models/{key}/*`, `/api/switch/{key}` and
`/api/adopt` key on the registry key, precheck into typed refusals and take no
body. `/api/server/{start,restart,stop,adopt}` key on `repo_id` + `backend`,
return untyped `{"error": "…"}`, and are the **only** route that accepts the
allocator's `util` and argv overrides — which is why they cannot be deleted
while v1's Configure panel is the page. `/api/server/adopt` also answers
synchronously where `/api/adopt` is 202 + SSE. Both families share
`_claim`/`_run_mutation`, so they cannot race, but they are two surfaces with
one meaning.

**The single point of failure**: nothing serves `/v1` while the process is
down, and every client knows only that URL.

**Unwired instrumentation.** `telemetry.py` and `xid_watch.py` are written and
tested, but nothing starts the sampler (the `telemetry` SSE event is
`legacy_page`'s strip, a different thing) and `cli_gpu.py` is not registered in
`cli.py`, so `servedeck gpu-log` does not exist. The reason they were written —
2026-09-12, the GPU fell off the bus and nothing on the box had recorded its
temperature — is still unaddressed on the running box.

**`needs_tty`** was retired from `models.toml` on 2026-09-18. It survives as a
`Model` field defaulting to `False`, so an older registry still loads; nothing
reads it and `/api/state` no longer publishes it.

## Where the older documents went

Six documents were deleted on 2026-09-18 once this set absorbed them. They are
in git history, and `git log --diff-filter=D --name-only -- docs/` finds the
commit; the last revision of each is `git show HEAD~1:<path>`. Test docstrings
that cite `SPEC.md §3` or `REDESIGN-2026-09-12.md §2.7` mean those files at
that revision.

| Deleted | Absorbed by | Why it went |
| --- | --- | --- |
| `docs/SPEC.md` | this file + [CONFIGURATION.md](CONFIGURATION.md) | An interface spec written before the code; its route list never gained `/api/server/*`, `/api/disk` or `/api/capacity/estimate`, which is the surface the served page actually runs on |
| `docs/TROUBLESHOOTING.md` | [OPERATIONS.md](OPERATIONS.md#recovery) | Its doctor table named rows that no longer exist and the wrong sysctl file; the playbooks are regenerated from `doctor.py` |
| `docs/REDESIGN-2026-09-12.md` | this file + [DECISIONS.md](DECISIONS.md) | The v2 design document. Its decisions are settled and now recorded as decisions, with dates |
| `docs/CUTOVER-2026-09-12.md` | — | A runbook for a cutover that completed on 2026-09-17 |
| `docs/PLAN-2026-09-17.md` | — | An audit and plan whose items are done, superseded, or in the owner's hands; the findings that outlived it are in these docs |
| `builds/README.md` | [BUILDS.md](BUILDS.md) | One docs home |

Keys and defaults: [CONFIGURATION.md](CONFIGURATION.md). Running it:
[OPERATIONS.md](OPERATIONS.md). Why things are the way they are:
[DECISIONS.md](DECISIONS.md).
