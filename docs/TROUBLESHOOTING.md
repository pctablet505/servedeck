# Troubleshooting (v2)

Every entry here is a failure that actually happened on this box, with what v2
does about it. The evidence is in
[REDESIGN-2026-09-12.md](REDESIGN-2026-09-12.md) §1 (R1–R5) and §4.

---

## A client gets `404 "the model does not exist"`

**Happened:** twice — the 27B rename on 09-11, Flash-Next on 09-12. A model's
public name lived in `chatLanguageModels.json`, `~/.codex/config.toml`, three
proxy units, `servedeck.toml`, `local_llm/.config` and several scripts. Change
it in one, and everything else 404s (R1).

**Fix:** `models.toml` carries `aliases`, and every alias is passed to
`--served-model-name`, so **every old name keeps working**. `GET /v1/models`
lists each alias and each preset as its own entry with `root` pointing at the
real id. Names are only ever added, never renamed.

**If you still see it:** the name is not in the registry at all. The 404 body
lists every name that would have worked. Add it to `aliases`, restart the
model, and run `servedeck wire --apply`.

---

## The wrong model's flags were used / `.config` was poisoned

**Happened:** three times. `llm` and servedeck both wrote
`local_llm/.config`, which four different parsers read (two of them by
`source`-ing it, i.e. executing it), last assignment winning. Flash-Next's
`EXTRA_ARGS` got handed to the 27B; the 27B was killed at boot by a flag that
belonged to another model (R2).

**Fix:** there is no `.config`, no launcher script and no `EXTRA_ARGS` stash.
The argv comes from `models.toml` and is rendered by a pure function. The `llm`
CLI is retired; `servedeck` is the CLI.

---

## "Which port is safe for this model?"

**Happened:** `:8003`, `:8005` and `:8006` were per-model side proxies, each
fronting one *fixed* upstream port, each doing one normalisation (reasoning
mirror, effort injection). A port swap silently broke one; `:8005` spent a day
proxying to a dead `:8001` (R3).

**Fix:** one URL, `http://127.0.0.1:8010/v1`, for every client and every model.
All three proxies' transforms are in `servedeck/policies.py` and are selected
*per route, by the registry*. No client is ever pointed at a model's own port
again. While the main slot is switching you get a `503` with `Retry-After: 15`
and a body naming what is in the slot — not a connection refused.

---

## servedeck crash-looped at boot and launched the model 70 times

**Happened:** 2026-09-11, 21:36–21:45. `servedeck.service` could not bind
`:8010` because a hand-started copy held it. The unit had no `StartLimit*`, so
it looped 67 times in 9 minutes — and because `reconcile_startup()` ran inside
the ASGI lifespan, which **uvicorn executes before it binds**, every single lap
launched a real vLLM boot before exiting with "address already in use" (R4).

**Fix:** `app.py`'s `_reconcile_after_bind` polls **our own** `/api/health` on
our own listen socket, and only then calls `control.reconcile`. If the port
never answers within 60 s, nothing is started and a `notice` event says
`bind_timeout`. The shipped unit also has `StartLimitIntervalSec=60` /
`StartLimitBurst=3`, so a persistently broken install gives up loudly.

**If it happens anyway:** `systemctl --user status servedeck` and
`ss -ltnp 'sport = :8010'`. Something else holds the port; stop that first.

---

## Stopping the dashboard killed the model

**Happened:** journal 2026-09-11 21:38:12. vLLM ran inside the dashboard's own
cgroup and the unit used the default `KillMode=control-group`, so every
dashboard stop SIGKILLed the model (R4).

**Fix:** models are **transient units in their own cgroup**
(`systemd-run --user --unit=model-<key> --collect`), reparented to the user
manager under `app.slice`. Restarting servedeck — or closing the terminal a
`servedeck start` was typed into — cannot touch a model. The shipped
`servedeck.service` additionally sets `KillMode=process` so the stop signals
only uvicorn.

Prove it: `systemctl --user show -p ControlGroup model-<key>` and compare with
`/proc/self/cgroup`.

---

## A crashed model reads as a clean stop

**Happened:** `--collect` sets `CollectMode=inactive-or-failed`, so systemd
unloads a transient unit *including when it failed*. `systemctl show` then
exits 0 for a unit it has never heard of and prints the **default** value of
every property asked for:

```
LoadState=not-found
ActiveState=inactive
Result=success          # a lie: the default, not this unit's outcome
NRestarts=0
```

The obvious failure predicate — `Result != "success"` — therefore reads *clean*
for every crash. A broken measurement fails downward.

**Fix:** every judgement in `control.py` pairs `units.show()` with
`units.exists()` (`LoadState=loaded`). *The unit vanished while we were waiting
for it* is a failure, not a clean stop. The journal survives collection, so the
diagnosis comes from `journalctl`, never from `Result=`.

---

## Stopping servedeck hangs for `TimeoutStopSec` and then gets SIGKILLed

**Happened:** every single stop. Each open page held an SSE generator parked on
`queue.get()` that nothing would ever complete; uvicorn waited for them on
shutdown.

**Fix:** `Hub.close()` hands every subscriber a `None` sentinel so its
generator returns, and the lifespan calls it **before** cancelling the
background tasks. `TimeoutStopSec=15` in the unit is now a backstop, not the
normal path.

---

## `servedeck doctor` is red

`doctor` runs five kinds of check. Read the failing row, not the summary:

| Row | Red means | Do this |
|---|---|---|
| `registry loads` | `models.toml` is invalid. Everything else is skipped. | Fix the one error it names. |
| `vscode: <id>` / `codex: <id>` / `kimi: <id>` | The configured URL answers, but does not serve that id (`missing`), or does not answer at all (`unreachable`). | `servedeck wire` for the diff, `--apply` to fix; or start the model. |
| `port N (<key>)` | The port answers **and serves a different model**. Not listening is green — that is just a stopped model. | Two models share a port, or a stray process holds one. `ss -ltnp 'sport = :N'`. |
| `systemd unit (<key>)` | No unit file for the model. | Expected until P5/P6 land: `model-<key>.service` is transient and has no file on disk. |
| `training marker` | A lock file exists — something else wants the GPU. | Leave the card alone. Remove the marker only once that run has really finished. |
| `ptrace_scope (<key>)` | A `needs_tty` model needs `kernel.yama.ptrace_scope = 0` for its PLE CUDA-IPC handoff. | Owner action: `/etc/sysctl.d/90-vllm.conf`. The launcher's own `sudo sysctl` silently no-ops under systemd — there is no tty. |

`servedeck doctor` exits 1 if any row is red, so it works in a script.

---

## Reading a model's journal

```bash
journalctl --user -u model-<key> -n 200 --no-pager        # last 200 lines
journalctl --user -u model-<key> -f                       # follow a boot
journalctl --user -u model-<key> --since "-1h" -o cat     # bare text, no prefix
```

`servedeck log <key> -n 200` prints the same thing and works whether or not the
dashboard is running.

The four lines a healthy boot prints, in order — these drive the progress bar,
not readiness:

1. `Loading weights took …`
2. `GPU KV cache size: …`
3. `Capturing CUDA graphs …`
4. `Application startup complete.`

**Readiness is the port probe**, not the markers: `GET /v1/models` on the
model's own port answering 200 with its id in the list. A model that skips a
marker (no CUDA graphs, a cached compile) still becomes ready — treating a
missing marker as "not ready" would be an instrument reporting on itself.

A boot that ends without marker 1 never loaded weights; without marker 2 the KV
cache did not fit. Both are in the last 40 journal lines that a failed
`StartResult` carries.

---

## A switch refuses with `vram_not_released`

`switch` stops the old model, then waits for `nvidia-smi` to report **80 %** of
what that unit's cgroup was holding back, up to 120 s, before booting the next
one. The driver frees a context asynchronously; a switch that trusted the
stop's exit code would launch a 90 GiB model into a card that still has the
last one in it, and the failure would surface minutes later as a CUDA OOM with
no obvious cause.

Red here means memory genuinely did not come back. `nvidia-smi` — look for a
process that is not in any `model-*` cgroup.

Note the accounting is over the **whole unit cgroup**, not `MainPID`: in vLLM
v1 the API server is `MainPID` and holds nothing, while the engine-core and
worker children hold all of it.

---

## The dashboard says a unit is running that it does not recognise

`/api/state` lists it under `unknown_units` and `servedeck status` prints a
warning. A `model-*` unit whose key is not in `models.toml` is **reported,
never acted on**: without a spec there is no slot, no port and no way to tell a
stray from a model. Either add it to the registry, or
`systemctl --user stop model-<key>`.

---

## `POST /api/…` comes back 409

The snapshot could already answer. `reason` says which:

- `busy` — a mutation is in flight. One lock serialises them.
- `already_live` — a unit for that key exists. Stop it first.
- `main_slot_busy` — use `POST /api/switch/<key>` instead of `start`.
- `not_main_slot` — `switch` only replaces the main slot; residents use `start`.
- `not_live` — nothing to stop.

Every one of these is re-checked inside `Control` against reality; the
pre-check only exists so the common refusals come back on the POST rather than
as an SSE notice you had to be watching for.
