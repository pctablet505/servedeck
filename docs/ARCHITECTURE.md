# Architecture (v2)

One repository, one Python package, one long-running unit, one transient unit
per model, one endpoint for every client. The *why* for all of it is
[REDESIGN-2026-09-12.md](REDESIGN-2026-09-12.md); this file is the map.

```
clients ──► servedeck (127.0.0.1:8010)
              ├── /v1/*    gateway: alias/preset routing, reasoning mirror
              ├── /api/*   control: state, start/stop/switch, wire, doctor, SSE
              └── /        the page (servedeck/web/)
                      │
                      │ systemd-run --user --unit=model-<key> --collect
                      ▼
              model-<key>.service ──► vllm serve   (its own cgroup)
                      ▲
                      └── models.toml: the single source of truth
```

## Layering

```
settings  limits                 no servedeck imports; env + hardware only
     │
     ▼
models.py (registry)             models.toml -> Model, render_argv/render_env
     │
     ▼
routes.py (the join)             registry names  ×  control liveness
     │                 │
     ▼                 ▼
gateway.py         control.py    the /v1 proxy   |   the supervisor
 policies.py         units.py                    |   systemd wrappers
                     desired.py                  |   what an operator asked for
     └──────┬────────────┘
            ▼
          app.py                 the HTTP surface: /api/*, SSE, the page
            │
      ┌─────┴─────┐
      ▼           ▼
  web/app.js   cli.py            the page  |  the terminal
```

## Module map

| Module | Packet | Why it exists |
|---|---|---|
| `settings.py` | P4 | Where to listen, where `models.toml` is, which unit namespace we own. Nothing else. |
| `limits.py` | P4 | The four *hardware* numbers (VRAM total, overhead, frag margin, training markers). |
| `models.py` | P1 | Loads and validates `models.toml`; renders one model's `vllm serve` argv and env. |
| `wire.py` | P1 | Rewrites VS Code / Codex / Kimi configs from the registry, in place, with a diff. |
| `doctor.py` | P1 | Proves the registry against reality: configs, ports, units, host state. |
| `cli.py` | P1 | The `servedeck` console script; falls back to in-process `Control` when the server is down. |
| `gateway.py` | P2 | The `/v1` proxy: resolve `model` → port, stream both ways, 404/503/502 envelopes. |
| `policies.py` | P2 | Pure byte transforms: reasoning mirror, effort overlay, output floor, model rewrite, `scan_model`. |
| `units.py` | P3 | Thin `systemd-run` / `systemctl` / `journalctl` wrappers. No policy. |
| `control.py` | P3 | The supervisor: start, wait-ready, stop, switch, adopt, reconcile. |
| `desired.py` | P3 | `state/desired.json` — what an operator last explicitly asked for. |
| `gpu.py` | kept, extended by P3 | `nvidia-smi` total / free / per-pid usage. |
| `routes.py` | P4 | The one join between registry, gateway and control. |
| `app.py` | P4 | ASGI app: `/api/*`, the SSE hub, the poller, the page. |
| `__main__.py` | P4 | `python -m servedeck` — starts uvicorn. |
| `discovery.py` | kept, renamed by P4 | The local hub cache: what is downloaded, servable, and how big. Was `registry.py`, a name the model registry now owns. |
| `capacity.py` `kvcalc.py` `parallelism.py` | kept | VRAM / KV / concurrency arithmetic, as a library. |
| `metrics.py` `tokens.py` `reqstats.py` | kept | Scrape and difference vLLM's Prometheus counters. |
| `disksize.py` | kept | Deduplicated on-disk size of a hub snapshot. |

Gone in v2, and not coming back: `config.py`, `paths.py`, `supervisor.py`,
`procctl.py`, `phases.py`, `updetect.py`, `shellconfig.py`, `legacy.py`,
`history.py`, `logtail.py`, `preflight.py`, `smoke.py`, `events.py`.

## The dependency rules, and what each one buys

**`settings.py` imports nothing from `servedeck`.** It is imported by
everything, so any import of its own would be a cycle to route around. It also
never reads a TOML file on import — asking it for a model's port, flags or
context is the bug the redesign removes.

**`models.render_argv` is pure.** It takes `util`, `ctx_tokens` and `port` as
already-resolved values and does no I/O, so `tests/test_models_golden.py` can
compare its output byte-for-byte against a legacy launcher's dry run.
Resolving `ctx = "native"` needs the hub cache and is therefore a separate
call (`native_ctx`), made by the caller.

**Nothing in a request handler blocks on a subprocess.** `systemctl`,
`journalctl` and `nvidia-smi` all run under `asyncio.to_thread`; the request
path only ever reads the snapshot the poller left behind
(`RegistryRoutes.refresh` pushes it every `POLL_INTERVAL_S` = 2 s). A
coroutine waiting on `systemctl` stops the whole server — including the
gateway that is streaming a model's tokens.

**Mutations answer immediately.** `POST /api/models/<key>/start` returns 202
and reports over SSE; anything decidable from the snapshot (unknown key, busy,
slot held) is refused synchronously with 404/409 so a client never has to
watch a stream to learn it was rejected. One `asyncio.Lock` serialises
mutations, which is what makes "the main slot is exclusive" true of the API
and not only of the GPU.

**The gateway router is mounted LAST.** Its `/v1/{path:path}` is a catch-all
and Starlette matches in registration order, so any route registered after it
is dead. Order in `create_app`: `_register_api` → `include_router(gateway)` →
`_register_page` (whose `/{asset:path}` is the second catch-all and must be
last of all).

**Resolution is total; liveness is a snapshot.** A name spelled in
`models.toml` always resolves, even with nothing running — that is what
separates a 404 ("reconfigure yourself") from a 503 ("wait"). An
un-refreshed table reports nothing live, which fails towards 503.

## Two invariants worth stating outright

**Intent is stored; state is computed.** `state/desired.json` is written only
by an explicit `start`/`stop`/`switch`/`adopt`. `reconcile` never writes it and
never stops anything, so servedeck restarting is not a reason for a model to
restart.

**Capacity has one implementation.** The page does no arithmetic of its own;
it renders `/api/state`, whose headroom block is `parallelism.recommend()`
over the engine's own reported KV pool. When that pool is unknown the answer
is "unknown" with a reason, never an estimate presented beside a measurement.
