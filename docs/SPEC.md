# Servedeck v2 — interface spec

What servedeck serves, and the exact shape of every answer. The *why* for all
of it is [REDESIGN-2026-09-12.md](REDESIGN-2026-09-12.md); this file does not
restate it.

Everything binds `127.0.0.1:8010` by default. There is no authentication, so do
not expose it.

Route order in `create_app` is load-bearing: `/api/*`, then the gateway
(`/v1/{path:path}` is a catch-all), then the page (`/{asset:path}` is the
second). See ARCHITECTURE.md.

---

## 1. The gateway — `/v1/*`

`servedeck/gateway.py`. One URL for every client and every model.

| Route | Methods | Behaviour |
|---|---|---|
| `GET /v1/models` | GET | The union of every **ready** model's served names. |
| `/v1/{path:path}` | GET POST PUT PATCH DELETE HEAD OPTIONS | Resolve `model` → port, apply that route's policies, stream both ways. |
| `/health` `/metrics` `/tokenize` `/detokenize` | same set | Routed the same way; no `model` in the body means the main slot. |

**Routing.** The request body is read only far enough to resolve the `model`
field (`policies.scan_model`, ceiling 8 MiB, cheap `b'"model"'` pre-filter);
the remainder is streamed. A name may be an id, an alias, a preset, or a
registry key. If it is an alias or preset, `model` is rewritten to the id vLLM
was actually started with — otherwise upstream 404s for a model it is serving.
A request that names no model goes to the main slot.

**Policies**, selected per route from the registry, all off by default:

| Policy | From | Effect |
|---|---|---|
| `mirror_reasoning` | `reasoning.mirror_content` | Copies `reasoning` ↔ `reasoning_content` in the JSON body and every SSE delta. Never applied to `/v1/responses`. |
| `effort_overlay` | the *preset's* table | Merges into `chat_template_kwargs` with `setdefault` — an explicit caller value always wins. |
| `min_output_tokens` | the model's field | Raises an **explicitly small** output budget (`max_tokens`, `max_completion_tokens`, `max_output_tokens`) to the floor, clamped to `ctx - prompt - 2048`. Never lowers one; never invents an absent one. |
| `ctx` | resolved context | Reported as `max_model_len`; the clamp for the floor. |

Streaming is decided by upstream's `Content-Type`, never by `stream: true`, so
a server that answers a streaming request with a JSON error is handled by the
JSON path automatically. A body is held in full only when a policy must rewrite
it.

### `GET /v1/models`

```json
{"object": "list", "data": [
  {"id": "qwen38-flash-next", "object": "model", "created": 1757664000,
   "owned_by": "servedeck", "root": "Qwen3.8-Flash-Next-Uncensored-NVFP4",
   "parent": null, "max_model_len": 262144}
]}
```

One entry per served name (id, every alias, every preset) of every **ready**
model. `root` is the id, which is what tells a client that three entries are
the same weights. `max_model_len` is `0` when `ctx = "native"` and the
checkpoint is not in the local hub cache — never a guess.

### Gateway refusals

All OpenAI-shaped, because every client already parses that.

| Status | When | Body |
|---|---|---|
| 404 | The name is in no registry entry. The client is misconfigured; waiting will not help. | `{"error": {"message": "The model 'X' does not exist. Known models: …", "type": "invalid_request_error", "param": "model", "code": "model_not_found"}}` |
| 503 | Registered, but not ready — booting, or something else holds the main slot. Header `Retry-After: 15`. | `{"error": {"message": "X is not running (main slot: Y, still starting)", "type": "model_not_running", "code": "not_running"}}` |
| 502 | The route table says live and the socket says otherwise. A disagreement between servedeck and reality, not a client error — so never dressed up as a 503 a client will silently retry. | `{"error": {"message": "upstream http://127.0.0.1:8004 for X is unreachable: …", "type": "upstream_unavailable", "code": "upstream_unavailable"}}` |

---

## 2. The control API — `/api/*`

`servedeck/app.py`, `_register_api`.

| Route | Method | Success |
|---|---|---|
| `/api/health` | GET | 200 `{"ok": true, "service": "servedeck", "port": 8010}` |
| `/api/state` | GET | 200 — the state document (§3) |
| `/api/models` | GET | 200 `{"models": [...], "cache": [...]}` |
| `/api/models/{key}/start` | POST | 202 accepted |
| `/api/models/{key}/stop` | POST | 202 accepted |
| `/api/switch/{key}` | POST | 202 accepted |
| `/api/adopt` | POST | 202 accepted (never refused) |
| `/api/log/{key}` | GET | 200 `{"key", "unit", "lines": [str]}` |
| `/api/wire` | GET | 200 — the wire payload, dry run |
| `/api/wire/apply` | POST | 200 — the wire payload, written |
| `/api/doctor` | GET | 200 `{"ok": bool, "checks": [{"name", "ok", "detail"}]}` |
| `/api/events` | GET | 200 `text/event-stream` (§4) |
| `/` and `/{asset:path}` | GET | The page and `servedeck/web/`, by explicit name. |

`/api/health` is deliberately the cheapest possible answer: no systemd, no GPU,
no registry walk. `_reconcile_after_bind` polls it to learn whether we won the
port, so a health check that could itself fail would make the reconcile
decision depend on something other than the bind.

### Accepted — `_accepted`, 202

```json
{"accepted": true, "action": "start", "model": "flashnext"}
```

The mutation runs in a worker thread; progress arrives on `/api/events`.

### Refused — `_refusal`

```json
{"error": {"reason": "main_slot_busy", "message": "…", "live_key": "qwen27b"}}
```

`reason` is the machine-readable half and uses exactly the vocabulary
`control.Refusal` uses, so a refusal decided here from the snapshot and one
decided in `Control` against reality are indistinguishable to a client.

`_precheck` answers whatever can be answered without a subprocess:

| Status | `reason` | Extra fields | When |
|---|---|---|---|
| 404 | `unknown_model` | `known: [keys]` | No such key in `models.toml`. |
| 409 | `busy` | `busy: <label>` | A mutation is already running. |
| 409 | `already_live` | `live_key` | `start`/`switch` and a unit for that key exists. |
| 409 | `main_slot_busy` | `live_key` | `start` on a `main` model while another holds the slot. |
| 409 | `not_main_slot` | — | `switch` on a `resident` model. |
| 409 | `not_live` | `live_key: null` | `stop` and no unit exists. |
| 409 | `ctx_unresolved` | — | `start`/`switch` on a model whose `ctx = "native"` could not be read (the checkpoint is not in the local hub cache). Refused here rather than launching with a context length nobody measured. |
| 404 | `unknown_model` | — | `/api/log/{key}` for an unregistered key. |

Reasons that can only be decided against reality arrive as an SSE `notice`
instead: `not_enough_vram`, `gpu_unavailable`, `no_vram_budget`,
`start_failed`, `stop_failed`, `vram_not_released`, `bad_key`, `boot_failed`,
`exception`.

### `/api/models`

The registry joined with what is actually on disk, so the main-slot dropdown is
not one round trip away from knowing whether a model would have to download
90 GiB first.

```json
{"models": [{"key", "id", "repo", "slot", "build",
             "on_disk": bool, "disk_gib": float|null, "reason": str}],
 "cache":  [{"repo_id", "servable": bool, "disk_gib": float,
             "arch": str|null, "in_registry": bool}]}
```

### `/api/wire` and `/api/wire/apply`

```json
{"applied": false,
 "targets": [{"name": "vscode chatLanguageModels.json", "path": "…",
              "changed": true, "diff": "--- …", "backup": null}]}
```

Same code path either way — the diff shown is literally the diff applied, which
is the property that makes an Apply button trustworthy. `backup` is set only
when a write happened: `<state_dir>/backups/<YYYY-MM-DD>/<flattened path>`.

---

## 3. The `/api/state` document

Rebuilt by the poller every 2 s and published on change; `GET /api/state`
returns the last one. Every field is a registry fact, a systemd fact, or a
number the running engine published about itself. Nothing is estimated.

```json
{
  "generated_at": 1757664000.0,
  "gateway_url": "http://127.0.0.1:8010/v1",
  "unit_prefix": "model-",
  "gpu": {"total_mib": 97887, "free_mib": 94100},
  "desired": {"main": "flashnext", "residents": ["lfm2"]},
  "busy": null,
  "models": [ … ],
  "unknown_units": [{"key", "unit", "unit_state"}],
  "headroom": { … }
}
```

`busy` is the label of the mutation in flight (`"switch flashnext"`) or `null`.

**`models[]`** — one row per registry entry, running or not:

| Field | Meaning |
|---|---|
| `key` `id` `aliases` `presets` `slot` `port` `build` `repo` `vram_mib` `needs_tty` | Straight from the registry. |
| `ctx` | Resolved context in tokens, or `null` when unknown. |
| `ctx_error` | Why it is unknown (`ctx = "native"`, checkpoint not in the hub cache), else `null`. |
| `live` | A unit exists. |
| `ready` | The port probe answers with this model's id. |
| `unit` | `model-<key>`. |
| `unit_state` | `"active (running)"`, or `"not started"`. |
| `restarts` `pid` | From `systemctl show`. |
| `uptime_s` | Seconds since `ExecMainStartTimestampMonotonic`, or `null`. |
| `metrics` | `null` unless ready; otherwise `reachable`, `running`, `waiting`, `kv_usage_perc`, `gen_tok_s`, `gen_tok_s_avg`, `gen_state`, `kv_cache_size_tokens`, `error`. |

**`headroom`** — how much more work the running engine can take:

`free_mib`, `model`, `key`, `full_ctx`, `pool_tokens`,
`full_context_requests`, `full_cost_tokens`, `small_request_tokens` (4096),
`small_requests`, `small_cost_tokens`, `fixed_cost_tokens`,
`headroom_fraction`, `source`, `unavailable`, `note`.

The pool is `vllm:cache_config_info`'s `kv_cache_size_tokens` — the engine's
own resolved KV capacity, the same number its boot log prints as "GPU KV cache
size". When it is absent, `unavailable` carries the reason and every derived
figure is `null`: an estimate presented beside a measurement is
indistinguishable from it on screen.

---

## 4. `/api/events` — SSE

`text/event-stream`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`. On
connect the stream replays the current state and the last 50 notices, so a page
opened after everything interesting happened is not blank.

Frames are `event: <type>\ndata: <json>\n\n`. Four kinds:

| Type | `data` |
|---|---|
| `state` | The whole §3 document. Published only when it *changes* — `generated_at`, `uptime_s` and the four volatile metric fields are excluded from the comparison, or "on change" would mean "every two seconds". |
| `progress` | `{"key", "kind", "text", "marker_index", "elapsed_s"}`. `kind` is `line`, `marker`, `ready` or `failed`. `key` is the model's key, or `"reconcile"` at startup. |
| `notice` | `{"level": "info"\|"error", "reason", "message", …}`. |
| keepalive | Not an event: the comment line `: keepalive`, every 15 s. |

`notice.reason` values and their extra fields:

- `bind_timeout` — our port never answered; nothing was reconciled.
- `reconciling` — `desired: {main, residents}`.
- `reconciled` — `already_live`, `booting`, `started`, `refused[{key, reason, message}]`.
- `started` — a mutation began.
- `ready` / `boot_failed` — `key`, `markers`, `journal` (last 40 lines).
- `stopped` — `key`.
- `adopted` — `adopted`, `unknown_units`.
- `wired` — which targets changed.
- `exception` — the mutation raised.
- any `control.Refusal` reason — `key`, `live_key`.

Every subscriber is handed a `None` sentinel at shutdown so its generator
returns; a per-subscriber queue of 256 drops its **oldest** event on overflow.

---

## 5. `state/desired.json`

`servedeck/desired.py`. The one thing servedeck remembers across restarts.

```json
{"version": 2, "main": "flashnext", "residents": ["lfm2"]}
```

That is the whole schema. `main` is a registry key or `null`; `residents` is a
list of keys, deduplicated in order.

It is **not** a mirror of what is running — it is what an operator last
explicitly asked for. Written only by `control.start` / `stop` / `switch` /
`adopt`, and written atomically (same-directory temp file, `fsync`,
`os.replace`). `reconcile` never writes it: a model that crashed must stay
desired so reconcile brings it back, and a model an operator stopped must stay
stopped even though adopting a stray unit would otherwise silently re-desire
it.

A missing, empty, unparseable or unknown-version file yields `Desired()` —
"want nothing" — with a warning, never an exception. A v1 file
(`{"version": 1, "desired_state": "RUNNING", "backend": "inline", …}`) is read
as `main = backend if desired_state == "RUNNING" else None`, no residents, and
a warning: v1's `backend` was a launcher name, not necessarily a v2 registry
key.

---

## 6. Unit lifecycle

A model is launched as:

```
systemd-run --user --unit=model-<key> --collect
            --description="servedeck model <id> (<key>)"
            -p Restart=on-failure -p RestartSec=10
            -p WorkingDirectory=<build root>
            --setenv=K=V …
            -- <build>/.venv/bin/vllm serve <repo>
               --served-model-name <id> <alias…> <preset…>
               --host 127.0.0.1
               [--enable-auto-tool-choice --tool-call-parser P]
               [--reasoning-parser P]
               --max-model-len N --gpu-memory-utilization U --port P
               <flags…>
```

Never `shell=True`; `--` terminates option parsing. Only unit names matching
`^model-[a-z0-9-]+$` or `^sd-test-[a-z0-9-]+$` are accepted, so no code path
here can reach a unit somebody else owns.

Readiness is `GET http://127.0.0.1:<port>/v1/models` answering 200 with one of
the model's served names in the list. The four journal markers
(`Loading weights took`, `GPU KV cache size:`, `Capturing CUDA graphs`,
`Application startup complete.`) drive the progress bar only.

A start fails when the unit goes to `failed`, restarts at least once while
booting, or **vanishes** — `--collect` deletes a failed unit, after which
`systemctl show` reports property defaults that look exactly like a clean stop.
See TROUBLESHOOTING.md.
