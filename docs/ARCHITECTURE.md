# Architecture

One process: a FastAPI app serving a static page, a JSON API, an SSE stream,
and a pass-through proxy. It shells out to your launcher; it never builds a
model-server command line.

```
browser ──► servedeck (127.0.0.1:8010)
              ├── web/            static dashboard
              ├── /api/*          state, models, capacity, events
              └── /v1/*           proxied to your model server
                        │
                        └──► your launcher ──► model server
```

## Modules

| Module | Responsibility |
|---|---|
| `config` | every machine-specific value; nothing else may hardcode one |
| `capacity` | the VRAM arithmetic. Pure — no I/O, no subprocess |
| `registry` | model discovery, and the measurement store |
| `kvcalc` | per-architecture KV cache arithmetic from a checkpoint's `config.json` |
| `metrics` | scrapes Prometheus `/metrics` |
| `phases` | boot-phase detection and failure classification from logs |
| `logtail` | follows a log across rotation and truncation |
| `procctl` | process control by PID and process group |
| `updetect` | which port the upstream is on, and why — see below |
| `supervisor` | intent state machine and auto-restart |
| `gateway` | holds requests while the backend restarts |
| `app` | HTTP surface |

## Which port the dashboard watches

Four sources, in this order, and `/api/state` says which one answered:

1. **A live process.** Every vLLM api-server process on the machine, found by
   walking `/proc` and reading the socket table — never a file. A process that
   is running is a fact; everything below it is somebody's intention.
2. **The shell config's `PORT`** (`local_llm/.config`), what the CLI last set up.
3. **`state/desired.json`**, what Servedeck last wanted.
4. **A configured backend's port**, as a last resort.

And one rule across all four: if the port a file names has nothing listening
while another known backend's port does, the dashboard follows the live one
and says so. The failure this replaces was silent — `.config` ended with
`BACKEND="glm53"` / `PORT="8002"`, GLM had been dead for a week, Flash-Next
was serving on :8001, and the dashboard reported the box as dead with every
figure blank and no indication of which port it had been looking at.

`upstream.resolution` in `/api/state` carries the chosen port, the source, a
sentence explaining it, and every candidate that was checked (with
`listening: null` meaning "not probed", never "nothing there").

## Where a KV figure comes from

Three sources, and the panel says which one it used:

1. **The running engine.** `vllm:cache_config_info` on `/metrics` carries
   `kv_cache_size_tokens` — the same figure the boot log prints once as
   "GPU KV cache size: N tokens". Labelled **measured**. Preferred whenever
   the model on screen is the one running, at the context it is running at.
2. **A recorded boot.** `state/measurements.json`, keyed on repo and context.
   Labelled **measured**, or **measured at another context** when the only
   record is from a different length.
3. **`kvcalc`.** Per-architecture arithmetic over the checkpoint's own
   `config.json`: attention K/V per layer, MLA latent, QSA compressed keys,
   and the Mamba/GDN recurrent state charged per sequence rather than per
   token. Labelled **estimated**.

`kvcalc` does not reimplement vLLM's block allocator. Each architecture family
carries one empirical correction for the padding it does not model, calibrated
against boots measured on the machine it runs on and shipped with the residual
it leaves — see the constants in `servedeck/kvcalc.py`, which name every boot
they were fitted to. A family with no such boot has a correction of 1.0 and
reports `calibrated: false`; the caller must present that as a floor.

## Two rules the design turns on

**Capacity has exactly one implementation.** The browser never computes it —
it calls the API. The estimate you see and the validation that blocks a start
are the same code path, so they cannot disagree.

**Intent is stored; state is computed.** `desired_state` is written to disk and
changed only by an explicit human action. `actual_state` is derived from
probes, never stored. This is what stops a supervisor from resurrecting a
server you deliberately stopped: it cannot tell "stopped" from "crashed" by
looking, so it doesn't try — it consults intent.

## Auto-restart

Restart only happens when **both**:

1. `desired_state == RUNNING`, and
2. the server had reached a serving state before it died.

The second condition is the important one. A boot that fails will fail the same
way forever, so restarting it hides the cause. A process that crashed while
serving is a different case, and worth retrying with backoff.

## Process control

`pgrep -f` and `pkill -f` match the calling process's own command line. A test
fails the build if either appears. `procctl` uses PIDs and process groups, and
refuses to signal a non-positive pgid — `killpg(0, …)` signals the caller's own
group, and `killpg(-1, …)` signals everything.
