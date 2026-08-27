# Architecture

One process: a FastAPI app serving a static page, a JSON API, an SSE stream,
and a pass-through proxy. It shells out to your launcher; it never builds a
model-server command line.

```
browser ──► coldstart (127.0.0.1:8010)
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
| `metrics` | scrapes Prometheus `/metrics` |
| `phases` | boot-phase detection and failure classification from logs |
| `logtail` | follows a log across rotation and truncation |
| `procctl` | process control by PID and process group |
| `supervisor` | intent state machine and auto-restart |
| `gateway` | holds requests while the backend restarts |
| `app` | HTTP surface |

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
