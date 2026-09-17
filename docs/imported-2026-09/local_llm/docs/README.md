# Local LLM — documentation index

A local, offline LLM served by vLLM on one RTX PRO 6000, wired to Codex CLI and (optionally)
VS Code Copilot Chat.

**Current backend:** `qwen38-flash-next` on `http://localhost:8001`, 262,144-token context,
started via `codex-qwen.sh` with `BACKEND="flashnext"`.

---

## Start here

| I want to… | Read |
|---|---|
| Start, stop, or check the server | [RUNBOOK.md](RUNBOOK.md) |
| Diagnose a crash, OOM, or hang | [RUNBOOK.md § Incident triage](RUNBOOK.md) |
| Decide context length / how many agents | [CAPACITY.md](CAPACITY.md) |
| Wire up Codex or VS Code Copilot | [CLIENTS.md](CLIENTS.md) |
| Rebuild Flash-Next from source | [../../vllm-qwen38next/SETUP.md](../../vllm-qwen38next/SETUP.md) |
| Understand the 27B history and tuning | [../LOCAL_LLM_SETUP.md](../LOCAL_LLM_SETUP.md) |

---

## The 60-second version

```bash
cd ~/Projects/local_llm
./codex-qwen.sh status     # is it up?
./codex-qwen.sh qwen       # start it + launch Codex
./codex-qwen.sh stop       # release the GPU
```

Health check that actually proves the engine is alive (`/v1/models` returns 200 even when it
isn't):

```bash
curl -s -o /dev/null -w '%{http_code}\n' -m 5 http://localhost:8001/health
```

---

## Coldstart — the management UI

A web UI for model switching, memory allocation, live utilization, and restart supervision.

> **Status (2026-08-27): runnable, read-only.**
> The dashboard works and shows live data — model registry, GPU telemetry, KV utilization,
> average context, preemptions, and server-side capacity estimates that reproduce the real boot
> numbers to within 0.02%.
> **Server control (start / stop / restart) is NOT wired yet** — `supervisor.py` and `gateway.py`
> are not built, so those buttons render disabled rather than pretending. Start and stop the
> model from a terminal with `./codex-qwen.sh` or `vllm-qwen38next/serve.sh`.

When it is finished:

```bash
cd ~/Projects/coldstart
./setup.sh          # one-time: creates .venv-gui, installs deps
./run.sh            # foreground; serves http://127.0.0.1:8010
```

Then open **http://127.0.0.1:8010**.

Or run it as a background service:

```bash
systemctl --user enable --now coldstart
systemctl --user status coldstart
journalctl --user -u coldstart -f
```

It binds `127.0.0.1` only, makes no external network requests, and never runs `sudo`.

Design reference: `~/Projects/coldstart/SPEC.md`.

**Resolved:** the API module is named `coldstart/app.py`, matching what `run.sh` and the systemd
unit already import.

---

## Hard-won facts worth knowing before you touch anything

- **`pgrep -f` / `pkill -f` match your own shell** on this box. They have caused false positives
  and killed the calling shell. Use `ps -eo comm` or explicit PIDs.
- **`/v1/models` returns 200 with a dead engine.** Use `/health`, which returns 503 on
  `EngineDeadError`.
- **Never load a second model.** 182 GB RAM, ~49–59 GB already held by Flash-Next's host-side
  n-gram table. A second load OOM-killed a 103 GiB process on 2026-08-27.
- **Flash-Next cannot restart unattended** — its startup needs a `sudo` sysctl for
  `ptrace_scope`, and `sudo -n` fails here.
- **262,144 is a model ceiling, not a memory limit.** More VRAM buys concurrency, never context.
- **Boot failures outnumber crashes** (44 of 72 recorded exits never reached serving). Read the
  error before restarting.
- **The watchdog is deliberately disabled** — it could not tell "stopped" from "crashed" and
  resurrected a server that had been stopped on purpose.
