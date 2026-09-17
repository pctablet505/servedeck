# Servedeck

One control plane for the local LLMs on this box. `models.toml` is the registry
(`qwen27b`, `flashnext`, `glm53`, `lfm2`); a user unit running `python -m servedeck`
serves a page, a JSON API and an OpenAI-compatible gateway on
**http://127.0.0.1:8010** (gateway under `/v1`); each model runs as its own
transient unit `model-<key>` started with `systemd-run`, so restarting servedeck
never touches a running engine. It replaced v1's `llm` shell launcher (now a shim
at `~/.local/bin/llm` that prints "servedeck owns this box" and exits 2), the two
reasoning-mirroring proxies on :8005/:8006, and hand-edited client configs.
Loopback only, no external network requests, never `sudo`.

Every client — Codex, Kimi CLI, VS Code chat — points at that `/v1` and asks for
the model named **`main`**, the gateway alias for whatever holds the main slot,
so nothing needs reconfiguring when the model changes.

## The five commands

```bash
servedeck status              # what is live: slot, ctx, KV %, run/wait, tok/s, uptime, headroom
servedeck switch flashnext    # put this model in the main slot, evicting whatever is there
servedeck start lfm2          # start (or stop) one model, streaming its boot log until READY
servedeck doctor              # prove the registry against reality before believing anything else
servedeck wire --apply        # rewrite the client configs (no flag = dry-run diff)
```

`start`, `stop`, `switch`, `log` and `adopt` drive the running dashboard over
HTTP and fall back to the same supervisor in-process when it is down, saying
which on the first line. `models`, `adopt`, `log` and `smoke` also exist — see
[docs/OPERATIONS.md](docs/OPERATIONS.md).

## The page

`http://127.0.0.1:8010` — v1's dashboard, served on the v2 backend via
`servedeck/legacy_page.py`: gateway URL and GPU memory in the header; what is
serving, with decode and prefill tok/s, time to first token, KV in use,
requests held back by a full pool, preemptions; the model list and machine
facts; a **Configure** panel (utilisation, max context, parallel agents, KV
offload, then Apply & restart or Stop); the prompt-size histogram the agent
count is sized on; traffic totals and a folded server log.

## Logs

```bash
journalctl --user -u servedeck -f          # servedeck itself, including its own logger
journalctl --user -u model-flashnext -f    # one model's engine (unit is model-<key>)
servedeck log flashnext -n 200             # the same journal, through the CLI
```

## When something is wrong

Run `servedeck doctor`. Besides the registry it checks the host rows that
explain most surprises: `kernel.yama.ptrace_scope` (must be 0, or the KV-offload
reaper cannot see a sibling engine and Flash-Next's PLE handoff fails), desired
state naming a model that is not running, weights on disk, leaked
`/dev/shm/vllm_offload_*.mmap` buffers at 40 GiB each, and the GPU power cap.
Playbooks: [docs/OPERATIONS.md](docs/OPERATIONS.md#recovery).

## The rest of the docs

| Doc | What it answers |
| --- | --- |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Daily use: switching, the page, reboot behaviour, logs, recovery |
| [docs/CLIENTS.md](docs/CLIENTS.md) | Codex, Kimi, VS Code wiring, `reasoning_content`, presets, adding a tool |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Registry, units, gateway, adoption, offload reaper, state, the poll |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | `models.toml` and `servedeck.toml`: every key and its default |
| [docs/HOST.md](docs/HOST.md) | GPU, power cap, Xid history, sysctl, RAM, /dev/shm, PCIe, disk |
| [docs/BUILDS.md](docs/BUILDS.md) | The three vLLM builds: forks, patches, venvs, rebuilding, drift checks |
| [docs/models/flashnext.md](docs/models/flashnext.md) | Flash-Next: flags and their reasons, measured performance, ceilings |
| [docs/models/qwen27b.md](docs/models/qwen27b.md) | Qwen3.8-27B: same |
| [docs/models/glm53.md](docs/models/glm53.md) | GLM-5.3: same |
| [docs/models/lfm2.md](docs/models/lfm2.md) | LFM2.5-350M: same |
| [docs/DECISIONS.md](docs/DECISIONS.md) | The owner's standing rules, dated, each with its reason |

Development: `uv venv .venv && uv pip install -e '.[dev]'`, then
`.venv/bin/python -m pytest -q`. The end-to-end tests drive the real
`systemd-run` path under the `sd-test-` unit namespace, so they can never
create, adopt or stop a real model unit. MIT — see [LICENSE](LICENSE).
