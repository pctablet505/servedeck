---
title: "Servedeck: a systemd control plane for local vLLM"
description: "Servedeck runs each local vLLM model as its own systemd user unit, serves one OpenAI-compatible /v1 endpoint, and shows live KV-cache use and throughput."
---

# Servedeck

Servedeck is a control plane for native vLLM on one Linux GPU workstation. It starts each model as its own `systemd-run --user` unit, serves one OpenAI-compatible `/v1` endpoint at `127.0.0.1:8010` that answers to the name `main`, and shows live KV-cache use and throughput on a local dashboard.

Restarting Servedeck never touches a running model. It needs no Docker and no `sudo`, and it makes no external network requests. It is MIT licensed and built by [RL Alpha Labs](https://rlalphalabs.com/); the source is on [GitHub](https://github.com/pctablet505/servedeck).

## What it does

- **Supervises models as systemd units.** `models.toml` is the registry. Each model is a transient unit `model-<key>`. Servedeck itself is a user unit with `KillMode=process`, so restarting it leaves the engines running. See [Architecture](ARCHITECTURE.md).
- **Gives every client one URL and one name.** Codex CLI, Kimi CLI and VS Code chat use `http://127.0.0.1:8010/v1` and ask for `main`, the alias for whichever model holds the main slot. `servedeck wire --apply` writes their configs. See [Clients](CLIENTS.md).
- **Shows capacity as it happens.** Decode and prefill tokens per second, time to first token, KV-cache use, requests held back by a full pool, preemptions, and the prompt-size histogram that the parallel-agent count is sized on. See [Operations](OPERATIONS.md).
- **Checks the host.** `servedeck doctor` compares the registry with reality: `kernel.yama.ptrace_scope`, leaked 40 GiB `/dev/shm/vllm_offload_*.mmap` buffers, weights on disk and the GPU power cap. See [the doctor rows](OPERATIONS.md#servedeck-doctor-row-by-row).

## The five commands

```bash
servedeck status              # what is live: slot, ctx, KV %, run/wait, tok/s, uptime, headroom
servedeck switch flashnext    # put this model in the main slot, evicting whatever is there
servedeck start lfm2          # start (or stop) one model, streaming its boot log until READY
servedeck doctor              # prove the registry against reality before believing anything else
servedeck wire --apply        # rewrite the client configs (no flag = dry-run diff)
```

## What it is not

- **Not zero-downtime.** A switch stops the model in the main slot, waits for the GPU to release its memory, then boots the next one. A measured Flash-Next boot took 137 seconds warm, and a cold boot was up to about 10 minutes when last measured. See [Operations](OPERATIONS.md).
- **Not an inference engine or a model downloader.** It supervises the vLLM builds you provide and checks that the weights are on disk. See [Builds](BUILDS.md).
- **Not multi-host and not authenticated.** It binds `127.0.0.1` only.
- **Not finished.** Version 0.1.0, alpha, built around one workstation. The docs are that machine's operating record: an RTX PRO 6000 Blackwell with 95.6 GiB, and a date on every measurement. Read the numbers as one machine's, not as a benchmark.

## Recovery and measurements

- A failed boot, an engine that died, a leaked 40 GiB `/dev/shm` buffer, a GPU that fell off the bus: the [recovery playbooks](OPERATIONS.md#recovery).
- Launch flags, capacity and measured throughput per model: [Flash-Next](models/flashnext.md), [Qwen3.8-27B](models/qwen27b.md), [GLM-5.3](models/glm53.md) and [LFM2.5-350M](models/lfm2.md).
- The machine itself: [power cap, Xid history, RAM, sysctl, PCIe and disk](HOST.md).
- Why it is built this way: [Decisions](DECISIONS.md).

## Run it

Servedeck is not on PyPI. Clone the repository, write a `models.toml` for your models and builds ([Configuration](CONFIGURATION.md), [Builds](BUILDS.md)), and install the shipped user unit, `systemd/servedeck.service`. The unit expects the checkout at `~/Projects/servedeck`; edit `WorkingDirectory` and `ExecStart` if you clone elsewhere. The steps are in [Operations](OPERATIONS.md#starting-servedeck-itself).
