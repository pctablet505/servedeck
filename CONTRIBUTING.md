# Contributing

## Ground rules

**Never use `pgrep -f` or `pkill -f`.** They match the calling process's own
command line. A test (`tests/test_procctl_no_pattern_kill.py`) greps the
package and fails on any occurrence. Use `ps -eo comm`, explicit PIDs, or
`procctl`.

**Never hardcode a path, GPU size, or model name.** Everything
machine-specific belongs in `servedeck/config.py`.

**Never run `sudo`.** Where privilege is needed, emit a copyable command for
the user to run.

**Every fix ships a test that fails before it.** Two of the bugs fixed in the
first review had been silently broken for the life of the module because
nothing exercised them.

## Numbers

Anything presented as measured must come from a real log or a live probe.
If it's derived, label it estimated. The UI distinguishes them because
predictions here have run optimistic before.

## Tests

```bash
.venv/bin/python -m pytest
```

Fixtures under `tests/fixtures/` are snapshots. Never point a test at a live
log: the server rewrites those on restart, and the test breaks for reasons
unrelated to the change.

## Quick reference (moved from the README)

These sections were on the front page until October 2026; they are kept here word for word.

### The five commands

```bash
servedeck status              # what is live: slot, ctx, KV %, run/wait, tok/s, uptime, headroom
servedeck switch flashnext    # put this model in the main slot, evicting whatever is there
servedeck start lfm2          # start (or stop) one model, streaming its boot log until READY
servedeck doctor              # prove the registry against reality before believing anything else
servedeck wire --apply        # rewrite the client configs (no flag = dry-run diff)
```

`start`, `stop`, `switch`, `log` and `adopt` drive the running dashboard over HTTP and fall back to the same
supervisor in-process when it is down, saying which on the first line. `models`, `adopt`, `log` and `smoke` also
exist. None of the verbs take tuning flags: `switch flashnext` launches at the `util` and context that
`models.toml` pins. To launch at other values, use the page's **Configure** panel.

The control plane is a **user** unit: `systemctl --user start servedeck` (`restart` after changing code or
`models.toml`), never `sudo`. Restarting it does not disturb a running model: engines are separate units and this
one sets `KillMode=process`.

### Logs

```bash
journalctl --user -u servedeck -f          # servedeck itself, including its own logger
journalctl --user -u model-flashnext -f    # one model's engine (unit is model-<key>)
servedeck log flashnext -n 200             # the same journal, through the CLI
```

### When something is wrong

Run `servedeck doctor`. Besides the registry it checks the host rows that explain most surprises:
`kernel.yama.ptrace_scope` (must be 0, or the KV-offload reaper cannot see a sibling engine and Flash-Next's PLE
handoff fails), desired state naming a model that is not running, weights on disk, leaked
`/dev/shm/vllm_offload_*.mmap` buffers at 40 GiB each, and the GPU power cap. Playbooks:
[Recovery](https://servedeck.rlalphalabs.com/operations/#recovery).

### Run the tests

`uv venv .venv && uv pip install -e '.[dev]'`, then the everyday run, which touches nothing on the box:

```bash
.venv/bin/python -m pytest -q -n 12 \
  --ignore=tests/test_e2e_real.py --ignore=tests/test_app_e2e.py \
  --ignore=tests/test_control_e2e.py --ignore=tests/test_gateway_e2e.py \
  --ignore=tests/test_cli_control.py
.venv/bin/python -m pytest -q -n 0 tests/test_cli_control.py
```

`test_cli_control.py` runs on its own because it binds a fixed port; under `-n 12` its cases contend with each
other and fail with `httpx.ConnectError` (10 of 16 on a clean tree). Nothing is skipped by splitting it out.

The four excluded files are the gate. Three drive the real `systemd-run` path, safely, under the `sd-test-` unit
namespace, which cannot name a real `model-*` unit; they bind fixed ports, so run them one at a time (`-n 0`).
`test_e2e_real.py` boots real models on the GPU and needs the card to have room. Run the gate on an idle card, not
as part of a normal check. How to contribute: [CONTRIBUTING.md](CONTRIBUTING.md).

| To | Read |
| --- | --- |
| Use it | [Operations](https://servedeck.rlalphalabs.com/operations/) · [Clients](https://servedeck.rlalphalabs.com/clients/) · [Configuration](https://servedeck.rlalphalabs.com/configuration/) |
| How it is built | [Architecture](https://servedeck.rlalphalabs.com/architecture/) · [Builds](https://servedeck.rlalphalabs.com/builds/) · [Host](https://servedeck.rlalphalabs.com/host/) · [Decisions](https://servedeck.rlalphalabs.com/decisions/) |
| Each model, measured | [Flash-Next](https://servedeck.rlalphalabs.com/models/flashnext/) · [Qwen3.8-27B](https://servedeck.rlalphalabs.com/models/qwen27b/) · [GLM-5.3](https://servedeck.rlalphalabs.com/models/glm53/) · [LFM2.5-350M](https://servedeck.rlalphalabs.com/models/lfm2/) |
