# Servedeck

One control plane for the local LLMs on one box: a registry, a supervisor, a
normalising gateway and a page.

- **One file describes every model** — `models.toml`. Names, aliases, port,
  context, parsers, flags, environment.
- **One URL for every client** — `http://127.0.0.1:8010/v1`. VS Code, Codex,
  Kimi, Claude Code and scripts all point at it, whichever model is loaded.
- **One command to run one** — `servedeck switch glm53`. The model runs as its
  own transient systemd unit, so restarting servedeck never touches it.

Binds loopback only, makes no external network requests, and never runs `sudo`.

---

## Requirements

- Linux with a user systemd instance (`systemctl --user`), Python 3.11+
- An NVIDIA GPU with `nvidia-smi` on `PATH`
- A vLLM venv per build, named in `[builds]` in `models.toml`

## Install

```bash
git clone https://github.com/pctablet505/servedeck && cd servedeck
uv venv .venv
uv pip install -e '.[dev]'
```

Then describe your models in `models.toml` — see
**[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** — and check it:

```bash
.venv/bin/servedeck models
.venv/bin/servedeck doctor
```

## Run the server

```bash
python -m servedeck                  # http://127.0.0.1:8010
python -m servedeck --no-reconcile   # serve, but start nothing from desired.json
```

Or as a unit, which is how it should run on a box you care about:

```bash
cp systemd/servedeck.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now servedeck
journalctl --user -u servedeck -f
```

The unit template assumes the checkout is at `~/Projects/servedeck`; edit
`WorkingDirectory` and `ExecStart` if it is not.

## The CLI

```
servedeck models              # the registry: key, id, slot, port, build, ctx
servedeck status              # what is live, KV usage, tok/s, uptime, headroom
servedeck start  <key>        # start a model and stream its boot
servedeck stop   <key>
servedeck switch <key>        # replace whatever holds the exclusive main slot
servedeck adopt               # record already-running units as desired
servedeck log    <key> -n 200 # tail its journal
servedeck smoke  <key>        # one chat + one tool call, through the gateway
servedeck wire [--apply]      # regenerate VS Code / Codex / Kimi configs
servedeck doctor              # prove the registry against reality
```

`start`, `stop`, `switch`, `log` and `adopt` drive the dashboard over HTTP when
it is up, and fall back to driving the same supervisor in-process when it is
not — saying which on the first line. A CLI whose only mode is "ask the thing
that is not running" fails exactly when it is needed.

`smoke` goes **through the gateway, on the model's public name**, never at the
model's own port. That is the point: what it proves is that the name a client
is configured with reaches the weights.

## What the page shows

Driven entirely by `/api/state`, with an SSE stream for boot progress.

- **Live models** — id and aliases, slot, port, context, KV usage, requests in
  flight, tok/s, uptime, restarts.
- **Headroom** — free VRAM, and how many full-context and 4k requests still fit
  in the engine's *own reported* KV pool. When that number is unknown the panel
  says so instead of estimating.
- **Controls** — start, stop, switch, wire, doctor. Each is one POST plus a
  stream of the unit's journal until READY or failure.

## Docs

- **[CONFIGURATION.md](docs/CONFIGURATION.md)** — `models.toml` and the five
  environment variables
- **[SPEC.md](docs/SPEC.md)** — every route, every response shape
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — the module map and the
  dependency rules
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** — the failures that
  actually happened, and what v2 does about them
- **[REDESIGN-2026-09-12.md](docs/REDESIGN-2026-09-12.md)** — why v2 looks like
  this

## Tests

```bash
.venv/bin/python -m pytest -q
```

The end-to-end tests drive the real `systemd-run` path under the `sd-test-`
unit namespace, so they can never create, adopt or stop a real model unit.

## License

MIT — see [LICENSE](LICENSE).
