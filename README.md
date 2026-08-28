# Servedeck

A local web dashboard for a self-hosted LLM server. It answers the questions
you actually have while running one:

- How much context can this model hold on my GPU?
- How many agents can I run in parallel before it starts thrashing?
- Is my prefix cache working?
- Did it crash, or did it never start?

It reads your GPU and your server's metrics. It does not replace your launch
script — it runs the one you already have.



---

## Why

Sizing a KV cache by hand is easy to get wrong, and wrong in an expensive
direction: you either waste half the card or discover at minute nine of a boot
that the context you asked for never fit.

Servedeck computes the same arithmetic vLLM does, before you start, and refuses
configurations that cannot work — with the reason and a suggested fix.

On the machine it was built for, its predictions match the engine's own
reported numbers to within **0.02%**.

---

## Requirements

- Linux, Python 3.11+
- An NVIDIA GPU with `nvidia-smi` on `PATH`
- A model server exposing an OpenAI-compatible API and Prometheus `/metrics`
  (vLLM, SGLang, or anything that speaks both)
- A script you already use to start it

---

## Install

```bash
git clone https://github.com/pctablet505/servedeck && cd servedeck
./setup.sh                    # creates .venv, installs deps
cp servedeck.toml.example servedeck.toml
$EDITOR servedeck.toml        # point it at your launcher
./run.sh                      # → http://127.0.0.1:8010
./stop.sh                     # stop it (or Ctrl-C in the terminal)
```

Nothing is installed system-wide. Servedeck binds `127.0.0.1` only, makes no
external network requests, and never runs `sudo`.

---

## Configure

The minimum is one backend — the launcher you already use:

```toml
[backends.vllm]
launcher = "~/serve.sh"
port = 8000
log_path = "~/serve.log"
architectures = ["Qwen3ForCausalLM", "LlamaForCausalLM"]
```

Servedeck passes settings to your launcher through environment variables, so
your script keeps owning the flags:

```toml
[backends.vllm.env_map]
port          = "PORT"
max_model_len = "MAX_LEN"
util          = "GPU_UTIL"
max_num_seqs  = "MAX_SEQS"
```

GPU size and model cache are auto-detected. Full reference:
**[docs/CONFIGURATION.md](docs/CONFIGURATION.md)**.

---

## What it shows

**Models** — everything in your Hugging Face cache, with the ones no backend
can load marked unservable and the reason why.

**Capacity** — for the model and settings you pick: KV cache size in tokens,
how many agents fit, and the largest context that will actually start. Numbers
measured from a real boot are labelled **measured**; the rest say **estimated**.

**Live utilization** — KV occupancy, running and queued requests, preemptions,
average context per request, and prefix cache hit rate.

**Agent sizing** — two numbers, never one:

| | meaning |
|---|---|
| Safe floor | every agent at full context — a guarantee |
| At observed average | `KV tokens ÷ measured average prompt` — a bet |

The second is usually several times larger, and it's a bet because one burst of
full-context requests will preempt. Watch `preemptions`: if it climbs, your
agent count is too high. That signal beats any estimate.

---

## Status

**Working:** the dashboard, capacity estimation, live metrics, model discovery,
start / stop / restart, and a pass-through proxy.

A server already running when Servedeck starts is deliberately left alone —
Servedeck does not assume a process it did not start is wanted. Click **Manage
running server** to adopt it; that is what enables Stop and crash detection
for it.

**Not wired yet:** the smoke-test button, and the request-holding gateway
(`gateway.py` exists and is tested, but is not in the request path).

---

## How capacity is computed

```
budget    = util × total_vram
kv        = budget − weights − overhead
kv_tokens = kv ÷ bytes_per_token
agents    = kv_tokens ÷ context_per_agent
```

Two things that trip people up, both handled:

**KV cost is not context-invariant.** The same model measured 30.4 KiB/token at
262k context and 33.9 at 131k — block sizing depends on the configured length.
Servedeck tracks the rate per context and says when it's reusing one measured
elsewhere.

**On-disk size is not loaded size.** A checkpoint with host-offloaded layers can
be 126 GB on disk and 78 GB in VRAM. Servedeck refuses to estimate weights for
architectures where that's known to be untrue, rather than being confidently
wrong by 37%.

---

## Docs

- **[CONFIGURATION.md](docs/CONFIGURATION.md)** — every setting
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** — when something breaks
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — how it's put together

---

## Tests

```bash
.venv/bin/python -m pytest
```

Capacity tests assert predictions match real engine output to within 0.2%,
replayed from boot logs committed under `tests/fixtures/`.

---

## License

MIT — see [LICENSE](LICENSE).
