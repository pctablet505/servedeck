# Configuration

Servedeck reads settings in this order — first match wins:

1. `SERVEDECK_*` environment variables
2. `servedeck.toml` (working directory, or `$SERVEDECK_CONFIG`)
3. Auto-detection
4. Defaults

Everything is optional except a backend.

---

## Backends

A backend is a script you already use to start a model server.

```toml
[backends.vllm]
launcher = "~/serve.sh"        # required
port     = 8000                # required
venv     = "~/venvs/vllm"      # optional
log_path = "~/serve.log"       # optional
writes_own_log = false         # optional
architectures = ["Qwen3ForCausalLM", "LlamaForCausalLM"]
needs_tty = false
```

**`architectures`** lists what this backend can load, matched against
`architectures[0]` in a model's `config.json`. Models no backend claims are
shown but not startable, with the reason. This is also how you teach Servedeck
a model family it does not know — no code change.

**`venv`** is checked before every start (a stale venv path is the largest
single boot-failure class there is) and is how Servedeck recognises a server
you started by hand, so it can adopt and supervise it.

**`log_path`** is where a *hand-started* run of this backend writes its log.
Servedeck reads it to recover an adopted server's boot numbers. **Leave it out
if your launcher has no fixed log** — Servedeck then falls back to the log it
opened itself, under `state/boot_logs/<backend>-*.log`, and to nothing at all
if there is none. Never point it at another backend's log: the phase machine
would match that model's error lines and file them as this backend's failure.

**`writes_own_log = true`** says the launcher redirects its *own* output into
`log_path` (`exec >> "$LOG" 2>&1`). Servedeck then tails that file during a
launch too, because the boot output stops arriving on the pipe it opened.
Leave it false when `log_path` is only a hand-launch convention — tailing it
during a fresh launch would replay a previous boot into the phase machine.

**`needs_tty = true`** means starting it requires a terminal — an interactive
`sudo` prompt, for example. Servedeck will not attempt an unattended restart;
it reports `blocked-needs-human` instead of looping.

### Passing settings to your launcher

Servedeck never builds a `vllm serve` command line. It sets environment
variables and runs your script, so your script keeps owning the flags.

```toml
[backends.vllm.env_map]
repo_id       = "MODEL"
port          = "PORT"
max_model_len = "MAX_LEN"
util          = "GPU_UTIL"
max_num_seqs  = "MAX_SEQS"
served_name   = "SERVED_NAME"
kv_dtype      = "KV_DTYPE"
```

Left side is Servedeck's setting name; right side is your variable. Your
launcher then reads them:

```bash
vllm serve "$MODEL" \
  --port "${PORT:-8000}" \
  --max-model-len "${MAX_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}"
```

Those seven are the whole set of settings Servedeck knows how to pass, and the
table above is also the default — omit `env_map` entirely and you get it. An
**empty** table is different from an omitted one: it means "this launcher reads
no environment", which is right for a launcher that takes its settings from a
file of its own.

`repo_id` matters more than it looks. Without it your launcher boots whatever
model its own `${MODEL:-...}` default names, no matter what was selected in the
UI.

### Machine-specific tuning

Anything else your launcher reads goes in a fixed `env` table, passed through
verbatim on every start:

```toml
[backends.vllm.env]
CPU_OFFLOAD_GB = "104"
KV_BYTES       = "8053063680"
```

Servedeck never interprets these — a knob it understands is a knob it can get
wrong. Two rules of thumb: put *sizing* knobs here, and leave *correctness*
knobs to your launcher's own defaults. A flag that decides which kernel gets
compiled is not something a dashboard should be able to override; if it is
wrong the server still loads, still serves, and quietly produces garbage.

---

## Hardware

```toml
gpu_total_mib = 24564    # detected via nvidia-smi if unset
overhead_gib  = 4.7      # VRAM that is neither weights nor KV
frag_margin_mib = 4096   # below this free, a config is "thin", not impossible
```

**`overhead_gib`** covers activations and CUDA graphs. Measured 4.3–4.5 GiB
across two very different models; 4.7 errs high so predictions stay
conservative. If yours come out optimistic, raise it.

**`frag_margin_mib`** is a warning threshold, never part of the requirement.
Adding it to the requirement makes any utilization above ~0.958 look
impossible on a card that runs 0.95 fine.

If GPU detection fails, `gpu_total_mib` is 0 and Servedeck refuses to compute
capacity rather than guessing.

---

## Paths

```toml
state_dir   = "./state"                       # desired state, history, measurements
model_cache = "~/.cache/huggingface/hub"      # from HF_HUB_CACHE / HF_HOME if unset
```

---

## Standing down for other GPU work

```toml
training_markers = ["~/run/training_in_progress"]
```

If any listed file exists, Servedeck refuses to start a server. Useful when a
training job needs the card. `~` is expanded; every listed path is checked on
each capacity estimate, and the ones that exist are named in the block reason
so you know what to remove.

`SERVEDECK_TRAINING_MARKERS` (colon-separated) overrides the list entirely,
for a one-off run.

---

## Serving

```toml
listen_host = "127.0.0.1"    # do not expose this; there is no auth
listen_port = 8010
```

---

## Environment variables

| Variable | Overrides |
|---|---|
| `SERVEDECK_CONFIG` | config file location |
| `SERVEDECK_STATE_DIR` | `state_dir` |
| `SERVEDECK_MODEL_CACHE` | `model_cache` |
| `SERVEDECK_GPU_TOTAL_MIB` | `gpu_total_mib` |
| `SERVEDECK_HOST` / `SERVEDECK_PORT` | listen address |
| `SERVEDECK_TRAINING_MARKERS` | `training_markers` (colon-separated) |
