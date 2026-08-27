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
log_path = "~/serve.log"       # default: <launcher dir>/<name>.log
venv     = "~/venvs/vllm"      # optional
architectures = ["Qwen3ForCausalLM", "LlamaForCausalLM"]
needs_tty = false
```

**`architectures`** lists what this backend can load, matched against
`architectures[0]` in a model's `config.json`. Models no backend claims are
shown but not startable, with the reason.

**`needs_tty = true`** means starting it requires a terminal — an interactive
`sudo` prompt, for example. Servedeck will not attempt an unattended restart;
it reports `blocked-needs-human` instead of looping.

### Passing settings to your launcher

Servedeck never builds a `vllm serve` command line. It sets environment
variables and runs your script, so your script keeps owning the flags.

```toml
[backends.vllm.env_map]
port          = "PORT"
max_model_len = "MAX_LEN"
util          = "GPU_UTIL"
max_num_seqs  = "MAX_SEQS"
served_name   = "SERVED_NAME"
```

Left side is Servedeck's setting name; right side is your variable. Your
launcher then reads them:

```bash
vllm serve "$MODEL" \
  --port "${PORT:-8000}" \
  --max-model-len "${MAX_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}"
```

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
training job needs the card.

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
