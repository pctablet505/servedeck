# Troubleshooting

## The page loads unstyled

Hard-refresh (`Ctrl+Shift+R`). A stylesheet served once with the wrong
content type stays cached.

## "No models found"

Servedeck scans `model_cache` for `models--*` directories. Check the path:

```bash
python -c "from servedeck import config; print(config.get().model_cache)"
```

A model needs `config.json` and at least one `.safetensors` file. GGUF-only
checkpoints are listed as unservable — vLLM cannot load them.

## A model shows "unservable"

Its `architectures[0]` matches no backend. Add it:

```toml
architectures = ["Qwen3ForCausalLM", "YourArchHere"]
```

Find the value with:

```bash
python -c "import json;print(json.load(open('<snapshot>/config.json'))['architectures'])"
```

## Capacity says "cannot compute"

GPU detection failed. Check `nvidia-smi` works, or set `gpu_total_mib`
manually.

## Every config is refused for VRAM

Your server is probably already running and holding the card. Servedeck
discounts VRAM held by *its own* backend, identified by the process tree of
whatever owns the configured port. If you started the server another way, it
may not be attributed.

## Predictions are optimistic

Raise `overhead_gib`. Servedeck also learns: after each successful boot it
records the real weights and KV size, and later estimates for that model are
labelled **measured** instead of **estimated**.

## Did it crash, or did it never start?

This decides whether restarting helps.

```bash
grep -nE "ValueError|RuntimeError|CUDA error|out of memory" <your-log> | tail -5
```

- **Crashed while serving** → a restart is likely to recover it.
- **Failed to boot** → restarting loops forever. Read the error first.

Servedeck makes the same distinction: it only auto-restarts a server that had
reached a serving state.

## KV usage jumps between 0% and some number

That's correct. `kv_cache_usage_perc` is occupancy of **in-flight** requests;
it must fall to 0 when idle. For "is caching working", look at the prefix
cache hit rate instead.

## The GPU is shared with your desktop

If a compositor runs on the same card, it needs VRAM too. `nvidia-smi` lists
it as a type `G` process — note that `--query-compute-apps` does **not** show
it, which makes it easy to conclude wrongly that the card is free.

Leave headroom, and prefer the lowest utilization that still fits the context
you need.
