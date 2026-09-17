# Capacity — measured numbers and how to size

Every figure here came from a real boot log or a live probe on this workstation. Where a number
is an estimate it says so. **Estimates on this box have run ~25% optimistic more than once** —
treat them as upper bounds, not budgets.

Hardware: NVIDIA RTX PRO 6000 Blackwell, **97,887 MiB** (95.59 GiB), 182 GB host RAM.

---

## The arithmetic

```
budget_gib    = util × 95.59
kv_gib        = budget_gib − weights_gib − overhead_gib
kv_tokens     = kv_gib × 1048576 ÷ kv_kib_per_token
agents_at_ctx = floor(kv_tokens ÷ context_per_agent)
```

`overhead_gib` is activations + CUDA graphs: **measured 4.47** (Flash-Next) and **4.33** (27B).
Use 4.7 when unmeasured — it errs low, which is the safe direction.

This reproduces reality to within 0.02%:

| | Flash-Next | 27B NVFP4 |
|---|---|---|
| util | 0.96 | 0.50 |
| weights | 78.47 GiB | 20.75 GiB |
| KV | 8.83 GiB | 22.72 GiB |
| predicted tokens | 304,637 | 627,003 |
| **vLLM printed** | **304,653** | **627,117** |
| error | 0.005% | 0.018% |
| concurrency | 1.16x | 2.39x |

---

## Per-model measured data

| model | backend | weights GiB | KV KiB/token | decode tok/s | trust |
|---|---|---|---|---|---|
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | flashnext | **78.47** | **30.39** @262k · **33.85** @131k | 99.3 | measured |
| `RadixArk/Qwen3.8-27B-NVFP4` | inline | **20.75** | **37.99** | 140.4 | measured |
| `Qwen/Qwen3.8-27B-FP8` | inline | **28.51** | 37.99 (family) | — | weights measured |
| `twolven/…-abliterated-AWQ-MTP` | inline | 18.21 (safetensors) | ~35.4 | 173.8 | estimated |
| `orcarouter/…-Uncensored-FP8` | inline | 28.75 (est) | 37.99 (est) | — | estimated |
| `OBLITERATUS/…-OBLITERATED` | — | — | — | — | **unservable** |

**`OBLITERATUS/Qwen3.8-27B-OBLITERATED` cannot be served.** It ships one 20.89 GiB
`Q6_K.gguf` with no `config.json` and no safetensors. Neither vLLM build can load it.

### Two traps in this table

**KV cost is not context-invariant.** Flash-Next measures 30.39 KiB/token at
`max_model_len=262144` but **33.85 at 131072** — vLLM's block sizing depends on the configured
length. Don't reuse a rate measured at a different context.

**On-disk size ≠ loaded weights.** Flash-Next is 125.91 GiB on disk but **78.47 GiB in VRAM**,
because the ~51 GB n-gram (PLE) table is offloaded to *host* RAM. A `disk × 1.01` estimator is
accurate for ordinary checkpoints (20.42→20.75, +1.6%) and **37% wrong** for this architecture.

---

## 262,144 is a model ceiling, not a memory limit

Flash-Next's `config.json` declares `max_position_embeddings: 262144` with
`rope_type: "default"` — no YaRN, no scaling. **Freeing memory buys concurrency, never more
context.**

Verified working at full length: needle retrieval from a 220k-token prompt at 10%, 50%, and
93% depth — **3/3 correct, uncached**. Prefill ~9,300–10,400 tok/s.

---

## Sizing for agents: two numbers, not one

**Safe floor** — every agent at full context. A guarantee:
```
agents = kv_tokens ÷ max_model_len
```

**At observed average** — a bet, and usually several times larger:
```
agents = kv_tokens ÷ (vllm:request_prompt_tokens_sum ÷ …_count)
```

Worked example, 27B NVFP4 at util 0.92 (KV 62.9 GiB = 1,735,170 tokens):

| context per agent | agents |
|---|---|
| 262,144 (full) | **6.6** |
| 100,000 | **17.4** |

Sizing for worst case leaves most of the card unused. But the average is a bet: a burst of
full-context requests will preempt. Watch `num_preemptions_total` — if it climbs, you set the
count too high. **That empirical signal beats any estimate.**

> `CODEX_MAX_SUBAGENTS` is a **Codex-side orchestration cap, not a GPU guarantee.** Setting 64
> against a backend serving 1.16x means 63 requests queue.

---

## Choosing a configuration

| goal | setting | result |
|---|---|---|
| Max context, one agent | `MAX_LEN=262144 MAX_SEQS=1 GPU_UTIL=0.95` | 262k, 1.04x — **current default** |
| Several agents | `MAX_LEN=65536 MAX_SEQS=4` | ~4 concurrent at 64k |
| Many agents | switch to 27B NVFP4 | ~7 at full context, 140 tok/s |

Flash-Next costs roughly **8× the concurrency** of the 27B and is slower per token. That is
inherent to 78.5 GiB of weights plus a host-resident n-gram table — not a tuning failure.
Choose deliberately.

---

## Blue-green (two instances at once)

Needs `2 × weights + 2 × overhead + 1 GiB` ≤ 95.59 GiB **and** enough KV left for both sides.

| model | 2 instances | verdict |
|---|---|---|
| Flash-Next | 166.3 GiB | **impossible** — 1.74× the card |
| 27B FP8 | 66.4 GiB | fits on weights, but **KV does not**: bf16 KV (~72 KiB/token, no `kv_cache_scheme` declared) leaves ~197k tokens/side against a 262k `max_model_len` — the engine refuses to start |
| 27B NVFP4 | 50.9 GiB | **feasible** — 2 × 0.47 util, the shipped default |
| AWQ-MTP | 45.8 GiB | **feasible** |

So zero-downtime model swaps are available on the 27B family and **not** on Flash-Next.

---

## Host RAM is a real constraint

182 GB total, with **~49–59 GB held by Flash-Next's PLE offload** while running. On 2026-08-27
a second model load reached 103 GiB of anon-rss and the kernel OOM-killer fired mid-request.
The server survived; the machine thrashed (93 → 2.8 tok/s) until memory was reclaimed.

**Never load a second model while the server is running.** Check `free -g` first.

---

## Why the default is 0.95, not 0.96

`0.95` is the **lowest utilization that still fits full 262,144 context** — 0.945 yields
255,162 KV tokens and the engine refuses to start. Measured at 0.95:

```
Available KV cache memory: 7.88 GiB
GPU KV cache size:         272,062 tokens
Maximum concurrency:       1.04x
free VRAM:                 5,641 MiB   (was 2,731 at util 0.96)
```

The extra ~2.9 GiB goes back to the desktop. **This GPU is shared with the display** —
`gnome-shell` appears in `nvidia-smi` as a type `G` process on `card1` (nvidia-drm), alongside
`card2` (amdgpu). An earlier note in this project claimed "no display attached"; that was
derived from `nvidia-smi --query-compute-apps`, which never lists the compositor. It was wrong.

> Raising utilization above 0.95 buys **concurrency only** — the context ceiling is the model's,
> not the card's. On a machine whose desktop shares this GPU, that is rarely the right trade.
