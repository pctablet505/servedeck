# LFM2.5-350M (`lfm2`)

A 350M-parameter resident that costs about 3 GiB of VRAM and boots in about a minute, kept for two
jobs: mechanical work that needs no intelligence (extraction, classification, structured output,
tool calls), and a real vLLM backend for end-to-end tests of servedeck itself. It is not for
knowledge or coding work — the model card says so and 2026-09-12 testing agreed. Its value is that
an experiment or a test no longer has to boot a 90 GiB model. It is also the only registry entry
with `slot = "resident"`: it co-resides with whatever holds the main slot instead of replacing it.

## Opt-in, and sticky once on

Owner directive, 2026-09-16: by default servedeck runs **one** model, and nothing auto-adds this
one. It is started explicitly (`servedeck start lfm2`, or the page's model card), and `desired.json`
on 2026-09-18 reads `residents: []`. A successful start writes the key into `residents`, so from
then on reconcile brings it back after a restart or reboot until it is stopped. Reconcile boots
`residents` **before** `main` — that is the order in which both fit; see the arithmetic below.

## Registry entry

| Key | Value | Why |
| --- | --- | --- |
| `slot` | `resident` | co-resident, budgeted; never takes the main slot |
| `vram_mib` | 3300 | the budget its utilisation is derived from, not a flag |
| `ctx` | 32768 | **not** the checkpoint's native 128,000 (`config.json: max_position_embeddings`, checked 2026-09-18): the whole 3.3 GiB budget is sized around a 32k KV cache |
| `port` | 8007 | inherited from the retired unit; clients that hardcoded it still work |
| `build` | `stock` | vLLM 0.29.0 in `~/Projects/local_llm/.venv-llm-029` |
| `max_output_tokens` | 4096 | |
| `tools.parser` | `lfm2` | with `--enable-auto-tool-choice`, emitted by the renderer |
| reasoning | none | LFM2.5 has no thinking mode, so there is no `reasoning` field to mirror into `reasoning_content` (see [../CLIENTS.md](../CLIENTS.md)) |
| `flags` | `--max-num-seqs 64`, `--dtype bfloat16` | 64 short mechanical requests at once; bf16 matches the checkpoint's own `dtype` |
| `host_ram_gib` | unset | it uses no host offload, so `start()`'s RAM preflight has nothing to check |

`VLLM_USE_FLASHINFER_SAMPLER=0` comes from `[defaults.env]` and is **required**, not a preference:
this box has no CUDA toolkit (no `nvcc`, no `/usr/local/cuda`), FlashInfer's top-k/top-p sampler is
JIT-compiled, and without the flag the engine core dies during the profiling run. A harmless
`deep_gemm` import warning at boot has the same cause. `HF_HUB_OFFLINE=1` is also a default now;
the 681 MiB snapshot is complete on disk (2026-09-18).

## The arithmetic that decides it fits

A resident's utilisation is derived, `floor2(vram_mib / total_mib)`: 3300 / 97887 → **0.03**. What
it will actually take is `ceil(97887 x 0.03)` = **2,937 MiB**, and `compute_util` requires that plus
a 512 MiB CUDA-context cushion (`RESIDENT_CUSHION_MIB`) to be free — **3,449 MiB**.

The cushion, not the global `margin_mib = 1024`, is the correct charge: the margin is the *main*
model's context cushion and has already been spent by the main model that is running. Charging the
resident for it too (budget 3,300 + margin 1,024 = 4,324 MiB against 3,735 MiB free) refused the
launch in exactly the situation the resident exists for. Fixed 2026-09-18.

Boot order decides whether both fit, because vLLM refuses to start when free VRAM < util x total:

- Resident first: it holds ~3.2 GiB, leaving enough for Flash-Next's 0.96 requirement of
  `ceil(97887 x 0.96)` = 93,972 MiB.
- Main first: whatever the main model leaves has to clear 3,449 MiB. Under the Flash-Next boot
  running on 2026-09-18 (util 0.96, ctx 262144) the card reports **2,815 MiB free**, which does not
  — a `start lfm2` now would be refused, naming the numbers. Stop and re-start in resident-first
  order, or lower the main model's utilisation for that boot.

Do not raise this resident's utilisation without lowering the main model's.

## Measured (2026-09-12, not re-measured since)

| | |
| --- | --- |
| Resident VRAM at util 0.03 | 3,204 MiB: weights 0.69 GiB + CUDA context + graphs + 1.56 GiB KV |
| KV cache | 135,878 tokens = 4.15 concurrent 32k requests |
| Slack against Flash-Next at 0.96 | 532 MiB, in either boot order, with the desktop's ~180 MiB |
| First boot | ~75 s (torch.compile 6 s, then cached AOT); later boots faster |
| Warm latency | 50-220 ms |
| Behaviour checked against the live server | plain chat, `tool_choice=auto` with parsed JSON arguments, forced named `tool_choice`, a tool-result round trip, `response_format: json_schema` returning schema-valid JSON |

**Refuted: util 0.04.** It was tried and measured at 4.1 GiB resident, which would have made
Flash-Next at 0.96 refuse to boot. 0.03 is the shipped value for that reason alone, not for
tidiness.

## Build: a consolidation choice, not a transcription

The retired `~/.config/systemd/user/lfm2-350m.service` (disabled and inactive, 2026-09-18) ran
`python -m vllm.entrypoints.openai.api_server` from `~/Projects/vllm-qwen38next/.venv-next`, the
Flash-Next fork's venv, because that build happened to carry `Lfm2ForCausalLM` and the `lfm2` tool
parser. The registry instead declares `build = "stock"` and servedeck renders `vllm serve` from
`.venv-llm-029`, which was checked on 2026-09-18 and carries both
(`vllm/model_executor/models/lfm2.py`, `vllm/tool_parsers/lfm2_tool_parser.py`, vLLM 0.29.0). Not
tying the small model to a fork it does not need is the point; the first real boot under the stock
build is still worth watching. `tests/test_e2e_real.py` already uses this checkpoint from the stock
build as its workhorse fixture, on `sd-test-vfy-*` units and ports 8050-8056, never the production
unit or port.

To use it by hand, point a client at `http://127.0.0.1:8010/v1` (or `:8007` directly) with model
`LFM2.5-350M`; it has no aliases and no presets. Related:
[../OPERATIONS.md](../OPERATIONS.md), [../CONFIGURATION.md](../CONFIGURATION.md),
[../ARCHITECTURE.md](../ARCHITECTURE.md), [flashnext.md](flashnext.md).
