# glm53 — GLM-5.3-Flash (ABLITERATED, NVFP4)

A 181.28 GiB checkpoint served from a 95.6 GiB card at 15-17 tok/s, because 163.27 GiB of it (90%)
is routed MoE expert weight that need not be resident. `--cpu-offload-params experts` keeps those
experts in pinned host RAM and the GPU reads over PCIe only the ones a token routes to; the other
18.01 GiB — attention projections, embeddings, lm_head, the shared expert, norms — stays on the
card. Everything below follows from that arrangement.

GLM is the **fallback, not the daily driver**: it tops out near 16 tok/s because it does not fit,
while Qwen3.8-Flash-Next runs at ~173 tok/s on the same card because it does
([flashnext.md](flashnext.md)). The owner's rule — a smaller model at 4-bit beats a bigger one at
lower precision — settles the choice. `servedeck switch glm53` stops the main-slot model first. GLM
needs `Glm5Next` support from vLLM PR #53906, in no released vLLM, so it runs from the `glm53`
build (`~/Projects/vllm-glm53/.venv-glm53`; see [../BUILDS.md](../BUILDS.md)).

## How it fits

| part | size | where |
|---|---|---|
| routed MoE experts | 163.27 GiB | 125 GiB pinned in host RAM, the rest resident |
| attention projections (MLA + KDA) | 11.74 GiB | resident, BF16 |
| embeddings + lm_head | 2.36 GiB | resident, BF16 |
| shared expert | 2.02 GiB | resident, BF16 |
| dense MLP, norms, gates | 1.58 GiB | resident, BF16 |
| vision tower | 1.05 GiB | not loaded (`--limit-mm-per-prompt '{"image":0,"video":0}'`) |

A decoded token touches ~4.43 GiB of expert weight (42 MoE layers x top-8 of 288 x ~13.5 MiB). At
the shipped 77% offload ~2.9 GiB crosses PCIe **per token** at ~23-25 GB/s effective, so decode is
bandwidth-bound and kernel choice moves it only ~6% (measured 2026-08-29 to 2026-09-02; not
re-measured). The NVFP4 quantiser left every non-expert tensor in BF16, so the resident 18.01 GiB
is read every token too — the second-largest line in the decode budget.

## Launch configuration, knob by knob

`[models.glm53]` in `models.toml` transcribes `~/Projects/vllm-glm53/serve-opt.sh` at that script's
own defaults; a copy is kept at
[../imported-2026-09/vllm-glm53/serve-opt.sh](../imported-2026-09/vllm-glm53/serve-opt.sh).

**`--cpu-offload-gb 125` and `VLLM_MOE_HOT_SLOTS=54` are one setting, not two.** Measured
2026-08-31: offload 125 with 68 hot slots 16.76 tok/s, offload 105 with no pinning 13.73, offload
125 with no pinning 11.34. Offloading *more* is faster only because the VRAM it frees funds the
hot-expert cache; without the cache, 125 is the worst of the three. The floor comes from the other
side: below ~100 GiB of offload the load OOMs, since 95.6 GiB minus 18.0 GiB of resident weights
minus KV and activations leaves room for only ~60-65 GiB of experts. Advice of the form "above ~100
GiB offload just costs speed" (still in the fork's `serve.sh` header and its stale RUNNING
document) predates pinning and is wrong.

**`--offload-backend uva`, not `prefetch`.** GLM is top-8-of-288, so a decode step reads only the
experts it routed and UVA's zero-copy reads exactly those; prefetch moves whole layers (~3.9 GiB
each) regardless of routing. The non-UVA fallback is worse than useless:
`VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1` dies with an illegal memory access and leaves a D-state
thread holding all VRAM, which only a reboot clears (`nvidia-smi --gpu-reset` refuses while the
card is primary). Never use it as a diagnostic step.

**`--kernel-config '{"moe_backend":"marlin"}'` is mandatory on SM120.** All five NVFP4 MoE backends
were swept on 2026-08-29 (source-only; it cannot be re-run without launching GLM). Both
CUTLASS-family kernels — `flashinfer_cutlass`, which `auto` picks, and vLLM's own `cutlass` —
accept the work and return garbage: the model loads, `/health` is green, and every prompt returns
one token forever with no error or warning. `flashinfer_trtllm` and `flashinfer_cutedsl` refuse
loudly. Marlin is also the fastest at GLM's dimensions, not a fallback
([Refuted](#refuted-with-evidence)); its unconditional "your GPU does not have native support for
FP4" warning proves nothing.

**The hot-expert profile is tied to the offload it was captured at.**
`VLLM_MOE_HOT_PROFILE=~/Projects/vllm-glm53/hot-profile-125.json` (36 KB, written 2026-09-02) holds
per-layer expert frequency counts, and layer *N* means "the *N*th offloaded layer" — a different
`--cpu-offload-gb` partitions the layers differently and the sets stop lining up. A mismatch is not
fatal: the pinner warns once and falls back per layer. Two cost figures circulate and the code's is
tighter — `dma_stage.py:298-302` records a 125-profile run at offload 127 measuring 15.09 against
16.76 tok/s, about 10% or ~1.7 tok/s, while `serve-opt.sh:117`'s "~5 tok/s" belongs to the
*missing*-profile case, where pinning is off entirely and throughput falls to ~11-14 tok/s.
Regenerate with `VLLM_MOE_SKEW_PERLAYER=<out.json>` at the same `CPU_OFFLOAD_GB`. `CPU_OFFLOAD_GB`
is also exported as an env var, and both belong there: vLLM takes the offload from the flag, the
pinner reads the env var only to compare the launch against the loaded profile. **The profile file
is untracked in the fork** (`git status`: `?? hot-profile-125.json`, 2026-09-18) — nothing in the
repository would restore it, and losing it costs a profiling run and ~2-5 tok/s until it is redone.

**`VLLM_MOE_PREFILL_GROUPS=1` is worth 2.9x on prefill.** A prefill chunk routes to nearly all 288
experts, exceeding the 64-row staging buffer, so ungrouped the layer is read zero-copy at ~23 GB/s
with tile re-reads: 614 tok/s prefill, ~7 min to first token at 262k. Grouped, the experts are
DMA-staged 64 at a time and reduced once: 1,762 tok/s, and 10k TTFT 16.8 s → 5.8 s. Equivalence was
proven by prompt-logprob comparison on fresh prompts, not greedy text
([Measuring a change](#measuring-a-change)). Summing BF16 partial outputs instead of reducing once
was **not** equivalent and must not be reintroduced.

**CUDA graphs are kept — no `--enforce-eager`.** The device-index Triton gather kernel made the
data-dependent expert staging graph-capturable, worth 11.77 → 13.53 tok/s (+13%, identical
staging); the earlier host-side design needed a device-to-host sync that capture forbids
("operation failed due to a previous error during capture"). `serve-opt.sh:298-305` still carries
the pre-kernel argument *for* eager while defaulting to graphs: the default is right, the comment
is stale. Do not copy this trade to a model that fits in VRAM, where it reverses.

**Pinned-memory accounting: two env vars, both counter-intuitive, both measured.**
`PYTORCH_ALLOC_CONF=...,pinned_use_cuda_host_register:True,pinned_num_register_threads:8` allocates
pinned host memory by `malloc` + `cudaHostRegister` instead of `cudaHostAlloc`, which hands out
shared 4 GiB granules — a 2.42 GiB `w13` and a 1.21 GiB `w2` expert tensor each rounded up, so
19.0 GiB of weights occupied 33.8 GiB of host RSS (1.85x); at full scale that turns a 100 GiB
offload into ~185 GiB and kills the box. `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1` then stops
Python pinning the weights *earlier*: handing `get_cuda_view_from_cpu_tensor` an already-pinned
tensor routes it through PyTorch's caching host allocator with power-of-two rounding, while an
**unpinned** tensor lets the C++ op pin an exact-size allocation itself. Same 19.0 GiB: 33.8 GiB
resident down to 21.5 GiB (1.94x → 1.13x). Pinning earlier is the pessimisation.

**`--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`; 2 is the only usable
depth.** Depth 1 returns correct text then dies with an illegal memory access partway through a
longer generation. Depth 4 hits a DeepGEMM `DG_HOST_ASSERT` on `block_kv` for arch 12 in the
sparse-attention KPOOL indexer. Depth 5 raised the per-request KV need to 4.27 GiB against the
then-current 4 GiB cap and failed at startup — that objection is against the old cap and the
shipped cap is 5.25 GiB, so depth 5 is *untested* rather than closed. MTP-2 measured 17.10 tok/s on
a coding prompt with 1.28 accepted tokens per step (64% cumulative: position 0 75%, position 1
53%). Speculation pays more than usual here because verifying several tokens per forward pass
reuses one expert fetch.

**`--max-num-batched-tokens 8192` is a hard floor.** 4096 kills the engine on the first short
decode request (HTTP 500, then engine dead) and 2048 likewise — re-confirmed 2026-09-02 on the
build with breakable graphs off and the staging bounds checks in place, so it is a defect separate
from the Xid faults. Consequence: shrinking the prefill chunk, the natural way to buy VRAM headroom
for long prompts, is closed. Headroom comes from hot slots instead.

The rest of the configuration, each with the reason it exists:

| setting | why |
|---|---|
| `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | Must be explicit, not absent: the fork auto-enables breakable graphs for `Glm5NextForConditionalGeneration` when the variable is unset (`config/vllm.py:688-697`, list at `:75-88`). Under them every build on 2026-09-02 faulted (Xid 31) after 2-24 requests. |
| `VLLM_USE_DEEP_GEMM=0` | Mandatory while MTP is on: the draft head routes through DeepGEMM's paged-MQA attention, which asserts `arch_major == 10` (SM100); this card is SM120. Costs nothing measurable on the plain decode path. |
| no `--block-size` | On SM120 DeepGEMM's paged-MQA needs `block_kv == 64` for the fp8 indexer, so `page_size = block_size / index_kpool` must be a multiple of 64 (block_size a multiple of 256). vLLM picks by LCM; spec-decode depth can shift it off that multiple, failing at an opaque C++ assert. |
| no `--attention-backend` | The SM120 sparse-MLA kernel refuses GLM's shapes, so the SM90 kernel is the working fallback ([Refuted](#refuted-with-evidence)). |
| `--max-num-seqs 1` | One agent at a time; at 16 tok/s batching buys nothing, and concurrency would spend KV budget the hot-expert cache is paid out of. |
| `--shutdown-timeout 30` | vLLM's default 0 means abort: the API server SIGTERMs the engine and force-kills it at once, so 125 GiB of pinned host memory is never `cudaHostUnregister`ed. 30 s also drains in-flight requests instead of dropping a generation mid-stream. |
| `--limit-mm-per-prompt '{"image":0,"video":0}'`, `vision = false` | Wired up as a coding backend; skipping the 1.05 GiB tower saves VRAM and a slice of startup, and clients are told the truth. |
| launch cwd | `~/Projects/vllm-glm53` is itself a second, uncompiled vLLM checkout, and Python puts cwd first on `sys.path`: launching from there fails as "vllm.vllm_flash_attn requires the CUDA flash attention extensions", naming the wrong subsystem. servedeck sets `WorkingDirectory=` to the venv, which has no `vllm/` — avoided by circumstance, not by an assertion. |

GLM needs **no `ptrace_scope` relaxation** of its own (UVA offload is in-process pinned memory,
with none of the CUDA-IPC handoff Flash-Next's PLE needs) and writes **no `/dev/shm` buffer**, so
the offload reaper has nothing of GLM's to free.

### Context and the KV cap

`ctx = 327680`, and that is **not** the checkpoint's limit: `config.json` says 1,048,576, and
servedeck's discovery reports `model_max_ctx = 1048576` for this repo. GLM's context here is
bounded by free VRAM, because long-prompt activation (attention plus the sparse indexer) is
allocated per request and scales with length rather than being reserved up front. 327,680 was
validated only after cutting hot slots to 54, and the largest prompt actually prefilled is 304,800
tokens — 93% of the advertised length (2026-09-02).

The KV cache must be capped explicitly. Uncapped, vLLM sizes it from the leftover budget: the
2026-08-29 run provisioned 2.1x concurrency and spent 14.5 GiB at 512k context for a single agent,
starving the expert cache the offload is paid for. Since 2026-09-18 servedeck **derives** the cap
from the launch context (`kv_cache_bytes_per_token = 17200`, so
`--kv-cache-memory-bytes = ctx x 17200`) rather than pinning a product that a moved context control
can contradict; 17,200 B/token is ~10% of headroom over the measured 15.0-15.7 KiB/token (4 GiB
gave 272,771 KV tokens, 5.25 GiB gave 375,543). The derived cap at 327,680 is 5,636,096,000 B,
1 MiB below the 5,637,144,576 the shell launcher used and validated at 304,800 tokens, so
`serve-opt.sh`'s claim that the formula "reproduces" that number is 1 MiB out. Moving the context
without the cap — which vLLM refuses with "N GiB KV cache is needed, larger than the available", as
happened when the old Coldstart UI passed `MAX_LEN` but not `KV_BYTES` — is no longer possible, and
a launch asking for more than 327,680 is refused outright rather than clamped, naming the pinned
value (`legacy_page.py:855-867`).

`--gpu-memory-utilization` is inert for GLM as configured: with `--kv-cache-memory-bytes` set the
worker skips memory profiling and logs "This does not respect the gpu_memory_utilization config",
and 0.95 against 0.96 produced byte-identical VRAM and the same 272,771-token KV cache. The
registry still pins `util = 0.95` (every measured GLM run used it) and since 2026-09-18
`compute_util` refuses to launch when the card cannot fit the pinned value, naming what holds it.
For GLM that refusal is the whole value of the number: a fit check, not a tuning knob.

`VLLM_MOE_HOT_SLOTS` is therefore GLM's only real VRAM-headroom lever, and nearly free at the top
of the ladder (measured 2026-09-02):

| hot slots | free VRAM after load | longest prompt | decode |
|---|---|---|---|
| 60 | 3,397 MiB | 180k dies ("Tried to allocate 116.00 MiB") | — |
| 54 (shipped) | 6,177 MiB | 304,800 tokens, TTFT 205 s at 1,487 tok/s | 14.99 tok/s |
| 48 | 8,817 MiB | 233,670 tokens, TTFT 158 s | 14.92 tok/s |

48 → 54 is ~3 points of cache hit rate, and 14.92 against 14.99 is inside the ±1 tok/s run-to-run
spread, so long-context headroom costs almost no throughput. Rule kept from that work: **never
advertise a context that has not been prefilled successfully at that length** — both earlier
settings (262,144 and the first 327,680) were advertised and never tested near their limit.

One trap belongs here: the ~29 GiB hot-expert cache is allocated **after** vLLM's memory profiling,
because `stage()` returns early when `n > MAX_STAGED_EXPERTS` (64, `dma_stage.py:51,382`) and the
profiling run's prefill is far above that cap. vLLM sizes KV and workspaces believing the VRAM is
free, and the server hits "CUDA out of memory. Tried to allocate 124.00 MiB" **mid-serving, after a
successful load** — three identical crashes on 2026-09-02. Moving the cap check after the
allocations was tried and reverted: startup then fails correctly, but serving hits an illegal
memory access.

## Host RAM is a hard precondition

A GLM launch needs `CPU_OFFLOAD_GB + ~30 GiB` of available host RAM — 155 GiB at the shipped
offload of 125, which is what `host_ram_gib = 155` carries. The +30 is calibrated against
measurement: the 2026-08-29 attempt offloaded 110 GiB and peaked near 168 GiB of host use, because
the loader's transient copies cost far more than the pinned weights. Steady state is offload +
~16 GiB (141 GiB used at offload 125 on 2026-09-02), so 30 keeps a ~14 GiB cushion.

**This box has no swap** (`swapon --show` empty, 2026-09-18) and pinned pages cannot be reclaimed,
so overshooting is an OOM kill of the session, not a slowdown; one real kill happened on 2026-08-28
when a second CUDA process ran alongside the pinned server. `serve-opt.sh` refused to start below
the threshold and servedeck had no equivalent until 2026-09-18, when `start()` gained the same
preflight and refuses with both numbers in the message.

This makes GLM and Flash-Next mutually exclusive on host RAM as well as VRAM. Flash-Next parks
40 GiB of evicted KV in a `/dev/shm` buffer, which is shmem and unreclaimable: with Flash-Next
serving on 2026-09-18, `/dev/shm` held 41 G of its 92 G and `MemAvailable` was ~70 GiB against
GLM's 155 GiB. `servedeck switch glm53` stops Flash-Next first and the reaper frees the unmapped
buffer before the next start, so the order is right; whether the remainder clears 155 GiB has not
been measured on the current configuration, and the preflight decides, naming what it read.

## Measured performance

Decode, single request, greedy or low temperature. Nothing here has been re-measured since
2026-09-02.

| measurement | tok/s | condition |
|---|---|---|
| plain UVA baseline | 10.77 | no staging kernel, no pinning (2026-09-01) |
| staging + MTP-2, eager | 11.77 | before the device-index gather kernel |
| + CUDA graphs restored | 13.53 | identical staging (2026-09-01) |
| + hot-expert cache, 66 slots | 16.50 | published headline (2026-09-01) |
| offload 125 + 68 slots | 16.76 | best measured (2026-08-31) |
| 40-request varied soak | 15.77 | T=0/0.7/1.0, four 9k prefills + a 30k prompt (2026-09-02) |
| shipped 54 slots | 14.99 | traded for long-context headroom (2026-09-02) |
| coding prompt, MTP-2 | 17.10 | 1.28 accepted tokens/step (2026-09-02) |

The published 16.50 headline was taken at 66 slots and is **not** a controlled A/B against the
shipped 54-slot 14.99; the public documents state that asymmetry rather than hiding it.

Where a decode step goes — torch profiler, median of 12 steps, 2026-09-02, not re-measured:
staging gather over PCIe 34.06 ms, dense BF16 GEMM 18.98 ms, MoE experts (Marlin) 3.58 ms,
norm/activation/other 1.38 ms, attention 0.47 ms. GPU busy exceeds the wall span by ~13.7 ms, so
some overlap already happens. Attention is 0.1% of GPU time and never the bottleneck here — worth
remembering before optimising it. The dense GEMM is the 18.01 GiB of BF16 non-expert weights read
every token; its pure-bandwidth floor is 10.74 ms at 1.8 TB/s, so it runs at ~57% of peak.

Prefill, fresh never-seen prompts, streaming, shipped build, 2026-09-02 (logged at
`~/Projects/glm53-single-gpu-notes/artifacts/prefill-fresh-2026-09-02.txt`): a 14-token prompt
0.85 s on the zero-copy path; 10k in 5.82-5.96 s (1,729-1,771 tok/s); 30k in 22.57 s (1,576 tok/s);
35k in 22.7 s with decode 17.5 tok/s afterwards; 304,800 in 205 s (1,487 tok/s). Prefix caching
(`--enable-prefix-caching`) helps only exact repeats — an identical 10k prompt re-sent prefills in
~3.2 s on the shipped build, against 16.9 s vs 16.6 s cold on the pre-grouping build. Worth having,
but Copilot resends the whole conversation every turn, so at large context every turn re-pays
prefill.

### Why 25 tok/s is unreachable

The offload-sensitivity experiment (2026-09-01) varied how much crosses PCIe: offload 102 GiB (63%
of experts remote) 78.3 ms/token, offload 125 GiB (77% remote) 87.6 ms/token. The linear fit is
**34.8 ms fixed + 68.6 ms x traffic_fraction**, so transfer is only ~55% of a step. Even with
infinite VRAM and zero PCIe traffic the stack caps near **29 tok/s**; and 25 tok/s needs ~55%
expert residency (90 GiB) where only ~37% (61 GiB) fits, once 17 GiB of dense/attention weights,
4-5 GiB of KV, ~5 GiB of activations and ~1 GiB of staging are paid for.

The fetches cannot be hidden behind compute: the MoE router depends on the *same* layer's attention
output, which is architectural. The GPU profile shows it — SM utilisation 96-100% at only ~210 W of
the then-600 W cap, with 14.6% memory-controller load. SMs stalled on PCIe, not computing. GLM
therefore cannot exploit a higher power cap, and the current one (375 W as `nvidia-smi` reports it
on 2026-09-18) costs it nothing; see [../HOST.md](../HOST.md).

**Do not promise more than ~16 tok/s for this model on this hardware**; ~16-17 tok/s is the
quality-neutral ceiling. Reducing top-k does not rescue it: router weight by rank is
`[.269 .186 .137 .109 .091 .078 .069 .061]`, and even top-4 — discarding 30% of routed weight —
projects only 17.7 tok/s.

**What the hot cache exploits** is per-layer skew: the hottest 36.2% of a layer's experts serve 75%
of its fetches. (An earlier "routing is near-uniform" claim was an artefact of summing per-expert
counts across layers; the memory note still carries the retracted version in its filename.)
Measured hit rate by slot count: 33 → 42.65%, 66 → 61.46%, 99 → 73.98%, 132 → 82.99%,
165 → 89.62%. Flat at the top, which is why the 54-slot long-context trade is nearly free.

## Stability: Xid 31 is unresolved

Four Xid 31 MMU faults hit on the evening of 2026-09-02 — 18:14:33, 18:27:27, 18:47:23, 19:05:48,
all `MMU Fault: ENGINE GRAPHICS GPC0 ... FAULT_PDE ACCESS_TYPE_VIRT_WRITE` — with
`VLLM_USE_BREAKABLE_CUDAGRAPH=0` and the staging bounds checks both in place. breakable=0 bought a
quiet 40-request soak, not immunity. Any document saying the fault is fixed is wrong, including
this box's own published write-up.

- **Breakable CUDA graphs made it much worse**: every build under breakable=1 on 2026-09-02 faulted
  by request 2-24, while breakable=0 ran a clean 40-request soak at unchanged speed. The flag stays
  off.
- **The fault needs the agent-loop shape**, not single long prompts: a conversation regrown and
  re-sent in full each turn, streamed, with ~50 tool definitions, so the prefix cache and the
  prefill path interact on an ever-growing prefix. A 60-turn reproducer faulted at turn ~33 with
  only ~13k of context in a single CUDA process — no second benchmark, no interactive client — so a
  bisect can run unattended at ~10 min per arm. That reproducer (`agentloop.py`) lived in a session
  scratchpad that no longer exists and would have to be rewritten from this description.
- **Smaller `--max-num-batched-tokens` is not the cause**: the same fault hit an 8192 run after 12
  requests, and smaller batches only changed the allocation layout so corruption surfaced sooner
  (2nd request instead of 12th).
- **The CPU-side stack is not evidence.** It blames the FlashInfer MLA attention backend, but the
  log itself warns that errors are reported asynchronously at a later call. Bounds checks were added
  to every id-indexed access in the staging path (both Triton gather kernels,
  `build_local_expert_map`'s `scatter_reduce_`, the Marlin call site), because a padded or sentinel
  expert id there produces exactly this signature. They did not stop the faults.
- **Do not correlate crash timing against an idle period.** The "14 hours clean" baseline that
  implicated reasoning restoration was a period with no load.
- GLM runs the **SM90** sparse-MLA attention kernel on Blackwell by fallback, and that is the
  backend the faults surface in.

The one untested decisive experiment: run the agent-loop reproducer with
`VLLM_MOE_DMA_STAGING=0`, which defaults to 1 in the fork (`marlin_moe.py:117-124`), so the arm is
a one-variable change. Faults stopping implicates our gather kernel; faults continuing implicates
driver or hardware. It costs the whole optimisation (~16 down to ~10.8 tok/s) for the duration of
the arm, which is why it needs the owner's go-ahead. Note when building any load generator that the
retired effort proxy answered HTTP 200 with an *empty stream* when the engine behind it was dead:
the first reproducer run scored 26 post-mortem turns as "60 ok, 0 failed".

## Client compatibility

Three client-facing defects were diagnosed on 2026-09-02 and are handled in
`servedeck/glm_policies.py` plus the registry; the `glm53-effort-proxy` that first carried them is
retired. The client side is in [../CLIENTS.md](../CLIENTS.md).

**Erased reasoning is the root cause of agent "amnesia"** — the largest single fix on this model.
OpenAI-style clients never echo `reasoning_content` back, so GLM's template renders every prior
assistant turn, *including the in-flight tool-calling round*, as an empty `<think></think>`; the
model reads its own blank thinking, concludes no task was ever given, and greets the user. The
template's `clear_thinking` gate cannot fire because `reasoning_content is defined` is false.
Measured two independent ways. A logprob probe on the rendered prompt (append the tokens for "The
user", ask for 1 token with logprobs=20) put 3.32% of the mass on an amnesia continuation as
Copilot sends it, against 0.00% with the reasoning restored — **33,000x on that probe** (`' wants'`
51.9% → 99.97%); across 12 varied conversation states the amnesia mass went from mean 0.32% / max
2.18% to 0.00% in all 12. The 3.32% predicted rate matches the 4.5% per-turn greeting rate measured
independently from 3,898 real transcript turns. It is **not** a long-thread effect (a flat per-turn
coin flip from turn 3 onward) and **not** the checkpoint: a near-matched control on this box —
Qwen3.8-Flash-Next, also abliterated, also NVFP4, same client, same 51 tools — did 3,788 turns with
zero greetings against GLM's 5 in 112 (Fisher p=1.8e-8). The mass is non-zero only on
**tool-terminated** turns and exactly 0.00% when the conversation ends on a user message, which is
why it bites agent mode and never chat.

`restore_reasoning` is nevertheless **OFF**, and that is an open decision rather than a settled one.
The recorded reason was the retired proxy's own note — "three Xid 31 GPU faults followed within 40
minutes of it first firing, after 14 hours clean" — and that correlation has been **refuted**: the
14-hour baseline was an idle period. What remains is a general argument (this box has an active,
unrelated GPU-fault problem, so nothing that plausibly perturbs the GPU defaults to on). Given the
measured size of the fix, turning it on for one session and watching `state/telemetry` is the
obvious next step. The gateway's output-side mirror (`mirror_content = true`) is no substitute: it
makes the thinking available to a client, it cannot make a client send it back.

**The thinking budget is the other half of the same symptom.** Reasoning shares `max_tokens` with
the answer and GLM's template opens a `<think>` block before the model writes a word, so a small
per-request budget is spent thinking and the turn returns empty — reproduced directly against :8002
with `max_tokens=200`: `finish_reason="length"`, `content_len=0`. That empty turn goes back into the
client's history and the thread reads as fresh. VS Code Copilot and Codex both send small budgets
for short turns. `min_output_tokens = 8192` raises any smaller budget, only ever upward, clamped so
prompt + budget still fits the context; a request that sent no budget at all is left alone, because
it already has the whole remaining context (`policies.py:288-328`). Calibration: measured GLM
reasoning usage on a hard problem was 183 tokens at effort low, 484-713 at high, 1,812 at max — max
is 7.2% of a 32,768 output budget, so the configured output ceiling was never the constraint. The
fork also supports a per-request `thinking_token_budget`, enforced by a sampler that forces the
think-end token when the budget is spent (`sampling_params.py`,
`v1/sample/thinking_budget_state.py`), which the retired proxy defaulted to `min(10,000, 75% of a
stated output budget)`. It is **not** ported to servedeck's gateway — a gap rather than a defect,
since the cap costs nothing until it binds.

**Effort levels are published as model aliases.** GLM's template resolves anything other than
`low`/`high` to `reasoning_effort="max"` (unbounded thinking), and the parameter can only be passed
through `chat_template_kwargs`, which neither VS Code's BYOK provider nor Codex exposes. The
registry ships presets `glm53-flash-low`, `glm53-flash-high` and `glm53-flash-max`, merged into
`chat_template_kwargs` as gap-filling defaults; the bare `glm53-flash` id still resolves to max.

**The tool-tag sanitiser is on.** GLM's streaming tool-call parser can miss a closing tag that
straddles a token boundary and the tag then leaks into the parsed argument — captured from a real
VS Code Copilot turn as `list_dir({"path</arg_key>": "/Users/user/Desktop/project"})`. The client
rejects the call ("must have required property 'path'"), the agent loses its step and starts
inventing paths. Stripping markup that is never valid argument text is a repair rather than a guess,
so `sanitize_tool_tags = true`, and servedeck repairs the **response** as well as the echoed request
so the broken key never reaches the client. Fixing this upstream in `glm47_moe.py`'s streaming state
machine is still open. `capture = false` and also needs `SERVEDECK_GLM_CAPTURE=1` in the
environment before anything is written, because bodies contain everything the user typed; the
predecessor defaulted capture to on and wrote full bodies to `/tmp/glm-capture` with no retention.

The `glm47` reasoning parser was checked and cleared: an online report says this adapter class
discards replies, and on this build it does not reproduce (5/5 sampled replies had reasoning and
content correctly separated, because the model re-emits `<think>`). Revisit `deepseek_r1` only if
leakage is actually observed.

## Refuted, with evidence

| idea | why it is dead |
|---|---|
| Native-FP4 MoE kernels (b12x / FlashInfer / CUTLASS) | Marlin is fastest at every batch size at GLM's dimensions: at M=1, Marlin 77.4 µs vs `flashinfer_cutlass` 99.8 (0.78x), `vllm_cutlass` 110.0 (0.70x), b12x 409.4 (0.19x); Marlin also wins at M=2, 4, 64. End-to-end b12x served 1.2-1.3 tok/s. b12x also `del w1, w2` and hard-raises on `expert_map`; FlashInfer accepts `expert_map` and never reads it, which is silently wrong. (The public table says 77.5 µs; pick one figure when migrating.) |
| The native SM120 sparse-MLA attention backend | `ATTN_BACKEND=FLASHINFER_MLA_SPARSE_SM120` fails at startup — "SM120 sparse MLA v32/GLM expects kv_lora_rank=512, qk_rope_head_dim=64, and query head dim 576" — and this model is NoPE MLA. Using it would need its shape assumptions ported the way the SM90 path already was. |
| FreeToken hybrid CPU co-execution | Measured 0.895x here (55.15 vs 61.62 tok/s in a Flash-Next comparison). Host memory bandwidth is 64.6 GB/s dual-channel against a 51.8 GB/s PCIe gather rate — 1.04x, where FreeToken's own policy needs 2.0x. Their published GLM numbers come from a 178 GB/s Xeon. |
| Router pruning (`VLLM_MOE_DECODE_TOPK=6`) | 20.48 vs 16.50 tok/s (+24%) but visibly degraded quality: repetition loops ("Jupiter (planet) (planet) (planet)") and word-salad explanations. Facts survived pruning; explanations did not. Off under the owner's no-quality-loss rule. The env var exists in the fork for a one-off. |
| FP8 for the 18 GiB of BF16 dense weights | Would save real time per token (5-6 ms by the decode-budget trace, ~10 ms by the quality note's estimate — the two disagree and neither was re-measured), but no checkpoint exists: a full-FP8 GLM-5.3-Flash is ~331 GB against 182 GiB host RAM + 96 GiB VRAM, so it cannot fit at all, and no FP4-experts/FP8-dense variant exists. Self-quantising is not free: properly scaled FP8 (per-tensor *and* per-channel) measured only ~31 dB SNR, because the BF16 tensors are genuine masters with non-zero FP8 round-trip error. Rules out `dealignai/...-UNCENSORED-FP8` and `orcarouter/...-Uncensored-FP8`; `LibertAIDAI/GLM-5.3-Flash-NVFP4` is the same layout as ours. |
| A bigger temporal (LRU) expert cache | 9.0% hit at 5.6% capacity, barely above random, because GLM activates 8 of 288 (2.8%) — far sparser than the Mixtral-class models the caching papers use. Static frequency skew is the property that pays. |
| Variable per-layer slot budgets (water-filling) | +0.07 pp for the same total slots: the layers' distribution *shapes* are alike, so uniform allocation is already optimal. |
| Re-profiling the hot set on mixed-temperature traffic | Worse on everything (13.02 vs 15.77 tok/s on prose) for +4 GiB of margin. Sampling at T>0 flattens the routing skew the profile exists to capture. |
| Deeper speculation (MTP past 2, n-gram) | Depth 4 asserts, depth 1 faults, depth 5 was refused by the old KV cap. A free drafter saves ~2-3% of a step at depth 2, and n-gram or suffix cannot beat a trained MTP head at that depth. |
| Lowering `--max-num-batched-tokens` for VRAM headroom | 2048 and 4096 crash the engine on the first request, even on the fixed build. |
| Raising `--gpu-memory-utilization` | 0.95 and 0.96 give byte-identical VRAM and the same 272,771-token KV cache, because the explicit KV cap makes profiling moot. |
| GPUDirect Storage / NVMe as additive bandwidth | The GPU's x16 ingress is the shared bottleneck and already at 92%; `nvidia-fs` is not installed. |
| More copy engines or streams | `asyncEngineCount = 2`. Engines are not the constraint, the x16 link is. |
| Buying more host RAM | Even at 256 GB, GLM's best case projects ~55 tok/s while Qwen3.8-Flash-Next already delivers 145-173 tok/s because it fits. When a model does not fit, the answer is a smaller model — and there is no hardware budget ([../DECISIONS.md](../DECISIONS.md)). |
| Offline expert pruning or merging; a smaller GLM-5.3 Flash variant | The first changes model quality irreversibly and needs calibration or retraining at 320B scale; the second does not exist in any documented form. |

## Measuring a change

**Greedy-text equivalence tests are invalid on this stack.** Long-prompt prefill is not run-to-run
deterministic — the same 3k and 9k prompts at T=0 produced output hashes A, B, A — because mamba
"align" mode caches state at scheduler-step boundaries and a cached-state continuation rounds
differently in BF16, flipping knife-edge argmaxes. Compare `prompt_logprobs` on **fresh** prompts
against the baseline's own fresh-vs-cached drift. This cost one good optimisation: the masked-group
prefill variant was withdrawn on misattributed nondeterminism and is still untested, having
measured 0.36 s TTFT on a 14-token prompt against 0.85 s shipped. Always separate TTFT from decode
when judging a change; a prefill regression once read as a decode regression.

**Profiling**: this fork has no `VLLM_TORCH_PROFILER_DIR`. Pass
`--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=DIR` at launch, then POST
`/start_profile` and `/stop_profile` to the engine port. `nsys` and `ncu` are not installed here.

## The public write-up

The full optimisation story is published at
[github.com/pctablet505/glm53-flash-single-gpu](https://github.com/pctablet505/glm53-flash-single-gpu)
(public; MIT for the docs, `src/` Apache-2.0 as vLLM-derived), working copy at
`~/Projects/glm53-single-gpu-notes`. Read it there rather than restating it;
`FURTHER-OPTIMIZATION-RESEARCH.md` §8 is the refuted table above in more detail.

**Where the public text is now wrong:** it states that the Xid 31 MMU fault was fixed by
`VLLM_USE_BREAKABLE_CUDAGRAPH=0` plus the staging bounds checks. The box disproved that the same
evening (2026-09-02) with four faults on the fixed build. The claim appears in several places in
that repository and has not been corrected upstream.

Framing rules held in the public text and worth preserving: "the best result we know of on a single
RTX PRO 6000", never "world record"; the one third-party number (a FreeToken ~14.9 tok/s GLM-5.2
figure) is marked second-hand, unlinked and unreproduced; and the two honest asymmetries — the
16.50 headline at 66 slots against the shipped 54-slot 14.99, not a controlled A/B — are stated
rather than hidden. The internal agent brief was deliberately moved out to
`~/Projects/glm53-notes-internal/` and stays out.
