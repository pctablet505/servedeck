# Configuration

`models.toml` in the repo root is the only file you edit. Everything else — the
gateway's routing table, each `vllm serve` argv, the generated client configs,
`doctor` — is derived from it. `servedeck.toml` no longer exists: no module
reads it, and a file of that name in the repo is ignored rather than merged.
`state/desired.json` is written by servedeck, not by you; the client configs are
written by `servedeck wire --apply` (see [CLIENTS.md](CLIENTS.md)). Every value
below was checked against `models.py`, `settings.py`, `limits.py` and
`desired.py` on 2026-09-18; a key not listed here is one the loader ignores.

## `models.toml`

Loaded and validated by `models.py:load()`. One problem is one `RegistryError`
saying what and where, never a bare `KeyError`; if the registry fails to load,
`doctor` skips every other check. **The loader ignores keys it does not know** —
there is no strict-schema pass, so `utl = 0.96` or `hostram_gib = 95` loads
cleanly and does nothing. Check a new key against the tables here, and confirm
it took effect with `servedeck models` or `servedeck doctor`.

### `[gpu]` — required

| Key | Type | Required | Default | For |
|---|---|---|---|---|
| `total_mib` | int | yes | — | The card's size, the denominator of every utilisation. `97887` here. |
| `margin_mib` | int | yes | — | Subtracted from free VRAM before a main model's utilisation is computed, never handed to any model. `1024` here. |
| `power_limit_w` | int | no | unset | The watt cap `doctor` expects. |

`margin_mib` is also the launching worker's CUDA-context cushion: the process
filling `--gpu-memory-utilization`'s fraction first creates a CUDA context,
loads cuBLAS/cuDNN kernels and allocates NCCL buffers, a few hundred MiB outside
that fraction and taken while the allocation runs, so `control.compute_util`
guarantees `ceil(total * util) <= free - 700`. `power_limit_w` is deliberately
unset: the cap is the owner's decision, and a number here that nobody decided
would read as one somebody had. Unset, doctor's power-cap row reports the live
watts and passes; set, it fails when the live cap differs by more than 1 W. See
[HOST.md](HOST.md).

### `[defaults.env]` — merged into every model

A table of string → string, optional, merged at load time into each model's own
`env`, where the model's own key wins. Quote the values: they are interpolated
into `--setenv=K=V`, so a TOML integer or boolean reaches the model as `4` or
`True` with no complaint.

`VLLM_USE_FLASHINFER_SAMPLER = "0"` — no system CUDA toolkit here, so
FlashInfer's JIT top-k/top-p sampler cannot compile and the engine core dies
during the profiling run. Every legacy launcher exported it; the harmless
`deep_gemm` import warning at boot has the same cause.

`HF_HUB_OFFLINE = "1"` (added 2026-09-18) — every model launches from the local
hub cache, so a launch cannot depend on the network, cannot resolve a revision
other than the one on disk, and needs no `HF_TOKEN`. The cost: a snapshot
genuinely missing a file fails fast with a local-files-only error instead of
fetching it, which is why doctor grew a `weights (<key>)` row per model. The 27B
and GLM have not been launched under it yet, so watch the first one.

### `[builds.<name>]` — the venv a model launches from

| Key | Type | Required | For |
|---|---|---|---|
| `venv` | string | yes | `vllm` is looked for at `<venv>/bin/vllm`, unless the last path segment is already `bin`, in which case it is used as written. `~` is expanded before it becomes argv[0]'s directory, because `execve` does not expand it. |
| `cuda_home` | string | yes | Becomes `CUDA_HOME`, and `<cuda_home>/bin` goes on the model's `PATH`. |

`cuda_home` exists because the box has no system CUDA toolkit: `nvcc`/`ptxas`
come from the venv's `nvidia/cu13` wheel and FlashInfer's JIT needs them at
request time, not only at process start. It is carried explicitly rather than
string-glued from `venv` plus a python version, because that version is a fact
about how the venv was built. Both keys are required, and the short forms the
older version of this document showed are **rejected**: `stock = "~/path"`
raises `builds.stock must be a table with 'venv' and 'cuda_home'`, a `venv`-only
table raises `builds.stock: missing 'cuda_home'` (`models.py:370-382`). The
three builds are in [BUILDS.md](BUILDS.md).

### `[models.<key>]`

The table key is the registry key: the CLI argument, the `/api/*` path segment
and the unit-name suffix (`model-<key>`). It must match `[a-z0-9-]+` or the load
fails, rather than systemd refusing three modules later.

| Key | Type | Required | Default | For |
|---|---|---|---|---|
| `id` | string | yes | — | The canonical public name; first entry in `--served-model-name`. |
| `repo` | string | yes | — | Hugging Face repo id; `vllm serve`'s positional argument. |
| `slot` | `"main"` \| `"resident"` | yes | — | `main` is exclusive, one at a time; `resident` is co-resident and budgeted. |
| `port` | int | yes | — | The loopback port this model's own vLLM listens on. Clients never see it. |
| `build` | string | yes | — | A key in `[builds]`. |
| `ctx` | int \| `"native"` | yes | — | `"native"` resolves to the checkpoint's own `max_position_embeddings` from the local hub cache (never fetched). An int pins a validated ceiling below it — GLM is `327680` because its context here is bounded by VRAM headroom, not by the checkpoint's 1,048,576. |
| `aliases` | list of strings | no | `[]` | Every name a client has ever used. Appended to `--served-model-name` and advertised in `GET /v1/models`. |
| `flags` | list of strings | no | `[]` | Appended verbatim to the argv after the computed flags. |
| `env` | table of strings | no | `{}` | The model's own environment, merged over `[defaults.env]`. |
| `presets` | table of tables | no | `{}` | See below. |
| `vram_mib` | int | residents only | — | The resident's VRAM budget, from which its utilisation is derived. |
| `util` | float in (0, 1] | no | unset | **Added 2026-09-18.** The utilisation this main model is *proven* to serve at; refused at load time if it is not a number in (0, 1]. |
| `host_ram_gib` | int | no | unset | **Added 2026-09-18.** `MemAvailable` GiB required before launch; `start()` refuses below it. No swap on this box and pinned pages cannot be reclaimed, so overshooting is an OOM kill of the desktop, not a slowdown — one happened on 2026-08-28. |
| `kv_cache_bytes_per_token` | int | no | unset | **Added 2026-09-18.** When set, `--kv-cache-memory-bytes` is emitted as `ctx_tokens * this`. The registry used to pin the product instead, so moving the context control left the KV cap sized for the old length and vLLM refused after a six-minute load. |
| `min_output_tokens` | int | no | unset | Floor the gateway raises an explicitly small output budget to, clamped by resolved `ctx`. GLM sets `8192`: a small `max_tokens` truncates mid-`<think>` and both `content` and `reasoning_content` come back empty. |
| `max_output_tokens` | int | no | unset | Advertised to clients by `servedeck wire` (VS Code's `maxOutputTokens`). |
| `vision` | bool | no | `false` | Advertised by `servedeck wire`. Coerced with `bool()`, so `vision = "false"` is **true** — write a real TOML boolean. |
| `needs_tty` | bool | no | `false` | **Retired 2026-09-18.** Still parsed so an older `models.toml` loads, read by nothing. Nothing ever enforced it; `/etc/sysctl.d/90-servedeck.conf` pins `kernel.yama.ptrace_scope` to 0 at boot, which made "cannot boot unattended" false for the one model that set it, and doctor's `ptrace_scope (host)` row is now unconditional. Delete it from your entries. |

`servedeck` never takes `--served-model-name`, `--host`, `--max-model-len`,
`--gpu-memory-utilization` or `--port` from `flags`: `render_argv` computes all
five, and the live Flash-Next argv on 2026-09-18 is byte-for-byte the rendered
registry entry. `Restart=always`, `RestartSec=10` and `TimeoutStopSec=120` are
unit properties in `units.py`, not registry keys; a model's own
`--shutdown-timeout 30` belongs in `flags`, because it is vLLM's setting.
Pinned values today: `util` 0.96 flashnext / 0.95 qwen27b / 0.95 glm53;
`host_ram_gib` 95 flashnext / 155 glm53; `kv_cache_bytes_per_token` 17200 glm53.

`[models.<key>.reasoning]` takes `parser` (required if the table exists) →
`--reasoning-parser`, and `mirror_content` (bool, default false) → the gateway
copies `reasoning` into `reasoning_content` for chat-completions clients, with
`servedeck wire` reporting the model as `thinking` to VS Code.
`[models.<key>.tools]` takes `parser` (required if present) →
`--enable-auto-tool-choice --tool-call-parser`.

### `[models.<key>.presets]`

Each `<name> = { … }` is a served name of its own: it appears in
`--served-model-name` and in `GET /v1/models`, routes to the same weights, and
carries its table as a **`chat_template_kwargs` overlay** on the request, merged
with `setdefault` so it only fills a gap — a preset is a default a client could
not otherwise express, not an override of one it could.
`glm53-flash-low = { reasoning_effort = "low" }` arrives upstream as
`chat_template_kwargs.reasoning_effort = "low"`.

Do not add `main` or `local` as an alias or preset. The gateway synthesises both
as aliases of whichever model holds the main slot (`routes.py:127`), resolved
before the registry is consulted; a registry entry of that name would load and
then be shadowed.

### GLM client-compatibility switches

Three booleans on `[models.<key>]`, read by `glm_policies.py`. They must be real
TOML booleans: `capture = "false"` is a truthy string, and a switch that turns
*on* when its config says "false" is the worst available failure for a flag
whose job is keeping request bodies off the disk.

| Key | Default | Second consent | Why |
|---|---|---|---|
| `sanitize_tool_tags` | `false`; `true` for glm53 | no | GLM's streaming tool-call parser can miss a closing tag that straddles a token boundary, and the tag leaks into the parsed argument — captured live as `list_dir({"path</arg_key>": …})`, which the client rejects, costing the agent its step. Stripping markup that is never valid argument text is a repair, not a guess. It also switches on the answerless-turn count for that route. |
| `restore_reasoning` | `false` everywhere | no, but evidence | Puts reasoning back on the in-flight assistant turns of a request whose client echoed the `tool_call` ids and dropped the thinking; the output-side mirror cannot, because it can make thinking available to a client but not make the client send it back. Off because the proxy it is ported from records "three Xid 31 GPU faults followed within 40 minutes of it first firing, after 14 hours clean" — correlation, not proof, but this box has an active unrelated GPU-fault problem, so nothing that plausibly perturbs the GPU defaults to on. Turn it on for one session, watch telemetry, decide. |
| `capture` | `false` | **yes** — `SERVEDECK_GLM_CAPTURE=1`, or `python -m servedeck --glm-capture` | Consent, not an enable. With both set, the gateway keeps a 6-turn ring of that route's request/response bodies under `state/captures/<model>/` (directory 0700, files 0600, credential headers redacted). Bodies contain everything the user typed, which is why one switch is in the registry and the other deliberately is not: the predecessor defaulted this on, writing full bodies to `/tmp` with no retention. |

**Known gap, 2026-09-18:** `routes.py:255` assembles `RoutePolicies` from
`mirror_content`, the preset overlay, `min_output_tokens` and `ctx` only, and
does not copy these three fields, so all three are inert in the live gateway
however `models.toml` sets them. Only tests construct a `RoutePolicies`
carrying them; the fix belongs in `routes.py:route_for`.

### What `_validate` enforces

`[gpu]` present with both required keys and at least one `[models.*]` table;
every required field present, with `slot` one of the two words, `ctx` an int or
`"native"`, `aliases`/`flags` lists of strings, `presets` and each overlay
tables, `vram_mib` on every resident and `util` a number in (0, 1]; registry
keys matching `[a-z0-9-]+`; every id, alias and preset name unique across the
whole file, with a preset forbidden from shadowing any id or alias including its
own model's; ports unique and never 8000 (ats-optimizer) or 8010 (the gateway);
every `build` a key in `[builds]`; and `sum(resident vram_mib) + margin_mib <
total_mib`, so a main model always has room.

### Utilisation: pinned in the registry, checked against the card

A main model's utilisation is its registry `util` when it has one, and
`floor2((free_mib - margin_mib) / total_mib)` when it does not. A resident's is
always `floor2(vram_mib / total_mib)` plus a CUDA-context cushion check, so a
resident can never make a big model refuse to boot — its memory is not free.

With `util` pinned, `compute_util` compares it against what fits now and
**refuses** if less fits, naming the free and total MiB. It neither hands the
model more nor silently launches it lower, because a lower utilisation changes
the KV budget, the context that fits and the agent count with nothing said. That
is why the key was added: free-VRAM arithmetic yields 0.98 on an idle card (a
launch was rendered at 0.98 on 2026-09-17), which boots and then dies of CUDA
OOM once agents arrive (measured 2026-09-03), while Flash-Next has only served
at 0.96 and the 27B at 0.95. A per-launch override from the page or the API is
still allowed and the VRAM refusal still applies; a per-launch
`--max-model-len` above a model's pinned integer `ctx` is refused with a 409
rather than clamped.

## Environment variables

`SERVEDECK_PORT` raises `SettingsError` on a non-integer; the three `limits.py`
overrides silently fall back to their default instead. Nothing here is required.

| Variable | Default | Read in | For |
|---|---|---|---|
| `SERVEDECK_MODELS` | `<repo>/models.toml` | `settings.py:129` | The registry path. `SERVEDECK_MODELS_TOML` is a synonym. |
| `SERVEDECK_STATE_DIR` | `<repo>/state` | `settings.py:132` | `desired.json`, `wire`'s dated backups, captures, the measurement store. |
| `SERVEDECK_UNIT_PREFIX` | `model-` | `settings.py:135` | The systemd namespace servedeck owns. Only `model-` (production) or `sd-test-` (the suite); anything else fails loudly at startup rather than producing a discovery glob that matches nothing and reports a serving box as empty. |
| `SERVEDECK_HOST` | `127.0.0.1` | `settings.py:144` | Listen address; no auth exists, so do not expose it. `python -m servedeck --host` sets it in the environment so the app and uvicorn agree. |
| `SERVEDECK_PORT` | `8010` | `settings.py:143` | Listen port. `--port` sets it the same way. |
| `SERVEDECK_LOG_LEVEL` | `INFO` | `__main__.py:27` | Level for the `servedeck` logger tree, whose one stderr handler is what puts reconcile's decisions in the journal. An unrecognised name falls back to `INFO`; uvicorn's access log is unaffected and stays on. |
| `SERVEDECK_GPU_TOTAL_MIB` | `nvidia-smi`, else `0` | `limits.py:130` | Total VRAM for the capacity arithmetic. `0` means "cannot compute" and capacity blocks rather than sizing a KV cache against a number nobody measured. |
| `SERVEDECK_OVERHEAD_GIB` | `4.7` | `limits.py:133` | VRAM that is neither weights nor KV (activations, CUDA graphs). Measured 4.3-4.5 GiB on two very different models; 4.7 errs high on purpose. |
| `SERVEDECK_FRAG_MARGIN_MIB` | `4096` | `limits.py:134` | Fragmentation margin, reported as "thin" and never as "impossible": adding it to the *requirement* makes any utilisation above ~0.958 unsatisfiable, a bug this project shipped once. |
| `SERVEDECK_TRAINING_MARKERS` | `~/Projects/local_llm/run/training_in_progress:~/.cache/algotrading/training_in_progress` | `limits.py:98` | Colon-separated lock files; if one exists, something else wants the GPU and `doctor` fails. Read at call time. The empty string leaves no markers at all. |
| `SERVEDECK_HF_HUB_DIR` | `~/.cache/huggingface/hub` | `discovery.py:101` | Where the hub cache is read, which is what makes `ctx = "native"` and doctor's weights rows resolvable. Its docstring marks it "tests only"; treat it as a test seam, not an operator knob. |
| `SERVEDECK_GLM_CAPTURE` | unset | `glm_policies.py:542` | The master capture switch (`1`, `true`, `yes`, `on`), needed alongside a route's own `capture = true`. |

`limits.get()` caches on first call, so changing the three capacity overrides
means restarting the unit.

## `state/desired.json` — schema 3

What an operator last explicitly asked for, and the one thing servedeck
remembers across a restart. Read it; do not hand-edit it — `control` rewrites it
atomically (temp file in the same directory, fsync, `os.replace`), so an edit
made while servedeck runs is overwritten by the next action.

```json
{
  "version": 3,
  "main": "flashnext",
  "residents": [],
  "launch": {"flashnext": {"util": 0.96,
                           "argv": {"--max-model-len": "262144",
                                    "--max-num-seqs": "16"}}}
}
```

`main` is a registry key or `null`, `residents` a list of keys. `launch` (new in
schema 3, 2026-09-18) holds the settings of each model's last boot that actually
reached READY, and reconcile replays them; an `argv` value of `null` means
"remove this flag". Before it existed the allocator was write-only: a tuned
utilisation, context, agent count and KV-offload size lived in the argv of a
running process and nowhere else, so the next restart or reboot relaunched at
the registry defaults with utilisation recomputed from free VRAM, and nothing
said so. Recording only ready boots means a configuration that fails to boot is
never the one a reboot repeats.

Written only by `control.start` / `stop` / `switch` / `adopt`, never derived
from a probe: a crashed model must stay desired so reconcile brings it back, and
a model an operator stopped must stay stopped even though adopting a stray unit
would otherwise re-desire it. A v2 file migrates to v3 silently and in place
(the change is purely additive); a v1 file migrates with a warning and leaves
`desired.json.v1` beside it, and its `backend` was a launcher name rather than a
registry key, so confirm it names a model. Any other `version` is treated as
empty rather than guessed at, as is a missing, empty or unparseable file — this
file must never be the reason servedeck will not start. Live content on
2026-09-18: `main = "flashnext"`, no residents, `launch` empty, so a reconcile
would relaunch Flash-Next at the pinned `util = 0.96` with its registry flags.

## Why one file

`models.toml`'s predecessor was `local_llm/.config`, a shell file whose parser
silently skipped any line that was not `KEY="value"` and any value containing a
double quote. It parsed to an empty set of settings at least once,
`--limit-mm-per-prompt` could never be expressed in it, and three incidents came
out of it (`docs/imported-2026-09/local_llm/docs/SCRIPTS.md` and the 2026-09 notes; not re-verified here). TOML
parses or fails, and `load()` rejects what it cannot use — except an unknown
key, as above.
