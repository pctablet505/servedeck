# Configuration (v2)

**`models.toml` in the repo root is the single source of truth.** Every model's
public name, aliases, port, context, parsers, launch flags and environment come
from it, and everything else — the gateway's routing table, the `vllm serve`
argv, the generated client configs, `doctor` — is derived. The reason it is one
file is root cause R1 in [REDESIGN-2026-09-12.md](REDESIGN-2026-09-12.md).

**`servedeck.toml` is gone.** So is `local_llm/.config`, the `[backends.*]`
tables, `env_map`, `launcher`, `log_path` and `writes_own_log`. Servedeck now
builds the command line itself; there is no launcher script to pass settings to.

---

## `models.toml`

Loaded and validated by `servedeck/models.py`. A problem is one
`RegistryError` with one line saying what and where — never a bare `KeyError`.

### `[gpu]` — required

```toml
[gpu]
total_mib  = 97887
margin_mib = 1024          # never handed to any model
```

Both keys are required. `margin_mib` is subtracted from free VRAM before a
`main` model's utilisation is computed.

### `[builds]` — build name → venv location

```toml
[builds]
stock      = "~/Projects/local_llm/.venv-llm-029"
qwen38next = "~/Projects/vllm-qwen38next/.venv-next"
```

Each model's `build` must be a key here. The value is the build's **venv**, and
the `vllm` binary is looked for at `<value>/bin/vllm` — unless the value's last
path segment is already `bin`, in which case it is used as written. `~` is
expanded.

The longer table form is also read, so a build can carry more than a path:

```toml
[builds.glm53]
venv = "~/Projects/vllm-glm53/.venv-glm53"
```

### `[defaults.env]` — merged into every model

```toml
[defaults.env]
VLLM_USE_FLASHINFER_SAMPLER = "0"
```

Merged at load time into each model's own `env`; a model's own key wins on
conflict. Downstream code only ever sees the merged result.

### `[models.<key>]`

The table key is the registry key: it is the CLI argument, the `/api/*` path
segment and the unit name suffix (`model-<key>`), so it must match
`[a-z0-9-]+` or `units.py` refuses to start it.

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | The canonical public name. First entry in `--served-model-name`. |
| `repo` | yes | Hugging Face repo id; `vllm serve`'s positional argument. |
| `slot` | yes | `"main"` (exclusive, one at a time) or `"resident"` (co-resident, budgeted). |
| `port` | yes | Loopback port this model's own vLLM listens on. Clients never see it. |
| `build` | yes | A key in `[builds]`. |
| `ctx` | yes | An integer, or `"native"` → the checkpoint's own `max_position_embeddings`. |
| `aliases` | no | List of strings. Every name a client has ever used. Added to `--served-model-name`. |
| `reasoning` | no | `{ parser = "...", mirror_content = true }`. `parser` required if the table is present. |
| `tools` | no | `{ parser = "..." }`. `parser` required if the table is present. Enables `--enable-auto-tool-choice`. |
| `flags` | no | List of strings appended verbatim to the argv, after the computed flags. |
| `env` | no | Table of string env vars for the unit. |
| `presets` | no | `{ <name> = { <request overlay> } }`. Each preset is a served name carrying a `chat_template_kwargs` overlay. |
| `vram_mib` | residents only | The resident's VRAM budget, from which its utilisation is derived. |
| `min_output_tokens` | no | Floor the gateway raises an explicitly-small output budget to. |
| `max_output_tokens` | no | Advertised to clients by `servedeck wire` (VS Code's `maxOutputTokens`). |
| `vision` | no | Advertised to clients by `servedeck wire`. Default `false`. |
| `needs_tty` | no | This model cannot boot unattended. `doctor` checks `ptrace_scope` for it. Default `false`. |

`servedeck` never writes `--served-model-name`, `--host`, `--max-model-len`,
`--gpu-memory-utilization` or `--port` from `flags`: it computes all five.

### Validation rules `_validate` enforces

- `[gpu]` is present with both `total_mib` and `margin_mib`.
- At least one `[models.*]` table exists.
- Every required field above is present; `slot` is `main` or `resident`;
  `ctx` is an int or `"native"`; `aliases` and `flags` are lists of strings;
  `presets` and each preset's overlay are tables.
- A `resident` has a `vram_mib`.
- **Every id, alias and preset name is unique across the whole file**, and a
  preset may not shadow any id or alias, including its own model's.
- **Ports are unique**, and never `8000` (ats-optimizer) or `8010` (the gateway).
- Every model's `build` is a key in `[builds]`.
- `sum(resident vram_mib) + margin_mib < total_mib`, so a main model always has
  room.

### Utilisation is computed, never configured

There is no `util` field. At launch, `control.py` asks `nvidia-smi`:

```
main:      floor2((free_mib - margin_mib) / total_mib)      # everything free
resident:  floor2(vram_mib / total_mib)                     # its own budget
```

Always floored to two decimals, never rounded up. This is decision 4 of the
design: a resident can no longer make a big model refuse to boot, because its
memory is simply not free.

---

## Environment variables

Five, and that is the whole surface (`servedeck/settings.py`):

| Variable | Default | Meaning |
|---|---|---|
| `SERVEDECK_MODELS` | `<repo>/models.toml` | Path to the registry. (`SERVEDECK_MODELS_TOML` is accepted as a synonym.) |
| `SERVEDECK_STATE_DIR` | `<repo>/state` | `desired.json`, `wire`'s dated backups, the measurement store. |
| `SERVEDECK_UNIT_PREFIX` | `model-` | The systemd namespace servedeck owns. **Only `model-` or `sd-test-`**; anything else is rejected at startup. |
| `SERVEDECK_HOST` | `127.0.0.1` | Listen address. Do not expose this; there is no auth. |
| `SERVEDECK_PORT` | `8010` | Listen port. |

A bad `SERVEDECK_UNIT_PREFIX` fails loudly at startup rather than producing a
`Control` whose discovery glob matches nothing — which would report a serving
box as empty.

Three capacity overrides plus the marker list live in `servedeck/limits.py`,
because they describe the *card*, not any model:

| Variable | Default | Meaning |
|---|---|---|
| `SERVEDECK_GPU_TOTAL_MIB` | `nvidia-smi`, else `0` | Total VRAM. `0` means "cannot compute", and capacity refuses to guess. |
| `SERVEDECK_OVERHEAD_GIB` | `4.7` | VRAM that is neither weights nor KV (activations, CUDA graphs). Measured 4.3–4.5 GiB on two very different models; 4.7 errs high on purpose. |
| `SERVEDECK_FRAG_MARGIN_MIB` | `4096` | Fragmentation margin. A *warning* threshold, never part of the requirement — adding it to the requirement makes any util above ~0.958 look impossible. |
| `SERVEDECK_TRAINING_MARKERS` | two built-in paths | Colon-separated lock files. If one exists, something else wants the GPU and `doctor` fails. |

The two built-in marker paths are `~/Projects/local_llm/run/training_in_progress`
and `~/.cache/algotrading/training_in_progress`.

---

## Client configs

Generated, not hand-written. `servedeck wire` rewrites, idempotently and with a
dated backup under `<state_dir>/backups/<YYYY-MM-DD>/`:

- `~/.config/Code/User/chatLanguageModels.json` — one `servedeck` group, one
  entry per served name and per preset, pointing at
  `http://localhost:8010/v1/chat/completions`.
- `~/.codex/config.toml` — one provider at `http://127.0.0.1:8010/v1`.
- `~/.kimi-code/config.toml` — same shape.

Only the tables servedeck owns are rewritten; every other group, provider and
hand-written comment is left byte-for-byte intact. Run `servedeck wire` with no
flags for the diff, `--apply` to write it. Then `servedeck doctor` proves it.
