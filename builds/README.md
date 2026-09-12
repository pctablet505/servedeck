# `builds/` — the three vLLM builds this box serves from

REDESIGN-2026-09-12.md §2.6: "don't lose the patches" becomes a property of
the repo. Every model in `models.toml` names a `build` (`[models.<key>].build`),
and every `build` is one key in `models.toml`'s `[builds]` table, pointing at a
venv:

| `[builds]` key | venv | serves |
|---|---|---|
| `stock` | `~/Projects/local_llm/.venv-llm-029` | `qwen27b` (the 27B, `qwen-server-run.sh`), `lfm2` (LFM2.5-350M — "borrows" this build per REDESIGN §5 decision 7, even though the live systemd unit happens to launch it from `vllm-qwen38next/.venv-next`'s python; see `builds/qwen38next/MANIFEST.toml`) |
| `qwen38next` | `~/Projects/vllm-qwen38next/.venv-next` | `flashnext` (`serve.sh`) |
| `glm53` | `~/Projects/vllm-glm53/.venv-glm53` | `glm53` (`serve-opt.sh`) |

Each `builds/<name>/` directory holds:

- **`MANIFEST.toml`** — what is actually installed in the venv right now: Python
  and package versions, the editable-install target (for the two forks), the
  fork tree's remote/branch/HEAD/`describe`, the upstream anchor where one is
  determinable, the ordered list of patch files that reproduce the running tree
  on top of that HEAD, which topic branches captured under `builds/patches/`
  are **not** part of what is actually running, and the CUDA/driver facts this
  box's builds depend on.
- **`check.sh`** — read-only. Verifies the live venv still matches
  `MANIFEST.toml`: Python version, package versions, the editable target, the
  tree's HEAD, and — the one that actually catches drift — that the tree's
  *current* `git diff` plus its current untracked non-`.bak` files, hashed,
  equal the hash of the exported patch files. If someone edits a fork checkout
  and forgets to re-export, this is what notices. Exit 0 if every check passes,
  1 otherwise, with one line per check.
- **`build.sh`** — the exact commands to reproduce the venv from scratch in a
  new directory: clone the fork remote at the recorded commit, `git apply` the
  patch files in order, create the venv, install, and any post-install step
  (the `lib64` shim). Supports `--dry-run`, which prints every step without
  running it.

## What "stock" means

`builds/stock/` is the plain `vllm==0.29.0` wheel from PyPI — no fork, no
patches, no source checkout. Its `MANIFEST.toml` leaves `[tree]` and
`[patches]` empty and says so; its `check.sh` skips the tree/patch checks
outright instead of failing on data that doesn't exist for a wheel install.

## The rule: patches are files, applied, never hand-edited in a checkout

`builds/patches/<name>/` (already populated, exported 2026-09-12) holds:

- `working-tree-tracked.patch` — `git diff` of every tracked file the fork
  checkout has modified relative to its own HEAD.
- `working-tree-untracked.patch` — the same shape, for untracked files the
  checkout has added (backup/scratch files — anything matching `*.bak*` or
  `*.reviewed` — are deliberately excluded; they are not part of what runs).
- `branches/<topic>/000N-*.patch` — `git format-patch` output for local topic
  branches that carry commits beyond the tree's HEAD but are **not merged
  into the branch that's actually installed**. `MANIFEST.toml`'s `[patches]`
  table names these explicitly and says they must **not** be applied by
  `build.sh`.
- `MANIFEST.txt` — the raw capture log from 2026-09-12 (branch, HEAD, remotes,
  `git status`, per-branch commit counts). `builds/<name>/MANIFEST.toml` is
  the structured, checked version of the same facts.

If a fork checkout needs to change, the change is made in the checkout, then
re-exported to these files (`git diff` / `git diff --no-index` / `git
format-patch`, same as this capture), then `check.sh` is re-run to confirm the
new export matches what's live. Editing a `.patch` file by hand, or "fixing"
the checkout without re-exporting, is exactly the failure `check.sh` exists to
catch.

## `builds/patches/glm53/GLM53-SM120-2026-08-29.patch`

Historical. It predates the September tuning pass and touches 9 files, all 9
of which are also in `working-tree-tracked.patch` (which touches 10 — the
extra file is `marlin_moe.py`, added later). Comparing `diff --git` file lists
between the two is sufficient to call it superseded — see
`builds/glm53/MANIFEST.toml`'s `[patches]` table; it is kept for history only
and is **not** in `build.sh`'s apply list.

## Rebuilding a fork is not routine

A fork rebuild is a multi-hour source compile (`vllm-qwen38next/SETUP.md`
quotes ~3-4 h first run, ~11 min warm/~40 min cold once the toolchain is
staged; `vllm-glm53/build.sh` is the same shape) that needs the GPU idle for
the whole thing and a matching CUDA toolchain pinned to what the driver on
this box actually supports (13.2.86 — see each `MANIFEST.toml`'s `[cuda]`
table). **`build.sh` has not been exercised end-to-end on this box** — only
`--dry-run` has been run, for all three builds, as part of this packet. Do not
run a real `build.sh` while the card is serving.

## Verification run, 2026-09-12

`check.sh` was run read-only against all three live venvs; see the packet
report for the three transcripts. `tests/test_builds.py` parses every
`MANIFEST.toml`, cross-checks `[models.*].build` in `models.toml` against
`[builds]`, confirms every listed patch file exists on disk, and runs
`check.sh --help` / `build.sh --dry-run` for all three builds (no network, no
GPU).
