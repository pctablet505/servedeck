# Merge notes — folding the `coldstart` fork back into Servedeck

**Date:** 2026-09-02

For a while there were two copies of this codebase: this repo (`servedeck`,
public) and a private fork at `~/Projects/coldstart` with the package renamed
to `coldstart/`. The fork predated `config.py`, so everything this repo makes
configurable, it hardcoded — and it also carried a month of genuine
improvement that belonged here.

This document records what moved, in which direction, and why. It is the
argument for every hunk, so a later reader does not have to re-derive it from
two diffs.

---

## Summary

Diffed all 17 shared modules pair-by-pair with the package rename normalised
away, plus `web/app.js`, `web/index.html`, and the test suite.

| Classification | Hunks | Where they went |
|---|---:|---|
| **(a) General** — improvement or bugfix that belongs upstream | **41** | ported into `servedeck/` |
| **(b) Local** — a fact about one machine | **13** | `servedeck.toml` |
| **Superseded** — this repo was already ahead; the fork's version discarded | **14** | nothing |
| **Rename noise** — `Coldstart`→`Servedeck` only, no behaviour | **21** | nothing |
| | **89** | |

Seven modules (`events`, `gpu`, `logtail`, `phases`, `smoke`, `shellconfig`,
`gateway`) differed **only** by the rename. They are byte-identical now.

Tests: **170 → 216 passing.**

---

## (a) General — ported into Servedeck

### `metrics.py` — prefill throughput (the flagship)

Taken wholesale. The fork's version is a strict superset.

- **Prefill throughput.** Derived from `vllm:prompt_tokens_total` minus
  `vllm:prompt_tokens_cached_total`, over the poll window — the same
  quantity vLLM's own log prints as "Avg prompt throughput". Subtracting the
  cached tokens is the point: a token served out of the prefix cache is
  counted in `prompt_tokens_total` but costs no prefill compute, so including
  it makes a cache hit look like a throughput record.
- **Lifetime prefill rate**, over `vllm:request_prefill_time_seconds_sum`.
  Prefill is bursty — at `--max-num-seqs 1` a 10k prompt prefills for ~10 s
  and then nothing prefills for minutes — so the *windowed* rate is unknown
  almost always. This one does not decay while the server idles, because its
  denominator is prefill seconds rather than wall seconds.
- **Idle is `None`, never `0.0`.** "0 tok/s" reads as "the machine got slow",
  which is the opposite of the truth. The UI renders `None` as an em dash.
- **The rate baseline is dropped on an unreachable window.** It used to
  survive, so the next successful scrape divided a fresh counter delta by a
  `dt` spanning the whole outage: a backend down for ten minutes came back
  reporting a plausible-looking throughput averaged over its own downtime.
  A 500 from a half-restarted `/metrics` is the same situation and drops it
  too.
- **A monotonic clock**, injectable. `time.time()` steps on an NTP
  correction, and a backwards step silently scales every throughput number
  the UI has ever shown.
- **A counter that went backwards means "new server"** — that window spans
  two processes and has no throughput, which is *unknown*, not 0.

Pinned by `tests/test_metrics.py` (11 tests, new — `metrics.py` had none,
which is why every bug above survived).

### `supervisor.py`

- **The adopted-server monitor fix.** `assert client is not None` in
  `_run_monitor` was reachable: `_adopt_ready()` calls the monitor with
  `already_ready=True`, which leaves `client` None, while the poll body is
  guarded on `self._tracker`, which `_adopt_ready()` has just set. The
  assertion fired on tick 1, the fire-and-forget task swallowed it, and
  liveness was never polled again — `/api/state` reported READY for a server
  whose port was dead. Now `if client is not None:`.
  Regression test: `test_adopted_server_monitor_notices_the_process_exiting`
  (verified to fail against the old line).
- **`repo_id` reaches the launcher.** `_build_launch` never passed the chosen
  model, so a launcher honouring `${MODEL:-...}` always booted its own
  hardcoded default no matter what the UI selected. `repo_id = "MODEL"` is
  now in the default `env_map`.
- **`BOOT_LOG_DIRNAME`** — `"boot_logs"` was spelled twice, in two files.
- **`_default_log_paths()` gained a fallback.** A backend that declares no
  `log_path` (its launcher has no log management of its own) now resolves to
  the newest log Servedeck opened for *that backend*, under
  `state/boot_logs/<name>-*.log`, and to nothing when there is none. Returning
  the wrong backend's log is worse than returning none: `phases.classify()`
  would match an error line from another model's run and file it as this
  run's `failure_code`.

### `app.py`

- **The upstream port is re-read on every poll**, not once at construction.
  Every start rewrites the shell config's `PORT`, but nothing re-read it — so
  starting a server on a different port left the metrics poller, the uptime
  lookup, the running-model probe, the own-VRAM discount and the `/v1` proxy
  all watching the old port for the life of the process, each reporting "not
  reachable" about a server that was serving fine. The poller is *rebuilt*,
  not re-pointed: it carries a throughput baseline belonging to the old
  server. A junk port (`0`, `99999`, non-numeric) is ignored rather than
  followed.
- **`max_model_len` comes from the RUNNING process**, read out of
  `/proc/<pid>/cmdline`, not from desired config. The serving line's "N ctx"
  had already been wrong twice from two stale sources (the UI slider, then
  `supervisor.max_model_len`); `desired` is what Servedeck *wants*, and for an
  adopted server the two need not agree at all.
- **`_running_model_id()`** — same argument for the model name.
  `--served-model-name` is an alias an operator may reuse across models.
- **`server_uptime_s`** from `/proc/<pid>/stat` field 22. The UI's "up Nm"
  was Servedeck's own uptime, so restarting the UI made a long-running server
  look freshly started.
- **`/api/models` returns `trust` and `weights_gib`.** The model card read
  both long before the payload carried either, so every model rendered
  "estimated" — a label SPEC.md §3 attaches "~25% optimistic historically"
  to, i.e. a claim, not decoration — and the weights-based arm of the
  serving-model match was dead code.
- **Boot facts are read from the backend's log, chosen by backend.** The old
  rule ordered candidates by which backend owns `rt.port`. Plus a `model_tag`
  guard: a log whose last `'model_tag'` is a different model is skipped, so a
  stale "GPU KV cache size: N tokens" line from another model's boot cannot
  reach the dashboard.
- **`serving_model` is cleared when the upstream goes down**, instead of
  showing the last model seen.
- **`_recover_if_server_returned()`** — a server started from a terminal after
  a blocker was cleared is adopted, instead of the dashboard sitting on a
  stale FAILED while the model serves happily (and a later crash being
  misfiled because nothing was tracking it).

### `web/`

- **Placeholder readings removed from `index.html`.** It shipped the design
  prototype's mock numbers as static text — `":8001 · 262,144 ctx · 99.3
  tok/s · up 12m"`, 92,571 MiB of VRAM in use, 8.8 GiB of KV, "6 requests ·
  synthetic test traffic", and a green **measured** provenance badge over all
  of it. Every one of those was on screen whenever its painter had not run,
  which includes the case where the backend is unreachable and the page is
  otherwise empty. Pinned by `test_page_placeholders_are_not_fabricated_readings`.
- The serving line was split into `paintServingMeta()` so it repaints on
  telemetry (2 s) rather than only on state (5 s) — it was showing readings up
  to 5 s old and skipping every other sample.
- `rateTxt` / `prefillTxt` / `uptimeTxt` / `trustTag` — null-aware formatters.
  The old line was `${liveMetrics.gen_tok_s || 0} tok/s`, which printed "0
  tok/s" both before the first telemetry event and whenever the server was
  idle.

### `capacity.py`

The `UNKNOWN_CAPACITY` finding's wording. The old text named
`model_type=='qwen4_exp'` and Flash-Next specifically in a message shown to
the user; the fork's is generic and says what to do about it ("Starting it is
how the real figure gets measured"). Strictly better for a model-agnostic
tool.

### `registry.py`

The `glm5_next` branch of the weights estimator — the same shape as the
`qwen4_exp` branch already there, and the same reason: routed experts are
host-offloaded, so on-disk size is not VRAM size, and estimating from it
produced a nonsensical negative KV budget. **Flagged:** this is a claim about
a public model family rather than about this box, which is why it went
upstream; but it is the second entry in a table that will keep growing, and
that table is arguably the wrong shape.

### `procctl.py`

- `process_uptime_s()` (new, general).
- `_detect_venv()` now builds its list of attributable trees from the
  configured backends, falling back to the two built-in ones. A backend that
  is registered everywhere else but missing from *this* check is a server
  nothing can adopt — so the supervisor can neither stop it nor notice it
  crashed.

---

## (b) Local — moved into `servedeck.toml`, not upstream

Everything here was a constant in the fork. None of it is in the package now.

| Fork location | What it was | Now |
|---|---|---|
| `paths.py` `VLLM_GLM53`, `SERVE_SH_GLM53`, `VENV_GLM53_DIR` | this box's GLM-5.3 tree, venv and launcher | `[backends.glm53]` `launcher` / `venv` |
| `paths.py` `SERVE_LOG` | `vllm-qwen38next/serve.log` | `[backends.flashnext] log_path` |
| `paths.py` `TRAINING_MARKERS` third entry | `~/Projects/AlgoTrading/run/training_in_progress` | `training_markers` |
| `registry.py` `KNOWN_ARCHS` `Glm5NextForConditionalGeneration` | one machine's model lineup in a public package | `[backends.glm53] architectures` |
| `preflight.py` `BACKEND_GLM53` | a backend name constant | (not needed — preflight reads the config) |
| `history.py` `SPEC_COLD_BOOT_RANGE_S["glm53"]` | measured 330–420 s cold boot | **UNRESOLVED, see below** |
| `supervisor.py` `_build_launch` glm53 branch | `SPEC`, `VLLM_MOE_DMA_STAGING`, `CPU_OFFLOAD_GB=104`, `KV_BYTES`, `AUTOTUNE` | `[backends.glm53.env]` |
| `supervisor.py` `_default_log_paths` glm53 branch | "glm53 has no fixed log, glob our own" | generalised: any backend with no `log_path` |
| `capacity.py` `GPU_TOTAL_MIB = 97887` | this card | `gpu_total_mib` (already config here) |
| `capacity.py` `TRAINING_MARKER_PATHS` (3 literals) | one developer's home dir | `training_markers` (already config here) |
| `web/app.js` `CTXS` `524288` | one model's ceiling, hand-added | derived from `model_max_ctx` |

The schema had to grow three things to express all of it:

- **`[backends.<name>.env]`** — fixed environment passed to the launcher
  verbatim. Servedeck never interprets it; a knob it understands is a knob it
  can get wrong. Note what is deliberately *absent* from the glm53 table:
  `MOE_BACKEND` and `VLLM_USE_DEEP_GEMM`, which `serve-opt.sh` sets and which
  are load-bearing for **correctness** on SM120. Overriding those from here
  would let the UI boot a server that loads, serves, and emits one token
  forever with no error.
- **`log_path` is now optional**, and **`writes_own_log`** says whether the
  launcher redirects into it itself. Two distinct facts that were previously
  conflated: which file an *adopted* server's boot numbers are in, versus
  which file must also be tailed during a *launch*. Tailing a hand-launch log
  during a fresh launch would replay a previous boot into the phase machine.
- **`env_map` accepts an empty table** as "this launcher reads no
  environment" (the inline backend takes its settings from a file). It used
  to fall back to the defaults on any falsy value.

---

## Fixes found while merging (neither tree had them)

- **`supervisor.py` used `config` without importing it.** `_default_log_paths()`
  raised `NameError` on every adoption. Nothing covered that path.
- **The start gate was `backend not in ("flashnext", "inline")`** — a literal
  that refused every backend added through `servedeck.toml`, which is the one
  thing configuration exists to allow. A model whose architecture resolved to
  a configured backend was listed as servable, offered in the UI, and then
  refused at Start with "no model/backend/port configured".
- **`training_markers` in the config was parsed and then read by nothing.**
  The only source was `$SERVEDECK_TRAINING_MARKERS`, and *nothing at all*
  stat'ed the paths, so `LiveFacts.training_markers` was always empty and the
  `TRAINING_MARKER` block could never fire. A configured guard that cannot
  fire is worse than an absent one: it reads as switched on. Now sourced from
  the config (env var still overrides) and actually stat'ed in `_estimate()`.
- **`VENV_MISSING` / `LAUNCHER_MISSING` preflight were skipped** for any
  backend not named `flashnext` or `inline` — i.e. for the backend most
  likely to be misconfigured, the one someone just added. C9 records the
  stale venv path as the single largest boot-failure class.
- **The unattended-restart gate was a backend name.** Whether a launcher needs
  a terminal is a property of that launcher (`sudo sysctl`, which silently
  no-ops without a tty) and differs from machine to machine. Now `needs_tty`,
  falling back to the old flashnext-only behaviour when nothing declares the
  backend.
- **`registry.KNOWN_ARCHS` could only be extended by editing the source.**
  `arch_backends()` now layers `[backends.<name>] architectures` over it.

---

## UI changes beyond the merge

1. **The context ladder is built from the model, not from a literal.**
   `const CTXS = [8192 … 262144]` was wrong in both directions: it offered
   lengths a small model cannot reach (the engine refuses at boot, minutes
   later) and it capped a large one. The registry has parsed
   `max_position_embeddings` all along — it reads **1,048,576** for GLM-5.3,
   so three quarters of that model's range was unreachable from the UI.
   `ctxChoices()` now walks powers of two up to the selected model's own
   ceiling, always ending on the ceiling itself, with
   `DEFAULT_MAX_CTX = 262144` only when nothing is known. Switching to a
   smaller model clamps the current selection down instead of leaving a value
   the model cannot serve.
2. **The serving line shows `http://localhost:PORT`, not `:PORT`.** It is the
   one place a user goes for the address to paste into a client, and a bare
   port is not an address.

Both are pinned by tests in `tests/test_ui.py`, each verified to fail against
the old code.

---

## Superseded — this repo was already ahead

The fork predates `config.py`, so its version of these was discarded outright:

- `capacity.py`'s hardware limits: the fork had `GPU_TOTAL_MIB = 97887` and
  three literal marker paths; this repo reads them from config.
- `app.py`'s SSE hub: the fork has a local `Hub` class with no `Last-Event-ID`
  replay, so a reconnecting browser silently lost the events it missed. This
  repo's `events.EventHub` replays them.
- `app.py`'s `_candidate_logs()`, `_default_port()`: config-driven here.
- `WEB = HERE / "web"` — the assets live *inside* the package here, so they
  ship in the wheel. The fork serves them from the project root.
- Test fixtures: this repo's are anonymised (`/home/user`), the fork's carry
  a real home directory. Kept this repo's.
- `tests/test_capacity.py::test_training_markers_are_configurable_not_hardcoded`
  and `test_registry.py`'s `REAL_MEASUREMENTS` placeholder — the fork reverted
  both to machine literals.

---

## Unresolved / flagged

1. **`history.SPEC_COLD_BOOT_RANGE_S`** still hardcodes `"flashnext":
   (240, 600)` upstream, and the fork's measured `"glm53": (420, 900)` was
   **not** ported. A cold-boot ETA is a per-machine, per-model measurement —
   it belongs in the config or in `state/measurements.json`, not in a literal
   keyed by backend name. Leaving it means a glm53 boot has no ETA range;
   adding it upstream would have been exactly the hardcoding this merge exists
   to remove. Needs a schema decision.
2. **`paths.py` still spells out `~/Projects/local_llm` and
   `~/Projects/vllm-qwen38next`.** Pre-existing; the config path now covers
   every *new* backend, and `preflight`/`procctl` prefer config over these,
   but the two legacy constants remain as the no-config fallback.
3. **`shellconfig` reads `SERVEDECK_URL`; `local_llm/.config` on this box
   sets `COLDSTART_URL`.** Only affects the base URL reported in the config
   summary — supervision and metrics use `PORT`, which is unchanged. Rename
   the key in `.config` (or add both) when retiring the fork.
4. **`preflight`'s ptrace checks are still keyed on `backend == "flashnext"`.**
   Correct on this box, and `needs_tty` now covers the restart gate, but the
   `PTRACE_BLOCKS_PLE` / `PTRACE_LEFT_RELAXED` checks would need a
   `needs_ptrace_relaxed` config flag to be truly model-agnostic. Not changed:
   out of scope, and the current behaviour is right here.
5. **`tests/conftest.py` is new and load-bearing.** `config._config_file()`
   picks up `servedeck.toml` next to the package, and that file is gitignored
   because it describes one machine — so without isolation the suite's results
   would depend on whether the person running it has a config and what is in
   it. Every test now runs against one small fixed configuration.
6. **`~/Projects/coldstart` was left in place and untouched**, per
   instruction. Nothing in it was modified, including its `.pytest_cache`.
   Retire it once this version has been run.
