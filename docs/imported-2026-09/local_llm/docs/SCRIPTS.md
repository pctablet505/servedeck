# The shell scripts of ~/Projects — what runs each one, and when

Written 2026-09-10 after the phase-2 estate cleanup. **29 shell scripts survive**
(31 before; `flashnext-reasoning-proxy/start.sh` and `stop.sh` were deleted —
see [Deleted](#deleted-in-this-pass)).

This file exists because the sprawl came from nobody being able to answer one
question per script: *what starts this, and when?* A script with no answer is
either dead or a trap. Every row below answers it. **Add a row when you add a
script; if you cannot fill in the "run by" column, do not add the script.**

Read with `docs/RUNBOOK.md` (operating procedures) and `LOCAL_LLM_SETUP.md`
(why the server kept dying). This file is only the map.

---

## What is actually serving right now

**As of 2026-09-10 22:25 the GPU is FREE — :8001 is down.** It was up when this
cleanup started (pid 3102188, parent `bash .../vllm-qwen38next/serve.sh` pid
3102177) and shut down gracefully at 22:25:44 — its log ends with
`Application shutdown complete`, not a fault — 25 seconds after another agent
rewrote `.config` (see [Blocker 2](#blocker-2--resolved-by-another-agent-at-2225)).
`nvidia-smi --query-compute-apps` is empty; `llm status` says `✗ server :8001 down`.
The table below is what the estate *runs*, with :8001's current state called out.

| Port | Process | Started by | Never touch without freeing the GPU |
|---|---|---|---|
| 8001 | vLLM, `mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4` — **currently DOWN** | `bash ~/Projects/vllm-qwen38next/serve.sh` (parent of the vLLM) | yes |
| 8005 | flashnext reasoning proxy | `flashnext-reasoning-proxy.service` (enabled, active) | no |
| 8010 | servedeck dashboard | `servedeck.service` (enabled, active) | no |
| 8000 | ats-optimizer | `ats-optimizer.service` — **unrelated project**, leave alone | no |

`~/Projects/vllm-qwen38next/serve.sh` is the single highest-risk file in the
estate: a live process is executing it. Never edit it in place and never move
it while :8001 is up. Edit by writing a new file and `mv`-ing it over.

---

## local_llm — the launcher layer

| Script | Run by | When |
|---|---|---|
| `llm` (not `.sh`, but the real entry point) | operator, by hand | `llm start/stop/status`. Reads `.config`, maps `BACKEND` → a serve script, or → `bin/serve-model.sh` when `LAUNCHER=unified`. |
| `codex-qwen.sh` | operator, by hand | drives Codex CLI against the local server; `qwen`/`resume`/`openai`/`stop`/`status`/`health`. Delegates `BACKEND=flashnext` to `vllm-qwen38next/serve.sh`. |
| `bin/serve-model.sh` | `llm` **only when `LAUNCHER=unified`** | the profile-driven unified launcher (profiles in `profiles/*.env`). **Currently dormant — `.config` has no `LAUNCHER` key, and `llm:34` defaults `LAUNCHER="legacy"`.** See [Cutover](#the-cutover-that-has-not-happened). |
| `bin/qwen-server-run.sh` | `qwen-vllm.service` `ExecStart` | **unit is disabled and inactive.** Guards (training-marker, GPU headroom), log rotation, then `exec vllm serve` for the Qwen3.8-27B "inline" backend. |
| `bin/qwen-server-record-death.sh` | `qwen-vllm.service` `ExecStopPost` | **unit disabled.** Writes the death record. Also the *reference document* for Xid classification: `servedeck/gpu.py` and `coldstart/gpu.py` both cite it by path, and `servedeck/paths.py:43` resolves it. Deleting it would orphan those citations. |
| `bin/qwen-vllm-watchdog.sh` | `qwen-vllm-watchdog.service`, driven by `.timer` | **both disabled.** Un-sticks `qwen-vllm.service` after a systemd burst-limit lockout. |
| `bin/soak-27b-overnight.sh` | operator, by hand | the 27B overnight Xid soak. Exercised in dry-run by `tests/test_soak_config_isolation.sh` and `tests/test_serve_model_golden.sh`. |
| `bin/battery-watchdog.sh` | **nothing** | manual only. Polls `upower`; on UPS battery writes `run/BATTERY_PAUSE` and tries to stop `qwen-vllm.service`. No unit, no timer, no cron references it. Kept because it is the only UPS-aware pause mechanism; see [Owner decisions](#owner-decisions). |

### local_llm/tests — the gates (8 scripts, all run by hand)

| Script | What it proves |
|---|---|
| `test_serve_model_golden.sh` | the unified launcher reproduces each legacy serve script's argv byte for byte. **The gate that guards the cutover — and it is currently running blind, see below.** |
| `test_unified_cutover_safety.sh` | `llm` dispatches to the right launcher, and `llm stop` recognises a `bin/serve-model.sh` wrapper. |
| `test_start_concurrency.sh` | two concurrent `llm start`s leave exactly one server. **Flaky as of 2026-09-10** (see [Owner decisions](#owner-decisions)). |
| `test_dry_run_no_side_effects.sh` | `DRY_RUN=1` writes nothing — no logs, no death records. |
| `test_soak_config_isolation.sh` | the soak script's config snapshot cannot leak into the live `.config`. |
| `test_llm_config_warn.sh` | `.config` lines that do not match `KEY="value"` are reported, not silently skipped. |
| `test_llm_log_rotation.sh` | log rotation keeps the symlink and the boot history straight. |
| `test_llm_port_pid.sh` | port/pid discovery never matches the calling shell (`pgrep -f` self-match). |

Run them all: `cd ~/Projects/local_llm && for t in tests/*.sh; do bash "$t"; done`

---

## vllm-qwen38next — the Flash-Next backend (the :8001 model; stopped since 22:25)

| Script | Run by | When |
|---|---|---|
| `serve.sh` | **the live server right now**; also `serve-abliterated.sh` (`exec`s it), `serve-tuned.sh`, and `codex-qwen.sh:115` | owns every flag, env var and ptrace handling for Flash-Next. Defaults `MM_LIMIT_JSON` to `{"image":2,"video":0}` (images ON since 2026-09-10). |
| `serve-abliterated.sh` | `llm` for `BACKEND=flashnext`; golden-tested | thin wrapper: picks the abliterated checkpoint, reports the PLE dtype, then `exec "$HERE/serve.sh"`. |
| `serve-tuned.sh` | operator, by hand; golden-tested | the tuned-profile variant (`LAUNCHER_PROFILE=flashnext-tuned` is the unified equivalent). |
| `build.sh` | operator, by hand | source build of the fork in `src/`. `MAX_JOBS=40 NVCC_THREADS=2 ./build.sh`. |
| `download.sh` | `qwen38next-download.service` `ExecStart` | **unit disabled and inactive.** Reboot-surviving resumable fetch of the ~130 GB checkpoint. Keep: it is the only thing that survives a reboot mid-download. |
| `fetch_fa.sh` | operator, by hand, before `build.sh` | clones vllm-flash-attn without the ROCm/composable_kernel submodule. Referenced by `build.sh`'s comments and `SETUP.md:191`. |

---

## vllm-glm53 — the GLM-5.3 backend (not currently serving)

| Script | Run by | When |
|---|---|---|
| `serve-opt.sh` | `llm` for `BACKEND=glm53`; golden-tested | the optimised GLM launcher (expert CPU-offload, hot-slot pinning, DMA staging, MoE backend, spec decode). |
| `serve.sh` | golden-tested (`glm53-legacy` profile) | the original, pre-optimisation GLM launcher. Kept because the `glm53-legacy` profile asserts parity against it. |
| `build.sh` | operator, by hand | source build of the GLM fork. |
| `fetch_fa.sh` | operator, by hand, before `build.sh` | as above, for this tree. |
| `build_rust.sh` | **upstream vLLM's own docker builds** | **NOT OURS.** Tracked on branch `glm-release`; referenced by `docker/Dockerfile.cpu`, `.rocm`, `.rocm_gfx1250`, `.xpu`, `.github/CODEOWNERS:53`, and `tests/tools/test_docker_build_metadata_args.py:174`. Deleting it dirties the fork. |
| `build_vllm_ppc64le.sh` | **nothing on this box** | **NOT OURS.** Also tracked upstream. Purpose-dead here (`uname -m` = `x86_64`) but deleting it dirties the fork checkout. Leave it. |
| `memwatch.sh` | operator, by hand | samples host/GPU memory during a model load; `VmPin` separates pinned-by-the-offloader from page cache. Unreferenced by anything automated. |
| `setup_swap.sh` | operator, by hand, **needs sudo** | creates a swapfile so the desktop survives a loader spike. Unreferenced. |
| `memtest.sh` | **nothing — and do not run it** | ⚠️ line 14 does `for p in $(pgrep -f "VLLM::"); do kill -KILL "$p"`. **Running it today kills the Flash-Next workers on :8001.** Unreferenced by anything. |
| `moe_sweep.sh` | **nothing — and do not run it** | ⚠️ matches `bin/vllm serv[e]\|VLLM:[:]` and sends TERM then KILL, unfiltered by model or venv. **Running it today kills the live server.** Unreferenced by anything. |

`memtest.sh` and `moe_sweep.sh` also violate the standing no-`pgrep -f` rule
(`pgrep -f` matches the invoking shell on this host). They were kept, not
deleted, because the audit proved them *unreferenced* but did not clear them as
*dead* — they are one-shot experiment harnesses with real content. See
[Owner decisions](#owner-decisions).

---

## servedeck — the live dashboard

| Script | Run by | When |
|---|---|---|
| `run.sh` | operator, by hand | **foreground dev runner only.** The deployed service is `servedeck.service`, which runs uvicorn directly and never calls this. Referenced by `docs/MIGRATION.md:121`. |
| `stop.sh` | operator, by hand | stops whatever listens on `:8010`, found by port and never by `pkill -f`. `systemctl --user stop servedeck` is the supported way when the unit is running. Referenced by `docs/MIGRATION.md:120` (as `coldstart/stop.sh`). |
| `setup.sh` | operator, by hand | ⚠️ creates `.venv` **and offers to install `systemd/servedeck.service`** — the shipped unit that assumes the repo lives at `$HOME/servedeck` and took the dashboard down with `status=203/EXEC` on 2026-09-10 20:29. The installed unit at `~/.config/systemd/user/servedeck.service` is the corrected one and carries a "do not let setup.sh overwrite this" header. **Read that header before running `setup.sh` again.** Referenced by `docs/MIGRATION.md:113`. |

---

## coldstart — the previous dashboard, kept for its state

**Do not delete or move this tree.** `servedeck` reads it *live*:
`servedeck/paths.py:101-102` sets `LEGACY_COLDSTART_STATE_DIR` to
`~/Projects/coldstart/state`, `legacy.py` merges it, `history.py:131` and
`registry.py:475` consume the merge, and `servedeck.service` sets **no**
`SERVEDECK_LEGACY_STATE_DIR` override. Verified 2026-09-10:
`coldstart/state/history.jsonl` holds 48 boot records, `servedeck/state/history.jsonl`
holds 24, they overlap in 16 — **32 records exist only in coldstart** and the
dashboard is displaying them right now. The path is hardcoded, so moving the
tree breaks the read as surely as deleting it. Absence does not crash anything
(`legacy.py:_legacy_file` returns `None`), it just makes the history quietly
shorter — the worst kind of loss, because nothing reports it.

`coldstart.service` is disabled but its `ExecStart` target
(`coldstart/.venv-gui/bin/uvicorn`) exists, so the documented rollback from
servedeck to coldstart is currently real.

| Script | Run by | When |
|---|---|---|
| `run.sh` | operator, by hand | foreground dev runner (`.venv-gui`, `coldstart.app:app`). Near-identical to `servedeck/run.sh` — differs only in name, venv and package. |
| `stop.sh` | operator, by hand | frees `:8010`; named in `servedeck/docs/MIGRATION.md:120` as the first step of the migration. |
| `setup.sh` | operator, by hand | creates `.venv-gui`, offers to install `coldstart.service`. Same shape and same hazard as servedeck's. |

These three are byte-for-byte the servedeck triple modulo the name, the venv
directory and the package. They were **not** consolidated into one shared
script: coldstart's value is being a frozen, self-contained rollback, and a
cross-tree dependency on a shared launcher is exactly what would make the
rollback fail on the day it is needed.

---

## flashnext-reasoning-proxy

No scripts. systemd is the only control path:

```
systemctl --user start  flashnext-reasoning-proxy
systemctl --user stop   flashnext-reasoning-proxy
systemctl --user status flashnext-reasoning-proxy
journalctl --user -u flashnext-reasoning-proxy -f     # or: tail -f proxy.log
curl -s http://127.0.0.1:8005/__proxy/health
```

`tests/test_systemd_unit_contract.py` pins the unit's `ExecStart` to the
invocation the deleted scripts documented, and fails if `start.sh`/`stop.sh`
reappear.

---

## Deleted in this pass

| Script | Proof it was safe to delete |
|---|---|
| `flashnext-reasoning-proxy/start.sh` | Its resolved argv, the unit's `ExecStart`, and `/proc/1094768/cmdline` (the process serving :8005) were **byte-identical**; cwd and `FLASHNEXT_UPSTREAM` matched too. The unit's own comment already said it must not call `start.sh` (nohup + own pidfile would leave systemd supervising a dead process). Only references anywhere were `proxy.py`'s docstring and that comment. Recoverable: `git show 694e1e2:start.sh`. |
| `flashnext-reasoning-proxy/stop.sh` | Redundant *and wrong*: it killed whatever `proxy.pid` named, but under systemd no `proxy.pid` is ever written, so `./stop.sh` printed "no pidfile; nothing to stop", exited 0, and left the proxy serving — it reported success for work it had not done. Recoverable: `git show 694e1e2:stop.sh`. |

Commit: `flashnext-reasoning-proxy` @ `185c6b6`, which carries the full proof
and the mutation results for the replacement guard.

---

## The cutover that has not happened

`bin/serve-model.sh` can replace `vllm-glm53/serve-opt.sh`,
`vllm-glm53/serve.sh`, `vllm-qwen38next/serve-abliterated.sh`,
`vllm-qwen38next/serve-tuned.sh` and `vllm-qwen38next/serve.sh` — **five of the
29 scripts retire the day `LAUNCHER=unified` is proven.** It has not been
flipped, and it must not be flipped from a session that cannot free the GPU.

### Blocker 1 — the gate is one live server away from seeing

`tests/test_serve_model_golden.sh` contains 48 assertions. It reported
`ALL 46 ASSERTIONS PASSED` at the start of this cleanup, with two skips. After
another agent corrected `.config` at 22:25 it reports **`ALL 47 ASSERTIONS
PASSED`** with one skip left:

```
SKIP: no Flash-Next server on :8001 to compare against, or .config's BACKEND is not flashnext
```

That one is the file's own strongest check (section 10): that the unified
launcher reproduces the argv of the vLLM **actually serving on :8001**. It reads
`MODEL_REPO`/`SERVED_NAME`/`PORT`/`MAX_MODEL_LEN`/`MAX_NUM_SEQS`/`GPU_MEM_UTIL`/
`EXTRA_ARGS` out of `.config` and diffs the result against `/proc/<pid>/cmdline`.
`.config` is now correct, so **the only thing still missing is a running
Flash-Next server on :8001.** Bring one up and the gate reaches 48/48 and the
cutover becomes validatable. Until then it is not.

A skip is not a pass. `grep -c '^SKIP'` on the gate's output is the number to
watch; the headline count moves up as skips resolve, which is exactly how a
"46 passed" line hid a missing check.

### Blocker 2 — RESOLVED by another agent at 22:25

**This is fixed. Recorded here because it was live for most of 2026-09-10 and
the failure mode will recur.** Until 22:25:19 today, `~/Projects/local_llm/.config`
had a header saying flashnext was active and an uncommented `EXTRA_ARGS` that was
flashnext's, while lines 72–78 uncommented set the GLM block — **immediately
below an identical, correctly commented-out copy of the same block** (lines
65–71), which is what made it an editing accident rather than a decision:

```
72:BACKEND="glm53"
73:MODEL_REPO="dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4"
74:SERVED_NAME="glm53-flash"
75:PORT="8002"
76:MAX_MODEL_LEN="327680"
77:MAX_NUM_SEQS="1"
78:GPU_MEM_UTIL="0.95"
```

Parsed with `llm`'s own `^[[:space:]]*([A-Z_]+)="([^"]*)"` regex, the file
yielded `BACKEND=glm53` while flashnext was the thing actually serving. **Two
consequences, both independent of any cleanup:**

1. A bare `llm start` would have launched GLM-5.3 on :8002 carrying flashnext's
   `EXTRA_ARGS` (`--mamba-ssm-cache-dtype`, `--prefix-match-unit`) — onto a card
   already held by Flash-Next.
2. A bare `llm stop` targeted :8002, so it would have reported success without
   stopping the live server on :8001.

This cleanup deliberately did **not** touch it — correcting it changes what
`llm start` and `llm stop` do to a running server. Another agent fixed it at
22:25:19, and the values it now carries match, field for field, the ground truth
this cleanup had independently read out of the live process (`/proc/3102188`)
before it stopped:

```
BACKEND="flashnext"
MODEL_REPO="mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4"
SERVED_NAME="qwen38-flash-next"
PORT="8001"
MAX_MODEL_LEN="262144"
MAX_NUM_SEQS="16"
GPU_MEM_UTIL="0.96"
```

`EXTRA_ARGS` at line 33 was already flashnext's and was correctly left alone.

**The lesson to keep:** this file has now caused the same class of incident three
times (see the `NOTE 2026-09-03` preserved in the file itself). The header and
the values are two statements of the same fact, and nothing checks that they
agree. `tests/test_llm_config_warn.sh` catches *unparseable* lines; nothing
catches a *parseable* line that contradicts the header or the running server.
That gap is worth a gate.

Note also that `.config` lines 22–32 are now **stale**: they warn that "A BARE
`llm start` WILL SERVE WITH IMAGES DISABLED", but `vllm-qwen38next/serve.sh:200`
has defaulted `MM_LIMIT_JSON` to `{"image":2,"video":0}` since 2026-09-10, and
`profiles/flashnext.env:39` carries the same `P_MM_LIMIT`. Images are ON by
default on both paths now.

### The cutover, in order

```bash
cd ~/Projects/local_llm
cp -a .config ".config.bak-precutover-$(date +%Y%m%d-%H%M%S)"

# 1. DONE (another agent, 22:25). Verify rather than repeat:
grep -n '^[A-Z_]\+="' .config     # BACKEND must be flashnext, PORT 8001

# 2. Bring the Flash-Next server up on :8001 on the LEGACY path, so the gate has
#    a live baseline to compare against. This is what unskips section 10:
./llm start                       # images default ON since 2026-09-10
curl -sf --retry 60 --retry-delay 5 --retry-all-errors http://127.0.0.1:8001/health

# 3. The gate must now SEE. Expect 48 assertions and ZERO skips:
bash tests/test_serve_model_golden.sh 2>&1 | tee /tmp/golden-live.txt
grep -c '^SKIP' /tmp/golden-live.txt      # must print 0  <-- the number that matters
tail -1 /tmp/golden-live.txt              # must say ALL 48 ASSERTIONS PASSED

# 4. Only then add the flag:
printf 'LAUNCHER="unified"\n' >> .config

# 5. Re-run the whole local_llm gate set. All must pass:
for t in tests/*.sh; do echo "== $t"; bash "$t" | tail -1; done

# 6. Prove the dispatch changed without starting anything. `llm` does not print
#    these, so read them out of the trace the way the gate itself does; `help`
#    starts nothing:
bash -x ./llm help 2>&1 | grep -E '^\+ (SERVE_SH|PROFILE)='
#    expect SERVE_SH=.../local_llm/bin/serve-model.sh and PROFILE=flashnext

# 7. THE LIVE STEP. Stop the server, restart it through the unified launcher,
#    and diff the argv against what was serving before:
tr '\0' '\n' < /proc/$(ss -H -ltnp 'sport = :8001' \
    | grep -oP 'pid=\K[0-9]+' | head -1)/cmdline | sed '/^$/d' > /tmp/argv.before
./llm stop
MM_LIMIT_JSON='{"image":2,"video":0}' ./llm start
# wait for :8001 to answer, then:
tr '\0' '\n' < /proc/$(ss -H -ltnp 'sport = :8001' \
    | grep -oP 'pid=\K[0-9]+' | head -1)/cmdline | sed '/^$/d' > /tmp/argv.after
diff /tmp/argv.before /tmp/argv.after && echo "IDENTICAL — cutover proven"

# 8. Health, not just argv:
curl -sf http://127.0.0.1:8001/health && echo
curl -sf http://127.0.0.1:8005/__proxy/health && echo   # proxy must see upstream_ok
```

**Rollback at any point:** remove the `LAUNCHER="unified"` line from `.config`
(or restore the backup taken at the top) and `./llm stop && ./llm start`. The
legacy serve scripts are untouched until step 9.

### Step 9 — only after the above passes on a real restart

Retire, in this order, each with its own commit:
`vllm-qwen38next/serve-abliterated.sh`, `serve-tuned.sh`, `serve.sh`,
`vllm-glm53/serve-opt.sh`, `serve.sh`.

Two things must be updated in the same commit as the first deletion, or the
retirement breaks them:

* `codex-qwen.sh:115` hardcodes `FLASHNEXT_SERVE="$HOME/Projects/vllm-qwen38next/serve.sh"`.
* `tests/test_serve_model_golden.sh` compares *against* these five scripts — the
  gate cannot outlive its own baseline. Decide before deleting whether the
  golden argv gets frozen into fixture files or the gate retires with them.
  **Freezing it is the right answer**: an argv gate with no baseline is a gate
  that reports success by measuring nothing.

---

## Owner decisions

1. **`.config` (Blocker 2) — already fixed by another agent at 22:25:19**, to
   exactly the values this cleanup had read out of the live process. Nothing to
   do; the open question is whether to add a gate that fails when `.config`'s
   header, its values, and the running server disagree. Three incidents so far.
   Separately: **:8001 is down and the GPU is free as of 22:25:44.** This cleanup
   did not stop it and has deliberately not restarted it — whoever freed the card
   owns that decision. `./llm start` brings it back on the legacy path.
2. **`vllm-glm53/memtest.sh` and `moe_sweep.sh`.** Unreferenced, and either one
   run today kills the live server. Delete them, or add a
   `[ -z "$(ss -H -ltn 'sport = :8001')" ] || exit 1` guard at the top of each?
   They were kept because the audit proved them unreferenced but not dead.
3. **`bin/battery-watchdog.sh`.** Nothing runs it and nothing but one archived
   handoff document mentions it. Keep as a manual tool, or delete?
4. **`tests/test_start_concurrency.sh` is flaky.** On 2026-09-10 it failed twice
   and passed once across three runs, always at the same assertion:
   `fail-before: run/server.pid (<pid>) is alive -- expected it to name the
   LOSER, which is dead`. A concurrency gate that reports a different answer
   each run is not a gate. This predates the cleanup.
5. **`servedeck` `main` is unpushed and drifting further** — `ahead 31` of
   `origin/main` at 22:31 (it was `ahead 29` at 22:10, while this cleanup ran),
   including the entire `ui-wave` merge. Those commits exist on one disk only.
   Not a cleanup issue; a backup one. Re-measure with
   `git -C ~/Projects/servedeck status -sb` rather than trusting this number.
6. **Merged servedeck branches.** `ui-disk-size`, `ui-wave`, `migrate-coldstart`,
   `ui-request-stats`, `ui-prefill-generate` are all fully merged into `main`
   (`git cherry -v main <branch>` is empty for each). Their worktrees are gone;
   the branches were deliberately kept. Delete them with
   `git -C ~/Projects/servedeck branch -d <branch>` whenever you want — `-d`
   will refuse if the merge claim is ever wrong.
