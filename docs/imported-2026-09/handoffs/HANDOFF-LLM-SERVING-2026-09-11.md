# Handoff — local LLM serving (27B upgrade, servedeck, Codex/VS Code wiring)

**Written:** 2026-09-11 20:50 IST, at the owner's request because the session's context is nearly exhausted.
**Scope:** everything about serving models on this box (RTX PRO 6000 Blackwell, 97,887 MiB, sm_120). The AlgoTrading audit is a separate workstream (`AUDIT-CODEBASE-2026-09-09.md`), and so is the servedeck full audit (`HANDOFF-SERVEDECK-FULL-AUDIT-2026-09-11.md`, another session).
**Memory:** the durable facts are in `~/.claude/projects/-home-pctablet505-Projects/memory/`. Read `qwen27b-xid-root-cause.md`, `servedeck-is-the-control-path.md`, `serve-full-native-context.md`, `flashnext-*.md` first.

---

## 1. What is running right now (verified 20:49)

| Port | What | Managed by |
|---|---|---|
| 8004 | **Qwen3.8-27B** `RadixArk/Qwen3.8-27B-NVFP4`, served name `Qwen3.8-27B-NVFP4`, pid 80969 | **Nothing** (servedeck is stopped; see §3) |
| 8006 | Reasoning-mirroring proxy → :8004 (VS Code uses this) | `qwen27b-reasoning-proxy.service` (systemd user, enabled) |
| 8005 | Reasoning-mirroring proxy → :8001 (Flash-Next; upstream currently down) | `flashnext-reasoning-proxy.service` (systemd user, enabled) |
| 8010 | **Nothing.** servedeck was stopped on the owner's order | `servedeck.service` is `failed`/stopped but still **enabled** |
| 8000 | Unrelated `ats-optimizer.service`. **Never touch** | — |

The 27B runs on **vLLM 0.29.0** from `~/Projects/local_llm/.venv-llm-029`, with `--max-model-len 262144 --gpu-memory-utilization 0.95 --max-num-seqs 16`, **CUDA graphs ON**, and CUDA core dumps armed (see §4). KV pool 1,817,628 tokens, 6.93x full-length concurrency. The rollback `.venv-llm` (0.27.1) is untouched on disk.

`~/Projects/local_llm/.config` currently says `BACKEND="inline"`, `PORT="8004"`, `SERVED_NAME="Qwen3.8-27B-NVFP4"`, `GPU_MEM_UTIL="0.95"`, `MAX_MODEL_LEN="262144"`, `EXTRA_ARGS=""` (Flash-Next's flags are stored aside in servedeck's `state/extra_args.json` and restored on switch-back).

---

## 2. Owner decisions — do not re-litigate

1. **The 27B runs at util 0.95 on purpose** (extra KV/context). Make it safe; never lower it.
2. **Always use the latest vLLM.** The 27B is on 0.29.0, the latest release.
3. **CUDA graphs off (`--enforce-eager`) is NOT an acceptable fix, not even temporarily.** If the 27B crashes on 0.29, root-cause and fix it properly. No retreat to 0.27.1, no bisect up front.
4. **Serve every model at its full native context** (262,144 for the 27B and Flash-Next). The workload is 2-3 large-context coding agents plus many small ones; `--max-model-len` is a per-request ceiling on a shared pool, so never cap it at pool ÷ agents.
5. **Reasoning stays enabled everywhere.** Chat-completions clients (VS Code) must get `reasoning_content`, hence the proxies.
6. **servedeck is the unified control path** for starting/stopping models, not `local_llm/llm`.
7. **Don't lose the patches.** GLM-5.3 and Flash-Next run from their own editable patched trees (`~/Projects/vllm-glm53/src`, `~/Projects/vllm-qwen38next/src`); the 27B is a stock wheel. Upgrading one never touches the others.

---

## 3. Traps that bit today — read before touching anything

- **Stopping or restarting servedeck kills the model it launched.** vLLM runs inside `servedeck.service`'s cgroup and the unit's default `KillMode=control-group` kills the whole cgroup. A runtime drop-in `$XDG_RUNTIME_DIR/systemd/user/servedeck.service.d/keep-model.conf` (`KillMode=process`) is in place now, but **it vanishes at reboot**. Permanent fix still needed (put `KillMode=process` in the unit, or launch models outside the dashboard's cgroup; servedeck re-adopts a running server on start).
- **servedeck resumes whatever `~/Projects/servedeck/state/desired.json` says after every login/reboot.** It currently says RUNNING, 27B, 0.95, 262144. That is intended, but check it before any reboot.
- **Two dashboards ran at once today** (the systemd one plus one started from a VS Code terminal), both on the same state dir, which risks double restarts on a crash. Start servedeck only through systemd.
- **`.config` is parsed last-assignment-wins** and has been poisoned three times by tools writing it. Always `grep -E '^[A-Z_]+="' .config` to see what actually applies.
- **Never `pgrep -f`/`pkill -f`** — they match your own shell, and a search loop containing the text it looks for reports itself as "alive". Use `ss -lntp` and `/proc/<pid>/cmdline`.
- **vLLM renames its engine process, which blanks `/proc/<pid>/environ`** unless `SPT_NOENV=1` is set (it now is, in servedeck.toml `[backends.inline.env]`).
- **A 27B crash can leave no kernel Xid.** 0.27.1's second death was a host-side "Segfault encountered" in `CUDAGraph::replay`. Always grep the launcher's runtime log, not just `journalctl`.
- **Edit possibly-running scripts by writing a new file and `mv`-ing it**, never in place.

---

## 4. If the owner reports a 27B crash

1. Evidence: `ls ~/Projects/local_llm/logs/cudacore/` (CUDA core dumps, armed via servedeck.toml `[backends.inline.env]`: `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`, skip_* generation flags); `grep -a -n 'Segfault encountered\|EngineDeadError\|illegal memory' $(ls -t ~/Projects/local_llm/logs/qwen_server-*.log | head -1)`; `journalctl -k -b | grep -i xid | grep -vi r8169` (the r8169 line is a network card).
2. Nothing restarts the 27B while servedeck is stopped, so it stays down until someone starts it.
3. Root-cause, don't mitigate. Background: upstream vllm-project/vllm#54331 (same GPU class, same `CUDAGraph::replay` SIGSEGV, 0.26-0.28 all crash, 0.24.0 clean, only enforce_eager survived) and #52225. The race fix PR #50729 is in 0.29.0 only (verified per tag). **This box has no cuda-gdb or compute-sanitizer and no sudo**, so analysing a CUDA core dump needs cuda-gdb obtained in user space first.

**Evidence so far on 0.29:** 37 minutes of stress across two boots, 810 requests all HTTP 200, 5 saturation bursts (≥7.6 min at the 16-request cap), prompts of 122k and 204k tokens answered correctly, no Xid, no crash. 0.27.1 died at ~3.6 min and ~60 min after ready. The owner stopped the stress before 60 min, so this is strong but partial.

**Measured 0.27.1 → 0.29.0** (ctx 110592): single stream 124 → 143 tok/s; 16-way 15 tok/s (two 130 s stalls) → 1,207 tok/s; same answers.

---

## 5. In flight when this was written (background agents; results arrive as task notifications)

1. **servedeck `ctx-full-native` — DONE and MERGED into servedeck main at 20:5x (merge `af75f73`, 633 passed, 28/28 mutants).** Launches now default to the model's full native context (the pool/agents cap lived in `capacity.py` `compute()`: `ctx_max_fit = min(model_max, kv_tokens // max_num_seqs)`), a "long + short at once" table replaces "context per agent", the page reads back the running server's real max-model-len, and Decode/Prefill now say "all requests together" vs "per request". **main is 3 commits ahead of origin (not pushed).** It takes effect the next time servedeck starts. Left for a decision: an API start/restart that omits `ctx` still reuses desired.json's max_model_len; unmeasured lengths reuse the latest measured rate (Flash-Next at util 0.94 defaults to 204,800 rather than ~233k, with the reason shown). Mutation-testing note: restore mutants with `PYTHONDONTWRITEBYTECODE=1`, or a same-size, same-second restore leaves a stale .pyc that Python still trusts.
2. **`codex-qwen.sh` model discovery — DONE (installed by rename, 21:0x).** The menu now resolves the model from what is serving: servedeck `GET :8010/api/state` for the port, then that port's `/v1/models` for the id; if servedeck is down it probes :8001/:8002/:8004 (never :8000); GLM only via the :8003 effort proxy. Menu shows e.g. "1) Launch Codex with the running model: Qwen3.8-27B-NVFP4 (:8004)"; with nothing serving it offers Flash-Next / 27B (GLM only if its proxy is up) and starts the pick through servedeck. Stop/restart/status go through servedeck. `CODEX_QWEN_DRY_RUN=1 ./codex-qwen.sh qwen` prints the resolution and changes nothing. Verified: the Codex wire API (`/v1/responses`) works against the 27B on 0.29; `tests/test_codex_qwen_model_discovery.sh` 64/64 (40 of its checks fail on the old script), 14/14 mutants. The interim `MODEL=` line was removed from `.config`. Original saved at `~/Projects/local_llm/backups/2026-09-11/codex-qwen.sh.bak-*`. **Not fixed:** option 8 (set-mem) still uses the old restart path; if it ever found a server it would run `systemctl --user restart qwen-vllm` next to servedeck's server. Real start/stop through the menu has only run against stubs so far.

**servedeck instances at 21:15:** two copies again, both started from a VS Code terminal (not systemd) after the owner's kill: pid 120486 since 20:55 (holds :8010) and pid 208603 since 21:13. Probably the servedeck full-audit session. Left running. Two copies on the same `state/` dir can both react to a crash, so whoever owns them should keep it to one.

---

## 6. Open items, ranked

1. **Review the codex-qwen.sh branch** (§5.2) when its agent reports. servedeck main is 3 commits ahead of origin after the ctx-full-native merge; push when the owner says so.
2. **Permanent fix for servedeck killing its model on stop** (§3). Belongs in the servedeck full audit.
3. **servedeck renamed the 27B's public name.** It launches with `--served-model-name Qwen3.8-27B-NVFP4`; the old launcher left vLLM's default, the full repo id, which client configs were written against. Fix: pass both names (vLLM accepts several). VS Code was switched to the short name as a workaround.
4. **Launcher tests: 3 of 8 fail, identically before and after the 0.29 switch.** `test_start_concurrency.sh` and `test_unified_cutover_safety.sh` stop at `llm` line ~435 `sudo -v` (Flash-Next PLE path, not stubbed in the sandbox); `test_llm_config_warn.sh` fails because `llm` has no `inline` backend. Fix the tests (and decide whether `llm` should support the 27B at all, given servedeck is the control path).
5. **`bin/soak-27b-overnight.sh` never re-sends reasoning** (reads `reasoning_content`, which :8004 does not return) and resets conversations by character count. **`bin/qwen-server-run.sh`** still has a comment saying the race is "unfixed as of vLLM 0.29.0" (false; #50729 is in 0.29.0).
6. **servedeck sweep leftovers F10-F14**: nothing measured is written back (27B uncalibrated), history records duplicated/misattributed (why the boot ETA is always blank), F12-F14 minor. Also a boot-log-rotation edge case can drop the READY latch.
7. **Consider more parallel small agents**: the 27B pool fits far more than 16 small requests alongside 2-3 full-length ones; `max_num_seqs` could rise once the owner wants it (re-check stability at saturation if so).
8. **GLM-5.3 and Flash-Next are on 0.29.0.dev0 forks.** The owner prefers latest versions; rebasing those patched trees onto current vLLM is a separate, larger job (GLM's rebase was previously estimated at 5 conflicts plus a new sync checker).

---

## 7. Upstream PRs and issues (owner's account pctablet505)

| Item | State |
|---|---|
| vllm-project/vllm #54971 (GLM-4.7 arg-tag leak) | Open, **approved** by an independent reviewer, blocked only by the first-contributor throttle |
| #54972 (UVA offloader empty_cache) | Open, throttle-blocked; dedup statement added |
| #56093 (Poolside arg-tag leak) | Open, throttle-blocked; model-evaluation note added |
| #56129 (reasoning round-trip tests) | Open, conflict resolved by merge (no force-push), mergeable |
| #56095 (NVFP4 w13 scale) | **Closed** as a duplicate of #55073; our tests offered as xininininin/vllm#1 |
| flashinfer-ai/flashinfer #5064 (cu13 `-lcudart`) | Claimed by a maintainer |

The throttle (`pre-run-check`) needs a maintainer-applied `ready`/`verified` label or 4 merged PRs; its message forbids an AI agent from requesting the label. Pre-commit is *skipped*, not failing; locally all five branches pass 31/31 hooks. A real upstream flake was root-caused (shared `MockHFConfig` class attribute in `test_serving_chat.py`); a one-line fix could be its own PR.

---

## 8. Key paths

- 27B launcher: `~/Projects/local_llm/bin/qwen-server-run.sh` (default venv now `.venv-llm-029`); unified launcher `bin/serve-model.sh` + `profiles/*.env` (behind `LAUNCHER=unified`, off).
- servedeck: `~/Projects/servedeck` (main), config `servedeck.toml` (gitignored; inline venv `.venv-llm-029`, `[backends.inline.env]` holds the CUDA_COREDUMP* entries and `SPT_NOENV`), state `state/`.
- Runtime logs: `~/Projects/local_llm/logs/qwen_server-<stamp>.log`; death log `qwen_deaths.log`.
- VS Code models: `~/.config/Code/User/chatLanguageModels.json` (27B `Qwen3.8-27B-NVFP4` → :8006, maxInput 226,144 + output 36,000 = 262,144).
- Proxy code: `~/Projects/flashnext-reasoning-proxy/` (one codebase, two systemd instances).
- Session scratchpad (RAM, lost on reboot): `/tmp/claude-1000/-home-pctablet505-Projects/5b534f7a-126f-4b00-9811-0dfe96e5eacc/scratchpad/` — upgrade backups under `upgrade27b/backups/`, config backups at the top level. Post-reboot backups (VS Code model config x2, and the upgrade set: `.config`, servedeck.toml, desired.json, extra_args.json, qwen-server-run.sh, qwen27b.env, codex-qwen.sh, all as of 19:44) were copied to real disk at `~/Projects/local_llm/backups/2026-09-11/`. Everything in the scratchpad from BEFORE the 18:08 reboot is gone, including the earlier `.config` backups some memory notes mention.
