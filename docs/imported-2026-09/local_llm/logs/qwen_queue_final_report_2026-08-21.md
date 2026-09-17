# Qwen local-agent work queue — final consolidated report (2026-08-21)

All six charter tasks (Q1–Q6) are complete, independently verified, and left as unmerged
branches. Nothing was pushed or merged; `main` is still `e93b0032`.

## Queue status

| Task | Branch | Commit | Status | Verified by |
|---|---|---|---|---|
| Q1 system_info GPU facts | `qwen/q1` | `2ea7ec60` | done | VERIFY agent A + orchestrator |
| Q2 branch/worktree prune | (git ops, no commit) | — | done | VERIFY agent A |
| Q3 max_holding_period default | `qwen/q3` | `64e57dcd` | done | VERIFY agent B |
| Q4 ADR renumber + land | `qwen/q4` | `c3aea6ad` | done | VERIFY agent B |
| Q5 stale doc citations | `qwen/q5-doc-citations` | `73bd4427` | done | orchestrator (direct) |
| Q6 unevaluable-check hunt | `qwen/q6` | `cf7470aa` | done + landed | orchestrator spot-checks + independent review session |

## Q1 — `qwen/q1` @ 2ea7ec60

Checkpoint `system_info` payload now records `gpu_name`, `gpu_vram_total_mb`,
`gpu_vram_peak_allocated_mb`, `gpu_utilization_pct` — always present, None-safe on CPU.
Files: `algotrading/src/utils/device.py` (+12, `total_gpu_vram_mb()` wrapper),
`algotrading/src/utils/system_info.py` (writer), `tests/utils/test_system_info_gpu.py` (new).

- Acceptance: CPU-pinned 6 passed / 2 skipped / 0 failed; `JAX_PLATFORMS=cuda` 8/8 passed
  (verified by VERIFY-A in the q1 worktree, import check inside the worktree).
- Fail-before (reproduced by VERIFY-A in a scratch worktree on main + copied test file):
  **6 failed / 0 passed / 2 skipped** — KeyError on each new key. Note: the original handoff
  said 5F/1P; the "1 passed" test (`test_payload_stays_json_serializable`) actually also fails
  on main (KeyError after the JSON round-trip). True fail-before is 6F; the fail-before
  property holds either way.
- Subtree: `tests/utils` → 0 failed / 2305 passed / 2 skipped / 1 xfailed (2308 collected).
  The handoff's "2758 collected" subtree invocation is not reconstructible from the record;
  closest repro: `tests/utils tests/golden tests/validation tests/quality tests/system` =
  2759 collected (±1). The failure-set-identity property holds for `tests/utils` (the subtree
  containing all changed modules): scratch-main 6F/2299P → q1 0F/2305P, same 2308 collected.
- Goldens: untouched (diff is exactly the 3 files; no baseline/fixture paths).
- UNCERTAIN: behaviour-affecting? No — payload-only, no learning or metric semantics change.

## Q2 — branch/worktree prune (no commits)

- 31 worktrees + 29 branches deleted; audit log `/tmp/q2_audit.log` (141 lines).
- VERIFY-A: every one of the 29 branch deletions is preceded by a recorded successful
  `git merge-base --is-ancestor <branch> main` and a clean `git status --porcelain`; all 31
  worktree removals preceded by a clean-status check; the single `-D`
  (`data-gate-range-2026-08-21`) is flanked by two recorded `git branch -d` failures, a
  re-verification, and a justification; its stated upstream ref is confirmed an ancestor of
  main. No missed deletions: all 27 non-main branches scanned; every merged-into-main branch
  is checked out in a retained worktree or protected.
- End state at Q2's last log write (18:35:04): 17 worktrees / 24 branches (reconstructs to
  the second). Today: 20 worktrees / 27 branches = that + `qwen/q3` + `qwen/q4` + `qwen/q6`.
- `main` tip: `e93b0032` — unchanged. All 4 forbidden worktrees + `AlgoTrading-llm-audit`
  intact.

## Q3 — `qwen/q3` @ 64e57dcd

`build_training_run_config()`: `max_holding_period` default is now `max_episode_steps + 1`
(was `NSE_MINUTES_PER_DAY` = 375 at every granularity), values below 1 raise ValueError.
Salvaged test `tests/cli/test_max_holding_period_derivation.py` landed byte-identical.

- Acceptance: 7 passed / 0 failed. Fail-before (scratch main + copied test): 6 failed / 1
  passed — both reproduced by VERIFY-B.
- Subtree `tests/cli tests/architecture`: 3 failed / 3244 passed (6 skipped, 4 xfailed). The
  3 failures are exactly the pre-existing orchestrator size-limit tests (same 3 fail on
  main; see "Pre-existing findings").
- `tests/golden`: 311 passed. Size markers match `wc -l` (file 866, function 305).
- Diff scope: exactly the 2 claimed files. UNCERTAIN: behaviour-affecting default — flagged
  per charter; change limited to the derivation + validation.

## Q4 — `qwen/q4` @ c3aea6ad

Salvage ADR landed as **ADR-0031** (`docs/adr/ADR-0031-retire-subdaily-rl.md`); 0031 was the
next free number (README allocation rule + hub sequence both checked); README index entry
added, next-free moved to 0032.

- Wording diff vs the salvage file: exactly 3 changed lines — title number, Sequence
  self-reference, one `<!-- dangling-ok: ... -->` on the out-of-repo `tools/week_kill_case.md`
  provenance line. Nothing else (VERIFY-B recorded the exact diff).
- No other ADR number moved (ls-tree main vs HEAD differ only by the added file).
- `tests/docs`: green except the known pre-existing DECISION-BRIEF→GATE_THRESHOLDS dangling
  (red on main pre-Q5; fixed on the Q5 branch). `python scripts/drift_check.py` → exit 0
  clean (VERIFY-B mutation-checked the gate: a planted violation makes it exit 1).
- Note: no dedicated "ADR index test" exists in the repo; acceptance is covered by the
  cross-reference gate + manual index consistency check (recorded by VERIFY-B).

## Q5 — `qwen/q5-doc-citations` @ 73bd4427

Branch name deviates from the charter's `qwen/q5` because the worktree
(`AlgoTrading-qwen-q5`) was pre-created clean/unclaimed at main and reused; flagged in the
Q5 evidence pack.

- Fixed 4 stale present-tense citations: `docs/bedrock/ARCHITECTURE.md:195` +
  `docs/technical/market_indices_pipeline.md:5` (index shims 2-line → 3-line; verified
  `wc -l` = 3 each, and the shims were exactly 2 lines at shim commit `1bd4ac6a`, so the
  dated log entries saying "2-line" were correct as written and were NOT touched);
  `docs/BACKLOG.md:210` S1-F03 (audit package 7,458→7,574 = sum `wc -l` over its 11 files;
  `profile_store.py` 1,616→1,868 — both re-measured by the orchestrator).
- `<!-- dangling-ok: ... -->` marker added on the citing line in
  `docs/DECISION-BRIEF-2026-08-21-fable-one-month-plan.md` for
  `docs/standards/GATE_THRESHOLDS.md` (deliberately not written yet; the marker
  self-cleans via `test_allowlist_has_no_stale_entries` when the file appears).
- Fail-before: 1 failed / 274 passed (the dangling test). Fix-after: **0 failed / 275
  passed** (re-run by the orchestrator on the committed tree). Diff = exactly 4 lines, no
  historical figure altered.
- UNCERTAIN (from the agent, reviewed and agreed by the orchestrator): BACKLOG S1-F03
  treated as present-tense (undated, describes current code); the C8 plan's 720/712 LOC
  treated as historical (pinned to base commit 32c3e423, where the numbers verify exactly);
  `workstreams/_findings` registers contain many now-stale sizes, left untouched as history
  per the charter rule; `.md`-file-size citations out of the `.py` scope of the charter.

## Q6 — `qwen/q6`, no commit (report-only)

Hunt for "check that cannot evaluate". Report:
`/home/pctablet505/Projects/AlgoTrading-qwen-q6/Q6-UNEVALUABLE-CHECKS-REPORT.md`
(26 KB, untracked; method: AST scan of 290 config-like getattr/hasattr reads vs
`TrainingRunConfig`'s 188 fields + call-site types; 107 `_should_*/_check_/_validate_*`
call-graph greps; 135-counter read/write traces; never-true-flag traces; 30-site
permissive-`except` sweep; complete dropped-candidate table with evidence).

**2 new findings** (spot-checked by the orchestrator with independent greps):
1. `algotrading/src/models/training/nifty_pipeline.py:312` —
   `getattr(cfg, "prefer_service_data", False)`: phantom field (single occurrence
   tree-wide, no CLI flag, `from_dict` strips unknown keys) → the `TrainingDataService`
   service-facade path has no reachable production call site. Previously owner-measured
   2026-08-05 (Q&A Q9, "Dormant") but never registered — register gap closed by the draft.
2. `algotrading/src/models/jax/training/orchestrator.py:421` — the portfolio-net eval actor
   gets `"n_aux": getattr(cfg, "n_aux_signals", 0)`; the field is on `TradingEnvConfig`, not
   `TrainingRunConfig`, and is never assigned on a config, so the eval always reads 0 while
   the encoder/critic were built with the real aux width (`:322/:339` fall back to
   `train_aux.shape[1]`; walk-forward mirror reads the real value at
   `multi_asset_walk_forward.py:138`). Trigger: `use_portfolio_net: true` (config-file only,
   no CLI flag) + `--sub-models` (the live aux route since ADR-0028). Runtime mode (crash vs
   silent mis-slice) not executed — no training runs permitted to the hunting agent.

## FLAW_REGISTER entries — landed on `qwen/q6` @ cf7470aa

The draft was reviewed by a second pass (record: `docs/audits/Q6_HUNT_REVIEW_HANDOFF_2026-08-21.md`,
now committed) which corrected three meta-claims of mine — the over-scoped "every line
re-verified" claim (I had re-run the re-detect commands, not every line number), a wrong
path, and, most substantively, the zero-call-site count: the Q6 report claimed 1 uncalled
`_should_*/_check_*/_validate_*` def, but there are **4** in `algotrading/src/`. The review
session applied the corrected register text; I then independently re-verified the new
claim (all 4 zero-call sites confirmed by my own greps, including the repo-wide call-site
search) and triaged the 3 previously-missed predicates:

- **FR-074 (S3) added:** `AdaptiveRegimeGater` is live (catalog-registered,
  `strategy_catalog.py:310`; auto-sim path `auto_simulation_logic.py:396`), but its
  entry/exit decision methods (`_should_entry_buy` :144, `_should_entry_short` :166,
  `_should_exit` :188) are never called — `generate_signal` (:206) decides inline with
  SMA-crossover + ATR stops, and the test files exercise the dead predicates directly.
- Register state: §2 shape bullet + §3 pass-04 subsection (recorded PARTIAL at pass time,
  completed in review; all 4 zero-call sites accounted for) + §4 coverage row + FR-072/073/074
  rows. Hunt report committed at `docs/audits/Q6_UNEVALUABLE_CHECK_HUNT_2026-08-21.md` with an
  append-only review addendum; superseded draft deleted.
- Two mechanical defects in the applied register text were fixed before committing: the §4
  row sat outside its table (after the `---` separator), and a blank line split the table.
- First commit attempt was blocked by the automated-review hook (detect-secrets false
  positive: high-entropy absolute path inside a proof block). Resolved with the hook's own
  `# pragma: allowlist secret` marker on the two affected lines; `.secrets.baseline` untouched.
- All hooks passed at commit time (incl. doc-drift gate and Contract ADR check); author
  pctablet505; worktree clean.

## Ratchets (final, official method)

`python scripts/quality/ratchet.py --check` (the pre-push gate's own counter, which excludes
`models/jax` from PLR2004/G004 and honours frozen sites/noqa) → **OK on main, q1, q3, q4**
("all metrics at or below baseline"). Q5 is docs-only; Q6 made no changes. The earlier
apparent PLR2004 (530 vs baseline 479) / G004 (1 vs 0) discrepancies were an artifact of a
naive `ruff check algotrading/src | wc -l` that did not apply the official exclusions —
**no ratchet moved on any task branch**, and no pre-existing violation exists.

## Pre-existing findings (untouched, per charter)

- 3 red architecture tests on main: `orchestrator.py` file 1311 > declared 1294,
  `JAXOrchestrator.fit` 376 > 359 (`test_file_size_limits`,
  `TestFunctionLength::test_no_function_exceeds_max_lines`, `test_function_size_limits`).
  The in-file `# size-ok` markers are stale; **raising or shrinking is an owner judgement
  call** — not done by the agents.
- Test-quality baseline stale on main (`scripts/check_test_quality_baseline.py`: unmarked
  406 > 387, assertion-free 280 > 279, shape-only 2816 > 2744). Re-baselining is a hard
  prohibition (§1.6) and a human-only decision.
- `tests/docs/test_doc_cross_references_resolve.py::test_no_unallowlisted_dangling_references`
  was red on main (DECISION-BRIEF → GATE_THRESHOLDS); fixed on `qwen/q5-doc-citations`,
  still red on main until that branch merges.
- `docs/improvement_plan/workstreams/*` and `_findings/*` contain many now-stale size
  figures, left as history per the "never rewrite the old number" rule (Q5 UNCERTAIN).

## Infrastructure

- vLLM `RadixArk/Qwen3.8-27B-NVFP4` now persistently configured at
  `--gpu-memory-utilization 0.89` (≈85 GiB of the 97.9 GiB card; `local_llm/.config`).
  Server process is owner-managed; agents were instructed to never touch it.
- Concurrency capacity at 0.89: one full 262K context ≈ 8.5 GiB KV (fitted from the
  script's measured table); KV pool ≈ 57 GiB → theoretical ceiling ~6–7 concurrent
  full-context sessions. 4 agents ran in parallel comfortably; 5 is fine; 6–7 is the
  OOM/preemption wall.
- Parallel runs used 4 agents: Q5, Q6, VERIFY-A (Q1+Q2), VERIFY-B (Q3+Q4), via
  `codex-qwen.sh qwen exec -C <worktree> --dangerously-bypass-approvals-and-sandbox -o
  /tmp/<id>_last_message.txt --json`; all four completed with exit 0.

## Artifacts

- Agent prompts: `/tmp/{q5,q6,verifya,verifyb}_prompt.md`
- Event streams: `/tmp/{q5,q6,verifya,verifyb}_events.jsonl`
- Final agent messages: `/tmp/{q5,q6,verifya,verifyb}_last_message.txt`
- Q2 audit log: `/tmp/q2_audit.log`
- Post-merge prune audit: `/tmp/qwen_postmerge_prune_2026-08-21.log`
- FR fix prompts/reports: `/tmp/{fr072,fr073,fr074}_prompt.md`, `*_last_message.txt`, `*_events.jsonl`
- Q6 report: `/home/pctablet505/Projects/AlgoTrading-qwen-q6/Q6-UNEVALUABLE-CHECKS-REPORT.md`
- FLAW_REGISTER draft: `/home/pctablet505/Projects/AlgoTrading-qwen-q6/Q6-FLAW_REGISTER-DRAFT.md`
- Set-mem log: `/tmp/setmem.log`

## Post-merge (20:14-21:05, 2026-08-21)

The owner merged all five `qwen/*` branches to `main` at 20:14 (merge commits
`d01d9427`, `dbdd53de`, `a21516d8`, `912b4f0f`, `faba3280`) and followed with
`ccc80c8a` (chore: refresh detect-secrets baseline line numbers).

**Mechanical post-merge prune (orchestrator, 20:22):** all five merged `qwen/*` worktrees +
branches removed, plus the owner's merge-scratch worktree under `/tmp/claude-1000/...`.
Every deletion preceded by a recorded `git merge-base --is-ancestor <branch> main` + clean
`git status --porcelain` (audit: `/tmp/qwen_postmerge_prune_2026-08-21.log`). End state:
16 worktrees / 22 branches.

**FR fix run (3 parallel local-Qwen agents + VERIFY agent):** the three open hunt-pass-04
rows were commissioned as checkable fix tasks, one worktree/branch each off `ccc80c8a`:

- **FR-072** `qwen/fr072` @ `1133a605` + `d2ab9b80`: removed the 8-line dead
  `prefer_service_data` gate from `load_and_split` (nifty_pipeline.py; marker re-declared
  357→349; stale narration reworded). `prefer_service_data` now 0 hits in src (was 1).
  `tests/models/training`+services+architecture failure set identical to main (3 pre-existing
  orchestrator size failures). Two commits (fix + register close, repo precedent).
- **FR-073** `qwen/fr073` @ `d5079bcf`: `orchestrator.py:421` now reads
  `algo_config.get("n_aux", 0)` — the same source the walk-forward mirror uses; the dict
  instance identity was traced and proven (comment at the fix site). Train sites :322/:339
  untouched (single diff hunk). New CPU-safe test `test_eval_actor_aux_width.py`: red on main
  (`assert 0 == 3`), green on branch. Subtree 406P → 408P (delta = exactly the 2 new tests).
  Bonus flag (out of scope, not fixed): `_pa_cfg["n_assets"]` reads the requested
  `getattr(cfg, "n_assets", 20)` where the resolved env count is what the mirror uses —
  same defect class, separate site.
- **FR-074** `qwen/fr074` @ `aa0353b5` + `0f0c45a4`: the three dead `AdaptiveRegimeGater`
  predicates deleted (62 lines; `generate_signal` untouched — single hunk). New
  characterization test (`test_adaptive_regime_gater_live_path.py`, 8 exact-signal tests)
  green on unmodified main AND after deletion (byte-identical file both runs = behavior
  invariance). Dead-predicate tests removed; one re-pointed at the live forced-exit branch.
  `tests/models/strategies` 379P/5S both sides. `_get_trend_direction` left: pre-existing
  orphan, not created by this deletion (flagged as separate finding). **Note:** the official
  ratchet's designed down-only auto-tighten updated `tools/ratchet/quality_baseline.json`
  PLR2004 479→473 with this commit (the deletion removed magic numbers); disclosed here per
  §1.6 sensitivity — revertible on the branch if the owner prefers the old ceiling.

All three: register rows moved §3→§5 with fix sha + evidence; ratchets OK via
`scripts/quality/ratchet.py --check`; no goldens/secrets touched; no push/merge.

## Owner decisions outstanding

1. Merge/push of `qwen/fr072` (`1133a605` + `d2ab9b80`), `qwen/fr073` (`d5079bcf` + `90e5dc16` register-sha cell), `qwen/fr074` (`aa0353b5` + `0f0c45a4`) (human-only; all three VERIFY-confirmed, see verdicts above).
2. Keep or revert the fr074 auto-tightened PLR2004 baseline (479→473, down-only; VERIFY reproduced the gate's auto-tighten mechanism — reverting is cosmetic since the next `ratchet --check` re-tightens it).
3. `orchestrator.py` size markers: raise budgets or shrink the function (judgement).
4. FR-073 runtime mode: a `use_portfolio_net: true` + `--sub-models` smoke run would settle
   crash-vs-mis-slice pre-fix runs (needs the card; not a Qwen task). The config wiring is
   now test-proven either way.
5. Test-quality baseline re-baseline (human-only, §1.6; still red on main).
6. New candidate rows for a future hunt pass: `_pa_cfg["n_assets"]` requested-vs-resolved
   (fr073 bonus flag), `_get_trend_direction` pre-existing orphan (fr074 note), and the
   tracked `docs/generated/CODEBASE_MAP.md` regeneration path problem (fr074 note).
7. Push `main` to origin (moved ~11 commits ahead of the remote).

## VERIFY-FR verdicts (independent agent, 20:47-21:06, 2026-08-21)

A VERIFY agent that did none of the fix work re-established every number:
tree state, import resolution, branch acceptance re-runs, fail-before
reproduction in a scratch worktree on main (`verify/fr-scratch`, removed on
completion), subtree failure-set identity vs main, goldens/baseline scan, and
register-closure format. Full verdict: `/tmp/verifyfr_last_message.txt`.

- **FR-072: CONFIRMED.** Fail-before main=1 hit (`nifty_pipeline.py:312`) vs
  branch=0; `prefer_service_data` confirmed absent from all 188
  `TrainingRunConfig` fields; marker 357→349 and file 800→792 confirmed;
  affected-set re-runs identical on both sides (3F/1519P/1S/4XF — the same 3
  pre-existing orchestrator size-gate failures); ratchet exit 0 at exact
  baseline; goldens untouched; register closure exactly 1 line replaced.
  Note: VERIFY's first branch-side run showed 2 extra failures in
  `tests/data/services/test_training_data_service.py` under two concurrent
  `-n auto` sessions; they did not reproduce on re-run (both sides identical)
  and the file passes 5/5 in isolation on the branch (orchestrator re-run,
  21:08). No root cause established — recorded, no impact on the verdict.
- **FR-073: CONFIRMED.** Copied test red on main (`assert 0 == 3`,
  file md5-identical), green on branch; subtree counts reconcile exactly
  (406 base + 2 new; failure-set difference = the fix's own new test);
  train sites `:322`/`:339` byte-unchanged; ratchet OK on both sides;
  goldens untouched. One documentation-level gap found: the "Closed by" cell
  carried a branch reference instead of the closing sha (the single-commit
  closure could not self-reference). Fixed post-verification by orchestrator
  commit `90e5dc16` on `qwen/fr073` (one cell → `d5079bcf`, doc-drift gate
  passed). Branch is now 2 commits: `d5079bcf` + `90e5dc16`.
- **FR-074: DISCREPANT on one check only, mechanism verified.**
  Invariance proven: characterization 8 passed on main (file copied in,
  pre-deletion) AND 8 passed on branch (post-deletion), test file
  md5-identical across runs; 3 def lines on main → 0 hits on branch;
  subtree reconciles (-7 dead-predicate tests, -1 re-pointed, +8 new = net 0;
  skip sets identical); register closure clean. The discrepancy:
  `tools/ratchet/quality_baseline.json` (PLR2004 479→473) traveled inside
  commit `aa0353b5`, outside the task's allowed file set. VERIFY reproduced
  the mechanism against main's baseline copy: the official gate's down-only
  auto-tighten writes+stages it (`--check` printed "PLR2004: 479 -> 473",
  exit 0, only that file would change). Not a hand re-baseline; the count
  genuinely dropped with the 62 deleted lines. Accept/reject left to owner
  (note: any future `ratchet --check` on this tree re-tightens to 473
  anyway, so reverting the file is cosmetic).
