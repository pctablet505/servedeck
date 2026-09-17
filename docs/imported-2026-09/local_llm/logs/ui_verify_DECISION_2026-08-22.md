# Decision — UI verification pass, 2026-08-21/22

**Decides:** `/home/pctablet505/Projects/local_llm/logs/ui_verify_2026-08-21.md`
**Branch:** `ui-deep-review-2026-08-21` @ `c62c5888`, 6 ahead / 0 behind `main` (`da01ce14`).
**Worktree:** `/home/pctablet505/Projects/AlgoTrading-ui-verify`
**Decided by:** Claude (orchestrator), 2026-08-22. **Not pushed. Not merged.**

---

## Verdict in one line

**Merge the branch. Four findings stay open, and only two of them are genuinely the owner's call —
the other two I am deciding here.**

---

## 1. The branch: MERGE

Verified independently, not taken from the log:

| check | result |
|---|---|
| Import resolves to the worktree under test | `/AlgoTrading-ui-verify/algotrading` ✓ |
| `tests/pyqt_ui -n auto` | **849 passed**, 0 failed, no segfault, 25.77 s |
| Ratchet baselines / `.secrets.baseline` / goldens touched | **0 files** |
| Position vs `main` | 6 ahead, **0 behind** — clean fast-forward |
| Fail-before proof per fix | present for all five, each with its own test file |

**The segfault question is settled, and it was the one that mattered.** Task #1 records that PR #42's
UI portion segfaults xdist workers, cause unidentified, do-not-merge. That is a standing reason to
distrust *any* PyQt change. This branch runs the full `tests/pyqt_ui` suite under `-n auto` — the
exact configuration that crashes on PR #42 — and completes clean at 849 passed. It is not the same
defect, and it does not reintroduce it. **This does not resolve task #1**; PR #42's UI delta stays
unmerged on `pr42-ui-2026-08-21`.

What lands:

| commit | finding | why it is a real defect |
|---|---|---|
| `76b5071b` | #1 (panel half) | 7 of 9 panel-indicator checkboxes silently rendered nothing — the viewer looked up column names `calculate_all_indicators` never writes, and CCI/WILLR/ROC/OBV were not computed at all. A control that appears to work and produces nothing is the worst class of UI defect |
| `d12e535a` | #3 | System Information dialog had no Close button and shipped placeholder text ("PyQt6: 6.x", "GPU Support: Check logs for details") |
| `b01ac6f9` | #7 | Both slippage spin boxes never set `singleStep`, so Qt's default of 1.0 made one arrow press jump 0.01 → 1.000 on a 0–1 % field. The arrows could only ever reach the default or the maximum |
| `f6adaba7` | #6 | Six search boxes fed raw user text to `str.contains` with `regex=True`. A single `(` raised `re.PatternError` inside a debounced slot: traceback to stderr, table frozen on the previous query, no user feedback |
| `4ec2aee5` | #8 | Position-sizing "Available Capital" defaulted to ₹100,000 while session capital defaulted to ₹1,000,000, syncing only after a session start — so pre-start sizing computed against **one-tenth** of the intended capital |
| `c62c5888` | — | no-op statement removed |

`f6adaba7` and `4ec2aee5` are the two worth caring about beyond tidiness: one is a crash on ordinary
input, the other silently sizes positions against the wrong capital.

**Do this:**

```bash
cd /home/pctablet505/Projects/AlgoTrading
git worktree add --detach /tmp/merge-ui main && cd /tmp/merge-ui
git merge --no-ff --no-verify -m "Merge ui-deep-review-2026-08-21" ui-deep-review-2026-08-21
export PATH="/home/pctablet505/Projects/AlgoTrading/.venv/bin:$PATH"
python -m pytest tests/pyqt_ui -n auto -p no:randomly --no-cov -q
git branch -f main HEAD && git push origin main    # never --no-verify on the push
git rev-parse main; git ls-remote origin main      # must match
```

Expect the push to fail once on a detect-secrets baseline refresh; commit
`.secrets.baseline` and push again. That has happened on every push today.

---

## 2. Finding #2 — CLOSED, and the close is correct

The log re-audited its own finding and withdrew it. I agree, and the reasoning is worth preserving:
`_resolve_override` writes the configured value **verbatim** when the field is untouched, so the
combo's nearest-preset display never corrupts the stored number, and `_remember_loaded` logs the
drift on every load. The real cause of the warning is that the workdir `.env` carries a stale
`0.001` while `TX_COST_DELIVERY` is `0.0011`.

**No action.** Withdrawing a finding on re-reading the code is the behaviour you want, not a
retraction to be embarrassed about.

---

## 3. Finding #4 — I am deciding this one: FIX IT, no ruling needed

A fresh clone has no `.env`, the app boots degraded with only a console banner, and `.env.example`
is shipped but never referenced in the `DEVELOPER_GUIDE.md` setup section.

This does not need the owner. It is a documentation gap with an obvious correct answer, and the log
itself records that it **cost the verification run a full re-run** (22:35–22:50) — the boot-environment
trap in its own notes. That is a measured cost, not a hypothetical.

**Do:** add the `.env.example` → `.env` step to the `DEVELOPER_GUIDE.md` setup section, and make the
degraded-boot banner name the file to create. Small, and it belongs to whoever next touches the docs.
**Not** a blocker for the merge.

---

## 4. Findings #1 (remainder) and #9 — these two are genuinely the owner's

Both are the same species: **the UI is faithfully displaying what the engine computes, and the
question is whether the engine's number means what its label says.** No UI change can settle either.
Neither is a bug in the branch above, and neither blocks the merge.

### #1 remainder — metric semantics on the stock viewer

Three separate labels, each arguably lying:

- **"Volatility (20d)"** applies the *daily* annualisation factor and `tail(20)` at every
  granularity. Same symbol: **22.67 % on day bars vs 1.27 % on 5-minute bars.** The 5-minute figure
  is not an annualised volatility of anything.
- **"52W High/Low"** silently falls back to period high/low when fewer than 252 rows are loaded — so
  it reports a 52-week extreme computed from whatever happens to be in the window.
- **"YTD Return"** is relative to the last *loaded* year, not the calendar year.

Each has a defensible fix and they are not the same fix. The pattern is this project's signature —
a number that reports a value it did not measure — but here the honest repair is a labelling and
semantics decision, and getting it wrong replaces one wrong number with a different wrong number.
**A patch is sketched; do not apply it without a ruling.**

### #9 — walk-forward aggregation over non-measurable windows

`mean OOS Sharpe = -inf, p-value = nan` on RELIANCE day with 5 splits. Mechanism fully traced: a
no-trade OOS window has zero return variance, `_limit_ratio`
(`models/analysis/backtesting.py:232`) returns ±inf **by explicit design** for zero denominators —
"non-measurements that fail closed" — and the numerator is negative, so the window is −inf.

The per-window rows are honest. What contaminates is the **aggregate**, which averages measured and
non-measured windows together and lets one non-measurement swallow the result. The options —
exclude them, report "n of m measured", or keep −inf as a deliberate fail-closed signal — are
research semantics with different meanings for every walk-forward number the project has produced.
**Reserved to the owner.** This one should go to Fable with the one-month plan; it is adjacent to
D1 (what number, against what benchmark) in the decision brief.

---

## 5. Housekeeping

- **`/home/pctablet505/Projects/AlgoTrading-ui-verify`** — the log says "remove when done". Keep it
  until the merge lands and the two open findings are ruled on; it is the only tree with the driver
  wired up.
- **`/tmp/uitour/`** — driver, `steps.jsonl`, `chart_probe.json`. **This is on `/tmp` and will not
  survive a reboot.** The driver is a genuinely reusable asset — it drives 21 user journeys
  end-to-end and caught five real defects. If it is worth keeping, move it into the repo
  (`scripts/uitour/`) or at least out of `/tmp`, before the machine restarts.
- The log's own **import-shadow warning** is worth promoting beyond this file: the first driver run
  silently tested the canonical dead-lineage checkout because a script in `/tmp` put its own
  directory on `sys.path` ahead of the worktree. Same hazard the `agy` guide records, third tool it
  has bitten. The `IMPORT_CHECK` record the driver now emits is the right shape of fix.

---

## 6. What I did not verify

Stated so nothing here is read as more checked than it is:

- I ran `tests/pyqt_ui` and the ratchet/golden check. I did **not** re-run the 21-journey driver, and
  I did not re-derive the individual findings' repro steps — those rest on the log.
- The WebEngine render probe (505 SVG paths, `window.Plotly` present) is quoted from the log,
  unverified by me.
- Findings #1 and #9 I read as *mechanism descriptions* and judged them owner-level on that basis; I
  did not independently confirm the 22.67 % / 1.27 % measurement or the `_limit_ratio` trace.
