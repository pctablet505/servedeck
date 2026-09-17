# Handoff — Servedeck UI wave + two audit passes

**Date:** 2026-09-10
**Repo / worktree:** `/home/pctablet505/Projects/servedeck-integrate`
**Branch:** `ui-wave` (24 commits ahead of `main`, 0 behind, **never pushed** — no `origin/ui-wave`)
**Task for the reviewer:** review the uncommitted working tree, then commit and merge.

---

## 0. Read this first — the state is UNCOMMITTED

`git status` shows 10 modified files and nothing staged. **HEAD does not contain
any of this work.** `git show HEAD:servedeck/web/app.js` is 45 kB; the working
file is 78 kB. So:

- **Do NOT `git checkout` / `git restore` any of these files to "get a clean
  baseline" — it destroys the work irrecoverably.** There is no stash, and the
  pre-fix state is not in git at all.
- Baselines for the fixes were reconstructed by string surgery into `/tmp`
  (`/tmp/app.js.fixed`, `/tmp/prefix_fix2.py`, `/tmp/prefix_fix3.py`).
- `servedeck.log` (314 kB) is untracked and **not** gitignored — it is runtime
  output from restarting the dev server. Do not commit it; consider adding it to
  `.gitignore` as part of this merge.

```
 M servedeck/app.py            |  62 ++-      (mostly _snap() consolidation)
 M servedeck/capacity.py       |  12 +-      (MiB grouping in two findings)
 M servedeck/reqstats.py       | 173 ++++-   (fine-bin histogram: earlier turn)
 M servedeck/supervisor.py     |  15 +       (earlier turn)
 M servedeck/web/app.js        | 869 +++++++---
 M servedeck/web/index.html    | 261 +--
 M servedeck/web/style.css     | 147 ++-
 M tests/test_capacity.py      |   4 +-      (grouped-number assertions)
 M tests/test_reqstats.py      | 171 +++-    (earlier turn)
 M tests/test_ui.py            | 1006 +++++++---
 10 files changed, 2347 insertions(+), 373 deletions(-)
```

---

## 1. What this work is

Three stacked passes over the dashboard (`http://127.0.0.1:8010/`), a local
vLLM control panel. Frontend is hand-written `index.html` / `style.css` /
`app.js` — no framework, no build step, served `Cache-Control: no-store`.

1. **UI wave (P0–P7)** — the owner's "badly organised / confusing" complaint,
   itemised as audit flaws A–H and N1–N28. Dead UI deleted, unwired markup
   wired, duplicated figures collapsed, every absent figure given a reason.
2. **Histogram granularity + refactor** — `reqstats` emits fine bins;
   `app.js`/`app.py` de-duplicated for LOC.
3. **Second audit pass (this turn)** — 8 more flaws, found by rendering the page
   and reading **computed styles** rather than reading source. This is where the
   headline defect lives.

---

## 2. The headline defect — invisible text from a CSS class collision

The serving line (`http://localhost:8001 · context 262,144 tokens · up 1h 0m`)
was **rendered, measured and invisible**. Markup correct, JS correct, text
correct. Cause:

```css
.seg{ … color:#fff; font-size:10px; … }   /* a segment of the VRAM bar */
.smeta .seg{white-space:nowrap}           /* the serving line's per-figure spans */
```

Two unrelated elements sharing the class name `seg`. The bar rule's
`color:#fff` painted the serving line **white on a `#FBFCFD` surface**.
`getComputedStyle` reported `rgb(255,255,255)` on every segment.

Fix: scope the bar rule to `.bar .seg`.

**Why this matters for review:** no amount of reading `app.js` finds this bug.
It is the argument for the browser-verification step in §6. If you review only
the diff, you will not find it, and you will not find its absence either.

---

## 3. Every flaw fixed, with its mechanism

### Pass 3 (this turn) — 8 flaws

| # | Flaw | Root cause | Fix | Test |
|---|---|---|---|---|
| 1 | Serving line invisible | `.seg` collision, white-on-white | Scope bar rule → `.bar .seg` | `test_the_vram_bar_segment_rule_does_not_leak_onto_the_serving_line` |
| 2 | `starting…` flickered | `paintState` wrote `#sMeta`; `paintTelemetry` repaints it every 2 s vs state's 5 s → note survived 1 frame in 3 | Render the phase inside `paintServingMeta`, the slot's only writer. Pure `busyPhase()` + `BUSY_PHASES`, also read by the button-disabling code (the state list was written twice) | `test_a_transition_is_said_once_and_by_the_element_that_owns_it` |
| 3 | `.pill.busy` was dead CSS | Shipped with no painter, exactly like `#dirty` had | Boot reads `starting…` amber, not grey `Not reachable` | same test (`'" busy"' in body`) |
| 4 | 29 unservable model cards keyboard-dead | Code set `disabled`; its own comment promised `aria-disabled`. A `disabled` button is skipped by Tab entirely → the reason, which lives on the card's note line, was unreadable without a mouse | `aria-disabled="true"` + `.mcard[aria-disabled="true"]` styling | `test_an_unservable_model_is_still_reachable_by_the_keyboard` (executes `renderModels`) |
| 5 | `#agents` displayed a value the page wasn't using | Type `99` → box showed `99`, `agents` held `64` (the number POSTed as `max_num_seqs`) | Echo the clamp back into the box; leave an **empty** box alone (mid-edit, not out-of-range) | `test_the_agents_box_shows_the_number_it_sends` |
| 6 | `KV fits 1` beside a field reading `16` | A contradiction styled as a caption, in quiet grey, with nothing connecting the two | `agentsFitNote()` → `{text, overfit, title}`; `overfit` → `.frow .s.overfit` (amber, bold) | `test_the_agents_field_shows_its_own_kv_limit` |
| 7 | `#dBadge` printed the Python enum | `measured_other_ctx` verbatim; `unknown` is the wrong word for a capacity figure | `badgeOf()` → `measured*` / `unpredictable`, each with a `title` | `test_the_provenance_badge_never_prints_the_enum` |
| 8 | `winMeta` said `1182s` | A duration without the duration formatter; every other duration on the page uses `durTxt()` | `winSpan()` → `19m 42s` | `test_the_window_span_is_a_duration_not_raw_seconds` |

Plus: `capacity.py`'s two VRAM findings now group MiB with `{:,}`
(`free 97,801 MiB`, not `97801`) to match `fmt()` used everywhere else.
`tests/test_capacity.py` updated to assert the grouped form.

### Passes 1–2 — the load-bearing ones

- **Agents read-back.** `#agents` showed `1` against a server running `8`, while
  the recommendation column below it disagreed with the input above it.
- **`userPicked` on every control input.** Moving a slider sprang back 5 s later
  under the next state event, and the Apply confirmation quoted a util the
  operator had already changed.
- **Preflight warn ≠ pass.** Every non-blocking finding was green with a tick, so
  "KV budget is tight" and "no blockers" were indistinguishable.
- **Boot phase bar wired.** The markup shipped with the prototype and never had a
  painter: a multi-minute boot sat on `Elapsed —` with an empty track while the
  supervisor published phase, per-phase times and a real ETA the whole time.
- **`#dirty` wired.** Same class of defect: markup + a `.on` CSS rule, no painter.
- **Throughput strip.** One number where the other had been. Now three cells,
  each with a window figure, a reason for an absent one, an age for a stale one,
  and a lifetime companion that never moves up into the window slot.
- **Dead UI deleted:** banner, autoR, notify, subN, setrow, the fixed ctx ladder,
  the disabled "Set subagents" control (no endpoint wrote it), the hardcoded
  `412 GiB disk` span (nothing measured it), 47 lines of orphan CSS.
- **Log → drawer.** Held 190 px permanently to show a stream nobody read while
  reading something else.
- **DPR canvases** (`fitCanvas`) + a y-axis count scale on the histogram.
- **ctx ceiling snapped to the step grid.** A range thumb only rests on
  `min + k*step`, so an off-grid ceiling (KV fit 19,100) parked the thumb at
  16,384 while the readout said 19,100 — the control disagreeing with its own
  value. `ctxTop()` snaps **down**, never up.
- **Percentile captions** on `#reqHist` were clipped off the top and overprinting
  each other. Two causes: canvas `textBaseline` **leaks between drawing blocks**
  (the count gridline leaves it `alphabetic`, so glyphs anchored upward off a
  96 px canvas — drawn, invisible), and a stateful `rowLast` that compared only
  against row 0, double-booking every crowded label onto row 1. Extracted pure
  `markRows(xs, gap, rows)`.
- **Fine-bin histogram.** `reqstats.WindowStats` emits `fine` bins on a round step
  ladder; interval-only observations return separately as `intervals` and are
  drawn as a translucent underlay, **never merged into a bar** (a request known
  only to `(20000, 50000]` placed in one 5k bin would be drawn at a position no
  measurement supports). `histData(w)` picks.

---

## 4. Test bar applied (read this before weakening anything)

Owner's standing bar: *"make the life of the code writer hell — a test that is
easy to pass is a defect."* Consequences you will see all over `test_ui.py`:

- **Prefer EXECUTING a pure function over grepping source.** A grep assertion
  survives every behaviour mutation, i.e. it is decorative. So every decision was
  extracted into a pure JS function and run in `dukpy` (QuickJS) against a DOM
  shim: `histData`, `railSigOf`, `railCountOf`, `dirtyBits`, `bootActive`,
  `bootNote`, `ctxTop`, `markRows`, `agentsFitNote`, `badgeOf`, `winSpan`,
  `busyPhase`.
- **Literal oracles, not recomputed properties.** `top(19_100) == 16_384` is
  written out. An assertion that recomputes the same `floor()` the function under
  test performs passes whatever it returns.
- **Painters are executed too**, not just grepped: `_rail_cards()` runs
  `renderModels()` and dumps the cards it built; `_render()` runs
  `paintThroughput`/`paintServingMeta` and dumps every element's text.
- **`_rendered_text()` strips comments** before any "does the page say X"
  assertion. Comments are for readers; only what survives it reaches the screen.

### Four defects in my own tests, all found by the mutation battery

Worth reviewing because they are the reason the battery exists:

1. `renderctx-stopped-calling-agentsfitnote` **survived** the first run. My own
   comment `// … see agentsFitNote()` kept the string alive, so grepping raw
   source passed with the call deleted. Fixed by routing the wiring test through
   `_rendered_text()`.
2. A pure-function test proves the decision is right and **nothing** about
   whether the page calls it. Added
   `test_the_pure_decisions_are_actually_wired_into_their_painters`.
3. Fail-before against a *reconstructed* pre-fix file can pass **vacuously**:
   reversing the call site still leaves a newly-introduced pure fn defined.
   `/tmp/prefix_fix3.py` deletes the fn body outright.
4. My own assertion `assert "s ·" not in span(...)` matched
   `request`**`s ·`**` 19m`. Write the literal you mean (`"1182s"`), not a proxy.

---

## 5. Proof already obtained — re-run it, don't trust this table

```bash
cd /home/pctablet505/Projects/servedeck-integrate

# Full suite. NOTE: inherited PYTEST_ADDOPTS carries `-n 6` but xdist is NOT
# installed, so the env -u is required, not cosmetic.
env -u PYTEST_ADDOPTS /home/pctablet505/Projects/servedeck/.venv/bin/python -m pytest -q
#   -> 489 passed, 3 skipped

# Mutation battery: 39 mutants, byte-replace + md5-verify restore.
/home/pctablet505/Projects/servedeck/.venv/bin/python /tmp/mutate_ui.py
#   -> 39/39 caught, all files restored (md5 verified)

# Fail-before, both passes (installs reconstructed pre-fix files, restores).
/home/pctablet505/Projects/servedeck/.venv/bin/python /tmp/prefix_fix2.py
/home/pctablet505/Projects/servedeck/.venv/bin/python /tmp/prefix_fix3.py

# Import smoke check. There is NO node/nodejs on this box and no ruff in the
# venv, so app.js gets its only syntax check from the browser and from
# test_ui.py's own _scan_js() brace/literal scanner.
/home/pctablet505/Projects/servedeck/.venv/bin/python -c "import servedeck.app; print('ok')"
```

Suite history: 464 → 480 (pass 1) → 481 (histogram captions) → **489** (pass 3).

The 13 mutants added this turn, each caught:
`css-seg-unscoped-again`, `card-disabled-not-aria`,
`agentsfit-comparison-flipped`, `agentsfit-always-overfit`, `badge-unknown-word`,
`badge-enum-leaks`, `winspan-raw-seconds`, `winspan-exact-count-inverted`,
`busyp-phase-always-empty`, `pill-never-shows-busy`,
`agents-box-clamp-not-echoed`, `renderctx-stopped-calling-agentsfitnote`,
`paintstate-stopped-calling-busyp`.

**Re-run the battery after ANY edit to `app.js` / `style.css` / `index.html`.**
It restores by writing original bytes back and md5-verifying — never
`git checkout`.

---

## 6. Live verification already done

Frontend is `no-store` → an html/css/js change needs only a browser reload.
A **Python** change needs a uvicorn restart.

Verified in the live page, with zero `pageerror`:

- `#sMeta .seg` computed colour `rgb(255,255,255)` → **`rgb(90,102,114)`**,
  font-size 10px → 12px. `#segW` (the bar) still `rgb(255,255,255)`.
- 29 cards `aria-disabled="true"`, `disabled` attribute count **0**,
  `tabIndex 0`, `focus()` lands on them, opacity 0.5.
- `#agents`: `99 → 64`, `0 → 1`, `8 → 8` (unchanged when in range).
- `#agentsFit`: `class="s mono overfit"`, title
  `at 16,384 context the KV budget holds 1 agent — the field above asks for 16 and is over budget`.
- `#winMeta`: `100/100 requests · 21m 06s · 98 exact`.
- `#dBadge`: `measured` / `measured*` with provenance titles, never an enum.
- Alert: `free 97,801 MiB leaves under 4,096 MiB above the 93,971 MiB budget`.
- Feeding a synthetic `STARTING` state through the real `paintState`:
  `#sMeta` → `starting…`, `#pill` → `pill busy` / `starting…`; a following
  `READY` state restores the serving line and `pill`.

### ⚠ The server I restarted — check before you conclude "the fix didn't work"

To pick up `capacity.py` I killed the manual uvicorn that held :8010. **A uvicorn
holding open SSE streams does not free the port immediately — it drains.** It took
~60 s, and a replacement started in the same command chain failed with
`[Errno 98] address already in use` while the *old* code kept serving.

Current owner of :8010: **pid 3286570**,
`servedeck/.venv/bin/python -m uvicorn servedeck.app:app --host 127.0.0.1 --port 8010 --no-access-log`
with `PYTHONPATH=/home/pctablet505/Projects/servedeck-integrate`, i.e. it serves
**this edited tree**. The systemd *user* unit on disk is stale
(`ExecStart=%h/servedeck/.venv/…`, path does not exist → `203/EXEC`); do not use
`systemctl --user restart servedeck.service` to bounce the dashboard.

If you need to restart it: kill, poll `ss -ltnp | grep 8010` until free, then start.

---

## 7. Review checklist — what I would want challenged

1. **`busyPhase` moved the busy note into `paintServingMeta`, which now
   early-`return`s.** Confirm nothing downstream of that `return` needed to run
   during a transition. The throughput strip is painted by a *separate* painter
   (`paintThroughput`, from telemetry), so it keeps updating — that is intended,
   and the `title` says the figures are the previous server's last ones.
2. **`BUSY_PHASES` is a single-line `const` on purpose** — the test harness's
   `_const_line()` helper lifts one line verbatim. Re-wrapping it multi-line
   breaks `_render`/`_exec` with a misleading `BUSY_PHASES is not defined`.
3. **`aria-disabled` cards are focusable but have no `onclick`.** Verify that is
   the intended interaction (reach + read the reason, cannot select) and that
   `role="option"` inside `role="listbox"` is acceptable with `aria-disabled`
   rather than needing `aria-selected` on every card.
4. **The empty-box carve-out in the `#agents` handler** (`value !== ""`) — is
   leaving `agents = 1` while the box is blank the right call, or should a blank
   box disable Apply?
5. **`badgeOf` renames `unknown` → `unpredictable`.** This is a user-visible
   string change on a page whose whole theme is provenance honesty. Check it
   against `docs/SPEC.md` §3 `UNKNOWN_CAPACITY`, which is the spec's own term.
6. **`capacity.py` `{:,}` grouping** changed two finding `detail` strings.
   `tests/test_capacity.py` asserts `"7,887"` / `"48,943"`. Grep for any other
   consumer of those strings (gateway error bodies, docs) before merging.
7. **`_DOM_SHIM.querySelector` now returns a stub element instead of `null`** so
   `renderModels` can execute. That is a harness-wide behaviour change — check no
   existing test relied on `null`.
8. **`servedeck.log`** — untracked, not ignored. Add to `.gitignore` or delete.

---

## 8. Known, deliberately NOT fixed

- **Dark-theme CSS block is a true duplicate** of the `prefers-color-scheme` one.
  Cannot be merged without a preprocessor (one sits inside `@media`, one does not).
- **The `hist` sparkline buffer is not cleared when the server restarts.** Old
  samples scroll out within ~80 s, so the case is weak — but it is the one place
  the page's "figures are never carried over from a different model" promise is
  not enforced in code.
- **`Smoke test` button is `disabled` with `title="not implemented yet"`** — the
  page's rule is that an inert control must say why, which it does. No endpoint
  behind it yet.

---

## 9. Suggested commit split

The tree is one coherent UI effort, but three reviewable chunks:

1. `reqstats` fine bins + `test_reqstats.py` (`servedeck/reqstats.py`,
   `tests/test_reqstats.py`, and `histData`/`reqHist`/`markRows` in `app.js`).
2. UI wave P0–P7 + refactor (`index.html`, `style.css`, most of `app.js`,
   `app.py`'s `_snap()`, `supervisor.py`).
3. Second audit pass (the 8 flaws above, `capacity.py`, `test_capacity.py`,
   the 8 new tests).

Commit-message style in this repo: imperative subject, then a body that argues
the *why* per hunk, ending with
`Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

Then: `git push -u origin ui-wave`, open a PR into `main`
(`https://github.com/pctablet505/servedeck.git`), and note in it that the branch
has never been pushed and carries 24 prior commits.

---

## 10. Reproducing the flaw class, for the next pass

The two most productive moves this turn, both of which read the *rendered* page:

```js
// 1. invisible text: computed colour vs the painted background, all elements
for (const el of document.querySelectorAll('body *')) {
  const t = /* own text nodes */; if (!t) continue;
  let bg = 'rgba(0, 0, 0, 0)', p = el;
  while (p && (bg === 'rgba(0, 0, 0, 0)')) { bg = getComputedStyle(p).backgroundColor; p = p.parentElement; }
  if (getComputedStyle(el).color === bg) report(el, 'INVISIBLE');
}
// 2. clipped / overflowing text
if (el.scrollWidth > el.clientWidth + 1 && cs.overflowX !== 'auto') report(el, 'CLIPPED');
```

Plus: enumerate `document.styleSheets` for **unscoped class rules that collide**
with a scoped rule elsewhere — that is how the `.seg` bug shows up as a
duplicated selector rather than as a mystery.
