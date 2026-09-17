# Handoff — Servedeck full-codebase defect audit

**Date:** 2026-09-11
**Repo:** `/home/pctablet505/Projects/servedeck`
**Branch / HEAD:** `main` @ `38b32c2` (clean tree, `git status` = 0 modified files)
**Task for the auditor (Codex):** find **every** defect in the codebase. This is a
whole-repo audit, not a UI pass. The UI was already audited twice on 2026-09-10
(see §5) — do **not** re-report those; look for what they missed and for
regressions.

---

## 0. Read this first — the state is clean and committed

Unlike the 2026-09-10 UI-wave handoff (which was an uncommitted working tree),
**this tree is clean.** `git status --short` returns nothing. HEAD contains all
the work. So:

- You **may** use `git checkout` / `git stash` freely — nothing uncommitted can
  be lost.
- The live dashboard (`127.0.0.1:8010`, uvicorn pid 88501) is running from
  `servedeck/.venv` with **no** `PYTHONPATH` override, i.e. it serves this same
  committed tree. Frontend is `Cache-Control: no-store` → a browser reload picks
  up JS/CSS/HTML changes with no restart.
- **Do not** `git checkout` files to "get a baseline" and then forget to
  restore — the tree is clean, so any edit you make is a real diff. Keep the
  audit **read-only** unless you are writing the fix + its test.

### Test baseline (verified 2026-09-11)

```
597 passed, 1 skipped, 4 warnings in 7.99s
```

Run with:

```bash
cd /home/pctablet505/Projects/servedeck
env -u PYTEST_ADDOPTS .venv/bin/python -m pytest -q
```

- The inherited `PYTEST_ADDOPTS` contains `-n 6` but **xdist is not installed**
  → you **must** `env -u PYTEST_ADDOPTS` or pytest errors out.
- No `ruff` in the venv, no pre-commit hook. Long lines >100 are pre-existing
  and tolerated.
- Smoke check that the app imports: `.venv/bin/python -c "import servedeck.app"`.

### Environment gotchas (do not trip on these)

- **Port 8010 is live.** SIGTERM to a uvicorn holding open SSE streams does NOT
  free the port immediately — it drains (~60 s). If you restart it: kill, poll
  `ss -ltnp | grep 8010` until free, then start. Starting a replacement in the
  same command chain fails with `[Errno 98] address already in use` and the
  *old* code keeps serving.
- The systemd **user** unit `servedeck.service` on disk is **stale**: its
  `ExecStart` points at `~/servedeck/.venv` which does not exist →
  `systemctl --user restart servedeck.service` fails with status=203/EXEC. Do
  not rely on it.
- The `/tmp/mutate_*.py` files in this workspace are for a **different project**
  (`contributor-value-index` / `cvi`), **not** servedeck. Ignore them.

---

## 1. What Servedeck is

A local web dashboard for a self-hosted LLM server (vLLM). It answers: how much
context fits on the GPU, how many agents run in parallel before thrashing, is
the prefix cache working, did it crash or never start. It reads the GPU and the
server's Prometheus `/metrics`; it does **not** replace the user's launch
script — it runs the one they already have. Binds `127.0.0.1` only, no external
network, never `sudo`.

Frontend is hand-written `index.html` / `style.css` / `app.js` — no framework,
no build step. Backend is FastAPI. Spec lives in `docs/SPEC.md`; module
docstrings cite SPEC sections.

---

## 2. Architecture map (module → responsibility)

Read the module docstring first; it states the SPEC contract and the invariants
the code is transcribed from. **Several modules are deliberately "verbatim"
transcriptions of SPEC.md — do not "improve" them; if a pattern is wrong, the
SPEC is wrong first.**

| Module | Lines | Responsibility | Key invariant |
|---|---|---|---|
| `app.py` | 1760 | FastAPI app: read-only observability + capacity estimation + pass-through proxy. Server control is **not** wired here (buttons render disabled). | Named `app` (not `api`) because `run.sh` + systemd import `servedeck.app:app`. |
| `supervisor.py` | 1747 | **TOP-PRIORITY.** Desired/actual state machine for the one vLLM server; startup reconciliation; boot-phase monitor; crash-vs-failed-boot; backoff auto-restart with crash-loop ceiling; death recording. | Auto-restart gated on `desired_state == "RUNNING"`, full stop. `stop()` persists `STOPPED` first. |
| `registry.py` | 855 | Model discovery over `~/.cache/huggingface/hub/models--*/`; servability + `KNOWN_ARCHS`; append-only `state/measurements.json`; `resolve_inputs()` three-tier lookup. | Pure discovery + bookkeeping: does not launch, does not compute VRAM budgets. |
| `procctl.py` | 773 | Process control. **The self-match trap:** pattern-matching lookup tools match the FULL command line including the calling shell's own. | Must never trip its own regex-anchored token match, comments included. |
| `capacity.py` | 710 | VRAM/KV budget arithmetic. **PURE** — no I/O, no subprocess, no network. | Every environmental fact arrives via the optional `live` (`LiveFacts`) param. Formulas transcribed from SPEC §3. |
| `metrics.py` | 709 | Hand-rolled Prometheus text-format parser for the live vLLM `/metrics`. | Metric names checked against a real source, never guessed; vLLM renames between versions. |
| `gateway.py` | 580 | Transparent reverse proxy from Servedeck's listener to the active backend's `port`. Two endpoints: `/v1/chat/completions`, `/v1/responses`. | Upstream port comes from the active `[backends.<name>]` in `servedeck.toml`, never named here. |
| `reqstats.py` | 509 | Rolling window over the last N *finished* requests, rebuilt from vLLM histograms. Emits `fine` bins + separate `intervals`. | Never merge an interval observation into a bar. |
| `phases.py` | 462 | Boot-log → ordered phase state machine (`PhaseTracker`) + typed `Failure` classification. | Every regex transcribed VERBATIM from SPEC §5. Do not generalize. |
| `history.py` | 390 | Append-only `state/history.jsonl`, one record per boot attempt / exit. | `supervisor.py` is the **only** writer. |
| `preflight.py` | 382 | Machine-level go/no-go checks (the I/O half of `capacity.py`). Feeds `LiveFacts`. | `capacity.py` stays pure; this is what collects the facts. |
| `updetect.py` | 340 | Which port the dashboard watches, and why. | Reads the live process on `rt.port`; a stale header must not point the dashboard at a dead port. |
| `smoke.py` | 340 | Two-request smoke test run **through the gateway** (not the backend port). | Proves the gateway path too. Pass iff tool-call name + JSON args. |
| `tokens.py` | 321 | Input/output token accounting + prefix-cache-served fraction, from 3 vLLM counters. | `prompt_tokens_total` includes cached; `prompt_tokens_cached_total` is the cached subset. |
| `kvcalc.py` | 317 | Per-architecture KV-cache arithmetic from the model's own `config.json`. | Replaces the old single-rate-per-repo + family-median fallback (both were wrong). |
| `shellconfig.py` | 292 | Servedeck's **only** writer of `local_llm/.config`. | Every mutation goes through `codex-qwen.sh set-*` as a subprocess; never edit the file directly. |
| `gpu.py` | 281 | `nvidia-smi` / `journalctl` wrappers + Xid classification. | Tolerates the tool being slow/absent/erroring → returns clearly-empty/None, never raises into a mid-render caller. |
| `disksize.py` | 264 | Disk-size accounting. | — |
| `parallelism.py` | 249 | "How many agents in parallel right now" — a KV-*admission* question, not a compute one. | The GPU is not the constraint; the KV pool is. |
| `logtail.py` | 213 | Log tailing. | — |
| `events.py` | 190 | In-process pub/sub SSE hub. One bounded `asyncio.Queue(maxsize=512)` per subscriber; a full queue drops the OLDEST event + increments a `dropped` counter. | A publish never blocks the publisher and never raises. |
| `legacy.py` | 181 | Every surviving `coldstart` spelling, in one module. Reads the old spelling as a DEPRECATED ALIAS, preferring the new one. | — |
| `config.py` | 283 | All machine-specific values. Resolution: `SERVEDECK_*` env → `servedeck.toml` → auto-detect → defaults. | Nothing else in the package may hardcode a path, GPU size, or model name. |
| `paths.py` | 104 | Every absolute path Servedeck touches, as module constants. | No other file may spell out one of these paths itself. |
| `web/app.js` | 1948 | Entire frontend logic: poll, render, SSE, canvas charts, user input. | — |
| `web/index.html` | 342 | Markup. | — |
| `web/style.css` | 395 | Styles. | — |

Total ≈ 27,878 lines (incl. tests).

---

## 3. How to audit (method)

For **each** finding, before reporting it:

1. **Read the surrounding 20 lines** to confirm the defect is real and not
   guarded elsewhere.
2. **Check the tests.** The suite is deep (≈ 5,900 lines of tests). If a passing
   test exercises the path and asserts correct behaviour, it is **not** a
   defect — drop it. Note: a test that only greps the source (asserts a function
   name appears) is **decorative** and does NOT count as coverage.
3. **Check `git log --oneline -20 -- <file>`** — recently changed code is more
   likely to be buggy.
4. If you are not 100% sure, mark it **NEEDS VERIFICATION** and say what would
   confirm it.

**No speculation.** Every finding cites exact code (file:line + a 1–5 line
quote). Better to report 30 real defects than 5 vague ones.

### Severity scale

- **P0** = security / data loss / crash / kill-wrong-process / wrong number shown to user.
- **P1** = real bug affecting users.
- **P2** = edge case / minor.
- **P3** = code smell that hints at a bug.

### Output format (per finding)

```
N. [P0|P1|P2|P3] file:line — short title
   Evidence: <quote the exact code, 1-5 lines>
   Why: <1-3 sentences on why this is a defect and what goes wrong at runtime>
   Fix: <1-2 sentences on the fix>
   Test coverage: <test name, or "none", or "decorative (grep-only)">
```

End each audit section with a **"NOT DEFECTS (checked and cleared)"** list of
3–5 things you suspected but confirmed fine, so the next reader knows you looked.

---

## 4. The six audit lanes (run in parallel if you can)

Split the repo into six independent lanes so agents can work concurrently. Each
lane is self-contained; a finding that crosses a lane boundary should be
reported by the lane that owns the file.

### Lane A — `app.py` + `gateway.py` + `events.py`
Hunt: async/await bugs (blocking sync I/O / `subprocess.run` / `time.sleep` /
`open()` inside async handlers; missing `await`); SSE connection leaks (client
disconnect not detected/cleaned up, unbounded subscriber lists, generator not
closed on error, no keepalive, no `Last-Event-Id`); race conditions on shared
mutable state from multiple async tasks; unvalidated input (`int()`/`float()`
without try/except, list index without bounds, dict access without `.get()` on
user keys); error handling (bare `except`, missing cleanup in `finally`, wrong
status code, internals leaked in the message); API contract (response shape
inconsistent across endpoints, 200-for-error, camelCase/snake_case mixed);
resource leaks (file handles, subprocess not waited, timer not cancelled);
security (command injection via `shell=True`, path traversal, SSRF, missing auth
on a mutating endpoint, CORS misconfig, secrets in response); logic
(off-by-one, inverted boolean, dead code, copy-paste-identical handlers that
should differ); performance (O(n²) hot path, whole-file read where streaming
would do, re-parsing config per request).

### Lane B — `supervisor.py` + `procctl.py` + `phases.py`
Hunt: process-lifecycle races (start-while-starting, stop-while-starting,
double-kill, restart that loses the PID, two coroutines writing the same state
file, check-then-act on process existence where the PID is reused between check
and kill); zombies/leaks (subprocess not waited after kill, `Popen` without
`wait`, pipe not closed → deadlock on a full pipe buffer, fd leak per
start/stop cycle, log handle kept open forever); signal handling (SIGTERM vs
SIGKILL ordering, missing grace period, **killing the wrong PID** — pattern match
too broad and hits the supervisor itself or an unrelated process, process-group
vs single-PID, no handling of "already dead" / `ProcessLookupError`); state
machine (impossible transition, missing transition, transition that skips
cleanup, state written to disk before the action completes → crash leaves
inconsistent state, no timeout on a transient state → stuck in "starting"
forever); health check (interval too tight/loose, no max-retry, always-passes
because it checks the wrong port/process, blocks the event loop, no
"not-up-yet" vs "permanently-failed" distinction); `/proc` parsing (assumed
field layout, process exits between read and parse, `int()` on an empty field,
symlink that disappears, `cmdline` null-byte separator); error handling (bare
`except` around subprocess, swallowing `ProcessLookupError`, catching `Exception`
and continuing as if healthy, missing cleanup when a start fails halfway);
concurrency (shared dict mutated without a lock, `asyncio.Lock` held across an
`await` that can raise → lock never released, long health check holding a lock
that blocks `stop()`, two supervisors for one model); resource limits (unbounded
log growth, unbounded retry, no cap on concurrent servers, history memory leak);
logic (off-by-one in retry count, inverted boolean, copy-paste between
start/stop, wrong PID stored — parent vs child, shell vs python).

### Lane C — `registry.py` + `preflight.py` + `updetect.py` + `smoke.py` + `legacy.py`
Hunt: path handling (traversal from a user-supplied model name into
`open()`/`Path()`, symlink following, relative-vs-absolute confusion, missing
`realpath`, TOCTOU where the path changes between check and use, non-UTF-8
filenames, empty-string path); file handling (reading a file mid-write, no size
check before reading into memory, assuming a file exists after a glob, no
permission-error handling, "file is a directory", broken symlink); registry
logic (duplicate names, case-sensitivity, add-twice, remove-while-in-use, stale
entry whose path was deleted, size in the wrong unit, a status value that can
never be reached); preflight gaps (a check that always passes — wrong units or
wrong path, a missing check, a check that blocks on slow I/O with no timeout,
boundary `>=` vs `>`, not accounting for already-allocated memory); update
detection (false positive on mtime-only touch, false negative when content
changes but mtime is preserved — `cp -p`/`rsync --preserve`, hash of a file still
being written, no handling of a vanished file, size-only comparison, stat/open
race); smoke test (always-passes by catching the exception and returning True,
wrong endpoint, no timeout, prompt too large, no response-content validation,
mutates server state); legacy migration (not idempotent, loses a field, crashes
on an empty file, no backup, silently skips a malformed line); error handling
(bare `except`, swallowing `FileNotFoundError` and returning a default that hides
the problem, transient network error with no retry); concurrency (registry dict
mutated while iterated, two preflight checks writing the same result file, smoke
test running mid-restart); logic (off-by-one, inverted boolean, `>` vs `>=`,
dead code, returns a value it never computes).

### Lane D — `capacity.py` + `kvcalc.py` + `parallelism.py` + `gpu.py` + `metrics.py` + `reqstats.py` + `tokens.py` + `history.py`
Hunt: unit errors (MiB vs MB vs bytes vs GiB, a value computed in bytes but
displayed as MiB, a threshold in MiB compared against a value in bytes, 1024 vs
1000, a missing `/1024` or `/1048576`); formula errors (KV-cache =
`seq_len × num_layers × num_heads × head_dim × dtype_bytes` — check each factor:
`num_heads` vs `num_kv_heads` for GQA/MQA, forgetting ×2 for K and V,
`hidden_size` vs `head_dim×num_heads`; integer division truncating a float;
dimensionally wrong); off-by-one/boundary (a window including one extra/missing
sample, a histogram bin that drops the max, a percentile with the wrong
interpolation, a "fits" check using `>` instead of `>=` at the exact boundary);
division by zero (a count that can be 0 — empty window, no requests yet; a value
that can be 0 — zero GPU memory, zero tokens; no guard); overflow/underflow
(numpy `int32`, a value that should be a float but is an int, a negative value
where only positive makes sense); GPU query (fragile `nvidia-smi` regex that
breaks on a locale decimal, no handling of `N/A`/`[Not Supported]`, a GPU removed
mid-query, a stale cached value, no timeout on the subprocess, assuming exactly
one GPU); metrics aggregation (a window that never closes, a counter not reset on
window roll, a moving average dividing by the wrong count, a p95/p99 on a list
that is not actually sorted, a metric computed but never emitted, or emitted but
never computed); histogram (bin too wide/narrow, the fine-vs-coarse path merging
two populations, an interval observation silently dropped, a histogram not
cleared on window reset); token counting (off by a constant factor, double-count
prompt+completion, not thread-safe, tokenizer loaded per call); history
(unbounded growth, a cap that evicts the wrong end, written but never read back,
read while locked by another process, an entry missing a field the reader
expects); concurrency (shared counter without a lock, a window torn-read, a dict
resized while iterated, two threads writing the same metric); logic (returns the
wrong variable, inverted comparison, dead branch, a default of `0` where `None`
means "unknown").

### Lane E — `config.py` + `paths.py` + `shellconfig.py` + `disksize.py` + `logtail.py` + `__main__.py` + `__init__.py` + `web/index.html` + `web/style.css` + `systemd/` + `run.sh`/`stop.sh`/`setup.sh` + `servedeck.toml(.example)` + `pyproject.toml` + `requirements.txt` + `.gitignore`
Hunt: config parsing (a TOML key read under the wrong name, a wrong default, a
value parsed as the wrong type, a missing key that crashes instead of using the
default, a key accepted but never used, a key used but undocumented in the
`.example`); path handling (relative where absolute is required, `~` not
expanded, used before created, two things writing the same file, hardcoded
instead of from config, breaks on a username with a space); shell config (a
command built by string formatting instead of a list, unquoted → breaks on a
path with spaces, unnecessary `shell=True`, inherits the environment when it
should not or vice versa, not found on a minimal `PATH`); disk size (wrong
units, not accounting for block size, computed before the file is fully written,
`du`/`stat` following symlinks when it should not, a stale cache); log tailing
(infinite spin when the file is not growing, reads the whole file into memory
every poll, loses lines on rotation, no handling of delete-and-recreate, blocks
the event loop with a sync read, does not seek to the end on start); entry point
(no clean SIGINT/SIGTERM, no missing-config check, starts the server before
validating config); HTML/CSS (a **CSS class-name collision** — a class used for
two purposes; this was a real 2026-09-10 bug, check for more; missing `alt`, an
input with no label, a button `disabled` but not `aria-disabled`, a `z-index`
that hides an element, insufficient contrast, a `<div>` where a semantic element
is required, missing viewport meta, a script that runs before the DOM is ready);
systemd (a unit pointing at a path that does not exist — **the user unit's
`ExecStart` points at `~/servedeck/.venv` which does not exist**; missing
`After=`/`Wants=`, `Restart=always` hammering a crasher, runs as root when it
should be the user, missing hardening where it matters); shell scripts (an
unquoted variable, no `set -e`, a relative path that depends on CWD, no exit-code
check on a dependency, installs without checking, not idempotent); packaging (an
unpinned or non-existent version, a dep in `requirements.txt` but not
`pyproject.toml` or vice versa, an import with no declared dep, a Python-version
range that is too broad); `.gitignore` (too broad / too narrow / missing a
generated-file pattern).

### Lane F — `web/app.js` (+ `web/index.html`)
Hunt: XSS/injection (`innerHTML`/`outerHTML`/`insertAdjacentHTML`/`document.write`
set from server- or user-controlled data — a model name, a log line, an error
message — without escaping; `textContent` is safe, `innerHTML` is not);
null/undefined access (a DOM query that can return null then dereferenced, a
missing JSON field then `.split()`/`.toFixed()`, a callback called with too few
args); event-listener leaks (a `setInterval` not cleared on re-init, a click
handler added on every render → fires N times after N renders); polling races
(two responses out of order, the older overwrites the newer, overlapping
fetches, a non-atomic state update with a render in between); SSE handling
(`EventSource` not closed on `beforeunload`, `onerror` does not reconnect or
reconnects in a tight loop with no backoff, an unknown event type silently
dropped, the stream opened before the initial poll populated state); state
management (a global written by two paths without coordination, reset on poll
but not on SSE or vice versa, a dirty flag set-but-never-cleared or
cleared-but-never-set, a cache not invalidated when its input changes); canvas
(not scaled for `devicePixelRatio`, context not cleared before the next draw →
ghosting, a stale size after a resize, wrong `textBaseline` — a 2026-09-10 bug
drew labels off-canvas, a chart that does not handle an empty data set →
divide-by-zero / NaN); user input (a number input that accepts "abc" with no NaN
check, a slider value used without clamping to `[min,max]`, a text input used in
a URL without `encodeURIComponent`, a form submitted on Enter when it should not
be, a button not disabled while its action is in flight → double-click → double
request); API-contract mismatch (a fetch expecting a field the backend does not
return — grep `app.py` to verify; a parameter the backend does not accept; an
array treated as an object or vice versa; a 4xx/5xx not checked via `res.ok`
then `.json()`'d and the error body treated as data); timer/lifecycle (a
`setInterval` never cleared, a `setTimeout` capturing a stale closure, a
recursive `setTimeout` that does not stop on error, an uncancelled
`requestAnimationFrame`); logic (off-by-one, inverted comparison, returns a value
it never computes, copy-paste-identical handlers that should differ, a fallback
that hides a real error); accessibility (a `disabled` button that keyboard users
cannot focus to read its tooltip — a 2026-09-10 bug; an input with no label, no
`aria-live` on a changing region, a missing focus trap).

---

## 5. What was already audited (do NOT re-report; DO regression-check)

Two audit passes ran on 2026-09-10 (branch `ui-wave`, now **merged into
`main`**). These defects were **found and fixed** — verify the fixes still hold,
then move on. Full detail: `HANDOFF-SERVEDECK-UI-WAVE-2026-09-10.md`.

- **CSS class-name collision** — `.seg` (VRAM bar segment, `color:#fff`) also
  matched `.smeta .seg` (the serving line's per-figure spans) → the serving line
  rendered white-on-`#FBFCFD`, i.e. invisible. Fixed by scoping the bar rule to
  `.bar .seg`. *This is the headline "read computed styles, not source" lesson.*
- **"Control disagrees with its own value"** — typing 99 in `#agents` showed 99
  but `agents` held 64 (the value POSTed). Fixed: echo the clamp back, but leave
  an empty box alone.
- **`disabled` vs `aria-disabled`** — a disabled button is skipped by Tab, so the
  29 unservable models' reason was unreadable. Fixed: `aria-disabled`.
- **Two writers, one slot** — `paintState()` and `paintTelemetry()` both wrote
  `#sMeta` on different cadences. Fixed: single owner, busy phase rendered from
  pure `busyPhase()`/`BUSY_PHASES`.
- **Backend enum leaked into prose** — `#dBadge` printed `measured_other_ctx`
  verbatim; raw seconds "1182s". Fixed: `badgeOf()` maps to "measured*"/
  "unpredictable"; `winSpan()` uses `durTxt()`.
- **Canvas `textBaseline` leak** — a gridline set "bottom" then "alphabetic", so
  percentile captions drew upward off the top of the 96px canvas. Fixed: state
  `textBaseline` before each `fillText` group.
- **Stagger/declutter double-booking** — a stateful `rowLast` comparing only
  against row 0 double-booked every crowded label onto row 1. Fixed: pure
  `markRows(xs,gap,rows)`.
- **Range-input thumb off-grid** — an off-grid max (KV fit 19,100) parked the
  thumb at 16,384 while the readout showed 19,100. Fixed: `ctxTop()` snaps down.
- **MiB grouping** — `capacity.py` now groups MiB with `{:,}` to match `fmt()`.

**Reusable test technique** (from those passes): to make a JS decision
testable, extract a **pure function** and execute it in `dukpy` (the `test_ui.py`
DOM shim). Prefer executing a pure fn over grepping source — grep assertions
survive every behaviour mutation (decorative). A pure-fn test proves the decision
is right but proves **nothing** about whether the page calls it — add a wiring
test that uses `_rendered_text()`, not a raw-source grep.

---

## 6. Deliverable

Produce a single consolidated report, `AUDIT-SERVEDECK-FULL-2026-09-11.md`, at
the workspace root (`/home/pctablet505/Projects/`), structured as:

1. **Summary table** — count by severity (P0/P1/P2/P3) and by lane.
2. **Findings** — the six lanes in order, each with its numbered findings in
   the §3 output format, sorted by severity within the lane.
3. **NOT DEFECTS (checked and cleared)** — per lane.
4. **Regression check** — which of the §5 2026-09-10 fixes still hold.
5. **Recommended fix order** — the P0s first, with a one-line "why now" each.

Do **not** apply fixes in this pass unless asked — this is an audit. If a P0 is
found that is actively dangerous (e.g. the stale systemd unit, or a
kill-wrong-process path), flag it at the top of the summary so it can be
triaged immediately.
