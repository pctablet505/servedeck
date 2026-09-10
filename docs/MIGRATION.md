# Migrating :8010 from `coldstart` to Servedeck

The dashboard on `127.0.0.1:8010` is currently served by `~/Projects/coldstart`
— a fork of this codebase that is not a git repository and whose
machine-specific values (paths, ports, GPU size, model names) are hardcoded in
its modules. Servedeck is the same program with those values moved into
`servedeck.toml`, plus the fixes listed below.

This document is the cutover. **Nothing here has been run**: the coldstart tree
is untouched and still serving. The owner performs the switch.

---

## Before you start

| | coldstart | servedeck |
|---|---|---|
| Tree | `~/Projects/coldstart` (not a git repo) | `~/Projects/servedeck` (github.com/pctablet505/servedeck) |
| Package | `coldstart/` | `servedeck/` |
| Venv | `.venv-gui` | `.venv` |
| ASGI app | `coldstart.app:app` | `servedeck.app:app` |
| Web assets | `web/` (outside the package) | `servedeck/web/` (ships in the wheel) |
| Machine config | hardcoded in `paths.py`, `registry.py`, `capacity.py`, `supervisor.py` | `servedeck.toml` |
| State | `~/Projects/coldstart/state/` | `state_dir` in `servedeck.toml` |
| systemd unit | `systemd/coldstart.service` | `systemd/servedeck.service` |

Both bind `127.0.0.1:8010`, so **only one can run at a time**.

---

## 0. The `coldstart` -> `servedeck` rename

The rename is complete inside this repository: module, package, class,
function and variable names, config keys, environment variables, the console
entry point, the systemd unit, the log and state paths, the gateway's error
`type`, the docs and the UI strings. Three things are DELIBERATELY still
spelled the old way, because the outside world has not been renamed with them.

### The deprecated aliases Servedeck still reads

| Old name | New name | Why the old one survives |
|---|---|---|
| `COLDSTART_URL` (in `.config`) | `SERVEDECK_URL` | `~/Projects/local_llm/.config` carries `COLDSTART_URL=""` and `codex-qwen.sh`'s `CONFIG_ALLOWED_KEYS` still lists it. Dropping it would make Servedeck read a key that does not exist and fall back to `http://localhost:$PORT/v1` — pointing Codex at the model server and around the gateway, with nothing reporting an error. |
| `USE_COLDSTART` (in `.config`) | `USE_SERVEDECK` | Same file, same allow list. |
| `COLDSTART_*` env vars | `SERVEDECK_*` env vars | A shell that exported the old prefix keeps working: `COLDSTART_PORT`, `COLDSTART_STATE_DIR`, `COLDSTART_CONFIG` and friends are read as their `SERVEDECK_*` equivalents. |

The new name **wins whenever both are set**, so a `.config` mid-migration is
never ambiguous. Every alias lives in one module, `servedeck/legacy.py`, so
that removing compatibility later is one file to read and one grep to trust;
`tests/test_rename.py` pins each of them.

### The pre-rename state directory is read, never written

`~/Projects/coldstart/state/` holds **47 boot records and 7 KV measurements**
— every boot this box has actually recorded, because the coldstart fork is
what has been serving `:8010`. Servedeck reads that directory alongside its
own `state_dir` and merges the two, oldest tree first, de-duplicated on
record content.

* It is **read-only**. Nothing in Servedeck writes inside `~/Projects/coldstart`
  — the fork may still be running out of it, and a second writer would be
  corrupting a live process's state. Rollback is therefore always clean.
* "Read the old file only when the new one is missing" was rejected: the first
  boot recorded after the cutover would make every earlier one disappear.
* Set `SERVEDECK_LEGACY_STATE_DIR=""` to turn the compatibility read off on a
  machine that never ran the fork. Point it elsewhere to read a different
  directory.


## 1. Check the config describes this box

`~/Projects/servedeck/servedeck.toml` already exists and is gitignored. Confirm
each `[backends.<name>]` block still matches reality — in particular that
`launcher`, `port` and `venv` are right, since the 27B moved to `8004` (`:8000`
is permanently held by `ats-optimizer.service`) and Flash-Next is on `:8001`.

```bash
cd ~/Projects/servedeck
grep -nE '^\[backends|^launcher|^port|^venv|^cwd' servedeck.toml
```

Two keys are new in this branch and worth adding:

```toml
[backends.inline]
# The launcher lives under bin/, so the default working directory (the
# launcher's own directory) is not the tree it belongs to. Precautionary
# rather than urgent: qwen-server-run.sh derives its root from
# ${BASH_SOURCE[0]}/.. and uses no relative paths, so this changes nothing
# today — it makes the launched process's cwd name the tree it came from.
cwd = "~/Projects/local_llm"

[backends.glm53]
# Carried over verbatim from the fork's history.py. Its own comment records a
# MEASURED 330-420 s cold boot (181 GiB load + Marlin repack + MTP graph
# capture) while the envelope it shipped is 420-900 -- deliberately pessimistic
# on both ends, because the repack is silent for minutes and an ETA that
# expires mid-boot reads as a hang. Kept as it was rather than "corrected" to
# the measurement: it is an ETA envelope, not a measurement, and it is used
# only until this model has boot history of its own.
cold_boot_range_s = [420.0, 900.0]
```

`EXTRA_ARGS` needs no config: Servedeck now reads it from
`~/Projects/local_llm/.config` and passes it through, but **only when that
file's `BACKEND` matches the backend being launched** — the same string `llm
start` exports.

## 2. Build the venv

```bash
cd ~/Projects/servedeck
./setup.sh                  # creates .venv and installs requirements.txt
.venv/bin/python -c "import servedeck.app; print('ok')"
```

## 3. Run it in the foreground first, with coldstart still stopped

```bash
~/Projects/coldstart/stop.sh          # frees :8010
cd ~/Projects/servedeck && ./run.sh   # Ctrl-C to stop
```

Open http://127.0.0.1:8010 and check three things that were broken before:

1. the **serving line names the model that is actually loaded** (it now comes
   from the live process's `--model` and from `/v1/models`, never from
   `.config`'s `BACKEND` header);
2. selecting a model whose backend differs from the last run and pressing Apply
   proposes **that backend's port**, not the previous one's;
3. starting onto a port something else holds is **refused up front**, instead
   of loading weights for ten minutes and then dying on "address already in
   use".

If anything looks wrong, Ctrl-C and go to Rollback. Nothing has been changed
that a restart of coldstart does not undo.

## 4. The run history carries over by itself

Since the rename, Servedeck reads `~/Projects/coldstart/state/` and merges it
with its own `state_dir` (see §0). Nothing has to be copied, and a first start
no longer has to be timed around a copy.

Copying the files anyway is still safe — the merge de-duplicates on record
content, so a boot present in both directories is counted once:

```bash
cp ~/Projects/coldstart/state/history.jsonl     ~/Projects/servedeck/state/
cp ~/Projects/coldstart/state/measurements.json ~/Projects/servedeck/state/
```

Verify the history is visible before switching the service over:

```bash
cd ~/Projects/servedeck
.venv/bin/python -c "from servedeck import history, registry; \
print(len(history.load_all()), 'boots,', len(registry.load_observations()), 'measurements')"
```

That must print a non-zero count. Zero means the compatibility read is off
(`SERVEDECK_LEGACY_STATE_DIR` set to empty) or `state_dir` points somewhere
unexpected — capacity predictions would silently revert to estimates, which is
a degradation nothing else reports.

## 5. Switch the systemd unit

The shipped unit assumes the checkout is at `%h/servedeck`. On this box it is
at `~/Projects/servedeck`, so the paths have to be rewritten — installing it
unedited gives `Command /home/<you>/servedeck/.venv/bin/uvicorn is not
executable: No such file or directory` (which `systemd-analyze verify` will
also tell you before you install it).

```bash
systemctl --user stop    coldstart.service
systemctl --user disable coldstart.service      # leaves the unit file in place

sed 's|%h/servedeck|%h/Projects/servedeck|g' \
    ~/Projects/servedeck/systemd/servedeck.service \
    > ~/.config/systemd/user/servedeck.service
systemd-analyze verify --user ~/.config/systemd/user/servedeck.service
systemctl --user daemon-reload
systemctl --user enable --now servedeck.service
systemctl --user status servedeck.service --no-pager
curl -sf http://127.0.0.1:8010/api/health && echo
```

`llm ui` still launches the coldstart tree by path: `cmd_ui` in
`~/Projects/local_llm/llm` hardcodes `$HOME/Projects/coldstart`, `.venv-gui`
and `coldstart.app:app`, and the `COLDSTART_PORT="8010"` near the top of that
file is what `llm status` probes. (Cited by name rather than line number: that
script is under active edit and the numbers move.) It is only a convenience
opener and the systemd unit makes it unnecessary, but if you want it to open
Servedeck, that is a three-line change in `cmd_ui` — a separate edit to a
script this branch deliberately does not touch.

## What the owner must change outside this repo

None of these files is touched by this branch — another workstream owns them,
and Servedeck keeps working as they are, on the deprecated aliases. This is the
list for whoever retires those aliases.

**`~/Projects/local_llm/.config`** (2 occurrences)

* line 15 — comment: "`COLDSTART_URL` is empty; Coldstart is optional…"
* line 16 — `COLDSTART_URL=""` -> `SERVEDECK_URL=""`

**`~/Projects/local_llm/codex-qwen.sh`** (16 occurrences)

* line 810 — `CONFIG_ALLOWED_KEYS="… COLDSTART_URL USE_COLDSTART"`; this is the
  allow list that rejects a `SERVEDECK_URL` write today, so it has to change
  first or nothing else can.
* lines 163-164, 177 — the `COLDSTART_URL` / `USE_COLDSTART` defaults and
  `recompute_derived()`'s `BASE_URL`
* lines 450-459, 677-680 — the delegate-to-the-dashboard branches in
  `start_server()` / `stop_server()`
* lines 68, 152, 156, 166, 446-447, 667 — comments naming the same keys

**`~/Projects/local_llm/llm`** (8 occurrences) — this one is not an alias;
it launches the OLD TREE by path and will keep doing so until it is edited:

* line 24 — `COLDSTART_PORT="8010"`
* line 603 — `coldstart.app:app` out of `$HOME/Projects/coldstart` with
  `.venv-gui`, logging to `$LOG_DIR/coldstart.log`
* lines 545-547, 584, 600, 606-607 — `llm status` / `llm up` probes of that port

Until line 603 is changed, `llm ui` starts the coldstart fork on `:8010`, which
will collide with a running Servedeck (`PORT_IN_USE`, refused up front) rather
than replace it.

## What changes for you

* **The model name in the header is the model that is running.** It is read
  from the process and cross-checked against `/v1/models`. When the two
  disagree — a `--served-model-name` reused from an earlier run — the
  dashboard says so instead of picking one.
* **Apply starts the model you selected**, on the port that model's backend
  declares, under a name derived from that model rather than the previous
  one's.
* **A start that cannot work is refused before it starts**, with the reason:
  port already in use, training marker present, venv or launcher missing.
* **Launches match `llm start`.** Same launcher, same working directory, same
  environment, `EXTRA_ARGS` included.
* Everything about this machine is in `servedeck.toml`. Adding a backend is a
  `[backends.<name>]` section, not a code change.

## Rollback

Servedeck writes only to its own `state_dir` and, through
`codex-qwen.sh`, to `~/Projects/local_llm/.config` — the same file coldstart
wrote, through the same script. There is nothing to undo but the service:

```bash
systemctl --user disable --now servedeck.service
systemctl --user enable  --now coldstart.service
curl -sf http://127.0.0.1:8010/api/health && echo
```

The coldstart tree is unmodified by this migration, including its `state/`:
Servedeck reads that directory and never writes to it, so a rollback finds
the fork's own history exactly as it left it. Boots recorded by Servedeck in
the meantime stay in Servedeck's `state_dir` and are not visible to the fork —
that gap is the only thing a rollback loses.

## After the cutover

`~/Projects/coldstart` becomes dead weight — but it is also the only copy of
its own `state/` and its `.venv-gui`, and it is not under version control, so
deleting it is not reversible. Recommendation: **keep it, stopped and
disabled, until Servedeck has served for a week and a cold boot has been
measured through it**; then delete. Do not delete on the day of the cutover.

**Before deleting it, copy its `state/` across.** Servedeck reads that
directory live (§0); deleting the tree takes 47 boot records and 7 KV
measurements with it, and the loss is silent — every model simply reverts to
"estimated". The copy is de-duplicated, so it is safe to run at any time:

```bash
cp ~/Projects/coldstart/state/history.jsonl     ~/Projects/servedeck/state/
cp ~/Projects/coldstart/state/measurements.json ~/Projects/servedeck/state/
export SERVEDECK_LEGACY_STATE_DIR=""   # then the tree is genuinely unreferenced
```

Retiring the deprecated `COLDSTART_*` aliases is a separate, later step, and
it is gated on the three files listed under "What the owner must change
outside this repo" — not on deleting the tree.
