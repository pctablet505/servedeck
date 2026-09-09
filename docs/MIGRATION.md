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

## 4. Carry the run history over

Servedeck's `state_dir` in `servedeck.toml` is `./state`, which already holds
the fork's `measurements.json` and `history.jsonl`. If you want the newest
coldstart history instead (it has kept running since), copy it before the
first Servedeck start — never during one:

```bash
cp ~/Projects/coldstart/state/history.jsonl     ~/Projects/servedeck/state/
cp ~/Projects/coldstart/state/measurements.json ~/Projects/servedeck/state/
```

Losing these is not fatal: capacity predictions fall back to estimates and
re-learn on the next boot. Losing them silently mid-run is worse, which is why
this is a stop-then-copy step.

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

The coldstart tree is unmodified by this migration, including its `state/`.

## After the cutover

`~/Projects/coldstart` becomes dead weight — but it is also the only copy of
its own `state/` and its `.venv-gui`, and it is not under version control, so
deleting it is not reversible. Recommendation: **keep it, stopped and
disabled, until Servedeck has served for a week and a cold boot has been
measured through it**; then delete. Do not delete on the day of the cutover.
