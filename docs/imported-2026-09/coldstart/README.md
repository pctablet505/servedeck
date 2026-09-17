# Coldstart

A local web GUI + API that manages a vLLM server on one RTX PRO 6000
(97,887 MiB) — start/stop/restart, live capacity estimation, boot-phase
progress, and a transparent proxy gateway for Codex CLI. Runs entirely on
`127.0.0.1`, no external network, no CDN, no auth (single-machine, v1).

**`SPEC.md` in this directory is the authoritative spec.** Every number in
it was measured from real boot logs — this README describes how the pieces
fit together and how to run things; it does not restate the spec.

## Quickstart

```bash
./setup.sh      # create/sync the venv, install pinned deps, install (but
                 # NOT enable) the systemd unit
./run.sh         # foreground dev server on http://127.0.0.1:8010
```

or, once you're ready to run it as a background service:

```bash
systemctl --user enable --now coldstart.service
journalctl --user -u coldstart -f
```

Run the test suite:

```bash
.venv-gui/bin/python -m pytest
```

## What this project will never do

These are absolute, not defaults you can override with a flag:

- **Never binds or touches port 8000/8001** — those are the sibling
  projects' vLLM servers, one of which is very likely live on this box
  right now. Coldstart only ever *proxies to* and *supervises* an existing
  server there; it never starts a model load itself except through the
  existing `serve.sh` / `qwen-server-run.sh` launchers, and never invents
  its own `vllm serve` command line (SPEC.md §1, "Delegation, not
  reimplementation").
- **Never installs into `.venv-llm` or `.venv-next`** — only into this
  project's own `.venv-gui` (SETUP.md:229's documented foot-gun).
- **Never runs `pgrep -f` / `pkill -f` / `pkill` / `killall`, anywhere** —
  that family of tools matches a process's full command line, including
  the calling shell's own, and has killed shells during this project's
  development. `coldstart/procctl.py` enumerates `/proc` directly instead;
  `tests/test_procctl_no_pattern_kill.py` greps the whole package for those
  names on every test run and fails the build on any hit.
- **Never runs `sudo`** — where a privileged action is genuinely needed
  (e.g. relaxing `ptrace_scope` for Flash-Next's PLE handoff), Coldstart
  surfaces a copyable command for a human to run, never runs it itself.
- **Never edits `local_llm/.config` directly** — every write goes through
  `codex-qwen.sh set-mem|set-subagents|set-config` as a subprocess
  (`coldstart/shellconfig.py`), preserving that script's own
  line-rewriting and validation logic.

## Process model

One `uvicorn` process, bound to `127.0.0.1:8010`, serves the GUI, the
`/api/*` control surface, and a transparent proxy to whichever upstream
vLLM port (`8001` flashnext / `8000` inline) is actually configured. The
vLLM server itself is launched with `start_new_session=True` (its own
session + process group), so it survives Coldstart being restarted;
Coldstart re-adopts it on its own startup by combining
`state/server.json` with a live port probe (SPEC.md §2 rule 4, §6
"Startup reconciliation").

## Repo layout (files this task owns)

| path | role |
|---|---|
| `coldstart/paths.py` | every absolute filesystem path Coldstart touches, as a module constant — no path literal appears anywhere else in the package |
| `coldstart/procctl.py` | process discovery/launch/stop, `/proc`-only, never a pattern-matching lookup tool (SPEC.md §2) |
| `coldstart/shellconfig.py` | the only writer of `local_llm/.config`, always via `codex-qwen.sh` subcommands |
| `coldstart/gpu.py` | `nvidia-smi` wrappers (`gpu_summary`, `compute_apps`, `gpu_alive`) and Xid classification from the kernel journal |
| `tests/test_procctl_no_pattern_kill.py` | the self-match-trap regression test |
| `setup.sh` / `run.sh` | environment setup / dev-mode launcher |
| `requirements.txt` | pinned runtime deps (`fastapi`, `uvicorn`, `httpx`; `pytest` unpinned) |
| `systemd/coldstart.service` | the `--user` unit `setup.sh` installs (but does not enable) |

Other modules under `coldstart/` (`capacity.py`, `phases.py`, `registry.py`,
`logtail.py`, and whatever assembles the FastAPI app / gateway / supervisor)
are owned by other agents working this same spec concurrently — this file
only documents the pieces above.

**Assumption flagged:** `run.sh` and `systemd/coldstart.service` both
invoke `coldstart.app:app` as the FastAPI application object. SPEC.md
names every `/api/*` endpoint but never states which module assembles them
into one importable `app` — that wiring belongs to whichever agent builds
the API layer. If it lands somewhere else, update the `ExecStart` /
`uvicorn` invocation in both files to match.

## `procctl.py` — the self-match trap, and how it's avoided

`SETUP.md:401` records a real incident: pattern-matching process lookups
(the family this project never uses — see above) match a process's full
command line, including the calling shell's own, and repeatedly killed
shells during development. `procctl.py` instead:

1. Enumerates `/proc/<pid>/{comm,cmdline,status,cwd,exe}` directly.
2. Uses `ss -H -ltnp` (a socket-table lookup, not a process-table pattern
   match) to find whatever holds a given port.
3. Classifies orphaned `VLLM::EngineCore`/`VLLM::Worker` children by
   `comm.startswith("VLLM::")` (the kernel truncates `comm` at 15 bytes,
   so `VLLM::EngineCore` reads back as `VLLM::EngineCor` — the prefix
   match is the correct predicate) — never by a cmdline pattern match on
   the candidate itself. An *ancestor's* cmdline is read only to decide
   orphan-hood (does it contain the literal 9 characters `vllm serve`?),
   never handed to any pattern-scanning tool.
4. Launches every server with `start_new_session=True`, so `stop()` can
   always signal a whole, isolated process group (`os.killpg`) — and
   refuses outright, before signalling anything, if that group ever turns
   out to be Coldstart's own.

### A real bug this surfaced, and the fix

Building the verification test for rule 2/3
(`launch()` really does give a child its own process group, and `stop()`
really can bring it down) found that `stop()` could time out on an
**already-dead** process. Root cause: `launch()`'s `Popen` object is never
retained, and nothing ever reaps the child — so once it exits, it becomes
a zombie that keeps existing in the process table until *something* calls
`wait()` on it. `os.killpg(pgid, 0)` (the liveness probe `stop()` polls)
reports a zombie as "alive" indefinitely, since the pid is still valid —
so `stop()` would run its full SIGTERM → wait → SIGKILL → wait cycle
against an already-exited process and still report failure at the end.

The fix (`procctl._pgid_alive`): attempt a non-blocking
`os.waitpid(pgid, os.WNOHANG)` first. Since the launched process's pid
*is* its own pgid (`start_new_session=True` makes pid == pgid == sid),
this reaps the leader the instant it exits, for as long as this same
Coldstart process is still its actual parent. A `ChildProcessError` (this
handle was re-adopted from a *previous* Coldstart process's
`state/server.json`, so we're not its parent) falls back to the original
`killpg`-based probe, which is the correct check for a pid we didn't fork.
`tests/test_procctl_no_pattern_kill.py::test_launch_gets_its_own_session_and_process_group`
exercises this against a real (harmless) child process and would have
caught the regression (it originally failed with a 15-second timeout
before the fix; it now passes in well under a second).

## `gpu.py` — Xid classification

`xid_events(since)` shells out to
`journalctl -k --no-pager --since <since>`, filters lines containing
`NVRM: Xid`, and classifies each by the numeric Xid code using **the exact
case statement in `bin/qwen-server-record-death.sh:54-69`**, read from
that file rather than re-derived:

| code | classification | restartable | meaning |
|---|---|---|---|
| 13, 31 | `mmu_fault` | yes | GPU MMU fault — the known, documented, MTP-correlated crash |
| 79 | `off_bus` | no | GPU has fallen off the bus — hardware/driver-level |
| 154 | `reboot_required` | no | driver GPU-recovery action asserted — often "Node Reboot Required" |
| anything else | `unrecognized` | no | not a code this script recognizes — do not assume it's the known pattern |

`gpu_alive()` backs SPEC.md §6's rule "`nvidia-smi -L` failing => never
restart" and §3's `GPU_UNRESPONSIVE` finding — every function in this
module returns an empty/`None` result on failure rather than raising, so a
flaky/absent `nvidia-smi` or an empty journal window never crashes a
caller mid-render.

## `shellconfig.py` — the only writer of `.config`

`local_llm/.config` holds `KEY="value"` lines written exclusively by
`codex-qwen.sh save_config_kv()` — editing it directly would bypass that
function's line-preserving rewrite and the script's own validators.
`shellconfig.py` never opens it for writing; every mutation is a
subprocess call into `codex-qwen.sh set-mem|set-subagents|set-config`.

The one hard rule (SPEC.md §1): `set-mem` **auto-restarts the server** if
one is already up. Coldstart's own restart sequence is always
*stop → write config → start*, owned by the supervisor — so
`set_util()` refuses outright with `ServerRunningError` if the server
answers as up, rather than ever letting `codex-qwen.sh`'s implicit restart
race Coldstart's own. By default it probes for itself
(`GET {BASE_URL}/models == 200`, the same criterion `codex-qwen.sh`'s own
`is_server_up()` uses); a caller that already knows its own `actual_state`
authoritatively (the supervisor, mid-sequence) may pass `server_up`
explicitly to skip the redundant probe.

## Gaps / things not implemented as written

Nothing assigned to this task was left unimplemented. Two things worth
flagging explicitly:

1. **The FastAPI app module name is an assumption**, not something SPEC.md
   states (see above) — `coldstart.app:app`. Fix `run.sh` /
   `systemd/coldstart.service` if it lands elsewhere.
2. **SPEC.md §9's shell patches are out of scope for this task's file
   list** and were not applied by `setup.sh` — those live in
   `codex-qwen.sh` / `serve.sh` / `qwen-server-run.sh`, none of which this
   task owns. Live inspection at setup time found §9(d) (the
   `BACKEND != flashnext` systemd gate) and the corrected retry values
   from the spec's own C1 correction already present in `codex-qwen.sh`;
   the rest of §9 was not checked further here.
