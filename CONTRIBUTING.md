# Contributing

## Ground rules

**Never use `pgrep -f` or `pkill -f`.** They match the calling process's own
command line. A test (`tests/test_hygiene.py`) AST-scans the
package and test suite and fails on any occurrence. Use `ps -eo comm`, explicit PIDs, or
unit cgroup inspection.

**Never hardcode a path, GPU size, or model name.** Everything
machine-specific belongs in `models.toml`, `servedeck/settings.py`, or `servedeck/limits.py`.

**Never run `sudo`.** Where privilege is needed, emit a copyable command for
the user to run.

**Every fix ships a test that fails before it.** Two of the bugs fixed in the
first review had been silently broken for the life of the module because
nothing exercised them.

## Numbers

Anything presented as measured must come from a real log or a live probe.
If it's derived, label it estimated. The UI distinguishes them because
predictions here have run optimistic before.

## Tests

```bash
.venv/bin/python -m pytest
```

Fixtures under `tests/fixtures/` are snapshots. Never point a test at a live
log: the server rewrites those on restart, and the test breaks for reasons
unrelated to the change.

## Quick reference (moved from the README)

These sections were on the front page until October 2026; they are kept here word for word.

### The five commands

```bash
servedeck status              # what is live: slot, ctx, KV %, run/wait, tok/s, uptime, headroom
servedeck switch flashnext    # put this model in the main slot, evicting whatever is there
servedeck start lfm2          # start (or stop) one model, streaming its boot log until READY
servedeck doctor              # prove the registry against reality before believing anything else
servedeck wire --apply        # rewrite the client configs (no flag = dry-run diff)
```

`start`, `stop`, `switch`, `log` and `adopt` drive the running dashboard over HTTP and fall back to the same
supervisor in-process when it is down, saying which on the first line. `models`, `adopt`, `log` and `smoke` also
exist. None of the verbs take tuning flags: `switch flashnext` launches at the `util` and context that
`models.toml` pins. To launch at other values, use the page's **Configure** panel.

The control plane is a **user** unit: `systemctl --user start servedeck` (`restart` after changing code or
`models.toml`), never `sudo`. Restarting it does not disturb a running model: engines are separate units and this
one sets `KillMode=process`.

### Logs

```bash
journalctl --user -u servedeck -f          # servedeck itself, including its own logger
journalctl --user -u model-flashnext -f    # one model's engine (unit is model-<key>)
servedeck log flashnext -n 200             # the same journal, through the CLI
```

### When something is wrong

Run `servedeck doctor`. Besides the registry it checks the host rows that explain most surprises:
`kernel.yama.ptrace_scope` (must be 0, or the KV-offload reaper cannot see a sibling engine and Flash-Next's PLE
handoff fails), desired state naming a model that is not running, weights on disk, leaked
`/dev/shm/vllm_offload_*.mmap` buffers at 40 GiB each, and the GPU power cap. Playbooks:
[Recovery](https://servedeck.rlalphalabs.com/operations/#recovery).

### Run the tests

`uv venv .venv && uv pip install -e '.[dev]'`, then the everyday run, which touches nothing on the box:

```bash
.venv/bin/python -m pytest -q -n 12 \
  --ignore=tests/test_e2e_real.py --ignore=tests/test_app_e2e.py \
  --ignore=tests/test_control_e2e.py --ignore=tests/test_gateway_e2e.py
```

The four excluded files are the gate. Three drive the real `systemd-run` path, safely, under the `sd-test-` unit
namespace, which cannot name a real `model-*` unit; they bind fixed ports, so run them one at a time (`-n 0`).
`test_e2e_real.py` boots real models on the GPU and needs the card to have room. Run the gate on an idle card, not
as part of a normal check. How to contribute: [CONTRIBUTING.md](CONTRIBUTING.md).

| To | Read |
| --- | --- |
| Use it | [Operations](https://servedeck.rlalphalabs.com/operations/) · [Clients](https://servedeck.rlalphalabs.com/clients/) · [Configuration](https://servedeck.rlalphalabs.com/configuration/) |
| How it is built | [Architecture](https://servedeck.rlalphalabs.com/architecture/) · [Builds](https://servedeck.rlalphalabs.com/builds/) · [Host](https://servedeck.rlalphalabs.com/host/) · [Decisions](https://servedeck.rlalphalabs.com/decisions/) |
| Each model, measured | [Flash-Next](https://servedeck.rlalphalabs.com/models/flashnext/) · [Qwen3.8-27B](https://servedeck.rlalphalabs.com/models/qwen27b/) · [GLM-5.3](https://servedeck.rlalphalabs.com/models/glm53/) · [LFM2.5-350M](https://servedeck.rlalphalabs.com/models/lfm2/) |

## Review & Verification Gate

Every pull request clears three sequential stages before being marked ready for review:

1. **Deterministic gate:**
   - CI workflow `.github/workflows/ci.yml` passes: editable install inside repo-local `.venv` (ensuring `tests/test_cli_control.py` executes all 16 console-script tests instead of skipping), `.venv/bin/python -m pytest -q -rs -n auto --maxprocesses=12 -p no:cacheprovider` excluding the 4 GPU/systemd e2e modules (covering all 35 non-e2e test modules including `tests/test_cli_control.py`, `tests/test_hygiene.py`, and `tests/test_site.py`), plus out-of-tree `.venv/bin/python -m mkdocs build --strict -d <out-dir>` (preventing generated HTML under `site/` from tripping `tests/test_hygiene.py`).
   - Behavioral changes include an executable `fail-before` proof (test fails against the base tree or reverted code) or `mutant-kill` proof (a one-line defect in a throwaway copy causes assertion exit `1`).
   - GitHub Actions workflows stay pinned to exact 40-character commit SHAs with minimal `permissions` and `persist-credentials: false` strictly on `actions/checkout`.
2. **Pass-1 author-blind diff audit:**
   - Start a fresh reviewer with zero authoring history and **without** reading PR rationalization prose first.
   - Read the diff and referenced callers directly, emit a **2-sentence paraphrase** of what the diff changes (catching unspoken assumptions early), audit correctness, systemd/process control safety, concurrency, packaging/hygiene invariants, and docs, and record every unrun check (e.g., GPU e2e gate or missing local `mkdocs`) under `COULD NOT VERIFY`.
   - Severity labels: **`Blocker:`** (correctness, safety, security, test-gate weakening, or ledger counter-shape — must fix), **`Warning:`** (high-risk defect or missing edge-case coverage — must fix or clear via Pass-2 proof), **`Nit:`** (non-blocking polish).
3. **Pass-2 proof-backed verification:**
   - No candidate `Blocker:` or `Warning:` finding may be dismissed with prose alone. Every disputed finding maps strictly to `VERIFIED` (must fix or escalate) or `DISPUTED_WITH_PROOF` backed by one of: (a) an executable counter-probe against the exact trigger input, (b) a caller-complete invariant proof quoting exact `file:line` guards across the repo, (c) a matching row in the `<base>` branch's `### Negative-Rule Ledger` below (never from self-edited PR lines), or (d) an exact verbatim documentation clause for non-executable `.md` prose.

### Negative-Rule Ledger

| Allowed Exact Shape (symbol/invariant) | Still-A-Bug Counter-Shape (never dismiss) |
| :--- | :--- |
| `systemctl --user` (for manual operator lifecycle of persistent `servedeck.service`), `servedeck/units.py` using `systemctl --user` / `systemd-run --user` / `journalctl --user` **exclusively** for transient `model-*` / `sd-test-*` user units (explicitly refusing `servedeck` and all other names via `_NAME_RE`), and read-only kernel Xid polling via `journalctl -k -b all --no-pager` (`servedeck/gpu.py`, `servedeck/xid_watch.py`). | Any invocation of `sudo`, `su`, or system-wide `systemctl` / `systemd-run` (without `--user`), widening `servedeck/units.py` `_NAME_RE` beyond `model-*` / `sd-test-*` (e.g., permitting `servedeck` or other persistent units), or wildcard unit stop globs such as `systemctl --user stop 'sd-test-*'` (`Blocker:`). |
| `sd-test-` transient unit prefix in isolated systemd e2e tests (`test_app_e2e.py`, `test_control_e2e.py`, `test_e2e_real.py`), plus the read-only `sd-netns-probe-<pid>` helper in `tests/conftest.py` (kept outside `sd-test-*` so e2e teardown sweeps do not race or stop it). | Starting, stopping, or mutating any live `model-*` unit or non-test production user unit (`servedeck`, `lfm2-350m`, `qwen-vllm`, `*-reasoning-proxy`) from automated tests (`Blocker:`). |
| Excluding the 4 fixed-port/GPU e2e modules (`test_e2e_real.py`, `test_app_e2e.py`, `test_control_e2e.py`, `test_gateway_e2e.py`) from the everyday `ci.yml` runner, and host-hardware/cache-gated individual test skips surfaced transparently by `-rs` (GPU/HF cache/multi-filesystem checks). | Weakening, causing module-/dependency-level skips (e.g., omitting `.venv` so `tests/test_cli_control.py` silently skips, or omitting `mkdocs` so `TestTheBuiltSite` in `tests/test_site.py` skips), or excluding any of the 35 non-e2e unit/hygiene/site test modules (`Blocker:`). |
