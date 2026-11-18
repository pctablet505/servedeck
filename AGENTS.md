# Servedeck — Agent & Contributor Card

Self-contained working guide for automated coding and review agents operating on this public repository. Full ground rules, test discipline, and the two-pass review gate live in [CONTRIBUTING.md](CONTRIBUTING.md).

## Key Commands

*(Create a repo-local `.venv` via `python3 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"` so `tests/test_cli_control.py` executes all 16 console-script tests instead of skipping.)*

- **Everyday non-e2e test suite (35 modules including `tests/test_cli_control.py`, `tests/test_hygiene.py`, and `tests/test_site.py`; caps parallel workers at `12`):**
  ```bash
  .venv/bin/python -m pytest -q -rs -n auto --maxprocesses=12 -p no:cacheprovider \
    --ignore=tests/test_e2e_real.py \
    --ignore=tests/test_app_e2e.py \
    --ignore=tests/test_control_e2e.py \
    --ignore=tests/test_gateway_e2e.py
  ```
- **Documentation site strict build (write output outside the repo tree so rendered HTML under `site/` does not trip `tests/test_hygiene.py` on subsequent local runs):**
  ```bash
  .venv/bin/python -m mkdocs build --strict -d <out-dir>
  ```
- **CI workflows:** `.github/workflows/ci.yml` runs on every `push` to `main` and every `pull_request` (`.venv` editable install, all 35 non-e2e modules with `-rs`, and out-of-tree `mkdocs build --strict`); `.github/workflows/pages.yml` builds and publishes the site with SHA-pinned GitHub Actions.

## Non-Negotiable Invariants

- **Never run `sudo`.** Control-plane operations use `systemctl --user`, `systemd-run --user` (managing transient `model-*` / `sd-test-*` user units in `servedeck/units.py`), and `journalctl --user`; read-only kernel Xid polling uses `journalctl -k -b all --no-pager` (`servedeck/gpu.py`, `servedeck/xid_watch.py`). Where host-level privilege is needed, emit a copyable command for the operator.
- **Never use `pgrep -f` or `pkill -f`.** Enforced by AST scan in `tests/test_hygiene.py`.
- **Keep transient e2e units strictly under the `sd-test-` namespace.** Never start, stop, or mutate live `model-*` units from automated tests (`test_app_e2e.py`, `test_control_e2e.py`, `test_e2e_real.py`).
- **No host-specific paths, machine names, or hardcoded GPU sizes** in code or docs (`models.toml`, `servedeck/settings.py`, and `servedeck/limits.py` own machine config). Keep documentation free of user-specific home paths and scratch temp paths.
- **Measured vs. estimated numbers:** Label any derived value explicitly as estimated; every measured figure must cite a real log or live probe.
- **Review & proof discipline:** Every behavioral fix ships a regression test seen to fail before the fix (`fail-before` / mutant proof). Follow the author-blind two-pass review gate and `### Negative-Rule Ledger` in [CONTRIBUTING.md](CONTRIBUTING.md).
- **Commit identity (maintainer / automated-agent commits):** Automated agent commits use sole repository owner identity with explicit staged paths and zero `Co-Authored-By:` or AI attribution trailers (external contributors retain their own authorship attribution).
