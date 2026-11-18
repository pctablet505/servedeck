# ServeDeck — Agent & Contributor Card

Self-contained working guide for automated coding and review agents operating on this public repository. Full ground rules, test discipline, and the two-pass review gate live in [CONTRIBUTING.md](CONTRIBUTING.md).

## Key Commands

- **Everyday non-e2e test suite (35 modules including `tests/test_hygiene.py` and `tests/test_site.py`; caps parallel workers at `12`):**
  ```bash
  python -m pytest -q -n auto --maxprocesses=12 -p no:cacheprovider \
    --ignore=tests/test_e2e_real.py \
    --ignore=tests/test_app_e2e.py \
    --ignore=tests/test_control_e2e.py \
    --ignore=tests/test_gateway_e2e.py
  ```
- **Documentation site strict build:**
  ```bash
  python -m mkdocs build --strict
  ```
- **CI workflows:** `.github/workflows/ci.yml` runs on every `push` to `main` and every `pull_request` (core non-e2e tests + `mkdocs build --strict`); `.github/workflows/pages.yml` builds and publishes the site with SHA-pinned GitHub Actions.

## Non-Negotiable Invariants

- **Never run `sudo`.** All control-plane operations use `systemctl --user` and `journalctl --user`. Where host-level privilege is needed, emit a copyable command for the operator.
- **Never use `pgrep -f` or `pkill -f`.** Enforced by AST scan in `tests/test_hygiene.py`.
- **Keep transient e2e units strictly under the `sd-test-` namespace.** Never start, stop, or mutate live `model-*` units from automated tests.
- **No host-specific paths, machine names, or hardcoded GPU sizes** in code or docs (`models.toml`, `servedeck/settings.py`, and `servedeck/limits.py` own machine config). Keep documentation free of user-specific home paths and scratch temp paths.
- **Measured vs. estimated numbers:** Label any derived value explicitly as estimated; every measured figure must cite a real log or live probe.
- **Review & proof discipline:** Every behavioral fix ships a regression test seen to fail before the fix (`fail-before` / mutant proof). Follow the author-blind two-pass review gate and `### Negative-Rule Ledger` in [CONTRIBUTING.md](CONTRIBUTING.md).
- **Commit identity:** Commit solely as the repository owner identity with explicit staged paths and zero `Co-Authored-By:` or AI attribution trailers.
