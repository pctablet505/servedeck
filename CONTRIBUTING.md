# Contributing

## Ground rules

**Never use `pgrep -f` or `pkill -f`.** They match the calling process's own
command line. A test (`tests/test_procctl_no_pattern_kill.py`) greps the
package and fails on any occurrence. Use `ps -eo comm`, explicit PIDs, or
`procctl`.

**Never hardcode a path, GPU size, or model name.** Everything
machine-specific belongs in `coldstart/config.py`.

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
log — the server rewrites those on restart, and the test breaks for reasons
unrelated to the change.
