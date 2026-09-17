# uitour — end-to-end PyQt UI driver (preserved 2026-08-22 from /tmp)

The decision doc (`ui_verify_DECISION_2026-08-22.md`) flagged that the driver
lived in /tmp and would not survive a reboot. These copies are the durable
master; a repo move (`scripts/uitour/`) is the suggested follow-up.

Files:
- `uitour_driver.py` — boots the real `run_app()` offscreen (only
  `QApplication.exec` replaced), drives 21 user journeys through all 24
  pages, auto-records + safely answers every modal, logs JSONL.
  Phases: `A` shell + 24-page tour, `B` core journeys (viewer/backtester/
  screener), `C` the 21 journeys (J4–J21 incl. live-sim order placement,
  auto-sim real run, CLI terminal real command, settings save, database
  stats). Usage: `python uitour_driver.py [AB|ABC|C]`.
- `chart_probe.py` — WebEngine render proof: boots the app WITH WebEngine,
  loads RELIANCE via the user path, reads the Plotly DOM back via
  `runJavaScript` (graph div + SVG path count). Verdict in
  `chart_probe.json` (RENDERED: 1 graph div, 505 SVG paths).
- `steps.jsonl` — final clean pass (driver v4, 2026-08-22 03:xx): all 21
  journeys green, EXIT:0, zero uncaught exceptions.

Run requirements:
```bash
cd <AlgoTrading worktree>
export PATH="/home/pctablet505/Projects/AlgoTrading/.venv/bin:$PATH"
export ALGOTRADING_WORKDIR=/home/pctablet505/Projects/AlgoTradingWorkingDir
python <this-dir>/uitour_driver.py C
```

Gotchas that cost real time (do not re-learn):
1. **Import shadow:** the driver force-inserts its worktree onto sys.path and
   records `IMPORT_CHECK` — verify it points at the tree under test before
   trusting a single record (a /tmp script once silently tested the dead
   canonical checkout).
2. **Killing the driver:** `pgrep -f 'uitour/driver[.]py'` then kill by PID.
   Never pkill a pattern that matches your own command line.
3. `DataFrameTableWidget` emits `row_selected` on **click**, not
   programmatic `selectRow` — simulate via `table._table.clicked.emit(
   model.index(0, 0))`.
4. The CLI terminal prompt is `algotrading >`: type the subcommand only
   (the page prefixes the executable).
5. Long runs: launch detached (`setsid nohup bash -c '... ; echo EXIT:$? >
   marker.done'`) or the exec session teardown kills the subshell.
