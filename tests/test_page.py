"""The page as served: v1's dashboard files, self-contained, on v2's routes.

The page is the owner's (restored 2026-09-17 from tag pre-cutover-2026-09-17);
its script is browser-only (async/await, canvas), so these tests check what
can be checked without a browser: the files exist and reference each other,
nothing is fetched from off the box, and every route the script calls is a
route the app registers.
"""

from __future__ import annotations

import re
from pathlib import Path

from servedeck import app as _app

WEB = Path(_app.__file__).parent / "web"
INDEX = (WEB / "index.html").read_text(encoding="utf-8")
APP_JS = (WEB / "app.js").read_text(encoding="utf-8")


def test_the_page_references_only_files_beside_it() -> None:
    refs = re.findall(r'(?:href|src)="([^"]+)"', INDEX)
    assert refs == ["style.css", "app.js"], refs
    for ref in refs:
        assert (WEB / ref).is_file(), f"{ref} is referenced but missing"


def test_the_page_ships_no_external_reference() -> None:
    for text in (INDEX, APP_JS, (WEB / "style.css").read_text(encoding="utf-8")):
        assert "http://" not in text.replace("http://localhost", "").replace("http://127.0.0.1", "")
        assert "https://" not in text
        assert "@import" not in text


class _FakeControl:
    def live(self):
        return []

    def adopt(self, **_kw):
        return None

    def reconcile(self, *_a, **_kw):
        return None


def _app_for(tmp_path):
    from servedeck import models as _models
    from servedeck.settings import Settings

    models_path = tmp_path / "models.toml"
    models_path.write_text(
        '[gpu]\ntotal_mib = 100000\nmargin_mib = 1024\n'
        '[builds.stock]\nvenv = "/opt/v"\ncuda_home = "/opt/cuda"\n'
        '[models.a]\nid = "A"\nrepo = "org/A"\nslot = "main"\nport = 8001\nbuild = "stock"\nctx = 4096\n'
    )
    settings = Settings(listen_host="127.0.0.1", listen_port=8099, models_path=models_path,
                        state_dir=tmp_path / "state", unit_prefix="sd-test-")
    return _app.create_app(settings, registry=_models.load(models_path), control=_FakeControl(),
                           reconcile=False, poll=False)


def test_every_route_the_script_calls_exists(tmp_path) -> None:
    app = _app_for(tmp_path)
    paths = {getattr(r, "path", "") for r in app.routes}
    called = set(re.findall(r'"(/api/[a-z/_]+)"', APP_JS))
    assert called, "the script calls no API at all?"
    missing = sorted(p for p in called if p not in paths)
    assert not missing, f"the page calls routes the app does not have: {missing}"
    assert "/api/events" in paths


def test_the_scripts_state_words_are_the_ones_the_backend_emits() -> None:
    for word in ("STARTING", "STOPPING", "READY", "STOPPED"):
        assert word in APP_JS
    for phase in _app._legacy.PHASES:
        assert f"{phase}:" in APP_JS or f'"{phase}"' in APP_JS, f"the page has no label for phase {phase}"


def test_the_2026_09_17_ux_pass_holds() -> None:
    """The owner's list: the allocator is one row of three controls, the
    recommendation column and the dead Smoke button are gone, running and
    waiting are on the status chip and nowhere else."""
    ids = set(re.findall(r'id="([A-Za-z0-9_]+)"', INDEX))
    for present in ("util", "ctx", "agents", "useRec", "agentsRec", "dBadge", "dKv", "dKvTok",
                    "kvOffload", "qN", "qW", "mPre", "hitRate", "pctP90", "oversub", "apply", "stop"):
        assert present in ids, present
    for gone in ("smoke", "mRun", "mWait", "recN", "recMath", "mixTbl", "recCal"):
        assert gone not in ids, gone
    assert INDEX.count('class="acell"') == 3
    render_ctx = APP_JS.split("function renderCtx")[1].split("\nfunction ")[0]
    assert "ctxLabel(" not in render_ctx, "the context bound must print the exact figure"
