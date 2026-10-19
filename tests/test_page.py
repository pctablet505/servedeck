"""The page as served: v1's dashboard files, self-contained, on v2's routes.

The page is the owner's (restored 2026-09-17 from tag pre-cutover-2026-09-17);
its script is browser-only (async/await, canvas), so these tests check what
can be checked without a browser: the files exist and reference each other,
nothing is fetched from off the box, and every route the script calls is a
route the app registers.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from servedeck import app as _app

WEB = Path(_app.__file__).parent / "web"
INDEX = (WEB / "index.html").read_text(encoding="utf-8")
APP_JS = (WEB / "app.js").read_text(encoding="utf-8")

#: Parses the page script without running it (``new Function`` compiles its
#: body; the two lines it wraps around the source are taken off the line).
_PARSE_ONLY = (
    "const [, b] = imports.gi.GLib.file_get_contents(ARGV[0]);"
    "try { new Function(new TextDecoder().decode(b)); }"
    "catch (e) { print(`${e.name}: ${e.message} (line ${e.lineNumber - 2})`);"
    " imports.system.exit(1); }"
)


@pytest.mark.skipif(shutil.which("gjs") is None, reason="needs gjs, GNOME's SpiderMonkey shell")
def test_the_page_script_parses() -> None:
    """A script that does not parse runs none of itself. On 2026-10-19 a second
    ``const stopBtn`` in paintState left the live page at "connecting..." with
    an empty model list, and every other test here passed, because they read
    app.js as text."""
    r = subprocess.run(
        ["gjs", "-c", _PARSE_ONLY, str(WEB / "app.js")],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stdout + r.stderr


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


def test_the_allocator_keeps_its_four_controls() -> None:
    ids = set(re.findall(r'id="([A-Za-z0-9_]+)"', INDEX))
    for present in ("util", "ctx", "agents", "useRec", "agentsRec", "offload", "offloadNote",
                    "dBadge", "vramTxt", "segW", "apply", "stop", "dirty"):
        assert present in ids, present
    for gone in ("smoke", "mRun", "mWait", "recN", "recMath", "mixTbl", "recCal", "dKv", "dKvTok"):
        assert gone not in ids, gone
    assert INDEX.count('class="acell"') == 4
    render_ctx = APP_JS.split("function renderCtx")[1].split("\nfunction ")[0]
    assert "ctxLabel(" not in render_ctx, "the context bound must print the exact figure"


def test_every_id_the_script_paints_exists_or_is_retired() -> None:
    """Retired ids are painted through set(), which no-ops on a missing element."""
    html = set(re.findall(r'id="([A-Za-z0-9_]+)"', INDEX))
    painted = set(re.findall(r'\$\("([A-Za-z0-9_]+)"\)', APP_JS)) | set(re.findall(r'set\("([A-Za-z0-9_]+)"', APP_JS))
    retired = {"adoptBtn", "dKv", "dKvTok", "mRun", "mWait", "mixFull", "mixNote", "mixTbl",
               "recBasis", "recCal", "recMath", "recN", "recP99", "smoke"}
    assert painted - html <= retired, sorted(painted - html - retired)


def test_the_page_reads_top_down_by_question() -> None:
    """2026-09-17 redesign: vital signs first (what serves, how fast, how full),
    then models beside the controls, then diagnostics and reference."""
    order = [INDEX.index(k) for k in ('class="panel hero"', 'class="grid"', 'id="trafficPanel"', 'id="logPanel"')]
    assert order == sorted(order)
    hero = INDEX.split('class="panel hero"')[1].split('class="grid"')[0]
    for kpi in ("thDecode", "thPrefill", "thTtft", "kvPct", "mWaitCap", "mPre", "sModel", "queued"):
        assert f'id="{kpi}"' in hero, kpi
    assert hero.count('class="kpi') >= 6
    grid = INDEX.split('class="grid"')[1].split('id="trafficPanel"')[0]
    left, right = grid.split('<div class="col">')[1:3]
    assert 'id="mlist"' in left and 'id="gpuName"' in left and 'id="hostRam"' in left
    assert 'class="panel notes"' in left, "the figures note fills the sidebar, not a footer"
    assert 'class="alloc"' in right and 'id="reqHist"' in right and 'id="oversub"' in right
    traffic = INDEX.split('id="trafficPanel"')[1].split('id="logPanel"')[0]
    assert 'id="tkIn"' in traffic and 'id="hitRate"' in traffic
    header = INDEX.split('<header class="top">')[1].split("</header>")[0]
    assert 'id="gpuUsed"' in header and 'id="gpuName"' not in header


def test_the_nesting_is_the_intended_tree() -> None:
    """A balanced div count is not enough (2026-09-17 twice): walk the tree."""
    depth = 0
    at = {}
    for line in INDEX.split("\n"):
        for key in ('class="panel hero"', 'class="grid"', 'id="trafficPanel"', 'id="logPanel"',
                    'class="alloc"', 'class="vram"', 'class="col"'):
            if key in line and key not in at:
                at[key] = depth
        depth += len(re.findall(r"<div\b", line)) - line.count("</div>")
        depth += len(re.findall(r"<section\b", line)) - line.count("</section>")
    assert depth == 0
    assert at['class="panel hero"'] == 1 and at['class="grid"'] == 1
    assert at['id="trafficPanel"'] == 1 and at['id="logPanel"'] == 1
    assert at['class="col"'] == 2
    assert at['class="vram"'] == at['class="alloc"'] + 1



def test_an_error_notice_unfolds_the_log() -> None:
    """A refusal written only into a folded log is a silent failure."""
    handler = APP_JS.split('addEventListener("notice"')[1].split("});")[0]
    assert 'n.level === "error"' in handler and "logFold" in handler and ".click()" in handler
    assert "n.replay" in handler, "replayed history on page load must not pop the log open"


#: Runs app.js's own paintTargetBanner in gjs against stub elements, and prints
#: the Apply button's label.
_PAINT_TARGET_BANNER = r"""
const [, b] = imports.gi.GLib.file_get_contents(ARGV[0]);
const src = new TextDecoder().decode(b);
const start = src.indexOf("function paintTargetBanner()");
eval(src.slice(start, src.indexOf("\n}\n", start) + 2));
const el = () => ({ textContent: "", className: "", title: "", style: {}, disabled: false });
const els = { stagedPill: el(), cfgModelName: el(), cfgStateTag: el(), cfgBannerSub: el(),
              revertActiveBtn: el(), apply: el() };
els.apply.textContent = "Apply & restart";  // index.html's label
els.apply.disabled = true;                   // the first paint, before /api/state
els.apply.title = "busy: STARTING";          // paintState's reason for disabling
var $ = (id) => els[id];
var sel = 0;
var MODELS = JSON.parse(ARGV[1]);
paintTargetBanner();
print(JSON.stringify([els.apply.textContent, els.apply.title]));
"""


@pytest.mark.skipif(shutil.which("gjs") is None, reason="needs gjs, GNOME's SpiderMonkey shell")
@pytest.mark.parametrize(
    "models, label",
    [
        ([{"name": "Alpha", "serving": False}], "Start Alpha"),
        ([{"name": "Alpha", "serving": False}, {"name": "Beta", "serving": True}], "Switch to Alpha"),
        ([{"name": "Alpha", "serving": True}], "Apply & restart"),
    ],
)
def test_the_apply_button_names_its_action_even_while_disabled(models, label) -> None:
    """The button is disabled until /api/state arrives, and with nothing
    serving the banner is not repainted after that, so a label set only on an
    enabled button stayed index.html's "Apply & restart" while the banner said
    the click would START the model (seen on the live page 2026-10-19). The
    title stays paintState's: it says why the button is disabled."""
    import json

    r = subprocess.run(
        ["gjs", "-c", _PAINT_TARGET_BANNER, str(WEB / "app.js"), json.dumps(models)],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout) == [label, "busy: STARTING"]
