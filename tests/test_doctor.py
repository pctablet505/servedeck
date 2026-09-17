"""servedeck.doctor — unit tests with a stubbed http_get, plus two tests against
the box's own live, read-only :8007 (LFM2, always on) and :8010 (gateway) —
the only two ports the hard rules allow this test suite to touch.
"""

from __future__ import annotations

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx
import pytest

from servedeck import doctor, models, units

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resp(status: int, data: dict | None = None) -> httpx.Response:
    return httpx.Response(status_code=status, json=data if data is not None else {})


def _models_response(*ids: str) -> httpx.Response:
    return _resp(200, {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]})


def _stub_get(mapping: dict[str, httpx.Response | Exception]):
    def get(url: str, timeout: float) -> httpx.Response:
        result = mapping[url]
        if isinstance(result, Exception):
            raise result
        return result

    return get


# --------------------------------------------------------------------------- #
# check_registry
# --------------------------------------------------------------------------- #


def test_check_registry_ok_on_real_models_toml():
    r = doctor.check_registry(REPO_ROOT / "models.toml")
    assert r.ok is True
    assert "4 model(s)" in r.detail


def test_check_registry_fails_on_bad_file(tmp_path):
    bad = tmp_path / "models.toml"
    bad.write_text("not toml [[[")
    r = doctor.check_registry(bad)
    assert r.ok is False


# --------------------------------------------------------------------------- #
# check_client_config
# --------------------------------------------------------------------------- #


def test_client_config_missing_file_is_ok(tmp_path):
    results = doctor.check_client_config("vscode", tmp_path / "nope.json")
    assert len(results) == 1 and results[0].ok is True
    assert "not present" in results[0].detail


def test_client_config_present_no_refs_is_ok(tmp_path):
    p = tmp_path / "chatLanguageModels.json"
    p.write_text(json.dumps([{"name": "x", "models": []}]))
    results = doctor.check_client_config("vscode", p)
    assert len(results) == 1 and results[0].ok is True
    assert "no model references" in results[0].detail


def test_client_config_vscode_ok_when_id_present(tmp_path):
    p = tmp_path / "chatLanguageModels.json"
    p.write_text(
        json.dumps(
            [{"name": "servedeck", "models": [{"id": "A", "url": "http://127.0.0.1:8099/v1/chat/completions"}]}]
        )
    )
    get = _stub_get({"http://127.0.0.1:8099/v1/models": _models_response("A", "B")})
    results = doctor.check_client_config("vscode", p, http_get=get)
    assert len(results) == 1
    assert results[0].ok is True
    assert results[0].name == "vscode: A"


def test_client_config_vscode_missing_when_id_absent(tmp_path):
    p = tmp_path / "chatLanguageModels.json"
    p.write_text(
        json.dumps(
            [{"name": "servedeck", "models": [{"id": "A", "url": "http://127.0.0.1:8099/v1/chat/completions"}]}]
        )
    )
    get = _stub_get({"http://127.0.0.1:8099/v1/models": _models_response("SOMETHING-ELSE")})
    results = doctor.check_client_config("vscode", p, http_get=get)
    assert results[0].ok is False
    assert "missing" in results[0].detail


def test_client_config_registered_but_stopped_model_is_wired_not_failing(tmp_path):
    """2026-09-17: every client is wired through the gateway, which lists only
    the models that are up. An entry for a registry model that is stopped is
    the normal state of most entries, not a failure."""
    p = tmp_path / "chatLanguageModels.json"
    p.write_text(
        json.dumps(
            [{"name": "servedeck", "models": [
                {"id": "Qwen3.8-27B-NVFP4", "url": "http://127.0.0.1:8010/v1/chat/completions"},
                {"id": "Nope", "url": "http://127.0.0.1:8010/v1/chat/completions"},
            ]}]
        )
    )
    get = _stub_get({"http://127.0.0.1:8010/v1/models": _models_response("Qwen3.8-Flash-Next")})
    results = doctor.check_client_config(
        "vscode", p, http_get=get, known_names={"Qwen3.8-27B-NVFP4", "qwen27b"}
    )
    by_name = {r.name: r for r in results}
    assert by_name["vscode: Qwen3.8-27B-NVFP4"].ok is True
    assert "not running" in by_name["vscode: Qwen3.8-27B-NVFP4"].detail
    assert by_name["vscode: Nope"].ok is False, "an id the registry does not know is still missing"


def test_client_config_unreachable(tmp_path):
    p = tmp_path / "chatLanguageModels.json"
    p.write_text(
        json.dumps(
            [{"name": "servedeck", "models": [{"id": "A", "url": "http://127.0.0.1:8099/v1/chat/completions"}]}]
        )
    )
    get = _stub_get({"http://127.0.0.1:8099/v1/models": httpx.ConnectError("refused")})
    results = doctor.check_client_config("vscode", p, http_get=get)
    assert results[0].ok is False
    assert "unreachable" in results[0].detail


def test_client_config_codex_resolves_provider_indirection(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[model_providers.servedeck]\nbase_url = "http://127.0.0.1:8099/v1"\n\n'
        '[profiles.qwen27b]\nmodel_provider = "servedeck"\nmodel = "Qwen3.8-27B-NVFP4"\n'
    )
    get = _stub_get({"http://127.0.0.1:8099/v1/models": _models_response("Qwen3.8-27B-NVFP4")})
    results = doctor.check_client_config("codex", p, http_get=get)
    assert len(results) == 1 and results[0].ok is True


def test_client_config_kimi_resolves_provider_indirection(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[providers.servedeck]\nbase_url = "http://127.0.0.1:8099/v1"\n\n'
        '[models."servedeck/lfm2"]\nprovider = "servedeck"\nmodel = "LFM2.5-350M"\n'
    )
    get = _stub_get({"http://127.0.0.1:8099/v1/models": _models_response("LFM2.5-350M")})
    results = doctor.check_client_config("kimi", p, http_get=get)
    assert len(results) == 1 and results[0].ok is True


# --------------------------------------------------------------------------- #
# check_port (stubbed)
# --------------------------------------------------------------------------- #


def _fake_model(**kw) -> models.Model:
    base = dict(key="m", id="M", repo="org/m", slot="main", port=9001, build="stock", ctx=1000)
    base.update(kw)
    return models.Model(**base)


class _ModelsHandler(BaseHTTPRequestHandler):
    """A real /v1/models — a stdlib HTTP server on an ephemeral port, standing
    in for a vLLM engine so the mismatch branch of check_port is proven
    against real HTTP transport and a real JSON body, not a stubbed
    ``http_get`` — without depending on the box's own :8007, which is not
    live by default any more (the LFM2 resident is opt-in, per the
    2026-09-16 owner directive)."""

    served_ids: tuple[str, ...] = ("Some-Other-Model",)

    def do_GET(self) -> None:  # noqa: N802 - stdlib method name
        if self.path != "/v1/models":
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(
            {"object": "list", "data": [{"id": i, "object": "model"} for i in self.served_ids]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        pass  # keep test output clean


class _RealModelsServer:
    """Context manager: a real HTTP server serving ``served_ids`` at
    /v1/models on an ephemeral 127.0.0.1 port, in a daemon thread."""

    def __init__(self, *served_ids: str) -> None:
        self._served_ids = served_ids or ("Some-Other-Model",)

    def __enter__(self) -> int:
        ids = self._served_ids

        class _Handler(_ModelsHandler):
            served_ids = ids

        self._server = HTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self._server.server_address[1]

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._thread.join(timeout=5)
        self._server.server_close()


def test_check_port_ok_when_id_listed():
    m = _fake_model(port=9001, id="M")
    get = _stub_get({"http://127.0.0.1:9001/v1/models": _models_response("M", "other")})
    r = doctor.check_port(m, http_get=get)
    assert r.ok is True and "listening" in r.detail


def test_check_port_not_listening_is_ok():
    m = _fake_model(port=9001)
    get = _stub_get({"http://127.0.0.1:9001/v1/models": httpx.ConnectError("refused")})
    r = doctor.check_port(m, http_get=get)
    assert r.ok is True
    assert r.detail == "not listening"


def test_check_port_wrong_model_is_a_failure():
    """The one real failure mode: something else answers on our port."""
    m = _fake_model(port=9001, id="Expected")
    get = _stub_get({"http://127.0.0.1:9001/v1/models": _models_response("SomeoneElse")})
    r = doctor.check_port(m, http_get=get)
    assert r.ok is False
    assert "SomeoneElse" in r.detail


def test_check_port_alias_counts_as_match():
    m = _fake_model(port=9001, id="Expected", aliases=("alias-only",))
    get = _stub_get({"http://127.0.0.1:9001/v1/models": _models_response("alias-only")})
    r = doctor.check_port(m, http_get=get)
    assert r.ok is True


def test_check_port_flags_wrong_expected_id_against_a_real_http_server():
    """Same shape as check_port's mismatch branch, proven against a real HTTP
    response instead of a stubbed http_get -- but hermetic: an in-test stdlib
    server stands in for the vLLM engine, on an ephemeral port, so this does
    not depend on anything actually listening on the box's own :8007 (moved
    off that dependency 2026-09-16; see _RealModelsServer above)."""
    with _RealModelsServer("Some-Other-Model") as port:
        m = _fake_model(key="lfm2", id="Not-The-Real-Model", port=port)
        r = doctor.check_port(m, timeout=2.0)
    assert r.ok is False
    assert "expected one of" in r.detail


# --------------------------------------------------------------------------- #
# check_port — against the box's own live, read-only :8007 / :8010
# --------------------------------------------------------------------------- #


def test_check_port_live_lfm2_on_8007():
    m = _fake_model(key="lfm2", id="LFM2.5-350M", port=8007)
    r = doctor.check_port(m, timeout=2.0)
    assert r.ok is True, r.detail
    assert "listening" in r.detail


def test_check_port_live_gateway_on_8010():
    """The gateway's own port, whatever is behind it right now.

    This must NOT assert that a particular model is live: :8010 is a gateway,
    its answer depends on which model the box happens to be serving, and during
    the v1 -> v2 cutover it changes owner entirely.  The property under test is
    that check_port reports honestly either way — ok when the expected id is
    served, and a detail naming the HTTP status when the gateway answers but
    has no model behind it (a real 503 seen on 2026-09-14).
    """
    m = _fake_model(key="gw", id="Qwen3.8-27B-NVFP4", port=8010)
    r = doctor.check_port(m, timeout=2.0)
    if r.ok:
        assert "Qwen3.8-27B-NVFP4" in r.detail or r.detail == "ok", r.detail
    else:
        assert "HTTP" in r.detail or "expected one of" in r.detail, r.detail
        assert "503" in r.detail or "expected one of" in r.detail, r.detail
def test_a_client_reference_to_a_remote_host_is_reported_but_never_probed(tmp_path):
    """doctor must not send a request off this box.

    Measured on this box: ``~/.kimi-code/config.toml`` declares four
    ``managed:kimi-code`` models behind ``https://api.kimi.com/coding/v1``.
    Before the filter, ``doctor`` sent an unauthenticated GET there on every
    run and reported each 401 as a servedeck failure — four permanent red
    lines for four correct entries, plus an outbound request from a local
    diagnostic. ``http_get`` raising here is the proof that the check did not
    call it: nothing stubs those URLs, so a probe would surface as the
    exception rather than as a passing assertion.
    """

    def must_not_be_called(url: str, timeout: float):  # pragma: no cover
        raise AssertionError(f"doctor probed a remote endpoint: {url}")

    p = tmp_path / "chatLanguageModels.json"
    p.write_text(
        json.dumps(
            [
                {
                    "name": "servedeck",
                    "models": [
                        {"id": "cloud", "url": "https://api.kimi.com/coding/v1/chat/completions"}
                    ],
                }
            ]
        )
    )
    results = doctor.check_client_config("vscode", p, http_get=must_not_be_called)
    assert len(results) == 1
    assert results[0].ok is True
    assert results[0].name == "vscode config (remote)"
    assert "api.kimi.com" in results[0].detail
    assert "cloud" in results[0].detail


def test_a_config_mixing_local_and_remote_checks_only_the_local_one(tmp_path):
    p = tmp_path / "chatLanguageModels.json"
    p.write_text(
        json.dumps(
            [
                {
                    "name": "servedeck",
                    "models": [
                        {"id": "cloud", "url": "https://api.kimi.com/coding/v1/chat/completions"},
                        {"id": "A", "url": "http://localhost:8099/v1/chat/completions"},
                    ],
                }
            ]
        )
    )
    get = _stub_get({"http://localhost:8099/v1/models": _models_response("A")})
    names = {r.name: r for r in doctor.check_client_config("vscode", p, http_get=get)}
    assert set(names) == {"vscode config (remote)", "vscode: A"}
    assert names["vscode: A"].ok is True
    # "localhost" counts as this box, the way VS Code's generated URL spells it.
    assert doctor.is_local_ref("http://localhost:8010/v1/models")
    assert doctor.is_local_ref("http://127.0.0.1:8010/v1/models")
    assert not doctor.is_local_ref("https://api.openai.com/v1/models")


def test_check_port_against_the_real_live_gateway_on_8010():
    """``check_port``'s three states against a REAL socket, not a stub.

    The claim under test is doctor's, not the box's: whatever :8010 answers
    right now, doctor must classify it into the right one of the three states
    and say which. The previous version of this test asserted ``ok is True``
    unconditionally, which made it a claim about whether the production 27B
    happened to be up — it fails today with "answers but response is unusable:
    HTTP 503", because the v1 gateway on :8010 is proxying to a :8004 that is
    not running. A test that goes red when a model is stopped is a broken
    instrument, and a broken instrument reports whatever the box is doing as a
    defect.
    """
    m = _fake_model(key="gw", id="Qwen3.8-27B-NVFP4", port=8010)
    r = doctor.check_port(m, timeout=2.0)

    try:
        live = httpx.get("http://127.0.0.1:8010/v1/models", timeout=2.0)
    except httpx.TransportError:
        assert r.ok is True and r.detail == "not listening", r
        return

    if live.status_code != 200:
        # Answering, but not with a model list: a real failure, and doctor must
        # name the status rather than call it "not listening".
        assert r.ok is False, r
        assert str(live.status_code) in r.detail, r.detail
        return

    ids = [e.get("id") for e in live.json().get("data", [])]
    if m.id in ids:
        assert r.ok is True and "listening" in r.detail, r
    else:
        assert r.ok is False and "expected one of" in r.detail, r


def test_check_port_not_listening_real_socket():
    """A port well outside the reserved 8000-8010 range, so this is not one of
    the box's live services — a genuine connection-refused case."""
    m = _fake_model(port=19999)
    r = doctor.check_port(m, timeout=1.0)
    assert r.ok is True
    assert r.detail == "not listening"


# --------------------------------------------------------------------------- #
# check_model_units — transient systemd-run --user units, never a directory
# listing (see doctor.py's module docstring for why).
# --------------------------------------------------------------------------- #


def _units_run(stdout: str = "", returncode: int = 0) -> units.Runner:
    def run(argv):
        return subprocess.CompletedProcess(list(argv), returncode, stdout, "")

    return run


def test_check_model_units_all_not_running_is_ok():
    """No live model-* unit at all is the normal, expected state — most
    registry models are not running most of the time."""
    reg = models.load(REPO_ROOT / "models.toml")
    results = doctor.check_model_units(reg, run=_units_run(stdout=""))
    by_name = {r.name: r for r in results}
    for key in reg.models:
        assert by_name[f"unit (model-{key})"].ok is True
        assert by_name[f"unit (model-{key})"].detail == "not running"


def test_check_model_units_reports_a_running_unit():
    reg = models.load(REPO_ROOT / "models.toml")
    stdout = "model-flashnext.service loaded active running Model flashnext\n"
    results = doctor.check_model_units(reg, run=_units_run(stdout=stdout))
    by_name = {r.name: r for r in results}
    assert by_name["unit (model-flashnext)"].ok is True
    assert by_name["unit (model-flashnext)"].detail == "running"
    assert by_name["unit (model-qwen27b)"].detail == "not running"


def test_check_model_units_flags_a_stray_unit_as_the_one_real_failure():
    """A model-* unit whose key the registry does not know at all — never
    'not running', which is always ok."""
    reg = models.load(REPO_ROOT / "models.toml")
    stdout = "model-ghost.service loaded active running Ghost\n"
    results = doctor.check_model_units(reg, run=_units_run(stdout=stdout))
    stray = [r for r in results if r.name == "unit (model-ghost)"]
    assert len(stray) == 1
    assert stray[0].ok is False
    assert "no model with key 'ghost'" in stray[0].detail
    # every registry-known model is still just "not running", not penalised by
    # the stray.
    assert all(r.ok for r in results if r.name != "unit (model-ghost)")


def test_check_model_units_list_failure_is_reported_not_raised():
    def raising_run(argv):
        raise units.UnitError("systemctl: not found")

    reg = models.load(REPO_ROOT / "models.toml")
    results = doctor.check_model_units(reg, run=raising_run)
    assert len(results) == 1
    assert results[0].ok is False
    assert "could not list model-* units" in results[0].detail


def test_check_model_units_live_real_systemctl():
    """Against the box's OWN systemctl --user (read-only `list-units`, never
    start/stop/enable/disable — the same carve-out the live :8007/:8010 GETs
    use). No model-* unit exists yet on this box (P1-P4's supervisor has never
    launched one), so every registry model is expected 'not running'."""
    reg = models.load(REPO_ROOT / "models.toml")
    results = doctor.check_model_units(reg)
    assert all(r.ok for r in results), results


# --------------------------------------------------------------------------- #
# run_doctor / all_ok / format_table
# --------------------------------------------------------------------------- #


def test_run_doctor_registry_failure_short_circuits(tmp_path):
    bad = tmp_path / "models.toml"
    bad.write_text("not toml [[[")
    results = doctor.run_doctor(bad)
    assert len(results) == 1
    assert results[0].ok is False


def test_run_doctor_full_pass_with_stubbed_network(tmp_path):
    reg = models.load(REPO_ROOT / "models.toml")
    responses = {
        f"http://127.0.0.1:{m.port}/v1/models": httpx.ConnectError("refused")
        for m in reg.models.values()
    }
    get = _stub_get(responses)
    results = doctor.run_doctor(
        REPO_ROOT / "models.toml",
        client_files={
            "vscode": tmp_path / "vscode.json",
            "codex": tmp_path / "codex.toml",
            "kimi": tmp_path / "kimi.toml",
        },
        unit_run=_units_run(stdout=""),
        http_get=get,
        # The two host checks P4 moved out of preflight.py take their input,
        # for the same reason http_get does: no test can set this box's
        # kernel.yama.ptrace_scope, and reading it here would make the result
        # depend on which machine the suite ran on. 0 and no markers are the
        # states a correctly-configured host is in (REDESIGN §6).
        marker_paths=(),
        ptrace_scope=0,
    )
    # Every port reports "not listening" (ok=True); every model reports "not
    # running" (ok=True; no stray units in the stub); missing client files are
    # also individually ok — nothing here should raise or fail.
    assert len(results) >= 1 + 3 + 4 + 4  # registry + 3 client files + 4 ports + 4 units
    port_results = [r for r in results if r.name.startswith("port ")]
    assert all(r.ok for r in port_results)
    unit_results = [r for r in results if r.name.startswith("unit (")]
    assert len(unit_results) == 4
    assert all(r.ok for r in unit_results)
    assert doctor.all_ok(results)


def test_all_ok():
    assert doctor.all_ok([doctor.CheckResult("a", True, "")]) is True
    assert doctor.all_ok([doctor.CheckResult("a", True, ""), doctor.CheckResult("b", False, "x")]) is False
    assert doctor.all_ok([]) is True


def test_format_table_shows_status_and_detail():
    out = doctor.format_table(
        [doctor.CheckResult("check a", True, "fine"), doctor.CheckResult("check b", False, "broken")]
    )
    assert "OK" in out and "FAIL" in out
    assert "fine" in out and "broken" in out
    assert "check a" in out and "check b" in out


def test_format_table_empty():
    assert doctor.format_table([]) == "(no checks ran)"


# --------------------------------------------------------------------------- #
# The two checks that moved out of preflight.py (P4)
# --------------------------------------------------------------------------- #

#: One model, no `needs_tty`, so `check_ptrace_scope` has nothing to report.
MINIMAL_TOML = """
[gpu]
total_mib = 100000
margin_mib = 1024

[builds.stock]
venv = "/opt/stock"
cuda_home = "/opt/stock/lib/python3.13/site-packages/nvidia/cu13"

[models.plain]
id = "Plain-Model"
repo = "org/plain"
slot = "resident"
vram_mib = 2000
port = 19011
build = "stock"
ctx = 4096
"""


def test_a_training_marker_is_a_failure_not_a_warning(tmp_path):
    """`qwen-server-run.sh`'s own guard 1. The marker files are written by
    tools outside this repo (an AlgoTrading training run, chiefly) and nothing
    else would notice them.

    A hit is a FAILURE: the correct response is to leave the card alone, and a
    warning is what gets scrolled past.
    """
    marker = tmp_path / "training_in_progress"
    assert doctor.check_training_marker((str(marker),)).ok is True
    marker.write_text("")
    result = doctor.check_training_marker((str(marker),))
    assert result.ok is False
    assert str(marker) in result.detail
    assert "do not start a model" in result.detail


def test_ptrace_scope_is_only_checked_for_models_that_need_a_tty(tmp_path):
    """Checked per model, so the answer names which model would fail — and not
    checked at all for a registry with no `needs_tty` model, rather than
    emitting a passing row nobody asked for."""
    reg = models.load(REPO_ROOT / "models.toml")
    needy = [m.key for m in reg.models.values() if m.needs_tty]
    assert needy, "models.toml has no needs_tty model; this test has no subject"

    ok = doctor.check_ptrace_scope(reg, 0)
    assert [r.name for r in ok] == [f"ptrace_scope ({k})" for k in needy]
    assert all(r.ok for r in ok)

    bad = doctor.check_ptrace_scope(reg, 1)
    assert all(not r.ok for r in bad)
    # The detail must say why the launcher's own sudo sysctl does not save it:
    # there is no tty under systemd, so the relaxation silently no-ops.
    assert "no tty" in bad[0].detail
    assert "sysctl" in bad[0].detail

    unreadable = doctor.check_ptrace_scope(reg, None)
    assert all(not r.ok for r in unreadable)
    assert "could not read" in unreadable[0].detail


def test_a_registry_with_no_needs_tty_model_gets_no_ptrace_row(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(MINIMAL_TOML)
    reg = models.load(path)
    assert not any(m.needs_tty for m in reg.models.values())
    assert doctor.check_ptrace_scope(reg, 1) == []


def test_none_scope_is_distinguishable_from_not_supplied(tmp_path):
    """Over-correction guard on the sentinel.

    `scope=None` is a real, reportable state ("could not read the sysctl"), so
    it cannot double as "not supplied" — collapsing the two would make an
    explicit `None` silently fall back to reading the host, and the test above
    would pass on a box where ptrace_scope happens to be 0.
    """
    reg = models.load(REPO_ROOT / "models.toml")
    explicit_none = doctor.check_ptrace_scope(reg, None)
    assert all(not r.ok for r in explicit_none)
    assert "could not read" in explicit_none[0].detail
    # Not supplied: reads the host, whatever it says — only the shape is fixed.
    from_host = doctor.check_ptrace_scope(reg)
    assert len(from_host) == len(explicit_none)
