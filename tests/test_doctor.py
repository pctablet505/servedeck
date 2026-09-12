"""servedeck.doctor — unit tests with a stubbed http_get, plus two tests against
the box's own live, read-only :8007 (LFM2, always on) and :8010 (gateway) —
the only two ports the hard rules allow this test suite to touch.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from servedeck import doctor, models

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
            [{"name": "servedeck", "models": [{"id": "A", "url": "http://x/v1/chat/completions"}]}]
        )
    )
    get = _stub_get({"http://x/v1/models": _models_response("A", "B")})
    results = doctor.check_client_config("vscode", p, http_get=get)
    assert len(results) == 1
    assert results[0].ok is True
    assert results[0].name == "vscode: A"


def test_client_config_vscode_missing_when_id_absent(tmp_path):
    p = tmp_path / "chatLanguageModels.json"
    p.write_text(
        json.dumps(
            [{"name": "servedeck", "models": [{"id": "A", "url": "http://x/v1/chat/completions"}]}]
        )
    )
    get = _stub_get({"http://x/v1/models": _models_response("SOMETHING-ELSE")})
    results = doctor.check_client_config("vscode", p, http_get=get)
    assert results[0].ok is False
    assert "missing" in results[0].detail


def test_client_config_unreachable(tmp_path):
    p = tmp_path / "chatLanguageModels.json"
    p.write_text(
        json.dumps(
            [{"name": "servedeck", "models": [{"id": "A", "url": "http://x/v1/chat/completions"}]}]
        )
    )
    get = _stub_get({"http://x/v1/models": httpx.ConnectError("refused")})
    results = doctor.check_client_config("vscode", p, http_get=get)
    assert results[0].ok is False
    assert "unreachable" in results[0].detail


def test_client_config_codex_resolves_provider_indirection(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[model_providers.servedeck]\nbase_url = "http://x/v1"\n\n'
        '[profiles.qwen27b]\nmodel_provider = "servedeck"\nmodel = "Qwen3.8-27B-NVFP4"\n'
    )
    get = _stub_get({"http://x/v1/models": _models_response("Qwen3.8-27B-NVFP4")})
    results = doctor.check_client_config("codex", p, http_get=get)
    assert len(results) == 1 and results[0].ok is True


def test_client_config_kimi_resolves_provider_indirection(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[providers.servedeck]\nbase_url = "http://x/v1"\n\n'
        '[models."servedeck/lfm2"]\nprovider = "servedeck"\nmodel = "LFM2.5-350M"\n'
    )
    get = _stub_get({"http://x/v1/models": _models_response("LFM2.5-350M")})
    results = doctor.check_client_config("kimi", p, http_get=get)
    assert len(results) == 1 and results[0].ok is True


# --------------------------------------------------------------------------- #
# check_port (stubbed)
# --------------------------------------------------------------------------- #


def _fake_model(**kw) -> models.Model:
    base = dict(key="m", id="M", repo="org/m", slot="main", port=9001, build="stock", ctx=1000)
    base.update(kw)
    return models.Model(**base)


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


# --------------------------------------------------------------------------- #
# check_port — against the box's own live, read-only :8007 / :8010
# --------------------------------------------------------------------------- #


def test_check_port_live_lfm2_on_8007():
    m = _fake_model(key="lfm2", id="LFM2.5-350M", port=8007)
    r = doctor.check_port(m, timeout=2.0)
    assert r.ok is True, r.detail
    assert "listening" in r.detail


def test_check_port_live_8007_flags_wrong_expected_id():
    """Same live server, wrong expectation — proves the drift-detection branch
    against a real vLLM /v1/models response, not just a stub."""
    m = _fake_model(key="lfm2", id="Not-The-Real-Model", port=8007)
    r = doctor.check_port(m, timeout=2.0)
    assert r.ok is False
    assert "expected one of" in r.detail


def test_check_port_live_gateway_on_8010():
    m = _fake_model(key="gw", id="Qwen3.8-27B-NVFP4", port=8010)
    r = doctor.check_port(m, timeout=2.0)
    assert r.ok is True, r.detail


def test_check_port_not_listening_real_socket():
    """A port well outside the reserved 8000-8010 range, so this is not one of
    the box's live services — a genuine connection-refused case."""
    m = _fake_model(port=19999)
    r = doctor.check_port(m, timeout=1.0)
    assert r.ok is True
    assert r.detail == "not listening"


# --------------------------------------------------------------------------- #
# check_systemd_units
# --------------------------------------------------------------------------- #


def test_systemd_units_known_name_found(tmp_path):
    (tmp_path / "qwen-vllm.service").write_text("[Unit]\n")
    reg = models.load(REPO_ROOT / "models.toml")
    results = doctor.check_systemd_units(reg, tmp_path)
    by_name = {r.name: r for r in results}
    assert by_name["systemd unit (qwen27b)"].ok is True


def test_systemd_units_missing_reports_expected_filename(tmp_path):
    reg = models.load(REPO_ROOT / "models.toml")
    results = doctor.check_systemd_units(reg, tmp_path)
    by_name = {r.name: r for r in results}
    assert by_name["systemd unit (qwen27b)"].ok is False
    assert "qwen-vllm.service" in by_name["systemd unit (qwen27b)"].detail
    # flashnext/glm53 have no known live unit yet -> fall back to the future
    # model-<key>.service name.
    assert "model-flashnext.service" in by_name["systemd unit (flashnext)"].detail


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
        systemd_dir=tmp_path,
        http_get=get,
    )
    # Every port reports "not listening" (ok=True); missing client files and
    # missing systemd units for flashnext/glm53 are also individually ok/fail,
    # but nothing here should raise.
    assert len(results) >= 1 + 3 + 4 + 4  # registry + 3 client files + 4 ports + 4 units
    port_results = [r for r in results if r.name.startswith("port ")]
    assert all(r.ok for r in port_results)


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
