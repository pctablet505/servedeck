"""servedeck.cli — the `servedeck` console command.

`wire`/`doctor` tests always point at a tmp models.toml with high, definitely-
unused ports (>19000, well outside the 8000-8010 reserved range) and monkeypatch
`servedeck.wire`'s path constants / WIRE_TARGETS / BACKUP_ROOT to tmp_path, so
nothing here ever touches ~/.config, ~/.codex, ~/.kimi-code, or a live port.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from servedeck import cli, models, wire

REPO_ROOT = Path(__file__).resolve().parent.parent

SAFE_TOML = """
[gpu]
total_mib = 100000
margin_mib = 1000

[builds.stock]
venv = "/opt/stock"
cuda_home = "/opt/stock/lib/python3.13/site-packages/nvidia/cu13"

[models.a]
id = "A"
repo = "org/a"
slot = "main"
port = 19991
build = "stock"
ctx = 1000

[models.a.tools]
parser = "x"
"""


@pytest.fixture
def safe_toml(tmp_path) -> Path:
    p = tmp_path / "models.toml"
    p.write_text(SAFE_TOML)
    return p


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #


def test_models_command_prints_table_from_real_registry(capsys):
    rc = cli.main(["models", "--models-toml", str(REPO_ROOT / "models.toml")])
    assert rc == 0
    out = capsys.readouterr().out
    for key in ("qwen27b", "flashnext", "glm53", "lfm2"):
        assert key in out
    assert "port" in out and "8004" in out


def test_models_command_reports_registry_error(tmp_path, capsys):
    bad = tmp_path / "models.toml"
    bad.write_text("not toml [[[")
    rc = cli.main(["models", "--models-toml", str(bad)])
    assert rc == 1
    assert "error" in capsys.readouterr().err


def test_models_command_uses_safe_toml_fixture(safe_toml, capsys):
    rc = cli.main(["models", "--models-toml", str(safe_toml)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "A" in out and "19991" in out


# --------------------------------------------------------------------------- #
# wire
# --------------------------------------------------------------------------- #


@pytest.fixture
def isolated_wire_targets(tmp_path, monkeypatch):
    """Redirects every wire target AND the backup root under tmp_path."""
    vscode = tmp_path / "clients" / "chatLanguageModels.json"
    codex = tmp_path / "clients" / "codex-config.toml"
    kimi = tmp_path / "clients" / "kimi-config.toml"
    targets = (
        wire.WireTarget("vscode chatLanguageModels.json", vscode, wire.render_vscode),
        wire.WireTarget("codex config.toml", codex, wire.render_codex),
        wire.WireTarget("kimi config.toml", kimi, wire.render_kimi),
    )
    monkeypatch.setattr(wire, "WIRE_TARGETS", targets)
    monkeypatch.setattr(wire, "BACKUP_ROOT", tmp_path / "state" / "backups")
    return {"vscode": vscode, "codex": codex, "kimi": kimi}


def test_wire_dry_run_prints_diff_and_does_not_write(safe_toml, isolated_wire_targets, capsys):
    rc = cli.main(["wire", "--models-toml", str(safe_toml)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "would change" in out
    assert not isolated_wire_targets["vscode"].exists()
    assert not isolated_wire_targets["codex"].exists()
    assert not isolated_wire_targets["kimi"].exists()


def test_wire_dry_run_is_the_default(safe_toml, isolated_wire_targets, capsys):
    """--apply must be opt-in (hard rule: wire defaults to --dry-run)."""
    rc1 = cli.main(["wire", "--models-toml", str(safe_toml)])
    out1 = capsys.readouterr().out
    assert rc1 == 0 and "would change" in out1
    assert not isolated_wire_targets["vscode"].exists()


def test_wire_apply_writes_and_backs_up(safe_toml, isolated_wire_targets, capsys):
    rc = cli.main(["wire", "--models-toml", str(safe_toml), "--apply"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "applied" in out
    assert isolated_wire_targets["vscode"].is_file()
    assert isolated_wire_targets["codex"].is_file()
    assert isolated_wire_targets["kimi"].is_file()

    backup_root = wire.BACKUP_ROOT
    backups = list(backup_root.rglob("*")) if backup_root.exists() else []
    backup_files = [b for b in backups if b.is_file()]
    assert backup_files, "expected at least one backup file under BACKUP_ROOT"


def test_wire_apply_backup_contains_pre_write_content(safe_toml, isolated_wire_targets, capsys):
    vscode_path = isolated_wire_targets["vscode"]
    vscode_path.parent.mkdir(parents=True, exist_ok=True)
    vscode_path.write_text("[]")
    cli.main(["wire", "--models-toml", str(safe_toml), "--apply"])
    backup_files = [b for b in wire.BACKUP_ROOT.rglob("*") if b.is_file()]
    contents = [b.read_text() for b in backup_files]
    assert "[]" in contents


def test_wire_second_apply_reports_no_changes(safe_toml, isolated_wire_targets, capsys):
    cli.main(["wire", "--models-toml", str(safe_toml), "--apply"])
    capsys.readouterr()
    rc = cli.main(["wire", "--models-toml", str(safe_toml), "--apply"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no changes" in out
    assert "applied" not in out


def test_wire_reports_registry_error(tmp_path, isolated_wire_targets, capsys):
    bad = tmp_path / "models.toml"
    bad.write_text("not toml [[[")
    rc = cli.main(["wire", "--models-toml", str(bad)])
    assert rc == 1
    assert "error" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


@pytest.fixture
def isolated_doctor_client_files(tmp_path, monkeypatch):
    monkeypatch.setattr(wire, "VSCODE_CHAT_LM_PATH", tmp_path / "no-vscode.json")
    monkeypatch.setattr(wire, "CODEX_CONFIG_PATH", tmp_path / "no-codex.toml")
    monkeypatch.setattr(wire, "KIMI_CONFIG_PATH", tmp_path / "no-kimi.toml")


def test_doctor_command_all_ok_with_safe_ports(safe_toml, isolated_doctor_client_files, tmp_path, capsys):
    # No --systemd-dir override: check_model_units does a real, read-only
    # `systemctl --user list-units 'model-*'` (never start/stop/enable —
    # allowed by the hard rules the same way the live :8007/:8010 GETs are).
    # There is no real model-a unit, so this is "not running" (ok), not a
    # failure — confirmed empty by `systemctl --user list-units 'model-*'`.
    rc = cli.main(["doctor", "--models-toml", str(safe_toml)])
    out = capsys.readouterr().out
    assert "registry loads" in out
    assert "port 19991" in out
    assert "unit (model-a)" in out
    # Port 19991 is not listening anywhere, and no model-a unit is running ->
    # both are OK states, not failures.
    assert rc == 0, out


def test_doctor_command_fails_on_bad_registry(tmp_path, isolated_doctor_client_files, capsys):
    bad = tmp_path / "models.toml"
    bad.write_text("not toml [[[")
    rc = cli.main(["doctor", "--models-toml", str(bad)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out


# --------------------------------------------------------------------------- #
# stub commands
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("command", cli.STUB_COMMANDS)
def test_stub_commands_exit_2(command, capsys):
    rc = cli.main([command])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not wired yet (P3/P4)" in err
    assert command in err


def test_no_command_is_a_usage_error():
    with pytest.raises(SystemExit) as exc_info:
        cli.main([])
    assert exc_info.value.code == 2


# --------------------------------------------------------------------------- #
# __main__.py must still work for `python -m servedeck` (the dashboard)
# --------------------------------------------------------------------------- #


def test_dunder_main_still_importable_and_has_main():
    from servedeck import __main__ as dashboard_main

    assert callable(dashboard_main.main)


def test_console_script_points_at_cli_main():
    import tomllib

    pyproject = REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    assert data["project"]["scripts"]["servedeck"] == "servedeck.cli:main"
