"""desired.json schema-1 migration persists once (2026-09-17)."""
import json
import logging

from servedeck import desired


V1 = {"version": 1, "desired_state": "RUNNING", "backend": "flashnext", "port": 8001}


def test_a_v1_file_is_rewritten_as_v2_on_first_read_and_the_original_kept(tmp_path) -> None:
    path = tmp_path / "desired.json"
    path.write_text(json.dumps(V1))
    got = desired.load(path)
    assert got.main == "flashnext" and got.residents == []
    on_disk = json.loads(path.read_text())
    assert on_disk["version"] == desired.SCHEMA_VERSION and on_disk["main"] == "flashnext"
    assert json.loads((tmp_path / "desired.json.v1").read_text()) == V1


def test_the_migration_warning_is_logged_once_not_on_every_poll(tmp_path, caplog) -> None:
    path = tmp_path / "desired.json"
    path.write_text(json.dumps(V1))
    with caplog.at_level(logging.WARNING, logger="servedeck.desired"):
        desired.load(path)
        desired.load(path)
        desired.load(path)
    assert sum("schema version 1" in r.message for r in caplog.records) == 1


def test_a_stopped_v1_file_migrates_to_no_main(tmp_path) -> None:
    path = tmp_path / "desired.json"
    path.write_text(json.dumps({**V1, "desired_state": "STOPPED"}))
    assert desired.load(path).main is None
    assert json.loads(path.read_text())["main"] is None
