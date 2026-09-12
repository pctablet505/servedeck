"""servedeck.wire — pure generator tests. Every render_* call here uses only
in-memory strings / tmp_path; nothing touches $HOME, ~/.codex, ~/.kimi-code, or
~/.config, per the packet's hard rules.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from servedeck import models, wire

REPO_ROOT = Path(__file__).resolve().parent.parent


def _fixed_resolver(ctx_by_key: dict[str, int]):
    def resolve(m: models.Model) -> int:
        if isinstance(m.ctx, int):
            return m.ctx
        return ctx_by_key[m.key]

    return resolve


@pytest.fixture
def registry() -> models.Registry:
    return models.load(REPO_ROOT / "models.toml")


@pytest.fixture
def resolve_ctx(registry):
    return _fixed_resolver(
        {"qwen27b": 262144, "flashnext": 262144, "glm53": 1048576}
    )


# --------------------------------------------------------------------------- #
# VS Code
# --------------------------------------------------------------------------- #


def test_vscode_empty_existing_creates_group(registry, resolve_ctx):
    out = wire.render_vscode(registry, "", resolve_ctx=resolve_ctx)
    data = json.loads(out)
    assert isinstance(data, list) and len(data) == 1
    assert data[0]["name"] == "servedeck"
    assert data[0]["vendor"] == "customendpoint"
    assert data[0]["apiType"] == "chat-completions"
    ids = [e["id"] for e in data[0]["models"]]
    # one per id and per preset: 4 base ids + 3 glm53 presets = 7.
    assert ids.count("Qwen3.8-27B-NVFP4") == 1
    assert "glm53-flash-low" in ids and "glm53-flash-high" in ids and "glm53-flash-max" in ids
    assert len(ids) == 7


def test_vscode_entry_fields(registry, resolve_ctx):
    out = wire.render_vscode(registry, "", resolve_ctx=resolve_ctx)
    data = json.loads(out)
    by_id = {e["id"]: e for e in data[0]["models"]}
    e = by_id["Qwen3.8-27B-NVFP4"]
    assert e["url"] == "http://localhost:8010/v1/chat/completions"
    assert e["toolCalling"] is True
    assert e["vision"] is True
    assert e["maxOutputTokens"] == 36000
    assert e["maxInputTokens"] == 262144 - 36000
    lfm2 = by_id["LFM2.5-350M"]
    assert lfm2["vision"] is False
    assert lfm2["maxOutputTokens"] == 4096
    assert lfm2["maxInputTokens"] == 32768 - 4096


def test_vscode_preserves_unrelated_vendor_groups(registry, resolve_ctx):
    existing = json.dumps(
        [{"name": "some-other-vendor", "vendor": "openai", "apiType": "chat", "models": [{"id": "keep-me"}]}]
    )
    out = wire.render_vscode(registry, existing, resolve_ctx=resolve_ctx)
    data = json.loads(out)
    names = [g["name"] for g in data]
    assert "some-other-vendor" in names
    assert "servedeck" in names
    other = next(g for g in data if g["name"] == "some-other-vendor")
    assert other["models"] == [{"id": "keep-me"}]


def test_vscode_replaces_legacy_local_llm_group_in_place(registry, resolve_ctx):
    existing = json.dumps(
        [
            {"name": "before", "vendor": "x", "apiType": "y", "models": []},
            {"name": "local_llm", "vendor": "customendpoint", "apiType": "chat-completions", "models": [{"id": "stale"}]},
            {"name": "after", "vendor": "x", "apiType": "y", "models": []},
        ]
    )
    out = wire.render_vscode(registry, existing, resolve_ctx=resolve_ctx)
    data = json.loads(out)
    names = [g["name"] for g in data]
    assert names == ["before", "servedeck", "after"]
    servedeck_group = data[1]
    assert all(e["id"] != "stale" for e in servedeck_group["models"])


def test_vscode_invalid_json_raises(registry, resolve_ctx):
    with pytest.raises(ValueError, match="not valid JSON"):
        wire.render_vscode(registry, "{not json", resolve_ctx=resolve_ctx)


def test_vscode_non_list_top_level_raises(registry, resolve_ctx):
    with pytest.raises(ValueError, match="JSON array"):
        wire.render_vscode(registry, json.dumps({"not": "a list"}), resolve_ctx=resolve_ctx)


def test_vscode_idempotent(registry, resolve_ctx):
    out1 = wire.render_vscode(registry, "", resolve_ctx=resolve_ctx)
    out2 = wire.render_vscode(registry, out1, resolve_ctx=resolve_ctx)
    assert out1 == out2


# --------------------------------------------------------------------------- #
# Codex
# --------------------------------------------------------------------------- #


def test_codex_empty_existing_adds_provider_and_profiles(registry, resolve_ctx):
    out = wire.render_codex(registry, "", resolve_ctx=resolve_ctx)
    parsed = tomllib.loads(out)
    assert parsed["model_providers"]["servedeck"]["base_url"] == "http://127.0.0.1:8010/v1"
    assert parsed["model_providers"]["servedeck"]["wire_api"] == "responses"
    assert set(parsed["profiles"]) == {"qwen27b", "flashnext", "glm53", "lfm2"}
    assert parsed["profiles"]["qwen27b"]["model"] == "Qwen3.8-27B-NVFP4"
    assert parsed["profiles"]["qwen27b"]["model_provider"] == "servedeck"


def test_codex_preserves_unrelated_provider_and_its_comments(registry, resolve_ctx):
    existing = (
        "# a hand-written comment explaining this provider\n"
        "[model_providers.glm53]\n"
        'name = "GLM-5.3-Flash (local)"\n'
        'base_url = "http://127.0.0.1:8003/v1"\n'
        'wire_api = "chat"\n'
    )
    out = wire.render_codex(registry, existing, resolve_ctx=resolve_ctx)
    assert "# a hand-written comment explaining this provider" in out
    assert '[model_providers.glm53]' in out
    assert 'base_url = "http://127.0.0.1:8003/v1"' in out
    parsed = tomllib.loads(out)
    assert parsed["model_providers"]["glm53"]["wire_api"] == "chat"
    assert "servedeck" in parsed["model_providers"]


def test_codex_rerun_replaces_in_place_without_duplicating(registry, resolve_ctx):
    out1 = wire.render_codex(registry, "", resolve_ctx=resolve_ctx)
    out2 = wire.render_codex(registry, out1, resolve_ctx=resolve_ctx)
    assert out1 == out2
    assert out2.count("[model_providers.servedeck]") == 1
    assert out2.count("[profiles.qwen27b]") == 1
    assert out2.count(wire._OWNED_COMMENT) == 1 + 4  # one provider + 4 profiles


def test_codex_updates_existing_servedeck_profile_when_registry_changes(registry, resolve_ctx):
    out1 = wire.render_codex(registry, "", resolve_ctx=resolve_ctx)
    # Change requires a fresh Model with a different max_output_tokens.
    import dataclasses

    changed = dataclasses.replace(registry.models["qwen27b"], max_output_tokens=1)
    new_models = dict(registry.models)
    new_models["qwen27b"] = changed
    changed_registry = dataclasses.replace(registry, models=new_models)
    out2 = wire.render_codex(changed_registry, out1, resolve_ctx=resolve_ctx)
    parsed = tomllib.loads(out2)
    assert parsed["profiles"]["qwen27b"]["model_max_output_tokens"] == 1


# --------------------------------------------------------------------------- #
# Kimi
# --------------------------------------------------------------------------- #


def test_kimi_empty_existing(registry, resolve_ctx):
    out = wire.render_kimi(registry, "", resolve_ctx=resolve_ctx)
    parsed = tomllib.loads(out)
    assert parsed["providers"]["servedeck"]["base_url"] == "http://127.0.0.1:8010/v1"
    assert parsed["models"]["servedeck/qwen27b"]["model"] == "Qwen3.8-27B-NVFP4"
    assert parsed["models"]["servedeck/qwen27b"]["max_context_size"] == 262144
    assert parsed["thinking"]["enabled"] is True


def test_kimi_preserves_unrelated_content_and_merges_thinking(registry, resolve_ctx):
    existing = (
        '[providers."managed:kimi-code"]\n'
        'type = "kimi"\n'
        'api_key = ""\n'
        "\n"
        '[models."kimi-code/kimi-for-coding"]\n'
        'provider = "managed:kimi-code"\n'
        'model = "kimi-for-coding"\n'
        "\n"
        "[thinking]\n"
        "enabled = false\n"
        'effort = "low"\n'
    )
    out = wire.render_kimi(registry, existing, resolve_ctx=resolve_ctx)
    parsed = tomllib.loads(out)
    assert parsed["providers"]["managed:kimi-code"]["type"] == "kimi"
    assert parsed["models"]["kimi-code/kimi-for-coding"]["model"] == "kimi-for-coding"
    # thinking.enabled flipped to true, but effort is preserved.
    assert parsed["thinking"]["enabled"] is True
    assert parsed["thinking"]["effort"] == "low"
    assert "servedeck" in parsed["providers"]
    assert "servedeck/lfm2" in parsed["models"]


def test_kimi_thinking_section_created_when_absent(registry, resolve_ctx):
    out = wire.render_kimi(registry, "", resolve_ctx=resolve_ctx)
    assert "[thinking]" in out
    assert tomllib.loads(out)["thinking"]["enabled"] is True


def test_kimi_capabilities_reflect_model_fields(registry, resolve_ctx):
    out = wire.render_kimi(registry, "", resolve_ctx=resolve_ctx)
    parsed = tomllib.loads(out)
    lfm2_caps = parsed["models"]["servedeck/lfm2"]["capabilities"]
    assert "tool_use" in lfm2_caps
    assert "thinking" not in lfm2_caps  # lfm2 has no reasoning parser
    qwen_caps = parsed["models"]["servedeck/qwen27b"]["capabilities"]
    assert set(qwen_caps) == {"tool_use", "thinking", "always_thinking", "image_in"}
    glm_caps = parsed["models"]["servedeck/glm53"]["capabilities"]
    assert "image_in" not in glm_caps  # glm53.vision = false


def test_kimi_rerun_replaces_in_place_without_duplicating(registry, resolve_ctx):
    out1 = wire.render_kimi(registry, "", resolve_ctx=resolve_ctx)
    out2 = wire.render_kimi(registry, out1, resolve_ctx=resolve_ctx)
    assert out1 == out2
    assert out2.count('[models."servedeck/qwen27b"]') == 1
    assert out2.count("[thinking]") == 1


# --------------------------------------------------------------------------- #
# unified_diff / read_existing / WIRE_TARGETS / make_default_ctx_resolver
# --------------------------------------------------------------------------- #


def test_unified_diff_empty_when_equal():
    assert wire.unified_diff("x", "same\n", "same\n") == ""


def test_unified_diff_shows_change():
    d = wire.unified_diff("x", "a\n", "b\n")
    assert "--- a/x" in d
    assert "+++ b/x" in d
    assert "-a" in d
    assert "+b" in d


def test_read_existing_missing_file_returns_empty(tmp_path):
    assert wire.read_existing(tmp_path / "nope.toml") == ""


def test_read_existing_reads_real_file(tmp_path):
    p = tmp_path / "f.toml"
    p.write_text("hello")
    assert wire.read_existing(p) == "hello"


def test_wire_targets_cover_all_three_clients():
    names = {t.name for t in wire.WIRE_TARGETS}
    assert len(wire.WIRE_TARGETS) == 3
    assert wire.VSCODE_CHAT_LM_PATH.name == "chatLanguageModels.json"
    assert wire.CODEX_CONFIG_PATH == Path.home() / ".codex" / "config.toml"
    assert wire.KIMI_CONFIG_PATH == Path.home() / ".kimi-code" / "config.toml"
    assert names  # sanity: non-empty, distinct names
    assert len({t.name for t in wire.WIRE_TARGETS}) == 3


def test_make_default_ctx_resolver_uses_hub_cache_for_native_ctx(tmp_path, monkeypatch, registry):
    """Hermetic: points the resolver at a FAKE hub dir under tmp_path via the
    env var registry.default_hub_dir() honours, never $HOME."""
    dirname = "models--" + registry.models["qwen27b"].repo.replace("/", "--")
    snap = tmp_path / dirname / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"max_position_embeddings": 999}))
    monkeypatch.setenv("SERVEDECK_HF_HUB_DIR", str(tmp_path))

    resolver = wire.make_default_ctx_resolver(registry)
    assert resolver(registry.models["qwen27b"]) == 999
    # An int ctx (lfm2) never touches the hub cache at all.
    assert resolver(registry.models["lfm2"]) == 32768


def test_make_default_ctx_resolver_caches_per_model(tmp_path, monkeypatch, registry):
    dirname = "models--" + registry.models["glm53"].repo.replace("/", "--")
    snap = tmp_path / dirname / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"max_position_embeddings": 12345}))
    monkeypatch.setenv("SERVEDECK_HF_HUB_DIR", str(tmp_path))

    resolver = wire.make_default_ctx_resolver(registry)
    m = registry.models["glm53"]
    assert resolver(m) == 12345
    # Delete the config; a cached resolver must not need to re-read it.
    (snap / "config.json").unlink()
    assert resolver(m) == 12345
