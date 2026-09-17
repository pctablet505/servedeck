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
    # glm53.ctx is now a pinned int (327680, not "native" — see models.toml),
    # so it needs no entry here: _fixed_resolver reads m.ctx directly for it.
    return _fixed_resolver({"qwen27b": 262144, "flashnext": 262144})


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
    # The slot first (the stable pick), then one per id and per preset:
    # 1 + 4 base ids + 3 glm53 presets = 8.
    assert ids[0] == "main", "the entry that survives a switch is the obvious pick"
    assert ids.count("Qwen3.8-27B-NVFP4") == 1
    assert "glm53-flash-low" in ids and "glm53-flash-high" in ids and "glm53-flash-max" in ids
    assert len(ids) == 8


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


def test_codex_points_a_bare_codex_at_the_main_slot(registry, resolve_ctx):
    """Reproduced against codex-cli 0.150.1 and 0.154.0-alpha.6.2 on
    2026-09-18: [profiles.<key>] tables are REFUSED ("move those settings into
    <home>/<name>.config.toml"), and with no top-level model/model_provider a
    bare `codex` silently used its cloud default — working, costing money, and
    leaving the local box idle while doctor reported codex OK."""
    out = wire.render_codex(registry, "", resolve_ctx=resolve_ctx)
    parsed = tomllib.loads(out)
    assert parsed["model_providers"]["servedeck"]["base_url"] == "http://127.0.0.1:8010/v1"
    assert parsed["model_providers"]["servedeck"]["wire_api"] == "responses"
    # The SLOT, not a model id: one model serves at a time, so a per-model
    # profile would be stale the moment the card is switched.
    assert parsed["model"] == "main"
    assert parsed["model_provider"] == "servedeck"
    assert parsed["model_context_window"] > 0
    assert "profiles" not in parsed, "the refused shape must not come back"
    assert "model_max_output_tokens" not in out, "neither installed binary knows that key"


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
    assert out2.count('model = "main"') == 1
    # Two owned blocks now: the top-level keys and the provider table.
    assert out2.count(wire._OWNED_COMMENT) == 2  # one provider + 4 profiles


def test_codex_keeps_hand_written_top_level_settings(registry, resolve_ctx):
    existing = (
        "# my own preferences\n"
        'approval_policy = "on-request"\n'
        'sandbox_mode = "workspace-write"\n'
    )
    out = wire.render_codex(registry, existing, resolve_ctx=resolve_ctx)
    parsed = tomllib.loads(out)
    assert parsed["approval_policy"] == "on-request"
    assert parsed["sandbox_mode"] == "workspace-write"
    assert parsed["model"] == "main"
    assert "# my own preferences" in out


def test_codex_context_window_follows_the_registry(registry, resolve_ctx):
    """The context Codex plans against must be the served one, not a guess:
    it is what its own auto-compaction is sized from."""
    import dataclasses

    out1 = wire.render_codex(registry, "", resolve_ctx=resolve_ctx)
    smaller = {m.key: dataclasses.replace(m, ctx=1024) for m in registry.models.values()}
    out2 = wire.render_codex(
        dataclasses.replace(registry, models=smaller), out1, resolve_ctx=lambda m: 1024
    )
    assert tomllib.loads(out1)["model_context_window"] != 1024
    assert tomllib.loads(out2)["model_context_window"] == 1024
    assert out2.count("model_context_window") == 1, "replaced in place, not appended"


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
    # flashnext.ctx is "native" (glm53's is now a pinned int, so it would never
    # touch the hub cache and couldn't exercise this path — see models.toml).
    dirname = "models--" + registry.models["flashnext"].repo.replace("/", "--")
    snap = tmp_path / dirname / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"max_position_embeddings": 12345}))
    monkeypatch.setenv("SERVEDECK_HF_HUB_DIR", str(tmp_path))

    resolver = wire.make_default_ctx_resolver(registry)
    m = registry.models["flashnext"]
    assert resolver(m) == 12345
    # Delete the config; a cached resolver must not need to re-read it.
    (snap / "config.json").unlink()
    assert resolver(m) == 12345


# --------------------------------------------------------------------------- #
# 2026-09-18 audit: the reasoning round trip, and wire eating comments
# --------------------------------------------------------------------------- #


def test_vscode_declares_thinking_for_a_reasoning_model(registry, resolve_ctx):
    """Without `thinking`, VS Code drops reasoning_content when it replays the
    previous turn and sends its own cot_summary/cot_id, which vLLM ignores —
    the model reads a multi-turn thread as fresh every turn. That erasure is
    the measured cause of agent amnesia here, and the gateway's output-side
    mirror cannot fix it."""
    out = json.loads(wire.render_vscode(registry, "", resolve_ctx=resolve_ctx))
    entries = {e["id"]: e for g in out if g["name"] == "servedeck" for e in g["models"]}
    thinker = registry.models["qwen27b"]
    assert thinker.reasoning is not None
    assert entries[thinker.id]["thinking"] is True
    assert entries[thinker.id]["contextWindow"] == resolve_ctx(thinker)
    plain = registry.models["lfm2"]
    assert plain.reasoning is None
    assert entries[plain.id]["thinking"] is False, "LFM2 has no thinking mode to declare"


def test_wire_does_not_eat_a_comment_between_two_tables(registry, resolve_ctx):
    """Every `wire --apply` silently deleted the operator's own notes: an
    owned table's span ran to the next "[", absorbing any comment below it."""
    existing = (
        "[providers.servedeck]\n"
        'base_url = "http://127.0.0.1:8010/v1"\n'
        "\n"
        "# my own note about the model below — do not delete\n"
        '[models."something/else"]\n'
        'provider = "elsewhere"\n'
    )
    out = wire.render_kimi(registry, existing, resolve_ctx=resolve_ctx)
    assert "# my own note about the model below — do not delete" in out
    assert '[models."something/else"]' in out
    assert 'provider = "elsewhere"' in out


def test_kimi_default_follows_the_slot_not_a_model(registry, resolve_ctx):
    """A per-model default is stale the moment the card is switched, and the
    owner then edits three client configs to get a local model back."""
    out = wire.render_kimi(registry, 'default_model = "servedeck/flashnext"\n', resolve_ctx=resolve_ctx)
    parsed = tomllib.loads(out)
    assert parsed["default_model"] == "servedeck/main"
    entry = parsed["models"]["servedeck/main"]
    assert entry["model"] == "main" and entry["provider"] == "servedeck"
    assert "thinking" in entry["capabilities"], "a reasoning model may hold the slot"


def test_kimi_keeps_a_cloud_default_the_operator_chose(registry, resolve_ctx):
    out = wire.render_kimi(registry, 'default_model = "kimi-code/k3"\n', resolve_ctx=resolve_ctx)
    assert tomllib.loads(out)["default_model"] == "kimi-code/k3"
    assert 'models."servedeck/main"' in out, "the entry is still offered"


def test_codex_retires_the_profile_tables_it_used_to_write(registry, resolve_ctx):
    """Upsert-only could not retire anything, and the [profiles.<key>] tables
    written before 2026-09-18 are the exact reason Codex refuses --profile."""
    legacy = (
        "# servedeck-generated by `servedeck wire` — edits here are overwritten.\n"
        "[profiles.qwen27b]\n"
        'model = "Qwen3.8-27B-NVFP4"\n'
        "\n"
        "# servedeck-generated by `servedeck wire` — edits here are overwritten.\n"
        "[profiles.flashnext]\n"
        'model_provider = "servedeck"\n'
        'model = "Qwen3.8-Flash-Next-Uncensored-NVFP4"\n'
        "\n"
        "# a profile the operator wrote by hand\n"
        "[profiles.mine]\n"
        'model = "something-else"\n'
    )
    out = wire.render_codex(registry, legacy, resolve_ctx=resolve_ctx)
    parsed = tomllib.loads(out)
    assert "flashnext" not in parsed.get("profiles", {}), "ours is retired"
    assert "qwen27b" not in parsed.get("profiles", {}), "every one of ours, not alternate ones"
    assert parsed["profiles"]["mine"]["model"] == "something-else", "theirs is not"
    assert "# a profile the operator wrote by hand" in out


def test_the_slot_advertises_the_narrowest_main_context(registry, resolve_ctx):
    """The slot may be holding any main model. A client that sized a prompt
    against GLM's 327,680 while Flash-Next (262,144) is serving gets the
    request rejected by the engine, so the smallest is the safe figure."""
    mains = [m for m in registry.models.values() if m.slot == "main"]
    smallest = min(resolve_ctx(m) for m in mains)
    assert smallest < max(resolve_ctx(m) for m in mains), "fixture needs two widths"
    codex = tomllib.loads(wire.render_codex(registry, "", resolve_ctx=resolve_ctx))
    assert codex["model_context_window"] == smallest
    kimi = tomllib.loads(wire.render_kimi(registry, "", resolve_ctx=resolve_ctx))
    assert kimi["models"]["servedeck/main"]["max_context_size"] == smallest
    vscode = json.loads(wire.render_vscode(registry, "", resolve_ctx=resolve_ctx))
    entry = next(e for g in vscode for e in g["models"] if e["id"] == "main")
    assert entry["contextWindow"] == smallest
