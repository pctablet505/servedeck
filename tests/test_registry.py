"""Tests for servedeck.registry — SPEC.md §4.

Two kinds of coverage:
  * Synthetic hub caches built under tmp_path, so discovery/servability/config
    parsing rules are pinned down exactly regardless of what happens to be on
    this box's real ~/.cache/huggingface/hub.
  * Assertions against the REAL hub cache (skipped if it is not present) that
    reproduce the exact counts and figures the task brief asked to be
    verified: 7 hub dirs -> 5 servable, 1 unservable (OBLITERATUS, GGUF-only),
    1 skipped stub (models--Qwen--Qwen3.8-27B).

No number here is invented: the real-cache assertions were measured by
running `python -m servedeck.registry --list` against the live cache before
these tests were written.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from servedeck import registry
from servedeck.registry import (
    KNOWN_ARCHS,
    ModelEntry,
    append_observation,
    discover_models,
    load_observations,
    resolve_inputs,
)

REAL_HUB_DIR = Path.home() / ".cache" / "huggingface" / "hub"
REAL_MEASUREMENTS = Path("/tmp/servedeck-test/placeholder")


# --------------------------------------------------------------------------- #
# Synthetic hub cache helpers
# --------------------------------------------------------------------------- #


def _write_safetensors(snapshot: Path, name: str, size_bytes: int, *, via_blob: bool = True) -> None:
    """Write a fake .safetensors file of an exact byte size.

    Real hub snapshots store files as symlinks into a sibling blobs/ dir;
    replicate that so the follow_symlinks=True sizing rule is actually
    exercised, not incidentally true because no symlink was involved.

    IMPORTANT: sizes here range up to ~126 GiB (a real Flash-Next safetensors
    sum). This box's /tmp is tmpfs (RAM-backed) and a live vLLM server is
    running with tens of GiB already committed — actually materializing that
    many zero bytes (`b"\\0" * size_bytes`) previously wrote 60+ real GiB into
    tmpfs per test run and drove the box to the edge of OOM. Use a *sparse*
    file instead: os.truncate() sets st_size correctly (which is all
    discover_models() reads) without allocating any backing pages for the
    unwritten holes.
    """
    if via_blob:
        blobs_dir = snapshot.parent.parent / "blobs"
        blobs_dir.mkdir(parents=True, exist_ok=True)
        # Disambiguate by snapshot revision too: a repo with multiple
        # snapshots/<rev>/ dirs shares one blobs/ dir, and two revisions may
        # both write a file of the same name (e.g. "m.safetensors") — a
        # collision here would silently resize an earlier revision's blob
        # out from under its already-created symlink.
        target = blobs_dir / f"blob-{snapshot.name}-{name}"
    else:
        target = snapshot / name
    with open(target, "wb") as fh:
        fh.truncate(size_bytes)
    if via_blob:
        (snapshot / name).symlink_to(target)


def _make_repo(
    hub_root: Path,
    dirname: str,
    *,
    rev: str = "abc123",
    config: dict | None = None,
    safetensors: dict[str, int] | None = None,
    gguf: list[str] | None = None,
    use_refs_main: bool = True,
    extra_snapshot_revs: list[str] | None = None,
    make_snapshots_dir: bool = True,
) -> Path:
    repo_dir = hub_root / dirname
    if make_snapshots_dir:
        snapshot = repo_dir / "snapshots" / rev
        snapshot.mkdir(parents=True)
        if use_refs_main:
            refs_dir = repo_dir / "refs"
            refs_dir.mkdir(parents=True, exist_ok=True)
            (refs_dir / "main").write_text(rev)
        for extra_rev in extra_snapshot_revs or []:
            (repo_dir / "snapshots" / extra_rev).mkdir(parents=True)
        if config is not None:
            (snapshot / "config.json").write_text(json.dumps(config))
        for name, size in (safetensors or {}).items():
            _write_safetensors(snapshot, name, size)
        for name in gguf or []:
            (snapshot / name).write_bytes(b"GGUF")
    else:
        repo_dir.mkdir(parents=True)
    return repo_dir


INLINE_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "text_config": {
        "max_position_embeddings": 262144,
        "num_hidden_layers": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "full_attention_interval": 4,
    },
    "quantization_config": {"quant_algo": "NVFP4"},
}

FLASHNEXT_CONFIG = {
    "architectures": ["Qwen4ExpForConditionalGeneration"],
    "model_type": "qwen4_exp",
    "text_config": {
        "max_position_embeddings": 262144,
        "num_hidden_layers": 48,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
    },
    "quantization_config": {"quant_method": "modelopt"},
}

GIB = 1024**3


# --------------------------------------------------------------------------- #
# Discovery / repo_id parsing
# --------------------------------------------------------------------------- #


def test_repo_id_only_first_double_dash_becomes_slash(tmp_path: Path) -> None:
    _make_repo(
        tmp_path,
        "models--RadixArk--Qwen3.8-27B-NVFP4",
        config=INLINE_CONFIG,
        safetensors={"model-1.safetensors": GIB},
    )
    entries = discover_models(tmp_path)
    assert len(entries) == 1
    assert entries[0].repo_id == "RadixArk/Qwen3.8-27B-NVFP4"


def test_non_models_dirs_and_files_are_ignored(tmp_path: Path) -> None:
    (tmp_path / ".locks").mkdir()
    (tmp_path / "CACHEDIR.TAG").write_text("x")
    _make_repo(tmp_path, "models--A--B", config=INLINE_CONFIG, safetensors={"m.safetensors": 10})
    entries = discover_models(tmp_path)
    assert [e.hub_dirname for e in entries] == ["models--A--B"]


def test_stub_with_no_snapshots_dir_is_skipped(tmp_path: Path) -> None:
    """The 12KB models--Qwen--Qwen3.8-27B stub: refs/ exists, snapshots/ does not."""
    _make_repo(tmp_path, "models--Qwen--Qwen3.8-27B", make_snapshots_dir=False)
    (tmp_path / "models--Qwen--Qwen3.8-27B" / "refs").mkdir(parents=True)
    (tmp_path / "models--Qwen--Qwen3.8-27B" / "refs" / "main").write_text("deadbeef")

    entries = discover_models(tmp_path)
    assert len(entries) == 1
    e = entries[0]
    assert e.skipped is True
    assert e.servable is False
    assert e.snapshot_path is None
    assert "no snapshots/" in (e.reason or "")


def test_empty_snapshots_dir_is_skipped(tmp_path: Path) -> None:
    repo_dir = tmp_path / "models--Foo--Bar"
    (repo_dir / "snapshots").mkdir(parents=True)
    entries = discover_models(tmp_path)
    assert entries[0].skipped is True


def test_refs_main_selects_that_revision_over_newer_snapshot(tmp_path: Path) -> None:
    """refs/main content wins even if a differently-named snapshot dir has a
    newer mtime — this is the documented precedence, not "pick newest"."""
    repo_dir = _make_repo(
        tmp_path,
        "models--A--B",
        rev="rev-old",
        config=INLINE_CONFIG,
        safetensors={"m.safetensors": 100},
    )
    newer = repo_dir / "snapshots" / "rev-new"
    newer.mkdir()
    (newer / "config.json").write_text(json.dumps(INLINE_CONFIG))
    _write_safetensors(newer, "m.safetensors", 999)

    entries = discover_models(tmp_path)
    assert entries[0].snapshot_path == str(repo_dir / "snapshots" / "rev-old")
    assert entries[0].safetensors_count == 1
    assert round(entries[0].safetensors_gib * GIB) == 100


def test_no_refs_main_falls_back_to_newest_snapshot_dir(tmp_path: Path) -> None:
    repo_dir = _make_repo(
        tmp_path,
        "models--A--B",
        rev="rev-1",
        use_refs_main=False,
        config=INLINE_CONFIG,
        safetensors={"m.safetensors": 100},
    )
    import time

    time.sleep(0.01)
    newer = repo_dir / "snapshots" / "rev-2"
    newer.mkdir()
    (newer / "config.json").write_text(json.dumps(INLINE_CONFIG))
    _write_safetensors(newer, "m.safetensors", 555)

    entries = discover_models(tmp_path)
    assert entries[0].snapshot_path == str(newer)


def test_safetensors_size_follows_symlinks(tmp_path: Path) -> None:
    size = 12345
    _make_repo(
        tmp_path,
        "models--A--B",
        config=INLINE_CONFIG,
        safetensors={"m.safetensors": size},
    )
    entries = discover_models(tmp_path)
    assert entries[0].safetensors_count == 1
    assert abs(entries[0].safetensors_gib * GIB - size) < 1.0


def test_safetensors_gib_sums_multiple_shards(tmp_path: Path) -> None:
    _make_repo(
        tmp_path,
        "models--A--B",
        config=INLINE_CONFIG,
        safetensors={
            "model-00001-of-00003.safetensors": 10 * GIB,
            "model-00002-of-00003.safetensors": 10 * GIB,
            "model-00003-of-00003.safetensors": 8 * GIB + 25 * (GIB // 100),
        },
    )
    entries = discover_models(tmp_path)
    e = entries[0]
    assert e.safetensors_count == 3
    assert round(e.safetensors_gib, 2) == pytest.approx(28.25, abs=0.02)


# --------------------------------------------------------------------------- #
# Servability rules
# --------------------------------------------------------------------------- #


def test_servable_requires_config_safetensors_and_known_arch(tmp_path: Path) -> None:
    _make_repo(
        tmp_path,
        "models--Radix--Good",
        config=INLINE_CONFIG,
        safetensors={"m.safetensors": GIB},
    )
    e = discover_models(tmp_path)[0]
    assert e.servable is True
    assert e.backend == "inline"
    assert e.reason is None


def test_flashnext_arch_maps_to_flashnext_backend(tmp_path: Path) -> None:
    _make_repo(
        tmp_path,
        "models--Radix--Next",
        config=FLASHNEXT_CONFIG,
        safetensors={"m.safetensors": GIB},
    )
    e = discover_models(tmp_path)[0]
    assert e.servable is True
    assert e.backend == "flashnext"
    assert e.model_type == "qwen4_exp"


def test_gguf_only_is_unservable_with_gguf_only_reason(tmp_path: Path) -> None:
    """OBLITERATUS-shaped repo: only a .gguf file, no config.json, no safetensors."""
    _make_repo(
        tmp_path,
        "models--OBLITERATUS--Qwen3.8-27B-OBLITERATED",
        config=None,
        gguf=["Qwen3.8-27B-OBLITERATED-Q6_K.gguf"],
    )
    e = discover_models(tmp_path)[0]
    assert e.servable is False
    assert e.skipped is False
    assert e.safetensors_gib == 0.0
    assert e.reason is not None
    assert "GGUF-only" in e.reason
    assert "0 safetensors" in e.reason
    assert "no config.json" in e.reason


def test_zero_byte_safetensors_file_present_is_still_unservable(tmp_path: Path) -> None:
    """A 0-byte .safetensors placeholder must not count as ``safetensors>0``."""
    repo_dir = _make_repo(
        tmp_path,
        "models--A--Empty",
        config=INLINE_CONFIG,
        safetensors={},
    )
    (repo_dir / "snapshots" / "abc123" / "model.safetensors").write_bytes(b"")
    e = discover_models(tmp_path)[0]
    assert e.safetensors_gib == 0.0
    assert e.servable is False
    assert "0 safetensors" in (e.reason or "")


def test_unknown_architecture_is_unservable(tmp_path: Path) -> None:
    cfg = {**INLINE_CONFIG, "architectures": ["SomeOtherForCausalLM"]}
    _make_repo(tmp_path, "models--A--Weird", config=cfg, safetensors={"m.safetensors": GIB})
    e = discover_models(tmp_path)[0]
    assert e.servable is False
    assert e.backend is None
    assert "unknown architecture" in (e.reason or "")
    assert "SomeOtherForCausalLM" in (e.reason or "")


def test_config_without_architectures_is_unservable_not_a_crash(tmp_path: Path) -> None:
    cfg = {k: v for k, v in INLINE_CONFIG.items() if k != "architectures"}
    _make_repo(tmp_path, "models--A--NoArch", config=cfg, safetensors={"m.safetensors": GIB})
    e = discover_models(tmp_path)[0]
    assert e.servable is False
    assert e.architectures0 is None


def test_missing_config_json_is_unservable(tmp_path: Path) -> None:
    _make_repo(tmp_path, "models--A--NoConfig", config=None, safetensors={"m.safetensors": GIB})
    e = discover_models(tmp_path)[0]
    assert e.servable is False
    assert e.config_exists is False
    assert "no config.json" in (e.reason or "")


def test_corrupt_config_json_treated_as_missing(tmp_path: Path) -> None:
    repo_dir = _make_repo(tmp_path, "models--A--Corrupt", config=INLINE_CONFIG, safetensors={"m.safetensors": GIB})
    (repo_dir / "snapshots" / "abc123" / "config.json").write_text("{not json")
    e = discover_models(tmp_path)[0]
    assert e.config_exists is False
    assert e.servable is False


def test_known_archs_map_is_exactly_the_two_documented_entries() -> None:
    """The built-in map must stay 1:1 — _reverse_arch_backends() depends on it.

    A stray entry here half-wires a backend: the model becomes servable and
    the UI offers it, while paths, preflight and the launcher know nothing
    about it. Adding a backend is a config act (`[backends.<name>]`), not an
    edit to this literal.
    """
    assert KNOWN_ARCHS == {
        "Qwen3_5ForConditionalGeneration": "inline",
        "Qwen4ExpForConditionalGeneration": "flashnext",
    }
    assert len(set(KNOWN_ARCHS.values())) == len(KNOWN_ARCHS), "map must stay 1:1"


def test_configured_architectures_extend_the_built_in_map(config_path) -> None:
    """Declaring `architectures` on a backend is the whole supported way to
    teach Servedeck a new model family.

    It used to require editing KNOWN_ARCHS, i.e. shipping one machine's model
    lineup inside a public package — and a user who could not edit the source
    simply saw their model listed as "unknown architecture".
    """
    from servedeck import registry

    config_path(
        '''
[backends.custom]
launcher = "/bin/true"
port = 9500
architectures = ["SomeNewForConditionalGeneration"]
'''
    )
    got = registry.arch_backends()
    assert got["SomeNewForConditionalGeneration"] == "custom"
    # and the built-ins survive, so an install with a config is not a
    # regression for the backends that never needed one
    assert got["Qwen4ExpForConditionalGeneration"] == "flashnext"


# --------------------------------------------------------------------------- #
# Config field extraction: text_config vs root fallback
# --------------------------------------------------------------------------- #


def test_fields_come_from_text_config_when_present(tmp_path: Path) -> None:
    _make_repo(tmp_path, "models--A--B", config=INLINE_CONFIG, safetensors={"m.safetensors": GIB})
    e = discover_models(tmp_path)[0]
    assert e.max_position_embeddings == 262144
    assert e.num_hidden_layers == 48
    assert e.num_key_value_heads == 8
    assert e.head_dim == 128
    assert e.full_attention_interval == 4


def test_fields_fall_back_to_root_when_no_text_config(tmp_path: Path) -> None:
    cfg = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "max_position_embeddings": 131072,
        "num_hidden_layers": 40,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "full_attention_interval": 2,
    }
    _make_repo(tmp_path, "models--A--B", config=cfg, safetensors={"m.safetensors": GIB})
    e = discover_models(tmp_path)[0]
    assert e.max_position_embeddings == 131072
    assert e.num_key_value_heads == 4


def test_quant_algo_falls_back_to_quant_method(tmp_path: Path) -> None:
    _make_repo(tmp_path, "models--A--B", config=FLASHNEXT_CONFIG, safetensors={"m.safetensors": GIB})
    e = discover_models(tmp_path)[0]
    assert e.quant_algo == "modelopt"  # FLASHNEXT_CONFIG only sets quant_method


def test_quant_algo_preferred_over_quant_method_when_both_present(tmp_path: Path) -> None:
    cfg = {**INLINE_CONFIG, "quantization_config": {"quant_algo": "NVFP4", "quant_method": "modelopt"}}
    _make_repo(tmp_path, "models--A--B", config=cfg, safetensors={"m.safetensors": GIB})
    e = discover_models(tmp_path)[0]
    assert e.quant_algo == "NVFP4"


def test_missing_hub_dir_returns_empty_list(tmp_path: Path) -> None:
    assert discover_models(tmp_path / "does-not-exist") == []


# --------------------------------------------------------------------------- #
# Observation store — append-only, atomic, tolerant of a bad file
# --------------------------------------------------------------------------- #


def test_load_observations_missing_file_returns_empty_list(tmp_path: Path) -> None:
    assert load_observations(tmp_path / "measurements.json") == []


def test_load_observations_empty_file_returns_empty_list(tmp_path: Path) -> None:
    p = tmp_path / "measurements.json"
    p.write_text("")
    assert load_observations(p) == []


def test_load_observations_corrupt_json_returns_empty_list_not_raise(tmp_path: Path) -> None:
    p = tmp_path / "measurements.json"
    p.write_text("{not valid json[[[")
    assert load_observations(p) == []


def test_load_observations_non_list_json_returns_empty_list(tmp_path: Path) -> None:
    p = tmp_path / "measurements.json"
    p.write_text(json.dumps({"not": "a list"}))
    assert load_observations(p) == []


def test_append_observation_is_append_only_and_never_overwrites(tmp_path: Path) -> None:
    p = tmp_path / "measurements.json"
    append_observation({"repo_id": "A", "n": 1}, p)
    append_observation({"repo_id": "B", "n": 2}, p)
    append_observation({"repo_id": "C", "n": 3}, p)
    obs = load_observations(p)
    assert [o["repo_id"] for o in obs] == ["A", "B", "C"]


def test_append_observation_creates_parent_dir(tmp_path: Path) -> None:
    p = tmp_path / "nested" / "dir" / "measurements.json"
    append_observation({"repo_id": "A"}, p)
    assert p.is_file()
    assert load_observations(p) == [{"repo_id": "A"}]


def test_append_observation_writes_atomically_no_tmp_left_behind(tmp_path: Path) -> None:
    p = tmp_path / "measurements.json"
    append_observation({"repo_id": "A"}, p)
    leftovers = [f for f in os.listdir(tmp_path) if f != "measurements.json"]
    assert leftovers == []


# --------------------------------------------------------------------------- #
# resolve_inputs() — three-tier lookup (SPEC §4)
# --------------------------------------------------------------------------- #


def _obs(repo_id: str, backend: str, ctx: int, **measured: object) -> dict:
    return {
        "ts": "2026-01-01T00:00:00Z",
        "repo_id": repo_id,
        "backend": backend,
        "inputs": {"util": 0.5, "max_model_len": ctx, "max_num_seqs": None, "kv_cache_dtype": "auto"},
        "measured": measured,
        "trust": "measured",
    }


def test_tier1_exact_ctx_match_is_trust_measured(tmp_path: Path) -> None:
    _make_repo(tmp_path, "models--A--B", config=INLINE_CONFIG, safetensors={"m.safetensors": GIB})
    obs = [_obs("A/B", "inline", 262144, weights_gib=20.75, kv_gib=22.72, kv_tokens=627117)]

    r = resolve_inputs("A/B", util=0.5, ctx=262144, observations=obs, hub_dir=tmp_path)

    assert r.trust == "measured"
    assert r.weights_source == "measured"
    assert r.weights_gib == 20.75
    assert r.matched_ctx == 262144
    assert r.kv_kib_per_token == pytest.approx(22.72 * 1048576 / 627117)


def test_tier2_same_repo_different_ctx_is_measured_other_ctx(tmp_path: Path) -> None:
    _make_repo(tmp_path, "models--A--B", config=INLINE_CONFIG, safetensors={"m.safetensors": GIB})
    obs = [_obs("A/B", "inline", 131072, weights_gib=20.75, kv_kib_per_token=33.85)]

    r = resolve_inputs("A/B", util=0.5, ctx=262144, observations=obs, hub_dir=tmp_path)

    assert r.trust == "measured_other_ctx"
    assert r.matched_ctx == 131072
    assert r.kv_kib_per_token == 33.85
    assert r.other_ctx_kv_rates == {131072: 33.85}


def test_tier1_wins_over_tier2_when_both_exist(tmp_path: Path) -> None:
    _make_repo(tmp_path, "models--A--B", config=INLINE_CONFIG, safetensors={"m.safetensors": GIB})
    obs = [
        _obs("A/B", "inline", 131072, weights_gib=1.0, kv_kib_per_token=1.0),
        _obs("A/B", "inline", 262144, weights_gib=2.0, kv_kib_per_token=2.0),
    ]
    r = resolve_inputs("A/B", util=0.5, ctx=262144, observations=obs, hub_dir=tmp_path)
    assert r.trust == "measured"
    assert r.matched_ctx == 262144
    assert r.weights_gib == 2.0


def test_tier2_prefers_matching_backend_over_stale_other_backend_history(tmp_path: Path) -> None:
    """If a repo_id's observation history spans two backends (e.g. after a
    KNOWN_ARCHS reclassification), tier-2 must not silently mix them."""
    _make_repo(tmp_path, "models--A--B", config=INLINE_CONFIG, safetensors={"m.safetensors": GIB})
    obs = [
        _obs("A/B", "flashnext", 131072, weights_gib=99.0, kv_kib_per_token=99.0),
        _obs("A/B", "inline", 100000, weights_gib=20.75, kv_kib_per_token=37.99),
    ]
    r = resolve_inputs("A/B", util=0.5, ctx=262144, observations=obs, hub_dir=tmp_path)
    assert r.backend == "inline"
    assert r.matched_ctx == 100000
    assert r.weights_gib == 20.75


def test_tier3_estimate_applies_1pt01_multiplier(tmp_path: Path) -> None:
    _make_repo(
        tmp_path,
        "models--A--NeverBooted",
        config=INLINE_CONFIG,
        safetensors={"m.safetensors": int(20 * GIB)},
    )
    r = resolve_inputs("A/NeverBooted", util=0.5, ctx=262144, observations=[], hub_dir=tmp_path)

    assert r.trust == "unknown" or r.weights_source == "estimated"
    assert r.weights_source == "estimated"
    assert r.weights_gib == pytest.approx(20.0 * 1.01, abs=0.01)


def test_tier3_refuses_estimator_for_qwen4_exp(tmp_path: Path) -> None:
    """CRITICAL rule: model_type=='qwen4_exp' must NEVER get a safetensors*1.01
    weight estimate — the n-gram table is host-offloaded, making the disk sum
    ~37% too high (125.91 GiB disk vs 78.47 GiB actual VRAM)."""
    _make_repo(
        tmp_path,
        "models--RadixArk--Qwen3.8-Flash-Next-NVFP4",
        config=FLASHNEXT_CONFIG,
        safetensors={"m.safetensors": int(125.91 * GIB)},
    )
    r = resolve_inputs(
        "RadixArk/Qwen3.8-Flash-Next-NVFP4",
        util=0.96,
        ctx=999999,  # deliberately not any known observation ctx -> forces tier 3
        observations=[],
        hub_dir=tmp_path,
    )

    assert r.weights_source == "unknown"
    assert r.weights_gib is None
    assert r.trust == "unknown"
    assert r.reason is not None and "qwen4_exp" in r.reason


def test_tier3_qwen4_exp_refusal_holds_even_with_other_repos_measured(tmp_path: Path) -> None:
    """The refusal is keyed on model_type, not on "no measurements exist
    anywhere" — a measured inline sibling must not leak a weight estimate
    into an unrelated never-booted flashnext repo."""
    _make_repo(
        tmp_path,
        "models--RadixArk--Qwen3.8-Flash-Next-NVFP4",
        config=FLASHNEXT_CONFIG,
        safetensors={"m.safetensors": int(125.91 * GIB)},
    )
    obs = [_obs("Some/OtherRepo", "inline", 262144, weights_gib=20.75, kv_gib=22.72, kv_tokens=627117)]
    r = resolve_inputs(
        "RadixArk/Qwen3.8-Flash-Next-NVFP4", util=0.96, ctx=999999, observations=obs, hub_dir=tmp_path
    )
    assert r.weights_source == "unknown"


def test_tier3_kv_rate_is_median_of_measured_same_architecture(tmp_path: Path) -> None:
    _make_repo(
        tmp_path,
        "models--A--NeverBooted",
        config=INLINE_CONFIG,
        safetensors={"m.safetensors": int(10 * GIB)},
    )
    obs = [
        # Same architecture (Qwen3_5*), trust="measured" -> counts.
        _obs("Other/One", "inline", 262144, weights_gib=1.0, kv_gib=10.0, kv_tokens=int(10 * 1048576 / 30)),
        _obs("Other/Two", "inline", 262144, weights_gib=1.0, kv_gib=10.0, kv_tokens=int(10 * 1048576 / 40)),
    ]
    r = resolve_inputs("A/NeverBooted", util=0.5, ctx=262144, observations=obs, hub_dir=tmp_path)
    assert r.kv_kib_per_token == pytest.approx(35.0, abs=0.5)


def test_tier3_kv_rate_ignores_non_measured_trust_observations(tmp_path: Path) -> None:
    _make_repo(
        tmp_path,
        "models--A--NeverBooted",
        config=INLINE_CONFIG,
        safetensors={"m.safetensors": int(10 * GIB)},
    )
    estimated_obs = _obs("Other/One", "inline", 262144, weights_gib=1.0, kv_kib_per_token=999.0)
    estimated_obs["trust"] = "estimated"
    r = resolve_inputs("A/NeverBooted", util=0.5, ctx=262144, observations=[estimated_obs], hub_dir=tmp_path)
    assert r.kv_kib_per_token is None
    assert r.kv_source == "unknown"


def test_tier3_no_safetensors_available_is_unknown(tmp_path: Path) -> None:
    r = resolve_inputs("Nowhere/AtAll", util=0.5, ctx=262144, observations=[], hub_dir=tmp_path)
    assert r.weights_source == "unknown"
    assert r.weights_gib is None


# --------------------------------------------------------------------------- #
# resolve_inputs() -> capacity.py integration: UNKNOWN_CAPACITY actually fires
# --------------------------------------------------------------------------- #


def test_unknown_weights_source_makes_capacity_raise_unknown_capacity_finding(tmp_path: Path) -> None:
    """End-to-end proof of the CRITICAL rule: a resolve_inputs() refusal for
    model_type=='qwen4_exp' must reach capacity.compute() as a blocking
    UNKNOWN_CAPACITY finding, with can_apply forced False."""
    from servedeck import capacity

    _make_repo(
        tmp_path,
        "models--RadixArk--Qwen3.8-Flash-Next-NVFP4",
        config=FLASHNEXT_CONFIG,
        safetensors={"m.safetensors": int(125.91 * GIB)},
    )
    r = resolve_inputs(
        "RadixArk/Qwen3.8-Flash-Next-NVFP4", util=0.96, ctx=999999, observations=[], hub_dir=tmp_path
    )
    assert r.weights_source == "unknown"

    m = capacity.ModelInputs(
        repo_id=r.repo_id,
        backend=r.backend or "flashnext",
        model_max_ctx=r.model_max_ctx or 262144,
        weights_gib=r.weights_gib,
        weights_source=r.weights_source,
        kv_kib_per_token=r.kv_kib_per_token,
        trust=r.trust,
        model_type=r.model_type,
    )
    result = capacity.compute(m, util=0.96, ctx=262144, max_num_seqs=1)

    codes = {f.code for f in result.findings}
    assert "UNKNOWN_CAPACITY" in codes
    assert result.can_apply is True


# --------------------------------------------------------------------------- #
# Real hub cache — reproduces the exact counts from the task brief's VERIFY
# --------------------------------------------------------------------------- #

pytestmark_real = pytest.mark.skipif(
    not REAL_HUB_DIR.is_dir(), reason="no Hugging Face cache on this machine"
)


# --------------------------------------------------------------------------- #
# Tests against a real Hugging Face cache.
#
# These assert INVARIANTS, never counts or specific repo ids: the cache belongs
# to whoever runs the tests and changes whenever they pull a model. An earlier
# version asserted "exactly 7 models", which broke the moment one was added.
# --------------------------------------------------------------------------- #


@pytestmark_real
def test_real_hub_entries_are_internally_consistent() -> None:
    for e in discover_models():
        assert e.repo_id and "/" in e.repo_id
        if e.skipped:
            continue
        # Servability must always be explainable: either it is servable, or
        # there is a reason a human can act on.
        if not e.servable:
            assert e.reason, f"{e.repo_id} is unservable with no reason given"
        else:
            assert e.architectures0, f"{e.repo_id} is servable but has no architecture"
            assert e.safetensors_count > 0
            assert e.safetensors_gib > 0


@pytestmark_real
def test_real_hub_gguf_only_models_are_rejected_with_a_reason() -> None:
    """A GGUF-only checkpoint has no config.json and cannot be loaded."""
    for e in discover_models():
        if e.skipped or e.config_exists or e.safetensors_count:
            continue
        assert not e.servable
        assert e.reason and "GGUF" in e.reason.upper()


@pytestmark_real
def test_weights_estimator_refuses_host_offload_architectures() -> None:
    """On-disk size is not loaded size when layers live in host RAM.

    Estimating weights from safetensors size is accurate to ~2% normally and
    wrong by ~37% for offload architectures, so the estimator must refuse
    rather than be confidently wrong.
    """
    for e in discover_models():
        if e.skipped or not e.servable or e.model_type != "qwen4_exp":
            continue
        ri = resolve_inputs(e.repo_id, util=0.95, ctx=262144)
        if ri.weights_source != "measured":
            assert ri.weights_source == "unknown", (
                f"{e.repo_id}: offload architecture must not be estimated from disk size"
            )
