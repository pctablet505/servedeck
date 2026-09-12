"""servedeck.models — registry load/validation, resolve(), rendering, arithmetic.

Every validator rule gets its own failing fixture (P1 deliverable 7: "Every
check in the validator gets a failing fixture"). All I/O is under tmp_path; the
real ``models.toml`` at the repo root is exercised separately by
``tests/test_models_golden.py`` and by ``test_real_models_toml_loads`` below.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from servedeck import models

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "models.toml"
    p.write_text(body)
    return p


BASE_GPU = """
[gpu]
total_mib = 100000
margin_mib = 1000

[builds]
stock = "/opt/stock"
"""


def _one_model(
    key: str = "a",
    *,
    id_: str = "A",
    repo: str = "org/a",
    slot: str = "main",
    port: int = 9001,
    build: str = "stock",
    ctx: int = 1000,
    extra: str = "",
) -> str:
    return f"""
[models.{key}]
id = "{id_}"
repo = "{repo}"
slot = "{slot}"
port = {port}
build = "{build}"
ctx = {ctx}
{extra}
"""


def _fake_hub(tmp_path: Path, repo_id: str, config: dict) -> Path:
    """<tmp_path>/models--Org--Name/snapshots/rev1/config.json, hub_dir=tmp_path."""
    dirname = "models--" + repo_id.replace("/", "--")
    snap = tmp_path / dirname / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps(config))
    return tmp_path


# --------------------------------------------------------------------------- #
# load(): the happy path
# --------------------------------------------------------------------------- #


def test_load_minimal_registry(tmp_path):
    p = _write(tmp_path, BASE_GPU + _one_model())
    reg = models.load(p)
    assert set(reg.models) == {"a"}
    m = reg.models["a"]
    assert m.id == "A"
    assert m.repo == "org/a"
    assert m.port == 9001
    assert m.aliases == ()
    assert m.reasoning is None
    assert m.tools is None
    assert m.flags == ()
    assert m.presets == {}


def test_load_full_model_fields(tmp_path):
    body = (
        BASE_GPU
        + _one_model(
            extra="""
aliases = ["a1", "a2"]
flags = ["--foo", "1"]
vision = true
max_output_tokens = 100
min_output_tokens = 10
needs_tty = true

[models.a.reasoning]
parser = "qwen3"
mirror_content = true

[models.a.tools]
parser = "qwen3_xml"

[models.a.presets]
a-low = { reasoning_effort = "low" }

[models.a.env]
FOO = "bar"
"""
        )
    )
    reg = models.load(_write(tmp_path, body))
    m = reg.models["a"]
    assert m.aliases == ("a1", "a2")
    assert m.flags == ("--foo", "1")
    assert m.vision is True
    assert m.max_output_tokens == 100
    assert m.min_output_tokens == 10
    assert m.needs_tty is True
    assert m.reasoning == models.Reasoning(parser="qwen3", mirror_content=True)
    assert m.tools == models.Tools(parser="qwen3_xml")
    assert m.presets == {"a-low": {"reasoning_effort": "low"}}
    assert m.env == {"FOO": "bar"}
    assert m.served_names() == ["A", "a1", "a2", "a-low"]


def test_defaults_env_merges_into_every_model_env(tmp_path):
    body = f"""
[gpu]
total_mib = 100000
margin_mib = 1000

[defaults.env]
VLLM_USE_FLASHINFER_SAMPLER = "0"

[builds]
stock = "/opt/stock"

[models.a]
id = "A"
repo = "org/a"
slot = "main"
port = 9001
build = "stock"
ctx = 1000

[models.a.env]
VLLM_USE_FLASHINFER_SAMPLER = "1"
FOO = "bar"

[models.b]
id = "B"
repo = "org/b"
slot = "main"
port = 9002
build = "stock"
ctx = 1000
"""
    reg = models.load(_write(tmp_path, body))
    # model's own env wins over the default on conflict.
    assert reg.models["a"].env == {"VLLM_USE_FLASHINFER_SAMPLER": "1", "FOO": "bar"}
    # a model with no [env] table still gets the default.
    assert reg.models["b"].env == {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
    assert reg.defaults_env == {"VLLM_USE_FLASHINFER_SAMPLER": "0"}


def test_resident_model_requires_vram_mib(tmp_path):
    body = BASE_GPU + _one_model(slot="resident")
    with pytest.raises(models.RegistryError, match="vram_mib"):
        models.load(_write(tmp_path, body))


def test_resident_model_with_vram_mib_loads(tmp_path):
    body = BASE_GPU + _one_model(slot="resident", extra="vram_mib = 500")
    reg = models.load(_write(tmp_path, body))
    assert reg.models["a"].vram_mib == 500


def test_real_models_toml_loads():
    """The repo-root models.toml (the actual deliverable) must load and validate."""
    reg = models.load(REPO_ROOT / "models.toml")
    assert set(reg.models) == {"qwen27b", "flashnext", "glm53", "lfm2"}
    assert reg.gpu.total_mib == 97887
    assert reg.gpu.margin_mib == 1024
    assert reg.defaults_env == {"VLLM_USE_FLASHINFER_SAMPLER": "0"}


# --------------------------------------------------------------------------- #
# Validation — one failing fixture per rule
# --------------------------------------------------------------------------- #


def test_missing_required_field(tmp_path):
    body = BASE_GPU + """
[models.a]
id = "A"
repo = "org/a"
slot = "main"
port = 9001
build = "stock"
# ctx missing
"""
    with pytest.raises(models.RegistryError, match="ctx"):
        models.load(_write(tmp_path, body))


def test_duplicate_id(tmp_path):
    body = BASE_GPU + _one_model("a", id_="SAME", port=9001) + _one_model(
        "b", id_="SAME", port=9002
    )
    with pytest.raises(models.RegistryError, match="duplicate id 'SAME'"):
        models.load(_write(tmp_path, body))


def test_duplicate_alias(tmp_path):
    body = (
        BASE_GPU
        + _one_model("a", id_="A", port=9001, extra='aliases = ["shared"]')
        + _one_model("b", id_="B", port=9002, extra='aliases = ["shared"]')
    )
    with pytest.raises(models.RegistryError, match="duplicate alias 'shared'"):
        models.load(_write(tmp_path, body))


def test_alias_collides_with_another_models_id(tmp_path):
    body = (
        BASE_GPU
        + _one_model("a", id_="A", port=9001)
        + _one_model("b", id_="B", port=9002, extra='aliases = ["A"]')
    )
    with pytest.raises(models.RegistryError, match="duplicate"):
        models.load(_write(tmp_path, body))


def test_preset_shadows_another_models_alias(tmp_path):
    body = (
        BASE_GPU
        + _one_model("a", id_="A", port=9001, extra='aliases = ["shadow-me"]')
        + _one_model(
            "b",
            id_="B",
            port=9002,
            extra="""
[models.b.presets]
"shadow-me" = {}
""",
        )
    )
    with pytest.raises(models.RegistryError, match="shadows the alias of models.a"):
        models.load(_write(tmp_path, body))


def test_preset_shadows_own_models_id(tmp_path):
    body = BASE_GPU + _one_model(
        "a",
        id_="A",
        port=9001,
        extra="""
[models.a.presets]
"A" = {}
""",
    )
    with pytest.raises(models.RegistryError, match="shadows the id of models.a"):
        models.load(_write(tmp_path, body))


def test_duplicate_port(tmp_path):
    body = BASE_GPU + _one_model("a", id_="A", port=9001) + _one_model(
        "b", id_="B", port=9001
    )
    with pytest.raises(models.RegistryError, match="duplicate port 9001"):
        models.load(_write(tmp_path, body))


@pytest.mark.parametrize("bad_port", [8000, 8010])
def test_reserved_port_rejected(tmp_path, bad_port):
    body = BASE_GPU + _one_model("a", id_="A", port=bad_port)
    with pytest.raises(models.RegistryError, match=f"port {bad_port} is reserved"):
        models.load(_write(tmp_path, body))


def test_unknown_build(tmp_path):
    body = BASE_GPU + _one_model("a", id_="A", port=9001, build="nonexistent")
    with pytest.raises(models.RegistryError, match="build 'nonexistent'"):
        models.load(_write(tmp_path, body))


def test_resident_vram_plus_margin_exceeds_total(tmp_path):
    body = f"""
[gpu]
total_mib = 10000
margin_mib = 1000

[builds]
stock = "/opt/stock"

[models.r]
id = "R"
repo = "org/r"
slot = "resident"
vram_mib = 9500
port = 9001
build = "stock"
ctx = 1000
"""
    with pytest.raises(models.RegistryError, match="no room would be left"):
        models.load(_write(tmp_path, body))


def test_resident_vram_plus_margin_exactly_equal_is_rejected(tmp_path):
    """The rule is '>=' (spec 2.1): leaving EXACTLY zero for a main model is
    still a rejection, not an edge case that slips through."""
    body = f"""
[gpu]
total_mib = 10000
margin_mib = 1000

[builds]
stock = "/opt/stock"

[models.r]
id = "R"
repo = "org/r"
slot = "resident"
vram_mib = 9000
port = 9001
build = "stock"
ctx = 1000
"""
    with pytest.raises(models.RegistryError, match="no room would be left"):
        models.load(_write(tmp_path, body))


def test_invalid_toml_raises_registry_error(tmp_path):
    p = tmp_path / "models.toml"
    p.write_text("this is not [ valid toml")
    with pytest.raises(models.RegistryError, match="invalid TOML"):
        models.load(p)


def test_missing_file_raises_registry_error(tmp_path):
    with pytest.raises(models.RegistryError, match="cannot read"):
        models.load(tmp_path / "does-not-exist.toml")


# --------------------------------------------------------------------------- #
# resolve()
# --------------------------------------------------------------------------- #


@pytest.fixture
def small_registry(tmp_path):
    body = (
        BASE_GPU
        + _one_model(
            "a",
            id_="A",
            port=9001,
            extra="""
aliases = ["a-alias"]

[models.a.presets]
a-preset = { x = 1 }
""",
        )
    )
    return models.load(_write(tmp_path, body))


def test_resolve_by_id(small_registry):
    r = small_registry.resolve("A")
    assert r is not None
    assert r.model.key == "a"
    assert r.preset is None
    assert r.overlay == {}


def test_resolve_by_alias(small_registry):
    r = small_registry.resolve("a-alias")
    assert r is not None and r.model.key == "a" and r.preset is None


def test_resolve_by_key(small_registry):
    r = small_registry.resolve("a")
    assert r is not None and r.model.id == "A"


def test_resolve_by_preset(small_registry):
    r = small_registry.resolve("a-preset")
    assert r is not None
    assert r.model.key == "a"
    assert r.preset == "a-preset"
    assert r.overlay == {"x": 1}


def test_resolve_unknown_returns_none(small_registry):
    assert small_registry.resolve("nope") is None


# --------------------------------------------------------------------------- #
# render_argv / render_env
# --------------------------------------------------------------------------- #


def _model(**kw) -> models.Model:
    base = dict(
        key="m",
        id="M",
        repo="org/m",
        slot="main",
        port=9001,
        build="stock",
        ctx="native",
    )
    base.update(kw)
    return models.Model(**base)


def test_render_argv_order_and_content():
    m = _model(
        aliases=("alias1",),
        reasoning=models.Reasoning(parser="qwen3", mirror_content=True),
        tools=models.Tools(parser="qwen3_xml"),
        flags=("--enable-prefix-caching", "--max-num-seqs", "16"),
    )
    argv = models.render_argv(m, "/venv/bin/vllm", 0.91, 262144, 8004)
    assert argv == [
        "/venv/bin/vllm",
        "serve",
        "org/m",
        "--served-model-name",
        "M",
        "alias1",
        "--host",
        "127.0.0.1",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--reasoning-parser",
        "qwen3",
        "--max-model-len",
        "262144",
        "--gpu-memory-utilization",
        "0.91",
        "--port",
        "8004",
        "--enable-prefix-caching",
        "--max-num-seqs",
        "16",
    ]


def test_render_argv_omits_tool_and_reasoning_flags_when_unset():
    m = _model()
    argv = models.render_argv(m, "/venv/bin/vllm", 0.5, 1000, 9001)
    joined = " ".join(argv)
    assert "--tool-call-parser" not in joined
    assert "--enable-auto-tool-choice" not in joined
    assert "--reasoning-parser" not in joined


def test_render_env_returns_models_merged_env():
    m = _model(env={"A": "1", "B": "2"})
    assert models.render_env(m) == {"A": "1", "B": "2"}
    # must be a copy, not the live dict
    out = models.render_env(m)
    out["C"] = "3"
    assert "C" not in m.env


# --------------------------------------------------------------------------- #
# main_util / resident_util
# --------------------------------------------------------------------------- #


def test_main_util_rounds_down_to_two_decimals():
    # (12345 - 1000) / 100000 = 0.11345 -> floor to 0.11, never rounds up.
    assert models.main_util(12345, 100000, 1000) == 0.11


def test_main_util_everything_free_rule():
    assert models.main_util(97887, 97887, 1024) == pytest.approx(0.98, abs=1e-9)


def test_resident_util_rounds_down_to_two_decimals():
    # 3300 / 97887 = 0.0337... -> floor to 0.03
    assert models.resident_util(3300, 97887) == 0.03


def test_util_functions_reject_nonpositive_total():
    with pytest.raises(ValueError):
        models.main_util(100, 0, 10)
    with pytest.raises(ValueError):
        models.resident_util(100, 0)


def test_floor_never_rounds_up():
    # 0.99999 would round UP to 1.00 under normal rounding; floor keeps it 0.99.
    assert models.main_util(99999, 100000, 0) == 0.99
    assert models.resident_util(999, 1000) == 0.99


# --------------------------------------------------------------------------- #
# native_ctx()
# --------------------------------------------------------------------------- #


def test_native_ctx_reads_root_max_position_embeddings(tmp_path):
    hub = _fake_hub(tmp_path, "org/model", {"max_position_embeddings": 262144})
    assert models.native_ctx("org/model", hub) == 262144


def test_native_ctx_reads_text_config_max_position_embeddings(tmp_path):
    hub = _fake_hub(
        tmp_path,
        "org/model",
        {"text_config": {"max_position_embeddings": 1048576}},
    )
    assert models.native_ctx("org/model", hub) == 1048576


def test_native_ctx_missing_repo_raises(tmp_path):
    with pytest.raises(models.RegistryError, match="no config.json"):
        models.native_ctx("org/nowhere", tmp_path)


def test_native_ctx_missing_field_raises(tmp_path):
    hub = _fake_hub(tmp_path, "org/model", {"some_other_field": 1})
    with pytest.raises(models.RegistryError, match="max_position_embeddings"):
        models.native_ctx("org/model", hub)


def test_native_ctx_matches_real_hub_cache_for_qwen27b():
    """Confirms the golden test's assumption: the 27B's native ctx (262144)
    equals the current live MAX_MODEL_LEN, so no normalisation is needed for
    that flag. Skips cleanly on a box without this repo in its hub cache."""
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub"
    dirname = "models--RadixArk--Qwen3.8-27B-NVFP4"
    if not (hub_dir / dirname).is_dir():
        pytest.skip("RadixArk/Qwen3.8-27B-NVFP4 not in this box's hub cache")
    assert models.native_ctx("RadixArk/Qwen3.8-27B-NVFP4") == 262144
