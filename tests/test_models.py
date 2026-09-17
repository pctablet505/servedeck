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

[builds.stock]
venv = "/opt/stock"
cuda_home = "/opt/stock/lib/python3.13/site-packages/nvidia/cu13"
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

[builds.stock]
venv = "/opt/stock"
cuda_home = "/opt/stock/lib/python3.13/site-packages/nvidia/cu13"

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
    assert reg.defaults_env == {"VLLM_USE_FLASHINFER_SAMPLER": "0"}  # this fixture's own [defaults.env]


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
    assert reg.defaults_env == {
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        # Offline for every model since 2026-09-18: all four snapshots are
        # complete on disk and doctor has a `weights (<key>)` row, so a launch
        # never depends on the network or resolves a revision that is not the
        # one on disk. models.toml carries the full history of this flag.
        "HF_HUB_OFFLINE": "1",
    }


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


@pytest.mark.parametrize("bad_key", ["UPPER", "has_underscore", "1_2_3", "qwen.27b"])
def test_invalid_model_key_shape_rejected(tmp_path, bad_key):
    """Registry keys become unit names (model-<key>); servedeck.units only
    accepts [a-z0-9-]+ after that prefix."""
    body = BASE_GPU + _one_model(f'"{bad_key}"', id_="A", port=9001)
    with pytest.raises(models.RegistryError, match="key must match"):
        models.load(_write(tmp_path, body))


def test_valid_model_key_shape_loads(tmp_path):
    body = BASE_GPU + _one_model("qwen-27b", id_="A", port=9001)
    reg = models.load(_write(tmp_path, body))
    assert "qwen-27b" in reg.models


def test_build_missing_venv_raises(tmp_path):
    body = (
        """
[gpu]
total_mib = 100000
margin_mib = 1000

[builds.stock]
cuda_home = "/opt/stock/lib/python3.13/site-packages/nvidia/cu13"
"""
        + _one_model()
    )
    with pytest.raises(models.RegistryError, match="missing 'venv'"):
        models.load(_write(tmp_path, body))


def test_build_missing_cuda_home_raises(tmp_path):
    body = (
        """
[gpu]
total_mib = 100000
margin_mib = 1000

[builds.stock]
venv = "/opt/stock"
"""
        + _one_model()
    )
    with pytest.raises(models.RegistryError, match="missing 'cuda_home'"):
        models.load(_write(tmp_path, body))


def test_build_table_wrong_type_raises(tmp_path):
    """The old flat `stock = "/path"` shape is rejected, not silently accepted
    with a missing cuda_home — [builds.<name>] must be a table."""
    body = (
        """
[gpu]
total_mib = 100000
margin_mib = 1000

[builds]
stock = "/opt/stock"
"""
        + _one_model()
    )
    with pytest.raises(models.RegistryError, match="must be a table"):
        models.load(_write(tmp_path, body))


def test_resident_vram_plus_margin_exceeds_total(tmp_path):
    body = f"""
[gpu]
total_mib = 10000
margin_mib = 1000

[builds.stock]
venv = "/opt/stock"
cuda_home = "/opt/stock/lib/python3.13/site-packages/nvidia/cu13"

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

[builds.stock]
venv = "/opt/stock"
cuda_home = "/opt/stock/lib/python3.13/site-packages/nvidia/cu13"

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


def _build(**kw) -> models.Build:
    base = dict(venv="/opt/venv", cuda_home="/opt/venv/lib/python3.13/site-packages/nvidia/cu13")
    base.update(kw)
    return models.Build(**base)


def test_render_env_merges_model_env_with_cuda_home_and_path():
    m = _model(env={"A": "1", "B": "2"})
    b = _build(venv="/opt/venv", cuda_home="/opt/cuda")
    out = models.render_env(m, b)
    assert out["A"] == "1"
    assert out["B"] == "2"
    assert out["CUDA_HOME"] == "/opt/cuda"
    assert out["PATH"] == "/opt/venv/bin:/opt/cuda/bin:/usr/local/bin:/usr/bin:/bin"


def test_render_env_is_a_copy_not_the_live_dict():
    m = _model(env={"A": "1"})
    b = _build()
    out = models.render_env(m, b)
    out["A"] = "changed"
    assert m.env["A"] == "1"


def test_render_env_expands_tilde_in_venv_and_cuda_home():
    m = _model()
    b = _build(venv="~/Projects/x/.venv", cuda_home="~/Projects/x/.venv/lib/python3.13/site-packages/nvidia/cu13")
    out = models.render_env(m, b)
    home = str(Path.home())
    assert out["CUDA_HOME"].startswith(home)
    assert out["PATH"].startswith(f"{home}/Projects/x/.venv/bin:{home}/Projects/x/.venv/lib")
    assert "~" not in out["CUDA_HOME"]
    assert "~" not in out["PATH"]


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



def test_every_model_launches_from_the_local_hub_cache() -> None:
    """2026-09-17: the first v2 launch of Flash-Next went online and died on a
    gated-repo 401; HF_HUB_OFFLINE=1 lived only in its v1 launcher and profile.
    It was then set for all four models unverified, narrowed back to flashnext
    on review, and on 2026-09-18 made the default for all four once the audit
    had confirmed every snapshot is complete on disk and doctor had grown a
    `weights (<key>)` row per model. A launch that depends on the network can
    resolve a revision other than the one on disk, and needs a token for a
    gated repo; this one cannot."""
    from pathlib import Path
    reg = models.load(Path(__file__).resolve().parent.parent / "models.toml")
    env = lambda k: models.render_env(reg.models[k], reg.builds[reg.models[k].build])  # noqa: E731
    for key in reg.models:
        assert env(key).get("HF_HUB_OFFLINE") == "1", key
