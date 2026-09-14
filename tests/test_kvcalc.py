"""Per-architecture KV arithmetic, checked against boots measured on this box.

The old sizing rule was one stored KiB/token per repo, with a family MEDIAN as
the fallback. Against the six boots below it gave:

    Qwen3.8-27B    util 0.47   531,529 tokens        +3.1%
    Flash-Next     util 0.95   290,925 tokens        -6.3%   (delivered config)
    GLM-5.3-Flash  327,680 ctx 375,543 tokens        -100%   (no rate at all)

The Flash-Next miss is the flag-blindness: the stored rate came from a boot
without --mamba-ssm-cache-dtype bfloat16, and the store keys observations on
(repo, ctx) only, so the flag that changes the cache layout is invisible. The
GLM miss is the family median having no sibling to take a median of.

Every ground truth here was read out of this machine's own records, not
invented: the two Flash-Next and the 27B figures come from
~/Projects/local_llm/logs/{server,qwen_server}.log and coldstart's
state/measurements.json; the GLM figure is serve-opt.sh's KV_BYTES
(MAX_LEN * 17200) and the token count the engine reported for it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from servedeck import discovery as registry
from servedeck import kvcalc

GIB = 1 << 30
HUB = Path.home() / ".cache" / "huggingface" / "hub"

#: (name, repo_id, mamba_ssm_dtype flag, kv_gib, measured tokens, ctx)
GROUND_TRUTHS = [
    ("27B at util 0.47", "RadixArk/Qwen3.8-27B-NVFP4", None, 19.27, 531_529, 262_144),
    ("27B at util 0.50", "RadixArk/Qwen3.8-27B-NVFP4", None, 22.72, 627_117, 262_144),
    (
        "Flash-Next, delivered config",
        "mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4",
        "bfloat16",
        7.98,
        290_925,
        262_144,
    ),
    (
        "Flash-Next, no cache flags",
        "mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4",
        None,
        7.86,
        272_062,
        262_144,
    ),
    # The RadixArk Flash-Next build's blobs are no longer on this filesystem
    # (its snapshot is all dangling symlinks), so its boot is checked against
    # the uncensored build's config. The two differ only in the dtype of the
    # host-offloaded PLE n-gram table; every KV-bearing tensor -- layer types,
    # KV heads, head dim, indexer -- is byte-identical.
    (
        "Flash-Next, RadixArk build",
        "mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4",
        None,
        8.83,
        304_653,
        262_144,
    ),
    (
        "GLM-5.3-Flash",
        "dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4",
        None,
        327_680 * 17_200 / GIB,
        375_543,
        327_680,
    ),
]


def _cfg(repo_id: str) -> dict:
    cfg = registry.load_model_config(repo_id, HUB)
    if cfg is None:
        pytest.skip(f"{repo_id} is not in the local hub cache")
    return cfg


@pytest.mark.parametrize(
    "name,repo_id,ssm,kv_gib,tokens,ctx",
    GROUND_TRUTHS,
    ids=[g[0] for g in GROUND_TRUTHS],
)
def test_calculator_reproduces_every_measured_boot(
    name: str, repo_id: str, ssm: str | None, kv_gib: float, tokens: int, ctx: int
) -> None:
    """Within 5% of what the engine actually allocated, on every boot this
    machine has a record of — across three architectures and two cache dtypes."""
    geo = kvcalc.geometry(_cfg(repo_id), mamba_ssm_dtype=ssm)
    predicted = kvcalc.tokens_at(geo, kv_gib, ctx)
    err = (predicted - tokens) / tokens
    assert abs(err) <= 0.05, (
        f"{name}: predicted {predicted:,} vs measured {tokens:,} ({err:+.1%})"
    )


def test_each_architecture_is_recognised_as_its_own_family() -> None:
    """A family median is wrong because KV size is a property of the layer
    stack, not of the family name. These three stacks must not be conflated."""
    assert kvcalc.geometry(_cfg("RadixArk/Qwen3.8-27B-NVFP4")).family == "hybrid_gdn"
    assert (
        kvcalc.geometry(_cfg("mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4")).family
        == "qsa_hybrid"
    )
    assert (
        kvcalc.geometry(_cfg("dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4")).family
        == "mla"
    )


def test_the_cache_dtype_the_checkpoint_declares_is_honoured() -> None:
    """Qwen3.8-27B's modelopt config declares kv_cache_scheme num_bits 8, so
    its KV is cached in fp8. Sizing it at the model dtype (bf16) doubles the
    per-token cost and halves the predicted context — a 2x error, larger than
    every other term in this module put together."""
    cfg = _cfg("RadixArk/Qwen3.8-27B-NVFP4")
    assert kvcalc.kv_cache_dtype_bytes(cfg) == 1
    fp8 = kvcalc.geometry(cfg)
    bf16 = kvcalc.geometry(cfg, kv_cache_dtype="bfloat16")
    assert bf16.attn_bytes_per_token == pytest.approx(2 * fp8.attn_bytes_per_token)
    # And the fp8 reading is the one that matches the boot.
    assert kvcalc.tokens_at(fp8, 19.27, 262_144) == pytest.approx(531_529, rel=0.05)
    assert kvcalc.tokens_at(bf16, 19.27, 262_144) < 300_000


def test_a_launch_flag_that_changes_the_cache_changes_the_answer() -> None:
    """--mamba-ssm-cache-dtype bfloat16 halves the recurrent state. The old
    store could not see it: the same repo at the same context has one row, so
    whichever of the two boots was appended last spoke for both."""
    cfg = _cfg("mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4")
    plain = kvcalc.geometry(cfg)
    bf16 = kvcalc.geometry(cfg, mamba_ssm_dtype="bfloat16")
    assert bf16.state_bytes_per_seq < plain.state_bytes_per_seq
    assert kvcalc.tokens_at(bf16, 7.98, 262_144) > kvcalc.tokens_at(plain, 7.98, 262_144)


def test_the_rate_is_not_treated_as_context_invariant() -> None:
    """Per-sequence state is a fixed cost spread over the context, so a short
    context costs MORE per token. The store's own "measured at a different
    context length" warning was this fact showing through; the calculator
    answers at the context it is asked about."""
    geo = kvcalc.geometry(_cfg("mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4"))
    assert geo.kib_per_token(131_072) > geo.kib_per_token(262_144)


def test_an_uncalibrated_family_says_it_is_uncalibrated() -> None:
    """No dense model has booted on this machine, so its allocator correction
    is a placeholder 1.0. Presenting that with the same confidence as a
    calibrated figure is the class of claim this repo does not make."""
    dense = {
        "architectures": ["LlamaForCausalLM"],
        "dtype": "bfloat16",
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    geo = kvcalc.geometry(dense)
    assert geo.family == "dense_gqa"
    assert geo.calibrated is False
    # 32 layers * 2 * 8 heads * 128 dims * 2 bytes = 131,072 B per token.
    assert geo.attn_bytes_per_token == 131_072
    assert geo.allocator_factor == 1.0


def test_a_config_with_no_attention_geometry_refuses_rather_than_guesses() -> None:
    geo = kvcalc.geometry({"architectures": ["Mystery"], "num_hidden_layers": 4})
    assert geo.family == "unknown"
    assert geo.attn_bytes_per_token == 0
    assert geo.calibrated is False


# --------------------------------------------------------------------------
# Integration: the defect the owner reported, end to end
# --------------------------------------------------------------------------
def _observations() -> list[dict]:
    """The real observation store this box has accumulated.

    Not `discovery.default_measurements_path()`: conftest points
    $SERVEDECK_STATE_DIR at a tmp directory for every test, which is exactly
    right for everything that WRITES and useless for the two tests below,
    whose whole subject is the real measurements. Both candidate locations are
    tried -- v2's own state dir and the pre-rename coldstart tree that was
    serving :8010 until 09-12 -- and the test skips if neither exists, because
    a machine without the store cannot have an opinion about it.
    """
    for base in (
        Path(__file__).resolve().parent.parent / "state",
        Path.home() / "Projects" / "coldstart" / "state",
    ):
        candidate = base / registry.MEASUREMENTS_FILENAME
        if candidate.is_file():
            return json.loads(candidate.read_text())
    pytest.skip("no observation store on this machine")


def test_a_model_with_no_sibling_observation_still_gets_a_kv_rate() -> None:
    """GLM-5.3-Flash: no Glm5Next checkpoint has ever booted here, so the
    family median had nothing to take a median of and the whole capacity panel
    read zero tokens for a model this box runs at 327,680 context."""
    ri = registry.resolve_inputs(
        "dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4",
        0.95,
        327_680,
        observations=_observations(),
        hub_dir=HUB,
    )
    assert ri.kv_kib_per_token is not None, "no KV rate at all for a model that runs here"
    # The measured rate: a 5.25 GiB cache held 375,543 tokens.
    measured = (327_680 * 17_200) / 375_543 / 1024
    assert ri.kv_kib_per_token == pytest.approx(measured, rel=0.05)
    assert ri.kv_source == "estimated", "an estimate must not be labelled measured"


def test_a_measured_observation_still_wins_over_the_calculator() -> None:
    """Over-correction guard. The calculator is a fallback, not a replacement:
    where the engine's own number exists it must be the answer, and the label
    must stay "measured"."""
    ri = registry.resolve_inputs(
        "RadixArk/Qwen3.8-27B-NVFP4",
        0.47,
        262_144,
        observations=_observations(),
        hub_dir=HUB,
    )
    assert ri.kv_source == "measured"
    assert ri.trust == "measured"
    assert ri.kv_kib_per_token == pytest.approx(37.99, abs=0.05)


def test_a_models_cache_flags_have_exactly_one_source(tmp_path: Path) -> None:
    """The flags that decide a model's KV layout must come from the same place
    the launch reads them.

    In v1 they came from two: a backend's fixed env in servedeck.toml, and
    EXTRA_ARGS out of local_llm/.config -- a mutable bash file two control
    planes both wrote (REDESIGN R2). When they disagreed the capacity panel
    described a server nobody starts; the delivered Flash-Next configuration
    carried --mamba-ssm-cache-dtype bfloat16, worth 6.9% of its token count,
    and the panel did not know.

    v2 has one source, `flags` in models.toml, and this pins that: whatever is
    written there reaches the argv verbatim and in order, and no other flag
    appears that the registry did not put there.
    """
    from servedeck import models as _models

    toml = tmp_path / "models.toml"
    toml.write_text(
        """
[gpu]
total_mib = 100000
margin_mib = 1000
[builds.stock]
venv = "/opt/stock"
cuda_home = "/opt/stock/lib/python3.13/site-packages/nvidia/cu13"
[models.m]
id = "M"
repo = "org/m"
slot = "main"
port = 19001
build = "stock"
ctx = 4096
flags = ["--mamba-ssm-cache-dtype", "bfloat16", "--language-model-only"]
"""
    )
    model = _models.load(toml).models["m"]
    argv = _models.render_argv(model, "/opt/stock/bin/vllm", 0.91, 4096, 19001)
    assert argv[-3:] == ["--mamba-ssm-cache-dtype", "bfloat16", "--language-model-only"]
    # Nothing else invented a flag: every `--x` in the argv is either one this
    # module renders by contract or one the registry supplied.
    rendered = {
        "--served-model-name", "--host", "--max-model-len",
        "--gpu-memory-utilization", "--port",
    }
    strays = [
        a for a in argv
        if a.startswith("--") and a not in rendered and a not in model.flags
    ]
    assert not strays, strays
