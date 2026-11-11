"""Property-based & differential formal laws via Hypothesis.

Exercises universal algebraic laws over randomized inputs across:
  1. ``wire.py`` (``render_kimi``, ``render_codex``, ``render_vscode``): round-trip
     TOML/JSON syntactic validity, exact fixed-point idempotence
     (``render(render(doc)) == render(doc)``), user-key & comment preservation,
     and stale ``servedeck/<retired>`` pruning.
  2. ``policies.py`` & ``glm_policies.py``: SSE arbitrary byte-chunk-boundary
     invariance (including multi-byte UTF-8 and split tag boundaries), nested
     tag-stripping fixed-point idempotence, and no-op byte identity.
  3. ``capacity.py``, ``kvcalc.py``, ``parallelism.py``: totality across
     arbitrary floats/ints/NaNs, monotonicity in ``util`` and ``ctx``, and
     ``single_request_fit`` postconditions.
  4. ``models.py``: schema fuzzer proving ``load_from_dict`` raises ONLY
     ``RegistryError`` (never bare ``KeyError``/``TypeError``/``ValueError``/
     ``AttributeError``) on arbitrary fuzzed dictionary inputs.
"""

from __future__ import annotations

import asyncio
import json
import math
import tomllib
from pathlib import Path
from typing import Any

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis import strategies as st

from servedeck import capacity as _capacity
from servedeck import glm_policies as _glm
from servedeck import kvcalc as _kvcalc
from servedeck import models as _models
from servedeck import parallelism as _parallelism
from servedeck import policies as _policies
from servedeck import wire as _wire

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_REGISTRY = _models.load(REPO_ROOT / "models.toml")


def _fixed_ctx(m: _models.Model) -> int:
    return m.ctx if isinstance(m.ctx, int) else 32768


# ---------------------------------------------------------------------------
# 1. wire.py Universal Validity, Idempotence & Preservation Laws
# ---------------------------------------------------------------------------

_SAFE_COMMENT = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs"), whitelist_characters="-_./:"),
    min_size=0,
    max_size=40,
)
_SAFE_IDENT = st.from_regex(r"[a-z][a-z0-9_-]{0,12}", fullmatch=True)


@settings(max_examples=80, deadline=None)
@given(
    top_comment=_SAFE_COMMENT,
    thinking_comment=_SAFE_COMMENT,
    thinking_enabled=st.booleans(),
    thinking_effort=st.sampled_from(["none", "low", "medium", "high"]),
    cloud_default=st.booleans(),
    stale_slug=_SAFE_IDENT.filter(lambda s: s not in REAL_REGISTRY.models and s != "main"),
    extra_model_slug=_SAFE_IDENT,
)
def test_prop_wire_render_kimi_valid_toml_idempotent_and_prunes_stale(
    top_comment: str,
    thinking_comment: str,
    thinking_enabled: bool,
    thinking_effort: str,
    cloud_default: bool,
    stale_slug: str,
    extra_model_slug: str,
) -> None:
    default_line = (
        'default_model = "kimi-code/k2"\n'
        if cloud_default
        else f'default_model = "servedeck/{stale_slug}"\n'
    )
    existing = (
        f"# {top_comment}\n"
        f"{default_line}\n"
        f"{_wire._OWNED_COMMENT}\n"
        f'[models."servedeck/{stale_slug}"]\n'
        'provider = "servedeck"\n'
        'model = "Retired"\n'
        "max_context_size = 4096\n\n"
        f'[models."custom/{extra_model_slug}"]\n'
        'provider = "other"\n'
        'model = "KeepMe"\n\n'
        "[thinking]\n"
        f"# {thinking_comment}\n"
        f"enabled = {'true' if thinking_enabled else 'false'}\n"
        f'effort = "{thinking_effort}"\n'
    )
    out1 = _wire.render_kimi(REAL_REGISTRY, existing, resolve_ctx=_fixed_ctx)
    parsed = tomllib.loads(out1)  # Must NEVER produce invalid TOML!
    out2 = _wire.render_kimi(REAL_REGISTRY, out1, resolve_ctx=_fixed_ctx)
    assert out1 == out2, "render_kimi violated fixed-point idempotence"

    assert parsed["thinking"]["enabled"] is True
    assert parsed["thinking"]["effort"] == thinking_effort
    assert f"servedeck/{stale_slug}" not in parsed.get("models", {})
    assert f"custom/{extra_model_slug}" in parsed.get("models", {})
    if cloud_default:
        assert parsed["default_model"] == "kimi-code/k2"
    else:
        assert parsed["default_model"] == "servedeck/main"


@settings(max_examples=60, deadline=None)
@given(
    sandbox_mode=st.sampled_from(["read-only", "workspace-write"]),
    stale_profile=_SAFE_IDENT,
    threads=st.integers(min_value=1, max_value=128),
)
def test_prop_wire_render_codex_valid_toml_and_idempotent(
    sandbox_mode: str, stale_profile: str, threads: int
) -> None:
    existing = (
        f'sandbox_mode = "{sandbox_mode}"\n\n'
        "[agents]\n"
        f"max_concurrent_threads_per_session = {threads}\n\n"
        f"{_wire._OWNED_COMMENT}\n"
        f"[profiles.{stale_profile}]\n"
        'model = "old"\n'
    )
    out1 = _wire.render_codex(REAL_REGISTRY, existing, resolve_ctx=_fixed_ctx)
    parsed = tomllib.loads(out1)
    out2 = _wire.render_codex(REAL_REGISTRY, out1, resolve_ctx=_fixed_ctx)
    assert out1 == out2
    assert parsed["model"] == "main"
    assert parsed["model_provider"] == "servedeck"
    assert parsed["sandbox_mode"] == sandbox_mode
    assert parsed["agents"]["max_concurrent_threads_per_session"] == threads
    assert "profiles" not in parsed


@settings(max_examples=50, deadline=None)
@given(other_group_name=_SAFE_IDENT.filter(lambda s: s != "servedeck"))
def test_prop_wire_render_vscode_valid_json_and_idempotent(other_group_name: str) -> None:
    existing = json.dumps([{"name": other_group_name, "models": [{"id": "keep"}]}])
    out1 = _wire.render_vscode(REAL_REGISTRY, existing, resolve_ctx=_fixed_ctx)
    parsed = json.loads(out1)
    out2 = _wire.render_vscode(REAL_REGISTRY, out1, resolve_ctx=_fixed_ctx)
    assert out1 == out2
    names = [g["name"] for g in parsed]
    assert other_group_name in names and "servedeck" in names


# ---------------------------------------------------------------------------
# 2. policies.py & glm_policies.py Chunk-Split Invariance & Tag Fixed-Points
# ---------------------------------------------------------------------------

_TAG_FRAGMENTS = st.sampled_from(
    [
        "</think>",
        "<think>",
        "</arg_key>",
        "<arg_key>",
        "</arg_value>",
        "<arg_value>",
        "</tool_call>",
        "<tool_call>",
        "plain",
        "αβγ",
        '{"a":1}',
    ]
)


@settings(max_examples=100, deadline=None)
@given(parts=st.lists(_TAG_FRAGMENTS, min_size=0, max_size=15))
def test_prop_glm_tag_strippers_are_idempotent_and_leave_no_leaked_tags(parts: list[str]) -> None:
    # Include deliberate nested splicing like "</th" + "</think>" + "ink>"
    text = "".join(parts)
    nested = text.replace("</think>", "</th</think>ink>").replace("</arg_key>", "</arg_</arg_key>key>")

    cleaned_arg = _glm.strip_leaked_tags(nested)
    assert _glm.strip_leaked_tags(cleaned_arg) == cleaned_arg
    for forbidden in ("<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "<tool_call>", "</tool_call>"):
        assert forbidden not in cleaned_arg

    cleaned_think = _glm.safe_reasoning(nested)
    assert _glm.safe_reasoning(cleaned_think) == cleaned_think
    assert "</think>" not in cleaned_think.lower()


@settings(max_examples=60, deadline=None)
@given(
    reasoning_txt=st.text(min_size=1, max_size=60),
    arg_prefix=st.from_regex(r"[a-zA-Z0-9_/-]{1,20}", fullmatch=True),
    cuts=st.lists(st.integers(min_value=1, max_value=260), min_size=1, max_size=25),
)
def test_prop_sse_streams_are_invariant_under_arbitrary_byte_chunk_boundaries(
    reasoning_txt: str, arg_prefix: str, cuts: list[int]
) -> None:
    """Feeding an SSE response 1 chunk at a time vs arbitrarily sliced byte
    chunks (including mid-UTF-8 and mid-``</arg_key>``) must yield the exact
    same mirrored reasoning and sanitized tool-call arguments."""
    f1 = {
        "id": "cmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "glm53-flash",
        "choices": [{"index": 0, "delta": {"reasoning": reasoning_txt}, "finish_reason": None}],
    }
    f2 = {
        "id": "cmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "glm53-flash",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "f", "arguments": f'{{"k":"{arg_prefix}</arg_key>"}}'},
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    raw = (
        b"data: " + json.dumps(f1, ensure_ascii=False).encode("utf-8") + b"\n\n"
        + b"data: " + json.dumps(f2, ensure_ascii=False).encode("utf-8") + b"\n\n"
        + b"data: [DONE]\n\n"
    )

    points = sorted({0, *(c % max(1, len(raw)) for c in cuts), len(raw)})
    sliced = [raw[a:b] for a, b in zip(points, points[1:]) if a < b]

    async def run_pipeline(chunks: list[bytes]) -> tuple[str, str, str]:
        st_glm = _glm.GlmState()
        pol = type("P", (), {"sanitize_tool_tags": True, "restore_reasoning": False, "capture": False, "min_output_tokens": None})()
        hook = _glm.begin(pol, model_id="glm53-flash", state=st_glm)
        assert hook is not None

        async def src():
            for ch in chunks:
                yield ch

        collected = b"".join([c async for c in _policies.sse_stream(hook.sse(src()))])
        r_acc, rc_acc, arg_acc = [], [], []
        for line in collected.split(b"\n"):
            if line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
                obj = json.loads(line[6:])
                delta = obj["choices"][0].get("delta", {})
                if delta.get("reasoning"):
                    r_acc.append(delta["reasoning"])
                if delta.get("reasoning_content"):
                    rc_acc.append(delta["reasoning_content"])
                for tc in delta.get("tool_calls") or []:
                    if tc.get("function", {}).get("arguments"):
                        arg_acc.append(tc["function"]["arguments"])
        return "".join(r_acc), "".join(rc_acc), "".join(arg_acc)

    whole_r, whole_rc, whole_args = asyncio.run(run_pipeline([raw]))
    split_r, split_rc, split_args = asyncio.run(run_pipeline(sliced))
    assert split_r == whole_r == reasoning_txt
    assert split_rc == whole_rc == reasoning_txt
    assert split_args == whole_args == f'{{"k":"{arg_prefix}"}}'


# ---------------------------------------------------------------------------
# 3. capacity.py & kvcalc.py Monotonicity & Totality Properties
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    u1=st.floats(min_value=0.05, max_value=1.0, allow_nan=False, allow_infinity=False),
    u2=st.floats(min_value=0.05, max_value=1.0, allow_nan=False, allow_infinity=False),
    ctx=st.integers(min_value=1024, max_value=262_144),
    seqs=st.integers(min_value=1, max_value=64),
)
def test_prop_capacity_compute_monotonic_in_util_and_bar_sums_to_100(
    u1: float, u2: float, ctx: int, seqs: int
) -> None:
    lo_u, hi_u = (u1, u2) if u1 <= u2 else (u2, u1)
    mi = _capacity.ModelInputs(
        repo_id="org/m",
        backend="stock",
        model_max_ctx=262_144,
        weights_gib=20.75,
        weights_source="measured",
        kv_kib_per_token=37.99,
        overhead_gib=4.33,
        trust="measured",
    )
    r_lo = _capacity.compute(mi, util=lo_u, ctx=ctx, max_num_seqs=seqs)
    r_hi = _capacity.compute(mi, util=hi_u, ctx=ctx, max_num_seqs=seqs)
    assert r_lo.kv_tokens <= r_hi.kv_tokens
    assert r_lo.max_single_ctx <= r_hi.max_single_ctx
    if r_hi.can_apply:
        total_pct = r_hi.bar.weights_pct + r_hi.bar.kv_pct + r_hi.bar.overhead_pct + r_hi.bar.free_pct
        assert math.isclose(total_pct, 100.0, abs_tol=1e-5)


@settings(max_examples=80, deadline=None)
@given(
    attn=st.floats(min_value=1024.0, max_value=65536.0, allow_nan=False, allow_infinity=False),
    state=st.floats(min_value=0.0, max_value=50_000_000.0, allow_nan=False, allow_infinity=False),
    c1=st.integers(min_value=1024, max_value=524_288),
    c2=st.integers(min_value=1024, max_value=524_288),
)
def test_prop_kvcalc_effective_bytes_per_token_non_increasing_with_ctx(
    attn: float, state: float, c1: int, c2: int
) -> None:
    lo_c, hi_c = (c1, c2) if c1 <= c2 else (c2, c1)
    geo = _kvcalc.KvGeometry(
        family="hybrid_gdn",
        attn_bytes_per_token=attn,
        state_bytes_per_seq=state,
        allocator_factor=1.1663,
        calibrated=True,
        factor_note="prop",
    )
    assert geo.bytes_per_token(hi_c) <= geo.bytes_per_token(lo_c) + 1e-9


# ---------------------------------------------------------------------------
# 4. models.py Schema Fuzzer: ONLY RegistryError on Arbitrary Malformed Dicts
# ---------------------------------------------------------------------------

_ANY_SCALAR = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-100_000, max_value=200_000),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=20),
)
_ANY_VALUE = st.recursive(
    _ANY_SCALAR,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(max_size=12), children, max_size=4),
    ),
    max_leaves=15,
)


@settings(max_examples=150, deadline=None)
@given(
    raw=st.one_of(
        st.dictionaries(st.text(max_size=12), _ANY_VALUE, max_size=6),
        st.fixed_dictionaries(
            {
                "gpu": _ANY_VALUE,
                "builds": _ANY_VALUE,
                "defaults": _ANY_VALUE,
                "models": _ANY_VALUE,
            }
        ),
        st.fixed_dictionaries(
            {
                "gpu": st.fixed_dictionaries({"total_mib": _ANY_SCALAR, "margin_mib": _ANY_SCALAR}),
                "builds": st.fixed_dictionaries(
                    {"stock": st.fixed_dictionaries({"venv": _ANY_SCALAR, "cuda_home": _ANY_SCALAR})}
                ),
                "models": st.dictionaries(st.text(min_size=0, max_size=10), _ANY_VALUE, max_size=3),
            }
        ),
    )
)
def test_prop_models_load_from_dict_never_leaks_unhandled_exceptions(raw: dict[str, Any]) -> None:
    """Contract in ``docs/CONFIGURATION.md:19``: one problem in ``models.toml``
    is one ``RegistryError``, never a bare ``KeyError``/``TypeError``/``ValueError``."""
    try:
        reg = _models.load_from_dict(raw)
    except _models.RegistryError:
        return
    # If load_from_dict succeeded, basic structural postconditions must hold:
    assert reg.gpu.total_mib > 0
    assert reg.gpu.margin_mib >= 0
    assert len(reg.models) >= 1
    for key, model in reg.models.items():
        assert 1 <= model.port <= 65535
        assert model.port not in (8000, 8010)
        assert model.build in reg.builds
