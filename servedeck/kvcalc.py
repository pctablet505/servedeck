"""Per-architecture KV cache arithmetic, from a model's own config.json.

WHY THIS EXISTS
---------------
Servedeck sized every model's KV cache as ``kv_gib / kv_kib_per_token`` with a
single measured rate per repo, and fell back — for a model it had never booted
— to the MEDIAN rate of other models sharing the same ``architectures[0]``.
Both halves are wrong in ways that were visible on this box:

* A model with no observation of its own and no family sibling got no rate at
  all, so the whole capacity panel read zero. GLM-5.3-Flash is exactly that
  case: measured here at 375,543 tokens on a 5.25 GiB cache, displayed as
  "0 tokens, capacity unknown".
* A family median treats KV size as a property of the family. It is a property
  of the layer stack: 16 full-attention layers of 4 KV heads (Qwen3.8-27B) and
  12 layers of 2 (Flash-Next) differ by 1.6x before any measurement.
* The stored rate is keyed on (repo, context) only, so launch flags that
  change the cache layout are invisible. The same Flash-Next checkpoint at the
  same util and context measures 30.29 KiB/token without
  ``--mamba-ssm-cache-dtype bfloat16`` and 28.76 KiB/token with it — a 6.5%
  error whichever of the two boots was appended last.

WHAT IT DOES NOT DO
-------------------
It does not reimplement vLLM's hybrid block allocator. That allocator pads
pages, groups layers and rounds block counts in ways this module deliberately
does not model — an attempt to do so from outside the engine would be a second
source of truth that goes stale on the next vLLM release. Instead each family
carries ONE empirical ``allocator_factor``, calibrated against boots measured
on this machine and reported with its residuals (see FACTORS below). Every
other term is derived from config.json arithmetic.

A number this module produces is therefore an ESTIMATE and must be labelled
one. When the running engine publishes its own resolved capacity — as
``kv_cache_size_tokens`` on ``vllm:cache_config_info``, or the boot log's
"GPU KV cache size: N tokens" — that measurement wins, always.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

Family = Literal["dense_gqa", "hybrid_gdn", "qsa_hybrid", "mla", "unknown"]

#: Bytes per element, by the name vLLM/HF use for the dtype.
_DTYPE_BYTES: dict[str, int] = {
    "float32": 4, "fp32": 4, "float": 4,
    "bfloat16": 2, "bf16": 2, "float16": 2, "fp16": 2, "half": 2,
    "float8_e4m3fn": 1, "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1, "int8": 1,
}

#: Empirical correction for the block/group padding vLLM's allocator applies
#: and this module does not model. One per family, each calibrated against
#: boots measured on THIS machine and carried with the residual it leaves.
#:
#: hybrid_gdn — Qwen3.8-27B-NVFP4, two boots at different cache sizes:
#:     19.27 GiB -> 531,529 tokens   predicted 531,713   +0.03%
#:     22.72 GiB -> 627,117 tokens   predicted 626,905   -0.03%
#:   Two points on one model at two cache sizes is a real check that the
#:   relationship is proportional in available memory. It is not a check
#:   across models: there is only one hybrid-GDN-without-QSA model here.
#:
#: qsa_hybrid — Qwen3.8-Flash-Next, three boots (the factor is their mean, so
#:   no single boot is reproduced exactly and none is privileged):
#:     7.98 GiB -> 290,925 tokens  (--mamba-ssm-cache-dtype bfloat16)  -3.0%
#:     7.86 GiB -> 272,062 tokens  (plain)                             +1.3%
#:     8.83 GiB -> 304,653 tokens  (RadixArk build, plain)             +1.7%
#:
#: mla — ONE boot: GLM-5.3-Flash at 327,680 ctx, a 5.25 GiB cache giving
#:   375,543 tokens. A single point cannot validate anything; it is
#:   calibration. The factor is near 1 only because the MLA latent cache
#:   dominates and is exactly derivable from kv_lora_rank.
#:
#: dense_gqa — NOT calibrated. No dense model has ever booted on this machine,
#: so the factor is 1.0, `calibrated` is False, and the figure is a FLOOR that
#: the caller must present as such.
_FACTORS: dict[str, tuple[float, bool, str]] = {
    "hybrid_gdn": (1.1663, True, "2 boots of one model, residuals +/-0.03%"),
    "qsa_hybrid": (1.1858, True, "mean of 3 boots, residuals -3.0%..+1.7%"),
    "mla": (1.0332, True, "1 boot (GLM-5.3-Flash, 5.25 GiB -> 375,543 tokens)"),
    "dense_gqa": (1.0, False, "no dense model has booted on this machine"),
    "unknown": (1.0, False, "architecture not recognised"),
}


@dataclass(frozen=True)
class KvGeometry:
    """How one checkpoint spends KV memory, from its config.json."""

    family: Family
    #: Cache bytes that scale with the length of the context, per token.
    attn_bytes_per_token: float
    #: Recurrent state (Mamba/GDN conv + SSM, or GLM's KDA) held per in-flight
    #: SEQUENCE regardless of its length. Amortised over the context length
    #: when converting to a per-token rate.
    state_bytes_per_seq: float
    allocator_factor: float
    calibrated: bool
    factor_note: str
    #: Named breakdown, for the UI and for anyone checking the arithmetic.
    terms: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def bytes_per_token(self, ctx: int) -> float:
        """Effective bytes of KV per token of context at this context length.

        Not context-invariant, and that is the point: the per-sequence state is
        a fixed cost spread over however long the context is, so the same model
        costs 33.9 KiB/token at 131,072 and 30.3 KiB/token at 262,144. The old
        single stored rate had to be right at one length and wrong at the other
        — the store's own "measured at a different context length" warning was
        the symptom.
        """
        if ctx <= 0:
            raise ValueError(f"ctx must be positive, got {ctx!r}")
        raw = self.attn_bytes_per_token + self.state_bytes_per_seq / ctx
        return raw * self.allocator_factor

    def kib_per_token(self, ctx: int) -> float:
        return self.bytes_per_token(ctx) / 1024.0

    def tokens_for(self, kv_bytes: float, ctx: int) -> int:
        """How many tokens of KV fit in `kv_bytes` at this context length."""
        per = self.bytes_per_token(ctx)
        return int(kv_bytes // per) if per > 0 else 0


def dtype_bytes(name: str | None, default: int = 2) -> int:
    """Element size for a dtype name. Unknown names fall back to `default`
    rather than raising: a checkpoint naming a dtype we have not seen must
    still get an estimate, and 2 (bf16) is what every local checkpoint uses."""
    if not name:
        return default
    return _DTYPE_BYTES.get(str(name).lower(), default)


def _text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    tc = cfg.get("text_config")
    return tc if isinstance(tc, dict) else cfg


def kv_cache_dtype_bytes(cfg: dict[str, Any], override: str | None = None) -> int:
    """Bytes per cached K/V element.

    `override` is the launcher's --kv-cache-dtype when it is not "auto". With
    "auto" the engine follows the checkpoint: a modelopt checkpoint that
    declares `quantization_config.kv_cache_scheme` with num_bits 8 caches in
    fp8, which is half the size the model dtype would suggest. Qwen3.8-27B
    declares exactly that, and reading only the model dtype overstates its KV
    by 2x.
    """
    if override and override.lower() not in ("auto", ""):
        return dtype_bytes(override)
    scheme = (cfg.get("quantization_config") or {}).get("kv_cache_scheme")
    if isinstance(scheme, dict) and scheme.get("num_bits"):
        try:
            return max(1, int(scheme["num_bits"]) // 8)
        except (TypeError, ValueError):
            pass
    return dtype_bytes(_text_config(cfg).get("dtype") or cfg.get("dtype"))


def _gdn_state_bytes(tc: dict[str, Any], ssm_bytes: int) -> float:
    """Conv + SSM recurrent state for ONE Gated DeltaNet / Mamba layer.

    Both terms verified against vLLM's derived attention block size on this
    box: with an fp32 SSM state this arithmetic gives a mamba page of
    3,268,608 B, which is 1,596 tokens of Flash-Next attention and rounds up
    to the block size 1600 the engine reported; at bfloat16 it gives 828 ->
    832, the value the engine reported with --mamba-ssm-cache-dtype bfloat16.
    Two independent confirmations, including that the conv state is fp32 and
    holds kernel-1 taps, not kernel.
    """
    v_heads = int(tc.get("linear_num_value_heads") or 0)
    k_heads = int(tc.get("linear_num_key_heads") or 0)
    k_dim = int(tc.get("linear_key_head_dim") or 0)
    v_dim = int(tc.get("linear_value_head_dim") or 0)
    kernel = int(tc.get("linear_conv_kernel_dim") or 0)
    if not (v_heads and k_dim and v_dim):
        return 0.0
    ssm = v_heads * k_dim * v_dim * ssm_bytes
    conv_dim = 2 * k_heads * k_dim + v_heads * v_dim
    conv = conv_dim * max(kernel - 1, 0) * 4  # conv state stays fp32
    return float(ssm + conv)


def geometry(
    cfg: dict[str, Any],
    *,
    kv_cache_dtype: str | None = None,
    mamba_ssm_dtype: str | None = None,
) -> KvGeometry:
    """Read a checkpoint's KV geometry out of its config.json.

    `kv_cache_dtype` and `mamba_ssm_dtype` are the LAUNCH flags
    (--kv-cache-dtype, --mamba-ssm-cache-dtype). They belong here because they
    change the cache layout, and leaving them out is what made one stored rate
    describe two different servers.
    """
    tc = _text_config(cfg)
    layer_types = list(tc.get("layer_types") or [])
    kvb = kv_cache_dtype_bytes(cfg, kv_cache_dtype)
    notes: list[str] = []
    terms: dict[str, float] = {}

    n_linear = sum(1 for t in layer_types if "linear" in str(t))
    n_layers = int(tc.get("num_hidden_layers") or len(layer_types) or 0)
    n_attn = (len(layer_types) - n_linear) if layer_types else n_layers

    # ---- MLA (GLM-5.3): one latent vector per token per attention layer.
    if tc.get("kv_lora_rank"):
        lora = int(tc["kv_lora_rank"])
        rope = int(tc.get("qk_rope_head_dim") or 0)
        latent = n_attn * (lora + rope) * kvb
        terms["mla_latent"] = float(latent)
        # The sparse-attention indexer keeps one compressed key per token on
        # every layer that has one. Its dtype is not declared in config.json;
        # the model dtype is assumed, which is what matches this box's only
        # GLM measurement. An fp8 indexer would be half this.
        idx_dim = int(tc.get("index_head_dim") or 0)
        idx = n_attn * idx_dim * kvb if idx_dim else 0
        if idx:
            terms["indexer_keys"] = float(idx)
            notes.append(
                "indexer key dtype is not declared in config.json; the model "
                "dtype is assumed (fp8 would halve this term)"
            )
        lin = tc.get("linear_attn_config") or {}
        state = 0.0
        if lin:
            heads = int(lin.get("num_heads") or 0)
            hd = int(lin.get("head_dim") or 0)
            kernel = int(lin.get("short_conv_kernel_size") or 0)
            n_kda = len(lin.get("kda_layers") or []) or n_linear
            ssm_b = dtype_bytes(mamba_ssm_dtype, 4)
            per = heads * hd * hd * ssm_b + heads * hd * max(kernel - 1, 0) * 4
            state = float(n_kda * per)
            terms["kda_state_per_seq"] = state
        f, cal, why = _FACTORS["mla"]
        return KvGeometry("mla", float(latent + idx), state, f, cal, why, terms, tuple(notes))

    # ---- Hybrid GDN (Qwen3.5 / Qwen4-Exp) and the dense case.
    head_dim = int(tc.get("head_dim") or 0)
    if not head_dim:
        hidden = int(tc.get("hidden_size") or 0)
        heads = int(tc.get("num_attention_heads") or 0)
        head_dim = hidden // heads if hidden and heads else 0
    kv_heads = int(tc.get("num_key_value_heads") or tc.get("num_attention_heads") or 0)
    if not (head_dim and kv_heads and n_attn):
        return KvGeometry("unknown", 0.0, 0.0, 1.0, False,
                          _FACTORS["unknown"][2], {},
                          ("config.json declares no usable attention geometry",))

    per_layer = 2 * kv_heads * head_dim * kvb
    attn = float(n_attn * per_layer)
    terms["attention_kv"] = attn

    if not n_linear:
        f, cal, why = _FACTORS["dense_gqa"]
        notes.append(
            "no measurement of a dense model exists on this machine, so the "
            "allocator correction is 1.0 and this figure is a floor"
        )
        return KvGeometry("dense_gqa", attn, 0.0, f, cal, why, terms, tuple(notes))

    ssm_b = dtype_bytes(mamba_ssm_dtype or tc.get("mamba_ssm_dtype"), 4)
    state = float(n_linear * _gdn_state_bytes(tc, ssm_b))
    terms["gdn_state_per_seq"] = state

    family: Family = "hybrid_gdn"
    idx_dim = int(tc.get("indexer_head_dim") or 0)
    if idx_dim:
        # QSA: alongside the full K/V, each attention layer keeps a COMPRESSED
        # key per token (the raw-key ring that feeds the indexer is bounded by
        # indexer_budget and is a per-sequence cost, not a per-token one).
        family = "qsa_hybrid"
        idx_heads = int(tc.get("indexer_kv_heads") or 1)
        ratio = int(tc.get("indexer_compress_ratio") or 1) or 1
        comp = n_attn * idx_heads * idx_dim * kvb / ratio
        terms["qsa_compressed_keys"] = float(comp)
        attn += comp
        budget = int(tc.get("indexer_budget") or 0)
        if budget:
            ring = n_attn * idx_heads * idx_dim * kvb * budget
            terms["qsa_key_ring_per_seq"] = float(ring)
            state += ring

    f, cal, why = _FACTORS[family]
    return KvGeometry(family, attn, state, f, cal, why, terms, tuple(notes))


def summarise(geo: KvGeometry, ctx: int) -> dict[str, Any]:
    """A JSON-safe description, for /api and for the panel's tooltip."""
    return {
        "family": geo.family,
        "kib_per_token": round(geo.kib_per_token(ctx), 3),
        "attn_bytes_per_token": round(geo.attn_bytes_per_token, 1),
        "state_bytes_per_seq": round(geo.state_bytes_per_seq, 1),
        "allocator_factor": geo.allocator_factor,
        "calibrated": geo.calibrated,
        "factor_note": geo.factor_note,
        "terms": {k: round(v, 1) for k, v in geo.terms.items()},
        "notes": list(geo.notes),
    }


def tokens_at(geo: KvGeometry, kv_gib: float, ctx: int) -> int:
    return geo.tokens_for(kv_gib * float(1 << 30), ctx)


def ceil_ctx_for_agents(kv_tokens: int, agents: int) -> int:
    """Largest per-agent context `agents` of them can hold at once."""
    if agents < 1:
        raise ValueError(f"agents must be >= 1, got {agents!r}")
    return max(0, math.floor(kv_tokens / agents))
