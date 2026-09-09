"""Tests for servedeck.capacity — SPEC.md §3.

capacity.py is pure (no file I/O, no subprocess, no network), so every test
here builds ModelInputs / LiveFacts by hand. Numbers asserted against are
either transcribed verbatim from SPEC.md §3 (the two real-boot acceptance
figures, the blue-green table) or computed once by hand from the same
formulas SPEC.md §3 specifies (budget/kv/weights/overhead arithmetic) and
hardcoded here so a regression shows up as a failing assertion, not a
silently-changed number.

THE ACCEPTANCE TEST IS NON-NEGOTIABLE: both real boots must reproduce to
<0.2% relative error against the vLLM-printed token counts.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from servedeck.capacity import (
    FLASHNEXT_MIN_UTIL,
    FRAG_MARGIN_GIB,
    GPU_TOTAL_GIB,
    GPU_TOTAL_MIB,
    MAMBA_MAX_NUM_SEQS_INLINE,
    OVERHEAD_GIB_DEFAULT,
    TRAINING_MARKER_PATHS,
    UTIL_THIN_MARGIN,
    VRAM_GUARD_HEADROOM_MIB,
    LiveFacts,
    ModelInputs,
    blue_green,
    compute,
)

# ---------------------------------------------------------------------------
# Shared fixtures — real, measured model configurations from SPEC.md §0.
# ---------------------------------------------------------------------------

FLASHNEXT = ModelInputs(
    repo_id="RadixArk/Qwen3.8-Flash-Next-NVFP4",
    backend="flashnext",
    model_max_ctx=262144,
    weights_gib=78.47,
    weights_source="measured",
    kv_kib_per_token=30.39,
    overhead_gib=4.47,
    trust="measured",
)

NVFP4_27B = ModelInputs(
    repo_id="RadixArk/Qwen3.8-27B-NVFP4",
    backend="inline",
    model_max_ctx=262144,
    weights_gib=20.75,
    weights_source="measured",
    kv_kib_per_token=37.99,
    overhead_gib=4.33,
    trust="measured",
)

FP8_27B = ModelInputs(
    repo_id="Qwen/Qwen3.8-27B-FP8",
    backend="inline",
    model_max_ctx=262144,
    weights_gib=28.51,
    weights_source="measured",
    kv_kib_per_token=37.99,
    trust="measured",
)

AWQ_MTP = ModelInputs(
    repo_id="twolven/Qwen3.8-27B-abliterated-AWQ-MTP",
    backend="inline",
    model_max_ctx=262144,
    weights_gib=18.21,
    weights_source="estimated",
    kv_kib_per_token=35.4,
    trust="estimated",
)


def _code(findings, code: str):
    """Return the single Finding with this code, or None."""
    matches = [f for f in findings if f.code == code]
    assert len(matches) <= 1, f"{code} fired more than once: {matches}"
    return matches[0] if matches else None


def _has(findings, code: str) -> bool:
    return any(f.code == code for f in findings)


# ---------------------------------------------------------------------------
# Constants — provenance sanity (SPEC.md §3).
# ---------------------------------------------------------------------------


def test_constants_match_spec():
    assert GPU_TOTAL_MIB == 97887
    assert GPU_TOTAL_GIB == pytest.approx(95.5927734375, abs=1e-9)
    assert OVERHEAD_GIB_DEFAULT == 4.7
    assert FRAG_MARGIN_GIB == 1.0
    assert UTIL_THIN_MARGIN == 0.97
    assert VRAM_GUARD_HEADROOM_MIB == 4096
    assert MAMBA_MAX_NUM_SEQS_INLINE == 128
    assert FLASHNEXT_MIN_UTIL == 0.90


def test_training_markers_are_configurable_not_hardcoded():
    """Marker paths come from config, so this works on any machine.

    They used to be three literals from one developer's home directory.
    """
    import os

    from servedeck import capacity as cap
    from servedeck import config

    os.environ["SERVEDECK_TRAINING_MARKERS"] = "/tmp/a-marker:/tmp/b-marker"
    try:
        config.reset()
        assert tuple(cap._cfg_markers()) == ("/tmp/a-marker", "/tmp/b-marker")
    finally:
        del os.environ["SERVEDECK_TRAINING_MARKERS"]
        config.reset()


# ---------------------------------------------------------------------------
# THE ACCEPTANCE TEST — both real boots, <0.2% relative error.
# ---------------------------------------------------------------------------


def test_acceptance_flashnext_reproduces_real_boot():
    r = compute(FLASHNEXT, util=0.96, ctx=262144, max_num_seqs=128)

    assert r.kv_gib == pytest.approx(8.83, rel=0.002)
    vllm_printed_tokens = 304_653
    assert abs(r.kv_tokens - vllm_printed_tokens) / vllm_printed_tokens < 0.002
    assert r.concurrency_x == pytest.approx(1.16, rel=0.002)

    # A well-formed request at these exact real-boot settings raises nothing.
    assert r.findings == ()
    assert r.can_apply is True
    assert r.confidence == "measured"
    assert r.agents_at_ctx == 1
    assert r.effective_parallel == 1
    assert r.max_single_ctx == 262144


def test_acceptance_27b_nvfp4_reproduces_real_boot():
    r = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=128)

    assert r.kv_gib == pytest.approx(22.72, rel=0.002)
    vllm_printed_tokens = 627_117
    assert abs(r.kv_tokens - vllm_printed_tokens) / vllm_printed_tokens < 0.002
    assert r.concurrency_x == pytest.approx(2.39, rel=0.002)

    assert r.findings == ()
    assert r.can_apply is True
    assert r.confidence == "measured"
    assert r.agents_at_ctx == 2
    assert r.effective_parallel == 2
    assert r.max_single_ctx == 262144


def test_acceptance_budget_and_bar_formulas():
    # budget_gib = util * GPU_TOTAL_GIB; kv_gib = budget - weights - overhead;
    # bar percentages are of GPU_TOTAL_GIB and sum to <=100.
    r = compute(FLASHNEXT, util=0.96, ctx=262144, max_num_seqs=128)
    assert r.budget_gib == pytest.approx(0.96 * GPU_TOTAL_GIB)
    assert r.kv_gib == pytest.approx(r.budget_gib - 78.47 - 4.47)
    total_pct = r.bar.weights_pct + r.bar.kv_pct + r.bar.overhead_pct + r.bar.free_pct
    assert total_pct == pytest.approx(100.0, abs=1e-6)
    assert r.bar.weights_pct == pytest.approx(78.47 / GPU_TOTAL_GIB * 100.0)


# ---------------------------------------------------------------------------
# blue_green() — reproduces the §3 table exactly.
# ---------------------------------------------------------------------------


def test_blue_green_flashnext_impossible():
    v = blue_green(FLASHNEXT, FLASHNEXT, util_a=0.47, util_b=0.47, ctx=262144)
    assert v.combined_gib == pytest.approx(166.3, abs=0.05)
    assert v.combined_gib == pytest.approx(2 * 78.47 + 2 * OVERHEAD_GIB_DEFAULT)
    assert v.gpu_total_gib == pytest.approx(95.6, abs=0.05)
    assert v.feasible is False
    assert v.remaining_kv_gib == 0.0
    assert v.kv_tokens == 0
    assert "impossible" in v.reason.lower()


def test_blue_green_27b_nvfp4_feasible():
    v = blue_green(NVFP4_27B, NVFP4_27B, util_a=0.47, util_b=0.47, ctx=262144)
    assert v.combined_gib == pytest.approx(50.9, abs=0.05)
    assert v.feasible is True
    assert v.remaining_kv_gib == pytest.approx(44.7, abs=0.05)


def test_blue_green_27b_fp8_feasible():
    v = blue_green(FP8_27B, FP8_27B, util_a=0.47, util_b=0.47, ctx=262144)
    assert v.combined_gib == pytest.approx(66.4, abs=0.05)
    assert v.feasible is True


def test_blue_green_awq_mtp_feasible():
    v = blue_green(AWQ_MTP, AWQ_MTP, util_a=0.47, util_b=0.47, ctx=262144)
    assert v.combined_gib == pytest.approx(45.8, abs=0.05)
    assert v.feasible is True


def test_blue_green_fits_formula_is_weights_plus_overhead_plus_margin():
    # fits = (w_a + w_b + 2*overhead + FRAG_MARGIN) <= GPU_TOTAL_GIB (SPEC §3).
    v = blue_green(NVFP4_27B, FP8_27B, util_a=0.47, util_b=0.47, ctx=262144)
    expected_combined = 20.75 + 28.51 + 2 * OVERHEAD_GIB_DEFAULT
    assert v.combined_gib == pytest.approx(expected_combined)
    assert v.required_gib == pytest.approx(expected_combined + FRAG_MARGIN_GIB)
    assert v.feasible == (v.required_gib <= GPU_TOTAL_GIB)


# ---------------------------------------------------------------------------
# BLOCKING findings — each fires on its documented trigger.
# ---------------------------------------------------------------------------


def test_model_unservable_blocks_and_short_circuits():
    m = ModelInputs(
        repo_id="OBLITERATUS/Qwen3.8-27B-OBLITERATED",
        backend="unknown",
        model_max_ctx=0,
        servable=False,
        unservable_reason="GGUF-only, no known launcher/backend",
    )
    r = compute(m, util=0.5, ctx=1000, max_num_seqs=128)
    f = _code(r.findings, "MODEL_UNSERVABLE")
    assert f is not None
    assert f.level == "block"
    assert "OBLITERATUS" in f.detail
    assert r.can_apply is False
    assert r.kv_tokens == 0
    assert r.confidence == "unknown"


def test_unknown_capacity_blocks_for_qwen4_exp_refused_estimate():
    m = ModelInputs(
        repo_id="RadixArk/Qwen3.8-Flash-Next-NVFP4",
        backend="flashnext",
        model_max_ctx=262144,
        weights_gib=None,
        weights_source="unknown",
        kv_kib_per_token=30.39,
        overhead_gib=4.47,
        trust="unknown",
        model_type="qwen4_exp",
    )
    r = compute(m, util=0.96, ctx=262144, max_num_seqs=128)
    f = _code(r.findings, "UNKNOWN_CAPACITY")
    assert f is not None
    assert f.level == "warn"
    assert "125.91" in f.detail and "78.47" in f.detail
    assert r.can_apply is True
    assert r.confidence == "unknown"


@pytest.mark.parametrize("bad_util", [1.5, 0.0, -0.1])
def test_util_out_of_range_blocks(bad_util):
    r = compute(NVFP4_27B, util=bad_util, ctx=262144, max_num_seqs=128)
    f = _code(r.findings, "UTIL_OUT_OF_RANGE")
    assert f is not None
    assert f.level == "block"
    assert r.can_apply is False


def test_util_in_range_does_not_trigger_util_out_of_range():
    r = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=128)
    assert not _has(r.findings, "UTIL_OUT_OF_RANGE")


@pytest.mark.parametrize("bad_seqs", [0, -1, -5])
def test_max_num_seqs_invalid_blocks(bad_seqs):
    r = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=bad_seqs)
    f = _code(r.findings, "MAX_NUM_SEQS_INVALID")
    assert f is not None
    assert f.level == "block"
    assert r.can_apply is False
    assert r.effective_parallel == 0


def test_ctx_above_ceiling_blocks():
    r = compute(NVFP4_27B, util=0.50, ctx=300_000, max_num_seqs=128)
    f = _code(r.findings, "CTX_ABOVE_CEILING")
    assert f is not None
    assert f.level == "block"
    assert "300,000" in f.detail and "262,144" in f.detail
    assert r.can_apply is False


def test_ctx_at_ceiling_does_not_block():
    r = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=128)
    assert not _has(r.findings, "CTX_ABOVE_CEILING")


def test_weights_exceed_budget_blocks_with_util_fix_action():
    # util=0.10 on the 27B NVFP4 config: budget << weights+overhead.
    r = compute(NVFP4_27B, util=0.10, ctx=262144, max_num_seqs=128)
    f = _code(r.findings, "WEIGHTS_EXCEED_BUDGET")
    assert f is not None
    assert f.level == "block"
    assert r.kv_gib < 0
    assert r.kv_tokens == 0
    assert r.can_apply is False
    assert f.fix_action == {"field": "util", "value": pytest.approx(0.27)}


def test_kv_too_small_for_one_ctx_blocks_with_ctx_fix_action():
    # util=0.30 leaves ~3.6 GiB of KV, far short of one 262144-token request.
    r = compute(NVFP4_27B, util=0.30, ctx=262144, max_num_seqs=128)
    f = _code(r.findings, "KV_TOO_SMALL_FOR_ONE_CTX")
    assert f is not None
    assert f.level == "block"
    assert r.kv_gib > 0  # there IS a KV budget, just not enough for one ctx
    assert r.kv_tokens < 262144
    assert f.fix_action == {"field": "ctx", "value": r.kv_tokens}
    assert r.can_apply is False


def test_flashnext_util_too_low_blocks():
    tiny = ModelInputs(
        repo_id="tiny-flashnext",
        backend="flashnext",
        model_max_ctx=262144,
        weights_gib=1.0,
        weights_source="measured",
        kv_kib_per_token=30.39,
        overhead_gib=1.0,
        trust="measured",
    )
    r = compute(tiny, util=0.85, ctx=1000, max_num_seqs=128)
    f = _code(r.findings, "FLASHNEXT_UTIL_TOO_LOW")
    assert f is not None
    assert f.level == "block"
    assert "0.90" in f.detail
    assert r.can_apply is False


def test_flashnext_util_at_minimum_does_not_block():
    tiny = ModelInputs(
        repo_id="tiny-flashnext",
        backend="flashnext",
        model_max_ctx=262144,
        weights_gib=1.0,
        weights_source="measured",
        kv_kib_per_token=30.39,
        overhead_gib=1.0,
        trust="measured",
    )
    r = compute(tiny, util=FLASHNEXT_MIN_UTIL, ctx=1000, max_num_seqs=128)
    assert not _has(r.findings, "FLASHNEXT_UTIL_TOO_LOW")


def test_flashnext_util_too_low_does_not_apply_to_inline_backend():
    # Same low util on an INLINE model must not trigger the flashnext-only rule.
    r = compute(NVFP4_27B, util=0.10, ctx=262144, max_num_seqs=128)
    assert not _has(r.findings, "FLASHNEXT_UTIL_TOO_LOW")


def test_not_enough_free_vram_blocks_same_arithmetic_as_qwen_server_run():
    # qwen-server-run.sh:104-135: free = total-used+own; want = int(total*util);
    # need = want + headroom; block iff free < need.
    live = LiveFacts(gpu_responsive=True, total_mib=97887, used_mib=90000, own_mib=0)
    r = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=128, live=live)
    f = _code(r.findings, "NOT_ENOUGH_FREE_VRAM")
    assert f is not None
    assert f.level == "block"
    assert "7887" in f.detail  # free_mib
    # need_mib is the utilization budget itself (48943), NOT budget + headroom.
    # Adding the 4096 MiB fragmentation margin on top of a fraction-of-total
    # budget makes the check unsatisfiable for any util >= 0.958 -- see
    # test_high_util_is_not_impossible below. The margin is reported separately
    # as the THIN_VRAM_MARGIN warning.
    assert "48943" in f.detail  # need_mib = int(97887 * 0.50)
    assert r.can_apply is False


def test_not_enough_free_vram_discounts_own_processes():
    # Same used_mib, but our own vllm holds 50000 MiB of it — discounting
    # that must turn a blocking case into a passing one.
    # need_mib = int(97887*0.50) + 4096 = 53039. free_mib = total-used+own.
    # own=0   -> free=7887  (< 53039, blocks — matches the sibling test above)
    # own=40000 -> free=47887 (still < 53039 — NOT enough to flip the verdict;
    #   an earlier version of this test used 40000 and asserted no-block,
    #   which was simply arithmetically wrong for this used_mib/util pair)
    # own=50000 -> free=57887 (>= 53039, correctly clears the guard)
    tight = LiveFacts(gpu_responsive=True, total_mib=97887, used_mib=90000, own_mib=0)
    r_tight = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=128, live=tight)
    assert _has(r_tight.findings, "NOT_ENOUGH_FREE_VRAM")

    discounted = LiveFacts(gpu_responsive=True, total_mib=97887, used_mib=90000, own_mib=50000)
    r_ok = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=128, live=discounted)
    assert not _has(r_ok.findings, "NOT_ENOUGH_FREE_VRAM")


def test_not_enough_free_vram_absent_with_plenty_of_headroom():
    live = LiveFacts(gpu_responsive=True, total_mib=97887, used_mib=1000, own_mib=0)
    r = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=128, live=live)
    assert not _has(r.findings, "NOT_ENOUGH_FREE_VRAM")


def test_training_marker_blocks_with_all_hit_paths_listed():
    """Every marker that was hit must be named, so the user knows what to remove."""
    hits = ["/tmp/marker-one", "/tmp/marker-two"]
    live = LiveFacts(training_markers=hits)
    r = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=128, live=live)
    f = _code(r.findings, "TRAINING_MARKER")
    assert f is not None
    assert f.level == "block"
    for hit in hits:
        assert hit in f.detail
    assert r.can_apply is False


def test_training_marker_absent_when_no_markers_present():
    live = LiveFacts(training_markers=[])
    r = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=128, live=live)
    assert not _has(r.findings, "TRAINING_MARKER")


def test_ptrace_blocks_ple_for_flashnext_with_nonzero_scope():
    tiny = ModelInputs(
        repo_id="tiny-flashnext",
        backend="flashnext",
        model_max_ctx=262144,
        weights_gib=1.0,
        weights_source="measured",
        kv_kib_per_token=30.39,
        overhead_gib=1.0,
        trust="measured",
    )
    live = LiveFacts(ptrace_scope=1)
    r = compute(tiny, util=0.95, ctx=1000, max_num_seqs=128, live=live)
    f = _code(r.findings, "PTRACE_BLOCKS_PLE")
    assert f is not None
    assert f.level == "block"
    # NEVER run sudo — must be a copyable command, not an executed one.
    assert f.fix_action == {
        "type": "manual_command",
        "command": "sudo sysctl -w kernel.yama.ptrace_scope=0",
    }
    assert r.can_apply is False


def test_ptrace_blocks_ple_absent_for_inline_backend():
    # Same relaxed ptrace_scope on an INLINE model must not fire the
    # flashnext-only PLE guard.
    live = LiveFacts(ptrace_scope=1)
    r = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=128, live=live)
    assert not _has(r.findings, "PTRACE_BLOCKS_PLE")


def test_ptrace_blocks_ple_absent_when_scope_already_zero():
    tiny = ModelInputs(
        repo_id="tiny-flashnext",
        backend="flashnext",
        model_max_ctx=262144,
        weights_gib=1.0,
        weights_source="measured",
        kv_kib_per_token=30.39,
        overhead_gib=1.0,
        trust="measured",
    )
    live = LiveFacts(ptrace_scope=0)
    r = compute(tiny, util=0.95, ctx=1000, max_num_seqs=128, live=live)
    assert not _has(r.findings, "PTRACE_BLOCKS_PLE")


def test_gpu_unresponsive_blocks_and_suppresses_vram_check():
    live = LiveFacts(gpu_responsive=False, used_mib=90000)
    r = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=128, live=live)
    f = _code(r.findings, "GPU_UNRESPONSIVE")
    assert f is not None
    assert f.level == "block"
    assert "nvidia-smi -L" in f.detail
    assert r.can_apply is False
    # gpu_responsive=False short-circuits the VRAM arithmetic entirely.
    assert not _has(r.findings, "NOT_ENOUGH_FREE_VRAM")


def test_gpu_responsive_true_does_not_block():
    live = LiveFacts(gpu_responsive=True, used_mib=1000)
    r = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=128, live=live)
    assert not _has(r.findings, "GPU_UNRESPONSIVE")


# ---------------------------------------------------------------------------
# WARNING findings — each fires on its documented trigger.
# ---------------------------------------------------------------------------


def test_thin_margin_warns_at_and_above_threshold():
    tiny = ModelInputs(
        repo_id="tiny-inline",
        backend="inline",
        model_max_ctx=262144,
        weights_gib=1.0,
        weights_source="measured",
        kv_kib_per_token=37.99,
        overhead_gib=1.0,
        trust="measured",
    )
    r = compute(tiny, util=UTIL_THIN_MARGIN, ctx=1000, max_num_seqs=128)
    f = _code(r.findings, "THIN_MARGIN")
    assert f is not None
    assert f.level == "warn"
    assert r.can_apply is True  # warn only, never blocks


def test_thin_margin_absent_below_threshold():
    tiny = ModelInputs(
        repo_id="tiny-inline",
        backend="inline",
        model_max_ctx=262144,
        weights_gib=1.0,
        weights_source="measured",
        kv_kib_per_token=37.99,
        overhead_gib=1.0,
        trust="measured",
    )
    r = compute(tiny, util=0.96, ctx=1000, max_num_seqs=128)
    assert not _has(r.findings, "THIN_MARGIN")


def test_estimated_only_warns_and_cites_the_25pct_figure():
    r = compute(AWQ_MTP, util=0.5, ctx=262144, max_num_seqs=128)
    f = _code(r.findings, "ESTIMATED_ONLY")
    assert f is not None
    assert f.level == "warn"
    assert "~25% optimistic" in f.detail
    assert r.can_apply is True


def test_estimated_only_absent_when_measured():
    r = compute(NVFP4_27B, util=0.5, ctx=262144, max_num_seqs=128)
    assert not _has(r.findings, "ESTIMATED_ONLY")


def test_measured_other_ctx_warns_and_shows_both_rates():
    m = ModelInputs(
        repo_id="RadixArk/Qwen3.8-Flash-Next-NVFP4",
        backend="flashnext",
        model_max_ctx=262144,
        weights_gib=78.47,
        weights_source="measured",
        kv_kib_per_token=33.85,  # measured at 131072, not the requested 262144
        overhead_gib=4.47,
        trust="measured_other_ctx",
        used_ctx_for_rate=131072,
        known_kv_rates={131072: 33.85},
    )
    r = compute(m, util=0.96, ctx=262144, max_num_seqs=128)
    f = _code(r.findings, "MEASURED_OTHER_CTX")
    assert f is not None
    assert f.level == "warn"
    assert "33.85" in f.detail
    assert "131,072" in f.detail  # both the used rate and its ctx are shown
    assert "262,144" in f.detail
    assert r.confidence == "measured_other_ctx"
    assert r.can_apply is True


def test_measured_other_ctx_absent_for_exact_ctx_match():
    r = compute(FLASHNEXT, util=0.96, ctx=262144, max_num_seqs=128)
    assert not _has(r.findings, "MEASURED_OTHER_CTX")


def test_seqs_below_agents_warns_and_caps_effective_parallel():
    r = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=1)
    f = _code(r.findings, "SEQS_BELOW_AGENTS")
    assert f is not None
    assert f.level == "warn"
    assert r.agents_at_ctx == 2
    assert r.effective_parallel == 1  # capped by max_num_seqs, not agents_at_ctx
    assert r.can_apply is True


def test_seqs_below_agents_absent_when_seqs_is_the_looser_bound():
    r = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=128)
    assert not _has(r.findings, "SEQS_BELOW_AGENTS")


def test_subagents_not_a_guarantee_exact_message_shape():
    live = LiveFacts(codex_max_subagents=8)
    r = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=128, live=live)
    f = _code(r.findings, "SUBAGENTS_NOT_A_GUARANTEE")
    assert f is not None
    assert f.level == "warn"
    assert r.effective_parallel == 2
    # REQUIRED label, verbatim (SPEC §3): c=8, e=2, c-e=6.
    assert f.detail == (
        "CODEX_MAX_SUBAGENTS=8 is a Codex-side orchestration cap, not a GPU "
        "guarantee. This configuration serves 2 in parallel; the other 6 "
        "will queue."
    )
    assert r.can_apply is True


def test_subagents_not_a_guarantee_absent_when_cap_fits():
    live = LiveFacts(codex_max_subagents=2)  # exactly effective_parallel, not above
    r = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=128, live=live)
    assert not _has(r.findings, "SUBAGENTS_NOT_A_GUARANTEE")


def test_mamba_seqs_cap_warns_for_inline_over_128():
    tiny = ModelInputs(
        repo_id="tiny-inline",
        backend="inline",
        model_max_ctx=262144,
        weights_gib=1.0,
        weights_source="measured",
        kv_kib_per_token=37.99,
        overhead_gib=1.0,
        trust="measured",
    )
    r = compute(tiny, util=0.90, ctx=1000, max_num_seqs=200)
    f = _code(r.findings, "MAMBA_SEQS_CAP")
    assert f is not None
    assert f.level == "warn"
    assert "128" in f.detail
    assert r.can_apply is True


def test_mamba_seqs_cap_absent_at_128_and_for_flashnext():
    tiny_inline = ModelInputs(
        repo_id="tiny-inline",
        backend="inline",
        model_max_ctx=262144,
        weights_gib=1.0,
        weights_source="measured",
        kv_kib_per_token=37.99,
        overhead_gib=1.0,
        trust="measured",
    )
    r_at_cap = compute(tiny_inline, util=0.90, ctx=1000, max_num_seqs=128)
    assert not _has(r_at_cap.findings, "MAMBA_SEQS_CAP")

    tiny_flashnext = ModelInputs(
        repo_id="tiny-flashnext",
        backend="flashnext",
        model_max_ctx=262144,
        weights_gib=1.0,
        weights_source="measured",
        kv_kib_per_token=30.39,
        overhead_gib=1.0,
        trust="measured",
    )
    r_flashnext = compute(tiny_flashnext, util=0.95, ctx=1000, max_num_seqs=200)
    assert not _has(r_flashnext.findings, "MAMBA_SEQS_CAP")


def test_ptrace_left_relaxed_warns_when_scope_zero_and_ready_or_stopped():
    for state in ("READY", "STOPPED"):
        live = LiveFacts(ptrace_scope=0, actual_state=state)
        r = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=128, live=live)
        f = _code(r.findings, "PTRACE_LEFT_RELAXED")
        assert f is not None, f"expected PTRACE_LEFT_RELAXED for actual_state={state}"
        assert f.level == "warn"
        assert f.fix_action == {
            "type": "manual_command",
            "command": "sudo sysctl -w kernel.yama.ptrace_scope=1",
        }
        assert r.can_apply is True


def test_ptrace_left_relaxed_absent_while_starting():
    live = LiveFacts(ptrace_scope=0, actual_state="STARTING")
    r = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=128, live=live)
    assert not _has(r.findings, "PTRACE_LEFT_RELAXED")


# ---------------------------------------------------------------------------
# Derived-value formulas, independent of any finding.
# ---------------------------------------------------------------------------


def test_concurrency_and_agents_formulas():
    r = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=128)
    assert r.concurrency_x == pytest.approx(r.kv_tokens / 262144)
    assert r.agents_at_ctx == math.floor(r.concurrency_x)
    assert r.max_single_ctx == min(262144, r.kv_tokens)
    assert r.effective_parallel == min(r.agents_at_ctx, 128)


def test_can_apply_false_iff_any_blocking_finding():
    ok = compute(NVFP4_27B, util=0.50, ctx=262144, max_num_seqs=128)
    assert ok.can_apply is True
    assert not any(f.level == "block" for f in ok.findings)

    blocked = compute(NVFP4_27B, util=0.50, ctx=300_000, max_num_seqs=128)
    assert blocked.can_apply is False
    assert any(f.level == "block" for f in blocked.findings)


def test_finding_levels_are_only_block_or_warn():
    live = LiveFacts(
        gpu_responsive=True,
        total_mib=97887,
        used_mib=1000,
        training_markers=["/tmp/servedeck-test/training_in_progress"],
        ptrace_scope=0,
        actual_state="READY",
        codex_max_subagents=8,
    )
    r = compute(NVFP4_27B, util=0.99, ctx=262144, max_num_seqs=1, live=live)
    assert r.findings  # sanity: this config actually triggers something
    for f in r.findings:
        assert f.level in ("block", "warn")


def test_high_util_is_not_impossible():
    """Regression: util 0.96 must be startable when the VRAM is our own.

    The old guard computed need = util*total + 4096, which exceeds the card for
    any util >= 0.958 -- so it blocked a configuration this workstation runs
    every day (Flash-Next at util 0.96, 94.5 GiB held by our own server).
    """
    live = LiveFacts(
        gpu_responsive=True,
        total_mib=97887,
        used_mib=94519,   # our own Flash-Next server, live figures
        own_mib=94486,
    )
    r = compute(FLASHNEXT, util=0.96, ctx=262144, max_num_seqs=1, live=live)
    assert _code(r.findings, "NOT_ENOUGH_FREE_VRAM") is None, (
        "util 0.96 was reported unstartable, but the server runs at 0.96"
    )


def test_thin_margin_warns_without_blocking():
    live = LiveFacts(gpu_responsive=True, total_mib=97887, used_mib=94519, own_mib=94486)
    r = compute(FLASHNEXT, util=0.96, ctx=262144, max_num_seqs=1, live=live)
    w = _code(r.findings, "THIN_VRAM_MARGIN")
    assert w is not None and w.level == "warn"


def test_unknown_weights_never_fabricate_a_kv_figure():
    """Regression: unknown weights were substituted with 0.0.

    That made the KV budget absorb the whole card — 2,811,134 tokens for a
    model whose size nobody knows, displayed next to real models reporting a
    tenth of that. Blocking the Apply button is not enough; the number itself
    must not be invented.
    """
    m = ModelInputs(
        repo_id="unknown/model", backend="flashnext", model_max_ctx=262144,
        weights_gib=None, weights_source="unknown", kv_kib_per_token=30.39,
        overhead_gib=4.7, trust="estimated", servable=True,
        unservable_reason=None, model_type="qwen4_exp",
        used_ctx_for_rate=262144, known_kv_rates={},
    )
    r = compute(m, util=0.95, ctx=262144, max_num_seqs=1)
    assert _code(r.findings, "UNKNOWN_CAPACITY") is not None
    assert r.kv_tokens == 0, f"fabricated {r.kv_tokens:,} tokens from unknown weights"
    assert r.kv_gib == 0.0
    assert r.agents_at_ctx == 0


def test_unknown_weights_do_not_block_the_launch():
    """Regression: UNKNOWN_CAPACITY used to be a blocker, which was a dead end.

    Booting is the only way to learn a model's real weight size, so blocking
    the launch made the condition permanent: unknown -> cannot start -> stays
    unknown forever. Refusing to PREDICT is right; refusing to TRY is not.
    """
    m = ModelInputs(
        repo_id="unknown/model", backend="flashnext", model_max_ctx=262144,
        weights_gib=None, weights_source="unknown", kv_kib_per_token=30.39,
        overhead_gib=4.7, trust="estimated", servable=True,
        unservable_reason=None, model_type="qwen4_exp",
        used_ctx_for_rate=262144, known_kv_rates={},
    )
    r = compute(m, util=0.95, ctx=262144, max_num_seqs=1)
    f = _code(r.findings, "UNKNOWN_CAPACITY")
    assert f is not None and f.level == "warn"
    assert r.can_apply is True, "a model whose size is unknown must still be startable"
    # but nothing may be fabricated from the weights we do not have
    assert r.kv_tokens == 0
    assert _code(r.findings, "KV_TOO_SMALL_FOR_ONE_CTX") is None, (
        "KV findings derived from unknown weights are meaningless and must be suppressed"
    )


def test_training_markers_come_from_the_config_file_too(config_path):
    """`training_markers` in servedeck.toml was parsed into Config and then
    read by nothing: the only source was $SERVEDECK_TRAINING_MARKERS.

    A configured guard that cannot fire is worse than an absent one — it reads
    as switched on. `~` is expanded, because a config file is exactly where
    someone writes `~/run/training_in_progress`.
    """
    from servedeck import capacity as cap

    config_path('training_markers = ["~/a-marker", "/tmp/b-marker"]\n')
    got = tuple(cap._cfg_markers())
    assert got == (str(Path.home() / "a-marker"), "/tmp/b-marker"), got
    assert cap.TRAINING_MARKER_PATHS == got, "refresh_limits() must re-read them"


def test_env_var_overrides_the_configured_markers(config_path, monkeypatch):
    """A test or a one-off run must be able to override without editing the
    file that describes the machine."""
    from servedeck import capacity as cap

    config_path('training_markers = ["/tmp/from-file"]\n')
    monkeypatch.setenv("SERVEDECK_TRAINING_MARKERS", "/tmp/x:/tmp/y")
    assert tuple(cap._cfg_markers()) == ("/tmp/x", "/tmp/y")


# ---------------------------------------------------------------------------
# Context bounds — "Context per agent is hardcoded and offered even for models
# that cannot run it"
# ---------------------------------------------------------------------------
def test_ctx_ceiling_is_the_model_ceiling_when_the_budget_is_generous() -> None:
    """The 27B at util 0.47 buys ~530k tokens of KV. One agent can therefore
    have the whole 262,144 the checkpoint allows, and nothing beyond it: the
    model ceiling is the binding limit here."""
    r = compute(NVFP4_27B, util=0.47, ctx=262144, max_num_seqs=1)
    assert r.ctx_max_model == 262144
    assert r.ctx_max_fit == 262144
    assert r.kv_tokens > 262144


def test_ctx_ceiling_falls_when_the_agents_have_to_share() -> None:
    """The same budget split four ways cannot give each agent the model's full
    context. The old control offered it anyway; the engine then loads weights
    for minutes and refuses."""
    one = compute(NVFP4_27B, util=0.47, ctx=262144, max_num_seqs=1)
    four = compute(NVFP4_27B, util=0.47, ctx=262144, max_num_seqs=4)
    assert four.ctx_max_fit == min(262144, one.kv_tokens // 4)
    assert four.ctx_max_fit < four.ctx_max_model, (
        "with four agents the KV budget, not the checkpoint, is the binding limit"
    )


def test_ctx_ceiling_is_what_fits_when_the_budget_is_the_smaller_limit() -> None:
    """Flash-Next at util 0.95 holds ~272k tokens for one agent — just over its
    262,144 ceiling. Two agents cannot each have that."""
    r = compute(FLASHNEXT, util=0.95, ctx=262144, max_num_seqs=2)
    assert r.ctx_max_fit == r.kv_tokens // 2
    assert r.ctx_max_fit < 262144


def test_unservable_model_offers_no_context_at_all() -> None:
    m = ModelInputs(
        repo_id="x/y", backend="inline", model_max_ctx=262144,
        servable=False, unservable_reason="GGUF-only",
    )
    r = compute(m, util=0.9, ctx=262144, max_num_seqs=1)
    assert r.ctx_max_fit == 0
