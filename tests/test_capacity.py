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


def test_training_markers_are_configurable_not_hardcoded(monkeypatch):
    """Marker paths come from the environment, so this works on any machine.

    They used to be three literals from one developer's home directory.
    """
    from servedeck import capacity as cap

    monkeypatch.setenv("SERVEDECK_TRAINING_MARKERS", "/var/tmp/a-marker:/var/tmp/b-marker")
    assert tuple(cap._cfg_markers()) == ("/var/tmp/a-marker", "/var/tmp/b-marker")


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
    assert "7,887" in f.detail  # free_mib, grouped like every other figure on the page
    # need_mib is the utilization budget itself (48943), NOT budget + headroom.
    # Adding the 4096 MiB fragmentation margin on top of a fraction-of-total
    # budget makes the check unsatisfiable for any util >= 0.958 -- see
    # test_high_util_is_not_impossible below. The margin is reported separately
    # as the THIN_VRAM_MARGIN warning.
    assert "48,943" in f.detail  # need_mib = int(97887 * 0.50)
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


def test_the_built_in_markers_are_used_when_nothing_is_configured(monkeypatch):
    """servedeck.toml is gone; the defaults live in servedeck.limits.

    A guard whose default is the empty tuple reads as switched on and fires
    never, which is the failure this list exists to avoid -- so the fallback
    has to be the real marker paths, not nothing.
    """
    from servedeck import capacity as cap
    from servedeck import limits

    monkeypatch.delenv("SERVEDECK_TRAINING_MARKERS", raising=False)
    assert tuple(cap._cfg_markers()) == limits.DEFAULT_TRAINING_MARKERS
    assert limits.DEFAULT_TRAINING_MARKERS, "an empty marker list is a guard that never fires"
    assert all(m.endswith("training_in_progress") for m in limits.DEFAULT_TRAINING_MARKERS)


def test_refresh_limits_re_reads_the_markers(monkeypatch):
    """`TRAINING_MARKER_PATHS` is read at import time; a long-lived process
    that changes the environment must be able to make it true again."""
    from servedeck import capacity as cap

    monkeypatch.setenv("SERVEDECK_TRAINING_MARKERS", "/var/tmp/x:/var/tmp/y")
    cap.refresh_limits()
    assert cap.TRAINING_MARKER_PATHS == ("/var/tmp/x", "/var/tmp/y")


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


@pytest.mark.parametrize("agents", [1, 4, 16])
def test_the_agent_count_never_lowers_the_context_ceiling(agents: int) -> None:
    """--max-model-len is a per-REQUEST ceiling. The KV pool is shared and
    vLLM's scheduler admits what fits and queues the rest, so the number of
    agents says nothing about how long one request may be.

    ctx_max_fit used to be kv_tokens // max_num_seqs. With the 27B at util 0.47
    (~530k tokens) and 4 agents that is ~132k, and on the live box (a pool of
    1,595,321 at util 0.95, 16 agents) it launched the server at 110,592 --
    and the next 110,593-token prompt failed outright. The model's own
    262,144 fits one request here, so that is the ceiling at every agent count.
    """
    r = compute(NVFP4_27B, util=0.47, ctx=262144, max_num_seqs=agents)
    assert r.kv_tokens > 262144, "precondition: one full-length request fits"
    assert r.ctx_max_fit == 262144, (
        f"with {agents} agents the context ceiling is {r.ctx_max_fit:,}: the pool "
        "was divided by the agent count again"
    )


def test_ctx_ceiling_is_what_one_request_can_hold_when_the_budget_is_smaller() -> None:
    """The only thing the KV budget may lower the ceiling for: a pool that
    cannot hold ONE request of the model's full length.

    Flash-Next at util 0.94: budget 0.94 x 95.5928 = 89.8572 GiB, minus 78.47
    weights and 4.47 overhead leaves 6.9172 GiB = 7,253,217 KiB, at 30.39
    KiB/token 238,671 tokens -- under its 262,144. One request can use all of
    them, and two agents do not halve that."""
    for agents in (1, 2):
        r = compute(FLASHNEXT, util=0.94, ctx=262144, max_num_seqs=agents)
        assert r.kv_tokens == 238_671, r.kv_tokens
        assert r.ctx_max_fit == 238_671, (agents, r.ctx_max_fit)


def test_ctx_ceiling_stays_at_the_model_when_one_request_just_fits() -> None:
    """Over-correction guard. Flash-Next at util 0.95 holds ~272k tokens, just
    over its 262,144: the model's own length is the ceiling, not the pool."""
    r = compute(FLASHNEXT, util=0.95, ctx=262144, max_num_seqs=2)
    assert r.kv_tokens > 262144
    assert r.ctx_max_fit == 262144


def test_unservable_model_offers_no_context_at_all() -> None:
    m = ModelInputs(
        repo_id="x/y", backend="inline", model_max_ctx=262144,
        servable=False, unservable_reason="GGUF-only",
    )
    r = compute(m, util=0.9, ctx=262144, max_num_seqs=1)
    assert r.ctx_max_fit == 0


# ---------------------------------------------------------------------------
# The context a launch defaults to: the longest ONE request can use
# ---------------------------------------------------------------------------
def _hybrid_pool(kv_bytes: int, attn: int, state: int):
    """Tokens of KV a hybrid server holds when configured for context L.

    A per-sequence recurrent state of `state` bytes is spread over the context,
    so each token costs attn + state / L and the pool shrinks as L shrinks
    (kvcalc.KvGeometry.bytes_per_token). Integer arithmetic, so the oracle
    below is exact."""
    return lambda length: (kv_bytes * length) // (attn * length + state)


def test_the_single_request_fit_is_the_longest_length_that_holds_one_request() -> None:
    """8 GiB of KV, 32 KiB of attention per token, a 1 GiB state per sequence.

    One request of L tokens needs 32768 * L + 2^30 bytes, so the longest that
    fits in 2^33 is (2^33 - 2^30) / 32768 = 229,376 exactly. Taking the pool at
    the model's 262,144 instead says 233,016 -- which does NOT fit (it needs
    233,016 * 32768 + 2^30 = 8,709,210,112 bytes of 8,589,934,592) and is the
    default this module's own KV_TOO_SMALL_FOR_ONE_CTX, or the engine, would
    refuse."""
    from servedeck.capacity import single_request_fit

    pool = _hybrid_pool(8 * 2**30, 32768, 2**30)
    assert pool(262_144) == 233_016, "precondition: the naive answer"
    assert single_request_fit(262_144, pool) == 229_376


def test_the_single_request_fit_keeps_the_model_length_when_it_fits() -> None:
    """Over-correction guard: nothing is lowered when one request fits, not
    even by a step."""
    from servedeck.capacity import single_request_fit

    assert single_request_fit(262_144, lambda length: 1_595_321) == 262_144
    assert single_request_fit(262_144, lambda length: 262_144) == 262_144


def test_the_single_request_fit_of_a_constant_rate_is_the_pool() -> None:
    """A measured observation carries one rate, so the pool does not move with
    the length and the answer is simply the pool."""
    from servedeck.capacity import single_request_fit

    assert single_request_fit(262_144, lambda length: 238_671) == 238_671
    assert single_request_fit(262_144, lambda length: 0) == 0


def test_the_reason_for_a_lowered_default_names_the_numbers() -> None:
    from servedeck.capacity import ctx_fit_reason

    why = ctx_fit_reason(model_max_ctx=262_144, pool_at_max=238_671, fit=238_671,
                         util=0.94, kv_source="measured")
    assert why is not None
    for part in ("262,144", "0.94", "238,671", "measured", "Raise GPU utilization"):
        assert part in why, (part, why)
    # No reason when nothing was lowered, or when nothing is known.
    assert ctx_fit_reason(model_max_ctx=262_144, pool_at_max=1_595_321, fit=262_144,
                          util=0.95, kv_source="measured") is None
    assert ctx_fit_reason(model_max_ctx=262_144, pool_at_max=0, fit=0,
                          util=0.95, kv_source="unknown") is None
    assert "estimated" in ctx_fit_reason(model_max_ctx=262_144, pool_at_max=1000,
                                         fit=1000, util=0.5, kv_source="estimated")
    # A fit below the quoted pool is explained, not left as a contradiction
    # (Flash-Next at util 0.94 on this box: 238,657 at full length, 207,150
    # fits, because the rate at a shorter context is higher).
    assert "shorter context costs more KV per token" in ctx_fit_reason(
        model_max_ctx=262_144, pool_at_max=238_657, fit=207_150, util=0.94,
        kv_source="measured")
    assert "shorter context" not in why
