"""Reading a boot's own memory accounting back off its journal.

Every fixture line here is verbatim from `journalctl --user -u model-flashnext`
on 2026-09-18, the evening the capacity panel green-lit a launch the engine
refused. They are the contract: if vLLM reformats them, these tests fail, and
`parse` returning None-everywhere is the designed response — never a zero that
would read as a measurement.
"""

from __future__ import annotations

import pytest

from servedeck import bootfacts

# The util 0.97 boot: 22:22:45, max_num_seqs 1, --kv-offloading-size 20.
BOOT_097 = [
    "(EngineCore pid=42623) INFO 09-18 22:23:41 [gpu_worker.py:298] Free memory on device "
    "(94.41/94.97 GiB) on startup. Desired GPU memory utilization is (0.97, 92.12 GiB). "
    "Actual usage is 81.7 GiB for consumed memory (weights + non-torch), 1.78 GiB for peak "
    "activation, and 0.28 GiB for CUDAGraph memory. Replace gpu_memory_utilization config "
    "with `--kv-cache-memory=8820325233` (8.21 GiB) to fit into requested memory, or "
    "`--kv-cache-memory=11273977344` (10.5 GiB) to fully utilize gpu memory. Current kv "
    "cache memory in use is 8.64 GiB.",
    "(EngineCore pid=42623) INFO 09-18 22:23:42 [kv_cache_utils.py:1229] GPU KV cache size: "
    "315,039 tokens, Maximum concurrency for 262,144 tokens per request: 1.20x",
]

# The util 0.98 boot: 22:30:30, registry defaults (16 seqs, 40 GiB offload).
BOOT_098 = [
    "(EngineCore pid=48111) INFO 09-18 22:31:20 [gpu_worker.py:298] Free memory on device "
    "(94.41/94.97 GiB) on startup. Desired GPU memory utilization is (0.98, 93.07 GiB). "
    "Actual usage is 81.7 GiB for consumed memory (weights + non-torch), 1.78 GiB for peak "
    "activation, and 0.66 GiB for CUDAGraph memory. Current kv cache memory in use is "
    "9.59 GiB.",
    "(EngineCore pid=48111) INFO 09-18 22:31:21 [kv_cache_utils.py:1229] GPU KV cache size: "
    "350,043 tokens, Maximum concurrency for 262,144 tokens per request: 1.34x",
]

#: capacity.GPU_TOTAL_GIB on this card: 97,887 MiB / 1024.
BASIS = 97887 / 1024

#: The weights figure every Flash-Next observation carries.
WEIGHTS = 78.47


def test_parses_the_memory_profile_and_the_kv_size() -> None:
    f = bootfacts.parse(BOOT_097)
    assert f.device_total_gib == 94.97
    assert f.device_free_gib == 94.41
    assert f.util == 0.97
    assert f.budget_gib == 92.12
    assert f.consumed_gib == 81.7
    assert f.activation_gib == 1.78
    assert f.cudagraph_gib == 0.28
    assert f.kv_gib == 8.64
    assert f.kv_tokens == 315_039  # the comma is vLLM's, not a typo
    assert f.max_concurrency == 1.20
    assert f.complete is True


def test_the_last_boot_in_the_journal_wins() -> None:
    """A unit restarted in place leaves several boots in one journal. The one
    that is running is the last one, and reading an earlier one would
    calibrate the panel against a configuration nothing is serving."""
    f = bootfacts.parse([*BOOT_097, *BOOT_098])
    assert (f.util, f.kv_gib, f.kv_tokens) == (0.98, 9.59, 350_043)


def test_a_release_that_reformats_the_lines_measures_nothing() -> None:
    """Not zero, not a default: None, so callers record nothing rather than
    poisoning the store with a fabricated measurement."""
    f = bootfacts.parse(["INFO: some future vLLM says something else entirely"])
    assert f == bootfacts.BootFacts()
    assert f.complete is False
    assert bootfacts.overhead_gib_from(f, WEIGHTS, BASIS) is None


def test_a_partial_parse_keeps_the_fields_it_did_get() -> None:
    """The patterns are independent on purpose: losing the tail of the usage
    sentence must not cost the KV size, which is the figure being predicted."""
    f = bootfacts.parse(BOOT_097[1:])
    assert f.kv_tokens == 315_039
    assert f.device_total_gib is None


@pytest.mark.parametrize(
    "lines, expected",
    [(BOOT_097, 5.615), (BOOT_098, 5.621)],
)
def test_overhead_is_one_constant_across_two_utilisations(lines, expected) -> None:
    """The residual is fitted, so it MUST come out the same at 0.97 and 0.98.
    0.006 GiB apart means it is a property of the model; a spread would mean
    the term is absorbing something that varies with utilisation, and fitting
    it from one boot would then be meaningless."""
    f = bootfacts.parse(lines)
    assert bootfacts.overhead_gib_from(f, WEIGHTS, BASIS) == pytest.approx(expected, abs=0.001)


def test_the_refitted_overhead_reproduces_the_engine_and_the_old_one_does_not() -> None:
    """The whole point, stated as arithmetic.

    Stored overhead was 4.47 GiB (fitted 2026-08-28). At util 0.95 that
    predicts a 7.87 GiB pool and `can_apply: true`; the engine measured
    6.74 GiB and refused, because one 262,144-token request needs 7.17 GiB.
    The re-fitted 5.62 predicts 6.72 -- under the requirement, so the panel
    refuses before the operator waits 2.5 minutes to be told.
    """
    stale, refitted = 4.47, 5.62
    needed_gib = 7.17  # vLLM: "(7.17 GiB KV cache is needed"
    measured_gib = 6.74  # vLLM: "available KV cache memory (6.74 GiB)"

    predicted_stale = 0.95 * BASIS - WEIGHTS - stale
    predicted_refit = 0.95 * BASIS - WEIGHTS - refitted

    assert predicted_stale > needed_gib, "the stale fit is why the panel said yes"
    assert predicted_refit < needed_gib, "the refitted one says no, as the engine did"
    assert predicted_refit == pytest.approx(measured_gib, abs=0.03)


def test_observation_is_shaped_like_the_entries_already_in_the_store() -> None:
    obs = bootfacts.observation(
        repo_id="mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4",
        backend="qwen38next",
        facts=bootfacts.parse(BOOT_097),
        weights_gib=WEIGHTS,
        basis_gib=BASIS,
        ctx=262144,
        max_num_seqs=1,
        kv_cache_dtype="auto",
        ts="2026-09-18T22:23:42+0530",
    )
    assert obs is not None
    assert obs["inputs"] == {
        "util": 0.97,  # what vLLM RESOLVED, not what was requested
        "max_model_len": 262144,
        "max_num_seqs": 1,
        "kv_cache_dtype": "auto",
    }
    assert obs["measured"]["kv_tokens"] == 315_039
    assert obs["measured"]["overhead_gib"] == pytest.approx(5.615, abs=0.001)
    assert obs["measured"]["weights_gib"] == WEIGHTS
    assert obs["measured"]["device_total_gib"] == 94.97


def test_a_boot_that_measured_nothing_is_not_recorded_at_all() -> None:
    """An empty record would still be the most recent entry for its repo, and
    the resolver takes the most recent -- so it would shadow a good one."""
    assert (
        bootfacts.observation(
            repo_id="org/x",
            backend="stock",
            facts=bootfacts.BootFacts(),
            weights_gib=WEIGHTS,
            basis_gib=BASIS,
            ctx=1024,
            max_num_seqs=1,
            kv_cache_dtype=None,
            ts="2026-09-18T00:00:00+0530",
        )
        is None
    )


def test_unknown_weights_still_record_the_kv_measurement() -> None:
    """A model booting for the first time has no weights figure yet. Its KV
    rate is still worth having; only the overhead residual is withheld."""
    obs = bootfacts.observation(
        repo_id="org/new",
        backend="stock",
        facts=bootfacts.parse(BOOT_097),
        weights_gib=None,
        basis_gib=BASIS,
        ctx=262144,
        max_num_seqs=1,
        kv_cache_dtype=None,
        ts="2026-09-18T22:23:42+0530",
    )
    assert obs is not None
    assert obs["measured"]["kv_tokens"] == 315_039
    assert obs["measured"]["overhead_gib"] is None
