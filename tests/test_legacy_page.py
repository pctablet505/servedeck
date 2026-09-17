"""legacy_page: the v1 page's payload shapes, built from v2's runtime."""

from __future__ import annotations

from dataclasses import dataclass

from servedeck import control, legacy_page as lp, models as _models, parallelism, reqstats


# --------------------------------------------------------------------------
# argv of the live process
# --------------------------------------------------------------------------


ARGV = ["/v/bin/vllm", "serve", "org/M", "--max-model-len", "262144",
        "--gpu-memory-utilization", "0.96", "--max-num-seqs", "16", "--port", "8001"]


def test_flags_are_read_from_the_process_argv_not_from_config() -> None:
    assert lp.int_flag(ARGV, "--max-model-len") == 262144
    assert lp.int_flag(ARGV, "--max-num-seqs") == 16
    assert lp.float_flag(ARGV, "--gpu-memory-utilization") == 0.96
    assert lp.int_flag(ARGV, "--nope") is None
    assert lp.int_flag(["--max-model-len"], "--max-model-len") is None, "a trailing flag has no value"
    assert lp.int_flag(["--max-model-len", "x"], "--max-model-len") is None


def test_apply_argv_overrides_replaces_in_place_or_appends() -> None:
    out = control.apply_argv_overrides(list(ARGV), {"--max-num-seqs": "4", "--kv-cache-dtype": "auto"})
    assert out[out.index("--max-num-seqs") + 1] == "4"
    assert out[-2:] == ["--kv-cache-dtype", "auto"]
    assert out.count("--max-num-seqs") == 1
    assert control.apply_argv_overrides(list(ARGV), None) == ARGV


def test_overrides_from_the_allocator_body_are_clamped_and_stringified() -> None:
    util, argv = lp.overrides_from({"util": 1.5, "ctx": 131072, "max_num_seqs": 0})
    assert util == 0.99
    assert argv == {"--max-model-len": "131072", "--max-num-seqs": "1"}
    assert lp.overrides_from({}) == (None, {})


# --------------------------------------------------------------------------
# boot phases
# --------------------------------------------------------------------------


def test_markers_become_the_phases_the_page_draws() -> None:
    t = lp.BootTracker()
    t.record("flashnext", "line", None, 2.0)
    t.record("flashnext", "marker", 0, 20.0)   # Loading weights took
    t.record("flashnext", "marker", 1, 70.0)   # GPU KV cache size
    assert t.booting
    p = t.payload("STARTING")
    assert p["phases"] == list(lp.PHASES)
    assert p["phase"] == "kv_cache" and p["phase_times"] == {"init": 0.0, "loading_weights": 20.0, "kv_cache": 70.0}
    assert p["elapsed_s"] == 70.0 and p["reached_ready"] is False
    t.record("flashnext", "ready", None, 109.0)
    assert not t.booting and t.payload("READY")["reached_ready"] is True


def test_a_new_boot_after_a_finished_one_starts_a_fresh_track() -> None:
    t = lp.BootTracker()
    t.record("a", "ready", None, 50.0)
    t.record("b", "marker", 0, 10.0)
    assert t.current is not None and t.current.key == "b" and t.current.phase == "loading_weights"


def test_actual_state_words_match_the_pages_gates() -> None:
    idle = lp.BootTracker()
    assert lp.actual_state(busy=None, boot=idle, holder_ready=True) == "READY"
    assert lp.actual_state(busy=None, boot=idle, holder_ready=None) == "STOPPED"
    assert lp.actual_state(busy=None, boot=idle, holder_ready=False) == "STARTING"
    assert lp.actual_state(busy={"action": "stop"}, boot=idle, holder_ready=True) == "STOPPING"
    booting = lp.BootTracker()
    booting.record("x", "marker", 0, 5.0)
    assert lp.actual_state(busy=None, boot=booting, holder_ready=True) == "STARTING", (
        "a boot in flight outranks the holder being replaced"
    )
    failed = lp.BootTracker()
    failed.record("x", "failed", None, 30.0)
    assert lp.actual_state(busy=None, boot=failed, holder_ready=None) == "FAILED"


# --------------------------------------------------------------------------
# sizing (parallelism's arithmetic, never the page's)
# --------------------------------------------------------------------------


def test_sizing_degrades_to_a_reason_never_to_a_number() -> None:
    down = lp.sizing_payload({"reachable": False}, served_name="m", max_num_seqs=16, full_ctx=262144)
    assert down["recommended"] is None and down["reason"] == "backend not reachable"
    nopool = lp.sizing_payload({"reachable": True}, served_name="m", max_num_seqs=16, full_ctx=262144)
    assert "KV pool" in nopool["reason"]
    quiet = lp.sizing_payload(
        {"reachable": True, "kv_cache_size_tokens": 297926, "prompt_stats": dict(reqstats.EMPTY_STATS)},
        served_name="m", max_num_seqs=16, full_ctx=262144,
    )
    assert quiet["recommended"] is None and "no requests observed" in quiet["reason"]
    assert quiet["mixed"] is not None, "the long+short table needs only the pool and the context"


def test_sizing_recommends_from_the_p90_with_parallelisms_own_formula() -> None:
    window = {**reqstats.EMPTY_STATS, "n": 40, "p90": {"lo": 8000, "hi": 8000, "exact": True},
              "p99": {"lo": 30000, "hi": 30000, "exact": True}}
    out = lp.sizing_payload(
        {"reachable": True, "kv_cache_size_tokens": 297926, "prompt_stats": window, "running": 2},
        served_name="m", max_num_seqs=16, full_ctx=262144,
    )
    want = parallelism.recommend(pool_tokens=297926, prompt_tokens=8000.0, max_num_seqs=16,
                                 basis="p90 of the last 40 requests").to_dict()
    assert out["recommended"] == want
    assert out["at_p99"]["prompt_tokens"] == 30000
    assert out["over_subscribed"] is False


# --------------------------------------------------------------------------
# the rail
# --------------------------------------------------------------------------


@dataclass
class Entry:
    repo_id: str
    servable: bool = True
    reason: str | None = None
    disk_bytes: int = 10 * 1024**3
    disk_local_bytes: int = 10 * 1024**3
    quant_algo: str | None = "NVFP4"
    max_position_embeddings: int | None = 262144
    safetensors_gib: float = 9.5


def _registry(tmp_path) -> _models.Registry:
    p = tmp_path / "models.toml"
    p.write_text(
        '[gpu]\ntotal_mib = 100000\nmargin_mib = 1024\n'
        '[builds.stock]\nvenv = "/opt/v"\ncuda_home = "/opt/cuda"\n'
        '[models.a]\nid = "A"\nrepo = "org/A"\nslot = "main"\nport = 8001\nbuild = "stock"\nctx = 4096\n'
        'aliases = ["aa"]\nflags = ["--max-num-seqs", "16"]\n'
        '[models.b]\nid = "B"\nrepo = "org/B"\nslot = "main"\nport = 8002\nbuild = "stock"\nctx = 8192\n'
    )
    return _models.load(p)


def test_model_rows_carry_what_the_rail_paints(tmp_path) -> None:
    reg = _registry(tmp_path)
    rows = lp.model_rows(reg, [Entry("org/A")], ctx_for=lambda m: m.ctx, live_keys={"a": True})
    a, b = rows
    assert a["repo_id"] == "org/A" and a["name"] == "A" and a["servable"] and a["serving"]
    assert a["disk_bytes"] == 10 * 1024**3 and a["quant"] == "NVFP4" and a["model_max_ctx"] == 4096
    assert a["trust"] == "measured" and a["weights_gib"] == 9.5
    assert not b["servable"] and b["unservable_reason"] == "not in the local hub cache"
    assert b["disk_bytes"] == 0 and b["trust"] == "estimated" and not b["serving"]


def test_key_for_repo_accepts_repo_id_name_or_key(tmp_path) -> None:
    reg = _registry(tmp_path)
    assert lp.key_for_repo(reg, "org/B", None) == "b"
    assert lp.key_for_repo(reg, "A", None) == "a"
    assert lp.key_for_repo(reg, "a", None) == "a"
    assert lp.key_for_repo(reg, "nope", "b") == "b", "v1 sent backend beside repo_id"
    assert lp.key_for_repo(reg, None, None) is None


def test_cache_flags_come_from_the_registry_flags(tmp_path) -> None:
    reg = _registry(tmp_path)
    assert lp.cache_flags(reg.models["a"]) == (None, None)
    assert lp.cache_flags(None) == (None, None)


def test_live_facts_prefer_the_engine_and_fall_back_to_argv() -> None:
    snap = {"reachable": True, "kv_cache_size_tokens": 297926, "kv_cache_max_concurrency": 1.136,
            "kv_cache_gpu_util": 0.965}
    f = lp.live_facts(snap, ARGV)
    assert f == {"kv_tokens": 297926, "kv_source": "engine", "kv_trust": "measured",
                 "concurrency_x": 1.136, "util_effective": 0.965, "ctx": 262144}
    assert lp.live_facts({"reachable": False}, ARGV) == {"ctx": 262144, "util_effective": 0.96}


def test_upstream_when_nothing_runs_names_the_port_and_the_reason() -> None:
    up = lp.upstream_payload(model=None, key=None, ready=False, pid=0, adopted=False, ctx_tokens=0,
                             snapshot=None, argv=[], fallback_port=8001,
                             reason_when_down="no model holds the main slot")
    assert up["up"] is False and up["port"] == 8001 and up["url"] == "http://localhost:8001"
    assert up["resolution"]["reason"] == "no model holds the main slot"


def test_process_uptime_reads_proc_stat_for_an_adopted_pid() -> None:
    import os
    up = lp.process_uptime_s(os.getpid())
    assert up is not None and 0 <= up < 24 * 3600
    assert lp.process_uptime_s(0) is None and lp.process_uptime_s(2**22 + 12345) is None


def test_a_pinned_ptrace_scope_is_the_configured_state(tmp_path) -> None:
    from servedeck import capacity
    (tmp_path / "90-servedeck.conf").write_text("kernel.yama.ptrace_scope = 0  # flash-next PLE\n")
    assert lp.ptrace_scope_pinned(tmp_path) is True
    (tmp_path / "90-servedeck.conf").write_text("kernel.yama.ptrace_scope = 1\n")
    assert lp.ptrace_scope_pinned(tmp_path) is False
    assert lp.ptrace_scope_pinned(tmp_path / "missing") is False
    m = capacity.ModelInputs(repo_id="org/M", backend="flashnext", model_max_ctx=4096,
                             weights_gib=10.0, kv_kib_per_token=24.0)
    codes = lambda live: {f.code for f in capacity.compute(m, util=0.5, ctx=4096, max_num_seqs=1, live=live).findings}  # noqa: E731
    loose = capacity.LiveFacts(ptrace_scope=0, actual_state="READY")
    pinned = capacity.LiveFacts(ptrace_scope=0, actual_state="READY", ptrace_scope_pinned=True)
    relaxed = [c for c in codes(loose) if "PTRACE" in c and c != "PTRACE_BLOCKS_PLE"]
    assert relaxed, "the unpinned case still warns (guard against silencing everything)"
    assert not any(c in codes(pinned) for c in relaxed)


def test_the_kv_offload_is_read_from_argv_and_sized_like_the_pool() -> None:
    argv = ARGV + ["--kv-offloading-size", "40"]
    assert lp.live_facts({"reachable": False}, argv)["kv_offload_gib"] == 40.0
    assert "kv_offload_gib" not in lp.live_facts({"reachable": False}, ARGV)
    # 9.3 GiB holds 297,926 tokens on the GPU; 40 GiB of host RAM at that
    # rate parks ~1.28M tokens. Unknown inputs stay unknown, never zero.
    assert lp.offload_tokens_for(40, 297926, 9.3) == int(40 * 297926 / 9.3)
    assert lp.offload_tokens_for(None, 297926, 9.3) is None
    assert lp.offload_tokens_for(40, 0, 9.3) is None
