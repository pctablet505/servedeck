"""app.py runtime regressions.

All of these are "the UI is looking at the wrong thing and says so with
confidence": a poller frozen on a stale port, boot facts read out of another
backend's log, a context length taken from desired config rather than from the
process that is actually serving. None of them raises; each just renders a
wrong number.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from servedeck import app as capp
from servedeck import supervisor as _sup


# --------------------------------------------------------------------------
# Which log the live boot facts come from
# --------------------------------------------------------------------------
def _cfg_with_logless_backend(config_path, tmp_path: Path):
    """A config where one backend declares a log and one deliberately does not.

    The second case is the real one this fixes: a launcher with no log
    management of its own (no tee, no `exec` redirect) has no fixed path to
    tail, so nothing but the log Servedeck opened itself can be named.
    """
    named_log = tmp_path / "named.log"
    named_log.write_text("")
    return named_log, config_path(
        f"""
[backends.withlog]
launcher = "/bin/true"
port = 9101
log_path = "{named_log}"

[backends.nolog]
launcher = "/bin/true"
port = 9102
"""
    )


def test_boot_log_candidates_are_chosen_by_backend_not_port(config_path, tmp_path) -> None:
    """Regression: the candidate order was decided by which backend owns
    rt.port, and a backend that declares no log could not be named at all.

    Any such deployment therefore opened some OTHER backend's log first. Those
    files are append-only across runs (and one of them is a symlink that gets
    re-pointed), so the first one on the list can carry a real
    "GPU KV cache size: N tokens" line belonging to a different model. The only
    thing standing between that number and the dashboard is
    _live_boot_facts()' model_tag guard — a backstop, not a selection rule.
    """
    named_log, _ = _cfg_with_logless_backend(config_path, tmp_path)
    d = tmp_path / "boot_logs"
    d.mkdir()
    ours = d / "nolog-20260902-034300.log"
    ours.write_text("")

    assert capp._boot_log_candidates("withlog")[0] == named_log
    assert capp._boot_log_candidates("nolog", boot_log_dir=d)[0] == ours, (
        "the log belonging to the backend we believe is running must lead; "
        "the others are fallbacks for a hand-launched server, behind it"
    )


def test_boot_log_candidates_prefer_our_own_boot_log(config_path, tmp_path) -> None:
    """When Servedeck launched the server itself, the log it opened is the
    authoritative one and must come first."""
    _cfg_with_logless_backend(config_path, tmp_path)
    d = tmp_path / "boot_logs"
    d.mkdir()
    mine = d / "nolog-20260902-034300.log"
    mine.write_text("GPU KV cache size: 123,456 tokens\n")

    got = capp._boot_log_candidates("nolog", boot_log_dir=d)
    assert got[0] == mine, got


def test_default_log_paths_never_return_another_backends_log(config_path, tmp_path) -> None:
    """No log is strictly better than the wrong log: with none, the phase
    machine simply stays quiet instead of classifying a foreign error line as
    this run's failure_code."""
    named_log, _ = _cfg_with_logless_backend(config_path, tmp_path)
    assert _sup._default_log_paths("nolog", boot_log_dir=tmp_path / "absent") == []
    assert _sup._default_log_paths("withlog") == [named_log]


def test_app_spells_no_absolute_path_of_its_own() -> None:
    """paths.py's rule: "No other file in this package may spell out one of
    these paths itself — import the constant instead. That is what makes it
    possible to trust a `grep` for a stray literal." app.py had two, both
    pointing into one developer's home directory.
    """
    src = (Path(capp.__file__)).read_text()
    strays = [ln for ln in src.splitlines() if "/home/" in ln or "Projects/" in ln]
    assert not strays, f"app.py hard-codes a machine-specific path: {strays}"


# --------------------------------------------------------------------------
# Which port the poller watches
# --------------------------------------------------------------------------
def test_runtime_retargets_when_the_configured_port_changes(monkeypatch) -> None:
    """Regression: rt.port was read from the shell config exactly once, at
    construction.

    Starting a server on a different port rewrites that PORT (the supervisor's
    _sync_shell_config does it on every start), but nothing re-read it. The
    metrics poller, the "up Nm" uptime lookup, the running-model probe, the
    own-VRAM discount and the /v1 proxy all kept pointing at the old port for
    the life of the process — every one of them reporting "not reachable" for
    a server that was serving fine.
    """
    monkeypatch.setattr(capp, "_safe_config", lambda: {"PORT": "8001"})
    rt = capp.Runtime()
    assert rt.port == 8001 and rt.upstream.endswith(":8001")
    first_poller = rt.poller

    monkeypatch.setattr(capp, "_safe_config", lambda: {"PORT": "8002"})
    changed = rt.retarget_from_config()

    assert changed is True
    assert rt.port == 8002
    assert rt.upstream.endswith(":8002")
    assert rt.poller.base_url.endswith(":8002")
    assert rt.poller is not first_poller, (
        "the poller carries a throughput baseline for the OLD server; "
        "reusing it would compute a rate across two different processes"
    )


def test_runtime_retarget_is_a_no_op_when_the_port_is_unchanged(monkeypatch) -> None:
    """The poller's rate baseline must survive an ordinary poll tick — it is
    two samples long, and rebuilding it every 2 s would mean no rate ever."""
    monkeypatch.setattr(capp, "_safe_config", lambda: {"PORT": "8002"})
    rt = capp.Runtime()
    poller = rt.poller
    assert rt.retarget_from_config() is False
    assert rt.poller is poller


def test_runtime_retarget_ignores_a_junk_port(monkeypatch) -> None:
    """A hand-edited shell config must not be able to point the UI at
    nothing."""
    monkeypatch.setattr(capp, "_safe_config", lambda: {"PORT": "8002"})
    rt = capp.Runtime()
    for junk in ("", "not-a-port", "0", "99999"):
        monkeypatch.setattr(capp, "_safe_config", lambda junk=junk: {"PORT": junk})
        assert rt.retarget_from_config() is False, junk
        assert rt.port == 8002, junk


# --------------------------------------------------------------------------
# Which context length and model the serving line reports
# --------------------------------------------------------------------------
_ARGV = [
    "/opt/venv/bin/vllm", "serve",
    "someorg/Some-Model-NVFP4",
    "--served-model-name", "some-model",
    "--host", "0.0.0.0", "--port", "8002",
    "--max-model-len", "262144",
    "--gpu-memory-utilization", "0.96",
    "--max-num-seqs", "1",
]


def test_argv_flag_reads_the_running_servers_own_flags() -> None:
    """The serving line's "N ctx" was fixed once already, from the UI slider
    to supervisor.max_model_len. That is still not the running server's own
    number: `desired` is what Servedeck WANTS, and for a server it adopted
    rather than launched the two need not agree at all. The process's own
    command line is the only authoritative source — the same reasoning
    _running_model_id() already applies to the model name.
    """
    assert capp._argv_flag(_ARGV, "--max-model-len") == "262144"
    assert capp._argv_flag(_ARGV, "--served-model-name") == "some-model"
    assert capp._argv_flag(_ARGV, "--not-a-flag") is None
    # A flag in trailing position has no value; it must not read off the end.
    assert capp._argv_flag(["vllm", "serve", "m", "--max-model-len"], "--max-model-len") is None


def test_running_max_model_len_is_an_int_or_none() -> None:
    """A non-numeric value in the command line is not a context length."""
    assert capp._max_model_len_from(_ARGV) == 262144
    assert capp._max_model_len_from(["vllm", "serve", "m"]) is None
    assert capp._max_model_len_from(["vllm", "serve", "m", "--max-model-len", "auto"]) is None
    assert capp._max_model_len_from(["vllm", "serve", "m", "--max-model-len", "-5"]) is None


def test_running_model_id_prefers_the_flag_then_the_positional(monkeypatch) -> None:
    """--served-model-name is an alias an operator may reuse across different
    models; --model (or `vllm serve <model>`) is the repo actually loaded."""
    monkeypatch.setattr(capp, "_listener_argv", lambda: _ARGV)
    assert capp._running_model_id() == "someorg/Some-Model-NVFP4"
    monkeypatch.setattr(
        capp, "_listener_argv",
        lambda: ["python", "-m", "vllm", "--model", "other/Repo", "--port", "8002"],
    )
    assert capp._running_model_id() == "other/Repo"
    monkeypatch.setattr(capp, "_listener_argv", lambda: [])
    assert capp._running_model_id() is None


# --------------------------------------------------------------------------
# Starting a backend that only configuration knows about
# --------------------------------------------------------------------------
def test_start_accepts_any_configured_backend(config_path, tmp_path, monkeypatch) -> None:
    """Regression: the start gate was `backend not in ("flashnext", "inline")`.

    That literal refused every backend added through servedeck.toml — which is
    the single thing configuration exists to allow. A model whose architecture
    resolved to a configured backend was listed as servable, offered in the
    UI, and then refused at Start with "no model/backend/port configured".
    """
    import asyncio
    import types

    from servedeck import preflight, procctl

    config_path(
        """
[backends.custom]
launcher = "/bin/true"
port = 9200
[backends.custom.env]
SOME_LOCAL_KNOB = "7"
"""
    )
    launches: list = []

    def fake_launch(argv, env, cwd, log_path):
        launches.append((argv, env, cwd))
        return procctl.ServerHandle(
            pid=4242, pgid=4242, argv=list(argv), cwd=cwd,
            log_path=str(log_path), started_at=0.0,
        )

    s = _sup.Supervisor(
        state_dir=tmp_path,
        clock=lambda: 1000.0,
        launch_fn=fake_launch,
        stop_fn=lambda h, **kw: procctl.StopResult(True, "term", 0.0, "faked"),
        history_path=tmp_path / "history.jsonl",
    )
    # monkeypatch, not assignment: a bare rebind here leaked into every test
    # that ran after it in the same session, so preflight was silently
    # disabled for the rest of the suite -- including tests whose entire
    # subject is a preflight check refusing a start.
    monkeypatch.setattr(preflight, "run_preflight", lambda **kw: [])
    monkeypatch.setattr(preflight, "blocking_failures", lambda checks: [])
    s._sync_shell_config = types.MethodType(lambda self, **kw: None, s)           # type: ignore[assignment]
    s._run_monitor = types.MethodType(lambda self, *a, **k: asyncio.sleep(0), s)  # type: ignore[assignment]

    asyncio.run(
        s.start(
            repo_id="someorg/Some-Model", backend="custom", served_name="x",
            port=9200, util=0.9, max_model_len=4096, max_num_seqs=1,
        )
    )

    assert len(launches) == 1, f"a configured backend must be startable: {s.last_error}"
    argv, env, cwd = launches[0]
    assert argv == ["/bin/true"]
    assert env["MODEL"] == "someorg/Some-Model", "env_map must carry the chosen repo"
    assert env["MAX_LEN"] == "4096"
    assert env["SOME_LOCAL_KNOB"] == "7", (
        "the backend's own `env` table is machine-specific tuning the launcher "
        "reads; Servedeck must pass it through verbatim"
    )


def test_start_refuses_an_undeclared_backend_by_name(tmp_path) -> None:
    """And says which file to fix — "no model configured" was the old message
    for a model that was perfectly well configured."""
    import asyncio

    from servedeck import procctl

    s = _sup.Supervisor(
        state_dir=tmp_path,
        clock=lambda: 1000.0,
        launch_fn=lambda *a: procctl.ServerHandle(1, 1, [], "/tmp", "", 0.0),
        stop_fn=lambda h, **kw: procctl.StopResult(True, "term", 0.0, ""),
        history_path=tmp_path / "history.jsonl",
    )
    asyncio.run(
        s.start(
            repo_id="someorg/M", backend="not-declared", served_name="x",
            port=9999, util=0.9, max_model_len=4096, max_num_seqs=1,
        )
    )
    assert s.actual_state == "FAILED"
    assert "not-declared" in (s.last_error or "")
    assert "servedeck.toml" in (s.last_error or "")


def test_configured_backend_gets_its_preflight_checks(config_path) -> None:
    """VENV_MISSING and LAUNCHER_MISSING used to be skipped for anything not
    named flashnext or inline — so the backend most likely to be misconfigured
    (the one someone just added) got no check at all. C9 records the stale
    venv path as the single largest boot-failure class."""
    from servedeck import preflight

    config_path(
        """
[backends.custom]
launcher = "/definitely/not/here.sh"
port = 9300
venv = "/definitely/not/here/.venv"
"""
    )
    venv = preflight.check_venv("custom")
    launcher = preflight.check_launcher("custom")
    assert venv is not None and venv.ok is False
    assert launcher is not None and launcher.ok is False
    assert "/definitely/not/here.sh" in launcher.detail


@pytest.mark.parametrize("declared_tty,expected_block", [(True, True), (False, False)])
def test_unattended_restart_gate_follows_needs_tty(config_path, declared_tty, expected_block) -> None:
    """Whether a launcher needs a terminal is a property of that launcher — a
    `sudo sysctl` that silently no-ops without a tty (SPEC.md correction C3) —
    and the answer differs from machine to machine. It was a hardcoded backend
    name, so the gate both over-blocked and under-blocked depending on how the
    box was set up."""
    config_path(
        f"""
[backends.custom]
launcher = "/bin/true"
port = 9400
needs_tty = {str(declared_tty).lower()}
"""
    )
    decision = _sup.decide_after_exit(
        desired_state="RUNNING",
        auto_restart=True,
        reached_ready=True,
        backend="custom",
        gpu_ok=True,
        xid_blocks_restart=False,
        xid_note=None,
        attempts=[],
        now=1000.0,
    )
    assert (decision.code == _sup.FLASHNEXT_HUMAN_GATE_CODE) is expected_block, decision


# --------------------------------------------------------------------------
# Where the MEASURED KV size comes from
# --------------------------------------------------------------------------
def test_live_kv_size_prefers_the_running_engine_over_a_boot_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A boot log outlives the server that wrote it, and a hand-launched
    server writes to no log Servedeck knows about at all. The engine publishes
    its own resolved capacity on vllm:cache_config_info, so read it there and
    let the log fill in only what /metrics does not carry."""
    from servedeck import metrics

    stale = tmp_path / "old.log"
    stale.write_text(
        "GPU KV cache size: 111,111 tokens, "
        "Maximum concurrency for 262,144 tokens per request: 0.42x\n"
        "Available KV cache memory: 3.21 GiB\n"
        "Model loading took 78.47 GiB memory and 90.6 seconds\n"
    )
    monkeypatch.setattr(capp, "_boot_log_candidates", lambda *a, **k: (stale,))
    monkeypatch.setattr(capp, "_running_model_id", lambda: None)

    live = (Path(__file__).parent / "fixtures" / "metrics_flashnext_live.txt").read_text()
    snap = metrics.MetricsSnapshot()
    parsed = metrics.parse_prometheus(live)
    labels = parsed[metrics.CACHE_CONFIG][0][0]
    snap.reachable = True
    snap.kv_cache_size_tokens = int(labels["kv_cache_size_tokens"])
    snap.kv_cache_max_concurrency = float(labels["kv_cache_max_concurrency"])
    snap.kv_cache_gpu_util = float(labels["gpu_memory_utilization"])
    monkeypatch.setattr(capp.rt, "metrics", snap.to_dict())

    facts = capp._live_boot_facts()
    assert facts["kv_tokens"] == 290_925, "the stale log's 111,111 won"
    assert facts["kv_source"] == "engine"
    assert facts["kv_trust"] == "measured"
    assert facts["util_effective"] == pytest.approx(0.95)
    # And the log still supplies what /metrics does not carry.
    assert facts["kv_gib"] == pytest.approx(3.21)
    assert facts["weights_gib"] == pytest.approx(78.47)


def test_live_kv_size_falls_back_to_the_boot_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An older vLLM that does not publish cache_config_info must still get
    its measured KV size out of the log."""
    log = tmp_path / "boot.log"
    log.write_text("GPU KV cache size: 272,062 tokens\n")
    monkeypatch.setattr(capp, "_boot_log_candidates", lambda *a, **k: (log,))
    monkeypatch.setattr(capp, "_running_model_id", lambda: None)
    monkeypatch.setattr(capp.rt, "metrics", {"reachable": True})

    facts = capp._live_boot_facts()
    assert facts["kv_tokens"] == 272_062
    assert facts["kv_source"] == "boot log"


# --------------------------------------------------------------------------
# The model scan must expire
# --------------------------------------------------------------------------
def _fake_entry(repo_id: str, disk_bytes: int):
    from servedeck.registry import ModelEntry

    return ModelEntry(
        repo_id=repo_id,
        hub_dirname="models--" + repo_id.replace("/", "--"),
        snapshot_path="/nowhere",
        skipped=False,
        servable=True,
        backend="flashnext",
        safetensors_gib=disk_bytes / (1024**3),
        safetensors_count=1,
        disk_bytes=disk_bytes,
        disk_local_bytes=disk_bytes,
        config_exists=True,
        architectures0="Qwen4ExpForConditionalGeneration",
        model_type="qwen4_exp",
        max_position_embeddings=262144,
        num_hidden_layers=48,
        num_key_value_heads=8,
        head_dim=128,
        full_attention_interval=None,
        quant_algo="NVFP4",
        reason=None,
    )


def test_the_model_scan_expires_so_a_deletion_becomes_visible(monkeypatch) -> None:
    """The cache had no expiry at all: scanned on the first /api/models and
    then served for the life of the process.

    On 2026-09-09 a 95.37 GiB BF16 PLE table was deleted and a 47.68 GiB FP8
    one hardlinked in while the dashboard was up. The panel went on reporting
    the pre-deletion size for hours, which is what "the size displayed is
    wrong" meant. A whole-hub scan is ~25 ms warm; there was nothing worth
    pinning forever.
    """
    from servedeck import app as capp
    from servedeck import registry as _reg

    sizes = [95 * 1024**3]
    monkeypatch.setattr(_reg, "discover_models", lambda *a, **k: [_fake_entry("A/B", sizes[0])])
    monkeypatch.setattr(
        _reg, "resolve_inputs", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no store"))
    )
    capp.rt._models_cache = None
    capp.rt._models_cache_at = 0.0

    first = capp._model_rows(now=1000.0)
    assert first[0]["disk_bytes"] == 95 * 1024**3

    # The table is deleted and a smaller one installed.
    sizes[0] = 47 * 1024**3

    # Within the TTL the cached answer stands -- that is the cache doing its
    # job, not the bug.
    assert capp._model_rows(now=1000.0 + capp.MODELS_CACHE_TTL_S / 2)[0]["disk_bytes"] == (
        95 * 1024**3
    )
    # Past it, the scan is retaken.
    fresh = capp._model_rows(now=1000.0 + capp.MODELS_CACHE_TTL_S + 0.1)
    assert fresh[0]["disk_bytes"] == 47 * 1024**3, (
        "the model scan never expired: a deleted 95 GiB table stayed on screen"
    )
    capp.rt._models_cache = None
    capp.rt._models_cache_at = 0.0


def test_the_disk_payload_is_statvfs_and_labels_its_unit(monkeypatch) -> None:
    """/api/disk must be the kernel's arithmetic, not an approximation of it.

    Pinned against a FIXED statvfs result rather than a live one: this box is
    writing continuously, so a real statvfs taken a millisecond after the
    payload disagrees by a few MiB and the test would be measuring the clock.
    The live agreement is checked separately, with a tolerance.
    """
    import os

    from servedeck import app as capp
    from servedeck import disksize as _dsz

    # 1 TiB filesystem, 4 KiB fragments, 200 GiB free of which 150 GiB is
    # available to a non-root user (the rest is ext4's root reserve).
    frag = 4096
    fake = os.statvfs_result(
        (
            frag,  # f_bsize
            frag,  # f_frsize
            (1024**4) // frag,  # f_blocks
            (200 * 1024**3) // frag,  # f_bfree
            (150 * 1024**3) // frag,  # f_bavail
            0, 0, 0, 0, 255,
        )
    )
    monkeypatch.setattr(_dsz.os, "statvfs", lambda _p: fake)
    capp.rt._models_cache = None
    capp.rt._models_cache_at = 0.0

    d = capp._disk_payload()

    assert d["total_bytes"] == 1024**4
    assert d["avail_bytes"] == 150 * 1024**3
    assert d["used_bytes"] == 1024**4 - 200 * 1024**3
    # df's Use%: used / (used + avail), which excludes the root reserve.
    assert d["used_pct"] == round(100 * (1024 - 200) / (1024 - 200 + 150), 1)
    assert d["unit"] == "bytes", "the payload must not leave its unit to be guessed"
    # Bytes on another mount are real but are not space on this filesystem.
    assert d["hub_bytes"] - d["hub_local_bytes"] == d["hub_foreign_bytes"]
    assert d["hub_local_bytes"] <= d["hub_bytes"]
    capp.rt._models_cache = None
    capp.rt._models_cache_at = 0.0


def test_the_disk_payload_agrees_with_the_live_kernel() -> None:
    """The over-correction guard for the fixture above: a payload that only
    ever matches a fake statvfs would pass while reading the wrong path or
    the wrong syscall. A live filesystem moves under us, so this allows drift
    -- but not the 5% a units or formula slip would produce."""
    import os

    from servedeck import app as capp

    capp.rt._models_cache = None
    capp.rt._models_cache_at = 0.0
    d = capp._disk_payload()
    s = os.statvfs(d["path"])
    frsize = s.f_frsize or s.f_bsize

    assert d["total_bytes"] == s.f_blocks * frsize, "total does not move; it must match exactly"
    tol = 1024**3  # 1 GiB of live churn
    assert abs(d["avail_bytes"] - s.f_bavail * frsize) < tol
    assert abs(d["used_bytes"] - (s.f_blocks - s.f_bfree) * frsize) < tol
    capp.rt._models_cache = None
    capp.rt._models_cache_at = 0.0
