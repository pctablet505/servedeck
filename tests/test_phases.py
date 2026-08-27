"""Tests for servedeck.phases and servedeck.logtail.

Replays REAL vLLM boot logs captured on disk (never fabricated) to prove the
phase FSM and classify() match SPEC.md §5 exactly. Fixture paths and the
numbers asserted against them come straight from the task brief / SPEC.md
ground truth table, not invented here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from servedeck import logtail, phases

# ---------------------------------------------------------------------------
# Real fixture logs on disk
# ---------------------------------------------------------------------------

# Fixtures are SNAPSHOTS committed under tests/fixtures/, never the live logs.
# The running server rewrites serve.log on every restart, so tests that read it
# directly break the moment anyone restarts the model - which is exactly what
# happened when the default utilization changed from 0.96 to 0.95.
FIXTURES = Path(__file__).resolve().parent / "fixtures"
LOCAL_LLM_LOGS = FIXTURES
FLASHNEXT_DIR = FIXTURES

SERVE_LOG = FLASHNEXT_DIR / "serve.log"  # full success, 262144 ctx
SERVE_LOG_131K = FLASHNEXT_DIR / "serve.log.131k"  # full success, 131072 ctx
SERVE_LOG_KVSMALL = FLASHNEXT_DIR / "serve.log.kvsmall"  # FAILED / KV_TOO_SMALL
SERVE_LOG_FP8FAIL = FLASHNEXT_DIR / "serve.log.fp8fail"  # FAILED / QSA-BF16 rejection
SERVE_LOG_NINJAFAIL = FLASHNEXT_DIR / "serve.log.ninjafail"  # source of the shm_broadcast line
SPLIT_27B_LOG = LOCAL_LLM_LOGS / "qwen_server-20260827-183107.log"  # split two-line KV layout

_REQUIRED_FIXTURES = [
    SERVE_LOG,
    SERVE_LOG_131K,
    SERVE_LOG_KVSMALL,
    SERVE_LOG_FP8FAIL,
    SERVE_LOG_NINJAFAIL,
    SPLIT_27B_LOG,
]
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in _REQUIRED_FIXTURES),
    reason="real boot-log fixtures not present on this machine",
)


def read_lines(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f]


def run_tracker(lines: list[str]) -> tuple[phases.PhaseTracker, list[phases.Failure]]:
    """Feed every line through a fresh PhaseTracker + classify().

    Simulates a supervisor whose GET /v1/models probe would succeed the
    instant it's asked (real probe timing is out of phases.py's scope —
    this module only owns the log-text half of the READY criterion).
    """
    tracker = phases.PhaseTracker()
    tracker.set_http_probe_ok(True)
    failures: list[phases.Failure] = []
    for line in lines:
        tracker.feed(line)
        failure = phases.classify(line, tracker.reached_ready)
        if failure is not None:
            failures.append(failure)
    return tracker, failures


def distinct_phase_sequence(lines: list[str]) -> list[phases.Phase]:
    tracker = phases.PhaseTracker()
    tracker.set_http_probe_ok(True)
    seq: list[phases.Phase] = []
    for line in lines:
        for event in tracker.feed(line):
            if event.advanced and (not seq or seq[-1] != event.phase):
                seq.append(event.phase)
    return seq


# ---------------------------------------------------------------------------
# serve.log — Flash-Next, full success, all 5(7) phases in order
# ---------------------------------------------------------------------------


def test_flashnext_full_boot_phase_order():
    lines = read_lines(SERVE_LOG)
    seq = distinct_phase_sequence(lines)
    assert seq == [
        phases.Phase.INIT,
        phases.Phase.LOADING_WEIGHTS,
        phases.Phase.COMPILING,
        phases.Phase.KV_CACHE,
        phases.Phase.CUDA_GRAPHS,
        phases.Phase.HTTP_START,
        phases.Phase.READY,
    ]


def test_flashnext_full_boot_measurements():
    lines = read_lines(SERVE_LOG)
    tracker, failures = run_tracker(lines)

    assert failures == []
    assert tracker.weights.gib == 78.47
    assert tracker.kv.available_gib == 8.83
    assert tracker.kv.tokens == 304653
    assert tracker.kv.concurrency_x == 1.16
    assert tracker.reached_ready is True
    assert tracker.phase is phases.Phase.READY
    # compile exit fires twice (backbone + eagle_head for MTP) and must
    # accumulate rather than overwrite.
    assert len(tracker.compile_exits) == 2
    assert tracker.http.port == 8001


def test_flashnext_kv_tokens_and_concurrency_share_one_line():
    """Flash-Next's combined layout: both values come off the SAME raw line."""
    lines = read_lines(SERVE_LOG)
    tokens_line = concurrency_line = None
    tracker = phases.PhaseTracker()
    for line in lines:
        for event in tracker.feed(line):
            if event.kind == "kv_tokens":
                tokens_line = event.line
            if event.kind == "kv_concurrency":
                concurrency_line = event.line
    assert tokens_line is not None and concurrency_line is not None
    assert tokens_line == concurrency_line
    assert "kv_cache_utils.py:2258" in tokens_line


# ---------------------------------------------------------------------------
# serve.log.131k — proves KV rate is context-dependent (not weights)
# ---------------------------------------------------------------------------


def test_flashnext_131k_context_dependent_kv():
    lines = read_lines(SERVE_LOG_131K)
    tracker, failures = run_tracker(lines)

    assert failures == []
    assert tracker.weights.gib == 78.47  # same weights as the 262k boot
    assert tracker.kv.tokens == 273596  # different KV token count at 131k ctx
    assert tracker.kv.max_ctx_per_request == 131072
    assert tracker.reached_ready is True


# ---------------------------------------------------------------------------
# serve.log.kvsmall — FAILED / KV_TOO_SMALL
# ---------------------------------------------------------------------------


def test_kvsmall_classifies_kv_too_small_with_estimated_max_len():
    lines = read_lines(SERVE_LOG_KVSMALL)
    tracker, failures = run_tracker(lines)

    assert tracker.reached_ready is False  # a failed boot never reaches READY
    kv_too_small = [f for f in failures if f.code == "KV_TOO_SMALL"]
    assert len(kv_too_small) >= 1
    failure = kv_too_small[0]

    max_seq_len, kv_needed_gib, kv_available_gib, estimated_max_len = failure.groups
    assert max_seq_len == "262144"
    assert kv_needed_gib == "7.57"
    assert kv_available_gib == "3.89"
    assert estimated_max_len == "118400"

    # fix_action offers the ctx fix, captured straight from group(4)
    assert failure.fix_action == {"ctx": "118400"}
    assert failure.auto_restart is False

    # it got as far as KV_CACHE (the phase whose own arithmetic doomed it)
    # but never reached CUDA_GRAPHS/HTTP_START/READY.
    assert tracker.kv.available_gib == 3.89
    assert tracker.phase in (phases.Phase.KV_CACHE, phases.Phase.COMPILING, phases.Phase.LOADING_WEIGHTS)
    assert tracker.phase is not phases.Phase.READY


# ---------------------------------------------------------------------------
# serve.log.fp8fail — the QSA/BF16 rejection
# ---------------------------------------------------------------------------


def test_fp8fail_classifies_engine_init_failed():
    lines = read_lines(SERVE_LOG_FP8FAIL)

    tracker = phases.PhaseTracker()
    tracker.set_http_probe_ok(True)
    engine_init_failed_idx: int | None = None
    engine_init_failed: phases.Failure | None = None
    for idx, line in enumerate(lines):
        tracker.feed(line)
        f = phases.classify(line, tracker.reached_ready)
        if f is not None and f.code == "ENGINE_INIT_FAILED":
            engine_init_failed_idx = idx
            engine_init_failed = f

    assert tracker.reached_ready is False
    assert engine_init_failed is not None
    failure = engine_init_failed
    assert "Engine core initialization failed" in failure.line
    assert failure.auto_restart is False
    assert failure.hint == "attach preceding 40 lines"

    # This boot dies constructing the model (QSA requires BF16 KV cache)
    # before it ever gets to load weights — confirm the FSM agrees.
    assert tracker.phase is phases.Phase.INIT
    assert tracker.weights.gib is None

    # The actual root cause (the QSA/BF16 rejection) precedes the final
    # "Engine core initialization failed" summary line by more than 40
    # lines in this real capture (a lot of multiprocess teardown logging
    # sits in between) — so verify it's present earlier in the same run,
    # not inside the fixed 40-line attachment window the hint proposes for
    # the UI. classify() only needs to correctly identify *this* failure as
    # ENGINE_INIT_FAILED; SPEC's 40-line figure is a UI display hint, not a
    # guarantee this module makes about window size.
    assert engine_init_failed_idx is not None
    root_cause_text = "\n".join(lines[:engine_init_failed_idx])
    assert "Qwen4Exp QSA requires a BF16 main KV cache" in root_cause_text
    assert "NotImplementedError" in root_cause_text


# ---------------------------------------------------------------------------
# qwen_server-20260827-183107.log — 27B SPLIT two-line KV layout
# ---------------------------------------------------------------------------


def test_27b_split_layout_boot():
    lines = read_lines(SPLIT_27B_LOG)
    tracker, failures = run_tracker(lines)

    assert failures == []
    assert tracker.weights.gib == 20.75
    assert tracker.kv.tokens == 627117
    assert tracker.kv.concurrency_x == 2.39
    assert tracker.kv.max_ctx_per_request == 262144
    assert tracker.reached_ready is True
    assert tracker.phase is phases.Phase.READY


def test_27b_split_layout_tokens_and_concurrency_are_different_lines():
    """The 27B splits GPU KV cache size / Maximum concurrency across
    kv_cache_utils.py:2235 and :2236 — two physical lines, unlike Flash-Next's
    single combined line. Two independent re.search calls (one per regex,
    applied per-line) must each fire on their own line."""
    lines = read_lines(SPLIT_27B_LOG)
    tokens_line = concurrency_line = None
    tracker = phases.PhaseTracker()
    for line in lines:
        for event in tracker.feed(line):
            if event.kind == "kv_tokens":
                tokens_line = event.line
            if event.kind == "kv_concurrency":
                concurrency_line = event.line

    assert tokens_line is not None and concurrency_line is not None
    assert tokens_line != concurrency_line
    assert "kv_cache_utils.py:2235" in tokens_line
    assert "kv_cache_utils.py:2236" in concurrency_line


def test_loading_weights_advances_without_explicit_anchor_on_flashnext():
    """Flash-Next never prints "Starting to load model ..." (it prints
    "Loading model from scratch..." instead) — the shard-progress /
    measurement lines alone must still advance the FSM into LOADING_WEIGHTS."""
    lines = read_lines(SERVE_LOG)
    assert not any(phases.RE_LOADING_WEIGHTS.search(line) for line in lines)

    tracker = phases.PhaseTracker()
    for line in lines:
        tracker.feed(line)
    assert tracker.weights.gib == 78.47  # still got there


def test_phase_tracker_never_regresses_backward():
    """A stray re-match of an earlier phase's anchor (e.g. a worker log
    interleaving lines from a different process, or a duplicate INIT-style
    line late in the stream) must never move ``phase`` backwards."""
    tracker = phases.PhaseTracker()
    tracker.feed("(Worker pid=1) INFO [gpu_worker.py:694] Available KV cache memory: 8.83 GiB")
    assert tracker.phase is phases.Phase.KV_CACHE

    # Now feed a line that matches an EARLIER phase's anchor regex.
    tracker.feed("(EngineCore pid=1) INFO [core.py:122] Initializing a V1 LLM engine")
    assert tracker.phase is phases.Phase.KV_CACHE  # must not regress to INIT

    events = tracker.feed("(EngineCore pid=1) INFO [core.py:122] Initializing a V1 LLM engine")
    assert events[0].advanced is False  # re-matching an earlier anchor never "advances"


def test_27b_has_explicit_loading_weights_anchor():
    """The 27B DOES print the anchor — confirms both code paths are real."""
    lines = read_lines(SPLIT_27B_LOG)
    matches = [line for line in lines if phases.RE_LOADING_WEIGHTS.search(line)]
    assert len(matches) == 1
    assert "RadixArk/Qwen3.8-27B-NVFP4" in matches[0]


# ---------------------------------------------------------------------------
# CUDA_GRAPHS — tqdm \r-joins many updates into one physical line; must take
# the LAST "NN%|" on the line, not the first (which would freeze at 0%).
# ---------------------------------------------------------------------------


def test_cuda_graphs_takes_last_percent_on_tqdm_joined_line():
    """Real fixture line 359 of serve.log: one physical line carries SIX
    "NN%|" occurrences (0, 25, 50, 75, 100, 100) for PIECEWISE alone, because
    tqdm overwrites in place with \\r and the log captures every write. The
    correct read is the last one (100), not the first (0)."""
    line = (
        "(Worker pid=620054) Capturing CUDA graphs (PIECEWISE):   0%|          | "
        "0/4 [00:00<?, ?it/s]Capturing CUDA graphs (PIECEWISE):  25%|██▌       | "
        "1/4 [00:01<00:04,  1.45s/it]Capturing CUDA graphs (PIECEWISE):  50%|█████     | "
        "2/4 [00:01<00:01,  1.44it/s]Capturing CUDA graphs (PIECEWISE):  75%|███████▌  | "
        "3/4 [00:02<00:00,  1.61it/s]Capturing CUDA graphs (PIECEWISE): 100%|██████████| "
        "4/4 [00:03<00:00,  1.34it/s]Capturing CUDA graphs (PIECEWISE): 100%|██████████| "
        "4/4 [00:03<00:00,  1.30it/s]"
    )
    assert phases.RE_CUDA_GRAPHS.search(line).group(1) == "PIECEWISE"
    assert phases._parse_cuda_graph_pct(line) == 100  # NOT 0 (the first match)

    tracker = phases.PhaseTracker()
    events = tracker.feed(line)
    assert tracker.cuda_graphs.progress == {"PIECEWISE": 100}
    assert any(e.kind == "cuda_graphs_progress" for e in events)


def test_cuda_graphs_progress_on_real_flashnext_boot_both_labels_reach_100():
    """SPEC's MTP note applies here too: PIECEWISE then FULL are two
    separate real lines (359, 360) in serve.log — both must end at 100,
    keyed by their own label, without one clobbering the other."""
    lines = read_lines(SERVE_LOG)
    tracker = phases.PhaseTracker()
    for line in lines:
        tracker.feed(line)
    assert tracker.cuda_graphs.progress == {"PIECEWISE": 100, "FULL": 100}


# ---------------------------------------------------------------------------
# shm_broadcast — INFORMATIONAL, never an error (SETUP.md:407)
# ---------------------------------------------------------------------------


def test_shm_broadcast_is_informational_never_an_error():
    lines = read_lines(SERVE_LOG_NINJAFAIL)
    shm_lines = [line for line in lines if "shm_broadcast" in line]
    assert len(shm_lines) >= 1, "fixture must actually contain the shm_broadcast line"

    for line in shm_lines:
        assert "No available shared memory broadcast block found in 60 seconds" in line
        failure = phases.classify(line, reached_ready=True)
        assert failure is not None
        assert failure.code == "INFORMATIONAL"
        assert failure.auto_restart is False

        # It's an INFO-level vLLM line, not an ERROR/WARNING one — the
        # severity classifier must not paint it red or amber.
        assert phases.line_severity(line) not in ("e", "w")


# ---------------------------------------------------------------------------
# classify() — remaining blocker codes (§5), verified against the exact
# literal text SPEC.md quotes for each. These conditions have no fixture
# file among the ones assigned; the strings below are transcribed directly
# from SPEC.md, not invented.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line,expected_code",
    [
        (
            "ERROR 08-27 12:00:00 [x.py:1] AttributeError: no module or parameter "
            "named 'ngram_embedding.weight_scale'",
            "PLE_FP8_PATCH_MISSING",
        ),
        (
            "OSError: [Errno 1] pidfd_getfd: Operation not permitted",
            "PTRACE_DENIED",
        ),
        (
            "CMake Error: Could NOT find CUDA_CUDART_LIBRARY (missing: CUDA_CUDART_LIBRARY)",
            "CUDA_SYMLINKS",
        ),
        (
            "RuntimeError: CUDA error: the provided PTX was compiled with an unsupported toolchain",
            "CUDA_TOOLCHAIN",
        ),
        (
            "RuntimeError: Ninja build failed. Ninja output:",
            "FLASHINFER_LINK",
        ),
        (
            "/usr/bin/ld: cannot find -lcudart: No such file or directory",
            "FLASHINFER_LINK",
        ),
        (
            "RuntimeError: Free memory on device (1.20 / 97.89 GiB) is less than desired "
            "GPU memory utilization (0.96, 93.97 GiB)",
            "STARTUP_OOM",
        ),
        (
            "RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}",
            "ENGINE_INIT_FAILED",
        ),
    ],
)
def test_classify_remaining_blocker_codes(line, expected_code):
    failure = phases.classify(line, reached_ready=False)
    assert failure is not None
    assert failure.code == expected_code


def test_classify_flashinfer_link_hint():
    failure = phases.classify("RuntimeError: Ninja build failed. Ninja output:", reached_ready=False)
    assert failure is not None
    assert failure.hint == 'grep -nE "FAILED:|cannot find -l" serve.log'


@pytest.mark.parametrize(
    "line,code",
    [
        ("torch.AcceleratorError: CUDA error: misaligned address", "CUDA_FAULT"),
        ("torch.AcceleratorError: CUDA error: an illegal memory access was encountered", "CUDA_FAULT"),
        ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", "RUNTIME_OOM"),
    ],
)
def test_classify_reached_ready_gates_auto_restart(line, code):
    failure_before_ready = phases.classify(line, reached_ready=False)
    assert failure_before_ready is not None
    assert failure_before_ready.code == code
    assert failure_before_ready.auto_restart is False  # a failed boot never auto-restarts

    failure_after_ready = phases.classify(line, reached_ready=True)
    assert failure_after_ready is not None
    assert failure_after_ready.code == code
    assert failure_after_ready.auto_restart is True  # crash-while-serving may auto-restart


def test_classify_returns_none_for_ordinary_info_lines():
    assert phases.classify("INFO 08-27 21:22:16 [scheduler.py:246] Chunked prefill is enabled.", False) is None


# ---------------------------------------------------------------------------
# Log line severity — §5 final paragraph
# ---------------------------------------------------------------------------


def test_line_severity_error_precedence():
    assert phases.line_severity("(Worker pid=1) ERROR 08-27 12:00:00 [x.py:1] boom") == "e"
    assert phases.line_severity("Traceback (most recent call last):") == "e"
    assert phases.line_severity("ValueError: something broke") == "e"
    assert phases.line_severity("RuntimeError: also broke") == "e"


def test_line_severity_warning():
    line = "(APIServer pid=619748) WARNING 08-27 21:22:16 [speculative.py:1005] Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer,which may result in lower acceptance rate"
    assert line in read_lines(SERVE_LOG)
    assert phases.line_severity(line) == "w"


def test_line_severity_phase_match_is_good():
    line = "(Worker pid=620054) INFO 08-27 21:24:00 [gpu_worker.py:694] Available KV cache memory: 8.83 GiB"
    assert phases.line_severity(line, is_phase_match=False) == ""
    assert phases.line_severity(line, is_phase_match=True) == "g"


def test_line_severity_default_for_plain_info():
    line = "(Worker pid=620054) INFO 08-27 21:22:25 [parallel_state.py:1638] world_size=1 rank=0 local_rank=0"
    assert phases.line_severity(line) == ""


def test_line_severity_error_beats_phase_match():
    # A line could in principle match both a phase regex and an error
    # marker; error must win per the precedence order in SPEC §5.
    line = "(Worker pid=1) ERROR [gpu_worker.py:694] Available KV cache memory: 8.83 GiB"
    assert phases.line_severity(line, is_phase_match=True) == "e"


# ---------------------------------------------------------------------------
# logtail.LogTailer — rotation / truncation survival
# ---------------------------------------------------------------------------


def test_logtail_first_poll_reads_existing_content_and_reports_rotated(tmp_path):
    target = tmp_path / "run-1.log"
    target.write_text("line one\nline two\n")

    tailer = logtail.LogTailer(target)
    result = tailer.poll()

    assert result.exists is True
    assert result.rotated is True  # first-ever sighting counts as a rotation
    assert result.truncated is False
    assert result.lines == ["line one", "line two"]


def test_logtail_incremental_append_is_not_a_rotation(tmp_path):
    target = tmp_path / "run-1.log"
    target.write_text("line one\n")
    tailer = logtail.LogTailer(target)
    first = tailer.poll()
    assert first.lines == ["line one"]

    with open(target, "a") as f:
        f.write("line two\nline three\n")

    second = tailer.poll()
    assert second.rotated is False
    assert second.truncated is False
    assert second.lines == ["line two", "line three"]


def test_logtail_partial_line_is_buffered_until_newline(tmp_path):
    target = tmp_path / "run-1.log"
    target.write_text("hello")  # no trailing newline yet
    tailer = logtail.LogTailer(target)
    result = tailer.poll()
    assert result.lines == []  # nothing complete yet

    with open(target, "a") as f:
        f.write(" world\n")

    result2 = tailer.poll()
    assert result2.lines == ["hello world"]


def test_logtail_survives_symlink_repoint_like_qwen_server_run_sh(tmp_path):
    """Mirrors `ln -sfn "$RUN_LOG" "$LOG_DIR/qwen_server.log"`: a symlink
    atomically re-pointed at a brand-new, differently-named regular file on
    every boot. The tailer must detect the inode change and read ONLY the
    new target's content, not the old target's tail."""
    log_dir = tmp_path
    boot_a = log_dir / "qwen_server-20260827-100000.log"
    boot_b = log_dir / "qwen_server-20260827-110000.log"
    boot_a.write_text("boot A line 1\nboot A line 2\n")

    symlink = log_dir / "qwen_server.log"
    os.symlink(boot_a, symlink)

    tailer = logtail.LogTailer(symlink)
    first = tailer.poll()
    assert first.rotated is True
    assert first.lines == ["boot A line 1", "boot A line 2"]

    # boot A gets a bit more output before the next boot starts...
    with open(boot_a, "a") as f:
        f.write("boot A line 3\n")
    mid = tailer.poll()
    assert mid.rotated is False
    assert mid.lines == ["boot A line 3"]

    # ...then qwen-server-run.sh starts a NEW boot and re-points the symlink
    # atomically (ln -sfn semantics: create a temp link, rename over).
    boot_b.write_text("boot B line 1\n")
    tmp_link = log_dir / ".qwen_server.log.tmp"
    os.symlink(boot_b, tmp_link)
    os.replace(tmp_link, symlink)

    after_rotate = tailer.poll()
    assert after_rotate.rotated is True
    assert after_rotate.truncated is False
    assert after_rotate.lines == ["boot B line 1"]  # NOT any leftover boot-A tail

    with open(boot_b, "a") as f:
        f.write("boot B line 2\n")
    follow_up = tailer.poll()
    assert follow_up.rotated is False
    assert follow_up.lines == ["boot B line 2"]


def test_logtail_survives_in_place_truncation_same_inode(tmp_path):
    target = tmp_path / "run.log"
    target.write_text("aaaaaaaaaa\nbbbbbbbbbb\ncccccccccc\n")
    tailer = logtail.LogTailer(target)
    first = tailer.poll()
    assert first.lines == ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc"]
    ino_before = os.stat(target).st_ino

    # open(path, "w") truncates the EXISTING inode in place (O_TRUNC),
    # unlike the symlink-repoint case above which swaps to a new inode.
    with open(target, "w") as f:
        f.write("short\n")
    ino_after = os.stat(target).st_ino
    assert ino_after == ino_before  # confirms this really is in-place truncation

    result = tailer.poll()
    assert result.truncated is True
    assert result.rotated is False
    assert result.lines == ["short"]


def test_logtail_missing_file_then_appears(tmp_path):
    target = tmp_path / "not-yet.log"
    tailer = logtail.LogTailer(target)

    absent = tailer.poll()
    assert absent.exists is False
    assert absent.lines == []
    assert absent.rotated is False

    target.write_text("first line\n")
    appeared = tailer.poll()
    assert appeared.exists is True
    assert appeared.rotated is True
    assert appeared.lines == ["first line"]


def test_logtail_file_deleted_then_recreated_is_a_rotation(tmp_path):
    target = tmp_path / "run.log"
    target.write_text("gen 1\n")
    tailer = logtail.LogTailer(target)
    tailer.poll()

    target.unlink()
    gone = tailer.poll()
    assert gone.exists is False

    target.write_text("gen 2\n")
    back = tailer.poll()
    assert back.rotated is True
    assert back.lines == ["gen 2"]


# ---------------------------------------------------------------------------
# End-to-end: LogTailer feeding a live PhaseTracker across a simulated
# rotation, replaying real fixture bytes rather than synthetic text.
# ---------------------------------------------------------------------------


def test_logtail_and_phase_tracker_end_to_end_across_rotation(tmp_path):
    live_log = tmp_path / "qwen_server.log"
    kvsmall_bytes = SERVE_LOG_KVSMALL.read_bytes()
    good_bytes = SERVE_LOG.read_bytes()

    # Boot 1 fails (kvsmall). Write it incrementally to prove polling mid-write works.
    live_log.write_bytes(kvsmall_bytes[: len(kvsmall_bytes) // 2])
    tailer = logtail.LogTailer(live_log)
    tracker = phases.PhaseTracker()

    def drain() -> list[phases.Failure]:
        result = tailer.poll()
        if result.rotated:
            tracker.__init__()  # type: ignore[misc]  # fresh FSM on rotation
            tracker.set_http_probe_ok(True)
        fails = []
        for line in result.lines:
            tracker.feed(line)
            f = phases.classify(line, tracker.reached_ready)
            if f is not None:
                fails.append(f)
        return fails

    fails = drain()
    with open(live_log, "ab") as f:
        f.write(kvsmall_bytes[len(kvsmall_bytes) // 2 :])
    fails += drain()

    assert any(f.code == "KV_TOO_SMALL" for f in fails)
    assert tracker.reached_ready is False

    # Boot 2 (a fresh run.sh invocation) re-points the log to a brand new
    # file with a successful boot's bytes — same filename this time
    # (in-place truncation), which the tailer must also survive.
    with open(live_log, "wb") as f:
        f.write(good_bytes)
    result = tailer.poll()
    assert result.truncated is True
    tracker = phases.PhaseTracker()
    tracker.set_http_probe_ok(True)
    fails2 = []
    for line in result.lines:
        tracker.feed(line)
        f = phases.classify(line, tracker.reached_ready)
        if f is not None:
            fails2.append(f)

    assert fails2 == []
    assert tracker.reached_ready is True
    assert tracker.kv.tokens == 304653
