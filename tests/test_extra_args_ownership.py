"""local_llm/.config's EXTRA_ARGS must never hand one backend's flags to another.

EXTRA_ARGS holds one backend's flags in one global key. Two real failures:

* 2026-09-10 22:54: a start of the 27B (backend ``inline``) from a .config
  describing Flash-Next launched with ``--prefix-match-unit 208``, and vLLM
  died 205 s in with "Invalid prefix_match_unit=208; ... block sizes=[848,
  848, 848, 848]". The same thing happened at the 2026-09-11 06:59 reboot,
  when startup reconciliation resumed the 27B.
* The first fix read EXTRA_ARGS before the sync rewrote BACKEND. That kept
  the flags out of the ENVIRONMENT for the first start only.
  bin/qwen-server-run.sh sources .config after its defaults, so the file's
  EXTRA_ARGS beats the environment and the 27B still read Flash-Next's flags.
  After that first start the file said ``BACKEND="inline"`` beside
  Flash-Next's flags. The guard read that pair as consistent, so every later
  start, every restart and every ``codex-qwen.sh start`` handed the flags on
  too.

These tests therefore run the REAL ``_sync_shell_config()`` against a real
file, through a stand-in codex-qwen.sh that has the same allow list and the
same ``save_config_kv()``. At each launch they check what the backend's
launcher would really read: the FILE for the 27B, the environment for
Flash-Next.
"""

from __future__ import annotations

import asyncio
import os
import re
import stat
import subprocess
import textwrap
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from servedeck import paths, preflight, procctl, shellconfig, supervisor

FLASHNEXT_EXTRA_ARGS = (
    "--mamba-ssm-cache-dtype bfloat16 --prefix-match-unit 208 "
    "--long-prefill-token-threshold 1024 --watermark 0.10"
)
FLASHNEXT_FLAGS = (
    "--mamba-ssm-cache-dtype", "--prefix-match-unit",
    "--long-prefill-token-threshold", "--watermark",
)

# The shape of this box's .config on 2026-09-11 (restored by hand at 07:05):
# a header, EXTRA_ARGS near the top, commented switch-back blocks, and the
# live Flash-Next block at the end.
BOX_CONFIG = textwrap.dedent(f"""\
    # Values MUST be double-quoted: llm and shellconfig._KV_LINE_RE only match
    # KEY="value", and an unquoted line is silently skipped.
    COLDSTART_URL=""
    EXTRA_ARGS="{FLASHNEXT_EXTRA_ARGS}"

    # --- Qwen3.8-27B ("inline") settings, kept for reference / fast switch-back -
    # BACKEND="inline"
    # MODEL_REPO="RadixArk/Qwen3.8-27B-NVFP4"
    # EXTRA_ARGS=""

    # --- Flash-Next (LIVE, matches the server running on :8001) ------------------
    BACKEND="flashnext"
    MODEL_REPO="mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4"
    SERVED_NAME="qwen38-flash-next"
    PORT="8001"
    MAX_MODEL_LEN="262144"
    MAX_NUM_SEQS="16"
    GPU_MEM_UTIL="0.96"
""")

INLINE = dict(
    repo_id="RadixArk/Qwen3.8-27B-NVFP4", backend="inline", served_name="Qwen3.8-27B-NVFP4",
    port=8004, util=0.47, max_model_len=262144, max_num_seqs=16,
)
FLASHNEXT = dict(
    repo_id="mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4", backend="flashnext",
    served_name="qwen38-flash-next", port=8001, util=0.96, max_model_len=262144, max_num_seqs=16,
)

# codex-qwen.sh's save_config_kv(), verbatim (codex-qwen.sh:251-258 on
# 2026-09-11). test_the_stand_in_matches_the_real_codex_qwen_sh pins it.
SAVE_CONFIG_KV = textwrap.dedent("""\
    save_config_kv() {
        local key="$1" value="$2"
        local tmp
        tmp=$(mktemp)
        [ -f "$CONFIG_FILE" ] && grep -v "^${key}=" "$CONFIG_FILE" > "$tmp"
        printf '%s="%s"\\n' "$key" "$value" >> "$tmp"
        mv "$tmp" "$CONFIG_FILE"
    }
""")


def _fake_codex_qwen(config_file: Path) -> str:
    """codex-qwen.sh with its real allow list and writer, and nothing that
    can reach a server (the real set-mem restarts a live one)."""
    return (
        "#!/usr/bin/env bash\n"
        f"CONFIG_FILE='{config_file}'\n"
        'CONFIG_ALLOWED_KEYS="BACKEND MODEL MODEL_REPO SERVED_NAME PORT MAX_MODEL_LEN '
        'MAX_NUM_SEQS COLDSTART_URL USE_COLDSTART"\n'
        + SAVE_CONFIG_KV
        + textwrap.dedent("""\
            case "$1" in
              set-config)
                for k in $CONFIG_ALLOWED_KEYS; do
                  if [ "$k" = "$2" ]; then save_config_kv "$2" "$3"; exit 0; fi
                done
                echo "Unknown config key: $2" >&2; exit 1 ;;
              set-mem) save_config_kv GPU_MEM_UTIL "$2" ;;
              save-kv) save_config_kv "$2" "$3" ;;   # test-only: parity check
              *) echo "fake codex-qwen.sh: unsupported $1" >&2; exit 1 ;;
            esac
        """)
    )


@dataclass
class Launch:
    backend: str
    env: dict
    config: dict          # .config as parsed at the instant of the launch
    config_text: str
    sees: str             # the EXTRA_ARGS the backend's own launcher would read


def _launcher_would_read(backend: str, env: dict, cfg: dict) -> str:
    if backend == "inline":
        # bin/qwen-server-run.sh: `[ -f "$CONFIG_FILE" ] && . "$CONFIG_FILE"`
        # runs AFTER its defaults, so a value in the file beats the environment.
        return cfg["EXTRA_ARGS"] if "EXTRA_ARGS" in cfg else env.get("EXTRA_ARGS", "")
    # serve.sh / serve-opt.sh interpolate ${EXTRA_ARGS:-} from the environment.
    return env.get("EXTRA_ARGS", "")


@pytest.fixture
def box(tmp_path: Path, monkeypatch):
    """A tmp local_llm/ holding BOX_CONFIG and the stand-in codex-qwen.sh."""
    local = tmp_path / "local_llm"
    local.mkdir()
    cfg = local / ".config"
    cfg.write_text(BOX_CONFIG)
    cfg.chmod(0o600)
    script = local / "codex-qwen.sh"
    script.write_text(_fake_codex_qwen(cfg))
    script.chmod(0o755)
    monkeypatch.setattr(paths, "LOCAL_LLM", local)
    monkeypatch.setattr(paths, "CONFIG_FILE", cfg)
    monkeypatch.setattr(paths, "CODEX_QWEN_SH", script)
    # start()/stop() also write run/desired_state; keep that out of the real tree.
    monkeypatch.setattr(paths, "RUN_DIR", local / "run")
    return cfg


def _supervisor(tmp_path: Path, monkeypatch) -> tuple[supervisor.Supervisor, list[Launch]]:
    """A Supervisor whose config sync is REAL; launch/stop/preflight/monitor faked."""
    launches: list[Launch] = []
    backend_of = {}

    def launch(argv, env, cwd, log_path):  # noqa: ANN001
        cfg = shellconfig.read_config()
        b = backend_of["now"]
        launches.append(Launch(b, dict(env), cfg, paths.CONFIG_FILE.read_text(),
                               _launcher_would_read(b, env, cfg)))
        n = len(launches)
        return procctl.ServerHandle(pid=7000 + n, pgid=7000 + n, argv=list(argv),
                                    cwd="/tmp", log_path="/tmp/fake.log", started_at=0.0)

    state = tmp_path / "state"
    s = supervisor.Supervisor(
        state_dir=state, clock=lambda: 1000.0, launch_fn=launch,
        stop_fn=lambda h, **kw: procctl.StopResult(True, "term", 0.0, "faked"),
        history_path=state / "history.jsonl",
    )
    monkeypatch.setattr(preflight, "run_preflight", lambda **kw: [])
    monkeypatch.setattr(preflight, "blocking_failures", lambda checks: [])
    s._run_monitor = types.MethodType(lambda self, *a, **k: asyncio.sleep(0), s)  # type: ignore[assignment]

    real_start = s.start

    async def start(**kw):  # noqa: ANN003 - remember which backend is launching
        backend_of["now"] = kw["backend"]
        await real_start(**kw)

    s.start = start  # type: ignore[assignment]
    return s, launches


def _leaked(text: str) -> list[str]:
    return [f for f in FLASHNEXT_FLAGS if f in (text or "")]


def _exited(s: supervisor.Supervisor) -> None:
    """The previous run's process is gone and its exit has been handled."""
    s.actual_state = "STOPPED"
    s._handle = None


# --------------------------------------------------------------------------
# The hole: the second start, and any restart, after a switch
# --------------------------------------------------------------------------
def test_the_27b_never_sees_flashnext_flags_on_the_first_or_the_second_start(
    box: Path, tmp_path: Path, monkeypatch
) -> None:
    s, launches = _supervisor(tmp_path, monkeypatch)

    async def scenario() -> None:
        await s.start(**INLINE)
        _exited(s)
        await s.start(**INLINE)

    asyncio.run(scenario())

    assert [launch.backend for launch in launches] == ["inline", "inline"]
    for n, launch in enumerate(launches, 1):
        assert launch.config.get("BACKEND") == "inline"
        assert not _leaked(launch.sees), (
            f"start #{n} of the 27B would launch with Flash-Next's {_leaked(launch.sees)} "
            f"(qwen-server-run.sh sources .config, which held EXTRA_ARGS="
            f"{launch.config.get('EXTRA_ARGS')!r} under BACKEND=\"inline\")"
        )
        assert not _leaked(launch.env.get("EXTRA_ARGS", "")), launch.env
    # And the file left behind can be launched by any other path, e.g.
    # `codex-qwen.sh start` through the qwen-vllm unit, without the flags.
    after = shellconfig.read_config()
    assert after["BACKEND"] == "inline" and not _leaked(after.get("EXTRA_ARGS", "")), after


def test_a_restart_after_a_switch_never_sees_flashnext_flags(
    box: Path, tmp_path: Path, monkeypatch
) -> None:
    s, launches = _supervisor(tmp_path, monkeypatch)

    async def scenario() -> None:
        await s.start(**INLINE)
        s.actual_state = "READY"
        await s.restart(mode="immediate")                         # plain restart
        s.actual_state = "READY"
        await s.restart(mode="immediate", max_model_len=131072)   # Apply & restart

    asyncio.run(scenario())

    assert len(launches) == 3, [launch.backend for launch in launches]
    for n, launch in enumerate(launches, 1):
        assert launch.backend == "inline"
        assert not _leaked(launch.sees), (
            f"launch #{n} (a restart after switching to the 27B) would carry "
            f"Flash-Next's {_leaked(launch.sees)}"
        )
    assert launches[2].config.get("MAX_MODEL_LEN") == "131072"


def test_switching_back_to_flashnext_restores_its_tuned_flags(
    box: Path, tmp_path: Path, monkeypatch
) -> None:
    s, launches = _supervisor(tmp_path, monkeypatch)

    async def scenario() -> None:
        await s.start(**INLINE)
        _exited(s)
        await s.start(**FLASHNEXT)

    asyncio.run(scenario())

    back = launches[1]
    assert back.backend == "flashnext"
    assert back.sees == FLASHNEXT_EXTRA_ARGS, (
        f"Flash-Next came back without its tuned flags: launcher would read {back.sees!r}"
    )
    assert back.config.get("EXTRA_ARGS") == FLASHNEXT_EXTRA_ARGS, (
        "`llm start` reads EXTRA_ARGS from the file, so the file must carry them too"
    )
    record = supervisor.load_extra_args_record(tmp_path / "state")
    assert record["written"] == {"backend": "flashnext", "extra_args": FLASHNEXT_EXTRA_ARGS}
    assert "flashnext" not in record["stash"], "live flags must not also sit in the stash"


def test_a_backend_only_rewrite_by_another_writer_does_not_reattribute_the_flags(
    box: Path, tmp_path: Path, monkeypatch
) -> None:
    """`codex-qwen.sh set-config BACKEND inline` rewrites BACKEND alone. The
    flags Servedeck last wrote for Flash-Next are still Flash-Next's."""
    s, launches = _supervisor(tmp_path, monkeypatch)
    asyncio.run(s.start(**FLASHNEXT))
    _exited(s)
    subprocess.run([str(paths.CODEX_QWEN_SH), "set-config", "BACKEND", "inline"], check=True)
    assert shellconfig.read_config()["BACKEND"] == "inline"

    asyncio.run(s.start(**INLINE))

    assert not _leaked(launches[1].sees), launches[1].config


# --------------------------------------------------------------------------
# Over-correction guards: nothing that already belongs to the backend moves
# --------------------------------------------------------------------------
def _count_extra_args_writes(monkeypatch) -> list[str]:
    calls: list[str] = []
    real = getattr(shellconfig, "set_extra_args", None)
    if real is not None:     # absent before the fix: then there are no writes to count
        monkeypatch.setattr(shellconfig, "set_extra_args", lambda v: (calls.append(v), real(v))[1])
    return calls


def test_restarting_flashnext_keeps_its_flags_and_never_rewrites_them(
    box: Path, tmp_path: Path, monkeypatch
) -> None:
    writes = _count_extra_args_writes(monkeypatch)
    s, launches = _supervisor(tmp_path, monkeypatch)

    async def scenario() -> None:
        await s.start(**FLASHNEXT)
        s.actual_state = "READY"
        await s.restart(mode="immediate")

    asyncio.run(scenario())

    assert [launch.sees for launch in launches] == [FLASHNEXT_EXTRA_ARGS] * 2
    assert writes == [], f"a same-backend start rewrote EXTRA_ARGS: {writes}"
    # The EXTRA_ARGS line is still where the operator put it, under its comment.
    assert f'EXTRA_ARGS="{FLASHNEXT_EXTRA_ARGS}"' in box.read_text().split("\n")[3]


def test_the_27bs_own_flags_survive_its_own_restarts(
    box: Path, tmp_path: Path, monkeypatch
) -> None:
    """qwen-server-run.sh documents EXTRA_ARGS="--enforce-eager" as the 27B's
    Xid escalation. Once set for the 27B it must not be stripped as foreign."""
    box.write_text(BOX_CONFIG.replace(f'EXTRA_ARGS="{FLASHNEXT_EXTRA_ARGS}"',
                                      'EXTRA_ARGS="--enforce-eager"')
                             .replace('BACKEND="flashnext"', 'BACKEND="inline"'))
    writes = _count_extra_args_writes(monkeypatch)
    s, launches = _supervisor(tmp_path, monkeypatch)

    async def scenario() -> None:
        await s.start(**INLINE)
        s.actual_state = "READY"
        await s.restart(mode="immediate")

    asyncio.run(scenario())

    assert [launch.sees for launch in launches] == ["--enforce-eager"] * 2
    assert writes == []


def test_flags_round_trip_through_a_third_backend(box: Path, tmp_path: Path, monkeypatch, config_path) -> None:
    """flashnext -> inline -> glm53 -> flashnext: each backend gets its own
    flags back, and none of them ever gets another's."""
    config_path(textwrap.dedent(f"""\
        [backends.flashnext]
        launcher = "/bin/true"
        port = 8001
        [backends.inline]
        launcher = "/bin/true"
        port = 8004
        [backends.inline.env_map]
        [backends.glm53]
        launcher = "/bin/true"
        port = 8002
    """))
    s, launches = _supervisor(tmp_path, monkeypatch)
    glm = dict(repo_id="dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4", backend="glm53",
               served_name="glm53-flash", port=8002, util=0.95, max_model_len=327680, max_num_seqs=1)

    async def scenario() -> None:
        await s.start(**INLINE)
        _exited(s)
        await s.start(**glm)
        _exited(s)
        await s.start(**FLASHNEXT)

    asyncio.run(scenario())

    assert [(launch.backend, launch.sees) for launch in launches] == [
        ("inline", ""), ("glm53", ""), ("flashnext", FLASHNEXT_EXTRA_ARGS),
    ]


# --------------------------------------------------------------------------
# The writer itself
# --------------------------------------------------------------------------
def test_set_extra_args_is_byte_identical_to_save_config_kv(box: Path, tmp_path: Path) -> None:
    ours = box
    theirs = tmp_path / "theirs.config"
    theirs.write_text(BOX_CONFIG)
    script = tmp_path / "kv.sh"
    script.write_text(_fake_codex_qwen(theirs))
    script.chmod(0o755)

    for value in ("", FLASHNEXT_EXTRA_ARGS, "--enforce-eager"):
        shellconfig.set_extra_args(value)
        subprocess.run([str(script), "save-kv", "EXTRA_ARGS", value], check=True)
        assert ours.read_bytes() == theirs.read_bytes(), value
    assert stat.S_IMODE(os.stat(ours).st_mode) == 0o600, ".config must stay 0600"


@pytest.mark.parametrize("value", ['--limit-mm-per-prompt {"image":2}', "--x $(id)", "--x `id`", "a\\b", "a\nb"])
def test_a_value_no_reader_can_read_back_is_refused_and_the_file_untouched(box: Path, value: str) -> None:
    before = box.read_bytes()
    with pytest.raises(ValueError):
        shellconfig.set_extra_args(value)
    assert box.read_bytes() == before


# --------------------------------------------------------------------------
# Against this box's real readers (skipped elsewhere)
# --------------------------------------------------------------------------
_REAL_RUN_SH = Path("~/Projects/local_llm/bin/qwen-server-run.sh").expanduser()
_REAL_LLM = Path("~/Projects/local_llm/llm").expanduser()
_REAL_CODEX = Path("~/Projects/local_llm/codex-qwen.sh").expanduser()


@pytest.mark.skipif(not _REAL_CODEX.is_file(), reason="local_llm is not on this machine")
def test_the_stand_in_matches_the_real_codex_qwen_sh() -> None:
    text = _REAL_CODEX.read_text()
    body = re.search(r"^save_config_kv\(\) \{\n.*?^\}\n", text, re.S | re.M)
    assert body, "save_config_kv() not found in codex-qwen.sh -- has it changed shape?"
    assert body.group(0) == SAVE_CONFIG_KV, "codex-qwen.sh's save_config_kv() changed"
    allowed = re.search(r'^CONFIG_ALLOWED_KEYS="([^"]*)"', text, re.M)
    assert allowed and "EXTRA_ARGS" not in allowed.group(1).split(), (
        "codex-qwen.sh now allows EXTRA_ARGS; shellconfig.set_extra_args() can go through it"
    )


@pytest.mark.skipif(not _REAL_RUN_SH.is_file(), reason="local_llm is not on this machine")
def test_the_real_27b_launcher_reads_no_flashnext_flags_after_a_switch(
    box: Path, tmp_path: Path, monkeypatch
) -> None:
    """Run the actual bin/qwen-server-run.sh in its DRY_RUN mode (which only
    resolves and prints its argv) against the .config the sync wrote."""
    s, launches = _supervisor(tmp_path, monkeypatch)

    async def scenario() -> None:
        await s.start(**INLINE)
        _exited(s)
        await s.start(**INLINE)

    asyncio.run(scenario())

    for n, launch in enumerate(launches, 1):
        snapshot = tmp_path / f"config-at-launch-{n}"
        snapshot.write_text(launch.config_text)
        env = {**os.environ, **launch.env, "CONFIG_FILE": str(snapshot), "DRY_RUN": "1",
               # Belt and braces: if DRY_RUN ever stopped short-circuiting,
               # there is no vllm to exec.
               "VLLM_VENV": str(tmp_path / "no-venv")}
        out = subprocess.run(["bash", str(_REAL_RUN_SH)], env=env, capture_output=True,
                             text=True, timeout=30)
        resolved = re.search(r"^RESOLVED .*?extra_args=\[(.*?)\]", out.stdout, re.M)
        assert resolved, (out.returncode, out.stdout[-500:], out.stderr[-500:])
        assert "model=RadixArk/Qwen3.8-27B-NVFP4 port=8004" in resolved.group(0)
        assert not _leaked(resolved.group(1)), (
            f"launch #{n}: the real 27B launcher resolved extra_args=[{resolved.group(1)}]"
        )


@pytest.mark.skipif(not _REAL_LLM.is_file(), reason="local_llm is not on this machine")
def test_llm_reads_back_exactly_what_the_sync_wrote(box: Path, tmp_path: Path, monkeypatch) -> None:
    """`llm` parses .config with its own regex, skips (and now warns about) any
    line it cannot read, and takes the last assignment. After a round trip it
    must read Flash-Next's flags back, with no warning."""
    block = re.search(r"# --- config_read begin ---\n(.*?)# --- config_read end ---",
                      _REAL_LLM.read_text(), re.S)
    assert block, "llm's config_read markers are gone -- has read_config_file moved?"
    s, _launches = _supervisor(tmp_path, monkeypatch)

    async def scenario() -> None:
        await s.start(**INLINE)
        _exited(s)
        await s.start(**FLASHNEXT)

    asyncio.run(scenario())

    reader = tmp_path / "read.sh"
    reader.write_text(block.group(1) + '\nread_config_file "$1"\nprintf "%s\\n" "$BACKEND" "$EXTRA_ARGS"\n')
    out = subprocess.run(["bash", str(reader), str(box)], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert "WARNING" not in out.stderr, out.stderr
    assert out.stdout.split("\n")[:2] == ["flashnext", FLASHNEXT_EXTRA_ARGS], out.stdout
