"""Golden test: the 27B's v2-rendered argv vs. the legacy launcher's real argv.

WHAT THIS RUNS AND WHY IT IS SAFE (see the P1 report for the full reasoning):

``~/Projects/local_llm/bin/qwen-server-run.sh`` has a documented ``DRY_RUN=1``
mode (its own header comment, and see the guard-skipping logic through the
whole file) that resolves every variable, builds the real ``vllm serve`` argv,
prints it, and exits 0 WITHOUT touching the GPU, a service, or the training
markers. It is invoked here with ``CONFIG_FILE`` pointed at a copy of the live
``.config`` written under ``tmp_path`` — never the real file — so the only
input to this test that comes from outside tmp_path is the launcher script
itself (read-only) and its own venv/model-repo string constants (also
read-only). The single side effect outside tmp_path is the script's own
unconditional ``mkdir -p "$HERE/logs" "$HERE/run"`` on the launcher's own
pre-existing scratch directories — idempotent, not one of the paths this
packet is forbidden from touching (~/.config, ~/.codex, ~/.kimi-code, systemd
units), and verified harmless by reading the script before this test was
written.

Flash-Next and GLM are NOT invoked: ``vllm-qwen38next/serve.sh`` relaxes
``kernel.yama.ptrace_scope`` via ``sudo`` unconditionally (not gated on
DRY_RUN), and both of that script and ``vllm-glm53/serve-opt.sh`` symlink
``$CUDA_HOME/lib64`` in their own (live, sibling) project trees before
reaching their DRY_RUN check — both are exactly what the hard rules forbid
("No sudo", never touch another project's live tree). Per the packet's own
fallback instruction, those two are compared against the literal argv text in
the serve scripts instead (see ``test_flashnext_argv_matches_serve_sh`` and
``test_glm53_argv_matches_serve_opt_sh`` below).

ARGUMENT ORDER: v2 renders every model in ONE fixed order (see
``models.render_argv``'s docstring). None of the three legacy launchers uses
that order, and no two of them use the SAME order as each other. So "compare
as an ordered list" is implemented here as: parse each argv into a mapping of
flag -> value-tokens (boolean flags -> None), then compare those mappings.
That is the property that actually matters to vLLM's argparse (it does not
care about flag order) and it is the only comparison that could ever pass
given the fixed-order design — this is the "intentional, named" difference
the packet's instructions ask for.

FROZEN SNAPSHOT, NOT A LIVE DEPENDENCY: the 27B comparison is split in two.
``test_qwen27b_argv_and_env_match_frozen_snapshot`` is the one that ALWAYS
RUNS — it reads ``tests/fixtures/qwen27b-launcher-{argv,env}.txt`` (captured
2026-09-12; see those files' headers for exactly which ``.config`` values
produced them) and never shells out, so it has no dependency on the live,
mutable ``~/Projects/local_llm/.config`` that R2 (REDESIGN §1) is retiring.
``test_qwen27b_launcher_snapshot_is_current`` (tests/test_e2e_real.py, moved
there 2026-09-16 so the always-run unit suite has no live dependency at all)
is the live half: it re-runs the real launcher and asserts the snapshot has
not gone stale — it skips cleanly (via ``_run_dry_run_launcher``'s own
guards, imported from this module) on a box without the launcher/.config, and
it is the test to consult (and the fixture files to regenerate) if
``.config`` or the script's own defaults change.

ENV COMPARISON: v2 now renders ``CUDA_HOME``, ``PATH`` and
``VLLM_USE_FLASHINFER_SAMPLER`` (``models.render_env``, launch-environment-
completeness fix) — ALL THREE are compared, not just one. ``CUDA_HOME`` and
``VLLM_USE_FLASHINFER_SAMPLER`` must match exactly. ``PATH`` does not: the
legacy script prepends its venv/CUDA bins onto WHATEVER PATH the invoking
shell happened to have, which is a property of the terminal that ran the
capture, not of the model — v2 instead renders a fixed, minimal,
self-sufficient PATH (a systemd transient unit inherits the user manager's
environment, never the invoking shell's, so relying on an inherited PATH
would not even find ``env``/``sh``). Only the ``<venv>/bin:<CUDA_HOME>/bin:``
PREFIX of PATH is therefore compared — that prefix is the part that is
actually a property of the model/build, and it is where the two ends agree.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from servedeck import models

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
QWEN27B_ARGV_FIXTURE = FIXTURES_DIR / "qwen27b-launcher-argv.txt"
QWEN27B_ENV_FIXTURE = FIXTURES_DIR / "qwen27b-launcher-env.txt"

LOCAL_LLM = Path.home() / "Projects" / "local_llm"
SERVER_RUN_SH = LOCAL_LLM / "bin" / "qwen-server-run.sh"
LIVE_CONFIG = LOCAL_LLM / ".config"

#: Every env var v2's registry renders for the 27B today (models.render_env).
#: Requested from the live launcher too, both when the fixtures were captured
#: and by the live "still current" check, so neither side can silently claim a
#: var the other never mentions.
QWEN27B_ENV_VARS = "CUDA_HOME PATH VLLM_USE_FLASHINFER_SAMPLER"

VLLM_QWEN38NEXT = Path.home() / "Projects" / "vllm-qwen38next"
SERVE_SH = VLLM_QWEN38NEXT / "serve.sh"
VLLM_GLM53 = Path.home() / "Projects" / "vllm-glm53"
SERVE_OPT_SH = VLLM_GLM53 / "serve-opt.sh"

#: Flags whose VALUE v2 computes rather than transcribes (packet deliverable 3).
#: --host is here too: legacy 27B never passes --host at all, v2 always does.
NORMALIZED_FLAGS = {"--gpu-memory-utilization", "--served-model-name", "--host"}


def _parse_argv(argv: list[str]) -> dict[str, list[str] | None]:
    """flag -> list of value tokens (until the next --flag), or None for a
    boolean flag. Non-flag leading tokens (the binary, "serve", the repo id)
    are not flags and are asserted separately by the caller."""
    out: dict[str, list[str] | None] = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            values: list[str] = []
            j = i + 1
            while j < len(argv) and not argv[j].startswith("--"):
                values.append(argv[j])
                j += 1
            out[tok] = values if values else None
            i = j
        else:
            i += 1
    return out


def _drop_normalized(parsed: dict[str, list[str] | None]) -> dict[str, list[str] | None]:
    return {k: v for k, v in parsed.items() if k not in NORMALIZED_FLAGS}


def _run_dry_run_launcher(
    tmp_path: Path, dry_env_vars: str = QWEN27B_ENV_VARS
) -> tuple[list[str], dict[str, str]]:
    """Copy the live .config to tmp_path, run qwen-server-run.sh DRY_RUN=1
    against that copy, and parse its ENV/ARGV dump lines."""
    if not SERVER_RUN_SH.is_file():
        pytest.skip(f"{SERVER_RUN_SH} not present on this box")
    if not LIVE_CONFIG.is_file():
        pytest.skip(f"{LIVE_CONFIG} not present on this box")

    tmp_config = tmp_path / "config-27b-snapshot"
    tmp_config.write_text(LIVE_CONFIG.read_text())

    env = dict(os.environ)
    env.update(
        {
            "DRY_RUN": "1",
            "CONFIG_FILE": str(tmp_config),
            "DRY_ENV_VARS": dry_env_vars,
        }
    )
    proc = subprocess.run(
        ["bash", str(SERVER_RUN_SH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"qwen-server-run.sh DRY_RUN failed: {proc.stderr}"

    env_out: dict[str, str] = {}
    argv_out: list[str] = []
    for line in proc.stdout.splitlines():
        if line.startswith("ENV "):
            name, _, value = line[len("ENV ") :].partition("=")
            env_out[name] = value
        elif line.startswith("ARGV "):
            argv_out.append(line[len("ARGV ") :])
    assert argv_out, f"no ARGV lines in launcher output:\n{proc.stdout}"
    return argv_out, env_out


def _load_argv_fixture(path: Path) -> list[str]:
    """One argv token per "ARGV <token>" line; comment/header lines ignored."""
    return [
        line[len("ARGV ") :] for line in path.read_text().splitlines() if line.startswith("ARGV ")
    ]


def _load_env_fixture(path: Path) -> dict[str, str]:
    """{name: value} from "ENV NAME=value" lines; comment/header lines ignored."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.startswith("ENV "):
            name, _, value = line[len("ENV ") :].partition("=")
            out[name] = value
    return out


def _assert_path_prefix_matches(label: str, path_value: str, expected_prefix: str) -> None:
    assert path_value.startswith(expected_prefix), (
        f"{label} PATH does not start with the expected <venv>/bin:<CUDA_HOME>/bin "
        f"prefix {expected_prefix!r}: {path_value!r}"
    )


@pytest.fixture
def registry() -> models.Registry:
    return models.load(REPO_ROOT / "models.toml")


def test_qwen27b_argv_and_env_match_frozen_snapshot(registry):
    """The always-run half of the golden test: reads the frozen fixtures, no
    subprocess, no dependency on the live/mutable ~/Projects/local_llm/.config.
    See the module docstring ("FROZEN SNAPSHOT...") and the fixture files'
    headers for what this compares and why PATH is a named exception."""
    legacy_argv = _load_argv_fixture(QWEN27B_ARGV_FIXTURE)
    legacy_env = _load_env_fixture(QWEN27B_ENV_FIXTURE)

    m = registry.models["qwen27b"]
    build = registry.builds[m.build]
    vllm_bin = str(Path(build.venv).expanduser() / "bin" / "vllm")
    ctx = models.native_ctx(m.repo)
    assert ctx == 262144, "native ctx must equal the snapshot's MAX_MODEL_LEN for this comparison to be meaningful"
    v2_argv = models.render_argv(m, vllm_bin, util=0.91, ctx_tokens=ctx, port=m.port)
    v2_env = models.render_env(m, build)

    # argv[0] (the vllm binary) and the repo id must match EXACTLY, not just as
    # a flag/value mapping — these are positional, not flags.
    assert legacy_argv[0] == vllm_bin, "models.toml's [builds.stock].venv must match the snapshot's venv"
    assert legacy_argv[1] == "serve"
    assert legacy_argv[2] == m.repo
    assert v2_argv[0] == vllm_bin
    assert v2_argv[1] == "serve"
    assert v2_argv[2] == m.repo

    legacy_flags = _drop_normalized(_parse_argv(legacy_argv[3:]))
    v2_flags = _drop_normalized(_parse_argv(v2_argv[3:]))

    assert v2_flags == legacy_flags, (
        f"registry-rendered flags differ from the snapshot's.\n"
        f"legacy only: { {k: v for k, v in legacy_flags.items() if k not in v2_flags} }\n"
        f"v2 only:     { {k: v for k, v in v2_flags.items() if k not in legacy_flags} }\n"
        f"value mismatches: "
        f"{ {k: (legacy_flags[k], v2_flags[k]) for k in legacy_flags if k in v2_flags and legacy_flags[k] != v2_flags[k]} }"
    )

    # Both must have exactly ONE served name in the legacy output (no aliasing
    # yet), and v2's served-model-name list must START with the same id.
    legacy_full = _parse_argv(legacy_argv[3:])
    assert legacy_full["--served-model-name"] == [m.id]
    v2_full = _parse_argv(v2_argv[3:])
    assert v2_full["--served-model-name"][0] == m.id

    # Confirms the premise behind treating --host as normalized: the legacy
    # launcher truly never passes --host (qwen-server-run.sh only ever sets
    # --port), so v2 adding --host 127.0.0.1 is a deliberate, one-way addition,
    # not a value v2 merely recomputes.
    assert "--host" not in legacy_argv

    # ENV: compare every var v2 renders (models.render_env) against the
    # snapshot — not just one. CUDA_HOME and VLLM_USE_FLASHINFER_SAMPLER must
    # match exactly; PATH only by its <venv>/bin:<CUDA_HOME>/bin prefix (the
    # rest is the capturing shell's own PATH — see the env fixture's header).
    assert set(legacy_env) == set(v2_env), (
        f"env var set differs: snapshot only {set(legacy_env) - set(v2_env)}, "
        f"v2 only {set(v2_env) - set(legacy_env)}"
    )
    assert legacy_env["CUDA_HOME"] == v2_env["CUDA_HOME"]
    assert legacy_env["VLLM_USE_FLASHINFER_SAMPLER"] == v2_env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    expected_prefix = f"{vllm_bin.rsplit('/bin/vllm', 1)[0]}/bin:{v2_env['CUDA_HOME']}/bin:"
    _assert_path_prefix_matches("snapshot", legacy_env["PATH"], expected_prefix)
    _assert_path_prefix_matches("v2", v2_env["PATH"], expected_prefix)


# test_qwen27b_snapshot_is_current — the live half that re-runs the real
# launcher against ~/Projects/local_llm/.config — moved to tests/test_e2e_real.py
# on 2026-09-16. Its whole job is comparing the frozen fixtures above against
# that LIVE, mutable file; replacing the live read with another committed
# snapshot would make it compare the frozen fixture against itself, which can
# never fail. That is a hermetic test's failure mode, not this one's job, so
# per the "genuinely requires the live file" case it stays a live test — just
# not in the always-run unit suite. See test_e2e_real.py for the moved test
# and why living there (not here) is what keeps it able to actually fail.


# --------------------------------------------------------------------------- #
# Flash-Next / GLM: cannot be dry-run safely (sudo / live-tree mutation before
# their own DRY_RUN check — see module docstring), so compare against the
# literal argv text in the serve scripts instead, as the packet permits.
# --------------------------------------------------------------------------- #


def test_flashnext_argv_matches_serve_sh_literal_text(registry):
    if not SERVE_SH.is_file():
        pytest.skip(f"{SERVE_SH} not present on this box")
    text = SERVE_SH.read_text()
    m = registry.models["flashnext"]

    # Static (non-env-conditional) tokens that models.toml's flags/reasoning/
    # tools MUST reproduce, transcribed by hand from serve.sh's VLLM_ARGV block
    # (see models.toml's citation comments for line numbers).
    for literal in (
        '--tool-call-parser qwen3_coder',
        '--reasoning-parser qwen3',
        '--tensor-parallel-size 1',
        '--distributed-executor-backend mp',
        '--enable-prefix-caching',
        '--enable-auto-tool-choice',
    ):
        assert literal in text, f"serve.sh no longer contains {literal!r}; re-check models.toml"

    assert m.tools is not None and m.tools.parser == "qwen3_coder"
    assert m.reasoning is not None and m.reasoning.parser == "qwen3"
    assert "--tensor-parallel-size" in m.flags and "1" in m.flags
    assert "--distributed-executor-backend" in m.flags and "mp" in m.flags
    assert "--enable-prefix-caching" in m.flags

    # The script-default values (MAX_SEQS, MAX_BATCHED, MM_LIMIT_JSON, KV_DTYPE,
    # SPEC_TOKENS) that models.toml pins as static flags.
    assert 'MAX_SEQS:-1' in text
    assert 'MAX_BATCHED:-8192' in text
    assert '{"image":2,"video":0}' in text
    assert 'KV_DTYPE:-auto' in text
    assert 'SPEC_TOKENS:-3' in text
    flags_text = " ".join(m.flags)
    assert "--max-num-seqs 1" in flags_text
    assert "--max-num-batched-tokens 8192" in flags_text
    assert '{"image":2,"video":0}' in flags_text
    assert "--kv-cache-dtype auto" in flags_text
    assert '"num_speculative_tokens":3' in flags_text


def test_glm53_argv_matches_serve_opt_sh_literal_text(registry):
    if not SERVE_OPT_SH.is_file():
        pytest.skip(f"{SERVE_OPT_SH} not present on this box")
    text = SERVE_OPT_SH.read_text()
    m = registry.models["glm53"]

    for literal in (
        "--tool-call-parser glm47",
        "--reasoning-parser glm47",
        "--offload-backend uva",
        "--cpu-offload-params experts",
        "--enable-prefix-caching",
    ):
        assert literal in text, f"serve-opt.sh no longer contains {literal!r}; re-check models.toml"

    assert m.tools is not None and m.tools.parser == "glm47"
    assert m.reasoning is not None and m.reasoning.parser == "glm47"
    flags_text = " ".join(m.flags)
    assert "--offload-backend uva" in flags_text
    assert "--cpu-offload-gb 125" in flags_text
    assert "--cpu-offload-params experts" in flags_text
    assert "--enable-prefix-caching" in flags_text
    # Script defaults pinned as static flags.
    assert "MAX_SEQS:-1" in text and "--max-num-seqs 1" in flags_text
    assert "MAX_BATCHED:-8192" in text and "--max-num-batched-tokens 8192" in flags_text
    assert 'MOE_BACKEND:-marlin' in text and '"moe_backend":"marlin"' in flags_text
    assert 'SPEC_TOKENS:-2' in text and '"num_speculative_tokens":2' in flags_text
    # serve-opt.sh derives --kv-cache-memory-bytes as MAX_LEN * 17200 at launch
    # time (serve-opt.sh:315-316); the registry pins ctx to the validated 327,680
    # and carries the product literally, so the two must agree by arithmetic.
    assert "KV_BYTES_PER_TOKEN:-17200" in text or "17200" in text
    assert m.ctx == 327680, "GLM ctx is pinned to the validated VRAM ceiling, not native"
    # Derived, not pinned (2026-09-18): the flag is emitted by render_argv from
    # the ctx the engine is actually given, because a pinned product plus a
    # movable context control is two sources of truth for one number — the page
    # moved ctx and left the cap sized for the old length, which vLLM refuses
    # only after a six-minute 181 GiB load.
    assert "--kv-cache-memory-bytes" not in flags_text
    assert m.kv_cache_bytes_per_token == 17200
    rendered = models.render_argv(m, "/v/bin/vllm", 0.95, 327680, m.port)
    assert rendered[rendered.index("--kv-cache-memory-bytes") + 1] == "5636096000", (
        "the derived cap at the validated ctx must equal what serve-opt.sh computed"
    )
    assert 327680 * 17200 == 5636096000
