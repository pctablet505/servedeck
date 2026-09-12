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
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from servedeck import models

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_LLM = Path.home() / "Projects" / "local_llm"
SERVER_RUN_SH = LOCAL_LLM / "bin" / "qwen-server-run.sh"
LIVE_CONFIG = LOCAL_LLM / ".config"

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


def _run_dry_run_launcher(tmp_path: Path) -> tuple[list[str], dict[str, str]]:
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
            "DRY_ENV_VARS": "VLLM_USE_FLASHINFER_SAMPLER",
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


@pytest.fixture
def registry() -> models.Registry:
    return models.load(REPO_ROOT / "models.toml")


def test_qwen27b_argv_matches_legacy_launcher(tmp_path, registry):
    legacy_argv, legacy_env = _run_dry_run_launcher(tmp_path)

    m = registry.models["qwen27b"]
    # The legacy launcher's own venv (VLLM_VENV default), matching models.toml's
    # [builds] entry for build="stock".
    vllm_bin = str(Path(registry.builds[m.build]).expanduser() / "bin" / "vllm")
    ctx = models.native_ctx(m.repo)
    assert ctx == 262144, "native ctx must equal the live MAX_MODEL_LEN for this comparison to be meaningful"
    v2_argv = models.render_argv(m, vllm_bin, util=0.91, ctx_tokens=ctx, port=m.port)

    # argv[0] (the vllm binary) and the repo id must match EXACTLY, not just as
    # a flag/value mapping — these are positional, not flags.
    assert legacy_argv[0] == vllm_bin, "the golden test's own [builds] path must match the live venv"
    assert legacy_argv[1] == "serve"
    assert legacy_argv[2] == m.repo
    assert v2_argv[0] == vllm_bin
    assert v2_argv[1] == "serve"
    assert v2_argv[2] == m.repo

    legacy_flags = _drop_normalized(_parse_argv(legacy_argv[3:]))
    v2_flags = _drop_normalized(_parse_argv(v2_argv[3:]))

    assert v2_flags == legacy_flags, (
        f"registry-rendered flags differ from the live launcher's.\n"
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

    # VLLM_USE_FLASHINFER_SAMPLER=0 is a [defaults.env] entry — confirm it is
    # actually what the live launcher exports too.
    assert legacy_env.get("VLLM_USE_FLASHINFER_SAMPLER") == "0"
    assert models.render_env(m)["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


def test_qwen27b_no_host_flag_in_legacy_output(tmp_path):
    """Confirms the premise behind treating --host as normalized: the legacy
    launcher truly never passes --host (qwen-server-run.sh only ever sets
    --port), so v2 adding --host 127.0.0.1 is a deliberate, one-way addition,
    not a value v2 merely recomputes."""
    legacy_argv, _ = _run_dry_run_launcher(tmp_path)
    assert "--host" not in legacy_argv


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
    assert "--kv-cache-memory-bytes 5636096000" in flags_text
    assert 327680 * 17200 == 5636096000
