"""The launch contract: what a dashboard start actually runs.

The owner's report was "I am not able to launch a vllm server for selected
model", and the failure is not in any one place — it is that a start from the
dashboard and a start from ``~/Projects/local_llm/llm`` produce two different
commands from the same configuration. ``llm start``'s own launch, verbatim
(``llm``:169-175)::

    ( cd "$(dirname "$SERVE_SH")" && setsid nohup env \\
        MODEL="$MODEL_REPO" SERVED_NAME="$SERVED_NAME" PORT="$PORT" \\
        MAX_LEN="$MAX_MODEL_LEN" MAX_SEQS="$MAX_NUM_SEQS" \\
        GPU_UTIL="$GPU_MEM_UTIL" EXTRA_ARGS="${EXTRA_ARGS:-}" \\
        "$SERVE_SH" >> "$SERVER_LOG" 2>&1 </dev/null & ...

Three things are asserted here, per backend, with the subprocess layer
mocked — nothing in this file starts a server, touches the GPU, or writes
outside tmp_path:

* the argv is the launcher and nothing else (Servedeck delegates, it does not
  build a ``vllm serve`` line);
* the environment carries every variable ``llm start`` exports, with the same
  values, including ``EXTRA_ARGS``;
* the cwd is the launcher's project root.

Plus the two resolvers that decide WHICH port and WHICH served name a start
gets when the dashboard's Apply button (which sends neither) is the caller.
"""

from __future__ import annotations

import asyncio
import re
import types
from pathlib import Path

import pytest

from servedeck import config, preflight, procctl, supervisor

# --------------------------------------------------------------------------- #
# A configuration shaped exactly like the one on the box this was found on:
# three backends, three ports, three launchers, one of them under bin/.
# --------------------------------------------------------------------------- #

def _box_config(tmp_path: Path, config_path) -> tuple[dict[str, Path], object]:
    trees = {}
    for name, rel in (
        ("glm53", "vllm-glm53/serve-opt.sh"),
        ("flashnext", "vllm-qwen38next/serve-abliterated.sh"),
        ("inline", "local_llm/bin/qwen-server-run.sh"),
    ):
        script = tmp_path / rel
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
        trees[name] = script

    body = f"""
gpu_total_mib = 97887

[backends.glm53]
launcher = "{trees['glm53']}"
port = 8002
architectures = ["Glm5NextForConditionalGeneration"]
cold_boot_range_s = [420.0, 900.0]

[backends.glm53.env_map]
repo_id = "MODEL"
port = "PORT"
max_model_len = "MAX_LEN"
util = "GPU_UTIL"
max_num_seqs = "MAX_SEQS"
served_name = "SERVED_NAME"

[backends.flashnext]
launcher = "{trees['flashnext']}"
port = 8001
needs_tty = true
architectures = ["Qwen4ExpForConditionalGeneration"]

[backends.flashnext.env_map]
repo_id = "MODEL"
port = "PORT"
max_model_len = "MAX_LEN"
util = "GPU_UTIL"
max_num_seqs = "MAX_SEQS"
served_name = "SERVED_NAME"

[backends.inline]
launcher = "{trees['inline']}"
# The launcher lives under bin/; the tree it belongs to is one level up.
cwd = "{tmp_path / 'local_llm'}"
port = 8004
architectures = ["Qwen3_5ForConditionalGeneration"]

[backends.inline.env_map]
"""
    return trees, config_path(body)


def _supervisor(tmp_path: Path) -> supervisor.Supervisor:
    s = supervisor.Supervisor(
        state_dir=tmp_path / "state",
        clock=lambda: 1000.0,
        launch_fn=lambda *a, **k: None,
        stop_fn=lambda h, **kw: procctl.StopResult(True, "term", 0.0, "faked"),
        history_path=tmp_path / "history.jsonl",
    )
    return s


def _build(s, backend: str, **over):
    kwargs = dict(
        backend=backend,
        repo_id="mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4",
        served_name="qwen38-flash-next",
        port=8001,
        util=0.95,
        max_model_len=262144,
        max_num_seqs=16,
    )
    kwargs.update(over)
    return s._build_launch(**kwargs)


# --------------------------------------------------------------------------- #
# argv / env / cwd
# --------------------------------------------------------------------------- #

def test_argv_is_the_launcher_alone_for_every_backend(tmp_path, config_path):
    """Servedeck delegates: it runs the launcher and passes settings through
    the environment. A ``vllm serve`` line built here would be a second source
    of truth for flags that the launcher already owns."""
    trees, _ = _box_config(tmp_path, config_path)
    s = _supervisor(tmp_path)
    for backend, script in trees.items():
        argv, _env, _cwd, _logs = _build(s, backend)
        assert argv == [str(script)], f"{backend}: {argv}"


@pytest.mark.parametrize("backend", ["glm53", "flashnext"])
def test_env_matches_what_llm_start_exports(tmp_path, config_path, backend, monkeypatch):
    """Every variable ``llm start`` exports, with the same value.

    ``EXTRA_ARGS`` is the one that was missing. On this box ``.config`` carries
    ``--language-model-only --mamba-ssm-cache-dtype bfloat16
    --prefix-match-unit 208`` for Flash-Next, and ``serve.sh`` interpolates it
    at the end of its ``vllm serve`` line (serve.sh:155). Dropping it means a
    dashboard launch boots a materially different server from the CLI's, with
    no error anywhere to say so.
    """
    _box_config(tmp_path, config_path)
    monkeypatch.setattr(
        supervisor.shellconfig,
        "read_config",
        lambda: {"BACKEND": backend, "EXTRA_ARGS": "--language-model-only --prefix-match-unit 208"},
    )
    s = _supervisor(tmp_path)
    _argv, env, _cwd, _logs = _build(s, backend)

    expected = {
        "MODEL": "mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4",
        "SERVED_NAME": "qwen38-flash-next",
        "PORT": "8001",
        "MAX_LEN": "262144",
        "MAX_SEQS": "16",
        "GPU_UTIL": "0.95",
        "EXTRA_ARGS": "--language-model-only --prefix-match-unit 208",
    }
    for key, value in expected.items():
        assert env.get(key) == value, f"{backend}: {key}={env.get(key)!r}, want {value!r}"
    # SPEC.md correction C9 — 17 of 72 recorded exits were the DNS class.
    assert env["HF_HUB_OFFLINE"] == "1"


def test_extra_args_is_not_taken_from_another_backends_config(tmp_path, config_path, monkeypatch):
    """``.config`` describes ONE backend at a time. Handing GLM's extra flags
    to a Qwen launcher is worse than handing it none."""
    _box_config(tmp_path, config_path)
    monkeypatch.setattr(
        supervisor.shellconfig,
        "read_config",
        lambda: {"BACKEND": "glm53", "EXTRA_ARGS": "--glm-only-flag"},
    )
    s = _supervisor(tmp_path)
    _argv, env, _cwd, _logs = _build(s, "flashnext")
    assert "EXTRA_ARGS" not in env, env


def test_cwd_is_the_project_root_not_the_bin_directory(tmp_path, config_path):
    """``llm`` runs the launcher from ``dirname "$SERVE_SH"``, which is the
    project root for a serve script that sits at one — and is NOT the project
    root for a launcher under ``bin/``.

    The default still matches the CLI exactly (see the two serve scripts
    below); what this pins is that a backend CAN declare a different working
    directory, because for a launcher under ``bin/`` there is no derivable
    right answer. NOT claimed: that this box's ``qwen-server-run.sh`` needs
    it — that script derives its own root from ``${BASH_SOURCE[0]}/..`` and is
    indifferent to cwd. A launcher that uses a relative path is not."""
    trees, _ = _box_config(tmp_path, config_path)
    s = _supervisor(tmp_path)

    for backend in ("glm53", "flashnext"):
        _argv, _env, cwd, _logs = _build(s, backend)
        assert Path(cwd) == trees[backend].parent

    _argv, _env, cwd, _logs = _build(s, "inline")
    assert Path(cwd) == tmp_path / "local_llm", cwd
    assert Path(cwd).name != "bin"


def test_inline_launcher_is_handed_no_settings_it_does_not_read(tmp_path, config_path):
    """An empty ``[backends.inline.env_map]`` is a deliberate "this launcher
    takes its settings from a file"; it must not be handed a MODEL it will
    ignore while its own ``.config`` says something else."""
    _box_config(tmp_path, config_path)
    s = _supervisor(tmp_path)
    _argv, env, _cwd, _logs = _build(s, "inline")
    assert set(env) <= {"HF_HUB_OFFLINE", "EXTRA_ARGS"}, env


@pytest.mark.skipif(
    not Path("~/Projects/local_llm/llm").expanduser().is_file(),
    reason="the local_llm CLI is not on this machine",
)
def test_the_real_llm_script_exports_nothing_servedeck_drops(tmp_path, config_path, monkeypatch):
    """Drift guard against the actual CLI, not a copy of it.

    Reads the `env ... "$SERVE_SH"` block out of ``~/Projects/local_llm/llm``
    and asserts Servedeck exports every variable named there. If someone adds
    a knob to the CLI's launch, this fails instead of the dashboard silently
    booting the old configuration.
    """
    text = Path("~/Projects/local_llm/llm").expanduser().read_text()
    block = re.search(r"setsid nohup env \\\n(.*?)\"\$SERVE_SH\"", text, re.S)
    assert block, "could not find llm's launch block — has cmd_start changed shape?"
    wanted = set(re.findall(r"\b([A-Z][A-Z0-9_]*)=\"\$", block.group(1)))
    assert wanted, "parsed no variables out of llm's launch block"
    # Variables the CLI passes that Servedeck deliberately does NOT, each with
    # a reason. Anything the CLI grows that is not on this list fails here --
    # which is the point: a new launcher knob must be a decision, not a
    # silent difference between the two ways of starting the same server.
    #
    # PROFILE (added to `llm` on 2026-09-09): selects a launcher profile in a
    # LAUNCHER/LAUNCHER_PROFILE scheme that is still being built. Servedeck
    # passing a knob whose contract it does not know is exactly what
    # servedeck.toml's [backends.*.env] comment warns against; when the scheme
    # settles, the answer is one line in that table, not code here.
    KNOWINGLY_NOT_PASSED = {"PROFILE"}
    wanted -= KNOWINGLY_NOT_PASSED

    _box_config(tmp_path, config_path)
    monkeypatch.setattr(
        supervisor.shellconfig,
        "read_config",
        lambda: {"BACKEND": "flashnext", "EXTRA_ARGS": "--x"},
    )
    s = _supervisor(tmp_path)
    _argv, env, _cwd, _logs = _build(s, "flashnext")
    missing = wanted - set(env)
    assert not missing, f"llm start exports {sorted(missing)}; a Servedeck launch does not"


# --------------------------------------------------------------------------- #
# Which port and which served name a dashboard start gets
# --------------------------------------------------------------------------- #

def _desired(**kw):
    d = supervisor.DesiredState()
    for k, v in kw.items():
        setattr(d, k, v)
    return d


def test_switching_backend_moves_to_that_backends_port(tmp_path, config_path):
    """The Apply button sends a model and a backend, never a port.

    The old fallback chain was ``body.port or desired.port or rt.port`` — the
    PREVIOUS run's port. With ``desired.json`` holding GLM's 8002 (which is
    exactly what it holds on this box), picking a Flash-Next model started
    Flash-Next on 8002 while the gateway went on proxying 8001, and the
    dashboard reported "backend not reachable" about a server that had booted
    perfectly.
    """
    _box_config(tmp_path, config_path)
    d = _desired(backend="glm53", port=8002, repo_id="dealignai/GLM-5.3-Flash", served_name="glm53-flash")
    assert supervisor.resolve_port("flashnext", d, fallback=8002) == 8001
    assert supervisor.resolve_port("inline", d, fallback=8002) == 8004


def test_an_explicit_port_and_a_same_backend_port_are_both_respected(tmp_path, config_path):
    """Config is the answer when the backend changes — not a policy that
    overrides a port somebody deliberately chose."""
    _box_config(tmp_path, config_path)
    d = _desired(backend="glm53", port=8099)
    assert supervisor.resolve_port("glm53", d) == 8099          # same backend: kept
    assert supervisor.resolve_port("flashnext", d, explicit=9001) == 9001


def test_a_new_model_does_not_inherit_the_previous_models_served_name(tmp_path, config_path):
    """The Apply button sends no served name either, and the old fallback was
    ``desired.served_name``. Starting a Qwen checkpoint after a GLM one
    therefore advertised it over ``/v1/models`` as ``glm53-flash``: two
    different models answering to one name, which is precisely why the UI
    could not say which model was serving."""
    _box_config(tmp_path, config_path)
    d = _desired(backend="glm53", repo_id="dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4", served_name="glm53-flash")

    got = supervisor.resolve_served_name(
        "flashnext", "mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4", d
    )
    assert got != "glm53-flash"
    assert got == "Qwen3.8-Flash-Next-Uncensored-NVFP4"

    # Same repo: the alias clients are configured against survives a restart.
    assert (
        supervisor.resolve_served_name(
            "glm53", "dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4", d
        )
        == "glm53-flash"
    )
    # An explicit name always wins.
    assert supervisor.resolve_served_name("flashnext", "a/b", d, explicit="mine") == "mine"


# --------------------------------------------------------------------------- #
# Preflight: the start that cannot succeed
# --------------------------------------------------------------------------- #

def test_a_port_already_in_use_blocks_the_start(tmp_path, config_path, monkeypatch):
    """vLLM binds its HTTP port LAST, after loading weights. Starting on a
    taken port spends four to ten minutes loading a model and then dies on
    "address already in use" — a long, healthy-looking boot that ends in a
    failure whose cause has scrolled past. ``llm start`` checks the port
    before it will launch anything; Servedeck did not check it at all, on a
    box where :8000 is permanently held by an unrelated service."""
    _box_config(tmp_path, config_path)
    monkeypatch.setattr(preflight, "_listener_pid", lambda port: 4242 if port == 8004 else None)

    blocked = preflight.blocking_failures(preflight.run_preflight(backend=None, port=8004))
    assert [c.id for c in blocked] == ["PORT_IN_USE"]
    assert "4242" in blocked[0].detail

    assert preflight.blocking_failures(preflight.run_preflight(backend=None, port=8001)) == []
    # No port to check is not a failure.
    assert preflight.check_port_free(None) is None


def test_start_refuses_when_the_port_is_taken(tmp_path, config_path, monkeypatch):
    """End to end through the supervisor: the launch must not happen."""
    _box_config(tmp_path, config_path)
    monkeypatch.setattr(preflight, "_listener_pid", lambda port: 999)
    monkeypatch.setattr(preflight, "check_gpu_responsive",
                        lambda: preflight.PreflightCheck("GPU_UNRESPONSIVE", True, "block", "ok", "ok", None))

    launched: list = []
    s = _supervisor(tmp_path)
    s._launch_fn = lambda argv, env, cwd, log: launched.append(argv)
    s._sync_shell_config = types.MethodType(lambda self, **kw: None, s)  # type: ignore[assignment]

    asyncio.run(
        s.start(
            repo_id="mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4",
            backend="flashnext", served_name="qwen38-flash-next", port=8001,
            util=0.95, max_model_len=262144, max_num_seqs=16,
        )
    )
    assert launched == [], "started a server on a port that is already taken"
    assert s.actual_state == "FAILED"
    assert "PORT_IN_USE" in (s.last_error or "")


# --------------------------------------------------------------------------- #
# Configured guards that must actually guard
# --------------------------------------------------------------------------- #

def test_a_configured_training_marker_blocks_a_start(tmp_path, config_path):
    """``training_markers`` in servedeck.toml was parsed, exposed, and then
    never consulted by the only code that can refuse a launch — a guard that
    reads as switched on and is not. On this box that was the AlgoTrading
    marker, which stopped blocking the GPU when the fork was folded in."""
    from servedeck import capacity

    marker = tmp_path / "AlgoTrading" / "run" / "training_in_progress"
    marker.parent.mkdir(parents=True)
    config_path(f'''
gpu_total_mib = 97887
training_markers = ["{marker}"]
''')
    capacity.refresh_limits()

    assert not marker.exists()
    assert preflight.check_training_marker().ok

    marker.write_text("")
    check = preflight.check_training_marker()
    assert not check.ok, "a declared training marker did not block a start"
    assert str(marker) in check.detail


def test_a_configured_cold_boot_range_is_used(tmp_path, config_path):
    """How long a checkpoint takes to load is a fact about one machine (181
    GiB of weights plus a Marlin repack behaves nothing like a 40 GiB model),
    so it is configuration. The fold-in dropped a measured 420-900 s envelope
    because it lived in the package as a literal."""
    from servedeck import history

    _box_config(tmp_path, config_path)
    assert history.cold_boot_range_s("glm53") == (420.0, 900.0)
    assert history.cold_boot_range_s("flashnext") == (240.0, 600.0)  # built-in fallback
    assert history.cold_boot_range_s("nosuch") is None


def test_config_declares_the_backend_a_boot_log_is_read_from(tmp_path, config_path):
    """Restored from the fork, generalised.

    A backend whose launcher has no log management of its own (no tee, no
    exec redirect) has no fixed path to tail, and returning ANOTHER backend's
    log is worse than returning none: phases.classify() would match an error
    line from a different model's run and file it as this run's failure_code.
    """
    import os

    _box_config(tmp_path, config_path)
    # glm53 declares no log_path in the config above.
    assert supervisor._default_log_paths("glm53", boot_log_dir=tmp_path / "nope") == []

    d = tmp_path / "boot_logs"
    d.mkdir()
    old, new = d / "glm53-20260101-000000.log", d / "glm53-20260902-000000.log"
    for i, f in enumerate((old, new)):
        f.write_text("x")
        os.utime(f, (1000 + i * 1000, 1000 + i * 1000))
    (d / "flashnext-20260903-000000.log").write_text("x")  # other backend, newer

    got = supervisor._default_log_paths("glm53", boot_log_dir=d)
    assert [Path(p) for p in got] == [new], got


def test_the_shell_configs_gateway_url_key_is_the_one_the_shell_writes():
    """``codex-qwen.sh``'s CONFIG_ALLOWED_KEYS spells this ``COLDSTART_URL``,
    and ``.config`` on this box carries ``COLDSTART_URL=""``. Renaming it in
    Python alone meant Servedeck read a key that never exists — and writing
    the new name would be re-rejected by the shell script's own allow list.
    Both spellings have to work while one file has two readers."""
    from servedeck import shellconfig

    assert "COLDSTART_URL" in shellconfig.ALLOWED_SET_KEYS
    assert "SERVEDECK_URL" in shellconfig.ALLOWED_SET_KEYS
    assert shellconfig._base_url({"COLDSTART_URL": "http://127.0.0.1:8010"}) == \
        "http://127.0.0.1:8010/v1"
    # The new name wins where both are set.
    assert shellconfig._base_url(
        {"COLDSTART_URL": "http://old:1", "SERVEDECK_URL": "http://new:2"}
    ) == "http://new:2/v1"
    # Empty stays empty: .config ships COLDSTART_URL="" and that must fall
    # through to the port-derived default, not to "/v1".
    assert shellconfig._base_url({"COLDSTART_URL": "", "PORT": "8001"}) == \
        "http://localhost:8001/v1"


# --------------------------------------------------------------------------- #
# The same thing again, through the endpoint the Apply button actually calls.
# These assert on VALUES, so they fail with the wrong port and the wrong name
# on the unfixed code rather than on a missing helper.
# --------------------------------------------------------------------------- #

def _capture_start(monkeypatch, tmp_path, desired_kwargs):
    """A supervisor whose start() records its kwargs and does nothing."""
    from servedeck import app as capp

    recorded: dict = {}

    class _Sup:
        def __init__(self):
            self.desired = _desired(**desired_kwargs)

        async def start(self, **kw):
            recorded.update(kw)

    s = _Sup()
    monkeypatch.setattr(capp, "_need_sup", lambda: s)
    monkeypatch.setattr(capp, "sup", lambda: s)
    monkeypatch.setattr(capp, "_state", lambda: {})
    monkeypatch.setattr(capp.rt, "port", 8002, raising=False)
    return capp, recorded


def _drive(capp):
    """POST the body the Apply button sends, and let the task it spawns run.

    api_start hands the supervisor coroutine to asyncio.create_task and
    returns 202 immediately, so the start has to be given a scheduling pass in
    the SAME loop -- a second asyncio.run() would never touch it.
    """
    async def go():
        await capp.api_start(
            {"repo_id": "mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4",
             "backend": "flashnext", "util": 0.95, "ctx": 262144, "max_num_seqs": 1}
        )
        for _ in range(5):
            await asyncio.sleep(0)

    asyncio.run(go())


def test_apply_starts_the_selected_backend_on_its_own_port(tmp_path, config_path, monkeypatch):
    """The bug the owner hit, at the endpoint the button posts to.

    ``state/desired.json`` on this box holds GLM's ``port: 8002``. Picking a
    Flash-Next model in the dashboard posts ``{repo_id, backend, util, ctx,
    max_num_seqs}`` and NO port, so the start inherited 8002 -- Flash-Next
    booted on GLM's port while the gateway went on proxying 8001, and the
    dashboard reported the model as unreachable.
    """
    _box_config(tmp_path, config_path)
    capp, recorded = _capture_start(
        monkeypatch, tmp_path,
        dict(backend="glm53", port=8002, repo_id="dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4",
             served_name="glm53-flash", util=0.95, max_model_len=327680, max_num_seqs=1),
    )
    _drive(capp)
    assert recorded.get("port") == 8001, (
        f"started the Flash-Next backend on port {recorded.get('port')} -- "
        "the port the PREVIOUS backend was using"
    )


def test_apply_does_not_advertise_the_new_model_under_the_old_ones_name(
    tmp_path, config_path, monkeypatch
):
    """Same post, same omission: no ``served_name`` is sent either, so the
    start carried ``glm53-flash`` onto a Qwen checkpoint. That name is what
    ``/v1/models`` then advertises, and it is why nothing downstream could say
    which model was serving."""
    _box_config(tmp_path, config_path)
    capp, recorded = _capture_start(
        monkeypatch, tmp_path,
        dict(backend="glm53", port=8002, repo_id="dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4",
             served_name="glm53-flash", util=0.95, max_model_len=327680, max_num_seqs=1),
    )
    _drive(capp)
    assert recorded.get("served_name") != "glm53-flash", (
        "the new model would be served under the previous model's name"
    )
    assert recorded.get("repo_id") == "mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4"


# --------------------------------------------------------------------------- #
# Multimodal: the one launch setting that cannot travel in .config
# --------------------------------------------------------------------------- #
# The owner's requirement is "we want images and multimodal to be enabled", and
# images are controlled by a single vLLM flag, ``--limit-mm-per-prompt``. Every
# launcher on this box defaulted it to zero images, so a start served text-only
# with nothing anywhere to say so.
#
# It cannot travel the usual way. ``EXTRA_ARGS`` comes out of
# ``local_llm/.config``, and both readers of that file (``llm``'s own loop and
# ``shellconfig._KV_LINE_RE``) accept only ``KEY="value"`` with no double quote
# inside the value -- while vLLM accepts only strict JSON here, i.e. a value
# that is nothing but double quotes. So the launchers read ``MM_LIMIT_JSON``
# from the ENVIRONMENT, and a dashboard start has to put it there too.
#
# These tests are about THIS BOX, like
# ``test_the_real_llm_script_exports_nothing_servedeck_drops`` above: the
# question is whether the dashboard and the CLI start the same server here, and
# that cannot be asked of a fixture.

REPO_ROOT = Path(__file__).resolve().parents[1]
BOX_CONFIG = REPO_ROOT / "servedeck.toml"
EXAMPLE_CONFIG = REPO_ROOT / "servedeck.toml.example"
FLASHNEXT_SERVE_SH = Path("~/Projects/vllm-qwen38next/serve.sh").expanduser()


def _serve_sh_mm_default() -> str:
    """The ``--limit-mm-per-prompt`` value ``serve.sh`` compiles in.

    Read out of the script rather than repeated here, so this file cannot
    become a second source of truth that quietly goes stale — the exact failure
    the ``[backends.*.env]`` comment in servedeck.toml warns about. Matches the
    defaulting line itself (``if [ -z "${MM_LIMIT_JSON:-}" ]; then
    MM_LIMIT_JSON='...'``), never the several comment lines that also mention
    the variable.
    """
    text = FLASHNEXT_SERVE_SH.read_text()
    m = re.search(
        r"""\[ -z "\$\{MM_LIMIT_JSON:-\}" \][^\n]*?MM_LIMIT_JSON='([^']*)'""", text
    )
    assert m, "serve.sh no longer defaults MM_LIMIT_JSON — has its shape changed?"
    return m.group(1)


def _box_toml() -> dict:
    import tomllib

    return tomllib.loads(BOX_CONFIG.read_text())


_needs_box = pytest.mark.skipif(
    not BOX_CONFIG.is_file() or not FLASHNEXT_SERVE_SH.is_file(),
    reason="this machine has no servedeck.toml / vllm-qwen38next/serve.sh",
)


@_needs_box
def test_the_launch_contract_names_the_multimodal_limit():
    """``MM_LIMIT_JSON`` has to be declared, and declared as what the launcher
    itself defaults to.

    Not an override — a contract made explicit. The value is pinned to
    ``serve.sh``'s own default so the two can never drift apart silently: if
    somebody retunes the launcher, this fails rather than the dashboard quietly
    starting the old configuration.
    """
    env = _box_toml()["backends"]["flashnext"].get("env", {})
    assert "MM_LIMIT_JSON" in env, (
        "servedeck.toml's flashnext backend does not declare MM_LIMIT_JSON, so a "
        "dashboard start says nothing about images while the operator's own "
        "command line does"
    )
    assert env["MM_LIMIT_JSON"] == _serve_sh_mm_default()


@_needs_box
def test_a_dashboard_start_and_llm_start_agree_on_the_multimodal_limit(
    tmp_path, config_path, monkeypatch
):
    """End to end through ``_build_launch``, against this box's real config.

    ``llm start`` exports no ``MM_LIMIT_JSON``, so the value ``llm start``
    produces is the one ``serve.sh`` compiles in. A dashboard start must put
    the SAME string in the launcher's environment — anything else, including
    nothing at all, is a second configuration nobody chose.
    """
    config_path(BOX_CONFIG.read_text())
    monkeypatch.setattr(
        supervisor.shellconfig, "read_config", lambda: {"BACKEND": "flashnext"}
    )
    s = _supervisor(tmp_path)
    _argv, env, _cwd, _logs = _build(s, "flashnext")

    llm_start_value = _serve_sh_mm_default()
    assert env.get("MM_LIMIT_JSON") == llm_start_value, (
        f"a dashboard start passes MM_LIMIT_JSON={env.get('MM_LIMIT_JSON')!r}; "
        f"`llm start` resolves {llm_start_value!r}"
    )
    # And it is images-ON, not merely equal: two paths that are both wrong in
    # the same way match perfectly.
    assert '"image":0' not in llm_start_value.replace(" ", ""), (
        "both paths agree, but on IMAGES OFF — the requirement is images enabled"
    )


@_needs_box
def test_the_multimodal_limit_is_not_handed_to_a_backend_that_ignores_it(
    tmp_path, config_path, monkeypatch
):
    """Over-correction guard.

    ``vllm-glm53/serve-opt.sh`` hardcodes ``--limit-mm-per-prompt`` and reads
    no environment for it, so declaring ``MM_LIMIT_JSON`` on that backend would
    advertise a knob that does nothing — a launch contract that lies. GLM's
    checkpoint does have a vision tower, but turning images on there costs VRAM
    on the one backend whose usable context is already bounded by free VRAM,
    which is a measurement on the card, not a config edit.
    """
    config_path(BOX_CONFIG.read_text())
    monkeypatch.setattr(
        supervisor.shellconfig, "read_config", lambda: {"BACKEND": "glm53"}
    )
    s = _supervisor(tmp_path)
    _argv, env, _cwd, _logs = _build(s, "glm53", port=8002)
    assert "MM_LIMIT_JSON" not in env, env


@pytest.mark.skipif(
    not EXAMPLE_CONFIG.is_file(), reason="no servedeck.toml.example in this checkout"
)
def test_the_shipped_example_documents_the_multimodal_limit():
    """The example config is the only thing a fresh install reads.

    A launcher setting that (a) changes what the server can do and (b) cannot
    be expressed in the shell config has to be visible there, or the next
    machine repeats this bug from scratch.
    """
    text = EXAMPLE_CONFIG.read_text()
    assert "MM_LIMIT_JSON" in text, (
        "servedeck.toml.example never mentions MM_LIMIT_JSON, so nothing tells a "
        "new install that images are a launcher setting Servedeck must pass"
    )
    assert "limit-mm-per-prompt" in text, (
        "the example names the variable but not the flag it becomes, which is "
        "what a reader would search for"
    )
