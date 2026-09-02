"""Test-wide configuration isolation.

``config._config_file()`` picks up ``servedeck.toml`` next to the package, and
that file is gitignored precisely because it describes ONE machine. Without
this fixture the suite's results would depend on whether the developer running
it happens to have a config, and on what is in it — a test that passes on a
clean checkout and fails on the maintainer's box, or worse, the reverse.

So every test runs against one small, fixed configuration declared here. It
names the two backends this project grew up with, at the paths ``paths.py``
already spells out, so nothing that used to work without a config changes
meaning.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from servedeck import capacity, config, paths

_TOML = f"""
gpu_total_mib = 97887
overhead_gib = 4.7
frag_margin_mib = 4096

[backends.flashnext]
launcher = "{paths.SERVE_SH}"
port = 8001
log_path = "{paths.VLLM_QWEN38NEXT / 'serve.log'}"
venv = "{paths.VENV_NEXT_DIR}"
architectures = ["Qwen4ExpForConditionalGeneration"]
needs_tty = true

[backends.flashnext.env_map]
port = "PORT"
max_model_len = "MAX_LEN"
util = "GPU_UTIL"
max_num_seqs = "MAX_SEQS"
served_name = "SERVED_NAME"
kv_dtype = "KV_DTYPE"

[backends.inline]
launcher = "{paths.SERVER_RUN_SH}"
port = 8000
log_path = "{paths.QWEN_LOG_FILE}"
writes_own_log = true
venv = "{paths.VENV_LLM_DIR}"
architectures = ["Qwen3_5ForConditionalGeneration"]

[backends.inline.env_map]
"""


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path_factory: pytest.TempPathFactory):
    """Point every test at the fixed configuration above, not at the box's."""
    path = tmp_path_factory.mktemp("cfg") / "servedeck.toml"
    path.write_text(_TOML)
    previous = os.environ.get("SERVEDECK_CONFIG")
    os.environ["SERVEDECK_CONFIG"] = str(path)
    config.reset()
    capacity.refresh_limits()
    try:
        yield config.get()
    finally:
        if previous is None:
            os.environ.pop("SERVEDECK_CONFIG", None)
        else:
            os.environ["SERVEDECK_CONFIG"] = previous
        config.reset()
        capacity.refresh_limits()


@pytest.fixture
def config_path(tmp_path: Path):
    """Write a one-off config for a test that needs a different one.

    Usage: ``cfg = config_path('[backends.x]\\nlauncher="/bin/true"\\nport=1')``
    """

    def _write(body: str):
        path = tmp_path / "servedeck.toml"
        path.write_text(body)
        os.environ["SERVEDECK_CONFIG"] = str(path)
        config.reset()
        capacity.refresh_limits()
        return config.get()

    return _write
