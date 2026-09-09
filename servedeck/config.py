"""Configuration: every machine-specific value lives here, and only here.

Resolution order, first match wins:

1. ``SERVEDECK_*`` environment variables
2. ``servedeck.toml`` — next to the package, or at ``$SERVEDECK_CONFIG``
3. Auto-detection (GPU size from ``nvidia-smi``, model cache from ``$HF_HOME``)
4. Documented defaults

Nothing else in the package may hardcode a path, a GPU size, or a model name.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A GPU size is only used when detection fails. 0 means "unknown", which makes
# capacity refuse to guess rather than invent a budget.
FALLBACK_GPU_TOTAL_MIB = 0

# Fragmentation margin above the utilization budget. Below this, a config is
# reported as thin — never as impossible. (Adding it to the *requirement*
# makes any util >= ~0.958 unsatisfiable on a card whose util is a fraction of
# total, which is a bug we shipped once.)
DEFAULT_FRAG_MARGIN_MIB = 4096

# Activations + CUDA graphs, i.e. VRAM that is neither weights nor KV.
# Measured 4.3-4.5 GiB for two very different models; 4.7 errs high, which
# makes predictions conservative.
DEFAULT_OVERHEAD_GIB = 4.7


@dataclass(frozen=True)
class Backend:
    """One way of launching a model server.

    Servedeck never builds a ``vllm serve`` command line itself. It runs your
    launcher and passes settings through the environment, so the launcher stays
    the single source of truth for flags.
    """

    name: str
    launcher: Path
    port: int
    #: The log this backend is conventionally started into by hand. Used when
    #: ADOPTING a server Servedeck did not launch, and when reading the boot
    #: numbers of one. None means "this launcher has no fixed log": Servedeck
    #: then falls back to the newest log it opened itself, under
    #: ``state/boot_logs/<name>-*.log``. Naming another backend's log here is
    #: worse than naming none — the phase machine would classify a different
    #: model's error lines as this run's failure.
    log_path: Path | None = None
    #: True when the launcher redirects its own output into ``log_path``
    #: (``exec >> "$LOG"``). Then a launch has to tail that file too, because
    #: the boot output stops arriving on the pipe Servedeck opened. False when
    #: ``log_path`` is only a hand-launch convention: tailing it during a fresh
    #: launch would replay a previous boot into the phase machine.
    writes_own_log: bool = False
    venv: Path | None = None
    # Environment variable names the launcher reads, so Servedeck can pass
    # settings without knowing the flags.
    env_map: dict[str, str] = field(
        default_factory=lambda: {
            "repo_id": "MODEL",
            "port": "PORT",
            "max_model_len": "MAX_LEN",
            "util": "GPU_UTIL",
            "max_num_seqs": "MAX_SEQS",
            "served_name": "SERVED_NAME",
        }
    )
    #: Fixed environment passed to the launcher verbatim on every start —
    #: tuning knobs that belong to your machine, not to Servedeck. Servedeck
    #: never interprets these; the launcher owns their meaning.
    env: dict[str, str] = field(default_factory=dict)
    # Architectures this backend can serve, from config.json's `architectures`.
    architectures: tuple[str, ...] = ()
    # True if starting it needs a terminal (e.g. an interactive sudo prompt),
    # which makes unattended restart impossible. Servedeck reports
    # blocked-needs-human rather than looping.
    needs_tty: bool = False
    #: Working directory for the launcher. Defaults to the directory the
    #: launcher lives in — which is what the shell CLI on this box does
    #: (``cd "$(dirname "$SERVE_SH")"``) and is right for a serve script that
    #: sits at a project root. It is NOT right for a launcher under ``bin/``:
    #: that directory is not the tree the server belongs to, and any relative
    #: path the launcher uses resolves one level down from where it means to.
    #:
    #: Whether that matters is a property of the launcher, which is why this is
    #: declared rather than derived. ``local_llm/bin/qwen-server-run.sh``, for
    #: one, derives its own root from ``${BASH_SOURCE[0]}/..`` and so does not
    #: care — setting ``cwd`` for it changes nothing today. A launcher that
    #: does care has nowhere else to say so.
    cwd: Path | None = None
    #: Cold-boot ETA envelope (low, high) in seconds, used only until this
    #: model has produced measured boot history of its own. A model family's
    #: cold boot is a fact about a checkpoint on one machine, so it is
    #: configuration, not a constant in the package.
    cold_boot_range_s: tuple[float, float] | None = None

    @property
    def run_cwd(self) -> Path:
        """Where to run the launcher from."""
        return self.cwd if self.cwd is not None else self.launcher.parent


@dataclass(frozen=True)
class Config:
    state_dir: Path
    model_cache: Path
    backends: tuple[Backend, ...]
    gpu_total_mib: int
    overhead_gib: float = DEFAULT_OVERHEAD_GIB
    frag_margin_mib: int = DEFAULT_FRAG_MARGIN_MIB
    listen_host: str = "127.0.0.1"
    listen_port: int = 8010
    # Optional: a shell script Servedeck shells out to for config writes, so a
    # CLI and the GUI cannot drift. None = Servedeck owns its own state only.
    shell_config_script: Path | None = None
    #: If any of these paths exists, something else wants the GPU and
    #: Servedeck stands down instead of starting a server.
    training_markers: tuple[str, ...] = ()

    def backend(self, name: str | None) -> Backend | None:
        return next((b for b in self.backends if b.name == name), None)

    def backend_for_arch(self, arch: str | None) -> Backend | None:
        if not arch:
            return None
        return next((b for b in self.backends if arch in b.architectures), None)

    @property
    def gpu_total_gib(self) -> float:
        return self.gpu_total_mib / 1024


def detect_gpu_total_mib() -> int:
    """Total VRAM in MiB, or 0 if it cannot be determined.

    Returning 0 rather than a guess is deliberate: capacity treats unknown
    total as "cannot compute" and blocks, which is safer than sizing a KV
    cache against a number nobody measured.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return FALLBACK_GPU_TOTAL_MIB
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if out.returncode != 0:
            return FALLBACK_GPU_TOTAL_MIB
        first = out.stdout.strip().splitlines()[0]
        return int(first.strip())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return FALLBACK_GPU_TOTAL_MIB


def default_model_cache() -> Path:
    if v := os.environ.get("SERVEDECK_MODEL_CACHE"):
        return Path(v).expanduser()
    if v := os.environ.get("HF_HUB_CACHE"):
        return Path(v).expanduser()
    if v := os.environ.get("HF_HOME"):
        return Path(v).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _config_file() -> Path | None:
    if v := os.environ.get("SERVEDECK_CONFIG"):
        p = Path(v).expanduser()
        return p if p.is_file() else None
    for candidate in (
        Path.cwd() / "servedeck.toml",
        Path(__file__).resolve().parent.parent / "servedeck.toml",
    ):
        if candidate.is_file():
            return candidate
    return None


def _parse_backend(name: str, raw: dict[str, Any]) -> Backend:
    missing = [k for k in ("launcher", "port") if k not in raw]
    if missing:
        raise ValueError(f"backend '{name}' is missing required key(s): {', '.join(missing)}")
    launcher = Path(str(raw["launcher"])).expanduser()
    port = int(raw["port"])
    # No invented default: a guessed path (launcher.parent/<name>.log) reads
    # as a real answer, and an adopted server's boot facts would be parsed out
    # of a file nothing writes. Absent means absent.
    log_path = Path(str(raw["log_path"])).expanduser() if raw.get("log_path") else None
    return Backend(
        name=name,
        launcher=launcher,
        port=port,
        log_path=log_path,
        writes_own_log=bool(raw.get("writes_own_log", False)),
        venv=Path(str(raw["venv"])).expanduser() if raw.get("venv") else None,
        # `in raw`, not truthiness: an EMPTY [backends.x.env_map] table is a
        # deliberate "this launcher reads no environment" (it takes its
        # settings from a file), and falling back to the default map there
        # would hand it settings it never asked for.
        env_map=(
            {str(k): str(v) for k, v in (raw["env_map"] or {}).items()}
            if "env_map" in raw
            else Backend.__dataclass_fields__["env_map"].default_factory()  # type: ignore[misc]
        ),
        env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
        architectures=tuple(raw.get("architectures", ())),
        needs_tty=bool(raw.get("needs_tty", False)),
        cwd=Path(str(raw["cwd"])).expanduser() if raw.get("cwd") else None,
        cold_boot_range_s=(
            (float(raw["cold_boot_range_s"][0]), float(raw["cold_boot_range_s"][1]))
            if raw.get("cold_boot_range_s")
            else None
        ),
    )


def load(path: Path | None = None) -> Config:
    """Build the effective configuration."""
    raw: dict[str, Any] = {}
    src = path or _config_file()
    if src is not None:
        with open(src, "rb") as fh:
            raw = tomllib.load(fh)

    state_dir = Path(
        os.environ.get("SERVEDECK_STATE_DIR")
        or raw.get("state_dir")
        or (Path(__file__).resolve().parent.parent / "state")
    ).expanduser()

    backends = tuple(
        _parse_backend(name, spec) for name, spec in (raw.get("backends") or {}).items()
    )

    gpu_total = int(os.environ.get("SERVEDECK_GPU_TOTAL_MIB") or raw.get("gpu_total_mib") or 0)
    if not gpu_total:
        gpu_total = detect_gpu_total_mib()

    shell_script = raw.get("shell_config_script")
    return Config(
        state_dir=state_dir,
        model_cache=Path(raw["model_cache"]).expanduser() if raw.get("model_cache") else default_model_cache(),
        backends=backends,
        gpu_total_mib=gpu_total,
        overhead_gib=float(raw.get("overhead_gib", DEFAULT_OVERHEAD_GIB)),
        frag_margin_mib=int(raw.get("frag_margin_mib", DEFAULT_FRAG_MARGIN_MIB)),
        listen_host=str(os.environ.get("SERVEDECK_HOST") or raw.get("listen_host", "127.0.0.1")),
        listen_port=int(os.environ.get("SERVEDECK_PORT") or raw.get("listen_port", 8010)),
        shell_config_script=Path(shell_script).expanduser() if shell_script else None,
        training_markers=tuple(
            str(Path(str(m)).expanduser()) for m in raw.get("training_markers", ())
        ),
    )


_cached: Config | None = None


def get() -> Config:
    """The process-wide configuration (loaded once)."""
    global _cached
    if _cached is None:
        _cached = load()
    return _cached


def reset() -> None:
    """Drop the cache. For tests."""
    global _cached
    _cached = None
