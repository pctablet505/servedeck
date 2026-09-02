"""servedeck.registry — model discovery, servability, and the observation store.

Implements SPEC.md §4 (REGISTRY):
  * Discovery over ``~/.cache/huggingface/hub/models--*/``.
  * Servability rules and ``KNOWN_ARCHS`` (which also selects backend/launcher).
  * The append-only ``state/measurements.json`` observation store.
  * ``resolve_inputs(repo_id, util, ctx)`` and its three-tier lookup.

This module is pure discovery + bookkeeping: it does not launch anything, does not
compute VRAM budgets (that is servedeck/capacity.py's job — see SPEC §3), and never
fabricates a number that was not either read from disk or explicitly given in
SPEC §0's ground-truth table.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from . import config as _config

# --------------------------------------------------------------------------- #
# Constants (SPEC §4)
# --------------------------------------------------------------------------- #

#: architectures[0] -> backend. This map is also how backend/launcher/venv is
#: selected for a servable model (SPEC §4, §1 "Delegation, not reimplementation").
KNOWN_ARCHS: dict[str, str] = {
    "Qwen3_5ForConditionalGeneration": "inline",
    "Qwen4ExpForConditionalGeneration": "flashnext",
}

def arch_backends() -> dict[str, str]:
    """architectures[0] -> backend, with servedeck.toml layered over the
    built-in map.

    Declaring `architectures` on a `[backends.<name>]` section is the whole
    supported way to teach Servedeck a new model family — no code change, and
    no architecture name belonging to one person's box baked into a public
    package. KNOWN_ARCHS remains the fallback for an install with no config.
    """
    merged = dict(KNOWN_ARCHS)
    try:
        backends = _config.get().backends
    except Exception:  # noqa: BLE001 - an unreadable config must not hide models
        return merged
    for b in backends:
        for arch in b.architectures:
            merged[arch] = b.name
    return merged


def _reverse_arch_backends() -> dict[str, str]:
    """Reverse of :func:`arch_backends`. Valid only where the map is 1:1 (each
    backend has exactly one known architecture). Used by the tier-3 KV-rate
    family estimate to classify *historical* observations by architecture even
    if their repo has since been deleted from the hub cache (backend survives
    in the observation; the architecture does not). A backend declaring several
    architectures simply has no single reverse answer — last one wins, and the
    estimate degrades to "no measured observation", never to a wrong family.
    """
    return {backend: arch for arch, backend in arch_backends().items()}

GIB = 1024**3

#: Default kv_cache_dtype used by the launchers when nothing overrides it
#: (SPEC §1: "flashnext: ... KV_DTYPE=auto ... serve.sh ALREADY reads ... KV_DTYPE from env").
DEFAULT_KV_CACHE_DTYPE = "auto"

WeightsSource = Literal["measured", "estimated", "unknown"]
KvSource = Literal["measured", "measured_other_ctx", "estimated", "unknown"]
Trust = Literal["measured", "measured_other_ctx", "estimated", "unknown"]

_CONFIG_FIELD_NAMES = (
    "architectures0",
    "model_type",
    "max_position_embeddings",
    "num_hidden_layers",
    "num_key_value_heads",
    "head_dim",
    "full_attention_interval",
    "quant_algo",
)


# --------------------------------------------------------------------------- #
# Hub cache location
# --------------------------------------------------------------------------- #


def default_hub_dir() -> Path:
    """~/.cache/huggingface/hub, overridable via SERVEDECK_HF_HUB_DIR (tests only)."""
    override = os.environ.get("SERVEDECK_HF_HUB_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "huggingface" / "hub"


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


@dataclass
class ModelEntry:
    """One ``models--*`` hub directory, resolved and (if possible) parsed."""

    repo_id: str
    hub_dirname: str
    snapshot_path: str | None
    skipped: bool
    servable: bool
    backend: str | None
    safetensors_gib: float
    safetensors_count: int
    config_exists: bool
    architectures0: str | None
    model_type: str | None
    max_position_embeddings: int | None
    num_hidden_layers: int | None
    num_key_value_heads: int | None
    head_dim: int | None
    full_attention_interval: int | None
    quant_algo: str | None
    reason: str | None


def _repo_id_from_dirname(dirname: str) -> str:
    """``models--Org--Name`` -> ``Org/Name``.

    Only the separator between namespace and repo name is a hub convention
    ("--"); repo names themselves use single hyphens (e.g. "Qwen3.8-27B-NVFP4"),
    so only the *first* "--" is replaced.
    """
    if not dirname.startswith("models--"):
        raise ValueError(f"not a models-- directory: {dirname!r}")
    rest = dirname[len("models--") :]
    return rest.replace("--", "/", 1)


def _resolve_snapshot(hub_repo_dir: Path) -> Path | None:
    """snapshot = refs/main content else newest snapshots/ dir; None if no snapshots/."""
    refs_main = hub_repo_dir / "refs" / "main"
    if refs_main.is_file():
        try:
            rev = refs_main.read_text().strip()
        except OSError:
            rev = ""
        if rev:
            candidate = hub_repo_dir / "snapshots" / rev
            if candidate.is_dir():
                return candidate

    snapshots_dir = hub_repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    try:
        candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    except OSError:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _scan_snapshot_files(snapshot: Path) -> tuple[float, int, int]:
    """(safetensors_gib, safetensors_count, gguf_count) for one snapshot dir.

    Sizes follow symlinks (hub snapshot files are symlinks into blobs/); this
    matches ``Path.stat()``'s default (follow_symlinks=True), per SPEC §4.
    """
    total_bytes = 0
    st_count = 0
    gguf_count = 0
    try:
        children = list(snapshot.iterdir())
    except OSError:
        return 0.0, 0, 0
    for p in children:
        name = p.name
        if name.endswith(".safetensors"):
            try:
                total_bytes += p.stat().st_size
            except OSError:
                continue
            st_count += 1
        elif name.endswith(".gguf"):
            gguf_count += 1
    return total_bytes / GIB, st_count, gguf_count


def _parse_config(snapshot: Path) -> dict[str, Any] | None:
    cfg_path = snapshot / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        return json.loads(cfg_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _extract_config_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    """architectures[0], model_type from root; the rest from text_config, falling
    back to root when text_config is absent (SPEC §4)."""
    text_config = cfg.get("text_config") or cfg
    quant = cfg.get("quantization_config") or {}
    architectures = cfg.get("architectures") or []
    return {
        "architectures0": architectures[0] if architectures else None,
        "model_type": cfg.get("model_type"),
        "max_position_embeddings": text_config.get("max_position_embeddings"),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "num_key_value_heads": text_config.get("num_key_value_heads"),
        "head_dim": text_config.get("head_dim"),
        "full_attention_interval": text_config.get("full_attention_interval"),
        "quant_algo": quant.get("quant_algo") or quant.get("quant_method"),
    }


def _build_entry(repo_id: str, hub_dirname: str, snapshot: Path | None) -> ModelEntry:
    if snapshot is None:
        return ModelEntry(
            repo_id=repo_id,
            hub_dirname=hub_dirname,
            snapshot_path=None,
            skipped=True,
            servable=False,
            backend=None,
            safetensors_gib=0.0,
            safetensors_count=0,
            config_exists=False,
            **{k: None for k in _CONFIG_FIELD_NAMES},
            reason="skipped: no snapshots/ (stub dir)",
        )

    safetensors_gib, st_count, gguf_count = _scan_snapshot_files(snapshot)
    cfg = _parse_config(snapshot)
    config_exists = cfg is not None
    fields = _extract_config_fields(cfg) if cfg is not None else {k: None for k in _CONFIG_FIELD_NAMES}
    known_archs = arch_backends()
    backend = known_archs.get(fields["architectures0"]) if fields["architectures0"] else None

    reasons: list[str] = []
    if safetensors_gib <= 0:
        reasons.append("0 safetensors")
    if not config_exists:
        reasons.append("no config.json")
    elif fields["architectures0"] not in known_archs:
        reasons.append(f"unknown architecture {fields['architectures0']!r}")

    servable = not reasons
    reason: str | None = None
    if reasons:
        prefix = "GGUF-only: " if gguf_count > 0 and safetensors_gib <= 0 else ""
        reason = prefix + ", ".join(reasons)

    return ModelEntry(
        repo_id=repo_id,
        hub_dirname=hub_dirname,
        snapshot_path=str(snapshot),
        skipped=False,
        servable=servable,
        backend=backend,
        # Not rounded: this is a raw measurement (SPEC §0 "measured, do not
        # substitute"). round(x, 4) has a resolution of ~105 KiB at GiB scale
        # (1024**3 * 0.0001), which is invisible for real multi-GB safetensors
        # shards but silently truncates small values to 0.0 — display-layer
        # rounding belongs in the CLI formatter (_fmt_gib), not in the
        # measurement itself.
        safetensors_gib=safetensors_gib,
        safetensors_count=st_count,
        config_exists=config_exists,
        reason=reason,
        **fields,
    )


def discover_models(hub_dir: Path | str | None = None) -> list[ModelEntry]:
    """Scan ``models--*`` directories under ``hub_dir`` (default: the real hub cache).

    Returns one ModelEntry per hub directory, including skipped stubs — callers
    that only want launchable models should filter on ``.servable``.
    """
    root = Path(hub_dir) if hub_dir is not None else default_hub_dir()
    if not root.is_dir():
        return []
    entries: list[ModelEntry] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.startswith("models--"):
            continue
        try:
            repo_id = _repo_id_from_dirname(child.name)
        except ValueError:
            continue
        snapshot = _resolve_snapshot(child)
        entries.append(_build_entry(repo_id, child.name, snapshot))
    return entries


# --------------------------------------------------------------------------- #
# Observation store — state/measurements.json (append-only)
# --------------------------------------------------------------------------- #


def default_measurements_path() -> Path:
    return Path(__file__).resolve().parent.parent / "state" / "measurements.json"


def load_observations(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Read the observation list. Missing/empty/corrupt file -> []; never raises."""
    p = Path(path) if path is not None else default_measurements_path()
    if not p.is_file():
        return []
    try:
        raw = p.read_text()
    except OSError:
        return []
    if not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return data


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".measurements-", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=False)
            fh.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def append_observation(observation: dict[str, Any], path: Path | str | None = None) -> None:
    """Append one observation. Never overwrites prior entries (append-only log,
    SPEC §4). Written after every boot reaching READY, and after a boot that
    fails at KV cache sizing (which still yields a weights measurement)."""
    p = Path(path) if path is not None else default_measurements_path()
    observations = load_observations(p)
    observations.append(observation)
    _atomic_write_json(p, observations)


def _kv_rate_of(obs: dict[str, Any]) -> float | None:
    """kv_kib_per_token for one observation: derived from kv_gib/kv_tokens when
    both are present (SPEC §4 "Derived, not stored twice"); otherwise falls back
    to a directly-stored rate (used by observations where only the rate itself
    is known, e.g. a secondary calibration boot at a different ctx)."""
    measured = obs.get("measured") or {}
    kv_gib = measured.get("kv_gib")
    kv_tokens = measured.get("kv_tokens")
    if kv_gib is not None and kv_tokens:
        try:
            return float(kv_gib) * 1048576.0 / float(kv_tokens)
        except (TypeError, ZeroDivisionError, ValueError):
            pass
    rate = measured.get("kv_kib_per_token")
    return float(rate) if rate is not None else None


# --------------------------------------------------------------------------- #
# resolve_inputs() — three-tier lookup (SPEC §4)
# --------------------------------------------------------------------------- #


@dataclass
class ResolvedInputs:
    """What resolve_inputs() found, and how much to trust it.

    Consumed by servedeck/capacity.py (not owned by this module) to build its
    own ModelInputs and raise UNKNOWN_CAPACITY when weights_source=="unknown".
    """

    repo_id: str
    backend: str | None
    requested_util: float
    requested_ctx: int
    weights_gib: float | None
    weights_source: WeightsSource
    kv_kib_per_token: float | None
    kv_source: KvSource
    trust: Trust
    overhead_gib: float | None
    model_max_ctx: int | None
    model_type: str | None
    architectures0: str | None
    matched_ctx: int | None
    other_ctx_kv_rates: dict[int, float] = field(default_factory=dict)
    reason: str | None = None


def _estimate_weights_gib(
    safetensors_gib: float | None, model_type: str | None
) -> tuple[float | None, WeightsSource, str | None]:
    """Tier-3 weight estimate: safetensors_gib*1.01 — refused for qwen4_exp.

    Refusal reason (SPEC §3/§4): the Flash-Next n-gram (MTP) table is
    host-offloaded, so its on-disk safetensors size overstates the actual VRAM
    footprint by ~37% (125.91 GiB disk vs 78.47 GiB VRAM, measured). Applying
    the *1.01 correction to that number would still be wildly wrong, so this
    tier refuses outright rather than emit a bad estimate.
    """
    if safetensors_gib is None or safetensors_gib <= 0:
        return None, "unknown", "no safetensors size available for this repo_id"
    if model_type == "qwen4_exp":
        return (
            None,
            "unknown",
            "safetensors*1.01 estimator refused for model_type='qwen4_exp': the "
            "n-gram (MTP) table is host-offloaded, so on-disk size overstates VRAM "
            "weights by ~37% (measured: 125.91 GiB disk vs 78.47 GiB VRAM on "
            "RadixArk/Qwen3.8-Flash-Next-NVFP4)",
        )
    if model_type == "glm5_next":
        return (
            None,
            "unknown",
            "safetensors*1.01 estimator refused for model_type='glm5_next': the "
            "routed experts are host-offloaded via --cpu-offload-gb, so VRAM "
            "weights are on-disk size MINUS whatever that knob is set to, and "
            "the knob is a tuning choice rather than a model property (measured: "
            "181.3 GiB disk, ~79 GiB VRAM at CPU_OFFLOAD_GB=104). Estimating "
            "from disk size produced 183.11 GiB and a nonsensical negative KV "
            "budget of -97 GiB",
        )
    return round(safetensors_gib * 1.01, 4), "estimated", None


def _estimate_kv_rate(
    architectures0: str | None, all_observations: list[dict[str, Any]]
) -> tuple[float | None, KvSource, str | None]:
    """Tier-3 KV rate estimate: median kv_kib_per_token of *measured* observations
    sharing this architecture (SPEC §4)."""
    if not architectures0:
        return None, "unknown", "unknown architecture; cannot estimate KV rate by family"
    rates: list[float] = []
    reverse_archs = _reverse_arch_backends()
    for obs in all_observations:
        if obs.get("trust") != "measured":
            continue
        arch = reverse_archs.get(obs.get("backend"))
        if arch != architectures0:
            continue
        rate = _kv_rate_of(obs)
        if rate is not None:
            rates.append(rate)
    if not rates:
        return None, "unknown", f"no measured KV-cache observation exists for architecture {architectures0!r}"
    return round(statistics.median(rates), 4), "estimated", None


def resolve_inputs(
    repo_id: str,
    util: float,
    ctx: int,
    *,
    observations: list[dict[str, Any]] | None = None,
    hub_dir: Path | str | None = None,
) -> ResolvedInputs:
    """Three-tier lookup (SPEC §4):

      1. exact (repo_id, backend, max_model_len==ctx)   -> trust="measured"
      2. same repo_id, any ctx                           -> trust="measured_other_ctx"
      3. estimate: weights = safetensors_gib*1.01 (refused for qwen4_exp);
         kv rate = median of measured observations sharing architectures[0]

    ``observations`` and ``hub_dir`` are injectable for tests; production
    callers pass neither and get the real measurements store / hub cache.
    """
    all_observations = load_observations() if observations is None else observations
    registry_entries = discover_models(hub_dir)
    registry_entry = next((e for e in registry_entries if e.repo_id == repo_id), None)

    backend = registry_entry.backend if registry_entry else None
    model_max_ctx = registry_entry.max_position_embeddings if registry_entry else None
    model_type = registry_entry.model_type if registry_entry else None
    architectures0 = registry_entry.architectures0 if registry_entry else None

    repo_obs = [o for o in all_observations if o.get("repo_id") == repo_id]
    if backend is None and repo_obs:
        # repo_id no longer present in the hub cache (e.g. deleted) — fall back
        # to whatever backend its own observation history recorded.
        backend = repo_obs[0].get("backend")
    if backend is not None:
        by_backend = [o for o in repo_obs if o.get("backend") == backend]
        repo_obs = by_backend or repo_obs

    other_ctx_kv_rates: dict[int, float] = {}
    for obs in repo_obs:
        obs_ctx = (obs.get("inputs") or {}).get("max_model_len")
        rate = _kv_rate_of(obs)
        if obs_ctx is not None and rate is not None:
            other_ctx_kv_rates[int(obs_ctx)] = rate

    # Tier 1: exact ctx match.
    exact = [o for o in repo_obs if (o.get("inputs") or {}).get("max_model_len") == ctx]
    if exact:
        obs = exact[-1]  # most recently appended wins
        measured = obs.get("measured") or {}
        weights_gib = measured.get("weights_gib")
        kv_rate = _kv_rate_of(obs)
        return ResolvedInputs(
            repo_id=repo_id,
            backend=backend,
            requested_util=util,
            requested_ctx=ctx,
            weights_gib=weights_gib,
            weights_source="measured" if weights_gib is not None else "unknown",
            kv_kib_per_token=kv_rate,
            kv_source="measured" if kv_rate is not None else "unknown",
            trust="measured",
            overhead_gib=measured.get("overhead_gib"),
            model_max_ctx=model_max_ctx,
            model_type=model_type,
            architectures0=architectures0,
            matched_ctx=ctx,
            other_ctx_kv_rates=other_ctx_kv_rates,
        )

    # Tier 2: same repo_id, any ctx.
    if repo_obs:
        obs = repo_obs[-1]
        measured = obs.get("measured") or {}
        weights_gib = measured.get("weights_gib")
        kv_rate = _kv_rate_of(obs)
        matched_ctx = (obs.get("inputs") or {}).get("max_model_len")
        return ResolvedInputs(
            repo_id=repo_id,
            backend=backend,
            requested_util=util,
            requested_ctx=ctx,
            weights_gib=weights_gib,
            weights_source="measured" if weights_gib is not None else "unknown",
            kv_kib_per_token=kv_rate,
            kv_source="measured_other_ctx" if kv_rate is not None else "unknown",
            trust="measured_other_ctx",
            overhead_gib=measured.get("overhead_gib"),
            model_max_ctx=model_max_ctx,
            model_type=model_type,
            architectures0=architectures0,
            matched_ctx=int(matched_ctx) if matched_ctx is not None else None,
            other_ctx_kv_rates=other_ctx_kv_rates,
        )

    # Tier 3: estimate.
    safetensors_gib = registry_entry.safetensors_gib if registry_entry else None
    weights_gib, weights_source, w_reason = _estimate_weights_gib(safetensors_gib, model_type)
    kv_rate, kv_source, kv_reason = _estimate_kv_rate(architectures0, all_observations)

    if weights_source == "unknown":
        trust: Trust = "unknown"
    elif weights_source == "estimated" or kv_source == "estimated":
        trust = "estimated"
    else:
        trust = "unknown"

    return ResolvedInputs(
        repo_id=repo_id,
        backend=backend,
        requested_util=util,
        requested_ctx=ctx,
        weights_gib=weights_gib,
        weights_source=weights_source,
        kv_kib_per_token=kv_rate,
        kv_source=kv_source,
        trust=trust,
        overhead_gib=None,
        model_max_ctx=model_max_ctx,
        model_type=model_type,
        architectures0=architectures0,
        matched_ctx=None,
        other_ctx_kv_rates=other_ctx_kv_rates,
        reason=w_reason or kv_reason,
    )


# --------------------------------------------------------------------------- #
# CLI — `python -m servedeck.registry --list`
# --------------------------------------------------------------------------- #


def _fmt_gib(x: float | None) -> str:
    return f"{x:.2f}" if x is not None else "—"


def _format_listing(entries: list[ModelEntry]) -> str:
    skipped = [e for e in entries if e.skipped]
    unservable = [e for e in entries if not e.skipped and not e.servable]
    servable = [e for e in entries if e.servable]

    lines = [
        f"Scanned {len(entries)} hub dir(s): "
        f"{len(servable)} servable, {len(unservable)} unservable, {len(skipped)} skipped",
        "",
    ]
    for e in entries:
        status = "SKIPPED" if e.skipped else ("SERVABLE" if e.servable else "UNSERVABLE")
        backend = e.backend or "-"
        arch = e.architectures0 or "-"
        lines.append(f"[{status:10}] {e.repo_id}")
        lines.append(
            f"             backend={backend:9} "
            f"safetensors={_fmt_gib(e.safetensors_gib)} GiB (n={e.safetensors_count}) "
            f"arch={arch}"
        )
        if e.reason:
            lines.append(f"             reason: {e.reason}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m servedeck.registry")
    parser.add_argument("--list", action="store_true", help="list discovered hub models")
    args = parser.parse_args(argv)

    if not args.list:
        parser.print_help()
        return 0

    entries = discover_models()
    print(_format_listing(entries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
