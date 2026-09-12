"""servedeck.models — the v2 model registry (REDESIGN-2026-09-12.md §2.1, §2.4, §2.7).

``models.toml`` is the single source of truth for every model this box can serve.
This module loads it, validates it, and turns one ``Model`` entry into the argv/env
of a ``vllm serve`` invocation — nothing here starts a process (that is P3/P4's
supervisor) and nothing here writes a client config (that is ``servedeck.wire``).

Design notes, so a later packet does not have to re-derive them:

* ``load()`` merges ``[defaults.env]`` into every model's own ``env`` at LOAD time
  (the model's own keys win on conflict), so ``render_env()`` only needs the model
  — no registry-level merge step downstream.
* ``render_argv()`` takes ``util``/``ctx_tokens``/``port`` as already-resolved
  values, not the registry or the hub cache: it never does I/O, which is what makes
  the golden test (``tests/test_models_golden.py``) able to compare it byte-for-byte
  against a legacy launcher's dry-run output. Resolving ``ctx="native"`` to an int is
  a separate step (``native_ctx()``), because that DOES need the hub cache.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import discovery as _discovery

__all__ = [
    "RegistryError",
    "Reasoning",
    "Tools",
    "Model",
    "GPU",
    "Registry",
    "Resolved",
    "load",
    "render_argv",
    "render_env",
    "main_util",
    "resident_util",
    "native_ctx",
]

#: Ports that belong to the gateway (:8010) and to the unrelated ats-optimizer.service
#: (:8000) — REDESIGN-2026-09-12.md §2.1 "ports unique and never 8000 or 8010".
RESERVED_PORTS: frozenset[int] = frozenset({8000, 8010})


class RegistryError(Exception):
    """Raised by :func:`load` (or a validator called directly) for anything wrong
    with ``models.toml``: a precise, one-line-per-problem message, never a bare
    KeyError/TypeError from a malformed table."""


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Reasoning:
    """A model's reasoning (thinking) configuration."""

    parser: str
    #: When true, the gateway (not this module) copies `reasoning` into
    #: `reasoning_content` for chat-completions clients — REDESIGN §2.3.
    mirror_content: bool = False


@dataclass(frozen=True)
class Tools:
    """A model's tool-call parser configuration."""

    parser: str


@dataclass(frozen=True)
class Model:
    """One ``[models.<key>]`` table, fully resolved (env already merged with
    ``[defaults.env]``; everything else exactly as written in ``models.toml``)."""

    key: str
    id: str
    repo: str
    slot: str  # "main" | "resident"
    port: int
    build: str
    ctx: int | str  # an int, or the literal string "native"
    aliases: tuple[str, ...] = ()
    reasoning: Reasoning | None = None
    tools: Tools | None = None
    flags: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    presets: dict[str, dict[str, Any]] = field(default_factory=dict)
    vram_mib: int | None = None  # resident slot only
    min_output_tokens: int | None = None
    max_output_tokens: int | None = None
    vision: bool = False
    needs_tty: bool = False

    def served_names(self) -> list[str]:
        """``[id, *aliases, *presets]`` — REDESIGN §2.4/§2.1: every name a client
        has ever used for this model, in the order they should appear on the vLLM
        command line and in ``GET /v1/models``."""
        return [self.id, *self.aliases, *self.presets.keys()]


@dataclass(frozen=True)
class GPU:
    total_mib: int
    margin_mib: int


@dataclass(frozen=True)
class Registry:
    """The loaded, validated ``models.toml``."""

    models: dict[str, Model]  # TOML key -> Model, insertion order preserved
    gpu: GPU
    builds: dict[str, str]
    defaults_env: dict[str, str]

    def resolve(self, name: str) -> Resolved | None:
        """Resolve ``name`` as a model id, alias, preset name, or registry key.

        Returns ``None`` when nothing served under this box answers to ``name`` —
        never raises, so callers (the gateway's model routing, ``doctor``) can
        treat "unknown model" as ordinary control flow.
        """
        for m in self.models.values():
            if name == m.key:
                return Resolved(model=m, preset=None, overlay={})
            if name == m.id or name in m.aliases:
                return Resolved(model=m, preset=None, overlay={})
            if name in m.presets:
                return Resolved(model=m, preset=name, overlay=dict(m.presets[name]))
        return None


@dataclass(frozen=True)
class Resolved:
    """What :meth:`Registry.resolve` found: the underlying model, the preset name
    used to reach it (``None`` for a plain id/alias/key lookup), and the request
    overlay that preset carries (``{}`` when there is none)."""

    model: Model
    preset: str | None
    overlay: dict[str, Any]


# --------------------------------------------------------------------------- #
# Loading + validation
# --------------------------------------------------------------------------- #


def _as_str_tuple(value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise RegistryError(f"{where} must be a list of strings, got {value!r}")
    return tuple(value)


def _load_model(key: str, raw: dict[str, Any], defaults_env: dict[str, str]) -> Model:
    for required in ("id", "repo", "slot", "port", "build", "ctx"):
        if required not in raw:
            raise RegistryError(f"models.{key}: missing required field {required!r}")

    reasoning_raw = raw.get("reasoning")
    reasoning = None
    if reasoning_raw is not None:
        if "parser" not in reasoning_raw:
            raise RegistryError(f"models.{key}.reasoning: missing 'parser'")
        reasoning = Reasoning(
            parser=reasoning_raw["parser"],
            mirror_content=bool(reasoning_raw.get("mirror_content", False)),
        )

    tools_raw = raw.get("tools")
    tools = None
    if tools_raw is not None:
        if "parser" not in tools_raw:
            raise RegistryError(f"models.{key}.tools: missing 'parser'")
        tools = Tools(parser=tools_raw["parser"])

    presets_raw = raw.get("presets") or {}
    if not isinstance(presets_raw, dict):
        raise RegistryError(f"models.{key}.presets must be a table")
    presets: dict[str, dict[str, Any]] = {}
    for pname, overlay in presets_raw.items():
        if not isinstance(overlay, dict):
            raise RegistryError(f"models.{key}.presets.{pname} must be a table")
        presets[pname] = dict(overlay)

    merged_env: dict[str, str] = dict(defaults_env)
    merged_env.update(raw.get("env") or {})

    slot = raw["slot"]
    if slot not in ("main", "resident"):
        raise RegistryError(f"models.{key}.slot must be 'main' or 'resident', got {slot!r}")

    vram_mib = raw.get("vram_mib")
    if slot == "resident" and vram_mib is None:
        raise RegistryError(f"models.{key}: slot='resident' requires vram_mib")

    ctx = raw["ctx"]
    if not (ctx == "native" or isinstance(ctx, int)):
        raise RegistryError(f"models.{key}.ctx must be an int or 'native', got {ctx!r}")

    return Model(
        key=key,
        id=raw["id"],
        repo=raw["repo"],
        slot=slot,
        port=int(raw["port"]),
        build=raw["build"],
        ctx=ctx,
        aliases=_as_str_tuple(raw.get("aliases"), f"models.{key}.aliases"),
        reasoning=reasoning,
        tools=tools,
        flags=_as_str_tuple(raw.get("flags"), f"models.{key}.flags"),
        env=merged_env,
        presets=presets,
        vram_mib=int(vram_mib) if vram_mib is not None else None,
        min_output_tokens=raw.get("min_output_tokens"),
        max_output_tokens=raw.get("max_output_tokens"),
        vision=bool(raw.get("vision", False)),
        needs_tty=bool(raw.get("needs_tty", False)),
    )


def _validate(models: dict[str, Model], gpu: GPU, builds: dict[str, str]) -> None:
    # -- names: id/alias/preset unique across the whole file, and a preset must
    # not shadow an id/alias (its own model's, or any other model's).
    owners: dict[str, tuple[str, str]] = {}  # name -> (model_key, kind)

    def claim(name: str, kind: str, model_key: str) -> None:
        prior = owners.get(name)
        if prior is None:
            owners[name] = (model_key, kind)
            return
        prior_key, prior_kind = prior
        if kind == "preset" or prior_kind == "preset":
            raise RegistryError(
                f"models.{model_key}: preset/id/alias {name!r} shadows the "
                f"{prior_kind} of models.{prior_key} (names must be unique across "
                "the whole registry)"
            )
        raise RegistryError(
            f"duplicate {kind} {name!r}: claimed by both models.{prior_key} "
            f"and models.{model_key}"
        )

    for key, m in models.items():
        claim(m.id, "id", key)
        for a in m.aliases:
            claim(a, "alias", key)
        for p in m.presets:
            claim(p, "preset", key)

    # -- ports: unique, and never 8000/8010.
    port_owners: dict[int, str] = {}
    for key, m in models.items():
        if m.port in RESERVED_PORTS:
            raise RegistryError(
                f"models.{key}: port {m.port} is reserved (8000 and 8010 are never "
                "a model's own port)"
            )
        prior_key = port_owners.get(m.port)
        if prior_key is not None:
            raise RegistryError(
                f"duplicate port {m.port}: claimed by both models.{prior_key} "
                f"and models.{key}"
            )
        port_owners[m.port] = key

    # -- unknown build.
    for key, m in models.items():
        if m.build not in builds:
            raise RegistryError(
                f"models.{key}: build {m.build!r} is not in [builds] "
                f"(known: {sorted(builds)})"
            )

    # -- resident VRAM budget must leave room for a main model.
    resident_total = sum(m.vram_mib or 0 for m in models.values() if m.slot == "resident")
    if resident_total + gpu.margin_mib >= gpu.total_mib:
        raise RegistryError(
            f"resident models use {resident_total} MiB + {gpu.margin_mib} MiB margin "
            f">= {gpu.total_mib} MiB total: no room would be left for a main model"
        )


def load(path: str | Path) -> Registry:
    """Load and validate ``models.toml``. Raises :class:`RegistryError` on the
    first problem found (never a bare KeyError/TypeError from tomllib's dict)."""
    p = Path(path)
    try:
        with p.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise RegistryError(f"{p}: invalid TOML: {e}") from e
    except OSError as e:
        raise RegistryError(f"{p}: cannot read: {e}") from e

    gpu_raw = data.get("gpu")
    if gpu_raw is None or "total_mib" not in gpu_raw or "margin_mib" not in gpu_raw:
        raise RegistryError("[gpu] table with total_mib and margin_mib is required")
    gpu = GPU(total_mib=int(gpu_raw["total_mib"]), margin_mib=int(gpu_raw["margin_mib"]))

    builds = dict(data.get("builds") or {})
    defaults_env = dict((data.get("defaults") or {}).get("env") or {})

    models_raw = data.get("models") or {}
    if not models_raw:
        raise RegistryError("[models.*] — at least one model is required")

    models: dict[str, Model] = {}
    for key, raw in models_raw.items():
        models[key] = _load_model(key, raw, defaults_env)

    _validate(models, gpu, builds)

    return Registry(models=models, gpu=gpu, builds=builds, defaults_env=defaults_env)


# --------------------------------------------------------------------------- #
# Utilisation arithmetic (REDESIGN §2.1: "computed at launch", never hand-tuned)
# --------------------------------------------------------------------------- #


def _floor_2dp(x: float) -> float:
    """Round DOWN to 2 decimals. Never round up: an over-estimated utilisation is
    a refused/OOM boot, an under-estimated one is merely a little headroom left
    on the table."""
    return math.floor(x * 100.0) / 100.0


def main_util(free_mib: float, total_mib: float, margin_mib: float) -> float:
    """A `main`-slot model's ``--gpu-memory-utilization``: everything free, minus
    the margin, as a fraction of the card — REDESIGN §2.1's "everything that is
    free" rule. Rounded down to 2 decimals (vLLM's own CLI precision)."""
    if total_mib <= 0:
        raise ValueError(f"total_mib must be positive, got {total_mib!r}")
    return _floor_2dp((free_mib - margin_mib) / total_mib)


def resident_util(vram_mib: float, total_mib: float) -> float:
    """A resident model's ``--gpu-memory-utilization``: its fixed budget as a
    fraction of the card. Rounded down to 2 decimals."""
    if total_mib <= 0:
        raise ValueError(f"total_mib must be positive, got {total_mib!r}")
    return _floor_2dp(vram_mib / total_mib)


# --------------------------------------------------------------------------- #
# Native context length
# --------------------------------------------------------------------------- #


def native_ctx(repo: str, hub_dir: str | Path | None = None) -> int:
    """A checkpoint's own ``max_position_embeddings`` (or the `text_config`
    equivalent for a multi-config checkpoint), read from its LOCAL hub cache
    ``config.json`` — never fetched, per :func:`servedeck.discovery.load_model_config`.

    This is the "full native context, always" owner rule (REDESIGN §0): a
    model's ``ctx = "native"`` in ``models.toml`` resolves through here.
    """
    cfg = _discovery.load_model_config(repo, hub_dir)
    if cfg is None:
        raise RegistryError(
            f"native_ctx({repo!r}): no config.json found in the local hub cache "
            "(model not downloaded, or repo id is wrong)"
        )
    text_config = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    value = text_config.get("max_position_embeddings") or cfg.get("max_position_embeddings")
    if not value:
        raise RegistryError(
            f"native_ctx({repo!r}): config.json has no max_position_embeddings"
        )
    return int(value)


# --------------------------------------------------------------------------- #
# Rendering a `vllm serve` invocation
# --------------------------------------------------------------------------- #


def render_argv(
    model: Model,
    venv_python_or_vllm_bin: str,
    util: float,
    ctx_tokens: int,
    port: int,
) -> list[str]:
    """The full ``vllm serve`` argv for one model, at an already-resolved
    utilisation/context/port. Pure — no I/O, no registry lookups — so it can be
    compared byte-for-byte against a legacy launcher's dry-run output (see
    ``tests/test_models_golden.py``).

    Fixed order (REDESIGN §2.4 deliverable spec): served-model-name, host,
    tool-choice+parser, reasoning-parser, max-model-len, gpu-memory-utilization,
    port, then the model's own ``flags``. This is ONE order for every model,
    which is not the order any of the three legacy launchers used (each used its
    own, mutually different, order) — the golden test therefore compares
    flag-to-value mappings, not raw positional sequence; see that test's
    docstring for why.
    """
    argv: list[str] = [venv_python_or_vllm_bin, "serve", model.repo]
    argv += ["--served-model-name", *model.served_names()]
    argv += ["--host", "127.0.0.1"]
    if model.tools is not None:
        argv += ["--enable-auto-tool-choice", "--tool-call-parser", model.tools.parser]
    if model.reasoning is not None:
        argv += ["--reasoning-parser", model.reasoning.parser]
    argv += ["--max-model-len", str(ctx_tokens)]
    argv += ["--gpu-memory-utilization", f"{util:.2f}"]
    argv += ["--port", str(port)]
    argv += list(model.flags)
    return argv


def render_env(model: Model) -> dict[str, str]:
    """A model's fully-resolved launch environment (``[defaults.env]`` already
    merged in by :func:`load`)."""
    return dict(model.env)
