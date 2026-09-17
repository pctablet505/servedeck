"""servedeck.models — the v2 model registry (REDESIGN-2026-09-12.md §2.1, §2.4, §2.7).

``models.toml`` is the single source of truth for every model this box can serve.
This module loads it, validates it, and turns one ``Model`` entry into the argv/env
of a ``vllm serve`` invocation — nothing here starts a process (that is P3/P4's
supervisor) and nothing here writes a client config (that is ``servedeck.wire``).

Design notes, so a later packet does not have to re-derive them:

* ``load()`` merges ``[defaults.env]`` into every model's own ``env`` at LOAD time
  (the model's own keys win on conflict), so ``render_env()`` only needs the model
  plus its resolved ``Build`` — no registry-level merge step downstream.
* ``render_argv()`` takes ``util``/``ctx_tokens``/``port`` as already-resolved
  values, not the registry or the hub cache: it never does I/O, which is what makes
  the golden test (``tests/test_models_golden.py``) able to compare it byte-for-byte
  against a legacy launcher's dry-run output. Resolving ``ctx="native"`` to an int is
  a separate step (``native_ctx()``), because that DOES need the hub cache.
* Utilisation arithmetic (``compute_util``/``floor2``) lives in
  ``servedeck.control`` (decision 4 of REDESIGN §2.1), not here — this module
  used to carry its own ``main_util``/``resident_util``, and two implementations
  of the same rule is exactly the kind of divergence that rule exists to avoid.
"""

from __future__ import annotations

import re
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
    "Build",
    "GPU",
    "Registry",
    "Resolved",
    "load",
    "render_argv",
    "render_env",
    "native_ctx",
]

#: Ports that belong to the gateway (:8010) and to the unrelated ats-optimizer.service
#: (:8000) — REDESIGN-2026-09-12.md §2.1 "ports unique and never 8000 or 8010".
RESERVED_PORTS: frozenset[int] = frozenset({8000, 8010})

#: Registry keys become unit names (``model-<key>``, servedeck.units._NAME_RE
#: only allows ``[a-z0-9-]+`` after that prefix) — enforced here too, so a bad
#: key is a load-time RegistryError instead of a refusal three modules later.
_KEY_RE = re.compile(r"^[a-z0-9-]+$")

#: Standard system dirs appended to a model's PATH after its venv/CUDA bins —
#: REDESIGN-2026-09-12.md launch-environment-completeness fix: a transient
#: systemd unit inherits the user manager's environment, never the invoking
#: shell's, so anything a model's own tooling shells out to (env, sh, coreutils)
#: must be reachable from an explicit, self-sufficient PATH.
_SYSTEM_PATH_DIRS = ("/usr/local/bin", "/usr/bin", "/bin")


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
    #: The GPU utilisation this model is PROVEN to serve at on this card, from
    #: the launcher that ran it (main slot only; a resident derives its own
    #: from ``vram_mib``). Without it, a plain start hands the model whatever
    #: the free-VRAM arithmetic yields — 0.98 on an idle card, which boots and
    #: then dies of CUDA OOM under concurrency (measured 2026-09-03), while
    #: Flash-Next has only ever been run at 0.96 and the 27B at 0.95. It is a
    #: ceiling, not a target: ``control.compute_util`` refuses rather than
    #: quietly launching at a lower utilisation than the proven one.
    util: float | None = None
    #: Bytes of KV cache this model needs per context token, when the engine
    #: has to be told explicitly (``--kv-cache-memory-bytes``). GLM's launcher
    #: derived it as MAX_LEN * 17200 at launch time (serve-opt.sh:315); the
    #: registry used to pin the PRODUCT instead, so moving the context control
    #: left the KV cap sized for the old length and vLLM refused to start
    #: after a six-minute load. Derived here, the two can never disagree.
    kv_cache_bytes_per_token: int | None = None
    #: Host RAM (GiB of MemAvailable) this model needs before it may launch.
    #: This box has NO swap, and pinned pages cannot be reclaimed, so
    #: overshooting host RAM is an OOM kill of the desktop rather than a
    #: slowdown — one happened on 2026-08-28. GLM's launcher refused to start
    #: below CPU_OFFLOAD_GB + 30 (serve-opt.sh:204-226, calibrated against a
    #: measured 168 GiB peak at 110 GiB of offload); nothing carried that
    #: guard into servedeck, so a badly timed switch could take the box down.
    host_ram_gib: int | None = None
    min_output_tokens: int | None = None
    max_output_tokens: int | None = None
    vision: bool = False
    #: Retired 2026-09-18 (see models.toml): nothing enforced it, the sysctl
    #: file makes it false for the one model that set it, and doctor's
    #: ptrace_scope row is now unconditional. Kept as a parsed field so an
    #: older models.toml still loads instead of failing on an unknown key.
    needs_tty: bool = False
    #: P9 — the GLM-5.3 client-compatibility switches (servedeck/glm_policies.py).
    #: They live on the MODEL, not in an env var, so they travel with the model
    #: through `servedeck wire`, `doctor` and a machine move; the gateway reads
    #: them straight onto `gateway.RoutePolicies`.
    #:
    #: `sanitize_tool_tags` — strip GLM template markup its tool-call parser
    #: leaks into parsed arguments (captured live: `list_dir({"path</arg_key>":
    #: ...})`, which the client then rejects). A pure repair of text that is
    #: never valid argument content, so it defaults TRUE for glm53.
    sanitize_tool_tags: bool = False
    #: `restore_reasoning` — put reasoning back on the in-flight assistant turns
    #: of a request whose client echoed the tool_call ids but dropped the
    #: thinking. Defaults FALSE, glm53 included, and must stay false until
    #: somebody decides otherwise with evidence: the source it is ported from
    #: records "three Xid 31 GPU faults followed within 40 minutes of it first
    #: firing, after 14 hours clean", and this box has an active, unrelated
    #: GPU-fault problem. Correlation, not proof — but nothing that plausibly
    #: perturbs the GPU may default to on.
    restore_reasoning: bool = False
    #: `capture` — this model CONSENTS to request/response capture. Not an
    #: enable on its own: capture writes whatever the user typed to disk, so it
    #: also needs the master switch `SERVEDECK_GLM_CAPTURE=1`
    #: (glm_policies.CAPTURE_ENV), which deliberately is not a registry field.
    capture: bool = False

    def served_names(self) -> list[str]:
        """``[id, *aliases, *presets]`` — REDESIGN §2.4/§2.1: every name a client
        has ever used for this model, in the order they should appear on the vLLM
        command line and in ``GET /v1/models``."""
        return [self.id, *self.aliases, *self.presets.keys()]


@dataclass(frozen=True)
class Build:
    """One ``[builds.<name>]`` table: the vLLM venv a model launches from, and
    the CUDA toolkit directory that ships inside it.

    Every legacy launcher exports ``CUDA_HOME=<venv>/lib/python3.13/site-packages/
    nvidia/cu13`` and prepends ``<venv>/bin:$CUDA_HOME/bin`` to ``PATH``
    (``qwen-server-run.sh:267-268``, ``serve.sh:63,72``, ``serve-opt.sh:168,177``)
    — nvcc/ptxas live there and FlashInfer's JIT needs them at request time, not
    just at process start. ``cuda_home`` is carried explicitly rather than
    derived by string-gluing a python version onto ``venv`` in code, because the
    python version is a fact about how the venv was built, not a constant this
    module should assume.
    """

    venv: str
    cuda_home: str


@dataclass(frozen=True)
class GPU:
    total_mib: int
    margin_mib: int
    #: The power cap this box is supposed to run at, in watts, for `servedeck
    #: doctor` to compare against the live one. Optional and deliberately
    #: unset by default: the cap is an owner decision (the card fell off the
    #: bus twice, at 600 W under load and at 450 W idle), and a number in this
    #: file that nobody has decided would read as one that somebody had.
    power_limit_w: int | None = None


@dataclass(frozen=True)
class Registry:
    """The loaded, validated ``models.toml``."""

    models: dict[str, Model]  # TOML key -> Model, insertion order preserved
    gpu: GPU
    builds: dict[str, Build]
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


def _as_util(raw: Mapping[str, Any], key: str) -> float | None:
    """``util`` as a fraction in (0, 1]. A typo here would be a launch at a
    utilisation nobody chose, so it is refused at load time."""
    value = raw.get("util")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegistryError(f"models.{key}.util must be a number, got {value!r}")
    out = float(value)
    if not 0.0 < out <= 1.0:
        raise RegistryError(f"models.{key}.util must be in (0, 1], got {out}")
    return out


def _as_str_tuple(value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise RegistryError(f"{where} must be a list of strings, got {value!r}")
    return tuple(value)


def _as_bool(raw: dict[str, Any], key: str, field_name: str) -> bool:
    """A registry switch must be a real TOML boolean.

    Not ``bool(value)``: ``capture = "false"`` is a truthy string, and a switch
    that turns *on* when its config says "false" is the worst possible failure
    for a flag whose whole job is to keep request bodies off the disk unless
    somebody asked for that.
    """
    if field_name not in raw:
        return False
    value = raw[field_name]
    if not isinstance(value, bool):
        raise RegistryError(
            f"models.{key}.{field_name} must be true or false, got {value!r}"
        )
    return value


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
        util=_as_util(raw, key),
        host_ram_gib=(
            int(raw["host_ram_gib"]) if raw.get("host_ram_gib") is not None else None
        ),
        kv_cache_bytes_per_token=(
            int(raw["kv_cache_bytes_per_token"])
            if raw.get("kv_cache_bytes_per_token") is not None
            else None
        ),
        min_output_tokens=raw.get("min_output_tokens"),
        max_output_tokens=raw.get("max_output_tokens"),
        vision=bool(raw.get("vision", False)),
        needs_tty=bool(raw.get("needs_tty", False)),
        sanitize_tool_tags=_as_bool(raw, key, "sanitize_tool_tags"),
        restore_reasoning=_as_bool(raw, key, "restore_reasoning"),
        capture=_as_bool(raw, key, "capture"),
    )


def _load_builds(raw: dict[str, Any]) -> dict[str, Build]:
    builds: dict[str, Build] = {}
    for name, table in raw.items():
        if not isinstance(table, dict):
            raise RegistryError(
                f"builds.{name} must be a table with 'venv' and 'cuda_home', "
                f"got {table!r}"
            )
        for required in ("venv", "cuda_home"):
            if not table.get(required):
                raise RegistryError(f"builds.{name}: missing {required!r}")
        builds[name] = Build(venv=table["venv"], cuda_home=table["cuda_home"])
    return builds


def _validate(models: dict[str, Model], gpu: GPU, builds: dict[str, Build]) -> None:
    # -- registry keys become unit names (`model-<key>`): restrict the shape
    # here rather than let a bad key surface as a systemd refusal later.
    for key in models:
        if not _KEY_RE.match(key):
            raise RegistryError(
                f"models.{key}: key must match {_KEY_RE.pattern!r} (it becomes "
                f"the unit name model-{key})"
            )

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
    gpu = GPU(
        total_mib=int(gpu_raw["total_mib"]),
        margin_mib=int(gpu_raw["margin_mib"]),
        power_limit_w=(
            int(gpu_raw["power_limit_w"]) if gpu_raw.get("power_limit_w") is not None else None
        ),
    )

    builds = _load_builds(data.get("builds") or {})
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
    if model.kv_cache_bytes_per_token:
        # Derived from the SAME ctx the engine is given, so the cap always
        # matches the context (see the field's docstring).
        argv += ["--kv-cache-memory-bytes", str(ctx_tokens * model.kv_cache_bytes_per_token)]
    argv += ["--port", str(port)]
    argv += list(model.flags)
    return argv


def render_env(model: Model, build: Build) -> dict[str, str]:
    """A model's COMPLETE launch environment: ``[defaults.env]`` and the
    model's own ``env`` (merged in by :func:`load`), plus ``CUDA_HOME`` and a
    full ``PATH`` derived from ``build``.

    Every legacy launcher exports these two (see :class:`Build`'s docstring for
    citations) and none of them relies on an inherited ``PATH`` doing the job —
    which matters here even more than it did for a shell script: a
    ``systemd-run --user`` transient unit inherits the USER MANAGER's
    environment, never the caller's shell, so a model launched without an
    explicit ``PATH`` would not even find ``env``/``sh``. ``venv``/``cuda_home``
    are expanded here (not left as ``~...``): a real ``CUDA_HOME`` environment
    variable is read literally by nvcc/torch, which do not expand ``~``.
    """
    env = dict(model.env)
    venv = str(Path(build.venv).expanduser())
    cuda_home = str(Path(build.cuda_home).expanduser())
    env["CUDA_HOME"] = cuda_home
    env["PATH"] = ":".join((f"{venv}/bin", f"{cuda_home}/bin", *_SYSTEM_PATH_DIRS))
    return env
