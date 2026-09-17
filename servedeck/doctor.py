"""servedeck.doctor — proves the registry against reality (REDESIGN-2026-09-12.md §2.4).

"Every client config drift becomes a red line in doctor instead of a 404 in the
editor an hour later." Four kinds of check, each yielding one or more
:class:`CheckResult`:

1. the registry loads and validates.
2. for each of the three client config files that EXISTS on disk, every model
   id it references answers at its configured URL's ``/v1/models`` — reported
   as ``ok`` / ``missing`` (something else answers there) / ``unreachable``.
3. for each model in the registry, its own port either answers ``/v1/models``
   with that model's id (or an alias) in the list, or is not listening at all
   — a THIRD state (a different model answering) is the one real failure.
4. which registry models currently have a live ``model-<key>`` transient unit
   (:func:`servedeck.units.list_units`), and whether any live ``model-*`` unit
   belongs to a key the registry does not know. A model with NO unit is not a
   failure — most registry models are not running most of the time (`main` is
   exclusive, `resident`s are opt-in) — only a STRAY unit (something running
   under a key this ``models.toml`` has never heard of) is.

   This deliberately does NOT look at ``~/.config/systemd/user/*.service``:
   v2's models run as ``systemd-run --user`` TRANSIENT units, which live in
   ``/run/user/<uid>/systemd/transient`` and are enumerated with
   ``systemctl --user list-units``, never as files in that directory — a
   file-existence check there can never pass after the cutover.
5. the two host-state checks that survived ``preflight.py``: no training
   marker claims the GPU, and ``ptrace_scope`` is 0 for any model that
   declares ``needs_tty``. Both are about the HOST, not about a model's
   configuration, which is why neither had anywhere else to go when
   ``preflight.py`` was deleted.

All network access is a plain ``httpx.get`` with a 2 s timeout, injectable via
``http_get`` so tests never need a real server — except the two tests that are
explicitly allowed to hit the box's own live, read-only ``:8007`` and ``:8010``.
Unit listing is a plain :class:`servedeck.units.Runner`, injectable the same
way (see ``servedeck.units``'s own test suite for the convention).
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Collection, Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import httpx

from . import control as _control
from . import desired as _desired
from . import discovery as _discovery
from . import gpu as _gpu
from . import limits as _limits
from . import models as _models
from . import settings as _settings
from . import units as _units
from . import wire as _wire

__all__ = [
    "CheckResult",
    "Unreachable",
    "check_registry",
    "check_client_config",
    "check_port",
    "check_model_units",
    "check_training_marker",
    "check_ptrace_scope",
    "run_doctor",
    "all_ok",
    "format_table",
]


class CheckResult(NamedTuple):
    name: str
    ok: bool
    detail: str


class Unreachable(Exception):
    """The endpoint did not answer at all (connection refused/timeout) —
    distinct from answering with the wrong content, which is a real failure."""


HttpGet = Callable[[str, float], httpx.Response]


def _default_get(url: str, timeout: float) -> httpx.Response:
    return httpx.get(url, timeout=timeout)


def _fetch_model_ids(url: str, timeout: float, http_get: HttpGet | None) -> list[str]:
    """GET ``url``, return the ``id`` of every entry in the standard
    ``{"data": [...]}`` /v1/models envelope. Raises :class:`Unreachable` on a
    connection failure/timeout, ``ValueError`` on a reachable-but-malformed
    response (never silently returns [])."""
    getter = http_get or _default_get
    try:
        resp = getter(url, timeout)
    except httpx.TimeoutException as e:
        raise Unreachable(f"timed out: {e}") from e
    except httpx.TransportError as e:
        raise Unreachable(f"connection failed: {e}") from e
    if resp.status_code != 200:
        raise ValueError(f"HTTP {resp.status_code}")
    try:
        data = resp.json()
    except ValueError as e:
        raise ValueError(f"non-JSON response: {e}") from e
    entries = data.get("data") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ValueError("response has no 'data' list")
    return [e["id"] for e in entries if isinstance(e, dict) and "id" in e]


# --------------------------------------------------------------------------- #
# 1. registry loads
# --------------------------------------------------------------------------- #


def check_registry(path: str | Path) -> CheckResult:
    try:
        reg = _models.load(path)
    except _models.RegistryError as e:
        return CheckResult("registry loads", False, str(e))
    return CheckResult("registry loads", True, f"{len(reg.models)} model(s) at {path}")


# --------------------------------------------------------------------------- #
# 2. client configs
# --------------------------------------------------------------------------- #


def _refs_from_vscode(text: str) -> list[tuple[str, str]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    refs: list[tuple[str, str]] = []
    if isinstance(data, list):
        for group in data:
            if not isinstance(group, dict):
                continue
            for m in group.get("models") or []:
                if isinstance(m, dict) and isinstance(m.get("id"), str) and isinstance(m.get("url"), str):
                    models_url = m["url"].rsplit("/chat/completions", 1)[0].rstrip("/") + "/models"
                    refs.append((m["id"], models_url))
    return refs


def _refs_via_provider_table(
    text: str, *, providers_key: str, entries_key: str, provider_field: str, model_field: str
) -> list[tuple[str, str]]:
    """Shared shape of Codex (model_providers/profiles) and Kimi
    (providers/models): a table of named providers with base_url, and a table
    of named entries each naming one provider and one model."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return []
    providers = data.get(providers_key) or {}
    refs: list[tuple[str, str]] = []
    for entry in (data.get(entries_key) or {}).values():
        if not isinstance(entry, dict):
            continue
        provider_name = entry.get(provider_field)
        model_name = entry.get(model_field)
        provider = providers.get(provider_name) if isinstance(provider_name, str) else None
        if not (isinstance(model_name, str) and isinstance(provider, dict) and provider.get("base_url")):
            continue
        base = str(provider["base_url"]).rstrip("/")
        refs.append((model_name, f"{base}/models"))
    return refs


def _refs_from_codex(text: str) -> list[tuple[str, str]]:
    return _refs_via_provider_table(
        text,
        providers_key="model_providers",
        entries_key="profiles",
        provider_field="model_provider",
        model_field="model",
    )


def _refs_from_kimi(text: str) -> list[tuple[str, str]]:
    return _refs_via_provider_table(
        text,
        providers_key="providers",
        entries_key="models",
        provider_field="provider",
        model_field="model",
    )


_PARSERS: dict[str, Callable[[str], list[tuple[str, str]]]] = {
    "vscode": _refs_from_vscode,
    "codex": _refs_from_codex,
    "kimi": _refs_from_kimi,
}

#: Hosts that are this box. A client reference to anything else is somebody
#: else's service and none of servedeck's business.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]", "0.0.0.0"})


def is_local_ref(url: str) -> bool:
    """True iff ``url``'s host is this machine's loopback.

    Measured on this box: ``~/.kimi-code/config.toml`` declares four
    ``managed:kimi-code`` models behind ``https://api.kimi.com/coding/v1``, and
    without this filter ``doctor`` sent an unauthenticated request to that host
    on every run and reported each 401 as a servedeck failure — four permanent
    red lines for four entries that are perfectly correct, and a doctor with
    four permanent red lines is a doctor nobody reads.

    Two independent reasons the filter belongs here, not in the report:

    * **Scope.** REDESIGN §2.4's claim is about *generated* entries: "for every
      generated entry, GET /v1/models at the configured URL must list that exact
      id", and everything ``servedeck wire`` generates points at
      ``127.0.0.1:8010``. A cloud provider a user configured by hand is not
      drift servedeck can create or fix.
    * **It is an outbound request.** ``doctor`` is a local diagnostic an
      operator runs freely, including in a loop. Reaching a third-party
      endpoint from it tells that endpoint the box exists on every run, and any
      day a client config gains an inline key in a header or query string, the
      same code path would carry it there.
    """
    from urllib.parse import urlsplit

    try:
        host = urlsplit(url).hostname
    except ValueError:
        return False
    return host is not None and host.lower() in _LOOPBACK_HOSTS


def check_client_config(
    kind: str,
    path: str | Path,
    *,
    timeout: float = 2.0,
    http_get: HttpGet | None = None,
    known_names: Collection[str] = (),
) -> list[CheckResult]:
    """``kind`` is one of "vscode" / "codex" / "kimi". Missing file is not a
    failure (nothing to check yet); a file present with no model references is
    reported as such rather than silently producing zero results.

    ``known_names`` is every id, alias and preset the registry serves through
    the gateway. A client entry for one of those that the endpoint does not
    list right now is *wired, not running* — the gateway answers it with a 503
    and a reason until the model is started — not a failure. Without it (the
    2026-09-17 cutover night) three stopped models made the page say "13 of
    29 failing" with nothing wrong."""
    p = Path(path)
    if not p.is_file():
        return [CheckResult(f"{kind} config", True, f"{p} not present")]
    text = p.read_text()
    refs = _PARSERS[kind](text)
    if not refs:
        return [CheckResult(f"{kind} config", True, f"{p} present, no model references found")]

    results: list[CheckResult] = []
    remote = [(m, u) for m, u in refs if not is_local_ref(u)]
    refs = [(m, u) for m, u in refs if is_local_ref(u)]
    if remote:
        # Reported, not silently dropped: an operator reading the table must be
        # able to see that these entries exist and were deliberately not probed.
        results.append(
            CheckResult(
                f"{kind} config (remote)",
                True,
                f"{len(remote)} entr{'y' if len(remote) == 1 else 'ies'} point off this box "
                f"and were not probed: "
                + ", ".join(f"{m} @ {u.rsplit('/models', 1)[0]}" for m, u in remote),
            )
        )
    if not refs:
        return results or [
            CheckResult(f"{kind} config", True, f"{p} present, no local model references found")
        ]
    for model_id, models_url in refs:
        name = f"{kind}: {model_id}"
        try:
            ids = _fetch_model_ids(models_url, timeout, http_get)
        except Unreachable as e:
            results.append(CheckResult(name, False, f"unreachable at {models_url}: {e}"))
            continue
        except ValueError as e:
            results.append(CheckResult(name, False, f"unreachable at {models_url}: {e}"))
            continue
        if model_id in ids:
            results.append(CheckResult(name, True, f"ok — {models_url} serves {model_id!r}"))
        elif model_id in known_names:
            results.append(
                CheckResult(
                    name, True, f"wired — {model_id!r} is in the registry but not running; "
                    f"{models_url} will list it once it is started"
                )
            )
        else:
            results.append(
                CheckResult(name, False, f"missing — {models_url} serves {ids!r}, not {model_id!r}")
            )
    return results


# --------------------------------------------------------------------------- #
# 3. registry ports
# --------------------------------------------------------------------------- #


def check_port(model: _models.Model, *, timeout: float = 2.0, http_get: HttpGet | None = None) -> CheckResult:
    name = f"port {model.port} ({model.key})"
    url = f"http://127.0.0.1:{model.port}/v1/models"
    try:
        ids = _fetch_model_ids(url, timeout, http_get)
    except Unreachable:
        return CheckResult(name, True, "not listening")
    except ValueError as e:
        return CheckResult(name, False, f"answers but response is unusable: {e}")
    served = {model.id, *model.aliases}
    hit = served & set(ids)
    if hit:
        return CheckResult(name, True, f"listening, serves {sorted(hit)}")
    return CheckResult(
        name, False, f"port {model.port} answers but serves {ids!r}, expected one of {sorted(served)!r}"
    )


# --------------------------------------------------------------------------- #
# 4. model units (transient systemd-run --user units — servedeck.units)
# --------------------------------------------------------------------------- #


def check_model_units(
    registry: _models.Registry, *, run: _units.Runner | None = None
) -> list[CheckResult]:
    """Which registry models have a live ``model-<key>`` unit right now, via
    ``systemctl --user list-units 'model-*'`` (:func:`servedeck.units.list_units`)
    — never a directory listing: transient units are not files.

    Not having a unit is NOT a failure — a `main`-slot model that lost the
    exclusive slot, or a `resident` nobody has started yet, both correctly
    report "not running". The one real failure this surfaces: a live
    ``model-*`` unit whose key this ``models.toml`` does not know at all (a
    stray — started by a since-removed registry entry, or a typo'd key).
    """
    try:
        live = set(_units.list_model_units(run=run))
    except _units.UnitError as e:
        return [CheckResult("model units", False, f"could not list model-* units: {e}")]

    results: list[CheckResult] = []
    for key in registry.models:
        unit = f"model-{key}"
        if unit in live:
            results.append(CheckResult(f"unit (model-{key})", True, "running"))
        else:
            results.append(CheckResult(f"unit (model-{key})", True, "not running"))

    known_units = {f"model-{key}" for key in registry.models}
    for stray in sorted(live - known_units):
        stray_key = stray[len("model-") :]
        results.append(
            CheckResult(
                f"unit ({stray})",
                False,
                f"{stray} is a live unit, but models.toml has no model with key {stray_key!r}",
            )
        )
    return results


# --------------------------------------------------------------------------- #
# 5. The two preflight checks that survived
# --------------------------------------------------------------------------- #
#
# ``preflight.py`` was 382 lines of launch gating built around a supervisor
# that no longer exists: venv presence, launcher scripts, log paths, GPU
# health, a Codex subagent cap. systemd, the registry and ``control`` cover all
# of it now. Two checks had no other home, and both share doctor's shape —
# something on this box is in a state that will make a boot fail, and an
# operator wants to know *before* burning four minutes discovering it.

#: A "lock file" convention: if one of these exists, something else wants the
#: GPU (a training run, a benchmark) and servedeck must stand down.
#: ``$SERVEDECK_TRAINING_MARKERS`` (colon-separated) overrides the defaults.
PTRACE_PATH = Path("/proc/sys/kernel/yama/ptrace_scope")

#: Flash-Next's PLE CUDA-IPC handoff needs ``pidfd_getfd``, which needs this.
PTRACE_SCOPE_REQUIRED = 0


def check_training_marker(marker_paths: Sequence[str] | None = None) -> CheckResult:
    """Is something else claiming the GPU right now?

    ``qwen-server-run.sh``'s own guard 1, kept because the marker files are
    written by tools outside this repo (an AlgoTrading training run, chiefly)
    and nothing else would notice them. A hit is a FAILURE, not a warning: the
    correct response is to leave the card alone, and a warning is what gets
    scrolled past.
    """
    candidates = _limits.training_markers() if marker_paths is None else tuple(marker_paths)
    hits = [p for p in candidates if Path(p).exists()]
    if not hits:
        return CheckResult(
            "training marker",
            True,
            f"none of the {len(candidates)} training-marker paths exist",
        )
    return CheckResult(
        "training marker",
        False,
        f"{hits[0]} exists — something else wants the GPU; do not start a model "
        f"(remove it only once that run has actually finished)",
    )


class _Unset:
    """Sentinel: ``scope=None`` means "unreadable", which is a real and
    reportable state, so it cannot double as "not supplied"."""


_UNSET = _Unset()


def _read_ptrace_scope() -> int | None:
    try:
        return int(PTRACE_PATH.read_text().strip())
    except (OSError, ValueError):
        return None


def check_ptrace_scope(
    registry: _models.Registry, scope: int | None | _Unset = _UNSET
) -> list[CheckResult]:
    """``kernel.yama.ptrace_scope`` — one HOST row, always.

    Two things on this box need it at 0, which is why it is no longer keyed on
    a model flag (``needs_tty``, removed 2026-09-18 — nothing enforced it, and
    /etc/sysctl.d/90-servedeck.conf made "cannot boot unattended" false):

    * Flash-Next's PLE offload hands a CUDA IPC fd between sibling worker
      processes (``pidfd_getfd``), which yama blocks at scope 1.
    * servedeck's own KV-offload reaper reads ``/proc/<pid>/maps`` of engines
      that are NOT its descendants, to prove a 40 GiB buffer is still mapped
      before deleting it. At scope 1 it cannot see them, so it fails closed
      and stops reaping — safe, but the leak then accumulates 40 GiB per
      crash until reboot.

    Unconditional, because a registry-keyed check disappears the moment the
    flag is removed, taking the only report of a host setting two subsystems
    depend on with it.
    """
    if isinstance(scope, _Unset):
        scope = _read_ptrace_scope()
    name = "ptrace_scope (host)"
    if scope is None:
        return [CheckResult(name, False, f"could not read {PTRACE_PATH}")]
    if scope != PTRACE_SCOPE_REQUIRED:
        return [
            CheckResult(
                name,
                False,
                f"kernel.yama.ptrace_scope={scope}, needs 0: Flash-Next's PLE handoff "
                f"(pidfd_getfd between sibling workers) fails, and the KV-offload "
                f"reaper cannot see a live engine's mappings so it stops reclaiming "
                f"leaked buffers. Owner action: /etc/sysctl.d/90-servedeck.conf",
            )
        ]
    return [
        CheckResult(name, True, "0 — the PLE handoff and the offload reaper can both see what they need")
    ]


# --------------------------------------------------------------------------- #
# Orchestration + presentation
# --------------------------------------------------------------------------- #



# --------------------------------------------------------------------------- #
# Host state the 2026-09-18 audit found nothing was reporting
# --------------------------------------------------------------------------- #


def check_offload_buffers(
    shm_dir: Path | None = None, in_use: Callable[[], frozenset[str]] | None = None
) -> CheckResult:
    """Leaked KV-offload buffers: 40 GiB of host RAM each, invisible.

    vLLM unlinks ``/dev/shm/vllm_offload_*.mmap`` only inside its own
    shutdown, which a kill, a crash or a lost GPU skips. servedeck reaps them,
    but the reaper fails closed when it cannot see every engine's mappings
    (ptrace_scope), so "reaped" is not guaranteed — and until this row nothing
    told an operator that 40 or 80 GiB of the 182 was simply gone.
    """
    directory = Path("/dev/shm") if shm_dir is None else shm_dir
    try:
        found = sorted(directory.glob(_control.OFFLOAD_REGION_GLOB))
    except OSError as exc:
        return CheckResult("offload buffers", False, f"could not read {directory}: {exc}")
    if not found:
        return CheckResult("offload buffers", True, f"none in {directory}")
    mapped = in_use() if in_use is not None else _control.offload_scan().paths
    orphans = [f for f in found if str(f) not in mapped]
    total = sum(f.stat().st_size for f in orphans if f.exists())
    if not orphans:
        return CheckResult(
            "offload buffers",
            True,
            f"{len(found)} buffer(s), all mapped by a live engine",
        )
    return CheckResult(
        "offload buffers",
        False,
        f"{len(orphans)} of {len(found)} buffer(s) hold {total / 2**30:.0f} GiB of host RAM "
        f"that no engine maps: {', '.join(f.name for f in orphans)}. A stop or start reaps "
        f"them; if they survive that, check kernel.yama.ptrace_scope",
    )


def check_desired_is_live(
    registry: _models.Registry,
    desired_path: Path,
    run: _units.Runner | None = None,
) -> list[CheckResult]:
    """Does what the box is configured to serve match what is running?

    The one question no check asked. A model whose engine died and whose unit
    systemd then collected leaves desired state naming it, no unit, and no
    complaint anywhere: every client gets "not running" while the page shows
    nothing wrong.
    """
    try:
        want = _desired.load(desired_path)
    except Exception as exc:  # noqa: BLE001
        return [CheckResult("desired state", False, f"{desired_path} unreadable: {exc}")]
    wanted = [k for k in [*want.residents, *([want.main] if want.main else [])]]
    if not wanted:
        return [CheckResult("desired state", True, "nothing is configured to run")]
    try:
        units_present = {
            u.removeprefix(_control.UNIT_PREFIX).removesuffix(".service")
            for u in _units.list_units(f"{_control.UNIT_PREFIX}*", run=run)
        }
    except _units.UnitError as exc:
        return [CheckResult("desired state", False, f"could not list model units: {exc}")]
    out: list[CheckResult] = []
    for key in wanted:
        name = f"desired ({key})"
        if key not in registry.models:
            out.append(CheckResult(name, False, f"{key} is wanted but not in the registry"))
        elif key in units_present:
            out.append(CheckResult(name, True, "wanted, and its unit exists"))
        else:
            out.append(
                CheckResult(
                    name,
                    False,
                    f"{key} is wanted but no {_control.UNIT_PREFIX}{key} unit exists — its "
                    f"engine died and systemd collected the unit, or it never started. "
                    f"`servedeck start {key}` and read the journal",
                )
            )
    return out


def check_weights_present(registry: _models.Registry, hub_dir: str | None = None) -> list[CheckResult]:
    """Is each model's checkpoint actually in the local hub cache?

    Models launch with ``HF_HUB_OFFLINE=1``, so a missing or half-downloaded
    snapshot is not a slow launch, it is a dead one — and the first v2 launch
    of Flash-Next died exactly this way (a gated-repo 401, because offline was
    NOT set and the hub was asked). One row per model, so the answer names
    which checkpoint to fetch.
    """
    try:
        entries = {e.repo_id: e for e in _discovery.discover_models(hub_dir)}
    except Exception as exc:  # noqa: BLE001
        return [CheckResult("weights", False, f"could not read the hub cache: {exc}")]
    out: list[CheckResult] = []
    for m in registry.models.values():
        entry = entries.get(m.repo)
        name = f"weights ({m.key})"
        if entry is None:
            out.append(CheckResult(name, False, f"{m.repo} is not in the local hub cache"))
        elif not entry.servable:
            out.append(CheckResult(name, False, f"{m.repo}: {entry.reason or 'not servable'}"))
        else:
            out.append(CheckResult(name, True, f"{m.repo} is local"))
    return out


def check_power_cap(expected_w: int | None, actual_w: float | None) -> CheckResult:
    """The GPU power cap, which a system unit re-applies at every boot.

    Reported always, because it is a number that silently changes under the
    operator: ``nvidia-power-limit.service`` pins one value, a hand
    ``nvidia-smi -pl`` lasts until reboot, and the difference is measurable
    throughput. It only FAILS when models.toml states an expected value.
    """
    if actual_w is None:
        return CheckResult("power cap", True, "nvidia-smi did not report a power limit")
    if expected_w is None:
        return CheckResult(
            "power cap",
            True,
            f"{actual_w:.0f} W (no expected value in models.toml [gpu] power_limit_w)",
        )
    if abs(actual_w - expected_w) > 1:
        return CheckResult(
            "power cap",
            False,
            f"{actual_w:.0f} W, but models.toml expects {expected_w} W. A boot unit "
            f"re-applies its own value: /etc/systemd/system/nvidia-power-limit.service",
        )
    return CheckResult("power cap", True, f"{actual_w:.0f} W, as models.toml expects")


def run_doctor(
    models_path: str | Path,
    *,
    client_files: dict[str, Path] | None = None,
    unit_run: _units.Runner | None = None,
    timeout: float = 2.0,
    http_get: HttpGet | None = None,
    marker_paths: Sequence[str] | None = None,
    ptrace_scope: int | None | _Unset = _UNSET,
    desired_path: Path | None = None,
    shm_dir: Path | None = None,
    hub_dir: str | None = None,
    power_limit_w: float | None | _Unset = _UNSET,
) -> list[CheckResult]:
    results: list[CheckResult] = [check_registry(models_path)]
    if not results[0].ok:
        return results

    registry = _models.load(models_path)

    files = (
        client_files
        if client_files is not None
        else {
            "vscode": _wire.VSCODE_CHAT_LM_PATH,
            "codex": _wire.CODEX_CONFIG_PATH,
            "kimi": _wire.KIMI_CONFIG_PATH,
        }
    )
    known_names: set[str] = set()
    for m in registry.models.values():
        known_names.add(m.id)
        known_names.update(m.aliases)
        known_names.update(m.presets)
    for kind, path in files.items():
        results.extend(
            check_client_config(
                kind, path, timeout=timeout, http_get=http_get, known_names=known_names
            )
        )

    for m in registry.models.values():
        results.append(check_port(m, timeout=timeout, http_get=http_get))

    results.extend(check_model_units(registry, run=unit_run))
    # Both host checks take their input rather than reading it, for the same
    # reason `http_get` exists: a test cannot set this box's
    # kernel.yama.ptrace_scope, and a check that reads it directly makes
    # `run_doctor`'s result depend on the machine the suite happens to run on.
    results.append(check_training_marker(marker_paths))
    results.extend(check_ptrace_scope(registry, ptrace_scope))
    results.extend(check_desired_is_live(registry, desired_path or _settings.get().desired_path, run=unit_run))
    results.extend(check_weights_present(registry, hub_dir))
    results.append(check_offload_buffers(shm_dir))
    results.append(
        check_power_cap(
            registry.gpu.power_limit_w,
            _gpu.power_limit_w() if power_limit_w is _UNSET else power_limit_w,
        )
    )
    return results


def all_ok(results: list[CheckResult]) -> bool:
    return all(r.ok for r in results)


def format_table(results: list[CheckResult]) -> str:
    if not results:
        return "(no checks ran)"
    name_w = max(len(r.name) for r in results)
    lines = []
    for r in results:
        status = "OK  " if r.ok else "FAIL"
        lines.append(f"[{status}] {r.name.ljust(name_w)}  {r.detail}")
    return "\n".join(lines)
