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

All network access is a plain ``httpx.get`` with a 2 s timeout, injectable via
``http_get`` so tests never need a real server — except the two tests that are
explicitly allowed to hit the box's own live, read-only ``:8007`` and ``:8010``.
Unit listing is a plain :class:`servedeck.units.Runner`, injectable the same
way (see ``servedeck.units``'s own test suite for the convention).
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import httpx

from . import models as _models
from . import units as _units
from . import wire as _wire

__all__ = [
    "CheckResult",
    "Unreachable",
    "check_registry",
    "check_client_config",
    "check_port",
    "check_model_units",
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


def check_client_config(
    kind: str, path: str | Path, *, timeout: float = 2.0, http_get: HttpGet | None = None
) -> list[CheckResult]:
    """``kind`` is one of "vscode" / "codex" / "kimi". Missing file is not a
    failure (nothing to check yet); a file present with no model references is
    reported as such rather than silently producing zero results."""
    p = Path(path)
    if not p.is_file():
        return [CheckResult(f"{kind} config", True, f"{p} not present")]
    text = p.read_text()
    refs = _PARSERS[kind](text)
    if not refs:
        return [CheckResult(f"{kind} config", True, f"{p} present, no model references found")]

    results: list[CheckResult] = []
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
# Orchestration + presentation
# --------------------------------------------------------------------------- #


def run_doctor(
    models_path: str | Path,
    *,
    client_files: dict[str, Path] | None = None,
    unit_run: _units.Runner | None = None,
    timeout: float = 2.0,
    http_get: HttpGet | None = None,
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
    for kind, path in files.items():
        results.extend(check_client_config(kind, path, timeout=timeout, http_get=http_get))

    for m in registry.models.values():
        results.append(check_port(m, timeout=timeout, http_get=http_get))

    results.extend(check_model_units(registry, run=unit_run))
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
