"""servedeck.cli — the `servedeck` command (REDESIGN-2026-09-12.md §2.4/§2.5).

Subcommands today: ``models`` (a table from the registry), ``wire`` (generate/
update client configs, dry-run by default), ``doctor`` (check the registry
against reality). ``status``/``start``/``stop``/``switch`` are stubs — P3/P4
own the supervisor these will drive.

``python -m servedeck`` (``servedeck/__main__.py``) is unrelated and unchanged:
it starts the DASHBOARD web server. This module is the ``servedeck`` console
script (``[project.scripts]`` in pyproject.toml).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

from . import doctor as _doctor
from . import models as _models
from . import wire as _wire

__all__ = ["build_parser", "main"]

STUB_COMMANDS = ("status", "start", "stop", "switch")


def default_models_toml() -> Path:
    override = os.environ.get("SERVEDECK_MODELS_TOML")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "models.toml"


def _load_or_report(models_toml: Path) -> _models.Registry | None:
    try:
        return _models.load(models_toml)
    except _models.RegistryError as e:
        print(f"error: {e}", file=sys.stderr)
        return None


def _cmd_models(args: argparse.Namespace) -> int:
    reg = _load_or_report(args.models_toml)
    if reg is None:
        return 1
    headers = ("key", "id", "slot", "port", "build", "ctx")
    rows = [
        (key, m.id, m.slot, str(m.port), m.build, str(m.ctx))
        for key, m in reg.models.items()
    ]
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
        for i, h in enumerate(headers)
    ]

    def fmt(row: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(row, widths))

    print(fmt(headers))
    print(fmt(tuple("-" * w for w in widths)))
    for row in rows:
        print(fmt(row))
    return 0


def _backup_and_write(path: Path, old_content: str) -> Path:
    backup_dir = _wire.BACKUP_ROOT / date.today().isoformat()
    backup_dir.mkdir(parents=True, exist_ok=True)
    safe_name = str(path).lstrip("/").replace("/", "_")
    backup_path = backup_dir / safe_name
    backup_path.write_text(old_content)
    return backup_path


def _cmd_wire(args: argparse.Namespace) -> int:
    reg = _load_or_report(args.models_toml)
    if reg is None:
        return 1
    resolve_ctx = _wire.make_default_ctx_resolver(reg)

    for target in _wire.WIRE_TARGETS:
        before = _wire.read_existing(target.path)
        after = target.render(reg, before, resolve_ctx=resolve_ctx)
        if before == after:
            print(f"== {target.name}: no changes")
            continue
        if args.apply:
            backup_path = _backup_and_write(target.path, before)
            target.path.parent.mkdir(parents=True, exist_ok=True)
            target.path.write_text(after)
            print(f"== {target.name}: applied (backup: {backup_path})")
        else:
            print(f"== {target.name}: would change (dry-run; pass --apply to write)")
            diff = _wire.unified_diff(target.name, before, after)
            sys.stdout.write(diff if diff.endswith("\n") else diff + "\n")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    results = _doctor.run_doctor(args.models_toml, systemd_dir=args.systemd_dir)
    print(_doctor.format_table(results))
    return 0 if _doctor.all_ok(results) else 1


def _cmd_stub(args: argparse.Namespace) -> int:
    print(f"servedeck {args.command}: not wired yet (P3/P4)", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--models-toml",
        type=Path,
        default=default_models_toml(),
        help="path to models.toml (default: the repo root's, or $SERVEDECK_MODELS_TOML)",
    )

    parser = argparse.ArgumentParser(prog="servedeck", description="servedeck v2 control CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_models = sub.add_parser("models", parents=[common], help="list models from the registry")
    p_models.set_defaults(func=_cmd_models)

    p_wire = sub.add_parser("wire", parents=[common], help="generate/update client configs")
    p_wire.add_argument(
        "--apply", action="store_true", help="write changes (default: dry-run diff only)"
    )
    p_wire.set_defaults(func=_cmd_wire)

    p_doctor = sub.add_parser("doctor", parents=[common], help="check the registry against reality")
    p_doctor.add_argument(
        "--systemd-dir",
        type=Path,
        default=None,
        help="override ~/.config/systemd/user (for tests / other machines)",
    )
    p_doctor.set_defaults(func=_cmd_doctor)

    for name in STUB_COMMANDS:
        p_stub = sub.add_parser(name, help=f"(not wired yet) {name}")
        p_stub.set_defaults(func=_cmd_stub, command=name)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
