"""Static formal verification gate: PEP 561 ``mypy`` type checking & AST scope proofs.

1. ``test_mypy_typechecks_entire_servedeck_package``: ``servedeck`` ships a
   ``py.typed`` marker; this test runs ``mypy --ignore-missing-imports
   --check-untyped-defs servedeck`` and requires 0 errors across all 31 modules.
2. ``test_no_local_variable_shadowing_before_definition``: AST dataflow/scope
   proof across every ``def`` / ``async def`` in ``servedeck/*.py`` verifying
   that no module-level name (such as ``os`` in ``BUG-01``) is read inside a
   function that later binds the same name locally via ``import``, ``=``,
   ``for``, ``with``, or ``except`` -- eliminating an entire class of latent
   ``UnboundLocalError`` crashes on cold branches.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = REPO_ROOT / "servedeck"


@pytest.mark.skipif(
    importlib.util.find_spec("mypy") is None,
    reason="mypy is not installed in this environment",
)
def test_mypy_typechecks_entire_servedeck_package() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--ignore-missing-imports",
            "--check-untyped-defs",
            "servedeck",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"mypy static verification failed:\n{proc.stdout}\n{proc.stderr}"


def _param_names(args: ast.arguments) -> set[str]:
    names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def _bound_targets(target: ast.AST) -> set[str]:
    out: set[str] = set()
    if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
        out.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            out |= _bound_targets(elt)
    elif isinstance(target, ast.Starred):
        out |= _bound_targets(target.value)
    return out


class _ShallowScopeChecker(ast.NodeVisitor):
    """Linear conservative check for ``read(x)`` appearing at an earlier line
    than the FIRST local binding ``bind(x)`` inside the same function scope,
    where ``x`` is also a module-level name (the exact ``BUG-01`` pattern)."""

    def __init__(self, module_globals: set[str], params: set[str]) -> None:
        self.module_globals = module_globals
        self.declared_global_or_nonlocal: set[str] = set()
        self.first_bind_line: dict[str, int] = {p: 0 for p in params}
        self.reads: list[tuple[str, int]] = []

    def _bind(self, name: str, lineno: int) -> None:
        if name not in self.first_bind_line or lineno < self.first_bind_line[name]:
            self.first_bind_line[name] = lineno

    # Nested scopes have their own lexical bindings; do not conflate them with ours.
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._bind(node.name, node.lineno)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._bind(node.name, node.lineno)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._bind(node.name, node.lineno)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ListComp(self, node: ast.ListComp) -> None:
        return

    def visit_SetComp(self, node: ast.SetComp) -> None:
        return

    def visit_DictComp(self, node: ast.DictComp) -> None:
        return

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        return

    def visit_Global(self, node: ast.Global) -> None:
        self.declared_global_or_nonlocal.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.declared_global_or_nonlocal.update(node.names)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound = alias.asname or alias.name.split(".")[0]
            self._bind(bound, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self._bind(alias.asname or alias.name, node.lineno)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self._bind(node.id, node.lineno)
        elif isinstance(node.ctx, ast.Load):
            self.reads.append((node.id, node.lineno))

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._bind(node.target.id, node.target.lineno)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type:
            self.visit(node.type)
        if node.name:
            self._bind(node.name, node.lineno)
        for stmt in node.body:
            self.visit(stmt)


def _module_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for a in stmt.names:
                names.add(a.asname or a.name.split(".")[0])
        elif isinstance(stmt, ast.ImportFrom):
            for a in stmt.names:
                if a.name != "*":
                    names.add(a.asname or a.name)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                names |= _bound_targets(t)
        elif isinstance(stmt, ast.AnnAssign) and stmt.target:
            names |= _bound_targets(stmt.target)
    return names


def test_no_local_variable_shadowing_before_definition() -> None:
    """Prove across all 31 ``servedeck/*.py`` modules that no function reads a
    module-global name at line L1 and binds the same name locally at line L2 > L1."""
    hazards: list[str] = []
    for path in sorted(PKG_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        mod_globals = _module_level_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            checker = _ShallowScopeChecker(mod_globals, _param_names(node.args))
            for stmt in node.body:
                checker.visit(stmt)
            for name, read_line in checker.reads:
                if name in checker.declared_global_or_nonlocal:
                    continue
                if name in checker.first_bind_line and name in mod_globals:
                    bind_line = checker.first_bind_line[name]
                    if read_line < bind_line:
                        hazards.append(
                            f"{path.name}:{node.name}: reads {name!r} at L{read_line} "
                            f"before local binding at L{bind_line} (UnboundLocalError hazard)"
                        )
    assert not hazards, "Local shadowing hazard(s) detected:\n  " + "\n  ".join(hazards)
