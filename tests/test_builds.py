"""Tests for builds/ — REDESIGN-2026-09-12.md §2.6, packet P5.

Three kinds of coverage, all offline (no network, no GPU):

1. Schema: every builds/<name>/MANIFEST.toml parses and has the tables/keys
   every consumer (check.sh, build.sh, a future packet) can rely on.
2. Cross-reference: every ``[models.*].build`` in models.toml resolves to a
   manifest directory, and every patch file a manifest names actually exists
   on disk under builds/patches/<name>/.
3. Executability: ``check.sh --help`` and ``build.sh --dry-run`` run and exit
   0 for all three builds -- these are the entry points a later packet (or a
   human) is expected to be able to call unconditionally, so a break here is
   a break for everyone downstream.

check.sh's own full read-only verification against the LIVE venvs (package
versions, tree HEAD, patch-drift hashing) is exercised by hand, not by this
suite -- it depends on this specific box's venvs and fork checkouts existing,
which a clean checkout or CI runner does not have.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import tomllib

from servedeck import models

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
BUILDS_DIR = ROOT / "builds"
PATCHES_DIR = BUILDS_DIR / "patches"
MODELS_TOML = ROOT / "models.toml"

BUILD_NAMES = sorted(
    p.name
    for p in BUILDS_DIR.iterdir()
    if p.is_dir() and p.name != "patches" and (p / "MANIFEST.toml").is_file()
)

REQUIRED_TOP_LEVEL_TABLES = {
    "venv",
    "packages",
    "install",
    "tree",
    "base",
    "patches",
    "cuda",
}
REQUIRED_VENV_KEYS = {"path", "python", "vllm"}
REQUIRED_PACKAGE_KEYS = {
    "torch",
    "transformers",
    "flashinfer_python",
    "triton",
    "numpy",
}
REQUIRED_INSTALL_KEYS = {"kind", "editable_target", "source"}
REQUIRED_TREE_KEYS = {"checkout", "remote", "branch", "head", "describe"}
REQUIRED_BASE_KEYS = {"anchor", "commits_ahead", "note"}
REQUIRED_PATCHES_KEYS = {"files", "excluded_branches", "note"}
REQUIRED_CUDA_KEYS = {
    "driver_header",
    "nvcc",
    "flashinfer_sampler",
    "lib64_shim",
}


def _load_manifest(name: str) -> dict:
    with (BUILDS_DIR / name / "MANIFEST.toml").open("rb") as f:
        return tomllib.load(f)


def test_at_least_the_three_documented_builds_exist() -> None:
    assert {"stock", "qwen38next", "glm53"} <= set(BUILD_NAMES)


@pytest.mark.parametrize("name", BUILD_NAMES)
class TestManifestSchema:
    def test_top_level_tables_present(self, name: str) -> None:
        manifest = _load_manifest(name)
        missing = REQUIRED_TOP_LEVEL_TABLES - manifest.keys()
        assert not missing, f"{name}: missing top-level table(s) {missing}"

    def test_venv_table(self, name: str) -> None:
        venv = _load_manifest(name)["venv"]
        missing = REQUIRED_VENV_KEYS - venv.keys()
        assert not missing, f"{name}: [venv] missing {missing}"
        assert isinstance(venv["path"], str) and venv["path"]
        assert isinstance(venv["python"], str) and venv["python"]
        assert isinstance(venv["vllm"], str) and venv["vllm"]

    def test_packages_table(self, name: str) -> None:
        packages = _load_manifest(name)["packages"]
        missing = REQUIRED_PACKAGE_KEYS - packages.keys()
        assert not missing, f"{name}: [packages] missing {missing}"
        for key, value in packages.items():
            assert isinstance(value, str) and value, f"{name}: packages.{key}"

    def test_install_table(self, name: str) -> None:
        install = _load_manifest(name)["install"]
        missing = REQUIRED_INSTALL_KEYS - install.keys()
        assert not missing, f"{name}: [install] missing {missing}"
        assert install["kind"] in ("wheel", "editable-source"), (
            f"{name}: install.kind must be 'wheel' or 'editable-source', "
            f"got {install['kind']!r}"
        )
        if install["kind"] == "editable-source":
            assert install["editable_target"], (
                f"{name}: editable-source build must set install.editable_target"
            )

    def test_tree_table(self, name: str) -> None:
        manifest = _load_manifest(name)
        tree = manifest["tree"]
        missing = REQUIRED_TREE_KEYS - tree.keys()
        assert not missing, f"{name}: [tree] missing {missing}"
        # A wheel build has no source tree; a source build must name one.
        if manifest["install"]["kind"] == "editable-source":
            assert tree["checkout"] and tree["head"], (
                f"{name}: editable-source build must set tree.checkout and tree.head"
            )

    def test_base_table(self, name: str) -> None:
        base = _load_manifest(name)["base"]
        missing = REQUIRED_BASE_KEYS - base.keys()
        assert not missing, f"{name}: [base] missing {missing}"
        assert base["anchor"], f"{name}: base.anchor must be set (a tag, or 'undetermined'/'n/a')"

    def test_patches_table(self, name: str) -> None:
        patches = _load_manifest(name)["patches"]
        missing = REQUIRED_PATCHES_KEYS - patches.keys()
        assert not missing, f"{name}: [patches] missing {missing}"
        assert isinstance(patches["files"], list)
        assert isinstance(patches["excluded_branches"], list)

    def test_cuda_table(self, name: str) -> None:
        cuda = _load_manifest(name)["cuda"]
        missing = REQUIRED_CUDA_KEYS - cuda.keys()
        assert not missing, f"{name}: [cuda] missing {missing}"
        for key, value in cuda.items():
            assert isinstance(value, str) and value, f"{name}: cuda.{key}"


@pytest.mark.parametrize("name", BUILD_NAMES)
def test_every_listed_patch_file_exists(name: str) -> None:
    manifest = _load_manifest(name)
    for rel in manifest["patches"]["files"]:
        path = PATCHES_DIR / name / rel
        assert path.is_file(), f"{name}: MANIFEST.toml lists patch file {rel!r}, not found at {path}"


def test_every_models_toml_build_has_a_manifest() -> None:
    """REDESIGN §2.6: every [models.*].build must resolve to builds/<name>/."""
    registry = models.load(MODELS_TOML)
    used_builds = {m.build for m in registry.models.values()}
    missing = {b for b in used_builds if not (BUILDS_DIR / b / "MANIFEST.toml").is_file()}
    assert not missing, f"models.toml uses build(s) with no manifest: {missing}"
    # And the reverse should hold too: every [builds] entry in models.toml
    # names a real build directory (keeps the registry and builds/ in sync).
    missing_dirs = {
        b for b in registry.builds if not (BUILDS_DIR / b / "MANIFEST.toml").is_file()
    }
    assert not missing_dirs, f"models.toml [builds] names build(s) with no manifest: {missing_dirs}"


@pytest.mark.parametrize("name", BUILD_NAMES)
def test_check_sh_help_exits_zero(name: str) -> None:
    script = BUILDS_DIR / name / "check.sh"
    assert script.is_file(), f"{name}: check.sh missing"
    result = subprocess.run(
        [str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{name}: check.sh --help exited {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "usage" in result.stdout.lower()


@pytest.mark.parametrize("name", BUILD_NAMES)
def test_build_sh_dry_run_exits_zero(name: str) -> None:
    script = BUILDS_DIR / name / "build.sh"
    assert script.is_file(), f"{name}: build.sh missing"
    result = subprocess.run(
        [str(script), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{name}: build.sh --dry-run exited {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # A dry run must never actually invoke a network/build tool -- it can
    # only ever print the plan. The strongest cheap proxy for "nothing ran"
    # is that no real target directory (which does not exist) was created.
    assert "dry" in result.stdout.lower() or "+" in result.stdout


@pytest.mark.parametrize("name", BUILD_NAMES)
def test_build_sh_dry_run_never_touches_the_filesystem(name: str, tmp_path) -> None:
    """--dry-run with an explicit target must not create that target."""
    script = BUILDS_DIR / name / "build.sh"
    target = tmp_path / "would-be-new-checkout"
    result = subprocess.run(
        [str(script), "--dry-run", str(target)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert not target.exists(), (
        f"{name}: build.sh --dry-run created {target} -- a dry run must only print"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
