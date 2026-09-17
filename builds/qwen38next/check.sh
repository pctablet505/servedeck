#!/usr/bin/env bash
# builds/qwen38next/check.sh -- read-only verification that the live venv AND
# its source tree still match builds/qwen38next/MANIFEST.toml. Never installs,
# never mutates the venv, never touches the tree's index or working files
# (only `git diff` / `git diff --no-index`, both read-only). No GPU use.
#
# The important check is the last one: it re-derives the tracked/untracked
# patch content straight from the live tree and hashes it against the
# exported patch files in builds/patches/qwen38next/ -- so an un-exported edit
# to the fork checkout is caught here, not discovered months later.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
MANIFEST="$HERE/MANIFEST.toml"
PATCH_DIR="$REPO_ROOT/builds/patches/qwen38next"

usage() {
    cat <<'EOF'
usage: check.sh [--help]

Read-only check that the live "qwen38next" venv and its editable source tree
still match builds/qwen38next/MANIFEST.toml:
  - venv: python version, key package versions, vllm version
  - editable install target (direct_url.json) matches [tree].checkout
  - tree HEAD matches [tree].head
  - the tree's CURRENT tracked diff + untracked non-.bak/.reviewed files,
    hashed, equal the hash of builds/patches/qwen38next/working-tree-*.patch
    (drift detection: an edit to the checkout that was never re-exported)
  - the 4 topic branches named in MANIFEST.toml [patches].excluded_branches
    are still NOT merged into the tree's HEAD

Exits 0 if every check passes, 1 otherwise. Makes no network calls, touches
no GPU, and does not install, apply a patch, or modify the tree.
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

FAIL=0
pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; FAIL=1; }

manifest_get() {
    python3 -c "
import tomllib
with open('$MANIFEST', 'rb') as f:
    d = tomllib.load(f)
v = d
for p in '$1'.split('.'):
    v = v[p]
print(v)
"
}

VENV_PATH="$(manifest_get venv.path)"
VENV="${VENV_PATH/#\~/$HOME}"
PY="$VENV/bin/python"
TREE="$(manifest_get tree.checkout)"
MANIFEST_HEAD="$(manifest_get tree.head)"

if [ ! -x "$PY" ]; then
    fail "venv python exists at $PY"
    echo "$FAIL checks failed" >&2
    exit 1
fi
pass "venv python exists at $PY"

if [ ! -d "$TREE/.git" ]; then
    fail "tree is a git checkout at $TREE"
    echo "$FAIL checks failed" >&2
    exit 1
fi
pass "tree is a git checkout at $TREE"

# --- venv / package versions --------------------------------------------------
ACTUAL_JSON="$("$PY" - <<'EOF'
import json, importlib.metadata as m, sys

out = {"python": "%d.%d.%d" % sys.version_info[:3]}
for dist in ("transformers", "flashinfer-python", "triton", "numpy", "vllm"):
    key = dist.replace("-", "_")
    try:
        out[key] = m.version(dist)
    except m.PackageNotFoundError:
        out[key] = None

# torch's dist-info METADATA "Version:" field omits the +cuXXX local suffix
# (it says plain "2.13.0"); torch.__version__ at import time is the one that
# actually distinguishes a cu130 build from a CPU-only one, which is the
# whole point of recording it -- see MANIFEST.toml [packages].torch.
try:
    import torch
    out["torch"] = torch.__version__
except Exception:
    out["torch"] = None

out["direct_url"] = None
try:
    d = m.distribution("vllm")
    try:
        raw = d.read_text("direct_url.json")
        out["direct_url"] = json.loads(raw)["url"] if raw else None
    except Exception:
        pass
except m.PackageNotFoundError:
    pass

print(json.dumps(out))
EOF
)"

if [ -z "$ACTUAL_JSON" ]; then
    fail "venv python ran and reported versions"
    echo "$FAIL checks failed" >&2
    exit 1
fi
pass "venv python ran and reported versions"

actual_get() {
    python3 -c "
import json
d = json.loads('''$ACTUAL_JSON''')
v = d.get('$1')
print('' if v is None else v)
"
}

check_eq() {
    local label="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        pass "$label: $actual"
    else
        fail "$label: manifest says $expected, venv has ${actual:-<missing>}"
    fi
}

check_eq "python version"    "$(manifest_get venv.python)"                "$(actual_get python)"
check_eq "vllm version"      "$(manifest_get venv.vllm)"                   "$(actual_get vllm)"
check_eq "torch"             "$(manifest_get packages.torch)"              "$(actual_get torch)"
check_eq "transformers"      "$(manifest_get packages.transformers)"       "$(actual_get transformers)"
check_eq "flashinfer-python" "$(manifest_get packages.flashinfer_python)"  "$(actual_get flashinfer_python)"
check_eq "triton"            "$(manifest_get packages.triton)"             "$(actual_get triton)"
check_eq "numpy"             "$(manifest_get packages.numpy)"              "$(actual_get numpy)"

# --- editable target -----------------------------------------------------
EXPECTED_URL="file://$(manifest_get install.editable_target)"
ACTUAL_URL="$(actual_get direct_url)"
if [ "$EXPECTED_URL" = "$ACTUAL_URL" ]; then
    pass "editable install target: $ACTUAL_URL"
else
    fail "editable install target: manifest says $EXPECTED_URL, venv has ${ACTUAL_URL:-<missing>}"
fi

# --- tree HEAD -------------------------------------------------------------
ACTUAL_HEAD="$(git -C "$TREE" rev-parse HEAD)"
check_eq "tree HEAD" "$MANIFEST_HEAD" "$ACTUAL_HEAD"

# --- patch drift: tracked diff ---------------------------------------------
TRACKED_ACTUAL_HASH="$(git -C "$TREE" diff | sha256sum | cut -d' ' -f1)"
TRACKED_MANIFEST_HASH="$(sha256sum "$PATCH_DIR/working-tree-tracked.patch" | cut -d' ' -f1)"
if [ "$TRACKED_ACTUAL_HASH" = "$TRACKED_MANIFEST_HASH" ]; then
    pass "tracked-file diff matches working-tree-tracked.patch ($TRACKED_ACTUAL_HASH)"
else
    fail "tracked-file diff does NOT match working-tree-tracked.patch (live $TRACKED_ACTUAL_HASH vs exported $TRACKED_MANIFEST_HASH) -- the checkout was edited and not re-exported"
fi

# --- patch drift: untracked files (excluding *.bak*/*.reviewed scratch files) --
UNTRACKED_FILES="$(git -C "$TREE" status --porcelain=v1 \
    | awk '/^\?\? / {print substr($0,4)}' \
    | grep -vE '\.(bak[0-9]*|reviewed)$' || true)"

UNTRACKED_ACTUAL_PATCH=""
for f in $UNTRACKED_FILES; do
    UNTRACKED_ACTUAL_PATCH+="$(git -C "$TREE" diff --no-index -- /dev/null "$f")"$'\n'
done
UNTRACKED_ACTUAL_HASH="$(printf '%s' "$UNTRACKED_ACTUAL_PATCH" | sha256sum | cut -d' ' -f1)"
UNTRACKED_MANIFEST_HASH="$(sha256sum "$PATCH_DIR/working-tree-untracked.patch" | cut -d' ' -f1)"
if [ "$UNTRACKED_ACTUAL_HASH" = "$UNTRACKED_MANIFEST_HASH" ]; then
    pass "untracked-file diff matches working-tree-untracked.patch ($UNTRACKED_ACTUAL_HASH)"
else
    fail "untracked-file diff does NOT match working-tree-untracked.patch (live $UNTRACKED_ACTUAL_HASH vs exported $UNTRACKED_MANIFEST_HASH) -- a new/changed untracked file was not re-exported, or one was removed"
fi

# --- excluded topic branches are still NOT merged into HEAD -----------------
for b in jit-warmup-and-template-errors mm-encoder-offload ple-fp8-conversion prefix-match-unit-validation; do
    if git -C "$TREE" rev-parse --verify -q "refs/heads/$b" >/dev/null; then
        if git -C "$TREE" merge-base --is-ancestor "$b" HEAD 2>/dev/null; then
            fail "topic branch $b is now merged into HEAD but MANIFEST.toml still lists it as excluded"
        else
            pass "topic branch $b still unmerged (excluded, as manifest says)"
        fi
    else
        pass "topic branch $b no longer exists locally (nothing to exclude)"
    fi
done

if [ "$FAIL" -eq 0 ]; then
    echo "all checks passed"
else
    echo "one or more checks FAILED" >&2
fi
exit "$FAIL"
