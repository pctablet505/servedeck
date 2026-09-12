#!/usr/bin/env bash
# builds/stock/check.sh -- read-only verification that the live venv still
# matches builds/stock/MANIFEST.toml. Never installs, never mutates the venv
# or any git tree. Prints one PASS/FAIL line per check; exits 0 iff all pass.
#
# stock is a plain PyPI wheel (no fork tree), so the tree/patch-drift checks
# that builds/qwen38next and builds/glm53 run are SKIPPED here, not failed --
# there is no source checkout to compare against.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$HERE/MANIFEST.toml"

usage() {
    cat <<'EOF'
usage: check.sh [--help]

Read-only check that the live "stock" venv (vllm==0.29.0, plain wheel) still
matches builds/stock/MANIFEST.toml: Python version, key package versions, the
vLLM version string, and that no direct_url.json has appeared (which would
mean this venv silently became an editable/source install).

Exits 0 if every check passes, 1 otherwise. Makes no network calls, touches
no GPU, and does not install or modify anything.
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

FAIL=0
pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; FAIL=1; }

# --- resolve venv path from the manifest -------------------------------------
VENV_PATH="$(python3 -c "
import tomllib
with open('$MANIFEST', 'rb') as f:
    print(tomllib.load(f)['venv']['path'])
")"
VENV="${VENV_PATH/#\~/$HOME}"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
    fail "venv python exists at $PY"
    echo "$FAIL checks failed" >&2
    exit 1
fi
pass "venv python exists at $PY"

# --- run one python process to gather everything comparable in one place ----
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

# direct_url.json presence for the vllm dist -- a wheel install has none.
out["vllm_direct_url"] = None
try:
    d = m.distribution("vllm")
    try:
        out["vllm_direct_url"] = d.read_text("direct_url.json")
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

read_manifest() {
    python3 -c "
import tomllib
with open('$MANIFEST', 'rb') as f:
    d = tomllib.load(f)
path = '$1'.split('.')
v = d
for p in path:
    v = v[p]
print(v)
"
}

read_actual() {
    python3 -c "
import json
d = json.loads('''$ACTUAL_JSON''')
v = d.get('$1')
print('' if v is None else v)
"
}

check_eq() {
    local label="$1" manifest_path="$2" actual_key="$3"
    local expected actual
    expected="$(read_manifest "$manifest_path")"
    actual="$(read_actual "$actual_key")"
    if [ "$expected" = "$actual" ]; then
        pass "$label: $actual"
    else
        fail "$label: manifest says $expected, venv has ${actual:-<missing>}"
    fi
}

check_eq "python version"        "venv.python"                 "python"
check_eq "vllm version"          "venv.vllm"                    "vllm"
check_eq "torch"                 "packages.torch"               "torch"
check_eq "transformers"          "packages.transformers"        "transformers"
check_eq "flashinfer-python"     "packages.flashinfer_python"   "flashinfer_python"
check_eq "triton"                "packages.triton"              "triton"
check_eq "numpy"                 "packages.numpy"                "numpy"

DIRECT_URL="$(read_actual vllm_direct_url)"
if [ -z "$DIRECT_URL" ]; then
    pass "vllm dist has no direct_url.json (still a plain wheel install)"
else
    fail "vllm dist NOW has a direct_url.json ($DIRECT_URL) -- this venv is no longer a plain wheel install; MANIFEST.toml is stale"
fi

echo "--- tree/patch checks: SKIPPED (stock has no source checkout; see [tree]/[patches] in MANIFEST.toml) ---"

if [ "$FAIL" -eq 0 ]; then
    echo "all checks passed"
else
    echo "one or more checks FAILED" >&2
fi
exit "$FAIL"
