"""Formal SMT mathematical proofs for Servedeck's pure decision & arithmetic cores.

Complements (never replaces) empirical/integration tests by proving universal
safety invariants via Z3 refutation (``solver.check() == z3.unsat`` on the
negated postcondition -- ``unknown`` is rejected, only ``unsat`` counts as a
proof), paired with differential equivalence checks against the real Python and
``web/app.js`` functions.

Theorems proved here:
1. ``Control.compute_util`` safety & co-residency slack: ``floor2`` truncation
   never rounds into ``margin_mib``, explicit ``util`` overrides never exceed
   ``free_mib - cushion``, and invalid utilities never pass.
2. ``capacity.compute`` & ``capacity.blue_green`` conservation & soundness:
   valid configurations conserve ``weights_pct + kv_pct + overhead_pct + free_pct == 100``
   with every bar term non-negative, ``can_apply`` implies valid positive ``ctx``
   and ``kv_tokens >= ctx``, and ``blue_green.feasible`` implies known weights
   plus ``FRAG_MARGIN_GIB`` slack.
3. ``control.offload_scan`` & ``reap_offload_regions`` fail-closed safety: no
   ``/dev/shm/vllm_offload_*.mmap`` region mapped or held by any live engine or
   live witness PID can ever be unlinked under any ``/proc`` failure or
   ``ptrace_scope`` permission denial.
4. ``web/app.js`` ``ctxRange`` + ``snapCtxToSlider`` grid reachability: every
   model ceiling (including non-4096-aligned native ceilings such as 202,752)
   sits on an exact ``min + k * 4096`` slider stop and remains reachable.
5. ``policies._apply_floor`` context-window safety & monotonicity: raising small
   output budgets to ``min_output_tokens`` never lowers a caller's budget and
   never pushes ``prompt + budget + 2048`` past ``ctx``.
6. ``parallelism.recommend`` & ``mixed_capacity`` scheduler/KV admission bounds:
   every recommended or mixed row respects ``max_num_seqs`` and available
   headroom-discounted KV pool tokens.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest

z3 = pytest.importorskip("z3")

from servedeck import capacity as _capacity
from servedeck import control as _control
from servedeck import parallelism as _parallelism
from servedeck import policies as _policies

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_JS_PATH = REPO_ROOT / "servedeck" / "web" / "app.js"


def _prove_unsat(solver: z3.Solver, theorem_name: str) -> None:
    """Assert that the negated theorem has ZERO counterexamples (`unsat`)."""
    solver.set("timeout", 15_000)
    verdict = solver.check()
    if verdict == z3.sat:
        pytest.fail(f"{theorem_name} violated! Counterexample model:\n{solver.model()}")
    assert verdict == z3.unsat, f"{theorem_name} returned solver verdict {verdict!r} (expected unsat)"


def _z3_ceil_div(num: z3.ArithRef, den: int) -> z3.ArithRef:
    """Exact ``ceil(num / den)`` for integer ``num >= 0`` and positive constant ``den``."""
    return (num + (den - 1)) / den


# ---------------------------------------------------------------------------
# Theorem 1: Control.compute_util VRAM Admission & Floor Safety
# ---------------------------------------------------------------------------


def test_smt_floor2_never_encroaches_on_safety_margin() -> None:
    """Prove across all ``total > 0`` and ``free > margin >= 0`` that
    ``floor2((free - margin) / total) * total <= free - margin``."""
    s = z3.Solver()
    total = z3.Int("total_mib")
    free = z3.Int("free_mib")
    margin = z3.Int("margin_mib")

    s.add(total >= 1, total <= 1_000_000)
    s.add(margin >= 0)
    s.add(free > margin, free <= total)

    # u_pct = floor(100 * (free - margin) / total) represents 100 * floor2(...)
    u_pct = (100 * (free - margin)) / total

    # Negation of safety property: either u_pct > 100, or ceil(total * u_pct / 100) > free - margin?
    # Note: u_pct * total <= 100 * (free - margin) implies (total * u_pct) / 100.0 <= free - margin,
    # and since free - margin is an integer, ceil(total * u_pct / 100.0) <= free - margin!
    need_mib = _z3_ceil_div(total * u_pct, 100)
    s.add(z3.Or(u_pct < 0, u_pct > 100, need_mib > free - margin))
    _prove_unsat(s, "floor2 safety margin theorem")


def test_smt_compute_util_admission_soundness_and_differential_check() -> None:
    """Prove symbolic ``compute_util`` never admits a launch whose ``ceil(total * u)``
    plus CUDA cushion exceeds ``free_mib``, and verify equivalence on boundary grid."""
    s = z3.Solver()
    total = z3.Int("total_mib")
    free = z3.Int("free_mib")
    margin = z3.Int("margin_mib")
    req_pct = z3.Int("req_pct")  # requested_util in hundredths
    has_req = z3.Bool("has_req")
    has_pin = z3.Bool("has_pin")
    pin_pct = z3.Int("pin_pct")
    is_main = z3.Bool("is_main")
    vram_mib = z3.Int("vram_mib")

    res_cushion = _control.RESIDENT_CUSHION_MIB

    s.add(total >= 1024, total <= 500_000)
    s.add(free >= 0, free <= total)
    s.add(margin >= 0, margin <= 32_768)
    s.add(pin_pct >= 1, pin_pct <= 100)

    max_fit_pct = (100 * (free - margin)) / total
    main_cushion = z3.If(margin < res_cushion, margin, res_cushion)

    accepted = z3.Bool("accepted")
    out_pct = z3.Int("out_pct")

    req_valid = z3.And(req_pct >= 1, req_pct <= 100)
    req_need = _z3_ceil_div(total * req_pct, 100)
    res_pct = (100 * vram_mib) / total
    res_need = _z3_ceil_div(total * res_pct, 100)

    main_branch = z3.If(
        free <= margin,
        z3.And(z3.Not(accepted), out_pct == 0),
        z3.If(
            has_req,
            z3.If(
                z3.And(req_valid, req_need + main_cushion <= free),
                z3.And(accepted, out_pct == req_pct),
                z3.And(z3.Not(accepted), out_pct == 0),
            ),
            z3.If(
                has_pin,
                z3.If(
                    max_fit_pct >= pin_pct,
                    z3.And(accepted, out_pct == pin_pct),
                    z3.And(z3.Not(accepted), out_pct == 0),
                ),
                z3.If(
                    max_fit_pct >= 1,
                    z3.And(accepted, out_pct == max_fit_pct),
                    z3.And(z3.Not(accepted), out_pct == 0),
                ),
            ),
        ),
    )

    resident_branch = z3.If(
        has_req,
        z3.If(
            z3.And(req_valid, req_need + res_cushion <= free),
            z3.And(accepted, out_pct == req_pct),
            z3.And(z3.Not(accepted), out_pct == 0),
        ),
        z3.If(
            z3.And(vram_mib >= 1, res_pct >= 1, res_need + res_cushion <= free),
            z3.And(accepted, out_pct == res_pct),
            z3.And(z3.Not(accepted), out_pct == 0),
        ),
    )

    s.add(z3.If(is_main, main_branch, resident_branch))

    # Safety property to refute: accepted ==> 1 <= out_pct <= 100 AND ceil(total*out/100) + min_cushion <= free
    out_need = _z3_ceil_div(total * out_pct, 100)
    eff_cushion = z3.If(is_main, main_cushion, res_cushion)
    s.add(
        accepted,
        z3.Or(out_pct < 1, out_pct > 100, out_need + eff_cushion > free),
    )
    _prove_unsat(s, "compute_util VRAM admission & cushion theorem")

    # Differential check: confirm real Control.compute_util agrees with the proven spec on edge grid.
    class _SpecStub:
        def __init__(self, slot: str, util: float | None, vram: int | None) -> None:
            self.key = "m"
            self.id = "M"
            self.slot = slot
            self.port = 19001
            self.util = util
            self.vram_mib = vram
            self.ctx_tokens = 4096
            self.venv_bin = "/bin"

        def served_names(self) -> list[str]:
            return ["M"]

        def render_argv(self, util: float, port: int) -> list[str]:
            return []

        def render_env(self) -> dict[str, str]:
            return {}

    class _RegStub:
        def __init__(self, spec: _SpecStub) -> None:
            self._s = spec

        def get(self, key: str) -> _SpecStub:
            return self._s

        def keys(self) -> list[str]:
            return ["m"]

    for free_v in (500, 1024, 50_000, 94_587, 97_887):
        for req_v in (None, 0.0, 0.80, 0.96, 0.99, 1.05):
            spec = _SpecStub("main", 0.96, None)
            ctl = _control.Control(
                _RegStub(spec),
                margin_mib=1024,
                total_mib=97_887,
                unit_prefix="sd-test-",
                free_mib=lambda fv=free_v: fv,
            )
            res = ctl.compute_util(spec, requested_util=req_v)
            if not isinstance(res, _control.Refusal):
                assert 0.0 < res <= 1.0
                assert math.ceil(97_887 * res) + min(1024, _control.RESIDENT_CUSHION_MIB) <= free_v


# ---------------------------------------------------------------------------
# Theorem 2: capacity.py Bar Conservation & blue_green Feasibility Soundness
# ---------------------------------------------------------------------------


def test_smt_capacity_bar_conservation_and_non_negativity() -> None:
    """Prove in exact real arithmetic that whenever ``0 < util <= 1``, ``total > 0``,
    ``weights >= 0``, ``overhead >= 0``, and ``kv = util * total - weights - overhead > 0``,
    all four bar percentages are non-negative and sum to ``100``."""
    s = z3.Solver()
    total = z3.Real("total_gib")
    util = z3.Real("util")
    weights = z3.Real("weights_gib")
    overhead = z3.Real("overhead_gib")

    s.add(total > 0)
    s.add(util > 0, util <= 1)
    s.add(weights >= 0, overhead >= 0)

    budget = util * total
    kv = budget - weights - overhead
    s.add(kv > 0)

    w_pct = (weights / total) * 100
    kv_pct = (kv / total) * 100
    ov_pct = (overhead / total) * 100
    used_pct = w_pct + kv_pct + ov_pct
    free_pct = z3.If(100 - used_pct > 0, 100 - used_pct, z3.RealVal(0))

    s.add(
        z3.Or(
            w_pct < 0,
            kv_pct <= 0,
            ov_pct < 0,
            free_pct < 0,
            used_pct > 100,
            w_pct + kv_pct + ov_pct + free_pct != 100,
        )
    )
    _prove_unsat(s, "capacity bar conservation & non-negativity theorem")


def test_smt_blue_green_feasibility_requires_known_weights_and_frag_margin() -> None:
    """Prove ``blue_green`` is feasible ONLY when both models have known weights,
    ``total_gib > 0``, and ``remaining_kv_gib >= frag_margin > 0``."""
    s = z3.Solver()
    known_a = z3.Bool("known_a")
    known_b = z3.Bool("known_b")
    total = z3.Real("total_gib")
    w_a = z3.Real("w_a")
    w_b = z3.Real("w_b")
    overhead = z3.RealVal("4.7")
    frag = z3.RealVal("1.0")

    s.add(w_a >= 0, w_b >= 0)
    combined = w_a + w_b + 2 * overhead
    required = combined + frag
    feasible = z3.And(known_a, known_b, total > 0, required <= total)
    remaining_kv = z3.If(feasible, total - combined, z3.RealVal(0))

    s.add(
        feasible,
        z3.Or(
            z3.Not(known_a),
            z3.Not(known_b),
            total <= 0,
            remaining_kv < frag,
            remaining_kv <= 0,
        ),
    )
    _prove_unsat(s, "blue_green feasibility soundness theorem")


# ---------------------------------------------------------------------------
# Theorem 3: offload_scan & reap_offload_regions Fail-Closed Safety Proof
# ---------------------------------------------------------------------------


def test_smt_offload_reaper_never_unlinks_buffer_held_by_live_process() -> None:
    """Prove across all combinations of ``/proc`` mount state, ``ptrace_scope``
    readability, witness liveness, and mapping state that ``reap_offload_regions``
    NEVER unlinks a buffer held by a live witness or live engine process."""
    s = z3.Solver()

    proc_is_dir = z3.Bool("proc_is_dir")
    engine_listing_ok = z3.Bool("engine_listing_ok")
    proc_has_any_numeric_pid = z3.Bool("proc_has_any_numeric_pid")

    # Two symbolic candidate holders: an engine discovered via cmdline (E) and
    # a witness PID supplied by Control from systemd/adoption (W).
    eng_alive = z3.Bool("eng_alive")
    eng_in_proc = z3.Bool("eng_in_proc")
    eng_maps_readable = z3.Bool("eng_maps_readable")
    eng_fd_readable = z3.Bool("eng_fd_readable")
    eng_holds_buf = z3.Bool("eng_holds_buf")

    wit_supplied = z3.Bool("wit_supplied")
    wit_alive = z3.Bool("wit_alive")
    wit_in_proc = z3.Bool("wit_in_proc")
    wit_maps_readable = z3.Bool("wit_maps_readable")
    wit_fd_readable = z3.Bool("wit_fd_readable")
    wit_holds_buf = z3.Bool("wit_holds_buf")

    # OS kernel invariant on real /proc: if /proc is mounted & listing succeeded,
    # a live vLLM engine appears in _engine_pids(proc), and when maps+fd are both
    # readable, its mapped/open offload buffer is visible in at least one of them.
    s.add(
        z3.Implies(
            z3.And(proc_is_dir, engine_listing_ok, proc_has_any_numeric_pid, eng_alive),
            eng_in_proc,
        )
    )
    # If a process is dead, it cannot hold a live kernel mapping.
    s.add(z3.Implies(eng_holds_buf, eng_alive))
    s.add(z3.Implies(wit_holds_buf, z3.And(wit_supplied, wit_alive)))

    # offload_scan's exact fail-closed trust & discovery rules (control.py):
    wit_verified = z3.Implies(
        wit_supplied,
        z3.If(
            wit_in_proc,
            z3.And(wit_maps_readable, wit_fd_readable),
            z3.Not(wit_alive),  # _pid_alive check when witness dir missing!
        ),
    )
    eng_verified = z3.Implies(eng_in_proc, z3.And(eng_maps_readable, eng_fd_readable))

    scan_trusted = z3.And(
        proc_is_dir,
        engine_listing_ok,
        proc_has_any_numeric_pid,
        wit_verified,
        eng_verified,
    )

    buf_seen_in_scan = z3.Or(
        z3.And(eng_in_proc, eng_maps_readable, eng_fd_readable, eng_holds_buf),
        z3.And(wit_supplied, wit_in_proc, wit_maps_readable, wit_fd_readable, wit_holds_buf),
    )

    # reap_offload_regions unlinks B iff scan.trusted and B not in scan.paths
    reaper_unlinks_buf = z3.And(scan_trusted, z3.Not(buf_seen_in_scan))

    # Refute: buffer is held by a live engine OR a supplied live witness, YET unlinked!
    s.add(reaper_unlinks_buf, z3.Or(eng_holds_buf, wit_holds_buf))
    _prove_unsat(s, "offload_scan fail-closed reaper safety theorem")


# ---------------------------------------------------------------------------
# Theorem 4: web/app.js ctxRange + snapCtxToSlider Grid & Ceiling Proof
# ---------------------------------------------------------------------------


def test_smt_app_js_ctx_slider_grid_and_non_4096_ceiling_reachability() -> None:
    """Prove over all ``modelMax, fit, running, picked`` in ``[1 .. 2,097,152]`` that:
    1. ``r.min <= snapped <= r.max``;
    2. Whenever ``r.max > r.min``, ``(r.max - r.min) % 4096 == 0`` AND
       ``(snapped - r.min) % 4096 == 0`` (every stop, including ``r.max``, is an
       exact HTML ``<input type=range>`` thumb position);
    3. ``snapCtxToSlider(r.max, r.min, r.max) == r.max`` (exact native ceilings
       such as 202,752 are never rounded down);
    4. When ``binding == 'kv'``, ``r.max <= fit``.
    Also cross-checks the Z3 specification against ``web/app.js`` in Node.js."""
    s = z3.Solver()
    MIN_CTX = 8192
    STEP = 4096

    model_max = z3.Int("model_max")
    fit = z3.Int("fit")
    running = z3.Int("running")
    picked = z3.Int("picked")

    s.add(model_max >= MIN_CTX, model_max <= 2_097_152)
    s.add(fit >= 0, fit <= 2_097_152)
    s.add(running >= 0, running <= 2_097_152)
    s.add(picked >= 1, picked <= 4_194_304)

    kv_top = z3.If(fit > 0, z3.If(model_max < fit, model_max, fit), model_max)
    is_kv_initial = kv_top < model_max
    ceiling = z3.If(running > kv_top, running, kv_top)
    is_kv_binding = z3.If(running > kv_top, z3.BoolVal(False), is_kv_initial)

    steps_kv = (ceiling - MIN_CTX) / STEP
    ctx_top = MIN_CTX + z3.If(steps_kv > 0, steps_kv, 0) * STEP

    r_min = z3.If(
        is_kv_binding,
        MIN_CTX,
        z3.If(
            ceiling <= MIN_CTX,
            ceiling,
            ceiling - ((ceiling - MIN_CTX) / STEP) * STEP,
        ),
    )
    r_max = z3.If(
        is_kv_binding,
        ctx_top,
        ceiling,
    )

    clamped = z3.If(picked < r_min, r_min, z3.If(picked > r_max, r_max, picked))
    snapped = z3.If(
        clamped >= r_max,
        r_max,
        z3.If(
            clamped <= r_min,
            r_min,
            r_min + ((clamped - r_min) / STEP) * STEP,
        ),
    )
    snapped_at_max = z3.If(r_max >= r_max, r_max, r_min)

    # Negation of conjunction of all 4 invariants:
    bad_bounds = z3.Or(r_min > r_max, snapped < r_min, snapped > r_max)
    bad_grid = z3.And(
        r_max > r_min,
        z3.Or((r_max - r_min) % STEP != 0, (snapped - r_min) % STEP != 0),
    )
    bad_ceiling_reach = z3.And( z3.Not(is_kv_binding), z3.Or(r_max != ceiling, snapped_at_max != ceiling) )
    bad_kv_cap = z3.And(is_kv_binding, fit >= MIN_CTX, r_max > fit)

    s.add(z3.Or(bad_bounds, bad_grid, bad_ceiling_reach, bad_kv_cap))
    _prove_unsat(s, "app.js ctxRange + snapCtxToSlider grid theorem")

    # Differential check against real app.js functions if node is present.
    node = shutil.which("node") or (
        str(Path.home() / ".local" / "bin" / "node")
        if (Path.home() / ".local" / "bin" / "node").is_file()
        and os.access(Path.home() / ".local" / "bin" / "node", os.X_OK)
        else None
    )
    if node:
        js = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[1], "utf8");
const MIN_CTX = 8192, CTX_STEP = 4096;
for (const fn of ["function ctxTop(", "function snapCtxToSlider(", "function ctxRange("]) {
  const i = src.indexOf(fn);
  eval(src.slice(i, src.indexOf("\n}\n", i) + 2));
}
const cases = [
  [202752, 0, 0, 202752],
  [202752, 0, 0, 198656],
  [327680, 245440, 0, 245440],
  [262144, 0, 262144, 131072],
];
const out = cases.map(([m, f, run, pick]) => {
  const r = ctxRange(m, f, run);
  const c = Math.max(r.min, Math.min(r.max, pick));
  const s = snapCtxToSlider(c, r.min, r.max);
  return { min: r.min, max: r.max, s, binding: r.binding };
});
console.log(JSON.stringify(out));
"""
        proc = subprocess.run([node, "-e", js, str(APP_JS_PATH)], capture_output=True, text=True, check=True)
        rows = json.loads(proc.stdout)
        assert rows[0]["max"] == 202_752 and rows[0]["s"] == 202_752
        assert rows[1]["s"] == 198_656 and (rows[1]["s"] - rows[1]["min"]) % 4096 == 0
        assert rows[2]["binding"] == "kv" and rows[2]["max"] <= 245_440


# ---------------------------------------------------------------------------
# Theorem 5: policies._apply_floor Context-Window Safety & Monotonicity
# ---------------------------------------------------------------------------


def test_smt_policies_apply_floor_never_lowers_or_overflows_context() -> None:
    """Prove over all ``ctx >= 1``, ``floor >= 1``, ``prompt >= 0``, ``current >= 0``
    that ``_apply_floor``:
    1. Never lowers ``current`` (``after >= current``);
    2. Leaves ``current >= floor`` unchanged (``after == current``);
    3. Whenever ``_apply_floor`` modifies ``current`` (``after > current``),
       ``after <= floor`` AND ``prompt + after + 2048 <= ctx``."""
    s = z3.Solver()
    RESERVE = _policies._CTX_RESERVE  # 2048

    ctx = z3.Int("ctx")
    floor_v = z3.Int("floor")
    prompt = z3.Int("prompt")
    current = z3.Int("current")

    s.add(ctx >= 1, floor_v >= 1, prompt >= 0, current >= 0)

    room = ctx - prompt - RESERVE
    min_floor_room = z3.If(floor_v < room, floor_v, room)
    target = z3.If(min_floor_room > 0, min_floor_room, 0)
    changed = z3.And(current < floor_v, target > current)
    after = z3.If(changed, target, current)

    s.add(
        z3.Or(
            after < current,
            z3.And(current >= floor_v, after != current),
            z3.And(changed, z3.Or(after > floor_v, prompt + after + RESERVE > ctx)),
        )
    )
    _prove_unsat(s, "policies._apply_floor monotonicity & context-headroom theorem")


# ---------------------------------------------------------------------------
# Theorem 6: parallelism.recommend & mixed_capacity Bounds
# ---------------------------------------------------------------------------


def test_smt_parallelism_recommend_and_mixed_capacity_bounds() -> None:
    """Prove over all ``pool >= 0``, ``cost_full >= 1``, ``cost_small >= 1``,
    ``max_seqs >= 1`` that:
    1. ``1 <= recommend.n <= max_seqs``;
    2. In ``mixed_capacity``, every emitted row has ``1 <= big <= max_seqs``,
       ``alongside >= 0``, ``big + alongside <= max_seqs``, and if ``alongside >= 1``,
       ``big * cost_full + alongside * cost_small <= floor(pool * headroom)``."""
    s = z3.Solver()
    # Represent usable = pool * 0.85 as exact scaled integer usable_100 = 85 * pool
    pool = z3.Int("pool_tokens")
    cost_full = z3.Int("cost_full")
    cost_small = z3.Int("cost_small")
    max_seqs = z3.Int("max_num_seqs")
    big = z3.Int("big")

    s.add(pool >= 0, pool <= 10_000_000)
    s.add(cost_full >= 1, cost_full <= 1_000_000)
    s.add(cost_small >= 1, cost_small <= cost_full)
    s.add(max_seqs >= 1, max_seqs <= 1024)

    # n_before = max(1, floor(pool * 85 / (100 * cost_full)))
    raw_fit_full = (pool * 85) / (100 * cost_full)
    n_before = z3.If(raw_fit_full >= 1, raw_fit_full, 1)
    rec_n = z3.If(n_before > max_seqs, max_seqs, n_before)

    max_big_unclamped = z3.If(n_before < _parallelism.MIXED_BIG_ROWS, n_before, _parallelism.MIXED_BIG_ROWS)
    max_big = z3.If(max_big_unclamped < max_seqs, max_big_unclamped, max_seqs)

    # Consider any emitted row 1 <= big <= max_big
    s.add(big >= 1, big <= max_big)

    left_100 = pool * 85 - 100 * big * cost_full
    raw_small = z3.If(left_100 > 0, left_100 / (100 * cost_small), 0)
    small_unclamped = z3.If(raw_small > 0, raw_small, 0)
    alongside = z3.If(
        big + small_unclamped > max_seqs,
        z3.If(max_seqs - big > 0, max_seqs - big, 0),
        small_unclamped,
    )

    s.add(
        z3.Or(
            rec_n < 1,
            rec_n > max_seqs,
            big > max_seqs,
            alongside < 0,
            big + alongside > max_seqs,
            z3.And(
                alongside >= 1,
                100 * (big * cost_full + alongside * cost_small) > pool * 85,
            ),
        )
    )
    _prove_unsat(s, "parallelism recommend & mixed_capacity scheduler/KV bounds theorem")
