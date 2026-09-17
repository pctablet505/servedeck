# Context-aware power management for an AI workstation — proposal

Status: **recommendation is NOT to build the thing that was asked for.** A smaller,
different thing is worth doing, and one measurement should decide even that.
Date: 2026-08-17. Evidence: 9-agent read-only investigation, no setting changed.

---

## 1. Verdict first

**Do not build a context-aware power-management daemon.** Three measured facts kill it.

**Idle is already handled by the silicon.** Sampled cores sit in the deepest C-state
**65.9–84.9%** of a 16,470 s uptime. Idle cores read *exactly* 624,194 kHz — the
hardware floor, reached with zero tuning. The GPU already drops its memory clock to
405 MHz and its PCIe link to Gen1 when unused. There is no idle CPU saving to recover
because nothing is failing to idle.

**A power manager is already running.** `power-profiles-daemon 0.30` is active and owns
EPP across all 32 CPUs (`balanced`, EPP `balance_performance` ×32). The complaint that
"the default OS mode is not context aware" is a complaint about *this daemon*. Anything
new either replaces it or drives it — it cannot ignore it, and in fact PPD will
overwrite direct EPP writes.

**The arithmetic does not support a daemon.** The cost handle is
**1 W recovered continuously = 8.76 kWh/yr ≈ ₹70/yr**. The largest idle saving anyone
could evidence was ~19 W of GPU memory-clock ramping — ₹1,140/yr — and it rests on four
minutes of contaminated sampling whose two windows disagreed by 33 W. An 8 W daemon
would be worth ₹560/yr and, given the noise floor below, could not even be *proven* to
work in under an hour of paired measurement.

---

## 2. The real finding, which is not what was asked about

**The context that matters is not idle-vs-busy. It is latency-sensitive vs
throughput-batch.**

Idle-vs-busy is already solved, and solved *faster than a daemon could poll*: the CPU
transitions in ~94 ms and the GPU in ~1.4 s. A control loop sampling at 1 Hz is
strictly worse than the hardware it would be second-guessing.

What the hardware genuinely cannot know is **whether a human is waiting.** Measured
during real mixed load, active cores ran **5325–5397 MHz against a 4300 MHz nominal** —
about **125% of nominal, the least efficient point on the voltage/frequency curve** —
because nothing distinguishes an overnight batch job from a keystroke a human is
blocking on. That is the one piece of context the OS is genuinely missing.

And the payoff for acting on it is mostly **not electricity**:

> Under sustained load the GPU sits at **598.76 W, 91 °C, 4 °C from its thermal limit,
> fan at 76%, SM clock 2565 of 3090 MHz, with `clocks_event_reasons.sw_power_cap =
> Active`.**

The card is **simultaneously power- and thermally-constrained**. It has no headroom to
surrender. Capping it for latency-insensitive work therefore costs performance that
*is not currently being delivered anyway*, and buys thermal margin, acoustics and
component life. Treat energy as the bonus, not the goal.

---

## 3. Where the power actually goes

Instrument: a **Corsair HX1200i** exposing USB-HID telemetry at `/sys/class/hwmon/hwmon3`
(`corsairpsu`), world-readable, ~2 ms per read. This was not known at the start and is
the single most valuable asset here — whole-system power is measurable with no root and
no external meter.

**Correction to an early reading of mine:** `power1` is **DC output** (≈ the rail sum),
not AC input. Verified over 152 samples at 4 Hz: median `total − railsum` = −1.0 W.
**Every watt below is DC; true AC draw is 8–12% higher** and all costs scale up
accordingly.

| state | DC total | GPU | non-GPU |
|---|---|---|---|
| quiet idle (desktop up) | 136–153 W | 21–27 W | ~110–128 W |
| GPU serving (701 samples, 6 min) | **752 W** (med 784) | **573 W** | ~179 W |
| 8 CPU cores pinned | ~280–288 W | quiet | ~130–160 W |

**A hard floor nobody had identified.** Across a 608 → 816 W swing in total power, the
small rails did not move at all:

```
                    total    +12V    +5V    +3.3V
lowest 20% of total  608.0   585.3   42.1    5.5
highest 20%          815.7   757.1   41.9    5.5
   +5V  sd = 0.9 W over 701 samples
   +3.3V sd = 0.1 W over 701 samples
```

**~47.4 W on +5V and +3.3V is invariant** — board logic, USB, NVMe, fans. That is about
**32% of the entire idle budget**, and no software lever in this proposal touches any of
it. It bounds how low idle can ever go.

**Baseline cost.** At an assumed duty cycle of 18 h idle / 4 h GPU-heavy / 2 h CPU-batch:
~**2,520 kWh/yr ≈ ₹20,150/yr**. Realistic total recoverable: **₹1,500–2,500/yr**.

---

## 4. Measurement reality — this is the binding constraint

Whole-system DC power has **sd ≈ 50 W** with an integrated autocorrelation time of
**~16 s** on a live box. Minimum detectable difference at 95% confidence:

| measurement time per arm | smallest provable effect |
|---|---|
| 10 min | ~22 W |
| 30 min | ~13 W |
| 60 min | ~9 W |

**Anything under ~10 W is effectively unfalsifiable here.** The instrument is not the
limit; workload variance is. This single table is why the daemon is a bad idea: most of
its claimed wins would be smaller than the noise it would be tuned against.

---

## 5. What can be controlled

All CPU sysfs knobs are root-owned `0644` and **none survive reboot** — no udev rule,
systemd unit, sysctl.d entry or kernel cmdline pins any of them.

| lever | range / current | root? | persists? | note |
|---|---|---|---|---|
| **PPD profile hold** (D-Bus `HoldProfile`) | performance / balanced / power-saver; now `balanced` | **no** — polkit `allow_active=yes` | auto-released on disconnect | Safest lever. Auto-release on crash *is* the safety property |
| **GPU power limit** `-pl` | 150–600 W; now 600 | yes | no, volatile | The key experiment. Card already `sw_power_cap = Active` |
| **CPU `scaling_max_freq`** | 624,194–5,756,452 kHz, 32 policies | yes | no | Hard ceiling, unlike EPP. Race-to-idle risk |
| **EPP** | 5 levels; now `balance_performance` | yes | no | **A hint, not a cap** — work still hit 5442–5687 MHz at the midpoint. PPD will overwrite you |
| `cpufreq/boost` | 0/1; now 1 | yes | no | Coarse version of `scaling_max_freq` |
| **GPU locked memory clock** `-lmc` | 405 / 810 / 7001 / 13365 / 14001 MHz | yes | no | The only knob targeting idle waste directly |

## 6. What cannot be controlled

| wanted | why not | nearest substitute |
|---|---|---|
| **CPU package power cap** | No `constraint_*` files in either RAPL zone. AMD RAPL is energy-*reporting* only; PPT/TDC/EDC live in the SMU, reachable only via BIOS | `scaling_max_freq` is the **only** way to bound CPU power |
| **CPU package energy reading** | Policy-blocked, not silicon-blocked: `energy_uj` is root-only, `perf_event_paranoid=4`, `/dev/cpu/*/msr` root-only. The PMUs *are* registered | **`CAP_PERFMON` on one helper binary** — highest-value action available, and it changes no power setting |
| **Per-process / per-cgroup power** | Architecturally impossible. cgroup v2 has no energy controller; NVML exposes per-process memory and SM util, never watts; RAPL is package-scoped even as root | cgroup `cpu.stat` + SM util as an explicitly-labelled *model*, never a measurement |
| **Sub-minute GPU vs non-GPU split by subtraction** | Refuted by measurement: regressing (total − NVML) on busy cores gives **R² = 0.008**, resid sd 60.9 W, and physically impossible **negative** power | Valid only on multi-minute steady-state averages |
| **DRAM / uncore / Infinity Fabric energy** | No `dram` or `psys` domain exists. 192 GB of DDR5 has no counter of any kind | None — permanently inside the ~110–128 W non-GPU bucket |
| **AC / wall power** | `curr1_label` exists but `curr1_input` does not | DC ÷ 0.89–0.92, or a ~₹2,000 plug meter (which would also calibrate everything else) |
| **GPU runtime power-down (D3cold)** | `Runtime D3: Disabled by default`; `runtime_suspended_time = 0` | none |

---

## 7. Recommended scope

**One afternoon of measurement, two static config changes, and one experiment that
decides whether ~150 lines of job wrapper ever get written. No daemon, no root service,
no polling control loop.**

**Phase 1 — measurement only, nothing changed (~2 h work, 1 day wall).**
Build `powersample` (~120 lines). On a **quiesced** box collect: an 8 h overnight idle
trace; a 10-minute DPMS test (blank the NVIDIA-attached panel and re-trace — if deep
idle goes to ~100%, the "desktop compositing holds the memory clock up" diagnosis is
confirmed *and* already captured overnight); and clean joules-per-job baselines for the
pytest suite and one training run. Decide the `CAP_PERFMON` question — without it the
110–128 W non-GPU bucket stays a black box.

**Phase 2 — the free static changes (~1 h).**
`systemctl disable nvidia-powerd` (provably inert here). Move the display to the iGPU
*only if* Phase 1 confirms the ramp diagnosis, and measure the net — the iGPU's draw
lands in the CPU package, so the net is not the full 19 W.

**Phase 3 — the one gating experiment (~1 afternoon).**
Sweep GPU `-pl` over 600/500/450/400/350/300 W on a fixed-step training job, measuring
**joules per job and seconds per job — never watts alone.** Same shape for a CPU
`scaling_max_freq` sweep against the pytest suite.

---

## 8. Pre-registered kill criteria

- **K1** — quiesced overnight trace shows GPU deep-idle ≥90%, or mean idle within 8 W of
  the 25.9 W floor → the 19 W was an artifact of a human at the desk. Drop it.
- **K2** — DPMS test does not restore deep idle → the mechanism is wrong; cancel the
  display-topology change.
- **K3** — display move nets < 10 W once the CPU package cost is counted → revert.
- **K4** — capping CPU frequency raises joules-per-suite, or costs >10% wall for <15%
  energy → **race-to-idle wins, the CPU lever is void.** Given 65.9–84.9% deep-idle
  residency this is the *expected* outcome.
- **K5** — the `-pl` sweep shows joules-per-job rising monotonically as the cap drops →
  no automation. Consider one static `-pl` for thermal reasons only.
- **K6** — Phases 1–3 together establish < ~₹1,000/yr → **do not write the daemon.**
  Apply the free static changes, archive the harness, close the project.
- **K7** — any intervention whose measured effect is below the detection floor for the
  time you are willing to spend measuring it.

---

## 9. Two things to act on regardless of this project

1. **The GPU runs at 91 °C with 4 °C of thermal margin and the fan at 76% under load.**
   That is worth attention on its own terms, independent of power. A static `-pl` in the
   450–500 W region is the cheapest intervention and costs little performance on a card
   that is *already* power-capped at its ceiling.
2. **Something is holding 50.5 GB of VRAM right now** (a vLLM server), alongside 24
   AlgoTrading worktrees on disk. Neither is a power problem, but both are worth knowing
   about.

---

## 10. Honest summary

The question assumed the machine wastes power at idle. It does not — the drivers and
silicon already do that job well, and a daemon would be re-litigating decisions made in
94 ms with a 1 s control loop. What the system genuinely cannot know is whether a human
is waiting, and the value of teaching it that is **thermal and acoustic first, monetary
second** — on the order of ₹1,500–2,500/yr against a ₹20,000/yr bill.

Build the measurement harness. Run one sweep. Let K1–K7 decide the rest.
