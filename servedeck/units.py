"""Thin wrappers over ``systemd-run --user`` / ``systemctl --user`` /
``journalctl --user`` — the v2 process-lifecycle substrate
(REDESIGN-2026-09-12.md §1 R4, §2.2).

Everything here is a *translation layer only*: build an argument list, run it,
parse the bytes back. No policy, no waiting on models, no GPU knowledge. That
lives in :mod:`servedeck.control`.

Rules this module keeps:

* **Never** ``shell=True``. Every call is an explicit ``list[str]`` argv, so a
  model id, a description or an env value can never be re-parsed as shell.
* Every function takes an injectable ``run: Runner`` (``(argv) ->
  CompletedProcess[str]``) so the unit tests assert on the exact argv without
  a systemd anywhere near them.
* Only two unit-name shapes are allowed (see :data:`_NAME_RE`): ``model-<key>``
  for real models and ``sd-test-<name>`` for the test suite's own transient
  units. A name that does not match is refused *before* any process is spawned
  — this is what keeps the suite from ever touching ``qwen-vllm``,
  ``lfm2-350m``, a ``*-reasoning-proxy`` or ``servedeck`` itself.

-------------------------------------------------------------------------
The ``--collect`` trap (measured on this box, systemd 259)
-------------------------------------------------------------------------
``--collect`` sets ``CollectMode=inactive-or-failed``: systemd unloads the
transient unit as soon as it goes inactive **including when it failed**. So
after a crash::

    systemctl --user show -p LoadState,ActiveState,Result,NRestarts sd-test-fail
    LoadState=not-found
    ActiveState=inactive
    Result=success          # <-- a LIE: these are the defaults for an
    NRestarts=0             #     unknown unit, not this unit's outcome

``systemctl show`` exits 0 for a unit it has never heard of and prints the
*default* value of every property asked for. That means the obvious failure
predicate — ``Result != "success" or ActiveState == "failed"`` — reads
**clean** for a unit that crashed and was collected. A caller that only looks
at :func:`show` will conclude "stopped normally" every single time a model
dies. Broken measurements fail downward.

The defence, used by :mod:`servedeck.control`: pair every :func:`show` with
:func:`gone` (``LoadState``, confirmed). *Unit vanished while we were waiting
for it* is a failure, not a clean stop. The journal survives collection, so the
diagnosis still comes from :func:`journal_tail`.

...and ``LoadState`` itself needs confirming, because a single read of it is
not reliable: see :func:`exists` for the measurement and :func:`gone` for the
predicate to actually use.
"""

from __future__ import annotations

import os
import re
import select
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "Runner",
    "UnitError",
    "UnitState",
    "JournalStream",
    "valid_unit_name",
    "default_runner",
    "start_transient",
    "stop",
    "show",
    "exists",
    "gone",
    "properties",
    "control_group",
    "list_model_units",
    "list_units",
    "list_units_argv",
    "manager_environment_names",
    "journal_tail",
    "journal_since",
    "journal_since_argv",
    "journal_follow",
    "start_transient_argv",
    "stop_argv",
    "show_argv",
    "list_model_units_argv",
    "journal_tail_argv",
    "journal_follow_argv",
    "SHOW_PROPERTIES",
    "DEFAULT_RESTART",
    "DEFAULT_RESTART_SEC",
    "DEFAULT_TIMEOUT_STOP_SEC",
    "DEFAULT_START_LIMIT_INTERVAL_SEC",
    "DEFAULT_START_LIMIT_BURST",
    "DEFAULT_MEMORY_MAX",
]

# --------------------------------------------------------------------------
# Types and constants
# --------------------------------------------------------------------------

#: A runner takes a complete argv and returns a finished process. Injected so
#: tests can assert the argv and return canned output.
Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]

#: Default subprocess timeout for the quick introspection calls.
DEFAULT_TIMEOUT_S = 10.0
#: `systemctl stop` on a vLLM unit has to wait for the process to die.
DEFAULT_STOP_TIMEOUT_S = 120.0

#: The only unit names this module will ever act on. ``model-*`` is v2's real
#: namespace; ``sd-test-*`` is reserved for the test suite. Anything else —
#: ``servedeck``, ``qwen-vllm``, ``lfm2-350m``, ``*-reasoning-proxy`` — is
#: refused, so no code path here can reach a unit somebody else owns.
_NAME_RE = re.compile(r"^model-[a-z0-9-]+$|^sd-test-[a-z0-9-]+$")

#: Values systemd accepts for ``Restart=``. Validated so a typo becomes a
#: refusal instead of a unit that silently never restarts.
_RESTART_VALUES = frozenset(
    {"no", "always", "on-success", "on-failure", "on-abnormal", "on-abort", "on-watchdog"}
)

#: A POSIX environment variable name. Anything else in ``unset_env`` is a
#: caller bug, and passing it through would make systemd reject the whole
#: unit — i.e. turn a typo in a redaction list into a model that will not boot.
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Launch defaults, declared ONCE. They were spelled out in both
#: `start_transient_argv` and `start_transient`, which meant changing one left
#: the other silently winning — the argv builder's value is dead, because the
#: wrapper always passes its own. A drift that no test could see.
#: ``always``, not ``on-failure``, and this is measured, not stylistic: when
#: vLLM's engine core dies, the watchdog sets ``server.should_exit`` and the
#: API server returns from ``serve_http`` normally, so the process exits **0**
#: (fork: ``vllm/entrypoints/launcher.py`` watchdog_loop + terminate_if_errored;
#: recorded in LOCAL_LLM_SETUP.md:294-298 from a live kill of EngineCore).
#: systemd reads 0 as a clean run, so ``on-failure`` fires zero times and a
#: model whose engine died stays down until somebody notices. v1's unit carried
#: ``Restart=always`` for exactly this reason. The start limit below is what
#: keeps ``always`` from crash-looping a model that cannot boot at all.
DEFAULT_RESTART = "always"
DEFAULT_RESTART_SEC = 10
#: Long enough for vLLM's own ``--shutdown-timeout`` drain plus the unwind of
#: pinned host memory: ``cudaHostUnregister`` on a 40 GiB KV-offload buffer and
#: the unlink of ``/dev/shm/vllm_offload_*.mmap`` happen inside the engine's
#: shutdown, and systemd killing the unit first is what orphaned 40 GiB of
#: host RAM on every restart. Must exceed the model's own shutdown timeout.
DEFAULT_TIMEOUT_STOP_SEC = 120
#: See `start_transient_argv`: systemd's own 5-per-10s limit can never fire
#: with a 10s restart delay, so a model that cannot boot restarts forever.
DEFAULT_START_LIMIT_INTERVAL_SEC = 300
DEFAULT_START_LIMIT_BURST = 3
#: Host-RAM ceiling for a model unit (hardware audit RAM-02, 2026-10-04). Above
#: the largest measured model peak (GLM: 168 GiB at 110 GiB of offload,
#: models.toml) and below the box's 182 GiB, so a leaking or over-offloaded
#: engine is OOM-killed inside its own cgroup instead of the kernel killing the
#: desktop. There is no swap here, so MemoryMax= is the whole limit.
DEFAULT_MEMORY_MAX = "172G"

#: Exactly the properties :func:`show` asks for, in the documented order.
SHOW_PROPERTIES = "ActiveState,SubState,Result,NRestarts,MainPID,ExecMainStartTimestamp"


class UnitError(RuntimeError):
    """A systemd command failed, or was refused before being run."""


@dataclass(frozen=True)
class UnitState:
    """The parse of ``systemctl --user show -p <SHOW_PROPERTIES> <name>``.

    ``load_state`` is NOT part of that call (see the ``--collect`` trap in the
    module docstring) — it is None unless a caller filled it in from
    :func:`exists` / :func:`properties`. Treat ``result``/``active_state`` as
    meaningful only when you have independently established that the unit is
    still loaded.
    """

    active_state: str
    sub_state: str
    result: str
    n_restarts: int
    main_pid: int
    exec_main_start_ts: str
    load_state: str | None = None

    @property
    def active(self) -> bool:
        return self.active_state == "active"

    @property
    def failed(self) -> bool:
        """True for a *still-loaded* unit that systemd considers failed.

        Deliberately narrow: for a collected unit this is False, because the
        values it is computed from are systemd's defaults rather than this
        unit's outcome. Pair it with :func:`exists`.
        """
        return self.active_state == "failed" or (self.result not in ("success", ""))


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def default_runner(timeout_s: float = DEFAULT_TIMEOUT_S) -> Runner:
    """A :data:`Runner` that actually spawns the process.

    Captures both streams as text and never raises on a nonzero exit — each
    wrapper below decides what a nonzero exit means. A missing binary or a
    timeout is surfaced as :class:`UnitError` because those are environment
    faults, not unit outcomes.
    """

    def _run(argv: Sequence[str]) -> "subprocess.CompletedProcess[str]":
        try:
            return subprocess.run(
                list(argv),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except FileNotFoundError as exc:  # systemctl/journalctl not installed
            raise UnitError(f"{argv[0]}: not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise UnitError(f"{' '.join(argv)}: timed out after {timeout_s}s") from exc

    return _run


def _checked(run: Runner, argv: Sequence[str]) -> "subprocess.CompletedProcess[str]":
    proc = run(argv)
    if proc.returncode != 0:
        raise UnitError(
            f"{' '.join(argv)} exited {proc.returncode}: "
            f"{(proc.stderr or proc.stdout or '').strip()}"
        )
    return proc


def valid_unit_name(name: str) -> bool:
    """True iff ``name`` is one this module is allowed to touch."""
    return bool(_NAME_RE.match(name))


def _require_name(name: str) -> str:
    if not valid_unit_name(name):
        raise UnitError(
            f"refusing to act on unit {name!r}: only 'model-<key>' and "
            f"'sd-test-<name>' (lowercase, digits, dashes) are allowed"
        )
    return name


# --------------------------------------------------------------------------
# systemd-run --user
# --------------------------------------------------------------------------


def start_transient_argv(
    name: str,
    argv: Sequence[str],
    env: Mapping[str, str],
    cwd: str | os.PathLike[str],
    restart: str = DEFAULT_RESTART,
    restart_sec: int = DEFAULT_RESTART_SEC,
    description: str | None = None,
    unset_env: Sequence[str] = (),
    start_limit_interval_sec: int = DEFAULT_START_LIMIT_INTERVAL_SEC,
    start_limit_burst: int = DEFAULT_START_LIMIT_BURST,
    timeout_stop_sec: int = DEFAULT_TIMEOUT_STOP_SEC,
    memory_max: str = DEFAULT_MEMORY_MAX,
) -> list[str]:
    """The exact argv :func:`start_transient` runs. Shape::

        systemd-run --user
                    --unit=<name>
                    --collect
                    --description=<description>
                    -p Restart=<restart>
                    -p RestartSec=<restart_sec>
                    -p StartLimitIntervalSec=<start_limit_interval_sec>
                    -p StartLimitBurst=<start_limit_burst>
                    -p TimeoutStopSec=<timeout_stop_sec>
                    -p MemoryMax=<memory_max>
                    -p WorkingDirectory=<cwd>
                    -p UnsetEnvironment=<NAME>   (one per name, sorted)
                    --setenv=<K>=<V>             (one per var, keys sorted)
                    --
                    <argv[0]> <argv[1]> ...

    ``--`` terminates option parsing so a model command may contain anything.
    Env keys and unset names are sorted purely so the argv is deterministic
    and assertable.

    **The start limit is not optional.** systemd's defaults are
    ``StartLimitIntervalSec=10s`` with ``StartLimitBurst=5``, and
    ``RestartSec=10`` puts each retry *outside* that 10-second window — so the
    ceiling can never be reached and a model that cannot boot at all restarts
    every ten seconds forever, unattended, holding a port and taking the GPU
    each time. That is R4's "the unit crash-loops at boot" exactly. 3 attempts
    per 300 s makes the unit reach ``failed`` after ~30 s of trying, which is
    what lets :meth:`control.Control.wait_ready` report a real failure instead
    of timing out against an eternally-restarting unit.

    ``unset_env`` exists because a transient unit inherits the **user
    manager's** environment, which on a workstation is the login session's —
    and that is where an operator's exported API keys end up (measured on this
    box: ``KITE_API_KEY`` and ``KITE_API_SECRET`` are in
    ``systemctl --user show-environment``, so every transient unit would
    inherit them). A model process has no business holding a credential for an
    unrelated service: it runs arbitrary user prompts, it logs, and it can be
    asked to print its own environment.

    systemd applies ``UnsetEnvironment=`` **after** the environment is
    assembled from the manager's, ``Environment=`` and ``--setenv``, so it
    beats everything — which is why naming a variable in both ``env`` and
    ``unset_env`` is refused here rather than silently resolved. Verified on
    this box (systemd 259): with the property, the variable is absent from the
    unit's ``/proc/<MainPID>/environ`` and from what ``/usr/bin/env`` logs to
    the journal, while ``HOME`` is still there.
    """
    _require_name(name)
    if not argv:
        raise UnitError(f"{name}: refusing to start a unit with an empty command")
    if restart not in _RESTART_VALUES:
        raise UnitError(
            f"{name}: Restart={restart!r} is not a systemd restart policy "
            f"({', '.join(sorted(_RESTART_VALUES))})"
        )
    if restart_sec < 0:
        raise UnitError(f"{name}: RestartSec must not be negative, got {restart_sec}")

    unset = sorted(set(unset_env))
    for entry in unset:
        if not _ENV_NAME_RE.match(entry):
            raise UnitError(f"{name}: {entry!r} is not a usable environment variable name")
    clash = sorted(set(unset) & set(env))
    if clash:
        raise UnitError(
            f"{name}: {', '.join(clash)} appear in both env and unset_env. "
            f"UnsetEnvironment= is applied last, so the unit would start WITHOUT "
            f"them and nothing would say so — decide in the caller which wins."
        )

    out = [
        "systemd-run",
        "--user",
        f"--unit={name}",
        "--collect",
        f"--description={description if description is not None else name}",
        "-p",
        f"Restart={restart}",
        "-p",
        f"RestartSec={restart_sec}",
        "-p",
        f"StartLimitIntervalSec={start_limit_interval_sec}",
        "-p",
        f"StartLimitBurst={start_limit_burst}",
        "-p",
        f"TimeoutStopSec={timeout_stop_sec}",
        "-p",
        f"MemoryMax={memory_max}",
        "-p",
        f"WorkingDirectory={os.fspath(cwd)}",
    ]
    for entry in unset:
        out.extend(["-p", f"UnsetEnvironment={entry}"])
    for key in sorted(env):
        value = env[key]
        if "=" in key or "\n" in key or "\n" in value:
            raise UnitError(f"{name}: refusing environment entry {key!r}")
        out.append(f"--setenv={key}={value}")
    out.append("--")
    out.extend(argv)
    return out


def start_transient(
    name: str,
    argv: Sequence[str],
    env: Mapping[str, str],
    cwd: str | os.PathLike[str],
    restart: str = DEFAULT_RESTART,
    restart_sec: int = DEFAULT_RESTART_SEC,
    description: str | None = None,
    unset_env: Sequence[str] = (),
    start_limit_interval_sec: int = DEFAULT_START_LIMIT_INTERVAL_SEC,
    start_limit_burst: int = DEFAULT_START_LIMIT_BURST,
    timeout_stop_sec: int = DEFAULT_TIMEOUT_STOP_SEC,
    run: Runner | None = None,
    memory_max: str = DEFAULT_MEMORY_MAX,
) -> None:
    """Launch ``argv`` as the transient user unit ``<name>.service``.

    The point of the whole exercise (R4): the child is reparented to the
    *user manager*, in its own cgroup under ``app.slice``. It is NOT in the
    cgroup of whatever started it, so restarting or killing servedeck — or
    closing the terminal a command was typed into — cannot take the model
    with it. ``tests/test_control_e2e.py`` asserts that cgroup difference
    directly.

    ``env`` is not a *supplement* to the caller's environment: a transient
    unit inherits the **user manager's** environment (``systemctl --user
    show-environment``), never the calling shell's. Anything the model needs
    must be in ``env`` — and anything ambient it must NOT see, such as an
    operator's API keys, must be in ``unset_env``.
    """
    run = run or default_runner()
    _checked(
        run,
        start_transient_argv(
            name,
            argv,
            env,
            cwd,
            restart,
            restart_sec,
            description,
            unset_env,
            start_limit_interval_sec,
            start_limit_burst,
            timeout_stop_sec,
            memory_max,
        ),
    )


# --------------------------------------------------------------------------
# systemctl --user
# --------------------------------------------------------------------------


def stop_argv(name: str) -> list[str]:
    """``systemctl --user stop <name>``."""
    _require_name(name)
    return ["systemctl", "--user", "stop", name]


def stop(name: str, timeout_s: float = DEFAULT_STOP_TIMEOUT_S, run: Runner | None = None) -> None:
    """Stop ``<name>.service``, blocking until systemd reports it down.

    Stopping a unit that is not loaded is a no-op, not an error: with
    ``--collect`` a crashed unit is already gone, and ``stop`` after a crash
    must not raise.
    """
    run = run or default_runner(timeout_s)
    argv = stop_argv(name)
    proc = run(argv)
    if proc.returncode == 0:
        return
    text = ((proc.stderr or "") + (proc.stdout or "")).lower()
    if "not loaded" in text or "not found" in text:
        return
    raise UnitError(f"{' '.join(argv)} exited {proc.returncode}: {(proc.stderr or '').strip()}")


def show_argv(name: str) -> list[str]:
    """``systemctl --user show -p <SHOW_PROPERTIES> <name>``."""
    _require_name(name)
    return ["systemctl", "--user", "show", "-p", SHOW_PROPERTIES, name]


def _parse_properties(stdout: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key.strip()] = value.strip()
    return out


def _as_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def show(name: str, run: Runner | None = None) -> UnitState:
    """The unit's current state.

    Remember the ``--collect`` trap: ``systemctl show`` exits 0 and prints
    property *defaults* for a unit it does not know, so a crashed-and-collected
    unit comes back as ``inactive/dead/success``. Confirm with :func:`exists`
    before believing "it stopped cleanly".
    """
    run = run or default_runner()
    proc = _checked(run, show_argv(name))
    props = _parse_properties(proc.stdout)
    return UnitState(
        active_state=props.get("ActiveState", ""),
        sub_state=props.get("SubState", ""),
        result=props.get("Result", ""),
        n_restarts=_as_int(props.get("NRestarts", "0")),
        main_pid=_as_int(props.get("MainPID", "0")),
        exec_main_start_ts=props.get("ExecMainStartTimestamp", ""),
    )


def properties(name: str, props: Sequence[str], run: Runner | None = None) -> dict[str, str]:
    """Arbitrary ``systemctl --user show`` properties, as a dict.

    argv: ``systemctl --user show -p <p1,p2,...> <name>``.
    """
    _require_name(name)
    if not props:
        raise UnitError(f"{name}: no properties requested")
    run = run or default_runner()
    argv = ["systemctl", "--user", "show", "-p", ",".join(props), name]
    return _parse_properties(_checked(run, argv).stdout)


def exists(name: str, run: Runner | None = None) -> bool:
    """ONE read of ``LoadState``. True iff it came back ``loaded``.

    **This read is not reliable on its own.** Measured on this box (systemd
    259, 2026-09-12): sampling a live, running transient unit every 20 ms,
    ``systemctl --user show -p LoadState`` returned ``LoadState=not-found``
    for 2 of ~1030 reads — exit status 0, empty stderr, unit perfectly
    healthy before and after. Roughly 0.2%.

    That rate is not negligible where it is used. A health check every 2 s
    across a five-minute model boot is ~150 reads, so trusting a single
    negative would declare a healthy 90 GiB boot dead about a quarter of the
    time — and, because the honest failure text is "it failed and --collect
    removed it", the report would be confident and wrong.

    Use :func:`gone` to ask whether a unit is really absent. This function
    stays a single honest read so the flakiness is visible rather than
    smeared across a primitive that also sleeps.
    """
    return properties(name, ("LoadState",), run=run).get("LoadState") == "loaded"


def gone(
    name: str,
    run: Runner | None = None,
    attempts: int = 3,
    delay_s: float = 0.3,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """True iff the unit is really absent — ``attempts`` consecutive
    ``LoadState != loaded`` reads.

    Short-circuits: the first read that says ``loaded`` returns False
    immediately, so the delay is only ever paid for a unit that is genuinely
    gone (where nobody is waiting on latency) and never for a healthy one.

    This is the predicate every caller actually wants; :func:`exists` is the
    raw sample it is built from. See :func:`exists` for the measurement that
    makes the confirmation necessary.
    """
    for attempt in range(attempts):
        if exists(name, run=run):
            return False
        if attempt + 1 < attempts:
            sleep(delay_s)
    return True


def control_group(name: str, run: Runner | None = None) -> str:
    """The unit's cgroup path, e.g.
    ``/user.slice/user-1000.slice/user@1000.service/app.slice/model-lfm2.service``.

    Compared against ``/proc/self/cgroup`` this is the whole R4 fix stated as
    a measurable fact.
    """
    return properties(name, ("ControlGroup",), run=run).get("ControlGroup", "")


def manager_environment_names(run: Runner | None = None) -> list[str]:
    """The NAMES of every variable in the user manager's environment, sorted.

    argv: ``systemctl --user show-environment``.

    Names only, and by construction: the value half of each line is discarded
    inside this function and never returned, so no caller — and no log line,
    exception message or debugger frame built from a caller's data — can leak
    a credential that happens to be in the session environment. The one thing
    a supervisor needs from this list is which names to redact, and that needs
    no values at all.

    Raises :class:`UnitError` if the manager cannot be asked. That is
    deliberate: an empty list would read as "nothing to redact" and would
    hand every ambient secret straight to the model.
    """
    run = run or default_runner()
    proc = _checked(run, ["systemctl", "--user", "show-environment"])
    names: list[str] = []
    for line in (proc.stdout or "").splitlines():
        name, sep, _value = line.partition("=")
        name = name.strip()
        if sep and _ENV_NAME_RE.match(name):
            names.append(name)
    return sorted(set(names))


def list_units_argv(pattern: str) -> list[str]:
    """``systemctl --user list-units <pattern> --all --plain --no-legend``."""
    return ["systemctl", "--user", "list-units", pattern, "--all", "--plain", "--no-legend"]


def list_model_units_argv() -> list[str]:
    """``systemctl --user list-units model-* --all --plain --no-legend``."""
    return list_units_argv("model-*")


def list_units(pattern: str, run: Runner | None = None) -> list[str]:
    """Units matching ``pattern``, WITHOUT the ``.service`` suffix, and only
    those :func:`valid_unit_name` would let us act on.

    The pattern is a namespace: ``model-*`` in production, ``sd-test-*`` for
    the suite's own units. Discovery and action therefore share one name
    predicate — a row that could not be stopped is never returned as something
    that could be.
    """
    run = run or default_runner()
    proc = run(list_units_argv(pattern))
    # `list-units` with a glob that matches nothing exits nonzero on some
    # systemd versions; an empty list is the right answer, not an exception.
    if proc.returncode != 0 and not (proc.stdout or "").strip():
        return []
    prefix = pattern.rstrip("*")
    names: list[str] = []
    for line in (proc.stdout or "").splitlines():
        fields = line.split()
        if not fields:
            continue
        unit = fields[0].lstrip("●*").strip()
        if not unit.startswith(prefix):
            continue
        if unit.endswith(".service"):
            unit = unit[: -len(".service")]
        if valid_unit_name(unit):
            names.append(unit)
    return names


def list_model_units(run: Runner | None = None) -> list[str]:
    """Every ``model-*`` unit systemd currently knows, WITHOUT the ``.service``
    suffix (``["model-lfm2", "model-flashnext"]``), so the caller can slice the
    ``model-`` prefix off to get the registry key.

    This replaces v1's ``/proc`` walk and its pattern matching entirely — and
    with it the ``pgrep -f`` self-match trap.
    """
    return list_units("model-*", run=run)


# --------------------------------------------------------------------------
# journalctl --user
# --------------------------------------------------------------------------


def journal_tail_argv(name: str, lines: int) -> list[str]:
    """``journalctl --user -u <name> --no-pager -n <lines> -o cat``."""
    _require_name(name)
    if lines <= 0:
        raise UnitError(f"{name}: journal_tail needs a positive line count, got {lines}")
    return ["journalctl", "--user", "-u", name, "--no-pager", "-n", str(lines), "-o", "cat"]


def journal_tail(name: str, lines: int = 40, run: Runner | None = None) -> list[str]:
    """The last ``lines`` journal lines for the unit.

    Survives ``--collect``: the journal outlives the unit object, which is why
    it, not ``Result=``, is what a failure report is built from.
    Returns ``[]`` rather than raising if journald has nothing (or is absent).
    """
    run = run or default_runner()
    proc = run(journal_tail_argv(name, lines))
    if proc.returncode != 0:
        return []
    return [ln.rstrip("\n") for ln in (proc.stdout or "").splitlines()]


def journal_since_argv(name: str, since: str) -> list[str]:
    """``journalctl --user -u <name> --no-pager -o cat --since <since>``."""
    _require_name(name)
    return ["journalctl", "--user", "-u", name, "--no-pager", "-o", "cat", "--since", since]


def journal_since(name: str, since: str, run: Runner | None = None) -> list[str]:
    """Every journal line for the unit from ``since`` onward — not a tail.

    A tail cannot be used to read a boot back: the line count needed depends on
    how chatty that boot was (torch.compile and FlashInfer JIT can emit
    thousands) and, on a unit that is already serving, on how much traffic has
    arrived since. Anchoring at the unit's own start timestamp is exact and
    costs nothing extra.
    """
    run = run or default_runner()
    proc = run(journal_since_argv(name, since))
    if proc.returncode != 0:
        return []
    return [ln.rstrip("\n") for ln in (proc.stdout or "").splitlines()]


def journal_follow_argv(name: str, since: str) -> list[str]:
    """``journalctl --user -u <name> --no-pager -o cat --since <since> -f``."""
    _require_name(name)
    return [
        "journalctl",
        "--user",
        "-u",
        name,
        "--no-pager",
        "-o",
        "cat",
        "--since",
        since,
        "-f",
    ]


class _Spawner(Protocol):
    def __call__(self, argv: Sequence[str]) -> Any: ...


def _default_spawn(argv: Sequence[str]) -> "subprocess.Popen[bytes]":
    return subprocess.Popen(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )


class JournalStream:
    """A live ``journalctl -f`` pipe, read with a deadline.

    Read straight off the raw fd with ``select`` + ``os.read`` rather than
    through a text wrapper: ``select`` cannot see bytes already sitting in a
    ``TextIOWrapper``'s buffer, so the obvious ``for line in proc.stdout``
    either blocks past the deadline or reports "no output" while a marker is
    buffered. Boot progress is exactly the case where that matters.

    Always use as a context manager — the follower must be killed, and it is
    a process we started, so killing it is ours to do.
    """

    def __init__(self, proc: Any) -> None:
        self._proc = proc
        self._buf = b""
        self._fd: int | None = None
        self.eof = False

    @property
    def proc(self) -> Any:
        return self._proc

    def _fileno(self) -> int | None:
        if self._fd is not None:
            return self._fd
        stdout = getattr(self._proc, "stdout", None)
        if stdout is None:
            return None
        self._fd = stdout.fileno()
        os.set_blocking(self._fd, False)
        return self._fd

    def poll_lines(self, timeout: float) -> list[str]:
        """One ``select`` tick: every COMPLETE line available within
        ``timeout`` seconds, possibly none.

        The caller gets control back every tick whether or not the journal
        said anything, which is what lets :meth:`control.Control.wait_ready`
        also poll the port and the unit's health while a silent model is
        loading weights. A generator that looped internally would hand back
        control only when a line arrived — and the interesting failure is a
        model that has stopped saying anything at all.
        """
        fd = self._fileno()
        if fd is None or self.eof:
            return []
        out: list[str] = []
        ready, _, _ = select.select([fd], [], [], timeout)
        if ready:
            try:
                chunk: bytes | None = os.read(fd, 65536)
            except BlockingIOError:
                chunk = None  # spurious readiness, not EOF
            except OSError:
                chunk = b""
            if chunk == b"":  # EOF: journalctl exited
                self.eof = True
                if self._buf:
                    out.append(self._buf.decode("utf-8", "replace"))
                    self._buf = b""
            elif chunk:
                self._buf += chunk
        while b"\n" in self._buf:
            raw, _, self._buf = self._buf.partition(b"\n")
            out.append(raw.decode("utf-8", "replace"))
        return out

    def lines(self, deadline: float | None = None) -> Iterator[str]:
        """Yield journal lines until ``deadline`` (a ``time.monotonic()``
        value) passes or the pipe reaches EOF."""
        import time

        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                wait = min(0.5, remaining)
            else:
                wait = 0.5
            yield from self.poll_lines(wait)
            if self.eof:
                return

    def close(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    def __enter__(self) -> "JournalStream":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def journal_follow(name: str, since: str, spawn: _Spawner | None = None) -> JournalStream:
    """Start following the unit's journal from ``since`` onwards.

    ``since`` is handed to ``journalctl --since`` untouched — any form that
    flag accepts ("now", "-1min", "2026-09-12 13:00:00").
    """
    spawn = spawn or _default_spawn
    return JournalStream(spawn(journal_follow_argv(name, since)))
