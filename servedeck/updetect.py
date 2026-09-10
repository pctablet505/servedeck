"""Which port the dashboard watches, and why it picked that one.

The migration's claim was that "model detection now reads the live process and
/v1/models, never a config header". It was half true: `_serving_identity()`
does read the process — but it reads the process *on `rt.port`*, and `rt.port`
itself came from one file and one file only, the shell config's ``PORT``. So a
stale header could no longer mislabel the model, but it could still point the
whole dashboard at a port nothing was serving on, and every reader downstream
then honestly reported an empty answer about a healthy box. That is what
happened on 2026-09-10: ``local_llm/.config`` ended with ``BACKEND="glm53"`` /
``PORT="8002"`` (its own header comment said flashnext, and the file's last
uncommented assignment wins), GLM was long dead, Flash-Next was serving on
:8001, and ``/api/state`` reported ``port 8002, up false, model_id null``.

The rule this module implements, in priority order:

1. **a live process this tooling recognises**, discovered from the system
   (``/proc`` + the socket table) rather than from any file. A process that is
   running is not an intention, it is a fact, and it outranks every file.
2. the shell config's ``BACKEND``/``PORT`` — what the CLI last set up;
3. ``state/desired.json`` — what Servedeck last wanted;
4. a configured backend's port, as a last resort.

And one rule that cuts across all four: **if the port chosen from a file has
nothing listening while another known backend's port does, follow the live
one and say so.** Reporting the box as dead while a server answers two ports
away is the failure this module exists to prevent.

Nothing here does I/O. The two probes -- "what vLLM processes are live" and
"is anything listening on this port" -- are injected by the caller, which is
what makes the precedence chain testable without a server, a socket, or a
``/proc`` that has to be arranged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

#: A port a listener can actually be on. A hand-edited shell config must not be
#: able to retarget the whole dashboard at nothing.
PORT_RANGE = range(1, 65536)

#: Watched when nothing -- no live process, no shell config, no desired state,
#: no configured backend -- names a port at all. A placeholder, never a claim.
DEFAULT_PORT = 8000

# Where the answer came from. These strings are part of /api/state's payload:
# the UI renders them, so they are named for a reader, not for a caller.
LIVE_PROCESS = "live_process"
LIVE_PORT = "live_port"
SHELL_CONFIG = "shell_config"
DESIRED = "desired"
BACKEND_CONFIG = "backend_config"
FALLBACK = "fallback"

#: How each source is named INSIDE a sentence about a port -- "nothing is
#: listening on :8002 (named by the shell config's PORT)". One phrasing, used
#: in every branch below, so no two branches can describe the same source
#: differently.
_NAMED_BY = {
    LIVE_PROCESS: "where a live vLLM process is serving",
    LIVE_PORT: "where a listener answers",
    SHELL_CONFIG: "named by the shell config's PORT",
    DESIRED: "named by state/desired.json",
    BACKEND_CONFIG: "declared by a configured backend",
    FALLBACK: "the built-in default",
}


@dataclass(frozen=True)
class LiveServer:
    """A vLLM api-server process found on this machine, with its port.

    ``backend`` is the label the process itself resolved to (its venv/cwd
    lives inside a configured backend's tree) -- not what any file says is
    running. ``None`` means the process is a vLLM server we can see but cannot
    attribute, which is a fact worth carrying rather than a reason to drop it.
    """

    pid: int
    port: int
    backend: str | None = None
    cmdline: tuple[str, ...] = ()


@dataclass(frozen=True)
class Candidate:
    """One port the resolver considered, and what it knows about it.

    ``listening`` is tri-state on purpose: ``None`` means the port was never
    probed (a live process already answered the question), and rendering that
    as "nothing there" would be a claim the resolver never made.
    """

    port: int
    source: str
    backend: str | None = None
    listening: bool | None = None
    pid: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "port": self.port,
            "source": self.source,
            "backend": self.backend,
            "listening": self.listening,
            "pid": self.pid,
        }


@dataclass(frozen=True)
class Upstream:
    """The resolved upstream, and the sentence that explains it.

    ``reason`` exists because the failure this module fixes was invisible: the
    dashboard showed blanks and no port, and there was nothing on the page or
    in the payload that said which port had been looked at or why. A resolver
    that cannot find a server must still be able to say what it tried.
    """

    port: int
    source: str
    reason: str
    backend: str | None = None
    pid: int | None = None
    live: bool = False
    candidates: tuple[Candidate, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "port": self.port,
            "source": self.source,
            "reason": self.reason,
            "backend": self.backend,
            "pid": self.pid,
            "live": self.live,
            "candidates": [c.to_dict() for c in self.candidates],
        }


def _valid(port: object) -> int | None:
    """`port` as an int in the range a listener can be on, or None."""
    try:
        value = int(port)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if value in PORT_RANGE else None


def _label(port: int, backends: Sequence[tuple[str, int]]) -> str:
    """How to name a port in a sentence: ":8001 (backends.flashnext)"."""
    for name, p in backends:
        if p == port:
            return f":{port} (backends.{name})"
    return f":{port}"


def _backend_at(port: int, backends: Sequence[tuple[str, int]]) -> str | None:
    for name, p in backends:
        if p == port:
            return name
    return None


def resolve(
    *,
    scan: Callable[[], Sequence[LiveServer]],
    probe: Callable[[int], int | None] | None = None,
    shell_port: object = None,
    shell_backend: str | None = None,
    desired_port: object = None,
    desired_backend: str | None = None,
    backends: Sequence[tuple[str, int]] = (),
    current_port: int | None = None,
    fallback_port: int = DEFAULT_PORT,
) -> Upstream:
    """Pick the upstream port, following the precedence in the module docstring.

    ``scan`` returns the live vLLM servers (empty when the caller does not want
    a process scan -- construction time, say, where a ``/proc`` walk on import
    would be paid by every test that so much as imports the app). ``probe``
    answers "which pid, if any, is listening on this port".

    ``probe=None`` means the socket table was not consulted at all, and the
    answer says so: an unprobed port is reported as ``listening: None`` and
    the reason claims nothing about what is or is not there. A resolver that
    reported "nothing is listening anywhere" from a check it never ran would
    be the same class of confident-and-wrong this module exists to end.
    """
    backends = tuple((name, p) for name, p in backends if _valid(p) is not None)

    # ---------------------------------------------------------------- files
    ordered: list[Candidate] = []
    seen: set[int] = set()

    def _consider(port: object, source: str, backend: str | None) -> None:
        value = _valid(port)
        if value is None or value in seen:
            return
        seen.add(value)
        ordered.append(
            Candidate(value, source, backend or _backend_at(value, backends))
        )

    _consider(shell_port, SHELL_CONFIG, shell_backend)
    _consider(desired_port, DESIRED, desired_backend)
    for name, port in backends:
        _consider(port, BACKEND_CONFIG, name)

    # -------------------------------------------------------- live process
    live = [s for s in scan() if _valid(s.port) is not None]
    if live:
        # Several servers can be up at once (a hand launch beside a managed
        # one). Prefer the one we are already watching -- moving off a healthy
        # port we are on would restart the throughput baseline for nothing --
        # then the one the shell config names, then desired, then the
        # configured order. Port number is the final tie-break so the answer
        # is stable rather than dependent on /proc iteration order.
        preference = [
            p for p in (
                _valid(current_port), _valid(shell_port), _valid(desired_port)
            ) if p is not None
        ] + [p for _, p in backends]
        rank = {port: i for i, port in enumerate(reversed(preference))}
        chosen = max(live, key=lambda s: (rank.get(s.port, -1), -s.port))
        live_candidates = tuple(
            Candidate(s.port, LIVE_PROCESS, s.backend, True, s.pid) for s in live
        )
        rest = tuple(c for c in ordered if c.port not in {s.port for s in live})
        named = _label(chosen.port, backends)
        who = f"pid {chosen.pid}" + (f", backend {chosen.backend}" if chosen.backend else "")
        overridden = ""
        shell = _valid(shell_port)
        if shell is not None and shell != chosen.port:
            overridden = (
                f" — the shell config's PORT says :{shell}, and a running "
                "process outranks a file"
            )
        return Upstream(
            port=chosen.port,
            source=LIVE_PROCESS,
            reason=f"a live vLLM server ({who}) is listening on {named}{overridden}",
            backend=chosen.backend,
            pid=chosen.pid,
            live=True,
            candidates=live_candidates + rest,
        )

    # -------------------------------------------------------- listening port
    if not ordered:
        port = _valid(fallback_port) or DEFAULT_PORT
        if probe is None:
            return Upstream(
                port=port,
                source=FALLBACK,
                reason=(
                    f"no backend is configured and no file names a port; watching "
                    f":{port}, {_NAMED_BY[FALLBACK]} — nothing has been probed yet"
                ),
                candidates=(Candidate(port, FALLBACK),),
            )
        pid = probe(port)
        return Upstream(
            port=port,
            source=FALLBACK,
            reason=(
                f"no backend is configured and no file names a port; watching "
                f":{port}, where " + (f"pid {pid} is listening" if pid else "nothing is listening")
            ),
            pid=pid,
            live=pid is not None,
            candidates=(Candidate(port, FALLBACK, None, pid is not None, pid),)
        )

    if probe is None:
        first_unprobed = ordered[0]
        return Upstream(
            port=first_unprobed.port,
            source=first_unprobed.source,
            reason=(
                f"watching {_label(first_unprobed.port, backends)}, "
                f"{_NAMED_BY[first_unprobed.source]} — nothing has been probed yet; "
                "the next poll checks what is actually listening"
            ),
            backend=first_unprobed.backend,
            candidates=tuple(ordered),
        )

    checked_list: list[Candidate] = []
    for c in ordered:
        pid = probe(c.port)
        checked_list.append(Candidate(c.port, c.source, c.backend, pid is not None, pid))
    probed = tuple(checked_list)
    first = probed[0]
    if first.listening:
        return Upstream(
            port=first.port,
            source=first.source,
            reason=(
                f"watching {_label(first.port, backends)}, "
                f"{_NAMED_BY[first.source]}, where pid {first.pid} is listening"
            ),
            backend=first.backend,
            pid=first.pid,
            live=True,
            candidates=probed,
        )

    # The port a file named is dead. Following a live one two ports away is the
    # whole point: reporting the box as dead while a server answers is the
    # failure, and doing it silently is the second failure.
    alive = next((c for c in probed[1:] if c.listening), None)
    if alive is not None:
        return Upstream(
            port=alive.port,
            source=LIVE_PORT,
            reason=(
                f"nothing is listening on {_label(first.port, backends)} "
                f"({_NAMED_BY[first.source]}); {_label(alive.port, backends)} has a "
                f"listener (pid {alive.pid}), so the dashboard followed it"
            ),
            backend=alive.backend,
            pid=alive.pid,
            live=True,
            candidates=probed,
        )

    checked = ", ".join(_label(c.port, backends) for c in probed)
    return Upstream(
        port=first.port,
        source=first.source,
        reason=(
            f"nothing is listening on any known port (checked {checked}); "
            f"showing {_label(first.port, backends)}, {_NAMED_BY[first.source]}"
        ),
        backend=first.backend,
        live=False,
        candidates=probed,
    )
