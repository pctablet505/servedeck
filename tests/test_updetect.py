"""Which port the dashboard watches, and whether it can say why.

The report: ``/api/state`` returned ``upstream.port 8002, up false,
model_id null`` and every throughput figure null, while a healthy Flash-Next
server was serving on :8001 and GLM (whose port 8002 is) had been dead for a
week. Correcting ``state/desired.json`` did not help, and neither did moving a
stale ``state/server.json`` aside, because neither file was the source: the
port came from ``local_llm/.config``, whose last uncommented assignments said
``BACKEND="glm53"`` / ``PORT="8002"`` — and that one file was the whole of the
answer.

The migration claimed "model detection now reads the live process and
/v1/models, never a config header". These tests are that claim, made true one
level up: the PORT is now resolved from the live process too, and when it
cannot be, the dashboard says which port it looked at and why instead of
rendering a blank.

Nothing here starts, stops or contacts a server. The two seams — the /proc
walk (``procctl.scan_vllm_processes``) and the socket table
(``procctl.listening_pids``) — plus ``/v1/models`` (``app._probe_models``) are
faked; everything else is the production path.
"""

from __future__ import annotations

import asyncio
import json
import types

import pytest

from servedeck import app as capp
from servedeck import paths, procctl, supervisor as _sup, updetect


# --------------------------------------------------------------------------
# The resolver itself — no app, no machine
# --------------------------------------------------------------------------
_BACKENDS = [("glm53", 8002), ("flashnext", 8001), ("inline", 8004)]


def _resolve(**kw):
    kw.setdefault("scan", lambda: [])
    kw.setdefault("backends", _BACKENDS)
    return updetect.resolve(**kw)


def test_a_live_process_outranks_every_file() -> None:
    """The precedence, at its most direct: the shell config names :8002, the
    desired state names :8002, and a vLLM process is serving on :8001. The
    process wins — it is a fact, and the files are intentions."""
    live = [updetect.LiveServer(pid=3102188, port=8001, backend="flashnext")]
    got = _resolve(
        scan=lambda: live,
        probe=lambda port: None,
        shell_port="8002", shell_backend="glm53", desired_port=8002,
    )
    assert got.port == 8001
    assert got.source == updetect.LIVE_PROCESS
    assert got.pid == 3102188 and got.backend == "flashnext" and got.live is True
    assert "8002" in got.reason, (
        "following a live process over the configured port is exactly the "
        "surprise that has to be explained on the page"
    )


def test_a_dead_configured_port_never_beats_a_live_one() -> None:
    """Nothing recognisable is running (a server we cannot attribute, another
    uid's process), but a known backend's port answers while the configured one
    does not. Reporting the box as dead there is the whole defect."""
    got = _resolve(
        probe=lambda port: 4242 if port == 8001 else None,
        shell_port=8002, shell_backend="glm53",
    )
    assert got.port == 8001 and got.source == updetect.LIVE_PORT and got.live is True
    assert "8002" in got.reason and "8001" in got.reason, got.reason


def test_a_foreign_listener_on_a_configured_port_is_not_serving() -> None:
    """2026-09-17: nothing vLLM was running, an unrelated project's web app
    held :8002 (the glm53 port), and the page said "serving". A listener the
    caller identifies as foreign is reported, but never followed as live."""
    got = _resolve(
        probe=lambda port: {8002: 1011710}.get(port),
        foreign=lambda pid: pid == 1011710,
        shell_port="8001", shell_backend="flashnext",
    )
    assert got.live is False
    assert got.source != updetect.LIVE_PORT
    assert got.port == 8001, "the page keeps watching the configured port"
    assert "1011710" in got.reason and "not a vLLM server" in got.reason
    held = [c for c in got.candidates if c.port == 8002]
    assert held and held[0].listening is True and held[0].pid == 1011710, (
        "the foreign holder is still a fact on the page, just not a model"
    )


def test_a_vllm_listener_is_still_followed_when_a_foreign_check_is_supplied() -> None:
    """Over-correction guard: the foreign check must only drop what it names."""
    got = _resolve(
        probe=lambda port: {8002: 777}.get(port),
        foreign=lambda pid: False,
        shell_port="8001", shell_backend="flashnext",
    )
    assert got.live is True and got.source == updetect.LIVE_PORT and got.pid == 777


def test_the_shell_config_wins_when_it_is_the_live_one() -> None:
    """Over-correction guard: discovery must not drag the dashboard off a
    perfectly good port just because it did the discovering."""
    got = _resolve(probe=lambda port: 77 if port == 8002 else None,
                   shell_port=8002, shell_backend="glm53")
    assert got.port == 8002 and got.source == updetect.SHELL_CONFIG and got.live is True


def test_desired_state_is_used_when_the_shell_config_names_nothing() -> None:
    got = _resolve(probe=lambda port: None, shell_port="", desired_port=8004,
                   desired_backend="inline")
    assert got.port == 8004 and got.source == updetect.DESIRED


def test_a_configured_backend_is_the_last_resort() -> None:
    got = _resolve(probe=lambda port: None)
    assert got.port == 8002 and got.source == updetect.BACKEND_CONFIG, (
        "with nothing else to go on, the first configured backend's port is "
        "the answer — and the reason must say that is all it is"
    )


def test_nothing_anywhere_still_says_what_was_checked() -> None:
    """A silent blank is the failure mode this whole module exists to end."""
    got = _resolve(probe=lambda port: None, shell_port=8002, shell_backend="glm53")
    assert got.live is False
    for port in (8002, 8001, 8004):
        assert f":{port}" in got.reason, f"{port} is missing from: {got.reason}"
    assert [c.listening for c in got.candidates] == [False, False, False]


def test_an_unprobed_port_is_not_reported_as_an_empty_one() -> None:
    """`probe=None` means the socket table was never read. Saying "nothing is
    listening" there would be a confident claim about a check that never ran —
    the same class of wrongness as the config header this replaces."""
    got = _resolve(shell_port=8002, shell_backend="glm53")
    assert got.port == 8002
    assert all(c.listening is None for c in got.candidates)
    assert "nothing is listening on" not in got.reason, got.reason
    assert "probed" in got.reason, got.reason


def test_a_junk_port_is_never_watched() -> None:
    """A hand-edited shell config must not be able to retarget the dashboard
    at a port a listener cannot even be on."""
    for junk in ("", "not-a-port", "0", "99999", None):
        got = _resolve(probe=lambda port: None, shell_port=junk)
        assert got.port == 8002 and got.source == updetect.BACKEND_CONFIG, junk


def test_the_port_we_are_already_on_breaks_a_tie_between_live_servers() -> None:
    """Two servers up at once (a hand launch beside a managed one). Moving off
    the healthy one we are already watching would throw away the poller's
    throughput baseline for no gain."""
    live = [
        updetect.LiveServer(pid=1, port=8001, backend="flashnext"),
        updetect.LiveServer(pid=2, port=8002, backend="glm53"),
    ]
    assert _resolve(scan=lambda: live, probe=lambda p: None, current_port=8002).port == 8002
    assert _resolve(scan=lambda: live, probe=lambda p: None, current_port=8001).port == 8001


# --------------------------------------------------------------------------
# The app, with the machine faked out
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _restore_runtime():
    """rt is process-wide. A test that repoints it must not repoint it for the
    rest of the session."""
    before = (capp.rt.port, capp.rt.upstream, capp.rt.poller,
              capp.rt.resolution, capp.rt.upstream_up, capp.rt.client)
    yield
    (capp.rt.port, capp.rt.upstream, capp.rt.poller,
     capp.rt.resolution, capp.rt.upstream_up, capp.rt.client) = before
    capp._invalidate_listener()


@pytest.fixture
def machine(monkeypatch):
    """Fake the two things the resolver reads about this box.

    ``live`` is what a /proc walk would find; ``listening`` is the socket
    table (port -> pid). Both are the real seams: procctl's own scan and its
    own one-shot ``ss`` reader.
    """

    def _set(*, live=(), listening=None, shell=None, desired=(None, None),
             backends=(("glm53", 8002), ("flashnext", 8001), ("inline", 8004))):
        listening = dict(listening or {})
        procs = [
            procctl.VllmProc(
                pid=s.pid, ppid=1, pgid=s.pid, comm="vllm", cmdline=list(s.cmdline),
                cwd=None, exe=None, role=procctl.ROLE_SERVER, venv=s.backend, port=s.port,
            )
            for s in live
        ]
        monkeypatch.setattr(procctl, "scan_vllm_processes", lambda: procs)
        monkeypatch.setattr(procctl, "listening_pids", lambda: dict(listening))
        monkeypatch.setattr(procctl, "listener_pid", lambda port: listening.get(port))
        monkeypatch.setattr(capp, "_safe_config", lambda: dict(shell or {}))
        monkeypatch.setattr(capp, "_configured_ports", lambda: list(backends))
        monkeypatch.setattr(capp, "_desired_target", lambda: desired)
        monkeypatch.setattr(capp, "_supervisor", None, raising=False)

    return _set


_FLASHNEXT_ARGV = [
    "/home/x/.venv-next/bin/vllm", "serve",
    "mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4",
    "--served-model-name", "qwen38-flash-next",
    "--port", "8001", "--max-model-len", "262144",
]


def test_the_dashboard_follows_the_live_server_not_the_configured_one(machine) -> None:
    """The reported defect, end to end.

    ``desired.json`` and the shell config both name the glm53 backend on
    :8002, GLM is not running, and Flash-Next is serving on :8001. The
    dashboard must watch :8001.
    """
    machine(
        live=[updetect.LiveServer(3102188, 8001, "flashnext", tuple(_FLASHNEXT_ARGV))],
        listening={8001: 3102188},
        shell={"BACKEND": "glm53", "PORT": "8002"},
        desired=(8002, "glm53"),
    )
    rt = capp.Runtime()
    assert rt.port == 8002, "construction is files-only; the poll is what discovers"

    assert rt.retarget() is True
    assert rt.port == 8001 and rt.upstream.endswith(":8001")
    assert rt.poller.base_url.endswith(":8001")
    assert rt.resolution.source == updetect.LIVE_PROCESS
    assert rt.resolution.backend == "flashnext"


def test_a_stale_server_json_naming_a_dead_pid_is_not_a_source(machine, tmp_path,
                                                               monkeypatch) -> None:
    """``state/server.json`` recorded pid 214759 — a GLM launch from
    2026-09-02 that had been dead for a week. Nothing may take a port, a pid
    or a backend from it: it is a record of what was launched once, and a
    launch is not a running process.
    """
    (tmp_path / "desired.json").write_text(json.dumps(
        {"version": 1, "desired_state": "RUNNING", "backend": "glm53", "port": 8002,
         "repo_id": "dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4"}
    ))
    (tmp_path / "server.json").write_text(json.dumps(
        {"pid": 214759, "pgid": 214759, "argv": ["/home/x/vllm-glm53/serve-opt.sh"],
         "cwd": "/home/x/vllm-glm53", "log_path": "", "started_at": 0.0}
    ))
    monkeypatch.setattr(paths, "STATE_DIR", tmp_path)

    def _never(*a, **k):
        raise AssertionError(
            "the upstream resolution read state/server.json; a file recording a "
            "dead launch must never decide which port the dashboard watches"
        )

    monkeypatch.setattr(procctl, "load_handle", _never)
    machine(
        live=[updetect.LiveServer(3102188, 8001, "flashnext", tuple(_FLASHNEXT_ARGV))],
        listening={8001: 3102188},
        shell={},
        desired=_sup.load_desired(tmp_path).port and (8002, "glm53") or (None, None),
    )
    rt = capp.Runtime()
    rt.retarget()

    assert rt.port == 8001, rt.resolution.reason
    assert rt.resolution.pid == 3102188
    assert 214759 not in [c.pid for c in rt.resolution.candidates]


def test_nothing_running_gives_a_reason_not_a_blank(machine) -> None:
    """"up: false, model: null, every figure null" is not an answer. Which
    port was looked at, and why that one, is."""
    machine(shell={"BACKEND": "glm53", "PORT": "8002"}, desired=(8002, "glm53"))
    rt = capp.Runtime()
    rt.retarget()

    assert rt.port == 8002 and rt.resolution.live is False
    assert "shell config" in rt.resolution.reason, rt.resolution.reason
    checked = {c.port for c in rt.resolution.candidates}
    assert {8001, 8002, 8004} <= checked, checked


def test_state_carries_the_resolution_for_the_page(machine, monkeypatch, tmp_path) -> None:
    """The reason has to reach the payload, or the page cannot render it."""
    monkeypatch.setattr(paths, "STATE_DIR", tmp_path)
    machine(shell={"BACKEND": "glm53", "PORT": "8002"})
    monkeypatch.setattr(capp, "_live_boot_facts", lambda: {})
    capp.rt.follow(capp._resolve_upstream(discover=True, current_port=capp.rt.port))

    payload = capp._state()["upstream"]

    assert payload["port"] == payload["resolution"]["port"]
    assert payload["resolution"]["reason"]
    assert payload["resolution"]["candidates"], "nothing to show for what was checked"


def test_the_first_payload_is_already_resolved(machine, monkeypatch) -> None:
    """Construction is files-only on purpose (a /proc walk on import would be
    paid by every importer), so the resolution has to happen before the app
    can answer anything. Without it the page loaded in the first two seconds
    is told about a port nothing has probed -- the wrong answer, arriving
    first, which is the one an operator reads.
    """
    machine(
        live=[updetect.LiveServer(3102188, 8001, "flashnext", tuple(_FLASHNEXT_ARGV))],
        listening={8001: 3102188},
        shell={"BACKEND": "glm53", "PORT": "8002"},
    )
    capp.rt.follow(capp._resolve_upstream(discover=False))
    assert capp.rt.port == 8002, "precondition: the files-only guess"

    async def _no_polling() -> None:
        return None

    monkeypatch.setattr(capp, "_poll_loop", _no_polling)
    monkeypatch.setattr(capp, "sup", lambda: None)

    async def _boot() -> None:
        await capp._startup()
        await asyncio.sleep(0)          # let the stubbed poll task retire
        await capp.rt.client.aclose()

    asyncio.run(_boot())

    assert capp.rt.port == 8001, capp.rt.resolution.reason


# --------------------------------------------------------------------------
# Adoption
# --------------------------------------------------------------------------
def _supervisor_for(tmp_path, monkeypatch):
    s = _sup.Supervisor(
        state_dir=tmp_path,
        clock=lambda: 0.0,
        launch_fn=lambda *a, **k: None,
        stop_fn=lambda h, **kw: procctl.StopResult(True, "term", 0.0, "faked"),
        history_path=tmp_path / "history.jsonl",
    )
    s._adopt_ready = types.MethodType(lambda self, pid: None, s)  # type: ignore[assignment]
    monkeypatch.setattr(capp, "sup", lambda: s)
    monkeypatch.setattr(capp, "_state", lambda: {})
    return s


def test_adopt_discovers_a_server_on_a_port_nothing_configured(machine, tmp_path,
                                                               monkeypatch) -> None:
    """``POST /api/server/adopt`` with no body used to probe
    ``desired.port or rt.port`` — files, both of them — so pressing Adopt while
    the configuration named a dead backend re-probed the dead port and answered
    "nothing is listening on port 8002" with a server running two ports away.

    It must scan the ports it knows, match on ``/v1/models`` AND on the
    process's command line, and adopt what it actually finds.
    """
    machine(
        live=[updetect.LiveServer(4242, 8099, "flashnext", tuple(_FLASHNEXT_ARGV))],
        listening={8099: 4242},
        shell={"BACKEND": "glm53", "PORT": "8002"},
        desired=(8002, "glm53"),
    )
    asked: list[int] = []

    async def _models(port: int):
        asked.append(port)
        return ["qwen38-flash-next"] if port == 8099 else []

    monkeypatch.setattr(capp, "_probe_models", _models)
    monkeypatch.setattr(procctl, "is_attributable", lambda pid: pid == 4242)
    monkeypatch.setattr(procctl, "backend_of_pid", lambda pid: "flashnext")
    monkeypatch.setattr(procctl, "cmdline_of", lambda pid: list(_FLASHNEXT_ARGV))
    s = _supervisor_for(tmp_path, monkeypatch)

    response = asyncio.run(capp.api_adopt({}))

    assert response.status_code == 200, getattr(response, "body", b"")
    assert json.loads(response.body) == {"adopted": True, "pid": 4242, "port": 8099}
    assert 8099 in asked, "adopt never asked /v1/models on the discovered port"
    assert s.desired.port == 8099 and s.desired.backend == "flashnext"
    assert s.desired.repo_id == "mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4", (
        "the repo must come from the process's own --model, not from the "
        "served name or the config header"
    )
    assert capp.rt.port == 8099, (
        "adoption left the dashboard watching the old port: a managed server "
        "it still could not see"
    )


def test_adopt_says_what_it_checked_when_it_finds_nothing(machine, tmp_path,
                                                          monkeypatch) -> None:
    """A refusal has to be actionable. "nothing is listening on port 8002"
    about a port the user never chose is not."""
    machine(shell={"BACKEND": "glm53", "PORT": "8002"}, desired=(8002, "glm53"))

    async def _models(port: int):
        return []

    monkeypatch.setattr(capp, "_probe_models", _models)
    _supervisor_for(tmp_path, monkeypatch)

    response = asyncio.run(capp.api_adopt({}))

    assert response.status_code == 409
    body = json.loads(response.body)
    for port in (8002, 8001, 8004):
        assert f":{port}" in body["error"], (port, body["error"])
    assert body["resolution"]["candidates"]


def test_adopt_still_honours_an_explicitly_named_port(machine, tmp_path,
                                                      monkeypatch) -> None:
    """Discovery is for the empty request. A port the operator named is
    probed exactly, and refused precisely — never silently swapped for
    something discovery liked better."""
    machine(listening={8001: 4242}, shell={"BACKEND": "glm53", "PORT": "8002"})
    monkeypatch.setattr(procctl, "is_attributable", lambda pid: False)
    _supervisor_for(tmp_path, monkeypatch)

    refused = asyncio.run(capp.api_adopt({"port": 8004}))
    assert refused.status_code == 409
    assert "nothing is listening on port 8004" in json.loads(refused.body)["error"]

    unattributable = asyncio.run(capp.api_adopt({"port": 8001}))
    assert unattributable.status_code == 409
    assert "pid 4242" in json.loads(unattributable.body)["error"]


def test_a_healthy_upstream_is_never_dragged_off_its_port_by_a_file(machine) -> None:
    """Over-correction guard, and a defect the fix itself had.

    While the dashboard is talking to a server, re-resolving from files alone
    (the cheap path taken when there is nothing to discover) would move it to
    whatever the shell config names — here the dead backend — and the next
    poll, finding that port dead, would move it back. A flap every 2 s, with
    the poller rebuilt each way, so no throughput figure ever survives long
    enough to be computed.
    """
    machine(
        live=[updetect.LiveServer(3102188, 8001, "flashnext", tuple(_FLASHNEXT_ARGV))],
        listening={8001: 3102188},
        shell={"BACKEND": "glm53", "PORT": "8002"},
        desired=(8002, "glm53"),
    )
    rt = capp.Runtime()
    rt.retarget()
    assert rt.port == 8001, "precondition: discovery found the live server"
    rt.upstream_up = True                      # the poller is scraping it happily
    poller = rt.poller

    for _ in range(5):
        assert rt.retarget() is False, (
            f"the dashboard left a healthy :8001 for :{rt.port} on a poll where "
            "nothing about the machine had changed"
        )
    assert rt.port == 8001 and rt.poller is poller
