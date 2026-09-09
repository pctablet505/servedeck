"""Which model is serving, and how the dashboard knows.

The owner's second report was "unable to detect the model correctly". Three
separate things were deciding it, none of them from the running server:

* the model list matched on ``--served-model-name`` — an alias an operator
  reuses across checkpoints, and one that a start could copy from the previous
  model outright (see test_launch_contract);
* failing that, it matched anything whose weights were within 0.5 GiB of the
  live figure — which marks a DIFFERENT model of similar size as the one
  running, and always finds something;
* adoption labelled the process with the shell config's ``BACKEND`` header,
  the header that said GLM while Qwen was serving.

The rule these tests hold to: identity comes from the live process's own
``--model`` first, from ``/v1/models`` on the live port second, and from the
config header never.
"""

from __future__ import annotations

import types

import pytest

from servedeck import app as capp
from servedeck import registry


class _Entry:
    """The two ModelEntry fields _repo_for_served_name reads."""

    def __init__(self, repo_id: str) -> None:
        self.repo_id = repo_id


@pytest.fixture
def cache(monkeypatch):
    """A model cache with two checkpoints of very similar size."""
    entries = [
        _Entry("mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4"),
        _Entry("RadixArk/Qwen3.8-Flash-Next-NVFP4"),
        _Entry("dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4"),
    ]
    monkeypatch.setattr(registry, "discover_models", lambda *a, **k: entries)
    return entries


@pytest.fixture
def live(monkeypatch):
    """Drive the two things _serving_identity reads: the live process's argv
    and what /v1/models advertises."""

    state = {"argv": [], "names": [], "pid": 4242, "backend": None, "up": True}

    def _apply():
        """Call after setting the fields above — they are read, not watched."""
        capp.rt.upstream_up = state["up"]
        capp.rt.serving_models = list(state["names"])
        monkeypatch.setattr(capp, "_listener_argv", lambda: list(state["argv"]))
        import servedeck.procctl as real

        monkeypatch.setattr(real, "listener_pid", lambda port: state["pid"])
        monkeypatch.setattr(real, "backend_of_pid", lambda pid: state["backend"])

    state["apply"] = _apply
    return state


# --------------------------------------------------------------------------- #

def test_identity_comes_from_the_running_process_not_its_alias(cache, live):
    """The live process was launched with ``--model <repo>``. That is what
    vLLM actually loaded, and nothing the operator can rename changes it."""
    live["argv"] = [
        "/venv/bin/python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4",
        "--served-model-name", "qwen38-flash-next",
    ]
    live["names"] = ["qwen38-flash-next"]
    live["backend"] = "flashnext"
    live["apply"]()

    ident = capp._serving_identity()
    assert ident["repo_id"] == "mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4"
    assert ident["source"] == "process"
    assert ident["backend"] == "flashnext"
    assert ident["served_names"] == ["qwen38-flash-next"]


def test_a_stale_alias_is_reported_as_a_mismatch_not_as_the_answer(cache, live):
    """The failure the owner saw: a Qwen checkpoint serving under the name of
    the GLM run before it, because the start carried the previous model's
    ``--served-model-name`` over. A name-matching dashboard shows the GLM row,
    or no row at all. Identity must still be the Qwen repo, and the
    disagreement must be visible rather than silently resolved."""
    live["argv"] = [
        "vllm", "serve", "mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4",
        "--served-model-name", "GLM-5.3-Flash-ABLITERATED-NVFP4",
    ]
    live["names"] = ["GLM-5.3-Flash-ABLITERATED-NVFP4"]
    live["apply"]()

    ident = capp._serving_identity()
    assert ident["repo_id"] == "mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4"
    assert ident["mismatch"] is True, (
        "the served name belongs to a different model than the one loaded; "
        "that has to surface, not be averaged away"
    )


def test_v1_models_answers_when_the_process_cannot_be_read(cache, live):
    """A server behind a socket we can see but a /proc entry we cannot (a
    different uid, a container) still tells us its name over the API. Second
    choice, and labelled as such."""
    live["argv"] = []
    live["names"] = ["GLM-5.3-Flash-ABLITERATED-NVFP4"]
    live["apply"]()

    ident = capp._serving_identity()
    assert ident["repo_id"] == "dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4"
    assert ident["source"] == "served_name"
    assert ident["mismatch"] is False


def test_an_ambiguous_alias_identifies_nothing(cache, live, monkeypatch):
    """Two cached repos whose final segment is the same name. Resolving to
    "the first one" would be a coin flip presented as a fact."""
    monkeypatch.setattr(
        registry, "discover_models",
        lambda *a, **k: [_Entry("a/Same-Name"), _Entry("b/Same-Name")],
    )
    live["argv"] = []
    live["names"] = ["Same-Name"]
    live["apply"]()

    ident = capp._serving_identity()
    assert ident["repo_id"] is None
    assert ident["source"] == "unknown"


def test_nothing_serving_claims_nothing(cache, live):
    live["argv"] = []
    live["names"] = []
    live["up"] = False
    live["apply"]()

    ident = capp._serving_identity()
    assert ident["repo_id"] is None
    assert ident["source"] == "unknown"
    assert ident["served_names"] == []


def test_all_served_names_are_reported_not_just_the_first(cache, live):
    """A vLLM server can advertise several ids for one loaded model. Taking
    ``data[0]`` made whichever alias sorted first the entire answer."""
    live["argv"] = ["vllm", "serve", "RadixArk/Qwen3.8-Flash-Next-NVFP4"]
    live["names"] = ["alias-one", "RadixArk/Qwen3.8-Flash-Next-NVFP4"]
    live["apply"]()

    ident = capp._serving_identity()
    assert ident["served_names"] == ["alias-one", "RadixArk/Qwen3.8-Flash-Next-NVFP4"]
    assert ident["repo_id"] == "RadixArk/Qwen3.8-Flash-Next-NVFP4"
    assert ident["mismatch"] is False


def test_state_carries_identity_for_the_dashboard(cache, live, monkeypatch):
    """The dashboard matches its model list on this; if /api/state stops
    carrying it the UI silently falls back to alias matching again."""
    live["argv"] = ["vllm", "serve", "RadixArk/Qwen3.8-Flash-Next-NVFP4"]
    live["names"] = ["qwen38-flash-next"]
    live["apply"]()
    monkeypatch.setattr(capp, "_live_boot_facts", lambda: {})
    monkeypatch.setattr(capp, "_safe_config", lambda: {})
    monkeypatch.setattr(capp, "_server_uptime_s", lambda: None)

    ident = capp._state()["upstream"]["identity"]
    assert ident["repo_id"] == "RadixArk/Qwen3.8-Flash-Next-NVFP4"


# --------------------------------------------------------------------------- #
# The dashboard's own matching rule
# --------------------------------------------------------------------------- #

def test_the_dashboard_matches_on_identity_and_not_on_weight_similarity():
    """A source-level assertion, because there is no JS runtime here.

    The removed rule was ``Math.abs(m.weights_gib - live.weights_gib) < 0.5``:
    an always-succeeding fallback that marks whichever cached model happens to
    be within half a gigabyte as the running one. Two NVFP4 conversions of the
    same checkpoint differ by far less than that.
    """
    src = (capp.HERE / "web" / "app.js").read_text()
    assert "up.identity" in src, "the dashboard no longer reads /api/state's identity"
    assert "liveFacts.weights_gib" not in src.split("function post(")[0], (
        "the weights-similarity fallback for 'which model is serving' is back"
    )


# --------------------------------------------------------------------------- #
# Adoption
# --------------------------------------------------------------------------- #

def test_adopting_labels_the_process_from_the_process(cache, live, tmp_path, monkeypatch):
    """``d.backend`` used to come from the shell config's ``BACKEND`` key.

    That header is what somebody last INTENDED, and on this box it said
    ``glm53`` for days while Qwen served on :8001 — `llm` launched GLM, the
    systemd unit tried to load GLM inside a venv that cannot parse it, and
    every reader that trusted the header was wrong together. Adoption must
    read the process.
    """
    import asyncio

    from servedeck import procctl, supervisor

    live["argv"] = [
        "vllm", "serve", "mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4",
    ]
    live["names"] = ["qwen38-flash-next"]
    live["backend"] = "flashnext"
    live["apply"]()

    s = supervisor.Supervisor(
        state_dir=tmp_path,
        clock=lambda: 0.0,
        launch_fn=lambda *a, **k: None,
        stop_fn=lambda h, **kw: procctl.StopResult(True, "term", 0.0, "faked"),
        history_path=tmp_path / "history.jsonl",
    )
    s._adopt_ready = types.MethodType(lambda self, pid: None, s)  # type: ignore[assignment]
    monkeypatch.setattr(capp, "_supervisor", s, raising=False)
    monkeypatch.setattr(capp, "sup", lambda: s)
    monkeypatch.setattr(procctl, "is_attributable", lambda pid: True)
    # The lying header.
    monkeypatch.setattr(capp, "_safe_config", lambda: {"BACKEND": "glm53"})
    monkeypatch.setattr(capp, "_state", lambda: {})

    asyncio.run(capp.api_adopt({"port": 8001}))

    assert s.desired.backend == "flashnext", (
        "adoption took the backend from the config header instead of the "
        "process that is actually serving"
    )
    assert s.desired.repo_id == "mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4"


def test_resolving_a_served_name_does_not_rescan_the_cache_every_poll(cache, live, monkeypatch):
    """/api/state runs on every dashboard poll and every SSE state publish.

    discover_models() walks the whole model cache — a directory listing and a
    config.json parse per repo — so resolving a served name through it on each
    call turns a status poll into a filesystem scan several times a second.
    The name changes only when a server restarts.
    """
    calls = {"n": 0}

    def counted(*a, **k):
        calls["n"] += 1
        return [_Entry("dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4")]

    monkeypatch.setattr(registry, "discover_models", counted)
    capp._SERVED_NAME_REPO.clear()

    live["argv"] = []
    live["names"] = ["GLM-5.3-Flash-ABLITERATED-NVFP4"]
    live["apply"]()

    for _ in range(20):
        assert capp._serving_identity()["repo_id"] == "dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4"
    assert calls["n"] == 1, f"scanned the model cache {calls['n']} times for one name"


# --------------------------------------------------------------------------- #
# What one status poll costs
# --------------------------------------------------------------------------- #

def _invalidate() -> None:
    """Drop any cached socket-table answer.

    Tolerates the helper being absent so the count assertion below reports the
    number of `ss` calls on code that has no cache at all, rather than an
    AttributeError that says nothing about the cost.
    """
    getattr(capp, "_invalidate_listener", lambda: None)()


def _count_socket_lookups(monkeypatch):
    """Replace the socket-table lookup with a counter.

    ``procctl.listener_pid`` shells out to ``ss`` — one fork+exec per call.
    The counter also stands in for the answer, so nothing downstream needs a
    real process.
    """
    from servedeck import procctl

    calls: list[int] = []

    def _fake(port: int) -> int:
        calls.append(port)
        return 424242            # a pid that does not exist: /proc reads fail

    monkeypatch.setattr(procctl, "listener_pid", _fake)
    return calls


@pytest.fixture
def quiet_state(monkeypatch):
    """Everything /api/state reads that is not the socket table."""
    monkeypatch.setattr(capp, "_live_boot_facts", lambda: {})
    monkeypatch.setattr(capp, "_safe_config", lambda: {})
    capp.rt.upstream_up = True
    capp.rt.serving_models = []
    _invalidate()


def test_one_status_poll_makes_one_socket_table_lookup(quiet_state, monkeypatch):
    """`ss` runs once per poll, not once per question asked of the poll.

    Four separate answers in ``_state()`` are read off the same listening
    socket: the running model id, the running max_model_len, the serving
    identity (which wants the pid as well as the argv) and the server's
    uptime. Each one used to spawn its own ``ss``. The dashboard polls
    /api/state on a timer and publishes the same payload over SSE, so that is
    four fork+execs several times a second for one fact that cannot change
    between them.
    """
    calls = _count_socket_lookups(monkeypatch)
    capp._state()
    assert len(calls) == 1, (
        f"one status poll spawned `ss` {len(calls)} times for one listening socket"
    )


def test_the_cached_listener_does_not_outlive_the_port_it_was_read_from(
    quiet_state, monkeypatch
):
    """Over-correction guard: a cache that answers for the WRONG port, or that
    never expires, makes the dashboard blind to the server it is supposed to
    be watching. A start rewrites the upstream port; the next lookup must go
    back to the socket table."""
    calls = _count_socket_lookups(monkeypatch)
    original_port = capp.rt.port
    try:
        capp._listener_argv()
        assert len(calls) == 1
        capp.rt.port = original_port + 1
        capp._listener_argv()
        assert calls == [original_port, original_port + 1], calls
        _invalidate()
        capp._listener_argv()
        assert len(calls) == 3, "an explicit invalidation must force a re-read"
    finally:
        capp.rt.port = original_port
        _invalidate()
