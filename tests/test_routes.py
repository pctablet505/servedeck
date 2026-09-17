"""servedeck.routes — the join between the registry, the gateway and control.

Everything here runs against a real ``models.Registry`` built from a tmp
``models.toml`` and a fake control, because the whole value of this module is
that it agrees with P1 about what a name means and with P2 about what a
``Route`` is. A hand-written stub registry would let the two drift apart
silently, which is the class of bug (R1) the redesign exists to remove.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from servedeck import models as _models
from servedeck.gateway import Route, RouteTable
from servedeck.routes import LiveView, RegistryRoutes, make_ctx_resolver

TOML = """
[gpu]
total_mib = 100000
margin_mib = 1024

[builds.stock]
venv = "/opt/stock"
cuda_home = "/opt/stock/lib/python3.13/site-packages/nvidia/cu13"

[defaults.env]
SHARED = "1"

[models.main1]
id = "Big-Model"
aliases = ["big", "bigmodel"]
repo = "org/big"
slot = "main"
port = 19001
build = "stock"
ctx = 32768
min_output_tokens = 2048
reasoning = { parser = "glm", mirror_content = true }

[models.main1.presets.big-high]
reasoning_effort = "high"

[models.main1.presets.big-low]
reasoning_effort = "low"
max_tokens = 512

[models.res1]
id = "Small-Model"
aliases = ["small"]
repo = "org/small"
slot = "resident"
vram_mib = 3300
port = 19002
build = "stock"
ctx = 4096

[models.main2]
id = "Other-Main"
repo = "org/other"
slot = "main"
port = 19003
build = "stock"
ctx = 8192
"""


@dataclass
class FakeLive:
    """The shape ``control.LiveModel`` has, as routes.set_live consumes it."""

    key: str
    unit: str
    ready: bool
    state: str = "active"
    sub_state: str = "running"
    pid: int = 4242
    restarts: int = 0
    port: int | None = None
    unknown: bool = False


class FakeControl:
    def __init__(self, rows: list[FakeLive] | None = None) -> None:
        self.rows = rows or []
        self.calls = 0
        self.raises: Exception | None = None

    def live(self) -> list[FakeLive]:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.rows


@pytest.fixture
def registry(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(TOML)
    return _models.load(path)


@pytest.fixture
def routes(registry):
    return RegistryRoutes(registry, FakeControl())


def live(routes_obj, *keys: str, ready: bool = True) -> None:
    routes_obj.set_live(
        [FakeLive(key=k, unit=f"model-{k}", ready=ready) for k in keys]
    )


# --------------------------------------------------------------------------
# The Protocol
# --------------------------------------------------------------------------


def test_registry_routes_satisfies_the_gateway_protocol(routes) -> None:
    """``RouteTable`` is ``runtime_checkable``, so this is a real check that
    all four methods exist — and it is the assertion that fails loudly if P2
    renames one, instead of the gateway failing quietly at request time."""
    assert isinstance(routes, RouteTable)


# --------------------------------------------------------------------------
# resolve(): id, alias, preset, key
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Big-Model", "big", "bigmodel", "main1"])
def test_every_name_for_a_model_resolves_to_it(routes, name) -> None:
    """The 404-after-rename class of bug, closed by construction: every name a
    client has ever been configured with reaches the same weights."""
    route = routes.resolve(name)
    assert route is not None
    assert route.model_id == "Big-Model"
    assert route.port == 19001


def test_an_unknown_name_resolves_to_none(routes) -> None:
    assert routes.resolve("no-such-model") is None


def test_resolution_does_not_depend_on_liveness(routes) -> None:
    """A registered model that is not running must still RESOLVE.

    This is the 404/503 distinction: 404 means "reconfigure yourself", 503
    means "wait". Collapsing them sends an operator hunting for a config file
    every time a model is booting, which is exactly what happened on 09-11.
    """
    assert routes.resolve("big") is not None
    assert routes.resolve("big").live is False
    live(routes, "main1")
    assert routes.resolve("big").live is True


def test_a_preset_carries_its_overlay_and_a_plain_alias_does_not(routes) -> None:
    """A preset is not a synonym: ``big-high`` is the same weights with a
    different request. An alias that carried the last preset's overlay would
    silently apply a reasoning effort nobody asked for."""
    preset = routes.resolve("big-high")
    assert preset.policies.effort_overlay == {"reasoning_effort": "high"}

    other = routes.resolve("big-low")
    assert other.policies.effort_overlay == {"reasoning_effort": "low", "max_tokens": 512}

    alias = routes.resolve("big")
    assert alias.policies.effort_overlay is None
    assert routes.resolve("Big-Model").policies.effort_overlay is None


def test_a_preset_route_keeps_the_underlying_model_id(routes) -> None:
    """The gateway rewrites the request's ``model`` to this before forwarding;
    if it were the preset name, upstream would 404 for a model it is serving."""
    assert routes.resolve("big-high").model_id == "Big-Model"
    assert routes.resolve("big-high").port == 19001


# --------------------------------------------------------------------------
# policies
# --------------------------------------------------------------------------


def test_mirror_reasoning_comes_from_the_models_reasoning_config(routes) -> None:
    assert routes.resolve("big").policies.mirror_reasoning is True
    assert routes.resolve("small").policies.mirror_reasoning is False


def test_min_output_tokens_and_ctx_travel_with_the_route(routes) -> None:
    """Both are needed together: the floor is clamped to ctx, so a route that
    carried one without the other could raise an output budget above the
    context the model has."""
    route = routes.resolve("big")
    assert route.policies.min_output_tokens == 2048
    assert route.policies.ctx == 32768
    assert routes.resolve("small").policies.min_output_tokens is None


def test_aliases_and_presets_are_kept_apart_on_the_route(routes) -> None:
    route = routes.resolve("Big-Model")
    assert route.aliases == ("big", "bigmodel")
    assert route.presets == ("big-high", "big-low")
    assert route.served_names() == ("Big-Model", "big", "bigmodel", "big-high", "big-low")


# --------------------------------------------------------------------------
# ctx = "native"
# --------------------------------------------------------------------------


def test_native_ctx_is_resolved_through_the_resolver(tmp_path) -> None:
    path = tmp_path / "models.toml"
    path.write_text(
        TOML.replace("[models.main1]\nid = \"Big-Model\"", "[models.main1]\nid = \"Big-Model\"").replace(
            "ctx = 32768", "ctx = \"native\"", 1
        )
    )
    registry = _models.load(path)
    routes = RegistryRoutes(registry, FakeControl(), ctx_resolver=lambda m: 262144)
    assert routes.resolve("big").policies.ctx == 262144


def test_an_unreadable_native_ctx_is_zero_and_says_why(tmp_path) -> None:
    """0, never a default.

    A substituted context length is indistinguishable on screen from a read
    one, and a client that sizes its prompts against it overruns a model that
    never had that much. The zero is paired with a sentence the page prints.
    """
    path = tmp_path / "models.toml"
    path.write_text(TOML.replace("ctx = 32768", 'ctx = "native"', 1))
    registry = _models.load(path)
    routes = RegistryRoutes(registry, FakeControl())  # real resolver, repo not on disk
    assert routes.resolve("big").policies.ctx == 0
    assert "hub cache" in (routes.ctx_error("main1") or "")


def test_the_ctx_resolver_reads_each_model_once() -> None:
    """``native_ctx`` reads a JSON file. ``/v1/models`` asks for ctx on every
    request; without the memo that is a stat+read per model per request for a
    value that cannot change while the process runs."""
    calls: list[str] = []

    def fake_native(repo, hub_dir=None):
        calls.append(repo)
        return 1024

    import servedeck.models as models_mod

    original = models_mod.native_ctx
    models_mod.native_ctx = fake_native
    try:
        resolver = make_ctx_resolver()
        model = _models.Model(
            key="k", id="I", repo="org/r", slot="main", port=1, build="b", ctx="native"
        )
        assert resolver(model) == 1024
        assert resolver(model) == 1024
        assert resolver(model) == 1024
    finally:
        models_mod.native_ctx = original
    assert calls == ["org/r"], calls


# --------------------------------------------------------------------------
# live_routes / main / known_names
# --------------------------------------------------------------------------


def test_live_routes_lists_only_ready_models(routes) -> None:
    """Ready, not merely started.

    A unit 40 s into a boot exists but answers nothing. Advertising it in
    ``/v1/models`` hands a client a name that yields a connection refused,
    instead of the 503 with Retry-After that tells it to wait.
    """
    routes.set_live(
        [
            FakeLive(key="main1", unit="model-main1", ready=False, sub_state="start"),
            FakeLive(key="res1", unit="model-res1", ready=True),
        ]
    )
    assert [r.model_id for r in routes.live_routes()] == ["Small-Model"]


def test_live_routes_is_empty_before_anything_refreshes(routes) -> None:
    """The safe direction. An un-refreshed table must report nothing live, so
    the gateway 503s rather than proxying to a port with nothing behind it."""
    assert routes.live_routes() == []
    assert routes.main() is None


def test_main_names_the_holder_even_while_it_is_still_booting(routes) -> None:
    """The whole content of the 503 body is "main slot: X, still starting"."""
    routes.set_live([FakeLive(key="main1", unit="model-main1", ready=False)])
    main = routes.main()
    assert main is not None and main.model_id == "Big-Model"
    assert main.live is False


def test_main_ignores_a_resident_that_is_running(routes) -> None:
    live(routes, "res1")
    assert routes.main() is None


def test_known_names_lists_every_name_resolve_accepts(routes) -> None:
    """The 404 body is built from this; a name missing here is a name the
    error message tells the client it cannot use while resolve accepts it."""
    names = routes.known_names()
    assert names == [
        "Big-Model", "big", "bigmodel", "big-high", "big-low",
        "Small-Model", "small",
        "Other-Main",
    ]
    for name in names:
        assert routes.resolve(name) is not None, name


# --------------------------------------------------------------------------
# refresh()
# --------------------------------------------------------------------------


def test_refresh_pulls_the_snapshot_from_control(registry) -> None:
    control = FakeControl([FakeLive(key="res1", unit="model-res1", ready=True)])
    routes = RegistryRoutes(registry, control)
    assert routes.live() == {}
    snapshot = routes.refresh()
    assert control.calls == 1
    assert set(snapshot) == {"res1"}
    assert routes.is_live("res1")


def test_a_failing_refresh_keeps_the_previous_snapshot(registry) -> None:
    """One flaky ``systemctl`` call must not take a serving model off the
    gateway. Blanking the table on an error would turn a transient failure into
    two seconds of 503 for every client."""
    control = FakeControl([FakeLive(key="res1", unit="model-res1", ready=True)])
    routes = RegistryRoutes(registry, control)
    routes.refresh()
    control.raises = OSError("systemctl exploded")
    routes.refresh()
    assert routes.is_live("res1"), "the snapshot was dropped on a transient failure"


def test_an_unknown_unit_is_reported_but_never_resolvable(registry) -> None:
    """A ``model-*`` unit whose key is not in the registry has no spec: no
    slot, no port, no way to tell a stray from a model. It shows up in the
    snapshot for the page to warn about and in nothing else."""
    control = FakeControl([FakeLive(key="mystery", unit="model-mystery", ready=False, unknown=True)])
    routes = RegistryRoutes(registry, control)
    routes.refresh()
    assert routes.live()["mystery"].unknown is True
    assert routes.resolve("mystery") is None
    assert routes.live_routes() == []
    assert routes.main() is None


def test_set_live_replaces_rather_than_merges(routes) -> None:
    """The snapshot is swapped wholesale so a reader on the event loop cannot
    see a half-updated table — and so a model that has gone away actually
    goes away instead of lingering as a stale True."""
    live(routes, "main1", "res1")
    assert set(routes.live()) == {"main1", "res1"}
    live(routes, "res1")
    assert set(routes.live()) == {"res1"}
    assert routes.is_live("main1") is False


def test_live_view_renders_the_unit_state_the_page_shows(routes) -> None:
    view = LiveView(key="k", unit="model-k", ready=True, state="active", sub_state="running", pid=1, restarts=0)
    assert view.unit_state == "active (running)"
    assert LiveView(key="k", unit="u", ready=False, state="failed", sub_state="", pid=0, restarts=2).unit_state == "failed"
    assert LiveView(key="k", unit="u", ready=False, state="", sub_state="", pid=0, restarts=0).unit_state == "unknown"


def test_routes_with_no_control_never_reports_anything_live(registry) -> None:
    routes = RegistryRoutes(registry, None)
    assert routes.refresh() == {}
    assert routes.live_routes() == []
    assert isinstance(routes.resolve("big"), Route)
