"""A ``RouteTable`` for tests, plus the fixture helpers both gateway suites use.

``FakeRouteTable`` is the whole of what ``servedeck.gateway`` needs from the
registry (P1/P3 will supply the real one), written out here so the gateway
tests depend on the *Protocol* rather than on a registry that does not exist
yet — and so a change to the Protocol breaks this file loudly instead of
breaking the gateway quietly.
"""

from __future__ import annotations

import json
import pathlib

from servedeck.gateway import Route, RoutePolicies

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "reasoning"


def load(name: str) -> bytes:
    """Bytes of a fixture recorded from a live vLLM (see test_policies.py)."""
    return (FIXTURES / name).read_bytes()


def sse_events(raw: bytes) -> list[dict]:
    """Parse every ``data:`` JSON object out of an SSE byte stream."""
    out = []
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload in (b"[DONE]", b""):
            continue
        out.append(json.loads(payload))
    return out


def reassemble(events: list[dict], field: str) -> str:
    """Concatenate one delta field across a parsed stream."""
    parts = []
    for ev in events:
        for ch in ev.get("choices") or []:
            v = (ch.get("delta") or {}).get(field)
            if isinstance(v, str):
                parts.append(v)
    return "".join(parts)


class FakeRouteTable:
    """An in-memory ``gateway.RouteTable``.

    ``main_name`` names which of the routes holds the exclusive main GPU slot;
    it may name a route that is **not** live, which is the state the 503 body
    exists to describe (the main model is booting).
    """

    def __init__(self, routes: list[Route], main_name: str | None = None) -> None:
        self.routes = list(routes)
        self.main_name = main_name
        self._by_name: dict[str, Route] = {}
        for route in self.routes:
            for name in route.served_names():
                self._by_name[name] = route

    # -- gateway.RouteTable ------------------------------------------------
    def resolve(self, name: str) -> Route | None:
        return self._by_name.get(name)

    def live_routes(self) -> list[Route]:
        return [r for r in self.routes if r.live]

    def main(self) -> Route | None:
        if self.main_name is None:
            return None
        return self._by_name.get(self.main_name)

    def known_names(self) -> list[str]:
        return list(self._by_name)


def lfm2_route(*, port: int = 8007, live: bool = True, **policy_kwargs) -> Route:
    """The live LFM2.5-350M resident, as the e2e suite and several unit tests
    describe it: one id, one alias, 32,768 tokens of context."""
    return Route(
        model_id="LFM2.5-350M",
        port=port,
        live=live,
        aliases=("lfm2",),
        policies=RoutePolicies(ctx=32768, **policy_kwargs),
    )
