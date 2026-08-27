"""Coldstart smoke test — SPEC.md §8 "SMOKE TEST".

Two requests, run THROUGH the gateway (http://127.0.0.1:8010 by default,
never directly at :8001/:8000) so this proves the gateway path too, not
just that vLLM itself is alive:

1. Tool call: one function ``get_time(timezone)``, ``tool_choice="auto"``,
   ``temperature=0``. Pass iff HTTP 200 and
   ``choices[0].message.tool_calls[0].function.name == "get_time"`` and
   its ``arguments`` string parses as JSON.
2. Plain: "Reply with exactly: OK". Pass iff ``message.content`` OR
   ``message.reasoning_content`` is non-empty — and this module always
   reports WHICH field it was. The qwen3 reasoning parser routes plain
   output to ``reasoning_content``; an empty ``content`` with everything
   actually in ``reasoning_content`` looks exactly like a model failure
   if you only ever check ``content`` (SETUP.md:277). Silently checking
   only ``content`` would misreport a healthy backend as broken.

Both steps report the real HTTP status and body on failure — never just
"request failed".

This module is import-safe from an API endpoint (`POST /api/smoke`, owned
by api.py — not this file) via :func:`run_smoke`, and directly runnable:
    python -m servedeck.smoke [--base-url URL] [--json]
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8010"

_GET_TIME_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current time in a given IANA timezone.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone name, e.g. 'Asia/Tokyo'.",
                },
            },
            "required": ["timezone"],
        },
    },
}

_TOOL_CALL_PROMPT = "What time is it right now in Tokyo? Use the get_time tool to find out."
_PLAIN_PROMPT = "Reply with exactly: OK"


@dataclass
class SmokeStepResult:
    name: str
    passed: bool
    http_status: int | None
    detail: str
    field_used: str | None = None  # "content" | "reasoning_content" | None
    raw_body: str | None = None  # populated on failure, so the real body is always visible


@dataclass
class SmokeResult:
    ok: bool
    model: str | None
    base_url: str
    steps: list[SmokeStepResult] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated, {len(text)} bytes total]"


async def _resolve_model(client: httpx.AsyncClient, base_url: str) -> tuple[str | None, SmokeStepResult | None]:
    """GET /v1/models through the gateway. Returns (model_id, None) on
    success, or (None, a failed SmokeStepResult) that the caller should
    surface as-is — both smoke steps need a real model id to send, and
    there is nothing meaningful to test if this fails."""
    url = f"{base_url}/v1/models"
    try:
        resp = await client.get(url, timeout=10.0)
    except httpx.HTTPError as exc:
        return None, SmokeStepResult(
            name="resolve_model",
            passed=False,
            http_status=None,
            detail=f"GET {url} failed: {exc}",
        )
    if resp.status_code != 200:
        return None, SmokeStepResult(
            name="resolve_model",
            passed=False,
            http_status=resp.status_code,
            detail=f"GET {url} -> HTTP {resp.status_code}",
            raw_body=_truncate(resp.text),
        )
    try:
        data = resp.json()
        model_id = data["data"][0]["id"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return None, SmokeStepResult(
            name="resolve_model",
            passed=False,
            http_status=resp.status_code,
            detail=f"GET {url} returned an unparsable body ({exc})",
            raw_body=_truncate(resp.text),
        )
    if not isinstance(model_id, str) or not model_id:
        return None, SmokeStepResult(
            name="resolve_model",
            passed=False,
            http_status=resp.status_code,
            detail=f"GET {url} -> data[0].id is not a usable model id: {model_id!r}",
            raw_body=_truncate(resp.text),
        )
    return model_id, None


async def _post_chat_completion(
    client: httpx.AsyncClient, base_url: str, payload: dict, *, timeout_s: float
) -> tuple[httpx.Response | None, str | None]:
    """Returns (response, None) on a completed HTTP exchange (any status
    code), or (None, error_detail) if the request never completed at all
    (connection refused, timed out, etc.)."""
    url = f"{base_url}/v1/chat/completions"
    try:
        resp = await client.post(url, json=payload, timeout=timeout_s)
    except httpx.HTTPError as exc:
        return None, f"POST {url} failed: {exc}"
    return resp, None


async def _step_tool_call(client: httpx.AsyncClient, base_url: str, model: str, *, timeout_s: float) -> SmokeStepResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _TOOL_CALL_PROMPT}],
        "tools": [_GET_TIME_TOOL],
        "tool_choice": "auto",
        "temperature": 0,
    }
    resp, err = await _post_chat_completion(client, base_url, payload, timeout_s=timeout_s)
    if resp is None:
        return SmokeStepResult(name="tool_call", passed=False, http_status=None, detail=err or "request failed")
    if resp.status_code != 200:
        return SmokeStepResult(
            name="tool_call",
            passed=False,
            http_status=resp.status_code,
            detail=f"HTTP {resp.status_code}",
            raw_body=_truncate(resp.text),
        )
    try:
        data = resp.json()
        message = data["choices"][0]["message"]
        tool_calls = message["tool_calls"]
        if not tool_calls:
            raise ValueError("tool_calls is empty")
        function = tool_calls[0]["function"]
        name = function["name"]
        args_raw = function["arguments"]
        parsed_args = json.loads(args_raw)
    except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        return SmokeStepResult(
            name="tool_call",
            passed=False,
            http_status=resp.status_code,
            detail=f"response did not contain a well-formed tool call: {exc}",
            raw_body=_truncate(resp.text),
        )
    if name != "get_time":
        return SmokeStepResult(
            name="tool_call",
            passed=False,
            http_status=resp.status_code,
            detail=f"tool_calls[0].function.name == {name!r}, expected 'get_time'",
            raw_body=_truncate(resp.text),
        )
    return SmokeStepResult(
        name="tool_call",
        passed=True,
        http_status=200,
        detail=f"called get_time(arguments={parsed_args!r})",
    )


async def _step_plain(client: httpx.AsyncClient, base_url: str, model: str, *, timeout_s: float) -> SmokeStepResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _PLAIN_PROMPT}],
    }
    resp, err = await _post_chat_completion(client, base_url, payload, timeout_s=timeout_s)
    if resp is None:
        return SmokeStepResult(name="plain", passed=False, http_status=None, detail=err or "request failed")
    if resp.status_code != 200:
        return SmokeStepResult(
            name="plain",
            passed=False,
            http_status=resp.status_code,
            detail=f"HTTP {resp.status_code}",
            raw_body=_truncate(resp.text),
        )
    try:
        data = resp.json()
        message = data["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return SmokeStepResult(
            name="plain",
            passed=False,
            http_status=resp.status_code,
            detail=f"response did not contain choices[0].message: {exc}",
            raw_body=_truncate(resp.text),
        )
    content = (message.get("content") or "").strip()
    reasoning = (message.get("reasoning_content") or "").strip()
    if content:
        return SmokeStepResult(
            name="plain", passed=True, http_status=200, detail="non-empty content", field_used="content"
        )
    if reasoning:
        return SmokeStepResult(
            name="plain",
            passed=True,
            http_status=200,
            detail=(
                "content was EMPTY; text landed in reasoning_content instead "
                "(qwen3 reasoning parser routes plain output there — SETUP.md:277, "
                "NOT a model failure)"
            ),
            field_used="reasoning_content",
        )
    return SmokeStepResult(
        name="plain",
        passed=False,
        http_status=200,
        detail="both content and reasoning_content are empty",
        raw_body=_truncate(resp.text),
    )


async def run_smoke(
    base_url: str | None = None,
    *,
    client: httpx.AsyncClient | None = None,
    timeout_s: float = 120.0,
) -> SmokeResult:
    """Run both smoke steps through the gateway at `base_url` (default:
    $SERVEDECK_URL or http://127.0.0.1:8010). Pass `client` to reuse an
    existing httpx.AsyncClient (e.g. a test's MockTransport-backed one,
    or an API endpoint's shared client) — otherwise one is created and
    closed here."""
    resolved_base_url = (base_url or os.environ.get("SERVEDECK_URL") or DEFAULT_BASE_URL).rstrip("/")
    started = time.time()
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient()
    try:
        model, resolve_failure = await _resolve_model(client, resolved_base_url)
        if model is None:
            steps = [resolve_failure] if resolve_failure is not None else []
            return SmokeResult(
                ok=False,
                model=None,
                base_url=resolved_base_url,
                steps=steps,
                started_at=started,
                finished_at=time.time(),
            )
        tool_step = await _step_tool_call(client, resolved_base_url, model, timeout_s=timeout_s)
        plain_step = await _step_plain(client, resolved_base_url, model, timeout_s=timeout_s)
        steps = [tool_step, plain_step]
        return SmokeResult(
            ok=all(s.passed for s in steps),
            model=model,
            base_url=resolved_base_url,
            steps=steps,
            started_at=started,
            finished_at=time.time(),
        )
    finally:
        if owns_client:
            await client.aclose()


def _format_report(result: SmokeResult) -> str:
    lines = [
        f"Coldstart smoke test — base_url={result.base_url} model={result.model!r} "
        f"— {'PASS' if result.ok else 'FAIL'} ({result.finished_at - result.started_at:.1f}s)"
    ]
    for s in result.steps:
        status = "PASS" if s.passed else "FAIL"
        extra = f" [field={s.field_used}]" if s.field_used else ""
        lines.append(f"  [{status}] {s.name}: {s.detail}{extra}")
        if not s.passed:
            lines.append(f"    http_status={s.http_status}")
            if s.raw_body:
                lines.append(f"    body: {s.raw_body}")
    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m servedeck.smoke")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Coldstart gateway base URL (default: $SERVEDECK_URL or http://127.0.0.1:8010)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a human-readable report")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="per-request timeout in seconds")
    args = parser.parse_args(argv)

    result = asyncio.run(run_smoke(args.base_url, timeout_s=args.timeout_s))

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(_format_report(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
