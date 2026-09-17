# Clients — wiring editors to the local model

Covers Codex CLI and VS Code Copilot Chat. Both talk to vLLM's OpenAI-compatible API; neither
needs server-side changes.

Current backend: **`qwen38-flash-next` on `http://localhost:8001`**.

---

## Codex CLI — `./codex-qwen.sh`

Working and verified end-to-end (`codex exec "Reply with exactly: WIRED"` → `WIRED`).

`write_provider_config()` generates `.codex/config.toml` on every launch, so edit the **script**,
not the generated file:

```toml
model_provider = "local-qwen"
model = "qwen38-flash-next"

[model_providers.local-qwen]
name = "local-qwen"
base_url = "http://localhost:8001/v1"
wire_api = "responses"
request_max_retries = 2
stream_max_retries = 2
stream_idle_timeout_ms = 900000

[agents]
max_concurrent_threads_per_session = 64
```

### Four things that will bite you

**`max_threads` is a serde alias, not a legacy fallback.** Writing both it and
`max_concurrent_threads_per_session` makes Codex reject the entire config:

```
Invalid configuration: duplicate field `max_concurrent_threads_per_session` in `agents`
```

The binary's field table contains them adjacently (`max_threadsmax_concurrent_threads_per_session`),
confirming the alias. **Write only the canonical name.**

**Keep retries low — 2, not 12.** A hung endpoint costs **~154 s per attempt**, and that
per-attempt ceiling is *not settable by any key* (`stream_idle_timeout_ms` at 3 s and 600 s
produced identical 155 s wall times). Twelve retries would freeze a turn for ~30 minutes.
`request_max_retries` exists but showed **no observable effect** on connect-level failures.

**All four keys only work inside `[model_providers.<name>]`.** At top level they parse silently
and do nothing. There is no global knob and no environment-variable override.

**`wire_api = "responses"` is mandatory.** The binary contains
`` `wire_api = "chat"` is no longer supported ``.

### Expected noise

- `warning: Model metadata for 'qwen38-flash-next' not found` — harmless; Codex has no built-in
  profile for a custom served-model name.
- **Corrected 2026-09-09 (measured against the live server, not inferred).** The `qwen3`
  reasoning parser DOES split thinking from the answer: `<think>` never appears in `content`.
  On `/v1/chat/completions` the thinking arrives in `message.reasoning`, and
  **`reasoning_content` is absent** — this build follows upstream vLLM's field rename, and no
  server flag emits the old name. On `/v1/responses` (the API Codex is forced onto) reasoning
  is a first-class `reasoning` output item carrying `summary`/`content`/`encrypted_content`,
  so Codex round-trips it structurally and is NOT exposed to the field-name mismatch.
- **Latent hazard for any chat-completions client that reads `reasoning_content`** (Copilot is
  the known example; it is not installed today). Such a client sees no thinking and cannot
  re-send it, which is exactly the mechanism behind GLM's multi-turn amnesia. The server's
  input side DOES accept a re-sent `reasoning_content` and normalizes it, so the fix is
  output-side only: front :8001 with a proxy that mirrors `reasoning` into `reasoning_content`
  (reuse `glm53-effort-proxy`, which already fixed this for GLM on :8003) before pointing any
  such client at Flash-Next.
- `-c model_reasoning_effort=…` is a Codex-side flag and does **not** reach this model's
  template. Flash-Next's own knob is `chat_template_kwargs.reasoning_effort` — see
  "Reasoning effort" below.

### Reasoning effort — already at maximum

Flash-Next's chat template supports exactly three levels, and **`xhigh` is both the maximum
and the default**. You cannot turn it up; you can only turn it down to go faster.

```json
{"model":"qwen38-flash-next","messages":[...],
 "chat_template_kwargs":{"reasoning_effort":"low"}}
```

Measured on one reasoning puzzle (same prompt, temperature 0):

| effort | completion tokens | time | outcome |
|---|---|---|---|
| `xhigh` (default) | 5,329 | 36.1 s | noticed the puzzle is self-contradictory |
| `low` | 1,266 | 10.6 s | straightforward step-by-step |

An unsupported value is rejected by the template with HTTP 400:
`Unexpected reasoning effort ultra. Supported types are xhigh (default), medium, and low.`

> ### ⚠ `xhigh` needs a large `max_tokens` or you get an EMPTY response
> At `max_tokens: 1200` this model returned **1,200 tokens billed and both `content` and
> `reasoning_content` empty** — generation was truncated mid-`<think>`, leaving the `qwen3`
> reasoning parser with an unterminated block, so it emitted nothing in either field.
> At 6,000 it completed normally.
>
> This looks exactly like a broken model or a dead server. It isn't. **Budget several thousand
> output tokens**, or drop to `low`/`medium`.

### Backend switching

`codex-qwen.sh` has `BACKEND="flashnext"`, which delegates to `vllm-qwen38next/serve.sh` (its own
venv, PLE env vars, ptrace handling). To revert to the 27B: set `BACKEND="inline"`,
`MODEL="RadixArk/Qwen3.8-27B-NVFP4"`, `PORT=8004` (moved from 8000 on 2026-09-09 — 8000 is
permanently held by the unrelated `ats-optimizer.service`).

`use_systemd()` gates the systemd paths on `BACKEND != flashnext` — without it, `start` would
launch the 27B unit on :8004 and then poll :8001 for 900 s.

---

## VS Code Copilot Chat (BYOK)

**Copilot is not currently installed.** Two extensions are present: `anthropic.claude-code` and
`openai.chatgpt`.

```bash
code --install-extension github.copilot-chat
```

Then: Command Palette → **`Chat: Manage Language Models`** → **Add Models** → **Custom Endpoint**.

| field | value |
|---|---|
| **vendor** | `customendpoint` — **not** `openai` |
| **url** | `http://localhost:8001/v1/chat/completions` |
| **apiType** | `chat-completions` |
| **apiKey** | any non-empty string (`local`) — vLLM doesn't check it |
| **id** / **name** | `qwen38-flash-next` |
| **toolCalling** | `true` |
| **vision** | `false` (see below) |
| **maxInputTokens** | `246144` |
| **maxOutputTokens** | `16000` |

**`vendor` must be `customendpoint`.** With `openai`, VS Code silently ignores
`maxInputTokens`/`maxOutputTokens` even when set
([microsoft/vscode#322216](https://github.com/microsoft/vscode/issues/322216)) — that was the
cause of a real "limited context length" report, not a server-side problem.

`maxInputTokens + maxOutputTokens` must total **≤ 262,144**; VS Code treats the sum as the
context window and defaults to something much smaller if omitted. 246144 + 16000 = exactly
262144.

`toolCalling: true` is **mandatory** — without it the model isn't shown in the picker at all.
Tool calling is verified working on this server (`finish_reason: tool_calls`, correct function
name and arguments).

`vision: false` was required because the server ran `--limit-mm-per-prompt
{"image":0,"video":0}`. **That is no longer true**: since 2026-09-10 every launch path
(`serve.sh`, `serve-tuned.sh` and `local_llm/bin/serve-model.sh`) defaults to
`{"image":2,"video":0}`, so the server accepts up to two images per prompt and the
server-side reason for `false` is gone. `vision: true` has NOT been tested through
Copilot BYOK from here, so the table above still says `false`; try `true` and expect it
to work. To go back to text-only (trading images for KV headroom), start with
`MM_LIMIT_JSON='{"image":0,"video":0}'`.

Also set **`chat.byokUtilityModelDefault`** to `Main Agent Model`, or Copilot reaches for a
cloud model for background tasks.

### Limitations worth knowing before you start

**BYOK does not cover inline code completions.** It applies to chat and utility tasks only;
completions, semantic search, and embeddings still require a GitHub account. If tab-completion
from the local model is what you wanted, BYOK doesn't provide it.

**No Copilot subscription is required** for BYOK chat itself, though org policy can restrict it.

**The server has one sequence slot** (`--max-num-seqs 1`). Copilot and Codex will serialize, and
Copilot fires background utility requests that contend. For both at once:

```bash
MAX_LEN=65536 MAX_SEQS=4 ./serve.sh
```

See [CAPACITY.md](CAPACITY.md) for the tradeoff.

### The VS Code Codex extension was rejected

It hardcodes `~/.codex/config.toml` (no `CODEX_HOME` override, so it can't be sandboxed the way
`codex-qwen.sh` is), and has an open bug where new conversations ignore custom providers and
silently fall back to `gpt-5-codex`
([codex#4558](https://github.com/openai/codex/issues/4558),
[#7971](https://github.com/openai/codex/issues/7971)). Only conversations started from the CLI
and continued in the extension are confirmed to work.

---

## Any OpenAI-compatible client

```
base_url : http://localhost:8001/v1
model    : qwen38-flash-next
api key  : any non-empty string
```

Endpoints verified live: `/v1/chat/completions`, `/v1/responses`, `/v1/models`, `/tokenize`,
`/metrics`, `/health`.

> Use `/health` for liveness, not `/v1/models` — the latter stays **200 with a dead engine**.
