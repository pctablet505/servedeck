# Clients

Every tool on this box talks to one URL — `http://127.0.0.1:8010/v1` — and asks for the
model named `main`. Nothing else is a supported wiring.

## The contract

| | |
|---|---|
| base URL | `http://127.0.0.1:8010/v1` (VS Code's file is written with `localhost:8010`; two deliberate constants in `wire.py:54-59` so an edit cannot make them agree by accident) |
| API key | none is checked — no model unit carries `--api-key` and the gateway has no auth. Send any non-empty string for clients that insist; `wire` writes `local-no-auth` for Kimi |
| model names | `GET /v1/models`, which lists every name each **ready** model answers to, with `root` = the id vLLM was started with and `max_model_len` = its context |
| `main` / `local` | the gateway aliases for whatever holds the exclusive main slot (`routes.py:127`). Client configs name these, so `servedeck switch` does not strand every client |
| reasoning | arrives as **both** `reasoning` and `reasoning_content`, same value, in the JSON body and in every SSE delta (verified live 2026-09-18: 73 characters in each, identical) |
| tool calls | parsed server-side by the engine, per model: `qwen3_coder` (flashnext), `qwen3_xml` (qwen27b), `glm47` (glm53), `lfm2`. A client sends ordinary OpenAI `tools` and gets `tool_calls` back |

Live 2026-09-18 with Flash-Next in the slot, `GET /v1/models` returns five names —
`Qwen3.8-Flash-Next-Uncensored-NVFP4`, `qwen38-flash-next`, `flashnext`, `main`, `local` —
all with the same `root`. A name may be an id, an alias, a preset or a registry key; an
alias or preset is rewritten to the started id before the request goes upstream, or the
engine would 404 for a model it is serving. Errors are OpenAI-shaped: **404** means the
client is misconfigured and the body lists every name that would have worked; **503** with
`Retry-After: 15` means the model is registered but booting, or another model holds the main
slot (the message names which); **502** means the registry says live and the socket
disagrees.

Effort presets are served *names*, not request flags: `glm53-flash-low` / `-high` / `-max`
appear as their own ids, and the gateway merges the preset's overlay into
`chat_template_kwargs` with `setdefault`, so an explicit caller value always wins — that is
how a client picks an effort level without knowing `chat_template_kwargs` exists. Flash-Next
instead takes `chat_template_kwargs.reasoning_effort` directly, with exactly three values:
`xhigh` (default and maximum), `medium`, `low`; anything else is an HTTP 400 naming the
three (verified live 2026-09-18). Codex's `-c model_reasoning_effort` never reaches a
template.

## The reasoning-erasure rule

A client that does not send the previous turn's reasoning back makes the model re-read the
thread as fresh. The gateway's mirror cannot fix that from the server side: the mirror makes
the thinking *available* to a client, it cannot make the client *send it back*.

Measured on GLM-5.3 on 2026-09-02, two independent ways, not re-measured since. As Copilot
sent it, a logprob probe put 3.32% of the mass on the model concluding no task had been
given; with the dropped reasoning restored, 0.00% — a 33,000x reduction, and the 3.32%
matches the 4.5% per-turn greeting rate measured separately over 3,898 real transcript
turns. Across 12 varied conversation states the restored arm was 0.00% in all 12. The
failure is specific to **tool-terminated** turns (0.12-2.18% there, exactly 0.00% in all six
cases ending on a user message), which is why it bites agent mode and never chat, and why it
looks like a per-turn coin flip rather than long-thread decay. A near-matched control
(Flash-Next: also abliterated, also NVFP4, same client, same 51 tools) did 3,788 turns with
zero greetings against GLM's 5 in 112, Fisher p=1.8e-8, ruling out the checkpoint. A
multi-hour session with the fix active is still unconfirmed.

The server-side counterpart (`restore_reasoning` in `models.toml`) is **off**, including for
glm53, on a GPU-fault correlation — see [models/glm53.md](models/glm53.md). The live fix is
the client-side one: make the client replay reasoning.

## Codex

`~/.codex/config.toml`, generated: top-level `model = "main"`,
`model_provider = "servedeck"`, `model_context_window = 262144`, plus
`[model_providers.servedeck]`.

`[profiles.<key>]` tables are **refused** by the installed binaries (codex-cli 0.150.1 and
0.154.0-alpha.6.2): ``--profile `flashnext` cannot be used while config.toml contains legacy
`profile = ...` or `[profiles.flashnext]` config; move those settings into
<home>/flashnext.config.toml``. Per-profile *files* are the supported shape now, and a
profile per model buys nothing here — one model serves at a time, so the name to select is
the slot. `wire` removes the profile tables it wrote before 2026-09-18.
`model_max_output_tokens` is gone with them: `strings | grep -c` finds that key in neither
installed binary, so it was noise.

`wire_api = "responses"` is mandatory — the binary contains ``` `wire_api = "chat"` is no
longer supported ```. Without explicit metadata Codex warns `Model metadata ... not found`,
guesses the context window and auto-compacts early, which is why `model_context_window` is
written; it carries the *narrowest* main-slot context (262,144), because the slot may hold
any main model and a prompt sized against GLM's 327,680 would be rejected by the engine.
The provider's three patience keys:

```toml
request_max_retries    = 2
stream_max_retries     = 2
stream_idle_timeout_ms = 900000
```

They work only inside `[model_providers.<name>]` — at top level they parse silently and do
nothing, and there is no environment override. `900000` rides out a cold boot (4-10 min
here; a mid-session restart is ordinary). The retry counts must stay at 2: a hung endpoint
costs ~154 s per attempt and that ceiling is not settable by any key —
`stream_idle_timeout_ms` at 3 s and at 600 s both gave 155 s wall times — so 12 retries
freezes a turn for ~30 minutes (measured 2026-09 in the coldstart investigation; not
re-measured). `request_max_retries` showed no observable effect on connect-level failures.

Codex speaks `/v1/responses`, where reasoning is a first-class output item, so it
round-trips thinking structurally and is not exposed to the erasure above. The gateway
treats that endpoint accordingly: the mirror is off there (nothing is missing), and so is
every GLM repair — `glm_policies.begin()` returns `None` for the responses API because the
envelope is a different shape and none of the transforms is written for it. The effort
overlay and output floor *do* still apply, so `glm53-flash-high` keeps meaning high effort;
a Codex session on GLM gets effort and floor but no tool-tag sanitising.

One hand-written table `wire` preserves: `[agents] max_concurrent_threads_per_session = 64`.
`max_threads` is a serde *alias* for that key, not a legacy fallback — writing both makes
Codex reject the whole config (``duplicate field `max_concurrent_threads_per_session` in
`agents` ``). `[code_mode_host]` and `[multi_agent]` were live-tested and silently ignored;
`features.multi_agent_v2` 400s every turn against this vLLM `/v1/responses` build.

`codex resume --last` replays the model name from its own session state, which produced a
live 404 storm after a model swap — the origin of the names-forever rule: a name any client
was ever configured with stays resolvable, and the 404 body lists them all.

**There are two Codex homes here.** `~/.codex` is the one `wire` maintains.
`~/Projects/local_llm/.codex` is reached whenever `CODEX_HOME` is exported, which
`codex-qwen.sh:106` does. Its config was hand-converted to the gateway on 2026-09-18, but
`codex-qwen.sh` regenerates the file on every launch with provider `local-qwen` and
`base_url = http://localhost:<model port>/v1` (`codex-qwen.sh:835,920-926`), undoing the
conversion and dropping the mirror. Do not launch through that script. Converging the homes
means moving ~1 GB of live session and thread history, so it waits for a moment when no
codex is running. Note that `doctor`'s `codex config` row reads model references out of
`[profiles]` only, so with the profiles retired it reports `present, no model references
found` — an OK that does not check the top-level `model`/`model_provider`.

## Kimi CLI

`~/.kimi-code/config.toml`, partly generated: `default_model = "servedeck/main"`, one
`[providers.servedeck]`, one `[models."servedeck/<key>"]` per registry model plus
`servedeck/main`, and `[thinking] enabled = true`.

Capabilities are derived from the registry — `tool_use` for a tool parser, `thinking` +
`always_thinking` for a reasoning parser, `image_in` when `vision` is true. `servedeck/main`
carries the *union* of the main models' capabilities and the narrowest main context
(262,144), since any of them may hold the slot. `[thinking] enabled = true` is the owner's
"reasoning always on" rule; other keys there are kept (`effort = "low"` is live). `wire`
repoints `default_model` only when it already names a `servedeck/` model or is absent — a
cloud default chosen on purpose is the operator's to keep — leaves the four `api.kimi.com`
entries untouched (`doctor` reports them off-box and unprobed), and preserves two
hand-written aliases (`local/flash-next`, `local/flash-next-proxy`, both the Flash-Next id
through the gateway) that keep pre-cutover sessions resolving.

## VS Code chat

`~/.config/Code/User/chatLanguageModels.json` — one group named `servedeck`; any other group
in the array passes through untouched. Live 2026-09-18 it holds eight entries: `main` first,
then the four model ids, then the three glm53 presets, all at
`http://localhost:8010/v1/chat/completions`.

| field | value | why |
|---|---|---|
| `vendor` | `customendpoint` | with `openai`, VS Code silently ignores `maxInputTokens`/`maxOutputTokens` ([vscode#322216](https://github.com/microsoft/vscode/issues/322216)) — the real cause of a "limited context length" report, not the server |
| `apiType` | `chat-completions` | |
| `toolCalling` | `true` | mandatory: without it the model never appears in the picker |
| `thinking` | `true` when the model has a reasoning parser | makes the extension **replay** the previous turn's reasoning |
| `contextWindow` | the model's context | the schema's own source of truth |
| `maxInputTokens` + `maxOutputTokens` | sums to exactly `contextWindow` | VS Code budgets prompts from the split |

The `thinking` flag is the fix for the erasure rule above. Until 2026-09-18 the generated
entries lacked it, and the extension then dropped `reasoning_content` on replay and sent its
own `cot_summary`/`cot_id`, which vLLM ignores — the measured cause of agent amnesia here.

The split has no margin: `wire.py:109` writes `maxInputTokens = ctx - maxOutputTokens`, so
Flash-Next is 230,144 + 32,000 = 262,144 and glm53 is 294,912 + 32,768 = 327,680. History
for whoever next sees a strange truncation: this file once advertised 500,000 for the glm53
aliases against a 262,144 server. A **window reload** is needed after `wire --apply`.

Two live gaps, 2026-09-18: `github.copilot-chat` is **not installed** (only
`anthropic.claude-code` and `openai.chatgpt` are), so these entries are wired ahead of the
client that consumes them — `code --install-extension github.copilot-chat`, then Command
Palette → `Chat: Manage Language Models`; and `chat.byokUtilityModelDefault` is not set in
`settings.json`, so set it to `Main Agent Model` or Copilot reaches for a cloud model for
background tasks.

BYOK covers **chat and utility tasks only** — inline completions, semantic search and
embeddings still require a GitHub account, so tab-completion from the local model is not
something it provides — and needs no subscription for chat itself, though org policy can
restrict it. The VS Code *Codex* extension was rejected: it hardcodes
`~/.codex/config.toml` with no `CODEX_HOME` override, and new conversations ignore custom
providers and fall back to a cloud model ([codex#4558](https://github.com/openai/codex/issues/4558),
[#7971](https://github.com/openai/codex/issues/7971)).

## Adding another tool

An OpenAI-compatible client needs three things: base URL `http://127.0.0.1:8010/v1`, model
name `main`, any non-empty API key. If it can be told a context window, read it from
`max_model_len` in `GET /v1/models` rather than a second config file. If it stores assistant
reasoning, make it send it back.

Never point a client at a model's own port: it loses the mirror and the effort and floor
policies, it dies the moment `servedeck switch` changes the card, and it is invisible to both
`wire` and `doctor`. Two on the box still do it — `ats-optimizer`
(`config/app.yaml:13-15`, `http://localhost:8001/v1`, model `qwen38-flash-next`) and
`~/Projects/local_llm/.codex` whenever its launcher regenerates it. Port 8000 is
ats-optimizer's own service, binds `0.0.0.0` rather than loopback, and servedeck never
touches it; `models.toml` validation rejects 8000 and 8010 as model ports outright.

What the gateway does not speak today: **no `/v1/embeddings`** — the request is forwarded and
the engine 404s `{"detail":"Not Found"}`, because a generation model registers no embeddings
route (verified live 2026-09-18), and nothing on this box serves embeddings. **`/v1/messages`
exists but is undocumented and unverified** — the engine registers it (with
`/v1/messages/count_tokens`) and the catch-all forwards it, but no client here uses it, no
policy is written for its shape, and a GET returns 405 from upstream. **No Ollama surface**,
and `/api/*` is servedeck's own control plane so it cannot host one; a tool that speaks only
Ollama has to be reconfigured or wrapped. Browser-based tools can GET anything and POST
`/v1/*` cross-origin, but a cross-origin write to `/api/*` is refused with 403
(`app.py:_same_origin_only`): the control API is writable only from its own page or a
terminal.

```bash
servedeck wire            # dry-run: the exact diff --apply would write
servedeck wire --apply    # writes it, after a dated backup under state/backups/<YYYY-MM-DD>/
servedeck doctor          # every wired name, probed against GET /v1/models
```

`wire` edits only the tables it owns, each marked `# servedeck-generated by servedeck wire`
— every other provider, group, comment and hand-written key survives byte-for-byte — and
only those three files are targets. Any other client on the box is invisible to it.
