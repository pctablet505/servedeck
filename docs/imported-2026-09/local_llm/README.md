# local_llm

One entry point for the local model stack: a large model served on this
workstation, plus the clients that talk to it.

```bash
./llm start                 # bring everything up, wait until it answers
./llm status                # what is running, and how much room is left
./llm chat "hello"          # prove it works
./llm codex                 # a coding agent wired to the local model
./llm stop
```

Everything else in this directory is either configuration or history. If you
only read one thing, read `./llm help`.

---

## What actually runs

| piece | port | what it is |
|---|---|---|
| **model server** | configured `PORT` | vLLM backend selected by `.config` (Qwen Flash-Next or GLM). |
| **effort proxy** | GLM only | Optional reasoning-effort proxy; disabled for Qwen. |
| **dashboard** | `http://localhost:8010` | Servedeck/Coldstart: start and stop the model, watch generation and prefill speed, see memory headroom. |

`llm start` brings up the configured model server and, for GLM, the effort
proxy. `llm ui` (and `llm codex`) also starts the dashboard.

## Configuration

`.config` is the only file you normally edit. Values **must** be quoted —
`KEY="value"` — because an unquoted line is skipped silently.

```
# Example Qwen configuration; use BACKEND="glm53" for the GLM backend.
BACKEND="flashnext"                 # which launcher to use
MODEL_REPO="mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4"
SERVED_NAME="qwen38-flash-next"
PORT="8001"                         # the Qwen server
COLDSTART_URL=""                    # direct client connection
```

For GLM, `COLDSTART_URL` can point clients at the effort proxy while `PORT`
continues to identify the real vLLM server. Qwen clients connect directly to
`PORT`.

`BACKEND` selects a launcher:

| backend | launcher |
|---|---|
| `glm53` | `~/Projects/vllm-glm53/serve-opt.sh` |
| `flashnext` | `~/Projects/vllm-qwen38next/serve-abliterated.sh` |

Model-specific tuning belongs in that launcher, not here. `llm` owns
orchestration only.

## GLM effort proxy

When `BACKEND="glm53"`, the GLM model **thinks before it answers, and the thinking comes out of the same
token budget as the answer.** Its chat template opens a `<think>` block before
the model writes a word.

So a client that asks for a small `max_tokens` gets this:

```
max_tokens=200  ->  finish_reason="length",  content: ""      (nothing at all)
```

An empty assistant turn is not harmless. It goes back into the conversation on
the next request, the template renders it as a blank turn, and the model reads
the thread as fresh and starts over. That is what "the agent forgot everything
after a few tool calls" looks like from the outside.

The proxy fixes this by **raising** a too-small budget (never lowering it, never
past what the context can hold), retrying an answerless turn with bounded
thinking, and — if even that comes back empty — saying so in the transcript
instead of emitting silence.

### Effort aliases

| model name | thinking | use it for |
|---|---|---|
| `glm53-flash-low` | terse | quick lookups |
| `glm53-flash-high` | bounded | **agents, tool loops, coding** |
| `glm53-flash` / `-max` | unbounded | one-shot hard questions, with a big budget |

Measured on one hard problem: `low` used 183 reasoning tokens, `high` 484, `max`
1,812. `max` took about twice the wall time of `high` and produced a *shorter*
answer. Agent loops want a decision, not an essay — use `-high`.

## Clients

**VS Code Copilot** — point directly at `:8001` for Qwen. For GLM, point at
`:8003` when the effort proxy is running. `maxInputTokens` must be the server's
context minus the output budget and a margin.

**Codex** — `./llm codex` sets `CODEX_HOME` to `.codex/` here, which wires the
provider to the configured server or GLM proxy. Two settings in `.codex/config.toml` are load-bearing:
`wire_api = "responses"` (this Codex build rejects `"chat"`), and the
`[models.*]` tables. Without the latter Codex warns that model metadata is
missing, guesses the context window, and compacts the conversation far too early
— the same "forgetting" symptom by a different route.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| agent repeats its first step, or offers "what would you like me to do?" | a blank GLM turn poisoned the history | use `-high`; make sure GLM clients go through `:8003` |
| `ERR_CONNECTION_REFUSED` in a client | server down or restarting | `llm status`, then `llm start` |
| `llm start` says a vllm is already running | a previous server is still alive or hung | `llm stop`, wait for RAM to come back, start again |
| server refuses to start, mentions KV cache | context asked for exceeds what KV can hold | the launcher derives KV from context; check `MAX_LEN` |
| first token takes minutes on a long prompt | that is prefill, and it is real work | `llm status` shows prefill throughput |

Logs: `logs/server.log`, `logs/proxy.log`. `llm logs -f` follows the server.

## Layout

```
llm                 the CLI — start/stop/status/logs/chat/codex/ui
.config             the only file you normally edit
.codex/             CODEX_HOME for `llm codex` (provider + model metadata)
logs/  run/         runtime output and pidfiles
bin/                helper scripts (watchdogs, recorders)
legacy/             superseded scripts, kept for reference
codex-qwen.sh       the previous launcher; superseded by ./llm
LOCAL_LLM_SETUP.md  long-form background: supervision, systemd, history
```

## Status

This is a personal workstation setup, not a product. It targets a specific vLLM
fork and specific checkpoints; numbers quoted here were measured on this machine
and carry the conditions they were measured under. The published write-up of the
model work lives at
[glm53-flash-single-gpu](https://github.com/pctablet505/glm53-flash-single-gpu).
