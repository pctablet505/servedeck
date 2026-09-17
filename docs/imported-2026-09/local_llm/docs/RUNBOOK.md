# Runbook — operating the local LLM

Day-to-day operation and incident triage. For building Flash-Next from source see
[vllm-qwen38next/SETUP.md](../../vllm-qwen38next/SETUP.md); for sizing decisions see
[CAPACITY.md](CAPACITY.md); for wiring editors see [CLIENTS.md](CLIENTS.md).

Current backend as of **2026-08-27**: `qwen38-flash-next` on **:8001**, started by
`vllm-qwen38next/serve.sh`, routed through `codex-qwen.sh` with `BACKEND="flashnext"`.

---

## The three commands you actually need

```bash
./codex-qwen.sh status      # is it up, which port, GPU, supervisor state
./codex-qwen.sh qwen        # start it (if needed) and launch Codex
./codex-qwen.sh stop        # release the GPU
```

`status` is safe to run at any time and never touches the server.

---

## Health checks — use the right endpoint

There are two endpoints and **they disagree on purpose**:

| endpoint | what it tells you |
|---|---|
| `GET /health` | **the real health gate.** Returns **503 on `EngineDeadError`** — the actual death mode. |
| `GET /v1/models` | served from app state. Stays **200 even with a dead engine**. |

```bash
curl -s -o /dev/null -w '%{http_code}\n' -m 5 http://localhost:8001/health
```

A 200 from `/v1/models` is **not** proof the model works. If you want one command that proves
end-to-end health, generate a token:

```bash
curl -s -m 30 http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-flash-next","messages":[{"role":"user","content":"say OK"}],"max_tokens":5}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"])'
```

---

## Live utilization

vLLM exposes Prometheus metrics at `/metrics`. The ones worth watching:

```bash
curl -s http://localhost:8001/metrics | grep -E '^vllm:(kv_cache_usage_perc|num_requests_running|num_requests_waiting|num_preemptions_total|request_prompt_tokens_(sum|count))'
```

| metric | meaning |
|---|---|
| `kv_cache_usage_perc` | 0–1. Sustained near 1.0 = at the edge. |
| `num_requests_running` / `_waiting` | current concurrency and backlog |
| `num_requests_waiting_by_reason{reason="capacity"}` | **>0 means you are KV-bound right now** |
| `num_preemptions_total` | **rising = over-subscribed.** vLLM is evicting and recomputing KV — real work thrown away. |
| `request_prompt_tokens_sum / _count` | average context per request — divide to size agent count |

Counters **reset on restart**. `num_preemptions_total` rising is the single most useful
oversubscription signal; it beats any capacity estimate.

---

## Incident triage

### 1. Is it the GPU or the host?

Answer this first — it splits the whole decision tree.

```bash
free -g                                          # host RAM
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
journalctl -k --no-pager --since "-2 hours" | grep -iE "oom-kill|killed process|NVRM: Xid"
```

**Host RAM OOM looks like an LLM crash but usually isn't.** On 2026-08-27 a 103 GiB Python
process was OOM-killed while the server kept serving normally — generation dropped from
93 to 2.8 tok/s from thrashing, then recovered. The log showed zero errors and
`request_success_total` kept climbing.

> **This machine has 182 GB RAM and roughly 49–59 GB is already held** by Flash-Next's PLE
> host-offload (the n-gram table lives in host RAM, not VRAM). Loading any second model —
> `vllm serve`, `from_pretrained`, a test harness — will OOM the box. Don't.

### 2. Did it crash, or did it never start?

This distinction decides whether restarting helps.

- **Crashed while serving** → a restart is likely to recover it.
- **Failed to boot** → restarting loops forever and hides the cause.

Historically **44 of 72 recorded exits never reached serving.** Boot failures dominate. So
always read the error before restarting.

```bash
grep -nE "ValueError|RuntimeError|NotImplementedError|cannot find -l|out of memory" \
  ~/Projects/vllm-qwen38next/serve.log | grep -v "raise " | tail -5
```

Six distinct boot failures are catalogued by symptom in
[SETUP.md §11](../../vllm-qwen38next/SETUP.md) — four of them report an error naming the
wrong subsystem, so match on the symptom, not your instinct.

### 3. GPU faults (Xid)

```bash
journalctl -k --no-pager | grep "NVRM: Xid" | tail -5
```

| Xid | meaning | restart? |
|---|---|---|
| 13, 31 | MMU fault | yes, a restart may recover |
| 79 | GPU fell off the bus | **no — reboot required** |
| 154 | node reboot required | **no** |
| other | unrecognised | **no — do not assume** |

If `nvidia-smi -L` fails at all, **never restart** — a reboot is the only fix. Six restarts
in 21 s once burned the systemd burst budget against a GPU that was off the bus.

### 4. Something is holding the GPU

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
ps -eo pid,comm --no-headers | awk '$2=="vllm" || $2 ~ /^VLLM::/'
```

Note `VLLM::EngineCore` is 16 characters and the kernel truncates `comm` at 15, so match on
the `VLLM::` prefix, never the full name.

---

## Process hygiene — the trap that bites here

**Never use `pgrep -f` or `pkill -f` on this box.** They match the calling shell's own command
line. This repeatedly produced false positives and once killed the invoking shell mid-command.

```bash
ps -eo pid,comm --no-headers | awk '$2=="vllm"'      # correct
ss -H -ltnp "sport = :8001"                          # who owns the port
```

---

## ptrace_scope — required at startup only

Flash-Next's PLE offload hands a GPU buffer to a sibling process over CUDA IPC, which needs
`pidfd_getfd()` → `PTRACE_MODE_ATTACH`. Ubuntu's default `ptrace_scope=1` only permits attach
to descendants.

`serve.sh` handles this: it relaxes the setting with `sudo`, then **re-hardens as soon as the
server answers**. Running the server therefore prompts for sudo, by design.

> **Consequence for automation:** `sudo -n` fails on this box ("interactive authentication is
> required") and a systemd `ExecStart` has no tty. **Flash-Next cannot be restarted
> unattended.** It currently appears to work only because `ptrace_scope` happens to be 0 —
> that does not survive a reboot.

Check and restore by hand:
```bash
cat /proc/sys/kernel/yama/ptrace_scope        # 1 = hardened (desired at rest)
sudo sysctl -w kernel.yama.ptrace_scope=1
```

---

## Restarting with different settings

`serve.sh` reads every knob from the environment:

```bash
MAX_LEN=65536 MAX_SEQS=4 ./serve.sh     # trade context for concurrency
GPU_UTIL=0.95 ./serve.sh                # default (lowest util that fits full 262k context)
```

Boot takes **~2 min warm, up to ~10 min cold** (torch.compile AOT artifacts and 97 FlashInfer
JIT objects cache under `~/.cache`). `shm_broadcast: No available shared memory broadcast
block found in 60 seconds` during startup is **informational, not an error**.

Config changes never apply to a running server — stop, change, start.

---

## Watchdog — deliberately disabled

`qwen-vllm-watchdog.timer` is **off**, and should stay off until it can distinguish intent.

A watchdog cannot tell *deliberately stopped* from *crashed*. `inactive` is exactly what a
clean stop leaves behind, so treating `inactive` as actionable resurrects a server you meant
to stop — this happened at 20:13:58 on 2026-08-27. It also called `systemctl reset-failed`,
which erases the `StartLimitBurst` backstop that protects against restart storms.

Any future supervisor must gate restarts on a persisted `desired_state`, not on observed state.
