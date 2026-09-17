#!/usr/bin/env bash
# soak-27b-overnight.sh -- long, unattended stability soak for the 27B on the
# v0.29.0 venv (.venv-llm-029), the candidate fix for the Mamba state-copy
# race (vllm-project/vllm#50729; see
# ~/.claude/projects/-home-pctablet505-Projects/memory/qwen27b-xid-root-cause.md).
#
# WHY THIS EXISTS (2026-09-09): the interactive validation soak was cut at
# ~42s of clean traffic by an owner change of plan (Flash-Next needed the
# card back). 42s is not evidence of anything -- historical Xid incidents
# took 30 min to 30 h to surface. This script reproduces the same 7-agent
# growing-multi-turn-context load and runs it unattended for hours instead.
#
# Usage:
#   setsid nohup ~/Projects/local_llm/bin/soak-27b-overnight.sh [duration_s] \
#     > /dev/null 2>&1 &
#   (default duration: 28800s = 8h. setsid detaches it from this shell's
#   session so it survives the terminal/session closing; nohup covers SIGHUP.
#   The script does not touch stdin/stdout/cwd after startup, so it is safe
#   to background this way.)
#
#   SOAK_WRITE_CONFIG_ONLY=1 ~/Projects/local_llm/bin/soak-27b-overnight.sh
#   writes the frozen 27B config snapshot (see CONFIG ISOLATION below) and
#   exits 0 immediately -- no GPU/server/network activity. Used by
#   tests/test_soak_config_isolation.sh to exercise write_soak_config()
#   without running a soak.
#
# CONFIG ISOLATION (2026-09-09): this script does NOT source the live
# .config -- .config is shared with every other backend on this box and can
# be switched to a different model/port/util at any time by an unrelated
# workflow (e.g. to Flash-Next: BACKEND="flashnext", PORT="8001",
# GPU_MEM_UTIL="0.95", EXTRA_ARGS="--language-model-only ..."). Sourcing it
# here would risk booting the wrong model on the wrong port -- or, worse,
# leaking GPU_MEM_UTIL="0.95" onto the 27B, which is exactly the utilization
# that produced the historical Xid 13/31 crashes (clean only at 0.47). This
# script instead OWNS a private, frozen snapshot -- $HERE/.config.27b-soak,
# rewritten from scratch by write_soak_config() at the top of every run --
# and passes it to bin/qwen-server-run.sh via CONFIG_FILE=..., which that
# script's launcher supports as an override of its own default ($HERE/.config)
# for exactly this reason. MODEL/PORT/GPU_MEM_UTIL/EXTRA_ARGS below are all
# read from that snapshot, once, so there is exactly one place in this file
# that names the 27B checkpoint literally: inside write_soak_config().
#
# What it does, in order:
#   1. Confirms the GPU is free (or that the target port is already OUR
#      27B/venv-029 process -- see idempotency below), then boots the 27B
#      from .venv-llm-029 with EXACTLY the flags bin/qwen-server-run.sh uses
#      (it just calls that script with VLLM_VENV and CONFIG_FILE overridden).
#   2. Runs 7 concurrent worker "agents", each holding a growing multi-turn
#      conversation (periodically reset once it gets large, like a new agent
#      session starting), against the server for $1 seconds.
#   3. Every 60s, counts `NVRM: Xid` lines in `journalctl -k -b` and logs the
#      delta since this soak started, plus running request/error counts.
#   4. On exit (normal, error, or signal) stops the server it booted.
#   5. Writes a final summary line: elapsed, requests, errors, tok/s, Xid
#      count (new Xid lines seen since this soak's baseline).
#
# Output:
#   logs/soak-27b-<timestamp>.log            -- human narrative + summary
#   logs/soak-27b-<timestamp>.requests.jsonl -- one JSON line per request
#
# Idempotent about a stale server on the port: if something is already
# listening there, this script reuses it IF its /proc/<pid>/cmdline
# identifies it as our own 27B process from .venv-llm-029 and it is healthy;
# kills-and-reboots it if it matches but is unhealthy (a stale leftover from
# a previous run of this script); and ABORTS without touching anything if
# the port is held by a process that does not match -- this box runs other
# people's/other scripts' servers too (ats-optimizer on 8000, GLM, Flash-Next)
# and this script must never guess at killing something it doesn't own.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$HERE/logs"
# SOAK_CONFIG: the frozen snapshot this script owns (see CONFIG ISOLATION
# above). Never the live .config -- write_soak_config() below is the only
# writer, and it overwrites this file at the start of every run.
SOAK_CONFIG="$HERE/.config.27b-soak"
VLLM_VENV="$HERE/.venv-llm-029"

mkdir -p "$LOG_DIR"

# write_soak_config PATH: (re)writes the frozen 27B snapshot at PATH,
# unconditionally overwriting any prior contents so a stale snapshot (from
# an earlier run, or a manual edit) can never leak into a soak. util 0.47 is
# the operating point that has run Xid-free for 9+ real days; 0.62 and 0.92
# are the two util settings that Xid'd during the Aug 2026 bisection -- see
# qwen-server-run.sh's EXTRA_ARGS comment for the full history. This is the
# ONLY place in this file that names the 27B checkpoint/port/util literally
# -- MODEL/PORT/GPU_MEM_UTIL/EXTRA_ARGS below are all derived from sourcing
# the file this writes, not from a second hardcoded copy.
write_soak_config() {
    cat > "$1" <<'EOF'
BACKEND="inline"
MODEL_REPO="RadixArk/Qwen3.8-27B-NVFP4"
PORT="8004"
GPU_MEM_UTIL="0.47"
MAX_NUM_SEQS="128"
MAX_MODEL_LEN="262144"
EXTRA_ARGS=""
EOF
}

write_soak_config "$SOAK_CONFIG"

# SOAK_WRITE_CONFIG_ONLY=1: write the snapshot above and exit 0 immediately,
# before touching the GPU, booting a server, or opening a network
# connection. See "Usage" in the header comment.
if [ "${SOAK_WRITE_CONFIG_ONLY:-0}" = 1 ]; then
    printf '%s\n' "$SOAK_CONFIG"
    exit 0
fi

PORT=8004
GPU_MEM_UTIL="0.47"
EXTRA_ARGS=""
# shellcheck disable=SC1090
[ -f "$SOAK_CONFIG" ] && . "$SOAK_CONFIG"
PORT="${PORT:-8004}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.47}"
# MODEL: derived from the snapshot's MODEL_REPO (see write_soak_config), not
# a second hardcoded literal -- one source of truth for the checkpoint name,
# used below both to boot the server and, in is_ours(), to recognise it.
MODEL="$MODEL_REPO"

DURATION_S="${1:-28800}"
STAMP="$(date +%Y%m%d-%H%M%S)"
SOAK_LOG="$LOG_DIR/soak-27b-$STAMP.log"
REQUESTS_LOG="$LOG_DIR/soak-27b-$STAMP.requests.jsonl"
: > "$REQUESTS_LOG"

log() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$SOAK_LOG" >&2 ; }

log "=== soak-27b-overnight.sh starting: duration=${DURATION_S}s port=$PORT venv=$VLLM_VENV model=$MODEL ==="
log "resolved config (from $SOAK_CONFIG): model=$MODEL port=$PORT util=$GPU_MEM_UTIL extra_args=[${EXTRA_ARGS:-}]"
log "requests log: $REQUESTS_LOG"

cmdline_of() { [ -r "/proc/$1/cmdline" ] && tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null; }

is_ours() {
    # $1 = pid. True if it's a vllm-serve process for our model from our venv.
    case "$(cmdline_of "$1")" in
        *"$VLLM_VENV/bin/vllm"*"serve"*"$MODEL"*) return 0 ;;
        *) return 1 ;;
    esac
}

find_port_pid() {
    ss -lntp 2>/dev/null | awk -v p=":$PORT" '$4 ~ (p"$") {print $0}' | \
        grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2
}

WE_BOOTED=0
SERVER_PID=""
CLEANED_UP=0

existing_pid="$(find_port_pid || true)"
if [ -n "${existing_pid:-}" ]; then
    if is_ours "$existing_pid"; then
        if curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            log "port $PORT already serving our model (pid $existing_pid) -- reusing, not re-booting"
            SERVER_PID="$existing_pid"
            WE_BOOTED=1   # we take ownership for stop-at-end purposes
        else
            log "port $PORT held by our own stale/unhealthy process (pid $existing_pid) -- stopping it before rebooting"
            kill -TERM "$existing_pid" 2>/dev/null || true
            for _ in $(seq 1 20); do
                kill -0 "$existing_pid" 2>/dev/null || break
                sleep 3
            done
            existing_pid=""
        fi
    else
        log "FATAL: port $PORT is held by pid $existing_pid, which is NOT this script's 27B/.venv-llm-029 process (cmdline: $(cmdline_of "$existing_pid")). Refusing to touch it -- aborting."
        exit 1
    fi
fi

cleanup() {
    [ "$CLEANED_UP" -eq 1 ] && return
    CLEANED_UP=1
    if [ "$WE_BOOTED" -eq 1 ] && [ -n "${SERVER_PID:-}" ]; then
        log "=== stopping server pid $SERVER_PID ==="
        # SERVER_PID is the launcher's pid, which `exec`s into the real vllm
        # process (same pid throughout -- see qwen-server-run.sh), so this IS
        # the APIServer pid; SIGTERM lets it shut down EngineCore cleanly.
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 3
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            log "server still alive after 60s -- sending SIGKILL"
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
    fi
}
trap cleanup EXIT
trap 'log "caught SIGTERM"; exit 143' TERM
trap 'log "caught SIGINT"; exit 130' INT

if [ -z "${SERVER_PID:-}" ]; then
    busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)"
    if [ -n "$busy" ]; then
        log "FATAL: nvidia-smi shows compute processes already running and no reusable server was found on port $PORT:"
        log "$busy"
        log "Refusing to boot a second server on the GPU. Aborting."
        exit 1
    fi
    log "GPU confirmed idle. Booting the 27B from $VLLM_VENV on port $PORT via qwen-server-run.sh"
    VLLM_VENV="$VLLM_VENV" CONFIG_FILE="$SOAK_CONFIG" nohup "$HERE/bin/qwen-server-run.sh" >> "$SOAK_LOG" 2>&1 &
    SERVER_PID=$!
    WE_BOOTED=1
    log "launcher/server pid $SERVER_PID"

    boot_elapsed=0
    up=0
    while [ "$boot_elapsed" -lt 900 ]; do
        if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            up=1
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "FATAL: server pid $SERVER_PID died during boot after ${boot_elapsed}s -- see $SOAK_LOG and logs/qwen_server.log"
            exit 1
        fi
        sleep 15
        boot_elapsed=$((boot_elapsed + 15))
        log "waiting for boot... ${boot_elapsed}s"
    done
    if [ "$up" -ne 1 ]; then
        log "FATAL: server did not answer /health within 900s"
        exit 1
    fi
    log "server up after ${boot_elapsed}s"
fi

# Verify via /proc/<pid>/cmdline, same as the interactive validation did.
log "cmdline: $(cmdline_of "$SERVER_PID")"

# ----------------------------------------------------- load + Xid monitor ---
# Embedded (not a separate file) so this script is the single self-contained
# deliverable. 7 workers, each a growing multi-turn conversation (reset once
# large, like a fresh agent session), for $DURATION_S seconds. A monitor
# thread samples `journalctl -k -b` for NVRM Xid lines every 60s.
BASE_URL="http://127.0.0.1:$PORT/v1/chat/completions" \
HEALTH_URL="http://127.0.0.1:$PORT/health" \
MODEL_NAME="$MODEL" \
DURATION_S="$DURATION_S" \
SOAK_LOG="$SOAK_LOG" \
REQUESTS_LOG="$REQUESTS_LOG" \
python3 - <<'PYEOF'
import json, os, random, subprocess, threading, time
import urllib.request, urllib.error

BASE_URL = os.environ["BASE_URL"]
HEALTH_URL = os.environ["HEALTH_URL"]
MODEL = os.environ["MODEL_NAME"]
DURATION_S = int(os.environ["DURATION_S"])
SOAK_LOG = os.environ["SOAK_LOG"]
REQUESTS_LOG = os.environ["REQUESTS_LOG"]

NUM_WORKERS = 7
# MAX_TOKENS (2026-09-09, owner addendum): was 300. Thinking stays enabled
# (no enable_thinking=false / chat_template_kwargs anywhere in this script),
# and at 300 tokens the reasoning budget consumed most or all of the
# completion before any visible answer, so responses were mostly
# finish_reason=length with empty content -- the "growing context" this soak
# exists to exercise barely grew. 2048 gives the model room to reason AND
# answer, so a realistic fraction of turns complete instead of truncating.
MAX_TOKENS = 2048
TEMPERATURE = 0.4
CONTEXT_RESET_CHARS = 550_000
REQUEST_TIMEOUT = 90

SEED_PROMPTS = [
    "You are helping refactor a Python data pipeline. Start by describing, step by step, how you would restructure a script that reads CSV files, cleans null values, and writes to Parquet.",
    "You are debugging a flaky pytest suite. Walk through a systematic approach to find why a test passes locally but fails in CI, one step at a time.",
    "You are reviewing a pull request that adds a new REST endpoint. List the things you'd check, then start checking the first one in detail.",
    "You are writing documentation for a CLI tool. Draft the first section (installation) and explain your reasoning for the structure.",
    "You are planning a database schema migration. Describe the first migration step and what could go wrong.",
    "You are optimizing a slow SQL query. Explain your first diagnostic step and what you'd look for.",
    "You are setting up CI/CD for a new microservice. Describe the first pipeline stage you'd configure and why.",
]
FOLLOWUPS = [
    "Continue to the next step in more detail.",
    "What could go wrong with that approach? Elaborate.",
    "Now write example code for what you just described.",
    "Summarize what we've covered so far, then continue.",
    "A colleague disagrees with that approach -- respond to their concern and continue.",
    "Add error handling considerations to the last step.",
    "Now consider the edge cases and continue the plan.",
]

lock = threading.Lock()
stop_event = threading.Event()
counters = {"requests": 0, "errors": 0, "resets": 0}


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(SOAK_LOG, "a") as f:
        f.write(line + "\n")


def post_chat(messages):
    payload = json.dumps({"model": MODEL, "messages": messages,
                           "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE}).encode()
    req = urllib.request.Request(BASE_URL, data=payload,
                                  headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            body = resp.read(); status = resp.status
        return status, time.time() - t0, json.loads(body), None
    except urllib.error.HTTPError as e:
        return e.code, time.time() - t0, None, e.read().decode(errors="replace")[:500]
    except Exception as e:
        return None, time.time() - t0, None, f"{type(e).__name__}: {e}"


def worker(worker_id):
    messages = [{"role": "user", "content": SEED_PROMPTS[worker_id % len(SEED_PROMPTS)]}]
    turn = 0
    while not stop_event.is_set():
        turn += 1
        status, latency, data, err = post_chat(messages)
        rec = {"ts": time.time(), "worker": worker_id, "turn": turn, "status": status,
               "latency_s": round(latency, 3),
               "context_chars": sum(len(m["content"]) for m in messages)}
        if data is not None:
            try:
                choice = data["choices"][0]
                rec["finish_reason"] = choice.get("finish_reason")
                message = choice["message"]
                content = message.get("content") or ""
                # reasoning_content (2026-09-09, owner addendum): the qwen3
                # reasoning parser returns the thinking trace separately from
                # content. Re-sending it on the assistant turn we append to
                # the growing history -- not just content -- is what the
                # model's chat template expects for a multi-turn thinking
                # conversation; dropping it here would silently reconstruct
                # a non-thinking-shaped history every turn.
                reasoning_content = message.get("reasoning_content") or ""
                usage = data.get("usage", {})
                rec["prompt_tokens"] = usage.get("prompt_tokens")
                rec["completion_tokens"] = usage.get("completion_tokens")
                assistant_turn = {"role": "assistant",
                                   "content": content or "(empty content -- reasoning-budget truncation)"}
                if reasoning_content:
                    assistant_turn["reasoning_content"] = reasoning_content
                messages.append(assistant_turn)
                messages.append({"role": "user", "content": random.choice(FOLLOWUPS)})
            except Exception as e:
                rec["parse_error"] = str(e)
                with lock: counters["errors"] += 1
        else:
            rec["error"] = err
            with lock: counters["errors"] += 1

        with lock: counters["requests"] += 1
        with open(REQUESTS_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")

        if sum(len(m["content"]) for m in messages) > CONTEXT_RESET_CHARS:
            messages = [{"role": "user", "content": SEED_PROMPTS[(worker_id + turn) % len(SEED_PROMPTS)]}]
            with lock: counters["resets"] += 1

        time.sleep(random.uniform(0.5, 2.0))


def xid_count():
    try:
        out = subprocess.run(["journalctl", "-k", "-b", "--no-pager"],
                              capture_output=True, text=True, timeout=30)
        return out.stdout.count("Xid")
    except Exception as e:
        return None


def health_ok():
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def monitor(start_time, baseline):
    while not stop_event.is_set():
        stop_event.wait(60)
        if stop_event.is_set():
            break
        xc = xid_count()
        delta = (xc - baseline) if (xc is not None and baseline is not None) else "N/A"
        with lock:
            snap = dict(counters)
        elapsed = int(time.time() - start_time)
        log(f"elapsed={elapsed}s health={'OK' if health_ok() else 'DOWN'} "
            f"requests={snap['requests']} errors={snap['errors']} resets={snap['resets']} "
            f"xid_cumulative={xc} xid_new_since_start={delta}")


def main():
    baseline = xid_count()
    log(f"soak load starting: workers={NUM_WORKERS} duration_s={DURATION_S} xid_baseline={baseline}")
    start_time = time.time()
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(NUM_WORKERS)]
    mon = threading.Thread(target=monitor, args=(start_time, baseline), daemon=True)
    for t in threads: t.start()
    mon.start()

    time.sleep(DURATION_S)
    stop_event.set()
    for t in threads: t.join(timeout=15)
    mon.join(timeout=5)

    elapsed = time.time() - start_time
    with lock:
        final = dict(counters)

    total_completion_tokens = 0
    finish_reasons = {}
    status_counts = {}
    with open(REQUESTS_LOG) as f:
        for line in f:
            try: r = json.loads(line)
            except Exception: continue
            total_completion_tokens += r.get("completion_tokens") or 0
            finish_reasons[r.get("finish_reason", "N/A")] = finish_reasons.get(r.get("finish_reason", "N/A"), 0) + 1
            status_counts[str(r.get("status"))] = status_counts.get(str(r.get("status")), 0) + 1

    final_xid = xid_count()
    xid_new = (final_xid - baseline) if (final_xid is not None and baseline is not None) else "N/A"
    tok_s = total_completion_tokens / elapsed if elapsed > 0 else 0.0

    summary = (f"SOAK SUMMARY elapsed_s={elapsed:.1f} requests={final['requests']} "
               f"errors={final['errors']} resets={final['resets']} "
               f"status_counts={status_counts} finish_reason_counts={finish_reasons} "
               f"total_completion_tokens={total_completion_tokens} tok_s={tok_s:.2f} "
               f"xid_baseline={baseline} xid_final={final_xid} xid_new_this_soak={xid_new} "
               f"capacity_md_single_stream_reference_tok_s=140.4")
    log(summary)


if __name__ == "__main__":
    main()
PYEOF
py_rc=$?
log "load generator exited rc=$py_rc"
log "=== soak-27b-overnight.sh finished ==="
exit "$py_rc"
