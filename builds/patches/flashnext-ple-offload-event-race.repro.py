#!/usr/bin/env python3
"""Reproduce the PLE offload freeze on the real GPU without loading a model (lead, 2026-10-10).

    PYTHONPATH=<a vllm source tree> <the build's python> builds/patches/flashnext-ple-offload-event-race.repro.py

Drives the tree's own PleOffloadConnector and CpuGpuSemaphore in the order that froze Flash-Next twice on 2026-10-10
(stacks: nse-knowledge-2026-10-08/reread/logs/freeze-160359):
  1. the model thread launches batch N;
  2. forward N parks on the model stream until the CPU worker answers request N (cuStreamWaitValue32);
  3. the request thread has taken request N off the queue but has not run yet (the delay CPU contention causes);
  4. the model thread launches batch N+1 (async scheduling keeps two batches in flight);
  5. the request thread now stages and publishes request N.
Step 5 must finish while forward N is still parked: the CPU worker cannot answer a request it never received.
Prints one verdict line. Exit 0 = request N was published; exit 1 = DEADLOCK (the request thread waited for forward N);
exit 2 = the scenario could not run. Works on the tree as it froze (one shared event) and on the fixed tree (one event
per request). Uses a few hundred MiB of GPU memory for a few seconds.
"""
import os
import queue
import sys
import threading
from multiprocessing.reduction import ForkingPickler
from unittest.mock import Mock

import torch
import torch.multiprocessing as torch_mp

import vllm
from vllm.model_executor.layers import ple_offload_layer
from vllm.model_executor.layers.ple_offload_layer import CpuGpuSemaphore
from vllm.v1.ple_offload import connector as connector_module
from vllm.v1.ple_offload.connector import PleOffloadConnector

WAIT_S = float(os.environ.get("REPRO_WAIT_S", "10"))
# control: REPRO_SKIP_NEXT_LAUNCH=1 leaves out step 4. Request N must then be published on every tree, which shows the
# early launch of batch N+1 is what the request thread ends up waiting for, not the parked forward by itself.
SKIP_NEXT_LAUNCH = os.environ.get("REPRO_SKIP_NEXT_LAUNCH") == "1"


def main() -> int:
    per_request_events = hasattr(PleOffloadConnector, "_enqueue_cuda_inputs")
    print("tree: %s" % os.path.dirname(os.path.dirname(vllm.__file__)))
    print("step 4 (batch N+1 launched before request N is published): %s" % ("LEFT OUT (control)" if SKIP_NEXT_LAUNCH else "done"))
    print("connector: %s (%s)" % (connector_module.__file__, "one event per request" if per_request_events else "one shared event"))
    if not torch.cuda.is_available():
        print("VERDICT could not run: no GPU")
        return 2
    socket = Mock()
    c = PleOffloadConnector.__new__(PleOffloadConnector)
    c.device = torch.device("cuda:0")
    c.tp_rank = 0
    c.dp_rank = 0
    strategy = torch_mp.get_sharing_strategy()
    try:
        torch_mp.set_sharing_strategy("file_descriptor")
        c._input_ids_buf = torch.full((4,), -99, dtype=torch.int32).share_memory_()
        c._query_start_loc_buf = torch.full((3,), -99, dtype=torch.int32).share_memory_()
        c._ngram_context_buf = None
        torch_mp.set_sharing_strategy("file_system")
        ForkingPickler.dumps((c._input_ids_buf, c._query_start_loc_buf))
    finally:
        torch_mp.set_sharing_strategy(strategy)
    c._pinned_input_buffers = []
    c._request_thread = None          # this script plays the request thread itself, in the order that froze
    published = threading.Event()
    with torch.accelerator.device_index(c.device.index):
        c._input_ids_source = torch.tensor([7, -1, 11, -1], dtype=torch.int32, device=c.device)
        c._query_start_loc_source = torch.tensor([0, 2, 4], dtype=torch.int32, device=c.device)
        c._ngram_context_source = None
        c._uses_cuda_inputs = True
        c._validate_input_sources()
        c._pin_input_buffers()
        if per_request_events:
            c._request_queue = queue.Queue(maxsize=2)
            c._d2h_event_pool = queue.Queue(maxsize=2)
            c._d2h_event_pool.put_nowait(torch.cuda.Event())
            c._d2h_event_pool.put_nowait(torch.cuda.Event())
        else:
            c._request_queue = queue.Queue(maxsize=1)          # as the tree's own __init__ sets them
            c._d2h_stream = torch.cuda.Stream(device=c.device)
            c._input_ready_event = torch.cuda.Event()
            c._d2h_done_event = torch.cuda.Event()
        semaphore = CpuGpuSemaphore(c.device)
        placeholder = torch.zeros(1, device=c.device)
        model_stream = torch.cuda.Stream(device=c.device)
        worker_stream = torch.cuda.Stream(device=c.device)
        torch.accelerator.synchronize(c.device)
        with torch.cuda.stream(model_stream):
            c._launch(num_reqs=2, num_tokens=4)                                                   # 1
            ple_offload_layer._ple_offload_wait_impl(semaphore.flag_tensor, placeholder, placeholder)   # 2
            first = c._request_queue.get_nowait()                                                 # 3
            if not SKIP_NEXT_LAUNCH:
                c._launch(num_reqs=1, num_tokens=2)                                               # 4

        def publish():
            c._process_request(first, socket)                                                     # 5
            published.set()

        t = threading.Thread(target=publish, daemon=True)
        t.start()
        ok = published.wait(timeout=WAIT_S)
        staged = c._input_ids_buf.tolist()
        sent = socket.send.call_count
        # the CPU worker's answer to request N; in the frozen server it never comes, here it lets the process end
        semaphore.signal(worker_stream)
        t.join(timeout=30)
        model_stream.synchronize()
        torch.accelerator.synchronize(c.device)
        c._unpin_input_buffers()
    if ok and sent == 1 and staged == [7, -1, 11, -1]:
        print("VERDICT ok: request N was staged and published while forward N was parked")
        return 0
    if not ok:
        print("VERDICT DEADLOCK: after %.0f s the request thread was still waiting; requests published %d, staged input ids %s. "
              "It was released only by the answer to the request it had not sent." % (WAIT_S, sent, staged))
        return 1
    print("VERDICT wrong result: published %d, staged %s" % (sent, staged))
    return 2


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
