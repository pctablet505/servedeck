"""Servedeck in-process pub/sub — SPEC.md §8's SSE hub.

One process, no external broker, no persistence beyond a short in-memory
replay buffer. Every subscriber (one per open `GET /api/events` connection)
gets its own bounded `asyncio.Queue(maxsize=512)`. A publish that finds a
subscriber's queue already full does not block the publisher and does not
raise: it drops that subscriber's OLDEST queued event to make room for the
new one and increments a per-subscriber `dropped` counter, so a slow
consumer loses history (oldest-first, as SSE ordering expects) rather than
stalling every other subscriber or the poller that is trying to publish.

A small ring buffer of recently-published events (independent of any one
subscriber's queue) backs `EventSource`'s automatic `Last-Event-ID`
reconnect: a client that reconnects with a `Last-Event-ID` header gets any
events newer than that id replayed from the ring buffer before live
delivery resumes, rather than silently missing whatever happened during the
gap. The ring buffer has its own fixed capacity and is best-effort — a
client that was gone longer than the buffer covers still resumes live, it
just cannot fully backfill the gap, which is the same limitation any
memory-bounded broker has.

No FastAPI/Starlette import here: this module is pure asyncio and is usable
(and unit-testable) without an ASGI server running.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass, field
from typing import Any

#: Per SPEC.md §8's explicit instruction: "per-subscriber bounded
#: asyncio.Queue(maxsize=512), drop-oldest on overflow with a dropped
#: counter."
QUEUE_MAXSIZE = 512

#: How many recently-published events the hub keeps around, independent of
#: subscriber queues, purely to answer a reconnecting client's
#: `Last-Event-ID`. Sized generously above QUEUE_MAXSIZE: a client that was
#: merely queue-starved (still connected, just slow) is already covered by
#: its own queue; this buffer exists for the *reconnect* case, where the
#: gap is measured in seconds of real disconnect time, not queue depth.
_REPLAY_BUFFER_SIZE = 2048

#: The event `type` values SPEC.md §8 names. Not enforced by the type
#: system (a dataclass field typed as `str` would reject nothing useful
#: here), just documented: state | phase | telemetry | log | gateway |
#: notice. `publish()` accepts any string — a caller inventing a new event
#: type is not this module's business to police.
EVENT_TYPES = ("state", "phase", "telemetry", "log", "gateway", "notice")


@dataclass(frozen=True)
class Event:
    """One published event. `id` is a process-wide monotonically increasing
    sequence number (1-based), used both as the SSE frame's `id:` field and
    as the cursor for Last-Event-ID replay."""

    id: int
    type: str
    data: Any
    ts: float = field(default_factory=time.time)


class Subscriber:
    """One open `/api/events` connection's private inbox."""

    __slots__ = ("id", "queue", "dropped")

    def __init__(self, sub_id: int) -> None:
        self.id = sub_id
        self.queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        self.dropped: int = 0

    def _offer(self, event: Event) -> None:
        """Non-blocking enqueue with drop-oldest-on-overflow.

        Never awaits and never raises out of this method: a full inbox is
        an expected steady-state condition for a stalled consumer, not an
        error the publisher (a metrics poller, a log tailer, a state
        machine transition) should ever have to handle specially.
        """
        try:
            self.queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass
        # Drop the oldest queued item to make room, then retry once. A
        # bounded retry loop (rather than a single attempt) tolerates the
        # pathological case of another coroutine draining the queue to
        # empty between our get_nowait() and put_nowait() — vanishingly
        # unlikely for a single-consumer-per-subscriber queue, but a loop
        # costs nothing and removes the need to reason about it further.
        for _ in range(4):
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:
                break
            try:
                self.queue.put_nowait(event)
                return
            except asyncio.QueueFull:
                continue
        # Every retry raced a full queue again (or emptied it without
        # succeeding, e.g. maxsize==0): count the drop and move on rather
        # than spin or block the publisher.
        self.dropped += 1


class EventHub:
    """Process-wide pub/sub. Create one instance and share it across every
    request handler and background poller in the process."""

    def __init__(
        self,
        *,
        queue_maxsize: int = QUEUE_MAXSIZE,
        replay_buffer_size: int = _REPLAY_BUFFER_SIZE,
    ) -> None:
        self._queue_maxsize = queue_maxsize
        self._replay_buffer_size = replay_buffer_size
        self._subscribers: dict[int, Subscriber] = {}
        self._next_sub_id = itertools.count(1)
        self._next_event_id = itertools.count(1)
        self._replay: list[Event] = []

    # -- publish -----------------------------------------------------------

    def publish(self, event_type: str, data: Any) -> Event:
        """Fan `data` out to every currently-subscribed queue and record it
        in the replay ring buffer. Synchronous and non-blocking — safe to
        call from any coroutine (or, since nothing here awaits, effectively
        synchronous code running on the event loop) without a lock: dict
        iteration below is over a snapshot list, so a concurrent
        subscribe()/unsubscribe() racing this call can only add or remove a
        subscriber that either does or doesn't see this particular event,
        never corrupt the fan-out.
        """
        event = Event(id=next(self._next_event_id), type=event_type, data=data)
        for sub in list(self._subscribers.values()):
            sub._offer(event)
        self._replay.append(event)
        if len(self._replay) > self._replay_buffer_size:
            del self._replay[: len(self._replay) - self._replay_buffer_size]
        return event

    # -- subscribe -----------------------------------------------------------

    def subscribe(self, *, last_event_id: int | None = None) -> tuple[Subscriber, list[Event]]:
        """Register a new subscriber and return it along with any
        replayable events newer than `last_event_id` (SSE's automatic
        `Last-Event-ID` reconnect header — EventSource resends whatever id
        it last saw on every reconnect attempt).

        Replay is a plain list handed back to the caller to emit BEFORE
        live queue delivery starts; it never touches the new subscriber's
        own queue, so there is no ordering race between "replayed backlog"
        and "the next live publish" — the caller controls that ordering
        explicitly by emitting the returned list first.
        """
        sub = Subscriber(next(self._next_sub_id))
        sub.queue = asyncio.Queue(maxsize=self._queue_maxsize)
        self._subscribers[sub.id] = sub
        backlog: list[Event] = []
        if last_event_id is not None:
            backlog = [e for e in self._replay if e.id > last_event_id]
        return sub, backlog

    def unsubscribe(self, sub: Subscriber) -> None:
        self._subscribers.pop(sub.id, None)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscriber_dropped_counts(self) -> dict[int, int]:
        """Diagnostic snapshot: {subscriber_id: dropped_count} for every
        currently-connected subscriber, e.g. for a /api/state debug field."""
        return {sid: sub.dropped for sid, sub in self._subscribers.items()}


#: Shared, process-wide instance. api.py imports this directly rather than
#: constructing its own EventHub, so every poller/background task and every
#: request handler in the process publishes to and subscribes from the same
#: hub. A dedicated EventHub() may still be constructed directly (e.g. in a
#: unit test) without touching this singleton.
hub = EventHub()
