"""In-process pub/sub between the graph worker thread and SSE subscribers.

One channel per test. Swap this module for Redis pub/sub when the API runs
on more than one worker process — the interface is deliberately tiny.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

_subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)
_loop: asyncio.AbstractEventLoop | None = None


def bind_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Remember the serving loop so worker threads can publish into it."""
    global _loop
    _loop = loop


def subscribe(channel: str) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    _subscribers[channel].add(queue)
    return queue


def unsubscribe(channel: str, queue: asyncio.Queue) -> None:
    _subscribers[channel].discard(queue)
    if not _subscribers[channel]:
        _subscribers.pop(channel, None)


def publish(channel: str, event: str, data: dict[str, Any]) -> None:
    """Thread-safe: called from the graph worker thread."""
    payload = {"event": event, "data": data}
    if _loop is None:
        return
    _loop.call_soon_threadsafe(_fanout, channel, payload)


def _fanout(channel: str, payload: dict) -> None:
    for queue in list(_subscribers.get(channel, ())):
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            # A stalled client must not block the run; it will resync via REST.
            pass
