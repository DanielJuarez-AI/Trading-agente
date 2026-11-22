"""
Core Events - Bus de Eventos SSE en Tiempo Real para el Pipeline Multi-Agente.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Optional, Set


class PipelineEventBus:
    """Publicador/Suscriptor thread-safe para Server-Sent Events (SSE)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscribers: Set[asyncio.Queue[Dict[str, Any]]] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: Optional[asyncio.AbstractEventLoop]) -> None:
        with self._lock:
            self._loop = loop

    def subscribe(self, maxsize: int = 128) -> asyncio.Queue[Dict[str, Any]]:
        q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, queue: asyncio.Queue[Dict[str, Any]]) -> None:
        with self._lock:
            self._subscribers.discard(queue)

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    @staticmethod
    def _put_nowait_drop_oldest(queue: asyncio.Queue[Dict[str, Any]], event: Dict[str, Any]) -> None:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        event = {
            "event": event_type,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "data": payload,
        }
        with self._lock:
            loop = self._loop
            subs = list(self._subscribers)

        if not subs:
            return

        if loop is not None and not loop.is_closed():
            for q in subs:
                try:
                    loop.call_soon_threadsafe(self._put_nowait_drop_oldest, q, event)
                except RuntimeError:
                    pass
        else:
            for q in subs:
                self._put_nowait_drop_oldest(q, event)

    async def stream(self, heartbeat_seconds: float = 15.0) -> AsyncIterator[Dict[str, Any]]:
        q = self.subscribe()
        try:
            yield {
                "event": "connected",
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "data": {"message": "SSE pipeline stream connected", "subscribers": self.subscriber_count()},
            }
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=heartbeat_seconds)
                    yield item
                except asyncio.TimeoutError:
                    yield {
                        "event": "heartbeat",
                        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "data": {"status": "alive"},
                    }
        finally:
            self.unsubscribe(q)
