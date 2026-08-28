"""Dual-tier priority request queue for Cinch FastAPI Gateway."""

from __future__ import annotations

import asyncio
import enum
import time
import uuid
from typing import Any, Dict


class RequestPriority(enum.IntEnum):
    """Priority level where lower integer denotes higher dispatch priority."""

    HIGH = 0  # Real-time / Interactive / VIP
    LOW = 1  # Batch / Background


class PriorityRequestQueue:
    """Async priority queue managing active concurrency and request preemption."""

    def __init__(self, max_active: int = 8, max_queue: int = 64) -> None:
        self.max_active = max_active
        self.max_queue = max_queue
        self._active_count = 0
        self._queue: asyncio.PriorityQueue[tuple[int, float, str, asyncio.Future[None]]] = asyncio.PriorityQueue()
        self._lock = asyncio.Lock()
        self._high_priority_queued = 0
        self._low_priority_queued = 0

    @property
    def queue_depth(self) -> int:
        """Current number of active waiting requests in queue."""
        return self._high_priority_queued + self._low_priority_queued

    @property
    def active_requests(self) -> int:
        """Current number of active requests being processed upstream."""
        return self._active_count

    async def acquire(
        self,
        priority: RequestPriority = RequestPriority.HIGH,
        timeout: float = 30.0,
    ) -> str:
        """Acquire a processing slot or wait in priority queue.

        Raises:
            asyncio.TimeoutError: If queued request exceeds timeout.
            RuntimeError: If queue is at maximum capacity.
        """
        request_id = str(uuid.uuid4())[:8]
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        async with self._lock:
            # If under active concurrency limit and queue is empty, acquire immediately
            if self._active_count < self.max_active and self.queue_depth == 0:
                self._active_count += 1
                return request_id

            if self.queue_depth >= self.max_queue:
                raise RuntimeError(f"Gateway priority queue full ({self.queue_depth}/{self.max_queue}). Load shed.")

            # Enqueue into priority queue
            if priority == RequestPriority.HIGH:
                self._high_priority_queued += 1
            else:
                self._low_priority_queued += 1

            entry = (int(priority), time.time(), request_id, fut)
            self._queue.put_nowait(entry)

        # Wait for dispatch future to be resolved or timeout
        try:
            await asyncio.wait_for(fut, timeout=timeout)
            return request_id
        except asyncio.TimeoutError:
            if not fut.done():
                fut.cancel()
            async with self._lock:
                if priority == RequestPriority.HIGH:
                    self._high_priority_queued = max(0, self._high_priority_queued - 1)
                else:
                    self._low_priority_queued = max(0, self._low_priority_queued - 1)
            raise

    async def release(self) -> None:
        """Release processing slot and dispatch next highest-priority queued request."""
        async with self._lock:
            self._active_count = max(0, self._active_count - 1)
            self._dispatch_next()

    def _dispatch_next(self) -> None:
        """Dispatch next waiting request from the queue if slots are available."""
        while self._active_count < self.max_active and not self._queue.empty():
            try:
                priority, _, _, fut = self._queue.get_nowait()
                if fut.cancelled():
                    continue

                if priority == int(RequestPriority.HIGH):
                    self._high_priority_queued = max(0, self._high_priority_queued - 1)
                else:
                    self._low_priority_queued = max(0, self._low_priority_queued - 1)

                self._active_count += 1
                fut.set_result(None)
            except asyncio.QueueEmpty:
                break

    def get_metrics(self) -> Dict[str, Any]:
        """Observability telemetry for queue status."""
        return {
            "active_requests": self._active_count,
            "max_active_limit": self.max_active,
            "total_queue_depth": self.queue_depth,
            "high_priority_queued": self._high_priority_queued,
            "low_priority_queued": self._low_priority_queued,
            "max_queue_capacity": self.max_queue,
        }
