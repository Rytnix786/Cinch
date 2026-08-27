"""Unit test suite for Dual-Tier Priority Request Queue."""

from __future__ import annotations

import asyncio
import pytest
from gateway.priority_queue import PriorityRequestQueue, RequestPriority


@pytest.mark.asyncio
async def test_priority_queue_immediate_acquire() -> None:
    """Verify immediate acquisition under active limit."""
    queue = PriorityRequestQueue(max_active=2, max_queue=10)
    req1 = await queue.acquire(RequestPriority.HIGH)
    req2 = await queue.acquire(RequestPriority.LOW)
    assert req1 != req2
    assert queue.active_requests == 2
    assert queue.queue_depth == 0

    await queue.release()
    assert queue.active_requests == 1
    await queue.release()
    assert queue.active_requests == 0


@pytest.mark.asyncio
async def test_priority_queue_preemption_ordering() -> None:
    """Verify high-priority requests jump ahead of low-priority backlog."""
    queue = PriorityRequestQueue(max_active=1, max_queue=10)

    # 1. Occupy active slot
    await queue.acquire(RequestPriority.HIGH)
    assert queue.active_requests == 1

    execution_order: list[str] = []

    async def task_worker(name: str, priority: RequestPriority) -> None:
        await queue.acquire(priority=priority)
        execution_order.append(name)
        await queue.release()

    # 2. Queue low priority task first
    low_task = asyncio.create_task(task_worker("low_1", RequestPriority.LOW))
    await asyncio.sleep(0.01)

    # 3. Queue high priority task second (should preempt low_1)
    high_task = asyncio.create_task(task_worker("high_1", RequestPriority.HIGH))
    await asyncio.sleep(0.01)

    assert queue.queue_depth == 2

    # 4. Release active slot -> high_1 should be dispatched before low_1
    await queue.release()
    await asyncio.gather(high_task, low_task)

    assert execution_order == ["high_1", "low_1"]


@pytest.mark.asyncio
async def test_priority_queue_timeout() -> None:
    """Verify timeout when queue wait threshold is exceeded."""
    queue = PriorityRequestQueue(max_active=1, max_queue=5)
    await queue.acquire(RequestPriority.HIGH)

    with pytest.raises(asyncio.TimeoutError):
        await queue.acquire(RequestPriority.LOW, timeout=0.05)

    assert queue.queue_depth == 0
    await queue.release()


@pytest.mark.asyncio
async def test_priority_queue_capacity_overflow() -> None:
    """Verify rejection when queue capacity is reached."""
    queue = PriorityRequestQueue(max_active=1, max_queue=2)
    await queue.acquire(RequestPriority.HIGH)

    # Fill queue to capacity (2)
    task1 = asyncio.create_task(queue.acquire(RequestPriority.LOW))
    task2 = asyncio.create_task(queue.acquire(RequestPriority.LOW))
    await asyncio.sleep(0.01)
    assert queue.queue_depth == 2

    # 3rd queued request should fail immediately with RuntimeError
    with pytest.raises(RuntimeError, match="priority queue full"):
        await queue.acquire(RequestPriority.LOW)

    await queue.release()
    await queue.release()
    await queue.release()
    await asyncio.gather(task1, task2, return_exceptions=True)
