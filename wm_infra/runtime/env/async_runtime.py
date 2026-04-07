"""Async dispatch/collect substrate for learned env stepping."""

from __future__ import annotations

import asyncio
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class TransitionDispatch:
    dispatch_id: str
    submitted_at: float
    item_count: int
    metadata: dict[str, Any]


class AsyncTransitionDispatcher:
    """Minimal send/collect-style dispatcher for transition batches."""

    def __init__(self, *, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_workers))
        self._pending: dict[str, Future[Any]] = {}

    def dispatch(
        self,
        *,
        item_count: int,
        fn: Callable[..., Any],
        metadata: dict[str, Any] | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> TransitionDispatch:
        dispatch = TransitionDispatch(
            dispatch_id=str(uuid.uuid4()),
            submitted_at=time.time(),
            item_count=item_count,
            metadata=dict(metadata or {}),
        )
        self._pending[dispatch.dispatch_id] = self._executor.submit(fn, *args, **(kwargs or {}))
        return dispatch

    def collect(self, dispatch_id: str, *, timeout: float | None = None) -> Any:
        future = self._pending.pop(dispatch_id)
        return future.result(timeout=timeout)

    async def collect_async(self, dispatch_id: str, *, timeout: float | None = None) -> Any:
        future = self._pending.pop(dispatch_id)
        wrapped = asyncio.wrap_future(future)
        if timeout is None:
            return await wrapped
        return await asyncio.wait_for(wrapped, timeout=timeout)

    def pending_count(self) -> int:
        return len(self._pending)


__all__ = ["AsyncTransitionDispatcher", "TransitionDispatch"]
