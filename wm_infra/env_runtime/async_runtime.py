"""Async dispatch/collect substrate for learned env stepping.

This module is intentionally queue-first rather than a thin threadpool wrapper.
The runtime can enqueue homogeneous transition work, seal it into batches by
``batch_key``, and collect results by dispatch or batch id.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable


@dataclass(slots=True)
class TransitionCall:
    """Compatibility payload for callable-based dispatches."""

    fn: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def execute(self) -> Any:
        return self.fn(*self.args, **self.kwargs)


@dataclass(slots=True)
class TransitionDispatch:
    """Handle for one queued transition item."""

    dispatch_id: str
    batch_key: Hashable
    submitted_at: float
    metadata: dict[str, Any]
    batch_id: str | None = None
    batch_size: int = 1
    batch_position: int = 0


@dataclass(slots=True)
class TransitionBatch:
    """A sealed homogeneous batch waiting for or running execution."""

    batch_id: str
    batch_key: Hashable
    dispatch_ids: list[str]
    submitted_at: float
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: float | None = None
    completed_at: float | None = None
    result_ready: bool = False


@dataclass(slots=True)
class _QueuedItem:
    dispatch: TransitionDispatch
    payload: Any
    future: Future[Any]
    batch_future: Future[list[Any]] | None = None


class AsyncTransitionDispatcher:
    """Queue-first send/collect dispatcher for homogeneous transition batches."""

    def __init__(
        self,
        *,
        max_workers: int = 2,
        max_batch_size: int = 8,
        batch_runner: Callable[[TransitionBatch, list[Any]], list[Any]] | None = None,
    ) -> None:
        self.max_batch_size = max(1, int(max_batch_size))
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_workers))
        self._batch_runner = batch_runner or self._default_batch_runner
        self._lock = threading.RLock()
        self._pending_by_key: dict[Hashable, deque[_QueuedItem]] = defaultdict(deque)
        self._dispatches: dict[str, _QueuedItem] = {}
        self._batch_futures: dict[str, Future[list[Any]]] = {}
        self._batch_records: dict[str, TransitionBatch] = {}

    def send(
        self,
        *,
        batch_key: Hashable,
        payload: Any,
        metadata: dict[str, Any] | None = None,
    ) -> TransitionDispatch:
        """Queue one homogeneous transition payload for later batch execution."""

        dispatch = TransitionDispatch(
            dispatch_id=str(uuid.uuid4()),
            batch_key=batch_key,
            submitted_at=time.time(),
            metadata=dict(metadata or {}),
        )
        item = _QueuedItem(dispatch=dispatch, payload=payload, future=Future())

        with self._lock:
            self._pending_by_key[batch_key].append(item)
            self._dispatches[dispatch.dispatch_id] = item
            if len(self._pending_by_key[batch_key]) >= self.max_batch_size:
                self._seal_ready_batches_locked(batch_key=batch_key)
        return dispatch

    def send_many(
        self,
        *,
        batch_key: Hashable,
        payloads: list[Any],
        metadata: dict[str, Any] | None = None,
        item_metadata: list[dict[str, Any] | None] | None = None,
    ) -> list[TransitionDispatch]:
        """Queue many homogeneous payloads that should share one batch key."""

        dispatches: list[TransitionDispatch] = []
        item_metadata = item_metadata or [None] * len(payloads)
        if len(item_metadata) != len(payloads):
            raise ValueError("item_metadata must match payloads length")
        for payload, payload_metadata in zip(payloads, item_metadata, strict=True):
            combined_metadata = dict(metadata or {})
            if payload_metadata:
                combined_metadata.update(payload_metadata)
            dispatches.append(self.send(batch_key=batch_key, payload=payload, metadata=combined_metadata))
        return dispatches

    def dispatch(
        self,
        *,
        item_count: int,
        fn: Callable[..., Any],
        metadata: dict[str, Any] | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> TransitionDispatch:
        """Compatibility wrapper for older callable-based dispatch call sites."""

        call = TransitionCall(fn=fn, args=args, kwargs=dict(kwargs or {}))
        batch_key = (metadata or {}).get("batch_key", ("callable", fn.__module__, getattr(fn, "__qualname__", fn.__name__)))
        return self.send(
            batch_key=batch_key,
            payload=call,
            metadata={**dict(metadata or {}), "item_count": item_count, "compat_dispatch": True},
        )

    def flush(self, batch_key: Hashable | None = None) -> list[TransitionBatch]:
        """Seal any pending items into executable batches."""

        with self._lock:
            return self._seal_ready_batches_locked(batch_key=batch_key)

    def collect(self, dispatch_id: str, *, timeout: float | None = None) -> Any:
        """Collect one dispatch result, flushing its batch if needed."""

        future = self._ensure_dispatch_future(dispatch_id)
        return future.result(timeout=timeout)

    async def collect_async(self, dispatch_id: str, *, timeout: float | None = None) -> Any:
        """Async collect for event-loop callers."""

        future = self._ensure_dispatch_future(dispatch_id)
        wrapped = asyncio.wrap_future(future)
        if timeout is None:
            return await wrapped
        return await asyncio.wait_for(wrapped, timeout=timeout)

    def collect_batch(self, batch_id: str, *, timeout: float | None = None) -> list[Any]:
        """Collect an entire sealed batch."""

        batch_future = self._ensure_batch_future(batch_id)
        return batch_future.result(timeout=timeout)

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for item in self._dispatches.values() if not item.future.done())

    def pending_batch_count(self) -> int:
        with self._lock:
            return sum(len(items) for items in self._pending_by_key.values())

    def snapshot(self) -> dict[str, Any]:
        """Return a lightweight queue snapshot for observability/tests."""

        with self._lock:
            pending_by_key = {str(key): len(items) for key, items in self._pending_by_key.items() if items}
            pending_count = sum(1 for item in self._dispatches.values() if not item.future.done())
            pending_item_count = sum(len(items) for items in self._pending_by_key.values())
            batches = {
                batch_id: {
                    "batch_key": str(batch.batch_key),
                    "dispatch_ids": list(batch.dispatch_ids),
                    "metadata": dict(batch.metadata),
                    "started_at": batch.started_at,
                    "completed_at": batch.completed_at,
                    "result_ready": batch.result_ready,
                }
                for batch_id, batch in self._batch_records.items()
            }
            return {
                "pending_count": pending_count,
                "pending_batch_count": pending_item_count,
                "pending_by_key": pending_by_key,
                "sealed_batches": batches,
            }

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _ensure_dispatch_future(self, dispatch_id: str) -> Future[Any]:
        with self._lock:
            item = self._dispatches.get(dispatch_id)
            if item is None:
                raise KeyError(dispatch_id)
            if item.batch_future is None:
                self._seal_ready_batches_locked(batch_key=item.dispatch.batch_key)
            return item.future

    def _ensure_batch_future(self, batch_id: str) -> Future[list[Any]]:
        with self._lock:
            batch_future = self._batch_futures.get(batch_id)
            if batch_future is None:
                raise KeyError(batch_id)
            return batch_future

    def _seal_ready_batches_locked(self, batch_key: Hashable | None = None) -> list[TransitionBatch]:
        keys = [batch_key] if batch_key is not None else [key for key, items in self._pending_by_key.items() if items]
        sealed_batches: list[TransitionBatch] = []
        for key in keys:
            queue = self._pending_by_key.get(key)
            while queue:
                batch_items = [queue.popleft() for _ in range(min(len(queue), self.max_batch_size))]
                if not batch_items:
                    break
                sealed_batches.append(self._submit_batch_locked(key, batch_items))
            if queue is not None and not queue:
                self._pending_by_key.pop(key, None)
        return sealed_batches

    def _submit_batch_locked(self, batch_key: Hashable, items: list[_QueuedItem]) -> TransitionBatch:
        batch_id = str(uuid.uuid4())
        dispatch_ids = [item.dispatch.dispatch_id for item in items]
        batch_metadata = self._batch_metadata(items)
        batch = TransitionBatch(
            batch_id=batch_id,
            batch_key=batch_key,
            dispatch_ids=dispatch_ids,
            submitted_at=time.time(),
            metadata=batch_metadata,
        )
        batch_future = self._executor.submit(self._run_batch, batch, items)
        self._batch_futures[batch_id] = batch_future
        self._batch_records[batch_id] = batch
        for index, item in enumerate(items):
            item.dispatch.batch_id = batch_id
            item.dispatch.batch_size = len(items)
            item.dispatch.batch_position = index
            item.batch_future = batch_future
        return batch

    def _run_batch(self, batch: TransitionBatch, items: list[_QueuedItem]) -> list[Any]:
        batch.started_at = time.time()
        payloads = [item.payload for item in items]
        try:
            results = self._batch_runner(batch, payloads)
            if len(results) != len(items):
                raise ValueError("batch_runner must return one result per queued item")
            for item, result in zip(items, results, strict=True):
                item.future.set_result(result)
            batch.completed_at = time.time()
            batch.result_ready = True
            return results
        except BaseException as exc:
            batch.completed_at = time.time()
            batch.result_ready = False
            for item in items:
                if not item.future.done():
                    item.future.set_exception(exc)
            raise

    @staticmethod
    def _batch_metadata(items: list[_QueuedItem]) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "item_count": len(items),
            "batch_item_metadata": [dict(item.dispatch.metadata) for item in items],
            "batch_item_ids": [item.dispatch.dispatch_id for item in items],
            "homogeneous_batch": True,
        }
        shared_keys = set.intersection(*(set(item.dispatch.metadata.keys()) for item in items)) if items else set()
        if shared_keys:
            metadata["shared_metadata"] = {
                key: items[0].dispatch.metadata[key]
                for key in sorted(shared_keys)
                if all(item.dispatch.metadata.get(key) == items[0].dispatch.metadata[key] for item in items)
            }
        return metadata

    @staticmethod
    def _default_batch_runner(batch: TransitionBatch, payloads: list[Any]) -> list[Any]:
        results: list[Any] = []
        for payload in payloads:
            if isinstance(payload, TransitionCall):
                results.append(payload.execute())
            elif callable(payload):
                results.append(payload())
            else:
                results.append(payload)
        return results


__all__ = [
    "AsyncTransitionDispatcher",
    "TransitionBatch",
    "TransitionCall",
    "TransitionDispatch",
]
