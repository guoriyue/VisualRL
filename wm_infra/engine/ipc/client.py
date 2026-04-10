"""Engine IPC client: ZMQ DEALER that talks to EngineIPCServer.

Provides an async interface matching the engine semantics:
``submit``, ``get_status``, ``get_result``, ``cancel``, ``health``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid

import zmq
import zmq.asyncio

from wm_infra.engine.ipc.protocol import decode_msg, encode_msg

logger = logging.getLogger(__name__)


class EngineIPCClient:
    """Async ZMQ DEALER client for the engine IPC protocol."""

    def __init__(self, ipc_path: str = "/tmp/wm-engine.sock") -> None:
        self.ipc_path = ipc_path
        self._ctx = zmq.asyncio.Context()
        self._sock: zmq.asyncio.Socket | None = None
        self._recv_task: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[tuple[str, dict]]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect DEALER socket and start recv loop."""
        self._sock = self._ctx.socket(zmq.DEALER)
        self._sock.connect(f"ipc://{self.ipc_path}")
        self._recv_task = asyncio.create_task(self._recv_loop())
        logger.info("EngineIPCClient connected to ipc://%s", self.ipc_path)

    async def stop(self) -> None:
        """Cancel recv loop and close socket."""
        if self._recv_task is not None:
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task
            self._recv_task = None
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
        self._ctx.term()
        # Cancel any remaining pending futures
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(self, request_id: str, num_steps: int, **kwargs) -> dict:
        """Send submit request, await SubmitAck."""
        payload = {"request_id": request_id, "num_steps": num_steps, **kwargs}
        _, resp = await self._send_recv("submit", payload)
        return resp

    async def get_status(self, request_id: str) -> dict:
        """Send status query, await StatusResp."""
        _, resp = await self._send_recv("status", {"request_id": request_id})
        return resp

    async def get_result(self, request_id: str) -> dict:
        """Send result query, await ResultResp with ArtifactRef."""
        _, resp = await self._send_recv("result", {"request_id": request_id})
        return resp

    async def cancel(self, request_id: str) -> dict:
        """Send cancel request, await CancelAck."""
        _, resp = await self._send_recv("cancel", {"request_id": request_id})
        return resp

    async def health(self) -> dict:
        """Send health query, await HealthResp."""
        _, resp = await self._send_recv("health", {})
        return resp

    async def submit_and_wait(
        self,
        request_id: str,
        num_steps: int,
        poll_interval: float = 0.1,
        **kwargs,
    ) -> dict:
        """Convenience: submit + poll status until done + get_result."""
        ack = await self.submit(request_id, num_steps, **kwargs)
        if not ack.get("accepted"):
            return ack
        while True:
            await asyncio.sleep(poll_interval)
            status = await self.get_status(request_id)
            if status.get("phase") == "done":
                return await self.get_result(request_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _send_recv(self, msg_type: str, payload: dict) -> tuple[str, dict]:
        """Send request with auto-generated cid, match response by cid."""
        assert self._sock is not None, "Client not started"
        cid = uuid.uuid4().hex
        data = encode_msg(msg_type, cid, payload)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[str, dict]] = loop.create_future()
        self._pending[cid] = future

        # DEALER framing: [b"", data]
        await self._sock.send_multipart([b"", data])
        return await future

    async def _recv_loop(self) -> None:
        """DEALER recv loop: recv_multipart -> route to pending future by cid."""
        assert self._sock is not None
        while True:
            try:
                frames = await self._sock.recv_multipart()
            except zmq.ZMQError:
                break
            except asyncio.CancelledError:
                break

            # DEALER framing: [b"", data]
            if len(frames) < 2:
                continue
            data = frames[1]

            try:
                resp_type, cid, resp_payload = decode_msg(data)
                future = self._pending.pop(cid, None)
                if future is not None and not future.done():
                    future.set_result((resp_type, resp_payload))
            except Exception:
                logger.exception("Error decoding IPC response")
