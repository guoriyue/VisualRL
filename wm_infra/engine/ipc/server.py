"""Engine IPC server: ZMQ ROUTER that wraps an EngineLoop.

Binds a ROUTER socket on ``ipc:///tmp/wm-engine.sock`` and dispatches
incoming JSON messages to the underlying ``EngineLoop``.  Submit is
non-blocking — the server registers a done-callback that writes artifacts
to tmpfs when the future resolves.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal

import zmq
import zmq.asyncio

from wm_infra.engine.ipc.artifacts import ArtifactStore
from wm_infra.engine.ipc.protocol import ArtifactRef, MsgType, decode_msg, encode_msg
from wm_infra.engine.managers.engine_loop import EngineLoop
from wm_infra.engine.types import EntityRequest, StepResult

logger = logging.getLogger(__name__)


class EngineIPCServer:
    """ZMQ ROUTER server that adapts the IPC protocol to ``EngineLoop``."""

    def __init__(
        self,
        engine_loop: EngineLoop,
        *,
        ipc_path: str = "/tmp/wm-engine.sock",
        artifact_root: str = "/dev/shm/wm-engine",
    ) -> None:
        self.engine = engine_loop
        self.ipc_path = ipc_path
        self.store = ArtifactStore(root=artifact_root)

        self._ctx = zmq.asyncio.Context()
        self._sock: zmq.asyncio.Socket | None = None
        self._recv_task: asyncio.Task[None] | None = None

        # Track pending futures so we can write artifacts on completion
        self._pending: dict[str, asyncio.Future[list[StepResult]]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start engine loop + ZMQ recv loop."""
        await self.engine.start()
        self._sock = self._ctx.socket(zmq.ROUTER)
        self._sock.bind(f"ipc://{self.ipc_path}")
        self._recv_task = asyncio.create_task(self._recv_loop())
        logger.info("EngineIPCServer listening on ipc://%s", self.ipc_path)

    async def stop(self) -> None:
        """Stop engine loop + close ZMQ socket."""
        if self._recv_task is not None:
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task
            self._recv_task = None
        await self.engine.stop()
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
        self._ctx.term()

    # ------------------------------------------------------------------
    # Recv loop
    # ------------------------------------------------------------------

    async def _recv_loop(self) -> None:
        """ROUTER recv loop: recv_multipart -> dispatch -> send_multipart."""
        assert self._sock is not None
        while True:
            try:
                frames = await self._sock.recv_multipart()
            except zmq.ZMQError:
                break
            except asyncio.CancelledError:
                break

            # ROUTER framing: [identity, b"", data]
            if len(frames) < 3:
                continue
            identity = frames[0]
            data = frames[2]

            try:
                msg_type, cid, payload = decode_msg(data)
                resp_type, resp_payload = await self._dispatch(msg_type, payload)
                resp_data = encode_msg(resp_type, cid, resp_payload)
                await self._sock.send_multipart([identity, b"", resp_data])
            except Exception:
                logger.exception("Error handling IPC message")

    async def _dispatch(self, msg_type: str, payload: dict) -> tuple[str, dict]:
        """Route a request to the appropriate handler."""
        if msg_type == MsgType.SUBMIT:
            return MsgType.SUBMIT_ACK, await self._handle_submit(payload)
        if msg_type == MsgType.CANCEL:
            return MsgType.CANCEL_ACK, self._handle_cancel(payload)
        if msg_type == MsgType.STATUS:
            return MsgType.STATUS_RESP, self._handle_status(payload)
        if msg_type == MsgType.RESULT:
            return MsgType.RESULT_RESP, self._handle_result(payload)
        if msg_type == MsgType.HEALTH:
            return MsgType.HEALTH_RESP, self._handle_health()
        return "error", {"error": f"unknown message type: {msg_type}"}

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _handle_submit(self, payload: dict) -> dict:
        """Submit a rollout. Non-blocking — returns ack immediately."""
        request_id: str = payload["request_id"]
        num_steps: int = payload["num_steps"]

        request = EntityRequest(
            request_id=request_id,
            num_steps=num_steps,
            action_sequence=payload.get("action_sequence", []),
            metadata=payload.get("metadata", {}),
            priority=payload.get("priority", 0.0),
            prefix_hash=payload.get("prefix_hash"),
        )

        try:
            future = self.engine.submit_nowait(request)
            self._pending[request_id] = future
            future.add_done_callback(
                lambda f, rid=request_id: asyncio.ensure_future(
                    self._on_request_done(rid, f)
                )
            )
            queue_pos = self.engine.num_pending() + self.engine.num_waiting()
            return {"request_id": request_id, "accepted": True, "queue_position": queue_pos}
        except Exception as exc:
            return {"request_id": request_id, "accepted": False, "error": str(exc)}

    def _handle_cancel(self, payload: dict) -> dict:
        """Cancel a rollout request."""
        request_id: str = payload["request_id"]
        cancelled = self.engine.scheduler.abort_request(request_id)
        self._pending.pop(request_id, None)
        return {"request_id": request_id, "cancelled": cancelled}

    def _handle_status(self, payload: dict) -> dict:
        """Return phase + step progress for a request."""
        request_id: str = payload["request_id"]
        state = self.engine.scheduler.get_state(request_id)
        if state is None:
            # Might be already completed and drained — check artifact store
            meta = self.store.read_meta(request_id)
            if meta is not None:
                return {
                    "request_id": request_id,
                    "phase": "done",
                    "step_index": len(meta),
                    "num_steps": len(meta),
                }
            return {"request_id": request_id, "phase": "unknown", "step_index": 0, "num_steps": 0}
        return {
            "request_id": request_id,
            "phase": state.phase.name.lower(),
            "step_index": state.step_index,
            "num_steps": state.request.num_steps,
        }

    def _handle_result(self, payload: dict) -> dict:
        """Return artifact ref if done, otherwise done=False."""
        request_id: str = payload["request_id"]
        meta = self.store.read_meta(request_id)
        if meta is not None:
            latent_path = self.store.read_latent_path(request_id)
            artifact = ArtifactRef(
                path=str(self.store.root / request_id),
                size_bytes=latent_path.stat().st_size if latent_path else 0,
            )
            if latent_path:
                import numpy as np

                arr = np.load(str(latent_path))
                artifact.shape = arr.shape
                artifact.dtype = str(arr.dtype)
            return {
                "request_id": request_id,
                "done": True,
                "artifact": artifact.to_dict(),
                "step_metas": meta,
            }
        return {"request_id": request_id, "done": False}

    def _handle_health(self) -> dict:
        """Return engine stats."""
        return {
            "num_pending": self.engine.num_pending(),
            "num_active": self.engine.num_active(),
            "num_waiting": self.engine.num_waiting(),
            "num_free_blocks": self.engine.pool.num_free_blocks,
        }

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    async def _on_request_done(
        self, request_id: str, future: asyncio.Future[list[StepResult]]
    ) -> None:
        """Write artifacts to tmpfs when an engine future resolves."""
        self._pending.pop(request_id, None)
        if future.cancelled():
            return
        exc = future.exception()
        if exc is not None:
            logger.error("Request %s failed: %s", request_id, exc)
            return
        results = future.result()
        try:
            self.store.write_results(request_id, results)
        except Exception:
            logger.exception("Failed to write artifacts for %s", request_id)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main() -> None:
    """Run the engine IPC server standalone."""
    from wm_infra.config import load_config
    from wm_infra.engine.model_executor.worker import DynamicsStage, EncodeStage
    from wm_infra.engine.types import EngineRunConfig

    config = load_config()
    run_config = EngineRunConfig(
        max_num_blocks=config.state_cache.max_batch_size * 4,
        block_size=1,
        latent_tokens=config.state_cache.num_latent_tokens,
        latent_dim=config.state_cache.latent_dim,
        max_batch_size=config.scheduler.max_batch_size,
        max_steps_per_entity=config.state_cache.max_rollout_steps,
        device=config.device.value if hasattr(config.device, "value") else str(config.device),
    )

    engine = EngineLoop(run_config)
    engine.register_stage(EncodeStage())
    engine.register_stage(DynamicsStage())

    server = EngineIPCServer(
        engine,
        ipc_path=config.ipc.socket_path,
        artifact_root=config.ipc.artifact_root,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run() -> None:
        await server.start()
        stop_event = asyncio.Event()

        def _signal_handler() -> None:
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)

        logger.info("Engine IPC server running. Press Ctrl+C to stop.")
        await stop_event.wait()
        logger.info("Shutting down engine IPC server...")
        await server.stop()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    loop.run_until_complete(run())


if __name__ == "__main__":
    main()
