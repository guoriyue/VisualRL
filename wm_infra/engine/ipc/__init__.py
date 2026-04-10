"""ZMQ IPC layer for cross-process gateway <-> engine communication."""

from wm_infra.engine.ipc.artifacts import ArtifactStore
from wm_infra.engine.ipc.client import EngineIPCClient
from wm_infra.engine.ipc.protocol import ArtifactRef, MsgType, decode_msg, encode_msg
from wm_infra.engine.ipc.server import EngineIPCServer

__all__ = [
    "ArtifactRef",
    "ArtifactStore",
    "EngineIPCClient",
    "EngineIPCServer",
    "MsgType",
    "decode_msg",
    "encode_msg",
]
