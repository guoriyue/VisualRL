"""Trainer-package fixtures: the one-rank gloo group the strategy tests share."""

from __future__ import annotations

import pytest

from tests.trainers._strategy_policies import free_port


@pytest.fixture(scope="module")
def cpu_process_group():
    """One gloo world_size=1 group for the collective tests in the requesting module.

    Uses a free ephemeral port and a self-restoring MonkeyPatch so the torchrun env
    vars do not leak into the rest of the suite.
    """

    import torch.distributed as dist

    mp = pytest.MonkeyPatch()
    mp.setenv("MASTER_ADDR", "127.0.0.1")
    mp.setenv("MASTER_PORT", str(free_port()))
    mp.setenv("RANK", "0")
    mp.setenv("WORLD_SIZE", "1")
    mp.setenv("LOCAL_RANK", "0")
    created = False
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        created = True
    yield
    if created and dist.is_initialized():
        dist.destroy_process_group()
    mp.undo()
