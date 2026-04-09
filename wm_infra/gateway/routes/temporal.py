"""Gateway routes for temporal entities."""

from __future__ import annotations

from fastapi import APIRouter, FastAPI, HTTPException, Request

from wm_infra.controlplane import BranchCreate, CheckpointCreate, EpisodeCreate, RolloutCreate, StateHandleCreate, TemporalStore
from wm_infra.gateway.state import get_gateway_runtime


def register_temporal_routes(app: FastAPI) -> None:
    """Register temporal entity routes on the Gateway app."""
    router = APIRouter()

    @router.post("/v1/episodes")
    async def create_episode(request: EpisodeCreate, http_request: Request):
        store: TemporalStore = get_gateway_runtime(http_request).temporal_store
        return store.create_episode(request).model_dump(mode="json")

    @router.get("/v1/episodes")
    async def list_episodes(request: Request):
        store: TemporalStore = get_gateway_runtime(request).temporal_store
        episodes = store.episodes.list()
        return {"episodes": [e.model_dump(mode="json") for e in episodes], "count": len(episodes)}

    @router.get("/v1/episodes/{episode_id}")
    async def get_episode(episode_id: str, request: Request):
        store: TemporalStore = get_gateway_runtime(request).temporal_store
        episode = store.episodes.get(episode_id)
        if episode is None:
            raise HTTPException(status_code=404, detail="Episode not found")
        return episode.model_dump(mode="json")

    @router.post("/v1/branches")
    async def create_branch(request: BranchCreate, http_request: Request):
        store: TemporalStore = get_gateway_runtime(http_request).temporal_store
        if store.episodes.get(request.episode_id) is None:
            raise HTTPException(status_code=404, detail="Episode not found")
        if request.parent_branch_id and store.branches.get(request.parent_branch_id) is None:
            raise HTTPException(status_code=404, detail="Parent branch not found")
        if request.forked_from_checkpoint_id and store.checkpoints.get(request.forked_from_checkpoint_id) is None:
            raise HTTPException(status_code=404, detail="Fork checkpoint not found")
        return store.create_branch(request).model_dump(mode="json")

    @router.get("/v1/branches")
    async def list_branches(request: Request, episode_id: str | None = None):
        store: TemporalStore = get_gateway_runtime(request).temporal_store
        branches = store.branches.list()
        if episode_id is not None:
            branches = [b for b in branches if b.episode_id == episode_id]
        return {"branches": [b.model_dump(mode="json") for b in branches], "count": len(branches)}

    @router.get("/v1/branches/{branch_id}")
    async def get_branch(branch_id: str, request: Request):
        store: TemporalStore = get_gateway_runtime(request).temporal_store
        branch = store.branches.get(branch_id)
        if branch is None:
            raise HTTPException(status_code=404, detail="Branch not found")
        return branch.model_dump(mode="json")

    @router.post("/v1/state-handles")
    async def create_state_handle(request: StateHandleCreate, http_request: Request):
        store: TemporalStore = get_gateway_runtime(http_request).temporal_store
        if store.episodes.get(request.episode_id) is None:
            raise HTTPException(status_code=404, detail="Episode not found")
        if request.branch_id and store.branches.get(request.branch_id) is None:
            raise HTTPException(status_code=404, detail="Branch not found")
        if request.rollout_id and store.rollouts.get(request.rollout_id) is None:
            raise HTTPException(status_code=404, detail="Rollout not found")
        if request.checkpoint_id and store.checkpoints.get(request.checkpoint_id) is None:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        return store.create_state_handle(request).model_dump(mode="json")

    @router.get("/v1/state-handles")
    async def list_state_handles(request: Request, episode_id: str | None = None, branch_id: str | None = None):
        store: TemporalStore = get_gateway_runtime(request).temporal_store
        items = store.state_handles.list()
        if episode_id is not None:
            items = [i for i in items if i.episode_id == episode_id]
        if branch_id is not None:
            items = [i for i in items if i.branch_id == branch_id]
        return {"state_handles": [i.model_dump(mode="json") for i in items], "count": len(items)}

    @router.get("/v1/state-handles/{state_handle_id}")
    async def get_state_handle(state_handle_id: str, request: Request):
        store: TemporalStore = get_gateway_runtime(request).temporal_store
        item = store.state_handles.get(state_handle_id)
        if item is None:
            raise HTTPException(status_code=404, detail="State handle not found")
        return item.model_dump(mode="json")

    @router.post("/v1/rollouts")
    async def create_temporal_rollout(request: RolloutCreate, http_request: Request):
        store: TemporalStore = get_gateway_runtime(http_request).temporal_store
        if store.episodes.get(request.episode_id) is None:
            raise HTTPException(status_code=404, detail="Episode not found")
        if request.branch_id and store.branches.get(request.branch_id) is None:
            raise HTTPException(status_code=404, detail="Branch not found")
        if request.input_state_handle_id and store.state_handles.get(request.input_state_handle_id) is None:
            raise HTTPException(status_code=404, detail="Input state handle not found")
        return store.create_rollout(request).model_dump(mode="json")

    @router.get("/v1/rollouts")
    async def list_temporal_rollouts(request: Request, episode_id: str | None = None, branch_id: str | None = None):
        store: TemporalStore = get_gateway_runtime(request).temporal_store
        items = store.rollouts.list()
        if episode_id is not None:
            items = [i for i in items if i.episode_id == episode_id]
        if branch_id is not None:
            items = [i for i in items if i.branch_id == branch_id]
        return {"rollouts": [i.model_dump(mode="json") for i in items], "count": len(items)}

    @router.get("/v1/rollouts/{rollout_id}")
    async def get_temporal_rollout(rollout_id: str, request: Request):
        store: TemporalStore = get_gateway_runtime(request).temporal_store
        item = store.rollouts.get(rollout_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Rollout not found")
        return item.model_dump(mode="json")

    @router.post("/v1/checkpoints")
    async def create_checkpoint(request: CheckpointCreate, http_request: Request):
        store: TemporalStore = get_gateway_runtime(http_request).temporal_store
        if store.episodes.get(request.episode_id) is None:
            raise HTTPException(status_code=404, detail="Episode not found")
        if request.rollout_id and store.rollouts.get(request.rollout_id) is None:
            raise HTTPException(status_code=404, detail="Rollout not found")
        if request.branch_id and store.branches.get(request.branch_id) is None:
            raise HTTPException(status_code=404, detail="Branch not found")
        if request.state_handle_id and store.state_handles.get(request.state_handle_id) is None:
            raise HTTPException(status_code=404, detail="State handle not found")
        checkpoint = store.create_checkpoint(request)
        if request.rollout_id:
            store.attach_checkpoint_to_rollout(request.rollout_id, checkpoint.checkpoint_id)
        return checkpoint.model_dump(mode="json")

    @router.get("/v1/checkpoints")
    async def list_checkpoints(request: Request, episode_id: str | None = None, rollout_id: str | None = None):
        store: TemporalStore = get_gateway_runtime(request).temporal_store
        items = store.checkpoints.list()
        if episode_id is not None:
            items = [i for i in items if i.episode_id == episode_id]
        if rollout_id is not None:
            items = [i for i in items if i.rollout_id == rollout_id]
        return {"checkpoints": [i.model_dump(mode="json") for i in items], "count": len(items)}

    @router.get("/v1/checkpoints/{checkpoint_id}")
    async def get_checkpoint(checkpoint_id: str, request: Request):
        store: TemporalStore = get_gateway_runtime(request).temporal_store
        item = store.checkpoints.get(checkpoint_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        return item.model_dump(mode="json")

    app.include_router(router)
