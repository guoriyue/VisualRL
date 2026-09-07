"""DINOv2 perceptual Video2World target-similarity reward (local, CPU/GPU).

Thin function-layer binding registered as ``target_dino_similarity`` (Future
Reward suite, SPRINT_future_reward): the registry builds it from YAML; the
model is constructed eagerly here and driven through the in-process scorer.
The scoring design — frozen DINOv2 cosine as the reward-hack-resistant
successor to the retired pixel-L1 ``target_video_similarity`` — is documented
in ``vrl.rewards.models.target_dino_similarity``.
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import InferenceRewardFunction
from vrl.rewards.models.target_dino_similarity import TargetDinoSimilarityModel
from vrl.rewards.runtime import InProcessRewardScorer


class TargetDinoSimilarityReward(InferenceRewardFunction):
    """Local reward comparing generated video to target media by DINOv2 cosine."""

    def __init__(
        self,
        device: str = "",
        reward_name: str = "target_dino_similarity",
        score_key: str = "target_dino_similarity",
        worker_config: dict[str, Any] | None = None,
    ) -> None:
        cfg = self.worker_config_with_device(worker_config, device=str(device))
        model = TargetDinoSimilarityModel(cfg)
        super().__init__(
            reward_name=reward_name,
            score_key=score_key,
            scorer=InProcessRewardScorer(model=model),
        )


__all__ = ["TargetDinoSimilarityReward"]
