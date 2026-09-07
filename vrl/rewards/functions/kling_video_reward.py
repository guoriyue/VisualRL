"""Kling VideoReward entry point for world-model RL.

``KlingVideoReward`` writes each sample's media to disk and scores it through
the configured in-process or HTTP runtime. ``DiskArtifactRewardFunction`` is the
transport capability boundary; this file only pins the Kling video-reward model
factory and its defaults.
"""

from __future__ import annotations

from vrl.rewards.base import DiskArtifactRewardFunction


class KlingVideoReward(DiskArtifactRewardFunction):
    """Kling VideoReward scored from disk artifacts."""

    model_factory = "vrl.rewards.models.kling_video_reward:KlingVideoRewardModel"
    request_prefix = "kling-video-reward"
    debug_basename = "kling_video_reward"
    default_reward_name = "kling_video_reward"
    default_score_key = "overall_reward"
    # The production gate (vrl/config/validation.py) validates this reward for
    # the three video prompt task types its datasets ship.
    production_task_types = frozenset({"text_to_video", "image_to_video", "video2world"})


__all__ = ["KlingVideoReward"]
