"""Loss-correctness tests for V-GRPO (algorithms/v_grpo.py).

Same harness as the DiffusionNFT tests: a tiny real ``WanTransformer3DModel``
behind real PEFT ``default`` / ``previous`` adapters and a real ``TrajectoryBatch``
from the production builder, so the two forwards the objective runs differ
because their LoRA weights genuinely differ. The math is not re-derived: the
sign tests take one real optimizer step and assert its direction, and the
lr=0 invariant is checked on the objective's own output.
"""

from __future__ import annotations

import pytest
import torch

from tests.algorithms.test_diffusion_nft import (
    _BATCH,
    _LATENT_SHAPE,
    _TEXT_DIM,
    _TEXT_LEN,
    _build_batch,
    _build_model,
    _default_forward,
)
from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
from vrl.algorithms.v_grpo import VGRPO, VGRPOConfig
from vrl.trainers.online import OnlineTrainer


def _batch(*, timestep: float | tuple[float, ...] = 500.0, seed: int = 1234):
    torch.manual_seed(seed)
    x0 = torch.randn(_LATENT_SHAPE)
    prompt_embeds = torch.randn(_BATCH, _TEXT_LEN, _TEXT_DIM)
    # The NFT harness stores a noise tensor the V-GRPO loss never reads: its
    # eps is the group-shared draw, not a replay tensor.
    return (
        x0,
        prompt_embeds,
        _build_batch(
            x0=x0, noise=torch.zeros(_LATENT_SHAPE), prompt_embeds=prompt_embeds, timestep=timestep
        ),
    )


def _synced_model():
    model = _build_model()
    model.sync_previous_policy_adapter(decay=0.0)
    return model


# ------------------------------------------------------------ advantages


@pytest.mark.parametrize("global_std", [False, True])
def test_v_grpo_advantages_share_the_grpo_group_contract(global_std: bool) -> None:
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    group_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    grpo = GRPO(GRPOConfig(eps=1e-4, global_std=global_std))
    vgrpo = VGRPO(VGRPOConfig(eps=1e-4, global_std=global_std))

    assert torch.equal(
        vgrpo.compute_advantages_from_tensors(rewards, group_ids),
        grpo.compute_advantages_from_tensors(rewards, group_ids),
    )


# ------------------------------------------------------------ objective sign


def _step_distances(*, advantage: float, config: VGRPOConfig | None = None, lr: float = 2.0):
    """One real SGD step; distance of the default forward to the reconstruction
    velocity ``noise - x0`` at the objective's own ``x_t`` before vs after.

    With ``previous`` synced the ratio is 1, so the gradient is ``-A * grad L``:
    a positive advantage must pull the x-prediction toward ``x0``, i.e. the
    velocity toward ``eps - x0`` for the eps the objective drew. The step is
    kept small: a large ascent step overshoots the curved loss and lands
    closer again, which would read as a sign error.
    """

    cfg = config or VGRPOConfig(adv_soft_clip=None, kl_coef=0.0)
    objective = VGRPO(cfg)
    x0, prompt_embeds, batch = _batch()
    model = _synced_model()
    noise = objective._group_shared_noise(x0, group_ids=batch.group_ids, timestep_index=0)
    t = 0.5
    xt = (1 - t) * x0 + t * noise
    target = noise - x0
    t_raw = torch.full((_BATCH,), 500.0)
    before = float((_default_forward(model, xt, prompt_embeds, t_raw).detach() - target).norm())
    loss, _ = objective.compute_batch_timestep_loss(model, batch, 0, torch.tensor([advantage]))
    trainable = [p for p in model.transformer.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=lr)
    opt.zero_grad()
    loss.backward()
    grad = torch.cat([p.grad.flatten() for p in trainable if p.grad is not None])
    opt.step()
    after = float((_default_forward(model, xt, prompt_embeds, t_raw).detach() - target).norm())
    return before, after, grad


def test_positive_advantage_pulls_the_prediction_toward_reconstruction() -> None:
    before, after, grad = _step_distances(advantage=5.0)
    assert grad.abs().sum() > 0
    assert after < before


def test_negative_advantage_pushes_the_prediction_away_from_reconstruction() -> None:
    before, after, _ = _step_distances(advantage=-5.0)
    assert after > before


def test_soft_clip_bounds_the_advantage_by_eta() -> None:
    """``eta * tanh(A / eta)``: a huge advantage moves the weights less than an
    unclipped one, but in the same direction."""
    _, after_clipped, grad_clipped = _step_distances(
        advantage=50.0, config=VGRPOConfig(adv_soft_clip=1.0, kl_coef=0.0)
    )
    before, _after_raw, grad_raw = _step_distances(
        advantage=50.0, config=VGRPOConfig(adv_soft_clip=None, kl_coef=0.0)
    )
    assert after_clipped < before
    assert grad_clipped.norm() < grad_raw.norm()
    # tanh(50) == 1 to fp32 precision, so the clipped gradient is the unit-advantage one.
    _, _, grad_unit = _step_distances(advantage=1.0)
    torch.testing.assert_close(grad_clipped, grad_unit, rtol=1e-4, atol=1e-6)


# ------------------------------------------------------------ ratio / clip / KL


def test_synced_behaviour_policy_gives_unit_ratio_and_zero_kl() -> None:
    """With ``previous == default`` the surrogate difference is exactly 0: the
    loss is ``-mean(A_soft)`` and the KL term vanishes."""
    _, _, batch = _batch()
    model = _synced_model()
    adv = torch.tensor([1.7])

    loss, metrics = VGRPO(VGRPOConfig(adv_soft_clip=3.0, kl_coef=0.3)).compute_batch_timestep_loss(
        model, batch, 0, adv
    )

    expected = -float(3.0 * torch.tanh(adv / 3.0))
    assert float(loss) == pytest.approx(expected, abs=1e-6)
    assert metrics.kl_penalty == 0.0
    assert metrics.weighted_kl_loss == 0.0
    assert metrics.update.approx_kl == 0.0
    assert metrics.update.clip_fraction == 0.0


def test_ratio_clipping_activates_once_the_policies_diverge() -> None:
    """A perturbed ``default`` moves the ratio off 1; with a tight epsilon the
    clipped surrogate is selected and reported, and the KL term is the x-pred
    MSE between the two policies' predictions."""
    _, _, batch = _batch()
    model = _synced_model()
    with torch.no_grad():
        for name, param in model.transformer.named_parameters():
            if ".default." in name and "lora_B" in name:
                param.add_(torch.randn_like(param) * 0.5)
    adv = torch.tensor([2.0])

    loose = VGRPO(VGRPOConfig(adv_soft_clip=None, clip_ratio=None, kl_coef=0.0))
    tight = VGRPO(VGRPOConfig(adv_soft_clip=None, clip_ratio=1e-6, kl_coef=1.0))
    loose_loss, loose_metrics = loose.compute_batch_timestep_loss(model, batch, 0, adv)
    tight_loss, tight_metrics = tight.compute_batch_timestep_loss(model, batch, 0, adv)

    assert loose_metrics.update.approx_kl > 0.0
    assert tight_metrics.update.clip_fraction == 1.0
    assert tight_metrics.kl_penalty > 0.0
    assert tight_metrics.weighted_kl_loss == pytest.approx(tight_metrics.kl_penalty)
    # min(rho A, clip(rho) A) with A > 0 never exceeds the unclipped surrogate.
    assert tight_metrics.policy_loss >= loose_metrics.policy_loss - 1e-6
    assert float(tight_loss) == pytest.approx(
        tight_metrics.policy_loss + tight_metrics.weighted_kl_loss, abs=1e-6
    )
    assert float(loose_loss) == pytest.approx(loose_metrics.policy_loss, abs=1e-6)


# ------------------------------------------------------------ group-shared noise


def test_noise_is_shared_within_a_group_and_fresh_across_groups_and_updates() -> None:
    objective = VGRPO()
    x0 = torch.zeros(4, 2, 3)
    same_group = objective._group_shared_noise(
        x0, group_ids=torch.tensor([3, 3, 3, 3]), timestep_index=1
    )
    two_groups = objective._group_shared_noise(
        x0, group_ids=torch.tensor([3, 3, 4, 4]), timestep_index=1
    )

    assert torch.equal(same_group[0], same_group[3])
    assert torch.equal(two_groups[0], same_group[0])  # same (update, group, index)
    assert not torch.equal(two_groups[2], two_groups[0])
    other_index = objective._group_shared_noise(
        x0, group_ids=torch.tensor([3, 3, 3, 3]), timestep_index=2
    )
    assert not torch.equal(other_index[0], same_group[0])
    objective._update_counter += 1
    next_update = objective._group_shared_noise(
        x0, group_ids=torch.tensor([3, 3, 3, 3]), timestep_index=1
    )
    assert not torch.equal(next_update[0], same_group[0])


def test_after_optimizer_step_syncs_previous_and_advances_the_noise_counter() -> None:
    model = _build_model()
    named = dict(model.transformer.named_parameters())
    a_name = next(n for n in named if ".previous." in n and "lora_A" in n)
    d_name = a_name.replace(".previous.", ".default.")
    with torch.no_grad():
        named[d_name].add_(1.0)
    assert not torch.allclose(named[a_name], named[d_name])
    objective = VGRPO(VGRPOConfig(weight_copy_decay=0.0))

    objective.after_optimizer_step(model, global_step=7)

    assert torch.allclose(named[a_name], named[d_name])
    assert objective._update_counter == 8


# ------------------------------------------------------------ lr=0 invariant


def test_first_step_invariant_holds_when_previous_is_synced() -> None:
    """Advantage antisymmetry: ``loss(A) == -loss(-A)`` at ratio 1 with no KL."""
    _, _, batch = _batch()
    record = VGRPO(VGRPOConfig(adv_soft_clip=3.0)).first_step_invariant_check(
        model=_synced_model(), batch=batch, advantages=torch.tensor([2.0]), timestep_index=0
    )

    assert record["event"] == "first_step_v_grpo_invariant"
    assert record["passed"] is True
    assert record["loss"] == pytest.approx(-record["flipped_loss"], abs=1e-6)


# ------------------------------------------------------------ guards


def test_timestep_index_outside_the_trajectory_is_rejected() -> None:
    _, _, batch = _batch(timestep=(250.0, 500.0))
    with pytest.raises(RuntimeError, match="timestep_index out of range"):
        VGRPO().compute_batch_timestep_loss(_synced_model(), batch, 2, torch.ones(_BATCH))


def test_edm_scale_timestep_grid_fails_loudly() -> None:
    _, _, batch = _batch(timestep=80000.0)
    with pytest.raises(RuntimeError, match=r"normalize into \[0, 1\]"):
        VGRPO().compute_batch_timestep_loss(_synced_model(), batch, 0, torch.ones(_BATCH))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clip_ratio": 0.0},
        {"clip_ratio": 1.5},
        {"kl_coef": -0.1},
        {"adv_soft_clip": 0.0},
        {"weight_copy_decay": 2.0},
    ],
)
def test_config_rejects_out_of_range_controls(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        VGRPOConfig(**kwargs)


# ------------------------------------------------------------ stratified selection

_pick = OnlineTrainer._train_timestep_indices


def test_stratified_draws_one_index_per_equal_interval() -> None:
    torch.manual_seed(0)
    for _ in range(20):
        picks = _pick(20, 0.2, "stratified")
        assert len(picks) == 4
        assert all(lo <= pick < lo + 5 for pick, lo in zip(picks, (0, 5, 10, 15), strict=True))


def test_stratified_resamples_across_calls_and_covers_the_ends() -> None:
    torch.manual_seed(0)
    draws = [tuple(_pick(20, 0.2, "stratified")) for _ in range(40)]
    assert len(set(draws)) > 1
    assert any(d[0] == 0 for d in draws) and any(d[-1] == 19 for d in draws)


def test_stratified_handles_uneven_intervals_and_full_fraction() -> None:
    torch.manual_seed(0)
    picks = _pick(10, 0.3, "stratified")  # 3 intervals over 10 steps: [0,3) [3,7) [7,10)
    assert len(picks) == 3
    assert 0 <= picks[0] < 3 and 3 <= picks[1] < 7 and 7 <= picks[2] < 10
    assert _pick(10, 1.0, "stratified") == list(range(10))
