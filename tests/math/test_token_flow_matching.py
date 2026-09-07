"""Velocity-contract + sample/replay consistency for NextStep-1 flow-matching.

Guards the contract verified against upstream ``FlowMatchingHead``: the velocity
predictor is ``image_head.net(x, t, c)`` and the head's ``forward`` is the
training-loss path, NOT velocity. ``_FakeHead.__call__`` raises, so any code that
reaches for ``image_head(...)`` instead of ``.net`` fails loudly here.
"""

from __future__ import annotations

import torch

from vrl.math.token.flow_matching import (
    _flow_terminal_mean,
    flow_logprob_at,
    flow_sample_with_logprob,
)


class _FakeHead:
    """Minimal stand-in for NextStep-1 FlowMatchingHead.

    Exposes ``.net(x, t, c)`` (a deterministic linear velocity) and
    ``input_dim``; calling the head directly raises so the logprob paths must
    use ``.net``.
    """

    def __init__(self, input_dim: int) -> None:
        self.input_dim = input_dim

    def net(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Deterministic velocity field of the right shape ([B, input_dim]).
        return 0.1 * x

    def __call__(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError(
            "flow-matching logprob must call image_head.net, not the head's "
            "forward (which is the training-loss path)"
        )


class _RecordingHead:
    """Identity-velocity head that records the conditioning it evaluated.

    ``.net(x, t, c)`` returns ``c`` so ``_flow_terminal_mean`` with num_steps=1
    yields the (guided) velocity directly, letting the CFG-combine tests assert
    which conditioning tensors the walk evaluated. Calling the head directly
    raises, mirroring ``_FakeHead``'s guard that logprob paths use ``.net``.
    """

    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def net(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        del x, t
        self.calls.append(c)
        return c

    def __call__(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError(
            "flow-matching logprob must call image_head.net, not the head's "
            "forward (which is the training-loss path)"
        )


def test_cfg_scale_one_uses_only_the_conditional_velocity() -> None:
    """CFG scale one is the no-guidance branch and must not evaluate uncond."""
    cond = torch.full((1, 2), 3.0)
    uncond = torch.full((1, 2), 1.0)
    head = _RecordingHead()

    mean = _flow_terminal_mean(
        image_head=head,
        cond=cond,
        x=torch.zeros_like(cond),
        num_steps=1,
        cfg_uncond=uncond,
        guidance_scale=1.0,
    )

    assert torch.equal(mean, cond)
    assert head.calls == [cond]


def test_cfg_scale_zero_does_not_enable_the_unconditional_branch() -> None:
    cond = torch.full((1, 2), 3.0)
    uncond = torch.full((1, 2), 1.0)
    head = _RecordingHead()

    mean = _flow_terminal_mean(
        image_head=head,
        cond=cond,
        x=torch.zeros_like(cond),
        num_steps=1,
        cfg_uncond=uncond,
        guidance_scale=0.0,
    )

    assert torch.equal(mean, cond)
    assert head.calls == [cond]


def test_cfg_scale_two_matches_upstream_unconditional_base_formula() -> None:
    """CFG uses uncond + scale * (cond - uncond), not an extra cond copy."""
    cond = torch.full((1, 2), 3.0)
    uncond = torch.full((1, 2), 1.0)

    mean = _flow_terminal_mean(
        image_head=_RecordingHead(),
        cond=cond,
        x=torch.zeros_like(cond),
        num_steps=1,
        cfg_uncond=uncond,
        guidance_scale=2.0,
    )

    assert torch.equal(mean, torch.full_like(cond, 5.0))


def test_sample_and_replay_go_through_net_and_agree() -> None:
    """Sampling through the head and replaying the sampled token with the saved prior reproduce
    the collection-time log-prob exactly: the GRPO ratio == 1 invariant.
    """
    b, token_dim, hidden = 2, 4, 6
    head = _FakeHead(token_dim)
    cond = torch.randn(b, hidden)
    gen = torch.Generator().manual_seed(0)

    token, log_prob, x0 = flow_sample_with_logprob(head, cond, num_steps=5, generator=gen)

    assert token.shape == (b, token_dim)
    assert log_prob.shape == (b,)
    assert x0.shape == (b, token_dim)
    assert torch.isfinite(log_prob).all()

    # Replaying the same token with the saved prior and the same (unchanged)
    # velocity field must reproduce the exact collection-time log-prob — this is
    # the GRPO ratio == 1 invariant the divergent velocity resolver would break.
    replay_log_prob = flow_logprob_at(head, cond, token, saved_noise=x0, num_steps=5)

    assert replay_log_prob.shape == (b,)
    assert torch.isfinite(replay_log_prob).all()
    assert torch.allclose(log_prob, replay_log_prob, atol=1e-5)
