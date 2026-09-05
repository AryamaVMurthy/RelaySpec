from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_proposal_kl_is_zero_for_identical_logits() -> None:
    from relayspec.losses import proposal_kl_loss

    logits = torch.tensor([[[1.0, 2.0, -1.0], [0.0, 0.5, 1.0]]])

    assert proposal_kl_loss(logits, logits).item() == pytest.approx(0.0, abs=1e-7)


def test_proposal_kl_backpropagates_only_through_student() -> None:
    from relayspec.losses import proposal_kl_loss

    student = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    teacher = torch.tensor([[[0.0, 1.0]]], requires_grad=True)
    loss = proposal_kl_loss(student, teacher)
    loss.backward()

    assert student.grad is not None
    assert teacher.grad is None


def test_relative_interface_mse_is_zero_for_identical_context() -> None:
    from relayspec.losses import relative_interface_mse

    context = torch.randn(2, 3, 5)

    assert relative_interface_mse(context, context).item() == pytest.approx(0.0)


def test_relative_interface_mse_matches_radial_angular_decomposition() -> None:
    from relayspec.losses import relative_interface_mse

    target = torch.tensor([[[3.0, 4.0]]])
    prediction = torch.tensor([[[0.0, 10.0]]])
    rho = prediction.norm() / target.norm()
    cosine = torch.nn.functional.cosine_similarity(prediction, target, dim=-1).item()
    expected = rho.item() ** 2 + 1 - 2 * rho.item() * cosine

    assert relative_interface_mse(prediction, target).item() == pytest.approx(expected)


def test_relative_interface_mse_uses_detached_float32_target() -> None:
    from relayspec.losses import relative_interface_mse

    prediction = torch.ones((1, 1, 2), dtype=torch.bfloat16, requires_grad=True)
    target = torch.zeros((1, 1, 2), dtype=torch.bfloat16, requires_grad=True)
    loss = relative_interface_mse(prediction, target)
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert target.grad is None


def test_interface_alignment_loss_requires_explicit_historical_weight() -> None:
    from relayspec.losses import interface_alignment_loss

    prediction = torch.ones((1, 1, 2))
    target = torch.zeros((1, 1, 2))

    with pytest.raises(ValueError, match="historical_cosine_weight"):
        interface_alignment_loss(
            prediction,
            target,
            objective="historical_mse_cosine",
        )


def test_interface_alignment_loss_selects_relative_objective() -> None:
    from relayspec.losses import interface_alignment_loss, relative_interface_mse

    prediction = torch.tensor([[[1.0, 2.0]]])
    target = torch.tensor([[[2.0, 1.0]]])

    assert interface_alignment_loss(
        prediction,
        target,
        objective="relative_interface_mse",
    ) == pytest.approx(relative_interface_mse(prediction, target).item())


def test_greedy_agreement_ce_is_zero_when_proposer_is_certain() -> None:
    from relayspec.losses import greedy_agreement_ce

    proposal_logits = torch.tensor([[[0.0, 60.0], [60.0, 0.0]]])
    accepted_tokens = torch.tensor([[1, 0]])

    loss = greedy_agreement_ce(proposal_logits, accepted_tokens)

    assert loss.item() == pytest.approx(0.0, abs=1e-9)


def test_greedy_agreement_ce_penalizes_disagreement() -> None:
    from relayspec.losses import greedy_agreement_ce

    logits = torch.tensor([[[0.0, 5.0]]])
    agreeing = greedy_agreement_ce(logits, torch.tensor([[1]]))
    disagreeing = greedy_agreement_ce(logits, torch.tensor([[0]]))

    assert disagreeing.item() > agreeing.item()


def test_greedy_agreement_ce_backpropagates_only_through_proposer() -> None:
    from relayspec.losses import greedy_agreement_ce

    proposal_logits = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    accepted_tokens = torch.tensor([[0]])
    greedy_agreement_ce(proposal_logits, accepted_tokens).backward()

    assert proposal_logits.grad is not None


def test_greedy_agreement_ce_rejects_mismatched_shapes() -> None:
    from relayspec.losses import greedy_agreement_ce

    with pytest.raises(ValueError, match="batch and positions"):
        greedy_agreement_ce(
            torch.zeros((1, 2, 3)), torch.zeros((1, 3), dtype=torch.long)
        )


def test_expected_accepted_length_recovers_full_block_when_certain() -> None:
    from relayspec.losses import expected_accepted_length_surrogate

    proposal_logits = torch.tensor([[[0.0, 60.0], [60.0, 0.0], [0.0, 60.0]]])
    accepted_tokens = torch.tensor([[1, 0, 1]])

    loss = expected_accepted_length_surrogate(proposal_logits, accepted_tokens)

    assert loss.item() == pytest.approx(-1.0, abs=1e-6)


def test_expected_accepted_length_matches_prefix_product_definition() -> None:
    from relayspec.losses import expected_accepted_length_surrogate

    proposal_logits = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    accepted_tokens = torch.tensor([[0, 1]])
    # Each position agrees with probability 0.5, so the expected accepted
    # prefix length is 0.5 + 0.25 = 0.75 over two positions.
    expected = -(0.5 + 0.25) / 2

    loss = expected_accepted_length_surrogate(proposal_logits, accepted_tokens)

    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_expected_accepted_length_discounts_later_positions() -> None:
    from relayspec.losses import expected_accepted_length_surrogate

    confident_first = torch.tensor([[[0.0, 5.0], [0.0, 0.0]]])
    confident_second = torch.tensor([[[0.0, 0.0], [0.0, 5.0]]])
    tokens = torch.tensor([[1, 1]])

    early = expected_accepted_length_surrogate(confident_first, tokens)
    late = expected_accepted_length_surrogate(confident_second, tokens)

    # Agreement at an early position unlocks later positions, so it is worth
    # strictly more than the same agreement placed later in the block.
    assert early.item() < late.item()


def test_hard_accepted_prefix_matches_leading_argmax_run() -> None:
    from relayspec.losses import hard_accepted_prefix_diagnostic

    # Position 0 agrees (argmax=1), position 1 disagrees (argmax=0, want 1),
    # position 2 would agree but is unreachable, so hard length is 1.
    proposal_logits = torch.tensor([[[0.0, 5.0], [5.0, 0.0], [0.0, 5.0]]])
    accepted_tokens = torch.tensor([[1, 1, 1]])

    diagnostic = hard_accepted_prefix_diagnostic(proposal_logits, accepted_tokens)

    assert diagnostic["hard_accepted_length"].item() == pytest.approx(1.0)
    assert diagnostic["hard_accepted_indicator"].tolist() == [[True, False, True]]


def test_hard_accepted_prefix_is_full_length_when_all_agree() -> None:
    from relayspec.losses import hard_accepted_prefix_diagnostic

    proposal_logits = torch.tensor([[[0.0, 5.0], [5.0, 0.0]]])
    accepted_tokens = torch.tensor([[1, 0]])

    diagnostic = hard_accepted_prefix_diagnostic(proposal_logits, accepted_tokens)

    assert diagnostic["hard_accepted_length"].item() == pytest.approx(2.0)


def test_hard_accepted_prefix_diverges_from_soft_surrogate_on_a_near_tie() -> None:
    """Pins the documented non-equivalence between A_soft and A_hard.

    With three or more classes, the argmax can hold a minority of the
    softmax mass (a_1 < 0.5) while still being the accepted, hard-agreeing
    token, so A_hard credits a position that A_soft discounts. A two-class
    softmax cannot exhibit this, since its argmax always carries at least
    half the mass, which is exactly why the general inequality fails only
    once the vocabulary is large enough, as it always is in practice.
    """
    from relayspec.losses import (
        expected_accepted_length_surrogate,
        hard_accepted_prefix_diagnostic,
    )

    # Position 0 is a clear win for token 0. Position 1 is a three-way near
    # tie where token 0 is still the argmax but holds under half the mass.
    proposal_logits = torch.tensor([[[5.0, 0.0, 0.0], [1.0, 0.9, 0.8]]])
    accepted_tokens = torch.tensor([[0, 0]])

    diagnostic = hard_accepted_prefix_diagnostic(proposal_logits, accepted_tokens)
    positions = proposal_logits.shape[1]
    soft_length = (
        -expected_accepted_length_surrogate(proposal_logits, accepted_tokens)
        * positions
    )

    assert diagnostic["agreement_mass"][0, 1].item() < 0.5
    assert diagnostic["hard_accepted_length"].item() == pytest.approx(2.0)
    assert soft_length.item() < diagnostic["hard_accepted_length"].item()


def test_hard_accepted_prefix_requires_no_grad() -> None:
    from relayspec.losses import hard_accepted_prefix_diagnostic

    proposal_logits = torch.tensor([[[0.0, 5.0]]], requires_grad=True)
    accepted_tokens = torch.tensor([[1]])

    diagnostic = hard_accepted_prefix_diagnostic(proposal_logits, accepted_tokens)

    assert not diagnostic["hard_accepted_length"].requires_grad
    assert not diagnostic["agreement_mass"].requires_grad


def test_scale_preserving_relay_does_not_normalize_its_input() -> None:
    from relayspec.relay import TargetFeatureRelay

    relay = TargetFeatureRelay(
        target_hidden_size=2,
        num_taps=1,
        draft_hidden_size=2,
        eps=1e-6,
        normalize_input=False,
    )
    with torch.no_grad():
        relay.projection.weight.copy_(torch.eye(2))
    features = torch.tensor([[[3.0, 4.0]]])

    assert torch.equal(relay(features), features)
