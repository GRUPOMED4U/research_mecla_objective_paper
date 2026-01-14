import torch
import pytest

from spesia_research.loss import PairwiseMECLALoss  # <-- adjust import


def _example_logits_labels():
    """
    [B=1, L=6, C=2]
    Span for class 0 at tokens 1..3
    """
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 2)
    labels = torch.zeros(1, 6, 2)
    labels[0, 1:4, 0] = 1
    return logits, labels


def test_pairwise_init_requires_pairs():
    # Not a pair -> should raise at init
    with pytest.raises(AssertionError):
        PairwiseMECLALoss(mutually_exclusive_classes_indices=[(0, 1, 2)])


def test_pairwise_forward_runs_and_shape_matches():
    logits, labels = _example_logits_labels()
    loss_fn = PairwiseMECLALoss(
        mutually_exclusive_classes_indices=[(0, 1)], mecla_amplification_factor=0.5
    )

    out = loss_fn(logits, labels)
    assert out.shape == logits.shape
    assert torch.isfinite(out).all()


def test_pairwise_boundary_penalty_hits_expected_tokens():
    """
    With apply_to_starts/ends enabled, the added boundary penalties should increase
    loss at:
      - token before start (0)
      - start token (1)
      - end token (3)
      - token after end (4)
    The middle token (2) should be unchanged by boundary penalties.
    """
    logits, labels = _example_logits_labels()
    loss_fn = PairwiseMECLALoss(
        mutually_exclusive_classes_indices=[(0, 1)], mecla_amplification_factor=0.5
    )

    base = loss_fn(
        logits, labels, apply_to_starts=False, apply_to_ends=False, token_dim=1
    )
    bumped = loss_fn(
        logits, labels, apply_to_starts=True, apply_to_ends=True, token_dim=1
    )

    for t in [0, 1, 3, 4]:
        assert (bumped[0, t, 0] > base[0, t, 0]).item()

    # token 2 is not start/end/before/after -> boundary penalty should not change it
    torch.testing.assert_close(bumped[0, 2, 0], base[0, 2, 0], rtol=0, atol=0)


def test_pairwise_no_crash_all_zero_labels():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 3)
    labels = torch.zeros_like(logits)

    loss_fn = PairwiseMECLALoss(
        mutually_exclusive_classes_indices=[(0, 1)], mecla_amplification_factor=0.5
    )

    out = loss_fn(logits, labels, apply_to_starts=True, apply_to_ends=True, token_dim=1)
    assert out.shape == logits.shape
    assert torch.isfinite(out).all()
