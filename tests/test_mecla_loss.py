import torch
import pytest

from spesia_research.loss import MECLALoss
# adjust import path ↑


# -------------------------
# Helpers
# -------------------------


def make_logits_and_labels():
    """
    Create a simple [B=1, L=6, C=2] example with a single span
    for class 0 from tokens 1..3.
    """
    torch.manual_seed(0)

    logits = torch.randn(1, 6, 2)
    labels = torch.zeros(1, 6, 2)

    # class 0 active on tokens [1, 2, 3]
    labels[0, 1:4, 0] = 1

    return logits, labels


# -------------------------
# Construction
# -------------------------


def test_init_requires_mutually_exclusive_classes():
    with pytest.raises(AssertionError):
        MECLALoss(mutually_exclusive_classes_indices=None)


def test_init_success():
    loss_fn = MECLALoss(mutually_exclusive_classes_indices=[(0, 1)])
    assert loss_fn.mecla_amplification_factor == 1.0


# -------------------------
# Mask correctness
# -------------------------


def test_starts_mask_detects_first_active_token():
    _, labels = make_logits_and_labels()

    starts = MECLALoss.get_starts_mask(labels, dim=1)
    idx = starts.nonzero(as_tuple=False).tolist()

    # start at token 1 for class 0
    assert idx == [[0, 1, 0]]


def test_ends_mask_detects_last_active_token():
    _, labels = make_logits_and_labels()

    ends = MECLALoss.get_ends_mask(labels, dim=1)
    idx = ends.nonzero(as_tuple=False).tolist()

    # end at token 3 for class 0
    assert idx == [[0, 3, 0]]


def test_single_token_span_is_both_start_and_end():
    labels = torch.zeros(1, 5, 1)
    labels[0, 2, 0] = 1

    starts = MECLALoss.get_starts_mask(labels, dim=1)
    ends = MECLALoss.get_ends_mask(labels, dim=1)

    assert starts.nonzero().tolist() == [[0, 2, 0]]
    assert ends.nonzero().tolist() == [[0, 2, 0]]


# -------------------------
# Boolean shift correctness
# -------------------------


def test_shift_tensor_before_and_after():
    mask = torch.zeros(1, 5, 1, dtype=torch.bool)
    mask[0, 2, 0] = True  # marker at token 2

    before = MECLALoss._shift_tensor(mask, dim=1, direction=-1)
    after = MECLALoss._shift_tensor(mask, dim=1, direction=+1)

    assert before.nonzero().tolist() == [[0, 1, 0]]
    assert after.nonzero().tolist() == [[0, 3, 0]]


# -------------------------
# Loss behavior
# -------------------------


def test_mecla_label_amplification_increases_loss():
    logits, labels = make_logits_and_labels()

    base_fn = MECLALoss(
        mutually_exclusive_classes_indices=[(0, 1)],
        mecla_amplification_factor=0.0,
    )
    mecla_fn = MECLALoss(
        mutually_exclusive_classes_indices=[(0, 1)],
        mecla_amplification_factor=1.0,
    )

    base_loss = base_fn(logits, labels)
    mecla_loss = mecla_fn(logits, labels)

    # MECLA should strictly increase loss for class 0
    assert torch.all(mecla_loss[..., 0] > base_loss[..., 0])


def test_start_and_end_amplification_hits_correct_tokens():
    logits, labels = make_logits_and_labels()

    loss_fn = MECLALoss(
        mutually_exclusive_classes_indices=[(0, 1)],
        mecla_amplification_factor=1.0,
    )

    base = loss_fn(
        logits,
        labels,
        apply_to_starts=False,
        apply_to_ends=False,
        token_dim=1,
    )

    bumped = loss_fn(
        logits,
        labels,
        apply_to_starts=True,
        apply_to_ends=True,
        token_dim=1,
    )

    # Tokens affected for class 0:
    # before-start: 0
    # start:        1
    # end:          3
    # after-end:    4
    affected_tokens = [0, 1, 3, 4]

    for t in affected_tokens:
        assert bumped[0, t, 0] > base[0, t, 0]

    # Middle token (inside span but not boundary) should not increase
    # only due to MECLA, not boundary logic
    assert torch.equal(bumped[0, 2, 0], base[0, 2, 0])


# -------------------------
# Safety checks
# -------------------------


def test_no_crash_on_all_zero_labels():
    logits = torch.randn(1, 4, 3)
    labels = torch.zeros_like(logits)

    loss_fn = MECLALoss(mutually_exclusive_classes_indices=[(0, 1)])

    loss = loss_fn(
        logits,
        labels,
        apply_to_starts=True,
        apply_to_ends=True,
    )

    assert torch.isfinite(loss).all()
