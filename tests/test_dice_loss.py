import pytest
import torch


from spesia_research.loss import MultilabelDiceLoss


def test_perfect_prediction_has_near_zero_loss():
    loss_fn = MultilabelDiceLoss()

    logits = torch.tensor(
        [[[10.0, -10.0], [-10.0, 10.0]]], dtype=torch.float32
    )  # [1, 2, 2]
    labels = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)

    loss = loss_fn(logits, labels)

    assert torch.isfinite(loss)
    assert loss.item() < 1e-3


def test_completely_wrong_prediction_has_near_one_loss():
    loss_fn = MultilabelDiceLoss()

    logits = torch.tensor(
        [[[-10.0, 10.0], [10.0, -10.0]]], dtype=torch.float32
    )  # opposite of labels
    labels = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)

    loss = loss_fn(logits, labels)

    assert torch.isfinite(loss)
    assert loss.item() > 0.99


def test_empty_prediction_and_empty_labels_give_zero_loss():
    loss_fn = MultilabelDiceLoss()

    logits = torch.full((1, 3, 2), -100.0, dtype=torch.float32)
    labels = torch.zeros((1, 3, 2), dtype=torch.float32)

    loss = loss_fn(logits, labels)

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)


def test_attention_mask_2d_ignores_padding_tokens():
    loss_fn = MultilabelDiceLoss()

    logits = torch.tensor([[[10.0], [10.0], [-10.0]]], dtype=torch.float32)  # [1, 3, 1]
    labels = torch.tensor([[[1.0], [1.0], [1.0]]], dtype=torch.float32)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.float32)

    masked_loss = loss_fn(logits, labels, mask)
    unmasked_loss = loss_fn(logits, labels)

    assert torch.isfinite(masked_loss)
    assert torch.isfinite(unmasked_loss)
    assert masked_loss.item() < unmasked_loss.item()
    assert masked_loss.item() < 1e-3


def test_attention_mask_3d_matches_equivalent_2d_mask():
    loss_fn = MultilabelDiceLoss()

    logits = torch.tensor(
        [[[2.0, -2.0], [1.0, -1.0], [3.0, -3.0]]], dtype=torch.float32
    )  # [1, 3, 2]
    labels = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
    mask_2d = torch.tensor([[1, 1, 0]], dtype=torch.float32)
    mask_3d = mask_2d.unsqueeze(-1).expand_as(logits)

    loss_2d = loss_fn(logits, labels, mask_2d)
    loss_3d = loss_fn(logits, labels, mask_3d)

    assert torch.isfinite(loss_2d)
    assert torch.isfinite(loss_3d)
    assert loss_2d.item() == pytest.approx(loss_3d.item(), rel=1e-6, abs=1e-8)


def test_manual_numeric_example_matches_expected_value():
    loss_fn = MultilabelDiceLoss(epsilon=1e-10)

    logits = torch.tensor([[[0.0], [0.0]]], dtype=torch.float32)  # sigmoid=0.5
    labels = torch.tensor([[[1.0], [0.0]]], dtype=torch.float32)

    loss = loss_fn(logits, labels)

    # Single sample, single class:
    # intersection = 0.5
    # denominator = (0.5 + 0.5) + (1 + 0) = 2.0
    # dice = 1.0 / 2.0 = 0.5
    # loss = 0.5
    expected = 0.5

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_weighted_loss_uses_normalized_class_average():
    weights = torch.tensor([1.0, 3.0], dtype=torch.float32)
    loss_fn = MultilabelDiceLoss(pos_weight=weights, from_logits=False)

    probs = torch.tensor([[[0.9, 0.2], [0.1, 0.8]]], dtype=torch.float32)  # [1, 2, 2]
    labels = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)

    loss = loss_fn(probs, labels)

    expected = 0.35

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_weighted_loss_does_not_go_negative_for_perfect_predictions():
    weights = torch.tensor([1.0, 5.0], dtype=torch.float32)
    loss_fn = MultilabelDiceLoss(pos_weight=weights)

    logits = torch.tensor([[[10.0, -10.0], [-10.0, 10.0]]], dtype=torch.float32)
    labels = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)

    loss = loss_fn(logits, labels)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    assert loss.item() < 1e-3


def test_batch_reduction_is_mean_of_per_sample_losses():
    loss_fn = MultilabelDiceLoss(from_logits=False)

    probs = torch.tensor(
        [
            [[1.0], [0.0]],  # perfect -> loss 0
            [[0.5], [0.5]],  # one positive, one negative -> dice 0.5 -> loss 0.5
        ],
        dtype=torch.float32,
    )  # [2, 2, 1]

    labels = torch.tensor(
        [
            [[1.0], [0.0]],
            [[1.0], [0.0]],
        ],
        dtype=torch.float32,
    )

    loss = loss_fn(probs, labels)

    expected = (0.0 + 0.5) / 2.0
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_raises_on_label_shape_mismatch():
    loss_fn = MultilabelDiceLoss()
    logits = torch.zeros((2, 4, 3), dtype=torch.float32)
    labels = torch.zeros((2, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="labels must have same shape as logits"):
        loss_fn(logits, labels)


def test_raises_on_invalid_2d_attention_mask_shape():
    loss_fn = MultilabelDiceLoss()
    logits = torch.zeros((2, 4, 3), dtype=torch.float32)
    labels = torch.zeros((2, 4, 3), dtype=torch.float32)
    bad_mask = torch.ones((2, 5), dtype=torch.float32)

    with pytest.raises(
        ValueError, match="2D attention_mask must have shape \\[B, T\\]"
    ):
        loss_fn(logits, labels, bad_mask)


def test_raises_on_invalid_3d_attention_mask_shape():
    loss_fn = MultilabelDiceLoss()
    logits = torch.zeros((2, 4, 3), dtype=torch.float32)
    labels = torch.zeros((2, 4, 3), dtype=torch.float32)
    bad_mask = torch.ones((2, 4, 2), dtype=torch.float32)

    with pytest.raises(
        ValueError, match="3D attention_mask must have shape \\[B, T, C\\]"
    ):
        loss_fn(logits, labels, bad_mask)


def test_raises_on_invalid_pos_weight_shape():
    loss_fn = MultilabelDiceLoss(
        pos_weight=torch.tensor([1.0, 2.0], dtype=torch.float32)
    )
    logits = torch.zeros((2, 4, 3), dtype=torch.float32)
    labels = torch.zeros((2, 4, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="pos_weight must have shape \\[C\\]"):
        loss_fn(logits, labels)
