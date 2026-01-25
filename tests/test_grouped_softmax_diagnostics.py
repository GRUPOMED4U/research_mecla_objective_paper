import math
from types import SimpleNamespace

import torch
from transformers import TrainingArguments

from spesia_research.trainers import MultiLabelTokenTrainer


class DummyDataset:
    def __init__(self, labels):
        self.id2label = {i: name for i, name in enumerate(labels)}
        self.label2id = {name: i for i, name in self.id2label.items()}

    def __len__(self):
        return 1


class FixedLogitsModel(torch.nn.Module):
    def __init__(self, init_logits: torch.Tensor):
        super().__init__()
        self.logits_param = torch.nn.Parameter(init_logits.clone().detach())
        self.config = SimpleNamespace(id2label=None, label2id=None)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        B, L = input_ids.shape
        logits = self.logits_param
        if logits.shape[1] == 1 and L != 1:
            logits = logits.expand(1, L, -1)
        logits = logits.expand(B, -1, -1)
        return SimpleNamespace(logits=logits)


def _make_trainer(tmp_path, model, train_ds, loss_type, mutually_exclusive_classes, mecla_amplification_factor=1.0):
    args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        per_device_train_batch_size=2,
        report_to=[],
    )
    return MultiLabelTokenTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        loss_type=loss_type,
        mutually_exclusive_classes=mutually_exclusive_classes,
        mecla_amplification_factor=mecla_amplification_factor,
    )


def _trainer_device(trainer: MultiLabelTokenTrainer) -> torch.device:
    return next(trainer.model.parameters()).device


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def test_grouped_softmax_loss_matches_log3(tmp_path):
    labels_list = ["HER_POS", "HER_NEG"]
    ds = DummyDataset(labels_list)

    init_logits = torch.zeros((1, 1, 2), dtype=torch.float32)
    model = FixedLogitsModel(init_logits)

    trainer = _make_trainer(
        tmp_path, model, ds,
        loss_type="bce_with_grouped_softmax",
        mutually_exclusive_classes=[["HER_POS", "HER_NEG"]],
    )
    device = _trainer_device(trainer)

    B, L, C = 1, 1, 2
    batch = {
        "input_ids": torch.ones(B, L, dtype=torch.long),
        "attention_mask": torch.ones(B, L, dtype=torch.long),
        "labels": torch.zeros(B, L, C, dtype=torch.float32),
    }
    batch["labels"][0, 0, ds.label2id["HER_POS"]] = 1.0

    loss = trainer.compute_loss(model, _to_device(dict(batch), device))
    assert abs(loss.item() - math.log(3.0)) < 1e-6


def test_padding_mask_ignores_padded_token(tmp_path):
    labels_list = ["HER_POS", "HER_NEG"]
    ds = DummyDataset(labels_list)

    init_logits = torch.tensor([[[0.0, 0.0],
                                [10.0, -10.0]]], dtype=torch.float32)  # [1,2,2]
    model = FixedLogitsModel(init_logits)

    trainer = _make_trainer(
        tmp_path, model, ds,
        loss_type="bce_with_grouped_softmax",
        mutually_exclusive_classes=[["HER_POS", "HER_NEG"]],
    )
    device = _trainer_device(trainer)

    B, L, C = 1, 2, 2
    batch = {
        "input_ids": torch.ones(B, L, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 0]], dtype=torch.long),  # second token padded
        "labels": torch.zeros(B, L, C, dtype=torch.float32),         # token0=NONE
    }

    loss = trainer.compute_loss(model, _to_device(dict(batch), device))
    assert abs(loss.item() - math.log(3.0)) < 1e-6


def test_grouped_softmax_gradient_signs(tmp_path):
    labels_list = ["HER_POS", "HER_NEG"]
    ds = DummyDataset(labels_list)

    init_logits = torch.tensor([[[0.2, -0.1]]], dtype=torch.float32)  # [1,1,2]
    model = FixedLogitsModel(init_logits)

    trainer = _make_trainer(
        tmp_path, model, ds,
        loss_type="bce_with_grouped_softmax",
        mutually_exclusive_classes=[["HER_POS", "HER_NEG"]],
    )
    device = _trainer_device(trainer)

    B, L, C = 1, 1, 2
    batch = {
        "input_ids": torch.ones(B, L, dtype=torch.long),
        "attention_mask": torch.ones(B, L, dtype=torch.long),
        "labels": torch.zeros(B, L, C, dtype=torch.float32),
    }
    batch["labels"][0, 0, ds.label2id["HER_POS"]] = 1.0

    loss = trainer.compute_loss(model, _to_device(dict(batch), device))
    loss.backward()

    g_pos = model.logits_param.grad[0, 0, ds.label2id["HER_POS"]].item()
    g_neg = model.logits_param.grad[0, 0, ds.label2id["HER_NEG"]].item()

    assert g_pos < 0.0
    assert g_neg > 0.0


def test_contradiction_is_ignored_no_nan(tmp_path):
    labels_list = ["HER_POS", "HER_NEG"]
    ds = DummyDataset(labels_list)

    init_logits = torch.tensor([[[3.0, 3.0]]], dtype=torch.float32)
    model = FixedLogitsModel(init_logits)

    trainer = _make_trainer(
        tmp_path, model, ds,
        loss_type="bce_with_grouped_softmax",
        mutually_exclusive_classes=[["HER_POS", "HER_NEG"]],
    )
    device = _trainer_device(trainer)

    B, L, C = 1, 1, 2
    batch = {
        "input_ids": torch.ones(B, L, dtype=torch.long),
        "attention_mask": torch.ones(B, L, dtype=torch.long),
        "labels": torch.zeros(B, L, C, dtype=torch.float32),
    }
    batch["labels"][0, 0, ds.label2id["HER_POS"]] = 1.0
    batch["labels"][0, 0, ds.label2id["HER_NEG"]] = 1.0  # contradiction -> ignored

    loss = trainer.compute_loss(model, _to_device(dict(batch), device))
    assert torch.isfinite(loss).item()
    assert abs(loss.item() - 0.0) < 1e-8


def test_bce_vs_grouped_softmax_all_zero_case(tmp_path):
    labels_list = ["HER_POS", "HER_NEG"]
    ds = DummyDataset(labels_list)

    init_logits = torch.zeros((1, 1, 2), dtype=torch.float32)
    model_bce = FixedLogitsModel(init_logits)
    model_grp = FixedLogitsModel(init_logits)

    trainer_bce = _make_trainer(
        tmp_path, model_bce, ds,
        loss_type="bce",
        mutually_exclusive_classes=[["HER_POS", "HER_NEG"]],
    )
    trainer_grp = _make_trainer(
        tmp_path, model_grp, ds,
        loss_type="bce_with_grouped_softmax",
        mutually_exclusive_classes=[["HER_POS", "HER_NEG"]],
    )

    B, L, C = 1, 1, 2
    base_batch = {
        "input_ids": torch.ones(B, L, dtype=torch.long),
        "attention_mask": torch.ones(B, L, dtype=torch.long),
        "labels": torch.zeros(B, L, C, dtype=torch.float32),  # NONE
    }

    loss_bce = trainer_bce.compute_loss(model_bce, _to_device(dict(base_batch), _trainer_device(trainer_bce))).item()
    loss_grp = trainer_grp.compute_loss(model_grp, _to_device(dict(base_batch), _trainer_device(trainer_grp))).item()

    # BCE sums across 2 labels, each has CE for negative at logit 0 => log(2)
    assert abs(loss_bce - (2.0 * math.log(2.0))) < 1e-6
    # Grouped softmax CE on NONE with [0,0,0] => log(3)
    assert abs(loss_grp - math.log(3.0)) < 1e-6


def test_mixed_nonexclusive_bce_plus_group_ce(tmp_path):
    labels_list = ["HER_POS", "HER_NEG", "X"]
    ds = DummyDataset(labels_list)

    init_logits = torch.zeros((1, 1, 3), dtype=torch.float32)
    model = FixedLogitsModel(init_logits)

    trainer = _make_trainer(
        tmp_path, model, ds,
        loss_type="bce_with_grouped_softmax",
        mutually_exclusive_classes=[["HER_POS", "HER_NEG"]],
        mecla_amplification_factor=1.0,
    )
    device = _trainer_device(trainer)

    B, L, C = 1, 1, 3
    batch = {
        "input_ids": torch.ones(B, L, dtype=torch.long),
        "attention_mask": torch.ones(B, L, dtype=torch.long),
        "labels": torch.zeros(B, L, C, dtype=torch.float32),
    }
    batch["labels"][0, 0, ds.label2id["X"]] = 1.0  # non-exclusive positive

    loss = trainer.compute_loss(model, _to_device(dict(batch), device)).item()

    # Group CE on NONE: log(3); BCE for X positive at logit 0: log(2)
    expected = math.log(3.0) + math.log(2.0)
    assert abs(loss - expected) < 1e-6
