import torch
from types import SimpleNamespace
from transformers import TrainingArguments

from spesia_research.trainers import MultiLabelTokenTrainer

class DummyDataset:
    # Trainer init needs these
    id2label = {0: "HER_POS", 1: "HER_NEG", 2: "X", 3: "Y"}
    label2id = {v: k for k, v in id2label.items()}

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        raise IndexError

class DummyModel(torch.nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.config = SimpleNamespace(id2label=None, label2id=None)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        B, L = input_ids.shape
        logits = torch.randn(B, L, self.num_labels, requires_grad=True)
        return SimpleNamespace(logits=logits)

def test_grouped_softmax_loss_runs_and_backprops():
    torch.manual_seed(0)

    num_labels = 4
    model = DummyModel(num_labels=num_labels)
    train_ds = DummyDataset()

    args = TrainingArguments(
        output_dir="tmp_test",
        per_device_train_batch_size=2,
        report_to=[],
    )

    trainer = MultiLabelTokenTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        loss_type="bce_with_grouped_softmax",
        mutually_exclusive_classes=[["HER_POS", "HER_NEG"]],
        mecla_amplification_factor=1.0,
    )

    B, L, C = 2, 6, num_labels
    input_ids = torch.ones(B, L, dtype=torch.long)
    attention_mask = torch.tensor([[1,1,1,1,0,0],[1,1,1,1,1,0]], dtype=torch.long)

    labels = torch.zeros(B, L, C)
    # a few positives
    labels[0, 1, train_ds.label2id["HER_POS"]] = 1
    labels[1, 2, train_ds.label2id["HER_NEG"]] = 1
    labels[1, 3, train_ds.label2id["X"]] = 1
    # inject a contradiction (both 1) -> should be ignored, not crash
    labels[0, 2, train_ds.label2id["HER_POS"]] = 1
    labels[0, 2, train_ds.label2id["HER_NEG"]] = 1

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    loss = trainer.compute_loss(model, inputs)
    assert torch.isfinite(loss).item()
    loss.backward()  # must not error
