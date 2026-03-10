from __future__ import annotations
from pathlib import Path
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
from tqdm import tqdm

from typing import Dict, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt


from spesia_research.datasets import (
    ClinicalRecordsDataset,
    DataCollatorForMultiLabelTokenClassification,
)

random_seed = 3
batch_size = 10

experiments = [
    "baseline_mmbert",
    "bce_with_mecla",
    "bce_with_pairwise_mecla",
    "bce_with_grouped_softmax",
    "bce_with_grouped_softmax_as_penalty",
]

rename_experiments = {
    "baseline_mmbert": "Baseline",
    "bce_with_mecla": "BCE with MECLA",
    "bce_with_pairwise_mecla": "BCE with Pairwise MECLA",
    "bce_with_grouped_softmax": "BCE with Grouped Softmax",
    "bce_with_grouped_softmax_as_penalty": "BCE with Grouped Softmax as Penalty",
}

rename_labels = {
    "BRCA_NEGATIVO": "Negative BRCA",
    "BRCA_POSITIVO": "Positive BRCA",
    "CIRURGIA": "Surgery",
    "HER2_NEGATIVO": "Negative HER2",
    "HER2_POSITIVO": "Positive HER2",
    "POS_MENOPAUSA": "Post-Menopause",
    "PRE_MENOPAUSA": "Pre-Menopause",
    "RE_NEGATIVO": "Negative ER",
    "RE_POSITIVO": "Positive ER",
    "RP_NEGATIVO": "Negative PR",
    "RP_POSITIVO": "Positive PR",
    "TIPO_HISTOPATOLOGICO": "Histopathological type",
}

Id2Label = Union[Sequence[str], Dict[int, str]]


# plot confusion matrices
def plot_mlcm(
    preds: torch.Tensor,
    labels: torch.Tensor,
    id2label: Optional[Id2Label] = None,
    *,
    normalize_rows: bool = False,
    annotate: bool = True,
    figsize=(10, 8),
    title: str = "MLCM (Multi-Label Confusion Matrix)",
    output_path: Optional[Path] = None,
):
    """
    MLCM implementation based on:
    Heydarian et al., "MLCM: Multi-Label Confusion Matrix" (IEEE Access, 2022).

    Expected inputs:
    - preds:  [num_tokens, num_classes] (binary; can be one-hot or multi-hot)
    - labels: [num_tokens, num_classes] (binary; multi-hot)

    Output matrix shape:
    - (num_classes + 1, num_classes + 1)
        last row  = NTL (No True Label)
        last col  = NPL (No Predicted Label)

    Notes:
    - Implements the paper's 2-step update:
        1) increment diagonal for correctly predicted true labels (Ti1)
        and handle (Ti=∅, Pi=∅) -> M(NTL, NPL)++
        2) category-based FN/FP scattering (Category 1/2/3)
    """

    if preds.shape != labels.shape or preds.ndim != 2:
        raise ValueError(
            "preds and labels must have shape [num_tokens, num_classes] and match."
        )

    num_tokens, q = preds.shape
    NTL = q  # last row
    NPL = q  # last col

    # Ensure binary on CPU (avoid int matmul on CUDA, and keep code simple)
    preds_b = (preds > 0).to(torch.uint8).cpu().numpy()
    labels_b = (labels > 0).to(torch.uint8).cpu().numpy()

    # MLCM has q+1 rows and q+1 cols (extra NTL row and NPL col) :contentReference[oaicite:4]{index=4}
    M = np.zeros((q + 1, q + 1), dtype=np.int64)

    for i in range(num_tokens):
        Ti = np.flatnonzero(labels_b[i])  # true label set
        Pi = np.flatnonzero(preds_b[i])  # predicted label set

        Ti_set = set(Ti.tolist())
        Pi_set = set(Pi.tolist())

        # Partition sets as in the paper: Ti1 = Pi1 = intersection; Ti2, Pi2 are misses/extra :contentReference[oaicite:5]{index=5}
        Ti1 = Ti_set & Pi_set
        Ti2 = Ti_set - Pi_set
        Pi2 = Pi_set - Ti_set

        # ----------------------------
        # Step 1 (Algorithm 2): TP on diagonal for all r in Ti1,
        # plus special case (Ti=∅ and Pi=∅) -> M(NTL,NPL)++ :contentReference[oaicite:6]{index=6}
        # ----------------------------
        for r in Ti1:
            M[r, r] += 1

        if len(Ti_set) == 0 and len(Pi_set) == 0:
            M[NTL, NPL] += 1
            continue

        # ----------------------------
        # Step 2: category-specific update
        # Category One: Pi ⊆ Ti  (no incorrect predictions; possibly missing true labels)
        # -> for r in Ti2: M(r, NPL)++  :contentReference[oaicite:7]{index=7}
        # Category Two: Ti ⊂ Pi (all true predicted; plus extra incorrect predictions)
        # -> for r in Ti: for c in Pi2: M(r,c)++
        #    and if Ti=∅: for c in Pi2: M(NTL,c)++ :contentReference[oaicite:8]{index=8}
        # Category Three: Ti2 != ∅ and Pi2 != ∅ (misses + extras)
        # -> for r in Ti2: for c in Pi2: M(r,c)++ :contentReference[oaicite:9]{index=9}
        # ----------------------------
        if len(Pi2) == 0:
            # Category 1 (Pi ⊆ Ti) in practice: Pi_set ⊆ Ti_set
            for r in Ti2:
                M[r, NPL] += 1

        elif len(Ti2) == 0:
            # Category 2 (Ti ⊂ Pi): all true predicted, plus extras
            if len(Ti_set) == 0:
                for c in Pi2:
                    M[NTL, c] += 1
            else:
                for r in Ti_set:
                    for c in Pi2:
                        M[r, c] += 1

        else:
            # Category 3: misses and extras
            for r in Ti2:
                for c in Pi2:
                    M[r, c] += 1

    # ---- labels for axes
    if id2label is None:
        class_names = [str(i) for i in range(q)]
    elif isinstance(id2label, dict):
        class_names = [rename_labels[id2label[i]] for i in range(q)]
    else:
        if len(id2label) != q:
            raise ValueError(
                f"id2label has length {len(id2label)} but num_classes is {q}."
            )
        class_names = list(id2label)

    x_names = class_names + ["NPL"]
    y_names = class_names + ["NTL"]

    plot_M = M.astype(float)
    if normalize_rows:
        row_sums = plot_M.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        plot_M = plot_M / row_sums

    # ---- plot
    plt.figure(figsize=figsize)
    im = plt.imshow(plot_M, cmap="GnBu")
    plt.colorbar(im)

    plt.xticks(np.arange(q + 1), x_names, rotation=90)
    plt.yticks(np.arange(q + 1), y_names)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)

    if annotate:
        for r in range(q + 1):
            for c in range(q + 1):
                val = plot_M[r, c]
                txt = f"{val:.2f}" if normalize_rows else f"{int(M[r, c])}"
                plt.text(c, r, txt, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return M


for exp in experiments:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(r"datasets\breast_cancer_dataset")
    model_id = Path(rf"experiments\{exp}_random_seed_3\best_model")
    model = AutoModelForTokenClassification.from_pretrained(model_id).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    mutually_exclusive_classes = [
        ["HER2_POSITIVO", "HER2_NEGATIVO"],
        ["RE_POSITIVO", "RE_NEGATIVO"],
        ["RP_POSITIVO", "RP_NEGATIVO"],
    ]
    test_dataset = ClinicalRecordsDataset(
        dataset_path, split="test", tokenizer=tokenizer, random_seed=random_seed
    )
    data_collator = DataCollatorForMultiLabelTokenClassification(
        max_length=1024,
        pad_token_id=tokenizer.pad_token_id,
        num_labels=test_dataset.num_labels,
        device=device,
    )

    exclusive_groups = [
        [test_dataset.label2id[label] for label in group]
        for group in mutually_exclusive_classes
    ]
    exclusive_idx = sorted({i for g in exclusive_groups for i in g})

    total_preds = None
    total_labels = None
    with torch.no_grad():
        if "grouped_softmax" in str(model_id):
            for i in tqdm(range(0, len(test_dataset), batch_size)):
                batch = test_dataset[i : i + batch_size]
                batch = data_collator(batch, include_labels=True)

                input_ids = batch["input_ids"].to("cuda")
                attention_mask = batch["attention_mask"].to("cuda")
                labels = batch["labels"].to("cuda")
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                B, L, C = logits.shape
                exclusive_mask = torch.zeros(
                    logits.size(-1), dtype=torch.bool, device=logits.device
                )
                exclusive_mask[exclusive_idx] = True
                preds = torch.zeros_like(logits)
                probs = torch.zeros_like(logits)
                probs_flat = probs.reshape(-1, probs.shape[-1])
                labels_flat = labels.reshape(-1, labels.shape[-1])
                # preds and probs for mutually exclusive groups
                for g in exclusive_groups:
                    group_logits = torch.cat(
                        [logits.new_zeros((B, L, 1)), logits[..., g]], dim=-1
                    )
                    # Softmax in the log space for numerical stability
                    group_probs = (
                        group_logits.exp()
                        / torch.logsumexp(group_logits, dim=-1).view(B, L, 1).exp()
                    )
                    group_preds = group_probs.argmax(dim=-1)
                    # Update preds
                    for g_idx, label_idx in enumerate(g):
                        preds[..., label_idx] = (group_preds == g_idx + 1).float()
                    # Update probs
                    probs[..., g] = group_probs[..., 1:]
                # preds and probs for non-exclusive labels
                non_exclusive_logits = logits[..., ~exclusive_mask]
                probs[..., ~exclusive_mask] = 1 / (1 + torch.exp(-non_exclusive_logits))

                best_thresholds = torch.full((C,), 0.5).to(device)
                for k in range(C):
                    best_thresholds[k] = model.config.thresholds.get(
                        test_dataset.id2label[k], {}
                    ).get("threshold", 0.5)
                non_exclusive_thresholds = best_thresholds[~exclusive_mask]
                non_exclusive_probs = probs[..., ~exclusive_mask]
                preds[..., ~exclusive_mask] = (
                    non_exclusive_probs > non_exclusive_thresholds
                ).float()
                preds_flat = preds.reshape(-1, preds.shape[-1])
                labels_flat = labels.reshape(-1, labels.shape[-1])

                if total_preds is None and total_labels is None:
                    total_preds = preds_flat
                    total_labels = labels_flat
                else:
                    total_preds = torch.cat((total_preds, preds_flat), dim=0)
                    total_labels = torch.cat((total_labels, labels_flat), dim=0)

        else:
            for i in tqdm(range(0, len(test_dataset), batch_size)):
                batch = test_dataset[i : i + batch_size]
                batch = data_collator(batch, include_labels=True)

                input_ids = batch["input_ids"].to("cuda")
                attention_mask = batch["attention_mask"].to("cuda")
                labels = batch["labels"].to("cuda")
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                B, L, C = logits.shape
                probs = torch.nn.functional.sigmoid(logits)
                best_thresholds = torch.full((C,), 0.5).to(device)
                for k in range(C):
                    best_thresholds[k] = model.config.thresholds.get(
                        test_dataset.id2label[k], {}
                    ).get("threshold", 0.5)
                preds = (probs > best_thresholds).float()

                preds_flat = preds.reshape(-1, preds.shape[-1])
                labels_flat = labels.reshape(-1, labels.shape[-1])

                if total_preds is None and total_labels is None:
                    total_preds = preds_flat
                    total_labels = labels_flat
                else:
                    total_preds = torch.cat((total_preds, preds_flat), dim=0)
                    total_labels = torch.cat((total_labels, labels_flat), dim=0)

        output_path = Path(f"plots/confusion_matrices/{exp}_random_seed_3.png")
        plot_mlcm(
            total_preds,
            total_labels,
            test_dataset.id2label,
            normalize_rows=True,
            title=rename_experiments[exp],
            output_path=output_path,
        )
