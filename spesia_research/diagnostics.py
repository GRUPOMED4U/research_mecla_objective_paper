# written with ChatGPT, tested with random data and trained model
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import torch


# ---------------------------
# Core: streaming diagnostics
# ---------------------------
@dataclass
class _StreamingECE:
    n_bins: int
    device: torch.device
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.edges = torch.linspace(0.0, 1.0, self.n_bins + 1, device=self.device, dtype=self.dtype)
        self.counts = torch.zeros(self.n_bins, device=self.device, dtype=self.dtype)
        self.sum_conf = torch.zeros(self.n_bins, device=self.device, dtype=self.dtype)
        self.sum_acc = torch.zeros(self.n_bins, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def update(self, probs: torch.Tensor, y: torch.Tensor) -> None:
        """
        probs: [N,K] simplex
        y:     [N] int
        """
        conf, pred = probs.max(dim=1)                 # [N]
        correct = (pred == y).to(self.dtype)          # [N]
        bin_idx = torch.bucketize(conf, self.edges[1:-1], right=True)  # [N] in 0..B-1

        self.counts += torch.bincount(bin_idx, minlength=self.n_bins).to(self.dtype)
        self.sum_conf.scatter_add_(0, bin_idx, conf.to(self.dtype))
        self.sum_acc.scatter_add_(0, bin_idx, correct)

    def compute(self) -> float:
        nz = self.counts > 0
        avg_conf = torch.zeros_like(self.sum_conf)
        avg_acc = torch.zeros_like(self.sum_acc)
        avg_conf[nz] = self.sum_conf[nz] / self.counts[nz]
        avg_acc[nz] = self.sum_acc[nz] / self.counts[nz]
        n = self.counts.sum().clamp_min(1.0)
        ece = ((self.counts / n) * (avg_acc - avg_conf).abs()).sum()
        return float(ece.item())

@dataclass
class _StreamingBrier:
    device: torch.device
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.sum_bs = torch.zeros((), device=self.device, dtype=self.dtype)
        self.count = torch.zeros((), device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def update(self, probs: torch.Tensor, y: torch.Tensor) -> None:
        """
        probs: [N,K] simplex
        y:     [N] int
        """
        probs = probs.to(self.dtype)
        y = y.to(torch.long)

        # bs_i = sum_k p_k^2 - 2*p_y + 1
        p2 = (probs * probs).sum(dim=1)                         # [N]
        py = probs.gather(1, y.view(-1, 1)).squeeze(1)          # [N]
        bs = p2 - 2.0 * py + 1.0                                # [N]

        self.sum_bs += bs.sum()
        self.count += torch.tensor(bs.numel(), device=self.device, dtype=self.dtype)

    def compute(self) -> float:
        denom = self.count.clamp_min(1.0)
        return float((self.sum_bs / denom).item())



@dataclass
class _StreamingConfusion:
    k: int
    device: torch.device
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.mat = torch.zeros((self.k, self.k), device=self.device, dtype=self.dtype)  # [true, pred]

    @torch.no_grad()
    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        """
        y_true, y_pred: [N] ints in 0..k-1
        """
        y_true = y_true.to(torch.long)
        y_pred = y_pred.to(torch.long)
        idx = y_true * self.k + y_pred
        counts = torch.bincount(idx, minlength=self.k * self.k).to(self.dtype)
        self.mat += counts.view(self.k, self.k)

    def compute(self) -> Dict[str, Any]:
        total = self.mat.sum().clamp_min(1.0)
        acc = (torch.diag(self.mat).sum() / total).item()
        return {
            "confusion": self.mat.detach().cpu().tolist(),
            "accuracy": float(acc),
            "support_true": self.mat.sum(dim=1).detach().cpu().tolist(),  # per true class
            "support_pred": self.mat.sum(dim=0).detach().cpu().tolist(),  # per predicted class
            "total": float(total.item()),
        }

# -------------------------
# Helpers
# -------------------------
def _infer_device(model: torch.nn.Module, device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def _groups_to_indices(
    groups: Sequence[Sequence[Union[int, str]]],
    label2id: Optional[Dict[str, int]],
) -> List[List[int]]:
    out: List[List[int]] = []
    for g in groups:
        if len(g) == 0:
            continue
        if isinstance(g[0], int):
            out.append([int(x) for x in g])  # type: ignore[arg-type]
        else:
            if label2id is None:
                raise ValueError("Groups are strings but label2id is missing.")
            out.append([label2id[str(x)] for x in g])
    return out


def _group_probs_and_targets(
    logits: torch.Tensor,   # [B,L,C]
    labels: torch.Tensor,   # [B,L,C] in {0,1}
    group: Sequence[int],   # indices into C (exclude implicit 0)
    unk_logit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      probs: [B,L, 1+|group|] for [0, group...]
      y:     [B,L] in {0..|group|}  (0 = none-of-group)
    """
    B, L, _ = logits.shape
    g_logits = torch.cat(
        [logits.new_full((B, L, 1), float(unk_logit)), logits[..., list(group)]],
        dim=-1,
    )
    probs = torch.softmax(g_logits, dim=-1)

    g_lab = labels[..., list(group)]                 # [B,L,|g|]
    any_on = g_lab.any(dim=-1)                       # [B,L]
    idx = g_lab.to(torch.float32).argmax(dim=-1)     # [B,L]
    y = torch.zeros_like(idx, dtype=torch.long)
    y[any_on] = idx[any_on] + 1
    return probs, y


# -------------------------
# Public diagnostic
# -------------------------
@torch.no_grad()
def ece(
    model: torch.nn.Module,
    test_data: Iterable[Dict[str, Any]],
    loss_type: str,
    **kwargs,
) -> Dict[str, float]:
    """
    diagnostic_function(model, test_data, loss_type, **kwargs) -> dict

    Supports:
      - loss_type == "bce_with_grouped_softmax": per-group ECE over implicit-[0]+group softmax.

    Required kwargs for grouped softmax:
      - mutually_exclusive_classes: list[list[int|str]]

    Optional kwargs:
      - n_bins: int = 15
      - unk_logit: float = 0.0
      - device: str|torch.device = None
      - max_batches: int|None
    """
    n_bins: int = int(kwargs.get("n_bins", 15))
    unk_logit: float = float(kwargs.get("unk_logit", 0.0))
    device = _infer_device(model, kwargs.get("device", None))
    max_batches = kwargs.get("max_batches", None)

    model = model.to(device)
    model.eval()

    if loss_type != "bce_with_grouped_softmax":
        raise NotImplementedError("ece() currently implemented for loss_type='bce_with_grouped_softmax' only.")

    groups_raw = kwargs.get("mutually_exclusive_classes", None)
    if groups_raw is None:
        raise ValueError("Provide mutually_exclusive_classes=... in kwargs.")

    label2id = getattr(getattr(model, "config", None), "label2id", None)
    groups = _groups_to_indices(groups_raw, label2id)

    aggs = [_StreamingECE(n_bins=n_bins, device=device) for _ in groups]

    for bi, batch in enumerate(test_data):
        if max_batches is not None and bi >= int(max_batches):
            break

        batch = _to_device(batch, device)
        labels = batch["labels"]
        attn = batch.get("attention_mask", None)
        if attn is None:
            attn = torch.ones_like(batch["input_ids"], dtype=torch.long, device=device)
        m = attn.bool()

        inputs = {k: v for k, v in batch.items() if k != "labels"}
        out = model(**inputs)
        logits = out.logits

        # normalize shapes to [B,L,C]
        if logits.ndim == 2:
            logits = logits.unsqueeze(1)     # [B,1,C]
            labels = labels.unsqueeze(1)     # [B,1,C]
            m = m.unsqueeze(1) if m.ndim == 1 else m

        labels = labels.to(torch.bool)

        for gi, g in enumerate(groups):
            probs, y = _group_probs_and_targets(logits, labels, g, unk_logit=unk_logit)
            p = probs[m].view(-1, probs.size(-1))
            yy = y[m].view(-1)
            aggs[gi].update(p, yy)

    out = {f"ece_group_{i}": aggs[i].compute() for i in range(len(aggs))}
    out["ece_macro"] = float(sum(out[f"ece_group_{i}"] for i in range(len(aggs))) / max(len(aggs), 1))
    return out

@torch.no_grad()
def brier(
    model: torch.nn.Module,
    test_data: Iterable[Dict[str, Any]],
    loss_type: str,
    **kwargs,
) -> Dict[str, float]:
    """
    diagnostic_function(model, test_data, loss_type, **kwargs) -> dict

    Supports:
      - loss_type == "bce_with_grouped_softmax": per-group Brier over implicit-[0]+group softmax.

    Required kwargs for grouped softmax:
      - mutually_exclusive_classes: list[list[int|str]]

    Optional kwargs:
      - unk_logit: float = 0.0
      - device: str|torch.device = None
      - max_batches: int|None
    """
    unk_logit: float = float(kwargs.get("unk_logit", 0.0))
    device = _infer_device(model, kwargs.get("device", None))
    max_batches = kwargs.get("max_batches", None)

    model = model.to(device)
    model.eval()

    if loss_type != "bce_with_grouped_softmax":
        raise NotImplementedError("brier() currently implemented for loss_type='bce_with_grouped_softmax' only.")

    groups_raw = kwargs.get("mutually_exclusive_classes", None)
    if groups_raw is None:
        raise ValueError("Provide mutually_exclusive_classes=... in kwargs.")

    label2id = getattr(getattr(model, "config", None), "label2id", None)
    groups = _groups_to_indices(groups_raw, label2id)

    aggs = [_StreamingBrier(device=device) for _ in groups]

    for bi, batch in enumerate(test_data):
        if max_batches is not None and bi >= int(max_batches):
            break

        batch = _to_device(batch, device)
        labels = batch["labels"]
        attn = batch.get("attention_mask", None)
        if attn is None:
            attn = torch.ones_like(batch["input_ids"], dtype=torch.long, device=device)
        m = attn.bool()

        inputs = {k: v for k, v in batch.items() if k != "labels"}
        out = model(**inputs)
        logits = out.logits

        # normalize shapes to [B,L,C]
        if logits.ndim == 2:
            logits = logits.unsqueeze(1)
            labels = labels.unsqueeze(1)
            m = m.unsqueeze(1) if m.ndim == 1 else m

        labels = labels.to(torch.bool)

        for gi, g in enumerate(groups):
            probs, y = _group_probs_and_targets(logits, labels, g, unk_logit=unk_logit)
            p = probs[m].view(-1, probs.size(-1))
            yy = y[m].view(-1)
            aggs[gi].update(p, yy)

    out = {f"brier_group_{i}": aggs[i].compute() for i in range(len(aggs))}
    out["brier_macro"] = float(sum(out[f"brier_group_{i}"] for i in range(len(aggs))) / max(len(aggs), 1))
    return out

@torch.no_grad()
def group_confusion(
    model: torch.nn.Module,
    test_data: Iterable[Dict[str, Any]],
    loss_type: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    diagnostic_function(model, test_data, loss_type, **kwargs) -> dict

    Current support:
      - loss_type == "bce_with_grouped_softmax": per-group 3x3 confusion over {0, +, -}
        where 0 is implicit (none-of-group).

    Required kwargs:
      - mutually_exclusive_classes: list[list[int|str]]

    Optional kwargs:
      - unk_logit: float = 0.0
      - device: str|torch.device = None
      - max_batches: int|None

    The question we can answer with this:
    1. What kind of mistakes is the model making: false mentions, missed mentions, polarity flips?
      - False mentions (true 0 -> pred +/−): 0 (never predicts +/−)
      - Polarity flips (+ <-> −): 0 (never predicts +/−).
      - Missed mentions (true +/− → pred 0): all non-zero truths are missed.
    2. Is the model collapsing to a default class for a group?
    3. What is the class-conditional performance (not just a single accuracy)? [kind of redundant but still useful]
    4. Which group(s) are the bottleneck, and in what way? [too specific for now]
      - i.e. Group A has higher missed mention than Group B.
    """
    unk_logit: float = float(kwargs.get("unk_logit", 0.0))
    device = _infer_device(model, kwargs.get("device", None))
    max_batches = kwargs.get("max_batches", None)

    model = model.to(device)
    model.eval()

    if loss_type != "bce_with_grouped_softmax":
        raise NotImplementedError("group_confusion() currently implemented for loss_type='bce_with_grouped_softmax' only.")

    groups_raw = kwargs.get("mutually_exclusive_classes", None)
    if groups_raw is None:
        raise ValueError("Provide mutually_exclusive_classes=... in kwargs.")

    label2id = getattr(getattr(model, "config", None), "label2id", None)
    groups = _groups_to_indices(groups_raw, label2id)

    # for your project these are 2-way groups -> K=3 with implicit 0
    aggs = [_StreamingConfusion(k=3, device=device) for _ in groups]

    for bi, batch in enumerate(test_data):
        if max_batches is not None and bi >= int(max_batches):
            break

        batch = _to_device(batch, device)
        labels = batch["labels"]
        attn = batch.get("attention_mask", None)
        if attn is None:
            attn = torch.ones_like(batch["input_ids"], dtype=torch.long, device=device)
        m = attn.bool()

        inputs = {k: v for k, v in batch.items() if k != "labels"}
        out = model(**inputs)
        logits = out.logits

        # normalize to [B,L,C]
        if logits.ndim == 2:
            logits = logits.unsqueeze(1)
            labels = labels.unsqueeze(1)
            m = m.unsqueeze(1) if m.ndim == 1 else m

        labels = labels.to(torch.bool)

        for gi, g in enumerate(groups):
            probs, y = _group_probs_and_targets(logits, labels, g, unk_logit=unk_logit)  # probs [B,L,3], y [B,L]
            pred = probs.argmax(dim=-1)                                                   # [B,L] in 0..2

            yy = y[m].view(-1)
            pp = pred[m].view(-1)
            aggs[gi].update(yy, pp)

    return {f"group_{i}": aggs[i].compute() for i in range(len(aggs))}

## sample call for ECE
# from transformers import AutoModelForTokenClassification
# from spesia_research.diagnostics import ece
#
# model = AutoModelForTokenClassification.from_pretrained("../best_model.pt", use_safetensors=True)
# # test_loader yields dicts with input_ids, attention_mask, labels
# report = ece(
#     model,
#     test_loader,
#     "bce_with_grouped_softmax",
#     mutually_exclusive_classes=[["HER2_POSITIVO","HER2_NEGATIVO"], ["RE_POSITIVO","RE_NEGATIVO"], ["RP_POSITIVO","RP_NEGATIVO"]],
#     n_bins=15,
# )
# print(report)

## sample call for Brier
# from spesia_research.diagnostics import brier
#
# report = brier(
#     model=model,
#     test_data=test_data,
#     loss_type="bce_with_grouped_softmax",
#     mutually_exclusive_classes=groups_idx,
# )
# report

## sample call for group_confusion (per-group confusion / error modes)
# from transformers import AutoModelForTokenClassification
# from spesia_research.diagnostics import group_confusion
#
# rep = group_confusion(
#     model=model,
#     test_data=test_loader,
#     loss_type="bce_with_grouped_softmax",
#     mutually_exclusive_classes=[
#         ["HER2_POSITIVO", "HER2_NEGATIVO"],
#         ["RE_POSITIVO", "RE_NEGATIVO"],
#         ["RP_POSITIVO", "RP_NEGATIVO"],
#     ],
# )
# print(rep)