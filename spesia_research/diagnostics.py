from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import torch


# -------------------------
# Core: streaming top-1 ECE
# -------------------------
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