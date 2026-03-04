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

@dataclass
class _StreamAttn:
    device: torch.device
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.sink_sum = torch.zeros((), device=self.device, dtype=self.dtype)
        self.ent_sum = torch.zeros((), device=self.device, dtype=self.dtype)
        self.topk_sum = torch.zeros((), device=self.device, dtype=self.dtype)
        self.q_count = torch.zeros((), device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def update(
        self,
        attn: torch.Tensor,           # [B,H,Q,K], softmax over K
        attention_mask: torch.Tensor, # [B,L] 1/0
        sink_idx: int,
        topk: int,
        normalize_entropy: bool,
        eps: float = 1e-12,
    ) -> None:
        B, H, Q, K = attn.shape
        m = attention_mask.to(torch.bool)

        # slice masks to Q/K lengths (BERT: Q==K==L)
        qmask = m[:, :Q]                              # [B,Q]
        kmask = m[:, :K]                              # [B,K]
        qmask_f = qmask[:, None, :].to(self.dtype)    # [B,1,Q]
        kmask_f = kmask[:, None, None, :].to(self.dtype)  # [B,1,1,K]

        if sink_idx < 0 or sink_idx >= K:
            raise ValueError(f"sink_idx={sink_idx} out of range for K={K}")

        # mask padded keys then renormalize over keys
        attn = attn.to(self.dtype) * kmask_f
        denom = attn.sum(dim=-1, keepdim=True).clamp_min(eps)
        attn = attn / denom

        # count valid queries (across batch*heads)
        q_count = (qmask_f.sum() * H).clamp_min(0.0)
        if q_count.item() == 0:
            return
        self.q_count += q_count

        # sink mass
        sink = attn[..., sink_idx]                    # [B,H,Q]
        self.sink_sum += (sink * qmask_f).sum()

        # entropy
        ent = -(attn * attn.clamp_min(eps).log()).sum(dim=-1)  # [B,H,Q]
        if normalize_entropy:
            n_keys = kmask.sum(dim=-1).clamp_min(1).to(self.dtype)  # [B]
            ent = ent / n_keys.log().view(B, 1, 1).clamp_min(eps)
        self.ent_sum += (ent * qmask_f).sum()

        # top-k mass
        k = int(min(max(topk, 1), K))
        topk_mass = attn.topk(k, dim=-1).values.sum(dim=-1)    # [B,H,Q]
        self.topk_sum += (topk_mass * qmask_f).sum()

    def compute(self) -> Dict[str, float]:
        denom = self.q_count.clamp_min(1.0)
        return {
            "sink_mass": float((self.sink_sum / denom).item()),
            "attn_entropy": float((self.ent_sum / denom).item()),
            "attn_topk_mass": float((self.topk_sum / denom).item()),
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
    
    The question we can answer with this:
    1. Are the model’s probabilities calibrated for each exclusive group (0/+/-)?
      - If the model predicts with confidence ~c, is it correct about ~c of the time?
    2. Which group is the most/least miscalibrated?
      - Compare ece_group_i across groups.
    3. Is the model’s confidence usable for thresholding/abstention?
      - Low ECE => confidence is a reliable risk indicator
      - High ECE => confidence is misleading.
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

    The question we can answer with this:
    1. Which model produces better probability estimates for each exclusive group (0/+/-)?
      - Lower brier_group_i is better.
    2. Which group contributes most to probability error?
      - Compare brier_group_i across groups.
    3. Did a loss improve probability quality even if accuracy/F1 stayed similar?
      - Brier is sensitive to “how wrong” and “how confident,” not just argmax correctness.
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
    1. What error mode dominates per group?
      - false mentions: true 0 -> pred (+/-)
      - missed mentions: true (+/-) -> pred 0
      - polarity flips: true + <-> pred -
    2. Is the model collapsing to a default class for a group?
      - e.g., always predicting 0.
    3. Which group is the bottleneck, and which error mode causes it? [probably too specific]
      - compare confusion matrices across groups.
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

@torch.no_grad()
def attention_sinks(
    model: torch.nn.Module,
    test_data: Iterable[Dict[str, Any]],
    loss_type: str,   # kept for uniform signature; unused
    **kwargs,
) -> Dict[str, Any]:
    """
    diagnostic_function(model, test_data, loss_type, **kwargs) -> dict

    Purpose:
      Attention concentration diagnostics (NOT causal explanations). Uses attention weights
      A^{(l)}[b,h,q,k] returned by model(..., output_attentions=True).

    Returns:
      - sink_mass: global mean attention paid to a fixed sink key position (sink_idx, default 0).
      - attn_entropy: mean Shannon entropy of attention rows (optionally normalized).
      - attn_topk_mass: mean sum of the top-k attention weights per query.
      - per_layer: same metrics as lists over layers (length = #layers).

    Note:
      - Needs access to the attention weights, currently forces it with:
        model.set_attn_implementation("eager")

    Required kwargs:
      - None

    Optional kwargs:
      - device: str|torch.device = None
      - max_batches: int|None
      - sink_idx: int = 0          (which key position is treated as the sink)
      - topk: int = 5             (k for top-k mass)
      - normalize_entropy: bool = True  (divide entropy by log(#valid_keys))

    Metric definitions (per attention row over keys k):
      - sink_mass(q) = A[q, sink_idx]
      - entropy(q) = - sum_k A[q,k] log A[q,k]
        normalized_entropy(q) = entropy(q) / log(#valid_keys)
      - topk_mass(q) = sum_{k in TopK(A[q,*])} A[q,k]

    Paper pointers:
      - sink_mass / attention sinks: "StreamingLLM" (Xiao et al., 2023), arXiv:2309.17453
        https://arxiv.org/abs/2309.17453
      - attention entropy (Shannon entropy on attention rows): Zhai et al., ICML 2023 (PMLR)
        https://proceedings.mlr.press/v202/zhai23a.html
      - top-k mass as attention concentration summary: commonly used as a sparsity/concentration proxy
        alongside entropy in attention analysis (e.g., Riffi-Aslett & Fell, 2026)
        https://link.springer.com/article/10.1007/s00138-025-01781-x

    The questions we can answer with this:
      1. Does the model exhibit sink behavior (e.g., excessive attention to [CLS] at position 0)?
         - Higher sink_mass suggests stronger sink usage.
      2. Is attention distributed or concentrated?
         - Lower attn_entropy and higher attn_topk_mass indicate more concentration.
      3. How does attention concentration evolve across layers?
         - Use per_layer curves to see early vs late layer behavior.

    Notes:
      - For long sequences, output_attentions can be heavy: attention tensors scale as O(L^2).
    """
    device = _infer_device(model, kwargs.get("device", None))
    max_batches = kwargs.get("max_batches", None)
    sink_idx = int(kwargs.get("sink_idx", 0))
    topk = int(kwargs.get("topk", 5))
    normalize_entropy = bool(kwargs.get("normalize_entropy", True))

    model = model.to(device)
    model.eval()
    # this is to make sure that we get the attention matrices
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")

    global_stats = _StreamAttn(device=device)
    per_layer_stats = None

    for bi, batch in enumerate(test_data):
        if max_batches is not None and bi >= int(max_batches):
            break

        batch = _to_device(batch, device)
        attn_mask = batch.get("attention_mask", None)
        if attn_mask is None:
            attn_mask = torch.ones_like(batch["input_ids"], dtype=torch.long, device=device)

        inputs = {k: v for k, v in batch.items() if k != "labels"}
        out = model(**inputs, output_attentions=True, return_dict=True)
        attns = getattr(out, "attentions", None)
        if attns is None:
            raise ValueError("No attentions returned. Model must support output_attentions=True.")

        if per_layer_stats is None:
            per_layer_stats = [_StreamAttn(device=device) for _ in range(len(attns))]

        for li, A in enumerate(attns):
            per_layer_stats[li].update(A, attn_mask, sink_idx, topk, normalize_entropy)
            global_stats.update(A, attn_mask, sink_idx, topk, normalize_entropy)

    out = global_stats.compute()
    # just in case: some models may not return attentions, this makes it clear.
    attns = getattr(out, "attentions", None)
    if not attns:  # None or empty tuple/list
        raise ValueError(
            "No attentions returned. Use attn_implementation='eager' (SDPA/Flash usually can't return weights)."
        )
    if per_layer_stats is not None:
        out["per_layer"] = {
            "sink_mass": [s.compute()["sink_mass"] for s in per_layer_stats],
            "attn_entropy": [s.compute()["attn_entropy"] for s in per_layer_stats],
            "attn_topk_mass": [s.compute()["attn_topk_mass"] for s in per_layer_stats],
        }
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

## sample call for attention_sinks
# from spesia_research.diagnostics import attention_sinks

# report = attention_sinks(
#     model=model,
#     test_data=test_data,
#     loss_type="bce_with_grouped_softmax",
#     sink_idx=0,
#     topk=5,
#     normalize_entropy=True,
#     max_batches=5,
# )
# report
## and then for plotting:
# import matplotlib.pyplot as plt
#
# layers = list(range(1, len(report["per_layer"]["sink_mass"]) + 1))
#
## sink mass
# plt.figure()
# plt.plot(layers, report["per_layer"]["sink_mass"], marker="o")
# plt.xlabel("Layer")
# plt.ylabel("Sink mass (to sink_idx)")
# plt.title("Attention sink mass by layer")
# plt.grid(True)
# plt.show()
#
## entropy
# plt.figure()
# plt.plot(layers, report["per_layer"]["attn_entropy"], marker="o")
# plt.xlabel("Layer")
# plt.ylabel("Attention entropy (normalized)")
# plt.title("Attention entropy by layer")
# plt.grid(True)
# plt.show()
#
## top-k mass
# plt.figure()
# plt.plot(layers, report["per_layer"]["attn_topk_mass"], marker="o")
# plt.xlabel("Layer")
# plt.ylabel(f"Top-k mass (k={5})")
# plt.title("Top-k attention mass by layer")
# plt.grid(True)
# plt.show()

# # global summary
# print(
#     "global:",
#     {k: report[k] for k in ["sink_mass", "attn_entropy", "attn_topk_mass"]}
# )