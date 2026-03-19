from typing import Literal
from typing import List
from abc import ABC, abstractmethod
import torch
from transformers import EvalPrediction
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import average_precision_score

from spesia_research.data_models import ThresholdMap


def compute_metrics(eval_pred: EvalPrediction):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids.astype(np.float32)
    # logits: [batch, seq_len, num_labels]
    # labels: [batch, seq_len, num_labels]

    probs = 1 / (1 + np.exp(-logits))
    labels = labels.astype(int)

    # Flatten batch and sequence for metrics
    probs_flat = probs.reshape(-1, probs.shape[-1])
    labels_flat = labels.reshape(-1, labels.shape[-1])

    metrics = {}

    if probs.shape[-1] > 1:
        # Multilabel/multiclass case
        thresholds = np.linspace(0.01, 0.99, 99)
        best_f1 = 0
        best_threshold = 0.5
        for t in thresholds:
            preds_flat = (probs_flat > t).astype(int)
            f1 = f1_score(labels_flat, preds_flat, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        # Use best threshold for all metrics
        preds_flat = (probs_flat > best_threshold).astype(int)
        metrics["best_threshold"] = best_threshold
        metrics["macro_precision"] = precision_score(
            labels_flat, preds_flat, average="macro", zero_division=0
        )
        metrics["macro_recall"] = recall_score(
            labels_flat, preds_flat, average="macro", zero_division=0
        )
        metrics["macro_f1"] = f1_score(
            labels_flat, preds_flat, average="macro", zero_division=0
        )

        metrics["micro_precision"] = precision_score(
            labels_flat, preds_flat, average="micro", zero_division=0
        )
        metrics["micro_recall"] = recall_score(
            labels_flat, preds_flat, average="micro", zero_division=0
        )
        metrics["micro_f1"] = f1_score(
            labels_flat, preds_flat, average="micro", zero_division=0
        )

        macro_ap = average_precision_score(labels_flat, probs_flat, average="macro")
        micro_ap = average_precision_score(labels_flat, probs_flat, average="micro")
        metrics["macro_pr_auc"] = macro_ap
        metrics["micro_pr_auc"] = micro_ap

    else:
        # Single-label (binary) case
        probs_flat = probs_flat.reshape(-1)
        labels_flat = labels_flat.reshape(-1)
        thresholds = np.linspace(0.01, 0.99, 99)
        best_f1 = 0
        best_threshold = 0.5
        for t in thresholds:
            preds_flat = (probs_flat > t).astype(int)
            f1 = f1_score(labels_flat, preds_flat, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        # Use best threshold for all metrics
        preds_flat = (probs_flat > best_threshold).astype(int)
        metrics["best_threshold"] = best_threshold
        metrics["precision"] = precision_score(
            labels_flat, preds_flat, average="binary", zero_division=0
        )
        metrics["recall"] = recall_score(
            labels_flat, preds_flat, average="binary", zero_division=0
        )
        metrics["f1"] = f1_score(
            labels_flat, preds_flat, average="binary", zero_division=0
        )

        pr_auc = average_precision_score(labels_flat, probs_flat)
        metrics["pr_auc"] = pr_auc

    return metrics


def compute_metrics_with_per_label_thresholds(
    eval_pred: EvalPrediction, include_per_label_thresholds=False
):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids.astype(np.float32)
    # logits: [batch, seq_len, num_labels]
    # labels: [batch, seq_len, num_labels]

    probs = 1 / (1 + np.exp(-logits))
    labels = labels.astype(int)

    # Flatten batch and sequence for metrics
    probs_flat = probs.reshape(-1, probs.shape[-1])
    labels_flat = labels.reshape(-1, labels.shape[-1])

    metrics = {}

    if probs.shape[-1] > 1:
        # Multilabel case — compute a threshold PER LABEL
        thresholds = np.linspace(0.01, 0.99, 99)
        n_labels = probs_flat.shape[-1]
        best_thresholds = np.full(n_labels, 0.5, dtype=np.float32)

        for k in range(n_labels):
            y_true = labels_flat[:, k]
            p = probs_flat[:, k]
            best_f1_k = -1.0
            best_t_k = 0.5
            # Skip labels that are all one class to avoid degenerate optimization
            # (we still keep default 0.5)
            if (y_true.sum() == 0) or (y_true.sum() == y_true.shape[0]):
                best_thresholds[k] = best_t_k
                continue
            for t in thresholds:
                y_pred_k = (p > t).astype(int)
                f1_k = f1_score(y_true, y_pred_k, average="binary", zero_division=0)
                if f1_k > best_f1_k:
                    best_f1_k = f1_k
                    best_t_k = t
            best_thresholds[k] = best_t_k

        # Use per-label thresholds for final predictions
        preds_flat = (probs_flat > best_thresholds).astype(int)

        if include_per_label_thresholds:
            metrics["best_thresholds"] = best_thresholds

        metrics["best_thresholds_mean"] = best_thresholds.mean().item()

        metrics["macro_precision"] = precision_score(
            labels_flat, preds_flat, average="macro", zero_division=0
        )
        metrics["macro_recall"] = recall_score(
            labels_flat, preds_flat, average="macro", zero_division=0
        )
        metrics["macro_f1"] = f1_score(
            labels_flat, preds_flat, average="macro", zero_division=0
        )
        try:
            metrics["macro_auc"] = roc_auc_score(
                labels_flat, probs_flat, average="macro"
            )
        except ValueError:
            metrics["macro_auc"] = 0.0

        metrics["micro_precision"] = precision_score(
            labels_flat, preds_flat, average="micro", zero_division=0
        )
        metrics["micro_recall"] = recall_score(
            labels_flat, preds_flat, average="micro", zero_division=0
        )
        metrics["micro_f1"] = f1_score(
            labels_flat, preds_flat, average="micro", zero_division=0
        )
        try:
            metrics["micro_auc"] = roc_auc_score(
                labels_flat, probs_flat, average="micro"
            )
        except ValueError:
            metrics["micro_auc"] = 0.0

    else:
        # Single-label (binary) case — unchanged
        probs_flat = probs_flat.reshape(-1)
        labels_flat = labels_flat.reshape(-1)
        thresholds = np.linspace(0.01, 0.99, 99)
        best_f1 = 0.0
        best_threshold = 0.5
        for t in thresholds:
            preds_flat = (probs_flat > t).astype(int)
            f1 = f1_score(labels_flat, preds_flat, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        preds_flat = (probs_flat > best_threshold).astype(int)
        metrics["best_threshold"] = float(best_threshold)
        metrics["precision"] = precision_score(
            labels_flat, preds_flat, average="binary", zero_division=0
        )
        metrics["recall"] = recall_score(
            labels_flat, preds_flat, average="binary", zero_division=0
        )
        metrics["f1"] = f1_score(
            labels_flat, preds_flat, average="binary", zero_division=0
        )
        try:
            metrics["auc"] = roc_auc_score(labels_flat, probs_flat)
        except ValueError:
            metrics["auc"] = 0.0

    return metrics


def get_best_threshold(
    threshold_map,
    min_precision=0.8,
    min_recall=0.7,
    prioritize: Literal["precision", "recall", "f1_score"] = "f1_score",
) -> ThresholdMap:
    """
    Get the best threshold for a given label based on precision and recall. Prioritizes precision by default.

    Args:
        threshold_map (dict):
            A dictionary containing the precision, recall and threshold for each label.
        min_precision (float, optional):
            The minimum precision to consider a threshold as valid. Defaults to 0.8.
        min_recall (float, optional):
            The minimum recall to consider a threshold as valid. Defaults to 0.7.
        prioritize (str, optional):
            The metric to prioritize when selecting the best threshold. Can be "precision", "recall" or "f1_score". Defaults to "f1_score".

    Returns:
        thresholds (ThresholdMap): ThresholdMap
            A ThresholdMap object containing the best threshold, precision and recall for each label.
    """
    thresholds = ThresholdMap()

    # precompute f1_scores
    for label in threshold_map.keys():
        threshold_map[label]["f1_score"] = 2 * (
            (threshold_map[label]["precision"] * threshold_map[label]["recall"])
            / (threshold_map[label]["precision"] + threshold_map[label]["recall"])
        )

    for label in threshold_map.keys():
        best_precision = np.array(0)
        best_threshold = np.array(1)
        best_f1 = np.array(0)
        best_recall = np.array(0)

        for i, t in enumerate(threshold_map[label]["threshold"]):
            if (
                threshold_map[label]["recall"][i] >= min_recall
                and threshold_map[label]["precision"][i] >= min_precision
            ):
                if prioritize == "precision":
                    if threshold_map[label]["precision"][i] >= best_precision:
                        best_precision = threshold_map[label]["precision"][i]
                        best_threshold = threshold_map[label]["threshold"][i]
                        best_recall = threshold_map[label]["recall"][i]
                        best_f1 = threshold_map[label]["f1_score"][i]
                elif prioritize == "recall":
                    if threshold_map[label]["recall"][i] >= best_recall:
                        best_precision = threshold_map[label]["precision"][i]
                        best_threshold = threshold_map[label]["threshold"][i]
                        best_recall = threshold_map[label]["recall"][i]
                        best_f1 = threshold_map[label]["f1_score"][i]
                elif prioritize == "f1_score":
                    if threshold_map[label]["f1_score"][i] >= best_f1:
                        best_precision = threshold_map[label]["precision"][i]
                        best_threshold = threshold_map[label]["threshold"][i]
                        best_recall = threshold_map[label]["recall"][i]
                        best_f1 = threshold_map[label]["f1_score"][i]
                else:
                    raise ValueError(
                        f"'prioritize' must be 'precision' or 'recall' or 'f1_score'. Received: {prioritize}"
                    )

        thresholds[label] = {
            "threshold": best_threshold.item(),
            "precision": best_precision.item(),
            "recall": best_recall.item(),
            "f1_score": best_f1.item(),
            "pr_auc": threshold_map[label]["pr_auc"],
        }

    return thresholds


class CustomMetricsForGroupedSoftmax:
    def __init__(self, mutually_exclusive_classes, label2id, *args, **kwargs):
        self.mutually_exclusive_classes = mutually_exclusive_classes
        self.label2id = label2id
        self.id2label = {v: k for k, v in self.label2id.items()}

    def __call__(
        self,
        eval_pred: EvalPrediction,
        include_per_label_thresholds=False,
        *args,
        **kwargs,
    ):
        logits = torch.tensor(eval_pred.predictions)
        labels = torch.tensor(eval_pred.label_ids)
        B, L, C = logits.shape

        exclusive_groups = [
            [self.label2id[label] for label in group]
            for group in self.mutually_exclusive_classes
        ]
        # Create exclusive groups mask
        exclusive_idx = sorted({i for g in exclusive_groups for i in g})
        exclusive_mask = torch.zeros(
            logits.size(-1), dtype=torch.bool, device=logits.device
        )
        exclusive_mask[exclusive_idx] = True

        preds = torch.zeros_like(logits)
        probs = torch.zeros_like(logits)

        # Flatten batch and sequence for metrics
        probs_flat = probs.reshape(-1, probs.shape[-1])
        labels_flat = labels.reshape(-1, labels.shape[-1])

        metrics = {}

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

        # find best threshold for each non exclusive label
        # Multilabel case — compute a threshold PER LABEL
        thresholds = np.linspace(0.01, 0.99, 99)
        best_thresholds = torch.full((C,), 0.5)
        for k in range(C):
            y_true = labels_flat[:, k]
            p = probs_flat[:, k]
            best_f1_k = -1.0
            best_t_k = 0.5
            # Skip labels that are all one class to avoid degenerate optimization
            # (we still keep default 0.5)
            if (y_true.sum() == 0) or (y_true.sum() == y_true.shape[0]):
                best_thresholds[k] = best_t_k
                continue
            for t in thresholds:
                y_pred_k = p > t
                f1_k = f1_score(y_true, y_pred_k, average="binary", zero_division=0)
                if f1_k > best_f1_k:
                    best_f1_k = f1_k
                    best_t_k = t
            best_thresholds[k] = best_t_k

        non_exclusive_thresholds = best_thresholds[~exclusive_mask]
        non_exclusive_probs = probs[..., ~exclusive_mask]
        preds[..., ~exclusive_mask] = (
            non_exclusive_probs > non_exclusive_thresholds
        ).float()
        preds_flat = preds.reshape(-1, preds.shape[-1])

        # compute metrics
        if include_per_label_thresholds:
            metrics["best_thresholds"] = best_thresholds

        metrics["best_thresholds_mean"] = best_thresholds.mean().item()

        metrics["macro_precision"] = precision_score(
            labels_flat, preds_flat, average="macro", zero_division=0
        )
        metrics["macro_recall"] = recall_score(
            labels_flat, preds_flat, average="macro", zero_division=0
        )
        metrics["macro_f1"] = f1_score(
            labels_flat, preds_flat, average="macro", zero_division=0
        )

        metrics["micro_precision"] = precision_score(
            labels_flat, preds_flat, average="micro", zero_division=0
        )
        metrics["micro_recall"] = recall_score(
            labels_flat, preds_flat, average="micro", zero_division=0
        )
        metrics["micro_f1"] = f1_score(
            labels_flat, preds_flat, average="micro", zero_division=0
        )

        macro_ap = average_precision_score(labels_flat, probs_flat, average="macro")
        micro_ap = average_precision_score(labels_flat, probs_flat, average="micro")
        metrics["macro_pr_auc"] = macro_ap
        metrics["micro_pr_auc"] = micro_ap

        # Metrics per label
        for k in range(C):
            y_true = labels_flat[:, k]
            y_pred = preds_flat[:, k]
            metrics[f"precision_{self.id2label[k]}"] = precision_score(
                y_true, y_pred, average="binary", zero_division=0
            )
            metrics[f"recall_{self.id2label[k]}"] = recall_score(
                y_true, y_pred, average="binary", zero_division=0
            )
            metrics[f"f1_{self.id2label[k]}"] = f1_score(
                y_true, y_pred, average="binary", zero_division=0
            )

        return metrics


class BaseSpanLevelMetrics(ABC):
    def __init__(
        self,
        id2label: dict[int, str] = None,
        mode: Literal["strict", "relaxed"] = "strict",
        device: str | torch.device = None,
    ):
        self.id2label = id2label if id2label is not None else {}
        self.mode = mode
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        if self.mode not in {"strict", "relaxed"}:
            raise ValueError(f"mode must be 'strict' or 'relaxed'. Got: {self.mode}")

    def get_start_end_indexes(self, input: torch.Tensor) -> list[set[tuple[int, int]]]:
        if len(input.shape) == 1:
            input = input.reshape(-1, 1)
        mask = torch.where(input == 1, 1, 0)
        end_shifted_mask = torch.zeros_like(mask)
        end_shifted_mask[:-1] = mask[1:]

        start_shifted_mask = torch.zeros_like(mask)
        start_shifted_mask[1:] = mask[:-1]

        spans_list = []
        for i in range(mask.shape[-1]):
            selected_start_idx = torch.where((mask - start_shifted_mask)[:, i] == 1)[0]
            selected_end_idx = torch.where((mask - end_shifted_mask)[:, i] == 1)[0]
            selected_end_idx += 1  # keep the right index exclusive
            spans_list.append(
                set(zip(selected_start_idx.tolist(), selected_end_idx.tolist()))
            )

        return spans_list

    def _span_contains(self, pred_span, target_span):
        """
        Returns True if pred_span contains target_span.

        Works for:
        - 2-tuples: (start, end)
        - 3-tuples: (batch, start, end)
        """
        if len(pred_span) != len(target_span):
            return False

        if len(pred_span) == 2:
            pred_start, pred_end = pred_span
            target_start, target_end = target_span
            return pred_start <= target_start and pred_end >= target_end

        if len(pred_span) == 3:
            pred_batch, pred_start, pred_end = pred_span
            target_batch, target_start, target_end = target_span
            return (
                pred_batch == target_batch
                and pred_start <= target_start
                and pred_end >= target_end
            )

        raise ValueError(f"Unsupported span format: {pred_span}")

    def _compute_span_counts_strict(self, pred_spans, target_spans):
        tp = len(pred_spans.intersection(target_spans))
        fp = len(pred_spans.difference(target_spans))
        fn = len(target_spans.difference(pred_spans))
        return tp, fp, fn

    def _compute_span_counts_relaxed(self, pred_spans, target_spans):
        matched_targets = set()
        tp = 0
        fp = 0

        for pred_span in pred_spans:
            found_match = False
            for target_span in target_spans:
                if target_span in matched_targets:
                    continue
                if self._span_contains(pred_span, target_span):
                    matched_targets.add(target_span)
                    tp += 1
                    found_match = True
                    break

            if not found_match:
                fp += 1

        fn = len(target_spans) - len(matched_targets)
        return tp, fp, fn

    def _compute_span_counts(self, pred_spans, target_spans):
        if self.mode == "strict":
            return self._compute_span_counts_strict(pred_spans, target_spans)
        elif self.mode == "relaxed":
            return self._compute_span_counts_relaxed(pred_spans, target_spans)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def compute_span_level_metrics(self, predictions, targets):
        if predictions.shape != targets.shape:
            raise ValueError(
                f"predictions and targets must have the same shape, got "
                f"{predictions.shape} and {targets.shape}"
            )

        # 1D: [T] -> treat as a single sequence with one class
        if predictions.ndim == 1:
            spans_x = self.get_start_end_indexes(predictions)
            spans_y = self.get_start_end_indexes(targets)

        # 2D: [T, C] -> treat as a single sequence with multiple classes
        elif predictions.ndim == 2:
            spans_x = self.get_start_end_indexes(predictions)
            spans_y = self.get_start_end_indexes(targets)

        # 3D: [B, T, C] -> process each sequence independently, then aggregate spans per class
        elif predictions.ndim == 3:
            batch_size, seq_len, num_classes = predictions.shape
            spans_x = [set() for _ in range(num_classes)]
            spans_y = [set() for _ in range(num_classes)]

            for b in range(batch_size):
                pred_spans_b = self.get_start_end_indexes(predictions[b])
                target_spans_b = self.get_start_end_indexes(targets[b])

                for c in range(num_classes):
                    spans_x[c].update((b, start, end) for start, end in pred_spans_b[c])
                    spans_y[c].update(
                        (b, start, end) for start, end in target_spans_b[c]
                    )
        else:
            raise ValueError(
                f"Expected predictions/targets with 1, 2, or 3 dimensions, got {predictions.ndim}"
            )

        entity_level_metrics = []
        for span_x, span_y in zip(spans_x, spans_y):
            tp, fp, fn = self._compute_span_counts(span_x, span_y)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1score = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            entity_level_metrics.append((tp, fp, fn, precision, recall, f1score))

        return entity_level_metrics

    def aggregate_span_metrics(self, entity_level_metrics):
        total_tp = sum(m[0] for m in entity_level_metrics)
        total_fp = sum(m[1] for m in entity_level_metrics)
        total_fn = sum(m[2] for m in entity_level_metrics)

        micro_precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        micro_recall = (
            total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        )
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )

        macro_precision = sum(m[3] for m in entity_level_metrics) / len(
            entity_level_metrics
        )
        macro_recall = sum(m[4] for m in entity_level_metrics) / len(
            entity_level_metrics
        )
        macro_f1 = sum(m[5] for m in entity_level_metrics) / len(entity_level_metrics)

        return {
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        }

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def __call__(self, eval_pred: EvalPrediction, *args, **kwargs):
        logits = torch.tensor(eval_pred.predictions)
        labels = torch.tensor(eval_pred.label_ids)
        B, L, C = logits.shape

        preds = self.predict(logits)
        span_level_metrics_per_label = self.compute_span_level_metrics(preds, labels)
        span_level_agg_metrics = self.aggregate_span_metrics(
            span_level_metrics_per_label
        )

        return {
            **span_level_agg_metrics,
            "metrics_per_entity": {
                self.id2label.get(label_idx, label_idx): {
                    "tp": span_level_metrics_per_label[label_idx][0],
                    "fp": span_level_metrics_per_label[label_idx][1],
                    "fn": span_level_metrics_per_label[label_idx][2],
                    "precision": span_level_metrics_per_label[label_idx][3],
                    "recall": span_level_metrics_per_label[label_idx][4],
                    "f1": span_level_metrics_per_label[label_idx][5],
                }
                for label_idx in range(C)
            },
        }


class BCESpanLevelMetrics(BaseSpanLevelMetrics):
    def __init__(
        self,
        prediction_thresholds: List[float] = None,
        id2label: dict[int, str] = None,
        device: str | torch.device = None,
        mode: Literal["strict", "relaxed"] = "strict",
    ):
        super().__init__(id2label=id2label, mode=mode, device=device)

        self.prediction_thresholds = (
            prediction_thresholds if prediction_thresholds is not None else []
        )
        self.prediction_thresholds = torch.tensor(self.prediction_thresholds).to(
            self.device
        )

    def predict(self, logits):
        logits = logits.to(self.device)
        probs = 1 / (1 + torch.exp(-logits))
        preds = (probs > self.prediction_thresholds).float()
        return preds


class GroupedSoftmaxSpanLevelMetrics(BaseSpanLevelMetrics):
    def __init__(
        self,
        mutually_exclusive_classes: List[List[str]],
        id2label: dict[int, str] = None,
        device: str | torch.device = None,
        prediction_thresholds: List[float] = None,
        mode: Literal["strict", "relaxed"] = "strict",
    ):
        super().__init__(id2label=id2label, mode=mode, device=device)
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.exclusive_groups = [
            [self.label2id[label] for label in group]
            for group in mutually_exclusive_classes
        ]
        self.prediction_thresholds = (
            prediction_thresholds if prediction_thresholds is not None else []
        )

    def predict(self, logits):
        exclusive_idx = sorted({i for g in self.exclusive_groups for i in g})
        exclusive_mask = torch.zeros(
            logits.size(-1), dtype=torch.bool, device=logits.device
        ).to(self.device)
        exclusive_mask[exclusive_idx] = True

        # Initialize preds
        preds = torch.zeros_like(logits)

        # preds for mutually exclusive groups
        B, L, C = logits.shape
        for g in self.exclusive_groups:
            group_logits = torch.cat(
                [logits.new_zeros((B, L, 1)), logits[..., g]], dim=-1
            )

            # Softmax in the log space for numerical stability
            group_preds = (
                group_logits.exp()
                / torch.logsumexp(group_logits, dim=-1).view(B, L, 1).exp()
            ).argmax(dim=-1)
            # Update preds
            for g_idx, label_idx in enumerate(g):
                preds[..., label_idx] = (group_preds == g_idx + 1).float()

        # preds for non-exclusive labels
        non_exclusive_logits = logits[..., ~exclusive_mask]
        non_exclusive_probs = 1 / (1 + torch.exp(-non_exclusive_logits))
        non_exclusive_thresholds = torch.tensor(
            self.prediction_thresholds, device=self.device
        )
        preds[..., ~exclusive_mask] = (
            non_exclusive_probs > non_exclusive_thresholds
        ).float()

        return preds


class BaseSequenceLevelMetrics(ABC):
    def __init__(self, id2label: dict[int, str] = None):
        self.id2label = id2label if id2label is not None else {}

    def compute_sequence_level_metrics(self, predictions, targets):
        if predictions.shape != targets.shape:
            raise ValueError(
                f"predictions and targets must have the same shape, got "
                f"{predictions.shape} and {targets.shape}"
            )

        # 1D: [T] -> one sequence, one class
        if predictions.ndim == 1:
            predictions_seq = predictions.reshape(1, -1, 1).max(dim=1).values
            targets_seq = targets.reshape(1, -1, 1).max(dim=1).values

        # 2D: [T, C] -> one sequence, multiple classes
        elif predictions.ndim == 2:
            predictions_seq = predictions.unsqueeze(0).max(dim=1).values
            targets_seq = targets.unsqueeze(0).max(dim=1).values

        # 3D: [B, T, C] -> multiple sequences, multiple classes
        elif predictions.ndim == 3:
            predictions_seq = predictions.max(dim=1).values
            targets_seq = targets.max(dim=1).values

        else:
            raise ValueError(
                f"Expected predictions/targets with 1, 2, or 3 dimensions, got {predictions.ndim}"
            )

        num_classes = predictions_seq.shape[-1]
        entity_level_metrics = []

        for c in range(num_classes):
            pred_c = predictions_seq[:, c]
            target_c = targets_seq[:, c]

            tp = int(((pred_c == 1) & (target_c == 1)).sum().item())
            fp = int(((pred_c == 1) & (target_c == 0)).sum().item())
            fn = int(((pred_c == 0) & (target_c == 1)).sum().item())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1score = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            entity_level_metrics.append((tp, fp, fn, precision, recall, f1score))

        return entity_level_metrics

    def aggregate_sequence_metrics(self, entity_level_metrics):
        # ---- MICRO ----
        total_tp = sum(m[0] for m in entity_level_metrics)
        total_fp = sum(m[1] for m in entity_level_metrics)
        total_fn = sum(m[2] for m in entity_level_metrics)

        micro_precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        micro_recall = (
            total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        )
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )

        # ---- MACRO ----
        macro_precision = sum(m[3] for m in entity_level_metrics) / len(
            entity_level_metrics
        )
        macro_recall = sum(m[4] for m in entity_level_metrics) / len(
            entity_level_metrics
        )
        macro_f1 = sum(m[5] for m in entity_level_metrics) / len(entity_level_metrics)

        return {
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        }

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def __call__(self, eval_pred: EvalPrediction, *args, **kwargs):
        logits = torch.tensor(eval_pred.predictions)
        labels = torch.tensor(eval_pred.label_ids)
        _, _, C = logits.shape

        preds = self.predict(logits)
        sequence_level_metrics_per_label = self.compute_sequence_level_metrics(
            preds, labels
        )
        sequence_level_agg_metrics = self.aggregate_sequence_metrics(
            sequence_level_metrics_per_label
        )

        return {
            **sequence_level_agg_metrics,
            "metrics_per_entity": {
                self.id2label.get(label_idx, label_idx): {
                    "tp": sequence_level_metrics_per_label[label_idx][0],
                    "fp": sequence_level_metrics_per_label[label_idx][1],
                    "fn": sequence_level_metrics_per_label[label_idx][2],
                    "precision": sequence_level_metrics_per_label[label_idx][3],
                    "recall": sequence_level_metrics_per_label[label_idx][4],
                    "f1": sequence_level_metrics_per_label[label_idx][5],
                }
                for label_idx in range(C)
            },
        }


class BCESequenceLevelMetrics(BaseSequenceLevelMetrics):
    def __init__(
        self,
        id2label: dict[int, str] = None,
        device=None,
        prediction_thresholds: List[float] = None,
    ):
        super().__init__(id2label=id2label)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.prediction_thresholds = prediction_thresholds

    def predict(self, logits):
        probs = 1 / (1 + torch.exp(-logits))

        if self.prediction_thresholds is None:
            thresholds = torch.full(
                (logits.shape[-1],), 0.5, device=logits.device, dtype=logits.dtype
            )
        else:
            thresholds = torch.tensor(
                self.prediction_thresholds, device=logits.device, dtype=logits.dtype
            )

        preds = (probs > thresholds).float()
        return preds


class GroupedSoftmaxSequenceLevelMetrics(BaseSequenceLevelMetrics):
    def __init__(
        self,
        mutually_exclusive_classes: List[List[str]],
        id2label: dict[int, str] = None,
        device=None,
        prediction_thresholds: List[float] = None,
    ):
        super().__init__(id2label=id2label)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.exclusive_groups = [
            [self.label2id[label] for label in group]
            for group in mutually_exclusive_classes
        ]
        self.prediction_thresholds = prediction_thresholds

    def predict(self, logits):
        exclusive_idx = sorted({i for g in self.exclusive_groups for i in g})
        exclusive_mask = torch.zeros(
            logits.size(-1), dtype=torch.bool, device=logits.device
        ).to(self.device)
        exclusive_mask[exclusive_idx] = True

        preds = torch.zeros_like(logits)

        B, L, C = logits.shape

        # mutually exclusive labels
        for g in self.exclusive_groups:
            group_logits = torch.cat(
                [logits.new_zeros((B, L, 1)), logits[..., g]], dim=-1
            )
            group_preds = (
                group_logits.exp()
                / torch.logsumexp(group_logits, dim=-1).view(B, L, 1).exp()
            ).argmax(dim=-1)

            for g_idx, label_idx in enumerate(g):
                preds[..., label_idx] = (group_preds == g_idx + 1).float()

        # non-exclusive labels
        non_exclusive_logits = logits[..., ~exclusive_mask]
        non_exclusive_probs = 1 / (1 + torch.exp(-non_exclusive_logits))

        if self.prediction_thresholds is None:
            non_exclusive_thresholds = torch.full(
                (non_exclusive_logits.shape[-1],),
                0.5,
                device=logits.device,
                dtype=logits.dtype,
            )
        else:
            non_exclusive_thresholds = torch.tensor(
                self.prediction_thresholds,
                device=self.device,
                dtype=logits.dtype,
            )

        preds[..., ~exclusive_mask] = (
            non_exclusive_probs > non_exclusive_thresholds
        ).float()

        return preds
