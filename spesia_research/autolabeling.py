from typing import List, Literal
import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm

from .data_models import Annotation
from .datasets import (
    ClinicalRecordsDataset,
    DataCollatorForMultiLabelTokenClassification,
)


class AutoAnnotator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForTokenClassification,
        best_thresholds: list[float] = None,
        idx_to_label: dict[int, str] = None,
        rename_labels: dict[str, str] = None,
        max_length: int = 512,
        batch_size: int = 10,
        prediction_type: Literal["sigmoid", "softmax", "grouped_softmax"] = "sigmoid",
        mutually_exclusive_classes: list[list[str, str]] = None,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.best_thresholds = best_thresholds
        self.idx_to_label = idx_to_label
        self.prediction_type = prediction_type
        self.mutually_exclusive_classes = mutually_exclusive_classes

        if self.idx_to_label is None:
            self.idx_to_label = model.config.id2label

        if rename_labels is not None:
            self.idx_to_label = {
                k: rename_labels.get(v, v) for k, v in self.idx_to_label.items()
            }

        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_collator = DataCollatorForMultiLabelTokenClassification(
            pad_token_id=tokenizer.pad_token_id,
            max_length=self.max_length,
            num_labels=len(self.idx_to_label),
            device=self.device,
        )
        self.batch_size = batch_size

        if self.best_thresholds is None:
            self.best_thresholds = torch.tensor(
                [v["threshold"] for k, v in model.config.thresholds.items()]
            )

        if isinstance(self.best_thresholds, List):
            self.best_thresholds = torch.tensor(self.best_thresholds)

        self.best_thresholds = self.best_thresholds.to(self.device)

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str | Path, *args, **kwargs
    ) -> "AutoAnnotator":
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        return cls(tokenizer=tokenizer, model=model, *args, **kwargs)

    @torch.no_grad()
    def _predict(
        self,
        records: (
            transformers.tokenization_utils_base.BatchEncoding
            | list[transformers.tokenization_utils_base.BatchEncoding]
        ),
    ) -> np.ndarray:
        if isinstance(records, transformers.tokenization_utils_base.BatchEncoding):
            records = [records]

        if self.prediction_type == "sigmoid":
            output = self.model(**self.data_collator(records, include_labels=False))
            logits = output.logits.detach()
            probs = 1 / (1 + torch.exp(-logits))
            predictions = probs > self.best_thresholds
            del output, logits
            return predictions  # [batch_size, seq_len, num_labels]

        elif self.prediction_type == "grouped_softmax":
            assert self.mutually_exclusive_classes is not None, (
                f"For grouped_softmax, mutually_exclusive_classes must be provided. Got {self.mutually_exclusive_classes}."
            )
            # Compute logits
            output = self.model(**self.data_collator(records, include_labels=False))
            logits = output.logits.detach()

            # Create exclusive groups max
            exclusive_groups = [
                [self.model.config.label2id[label] for label in group]
                for group in self.mutually_exclusive_classes
            ]
            exclusive_idx = sorted({i for g in exclusive_groups for i in g})
            exclusive_mask = torch.zeros(
                logits.size(-1), dtype=torch.bool, device=logits.device
            ).to(self.device)
            exclusive_mask[exclusive_idx] = True

            # Initialize preds
            preds = torch.zeros_like(logits)

            # preds for mutually exclusive groups
            B, L, C = logits.shape
            for g in exclusive_groups:
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
            best_thresholds = [
                v["threshold"] for k, v in self.model.config.thresholds.items()
            ]
            non_exclusive_logits = logits[..., ~exclusive_mask]
            non_exclusive_probs = 1 / (1 + torch.exp(-non_exclusive_logits))
            non_exclusive_thresholds = torch.tensor(best_thresholds, device=self.device)
            preds[..., ~exclusive_mask] = (
                non_exclusive_probs > non_exclusive_thresholds
            ).float()

            return preds  # [batch_size, seq_len, num_labels]

    def _map_prediction_to_tokens(
        self,
        predictions: np.ndarray,  # [batch_size, seq_len, num_labels]
        records: List[transformers.tokenization_utils_base.BatchEncoding],
    ) -> list[list[Annotation]]:
        mapped_annotations = []

        for current_record, current_record_predictions in zip(records, predictions):
            annotations = []
            current_str_position = 0

            for token_index, pred in enumerate(current_record_predictions):
                text = self.tokenizer.decode(
                    current_record.input_ids[token_index], skip_special_tokens=True
                )
                if token_index >= len(current_record.input_ids):
                    break
                label_ids = torch.where(pred)[0]
                start_pos = current_str_position
                end_pos = current_str_position + len(text)
                for label_id in label_ids.tolist():
                    annotations.append(
                        Annotation(
                            id=str(token_index),
                            tags=set([self.idx_to_label[label_id]]),
                            start=start_pos,
                            end=end_pos,
                            text=text,
                        )
                    )
                current_str_position = end_pos

            mapped_annotations.append(annotations)

        return mapped_annotations

    def _merge_annotations(self, annotations: list[Annotation]):
        annotations = deepcopy(annotations)
        merged_annotations = {}
        for ann in annotations:
            for tag in ann.tags:
                if (ann.start, tag) in merged_annotations:
                    old_ann = merged_annotations[(ann.start, tag)]
                    new_ann = Annotation(
                        id=old_ann.id,
                        tags=set([tag]),
                        start=old_ann.start,
                        end=ann.end,
                        text=old_ann.text + ann.text,
                    )
                    merged_annotations[(new_ann.end, tag)] = new_ann
                    del merged_annotations[(ann.start, tag)]
                else:
                    new_ann = Annotation(
                        id=ann.id,
                        tags=set([tag]),
                        start=ann.start,
                        end=ann.end,
                        text=ann.text,
                    )
                    merged_annotations[(new_ann.end, tag)] = new_ann

        # fix token shift due to <bos> token
        annotations = list(merged_annotations.values())
        for ann in annotations:
            ann.end = ann.end - 1
            ann.start = ann.start - 1

        return annotations

    def annotate(
        self,
        dataset_path: Path | str = None,
        dataset: ClinicalRecordsDataset = None,
        texts: List[str] = None,
    ) -> ClinicalRecordsDataset:
        """
        Annotates the records in the dataset using the model.

        Args:
            dataset_path: Path to the dataset to annotate, or None to use the dataset provided in the constructor.
            dataset: ClinicalRecordsDataset to annotate, or None to use the dataset provided in the constructor.

        Returns:
            ClinicalRecordsDataset: The annotated dataset.

        Raises:
            ValueError: If neither dataset_path nor dataset is provided.
        """
        if dataset is not None:
            self.dataset = dataset
        elif dataset_path is not None:
            self.dataset = ClinicalRecordsDataset(
                dataset_path, tokenizer=self.tokenizer
            )
        elif texts is not None:
            self.dataset = ClinicalRecordsDataset.from_list_of_strings(
                texts, tokenizer=self.tokenizer
            )
        else:
            raise ValueError("Either dataset_path or dataset must be provided")

        # send model to GPU if available
        self.model.to(self.device)

        for idx in tqdm(
            range(0, len(self.dataset), self.batch_size),
            desc="Annotating records",
            total=(len(self.dataset)) // self.batch_size,
        ):
            batch = self.dataset[idx : idx + self.batch_size]
            predictions = self._predict(batch)
            annotations = self._map_prediction_to_tokens(predictions, batch)
            merged_annotations = [self._merge_annotations(ann) for ann in annotations]
            for record_idx in range(idx, min(idx + self.batch_size, len(self.dataset))):
                self.dataset.records[record_idx].annotations = merged_annotations[
                    record_idx - idx
                ]

        # send model back to CPU to free up memory and empty cache
        self.model.to("cpu")
        torch.cuda.empty_cache()

        return self.dataset
