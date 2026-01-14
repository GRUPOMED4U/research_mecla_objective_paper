from abc import ABC, abstractmethod
import copy
from pathlib import Path
import math
from collections import Counter
import random

from scipy import stats
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
)
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal
from datetime import datetime, timezone
import json
from nltk.translate.bleu_score import sentence_bleu

import yaml

from spesia_research.config import load_config
from spesia_research.datasets import (
    ClinicalRecordsDataset,
    DataCollatorForMultiLabelTokenClassification,
)
from spesia_research.metrics import compute_metrics
from spesia_research.trainers import EarlyStoppingCallback, MultiLabelTokenTrainer


class BaseDatasetEvaluator(ABC):
    def __init__(self, tokenizer_id: str = "jhu-clsp/mmBERT-base"):
        self.metrics = {}
        self.tokenizer_id = tokenizer_id
        self.sample_from_splits = {
            "train": 1.0,
            "val": 1.0,
            "test": 1.0,
        }

    @abstractmethod
    def evaluate(self, dataset_path: Path | str = None):
        pass


class DatasetEvaluatorForNER(BaseDatasetEvaluator):
    """
    Framework for evaluating NER datasets based on the paper https://doi.org/10.1017/nlp.2024.37
    """

    def _load_datasets_and_tokenizer(self, dataset_path: Path | str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, use_fast=True)

        self.splits: Dict[str, ClinicalRecordsDataset] = {
            "train": None,
            "val": None,
            "test": None,
        }
        for split in self.splits:
            self.splits[split] = ClinicalRecordsDataset(
                dataset_path,
                split=split,
                tokenizer=self.tokenizer,
                label_type=self.label_type,
                tags_to_consider=self.tags_to_consider,
                labels_to_ignore=self.labels_to_ignore,
                semantic_groups_to_consider=self.semantic_groups_to_consider,
                sample_from_split=self.sample_from_splits[split],
            )

        # empty dataset
        self.complete_dataset = ClinicalRecordsDataset(
            tokenizer=self.tokenizer,
            label_type=self.label_type,
            tags_to_consider=self.tags_to_consider,
            labels_to_ignore=self.labels_to_ignore,
            semantic_groups_to_consider=self.semantic_groups_to_consider,
        )
        # extend with every split
        for split in self.splits:
            self.complete_dataset.extend(self.splits[split])

    def _compute_redundancy(self) -> None:
        """
        **Redundancy**: reveal duplicates in the dataset. A smaller redundancy is better and indicates more diversity in the dataset.
        """
        self.metrics["redundancy"] = {}
        for split in self.splits:
            seen = set()
            duplicates = 0
            for record in self.splits[split].records:
                if record.text not in seen:
                    seen.add(record.text)
                else:
                    duplicates += 1
            self.metrics["redundancy"][split] = {
                "duplicates": duplicates,
                "total": len(self.splits[split].records),
                "redundancy": duplicates / len(self.splits[split].records),
            }

    def _compute_leakage_ratio(self) -> None:
        """
        **Leakage Ratio**: how many instances in the test set (Te) have incorrectly appeared in the training set (Tr) or development set (De). A lower Leakage Ratio is indicative of better dataset partitioning as it suggests that there is minimal to no overlap between the sets.
        """
        # set with train and val records
        train_val_records_set = set(
            [record.text for record in self.splits["train"].records]
            + [record.text for record in self.splits["val"].records]
        )

        n_leaked_records = 0
        for record in self.splits["test"].records:
            if record.text in train_val_records_set:
                n_leaked_records += 1

        self.metrics["leakage_ratio"] = {
            "n_leaked_records": n_leaked_records,
            "total": len(self.splits["test"].records),
            "leakage_ratio": n_leaked_records / len(self.splits["test"].records),
        }

    def _compute_unseen_entity_ratio(self) -> None:
        """
        **Unseen Entity Ratio**: proportion of new entities in the test set labels that are not present in the training set, promoting the model's ability to generalize. A higher Unseen Entity Ratio is desirable as it indicates a greater challenge for the model to recognize entities it has not encountered during training.
        """
        # train entities set -> Tuple[label, text]
        train_entities_set = set()
        for record in self.splits["train"].records:
            for annotation in record.annotations:
                for tag in getattr(annotation, self.label_type):
                    train_entities_set.add((tag, annotation.text))

        n_unseen_entities = 0
        n_test_entities = 0

        # initialize unseen_entities_per_class
        unseen_entities_per_class = {}

        # gather n_unseen per class and class total
        for record in self.splits["test"].records:
            for annotation in record.annotations:
                for tag in getattr(annotation, self.label_type):
                    if tag not in unseen_entities_per_class:
                        unseen_entities_per_class[tag] = {
                            "n_unseen_entities": 0,
                            "total": 0,
                        }
                    unseen_entities_per_class[tag]["total"] += 1
                    entity = (tag, annotation.text)
                    n_test_entities += 1
                    if entity not in train_entities_set:
                        n_unseen_entities += 1
                        unseen_entities_per_class[tag]["n_unseen_entities"] += 1

        # compute ratios
        for tag in unseen_entities_per_class:
            if unseen_entities_per_class[tag]["total"] == 0:
                unseen_entities_per_class[tag]["unseen_entity_ratio"] = 0
                continue

            unseen_entities_per_class[tag]["unseen_entity_ratio"] = (
                unseen_entities_per_class[tag]["n_unseen_entities"]
                / unseen_entities_per_class[tag]["total"]
            )

        self.metrics["unseen_entity_ratio"] = {
            "n_unseen_entities": n_unseen_entities,
            "total": n_test_entities,
            "unseen_entity_ratio": n_unseen_entities / n_test_entities,
            "unseen_entities_per_class": unseen_entities_per_class,
        }

    def _compute_entity_ambiguity_degree(self) -> None:
        """
        **Entity Ambiguity Degree**: measure how many entities are labeled with more than one kind of entities types. For example, if “apple” is labeled as “Fruit” in one instance and labeled as “Company” in another instance, then there is a conflict in D. A higher Entity Ambiguity Degree represents a more challenging dataset because it indicates more instances where an entity is labeled with different types, thereby confusing NER models. $e^{*}(D)$ represents the number of conflict entities in dataset D.
        """
        self.metrics["entity_ambiguity_degree"] = {}
        entity_ambiguity_track = {}
        entity_ambiguity_track_per_label = {}
        n_records_per_label = {}

        for record in self.complete_dataset.records:
            for annotation in record.annotations:
                # initialize object to track entity text and associated tags
                if annotation.text not in entity_ambiguity_track:
                    entity_ambiguity_track[annotation.text] = {
                        "entities": [],
                        "tags": set(),
                    }

                for tag in getattr(annotation, self.label_type):
                    # initialize object to track entity text per tag
                    if tag not in entity_ambiguity_track_per_label:
                        entity_ambiguity_track_per_label[tag] = {}
                        n_records_per_label[tag] = 0

                    # for each tag, keep track of entity text
                    if annotation.text not in entity_ambiguity_track_per_label[tag]:
                        entity_ambiguity_track_per_label[tag][annotation.text] = {
                            "entities": [],
                            "tags": set(),
                        }

                    # append entities and tags to global ambiguity object
                    entity_ambiguity_track[annotation.text]["entities"].append(
                        (tag, annotation.text)
                    )
                    entity_ambiguity_track[annotation.text]["tags"].add(tag)

                    # append entities and tags to ambiguity object per label
                    entity_ambiguity_track_per_label[tag][annotation.text][
                        "entities"
                    ].append((tag, annotation.text))
                    entity_ambiguity_track_per_label[tag][annotation.text]["tags"].add(
                        tag
                    )

            for tag in getattr(record, self.label_type):
                n_records_per_label[tag] += 1

        self.metrics["entity_ambiguity_degree"] = {
            "n_conflict_entities": 0,
            "total_unique_entities": len(entity_ambiguity_track),
            "entity_ambiguity_degree": 0,
            "n_records": len(self.complete_dataset.records),
        }

        for _, entity_data in entity_ambiguity_track.items():
            if len(entity_data["entities"]) > 1:
                self.metrics["entity_ambiguity_degree"]["n_conflict_entities"] += 1

        self.metrics["entity_ambiguity_degree"]["entity_ambiguity_degree"] = (
            self.metrics["entity_ambiguity_degree"]["n_conflict_entities"]
            / len(self.complete_dataset.records)
        )

        # compute entity ambiguity degree per label
        for label in entity_ambiguity_track_per_label:
            self.metrics["entity_ambiguity_degree"][label] = {}
            self.metrics["entity_ambiguity_degree"][label]["n_conflict_entities"] = 0
            self.metrics["entity_ambiguity_degree"][label]["total_unique_entities"] = (
                len(entity_ambiguity_track_per_label[label])
            )

            for _, entity_data in entity_ambiguity_track_per_label[label].items():
                if len(entity_data["entities"]) > 1:
                    self.metrics["entity_ambiguity_degree"][label][
                        "n_conflict_entities"
                    ] += 1

            self.metrics["entity_ambiguity_degree"][label][
                "entity_ambiguity_degree"
            ] = self.metrics["entity_ambiguity_degree"][label][
                "n_conflict_entities"
            ] / len(self.complete_dataset.records)

    def _compute_entity_density(self) -> None:
        """
        **Text Complexity**: Entity Density in sentences within the dataset. Higher Text Complexity signals a more difficult dataset because it implies that sentences are densely packed with entities, requiring more nuanced understanding and recognition by the model.
        """
        entity_density = 0
        for i, record in enumerate(self.complete_dataset):
            m_i = len(record["input_ids"])
            e_y_i = record["labels"].sum().item()
            entity_density += e_y_i / m_i

        entity_density = entity_density / len(self.complete_dataset)
        self.metrics["entity_density"] = entity_density

    def _compute_entity_imbalance_degree(self) -> None:
        """
        **Entity Imbalance Degree**:  measures the unevenness of the distribution of different entities in D. A lower Entity Imbalance Degree is better as it indicates a more balanced distribution of entity types, which is desirable for ensuring that the model is equally exposed to all categories and does not develop a bias toward the more frequent ones.
        """

        labels = [record["labels"] for record in self.complete_dataset]
        concat_labels = torch.cat(labels, dim=0)

        if concat_labels.shape[-1] == 1:
            self.metrics["entity_imbalance_degree"] = 0.0
            return

        self.metrics["entity_imbalance_degree"] = concat_labels.mean(dim=0).std().item()

    def _compute_entity_null_rate(self) -> None:
        """
        **Entity-Null Rate**: evaluates the proportion of instances in the dataset that do not contain any entity. A lower Entity-Null Rate is preferred because it suggests that the dataset contains a richer set of examples for the model to learn from, with more instances that include entity information.
        """
        self.metrics["entity_null_rate"] = sum(
            [
                1 if record["labels"].sum().item() == 0 else 0
                for record in self.complete_dataset
            ]
        ) / len(self.complete_dataset)

    def _dont_compute_entity_self_bleu(self) -> None:
        """
        Calculates Self-BLEU: Diversity intra-class.
        """
        global_self_bleu = None
        entity_self_bleu = {}
        entities_per_class = {}

        # Collect entities for each class
        for record in self.complete_dataset.records:
            for label in getattr(record, self.label_type):
                if label not in entities_per_class:
                    entities_per_class[label] = []
                entities_per_class[label].append(record.text)

        # Compute self bleu per label
        for label, entities in tqdm(entities_per_class.items(), desc="Self-BLEU"):
            print("Calculating self bleu for label:", label)
            entity_self_bleu[label] = self._calculate_self_bleu_for_single_class(
                entities
            )

        # Compute global self bleu average
        if len(entity_self_bleu) > 0:
            global_self_bleu = sum(entity_self_bleu.values()) / len(entity_self_bleu)

        self.metrics["entity_self_bleu"] = entity_self_bleu
        self.metrics["global_self_bleu"] = global_self_bleu

    def _get_ngrams(self, tokens, n):
        """Generates a list of n-grams from a list of tokens."""
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def _modified_precision(self, candidate, references, n):
        """Calculates modified n-gram precision."""
        counts = Counter(self._get_ngrams(candidate, n))
        if not counts:
            return 0

        max_counts = {}
        for ref in references:
            ref_counts = Counter(self._get_ngrams(ref, n))
            for ngram in counts:
                max_counts[ngram] = max(
                    max_counts.get(ngram, 0), ref_counts.get(ngram, 0)
                )

        clipped_counts = {
            ngram: min(count, max_counts.get(ngram, 0))
            for ngram, count in counts.items()
        }
        return sum(clipped_counts.values()) / sum(counts.values())

    def _calculate_bleu_score(
        self, candidate, references, weights=(0.25, 0.25, 0.25, 0.25)
    ):
        """Calculates the BLEU score for a single candidate against references."""
        p_n = []
        for i in range(1, len(weights) + 1):
            precision = self._modified_precision(candidate, references, i)
            # Avoid math domain error with log(0) by using a tiny smoothing value
            p_n.append(max(precision, 1e-9))

        # Geometric mean of precisions
        score = math.exp(sum(w * math.log(p) for w, p in zip(weights, p_n)))

        # Brevity Penalty
        c = len(candidate)
        # Find the reference length closest to the candidate length
        r = min((abs(len(ref) - c), len(ref)) for ref in references)[1]

        bp = 1 if c > r else math.exp(1 - r / c)
        return bp * score

    def _calculate_self_bleu_for_single_class(self, texts):
        """Calculates Self-BLEU: Diversity intra-class."""
        # 1. Tokenization
        tokenized_texts = [
            self.tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts
        ]

        scores = []
        # 2. Sample examples to limit computation time
        for i in random.choices(range(len(tokenized_texts)), k=100):
            # Leave-one-out: One is the candidate, the rest are references
            candidate = tokenized_texts[i]
            references = tokenized_texts[:i] + tokenized_texts[i + 1 :]

            # Calculate BLEU for this candidate
            scores.append(self._calculate_bleu_score(candidate, references))

        return sum(scores) / len(scores)

    def _calculate_bleu_score_nltk(self, tokenized_list) -> float:
        bleu_scores = []

        for i in range(len(tokenized_list)):
            # 1. Candidate is the current entity
            candidate = tokenized_list[i]

            # 2. References are all other entities in the list
            references = tokenized_list[:i] + tokenized_list[i + 1 :]

            # 3. Calculate BLEU
            score = sentence_bleu(references, candidate)
            bleu_scores.append(score)

        return sum(bleu_scores) / len(bleu_scores)

    def _generate_report(self, output_path: Path = None) -> None:
        """
        Generates a human-readable markdown report containing all of the computed metrics.
        Works with any label set (tables are built from whatever labels appear in the metrics dicts).
        """
        import json
        from datetime import datetime, timezone

        if output_path is None:
            output_path = Path("datasets", "reports")

        report_base_path = output_path
        report_base_path.mkdir(exist_ok=True, parents=True)

        report_path = (
            report_base_path / f"{self.dataset_name}_{self.label_type}_report.md"
        )

        def _fmt_float(x, nd=4):
            try:
                return f"{float(x):.{nd}f}"
            except Exception:
                return str(x)

        def _fmt_pct(x, nd=2):
            try:
                return f"{100.0 * float(x):.{nd}f}%"
            except Exception:
                return str(x)

        def _json_pretty(obj) -> str:
            # fallback-safe pretty print
            try:
                return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
            except Exception:
                return str(obj)

        def _md_table(headers, rows):
            # headers: list[str], rows: list[list[str]]
            out = []
            out.append("| " + " | ".join(headers) + " |")
            out.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for r in rows:
                out.append("| " + " | ".join(r) + " |")
            return "\n".join(out)

        # ---------- basic dataset info ----------
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        split_sizes = {}
        if hasattr(self, "splits") and isinstance(self.splits, dict):
            for split, ds in self.splits.items():
                if ds is None:
                    continue
                try:
                    split_sizes[split] = len(ds.records)
                except Exception:
                    split_sizes[split] = None

        n_total = None
        try:
            n_total = len(self.complete_dataset.records)
        except Exception:
            pass

        with open(report_path, "w", encoding="utf-8") as f:
            # Title + metadata
            f.write(f"# Dataset Evaluation Report\n\n")
            f.write(f"- **Dataset:** `{self.dataset_path}`\n")
            f.write(f"- **Tokenizer:** `{self.tokenizer_id}`\n")
            f.write(f"- **Generated:** {now_utc}\n")
            if n_total is not None:
                f.write(f"- **Total records (all splits combined):** {n_total}\n")
            if split_sizes:
                f.write(
                    "- **Split sizes:** "
                    + ", ".join([f"{k}={v}" for k, v in split_sizes.items()])
                    + "\n"
                )
            f.write("\n---\n\n")

            # ---------- Summary (quick read) ----------
            f.write("## Summary\n\n")
            # Pull the main metrics if present
            leakage = self.metrics.get("leakage_ratio", {})
            redundancy = self.metrics.get("redundancy", {})
            null_rate = self.metrics.get("entity_null_rate", None)
            density = self.metrics.get("entity_density", None)
            imbalance = self.metrics.get("entity_imbalance_degree", None)
            unseen = self.metrics.get("unseen_entity_ratio", {})
            ambiguity = self.metrics.get("entity_ambiguity_degree", {})

            summary_rows = []

            if isinstance(leakage, dict) and "leakage_ratio" in leakage:
                summary_rows.append(
                    [
                        "Leakage ratio (test in train/val)",
                        _fmt_pct(leakage["leakage_ratio"]),
                    ]
                )
            if isinstance(null_rate, (int, float)):
                summary_rows.append(
                    ["Entity-null rate (no entities)", _fmt_pct(null_rate)]
                )
            if isinstance(density, (int, float)):
                summary_rows.append(
                    ["Entity density (avg entities per token)", _fmt_float(density, 6)]
                )
            if isinstance(imbalance, (int, float)):
                summary_rows.append(
                    ["Entity imbalance (std across labels)", _fmt_float(imbalance, 6)]
                )
            if isinstance(unseen, dict) and "unseen_entity_ratio" in unseen:
                summary_rows.append(
                    [
                        "Unseen entity ratio (test entities not in train)",
                        _fmt_pct(unseen["unseen_entity_ratio"]),
                    ]
                )
            if isinstance(ambiguity, dict) and "entity_ambiguity_degree" in ambiguity:
                summary_rows.append(
                    [
                        "Entity ambiguity degree (conflicts / records)",
                        _fmt_float(ambiguity["entity_ambiguity_degree"], 6),
                    ]
                )

            if summary_rows:
                f.write(_md_table(["Metric", "Value"], summary_rows) + "\n\n")
            else:
                f.write("_No summary metrics found._\n\n")

            f.write("---\n\n")

            # ---------- Redundancy ----------
            if isinstance(redundancy, dict) and redundancy:
                f.write("## Redundancy\n\n")
                rows = []
                for split, info in redundancy.items():
                    if not isinstance(info, dict):
                        continue
                    rows.append(
                        [
                            str(split),
                            str(info.get("total", "")),
                            str(info.get("duplicates", "")),
                            _fmt_pct(info.get("redundancy", "")),
                        ]
                    )
                if rows:
                    rows = sorted(rows, key=lambda r: r[0])
                    f.write(
                        _md_table(["Split", "Total", "Duplicates", "Redundancy"], rows)
                        + "\n\n"
                    )
                else:
                    f.write(
                        "_Redundancy metric present but not in expected format._\n\n"
                    )
                f.write("---\n\n")

            # ---------- Leakage ----------
            if isinstance(leakage, dict) and leakage:
                f.write("## Leakage\n\n")
                f.write(
                    f"- **Leaked test records:** {leakage.get('n_leaked_records', 'NA')} / {leakage.get('total', 'NA')} "
                    f"(**{_fmt_pct(leakage.get('leakage_ratio', 'NA'))}**)\n\n"
                )
                f.write("---\n\n")

            # ---------- Entity presence / density / imbalance ----------
            f.write("## Entity Presence and Distribution\n\n")
            bullets = []
            if isinstance(null_rate, (int, float)):
                bullets.append(
                    f"- **Entity-null rate:** {_fmt_pct(null_rate)} (fraction of records with zero entities)"
                )
            if isinstance(density, (int, float)):
                bullets.append(
                    f"- **Entity density:** {_fmt_float(density, 6)} (average entity labels per token)"
                )
            if isinstance(imbalance, (int, float)):
                bullets.append(
                    f"- **Entity imbalance degree:** {_fmt_float(imbalance, 6)} (std of label frequency across labels)"
                )
            if bullets:
                f.write("\n".join(bullets) + "\n\n")
            else:
                f.write("_No entity presence/distribution metrics found._\n\n")
            f.write("---\n\n")

            # ---------- Unseen entity ratio ----------
            if isinstance(unseen, dict) and unseen:
                f.write("## Unseen Entities (Test vs Train)\n\n")
                if "unseen_entity_ratio" in unseen:
                    f.write(
                        f"- **Overall:** {unseen.get('n_unseen_entities', 'NA')} / {unseen.get('total', 'NA')} "
                        f"(**{_fmt_pct(unseen.get('unseen_entity_ratio', 'NA'))}**)\n\n"
                    )

                per_class = unseen.get("unseen_entities_per_class", {})
                if isinstance(per_class, dict) and per_class:
                    rows = []
                    for label, info in per_class.items():
                        if not isinstance(info, dict):
                            continue
                        rows.append(
                            [
                                str(label),
                                str(info.get("total", "")),
                                str(info.get("n_unseen_entities", "")),
                                _fmt_pct(info.get("unseen_entity_ratio", "")),
                            ]
                        )

                    # sort by unseen ratio desc when possible
                    def _sort_key(r):
                        try:
                            return float(r[3].replace("%", ""))  # percent string
                        except Exception:
                            return -1.0

                    rows = sorted(rows, key=_sort_key, reverse=True)
                    f.write("### Per-label breakdown\n\n")
                    f.write(
                        _md_table(
                            [
                                "Label",
                                "Total entities",
                                "Unseen entities",
                                "Unseen ratio",
                            ],
                            rows,
                        )
                        + "\n\n"
                    )
                else:
                    f.write("_No per-label unseen-entity breakdown available._\n\n")
                f.write("---\n\n")

            # ---------- Entity ambiguity degree ----------
            if isinstance(ambiguity, dict) and ambiguity:
                f.write("## Entity Ambiguity (Same surface form, different labels)\n\n")
                if (
                    "n_conflict_entities" in ambiguity
                    and "total_unique_entities" in ambiguity
                ):
                    f.write(
                        f"- **Conflicting entities:** {ambiguity.get('n_conflict_entities', 'NA')} / {ambiguity.get('total_unique_entities', 'NA')} unique entities\n"
                    )
                if "entity_ambiguity_degree" in ambiguity:
                    f.write(
                        f"- **Ambiguity degree:** {_fmt_float(ambiguity.get('entity_ambiguity_degree', 'NA'), 6)} (conflicts / records)\n"
                    )
                if "n_records" in ambiguity:
                    f.write(f"- **Records:** {ambiguity.get('n_records', 'NA')}\n")
                f.write("\n")

                # Per-label ambiguity lives as sibling keys in your current structure
                per_label_rows = []
                for k, v in ambiguity.items():
                    if k in {
                        "n_conflict_entities",
                        "total_unique_entities",
                        "entity_ambiguity_degree",
                        "n_records",
                    }:
                        continue
                    if isinstance(v, dict) and "entity_ambiguity_degree" in v:
                        per_label_rows.append(
                            [
                                str(k),
                                str(v.get("total_unique_entities", "")),
                                str(v.get("n_conflict_entities", "")),
                                _fmt_float(v.get("entity_ambiguity_degree", ""), 6),
                            ]
                        )

                if per_label_rows:
                    # sort by ambiguity degree desc
                    def _sort_key2(r):
                        try:
                            return float(r[3])
                        except Exception:
                            return -1.0

                    per_label_rows = sorted(
                        per_label_rows, key=_sort_key2, reverse=True
                    )
                    f.write("### Per-label breakdown\n\n")
                    f.write(
                        _md_table(
                            [
                                "Label",
                                "Unique entities",
                                "Conflicting entities",
                                "Ambiguity degree",
                            ],
                            per_label_rows,
                        )
                        + "\n\n"
                    )

                f.write("---\n\n")

            # ---------- Appendix: raw metrics (optional, but nicely formatted) ----------
            f.write("## Appendix: Raw Metrics (for debugging)\n\n")
            f.write(
                "If you need the exact raw structures, they are preserved below.\n\n"
            )
            for metric_name, metric_value in self.metrics.items():
                f.write(f"### {metric_name}\n\n")
                f.write("```json\n")
                f.write(_json_pretty(metric_value))
                f.write("\n```\n\n")

        print(f"Report generated at: {report_path.resolve()}")

    def _load_training_config(self, path: Path) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)

        cfg = {}
        if p.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML not installed. pip install pyyaml")
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

        if p.suffix.lower() == ".json":
            cfg = json.loads(p.read_text(encoding="utf-8"))

        cfg["run_name"] = cfg.get("run_name", Path(path).stem)
        return cfg

    def _train_token_classification_model(self, cfg: dict):
        training_args = TrainingArguments(
            output_dir="output",
            **cfg["training_args"],
        )
        model = AutoModelForTokenClassification.from_pretrained(
            cfg["model_id"], num_labels=self.splits["train"].num_labels
        )
        tokenizer = self.tokenizer

        data_collator = DataCollatorForMultiLabelTokenClassification(
            pad_token_id=tokenizer.pad_token_id,
            max_length=cfg["dataset_args"]["max_length"],
            num_labels=self.splits["train"].num_labels,
        )
        # Load trainer
        trainer = MultiLabelTokenTrainer(
            model=model,
            args=training_args,
            train_dataset=self.splits["train"],
            eval_dataset=(
                self.splits["val"]
                if self.splits["val"].split_ratio["val"] > 0
                else self.splits["test"]
            ),
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(patience=cfg["early_stopping_patience"])],
            pos_weight=self.splits["train"].pos_weight,
            **cfg.get("trainer_args", {}),
        )
        trainer.train()
        trainer.eval_dataset = self.splits["test"]
        test_set_metrics = trainer.evaluate()
        test_set_metrics["run_name"] = cfg["run_name"]
        test_set_metrics.update(
            {
                "model_id": cfg["model_id"],
                "batch_effective": cfg["training_args"]["per_device_train_batch_size"]
                * cfg["training_args"]["gradient_accumulation_steps"],
                "patience": cfg["early_stopping_patience"],
                "pos_weight": self.splits["train"].pos_weight.tolist(),
            }
        )
        test_set_metrics["annotation_type"] = cfg["dataset_args"]["annotation_scheme"]
        pred_output = trainer.predict(self.splits["test"])
        probs = 1 / (1 + np.exp(-pred_output.predictions))
        labels = pred_output.label_ids.astype(int)
        threshold_map = {}

        # For single-label: flatten arrays
        if probs.shape[-1] == 1:
            probs_flat = probs.reshape(-1)
            labels_flat = labels.reshape(-1)
        else:
            # For multilabel: plot for each label
            probs_flat = probs.reshape(-1, probs.shape[-1])
            labels_flat = labels.reshape(-1, labels.shape[-1])

        # --- Precision–Recall ---
        for i in range(probs_flat.shape[-1] if probs_flat.ndim > 1 else 1):
            y_score = probs_flat[:, i] if probs_flat.ndim > 1 else probs_flat
            y_true = labels_flat[:, i] if probs_flat.ndim > 1 else labels_flat
            precision, recall, thresholds = precision_recall_curve(y_true, y_score)
            pr_auc = average_precision_score(y_true, y_score)
            threshold_map[self.splits["test"].labels_to_consider[i]] = {
                "threshold": thresholds,
                "precision": precision,
                "recall": recall,
                "pr_auc": pr_auc,
            }
        test_set_metrics["threshold_map"] = threshold_map
        macro_ap = average_precision_score(labels_flat, probs_flat, average="macro")
        micro_ap = average_precision_score(labels_flat, probs_flat, average="micro")
        test_set_metrics["macro_ap"] = macro_ap
        test_set_metrics["micro_ap"] = micro_ap
        self.metrics["test_set_metrics"] = test_set_metrics

    def _process_metrics(self) -> pd.DataFrame:
        processed_metrics = []

        # general info
        general_info = {
            "dataset_name": self.metrics["dataset_name"],
            "label_type": self.metrics["label_type"],
            "num_classes": len(self.metrics["entities"]),
            "train_dataset_size": self.metrics["train_dataset_size"],
            "val_dataset_size": self.metrics["val_dataset_size"],
            "test_dataset_size": self.metrics["test_dataset_size"],
        }

        # extract dataset specific metrics
        dataset_specific_metrics = {}
        dataset_specific_metrics["entity_ambiguity_degree"] = self.metrics[
            "entity_ambiguity_degree"
        ]["entity_ambiguity_degree"]
        dataset_specific_metrics["entity_density"] = self.metrics["entity_density"]
        dataset_specific_metrics["entity_imbalance_degree"] = self.metrics[
            "entity_imbalance_degree"
        ]
        dataset_specific_metrics["entity_null_rate"] = self.metrics["entity_null_rate"]
        dataset_specific_metrics["dataset_name"] = self.metrics["dataset_name"]
        dataset_specific_metrics["leakage_ratio"] = self.metrics["leakage_ratio"][
            "leakage_ratio"
        ]
        dataset_specific_metrics["redundancy_train"] = self.metrics["redundancy"][
            "train"
        ]["redundancy"]
        dataset_specific_metrics["redundancy_test"] = self.metrics["redundancy"][
            "test"
        ]["redundancy"]
        dataset_specific_metrics["unseen_entity_ratio"] = self.metrics[
            "unseen_entity_ratio"
        ]["unseen_entity_ratio"]

        # general performance metrics
        general_performance_metrics = {
            "macro_ap": self.metrics["test_set_metrics"]["macro_ap"],
            "micro_ap": self.metrics["test_set_metrics"]["micro_ap"],
        }

        # extract entity specific metrics
        for e in self.metrics["entities"]:
            entity_specific_metrics = {}
            entity_specific_metrics["entity_name"] = e
            if e in self.metrics["entity_ambiguity_degree"]:
                entity_specific_metrics["entity_specific_ambiguity_degree"] = (
                    self.metrics["entity_ambiguity_degree"][e][
                        "entity_ambiguity_degree"
                    ]
                )

            if e in self.metrics["unseen_entity_ratio"]["unseen_entities_per_class"]:
                entity_specific_metrics["entity_specific_unseen_ratio"] = self.metrics[
                    "unseen_entity_ratio"
                ]["unseen_entities_per_class"][e]["unseen_entity_ratio"]

            if e in self.metrics["test_set_metrics"]["threshold_map"]:
                entity_specific_metrics["entity_specific_pr_auc"] = self.metrics[
                    "test_set_metrics"
                ]["threshold_map"][e]["pr_auc"]

            processed_metrics.append(
                {
                    **general_info,
                    **dataset_specific_metrics,
                    **general_performance_metrics,
                    **entity_specific_metrics,
                }
            )

        return pd.DataFrame(processed_metrics)

    def evaluate(
        self,
        dataset_path: Path | str,
        label_type: Literal["tags", "semantic_groups"] = "tags",
        dataset_name: str = None,
        labels_to_consider: List[str] = None,
        labels_to_ignore: List[str] = None,
        generate_eval_report: bool = True,
        train_model: bool = False,
        training_config_path: Path | str = None,
        sample_from_splits: Dict[str, float] = None,
        output_path: Path | None = None,
        **kwargs,
    ):
        """
        Loads datasets and tokenizer and runs all methods that start with `_compute_`
        """

        if sample_from_splits is not None:
            self.sample_from_splits.update(sample_from_splits)

        # Prepare directory
        if output_path is not None:
            analysis_results_path = output_path / "analysis_results.csv"
            analysis_results_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            analysis_results_path = Path("analysis_results.csv")
            analysis_results_path.parent.mkdir(parents=True, exist_ok=True)

        if not analysis_results_path.exists():
            results_df = pd.DataFrame()
        else:
            results_df = pd.read_csv(analysis_results_path)

        # Set attributes
        self.dataset_path = (
            Path(dataset_path) if isinstance(dataset_path, str) else dataset_path
        )
        self.dataset_name = dataset_name if dataset_name else self.dataset_path.stem

        # Skip if dataset has already been evaluated
        if (
            len(results_df) > 0
            and self.dataset_name in results_df["dataset_name"].unique()
        ):
            print(
                f"Dataset {self.dataset_name} has already been evaluated. Skipping..."
            )
            return

        self.label_type = label_type
        self.tags_to_consider = None
        self.semantic_groups_to_consider = None
        self.labels_to_ignore = labels_to_ignore
        setattr(self, f"{label_type}_to_consider", labels_to_consider)
        self._load_datasets_and_tokenizer(dataset_path)
        self.metrics = {
            "dataset_name": self.dataset_name,
            "entities": getattr(
                self.complete_dataset, f"{self.label_type}_to_consider"
            ),
            "label_type": self.label_type,
            "train_dataset_size": len(self.splits["train"]),
            "val_dataset_size": len(self.splits["val"]),
            "test_dataset_size": len(self.splits["test"]),
        }

        # run all methods that start with `_compute_`
        for method in dir(self):
            if method.startswith("_compute_"):
                getattr(self, method)()

        if train_model:
            assert training_config_path is not None
            cfg = self._load_training_config(training_config_path)
            self._train_token_classification_model(cfg)
            results_df = pd.concat([results_df, self._process_metrics()], axis=0)
            results_df.to_csv(analysis_results_path, index=False)

        if generate_eval_report:
            self._generate_report(output_path=output_path)

    def compare(
        self,
        datasets_args: List[dict],
        output_path: Path | None = None,
        generate_eval_report: bool = True,
        generate_comparison_report: bool = True,
        train_model: bool = False,
        training_config_path: Path | str = None,
    ):
        """
        Loads datasets and tokenizer and runs all methods that start with `_compute_`. Creates a single object comparing overall metrics from multiple datasets.
        """
        self.combined_metrics = []
        self.dataset_names = []
        self.dataset_args = datasets_args

        for ds in tqdm(self.dataset_args, desc="Comparing datasets"):
            self.evaluate(
                **ds,
                generate_eval_report=generate_eval_report,
                train_model=train_model,
                training_config_path=training_config_path,
                output_path=output_path,
            )
            self.dataset_names.append(self.dataset_name)
            self.combined_metrics.append(copy.deepcopy(self.metrics))

        if output_path is not None:
            if isinstance(output_path, str):
                output_path = Path(output_path)
        else:
            output_path = Path("datasets/reports/dataset_comparison_report.md")

        if generate_comparison_report:
            self.generate_comparison_report(
                self.combined_metrics,
                self.dataset_names,
                output_path / "comparison_report.md",
                top_k=10,
            )

        return self.combined_metrics

    @staticmethod
    def generate_comparison_report(
        metrics_list: Sequence[Dict[str, Any]],
        dataset_names: Optional[Sequence[str]] = None,
        output_path: Optional[Path] = None,
        top_k: int = 10,
    ) -> Path:
        """
        Generate a Markdown comparison report across multiple datasets.

        This method is label-agnostic: it never assumes the same label set across datasets.
        For per-label metrics, it reports distribution summaries and top-K labels within each dataset.

        Parameters
        ----------
        metrics_list:
            List of metrics dicts produced by DatasetEvaluatorForNER (one per dataset).
        dataset_names:
            Optional list of dataset names (same length as metrics_list). If None, uses "Dataset 1", ...
        output_path:
            Where to write the markdown. If None, writes "dataset_comparison_report.md" in CWD.
        top_k:
            Top-K labels to show for per-label tables (unseen ratio, ambiguity degree).

        Returns
        -------
        Path to the generated markdown file.
        """
        if dataset_names is None:
            dataset_names = [f"Dataset {i + 1}" for i in range(len(metrics_list))]
        if len(dataset_names) != len(metrics_list):
            raise ValueError("dataset_names must have the same length as metrics_list")

        if output_path is None:
            output_path = Path("dataset_comparison_report.md")
        else:
            output_path = Path(output_path)

        # ---------------- helpers ----------------
        def _safe_float(x) -> Optional[float]:
            try:
                if x is None:
                    return None
                v = float(x)
                if math.isnan(v) or math.isinf(v):
                    return None
                return v
            except Exception:
                return None

        def _fmt_float(x: Any, nd: int = 6) -> str:
            v = _safe_float(x)
            return f"{v:.{nd}f}" if v is not None else "NA"

        def _fmt_pct(x: Any, nd: int = 2) -> str:
            v = _safe_float(x)
            return f"{100.0 * v:.{nd}f}%" if v is not None else "NA"

        def _md_table(headers: List[str], rows: List[List[str]]) -> str:
            out = []
            out.append("| " + " | ".join(headers) + " |")
            out.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for r in rows:
                out.append("| " + " | ".join(r) + " |")
            return "\n".join(out)

        def _quantiles(values: List[float]) -> Dict[str, float]:
            if not values:
                return {}
            values_sorted = sorted(values)
            n = len(values_sorted)

            def q(p: float) -> float:
                # simple nearest-rank
                idx = max(0, min(n - 1, int(math.ceil(p * n)) - 1))
                return values_sorted[idx]

            return {
                "count": float(n),
                "mean": sum(values_sorted) / n,
                "median": q(0.5),
                "p90": q(0.9),
                "max": values_sorted[-1],
            }

        def _json_pretty(obj: Any) -> str:
            try:
                return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
            except Exception:
                return str(obj)

        # ---------------- extract “overview” metrics ----------------
        # These are always comparable across datasets (no labels involved).
        overview_rows: List[List[str]] = []
        overview_headers = [
            "Dataset",
            "Entity null rate",
            "Entity density",
            "Imbalance degree",
            "Unseen entity ratio",
            "Ambiguity degree",
            "Leakage ratio (test)",
        ]

        for name, m in zip(dataset_names, metrics_list):
            null_rate = m.get("entity_null_rate")
            density = m.get("entity_density")
            imbalance = m.get("entity_imbalance_degree")

            unseen = m.get("unseen_entity_ratio") or {}
            unseen_ratio = unseen.get("unseen_entity_ratio")

            amb = m.get("entity_ambiguity_degree") or {}
            amb_degree = amb.get("entity_ambiguity_degree")

            leak = m.get("leakage_ratio") or {}
            leak_ratio = leak.get("leakage_ratio")

            overview_rows.append(
                [
                    str(name),
                    _fmt_pct(null_rate),
                    _fmt_float(density, 8),
                    _fmt_float(imbalance, 8),
                    _fmt_pct(unseen_ratio),
                    _fmt_float(amb_degree, 6),
                    _fmt_pct(leak_ratio),
                ]
            )

        # ---------------- redundancy / leakage tables ----------------
        redundancy_rows: List[List[str]] = []
        leakage_rows: List[List[str]] = []

        for name, m in zip(dataset_names, metrics_list):
            red = m.get("redundancy") or {}
            # expected keys train/val/test, but handle arbitrary
            if isinstance(red, dict) and red:
                for split, info in red.items():
                    if not isinstance(info, dict):
                        continue
                    redundancy_rows.append(
                        [
                            str(name),
                            str(split),
                            str(info.get("total", "NA")),
                            str(info.get("duplicates", "NA")),
                            _fmt_pct(info.get("redundancy")),
                        ]
                    )

            leak = m.get("leakage_ratio") or {}
            if isinstance(leak, dict) and leak:
                leakage_rows.append(
                    [
                        str(name),
                        str(leak.get("n_leaked_records", "NA")),
                        str(leak.get("total", "NA")),
                        _fmt_pct(leak.get("leakage_ratio")),
                    ]
                )

        # ---------------- label-agnostic distributions (per dataset) ----------------
        # Unseen per class distribution
        unseen_dist_rows: List[List[str]] = []
        # Ambiguity per class distribution
        amb_dist_rows: List[List[str]] = []

        # Top-K per dataset (still label-agnostic)
        top_unseen_blocks: List[Tuple[str, str]] = []  # (dataset_name, markdown)
        top_amb_blocks: List[Tuple[str, str]] = []

        for name, m in zip(dataset_names, metrics_list):
            unseen = m.get("unseen_entity_ratio") or {}
            per_class_unseen = (
                unseen.get("unseen_entities_per_class", {})
                if isinstance(unseen, dict)
                else {}
            )

            unseen_ratios = []
            if isinstance(per_class_unseen, dict):
                for _lbl, info in per_class_unseen.items():
                    if isinstance(info, dict):
                        v = _safe_float(info.get("unseen_entity_ratio"))
                        if v is not None:
                            unseen_ratios.append(v)

            uq = _quantiles(unseen_ratios)
            unseen_dist_rows.append(
                [
                    str(name),
                    str(int(uq["count"])) if uq else "0",
                    _fmt_pct(uq.get("mean")) if uq else "NA",
                    _fmt_pct(uq.get("median")) if uq else "NA",
                    _fmt_pct(uq.get("p90")) if uq else "NA",
                    _fmt_pct(uq.get("max")) if uq else "NA",
                ]
            )

            # Top-K unseen
            if isinstance(per_class_unseen, dict) and per_class_unseen:
                rows = []
                for lbl, info in per_class_unseen.items():
                    if not isinstance(info, dict):
                        continue
                    rows.append(
                        (
                            lbl,
                            _safe_float(info.get("unseen_entity_ratio")) or -1.0,
                            info.get("n_unseen_entities", "NA"),
                            info.get("total", "NA"),
                        )
                    )
                rows.sort(key=lambda x: x[1], reverse=True)
                rows = rows[:top_k]
                md = _md_table(
                    ["Label", "Unseen ratio", "Unseen", "Total"],
                    [
                        [str(lbl), _fmt_pct(r), str(nu), str(tot)]
                        for (lbl, r, nu, tot) in rows
                    ],
                )
                top_unseen_blocks.append((str(name), md))
            else:
                top_unseen_blocks.append(
                    (str(name), "_No per-label unseen breakdown available._")
                )

            # Ambiguity per class distribution + top-K
            amb = m.get("entity_ambiguity_degree") or {}
            per_class_amb = {}
            if isinstance(amb, dict):
                # per-class ambiguity entries are “all keys except the global fields”
                global_keys = {
                    "n_conflict_entities",
                    "total_unique_entities",
                    "entity_ambiguity_degree",
                    "n_records",
                }
                per_class_amb = {
                    k: v
                    for k, v in amb.items()
                    if k not in global_keys and isinstance(v, dict)
                }

            amb_values = []
            for _lbl, info in per_class_amb.items():
                v = _safe_float(info.get("entity_ambiguity_degree"))
                if v is not None:
                    amb_values.append(v)

            aq = _quantiles(amb_values)
            amb_dist_rows.append(
                [
                    str(name),
                    str(int(aq["count"])) if aq else "0",
                    _fmt_float(aq.get("mean"), 6) if aq else "NA",
                    _fmt_float(aq.get("median"), 6) if aq else "NA",
                    _fmt_float(aq.get("p90"), 6) if aq else "NA",
                    _fmt_float(aq.get("max"), 6) if aq else "NA",
                ]
            )

            if per_class_amb:
                rows = []
                for lbl, info in per_class_amb.items():
                    rows.append(
                        (
                            lbl,
                            _safe_float(info.get("entity_ambiguity_degree")) or -1.0,
                            info.get("n_conflict_entities", "NA"),
                            info.get("total_unique_entities", "NA"),
                        )
                    )
                rows.sort(key=lambda x: x[1], reverse=True)
                rows = rows[:top_k]
                md = _md_table(
                    ["Label", "Ambiguity degree", "Conflicting", "Unique entities"],
                    [
                        [str(lbl), _fmt_float(v, 6), str(nc), str(tu)]
                        for (lbl, v, nc, tu) in rows
                    ],
                )
                top_amb_blocks.append((str(name), md))
            else:
                top_amb_blocks.append(
                    (str(name), "_No per-label ambiguity breakdown available._")
                )

        # ---------------- write markdown ----------------
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines: List[str] = []
        lines.append("# Dataset Comparison Report\n")
        lines.append(f"- **Generated:** {now_utc}\n")
        lines.append(f"- **Datasets compared:** {len(metrics_list)}\n")
        lines.append("---\n")

        lines.append("## Overview (comparable metrics)\n")
        lines.append(_md_table(overview_headers, overview_rows))
        lines.append("\n---\n")

        if redundancy_rows:
            lines.append("## Redundancy by split\n")
            lines.append(
                _md_table(
                    ["Dataset", "Split", "Total", "Duplicates", "Redundancy"],
                    redundancy_rows,
                )
            )
            lines.append("\n---\n")

        if leakage_rows:
            lines.append("## Leakage (test overlap)\n")
            lines.append(
                _md_table(
                    ["Dataset", "Leaked records", "Test total", "Leakage ratio"],
                    leakage_rows,
                )
            )
            lines.append("\n---\n")

        lines.append("## Per-label Unseen Entity Ratio (label-agnostic summaries)\n")
        lines.append(
            "These stats summarize the *distribution of per-label unseen ratios* inside each dataset "
            "(so they work even when label sets differ).\n"
        )
        lines.append(
            _md_table(
                ["Dataset", "#Labels", "Mean", "Median", "P90", "Max"], unseen_dist_rows
            )
        )
        lines.append("\n### Top labels by unseen ratio (within each dataset)\n")
        for ds, block in top_unseen_blocks:
            lines.append(f"#### {ds}\n")
            lines.append(block + "\n")
        lines.append("\n---\n")

        lines.append("## Per-label Ambiguity Degree (label-agnostic summaries)\n")
        lines.append(
            "These stats summarize the *distribution of per-label ambiguity degrees* inside each dataset.\n"
        )
        lines.append(
            _md_table(
                ["Dataset", "#Labels", "Mean", "Median", "P90", "Max"], amb_dist_rows
            )
        )
        lines.append("\n### Top labels by ambiguity degree (within each dataset)\n")
        for ds, block in top_amb_blocks:
            lines.append(f"#### {ds}\n")
            lines.append(block + "\n")
        lines.append("\n---\n")

        lines.append("## Appendix: Raw Metrics (debug)\n")
        lines.append("```json\n")
        lines.append(
            _json_pretty(
                {"datasets": list(dataset_names), "metrics_list": list(metrics_list)}
            )
        )
        lines.append("\n```\n")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    def analyse_sample_size_effect_on_model_performance(
        self, config_path: Path | str
    ) -> None:
        """
        Analyse how *training-set sample size* impacts model performance and dataset difficulty metrics.

        This method reads a YAML/JSON config file, progressively subsamples the **train** split
        (from ``1/n_parts`` up to ``n_parts/n_parts``), and for each fraction it:

        - Evaluates the dataset (all ``_compute_*`` dataset metrics).
        - Trains a token-classification model (with early stopping) and evaluates on the test split.
        - Writes per-fraction evaluation reports and a combined comparison report under::

            datasets/reports/sample_size_effect_on_class_performance/<dataset_name>/

        Parameters
        ----------
        config_path:
            Path to a YAML (``.yml``/``.yaml``) configuration file.

        Expected config file schema
        ---------------------------
        Top-level keys
        ~~~~~~~~~~~~~~
        dataset_path : str
            Path to the dataset folder (must be readable by ``ClinicalRecordsDataset``).
        model_id : str
            Hugging Face model identifier for both tokenizer and model (e.g. ``jhu-clsp/mmBERT-base``).
            This value is also assigned to ``self.tokenizer_id``.
        early_stopping_patience : int
            Number of evaluation steps/epochs with no improvement before stopping training.
        n_parts : int
            Number of fractions to evaluate. The train split is sampled with fractions::

                i / n_parts  for i in [1, 2, ..., n_parts]

        dataset_args : dict
            Dataset-related arguments used during training:

            - max_length : int
                Maximum sequence length for tokenization/padding.
            - annotation_scheme : str
                Annotation scheme identifier (e.g. ``"IO"``). Stored as ``annotation_type`` in metrics.

        training_args : dict
            Hugging Face ``TrainingArguments`` fields (passed via ``TrainingArguments(output_dir="output", **training_args)``).
            Common keys include:

            - per_device_train_batch_size : int
            - gradient_accumulation_steps : int
            - num_train_epochs : int
            - learning_rate : float
            - weight_decay : float
            - eval_strategy : str
            - logging_strategy : str
            - save_strategy : str
            - load_best_model_at_end : bool
            - metric_for_best_model : str

        Optional keys (currently not consumed by this method directly)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        correlation_study_features : list[str]
            List of feature/column names intended for downstream correlation studies.
        variables_to_plot : list[str]
            List of metric names intended for downstream plotting.

        Returns
        -------
        None
            Reports and CSV outputs are written to disk; the method returns ``None``.

        Notes
        -----
        - Only the **train** split is subsampled; validation and test splits remain at full size.
        - Training is executed for each fraction, and test performance includes macro/micro AP
        plus per-entity PR-AUC in the saved metrics.
        """
        if isinstance(config_path, str):
            config_path = Path(config_path)

        config = load_config(config_path)

        # Check tokenizer
        self.tokenizer_id = config["model_id"]

        # Setup relevant paths
        dataset_path = Path(config["dataset_path"])
        results_base_path = (
            Path("datasets/reports/sample_size_effect_on_class_performance")
            / config_path.stem
        )
        results_base_path.mkdir(exist_ok=True, parents=True)

        # Prepare multiple dataset args for comparison
        n_parts = config["n_parts"]
        datasets_args = [
            {
                "dataset_path": dataset_path,
                "dataset_name": f"{dataset_path.stem}_frac_{i / n_parts:.2f}",
                "sample_from_splits": {"train": i / n_parts},
                **config["dataset_args"],
            }
            for i in range(1, n_parts + 1)
        ]

        # Run comparison
        self.compare(
            datasets_args,
            output_path=results_base_path,
            generate_eval_report=True,
            generate_comparison_report=True,
            train_model=True,
            training_config_path=config_path,
        )

        # Compute correlations and variable importance
        ## Load metrics
        df = pd.read_csv(results_base_path / "analysis_results.csv")
        df = df.dropna()
        target = "entity_specific_pr_auc"
        entities = df["entity_name"].unique()
        features = config["correlation_study_features"]
        variables_to_plot = config["variables_to_plot"]
        ## Filter for numeric features present in the dataset
        available_features = [
            f
            for f in features
            if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
        ]
        ## Future train size for estimation
        future_train_size = df["train_dataset_size"].max() * 2

        # Correlation Analysis
        corr_matrix = df[available_features + [target]].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap: Features vs. PR AUC")
        plt.tight_layout()
        plt.savefig(results_base_path / "correlation_heatmap.png")

        # Variable Importance Analysis
        X = df[available_features].fillna(0)
        y = df[target]
        print(X)
        print(y)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # SHAP Analysis
        for var in variables_to_plot:
            shap.partial_dependence_plot(
                var,
                model.predict,
                X,
                ice=False,
                model_expected_value=True,
                feature_expected_value=True,
            )

        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(X)
        shap.plots.beeswarm(shap_values)

        importances = pd.Series(
            model.feature_importances_, index=available_features
        ).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances.values, y=importances.index)
        plt.title("Variable Importance (Random Forest)")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(results_base_path / "variable_importance.png")

        # Setup for plot
        rows = (len(entities) + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(20, 5 * rows))
        axes = axes.flatten()

        results = []

        for i, entity in enumerate(entities):
            ax = axes[i]
            ent_data = df[df["entity_name"] == entity].sort_values("train_dataset_size")

            # Prepare log-transformed features
            x = np.log(ent_data["train_dataset_size"].values)
            y = ent_data[target].values
            n = len(x)

            # Fit model
            model = LinearRegression()
            model.fit(x.reshape(-1, 1), y)

            # Range for plotting (observed to future)
            x_range = np.linspace(
                ent_data["train_dataset_size"].min(), future_train_size, 100
            )
            x_range_log = np.log(x_range)

            # Predictions
            y_pred = model.predict(x_range_log.reshape(-1, 1))

            # Calculate Confidence Interval for the mean response
            # Formula: y_pred +/- t * SE
            # SE = sqrt( MSE * (1/n + (x_h - mean_x)^2 / sum((x_i - mean_x)^2)) )

            y_fitted = model.predict(x.reshape(-1, 1))
            mse = np.sum((y - y_fitted) ** 2) / (n - 2) if n > 2 else 0

            mean_x = np.mean(x)
            ssx = np.sum((x - mean_x) ** 2)

            # Calculate t-value for 95%
            t_val = stats.t.ppf(0.975, n - 2) if n > 2 else 2.0

            se_mean = (
                np.sqrt(mse * (1.0 / n + (x_range_log - mean_x) ** 2 / ssx))
                if ssx > 0
                else 0
            )

            ci_upper = y_pred + t_val * se_mean
            ci_lower = y_pred - t_val * se_mean

            # Clip values
            y_pred_clipped = np.clip(y_pred, 0, 1)
            ci_upper_clipped = np.clip(ci_upper, 0, 1)
            ci_lower_clipped = np.clip(ci_lower, 0, 1)

            # Plotting
            ax.scatter(
                ent_data["train_dataset_size"],
                y,
                color="blue",
                label="Observed",
                alpha=0.6,
            )
            ax.plot(x_range, y_pred_clipped, color="red", label="Log-Linear Trend")
            ax.fill_between(
                x_range,
                ci_lower_clipped,
                ci_upper_clipped,
                color="red",
                alpha=0.15,
                label="95% CI",
            )

            # Point estimate at future size
            future_pred = model.predict(np.log([[future_train_size]]))[0]
            future_se = (
                np.sqrt(
                    mse * (1.0 / n + (np.log(future_train_size) - mean_x) ** 2 / ssx)
                )
                if ssx > 0
                else 0
            )
            future_ci_low = np.clip(future_pred - t_val * future_se, 0, 1)
            future_ci_high = np.clip(future_pred + t_val * future_se, 0, 1)
            future_pred = np.clip(future_pred, 0, 1)

            ax.scatter(
                [future_train_size],
                [future_pred],
                color="black",
                marker="x",
                s=100,
                zorder=5,
            )

            ax.set_title(f"Entity: {entity}")
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Train Size")
            ax.set_ylabel("PR AUC")
            ax.legend(fontsize="x-small")

            results.append(
                {
                    "entity_name": entity,
                    "current_max_auc": y.max(),
                    f"estimated_auc_{future_train_size}": future_pred,
                    f"ci_lower_{future_train_size}": future_ci_low,
                    f"ci_upper_{future_train_size}": future_ci_high,
                    "expected_improvement": max(0, future_pred - y.max()),
                }
            )

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(results_base_path / "estimates_with_extended_ci.png")

        # 3. Variable Trends and Relationship Plots (with default CI)

        # Variable Trends vs Size
        fig_var, axes_var = plt.subplots(1, len(variables_to_plot), figsize=(18, 5))
        for i, var in enumerate(variables_to_plot):
            sns.lineplot(
                data=df,
                x="train_dataset_size",
                y=var,
                hue="entity_name",
                ax=axes_var[i],
                marker="s",
                legend=(i == len(variables_to_plot) - 1),
            )
            axes_var[i].set_title(f"Trend: {var} vs Size")
            axes_var[i].set_ylim(0, 1.05)
            axes_var[i].grid(True, linestyle=":", alpha=0.5)
        if axes_var[1].get_legend():
            axes_var[1].legend(
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize="x-small",
                title="Entity",
            )
        plt.tight_layout()
        plt.savefig(results_base_path / "variable_trends_with_ci.png")

        # Performance vs Variables Relationship
        fig_rel, axes_rel = plt.subplots(1, len(variables_to_plot), figsize=(18, 5))
        for i, var in enumerate(variables_to_plot):
            # sns.regplot includes 95% CI by default
            sns.regplot(
                data=df,
                x=var,
                y=target,
                ax=axes_rel[i],
                scatter_kws={"alpha": 0.4},
                line_kws={"color": "red"},
            )
            axes_rel[i].set_title(f"PR AUC vs {var} (with CI)")
            axes_rel[i].set_ylim(0, 1.05)
            axes_rel[i].set_xlim(left=0)
            axes_rel[i].grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(results_base_path / "performance_vs_variables_with_ci.png")

        estimates_df = pd.DataFrame(results).sort_values(
            "expected_improvement", ascending=False
        )
        estimates_df.to_csv(
            results_base_path / "pr_auc_estimates_with_extended_ci.csv", index=False
        )

        print("Analysis complete. Check 'estimates_with_extended_ci.png'.")
