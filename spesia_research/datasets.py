"""
This module contains the `ClinicalRecordsDataset` class for annotated clinical records (SemClinBr, Argilla, Docanno).
It loads records from different sources and prepares data for multilabel NER model training.

Functions:
    overlaps: Check strict overlap of half-open intervals [start, end).

Classes:
    ClinicalRecordsDataset: Dataset class for annotated clinical records.
"""

# Standard libraries
import functools
from pathlib import Path
from typing import Dict, List, Literal, Set, Tuple
import random
import copy
from itertools import combinations

# Third-party libraries
import jsonlines
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from pydantic import BaseModel
import datasets
from datasets import load_dataset

# Custom libraries
from .config import load_exp_config
from .data_models import Annotation, Record


# ==============================================================
# Utility functions
# ==============================================================


def overlaps(a_start, a_end, b_start, b_end) -> bool:
    """Check strict overlap of half-open intervals [start, end)."""
    return (a_start < b_end) and (b_start < a_end)


def reconstruct_text_from_tokens(tokens: List[str]) -> str:
    no_space_before = {".", ",", ";", ":", "!", "?", ")", "]", "}"}
    no_space_after = {"(", "[", "{"}

    text = ""
    for tok in tokens:
        if not text:
            text = tok
        elif tok in no_space_before:
            text += tok
        elif text[-1] in no_space_after:
            text += tok
        else:
            text += " " + tok

    return text


# ==============================================================
# Registry of funcions that map HF datasets to ClinicalRecordsDataset objects
# ==============================================================
HF_TO_CLINICAL_RECORDS_DATASET_MAPPING = {}


def hf_dataset_registry(hf_dataset_id: str):
    def decorator(func):
        # Register the callable at import time
        HF_TO_CLINICAL_RECORDS_DATASET_MAPPING[hf_dataset_id] = func
        return func

    return decorator


# ==============================================================
# Main dataset class
# ==============================================================


class ClinicalRecordsDataset(Dataset):
    """
    Dataset class for annotated clinical records (SemClinBr, Argilla, Docanno).
    Loads records from different sources and prepares data for multilabel NER model training.
    """

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        file_format: Literal["semclinbr", "argilla", "docanno", "conll"] = None,
        text_column: str = None,
        label_column: str = None,
        records: list[Record] | None = None,
        tokenizer: AutoTokenizer | None = None,
        label_type: Literal["tags", "semantic_groups"] = "tags",
        tags_to_consider: list[str] = None,
        semantic_groups_to_consider: list[str] = None,
        labels_to_ignore: list[str] = None,
        max_length=512,
        random_seed=42,
        split: Literal["train", "val", "test"] | None = None,
        split_ratio=None,
        min_samples_per_label: int = 10,
        semantic_group_standard: Literal["UMLS"] = "UMLS",
        annotation_scheme: Literal["IO", "BIO"] = "IO",
        begin_token_weight_scaler: float = 1.0,
        data_split_method: Literal["random", "iterative"] = "iterative",
        task: Literal[
            "token_classification",
            "masked_language_modeling",  # Does not use token labels
            "sequence_classification",  # When set, use token labels as sequence labels
            "supervised_fine_tuning",  # When set, use completions as labels for language modeling
        ] = "token_classification",
        custom_label_mapping: (
            dict | None
        ) = None,  # When set, changes labels in the dataset to match the custom label mapping
        # supervised_fine_tuning parameters
        prompt_template: list[dict[str, str]] = None,
        prompt_args: dict[str, str] = None,
        structured_output_model: BaseModel = None,
        count_tokens: bool = False,
        augment_data_with_dropped_entities: bool = False,
        data_augmentation_limit_per_record: int = 100,
        sample_from_split: float = 1.0,
        **kwargs,
    ) -> None:
        # --------------------------
        # Sanity checks
        # --------------------------
        assert sample_from_split <= 1 and sample_from_split > 0, (
            "sample_from_split must be between 0 and 1"
        )

        if isinstance(dataset_path, str):
            self.path = Path(dataset_path)
        else:
            self.path = dataset_path

        assert label_type in [
            "tags",
            "semantic_groups",
        ], "label_type must be 'tags' or 'semantic_groups'"

        assert annotation_scheme in [
            "IO",
            "BIO",
        ], "annotation_scheme must be 'IO' or 'BIO'"

        # --------------------------
        # Main attributes
        # --------------------------
        self.label_type = label_type
        self.file_format = file_format
        self.records: list[Record] = []
        self.all_records: list[Record] = []
        self.split = split
        self.sample_from_split = sample_from_split
        self.random_seed = random_seed
        self.min_samples_per_label = min_samples_per_label
        self.semantic_group_standard = semantic_group_standard
        self.tags: List[str] = []
        self.semantic_groups: List[str] = []
        self.labels_to_ignore: List[str] = (
            labels_to_ignore if labels_to_ignore is not None else []
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.annotation_scheme = annotation_scheme
        self.begin_token_weight_scaler = begin_token_weight_scaler
        self.data_split_method = data_split_method
        self.task = task
        self.total_tokens = 0
        self.custom_label_mapping = custom_label_mapping
        self.prompt_template = prompt_template
        self.prompt_args = copy.deepcopy(prompt_args)
        self.structured_output_model = structured_output_model
        self.column_names = ["prompt", "completion"]
        self.count_tokens = count_tokens
        self.augment_data_with_dropped_entities = augment_data_with_dropped_entities
        self.data_augmentation_limit_per_record = data_augmentation_limit_per_record
        self.semgroups_path = Path("datasets/ner/SemClinBr/SemGroups.txt")

        if split_ratio is None:
            self.split_ratio = {"train": 0.6, "val": 0.2, "test": 0.2}
        else:
            assert isinstance(split_ratio, dict)
            assert split_ratio.keys() == {"train", "val", "test"}
            self.split_ratio = split_ratio

        if self.path is not None:
            # --------------------------
            # Load files
            # --------------------------
            self._load_all_records(text_column=text_column, label_column=label_column)

        elif records is not None:
            self.all_records = records

        # --------------------------
        # Get labels
        # --------------------------
        self._get_records_tags()

        if self.label_type == "semantic_groups":
            self._get_semantic_groups()
            self._map_tags_to_semantic_groups(standard=self.semantic_group_standard)

        # --------------------------
        # Define labels
        # --------------------------
        self.tags_to_consider = (
            self.tags
            if tags_to_consider is None or len(tags_to_consider) == 0
            else tags_to_consider
        )

        self.semantic_groups_to_consider = (
            self.semantic_groups
            if semantic_groups_to_consider is None
            or len(semantic_groups_to_consider) == 0
            else semantic_groups_to_consider
        )

        self.tags_to_consider = [
            t for t in self.tags_to_consider if t not in self.labels_to_ignore
        ]

        self.semantic_groups_to_consider = [
            t
            for t in self.semantic_groups_to_consider
            if t not in self.labels_to_ignore
        ]

        # --------------------------
        # Create label map
        # --------------------------
        self._create_label_map()
        self.num_labels = len(self.labels_to_consider)

        # --------------------------
        # Split data
        # --------------------------
        if self.split is not None:
            self._split_records()
        else:
            self.records = self.all_records
            self.all_records = []

        # --------------------------
        # Keep only labels of interest
        # --------------------------
        for record in self.records:
            annotations_to_remove = []
            for ann in record.annotations:
                # track not wanted labels and remove them later
                labels_to_remove = []
                for label in getattr(ann, self.label_type):
                    if (
                        label not in self.labels_to_consider
                        or label in self.labels_to_ignore
                    ):
                        labels_to_remove.append(label)
                for label in labels_to_remove:
                    getattr(ann, self.label_type).remove(label)
                # track empty annotations
                if len(getattr(ann, self.label_type)) == 0:
                    annotations_to_remove.append(ann)
            # remove empty annotations
            for ann in annotations_to_remove:
                record.annotations.remove(ann)

        self._update_label_mapping()

        # --------------------------
        # Apply custom label mapping if set
        # --------------------------
        if self.custom_label_mapping is not None:
            self._apply_custom_label_mapping()

        # --------------------------
        # Generate prompts in records if supervised fine-tuning task
        # --------------------------
        if self.task == "supervised_fine_tuning":
            self._generate_prompts()

    # ==============================================================
    # Main methods
    # ==============================================================

    def _load_all_records(
        self, text_column: str = None, label_column: str = None
    ) -> None:
        """
        Load all records from different sources. Handles different file formats. Raises error file format is not supported.
        """
        if self.file_format == "conll":
            self.all_records = Record.from_conll(self.path)

        file_paths = list(self.path.glob("*"))
        if not file_paths:
            raise FileNotFoundError(f"No files found in {self.path}")

        for file_path in tqdm(file_paths, desc="Loading all Records"):
            try:
                records = Record.from_file(file_path, text_column, label_column)

                # Count tokens:
                if self.tokenizer is not None:
                    self.total_tokens += sum(
                        [
                            len(self.tokenizer(record.text)["input_ids"])
                            for record in records
                        ]
                    )

                self.all_records.extend(records)
            except Exception as e:
                print(f"[WARN] Erro ao processar {file_path.name}: {e}")

    @classmethod
    def from_exp_config(
        cls,
        config: dict | None = None,
        config_path: str | Path | None = None,
        split: str | None = None,
        tokenizer: AutoTokenizer | None = None,
    ) -> "ClinicalRecordsDataset":
        if config is not None:
            return cls(**config["dataset_args"], split=split, tokenizer=tokenizer)
        elif config_path is not None:
            config = load_exp_config(config_path)
            return cls(**config["dataset_args"], split=split, tokenizer=tokenizer)
        else:
            raise ValueError("Either config or config_path must be provided")

    @classmethod
    def from_list_of_strings(
        cls, texts: list[str], **kwargs
    ) -> "ClinicalRecordsDataset":
        return cls(
            records=[Record(text=text, annotations=[]) for text in texts], **kwargs
        )

    @classmethod
    def from_list_of_records(
        cls, records: list[Record], **kwargs
    ) -> "ClinicalRecordsDataset":
        return cls(records=records, **kwargs)

    def _split_records(self) -> None:
        if self.data_split_method == "random":
            self._random_data_split()
        elif self.data_split_method == "iterative":
            self._iterative_stratification()

    def _random_data_split(self) -> None:
        random.seed(self.random_seed)
        random.shuffle(self.all_records)

        # Compute split indices
        n_total = len(self.all_records)
        n_train = int(n_total * self.split_ratio["train"])
        n_val = int(n_total * self.split_ratio["val"])

        split_records = {
            "train": self.all_records[:n_train],
            "val": self.all_records[n_train : n_train + n_val],
            "test": self.all_records[n_train + n_val :],
        }

        self.records = split_records[self.split]

    def _iterative_stratification(self) -> None:
        np.random.seed(self.random_seed)

        # The same record may have more than one set of annotations
        ## Index records by unique text
        labels = getattr(self, self.label_type)
        unique_texts = {r.text: [0 for _ in labels] for r in self.all_records}

        ## Map annotations to index
        for r in self.all_records:
            for label in labels:
                if label in getattr(r, self.label_type):
                    unique_texts[r.text][labels.index(label)] = 1

        ## Create X and y
        X = np.array(list(unique_texts.keys())).reshape(-1, 1)
        y = np.array(list(unique_texts.values()))

        ## Remove labels with low sample count
        col_sums = y.sum(axis=0)
        counts = pd.Series(col_sums, index=labels)
        labels_to_ignore = counts[counts < self.min_samples_per_label].index.tolist()
        labels_to_keep = counts[counts >= self.min_samples_per_label].index.tolist()
        self.labels_to_ignore = labels_to_ignore
        if len(labels_to_ignore) > 0:
            print(
                "Ignored labels during data split due to low sample count:",
                labels_to_ignore,
            )

        ## Select only frequent labels
        keep_idx = [getattr(self, self.label_type).index(l) for l in labels_to_keep]
        y_filtered = y[:, keep_idx]

        ## Stratified split
        X_temp, y_temp, X_test, y_test = iterative_train_test_split(
            X, y_filtered, test_size=self.split_ratio["test"]
        )
        val_prop = self.split_ratio["val"] / (
            self.split_ratio["train"] + self.split_ratio["val"]
        )
        X_train, y_train, X_val, y_val = iterative_train_test_split(
            X_temp, y_temp, test_size=val_prop
        )

        # Get fraction of desired split if wanted
        if self.sample_from_split < 1.0:
            # Get the correct split
            if self.split == "train":
                X = X_train
                y = y_train
            elif self.split == "val":
                X = X_val
                y = y_val
            elif self.split == "test":
                X = X_test
                y = y_test

            # Get fraction with iterative split
            X, y, _, _ = iterative_train_test_split(
                X, y, test_size=(1 - self.sample_from_split)
            )

            # Reassign to the correct split
            if self.split == "train":
                X_train = X
                y_train = y
            elif self.split == "val":
                X_val = X
                y_val = y
            elif self.split == "test":
                X_test = X
                y_test = y

        records_split = {
            "train": X_train.reshape(-1).tolist(),
            "val": X_val.reshape(-1).tolist(),
            "test": X_test.reshape(-1).tolist(),
        }

        ## Define records to include in this split
        records_to_include_in_split = set(records_split[self.split])
        self.records = [
            r for r in self.all_records if r.text in records_to_include_in_split
        ]

        ## Save memory
        self.all_records = []

    def adaptive_oversample(
        self,
        target_ratio: float = 0.8,
        random_state: int = 42,
        max_replication: int = 15,
    ):
        """
        Adaptive oversampling using roulette wheel selection for individual sampling.
        Prioritizes records with multiple minority labels through probabilistic selection.

        Args:
            target_ratio (float): Percentage of the average to be reached (0.8 = 80%)
            random_state (int): Seed for reproducibility
            max_replication (int): Maximum number of replications per record
        """

        if self.split != "train":
            return

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        label_count_dict = self.get_label_count(self.label_type).to_dict()
        total_ocurrencias = sum(label_count_dict.values())
        num_labels = len(label_count_dict)
        avg_ocurrencias = total_ocurrencias / num_labels
        target_count = int(avg_ocurrencias * target_ratio)

        # Identify which labels need oversampling
        labels_needing_oversample = {}
        for label, count in label_count_dict.items():
            if count < target_count:
                deficit = target_count - count
                replication_factor = max(
                    1, min(max_replication, deficit // (count + 1))
                )
                labels_needing_oversample[label] = {
                    "current_count": count,
                    "target_count": target_count,
                    "deficit": deficit,
                    "replication_factor": replication_factor,
                }

        # WHEEL ROULETTE
        minority_labels_set = set(labels_needing_oversample.keys())

        high_value_records = []
        medium_value_records = []
        low_value_records = []

        for record in self.records:
            record_labels = getattr(record, self.label_type)
            minority_labels_in_record = [
                label for label in record_labels if label in minority_labels_set
            ]

            if not minority_labels_in_record:
                continue

            num_minority_labels = len(minority_labels_in_record)

            if num_minority_labels >= 3:
                high_value_records.append(
                    {
                        "record": record,
                        "minority_labels": minority_labels_in_record,
                        "fitness": 3.0,
                    }
                )
            elif num_minority_labels == 2:
                other_labels = [
                    label for label in record_labels if label not in minority_labels_set
                ]

                if len(other_labels) >= 3:
                    medium_value_records.append(
                        {
                            "record": record,
                            "minority_labels": minority_labels_in_record,
                            "fitness": 1.5,
                        }
                    )
                else:
                    low_value_records.append(
                        {
                            "record": record,
                            "minority_labels": minority_labels_in_record,
                            "fitness": 1.0,
                        }
                    )

        all_candidates = high_value_records + medium_value_records + low_value_records

        if not all_candidates:
            print("⚠️ There is no samples of minorities classes for oversampling")
            return

        def wheel_roulette_selection(candidates, num_selections):
            if not candidates:
                return []

            total_fitness = sum(candidate["fitness"] for candidate in candidates)

            wheel = []
            cumulative = 0.0

            for candidate in candidates:
                probability = candidate["fitness"] / total_fitness
                cumulative += probability
                wheel.append((cumulative, candidate))

            selected = []
            for _ in range(num_selections):
                spin = random.random()

                for threshold, candidate in wheel:
                    if spin <= threshold:
                        selected.append(candidate)
                        break

            return selected

        replication_needs = {}
        for label, info in labels_needing_oversample.items():
            replication_needs[label] = info["deficit"]

        new_records = self.records.copy()
        replication_stats = {label: 0 for label in labels_needing_oversample.keys()}
        remaining_deficits = {
            label: info["deficit"] for label, info in labels_needing_oversample.items()
        }

        max_rounds = 15
        for round_num in range(max_rounds):
            if all(deficit <= 0 for deficit in remaining_deficits.values()):
                break

            total_remaining_deficit = sum(remaining_deficits.values())
            if total_remaining_deficit <= 0:
                break

            max_replicas_per_round = min(
                total_remaining_deficit, len(all_candidates) * 2
            )

            selected_candidates = wheel_roulette_selection(
                all_candidates, max_replicas_per_round
            )

            for candidate in selected_candidates:
                record = candidate["record"]
                minority_labels = candidate["minority_labels"]

                needed_for_any_label = any(
                    remaining_deficits.get(label, 0) > 0 for label in minority_labels
                )

                if needed_for_any_label:
                    new_records.append(record)

                    for label in minority_labels:
                        if (
                            label in replication_stats
                            and remaining_deficits.get(label, 0) > 0
                        ):
                            replication_stats[label] += 1
                            remaining_deficits[label] = max(
                                0, remaining_deficits[label] - 1
                            )

        remaining_significant_deficits = {
            label: deficit
            for label, deficit in remaining_deficits.items()
            if deficit > len(all_candidates) // 2
        }

        if remaining_significant_deficits:
            for label, deficit in remaining_significant_deficits.items():
                label_candidates = [
                    candidate
                    for candidate in all_candidates
                    if label in candidate["minority_labels"]
                ]

                if label_candidates:
                    selected_for_label = wheel_roulette_selection(
                        label_candidates, deficit
                    )

                    for candidate in selected_for_label:
                        new_records.append(candidate["record"])
                        replication_stats[label] += 1
                        remaining_deficits[label] -= 1

                        if remaining_deficits[label] <= 0:
                            break

        if random_state is not None:
            random.shuffle(new_records)

        self.records = new_records

    # ==============================================================
    # Label getters and mapping methods
    # ==============================================================

    @property
    def pos_weight(self) -> torch.Tensor:
        """
        Returns the class weights for weighted loss.

        This property calculates the class weights based on the label counts.
        The class weights are used for weighted loss in machine learning models.

        Returns:
            pos_weight (torch.Tensor): A tensor containing class weights, where each weight corresponds to a label.
        """
        # Class weights for weighted loss if needed
        label_count = self.get_label_count(self.label_type)
        pos_weight = []
        for label, idx in self.label2id.items():
            base_label = label.split("-")[-1]
            curr_label_count = label_count.get(base_label, 0)

            # evita divisão por zero
            if curr_label_count == 0:
                curr_weight = 1.0
            else:
                curr_weight = (len(self) - curr_label_count) / max(curr_label_count, 1)

            if label.startswith("B-"):
                curr_weight *= self.begin_token_weight_scaler

            pos_weight.append(curr_weight)

        return torch.tensor(pos_weight)

    def _get_records_tags(self) -> None:
        tags = set()
        if len(self.all_records) > 0:
            for record in self.all_records:
                tags.update(record.tags)
        else:
            for record in self.records:
                tags.update(record.tags)
        self.tags = list(sorted(tags))

    def _get_semantic_groups(self, standard: str = "UMLS") -> None:
        if standard == "UMLS":
            semgroups = pd.read_csv(self.semgroups_path, sep="|", header=None)
            self.semantic_groups.extend(semgroups[1])
            self.semantic_groups = sorted(set(self.semantic_groups))
        else:
            raise NotImplementedError(
                f"Semantic group mapping to {standard} not implemented."
            )

    def _map_tags_to_semantic_groups(self, standard: str = "UMLS") -> None:
        if standard == "UMLS":
            semgroups = pd.read_csv(self.semgroups_path, sep="|", header=None)
            tag_to_semgroup = dict(zip(semgroups[3], semgroups[1]))

            for record in tqdm(
                self.all_records, desc="Mapping tags to UMLS semantic groups"
            ):
                for annotation in record.annotations:
                    annotation.semantic_groups = list(
                        {
                            tag_to_semgroup[tag]
                            for tag in annotation.tags
                            if tag in tag_to_semgroup
                        }
                    )
        else:
            raise NotImplementedError(
                f"Semantic group mapping to {standard} not implemented."
            )

    # ==============================================================
    # Tokenização e geração de labels
    # ==============================================================

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, index: int
    ) -> (
        transformers.tokenization_utils_base.BatchEncoding
        | List[transformers.tokenization_utils_base.BatchEncoding]
    ):
        if isinstance(index, int):
            records = [self.records[index]]

        elif isinstance(index, slice):
            records = self.records[index]

        else:
            raise ValueError("Index must be an integer or a slice.")

        if self.task == "supervised_fine_tuning":
            output = []
            for record in records:
                output.append(
                    {"prompt": record.prompt, "completion": record.completion}
                )

            if isinstance(index, int):
                return output[0]

            return output

        encoded_texts = []
        for record in records:
            if self.task in ["token_classification", "masked_language_modeling"]:
                encoded_text, token_vectors = self._spans_to_multilabels(record)
                labels = torch.tensor(token_vectors, dtype=torch.float)
                encoded_text["labels"] = labels
                del encoded_text["offset_mapping"]
                encoded_texts.append(encoded_text)

                if self.task == "masked_language_modeling":
                    del encoded_text["labels"]

            elif self.task == "sequence_classification":
                encoded_text = self.tokenizer(
                    record.text,
                    return_offsets_mapping=False,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                )
                labels = [0 for _ in range(self.num_labels)]
                for label in record.sequence_labels:
                    if label in self.label2id:
                        labels[self.label2id[label]] = 1
                labels = torch.tensor(labels, dtype=torch.float)
                encoded_text["labels"] = labels
                encoded_texts.append(encoded_text)
            else:
                raise ValueError(f"Unknown task: {self.task}")

        if isinstance(index, int):
            return encoded_texts[0]

        return encoded_texts

    def _create_label_map(self) -> None:
        self.labels_to_consider = getattr(self, f"{self.label_type}_to_consider")

        if self.annotation_scheme == "BIO":
            extended_labels = []
            for label in self.labels_to_consider:
                extended_labels += [f"B-{label}", f"I-{label}"]
            self.labels_to_consider = extended_labels

        self.label2id = {tag: i for i, tag in enumerate(self.labels_to_consider)}
        self.id2label = {i: tag for i, tag in enumerate(self.labels_to_consider)}

    def _spans_to_multilabels(
        self, record: Record
    ) -> Tuple[transformers.tokenization_utils_base.BatchEncoding, List[List[int]]]:
        encoded_text = self.tokenizer(
            record.text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        offsets = encoded_text["offset_mapping"]
        word_ids = encoded_text.word_ids()
        token_labels = [set() for _ in offsets]

        if self.annotation_scheme == "IO":
            for annotation in record.annotations:
                labels_source = (
                    annotation.tags
                    if self.label_type == "tags"
                    else annotation.semantic_groups
                )
                for label in labels_source:
                    if label not in getattr(self, f"{self.label_type}_to_consider"):
                        continue
                    for i, (token_start, token_end) in enumerate(offsets):
                        if token_start == token_end:
                            continue
                        if overlaps(
                            token_start, token_end, annotation.start, annotation.end
                        ):
                            token_labels[i].add(label)

        elif self.annotation_scheme == "BIO":
            for annotation in record.annotations:
                labels_source = (
                    annotation.tags
                    if self.label_type == "tags"
                    else annotation.semantic_groups
                )
                for label in labels_source:
                    if label not in getattr(self, f"{self.label_type}_to_consider"):
                        continue
                    for i, (token_start, token_end) in enumerate(offsets):
                        if token_start == token_end:
                            continue
                        if overlaps(
                            token_start, token_end, annotation.start, annotation.end
                        ):
                            tag = (
                                f"B-{label}"
                                if token_start == annotation.start
                                else f"I-{label}"
                            )
                            token_labels[i].add(tag)

        token_vectors = []
        for i, word_id in enumerate(word_ids):
            token_vector = [0] * self.num_labels
            if word_id is not None:
                for label in token_labels[i]:
                    token_vector[self.label2id[label]] = 1
            token_vectors.append(token_vector)

        return encoded_text, token_vectors

    # ==============================================================
    # Statistics and visualization
    # ==============================================================

    def get_label_count(
        self, label_type: Literal["tags", "semantic_groups"] = "tags"
    ) -> pd.Series:
        labels = getattr(self, label_type)
        y_arr = np.array(
            [
                [1 if label in getattr(r, label_type) else 0 for label in labels]
                for r in self.records
            ],
            dtype=int,
        )
        col_sums = y_arr.sum(axis=0)
        return pd.Series(col_sums, index=labels).sort_values(ascending=False)

    def print_record(self, index: int):
        record = self[index]
        for token, labels in zip(record.input_ids, record.labels):
            curr_labels = (labels == 1).nonzero().view(-1)
            if len(curr_labels) == 0:
                labels_text = "O"
            else:
                labels_text = "|".join([self.id2label[i.item()] for i in curr_labels])
            print(f"{self.tokenizer.decode(token):20} - {labels_text}")

    # ==============================================================
    #  Data export methods
    # ==============================================================

    def export(
        self,
        path: Path | str,
        export_format: Literal["jsonl"],
        include_annotations: bool = False,
        exclude_duplicates: bool = True,
        task: Literal[
            "token_classification", "sequence_classification"
        ] = "token_classification",
        in_shards: int = -1,
    ) -> None:
        """
        Exports the record to a file in the specified format and path.
        """
        if isinstance(path, str):
            path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)

        records_to_export = self.records

        if export_format == "jsonl":
            records_in_jsonl_format = []
            seen = set()

            for r in records_to_export:
                if r.text == "":
                    continue

                r_to_export = {}
                r_to_export["text"] = r.text
                r_to_export["label"] = []

                if include_annotations:
                    if task == "token_classification":
                        labels = []
                        for ann in r.annotations:
                            for label in getattr(ann, self.label_type):
                                labels.append([ann.start, ann.end, label])
                        r_to_export["label"] = labels

                    elif task == "sequence_classification":
                        labels = []
                        for ann in r.annotations:
                            for label in getattr(ann, self.label_type):
                                if label not in labels:
                                    labels.append(label)
                        r_to_export["label"] = labels

                if exclude_duplicates and r_to_export["text"] in seen:
                    continue

                seen.add(r_to_export["text"])
                records_in_jsonl_format.append(r_to_export)

            if in_shards > 0:
                records_in_jsonl_format = [
                    records_in_jsonl_format[i : i + in_shards]
                    for i in range(0, len(records_in_jsonl_format), in_shards)
                ]
                for i, shard in enumerate(records_in_jsonl_format):
                    with jsonlines.open(
                        path.parent / f"{path.stem}_{i}.jsonl", mode="w"
                    ) as writer:
                        for obj in shard:
                            writer.write(obj)
                return

            with jsonlines.open(path, mode="w") as writer:
                for obj in records_in_jsonl_format:
                    writer.write(obj)

    def _apply_custom_label_mapping(self):
        print("Applying custom label mapping:", self.custom_label_mapping)
        # iterate over records and change labels accordingly
        for record in self.records:
            for ann in record.annotations:
                for label in getattr(ann, self.label_type):
                    if label in self.custom_label_mapping:
                        setattr(
                            ann, self.label_type, [self.custom_label_mapping[label]]
                        )

        # update labels to ignore
        if self.labels_to_ignore is not None:
            self.labels_to_ignore += [k for k, v in self.custom_label_mapping.items()]
        else:
            self.labels_to_ignore = [k for k, v in self.custom_label_mapping.items()]

        # update label mapping
        self._update_label_mapping()

    def _update_label_mapping(self):
        self._get_records_tags()
        self.tags_to_consider = (
            self.tags
            if self.tags_to_consider is None or len(self.tags_to_consider) == 0
            else self.tags_to_consider
        )

        self.semantic_groups_to_consider = (
            self.semantic_groups
            if self.semantic_groups_to_consider is None
            or len(self.semantic_groups_to_consider) == 0
            else self.semantic_groups_to_consider
        )

        if self.labels_to_ignore is not None:
            self.tags_to_consider = [
                t for t in self.tags_to_consider if t not in self.labels_to_ignore
            ]

            self.semantic_groups_to_consider = [
                t
                for t in self.semantic_groups_to_consider
                if t not in self.labels_to_ignore
            ]

        self._create_label_map()
        self.num_labels = len(self.labels_to_consider)

    def _apply_prompt_args_to_messages(
        self, messages: List[Dict[str, str]], prompt_args: dict
    ):
        """
        Apply prompt args to messages. Ignore keys that are not present in the message. Keys must be wrapped in {}.
        """
        for message in messages:
            message["content"] = message["content"].format(
                **{
                    k: v
                    for k, v in prompt_args.items()
                    if "{" + k + "}" in message["content"]
                }
            )

        return messages

    def _generate_prompts(self):
        """
        Attaches prompts to the records based on prompt template, prompt args and output format if provided
        """

        # input validation
        assert self.prompt_template is not None, "Prompt template not provided"
        assert (
            isinstance(self.prompt_template, list)
            and len(self.prompt_template) > 0
            and isinstance(self.prompt_template[0], dict)
            and "role" in self.prompt_template[0]
            and "content" in self.prompt_template[0]
        ), (
            "Prompt template must be a non-empty list of dictionaries with 'role' and 'content' keys"
        )

        assert any(["{text}" in msg["content"] for msg in self.prompt_template]), (
            "Prompt template must contain '{text}' in at least one message"
        )

        reserved_keys = ["text", "json_format", "json_response"]
        if self.prompt_args is not None:
            for k in self.prompt_args.keys():
                assert k not in reserved_keys, (
                    f"Prompt args cannot contain '{k}' key. It is reserved for internal use. Reserved keys: {reserved_keys}."
                )

        self.prompt_args = self.prompt_args or {}

        if "entities" in self.prompt_args:
            for entity in self.labels_to_consider:
                assert entity in self.prompt_args["entities"], (
                    f"Entity '{entity}' not found with its description in prompt args. Available entities: {self.prompt_args['entities']}. Please provide it."
                )

        # add json_format if provided
        if self.structured_output_model is not None:
            assert issubclass(self.structured_output_model, BaseModel), (
                "Structured output model must be a pydantic BaseModel"
            )
            self.prompt_args["json_format"] = (
                self.structured_output_model.model_json_schema()
            )

            # make sure the provided model can be instantiated from a Record class object
            def has_method(obj, method_name):
                """Check if an object has a callable method with the given name."""
                # Use getattr with a default (None) to avoid AttributeError if the attribute doesn't exist
                attribute = getattr(obj, method_name, None)
                # Check if the attribute exists and is callable
                return attribute is not None and callable(attribute)

            assert has_method(self.structured_output_model, "from_record"), (
                "Structured output model must have a from_record method"
            )

        elif self.structured_output_model is None:
            for msg in self.prompt_template:
                assert "json_format" not in msg["content"], (
                    "json_format key not expected in prompt template. Please provide structured_output_model or remove it from prompt template."
                )
                assert "json_response" not in msg["content"], (
                    "json_response key not expected in prompt template. Please provide structured_output_model or remove it from prompt template."
                )

        for record in tqdm(
            self.records, desc="Formating prompts and adding to records"
        ):
            prompt_args = {**self.prompt_args, "text": record.text}
            if self.structured_output_model is not None:
                prompt_args["json_response"] = self.structured_output_model.from_record(
                    record
                ).model_dump_json()
            new_prompt = copy.deepcopy(self.prompt_template)
            new_prompt = self._apply_prompt_args_to_messages(new_prompt, prompt_args)
            record.prompt = new_prompt[:-1]
            record.completion = new_prompt[-1:]

        if self.augment_data_with_dropped_entities and self.split == "train":
            self._augment_data_with_dropped_entities()

    def _augment_data_with_dropped_entities(self):
        print("Augmenting data with dropped entities")
        augmented_records: List[Record] = []
        for record in tqdm(
            self.records, desc="Formating prompts and adding to records"
        ):
            entities = self.prompt_args["entities"].keys()
            entity_combinations: List[Tuple[str, ...]] = []
            for i in range(1, len(entities)):
                entity_combinations.extend(list(combinations(entities, i)))

            # limit number of combinations per record
            if len(entity_combinations) > self.data_augmentation_limit_per_record:
                entity_combinations = random.sample(
                    entity_combinations, k=self.data_augmentation_limit_per_record
                )

            # create new records using every combination of dropped entities
            for combination in entity_combinations:
                current_prompt_args = copy.deepcopy(self.prompt_args)
                current_prompt_args["text"] = record.text

                for entity in combination:
                    if entity in current_prompt_args["entities"]:
                        del current_prompt_args["entities"][entity]

                new_record = copy.deepcopy(record)

                if self.structured_output_model is not None:
                    # remove dropped entities from new record annotations
                    new_annotations = []
                    for annotation in new_record.annotations:
                        annotation.tags = [
                            entity
                            for entity in annotation.tags
                            if entity not in combination
                        ]
                        new_annotations.append(annotation)

                    new_record.annotations = new_annotations

                    # apply to json_response
                    current_prompt_args["json_response"] = (
                        self.structured_output_model.from_record(
                            new_record
                        ).model_dump_json()
                    )

                new_prompt = copy.deepcopy(self.prompt_template)
                new_prompt = self._apply_prompt_args_to_messages(
                    new_prompt, current_prompt_args
                )

                new_record.prompt = new_prompt[:-1]
                new_record.completion = new_prompt[-1:]
                augmented_records.append(new_record)

        self.records.extend(augmented_records)

    def to_hf_dataset(self):
        if self.task == "supervised_fine_tuning":
            # define generator
            def gen():
                for i in range(len(self.records)):
                    yield self[i]

            return datasets.Dataset.from_generator(gen)
        else:
            raise NotImplementedError(
                f"Task {self.task} not supported. Supported tasks: supervised_fine_tuning"
            )

    def extend(
        self, other: "ClinicalRecordsDataset" | List["ClinicalRecordsDataset"]
    ) -> None:
        if isinstance(other, ClinicalRecordsDataset):
            other = [other]

        for o in other:
            assert isinstance(o, ClinicalRecordsDataset)
            self.records.extend(o.records)
            self.total_tokens += o.total_tokens

        self._update_label_mapping()

    def get_texts(self) -> Set[str]:
        return {record.text for record in self.records}

    def get_text_index(self, text: str) -> int:
        return list(self.get_texts()).index(text)

    def get_overlapping_tags(self) -> set[set[str]]:
        """Returns a dictionary where keys are tuples of overlapping tags and values are lists of record indices that contain the overlapping tags.

        The dictionary is constructed by iterating over the records and their annotations, and checking for overlaps between adjacent annotations. If an overlap is found, the tags of the overlapping annotations are added to the dictionary as a tuple key, and the record index is added to the list of values for that key.

        The purpose of this function is to provide a way to identify records that contain overlapping annotations, which can be useful for tasks such as annotation deduplication or identifying mutually exclusive labels.

        Returns:
            A dictionary where keys are tuples of overlapping tags and values are lists of record indices that contain the overlapping tags.
        """
        overlapping_registry = {}
        for record_idx, record in enumerate(self.records):
            for i, annotation in enumerate(record.annotations):
                for j, other_annotation in enumerate(record.annotations[i + 1 :]):
                    if overlaps(
                        annotation.start,
                        annotation.end,
                        other_annotation.start,
                        other_annotation.end,
                    ):
                        overlap_key = set([*annotation.tags, *other_annotation.tags])
                        overlap_key = tuple(list(sorted(overlap_key)))
                        if overlap_key not in overlapping_registry:
                            overlapping_registry[overlap_key] = []
                            overlapping_registry[overlap_key].append(record_idx)

                        else:
                            if record not in overlapping_registry[overlap_key]:
                                overlapping_registry[overlap_key].append(record_idx)

        return overlapping_registry

    @classmethod
    def from_hf_dataset(cls, dataset_id: str, **kwargs) -> "ClinicalRecordsDataset":
        """
        Instantiates a ClinicalRecordsDataset object from a Hugging Face dataset ID.

        Args:
            dataset_id (str): The ID of the Hugging Face dataset to instantiate.
            **kwargs: Additional keyword arguments to pass to the ClinicalRecordsDataset constructor.

        Returns:
            ClinicalRecordsDataset: A new ClinicalRecordsDataset object instantiated from the Hugging Face dataset with the specified ID.
        """
        if dataset_id not in HF_TO_CLINICAL_RECORDS_DATASET_MAPPING:
            raise ValueError(f"Dataset {dataset_id} not supported yet.")

        return HF_TO_CLINICAL_RECORDS_DATASET_MAPPING[dataset_id](**kwargs)


# ==============================================================
# Data Collator
# ==============================================================


class DataCollatorForMultiLabelTokenClassification:
    """
    Custom data collator for consistent padding and batching.
    """

    def __init__(self, pad_token_id=0, max_length=512, num_labels=1, device="cpu"):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.num_labels = num_labels
        self.device = device

    def __call__(self, features, include_labels=True):
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            labels = feature["labels"]

            pad_len = self.max_length - len(input_ids)
            if pad_len > 0:
                input_ids += [self.pad_token_id] * pad_len
                attention_mask += [0] * pad_len
            else:
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]

            batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            batch_attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))

            if include_labels:
                if labels.shape[0] < self.max_length:
                    pad_labels = torch.zeros(
                        self.max_length - labels.shape[0], self.num_labels
                    )
                    labels = torch.cat([labels, pad_labels], dim=0)
                elif labels.shape[0] > self.max_length:
                    labels = labels[: self.max_length]
                labels = labels.to(dtype=torch.long)
                batch_labels.append(labels)

        if include_labels:
            return {
                "input_ids": torch.stack(batch_input_ids).to(self.device),
                "attention_mask": torch.stack(batch_attention_mask).to(self.device),
                "labels": torch.stack(batch_labels).to(self.device),
            }

        return {
            "input_ids": torch.stack(batch_input_ids).to(self.device),
            "attention_mask": torch.stack(batch_attention_mask).to(self.device),
        }


# ==============================================================
# HF datasets registry
# ==============================================================


@hf_dataset_registry("Aunderline/genia")
def load_aunderline_genia(split: str = None, **kwargs) -> "ClinicalRecordsDataset":
    if split == "val":
        split = "validation"
    dataset_id = "Aunderline/genia"
    datasets = load_dataset(dataset_id)
    converted_datasets = []
    splits_of_interest = datasets.keys() if split is None else [split]
    for split in splits_of_interest:
        dataset = datasets[split]
        records = []
        for record in dataset:
            text = reconstruct_text_from_tokens(record["tokens"])
            annotations = []
            for entity in sorted(record["entities"], key=lambda x: x["start"]):
                entity_text = reconstruct_text_from_tokens(
                    record["tokens"][entity["start"] : entity["end"]]
                )
                start = text.find(entity_text)
                end = start + len(entity_text)
                annotations.append(
                    Annotation(
                        id="",
                        tags=[entity["type"]],
                        start=start,
                        end=end,
                        text=entity_text,
                    )
                )

            records.append(Record(text=text, annotations=annotations))

        dataset = ClinicalRecordsDataset.from_list_of_records(records, **kwargs)
        converted_datasets.append(dataset)

    if len(converted_datasets) == 1:
        return converted_datasets[0]
    else:
        for ds in converted_datasets[1:]:
            converted_datasets[0].extend(ds)
        return converted_datasets[0]
