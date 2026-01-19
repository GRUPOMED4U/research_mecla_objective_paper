#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict
from sklearn.metrics import average_precision_score, precision_recall_curve
from transformers import AutoModelForSequenceClassification
import copy
import jsonlines

from spesia_research.config import load_exp_config
from spesia_research.logs import configure_logging, get_logger
from spesia_research.trainers import EarlyStoppingCallback


def coerce_scalar(s: str) -> Any:
    # Convenience: bare true/false/none
    """
    Convenience function to convert a string to a scalar type (int, float, bool, None)
    If the string is "true" or "false", it will be converted to a bool.
    If the string is "none" or "null", it will be converted to None.
    If the string contains a ".", "e", or "E", it will be converted to a float.
    Otherwise, it will be converted to an int if possible, or left as a string if not.

    Args:
        s (str): The string to convert.

    Returns:
        Any: The converted scalar type.
    """
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None

    # Try int/float
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except Exception:
        return s  # fallback to raw string


def deep_merge(dict1, dict2):
    """
    Recursively merges dict2 into dict1.
    Values in dict2 will overwrite values in dict1 for non-dict types.
    """
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge if both values are dictionaries
            merged[key] = deep_merge(merged[key], value)
        else:
            # Overwrite or add the value from dict2
            merged[key] = value
    return merged


def set_nested(d: dict, dotted_key: str, value: Any) -> dict:
    """
    Set a value in a nested dictionary using a dotted key.

    For example, if d = {} and dotted_key = "a.b.c", then set_nested(d, dotted_key, 1)
    will result in d = {"a": {"b": {"c": 1}}.

    Args:
        d (dict): The dictionary to modify.
        dotted_key (str): The dotted key to set the value for.
        value (Any): The value to set.

    Returns:
        None
    """
    parts = dotted_key.split(".")
    new_dict = value
    for p in parts[::-1]:
        new_dict = {p: new_dict}
    d = deep_merge(d, new_dict)
    return d


def parse_kv_list(kvs: list[str]) -> dict:
    """
    Parse a list of key-value pairs into a dictionary.

    Args:
        kvs (list[str]): A list of key-value pairs in the format "key=value".

    Returns:
        dict: A dictionary containing the parsed key-value pairs.

    Raises:
        SystemExit: If a key-value pair is malformed (e.g. lacks "=").
    """
    out: dict = {}
    for item in kvs:
        if "=" not in item:
            raise SystemExit(f"Invalid --set '{item}'. Expected key=value.")
        k, v = item.split("=", 1)
        out = set_nested(out, k.strip(), coerce_scalar(v.strip()))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hpsearch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=str, help="Path to YAML/JSON config file")
    p.add_argument(
        "--set",
        "-s",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config value. Repeatable. Supports dotted keys.",
    )

    return p


def parse_config() -> Dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args()
    overrides = parse_kv_list(args.set)
    config = load_exp_config(args.config)
    config = deep_merge(config, overrides)
    results_path = Path("experiments") / Path(args.config).stem
    results_path.mkdir(exist_ok=True)
    config["results_path"] = results_path
    return config


def is_not_empty(p: Path) -> bool:
    """Check if the path exists and the directory is not empty."""
    if p.is_dir():
        # any() returns True if the iterator yields any item, False otherwise.
        # This is the most efficient way as it short-circuits.
        return any(p.iterdir())
    elif p.is_file():
        # You might also want to check for zero-byte files
        return p.stat().st_size > 0
    else:
        # Path does not exist or is a broken symlink, etc.
        return False


if __name__ == "__main__":
    from pathlib import Path
    from transformers import AutoTokenizer
    from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
    from transformers import AutoModelForMaskedLM, AutoModelForCausalLM
    from transformers import DataCollatorForLanguageModeling
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    import numpy as np
    from trl import SFTConfig, SFTTrainer

    from spesia_research.metrics import compute_metrics, get_best_threshold
    from spesia_research.datasets import (
        ClinicalRecordsDataset,
        DataCollatorForMultiLabelTokenClassification,
    )
    from spesia_research.trainers import (
        MultiLabelTokenTrainer,
        MultilabelSequenceClassificationTrainer,
    )
    from spesia_research.custom_models.bidirectional_llama import (
        BidirectionalLlamaForTokenClassification,
        BidirectionalLlamaForCausalLM,
    )
    from spesia_research.data_models import AgentAnnotationsList

    # Configure logging
    configure_logging()
    logger = get_logger("train")

    config = parse_config()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], use_fast=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Extend tokenizer option
    if config.get("extend_tokenizer"):
        print("Extending tokenizer...")
        # Prepare training data
        tokenizer_training_data_filepaths = config.get(
            "tokenizer_training_data_filepaths", []
        )
        assert len(tokenizer_training_data_filepaths) > 0, (
            "If you want to extend the tokenizer, please provide at least one "
            "training data file path."
        )
        assert all(
            p.endswith(".jsonl") or p.endswith(".json")
            for p in tokenizer_training_data_filepaths
        ), "All training data files must be .json or .jsonl"

        tokenizer_training_dataset = load_dataset(
            "json", data_files=tokenizer_training_data_filepaths, split="train"
        )

        # Train new tokenizer
        new_tokenizer_vocab_size = config.get("new_tokenizer_vocab_size", 3000)
        new_tokenizer = tokenizer.train_new_from_iterator(
            tokenizer_training_dataset["text"], new_tokenizer_vocab_size
        )

        # Incorporate new tokens into old tokenizer
        new_tokens = set(new_tokenizer.get_vocab().keys())
        old_tokens = set(tokenizer.get_vocab().keys())

        added_tokens = []
        for token in new_tokens:
            if token not in old_tokens:
                added_tokens.append(token)
        tokenizer.add_tokens(added_tokens)
        print("Total number of new tokens incorporated:", len(added_tokens))

    # Load datasets
    # Handle the possibility of merging multiple datasets
    if isinstance(config["dataset_args"]["dataset_path"], str):
        config["dataset_args"]["dataset_path"] = [
            config["dataset_args"]["dataset_path"]
        ]

    paths = config["dataset_args"]["dataset_path"]
    dataset_args = copy.deepcopy(config["dataset_args"])
    del config["dataset_args"]["dataset_path"]

    datasets: dict[str, ClinicalRecordsDataset | None] = {
        "train": None,
        "val": None,
        "test": None,
    }

    logger.info(f"Dataset args: {dataset_args}")

    for dataset_path in paths:
        dataset_args["dataset_path"] = dataset_path
        if config["dataset_args"].get("task") == "masked_language_modeling":
            current_train_dataset = ClinicalRecordsDataset(
                tokenizer=tokenizer, **dataset_args
            )

            if datasets["train"] is None:
                datasets["train"] = current_train_dataset
            else:
                datasets["train"].extend(current_train_dataset)

        elif config["dataset_args"].get("task") == "supervised_fine_tuning":
            for split, split_dataset in datasets.items():
                current_dataset = ClinicalRecordsDataset(
                    tokenizer=tokenizer,
                    split=split,
                    structured_output_model=AgentAnnotationsList,
                    **dataset_args,
                )

                if split_dataset is None:
                    datasets[split] = current_dataset
                else:
                    datasets[split].extend(current_dataset)

        else:
            for split, split_dataset in datasets.items():
                current_dataset = ClinicalRecordsDataset(
                    tokenizer=tokenizer, split=split, **dataset_args
                )

                if split_dataset is None:
                    datasets[split] = current_dataset
                else:
                    datasets[split].extend(current_dataset)

    # Extend datasets if set in config
    if config.get("dataset_extensions") is not None:
        print("Extending datasets...")
        for split in config["dataset_extensions"]:
            for dataset_path in config["dataset_extensions"][split]:
                dataset_args["dataset_path"] = dataset_path
                current_dataset = ClinicalRecordsDataset(
                    tokenizer=tokenizer,
                    split=None,  # The extension is completely loaded into the designed split
                    **dataset_args,
                )
                datasets[split].extend(current_dataset)

    train_dataset, val_dataset, test_dataset = (
        datasets["train"],
        datasets["val"],
        datasets["test"],
    )

    logger.info(f"Dataset args: {config['dataset_args']}")

    if config["dataset_args"].get("count_tokens"):
        print(f"Total training tokens: {train_dataset.total_tokens}")

    # Load best params from hpsearch if available
    if config.get("load_best_params_from_hpsearch", False):
        logger.info("Loading best params from hpsearch...")
        for path in config["best_params_paths"].get("training_args", []):
            best_training_args_params_path = Path(path)
            best_training_args_params_path = best_training_args_params_path.parent / (
                best_training_args_params_path.stem
                + f"_data_split_seed_{train_dataset.random_seed}.jsonl"
            )
            if not best_training_args_params_path.exists():
                logger.warning(
                    f"Best training args params path {best_training_args_params_path} does not exist. Skipping..."
                )
                continue
            best_training_args_params = json.load(
                open(best_training_args_params_path, "r")
            )
            config["training_args"].update(
                best_training_args_params["best_trial_params"]
            )
            logger.info(
                f"Training args: Loaded best params from {best_training_args_params_path}"
            )
            logger.info(
                f"Best params: {best_training_args_params['best_trial_params']}"
            )
            logger.info(f"Training args: {config['training_args']}")

        for path in config["best_params_paths"].get("trainer_args", []):
            best_trainer_args_params_path = Path(path)
            best_trainer_args_params_path = best_trainer_args_params_path.parent / (
                best_trainer_args_params_path.stem
                + f"_data_split_seed_{train_dataset.random_seed}.jsonl"
            )
            if not best_trainer_args_params_path.exists():
                logger.warning(
                    f"Best trainer args params path {best_trainer_args_params_path} does not exist. Skipping..."
                )
                continue
            best_trainer_args_params = json.load(
                open(best_trainer_args_params_path, "r")
            )
            config["trainer_args"].update(best_trainer_args_params["best_trial_params"])
            logger.info(
                f"Trainer args: Loaded best params from {best_trainer_args_params_path}"
            )
            logger.info(f"Best params: {best_trainer_args_params['best_trial_params']}")
            logger.info(f"Trainer args: {config['trainer_args']}")

    # Load training arguments
    if config["dataset_args"].get("task") == "supervised_fine_tuning":
        training_args = SFTConfig(
            output_dir=config["results_path"] / "checkpoints",
            max_length=config["dataset_args"]["max_length"],
            completion_only_loss=True,
            **config["training_args"],
        )

    else:
        training_args = TrainingArguments(
            output_dir=config["results_path"] / "checkpoints",
            **config["training_args"],
        )

    # Prepare model, data collator and trainer
    if config["dataset_args"].get("task") == "masked_language_modeling":
        # Load model
        if config.get("use_bidirectional_llama"):
            model = BidirectionalLlamaForCausalLM.from_pretrained(config["model_id"])

        else:
            model = AutoModelForMaskedLM.from_pretrained(config["model_id"])

        # Check if tokenizer has a mask token
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
            model.resize_token_embeddings(len(tokenizer))

        # Resize model if tokenizer has been extended
        if config.get("extend_tokenizer"):
            model.resize_token_embeddings(len(tokenizer))

        # Load lora if enabled
        if config.get("use_lora"):
            lora_config = LoraConfig(**config["lora_config"])
            model = get_peft_model(model, lora_config)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=config["data_collator_args"]["mlm_probability"],
        )

        # Load trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(patience=config["early_stopping_patience"])
            ],
            **config.get("trainer_args", {}),
        )

    elif config["dataset_args"].get("task") == "token_classification":
        if config.get("use_bidirectional_llama"):
            model = BidirectionalLlamaForTokenClassification.from_pretrained(
                config["model_id"], num_labels=train_dataset.num_labels
            )

        else:
            model = AutoModelForTokenClassification.from_pretrained(
                config["model_id"], num_labels=train_dataset.num_labels
            )

        # Load lora if enabled
        if config.get("use_lora"):
            lora_config = LoraConfig(**config["lora_config"])
            model = get_peft_model(model, lora_config)

        data_collator = DataCollatorForMultiLabelTokenClassification(
            pad_token_id=tokenizer.pad_token_id,
            max_length=config["dataset_args"]["max_length"],
            num_labels=train_dataset.num_labels,
        )

        # Load trainer
        trainer = MultiLabelTokenTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=(
                val_dataset if val_dataset.split_ratio["val"] > 0 else test_dataset
            ),
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(patience=config["early_stopping_patience"])
            ],
            pos_weight=train_dataset.pos_weight,
            **config.get("trainer_args", {}),
        )

    elif config["dataset_args"].get("task") == "sequence_classification":
        if config.get("use_bidirectional_llama"):
            raise ValueError(
                "Bidirectional LLaMA not supported for sequence classification"
            )

        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                config["model_id"], num_labels=train_dataset.num_labels
            )

        # Load lora if enabled
        if config.get("use_lora"):
            lora_config = LoraConfig(**config["lora_config"])
            model = get_peft_model(model, lora_config)

        data_collator = DataCollatorForMultiLabelTokenClassification(
            pad_token_id=tokenizer.pad_token_id,
            max_length=config["dataset_args"]["max_length"],
            num_labels=train_dataset.num_labels,
        )

        # Load trainer
        trainer = MultilabelSequenceClassificationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=(
                val_dataset if val_dataset.split_ratio["val"] > 0 else test_dataset
            ),
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(patience=config["early_stopping_patience"])
            ],
            pos_weight=train_dataset.pos_weight,
            **config.get("trainer_args", {}),
        )

    elif config["dataset_args"].get("task") == "supervised_fine_tuning":
        # Convert datasets to huggingface datasets
        train_dataset = train_dataset.to_hf_dataset()
        val_dataset = val_dataset.to_hf_dataset()
        # TODO: Add evaluation pipeline for gen AI models
        # test_dataset = test_dataset.to_hf_dataset()

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(config["model_id"])
        tokenizer = AutoTokenizer.from_pretrained(config["model_id"])

        # Load lora if enabled
        if config.get("use_lora"):
            lora_config = LoraConfig(**config["lora_config"])
            model = get_peft_model(model, lora_config)

        # Load trainer
        # TODO: Add support for early stopping
        # TODO: Add specific compute metrics
        # TODO: Add specific compute loss
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
            processing_class=tokenizer,
            **config.get("trainer_args", {}),
        )

    else:
        raise ValueError("Task not supported")

    # Execute training
    ## Either resume or start new training loop
    if config.get("resume_from_checkpoint") is not None:
        resume_from_checkpoint = config["resume_from_checkpoint"]
    else:
        resume_from_checkpoint = (
            True if (is_not_empty(config["results_path"] / "checkpoints")) else False
        )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Evaluate on test set if necessary
    if config["dataset_args"].get("task") in [
        "token_classification",
        "sequence_classification",
    ]:
        trainer.eval_dataset = test_dataset
        test_set_metrics = trainer.evaluate()
        test_set_metrics["run_name"] = config["run_name"]
        test_set_metrics.update(
            {
                "model_id": config["model_id"],
                "batch_effective": config["training_args"][
                    "per_device_train_batch_size"
                ]
                * config["training_args"]["gradient_accumulation_steps"],
                "patience": config["early_stopping_patience"],
                "pos_weight": train_dataset.pos_weight.tolist(),
            }
        )

        if config["dataset_args"].get("task") == "token_classification":
            test_set_metrics["annotation_type"] = config["dataset_args"][
                "annotation_scheme"
            ]

        # Compute metrics
        pred_output = trainer.predict(test_dataset)
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
            label_type = config["dataset_args"].get("label_type", "tags")
            label_name = (
                getattr(test_dataset, f"{label_type}_to_consider")[i]
                if probs_flat.ndim > 1
                else "Label"
            )
            precision, recall, thresholds = precision_recall_curve(y_true, y_score)
            pr_auc = average_precision_score(y_true, y_score)
            threshold_map[test_dataset.labels_to_consider[i]] = {
                "threshold": thresholds,
                "precision": precision,
                "recall": recall,
                "pr_auc": pr_auc,
            }

        macro_ap = average_precision_score(labels_flat, probs_flat, average="macro")
        micro_ap = average_precision_score(labels_flat, probs_flat, average="micro")
        test_set_metrics["macro_ap"] = macro_ap
        test_set_metrics["micro_ap"] = micro_ap
        print(f"Macro AP: {macro_ap:.3f}")
        print(f"Micro AP: {micro_ap:.3f}")

        # Select best tresholds
        thresholds = get_best_threshold(
            threshold_map, **config.get("threshold_selection", {})
        )
        trainer.model.config.thresholds = thresholds.to_dict()

        # Save metrics
        metrics_path = (
            config["results_path"]
            / f"metrics_data_split_seed_{train_dataset.random_seed}.jsonl"
        )
        write_mode = (
            "a" if metrics_path.is_file() and is_not_empty(metrics_path) else "w"
        )
        with jsonlines.open(metrics_path, write_mode) as writer:
            # convert np.arrays to lists
            threshold_map = {
                k: {
                    "threshold": v["threshold"].tolist(),
                    "precision": v["precision"].tolist(),
                    "recall": v["recall"].tolist(),
                    "pr_auc": v["pr_auc"],
                }
                for k, v in threshold_map.items()
            }

            test_set_metrics["run_name"] = config["run_name"]
            test_set_metrics["threshold_map"] = threshold_map
            test_set_metrics["selected_thresholds"] = thresholds.to_dict()
            writer.write(test_set_metrics)

    # Save best model
    trainer.save_model(config["results_path"] / "best_model")
    trainer.save_state()
