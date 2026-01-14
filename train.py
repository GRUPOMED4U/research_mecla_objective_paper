#!/usr/bin/env python3
from __future__ import annotations
import uuid
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve
from transformers import AutoModelForSequenceClassification
import yaml
import copy
import jsonlines

from spesia_research.trainers import EarlyStoppingCallback


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=str, help="Path to YAML/JSON config file")
    mode.add_argument(
        "--cli",
        action="store_true",
        help="Provide full config via CLI options (no --config)",
    )

    # ----- Top-level -----
    p.add_argument("--model-id", type=str)
    p.add_argument("--use-bidirectional-llama", action="store_true")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--early-stopping-patience", type=int, default=2)
    p.add_argument("--run-name", type=str, default=uuid.uuid4().hex[:8])

    # ----- Dataset args -----
    dataset_args_group = p.add_argument_group("Dataset args")
    dataset_args_group.add_argument("--dataset-path", type=str)
    dataset_args_group.add_argument("--labels-to-consider", nargs="+")
    dataset_args_group.add_argument("--annotation-scheme", choices=["IO", "BIO"])

    # ----- Training args -----
    training_args_group = p.add_argument_group("Training args")
    training_args_group.add_argument("--output-dir", type=str, default="./output")
    training_args_group.add_argument("--per-device-train-batch-size", type=int)
    training_args_group.add_argument("--gradient-accumulation-steps", type=int)
    training_args_group.add_argument("--num-train-epochs", type=int)
    training_args_group.add_argument("--learning-rate", type=float)
    training_args_group.add_argument("--weight-decay", type=float, default=None)
    training_args_group.add_argument("--include-for-metrics", nargs="*", default=None)
    training_args_group.add_argument(
        "--eval-strategy", choices=["no", "steps", "epoch"], default="epoch"
    )
    training_args_group.add_argument(
        "--logging-strategy", choices=["no", "steps", "epoch"], default="epoch"
    )
    training_args_group.add_argument(
        "--save-strategy", choices=["no", "steps", "epoch"], default="epoch"
    )
    training_args_group.add_argument(
        "--load-best-model-at-end", action="store_true", default=True
    )
    training_args_group.add_argument(
        "--metric-for-best-model", type=str, default="eval_loss"
    )

    # ----- LoRA args -----
    lora_args_group = p.add_argument_group("LoRA args")
    lora_args_group.add_argument("--use-lora", action="store_true")
    lora_args_group.add_argument("--lora-r", type=int, default=16)
    lora_args_group.add_argument("--lora-alpha", type=int, default=16)
    lora_args_group.add_argument(
        "--lora-target-modules", nargs="+", default="all-linear"
    )
    lora_args_group.add_argument("--lora-dropout", type=float, default=0.1)
    lora_args_group.add_argument("--lora-bias", type=str, default="none")
    lora_args_group.add_argument(
        "--lora-modules-to-save", nargs="+", default=["classifier"]
    )

    return p


def _load_config(path: str) -> Dict[str, Any]:
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


def _require_all(
    parser: argparse.ArgumentParser, args: argparse.Namespace, fields: List[str]
) -> None:
    missing = [f for f in fields if getattr(args, f) in (None, [])]
    if missing:
        parser.error(
            "Missing required CLI args (in --cli mode): "
            + ", ".join(f"--{m.replace('_', '-')}" for m in missing)
        )


def parse_config() -> Dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        results_path = Path("experiments") / Path(args.config).stem
        results_path.mkdir(exist_ok=True)
        parser_data = _load_config(args.config)
        parser_data["results_path"] = results_path
        return parser_data

    # --cli mode: enforce everything is present
    required_fields = [
        "model_id",
        "dataset_path",
        "tags_to_consider",
        "annotation_scheme",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "num_train_epochs",
        "learning_rate",
    ]
    _require_all(parser, args, required_fields)

    parser_data = {
        "model_id": args.model_id,
        "use_bidirectional_llama": args.use_bidirectional_llama,
        "max_length": args.max_length,
        "early_stopping_patience": args.early_stopping_patience,
        "run_name": args.run_name,
        "dataset_args": {
            "dataset_path": args.dataset_path,
            "tags_to_consider": args.tags_to_consider,
            "annotation_scheme": args.annotation_scheme,
        },
        "training_args": {
            "output_dir": args.output_dir,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "include_for_metrics": args.include_for_metrics or [],
            "eval_strategy": args.eval_strategy,
            "logging_strategy": args.logging_strategy,
            "save_strategy": args.save_strategy,
            "load_best_model_at_end": bool(args.load_best_model_at_end),
            "metric_for_best_model": args.metric_for_best_model,
        },
    }

    if args.use_lora:
        parser_data["lora_config"] = {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "target_modules": args.lora_target_modules,
            "dropout": args.lora_dropout,
            "bias": args.lora_bias,
            "modules_to_save": args.lora_modules_to_save,
        }

    results_path = Path("experiments") / parser_data["run_name"]
    results_path.mkdir(exist_ok=True)
    parser_data = _load_config(args.config)
    parser_data["results_path"] = results_path
    return parser_data


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

    cfg = parse_config()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], use_fast=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Extend tokenizer option
    if cfg.get("extend_tokenizer"):
        print("Extending tokenizer...")
        # Prepare training data
        tokenizer_training_data_filepaths = cfg.get(
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
        new_tokenizer_vocab_size = cfg.get("new_tokenizer_vocab_size", 3000)
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
    if isinstance(cfg["dataset_args"]["dataset_path"], str):
        cfg["dataset_args"]["dataset_path"] = [cfg["dataset_args"]["dataset_path"]]

    paths = cfg["dataset_args"]["dataset_path"]
    dataset_args = copy.deepcopy(cfg["dataset_args"])
    del cfg["dataset_args"]["dataset_path"]

    datasets: dict[str, ClinicalRecordsDataset | None] = {
        "train": None,
        "val": None,
        "test": None,
    }

    for dataset_path in paths:
        dataset_args["dataset_path"] = dataset_path
        if cfg["dataset_args"].get("task") == "masked_language_modeling":
            current_train_dataset = ClinicalRecordsDataset(
                tokenizer=tokenizer, **dataset_args
            )

            if datasets["train"] is None:
                datasets["train"] = current_train_dataset
            else:
                datasets["train"].extend(current_train_dataset)

        elif cfg["dataset_args"].get("task") == "supervised_fine_tuning":
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
    if cfg.get("dataset_extensions") is not None:
        print("Extending datasets...")
        for split in cfg["dataset_extensions"]:
            for dataset_path in cfg["dataset_extensions"][split]:
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

    if cfg["dataset_args"].get("count_tokens"):
        print(f"Total training tokens: {train_dataset.total_tokens}")

    # Load training arguments
    if cfg["dataset_args"].get("task") == "supervised_fine_tuning":
        training_args = SFTConfig(
            output_dir=cfg["results_path"] / "checkpoints",
            max_length=cfg["dataset_args"]["max_length"],
            completion_only_loss=True,
            **cfg["training_args"],
        )

    else:
        training_args = TrainingArguments(
            output_dir=cfg["results_path"] / "checkpoints",
            **cfg["training_args"],
        )

    # Prepare model, data collator and trainer
    if cfg["dataset_args"].get("task") == "masked_language_modeling":
        # Load model
        if cfg.get("use_bidirectional_llama"):
            model = BidirectionalLlamaForCausalLM.from_pretrained(cfg["model_id"])

        else:
            model = AutoModelForMaskedLM.from_pretrained(cfg["model_id"])

        # Check if tokenizer has a mask token
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
            model.resize_token_embeddings(len(tokenizer))

        # Resize model if tokenizer has been extended
        if cfg.get("extend_tokenizer"):
            model.resize_token_embeddings(len(tokenizer))

        # Load lora if enabled
        if cfg.get("use_lora"):
            lora_config = LoraConfig(**cfg["lora_config"])
            model = get_peft_model(model, lora_config)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=cfg["data_collator_args"]["mlm_probability"],
        )

        # Load trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(patience=cfg["early_stopping_patience"])],
            **cfg.get("trainer_args", {}),
        )

    elif cfg["dataset_args"].get("task") == "token_classification":
        if cfg.get("use_bidirectional_llama"):
            model = BidirectionalLlamaForTokenClassification.from_pretrained(
                cfg["model_id"], num_labels=train_dataset.num_labels
            )

        else:
            model = AutoModelForTokenClassification.from_pretrained(
                cfg["model_id"], num_labels=train_dataset.num_labels
            )

        # Load lora if enabled
        if cfg.get("use_lora"):
            lora_config = LoraConfig(**cfg["lora_config"])
            model = get_peft_model(model, lora_config)

        data_collator = DataCollatorForMultiLabelTokenClassification(
            pad_token_id=tokenizer.pad_token_id,
            max_length=cfg["dataset_args"]["max_length"],
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
            callbacks=[EarlyStoppingCallback(patience=cfg["early_stopping_patience"])],
            pos_weight=train_dataset.pos_weight,
            **cfg.get("trainer_args", {}),
        )

    elif cfg["dataset_args"].get("task") == "sequence_classification":
        if cfg.get("use_bidirectional_llama"):
            raise ValueError(
                "Bidirectional LLaMA not supported for sequence classification"
            )

        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                cfg["model_id"], num_labels=train_dataset.num_labels
            )

        # Load lora if enabled
        if cfg.get("use_lora"):
            lora_config = LoraConfig(**cfg["lora_config"])
            model = get_peft_model(model, lora_config)

        data_collator = DataCollatorForMultiLabelTokenClassification(
            pad_token_id=tokenizer.pad_token_id,
            max_length=cfg["dataset_args"]["max_length"],
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
            callbacks=[EarlyStoppingCallback(patience=cfg["early_stopping_patience"])],
            pos_weight=train_dataset.pos_weight,
            **cfg.get("trainer_args", {}),
        )

    elif cfg["dataset_args"].get("task") == "supervised_fine_tuning":
        # Convert datasets to huggingface datasets
        train_dataset = train_dataset.to_hf_dataset()
        val_dataset = val_dataset.to_hf_dataset()
        # TODO: Add evaluation pipeline for gen AI models
        # test_dataset = test_dataset.to_hf_dataset()

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(cfg["model_id"])
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])

        # Load lora if enabled
        if cfg.get("use_lora"):
            lora_config = LoraConfig(**cfg["lora_config"])
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
            **cfg.get("trainer_args", {}),
        )

    else:
        raise ValueError("Task not supported")

    # Execute training
    ## Either resume or start new training loop
    if cfg.get("resume_from_checkpoint") is not None:
        resume_from_checkpoint = cfg["resume_from_checkpoint"]
    else:
        resume_from_checkpoint = (
            True if (is_not_empty(cfg["results_path"] / "checkpoints")) else False
        )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Evaluate on test set if necessary
    if cfg["dataset_args"].get("task") in [
        "token_classification",
        "sequence_classification",
    ]:
        trainer.eval_dataset = test_dataset
        test_set_metrics = trainer.evaluate()
        test_set_metrics["run_name"] = cfg["run_name"]
        test_set_metrics.update(
            {
                "model_id": cfg["model_id"],
                "batch_effective": cfg["training_args"]["per_device_train_batch_size"]
                * cfg["training_args"]["gradient_accumulation_steps"],
                "patience": cfg["early_stopping_patience"],
                "pos_weight": train_dataset.pos_weight.tolist(),
            }
        )

        if cfg["dataset_args"].get("task") == "token_classification":
            test_set_metrics["annotation_type"] = cfg["dataset_args"][
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
            label_type = cfg["dataset_args"].get("label_type", "tags")
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
            threshold_map, **cfg.get("threshold_selection", {})
        )
        trainer.model.config.thresholds = thresholds.to_dict()

        # Save metrics
        metrics_path = cfg["results_path"] / "metrics.jsonl"
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

            test_set_metrics["run_name"] = cfg["run_name"]
            test_set_metrics["threshold_map"] = threshold_map
            test_set_metrics["selected_thresholds"] = thresholds.to_dict()
            writer.write(test_set_metrics)

    # Save best model
    trainer.save_model(cfg["results_path"] / "best_model")
    trainer.save_state()
