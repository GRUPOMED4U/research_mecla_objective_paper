"""
This script is for hyperparameter search with optuna for an experiment defined as a .yaml file
"""

import argparse
import json
from pathlib import Path
import shutil
from typing import Dict
import optuna
from optuna.storages import RDBStorage
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
)
import jsonlines

# Custom modules
from spesia_research.config import load_exp_config
from spesia_research.datasets import (
    ClinicalRecordsDataset,
    DataCollatorForMultiLabelTokenClassification,
)
from spesia_research.trainers import MultiLabelTokenTrainer
from spesia_research.metrics import compute_metrics
from spesia_research.logs import configure_logging, get_logger
from spesia_research.parser_utils import deep_merge, parse_kv_list


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


if __name__ == "__main__":
    # Configure logging
    configure_logging()
    logger = get_logger("hpsearch")

    # Load config and custom options
    args = build_parser().parse_args()
    overrides = parse_kv_list(args.set)
    config = load_exp_config(args.config)
    config = deep_merge(config, overrides)

    # Global variables and set directories
    exp_path = Path(args.config)
    results_path = Path("experiments") / exp_path.stem
    results_path.mkdir(exist_ok=True)
    model_id = config["model_id"]

    logger.debug(f"Loaded config: {exp_path}")
    logger.debug(config)

    # Load datasets
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets: Dict[str, ClinicalRecordsDataset] = {
        "train": None,
        "val": None,
        "test": None,
    }
    for split in datasets.keys():
        datasets[split] = ClinicalRecordsDataset.from_exp_config(
            config=config,
            split=split,
            tokenizer=tokenizer,
        )

    train_dataset, val_dataset, test_dataset = (
        datasets["train"],
        datasets["val"],
        datasets["test"],
    )

    logger.info(f"Dataset args: {config['dataset_args']}")

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

    # Define optuna study
    for hpsearch_config in config.get("hyperparameter_search", {}).values():
        study_name = f"{exp_path.stem}_{'_'.join(hpsearch_config['metrics'])}_data_split_seed_{train_dataset.random_seed}"
        storage_path = "sqlite:///optuna.db"

        # skip if already done
        hpsearch_path = (
            results_path
            / f"hpsearch_{'_'.join(hpsearch_config['metrics'])}_data_split_seed_{train_dataset.random_seed}.jsonl"
        )
        if hpsearch_path.exists():
            logger.info(
                f"Skipping hyperparameter search for {study_name}. File {hpsearch_path} already exists."
            )
            continue

        logger.info(f"Running hyperparameter search for {study_name}")

        # Define persistent storage
        storage = RDBStorage(storage_path)

        study = optuna.create_study(
            study_name=study_name,
            directions=hpsearch_config["directions"],
            storage=storage,
            load_if_exists=hpsearch_config["load_if_exists"],
        )

        def objective(trial):
            model = AutoModelForTokenClassification.from_pretrained(
                model_id, num_labels=train_dataset.num_labels
            )

            config["training_args"]["num_train_epochs"] = 3

            # Suggest hyperparameters in training args
            training_args_hpsearch_params = ["learning_rate", "weight_decay"]
            logger.info("Suggested hyperparameters:")
            for param in training_args_hpsearch_params:
                if param in hpsearch_config["parameters"]:
                    # get suggestion method
                    config["training_args"][param] = getattr(
                        trial, hpsearch_config["parameters"][param]["method"]
                    )
                    # get suggested value
                    config["training_args"][param] = config["training_args"][param](
                        **hpsearch_config["parameters"][param]["args"]
                    )
                    logger.info(
                        f"Selected hyperparameter -> {param}: {config['training_args'][param]}"
                    )

            training_args = TrainingArguments(
                output_dir="./output", **config["training_args"]
            )
            training_args.save_total_limit = 0

            data_collator = DataCollatorForMultiLabelTokenClassification(
                pad_token_id=tokenizer.pad_token_id,
                max_length=config["dataset_args"]["max_length"],
                num_labels=train_dataset.num_labels,
            )

            trainer_args_hpsearch_params = ["mecla_amplification_factor"]
            for param in trainer_args_hpsearch_params:
                if param in hpsearch_config["parameters"]:
                    # get suggestion method
                    config["trainer_args"][param] = getattr(
                        trial, hpsearch_config["parameters"][param]["method"]
                    )
                    # get suggested value
                    config["trainer_args"][param] = config["trainer_args"][param](
                        **hpsearch_config["parameters"][param]["args"]
                    )
                    logger.info(
                        f"Selected hyperparameter -> {param}: {config['trainer_args'][param]}"
                    )

            trainer = MultiLabelTokenTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                processing_class=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                pos_weight=train_dataset.pos_weight,
                **config.get("trainer_args", {}),
            )

            # Train
            trainer.train()

            # Get the metrics you want to optimize
            eval_results = trainer.evaluate()

            # Return the metrics to Optuna
            if len(hpsearch_config["metrics"]) == 1:
                return eval_results[hpsearch_config["metrics"][0]]

            return (eval_results[metric] for metric in hpsearch_config["metrics"])

        # Execute hpsearch
        study.optimize(objective, n_trials=hpsearch_config["n_trials"])

        # Save best params
        with jsonlines.open(hpsearch_path, "w") as writer:
            writer.write(
                {
                    "best_trial_number": study.best_trial.number,
                    "best_trial_value": study.best_trial.value,
                    "best_trial_params": study.best_trial.params,
                }
            )

        # Clear output directory
        try:
            shutil.rmtree("./output")
            logger.info(f"Folder '{'./output'}' and all its contents deleted.")
        except OSError as e:
            logger.error(f"Error deleting non-empty folder: {e}")
