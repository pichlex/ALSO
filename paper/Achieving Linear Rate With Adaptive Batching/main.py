import argparse
import json
from copy import deepcopy
from typing import Dict, Any

from data import get_dataloaders
from train import train_model
from tune import run_tuning, load_tuned_params
from tune import _optimizer_signature


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def apply_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "dataset": "cifar10",
        "model": "resnet18",
        "batch_size": 64,
        "epochs": 20,
        "optimizer": "sgd",
        "lr": 0.1,
        "momentum": 0.9,
        "nesterov": False,
        "weight_decay": 5e-4,
        "adaptive_batch": False,
        "adaptive_batch_beta": 0.0,
        "batch_size_multiplier": 1.0,
        "adaptive_batch_min": 10,
        "adaptive_batch_max": 1024,
        "epoch_start_ab": 2,
        "adaptive_batch_strategy": "variance_ratio",  # "variance_ratio" | "adabatchgrad" | "divebatch"
        "adabatchgrad_batch_test": "random_increase",
        "adabatchgrad_theta": 0.1,
        "adabatchgrad_nu": 0.1,
        "adabatchgrad_prob_new": 0.005,
        "adabatchgrad_k": 5,
        "adabatchgrad_alpha": 1.0,
        "adabatchgrad_beta": 1.0,
        "adabatchgrad_power_eps": 0.0,
        "divebatch_delta": 0.1,
        "divebatch_max_batch": 2048,
        "divebatch_lr_rescale": False,
        "divebatch_eps": 1e-12,
        "divebatch_microbatch": 8,
        "augment": True,
        "num_workers": 2,
        "seed": 42,
        "mlflow_experiment": "Achieving Linear Rate",
        "report_to": "mlflow",
        "tune_runs": 100,
        "n_epoches_tune": 5,
        "tune_name": "study",
        "use_old_tune_params": True,
        "scheduler": None,
        "scheduler_milestones": [30, 60],
        "scheduler_gamma": 0.1,
    }
    merged = defaults.copy()
    merged.update(cfg)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Adaptive batching experiments.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning (fixed batch).")
    parser.add_argument(
        "--use-tuned",
        action="store_true",
        help="Load tuned hyperparameters before training.",
    )
    args = parser.parse_args()

    raw_cfg = load_config(args.config)
    config = apply_defaults(raw_cfg)

    if args.tune:
        tuned_cfg = run_tuning(config)
        print("Tuning complete. Best params:", tuned_cfg)
        return

    if args.use_tuned:
        signature = _optimizer_signature(config)
        tuned = load_tuned_params(
            config.get("dataset", "cifar10"), config.get("tune_name", "study"), signature
        )
        if tuned is not None:
            config.update(tuned)

    train_loader, val_loader, test_loader = get_dataloaders(config)
    result = train_model(config, train_loader, val_loader, test_loader)
    print("Training finished. Best metrics:", result["best_test"])


if __name__ == "__main__":
    main()
