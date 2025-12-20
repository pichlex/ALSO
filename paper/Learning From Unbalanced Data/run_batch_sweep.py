"""Batch sweep runner for Unbalanced CIFAR10 experiments.

Runs the existing experiment pipeline sequentially for a single seed across
multiple batch sizes and logs aggregated metrics to a JSONL file.
"""

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from main import run_optimization, tune_params


def _build_base_config(config_path: Path, seed: int, device: int, balanced_test: bool, augment: bool) -> Dict:
    with config_path.open() as f:
        base_config = json.load(f)

    base_config["device_id"] = device
    base_config["balanced_test"] = balanced_test
    base_config["augment"] = augment
    base_config["seed"] = seed
    base_config.setdefault("mode", "optimistic")
    base_config.setdefault("optimizer_mode", "optimistic")
    base_config.setdefault("use_adam", True)
    base_config.setdefault("pi_temperature", 1.0)
    base_config.setdefault("cached_batch", False)
    base_config.setdefault("report_to", "mlflow")
    base_config.setdefault("mlflow_experiment", "Learning From Unbalanced Data Batch Sweep")
    return base_config


def _experiment_list() -> List[tuple]:
    # Each tuple: (optimizer_name, use_sampler, use_static_weights, use_exp, use_init_static_weights, use_ls_dro)
    return [
        ("also", True, False, False, True, False),  # ALSO with initialized static weights
    ]


def _format_run_name(config: Dict) -> str:
    run_name = config["optimizer"]
    if config["use_sampler"]:
        run_name += "_upsampling"
    if config["use_static_weights"]:
        run_name += "_static_weights"
    if config["use_exp"]:
        run_name += "_exp"
    if config["use_init_static_weights"]:
        run_name += "_init_sw"
    if config["use_ls_dro"]:
        run_name += "_ls_dro"
    return run_name


def _aggregate_metrics(metric_lists: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    aggregated = {}
    for metric_name, values in metric_lists.items():
        res = np.array(values)
        aggregated[metric_name] = {"mean": float(res.mean()), "std": float(res.std())}
    return aggregated


def run_batch_sweep(
    config_path: Path,
    seed: int,
    batch_sizes: Iterable[int],
    output_path: Path,
    device: int,
    balanced_test: bool,
    augment: bool,
    tune: bool,
    use_old_tune_params: bool,
):
    base_config = _build_base_config(config_path, seed, device, balanced_test, augment)
    unbalance_coef_list = base_config["unbalance_coefs"]
    dataset_name = base_config.get("dataset", "cifar10").lower()
    experiment_list = _experiment_list()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f_out:
        for batch_size in batch_sizes:
            for unbalance_coef in unbalance_coef_list:
                config = deepcopy(base_config)
                config["batch_size"] = batch_size
                config["unbalance_coef"] = unbalance_coef
                if config.get("pi_strategy") == "uc":
                    config["batch_size"] = int(batch_size * unbalance_coef)

                print(
                    f"Batch size {batch_size} | unbalance_coefficient: {unbalance_coef}"
                )

                metrics: Dict[str, Dict[str, List[float]]] = {}
                for onm, usmlpr, ustw, uex, uistw, ulsdro in experiment_list:
                    config["optimizer"] = onm
                    config["use_sampler"] = usmlpr
                    config["use_static_weights"] = ustw
                    config["use_exp"] = uex
                    config["use_init_static_weights"] = uistw
                    config["use_ls_dro"] = ulsdro

                    run_name = _format_run_name(config)
                    tune_name = run_name

                    if run_name not in metrics:
                        metrics[run_name] = defaultdict(list)
                    config["run_name"] = run_name

                    for i in range(config["eval_runs"]):
                        config["seed"] = seed
                        if tune or i == 0:
                            config = tune_params(config, tune_name, use_old_tune_params if i == 0 else True)
                        metrics[run_name] = run_optimization(config, metrics[run_name])

                    aggregated = _aggregate_metrics(metrics[run_name])
                    record = {
                        "seed": seed,
                        "batch_size": batch_size,
                        "unbalance_coef": unbalance_coef,
                        "run_name": run_name,
                        "dataset": dataset_name,
                        "metrics": aggregated,
                    }
                    f_out.write(json.dumps(record) + "\n")
                    f_out.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch size sweep for a single seed and log metrics to JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("paper/Learning From Unbalanced Data/configs/config_use_sampler_bs=64.json"),
        help="Path to the base JSON config.",
    )
    parser.add_argument("--seed", type=int, required=True, help="Seed to run.")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        help="Batch sizes to sweep.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to runs/batch_sweep_seed{seed}.jsonl next to main.py.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Device index for GPU (None for default).",
    )
    parser.add_argument(
        "--balanced-test",
        action="store_true",
        help="Use a balanced test set.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna tuning before evaluation.",
    )
    parser.add_argument(
        "--use-old-tune-params",
        action="store_true",
        help="Reuse existing tuned params if available.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    output = args.output
    if output is None:
        default_runs = script_dir / "batch_sweep" / "runs"
        # Determine dataset from config to separate outputs
        try:
            with args.config_path.open() as f:
                cfg = json.load(f)
                dataset_name = cfg.get("dataset", "cifar10").lower()
        except Exception:
            dataset_name = "cifar10"
        output = default_runs / dataset_name / f"batch_sweep_seed{args.seed}.jsonl"
    run_batch_sweep(
        config_path=args.config_path,
        seed=args.seed,
        batch_sizes=args.batch_sizes,
        output_path=output,
        device=args.device,
        balanced_test=args.balanced_test,
        augment=args.augment,
        tune=args.tune,
        use_old_tune_params=args.use_old_tune_params,
    )
