"""
Main experiment script for unbalanced CIFAR10 binary classification experiments.

This script implements the experiments from Section 5.1 of the paper
"Aligning Distributionally Robust Optimization with Practical Deep Learning Needs".
It compares various optimization approaches on an artificially unbalanced CIFAR10 dataset.

This version has been refactored to use the 'click' library for command-line
argument parsing.
"""

import json
import os.path
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Union

import click
import numpy as np
import mlflow
import optuna
from scipy import stats

from optimizers.large_scale_dro import RobustLoss
from problems import get_problem
from utils import ImportanceLoss, train


def run_optimization(
    config: Dict[str, Any],
    metrics: Optional[Dict[str, List[float]]] = None,
    tuning: bool = False,
) -> Union[float, Dict[str, List[float]]]:
    """
    Run a single optimization experiment with the given configuration.

    This function sets up the model, optimizer, and datasets according to the provided
    configuration, then trains the model and evaluates its performance.

    Args:
        config: Dictionary containing experiment configuration parameters
        metrics: Optional dictionary to store evaluation metrics
        tuning: Whether this run is for hyperparameter tuning

    Returns:
        If tuning is True, returns the maximum validation F1 score.
        Otherwise, returns the updated metrics dictionary.
    """
    (
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        loss_fn,
        device,
        compute_weights_fn,
    ) = get_problem(config)

    mlflow_enabled = config.get("report_to") == "mlflow" and not tuning
    mlflow_ctx = nullcontext()
    if mlflow_enabled:
        experiment_base = config.get("mlflow_experiment", "Learning From Unbalanced Data")
        mlflow.set_experiment(f"{experiment_base}_seed{config['seed']}")
        run_title = (
            f"{config.get('run_name', config['optimizer'])}"
            f"_uc{config.get('unbalance_coef', 'na')}_seed{config['seed']}"
        )
        mlflow_ctx = mlflow.start_run(run_name=run_title)

    start = time.monotonic()
    with mlflow_ctx:
        if mlflow_enabled:
            loggable_params = {
                k: v
                for k, v in config.items()
                if isinstance(v, (int, float, str, bool))
            }
            mlflow.log_params(loggable_params)
            mlflow.set_tags(
                {
                    "optimizer": config.get("optimizer"),
                    "unbalance_coef": config.get("unbalance_coef"),
                    "seed": config.get("seed"),
                }
            )
        if config["use_exp"]:
            # Apply importance-weighted loss with exponential transformation
            loss_fn = ImportanceLoss(
                loss_fn,
                tau=config["tau"],
                C=config["C"],
                warmup_steps=config["exp_warmup_steps"],
            )
        if config["use_ls_dro"]:
            # Apply large-scale DRO loss wrapper
            loss_fn = RobustLoss(
                base_loss_fn=loss_fn, size=0.99, reg=1e-5, geometry="cvar"
            )
        _, val_results, test_results, pi_history = train(
            model,
            optimizer,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            loss_fn,
            device,
            config,
            tuning=tuning,
            compute_weights_fn=compute_weights_fn,
        )

        if tuning:
            # For hyperparameter tuning, return the best validation F1 score
            return np.max(val_results["f1"])

        # For evaluation, record metrics at the epoch with best validation F1
        idx = np.argmax(val_results["f1"])
        for metric in test_results:
            metrics[metric].append(test_results[metric][idx])
        end = time.monotonic()
        metrics["time"].append(end - start)

        if mlflow_enabled:
            mlflow.log_metric("best_epoch", int(idx))
            for metric in test_results:
                mlflow.log_metric(f"best_test_{metric}", test_results[metric][idx])
                mlflow.log_metric(f"best_val_{metric}", val_results[metric][idx])
            mlflow.log_metric("duration_sec", end - start)
            mlflow.log_dict(
                {"val": val_results, "test": test_results}, "metrics_history.json"
            )
            if pi_history:
                mlflow.log_dict({"pi_history": pi_history}, "pi_history.json")
        return metrics


def tune_params(
    config: Dict[str, Any], name: Optional[str] = None, use_old_tune_params: bool = True
) -> Dict[str, Any]:
    """
    Tune hyperparameters using Optuna and save/load results.

    This function either loads previously tuned parameters from disk or
    runs a new hyperparameter tuning process using Optuna, based on the
    specified optimizer and configuration.

    Args:
        config: Dictionary containing base experiment configuration
        name: Name prefix for the saved parameter file
        use_old_tune_params: Whether to load existing parameters if available

    Returns:
        Updated config dictionary with tuned hyperparameters
    """
    f_name = f'tuned_params/{name}_{config["unbalance_coef"]}.json'
    if os.path.exists(f_name) and use_old_tune_params:
        try:
            with open(f_name) as f:
                params = json.load(f)
                for key in params.keys():
                    config[key] = params[key]
            return config
        except json.decoder.JSONDecodeError:
            pass

    study = optuna.create_study(
        direction="maximize", study_name=f"{name}"}"
    )

    def tune_function(trial):
        """Define parameter search space for Optuna."""
        # Learning rate and weight decay for all optimizers
        config["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        config["weight_decay"] = trial.suggest_float(
            "weight_decay", 1e-6, 1e-2, log=True
        )

        # Additional parameters for ALSO
        if config["optimizer"] in ["also"]:
            config["pi_lr"] = trial.suggest_float("pi_lr", 1e-5, 1e-3, log=True)
            config["pi_decay"] = trial.suggest_float("pi_decay", 1e-3, 1.0, log=True)

        # Parameters for exponential loss transformation
        if config["use_exp"]:
            config["C"] = trial.suggest_float("C", 1e-1, 1e1, log=True)
            config["tau"] = trial.suggest_float("tau", 1e-4, 1e1, log=True)
            config["exp_warmup_steps"] = trial.suggest_int("exp_warmup_steps", 10, 200)

        return run_optimization(config, None, tuning=True)

    # Run optimization with specified number of trials
    study.optimize(tune_function, n_trials=config["tune_runs"])

    # Save best parameters to disk
    with open(f_name, "w") as f:
        json.dump(study.best_trial.params, f)

    # Load parameters and update config
    with open(f_name) as f:
        params = json.load(f)
        for key in params.keys():
            config[key] = params[key]

    return config


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--device", type=int, default=None, help="Device index for GPU.", show_default=True)
@click.option("--config-path", "config_path", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True), default="config.json", help="Path to the JSON configuration file.", show_default=True)
@click.option("--balanced-test", is_flag=True, help="If set, use a balanced test set.")
@click.option("--tune", is_flag=True, help="If set, run hyperparameter tuning using Optuna.")
@click.option("--use-old-tune-params", is_flag=True, help="If set, use previously saved tuning parameters if they exist.")
@click.option("--augment", is_flag=True, help="If set, use data augmentation.")
@click.option("--seed", type=int, default=None, help="Base seed for experiments.")
def main(
    device: Optional[int],
    config_path: str,
    balanced_test: bool,
    tune: bool,
    use_old_tune_params: bool,
    augment: bool,
    seed: Optional[int],
):
    """Main entry point for running the Unbalanced CIFAR10 experiments.

    This function sets up and runs experiments based on a JSON config file
    and command-line arguments. It supports hyperparameter tuning and
    evaluation across different optimization strategies.
    """
    with open(config_path) as f:
        config = json.load(f)

    # Update config with command-line arguments
    config["device_id"] = device
    config["balanced_test"] = balanced_test
    config["augment"] = augment
    if seed is not None:
        config["seed"] = seed
    base_seed = config.get("seed", 0)
    unbalance_coef_list = config["unbalance_coefs"]

    # Set default optimization parameters
    config["mode"] = "optimistic"
    config["optimizer_mode"] = "optimistic"
    config["use_adam"] = True
    if "report_to" not in config.keys():
        config["report_to"] = "mlflow"
    config.setdefault("mlflow_experiment", "Learning From Unbalanced Data")

    # Define experiment configurations to run
    # Each tuple represents: (optimizer_name, use_sampler, use_static_weights, use_exp, use_init_static_weights, use_ls_dro)
    experiment_list = [
        # Main method from the paper:
        ("also", False, False, False, True, False),  # ALSO with initialized static weights
        # To run other experiments, uncomment them below:
        # --- Standard approaches ---
        # ("adam", False, False, False, False, False),   # standard Adam
        # ("adam", True, False, False, False, False),    # Adam with upsampling
        # ("adam", False, True, False, False, False),    # Adam with static weights
        # --- Other DRO methods ---
        # ("dro_loss", False, False, False, False, False),   # DRO loss function
        # ("adam", False, False, False, False, True),   # Large Scale loss function
        # ("adam", False, False, True, False, False),    # RECOVER
        # ("drago", False, False, False, False, False),   # DRAGO optimizer
    ]

    # Run experiments for each imbalance coefficient
    for unbalance_coef in unbalance_coef_list:
        config["unbalance_coef"] = unbalance_coef
        print(
            f"===================== unbalance_coefficient: {unbalance_coef} ======================="
        )
        metrics = {}

        # Run each experiment configuration
        for onm, usmlpr, ustw, uex, uistw, ulsdro in experiment_list:
            config["optimizer"] = onm
            config["use_sampler"] = usmlpr
            config["use_static_weights"] = ustw
            config["use_exp"] = uex
            config["use_init_static_weights"] = uistw
            config["use_ls_dro"] = ulsdro

            # Build run name based on configuration
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

            tune_name = run_name
            run_name = f"{run_name}"
            if run_name not in metrics.keys():
                metrics[run_name] = defaultdict(list)
            config["run_name"] = run_name

            # Run multiple trials with different random seeds
            for i in range(config["eval_runs"]):
                config["seed"] = base_seed

                # Tune hyperparameters if requested or on the first run of a new experiment
                if tune or i == 0:
                    if i == 0:
                        config = tune_params(config, tune_name, use_old_tune_params)
                    else:
                        config = tune_params(config, tune_name, True)

                # Run the optimization and collect metrics
                metrics[run_name] = run_optimization(config, metrics[run_name])

            # Report results for this run configuration
            print(f"~~~~~~~~~~~ Run name {run_name} ~~~~~~~~~~~")
            for metric_name in metrics[run_name].keys():
                res = np.array(metrics[run_name][metric_name])
                print(f"{metric_name}: {res.mean()}+-{res.std()}")


if __name__ == "__main__":
    main()
