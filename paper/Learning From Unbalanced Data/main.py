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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click
import numpy as np
import optuna
from scipy import stats

from optimizers.large_scale_dro import RobustLoss
from problems import get_problem
from utils import ImportanceLoss, train


def _to_serializable(value: Any) -> Any:
    """Convert values to JSON/MLflow friendly formats."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return str(value)


def _filter_params_for_logging(config: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only simple types for parameter logging."""
    params = {}
    for k, v in config.items():
        params[k] = _to_serializable(v)
    return params


def run_optimization(
    config: Dict[str, Any],
    metrics: Optional[Dict[str, List[float]]] = None,
    tuning: bool = False,
    run_dir: Optional[Path] = None,
    mlflow_client: Optional[Any] = None,
    log_artifacts: bool = False,
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
    config_snapshot = _filter_params_for_logging(config)
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

    start = time.monotonic()
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
        loss_fn = RobustLoss(base_loss_fn=loss_fn, size=0.99, reg=1e-5, geometry="cvar")
    model, val_results, test_results, train_losses = train(
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
        mlflow_client=mlflow_client,
    )

    if tuning:
        # For hyperparameter tuning, return the best validation F1 score
        return np.max(val_results["f1"])

    # For evaluation, record metrics at the epoch with best validation F1
    idx = np.argmax(val_results["f1"])
    for metric in test_results:
        metrics[metric].append(test_results[metric][idx])
    end = time.monotonic()
    run_time = end - start
    metrics["time"].append(run_time)

    if mlflow_client is not None:
        mlflow_client.log_metric("best_val_f1", float(val_results["f1"][idx]), step=int(idx))
        mlflow_client.log_metrics(
            {f"best_test_{metric}": float(test_results[metric][idx]) for metric in test_results},
            step=int(idx),
        )
        mlflow_client.log_metric("runtime_sec", float(run_time))

    if log_artifacts and run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

        history = {
            "train_loss": [float(x) for x in train_losses],
            "val": {k: [float(x) for x in vals] for k, vals in val_results.items()},
            "test": {k: [float(x) for x in vals] for k, vals in test_results.items()},
        }
        history_path = run_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f)

        best_metrics = {
            "best_epoch": int(idx),
            "best_val": {k: float(val_results[k][idx]) for k in val_results},
            "best_test": {k: float(test_results[k][idx]) for k in test_results},
            "runtime_sec": float(run_time),
        }
        best_metrics_path = run_dir / "best_metrics.json"
        with open(best_metrics_path, "w") as f:
            json.dump(best_metrics, f)

        config_path = run_dir / "config_used.json"
        with open(config_path, "w") as f:
            json.dump(config_snapshot, f)

        tuned_params_path = None
        if "run_name" in config and "unbalance_coef" in config:
            candidate = (
                Path("tuned_params") / f"{config['run_name']}_{config['unbalance_coef']}.json"
            )
            if candidate.exists():
                tuned_params_path = candidate

        if mlflow_client is not None:
            mlflow_client.log_artifact(str(history_path))
            mlflow_client.log_artifact(str(best_metrics_path))
            mlflow_client.log_artifact(str(config_path))
            if tuned_params_path is not None:
                mlflow_client.log_artifact(str(tuned_params_path))

    return metrics


def tune_params(
    config: Dict[str, Any],
    name: Optional[str] = None,
    use_old_tune_params: bool = True,
    mlflow_client: Optional[Any] = None,
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
            if mlflow_client is not None:
                mlflow_client.log_artifact(f_name)
            return config
        except json.decoder.JSONDecodeError:
            pass

    study = optuna.create_study(
        direction="maximize", study_name=f"{name}_{config['unbalance_coef']}"
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

    if mlflow_client is not None:
        mlflow_client.log_artifact(f_name)

    return config


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--device", type=int, default=None, help="Device index for GPU.", show_default=True)
@click.option("--config-path", "config_path", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True), default="config.json", help="Path to the JSON configuration file.", show_default=True)
@click.option("--balanced-test", is_flag=True, help="If set, use a balanced test set.")
@click.option("--tune", is_flag=True, help="If set, run hyperparameter tuning using Optuna.")
@click.option("--use-old-tune-params", is_flag=True, help="If set, use previously saved tuning parameters if they exist.")
@click.option("--augment", is_flag=True, help="If set, use data augmentation.")
@click.option("--mlflow", "use_mlflow", is_flag=True, help="If set, log metrics and artifacts to MLflow.")
@click.option("--mlflow-uri", default="file:./mlruns", show_default=True, help="MLflow tracking URI.")
@click.option(
    "--mlflow-experiment",
    default="learning_unbalanced_cifar",
    show_default=True,
    help="MLflow experiment name.",
)
@click.option(
    "--mlflow-run-dir",
    default="runs",
    show_default=True,
    help="Local directory to store run artifacts before uploading to MLflow.",
)
def main(
    device: Optional[int],
    config_path: str,
    balanced_test: bool,
    tune: bool,
    use_old_tune_params: bool,
    augment: bool,
    use_mlflow: bool,
    mlflow_uri: str,
    mlflow_experiment: str,
    mlflow_run_dir: str,
):
    """Main entry point for running the Unbalanced CIFAR10 experiments.

    This function sets up and runs experiments based on a JSON config file
    and command-line arguments. It supports hyperparameter tuning and
    evaluation across different optimization strategies.
    """
    with open(config_path) as f:
        config = json.load(f)

    mlflow_client = None
    run_root = Path(mlflow_run_dir)
    if use_mlflow:
        try:
            import mlflow as mlflow_module
        except ImportError as exc:
            raise click.ClickException(
                "MLflow is not installed. Install it (e.g., `uv pip install mlflow`) or disable --mlflow."
            ) from exc
        mlflow_client = mlflow_module
        mlflow_client.set_tracking_uri(mlflow_uri)
        mlflow_client.set_experiment(mlflow_experiment)
        run_root.mkdir(parents=True, exist_ok=True)

    # Update config with command-line arguments
    config["device_id"] = device
    config["balanced_test"] = balanced_test
    config["augment"] = augment
    unbalance_coef_list = config["unbalance_coefs"]

    # Set default optimization parameters
    config["mode"] = "optimistic"
    config["optimizer_mode"] = "optimistic"
    config["use_adam"] = True
    if "report_to" not in config.keys():
        config["report_to"] = "none"

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
            config["run_name"] = run_name

            if run_name not in metrics.keys():
                metrics[run_name] = defaultdict(list)

            summary_tags = {
                "optimizer": config["optimizer"],
                "unbalance_coef": unbalance_coef,
                "balanced_test": balanced_test,
                "augment": augment,
            }
            outer_ctx = (
                mlflow_client.start_run(
                    run_name=f"{run_name}_k{unbalance_coef}_summary", tags=summary_tags
                )
                if mlflow_client is not None
                else nullcontext()
            )

            with outer_ctx:
                # Run multiple trials with different random seeds
                for i, seed in enumerate(range(config["eval_runs"])):
                    config["seed"] = seed
                    run_dir = run_root / f"{run_name}_k{unbalance_coef}_seed{seed}"

                    inner_ctx = (
                        mlflow_client.start_run(
                            run_name=f"{run_name}_seed{seed}_k{unbalance_coef}",
                            nested=True,
                            tags={**summary_tags, "seed": seed},
                        )
                        if mlflow_client is not None
                        else nullcontext()
                    )

                    with inner_ctx:
                        # Tune hyperparameters if requested or on the first run of a new experiment
                        if tune or i == 0:
                            if i == 0:
                                config = tune_params(
                                    config,
                                    run_name,
                                    use_old_tune_params,
                                    mlflow_client=mlflow_client,
                                )
                            else:
                                config = tune_params(
                                    config, run_name, True, mlflow_client=mlflow_client
                                )

                        if mlflow_client is not None:
                            mlflow_client.log_params(
                                _filter_params_for_logging(
                                    {
                                        "optimizer": config["optimizer"],
                                        "use_sampler": config["use_sampler"],
                                        "use_static_weights": config["use_static_weights"],
                                        "use_exp": config["use_exp"],
                                        "use_init_static_weights": config[
                                            "use_init_static_weights"
                                        ],
                                        "use_ls_dro": config["use_ls_dro"],
                                        "lr": config.get("lr"),
                                        "weight_decay": config.get("weight_decay"),
                                        "pi_lr": config.get("pi_lr"),
                                        "pi_decay": config.get("pi_decay"),
                                        "C": config.get("C"),
                                        "tau": config.get("tau"),
                                        "exp_warmup_steps": config.get("exp_warmup_steps"),
                                        "batch_size": config["batch_size"],
                                        "n_epoches": config["n_epoches"],
                                        "seed": seed,
                                        "unbalance_coef": unbalance_coef,
                                        "balanced_test": balanced_test,
                                        "augment": augment,
                                        "model": config["model"],
                                    }
                                )
                            )

                        # Run the optimization and collect metrics
                        metrics[run_name] = run_optimization(
                            config,
                            metrics[run_name],
                            run_dir=run_dir,
                            mlflow_client=mlflow_client,
                            log_artifacts=use_mlflow,
                        )

                if mlflow_client is not None and metrics.get(run_name):
                    summary_payload = {
                        metric_name: {
                            "mean": float(np.array(metrics[run_name][metric_name]).mean()),
                            "std": float(np.array(metrics[run_name][metric_name]).std()),
                        }
                        for metric_name in metrics[run_name].keys()
                    }
                    mlflow_client.log_metrics(
                        {
                            f"{metric_name}_mean": payload["mean"]
                            for metric_name, payload in summary_payload.items()
                        }
                    )
                    mlflow_client.log_metrics(
                        {
                            f"{metric_name}_std": payload["std"]
                            for metric_name, payload in summary_payload.items()
                        }
                    )
                    summary_path = run_root / f"{run_name}_k{unbalance_coef}_summary.json"
                    with open(summary_path, "w") as f:
                        json.dump(summary_payload, f)
                    mlflow_client.log_artifact(str(summary_path))

            # Report results for this run configuration
            print(f"~~~~~~~~~~~ Run name {run_name} ~~~~~~~~~~~")
            for metric_name in metrics[run_name].keys():
                res = np.array(metrics[run_name][metric_name])
                print(f"{metric_name}: {res.mean()}+-{res.std()}")


if __name__ == "__main__":
    main()
