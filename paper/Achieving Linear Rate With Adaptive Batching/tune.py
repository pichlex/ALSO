import json
import os
from typing import Dict, Any, Optional

import optuna

from data import get_dataloaders
from train import train_model


def _optimizer_signature(config: Dict[str, Any]) -> str:
    opt = config.get("optimizer", "sgd").lower()
    if opt == "sgd":
        mom_on = config.get("momentum", 0) > 0
        nest_on = bool(config.get("nesterov", False))
        wd_on = config.get("weight_decay", 0) > 0
        return f"sgd_mom-{'on' if mom_on else 'off'}_nest-{'on' if nest_on else 'off'}_wd-{'on' if wd_on else 'off'}"
    return opt


def _tuned_params_path(dataset: str, name: str, signature: str) -> str:
    base_dir = os.path.join(os.path.dirname(__file__), "tuned_params", dataset, signature)
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{name}.json")


def load_tuned_params(dataset: str, name: str, signature: str) -> Optional[Dict[str, Any]]:
    path = _tuned_params_path(dataset, name, signature)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_tuned_params(dataset: str, name: str, signature: str, params: Dict[str, Any]) -> None:
    path = _tuned_params_path(dataset, name, signature)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)


def run_tuning(base_config: Dict[str, Any]) -> Dict[str, Any]:
    dataset = base_config.get("dataset", "cifar10").lower()
    tune_name = base_config.get("tune_name", "study")
    use_old = bool(base_config.get("use_old_tune_params", True))
    signature = _optimizer_signature(base_config)
    existing = load_tuned_params(dataset, tune_name, signature) if use_old else None
    if existing is not None:
        cfg = base_config.copy()
        cfg.update(existing)
        return cfg

    def objective(trial: optuna.trial.Trial) -> float:
        config = base_config.copy()
        config["adaptive_batch"] = False
        config["epochs"] = int(config.get("n_epoches_tune", 5))
        config["report_to"] = None
        config["optimizer"] = config.get("optimizer", "sgd").lower()
        if config["optimizer"] == "sgd":
            config["lr"] = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
            config["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            config["momentum"] = trial.suggest_float("momentum", 0.5, 0.99)
            config["nesterov"] = trial.suggest_categorical("nesterov", [True, False])
        elif config["optimizer"] == "adamw":
            config["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            config["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        train_loader, val_loader, test_loader = get_dataloaders(config)
        result = train_model(config, train_loader, val_loader, test_loader)
        return result["best_val_f1"]

    study = optuna.create_study(direction="maximize", study_name=tune_name)
    study.optimize(objective, n_trials=int(base_config.get("tune_runs", 100)))
    best_params = study.best_params
    save_tuned_params(dataset, tune_name, signature, best_params)
    cfg = base_config.copy()
    cfg.update(best_params)
    return cfg
