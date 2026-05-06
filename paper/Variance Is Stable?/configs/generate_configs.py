from __future__ import annotations

from pathlib import Path

import yaml


BATCH_SIZES = [16, 64, 256, 1024]
CONFIG_DIR = Path(__file__).resolve().parent


def _metric_microbatch_size(batch_size: int) -> int:
    return min(batch_size, 16)


def _write_config(filename: str, payload: dict) -> None:
    path = CONFIG_DIR / filename
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _resnet_config(dataset: str, optimizer: str, batch_size: int) -> tuple[str, dict]:
    if dataset == "cifar10":
        model = "resnet18"
        epochs = 40
        scheduler_eta_min = 1e-6
        if optimizer == "sgd":
            lr = 1e-2
            weight_decay = 5e-4
        elif optimizer == "adamw":
            lr = 5e-4
            weight_decay = 5e-4
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
    elif dataset == "cifar100":
        model = "resnet34"
        epochs = 40
        if optimizer == "sgd":
            lr = 0.005
            weight_decay = 0.005
            scheduler_eta_min = 1e-4
        elif optimizer == "adamw":
            lr = 1e-4
            weight_decay = 0.005
            scheduler_eta_min = 1e-5
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    payload = {
        "dataset": dataset,
        "model": model,
        "optimizer": optimizer,
        "seed": 1,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "betas": [0.9, 0.999],
        "momentum": 0.9,
        "weight_decay": weight_decay,
        "nesterov": False,
        "scheduler": "cosine",
        "scheduler_T_max": epochs,
        "scheduler_eta_min": scheduler_eta_min,
        "num_workers": 2,
        "transform_mode": "all",
        "device": "auto",
        "output_dir": f"paper/Variance Is Stable?/runs/{dataset}_{model}_{optimizer}_bs{batch_size}/",
        "metric_microbatch_size": _metric_microbatch_size(batch_size),
        "eval_every": 1,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "non_blocking_transfers": True,
        "cudnn_benchmark": True,
        "profile_timing": True,
    }
    filename = f"{dataset}_{model}_{optimizer}_bs{batch_size}.yaml"
    return filename, payload


def _cabs_config(batch_size: int) -> tuple[str, dict]:
    payload = {
        "dataset": "cifar10",
        "model": "cabs_2conv_3dense",
        "optimizer": "sgd",
        "seed": 0,
        "batch_size": batch_size,
        "epochs": 10_000_000,
        "lr": 0.1,
        "betas": [0.9, 0.999],
        "momentum": 0.0,
        "weight_decay": 5e-4,
        "nesterov": False,
        "scheduler": "cosine",
        "scheduler_T_max": 10_000_000,
        "scheduler_eta_min": 0.001,
        "num_workers": 2,
        "transform_mode": "all",
        "device": "auto",
        "output_dir": f"paper/Variance Is Stable?/runs/cifar10_cabs_2conv_3dense_sgd_bs{batch_size}/",
        "metric_microbatch_size": _metric_microbatch_size(batch_size),
        "eval_every": 1,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "non_blocking_transfers": True,
        "cudnn_benchmark": True,
        "profile_timing": True,
    }
    filename = f"cifar10_cabs_2conv_3dense_sgd_bs{batch_size}.yaml"
    return filename, payload


def main() -> None:
    plan: list[tuple[str, dict]] = []
    for dataset in ("cifar10", "cifar100"):
        for optimizer in ("sgd", "adamw"):
            for batch_size in BATCH_SIZES:
                plan.append(_resnet_config(dataset, optimizer, batch_size))
    for batch_size in BATCH_SIZES:
        plan.append(_cabs_config(batch_size))

    for filename, payload in plan:
        _write_config(filename, payload)


if __name__ == "__main__":
    main()
