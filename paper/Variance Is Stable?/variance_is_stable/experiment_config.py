from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ExperimentConfig:
    dataset: str = "cifar10"
    model: str = "resnet18"
    optimizer: str = "sgd"
    seed: int = 42
    batch_size: int = 64
    epochs: int = 40
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 5e-3
    nesterov: bool = False
    scheduler: str = "cosine"
    scheduler_T_max: int = 40
    scheduler_eta_min: float = 1e-4
    num_workers: int = 2
    transform_mode: str = "all"
    device: str = "auto"
    output_dir: str = "paper/Variance Is Stable?/runs"
    run_name: str | None = None
    metric_microbatch_size: int = 16
    eval_every: int = 1
    persistent_workers: bool = True
    prefetch_factor: int = 2
    non_blocking_transfers: bool = True
    cudnn_benchmark: bool = True
    profile_timing: bool = True

    def validate(self) -> "ExperimentConfig":
        if self.dataset != "cifar10":
            raise ValueError("Only dataset='cifar10' is supported.")
        if self.model != "resnet18":
            raise ValueError("Only model='resnet18' is supported.")
        if self.optimizer != "sgd":
            raise ValueError("Only optimizer='sgd' is supported.")
        if self.transform_mode not in {"none", "normalize", "all"}:
            raise ValueError("transform_mode must be one of: none, normalize, all.")
        if self.scheduler not in {"none", "cosine"}:
            raise ValueError("scheduler must be one of: none, cosine.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.metric_microbatch_size <= 0:
            raise ValueError("metric_microbatch_size must be positive.")
        if self.eval_every <= 0:
            raise ValueError("eval_every must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            return json.load(handle)
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(handle)
            return {} if data is None else data
    raise ValueError(f"Unsupported config format: {path}")


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = _read_config(path)
    config = ExperimentConfig(**raw)
    if config.scheduler_T_max <= 0:
        config.scheduler_T_max = config.epochs
    return config.validate()
