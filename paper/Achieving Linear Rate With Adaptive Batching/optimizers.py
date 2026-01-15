from typing import Dict, Any, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> Optimizer:
    name = config.get("optimizer", "sgd").lower()
    lr = float(config.get("lr", 0.1))
    weight_decay = float(config.get("weight_decay", 0.0))

    if name == "sgd":
        momentum = float(config.get("momentum", 0.9))
        nesterov = bool(config.get("nesterov", False))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )
    if name == "adamw":
        betas = config.get("betas", (0.9, 0.999))
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: Optimizer, config: Dict[str, Any]) -> Optional[_LRScheduler]:
    """Create LR scheduler if configured; returns None when not requested."""
    name = config.get("scheduler")
    if name is None:
        return None

    name = str(name).lower()
    if name == "multistep":
        milestones = config.get("scheduler_milestones", [30, 60])
        gamma = float(config.get("scheduler_gamma", 0.1))
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    raise ValueError(f"Unsupported scheduler: {name}")
