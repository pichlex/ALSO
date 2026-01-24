from typing import Dict, Any, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, _LRScheduler
from adabatchgrad import AdaBatchGrad


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
    if name == "adam":
        betas = config.get("betas", (0.9, 0.999))
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )
    if name == "adamw":
        betas = config.get("betas", (0.9, 0.999))
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )
    if name == "adabatchgrad":
        alpha = float(config.get("adabatchgrad_alpha", 1.0))
        beta = float(config.get("adabatchgrad_beta", 1.0))
        power_eps = float(config.get("adabatchgrad_power_eps", 0.0))
        return AdaBatchGrad(
            model.parameters(),
            alpha=alpha,
            beta=beta,
            power_eps=power_eps,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: Optimizer, config: Dict[str, Any]) -> Optional[_LRScheduler]:
    """Create LR scheduler if configured; returns None when not requested."""
    name = config.get("scheduler")
    if name is None:
        return None

    # AdaBatchGrad does not expose per-group lr expected by schedulers; skip.
    if str(config.get("optimizer", "")).lower() == "adabatchgrad":
        return None

    name = str(name).lower()
    if name == "multistep":
        milestones = config.get("scheduler_milestones", [30, 60])
        gamma = float(config.get("scheduler_gamma", 0.1))
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    if name == "cosine":
        # Default T_max to total epochs if not provided.
        T_max = int(config.get("scheduler_T_max", config.get("epochs", 20)))
        eta_min = float(config.get("scheduler_eta_min", 0.0))
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    raise ValueError(f"Unsupported scheduler: {name}")
