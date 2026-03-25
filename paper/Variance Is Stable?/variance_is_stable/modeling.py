from __future__ import annotations

import logging

import torch

from .experiment_config import ExperimentConfig


def resolve_device(requested: str, logger: logging.Logger | None = None) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        if logger is not None:
            logger.warning("CUDA requested but unavailable, falling back to CPU.")
        return torch.device("cpu")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        if logger is not None:
            logger.warning("MPS requested but unavailable, falling back to CPU.")
        return torch.device("cpu")
    return device


def build_model(config: ExperimentConfig) -> torch.nn.Module:
    import torchvision

    if config.model != "resnet18":
        raise ValueError(f"Unsupported model: {config.model}")
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    return model
