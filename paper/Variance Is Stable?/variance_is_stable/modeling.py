from __future__ import annotations

import logging

import torch

from .experiment_config import ExperimentConfig


class CABS2Conv3Dense(torch.nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fc1 = torch.nn.Linear(2304, 384)
        self.fc2 = torch.nn.Linear(384, 192)
        self.fc3 = torch.nn.Linear(192, num_classes)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        torch.nn.init.trunc_normal_(self.conv1.weight, std=5e-2)
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.trunc_normal_(self.conv2.weight, std=5e-2)
        torch.nn.init.constant_(self.conv2.bias, 0.1)
        torch.nn.init.trunc_normal_(self.fc1.weight, std=0.04)
        torch.nn.init.constant_(self.fc1.bias, 0.1)
        torch.nn.init.trunc_normal_(self.fc2.weight, std=0.04)
        torch.nn.init.constant_(self.fc2.bias, 0.1)
        torch.nn.init.trunc_normal_(self.fc3.weight, std=1.0 / 192.0)
        torch.nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def _num_classes(dataset: str) -> int:
    if dataset == "cifar10":
        return 10
    if dataset == "cifar100":
        return 100
    raise ValueError(f"Unsupported dataset: {dataset}")


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
    if config.model == "cabs_2conv_3dense":
        return CABS2Conv3Dense(num_classes=_num_classes(config.dataset))
    import torchvision

    if config.model == "resnet18":
        model = torchvision.models.resnet18(weights=None)
    elif config.model == "resnet34":
        model = torchvision.models.resnet34(weights=None)
    else:
        raise ValueError(f"Unsupported model: {config.model}")

    model.conv1 = torch.nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(model.fc.in_features, _num_classes(config.dataset))
    return model
