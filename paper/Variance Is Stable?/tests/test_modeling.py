from __future__ import annotations

import sys
import types

import torch
import pytest

from variance_is_stable.experiment_config import ExperimentConfig
from variance_is_stable.modeling import CABS2Conv3Dense, build_model


class _FakeResNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fc = torch.nn.Linear(64, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.mean(self.conv1(x), dim=(-1, -2)))


@pytest.fixture
def fake_torchvision(monkeypatch: pytest.MonkeyPatch):
    models = types.SimpleNamespace(
        resnet18=lambda weights=None: _FakeResNet(),
        resnet34=lambda weights=None: _FakeResNet(),
    )
    module = types.SimpleNamespace(models=models)
    monkeypatch.setitem(sys.modules, "torchvision", module)
    return module


def test_resnet_models_use_small_image_stem(fake_torchvision) -> None:
    del fake_torchvision
    for model_name in ("resnet18", "resnet34"):
        model = build_model(
            ExperimentConfig(model=model_name, optimizer="sgd", scheduler="none")
        )

        assert model.conv1.kernel_size == (3, 3)
        assert model.conv1.stride == (1, 1)
        assert model.conv1.padding == (1, 1)
        assert isinstance(model.maxpool, torch.nn.Identity)
        assert model.fc.out_features == 10

    cifar100_model = build_model(
        ExperimentConfig(dataset="cifar100", model="resnet34", optimizer="sgd", scheduler="none")
    )
    assert cifar100_model.fc.out_features == 100


def test_cabs_model_returns_cifar10_logits(fake_torchvision) -> None:
    del fake_torchvision
    model = build_model(
        ExperimentConfig(model="cabs_2conv_3dense", optimizer="sgd", scheduler="none")
    )

    assert isinstance(model, CABS2Conv3Dense)
    outputs = model(torch.randn(3, 3, 24, 24))
    assert outputs.shape == (3, 10)
