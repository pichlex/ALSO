from __future__ import annotations

import pytest

from variance_is_stable.experiment_config import ExperimentConfig


def test_config_accepts_new_models_and_optimizers() -> None:
    for model_name in ("resnet18", "resnet34", "cabs_2conv_3dense"):
        ExperimentConfig(model=model_name, optimizer="sgd").validate()

    ExperimentConfig(optimizer="adamw", betas=(0.8, 0.95)).validate()
    ExperimentConfig(dataset="cifar100", model="resnet34").validate()


def test_config_rejects_unknown_model_optimizer_and_invalid_betas() -> None:
    with pytest.raises(ValueError, match="model must be one of"):
        ExperimentConfig(model="resnet50").validate()

    with pytest.raises(ValueError, match="optimizer must be one of"):
        ExperimentConfig(optimizer="adam").validate()

    with pytest.raises(ValueError, match="betas must contain exactly two values"):
        ExperimentConfig(optimizer="adamw", betas=(0.9,)).validate()

    with pytest.raises(ValueError, match="only supported for dataset='cifar10'"):
        ExperimentConfig(dataset="cifar100", model="cabs_2conv_3dense").validate()
