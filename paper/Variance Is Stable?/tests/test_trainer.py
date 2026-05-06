from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from variance_is_stable.experiment_config import ExperimentConfig
from variance_is_stable.trainer import build_optimizer, train_one_epoch


class IndexedTensorDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index], index


def _manual_epoch_hat_grad(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> OrderedDict[str, torch.Tensor]:
    model.train()
    grad_sums = OrderedDict(
        (name, torch.zeros_like(parameter))
        for name, parameter in model.named_parameters()
    )
    steps = 0

    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        batch_inputs = inputs[start:stop]
        batch_targets = targets[start:stop]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch_inputs), batch_targets, reduction="mean")
        loss.backward()
        for name, parameter in model.named_parameters():
            grad_sums[name].add_(parameter.grad.detach())
        optimizer.step()
        steps += 1

    return OrderedDict((name, tensor / float(steps)) for name, tensor in grad_sums.items())


def test_train_one_epoch_keeps_hat_grad_semantics() -> None:
    torch.manual_seed(7)
    inputs = torch.randn(6, 4)
    targets = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    dataset = IndexedTensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model_for_train = torch.nn.Sequential(
        torch.nn.Linear(4, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )
    model_for_manual = torch.nn.Sequential(
        torch.nn.Linear(4, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )
    model_for_manual.load_state_dict(model_for_train.state_dict())

    config = ExperimentConfig(batch_size=2, epochs=1, scheduler="none", transform_mode="none")
    optimizer = build_optimizer(model_for_train, config)
    manual_optimizer = build_optimizer(model_for_manual, config)

    train_loss, hat_grad, timing, metric_3 = train_one_epoch(
        model=model_for_train,
        optimizer=optimizer,
        train_loader=loader,
        device=torch.device("cpu"),
        epoch_index=0,
        total_epochs=1,
        non_blocking_transfers=False,
        profile_timing=False,
    )
    expected_hat_grad = _manual_epoch_hat_grad(
        model=model_for_manual,
        optimizer=manual_optimizer,
        inputs=inputs,
        targets=targets,
        batch_size=2,
    )

    for name in expected_hat_grad:
        assert torch.allclose(hat_grad[name], expected_hat_grad[name], atol=1e-6)
        assert hat_grad[name].device.type == "cpu"
    assert train_loss > 0
    assert timing["train_time_sec"] > 0
    assert metric_3 >= 0


def test_train_one_epoch_returns_first_step_parameter_shift() -> None:
    torch.manual_seed(11)
    inputs = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    dataset = IndexedTensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = torch.nn.Linear(3, 2)
    manual_model = torch.nn.Linear(3, 2)
    manual_model.load_state_dict(model.state_dict())

    config = ExperimentConfig(
        batch_size=2,
        epochs=1,
        scheduler="none",
        transform_mode="none",
        lr=0.05,
        momentum=0.9,
        weight_decay=0.01,
    )
    optimizer = build_optimizer(model, config)
    manual_optimizer = build_optimizer(manual_model, config)

    _, _, _, metric_3 = train_one_epoch(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        device=torch.device("cpu"),
        epoch_index=0,
        total_epochs=1,
        non_blocking_transfers=False,
        profile_timing=False,
    )

    start_params = OrderedDict(
        (name, parameter.detach().clone())
        for name, parameter in manual_model.named_parameters()
    )
    batch_inputs, batch_targets, _ = next(iter(loader))
    manual_optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(manual_model(batch_inputs), batch_targets, reduction="mean")
    loss.backward()
    manual_optimizer.step()
    end_params = OrderedDict(
        (name, parameter.detach().clone())
        for name, parameter in manual_model.named_parameters()
    )
    expected_metric_3 = sum(
        ((end_params[name] - start_params[name]) ** 2).sum().item()
        for name in start_params
    )

    assert metric_3 == pytest.approx(expected_metric_3)


def test_build_optimizer_supports_adamw() -> None:
    model = torch.nn.Linear(3, 2)
    config = ExperimentConfig(
        optimizer="adamw",
        scheduler="none",
        transform_mode="none",
        betas=(0.8, 0.95),
    )

    optimizer = build_optimizer(model, config)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["betas"] == (0.8, 0.95)


def test_train_one_epoch_keeps_hat_grad_semantics_with_adamw() -> None:
    torch.manual_seed(13)
    inputs = torch.randn(6, 4)
    targets = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    dataset = IndexedTensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model_for_train = torch.nn.Sequential(
        torch.nn.Linear(4, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )
    model_for_manual = torch.nn.Sequential(
        torch.nn.Linear(4, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )
    model_for_manual.load_state_dict(model_for_train.state_dict())

    config = ExperimentConfig(
        batch_size=2,
        epochs=1,
        scheduler="none",
        transform_mode="none",
        optimizer="adamw",
        lr=1e-3,
    )
    optimizer = build_optimizer(model_for_train, config)
    manual_optimizer = build_optimizer(model_for_manual, config)

    _, hat_grad, _, metric_3 = train_one_epoch(
        model=model_for_train,
        optimizer=optimizer,
        train_loader=loader,
        device=torch.device("cpu"),
        epoch_index=0,
        total_epochs=1,
        non_blocking_transfers=False,
        profile_timing=False,
    )
    expected_hat_grad = _manual_epoch_hat_grad(
        model=model_for_manual,
        optimizer=manual_optimizer,
        inputs=inputs,
        targets=targets,
        batch_size=2,
    )

    for name in expected_hat_grad:
        assert torch.allclose(hat_grad[name], expected_hat_grad[name], atol=1e-6)
    assert metric_3 >= 0
