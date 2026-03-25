from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from variance_is_stable.metrics import (
    clone_named_parameters,
    compute_epoch_start_metrics,
)
from variance_is_stable.trainer import parameter_shift_sq


class IndexedTensorDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index], index


def _flatten_named_grads(named_grads: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in named_grads.values()])


def _manual_per_sample_grads(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> list[torch.Tensor]:
    model.eval()
    grads = []
    for sample, target in zip(inputs, targets):
        model.zero_grad(set_to_none=True)
        logits = model(sample.unsqueeze(0))
        loss = F.cross_entropy(logits, target.unsqueeze(0), reduction="mean")
        loss.backward()
        grads.append(
            torch.cat([parameter.grad.detach().reshape(-1) for parameter in model.parameters()])
        )
    return grads


def test_epoch_start_metrics_match_bruteforce() -> None:
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.5, -0.25], [-0.1, 0.3]]))
        model.bias.copy_(torch.tensor([0.2, -0.4]))

    inputs = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, -0.5],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    dataset = IndexedTensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    reference_grad = OrderedDict(
        (name, torch.full_like(parameter, 0.1))
        for name, parameter in model.named_parameters()
    )

    metrics = compute_epoch_start_metrics(
        model=model,
        train_loader=loader,
        device=torch.device("cpu"),
        metric_microbatch_size=1,
        reference_grad=reference_grad,
    )

    per_sample_grads = _manual_per_sample_grads(model, inputs, targets)
    grad_matrix = torch.stack(per_sample_grads)
    grad_mean = grad_matrix.mean(dim=0)
    first_batch_mean = grad_matrix[:2].mean(dim=0)
    flat_reference = _flatten_named_grads(reference_grad)

    expected_metric_1 = ((grad_matrix - grad_mean).pow(2).sum(dim=1)).mean().item()
    expected_metric_2 = torch.sum((first_batch_mean - grad_mean) ** 2).item()
    expected_metric_4 = ((grad_matrix - flat_reference) ** 2).sum(dim=1).mean().item()
    expected_metric_5 = torch.sum((first_batch_mean - flat_reference) ** 2).item()

    assert metrics.metric_1 == pytest.approx(expected_metric_1)
    assert metrics.metric_2 == pytest.approx(expected_metric_2)
    assert metrics.metric_4 == pytest.approx(expected_metric_4)
    assert metrics.metric_5 == pytest.approx(expected_metric_5)


def test_parameter_shift_sq_uses_epoch_start_snapshots() -> None:
    model = torch.nn.Linear(2, 2)
    previous = clone_named_parameters(model)
    with torch.no_grad():
        model.weight.add_(1.0)
    current = clone_named_parameters(model)

    expected = sum(((current[name] - previous[name]) ** 2).sum().item() for name in current)
    assert parameter_shift_sq(previous, current) == pytest.approx(expected)
    assert torch.isnan(torch.tensor(parameter_shift_sq(None, current)))
