from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from tqdm.auto import tqdm


NamedTensorDict = OrderedDict[str, torch.Tensor]


def clone_named_parameters(model: torch.nn.Module) -> NamedTensorDict:
    return OrderedDict(
        (name, parameter.detach().clone()) for name, parameter in model.named_parameters()
    )


def clone_named_parameters_for_grad(model: torch.nn.Module) -> NamedTensorDict:
    return OrderedDict(
        (name, parameter.detach().clone().requires_grad_(True))
        for name, parameter in model.named_parameters()
    )


def clone_named_buffers(model: torch.nn.Module) -> NamedTensorDict:
    return OrderedDict(
        (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
    )


def zeros_like_named_tensors(tensors: NamedTensorDict, device: torch.device) -> NamedTensorDict:
    return OrderedDict(
        (name, torch.zeros_like(tensor, device=device)) for name, tensor in tensors.items()
    )


def named_tensor_norm_sq(tensors: NamedTensorDict) -> float:
    return float(
        sum(torch.sum(tensor * tensor).item() for tensor in tensors.values())
    )


def named_tensor_distance_sq(lhs: NamedTensorDict, rhs: NamedTensorDict) -> float:
    total = 0.0
    for name in lhs:
        diff = lhs[name] - rhs[name]
        total += torch.sum(diff * diff).item()
    return float(total)


def divide_named_tensors(tensors: NamedTensorDict, divisor: float) -> NamedTensorDict:
    return OrderedDict((name, tensor / divisor) for name, tensor in tensors.items())


def move_named_tensors(tensors: NamedTensorDict, device: torch.device) -> NamedTensorDict:
    return OrderedDict((name, tensor.to(device)) for name, tensor in tensors.items())


def _single_sample_loss(
    model: torch.nn.Module,
    params: NamedTensorDict,
    buffers: NamedTensorDict,
    sample: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    logits = functional_call(model, (params, buffers), (sample.unsqueeze(0),))
    return F.cross_entropy(logits, target.unsqueeze(0), reduction="mean")


@dataclass(slots=True)
class EpochStartMetrics:
    metric_1: float
    metric_2: float
    metric_4: float
    metric_5: float


def compute_epoch_start_metrics(
    model: torch.nn.Module,
    train_loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    metric_microbatch_size: int,
    reference_grad: NamedTensorDict | None = None,
    progress_desc: str | None = None,
) -> EpochStartMetrics:
    previous_mode = model.training
    model.eval()

    params = clone_named_parameters_for_grad(model)
    buffers = clone_named_buffers(model)
    param_sums = zeros_like_named_tensors(params, device=device)
    first_batch_sum = zeros_like_named_tensors(params, device=device)
    reference_on_device = (
        move_named_tensors(reference_grad, device=device) if reference_grad is not None else None
    )

    params = move_named_tensors(params, device=device)
    buffers = move_named_tensors(buffers, device=device)

    batched_grad_fn = vmap(
        grad(lambda current_params, current_buffers, sample, target: _single_sample_loss(
            model, current_params, current_buffers, sample, target
        )),
        in_dims=(None, None, 0, 0),
    )

    dataset_size = 0
    grad_norm_sq_sum = 0.0
    reference_var_sum = 0.0
    first_batch_size = 0

    iterator = train_loader
    if progress_desc is not None:
        iterator = tqdm(train_loader, desc=progress_desc, leave=False)

    for loader_batch_index, batch in enumerate(iterator):
        inputs, targets, _ = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.shape[0]
        if loader_batch_index == 0:
            first_batch_size = batch_size

        for start in range(0, batch_size, metric_microbatch_size):
            stop = min(start + metric_microbatch_size, batch_size)
            micro_inputs = inputs[start:stop]
            micro_targets = targets[start:stop]
            per_sample_grads = batched_grad_fn(params, buffers, micro_inputs, micro_targets)

            for name, grad_batch in per_sample_grads.items():
                grad_batch = grad_batch.detach()
                param_sums[name].add_(grad_batch.sum(dim=0))
                grad_norm_sq_sum += grad_batch.flatten(start_dim=1).square().sum().item()
                if loader_batch_index == 0:
                    first_batch_sum[name].add_(grad_batch.sum(dim=0))
                if reference_on_device is not None:
                    diff = grad_batch - reference_on_device[name].unsqueeze(0)
                    reference_var_sum += diff.flatten(start_dim=1).square().sum().item()
            dataset_size += stop - start

    mean_grads = divide_named_tensors(param_sums, float(dataset_size))
    mean_norm_sq = named_tensor_norm_sq(mean_grads)
    metric_1 = max(0.0, grad_norm_sq_sum / float(dataset_size) - mean_norm_sq)

    metric_2 = 0.0
    for name, batch_sum in first_batch_sum.items():
        batch_mean = batch_sum / float(first_batch_size)
        diff = batch_mean - mean_grads[name]
        metric_2 += torch.sum(diff * diff).item()
    metric_2 = float(metric_2)

    metric_4 = float("nan")
    metric_5 = float("nan")
    if reference_on_device is not None:
        metric_4 = reference_var_sum / float(dataset_size)
        metric_5 = 0.0
        for name, batch_sum in first_batch_sum.items():
            batch_mean = batch_sum / float(first_batch_size)
            diff = batch_mean - reference_on_device[name]
            metric_5 += torch.sum(diff * diff).item()
        metric_5 = float(metric_5)

    if previous_mode:
        model.train()
    return EpochStartMetrics(
        metric_1=float(metric_1),
        metric_2=float(metric_2),
        metric_4=float(metric_4),
        metric_5=float(metric_5),
    )
