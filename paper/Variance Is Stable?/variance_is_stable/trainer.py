from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .experiment_config import ExperimentConfig
from .metrics import (
    EpochStartMetrics,
    NamedTensorDict,
    clone_named_parameters,
    named_tensor_distance_sq,
    zeros_like_named_tensors,
)


def build_optimizer(
    model: torch.nn.Module,
    config: ExperimentConfig,
) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.scheduler == "none":
        return None
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler_T_max,
            eta_min=config.scheduler_eta_min,
        )
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def evaluate(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> tuple[float, float]:
    previous_mode = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == targets).sum().item()
            total_examples += targets.numel()

    if previous_mode:
        model.train()
    return total_loss / max(1, total_examples), total_correct / max(1, total_examples)


def _collect_batch_mean_grads(model: torch.nn.Module) -> NamedTensorDict:
    return OrderedDict(
        (name, parameter.grad.detach().clone())
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    )


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    device: torch.device,
    epoch_index: int,
    total_epochs: int,
) -> tuple[float, NamedTensorDict]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_steps = 0

    grad_accumulator = zeros_like_named_tensors(
        OrderedDict((name, parameter.detach()) for name, parameter in model.named_parameters()),
        device=torch.device("cpu"),
    )

    progress = tqdm(
        train_loader,
        desc=f"Train {epoch_index + 1}/{total_epochs}",
        leave=False,
    )
    for inputs, targets, _ in progress:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction="mean")
        loss.backward()

        batch_mean_grads = _collect_batch_mean_grads(model)
        for name, grad_tensor in batch_mean_grads.items():
            grad_accumulator[name].add_(grad_tensor.detach().cpu())

        optimizer.step()

        batch_size = targets.numel()
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        total_steps += 1

    hat_grad = OrderedDict(
        (name, tensor / float(total_steps)) for name, tensor in grad_accumulator.items()
    )
    return total_loss / max(1, total_examples), hat_grad


def build_epoch_record(
    epoch: int,
    epoch_metrics: EpochStartMetrics,
    metric_3: float,
    train_loss: float,
    val_loss: float,
    val_acc: float,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "metric_1": epoch_metrics.metric_1,
        "metric_2": epoch_metrics.metric_2,
        "metric_3": metric_3,
        "metric_4": epoch_metrics.metric_4,
        "metric_5": epoch_metrics.metric_5,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }


def log_epoch_summary(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    epoch_record: dict[str, Any],
) -> None:
    logger.info(
        (
            "Epoch %s/%s | m1=%.6f | m2=%.6f | m3=%.6f | m4=%s | m5=%s | "
            "train_loss=%.6f | val_loss=%.6f | val_acc=%.4f"
        ),
        epoch + 1,
        total_epochs,
        epoch_record["metric_1"],
        epoch_record["metric_2"],
        epoch_record["metric_3"],
        f"{epoch_record['metric_4']:.6f}" if not torch.isnan(torch.tensor(epoch_record["metric_4"])) else "nan",
        f"{epoch_record['metric_5']:.6f}" if not torch.isnan(torch.tensor(epoch_record["metric_5"])) else "nan",
        epoch_record["train_loss"],
        epoch_record["val_loss"],
        epoch_record["val_acc"],
    )


def parameter_shift_sq(
    previous_epoch_start: NamedTensorDict | None,
    current_epoch_start: NamedTensorDict,
) -> float:
    if previous_epoch_start is None:
        return float("nan")
    return named_tensor_distance_sq(previous_epoch_start, current_epoch_start)
