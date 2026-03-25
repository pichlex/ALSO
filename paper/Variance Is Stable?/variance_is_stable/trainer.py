from __future__ import annotations

import math
import logging
import time
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


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    device: torch.device,
    epoch_index: int,
    total_epochs: int,
    non_blocking_transfers: bool,
    profile_timing: bool,
) -> tuple[float, NamedTensorDict, dict[str, float], float]:
    model.train()
    total_examples = 0
    total_steps = 0
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    first_step_shift_sq = float("nan")
    first_step_start_params = None

    grad_accumulator = zeros_like_named_tensors(
        OrderedDict((name, parameter.detach()) for name, parameter in model.named_parameters()),
        device=device,
    )
    transfer_time = 0.0
    forward_backward_time = 0.0
    optimizer_step_time = 0.0
    epoch_start_time = time.perf_counter()

    progress = tqdm(
        train_loader,
        desc=f"Train {epoch_index + 1}/{total_epochs}",
        leave=False,
    )
    for inputs, targets, _ in progress:
        if total_steps == 0:
            first_step_start_params = OrderedDict(
                (name, parameter.detach().clone())
                for name, parameter in model.named_parameters()
            )
        if profile_timing:
            transfer_start = time.perf_counter()
        inputs = inputs.to(device, non_blocking=non_blocking_transfers)
        targets = targets.to(device, non_blocking=non_blocking_transfers)
        if profile_timing:
            _synchronize_device(device)
            transfer_time += time.perf_counter() - transfer_start

        optimizer.zero_grad(set_to_none=True)
        if profile_timing:
            forward_backward_start = time.perf_counter()
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction="mean")
        loss.backward()
        if profile_timing:
            _synchronize_device(device)
            forward_backward_time += time.perf_counter() - forward_backward_start

        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                grad_accumulator[name].add_(parameter.grad.detach())

        if profile_timing:
            optimizer_step_start = time.perf_counter()
        optimizer.step()
        if profile_timing:
            _synchronize_device(device)
            optimizer_step_time += time.perf_counter() - optimizer_step_start
        if total_steps == 0 and first_step_start_params is not None:
            first_step_end_params = OrderedDict(
                (name, parameter.detach().clone())
                for name, parameter in model.named_parameters()
            )
            first_step_shift_sq = named_tensor_distance_sq(
                first_step_start_params,
                first_step_end_params,
            )

        batch_size = targets.numel()
        total_loss += loss.detach() * batch_size
        total_examples += batch_size
        total_steps += 1

    _synchronize_device(device)
    train_time = time.perf_counter() - epoch_start_time
    hat_grad = OrderedDict(
        (name, (tensor / float(total_steps)).detach().cpu())
        for name, tensor in grad_accumulator.items()
    )
    timing = {
        "train_time_sec": train_time,
        "data_to_device_time_sec": transfer_time if profile_timing else float("nan"),
        "forward_backward_time_sec": forward_backward_time if profile_timing else float("nan"),
        "optimizer_step_time_sec": optimizer_step_time if profile_timing else float("nan"),
        "train_steps": float(total_steps),
        "samples_per_sec": (
            float(total_examples) / train_time if train_time > 0 else float("nan")
        ),
    }
    return total_loss.div(max(1, total_examples)).item(), hat_grad, timing, first_step_shift_sq


def build_epoch_record(
    epoch: int,
    epoch_metrics: EpochStartMetrics,
    metric_3: float,
    train_loss: float,
    val_loss: float,
    val_acc: float,
    metrics_time_sec: float,
    train_timing: dict[str, float],
    eval_time_sec: float,
    cuda_memory_stats: dict[str, float],
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
        "metrics_time_sec": metrics_time_sec,
        "train_time_sec": train_timing["train_time_sec"],
        "eval_time_sec": eval_time_sec,
        "data_to_device_time_sec": train_timing["data_to_device_time_sec"],
        "forward_backward_time_sec": train_timing["forward_backward_time_sec"],
        "optimizer_step_time_sec": train_timing["optimizer_step_time_sec"],
        "train_steps": train_timing["train_steps"],
        "samples_per_sec": train_timing["samples_per_sec"],
        "cuda_mem_alloc_mb": cuda_memory_stats["cuda_mem_alloc_mb"],
        "cuda_mem_peak_mb": cuda_memory_stats["cuda_mem_peak_mb"],
    }


def log_epoch_summary(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    epoch_record: dict[str, Any],
) -> None:
    metric_4 = epoch_record["metric_4"]
    metric_5 = epoch_record["metric_5"]
    logger.info(
        (
            "Epoch %s/%s | m1=%.6f | m2=%.6f | m3=%.6f | m4=%s | m5=%s | "
            "train_loss=%.6f | val_loss=%.6f | val_acc=%.4f | "
            "metrics=%.2fs | train=%.2fs | eval=%.2fs | samples/s=%.2f | "
            "cuda_mem=%.1fMB | cuda_peak=%.1fMB"
        ),
        epoch + 1,
        total_epochs,
        epoch_record["metric_1"],
        epoch_record["metric_2"],
        epoch_record["metric_3"],
        f"{metric_4:.6f}" if not math.isnan(metric_4) else "nan",
        f"{metric_5:.6f}" if not math.isnan(metric_5) else "nan",
        epoch_record["train_loss"],
        epoch_record["val_loss"],
        epoch_record["val_acc"],
        epoch_record["metrics_time_sec"],
        epoch_record["train_time_sec"],
        epoch_record["eval_time_sec"],
        epoch_record["samples_per_sec"],
        epoch_record["cuda_mem_alloc_mb"],
        epoch_record["cuda_mem_peak_mb"],
    )


def parameter_shift_sq(
    previous_epoch_start: NamedTensorDict | None,
    current_epoch_start: NamedTensorDict,
) -> float:
    if previous_epoch_start is None:
        return float("nan")
    return named_tensor_distance_sq(previous_epoch_start, current_epoch_start)
