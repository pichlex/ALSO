import math
from typing import Dict, Any, List, Optional, Tuple

import mlflow
import numpy as np
import torch
import torchvision
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, BatchSampler

from adabatchgrad import (
    DynamicBatchSampler,
    calculate_batch_size,
    per_sample_cross_entropy_grads,
)
from adaptive_batch import AdaptiveBatchTracker
from data import get_device, set_seed, IndexedDataset
from optimizers import build_optimizer, build_scheduler


class FixedOrderSampler(torch.utils.data.Sampler[int]):
    """Yields a predefined list of indices in order."""

    def __init__(self, indices: List[int]):
        super().__init__(None)
        self.indices = [int(i) for i in indices]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _get_param_list(optimizer: torch.optim.Optimizer):
    return [p for group in optimizer.param_groups for p in group["params"]]


def _evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: str,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_true = []
    total_pred = []
    with torch.no_grad():
        for Xy, _ in dataloader:
            X, y = Xy
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            losses = loss_fn(logits, y)
            loss = losses.mean()
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            total_true.append(y.detach().cpu().numpy())
            total_pred.append(preds.detach().cpu().numpy())
    total_true_np = np.concatenate(total_true)
    total_pred_np = np.concatenate(total_pred)
    average = "weighted" if len(np.unique(total_true_np)) > 2 else "binary"
    metrics = {
        "f1": f1_score(total_true_np, total_pred_np, average=average),
        "precision": precision_score(
            total_true_np, total_pred_np, average=average, zero_division=0.0
        ),
        "recall": recall_score(total_true_np, total_pred_np, average=average),
        "accuracy": float(np.mean(total_true_np == total_pred_np)),
    }
    return total_loss / max(1, len(dataloader)), metrics


def _make_adaptive_loader(
    train_dataset: IndexedDataset,
    batch_size: int,
    fixed_indices: List[int],
    num_workers: int,
    seed_worker,
    generator: torch.Generator,
) -> DataLoader:
    sampler = FixedOrderSampler(fixed_indices)
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)
    return DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
    )


def _make_shuffled_loader(
    train_dataset: IndexedDataset,
    batch_size: int,
    num_workers: int,
    seed_worker,
    generator: torch.Generator,
) -> DataLoader:
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
    )


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _simulate_cosine_lrs(
    base_lr: float, eta_min: float, T_max: int, epochs: int
) -> List[float]:
    # Use the same CosineAnnealingLR logic PyTorch provides to find when LR crosses thresholds.
    # T_max defaults to epochs when not provided; clamp to >=1 to avoid div-by-zero.
    T_max = max(1, T_max)
    dummy_param = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.SGD([dummy_param], lr=base_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)
    lrs: List[float] = []
    for _ in range(epochs):
        lrs.append(opt.param_groups[0]["lr"])
        scheduler.step()
    return lrs


def _compute_seesaw_cut_epochs(alpha: float, lr0: float, lr_schedule: List[float]) -> List[int]:
    if alpha <= 1.0:
        return []
    cuts: List[int] = []
    threshold = lr0 / alpha
    for idx, lr in enumerate(lr_schedule):
        if lr <= threshold:
            cuts.append(idx)
            threshold /= alpha
    return cuts


def _clone_model_params(model: torch.nn.Module) -> List[torch.Tensor]:
    # Snapshot parameters to CPU to measure cross-epoch movement without holding extra GPU memory.
    return [p.detach().clone().cpu() for p in model.parameters()]


def train_model(
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> Dict[str, Any]:
    preferred_device = config.get("device")
    auto_device = get_device()
    device = preferred_device or auto_device
    if preferred_device and preferred_device.startswith("cuda") and not torch.cuda.is_available():
        # Fall back gracefully if CUDA requested but unavailable.
        device = "cpu"
    model_name = config.get("model", "resnet18").lower()
    if model_name == "resnet18":
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    adaptive_enabled = bool(config.get("adaptive_batch", False))
    batch_size_min = int(config.get("adaptive_batch_min", 10))
    batch_size_max = int(config.get("adaptive_batch_max", 65536))
    batch_size_init = int(config.get("batch_size", 64))
    adaptive_beta = float(config.get("adaptive_batch_beta", 0.0))
    batch_size_multiplier = float(config.get("batch_size_multiplier", 1.0))
    epoch_start_ab = int(config.get("epoch_start_ab", 2))
    adaptive_strategy = str(config.get("adaptive_batch_strategy", "variance_ratio")).lower()
    adabatchgrad_batch_test = config.get("adabatchgrad_batch_test", "random_increase")
    adabatchgrad_theta = float(config.get("adabatchgrad_theta", 0.1))
    adabatchgrad_nu = float(config.get("adabatchgrad_nu", 0.1))
    adabatchgrad_prob_new = float(config.get("adabatchgrad_prob_new", 0.005))
    adabatchgrad_k = int(config.get("adabatchgrad_k", 5))
    epochs = int(config.get("epochs", 20))

    log_to_mlflow = config.get("report_to") == "mlflow"
    if log_to_mlflow:
        experiment_name = config.get("mlflow_experiment", "Achieving Linear Rate")
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=config.get("run_name"))
        params_to_log = {
            k: v
            for k, v in config.items()
            if isinstance(v, (int, float, str, bool))
        }
        mlflow.log_params(params_to_log)

    g, seed_worker = set_seed(int(config.get("seed", 42)))
    train_dataset: IndexedDataset = train_loader.dataset
    use_adabatchgrad_strategy = adaptive_enabled and adaptive_strategy == "adabatchgrad"
    use_seesaw_strategy = adaptive_enabled and adaptive_strategy == "seesaw"
    use_variance_strategy = adaptive_enabled and adaptive_strategy not in ("adabatchgrad", "seesaw")
    if use_seesaw_strategy:
        scheduler = None

    adaptive_sampler: Optional[DynamicBatchSampler] = None
    if use_adabatchgrad_strategy:
        adaptive_sampler = DynamicBatchSampler(
            train_dataset,
            batch_size_init,
            generator=g,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=adaptive_sampler,
            num_workers=int(config.get("num_workers", 2)),
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
        )

    fixed_indices = (
        torch.randperm(len(train_dataset), generator=g).tolist() if use_variance_strategy else []
    )

    optimizer_name = str(config.get("optimizer", "sgd")).lower()
    seesaw_alpha = float(config.get("seesaw_alpha", 2.0))
    seesaw_cut_epochs: List[int] = []
    current_lr = float(config.get("lr", 0.1))
    if use_seesaw_strategy:
        eta_min = float(config.get("scheduler_eta_min", 0.0))
        T_max = int(config.get("scheduler_T_max", config.get("epochs", 20)))
        lr_schedule = _simulate_cosine_lrs(current_lr, eta_min, T_max, epochs)
        seesaw_cut_epochs = _compute_seesaw_cut_epochs(seesaw_alpha, current_lr, lr_schedule)

    hat_norm_history: List[float] = []
    var_history: List[float] = []
    prev_hat_grad: Optional[List[Optional[torch.Tensor]]] = None
    prev_batch_size: Optional[float] = batch_size_init
    prev_ratio_ema: Optional[float] = None
    prev_params: Optional[List[torch.Tensor]] = None
    prev_prev_params: Optional[List[torch.Tensor]] = None

    best_val_f1 = -1.0
    best_val_acc = -1.0
    best_test = {}
    best_test_acc = {}
    train_step = 0

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0].get("lr")
        # For AdaBatchGrad, log the current step size instead of lr (lr is undefined).
        if log_to_mlflow:
            if current_lr is not None:
                mlflow.log_metric("lr", current_lr, step=epoch)
            else:
                # Use the shared step size from the optimizer state if present.
                first_param = next(iter(model.parameters()))
                step_size = optimizer.state.get(first_param, {}).get("step_size")
                if step_size is not None:
                    mlflow.log_metric("lr", float(step_size), step=epoch)

        if use_adabatchgrad_strategy:
            if adaptive_sampler is not None:
                adaptive_sampler.reset_epoch_state()
            current_loader = train_loader
        elif use_seesaw_strategy:
            batch_size_epoch = int(prev_batch_size) if prev_batch_size is not None else batch_size_init
            if epoch in seesaw_cut_epochs:
                lr_divisor = seesaw_alpha if optimizer_name == "sgd" else math.sqrt(seesaw_alpha)
                if lr_divisor > 0:
                    current_lr = current_lr / lr_divisor if current_lr is not None else None
                if current_lr is not None:
                    _set_optimizer_lr(optimizer, current_lr)
                batch_size_epoch = int(math.ceil(batch_size_epoch * seesaw_alpha))
            batch_size_epoch = max(batch_size_min, min(batch_size_max, batch_size_epoch))
            batch_size_epoch = max(1, min(batch_size_epoch, len(train_dataset)))
            prev_batch_size = batch_size_epoch
            current_loader = _make_shuffled_loader(
                train_dataset,
                batch_size_epoch,
                int(config.get("num_workers", 2)),
                seed_worker,
                g,
            )
            if log_to_mlflow:
                mlflow.log_metric("seesaw/batch_size", batch_size_epoch, step=epoch)
                if current_lr is not None:
                    mlflow.log_metric("seesaw/lr", current_lr, step=epoch)
        elif use_variance_strategy and epoch >= epoch_start_ab:
            var_used = var_history[epoch - 1] if (epoch - 1) < len(var_history) else None
            theta_diff_norm_sq = None
            if prev_params is not None and prev_prev_params is not None:
                theta_diff_norm_sq = 0.0
                for p1, p0 in zip(prev_params, prev_prev_params):
                    diff = p1 - p0
                    theta_diff_norm_sq += torch.sum(diff * diff).item()
            ratio_raw = None
            ratio_smoothed = None
            if var_used is not None and theta_diff_norm_sq is not None and theta_diff_norm_sq > 0:
                ratio_raw = math.sqrt(var_used / theta_diff_norm_sq)
                ratio_for_batch = ratio_raw
                if adaptive_beta > 0 and prev_batch_size is not None:
                    ratio_for_batch = adaptive_beta * prev_batch_size + (1 - adaptive_beta) * ratio_raw
                    ratio_smoothed = ratio_for_batch
                ratio_for_batch *= batch_size_multiplier
                batch_size_epoch = int(math.floor(ratio_for_batch))
            else:
                batch_size_epoch = batch_size_init
            batch_size_epoch = max(batch_size_min, min(batch_size_max, batch_size_epoch))
            batch_size_epoch = max(1, min(batch_size_epoch, len(train_dataset)))
            prev_batch_size = batch_size_epoch
            current_loader = _make_adaptive_loader(
                train_dataset,
                batch_size_epoch,
                fixed_indices,
                int(config.get("num_workers", 2)),
                seed_worker,
                g,
            )
            if log_to_mlflow:
                mlflow.log_metric("adaptive_batch/batch_size", batch_size_epoch, step=epoch)
                if var_used is not None and theta_diff_norm_sq is not None:
                    mlflow.log_metric("adaptive_batch/var_sum", var_used, step=epoch)
                    mlflow.log_metric(
                        "adaptive_batch/theta_diff_norm_sq", theta_diff_norm_sq, step=epoch
                    )
                if ratio_raw is not None:
                        mlflow.log_metric("adaptive_batch/ratio_raw", ratio_raw, step=epoch)
                if ratio_smoothed is not None:
                    mlflow.log_metric("adaptive_batch/ratio_ema", ratio_smoothed, step=epoch)
        else:
            current_loader = train_loader
            prev_batch_size = batch_size_init

        tracker = AdaptiveBatchTracker(device) if use_variance_strategy else None
        if tracker is not None:
            tracker.reset(prev_hat_grad)

        model.train()
        total_loss = 0.0
        steps = 0
        param_list = _get_param_list(optimizer)

        if use_adabatchgrad_strategy:
            pbar = tqdm(
                total=len(current_loader),
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=False,
            )
            batch_iter = iter(current_loader)
            while True:
                try:
                    (X, y), _ = next(batch_iter)
                except StopIteration:
                    break

                X = X.to(device)
                y = y.to(device)

                if epoch >= epoch_start_ab and adaptive_sampler is not None:
                    prev_mode = model.training
                    model.eval()
                    with torch.no_grad():
                        grads_per_sample = per_sample_cross_entropy_grads(model, X, y)
                    if prev_mode:
                        model.train()
                    new_batch_size = calculate_batch_size(
                        adabatchgrad_k,
                        adaptive_sampler.state,
                        grads_per_sample,
                        len(train_dataset),
                        adabatchgrad_batch_test,
                        None,
                        adabatchgrad_theta,
                        adabatchgrad_prob_new,
                        adabatchgrad_nu,
                    )
                    new_batch_size = max(batch_size_min, min(batch_size_max, new_batch_size))
                    new_batch_size = max(1, min(new_batch_size, len(train_dataset)))
                    if new_batch_size > adaptive_sampler.batch_size:
                        adaptive_sampler.update_batch_size(new_batch_size)
                        try:
                            (X, y), _ = next(batch_iter)
                        except StopIteration:
                            break
                        X = X.to(device)
                        y = y.to(device)
                    if log_to_mlflow:
                        mlflow.log_metric("adaptive_batch/batch_size_candidate", new_batch_size, step=train_step)
                        mlflow.log_metric("adaptive_batch/batch_size", adaptive_sampler.batch_size, step=train_step)
                        inner_bs = adaptive_sampler.state.get("inner_batch_size")
                        ortho_bs = adaptive_sampler.state.get("ortho_batch_size")
                        if inner_bs is not None:
                            mlflow.log_metric(
                                "adaptive_batch/inner_batch_size", inner_bs, step=train_step
                            )
                        if ortho_bs is not None:
                            mlflow.log_metric(
                                "adaptive_batch/ortho_batch_size", ortho_bs, step=train_step
                            )

                optimizer.zero_grad()
                logits = model(X)
                losses = loss_fn(logits, y)
                loss = losses.mean()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps += 1
                train_step += 1
                pbar.update(1)
            pbar.close()
            if log_to_mlflow:
                epoch_batch_size = (
                    adaptive_sampler.batch_size
                    if adaptive_sampler is not None
                    else batch_size_init
                )
                mlflow.log_metric(
                    "adaptive_batch/batch_size_epoch", epoch_batch_size, step=epoch
                )
        else:
            batch_iter = tqdm(
                current_loader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=False,
            )
            for (X, y), _ in batch_iter:
                X = X.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                logits = model(X)
                losses = loss_fn(logits, y)
                loss = losses.mean()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps += 1
                train_step += 1

                if tracker is not None:
                    grads_now = [p.grad for p in param_list]
                    tracker.update(grads_now)

            if tracker is not None:
                hat_grad, hat_norm_sq, var_sum = tracker.finalize()
                prev_hat_grad = hat_grad
                hat_norm_history.append(hat_norm_sq)
                var_history.append(var_sum)
                if log_to_mlflow:
                    mlflow.log_metric("adaptive_batch/F_hat_norm_sq", hat_norm_sq, step=epoch)
                    mlflow.log_metric("adaptive_batch/var_sum", var_sum, step=epoch)
                    if hat_norm_sq > 0:
                        ratio_now = var_sum / hat_norm_sq
                        mlflow.log_metric("adaptive_batch/ratio_raw", ratio_now, step=epoch)

            # Snapshot model parameters at the end of the epoch to measure movement between epochs.
            prev_prev_params = prev_params
            prev_params = _clone_model_params(model)

        train_loss_epoch = total_loss / max(1, steps)
        val_loss, val_metrics = _evaluate(model, val_loader, loss_fn, device)
        test_loss, test_metrics = _evaluate(model, test_loader, loss_fn, device)

        if log_to_mlflow:
            mlflow.log_metric("train_loss", train_loss_epoch, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v, step=epoch)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v, step=epoch)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_test = {
                "test_loss": test_loss,
                **test_metrics,
                "epoch": epoch,
            }
        if val_metrics.get("accuracy", -1.0) > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_test_acc = {
                "test_loss": test_loss,
                **test_metrics,
                "epoch": epoch,
            }

        if scheduler is not None:
            scheduler.step()

    if log_to_mlflow:
        mlflow.log_metric("best_val_f1", best_val_f1)
        mlflow.log_metric("best_val_acc", best_val_acc)
        if best_test:
            mlflow.log_metrics({f"best_{k}": v for k, v in best_test.items() if k != "epoch"})
            mlflow.log_metric("best_epoch", best_test.get("epoch", -1))
        if best_test_acc:
            mlflow.log_metrics(
                {f"best_acc_{k}": v for k, v in best_test_acc.items() if k != "epoch"}
            )
            mlflow.log_metric("best_acc_epoch", best_test_acc.get("epoch", -1))
        mlflow.end_run()

    return {
        "best_val_f1": best_val_f1,
        "best_val_acc": best_val_acc,
        "best_test": best_test,
        "best_test_acc": best_test_acc,
        "hat_norm_history": hat_norm_history,
        "var_history": var_history,
    }
