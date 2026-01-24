import math
import types
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
    dataset_name = str(config.get("dataset", "cifar10")).lower()

    adaptive_enabled = bool(config.get("adaptive_batch", False))
    batch_size_min = int(config.get("adaptive_batch_min", 10))
    batch_size_max = int(config.get("adaptive_batch_max", 1024))
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
    dive_delta = float(config.get("divebatch_delta", 0.1))
    dive_max_batch = int(config.get("divebatch_max_batch", batch_size_max))
    dive_lr_rescale = bool(config.get("divebatch_lr_rescale", False))
    dive_eps = float(config.get("divebatch_eps", 1e-12))
    dive_microbatch = int(config.get("divebatch_microbatch", 8))
    epochs = int(config.get("epochs", 20))

    use_adabatchgrad_strategy = adaptive_enabled and adaptive_strategy == "adabatchgrad"
    use_divebatch_strategy = adaptive_enabled and adaptive_strategy == "divebatch"
    use_variance_strategy = adaptive_enabled and adaptive_strategy == "variance_ratio"

    if use_divebatch_strategy:
        try:
            from backpack import backpack, extend
            from backpack.extensions import BatchGrad
        except ImportError as exc:
            raise ImportError(
                "backpack-for-pytorch is required for the divebatch strategy. "
                "Install it with `pip install backpack-for-pytorch`."
            ) from exc
    model_name = config.get("model", "resnet18").lower()
    if model_name == "resnet18":
        weights = None if dataset_name in {"cifar10", "cifar100"} else torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, 10 if dataset_name == "cifar10" else model.fc.out_features)
        if dataset_name in {"cifar10", "cifar100"}:
            # CIFAR-friendly stem: smaller kernel, no early downsampling to keep index math manageable for Backpack.
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = torch.nn.Identity()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Backpack is incompatible with in-place activations; disable them.
    for m in model.modules():
        if isinstance(m, torch.nn.ReLU):
            m.inplace = False
        if hasattr(torchvision.models.resnet, "BasicBlock") and isinstance(
            m, torchvision.models.resnet.BasicBlock
        ):
            if not hasattr(m, "_patched_for_backpack"):
                def _forward_no_inplace(self, x):
                    identity = x
                    out = self.conv1(x)
                    out = self.bn1(out)
                    out = self.relu(out)

                    out = self.conv2(out)
                    out = self.bn2(out)

                    if self.downsample is not None:
                        identity = self.downsample(x)

                    out = out + identity
                    out = self.relu(out)
                    return out

                m.forward = types.MethodType(_forward_no_inplace, m)
                m._patched_for_backpack = True

    if use_divebatch_strategy:
        model = extend(model)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    if use_divebatch_strategy:
        loss_fn = extend(loss_fn)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

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
    use_divebatch_strategy = adaptive_enabled and adaptive_strategy == "divebatch"
    use_variance_strategy = adaptive_enabled and adaptive_strategy == "variance_ratio"

    if use_divebatch_strategy:
        try:
            from backpack import backpack, extend
            from backpack.extensions import BatchGrad
        except ImportError as exc:
            raise ImportError(
                "backpack-for-pytorch is required for the divebatch strategy. "
                "Install it with `pip install backpack-for-pytorch`."
            ) from exc

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

    dive_batch_size = batch_size_init
    dive_dataset_size = len(train_dataset)

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
        elif use_divebatch_strategy:
            current_loader = DataLoader(
                train_dataset,
                batch_size=dive_batch_size,
                shuffle=True,
                num_workers=int(config.get("num_workers", 2)),
                worker_init_fn=seed_worker,
                generator=g,
                pin_memory=True,
                drop_last=False,
            )
            if log_to_mlflow:
                mlflow.log_metric("adaptive_batch/batch_size", dive_batch_size, step=epoch)
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

        dive_grad_sums: Optional[List[Optional[torch.Tensor]]] = None
        dive_grad_sq_sum = 0.0

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
        elif use_divebatch_strategy:
            batch_iter = tqdm(
                current_loader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=False,
            )
            prev_bs = dive_batch_size
            for (X, y), _ in batch_iter:
                X = X.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                batch_size_now = X.size(0)
                scale_factor = float(batch_size_now)
                if dive_grad_sums is None:
                    dive_grad_sums = [
                        torch.zeros_like(p, device=p.device) if p is not None else None
                        for p in param_list
                    ]

                # Process batch in micro-chunks to avoid huge per-sample Jacobians.
                last_loss_value = 0.0
                for start in range(0, batch_size_now, max(1, dive_microbatch)):
                    end = min(batch_size_now, start + max(1, dive_microbatch))
                    X_chunk = X[start:end]
                    y_chunk = y[start:end]
                    micro_size = X_chunk.size(0)
                    if micro_size == 0:
                        continue
                    factor = micro_size / float(batch_size_now)

                    for p in param_list:
                        if hasattr(p, "grad_batch"):
                            p.grad_batch = None

                    logits = model(X_chunk)
                    losses = loss_fn(logits, y_chunk)
                    loss = losses.mean() * factor
                    last_loss_value = losses.mean().item()
                    with backpack(BatchGrad()):
                        loss.backward()

                    with torch.no_grad():
                        for idx, p in enumerate(param_list):
                            grad_batch = getattr(p, "grad_batch", None)
                            if grad_batch is None:
                                raise RuntimeError(
                                    "grad_batch is missing. Ensure Backpack BatchGrad extension is applied."
                                )
                            # Undo scaling from mean and factor to recover per-sample grads.
                            grad_batch = grad_batch * batch_size_now
                            grad_batch_flat = grad_batch.reshape(grad_batch.shape[0], -1)
                            dive_grad_sq_sum += grad_batch_flat.pow(2).sum().item()
                            batch_grad_sum_flat = grad_batch_flat.sum(dim=0)
                            dive_grad_sums[idx].add_(batch_grad_sum_flat.view_as(p))

                    for p in param_list:
                        if hasattr(p, "grad_batch"):
                            p.grad_batch = None

                optimizer.step()

                total_loss += last_loss_value
                steps += 1
                train_step += 1

            denom_norm_sq = 0.0
            if dive_grad_sums is not None:
                for summed in dive_grad_sums:
                    if summed is None:
                        continue
                    denom_norm_sq += torch.sum(summed * summed).item()
            delta_hat = None
            if epoch >= epoch_start_ab and denom_norm_sq > dive_eps:
                delta_hat = dive_grad_sq_sum / max(denom_norm_sq, dive_eps)
                candidate_bs = int(math.floor(dive_delta * dive_dataset_size * delta_hat))
                candidate_bs = max(batch_size_min, candidate_bs)
                candidate_bs = min(candidate_bs, dive_max_batch, len(train_dataset))
                if candidate_bs < 1:
                    candidate_bs = batch_size_init
                if dive_lr_rescale and prev_bs > 0 and candidate_bs > 0:
                    lr_scale = candidate_bs / float(prev_bs)
                    for group in optimizer.param_groups:
                        if "lr" in group and group["lr"] is not None:
                            group["lr"] *= lr_scale
                    if log_to_mlflow:
                        mlflow.log_metric("adaptive_batch/lr_scale", lr_scale, step=epoch)
                dive_batch_size = candidate_bs
            if log_to_mlflow:
                mlflow.log_metric("adaptive_batch/numerator_norm_sq", dive_grad_sq_sum, step=epoch)
                mlflow.log_metric("adaptive_batch/denominator_norm_sq", denom_norm_sq, step=epoch)
                if delta_hat is not None:
                    mlflow.log_metric("adaptive_batch/ratio_raw", delta_hat, step=epoch)
                    mlflow.log_metric("adaptive_batch/batch_size_next", dive_batch_size, step=epoch)
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
