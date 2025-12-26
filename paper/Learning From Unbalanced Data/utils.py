"""
Utility functions for training and evaluating models on unbalanced datasets.

This module provides helper classes and functions for working with unbalanced datasets,
specifically for the CIFAR-10 dataset with binary classification setup.
"""

import math
import torch
import numpy as np
import mlflow
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional, Callable, Any

from sklearn.model_selection import train_test_split
from collections import defaultdict


class UnbalancedDataset(torch.utils.data.Dataset):
    """
    Creates an unbalanced dataset from a balanced dataset by grouping classes.

    By default, it groups classes by even/odd labels (binary) and downsamples
    the minority class by factor ``k``. For multiclass mode, custom
    ``class_groups`` and ``class_ratios`` can be provided to control target
    proportions without oversampling.

    Args:
        balanced_dataset: Original balanced dataset that provides (data, target) tuples
        seed: Random seed for reproducibility
        k: Imbalance factor for binary mode (default: 2)
        class_groups: Optional list of lists defining groups of original labels
        class_ratios: Optional list/tuple with desired ratios per group. If
            provided, it overrides ``k`` and enforces ratios via downsampling
            only (no oversampling).
    """

    def __init__(
        self,
        balanced_dataset: torch.utils.data.Dataset,
        seed: int,
        k: int = 2,
        class_groups: Optional[List[List[int]]] = None,
        class_ratios: Optional[List[int]] = None,
    ):
        X, y = [], []
        for el_x, el_y in balanced_dataset:
            X.append(el_x)
            y.append(el_y)
        X = torch.stack(X)
        y = torch.tensor(y)
        unique_labels = sorted(set(y.tolist()))
        default_groups = [
            [label for label in unique_labels if label % 2 == 0],
            [label for label in unique_labels if label % 2 == 1],
        ]
        groups = class_groups or default_groups
        label_to_group = {}
        for group_idx, labels in enumerate(groups):
            for label in labels:
                if label in label_to_group:
                    raise ValueError(f"Label {label} is assigned to multiple groups")
                label_to_group[int(label)] = group_idx
        missing_labels = [label for label in unique_labels if label not in label_to_group]
        if missing_labels:
            raise ValueError(
                f"Labels {missing_labels} are not assigned to any class_group"
            )

        new_targets = torch.tensor([label_to_group[int(label)] for label in y.tolist()])
        group_indices = [np.where(new_targets.numpy() == i)[0] for i in range(len(groups))]
        if any(len(idxs) == 0 for idxs in group_indices):
            empty = [i for i, idxs in enumerate(group_indices) if len(idxs) == 0]
            raise ValueError(f"Groups {empty} have no samples in the dataset")

        ratios: Optional[List[int]] = None
        if class_ratios is not None:
            ratios = [int(r) for r in class_ratios]
            if len(ratios) != len(groups):
                raise ValueError("class_ratios length must match class_groups length")
            if any(r <= 0 for r in ratios):
                raise ValueError("class_ratios must contain positive values only")
        elif k is not None and len(groups) == 2:
            # Maintain original binary behavior via ratios [k, 1]
            ratios = [k, 1]

        rng = np.random.default_rng(seed)
        selected_indexes: List[np.ndarray] = []

        if ratios is None:
            selected_indexes = group_indices
        else:
            anchor_idx = int(np.argmax(ratios))
            anchor_size = len(group_indices[anchor_idx])
            t = anchor_size / ratios[anchor_idx]
            for idxs, ratio in zip(group_indices, ratios):
                target = int(np.floor(t * ratio))
                target = min(target, len(idxs))
                target = max(1, target)
                chosen = rng.choice(idxs, size=target, replace=False)
                selected_indexes.append(chosen)

        selected = np.concatenate(selected_indexes)
        self._X = X[selected]
        self._y = new_targets[selected]
        self.n_classes = len(groups)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self._X)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the i-th sample from the dataset.

        Args:
            i: Index of the sample to fetch

        Returns:
            Tuple containing (data, target)
        """
        el = (self._X[i], self._y[i])
        return el


class IndexedDataset(torch.utils.data.Dataset):
    """
    Wraps a dataset to additionally return indices of samples.

    This wrapper allows tracking sample indices during training, which is useful
    for implementing sample-based weighting schemes.

    Args:
        dataset: Original dataset that provides (data, target) tuples
        transform: Optional transform to apply to the data
    """

    def __init__(
        self, dataset: torch.utils.data.Dataset, transform: Optional[Callable] = None
    ):
        self._dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self._dataset)

    def __getitem__(self, i: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], int]:
        """
        Returns the i-th sample along with its index.

        Args:
            i: Index of the sample to fetch

        Returns:
            Tuple containing ((data, target), index)
        """
        X, y = self._dataset[i]
        if self.transform is not None:
            X = self.transform(X)
        return (X, y), i


class FixedOrderSampler(torch.utils.data.Sampler[int]):
    """Simple sampler that yields a fixed, precomputed order of indexes."""

    def __init__(self, indices: List[int]):
        super().__init__(None)
        self._indices = [int(i) for i in indices]

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


class AdaptiveBatchTracker:
    """
    Tracks hat{F} and variance terms for adaptive batch size computation.

    We only store running sums to avoid extra memory, and we keep gradients on
    the model device to stay inexpensive on a single GPU.
    """

    def __init__(self, device: str):
        self.device = device
        self.reset(None, None)

    def reset(
        self,
        hat_ref_grad: Optional[List[Optional[torch.Tensor]]],
        hat_ref_loss: Optional[float],
    ) -> None:
        self.hat_ref_grad = hat_ref_grad
        self.hat_ref_loss = hat_ref_loss
        self.grad_sums: Optional[List[Optional[torch.Tensor]]] = None
        self.loss_max_sum: float = 0.0
        self.var_sum: float = 0.0
        self.steps: int = 0

    def _accumulate_grads(self, grads: List[Optional[torch.Tensor]]) -> None:
        if self.grad_sums is None:
            self.grad_sums = [
                g.detach().clone() if g is not None else None for g in grads
            ]
            return
        for idx, g in enumerate(grads):
            if g is None:
                continue
            if self.grad_sums[idx] is None:
                self.grad_sums[idx] = g.detach().clone()
            else:
                self.grad_sums[idx].add_(g.detach())

    def _accumulate_variance(
        self, grads: List[Optional[torch.Tensor]], losses_raw: torch.Tensor
    ) -> None:
        if self.hat_ref_grad is None or self.hat_ref_loss is None:
            return
        grad_diff_sq = 0.0
        for g, h in zip(grads, self.hat_ref_grad):
            if g is None or h is None:
                continue
            grad_diff = g.detach() - h
            grad_diff_sq += torch.sum(grad_diff * grad_diff).item()
        loss_diff = torch.max(torch.abs(losses_raw - self.hat_ref_loss))
        self.var_sum += 2.0 * grad_diff_sq + 2.0 * float(loss_diff.item() ** 2)

    def update(
        self, grads: List[Optional[torch.Tensor]], losses_raw: Optional[torch.Tensor]
    ) -> None:
        if losses_raw is None:
            return
        loss_tensor = losses_raw.detach()
        loss_max = loss_tensor.max()
        self._accumulate_grads(grads)
        self._accumulate_variance(grads, loss_tensor)
        self.loss_max_sum += loss_max.item()
        self.steps += 1

    def finalize(
        self,
    ) -> Tuple[
        Optional[List[Optional[torch.Tensor]]], Optional[float], float, float
    ]:
        if self.steps == 0 or self.grad_sums is None:
            return None, None, 0.0, self.var_sum
        hat_grad: List[Optional[torch.Tensor]] = []
        hat_grad_norm_sq = 0.0
        for g_sum in self.grad_sums:
            if g_sum is None:
                hat_grad.append(None)
                continue
            g_hat = g_sum / float(self.steps)
            hat_grad.append(g_hat)
            hat_grad_norm_sq += torch.sum(g_hat * g_hat).item()
        hat_loss = self.loss_max_sum / float(self.steps)
        hat_norm_sq = 2.0 * hat_grad_norm_sq + 2.0 * (hat_loss ** 2)
        return hat_grad, hat_loss, hat_norm_sq, self.var_sum


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    device: str,
    config: Dict[str, Any],
    compute_weights_fn: Optional[Callable] = None,
    tuning: bool = False,
    adaptive_tracker: Optional[AdaptiveBatchTracker] = None,
) -> Tuple[float, int]:
    """
    Performs a single training step (one epoch) for the model.

    Implements flexible loss weighting mechanisms that can be used for
    distributionally robust optimization.

    Args:
        model: The neural network model to train
        optimizer: The optimizer for updating model parameters
        dataloader: DataLoader providing batches of training data
        loss_fn: Loss function that returns per-sample losses
        device: Device to run computations on ('cpu' or 'cuda')
        config: Dictionary containing configuration parameters
        compute_weights_fn: Optional function to compute sample weights
        tuning: Whether the model is being tuned (controls logging behavior)
        adaptive_tracker: Optional tracker for adaptive batching statistics

    Returns:
        Tuple of (average loss value for the epoch, number of batches processed)
    """
    use_dynamic = config.get("dynamic_batch", False)
    threshold = float(config.get("pi_threshold", 0.9))
    sampling = config.get("pi_sampling", "pi")
    pi_order = config.get("pi_strategy", "desc")
    generator = torch.Generator(device=device)
    if "seed" in config:
        generator.manual_seed(config["seed"])
    log_to_mlflow = config.get("report_to") == "mlflow" and not tuning
    param_list = [p for group in optimizer.param_groups for p in group["params"]]

    model.train()
    total_loss = 0.0
    steps = 0

    if use_dynamic:
        if not hasattr(optimizer, "select_batch"):
            raise AttributeError("dynamic_batch requires optimizer.select_batch")
        dataset = dataloader.dataset
        total_samples = len(dataset)
        use_cached_batch = config.get("cached_batch", False)
        cache_mask = torch.zeros(total_samples, dtype=torch.bool, device=device) if use_cached_batch else None
        consumed = 0
        while consumed < total_samples:
            batch_info: Dict[str, Any] = {}
            _, batch_idx = optimizer.select_batch(
                threshold=threshold,
                strategy=sampling,
                order=pi_order,
                generator=generator,
                excluded_mask=cache_mask,
            )
            if batch_idx.numel() == 0:
                break
            batch = [dataset[int(i)] for i in batch_idx.tolist()]
            data_list, idx_list = zip(*batch)
            X_list, y_list = zip(*data_list)
            X = torch.stack(X_list).to(device)
            y = torch.tensor(y_list, device=device)
            indexes = torch.tensor(idx_list, device=device)
            batch_size = len(batch_idx)
            if log_to_mlflow:
                mlflow.log_metric("batch_size", batch_size, step=config["train_step"])
            if cache_mask is not None:
                cache_mask[batch_idx] = True

            def closure(w=None, scale=None):
                """
                Closure function for optimizers that support it.

                Computes loss and gradients, with optional sample weighting.

                Args:
                    w: Optional tensor of weights for each sample
                    scale: Optional scaling factor for losses

                Returns:
                    Tuple of (per-sample losses, logged loss value)
                """
                optimizer.zero_grad()
                preds = model(X)
                losses = loss_fn(preds, y)
                if adaptive_tracker is not None:
                    batch_info["losses_raw"] = losses.detach()
                if w is None and compute_weights_fn is not None:
                    w = compute_weights_fn(losses)
                    scale = 1
                if w is not None and scale is not None:
                    losses = losses * scale
                    loss = (w * losses).sum()
                    loss.backward()
                    loss_log = losses.mean().item()
                else:
                    loss = losses.mean()
                    loss.backward()
                    loss_log = loss.item()
                return losses, loss_log

            closure.device = device

            loss_val = optimizer.step(closure=closure, groups_indexes=indexes)
            if loss_val is not None:
                total_loss += loss_val
            consumed += batch_size
            steps += 1
            if log_to_mlflow:
                config["train_step"] += 1
            if adaptive_tracker is not None:
                grads_now = [p.grad for p in param_list]
                adaptive_tracker.update(grads_now, batch_info.get("losses_raw"))
    else:
        for steps, ((X, y), indexes) in enumerate(dataloader, start=1):
            batch_info: Dict[str, Any] = {}
            X, y = X.to(device), y.to(device)
            batch_size = y.size(0)
            if log_to_mlflow:
                mlflow.log_metric("batch_size", batch_size, step=config["train_step"])

            def closure(w=None, scale=None):
                """
                Closure function for optimizers that support it.

                Computes loss and gradients, with optional sample weighting.

                Args:
                    w: Optional tensor of weights for each sample
                    scale: Optional scaling factor for losses

                Returns:
                    Tuple of (per-sample losses, logged loss value)
                """
                optimizer.zero_grad()
                preds = model(X)
                losses = loss_fn(preds, y)
                if adaptive_tracker is not None:
                    batch_info["losses_raw"] = losses.detach()
                if w is None and compute_weights_fn is not None:
                    w = compute_weights_fn(losses)
                    scale = 1
                if w is not None and scale is not None:
                    losses = losses * scale
                    loss = (w * losses).sum()
                    loss.backward()
                    loss_log = losses.mean().item()
                else:
                    loss = losses.mean()
                    loss.backward()
                    loss_log = loss.item()
                return losses, loss_log

            closure.device = device

            try:
                # For optimizers supporting group-based weights (like ALSO)
                loss_val = optimizer.step(closure=closure, groups_indexes=indexes.to(device))
            except TypeError:
                try:
                    loss_val = optimizer.step(closure=closure, dataset_indexes=indexes.to(device))
                except TypeError:
                    # Fall back for standard optimizers (Adam, SGD, etc.)
                    loss_val = optimizer.step(closure=closure)
            if loss_val is not None:
                total_loss += loss_val
            if log_to_mlflow:
                config["train_step"] += 1
            if adaptive_tracker is not None:
                grads_now = [p.grad for p in param_list]
                adaptive_tracker.update(grads_now, batch_info.get("losses_raw"))

    return total_loss / max(1, steps), steps


@torch.no_grad()
def eval_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    device: str,
    config: Dict[str, Any],
    validation: bool = True,
    tuning: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluates the model on validation or test data.

    Computes loss and classification metrics (F1 score, precision, recall).

    Args:
        model: The neural network model to evaluate
        dataloader: DataLoader providing batches of evaluation data
        loss_fn: Loss function that returns per-sample losses
        device: Device to run computations on ('cpu' or 'cuda')
        config: Dictionary containing configuration parameters
        validation: Whether this is validation data (True) or test data (False)
        tuning: Whether the model is being tuned (controls logging behavior)

    Returns:
        Tuple containing (average loss, dictionary of evaluation metrics)
    """
    model.eval()
    total_loss = 0
    total_true = np.array([])
    total_pred = np.array([])
    for t, ((X, y), _) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        preds = model(X)
        losses = loss_fn(preds, y)
        y_pred = preds.argmax(dim=-1)
        total_true = np.append(total_true, y.cpu().detach().numpy())
        total_pred = np.append(total_pred, y_pred.cpu().detach().numpy())
        loss = losses.mean()
        total_loss += loss.item()

    # Determine whether to use binary or weighted averaging for metrics
    average = "weighted" if len(np.unique(total_true)) > 2 else "binary"
    f1 = f1_score(total_true, total_pred, average=average)
    precision = precision_score(
        total_true, total_pred, zero_division=0.0, average=average
    )
    recall = recall_score(total_true, total_pred, average=average)
    results = {"f1": f1, "precision": precision, "recall": recall}
    return total_loss / t, results


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    device: str,
    config: Dict[str, Any],
    tuning: bool = False,
    compute_weights_fn: Optional[Callable] = None,
) -> Tuple[
    torch.nn.Module,
    Dict[str, List[float]],
    Dict[str, List[float]],
    List[Optional[List[float]]],
]:
    """
    Trains the model for multiple epochs and evaluates on validation and test data.

    Args:
        model: The neural network model to train
        optimizer: The optimizer for updating model parameters
        train_dataloader: DataLoader providing batches of training data
        val_dataloader: DataLoader providing batches of validation data
        test_dataloader: DataLoader providing batches of test data
        loss_fn: Loss function that returns per-sample losses
        device: Device to run computations on ('cpu' or 'cuda')
        config: Dictionary containing configuration parameters
        tuning: Whether the model is being tuned (controls logging behavior)
        compute_weights_fn: Optional function to compute sample weights

    Returns:
        Tuple containing (trained model, validation metrics history, test metrics history, pi snapshots)
    """
    val_metrics = defaultdict(list)
    test_metrics = defaultdict(list)
    pi_history: List[Optional[List[float]]] = []
    config["train_step"], config["val_step"], config["test_step"] = 0, 0, 0
    log_to_mlflow = config.get("report_to") == "mlflow" and not tuning
    log_pi_snapshots = config.get("log_pi", True)

    # Use different number of epochs for tuning if specified
    if "n_epoches_tune" not in config:
        config["n_epoches_tune"] = config["n_epoches"]

    adaptive_enabled = config.get("adaptive_batch", False)
    batch_size_min = int(config.get("adaptive_batch_min", 10))
    batch_size_max = int(config.get("adaptive_batch_max", 512))
    init_batch_size = int(config.get("batch_size", 1))
    epoch_start_ab = int(config.get("epoch_start_ab", 2))
    train_dataset = train_dataloader.dataset
    prev_hat_grad: Optional[List[Optional[torch.Tensor]]] = None
    prev_hat_loss: Optional[float] = None
    hat_norm_history: List[float] = []
    var_history: List[float] = []

    def _build_fixed_order_indices() -> List[int]:
        g = torch.Generator()
        g.manual_seed(config.get("seed", 0))
        total = len(train_dataset)
        if config.get("use_sampler", False):
            base_ds = getattr(train_dataset, "_dataset", train_dataset)
            labels = getattr(base_ds, "_y", None)
            n_classes = getattr(base_ds, "n_classes", None)
            if labels is None or n_classes is None:
                raise ValueError(
                    "Adaptive batching with sampler requires dataset labels."
                )
            labels = labels.to(torch.long)
            class_counts = torch.stack(
                [(labels == c).sum() for c in range(int(n_classes))]
            ).float()
            weights = (1.0 / class_counts)[labels]
            weights = weights / weights.sum()
            return torch.multinomial(
                weights, total, replacement=True, generator=g
            ).tolist()
        return torch.randperm(total, generator=g).tolist()

    if adaptive_enabled:
        if config.get("dynamic_batch", False):
            # The adaptive batch schedule controls batch sizes; disable competing mode.
            config["dynamic_batch"] = False
        fixed_indices = _build_fixed_order_indices()

        def _make_adaptive_loader(batch_size: int) -> torch.utils.data.DataLoader:
            sampler = FixedOrderSampler(fixed_indices)
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, batch_size, drop_last=False
            )
            return torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                num_workers=train_dataloader.num_workers,
                worker_init_fn=train_dataloader.worker_init_fn,
                pin_memory=getattr(train_dataloader, "pin_memory", False),
                collate_fn=train_dataloader.collate_fn,
            )
    else:
        fixed_indices = None

    # Use tqdm progress bar during regular training but not when tuning
    e_list = (
        range(config["n_epoches_tune"]) if tuning else tqdm(range(config["n_epoches"]))
    )

    for e in e_list:
        adaptive_active = adaptive_enabled and e >= epoch_start_ab
        if adaptive_active:
            var_used = var_history[e - 1] if (e > 0 and len(var_history) >= e) else None
            denom_used = hat_norm_history[e - 2] if (e > 1 and len(hat_norm_history) >= e - 1) else None
            if e >= 2 and var_used is not None and denom_used is not None and denom_used > 0:
                batch_size_epoch = math.floor(var_used / denom_used)  # floor as requested
            else:
                batch_size_epoch = init_batch_size
            batch_size_epoch = int(max(batch_size_min, min(batch_size_max, batch_size_epoch)))
            batch_size_epoch = max(1, min(batch_size_epoch, len(train_dataset)))
            current_train_loader = _make_adaptive_loader(batch_size_epoch)
            if hasattr(optimizer, "loss_scale"):
                optimizer.loss_scale = len(train_dataset) / float(batch_size_epoch)
            if log_to_mlflow:
                mlflow.log_metric("adaptive_batch/batch_size", batch_size_epoch, step=e)
                if var_used is not None and denom_used is not None:
                    mlflow.log_metric("adaptive_batch/var_sum", var_used, step=e)
                    mlflow.log_metric("adaptive_batch/F_hat_norm_sq", denom_used, step=e)
        else:
            current_train_loader = train_dataloader
            var_used = None
            denom_used = None

        tracker = AdaptiveBatchTracker(device) if adaptive_enabled else None
        if tracker is not None:
            tracker.reset(prev_hat_grad, prev_hat_loss)
        # Train for one epoch
        train_loss, n_batches = train_step(
            model,
            optimizer,
            current_train_loader,
            loss_fn,
            device,
            config,
            tuning=tuning,
            compute_weights_fn=compute_weights_fn,
            adaptive_tracker=tracker,
        )
        if tracker is not None:
            hat_grad, hat_loss, hat_norm_sq, var_sum = tracker.finalize()
            prev_hat_grad, prev_hat_loss = hat_grad, hat_loss
            hat_norm_history.append(hat_norm_sq)
            var_history.append(var_sum)
        if log_to_mlflow:
            mlflow.log_metric("train_loss", train_loss, step=e)
            mlflow.log_metric("batches_per_epoch", n_batches, step=e)

        # Evaluate on validation set
        val_loss, val_results = eval_step(
            model, val_dataloader, loss_fn, device, config, tuning=tuning
        )
        for key in val_results:
            val_metrics[key].append(val_results[key])

        # Evaluate on test set
        test_loss, test_results = eval_step(
            model,
            test_dataloader,
            loss_fn,
            device,
            config,
            validation=False,
            tuning=tuning,
        )
        for key in test_results:
            test_metrics[key].append(test_results[key])

        # Log metrics and pi snapshots to MLflow when requested
        if log_to_mlflow:
            mlflow.log_metric("val_loss", val_loss, step=e)
            mlflow.log_metric("test_loss", test_loss, step=e)
            for key, value in val_results.items():
                mlflow.log_metric(f"val_{key}", value, step=e)
            for key, value in test_results.items():
                mlflow.log_metric(f"test_{key}", value, step=e)

        if log_pi_snapshots and not tuning and hasattr(optimizer, "pi"):
            pi_snapshot = optimizer.pi.detach().clone()
            pi_history.append(pi_snapshot.cpu().tolist())
            if log_to_mlflow:
                uc = config.get("unbalance_coef", "na")
                if isinstance(uc, (list, tuple)):
                    uc = "-".join(str(x) for x in uc)
                mlflow.log_dict(
                    {"epoch": int(e), "pi": pi_snapshot.cpu().tolist()},
                    f"pi/uc_{uc}/epoch_{int(e)}.json",
                )

    return model, val_metrics, test_metrics, pi_history


class ImportanceLoss(torch.nn.Module):
    """
    Importance-weighted loss function with clipping and warm-up.

    This loss function modifies a base loss function by applying an exponential
    transformation to emphasize difficult examples. It includes clipping to prevent
    exploding gradients and a warm-up period during which the original loss is used.

    Args:
        loss_fn: Base loss function
        tau: Temperature parameter for scaling (default: 1)
        C: Clipping parameter to prevent exploding gradients (default: 10)
        warmup_steps: Number of iterations to use original loss before switching (default: 100)
    """

    def __init__(
        self, loss_fn: Callable, tau: float = 1, C: float = 10, warmup_steps: int = 100
    ):
        super().__init__()
        self.tau = tau
        self.C = C
        self.base_loss = loss_fn
        self.warmup_steps = warmup_steps
        self.__current_iter = 0

    def clip_importance_loss_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Applies the importance-weighted loss transformation with clipping.

        The transformation is: (exp(τ * clip(loss, -C, C)) - 1) / τ

        Args:
            x: Model predictions
            y: Ground truth labels

        Returns:
            Transformed loss value
        """
        return (
            self.base_loss(x, y)
            .mul(self.tau)
            .clip(-self.C, self.C)
            .exp()
            .mean()
            .sub(1)
            .divide(self.tau)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss, using the base loss during warm-up and
        the transformed loss afterward.

        Args:
            x: Model predictions
            y: Ground truth labels

        Returns:
            Loss value
        """
        if self.__current_iter < self.warmup_steps:
            self.__current_iter += 1
            return self.base_loss(x, y)
        else:
            return self.clip_importance_loss_fn(x, y)
