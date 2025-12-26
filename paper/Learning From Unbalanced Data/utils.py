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
from dataclasses import dataclass


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


def flatten_gradients(model: torch.nn.Module) -> torch.Tensor:
    """
    Flatten all parameter gradients into a single vector.

    Returns an empty tensor if no gradients are present.
    """
    grads = []
    device = None
    for p in model.parameters():
        if p.grad is None:
            continue
        if device is None:
            device = p.grad.device
        grads.append(p.grad.view(-1))
    if not grads:
        device = device or next(model.parameters()).device
        return torch.empty(0, device=device)
    return torch.cat(grads)


@dataclass
class AdaptiveHat:
    theta: torch.Tensor
    pi: torch.Tensor
    norm_sq: float


class AdaptiveBatchTracker:
    """
    Tracks \\hat{F} and variance terms for adaptive batching across one epoch.
    """

    def __init__(
        self,
        pi_template: torch.Tensor,
        hat_reference: Optional[AdaptiveHat],
        eps: float,
    ):
        self.count = 0
        self.theta_mean: Optional[torch.Tensor] = None
        self.pi_sum = torch.zeros_like(pi_template)
        self.var_sum = 0.0
        self.hat_reference = hat_reference
        self.eps = eps

    def update(
        self,
        theta_vec: torch.Tensor,
        losses: torch.Tensor,
        pi_selected: torch.Tensor,
        batch_idx: torch.Tensor,
        batch_size: int,
    ) -> None:
        self.count += 1
        if self.theta_mean is None:
            self.theta_mean = theta_vec.detach().clone()
        else:
            # Running mean update to avoid large intermediate buffers
            self.theta_mean.add_(theta_vec - self.theta_mean, alpha=1 / float(self.count))

        contrib = (-losses.detach() / (pi_selected.detach() + self.eps)) / float(batch_size)
        self.pi_sum.index_add_(0, batch_idx, contrib)

        if self.hat_reference is None:
            return

        # Variance accumulation: ||F_i - hat_F||_*^2 with hat_F from previous epoch
        diff_theta = theta_vec - self.hat_reference.theta
        theta_l2_sq = torch.dot(diff_theta, diff_theta)

        diff_pi = -self.hat_reference.pi.clone()
        diff_pi.index_add_(0, batch_idx, contrib)
        pi_linf = diff_pi.abs().max()

        self.var_sum += float(2 * theta_l2_sq + 2 * (pi_linf ** 2))

    def finalize(self) -> Tuple[Optional[AdaptiveHat], Optional[float]]:
        if self.count == 0 or self.theta_mean is None:
            return None, None

        pi_mean = self.pi_sum / float(self.count)
        theta_l2_sq = torch.dot(self.theta_mean, self.theta_mean)

        pi_l2_sq = torch.dot(pi_mean, pi_mean)
        pi_linf = pi_mean.abs().max()

        norm_sq = float(2 * theta_l2_sq + 2 * (pi_linf ** 2))

        hat = AdaptiveHat(theta=self.theta_mean, pi=pi_mean, norm_sq=norm_sq)
        return hat, self.var_sum


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

    Returns:
        Tuple of (average loss value for the epoch, number of batches processed)
    """
    use_dynamic = config.get("dynamic_batch", False)
    generator = torch.Generator(device=device)
    if "seed" in config and config["seed"] is not None:
        generator.manual_seed(int(config["seed"]))
    if "seed" in config:
        generator.manual_seed(config["seed"])
    log_to_mlflow = config.get("report_to") == "mlflow" and not tuning

    model.train()
    total_loss = 0.0
    steps = 0

    if use_dynamic:
        if not hasattr(optimizer, "select_batch"):
            raise AttributeError("dynamic_batch requires optimizer.select_batch")
        dataset = dataloader.dataset
        total_samples = len(dataset)
        batch_size = int(config.get("current_batch_size", config.get("batch_size", 1)))
        num_batches = max(1, math.ceil(total_samples / batch_size))

        for _ in range(num_batches):
            _, batch_idx = optimizer.select_batch(
                batch_size=batch_size,
                generator=generator,
            )
            if batch_idx.numel() == 0:
                continue
            batch = [dataset[int(i)] for i in batch_idx.tolist()]
            data_list, idx_list = zip(*batch)
            X_list, y_list = zip(*data_list)
            X = torch.stack(X_list).to(device)
            y = torch.tensor(y_list, device=device)
            indexes = torch.tensor(idx_list, device=device)
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
            raw_losses: Optional[torch.Tensor] = None
            pi_selected: Optional[torch.Tensor] = None
            if isinstance(loss_val, tuple):
                loss_val, raw_losses, pi_selected = loss_val

            if loss_val is not None:
                total_loss += loss_val
            steps += 1
            if adaptive_tracker is not None:
                if raw_losses is None or pi_selected is None:
                    raise RuntimeError("Adaptive batching requires optimizer to return losses and pi_selected.")
                theta_vec = flatten_gradients(model)
                adaptive_tracker.update(
                    theta_vec=theta_vec,
                    losses=raw_losses,
                    pi_selected=pi_selected,
                    batch_idx=indexes,
                    batch_size=batch_size,
                )
            if log_to_mlflow:
                config["train_step"] += 1
    else:
        for steps, ((X, y), indexes) in enumerate(dataloader, start=1):
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
            raw_losses: Optional[torch.Tensor] = None
            pi_selected: Optional[torch.Tensor] = None
            if isinstance(loss_val, tuple):
                loss_val, raw_losses, pi_selected = loss_val
            if loss_val is not None:
                total_loss += loss_val
            if adaptive_tracker is not None and raw_losses is not None and pi_selected is not None:
                theta_vec = flatten_gradients(model)
                adaptive_tracker.update(
                    theta_vec=theta_vec,
                    losses=raw_losses,
                    pi_selected=pi_selected,
                    batch_idx=indexes.to(device),
                    batch_size=batch_size,
                )
            if log_to_mlflow:
                config["train_step"] += 1

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
    adaptive_enabled = config.get("adaptive_batching", False)
    init_batch_size = config.get("init_batch_size", config.get("batch_size"))
    min_batch_size = config.get("min_batch_size", 10)
    max_batch_size = config.get("max_batch_size", 512)
    adaptive_state: Dict[str, Any] = {
        "hat_history": [],
        "hat_norm_history": [],
        "var_history": [],
    }

    # Use different number of epochs for tuning if specified
    if "n_epoches_tune" not in config:
        config["n_epoches_tune"] = config["n_epoches"]

    # Use tqdm progress bar during regular training but not when tuning
    e_list = (
        range(config["n_epoches_tune"]) if tuning else tqdm(range(config["n_epoches"]))
    )

    for e in e_list:
        # Determine batch size for this epoch under adaptive batching
        if adaptive_enabled:
            if e < 2:
                current_batch_size = init_batch_size
            elif len(adaptive_state["var_history"]) >= 1 and len(adaptive_state["hat_norm_history"]) >= 2:
                numerator = adaptive_state["var_history"][-1]
                denom = adaptive_state["hat_norm_history"][-2]
                if denom <= 0:
                    current_batch_size = max_batch_size
                else:
                    current_batch_size = max(
                        min_batch_size,
                        min(max_batch_size, int(math.floor(numerator / denom))),
                    )
            else:
                current_batch_size = init_batch_size
            config["current_batch_size"] = current_batch_size
            config["batch_size"] = current_batch_size
            if log_to_mlflow:
                mlflow.log_metric("adaptive_batch_size", current_batch_size, step=e)

        adaptive_tracker: Optional[AdaptiveBatchTracker] = None
        if adaptive_enabled:
            pi_template = getattr(optimizer, "pi", None)
            eps_val = getattr(optimizer, "eps", 1e-12)
            if pi_template is None:
                raise AttributeError("Adaptive batching requires optimizer with `pi` attribute.")
            hat_ref = adaptive_state["hat_history"][-2] if len(adaptive_state["hat_history"]) >= 2 else None
            adaptive_tracker = AdaptiveBatchTracker(
                pi_template=pi_template.detach(),
                hat_reference=hat_ref,
                eps=eps_val,
            )

        # Train for one epoch
        train_loss, n_batches = train_step(
            model,
            optimizer,
            train_dataloader,
            loss_fn,
            device,
            config,
            tuning=tuning,
            compute_weights_fn=compute_weights_fn,
            adaptive_tracker=adaptive_tracker,
        )
        if log_to_mlflow:
            mlflow.log_metric("batches_per_epoch", n_batches, step=e)
        hat_info, var_sum = (None, None)
        if adaptive_tracker is not None:
            hat_info, var_sum = adaptive_tracker.finalize()
            if hat_info is not None:
                adaptive_state["hat_history"].append(hat_info)
                adaptive_state["hat_norm_history"].append(hat_info.norm_sq)
                if log_to_mlflow:
                    mlflow.log_metric("hat_norm_sq", hat_info.norm_sq, step=e)
            elif adaptive_enabled:
                adaptive_state["hat_norm_history"].append(0.0)
            if var_sum is not None:
                adaptive_state["var_history"].append(var_sum)
                if log_to_mlflow:
                    mlflow.log_metric("var_sum_sq", var_sum, step=e)
        elif adaptive_enabled:
            adaptive_state["var_history"].append(None)
            if adaptive_state["hat_prev"] is not None:
                adaptive_state["hat_norm_history"].append(adaptive_state["hat_prev"].norm_sq)

        # Evaluate on validation set
        _, val_results = eval_step(
            model, val_dataloader, loss_fn, device, config, tuning=tuning
        )
        for key in val_results:
            val_metrics[key].append(val_results[key])

        # Evaluate on test set
        _, test_results = eval_step(
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
