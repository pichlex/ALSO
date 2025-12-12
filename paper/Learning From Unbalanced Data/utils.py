"""
Utility functions for training and evaluating models on unbalanced datasets.

This module provides helper classes and functions for working with unbalanced datasets,
specifically for the CIFAR-10 dataset with binary classification setup.
"""

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
    Creates an unbalanced dataset from a balanced dataset by reducing one class.

    This class converts a multi-class dataset to a binary classification problem
    by grouping classes with even/odd labels and then reduces the minority class
    according to the provided imbalance factor.

    Args:
        balanced_dataset: Original balanced dataset that provides (data, target) tuples
        seed: Random seed for reproducibility
        k: Imbalance factor - larger values create more imbalance (default: 2)
    """

    def __init__(
        self, balanced_dataset: torch.utils.data.Dataset, seed: int, k: int = 2
    ):
        X, y = [], []
        for el_x, el_y in balanced_dataset:
            X.append(el_x)
            y.append(el_y)
        X = torch.stack(X)
        y = torch.tensor(y)
        # Convert to binary classification (even/odd classes)
        new_targets = y % 2
        X_first_class = X[new_targets == 1]
        X_zero_class = X[new_targets == 0]
        if k == 1:
            # No imbalance when k=1
            compressed_indexes = np.arange(X_first_class.shape[0])
        else:
            # Reduce class 1 by factor k (keeping only 1/k of samples)
            _, compressed_indexes = train_test_split(
                np.arange(X_first_class.shape[0]),
                test_size=1.0 / k,
                stratify=y[new_targets == 1],
                random_state=seed,
            )
        X_first_class = X_first_class[compressed_indexes]
        self._X = torch.cat([X_zero_class, X_first_class])
        self._y = torch.cat(
            [
                torch.zeros(X_zero_class.shape[0], dtype=y.dtype),
                torch.ones(X_first_class.shape[0], dtype=y.dtype),
            ]
        )

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


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    device: str,
    config: Dict[str, Any],
    compute_weights_fn: Optional[Callable] = None,
    tuning: bool = False,
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
    threshold = float(config.get("pi_threshold", 0.9))
    sampling = config.get("pi_sampling", "pi")
    generator = torch.Generator(device=device)
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
        consumed = 0
        while consumed < total_samples:
            _, batch_idx = optimizer.select_batch(
                threshold=threshold, strategy=sampling, generator=generator
            )
            batch = [dataset[int(i)] for i in batch_idx.tolist()]
            data_list, idx_list = zip(*batch)
            X_list, y_list = zip(*data_list)
            X = torch.stack(X_list).to(device)
            y = torch.tensor(y_list, device=device)
            indexes = torch.tensor(idx_list, device=device)
            batch_size = len(batch_idx)
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
            if loss_val is not None:
                total_loss += loss_val
            consumed += batch_size
            steps += 1
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
            if loss_val is not None:
                total_loss += loss_val
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

    # Use different number of epochs for tuning if specified
    if "n_epoches_tune" not in config:
        config["n_epoches_tune"] = config["n_epoches"]

    # Use tqdm progress bar during regular training but not when tuning
    e_list = (
        range(config["n_epoches_tune"]) if tuning else tqdm(range(config["n_epoches"]))
    )

    for e in e_list:
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
        )
        if log_to_mlflow:
            mlflow.log_metric("batches_per_epoch", n_batches, step=e)

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
