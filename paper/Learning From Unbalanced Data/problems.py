"""
Problem definition module for unbalanced CIFAR10 binary classification experiments.

This module provides functions to set up and configure the unbalanced CIFAR10
classification task using various optimization approaches described in Section 5.1
of the paper "Aligning Distributionally Robust Optimization with Practical Deep Learning Needs".
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from typing import Dict, Tuple, Any, Optional, Callable
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import numpy as np
import random

from utils import IndexedDataset, UnbalancedDataset

from optimizers.also import ALSO
from optimizers.deshift import make_spectral_risk_measure, make_extremile_spectrum
from optimizers.drago import DRAGO
import os


def set_global_seed(seed: int = 42) -> Tuple[torch.Generator, Callable]:
    """
    Sets global random seed for reproducibility across all random number generators.

    Args:
        seed: Random seed value (default: 42)

    Returns:
        Tuple containing:
        - Seeded PyTorch generator
        - Worker seed function for DataLoader workers
    """

    def seed_worker(worker_id: int) -> None:
        """Sets random seed for DataLoader workers."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return g, seed_worker


def get_problem(config: Dict[str, Any]) -> Tuple[
    nn.Module,
    torch.optim.Optimizer,
    DataLoader,
    DataLoader,
    DataLoader,
    Callable,
    str,
    Optional[Callable],
]:
    """
    Configures the unbalanced CIFAR10 classification problem based on provided configuration.

    This function:
    1. Sets up reproducible randomness
    2. Prepares datasets (train/validation/test) with specified imbalance
    3. Creates the model architecture
    4. Configures the optimizer based on specified strategy
    5. Returns all components needed for training

    Args:
        config: Dictionary containing experiment configuration parameters

    Returns:
        Tuple containing:
        - Neural network model
        - Optimizer
        - Training data loader
        - Validation data loader
        - Test data loader
        - Loss function
        - Device string ("cpu" or "cuda")
        - Optional weight computation function
    """
    g, seed_worker = set_global_seed(config["seed"])
    sampler = None
    if config["device_id"] is None:
        device = "cpu"
    else:
        device = f"cuda" if torch.cuda.is_available() else "cpu"
    # Define data transformations
    transform_base = transforms.Compose(
        [
            transforms.ToTensor(),  # Basic transformation to tensor
        ]
    )

    # Standard normalization for CIFAR10 with proper mean/std values
    transform_test = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Optionally add data augmentation for training
    if config["augment"]:
        transform_train = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),  # Random crops with padding
                transforms.RandomHorizontalFlip(),  # Random horizontal flips
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    else:
        transform_train = transform_test
    gen = torch.Generator().manual_seed(config["seed"])
    # Load CIFAR10 dataset
    ds = torchvision.datasets.CIFAR10(
        "../datasets", train=True, transform=transform_base, download=True,
    )

    # Split into training and validation sets with stratification
    train_idx, val_idx = train_test_split(
        np.arange(len(ds)),
        test_size=0.2,
        stratify=ds.targets,
        random_state=config["seed"],
    )

    # Create dataset subsets
    ds_train = Subset(ds, train_idx)
    ds_val = Subset(ds, val_idx)

    # Create unbalanced datasets with specified imbalance factor
    ds_train = IndexedDataset(
        UnbalancedDataset(ds_train, seed=config["seed"], k=config["unbalance_coef"]),
        transform=transform_train,
    )
    ds_val = IndexedDataset(
        UnbalancedDataset(ds_val, seed=config["seed"], k=config["unbalance_coef"]),
        transform=transform_test,
    )

    # Create test dataset (either balanced when k=1 or with the same imbalance as training)
    if config["balanced_test"]:
        # Balanced test set (k=1 means no imbalance)
        ds_test = IndexedDataset(
            UnbalancedDataset(
                torchvision.datasets.CIFAR10(
                    "../datasets", train=False, transform=transform_base, download=True
                ),
                seed=config["seed"],
                k=1,  # No imbalance
            ),
            transform=transform_test,
        )
    else:
        # Test set with same imbalance as training
        ds_test = IndexedDataset(
            UnbalancedDataset(
                torchvision.datasets.CIFAR10(
                    "../datasets", train=False, transform=transform_base
                ),
                seed=config["seed"],
                k=config["unbalance_coef"],
            ),
            transform=transform_test,
        )
    # Configure class weighting strategies
    if config["use_sampler"]:
        # Create a weighted sampler to handle class imbalance during batch sampling
        class_counts = [sum(ds_train._dataset._y == 0), sum(ds_train._dataset._y == 1)]
        weights = [1 / float(class_counts[i]) for i in range(len(class_counts))]
        samples_weights_train = np.array(
            [weights[int(t)] for t in ds_train._dataset._y]
        )
        samples_weights_train /= samples_weights_train.sum()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            samples_weights_train, len(samples_weights_train), replacement=True
        )

    if config["use_static_weights"]:
        # Calculate static class weights for loss weighting
        class_counts = [sum(ds_train._dataset._y == 0), sum(ds_train._dataset._y == 1)]
        weights = [1 / float(class_counts[i]) for i in range(len(class_counts))]
        weights_train = torch.tensor(
            [weights[int(t)] for t in ds_train._dataset._y], dtype=torch.float32
        ).to(device)
        config["weights"] = weights_train / weights_train.sum()
    else:
        config["weights"] = None
    # Configure model output size and loss function
    d_out = 2  # Binary classification
    metric_fn = f1_score
    loss_fn = nn.CrossEntropyLoss(reduction="none")  # Per-sample losses for weighting

    # Calculate loss scaling factor
    config["loss_scale"] = len(ds_train) / config["batch_size"]

    # Create data loaders with reproducible randomness
    train_dataloader = DataLoader(
        ds_train,
        batch_size=config["batch_size"],
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=sampler is None,  # Only shuffle if not using weighted sampler
        sampler=sampler,
    )

    val_dataloader = DataLoader(
        ds_val,
        batch_size=config["batch_size"],
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )

    test_dataloader = DataLoader(
        ds_test,
        batch_size=config["batch_size"],
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    # Initialize model architecture
    if config["model"] == "resnet18":
        model = torchvision.models.resnet18().train()
        # Replace final fully connected layer for binary classification
        model.fc = nn.Linear(model.fc.in_features, d_out)

    # Move model to appropriate device (CPU or GPU)
    model.to(device)
    # Configure optimizer based on experiment configuration
    compute_weights_fn = None
    if config["optimizer"] == "adam":
        # Standard Adam optimizer
        optimizer = optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )

    elif config["optimizer"] == "also":
        # ALSO optimizer (Adaptive Loss Scaling Optimizer) from the paper
        if "pi_lr" in config:
            pi_lr = config["pi_lr"]
        else:
            pi_lr = None

        # Optionally initialize with class-balanced weights
        if config["use_init_static_weights"]:
            class_counts = [
                sum(ds_train._dataset._y == 0),
                sum(ds_train._dataset._y == 1),
            ]
            weights = [1 / float(class_counts[i]) for i in range(len(class_counts))]
            pi_reg = torch.tensor(
                [weights[int(t)] for t in ds_train._dataset._y], dtype=torch.float32
            ).to(device)
        else:
            pi_reg = None

        # Initialize ALSO optimizer
        optimizer = ALSO(
            model.parameters(),
            n_groups=len(ds_train),  # One weight per training sample
            batch_size=config["batch_size"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            pi_decay=config["pi_decay"],
            mode=config["optimizer_mode"],
            pi_lr=pi_lr,
            pi_reg=pi_reg,
            pi_init=pi_reg,
            loss_scale=1,
            eps=1e-20,
        )

    elif config["optimizer"] == "dro_loss":
        # Standard optimizer with distributionally robust loss weighting
        optimizer = optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )

        # Configure DRO parameters
        shift_cost = 1.0
        penalty = "kl"  # Options: 'chi2', 'kl'

        # Create extremile spectrum and DRO weight calculation function
        spectrum = make_extremile_spectrum(config["batch_size"], n_draws=2.0)
        compute_weights_fn = make_spectral_risk_measure(
            spectrum, penalty=penalty, shift_cost=shift_cost
        )
    elif config["optimizer"] == "drago":
        # DRAGO - Distributionally Robust Adaptive Gradient Optimizer
        optimizer = DRAGO(
            model.parameters(),
            lr=config["lr"],
            batch_size=config["batch_size"],
            weight_decay=config["weight_decay"],
            data_len=len(ds_train),
        )
        compute_weights_fn = None

    else:
        raise NotImplementedError(f"Unknown optimizer {config['optimizer']}")

    # Only use weight computation function for DRO loss
    if config["optimizer"] != "dro_loss":
        compute_weights_fn = None

    return (
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        loss_fn,
        device,
        compute_weights_fn,
    )
