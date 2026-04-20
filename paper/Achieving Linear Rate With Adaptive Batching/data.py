import os
import random
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset


class IndexedDataset(torch.utils.data.Dataset):
    """Wraps a dataset to return (data, target) along with index."""

    def __init__(self, dataset: torch.utils.data.Dataset, transform=None):
        self._dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        x, y = self._dataset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return (x, y), idx


def set_seed(seed: int) -> Tuple[torch.Generator, Any]:
    """Seed python, numpy, torch; return torch.Generator and worker_init_fn."""

    def seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g, seed_worker


def _build_transforms(dataset: str, augment: bool):
    if dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == "cifar100":
        # Statistics from torchvision docs for CIFAR-100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset == "svhn":
        mean = tuple(x / 255.0 for x in [109.9, 109.7, 113.8])
        std = tuple(x / 255.0 for x in [50.1, 50.6, 50.8])
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    base = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    if augment:
        aug_prefix = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        transform_train = transforms.Compose(aug_prefix + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        transform_train = base

    transform_test = base
    return transform_train, transform_test


def _load_dataset(dataset: str, transform):
    if dataset == "cifar10":
        train_base = torchvision.datasets.CIFAR10(
            root="../datasets", train=True, transform=transform, download=True
        )
        test_base = torchvision.datasets.CIFAR10(
            root="../datasets", train=False, transform=transform, download=True
        )
        targets = train_base.targets
    elif dataset == "cifar100":
        train_base = torchvision.datasets.CIFAR100(
            root="../datasets", train=True, transform=transform, download=True
        )
        test_base = torchvision.datasets.CIFAR100(
            root="../datasets", train=False, transform=transform, download=True
        )
        targets = train_base.targets
    elif dataset == "svhn":
        train_base = torchvision.datasets.SVHN(
            root="../datasets", split="train", transform=transform, download=True
        )
        test_base = torchvision.datasets.SVHN(
            root="../datasets", split="test", transform=transform, download=True
        )
        targets = train_base.labels
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return train_base, test_base, np.array(targets)


def get_dataloaders(config: Dict[str, Any]):
    dataset = config.get("dataset", "cifar10").lower()
    augment = bool(config.get("augment", False))
    batch_size = int(config.get("batch_size", 64))
    num_workers = int(config.get("num_workers", 2))
    seed = int(config.get("seed", 42))

    g, seed_worker = set_seed(seed)
    transform_train, transform_test = _build_transforms(dataset, augment)
    train_base, test_base, targets = _load_dataset(dataset, transform=None)

    # train/val split with stratification on original labels
    train_idx, val_idx = train_test_split(
        np.arange(len(train_base)),
        test_size=0.2,
        stratify=targets,
        random_state=seed,
    )
    train_subset = Subset(train_base, train_idx)
    val_subset = Subset(train_base, val_idx)

    train_ds = IndexedDataset(train_subset, transform=transform_train)
    val_ds = IndexedDataset(val_subset, transform=transform_test)
    test_ds = IndexedDataset(test_base, transform=transform_test)
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=num_workers,
        pin_memory=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=num_workers,
        pin_memory=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=num_workers,
        pin_memory=True,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
