from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler

from .experiment_config import ExperimentConfig


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class FixedOrderSampler(Sampler[int]):
    def __init__(self, indices: list[int]):
        super().__init__()
        self._indices = [int(index) for index in indices]

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


class DeterministicCIFAR10Transform:
    def __init__(self, mode: str, train: bool, seed: int):
        self.mode = mode
        self.train = train
        self.seed = seed
        self.mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(CIFAR10_STD, dtype=torch.float32).view(3, 1, 1)

    def _to_tensor(self, image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def _train_augment(self, image, sample_index: int):
        rng = random.Random(self.seed + sample_index * 1_000_003)
        top = rng.randint(0, 8)
        left = rng.randint(0, 8)
        flip = bool(rng.randint(0, 1))

        padded = ImageOps.expand(image, border=4, fill=0)
        cropped = padded.crop((left, top, left + 32, top + 32))
        if flip:
            cropped = cropped.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
        return cropped

    def __call__(self, image, sample_index: int) -> torch.Tensor:
        if self.mode == "all" and self.train:
            image = self._train_augment(image, sample_index)

        tensor = self._to_tensor(image)
        if self.mode in {"normalize", "all"}:
            tensor = self._normalize(tensor)
        return tensor


class IndexedSubset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        indices: np.ndarray,
        transform: Callable[[object, int], torch.Tensor],
    ):
        self.base_dataset = base_dataset
        self.indices = [int(index) for index in indices.tolist()]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        sample_index = self.indices[idx]
        image, label = self.base_dataset[sample_index]
        tensor = self.transform(image, sample_index)
        return tensor, int(label), sample_index


@dataclass(slots=True)
class ExperimentData:
    train_dataset: IndexedSubset
    val_dataset: IndexedSubset
    train_loader: DataLoader
    val_loader: DataLoader
    train_order: list[int]


def build_fixed_order(length: int, seed: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randperm(length, generator=generator).tolist()


def _load_cifar10(root: Path):
    import torchvision

    return torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        transform=None,
        download=True,
    )


def build_experiment_data(config: ExperimentConfig, device: torch.device) -> ExperimentData:
    dataset_root = (Path(__file__).resolve().parents[2] / "datasets").resolve()
    base_dataset = _load_cifar10(dataset_root)
    targets = np.asarray(base_dataset.targets)

    indices = np.arange(len(base_dataset))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=targets,
        random_state=config.seed,
    )

    train_transform = DeterministicCIFAR10Transform(
        mode=config.transform_mode,
        train=True,
        seed=config.seed,
    )
    val_mode = "normalize" if config.transform_mode == "all" else config.transform_mode
    val_transform = DeterministicCIFAR10Transform(
        mode=val_mode,
        train=False,
        seed=config.seed,
    )

    train_dataset = IndexedSubset(base_dataset, train_indices, train_transform)
    val_dataset = IndexedSubset(base_dataset, val_indices, val_transform)

    train_order = build_fixed_order(len(train_dataset), config.seed)
    train_sampler = FixedOrderSampler(train_order)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=False,
    )

    return ExperimentData(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        train_order=train_order,
    )
