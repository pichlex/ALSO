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
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


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


class DeterministicCIFARTransform:
    def __init__(self, mode: str, train: bool, seed: int, mean: tuple[float, float, float], std: tuple[float, float, float]):
        self.mode = mode
        self.train = train
        self.seed = seed
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

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


class DeterministicCIFAR10Transform(DeterministicCIFARTransform):
    def __init__(self, mode: str, train: bool, seed: int):
        super().__init__(mode=mode, train=train, seed=seed, mean=CIFAR10_MEAN, std=CIFAR10_STD)


class PerImageStandardize:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        mean = image.mean()
        std = image.std(unbiased=False)
        min_std = 1.0 / np.sqrt(float(image.numel()))
        return (image - mean) / torch.clamp(std, min=min_std)


class CABSReferenceTransform:
    def __init__(self, train: bool, seed: int):
        self.train = train
        self.seed = seed
        self.standardize = PerImageStandardize()

    def _to_tensor(self, image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def _random_brightness(self, image: torch.Tensor, rng: random.Random) -> torch.Tensor:
        delta = rng.uniform(-63.0, 63.0)
        return image + delta

    def _random_contrast(self, image: torch.Tensor, rng: random.Random) -> torch.Tensor:
        factor = rng.uniform(0.2, 1.8)
        channel_mean = image.mean(dim=(-2, -1), keepdim=True)
        return (image - channel_mean) * factor + channel_mean

    def _crop(self, image, sample_index: int):
        if self.train:
            rng = random.Random(self.seed + sample_index * 1_000_003)
            top = rng.randint(0, 8)
            left = rng.randint(0, 8)
            cropped = image.crop((left, top, left + 24, top + 24))
            if bool(rng.randint(0, 1)):
                cropped = cropped.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
            return cropped, rng

        left = 4
        top = 4
        return image.crop((left, top, left + 24, top + 24)), None

    def __call__(self, image, sample_index: int) -> torch.Tensor:
        image, rng = self._crop(image, sample_index)
        tensor = self._to_tensor(image) * 255.0
        if self.train and rng is not None:
            tensor = self._random_brightness(tensor, rng)
            tensor = self._random_contrast(tensor, rng)
        return self.standardize(tensor)


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


def _load_cifar100(root: Path):
    import torchvision

    return torchvision.datasets.CIFAR100(
        root=root,
        train=True,
        transform=None,
        download=True,
    )


def build_experiment_data(config: ExperimentConfig, device: torch.device) -> ExperimentData:
    dataset_root = (Path(__file__).resolve().parents[2] / "datasets").resolve()
    if config.dataset == "cifar10":
        base_dataset = _load_cifar10(dataset_root)
        mean = CIFAR10_MEAN
        std = CIFAR10_STD
    elif config.dataset == "cifar100":
        base_dataset = _load_cifar100(dataset_root)
        mean = CIFAR100_MEAN
        std = CIFAR100_STD
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    targets = np.asarray(base_dataset.targets)

    indices = np.arange(len(base_dataset))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=targets,
        random_state=config.seed,
    )

    if config.model == "cabs_2conv_3dense":
        train_transform = CABSReferenceTransform(train=True, seed=config.seed)
        val_transform = CABSReferenceTransform(train=False, seed=config.seed)
    else:
        train_transform = DeterministicCIFARTransform(
            mode=config.transform_mode,
            train=True,
            seed=config.seed,
            mean=mean,
            std=std,
        )
        val_mode = "normalize" if config.transform_mode == "all" else config.transform_mode
        val_transform = DeterministicCIFARTransform(
            mode=val_mode,
            train=False,
            seed=config.seed,
            mean=mean,
            std=std,
        )

    train_dataset = IndexedSubset(base_dataset, train_indices, train_transform)
    val_dataset = IndexedSubset(base_dataset, val_indices, val_transform)

    train_order = build_fixed_order(len(train_dataset), config.seed)
    train_sampler = FixedOrderSampler(train_order)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    pin_memory = device.type == "cuda"
    use_persistent_workers = config.num_workers > 0 and config.persistent_workers
    loader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker,
        "generator": generator,
        "drop_last": False,
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = use_persistent_workers
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    return ExperimentData(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        train_order=train_order,
    )
