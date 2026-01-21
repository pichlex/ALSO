import os
import random
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets.utils import download_and_extract_archive

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


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


def _build_transforms(image_size: int, augment: bool):
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    test_tfms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    if not augment:
        return test_tfms, test_tfms
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tfms, test_tfms


def _extract_targets(dataset: torch.utils.data.Dataset) -> np.ndarray:
    for attr in ("targets", "labels", "_labels", "y"):
        if hasattr(dataset, attr):
            values = getattr(dataset, attr)
            if isinstance(values, np.ndarray):
                return values
            return np.array(values)
    if hasattr(dataset, "samples"):
        return np.array([s[1] for s in dataset.samples])
    raise ValueError("Could not extract targets from dataset for stratified split.")


class TinyImageNetDataset(torch.utils.data.Dataset):
    """Minimal Tiny ImageNet loader with optional download."""

    def __init__(self, root: str, split: str = "train", transform=None, download: bool = False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.base_dir = os.path.join(self.root, "tiny-imagenet-200")
        if download:
            self._download()
        if not os.path.isdir(self.base_dir):
            raise FileNotFoundError(
                f"Tiny ImageNet not found at {self.base_dir}. Set download=True to fetch it."
            )
        self._load_metadata()
        self._load_samples()

    def _download(self) -> None:
        if os.path.isdir(self.base_dir):
            return
        os.makedirs(self.root, exist_ok=True)
        download_and_extract_archive(
            _TINY_IMAGENET_URL,
            download_root=self.root,
            filename="tiny-imagenet-200.zip",
            remove_finished=True,
        )

    def _load_metadata(self) -> None:
        wnids_path = os.path.join(self.base_dir, "wnids.txt")
        with open(wnids_path) as f:
            self.classes = [line.strip() for line in f if line.strip()]
        self.class_to_idx = {c: idx for idx, c in enumerate(self.classes)}

    def _load_samples(self) -> None:
        self.samples: List[Tuple[str, int]] = []
        if self.split == "train":
            train_dir = os.path.join(self.base_dir, "train")
            for wnid in self.classes:
                img_dir = os.path.join(train_dir, wnid, "images")
                if not os.path.isdir(img_dir):
                    continue
                for fname in os.listdir(img_dir):
                    if not fname.lower().endswith(".jpeg"):
                        continue
                    path = os.path.join(img_dir, fname)
                    self.samples.append((path, self.class_to_idx[wnid]))
        elif self.split == "val":
            ann_path = os.path.join(self.base_dir, "val", "val_annotations.txt")
            img_dir = os.path.join(self.base_dir, "val", "images")
            with open(ann_path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    fname, wnid = parts[0], parts[1]
                    path = os.path.join(img_dir, fname)
                    if wnid not in self.class_to_idx:
                        continue
                    self.samples.append((path, self.class_to_idx[wnid]))
        else:
            raise ValueError(f"Unsupported split: {self.split}")
        self.targets = [s[1] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_dataloaders(config: Dict[str, Any]):
    dataset = config.get("dataset", "food101").lower()
    augment = bool(config.get("augment", True))
    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 4))
    seed = int(config.get("seed", 42))
    image_size = int(config.get("image_size", 224))
    val_split = float(config.get("val_split", 0.2))
    data_dir = config.get("data_dir", "../datasets")
    download = bool(config.get("download", True))

    g, seed_worker = set_seed(seed)
    train_tfms, test_tfms = _build_transforms(image_size, augment)

    if dataset == "food101":
        import torchvision

        base_train = torchvision.datasets.Food101(root=data_dir, split="train", download=download)
        base_test = torchvision.datasets.Food101(root=data_dir, split="test", download=download)
        targets = _extract_targets(base_train)
        classes = base_train.classes
    elif dataset in ("tiny_imagenet", "tiny-imagenet"):
        base_train = TinyImageNetDataset(root=data_dir, split="train", transform=None, download=download)
        base_test = TinyImageNetDataset(root=data_dir, split="val", transform=None, download=download)
        targets = np.array(base_train.targets)
        classes = base_train.classes
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    val_split = max(0.01, min(0.5, val_split))
    stratify = targets if len(np.unique(targets)) > 1 else None
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=val_split,
        stratify=stratify,
        random_state=seed,
        shuffle=True,
    )

    train_ds = IndexedDataset(Subset(base_train, train_idx), transform=train_tfms)
    val_ds = IndexedDataset(Subset(base_train, val_idx), transform=test_tfms)
    test_ds = IndexedDataset(base_test, transform=test_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=num_workers,
        pin_memory=True,
    )
    num_classes = len(classes)
    return train_loader, val_loader, test_loader, num_classes


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
