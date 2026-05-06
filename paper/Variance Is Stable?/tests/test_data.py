from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from variance_is_stable.data import (
    CABSReferenceTransform,
    DeterministicCIFAR10Transform,
    FixedOrderSampler,
    build_fixed_order,
)


class IndexDataset(Dataset):
    def __len__(self) -> int:
        return 10

    def __getitem__(self, index: int):
        return torch.tensor([index], dtype=torch.float32), 0, index


def test_train_order_same_first_batch_across_epochs() -> None:
    dataset = IndexDataset()
    order = build_fixed_order(len(dataset), seed=123)
    loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=FixedOrderSampler(order),
        shuffle=False,
    )

    first_epoch_batch = next(iter(loader))[2].tolist()
    second_epoch_batch = next(iter(loader))[2].tolist()

    assert first_epoch_batch == second_epoch_batch


def test_deterministic_transforms_all() -> None:
    array = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    image = Image.fromarray(array)

    transform_a = DeterministicCIFAR10Transform(mode="all", train=True, seed=77)
    transform_b = DeterministicCIFAR10Transform(mode="all", train=True, seed=77)

    tensor_a1 = transform_a(image, sample_index=5)
    tensor_a2 = transform_a(image, sample_index=5)
    tensor_b = transform_b(image, sample_index=5)

    assert torch.allclose(tensor_a1, tensor_a2)
    assert torch.allclose(tensor_a1, tensor_b)


def test_cabs_reference_transform_is_deterministic_and_crops_to_24() -> None:
    array = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    image = Image.fromarray(array)

    transform_a = CABSReferenceTransform(train=True, seed=91)
    transform_b = CABSReferenceTransform(train=True, seed=91)
    val_transform = CABSReferenceTransform(train=False, seed=91)

    tensor_a1 = transform_a(image, sample_index=7)
    tensor_a2 = transform_a(image, sample_index=7)
    tensor_b = transform_b(image, sample_index=7)
    val_tensor = val_transform(image, sample_index=7)

    assert tensor_a1.shape == (3, 24, 24)
    assert val_tensor.shape == (3, 24, 24)
    assert torch.allclose(tensor_a1, tensor_a2)
    assert torch.allclose(tensor_a1, tensor_b)
