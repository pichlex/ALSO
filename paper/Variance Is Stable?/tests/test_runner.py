from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from variance_is_stable.data import ExperimentData, FixedOrderSampler, build_fixed_order
from variance_is_stable.experiment_config import ExperimentConfig
from variance_is_stable.results import configure_logging, prepare_output_dir
from variance_is_stable.runner import run_experiment


class IndexedTensorDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index], index


def _build_synthetic_data(config: ExperimentConfig, device: torch.device) -> ExperimentData:
    del device
    inputs = torch.randn(8, 4)
    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    train_dataset = IndexedTensorDataset(inputs, targets)
    val_dataset = IndexedTensorDataset(inputs[:4], targets[:4])
    train_order = build_fixed_order(len(train_dataset), config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=FixedOrderSampler(train_order),
        shuffle=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    return ExperimentData(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        train_order=train_order,
    )


def _build_tiny_model(config: ExperimentConfig) -> torch.nn.Module:
    del config
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )


def test_runner_smoke_saves_expected_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = ExperimentConfig(
        batch_size=4,
        epochs=2,
        metric_microbatch_size=2,
        num_workers=0,
        output_dir=str(tmp_path),
        run_name="smoke",
        transform_mode="none",
        scheduler="none",
    )
    output_dir = prepare_output_dir(config)
    logger = configure_logging(output_dir)

    monkeypatch.setattr("variance_is_stable.runner.build_experiment_data", _build_synthetic_data)
    monkeypatch.setattr("variance_is_stable.runner.build_model", _build_tiny_model)

    run_experiment(config=config, output_dir=output_dir, logger=logger)

    wide_path = output_dir / "metrics_wide.csv"
    long_path = output_dir / "metrics_long.csv"
    config_path = output_dir / "config_used.json"
    log_path = output_dir / "run.log"

    assert wide_path.exists()
    assert long_path.exists()
    assert config_path.exists()
    assert log_path.exists()

    wide_df = pd.read_csv(wide_path)
    assert set(
        [
            "epoch",
            "metric_1",
            "metric_2",
            "metric_3",
            "metric_4",
            "metric_5",
            "train_loss",
            "val_loss",
            "val_acc",
            "metrics_time_sec",
            "train_time_sec",
            "eval_time_sec",
            "data_to_device_time_sec",
            "forward_backward_time_sec",
            "optimizer_step_time_sec",
            "train_steps",
            "samples_per_sec",
            "cuda_mem_alloc_mb",
            "cuda_mem_peak_mb",
        ]
    ).issubset(wide_df.columns)
    assert len(wide_df) == 2
    assert pd.isna(wide_df.loc[0, "metric_4"])
    assert pd.isna(wide_df.loc[0, "metric_5"])
    assert not pd.isna(wide_df.loc[1, "metric_4"])
    assert not pd.isna(wide_df.loc[1, "metric_5"])
    assert (wide_df["train_time_sec"] > 0).all()
    assert (wide_df["metrics_time_sec"] > 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable.")
def test_runner_smoke_on_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = ExperimentConfig(
        batch_size=4,
        epochs=1,
        metric_microbatch_size=2,
        num_workers=0,
        output_dir=str(tmp_path),
        run_name="cuda-smoke",
        transform_mode="none",
        scheduler="none",
        device="cuda",
    )
    output_dir = prepare_output_dir(config)
    logger = logging.getLogger("cuda-smoke")

    monkeypatch.setattr("variance_is_stable.runner.build_experiment_data", _build_synthetic_data)
    monkeypatch.setattr("variance_is_stable.runner.build_model", _build_tiny_model)
    run_experiment(config=config, output_dir=output_dir, logger=logger)

    assert (output_dir / "metrics_wide.csv").exists()


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS is unavailable.",
)
def test_runner_smoke_on_mps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = ExperimentConfig(
        batch_size=4,
        epochs=1,
        metric_microbatch_size=2,
        num_workers=0,
        output_dir=str(tmp_path),
        run_name="mps-smoke",
        transform_mode="none",
        scheduler="none",
        device="mps",
    )
    output_dir = prepare_output_dir(config)
    logger = logging.getLogger("mps-smoke")

    monkeypatch.setattr("variance_is_stable.runner.build_experiment_data", _build_synthetic_data)
    monkeypatch.setattr("variance_is_stable.runner.build_model", _build_tiny_model)
    run_experiment(config=config, output_dir=output_dir, logger=logger)

    assert (output_dir / "metrics_wide.csv").exists()
