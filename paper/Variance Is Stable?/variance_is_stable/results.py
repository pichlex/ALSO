from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .experiment_config import ExperimentConfig


def prepare_output_dir(config: ExperimentConfig) -> Path:
    root = Path(config.output_dir)
    run_name = config.run_name or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    output_dir = root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("variance_is_stable")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def save_run_artifacts(
    output_dir: Path,
    config: ExperimentConfig,
    epoch_records: list[dict[str, Any]],
) -> None:
    with (output_dir / "config_used.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)

    wide_df = pd.DataFrame(epoch_records)
    wide_df.to_csv(output_dir / "metrics_wide.csv", index=False)

    metric_columns = ["metric_1", "metric_2", "metric_3", "metric_4", "metric_5"]
    long_df = wide_df.melt(
        id_vars=[col for col in wide_df.columns if col not in metric_columns],
        value_vars=metric_columns,
        var_name="metric_name",
        value_name="value",
    )
    long_df.to_csv(output_dir / "metrics_long.csv", index=False)
