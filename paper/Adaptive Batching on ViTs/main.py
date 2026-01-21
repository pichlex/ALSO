import argparse
import json
from typing import Dict, Any

from data import get_dataloaders
from train import train_model


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def apply_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "dataset": "food101",
        "model_name": "faster_vit_0_224",
        "batch_size": 32,
        "epochs": 20,
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 0.01,
        "adaptive_batch": False,
        "adaptive_batch_beta": 0.0,
        "batch_size_multiplier": 1.0,
        "adaptive_batch_min": 8,
        "adaptive_batch_max": 512,
        "epoch_start_ab": 1,
        "augment": True,
        "num_workers": 4,
        "seed": 42,
        "mlflow_experiment": "Adaptive Batching on ViTs",
        "report_to": "mlflow",
        "scheduler": "cosine",
        "scheduler_T_max": 20,
        "scheduler_eta_min": 5e-6,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "lora_target_modules": ["qkv"],
        "val_split": 0.2,
        "image_size": 224,
        "data_dir": "../datasets",
        "download": True,
    }
    merged = defaults.copy()
    merged.update(cfg)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Adaptive batching with ViTs and LoRA.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    args = parser.parse_args()

    raw_cfg = load_config(args.config)
    config = apply_defaults(raw_cfg)

    train_loader, val_loader, test_loader, num_classes = get_dataloaders(config)
    result = train_model(config, train_loader, val_loader, test_loader, num_classes)
    print("Training finished. Best metrics:", result["best_test"])


if __name__ == "__main__":
    main()
