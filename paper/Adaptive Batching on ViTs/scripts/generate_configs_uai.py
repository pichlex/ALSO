#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ViT Food101 SGD configs for basic, seesaw, and aboba variants."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("paper/Adaptive Batching on ViTs/configs-uai/sgd/food101"),
        help="Root directory to store generated configs (variant subfolders are created).",
    )
    parser.add_argument("--dataset", default="food101")
    parser.add_argument("--optimizer", default="sgd")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--scheduler-T-max", dest="scheduler_T_max", type=int, default=25)
    parser.add_argument("--scheduler-eta-min", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-path", default="models/faster_vit_0.pth.tar")
    parser.add_argument("--adaptive-min", type=int, default=8, help="Lower cap for adaptive batch.")
    parser.add_argument("--adaptive-max", type=int, default=4096, help="Upper cap for adaptive batch.")
    parser.add_argument("--adaptive-beta", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora-targets", nargs="+", default=["qkv"], help="Target modules for LoRA tuning."
    )
    parser.add_argument("--report-to", default="mlflow")
    parser.add_argument("--mlflow-pattern", default="{optimizer}-{dataset}-{variant}-bs{batch_size}-seed{seed}")
    parser.add_argument(
        "--run-name-pattern",
        default=None,
        help="Optional run_name format string. Same placeholders as mlflow pattern.",
    )

    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--basic-batch-sizes", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--seesaw-batch-sizes", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument(
        "--aboba-batch-sizes", type=int, nargs="+", default=[8, 16, 64, 256, 512, 1024]
    )

    parser.add_argument("--seesaw-alpha", type=float, default=3.0)
    parser.add_argument("--seesaw-epoch-start", type=int, default=2)
    parser.add_argument(
        "--seesaw-batch-multiplier",
        type=float,
        default=1.0,
        help="Multiplier kept for completeness; Seesaw ignores it in training.",
    )

    parser.add_argument("--aboba-strategy", default="variance_ratio")
    parser.add_argument("--aboba-batch-multiplier", type=float, default=2.0)
    parser.add_argument("--aboba-epoch-start", type=int, default=2)

    parser.add_argument("--dry-run", action="store_true", help="Print planned files without writing.")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to overwrite existing config files.",
    )
    return parser.parse_args()


def _base_config(args: argparse.Namespace) -> Dict:
    return {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": "cosine",
        "scheduler_T_max": args.scheduler_T_max,
        "scheduler_eta_min": args.scheduler_eta_min,
        "augment": args.augment,
        "image_size": args.image_size,
        "adaptive_batch_min": args.adaptive_min,
        "adaptive_batch_max": args.adaptive_max,
        "adaptive_batch_beta": args.adaptive_beta,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_targets,
        "pretrained": args.pretrained,
        "model_path": args.model_path,
        "report_to": args.report_to,
    }


def _apply_variant(
    base: Dict,
    variant: str,
    batch_size: int,
    seed: int,
    args: argparse.Namespace,
) -> Dict:
    cfg = dict(base)
    cfg["batch_size"] = batch_size
    cfg["seed"] = seed
    cfg["mlflow_experiment"] = args.mlflow_pattern.format(
        optimizer=args.optimizer,
        dataset=args.dataset,
        variant=variant,
        batch_size=batch_size,
        seed=seed,
    )
    if args.run_name_pattern:
        cfg["run_name"] = args.run_name_pattern.format(
            optimizer=args.optimizer,
            dataset=args.dataset,
            variant=variant,
            batch_size=batch_size,
            seed=seed,
        )

    if variant == "basic":
        cfg.update(
            {
                "adaptive_batch": False,
                "batch_size_multiplier": 1.0,
                "epoch_start_ab": 1,
            }
        )
    elif variant == "seesaw":
        cfg.update(
            {
                "adaptive_batch": True,
                "adaptive_batch_strategy": "seesaw",
                "seesaw_alpha": args.seesaw_alpha,
                "batch_size_multiplier": args.seesaw_batch_multiplier,
                "epoch_start_ab": args.seesaw_epoch_start,
            }
        )
    elif variant == "aboba":
        cfg.update(
            {
                "adaptive_batch": True,
                "adaptive_batch_strategy": args.aboba_strategy,
                "batch_size_multiplier": args.aboba_batch_multiplier,
                "epoch_start_ab": args.aboba_epoch_start,
            }
        )
    else:
        raise ValueError(f"Unknown variant {variant}")
    return cfg


def _pairs(
    seeds: Iterable[int], batch_sizes: Iterable[int]
) -> Iterable[Tuple[int, int]]:
    for seed in seeds:
        for bs in batch_sizes:
            yield seed, bs


def _write_config(path: Path, cfg: Dict, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    with path.open("w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    base = _base_config(args)
    plan: List[Tuple[str, Path, Dict]] = []

    for variant, batches in [
        ("basic", args.basic_batch_sizes),
        ("seesaw", args.seesaw_batch_sizes),
        ("aboba", args.aboba_batch_sizes),
    ]:
        for seed, bs in _pairs(args.seeds, batches):
            cfg = _apply_variant(base, variant, bs, seed, args)
            filename = f"bs{bs}-seed{seed}.json"
            path = args.output_root / variant / filename
            plan.append((variant, path, cfg))

    if args.dry_run:
        for variant, path, _ in plan:
            print(f"[dry-run] {variant}: {path}")
        print(f"Total configs: {len(plan)}")
        return

    for variant, path, cfg in plan:
        _write_config(path, cfg, args.overwrite)

    counts = {}
    for variant, _, _ in plan:
        counts[variant] = counts.get(variant, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"Wrote {len(plan)} configs -> {args.output_root} ({summary})")


if __name__ == "__main__":
    main()
