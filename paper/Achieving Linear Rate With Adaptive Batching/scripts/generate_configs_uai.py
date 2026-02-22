#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate configs for basic, seesaw, and aboba variants "
            "(default profile: CIFAR100 + ResNet34 + AdamW)."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "paper/Achieving Linear Rate With Adaptive Batching/configs-uai/adamw/cifar100"
        ),
        help="Root directory to store generated configs (variant subfolders are created).",
    )
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--model", default="resnet34")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum to write into configs when optimizer=sgd.",
    )
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--scheduler-T-max", dest="scheduler_T_max", type=int, default=40)
    parser.add_argument("--scheduler-eta-min", type=float, default=1e-5)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaptive-min", type=int, default=10, help="Lower cap for adaptive batch.")
    parser.add_argument("--adaptive-max", type=int, default=4096, help="Upper cap for adaptive batch.")
    parser.add_argument("--adaptive-beta", type=float, default=0.0)
    parser.add_argument("--report-to", default="mlflow")
    parser.add_argument("--use-old-tune-params", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--mlflow-pattern",
        default="{optimizer}-{dataset}-{variant}-bs{batch_size}-seed{seed}",
    )
    parser.add_argument(
        "--run-name-pattern",
        default=None,
        help="Optional run_name format string. Same placeholders as mlflow pattern.",
    )

    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--basic-batch-sizes", type=int, nargs="+", default=[128])
    parser.add_argument("--seesaw-batch-sizes", type=int, nargs="+", default=[128])
    parser.add_argument("--aboba-batch-sizes", type=int, nargs="+", default=[128])

    parser.add_argument("--seesaw-alpha", type=float, default=3.0)
    parser.add_argument("--seesaw-epoch-start", type=int, default=2)
    parser.add_argument(
        "--seesaw-batch-multiplier",
        type=float,
        default=1.0,
        help="Multiplier kept for completeness; Seesaw ignores it in training.",
    )

    parser.add_argument("--aboba-strategy", default="variance_ratio")
    parser.add_argument(
        "--aboba-batch-multipliers",
        type=float,
        nargs="+",
        default=[2.0, 4.0, 8.0, 16.0],
    )
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
    base = {
        "dataset": args.dataset,
        "model": args.model,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": "cosine",
        "scheduler_T_max": args.scheduler_T_max,
        "scheduler_eta_min": args.scheduler_eta_min,
        "augment": args.augment,
        "adaptive_batch_min": args.adaptive_min,
        "adaptive_batch_max": args.adaptive_max,
        "adaptive_batch_beta": args.adaptive_beta,
        "report_to": args.report_to,
        "use_old_tune_params": args.use_old_tune_params,
    }
    if str(args.optimizer).lower() == "sgd":
        base["momentum"] = args.momentum
    return base


def _format_name(
    pattern: str,
    variant: str,
    batch_size: int,
    seed: int,
    args: argparse.Namespace,
    aboba_batch_multiplier: Optional[float] = None,
) -> str:
    name = pattern.format(
        optimizer=args.optimizer,
        dataset=args.dataset,
        variant=variant,
        batch_size=batch_size,
        seed=seed,
        aboba_batch_multiplier=aboba_batch_multiplier,
    )
    if variant == "aboba" and "{aboba_batch_multiplier}" not in pattern:
        name = f"{name}-abm{aboba_batch_multiplier}"
    return name


def _apply_variant(
    base: Dict,
    variant: str,
    batch_size: int,
    seed: int,
    args: argparse.Namespace,
    aboba_batch_multiplier: Optional[float] = None,
) -> Dict:
    cfg = dict(base)
    cfg["batch_size"] = batch_size
    cfg["seed"] = seed
    cfg["mlflow_experiment"] = _format_name(
        args.mlflow_pattern,
        variant=variant,
        batch_size=batch_size,
        seed=seed,
        args=args,
        aboba_batch_multiplier=aboba_batch_multiplier,
    )
    if args.run_name_pattern:
        cfg["run_name"] = _format_name(
            args.run_name_pattern,
            variant=variant,
            batch_size=batch_size,
            seed=seed,
            args=args,
            aboba_batch_multiplier=aboba_batch_multiplier,
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
                "batch_size_multiplier": aboba_batch_multiplier,
                "epoch_start_ab": args.aboba_epoch_start,
            }
        )
    else:
        raise ValueError(f"Unknown variant {variant}")
    return cfg


def _pairs(seeds: Iterable[int], batch_sizes: Iterable[int]) -> Iterable[Tuple[int, int]]:
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


def _fmt_multiplier(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def main() -> None:
    args = parse_args()
    base = _base_config(args)
    plan: List[Tuple[str, Path, Dict]] = []

    for seed, bs in _pairs(args.seeds, args.basic_batch_sizes):
        cfg = _apply_variant(base, "basic", bs, seed, args)
        path = args.output_root / "basic" / f"bs{bs}-seed{seed}.json"
        plan.append(("basic", path, cfg))

    for seed, bs in _pairs(args.seeds, args.seesaw_batch_sizes):
        cfg = _apply_variant(base, "seesaw", bs, seed, args)
        path = args.output_root / "seesaw" / f"bs{bs}-seed{seed}.json"
        plan.append(("seesaw", path, cfg))

    for mult in args.aboba_batch_multipliers:
        for seed, bs in _pairs(args.seeds, args.aboba_batch_sizes):
            cfg = _apply_variant(base, "aboba", bs, seed, args, aboba_batch_multiplier=mult)
            suffix = _fmt_multiplier(mult)
            path = args.output_root / "aboba" / f"abm{suffix}" / f"bs{bs}-seed{seed}.json"
            plan.append(("aboba", path, cfg))

    if args.dry_run:
        for variant, path, _ in plan:
            print(f"[dry-run] {variant}: {path}")
        print(f"Total configs: {len(plan)}")
        return

    for _, path, cfg in plan:
        _write_config(path, cfg, args.overwrite)

    counts = {}
    for variant, _, _ in plan:
        counts[variant] = counts.get(variant, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"Wrote {len(plan)} configs -> {args.output_root} ({summary})")


if __name__ == "__main__":
    main()
