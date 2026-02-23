#!/usr/bin/env python3
import argparse
import json
import os
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
        default="{optimizer}-{dataset}-{variant}-prcd-bs{batch_size}-seed{seed}",
    )
    parser.add_argument(
        "--run-name-pattern",
        default=None,
        help="Optional run_name format string. Same placeholders as mlflow pattern.",
    )

    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--basic-batch-sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--seesaw-batch-sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--aboba-batch-sizes", type=int, nargs="+", default=[64, 128, 256])

    parser.add_argument("--seesaw-alpha", type=float, default=3.0)
    parser.add_argument("--seesaw-epoch-start", type=int, default=2)
    parser.add_argument(
        "--seesaw-batch-multiplier",
        type=float,
        default=1.0,
        help="Multiplier kept for completeness; Seesaw ignores it in training.",
    )

    parser.add_argument("--aboba-strategy", default="variance_ratio_preconditioned")
    parser.add_argument(
        "--aboba-batch-multipliers",
        type=float,
        nargs="+",
        default=[0.04],
    )
    parser.add_argument("--aboba-epoch-start", type=int, default=2)

    parser.add_argument("--dry-run", action="store_true", help="Print planned files without writing.")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to overwrite existing config files.",
    )
    parser.add_argument(
        "--write-run-scripts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to generate run-all.sh and run-part*.sh in output root.",
    )
    parser.add_argument(
        "--run-split-parts",
        type=int,
        default=2,
        help="How many run-part*.sh scripts to split run-all commands into.",
    )
    parser.add_argument(
        "--prune-stale-configs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete JSON configs under output root that are not in the generated plan.",
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


def _run_script_header() -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "UV_BIN=${UV_BIN:-uv}\n"
        "SCRIPT_DIR=$(cd -- \"$(dirname -- \"${BASH_SOURCE[0]}\")\" && pwd)\n"
        "WORKDIR=${WORKDIR:-$(cd -- \"${SCRIPT_DIR}/../../..\" && pwd)}\n"
        "CONFIG_DIR=${CONFIG_DIR:-${SCRIPT_DIR}}\n"
        "MAIN_PATH=${MAIN_PATH:-\"main.py\"}\n"
        "\n"
        "cd \"${WORKDIR}\"\n"
    )


def _run_command(output_root: Path, config_path: Path) -> str:
    rel = config_path.relative_to(output_root).as_posix()
    return f"\"${{UV_BIN}}\" run \"${{MAIN_PATH}}\" --config \"${{CONFIG_DIR}}/{rel}\""


def _write_run_script(path: Path, commands: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = _run_script_header() + "\n".join(commands) + ("\n" if commands else "")
    path.write_text(body)
    os.chmod(path, 0o755)


def _find_stale_json_configs(output_root: Path, planned_paths: List[Path]) -> List[Path]:
    planned_set = {p.resolve() for p in planned_paths}
    existing = [p for p in output_root.rglob("*.json") if p.is_file()]
    stale = [p for p in existing if p.resolve() not in planned_set]
    return sorted(stale, key=lambda p: p.as_posix())


def main() -> None:
    args = parse_args()
    if args.run_split_parts < 1:
        raise ValueError("--run-split-parts must be >= 1.")
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

    run_order = sorted(plan, key=lambda item: item[1].as_posix())
    run_commands = [_run_command(args.output_root, path) for _, path, _ in run_order]
    planned_paths = [path for _, path, _ in plan]
    stale_configs = _find_stale_json_configs(args.output_root, planned_paths)
    part_commands: List[List[str]] = []
    if args.write_run_scripts:
        for part_idx in range(args.run_split_parts):
            cmds = [cmd for cmd_idx, cmd in enumerate(run_commands) if cmd_idx % args.run_split_parts == part_idx]
            part_commands.append(cmds)

    if args.dry_run:
        for variant, path, _ in plan:
            print(f"[dry-run] {variant}: {path}")
        print(f"Total configs: {len(plan)}")
        if args.prune_stale_configs:
            print(f"[dry-run] stale-json-to-delete: {len(stale_configs)}")
        if args.write_run_scripts:
            print(f"[dry-run] run-all.sh commands: {len(run_commands)}")
            for idx, cmds in enumerate(part_commands, start=1):
                print(f"[dry-run] run-part{idx}.sh commands: {len(cmds)}")
        return

    for _, path, cfg in plan:
        _write_config(path, cfg, args.overwrite)

    pruned_count = 0
    if args.prune_stale_configs:
        for stale in stale_configs:
            stale.unlink()
            pruned_count += 1

    if args.write_run_scripts:
        _write_run_script(args.output_root / "run-all.sh", run_commands)
        for idx, cmds in enumerate(part_commands, start=1):
            _write_run_script(args.output_root / f"run-part{idx}.sh", cmds)

    counts = {}
    for variant, _, _ in plan:
        counts[variant] = counts.get(variant, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"Wrote {len(plan)} configs -> {args.output_root} ({summary})")
    if args.prune_stale_configs:
        print(f"Pruned stale configs: {pruned_count}")
    if args.write_run_scripts:
        part_summary = ", ".join(
            f"run-part{idx + 1}={len(cmds)}" for idx, cmds in enumerate(part_commands)
        )
        print(
            f"Wrote run scripts -> {args.output_root} "
            f"(run-all={len(run_commands)}, {part_summary})"
        )


if __name__ == "__main__":
    main()
