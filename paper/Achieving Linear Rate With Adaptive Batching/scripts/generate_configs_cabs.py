#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "configs-cabs" / "sgd" / "cifar10"
DEFAULT_PAPER_OUTPUT_ROOT = PROJECT_ROOT / "configs-cabs-paper" / "sgd" / "cifar10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CABS-style CIFAR-10 comparison configs."
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--paper-scale", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--adaptive-batch-size", type=int, default=16)
    parser.add_argument("--max-train-steps", type=int, default=8000)
    parser.add_argument("--max-examples-accessed", type=int, default=10_000_000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--eta-min", type=float, default=0.001)
    parser.add_argument("--adaptive-min", type=int, default=16)
    parser.add_argument("--adaptive-max", type=int, default=2048)
    parser.add_argument("--seesaw-alpha", type=float, default=2.0)
    parser.add_argument("--aboba-batch-multiplier", type=float, default=1.0)
    parser.add_argument("--run-split-parts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _base_config(args: argparse.Namespace) -> Dict:
    cfg = {
        "dataset": "cifar10",
        "model": "cabs_2conv_3dense",
        "seed": args.seed,
        "augment": True,
        "preprocessing": "cabs_reference",
        "batch_size": args.batch_size,
        "epochs": args.max_examples_accessed if args.paper_scale else args.max_train_steps,
        "optimizer": "sgd",
        "lr": args.lr,
        "momentum": 0.0,
        "nesterov": False,
        "weight_decay": 0.0,
        "scheduler": "cosine",
        "scheduler_step_unit": "examples" if args.paper_scale else "step",
        "scheduler_eta_min": args.eta_min,
        "adaptive_batch_min": args.adaptive_min,
        "adaptive_batch_max": args.adaptive_max,
        "adaptive_batch_beta": 0.0,
        "epoch_start_ab": 1,
        "report_to": "mlflow",
        "tune_runs": 100,
        "n_epoches_tune": 10,
        "tune_name": "cifar10_cabs_setup_sgd",
        "use_old_tune_params": False,
    }
    if args.paper_scale:
        cfg["max_examples_accessed"] = args.max_examples_accessed
        cfg["scheduler_T_max_examples"] = args.max_examples_accessed
    else:
        cfg["max_train_steps"] = args.max_train_steps
        cfg["scheduler_T_max"] = args.max_train_steps
    return cfg


def _variant_config(base: Dict, variant: str, args: argparse.Namespace) -> Dict:
    cfg = dict(base)
    if args.paper_scale and variant != "basic":
        cfg["batch_size"] = args.adaptive_batch_size
    cfg["mlflow_experiment"] = (
        f"cabs-setup-cifar10-{variant}-bs{cfg['batch_size']}-seed{args.seed}"
    )
    cfg["run_name"] = variant

    if variant == "basic":
        cfg["adaptive_batch"] = False
        cfg["batch_size_multiplier"] = 1.0
    elif variant == "cabs":
        cfg["adaptive_batch"] = True
        cfg["adaptive_batch_strategy"] = "cabs"
        cfg["cabs_running_avg_constant"] = 0.95
        cfg["cabs_eps"] = 0.0
        cfg["cabs_c"] = 1.0
        cfg["cabs_batch_lr"] = args.lr
        cfg["batch_size_multiplier"] = 1.0
    elif variant == "variance_ratio_iter":
        cfg["adaptive_batch"] = True
        cfg["adaptive_batch_strategy"] = "variance_ratio_iter"
        cfg["batch_size_multiplier"] = args.aboba_batch_multiplier
    elif variant == "seesaw":
        cfg["adaptive_batch"] = True
        cfg["adaptive_batch_strategy"] = "seesaw"
        cfg["seesaw_alpha"] = args.seesaw_alpha
        cfg["batch_size_multiplier"] = 1.0
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return cfg


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


def _write_json(path: Path, cfg: Dict, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    path.write_text(json.dumps(cfg, indent=2) + "\n")


def _write_run_script(path: Path, commands: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_run_script_header() + "\n".join(commands) + "\n")
    os.chmod(path, 0o755)


def main() -> None:
    args = parse_args()
    if args.run_split_parts < 1:
        raise ValueError("--run-split-parts must be >= 1.")

    output_root = args.output_root or (
        DEFAULT_PAPER_OUTPUT_ROOT if args.paper_scale else DEFAULT_OUTPUT_ROOT
    )
    base = _base_config(args)
    variants = ["basic", "cabs", "variance_ratio_iter", "seesaw"]
    plan: List[Tuple[str, Path, Dict]] = []
    for variant in variants:
        cfg = _variant_config(base, variant, args)
        path = output_root / variant / f"bs{cfg['batch_size']}-seed{args.seed}.json"
        plan.append((variant, path, cfg))

    run_order = sorted(plan, key=lambda item: item[1].as_posix())
    commands = [_run_command(output_root, path) for _, path, _ in run_order]
    part_commands = [
        [cmd for idx, cmd in enumerate(commands) if idx % args.run_split_parts == part_idx]
        for part_idx in range(args.run_split_parts)
    ]

    if args.dry_run:
        for variant, path, _ in plan:
            print(f"[dry-run] {variant}: {path}")
        print(f"Total configs: {len(plan)}")
        return

    for _, path, cfg in plan:
        _write_json(path, cfg, args.overwrite)

    _write_run_script(output_root / "run-all.sh", commands)
    for idx, cmds in enumerate(part_commands, start=1):
        _write_run_script(output_root / f"run-part{idx}.sh", cmds)

    print(f"Wrote {len(plan)} configs -> {output_root}")
    print(
        f"Wrote run scripts -> {output_root} "
        f"(run-all={len(commands)}, "
        + ", ".join(f"run-part{idx + 1}={len(cmds)}" for idx, cmds in enumerate(part_commands))
        + ")"
    )


if __name__ == "__main__":
    main()
