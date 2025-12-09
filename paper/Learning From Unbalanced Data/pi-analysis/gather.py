#!/usr/bin/env python3
"""
Collect pi trajectory epoch JSONs from MLflow runs into a structured folder tree.

Expected MLflow layout (per run):
<mlruns>/<exp_id>/<run_id>/
    params/seed
    params/unbalance_coef
    artifacts/pi/uc_<UC>/epoch_*.json

Resulting layout:
pi-analysis/data/uc-<UC>/seed-<SEED>/epoch_*.json

UC and seed are taken from MLflow params when possible, falling back to the
run name in meta.yaml and the artifact folder names.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple


def read_param(run_dir: Path, name: str) -> Optional[str]:
    """Read a single MLflow param value if it exists."""
    path = run_dir / "params" / name
    if path.exists():
        return path.read_text().strip() or None
    return None


def read_run_name(run_dir: Path) -> Optional[str]:
    """Extract run_name from MLflow meta.yaml without requiring PyYAML."""
    meta_path = run_dir / "meta.yaml"
    if not meta_path.exists():
        return None

    for line in meta_path.read_text().splitlines():
        if line.startswith("run_name:"):
            # Strip the key, colon, optional space, and wrapping quotes
            raw = line.split(":", 1)[1].strip()
            return raw.strip("\"'") if raw else None
    return None


def parse_seed_uc(run_dir: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    seed = read_param(run_dir, "seed")
    uc = read_param(run_dir, "unbalance_coef")
    run_name = read_run_name(run_dir)

    if run_name:
        if seed is None:
            match = re.search(r"seed(\d+)", run_name)
            if match:
                seed = match.group(1)
        if uc is None:
            match = re.search(r"uc(\d+)", run_name)
            if match:
                uc = match.group(1)

    return seed, uc, run_name


def gather_run(run_dir: Path, output_root: Path) -> int:
    """Copy all epoch_*.json files for a single run. Returns copied file count."""
    pi_root = run_dir / "artifacts" / "pi"
    if not pi_root.exists():
        return 0

    seed, uc_from_meta, run_name = parse_seed_uc(run_dir)

    copied = 0
    for uc_dir in pi_root.iterdir():
        if not uc_dir.is_dir() or not uc_dir.name.startswith("uc"):
            continue

        uc_match = re.search(r"(\d+)", uc_dir.name)
        uc_value = uc_match.group(1) if uc_match else uc_from_meta
        if uc_value is None:
            print(f"[warn] Skipping {uc_dir} (missing UC value)", file=sys.stderr)
            continue

        seed_value = seed or "unknown"
        dest_dir = output_root / f"uc-{uc_value}" / f"seed-{seed_value}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        for epoch_file in sorted(uc_dir.glob("epoch_*.json")):
            shutil.copy2(epoch_file, dest_dir / epoch_file.name)
            copied += 1

    if copied == 0 and run_name:
        print(f"[info] No epoch files found for run '{run_name}'", file=sys.stderr)

    return copied


def gather_all(mlruns_root: Path, output_root: Path) -> int:
    total_copied = 0
    run_dirs = [p for p in mlruns_root.glob("*/*") if (p / "artifacts").exists()]
    for run_dir in run_dirs:
        copied = gather_run(run_dir, output_root)
        if copied:
            total_copied += copied
            print(f"[ok] {copied} files from {run_dir.relative_to(mlruns_root)}")
    return total_copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect pi epoch JSONs from MLflow runs")
    default_mlruns = Path(__file__).resolve().parents[1] / "mlruns"
    default_output = Path(__file__).resolve().parent / "data"

    parser.add_argument(
        "--mlruns",
        type=Path,
        default=default_mlruns,
        help=f"Path to mlruns root (default: {default_mlruns})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Destination root for collected JSONs (default: {default_output})",
    )

    args = parser.parse_args()
    if not args.mlruns.exists():
        parser.error(f"mlruns path does not exist: {args.mlruns}")

    args.output.mkdir(parents=True, exist_ok=True)
    total = gather_all(args.mlruns, args.output)
    print(f"Copied {total} epoch files into {args.output}")


if __name__ == "__main__":
    main()
