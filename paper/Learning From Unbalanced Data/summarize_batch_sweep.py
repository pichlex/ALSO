"""Summarize batch sweep JSONL logs across seeds.

Reads one or more batch_sweep_seed*.jsonl files and computes mean/std
across seeds for each metric, batch size, and unbalance coefficient.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _load_records(paths: Iterable[Path]) -> Iterable[Dict]:
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _group_values(records: Iterable[Dict]) -> Dict[Tuple[int, int, str, str], List[float]]:
    grouped: Dict[Tuple[int, int, str, str], List[float]] = defaultdict(list)
    for rec in records:
        batch_size = rec["batch_size"]
        uc = rec["unbalance_coef"]
        run_name = rec.get("run_name", "")
        metrics = rec["metrics"]
        for metric_name, stats in metrics.items():
            # Expect stats like {"mean": float, "std": float}; fall back to raw value
            mean_value = stats["mean"] if isinstance(stats, dict) and "mean" in stats else float(stats)
            grouped[(batch_size, uc, run_name, metric_name)].append(float(mean_value))
    return grouped


def _summaries(grouped: Dict[Tuple[int, int, str, str], List[float]]) -> List[Dict]:
    summary_rows = []
    for (batch_size, uc, run_name, metric_name), values in sorted(grouped.items()):
        arr = np.array(values, dtype=float)
        summary_rows.append(
            {
                "batch_size": batch_size,
                "unbalance_coef": uc,
                "run_name": run_name,
                "metric": metric_name,
                "count": int(arr.size),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
            }
        )
    return summary_rows


def summarize(inputs: List[Path], output: Path) -> None:
    records = _load_records(inputs)
    grouped = _group_values(records)
    rows = _summaries(grouped)

    if output == Path("-"):
        for row in rows:
            print(json.dumps(row))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize batch sweep JSONL logs across seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_root = Path(__file__).resolve().parent
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="*",
        default=None,
        help="Paths or glob patterns to batch_sweep JSONL files. If omitted, uses runs/batch_sweep_seed*.jsonl next to this script.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for summary JSONL (use '-' for stdout).",
    )
    args = parser.parse_args()

    if args.inputs:
        paths: List[Path] = []
        for pattern in args.inputs:
            p = Path(pattern)
            if any(ch in pattern for ch in "*?[]"):
                paths.extend(sorted(Path().glob(pattern)))
            elif p.is_file():
                paths.append(p)
            else:
                # Try relative to script directory
                paths.extend(sorted(default_root.glob(pattern)))
    else:
        paths = sorted((default_root / "runs").glob("batch_sweep_seed*.jsonl"))

    if not paths:
        raise SystemExit("No input JSONL files found. Provide --inputs or place files in runs/.")

    if args.output is None:
        args.output = default_root / "runs" / "batch_sweep_summary.jsonl"

    # Normalize output
    output_path = Path(args.output)
    return argparse.Namespace(inputs=paths, output=output_path)


if __name__ == "__main__":
    cli_args = parse_args()
    summarize(cli_args.inputs, cli_args.output)
