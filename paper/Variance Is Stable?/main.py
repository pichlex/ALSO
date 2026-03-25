from __future__ import annotations

import argparse

from variance_is_stable.experiment_config import load_config
from variance_is_stable.results import configure_logging, prepare_output_dir
from variance_is_stable.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Variance Is Stable experiments.")
    parser.add_argument("--config", required=True, help="Path to a JSON or YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = prepare_output_dir(config)
    logger = configure_logging(output_dir)
    run_experiment(config=config, output_dir=output_dir, logger=logger)


if __name__ == "__main__":
    main()
