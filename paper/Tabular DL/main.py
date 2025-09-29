from dataclasses import dataclass

import json
from pathlib import Path
import pandas as pd
import tomli
import tomli_w
import os
import click


TABRED_DATASETS = [
    "cooking-time",
    "delivery-eta",
    "ecom-offers",
    "homecredit-default",
    "homesite-insurance",
    "maps-routing",
    "sberbank-housing",
    "weather",
]

@dataclass
class Config:
    optimizer_name: str = 'also'
    dataset: str = "california"
    model: str = "mlp-periodic"

    def initialize(self, device):
        if device is None:
            device = "''"
        dataset_prefix = ""
        if self.dataset in TABRED_DATASETS:
            dataset_prefix = "tabred/"
        self.device = device
        self.optimization_configs_path = f'exp/{self.optimizer_name}.toml'
        self.base_path = f'exp/{self.model}/{dataset_prefix}{self.dataset}'


def run_experiment(config: Config):
    opt_config = tomli.loads(Path(config.optimization_configs_path).read_text())
    model_config = tomli.loads(Path(f'{config.base_path}/0-tuning.toml').read_text())
    model_config['space']['optimizer'] = opt_config['optimizer']
    model_config['n_trials'] = 100
    try:
        os.mkdir(f"{config.base_path}/{config.optimizer_name}")
    except FileExistsError:
        pass
    tuning_config_path = f"{config.base_path}/{config.optimizer_name}/0-tuning.toml"
    Path(tuning_config_path).write_text(tomli_w.dumps(model_config))
    run_cmd = f"CUDA_VISIBLE_DEVICES={config.device} python bin/go.py {tuning_config_path} --force"
    print(run_cmd)
    os.system(run_cmd)
    return f"{config.base_path}/{config.optimizer_name}"


def read_results(results_path, postfix=''):
    df = pd.json_normalize([
        json.loads(x.read_text())
        for x in Path(results_path).glob('0-evaluation/*/report.json')
    ])
    mean = df.groupby('config.data.path')['metrics.test.score'].mean().iloc[0]
    std = df.groupby('config.data.path')['metrics.test.score'].std().iloc[0]
    results = {
        f"mean{postfix}": mean,
        f"std{postfix}": std,
    }
    return results


@click.command(help="Run ALSO (or other optimizer) experiment on a tabular dataset.")
@click.option("--device", type=int, default=None, help="The GPU device index to use.")
@click.option("--optimizer_name", default="also", show_default=True, help="The name of the optimizer configuration.")
@click.option("--dataset", default="california", show_default=True, help="The name of the dataset to use.")
@click.option("--results_path", default="results.json", show_default=True, help="The path for results saving")
def main(device, optimizer_name: str, dataset: str, results_path: str):
    """
    This script runs the architecture tuning and evaluation for a given
    optimizer and dataset, then saves the results to a JSON file.
    """
    config = Config(
        optimizer_name=optimizer_name,
        dataset=dataset,
    )
    config.initialize(device)

    print(f"Starting experiment for dataset '{config.dataset}' with optimizer '{config.optimizer_name}'...")
    
    results_path = run_experiment(config)
    result_dict = read_results(results_path)

    print(f"Saving results to '{results_path}'")
    with open(results_path, "w") as f:
        json.dump(result_dict, f, indent=4)

    print("Experiment finished.")


if __name__ == "__main__":
    main()
