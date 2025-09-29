# Tabular DL Experiments

This directory contains the implementation of experiments from Section 6.2 of the paper ["Aligning Distributionally Robust Optimization with Practical Deep Learning Needs"](https://arxiv.org/abs/2508.16734).

## Overview

These experiments compare various optimization approaches in training DL models on tabular data. The main goal is to evaluate the effectiveness of different optimization techniques in handling challanging, diverse real-world heterogeneities, which presence in tabular data: one of the most popular domain in practical machine learning.

## File Structure

- The directories `bin`, `lib`, `tools` are the same as in [TabM repo](https://github.com/yandex-research/tabm?)
- `exp` is directory, contaning configs for optimization methods, models and datasets
- `main.py`: the file, which is used to automatically run a single experiment

## Setup

### Requirements

You can use `environment.yaml` from the parent folder

### Data Preparation

**The same instruction as in TabM**. The data consists of two parts.

**Part 1.** Go to the `Tabular DL` folder and run:

```
mkdir local
wget https://huggingface.co/datasets/rototoHF/tabm-data/resolve/main/data.tar -O local/tabm-data.tar.gz
mkdir data
tar -xvf local/tabm-data.tar.gz -C data
```

**Part 2.** Create the `local` directory
and download the [TabReD](https://github.com/yandex-research/tabred) benchmark to `local/tabred`
(you will need an account on Kaggle).
Then, run:

```
python tools/prepare_tabred.py local/tabred data
```

## Running Experiments

To run one experiment one can use the following command:

```
python main.py --device 0 --optimizer_name also --dataset california --results_path results.json
```

Additional command-line options:
- `--device <id>`: GPU device ID (None for CPU)
- `--optimizer_name`: Name of the optimizer (see, **Experiment Configurations** section)
- `--dataset`: Name of the dataset (see, **Experiment Configurations** section)
- `--results_path`: the path to the `.json` file, where to save the results

## Experiment Configurations

The experiments compare different optimization strategies:

1. Standard AdamW (`--optimizer_name adamw`)
3. Adam with static weights (`--optimizer_name static_weights_adamw`)
4. ALSO (Adaptive Loss Scaling Optimizer) (`--optimizer_name also`)
5. Spectral Risk loss (`--optimizer_name dro`)
6. RECOVER (`--optimizer_name recover`)
7. Large Scale DRO Loss (`--optimizer_name large_scale_dro_chi_square` or `--optimizer_name large_scale_dro_cvar`)
8. DRAGO optimizer (`--optimizer_name drago`)

On several standard datasets: `adult`, `black-friday`, `california`, `churn`, `higgs-small`, `microsoft`, `cooking-time`, `delivery-eta`, `ecom-offers`, `homecredit-default`, `homesite-insurance`, `maps-routing`, `sberbank-housing`, `weather`.

## Results
The results are presented in `--results_path` file (i.e. `results.json`) as json file with `mean` and `std` fiels computed over 15 splits. The metric is either RMSE for regression or ROC AUC for classification

## Hyperparameter Tuning

The script uses Optuna for hyperparameter tuning.
