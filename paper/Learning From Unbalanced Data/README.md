# Unbalanced CIFAR10 Binary Classification Experiments

This directory contains the implementation of experiments from Section 6.1 of the paper ["Aligning Distributionally Robust Optimization with Practical Deep Learning Needs"](https://arxiv.org/abs/2508.16734).

## Overview

These experiments compare various optimization approaches on an artificially unbalanced CIFAR10 dataset. The main goal is to evaluate the effectiveness of different optimization techniques in handling class imbalance, which is a common challenge in real-world machine learning applications.

## File Structure

- `main.py`: Main script for running experiments
- `utils.py`: Utility functions for dataset creation, training, and evaluation
- `problems.py`: Problem definition and configuration for the unbalanced CIFAR10 task

## Setup

### Requirements

You can use `environment.yaml` from the parent folder

### Configuration

For configuration you can use `config.json` file with the following structure:

```json
{
    "model": "resnet18",
    "unbalance_coefs": [5, 10],
    "batch_size": 256,
    "n_epoches": 30,
    "n_epoches_tune": 5,
    "eval_runs": 3,
    "tune_runs": 20,
}
```

The main configuration parameters are:
- `model`: Model architecture to use (currently only "resnet18" is supported)
- `unbalance_coefs`: List of imbalance coefficients to test (higher means more imbalance)
- `batch_size`: Training batch size
- `n_epoches`: Number of training epochs for evaluation runs
- `n_epoches_tune`: Number of training epochs for hyperparameter tuning
- `eval_runs`: Number of evaluation runs with different random seeds
- `tune_runs`: Number of hyperparameter tuning trials

## Running Experiments

To run experiments:

```bash
python unbalanced_dataset.py --config config.json
```

Additional command-line options:
- `--device <id>`: GPU device ID (None for CPU)
- `--balanced_test`: Use balanced test set regardless of training imbalance
- `--tune`: Force hyperparameter tuning even for repeated runs
- `--use_old_tune_params`: Use previously tuned parameters if available
- `--augment`: Use data augmentation during training

## Experiment Configurations

The experiments compare different optimization strategies:

1. Standard Adam
2. Adam with upsampling
3. Adam with static weights
4. ALSO (Adaptive Loss Scaling Optimizer)
5. Spectral Risk loss
6. RECOVER
7. Large Scale DRO Loss
8. DRAGO optimizer

To run specific experiments, modify the `experiment_list` in `unbalanced_dataset.py`.

## Results

The script evaluates models using multiple metrics:
- F1 score
- Precision
- Recall
- Training time

Results are printed for each experiment configuration.

## Hyperparameter Tuning

The script uses Optuna for hyperparameter tuning. Tuned parameters are saved to disk in the `tuned_params` directory (which should exist) and can be reused for future runs with the same configuration.
