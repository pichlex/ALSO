# Split Learning on Food101 an Flowers102 Datasets Experiments

This directory contains the implementation of experiments from Section 6.5 of the paper ["Aligning Distributionally Robust Optimization with Practical Deep Learning Needs"](https://arxiv.org/abs/2508.16734).

## Overview

These experiments compare various optimization approaches in Split Learning setup, where one worker joins the training later.

## File Structure

- `main.py`: Main script for running experiments
- `optuna_configs.py`: Configs for tuning
- `utils.py`: Utility functions for reproducibility and evaluation
- `losses.py`: Split Learning related functions
- `run_all.sh`: Script, which runs tuning for all methods
- `run_optuna_all.sh`: Script, which runs evaluation of all methods with params, which were used in paper

## Setup

### Requirements

You can use `environment.yaml` from the parent folder

## Running Experiments

See `run_all.sh`

## Experiment Configurations

The experiments compare different optimization strategies:

1. Standard AdamW
2. ALSO (Adaptive Loss Scaling Optimizer)
3. Spectral Risk loss
4. RECOVER
5. Large Scale DRO Loss
6. DRAGO optimizer

To run specific experiments, you can use one of the following lines:
```
# AdamW
python main.py --optim-mode adamw --num_epochs_food 3 --num_epochs_joint 7 --learning_rate 0.0003 --seeds '[42, 43, 44, 45, 46]'
# ALSO
python main.py --optim-mode also --num_epochs_food 3 --num_epochs_joint 7 --learning_rate 0.0003 --seeds '[42, 43, 44, 45, 46]'
# RECOVER
python main.py --optim-mode recover --num_epochs_food 3 --num_epochs_joint 7 --learning_rate 0.0003 --seeds '[42, 43, 44, 45, 46]'
# DRAGO
python main.py --optim-mode drago --num_epochs_food 3 --num_epochs_joint 7 --learning_rate 0.0003 --seeds '[42, 43, 44, 45, 46]'
# Spectral Risk
python main.py --optim-mode dro --num_epochs_food 3 --num_epochs_joint 7 --learning_rate 0.0003 --seeds '[42, 43, 44, 45, 46]'
# Large Scale DRO
python main.py --optim-mode dro_largescale --num_epochs_food 3 --num_epochs_joint 7 --learning_rate 0.0003 --seeds '[42, 43, 44, 45, 46]'
```

## Results

The script evaluates models using Accuracy@5

## Hyperparameter Tuning

The script uses Optuna for hyperparameter tuning.
