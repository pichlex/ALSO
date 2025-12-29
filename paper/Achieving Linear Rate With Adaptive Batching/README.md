# Achieving Linear Rate With Adaptive Batching

Experiments on CIFAR10 and SVHN with two modes:

- Baseline: fixed batch size training.
- Adaptive: three-phase adaptive batching with gradient-only statistics.

## Key details
- `hat_norm_sq = ||hat_grad||^2`, `var_sum = ||g_t - hat_grad||^2` (no loss terms).
- Adaptive batching kicks in from epoch `epoch_start_ab` (default 2): epoch 1 collects grads, epoch 2 accumulates variance, epoch 3+ uses `floor(var_sum / hat_norm_sq)` clamped to `[adaptive_batch_min, adaptive_batch_max]`, with optional EMA smoothing `adaptive_batch_beta`.
- CIFAR10/SVHN, 10 classes, no imbalance or class aggregation. Augmentation optional via `augment` flag (default on).
- MLflow logging (default experiment `Achieving Linear Rate`): batch size, numerator/denominator, train/val/test loss, precision/recall/f1/accuracy.
- Hyperparameter tuning (fixed batch only) with Optuna: 5 epochs, 100 trials by default; tuned params saved under `tuned_params/{dataset}/{tune_name}.json` and can be reused.

## Usage Example
```bash
python main.py --config configs/cifar10_baseline.json          # train with config
python main.py --config configs/cifar10_baseline.json --tune   # run tuning only
python main.py --config configs/cifar10_adaptive.json --use-tuned  # load tuned params before training
```

## Config fields (JSON)
Common:
- `dataset`: `cifar10` | `svhn`
- `model`: `resnet18`
- `batch_size`, `epochs`, `optimizer` (`sgd` or `adamw`), `lr`, `momentum`, `nesterov`, `weight_decay`
- `device` (optional): e.g. `"cuda"`, `"cuda:0"`, `"cpu"`; defaults to auto-detect CUDA
- `augment` (bool, default true), `num_workers`, `seed`
- `mlflow_experiment`, `report_to`

Adaptive:
- `adaptive_batch` (bool), `adaptive_batch_beta` (default 0.0), `adaptive_batch_min` (10), `adaptive_batch_max` (1024), `epoch_start_ab` (2)

Tuning:
- `tune_runs` (100), `n_epoches_tune` (5), `tune_name`, `use_old_tune_params` (reuse saved), `use_tuned_params` (when loading)
- Tuned params saved to `tuned_params/{dataset}/{tune_name}.json`; for SGD the path includes optimizer signature: `tuned_params/{dataset}/sgd_mom-{on/off}_nest-{on/off}_wd-{on/off}/{tune_name}.json`
- Tuning objective uses validation accuracy (`best_val_acc`).
