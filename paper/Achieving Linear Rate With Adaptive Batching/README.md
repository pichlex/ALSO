# Achieving Linear Rate With Adaptive Batching

Experiments on CIFAR10 and SVHN with two modes:

- Baseline: fixed batch size training.
- Adaptive: three-phase adaptive batching with gradient-only statistics.

## Key details
- `hat_norm_sq = ||hat_grad||^2`, `var_sum = ||g_t - hat_grad||^2` (no loss terms).
- Adaptive batching kicks in from epoch `epoch_start_ab` (default 2): epoch 1 collects grads, epoch 2 accumulates variance, epoch 3+ sets batch size from ratio-based rules (all clamped to `[adaptive_batch_min, adaptive_batch_max]`, with optional smoothing `adaptive_batch_beta`).
- `variance_ratio`: `batch = floor(batch_size_multiplier * sqrt(var_used / theta_diff_norm_sq))`, where `theta_diff_norm_sq = ||theta_{e-1} - theta_{e-2}||^2`.
- `variance_ratio_preconditioned` (AdamW only): `batch = floor(batch_size_multiplier * sqrt(var_used_update / theta_diff_norm_sq))`, where `var_used_update` is variance accumulated from `u_t = m_hat_t / (sqrt(v_hat_t) + eps)` with `m_hat_t = m_t / (1 - beta1^t)` and `v_hat_t = v_t / (1 - beta2^t)`, and `theta_diff_norm_sq = ||theta_{e-1} - theta_{e-2}||^2`; AMSGrad is not supported for this strategy.
- `variance_ratio_iter`: ABOBA step-level variant. `hat_ref_grad` is still computed as the epoch mean signal, but the next batch is chosen from the last available step via `batch_{t+1} = floor(batch_size_multiplier * sqrt(var_t / ||theta_t - theta_{t-1}||^2))`, where `var_t = ||s_t - hat_ref_grad_epoch||^2`; invalid or infinite ratios clamp to `adaptive_batch_max`, the first adaptive batch of a new epoch reuses the previous epoch's last-step statistics instead of resetting to the config batch size, and the step denominator is computed from the exact optimizer update norm without CPU parameter snapshots.
- `cifar10`, `cifar100`, and `svhn` with `resnet18`/`resnet34` use a CIFAR-style ResNet stem automatically: `conv1` becomes `3x3, stride=1, padding=1`, `maxpool` is removed, and these runs use `weights=None` instead of ImageNet pretrained weights.
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
- `dataset`: `cifar10` | `cifar100` | `svhn`
- `model`: `resnet18` | `resnet34`
- `batch_size`, `epochs`, `optimizer` (`sgd` or `adamw`), `lr`, `momentum`, `nesterov`, `weight_decay`
- `device` (optional): e.g. `"cuda"`, `"cuda:0"`, `"cpu"`; defaults to auto-detect CUDA
- `augment` (bool, default true), `num_workers`, `seed`
- `mlflow_experiment`, `report_to`

Adaptive:
- `adaptive_batch` (bool), `adaptive_batch_beta` (default 0.0), `batch_size_multiplier` (1.0), `adaptive_batch_min` (10), `adaptive_batch_max` (65536), `epoch_start_ab` (2)
- `adaptive_batch_strategy`:  
  - `variance_ratio` (default, legacy three-phase)  
  - `variance_ratio_sq` (legacy squared-ratio scaling)  
  - `variance_ratio_preconditioned` (AdamW adaptive-update numerator with plain parameter-diff denominator)  
  - `variance_ratio_iter` (ABOBA step-level variant: epoch-level `hat_ref_grad`, per-step `var_t = ||s_t - hat_ref_grad_epoch||^2`, denominator from neighbouring parameter updates `||theta_t - theta_{t-1}||^2`, fixed order, no `adaptive_batch_beta` smoothing)  
  - `adabatchgrad` (per-step sizing via AdaBatchGrad tests). For `adabatchgrad`, knobs mirror the reference defaults: `adabatchgrad_batch_test` (`random_increase` or `inner_ortho_nn`), `adabatchgrad_theta` (0.1), `adabatchgrad_nu` (0.1), `adabatchgrad_prob_new` (0.005), `adabatchgrad_k` (5), `adabatchgrad_alpha` (1.0), `adabatchgrad_beta` (1.0), `adabatchgrad_power_eps` (0.0).  
  - `seesaw` (epoch-based cosine-threshold schedule): detect cosine LR crossings at `lr0 / alpha^k`; on each crossing multiply batch size by `alpha` (clamped by `adaptive_batch_max`) and scale LR by `1/alpha` for SGD or `1/sqrt(alpha)` for Adam/AdamW. Config knob: `seesaw_alpha` (default 2.0). No warmup support is expected.

Tuning:
- `tune_runs` (100), `n_epoches_tune` (5), `tune_name`, `use_old_tune_params` (reuse saved), `use_tuned_params` (when loading)
- Tuned params saved to `tuned_params/{dataset}/{tune_name}.json`; for SGD the path includes optimizer signature: `tuned_params/{dataset}/sgd_mom-{on/off}_nest-{on/off}_wd-{on/off}/{tune_name}.json`
- Tuning objective uses validation accuracy (`best_val_acc`).
