# Adaptive Batch Sizing for ALSO

This notes the stochastic operator view, the norm we use, and how the adaptive batch size is computed and wired into the training loop.

## Operator
- For index `i` drawn with probability `pi_i`, the per-sample operator is  
  `F_i(z) = [∇f_i(θ), 0, …, -f_i(θ)/pi_i, …, 0]^T`.
- For a batch `B` drawn i.i.d. from `pi`, the batched operator is the average  
  `F_B(z) = (1/|B|) Σ_{i in B} F_i(z)`. This is unbiased: `E[F_B] = Σ pi_i F_i`.

## Norm (dual to the Bregman divergence we use)
- Bregman divergence: `V(z1, z2) = 0.5 * ||θ1-θ2||_2^2 + KL[pi1 || pi2]`.
- Corresponding dual norm squared for the operator block `[∑ π_i ∇f_i, -f_1, …, -f_n]`:  
  `||F||_*^2 = 2 * ||∑ π_i ∇f_i||_2^2 + 2 * max_i |f_i|^2` (only L2 for θ-block, only L∞ for the π-block).

## Epoch schedule
1) **Epoch 0 and 1**: fixed batch size `init_batch_size`.
   - Epoch 0: build `hat_F` as the running mean of `F_B`.
   - Epoch 1: accumulate `Σ ||F_B - hat_F_prevprev||_*^2` using `hat_F` from epoch 0.
2) **Epoch 2 and onward**:
   - Before epoch e, compute `B = floor(var_{e-1} / ||hat_F_{e-2}||_*^2)`, clamped to `[min_batch_size, max_batch_size]`.
   - Train epoch e with that `B`, build new `hat_F_e`, and accumulate variance vs `hat_F_{e-2}`.
   - Keep sliding: variance from the last epoch, `hat_F` norm from two epochs back.

## Implementation highlights
- **Sampling**: `ALSO.select_batch(batch_size, generator)` draws indices i.i.d. by current `pi` (with replacement). The tail-mass cutoff logic is removed.
- **Unbiased updates**: optimizer returns `(loss_mean, raw_losses, pi_selected)`; `pi` update uses `-losses / pi_selected`, and the closure averages `(pi_selected * losses / batch_size)` for θ-gradients.
- **Tracking** (`utils.AdaptiveBatchTracker`):
  - Builds `hat_F` as running mean of flattened θ-gradients and the `-f_i/pi_i` contribution (batch-averaged).
  - Accumulates `Σ ||F_B - hat_F_ref||_*^2` with `hat_F_ref` taken from two epochs back.
  - Stores per-epoch `hat_norm_sq` and `var_sum` only (no per-iteration storage).
- **Wiring**: `train()` manages `hat_history`, `var_history`, computes `current_batch_size` each epoch, and logs metrics. Adaptive mode sets `dynamic_batch=True`.

## Config keys (set in your JSON)
```json
{
  "adaptive_batching": true,
  "init_batch_size": 64,   // default: batch_size
  "min_batch_size": 10,    // default 10
  "max_batch_size": 512    // default 512
}
```
No other changes are needed; `batch_size` will be overridden each epoch in adaptive mode.

## Logging
- Per epoch (when `report_to = "mlflow"`): `adaptive_batch_size`, `hat_norm_sq`, `var_sum_sq`.

## Quick start
1) In your experiment config, enable `"adaptive_batching": true` and choose `init_batch_size` if needed.
2) Run as usual (e.g., `python main.py --config-path ...`). First two epochs stay fixed; starting epoch 2 the batch size adapts by the rule above.
3) Inspect mlflow for the chosen batch sizes and norms to confirm the schedule is behaving as expected.
