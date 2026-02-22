import math
from typing import List, Optional, Tuple

import torch


class AdaptiveBatchTracker:
    """
    Tracks reference gradients and variance terms for adaptive batch sizing.

    At epoch e we accumulate gradients g_t for each step. After the epoch, we
    compute hat_grad = mean_t g_t and hat_norm_sq = ||hat_grad||^2.
    While accumulating, if a reference gradient hat_ref_grad is provided
    (typically from the previous epoch), we track var_sum = Σ ||g_t - hat_ref_grad||^2.
    """

    def __init__(self, device: str):
        self.device = device
        self.reset(None)

    def reset(self, hat_ref_grad: Optional[List[Optional[torch.Tensor]]]) -> None:
        self.hat_ref_grad = hat_ref_grad
        self.grad_sums: Optional[List[Optional[torch.Tensor]]] = None
        self.steps: int = 0
        self.var_sum: float = 0.0

    def _accumulate_grads(self, grads: List[Optional[torch.Tensor]]) -> None:
        if self.grad_sums is None:
            self.grad_sums = [
                g.detach().clone() if g is not None else None for g in grads
            ]
            return
        for idx, g in enumerate(grads):
            if g is None:
                continue
            if self.grad_sums[idx] is None:
                self.grad_sums[idx] = g.detach().clone()
            else:
                self.grad_sums[idx].add_(g.detach())

    def _accumulate_variance(self, grads: List[Optional[torch.Tensor]]) -> None:
        if self.hat_ref_grad is None:
            return
        for g, h in zip(grads, self.hat_ref_grad):
            if g is None or h is None:
                continue
            diff = g.detach() - h
            self.var_sum += torch.sum(diff * diff).item()

    def update(self, grads: List[Optional[torch.Tensor]]) -> None:
        self._accumulate_grads(grads)
        self._accumulate_variance(grads)
        self.steps += 1

    def finalize(
        self,
    ) -> Tuple[Optional[List[Optional[torch.Tensor]]], float, float]:
        if self.steps == 0 or self.grad_sums is None:
            return None, 0.0, 0.0
        hat_grad: List[Optional[torch.Tensor]] = []
        hat_grad_norm_sq = 0.0
        for g_sum in self.grad_sums:
            if g_sum is None:
                hat_grad.append(None)
                continue
            g_hat = g_sum / float(self.steps)
            hat_grad.append(g_hat)
            hat_grad_norm_sq += torch.sum(g_hat * g_hat).item()
        hat_norm_sq = hat_grad_norm_sq
        return hat_grad, hat_norm_sq, self.var_sum


def ensure_preconditioned_strategy_compat(
    adaptive_enabled: bool,
    adaptive_strategy: str,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Validate optimizer compatibility for adaptive strategies."""
    if adaptive_enabled and adaptive_strategy == "variance_ratio_preconditioned":
        if not isinstance(optimizer, torch.optim.AdamW):
            raise ValueError(
                "adaptive_batch_strategy='variance_ratio_preconditioned' requires optimizer='adamw'."
            )
        if any(bool(group.get("amsgrad", False)) for group in optimizer.param_groups):
            raise ValueError(
                "adaptive_batch_strategy='variance_ratio_preconditioned' does not support amsgrad=True."
            )


def compute_adamw_preconditioned_theta_diff_norm_sq(
    optimizer: torch.optim.AdamW,
    optimizer_params: List[torch.Tensor],
    prev_params: Optional[List[torch.Tensor]],
    prev_prev_params: Optional[List[torch.Tensor]],
) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
    """
    Compute || (theta_{e-1} - theta_{e-2}) / (sqrt(v_hat_t) + eps) ||^2 for AdamW.

    Returns:
      - preconditioned norm squared when computable
      - mean(v_t) over all used tensor elements
      - ||v_t||_2 over all used tensor elements
      - valid flag indicating whether any usable parameters contributed
    """
    if prev_params is None or prev_prev_params is None:
        return None, None, None, False
    if len(prev_params) != len(prev_prev_params):
        return None, None, None, False
    if len(optimizer_params) != len(prev_params):
        return None, None, None, False

    index_by_param_id = {id(p): idx for idx, p in enumerate(optimizer_params)}

    norm_sq = 0.0
    v_t_sum = 0.0
    v_t_sq_sum = 0.0
    v_t_count = 0
    used_any = False

    for group in optimizer.param_groups:
        amsgrad = bool(group.get("amsgrad", False))
        if amsgrad:
            continue
        eps = float(group.get("eps", 1e-8))
        beta2 = float(group.get("betas", (0.9, 0.999))[1])
        v_key = "exp_avg_sq"

        for p in group["params"]:
            idx = index_by_param_id.get(id(p))
            if idx is None:
                continue

            theta_prev = prev_params[idx]
            theta_prev_prev = prev_prev_params[idx]
            if theta_prev.shape != theta_prev_prev.shape:
                continue

            state = optimizer.state.get(p, {})
            v_t = state.get(v_key)
            step_t = state.get("step")
            if v_t is None:
                continue
            if step_t is None:
                continue
            if v_t.shape != theta_prev.shape:
                continue

            theta_diff = theta_prev - theta_prev_prev
            if theta_diff.numel() == 0:
                continue

            v_t_cpu = v_t.detach().to(device=theta_diff.device, dtype=theta_diff.dtype)
            if not torch.isfinite(v_t_cpu).all().item():
                continue
            if torch.any(v_t_cpu < 0).item():
                continue

            if torch.is_tensor(step_t):
                step = float(step_t.detach().item())
            else:
                step = float(step_t)
            if not math.isfinite(step) or step <= 0:
                continue

            bias_correction2 = 1.0 - (beta2 ** step)
            if not math.isfinite(bias_correction2) or bias_correction2 <= 0:
                continue

            v_hat_t = v_t_cpu / bias_correction2
            if not torch.isfinite(v_hat_t).all().item():
                continue
            if torch.any(v_hat_t < 0).item():
                continue

            denom = torch.sqrt(v_hat_t) + eps
            if not torch.isfinite(denom).all().item():
                continue

            scaled_diff = theta_diff / denom
            if not torch.isfinite(scaled_diff).all().item():
                continue

            norm_sq += torch.sum(scaled_diff * scaled_diff).item()
            v_t_sum += torch.sum(v_t_cpu).item()
            v_t_sq_sum += torch.sum(v_t_cpu * v_t_cpu).item()
            v_t_count += int(v_t_cpu.numel())
            used_any = True

    if not used_any:
        return None, None, None, False
    if not math.isfinite(norm_sq) or norm_sq <= 0:
        return None, None, None, False

    v_t_mean = (v_t_sum / float(v_t_count)) if v_t_count > 0 else None
    if v_t_mean is not None and not math.isfinite(v_t_mean):
        v_t_mean = None

    v_t_norm = math.sqrt(v_t_sq_sum)
    if not math.isfinite(v_t_norm) or v_t_norm <= 0:
        return None, v_t_mean, None, False

    return norm_sq, v_t_mean, v_t_norm, True
