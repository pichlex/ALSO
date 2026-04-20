import math
from typing import Iterable, List, Optional, Sequence, Tuple

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
            self.grad_sums = clone_optional_tensors(grads)
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


def clone_optional_tensors(
    tensors: Sequence[Optional[torch.Tensor]],
) -> List[Optional[torch.Tensor]]:
    return [tensor.detach().clone() if tensor is not None else None for tensor in tensors]


def compute_signal_reference_variance(
    signal: Sequence[Optional[torch.Tensor]],
    hat_ref_grad: Optional[Sequence[Optional[torch.Tensor]]],
) -> float:
    if hat_ref_grad is None:
        return 0.0

    variance = 0.0
    for current, reference in zip(signal, hat_ref_grad):
        if current is None or reference is None:
            continue
        diff = current.detach() - reference
        variance += torch.sum(diff * diff).item()
    return variance


def compute_step_theta_diff_norm_sq(
    previous_params: Sequence[torch.Tensor],
    current_params: Sequence[torch.Tensor],
) -> float:
    theta_diff_norm_sq = 0.0
    for previous, current in zip(previous_params, current_params):
        diff = current.detach().cpu() - previous
        theta_diff_norm_sq += torch.sum(diff * diff).item()
    return theta_diff_norm_sq


def _compute_sgd_step_norm_sq(optimizer: torch.optim.SGD) -> float:
    step_norm_sq = 0.0
    for group in optimizer.param_groups:
        lr = float(group.get("lr", 0.0))
        momentum = float(group.get("momentum", 0.0))
        dampening = float(group.get("dampening", 0.0))
        weight_decay = float(group.get("weight_decay", 0.0))
        nesterov = bool(group.get("nesterov", False))
        if lr == 0.0:
            continue

        for p in group["params"]:
            grad = p.grad
            if grad is None:
                continue
            grad_to_use = grad.detach()
            if weight_decay != 0.0:
                grad_to_use = grad_to_use.add(p.detach(), alpha=weight_decay)

            if momentum != 0.0:
                state = optimizer.state.get(p, {})
                momentum_buffer = state.get("momentum_buffer")
                if momentum_buffer is None:
                    buf = grad_to_use.clone()
                else:
                    buf = momentum_buffer.detach().mul(momentum).add(
                        grad_to_use, alpha=1.0 - dampening
                    )
                if nesterov:
                    grad_to_use = grad_to_use.add(buf, alpha=momentum)
                else:
                    grad_to_use = buf

            step_norm_sq += (lr * lr) * torch.sum(grad_to_use * grad_to_use).item()
    return step_norm_sq


def _compute_adamw_step_norm_sq(optimizer: torch.optim.AdamW) -> float:
    step_norm_sq = 0.0
    for group in optimizer.param_groups:
        lr = float(group.get("lr", 0.0))
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        beta1 = float(beta1)
        beta2 = float(beta2)
        eps = float(group.get("eps", 1e-8))
        weight_decay = float(group.get("weight_decay", 0.0))
        amsgrad = bool(group.get("amsgrad", False))
        if amsgrad:
            raise ValueError(
                "adaptive_batch_strategy='variance_ratio_iter' does not support AdamW with amsgrad=True."
            )
        if lr == 0.0:
            continue

        for p in group["params"]:
            grad = p.grad
            if grad is None:
                continue
            state = optimizer.state.get(p, {})
            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            step_t = state.get("step", 0)

            if exp_avg is None:
                exp_avg = torch.zeros_like(p, memory_format=torch.preserve_format)
            else:
                exp_avg = exp_avg.detach()
            if exp_avg_sq is None:
                exp_avg_sq = torch.zeros_like(p, memory_format=torch.preserve_format)
            else:
                exp_avg_sq = exp_avg_sq.detach()

            if torch.is_tensor(step_t):
                step = float(step_t.detach().item())
            else:
                step = float(step_t)
            next_step = step + 1.0

            grad_to_use = grad.detach()
            exp_avg_new = exp_avg.mul(beta1).add(grad_to_use, alpha=1.0 - beta1)
            exp_avg_sq_new = exp_avg_sq.mul(beta2).addcmul(
                grad_to_use, grad_to_use, value=1.0 - beta2
            )

            bias_correction1 = 1.0 - beta1**next_step
            bias_correction2 = 1.0 - beta2**next_step
            if bias_correction1 <= 0.0 or bias_correction2 <= 0.0:
                continue

            step_size = lr / bias_correction1
            denom = exp_avg_sq_new.sqrt().div(math.sqrt(bias_correction2)).add_(eps)
            adaptive_update = exp_avg_new / denom
            total_update = adaptive_update.mul(-step_size)
            if weight_decay != 0.0:
                total_update = total_update.add(p.detach(), alpha=-lr * weight_decay)
            step_norm_sq += torch.sum(total_update * total_update).item()
    return step_norm_sq


def compute_optimizer_step_norm_sq(
    optimizer: torch.optim.Optimizer,
    optimizer_name: str,
    optimizer_params: Sequence[torch.Tensor],
) -> float:
    del optimizer_params
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "sgd":
        if not isinstance(optimizer, torch.optim.SGD):
            raise ValueError("optimizer_name='sgd' requires torch.optim.SGD.")
        return _compute_sgd_step_norm_sq(optimizer)
    if optimizer_name == "adamw":
        if not isinstance(optimizer, torch.optim.AdamW):
            raise ValueError("optimizer_name='adamw' requires torch.optim.AdamW.")
        return _compute_adamw_step_norm_sq(optimizer)
    raise ValueError(
        "adaptive_batch_strategy='variance_ratio_iter' only supports exact step norms for optimizer='sgd' or 'adamw'."
    )


def compute_variance_ratio_iter_batch_size(
    numerator: float,
    denominator: float,
    batch_size_multiplier: float,
    batch_size_min: int,
    batch_size_max: int,
) -> Tuple[int, Optional[float], Optional[float]]:
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or numerator < 0.0
        or denominator <= 0.0
    ):
        return int(batch_size_max), None, None

    ratio_raw = math.sqrt(numerator / denominator)
    raw_batch = batch_size_multiplier * ratio_raw
    if not math.isfinite(raw_batch):
        return int(batch_size_max), ratio_raw, None

    batch_size = int(math.floor(raw_batch))
    batch_size = max(int(batch_size_min), min(int(batch_size_max), batch_size))
    return batch_size, ratio_raw, raw_batch


class DynamicFixedOrderBatchSampler(torch.utils.data.Sampler[List[int]]):
    """
    Fixed-order sampler with per-step batch-size updates for ABOBA variants.

    Updates apply to the next batch in the same epoch, and incomplete tails are
    dropped to match the existing BatchSampler(drop_last=True) behaviour.
    """

    def __init__(self, fixed_indices: Iterable[int], initial_batch_size: int):
        self.indices = [int(idx) for idx in fixed_indices]
        self.batch_size = max(1, int(initial_batch_size))
        self.id = 0

    def __iter__(self):
        self.id = 0
        while self.id + self.batch_size <= len(self.indices):
            start_id = self.id
            self.id += self.batch_size
            yield self.indices[start_id : self.id]

    def __len__(self) -> int:
        return len(self.indices) // max(1, int(self.batch_size))

    def set_batch_size(self, new_batch_size: int) -> None:
        self.batch_size = max(1, int(new_batch_size))

    def update_batch_size(self, new_batch_size: int) -> None:
        self.set_batch_size(new_batch_size)


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


def compute_adamw_adaptive_update_signal(
    optimizer: torch.optim.AdamW,
    optimizer_params: List[torch.Tensor],
) -> List[Optional[torch.Tensor]]:
    """
    Build a parameter-ordered list of AdamW adaptive-part update signals u_t.

    For each parameter p this returns:
      u_t = m_hat_t / (sqrt(v_hat_t) + eps)
    when available and finite; otherwise None.
    """
    index_by_param_id = {id(p): idx for idx, p in enumerate(optimizer_params)}
    signal: List[Optional[torch.Tensor]] = [None] * len(optimizer_params)

    for group in optimizer.param_groups:
        beta1 = float(group.get("betas", (0.9, 0.999))[0])
        beta2 = float(group.get("betas", (0.9, 0.999))[1])
        eps = float(group.get("eps", 1e-8))
        for p in group["params"]:
            idx = index_by_param_id.get(id(p))
            if idx is None:
                continue

            state = optimizer.state.get(p, {})
            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            step_t = state.get("step")
            if exp_avg is None or exp_avg_sq is None or step_t is None:
                continue
            if exp_avg.shape != p.shape or exp_avg_sq.shape != p.shape:
                continue

            if torch.is_tensor(step_t):
                step = float(step_t.detach().item())
            else:
                step = float(step_t)
            if not math.isfinite(step) or step <= 0:
                continue

            bias_correction1 = 1.0 - (beta1 ** step)
            if not math.isfinite(bias_correction1) or bias_correction1 <= 0:
                continue

            bias_correction2 = 1.0 - (beta2 ** step)
            if not math.isfinite(bias_correction2) or bias_correction2 <= 0:
                continue

            m_t = exp_avg.detach().to(device=p.device, dtype=p.dtype)
            if not torch.isfinite(m_t).all().item():
                continue

            v_t = exp_avg_sq.detach().to(device=p.device, dtype=p.dtype)
            if not torch.isfinite(v_t).all().item():
                continue
            if torch.any(v_t < 0).item():
                continue

            m_hat = m_t / bias_correction1
            if not torch.isfinite(m_hat).all().item():
                continue

            v_hat = v_t / bias_correction2
            if not torch.isfinite(v_hat).all().item():
                continue
            if torch.any(v_hat < 0).item():
                continue

            denom = torch.sqrt(v_hat) + eps
            if not torch.isfinite(denom).all().item():
                continue

            u_t = m_hat / denom
            if not torch.isfinite(u_t).all().item():
                continue
            signal[idx] = u_t

    return signal
