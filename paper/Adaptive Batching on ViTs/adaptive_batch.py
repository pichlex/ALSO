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
