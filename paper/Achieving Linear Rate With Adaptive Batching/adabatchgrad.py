import math
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


class DynamicBatchSampler(torch.utils.data.Sampler[List[int]]):
    """
    Sampler that allows increasing batch size on the fly.

    Updates to the batch size affect the remainder of the current epoch. This
    mirrors the behaviour in the AdaBatchGrad reference implementation.
    """

    def __init__(
        self,
        data_source: torch.utils.data.Dataset,
        initial_batch_size: int,
        generator: Optional[torch.Generator] = None,
    ):
        self.data_source = data_source
        self.batch_size = max(1, int(initial_batch_size))
        self.generator = generator
        self.id = 0
        self.state = {"batch_size": self.batch_size, "new_batch_size_list": []}

    def __iter__(self):
        self.id = 0
        indices = torch.randperm(len(self.data_source), generator=self.generator)
        while self.id < len(self.data_source):
            start_id = self.id
            self.id += self.batch_size
            yield indices[start_id : self.id].tolist()

    def __len__(self) -> int:
        return math.ceil(len(self.data_source) / float(self.batch_size))

    def update_batch_size(self, new_batch_size: int) -> None:
        new_bs = int(new_batch_size)
        if new_bs <= self.batch_size:
            return
        # Rewind the cursor so the new batch starts at the previous boundary.
        self.id -= self.batch_size
        self.batch_size = new_bs
        self.state["batch_size"] = new_bs

    def reset_epoch_state(self) -> None:
        self.state["new_batch_size_list"] = []


def random_increase(
    grad_list,
    grad_full_: Optional[torch.Tensor] = None,
    mul_scalar: float = 1.01,
    prob_new: float = 0.005,
    nu: Optional[float] = None,
) -> int:
    if prob_new <= 0:
        raise ValueError(f"Invalid probability value for new method: prob_new = {prob_new}")
    batch_size = len(grad_list)
    if torch.rand(1).item() < prob_new:
        batch_size = math.ceil(mul_scalar * batch_size)
    return int(batch_size)


def inner_ortho_test_nn(
    param_grad_per_sample: Dict[str, torch.Tensor],
    grad_full_: Optional[torch.Tensor] = None,
    theta: float = 0.9,
    prob_new: Optional[float] = None,
    nu: float = 5.84,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(iter(param_grad_per_sample.values())).device
    len_S = next(iter(param_grad_per_sample.values())).shape[0]

    total_grad_per_sample_on_grad_mean = torch.zeros(len_S, device=device)
    total_inner_diff = torch.zeros(len_S, device=device)
    total_norm_sq = 0.0
    ortho_numerator = 0.0

    for grad_per_sample in param_grad_per_sample.values():
        grad_mean = grad_per_sample.mean(dim=0)
        grad_mean_norm_sq = torch.norm(grad_mean).item() ** 2
        total_norm_sq += grad_mean_norm_sq

        dims = len(grad_per_sample.shape) - 1
        grad_per_sample_on_grad_mean = torch.tensordot(grad_per_sample, grad_mean, dims=dims)
        total_grad_per_sample_on_grad_mean += grad_per_sample_on_grad_mean
        total_inner_diff += grad_per_sample_on_grad_mean - grad_mean_norm_sq

    inner_numerator = torch.sum(total_inner_diff**2)

    for grad_per_sample in param_grad_per_sample.values():
        grad_mean = grad_per_sample.mean(dim=0)
        fst_term = grad_per_sample
        snd_term = grad_mean * total_grad_per_sample_on_grad_mean.view(
            -1, *([1] * len(grad_mean.shape))
        )
        snd_term /= total_norm_sq + 1e-12

        ortho_numerator += torch.sum((fst_term - snd_term) ** 2)

    new_Sk_size_inner = torch.ceil(
        inner_numerator / ((len_S - 1) * theta**2 * (total_norm_sq + 1e-12) ** 2)
    )
    new_Sk_size_ortho = torch.ceil(
        ortho_numerator / ((len_S - 1) * nu**2 * (total_norm_sq + 1e-12))
    )

    return new_Sk_size_inner, new_Sk_size_ortho


def choose_batch_test_func(batch_test_name: Optional[str]):
    if batch_test_name is not None:
        if batch_test_name == "random_increase":
            batch_test_func = random_increase
        elif batch_test_name == "inner_ortho_nn":
            batch_test_func = inner_ortho_test_nn
        else:
            raise ValueError(f"Unknown batch test: {batch_test_name}")
    else:
        batch_test_func = None
    return batch_test_func


def calculate_batch_size(
    k: int,
    state: Dict[str, Iterable],
    grad_list,
    data_size: int,
    batch_test: Optional[str],
    grad_full,
    theta: float,
    prob_new: Optional[float],
    nu: Optional[float],
) -> int:
    batch_test_func = choose_batch_test_func(batch_test)
    if batch_test_func is None:
        return state["batch_size"]

    if state["batch_size"] < data_size:
        if batch_test == "inner_ortho_nn":
            new_bs_inner, new_bs_ortho = batch_test_func(
                grad_list, grad_full, theta, prob_new, nu
            )
            new_batch_size = int(max(new_bs_inner.item(), new_bs_ortho.item()))
            state["inner_batch_size"] = int(new_bs_inner.item())
            state["ortho_batch_size"] = int(new_bs_ortho.item())
            state["new_batch_size_list"].append(new_batch_size)

            if len(state["new_batch_size_list"]) == k:
                min_bs = min(state["new_batch_size_list"])
                if state["batch_size"] < min_bs:
                    new_batch_size = min_bs
                    state["batch_size"] = new_batch_size
                state["new_batch_size_list"] = []
            else:
                new_batch_size = state["batch_size"]
        else:
            new_batch_size = int(
                batch_test_func(grad_list, grad_full, theta, prob_new, nu)
            )

        if new_batch_size >= data_size:
            new_batch_size = data_size
    else:
        new_batch_size = data_size
        state["inner_batch_size"] = data_size
        state["ortho_batch_size"] = data_size
        state["new_batch_size_list"].append(new_batch_size)

    return int(new_batch_size)


def _ce_loss(params: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], model, X, y):
    params_and_buffers = {**params, **buffers}
    if X.dim() == 3:
        X = X.unsqueeze(0)
    if y.dim() == 0:
        y = y.unsqueeze(0)
    logits = functional_call(model, params_and_buffers, (X,))
    return F.cross_entropy(logits, y)


_ce_grad = grad(_ce_loss)
_ce_grad_per_sample = vmap(_ce_grad, in_dims=(None, None, None, 0, 0))


def per_sample_cross_entropy_grads(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """
    Compute per-sample gradients of cross-entropy loss with respect to model parameters.
    """
    params = {name: p for name, p in model.named_parameters()}
    buffers = {name: b for name, b in model.named_buffers()}
    return _ce_grad_per_sample(params, buffers, model, X, y)


class AdaBatchGrad(torch.optim.Optimizer):
    """
    AdaBatchGrad optimizer (batch-size aware AdaGrad).

    This mirrors the logic from `adabatchgrad_nn.py`: a single global step size
    is computed from the sum of squared gradient norms accumulated over steps.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        alpha: float = 1.0,
        beta: float = 1.0,
        power_eps: float = 0.0,
        weight_decay: float = 0.0,
    ):
        if beta < 0:
            raise ValueError(f"Invalid learning rate denominator: beta = {beta}")

        defaults = dict(alpha=alpha, beta=beta, power_eps=power_eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        for group in self.param_groups:
            params_list = group["params"]
            if not params_list:
                continue
            p = params_list[0]
            state = self.state[p]
            state["gradient_square_sum"] = torch.tensor(0.0, device=p.device)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            alpha = group["alpha"]
            beta = group["beta"]
            power_eps = group["power_eps"]
            weight_decay = group["weight_decay"]

            params = group["params"]
            if len(params) == 0:
                continue
            # Use the first param to access shared state.
            p0 = params[0]
            state = self.state[p0]

            grad_sq_sum = state.get("gradient_square_sum", torch.tensor(0.0, device=p0.device))
            grad_norm_sq = torch.tensor(0.0, device=p0.device)

            for p in params:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)
                grad_norm_sq = grad_norm_sq + grad.pow(2).sum()

            # Accumulate and compute step size (shared across params).
            grad_sq_sum = grad_sq_sum + state.get("prev_grad_square", torch.tensor(0.0, device=p0.device))
            state["prev_grad_square"] = grad_norm_sq
            h_k = alpha / torch.pow(beta + grad_sq_sum, 0.5 + power_eps)
            state["gradient_square_sum"] = grad_sq_sum
            state["step_size"] = h_k

            for p in params:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)
                p.add_(grad, alpha=-h_k)

        return loss
