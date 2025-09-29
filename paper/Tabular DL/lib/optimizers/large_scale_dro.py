"""
Adopted from https://github.com/daniellevy/fast-dro/blob/main/robust_losses.py


PyTorch modules for computing robust losses with
for (KL-regularized) CVaR, constrained-chi^2 and penalized-chi^2
uncertainty sets.
Includes losses appropriate for our porposed batch and MLMC gradient estimators
as well as losses for the dual-SGM and primal-dual methods.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from scipy import optimize, interpolate


def project_to_cs_ball(v, rho):
    """Numpy/Scipy projection to chi-square ball of radius rho"""
    n = len(v)
    def cs_div(p):
        return 0.5 * np.mean((n * p - 1)**2)

    # first, check if a simplex projection is within the chi-square ball
    target_simplex = lambda eta: np.sum(np.maximum(v - eta, 0)) - 1.0
    eta_min_simplex = v.min() - 1 / n
    eta_max_simplex = v.max()
    eta_simplex = optimize.brentq(
        target_simplex, eta_min_simplex, eta_max_simplex)
    p_candidate = np.maximum(v - eta_simplex, 0)
    if cs_div(p_candidate) <= rho:
        return p_candidate

    # second, compute a chi-square best response
    def target_cs(eta, return_p=False):
        p = np.maximum(v - eta, 0)
        if p.sum() == 0.0:
            p[np.argmax(v)] = 1.0
        else:
            p /= p.sum()
        err = cs_div(p) - rho
        return p if return_p else err
    eta_max_cs = v.max()
    eta_min_cs = v.min()
    if target_cs(eta_max_cs) <= 0:
        return target_cs(eta_max_cs, return_p=True)
    while target_cs(eta_min_cs) > 0.0:  # find left interval edge for bisection
        eta_min_cs = 2 * eta_min_cs - eta_max_cs
    eta_cs = optimize.brentq(
        target_cs, eta_min_cs, eta_max_cs)
    p_candidate = target_cs(eta_cs, return_p=True)
    assert np.abs(cs_div(p_candidate) - rho) < rho * 1e-2
    return p_candidate


def project_to_cvar_ball(w, alpha):
    if alpha == 1.0:
        return np.ones(n) / n
    n = len(w)
    k = alpha * n
    w = w + 1e-12  # slight padding to avoid numerical issues
    logw = np.log(w) - np.log(w.max())  # offset so maximum value is 0.0

    def target_cvar(nu, return_p=False):
        w_ = np.exp(np.minimum(logw, nu))
        p = w_ / w_.sum()
        return p if return_p else p.max() - 1 / k

    if target_cvar(0.0) <= 0.0:
        p = w / w.sum()
    else:
        nu_max = 0.0
        nu_min = logw.min()
        nu = optimize.brentq(
            target_cvar, nu_min, nu_max)
        p = target_cvar(nu, return_p=True)
    if p.max() > 1 / k * 1.01:
        logging.warning(f'project_to_cvar_ball: Maximum element of p'
                        f' is {p.max()}; supposed to be {1/k}')
    if np.abs(p.sum() - 1.0) > 1e-2:
        logging.warning(f'project_to_cvar_ball: Elements of p sum to'
                        f'{p.sum()} instead of 1.0')
    return p


GEOMETRIES = ('cvar', 'chi-square')
MIN_REL_DIFFERENCE = 1e-5


def chi_square_value(p, v, reg):
    """Returns <p, v> - reg * chi^2(p, uniform) for Torch tensors"""
    m = p.shape[0]

    with torch.no_grad():
        chi2 = (0.5 / m) * reg * (torch.norm(m * p - torch.ones(m, device=p.device), p=2) ** 2)

    return torch.dot(p, v) - chi2


def cvar_value(p, v, reg):
    """Returns <p, v> - reg * KL(p, uniform) for Torch tensors"""
    m = p.shape[0]

    with torch.no_grad():
        idx = torch.nonzero(p)  # where is annoyingly backwards incompatible
        kl = np.log(m) + (p[idx] * torch.log(p[idx])).sum()

    return torch.dot(p, v) - reg * kl


def fenchel_kl_cvar(v, alpha):
    """Returns the empirical mean of the Fenchel dual for KL CVaR"""
    v -= np.log(1 / alpha)
    v1 = v[torch.lt(v, 0)]
    v2 = v[torch.ge(v, 0)]
    w1 = torch.exp(v1) / alpha - 1
    w2 = (v2 + 1) * (1 / alpha) - 1
    return (w1.sum() + w2.sum()) / v.shape[0]


def bisection(eta_min, eta_max, f, tol=1e-6, max_iter=500):
    """Expects f an increasing function and return eta in [eta_min, eta_max] 
    s.t. |f(eta)| <= tol (or the best solution after max_iter iterations"""
    lower = f(eta_min)
    upper = f(eta_max)

    # until the root is between eta_min and eta_max, double the length of the 
    # interval starting at either endpoint.
    while lower > 0 or upper < 0:
        length = eta_max - eta_min
        if lower > 0:
            eta_max = eta_min
            eta_min = eta_min - 2 * length
        if upper < 0:
            eta_min = eta_max
            eta_max = eta_max + 2 * length

        lower = f(eta_min)
        upper = f(eta_max)

    for _ in range(max_iter):
        eta = 0.5 * (eta_min + eta_max)

        v = f(eta)

        if torch.abs(v) <= tol:
            return eta

        if v > 0:
            eta_max = eta
        elif v < 0:
            eta_min = eta

    # if the minimum is not reached in max_iter, returns the current value
    logging.warning('Maximum number of iterations exceeded in bisection')
    return 0.5 * (eta_min + eta_max)


def huber_loss(x, delta=1.):
    """ Standard Huber loss of parameter delta

    https://en.wikipedia.org/wiki/Huber_loss

    returns 0.5 * x^2 if |a| <= \delta
            \delta * (|a| - 0.5 * \delta) o.w.
    """
    if torch.abs(x) <= delta:
        return 0.5 * (x ** 2)
    else:
        return delta * (torch.abs(x) - 0.5 * delta)


class RobustLoss(nn.Module):
    """PyTorch module for the batch robust loss estimator"""
    def __init__(self, base_loss_fn, size, reg, geometry, tol=1e-4,
                 max_iter=1000, debugging=False):
        """
        Parameters
        ----------

        base_loss_fn : callable
            Base loss function with reduction `none`

        size : float
            Size of the uncertainty set (\rho for \chi^2 and \alpha for CVaR)
            Set float('inf') for unconstrained
        reg : float
            Strength of the regularizer, entropy if geometry == 'cvar'
            $\chi^2$ divergence if geometry == 'chi-square'
        geometry : string
            Element of GEOMETRIES
        tol : float, optional
            Tolerance parameter for the bisection
        max_iter : int, optional
            Number of iterations after which to break the bisection

        """
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.size = size
        self.reg = reg
        self.geometry = geometry
        self.tol = tol
        self.max_iter = max_iter
        self.debugging = debugging

        self.is_erm = size == 0

        if geometry not in GEOMETRIES:
            raise ValueError('Geometry %s not supported' % geometry)

        if geometry == 'cvar' and self.size > 1:
            raise ValueError(f'alpha should be < 1 for cvar, is {self.size}')

    def best_response(self, v):
        size = self.size
        reg = self.reg
        m = v.shape[0]

        if self.geometry == 'cvar':
            if self.reg > 0:
                if size == 1.0:
                    return torch.ones_like(v) / m

                def p(eta):
                    x = (v - eta) / reg
                    return torch.min(torch.exp(x), torch.tensor([1 / size]).to(dtype=x.dtype, device=x.device)) / m

                def bisection_target(eta):
                    return 1.0 - p(eta).sum()

                eta_min = reg * torch.logsumexp(v / reg - np.log(m), 0)
                eta_max = v.max()

                if torch.abs(bisection_target(eta_min)) <= self.tol:
                    return p(eta_min)
            else:
                cutoff = int(size * m)
                surplus = 1.0 - cutoff / (size * m)

                p = torch.zeros_like(v)
                idx = torch.argsort(v, descending=True)
                p[idx[:cutoff]] = 1.0 / (size * m)
                if cutoff < m:
                    p[idx[cutoff]] = surplus
                return p

        if self.geometry == 'chi-square':
            if (v.max() - v.min()) / v.max() <= MIN_REL_DIFFERENCE:
                return torch.ones_like(v) / m

            if size == float('inf'):
                assert reg > 0

                def p(eta):
                    return torch.relu(v - eta) / (reg * m)

                def bisection_target(eta):
                    return 1.0 - p(eta).sum()

                eta_min = min(v.sum() - reg * m, v.min())
                eta_max = v.max()

            else:
                assert size < float('inf')

                # failsafe for batch sizes small compared to
                # uncertainty set size
                if m <= 1 + 2 * size:
                    out = (v == v.max()).float()
                    out /= out.sum()
                    return out

                if reg == 0:
                    def p(eta):
                        pp = torch.relu(v - eta)
                        return pp / pp.sum()

                    def bisection_target(eta):
                        pp = p(eta)
                        w = m * pp - torch.ones_like(pp)
                        return 0.5 * torch.mean(w ** 2) - size

                    eta_min = -(1.0 / (np.sqrt(2 * size + 1) - 1)) * v.max()
                    eta_max = v.max()
                else:
                    def p(eta):
                        pp = torch.relu(v - eta)

                        opt_lam = max(
                            reg, torch.norm(pp) / np.sqrt(m * (1 + 2 * size))
                        )

                        return pp / (m * opt_lam)

                    def bisection_target(eta):
                        return 1 - p(eta).sum()

                    eta_min = v.min() - 1
                    eta_max = v.max()

        eta_star = bisection(
            eta_min, eta_max, bisection_target,
            tol=self.tol, max_iter=self.max_iter)

        if self.debugging:
            return p(eta_star), eta_star
        return p(eta_star)

    def forward(self, preds, targets):
        """Value of the robust loss

        Note that the best response is computed without gradients

        Parameters
        ----------

        preds : torch.Tensor
            Tensor containing predictions on the batch of examples
        
        targets : torch.Tensor
            Tensor containing targets for the batch of examples

        Returns
        -------
        loss : torch.float
            Value of the robust loss on the batch of examples
        """
        v = self.base_loss_fn(preds, targets)
        if self.is_erm:
            return v.mean()
        else:
            with torch.no_grad():
                p = self.best_response(v)

            if self.geometry == 'cvar':
                return cvar_value(p, v, self.reg)
            elif self.geometry == 'chi-square':
                return chi_square_value(p, v, self.reg)
