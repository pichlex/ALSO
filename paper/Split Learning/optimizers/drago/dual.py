import numpy as np
import torch


def get_smooth_weights(losses, spectrum, smooth_coef, smoothing="l2"):
    if smooth_coef < 1e-16:
        return spectrum[torch.argsort(torch.argsort(losses))]
    n = len(losses)
    scaled_losses = losses / smooth_coef
    perm = torch.argsort(losses)
    sorted_losses = scaled_losses[perm]

    if smoothing == "l2":
        primal_sol = l2_centered_isotonic_regression(
            sorted_losses.numpy(), spectrum.numpy()
        )
    elif smoothing == "neg_entropy":
        primal_sol = neg_entropy_centered_isotonic_regression(sorted_losses, spectrum)
    else:
        raise NotImplementedError
    inv_perm = torch.argsort(perm)
    primal_sol = primal_sol[inv_perm]
    if smoothing == "l2":
        smooth_weights = scaled_losses - primal_sol + 1 / n
    elif smoothing == "neg_entropy":
        smooth_weights = torch.exp(scaled_losses - primal_sol) / n
    else:
        raise NotImplementedError
    return smooth_weights

def get_smooth_weights_sorted(losses, spectrum, smooth_coef, smoothing="l2", tol=1e-16):
    if smooth_coef < 1e-16:
        return spectrum

    n = len(losses)
    sorted_losses = losses / smooth_coef

    if smoothing == "l2":
        primal_sol = torch.tensor(
            l2_centered_isotonic_regression(sorted_losses.numpy(), spectrum.numpy())
        )
    elif smoothing == "neg_entropy":
        primal_sol = torch.tensor(
            neg_entropy_centered_isotonic_regression(
                sorted_losses.numpy(), spectrum.numpy()
            )
        )
    else:
        raise NotImplementedError
    if smoothing == "l2":
        smooth_weights = sorted_losses - primal_sol + 1 / n
    elif smoothing == "neg_entropy":
        smooth_weights = torch.exp(sorted_losses - primal_sol) / n
    else:
        raise NotImplementedError
    return smooth_weights


def l2_centered_isotonic_regression(losses, spectrum):
    n = len(losses)
    means = [losses[0] + 1 / n - spectrum[0]]
    counts = [1]
    end_points = [0]
    for i in range(1, n):
        means.append(losses[i] + 1 / n - spectrum[i])
        counts.append(1)
        end_points.append(i)
        while len(means) > 1 and means[-2] >= means[-1]:
            prev_mean, prev_count, prev_end_point = (
                means.pop(),
                counts.pop(),
                end_points.pop(),
            )
            means[-1] = (counts[-1] * means[-1] + prev_count * prev_mean) / (
                counts[-1] + prev_count
            )
            counts[-1] = counts[-1] + prev_count
            end_points[-1] = prev_end_point

    # Expand function so numba understands.
    sol = np.zeros((n,))
    i = 0
    for j in range(len(end_points)):
        end_point = end_points[j]
        sol[i : end_point + 1] = means[j]
        i = end_point + 1
    return sol


def neg_entropy_centered_isotonic_regression(losses, spectrum):
    n = len(losses)
    # logn = torch.log(torch.tensor(n))
    # log_spectrum = torch.log(spectrum)
    logn = np.log(n)
    log_spectrum = -np.inf * np.ones(len(spectrum))
    log_spectrum[spectrum > 0] = np.log(spectrum[spectrum > 0])
    # log_spectrum = np.log(spectrum)
    

    lse_losses = [losses[0]]
    lse_log_spectrum = [log_spectrum[0]]
    means = [losses[0] - log_spectrum[0] - logn]
    end_points = [0]
    for i in range(1, n):
        means.append(losses[i] - log_spectrum[i] - logn)
        lse_losses.append(losses[i])
        lse_log_spectrum.append(log_spectrum[i])
        end_points.append(i)
        while len(means) > 1 and means[-2] >= means[-1]:
            prev_mean, prev_lse_loss, prev_lse_log_spectrum, prev_end_point = (
                means.pop(),
                lse_losses.pop(),
                lse_log_spectrum.pop(),
                end_points.pop(),
            )
            # new_lse_loss = torch.logsumexp(
            #     torch.tensor([lse_losses[-1], prev_lse_loss]), dim=0
            # )
            if lse_losses[-1] == -np.inf and prev_lse_loss == -np.inf:
                new_lse_loss = -np.inf
            else:
                new_lse_loss = np.logaddexp(lse_losses[-1], prev_lse_loss)
            # new_lse_log_spectrum = torch.logsumexp(
            #     torch.tensor([lse_log_spectrum[-1], prev_lse_log_spectrum]), dim=0
            # )
            if lse_log_spectrum[-1] == -np.inf and prev_lse_log_spectrum == -np.inf:
                new_lse_log_spectrum = -np.inf
            else:
                new_lse_log_spectrum = np.logaddexp(lse_log_spectrum[-1], prev_lse_log_spectrum)

            means[-1] = new_lse_loss - new_lse_log_spectrum - logn
            lse_losses[-1], lse_log_spectrum[-1] = new_lse_loss, new_lse_log_spectrum
            end_points[-1] = prev_end_point

    # Expand function so numba understands.
    sol = np.zeros((n,))
    i = 0
    for j in range(len(end_points)):
        end_point = end_points[j]
        sol[i : end_point + 1] = means[j]
        i = end_point + 1
    return sol

def projection_simplex_sorted(v, u, z=1):
    # assume that u is sorted version of v in decreasing order
    n_features = v.shape[0]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    if not np.any(cond):
        ret = np.zeros(shape=(n_features,), dtype=np.float64)
        ret[np.argmax(v)] = 1.
        return ret
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

def l2_centered_chi2_projection(losses, losses_sorted, rho, nu, lam):
    n = len(losses)
    q = projection_simplex_sorted(losses / (nu + lam), losses_sorted / (nu + lam))
    val = 0.5 * (nu + lam) * (q ** 2).sum() - q @ losses - lam * (rho + 0.5 / n)
    grad = 0.5 * (q ** 2).sum() - (rho + 0.5 / n)
    return val, grad, q


def exponential_search(losses, losses_sorted, rho, nu, tol=1e-10):
    lmin = 0.0
    lmax = 1.0
    gmax = l2_centered_chi2_projection(losses, losses_sorted, rho, nu, lmax)[1]
    i = 0
    while gmax > tol:
        lmin = lmax
        lmax *= 2
        gmax = l2_centered_chi2_projection(losses, losses_sorted, rho, nu, lmax)[1]
        i += 1
    return lmin, lmax


def chi2_divergence_oracle(radius, shift_cost, losses, tol=1e-10):
    radius = radius / len(losses)
    losses_sorted = np.sort(losses)[::-1]
    lmin, lmax =  exponential_search(losses, losses_sorted, radius, shift_cost)
    while lmax - lmin > tol:
        mid = (lmin + lmax) / 2
        val, grad, q = l2_centered_chi2_projection(losses, losses_sorted, radius, shift_cost, mid)
        if grad > tol:
            lmin = mid
        elif grad < -tol:
            lmax = mid
        else:
            break
    return q
