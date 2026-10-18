"""Covariance Intersection (CI) Track-to-Track Fusion and OSPA Evaluation.

Provides:
1. `fuse_covariance_intersection`: Robust, correlation-agnostic track fusion minimizing
   trace of combined covariance matrix via 1D convex optimization.
2. `compute_ospa_metric`: Optimal Subpattern Assignment metric for multitarget tracking benchmark.
"""

from typing import Tuple
import numpy as np
from scipy.optimize import minimize_scalar, linear_sum_assignment


def fuse_covariance_intersection(
    x1: np.ndarray,
    P1: np.ndarray,
    x2: np.ndarray,
    P2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fuse two track estimates (x1, P1) and (x2, P2) using Covariance Intersection.

    Returns:
        (x_fused, P_fused, optimal_omega)
    """
    x1 = np.asarray(x1, dtype=float).flatten()
    x2 = np.asarray(x2, dtype=float).flatten()
    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)

    inv_P1 = np.linalg.pinv(P1)
    inv_P2 = np.linalg.pinv(P2)

    def cost_fn(omega: float) -> float:
        P_ci_inv = omega * inv_P1 + (1.0 - omega) * inv_P2
        # Return trace of covariance
        try:
            P_ci = np.linalg.pinv(P_ci_inv)
            return float(np.trace(P_ci))
        except Exception:
            return 1e12

    res = minimize_scalar(cost_fn, bounds=(0.0, 1.0), method="bounded")
    best_omega = float(np.clip(res.x, 0.0, 1.0))

    P_ci_inv = best_omega * inv_P1 + (1.0 - best_omega) * inv_P2
    P_fused = np.linalg.pinv(P_ci_inv)
    x_fused = P_fused @ (best_omega * inv_P1 @ x1 + (1.0 - best_omega) * inv_P2 @ x2)

    return x_fused, P_fused, best_omega


def compute_ospa_metric(
    gt_points: np.ndarray,
    est_points: np.ndarray,
    c: float = 100.0,
    p: float = 1.0,
) -> float:
    """Compute the Optimal Subpattern Assignment (OSPA) distance metric.

    Args:
        gt_points: Array of ground-truth 2D positions (M, 2)
        est_points: Array of estimated 2D positions (N, 2)
        c: Cut-off parameter (meters)
        p: Order parameter (p >= 1)

    Returns:
        OSPA distance (scalar >= 0)
    """
    gt = np.atleast_2d(gt_points)
    est = np.atleast_2d(est_points)

    m = len(gt) if gt.size > 0 else 0
    n = len(est) if est.size > 0 else 0

    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return float(c)

    # Compute pairwise distance matrix truncated at c
    dist_mat = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            d = np.linalg.norm(gt[i] - est[j])
            dist_mat[i, j] = min(float(d), c)

    if m <= n:
        row_ind, col_ind = linear_sum_assignment(dist_mat)
        sum_matched_p = np.sum(dist_mat[row_ind, col_ind] ** p)
        sum_unmatched_p = (n - m) * (c**p)
        ospa = ((1.0 / n) * (sum_matched_p + sum_unmatched_p)) ** (1.0 / p)
    else:
        # Swap roles
        col_ind, row_ind = linear_sum_assignment(dist_mat.T)
        sum_matched_p = np.sum(dist_mat.T[col_ind, row_ind] ** p)
        sum_unmatched_p = (m - n) * (c**p)
        ospa = ((1.0 / m) * (sum_matched_p + sum_unmatched_p)) ** (1.0 / p)

    return float(ospa)
