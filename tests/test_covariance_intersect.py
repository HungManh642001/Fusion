import pytest
import numpy as np
from src.naval_isr.simulation.covariance_intersect import fuse_covariance_intersection, compute_ospa_metric

def test_ci_optimal_bounding():
    x1 = np.array([100.0, 10.0])
    P1 = np.diag([4.0, 1.0])
    x2 = np.array([102.0, 9.5])
    P2 = np.diag([2.0, 2.0])
    x_ci, P_ci, omega = fuse_covariance_intersection(x1, P1, x2, P2)
    assert 0.0 <= omega <= 1.0
    # CI covariance trace should be bounded
    assert np.trace(P_ci) <= max(np.trace(P1), np.trace(P2))
    # State estimate should lie between x1 and x2
    assert min(x1[0], x2[0]) <= x_ci[0] <= max(x1[0], x2[0])

def test_ci_identical_estimates():
    x1 = np.array([50.0, -20.0])
    P1 = np.diag([10.0, 10.0])
    x2 = np.array([50.0, -20.0])
    P2 = np.diag([10.0, 10.0])
    x_ci, P_ci, omega = fuse_covariance_intersection(x1, P1, x2, P2)
    np.testing.assert_allclose(x_ci, x1, atol=1e-5)
    np.testing.assert_allclose(P_ci, P1, atol=1e-5)

def test_ospa_metric_calculation():
    # True positions: 2 targets at [0, 0] and [100, 100]
    gt = np.array([[0.0, 0.0], [100.0, 100.0]])
    # Estimated positions with slight error
    est = np.array([[2.0, 0.0], [98.0, 101.0]])
    ospa = compute_ospa_metric(gt, est, c=50.0, p=1.0)
    assert 0.0 < ospa < 10.0
