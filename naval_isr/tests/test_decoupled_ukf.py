"""Unit tests for Decoupled Unscented Kalman Filter Core."""
import numpy as np
import pytest
try:
    from naval_isr.tracking.decoupled_ukf import DecoupledUKF
except ImportError:
    from naval_isr.src.tracking.decoupled_ukf import DecoupledUKF

def test_ukf_prediction_and_velocity_decoupling():
    init_pos = np.array([100.0, 50.0])
    tracker = DecoupledUKF(init_pos=init_pos, is_large_ship=False)
    tracker.x[1] = 3.0  # v_N
    tracker.x[3] = 4.0  # v_E
    
    # 1. Predict 1.0 second
    tracker.predict(1.0)
    assert np.isclose(tracker.x[0], 103.0)
    assert np.isclose(tracker.x[2], 54.0)
    
    # 2. Bearing update must decouple velocity: velocity must remain strictly [3.0, 4.0]
    own_pos = np.array([0.0, 0.0])
    true_brg = np.arctan2(54.0, 103.0)
    noisy_brg = true_brg + np.radians(2.0)  # 2 degrees error
    
    updated = tracker.update_bearing(noisy_brg, own_pos)
    assert updated is True
    assert np.isclose(tracker.x[1], 3.0, atol=1e-6)
    assert np.isclose(tracker.x[3], 4.0, atol=1e-6)

def test_ukf_velocity_clamping():
    tracker = DecoupledUKF(init_pos=np.array([0.0, 0.0]))
    tracker.x[1] = 20.0
    tracker.x[3] = 20.0
    tracker.predict(0.1)
    assert tracker.speed <= 12.01

def test_ukf_zero_range_singularity():
    tracker = DecoupledUKF(init_pos=np.array([0.0, 0.0]))
    own_pos = np.array([0.0, 0.0])
    # Range is 0, should safely return False without ZeroDivisionError
    res = tracker.update_bearing(0.5, own_pos)
    assert res is False
