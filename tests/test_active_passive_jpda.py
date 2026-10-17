import pytest
import numpy as np
from src.naval_isr.simulation.active_passive_jpda import ActivePassiveJPDA
from src.naval_isr.simulation.imm_tracker import IMMTracker

def test_bearing_only_gating_match():
    jpda = ActivePassiveJPDA(gate_threshold=9.0)
    # Track at x=10000, y=10000 -> true bearing = 45 degrees (pi/4 rad)
    track_state = np.array([10000.0, 0.0, 10000.0, 0.0])
    esm_bearing_rad = np.pi / 4.0
    dist = jpda.compute_angular_gating(track_state, esm_bearing_rad, sigma_theta=0.01, sigma_esm=0.02)
    assert dist < 1.0  # High confidence match

def test_bearing_only_gating_reject():
    jpda = ActivePassiveJPDA(gate_threshold=9.0)
    track_state = np.array([10000.0, 0.0, 10000.0, 0.0])  # 45 deg
    esm_bearing_rad = np.pi / 2.0  # 90 deg
    dist = jpda.compute_angular_gating(track_state, esm_bearing_rad, sigma_theta=0.01, sigma_esm=0.02)
    assert dist > 9.0  # Out of gate

def test_imm_tracker_cv_and_ct_modes():
    # Initialize IMM tracker at [0, 0] with velocity [10, 0]
    tracker = IMMTracker(init_pos=np.array([0.0, 0.0]), init_vel=np.array([10.0, 0.0]), dt=2.0)
    
    # Predict step
    x_pred, P_pred = tracker.predict()
    assert x_pred[0] > 0.0  # moved forward in X
    
    # Update with straight line radar measurements
    for step in range(5):
        tracker.predict()
        meas = np.array([(step + 1) * 20.0, 0.0])  # 10 m/s * 2s = 20m per step
        x_est, P_est = tracker.update_radar(meas)
    
    # CV model probability should remain high for straight line motion
    assert tracker.mu[0] > tracker.mu[1]

def test_jpda_joint_association():
    jpda = ActivePassiveJPDA(gate_threshold=9.0)
    # 2 tracks: Track 1 at [1000, 1000] (45 deg), Track 2 at [-1000, 1000] (135 deg)
    tracks = [
        {"track_id": 1, "state": np.array([1000.0, 0.0, 1000.0, 0.0]), "cov": np.eye(4) * 100.0},
        {"track_id": 2, "state": np.array([-1000.0, 0.0, 1000.0, 0.0]), "cov": np.eye(4) * 100.0},
    ]
    radar_meas = [
        {"meas_cartesian": np.array([1005.0, 995.0])},
        {"meas_cartesian": np.array([-998.0, 1002.0])},
    ]
    esm_meas = [
        {"bearing_rad": np.pi / 4.0 + 0.005, "emitter_id": "RADAR-MPQ-TYPE052D"},
    ]
    
    associations = jpda.associate(tracks, radar_meas, esm_meas)
    assert len(associations) == 2
    # Track 1 should be associated with ESM emitter
    t1_assoc = next(a for a in associations if a["track_id"] == 1)
    assert t1_assoc["associated_esm"] is not None
    assert t1_assoc["associated_esm"]["emitter_id"] == "RADAR-MPQ-TYPE052D"
