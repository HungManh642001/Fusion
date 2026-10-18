"""Integration test for Modular Maritime Fusion Pipeline."""
import numpy as np
import pytest
try:
    from naval_isr.tracking.modular_fusion_pipeline import ModularMaritimeFusionPipeline
except ImportError:
    from naval_isr.src.tracking.modular_fusion_pipeline import ModularMaritimeFusionPipeline


def test_pipeline_event_processing():
    p1_0 = np.array([100.0, 50.0])
    p2_0 = np.array([-100.0, -50.0])
    pipeline = ModularMaritimeFusionPipeline(p1_0, p2_0, is_t2_large=True)
    pipeline.initialize_velocities(np.array([2.0, 1.0]), np.array([-1.0, 0.5]))
    
    # Send a radar event (sensorID=2)
    event_radar = {
        "time": 1000.1,
        "sensorID": 2,
        "ownshipPosition": [0.0, 0.0],
        "measurement": [[100.2, -99.8], [50.1, -49.9]]
    }
    t1_pos, t2_pos = pipeline.process_event(event_radar)
    assert t1_pos.shape == (2,)
    assert t2_pos.shape == (2,)
    assert np.linalg.norm(t1_pos - p1_0) < 5.0
    assert np.linalg.norm(t2_pos - p2_0) < 5.0


def test_pipeline_trajectory_finalization():
    p1_0 = np.array([10.0, 20.0])
    p2_0 = np.array([-30.0, -40.0])
    pipeline = ModularMaritimeFusionPipeline(p1_0, p2_0, is_t2_large=False)
    for i in range(10):
        ev = {
            "time": 1000.0 + i * 0.1,
            "sensorID": 1,
            "ownshipPosition": [0.0, 0.0],
            "measurement": [[10.0 + i * 0.2, -30.0 - i * 0.1], [20.0 + i * 0.1, -40.0 - i * 0.2]]
        }
        pipeline.process_event(ev)
    s1, s2 = pipeline.finalize_trajectories()
    assert len(s1) == 10
    assert len(s2) == 10
    assert s1.shape[1] == 2
    assert s2.shape[1] == 2
