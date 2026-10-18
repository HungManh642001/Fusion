"""Unit tests for Extended Object Tracking Offset & Hydrodynamic Smoother."""
import numpy as np
import pytest
try:
    from naval_isr.tracking.eot_compensator import EOTCompensator
except ImportError:
    from naval_isr.src.tracking.eot_compensator import EOTCompensator


def test_eot_antenna_offset_projection():
    comp = EOTCompensator()
    own_pos = np.array([0.0, 0.0])
    target_pos = np.array([100.0, 0.0])  # due North
    target_vel = np.array([5.0, 0.0])    # heading North (psi = 0)
    
    # Length = 31.25m, Beam = 9.6m
    offset = comp.compute_antenna_offset(own_pos, target_pos, target_vel, length=31.25, beam=9.6)
    # The sensor illuminates the stern/facing hull, offset should shift towards center
    assert len(offset) == 2
    assert abs(offset[0]) > 2.0  # non-trivial offset along longitudinal axis


def test_telemetry_conversion():
    comp = EOTCompensator()
    lat, lon, sog, cog = comp.to_wgs84_telemetry(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
    assert np.isclose(lat, 63.4389029, atol=1e-5)
    assert np.isclose(lon, 10.3990828, atol=1e-5)
    assert np.isclose(sog, 5.0 * 1.94384, atol=0.01)  # 5 m/s in knots
    assert np.isclose(cog, np.degrees(np.arctan2(4.0, 3.0)), atol=0.1)


def test_trajectory_smoothing():
    comp = EOTCompensator()
    t = np.linspace(0, 10, 100)
    # Smooth line + high frequency jitter
    clean = np.column_stack([t * 2.0, t * 1.5])
    jitter = np.random.normal(0, 0.5, clean.shape)
    noisy = clean + jitter
    
    smoothed = comp.smooth_trajectory(noisy, window_len=21, polyorder=2)
    assert smoothed.shape == noisy.shape
    # Variance of diff should be smaller in smoothed than noisy
    jitter_var = np.var(np.diff(noisy, axis=0))
    smoothed_var = np.var(np.diff(smoothed, axis=0))
    assert smoothed_var < jitter_var
