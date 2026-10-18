"""Unit tests for Sensor Shadow Ray-Casting & Merged Rayleigh Manager."""
import numpy as np
import pytest
try:
    from naval_isr.tracking.shadow_manager import ShadowAndRayleighManager
except ImportError:
    from naval_isr.src.tracking.shadow_manager import ShadowAndRayleighManager


def test_ray_casting_occlusion_detection():
    mgr = ShadowAndRayleighManager(shadow_angle_deg=4.5)
    own = np.array([0.0, 0.0])
    # T1 is front at 100m bearing 0.5 rad; T2 is behind at 200m bearing 0.52 rad (~1.1 deg diff)
    pos1 = np.array([100.0 * np.cos(0.5), 100.0 * np.sin(0.5)])
    pos2 = np.array([200.0 * np.cos(0.52), 200.0 * np.sin(0.52)])
    
    is_occ, occ_id = mgr.check_occlusion(own, pos1, pos2)
    assert is_occ is True
    assert occ_id == 2  # Target 2 is in the shadow
    
    # Bearing to front ship should block T2 bearing update
    assert mgr.is_bearing_blocked_by_shadow(0.51, own, pos1) is True


def test_merged_radar_resolution():
    mgr = ShadowAndRayleighManager()
    pos1 = np.array([500.0, 200.0])
    pos2 = np.array([508.0, 205.0])  # 9.4m separation (unresolved)
    
    pts = [np.array([504.0, 202.5])]  # 1 merged centroid hit
    is_merged, weights = mgr.resolve_merged_radar(pts, pos1, pos2, gate=25.0)
    assert is_merged is True
    assert weights is not None
    w1, w2 = weights
    assert np.isclose(w1 + w2, 1.0)
    assert 0.3 < w1 < 0.7  # balanced weighting


def test_no_occlusion_when_angles_separated():
    mgr = ShadowAndRayleighManager(shadow_angle_deg=4.5)
    own = np.array([0.0, 0.0])
    # T1 at bearing 0.2 rad, T2 at bearing 0.6 rad (~23 deg diff)
    pos1 = np.array([100.0 * np.cos(0.2), 100.0 * np.sin(0.2)])
    pos2 = np.array([120.0 * np.cos(0.6), 120.0 * np.sin(0.6)])
    is_occ, occ_id = mgr.check_occlusion(own, pos1, pos2)
    assert is_occ is False
    assert occ_id == 0
