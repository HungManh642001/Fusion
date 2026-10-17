import pytest
import numpy as np
from src.cascade_fusion import CascadeFusion, TrajectoryState, FusedTrack
from src.coord_transform import CoordinateTransformer

def test_cost_matrix_calculation():
    fusion = CascadeFusion()
    # State: x, y, vx, vy, azimuth, entry_time, length
    s1 = TrajectoryState(x=0.5, y=0.5, vx=1.0, vy=0.0, azimuth=0.0, entry_time=1000, length=10)
    s2 = TrajectoryState(x=0.51, y=0.51, vx=1.0, vy=0.0, azimuth=0.02, entry_time=1000, length=10)
    # Different heading
    s3 = TrajectoryState(x=0.51, y=0.51, vx=-1.0, vy=0.0, azimuth=3.14, entry_time=1000, length=10)
    
    cost_same_heading = fusion.compute_cost(s1, s2, k=0.5)
    cost_opp_heading = fusion.compute_cost(s1, s3, k=0.5)
    
    assert cost_same_heading < 0.2
    assert cost_opp_heading > cost_same_heading * 5.0

def test_cascade_association_three_way():
    fusion = CascadeFusion()
    # Mock AIS trajectory for MMSI 413806072
    ais_tracks = [{
        'mmsi': 413806072,
        'lat': 30.570142,
        'lon': 114.297007,
        'sog': 5.7,
        'cog': 34.5,
        'ts': 2365011758
    }]
    # Mock Radar track 26
    radar_tracks = [{
        'track_id': 23,
        'bbox': [1723.0, 1134.0, 29.0, 56.0],
        'conf': 0.9,
        'ts': 2365011758
    }]
    # Mock Visible track 20
    visible_tracks = [{
        'track_id': 20,
        'bbox': [727.0, 761.0, 377.0, 61.0],
        'conf': 1.0,
        'ts': 2365011758
    }]
    
    fused_results = fusion.fuse(ais_tracks, radar_tracks, visible_tracks, ts=2365011758)
    assert len(fused_results) >= 1
    tri_fused = [f for f in fused_results if f.fusion_type == 'tri']
    assert len(tri_fused) == 1
    f = tri_fused[0]
    assert f.mmsi == 413806072
    assert f.radar_id == 23
    assert f.camera_id == 20
    assert pytest.approx(f.sog, abs=0.1) == 5.7
    assert pytest.approx(f.cog, abs=0.5) == 34.5
