import pytest
import numpy as np
import cv2
from src.cascade_fusion import FusedTrack
from src.visualizer import VideoVisualizer

def test_visualizer_draw_frame():
    visualizer = VideoVisualizer(target_size=(1280, 720))
    # Blank frame 1280x720
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Create 3 tracks: one tri (red), one dual (yellow), one single (green)
    tracks = [
        FusedTrack(
            fusion_id=1,
            fusion_type='tri',
            mmsi=413806072,
            radar_id=23,
            camera_id=20,
            camera_conf=1.0,
            bbox_vis=[727.0, 761.0, 377.0, 61.0],  # in 2560x1440 space
            bbox_rad=[1723.0, 1134.0, 29.0, 56.0],
            lat=30.570142,
            lon=114.297007,
            sog=5.7,
            cog=34.5
        ),
        FusedTrack(
            fusion_id=2,
            fusion_type='dual_ais_vis',
            mmsi=413843904,
            radar_id=None,
            camera_id=5,
            camera_conf=1.0,
            bbox_vis=[431.0, 771.0, 181.0, 28.0],
            bbox_rad=None,
            lat=30.5716,
            lon=114.2902,
            sog=0.0,
            cog=0.0
        ),
        FusedTrack(
            fusion_id=3,
            fusion_type='single_vis',
            mmsi=None,
            radar_id=None,
            camera_id=6,
            camera_conf=1.0,
            bbox_vis=[2168.0, 759.0, 113.0, 30.0],
            bbox_rad=None,
            lat=0.0,
            lon=0.0,
            sog=0.0,
            cog=0.0
        )
    ]
    
    annotated = visualizer.draw_frame(frame, tracks, scale=0.5)
    assert annotated.shape == (720, 1280, 3)
    # Check that pixels have been drawn (non-zero)
    assert np.count_nonzero(annotated) > 1000
    
    # Check presence of red, yellow, and green colors
    # Red: high R, low B/G
    # Yellow: high R & G, low B
    # Green: high G, low R & B
    has_red = np.any((annotated[:, :, 2] > 180) & (annotated[:, :, 0] < 80))
    has_yellow = np.any((annotated[:, :, 2] > 180) & (annotated[:, :, 1] > 180))
    has_green = np.any((annotated[:, :, 1] > 180) & (annotated[:, :, 2] < 80))
    
    assert has_red, "Should contain red box for tri-fused track"
    assert has_yellow, "Should contain yellow box for dual-fused track"
    assert has_green, "Should contain green box for single-modality track"
