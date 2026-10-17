"""Kiểm thử đơn vị cho mô-đun trực quan hóa VideoVisualizer."""

import numpy as np

from src.cascade_fusion import FusedTrack
from src.config import FusionType
from src.visualizer import VideoVisualizer


def test_draw_frame_with_tri_dual_and_single_tracks_renders_correct_colors() -> None:
    # Arrange
    visualizer = VideoVisualizer(target_size=(1280, 720))
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    tracks = [
        FusedTrack(
            fusion_id=1,
            fusion_type=FusionType.TRI,
            mmsi=413806072,
            radar_id=23,
            camera_id=20,
            camera_conf=1.0,
            bbox_vis=[727.0, 761.0, 377.0, 61.0],
            bbox_rad=[1723.0, 1134.0, 29.0, 56.0],
            lat=30.570142,
            lon=114.297007,
            sog=5.7,
            cog=34.5,
        ),
        FusedTrack(
            fusion_id=2,
            fusion_type=FusionType.DUAL_AIS_VIS,
            mmsi=413843904,
            radar_id=None,
            camera_id=5,
            camera_conf=1.0,
            bbox_vis=[431.0, 771.0, 181.0, 28.0],
            bbox_rad=None,
            lat=30.5716,
            lon=114.2902,
            sog=0.0,
            cog=0.0,
        ),
        FusedTrack(
            fusion_id=3,
            fusion_type=FusionType.SINGLE_VIS,
            mmsi=None,
            radar_id=None,
            camera_id=6,
            camera_conf=1.0,
            bbox_vis=[2168.0, 759.0, 113.0, 30.0],
            bbox_rad=None,
            lat=0.0,
            lon=0.0,
            sog=0.0,
            cog=0.0,
        ),
    ]

    # Act
    annotated = visualizer.draw_frame(frame, tracks, scale=0.5)

    # Assert
    assert annotated.shape == (720, 1280, 3)
    assert np.count_nonzero(annotated) > 1000

    has_red = bool(np.any((annotated[:, :, 2] > 180) & (annotated[:, :, 0] < 80)))
    has_yellow = bool(np.any((annotated[:, :, 2] > 180) & (annotated[:, :, 1] > 180)))
    has_green = bool(np.any((annotated[:, :, 1] > 180) & (annotated[:, :, 2] < 80)))

    assert has_red, "Cần chứa hộp bao màu đỏ cho vết hợp nhất 3 nguồn"
    assert has_yellow, "Cần chứa hộp bao màu vàng cho vết hợp nhất 2 nguồn"
    assert has_green, "Cần chứa hộp bao màu xanh lá cho vết nhận dạng đơn nguồn"
