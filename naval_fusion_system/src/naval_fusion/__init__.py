"""Phân hệ hợp nhất động học đa cảm biến tàu chiến (Naval Kinematic Fusion Engine).

Module cung cấp lõi hợp nhất theo dõi mục tiêu phi hợp tác trên biển, hỗ trợ
bù chuyển động tàu mẹ trong hệ tọa độ Piren NED và xử lý cảm biến dị thể
(Radar, LiDAR, IR, EO).
"""

from naval_fusion.loader import AutoferryLoader
from naval_fusion.models import (
    Detection,
    EvaluationMetrics,
    OwnshipState,
    SensorType,
    TrackState,
)
from naval_fusion.pipeline import NavalFusionPipeline
from naval_fusion.visualizer import TacticalVisualizer

__all__ = [
    "AutoferryLoader",
    "Detection",
    "EvaluationMetrics",
    "NavalFusionPipeline",
    "OwnshipState",
    "SensorType",
    "TacticalVisualizer",
    "TrackState",
]

