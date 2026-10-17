"""Gói phần mềm hợp nhất dữ liệu nhận thức và quỹ đạo tàu thuyền đa nguồn.

Bao gồm các mô-đun giải mã AIS, chuyển đổi tọa độ, bám bắt đa mục tiêu MSTrack,
thuật toán hợp nhất tầng Cascade Association Matching và trực quan hóa video.
"""

from src.ais_decoder import AISDecoder, AISMessage, extrapolate_kinematic
from src.cascade_fusion import CascadeFusion, FusedTrack, TrajectoryState
from src.config import FusionType, IDDisplayMode, Modality, PipelineSettings
from src.coord_transform import CoordinateTransformer
from src.evaluator import PipelineEvaluator
from src.kalman_tracker import KalmanBoxTracker
from src.mstrack import MSTrack, Track
from src.visualizer import VideoVisualizer

__all__ = [
    "AISDecoder",
    "AISMessage",
    "CascadeFusion",
    "CoordinateTransformer",
    "FusedTrack",
    "FusionType",
    "IDDisplayMode",
    "KalmanBoxTracker",
    "MSTrack",
    "Modality",
    "PipelineEvaluator",
    "PipelineSettings",
    "Track",
    "TrajectoryState",
    "VideoVisualizer",
    "extrapolate_kinematic",
]
