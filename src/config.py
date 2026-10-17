"""Cấu hình và hằng số toàn cục cho hệ thống hợp nhất quỹ đạo tàu thuyền.

Mô-đun định nghĩa các hằng số vật lý, ma trận chiếu hiệu chuẩn, các tập giá trị
hữu hạn dạng Enum và cấu hình bất biến PipelineSettings.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

# Bán kính Trái Đất theo mét
EARTH_RADIUS_M: float = 6371000.0

# Hệ số chuyển đổi vận tốc từ hải lý/giờ (knots) sang mét/giây (m/s)
KNOTS_TO_MS: float = 0.51444444

# Ma trận Affine hiệu chuẩn chuyển đổi tọa độ trắc địa [lon, lat, 1]
# sang pixel radar [xr, yr]. Sai số tái chiếu trung bình là 1.14 px.
AFFINE_GEO_TO_RADAR: npt.NDArray[np.float64] = np.array(
    [
        [-5987.59411024, -13375.50157462, 1094999.35764672],
        [-1240.64398706, -4834.35908282, 290753.26076505],
    ],
    dtype=np.float64,
)

# Ma trận chiếu hiệu chuẩn F từ tọa độ radar [xr, yr, 1]
# sang hộp bao camera [xv, yv, wv, hv] trên độ phân giải camera 2560x1440.
F_RADAR_TO_VISIBLE: npt.NDArray[np.float64] = np.array(
    [
        [-1.38157877e01, -8.78508291e00, 3.51290525e04],
        [2.31098562e-01, -2.83131713e-01, 6.87422028e02],
        [2.00502139e00, 1.84434496e00, -5.27668883e03],
        [2.13125082e-01, 2.51459832e-01, -6.02300267e02],
    ],
    dtype=np.float64,
)

# Tọa độ giới hạn hành lang sông Trường Giang của tập dữ liệu Scene01
LAT_BOUNDS_DEFAULT: tuple[float, float] = (30.50, 30.62)
LON_BOUNDS_DEFAULT: tuple[float, float] = (114.20, 114.40)


class Modality(StrEnum):
    """Phương thức cảm biến thu thập dữ liệu."""

    RADAR = "radar"
    VISIBLE = "visible"


class FusionType(StrEnum):
    """Phân loại mức độ hợp nhất đa cảm biến."""

    TRI = "tri"
    DUAL_RAD_VIS = "dual_rad_vis"
    DUAL_AIS_VIS = "dual_ais_vis"
    DUAL_AIS_RAD = "dual_ais_rad"
    SINGLE_VIS = "single_vis"
    SINGLE_RAD = "single_rad"
    SINGLE_AIS = "single_ais"


class IDDisplayMode(StrEnum):
    """Chế độ hiển thị định danh mục tiêu trên video trực quan hóa."""

    SAMPLE_DEMO = "sample_demo"
    TRACKER = "tracker"
    GT = "gt"


@dataclass(frozen=True)
class PipelineSettings:
    """Cấu hình bất biến cho toàn bộ pipeline thực thi hợp nhất quỹ đạo.

    Attributes:
        scene_dir: Đường dẫn thư mục chứa dữ liệu kịch bản.
        output_dir: Thư mục lưu trữ kết quả đầu ra.
        start_frame: Chỉ số khung hình video bắt đầu kết xuất.
        num_frames: Số lượng khung hình xử lý.
        render_full: Cờ chỉ định kết xuất toàn bộ video.
        id_mode: Chế độ hiển thị ID tàu thuyền.
        fps: Tốc độ khung hình của video đầu ra.
        target_resolution: Độ phân giải video kết xuất (chiều rộng, chiều cao).
    """

    scene_dir: str = "datasets/scene01"
    output_dir: str = "output"
    start_frame: int = 7450
    num_frames: int = 833
    render_full: bool = False
    id_mode: IDDisplayMode = IDDisplayMode.SAMPLE_DEMO
    fps: float = 25.0
    target_resolution: tuple[int, int] = (1280, 720)
