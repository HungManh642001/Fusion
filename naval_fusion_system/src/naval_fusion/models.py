"""Cấu trúc dữ liệu kiểu mạnh phục vụ phân hệ hợp nhất động học hải quân.

Module này định nghĩa các thực thể dữ liệu bất biến và có cấu trúc, bao gồm
loại cảm biến, trạng thái tàu mẹ, điểm đo dị thể, trạng thái vết mục tiêu
và các chỉ số đánh giá định lượng.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


class SensorType(IntEnum):
    """Mã định danh cảm biến theo quy chuẩn Autoferry.

    Attributes:
        LIDAR: LiDAR 3D đo tọa độ Descartes 2D [x, y] trong hệ thân tàu.
        RADAR: Marine Radar 360 độ đo tọa độ Descartes 2D [x, y] trong hệ thân tàu.
        IR: Camera ảnh nhiệt hồng ngoại đo góc phương vị 1D [theta] (radian).
        EO: Camera quang điện tử đo góc phương vị 1D [theta] (radian).
    """

    LIDAR = 1
    RADAR = 2
    IR = 3
    EO = 4


@dataclass(frozen=True)
class OwnshipState:
    """Trạng thái động học của tàu mẹ trong hệ tọa độ Piren NED.

    Attributes:
        timestamp: Thời gian ghi nhận theo chuẩn Unix epoch (giây).
        pos_north: Tọa độ vị trí trên trục Bắc (mét).
        pos_east: Tọa độ vị trí trên trục Đông (mét).
    """

    timestamp: float
    pos_north: float
    pos_east: float


@dataclass
class Detection:
    """Điểm đo chuẩn hóa nhận được từ các cảm biến dị thể.

    Attributes:
        sensor_type: Loại cảm biến cung cấp điểm đo.
        timestamp: Thời điểm ghi nhận điểm đo (giây).
        ownship_pos: Tọa độ tàu mẹ tại thời điểm đo (pos_north, pos_east) tính bằng mét.
        raw_measurement: Điểm đo thô trong hệ thân tàu ([x, y] hoặc [theta]).
        measurement_ned: Tọa độ bù về hệ NED thế giới ([pN, pE] hoặc [theta_NED]).
    """

    sensor_type: SensorType
    timestamp: float
    ownship_pos: tuple[float, float]
    raw_measurement: list[float]
    measurement_ned: np.ndarray


@dataclass
class TrackState:
    """Trạng thái một vết mục tiêu được quản lý trong hệ quy chiếu Piren NED.

    Attributes:
        track_id: Định danh duy nhất toàn cục của vết.
        state_vector: Vector trạng thái 4D [pN, pE, vN, vE]^T kích thước (4,).
        covariance: Ma trận hiệp phương sai sai số ước lượng kích thước (4, 4).
        last_updated: Thời điểm cuối cùng vết được cập nhật (giây).
        hits: Số lần ghép nối thành công với điểm đo cảm biến.
        misses: Số bước quét liên tiếp không nhận được điểm đo liên kết.
        is_confirmed: Trạng thái khẳng định vết (True nếu vết đã xác nhận).
        contributing_sensors: Tập hợp các cảm biến đã đóng góp dữ liệu cho vết.
    """

    track_id: int
    state_vector: np.ndarray
    covariance: np.ndarray
    last_updated: float
    hits: int
    misses: int
    is_confirmed: bool
    contributing_sensors: set[SensorType]


def _default_details() -> dict[str, Any]:
    return {}


@dataclass
class EvaluationMetrics:
    """Cấu trúc lưu trữ các chỉ số đánh giá chất lượng bám vết định lượng.

    Attributes:
        rmse_position: Sai số toàn phương trung bình vị trí (mét).
        rmse_velocity: Sai số toàn phương trung bình vận tốc (mét/giây).
        num_confirmed_tracks: Tổng số vết được xác nhận trong kịch bản.
        id_switches: Số lần tráo đổi định danh vết (ID switches).
        mostly_tracked_ratio: Tỷ lệ quỹ đạo mục tiêu được bám bắt liên tục (>=80%).
        details: Từ điển chứa các thông tin và chỉ số phụ trợ chi tiết.
    """

    rmse_position: float
    rmse_velocity: float
    num_confirmed_tracks: int
    id_switches: int
    mostly_tracked_ratio: float
    details: dict[str, Any] = field(default_factory=_default_details)
