"""Bộ lọc Kalman theo vết hộp bao mục tiêu trong không gian ảnh 2D.

Mô hình hóa trạng thái động học 8 chiều: [xc, yc, s, h, vx, vy, vs, vh]
trong đó (xc, yc) là tâm hộp bao, s là tỷ lệ khung hình (w/h), h là chiều cao,
và các thành phần vận tốc tương ứng.
"""

import logging
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def convert_bbox_to_z(bbox: Sequence[float]) -> npt.NDArray[np.float64]:
    """Chuyển đổi hộp bao [x, y, w, h] sang vector quan sát z = [xc, yc, s, h].

    Args:
        bbox: Chuỗi 4 phần tử biểu diễn [xmin, ymin, w, h].

    Returns:
        Vector cột 4 chiều biểu diễn không gian quan sát của bộ lọc Kalman.
    """
    width = max(float(bbox[2]), 1.0)
    height = max(float(bbox[3]), 1.0)
    center_x = float(bbox[0]) + width / 2.0
    center_y = float(bbox[1]) + height / 2.0
    aspect_ratio = width / height
    z_vec = np.array([center_x, center_y, aspect_ratio, height], dtype=np.float64)
    return z_vec.reshape((4, 1))


def convert_x_to_bbox(state_x: npt.NDArray[np.float64]) -> list[float]:
    """Chuyển đổi vector trạng thái Kalman sang hộp bao chuẩn [xmin, ymin, w, h].

    Args:
        state_x: Vector trạng thái 8 chiều của bộ lọc Kalman.

    Returns:
        Danh sách 4 phần tử [xmin, ymin, w, h] theo pixel ảnh.
    """
    center_x = float(state_x[0, 0])
    center_y = float(state_x[1, 0])
    aspect_ratio = float(state_x[2, 0])
    height = float(state_x[3, 0])

    width = max(aspect_ratio * height, 1.0)
    height_safe = max(height, 1.0)
    xmin = center_x - width / 2.0
    ymin = center_y - height_safe / 2.0
    return [xmin, ymin, width, height_safe]


class KalmanBoxTracker:
    """Bộ bám bắt Kalman cho từng đối tượng hộp bao đơn lẻ."""

    _count: int = 0

    def __init__(self, bbox: list[float], *, track_id: int | None = None) -> None:
        """Khởi tạo trạng thái và ma trận hiệp phương sai cho bộ bám bắt.

        Args:
            bbox: Hộp bao phát hiện ban đầu [xmin, ymin, w, h].
            track_id: Mã định danh quỹ đạo nếu có, hoặc tự động cấp phát tăng dần.
        """
        if track_id is None:
            KalmanBoxTracker._count += 1
            self.track_id: int = KalmanBoxTracker._count
        else:
            self.track_id = track_id

        self.dim_x: int = 8
        self.dim_z: int = 4

        # Khởi tạo vector trạng thái x
        self.x: npt.NDArray[np.float64] = np.zeros((self.dim_x, 1), dtype=np.float64)
        self.x[:4] = convert_bbox_to_z(bbox)

        # Ma trận chuyển trạng thái F
        self.f: npt.NDArray[np.float64] = np.eye(self.dim_x, dtype=np.float64)
        for i in range(4):
            self.f[i, i + 4] = 1.0

        # Ma trận quan sát H
        self.h: npt.NDArray[np.float64] = np.zeros(
            (self.dim_z, self.dim_x), dtype=np.float64
        )
        for i in range(4):
            self.h[i, i] = 1.0

        # Ma trận hiệp phương sai sai số ước lượng P
        self.p: npt.NDArray[np.float64] = np.eye(self.dim_x, dtype=np.float64) * 10.0
        self.p[4:, 4:] *= 100.0

        # Ma trận hiệp phương sai nhiễu quá trình Q
        self.q: npt.NDArray[np.float64] = np.eye(self.dim_x, dtype=np.float64) * 1.0
        self.q[4:, 4:] *= 0.01

        # Ma trận hiệp phương sai nhiễu đo R
        self.r: npt.NDArray[np.float64] = np.eye(self.dim_z, dtype=np.float64) * 1.0

        self.time_since_update: int = 0
        self.history: list[list[float]] = [bbox]
        self.hits: int = 1
        self.age: int = 1
        self.confidence_history: list[float] = [1.0]

    def predict(self) -> list[float]:
        """Thực hiện bước ngoại suy dự báo trạng thái và trả về hộp bao dự đoán.

        Returns:
            Hộp bao dự đoán [xmin, ymin, w, h] tại bước thời gian hiện tại.
        """
        # Đảm bảo tỷ lệ khung hình và chiều cao không bị âm do nhiễu số học
        if self.x[3, 0] + self.x[7, 0] <= 0:
            self.x[7, 0] = 0.0

        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q
        self.age += 1
        self.time_since_update += 1
        return convert_x_to_bbox(self.x)

    def update(self, bbox: list[float], *, conf: float = 1.0) -> None:
        """Cập nhật trạng thái bộ lọc khi có hộp bao đo đạc mới.

        Args:
            bbox: Hộp bao đo đạc quan sát được [xmin, ymin, w, h].
            conf: Độ tin cậy của phép phát hiện đo đạc.
        """
        self.time_since_update = 0
        self.hits += 1
        self.confidence_history.append(conf)
        z = convert_bbox_to_z(bbox)

        # Tính toán sai lệch quan sát (Innovation)
        y = z - self.h @ self.x
        innovation_cov = self.h @ self.p @ self.h.T + self.r
        kalman_gain = self.p @ self.h.T @ np.linalg.inv(innovation_cov)

        self.x = self.x + kalman_gain @ y
        identity_mat = np.eye(self.dim_x, dtype=np.float64)
        self.p = (identity_mat - kalman_gain @ self.h) @ self.p

        current_bbox = convert_x_to_bbox(self.x)
        self.history.append(current_bbox)

    def get_state(self) -> list[float]:
        """Lấy vị trí hộp bao ước lượng tối ưu hiện tại.

        Returns:
            Hộp bao ước lượng [xmin, ymin, w, h].
        """
        return convert_x_to_bbox(self.x)
