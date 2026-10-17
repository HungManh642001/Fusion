"""Mô-đun chuyển đổi không - thời gian giữa các hệ tọa độ quan sát.

Hỗ trợ chuyển đổi tương hỗ giữa ba hệ tọa độ:
1. Hệ tọa độ trắc địa toàn cầu WGS-84 (Vĩ độ, Kinh độ theo độ thập phân).
2. Hệ tọa độ điểm ảnh radar PPI (xr, yr).
3. Hệ tọa độ điểm ảnh camera trường nhìn khả kiến (xv, yv) và cổng bám bắt F.
"""

import logging

import numpy as np
import numpy.typing as npt

from src.config import AFFINE_GEO_TO_RADAR, F_RADAR_TO_VISIBLE

logger = logging.getLogger(__name__)


class CoordinateTransformer:
    """Bộ chuyển đổi tọa độ đa cảm biến phục vụ đồng bộ không gian quan sát."""

    def __init__(
        self,
        *,
        platform_lat: float = 30.563549,
        platform_lon: float = 114.306087,
        platform_h: float = 20.0,
    ) -> None:
        """Khởi tạo bộ chuyển đổi tọa độ với vị trí trạm quan trắc bờ.

        Args:
            platform_lat: Vĩ độ vị trí trạm quan trắc (độ thập phân).
            platform_lon: Kinh độ vị trí trạm quan trắc (độ thập phân).
            platform_h: Độ cao đặt cảm biến so với mực nước biển (mét).
        """
        self.platform_lat = platform_lat
        self.platform_lon = platform_lon
        self.platform_h = platform_h
        self.m_geo2rad: npt.NDArray[np.float64] = AFFINE_GEO_TO_RADAR
        self.f_rad2vis: npt.NDArray[np.float64] = F_RADAR_TO_VISIBLE

    def geodetic_to_radar(self, lat: float, lon: float) -> tuple[float, float]:
        """Chiếu tọa độ trắc địa (vĩ độ, kinh độ) sang tọa độ điểm ảnh radar (xr, yr).

        Args:
            lat: Vĩ độ trắc địa WGS-84 theo độ thập phân.
            lon: Kinh độ trắc địa WGS-84 theo độ thập phân.

        Returns:
            Bộ giá trị tọa độ điểm ảnh trên radar (xr, yr).
        """
        point_homo: npt.NDArray[np.float64] = np.array(
            [lon, lat, 1.0], dtype=np.float64
        )
        res = self.m_geo2rad @ point_homo
        return float(res[0]), float(res[1])

    def radar_to_geodetic(self, xr: float, yr: float) -> tuple[float, float]:
        """Ánh xạ ngược tọa độ điểm ảnh radar (xr, yr) về tọa độ trắc địa (lat, lon).

        Args:
            xr: Tọa độ hoành độ pixel trên ảnh radar.
            yr: Tọa độ tung độ pixel trên ảnh radar.

        Returns:
            Bộ giá trị tọa độ trắc địa (vĩ độ, kinh độ).
        """
        point_centered = np.array(
            [xr - self.m_geo2rad[0, 2], yr - self.m_geo2rad[1, 2]],
            dtype=np.float64,
        )
        affine_2x2 = self.m_geo2rad[:, :2]
        inv_affine = np.linalg.inv(affine_2x2)
        lon_lat = inv_affine @ point_centered
        return float(lon_lat[1]), float(lon_lat[0])

    def radar_to_visible(self, xr: float, yr: float) -> tuple[float, float]:
        """Chiếu tọa độ điểm ảnh radar (xr, yr) sang tâm hộp bao trên camera khả kiến.

        Args:
            xr: Hoành độ pixel trên radar.
            yr: Tung độ pixel trên radar.

        Returns:
            Bộ giá trị tọa độ tâm mục tiêu trên ảnh camera (xv_center, yv_center).
        """
        point_homo: npt.NDArray[np.float64] = np.array([xr, yr, 1.0], dtype=np.float64)
        bbox = self.f_rad2vis @ point_homo
        xv_center = bbox[0] + bbox[2] / 2.0
        yv_center = bbox[1] + bbox[3] / 2.0
        return float(xv_center), float(yv_center)

    def radar_bbox_to_visible_gate(
        self,
        bbox_rad: list[float],
        *,
        expansion_factor: float = 1.2,
    ) -> list[float]:
        """Chiếu hộp bao radar sang cổng bám bắt không gian trên ảnh camera khả kiến.

        Áp dụng hệ số mở rộng để bù trừ sai số ngoại suy và dao động sóng nước.

        Args:
            bbox_rad: Hộp bao mục tiêu radar dạng [xmin, ymin, w, h].
            expansion_factor: Hệ số mở rộng kích thước cổng bám bắt.

        Returns:
            Hộp cổng bám bắt [xmin, ymin, w, h] trên khung hình camera khả kiến.
        """
        xr_center = bbox_rad[0] + bbox_rad[2] / 2.0
        yr_center = bbox_rad[1] + bbox_rad[3] / 2.0
        point_homo: npt.NDArray[np.float64] = np.array(
            [xr_center, yr_center, 1.0], dtype=np.float64
        )
        v_box = self.f_rad2vis @ point_homo

        x_raw, y_raw, w_raw, h_raw = v_box[0], v_box[1], v_box[2], v_box[3]
        w_expanded = max(w_raw * expansion_factor, 60.0)
        h_expanded = max(h_raw * expansion_factor, 30.0)
        x_expanded = x_raw - (w_expanded - w_raw) / 2.0
        y_expanded = y_raw - (h_expanded - h_raw) / 2.0

        # Giới hạn trong kích thước khung hình video gốc 2560x1440
        x_clamped = max(0.0, min(2550.0, x_expanded))
        y_clamped = max(0.0, min(1430.0, y_expanded))
        return [
            float(x_clamped),
            float(y_clamped),
            float(w_expanded),
            float(h_expanded),
        ]

    def geodetic_to_visible(self, lat: float, lon: float) -> tuple[float, float]:
        """Chiếu trực tiếp tọa độ địa lý sang camera thông qua trung gian radar.

        Args:
            lat: Vĩ độ trắc địa.
            lon: Kinh độ trắc địa.

        Returns:
            Bộ giá trị tọa độ điểm ảnh camera (xv, yv).
        """
        xr, yr = self.geodetic_to_radar(lat, lon)
        return self.radar_to_visible(xr, yr)

    def is_in_msf_fov(self, lat: float, lon: float) -> bool:
        """Kiểm tra mục tiêu tàu thuyền có nằm trong trường quan sát camera hay không.

        Camera có trường nhìn ngang 54.3 độ bao quát luồng hàng hải sông Trường Giang.

        Args:
            lat: Vĩ độ trắc địa.
            lon: Kinh độ trắc địa.

        Returns:
            True nếu mục tiêu nằm trong tầm quan sát của camera, ngược lại False.
        """
        xr, yr = self.geodetic_to_radar(lat, lon)
        xv, _ = self.radar_to_visible(xr, yr)
        return -100.0 <= xv <= 2660.0
