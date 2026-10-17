"""Thuật toán ghép nối dữ liệu Mahalanobis Gating và phạt góc Cosine vận tốc.

Module này cung cấp lớp `DataAssociator` thực hiện:
- Kiểm tra cổng không gian (spatial gating) bằng khoảng cách Mahalanobis 2D
  theo phân phối Chi-bình-phương (Chi-square distribution).
- Phạt chi phí liên kết đối với các mục tiêu và điểm đo di chuyển ngược hướng
  dựa trên tích vô hướng Cosine vận tốc.
- Kiểm tra cổng góc phương vị 1D cho các cảm biến quang điện tử / hồng ngoại (IR/EO).
- Giải bài toán phân công tối ưu 1-1 bằng thuật toán Hungary
  (Hungarian algorithm thông qua `scipy.optimize.linear_sum_assignment`).
- Phân loại và trả về danh sách các cặp ghép thành công (matches), vết không
  được ghép nối (unmatched tracks) và điểm đo chưa liên kết (unmatched detections).
"""

import logging
import math
from typing import cast

import numpy as np
from scipy.optimize import linear_sum_assignment  # pyright: ignore[reportUnknownVariableType]

from src.naval_fusion.coord_ned import normalize_bearing_rad
from src.naval_fusion.models import Detection, SensorType, TrackState

logger = logging.getLogger(__name__)

# Hằng số cấu hình mặc định theo quy chuẩn chiến thuật hải quân
DEFAULT_CHI2_GATE_2D: float = 9.21  # Bậc tự do 2, mức ý nghĩa p = 0.01 (99% gating)
DEFAULT_CHI2_GATE_1D: float = 9.0  # Bậc tự do 1, kiểm tra 3-sigma góc phương vị
DEFAULT_COSINE_WEIGHT: float = 3.0  # Trọng số phạt vận tốc ngược chiều lambda
DEFAULT_R_RADAR: np.ndarray = np.diag([25.0, 25.0])  # Ma trận nhiễu đo Radar (5m sigma)
DEFAULT_R_LIDAR: np.ndarray = np.diag([4.0, 4.0])  # Ma trận nhiễu đo LiDAR (2m sigma)
DEFAULT_R_ANGLE: float = 1e-4  # Phương sai nhiễu đo góc IR/EO (~0.57 độ sigma)


class DataAssociator:
    """Bộ ghép nối dữ liệu dị thể sử dụng Mahalanobis Gating và phạt Cosine.

    Attributes:
        chi2_gate_2d: Ngưỡng cổng Chi-square cho điểm đo không gian 2D (mặc định 9.21).
        chi2_gate_1d: Ngưỡng cổng Chi-square cho điểm đo góc 1D (mặc định 9.0).
        cosine_weight: Hệ số phạt góc lệch vận tốc lambda (mặc định 3.0).
        r_radar: Ma trận hiệp phương sai nhiễu đo mặc định của Radar kích thước (2, 2).
        r_lidar: Ma trận hiệp phương sai nhiễu đo mặc định của LiDAR kích thước (2, 2).
        r_angle: Phương sai nhiễu đo góc mặc định của IR/EO (rad^2).
    """

    def __init__(
        self,
        chi2_gate_2d: float = DEFAULT_CHI2_GATE_2D,
        chi2_gate_1d: float = DEFAULT_CHI2_GATE_1D,
        cosine_weight: float = DEFAULT_COSINE_WEIGHT,
        r_radar: np.ndarray | None = None,
        r_lidar: np.ndarray | None = None,
        r_angle: float = DEFAULT_R_ANGLE,
    ) -> None:
        """Khởi tạo thuật toán ghép nối dữ liệu.

        Args:
            chi2_gate_2d: Ngưỡng kiểm tra cổng 2D. Phải là số dương > 0.
            chi2_gate_1d: Ngưỡng kiểm tra cổng 1D. Phải là số dương > 0.
            cosine_weight: Hệ số trọng số phạt góc cosine. Phải không âm >= 0.
            r_radar: Ma trận nhiễu đo 2D cho cảm biến Radar (mặc định 25.0 * I_2).
            r_lidar: Ma trận nhiễu đo 2D cho cảm biến LiDAR (mặc định 4.0 * I_2).
            r_angle: Phương sai nhiễu đo góc 1D cho cảm biến IR/EO (mặc định 1e-4).

        Raises:
            ValueError: Nếu bất kỳ tham số ngưỡng nào không hợp lệ.
        """
        if chi2_gate_2d <= 0.0:
            msg = f"Ngưỡng chi2_gate_2d phải lớn hơn 0, nhận được: {chi2_gate_2d}"
            logger.error(msg)
            raise ValueError(msg)

        if chi2_gate_1d <= 0.0:
            msg = f"Ngưỡng chi2_gate_1d phải lớn hơn 0, nhận được: {chi2_gate_1d}"
            logger.error(msg)
            raise ValueError(msg)

        if cosine_weight < 0.0:
            msg = f"Trọng số cosine_weight phải không âm, nhận được: {cosine_weight}"
            logger.error(msg)
            raise ValueError(msg)

        if r_angle <= 0.0:
            msg = f"Phương sai nhiễu góc r_angle phải lớn hơn 0, nhận được: {r_angle}"
            logger.error(msg)
            raise ValueError(msg)

        self.chi2_gate_2d: float = float(chi2_gate_2d)
        self.chi2_gate_1d: float = float(chi2_gate_1d)
        self.cosine_weight: float = float(cosine_weight)

        self.r_radar: np.ndarray = (
            np.copy(r_radar) if r_radar is not None else np.copy(DEFAULT_R_RADAR)
        )
        self.r_lidar: np.ndarray = (
            np.copy(r_lidar) if r_lidar is not None else np.copy(DEFAULT_R_LIDAR)
        )
        self.r_angle: float = float(r_angle)

    def calc_cost_2d(
        self,
        track: TrackState,
        meas_pos_ned: np.ndarray,
        meas_vel: np.ndarray | None = None,
        r_2d: np.ndarray | None = None,
        R_2d: np.ndarray | None = None,  # noqa: N803
    ) -> float:
        """Tính chi phí liên kết 2D giữa vết và điểm đo vị trí Descartes.

        Công thức khoảng cách Mahalanobis:
            y = z_pos - x_pos
            S = P_pos + R_2d
            d_M^2 = y^T * S^-1 * y

        Nếu d_M^2 > chi2_gate_2d, điểm đo bị loại và trả về float('inf').
        Ngược lại, nếu có thông tin vận tốc đo meas_vel, cộng thêm hệ số phạt:
            Cost = d_M + cosine_weight * (1 - cos(v_trk, v_meas))

        Args:
            track: Bản ghi trạng thái vết mục tiêu hiện tại.
            meas_pos_ned: Tọa độ vị trí đo 2D [p_N, p_E] trong hệ Piren NED (mét).
            meas_vel: Vector vận tốc đo được [v_N, v_E] (tùy chọn).
            r_2d: Ma trận hiệp phương sai nhiễu đo kích thước (2, 2).
            R_2d: Tên đối số tương thích ngược cho r_2d.

        Returns:
            float: Chi phí liên kết (hữu hạn nếu trong cổng, hoặc
                float('inf') nếu ngoài).

        Raises:
            ValueError: Nếu meas_pos_ned không có ít nhất 2 phần tử.
        """
        pos_arr = np.asarray(meas_pos_ned, dtype=np.float64).reshape(-1)
        if pos_arr.shape[0] < 2:
            msg = f"meas_pos_ned phải có ít nhất 2 phần tử, nhận được: {pos_arr.shape}"
            logger.error(msg)
            raise ValueError(msg)

        # Vector sai số vị trí y = z_pos - x_pos
        y_diff = pos_arr[:2] - track.state_vector[:2]

        # Ma trận hiệp phương sai sai số vị trí P_pos (2x2)
        p_pos = track.covariance[0:2, 0:2]
        if R_2d is not None:
            r_mat = R_2d
        elif r_2d is not None:
            r_mat = r_2d
        else:
            r_mat = self.r_radar

        s_mat = p_pos + r_mat
        s_mat = 0.5 * (s_mat + s_mat.T)

        # Tính khoảng cách Mahalanobis bình phương d_M^2 = y^T * S^-1 * y
        try:
            d_m_sq = float(y_diff.T @ np.linalg.solve(s_mat, y_diff))
        except np.linalg.LinAlgError:
            # Nếu ma trận suy biến, thêm jitter nhỏ vào đường chéo
            s_jitter = s_mat + np.eye(2, dtype=np.float64) * 1e-6
            d_m_sq = float(y_diff.T @ np.linalg.solve(s_jitter, y_diff))

        d_m_sq = max(0.0, d_m_sq)

        # Kiểm tra ngưỡng cổng Chi-square 2D
        if d_m_sq > self.chi2_gate_2d:
            return float("inf")

        d_m = math.sqrt(d_m_sq)

        # Tính hệ số phạt Cosine nếu có thông tin vận tốc đo
        if meas_vel is not None:
            vel_arr = np.asarray(meas_vel, dtype=np.float64).reshape(-1)
            if vel_arr.shape[0] >= 2:
                v_trk = track.state_vector[2:4]
                v_meas = vel_arr[:2]

                norm_trk = float(np.linalg.norm(v_trk))
                norm_meas = float(np.linalg.norm(v_meas))

                # Chỉ áp dụng phạt hướng nếu cả hai đều có vận tốc khác 0 (> 1e-6 m/s)
                if norm_trk > 1e-6 and norm_meas > 1e-6:
                    dot_prod = float(np.dot(v_trk, v_meas))
                    cos_theta = dot_prod / (norm_trk * norm_meas)
                    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
                    cosine_penalty = self.cosine_weight * (1.0 - cos_theta)
                    return d_m + cosine_penalty

        return d_m

    def calc_cost_bearing_1d(
        self,
        track: TrackState,
        z_bearing: float,
        ownship_pos: tuple[float, float],
        r_angle: float | None = None,
        R_angle: float | None = None,  # noqa: N803
    ) -> float:
        """Tính chi phí liên kết góc phương vị 1D giữa vết và điểm đo IR/EO.

        Công thức:
            Delta_theta = normalize_bearing_rad(z_bearing - h(x))
            S_theta = H_j * P * H_j^T + R_angle
            d_angle^2 = Delta_theta^2 / S_theta

        Nếu d_angle^2 > chi2_gate_1d, điểm đo bị loại và trả về float('inf').
        Ngược lại, trả về d_angle = sqrt(d_angle^2).

        Args:
            track: Bản ghi trạng thái vết mục tiêu hiện tại.
            z_bearing: Góc phương vị đo được từ cảm biến (radian).
            ownship_pos: Tọa độ tàu mẹ [N_own, E_own] tại thời điểm đo (mét).
            r_angle: Phương sai nhiễu đo góc (rad^2).
            R_angle: Tên đối số tương thích ngược cho r_angle.

        Returns:
            float: Chi phí liên kết góc (khoảng cách Mahalanobis 1D hoặc float('inf')).

        Raises:
            ValueError: Nếu ownship_pos không đúng 2 phần tử hoặc phương sai nhiễu <= 0.
        """
        if len(ownship_pos) != 2:
            msg = f"ownship_pos phải có đúng 2 phần tử, nhận được: {len(ownship_pos)}"
            logger.error(msg)
            raise ValueError(msg)

        if R_angle is not None:
            r_val = R_angle
        elif r_angle is not None:
            r_val = r_angle
        else:
            r_val = self.r_angle

        if r_val <= 0.0:
            msg = f"Phương sai nhiễu góc phải > 0, nhận được: {r_val}"
            logger.error(msg)
            raise ValueError(msg)

        delta_n = float(track.state_vector[0]) - float(ownship_pos[0])
        delta_e = float(track.state_vector[1]) - float(ownship_pos[1])
        d_sq = delta_n * delta_n + delta_e * delta_e
        d_sq = max(d_sq, 1e-12)

        # Góc phương vị dự báo từ vị trí tàu mẹ tới vết mục tiêu h(x)
        h_pred = math.atan2(delta_e, delta_n)

        # Độ lệch góc chuẩn hóa về khoảng [-pi, pi]
        delta_theta = normalize_bearing_rad(float(z_bearing) - h_pred)

        # Ma trận Jacobian H_j kích thước (1, 4)
        h_jac = np.array(
            [[-delta_e / d_sq, delta_n / d_sq, 0.0, 0.0]],
            dtype=np.float64,
        )

        # Phương sai phần dư quan sát S_theta = H_j * P * H_j^T + R_angle
        s_theta = float((h_jac @ track.covariance @ h_jac.T).item()) + r_val
        s_theta = max(s_theta, 1e-12)

        # Khoảng cách Mahalanobis 1D bình phương
        d_angle_sq = (delta_theta * delta_theta) / s_theta

        if d_angle_sq > self.chi2_gate_1d:
            return float("inf")

        return math.sqrt(max(0.0, d_angle_sq))

    def associate(
        self,
        tracks: list[TrackState],
        detections: list[Detection],
        ownship_pos: tuple[float, float],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Thực hiện ghép nối dữ liệu tối ưu giữa danh sách vết và tập điểm đo.

        Quy trình xử lý:
        1. Xây dựng ma trận chi phí kích thước (N_tracks x N_dets).
        2. Tính chi phí cho từng cặp (vết i, điểm đo j) phụ thuộc vào loại cảm biến
           (2D Descartes cho Radar/LiDAR, 1D góc phương vị cho IR/EO).
        3. Sử dụng thuật toán Hungary (scipy.optimize.linear_sum_assignment)
           để giải bài toán phân công tối ưu toàn cục.
        4. Lọc bỏ các cặp ghép có chi phí bằng vô cùng (bị loại trừ bởi cổng Gating).
        5. Xác định danh sách vết chưa ghép nối và điểm đo chưa được liên kết.

        Args:
            tracks: Danh sách các vết mục tiêu hiện có trong hệ thống.
            detections: Danh sách các điểm đo cảm biến tại bước quét hiện tại.
            ownship_pos: Tọa độ tàu mẹ [pos_north, pos_east] tại thời điểm quét.

        Returns:
            tuple[list[tuple[int, int]], list[int], list[int]]:
                - matches: Danh sách các cặp chỉ số ghép nối (track_idx, det_idx).
                - unmatched_tracks: Danh sách chỉ số các vết không nhận được điểm đo.
                - unmatched_detections: Danh sách chỉ số các điểm đo chưa ghép vào vết.
        """
        num_tracks = len(tracks)
        num_dets = len(detections)

        # Xử lý các trường hợp biên danh sách rỗng
        if num_tracks == 0:
            return [], [], list(range(num_dets))

        if num_dets == 0:
            return [], list(range(num_tracks)), []

        # Khởi tạo ma trận chi phí (num_tracks, num_dets)
        cost_matrix = np.full(
            (num_tracks, num_dets), fill_value=np.inf, dtype=np.float64
        )

        for i, trk in enumerate(tracks):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = self._calc_single_cost(trk, det, ownship_pos)

        # Nếu toàn bộ ma trận đều là inf (tất cả đều bị cổng gating chặn)
        if np.all(np.isinf(cost_matrix)):
            return [], list(range(num_tracks)), list(range(num_dets))

        # Để tránh lỗi 'cost matrix is infeasible' từ linear_sum_assignment khi một số
        # hàng hoặc cột toàn là inf, thay thế inf bằng một giá trị chi phí rất lớn
        finite_mask = ~np.isinf(cost_matrix)
        if not np.all(finite_mask):
            finite_vals = cost_matrix[finite_mask]
            max_finite = float(np.max(finite_vals)) if len(finite_vals) > 0 else 0.0
            large_cost = max(max_finite * 100.0 + 1000.0, 1e6)
            assign_matrix = np.where(np.isinf(cost_matrix), large_cost, cost_matrix)
        else:
            large_cost = float("inf")
            assign_matrix = cost_matrix

        # Phân công tối ưu 1-1 bằng thuật toán Hungary
        result = cast(
            tuple[np.ndarray, np.ndarray],
            linear_sum_assignment(assign_matrix),
        )
        row_ind: list[int] = [int(idx) for idx in result[0]]
        col_ind: list[int] = [int(idx) for idx in result[1]]

        matches: list[tuple[int, int]] = []
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        for r_idx, c_idx in zip(row_ind, col_ind, strict=False):
            cost_val = float(cost_matrix[r_idx, c_idx])

            # Chỉ chấp nhận ghép nối nếu chi phí hữu hạn và nhỏ hơn ngưỡng phạt thay thế
            if not np.isinf(cost_val) and cost_val < large_cost:
                matches.append((r_idx, c_idx))
                matched_tracks.add(r_idx)
                matched_dets.add(c_idx)

        unmatched_tracks = [i for i in range(num_tracks) if i not in matched_tracks]
        unmatched_detections = [j for j in range(num_dets) if j not in matched_dets]

        return matches, unmatched_tracks, unmatched_detections

    def _calc_single_cost(
        self,
        track: TrackState,
        detection: Detection,
        ownship_pos: tuple[float, float],
    ) -> float:
        """Tính chi phí liên kết giữa một vết và một điểm đo cụ thể.

        Args:
            track: Trạng thái vết mục tiêu.
            detection: Điểm đo từ cảm biến.
            ownship_pos: Tọa độ vị trí tàu mẹ.

        Returns:
            float: Chi phí liên kết tính toán được (hữu hạn hoặc float('inf')).
        """
        sensor_type = detection.sensor_type

        if sensor_type == SensorType.RADAR:
            meas_vel = (
                detection.measurement_ned[2:4]
                if detection.measurement_ned.shape[0] >= 4
                else None
            )
            return self.calc_cost_2d(
                track=track,
                meas_pos_ned=detection.measurement_ned[:2],
                meas_vel=meas_vel,
                R_2d=self.r_radar,
            )

        if sensor_type == SensorType.LIDAR:
            meas_vel = (
                detection.measurement_ned[2:4]
                if detection.measurement_ned.shape[0] >= 4
                else None
            )
            return self.calc_cost_2d(
                track=track,
                meas_pos_ned=detection.measurement_ned[:2],
                meas_vel=meas_vel,
                R_2d=self.r_lidar,
            )

        if sensor_type in (SensorType.IR, SensorType.EO):
            pos = detection.ownship_pos
            bearing_rad = float(detection.measurement_ned[0])
            return self.calc_cost_bearing_1d(
                track=track,
                z_bearing=bearing_rad,
                ownship_pos=pos,
                r_angle=self.r_angle,
            )

        # Trường hợp dự phòng nếu kiểu cảm biến không nằm trong enum chuẩn
        if detection.measurement_ned.shape[0] >= 2:
            return self.calc_cost_2d(
                track=track, meas_pos_ned=detection.measurement_ned[:2]
            )

        bearing_rad = float(detection.measurement_ned[0])
        return self.calc_cost_bearing_1d(
            track=track,
            z_bearing=bearing_rad,
            ownship_pos=ownship_pos,
            R_angle=self.r_angle,
        )
