"""Bộ lọc Kalman mở rộng dị thể (Heterogeneous Extended Kalman Filter).

Module này triển khai bộ lọc EKF đa chế độ chuyên dụng cho tác chiến hải quân,
hợp nhất điểm đo từ nhiều loại cảm biến dị thể (heterogeneous sensors) bao gồm:
- Cảm biến đo tọa độ Descartes 2D tuyến tính: Marine Radar và LiDAR 3D.
- Cảm biến đo góc phương vị 1D phi tuyến: Camera ảnh nhiệt hồng ngoại IR
  và Camera quang điện tử EO.

Bộ lọc sử dụng mô hình động học vận tốc không đổi CV (Constant Velocity)
với nhiễu gia tốc trắng liên tục (Continuous White Noise Acceleration - CWNA).
Tất cả các bước cập nhật hiệp phương sai đều sử dụng dạng ổn định số học Joseph
kết hợp ép đối xứng và kiểm tra dương xác định.
"""

import logging
import math
from collections.abc import Sequence

import numpy as np

from src.naval_fusion.coord_ned import normalize_bearing_rad
from src.naval_fusion.models import SensorType, TrackState

logger = logging.getLogger(__name__)


class HeterogeneousEKF:
    """Bộ lọc Kalman mở rộng dị thể hỗ trợ cập nhật 2D Descartes và 1D góc phương vị.

    Attributes:
        q_noise: Cường độ mật độ phổ công suất nhiễu gia tốc quá trình (m^2/s^3).
        H_CARTESIAN: Ma trận quan sát tuyến tính 2D cho Radar/LiDAR kích thước (2, 4).
    """

    H_CARTESIAN: np.ndarray = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    def __init__(self, q_noise: float = 0.5) -> None:
        """Khởi tạo bộ lọc Kalman mở rộng dị thể.

        Args:
            q_noise: Cường độ nhiễu quá trình gia tốc trắng (m^2/s^3). Mặc định là 0.5.

        Raises:
            ValueError: Nếu q_noise không phải là số dương lớn hơn 0.
        """
        if q_noise <= 0.0:
            msg = f"Cường độ nhiễu q_noise phải lớn hơn 0, nhận được: {q_noise}"
            logger.error(msg)
            raise ValueError(msg)

        self.q_noise: float = float(q_noise)

    def predict(self, track: TrackState, dt: float) -> TrackState:
        """Ngoại suy trạng thái động học và ma trận hiệp phương sai theo mô hình CV.

        Sử dụng ma trận chuyển trạng thái F(dt) và ma trận nhiễu quá trình gia tốc
        trắng liên tục Q(dt).

        Args:
            track: Trạng thái vết mục tiêu hiện tại trước khi ngoại suy.
            dt: Khoảng thời gian ngoại suy bước nhảy (giây). Phải thỏa mãn dt >= 0.

        Returns:
            TrackState: Bản ghi trạng thái vết mới sau khi ngoại suy.

        Raises:
            ValueError: Nếu dt < 0.
        """
        if dt < 0.0:
            msg = f"Khoảng thời gian ngoại suy dt phải không âm, nhận được: {dt}"
            logger.error(msg)
            raise ValueError(msg)

        if dt == 0.0:
            return TrackState(
                track_id=track.track_id,
                state_vector=np.copy(track.state_vector),
                covariance=np.copy(track.covariance),
                last_updated=track.last_updated,
                hits=track.hits,
                misses=track.misses,
                is_confirmed=track.is_confirmed,
                contributing_sensors=set(track.contributing_sensors),
            )

        # 1. Ma trận chuyển trạng thái F(dt)
        f_matrix = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # 2. Ma trận nhiễu quá trình Q(dt) theo mô hình CWNA
        dt2 = (dt * dt) / 2.0
        dt3 = (dt * dt * dt) / 3.0
        q_matrix = self.q_noise * np.array(
            [
                [dt3, 0.0, dt2, 0.0],
                [0.0, dt3, 0.0, dt2],
                [dt2, 0.0, dt, 0.0],
                [0.0, dt2, 0.0, dt],
            ],
            dtype=np.float64,
        )

        # 3. Ngoại suy vector trạng thái x_pred = F * x
        state_pred = f_matrix @ track.state_vector

        # 4. Ngoại suy ma trận hiệp phương sai P_pred = F * P * F^T + Q
        cov_pred = f_matrix @ track.covariance @ f_matrix.T + q_matrix

        # Ép đối xứng và kiểm tra dương xác định
        cov_pred = self._enforce_symmetric_pd(cov_pred)

        return TrackState(
            track_id=track.track_id,
            state_vector=state_pred,
            covariance=cov_pred,
            last_updated=track.last_updated + dt,
            hits=track.hits,
            misses=track.misses,
            is_confirmed=track.is_confirmed,
            contributing_sensors=set(track.contributing_sensors),
        )

    def update_cartesian_2d(
        self,
        track: TrackState,
        z_2d: np.ndarray | Sequence[float],
        R_cov: np.ndarray,  # noqa: N803
        sensor_type: SensorType | None = None,
        timestamp: float | None = None,
    ) -> TrackState:
        """Cập nhật tuyến tính Kalman 2D cho điểm đo Radar hoặc LiDAR.

        Mô hình quan sát tuyến tính:
            z = H * x + v, với H = [[1, 0, 0, 0], [0, 1, 0, 0]]
        Công thức hiệp phương sai được cập nhật dưới dạng Joseph:
            P = (I - K*H) * P * (I - K*H)^T + K * R * K^T

        Args:
            track: Trạng thái vết mục tiêu cần cập nhật.
            z_2d: Vector đo vị trí 2D [p_N, p_E] trong hệ Piren NED (mét).
            R_cov: Ma trận hiệp phương sai nhiễu đo kích thước (2, 2) (m^2).
            sensor_type: Loại cảm biến thực hiện cập nhật (tùy chọn).
            timestamp: Thời điểm ghi nhận điểm đo (tùy chọn).

        Returns:
            TrackState: Bản ghi trạng thái vết mới sau khi cập nhật.

        Raises:
            ValueError: Nếu z_2d không đúng 2 phần tử hoặc R_cov không đúng dạng (2, 2).
        """
        z_arr = np.asarray(z_2d, dtype=np.float64).reshape(-1)
        if z_arr.shape != (2,):
            msg = f"Điểm đo z_2d phải là vector 2D, nhận được hình dạng {z_arr.shape}"
            logger.error(msg)
            raise ValueError(msg)

        r_mat = np.asarray(R_cov, dtype=np.float64)
        if r_mat.shape != (2, 2):
            msg = f"Ma trận R_cov phải có kích thước (2, 2), nhận được {r_mat.shape}"
            logger.error(msg)
            raise ValueError(msg)

        h_mat = self.H_CARTESIAN
        x_state = track.state_vector
        p_cov = track.covariance

        # Innovation (phần dư quan sát): y = z - H * x
        innovation = z_arr - h_mat @ x_state

        # Innovation covariance: S = H * P * H^T + R
        s_cov = h_mat @ p_cov @ h_mat.T + r_mat
        s_cov = 0.5 * (s_cov + s_cov.T)

        # Kalman gain: K = P * H^T * S^-1
        s_inv = np.linalg.inv(s_cov)
        k_gain = p_cov @ h_mat.T @ s_inv

        # Cập nhật trạng thái: x = x + K * y
        x_updated = x_state + k_gain @ innovation

        # Cập nhật hiệp phương sai theo dạng Joseph ổn định số học:
        # P = (I - K * H) * P * (I - K * H)^T + K * R * K^T
        eye4 = np.eye(4, dtype=np.float64)
        i_kh = eye4 - k_gain @ h_mat
        p_updated = i_kh @ p_cov @ i_kh.T + k_gain @ r_mat @ k_gain.T

        # Ép đối xứng và dương xác định
        p_updated = self._enforce_symmetric_pd(p_updated)

        sensors = set(track.contributing_sensors)
        if sensor_type is not None:
            sensors.add(sensor_type)

        new_last_updated = timestamp if timestamp is not None else track.last_updated

        return TrackState(
            track_id=track.track_id,
            state_vector=x_updated,
            covariance=p_updated,
            last_updated=new_last_updated,
            hits=track.hits,
            misses=track.misses,
            is_confirmed=track.is_confirmed,
            contributing_sensors=sensors,
        )

    def calc_bearing_jacobian(
        self,
        state_vector: np.ndarray | Sequence[float],
        ownship_pos: tuple[float, float] | Sequence[float],
    ) -> np.ndarray:
        """Tính ma trận đạo hàm riêng Jacobian H_j kích thước (1, 4) cho hàm đo góc.

        Hàm đo góc phương vị:
            h(x) = atan2(p_E - E_own, p_N - N_own)
        Đạo hàm riêng:
            d^2 = (p_N - N_own)^2 + (p_E - E_own)^2
            H_j = [- (p_E - E_own) / d^2, (p_N - N_own) / d^2, 0, 0]

        Args:
            state_vector: Vector trạng thái mục tiêu [p_N, p_E, v_N, v_E].
            ownship_pos: Tọa độ tàu mẹ [N_own, E_own] tính bằng mét.

        Returns:
            np.ndarray: Ma trận Jacobian kích thước (1, 4) dtype float64.

        Raises:
            ValueError: Nếu state_vector hoặc ownship_pos không đúng số chiều.
        """
        if len(state_vector) < 2:
            msg = (
                f"state_vector phải có ít nhất 2 phần tử, nhận được {len(state_vector)}"
            )
            logger.error(msg)
            raise ValueError(msg)

        if len(ownship_pos) != 2:
            msg = (
                f"ownship_pos phải có đúng 2 phần tử (pos_north, pos_east), "
                f"nhận được {len(ownship_pos)}"
            )
            logger.error(msg)
            raise ValueError(msg)

        delta_n = float(state_vector[0]) - float(ownship_pos[0])
        delta_e = float(state_vector[1]) - float(ownship_pos[1])
        d_sq = delta_n * delta_n + delta_e * delta_e

        # Tránh lỗi chia cho 0 khi mục tiêu trùng vị trí tàu mẹ
        d_sq = max(d_sq, 1e-12)

        dh_dpn = -delta_e / d_sq
        dh_dpe = delta_n / d_sq

        return np.array([[dh_dpn, dh_dpe, 0.0, 0.0]], dtype=np.float64)

    def calc_bearing_h(
        self,
        state_vector: np.ndarray | Sequence[float],
        ownship_pos: tuple[float, float] | Sequence[float],
    ) -> float:
        """Tính góc phương vị dự báo từ vị trí tàu mẹ tới mục tiêu (hàm phi tuyến h(x)).

        Args:
            state_vector: Vector trạng thái mục tiêu [p_N, p_E, ...].
            ownship_pos: Tọa độ tàu mẹ (pos_north, pos_east).

        Returns:
            float: Góc phương vị dự báo (radian) trong khoảng [-pi, pi].
        """
        delta_n = float(state_vector[0]) - float(ownship_pos[0])
        delta_e = float(state_vector[1]) - float(ownship_pos[1])
        return float(math.atan2(delta_e, delta_n))

    def update_bearing_1d(
        self,
        track: TrackState,
        z_bearing: float,
        ownship_pos: tuple[float, float] | Sequence[float],
        R_angle: float,  # noqa: N803
        sensor_type: SensorType | None = None,
        timestamp: float | None = None,
    ) -> TrackState:
        """Cập nhật phi tuyến EKF 1D cho cảm biến góc IR hoặc EO.

        Thực hiện chuẩn hóa sai lệch góc (innovation) qua `normalize_bearing_rad`
        để triệt tiêu hiện tượng nhảy bậc chu kỳ khi qua biên [-pi, pi].
        Áp dụng dạng Joseph ổn định số học cho cập nhật ma trận hiệp phương sai.

        Args:
            track: Trạng thái vết mục tiêu hiện tại.
            z_bearing: Góc phương vị đo được từ cảm biến (radian).
            ownship_pos: Tọa độ tàu mẹ (pos_north, pos_east) tại thời điểm đo.
            R_angle: Phương sai nhiễu đo góc (rad^2). Phải lớn hơn 0.
            sensor_type: Loại cảm biến thực hiện cập nhật (IR hoặc EO).
            timestamp: Thời điểm ghi nhận điểm đo (tùy chọn).

        Returns:
            TrackState: Bản ghi trạng thái vết mới sau khi cập nhật.

        Raises:
            ValueError: Nếu ownship_pos không đúng 2 phần tử hoặc R_angle <= 0.
        """
        if len(ownship_pos) != 2:
            msg = f"ownship_pos phải có đúng 2 phần tử, nhận được {len(ownship_pos)}"
            logger.error(msg)
            raise ValueError(msg)

        if R_angle <= 0.0:
            msg = f"Phương sai nhiễu đo góc R_angle phải > 0, nhận được: {R_angle}"
            logger.error(msg)
            raise ValueError(msg)

        x_state = track.state_vector
        p_cov = track.covariance

        # 1. Tính toán Jacobian H_j và giá trị đo dự báo h(x)
        h_jacobian = self.calc_bearing_jacobian(x_state, ownship_pos)
        h_predicted = self.calc_bearing_h(x_state, ownship_pos)

        # 2. Sai lệch đo (innovation) kèm chuẩn hóa góc qua [-pi, pi]
        innovation_angle = normalize_bearing_rad(float(z_bearing) - h_predicted)

        # 3. Innovation covariance: S = H_j * P * H_j^T + R_angle (scalar)
        s_scalar = float((h_jacobian @ p_cov @ h_jacobian.T).item()) + float(R_angle)
        s_scalar = max(s_scalar, 1e-12)

        # 4. Kalman gain: K = P * H_j^T / S (kích thước 4x1)
        k_gain = (p_cov @ h_jacobian.T) / s_scalar

        # 5. Cập nhật trạng thái: x = x + K * y
        x_updated = x_state + (k_gain * innovation_angle).flatten()

        # 6. Cập nhật ma trận hiệp phương sai theo dạng Joseph 1D:
        # P = (I - K * H_j) * P * (I - K * H_j)^T + K * [R_angle] * K^T
        eye4 = np.eye(4, dtype=np.float64)
        kh_mat = k_gain @ h_jacobian
        i_kh = eye4 - kh_mat
        r_mat = np.array([[float(R_angle)]], dtype=np.float64)
        p_updated = i_kh @ p_cov @ i_kh.T + k_gain @ r_mat @ k_gain.T

        # Ép đối xứng và kiểm tra dương xác định
        p_updated = self._enforce_symmetric_pd(p_updated)

        sensors = set(track.contributing_sensors)
        if sensor_type is not None:
            sensors.add(sensor_type)

        new_last_updated = timestamp if timestamp is not None else track.last_updated

        return TrackState(
            track_id=track.track_id,
            state_vector=x_updated,
            covariance=p_updated,
            last_updated=new_last_updated,
            hits=track.hits,
            misses=track.misses,
            is_confirmed=track.is_confirmed,
            contributing_sensors=sensors,
        )

    @staticmethod
    def _enforce_symmetric_pd(
        matrix: np.ndarray,
        min_eigenval: float = 1e-9,
    ) -> np.ndarray:
        """Ép đối xứng ma trận và đảm bảo tất cả các trị riêng đều dương xác định.

        Args:
            matrix: Ma trận hiệp phương sai kích thước (4, 4).
            min_eigenval: Ngưỡng trị riêng nhỏ nhất cho phép để đảm bảo khả nghịch.

        Returns:
            np.ndarray: Ma trận đối xứng và dương xác định.
        """
        # Ép đối xứng hoàn toàn P = 0.5 * (P + P^T)
        sym_matrix = 0.5 * (matrix + matrix.T)

        # Kiểm tra trị riêng
        eigenvalues = np.linalg.eigvalsh(sym_matrix)
        lowest_eig = float(eigenvalues[0])

        if lowest_eig < min_eigenval:
            dim = sym_matrix.shape[0]
            jitter = (min_eigenval - lowest_eig) * np.eye(dim, dtype=np.float64)
            sym_matrix = sym_matrix + jitter

        return sym_matrix
