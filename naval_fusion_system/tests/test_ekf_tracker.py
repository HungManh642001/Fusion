"""Kiểm thử bộ lọc Kalman mở rộng dị thể (HeterogeneousEKF).

Module này kiểm thử tính năng ngoại suy động học (predict) theo mô hình
vận tốc không đổi CV (Constant Velocity), cập nhật tọa độ Descartes 2D
tuyến tính cho Radar/LiDAR, và cập nhật góc phương vị 1D phi tuyến cho
Camera ảnh nhiệt IR / Quang điện tử EO.
"""

import numpy as np
import pytest

from naval_fusion.ekf_tracker import HeterogeneousEKF
from naval_fusion.models import SensorType, TrackState


@pytest.fixture
def sample_track() -> TrackState:
    """Khởi tạo một vết mục tiêu mẫu với ma trận hiệp phương sai xác định."""
    return TrackState(
        track_id=1,
        state_vector=np.array([100.0, 200.0, 2.0, 1.0], dtype=np.float64),
        covariance=np.eye(4, dtype=np.float64) * 10.0,
        last_updated=0.0,
        hits=1,
        misses=0,
        is_confirmed=True,
        contributing_sensors={SensorType.RADAR},
    )


def test_ekf_predict_constant_velocity_updates_position_and_covariance(
    sample_track: TrackState,
) -> None:
    """Kiểm tra ngoại suy động học: vị trí theo vận tốc và hiệp phương sai tăng."""
    # Arrange
    ekf = HeterogeneousEKF(q_noise=0.5)
    dt = 2.0

    # Act
    pred_track = ekf.predict(sample_track, dt=dt)

    # Assert
    # Vị trí mới p_N = 100 + 2.0 * 2 = 104.0; p_E = 200 + 1.0 * 2 = 202.0
    assert np.isclose(pred_track.state_vector[0], 104.0, atol=1e-9)
    assert np.isclose(pred_track.state_vector[1], 202.0, atol=1e-9)
    assert np.isclose(pred_track.state_vector[2], 2.0, atol=1e-9)
    assert np.isclose(pred_track.state_vector[3], 1.0, atol=1e-9)

    # Hiệp phương sai vị trí phải tăng sau bước dự báo do có nhiễu quá trình Q
    assert pred_track.covariance[0, 0] > sample_track.covariance[0, 0]
    assert pred_track.covariance[1, 1] > sample_track.covariance[1, 1]

    # Ma trận phải giữ tính đối xứng và các trị riêng dương
    np.testing.assert_allclose(
        pred_track.covariance,
        pred_track.covariance.T,
        atol=1e-12,
    )
    eigenvalues = np.linalg.eigvalsh(pred_track.covariance)
    assert np.all(eigenvalues > 0)

    # Thời gian cập nhật phải được cộng thêm dt
    assert np.isclose(
        pred_track.last_updated,
        sample_track.last_updated + dt,
        atol=1e-9,
    )


def test_ekf_predict_zero_dt_preserves_state_and_covariance(
    sample_track: TrackState,
) -> None:
    """Kiểm tra ngoại suy với dt = 0 giữ nguyên trạng thái và hiệp phương sai."""
    # Arrange
    ekf = HeterogeneousEKF(q_noise=0.5)

    # Act
    pred_track = ekf.predict(sample_track, dt=0.0)

    # Assert
    np.testing.assert_allclose(
        pred_track.state_vector,
        sample_track.state_vector,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        pred_track.covariance,
        sample_track.covariance,
        atol=1e-12,
    )
    assert np.isclose(pred_track.last_updated, sample_track.last_updated, atol=1e-12)


def test_ekf_predict_negative_dt_raises_value_error(
    sample_track: TrackState,
) -> None:
    """Kiểm tra ngoại lệ ValueError khi dt âm."""
    # Arrange
    ekf = HeterogeneousEKF(q_noise=0.5)

    # Act & Assert
    with pytest.raises(ValueError, match="dt"):
        ekf.predict(sample_track, dt=-0.5)


def test_ekf_update_cartesian_decreases_covariance_and_maintains_symmetry(
    sample_track: TrackState,
) -> None:
    """Kiểm tra cập nhật tuyến tính 2D: hiệp phương sai giảm và duy trì đối xứng."""
    # Arrange
    ekf = HeterogeneousEKF()
    z = np.array([102.0, 198.0], dtype=np.float64)
    r_cov = np.eye(2, dtype=np.float64) * 4.0

    # Act
    updated_track = ekf.update_cartesian_2d(sample_track, z, R_cov=r_cov)

    # Assert
    # Hiệp phương sai phải giảm sau khi có điểm đo
    assert updated_track.covariance[0, 0] < sample_track.covariance[0, 0]
    assert updated_track.covariance[1, 1] < sample_track.covariance[1, 1]

    # Ma trận hiệp phương sai phải đối xứng
    np.testing.assert_allclose(
        updated_track.covariance,
        updated_track.covariance.T,
        atol=1e-12,
    )

    # Trị riêng phải dương xác định
    eigenvalues = np.linalg.eigvalsh(updated_track.covariance)
    assert np.all(eigenvalues > 0)

    # Tọa độ ước lượng được kéo về phía điểm đo
    assert updated_track.state_vector[0] > sample_track.state_vector[0]
    assert updated_track.state_vector[1] < sample_track.state_vector[1]


def test_ekf_update_cartesian_with_metadata_updates_sensors_and_timestamp(
    sample_track: TrackState,
) -> None:
    """Kiểm tra truyền sensor_type và timestamp vào update_cartesian_2d."""
    # Arrange
    ekf = HeterogeneousEKF()
    z = np.array([101.0, 201.0], dtype=np.float64)
    r_cov = np.eye(2, dtype=np.float64) * 25.0

    # Act
    updated = ekf.update_cartesian_2d(
        sample_track,
        z,
        R_cov=r_cov,
        sensor_type=SensorType.LIDAR,
        timestamp=5.5,
    )

    # Assert
    assert updated.last_updated == 5.5
    assert SensorType.LIDAR in updated.contributing_sensors
    assert SensorType.RADAR in updated.contributing_sensors


def test_ekf_update_cartesian_invalid_dimensions_raises_value_error(
    sample_track: TrackState,
) -> None:
    """Kiểm tra ngoại lệ ValueError khi z hoặc R_cov sai kích thước."""
    # Arrange
    ekf = HeterogeneousEKF()

    # Act & Assert
    with pytest.raises(ValueError, match="z_2d"):
        ekf.update_cartesian_2d(sample_track, np.array([100.0]), R_cov=np.eye(2))

    with pytest.raises(ValueError, match="R_cov"):
        ekf.update_cartesian_2d(sample_track, np.array([100.0, 200.0]), R_cov=np.eye(3))


def test_ekf_update_bearing_only_maintains_symmetry_and_positive_definiteness(
    sample_track: TrackState,
) -> None:
    """Kiểm tra cập nhật phi tuyến EKF 1D: ma trận đối xứng và dương xác định."""
    # Arrange
    ekf = HeterogeneousEKF()
    own_pos = (0.0, 0.0)
    # Góc thực tế tới (100, 200) là atan2(200, 100)
    true_angle = float(np.arctan2(200.0, 100.0))
    # Điểm đo bị lệch nhẹ +0.02 rad
    z_angle = true_angle + 0.02
    r_angle = 1e-4

    # Act
    updated_track = ekf.update_bearing_1d(
        sample_track,
        z_bearing=z_angle,
        ownship_pos=own_pos,
        R_angle=r_angle,
    )

    # Assert
    np.testing.assert_allclose(
        updated_track.covariance,
        updated_track.covariance.T,
        atol=1e-12,
    )
    eigenvalues = np.linalg.eigvalsh(updated_track.covariance)
    assert np.all(eigenvalues > 0)

    # Vị trí mới phải dịch chuyển góc về phía điểm đo (góc ước lượng tăng lên)
    updated_angle = float(
        np.arctan2(updated_track.state_vector[1], updated_track.state_vector[0]),
    )
    assert updated_angle > true_angle


def test_ekf_update_bearing_wraparound_handles_boundary_correctly() -> None:
    """Kiểm tra xử lý biên [-pi, pi]: góc đo nhảy qua biên không gây nổ bộ lọc."""
    # Arrange
    ekf = HeterogeneousEKF()
    # Tàu mẹ tại gốc (0, 0), mục tiêu ở hướng Tây Bắc gần -pi (-179 deg)
    angle_pred = np.deg2rad(179.0)  # +179 độ
    dist = 500.0
    p_n = dist * np.cos(angle_pred)
    p_e = dist * np.sin(angle_pred)

    track = TrackState(
        track_id=2,
        state_vector=np.array([p_n, p_e, 0.0, 0.0], dtype=np.float64),
        covariance=np.eye(4, dtype=np.float64) * 20.0,
        last_updated=1.0,
        hits=3,
        misses=0,
        is_confirmed=True,
        contributing_sensors={SensorType.RADAR},
    )

    # Điểm đo góc thực tế là -179 độ (nhảy qua mốc pi)
    angle_meas = np.deg2rad(-179.0)  # -179 độ

    # Act
    updated = ekf.update_bearing_1d(
        track,
        z_bearing=angle_meas,
        ownship_pos=(0.0, 0.0),
        R_angle=1e-4,
    )

    # Assert
    # Bộ lọc không bị nổ giá trị (không NaN, không inf)
    assert np.all(np.isfinite(updated.state_vector))
    assert np.all(np.isfinite(updated.covariance))

    # Góc phương vị sau cập nhật phải dịch chuyển nhẹ về phía -179 deg
    updated_angle = float(np.arctan2(updated.state_vector[1], updated.state_vector[0]))
    # Khoảng cách góc giữa updated và angle_pred phải nhỏ (< 5 độ)
    angle_diff = np.arctan2(
        np.sin(updated_angle - angle_pred),
        np.cos(updated_angle - angle_pred),
    )
    angular_step = abs(float(angle_diff))
    assert angular_step < np.deg2rad(5.0)


def test_ekf_update_bearing_invalid_inputs_raises_value_error(
    sample_track: TrackState,
) -> None:
    """Kiểm tra ngoại lệ ValueError khi ownship_pos hoặc R_angle không hợp lệ."""
    # Arrange
    ekf = HeterogeneousEKF()

    # Act & Assert
    with pytest.raises(ValueError, match="ownship_pos"):
        ekf.update_bearing_1d(
            sample_track,
            z_bearing=0.5,
            ownship_pos=(0.0,),  # Chỉ có 1 phần tử
            R_angle=1e-4,
        )

    with pytest.raises(ValueError, match="R_angle"):
        ekf.update_bearing_1d(
            sample_track,
            z_bearing=0.5,
            ownship_pos=(0.0, 0.0),
            R_angle=-1e-4,  # R âm
        )


def test_ekf_init_invalid_q_noise_raises_value_error() -> None:
    """Kiểm tra ngoại lệ khi q_noise <= 0."""
    # Act & Assert
    with pytest.raises(ValueError, match="q_noise"):
        HeterogeneousEKF(q_noise=0.0)

    with pytest.raises(ValueError, match="q_noise"):
        HeterogeneousEKF(q_noise=-1.0)


def test_ekf_jacobian_matches_finite_differences() -> None:
    """Kiểm tra độ chính xác của Jacobian H_j bằng giải thuật sai phân hữu hạn."""
    # Arrange
    ekf = HeterogeneousEKF()
    p_n, p_e = 150.0, -80.0
    own_n, own_e = 20.0, 30.0
    own_pos = (own_n, own_e)
    state = np.array([p_n, p_e, 5.0, -3.0], dtype=np.float64)

    # Act
    h_j = ekf.calc_bearing_jacobian(state, own_pos)
    assert h_j.shape == (1, 4)

    # Tính gradient số học (finite difference)
    eps = 1e-6
    # Đạo hàm theo pN
    h_plus_n = float(np.arctan2(p_e - own_e, (p_n + eps) - own_n))
    h_minus_n = float(np.arctan2(p_e - own_e, (p_n - eps) - own_n))
    dh_dpn = (h_plus_n - h_minus_n) / (2 * eps)

    # Đạo hàm theo pE
    h_plus_e = float(np.arctan2((p_e + eps) - own_e, p_n - own_n))
    h_minus_e = float(np.arctan2((p_e - eps) - own_e, p_n - own_n))
    dh_dpe = (h_plus_e - h_minus_e) / (2 * eps)

    # Assert
    assert np.isclose(h_j[0, 0], dh_dpn, rtol=1e-5)
    assert np.isclose(h_j[0, 1], dh_dpe, rtol=1e-5)
    assert h_j[0, 2] == 0.0
    assert h_j[0, 3] == 0.0
