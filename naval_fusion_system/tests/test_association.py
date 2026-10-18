"""Kiểm thử thuật toán ghép nối dữ liệu Mahalanobis Gating và phạt Cosine.

Module này kiểm thử các tính năng:
- Tính khoảng cách Mahalanobis 2D và kiểm tra ngưỡng gating Chi-bình-phương.
- Phạt chi phí theo góc lệch cosine vận tốc giữa vết và điểm đo.
- Tính khoảng cách Mahalanobis góc 1D cho cảm biến hồng ngoại / quang điện tử.
- Thuật toán phân công tối ưu Hungary (scipy.optimize.linear_sum_assignment).
- Quản lý danh sách các vết và điểm đo không được ghép nối (unmatched).
"""

import math

import numpy as np
import pytest

from naval_fusion.association import DataAssociator
from naval_fusion.models import Detection, SensorType, TrackState


@pytest.fixture
def sample_track_north() -> TrackState:
    """Tạo vết mục tiêu di chuyển về hướng Bắc với vận tốc 5 m/s."""
    return TrackState(
        track_id=1,
        state_vector=np.array([100.0, 100.0, 5.0, 0.0], dtype=np.float64),
        covariance=np.eye(4, dtype=np.float64) * 2.0,
        last_updated=0.0,
        hits=3,
        misses=0,
        is_confirmed=True,
        contributing_sensors={SensorType.RADAR},
    )


@pytest.fixture
def sample_track_east() -> TrackState:
    """Tạo vết mục tiêu di chuyển về hướng Đông với vận tốc 4 m/s."""
    return TrackState(
        track_id=2,
        state_vector=np.array([200.0, 300.0, 0.0, 4.0], dtype=np.float64),
        covariance=np.eye(4, dtype=np.float64) * 4.0,
        last_updated=0.0,
        hits=4,
        misses=0,
        is_confirmed=True,
        contributing_sensors={SensorType.RADAR, SensorType.IR},
    )


def test_calc_cost_2d_within_gate_returns_mahalanobis_distance(
    sample_track_north: TrackState,
) -> None:
    """Kiểm tra chi phí 2D trong gating trả về khoảng cách Mahalanobis hữu hạn."""
    # Arrange
    associator = DataAssociator(chi2_gate_2d=9.21)
    meas_pos = np.array([101.0, 101.0], dtype=np.float64)

    # Act
    cost = associator.calc_cost_2d(sample_track_north, meas_pos)

    # Assert
    # Chi phí phải là số thực dương và nhỏ hơn ngưỡng căn bậc 2 của gate
    assert math.isfinite(cost)
    assert 0.0 < cost < math.sqrt(9.21)


def test_calc_cost_2d_outside_gate_returns_infinity(
    sample_track_north: TrackState,
) -> None:
    """Kiểm tra điểm đo nằm ngoài cổng gating 2D trả về vô cùng (float('inf'))."""
    # Arrange
    associator = DataAssociator(chi2_gate_2d=9.21)
    # Điểm đo cách xa 100m, chắc chắn vượt ngưỡng gating 9.21
    meas_pos = np.array([200.0, 200.0], dtype=np.float64)

    # Act
    cost = associator.calc_cost_2d(sample_track_north, meas_pos)

    # Assert
    assert math.isinf(cost)


def test_calc_cost_2d_velocity_same_direction_no_penalty(
    sample_track_north: TrackState,
) -> None:
    """Kiểm tra vận tốc cùng hướng không bị phạt cosine."""
    # Arrange
    associator = DataAssociator(cosine_weight=5.0)
    meas_pos = np.array([100.0, 100.0], dtype=np.float64)
    # Vận tốc đo trùng hướng Bắc với vết
    meas_vel = np.array([5.0, 0.0], dtype=np.float64)

    # Act
    cost = associator.calc_cost_2d(sample_track_north, meas_pos, meas_vel=meas_vel)

    # Assert
    # Độ lệch vị trí bằng 0, độ lệch hướng bằng 0 -> chi phí bằng 0
    assert np.isclose(cost, 0.0, atol=1e-6)


def test_calc_cost_2d_velocity_opposite_direction_heavily_penalized(
    sample_track_north: TrackState,
) -> None:
    """Kiểm tra vận tốc ngược hướng bị phạt nặng bởi hệ số cosine."""
    # Arrange
    associator = DataAssociator(cosine_weight=5.0)
    meas_pos = np.array([100.0, 100.0], dtype=np.float64)
    meas_vel_same = np.array([5.0, 0.0], dtype=np.float64)
    meas_vel_opp = np.array([-5.0, 0.0], dtype=np.float64)

    # Act
    cost_same = associator.calc_cost_2d(
        sample_track_north, meas_pos, meas_vel=meas_vel_same
    )
    cost_opp = associator.calc_cost_2d(
        sample_track_north, meas_pos, meas_vel=meas_vel_opp
    )

    # Assert
    # Cùng hướng phạt 0, ngược hướng phạt 5.0 * (1 - (-1)) = 10.0
    assert np.isclose(cost_same, 0.0, atol=1e-6)
    assert np.isclose(cost_opp, 10.0, atol=1e-6)
    assert cost_opp > cost_same + 8.0


def test_calc_cost_2d_stationary_velocity_zero_penalty(
    sample_track_north: TrackState,
) -> None:
    """Kiểm tra khi điểm đo hoặc vết đứng yên thì không áp dụng phạt cosine."""
    # Arrange
    associator = DataAssociator(cosine_weight=5.0)
    meas_pos = np.array([100.0, 100.0], dtype=np.float64)
    meas_vel_zero = np.array([0.0, 0.0], dtype=np.float64)

    # Act
    cost = associator.calc_cost_2d(sample_track_north, meas_pos, meas_vel=meas_vel_zero)

    # Assert
    assert np.isclose(cost, 0.0, atol=1e-6)


def test_calc_cost_bearing_1d_within_gate_returns_distance(
    sample_track_north: TrackState,
) -> None:
    """Kiểm tra chi phí góc 1D trong gating trả về khoảng cách hữu hạn."""
    # Arrange
    associator = DataAssociator(chi2_gate_1d=9.0)
    ownship_pos = (0.0, 0.0)
    # Vết tại (100, 100), góc thực là pi/4 = 0.785398 rad
    true_bearing = math.atan2(100.0, 100.0)
    meas_bearing = true_bearing + 0.005  # Lệch nhỏ 5 mrad

    # Act
    cost = associator.calc_cost_bearing_1d(
        sample_track_north, meas_bearing, ownship_pos=ownship_pos, r_angle=1e-4
    )

    # Assert
    assert math.isfinite(cost)
    assert 0.0 < cost < math.sqrt(9.0)


def test_calc_cost_bearing_1d_outside_gate_returns_infinity(
    sample_track_north: TrackState,
) -> None:
    """Kiểm tra góc đo lệch lớn hơn ngưỡng gating 1D trả về vô cùng."""
    # Arrange
    associator = DataAssociator(chi2_gate_1d=9.0)
    ownship_pos = (0.0, 0.0)
    # Góc thực pi/4, điểm đo lệch 0.5 rad (khoảng 28 độ)
    true_bearing = math.atan2(100.0, 100.0)
    meas_bearing = true_bearing + 0.5

    # Act
    cost = associator.calc_cost_bearing_1d(
        sample_track_north, meas_bearing, ownship_pos=ownship_pos, r_angle=1e-4
    )

    # Assert
    assert math.isinf(cost)


def test_calc_cost_bearing_1d_wraparound_pi(
    sample_track_north: TrackState,
) -> None:
    """Kiểm tra xử lý góc tuần hoàn qua biên [-pi, pi] không bị nhảy đột biến."""
    # Arrange
    associator = DataAssociator(chi2_gate_1d=9.0)
    ownship_pos = (200.0, 100.0)
    # Target (100, 100), ownship (200, 100) -> dN = -100, dE = 0 -> Bearing = pi
    meas_bearing = -math.pi + 0.005  # Rất gần pi khi chuẩn hóa qua vòng tròn

    # Act
    cost = associator.calc_cost_bearing_1d(
        sample_track_north, meas_bearing, ownship_pos=ownship_pos, r_angle=1e-4
    )

    # Assert
    assert math.isfinite(cost)
    assert cost < math.sqrt(9.0)


def test_associate_cartesian_matches_optimal_assignment(
    sample_track_north: TrackState,
) -> None:
    """Kiểm tra ghép nối thành công 1 vết và 1 điểm đo Radar gần kề."""
    # Arrange
    associator = DataAssociator(chi2_gate_2d=9.21)
    det = Detection(
        sensor_type=SensorType.RADAR,
        timestamp=0.1,
        ownship_pos=(0.0, 0.0),
        raw_measurement=[101.0, 102.0],
        measurement_ned=np.array([101.0, 102.0], dtype=np.float64),
    )

    # Act
    matches, unmatched_trks, unmatched_dets = associator.associate(
        tracks=[sample_track_north],
        detections=[det],
        ownship_pos=(0.0, 0.0),
    )

    # Assert
    assert len(matches) == 1
    assert matches[0] == (0, 0)
    assert len(unmatched_trks) == 0
    assert len(unmatched_dets) == 0


def test_associate_multiple_tracks_and_detections_hungarian(
    sample_track_north: TrackState,
    sample_track_east: TrackState,
) -> None:
    """Kiểm tra thuật toán Hungary phân bổ tối ưu toàn cục giữa 2 vết và 2 điểm đo."""
    # Arrange
    associator = DataAssociator()
    # det0 gần track_north, det1 gần track_east
    det0 = Detection(
        sensor_type=SensorType.RADAR,
        timestamp=0.1,
        ownship_pos=(0.0, 0.0),
        raw_measurement=[100.5, 100.5],
        measurement_ned=np.array([100.5, 100.5], dtype=np.float64),
    )
    det1 = Detection(
        sensor_type=SensorType.RADAR,
        timestamp=0.1,
        ownship_pos=(0.0, 0.0),
        raw_measurement=[201.0, 299.0],
        measurement_ned=np.array([201.0, 299.0], dtype=np.float64),
    )

    # Act
    matches, unmatched_trks, unmatched_dets = associator.associate(
        tracks=[sample_track_north, sample_track_east],
        detections=[det0, det1],
        ownship_pos=(0.0, 0.0),
    )

    # Assert
    assert len(matches) == 2
    # track 0 -> det 0, track 1 -> det 1
    assert (0, 0) in matches
    assert (1, 1) in matches
    assert len(unmatched_trks) == 0
    assert len(unmatched_dets) == 0


def test_associate_gated_out_detections_unmatched(
    sample_track_north: TrackState,
) -> None:
    """Kiểm tra điểm đo quá xa bị gating loại trừ và đưa vào danh sách unmatched."""
    # Arrange
    associator = DataAssociator(chi2_gate_2d=9.21)
    det_far = Detection(
        sensor_type=SensorType.RADAR,
        timestamp=0.1,
        ownship_pos=(0.0, 0.0),
        raw_measurement=[500.0, 500.0],
        measurement_ned=np.array([500.0, 500.0], dtype=np.float64),
    )

    # Act
    matches, unmatched_trks, unmatched_dets = associator.associate(
        tracks=[sample_track_north],
        detections=[det_far],
        ownship_pos=(0.0, 0.0),
    )

    # Assert
    assert len(matches) == 0
    assert unmatched_trks == [0]
    assert unmatched_dets == [0]


def test_associate_heterogeneous_radar_and_ir(
    sample_track_north: TrackState,
    sample_track_east: TrackState,
) -> None:
    """Kiểm tra ghép nối dị thể đồng thời: Radar 2D và Camera hồng ngoại IR 1D."""
    # Arrange
    associator = DataAssociator()
    own_pos = (0.0, 0.0)

    # det_radar khớp với track_north tại (100, 100)
    det_radar = Detection(
        sensor_type=SensorType.RADAR,
        timestamp=0.1,
        ownship_pos=own_pos,
        raw_measurement=[100.5, 99.5],
        measurement_ned=np.array([100.5, 99.5], dtype=np.float64),
    )

    # det_ir là tia góc của track_east tại (200, 300) -> atan2(300, 200)
    bearing_east = math.atan2(300.0, 200.0)
    det_ir = Detection(
        sensor_type=SensorType.IR,
        timestamp=0.1,
        ownship_pos=own_pos,
        raw_measurement=[bearing_east],
        measurement_ned=np.array([bearing_east], dtype=np.float64),
    )

    # Act
    matches, unmatched_trks, unmatched_dets = associator.associate(
        tracks=[sample_track_north, sample_track_east],
        detections=[det_radar, det_ir],
        ownship_pos=own_pos,
    )

    # Assert
    assert len(matches) == 2
    assert (0, 0) in matches
    assert (1, 1) in matches
    assert len(unmatched_trks) == 0
    assert len(unmatched_dets) == 0


def test_associate_empty_inputs_returns_empty_results() -> None:
    """Kiểm tra xử lý danh sách rỗng an toàn không phát sinh ngoại lệ."""
    # Arrange
    associator = DataAssociator()

    # Act 1: Cả hai đều rỗng
    matches, un_trks, un_dets = associator.associate(
        tracks=[], detections=[], ownship_pos=(0.0, 0.0)
    )
    assert matches == []
    assert un_trks == []
    assert un_dets == []

    # Act 2: Vết có nhưng không có điểm đo
    dummy_track = TrackState(
        track_id=1,
        state_vector=np.zeros(4, dtype=np.float64),
        covariance=np.eye(4, dtype=np.float64),
        last_updated=0.0,
        hits=1,
        misses=0,
        is_confirmed=False,
        contributing_sensors={SensorType.RADAR},
    )
    matches2, un_trks2, un_dets2 = associator.associate(
        tracks=[dummy_track], detections=[], ownship_pos=(0.0, 0.0)
    )
    assert matches2 == []
    assert un_trks2 == [0]
    assert un_dets2 == []


def test_associator_invalid_parameters_raises_value_error() -> None:
    """Kiểm tra ngoại lệ ValueError khi tham số khởi tạo không hợp lệ."""
    with pytest.raises(ValueError, match="chi2_gate_2d"):
        DataAssociator(chi2_gate_2d=-1.0)

    with pytest.raises(ValueError, match="chi2_gate_1d"):
        DataAssociator(chi2_gate_1d=0.0)

    with pytest.raises(ValueError, match="cosine_weight"):
        DataAssociator(cosine_weight=-2.0)
