"""Kiểm thử tính năng chuyển đổi tọa độ NED và bù chuyển động tàu mẹ (coord_ned).

Module này kiểm thử các hàm bù vector tọa độ không gian 2D, chuẩn hóa góc phương vị
về khoảng [-pi, pi], và tính góc phương vị từ tàu mẹ tới mục tiêu trong hệ Piren NED.
"""

import numpy as np
import pytest

from naval_fusion.coord_ned import (
    calc_bearing_from_ownship,
    compensate_ownship_2d,
    normalize_bearing_rad,
)


def test_compensate_ownship_2d_valid_inputs_returns_correct_ned_position() -> None:
    """Kiểm tra bù vị trí tàu mẹ với điểm đo dạng numpy ndarray."""
    # Arrange
    own_pos = (100.0, -200.0)
    meas_vessel = np.array([50.0, 30.0], dtype=np.float64)

    # Act
    target_ned = compensate_ownship_2d(own_pos, meas_vessel)

    # Assert
    assert isinstance(target_ned, np.ndarray)
    np.testing.assert_allclose(target_ned, np.array([150.0, -170.0], dtype=np.float64))


def test_compensate_ownship_2d_with_list_input_returns_ndarray() -> None:
    """Kiểm tra chấp nhận meas_vessel dạng list float và trả về ndarray float64."""
    # Arrange
    own_pos = (-50.0, 80.0)
    meas_vessel = [25.5, -10.5]

    # Act
    target_ned = compensate_ownship_2d(own_pos, meas_vessel)

    # Assert
    assert isinstance(target_ned, np.ndarray)
    assert target_ned.dtype == np.float64
    np.testing.assert_allclose(target_ned, np.array([-24.5, 69.5], dtype=np.float64))


def test_compensate_ownship_2d_invalid_shape_raises_value_error() -> None:
    """Kiểm tra ngoại lệ ValueError khi kích thước vector đầu vào không đúng 2D."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="ownship_pos phải có đúng 2 phần tử"):
        compensate_ownship_2d((100.0,), np.array([50.0, 30.0]))

    with pytest.raises(ValueError, match="vessel_meas phải có đúng 2 phần tử"):
        compensate_ownship_2d((100.0, 200.0), np.array([50.0, 30.0, 10.0]))


def test_normalize_bearing_rad_within_bounds_preserves_angle() -> None:
    """Kiểm tra các góc đã nằm trong khoảng [-pi, pi] được giữ nguyên giá trị."""
    # Arrange
    angles = [0.0, np.pi / 4, -np.pi / 3, np.pi / 2, -np.pi / 2]

    # Act & Assert
    for angle in angles:
        normalized = normalize_bearing_rad(angle)
        assert np.isclose(normalized, angle, atol=1e-12)


def test_normalize_bearing_rad_exceeding_pi_wraps_to_negative_range() -> None:
    """Kiểm tra góc vượt quá pi (pi + 0.1) được wrap về khoảng âm (-pi + 0.1)."""
    # Arrange
    angle = np.pi + 0.1

    # Act
    normalized = normalize_bearing_rad(angle)

    # Assert
    assert np.isclose(normalized, -np.pi + 0.1, atol=1e-12)


def test_normalize_bearing_rad_below_minus_pi_wraps_to_positive_range() -> None:
    """Kiểm tra góc nhỏ hơn -pi (-pi - 0.1) được wrap về khoảng dương (pi - 0.1)."""
    # Arrange
    angle = -np.pi - 0.1

    # Act
    normalized = normalize_bearing_rad(angle)

    # Assert
    assert np.isclose(normalized, np.pi - 0.1, atol=1e-12)


def test_normalize_bearing_rad_multi_rotations_normalizes_correctly() -> None:
    """Kiểm tra góc trải qua nhiều vòng quay (> 2*pi) được chuẩn hóa chuẩn xác."""
    # Arrange
    angle = 3 * np.pi + 0.25

    # Act
    normalized = normalize_bearing_rad(angle)

    # Assert
    assert np.isclose(normalized, -np.pi + 0.25, atol=1e-12)


def test_calc_bearing_from_ownship_cardinal_directions() -> None:
    """Kiểm tra tính góc phương vị từ tàu mẹ theo 4 hướng chính và hướng Đông Bắc."""
    # Arrange
    own_pos = (0.0, 0.0)

    # Act & Assert
    # Hướng Bắc (pN > 0, pE = 0) -> 0 rad
    bearing_north = calc_bearing_from_ownship(own_pos, np.array([100.0, 0.0]))
    assert np.isclose(bearing_north, 0.0, atol=1e-12)

    # Hướng Đông (pN = 0, pE > 0) -> pi / 2 rad
    bearing_east = calc_bearing_from_ownship(own_pos, np.array([0.0, 100.0]))
    assert np.isclose(bearing_east, np.pi / 2, atol=1e-12)

    # Hướng Nam (pN < 0, pE = 0) -> pi rad hoặc -pi rad
    bearing_south = calc_bearing_from_ownship(own_pos, np.array([-100.0, 0.0]))
    assert np.isclose(abs(bearing_south), np.pi, atol=1e-12)

    # Hướng Tây (pN = 0, pE < 0) -> -pi / 2 rad
    bearing_west = calc_bearing_from_ownship(own_pos, np.array([0.0, -100.0]))
    assert np.isclose(bearing_west, -np.pi / 2, atol=1e-12)

    # Hướng Đông Bắc (pN = 50, pE = 50) -> pi / 4 rad
    bearing_ne = calc_bearing_from_ownship(own_pos, np.array([50.0, 50.0]))
    assert np.isclose(bearing_ne, np.pi / 4, atol=1e-12)


def test_calc_bearing_from_ownship_with_target_state_vector_4d() -> None:
    """Kiểm tra tính góc phương vị khi mục tiêu là vector trạng thái 4D."""
    # Arrange
    own_pos = (100.0, 50.0)
    target_state = np.array([200.0, 150.0, 5.0, -2.0], dtype=np.float64)

    # Act
    bearing = calc_bearing_from_ownship(own_pos, target_state)

    # Assert
    # delta_N = 200 - 100 = 100, delta_E = 150 - 50 = 100 -> bearing = pi/4
    assert np.isclose(bearing, np.pi / 4, atol=1e-12)


def test_calc_bearing_from_ownship_colocated_returns_zero() -> None:
    """Kiểm tra khi mục tiêu trùng vị trí với tàu mẹ, góc phương vị trả về 0.0."""
    # Arrange
    own_pos = (123.4, 567.8)
    target_pos = (123.4, 567.8)

    # Act
    bearing = calc_bearing_from_ownship(own_pos, target_pos)

    # Assert
    assert np.isclose(bearing, 0.0, atol=1e-12)


def test_calc_bearing_from_ownship_invalid_dimensions_raises_value_error() -> None:
    """Kiểm tra ngoại lệ ValueError khi vị trí tàu mẹ hoặc mục tiêu không đủ 2 số."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="ownship_pos phải có đúng 2 phần tử"):
        calc_bearing_from_ownship((10.0,), np.array([100.0, 50.0]))

    with pytest.raises(ValueError, match="target_pos_ned phải có ít nhất 2 phần tử"):
        calc_bearing_from_ownship((10.0, 20.0), np.array([100.0]))
