"""Kiểm thử tính toàn vẹn và các ràng buộc dữ liệu của module models.

Tuân thủ kiến trúc kiểm thử AAA (Arrange, Act, Assert) và quy chuẩn kiểm thử dự án.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.naval_fusion.models import (
    Detection,
    EvaluationMetrics,
    OwnshipState,
    SensorType,
    TrackState,
)


def test_sensor_type_enum_matches_autoferry_spec() -> None:
    """Kiểm tra giá trị enum của các loại cảm biến chuẩn theo thiết kế."""
    # Arrange & Act & Assert
    assert int(SensorType.LIDAR) == 1
    assert int(SensorType.RADAR) == 2
    assert int(SensorType.IR) == 3
    assert int(SensorType.EO) == 4


def test_ownship_state_immutability_raises_frozen_error() -> None:
    """Kiểm tra tính bất biến của OwnshipState khi cố tình gán lại giá trị."""
    # Arrange
    state = OwnshipState(timestamp=100.0, pos_north=10.5, pos_east=20.2)

    # Act & Assert
    assert state.timestamp == 100.0
    assert state.pos_north == 10.5
    assert state.pos_east == 20.2
    with pytest.raises(FrozenInstanceError):
        setattr(state, "pos_north", 15.0)  # noqa: B010


def test_detection_creation_with_valid_fields_stores_correctly() -> None:
    """Kiểm tra khởi tạo đối tượng Detection với các trường dữ liệu hợp lệ."""
    # Arrange
    sensor = SensorType.RADAR
    timestamp = 100.5
    ownship_pos = (15.0, 25.0)
    raw_measurement = [50.0, 30.0]
    measurement_ned = np.array([65.0, 55.0], dtype=np.float64)

    # Act
    detection = Detection(
        sensor_type=sensor,
        timestamp=timestamp,
        ownship_pos=ownship_pos,
        raw_measurement=raw_measurement,
        measurement_ned=measurement_ned,
    )

    # Assert
    assert detection.sensor_type == SensorType.RADAR
    assert detection.timestamp == 100.5
    assert detection.ownship_pos == (15.0, 25.0)
    assert detection.raw_measurement == [50.0, 30.0]
    assert np.array_equal(detection.measurement_ned, measurement_ned)


def test_track_state_creation_maintains_covariance_and_state() -> None:
    """Kiểm tra khởi tạo TrackState với vector trạng thái và ma trận hiệp phương sai."""
    # Arrange
    track_id = 42
    state_vec = np.array([100.0, 200.0, 1.5, -0.5], dtype=np.float64)
    cov = np.eye(4, dtype=np.float64) * 10.0
    last_updated = 12345.67
    hits = 3
    misses = 0
    is_confirmed = True
    contributing = {SensorType.RADAR, SensorType.IR}

    # Act
    track = TrackState(
        track_id=track_id,
        state_vector=state_vec,
        covariance=cov,
        last_updated=last_updated,
        hits=hits,
        misses=misses,
        is_confirmed=is_confirmed,
        contributing_sensors=contributing,
    )

    # Assert
    assert track.track_id == 42
    assert track.is_confirmed is True
    assert track.hits == 3
    assert track.misses == 0
    assert SensorType.IR in track.contributing_sensors
    assert np.array_equal(track.state_vector, state_vec)
    assert np.array_equal(track.covariance, cov)


def test_evaluation_metrics_stores_quantitative_indicators() -> None:
    """Kiểm tra cấu trúc chứa chỉ số đánh giá định lượng EvaluationMetrics."""
    # Arrange & Act
    metrics = EvaluationMetrics(
        rmse_position=2.45,
        rmse_velocity=0.35,
        num_confirmed_tracks=2,
        id_switches=0,
        mostly_tracked_ratio=1.0,
        details={"scenario": "scenario2"},
    )

    # Assert
    assert metrics.rmse_position == 2.45
    assert metrics.rmse_velocity == 0.35
    assert metrics.num_confirmed_tracks == 2
    assert metrics.id_switches == 0
    assert metrics.mostly_tracked_ratio == 1.0
    assert metrics.details["scenario"] == "scenario2"


def test_package_exports_all_models() -> None:
    """Kiểm tra package src.naval_fusion export đầy đủ các mô hình dữ liệu chính."""
    # Arrange & Act
    import src.naval_fusion as nf

    # Assert
    assert hasattr(nf, "SensorType")
    assert hasattr(nf, "OwnshipState")
    assert hasattr(nf, "Detection")
    assert hasattr(nf, "TrackState")
    assert hasattr(nf, "EvaluationMetrics")
