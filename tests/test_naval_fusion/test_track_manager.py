"""Kiểm thử vòng đời vết mục tiêu chuẩn quân sự và luật M/N (TrackManager).

Tuân thủ kiến trúc kiểm thử AAA (Arrange, Act, Assert) và quy chuẩn kiểm thử dự án.
"""

import numpy as np
import pytest

from src.naval_fusion.models import Detection, SensorType, TrackState
from src.naval_fusion.track_manager import (
    DEFAULT_CONFIRM_HITS,
    DEFAULT_MAX_COASTING_TIME,
    DEFAULT_MAX_MISSES,
    DEFAULT_WINDOW_SIZE,
    TrackManager,
)


def test_ir_bearing_cannot_init_track_returns_none() -> None:
    """Kiểm tra nguyên tắc quân sự: Cảm biến góc IR 1D không được tự tạo vết."""
    # Arrange
    manager = TrackManager()
    ir_det = Detection(
        sensor_type=SensorType.IR,
        timestamp=100.0,
        ownship_pos=(10.0, 20.0),
        raw_measurement=[0.45],
        measurement_ned=np.array([0.45], dtype=np.float64),
    )

    # Act
    track = manager.init_track(ir_det)

    # Assert
    assert track is None


def test_eo_bearing_cannot_init_track_returns_none() -> None:
    """Kiểm tra nguyên tắc quân sự: Cảm biến quang EO 1D không được tạo vết."""
    # Arrange
    manager = TrackManager()
    eo_det = Detection(
        sensor_type=SensorType.EO,
        timestamp=100.0,
        ownship_pos=(10.0, 20.0),
        raw_measurement=[-0.25],
        measurement_ned=np.array([-0.25], dtype=np.float64),
    )

    # Act
    track = manager.init_track(eo_det)

    # Assert
    assert track is None


def test_radar_inits_tentative_track_with_zero_velocity() -> None:
    """Kiểm tra Radar 2D khởi tạo vết nghi ngờ với vận tốc ban đầu bằng 0."""
    # Arrange
    manager = TrackManager()
    radar_det = Detection(
        sensor_type=SensorType.RADAR,
        timestamp=100.0,
        ownship_pos=(0.0, 0.0),
        raw_measurement=[150.0, 250.0],
        measurement_ned=np.array([150.0, 250.0], dtype=np.float64),
    )

    # Act
    track = manager.init_track(radar_det)

    # Assert
    assert track is not None
    assert track.track_id == 1
    assert track.is_confirmed is False
    assert track.hits == 1
    assert track.misses == 0
    assert track.last_updated == 100.0
    assert float(track.state_vector[0]) == pytest.approx(150.0)
    assert float(track.state_vector[1]) == pytest.approx(250.0)
    assert float(track.state_vector[2]) == pytest.approx(0.0)
    assert float(track.state_vector[3]) == pytest.approx(0.0)
    assert track.covariance.shape == (4, 4)


def test_lidar_inits_tentative_track_with_incremented_id() -> None:
    """Kiểm tra LiDAR 2D khởi tạo vết và ID của vết tăng tuần tự."""
    # Arrange
    manager = TrackManager()
    det1 = Detection(
        sensor_type=SensorType.RADAR,
        timestamp=10.0,
        ownship_pos=(0.0, 0.0),
        raw_measurement=[10.0, 20.0],
        measurement_ned=np.array([10.0, 20.0], dtype=np.float64),
    )
    det2 = Detection(
        sensor_type=SensorType.LIDAR,
        timestamp=10.0,
        ownship_pos=(0.0, 0.0),
        raw_measurement=[30.0, 40.0],
        measurement_ned=np.array([30.0, 40.0], dtype=np.float64),
    )

    # Act
    track1 = manager.init_track(det1)
    track2 = manager.init_track(det2)

    # Assert
    assert track1 is not None and track1.track_id == 1
    assert track2 is not None and track2.track_id == 2
    assert track2.contributing_sensors == {SensorType.LIDAR}


def test_init_track_invalid_measurement_dimension_returns_none() -> None:
    """Kiểm tra điểm đo kích thước không hợp lệ (< 2) sẽ bị từ chối an toàn."""
    # Arrange
    manager = TrackManager()
    invalid_det = Detection(
        sensor_type=SensorType.RADAR,
        timestamp=10.0,
        ownship_pos=(0.0, 0.0),
        raw_measurement=[100.0],
        measurement_ned=np.array([100.0], dtype=np.float64),
    )

    # Act
    track = manager.init_track(invalid_det)

    # Assert
    assert track is None


def test_mn_rule_confirmation_lifecycle() -> None:
    """Kiểm tra quy tắc M/N: Chuyển sang Confirmed khi đủ confirm_hits."""
    # Arrange
    manager = TrackManager(confirm_hits=3, window_size=5)
    rad_det = Detection(
        sensor_type=SensorType.RADAR,
        timestamp=1.0,
        ownship_pos=(0.0, 0.0),
        raw_measurement=[100.0, 200.0],
        measurement_ned=np.array([100.0, 200.0], dtype=np.float64),
    )
    track = manager.init_track(rad_det)
    assert track is not None
    assert track.hits == 1
    assert track.is_confirmed is False
    tracks = [track]

    # Act - Lần trúng thứ 2 (chưa đủ 3)
    tracks = manager.update_matches(tracks, [(0, 0)], [rad_det], current_time=2.0)

    # Assert - Lần 2
    assert tracks[0].hits == 2
    assert tracks[0].misses == 0
    assert tracks[0].last_updated == 2.0
    assert tracks[0].is_confirmed is False

    # Act - Lần trúng thứ 3 (đủ 3/5 -> Confirmed!)
    tracks = manager.update_matches(tracks, [(0, 0)], [rad_det], current_time=3.0)

    # Assert - Lần 3
    assert tracks[0].hits == 3
    assert tracks[0].misses == 0
    assert tracks[0].last_updated == 3.0
    assert tracks[0].is_confirmed is True


def test_update_matches_accumulates_contributing_sensors() -> None:
    """Kiểm tra điểm đo ghép vào vết giúp tích lũy contributing_sensors."""
    # Arrange
    manager = TrackManager()
    rad_det = Detection(
        sensor_type=SensorType.RADAR,
        timestamp=1.0,
        ownship_pos=(0.0, 0.0),
        raw_measurement=[100.0, 200.0],
        measurement_ned=np.array([100.0, 200.0], dtype=np.float64),
    )
    ir_det = Detection(
        sensor_type=SensorType.IR,
        timestamp=1.5,
        ownship_pos=(0.0, 0.0),
        raw_measurement=[0.5],
        measurement_ned=np.array([0.5], dtype=np.float64),
    )
    track = manager.init_track(rad_det)
    assert track is not None
    tracks = [track]

    # Act
    tracks = manager.update_matches(tracks, [(0, 0)], [ir_det], current_time=1.5)

    # Assert
    assert tracks[0].contributing_sensors == {SensorType.RADAR, SensorType.IR}
    assert tracks[0].hits == 2
    assert tracks[0].misses == 0


def test_update_unmatched_increments_misses() -> None:
    """Kiểm tra vết không được ghép nối sẽ bị tăng số lần trượt misses."""
    # Arrange
    manager = TrackManager(max_misses=5, max_coasting_time=10.0)
    track = TrackState(
        track_id=1,
        state_vector=np.array([100.0, 200.0, 1.0, 0.5], dtype=np.float64),
        covariance=np.eye(4, dtype=np.float64) * 5.0,
        last_updated=1.0,
        hits=3,
        misses=0,
        is_confirmed=True,
        contributing_sensors={SensorType.RADAR},
    )
    tracks = [track]

    # Act - Trượt lần 1
    surviving = manager.update_unmatched(tracks, [0], current_time=2.0)

    # Assert
    assert len(surviving) == 1
    assert surviving[0].misses == 1

    # Act - Trượt lần 2
    surviving = manager.update_unmatched(surviving, [0], current_time=3.0)

    # Assert
    assert len(surviving) == 1
    assert surviving[0].misses == 2


def test_track_deleted_when_max_misses_exceeded() -> None:
    """Kiểm tra vết bị xóa khi số lần trượt liên tiếp đạt max_misses."""
    # Arrange
    manager = TrackManager(max_misses=3, max_coasting_time=20.0)
    track = TrackState(
        track_id=1,
        state_vector=np.array([100.0, 200.0, 1.0, 0.5], dtype=np.float64),
        covariance=np.eye(4, dtype=np.float64) * 5.0,
        last_updated=1.0,
        hits=3,
        misses=2,
        is_confirmed=True,
        contributing_sensors={SensorType.RADAR},
    )
    tracks = [track]

    # Act - misses tăng từ 2 lên 3 (bằng max_misses=3)
    surviving = manager.update_unmatched(tracks, [0], current_time=2.0)

    # Assert
    assert len(surviving) == 0


def test_track_deleted_when_max_coasting_time_exceeded() -> None:
    """Kiểm tra vết bị xóa khi thời gian trôi vượt quá max_coasting_time."""
    # Arrange
    manager = TrackManager(max_misses=10, max_coasting_time=5.0)
    track = TrackState(
        track_id=1,
        state_vector=np.array([100.0, 200.0, 1.0, 0.5], dtype=np.float64),
        covariance=np.eye(4, dtype=np.float64) * 5.0,
        last_updated=1.0,
        hits=3,
        misses=1,
        is_confirmed=True,
        contributing_sensors={SensorType.RADAR},
    )
    tracks = [track]

    # Act - current_time = 6.5s -> dt = 5.5s > 5.0s
    surviving = manager.update_unmatched(tracks, [0], current_time=6.5)

    # Assert
    assert len(surviving) == 0


def test_step_full_lifecycle_with_6_arguments() -> None:
    """Kiểm tra step toàn diện: cập nhật vết, sinh vết 2D mới và chặn IR."""
    # Arrange
    manager = TrackManager(confirm_hits=3, max_misses=3, max_coasting_time=5.0)
    # Vết 1 đang bám (sẽ trúng)
    track1 = TrackState(
        track_id=1,
        state_vector=np.array([100.0, 100.0, 0.0, 0.0], dtype=np.float64),
        covariance=np.eye(4, dtype=np.float64) * 5.0,
        last_updated=1.0,
        hits=2,
        misses=0,
        is_confirmed=False,
        contributing_sensors={SensorType.RADAR},
    )
    # Vết 2 đang trượt (sẽ bị xóa vì misses lên 3)
    track2 = TrackState(
        track_id=2,
        state_vector=np.array([500.0, 500.0, 0.0, 0.0], dtype=np.float64),
        covariance=np.eye(4, dtype=np.float64) * 5.0,
        last_updated=1.0,
        hits=1,
        misses=2,
        is_confirmed=False,
        contributing_sensors={SensorType.RADAR},
    )
    tracks = [track1, track2]

    # Điểm đo: det0 ghép với track1, det1 là radar chưa ghép, det2 là IR chưa ghép
    det0 = Detection(
        sensor_type=SensorType.RADAR,
        timestamp=2.0,
        ownship_pos=(0.0, 0.0),
        raw_measurement=[102.0, 101.0],
        measurement_ned=np.array([102.0, 101.0], dtype=np.float64),
    )
    det1 = Detection(
        sensor_type=SensorType.RADAR,
        timestamp=2.0,
        ownship_pos=(0.0, 0.0),
        raw_measurement=[300.0, 400.0],
        measurement_ned=np.array([300.0, 400.0], dtype=np.float64),
    )
    det2 = Detection(
        sensor_type=SensorType.IR,
        timestamp=2.0,
        ownship_pos=(0.0, 0.0),
        raw_measurement=[0.8],
        measurement_ned=np.array([0.8], dtype=np.float64),
    )
    detections = [det0, det1, det2]
    matches = [(0, 0)]
    unmatched_tracks = [1]  # track2 trượt
    unmatched_detections = [1, 2]  # det1 (radar) và det2 (IR)

    # Act
    surviving = manager.step(
        tracks=tracks,
        matches=matches,
        unmatched_tracks=unmatched_tracks,
        unmatched_detections=unmatched_detections,
        detections=detections,
        current_time=2.0,
    )

    # Assert
    # track2 bị xóa do misses >= 3.
    # track1 sống và đạt hits=3 -> is_confirmed = True.
    # det1 (radar) sinh ra vết mới.
    # det2 (IR) bị chặn không sinh vết.
    # Tổng cộng có 2 vết còn sống (track1 và vết mới từ det1).
    assert len(surviving) == 2
    confirmed = [t for t in surviving if t.is_confirmed]
    tentative = [t for t in surviving if not t.is_confirmed]
    assert len(confirmed) == 1
    assert confirmed[0].track_id == 1
    assert confirmed[0].hits == 3

    assert len(tentative) == 1
    assert tentative[0].hits == 1
    assert SensorType.RADAR in tentative[0].contributing_sensors


def test_step_convenience_with_4_arguments() -> None:
    """Kiểm tra gọi step theo cú pháp ngắn gọn 4 tham số tự suy luận unmatched."""
    # Arrange
    manager = TrackManager(confirm_hits=3, max_misses=5)
    det_radar = Detection(
        sensor_type=SensorType.RADAR,
        timestamp=1.0,
        ownship_pos=(0.0, 0.0),
        raw_measurement=[100.0, 200.0],
        measurement_ned=np.array([100.0, 200.0], dtype=np.float64),
    )

    # Act - Khởi tạo từ step khi chưa có vết nào
    surviving = manager.step(
        tracks=[],
        matches=[],
        detections=[det_radar],
        current_time=1.0,
    )

    # Assert
    assert len(surviving) == 1
    assert surviving[0].track_id == 1
    assert surviving[0].hits == 1
    assert surviving[0].is_confirmed is False


def test_track_manager_invalid_parameters_raise_value_error() -> None:
    """Kiểm tra cấu hình tham số sai sẽ báo lỗi ValueError."""
    # Act & Assert
    with pytest.raises(ValueError, match="confirm_hits"):
        TrackManager(confirm_hits=0)

    with pytest.raises(ValueError, match="window_size"):
        TrackManager(confirm_hits=5, window_size=3)

    with pytest.raises(ValueError, match="max_misses"):
        TrackManager(max_misses=0)

    with pytest.raises(ValueError, match="max_coasting_time"):
        TrackManager(max_coasting_time=-1.0)


def test_get_confirmed_and_tentative_tracks_helper() -> None:
    """Kiểm tra các phương thức lọc vết xác nhận và vết nghi ngờ."""
    # Arrange
    manager = TrackManager()
    t1 = TrackState(
        track_id=1,
        state_vector=np.zeros(4),
        covariance=np.eye(4),
        last_updated=0.0,
        hits=3,
        misses=0,
        is_confirmed=True,
        contributing_sensors={SensorType.RADAR},
    )
    t2 = TrackState(
        track_id=2,
        state_vector=np.zeros(4),
        covariance=np.eye(4),
        last_updated=0.0,
        hits=1,
        misses=0,
        is_confirmed=False,
        contributing_sensors={SensorType.RADAR},
    )
    tracks = [t1, t2]

    # Act & Assert
    assert len(manager.get_confirmed_tracks(tracks)) == 1
    assert manager.get_confirmed_tracks(tracks)[0].track_id == 1
    assert len(manager.get_tentative_tracks(tracks)) == 1
    assert manager.get_tentative_tracks(tracks)[0].track_id == 2


def test_track_manager_reset_restores_state() -> None:
    """Kiểm tra phương thức reset đưa bộ quản lý vết về trạng thái ban đầu."""
    # Arrange
    manager = TrackManager()
    det = Detection(
        sensor_type=SensorType.RADAR,
        timestamp=1.0,
        ownship_pos=(0.0, 0.0),
        raw_measurement=[10.0, 20.0],
        measurement_ned=np.array([10.0, 20.0], dtype=np.float64),
    )
    t = manager.init_track(det)
    assert t is not None and t.track_id == 1
    assert manager.next_track_id == 2

    # Act
    manager.reset(initial_track_id=1)

    # Assert
    assert manager.next_track_id == 1
    t_new = manager.init_track(det)
    assert t_new is not None and t_new.track_id == 1


def test_default_constants_match_naval_standards() -> None:
    """Kiểm tra các hằng số mặc định tuân thủ quy chuẩn tác chiến hải quân."""
    # Arrange & Act & Assert
    assert DEFAULT_CONFIRM_HITS == 3
    assert DEFAULT_WINDOW_SIZE == 5
    assert DEFAULT_MAX_MISSES == 5
    assert DEFAULT_MAX_COASTING_TIME == 5.0
