"""Kiểm thử đơn vị cho thuật toán bám bắt đa mục tiêu MSTrack và KalmanBoxTracker."""

import pytest

from src.config import Modality
from src.kalman_tracker import KalmanBoxTracker
from src.mstrack import MSTrack


def test_kalman_box_tracker_predict_and_update_maintains_consistent_state() -> None:
    # Arrange
    bbox = [100.0, 100.0, 50.0, 20.0]
    tracker = KalmanBoxTracker(bbox, track_id=1)

    # Act
    pred_bbox = tracker.predict()

    # Assert
    assert len(pred_bbox) == 4
    assert pytest.approx(pred_bbox[0], abs=5.0) == 100.0

    # Act 2 (Update)
    new_bbox = [105.0, 102.0, 50.0, 20.0]
    tracker.update(new_bbox, conf=0.95)

    # Assert 2
    assert tracker.hits == 2
    assert tracker.time_since_update == 0


def test_mstrack_feedback_visible_redetection_recovers_missing_track() -> None:
    # Arrange
    mstrack = MSTrack(modality=Modality.VISIBLE, max_age=30)
    for t in range(5):
        dets: list[dict[str, object]] = [
            {"bbox": [100.0 + t * 5, 200.0, 80.0, 30.0], "conf": 0.95}
        ]
        mstrack.update(dets, frame_idx=t, ts=1000 + t * 1000)

    # Act
    tracks = mstrack.get_active_tracks()
    assert len(tracks) == 1
    t_id = tracks[0].track_id

    # Mô phỏng mất phát hiện ở t=5 nhưng có cổng bám bắt từ radar
    radar_gates = {t_id: [120.0, 195.0, 90.0, 40.0]}
    recovered_tracks = mstrack.update([], frame_idx=5, ts=6000, radar_gates=radar_gates)

    # Assert
    assert len(recovered_tracks) == 1
    assert recovered_tracks[0].track_id == t_id
    assert recovered_tracks[0].time_since_update == 0
