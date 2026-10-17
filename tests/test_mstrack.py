import pytest
import numpy as np
from src.kalman_tracker import KalmanBoxTracker
from src.mstrack import MSTrack, Track

def test_kalman_box_tracker_predict_and_update():
    bbox = [100.0, 100.0, 50.0, 20.0]  # [x, y, w, h]
    tracker = KalmanBoxTracker(bbox, track_id=1)
    
    # Predict
    pred_bbox = tracker.predict()
    assert len(pred_bbox) == 4
    assert pytest.approx(pred_bbox[0], abs=5.0) == 100.0
    
    # Update with moved bbox
    new_bbox = [105.0, 102.0, 50.0, 20.0]
    tracker.update(new_bbox, conf=0.95)
    assert tracker.hits == 2
    assert tracker.time_since_update == 0

def test_mstrack_feedback_visible_redetection():
    mstrack = MSTrack(modality='visible', max_age=30)
    
    # Feed 5 detections to establish high confidence track
    for t in range(5):
        dets = [{'bbox': [100.0 + t*5, 200.0, 80.0, 30.0], 'conf': 0.95}]
        mstrack.update(dets, frame_idx=t, ts=1000 + t*1000)
    
    tracks = mstrack.get_active_tracks()
    assert len(tracks) == 1
    t_id = tracks[0].track_id
    
    # Simulate missing detection at t=5 (e.g. occlusion or rain blur)
    # Provide a complementary radar gate covering the target area
    radar_gates = {t_id: [120.0, 195.0, 90.0, 40.0]}
    recovered_tracks = mstrack.update([], frame_idx=5, ts=6000, radar_gates=radar_gates)
    
    # Track should be preserved and updated via feedback re-detection!
    assert len(recovered_tracks) == 1
    assert recovered_tracks[0].track_id == t_id
    assert recovered_tracks[0].time_since_update == 0
