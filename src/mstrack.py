import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from src.kalman_tracker import KalmanBoxTracker

@dataclass
class Track:
    track_id: int
    bbox: List[float]  # [xmin, ymin, w, h]
    conf: float
    hits: int
    age: int
    time_since_update: int
    trajectory: List[List[float]] = field(default_factory=list)
    timestamps: List[int] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)

def compute_iou(bb_test: List[float], bb_gt: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [xmin, ymin, w, h]."""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[0] + bb_test[2], bb_gt[0] + bb_gt[2])
    yy2 = min(bb_test[1] + bb_test[3], bb_gt[1] + bb_gt[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_test = max(bb_test[2] * bb_test[3], 1e-6)
    area_gt = max(bb_gt[2] * bb_gt[3], 1e-6)
    union = area_test + area_gt - inter
    return inter / union

class MSTrack:
    """
    Multi-Stage Detection and Tracking Method (MSTrack) with multi-source feedback re-detection
    (Sensors 2025, 25, 2179, Section 3.1).
    """
    def __init__(self, modality: str = 'visible', max_age: int = 300, min_hits: int = 1, iou_threshold: float = 0.2):
        self.modality = modality
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        self.tracks_metadata: Dict[int, Dict] = {}

    def update(
        self,
        detections: List[Dict],
        frame_idx: int,
        ts: int,
        radar_gates: Optional[Dict[int, List[float]]] = None,
        ais_seeds: Optional[Dict[int, Tuple[float, float]]] = None,
        radar_image: Optional[np.ndarray] = None
    ) -> List[Track]:
        """
        Updates trackers with detections. If detections are missing for previously established
        tracks, performs feedback-based re-detection (radar via AIS seed or visible via radar gate).
        """
        self.frame_count += 1
        
        # 1. Predict new locations for existing trackers
        predicted_boxes = []
        to_del = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict()
            if np.any(np.isnan(pred)):
                to_del.append(i)
            else:
                predicted_boxes.append(pred)

        for i in reversed(to_del):
            self.trackers.pop(i)

        # 2. Associate detections with predicted trackers via Hungarian algorithm
        det_boxes = [d['bbox'] for d in detections]
        det_confs = [d.get('conf', 1.0) for d in detections]
        
        matched_indices = []
        unmatched_dets = list(range(len(det_boxes)))
        unmatched_trks = list(range(len(self.trackers)))

        if len(self.trackers) > 0 and len(det_boxes) > 0:
            iou_matrix = np.zeros((len(det_boxes), len(self.trackers)), dtype=np.float32)
            for d, det in enumerate(det_boxes):
                for t, trk_box in enumerate(predicted_boxes):
                    iou_matrix[d, t] = compute_iou(det, trk_box)

            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched_indices.append((r, c))
                    if r in unmatched_dets:
                        unmatched_dets.remove(r)
                    if c in unmatched_trks:
                        unmatched_trks.remove(c)

        # 3. Update matched trackers
        for d_idx, t_idx in matched_indices:
            self.trackers[t_idx].update(det_boxes[d_idx], conf=det_confs[d_idx])
            t_id = self.trackers[t_idx].track_id
            if t_id not in self.tracks_metadata:
                self.tracks_metadata[t_id] = {'timestamps': [], 'trajectory': []}
            self.tracks_metadata[t_id]['timestamps'].append(ts)
            self.tracks_metadata[t_id]['trajectory'].append(det_boxes[d_idx])

        # 4. Feedback Re-detection for unmatched trackers (Section 3.1.2 & 3.1.3)
        for t_idx in unmatched_trks:
            trk = self.trackers[t_idx]
            t_id = trk.track_id
            recovered_bbox = None

            if self.modality == 'visible' and radar_gates is not None and t_id in radar_gates:
                # Visible feedback re-detection (Section 3.1.3, Eq. 4-7)
                gate = radar_gates[t_id]  # [xgate, ygate, wgate, hgate]
                recent_confs = trk.confidence_history[-10:] if trk.confidence_history else [1.0]
                mconf = float(np.mean(recent_confs))
                
                # Kinematic extrapolation
                if mconf > 0.8 and len(trk.history) >= 2:
                    last_box = trk.history[-1]
                    prev_box = trk.history[-2]
                    vx = last_box[0] - prev_box[0]
                    vy = last_box[1] - prev_box[1]
                    vw = last_box[2] - prev_box[2]
                    vh = last_box[3] - prev_box[3]
                    pred_box = [last_box[0] + vx, last_box[1] + vy, max(last_box[2] + vw, 10.0), max(last_box[3] + vh, 10.0)]
                else:
                    pred_box = trk.get_state()

                # Verify within radar tracking gate (Eq. 7)
                xg, yg, wg, hg = gate[0], gate[1], gate[2], gate[3]
                xd, yd, wd, hd = pred_box[0], pred_box[1], pred_box[2], pred_box[3]
                
                # Check spatial overlap with gate
                iou_with_gate = compute_iou(pred_box, gate)
                center_inside = (xg <= xd + wd/2 <= xg + wg) and (yg <= yd + hd/2 <= yg + hg)
                if iou_with_gate > 0.1 or center_inside:
                    recovered_bbox = pred_box

            elif self.modality == 'radar' and ais_seeds is not None and t_id in ais_seeds:
                # Radar feedback re-detection (Section 3.1.2, Eq. 2-3)
                seed_x, seed_y = ais_seeds[t_id]
                pred_box = trk.get_state()
                # Check if seed is inside gate
                if pred_box[0] <= seed_x <= pred_box[0] + pred_box[2] and pred_box[1] <= seed_y <= pred_box[1] + pred_box[3]:
                    recovered_bbox = pred_box

            if recovered_bbox is not None:
                trk.update(recovered_bbox, conf=0.8)
                if t_id not in self.tracks_metadata:
                    self.tracks_metadata[t_id] = {'timestamps': [], 'trajectory': []}
                self.tracks_metadata[t_id]['timestamps'].append(ts)
                self.tracks_metadata[t_id]['trajectory'].append(recovered_bbox)

        # 5. Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            new_trk = KalmanBoxTracker(det_boxes[d_idx])
            self.trackers.append(new_trk)
            t_id = new_trk.track_id
            self.tracks_metadata[t_id] = {
                'timestamps': [ts],
                'trajectory': [det_boxes[d_idx]]
            }

        # 6. Extract active output tracks
        active_tracks = []
        for trk in self.trackers:
            if trk.time_since_update == 0 and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                t_id = trk.track_id
                meta = self.tracks_metadata.get(t_id, {'timestamps': [ts], 'trajectory': [trk.get_state()]})
                active_tracks.append(Track(
                    track_id=t_id,
                    bbox=trk.get_state(),
                    conf=float(trk.confidence_history[-1]) if trk.confidence_history else 1.0,
                    hits=trk.hits,
                    age=trk.age,
                    time_since_update=trk.time_since_update,
                    trajectory=meta['trajectory'],
                    timestamps=meta['timestamps'],
                    confidence_history=trk.confidence_history
                ))

        # Remove dead trackers exceeding max_age
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return active_tracks

    def get_active_tracks(self) -> List[Track]:
        return [
            Track(
                track_id=t.track_id,
                bbox=t.get_state(),
                conf=float(t.confidence_history[-1]) if t.confidence_history else 1.0,
                hits=t.hits,
                age=t.age,
                time_since_update=t.time_since_update,
                trajectory=self.tracks_metadata.get(t.track_id, {}).get('trajectory', []),
                timestamps=self.tracks_metadata.get(t.track_id, {}).get('timestamps', []),
                confidence_history=t.confidence_history
            )
            for t in self.trackers if t.time_since_update == 0
        ]
