"""Thuật toán bám bắt đa mục tiêu đa giai đoạn MSTrack kết hợp cơ chế phản hồi.

Triển khai cấu trúc bám bắt đa mục tiêu MSTrack theo bài báo Sensors 2025, 25, 2179
(Mục 3.1), hỗ trợ tái phát hiện mục tiêu bị mất dấu thông qua phản hồi tín hiệu
từ cảm biến khác (radar gate cho camera hoặc AIS seed cho radar).
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import numpy.typing as npt
import scipy.optimize

from src.config import Modality
from src.kalman_tracker import KalmanBoxTracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Track:
    """Biểu diễn quỹ đạo bám bắt của một tàu thuyền đơn lẻ.

    Attributes:
        track_id: Mã định danh quỹ đạo bám bắt duy nhất.
        bbox: Hộp bao hiện tại [xmin, ymin, w, h].
        conf: Điểm độ tin cậy của quỹ đạo.
        hits: Số lần quan sát thành công liên tiếp.
        age: Tuổi thọ của quỹ đạo tính theo số khung hình.
        time_since_update: Số khung hình trôi qua kể từ lần cập nhật gần nhất.
        trajectory: Danh sách tọa độ lịch sử các hộp bao.
        timestamps: Danh sách dấu thời gian mili-giây tương ứng.
        confidence_history: Lịch sử độ tin cậy qua từng khung hình.
    """

    track_id: int
    bbox: list[float]
    conf: float
    hits: int
    age: int
    time_since_update: int
    trajectory: list[list[float]] = field(default_factory=list[list[float]])
    timestamps: list[int] = field(default_factory=list[int])
    confidence_history: list[float] = field(default_factory=list[float])


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Tính toán chỉ số giao trên hợp (Intersection over Union - IoU) giữa hai hộp bao.

    Args:
        box_a: Hộp bao thứ nhất định dạng [xmin, ymin, w, h].
        box_b: Hộp bao thứ hai định dạng [xmin, ymin, w, h].

    Returns:
        Tỷ lệ IoU nằm trong khoảng [0.0, 1.0].
    """
    x_left = max(float(box_a[0]), float(box_b[0]))
    y_top = max(float(box_a[1]), float(box_b[1]))
    x_right = min(float(box_a[0]) + float(box_a[2]), float(box_b[0]) + float(box_b[2]))
    y_bottom = min(float(box_a[1]) + float(box_a[3]), float(box_b[1]) + float(box_b[3]))

    inter_width = max(0.0, x_right - x_left)
    inter_height = max(0.0, y_bottom - y_top)
    intersection = inter_width * inter_height

    area_a = max(float(box_a[2]) * float(box_a[3]), 1e-6)
    area_b = max(float(box_b[2]) * float(box_b[3]), 1e-6)
    union = area_a + area_b - intersection

    return intersection / union


_LSA_FUNC = cast(
    Callable[[object], tuple[Sequence[int], Sequence[int]]],
    scipy.optimize.linear_sum_assignment,
)


def _solve_hungarian(
    cost_matrix: npt.NDArray[np.float32],
) -> tuple[list[int], list[int]]:
    """Giải bài toán gán cặp tối ưu bằng thuật toán Hungarian.

    Args:
        cost_matrix: Ma trận chi phí 2 chiều.

    Returns:
        Bộ đôi danh sách chỉ số (hàng đã gán, cột đã gán).
    """
    row_ind_raw, col_ind_raw = _LSA_FUNC(cost_matrix)
    rows: list[int] = [int(i) for i in row_ind_raw]
    cols: list[int] = [int(j) for j in col_ind_raw]
    return rows, cols


class MSTrack:
    """Bộ bám bắt đa mục tiêu MSTrack hỗ trợ cơ chế phản hồi đa cảm biến."""

    def __init__(
        self,
        *,
        modality: Modality = Modality.VISIBLE,
        max_age: int = 300,
        min_hits: int = 1,
        iou_threshold: float = 0.2,
    ) -> None:
        """Khởi tạo cấu hình bộ bám bắt.

        Args:
            modality: Loại cảm biến quan sát (radar hoặc visible).
            max_age: Số khung hình tối đa cho phép mất dấu trước khi xóa quỹ đạo.
            min_hits: Số lần bám thành công tối thiểu để xuất kết quả.
            iou_threshold: Ngưỡng IoU tối thiểu để liên kết phát hiện với vết.
        """
        self.modality = modality
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count: int = 0
        self.tracks_metadata: dict[int, dict[str, list[object]]] = {}

    def update(
        self,
        detections: Sequence[Mapping[str, object]],
        *,
        frame_idx: int,
        ts: int,
        radar_gates: dict[int, list[float]] | None = None,
        ais_seeds: dict[int, tuple[float, float]] | None = None,
    ) -> list[Track]:
        """Cập nhật quỹ đạo bám bắt tại khung hình mới.

        Args:
            detections: Danh sách các phát hiện mục tiêu trong khung hình.
            frame_idx: Chỉ số khung hình hiện tại.
            ts: Dấu thời gian mili-giây.
            radar_gates: Bảng tra cứu cổng bám bắt radar theo ID (cho visible).
            ais_seeds: Bảng tra cứu điểm hạt nhân AIS theo ID (cho radar).

        Returns:
            Danh sách các quỹ đạo bám bắt đang hoạt động.
        """
        self.frame_count += 1
        predicted_boxes = self._predict_trackers()

        det_boxes: list[list[float]] = []
        det_confs: list[float] = []
        for d in detections:
            raw_box = d.get("bbox")
            if isinstance(raw_box, Sequence) and not isinstance(raw_box, (str, bytes)):
                typed_box: list[float] = [
                    float(cast(float | int | str, x))
                    for x in cast(Sequence[object], raw_box)
                ]
                if len(typed_box) == 4:
                    det_boxes.append(typed_box)
            raw_conf = d.get("conf", 1.0)
            det_confs.append(
                float(raw_conf) if isinstance(raw_conf, (int, float, str)) else 1.0
            )

        matched, unmatched_dets, unmatched_trks = self._associate(
            det_boxes, predicted_boxes
        )

        self._update_matched(matched, det_boxes, det_confs, ts)
        self._handle_feedback(unmatched_trks, ts, radar_gates, ais_seeds)
        self._spawn_new_trackers(unmatched_dets, det_boxes, ts)
        self._prune_dead_trackers()

        return self._collect_active_tracks(ts)

    def get_active_tracks(self) -> list[Track]:
        """Lấy danh sách các quỹ đạo bám bắt đang hoạt động hiện thời.

        Returns:
            Danh sách đối tượng Track đang hoạt động.
        """
        return self._collect_active_tracks(0)

    def _predict_trackers(self) -> list[list[float]]:
        """Dự báo vị trí kế tiếp cho toàn bộ các tracker hiện có.

        Returns:
            Danh sách các hộp bao dự đoán hợp lệ.
        """
        predicted_boxes: list[list[float]] = []
        valid_trackers: list[KalmanBoxTracker] = []

        for trk in self.trackers:
            pred = trk.predict()
            if not np.any(np.isnan(pred)):
                predicted_boxes.append(pred)
                valid_trackers.append(trk)

        self.trackers = valid_trackers
        return predicted_boxes

    def _associate(
        self,
        det_boxes: list[list[float]],
        predicted_boxes: list[list[float]],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Liên kết giữa hộp phát hiện và hộp dự báo bằng thuật toán Hungarian.

        Args:
            det_boxes: Danh sách hộp bao phát hiện mới.
            predicted_boxes: Danh sách hộp bao dự báo từ Kalman.

        Returns:
            Bộ 3: (cặp chỉ số ghép nối, chỉ số phát hiện thừa, chỉ số vết chưa ghép).
        """
        unmatched_dets = list(range(len(det_boxes)))
        unmatched_trks = list(range(len(self.trackers)))
        matched_indices: list[tuple[int, int]] = []

        if not self.trackers or not det_boxes:
            return matched_indices, unmatched_dets, unmatched_trks

        iou_matrix: npt.NDArray[np.float32] = np.zeros(
            (len(det_boxes), len(self.trackers)), dtype=np.float32
        )
        for d, det in enumerate(det_boxes):
            for t, trk_box in enumerate(predicted_boxes):
                iou_matrix[d, t] = compute_iou(det, trk_box)

        row_ind, col_ind = _solve_hungarian(-iou_matrix)
        for r, c in zip(row_ind, col_ind, strict=False):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched_indices.append((r, c))
                if r in unmatched_dets:
                    unmatched_dets.remove(r)
                if c in unmatched_trks:
                    unmatched_trks.remove(c)

        return matched_indices, unmatched_dets, unmatched_trks

    def _update_matched(
        self,
        matched_indices: list[tuple[int, int]],
        det_boxes: list[list[float]],
        det_confs: list[float],
        ts: int,
    ) -> None:
        """Cập nhật các tracker đã được ghép cặp thành công.

        Args:
            matched_indices: Danh sách cặp chỉ số (phát hiện, tracker).
            det_boxes: Danh sách hộp phát hiện.
            det_confs: Điểm tin cậy phát hiện tương ứng.
            ts: Dấu thời gian mili-giây.
        """
        for d_idx, t_idx in matched_indices:
            self.trackers[t_idx].update(det_boxes[d_idx], conf=det_confs[d_idx])
            t_id = self.trackers[t_idx].track_id
            if t_id not in self.tracks_metadata:
                self.tracks_metadata[t_id] = {"timestamps": [], "trajectory": []}
            self.tracks_metadata[t_id]["timestamps"].append(ts)
            self.tracks_metadata[t_id]["trajectory"].append(det_boxes[d_idx])

    def _handle_feedback(
        self,
        unmatched_trks: list[int],
        ts: int,
        radar_gates: dict[int, list[float]] | None,
        ais_seeds: dict[int, tuple[float, float]] | None,
    ) -> None:
        """Tái phát hiện các tracker chưa được ghép nối thông qua cơ chế phản hồi.

        Args:
            unmatched_trks: Danh sách chỉ số tracker chưa có quan sát.
            ts: Dấu thời gian mili-giây.
            radar_gates: Cổng bám bắt radar.
            ais_seeds: Tọa độ hạt nhân AIS.
        """
        for t_idx in unmatched_trks:
            trk = self.trackers[t_idx]
            recovered_bbox = self._try_recover_tracker(trk, radar_gates, ais_seeds)
            if recovered_bbox is not None:
                trk.update(recovered_bbox, conf=0.8)
                t_id = trk.track_id
                if t_id not in self.tracks_metadata:
                    self.tracks_metadata[t_id] = {"timestamps": [], "trajectory": []}
                self.tracks_metadata[t_id]["timestamps"].append(ts)
                self.tracks_metadata[t_id]["trajectory"].append(recovered_bbox)

    def _try_recover_tracker(
        self,
        trk: KalmanBoxTracker,
        radar_gates: dict[int, list[float]] | None,
        ais_seeds: dict[int, tuple[float, float]] | None,
    ) -> list[float] | None:
        """Thử nghiệm tái phát hiện một tracker theo từng phương thức cảm biến.

        Args:
            trk: Đối tượng KalmanBoxTracker đang xét.
            radar_gates: Cổng bám bắt radar.
            ais_seeds: Hạt nhân AIS.

        Returns:
            Hộp bao phục hồi được hoặc None nếu không thỏa mãn điều kiện.
        """
        t_id = trk.track_id
        if self.modality == Modality.VISIBLE and radar_gates and t_id in radar_gates:
            return self._recover_visible_track(trk, radar_gates[t_id])

        if self.modality == Modality.RADAR and ais_seeds and t_id in ais_seeds:
            return self._recover_radar_track(trk, ais_seeds[t_id])

        return None

    @staticmethod
    def _recover_visible_track(
        trk: KalmanBoxTracker,
        gate: list[float],
    ) -> list[float] | None:
        """Tái khôi phục mục tiêu camera dựa trên cổng bám bắt radar (Công thức 4-7).

        Args:
            trk: Tracker camera bị mất quan sát.
            gate: Hộp cổng bám bắt radar [xg, yg, wg, hg].

        Returns:
            Hộp bao phục hồi hoặc None.
        """
        recent_confs = trk.confidence_history[-10:] if trk.confidence_history else [1.0]
        mean_conf = float(np.mean(recent_confs))

        if mean_conf > 0.8 and len(trk.history) >= 2:
            last_box = trk.history[-1]
            prev_box = trk.history[-2]
            vx = last_box[0] - prev_box[0]
            vy = last_box[1] - prev_box[1]
            vw = last_box[2] - prev_box[2]
            vh = last_box[3] - prev_box[3]
            pred_box = [
                last_box[0] + vx,
                last_box[1] + vy,
                max(last_box[2] + vw, 10.0),
                max(last_box[3] + vh, 10.0),
            ]
        else:
            pred_box = trk.get_state()

        xg, yg, wg, hg = gate[0], gate[1], gate[2], gate[3]
        xd, yd, wd, hd = pred_box[0], pred_box[1], pred_box[2], pred_box[3]
        center_inside = (xg <= xd + wd / 2.0 <= xg + wg) and (
            yg <= yd + hd / 2.0 <= yg + hg
        )
        if compute_iou(pred_box, gate) > 0.1 or center_inside:
            return pred_box
        return None

    @staticmethod
    def _recover_radar_track(
        trk: KalmanBoxTracker,
        seed: tuple[float, float],
    ) -> list[float] | None:
        """Tái khôi phục mục tiêu radar dựa trên hạt nhân AIS (Công thức 2-3).

        Args:
            trk: Tracker radar bị mất quan sát.
            seed: Tọa độ hạt nhân AIS (seed_x, seed_y) trên ảnh radar.

        Returns:
            Hộp bao phục hồi hoặc None.
        """
        seed_x, seed_y = seed
        pred_box = trk.get_state()
        inside = (pred_box[0] <= seed_x <= pred_box[0] + pred_box[2]) and (
            pred_box[1] <= seed_y <= pred_box[1] + pred_box[3]
        )
        return pred_box if inside else None

    def _spawn_new_trackers(
        self,
        unmatched_dets: list[int],
        det_boxes: list[list[float]],
        ts: int,
    ) -> None:
        """Khởi tạo tracker mới cho các phát hiện chưa thuộc về vết nào.

        Args:
            unmatched_dets: Danh sách chỉ số phát hiện chưa ghép.
            det_boxes: Danh sách tọa độ hộp bao phát hiện.
            ts: Dấu thời gian mili-giây.
        """
        for d_idx in unmatched_dets:
            new_trk = KalmanBoxTracker(det_boxes[d_idx])
            self.trackers.append(new_trk)
            t_id = new_trk.track_id
            self.tracks_metadata[t_id] = {
                "timestamps": [ts],
                "trajectory": [det_boxes[d_idx]],
            }

    def _prune_dead_trackers(self) -> None:
        """Xóa bỏ các tracker đã mất quan sát vượt quá ngưỡng max_age."""
        self.trackers = [
            t for t in self.trackers if t.time_since_update <= self.max_age
        ]

    def _collect_active_tracks(self, ts: int) -> list[Track]:
        """Thu thập danh sách các quỹ đạo hợp lệ đang hoạt động để xuất đầu ra.

        Args:
            ts: Dấu thời gian hiện tại.

        Returns:
            Danh sách các đối tượng Track hợp lệ.
        """
        active_tracks: list[Track] = []
        for trk in self.trackers:
            if trk.time_since_update == 0 and (
                trk.hits >= self.min_hits or self.frame_count <= self.min_hits
            ):
                t_id = trk.track_id
                meta = self.tracks_metadata.get(
                    t_id, {"timestamps": [ts], "trajectory": [trk.get_state()]}
                )
                raw_traj = meta.get("trajectory", [])
                raw_ts = meta.get("timestamps", [])
                traj_list: list[list[float]] = []
                for pt in raw_traj:
                    if isinstance(pt, Sequence) and not isinstance(pt, (str, bytes)):
                        traj_list.append(
                            [
                                float(cast(float | int | str, coord))
                                for coord in cast(Sequence[object], pt)
                            ]
                        )

                ts_list: list[int] = [
                    int(cast(int | float | str, t))
                    for t in cast(Sequence[object], raw_ts)
                ]

                active_tracks.append(
                    Track(
                        track_id=t_id,
                        bbox=trk.get_state(),
                        conf=float(trk.confidence_history[-1])
                        if trk.confidence_history
                        else 1.0,
                        hits=trk.hits,
                        age=trk.age,
                        time_since_update=trk.time_since_update,
                        trajectory=traj_list,
                        timestamps=ts_list,
                        confidence_history=list(trk.confidence_history),
                    )
                )
        return active_tracks
