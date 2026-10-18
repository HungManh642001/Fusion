"""Đánh giá định lượng và so sánh đối chứng hiệu năng bám vết hải quân.

Module này cung cấp các chỉ số đo lường chuẩn quân sự và khoa học bao gồm:
- Sai số toàn phương trung bình vị trí (RMSE Position).
- Sai số toàn phương trung bình vận tốc (RMSE Velocity).
- Số lần nhảy định danh vết (ID Switches - IDSw).
- Tỷ lệ bám vết liên tục (Mostly Tracked - MT ratio >= 80%).
- Xuất báo cáo so sánh đối chứng (Ablation Study) dưới định dạng bảng Markdown.
"""

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy.optimize import (
    linear_sum_assignment,  # pyright: ignore[reportUnknownVariableType]
)

from naval_fusion.models import EvaluationMetrics, TrackState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _GTTarget:
    """Đối tượng mục tiêu Ground Truth nội bộ phục vụ đánh giá."""

    target_id: int
    time: float
    position: np.ndarray
    velocity: np.ndarray | None


@dataclass(frozen=True)
class _GTFrame:
    """Khung thời gian chuẩn Ground Truth chứa danh sách các mục tiêu."""

    frame_index: int
    timestamp: float
    targets: list[_GTTarget]


class NavalEvaluator:
    """Bộ đánh giá định lượng chất lượng bám vết phân hệ hợp nhất hải quân."""

    def calc_rmse_pos(
        self,
        est_positions: np.ndarray,
        gt_positions: np.ndarray,
    ) -> float:
        """Tính căn bậc hai sai số bình phương trung bình khoảng cách vị trí (mét).

        Hỗ trợ mảng 2D kích thước (K, 2) hoặc vector 1D kích thước (2,).

        Args:
            est_positions: Tọa độ vị trí ước lượng [pN, pE].
            gt_positions: Tọa độ vị trí chuẩn Ground Truth [pN, pE].

        Returns:
            float: Sai số RMSE vị trí tính bằng mét.

        Raises:
            ValueError: Nếu số lượng điểm đo hoặc số chiều không tương thích.
        """
        est = np.asarray(est_positions, dtype=np.float64)
        gt = np.asarray(gt_positions, dtype=np.float64)

        if est.size == 0 or gt.size == 0:
            return 0.0

        if est.ndim == 1:
            est = est.reshape(1, -1)
        if gt.ndim == 1:
            gt = gt.reshape(1, -1)

        if est.shape[0] != gt.shape[0]:
            msg = (
                f"Số lượng điểm ước lượng ({est.shape[0]}) và "
                f"ground truth ({gt.shape[0]}) phải bằng nhau."
            )
            logger.error(msg)
            raise ValueError(msg)

        if est.shape[1] < 2 or gt.shape[1] < 2:
            msg = "Tọa độ vị trí phải có ít nhất 2 chiều (pN, pE)."
            logger.error(msg)
            raise ValueError(msg)

        diff_n = est[:, 0] - gt[:, 0]
        diff_e = est[:, 1] - gt[:, 1]
        dist_sq = diff_n * diff_n + diff_e * diff_e
        return float(np.sqrt(np.mean(dist_sq)))

    def calc_rmse_vel(
        self,
        est_velocities: np.ndarray,
        gt_velocities: np.ndarray,
    ) -> float:
        """Tính căn bậc hai sai số bình phương trung bình vận tốc (m/s).

        Hỗ trợ mảng 2D kích thước (K, 2) hoặc vector 1D kích thước (2,).

        Args:
            est_velocities: Vector vận tốc ước lượng [vN, vE].
            gt_velocities: Vector vận tốc chuẩn Ground Truth [vN, vE].

        Returns:
            float: Sai số RMSE vận tốc tính bằng mét/giây.

        Raises:
            ValueError: Nếu số lượng điểm đo hoặc số chiều không tương thích.
        """
        est = np.asarray(est_velocities, dtype=np.float64)
        gt = np.asarray(gt_velocities, dtype=np.float64)

        if est.size == 0 or gt.size == 0:
            return 0.0

        if est.ndim == 1:
            est = est.reshape(1, -1)
        if gt.ndim == 1:
            gt = gt.reshape(1, -1)

        if est.shape[0] != gt.shape[0]:
            msg = (
                f"Số lượng điểm ước lượng ({est.shape[0]}) và "
                f"ground truth ({gt.shape[0]}) phải bằng nhau."
            )
            logger.error(msg)
            raise ValueError(msg)

        if est.shape[1] < 2 or gt.shape[1] < 2:
            msg = "Vector vận tốc phải có ít nhất 2 chiều (vN, vE)."
            logger.error(msg)
            raise ValueError(msg)

        diff_vn = est[:, 0] - gt[:, 0]
        diff_ve = est[:, 1] - gt[:, 1]
        vel_sq = diff_vn * diff_vn + diff_ve * diff_ve
        return float(np.sqrt(np.mean(vel_sq)))

    def evaluate_tracks(
        self,
        confirmed_tracks_by_time: dict[float, list[TrackState]],
        ground_truth: Sequence[dict[str, Any] | Sequence[dict[str, Any]]],
        *,
        distance_threshold: float = 25.0,
        time_tolerance: float = 0.5,
    ) -> EvaluationMetrics:
        """So sánh các vết đã xác nhận với Ground Truth theo từng mốc thời gian.

        Tính toán đầy đủ các chỉ số RMSE vị trí, RMSE vận tốc, số vết xác nhận,
        số lần nhảy ID (IDSw), và tỷ lệ bám vết liên tục (Mostly Tracked - MT ratio).

        Args:
            confirmed_tracks_by_time: Từ điển ánh xạ thời gian (giây) sang danh sách
                các vết mục tiêu đã xác nhận (TrackState).
            ground_truth: Danh sách bản ghi chuẩn kiểm chứng Ground Truth từ Autoferry.
            distance_threshold: Ngưỡng khoảng cách tối đa (mét) để chấp nhận liên kết.
            time_tolerance: Sai số thời gian tối đa (giây) để ghép mốc đo với GT.

        Returns:
            EvaluationMetrics: Đối tượng chứa các chỉ số đo lường định lượng.
        """
        gt_frames = self._parse_ground_truth(ground_truth)
        unique_gt_ids: set[int] = {
            tgt.target_id for frame in gt_frames for tgt in frame.targets
        }

        total_gt_frames_count: dict[int, int] = {gid: 0 for gid in unique_gt_ids}
        for frame in gt_frames:
            for tgt in frame.targets:
                total_gt_frames_count[tgt.target_id] += 1

        all_confirmed_track_ids: set[int] = set()
        for trks in confirmed_tracks_by_time.values():
            for trk in trks:
                if getattr(trk, "is_confirmed", True):
                    all_confirmed_track_ids.add(trk.track_id)

        if not gt_frames or not confirmed_tracks_by_time:
            return EvaluationMetrics(
                rmse_position=0.0,
                rmse_velocity=0.0,
                num_confirmed_tracks=len(all_confirmed_track_ids),
                id_switches=0,
                mostly_tracked_ratio=0.0,
                details={
                    "total_gt_targets": len(unique_gt_ids),
                    "mostly_tracked_targets": 0,
                    "matched_points": 0,
                    "unique_track_ids": sorted(list(all_confirmed_track_ids)),
                },
            )

        track_timestamps = np.array(
            sorted(confirmed_tracks_by_time.keys()), dtype=np.float64
        )
        tracked_gt_frames_count: dict[int, int] = {gid: 0 for gid in unique_gt_ids}
        target_last_track_id: dict[int, int] = {}
        id_switches = 0
        pos_errors_sq: list[float] = []
        vel_errors_sq: list[float] = []

        # Đánh giá tuần tự theo từng khung Ground Truth
        for frame in gt_frames:
            t_gt = frame.timestamp
            targets = frame.targets
            if not targets:
                continue

            diffs = np.abs(track_timestamps - t_gt)
            closest_idx = int(np.argmin(diffs))
            if float(diffs[closest_idx]) > time_tolerance:
                continue

            closest_track_time = float(track_timestamps[closest_idx])
            trks_at_t = [
                trk
                for trk in confirmed_tracks_by_time[closest_track_time]
                if getattr(trk, "is_confirmed", True)
            ]
            if not trks_at_t:
                continue

            # Bipartite matching giữa vết ước lượng và ground truth
            cost_matrix = np.zeros((len(trks_at_t), len(targets)), dtype=np.float64)
            for i, trk in enumerate(trks_at_t):
                trk_pos = trk.state_vector[:2]
                for j, tgt in enumerate(targets):
                    cost_matrix[i, j] = float(np.linalg.norm(trk_pos - tgt.position))

            assign_result = cast(
                tuple[np.ndarray, np.ndarray],
                linear_sum_assignment(cost_matrix),
            )
            row_indices: list[int] = [int(idx) for idx in assign_result[0]]
            col_indices: list[int] = [int(idx) for idx in assign_result[1]]
            for r_i, c_j in zip(row_indices, col_indices, strict=False):
                dist = float(cost_matrix[r_i, c_j])
                if dist > distance_threshold:
                    continue

                matched_trk: TrackState = trks_at_t[r_i]
                matched_tgt: _GTTarget = targets[c_j]

                pos_errors_sq.append(float(dist * dist))

                if matched_tgt.velocity is not None:
                    trk_vel: np.ndarray = matched_trk.state_vector[2:4]
                    v_err = float(np.linalg.norm(trk_vel - matched_tgt.velocity))
                    vel_errors_sq.append(float(v_err * v_err))

                tracked_gt_frames_count[matched_tgt.target_id] += 1

                if (
                    matched_tgt.target_id in target_last_track_id
                    and target_last_track_id[matched_tgt.target_id]
                    != matched_trk.track_id
                ):
                    id_switches += 1
                target_last_track_id[matched_tgt.target_id] = matched_trk.track_id

        rmse_pos = float(np.sqrt(np.mean(pos_errors_sq))) if pos_errors_sq else 0.0
        rmse_vel = float(np.sqrt(np.mean(vel_errors_sq))) if vel_errors_sq else 0.0

        mt_targets = 0
        for gid in unique_gt_ids:
            total_f = total_gt_frames_count[gid]
            if total_f > 0:
                tracked_f = tracked_gt_frames_count[gid]
                if tracked_f / total_f >= 0.80:
                    mt_targets += 1

        mostly_tracked_ratio = (
            float(mt_targets / len(unique_gt_ids)) if unique_gt_ids else 0.0
        )

        details = {
            "total_gt_targets": len(unique_gt_ids),
            "mostly_tracked_targets": mt_targets,
            "matched_points": len(pos_errors_sq),
            "total_gt_points": sum(len(f.targets) for f in gt_frames),
            "id_switches": id_switches,
            "unique_track_ids": sorted(list(all_confirmed_track_ids)),
        }

        return EvaluationMetrics(
            rmse_position=rmse_pos,
            rmse_velocity=rmse_vel,
            num_confirmed_tracks=len(all_confirmed_track_ids),
            id_switches=id_switches,
            mostly_tracked_ratio=mostly_tracked_ratio,
            details=details,
        )

    def evaluate_scenario(
        self,
        confirmed_tracks_by_time: dict[float, list[TrackState]],
        ground_truth: Sequence[dict[str, Any] | Sequence[dict[str, Any]]],
        *,
        distance_threshold: float = 25.0,
        time_tolerance: float = 0.5,
    ) -> EvaluationMetrics:
        """Bí danh của evaluate_tracks phục vụ tương thích ngược."""
        return self.evaluate_tracks(
            confirmed_tracks_by_time=confirmed_tracks_by_time,
            ground_truth=ground_truth,
            distance_threshold=distance_threshold,
            time_tolerance=time_tolerance,
        )

    def generate_markdown_report(
        self,
        results: dict[str, EvaluationMetrics],
        scenario_name: str = "Scenario",
    ) -> str:
        """Xuất báo cáo định dạng bảng Markdown so sánh các chế độ thử nghiệm.

        Args:
            results: Từ điển chứa kết quả EvaluationMetrics theo tên cấu hình/chế độ.
            scenario_name: Tên kịch bản thử nghiệm (ví dụ: `scenario2`).

        Returns:
            str: Chuỗi văn bản bảng Markdown hoàn chỉnh.
        """
        lines: list[str] = [
            "# Báo Cáo Đánh Giá Hiệu Năng Phân Hệ Hợp Nhất Động Học Hải Quân",
            f"**Kịch bản:** `{scenario_name}`",
            "",
        ]

        if not results:
            lines.append("> *Không có dữ liệu thử nghiệm để xuất báo cáo đối chứng.*")
            return "\n".join(lines) + "\n"

        table_header = (
            "| Chế độ thử nghiệm | RMSE Vị trí (m) | RMSE Vận tốc (m/s) | "
            "Số vết xác nhận | ID Switches | Tỷ lệ bám vết (MT) |"
        )
        lines.extend(
            [
                table_header,
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
            ]
        )

        for mode_name, metrics in results.items():
            mt_str = f"{metrics.mostly_tracked_ratio * 100:.1f}%"
            row = (
                f"| {mode_name} | {metrics.rmse_position:.2f} | "
                f"{metrics.rmse_velocity:.2f} | {metrics.num_confirmed_tracks} | "
                f"{metrics.id_switches} | {mt_str} |"
            )
            lines.append(row)

        lines.extend(
            [
                "",
                "### Tóm tắt & Phân tích đối chứng:",
            ]
        )

        # Tìm cấu hình có RMSE vị trí tốt nhất
        best_mode = min(results.items(), key=lambda item: item[1].rmse_position)
        lines.append(
            f"- **Độ chính xác vị trí cao nhất:** Cấu hình `{best_mode[0]}` "
            f"đạt RMSE vị trí `{best_mode[1].rmse_position:.2f}m`."
        )

        # Đánh giá tính ổn định định danh vết
        total_switches = sum(m.id_switches for m in results.values())
        if total_switches == 0:
            lines.append(
                "- **Tính toàn vẹn quỹ đạo:** Không phát sinh hiện tượng nhảy "
                "định danh vết (0 ID Switches) trên mọi cấu hình."
            )
        else:
            lines.append(
                f"- **Nhảy định danh vết:** Tổng cộng {total_switches} "
                "lần nhảy ID phát hiện giữa các chế độ thử nghiệm."
            )

        mt_ratios = [m.mostly_tracked_ratio * 100 for m in results.values()]
        lines.append(
            f"- **Tỷ lệ duy trì bám vết (MT):** Dao động từ "
            f"`{min(mt_ratios):.1f}%` đến `{max(mt_ratios):.1f}%`."
        )

        return "\n".join(lines) + "\n"

    def _parse_ground_truth(
        self,
        ground_truth: Sequence[dict[str, Any] | Sequence[dict[str, Any]]],
    ) -> list[_GTFrame]:
        """Phân tích dữ liệu Ground Truth thô thành danh sách các khung thời gian chuẩn.

        Tự động trích xuất hoặc ước lượng vector vận tốc cho các mục tiêu
        nếu không có sẵn trong tệp dữ liệu thô.

        Args:
            ground_truth: Danh sách bản ghi từ điển hoặc danh sách con từ JSON.

        Returns:
            list[_GTFrame]: Danh sách khung thời gian chuẩn hóa sắp xếp theo thời gian.
        """
        if not ground_truth:
            return []

        # Tách dữ liệu thành từng khung thời gian
        raw_frames: list[list[dict[str, Any]]] = []
        first_elem = ground_truth[0]
        if isinstance(first_elem, (list, tuple)):
            for step in ground_truth:
                if isinstance(step, (list, tuple)):
                    raw_frames.append(list(step))
        else:
            # Dạng danh sách phẳng các từ điển: gom nhóm theo mốc thời gian
            time_groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
            for item in ground_truth:
                if isinstance(item, dict):
                    t_val = item.get("time")
                    if t_val is None:
                        t_val = item.get("timestamp", 0.0)
                    time_groups[float(t_val)].append(item)
            for t_key in sorted(time_groups.keys()):
                raw_frames.append(time_groups[t_key])

        # Tập hợp quỹ đạo của từng mục tiêu để tính vận tốc nếu cần
        target_trajectories: dict[int, list[tuple[float, np.ndarray]]] = defaultdict(
            list
        )
        for frame_items in raw_frames:
            for it in frame_items:
                target_id = int(it.get("targetID", it.get("target_id", 0)))
                t_val = float(it.get("time", it.get("timestamp", 0.0)))
                pos_raw = it.get("position", [0.0, 0.0])
                pos_arr = np.array(
                    [float(pos_raw[0]), float(pos_raw[1])], dtype=np.float64
                )
                target_trajectories[target_id].append((t_val, pos_arr))

        # Ước lượng vận tốc theo chuỗi thời gian nếu không có sẵn trường velocity
        target_velocities: dict[int, dict[float, np.ndarray]] = defaultdict(dict)
        for tid, traj in target_trajectories.items():
            traj.sort(key=lambda x: x[0])
            num_pts = len(traj)
            for k in range(num_pts):
                t_k, p_k = traj[k]
                if k < num_pts - 1:
                    t_next, p_next = traj[k + 1]
                    dt = t_next - t_k
                    if dt > 1e-5:
                        v_k = (p_next - p_k) / dt
                    else:
                        v_k = np.zeros(2, dtype=np.float64)
                elif num_pts >= 2:
                    t_prev, p_prev = traj[k - 1]
                    dt = t_k - t_prev
                    v_k = (
                        (p_k - p_prev) / dt
                        if dt > 1e-5
                        else np.zeros(2, dtype=np.float64)
                    )
                else:
                    v_k = np.zeros(2, dtype=np.float64)
                target_velocities[tid][t_k] = v_k

        # Xây dựng danh sách _GTFrame
        parsed_frames: list[_GTFrame] = []
        for f_idx, frame_items in enumerate(raw_frames):
            if not frame_items:
                continue

            frame_targets: list[_GTTarget] = []
            frame_times: list[float] = []

            for it in frame_items:
                target_id = int(it.get("targetID", it.get("target_id", 0)))
                t_val = float(it.get("time", it.get("timestamp", 0.0)))
                frame_times.append(t_val)

                pos_raw = it.get("position", [0.0, 0.0])
                pos_arr = np.array(
                    [float(pos_raw[0]), float(pos_raw[1])], dtype=np.float64
                )

                if "velocity" in it and it["velocity"] is not None:
                    vel_raw = it["velocity"]
                    vel_arr: np.ndarray | None = np.array(
                        [float(vel_raw[0]), float(vel_raw[1])], dtype=np.float64
                    )
                else:
                    vel_arr = target_velocities[target_id].get(t_val)

                frame_targets.append(
                    _GTTarget(
                        target_id=target_id,
                        time=t_val,
                        position=pos_arr,
                        velocity=vel_arr,
                    )
                )

            mean_time = float(np.mean(frame_times))
            parsed_frames.append(
                _GTFrame(
                    frame_index=f_idx,
                    timestamp=mean_time,
                    targets=frame_targets,
                )
            )

        parsed_frames.sort(key=lambda f: f.timestamp)
        return parsed_frames
