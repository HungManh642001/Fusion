"""Bộ điều phối luồng tích hợp hợp nhất động học hải quân (NavalFusionPipeline).

Module này kết nối và điều phối toàn bộ các thành phần trong phân hệ hợp nhất:
- Nạp và tiền xử lý dữ liệu cảm biến đa phương thức Autoferry (`AutoferryLoader`).
- Bù chuyển động tàu mẹ trong hệ tọa độ Piren NED (`compensate_ownship_2d`).
- Bộ lọc Kalman mở rộng dị thể (`HeterogeneousEKF`) với mô hình động học CV.
- Ghép nối dữ liệu tối ưu Mahalanobis Gating và phạt Cosine (`DataAssociator`).
- Quản lý vòng đời vết mục tiêu theo luật M/N quân sự (`TrackManager`).
- Đánh giá định lượng RMSE và kiểm chứng đối chứng (`NavalEvaluator`).
"""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.naval_fusion.association import DataAssociator
from src.naval_fusion.ekf_tracker import HeterogeneousEKF
from src.naval_fusion.evaluator import NavalEvaluator
from src.naval_fusion.loader import AutoferryLoader
from src.naval_fusion.models import Detection, EvaluationMetrics, SensorType, TrackState
from src.naval_fusion.track_manager import TrackManager

logger = logging.getLogger(__name__)


class NavalFusionPipeline:
    """Bộ điều phối luồng xử lý hợp nhất động học đa cảm biến tàu chiến.

    Attributes:
        loader: Bộ nạp và chuẩn hóa dữ liệu kịch bản Autoferry.
        ekf: Bộ lọc Kalman mở rộng dị thể xử lý cập nhật 2D Descartes và 1D góc.
        associator: Bộ ghép nối dữ liệu tối ưu với gating và phạt Cosine.
        track_manager: Bộ quản lý máy trạng thái vòng đời vết chuẩn quân sự M/N.
        evaluator: Bộ đánh giá định lượng chất lượng bám vết RMSE/MOTA.
    """

    def __init__(
        self,
        loader: AutoferryLoader | None = None,
        ekf: HeterogeneousEKF | None = None,
        associator: DataAssociator | None = None,
        track_manager: TrackManager | None = None,
        evaluator: NavalEvaluator | None = None,
        q_noise: float = 0.5,
    ) -> None:
        """Khởi tạo pipeline hợp nhất động học hải quân.

        Args:
            loader: Thể hiện AutoferryLoader tùy chỉnh hoặc None để tạo mặc định.
            ekf: Thể hiện HeterogeneousEKF tùy chỉnh hoặc None để tạo mặc định.
            associator: Thể hiện DataAssociator tùy chỉnh hoặc None để tạo mặc định.
            track_manager: Thể hiện TrackManager tùy chỉnh hoặc None để tạo mặc định.
            evaluator: Thể hiện NavalEvaluator tùy chỉnh hoặc None để tạo mặc định.
            q_noise: Cường độ nhiễu quá trình CWNA khi khởi tạo EKF mặc định.
        """
        self.loader = loader or AutoferryLoader()
        self.ekf = ekf or HeterogeneousEKF(q_noise=q_noise)
        self.associator = associator or DataAssociator()
        self.track_manager = track_manager or TrackManager()
        self.evaluator = evaluator or NavalEvaluator()

    def run(
        self,
        scenario_dir: str | Path,
        enabled_sensors: set[SensorType] | None = None,
    ) -> tuple[dict[float, list[TrackState]], EvaluationMetrics]:
        """Thực thi toàn bộ luồng tích hợp trên một kịch bản cảm biến Autoferry.

        Quy trình xử lý tuần tự:
        1. Nạp và phân tích dữ liệu kịch bản (detections và ground truth).
        2. Lọc danh sách điểm đo theo tập hợp `enabled_sensors` (nếu có).
        3. Gom nhóm điểm đo theo từng mốc thời gian và sắp xếp tăng dần.
        4. Duyệt qua từng khung thời gian:
           a. Dự báo (Predict) trạng thái toàn bộ vết hiện có tới thời điểm hiện tại.
           b. Ghép nối (Associate) giữa danh sách vết và điểm đo qua Mahalanobis.
           c. Cập nhật (Update) vết với điểm đo 2D (Radar/LiDAR) hoặc 1D góc (IR/EO).
           d. Quản lý vết (TrackManager.step): cập nhật trúng/trượt, xóa vết chết và
              khởi tạo vết mới (chỉ từ cảm biến 2D).
           e. Lưu trữ các vết đã xác nhận (is_confirmed=True) vào lịch sử bám vết.
        5. Đánh giá chất lượng bám vết định lượng qua `NavalEvaluator`.

        Args:
            scenario_dir: Đường dẫn tới thư mục kịch bản (ví dụ: `scenario2`).
            enabled_sensors: Tập hợp các cảm biến được kích hoạt; None nếu dùng tất cả.

        Returns:
            tuple[dict[float, list[TrackState]], EvaluationMetrics]:
                - confirmed_history: Từ điển ánh xạ thời gian (float) -> danh sách
                  các vết mục tiêu đã xác nhận tại mốc đó.
                - metrics: Cấu trúc số liệu định lượng (RMSE pos, vel, IDSw, MT ratio).

        Raises:
            FileNotFoundError: Nếu thư mục kịch bản không tồn tại hoặc thiếu tệp JSON.
        """
        detections, ground_truth = self.loader.load_scenario(scenario_dir)

        if enabled_sensors is not None:
            detections = [d for d in detections if d.sensor_type in enabled_sensors]

        # Đặt lại bộ quản lý vết cho kịch bản mới
        self.track_manager.reset()

        if not detections:
            logger.warning(
                "Không có điểm đo nào trong kịch bản %s sau khi lọc.",
                scenario_dir,
            )
            empty_history: dict[float, list[TrackState]] = {}
            metrics = self.evaluator.evaluate_tracks(empty_history, ground_truth)
            return empty_history, metrics

        # Gom nhóm điểm đo theo mốc thời gian
        time_groups: dict[float, list[Detection]] = defaultdict(list)
        for det in detections:
            time_groups[det.timestamp].append(det)

        sorted_times = sorted(time_groups.keys())
        active_tracks: list[TrackState] = []
        confirmed_history: dict[float, list[TrackState]] = {}

        for t_curr in sorted_times:
            frame_dets = time_groups[t_curr]
            ownship_pos = frame_dets[0].ownship_pos
            primary_sensor = frame_dets[0].sensor_type

            # Bước 1: Ngoại suy EKF cho tất cả vết đang hoạt động tới mốc t_curr
            predicted_tracks: list[TrackState] = []
            for trk in active_tracks:
                dt = t_curr - trk.last_updated
                pred_trk = self.ekf.predict(trk, dt) if dt > 0.0 else trk
                predicted_tracks.append(pred_trk)
            active_tracks = predicted_tracks

            # Bước 2: Ghép nối dữ liệu tối ưu qua Mahalanobis gating và phạt Cosine
            matches, unmatched_trks, unmatched_dets = self.associator.associate(
                tracks=active_tracks,
                detections=frame_dets,
                ownship_pos=ownship_pos,
            )

            # Bước 3: Cập nhật điểm đo cho các vết được ghép nối thành công
            for trk_idx, det_idx in matches:
                trk = active_tracks[trk_idx]
                det = frame_dets[det_idx]

                if det.sensor_type in (SensorType.RADAR, SensorType.LIDAR):
                    r_cov = (
                        self.associator.r_radar
                        if det.sensor_type == SensorType.RADAR
                        else self.associator.r_lidar
                    )
                    updated_trk = self.ekf.update_cartesian_2d(
                        track=trk,
                        z_2d=det.measurement_ned[:2],
                        R_cov=r_cov,
                        sensor_type=det.sensor_type,
                        timestamp=t_curr,
                    )
                else:
                    bearing_rad = float(det.measurement_ned[0])
                    updated_trk = self.ekf.update_bearing_1d(
                        track=trk,
                        z_bearing=bearing_rad,
                        ownship_pos=det.ownship_pos,
                        R_angle=self.associator.r_angle,
                        sensor_type=det.sensor_type,
                        timestamp=t_curr,
                    )

                active_tracks[trk_idx] = updated_trk

            # Bước 4: Quản lý vòng đời vết chuẩn quân sự
            # Cảm biến góc 1D (IR/EO) trường nhìn hẹp không làm tăng số lần trượt (miss)
            # của các vết nằm ngoài góc quét; chỉ cảm biến quét 2D (Radar/LiDAR) mới
            # xác định vết trượt thực sự.
            effective_unmatched_trks = (
                unmatched_trks
                if primary_sensor in (SensorType.RADAR, SensorType.LIDAR)
                else []
            )

            active_tracks = self.track_manager.step(
                tracks=active_tracks,
                matches=matches,
                unmatched_tracks=effective_unmatched_trks,
                unmatched_detections=unmatched_dets,
                detections=frame_dets,
                current_time=t_curr,
            )

            # Bước 5: Sao lưu trạng thái các vết đã khẳng định (CONFIRMED)
            confirmed_snapshot = [
                TrackState(
                    track_id=trk.track_id,
                    state_vector=np.copy(trk.state_vector),
                    covariance=np.copy(trk.covariance),
                    last_updated=trk.last_updated,
                    hits=trk.hits,
                    misses=trk.misses,
                    is_confirmed=trk.is_confirmed,
                    contributing_sensors=set(trk.contributing_sensors),
                )
                for trk in active_tracks
                if trk.is_confirmed
            ]

            if confirmed_snapshot:
                confirmed_history[t_curr] = confirmed_snapshot

        # Đánh giá định lượng kết quả với Ground Truth
        metrics = self.evaluator.evaluate_tracks(confirmed_history, ground_truth)
        logger.info(
            "Hoàn tất kịch bản %s: RMSE pos=%.2fm, vel=%.2fm/s, "
            "%d vết xác nhận, %d ID switches.",
            scenario_dir,
            metrics.rmse_position,
            metrics.rmse_velocity,
            metrics.num_confirmed_tracks,
            metrics.id_switches,
        )

        return confirmed_history, metrics

    def run_ablation(
        self,
        scenario_dir: str | Path,
    ) -> dict[str, EvaluationMetrics]:
        """Thực thi thử nghiệm đối chứng phân tích giá trị cảm biến (Ablation Study).

        Chạy 3 cấu hình thử nghiệm chuẩn khoa học:
        1. Radar-Only: Chỉ sử dụng cảm biến Radar hàng hải.
        2. Radar + IR: Hợp nhất Radar và Camera nhiệt hồng ngoại góc 1D.
        3. Hợp nhất Toàn diện: Radar + IR + EO + LiDAR.

        Args:
            scenario_dir: Đường dẫn kịch bản thử nghiệm.

        Returns:
            dict[str, EvaluationMetrics]: Bảng kết quả định lượng theo tên cấu hình.
        """
        results: dict[str, EvaluationMetrics] = {}

        # 1. Chế độ Radar-Only
        logger.info("Chạy thử nghiệm đối chứng: Radar-Only")
        _, metrics_radar = self.run(scenario_dir, enabled_sensors={SensorType.RADAR})
        results["Radar-Only"] = metrics_radar

        # 2. Chế độ Radar + IR Fusion
        logger.info("Chạy thử nghiệm đối chứng: Radar + IR Fusion")
        _, metrics_radar_ir = self.run(
            scenario_dir, enabled_sensors={SensorType.RADAR, SensorType.IR}
        )
        results["Radar + IR"] = metrics_radar_ir

        # 3. Chế độ Hợp nhất Toàn diện
        logger.info(
            "Chạy thử nghiệm đối chứng: Hợp nhất Toàn diện (Radar + IR + EO + LiDAR)"
        )
        _, metrics_all = self.run(scenario_dir, enabled_sensors=None)
        results["Hợp nhất Toàn diện (Radar + IR + EO + LiDAR)"] = metrics_all

        return results

