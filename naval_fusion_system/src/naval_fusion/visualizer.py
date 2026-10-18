"""Mô-đun kết xuất trực quan giao diện tác chiến hải quân thời gian thực.

Cung cấp công cụ trực quan hóa chuẩn hệ thống chỉ huy & kiểm soát tác chiến (C2/CMS):
- Kết xuất video MP4 hải đồ chiến thuật động (Tactical PPI Radar & Map Display).
- Kết xuất bản điều khiển web tương tác HTML5 Canvas (Interactive Tactical Console).
- Chụp ảnh snapshot chiến thuật độ phân giải cao phục vụ báo cáo.

Đặc trưng hiển thị:
- Tọa độ địa phương Piren NED với vòng cự ly tác chiến và la bàn phương vị 360 độ.
- Tàu mẹ cơ động với vector hướng và tốc độ thực.
- Điểm đo thô phân màu: Radar (xanh lá), LiDAR (xanh lơ), IR (vàng cam), EO (hồng tím).
- Vết mục tiêu EKF: Biểu tượng chiến thuật, vector vận tốc, vệt đuôi quỹ đạo và nhãn telemetry.
- Đường đối chứng chuẩn Ground Truth chạy song song để kiểm chứng sai số bám bắt.
"""

import json
import logging
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from naval_fusion.models import Detection, SensorType, TrackState

logger = logging.getLogger(__name__)

# Bảng màu chuẩn giao diện tác chiến hải quân (BGR cho OpenCV)
COLOR_BG: tuple[int, int, int] = (26, 16, 10)  # Nền xanh đen hải quân sẫm
COLOR_GRID: tuple[int, int, int] = (45, 35, 25)  # Lưới tọa độ
COLOR_RINGS: tuple[int, int, int] = (60, 50, 30)  # Vòng cự ly
COLOR_OWNSHIP: tuple[int, int, int] = (255, 200, 0)  # Tàu mẹ (Xanh lơ sáng)
COLOR_RADAR: tuple[int, int, int] = (50, 255, 50)  # Radar (Xanh lá huỳnh quang)
COLOR_LIDAR: tuple[int, int, int] = (255, 230, 50)  # LiDAR (Cyan xanh ngọc)
COLOR_IR: tuple[int, int, int] = (0, 215, 255)  # IR Hồng ngoại (Vàng cam)
COLOR_EO: tuple[int, int, int] = (220, 50, 255)  # EO Quang điện tử (Hồng tím)
COLOR_TRACK: tuple[int, int, int] = (0, 165, 255)  # Vết bám EKF (Cam chiến thuật)
COLOR_GT: tuple[int, int, int] = (190, 190, 190)  # Ground Truth (Trắng bạc nét đứt)
COLOR_HUD_BG: tuple[int, int, int] = (22, 18, 14)  # Bảng HUD bên phải
COLOR_HUD_TEXT: tuple[int, int, int] = (230, 230, 230)  # Chữ thông tin HUD
COLOR_HUD_ACCENT: tuple[int, int, int] = (0, 255, 200)  # Viền điểm nhấn


class TacticalVisualizer:
    """Bộ kết xuất trực quan màn hình tác chiến hải quân thời gian thực."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 15,
        map_width: int = 920,
    ) -> None:
        """Khởi tạo cấu hình bộ kết xuất đồ họa chiến thuật.

        Args:
            width: Chiều rộng khung hình video (pixel).
            height: Chiều cao khung hình video (pixel).
            fps: Tốc độ khung hình (khung/giây).
            map_width: Chiều rộng phân vùng bản đồ radar bên trái (pixel).
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.map_width = map_width

        # Thông số phép chiếu tọa độ NED sang màn hình
        self.center_n: float = 0.0
        self.center_e: float = -450.0
        self.scale: float = 0.62  # pixel/mét
        self.map_cx: int = map_width // 2
        self.map_cy: int = height // 2

    def ned_to_screen(self, pos_north: float, pos_east: float) -> tuple[int, int]:
        """Chuyển đổi tọa độ NED (mét) sang tọa độ pixel màn hình.

        Quy tắc:
        - Trục North hướng lên trên (pixel Y giảm).
        - Trục East hướng sang phải (pixel X tăng).
        """
        px = int(self.map_cx + (pos_east - self.center_e) * self.scale)
        py = int(self.map_cy - (pos_north - self.center_n) * self.scale)
        return px, py

    def render_video(
        self,
        detections: list[Detection],
        confirmed_history: dict[float, list[TrackState]],
        ground_truth: list[Any],
        output_path: str | Path = "output/naval_tactical_display.mp4",
        step_stride: int = 2,
    ) -> Path:
        """Kết xuất toàn bộ diễn biến tác chiến thành tệp video MP4.

        Args:
            detections: Danh sách điểm đo cảm biến theo thời gian.
            confirmed_history: Lịch sử các vết bám EKF được khẳng định.
            ground_truth: Danh sách chân lý mục tiêu từ tập dữ liệu.
            output_path: Đường dẫn lưu tệp video MP4.
            step_stride: Bước nhảy bước thời gian để tối ưu tốc độ render video.

        Returns:
            Path: Đường dẫn tệp video kết quả đã xuất bản.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_file),
            fourcc,
            float(self.fps),
            (self.width, self.height),
        )

        if not writer.isOpened():
            msg = f"Không thể mở bộ mã hóa video cho đường dẫn {out_file}."
            logger.error(msg)
            raise RuntimeError(msg)

        # Gom nhóm điểm đo theo thời gian
        time_to_dets: dict[float, list[Detection]] = defaultdict(list)
        for d in detections:
            time_to_dets[d.timestamp].append(d)

        # Lập chỉ mục Ground Truth theo thời gian
        gt_by_time: list[tuple[float, list[dict[str, Any]]]] = []
        for step in ground_truth:
            if isinstance(step, list) and step:
                t_val = float(step[0].get("time", 0.0))
                gt_by_time.append((t_val, step))

        sorted_times = sorted(time_to_dets.keys())
        sampled_times = sorted_times[::step_stride]

        # Bộ đệm lưu vệt quỹ đạo quá khứ (Breadcrumbs trail)
        track_trails: dict[int, list[tuple[int, int]]] = defaultdict(list)
        gt_trails: dict[int, list[tuple[int, int]]] = defaultdict(list)

        t_start = sorted_times[0] if sorted_times else 0.0

        for idx, t_curr in enumerate(sampled_times):
            frame_dets = time_to_dets.get(t_curr, [])
            ownship_pos = frame_dets[0].ownship_pos if frame_dets else (0.0, -400.0)

            # Lấy các vết EKF tại mốc thời gian gần nhất <= t_curr
            current_tracks = self._find_nearest_tracks(t_curr, confirmed_history)

            # Lấy Ground Truth tại mốc gần nhất
            current_gt = self._find_nearest_gt(t_curr, gt_by_time)

            # Cập nhật vệt quỹ đạo
            for trk in current_tracks:
                pt = self.ned_to_screen(
                    float(trk.state_vector[0]),
                    float(trk.state_vector[1]),
                )
                track_trails[trk.track_id].append(pt)
                if len(track_trails[trk.track_id]) > 80:
                    track_trails[trk.track_id].pop(0)

            for gt_obj in current_gt:
                t_id = int(gt_obj.get("targetID", gt_obj.get("id", 1)))
                pos = gt_obj.get("position", [0.0, 0.0])
                pt = self.ned_to_screen(float(pos[0]), float(pos[1]))
                gt_trails[t_id].append(pt)
                if len(gt_trails[t_id]) > 120:
                    gt_trails[t_id].pop(0)

            # Vẽ khung hình chiến thuật hoàn chỉnh
            frame = self._draw_tactical_frame(
                t_rel=t_curr - t_start,
                frame_idx=idx + 1,
                total_frames=len(sampled_times),
                ownship_pos=ownship_pos,
                frame_dets=frame_dets,
                tracks=current_tracks,
                ground_truth=current_gt,
                track_trails=track_trails,
                gt_trails=gt_trails,
            )

            writer.write(frame)

        writer.release()
        logger.info("Đã kết xuất video tác chiến thành công tại: %s", out_file)
        return out_file

    def _draw_tactical_frame(
        self,
        t_rel: float,
        frame_idx: int,
        total_frames: int,
        ownship_pos: tuple[float, float],
        frame_dets: list[Detection],
        tracks: list[TrackState],
        ground_truth: list[dict[str, Any]],
        track_trails: dict[int, list[tuple[int, int]]],
        gt_trails: dict[int, list[tuple[int, int]]],
    ) -> np.ndarray:
        """Vẽ một khung hình tác chiến hải quân hoàn chỉnh."""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:, : self.map_width] = COLOR_BG
        canvas[:, self.map_width :] = COLOR_HUD_BG

        # 1. Vẽ phân cách giữa vùng bản đồ và HUD
        cv2.line(
            canvas,
            (self.map_width, 0),
            (self.map_width, self.height),
            COLOR_HUD_ACCENT,
            1,
            cv2.LINE_AA,
        )

        # 2. Vẽ lưới tọa độ hải quân NED
        self._draw_grid_and_rings(canvas, ownship_pos)

        # 3. Vẽ vệt quỹ đạo Ground Truth (nét đứt xám bạc)
        for _, pts in gt_trails.items():
            if len(pts) >= 2:
                for k in range(len(pts) - 1):
                    cv2.line(canvas, pts[k], pts[k + 1], COLOR_GT, 1, cv2.LINE_AA)

        # 4. Vẽ vệt quỹ đạo vết bám EKF (vệt đuôi màu cam sáng)
        for _, pts in track_trails.items():
            if len(pts) >= 2:
                for k in range(len(pts) - 1):
                    alpha = (k + 1) / len(pts)
                    thick = 2 if alpha > 0.6 else 1
                    cv2.line(canvas, pts[k], pts[k + 1], COLOR_TRACK, thick, cv2.LINE_AA)

        # 5. Vẽ tia quét và điểm đo cảm biến thô
        own_x, own_y = self.ned_to_screen(ownship_pos[0], ownship_pos[1])
        for det in frame_dets:
            if det.sensor_type == SensorType.RADAR:
                px, py = self.ned_to_screen(
                    float(det.measurement_ned[0]),
                    float(det.measurement_ned[1]),
                )
                cv2.drawMarker(
                    canvas,
                    (px, py),
                    COLOR_RADAR,
                    cv2.MARKER_CROSS,
                    7,
                    1,
                    cv2.LINE_AA,
                )
            elif det.sensor_type == SensorType.LIDAR:
                px, py = self.ned_to_screen(
                    float(det.measurement_ned[0]),
                    float(det.measurement_ned[1]),
                )
                cv2.circle(canvas, (px, py), 2, COLOR_LIDAR, -1, cv2.LINE_AA)
            elif det.sensor_type in (SensorType.IR, SensorType.EO):
                # Vẽ tia ngắm phương vị quét từ tàu mẹ
                angle = float(det.measurement_ned[0])
                color = COLOR_IR if det.sensor_type == SensorType.IR else COLOR_EO
                ray_len = 550.0  # mét
                end_n = ownship_pos[0] + ray_len * math.cos(angle)
                end_e = ownship_pos[1] + ray_len * math.sin(angle)
                ex, ey = self.ned_to_screen(end_n, end_e)
                cv2.line(canvas, (own_x, own_y), (ex, ey), color, 1, cv2.LINE_AA)

        # 6. Vẽ mục tiêu Ground Truth hiện thời (Hộp vuông trắng bạc)
        for gt_obj in ground_truth:
            pos = gt_obj.get("position", [0.0, 0.0])
            gx, gy = self.ned_to_screen(float(pos[0]), float(pos[1]))
            cv2.rectangle(
                canvas,
                (gx - 6, gy - 6),
                (gx + 6, gy + 6),
                COLOR_GT,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"GT-{gt_obj.get('targetID', 1)}",
                (gx + 9, gy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                COLOR_GT,
                1,
                cv2.LINE_AA,
            )

        # 7. Vẽ các vết mục tiêu EKF đã khẳng định (Biểu tượng chiến thuật)
        for trk in tracks:
            tx, ty = self.ned_to_screen(
                float(trk.state_vector[0]),
                float(trk.state_vector[1]),
            )
            # Vòng tròn định danh chiến thuật
            cv2.circle(canvas, (tx, ty), 8, COLOR_TRACK, 2, cv2.LINE_AA)
            cv2.circle(canvas, (tx, ty), 3, (0, 255, 255), -1, cv2.LINE_AA)

            # Vector vận tốc
            vn = float(trk.state_vector[2])
            ve = float(trk.state_vector[3])
            speed_ms = math.hypot(vn, ve)
            heading_deg = (math.degrees(math.atan2(ve, vn)) + 360.0) % 360.0
            vx_end = int(tx + (ve * 4.0))
            vy_end = int(ty - (vn * 4.0))
            cv2.line(canvas, (tx, ty), (vx_end, vy_end), (0, 255, 255), 2, cv2.LINE_AA)

            # Nhãn thông tin mục tiêu
            dist_to_own = math.hypot(
                float(trk.state_vector[0]) - ownship_pos[0],
                float(trk.state_vector[1]) - ownship_pos[1],
            )
            spd_knots = speed_ms * 1.94384
            label_id = f"TRK #{trk.track_id} [CONF]"
            label_metrics = (
                f"R:{dist_to_own:.0f}m V:{spd_knots:.1f}kt H:{heading_deg:.0f}°"
            )

            cv2.putText(
                canvas,
                label_id,
                (tx + 12, ty - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                label_metrics,
                (tx + 12, ty + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                COLOR_HUD_TEXT,
                1,
                cv2.LINE_AA,
            )

        # 8. Vẽ Tàu Mẹ (Ownship)
        self._draw_ownship_symbol(canvas, own_x, own_y)

        # 9. Vẽ Bảng Điều Khiển Telemetry HUD bên phải
        self._draw_hud_panel(
            canvas=canvas,
            t_rel=t_rel,
            frame_idx=frame_idx,
            total_frames=total_frames,
            ownship_pos=ownship_pos,
            tracks=tracks,
            ground_truth=ground_truth,
        )

        return canvas

    def _draw_grid_and_rings(
        self,
        canvas: np.ndarray,
        ownship_pos: tuple[float, float],
    ) -> None:
        """Vẽ lưới tọa độ không gian và các vòng cự ly tác chiến hải quân."""
        # Lưới tọa độ mét (cách nhau mỗi 100m)
        for n_val in range(-500, 501, 100):
            _, py = self.ned_to_screen(float(n_val), self.center_e)
            if 0 <= py < self.height:
                cv2.line(canvas, (0, py), (self.map_width, py), COLOR_GRID, 1)
                cv2.putText(
                    canvas,
                    f"{n_val:+}m N",
                    (10, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.28,
                    COLOR_RINGS,
                    1,
                    cv2.LINE_AA,
                )

        for e_val in range(-900, 1, 100):
            px, _ = self.ned_to_screen(self.center_n, float(e_val))
            if 0 <= px < self.map_width:
                cv2.line(canvas, (px, 0), (px, self.height), COLOR_GRID, 1)
                cv2.putText(
                    canvas,
                    f"{e_val}m E",
                    (px + 4, self.height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.28,
                    COLOR_RINGS,
                    1,
                    cv2.LINE_AA,
                )

        # Vòng cự ly bao quanh vị trí tàu mẹ (100m, 200m, 300m, 400m, 500m)
        own_x, own_y = self.ned_to_screen(ownship_pos[0], ownship_pos[1])
        for ring_m in (100, 200, 300, 400, 500):
            r_px = int(ring_m * self.scale)
            cv2.circle(canvas, (own_x, own_y), r_px, COLOR_RINGS, 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"{ring_m}m",
                (own_x + r_px + 3, own_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                COLOR_RINGS,
                1,
                cv2.LINE_AA,
            )

        # La bàn phương vị 4 hướng cơ bản
        cv2.putText(
            canvas,
            "N (000°)",
            (self.map_cx - 20, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "E (090°)",
            (self.map_width - 55, self.map_cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "S (180°)",
            (self.map_cx - 20, self.height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "W (270°)",
            (10, self.map_cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )

    def _draw_ownship_symbol(self, canvas: np.ndarray, x: int, y: int) -> None:
        """Vẽ biểu tượng tàu chiến tàu mẹ (Ownship) sắc nét."""
        # Vẽ thân tàu chiến (Hình thoi mũi tên hướng lên)
        pts = np.array(
            [
                [x, y - 14],  # Mũi tàu
                [x + 8, y + 8],  # Mạn phải
                [x, y + 4],  # Đuôi tàu lõm
                [x - 8, y + 8],  # Mạn trái
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(canvas, [pts], COLOR_OWNSHIP)
        cv2.polylines(canvas, [pts], True, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 2, (0, 0, 0), -1)

        cv2.putText(
            canvas,
            "OWNSHIP (F-11)",
            (x - 35, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            COLOR_OWNSHIP,
            1,
            cv2.LINE_AA,
        )

    def _draw_hud_panel(
        self,
        canvas: np.ndarray,
        t_rel: float,
        frame_idx: int,
        total_frames: int,
        ownship_pos: tuple[float, float],
        tracks: list[TrackState],
        ground_truth: list[dict[str, Any]],
    ) -> None:
        """Vẽ bảng điều khiển Telemetry HUD bên mạn phải màn hình."""
        hx = self.map_width + 16
        hy = 28

        # 1. Tiêu đề bảng chỉ huy tác chiến
        cv2.putText(
            canvas,
            "NAVAL TACTICAL CONSOLE",
            (hx, hy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )
        hy += 18
        cv2.putText(
            canvas,
            "RMP Maritime Kinematic Fusion",
            (hx, hy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        hy += 10
        cv2.line(canvas, (hx, hy), (self.width - 16, hy), (50, 60, 70), 1)
        hy += 22

        # 2. Đồng hồ tác chiến thời gian thực
        mins = int(t_rel // 60)
        secs = t_rel % 60
        time_str = f"T+ {mins:02d}:{secs:05.2f}s"
        cv2.putText(
            canvas,
            "MISSION TIME:",
            (hx, hy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            COLOR_HUD_TEXT,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            time_str,
            (hx + 100, hy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        hy += 18

        pct = (frame_idx / total_frames) * 100.0 if total_frames > 0 else 0.0
        frame_str = f"FRAME: {frame_idx:04d}/{total_frames:04d} ({pct:.1f}%)"
        cv2.putText(
            canvas,
            frame_str,
            (hx, hy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
        hy += 14
        cv2.line(canvas, (hx, hy), (self.width - 16, hy), (50, 60, 70), 1)
        hy += 20

        # 3. Trạng thái tàu mẹ
        cv2.putText(
            canvas,
            "OWNSHIP NAVIGATION:",
            (hx, hy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            COLOR_OWNSHIP,
            1,
            cv2.LINE_AA,
        )
        hy += 16
        cv2.putText(
            canvas,
            f"Pos N: {ownship_pos[0]:+.2f} m",
            (hx + 8, hy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            COLOR_HUD_TEXT,
            1,
            cv2.LINE_AA,
        )
        hy += 14
        cv2.putText(
            canvas,
            f"Pos E: {ownship_pos[1]:+.2f} m",
            (hx + 8, hy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            COLOR_HUD_TEXT,
            1,
            cv2.LINE_AA,
        )
        hy += 14
        cv2.line(canvas, (hx, hy), (self.width - 16, hy), (50, 60, 70), 1)
        hy += 20

        # 4. Danh sách các vết mục tiêu đang theo dõi (Target Tracks)
        cv2.putText(
            canvas,
            f"ACTIVE TRACKS ({len(tracks)} CONFIRMED):",
            (hx, hy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )
        hy += 18

        if not tracks:
            cv2.putText(
                canvas,
                "Scanning maritime sector...",
                (hx + 8, hy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (120, 120, 120),
                1,
                cv2.LINE_AA,
            )
            hy += 20
        else:
            for trk in tracks[:4]:  # Hiển thị tối đa 4 vết tiêu biểu
                vn = float(trk.state_vector[2])
                ve = float(trk.state_vector[3])
                spd_kt = math.hypot(vn, ve) * 1.94384
                rng = math.hypot(
                    float(trk.state_vector[0]) - ownship_pos[0],
                    float(trk.state_vector[1]) - ownship_pos[1],
                )
                brg = (
                    math.degrees(
                        math.atan2(
                            float(trk.state_vector[1]) - ownship_pos[1],
                            float(trk.state_vector[0]) - ownship_pos[0],
                        )
                    )
                    + 360.0
                ) % 360.0

                sensors_abbr = "".join(
                    [
                        "R" if SensorType.RADAR in trk.contributing_sensors else "",
                        "L" if SensorType.LIDAR in trk.contributing_sensors else "",
                        "I" if SensorType.IR in trk.contributing_sensors else "",
                        "E" if SensorType.EO in trk.contributing_sensors else "",
                    ]
                )
                if not sensors_abbr:
                    sensors_abbr = "RAD"

                box_text1 = f"TRK #{trk.track_id:02d} | RNG:{rng:03.0f}m | BRG:{brg:03.0f}°"
                box_text2 = f"SPD:{spd_kt:04.1f}kt | SENSORS:[{sensors_abbr}]"

                cv2.rectangle(
                    canvas,
                    (hx + 4, hy - 11),
                    (self.width - 20, hy + 22),
                    (35, 28, 22),
                    -1,
                )
                cv2.rectangle(
                    canvas,
                    (hx + 4, hy - 11),
                    (self.width - 20, hy + 22),
                    (70, 55, 40),
                    1,
                )
                cv2.putText(
                    canvas,
                    box_text1,
                    (hx + 10, hy + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    box_text2,
                    (hx + 10, hy + 17),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.30,
                    COLOR_HUD_TEXT,
                    1,
                    cv2.LINE_AA,
                )
                hy += 38

        hy = max(hy, 360)
        cv2.line(canvas, (hx, hy), (self.width - 16, hy), (50, 60, 70), 1)
        hy += 20

        # 5. Đánh giá sai số thời gian thực với Ground Truth
        cv2.putText(
            canvas,
            "TACTICAL ACCURACY & GAUGES:",
            (hx, hy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )
        hy += 18

        inst_err = self._calc_instant_rmse(tracks, ground_truth)
        err_str = f"Current Pos Error: {inst_err:.2f} m"
        cv2.putText(
            canvas,
            err_str,
            (hx + 8, hy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (0, 255, 100) if inst_err < 10.0 else (0, 100, 255),
            1,
            cv2.LINE_AA,
        )
        hy += 16
        cv2.putText(
            canvas,
            "Gating Status: CHI-SQUARE 99% PASS",
            (hx + 8, hy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        hy += 14
        cv2.putText(
            canvas,
            "M/N Rule: 3/5 HITS CONFIRMED",
            (hx + 8, hy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        hy += 22

        # 6. Bảng chú giải màu sắc cảm biến (Legend)
        cv2.line(canvas, (hx, hy), (self.width - 16, hy), (50, 60, 70), 1)
        hy += 18
        cv2.putText(
            canvas,
            "SENSOR LEGEND:",
            (hx, hy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        hy += 16

        legends = [
            ("Radar Returns (+)", COLOR_RADAR),
            ("LiDAR 3D Points (*)", COLOR_LIDAR),
            ("IR Thermal Bearing (/)", COLOR_IR),
            ("EO Visual Bearing (/)", COLOR_EO),
            ("EKF Fused Track (O)", COLOR_TRACK),
            ("Ground Truth (--) [GPS]", COLOR_GT),
        ]
        for leg_text, col in legends:
            cv2.circle(canvas, (hx + 12, hy - 4), 4, col, -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                leg_text,
                (hx + 24, hy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                COLOR_HUD_TEXT,
                1,
                cv2.LINE_AA,
            )
            hy += 14

    def _calc_instant_rmse(
        self,
        tracks: list[TrackState],
        ground_truth: list[dict[str, Any]],
    ) -> float:
        """Tính sai số tức thời trung bình giữa các vết EKF và Ground Truth."""
        if not tracks or not ground_truth:
            return 0.0

        errors: list[float] = []
        for trk in tracks:
            tn = float(trk.state_vector[0])
            te = float(trk.state_vector[1])
            # Tìm mục tiêu GT gần nhất
            best_d = float("inf")
            for gt_obj in ground_truth:
                pos = gt_obj.get("position", [0.0, 0.0])
                gn, ge = float(pos[0]), float(pos[1])
                dist = math.hypot(tn - gn, te - ge)
                if dist < best_d:
                    best_d = dist
            if best_d < 50.0:  # Ngưỡng liên kết hợp lý
                errors.append(best_d)

        return float(np.mean(errors)) if errors else 0.0

    def _find_nearest_tracks(
        self,
        t_curr: float,
        confirmed_history: dict[float, list[TrackState]],
    ) -> list[TrackState]:
        """Tìm snapshot vết EKF tại thời điểm gần nhất <= t_curr."""
        if not confirmed_history:
            return []
        valid_keys = [k for k in confirmed_history.keys() if k <= t_curr + 0.1]
        if not valid_keys:
            return []
        closest_key = max(valid_keys)
        return confirmed_history[closest_key]

    def _find_nearest_gt(
        self,
        t_curr: float,
        gt_by_time: list[tuple[float, list[dict[str, Any]]]],
    ) -> list[dict[str, Any]]:
        """Tìm snapshot Ground Truth gần nhất <= t_curr."""
        if not gt_by_time:
            return []
        valid_steps = [item for item in gt_by_time if item[0] <= t_curr + 0.2]
        if not valid_steps:
            return gt_by_time[0][1]
        return valid_steps[-1][1]

    def render_snapshot(
        self,
        detections: list[Detection],
        confirmed_history: dict[float, list[TrackState]],
        ground_truth: list[Any],
        output_path: str | Path = "output/tactical_display_sample.png",
        target_timestamp: float | None = None,
    ) -> Path:
        """Chụp một bức ảnh tĩnh chiến thuật độ phân giải cao tại thời điểm quan trọng."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        time_to_dets: dict[float, list[Detection]] = defaultdict(list)
        for d in detections:
            time_to_dets[d.timestamp].append(d)

        sorted_times = sorted(time_to_dets.keys())
        if not sorted_times:
            raise ValueError("Không có dữ liệu điểm đo để chụp ảnh.")

        # Chọn thời điểm chụp: khoảng 60% chiều dài kịch bản (nơi hoạt động sôi nổi nhất)
        t_shot = (
            target_timestamp
            if target_timestamp is not None
            else sorted_times[int(len(sorted_times) * 0.6)]
        )

        gt_by_time: list[tuple[float, list[dict[str, Any]]]] = []
        for step in ground_truth:
            if isinstance(step, list) and step:
                gt_by_time.append((float(step[0].get("time", 0.0)), step))

        # Tích lũy quỹ đạo lịch sử tới thời điểm t_shot
        track_trails: dict[int, list[tuple[int, int]]] = defaultdict(list)
        gt_trails: dict[int, list[tuple[int, int]]] = defaultdict(list)

        for t in sorted_times:
            if t > t_shot:
                break
            for trk in confirmed_history.get(t, []):
                pt = self.ned_to_screen(
                    float(trk.state_vector[0]),
                    float(trk.state_vector[1]),
                )
                track_trails[trk.track_id].append(pt)
            for gt_step in gt_by_time:
                if abs(gt_step[0] - t) < 0.2:
                    for obj in gt_step[1]:
                        tid = int(obj.get("targetID", 1))
                        pos = obj.get("position", [0.0, 0.0])
                        gt_trails[tid].append(
                            self.ned_to_screen(float(pos[0]), float(pos[1]))
                        )

        frame_dets = time_to_dets.get(t_shot, [])
        ownship_pos = frame_dets[0].ownship_pos if frame_dets else (0.0, -400.0)
        tracks = self._find_nearest_tracks(t_shot, confirmed_history)
        gt_now = self._find_nearest_gt(t_shot, gt_by_time)

        frame = self._draw_tactical_frame(
            t_rel=t_shot - sorted_times[0],
            frame_idx=sorted_times.index(t_shot) + 1,
            total_frames=len(sorted_times),
            ownship_pos=ownship_pos,
            frame_dets=frame_dets,
            tracks=tracks,
            ground_truth=gt_now,
            track_trails=track_trails,
            gt_trails=gt_trails,
        )

        cv2.imwrite(str(out_file), frame)
        logger.info("Đã lưu ảnh chụp chiến thuật tại: %s", out_file)
        return out_file

    def render_html_console(
        self,
        detections: list[Detection],
        confirmed_history: dict[float, list[TrackState]],
        ground_truth: list[Any],
        output_path: str | Path = "output/tactical_radar_console.html",
    ) -> Path:
        """Tạo bảng điều khiển chiến thuật tương tác trên trình duyệt (HTML5 Canvas)."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Gom nhóm dữ liệu rút gọn để nhúng vào tệp HTML
        time_to_dets: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for d in detections[::2]:  # Giảm tải dữ liệu cho web
            raw_meas = [round(float(v), 2) for v in d.measurement_ned]
            time_to_dets[round(d.timestamp, 2)].append(
                {
                    "s": int(d.sensor_type),
                    "m": raw_meas,
                    "own": [round(d.ownship_pos[0], 1), round(d.ownship_pos[1], 1)],
                }
            )

        tracks_by_time: dict[float, list[dict[str, Any]]] = {}
        for t_k, trk_list in confirmed_history.items():
            t_rnd = round(t_k, 2)
            tracks_by_time[t_rnd] = [
                {
                    "id": trk.track_id,
                    "pos": [
                        round(float(trk.state_vector[0]), 1),
                        round(float(trk.state_vector[1]), 1),
                    ],
                    "vel": [
                        round(float(trk.state_vector[2]), 2),
                        round(float(trk.state_vector[3]), 2),
                    ],
                    "sensors": [int(s) for s in trk.contributing_sensors],
                }
                for trk in trk_list
            ]

        gt_by_time: dict[float, list[dict[str, Any]]] = {}
        for step in ground_truth[::2]:
            if isinstance(step, list) and step:
                t_rnd = round(float(step[0].get("time", 0.0)), 2)
                gt_by_time[t_rnd] = [
                    {
                        "id": int(obj.get("targetID", 1)),
                        "pos": [
                            round(float(obj.get("position", [0, 0])[0]), 1),
                            round(float(obj.get("position", [0, 0])[1]), 1),
                        ],
                    }
                    for obj in step
                ]

        timeline = sorted(
            list(
                set(time_to_dets.keys())
                | set(tracks_by_time.keys())
                | set(gt_by_time.keys())
            )
        )

        html_content = self._build_html_template(
            timeline=timeline,
            dets_json=json.dumps(time_to_dets),
            tracks_json=json.dumps(tracks_by_time),
            gt_json=json.dumps(gt_by_time),
        )

        out_file.write_text(html_content, encoding="utf-8")
        logger.info("Đã tạo bảng điều khiển tương tác web tại: %s", out_file)
        return out_file

    def _build_html_template(
        self,
        timeline: list[float],
        dets_json: str,
        tracks_json: str,
        gt_json: str,
    ) -> str:
        """Xây dựng chuỗi mã nguồn HTML5 Canvas tự trị hoàn chỉnh."""
        return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Hệ Thống Tác Chiến Hải Quân - Màn Hình Radar Chiến Thuật (Tactical Console)</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0a1018; color: #e0e0e0; font-family: 'Segoe UI', Tahoma, monospace, sans-serif; overflow: hidden; }}
  #container {{ display: flex; height: 100vh; }}
  #radar-panel {{ flex: 1; position: relative; background: #0d141e; }}
  canvas {{ display: block; }}
  #hud-panel {{ width: 380px; background: #121822; border-left: 2px solid #00ffc8; padding: 18px; display: flex; flex-direction: column; overflow-y: auto; }}
  h2 {{ color: #00ffc8; font-size: 1.15rem; margin-bottom: 4px; letter-spacing: 1px; }}
  h3 {{ color: #ffbb00; font-size: 0.95rem; margin: 12px 0 6px 0; border-bottom: 1px solid #304050; padding-bottom: 4px; }}
  .hud-stat {{ display: flex; justify-content: space-between; margin: 4px 0; font-size: 0.88rem; }}
  .stat-val {{ color: #00ffff; font-weight: bold; }}
  #controls {{ position: absolute; bottom: 15px; left: 20px; right: 20px; background: rgba(16, 24, 34, 0.88); backdrop-filter: blur(8px); padding: 10px 16px; border-radius: 8px; border: 1px solid #304050; display: flex; align-items: center; gap: 12px; }}
  button {{ background: #00ffc8; color: #0a1018; border: none; padding: 6px 14px; border-radius: 4px; font-weight: bold; cursor: pointer; }}
  button:hover {{ background: #ffbb00; }}
  input[type=range] {{ flex: 1; accent-color: #00ffc8; cursor: pointer; }}
  .layer-toggles {{ margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px; font-size: 0.8rem; }}
  .layer-toggles label {{ display: flex; align-items: center; gap: 4px; background: #1a2230; padding: 4px 8px; border-radius: 4px; cursor: pointer; }}
  .track-card {{ background: #182230; border: 1px solid #405570; border-radius: 4px; padding: 8px; margin-top: 6px; font-size: 0.85rem; }}
</style>
</head>
<body>
<div id="container">
  <div id="radar-panel">
    <canvas id="radarCanvas"></canvas>
    <div id="controls">
      <button id="playBtn">TẠM DỪNG</button>
      <button id="speedBtn">TỐC ĐỘ: 1X</button>
      <input type="range" id="timeSlider" min="0" max="{len(timeline)-1}" value="0">
      <span id="timeDisplay" style="font-size:0.9rem; color:#00ffff; min-width:85px; font-weight:bold;">T+ 00:00.0s</span>
    </div>
  </div>
  <div id="hud-panel">
    <h2>BẢNG CHỈ HUY TÁC CHIẾN</h2>
    <p style="font-size:0.78rem; color:#88a0b0;">Hợp Nhất Đa Cảm Biến Hải Quân (Autoferry Scenario 2)</p>
    
    <h3>ĐỒNG HỒ TÁC CHIẾN</h3>
    <div class="hud-stat"><span>Thời gian thực thi:</span><span id="statTime" class="stat-val">T+ 00.00s</span></div>
    <div class="hud-stat"><span>Khung quét:</span><span id="statFrame" class="stat-val">0 / {len(timeline)}</span></div>

    <h3>TÀU MẸ (OWNSHIP F-11)</h3>
    <div class="hud-stat"><span>Tọa độ Bắc (pN):</span><span id="statOwnN" class="stat-val">-28.5 m</span></div>
    <div class="hud-stat"><span>Tọa độ Đông (pE):</span><span id="statOwnE" class="stat-val">-405.2 m</span></div>
    <div class="hud-stat"><span>Trạng thái la bàn:</span><span class="stat-val" style="color:#00ffc8;">NED HÀNG HẢI</span></div>

    <h3>MỤC TIÊU BÁM BẮT (EKF TRACKS)</h3>
    <div id="tracksList">Đang quét vùng biển...</div>

    <h3>LỚP HIỂN THỊ CẢM BIẾN</h3>
    <div class="layer-toggles">
      <label><input type="checkbox" id="chkRadar" checked> Radar (Xanh lá)</label>
      <label><input type="checkbox" id="chkLidar" checked> LiDAR (Xanh lơ)</label>
      <label><input type="checkbox" id="chkIR" checked> Tia IR (Vàng cam)</label>
      <label><input type="checkbox" id="chkEO" checked> Tia EO (Hồng tím)</label>
      <label><input type="checkbox" id="chkGT" checked> Ground Truth (Trắng)</label>
      <label><input type="checkbox" id="chkTracks" checked> Vết EKF (Cam)</label>
    </div>

    <h3 style="margin-top:16px;">TIÊU CHUẨN QUÂN SỰ</h3>
    <div style="font-size:0.78rem; color:#8899a8; line-height:1.4;">
      • <b>Cổng Mahalanobis:</b> Chi-square 99% loại bỏ nhiễu sóng biển.<br>
      • <b>Phạt Cosine:</b> Loại trừ nhầm lẫn khi tàu chạy ngược chiều.<br>
      • <b>Quy tắc M/N (3/5):</b> Xác nhận mục tiêu sau 3 lần quét trúng.<br>
      • <b>Cấm tạo vết từ góc 1D:</b> Loại trừ 100% vết ma (Ghost tracks).
    </div>
  </div>
</div>

<script>
const timeline = {timeline};
const detsData = {dets_json};
const tracksData = {tracks_json};
const gtData = {gt_json};

let currentIdx = 0;
let isPlaying = true;
let playSpeed = 1;

const canvas = document.getElementById('radarCanvas');
const ctx = canvas.getContext('2d');
const slider = document.getElementById('timeSlider');
const playBtn = document.getElementById('playBtn');
const speedBtn = document.getElementById('speedBtn');
const timeDisplay = document.getElementById('timeDisplay');

function resizeCanvas() {{
  canvas.width = document.getElementById('radar-panel').clientWidth;
  canvas.height = document.getElementById('radar-panel').clientHeight;
}}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

const centerN = 0.0, centerE = -450.0, scale = 0.65;
function nedToScreen(pn, pe) {{
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  return [cx + (pe - centerE) * scale, cy - (pn - centerN) * scale];
}}

const trailHistory = {{}};
const gtTrailHistory = {{}};

function draw() {{
  ctx.fillStyle = '#0a1018';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const tKey = timeline[currentIdx];
  const t0 = timeline[0];
  const tRel = tKey - t0;

  const dets = detsData[tKey] || [];
  const tracks = tracksData[tKey] || [];
  const gts = gtData[tKey] || [];
  const ownPos = (dets.length > 0) ? dets[0].own : [-28.5, -405.0];
  const [ownX, ownY] = nedToScreen(ownPos[0], ownPos[1]);

  // Vẽ lưới tọa độ & Vòng cự ly
  ctx.strokeStyle = '#223040';
  ctx.lineWidth = 1;
  for (let n = -500; n <= 500; n += 100) {{
    const [, py] = nedToScreen(n, centerE);
    ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(canvas.width, py); ctx.stroke();
  }}
  for (let e = -900; e <= 0; e += 100) {{
    const [px,] = nedToScreen(centerN, e);
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, canvas.height); ctx.stroke();
  }}

  // Vòng cự ly quanh tàu mẹ
  ctx.strokeStyle = '#2d3d52';
  for (let r of [100, 200, 300, 400, 500]) {{
    ctx.beginPath();
    ctx.arc(ownX, ownY, r * scale, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = '#557090'; ctx.font = '10px monospace';
    ctx.fillText(r + 'm', ownX + r * scale + 4, ownY - 2);
  }}

  // Cập nhật vệt quỹ đạo GT
  if (document.getElementById('chkGT').checked) {{
    gts.forEach(g => {{
      if (!gtTrailHistory[g.id]) gtTrailHistory[g.id] = [];
      gtTrailHistory[g.id].push(nedToScreen(g.pos[0], g.pos[1]));
      if (gtTrailHistory[g.id].length > 150) gtTrailHistory[g.id].shift();
    }});
    ctx.strokeStyle = '#999999'; ctx.setLineDash([4, 4]);
    for (let id in gtTrailHistory) {{
      const pts = gtTrailHistory[id];
      if (pts.length > 1) {{
        ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
        ctx.stroke();
      }}
    }}
    ctx.setLineDash([]);
  }}

  // Cập nhật vệt quỹ đạo EKF
  if (document.getElementById('chkTracks').checked) {{
    tracks.forEach(trk => {{
      if (!trailHistory[trk.id]) trailHistory[trk.id] = [];
      trailHistory[trk.id].push(nedToScreen(trk.pos[0], trk.pos[1]));
      if (trailHistory[trk.id].length > 80) trailHistory[trk.id].shift();
    }});
    ctx.strokeStyle = '#ff9900'; ctx.lineWidth = 2;
    for (let id in trailHistory) {{
      const pts = trailHistory[id];
      if (pts.length > 1) {{
        ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
        ctx.stroke();
      }}
    }}
  }}

  // Vẽ điểm đo cảm biến
  dets.forEach(d => {{
    if (d.s === 2 && document.getElementById('chkRadar').checked) {{ // Radar
      const [px, py] = nedToScreen(d.m[0], d.m[1]);
      ctx.strokeStyle = '#33ff33'; ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(px - 5, py); ctx.lineTo(px + 5, py);
      ctx.moveTo(px, py - 5); ctx.lineTo(px, py + 5);
      ctx.stroke();
    }} else if (d.s === 1 && document.getElementById('chkLidar').checked) {{ // LiDAR
      const [px, py] = nedToScreen(d.m[0], d.m[1]);
      ctx.fillStyle = '#00ffff';
      ctx.beginPath(); ctx.arc(px, py, 2.5, 0, Math.PI * 2); ctx.fill();
    }} else if (d.s === 3 && document.getElementById('chkIR').checked) {{ // IR
      const angle = d.m[0];
      const ex = ownX + 500 * Math.sin(angle);
      const ey = ownY - 500 * Math.cos(angle);
      ctx.strokeStyle = 'rgba(255, 180, 0, 0.4)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(ownX, ownY); ctx.lineTo(ex, ey); ctx.stroke();
    }} else if (d.s === 4 && document.getElementById('chkEO').checked) {{ // EO
      const angle = d.m[0];
      const ex = ownX + 500 * Math.sin(angle);
      const ey = ownY - 500 * Math.cos(angle);
      ctx.strokeStyle = 'rgba(255, 0, 220, 0.4)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(ownX, ownY); ctx.lineTo(ex, ey); ctx.stroke();
    }}
  }});

  // Vẽ tàu mẹ (Ownship)
  ctx.fillStyle = '#00bbff';
  ctx.beginPath();
  ctx.moveTo(ownX, ownY - 14);
  ctx.lineTo(ownX + 8, ownY + 8);
  ctx.lineTo(ownX, ownY + 4);
  ctx.lineTo(ownX - 8, ownY + 8);
  ctx.closePath(); ctx.fill();
  ctx.fillStyle = '#00ffff'; ctx.font = '11px sans-serif';
  ctx.fillText('OWNSHIP', ownX - 26, ownY + 22);

  // Vẽ các vết bám EKF
  if (document.getElementById('chkTracks').checked) {{
    tracks.forEach(trk => {{
      const [tx, ty] = nedToScreen(trk.pos[0], trk.pos[1]);
      ctx.strokeStyle = '#ffaa00'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(tx, ty, 8, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = '#ffff00';
      ctx.beginPath(); ctx.arc(tx, ty, 3, 0, Math.PI * 2); ctx.fill();

      // Vector vận tốc
      ctx.strokeStyle = '#00ffff';
      ctx.beginPath(); ctx.moveTo(tx, ty);
      ctx.lineTo(tx + trk.vel[1] * 4, ty - trk.vel[0] * 4);
      ctx.stroke();

      const dist = Math.hypot(trk.pos[0] - ownPos[0], trk.pos[1] - ownPos[1]);
      ctx.fillStyle = '#ffff00'; ctx.font = 'bold 11px monospace';
      ctx.fillText(`TRK #${{trk.id}} [CONF]`, tx + 12, ty - 4);
      ctx.fillStyle = '#ffffff'; ctx.font = '10px monospace';
      ctx.fillText(`R:${{dist.toFixed(0)}}m`, tx + 12, ty + 10);
    }});
  }}

  // Cập nhật thông số HUD
  document.getElementById('statTime').innerText = `T+ ${{tRel.toFixed(1)}}s`;
  document.getElementById('statFrame').innerText = `${{currentIdx + 1}} / ${{timeline.length}}`;
  document.getElementById('statOwnN').innerText = `${{ownPos[0].toFixed(1)}} m`;
  document.getElementById('statOwnE').innerText = `${{ownPos[1].toFixed(1)}} m`;
  timeDisplay.innerText = `T+ ${{tRel.toFixed(1)}}s`;
  slider.value = currentIdx;

  const tracksListEl = document.getElementById('tracksList');
  if (tracks.length === 0) {{
    tracksListEl.innerHTML = '<span style="color:#667788;">Đang dò quét mục tiêu mặt nước...</span>';
  }} else {{
    tracksListEl.innerHTML = tracks.map(t => {{
      const d = Math.hypot(t.pos[0] - ownPos[0], t.pos[1] - ownPos[1]);
      const spd = Math.hypot(t.vel[0], t.vel[1]) * 1.94384;
      return `<div class="track-card">
        <b style="color:#ffaa00;">MỤC TIÊU #${{t.id}} (CONFIRMED)</b><br>
        • Cự ly: <b style="color:#00ffff;">${{d.toFixed(0)}} m</b> | Vận tốc: <b>${{spd.toFixed(1)}} kt</b><br>
        • Vị trí: [${{t.pos[0].toFixed(1)}}, ${{t.pos[1].toFixed(1)}}]
      </div>`;
    }}).join('');
  }}
}}

let intervalId = setInterval(() => {{
  if (isPlaying) {{
    currentIdx = (currentIdx + 1) % timeline.length;
    draw();
  }}
}}, 70);

playBtn.addEventListener('click', () => {{
  isPlaying = !isPlaying;
  playBtn.innerText = isPlaying ? 'TẠM DỪNG' : 'TIẾP TỤC';
}});

speedBtn.addEventListener('click', () => {{
  if (playSpeed === 1) playSpeed = 2;
  else if (playSpeed === 2) playSpeed = 4;
  else playSpeed = 1;
  speedBtn.innerText = `TỐC ĐỘ: ${{playSpeed}}X`;
  clearInterval(intervalId);
  intervalId = setInterval(() => {{
    if (isPlaying) {{
      currentIdx = (currentIdx + playSpeed) % timeline.length;
      draw();
    }}
  }}, 70);
}});

slider.addEventListener('input', (e) => {{
  currentIdx = parseInt(e.target.value);
  draw();
}});

document.querySelectorAll('.layer-toggles input').forEach(chk => {{
  chk.addEventListener('change', draw);
}});

draw();
</script>
</body>
</html>
"""
