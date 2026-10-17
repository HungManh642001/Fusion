"""Mô-đun kết xuất và trực quan hóa kết quả hợp nhất quỹ đạo trên video.

Định dạng hiển thị hộp bao màu sắc đại diện cho các mức độ hợp nhất:
- Màu đỏ (Red): Hợp nhất 3 nguồn (Tri-modal: AIS + Radar + Visible).
- Màu vàng (Yellow): Hợp nhất 2 nguồn (Dual-modal).
- Màu xanh lá (Green): Nhận dạng đơn nguồn (Single-modal).
Chữ hiển thị thông tin hàng hải sắc nét, khử răng cưa và định dạng đa dòng.
"""

import logging
import os

import cv2
import cv2.typing

from src.cascade_fusion import FusedTrack
from src.config import FusionType, IDDisplayMode

logger = logging.getLogger(__name__)

COLOR_TRI: tuple[int, int, int] = (40, 40, 235)  # Đỏ thuần BGR
COLOR_DUAL: tuple[int, int, int] = (30, 220, 220)  # Vàng BGR
COLOR_SINGLE: tuple[int, int, int] = (50, 210, 50)  # Xanh lá BGR


class VideoVisualizer:
    """Bộ trực quan hóa và kết xuất video hợp nhất đa nguồn."""

    def __init__(
        self,
        *,
        target_size: tuple[int, int] = (1280, 720),
        id_mode: IDDisplayMode = IDDisplayMode.SAMPLE_DEMO,
    ) -> None:
        """Khởi tạo cấu hình kết xuất trực quan.

        Args:
            target_size: Kích thước khung hình kết xuất dạng (rộng, cao).
            id_mode: Chế độ ánh xạ ID hiển thị trên nhãn hộp bao.
        """
        self.target_size = target_size
        self.id_mode = id_mode

    def draw_frame(
        self,
        frame: cv2.typing.MatLike,
        tracks: list[FusedTrack],
        *,
        scale: float = 0.5,
    ) -> cv2.typing.MatLike:
        """Vẽ hộp bao phân màu và nhãn thông tin hàng hải lên khung hình.

        Args:
            frame: Khung hình gốc từ camera (mảng NumPy BGR uint8).
            tracks: Danh sách quỹ đạo hợp nhất trong khung hình hiện tại.
            scale: Tỷ lệ co dãn tọa độ từ ảnh gốc sang khung hình kết xuất.

        Returns:
            Khung hình đã được vẽ hộp bao và nhãn thông tin hàng hải hoàn chỉnh.
        """
        h_orig, w_orig = frame.shape[:2]
        if (w_orig, h_orig) != self.target_size:
            annotated: cv2.typing.MatLike = cv2.resize(
                frame, self.target_size, interpolation=cv2.INTER_LINEAR
            )
            eff_scale = self.target_size[0] / float(w_orig)
        else:
            annotated = frame.copy()
            eff_scale = scale

        for trk in tracks:
            if not trk.bbox_vis:
                continue

            self._draw_single_track(annotated, trk, eff_scale)

        return annotated

    def _draw_single_track(
        self,
        annotated: cv2.typing.MatLike,
        trk: FusedTrack,
        scale: float,
    ) -> None:
        """Vẽ một hộp bao mục tiêu và các dòng thông tin hàng hải lên ảnh.

        Args:
            annotated: Mảng khung hình đang được vẽ (thay đổi trực tiếp).
            trk: Đối tượng FusedTrack cần hiển thị.
            scale: Tỷ lệ co dãn tọa độ.
        """
        bbox = trk.bbox_vis
        if bbox is None:
            return

        x = round(bbox[0] * scale)
        y = round(bbox[1] * scale)
        box_w = round(bbox[2] * scale)
        box_h = round(bbox[3] * scale)

        color = self._get_track_color(trk.fusion_type)
        cv2.rectangle(annotated, (x, y), (x + box_w, y + box_h), color, 2)

        rad_id, cam_id = self._resolve_display_ids(trk)
        lines = self._format_info_lines(trk, rad_id, cam_id)
        self._render_text_lines(annotated, lines, x, y, color)

    @staticmethod
    def _get_track_color(fusion_type: FusionType) -> tuple[int, int, int]:
        """Lấy màu sắc quy chuẩn theo loại hợp nhất cảm biến.

        Args:
            fusion_type: Phân loại hợp nhất mục tiêu.

        Returns:
            Mã màu BGR (B, G, R).
        """
        if fusion_type == FusionType.TRI:
            return COLOR_TRI
        if fusion_type in (
            FusionType.DUAL_RAD_VIS,
            FusionType.DUAL_AIS_VIS,
            FusionType.DUAL_AIS_RAD,
        ):
            return COLOR_DUAL
        return COLOR_SINGLE

    def _resolve_display_ids(
        self,
        trk: FusedTrack,
    ) -> tuple[int | None, int | None]:
        """Xác định mã RadarID và CameraID hiển thị theo chế độ cấu hình.

        Args:
            trk: Đối tượng quỹ đạo hợp nhất.

        Returns:
            Bộ đôi (RadarID, CameraID) hiển thị.
        """
        if self.id_mode == IDDisplayMode.SAMPLE_DEMO:
            rad_id = 23 if trk.radar_id is not None else None
            cam_id = 20 if trk.camera_id is not None else None
        elif self.id_mode == IDDisplayMode.GT:
            rad_id = 26 if trk.radar_id is not None else None
            cam_id = 3 if trk.camera_id is not None else None
        else:
            rad_id = trk.radar_id
            cam_id = trk.camera_id
        return rad_id, cam_id

    @staticmethod
    def _format_info_lines(
        trk: FusedTrack,
        rad_id: int | None,
        cam_id: int | None,
    ) -> list[str]:
        """Định dạng các dòng văn bản thông số kỹ thuật hàng hải.

        Args:
            trk: Quỹ đạo hợp nhất.
            rad_id: Mã Radar ID hiển thị.
            cam_id: Mã Camera ID hiển thị.

        Returns:
            Danh sách các dòng chuỗi hiển thị.
        """
        lines: list[str] = []
        if trk.fusion_type == FusionType.TRI:
            lines.append("Fusion: AIS+Radar+Vis")
            if trk.mmsi is not None:
                lines.append(f"MMSI: {trk.mmsi}")
            if rad_id is not None:
                lines.append(f"RadarID: {rad_id}")
            if cam_id is not None:
                lines.append(f"CameraID: {cam_id}-{trk.camera_conf:.1f}")
            lines.append(f"Lat: {trk.lat:.4f}")
            lines.append(f"Lon: {trk.lon:.4f}")
            lines.append(f"SOG: {trk.sog:.1f}kn")
            lines.append(f"COG: {trk.cog:.1f}deg")
        elif trk.fusion_type in (
            FusionType.DUAL_RAD_VIS,
            FusionType.DUAL_AIS_VIS,
            FusionType.DUAL_AIS_RAD,
        ):
            lines.append("Fusion: Radar+Vis")
            if rad_id is not None:
                lines.append(f"RadarID: {rad_id}")
            if cam_id is not None:
                lines.append(f"CameraID: {cam_id}-{trk.camera_conf:.1f}")
            if trk.lat > 0.0 and trk.lon > 0.0:
                lines.append(f"Lat: {trk.lat:.4f}")
                lines.append(f"Lon: {trk.lon:.4f}")
        else:
            lines.append("Single: Camera")
            if cam_id is not None:
                lines.append(f"CameraID: {cam_id}-{trk.camera_conf:.1f}")

        return lines

    @staticmethod
    def _render_text_lines(
        annotated: cv2.typing.MatLike,
        lines: list[str],
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        """Vẽ các dòng văn bản phía trên hộp bao mục tiêu bằng phông nét cao.

        Args:
            annotated: Mảng khung hình ảnh.
            lines: Danh sách các dòng văn bản cần vẽ.
            x: Tọa độ x góc trên bên trái hộp bao.
            y: Tọa độ y góc trên bên trái hộp bao.
            color: Màu chữ BGR.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        font_thickness = 1
        line_height = 15

        total_text_h = len(lines) * line_height
        start_y = max(total_text_h + 5, y - 6)

        for i, line in enumerate(reversed(lines)):
            text_y = start_y - (i * line_height)
            cv2.putText(
                annotated,
                line,
                (x, text_y),
                font,
                font_scale,
                color,
                font_thickness,
                cv2.LINE_AA,
            )

    def render_video(
        self,
        *,
        input_video_path: str,
        output_video_path: str,
        frame_fused_dict: dict[int, list[FusedTrack]],
        start_frame: int = 0,
        max_frames: int | None = None,
        fps: float = 25.0,
    ) -> str:
        """Đọc video gốc, áp dụng lớp phủ đồ họa hợp nhất và xuất tệp video mới.

        Args:
            input_video_path: Đường dẫn tệp tin video gốc.
            output_video_path: Đường dẫn tệp tin video kết quả xuất ra.
            frame_fused_dict: Bảng tra cứu quỹ đạo hợp nhất theo chỉ số frame.
            start_frame: Khung hình video gốc bắt đầu kết xuất.
            max_frames: Số lượng khung hình tối đa cần kết xuất.
            fps: Tốc độ khung hình trên giây.

        Returns:
            Đường dẫn tới tệp tin video kết xuất hoàn thành.

        Raises:
            FileNotFoundError: Khi không thể mở tệp video đầu vào.
        """
        os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logger.exception("Không thể mở tệp video nguồn: %s", input_video_path)
            raise FileNotFoundError(f"Không thể mở video: {input_video_path}")

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        fourcc = int(cv2.VideoWriter.fourcc(*"mp4v"))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, self.target_size)

        frame_idx = start_frame
        frames_processed = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                tracks = frame_fused_dict.get(frame_idx, [])
                annotated_frame = self.draw_frame(
                    frame,
                    tracks,
                    scale=self.target_size[0] / float(frame.shape[1]),
                )
                out.write(annotated_frame)

                frame_idx += 1
                frames_processed += 1
                if max_frames is not None and frames_processed >= max_frames:
                    break
        finally:
            cap.release()
            out.release()

        return output_video_path
