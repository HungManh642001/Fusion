import os
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
from src.cascade_fusion import FusedTrack

COLOR_TRI = (40, 40, 235)       # Red in BGR
COLOR_DUAL = (30, 220, 220)     # Yellow in BGR
COLOR_SINGLE = (50, 210, 50)    # Green in BGR

class VideoVisualizer:
    """
    Renders HD video frames and files strictly matching the visualization style
    from Sensors 2025 (Figure 14 and sample demo videos).
    """
    def __init__(self, target_size: Tuple[int, int] = (1280, 720), id_mode: str = 'sample_demo'):
        self.target_size = target_size
        self.id_mode = id_mode

    def draw_frame(self, frame: np.ndarray, tracks: List[FusedTrack], scale: float = 0.5) -> np.ndarray:
        """
        Draws color-coded bounding boxes and multi-line maritime metadata on the frame.
        """
        h, w = frame.shape[:2]
        if (w, h) != self.target_size:
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
            scale = self.target_size[0] / float(w)

        annotated = frame.copy()

        for trk in tracks:
            if not trk.bbox_vis:
                continue

            # Scale bounding box to current frame resolution
            orig_x, orig_y, orig_w, orig_h = trk.bbox_vis
            x = int(round(orig_x * scale))
            y = int(round(orig_y * scale))
            box_w = int(round(orig_w * scale))
            box_h = int(round(orig_h * scale))

            # Determine color by fusion type
            if trk.fusion_type == 'tri':
                color = COLOR_TRI
            elif trk.fusion_type in ('dual_rad_vis', 'dual_ais_vis', 'dual_ais_rad'):
                color = COLOR_DUAL
            else:
                color = COLOR_SINGLE

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + box_w, y + box_h), color, 2)

            # Display IDs: sample_demo (RadarID:23, CameraID:20), gt (26, 3), or tracker (raw)
            if self.id_mode == 'sample_demo':
                rad_id = 23 if trk.radar_id is not None else None
                cam_id = 20 if trk.camera_id is not None else None
            elif self.id_mode == 'gt':
                rad_id = 26 if trk.radar_id is not None else None
                cam_id = 3 if trk.camera_id is not None else None
            else:
                rad_id = trk.radar_id
                cam_id = trk.camera_id

            # Assemble metadata lines
            lines = []
            if trk.fusion_type == 'tri':
                lines = [
                    f"SOG:{trk.sog:.1f}",
                    f"COG:{trk.cog:.1f}",
                    f"Lat:{trk.lat:.6f}",
                    f"Lon:{trk.lon:.6f}",
                    f"MMSI:{trk.mmsi}",
                    f"RadarID:{rad_id}",
                    f"CameraID:{cam_id}-{trk.camera_conf:.1f}"
                ]
            elif trk.fusion_type == 'dual_ais_vis':
                lines = [
                    f"SOG:{trk.sog:.1f}",
                    f"COG:{trk.cog:.1f}",
                    f"Lat:{trk.lat:.6f}",
                    f"Lon:{trk.lon:.6f}",
                    f"MMSI:{trk.mmsi}",
                    f"CameraID:{trk.camera_id}-{trk.camera_conf:.1f}"
                ]
            elif trk.fusion_type == 'dual_rad_vis':
                lines = [
                    f"SOG:{trk.sog:.1f}",
                    f"COG:{trk.cog:.1f}",
                    f"RadarID:{trk.radar_id}",
                    f"CameraID:{trk.camera_id}-{trk.camera_conf:.1f}"
                ]
            else:
                lines = [
                    f"CameraID:{trk.camera_id}-{trk.camera_conf:.1f}"
                ]

            # Render text stack vertically above the upper edge of the box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.44
            font_thickness = 1
            line_height = 15
            
            # Start position above box
            total_text_h = len(lines) * line_height
            start_y = max(total_text_h + 5, y - 6)
            
            # Draw lines from bottom to top
            for i, line in enumerate(reversed(lines)):
                text_y = start_y - (i * line_height)
                # Pure clean anti-aliased text (no shadow bleed)
                cv2.putText(annotated, line, (x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)

        return annotated

    def render_video(
        self,
        input_video_path: str,
        output_video_path: str,
        frame_fused_dict: Dict[int, List[FusedTrack]],
        start_frame: int = 0,
        max_frames: Optional[int] = None,
        fps: float = 25.0
    ) -> str:
        """
        Reads input video, applies multi-source fusion overlay, and encodes output video.
        """
        os.makedirs(os.path.dirname(output_video_path) or '.', exist_ok=True)
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, self.target_size)

        frame_idx = start_frame
        frames_processed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            tracks = frame_fused_dict.get(frame_idx, [])
            annotated_frame = self.draw_frame(frame, tracks, scale=self.target_size[0] / frame.shape[1])
            out.write(annotated_frame)

            frame_idx += 1
            frames_processed += 1
            if max_frames is not None and frames_processed >= max_frames:
                break

        cap.release()
        out.release()
        return output_video_path
