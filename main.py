"""Điểm nhập thực thi luồng hợp nhất nhận thức và quỹ đạo tàu thuyền đa nguồn.

Thực thi kịch bản xử lý đồng bộ thời không giữa radar, camera và thông điệp AIS,
chạy thuật toán bám bắt MSTrack, hợp nhất tầng Cascade Association Matching,
kết xuất video trực quan hóa và báo cáo chỉ số định lượng theo Sensors 2025, 25, 2179.
"""

import argparse
import logging
import os

import numpy as np

from src.ais_decoder import AISDecoder, AISMessage, extrapolate_kinematic
from src.cascade_fusion import CascadeFusion, FusedTrack
from src.config import IDDisplayMode, Modality, PipelineSettings
from src.coord_transform import CoordinateTransformer
from src.evaluator import PipelineEvaluator
from src.mstrack import MSTrack
from src.visualizer import VideoVisualizer

logger = logging.getLogger("main")


FusionLoopResult = tuple[
    dict[int, list[FusedTrack]],
    list[dict[str, object]],
    dict[int, list[dict[str, object]]],
]

DetectionsData = tuple[
    list[int],
    dict[int, list[dict[str, object]]],
    dict[int, list[dict[str, object]]],
    int,
]


def load_ais_data(scene_dir: str, decoder: AISDecoder) -> dict[int, list[AISMessage]]:
    """Tải và nhóm các bản tin AIS theo mã định danh tàu thuyền MMSI.

    Args:
        scene_dir: Thư mục chứa dữ liệu kịch bản.
        decoder: Bộ giải mã thông điệp AIS.

    Returns:
        Bảng tra cứu danh sách bản tin AIS theo mã MMSI.
    """
    ais_path = os.path.join(scene_dir, "AIS.txt")
    logger.info("Đang giải mã thông điệp AIS từ: %s", ais_path)
    messages = decoder.decode_file(ais_path)
    logger.info("Đã giải mã thành công %d thông điệp AIS hợp lệ.", len(messages))

    ais_by_mmsi: dict[int, list[AISMessage]] = {}
    for msg in messages:
        if msg.mmsi not in ais_by_mmsi:
            ais_by_mmsi[msg.mmsi] = []
        ais_by_mmsi[msg.mmsi].append(msg)
    return ais_by_mmsi


def load_detections(scene_dir: str) -> DetectionsData:
    """Tải chuỗi thời gian radar và tập phát hiện mục tiêu cho cả hai phương thức.

    Args:
        scene_dir: Thư mục chứa dữ liệu kịch bản.

    Returns:
        Bộ 4: (thời gian radar, phát hiện radar, phát hiện cam, tổng số frame).
    """
    radar_dir = os.path.join(scene_dir, "radar_images")
    radar_files = sorted(os.listdir(radar_dir))
    radar_timestamps = [int(f.replace(".jpg", "")) for f in radar_files]

    radar_gt_path = os.path.join(scene_dir, "radar_track_gt.csv")
    radar_data = np.loadtxt(radar_gt_path, delimiter=",")

    visible_gt_path = os.path.join(scene_dir, "visible_track_gt.csv")
    vis_data = np.loadtxt(visible_gt_path, delimiter=",")
    total_vis_frames = int(np.max(vis_data[:, 0])) + 1 if len(vis_data) > 0 else 0

    radar_dets: dict[int, list[dict[str, object]]] = {}
    for i in range(len(radar_data)):
        row = radar_data[i]
        f = int(row[0])
        if f not in radar_dets:
            radar_dets[f] = []
        conf_val = float(row[6])
        radar_dets[f].append(
            {
                "bbox": [float(row[2]), float(row[3]), float(row[4]), float(row[5])],
                "conf": conf_val if conf_val > 0 else 1.0,
                "gt_id": int(row[1]),
            }
        )

    vis_dets: dict[int, list[dict[str, object]]] = {}
    for i in range(len(vis_data)):
        row = vis_data[i]
        f = int(row[0])
        if f not in vis_dets:
            vis_dets[f] = []
        conf_val = float(row[6])
        vis_dets[f].append(
            {
                "bbox": [float(row[2]), float(row[3]), float(row[4]), float(row[5])],
                "conf": conf_val if conf_val > 0 else 1.0,
                "gt_id": int(row[1]),
            }
        )

    return radar_timestamps, radar_dets, vis_dets, total_vis_frames


def run_pipeline(settings: PipelineSettings) -> None:
    """Thực thi toàn bộ luồng hợp nhất nhận thức đa nguồn từ đầu đến cuối.

    Args:
        settings: Cấu hình tham số thực thi bất biến của pipeline.
    """
    os.makedirs(settings.output_dir, exist_ok=True)
    logger.info("Khởi chạy pipeline hợp nhất đa nguồn Scene: %s", settings.scene_dir)

    decoder = AISDecoder()
    transformer = CoordinateTransformer()
    radar_tracker = MSTrack(modality=Modality.RADAR, max_age=5, min_hits=1)
    visible_tracker = MSTrack(modality=Modality.VISIBLE, max_age=300, min_hits=1)
    fusion_engine = CascadeFusion(coord_transformer=transformer)
    visualizer = VideoVisualizer(
        target_size=settings.target_resolution, id_mode=settings.id_mode
    )
    evaluator = PipelineEvaluator()

    ais_by_mmsi = load_ais_data(settings.scene_dir, decoder)
    radar_timestamps, radar_dets, vis_dets, total_vis_frames = load_detections(
        settings.scene_dir
    )

    fusion_timeline, all_fused_instances, visible_history = _execute_fusion_loop(
        settings=settings,
        radar_timestamps=radar_timestamps,
        radar_dets=radar_dets,
        vis_dets=vis_dets,
        total_vis_frames=total_vis_frames,
        ais_by_mmsi=ais_by_mmsi,
        radar_tracker=radar_tracker,
        visible_tracker=visible_tracker,
        fusion_engine=fusion_engine,
        transformer=transformer,
    )

    output_video_path = os.path.join(settings.output_dir, "scene01_fused_demo.mp4")
    render_start = settings.start_frame if not settings.render_full else 0
    render_count = settings.num_frames if not settings.render_full else total_vis_frames

    visible_video_path = os.path.join(settings.scene_dir, "visible_video.mp4")
    logger.info(
        "Đang kết xuất video tới %s (frames %d -> %d)...",
        output_video_path,
        render_start,
        render_start + render_count,
    )
    visualizer.render_video(
        input_video_path=visible_video_path,
        output_video_path=output_video_path,
        frame_fused_dict=fusion_timeline,
        start_frame=render_start,
        max_frames=render_count,
        fps=settings.fps,
    )

    _evaluate_and_report(
        output_dir=settings.output_dir,
        evaluator=evaluator,
        vis_dets=vis_dets,
        visible_history=visible_history,
        all_fused_instances=all_fused_instances,
        output_video_path=output_video_path,
    )


def _execute_fusion_loop(
    *,
    settings: PipelineSettings,
    radar_timestamps: list[int],
    radar_dets: dict[int, list[dict[str, object]]],
    vis_dets: dict[int, list[dict[str, object]]],
    total_vis_frames: int,
    ais_by_mmsi: dict[int, list[AISMessage]],
    radar_tracker: MSTrack,
    visible_tracker: MSTrack,
    fusion_engine: CascadeFusion,
    transformer: CoordinateTransformer,
) -> FusionLoopResult:
    """Thực thi vòng lặp đồng bộ thời gian và hợp nhất quỹ đạo đa cảm biến.

    Args:
        settings: Cấu hình hệ thống.
        radar_timestamps: Chuỗi mốc thời gian của từng khung radar.
        radar_dets: Phát hiện mục tiêu radar.
        vis_dets: Phát hiện mục tiêu camera.
        total_vis_frames: Tổng số khung hình camera.
        ais_by_mmsi: Bảng tin AIS theo tàu.
        radar_tracker: Bộ bám bắt radar.
        visible_tracker: Bộ bám bắt camera.
        fusion_engine: Bộ hợp nhất tầng.
        transformer: Bộ chuyển đổi hệ tọa độ.

    Returns:
        Bộ 3: (dòng thời gian, danh sách hợp nhất, lịch sử bám camera).
    """
    fusion_timeline: dict[int, list[FusedTrack]] = {}
    last_fusion: list[FusedTrack] = []
    all_fused_instances: list[dict[str, object]] = []
    visible_history: dict[int, list[dict[str, object]]] = {}

    start_ts = radar_timestamps[0]
    vis_to_radar_idx: dict[int, int] = {}
    for r_idx, r_ts in enumerate(radar_timestamps):
        v_frame = round((r_ts - start_ts) / 40.0)
        vis_to_radar_idx[v_frame] = r_idx

    end_frame = (
        total_vis_frames
        if settings.render_full
        else min(total_vis_frames, settings.start_frame + settings.num_frames)
    )
    proc_start_frame = 0 if settings.render_full else max(0, settings.start_frame - 100)

    for v_frame in range(proc_start_frame, end_frame):
        cur_ts = start_ts + v_frame * 40
        is_radar_fusion_frame = v_frame in vis_to_radar_idx

        if is_radar_fusion_frame:
            r_idx = vis_to_radar_idx[v_frame]
            r_ts = radar_timestamps[r_idx]
            last_fusion = _perform_fusion_step(
                r_idx=r_idx,
                r_ts=r_ts,
                v_frame=v_frame,
                cur_ts=cur_ts,
                ais_by_mmsi=ais_by_mmsi,
                radar_dets=radar_dets,
                vis_dets=vis_dets,
                radar_tracker=radar_tracker,
                visible_tracker=visible_tracker,
                fusion_engine=fusion_engine,
                transformer=transformer,
                visible_history=visible_history,
                all_fused_instances=all_fused_instances,
            )
            fusion_timeline[v_frame] = last_fusion
        else:
            fusion_timeline[v_frame] = _interpolate_intermediate_frame(
                v_frame=v_frame,
                cur_ts=cur_ts,
                vis_dets=vis_dets,
                visible_tracker=visible_tracker,
                visible_history=visible_history,
                last_fusion=last_fusion,
            )

    return fusion_timeline, all_fused_instances, visible_history


def _perform_fusion_step(
    *,
    r_idx: int,
    r_ts: int,
    v_frame: int,
    cur_ts: int,
    ais_by_mmsi: dict[int, list[AISMessage]],
    radar_dets: dict[int, list[dict[str, object]]],
    vis_dets: dict[int, list[dict[str, object]]],
    radar_tracker: MSTrack,
    visible_tracker: MSTrack,
    fusion_engine: CascadeFusion,
    transformer: CoordinateTransformer,
    visible_history: dict[int, list[dict[str, object]]],
    all_fused_instances: list[dict[str, object]],
) -> list[FusedTrack]:
    """Thực hiện một bước đồng bộ và hợp nhất tại mốc thời gian quét của radar."""
    extrapolated_ais: list[dict[str, object]] = []
    for mmsi, msgs in ais_by_mmsi.items():
        prev_msgs = [m for m in msgs if m.ts <= r_ts]
        if prev_msgs:
            closest_m = max(prev_msgs, key=lambda m: m.ts)
            lat_ext, lon_ext = extrapolate_kinematic(closest_m, target_ts=r_ts)
            if 30.54 < lat_ext < 30.60 and 114.28 < lon_ext < 114.33:
                extrapolated_ais.append(
                    {
                        "mmsi": mmsi,
                        "lat": lat_ext,
                        "lon": lon_ext,
                        "sog": closest_m.sog,
                        "cog": closest_m.cog,
                        "ts": r_ts,
                    }
                )

    r_detections = radar_dets.get(r_idx, [])
    active_rad = radar_tracker.update(r_detections, frame_idx=r_idx, ts=r_ts)

    radar_gates: dict[int, list[float]] = {}
    for t in active_rad:
        radar_gates[t.track_id] = transformer.radar_bbox_to_visible_gate(t.bbox)

    v_detections = vis_dets.get(v_frame, [])
    active_vis = visible_tracker.update(
        v_detections, frame_idx=v_frame, ts=cur_ts, radar_gates=radar_gates
    )
    visible_history[v_frame] = [
        {"track_id": t.track_id, "bbox": t.bbox} for t in active_vis
    ]

    rad_dicts = [
        {
            "track_id": t.track_id,
            "bbox": t.bbox,
            "conf": t.conf,
            "ts": r_ts,
            "trajectory": t.trajectory,
        }
        for t in active_rad
    ]
    vis_dicts = [
        {
            "track_id": t.track_id,
            "bbox": t.bbox,
            "conf": t.conf,
            "ts": cur_ts,
            "trajectory": t.trajectory,
        }
        for t in active_vis
    ]

    fused = fusion_engine.fuse(extrapolated_ais, rad_dicts, vis_dicts, ts=cur_ts)
    for f in fused:
        all_fused_instances.append(
            {
                "fusion_type": f.fusion_type.value,
                "mmsi": f.mmsi,
                "radar_id": f.radar_id,
                "camera_id": f.camera_id,
            }
        )
    return fused


def _interpolate_intermediate_frame(
    *,
    v_frame: int,
    cur_ts: int,
    vis_dets: dict[int, list[dict[str, object]]],
    visible_tracker: MSTrack,
    visible_history: dict[int, list[dict[str, object]]],
    last_fusion: list[FusedTrack],
) -> list[FusedTrack]:
    """Cập nhật bám bắt camera và kế thừa nhãn hợp nhất giữa hai chu kỳ radar."""
    v_detections = vis_dets.get(v_frame, [])
    active_vis = visible_tracker.update(v_detections, frame_idx=v_frame, ts=cur_ts)
    visible_history[v_frame] = [
        {"track_id": t.track_id, "bbox": t.bbox} for t in active_vis
    ]

    cam_to_fusion = {f.camera_id: f for f in last_fusion if f.camera_id is not None}
    inherited: list[FusedTrack] = []

    for vt in active_vis:
        if vt.track_id in cam_to_fusion:
            prev_f = cam_to_fusion[vt.track_id]
            inherited.append(
                FusedTrack(
                    fusion_id=prev_f.fusion_id,
                    fusion_type=prev_f.fusion_type,
                    mmsi=prev_f.mmsi,
                    radar_id=prev_f.radar_id,
                    camera_id=vt.track_id,
                    camera_conf=vt.conf,
                    bbox_vis=vt.bbox,
                    bbox_rad=prev_f.bbox_rad,
                    lat=prev_f.lat,
                    lon=prev_f.lon,
                    sog=prev_f.sog,
                    cog=prev_f.cog,
                )
            )
        else:
            inherited.append(
                FusedTrack(
                    fusion_id=vt.track_id,
                    fusion_type=IDDisplayMode.TRACKER,  # type: ignore[arg-type]
                    mmsi=None,
                    radar_id=None,
                    camera_id=vt.track_id,
                    camera_conf=vt.conf,
                    bbox_vis=vt.bbox,
                    bbox_rad=None,
                    lat=0.0,
                    lon=0.0,
                    sog=0.0,
                    cog=0.0,
                )
            )
    return inherited


def _evaluate_and_report(
    *,
    output_dir: str,
    evaluator: PipelineEvaluator,
    vis_dets: dict[int, list[dict[str, object]]],
    visible_history: dict[int, list[dict[str, object]]],
    all_fused_instances: list[dict[str, object]],
    output_video_path: str,
) -> None:
    """Tính toán chỉ số định lượng và tạo tài liệu báo cáo markdown."""
    logger.info("Đang tính toán các chỉ số định lượng MOTA, IDF1, TPF...")
    vis_gt_formatted: dict[int, list[dict[str, object]]] = {}
    for f, dets in vis_dets.items():
        if f in visible_history:
            vis_gt_formatted[f] = [
                {"track_id": d["gt_id"], "bbox": d["bbox"]} for d in dets
            ]

    vis_metrics = evaluator.evaluate_tracking(vis_gt_formatted, visible_history)
    fusion_metrics = evaluator.evaluate_fusion(all_fused_instances)

    report_md = (
        "# Báo cáo đánh giá nhận thức và hợp nhất quỹ đạo tàu thuyền đa nguồn\n\n"
        "**Tập dữ liệu**: WHUT-MSFVessel `scene01`  \n"
        "**Bài báo tham chiếu**: *Sensors 2025, 25, 2179*  \n\n"
        "## 1. Hiệu năng bám bắt đa mục tiêu MSTrack (Bảng 4)\n\n"
        "| Chỉ số | Đo đạc trên Scene01 | Bài báo công bố (Bảng 4) |\n"
        "|---|---|---|\n"
        f"| **MOTA** (Tracking Accuracy) | **{vis_metrics['MOTA']}** | 0.941 |\n"
        f"| **IDF1** (ID F1-Score) | **{vis_metrics['IDF1']}** | 0.942 |\n"
        f"| **IDP** (ID Precision) | **{vis_metrics['IDP']}** | 0.979 |\n"
        f"| **IDR** (ID Recall) | **{vis_metrics['IDR']}** | 0.907 |\n"
        f"| **IDSw** (Identity Switches) | **{int(vis_metrics['IDSw'])}** | 3 |\n\n"
        "## 2. Kết quả hợp nhất quỹ đạo đa nguồn (Bảng 5)\n\n"
        "| Chỉ số | Kết quả thực nghiệm | Mô tả |\n"
        "|---|---|---|\n"
        f"| **TPF** (True Positive) | **{fusion_metrics['TPF'] * 100:.1f}%** | "
        "Hợp nhất 3 nguồn (Hộp đỏ) |\n"
        f"| **PPF** (Part Positive) | **{fusion_metrics['PPF'] * 100:.1f}%** | "
        "Hợp nhất 2 nguồn (Hộp vàng) |\n"
        f"| **Single Modality** | **{fusion_metrics['SingleModality'] * 100:.1f}%** | "
        "Đơn cảm biến (Hộp xanh lá) |\n"
        f"| **Total Instances** | **{int(fusion_metrics['TotalFusedInstances'])}** | "
        "Tổng thể hiện quỹ đạo |\n\n"
        "## 3. Video kết quả\n"
        f"- **Video tạo ra**: `{output_video_path}` (1280x720 @ 25 fps).\n"
        "- **Quy chuẩn hiển thị**: Đồng bộ với `sample_radar_demo.mp4`.\n"
    )

    report_path = os.path.join(output_dir, "metrics_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info("Đã xuất báo cáo đánh giá định lượng tới: %s", report_path)
    logger.info(
        "Tóm tắt: MOTA=%s, IDF1=%s, IDSw=%d, TPF=%.1f%%, PPF=%.1f%%",
        vis_metrics["MOTA"],
        vis_metrics["IDF1"],
        int(vis_metrics["IDSw"]),
        fusion_metrics["TPF"] * 100,
        fusion_metrics["PPF"] * 100,
    )


def main() -> None:
    """Điểm nhập phân tích tham số dòng lệnh CLI và chạy pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Chạy hệ thống hợp nhất nhận thức tàu thuyền đa nguồn trên Scene01"
    )
    parser.add_argument("--scene", type=str, default="datasets/scene01")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--start_frame", type=int, default=7450)
    parser.add_argument("--num_frames", type=int, default=833)
    parser.add_argument("--full", action="store_true", default=False)
    parser.add_argument(
        "--id_mode",
        type=str,
        choices=["sample_demo", "tracker", "gt"],
        default="sample_demo",
        help="Chế độ hiển thị ID: 'sample_demo', 'tracker', 'gt'",
    )
    args = parser.parse_args()

    settings = PipelineSettings(
        scene_dir=args.scene,
        output_dir=args.output,
        start_frame=args.start_frame,
        num_frames=args.num_frames,
        render_full=args.full,
        id_mode=IDDisplayMode(args.id_mode),
    )

    run_pipeline(settings)


if __name__ == "__main__":
    main()
