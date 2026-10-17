#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/calibrate_matrices.py
=============================
Công cụ hiệu chuẩn không gian đa cảm biến (Multi-Sensor Spatial Calibration)
dành cho hệ thống giám sát tàu thuyền thông minh (Sensors 2025, 25, 2179).

Mục tiêu:
1. Tính ma trận Affine M_geo2rad (GPS [lon, lat, 1] -> Radar [xr, yr]) theo Công thức 10.
2. Tính ma trận Chiếu F_rad2vis (Radar [xr, yr, 1] -> Visible [xv, yv, wv, hv]) theo Công thức 6.
3. Đánh giá sai số hồi quy (Reprojection Error RMSE tính bằng pixel và mét).
4. Xuất mã nguồn Python định dạng sẵn để dán trực tiếp vào `src/coord_transform.py`.

Sử dụng:
    python scripts/calibrate_matrices.py --scene datasets/scene01
    python scripts/calibrate_matrices.py --demo
"""

import os
import sys
import argparse
from typing import Tuple, List, Dict
import numpy as np
import cv2

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Hằng số quy đổi vật lý trắc địa
EARTH_RADIUS = 6371000.0  # mét
# Đối với ảnh Radar 2048x2048 bao quát bán kính 5 km (đường kính 10 km = 10,000 m):
RADAR_METERS_PER_PIXEL = 10000.0 / 2048.0  # ~4.8828 mét / pixel


# ==============================================================================
# 1. HÀM TOÁN HỌC: HIỆU CHUẨN MA TRẬN AFFINE (GPS -> RADAR)
# ==============================================================================
def estimate_affine_geo_to_radar(
    pts_gps: np.ndarray,
    pts_radar: np.ndarray,
    use_ransac: bool = True,
    ransac_thresh_px: float = 3.0
) -> Tuple[np.ndarray, float, float]:
    """
    Tính ma trận Affine 2x3 (M_geo2rad) ánh xạ từ [lon, lat, 1]^T sang [xr, yr]^T
    theo Công thức (10) trong bài báo Sensors 2025, 25, 2179:
        [xr, yr]^T = M_geo2rad @ [lon, lat, 1]^T

    Tham số:
        pts_gps: Mảng Nx2 chứa tọa độ kinh độ - vĩ độ [[lon_1, lat_1], ..., [lon_N, lat_N]]
        pts_radar: Mảng Nx2 chứa tọa độ pixel radar [[xr_1, yr_1], ..., [xr_N, yr_N]]
        use_ransac: True để dùng RANSAC loại bỏ nhiễu/outlier; False dùng Least Squares thuần
        ransac_thresh_px: Ngưỡng RANSAC (pixel)

    Trả về:
        M_geo2rad: Ma trận Affine kích thước 2x3 (float64)
        rmse_pixels: Sai số bình phương trung bình (RMSE) tính bằng pixel
        rmse_meters: Sai số tương đương ước tính trên thực địa (mét)
    """
    assert len(pts_gps) >= 3, "Cần tối thiểu 3 cặp điểm mốc không thẳng hàng để giải ma trận Affine 2D!"
    assert len(pts_gps) == len(pts_radar), "Số lượng điểm GPS và Radar phải bằng nhau!"

    pts_gps = np.asarray(pts_gps, dtype=np.float64)
    pts_radar = np.asarray(pts_radar, dtype=np.float64)

    if use_ransac and len(pts_gps) >= 4:
        # Dùng RANSAC của OpenCV để loại trừ các điểm GPS bị trôi sóng
        M_est, inliers = cv2.estimateAffine2D(
            pts_gps.astype(np.float32),
            pts_radar.astype(np.float32),
            ransacReprojThreshold=ransac_thresh_px,
            confidence=0.99
        )
        if M_est is None:
            # Fallback về OLS nếu RANSAC không hội tụ
            A = np.hstack([pts_gps, np.ones((len(pts_gps), 1), dtype=np.float64)])
            M_trans, _, _, _ = np.linalg.lstsq(A, pts_radar, rcond=None)
            M_est = M_trans.T
        else:
            M_est = M_est.astype(np.float64)
    else:
        # Giải hệ phương trình tuyến tính cực tiểu hóa sai số bình phương (OLS)
        # Phương trình: A * M^T = B
        # Trong đó: A có kích thước Nx3 [lon, lat, 1], B có kích thước Nx2 [xr, yr]
        A = np.hstack([pts_gps, np.ones((len(pts_gps), 1), dtype=np.float64)])
        M_trans, _, _, _ = np.linalg.lstsq(A, pts_radar, rcond=None)
        M_est = M_trans.T  # Kích thước 2x3

    # Đánh giá sai số hồi quy (Reprojection Error)
    pts_homo = np.hstack([pts_gps, np.ones((len(pts_gps), 1), dtype=np.float64)])
    pts_pred = pts_homo @ M_est.T
    diff = pts_radar - pts_pred
    rmse_pixels = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    rmse_meters = rmse_pixels * RADAR_METERS_PER_PIXEL

    return M_est, rmse_pixels, rmse_meters


# ==============================================================================
# 2. HÀM TOÁN HỌC: HIỆU CHUẨN MA TRẬN CHIẾU (RADAR -> CAMERA)
# ==============================================================================
def estimate_projection_radar_to_visible(
    radar_centers: np.ndarray,
    visible_bboxes: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Tính ma trận Chiếu F kích thước 4x3 ánh xạ từ tọa độ tâm Radar [xr, yr, 1]^T
    sang Bounding Box Camera [xv, yv, wv, hv]^T theo Công thức (6) của bài báo:
        Bbox_gate_vis = F @ [xr_center, yr_center, 1]^T

    Tham số:
        radar_centers: Mảng Nx2 chứa tọa độ tâm vệt radar [[xr_c1, yr_c1], ...]
        visible_bboxes: Mảng Nx4 chứa hộp bao camera [[xv_1, yv_1, wv_1, hv_1], ...]

    Trả về:
        F_rad2vis: Ma trận chiếu kích thước 4x3 (float64)
        rmse_center_px: Sai số định vị tâm hộp bao trên camera (pixel)
        mean_iou: Độ trùng khớp IoU trung bình giữa cổng dự báo và hộp thực tế
    """
    assert len(radar_centers) >= 3, "Cần tối thiểu 3 cặp hộp bao đồng bộ để giải ma trận chiếu F!"
    assert len(radar_centers) == len(visible_bboxes), "Số lượng mẫu Radar và Visible phải bằng nhau!"

    radar_centers = np.asarray(radar_centers, dtype=np.float64)
    visible_bboxes = np.asarray(visible_bboxes, dtype=np.float64)

    # Xây dựng hệ phương trình: X * F^T = Y
    # X có kích thước Nx3 [xr, yr, 1]
    # Y có kích thước Nx4 [xv, yv, wv, hv]
    # F^T có kích thước 3x4 => F có kích thước 4x3
    X_homo = np.hstack([radar_centers, np.ones((len(radar_centers), 1), dtype=np.float64)])
    F_trans, _, _, _ = np.linalg.lstsq(X_homo, visible_bboxes, rcond=None)
    F_est = F_trans.T  # Kích thước 4x3

    # Đánh giá sai số hồi quy
    y_pred = X_homo @ F_est.T  # Nx4 dự báo [xv, yv, wv, hv]
    
    # 1. Sai số tâm (Center distance error)
    center_pred_x = y_pred[:, 0] + y_pred[:, 2] / 2.0
    center_pred_y = y_pred[:, 1] + y_pred[:, 3] / 2.0
    center_gt_x = visible_bboxes[:, 0] + visible_bboxes[:, 2] / 2.0
    center_gt_y = visible_bboxes[:, 1] + visible_bboxes[:, 3] / 2.0
    dist_sq = (center_pred_x - center_gt_x) ** 2 + (center_pred_y - center_gt_y) ** 2
    rmse_center_px = float(np.sqrt(np.mean(dist_sq)))

    # 2. Tính chỉ số IoU trung bình
    ious = []
    for i in range(len(visible_bboxes)):
        b_gt = visible_bboxes[i]
        b_pr = y_pred[i]
        xx1 = max(b_gt[0], b_pr[0])
        yy1 = max(b_gt[1], b_pr[1])
        xx2 = min(b_gt[0] + b_gt[2], b_pr[0] + b_pr[2])
        yy2 = min(b_gt[1] + b_gt[3], b_pr[1] + b_pr[3])
        w = max(0.0, xx2 - xx1)
        h = max(0.0, yy2 - yy1)
        inter = w * h
        union = b_gt[2] * b_gt[3] + b_pr[2] * b_pr[3] - inter
        iou = inter / max(union, 1e-6)
        ious.append(iou)
    mean_iou = float(np.mean(ious))

    return F_est, rmse_center_px, mean_iou


# ==============================================================================
# 3. HÀM TỰ ĐỘNG KHAI THÁC & TRÍCH XUẤT CẶP ĐIỂM TỪ THƯ MỤC DỮ LIỆU SCENE
# ==============================================================================
def extract_calibration_pairs_from_scene(scene_dir: str):
    """
    Tự động đọc AIS.txt, radar_track_gt.csv và visible_track_gt.csv trong thư mục
    scene để trích xuất các cặp điểm mốc phục vụ hiệu chuẩn.
    """
    import pandas as pd
    from src.ais_decoder import AISDecoder, extrapolate_kinematic

    print(f"[*] Đang nạp dữ liệu từ thư mục: {scene_dir}...")
    
    # 1. Đọc danh sách thời gian radar
    radar_dir = os.path.join(scene_dir, 'radar_images')
    radar_files = sorted([f for f in os.listdir(radar_dir) if f.endswith('.jpg')])
    radar_timestamps = [int(f.replace('.jpg', '')) for f in radar_files]
    start_ts = radar_timestamps[0]

    # 2. Đọc Radar GT
    radar_gt_path = os.path.join(scene_dir, 'radar_track_gt.csv')
    df_rad = pd.read_csv(radar_gt_path, header=None)

    # 3. Đọc Visible GT
    vis_gt_path = os.path.join(scene_dir, 'visible_track_gt.csv')
    df_vis = pd.read_csv(vis_gt_path, header=None)

    # 4. Giải mã AIS
    ais_path = os.path.join(scene_dir, 'AIS.txt')
    decoder = AISDecoder()
    ais_messages = decoder.decode_file(ais_path)

    # Lọc các bản tin AIS của tàu mẫu trong luồng chính (MMSI 413806072)
    target_mmsi = 413806072
    mmsi_msgs = [m for m in ais_messages if m.mmsi == target_mmsi]

    # A. Trích xuất cặp điểm đồng bộ giữa AIS, Radar Track 26 và Visible Track 3
    gps_pts = []
    radar_pts = []
    radar_boxes = []
    vis_boxes = []

    # Tàu MMSI 413806072 di chuyển qua các vòng quét của radar (Track ID = 26 trong nhãn ground truth)
    rad_t26 = df_rad[df_rad[1] == 26]
    print(f"[*] Tìm thấy {len(rad_t26)} khung quét chứa vết Radar Track ID 26 (MMSI: {target_mmsi})...")

    for _, r_row in rad_t26.iterrows():
        r_frame = int(r_row[0])
        r_ts = radar_timestamps[r_frame]
        r_xc = float(r_row[2]) + float(r_row[4]) / 2.0
        r_yc = float(r_row[3]) + float(r_row[5]) / 2.0

        # 1. Ghép với AIS (tìm bản tin gần nhất rồi ngoại suy về r_ts)
        close_m = min(mmsi_msgs, key=lambda m: abs(m.ts - r_ts))
        lat_ext, lon_ext = extrapolate_kinematic(close_m, target_ts=r_ts)
        gps_pts.append([lon_ext, lat_ext])
        radar_pts.append([r_xc, r_yc])

        # 2. Ghép với Visible Track ID 3 tại khung hình camera đồng bộ
        v_frame = int(round((r_ts - start_ts) / 40.0))
        v_sub = df_vis[(df_vis[0] == v_frame) & (df_vis[1] == 3)]
        if len(v_sub) > 0:
            v_row = v_sub.iloc[0]
            v_box = [float(v_row[2]), float(v_row[3]), float(v_row[4]), float(v_row[5])]
            radar_boxes.append([r_xc, r_yc])
            vis_boxes.append(v_box)

    print(f"[*] Đã trích xuất thành công {len(gps_pts)} cặp GPS-Radar và {len(radar_boxes)} cặp Radar-Visible đồng bộ!")
    return np.array(gps_pts), np.array(radar_pts), np.array(radar_boxes), np.array(vis_boxes)


# ==============================================================================
# 4. HÀM DEMO: HIỆU CHUẨN BẰNG ĐIỂM MỐC THỦ CÔNG (DÙNG CHO TRẠM MỚI BẤT KỲ)
# ==============================================================================
def run_manual_calibration_demo():
    """
    Mô phỏng trường hợp triển khai trên trạm mới: Kỹ sư nhập 5 điểm mốc thực tế
    được đo đạc độc lập để sinh ra ngay lập tức ma trận biến đổi.
    """
    print("\n" + "="*70)
    print("DEMO: HIỆU CHUẨN MA TRẬN CHO TRẠM QUAN SÁT MỚI BẰNG ĐIỂM MỐC THỦ CÔNG")
    print("="*70)
    
    # 5 điểm mốc thực tế (ví dụ: phao số 1, phao số 2, trụ cầu 1, trụ cầu 2, ngọn hải đăng)
    # Tọa độ GPS [Kinh độ, Vĩ độ]
    custom_gps = np.array([
        [114.297007, 30.570142],
        [114.290170, 30.571580],
        [114.309690, 30.589050],
        [114.286900, 30.561360],
        [114.315750, 30.601410]
    ], dtype=np.float64)

    # Tọa độ tương ứng trên màn ảnh Radar 2048x2048 [xr, yr]
    custom_radar = np.array([
        [1737.5, 1162.0],
        [1759.8, 1164.2],
        [1410.2, 1025.1],
        [1920.4, 1228.6],
        [1208.5,  930.4]
    ], dtype=np.float64)

    print(f"[*] Đang giải ma trận Affine M_geo2rad từ {len(custom_gps)} điểm mốc...")
    M_calib, rmse_px, rmse_m = estimate_affine_geo_to_radar(custom_gps, custom_radar, use_ransac=False)

    print("\n[+] KẾT QUẢ HIỆU CHUẨN MA TRẬN AFFINE (GPS -> RADAR):")
    print("-" * 50)
    print(f"Sai số bình phương trung bình (RMSE): {rmse_px:.3f} pixels")
    print(f"Sai số tương đương thực địa ước tính: {rmse_m:.2f} mét")
    print("-" * 50)
    print("Đoạn mã cấu hình (Copy & dán vào `src/coord_transform.py`):")
    print("AFFINE_GEO_TO_RADAR = np.array([")
    for row in M_calib:
        print(f"    [{row[0]:.8e}, {row[1]:.8e}, {row[2]:.8e}],")
    print("], dtype=np.float64)\n")

    # Hiệu chuẩn ma trận chiếu F từ Radar sang Camera
    custom_radar_pts = np.array([
        [1723.0, 1134.0],
        [1650.0, 1110.0],
        [1580.0, 1080.0],
        [1500.0, 1050.0]
    ], dtype=np.float64)

    custom_vis_bboxes = np.array([
        [727.0, 761.0, 377.0, 61.0],
        [810.0, 755.0, 340.0, 58.0],
        [905.0, 750.0, 300.0, 52.0],
        [1020.0, 742.0, 260.0, 48.0]
    ], dtype=np.float64)

    print(f"[*] Đang giải ma trận Chiếu F_rad2vis từ {len(custom_radar_pts)} cặp hộp bao...")
    F_calib, rmse_center, mean_iou = estimate_projection_radar_to_visible(custom_radar_pts, custom_vis_bboxes)

    print("\n[+] KẾT QUẢ HIỆU CHUẨN MA TRẬN CHIẾU (RADAR -> CAMERA):")
    print("-" * 50)
    print(f"Sai số định vị tâm hộp bao: {rmse_center:.2f} pixels")
    print(f"Chỉ số IoU trung bình giữa cổng dự báo và thực tế: {mean_iou * 100:.1f}%")
    print("-" * 50)
    print("Đoạn mã cấu hình (Copy & dán vào `src/coord_transform.py`):")
    print("F_RADAR_TO_VISIBLE = np.array([")
    for row in F_calib:
        print(f"    [{row[0]:.8e}, {row[1]:.8e}, {row[2]:.8e}],")
    print("], dtype=np.float64)")
    print("="*70)


# ==============================================================================
# HÀM MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Multi-Sensor Spatial Calibration Tool (Sensors 2025, 25, 2179)")
    parser.add_argument('--scene', type=str, default=None, help="Đường dẫn thư mục cảnh (ví dụ: datasets/scene01)")
    parser.add_argument('--demo', action='store_true', help="Chạy demo hiệu chuẩn với tập điểm mốc mẫu")
    args = parser.parse_args()

    if args.scene and os.path.exists(args.scene):
        gps_pts, rad_pts, r_boxes, v_boxes = extract_calibration_pairs_from_scene(args.scene)
        
        print("\n" + "=" * 70)
        print(f"KẾT QUẢ HIỆU CHUẨN MA TRẬN CHO BỘ DỮ LIỆU: {args.scene}")
        print("=" * 70)

        if len(gps_pts) >= 3:
            M_est, rmse_px, rmse_m = estimate_affine_geo_to_radar(gps_pts, rad_pts, use_ransac=False)
            print("\n[+] 1. MA TRẬN AFFINE (GPS -> RADAR) [Công thức 10]:")
            print("-" * 50)
            print(f"Số lượng cặp điểm mốc đồng bộ: {len(gps_pts)}")
            print(f"Sai số hồi quy bình phương (RMSE): {rmse_px:.3f} pixels")
            print(f"Sai số tương đương trên mặt nước: {rmse_m:.2f} mét")
            print("-" * 50)
            print("Đoạn mã cấu hình (Copy & dán vào `src/coord_transform.py`):")
            print("AFFINE_GEO_TO_RADAR = np.array([")
            for row in M_est:
                print(f"    [{row[0]:.8e}, {row[1]:.8e}, {row[2]:.8e}],")
            print("], dtype=np.float64)\n")

        if len(r_boxes) >= 3:
            F_est, rmse_c, iou = estimate_projection_radar_to_visible(r_boxes, v_boxes)
            print("[+] 2. MA TRẬN CHIẾU (RADAR -> CAMERA) [Công thức 6]:")
            print("-" * 50)
            print(f"Số lượng cặp hộp bao đồng bộ: {len(r_boxes)}")
            print(f"Sai số định vị tâm hộp bao: {rmse_c:.2f} pixels")
            print(f"Chỉ số IoU trùng khớp trung bình: {iou * 100:.1f}%")
            print("-" * 50)
            print("Đoạn mã cấu hình (Copy & dán vào `src/coord_transform.py`):")
            print("F_RADAR_TO_VISIBLE = np.array([")
            for row in F_est:
                print(f"    [{row[0]:.8e}, {row[1]:.8e}, {row[2]:.8e}],")
            print("], dtype=np.float64)")
            print("=" * 70)
    else:
        run_manual_calibration_demo()


if __name__ == '__main__':
    main()
