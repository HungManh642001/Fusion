"""Phase 2 CLI Runner: Spaceborne SAR Reconnaissance & Dark Vessel Classification.

Executes:
1. Synthetic / xView3-SAR benchmark vessel detection simulation
2. AIS Dead-Reckoning Geodesic Extrapolation
3. Correlation and Multi-Hypothesis Classification:
   - Cooperative vessels
   - Dark Vessels (Red Alert)
   - Spoofed Ghost Targets (Yellow Alert)
4. Renders interactive GIS Folium map (tactical_sar_map.html)
5. Generates Phase 2 technical assessment report.
"""

import os
import sys
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.naval_isr.sar_recon.sar_preprocessor import SARPreprocessor
from src.naval_isr.sar_recon.ais_correlation import AISCorrelationEngine
from src.naval_isr.sar_recon.dark_vessel_detector import DarkVesselDetector
from src.naval_isr.visualization.gis_map_renderer import GISTacticalMapRenderer


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2 SAR Reconnaissance & Dark Vessel Demo.")
    parser.add_argument("--output", type=str, default="output/naval_isr", help="Directory to save artifacts.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("🛰️  [Phase 2] Spaceborne SAR Reconnaissance & Dark Vessel Correlation")
    print("=" * 70)

    # 1. Simulate SAR Satellite Overpass at Epoch T = 1718000000 (Epoch seconds)
    sar_epoch = 1718000000.0
    print(f"📡 Satellite Observation Epoch: {sar_epoch} (UTC)")

    # Ground truth targets in surveillance AOI:
    # Target A: Large Container Carrier (180m, cooperative)
    # Target B: Dark Warship (Destroyer 155m, AIS turned off)
    # Target C: Fast Attack Craft (65m, AIS turned off)
    # Target D: Ghost AIS Ship (AIS broadcasting false pos, no vessel present)
    sar_detections = [
        {"detection_id": 1, "lat": 30.6800, "lon": 114.4100, "length_m": 182.0, "peak_sigma0_db": 18.5},
        {"detection_id": 2, "lat": 30.7900, "lon": 114.2200, "length_m": 156.0, "peak_sigma0_db": 24.2}, # Dark Warship
        {"detection_id": 3, "lat": 30.6100, "lon": 114.1500, "length_m": 68.0,  "peak_sigma0_db": 19.8}, # Dark Fast Craft
        {"detection_id": 4, "lat": 30.5200, "lon": 114.3300, "length_m": 22.0,  "peak_sigma0_db": 12.0}, # Skiff
    ]

    # Asynchronous AIS reports received 120s prior to SAR overpass
    ais_reports = [
        # Target A: Container ship broadcast 120s ago at slightly lower latitude, moving North at 12 knots
        {
            "mmsi": 413806072, "vessel_name": "COSCO_HARMONY", "lat": 30.6738, "lon": 114.4100,
            "sog_knots": 12.0, "cog_deg": 0.0, "ts": sar_epoch - 120.0
        },
        # Ghost ship broadcasting AIS at 30.7500, 114.5000 (no SAR echo there!)
        {
            "mmsi": 413999888, "vessel_name": "GHOST_PHANTOM", "lat": 30.7500, "lon": 114.5000,
            "sog_knots": 8.0, "cog_deg": 45.0, "ts": sar_epoch - 60.0
        },
    ]

    print(f"🔍 Detections extracted from SAR: {len(sar_detections)} contacts")
    print(f"📡 AIS reports in AOI window: {len(ais_reports)} vessels")

    # 2. Correlate with Geodesic Dead-Reckoning
    corr_engine = AISCorrelationEngine()
    matched, unmatched_sar, unmatched_ais = corr_engine.match_sar_to_ais(
        sar_detections=sar_detections,
        ais_records=ais_reports,
        target_ts=sar_epoch,
        gate_dist_km=1.5,
    )

    print(f"\n📊 Correlation Results:")
    print(f"   - Confirmed Matched: {len(matched)}")
    print(f"   - Unmatched SAR Detections: {len(unmatched_sar)}")
    print(f"   - Unmatched AIS Broadcasts: {len(unmatched_ais)}")

    # 3. Classify with Dark Vessel Detector
    detector = DarkVesselDetector(min_warship_length_m=50.0)
    classified = detector.classify_vessels(
        matched=matched,
        unmatched_sar=unmatched_sar,
        unmatched_ais=unmatched_ais,
    )

    dark_vessels = classified["dark_vessels"]
    ghosts = classified["spoofed_ghosts"]
    coops = classified["cooperative"]
    skiffs = classified["unidentified_small_craft"]

    print(f"\n🚨 Tactical Assessment Summary:")
    print(f"   - [GREEN] Cooperative: {len(coops)}")
    print(f"   - [RED]   Dark Vessels (Warships/Unannounced): {len(dark_vessels)}")
    for d in dark_vessels:
        print(f"             * SAR #{d['detection_id']} | Length: {d['length_m']:.1f}m | Loc: ({d['lat']:.4f}°N, {d['lon']:.4f}°E)")
    print(f"   - [YELLOW] Spoofed Ghost Ships: {len(ghosts)}")
    for g in ghosts:
        print(f"             * MMSI {g['mmsi']} | Rep Speed: {g['sog_knots']} kts | Loc: ({g['lat']:.4f}°N, {g['lon']:.4f}°E)")
    print(f"   - [INFO]  Small Non-SOLAS Craft: {len(skiffs)}")

    # 4. Render Interactive Folium Map
    renderer = GISTacticalMapRenderer()
    map_path = os.path.join(args.output, "tactical_sar_map.html")
    renderer.render_interactive_map(classified, output_html_path=map_path)
    print(f"\n🗺️  Interactive Tactical GIS Map saved to: {map_path}")

    # 5. Generate Technical Report
    report_path = os.path.join(args.output, "phase2_dark_vessel_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Thử Nghiệm Giai Đoạn 2: Trinh Sát Vệ Tinh SAR & Nhận Diện Mục Tiêu Tắt AIS (Dark Vessels)\n\n")
        f.write(f"- **Thời điểm quan sát vệ tinh (Epoch):** {sar_epoch} UTC\n")
        f.write(f"- **Số lượng mục tiêu phát hiện qua ảnh SAR:** {len(sar_detections)}\n")
        f.write(f"- **Số lượng bản tin AIS trong khu vực:** {len(ais_reports)}\n")
        f.write(f"- **Ngưỡng chiều dài tàu quân sự:** $\\ge 50.0$ m\n\n")
        f.write("## 1. Kết Quả Phân Loại Chiến Thuật Tự Động\n\n")
        f.write("| Loại Mục Tiêu | Số lượng | Mức Độ Cảnh Báo | Hành Động Tác Chiến Khuyến Nghị |\n")
        f.write("| :--- | :---: | :---: | :--- |\n")
        f.write(f"| **Tàu Hợp Tác (Cooperative)** | {len(coops)} | <span style='color:green;'>XANH (Bình thường)</span> | Duy trì giám sát định kỳ |\n")
        f.write(f"| **Mục Tiêu Đen (Dark Vessels)** | {len(dark_vessels)} | <span style='color:red;'>**ĐỎ (Nguy hiểm cao)**</span> | **Chỉ thị Radar bờ và UAV trinh sát quang học bay đến xác minh khẩn cấp** |\n")
        f.write(f"| **Tàu Ma AIS (Ghost Ships)** | {len(ghosts)} | <span style='color:orange;'>**VÀNG (Cảnh báo)**</span> | Điều tra nguồn phát xạ tác chiến điện tử (EW Deception) |\n")
        f.write(f"| **Thuyền nhỏ dân sự (Skiff)** | {len(skiffs)} | LAM (Thông tin) | Giám sát ven bờ |\n\n")
        f.write("## 2. Chi Tiết Mục Tiêu Đen Phát Hiện Được\n\n")
        for d in dark_vessels:
            f.write(f"- **Mã SAR #{d['detection_id']}:** Chiều dài ước tính **{d['length_m']:.1f} m** (kích thước khu trục hạm/tàu hộ vệ). Tọa độ: `{d['lat']:.4f}°N, {d['lon']:.4f}°E`. Không phát bất kỳ tín hiệu nhận dạng hàng hải nào.\n")
        f.write("\n## 3. Bản Đồ Tương Tác Quân Sự\n\n")
        f.write("Bản đồ GIS tương tác toàn cảnh khu vực tác chiến đã được xuất tại file: `tactical_sar_map.html` (có thể mở trực tiếp bằng trình duyệt Chrome/Firefox/Edge để phóng to và bấm xem thuộc tính từng mục tiêu).\n")

    print(f"📄 Báo cáo Phase 2 đã được lưu tại: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
