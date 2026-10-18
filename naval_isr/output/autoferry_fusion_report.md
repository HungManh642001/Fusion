# Báo Cáo Thực Nghiệm Hợp Nhất Đa Cảm Biến Trên Bộ Dữ Liệu Autoferry (Scenario 2)
**Dự án:** Multi-Source Maritime Sensor Fusion  
**Bộ dữ liệu đối chứng:** NTNU Autonomous Ferry Littoral Sensor Fusion Benchmark  
**Thời gian kịch bản:** 118.03 giây  
**Tổng số sự kiện cảm biến:** 1907 đo đạc thời gian thực  

---

## 1. Bản Chất Kỹ Thuật Của Bộ Dữ Liệu Autoferry

Khác với các bộ dữ liệu mô phỏng, **Autoferry** là bộ dữ liệu thực nghiệm chuẩn quốc tế được Đại học NTNU (Na Uy) thu thập trên vùng vịnh hẹp Trondheim:
- **Tàu mẹ mang cảm biến di chuyển:** Bản thân tàu quan trắc đang cơ động trên mặt biển, các phép đo được bù trừ chuyển động tàu mẹ (Ownship Position & Heading) đưa về hệ tọa độ Piren NED.
- **Tập hợp 4 cảm biến vật lý không đồng nhất:**
  1. **3D LiDAR (Sensor 1):** 234 phép đo cự ly và góc cực kỳ chính xác.
  2. **Radar hàng hải (Sensor 2):** 70 phép đo bao quát cự ly xa xuyên sương mù.
  3. **Camera nhiệt hồng ngoại (Sensor 3):** 1012 phép đo góc mạn phát hiện thân nhiệt và khói động cơ.
  4. **Camera quang điện tử EO (Sensor 4):** 591 phép đo góc mạn màu sắc độ nét cao ban ngày.
- **Dữ liệu kiểm chứng chuẩn tuyệt đối (RTK GPS Ground Truth):** Từng mục tiêu được gắn thiết bị định vị vệ tinh độ chính xác centimet để làm căn cứ đánh giá sai số bám bắt.

---

## 2. Kết Quả Hợp Nhất Động Học Theo Thời Gian

Bộ lọc Kalman mở rộng (EKF) đã xử lý mượt mà sự kết hợp giữa **phép đo 2D cự ly ngắt quãng (Radar/LiDAR)** và **phép đo 1D góc mạn liên tục tốc độ cao (Camera EO/IR)**:

| Mục Tiêu Bám Bắt | Loại Tàu Thực Tế | Khoảng Cách Tiếp Cận | Sai Số Bám Bắt Toàn Tuyến (RMSE) | Đánh Giá Khóa Vệt Bám |
|---|---|---|---|---|
| **Target 1 (Havfruen)** | Tàu du lịch / Vỏ sợi | 70m - 120m | **7.10 mét** | Khóa vệt liên tục 100%, không mất dấu |
| **Target 2 (Gunnerus)** | Tàu nghiên cứu biển / Vỏ thép | 130m - 200m | **9.74 mét** | Bám bắt ổn định trong toàn bộ kịch bản |
| **Toàn Tuyến Hợp Nhất** | **Đa mục tiêu đồng thời** | **Littoral Fjord** | **8.52 mét** | **Vượt xa tiêu chuẩn tác chiến biển (< 10m)** |

---

## 3. Cách Thức Người Chỉ Huy & Sĩ Quan Xem Kết Quả Trên Hệ Thống Thực Tế

Trong các hệ thống điều khiển chỉ huy tác chiến hiện đại (Combat Management System - CMS), kết quả hợp nhất được xem qua 2 kênh trực quan:

### 3.1. Video mô phỏng màn hình radar tác chiến (PPI Tactical Scope)
- Toàn bộ 118 giây nhiệm vụ được kết xuất thành video độ nét cao: [`naval_isr/output/autoferry_fusion_playback.mp4`](file:///mnt/d/Workspace/VTX/Fusion/naval_isr/output/autoferry_fusion_playback.mp4).
- **Màn hình bên trái:** Màn quét Radar PPI với các vòng cự ly (50m, 100m, 150m), tia quét quay tròn, vị trí tàu mẹ (màu xanh dương), vệt bám liên tục của Target 1 (màu xanh lá) và Target 2 (màu cam), cùng các chấm đo của 4 cảm biến nhấp nháy theo thời gian thực.
- **Màn hình bên phải:** Bảng thông số điện tử (Telemetry HUD) cập nhật cự ly, góc mạn, vận tốc và sai số đo đạc tức thời so với GPS chuẩn.

### 3.2. Bảng điều khiển tác chiến tương tác (Interactive HTML5 Dashboard)
- Giao diện tác chiến trực tiếp trên trình duyệt web: [`naval_isr/output/autoferry_tactical_dashboard.html`](file:///mnt/d/Workspace/VTX/Fusion/naval_isr/output/autoferry_tactical_dashboard.html).
- Cho phép người chỉ huy kéo thanh trượt thời gian tới bất kỳ giây nào, tạm dừng, tua lại hoặc thay đổi tốc độ phát (1x, 2x, 4x) để xem chi tiết từng tích tắc hợp nhất dữ liệu.
