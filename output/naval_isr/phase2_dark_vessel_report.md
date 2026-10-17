# Báo Cáo Thử Nghiệm Giai Đoạn 2: Trinh Sát Vệ Tinh SAR & Nhận Diện Mục Tiêu Tắt AIS (Dark Vessels)

- **Thời điểm quan sát vệ tinh (Epoch):** 1718000000.0 UTC
- **Số lượng mục tiêu phát hiện qua ảnh SAR:** 4
- **Số lượng bản tin AIS trong khu vực:** 2
- **Ngưỡng chiều dài tàu quân sự:** $\ge 50.0$ m

## 1. Kết Quả Phân Loại Chiến Thuật Tự Động

| Loại Mục Tiêu | Số lượng | Mức Độ Cảnh Báo | Hành Động Tác Chiến Khuyến Nghị |
| :--- | :---: | :---: | :--- |
| **Tàu Hợp Tác (Cooperative)** | 1 | <span style='color:green;'>XANH (Bình thường)</span> | Duy trì giám sát định kỳ |
| **Mục Tiêu Đen (Dark Vessels)** | 2 | <span style='color:red;'>**ĐỎ (Nguy hiểm cao)**</span> | **Chỉ thị Radar bờ và UAV trinh sát quang học bay đến xác minh khẩn cấp** |
| **Tàu Ma AIS (Ghost Ships)** | 1 | <span style='color:orange;'>**VÀNG (Cảnh báo)**</span> | Điều tra nguồn phát xạ tác chiến điện tử (EW Deception) |
| **Thuyền nhỏ dân sự (Skiff)** | 1 | LAM (Thông tin) | Giám sát ven bờ |

## 2. Chi Tiết Mục Tiêu Đen Phát Hiện Được

- **Mã SAR #2:** Chiều dài ước tính **156.0 m** (kích thước khu trục hạm/tàu hộ vệ). Tọa độ: `30.7900°N, 114.2200°E`. Không phát bất kỳ tín hiệu nhận dạng hàng hải nào.
- **Mã SAR #3:** Chiều dài ước tính **68.0 m** (kích thước khu trục hạm/tàu hộ vệ). Tọa độ: `30.6100°N, 114.1500°E`. Không phát bất kỳ tín hiệu nhận dạng hàng hải nào.

## 3. Bản Đồ Tương Tác Quân Sự

Bản đồ GIS tương tác toàn cảnh khu vực tác chiến đã được xuất tại file: `tactical_sar_map.html` (có thể mở trực tiếp bằng trình duyệt Chrome/Firefox/Edge để phóng to và bấm xem thuộc tính từng mục tiêu).
