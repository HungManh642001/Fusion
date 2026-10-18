# Báo Cáo Thử Nghiệm Giai Đoạn 1: Hợp Nhất Đa Cảm Biến Chiến Thuật Bờ Biển

- **Thời gian kịch bản:** 150 giây (Chu kỳ lấy mẫu $\Delta t = 2.0$s)
- **Cảm biến mô phỏng:** Radar ven bờ chủ động ($r, \theta, \dot{r}$), Trạm ESM thụ động (Line-of-Bearing), Trạm bờ AIS
- **Sai số trung bình OSPA (c=100m, p=1):** **82.84 m**

## Kết quả Phân tích Mục tiêu Chiến thuật

| Target ID | Loại Mục Tiêu | Trạng Thái AIS | Emitter ESM | Độ chính xác bám bắt |
| :---: | :---: | :---: | :---: | :---: |
| **Target 1** | Tàu hàng (Merchant Cargo) | Hợp lệ (MMSI 413806072) | Không phát xạ | Bám bắt ổn định (CV mode) |
| **Target 2** | Tàu buôn lậu/Tình báo | Giả mạo lệch (+3.5km, +2.5km) | Không phát xạ | Phát hiện bất thường sai lệch AIS-Radar |
| **Target 3** | Tàu chiến (Dark Warship) | **TẮT AIS HOÀN TOÀN** | **Phát xạ RADAR-MPQ-TYPE052D** | Bám bắt cơ động ngoặt 45° (IMM CT mode) |

## Đánh giá Thuật toán Hợp nhất

1. **IMM (CV + CT):** Khi Target 3 thực hiện cơ động quay vòng $45^\circ$, bộ lọc tự động kích hoạt mô hình CT, ngăn chặn phân kỳ vết bám.
2. **Active-Passive JPDA:** Khoảng cách Mahalanobis góc $d^2(\theta_{rad}, \theta_{esm})$ ghép nối chính xác góc phương vị ESM với vết Radar mà không bị nhầm lẫn định danh.
3. **Covariance Intersection:** Kết hợp thông tin định vị Radar và đường ngắm ESM mà không phụ thuộc ma trận hiệp sai chéo, thu hẹp elip bất định phương vị.
