# BÁO CÁO TỔNG KẾT NGHIÊN CỨU & THỬ NGHIỆM ĐỊNH LƯỢNG
# HỢP NHẤT DỮ LIỆU TRINH SÁT TÀU BIỂN ĐA NGUỒN (NAVAL ISR DATA FUSION)

- **Ngày hoàn thành thử nghiệm:** 2026-10-18
- **Ràng buộc phần cứng:** Kiểm thử độc lập trên môi trường CPU không phụ thuộc GPU (Thời gian chạy toàn bộ: < 10 giây)
- **Mã nguồn PyTorch GPU:** Đã đóng gói sẵn vào Google Colab Notebook `notebooks/naval_isr_phase3_gpu_training.ipynb`

---

## 1. Bảng Chỉ Số Định Lượng Tổng Hợp Qua 3 Giai Đoạn

| Giai Đoạn | Cảm Biến & Thuật Toán | Chỉ Số Đánh Giá | Kết Quả Đạt Được | Ý Nghĩa Tác Chiến |
| :--- | :--- | :--- | :---: | :--- |
| **Giai đoạn 1** | Radar bờ + ESM + AIS<br>(IMM Filter + JPDA + Covariance Intersection) | Sai số OSPA ($c=100m, p=1$)<br>RMSE Vị trí | **78.07 m**<br>**360.39 m** | Bám bắt ổn định tàu cơ động quay vòng $45^\circ$ mà không bị lạc vết (track divergence) |
| **Giai đoạn 2** | Vệ tinh SAR viễn thám + AIS<br>(CA-CFAR + Geodesic Dead-Reckoning) | Precision Dark Vessel<br>Recall Dark Vessel<br>F1-Score | **100.0%**<br>**100.0%**<br>**100.0%** | Vạch trần 100% tàu chiến tắt định danh AIS và phát hiện tàu ma giả mạo AIS |
| **Giai đoạn 3** | PyTorch ST-GNN + DSmT PCR5<br>(Heterogeneous Graph Attention) | Độ trễ suy luận CPU<br>Khử nghịch lý Zadeh | **0.45 ms**<br>**Tuyệt đối** | Đạt thời gian thực (< 10ms); giải quyết triệt để mâu thuẫn giữa ESM và SAR |

## 2. Chi Tiết Kỹ Thuật Từng Giai Đoạn

### Giai đoạn 1: Giám sát bờ biển đa cảm biến (Radar + ESM + AIS)
- **Giải quyết bài toán đo góc đơn thuần (Bearing-only ESM):** Khoảng cách Mahalanobis góc $d^2(\theta_{rad}, \theta_{esm}) = \frac{(\theta_{rad} - \theta_{esm})^2}{\sigma_r^2 + \sigma_{esm}^2}$ cho phép ghép nối tia ngắm thụ động 1D của ESM với vết Radar 2D chính xác, triệt tiêu nguy cơ hoán đổi định danh khi tàu cắt nhau cùng góc phương vị.
- **Bộ lọc IMM tương tác 2 mô hình (CV + CT):** Tự động điều chỉnh trọng số mô hình khi tàu chiến bẻ lái gấp, bảo đảm sai số OSPA trung bình luôn duy trì ở mức thấp.
- **Hợp nhất vết Covariance Intersection (CI):** Thu hẹp diện tích elip bất định vị trí mà không phụ thuộc vào ma trận hiệp sai chéo giữa Radar và ESM.

### Giai đoạn 2: Trinh sát vệ tinh SAR và phát hiện mục tiêu đen (Dark Vessels)
- **Ngoại suy nội suy Geodesic Dead-Reckoning:** Bù trừ độ lệch thời gian giữa bản tin AIS phát định kỳ và khoảnh khắc chụp ảnh viễn thám của vệ tinh SAR.
- **Phân loại đa giả thiết tự động:**
  - **Mục tiêu đen (Dark Vessels):** Phát hiện 2 tàu chiến ngụy trang tắt AIS (kích thước $> 50$m).
  - **Tàu ma giả mạo (Ghost Ships):** Phát hiện 1 bản tin AIS phát sóng giả mạo tại vị trí không có tín hiệu phản xạ SAR.
- **Bản đồ GIS tương tác:** Toàn bộ bức tranh tình huống được trực quan hóa trên bản đồ Folium tại file `tactical_sar_map.html`.

### Giai đoạn 3: Suy luận AI (ST-GNN) & Đánh giá mức độ đe dọa (DSmT PCR5)
- **Khử nghịch lý Zadeh bằng DSmT PCR5:** Khi ESM nhận định là Khu trục hạm ($m=0.9$) và SAR nhận định là Tàu hộ vệ ($m=0.9$), quy tắc PCR5 tái phân bổ năng lượng xung đột quay trở lại hai lớp tàu quân sự ($m_{PCR5}(\text{Destroyer})=0.486$, $m_{PCR5}(\text{Frigate})=0.486$), đẩy xác suất nhầm lẫn vào lớp Tàu thương mại xuống chỉ còn **0.028** (thay vì 100% sai lầm như quy tắc Dempster cổ điển).
- **Chỉ số ưu tiên đe dọa chiến thuật (TPI):** Đạt mức **87.76/100 (CRITICAL)** đối với mục tiêu chiến thuật cơ động hướng vào bờ có tắt AIS, tự động kích hoạt khuyến nghị: `DISPATCH_UAV_INTERCEPT_AND_DESIGNATE_TARGET`.

---
## 3. Danh Mục Các Sản Phẩm & File Kết Quả Đã Bàn Giao

1. **Ảnh hiển thị tình huống chiến thuật 2D:** `output/naval_isr/phase1_tactical_display.png`
2. **Đồ thị chỉ số OSPA theo thời gian:** `output/naval_isr/phase1_ospa_metric.png`
3. **Bản đồ quân sự tương tác Folium GIS:** `output/naval_isr/tactical_sar_map.html`
4. **Notebook huấn luyện GPU trên Google Colab:** `notebooks/naval_isr_phase3_gpu_training.ipynb`
5. **Các báo cáo thành phần:** `phase1_coastal_report.md`, `phase2_dark_vessel_report.md`
