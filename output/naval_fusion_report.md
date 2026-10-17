# Báo Cáo Đánh Giá Hiệu Năng Phân Hệ Hợp Nhất Động Học Hải Quân
**Kịch bản:** `scenario2`

| Chế độ thử nghiệm | RMSE Vị trí (m) | RMSE Vận tốc (m/s) | Số vết xác nhận | ID Switches | Tỷ lệ bám vết (MT) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Radar-Only | 7.29 | 0.86 | 48 | 0 | 0.0% |
| Radar + IR | 7.99 | 2.43 | 84 | 21 | 100.0% |
| Hợp nhất Toàn diện (Radar + IR + EO + LiDAR) | 6.52 | 2.23 | 110 | 29 | 100.0% |

### Tóm tắt & Phân tích đối chứng:
- **Độ chính xác vị trí cao nhất:** Cấu hình `Hợp nhất Toàn diện (Radar + IR + EO + LiDAR)` đạt RMSE vị trí `6.52m`.
- **Nhảy định danh vết:** Tổng cộng 50 lần nhảy ID phát hiện giữa các chế độ thử nghiệm.
- **Tỷ lệ duy trì bám vết (MT):** Dao động từ `0.0%` đến `100.0%`.
