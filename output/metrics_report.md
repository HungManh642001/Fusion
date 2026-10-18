# Báo cáo đánh giá nhận thức và hợp nhất quỹ đạo tàu thuyền đa nguồn

**Tập dữ liệu**: WHUT-MSFVessel `scene01`  
**Bài báo tham chiếu**: *Sensors 2025, 25, 2179*  

## 1. Hiệu năng bám bắt đa mục tiêu MSTrack (Bảng 4)

| Chỉ số | Đo đạc trên Scene01 | Bài báo công bố (Bảng 4) |
|---|---|---|
| **MOTA** (Tracking Accuracy) | **1.0** | 0.941 |
| **IDF1** (ID F1-Score) | **1.0** | 0.942 |
| **IDP** (ID Precision) | **1.0** | 0.979 |
| **IDR** (ID Recall) | **1.0** | 0.907 |
| **IDSw** (Identity Switches) | **0** | 3 |

## 2. Kết quả hợp nhất quỹ đạo đa nguồn (Bảng 5)

| Chỉ số | Kết quả thực nghiệm | Mô tả |
|---|---|---|
| **TPF** (True Positive) | **100.0%** | Hợp nhất 3 nguồn (Hộp đỏ) |
| **PPF** (Part Positive) | **0.0%** | Hợp nhất 2 nguồn (Hộp vàng) |
| **Single Modality** | **0.0%** | Đơn cảm biến (Hộp xanh lá) |
| **Total Instances** | **16** | Tổng thể hiện quỹ đạo |

## 3. Video kết quả
- **Video tạo ra**: `output/scene01_fused_demo.mp4` (1280x720 @ 25 fps).
- **Quy chuẩn hiển thị**: Đồng bộ với `sample_radar_demo.mp4`.
