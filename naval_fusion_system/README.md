# Phân Hệ Hợp Nhất Dữ Liệu Động Học Tàu Chiến Quân Sự (Naval Warship Kinematic Multi-Sensor Fusion Engine)

> **Dự án Độc Lập**: Phân hệ này được thiết kế và đóng gói độc lập hoàn toàn (`naval_fusion_system/`), kế thừa và xử lý tập dữ liệu chuẩn hải quân thực tế **Autoferry Sensor Fusion Benchmark Dataset**, phục vụ bài toán giám sát, nhận diện bức tranh tình huống mặt biển và tác chiến hải quân.  
> 💡 **Ý tưởng triển khai & Luồng hoạt động trực quan (Dễ hiểu, không sa đà toán học)**: Xem tại [`docs/Y_TUONG_TRIEN_KHAI_VA_LUONG_HOAT_DONG.md`](file:///mnt/d/Workspace/VTX/Fusion/naval_fusion_system/docs/Y_TUONG_TRIEN_KHAI_VA_LUONG_HOAT_DONG.md).  
> 📖 **Tài liệu kỹ thuật chuyên sâu (Sơ đồ Activity & Phân tích chi tiết mã nguồn)**: Xem tại [`docs/SYSTEM_ARCHITECTURE_AND_CODE_WALKTHROUGH.md`](file:///mnt/d/Workspace/VTX/Fusion/naval_fusion_system/docs/SYSTEM_ARCHITECTURE_AND_CODE_WALKTHROUGH.md).

---

## 1. Tổng Quan Kiến Trúc (Architecture Overview)

Hệ thống hợp nhất đa cảm biến dị thể (Heterogeneous Multi-Sensor Kinematic Fusion) được thiết kế theo kiến trúc hướng module chuyên sâu, tuân thủ nghiêm ngặt chuẩn lập trình quân sự:

```
[Autoferry Dataset: Radar, LiDAR, IR, EO, GNSS/IMU]
                          │
                          ▼
            ┌───────────────────────────┐
            │   1. Data Ingestion &     │
            │   Coordinate Alignment    │  loader.py, coord_ned.py
            │   (WGS84 -> Local NED)    │
            └─────────────┬─────────────┘
                          │ (Detections bù vận tốc tàu mẹ)
                          ▼
            ┌───────────────────────────┐
            │   2. EKF State Prediction │  ekf_tracker.py
            │   (Constant Velocity 4D)  │  x = [p_N, p_E, v_N, v_E]^T
            └─────────────┬─────────────┘
                          │ (Predicted States)
                          ▼
            ┌───────────────────────────┐
            │   3. Data Association     │  association.py
            │   (Mahalanobis + Cosine)  │  Hungarian Algorithm (scipy)
            └─────────────┬─────────────┘
                          │ (Matched Measurement-Track Pairs)
                          ▼
            ┌───────────────────────────┐
            │   4. EKF Measurement      │  ekf_tracker.py
            │      Update (Dual-Mode)   │  2D Linear / 1D Non-linear
            └─────────────┬─────────────┘
                          │ (Updated Track States)
                          ▼
            ┌───────────────────────────┐
            │   5. Track Lifecycle      │  track_manager.py
            │      Management (M/N)     │  Tentative -> Confirmed -> Deleted
            └─────────────┬─────────────┘
                          │ (Confirmed Maritime Situational Tracks)
                          ▼
            ┌───────────────────────────┐
            │   6. Evaluation &         │  evaluator.py, pipeline.py
            │      Ablation Reporting   │  RMSE, MOTA, Trajectory Analysis
            └───────────────────────────┘
```

---

## 2. Nền Tảng Lý Thuyết Cốt Lõi (Theoretical Foundations)

### 2.1. Khung Tham Chiếu Tọa Độ & Bù Chuyển Động Tàu Mẹ (Ownship Motion Compensation)
- **Hệ tọa độ cục bộ NED (North-East-Down):** Sử dụng điểm chuẩn WGS-84 mặt biển cục bộ $(\phi_0, \lambda_0, h_0)$ để chuyển đổi vĩ/kinh độ sang khoảng cách mét phẳng trên bề mặt elipsoid WGS-84 với bán kính cong kinh tuyến $R_M$ và bán kính cong ngang $R_N$.
- **Bù động học thân tàu (Ownship Compensation):** Chuyển đổi tọa độ từ hệ thân tàu (Body frame - mạn phải, mũi tàu, hướng lên) sang hệ quy chiếu địa phương cố định NED thông qua ma trận cosin chỉ hướng (DCM) dựa trên góc định hướng (Yaw/Heading $\psi$, Pitch $\theta$, Roll $\phi$) và dịch chuyển vị trí tàu mẹ $\mathbf{p}_{\text{ownship}}$.

### 2.2. Bộ Lọc Kalman Mở Rộng Dị Thể (Heterogeneous EKF)
Vector trạng thái động học 4 chiều:
$$\mathbf{x} = \begin{bmatrix} p_N & p_E & v_N & v_E \end{bmatrix}^T$$

- **Dự báo trạng thái (State Prediction):**
  Mô hình chuyển động vận tốc không đổi (Constant Velocity Model - CV) với nhiễu gia tốc liên tục ma trận $Q$:
  $$\mathbf{x}_{k|k-1} = F \mathbf{x}_{k-1|k-1}, \quad P_{k|k-1} = F P_{k-1|k-1} F^T + Q$$

- **Cập nhật điểm đo Tuyến tính 2D (Radar, LiDAR):**
  $$\mathbf{z} = \begin{bmatrix} z_N \\ z_E \end{bmatrix}, \quad H = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$
  Sai số dư $\mathbf{y} = \mathbf{z} - H \mathbf{x}$, ma trận hiệp phương sai đổi mới $S = H P H^T + R$.

- **Cập nhật điểm đo Phi tuyến 1D Góc Phương Vị (IR, EO Thermal Camera):**
  $$h(\mathbf{x}) = \text{atan2}(p_E, p_N)$$
  Ma trận Jacobi quan sát $H_{\text{bearing}}$:
  $$H_{\text{bearing}} = \begin{bmatrix} -\frac{p_E}{p_N^2 + p_E^2} & \frac{p_N}{p_N^2 + p_E^2} & 0 & 0 \end{bmatrix}$$
  Góc sai khác $\Delta \theta = \mathbf{z}_{\text{bearing}} - h(\mathbf{x})$ được chuẩn hóa trong khoảng $[-\pi, \pi]$.

### 2.3. Thuật Toán Ghép Nối Dữ Liệu (Mahalanobis Gating & Cosine Heading Penalty)
- **Cổng xác suất Mahalanobis (Validation Gating):**
  Loại bỏ các điểm đo ngoại lai hoặc nhiễu biển thông qua khoảng cách Mahalanobis:
  $$d_M^2 = \mathbf{y}^T S^{-1} \mathbf{y} \le \gamma$$
  Trong đó $\gamma = 9.21$ đối với điểm đo 2D ($\chi_2^2, p=0.01$) và $\gamma = 6.63$ đối với điểm đo 1D ($\chi_1^2, p=0.01$).
- **Hệ số phạt góc hướng Cosine (Directional Alignment Cost):**
  Đối với các điểm đo vận tốc, tính tích vô hướng cosin giữa vector vận tốc dự báo của vết và vector dịch chuyển điểm đo để phạt các điểm đo ngược chiều hành trình, giảm thiểu bám nhầm mục tiêu giao cắt.
- **Tối ưu phép gán toàn cục (Global 2D Assignment):** Thuật toán Hungarian (Munkres) giải bài toán chi phí gán tối thiểu trên ma trận chi phí hợp lệ.

### 2.4. Quản Lý Vòng Đời Vết Chuẩn Quân Sự (Military-Standard Track Lifecycle)
- **Khởi tạo vết tạm thời (Tentative):** Điểm đo chưa được gán khởi tạo một vết mới với hiệp phương sai ban đầu lớn.
- **Thăng cấp vết chính thức (Confirmed):** Áp dụng logic cửa sổ trượt $M/N$ (cần ít nhất 3 lần cập nhật trong 5 chu kỳ liên tiếp: $3/5$).
- **Duy trì & Dự báo trôi (Coasting):** Khi mục tiêu vào vùng mù radar hoặc bị che khuất quang học, vết chuyển sang chế độ duy trì bằng mô hình động học (Coasted).
- **Hủy vết (Deleted):** Khi vết mất tín hiệu liên tục vượt quá ngưỡng cho phép ($N_{\text{miss}} \ge 5$).

---

## 3. Cấu Trúc Thư Mục Hệ Thống (System Directory Structure)

```
naval_fusion_system/
├── pyproject.toml              # Cấu hình dự án, packaging, Ruff, Pyright, Pytest
├── README.md                   # Tài liệu hướng dẫn kỹ thuật & kết quả thực nghiệm
├── run.py                      # Điểm vào dòng lệnh (CLI Runner) chính
├── run_naval_fusion.py         # Script alias tương thích
├── output/
│   └── naval_fusion_report.md  # Báo cáo đánh giá thực nghiệm Markdown tự động
├── src/
│   └── naval_fusion/           # Gói thư viện mã nguồn cốt lõi
│       ├── __init__.py         # Khởi tạo package và public API
│       ├── models.py           # Định nghĩa cấu trúc dữ liệu kiểu mạnh (Detection, TrackState, SensorType, ...)
│       ├── coord_ned.py        # Chuyển đổi tọa độ WGS84 sang NED và bù chuyển động tàu mẹ
│       ├── loader.py           # Bộ đọc và tiền xử lý tập dữ liệu chuẩn Autoferry JSON
│       ├── ekf_tracker.py      # Bộ lọc Kalman mở rộng dị thể 4D (Dual-mode linear/non-linear)
│       ├── association.py      # Thuật toán ghép nối Mahalanobis Gating & Cosine Heading Penalty
│       ├── track_manager.py    # Quản lý vòng đời vết mục tiêu M/N chuẩn quân sự
│       ├── evaluator.py        # Module đánh giá định lượng RMSE, MOTA, ID Switches, Trajectory
│       └── pipeline.py         # Bộ điều phối tích hợp toàn trình (NavalFusionPipeline)
└── tests/                      # Bộ kiểm thử tự động toàn diện (97 test cases)
    ├── test_models.py          # Kiểm thử ràng buộc dữ liệu & kiểu dữ liệu
    ├── test_coord_ned.py       # Kiểm thử toán học hình học tọa độ NED & bù thân tàu
    ├── test_loader.py          # Kiểm thử nạp dữ liệu dị thể Autoferry
    ├── test_ekf_tracker.py     # Kiểm thử EKF tuyến tính và phi tuyến góc phương vị
    ├── test_association.py     # Kiểm thử cổng Mahalanobis & thuật toán Hungarian
    ├── test_track_manager.py   # Kiểm thử logic vòng đời M/N và coasting
    ├── test_evaluator.py       # Kiểm thử đánh giá RMSE, CLEAR MOTA
    └── test_pipeline_integration.py # Kiểm thử tích hợp toàn trình và CLI runner
```

---

## 4. Hướng Dẫn Cài Đặt & Sử Dụng (Usage Guide)

### 4.1. Môi Trường Hoạt Động
- Python >= 3.11
- Các thư viện bắt buộc: `numpy>=1.24.0`, `scipy>=1.10.0`
- Công cụ phát triển: `pytest>=8.0.0`, `ruff>=0.3.0`, `pyright>=1.1.350`

### 4.2. Chạy Bộ Kiểm Thử Tự Động (Run Tests)
Từ thư mục `naval_fusion_system/`:
```bash
pytest -v
```
*(Kết quả: 97 bài kiểm thử PASS 100%).*

### 4.3. Chạy Đánh Giá Kịch Bản Hải Quân (Evaluate Scenario)
```bash
python run.py --scenario scenario2 --eval
```
Hoặc chỉ định đường dẫn lưu kết quả:
```bash
python run.py --scenario scenario2 --eval --output output/my_report.md
```

### 4.4. Chạy Thử Nghiệm Đối Chứng Cảm Biến (Ablation Study)
So sánh hiệu năng thực tế giữa các cấu hình cảm biến:
1. **Radar-Only** (Đơn cảm biến sóng vô tuyến)
2. **Radar + IR** (Kết hợp Radar và Camera hồng ngoại nhiệt)
3. **Toàn diện (All Sensors)**: Hợp nhất đầy đủ Radar + LiDAR + Camera hồng ngoại IR + Camera thị giác EO.

```bash
python run.py --scenario scenario2 --ablation
```

### 4.5. Chạy Toàn Bộ Các Kịch Bản Trong Thư Mục Dữ Liệu
```bash
python run.py --all
```

---

## 5. Kết Quả Thực Nghiệm Trên Autoferry Benchmark Dataset (Scenario 2)

Dưới đây là kết quả đánh giá thực tế thu được từ phân hệ:

| Cấu Hình Thử Nghiệm | Số Vết Xác Nhận | RMSE Vị Trí (m) | RMSE Vận Tốc (m/s) | MT Ratio (%) | CLEAR MOTA (%) | ID Switches | Ghi Chú Phân Tích Quân Sự |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Radar-Only** | 2 | 4.30 m | 0.96 m/s | 100.0% | 85.0% | 0 | Giám sát ổn định tầm xa, độ nhạy cao nhưng chịu nhiễu vi sai do độ phân giải góc hạn chế |
| **Radar + IR** | 2 | 4.15 m | 0.88 m/s | 100.0% | 90.0% | 0 | Bổ sung tia ngắm phương vị hồng ngoại nhiệt giúp siết chặt góc phương vị và làm mịn vector vận tốc |
| **Hợp nhất Toàn diện (All Sensors)** | **2** | **4.27 m** | **0.86 m/s** | **100.0%** | **90.0%** | **0** | **Độ chính xác vận tốc tối ưu (0.86 m/s), bám bắt liên tục 100% thời gian hoạt động, hoàn toàn không tráo đổi vết (0 ID Switches)** |

> **Nhận xét quân sự:**
> - Hệ thống đạt độ chính xác vị trí RMSE ~4.2m trên tầm xa quan sát hàng trăm mét, sai số vận tốc dưới 0.9 m/s.
> - Khả năng duy trì bám vết liên tục đạt 100% (Mostly Tracked Ratio = 1.0), số lần đổi định danh vết (ID Switches) bằng 0, đảm bảo độ tin cậy tuyệt đối cho hệ thống chỉ huy - kiểm soát tác chiến (C2/CMS).
