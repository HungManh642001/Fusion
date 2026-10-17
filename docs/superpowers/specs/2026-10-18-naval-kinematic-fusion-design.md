# TÀI LIỆU ĐẶC TẢ THIẾT KẾ KIẾN TRÚC (DESIGN SPEC)
# Phân Hệ Hợp Nhất Động Học Đa Cảm Biến Tàu Chiến (Naval Kinematic Fusion Engine)

> **Mã định danh:** `SPEC-2026-10-18-NAVAL-FUSION`  
> **Ngày lập:** 18/10/2026  
> **Trạng thái:** Chờ phê duyệt (Pending Review)  
> **Bộ dữ liệu mục tiêu:** `Autoferry Sensor Fusion Benchmark Dataset` (9 kịch bản trên biển Trondheim)  
> **Nguyên tắc bất biến:** **Độc lập 100%** – Tuyệt đối không sửa đổi, thay thế hay phụ thuộc vào mã nguồn cũ của bài báo Sensors 2025 (`src/*.py`, `main.py`).

---

## 1. TỔNG QUAN & BỐI CẢNH DỰ ÁN

### 1.1. Mục Tiêu Nghiệp Vụ
Xây dựng một phân hệ phần mềm độc lập mô phỏng **Lõi Hợp nhất Động học Đa cảm biến Tàu mẹ (Kinematic Naval Fusion Core)** phục vụ bài toán quản lý chiến đấu hải quân (CMS / C4ISR). Phân hệ giải quyết bài toán bám bắt các mục tiêu trên mặt nước hoàn toàn phi hợp tác (Non-cooperative tracking – không phụ thuộc vào AIS), có khả năng bù chuyển động của tàu mẹ trên biển theo thời gian thực và xử lý đồng thời các cảm biến dị thể (Đo cự ly/vị trí từ Radar/LiDAR và đo góc phương vị từ Camera nhiệt Hồng ngoại IR / Camera Quang học EO).

### 1.2. Các Rào Cản Kỹ Thuật Đã Giải Quyết
1. **Khử hoàn toàn sự phụ thuộc vào AIS:** Không sử dụng AIS làm hạt giống (seed) hay khóa danh tính; mục tiêu được khởi tạo và bám bắt hoàn toàn bằng các cảm biến vô tuyến và quang điện tử.
2. **Khắc phục trạm bờ tĩnh:** Thay thế ma trận Affine phẳng tĩnh bằng mô hình bù chuyển động tàu mẹ đa bậc tự do trong hệ quy chiếu quân sự **Piren NED (North-East-Down)**.
3. **Giải quyết sự khập khiễng cảm biến (Heterogeneous Modality Gap):** Tích hợp bộ lọc Kalman mở rộng (EKF) hỗ trợ 2 chế độ cập nhật: Cập nhật đầy đủ 2D (Linear Kalman) cho Radar/LiDAR và Cập nhật góc phi tuyến 1D (Nonlinear EKF) cho Camera Hồng ngoại nhiệt IR và Quang học EO.

---

## 2. BỐ CỤC KIẾN TRÚC & PHÂN RÃ MODULE

Hệ thống được tổ chức khép kín bên trong gói `src/naval_fusion/`, có điểm vào thực thi độc lập `run_naval_fusion.py` và bộ kiểm thử riêng `tests/test_naval_fusion/`.

```
VTX/Fusion/
├── src/
│   ├── (Giữ nguyên vẹn 100% mã nguồn cũ: ais_decoder, mstrack, cascade_fusion, coord_transform, ...)
│   └── naval_fusion/                   # Gói phân hệ mới độc lập hoàn toàn
│       ├── __init__.py                 # Xuất các interface và models công khai
│       ├── models.py                   # Cấu trúc dữ liệu kiểu mạnh (Dataclass / Typed)
│       ├── coord_ned.py                # Chuyển đổi tọa độ & bù vị trí tàu mẹ NED
│       ├── loader.py                   # Nạp và tiền xử lý dữ liệu Autoferry (JSON/MAT)
│       ├── ekf_tracker.py              # Bộ lọc EKF bám vết dị thể (Full 2D & Bearing-only)
│       ├── association.py              # Ghép nối Mahalanobis Gating + Phạt Cosine vận tốc
│       ├── track_manager.py            # Máy trạng thái quản lý vòng đời vết (Quy tắc M/N)
│       ├── evaluator.py                # Đánh giá sai số định lượng (RMSE Vị trí, Vận tốc, MOTA)
│       └── pipeline.py                 # Bộ điều phối luồng tích hợp toàn chu trình
│
├── tests/
│   ├── (Giữ nguyên các test cũ)
│   └── test_naval_fusion/              # Kiểm thử tự động độc lập
│       ├── test_coord_ned.py           # Unit test chuyển đổi tọa độ NED
│       ├── test_ekf_tracker.py         # Unit test toán học bộ lọc EKF (Full & Bearing)
│       ├── test_association.py         # Unit test ghép nối và phạt Cosine
│       └── test_pipeline_integration.py# Integration test chạy trên scenario2
│
└── run_naval_fusion.py                 # Điểm vào thực thi dòng lệnh (CLI Runner)
```

---

## 3. ĐẶC TẢ CẤU TRÚC DỮ LIỆU (`models.py`)

Hệ thống sử dụng các Dataclass bất biến để đảm bảo tính an toàn kiểu dữ liệu và ngăn ngừa tác dụng phụ (side-effects):

```python
from dataclasses import dataclass
from enum import IntEnum
import numpy as np

class SensorType(IntEnum):
    """Mã định danh cảm biến theo quy chuẩn Autoferry."""
    LIDAR = 1   # LiDAR 3D: Đo 2D Cartesian [x, y] trong hệ thân tàu
    RADAR = 2   # Marine Radar 360: Đo 2D Cartesian [x, y] trong hệ thân tàu
    IR = 3      # Thermal Infrared: Đo 1D Bearing angle [theta] (Radian)
    EO = 4      # Electro-Optical: Đo 1D Bearing angle [theta] (Radian)

@dataclass(frozen=True)
class OwnshipState:
    """Trạng thái động học tàu mẹ trong hệ tọa độ Piren NED."""
    timestamp: float          # Thời gian ghi nhận (Unix epoch)
    pos_north: float          # Tọa độ trục Bắc (mét)
    pos_east: float           # Tọa độ trục Đông (mét)

@dataclass
class Detection:
    """Điểm đo chuẩn hóa nhận được từ cảm biến."""
    sensor_type: SensorType
    timestamp: float
    ownship_pos: tuple[float, float]      # [N_own, E_own]
    raw_measurement: list[float]          # [x, y] hoặc [theta]
    measurement_ned: np.ndarray           # Tọa độ bù về NED thế giới: [pN, pE] hoặc [theta_NED]

@dataclass
class TrackState:
    """Trạng thái một vết mục tiêu được quản lý trong hệ NED."""
    track_id: int                         # ID duy nhất toàn cục
    state_vector: np.ndarray              # [pN, pE, vN, vE]^T (Kích thước 4x1)
    covariance: np.ndarray                # Ma trận hiệp phương sai P (Kích thước 4x4)
    last_updated: float                   # Thời điểm cập nhật cuối
    hits: int                             # Số lần ghép nối thành công
    misses: int                           # Số bước quét liên tiếp bị mất tín hiệu
    is_confirmed: bool                    # Đã xác nhận (True) hay nghi ngờ (False)
    contributing_sensors: set[SensorType] # Tập hợp cảm biến đang hỗ trợ vết
```

---

## 4. BÙ CHUYỂN ĐỘNG TÀU MẸ & TỌA ĐỘ NED (`coord_ned.py`)

- **Gốc tọa độ Piren NED:** Được cố định tại kinh vĩ độ $PirenLLA = [63.4389029^\circ N, 10.39908278^\circ E, 39.923m]$.
- **Bù tọa độ Radar / LiDAR (2D):**
  Điểm đo trả về vector $[x_{vessel}, y_{vessel}]^T$ trong hệ thân tàu. Tọa độ mục tiêu trong hệ Piren NED:
  $$\begin{bmatrix} p_N \\ p_E \end{bmatrix} = \begin{bmatrix} N_{own}(t) \\ E_{own}(t) \end{bmatrix} + \begin{bmatrix} x_{vessel} \\ y_{vessel} \end{bmatrix}$$
- **Bù góc phương vị Hồng ngoại IR / EO (1D):**
  Camera trả về góc phương vị tương đối $\theta_{meas}$ (tính từ thân tàu). Thuật toán chuẩn hóa góc về hệ quy chiếu hướng Bắc địa phương (Wrap to $[-\pi, \pi]$).

---

## 5. MÔ HÌNH TOÁN HỌC BỘ LỌC EKF DỊ THỂ (`ekf_tracker.py`)

### 5.1. Mô hình Động học & Dự báo (Prediction)
- **Vector trạng thái 4D:** $x = [p_N, p_E, v_N, v_E]^T$.
- **Ma trận chuyển trạng thái:**
  $$F(\Delta t) = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
- **Hiệp phương sai nhiễu quá trình $Q(\Delta t)$ (Continuous White Noise Acceleration):**
  $$Q(\Delta t) = q \cdot \begin{bmatrix} \frac{\Delta t^3}{3} & 0 & \frac{\Delta t^2}{2} & 0 \\ 0 & \frac{\Delta t^3}{3} & 0 & \frac{\Delta t^2}{2} \\ \frac{\Delta t^2}{2} & 0 & \Delta t & 0 \\ 0 & \frac{\Delta t^2}{2} & 0 & \Delta t \end{bmatrix}, \quad q = 0.5\text{ m/s}^2$$
- **Bước dự báo:**
  $$\hat{x}_{t|t-1} = F(\Delta t) \hat{x}_{t-1|t-1}, \quad P_{t|t-1} = F(\Delta t) P_{t-1|t-1} F(\Delta t)^T + Q(\Delta t)$$

### 5.2. Chế độ Cập nhật 1: Điểm đo Tuyến tính Đầy đủ 2D (Radar / LiDAR)
- Vector đo: $z = [p_N^{meas}, p_E^{meas}]^T$.
- Ma trận quan sát: $H = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$.
- Ma trận nhiễu đo: $R_{radar} = \text{diag}([25.0, 25.0])\text{ m}^2$ (Radar) hoặc $\text{diag}([4.0, 4.0])\text{ m}^2$ (LiDAR).
- Cập nhật Kalman chuẩn:
  $$y = z - H \hat{x}_{t|t-1}, \quad S = H P_{t|t-1} H^T + R$$
  $$K = P_{t|t-1} H^T S^{-1}, \quad \hat{x}_{t|t} = \hat{x}_{t|t-1} + K y, \quad P_{t|t} = (I - K H) P_{t|t-1}$$

### 5.3. Chế độ Cập nhật 2: Điểm đo Góc Phi tuyến 1D (Hồng ngoại IR / EO)
- Điểm đo: $z = [\theta_{meas}]$.
- Hàm quan sát phi tuyến:
  $$h(x) = \text{atan2}(p_E - E_{own}(t), p_N - N_{own}(t))$$
- Ma trận đạo hàm riêng Jacobian $H_j$ (Kích thước $1 \times 4$):
  $$d^2 = (p_N - N_{own})^2 + (p_E - E_{own})^2$$
  $$H_j = \begin{bmatrix} -\frac{p_E - E_{own}}{d^2} & \frac{p_N - N_{own}}{d^2} & 0 & 0 \end{bmatrix}$$
- Nhiễu đo góc: $R_{angle} = (0.5^\circ \cdot \frac{\pi}{180})^2 \approx 7.6 \times 10^{-5}\text{ rad}^2$.
- Cập nhật EKF:
  $$y = \text{wrap\_to\_pi}(\theta_{meas} - h(\hat{x}_{t|t-1}))$$
  $$S = H_j P_{t|t-1} H_j^T + R_{angle}, \quad K = P_{t|t-1} H_j^T S^{-1}$$
  $$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K y, \quad P_{t|t} = (I - K H_j) P_{t|t-1}$$

---

## 6. GHÉP NỐI DỮ LIỆU & QUẢN LÝ VÒNG ĐỜI VẾT (`association.py` & `track_manager.py`)

### 6.1. Ma trận Chi phí Liên kết & Gating
- **Khoảng cách Mahalanobis:** $d_M^2 = y^T S^{-1} y$.
  - Ngưỡng gating 2D (Radar): $d_M^2 \le 9.21$ (Phân phối $\chi^2$ bậc 2, $p=0.01$).
  - Ngưỡng gating 1D (Góc IR/EO): $d_{angle}^2 \le 9.0$ ($3\sigma$ kiểm tra góc).
  - Vượt ngưỡng $\rightarrow$ Gán $Cost = \infty$.
- **Hệ số phạt Cosine vận tốc:**
  $$Cost_{final} = d_M + \lambda \cdot (1 - \cos(\vec{v}_{track}, \vec{v}_{meas})), \quad \lambda = 3.0$$
- **Phân công tối ưu 1-1:** Sử dụng thuật toán Hungary (`scipy.optimize.linear_sum_assignment`).

### 6.2. Máy Trạng thái Vòng đời Vết (Luật $M/N$)
- **Khởi tạo (TENTATIVE):** Chỉ các điểm đo Radar hoặc LiDAR (có cự ly) chưa ghép mới được khởi tạo vết. Tia góc hồng ngoại đơn lẻ bị cấm tự ý khởi tạo vết mới.
- **Xác nhận (CONFIRMED):** Thỏa mãn quy tắc $3/5$: Có ít nhất 3 lần ghép nối thành công trong 5 bước đo liên tiếp. Chỉ vết `CONFIRMED` mới được xuất ra bảng điều khiển tác chiến.
- **Duy trì quán tính (COASTING):** Khi tạm thời mất tín hiệu, vết chạy quán tính theo dự báo EKF.
- **Xóa vết (DELETED):** Nếu số bước liên tiếp không có đo lường $misses \ge 5$ chu kỳ hoặc thời gian trôi $> 5.0$ giây $\rightarrow$ Xóa khỏi danh sách theo dõi.

---

## 7. BỘ CHỈ SỐ ĐÁNH GIÁ ĐỊNH LƯỢNG (`evaluator.py`)

Hệ thống so sánh trực tiếp tọa độ vết với Ground Truth có sẵn trong Autoferry:
1. **RMSE Vị trí:** $RMSE_{pos} = \sqrt{\frac{1}{K} \sum_{k=1}^K ((p_{N,k}^{est} - p_{N,k}^{gt})^2 + (p_{E,k}^{est} - p_{E,k}^{gt})^2)}$ (Mục tiêu: $< 3.0\text{ m}$).
2. **RMSE Vận tốc:** $RMSE_{vel} = \sqrt{\frac{1}{K} \sum_{k=1}^K ((v_{N,k}^{est} - v_{N,k}^{gt})^2 + (v_{E,k}^{est} - v_{E,k}^{gt})^2)}$.
3. **Chỉ số Nhảy ID (IDSw):** Mục tiêu $= 0$.
4. **Thử nghiệm Đối chứng (Ablation Study):**
   - Chạy 1: Radar-Only.
   - Chạy 2: Radar + Camera nhiệt Hồng ngoại (Radar + IR Fusion).
   - Chạy 3: Hợp nhất Toàn diện (Radar + IR + EO + LiDAR).

---

## 8. ĐIỂM VÀO THỰC THI & TÍCH HỢP HỆ THỐNG (`run_naval_fusion.py`)

Hỗ trợ giao diện dòng lệnh độc lập:
```bash
# Chạy kịch bản cơ sở scenario2 kèm đánh giá định lượng
python run_naval_fusion.py --scenario scenario2 --eval

# Chạy thử nghiệm đối chứng phân tích giá trị của cảm biến hồng ngoại
python run_naval_fusion.py --scenario scenario2 --ablation

# Chạy kiểm chứng toàn bộ 9 kịch bản
python run_naval_fusion.py --all
```
Kết quả báo cáo tự động được xuất ra màn hình và lưu thành file Markdown tại `output/naval_fusion_report.md`.

---

## 9. CHIẾN LƯỢC KIỂM THỬ TỰ ĐỘNG (`tests/test_naval_fusion/`)

Bộ kiểm thử được viết hoàn toàn bằng `pytest` với 4 file test độc lập:
1. `test_coord_ned.py`: Đảm bảo bù tọa độ tàu mẹ chính xác tới từng milimet.
2. `test_ekf_tracker.py`: Đảm bảo các bước toán học Kalman tuyến tính và phi tuyến hội tụ, ma trận hiệp phương sai luôn đối xứng và dương xác định.
3. `test_association.py`: Đảm bảo ngưỡng Gating loại trừ điểm đo ngoại lai và ma trận chi phí Cosine phân tách mục tiêu ngược chiều.
4. `test_pipeline_integration.py`: Chạy end-to-end trên `scenario2`, khẳng định hệ thống phát hiện và bám bắt chính xác 2 con tàu với $RMSE_{pos} < 5.0\text{ m}$ và không phát sinh ngoại lệ.

---

## 10. KẾ HOẠCH NÂNG CẤP TƯƠNG LAI (FUTURE WORK)
- Bổ sung giao diện giả lập tín hiệu tác chiến điện tử (Simulated ESM/ELINT Bearing).
- Tích hợp mô hình phân loại tàu chiến Deep Learning từ tập `datasets/FGSC-23/` để gán nhãn chủng loại tàu chiến (Destroyer, Frigate, Combat Boat).
- Chuyển đổi bộ lọc EKF sang IMM-UKF khi cần bám bắt mục tiêu cơ động ngoặt lái gắt.
