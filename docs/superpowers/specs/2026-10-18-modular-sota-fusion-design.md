# Đặc Tả Thiết Kế Kỹ Thuật (Technical Design Spec): SOTA Modular Maritime Fusion Engine

**Dự án:** Multi-Source Naval Intelligence, Surveillance & Reconnaissance (ISR) Data Fusion  
**Ngày thiết kế:** 2026-10-18  
**Tác giả:** Antigravity AI & Nhóm Kỹ sư Dự án  
**Trạng thái:** Chờ phê duyệt (Pending User Review)  
**Tập dữ liệu kiểm chuẩn:** NTNU Autonomous Ferry Littoral Sensor Fusion Benchmark (9 kịch bản biển)  

---

## 1. Giới Thiệu & Mục Tiêu Kỹ Thuật

### 1.1. Bối cảnh
Qua kiểm chuẩn độc lập trên toàn bộ 9 kịch bản (17,771 sự kiện đo đạc thực tế từ Radar Simrad 4G, LiDAR Velodyne Puck, Camera quang học Basler và Camera nhiệt FLIR) của bộ dữ liệu Autoferry, hệ thống nhận diện 3 điểm nghẽn vật lý lớn trong môi trường biển ven bờ:
1. **Bóng râm cảm biến (Occlusion Shadow):** Trong Kịch bản 3, 5, 6, tàu `Havfruen` bị che khuất sau tàu lớn `Gunnerus` trong 15-20 giây. Bộ lọc EKF kinh điển bị nhầm lẫn góc camera, gây tích tụ sai số vận tốc giả và kéo vệt dự báo văng xa (RMSE vọt lên $> 200\text{m}$).
2. **Chập vệt phản xạ Radar (Merged Rayleigh Hits):** Trong Kịch bản 17, hai tàu chạy song song cách nhau $10-13\text{m}$ ở cự ly $> 700\text{m}$. Búp sóng Radar rộng $63.5\text{m}$ không thể phân giải 2 mục tiêu riêng biệt, chỉ trả về 1 đốm đo duy nhất. Phép gán đơn định Hungary gán điểm đo cho một tàu, khiến tàu còn lại bị "bỏ đói" dữ liệu vị trí.
3. **Độ lệch tâm thân vỏ mục tiêu mở rộng (Extended Object Glint):** Tàu `Gunnerus` dài tới $31.25\text{m}$ và rộng $9.6\text{m}$. Sóng Radar/LiDAR tán xạ từ mép mạn hoặc mũi tàu tạo ra sai số tự nhiên $5-10\text{m}$ so với ăng-ten GPS cột buồm.

### 1.2. Mục tiêu hệ thống thế hệ mới
Thiết kế và triển khai một kiến trúc hợp nhất đa cảm biến 3 tầng độc lập (**SOTA Modular Maritime Fusion Engine**), hoàn toàn tất định (deterministic), thực thi thuần túy trên **CPU** với độ trễ $< 0.4\text{ ms / sự kiện}$, khắc phục triệt để cả 3 điểm nghẽn vật lý trên.

---

## 2. Kiến Trúc Hệ Thống (System Architecture)

Hệ thống được chia tách thành 3 tầng chức năng rõ rệt theo sơ đồ khối dưới đây:

```
                          [ Dữ liệu Cảm biến Dị loại Đầu vào ]
                     (Radar 2.5Hz, LiDAR 10Hz, Camera EO/IR 20Hz)
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TẦNG 2: BỘ QUẢN LÝ LIÊN KẾT, BÓNG RÂM & PHÂN GIẢI CHẬP VỆT (ASSOCIATION)   │
│  - Phép thử cổng Mahalanobis mở rộng (Adaptive Gating)                      │
│  - Bộ phát hiện góc khuất Ray-Casting (LoS Occlusion Detection)             │
│  - Cơ chế nhận diện & chia sẻ đốm đo chập (Merged Rayleigh Resolution)     │
│  - Phân bổ tối ưu ma trận chi phí (Kuhn-Munkres Hungarian)                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TẦNG 1: BỘ LỌC PHI TUYẾN TẤT ĐỊNH TÁCH BIỆT VẬN TỐC (DECOUPLED UKF CORE)   │
│  - Lấy mẫu 9 điểm Sigma (Unscented Transform không cần vi phân Jacobian)   │
│  - Cập nhật vị trí 2D (Radar / LiDAR)                                       │
│  - Cập nhật góc 1D (EO / IR) với độ lợi vận tốc triệt tiêu (K_vel = 0)     │
│  - Dự báo quán tính thủy động học (Dead-Reckoning Coasting) khi mất dấu     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TẦNG 3: BÙ LỆCH TÂM THÂN VỎ (EOT) & LÀM MỊN QUỸ ĐẠO THỦY ĐỘNG LỰC HỌC     │
│  - Mô hình elip định hướng thân tàu (Extended Object Ellipse Offset)        │
│  - Chiếu tâm phản xạ mép vỏ về ăng-ten GPS cột buồm (Mast Antenna)          │
│  - Bộ lọc làm mịn động học bậc cao Savitzky-Golay / RTS Smoother            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
                     [ Dữ liệu Đầu ra Chuẩn Hóa WGS-84 ]
           (Lat, Lon, SOG knots, COG deg, Video Playback, Dashboard)
```

---

## 3. Đặc Tả Chi Tiết Từng Module (Component Specifications)

### 3.1. Tầng 1: Decoupled Unscented Kalman Filter (`decoupled_ukf.py`)

#### Vector trạng thái & Dự báo động học
* Vector trạng thái: $x_k = [p_N, v_N, p_E, v_E]^T \in \mathbb{R}^4$ (Vị trí Bắc, Đông theo mét; Vận tốc theo m/s).
* Ma trận chuyển trạng thái chuyển động vận tốc không đổi (CV Model):
  $$F = \begin{bmatrix} 1 & \Delta t & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & \Delta t \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad Q_k = \begin{bmatrix} Q_1 & 0 \\ 0 & Q_1 \end{bmatrix}, \quad Q_1 = q \begin{bmatrix} \frac{\Delta t^3}{3} & \frac{\Delta t^2}{2} \\ \frac{\Delta t^2}{2} & \Delta t \end{bmatrix}$$
  Với $q = 0.04\text{ m}^2/\text{s}^3$ cho tàu nhỏ linh hoạt, $q = 0.02\text{ m}^2/\text{s}^3$ cho tàu lớn có quán tính cao.
* Khởi tạo vector vận tốc ban đầu thông qua phép vi phân 2 mốc thời gian đầu tiên ($\Delta t > 0.4\text{s}$) thay vì giả định vận tốc bằng $0$.

#### Lấy mẫu điểm Sigma (Unscented Transform)
* Số điểm Sigma: $2n + 1 = 9$ điểm với $n = 4$:
  $$\mathcal{X}_0 = \hat{x}, \quad \mathcal{X}_i = \hat{x} + \left(\sqrt{(n + \lambda)P}\right)_i, \quad \mathcal{X}_{i+n} = \hat{x} - \left(\sqrt{(n + \lambda)P}\right)_i \quad (i = 1, \dots, n)$$
  Tham số tỉ lệ: $\alpha = 10^{-3}, \beta = 2, \kappa = 0, \lambda = \alpha^2(n + \kappa) - n$.

#### Cập nhật đo đạc góc Camera & Tách biệt vận tốc (Velocity Decoupling)
* Hàm đo góc phi tuyến: $h_{\text{brg}}(x) = \arctan2(p_E - E_{\text{own}}, p_N - N_{\text{own}})$.
* Sai lệch đo lường: $\tilde{y} = (z_{\text{brg}} - \hat{z}_{\text{brg}} + \pi) \pmod{2\pi} - \pi$.
* Cổng kiểm tra tính hợp lệ của góc: $|\tilde{y}| < 5.0^\circ$.
* Vector độ lợi Kalman tiêu chuẩn $K = P_{xz} S_{zz}^{-1} = [K_{p_N}, K_{v_N}, K_{p_E}, K_{v_E}]^T$.
* **Quy tắc tách biệt vận tốc Aidala-Hammel:**
  $$K_{\text{decoupled}} = \begin{bmatrix} K_{p_N} \\ 0.0 \\ K_{p_E} \\ 0.0 \end{bmatrix}$$
  * Đảm bảo góc camera chỉ tác động lên vị trí vuông góc tia ngắm (Cross-LOS).
  * Vận tốc $v_N, v_E$ được bảo toàn nguyên vẹn, không bị ô nhiễm bởi nhiễu góc.

#### Ràng buộc vật lý tốc độ tối đa (Velocity Clamping)
* $v_{\text{mag}} = \sqrt{v_N^2 + v_E^2}$. Nếu $v_{\text{mag}} > 12.0\text{ m/s}$ ($23.3\text{ knots}$), tự động co tỉ lệ $v \leftarrow v \cdot \frac{12.0}{v_{\text{mag}}}$.

---

### 3.2. Tầng 2: Quản Lý Bóng Râm & Phân Giải Chập Vệt (`shadow_manager.py`)

#### Bắn tia LoS Ray-Casting phát hiện che khuất
* Góc nhìn tàu mẹ tới Tàu 1 và Tàu 2: $\theta_1, \theta_2$.
* Góc kẹp: $\Delta \theta = |(\theta_1 - \theta_2 + \pi) \pmod{2\pi} - \pi|$.
* Cự ly tới tàu mẹ: $R_1 = \|p_1 - P_{\text{own}}\|, R_2 = \|p_2 - P_{\text{own}}\|$.
* **Quy tắc bóng râm:**
  - Nếu $\Delta \theta < 4.5^\circ$: Tàu có cự ly xa hơn $\max(R_1, R_2)$ được phân loại là **Bị Che Khuất (Occluded)**.
  - Tàu bị che khuất sẽ **từ chối toàn bộ cập nhật góc camera** nằm trong phạm vi $|\theta_{\text{meas}} - \theta_{\text{front}}| < 5.0^\circ$.
  - Tàu bị che khuất chuyển sang chế độ **Dự báo quán tính (Coasting)**. Cổng bám 2D mở rộng thích ứng theo thời gian mất dấu:
    $$\text{Gate}(t) = \min(30.0 + 3.0 \cdot \Delta t_{\text{coast}}, 80.0\text{ m})$$
  - Khi tàu trước bẻ lái hoặc tàu sau lộ diện ($\Delta \theta > 5.0^\circ$), cổng bám lập tức thu nhận lại đo đạc Radar/LiDAR mà không làm đứt gãy ID theo dõi.

#### Phân giải chập vệt phản xạ Radar (Rayleigh Mixture Model)
* Điều kiện phát hiện chập vệt:
  - Cự ly giữa 2 tàu $d_{12} = \|p_1 - p_2\| < 25.0\text{m}$.
  - Số lượng điểm phản xạ Radar trong khu vực chỉ có duy nhất 1 điểm $z_{\text{merged}}$.
* Trọng số tiệm cận gán mềm:
  $$d_1 = \|z_{\text{merged}} - p_1\|, \quad d_2 = \|z_{\text{merged}} - p_2\|$$
  $$w_1 = \frac{\exp\left(-\frac{d_1^2}{2\sigma_r^2}\right)}{\exp\left(-\frac{d_1^2}{2\sigma_r^2}\right) + \exp\left(-\frac{d_2^2}{2\sigma_r^2}\right)}, \quad w_2 = 1 - w_1$$
* Hiệp phương sai mở rộng: $R_{\text{merged}} = R_{\text{radar}} + \frac{1}{4}(p_1 - p_2)(p_1 - p_2)^T$.
* Cập nhật đồng thời cho cả hai tàu: $\Delta x_1 = w_1 K_1 (z - \hat{z}_1)$ và $\Delta x_2 = w_2 K_2 (z - \hat{z}_2)$.

---

### 3.3. Tầng 3: Bù Lệch Tâm Thân Tàu Mở Rộng & Làm Mịn (`eot_compensator.py`)

#### Bù lệch tâm thân vỏ elip (Extended Object Tracking Offset)
* Hướng mũi tàu: $\psi = \arctan2(v_E, v_N)$.
* Kích thước định danh:
  - Tàu lớn `R/V Gunnerus`: Chiều dài $L = 31.25\text{m}$, Chiều rộng $W = 9.6\text{m}$.
  - Tàu nhỏ `Havfruen` / `Jet boat`: Xem như chất điểm ($L = 7.0\text{m}, W = 2.5\text{m}$).
* Vector tia ngắm từ tàu mẹ tới mục tiêu: $u_{\text{los}} = \frac{p - P_{\text{own}}}{\|p - P_{\text{own}}\|}$.
* Vector pháp tuyến mép vỏ đối diện tia ngắm trong hệ quy chiếu thân tàu:
  $$\delta_{\text{body}} = \begin{bmatrix} -\text{sign}(u_{\text{los}} \cdot \mathbf{x}_{\text{body}}) \cdot \frac{L}{4} \\ -\text{sign}(u_{\text{los}} \cdot \mathbf{y}_{\text{body}}) \cdot \frac{W}{4} \end{bmatrix}$$
* Vector dịch chuyển bù trong hệ NED: $\Delta p_{\text{offset}} = R(\psi) \delta_{\text{body}}$.
* Điểm đo sau hiệu chỉnh đưa vào bộ lọc: $z_{\text{corrected}} = z_{\text{raw}} - \Delta p_{\text{offset}}$.

#### Bộ làm mịn Savitzky-Golay & Trắc địa WGS-84
* Bộ lọc đa thức bậc 2 với cửa sổ thích ứng: $W = 51$ nếu số mẫu $> 55$, ngược lại $W = 21$.
* Tính toán chuẩn các trường hải hành:
  - $\text{Lat}(t) = 63.4389029 + \frac{p_N(t)}{111319.5}$ (độ thập phân).
  - $\text{Lon}(t) = 10.3990828 + \frac{p_E(t)}{111319.5 \cdot \cos(63.4389029^\circ)}$ (độ thập phân).
  - $\text{SOG}(t) = \sqrt{\dot{p}_N^2 + \dot{p}_E^2} \times 1.94384$ (hải lý/giờ).
  - $\text{COG}(t) = (\arctan2(\dot{p}_E, \dot{p}_N) \times \frac{180}{\pi} + 360) \pmod{360}$ (độ).

---

## 4. Cấu Trúc Mã Nguồn & Tệp Tin (File Organization)

Toàn bộ mã nguồn mới được tổ chức gọn gàng trong thư mục [`naval_isr/src/tracking/`](file:///mnt/d/Workspace/VTX/Fusion/naval_isr):

```
naval_isr/
├── src/
│   └── tracking/
│       ├── __init__.py
│       ├── decoupled_ukf.py           # Bộ lọc phi tuyến Decoupled UKF
│       ├── shadow_manager.py          # Quản lý bóng râm LoS & Phân giải chập vệt
│       ├── eot_compensator.py         # Bù lệch tâm thân vỏ tàu mở rộng
│       └── modular_fusion_pipeline.py # Điều phối toàn bộ pipeline
├── tests/
│   └── test_modular_tracker.py        # Kiểm thử đơn vị tự động
└── scripts/
    └── evaluate_all_autoferry_scenarios.py # Benchmark tự động 9 kịch bản
```

---

## 5. Kế Hoạch Kiểm Thử & Tiêu Chí Thành Công (Testing & KPIs)

### 5.1. Kiểm thử đơn vị (Unit Tests)
1. `test_decoupled_ukf_gain`: Xác nhận độ lợi $K[1] == 0.0$ và $K[3] == 0.0$ khi cập nhật góc, vận tốc không đổi.
2. `test_shadow_manager_suppression`: Xác nhận khi $\Delta \theta = 2.0^\circ$ và $R_2 > R_1$, Tàu 2 bị khóa góc camera và chuyển sang Coasting.
3. `test_merged_radar_resolution`: Xác nhận khi khoảng cách 2 tàu $= 12\text{m}$ và có 1 điểm đo radar, cả hai tàu đều nhận cập nhật vị trí tương ứng trọng số tiệm cận.
4. `test_eot_projection`: Xác nhận điểm đo mép mạn tàu lớn $31\text{m}$ được dịch chuyển chính xác về tâm ăng-ten cột buồm.

### 5.2. Tiêu chí định lượng thành công trên 9 kịch bản Autoferry
1. **Triệt tiêu hiện tượng văng vệt:** `scenario3` và `scenario6` không bị mất dấu hay vọt sai số $> 400\text{m}$.
2. **Khắc phục chập vệt Kịch bản 17:** Sai số hợp nhất giảm từ $65.07\text{m}$ xuống dưới $15.0\text{m}$.
3. **Giảm sai số tàu lớn `Gunnerus`:** Sai số RMSE tàu 2 trong các kịch bản thuận lợi giảm từ $\approx 10\text{m}$ xuống dưới $5.5\text{m}$.
4. **Hiệu năng CPU:** Toàn bộ 17,771 sự kiện chạy xong trong $< 5.0\text{ giây}$ trên CPU ($< 0.35\text{ ms / sự kiện}$).
