# Phân Hệ Hợp Nhất Động Học Đa Cảm Biến Tàu Chiến (Naval Kinematic Fusion Engine) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng phân hệ phần mềm độc lập `src/naval_fusion/` thực thi bám bắt mục tiêu mặt nước phi hợp tác trên biển (không dùng AIS), bù chuyển động tàu mẹ trong hệ tọa độ Piren NED, hợp nhất dị thể điểm đo Radar/LiDAR 2D và Camera Hồng ngoại nhiệt IR / Quang học EO 1D bằng bộ lọc EKF, quản lý vết theo luật $M/N$ và đánh giá định lượng so với Ground Truth của bộ dữ liệu Autoferry.

**Architecture:** Kiến trúc 6 tầng ghép nối lỏng (Decoupled Plot-to-Track Architecture): Loader $\rightarrow$ Coordinate Transformer (NED) $\rightarrow$ Heterogeneous EKF Tracker $\rightarrow$ Mahalanobis/Cosine Association Engine $\rightarrow$ M/N Track Lifecycle Manager $\rightarrow$ Evaluator & CLI Runner.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`scipy.optimize.linear_sum_assignment`), Pytest.

**Spec:** [`docs/superpowers/specs/2026-10-18-naval-kinematic-fusion-design.md`](file:///mnt/d/Workspace/VTX/Fusion/docs/superpowers/specs/2026-10-18-naval-kinematic-fusion-design.md)

---

## Global Constraints

- Toàn bộ mã nguồn mới phải nằm độc lập bên trong `src/naval_fusion/`, bộ kiểm thử tại `tests/test_naval_fusion/`, điểm vào thực thi tại `run_naval_fusion.py`.
- Tuyệt đối không import hoặc sửa đổi mã nguồn cũ của bài báo Sensors 2025 (`src/ais_decoder.py`, `src/cascade_fusion.py`, `src/coord_transform.py`, `src/mstrack.py`, `src/config.py`, `main.py`).
- Không có bất kỳ phụ thuộc nào vào hệ thống nhận dạng tự động AIS; bám bắt hoàn toàn phi hợp tác.
- Hệ quy chuẩn không gian: Piren NED thế giới cố định tại $PirenLLA = [63.4389029^\circ N, 10.39908278^\circ E, 39.923m]$.
- Bộ lọc EKF phải hỗ trợ 2 chế độ cập nhật riêng biệt: Cập nhật 2D Cartesian tuyến tính (Radar/LiDAR) và Cập nhật 1D Góc phương vị phi tuyến (IR/EO).

## Review Focus

1. **Khởi tạo vết sai quy tắc bằng tia góc (Bearing-Only False Track):** Điểm đo góc 1D từ Camera Hồng ngoại hoặc EO tuyệt đối không được tự ý tạo vết mới nếu chưa có cự ly từ Radar/LiDAR $\rightarrow$ Kiểm thử tại Task 6.
2. **Góc phương vị bị đứt đoạn qua mốc biên $[-\pi, \pi]$ (Angle wrap-around):** Khi góc đo nhảy từ $+179^\circ$ sang $-179^\circ$, innovation góc $y = \text{wrap\_to\_pi}(\theta_{meas} - \theta_{pred})$ phải trả về $+2^\circ$ chứ không phải $-358^\circ$ $\rightarrow$ Kiểm thử tại Task 4.
3. **Ma trận hiệp phương sai mất tính đối xứng / dương xác định:** Cập nhật EKF qua hàng ngàn bước lặp có thể bị trôi số học dấu phẩy động $\rightarrow$ Phải ép đối xứng $P = \frac{1}{2}(P + P^T)$ sau mỗi bước cập nhật $\rightarrow$ Kiểm thử tại Task 4.
4. **Phân bổ sai nhầm khi hai tàu lướt qua nhau ngược chiều:** Khi khoảng cách 2 tàu rất gần, ma trận chi phí phải kích hoạt thành công hệ số phạt Cosine vận tốc để ngăn tráo ID $\rightarrow$ Kiểm thử tại Task 5.
5. **Độ trễ thời gian giữa các gói tin không đều ($\Delta t$ biến thiên):** Các cảm biến quét ở tần số khác nhau (Radar ~0.5 Hz, IR ~10 Hz), bộ lọc phải tính động $\Delta t = t_{curr} - t_{last}$ cho từng bước ngoại suy $\rightarrow$ Kiểm thử tại Task 4.

---

### Task 1: Thiết Lập Gói & Cấu Trúc Dữ Liệu Kiểu Mạnh (`models.py`, `__init__.py`)

**Files:**
- Create: `src/naval_fusion/__init__.py`
- Create: `src/naval_fusion/models.py`
- Test: `tests/test_naval_fusion/test_models.py`

**Interfaces:**
- Produces: `SensorType` (Enum: `LIDAR=1`, `RADAR=2`, `IR=3`, `EO=4`), `OwnshipState`, `Detection`, `TrackState`, `EvaluationMetrics`.

- [ ] **Step 1: Viết test kiểm tra tính toàn vẹn của mô hình dữ liệu**

```python
# tests/test_naval_fusion/test_models.py
import numpy as np
import pytest
from src.naval_fusion.models import SensorType, OwnshipState, Detection, TrackState

def test_sensor_type_values():
    assert SensorType.LIDAR == 1
    assert SensorType.RADAR == 2
    assert SensorType.IR == 3
    assert SensorType.EO == 4

def test_detection_creation():
    det = Detection(
        sensor_type=SensorType.RADAR,
        timestamp=100.0,
        ownship_pos=(10.0, 20.0),
        raw_measurement=[100.0, 50.0],
        measurement_ned=np.array([110.0, 70.0])
    )
    assert det.sensor_type == SensorType.RADAR
    assert det.measurement_ned.shape == (2,)

def test_track_state_defaults():
    track = TrackState(
        track_id=1,
        state_vector=np.array([100.0, 200.0, 2.0, 0.5]),
        covariance=np.eye(4),
        last_updated=100.0,
        hits=1,
        misses=0,
        is_confirmed=False,
        contributing_sensors={SensorType.RADAR}
    )
    assert track.track_id == 1
    assert not track.is_confirmed
```

- [ ] **Step 2: Chạy kiểm thử để xác nhận thất bại**

Run: `pytest tests/test_naval_fusion/test_models.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'src.naval_fusion')

- [ ] **Step 3: Triển khai `models.py` và `__init__.py`**

Tạo `src/naval_fusion/models.py` chứa đầy đủ các Dataclass theo đúng thiết kế của mục 3 trong Design Spec. Tạo `src/naval_fusion/__init__.py` export các lớp này.

- [ ] **Step 4: Chạy kiểm thử để xác nhận vượt qua**

Run: `pytest tests/test_naval_fusion/test_models.py -v`  
Expected: PASS

- [ ] **Step 5: Commit Git**

```bash
git add src/naval_fusion/__init__.py src/naval_fusion/models.py tests/test_naval_fusion/test_models.py
git commit -m "feat(naval_fusion): define strong-typed data contracts and models"
```

---

### Task 2: Chuyển Đổi Tọa Độ NED & Bù Vị Trí Tàu Mẹ (`coord_ned.py`)

**Files:**
- Create: `src/naval_fusion/coord_ned.py`
- Test: `tests/test_naval_fusion/test_coord_ned.py`

**Interfaces:**
- Consumes: `SensorType`, `OwnshipState` từ `models.py`.
- Produces: `compensate_ownship_2d(ownship_pos, vessel_meas) -> np.ndarray`, `normalize_bearing_rad(angle_rad) -> float`, `calc_bearing_from_ownship(ownship_pos, target_pos_ned) -> float`.

- [ ] **Step 1: Viết test kiểm tra chuyển đổi tọa độ và bù vị trí tàu mẹ**

```python
# tests/test_naval_fusion/test_coord_ned.py
import numpy as np
import pytest
from src.naval_fusion.coord_ned import (
    compensate_ownship_2d,
    normalize_bearing_rad,
    calc_bearing_from_ownship,
)

def test_compensate_ownship_2d():
    own_pos = (100.0, -200.0)
    meas_vessel = np.array([50.0, 30.0])
    target_ned = compensate_ownship_2d(own_pos, meas_vessel)
    np.testing.assert_allclose(target_ned, np.array([150.0, -170.0]))

def test_normalize_bearing_rad():
    assert np.isclose(normalize_bearing_rad(np.pi + 0.1), -np.pi + 0.1)
    assert np.isclose(normalize_bearing_rad(-np.pi - 0.1), np.pi - 0.1)
    assert np.isclose(normalize_bearing_rad(0.0), 0.0)

def test_calc_bearing_from_ownship():
    own_pos = (0.0, 0.0)
    # Target directly North -> 0 rad
    assert np.isclose(calc_bearing_from_ownship(own_pos, np.array([100.0, 0.0])), 0.0)
    # Target directly East -> pi/2 rad
    assert np.isclose(calc_bearing_from_ownship(own_pos, np.array([0.0, 100.0])), np.pi / 2)
```

- [ ] **Step 2: Chạy kiểm thử để xác nhận thất bại**

Run: `pytest tests/test_naval_fusion/test_coord_ned.py -v`  
Expected: FAIL (ModuleNotFoundError / ImportError)

- [ ] **Step 3: Triển khai `coord_ned.py`**

Thực hiện các phép toán bù vector tuyến tính $[N_{own}, E_{own}] + [x_v, y_v]$ và hàm chuẩn hóa góc $\text{atan2}(\sin \theta, \cos \theta)$.

- [ ] **Step 4: Chạy kiểm thử để xác nhận vượt qua**

Run: `pytest tests/test_naval_fusion/test_coord_ned.py -v`  
Expected: PASS

- [ ] **Step 5: Commit Git**

```bash
git add src/naval_fusion/coord_ned.py tests/test_naval_fusion/test_coord_ned.py
git commit -m "feat(naval_fusion): implement ownship motion compensation and NED geometry"
```

---

### Task 3: Bộ Nạp & Tiền Xử Lý Dữ Liệu Autoferry (`loader.py`)

**Files:**
- Create: `src/naval_fusion/loader.py`
- Test: `tests/test_naval_fusion/test_loader.py`

**Interfaces:**
- Consumes: `Detection`, `SensorType` từ `models.py`, `compensate_ownship_2d` từ `coord_ned.py`.
- Produces: `AutoferryLoader.load_scenario(scenario_dir) -> tuple[list[Detection], list[dict]]`.

- [ ] **Step 1: Viết test kiểm tra nạp dữ liệu từ thư mục `Autoferry-sensor_fusion_dataset`**

```python
# tests/test_naval_fusion/test_loader.py
import os
import pytest
from src.naval_fusion.loader import AutoferryLoader
from src.naval_fusion.models import SensorType

def test_load_scenario2_valid():
    scenario_dir = "datasets/Autoferry-sensor_fusion_dataset/scenario2"
    if not os.path.exists(scenario_dir):
        pytest.skip("Autoferry scenario2 not available")
    
    loader = AutoferryLoader()
    detections, ground_truth = loader.load_scenario(scenario_dir)
    
    assert len(detections) > 1000
    assert len(ground_truth) > 1000
    
    # Check that measurements are properly compensated
    radar_dets = [d for d in detections if d.sensor_type == SensorType.RADAR]
    assert len(radar_dets) > 0
    assert radar_dets[0].measurement_ned.shape == (2,)
```

- [ ] **Step 2: Chạy kiểm thử để xác nhận thất bại**

Run: `pytest tests/test_naval_fusion/test_loader.py -v`  
Expected: FAIL (No module named loader)

- [ ] **Step 3: Triển khai `loader.py`**

Đọc file `scenarioX_detections.json` và `scenarioX_groundTruth.json`. Phân tích trường `measurement` (nếu là list lồng nhau đối với Radar/LiDAR thì tách thành từng Detection riêng lẻ có bù vị trí tàu mẹ; nếu là float đối với IR/EO thì gán góc phương vị).

- [ ] **Step 4: Chạy kiểm thử để xác nhận vượt qua**

Run: `pytest tests/test_naval_fusion/test_loader.py -v`  
Expected: PASS

- [ ] **Step 5: Commit Git**

```bash
git add src/naval_fusion/loader.py tests/test_naval_fusion/test_loader.py
git commit -m "feat(naval_fusion): implement Autoferry JSON dataset loader and parser"
```

---

### Task 4: Bộ Lọc Kalman Mở Rộng Dị Thể (`ekf_tracker.py`)

**Files:**
- Create: `src/naval_fusion/ekf_tracker.py`
- Test: `tests/test_naval_fusion/test_ekf_tracker.py`

**Interfaces:**
- Consumes: `TrackState`, `Detection`, `SensorType` từ `models.py`, `normalize_bearing_rad` từ `coord_ned.py`.
- Produces: `HeterogeneousEKF.predict(track, dt) -> TrackState`, `HeterogeneousEKF.update_cartesian_2d(track, z_2d, R) -> TrackState`, `HeterogeneousEKF.update_bearing_1d(track, z_bearing, ownship_pos, R_angle) -> TrackState`.

- [ ] **Step 1: Viết test kiểm thử toán học EKF (Predict, Full Update 2D, Bearing Update 1D)**

```python
# tests/test_naval_fusion/test_ekf_tracker.py
import numpy as np
import pytest
from src.naval_fusion.ekf_tracker import HeterogeneousEKF
from src.naval_fusion.models import TrackState, SensorType

@pytest.fixture
def sample_track():
    return TrackState(
        track_id=1,
        state_vector=np.array([100.0, 200.0, 2.0, 1.0]),
        covariance=np.eye(4) * 10.0,
        last_updated=0.0,
        hits=1,
        misses=0,
        is_confirmed=True,
        contributing_sensors={SensorType.RADAR}
    )

def test_ekf_predict(sample_track):
    ekf = HeterogeneousEKF(q_noise=0.5)
    pred_track = ekf.predict(sample_track, dt=2.0)
    # Position = p0 + v * dt
    assert np.isclose(pred_track.state_vector[0], 104.0)
    assert np.isclose(pred_track.state_vector[1], 202.0)
    # Covariance must grow after prediction
    assert pred_track.covariance[0, 0] > sample_track.covariance[0, 0]

def test_ekf_update_cartesian(sample_track):
    ekf = HeterogeneousEKF()
    z = np.array([102.0, 198.0])
    updated_track = ekf.update_cartesian_2d(sample_track, z, R_cov=np.eye(2)*4.0)
    # Covariance must decrease after update
    assert updated_track.covariance[0, 0] < sample_track.covariance[0, 0]
    # Matrix must remain symmetric
    np.testing.assert_allclose(updated_track.covariance, updated_track.covariance.T)

def test_ekf_update_bearing_only(sample_track):
    ekf = HeterogeneousEKF()
    own_pos = (0.0, 0.0)
    # True bearing to (100, 200) is atan2(200, 100)
    true_angle = np.arctan2(200.0, 100.0)
    # Measurement slightly shifted by +0.02 rad
    z_angle = true_angle + 0.02
    updated_track = ekf.update_bearing_1d(sample_track, z_angle, own_pos, R_angle=1e-4)
    # Covariance must remain symmetric and positive definite
    np.testing.assert_allclose(updated_track.covariance, updated_track.covariance.T)
    eigenvalues = np.linalg.eigvals(updated_track.covariance)
    assert np.all(eigenvalues > 0)
```

- [ ] **Step 2: Chạy kiểm thử để xác nhận thất bại**

Run: `pytest tests/test_naval_fusion/test_ekf_tracker.py -v`  
Expected: FAIL (No module named ekf_tracker)

- [ ] **Step 3: Triển khai `ekf_tracker.py`**

Cài đặt đầy đủ công thức toán học EKF: Mô hình CV cho `predict`, cập nhật tuyến tính cho `update_cartesian_2d`, và Jacobian $H_j = [-\frac{dE}{d^2}, \frac{dN}{d^2}, 0, 0]$ cho `update_bearing_1d`. Ép đối xứng ma trận $P = 0.5(P + P^T)$.

- [ ] **Step 4: Chạy kiểm thử để xác nhận vượt qua**

Run: `pytest tests/test_naval_fusion/test_ekf_tracker.py -v`  
Expected: PASS

- [ ] **Step 5: Commit Git**

```bash
git add src/naval_fusion/ekf_tracker.py tests/test_naval_fusion/test_ekf_tracker.py
git commit -m "feat(naval_fusion): implement dual-mode heterogeneous EKF tracker"
```

---

### Task 5: Thuật Toán Ghép Nối Mahalanobis Gating & Phạt Cosine (`association.py`)

**Files:**
- Create: `src/naval_fusion/association.py`
- Test: `tests/test_naval_fusion/test_association.py`

**Interfaces:**
- Consumes: `TrackState`, `Detection`, `SensorType` từ `models.py`.
- Produces: `DataAssociator.associate(tracks, detections, ownship_pos) -> tuple[list[tuple[int, int]], list[int], list[int]]` (Matches, Unmatched Tracks, Unmatched Detections).

- [ ] **Step 1: Viết test kiểm tra ghép nối, cổng gating và hệ số phạt Cosine**

```python
# tests/test_naval_fusion/test_association.py
import numpy as np
import pytest
from src.naval_fusion.association import DataAssociator
from src.naval_fusion.models import TrackState, Detection, SensorType

def test_associate_cartesian_matches():
    associator = DataAssociator(chi2_gate_2d=9.21)
    track1 = TrackState(
        track_id=1,
        state_vector=np.array([100.0, 100.0, 2.0, 0.0]),
        covariance=np.eye(4)*5.0,
        last_updated=0.0, hits=3, misses=0, is_confirmed=True,
        contributing_sensors={SensorType.RADAR}
    )
    det1 = Detection(
        sensor_type=SensorType.RADAR,
        timestamp=0.1,
        ownship_pos=(0.0, 0.0),
        raw_measurement=[101.0, 102.0],
        measurement_ned=np.array([101.0, 102.0])
    )
    matches, unmatched_trks, unmatched_dets = associator.associate(
        tracks=[track1], detections=[det1], ownship_pos=(0.0, 0.0)
    )
    assert len(matches) == 1
    assert matches[0] == (0, 0)
    assert len(unmatched_trks) == 0
    assert len(unmatched_dets) == 0

def test_cosine_penalty_opposite_direction():
    associator = DataAssociator(cosine_weight=5.0)
    # Track going North (vN = 5.0, vE = 0.0)
    track = TrackState(
        track_id=1,
        state_vector=np.array([100.0, 100.0, 5.0, 0.0]),
        covariance=np.eye(4)*2.0,
        last_updated=0.0, hits=3, misses=0, is_confirmed=True,
        contributing_sensors={SensorType.RADAR}
    )
    # Measurement position identical, but estimated motion is South
    cost_same = associator.calc_cost_2d(track, np.array([100.0, 100.0]), meas_vel=np.array([5.0, 0.0]))
    cost_opp = associator.calc_cost_2d(track, np.array([100.0, 100.0]), meas_vel=np.array([-5.0, 0.0]))
    # Opposite direction must be penalized heavily
    assert cost_opp > cost_same + 8.0
```

- [ ] **Step 2: Chạy kiểm thử để xác nhận thất bại**

Run: `pytest tests/test_naval_fusion/test_association.py -v`  
Expected: FAIL (No module named association)

- [ ] **Step 3: Triển khai `association.py`**

Cài đặt hàm tính khoảng cách Mahalanobis, kiểm tra cổng $\chi^2$ (2D gate = 9.21, 1D gate = 9.0), hàm phạt Cosine vận tốc, và gọi `scipy.optimize.linear_sum_assignment` để phân công tối ưu 1-1.

- [ ] **Step 4: Chạy kiểm thử để xác nhận vượt qua**

Run: `pytest tests/test_naval_fusion/test_association.py -v`  
Expected: PASS

- [ ] **Step 5: Commit Git**

```bash
git add src/naval_fusion/association.py tests/test_naval_fusion/test_association.py
git commit -m "feat(naval_fusion): implement Mahalanobis gating and Cosine association"
```

---

### Task 6: Quản Lý Vòng Đời Vết Chuẩn Quân Sự (`track_manager.py`)

**Files:**
- Create: `src/naval_fusion/track_manager.py`
- Test: `tests/test_naval_fusion/test_track_manager.py`

**Interfaces:**
- Consumes: `TrackState`, `Detection`, `SensorType` từ `models.py`.
- Produces: `TrackManager.init_track(detection) -> TrackState`, `TrackManager.step(tracks, matches, detections, current_time) -> list[TrackState]`.

- [ ] **Step 1: Viết test kiểm tra quy tắc $M/N$ và cấm khởi tạo vết bằng tia góc**

```python
# tests/test_naval_fusion/test_track_manager.py
import numpy as np
import pytest
from src.naval_fusion.track_manager import TrackManager
from src.naval_fusion.models import Detection, SensorType

def test_ir_bearing_cannot_init_track():
    manager = TrackManager()
    ir_det = Detection(
        sensor_type=SensorType.IR,
        timestamp=1.0,
        ownship_pos=(0.0, 0.0),
        raw_measurement=[0.5],
        measurement_ned=np.array([0.5])
    )
    new_track = manager.init_track(ir_det)
    # Bearing-only detection must NOT initialize a new track
    assert new_track is None

def test_radar_inits_tentative_track():
    manager = TrackManager()
    rad_det = Detection(
        sensor_type=SensorType.RADAR,
        timestamp=1.0,
        ownship_pos=(0.0, 0.0),
        raw_measurement=[100.0, 200.0],
        measurement_ned=np.array([100.0, 200.0])
    )
    new_track = manager.init_track(rad_det)
    assert new_track is not None
    assert not new_track.is_confirmed  # Must be tentative
    assert new_track.hits == 1

def test_mn_rule_confirmation():
    manager = TrackManager(confirm_hits=3, window_size=5)
    rad_det = Detection(
        sensor_type=SensorType.RADAR,
        timestamp=1.0,
        ownship_pos=(0.0, 0.0),
        raw_measurement=[100.0, 200.0],
        measurement_ned=np.array([100.0, 200.0])
    )
    track = manager.init_track(rad_det)
    tracks = [track]
    
    # 2nd hit
    tracks = manager.update_matches(tracks, [(0, 0)], [rad_det], current_time=2.0)
    assert not tracks[0].is_confirmed
    # 3rd hit -> Confirmed!
    tracks = manager.update_matches(tracks, [(0, 0)], [rad_det], current_time=3.0)
    assert tracks[0].is_confirmed
```

- [ ] **Step 2: Chạy kiểm thử để xác nhận thất bại**

Run: `pytest tests/test_naval_fusion/test_track_manager.py -v`  
Expected: FAIL (No module named track_manager)

- [ ] **Step 3: Triển khai `track_manager.py`**

Cài đặt quy tắc $M/N$ (chuyển trạng thái từ Tentative $\rightarrow$ Confirmed khi đạt 3/5 hits), cơ chế xóa vết khi `misses >= max_misses` (mặc định 5), và điều kiện chặn không cho phép khởi tạo vết mới từ cảm biến góc IR/EO.

- [ ] **Step 4: Chạy kiểm thử để xác nhận vượt qua**

Run: `pytest tests/test_naval_fusion/test_track_manager.py -v`  
Expected: PASS

- [ ] **Step 5: Commit Git**

```bash
git add src/naval_fusion/track_manager.py tests/test_naval_fusion/test_track_manager.py
git commit -m "feat(naval_fusion): implement military-standard M/N track lifecycle manager"
```

---

### Task 7: Đánh Giá Định Lượng & So Sánh Đối Chứng (`evaluator.py`)

**Files:**
- Create: `src/naval_fusion/evaluator.py`
- Test: `tests/test_naval_fusion/test_evaluator.py`

**Interfaces:**
- Consumes: `TrackState`, `EvaluationMetrics` từ `models.py`.
- Produces: `NavalEvaluator.evaluate_scenario(confirmed_history, ground_truth) -> EvaluationMetrics`, `NavalEvaluator.generate_markdown_report(metrics_dict) -> str`.

- [ ] **Step 1: Viết test kiểm tra tính toán RMSE vị trí, vận tốc và tỷ lệ bám vết**

```python
# tests/test_naval_fusion/test_evaluator.py
import numpy as np
import pytest
from src.naval_fusion.evaluator import NavalEvaluator

def test_calc_rmse_position():
    evaluator = NavalEvaluator()
    # Estimated positions vs Ground truth positions
    est_pos = np.array([[10.0, 20.0], [15.0, 25.0]])
    gt_pos = np.array([[10.0, 23.0], [15.0, 21.0]])  # errors: 3m and 4m
    # RMSE = sqrt((3^2 + 4^2)/2) = sqrt(25/2) = 3.5355m
    rmse = evaluator.calc_rmse_pos(est_pos, gt_pos)
    assert np.isclose(rmse, 3.5355339)
```

- [ ] **Step 2: Chạy kiểm thử để xác nhận thất bại**

Run: `pytest tests/test_naval_fusion/test_evaluator.py -v`  
Expected: FAIL (No module named evaluator)

- [ ] **Step 3: Triển khai `evaluator.py`**

Cài đặt hàm tính RMSE vị trí, RMSE vận tốc, số lần nhảy ID (IDSw), tỷ lệ bám vết (Mostly Tracked - MT), và hàm xuất báo cáo định dạng bảng Markdown.

- [ ] **Step 4: Chạy kiểm thử để xác nhận vượt qua**

Run: `pytest tests/test_naval_fusion/test_evaluator.py -v`  
Expected: PASS

- [ ] **Step 5: Commit Git**

```bash
git add src/naval_fusion/evaluator.py tests/test_naval_fusion/test_evaluator.py
git commit -m "feat(naval_fusion): implement RMSE and MOTA quantitative evaluator"
```

---

### Task 8: Bộ Điều Phối Pipeline Tích Hợp & Điểm Vào CLI (`pipeline.py`, `run_naval_fusion.py`)

**Files:**
- Create: `src/naval_fusion/pipeline.py`
- Create: `run_naval_fusion.py`
- Test: `tests/test_naval_fusion/test_pipeline_integration.py`

**Interfaces:**
- Consumes: Mọi module trong `src/naval_fusion/`.
- Produces: `NavalFusionPipeline.run(scenario_dir, enabled_sensors) -> tuple[dict, EvaluationMetrics]`, CLI entrypoint `run_naval_fusion.py`.

- [ ] **Step 1: Viết test tích hợp toàn diện trên Autoferry `scenario2`**

```python
# tests/test_naval_fusion/test_pipeline_integration.py
import os
import pytest
from src.naval_fusion.pipeline import NavalFusionPipeline

def test_pipeline_run_scenario2():
    scenario_dir = "datasets/Autoferry-sensor_fusion_dataset/scenario2"
    if not os.path.exists(scenario_dir):
        pytest.skip("Autoferry scenario2 not available")
    
    pipeline = NavalFusionPipeline()
    history, metrics = pipeline.run(scenario_dir)
    
    # Must successfully track the 2 vessels
    assert metrics.num_confirmed_tracks >= 2
    # Position RMSE must be bounded (less than 10 meters on 500m scale)
    assert metrics.rmse_position < 10.0
```

- [ ] **Step 2: Chạy kiểm thử để xác nhận thất bại**

Run: `pytest tests/test_naval_fusion/test_pipeline_integration.py -v`  
Expected: FAIL (No module named pipeline)

- [ ] **Step 3: Triển khai `pipeline.py` và `run_naval_fusion.py`**

Kết nối toàn bộ chu trình xử lý: Nạp dữ liệu `loader` $\rightarrow$ duyệt từng mốc thời gian $\rightarrow$ dự báo `ekf` $\rightarrow$ ghép nối `association` $\rightarrow$ cập nhật đo lường $\rightarrow$ quản lý vết `track_manager` $\rightarrow$ xuất kết quả `evaluator`. Cài đặt giao diện dòng lệnh với `argparse` trong `run_naval_fusion.py`.

- [ ] **Step 4: Chạy kiểm thử tích hợp để xác nhận vượt qua**

Run: `pytest tests/test_naval_fusion/ -v`  
Expected: 100% PASS trên tất cả các unit tests và integration tests.

- [ ] **Step 5: Chạy thử nghiệm thực tế từ dòng lệnh trên `scenario2`**

Run: `python run_naval_fusion.py --scenario scenario2 --eval`  
Expected: Chạy mượt mà, in báo cáo bảng số liệu RMSE lên Terminal và lưu vào `output/naval_fusion_report.md`.

- [ ] **Step 6: Chạy kiểm tra đảm bảo mã nguồn cũ không bị ảnh hưởng**

Run: `python -m pytest tests/` (chạy toàn bộ test của repo)  
Expected: Toàn bộ test cũ của dự án vẫn pass 100%.

- [ ] **Step 7: Commit Git hoàn tất kế hoạch**

```bash
git add src/naval_fusion/pipeline.py run_naval_fusion.py tests/test_naval_fusion/
git commit -m "feat(naval_fusion): complete integrated pipeline and CLI runner"
```
