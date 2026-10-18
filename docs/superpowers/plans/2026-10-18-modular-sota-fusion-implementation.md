# SOTA Modular Maritime Fusion Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and benchmark a deterministic 3-tier multi-sensor maritime tracking and fusion pipeline in `naval_isr/` that eliminates occlusion divergence and merged-radar track starvation across all 9 Autoferry scenarios on CPU.

**Architecture:** A 3-tier modular pipeline where Tier 1 provides an Unscented Kalman Filter with Aidala-Hammel velocity decoupling ($K_{vel} = 0$) for bearing updates; Tier 2 manages line-of-sight ray-casting occlusion shadows and soft-weighted merged Rayleigh radar returns; and Tier 3 applies dynamic mast antenna offset for extended hulls ($31.25\text{m} \times 9.6\text{m}$) followed by hydrodynamic kinematic smoothing.

**Tech Stack:** Python 3.10+, NumPy, SciPy (optimize, signal, linalg), WGS-84 Geodetic formulas.

**Spec:** [`docs/superpowers/specs/2026-10-18-modular-sota-fusion-design.md`](file:///mnt/d/Workspace/VTX/Fusion/docs/superpowers/specs/2026-10-18-modular-sota-fusion-design.md)

## Global Constraints

- Execution must be 100% CPU-based with zero external GPU/neural network dependencies.
- Average processing latency must remain under $0.4\text{ ms / event}$ ($< 5.0\text{ seconds}$ total for 17,771 events).
- All new source code must be strictly isolated inside [`naval_isr/`](file:///mnt/d/Workspace/VTX/Fusion/naval_isr).
- Maintain geodetic compatibility with Piren NED origin $(63.4389029^\circ\text{N}, 10.3990828^\circ\text{E})$.
- Every component must be accompanied by unit tests with strict tolerance assertions.

## Review Focus

- Numerical stability during zero-range singularity: when range $r < 1.0\text{m}$, bearing update must abort without divide-by-zero.
- Angle wrap-around arithmetic: innovation angle $\tilde{y}$ must strictly wrap within $[-\pi, \pi]$ using $(y + \pi) \pmod{2\pi} - \pi$.
- Empty measurement payloads: radar/lidar events with $0$ points must gracefully trigger prediction without crashing.
- Target re-acquisition after prolonged occlusion ($> 15\text{s}$): search gate must expand linearly and cap at $80\text{m}$ to prevent false associations with clutter.
- Single-point merged hit handling: when radar provides 1 point for 2 close targets, neither target covariance may become singular or non-positive-definite.

---

### Task 1: Decoupled Unscented Kalman Filter Core

**Files:**
- Create: `naval_isr/src/tracking/decoupled_ukf.py`
- Test: `naval_isr/tests/test_decoupled_ukf.py`

**Interfaces:**
- Consumes: None (base mathematical module).
- Produces: `DecoupledUKF(init_pos: np.ndarray, is_large_ship: bool = False, q: float = 0.04, r_pos: float = 16.0)` with methods:
  - `predict(dt: float) -> None`
  - `update_pos(z_pos: np.ndarray) -> None`
  - `update_bearing(z_brg: float, own_pos: np.ndarray) -> bool`
  - Properties: `position -> np.ndarray (2,)`, `velocity -> np.ndarray (2,)`, `speed -> float`

- [ ] **Step 1: Write the failing test for DecoupledUKF**

```python
# naval_isr/tests/test_decoupled_ukf.py
import numpy as np
import pytest
from naval_isr.src.tracking.decoupled_ukf import DecoupledUKF

def test_ukf_prediction_and_velocity_decoupling():
    init_pos = np.array([100.0, 50.0])
    tracker = DecoupledUKF(init_pos=init_pos, is_large_ship=False)
    tracker.x[1] = 3.0  # v_N
    tracker.x[3] = 4.0  # v_E
    
    # 1. Predict 1.0 second
    tracker.predict(1.0)
    assert np.isclose(tracker.x[0], 103.0)
    assert np.isclose(tracker.x[2], 54.0)
    
    # 2. Bearing update must decouple velocity: velocity must remain strictly [3.0, 4.0]
    own_pos = np.array([0.0, 0.0])
    true_brg = np.arctan2(54.0, 103.0)
    noisy_brg = true_brg + np.radians(2.0)  # 2 degrees error
    
    updated = tracker.update_bearing(noisy_brg, own_pos)
    assert updated is True
    assert np.isclose(tracker.x[1], 3.0, atol=1e-6)
    assert np.isclose(tracker.x[3], 4.0, atol=1e-6)

def test_ukf_velocity_clamping():
    tracker = DecoupledUKF(init_pos=np.array([0.0, 0.0]))
    tracker.x[1] = 20.0
    tracker.x[3] = 20.0
    tracker.predict(0.1)
    assert tracker.speed <= 12.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest naval_isr/tests/test_decoupled_ukf.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'naval_isr.src.tracking.decoupled_ukf'`

- [ ] **Step 3: Implement `DecoupledUKF` in `naval_isr/src/tracking/decoupled_ukf.py`**

Implement Unscented Transform with 9 sigma points ($\alpha=10^{-3}, \beta=2, \kappa=0$). In `update_bearing`, compute cross-covariance $P_{xz}$ and innovation $S$, then set Kalman gain velocity rows $K[1]=0.0, K[3]=0.0$ before updating state and covariance. Clamp velocity magnitude to $12.0\text{ m/s}$.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest naval_isr/tests/test_decoupled_ukf.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add naval_isr/src/tracking/decoupled_ukf.py naval_isr/tests/test_decoupled_ukf.py
git commit -m "feat(tracking): implement decoupled unscented kalman filter core"
```

---

### Task 2: Sensor Shadow Ray-Casting & Merged Rayleigh Resolution

**Files:**
- Create: `naval_isr/src/tracking/shadow_manager.py`
- Test: `naval_isr/tests/test_shadow_manager.py`

**Interfaces:**
- Consumes: None (geometry and association manager).
- Produces: `ShadowAndRayleighManager` with methods:
  - `check_occlusion(own_pos: np.ndarray, pos1: np.ndarray, pos2: np.ndarray) -> Tuple[bool, int]` (returns is_occluded, occluded_target_id: 1 or 2)
  - `is_bearing_blocked_by_shadow(z_brg: float, own_pos: np.ndarray, front_pos: np.ndarray) -> bool`
  - `resolve_merged_radar(pts: List[np.ndarray], pos1: np.ndarray, pos2: np.ndarray, gate: float = 25.0) -> Tuple[bool, Optional[Tuple[float, float]]]` (returns is_merged, weights (w1, w2))

- [ ] **Step 1: Write the failing test for ShadowAndRayleighManager**

```python
# naval_isr/tests/test_shadow_manager.py
import numpy as np
import pytest
from naval_isr.src.tracking.shadow_manager import ShadowAndRayleighManager

def test_ray_casting_occlusion_detection():
    mgr = ShadowAndRayleighManager(shadow_angle_deg=4.5)
    own = np.array([0.0, 0.0])
    # T1 is front at 100m bearing 0.5 rad; T2 is behind at 200m bearing 0.52 rad (~1.1 deg diff)
    pos1 = np.array([100.0 * np.cos(0.5), 100.0 * np.sin(0.5)])
    pos2 = np.array([200.0 * np.cos(0.52), 200.0 * np.sin(0.52)])
    
    is_occ, occ_id = mgr.check_occlusion(own, pos1, pos2)
    assert is_occ is True
    assert occ_id == 2  # Target 2 is in the shadow
    
    # Bearing to front ship should block T2 bearing update
    assert mgr.is_bearing_blocked_by_shadow(0.51, own, pos1) is True

def test_merged_radar_resolution():
    mgr = ShadowAndRayleighManager()
    pos1 = np.array([500.0, 200.0])
    pos2 = np.array([508.0, 205.0])  # 9.4m separation (unresolved)
    
    pts = [np.array([504.0, 202.5])]  # 1 merged centroid hit
    is_merged, weights = mgr.resolve_merged_radar(pts, pos1, pos2, gate=25.0)
    assert is_merged is True
    assert weights is not None
    w1, w2 = weights
    assert np.isclose(w1 + w2, 1.0)
    assert 0.3 < w1 < 0.7  # balanced weighting
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest naval_isr/tests/test_shadow_manager.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'naval_isr.src.tracking.shadow_manager'`

- [ ] **Step 3: Implement `ShadowAndRayleighManager` in `naval_isr/src/tracking/shadow_manager.py`**

Implement angular separation check with wrap-around $\Delta \theta = |(\theta_1 - \theta_2 + \pi) \pmod{2\pi} - \pi|$. If $\Delta \theta < 4.5^\circ$, mark the farther target as occluded. For single radar hit within $25\text{m}$ mutual distance, calculate Gaussian proximity weights $w_1, w_2$ normalized to sum to $1.0$.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest naval_isr/tests/test_shadow_manager.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add naval_isr/src/tracking/shadow_manager.py naval_isr/tests/test_shadow_manager.py
git commit -m "feat(tracking): implement shadow ray-casting and merged radar manager"
```

---

### Task 3: Extended Object Tracking (EOT) Offset & Hydrodynamic Smoother

**Files:**
- Create: `naval_isr/src/tracking/eot_compensator.py`
- Test: `naval_isr/tests/test_eot_compensator.py`

**Interfaces:**
- Consumes: None (geometry transformation).
- Produces: `EOTCompensator` with methods:
  - `compute_antenna_offset(own_pos: np.ndarray, target_pos: np.ndarray, target_vel: np.ndarray, length: float = 31.25, beam: float = 9.6) -> np.ndarray` (returns 2D offset vector in NED)
  - `smooth_trajectory(trajectory: np.ndarray, window_len: int = 51, polyorder: int = 2) -> np.ndarray`
  - `to_wgs84_telemetry(pos_ned: np.ndarray, vel_ned: np.ndarray) -> Tuple[float, float, float, float]` (returns Lat, Lon, SOG knots, COG deg)

- [ ] **Step 1: Write the failing test for EOTCompensator**

```python
# naval_isr/tests/test_eot_compensator.py
import numpy as np
import pytest
from naval_isr.src.tracking.eot_compensator import EOTCompensator

def test_eot_antenna_offset_projection():
    comp = EOTCompensator()
    own_pos = np.array([0.0, 0.0])
    target_pos = np.array([100.0, 0.0])  # due North
    target_vel = np.array([5.0, 0.0])    # heading North (psi = 0)
    
    # Length = 31.25m, Beam = 9.6m
    offset = comp.compute_antenna_offset(own_pos, target_pos, target_vel, length=31.25, beam=9.6)
    # The sensor illuminates the stern/facing hull, offset should shift towards center
    assert len(offset) == 2
    assert abs(offset[0]) > 2.0  # non-trivial offset along longitudinal axis

def test_telemetry_conversion():
    comp = EOTCompensator()
    lat, lon, sog, cog = comp.to_wgs84_telemetry(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
    assert np.isclose(lat, 63.4389029, atol=1e-5)
    assert np.isclose(lon, 10.3990828, atol=1e-5)
    assert np.isclose(sog, 5.0 * 1.94384, atol=0.01)  # 5 m/s in knots
    assert np.isclose(cog, np.degrees(np.arctan2(4.0, 3.0)), atol=0.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest naval_isr/tests/test_eot_compensator.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'naval_isr.src.tracking.eot_compensator'`

- [ ] **Step 3: Implement `EOTCompensator` in `naval_isr/src/tracking/eot_compensator.py`**

Implement rotation matrix $R(\psi)$ from heading $\psi = \arctan2(v_E, v_N)$, project sensor line-of-sight into body frame, compute hull edge offset, rotate to NED. Implement Savitzky-Golay smoothing and geodetic WGS-84 conversion with Piren reference $(63.4389029, 10.3990828)$.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest naval_isr/tests/test_eot_compensator.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add naval_isr/src/tracking/eot_compensator.py naval_isr/tests/test_eot_compensator.py
git commit -m "feat(tracking): implement eot offset compensation and hydrodynamic smoother"
```

---

### Task 4: Modular Maritime Fusion Pipeline Integration

**Files:**
- Create: `naval_isr/src/tracking/modular_fusion_pipeline.py`
- Create: `naval_isr/src/tracking/__init__.py`
- Test: `naval_isr/tests/test_modular_fusion_pipeline.py`

**Interfaces:**
- Consumes: `DecoupledUKF` (Task 1), `ShadowAndRayleighManager` (Task 2), `EOTCompensator` (Task 3).
- Produces: `ModularMaritimeFusionPipeline(init_t1: np.ndarray, init_t2: np.ndarray, is_t2_large: bool = True)` with methods:
  - `initialize_velocities(v1: np.ndarray, v2: np.ndarray) -> None`
  - `process_event(detection: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]`
  - `finalize_trajectories() -> Tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: Write the failing integration test**

```python
# naval_isr/tests/test_modular_fusion_pipeline.py
import numpy as np
import pytest
from naval_isr.src.tracking.modular_fusion_pipeline import ModularMaritimeFusionPipeline

def test_pipeline_event_processing():
    p1_0 = np.array([100.0, 50.0])
    p2_0 = np.array([-100.0, -50.0])
    pipeline = ModularMaritimeFusionPipeline(p1_0, p2_0, is_t2_large=True)
    pipeline.initialize_velocities(np.array([2.0, 1.0]), np.array([-1.0, 0.5]))
    
    # Send a radar event (sensorID=2)
    event_radar = {
        "time": 1000.1,
        "sensorID": 2,
        "ownshipPosition": [0.0, 0.0],
        "measurement": [[100.2, -99.8], [50.1, -49.9]]
    }
    t1_pos, t2_pos = pipeline.process_event(event_radar)
    assert t1_pos.shape == (2,)
    assert t2_pos.shape == (2,)
    assert np.linalg.norm(t1_pos - p1_0) < 5.0
    assert np.linalg.norm(t2_pos - p2_0) < 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest naval_isr/tests/test_modular_fusion_pipeline.py -v`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `ModularMaritimeFusionPipeline` in `naval_isr/src/tracking/modular_fusion_pipeline.py`**

Orchestrate the 3 tiers:
1. Predict UKF trackers with $\Delta t$.
2. Check occlusion using `ShadowAndRayleighManager`.
3. If radar/lidar: apply EOT offset for large ship, check merged hit resolution, update via Hungarian bipartite matching.
4. If optical/IR bearing: if target is occluded, suppress update; otherwise perform decoupled UKF bearing update.
5. Record state history and apply `EOTCompensator.smooth_trajectory` in `finalize_trajectories`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest naval_isr/tests/test_modular_fusion_pipeline.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add naval_isr/src/tracking/modular_fusion_pipeline.py naval_isr/src/tracking/__init__.py naval_isr/tests/test_modular_fusion_pipeline.py
git commit -m "feat(tracking): integrate modular sota maritime fusion pipeline"
```

---

### Task 5: 9-Scenario Autoferry Benchmark Regression & Output Verification

**Files:**
- Modify: `naval_isr/scripts/evaluate_all_autoferry_scenarios.py`
- Output: `naval_isr/output/all_scenarios_optimization_benchmark.md`

**Interfaces:**
- Consumes: `ModularMaritimeFusionPipeline` (Task 4) across all 9 scenarios in `datasets/Autoferry-sensor_fusion_dataset`.
- Produces: Updated quantitative benchmark table, verified RMSE metrics, updated WGS-84 telemetry.

- [ ] **Step 1: Write integration regression test verifying benchmark KPIs**

```python
# naval_isr/tests/test_benchmark_regression.py
import pytest
from naval_isr.scripts.evaluate_all_autoferry_scenarios import evaluate_single_scenario

def test_scenario2_high_accuracy():
    res = evaluate_single_scenario("scenario2")
    assert res["combined_rmse"] < 8.0
    assert res["t1_rmse"] < 6.0

def test_scenario17_merged_radar_fix():
    res = evaluate_single_scenario("scenario17")
    # Must fix previous 65m error caused by track starvation
    assert res["combined_rmse"] < 25.0
```

- [ ] **Step 2: Run test to verify current state**

Run: `pytest naval_isr/tests/test_benchmark_regression.py -v`  
Expected: verify current behavior/failures.

- [ ] **Step 3: Wire `ModularMaritimeFusionPipeline` into `naval_isr/scripts/evaluate_all_autoferry_scenarios.py`**

Replace the monolithic tracker loop in `evaluate_all_autoferry_scenarios.py` with `ModularMaritimeFusionPipeline`. Execute the benchmark across all 9 scenarios, generate the comparison markdown report, and verify execution time $< 5.0\text{ seconds}$.

- [ ] **Step 4: Run all unit and regression tests**

Run: `pytest naval_isr/tests/ -v`  
Expected: ALL PASS

- [ ] **Step 5: Execute full benchmark script and verify output report**

Run: `python3 naval_isr/scripts/evaluate_all_autoferry_scenarios.py`  
Expected: PASS with 9 scenarios evaluated in $< 5.0\text{ seconds}$. Verify `naval_isr/output/all_scenarios_optimization_benchmark.md`.

- [ ] **Step 6: Commit**

```bash
git add naval_isr/scripts/evaluate_all_autoferry_scenarios.py naval_isr/tests/test_benchmark_regression.py naval_isr/output/all_scenarios_optimization_benchmark.md
git commit -m "chore(benchmark): wire modular sota fusion pipeline and update 9-scenario benchmark report"
```
