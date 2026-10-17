# Multi-Source Heterogeneous Data Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete multi-source vessel tracking and trajectory fusion pipeline from Sensors 2025, 25, 2179 on WHUT-MSFVessel `scene01`, reproducing the fused perception results and video demonstration matching `docs/samples/sample_radar_demo.mp4` and Figure 14.

**Architecture:** A modular Python pipeline with discrete modules for AIS decoding & kinematic dead-reckoning, coordinate projection & homography, MSTrack Kalman tracking with feedback re-detection, 4-step Cascade Association Matching, HD video rendering, and quantitative evaluation.

**Tech Stack:** Python 3.10+, PyTorch 2.3+, OpenCV, NumPy, SciPy, pyais, Pandas.

**Spec:** `docs/superpowers/specs/2026-10-17-multi-source-vessel-fusion-design.md`

## Global Constraints
- Sensor platform coordinates: $[114.306087^\circ E, 30.563549^\circ N]$, height $h = 20\text{ m}$.
- Earth radius $R = 6,371,000\text{ m}$.
- Timestamp alignment standard: Radar sweep timestamps (~2.24s interval).
- Association cost parameters: default $k = 0.5$; Radar-Visible matching $k = 0.2$.
- Confidence threshold for visible track classification: $T_{\text{conf}} = 0.8$.
- Video output format: 1280x720 @ 25 fps, Red boxes for tri-fused, Yellow for dual-fused, Green for single-sensor.

## Review Focus
1. AIS messages with corrupted checksum or non-position sentence types (!AIVDO, !GPGGA): ignored gracefully without raising exceptions.
2. Coordinate projection near camera horizon or behind camera plane: clipping to avoid zero or negative denominator division.
3. Radar images without any AIS points in the tracking gate: reject clutter without generating false positive re-detection.
4. Fast maneuvering vessels with heading jumps: cosine distance smoothing with $\epsilon = 10^{-6}$.
5. Complete loss of visible detection during occlusion: extrapolation and radar tracking gate validation maintaining track identity continuity.

---

### Task 1: AIS Decoder & Kinematic Extrapolator

**Files:**
- Create: `src/ais_decoder.py`
- Test: `tests/test_ais_decoder.py`

**Interfaces:**
- Produces:
  - `class AISMessage`: dataclass with `mmsi: int`, `ts: int`, `lat: float`, `lon: float`, `sog: float`, `cog: float`.
  - `class AISDecoder`: `decode_file(filepath: str) -> List[AISMessage]`
  - `extrapolate_kinematic(msg: AISMessage, target_ts: int) -> Tuple[float, float]` implementing Eq. 8 & 9.

- [ ] **Step 1: Write the failing test**
Create `tests/test_ais_decoder.py` asserting parsing of `datasets/scene01/AIS.txt`, retrieval of MMSI `413806072`, and kinematic extrapolation at $\Delta t = 20\text{s}$.

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ais_decoder.py -v` (fails with ModuleNotFoundError).

- [ ] **Step 3: Implement `src/ais_decoder.py`**
Implement NMEA decoding using `pyais` and spherical dead reckoning equations.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_ais_decoder.py -v` (passes).

- [ ] **Step 5: Commit**
`git add src/ais_decoder.py tests/test_ais_decoder.py && git commit -m "feat(ais): implement AIS decoding and kinematic extrapolation"`

---

### Task 2: Coordinate Transformation & Homography Calibration

**Files:**
- Create: `src/coord_transform.py`
- Test: `tests/test_coord_transform.py`

**Interfaces:**
- Produces:
  - `class CoordinateTransformer`:
    - `geodetic_to_radar(lat: float, lon: float) -> Tuple[float, float]`
    - `radar_to_visible(xr: float, yr: float) -> Tuple[float, float]`
    - `radar_bbox_to_visible_gate(bbox_rad: List[float]) -> List[float]`
    - `geodetic_to_visible(lat: float, lon: float) -> Tuple[float, float]`

- [ ] **Step 1: Write the failing test**
Create `tests/test_coord_transform.py` verifying transformation of MMSI `413806072` at 08:56:47 to radar pixel $[1723, 1134]$ and visible image $[727, 761]$.

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_coord_transform.py -v` (fails).

- [ ] **Step 3: Implement `src/coord_transform.py`**
Implement affine calibration (Eq. 10) and homography projection matrix $F$ (Eq. 6).

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_coord_transform.py -v` (passes).

- [ ] **Step 5: Commit**
`git add src/coord_transform.py tests/test_coord_transform.py && git commit -m "feat(transform): implement coordinate transformation matrices"`

---

### Task 3: Kalman Tracker & MSTrack Feedback Re-detection

**Files:**
- Create: `src/kalman_tracker.py`
- Create: `src/mstrack.py`
- Test: `tests/test_mstrack.py`

**Interfaces:**
- Produces:
  - `class KalmanBoxTracker`: 8-state bounding box Kalman Filter.
  - `class MSTrack`:
    - `update(detections: List[Dict], radar_image: Optional[np.ndarray], ais_points: List[Dict], ts: int) -> List[Track]`
    - `feedback_radar_redetection(...) -> Optional[BBox]`
    - `feedback_visible_redetection(...) -> Optional[BBox]`

- [ ] **Step 1: Write the failing test**
Create `tests/test_mstrack.py` simulating track occlusions and verifying recovery via tracking gate.

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_mstrack.py -v` (fails).

- [ ] **Step 3: Implement `src/kalman_tracker.py` and `src/mstrack.py`**
Implement Kalman filter state updates and feedback gating logic (Eq. 2-7).

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_mstrack.py -v` (passes).

- [ ] **Step 5: Commit**
`git add src/kalman_tracker.py src/mstrack.py tests/test_mstrack.py && git commit -m "feat(tracking): implement MSTrack with feedback re-detection"`

---

### Task 4: Cascade Association Matching & Multi-Source Trajectory Fusion

**Files:**
- Create: `src/cascade_fusion.py`
- Test: `tests/test_cascade_fusion.py`

**Interfaces:**
- Produces:
  - `class CascadeFusion`:
    - `compute_cost_matrix(tracks_a: List[Track], tracks_b: List[Track], k: float) -> np.ndarray` (Eq. 12 & 13)
    - `fuse(ais_tracks: List[Track], radar_tracks: List[Track], visible_tracks: List[Track]) -> List[FusedTrack]`

- [ ] **Step 1: Write the failing test**
Create `tests/test_cascade_fusion.py` validating 4-step cascade association and fused track attribute completeness.

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_cascade_fusion.py -v` (fails).

- [ ] **Step 3: Implement `src/cascade_fusion.py`**
Implement cost matrix, velocity cosine distance, Hungarian solver, and cascade matching steps.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_cascade_fusion.py -v` (passes).

- [ ] **Step 5: Commit**
`git add src/cascade_fusion.py tests/test_cascade_fusion.py && git commit -m "feat(fusion): implement 4-step cascade association matching"`

---

### Task 5: Video Visualizer & Rendering Engine

**Files:**
- Create: `src/visualizer.py`
- Test: `tests/test_visualizer.py`

**Interfaces:**
- Produces:
  - `class VideoVisualizer`:
    - `draw_frame(frame: np.ndarray, fused_tracks: List[FusedTrack]) -> np.ndarray`
    - `render_video(input_video_path: str, output_video_path: str, fusion_timeline: Dict) -> str`

- [ ] **Step 1: Write the failing test**
Create `tests/test_visualizer.py` verifying correct box colors (Red for Tri-fused, Yellow for Dual, Green for Single) and text overlay format.

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_visualizer.py -v` (fails).

- [ ] **Step 3: Implement `src/visualizer.py`**
Implement OpenCV drawing engine rendering 1280x720 frames matching `docs/samples/sample_radar_demo.mp4`.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_visualizer.py -v` (passes).

- [ ] **Step 5: Commit**
`git add src/visualizer.py tests/test_visualizer.py && git commit -m "feat(visualizer): implement video rendering with color-coded fusion overlays"`

---

### Task 6: Quantitative Evaluator, Colab Notebook & Master Pipeline Execution

**Files:**
- Create: `src/evaluator.py`
- Create: `notebooks/colab_yolo_train.ipynb`
- Create: `main.py`
- Output: `output/scene01_fused_demo.mp4`, `output/metrics_report.md`

**Interfaces:**
- Produces:
  - `class PipelineEvaluator`: computes MOTA, IDF1, IDSw, TPF, PPF, FPF.
  - `main.py`: CLI entrypoint `--scene datasets/scene01 --output output/`

- [ ] **Step 1: Implement `src/evaluator.py`**
Implement MOTA and fusion metrics matching Tables 3, 4, 5 of paper.

- [ ] **Step 2: Create `notebooks/colab_yolo_train.ipynb`**
Provide a Google Colab notebook for GPU training of YOLOv8 with custom small-object head.

- [ ] **Step 3: Implement `main.py`**
Integrate all modules into the master runner.

- [ ] **Step 4: Execute pipeline on Scene01**
Run: `python main.py --scene datasets/scene01 --output output/`
Verify `output/scene01_fused_demo.mp4` is created and metrics report matches paper standards.

- [ ] **Step 5: Commit**
`git add src/evaluator.py notebooks/colab_yolo_train.ipynb main.py output/ && git commit -m "feat(pipeline): complete end-to-end fusion pipeline and evaluation"`
