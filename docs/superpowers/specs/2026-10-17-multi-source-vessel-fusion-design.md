# Multi-Source Heterogeneous Data Fusion System for Surface Vessels (Sensors 2025, 25, 2179)

**Date**: 2026-10-17  
**Author**: Antigravity Assistant & User  
**Target Paper**: *Surface Vessels Detection and Tracking Method and Datasets with Multi-Source Data Fusion in Real-World Complex Scenarios* (Sensors 2025, 25, 2179)  
**Dataset**: WHUT-MSFVessel `scene01`  

---

## 1. System Overview & Objective

The objective of this system is to realize the complete end-to-end multi-source perception, tracking, and trajectory fusion pipeline proposed in Sensors 2025 (25, 2179), operating on real-world maritime data (AIS, Marine Radar, and Visible Video Camera).

The pipeline ingests:
1. Continuous visible video (`visible_video.mp4`, 2560x1440 @ 25 Hz, 13,650 frames).
2. Radar sweep image sequence (`radar_images/`, 2048x2048 @ ~2.25s interval, 228 frames).
3. Raw NMEA AIVDM/AIVDO text stream (`AIS.txt`).

And outputs:
1. Unified fused trajectories containing MMSI, Radar Tracking ID, Visible Tracking ID, Latitude, Longitude, Speed Over Ground (SOG), and Course Over Ground (COG).
2. Rendered visualization video strictly matching the official demo samples (`sample_radar_demo.mp4` & Figure 14) with:
   - **Red bounding box**: Tri-fused targets (AIS + Radar + Visible).
   - **Yellow bounding box**: Partially fused targets (AIS + Visible or Radar + Visible).
   - **Green bounding box**: Single modality (Visible only).
   - Multi-line text overlay: `SOG`, `COG`, `Lat`, `Lon`, `MMSI`, `RadarID`, `CameraID`.
3. Quantitative evaluation metrics (MOTA, IDF1, IDSw, TPF, PPF, FPF, FNF) compliant with Tables 3, 4, and 5 of the paper.

---

## 2. Mathematical Formulations & Algorithms

### 2.1 Spatiotemporal Alignment
- **Temporal Alignment**:
  - The radar rotation period is $\approx 2.24\text{ s}$ ($0.45\text{ Hz}$).
  - All data sources are synchronized to the radar update timestamps $T_{\text{rad}}$.
  - Visible frames: For each radar timestamp $T_{\text{rad}}$, the visible frame with the closest timestamp is chosen as the fusion keyframe. Non-fusion intermediate visible frames (25 Hz) update tracking and inherit the last fused attributes.
  - AIS: Linearly extrapolated from the last received packet to $T_{\text{rad}}$ via spherical Earth kinematic dead-reckoning equations:
    $$\text{lat}_{rt} = \text{lat}_{\text{last}} + \frac{180 \cdot \text{SOG} \cdot \Delta t \cdot \cos(\text{COG}_{\text{rad}})}{\pi \cdot R}$$
    $$\text{lon}_{rt} = \text{lon}_{\text{last}} + \frac{180 \cdot \text{SOG} \cdot \Delta t \cdot \sin(\text{COG}_{\text{rad}})}{\pi \cdot R \cdot \cos(\pi \cdot \text{lat}_{\text{last}} / 180)}$$
    where $R = 6,371,000\text{ m}$, $\text{SOG}$ is converted to $\text{m/s}$ ($\text{knots} \times 0.514444$).

### 2.2 Coordinate Transformations
- Platform Geodetic Coordinates: $[114.306087^\circ E, 30.563549^\circ N]$, height $h = 20\text{ m}$.
- **Geodetic to Radar Pixels** (Eq. 10): Affine transformation mapping $(\text{lon}, \text{lat}) \to (x_r, y_r)$.
- **Radar to Visible Pixels** (Eq. 6 & 11): Projection / Homography mapping matrix $F$ transforming radar coordinates to visible bounding box gates:
  $$Bbox_{\text{gate}|t}^{\text{vis}} = F \cdot Bbox_{\text{det}|t}^{\text{rad}}$$

### 2.3 MSTrack Multi-Stage Detection & Tracking
1. **Layered Tracking**:
   - Initial tracking on each modality using Kalman filter state vectors $\mathbf{x} = [x, y, a, h, \dot{x}, \dot{y}, \dot{a}, \dot{h}]$.
2. **Radar Feedback Re-detection**:
   - When a previously fused radar track is missing: generate tracking gate $Bbox_{\text{gate}}^{\text{rad}}$ via Kalman 1-step prediction.
   - If an extrapolated AIS point falls inside the gate, perform connected component extraction on radar image pixels $P(x,y) > T_p$ seeded at the AIS coordinate to recover the missing radar detection.
3. **Visible Feedback Re-detection**:
   - Classify missing trajectory confidence: $mconf = \frac{1}{n} \sum_{i=t-n+1}^t conf_i$.
   - If $mconf > 0.8$: extrapolate linearly from latest step.
   - If $mconf \le 0.8$: trace back to the latest high-confidence state and linearly extrapolate.
   - Project corresponding radar target into visible space via $F$ as tracking gate $Bbox_{\text{gate}|t}^{\text{vis}}$. Verify $Bbox_{\text{det}}^{\text{vis}}$ falls inside the gate before updating the trajectory.

### 2.4 Multi-Source Trajectory Cost Calculation & Cascade Association
- **Kinematic State**: $S = (x, y, \vec{v}, a, t, l)$ where $(x,y)$ is normalized position, $\vec{v}$ velocity vector, $a$ azimuth, $t$ entry time, $l$ track length.
- **Cost Formula** (Eq. 12 & 13):
  $$c_{i,j}^{(A,B)} = \begin{cases} +\infty & \text{if } |l_i - l_j| > l_{\text{limit}} \text{ or } |t_i - t_j| > t_{\text{limit}} \\ \frac{k \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2} + (1 - k)|a_i - a_j| + ||\vec{v}_i| - |\vec{v}_j||}{\cos(\vec{v}_i, \vec{v}_j)} & \text{otherwise} \end{cases}$$
  where $\cos(\vec{v}_i, \vec{v}_j) = \frac{1}{2} \left( \frac{\vec{v}_i \cdot \vec{v}_j}{|\vec{v}_i| |\vec{v}_j|} + 1 + \epsilon \right) \in (0, 1]$.
- **Cascade Matching Procedure**:
  - **Step 1**: Pairwise association between AIS and Radar trajectories via Hungarian bipartite matching ($k = 0.5$).
  - **Step 2**: Pairwise association between Radar and Visible trajectories ($k = 0.2$ to prioritize azimuth over depth).
  - **Step 3**: Unassociated AIS ($TU_{AA}$) matched against unassociated Visible ($TU_{AV}$) via visible projection.
  - **Step 4**: Integrated matching: AIS and Visible trajectories sharing the identical Radar ID are transitively bound into a 3-way fusion entity.

---

## 3. Module Architecture & File Structure

```
Fusion/
├── src/
│   ├── __init__.py
│   ├── ais_decoder.py        # NMEA AIVDM decoding, dead reckoning extrapolation
│   ├── coord_transform.py    # Geodetic, Radar, and Camera projection & Homography
│   ├── kalman_tracker.py     # Kalman filter state estimation & gating
│   ├── mstrack.py            # Feedback-based re-detection and tracking loop
│   ├── cascade_fusion.py     # Cost matrix calculation & 4-step Hungarian cascade
│   ├── visualizer.py         # 1280x720 video rendering with exact styling
│   └── evaluator.py          # MOTA, IDF1, TPF, PPF, FPF metrics calculator
├── tests/
│   ├── test_ais_decoder.py   # Verify AIS parsing against MMSI 413806072
│   ├── test_coord_transform.py # Verify projection accuracy
│   └── test_cascade_fusion.py # Unit tests for cost matrix & cascade matching
├── notebooks/
│   └── colab_yolo_train.ipynb # Free Google Colab GPU training notebook for YOLOv8
├── main.py                   # Master CLI runner to execute pipeline on Scene01
└── output/                   # Target directory for generated fusion video & metrics
```

---

## 4. Video Rendering Specification (Figure 14 & Demo Alignment)

- Output resolution: 1280 x 720 (half-scale of 2560x1440) @ 25 fps.
- Visual Bounding Boxes:
  - Thickness: 2 px.
  - Colors:
    - Tri-fused (AIS + Radar + Visible): Red `(40, 40, 230)` BGR.
    - Dual-fused (AIS + Visible or Radar + Visible): Yellow `(30, 220, 220)` BGR.
    - Single modality (Visible only): Green `(50, 210, 50)` BGR.
- Metadata Text Display (placed above upper bounding box edge):
  - Font: `cv2.FONT_HERSHEY_SIMPLEX`, scale `0.45`, thickness `1`.
  - Format:
    ```
    SOG:<speed_knots:.1f>
    COG:<course_deg:.1f>
    Lat:<latitude:.6f>
    Lon:<longitude:.6f>
    MMSI:<mmsi_id>
    RadarID:<radar_track_id>
    CameraID:<vis_track_id>-<conf:.1f>
    ```

---

## 5. Verification & Validation Plan

1. **Unit Verification**:
   - Verify AIS decoder parses all 1,953 AIVDM packets and extracts MMSI `413806072` and `413843904`.
   - Verify coordinate calibration error $< 15\text{ px}$ on key visible landmarks.
2. **Trajectory Association Verification**:
   - Compare fused track of MMSI `413806072` at 08:56:47 with ground truth:
     Expected: `Lat ~ 30.570142, Lon ~ 114.297007, SOG ~ 5.7, COG ~ 34.5, RadarID ~ 23`.
3. **Quantitative Metrics**:
   - Compute MOTA, IDF1, IDSw on Scene01 radar and visible tracks.
   - Verify MOTA $> 0.85$ and IDF1 $> 0.85$ as reported in Tables 3 & 4.
4. **Visual Artifact**:
   - Render the output video `output/scene01_fused_demo.mp4` and verify keyframes against `docs/samples/sample_radar_demo.mp4`.
