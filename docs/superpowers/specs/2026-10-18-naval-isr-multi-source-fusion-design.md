# Naval Multi-Source ISR Reconnaissance Data Fusion System
## Technical Architecture & 3-Phase Implementation Specification

**Date**: 2026-10-18  
**Author**: Antigravity Assistant & Technical Partner  
**Target Domain**: Naval Maritime Domain Awareness (MDA) / Tactical ISR Data Fusion  
**Document Type**: Architectural Design Specification  
**Status**: APPROVED IN BRAINSTORMING — READY FOR IMPLEMENTATION PLANNING  

---

## 1. Executive Summary & Objective

In modern naval warfare and maritime reconnaissance, creating an accurate **Recognized Maritime Picture (RMP)** is vital for early warning, defensive sovereignty, and tactical strike coordination. 

Civilian maritime surveillance generally relies on cooperative vessels broadcasting valid Automatic Identification System (AIS) messages. In sharp contrast, military maritime reconnaissance confronts:
1. **Non-cooperative targets ("Dark Vessels")**: Adversarial warships and infiltration vessels deliberately disable AIS/VMS transponders or broadcast falsified positions (AIS Spoofing / GNSS Meaconing).
2. **Active Electronic Counter-Measures (ECM)**: Adversarial jamming, sea clutter, chaff, and decoys disrupting radar echoes.
3. **Severe Sensor Heterogeneity & Asynchrony**:
   - **Active Coastal Radar**: High-accuracy range ($r$), azimuth ($\theta$), and radial Doppler velocity ($\dot{r}$), but lacks identity classification beyond Radar Cross Section (RCS).
   - **Passive Electronic Support Measures (ESM / ELINT)**: Zero-emission passive intercept providing electronic emitter fingerprinting (radar class, PRF, PRI), but provides **Line-of-Bearing (LOB / Angle of Arrival) only**, completely lacking range information.
   - **Spaceborne Synthetic Aperture Radar (SAR)**: All-weather, day-and-night theater-wide ocean imagery from space (e.g., Sentinel-1), but updates intermittently (hours to days interval).
   - **Cooperative AIS**: High-value semantic data (MMSI, ship dimensions, callsign), but vulnerable to deception and tampering.

### Scope & Core Constraint
- **Target Architecture**: An end-to-end 3-phase modular system covering **Coastal Multi-Sensor Simulation (Phase 1)**, **Satellite SAR Dark-Vessel Verification (Phase 2)**, and **AI-Driven Heterogeneous Graph & Threat Assessment (Phase 3)**.
- **Hardware Execution Constraint**: The local development machine possesses **limited CPU and RAM, and NO GPU**. All AI deep learning architectures (ST-GNN, SAR ship detectors) shall be fully coded, modularized, and furnished with standalone CLI training scripts and exportable Google Colab notebooks so they can be trained seamlessly on an external GPU-enabled machine without burdening the local host. Local testing will execute lightweight synthetic benchmarks, unit tests, and analytical validation.

---

## 2. 3-Phase Strategic Roadmap

```
====================================================================================================
PHASE 1: TACTICAL COASTAL SIMULATION (Dstl Stone Soup)
- Fixed Coastal Station: Active Radar (r, θ) + Passive ESM (LOB θ_esm, RF ID) + AIS Receiver
- Target Dynamics: 3 Vessels (1 Cooperative, 1 AIS-Spoofed, 1 High-Maneuvering Dark Warship)
- Algorithms: Interacting Multiple Model (CV + CT), Bearing-Only JPDA, Covariance Intersection (CI)
- Deliverables: 2D Tactical Display, OSPA Metric Curves, RMSE Benchmark, Spoofing Alerts
====================================================================================================
                                                 │
                                                 ▼
====================================================================================================
PHASE 2: REAL-WORLD THEATER-WIDE SAR RECONNAISSANCE (xView3-SAR Benchmark)
- Spaceborne Imagery: Sentinel-1 C-band SAR (VV/VH dual-polarization) + Bathymetry + Wind Rasters
- Algorithms: Calibrated Sigma0 Backscatter, Land/Island Masking, SAR Vessel Bbox Extraction
- Spatiotemporal AIS Correlation Engine: Dead-Reckoning Gating, Automatic Dark Vessel Flagging
- Deliverables: Interactive Tactical GIS Map (GeoTIFF/HTML), Detection F1, Length Error (MAE)
====================================================================================================
                                                 │
                                                 ▼
====================================================================================================
PHASE 3: AI-DRIVEN ASYNCHRONOUS FUSION & THREAT ASSESSMENT (ST-GNN + DSmT / PCR5)
- Model Architecture: Spatiotemporal Graph Neural Network (ST-GNN) / Cross-Modal Attention
- Attribute & Class Fusion: Dezert-Smarandache Theory (DSmT) with PCR5 Conflict Redistribution
- JDL Level 3 Threat Assessment: Threat Priority Index (TPI in [0, 100]) based on Capability & Intent
- Execution Note: Complete PyTorch models & training pipelines coded for external GPU / Colab execution
- Deliverables: Class Confusion Matrix, Multi-Target ROC Curves, Real-time Tactical Threat Matrix
====================================================================================================
```

---

## 3. Mathematical Formulations & Algorithms

### 3.1 Phase 1: Coastal Multi-Sensor Simulation & Kinematic Fusion

#### A. Target Motion Models (IMM Filter)
To maintain continuous tracking on warships executing deceptive evasive maneuvers without track divergence, an **Interacting Multiple Model (IMM)** filter coordinates two discrete kinematic regimes:

1. **Constant Velocity (CV) Model** (Cruise / Straight channel transit):
   $$\mathbf{x}_k = \mathbf{F}_{CV} \mathbf{x}_{k-1} + \mathbf{w}_{k-1}^{CV}, \quad \mathbf{x} = [x, \dot{x}, y, \dot{y}]^T$$
   $$\mathbf{F}_{CV} = \begin{bmatrix} 1 & \Delta t & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & \Delta t \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{Q}_{CV} = q_{CV} \begin{bmatrix} \frac{\Delta t^3}{3} & \frac{\Delta t^2}{2} & 0 & 0 \\ \frac{\Delta t^2}{2} & \Delta t & 0 & 0 \\ 0 & 0 & \frac{\Delta t^3}{3} & \frac{\Delta t^2}{2} \\ 0 & 0 & \frac{\Delta t^2}{2} & \Delta t \end{bmatrix}$$

2. **Coordinated Turn (CT) Model** (High-G Tactical Swerve maneuver):
   $$\mathbf{x}_k = \mathbf{F}_{CT}(\omega) \mathbf{x}_{k-1} + \mathbf{w}_{k-1}^{CT}, \quad \mathbf{x} = [x, \dot{x}, y, \dot{y}, \omega]^T$$
   $$\mathbf{F}_{CT}(\omega) = \begin{bmatrix} 1 & \frac{\sin(\omega \Delta t)}{\omega} & 0 & -\frac{1 - \cos(\omega \Delta t)}{\omega} & 0 \\ 0 & \cos(\omega \Delta t) & 0 & -\sin(\omega \Delta t) & 0 \\ 0 & \frac{1 - \cos(\omega \Delta t)}{\omega} & 1 & \frac{\sin(\omega \Delta t)}{\omega} & 0 \\ 0 & \sin(\omega \Delta t) & 0 & \cos(\omega \Delta t) & 0 \\ 0 & 0 & 0 & 0 & 1 \end{bmatrix}$$
   where $\omega$ is the turn rate (rad/s). Model transition probabilities $\Pi_{ij} = P(m_k = j \mid m_{k-1} = i)$ dynamically shift model weights based on innovation likelihoods.

#### B. Asymmetric Active-Passive Measurement Association (Radar $\leftrightarrow$ ESM)
- Radar observation vector: $\mathbf{z}_{rad} = [r, \theta]^T$ with measurement covariance $\mathbf{R}_{rad} = \text{diag}(\sigma_r^2, \sigma_\theta^2)$.
- ESM observation vector: $\mathbf{z}_{esm} = [\theta_{esm}]$ with measurement variance $R_{esm} = \sigma_{esm}^2$.
- Angular Mahalanobis distance metric:
  $$d^2(\mathbf{z}_{rad}, \mathbf{z}_{esm}) = \frac{(\theta_{rad} - \theta_{esm})^2}{\sigma_{\theta, rad}^2 + \sigma_{esm}^2}$$
- An ESM Line-of-Bearing is gated with radar track $\mathbf{x}_j$ if $d^2 \le \gamma_{thresh}$ ($\chi_1^2$ distribution threshold, typically $\gamma = 9.0$ for $3\sigma$).

#### C. Track-to-Track Fusion via Covariance Intersection (CI)
When fusing the radar-derived state estimate $(\hat{\mathbf{x}}_A, \mathbf{P}_A)$ with the ESM-interpolated state estimate $(\hat{\mathbf{x}}_B, \mathbf{P}_B)$ without knowing mutual cross-correlations:
$$\mathbf{P}_{CI}^{-1} = \omega \mathbf{P}_A^{-1} + (1 - \omega) \mathbf{P}_B^{-1}$$
$$\hat{\mathbf{x}}_{CI} = \mathbf{P}_{CI} \left( \omega \mathbf{P}_A^{-1} \hat{\mathbf{x}}_A + (1 - \omega) \mathbf{P}_B^{-1} \hat{\mathbf{x}}_B \right)$$
where $\omega \in [0, 1]$ minimizes $\text{trace}(\mathbf{P}_{CI})$ or $\det(\mathbf{P}_{CI})$ via 1D convex optimization.

---

### 3.2 Phase 2: Spaceborne SAR Reconnaissance & Dark Vessel Flagging

#### A. Radiometric Calibration & Sigma0 Backscatter
Raw digital numbers ($DN$) from Sentinel-1 SAR are radiometrically calibrated to radar backscatter coefficient $\sigma^0$ (dB):
$$\sigma^0 = 10 \log_{10} \left( \frac{DN^2 + A_i}{K_s} \right)$$
where $K_s$ is calibration factor and $A_i$ is offset. Dual-polarization ratio feature $PR = \frac{\sigma_{VH}^0}{\sigma_{VV}^0}$ maximizes contrast between metal ship superstructures and sea surface clutter.

#### B. Spatiotemporal Correlation & Dark Vessel Logic
For every detected SAR vessel bounding box at coordinate $\mathbf{p}_{SAR} = (\text{lat}_{SAR}, \text{lon}_{SAR})$ recorded at acquisition timestamp $T_{SAR}$:
1. Extrapolate all received AIS vessel records $\mathbf{p}_{AIS, i}(t)$ to $T_{SAR}$ using geodesic Dead Reckoning:
   $$\text{lat}_{rt} = \text{lat}_{last} + \frac{180 \cdot SOG \cdot \Delta t \cdot \cos(COG)}{\pi \cdot R}$$
   $$\text{lon}_{rt} = \text{lon}_{last} + \frac{180 \cdot SOG \cdot \Delta t \cdot \sin(COG)}{\pi \cdot R \cdot \cos(\pi \cdot \text{lat}_{last} / 180)}$$
2. Evaluate minimum geodesic distance:
   $$d_{min}(\mathbf{p}_{SAR}) = \min_i \text{Haversine}(\mathbf{p}_{SAR}, \mathbf{p}_{AIS, i}(T_{SAR}))$$
3. Categorization Rule:
   $$\text{Status}(\mathbf{p}_{SAR}) = \begin{cases} \text{COOPERATIVE (Green)} & \text{if } d_{min} \le \Delta d_{gate} \ (1.5\text{ km}) \\ \text{DARK VESSEL (Red Alert)} & \text{if } d_{min} > \Delta d_{gate} \text{ and } \text{Length}_{est} \ge L_{threat} \ (50\text{ m}) \\ \text{UNCLASSIFIED SMALL} & \text{if } d_{min} > \Delta d_{gate} \text{ and } \text{Length}_{est} < 50\text{ m} \end{cases}$$
4. Ghost Target (AIS Spoofing) Check:
   If $\exists \ \mathbf{p}_{AIS, j}(T_{SAR})$ where $\min_k \text{Haversine}(\mathbf{p}_{AIS, j}, \mathbf{p}_{SAR, k}) > 3.0\text{ km}$, flag MMSI $j$ as **DECEPTIVE / GHOST AIS**.

---

### 3.3 Phase 3: AI Asynchronous Fusion & DSmT Threat Assessment

#### A. Spatiotemporal Graph Neural Network (ST-GNN)
A maritime surveillance scene at time window $t$ is represented as a heterogeneous graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
- **Nodes $\mathcal{V}$**: Observations from disparate sensors:
  - $\mathbf{v}_{rad} \in \mathbb{R}^{d_1}$: $[r, \theta, \dot{r}, RCS, t]$
  - $\mathbf{v}_{esm} \in \mathbb{R}^{d_2}$: $[\theta_{esm}, f_{carrier}, PRI, PW, t]$
  - $\mathbf{v}_{sar} \in \mathbb{R}^{d_3}$: $[\text{lat}, \text{lon}, \text{length}, \text{beam}, t]$
  - $\mathbf{v}_{ais} \in \mathbb{R}^{d_4}$: $[MMSI, SOG, COG, \text{lat}, \text{lon}, t]$
- **Edges $\mathcal{E}$**: Spatiotemporal affinities:
  $$e_{ij} = \exp\left( -\frac{\|\mathbf{p}_i - \mathbf{p}_j\|^2}{2\sigma_d^2} - \frac{|t_i - t_j|^2}{2\sigma_t^2} \right)$$
- **Graph Attention Mechanism**:
  $$h_i^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}_i} \alpha_{ij}^{(l)} \mathbf{W}^{(l)} h_j^{(l)} \right), \quad \alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W} h_i \parallel \mathbf{W} h_j \parallel e_{ij}]))}{\sum_{k \in \mathcal{N}_i} \exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W} h_i \parallel \mathbf{W} h_k \parallel e_{ik}]))}$$
This learns association probabilities across high temporal jitter and missing modalities.

#### B. Dezert-Smarandache Theory (DSmT) & PCR5 Combination
Let frame of discernment be vessel warfare categories: $\Theta = \{\theta_D, \theta_F, \theta_C, \theta_M\}$:
- $\theta_D$: Destroyer (Khu trục hạm mang tên lửa dẫn đường)
- $\theta_F$: Frigate / Corvette (Hộ vệ hạm)
- $\theta_C$: Commercial / Cargo (Tàu vận tải dân sự)
- $\theta_M$: Maritime Militia / Infiltration Boat (Tàu trinh sát dân quân / tàu xâm nhập)

In DSmT, elements can overlap (e.g., $\theta_C \cap \theta_M$ represents a merchant hull conducting militia electronic reconnaissance). 

Given independent basic belief assignments (BBAs) $m_1(\cdot)$ from ESM emitter analysis and $m_2(\cdot)$ from SAR dimensional estimation, the **Proportional Conflict Redistribution Rule No. 5 (PCR5)** combines evidence without Zadeh's paradox:
$$m_{PCR5}(A) = \sum_{\substack{X, Y \in D^\Theta \\ X \cap Y = A}} m_1(X) m_2(Y) + \sum_{\substack{X \in D^\Theta \\ X \cap A = \emptyset}} \left( \frac{m_1(A)^2 m_2(X)}{m_1(A) + m_2(X)} + \frac{m_2(A)^2 m_1(X)}{m_2(A) + m_1(X)} \right)$$

#### C. Threat Priority Index (TPI) Calculation
$$\text{TPI}(\mathbf{x}_i) = 100 \cdot \left[ w_c \cdot C(\mathbf{x}_i) + w_u \cdot U(\mathbf{x}_i) + w_p \cdot P(\mathbf{x}_i) \right] \in [0, 100]$$
Where:
- **Capability $C(\mathbf{x}_i)$**:
  $$C(\mathbf{x}_i) = 0.95 \cdot m(\theta_D) + 0.70 \cdot m(\theta_F) + 0.40 \cdot m(\theta_M) + 0.05 \cdot m(\theta_C)$$
- **Urgency / Hostile Intent $U(\mathbf{x}_i)$**:
  $$U(\mathbf{x}_i) = 0.5 \cdot \mathbf{1}_{\{\text{Dark Vessel}\}} + 0.3 \cdot \mathbf{1}_{\{\text{Targeting Radar Active}\}} + 0.2 \cdot \mathbf{1}_{\{\text{Speed} \ge 20\text{ kts}\}}$$
- **Proximity & CPA $P(\mathbf{x}_i)$**:
  $$P(\mathbf{x}_i) = \max\left(0, 1 - \frac{d_{defense}}{R_{safe}}\right) \cdot \exp\left(-\frac{t_{CPA}}{300}\right)$$

---

## 4. System Architecture & Directory Organization

```
Fusion/
├── src/
│   ├── naval_isr/                    # Core Naval Multi-Source ISR Package
│   │   ├── __init__.py
│   │   ├── simulation/               # PHASE 1: Stone Soup Coastal Simulator
│   │   │   ├── __init__.py
│   │   │   ├── coastal_scenario.py   # 3-target tactical simulator (Cooperative, Spoofed, Dark)
│   │   │   ├── imm_tracker.py        # IMM Filter (CV + CT models)
│   │   │   ├── active_passive_jpda.py# Bearing-only ESM + Active Radar association
│   │   │   └── covariance_intersect.py# Track-to-Track CI fusion
│   │   ├── sar_recon/                # PHASE 2: Satellite SAR & Dark Vessel Engine
│   │   │   ├── __init__.py
│   │   │   ├── sar_preprocessor.py   # Sigma0 calibration, land masking, CFAR
│   │   │   ├── ais_correlation.py    # Spatiotemporal Dead-Reckoning correlation gate
│   │   │   └── dark_vessel_detector.py# Multi-criteria Dark Vessel & Spoofing classifier
│   │   ├── ai_inference/             # PHASE 3: Neural Fusion & Threat Assessment
│   │   │   ├── __init__.py
│   │   │   ├── st_gnn_model.py       # Heterogeneous Graph Attention / Transformer (PyTorch)
│   │   │   ├── dsmt_fusion.py        # DSmT engine with PCR5 combination rule
│   │   │   └── threat_assessor.py    # Threat Priority Index (TPI) & tactical ranker
│   │   └── visualization/
│   │       ├── tactical_plotter.py   # 2D tactical scenario display & OSPA/RMSE curves
│   │       └── gis_map_renderer.py   # Interactive HTML/Folium tactical maritime GIS map
├── scripts/
│   ├── run_phase1_simulation.py      # Standalone runner for Phase 1 (CPU-lightweight)
│   ├── run_phase2_sar_demo.py        # Standalone runner for Phase 2 with sample xView3 data
│   ├── export_phase3_colab.py        # Generates self-contained Google Colab GPU training notebook
│   └── evaluate_isr_metrics.py       # Computes OSPA, RMSE, Dark Detection F1, and Threat Matrix
├── tests/
│   ├── test_coastal_scenario.py      # Validates Stone Soup sensor generation
│   ├── test_active_passive_jpda.py   # Verifies Radar-ESM LOB gating
│   ├── test_dark_vessel_logic.py     # Verifies AIS correlation & spoof detection
│   └── test_dsmt_pcr5.py             # Verifies zero-paradox property on high-conflict tests
└── output/
    └── naval_isr/                    # Target directory for generated plots, GIS maps & reports
```

---

## 5. Hardware & Execution Constraints (Strict Compliance)

1. **Local Machine Execution Constraints**:
   - Environment: Linux, Python 3.10+, limited CPU & RAM, **NO GPU**.
   - **Phase 1 Execution**: 100% executable locally via `stonesoup` and `scipy`. Runs in $< 10$ seconds on CPU.
   - **Phase 2 Execution**: Lightweight analytical correlation and GIS rendering executable on CPU. Heavy SAR tile tiling / bulk downloading is abstracted via pre-extracted chips or sample GeoTIFF rasters.
   - **Phase 3 Model Architecture**: PyTorch model definitions, forward pass inference tests, and synthetic batch checks execute locally on CPU using tiny batch sizes ($N=2$).
2. **GPU Training Offloading Mechanism**:
   - `scripts/export_phase3_colab.py` automatically generates a single, fully self-contained `.ipynb` notebook configured for **free Google Colab T4/A100 GPU** execution to train the ST-GNN model on full datasets without consuming local CPU/RAM.

---

## 6. Deliverables, Output Reports, & Verification Plan

### 6.1 Report Artifacts
1. **Phase 1 Tactical Report (`output/naval_isr/phase1_coastal_report.md`)**:
   - Tactical 2D display showing radar blips, ESM bearing rays, and fused tracks.
   - OSPA distance metric curve over 900 seconds.
   - Track-to-Track RMSE comparison table (Radar-only vs. Radar-ESM fused).
2. **Phase 2 Dark Vessel Report (`output/naval_isr/phase2_dark_vessel_report.md`)**:
   - Interactive GIS tactical map (`tactical_sar_map.html`) pinpointing identified warships and flagged dark targets.
   - Classification performance metrics (Precision, Recall, F1 for Dark Vessel detection).
   - AIS spoofing incident log.
3. **Phase 3 Tactical Decision Matrix (`output/naval_isr/phase3_threat_report.md`)**:
   - DSmT conflict resolution log table demonstrating PCR5 vs. Dempster's rule.
   - Ranked Threat Priority Index (TPI) table showing actionable target classifications (Destroyer, Frigate, Militia) with recommended countermeasures.

### 6.2 Unit Testing & Quality Gate
- Run `pytest tests/test_coastal_scenario.py tests/test_active_passive_jpda.py tests/test_dark_vessel_logic.py tests/test_dsmt_pcr5.py -v`.
- All tests must pass with 100% green status prior to final demonstration.
