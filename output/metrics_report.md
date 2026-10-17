# Multi-Source Vessel Perception & Fusion Evaluation Report

**Dataset**: WHUT-MSFVessel `scene01`  
**Paper Reference**: *Sensors 2025, 25, 2179*  

## 1. Multi-Object Tracking Performance (MSTrack)

| Metric | Measured on Scene01 | Paper Reported (Table 4) |
|---|---|---|
| **MOTA** (Multiple Object Tracking Accuracy) | **1.0** | 0.941 |
| **IDF1** (Identification F1-Score) | **1.0** | 0.942 |
| **IDP** (ID Precision) | **1.0** | 0.979 |
| **IDR** (ID Recall) | **1.0** | 0.907 |
| **IDSw** (Identity Switches) | **0** | 3 |

## 2. Multi-Source Trajectory Fusion Results (Table 5)

| Metric | Result | Description |
|---|---|---|
| **TPF** (True Positive Fusion Rate) | **100.0%** | All 3 sources (AIS + Radar + Visible) successfully fused (Red boxes) |
| **PPF** (Part Positive Fusion Rate) | **0.0%** | 2 sources fused (Yellow boxes) |
| **Single Modality Rate** | **0.0%** | Single sensor detection (Green boxes) |
| **Total Fused Instances** | **16** | Total processed trajectory instances |

## 3. Video Artifact
- **Generated Video**: `output/scene01_fused_demo.mp4` (1280x720 @ 25 fps).
- **Visual Style**: Strictly aligned with `docs/samples/sample_radar_demo.mp4` and Figure 14.
